"""Default-off source-interval interior timing with fixed native row ownership.

The continuous optimizer may move an impact, never a picture. The admissible
interval is inside the original incoming/outgoing observation gap, including
withheld exposures. No fitted input, label lookup, timing grid or case rule.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np

POLICIES = ("off", "source_interval")
FIELD = "interior_contact_epoch_fit"
MARGIN_FRAMES = 1e-6


def prepare(scene, events: list[dict], native, duration: float | None) -> dict:
    """Derive all bounds from source intervals, physical barriers and native rows."""
    from cv.experiments.connected_shooting.observation_operator import support_span

    contacts = [e for e in events if e["event_type"] == "contact"]
    closing_contact = None
    if getattr(scene, "right_boundary_kind", "supplied_end") == "original_contact":
        if not contacts or float(contacts[-1]["frame"]) != float(scene.contact_frames[-1]):
            raise ValueError("contact timing requires the original closing contact boundary")
        closing_contact = deepcopy(contacts[-1])
        contacts = contacts[:-1]
    if len(contacts) != len(scene.contact_frames) - 1 or len(native) != len(scene.pixels):
        raise ValueError("contact timing requires the complete original event and row inventory")
    if not np.array_equal([e["frame"] for e in contacts], scene.contact_frames[:-1]):
        raise ValueError("contact timing source epochs differ from the original scene")
    rows = []
    for i in range(1, len(contacts)):
        event = contacts[i]
        target = float(event["frame"])
        row = dict(
            contact_index=i,
            nominal_frame=target,
            source_event=deepcopy(event),
            status="frozen",
            reason="invalid_or_missing_source_interval",
        )
        rows.append(row)
        interval = event.get("frame_interval")
        try:
            low, high = map(float, interval)
        except (TypeError, ValueError):
            continue
        if not np.isfinite([low, high, target]).all() or not low <= target <= high or low >= high:
            continue
        incoming, outgoing = np.asarray(native[i - 1], float), np.asarray(native[i], float)
        if not len(incoming) or not len(outgoing):
            row["reason"] = "missing_two_sided_native_support"
            continue
        gap = [float(np.max(incoming)) + support_span(duration), float(np.min(outgoing))]
        row.update(source_interval_frames=[low, high], observation_gap_frames=gap)
        low, high = max(low, gap[0] + MARGIN_FRAMES), min(high, gap[1] - MARGIN_FRAMES)
        # Preserve supplied bounce/net inventory on both sides. Intervals, not
        # just nominal impacts, form barriers; no contact may reorder evidence.
        for other in events:
            if other["event_type"] not in ("bounce", "net_hit"):
                continue
            a, b = map(float, other.get("frame_interval", [other["frame"]] * 2))
            if float(other["frame"]) < target:
                low = max(low, b + MARGIN_FRAMES)
            else:
                high = min(high, a - MARGIN_FRAMES)
        row["bounds_frames"] = [low, high]
        if not low < high:
            # Keep the existing refusal receipt for unchanged empty intersections.
            row["reason"] = "source_epoch_outside_open_observation_or_event_gap"
            continue
        row.update(
            status="movable",
            reason=(
                "source_interval_inside_native_gap"
                if low < target < high
                else "source_interval_intersects_native_gap_projected_initializer"
            ),
            sigma_frames=(row["source_interval_frames"][1] - row["source_interval_frames"][0]) / 2,
        )
    return dict(
        policy="source_interval",
        contacts=rows,
        original_contact_frames=np.asarray(scene.contact_frames, float).tolist(),
        native_observation_frames=[np.asarray(f, float).tolist() for f in native],
        exposure_duration_frames=duration,
        observations_or_native_timestamps_changed=False,
        scope="continuous connected solve; fixed initializer and native ownership",
        **({"closing_contact": closing_contact} if closing_contact is not None else {}),
    )


def initial_frame(row: dict) -> float:
    """Project only the source nominal; an explicit in-search seed must be valid."""
    nominal = float(row["nominal_frame"])
    initial = row.get("initial_frame")
    if row["status"] == "frozen":
        if initial is not None and (not np.isfinite(initial) or initial != nominal):
            raise ValueError("frozen contact initializer differs from source epoch")
        return nominal
    low, high = row["bounds_frames"]
    if initial is None:
        return float(np.clip(nominal, low, high))
    if not np.isfinite(initial) or not low <= initial <= high:
        raise ValueError("interior timing initializer must lie inside its source bounds")
    return float(initial)


def validate_plan(scene, plan: dict) -> list[dict]:
    """Fail closed on malformed numerical timing metadata."""
    if plan.get("policy") != "source_interval" or plan["original_contact_frames"] != list(
        scene.contact_frames
    ):
        raise ValueError("interior timing plan differs from source scene")
    closing = plan.get("closing_contact")
    if getattr(scene, "right_boundary_kind", "supplied_end") == "original_contact":
        if (
            not isinstance(closing, dict)
            or closing.get("event_type") != "contact"
            or closing.get("frame") != scene.contact_frames[-1]
        ):
            raise ValueError("interior timing closing contact differs from source boundary")
    elif closing is not None:
        raise ValueError("closing contact metadata requires an original contact boundary")
    rows = plan["contacts"]
    if [r["contact_index"] for r in rows] != list(range(1, len(scene.contact_frames) - 1)):
        raise ValueError("interior timing plan must retain every contact")
    movable = []
    for row in rows:
        i = row["contact_index"]
        if row["nominal_frame"] != scene.contact_frames[i]:
            raise ValueError("interior timing nominal epoch changed")
        if row["status"] == "frozen":
            initial_frame(row)
            continue
        if row["status"] != "movable":
            raise ValueError("unknown interior timing status")
        lo, hi = row["bounds_frames"]
        sigma = row["sigma_frames"]
        original_low, original_high = row["source_interval_frames"]
        if lo < original_low or hi > original_high:
            raise ValueError("interior timing bounds exceed the original source interval")
        if (
            not np.isfinite([lo, hi, sigma, original_low, original_high]).all()
            or not lo < hi
            or not original_low <= row["nominal_frame"] <= original_high
            or sigma <= 0
        ):
            raise ValueError("invalid interior timing bounds")
        from cv.experiments.connected_shooting.observation_operator import support_span

        if (
            max(plan["native_observation_frames"][i - 1])
            + support_span(plan["exposure_duration_frames"])
            >= lo
            or min(plan["native_observation_frames"][i]) <= hi
        ):
            raise ValueError("interior timing would reassign a native observation")
        initial_frame(row)
        movable.append(row)
    return movable


def source_plan(receipt: dict) -> dict:
    """Strip only numerical outcomes/seeds; immutable source evidence remains."""
    original = deepcopy(receipt)
    original.pop("fitted_contact_frames", None)
    for row in original["contacts"]:
        for field in ("fitted_frame", "shift_frames", "soft_prior_residual", "initial_frame"):
            row.pop(field, None)
    return original


def seed_plan(scene, plan: dict, fit: dict | None) -> dict:
    """Carry an in-search numerical epoch seed without changing source bounds/priors."""
    if fit is None:
        return plan
    receipt = fit.get(FIELD)
    if receipt is None or source_plan(receipt) != plan:
        raise ValueError("interior timing seed differs from its source-derived plan")
    active = fitted_scene(scene, fit)
    seeded = deepcopy(plan)
    for row in validate_plan(scene, seeded):
        row["initial_frame"] = float(active.contact_frames[row["contact_index"]])
    validate_plan(scene, seeded)
    return seeded


def fitted_scene(scene, fit: dict):
    """Replay fitted impact epochs; physical parameters and original rows stay intact."""
    frames = np.asarray(scene.contact_frames, float).copy()
    if fit.get("first_contact_epoch_fit") is not None:
        frames[0] = float(fit["first_contact_epoch_fit"]["fitted_frame"])
    receipt = fit.get(FIELD)
    if receipt is not None:
        validate_plan(scene, receipt)
        declared = receipt["fitted_contact_frames"]
        if len(declared) != len(frames) or declared[-1] != frames[-1] or declared[0] != frames[0]:
            raise ValueError(
                "interior fit changed first/final timing without its declared mechanism"
            )
        for row in receipt["contacts"]:
            i = row["contact_index"]
            value = float(declared[i])
            if row["fitted_frame"] != value or not np.isfinite(value):
                raise ValueError("inconsistent fitted contact epoch")
            if row["status"] == "frozen":
                if value != frames[i]:
                    raise ValueError("fitted contact moved a frozen epoch")
            elif not row["bounds_frames"][0] <= value <= row["bounds_frames"][1]:
                raise ValueError("fitted contact escaped its source/native bracket")
            frames[i] = value
    active = replace(scene, contact_frames=frames)
    active.validate()
    return active


def apply_context(context: dict, fit: dict) -> dict:
    receipt = fit.get(FIELD)
    if receipt is None and fit.get("first_contact_epoch_fit") is None:
        return context
    if receipt is not None:
        duration = context.get("observation_operator", {}).get("exposure_duration_frames")
        if receipt["exposure_duration_frames"] != duration:
            raise ValueError("contact timing exposure duration differs from source context")
        expected = prepare(context["scene"], context["events"], context["native"], duration)
        actual = source_plan(receipt)
        if actual != expected:
            raise ValueError("contact timing receipt does not reproduce from source inputs")
    return (
        context
        | {name: fitted_scene(context[name], fit) for name in ("scene", "heldout")}
        | ({FIELD: deepcopy(receipt)} if receipt is not None else {})
        | (
            {"first_contact_epoch_fit": deepcopy(fit["first_contact_epoch_fit"])}
            if fit.get("first_contact_epoch_fit") is not None
            else {}
        )
    )


def original_contact_inventory(
    context: dict, duration: float | None
) -> tuple[np.ndarray, list[dict]]:
    """Recover event identities after timing fits without moving source events.

    Downstream impact inventory uses these original boundaries; propagation and
    separation constraints still use the active fitted epochs. A serve-prefix
    stage may independently move the first contact, but cannot change any of
    the interior or final epochs attested by this receipt.
    """
    scene = context["scene"]
    receipt = context.get(FIELD)
    if receipt is None:
        return scene.contact_frames, context["events"]
    if receipt["exposure_duration_frames"] != duration:
        raise ValueError("contact boundary receipt has a different exposure duration")
    original = np.asarray(receipt["original_contact_frames"], float)
    events = deepcopy(context["events"])
    first = next(e for e in events if e["event_type"] == "contact")
    if first["frame"] != original[0]:
        if (
            first.get("profiled_within_original_interval") is not True
            or first.get("original_representative_frame") != original[0]
            or first["frame"] != scene.contact_frames[0]
        ):
            raise ValueError("first contact profile lacks its original event identity")
        first["frame"] = float(original[0])
    source = replace(scene, contact_frames=original)
    plan = prepare(source, events, context["native"], duration)
    if source_plan(receipt) != plan:
        raise ValueError("contact boundary receipt differs from original event/native evidence")
    validate_plan(source, plan)
    fitted = fitted_scene(
        source,
        {
            FIELD: receipt,
            "first_contact_epoch_fit": context.get("first_contact_epoch_fit"),
        },
    )
    for name in ("scene", "heldout"):
        if not np.array_equal(context[name].contact_frames[1:], fitted.contact_frames[1:]):
            raise ValueError("active contact boundary differs from its fitted timing receipt")
    return original, events


def preserve_in_extended_context(context: dict, extended: dict, duration: float | None) -> dict:
    """Carry fitted contacts through a source-observed terminal-only extension.

    The original receipt is validated before rebinding its native inventory and
    final horizon. Every contact bound and fitted epoch must remain unchanged.
    """
    receipt = context.get(FIELD)
    if receipt is None:
        return extended
    original, events = original_contact_inventory(context, duration)
    rebuilt = extended["scene"]
    if not np.array_equal(rebuilt.contact_frames, extended["heldout"].contact_frames):
        raise ValueError("terminal extension training/check boundaries disagree")
    if len(rebuilt.contact_frames) != len(original) or not np.array_equal(
        rebuilt.contact_frames[1:-1], original[1:-1]
    ):
        raise ValueError("terminal extension changed original contact topology")
    horizon = float(rebuilt.contact_frames[-1])
    if horizon < context["scene"].contact_frames[-1]:
        raise ValueError("terminal extension shortened the fitted observation horizon")
    before, after = context["native"], extended["native"]
    if len(before) != len(after) or any(
        not np.array_equal(a, b) for a, b in zip(before[:-1], after[:-1], strict=True)
    ):
        raise ValueError("terminal extension changed earlier native ownership")
    old_tail, new_tail = np.asarray(before[-1]), np.asarray(after[-1])
    additions = np.setdiff1d(new_tail, old_tail)
    if not np.isin(old_tail, new_tail).all() or np.any(
        additions <= context["scene"].contact_frames[-1]
    ):
        raise ValueError("terminal extension changed original native support")
    boundaries = original.copy()
    boundaries[-1] = horizon
    source = replace(rebuilt, contact_frames=boundaries)
    plan = prepare(source, events, after, duration)
    if plan["contacts"] != source_plan(receipt)["contacts"]:
        raise ValueError("terminal extension changed contact timing evidence or bounds")
    rebased = deepcopy(receipt)
    rebased["original_contact_frames"] = plan["original_contact_frames"]
    rebased["native_observation_frames"] = plan["native_observation_frames"]
    rebased["fitted_contact_frames"][-1] = horizon
    fit = {FIELD: rebased, "first_contact_epoch_fit": context.get("first_contact_epoch_fit")}
    active = fitted_scene(source, fit).contact_frames
    # A later source-qualified serve profile can independently update the first
    # contact; original_contact_inventory has already checked its event identity.
    active[0] = context["scene"].contact_frames[0]
    if not np.array_equal(active[:-1], context["scene"].contact_frames[:-1]):
        raise ValueError("terminal extension changed a fitted contact epoch")
    result = extended | {
        name: replace(extended[name], contact_frames=active.copy()) for name in ("scene", "heldout")
    }
    result[FIELD] = rebased
    result["interior_timing_extension"] = dict(
        previous_horizon=float(context["scene"].contact_frames[-1]),
        extended_horizon=horizon,
        added_native_frames=additions.tolist(),
        fitted_contacts_preserved=True,
        native_timestamps_changed=False,
    )
    original_contact_inventory(result, duration)
    return result
