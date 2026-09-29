"""Optional input-owned terminal rebound context, admitted as one replay transaction.

No fitting, gate selection, file IO, or manual horizon is performed here. The
ordinary shared terminal-response fitter consumes any admitted original wing.
Replay preserves observations already inside the original ending as well as the
qualified extension; it must not reinterpret original frames as newly added context.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from cv.pipeline import resolution
from cv.pipeline.bounce_detect import split_runs

POLICY = "initial_native_rebound_component_v1"
GAP_SECONDS = 0.12
JUMP_PIXELS_PER_SECOND_1920 = 2750.0


def ownership(labels: dict, search: dict) -> dict:
    """Qualify the first post-ending observation component, before fitting."""
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    attempt = labels["attempt"]
    clip = search["clip"]
    groups = [r for r in labels["ball"]["records"] if r["clip"] == clip]
    if len(groups) != 1:
        raise ValueError("one original ball window required")
    rows = groups[0]["frames"]
    source = dict(
        attempt=attempt,
        rows=rows,
        events=labels["events"]["records"],
        native_size=labels.get("native_size"),
        fps=search["configuration"]["fps"],
    )
    receipt = dict(
        policy=POLICY,
        source_sha256=hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        status="unsupported",
        supported_native_frames=[],
        later_unowned_native_frames=[],
        full_video_context=[min(r["frame"] for r in rows), max(r["frame"] for r in rows)],
        native_timestamps_changed=False,
        physical_ending_inferred=False,
        fit_selected_horizon=False,
        gap_seconds=GAP_SECONDS,
        jump_pixels_per_second_1920=JUMP_PIXELS_PER_SECOND_1920,
    )
    if attempt.get("observation_scope") is not None:
        return receipt | dict(reason="original_unknown_ending_scope_preserved")
    if search["configuration"]["termination_kind"] not in {"terminal_bounce", "second_bounce"}:
        return receipt | dict(reason="no_original_ground_ending")
    endpoint = float(attempt["ending_frame"])
    grounds = [
        e for e in search["events"] if e["event_type"] == "bounce" and float(e["frame"]) <= endpoint
    ]
    if not grounds:
        return receipt | dict(reason="no_supplied_terminal_ground")
    ground = max(grounds, key=lambda e: float(e["frame"]))
    epoch = float(ground["frame"])
    interval = ground.get("frame_interval", [epoch, epoch])
    receipt.update(
        supplied_terminal_bounce_frame=epoch,
        supplied_competitive_endpoint=endpoint,
        original_ground_event=deepcopy(ground),
    )
    size = labels.get("native_size")
    if not isinstance(size, list) or len(size) != 2:
        return receipt | dict(reason="declared_native_dimensions_unavailable")
    size = resolution.FrameSize(*size)
    fps = float(search["configuration"]["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("finite native cadence required")
    visible = [
        r
        for r in rows
        if r["status"] == "visible"
        and float(r["frame"]) > epoch
        and resolution.points_inside_image([r["x1080"], r["y1080"]], size)
    ]
    visible.sort(key=lambda r: float(r["frame"]))
    if not visible:
        return receipt | dict(reason="no_native_postground_observations")
    frames = np.array([r["frame"] for r in visible], float)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("unique ordered native observations required")
    gap = max(1, int(math.ceil(GAP_SECONDS * fps)))
    groups = split_runs(
        frames,
        np.array([r["x1080"] for r in visible]),
        np.array([r["y1080"] for r in visible]),
        gap_max=gap,
        jump_px=JUMP_PIXELS_PER_SECOND_1920 * size.width / 1920 / fps,
    )
    components = [frames[g].tolist() for g in groups]
    receipt["continuity_components"] = components
    if components[0][0] > float(interval[1]) + gap:
        return receipt | dict(reason="initial_postground_component_not_supported")
    # A missing detection is not a change of ownership. In-picture occlusions
    # remain in this same observation component unless native motion supplies
    # a departure or incompatible reappearance witness.
    first = frames.tolist()
    gaps = []
    boundary = None
    speed_limit = JUMP_PIXELS_PER_SECOND_1920 * size.width / 1920 / fps
    pixels = np.array([[r["x1080"], r["y1080"]] for r in visible], float)
    for index in range(1, len(frames)):
        elapsed = frames[index] - frames[index - 1]
        displacement = pixels[index] - pixels[index - 1]
        if np.linalg.norm(displacement) / elapsed > speed_limit:
            boundary = dict(
                reason="incompatible_native_motion",
                before_frame=frames[index - 1],
                after_frame=frames[index],
                native_displacement=displacement.tolist(),
                elapsed_frames=elapsed,
                maximum_pixels_per_native_frame=speed_limit,
            )
        elif elapsed > gap:
            departure = None
            if index >= 3 and np.all(np.diff(frames[index - 3 : index]) <= gap):
                velocities = (
                    np.diff(pixels[index - 3 : index], axis=0)
                    / np.diff(frames[index - 3 : index])[:, None]
                )
                last = pixels[index - 1]
                edges = []
                for axis, limit in enumerate((size.width, size.height)):
                    for side, sign, distance in (
                        ("low", -1, last[axis]),
                        ("high", 1, limit - last[axis]),
                    ):
                        outward = sign * velocities[:, axis]
                        if np.all(outward > 0) and distance <= float(np.min(outward)) * gap:
                            edges.append(
                                dict(axis=axis, side=side, distance_pixels=float(distance))
                            )
                if edges:
                    departure = dict(
                        native_frames=frames[index - 3 : index].tolist(),
                        native_centers=pixels[index - 3 : index].tolist(),
                        secant_pixels_per_frame=velocities.tolist(),
                        edges=edges,
                        crossing_bracket_frames=gap,
                        exact_exit_epoch_inferred=False,
                    )
            gaps.append(
                dict(
                    before_frame=frames[index - 1],
                    after_frame=frames[index],
                    departure_evidence=departure,
                    status="unobserved_offscreen_continuation"
                    if departure
                    else "uncertain_in_picture_continuity_retained",
                )
            )
            if departure:
                boundary = dict(reason="native_edge_departure_before_gap", **gaps[-1])
        if boundary is not None:
            first = frames[:index].tolist()
            break
    receipt["native_continuity_boundary"] = boundary
    receipt["native_gap_evidence"] = gaps
    bound = exposure.before_next_physical_event(
        dict(
            postbounce_labeled_frames=first,
            postbounce_labeled_frame_count=len(first),
            last_postbounce_labeled_frame=first[-1],
            status="supported",
        ),
        labels,
        clip,
        epoch,
        search["configuration"]["exposure_duration_frames"],
    )
    owned = bound["postbounce_labeled_frames"]
    receipt.update(
        supported_native_frames=owned,
        later_unowned_native_frames=[f for f in frames.tolist() if f not in owned],
        next_physical_event_boundary=bound.get("next_physical_event_boundary"),
    )
    if not owned or owned[-1] <= endpoint:
        return receipt | dict(reason="no_supported_extension_beyond_source_endpoint")
    return receipt | dict(
        status="qualified",
        reason="supported_initial_rebound_component",
        observation_horizon=owned[-1],
    )


def limit_inventory(receipt: dict, labels: dict, search: dict) -> dict:
    """Apply the same input ownership before the first numerical search objective."""
    spec = ownership(labels, search)
    if labels["attempt"].get("observation_scope") is not None:
        # An existing unresolved physical scope has its own original horizon.
        return receipt | {"native_context_ownership": spec}
    frames = [
        f for f in receipt["postbounce_labeled_frames"] if f in spec["supported_native_frames"]
    ]
    return receipt | dict(
        native_context_ownership=spec,
        status="supported" if frames else "no_owned_native_postbounce_frames",
        postbounce_labeled_frames=frames,
        postbounce_labeled_frame_count=len(frames),
        last_postbounce_labeled_frame=max(frames) if frames else None,
    )


def require_initial_ownership(labels: dict, search: dict) -> None:
    """Replay cannot restore an unowned source tail by trusting a cached horizon."""
    if search["configuration"].get("terminal_context_ownership", "off") != "on":
        return
    receipt = search["configuration"].get("terminal_rebound") or {"mode": "off"}
    declared = receipt.get("native_context_ownership")
    supported = receipt.get("mode") == "on" and receipt.get("status") == "supported"
    if declared is None:
        if supported:
            raise ValueError("owned initial rebound requires its input ownership receipt")
        return
    actual = ownership(labels, search)
    if declared != actual:
        raise ValueError("initial terminal ownership differs from original inputs")
    if supported and labels["attempt"].get("observation_scope") is None:
        endpoint = float(labels["attempt"]["ending_frame"])
        recorded = receipt["postbounce_labeled_frames"]
        ground_epoch = actual.get("supplied_terminal_bounce_frame", endpoint)
        # A search can retain post-bounce pictures already inside the original
        # competitive window without extending that window. The terminal-tail
        # rule certifies added context, not deletion of existing observations.
        original = {
            float(row["frame"])
            for group in labels["ball"]["records"]
            if group["clip"] == search["clip"]
            for row in group["frames"]
            if row["status"] == "visible" and ground_epoch < float(row["frame"]) <= endpoint
        }
        extension = [frame for frame in actual["supported_native_frames"] if frame > endpoint]
        if (
            any(
                not isinstance(frame, (int, float)) or not math.isfinite(frame)
                for frame in recorded
            )
            or recorded != sorted(set(recorded))
            or not set(actual["supported_native_frames"]).issubset(recorded)
            or any(frame <= endpoint and frame not in original for frame in recorded)
            or [frame for frame in recorded if frame > endpoint] != extension
        ):
            raise ValueError("initial search reattached unowned native observations")
        horizon = max(endpoint, actual.get("observation_horizon", endpoint))
        if float(receipt["modeled_scene_end_frame"]) != horizon:
            raise ValueError("initial search reattached an unowned modeled horizon")


def preserved_segment(scene, heldout, bounces, axes):
    """Structural rebound witnesses remain original training rows; checks stay checks."""
    if scene.dynamics != "measured_240hz" or not len(bounces[-1]):
        raise ValueError("terminal rebound needs a supplied measured-dynamics bounce")
    bounce = float(bounces[-1][-1])
    native = np.unique(np.r_[scene.observation_frames[-1], heldout.observation_frames[-1]])
    native = native[native > bounce]
    frames = np.asarray(scene.observation_frames[-1], float)
    frames = frames[frames > bounce]
    if not len(frames):
        raise ValueError("no original training exposure supports terminal rebound")
    return (
        scene,
        axes,
        frames,
        dict(
            mode="on",
            status="supported",
            postbounce_labeled_frames=native.tolist(),
            postbounce_labeled_frame_count=len(native),
            supplied_terminal_bounce_frame=bounce,
            terminal_structural_training_frames=frames.tolist(),
            original_partition_preserved=True,
            withheld_rows_activated_count=0,
            withheld_rows_activated_frames=[],
            native_timestamps_changed=False,
        ),
    )


def physical_replay(context: dict, parameters, duration) -> None:
    """Query all original rows and the full physical interval, without acceptance gates."""
    from cv.experiments.connected_shooting import full_native_continuation as full
    from cv.experiments.connected_shooting.labeled_preparation_net_followup import fit_check_copy

    checks, _ = fit_check_copy(context["scene"], context["heldout"])
    active, axes, _ = full.merge_scene(
        context["scene"], checks, context["bounces"], context["axes"]
    )
    # Temporary replay union only; original train/check scenes are not modified.
    # chain integrates through each endpoint, even if native queries are sparse.
    full.model.chain(active, parameters)
    prediction = full.exposure.prediction(
        active, parameters, axes, duration, termination_kind=context["termination_kind"]
    )
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite native aftermath projection")


def partition_record(context: dict) -> dict:
    """Store source frame identities separately; an all-native check copy stays explicit."""
    return {
        name: [np.asarray(frames, float).tolist() for frames in context[name].observation_frames]
        for name in ("scene", "heldout")
    }


def require_original_partition(source: dict, proposed: dict) -> None:
    before, after = partition_record(source), partition_record(proposed)
    horizon = float(source["scene"].contact_frames[-1])
    for name in before:
        if len(before[name]) != len(after[name]):
            raise ValueError("extension changed original flight partition")
        for old, new in zip(before[name], after[name], strict=True):
            if old != [f for f in new if f <= horizon]:
                raise ValueError("extension changed original native train/check identities")


def prepare(
    context: dict, parameters, labels: dict, cameras: dict, search: dict
) -> tuple[dict, dict]:
    """Atomically admit an input-owned extension or retain the finite source scene."""
    from cv.experiments.connected_shooting import labeled_context_census as census

    if context.get("terminal_context_support") is not None:
        raise ValueError("terminal context transaction must run once per invocation")
    spec = ownership(labels, search)
    source_end = float(context["scene"].contact_frames[-1])
    receipt = dict(
        policy=POLICY,
        ownership=spec,
        status="source_retained",
        source_horizon=source_end,
        modeled_horizon=source_end,
        supplied_terminal_bounce_frame=spec.get("supplied_terminal_bounce_frame"),
        parameters_changed=False,
        gate_selection=False,
        original_partition_preserved=True,
        original_native_observation_partition=partition_record(context),
    )
    # Never substitute an invalid source state merely because optional extension failed.
    physical_replay(context, parameters, search["configuration"]["exposure_duration_frames"])
    if spec["status"] == "qualified" and source_end > spec["observation_horizon"]:
        raise ValueError("initial search consumed unowned native context; rerun shared preparation")
    if spec["status"] == "qualified" and source_end == spec["observation_horizon"]:
        if search["configuration"].get("terminal_context_ownership") != "on":
            raise ValueError("initial rebound search lacks matching ownership/partition policy")
        receipt.update(status="committed", extension={"status": "initial_search_already_owned"})
    elif spec["status"] == "qualified":
        try:
            proposed, details = census.extended_context(
                context,
                labels,
                cameras,
                search,
                last_context_frame=spec["observation_horizon"],
                preserve_partition=True,
            )
            if proposed is None:
                raise ValueError(f"qualified extension was not prepared: {details}")
            require_original_partition(context, proposed)
            physical_replay(
                proposed, parameters, search["configuration"]["exposure_duration_frames"]
            )
            context = proposed
            receipt.update(
                status="committed",
                modeled_horizon=float(context["scene"].contact_frames[-1]),
                extension=details,
            )
        except (ValueError, FloatingPointError, OverflowError) as error:
            receipt["reason"] = f"unsupported_modeled_aftermath: {error}"
    else:
        receipt["reason"] = spec["reason"]
    receipt["native_observation_partition"] = partition_record(context)
    return context | {"terminal_context_support": deepcopy(receipt)}, receipt


def validate_result(result: dict, labels: dict, search: dict, policy: dict) -> None:
    """Recompute ownership from bound input evidence; never infer a new export horizon."""
    marker = result.get("evaluation_context", {}).get("terminal_context_support")
    if policy.get("terminal_context_ownership", "off") == "off":
        if marker is not None:
            raise ValueError("owned terminal context requires explicit shared policy")
        return
    if search["configuration"].get("terminal_context_ownership") != "on":
        raise ValueError("initial search did not apply declared terminal ownership policy")
    if marker is None or marker != result.get("context") or marker.get("policy") != POLICY:
        raise ValueError("missing or conflicting terminal context transaction receipt")
    if marker.get("ownership") != ownership(labels, search):
        raise ValueError("terminal context ownership differs from original observations")
    if marker.get("native_observation_partition") != {
        name: result["evaluation_context"][name]["observation_frames"]
        for name in ("scene", "heldout")
    }:
        raise ValueError("final scene changed owned native observation identities")
    end = float(result["evaluation_context"]["scene"]["contact_frames"][-1])
    if end != marker["modeled_horizon"]:
        raise ValueError("final scene reattached unowned terminal observations")
    if marker["status"] == "committed":
        if end != marker["ownership"].get("observation_horizon"):
            raise ValueError("committed context differs from owned horizon")
    elif marker["status"] != "source_retained" or end != marker["source_horizon"]:
        raise ValueError("failed extension changed the original source horizon")
