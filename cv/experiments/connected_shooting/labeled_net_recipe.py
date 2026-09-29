"""Explicit-input terminal-net recipe with input-ranked source and matched evidence.

Opened labeled development, not automatic inference. Reuses the existing free
net-velocity fitter and continuous ground guidance; no acceptance-winner lookup.
Requires an original terminal net, supplied later ground and observed rebound.
Example: python -m cv.experiments.connected_shooting.labeled_net_recipe --help
"""

from __future__ import annotations

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import traceback

import numpy as np

from cv.experiments.connected_shooting import (
    labeled_common_source as source_loader,
    labeled_net_context_fit as original,
    labeled_net_free_response as free_response,
    labeled_net_ray_seed as ray_seed,
    labeled_serve_recipe as recipe,
)
from cv.pipeline import provenance

epoch, census, scope = original.epoch, original.census, original.scope

#: The single ownership refusal s6_terminal_context.ownership records for an
#: original attempt whose physical ending is unresolved; see its early return.
SCOPE_RETENTION_REASON = "original_unknown_ending_scope_preserved"


class UnsupportedNet(ValueError):
    """Original topology or required native support is outside this adapter."""


def declared_horizon(record: dict) -> float | None:
    """Read an explicitly declared native horizon; never infer one from a scene."""
    value = record.get("observation_horizon")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def retained_observation_scope(context: dict, receipt: dict) -> bool:
    """Is this exactly the unchanged original observation-scoped source context?

    The owned-context transaction is reported before the observation horizon, so
    a source-retained scope transaction arrives here as a preserved-context
    receipt. That is the original source with nothing added, not an unsupported
    one. Admit it only when the transaction, the ownership refusal it records,
    the declared horizon and the current native partition all still agree; a
    failed, malformed or stale transaction is refused like any other context.
    """
    from cv.experiments.connected_shooting import observed_horizon_tail as tail_scope
    from cv.pipeline import s6_terminal_context as owned

    attempt = context["attempt"]
    scope_contract = attempt.get("observation_scope")
    transaction = context.get("terminal_context_support")
    tail = attempt.get("observed_horizon_tail")
    if (
        not isinstance(scope_contract, dict)
        or not isinstance(transaction, dict)
        or (tail is not None and not isinstance(tail, dict))
        or receipt.get("status") != "owned_native_context_preserved"
        or receipt.get("transaction") != transaction
    ):
        return False
    ownership = transaction.get("ownership")
    if (
        transaction.get("policy") != owned.POLICY
        or transaction.get("status") != "source_retained"
        or "extension" in transaction
        or not isinstance(ownership, dict)
        or ownership.get("status") != "unsupported"
        or ownership.get("reason") != SCOPE_RETENTION_REASON
        or transaction.get("reason") != ownership.get("reason")
    ):
        return False
    if declared_horizon(scope_contract) is None or (
        tail is not None and declared_horizon(tail) is None
    ):
        return False
    horizon = tail_scope.modeled_horizon(attempt, scope_contract)
    partition = owned.partition_record(context)
    return (
        float(context["scene"].contact_frames[-1]) == horizon
        and transaction.get("source_horizon") == horizon
        and transaction.get("modeled_horizon") == horizon
        and transaction.get("original_native_observation_partition") == partition
        and transaction.get("native_observation_partition") == partition
    )


def ground_bounded_frames(event: dict, available, ground_events: list[dict]):
    """Keep each ground witness on its adjacent incoming and outgoing arcs."""
    epoch = float(event["frame"])
    before = [float(e["frame_interval"][1]) for e in ground_events if float(e["frame"]) < epoch]
    after = [float(e["frame_interval"][0]) for e in ground_events if float(e["frame"]) > epoch]
    low, high = max(before, default=-np.inf), min(after, default=np.inf)
    kept = [f for f in available if low < f < high]
    return kept, dict(
        previous_ground_upper_frame=low if np.isfinite(low) else None,
        next_ground_lower_frame=high if np.isfinite(high) else None,
        excluded_across_ground_or_uncertain_frames=[f for f in available if f not in kept],
    )


def ground_occurrence_supported(impact: dict, context: dict) -> bool:
    """Consume the original reviewed occurrence separately from exact timing."""
    # Timing-uncertain source rows still require the existing unique source join.
    # A valid prepared occurrence receipt cannot override a changed source ledger.
    if impact.get("status") != "ambiguous" and occurrence.resolved_membership(impact):
        return True
    if impact.get("status") != "ambiguous" or impact.get("occurrence_status") in (
        "unsupported",
        "ambiguous",
        "absent",
    ):
        return False
    from cv.experiments.connected_shooting.labeled_event_occurrence import timing_admission

    attempt = context.get("attempt", {})
    clip = attempt.get("point_clip")
    matches = [
        row
        for row in attempt.get("agent_event_rows", [])
        if row["event_type"] == "bounce"
        and row.get("clip", clip) == clip
        and float(row["frame"]) == float(impact["frame"])
        and row.get("frame_interval") == impact.get("frame_interval")
    ]
    return len(matches) == 1 and timing_admission(matches[0]) is not None


def qualify(context: dict) -> dict:
    """A terminal net family only; absent or interior events are not reinterpreted."""
    start, end = map(float, context["scene"].contact_frames[-2:])
    nets = [e for e in context["events"] if e["event_type"] == "net_hit"]
    if len(nets) != 1 or not start < float(nets[0]["frame"]) < end:
        raise UnsupportedNet("exactly one original terminal-flight net and no other nets required")
    event = nets[0]
    low, high = map(float, event["frame_interval"])
    if not occurrence.resolved_membership(event) or not (
        start < low <= float(event["frame"]) <= high < end
    ):
        raise UnsupportedNet("resolved net membership and an original in-flight interval required")
    ground = sorted(
        [
            e
            for e in context["events"]
            if e["event_type"] == "bounce" and high < float(e["frame"]) <= end
        ],
        key=lambda e: float(e["frame"]),
    )
    if not ground:
        raise UnsupportedNet(
            "supplied post-net ground required; net-ending evidence is a separate arm"
        )
    if len(ground) not in (1, 2) or not np.array_equal(
        np.asarray(context["bounces"][-1], float), [float(e["frame"]) for e in ground]
    ):
        raise UnsupportedNet(
            "one or two supplied terminal bounces matching the physical inventory required"
        )
    previous_high = high
    for impact in ground:
        lo, hi = map(float, impact["frame_interval"])
        if not ground_occurrence_supported(impact, context) or not (
            previous_high < lo <= float(impact["frame"]) <= hi
        ):
            raise UnsupportedNet("resolved ordered ground membership and intervals required")
        previous_high = hi
    return dict(net=event, ground_events=ground, terminal_flight=len(context["scene"].pixels) - 1)


def prepare(bundle: dict) -> dict:
    """Construct the existing native net-bounded witnesses before fitting/scoring."""
    context = bundle["context"]
    if getattr(context["scene"], "terminal_net_tail", None) is not None:
        from cv.experiments.connected_shooting import (
            labeled_terminal_tail_response as tail_response,
        )

        eligibility = tail_response.qualify(context)
        return dict(
            context=context,
            eligibility=eligibility,
            target_receipts=[],
            context_receipt=dict(
                status="original_observed_tail",
                observations_changed=False,
                ground_epochs_supplied=False,
                occurrence_inferred=False,
            ),
        )
    eligibility = qualify(context)
    labels, cameras_doc, search = bundle["labels"], bundle["cameras"], bundle["search"]
    extended, receipt = census.extended_context(context, labels, cameras_doc, search)
    if extended is not None:
        context = extended
    elif (
        context.get("terminal_context_support", {}).get("status") == "committed"
        and receipt.get("status") == "owned_native_context_preserved"
    ):
        pass
    elif retained_observation_scope(context, receipt):
        # The owned-context transaction already retained this original observed
        # scope, so the scope check inside the census is never reached. Nothing
        # is extended, shortened or re-owned here; keep the source exactly.
        pass
    elif (
        context["attempt"].get("observation_scope") is not None
        and receipt.get("status") == "original_observation_horizon_preserved"
    ):
        # Scope preparation already retained the original observed aftermath.
        # No extension is needed; None here means unchanged, not unsupported.
        # Keep the supplied boundary and every observation exactly as prepared.
        pass
    else:
        raise UnsupportedNet(f"observed post-net rebound context unsupported: {receipt}")
    targets = deepcopy(context["targets"])
    cameras = {
        int(row["frame"]): np.asarray(row["P"])
        for row in cameras_doc["cameras"]
        if "P" in row and row.get("supported", True)
    }
    clip = context["attempt"]["point_clip"]
    records = [
        row
        for group in labels["ball"]["records"]
        if group["clip"] == clip
        for row in group["frames"]
        if row["status"] == "visible"
    ]
    pixels = {int(row["frame"]): np.array([row["x1080"], row["y1080"]]) for row in records}
    # Automatic centers can explicitly declare unmeasured uncertainty. Keep it
    # absent from the witness lookup, as in ordinary ground preparation; a null
    # value is neither a measured radius nor the legacy missing-field default.
    radii = {
        int(row["frame"]): radius
        for row in records
        if (radius := row.get("uncertainty_radius_px1080", 2.0)) is not None
    }
    nets = [e for e in context["events"] if e["event_type"] == "net_hit"]
    target_receipts = []
    for i, group in enumerate(targets):
        if not len(context["scene"].net_hit_frames[i]):
            continue
        available = np.unique(
            np.r_[context["scene"].observation_frames[i], context["heldout"].observation_frames[i]]
        )
        for j, target in enumerate(group):
            event = next(
                e
                for e in context["events"]
                if e["event_type"] == "bounce" and e["frame"] == target["event_frame"]
            )
            eligible, bound = original.bounded.net_bounded_frames(event, available, nets)
            if len(eligibility["ground_events"]) == 2:
                eligible, ground_bounds = ground_bounded_frames(
                    event, eligible, eligibility["ground_events"]
                )
                bound = {**bound, **ground_bounds, "bounded_eligible_frames": eligible}
            new = original.whole.event_ground_target(
                event,
                cameras,
                pixels,
                radii,
                camera_distortion=original.whole.camera_geometry.camera_radial_map(cameras_doc),
                mode="subframe_graded_circle",
                observation_fallback=True,
                eligible_frames=eligible,
            )
            group[j] = new
            target_receipts.append(dict(flight=i, bounds=bound, before=target, after=new))
    active = context | {"targets": targets}
    checks, consumed = original.followup.fit_check_copy(active["scene"], active["heldout"])
    return dict(
        context=active,
        original_target_context=context,
        fit_context=active | {"heldout": checks},
        eligibility=eligibility,
        context_receipt=receipt,
        target_receipts=target_receipts,
        consumed_check_rows=consumed,
    )


@contextmanager
def continuous_ground(context: dict):
    """Scope the existing continuous ground forecast to this worker's fit call."""
    evaluate = epoch.interval.block.event_constraints.evaluate
    intervals = [
        next(
            e["frame_interval"]
            for e in context["events"]
            if e["event_type"] == "bounce" and e["frame"] == frame
        )
        for frame in context["bounces"][-1]
    ]

    def guided(scene, params, *args, **kwargs):
        native, residuals, evidence = evaluate(scene, params, *args, **kwargs)
        return original.horizon.apply(scene, params, native, residuals, evidence, intervals)

    try:
        epoch.interval.block.event_constraints.evaluate = guided
        yield
    finally:
        epoch.interval.block.event_constraints.evaluate = evaluate


def score(context: dict, parameters, bundle: dict) -> tuple[dict, dict]:
    return census.score(
        context,
        parameters,
        bundle["threshold"],
        bundle["duration"],
        bundle["ending"],
        scorer=scope.LOCAL_SCORE,
    )


def measure_result(prepared: dict, parameters, bundle: dict, response: dict) -> dict:
    """Store final verdict and measurement together; old targets are only diagnostic."""
    law = free_response.response_from_record(response)
    with epoch.tape.using_response(law):
        after, measurement = score(prepared["context"], parameters, bundle)
        old_after, old_measurement = score(prepared["original_target_context"], parameters, bundle)
        chain = recipe.full.model.chain(prepared["context"]["scene"], parameters)
    return dict(
        after=after,
        measurement=measurement,
        original_targets_diagnostic=dict(after=old_after, measurement=old_measurement),
        final_net_hits=chain[-1]["net_hits"],
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source_loader.add_arguments(p)
    p.add_argument("--max-nfev", type=int, default=120)
    p.add_argument("--seconds", type=float, default=300.0)
    p.add_argument("--net-initializer", choices=["source", "native-plane"], default="source")
    p.add_argument("--preflight-only", action="store_true")
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        automatic_inference_eligible=False,
        human_derived=True,
    )
    stage = "source_loading"
    try:
        bundle = source_loader.load(args)
        modules = [
            __file__,
            source_loader.__file__,
            recipe.__file__,
            original.__file__,
            epoch.__file__,
            epoch.chart.__file__,
            epoch.height_chart.__file__,
            epoch.tape.__file__,
            original.followup.__file__,
            original.bounded.__file__,
            original.horizon.__file__,
            free_response.__file__,
            ray_seed.__file__,
            census.__file__,
            scope.__file__,
            recipe.acceptance.__file__,
            recipe.rescore.__file__,
            recipe.full.model.__file__,
            recipe.full.exposure.__file__,
        ]
        result.update(
            sources=bundle["sources"] + [provenance.file_record(Path(p)) for p in modules],
            seed_selection=bundle["selection"],
            source_reproduction=bundle["reproduction"],
            source_projection_replay=bundle["projection_replay"],
            association_preparation=bundle["association_preparation"],
            physics=bundle["physics"],
            original_parameters=bundle["parameters"],
            independent_xyz_truth_available=False,
        )
        stage = "context_preparation"
        prepared = prepare(bundle)
        source_before, source_measurement = score(bundle["context"], bundle["parameters"], bundle)
        before, before_measurement = score(prepared["context"], bundle["parameters"], bundle)
        result.update(
            source_before=source_before,
            source_measurement=source_measurement,
            before=before,
            original_measurement=before_measurement,
            eligibility=prepared["eligibility"],
            context_receipt=prepared["context_receipt"],
            targets=prepared["target_receipts"],
            fit_only_duplicate_check_rows_removed=prepared["consumed_check_rows"],
            policy=dict(
                free_net_velocity=True,
                continuous_ground=True,
                quadratic_images=True,
                joint=False,
                upper_net=False,
                free_net_spin=False,
                net_band=False,
                first_ground_seed=False,
                jacobian_step=1e-6,
                scorer="existing_labeled_serve_covariance_scope",
                same_final_measurement_verdict_context=True,
            ),
        )
        recipe.save(args.output / "start.json", result)
        if args.preflight_only:
            result.update(status="preflight_only_no_fit")
        else:
            stage = "fit"
            initial_parameters = bundle["parameters"]
            if args.net_initializer == "native-plane":
                initial_parameters, initialization = ray_seed.seed(
                    prepared["fit_context"],
                    initial_parameters,
                    bundle["labels"],
                    bundle["cameras"],
                    prepared["eligibility"]["net"],
                )
                result["input_initialization"] = initialization
                result["initial_parameters"] = initial_parameters
            with continuous_ground(prepared["fit_context"]):
                fitted = epoch.fit(
                    prepared["fit_context"],
                    initial_parameters,
                    bundle["duration"],
                    maxiter=args.max_nfev,
                    seconds=args.seconds,
                    free_net_velocity=True,
                    jacobian_step=1e-6,
                    quadratic_images=True,
                )
            recipe.save(args.output / "fit.json", fitted)
            stage = "measure"
            parameters = fitted["best"]["parameters"]
            result.update(measure_result(prepared, parameters, bundle, fitted["best"]["response"]))
            context = prepared["context"]
            result.update(
                status="measured_requires_native_review",
                fit=fitted,
                parameters=parameters,
                scene_contact_frames=context["scene"].contact_frames,
                terminal_rebound_frames=context["terminal_rebound_frames"],
                evaluation_scene=asdict(context["scene"]),
                evaluation_heldout=asdict(context["heldout"]),
            )
    except Exception as error:
        result.update(
            status="unsupported_input"
            if isinstance(error, (UnsupportedNet, ray_seed.UnsupportedSeed))
            else "execution_failed",
            stage=stage,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    recipe.save(args.output / "report.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
