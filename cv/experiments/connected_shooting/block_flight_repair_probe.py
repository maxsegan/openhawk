"""Can three-flight block repair recover a missing interior flight?

Opened-development diagnostic only. Freeze outer contacts, contact epochs,
rebound coefficients and every unrelated parameter. Move the two internal
contact positions and three launch velocity/spin blocks. Original native
leading-front pixels constrain contact neighborhoods; no true XYZ is supplied.
Export the complete single-shooting vector and reuse unchanged acceptance and
previously-accepted-flight guards. No newly constructed legacy witness fallback.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    event_constraints,
    local_flight_repair_probe as local,
    model,
    real_exposure_replay as exposure,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance


def block_indices(scene, flight_index):
    count = len(scene.contact_frames) - 1
    if not 0 < flight_index < count - 1:
        raise ValueError("a three-flight block requires a nonterminal interior target")
    blocks = model.shared_parameter_slices(scene)
    contacts = np.arange(3 * flight_index, 3 * (flight_index + 2))
    velocities = np.arange(
        blocks["velocities"].start + 3 * (flight_index - 1),
        blocks["velocities"].start + 3 * (flight_index + 2),
    )
    spins = np.arange(
        blocks["spins"].start + 3 * (flight_index - 1),
        blocks["spins"].start + 3 * (flight_index + 2),
    )
    return np.r_[contacts, velocities, spins]


def contact_image_groups(scene, flight_index):
    """Separate native picture wings within four frames of each moving contact."""
    offsets = np.r_[0, np.cumsum([len(frames) for frames in scene.observation_frames])]
    groups = []
    for contact_index in (flight_index, flight_index + 1):
        epoch = scene.contact_frames[contact_index]
        for flight in (contact_index - 1, contact_index):
            frames = scene.observation_frames[flight]
            chosen = np.flatnonzero(np.abs(frames - epoch) <= 4.0)
            if not len(chosen):
                raise ValueError("moving contact has an unsupported original image wing")
            groups.append(offsets[flight] + chosen)
    return groups


def image_budgets(residual, groups):
    return np.asarray([np.mean(np.sum(residual[group] ** 2, axis=1)) for group in groups])


def unrelated_path_drift(before, after, changed_flights):
    """Certify that exporting the block did not propagate a seam downstream."""
    if len(before) != len(after):
        raise ValueError("full exported flight inventory changed")
    unchanged = [index for index in range(len(before)) if index not in changed_flights]
    return max(
        (
            float(
                np.linalg.norm(
                    np.asarray(after[index]["positions"]) - np.asarray(before[index]["positions"]),
                    axis=1,
                ).max()
            )
            for index in unchanged
        ),
        default=0.0,
    )


def frozen_bounce_plan(context, flight_index):
    """Use the independently frozen witness and the unchanged reproduced rung."""
    threshold = context["repair_thresholds"]
    witnesses = context["repair_reference_verdict"]["flights"][flight_index]["bounce_witness"]
    if not witnesses:
        raise ValueError("bounce feasibility requires a supplied independent witness")
    plan = []
    for witness in witnesses:
        if not all(
            witness.get(key, False)
            for key in (
                "two_sided_reversal",
                "covariance_resolved",
                "timing_witness_resolved",
                "serve_witness_covariance_resolved",
            )
        ):
            raise ValueError("bounce feasibility requires a resolved frozen two-sided witness")
        plan.append(
            dict(
                xy_m=witness["witness_xyz_m"][:2],
                event_frame=witness["supplied_frame"],
                maximum_distance_m=min(
                    witness["raw_distance_limit_m"],
                    threshold.bounce_ray_limit_m + witness["graded_circle_radius_m"],
                )
                - 1e-4,
                maximum_time_error_frames=threshold.bounce_uncertainty_frames - 1e-5,
                strict_search_interior_margin_m=1e-4,
            )
        )
    return plan


def bounce_feasibility_slack(bounces, plan):
    """Positive means inside the same spatial/timing gate, with a tiny margin."""
    if len(bounces) != len(plan):
        return np.full(2 * len(plan) + 1, -1.0)
    slack = []
    for observed, witness in zip(bounces, plan, strict=True):
        distance = np.linalg.norm(np.asarray(observed["x"])[:2] - witness["xy_m"])
        dt = observed["frame"] - witness["event_frame"]
        slack.extend(
            [
                1 - (distance / witness["maximum_distance_m"]) ** 2,
                1 - (dt / witness["maximum_time_error_frames"]) ** 2,
            ]
        )
    return np.r_[slack, 0.0]


def accepted_neighbor_bounce_plan(context, flight_index):
    """Preserve an already accepted frozen witness route, including old chords."""
    threshold = context["repair_thresholds"]
    flight = context["repair_reference_verdict"]["flights"][flight_index]
    ray_limit = (
        threshold.serve_bounce_ray_limit_m
        if flight.get("role") == "serve" and threshold.serve_bounce_ray_limit_m is not None
        else threshold.bounce_ray_limit_m
    )
    if not flight["accepted"]:
        raise ValueError("neighbor preservation requires an originally accepted flight")
    plan = []
    for witness in flight["bounce_witness"]:
        if witness["from_terminal_completion"]:
            raise ValueError("terminal-completion witness unsupported by bounded repair policy")
        if witness["reversal_agrees"]:
            xy = witness["witness_xyz_m"][:2]
            limit = min(
                witness["raw_distance_limit_m"],
                ray_limit + witness["graded_circle_radius_m"],
            )
            route = "frozen_two_sided_reversal"
        elif witness["legacy_agrees"] and witness.get("legacy_subframe_witness"):
            xy = witness["legacy_subframe_witness"]["xyz_m"][:2]
            limit = ray_limit + witness["legacy_graded_circle_radius_m"]
            route = "already_accepted_frozen_legacy"
        else:
            raise ValueError("accepted neighbor has no reproducible frozen bounce route")
        plan.append(
            dict(
                xy_m=xy,
                event_frame=witness["supplied_frame"],
                maximum_distance_m=limit - 1e-4,
                maximum_time_error_frames=threshold.bounce_uncertainty_frames - 1e-5,
                strict_search_interior_margin_m=1e-4,
                witness_route=route,
            )
        )
    return plan


def block_window_plan(context, flights):
    """Freeze native membership and an allowed timing shift, never observations."""
    scene = context["scene"]
    threshold = context["repair_thresholds"]
    offsets = np.r_[0, np.cumsum([len(frames) for frames in scene.observation_frames])]
    plan = []
    seen = set()
    for index in flights:
        frames = scene.observation_frames[index]
        for window in context["repair_reference_verdict"]["flights"][index]["directional_windows"]:
            horizon = {"short": 4.0, "medium": 10.0}[window["horizon"]]
            epoch = window["event_frame"]
            low, high = (
                (epoch - horizon, epoch)
                if window["direction"] == "backward"
                else (epoch, epoch + horizon)
            )
            selected = np.flatnonzero((frames >= low) & (frames <= high)) + offsets[index]
            if len(selected) != window["native_training_pictures"]:
                raise ValueError("frozen native window membership failed to reproduce")
            shift = (
                0.0
                if window["zero_shift_rms_px"] <= threshold.directional_rms_limit_px
                else window["selected_time_shift_frames"]
            )
            if abs(shift) > threshold.directional_time_shift_frames:
                raise ValueError("frozen window shift exceeds unchanged rung allowance")
            key = (tuple(selected), shift)
            if key not in seen:
                plan.append(
                    dict(
                        indices=selected.tolist(),
                        shift_frames=shift,
                        rms_limit_px=threshold.directional_rms_limit_px - 1e-4,
                        flight_index=int(index),
                        kind="event_window",
                    )
                )
                seen.add(key)
        plan.append(
            dict(
                indices=list(range(int(offsets[index]), int(offsets[index + 1]))),
                shift_frames=0.0,
                rms_limit_px=threshold.flight_reprojection_rms_limit_px - 1e-4,
                flight_index=int(index),
                kind="flight_reprojection",
            )
        )
    return plan


def optimize_block(
    context,
    parameters,
    flight_index,
    *,
    maxiter,
    bounce_bracket_frames,
    duration,
    bounce_feasibility=False,
    preserve_block_gates=False,
):
    scene = replace(context["scene"], parameterization="shared_contact_states")
    initial = model.shared_contact_seed(scene, parameters)
    indices = block_indices(scene, flight_index)
    flights_in_block = np.arange(flight_index - 1, flight_index + 2)
    groups = contact_image_groups(scene, flight_index)
    gate_plan = frozen_bounce_plan(context, flight_index) if bounce_feasibility else []
    gate_plans = {flight_index: gate_plan} if gate_plan else {}
    if preserve_block_gates:
        if not gate_plan:
            raise ValueError("block gate preservation requires explicit bounce feasibility")
        for index in flights_in_block:
            if (
                index != flight_index
                and context["repair_reference_verdict"]["flights"][index]["accepted"]
            ):
                gate_plans[int(index)] = accepted_neighbor_bounce_plan(context, int(index))
    window_plan = block_window_plan(context, flights_in_block) if preserve_block_gates else []
    constraint_count = sum(2 * len(plan) + 1 for plan in gate_plans.values()) + len(window_plan)
    scale = np.r_[[1.0] * 6, [30.0] * 9, [3.0] * 9]
    # A bounded development trust region, not an acceptance threshold. Height
    # bounds match the existing shared-contact fitter. Outer contact XYZ never
    # appears in the optimized vector.
    lower = np.r_[initial[indices[:6]] - 1.5, [-75.0] * 9, [-6.0] * 9]
    upper = np.r_[initial[indices[:6]] + 1.5, [75.0] * 9, [6.0] * 9]
    lower[[2, 5]] = np.maximum(lower[[2, 5]], model.R_BALL + 1e-4)
    upper[[2, 5]] = np.minimum(upper[[2, 5]], 4.0)
    if np.any(initial[indices] < lower) or np.any(initial[indices] > upper):
        raise ValueError("reference outside the original velocity/spin/height bounds")
    target = np.concatenate(scene.pixels)
    loss = event_constraints.mixed_loss(2 * len(target))
    cache = FlightCache(entries_per_flight=80)
    anchor = exposure.anchor_plan(context["targets"], [None] * len(scene.pixels))
    memo = {}

    def expand(q):
        full = initial.copy()
        full[indices] = np.asarray(q) * scale
        return full

    invalid_evaluations = {}

    def raw_evaluate(q):
        key = np.asarray(q).tobytes()
        if key in memo:
            return memo[key]
        full = expand(q)
        flights, physical, _ = event_constraints.evaluate(
            scene, full, context["bounces"], bounce_bracket_frames, simulation_cache=cache
        )
        image = (
            exposure.prediction(
                scene,
                full,
                context["axes"],
                duration,
                cache,
                termination_kind=context["termination_kind"],
            )
            - target
        )
        anchors = []
        for flight in flights_in_block:
            for j, (xy, sigma) in enumerate(anchor["bounce"][flight]):
                observed = flights[flight]["bounces"]
                anchors.extend(
                    (np.asarray(observed[j]["x"])[:2] - xy) / sigma
                    if j < len(observed)
                    else [0.0, 0.0]
                )
        residual = np.r_[image.ravel(), physical, anchors]
        gate_slack = [
            bounce_feasibility_slack(flights[index]["bounces"], plan)
            for index, plan in gate_plans.items()
        ]
        shifted_images = {0.0: image}
        for shift in {row["shift_frames"] for row in window_plan} - {0.0}:
            shifted_images[shift] = (
                exposure.time_shifted_prediction(
                    scene,
                    full,
                    context["axes"],
                    duration,
                    shift,
                    termination_kind=context["termination_kind"],
                )
                - target
            )
        window_slack = []
        for plan in window_plan:
            residuals = shifted_images[plan["shift_frames"]][plan["indices"]]
            mse = np.mean(np.sum(residuals**2, axis=1))
            window_slack.append(1 - mse / plan["rms_limit_px"] ** 2 if np.isfinite(mse) else -1e6)
        value = (
            float(2 * np.sum(loss((residual / 2) ** 2)[0])),
            model.shared_contact_gaps(flights)[flights_in_block].ravel(),
            image_budgets(image, groups),
            np.concatenate([*gate_slack, np.asarray(window_slack)])
            if constraint_count
            else np.empty(0),
        )
        if len(memo) > 300:
            memo.clear()
        memo[key] = value
        return value

    def evaluate(q):
        try:
            return raw_evaluate(q)
        except ValueError as error:
            # A search proposal can violate simulator feasibility. Penalize
            # these named physical failures, while unknown errors still stop.
            message = str(error)
            if message not in {
                "descending initial ground state has no supported flight",
                "bounce cap reached; ground-clamped continuation is unsupported",
                "measured dynamics bounce cap reached",
            }:
                raise
            invalid_evaluations[message] = invalid_evaluations.get(message, 0) + 1
            return (
                1e12 + float(np.sum(np.asarray(q) ** 2)),
                np.full(9, 1e3),
                np.full(len(groups), 1e12),
                np.full(constraint_count, -1e6),
            )

    q0 = initial[indices] / scale
    baseline = evaluate(q0)
    # The incumbent is always feasible. A contact wing already above 16px may
    # not get worse; all other moving-contact wings remain within that original
    # search objective's 16px budget. Acceptance still uses the full fixed rung.
    budgets = np.maximum(baseline[2] + 1e-8, 16.0**2)
    flat = q0.copy()
    flat[15:] = 0
    single_scene = replace(scene, parameterization="single_shooting")
    original_flights = model.chain(single_scene, parameters)
    trials = []
    for name, start in (("reference", q0), ("flat_spin", flat)):
        began = time.monotonic()
        constraints = [
            dict(type="eq", fun=lambda q: evaluate(q)[1]),
            dict(type="ineq", fun=lambda q: budgets - evaluate(q)[2]),
        ]
        if gate_plan:
            constraints.append(dict(type="ineq", fun=lambda q: evaluate(q)[3]))
        solved = minimize(
            lambda q: evaluate(q)[0],
            start,
            method="SLSQP",
            bounds=list(zip(lower / scale, upper / scale)),
            constraints=constraints,
            options=dict(maxiter=maxiter, ftol=1e-8),
        )
        full = expand(solved.x)
        frozen = np.ones(len(full), bool)
        frozen[indices] = False
        if not np.array_equal(full[frozen], initial[frozen]):
            raise AssertionError("block repair mutated a frozen parameter")
        gaps = model.shared_contact_gaps(model.chain(scene, full))
        contact_slack = budgets - evaluate(solved.x)[2]
        single = model.shared_to_single_parameters(scene, full)
        # Byte-identical external launch/rebound parameters are necessary;
        # sub-millimetre residuals are separately certified before replacement.
        n = len(scene.contact_frames) - 1
        changing_single = np.r_[
            np.arange(3 + 3 * (flight_index - 1), 3 + 3 * (flight_index + 2)),
            np.arange(3 + 3 * n + 3 * (flight_index - 1), 3 + 3 * n + 3 * (flight_index + 2)),
        ]
        unchanged = np.ones(len(single), bool)
        unchanged[changing_single] = False
        if not np.array_equal(single[unchanged], parameters[unchanged]):
            raise AssertionError("export mutated unrelated single-shooting parameters")
        drift = unrelated_path_drift(
            original_flights, model.chain(single_scene, single), flights_in_block
        )
        trials.append(
            dict(
                start=name,
                success=bool(solved.success),
                message=str(solved.message),
                iterations=int(solved.nit),
                function_evaluations=int(solved.nfev),
                cost=float(solved.fun),
                initial_cost=baseline[0],
                maximum_endpoint_gap_m=float(np.linalg.norm(gaps, axis=1).max()),
                endpoint_residuals_m=gaps.tolist(),
                frozen_parameters_identical=True,
                unrelated_export_parameters_identical=True,
                frozen_bounce_feasibility_plan=gate_plan,
                all_block_bounce_plans=gate_plans,
                block_window_plans=window_plan,
                bounce_feasibility_initial_slack=baseline[3].tolist(),
                bounce_feasibility_final_slack=evaluate(solved.x)[3].tolist(),
                bounce_feasibility_valid=bool(
                    not gate_plan or np.min(evaluate(solved.x)[3]) >= -1e-6
                ),
                invalid_physical_evaluations=dict(invalid_evaluations),
                maximum_unrelated_path_drift_m=drift,
                unrelated_path_preserved=bool(drift <= local.ENDPOINT_LIMIT_M),
                contact_image_budgets_rms_px=np.sqrt(budgets).tolist(),
                contact_image_after_rms_px=np.sqrt(evaluate(solved.x)[2]).tolist(),
                contact_image_constraints_valid=bool(np.min(contact_slack) >= -1e-4),
                contact_displacement_xyz_m=(full[indices[:6]] - initial[indices[:6]])
                .reshape(2, 3)
                .tolist(),
                shared_parameters=full.tolist(),
                parameters=single.tolist(),
                wall_seconds=time.monotonic() - began,
            )
        )
    return trials


def run_case(job):
    result = local.run_case(
        job,
        optimizer=partial(
            optimize_block,
            bounce_feasibility=job.get("bounce_feasibility", "off") == "on",
            preserve_block_gates=job.get("preserve_block_gates", "off") == "on",
        ),
    )
    # Fail closed if the optimizer violated an explicit contact image condition,
    # even if the independent, unchanged full-flight gates would accept it.
    if result.get("status") == "measured":
        for trial in result["trials"]:
            trial["eligible_improvement"] &= (
                trial["contact_image_constraints_valid"]
                and trial["unrelated_path_preserved"]
                and trial["bounce_feasibility_valid"]
            )
        improving = [row for row in result["trials"] if row["eligible_improvement"]]
        chosen = min(
            improving,
            key=lambda row: (-row["verdict"]["accepted_flight_count"], row["cost"]),
            default=None,
        )
        result["selected"] = "reference" if chosen is None else chosen["start"]
        result["selected_source"] = "original_reference" if chosen is None else "repair_trial"
        result["selected_trial_start"] = None if chosen is None else chosen["start"]
        result["selected_verdict"] = result["before"] if chosen is None else chosen["verdict"]
    result["experiment"] = "three_flight_movable_internal_contacts"
    output = Path(job["output"]) / f"{job['key']}_f{job['flight_index']:02d}"
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument(
        "--case", action="append", required=True, help="attempt_key:interior_flight"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--bounce-feasibility",
        choices=("off", "on"),
        default="off",
        help="Explicitly constrain target to its unchanged frozen bounce ray/time gates",
    )
    parser.add_argument(
        "--preserve-block-gates",
        choices=("off", "on"),
        default="off",
        help="Constrain already accepted neighbor bounce gates and all block image windows",
    )
    args = parser.parse_args()
    if args.maxiter < 1 or args.timeout < 1:
        parser.error("positive iteration and time budgets required")
    args.output.mkdir(parents=True, exist_ok=False)
    source_records = [
        provenance.file_record(Path(__file__)),
        provenance.file_record(Path(local.__file__)),
    ]
    for source_path in (Path(__file__), Path(local.__file__)):
        (args.output / source_path.name).write_bytes(source_path.read_bytes())
    jobs = []
    for case in args.case:
        key, flight = case.rsplit(":", 1)
        jobs.append(
            dict(
                key=key,
                flight_index=int(flight),
                baseline_report=str(args.baseline_report),
                rung=args.rung,
                output=str(args.output),
                maxiter=args.maxiter,
                timeout=args.timeout,
                bounce_feasibility=args.bounce_feasibility,
                preserve_block_gates=args.preserve_block_gates,
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, jobs))
    report = dict(
        schema="three_flight_block_repair_probe_v1",
        scope=__doc__,
        jobs=jobs,
        results=results,
        sources=source_records,
        code=provenance.git_record(paths.REPO_ROOT),
        human_derived=True,
        automatic_inference_eligible=False,
        promoted=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                dict(
                    key=row["job"]["key"],
                    flight=row["job"]["flight_index"],
                    status=row["status"],
                    selected=row.get("selected"),
                    blocker=row.get("blocker"),
                )
                for row in results
            ]
        )
    )


if __name__ == "__main__":
    main()
