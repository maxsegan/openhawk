"""Bounded two-flight serve repair with free initial contact and fixed right end.

Opened-development only. Preserve original serve depth bounds, stature/reach
and native player/ball evidence. Use available native pre-contact toss evidence
as an explicitly uncertain soft contact estimate; never invent missing toss
frames. Existing frozen chord witnesses may keep their original acceptance
route; no new fallback or independent XYZ truth is introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    athlete_priors,
    block_flight_repair_probe as block,
    event_constraints,
    local_flight_repair_probe as local,
    model,
    joint_toss_residual,
    real_exposure_replay as exposure,
    serve_reach_cylinder,
    toss_witness,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance


def serve_indices(scene):
    if len(scene.contact_frames) - 1 != 2:
        raise ValueError("this bounded arm requires exactly two flights")
    slices = model.shared_parameter_slices(scene)
    return np.r_[
        np.arange(6),
        np.arange(slices["velocities"].start, slices["velocities"].stop),
        np.arange(slices["spins"].start, slices["spins"].stop),
    ]


def existing_serve_bounce_plan(context):
    """Target the exact old chord only if it existed in the frozen baseline."""
    flight = context["repair_reference_verdict"]["flights"][0]
    threshold = context["repair_thresholds"]
    ray_limit = threshold.serve_bounce_ray_limit_m or threshold.bounce_ray_limit_m
    plan = []
    for witness in flight["bounce_witness"]:
        legacy = witness.get("legacy_subframe_witness")
        if not legacy or not legacy.get("available", False):
            raise ValueError("serve boundary probe requires an existing frozen chord")
        plan.append(
            dict(
                xy_m=legacy["xyz_m"][:2],
                event_frame=witness["supplied_frame"],
                maximum_distance_m=ray_limit + witness["legacy_graded_circle_radius_m"] - 1e-4,
                maximum_time_error_frames=threshold.bounce_uncertainty_frames - 1e-5,
                witness_route="existing_frozen_legacy_not_new_fallback",
                primary_covariance_resolved=witness["covariance_resolved"],
            )
        )
    if not plan:
        raise ValueError("serve boundary probe needs a frozen bounce witness")
    return plan


def serve_contact_image_groups(scene):
    start, contact, _ = scene.contact_frames
    first, second = scene.observation_frames
    groups = [
        np.flatnonzero(first <= start + 4),
        np.flatnonzero(first >= contact - 4),
        len(first) + np.flatnonzero(second <= contact + 4),
    ]
    if any(not len(group) for group in groups):
        raise ValueError("serve or internal contact lacks original native image support")
    return groups


def toss_evidence(context, labels, cameras, pose_csv, pose_scale, low, high):
    contact = next(e for e in context["events"] if e["event_type"] == "contact")
    observations = toss_witness.contact_constraint_observations(
        labels,
        cameras,
        clip=context["attempt"]["point_clip"],
        contact_frame=contact["frame"],
        contact_frame_interval=tuple(contact["frame_interval"]),
    )
    if observations["status"] != "supported":
        return dict(
            status="abstained",
            observations=observations,
            estimate=None,
            note="No precontact wing invented; original outgoing/player constraints retained.",
        )
    pixel = np.asarray(context["scene"].pixels[0][0])
    feet = toss_witness.server_feet(
        pose_csv,
        context["attempt"]["point_clip"],
        contact["frame"],
        pixel,
        image_coordinate_scale=pose_scale,
        observation_fallback=True,
        fallback_receipt=[],
    )
    estimate = toss_witness.fit_contact(
        observations,
        feet,
        context["scene"].fps,
        contact_frame_interval=tuple(contact["frame_interval"]),
        contact_bounds=(low, high),
        config=toss_witness.CONTACT_CONSTRAINT_CONFIG,
    )
    if estimate["status"] != "supported":
        raise ValueError(
            "available toss contact inference held: " + ",".join(estimate.get("failures", []))
        )
    return dict(
        status="supported",
        observations=observations,
        feet=feet,
        estimate=estimate,
        note="Toss-derived uncertain soft contact, not independent XYZ truth.",
    )


def optimize_serve(
    context,
    parameters,
    flight_index,
    *,
    maxiter,
    bounce_bracket_frames,
    duration,
    labels,
    cameras,
    pose_csv,
    pose_scale,
    raw_toss=False,
):
    if flight_index != 0:
        raise ValueError("serve boundary optimizer targets flight zero only")
    scene = replace(context["scene"], parameterization="shared_contact_states")
    initial = (
        np.asarray(context["repair_initial_shared_parameters"], float).copy()
        if "repair_initial_shared_parameters" in context
        else model.shared_contact_seed(scene, parameters)
    )
    indices = serve_indices(scene)
    cylinder = serve_reach_cylinder.build(context["players"][0])
    centre = np.asarray(cylinder.centre_xy_m)
    lower = np.r_[initial[indices[:6]] - 1.5, [-75.0] * 6, [-6.0] * 6]
    upper = np.r_[initial[indices[:6]] + 1.5, [75.0] * 6, [6.0] * 6]
    lower[0] = max(lower[0], centre[0] - cylinder.radius_m)
    upper[0] = min(upper[0], centre[0] + cylinder.radius_m)
    lower[1] = max(context["depth_bounds"][0], centre[1] - cylinder.radius_m)
    upper[1] = min(context["depth_bounds"][1], centre[1] + cylinder.radius_m)
    lower[2], upper[2] = cylinder.height_interval_m
    lower[5] = max(lower[5], model.R_BALL + 1e-4)
    upper[5] = min(upper[5], 4.0)
    if np.any(initial[indices] < lower) or np.any(initial[indices] > upper):
        raise ValueError("incumbent outside original stature/reach/depth bounds")
    toss = (
        context["repair_toss_evidence"]
        if "repair_toss_evidence" in context
        else toss_evidence(context, labels, cameras, pose_csv, pose_scale, lower[:3], upper[:3])
    )
    if raw_toss and toss["status"] != "supported":
        raise ValueError("joint raw toss requires supported original evidence")
    plans = {
        0: existing_serve_bounce_plan(context),
        1: block.accepted_neighbor_bounce_plan(context, 1),
    }
    windows = block.block_window_plan(context, [0, 1])
    groups = serve_contact_image_groups(scene)
    target = np.concatenate(scene.pixels)
    scale = np.r_[[1.0] * 6, [30.0] * 6, [3.0] * 6]
    if raw_toss:
        first_frame = toss["observations"]["rows"][0]["frame"]
        minimum_tau = max((scene.contact_frames[0] - first_frame + 1) / scene.fps, 0.30)
        lower = np.r_[lower, [-8, -8, -8, minimum_tau]]
        upper = np.r_[upper, [8, 8, 5, max(1.25, minimum_tau + 0.05)]]
        scale = np.r_[scale, [3.0, 3.0, 3.0, 1.0]]
    loss = event_constraints.mixed_loss(2 * len(target))
    cache = FlightCache(entries_per_flight=80)
    anchor = exposure.anchor_plan(context["targets"], [None] * 2)
    memo = {}
    invalid = {}
    constraint_count = (
        sum(2 * len(plan) + 1 for plan in plans.values())
        + len(windows)
        + 1
        + (7 if raw_toss else 0)
    )

    def expand(q):
        full = initial.copy()
        full[indices] = (np.asarray(q) * scale)[:18]
        return full

    def evaluate(q):
        key = np.asarray(q).tobytes()
        if key in memo:
            return memo[key]
        try:
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
            contacts = full[model.shared_parameter_slices(scene)["contacts"]].reshape(3, 3)
            anchors = []
            for i in [0, 1]:
                for j, (xy, sigma) in enumerate(anchor["bounce"][i]):
                    anchors.extend(
                        (np.asarray(flights[i]["bounces"][j]["x"])[:2] - xy) / sigma
                        if j < len(flights[i]["bounces"])
                        else [0.0, 0.0]
                    )
            soft = athlete_priors.optimization_residuals(
                contacts[: len(context["players"])], context["players"]
            )
            if toss["status"] == "supported" and not raw_toss:
                estimate = toss["estimate"]
                soft = np.r_[
                    soft, (contacts[0] - estimate["contact_xyz_m"]) / estimate["contact_sigma_m"]
                ]
            raw_residual = []
            if raw_toss:
                raw_residual, raw_slack, _ = joint_toss_residual.evaluate(
                    contacts[0],
                    (np.asarray(q) * scale)[18:],
                    toss["observations"],
                    toss["feet"],
                    scene.fps,
                )
            residual = np.r_[image.ravel(), physical, anchors, soft, raw_residual]
            slack = [
                block.bounce_feasibility_slack(flights[i]["bounces"], plan)
                for i, plan in plans.items()
            ]
            if raw_toss:
                slack.append(raw_slack)
            shifted = {0.0: image}
            for shift in {p["shift_frames"] for p in windows} - {0.0}:
                shifted[shift] = (
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
            for plan in windows:
                errors = shifted[plan["shift_frames"]][plan["indices"]]
                mse = np.mean(np.sum(errors**2, axis=1))
                slack.append(
                    np.array([1 - mse / plan["rms_limit_px"] ** 2 if np.isfinite(mse) else -1e6])
                )
            slack.append(
                np.array([1 - np.sum((contacts[0, :2] - centre) ** 2) / cylinder.radius_m**2])
            )
            value = (
                float(2 * np.sum(loss((residual / 2) ** 2)[0])),
                model.shared_contact_gaps(flights).ravel(),
                block.image_budgets(image, groups),
                np.concatenate(slack),
            )
        except ValueError as error:
            message = str(error)
            if message not in {
                "descending initial ground state has no supported flight",
                "bounce cap reached; ground-clamped continuation is unsupported",
                "measured dynamics bounce cap reached",
            }:
                raise
            invalid[message] = invalid.get(message, 0) + 1
            value = (
                1e12 + float(np.sum(np.asarray(q) ** 2)),
                np.full(6, 1e3),
                np.full(3, 1e12),
                np.full(constraint_count, -1e6),
            )
        if len(memo) > 250:
            memo.clear()
        memo[key] = value
        return value

    initial_variables = initial[indices]
    if raw_toss:
        conditioned = toss_witness.fit(initial[:3], toss["observations"], toss["feet"], scene.fps)
        initial_variables = np.r_[
            initial_variables,
            conditioned["incoming_contact_velocity_mps"],
            conditioned["release_seconds_before_contact"],
        ]
    q0 = initial_variables / scale
    baseline = evaluate(q0)
    budgets = np.asarray(
        context.get("repair_contact_image_budgets", np.maximum(baseline[2] + 1e-8, 16.0**2))
    )
    starts = [("reference", q0)]
    if toss["status"] == "supported":
        proposed = q0.copy()
        proposed[:3] = np.clip(toss["estimate"]["contact_xyz_m"], lower[:3], upper[:3])
        if raw_toss:
            estimate = toss["estimate"]
            proposed[18:] = (
                np.r_[
                    estimate["incoming_contact_velocity_mps"],
                    estimate["release_seconds_before_contact"],
                ]
                / scale[18:]
            )
        starts.append(("toss_contact", proposed))
    trials = []
    for name, start in starts:
        began = time.monotonic()
        solved = minimize(
            lambda q: evaluate(q)[0],
            start,
            method="SLSQP",
            bounds=list(zip(lower / scale, upper / scale)),
            constraints=[
                dict(type="eq", fun=lambda q: evaluate(q)[1]),
                dict(type="ineq", fun=lambda q: budgets - evaluate(q)[2]),
                dict(type="ineq", fun=lambda q: evaluate(q)[3]),
            ],
            options=dict(maxiter=maxiter, ftol=1e-8),
        )
        full = expand(solved.x)
        frozen = np.ones(len(full), bool)
        frozen[indices] = False
        if not np.array_equal(full[frozen], initial[frozen]):
            raise AssertionError("serve repair mutated frozen endpoint/rebound")
        gaps = model.shared_contact_gaps(model.chain(scene, full))
        single = model.shared_to_single_parameters(scene, full)
        raw_record = None
        if raw_toss:
            _, _, raw_record = joint_toss_residual.evaluate(
                full[:3], (solved.x * scale)[18:], toss["observations"], toss["feet"], scene.fps
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
                maximum_unrelated_path_drift_m=0.0,
                contact_image_constraints_valid=bool(
                    np.min(budgets - evaluate(solved.x)[2]) >= -1e-4
                ),
                all_frozen_gate_constraints_valid=bool(np.min(evaluate(solved.x)[3]) >= -1e-6),
                gate_slack=evaluate(solved.x)[3].tolist(),
                original_depth_bounds_m=list(context["depth_bounds"]),
                serve_reach=cylinder.record(),
                toss_evidence=toss,
                joint_raw_toss=raw_record,
                raw_toss_constraints_valid=(
                    not raw_toss
                    or (bool(solved.success) and all(raw_record["constraints"].values()))
                ),
                bounce_plans=plans,
                contact_displacement_xyz_m=(full[:6] - initial[:6]).reshape(2, 3).tolist(),
                first_contact_xyz_m=single[:3].tolist(),
                shared_parameters=full.tolist(),
                parameters=single.tolist(),
                invalid_physical_evaluations=dict(invalid),
                wall_seconds=time.monotonic() - began,
            )
        )
    return trials


def finalize_selection(result):
    """Apply extra serve constraints and keep every selection field consistent."""
    if result["status"] != "measured":
        return
    if result["status"] == "measured":
        for trial in result["trials"]:
            trial["eligible_improvement"] &= (
                trial["contact_image_constraints_valid"]
                and trial["all_frozen_gate_constraints_valid"]
                and trial.get("raw_toss_constraints_valid", True)
            )
        chosen = min(
            (t for t in result["trials"] if t["eligible_improvement"]),
            key=lambda t: (-t["verdict"]["accepted_flight_count"], t["cost"]),
            default=None,
        )
        result.update(
            selected="reference" if chosen is None else chosen["start"],
            selected_source="original_reference" if chosen is None else "repair_trial",
            selected_trial_start=None if chosen is None else chosen["start"],
            selected_verdict=result["before"] if chosen is None else chosen["verdict"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-toss",
        choices=["on", "off"],
        default="off",
        help="Join original native toss pixels with original physical checks; no soft XYZ target.",
    )
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.maxiter < 1 or args.timeout < 1:
        parser.error("positive iteration and time budgets required")
    args.output.mkdir(parents=True, exist_ok=False)
    aggregate = json.loads(args.baseline_report.read_text())
    item = next(a for a in aggregate["attempts"] if a["key"] == args.case)
    perflight = json.loads(Path(item["document"]).read_text())
    source = local._resolve(perflight["search_report"]).parent.parent
    labelpath = (
        paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1" / item["metadata"]["label_file"]
    )
    camerapath = source / "inputs/cameras.json"
    labels = json.loads(labelpath.read_text())
    cameras = json.loads(camerapath.read_text())
    sources = [
        provenance.file_record(p)
        for p in [
            Path(__file__),
            Path(local.__file__),
            Path(block.__file__),
            Path(joint_toss_residual.__file__),
            Path(toss_witness.__file__),
            labelpath,
            camerapath,
        ]
    ]
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    job = dict(
        key=args.case,
        flight_index=0,
        baseline_report=str(args.baseline_report),
        rung=args.rung,
        output=str(args.output),
        maxiter=args.maxiter,
        timeout=args.timeout,
        raw_toss=args.raw_toss,
    )
    result = local.run_case(
        job,
        optimizer=partial(
            optimize_serve,
            labels=labels,
            cameras=cameras,
            pose_csv=paths.data_root() / "processed" / item["metadata"]["player_localization"],
            pose_scale=item["metadata"]["pose_image_scale"],
            raw_toss=args.raw_toss == "on",
        ),
    )
    finalize_selection(result)
    result.update(
        schema="two_flight_serve_block_repair_v1", sources=sources, promoted=False, scope=__doc__
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / f"{args.case}_f00" / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result["status"], result.get("selected_source"), result.get("blocker"))


if __name__ == "__main__":
    main()
