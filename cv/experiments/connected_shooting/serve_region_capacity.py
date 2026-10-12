"""Does a known-valid initial-contact sphere improve complete synthetic trajectories?

Evaluation-only upper-bound experiment. The sphere is centred on the generating
first contact, NOT on a recovered player. This deliberately privileged region
tests S6 capacity, not an anatomical bound or real-video accuracy. Only seeds
outside the explicit sphere are repaired; every source point stays in the count.
The paired restart has identical terminal constraints and no contact constraint.
Neither optimizer success nor sphere membership constitutes point acceptance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
from pathlib import Path
import signal
import time

import numpy as np

from cv.experiments.connected_shooting import (
    model,
    oracle_benchmark as oracle,
    physical_audit,
    physical_compatibility,
    pose_contact_feasibility as contact,
    terminal_completion,
    terminal_feasibility as terminal,
)
from cv.pipeline import paths, provenance
from cv.validation import s6_owner_ground_camera as ground


def sphere_slack(position, center, radius):
    if isinstance(radius, (bool, np.bool_)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("explicit positive finite sphere radius required")
    return float(radius - contact.wrist_distance(position, [center]))


def score(point, scene, heldout, truth_paths, parameters, bounces, native):
    """Reuse the frozen spatial, impact, net, passivity and completed-ending checks."""
    parameters = np.asarray(parameters, float)
    connected = model.chain(scene, parameters)
    state = dict(
        parameters=parameters,
        junction_gaps_m=[
            float(np.linalg.norm(a["end_xyz"] - b["start_xyz"]))
            for a, b in zip(connected, connected[1:])
        ],
        optimizer_success=False,
        objective_calls=0,
        final_pixel_rms=float(np.sqrt(np.mean(model.image_residual(scene, parameters) ** 2))),
    )
    spatial = oracle.measure(scene, state, truth_paths, heldout)
    completion = terminal_completion.complete(
        scene,
        parameters,
        point["termination_kind"],
        uncertainty_frames=1,
        last_observation_frame=float(native[-1][-1]),
    )
    audit = physical_audit.audit_point(
        point,
        scene,
        parameters,
        truth_paths,
        spatial["trajectory_and_endpoint_tolerances_met"],
        fitted_end_frame=completion["end_frame"] if completion["status"] == "completed" else None,
    )
    audit["conditions"]["bounded_ground_completion"] = completion["status"] == "completed"
    if completion["status"] == "completed":
        error = float(
            np.linalg.norm(np.asarray(completion["end_xyz"]) - truth_paths[-1]["positions"][-1])
        )
        audit["conditions"]["completed_endpoint_position"] = (
            error <= oracle.CONTRACT["termination_tolerance_m"]
        )
    compatible = physical_compatibility.evaluate(
        scene,
        parameters,
        bounces,
        point["termination_kind"],
        native,
    )
    return dict(
        spatial=spatial,
        physical_conditions=audit["conditions"],
        input_compatible=compatible["compatible"],
        input_failures=compatible["failures"],
        synthetic_correct=bool(all(audit["conditions"].values())),
        completion=completion,
    )


def prepare(point, report):
    cfg = report["configuration"]
    camera = Path(cfg["cameras_root"]) / point["match_id"] / "camera_P_per_frame_v1.npz"
    scene, _, _, truth_paths, heldout = oracle.prepare(point, camera)
    scene, heldout = [oracle.configured_scene(s, cfg) for s in (scene, heldout)]
    scene = oracle.noisy_pixels(scene, point["point"], cfg["noise_seed"])
    bounces, _ = oracle.in_scope_bounce_evidence(point, scene)
    native = tuple(
        np.unique(np.r_[a, b])
        for a, b in zip(scene.observation_frames, heldout.observation_frames, strict=True)
    )
    return scene, heldout, truth_paths, bounces, native


def run_point(point, report, old, old_audit, radius, maxiter, timeout):
    started = time.monotonic()
    row = dict(point=point["point"], status="held", arms={}, complete_real_point_accepted=False)
    scene, heldout, truth_paths, bounces, native = prepare(point, report)
    initial = np.array(old["parameters"])
    center = truth_paths[0]["positions"][0].copy()
    baseline = score(point, scene, heldout, truth_paths, initial, bounces, native)
    if baseline["synthetic_correct"] != old_audit["oracle_physical_diagnostic_met"] or any(
        not np.isclose(a["trajectory_rms_m"], b["trajectory_rms_m"], rtol=1e-8, atol=1e-8)
        for a, b in zip(baseline["spatial"]["flights"], old["flights"], strict=True)
    ):
        raise ValueError("frozen baseline geometry or complete-point score did not reproduce")
    original_cost = terminal.objective(
        scene, initial, bounces, float(native[-1][-1]), terminal_guidance=False
    )
    if not np.isclose(original_cost, old["optimizer_evidence"]["cost"], rtol=1e-8, atol=1e-7):
        raise ValueError("original image/event/rebound objective did not reproduce")
    slack = sphere_slack(initial[:3], center, radius)
    row.update(
        status="measured",
        baseline=baseline,
        sphere_center_xyz=center.tolist(),
        radius_m=radius,
        initial_sphere_slack_m=slack,
        repair_triggered=slack < 0,
        original_cost=original_cost,
        baseline_numerically_reproduced=True,
        oracle_contact_center=True,
        observed_player_input=False,
    )
    if slack >= 0:
        row["policy"] = "retain_original_inside_sphere_without_refitting"
    else:

        def deadline(*_):
            raise TimeoutError("bounded per-arm contact-capacity refinement")

        for arm in ("terminal_only", "contact_sphere"):
            before = time.monotonic()
            previous = signal.signal(signal.SIGALRM, deadline)
            signal.alarm(timeout)
            try:
                if arm == "contact_sphere":
                    fit = contact.refine(
                        scene,
                        initial,
                        bounces,
                        float(native[-1][-1]),
                        [center],
                        radius,
                        maxiter=maxiter,
                        guide=False,
                    )
                else:
                    fit = terminal.refine(
                        scene,
                        initial,
                        bounces,
                        float(native[-1][-1]),
                        maxiter=maxiter,
                        terminal_guidance=False,
                    )
                measured = score(
                    point, scene, heldout, truth_paths, fit["parameters"], bounces, native
                )
                row["arms"][arm] = dict(
                    status="measured",
                    fit=fit,
                    score=measured,
                    sphere_slack_m=sphere_slack(fit["parameters"][:3], center, radius),
                )
            except (ValueError, FloatingPointError, OverflowError, TimeoutError) as error:
                row["arms"][arm] = dict(status="held", reason=str(error))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
                row["arms"][arm]["wall_seconds"] = time.monotonic() - before
    row["wall_seconds"] = time.monotonic() - started
    return row


def summarize(rows):
    baseline = {r["point"] for r in rows if r.get("baseline", {}).get("synthetic_correct", False)}
    output = dict(
        points=len(rows),
        baseline_correct=len(baseline),
        repair_triggered=sum(r.get("repair_triggered", False) for r in rows),
        arms={},
    )
    for name in ("terminal_only", "contact_sphere"):
        correct, held, feasible, converged = set(), 0, 0, 0
        for row in rows:
            if row["status"] != "measured":
                held += 1
            elif not row["repair_triggered"]:
                if row["baseline"]["synthetic_correct"]:
                    correct.add(row["point"])
            else:
                arm = row["arms"][name]
                if arm["status"] != "measured":
                    held += 1
                    continue
                # This numerical membership check is reported separately, not an accuracy threshold.
                member = arm["sphere_slack_m"] >= -1e-6
                feasible += member
                converged += bool(arm["fit"]["optimizer"]["success"])
                if arm["score"]["synthetic_correct"] and (name != "contact_sphere" or member):
                    correct.add(row["point"])
        output["arms"][name] = dict(
            synthetic_correct=len(correct),
            gained=sorted(correct - baseline),
            lost=sorted(baseline - correct),
            held=held,
            repaired_sphere_members=feasible,
            optimizer_converged=converged,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Frozen report.json; adjacent physical_audit.json required",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    sphere_slack([0, 0, 0], [0, 0, 0], args.radius_m)
    if args.output.exists() or min(args.maxiter, args.timeout, args.jobs) <= 0:
        raise ValueError("new output and positive resource limits required")
    report = json.loads(args.baseline.read_text())
    audit_path = args.baseline.with_name("physical_audit.json")
    audit = json.loads(audit_path.read_text())
    cfg = report["configuration"]
    if (
        report["status"] != "complete"
        or cfg["arms"] != ["pixels_noise2"]
        or cfg["training_visibility"] != "all"
        or cfg["bounce_time_conditioning"] != "exact"
        or cfg["rebound_mode"] != "point_scales"
    ):
        raise ValueError("completed all-visible exact-event noisy point-scale baseline required")
    if (
        provenance.file_record(args.baseline) not in audit["inputs"]
        or audit["ground_completion_uncertainty_frames"] != 1
    ):
        raise ValueError("matching frozen complete-ground physical audit required")
    truth_path = Path(cfg["truth"])
    if provenance.file_record(truth_path) != report["truth"]:
        raise ValueError("frozen truth bytes changed")
    points = {p["point"]: p for p in json.loads(truth_path.read_text())["points"]}
    old = {r["point"]: r for r in report["results"]}
    old_audit = {r["point"]: r for r in audit["results"]}
    inventory = report["point_inventory"]
    if (
        len(set(inventory)) != len(inventory)
        or set(old) != set(inventory)
        or set(old_audit) != set(inventory)
        or len(old) != len(report["results"])
        or len(old_audit) != len(audit["results"])
    ):
        raise ValueError("complete unique baseline/audit point inventory required")
    if any(r["status"] != "measured" for r in old.values()):
        raise ValueError(
            "this replay requires a fully measured baseline; do not drop failed source rows"
        )
    # Historical producer code is not asserted to match main. Immutable data bindings
    # and actual baseline geometry/objective/score are independently checked above.
    data = [r for r in report["inputs"] if r["path"].endswith(".npz")]
    for record in data:
        if provenance.file_record(ground.resolve_record(record)) != record:
            raise ValueError("frozen camera bytes changed")
    camera_files = [
        Path(cfg["cameras_root"]) / points[p]["match_id"] / "camera_P_per_frame_v1.npz"
        for p in inventory
    ]
    if any(provenance.file_record(p) not in data for p in camera_files):
        raise ValueError("every camera must be explicitly bound by baseline")
    current_dependencies = []
    for record in report["inputs"]:
        if not record["path"].endswith(".py"):
            continue
        current_path = (paths.REPO_ROOT / record["path"]).resolve()
        if record["path_base"] != "repository" or not current_path.is_relative_to(
            paths.REPO_ROOT.resolve()
        ):
            raise ValueError("explicit repository dependency required for current-code replay")
        current_dependencies.append(current_path)
    files = list(
        dict.fromkeys(
            [
                args.baseline,
                audit_path,
                truth_path,
                *camera_files,
                *current_dependencies,
                Path(__file__),
                *[
                    Path(m.__file__)
                    for m in (
                        model,
                        oracle,
                        physical_audit,
                        physical_compatibility,
                        contact,
                        terminal_completion,
                        terminal,
                    )
                ],
            ]
        )
    )
    bindings = [provenance.file_record(p) for p in files]
    result = dict(
        schema="s6_serve_region_capacity_v1",
        scope=__doc__,
        inputs=bindings,
        code=provenance.git_record(paths.REPO_ROOT),
        configuration=vars(args),
        status="running",
        oracle_contact_center=True,
        automatic_inference_eligible=False,
        complete_real_points_accepted=0,
        results=[],
    )
    args.output.mkdir(parents=True)

    def write():
        (args.output / "report.json").write_text(
            json.dumps(
                result,
                default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    write()
    with (
        ProcessPoolExecutor(max_workers=args.jobs) as pool,
        (args.output / "progress.jsonl").open("w") as log,
    ):
        tasks = {
            pool.submit(
                run_point,
                points[p],
                report,
                old[p],
                old_audit[p],
                args.radius_m,
                args.maxiter,
                args.timeout,
            ): p
            for p in inventory
        }
        for future in as_completed(tasks):
            row = (
                future.result()
            )  # source/numerical replay errors abort, never silently become passes
            result["results"].append(row)
            progress = dict(
                point=row["point"],
                repair_triggered=row["repair_triggered"],
                baseline_correct=row["baseline"]["synthetic_correct"],
                arms={
                    k: dict(
                        status=v["status"],
                        correct=v.get("score", {}).get("synthetic_correct"),
                        optimizer=v.get("fit", {}).get("optimizer", {}).get("success"),
                    )
                    for k, v in row["arms"].items()
                },
            )
            log.write(json.dumps(progress) + "\n")
            log.flush()
            print(json.dumps(progress), flush=True)
            write()
    if bindings != [provenance.file_record(p) for p in files]:
        raise ValueError("capacity inputs changed during execution")
    result["results"].sort(key=lambda r: inventory.index(r["point"]))
    result.update(status="complete", summary=summarize(result["results"]))
    write()
    body = html.escape(json.dumps(result["summary"], indent=2))
    (args.output / "index.html").write_text(
        f'<!doctype html><title>S6 oracle serve sphere</title><h1>Known-valid contact region: synthetic capacity only</h1><p>The centre is generating truth, not a player estimate. All points remain in the denominator. Baseline retention inside the region is explicit; failed repairs are held. No real point is accepted.</p><pre>{body}</pre><p><a href="report.json">Frozen inputs, every condition and full physical/spatial checks</a></p>'
    )
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
