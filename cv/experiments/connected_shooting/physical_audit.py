"""Audit frozen Stage 6 oracle fits and truth for physical point defects.

Keeps the frozen inventory, including impossible synthetic targets. Does not
change a fit, snap endpoints, inject events or authorize production acceptance.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import html
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import model, oracle_benchmark
from cv.experiments.connected_shooting.physical_compatibility import crossings, impact_passivity
from cv.pipeline import camera_cal, trajectory_contract
from cv.pipeline.provenance import file_record, git_record


def physical_queries(
    scene: model.Scene,
    paths: list[dict],
    fitted_end_frame: float | None = None,
) -> tuple[model.Scene, tuple[np.ndarray, ...]]:
    """Sample the returned physical domain, without changing spatial truth scoring."""
    queries = [p["frames"] for p in paths]
    if fitted_end_frame is not None:
        times = scene.contact_frames.copy()
        times[-1] = fitted_end_frame
        scene = replace(scene, contact_frames=times)
        scene.validate()
        original_end = float(paths[-1]["frames"][-1])
        extra = np.arange(original_end, fitted_end_frame, scene.fps / 240.0)
        queries[-1] = np.unique(
            np.r_[queries[-1][queries[-1] <= fitted_end_frame], extra, fitted_end_frame]
        )
    return scene, tuple(queries)


def audit_point(
    point: dict,
    scene: model.Scene,
    parameters: np.ndarray,
    paths: list[dict],
    spatial_pass: bool,
    *,
    fitted_end_frame: float | None = None,
) -> dict:
    scene, queries = physical_queries(scene, paths, fitted_end_frame)
    fitted = model.chain(scene, parameters, query_frames=queries)
    # The numerical endpoint tolerance must also apply when asking whether an
    # impact occurred. A last impact 1e-12 frames beyond an exact end must not
    # vanish from the event list while its endpoint passes the ground contract.
    # This probes existing dynamics only; returned positions/end times are NOT
    # changed, no image is added, and no ordinary event-time uncertainty is used.
    probed_bounces = endpoint_bounce_probe(scene, parameters)
    flights = []
    contracts = []
    for i, (fit, path, target) in enumerate(zip(fitted, paths, point["flights"], strict=True)):
        expected = [b for b in target.get("bounces", []) if b["frame"] <= path["frames"][-1] + 1e-6]
        candidates = probed_bounces if i == len(paths) - 1 else fit["bounces"]
        observed = [b for b in candidates if b["frame"] <= fit["end_frame"] + 1e-6]
        pairs = [
            {
                "frame_error": abs(float(a["frame"]) - float(b["frame"])),
                "position_error_m": float(np.linalg.norm(np.asarray(a["xyz"]) - b["x"])),
            }
            for a, b in zip(expected, observed)
        ]
        bounce_ok = len(expected) == len(observed) and all(
            p["frame_error"] <= 1.0 and p["position_error_m"] <= 0.3048 for p in pairs
        )
        net = crossings(queries[i], fit["positions"])
        truth_net = crossings(path["frames"], path["positions"])
        flights.append(
            {
                "flight_index": i,
                "expected_bounces": len(expected),
                "modeled_bounces": len(observed),
                "bounce_pairs": pairs,
                "bounce_contract_met": bounce_ok,
                "net_crossings": net,
                "truth_net_crossings": truth_net,
                "minimum_height_m": float(fit["positions"][:, 2].min()),
                "impact_passivity": [
                    impact_passivity(b, fit["end_frame"], scene.fps) for b in observed
                ],
            }
        )
        contracts.append(
            {
                "start_frame": fit["start_frame"],
                "end_frame": fit["end_frame"],
                "start_xyz": fit["start_xyz"].tolist(),
                "end_xyz": fit["end_xyz"].tolist(),
                "terminal_end": i == len(paths) - 1,
            }
        )
    connection = trajectory_contract.trajectory_connection_report(contracts)
    ending = trajectory_contract.terminal_ground_endpoint_report(
        point["termination_kind"], contracts
    )
    truth_clear = not any(c["penetration"] for f in flights for c in f["truth_net_crossings"])
    conditions = {
        "trajectory_and_endpoint_tolerances": bool(spatial_pass),
        "structural_connections": connection["valid"],
        "bounce_count_timing_position": all(f["bounce_contract_met"] for f in flights),
        "no_net_penetration": not any(
            c["penetration"] for f in flights for c in f["net_crossings"]
        ),
        "no_underground_path": all(f["minimum_height_m"] >= model.R_BALL - 1e-3 for f in flights),
        "physical_ground_ending": ending["valid"],
        "synthetic_truth_no_net_penetration": truth_clear,
        "passive_court_impacts": all(b["valid"] for f in flights for b in f["impact_passivity"]),
    }
    return {
        "conditions": conditions,
        "oracle_physical_diagnostic_met": all(conditions.values()),
        "failures": [k for k, v in conditions.items() if not v],
        "flights": flights,
        "connections": connection,
        "ending": ending,
        "modeled_ending": contracts[-1],
        "complete_real_point_accepted": False,
        "scope": "oracle-known topology/start; sampled net/ground and internal impact-energy screen, not independent law/reach certificate",
        "endpoint_event_probe_tolerance_frames": trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES,
    }


def endpoint_bounce_probe(scene: model.Scene, parameters: np.ndarray) -> list[dict]:
    times = scene.contact_frames.copy()
    times[-1] += trajectory_contract.TIME_NUMERICAL_TOLERANCE_FRAMES
    return model.chain(replace(scene, contact_frames=times), parameters)[-1]["bounces"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--cameras-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-compatibility",
        action="store_true",
        help="Separately check supplied-input physical compatibility without spatial truth",
    )
    parser.add_argument(
        "--ground-completion-frames",
        type=float,
        help="explicit research ending uncertainty, at most one native frame; no refit",
    )
    args = parser.parse_args()
    if args.input_compatibility and args.ground_completion_frames is None:
        parser.error("input compatibility comparison requires explicit ground completion")
    if args.ground_completion_frames is not None and not 0 < args.ground_completion_frames <= 1:
        parser.error("ground completion requires uncertainty in (0, 1] native frames")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input_compatibility and args.output.with_suffix(".html").exists():
        raise FileExistsError(args.output.with_suffix(".html"))
    report = json.loads(args.report.read_text())
    if args.cameras_root.resolve() != Path(report["configuration"]["cameras_root"]).resolve():
        raise ValueError("physical audit must use the frozen generating-camera root")
    if file_record(args.truth)["sha256"] != report["truth"]["sha256"]:
        raise ValueError("frozen report truth mismatch")
    truth = {p["point"]: p for p in json.loads(args.truth.read_text())["points"]}
    sources = [
        args.report,
        args.truth,
        Path(oracle_benchmark.s6_point_bench.__file__),
        Path(oracle_benchmark.owner_spatial_audit.__file__),
        Path(__file__),
        Path(model.__file__),
        Path(oracle_benchmark.__file__),
        Path(camera_cal.__file__),
        Path(trajectory_contract.__file__),
    ]
    sources.extend(
        args.cameras_root / truth[p]["match_id"] / "camera_P_per_frame_v1.npz"
        for p in report["point_inventory"]
    )
    from cv.experiments.connected_shooting import measured_dynamics
    from cv.experiments.connected_shooting import terminal_completion
    from cv.experiments.connected_shooting import physical_compatibility, event_constraints

    sources.extend(
        [
            Path(measured_dynamics.__file__),
            Path(terminal_completion.__file__),
            Path(physical_compatibility.__file__),
            Path(event_constraints.__file__),
            Path(measured_dynamics.bounce_reference.__file__),
            Path(model.rich_ball_physics.__file__),
            Path(model.rich_ball_physics.pe.__file__),
            Path(model.rich_ball_physics.flight.__file__),
            Path(model.rich_ball_physics.impact.__file__),
        ]
    )
    records = [file_record(p) for p in sources]
    original_hashes = {r["sha256"] for r in report["inputs"]}
    if any(
        file_record(args.cameras_root / truth[p]["match_id"] / "camera_P_per_frame_v1.npz")[
            "sha256"
        ]
        not in original_hashes
        for p in report["point_inventory"]
    ):
        raise ValueError("camera content absent from frozen report inputs")
    rows = []
    keys = set()
    for row in report["results"]:
        key = (row["point"], row["arm"])
        if key in keys:
            raise ValueError("duplicate point/arm")
        keys.add(key)
        result = {
            "point": row["point"],
            "arm": row["arm"],
            "status": row["status"],
            "oracle_physical_diagnostic_met": False,
        }
        if row["status"] == "measured":
            point = truth[row["point"]]
            scene, _, _, paths, heldout = oracle_benchmark.prepare(
                point, args.cameras_root / point["match_id"] / "camera_P_per_frame_v1.npz"
            )
            scene = oracle_benchmark.configured_scene(scene, report["configuration"])
            heldout = oracle_benchmark.configured_scene(heldout, report["configuration"])
            full_native = tuple(
                np.unique(np.r_[a, b])
                for a, b in zip(scene.observation_frames, heldout.observation_frames, strict=True)
            )
            scene, _, _ = oracle_benchmark.apply_training_visibility(
                scene, point, report["configuration"].get("training_visibility", "all")
            )
            parameters = np.asarray(row["parameters"])
            if args.input_compatibility:
                supplied, _ = oracle_benchmark.conditioning_bounces(
                    point, scene, report["configuration"].get("bounce_time_conditioning", "exact")
                )
                result["input_physical_compatibility"] = physical_compatibility.evaluate(
                    scene,
                    parameters,
                    oracle_benchmark.in_scope_bounce_evidence(supplied, scene)[0],
                    point["termination_kind"],
                    full_native,
                    uncertainty_frames=report["configuration"].get(
                        "bounce_uncertainty_frames", 1.0
                    ),
                    ending_uncertainty_frames=args.ground_completion_frames,
                )
            replay = oracle_benchmark.measure(
                scene,
                {**row, "parameters": parameters, "final_pixel_rms": row["training_pixel_rms"]},
                paths,
                heldout,
            )
            for a, b in zip(replay["flights"], row["flights"], strict=True):
                if not np.isclose(
                    a["trajectory_rms_m"], b["trajectory_rms_m"], atol=1e-8, rtol=1e-8
                ):
                    raise ValueError("frozen trajectory numerical replay drift")
            result.update(
                audit_point(
                    point, scene, parameters, paths, row["trajectory_and_endpoint_tolerances_met"]
                )
            )
            if args.ground_completion_frames is not None:
                result["baseline_oracle_physical_diagnostic_met"] = result[
                    "oracle_physical_diagnostic_met"
                ]
                completion = terminal_completion.complete(
                    scene,
                    parameters,
                    point["termination_kind"],
                    uncertainty_frames=args.ground_completion_frames,
                    last_observation_frame=max(
                        scene.observation_frames[-1][-1], heldout.observation_frames[-1][-1]
                    ),
                )
                result["terminal_completion"] = completion
                if completion["status"] == "completed":
                    result.update(
                        audit_point(
                            point,
                            scene,
                            parameters,
                            paths,
                            row["trajectory_and_endpoint_tolerances_met"],
                            fitted_end_frame=completion["end_frame"],
                        )
                    )
                    error = float(
                        np.linalg.norm(
                            np.asarray(completion["end_xyz"]) - paths[-1]["positions"][-1]
                        )
                    )
                    result["completed_endpoint_error_m"] = error
                    result["conditions"]["completed_endpoint_position"] = (
                        error <= oracle_benchmark.CONTRACT["termination_tolerance_m"]
                    )
                result["conditions"]["bounded_ground_completion"] = (
                    completion["status"] == "completed"
                )
                result["oracle_physical_diagnostic_met"] = all(result["conditions"].values())
                result["failures"] = [k for k, v in result["conditions"].items() if not v]
        rows.append(result)
    expected = {(p, a) for p in report["point_inventory"] for a in report["configuration"]["arms"]}
    if keys != expected:
        raise ValueError("incomplete frozen point/arm inventory")
    if records != [file_record(p) for p in sources]:
        raise ValueError("inputs changed during physical audit")
    from collections import Counter

    summary = {
        a: {
            "points": sum(r["arm"] == a for r in rows),
            "oracle_physical_diagnostic_met": sum(
                r["oracle_physical_diagnostic_met"] for r in rows if r["arm"] == a
            ),
            **(
                {
                    "baseline_oracle_physical_diagnostic_met": sum(
                        r.get("baseline_oracle_physical_diagnostic_met", False)
                        for r in rows
                        if r["arm"] == a
                    ),
                    "completion_status_counts": dict(
                        Counter(
                            r.get("terminal_completion", {}).get(
                                "reason",
                                "completed"
                                if r.get("terminal_completion", {}).get("status") == "completed"
                                else "not_measured",
                            )
                            for r in rows
                            if r["arm"] == a
                        )
                    ),
                }
                if args.ground_completion_frames is not None
                else {}
            ),
            "overlapping_failures": dict(
                Counter(
                    f for r in rows if r["arm"] == a for f in r.get("failures", ["not_measured"])
                )
            ),
        }
        for a in report["configuration"]["arms"]
    }
    payload = {
        "schema": "s6_backward_oracle_ground_completion_audit_v3"
        if args.ground_completion_frames is not None
        else "s6_backward_oracle_physical_audit_v3",
        "impact_energy_contract": "returned outgoing flights require finite velocity/spin and total rigid-body kinetic energy ratio <= 1.000001; terminal unused outgoing states are not scored",
        "ground_completion_uncertainty_frames": args.ground_completion_frames,
        "spatial_scoring": "unchanged original dense truth horizon; completed endpoint scored additionally; physical screens use the returned path domain",
        "code": git_record(Path(__file__).resolve().parents[3]),
        "inputs": records,
        "input_bindings": [
            {"resolved_path": str(p.resolve()), "record": r}
            for p, r in zip(sources, records, strict=True)
        ],
        "summaries": summary,
        "results": rows,
        "scope": "synthetic oracle diagnostics only; impossible targets and observation exits remain in denominator",
    }
    if args.input_compatibility:
        payload["input_compatibility_summaries"] = {
            arm: {
                "conditions": sum(r["arm"] == arm for r in rows),
                "compatible": sum(
                    r["arm"] == arm
                    and r.get("input_physical_compatibility", {}).get("compatible", False)
                    for r in rows
                ),
                "compatible_and_oracle_correct": sum(
                    r["arm"] == arm
                    and r.get("input_physical_compatibility", {}).get("compatible", False)
                    and r["oracle_physical_diagnostic_met"]
                    for r in rows
                ),
                "compatible_but_oracle_wrong": sum(
                    r["arm"] == arm
                    and r.get("input_physical_compatibility", {}).get("compatible", False)
                    and not r["oracle_physical_diagnostic_met"]
                    for r in rows
                ),
                "oracle_correct_but_incompatible": sum(
                    r["arm"] == arm
                    and not r.get("input_physical_compatibility", {}).get("compatible", False)
                    and r["oracle_physical_diagnostic_met"]
                    for r in rows
                ),
            }
            for arm in report["configuration"]["arms"]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.input_compatibility:
        args.output.with_suffix(".html").write_text(
            "<!doctype html><title>S6 physical compatibility</title><h1>Compatibility versus oracle accuracy</h1>"
            "<p>Supplied-input research only. Compatibility is not correct-point acceptance. "
            "Camera/event accuracy, ball identity, contact reach and real-video yield remain unvalidated.</p><pre>"
            + html.escape(json.dumps(payload["input_compatibility_summaries"], indent=2))
            + "</pre><details><summary>Every point and disposition</summary><pre>"
            + html.escape(
                json.dumps(
                    [
                        dict(
                            point=r["point"],
                            arm=r["arm"],
                            compatible=r.get("input_physical_compatibility", {}).get(
                                "compatible", False
                            ),
                            oracle_correct=r["oracle_physical_diagnostic_met"],
                            failures=r.get("input_physical_compatibility", {}).get(
                                "failures", ["not_measured"]
                            ),
                        )
                        for r in rows
                    ],
                    indent=2,
                )
            )
            + '</pre></details><p><a href="'
            + html.escape(args.output.name, quote=True)
            + '">Bound report</a></p>'
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
