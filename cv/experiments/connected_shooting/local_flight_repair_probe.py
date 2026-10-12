"""Can a fixed-contact local refit recover one missing interior flight?

Non-default opened-development probe. Reuse an explicitly named per-flight
reference, freeze every shared contact and nonlocal parameter, and optimize
only one launch velocity/spin block. Certify its endpoint, export through the
single-shooting representation, and remeasure every flight at unchanged gates.
The reference remains selected unless accepted-flight membership improves
without losing any accepted flight. This is not an automatic inference input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    athlete_priors,
    event_constraints,
    model,
    per_flight_acceptance as acceptance,
    per_flight_pictures,
    per_flight_rescore as rescore,
    real_exposure_replay as exposure,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance

ENDPOINT_LIMIT_M = 0.001


def local_indices(scene: model.Scene, flight_index: int) -> np.ndarray:
    """The six launch parameters; contact and rebound variables are excluded."""
    count = len(scene.contact_frames) - 1
    if not 0 < flight_index < count - 1:
        raise ValueError("probe requires an interior flight, excluding serve and ending")
    blocks = model.shared_parameter_slices(scene)
    return np.r_[
        np.arange(
            blocks["velocities"].start + 3 * flight_index,
            blocks["velocities"].start + 3 * flight_index + 3,
        ),
        np.arange(
            blocks["spins"].start + 3 * flight_index, blocks["spins"].start + 3 * flight_index + 3
        ),
    ]


def improvement_allowed(before: dict, after: dict, endpoint_gap_m: float) -> bool:
    """Gate candidate replacement using full exported-flight membership."""
    if not np.isfinite(endpoint_gap_m) or endpoint_gap_m > ENDPOINT_LIMIT_M:
        return False
    old, new = before["flights"], after["flights"]
    if len(old) != len(new):
        return False
    return bool(
        after.get("structural_connections_valid", False)
        and all(not a["accepted"] or b["accepted"] for a, b in zip(old, new, strict=True))
        and sum(row["accepted"] for row in new) > sum(row["accepted"] for row in old)
        and all(row["checks"].get("continuity_into_neighbours", False) for row in new)
    )


def optimize_local(
    context: dict,
    parameters: np.ndarray,
    flight_index: int,
    *,
    maxiter: int,
    bounce_bracket_frames: float,
    duration: float,
) -> list[dict]:
    """Two bounded SLSQP starts with exact endpoint equality constraints."""
    scene = replace(context["scene"], parameterization="shared_contact_states")
    initial = model.shared_contact_seed(scene, parameters)
    indices = local_indices(scene, flight_index)
    scale = np.r_[[30.0] * 3, [3.0] * 3]
    lower, upper = np.r_[[-75.0] * 3, [-6.0] * 3], np.r_[[75.0] * 3, [6.0] * 3]
    if np.any(initial[indices] < lower) or np.any(initial[indices] > upper):
        raise ValueError("reference local parameters are outside the unchanged solver bounds")
    target = np.concatenate(scene.pixels)
    image_rows = 2 * len(target)
    loss = event_constraints.mixed_loss(image_rows)
    cache = FlightCache(entries_per_flight=24)
    anchor = exposure.anchor_plan(context["targets"], [None] * len(scene.pixels))

    def expand(q):
        full = initial.copy()
        full[indices] = np.asarray(q) * scale
        return full

    def evaluate(q):
        full = expand(q)
        flights, physical, _ = event_constraints.evaluate(
            scene, full, context["bounces"], bounce_bracket_frames, simulation_cache=cache
        )
        return full, flights, physical

    def objective(q):
        full, flights, physical = evaluate(q)
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
        for j, (xy, sigma) in enumerate(anchor["bounce"][flight_index]):
            observed = flights[flight_index]["bounces"]
            anchors.extend(
                (np.asarray(observed[j]["x"])[:2] - xy) / sigma if j < len(observed) else [0.0, 0.0]
            )
        residual = np.r_[image.ravel(), physical, anchors]
        return float(2 * np.sum(loss((residual / 2) ** 2)[0]))

    def endpoint(q):
        _, flights, _ = evaluate(q)
        return model.shared_contact_gaps(flights)[flight_index]

    q0 = initial[indices] / scale
    flat = q0.copy()
    flat[3:] = 0.0
    trials = []
    for name, start in (("reference", q0), ("flat_spin", flat)):
        began = time.monotonic()
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=list(zip(lower / scale, upper / scale)),
            constraints=[{"type": "eq", "fun": endpoint}],
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        full = expand(result.x)
        frozen = np.ones(len(full), bool)
        frozen[indices] = False
        if not np.array_equal(full[frozen], initial[frozen]):
            raise AssertionError("local optimization mutated a frozen parameter")
        gaps = model.shared_contact_gaps(model.chain(scene, full))
        trials.append(
            {
                "start": name,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "cost": float(result.fun),
                "initial_cost": objective(q0),
                "maximum_endpoint_gap_m": float(np.linalg.norm(gaps, axis=1).max()),
                "endpoint_residuals_m": gaps.tolist(),
                "frozen_parameters_identical": True,
                "shared_parameters": full.tolist(),
                "parameters": model.shared_to_single_parameters(scene, full).tolist(),
                "wall_seconds": time.monotonic() - began,
            }
        )
    return trials


def _resolve(record: dict) -> Path:
    return (paths.REPO_ROOT if record["path_base"] == "repository" else paths.data_root()) / record[
        "path"
    ]


def run_case(job: dict, *, optimizer=None) -> dict:
    began = time.monotonic()
    output = Path(job["output"]) / f"{job['key']}_f{job['flight_index']:02d}"
    output.mkdir(parents=True, exist_ok=False)

    def timeout(*_args):
        raise TimeoutError("bounded local-flight repair deadline")

    prior_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(job["timeout"])
    try:
        aggregate = json.loads(Path(job["baseline_report"]).read_text())
        item = next(row for row in aggregate["attempts"] if row["key"] == job["key"])
        perflight = json.loads(Path(item["document"]).read_text())
        saved = next(row for row in perflight["rungs"] if row["rung"] == job["rung"])
        search_path = _resolve(perflight["search_report"])
        search = json.loads(search_path.read_text())
        configuration_path = next(
            _resolve(record)
            for record in search["inputs"]
            if record["path"].endswith("/fixed_configuration.json")
        )
        configuration = json.loads(configuration_path.read_text())
        candidates = [
            row
            for row in search["refined_candidates"]
            if row["depth_hypothesis_m"] == saved["selected_depth_m"]
        ]
        if len(candidates) != 1:
            raise ValueError("saved selected depth does not uniquely identify a candidate")
        reference = candidates[0]
        source = search_path.parent.parent
        labels_path = (
            paths.REPO_ROOT
            / "cv/validation/labels/s6_agent_inputs_v1"
            / item["metadata"]["label_file"]
        )
        args = SimpleNamespace(
            labels=labels_path,
            packet=source / "inputs/packet.json",
            cameras=source / "inputs/cameras.json",
            pose_csv=paths.data_root() / "processed" / item["metadata"]["player_localization"],
            pose_image_scale=item["metadata"]["pose_image_scale"],
            athlete_prior_mode="stature_pose_soft",
            player_order=[name for name, _ in item["metadata"]["player_statures_m"]],
            observation_fallback="on",
            dense_labels=None,
            player_ledger=None,
            witness_surface=item["metadata"]["surface"],
        )
        context = rescore.build_context(args, search)
        receipt = rescore.reproduction_receipt(context, reference, args.athlete_prior_mode)
        if not receipt["identical"]:
            raise ValueError("frozen scene reproduction failed")
        thresholds = next(
            row["thresholds"] for row in rescore.ladder() if row["name"] == job["rung"]
        )
        duration = float(search["configuration"]["exposure_duration_frames"])

        def score(parameters):
            measurement = exposure.measure(
                context["scene"],
                context["heldout"],
                context["bounces"],
                context["native"],
                parameters,
                context["axes"],
                duration,
                [],
                termination_kind=context["termination_kind"],
            )
            athlete = athlete_priors.evaluate(
                np.asarray(measurement["contact_xyz"]), context["players"]
            )
            measured = acceptance.measure(
                context["scene"],
                context["heldout"],
                parameters,
                context["bounces"],
                context["native"],
                context["axes"],
                context["targets"],
                context["players"],
                context["events"],
                termination_kind=context["termination_kind"],
                duration=duration,
                athlete=athlete,
                ending_passive_context_frames=thresholds.ending_passive_context_frames,
                measurement=measurement,
            )
            measured["surface"] = context["witness_surface"]
            verdict = acceptance.score(
                measured,
                thresholds,
                depth_bounds=context["depth_bounds"],
                athlete_prior_mode=args.athlete_prior_mode,
                ending_kind=item["metadata"]["ending_kind"],
            )
            return verdict, measurement

        parameters = np.asarray(reference["measurement"]["fit"]["parameters"], float)
        before, before_measurement = score(parameters)
        if [f["accepted"] for f in before["flights"]] != [
            f["accepted"] for f in saved["verdict"]["flights"]
        ]:
            raise ValueError("fresh baseline measurement does not reproduce adopted membership")
        trials = (optimize_local if optimizer is None else optimizer)(
            {**context, "repair_thresholds": thresholds, "repair_reference_verdict": before},
            parameters,
            job["flight_index"],
            maxiter=job["maxiter"],
            duration=duration,
            bounce_bracket_frames=float(
                configuration["solver_experimental_arm"]["bounce_bracket_frames"]
            ),
        )
        for trial in trials:
            trial["verdict"], trial["measurement"] = score(np.asarray(trial["parameters"]))
            trial["eligible_improvement"] = improvement_allowed(
                before, trial["verdict"], trial["maximum_endpoint_gap_m"]
            )
        improving = [row for row in trials if row["eligible_improvement"]]
        chosen = min(
            improving,
            key=lambda row: (-row["verdict"]["accepted_flight_count"], row["cost"]),
            default=None,
        )
        labels = json.loads(labels_path.read_text())
        index = job["flight_index"]
        start, end = context["scene"].contact_frames[index : index + 2]
        native_rows = [
            row for row in before_measurement["native_projection"] if start <= row["frame"] <= end
        ]
        focus_epochs = [
            start,
            end,
            *(
                epoch + offset
                for epoch in context["bounces"][index]
                for offset in (-2, -1, 0, 1, 2)
            ),
        ]
        focus_frames = {
            min(native_rows, key=lambda row: abs(row["frame"] - epoch))["frame"]
            for epoch in focus_epochs
        }
        focus_frames.add(max(native_rows, key=lambda row: row["error_px"])["frame"])
        for name, verdict, measurement in [
            ("before", before, before_measurement),
            *((f"trial_{row['start']}", row["verdict"], row["measurement"]) for row in trials),
        ]:
            plotted = deepcopy(search)
            plotted["refined_candidates"] = [
                {
                    **reference,
                    "measurement": {
                        **measurement,
                        "native_projection": [
                            row
                            for row in measurement["native_projection"]
                            if row["frame"] in focus_frames
                        ],
                    },
                }
            ]
            pf = {
                "rungs": [
                    {
                        "rung": job["rung"],
                        "selected_depth_m": saved["selected_depth_m"],
                        "verdict": verdict,
                    }
                ]
            }
            per_flight_pictures.flight_pictures(
                plotted,
                pf,
                labels,
                job["rung"],
                job["flight_index"],
                output / name,
                maximum_pictures=8,
            )
        for trial in trials:
            trial.pop("measurement")
        result = {
            "status": "measured",
            "job": job,
            "before": before,
            "trials": trials,
            "selected": "reference" if chosen is None else chosen["start"],
            "selected_source": "original_reference" if chosen is None else "repair_trial",
            "selected_trial_start": None if chosen is None else chosen["start"],
            "selected_verdict": before if chosen is None else chosen["verdict"],
            "reference_retained": True,
            "reproduction": receipt,
            "inputs": [
                provenance.file_record(p)
                for p in (
                    Path(job["baseline_report"]),
                    search_path,
                    configuration_path,
                    labels_path,
                    args.packet,
                    args.cameras,
                )
            ],
        }
    except (ValueError, KeyError, TimeoutError, FloatingPointError) as error:
        result = {"status": "held", "job": job, "blocker": f"{type(error).__name__}: {error}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_alarm)
    result.update(
        wall_seconds=time.monotonic() - began,
        human_derived=True,
        automatic_inference_eligible=False,
    )
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument(
        "--case", action="append", required=True, help="attempt_key:zero_based_interior_flight"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.maxiter < 1 or args.timeout < 1:
        parser.error("positive iteration and time budgets required")
    args.output.mkdir(parents=True, exist_ok=False)
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
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, jobs))
    report = {
        "schema": "fixed_contact_local_flight_repair_probe_v1",
        "scope": __doc__,
        "jobs": jobs,
        "results": results,
        "code": provenance.git_record(paths.REPO_ROOT),
        "source": provenance.file_record(Path(__file__)),
        "human_derived": True,
        "automatic_inference_eligible": False,
        "promoted": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    "key": r["job"]["key"],
                    "flight": r["job"]["flight_index"],
                    "status": r["status"],
                    "selected": r.get("selected"),
                    "blocker": r.get("blocker"),
                }
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
