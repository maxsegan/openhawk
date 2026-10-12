"""Owner-aligned spatial diagnostics alongside the historical synthetic benchmark.

Question: how many complete known-truth points meet foot-scale error tolerances
AND the production connection/ending contracts? This is development evaluation,
not raw-video yield or independent validation of real airborne depth.

Run with --truth, --report, --cameras-root and --output; --cohort optionally fixes
a subset before scoring. Missing reconstructions remain in that denominator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from cv.pipeline.artifact_cache import stage_identity
from cv.pipeline.trajectory_contract import (
    terminal_ground_endpoint_report,
    trajectory_connection_report,
)
from cv.pipeline.flight_ledger import classify_flight
from cv.validation import s6_point_bench as bench

CONTRACT = {
    "schema": "owner_spatial_working_v1",
    "bounce_tolerance_m": 0.3048,
    "contact_tolerance_m": 0.9144,
    "trajectory_rms_tolerance_m": 0.3048,
    "trajectory_peak_ceiling_m": 0.9144,
    "termination_tolerance_m": 0.3048,
    "note": (
        "Working research interpretation of the owner's guidance, not line-call precision. "
        "Three-foot contact/peak limits are permissive ceilings, not typical-error targets. "
        "Missing required witnesses fail. Spatial tolerance never permits a discontinuity."
    ),
}


def _within(value, limit: float) -> bool:
    try:
        return math.isfinite(float(value)) and 0.0 <= float(value) <= limit
    except (ValueError, TypeError, OverflowError):
        return False


def flight_checks(row: dict) -> dict[str, bool]:
    """Require measured witnesses; an absent value is not an automatic pass."""
    terminal = row.get("terminal") is True
    ending_error = (
        row.get("exit_error_m")
        if row.get("termination_kind") == "out_of_view"
        else row.get("termination_error_m")
    )
    return {
        "solved": row.get("solved") is True,
        "bounce": row.get("scored_bounces") == 0
        or _within(row.get("bounce_error_m"), CONTRACT["bounce_tolerance_m"]),
        "start_contact": _within(row.get("start_contact_error_m"), CONTRACT["contact_tolerance_m"]),
        "end_contact": terminal
        or _within(row.get("end_contact_error_m"), CONTRACT["contact_tolerance_m"]),
        "trajectory_rms": _within(
            row.get("trajectory_rms_m"), CONTRACT["trajectory_rms_tolerance_m"]
        ),
        "trajectory_peak": _within(
            row.get("trajectory_max_m"), CONTRACT["trajectory_peak_ceiling_m"]
        ),
        "trajectory_coverage": _within(row.get("trajectory_uncovered_frames"), 0.0)
        and _within(row.get("trajectory_frames"), 1e9)
        and float(row["trajectory_frames"]) >= 2,
        "termination": not terminal or _within(ending_error, CONTRACT["termination_tolerance_m"]),
        "no_early_stop": row.get("mid_air_stop") is False,
    }


def summarize(truth: dict, report: dict, historical: dict) -> dict:
    """Keep every selected truth point/flight, including absent or held output."""
    records = {row["point"]: row for row in report["points_detail"]}
    measurements = {(row["point"], row["flight_index"]): row for row in historical["flight_rows"]}
    if len(records) != len(report["points_detail"]) or len(measurements) != len(
        historical["flight_rows"]
    ):
        raise ValueError("duplicate reconstruction points or scored flights")
    if len({row["point"] for row in truth["points"]}) != len(truth["points"]):
        raise ValueError("duplicate truth points")
    points, flights = [], []
    for expected in truth["points"]:
        key = expected["point"]
        record = records.get(key, {})
        fits_by_index = {int(fit["flight_index"]): fit for fit in record.get("fits", [])}
        attempts_by_index = {
            int(attempt["flight_index"]): attempt for attempt in record.get("flight_attempts", [])
        }
        if len(fits_by_index) != len(record.get("fits", [])) or len(attempts_by_index) != len(
            record.get("flight_attempts", [])
        ):
            raise ValueError("duplicate fit or attempt flight index")
        local = []
        for entry in expected["flights"]:
            index = int(entry["flight_index"])
            measured = measurements.get((key, index), {})
            checks = flight_checks(measured)
            disposition, gate_reasons = (
                classify_flight(record, attempts_by_index[index], fits_by_index.get(index))
                if index in attempts_by_index
                else ("unknown", ["recorded_flight_attempt_missing"])
            )
            local.append(
                {
                    "point": key,
                    "flight_index": index,
                    "flight_gate_accepted": disposition == "provisional_valid",
                    "flight_gate_disposition": disposition,
                    "flight_gate_reasons": gate_reasons,
                    "historical_held_out_accepted": measured.get("accepted") is True,
                    "spatially_within_tolerance": all(checks.values()),
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                }
            )
        flights.extend(local)
        fits = record.get("fits", [])
        connections = trajectory_connection_report(fits)
        ending_geometry = terminal_ground_endpoint_report(
            (record.get("terminal_coverage") or {}).get("termination_kind"), fits
        )
        spatial = bool(local) and all(row["spatially_within_tolerance"] for row in local)
        exact = (expected.get("trajectory_truth") or {}).get(
            "schema"
        ) == "simulation_trajectory_samples_v1"
        topology = len(fits) == len(expected["flights"]) and sorted(
            row.get("flight_index", -1) for row in fits
        ) == sorted(int(row["flight_index"]) for row in expected["flights"])
        conditions = {
            "recorded_simulation_truth": exact,
            "all_flights_spatially_within_tolerance": spatial,
            "one_fit_per_expected_flight": topology,
            "continuous_connections": connections["valid"],
            "physical_ending_covered": (record.get("terminal_coverage") or {}).get("valid") is True,
            "court_ending_geometry_consistent": ending_geometry["valid"],
            "production_point_retained": record.get("decision") == "retain",
            "constituent_flights_accepted": bool(local)
            and all(row["flight_gate_accepted"] for row in local),
        }
        points.append(
            {
                "point": key,
                "source_match": expected.get("source_match", key.split("__", 1)[0]),
                "flights": len(local),
                "qualified": all(conditions.values()),
                "spatially_within_tolerance": spatial,
                "conditions": conditions,
                "failed_conditions": [name for name, passed in conditions.items() if not passed],
                "connections": connections,
                "ending_geometry": ending_geometry,
            }
        )
    slices = []
    for source in sorted({row["source_match"] for row in points}):
        members = [row for row in points if row["source_match"] == source]
        slices.append(
            {
                "source_match": source,
                "points": len(members),
                "qualified": sum(row["qualified"] for row in members),
            }
        )
    return {
        "schema": "owner_synthetic_spatial_audit_v3",
        "flight_acceptance_definition": "current classify_flight on recorded attempt and fit; historical held-out pixel acceptance reported separately",
        "contract": CONTRACT,
        "evaluation_status": "synthetic_development_not_full_match_yield",
        "points": len(points),
        "flights": len(flights),
        "qualified_points": sum(row["qualified"] for row in points),
        "qualified_fraction": sum(row["qualified"] for row in points) / len(points)
        if points
        else None,
        "spatially_within_tolerance_points": sum(
            row["spatially_within_tolerance"] for row in points
        ),
        "spatially_within_tolerance_flights": sum(
            row["spatially_within_tolerance"] for row in flights
        ),
        "flight_gate_accepted": sum(row["flight_gate_accepted"] for row in flights),
        "historical_held_out_accepted": sum(row["historical_held_out_accepted"] for row in flights),
        "historical_held_out_accepted_spatially_wrong": sum(
            row["historical_held_out_accepted"] and not row["spatially_within_tolerance"]
            for row in flights
        ),
        "flight_gate_accepted_spatially_wrong": sum(
            row["flight_gate_accepted"] and not row["spatially_within_tolerance"] for row in flights
        ),
        "failed_condition_counts": dict(
            Counter(name for row in points for name in row["failed_conditions"])
        ),
        "historical_criteria": historical["by_criterion"],
        "by_source_match": slices,
        "point_rows": points,
        "flight_rows": flights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cameras-root", type=Path, required=True)
    parser.add_argument("--anchors-root", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.truth, args.report, args.cameras_root]
    inputs.extend(path for path in (args.anchors_root, args.cohort) if path is not None)
    identity_args = {
        "stage": "owner_synthetic_spatial_audit",
        "command": [sys.executable, "-m", "cv.validation.owner_spatial_audit"],
        "inputs": inputs,
        "configuration": CONTRACT,
    }
    before = stage_identity(**identity_args)
    truth = json.loads(args.truth.read_text())
    report = json.loads(args.report.read_text())
    if args.cohort:
        keys = [row["point"] for row in json.loads(args.cohort.read_text())["points"]]
        available = {row["point"] for row in truth["points"]}
        if len(keys) != len(set(keys)) or set(keys) - available:
            raise ValueError("cohort has duplicated or unavailable truth points")
        truth["points"] = [row for row in truth["points"] if row["point"] in set(keys)]
    historical = bench.audit_report(truth, report, args.cameras_root, args.anchors_root)
    result = summarize(truth, report, historical)
    after = stage_identity(**identity_args)
    if before != after:
        raise RuntimeError("scoring inputs/code changed during evaluation")
    result["identity"] = before
    if args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output cannot replace an evaluation input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"point_rows", "flight_rows", "identity"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
