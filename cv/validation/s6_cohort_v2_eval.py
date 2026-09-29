"""Measure automatic Stage-6 reconstruction on the regenerated cohort-v2 substrate.

The module builds a read-only symlink mirror, places the fixed event-model-v3
emissions at the same root-level path used by the broadcast runner, executes the
production reconstruction CLI, and reports label-blind and offline-truth metrics
separately.  Owner truth is never read by ``build`` or ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cv.pipeline.flight_ledger import build as build_flight_ledger
from cv.pipeline.flight_ledger import net_clearance_m, write_csv
from cv.validation.oracle_3d_ceiling import SEED_20260902_POINTS
from cv.validation.s6root_metric_acceptance import run as run_metric_acceptance
from cv.validation.wk3_s6_cohort import residual_diagnostics

POINTS = 198
MATCHES = 46
PHYSICAL_EVENTS = {"contact", "bounce", "net_hit"}
RUNNER_EVENT_NAME = "event_emissions.json"
MIRROR_MANIFEST_NAME = "s6_cohort_v2_mirror_manifest.json"
HELD_OUT_BROADCASTS = {
    "ao2019f_w_osaka_kvitova",
    "atpf2024rr_m_alcaraz_zverev",
    "rg2025f_m_alcaraz_sinner",
    "rome2026f_w_svitolina_gauff",
    "uso2025r128_m_khachanov_basavareddy",
    "wim2024f_w_krejcikova_paolini",
    "wim2025f_m_sinner_alcaraz",
    "wtaf2025f_w_rybakina_sabalenka",
}


def _data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def percentile_pair(values: Iterable[float]) -> list[float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return [None, None]
    return [float(np.median(array)), float(np.percentile(array, 90.0))]


def point_match(point: str) -> str:
    return point.rsplit("__", 1)[0]


def build_mirror(
    source_root: Path,
    emissions_path: Path,
    emissions_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Link the immutable cohort and materialize the runner's accepted event view."""
    source_root = source_root.resolve()
    emissions_path = emissions_path.resolve()
    emissions_manifest_path = emissions_manifest_path.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing mirror root: {output_root}")

    cohort = json.loads((source_root / "manifest.json").read_text())
    match_ids = [str(row["id"]) for row in cohort["matches"]]
    point_count = sum(len(row.get("point_ids", [])) for row in cohort["matches"])
    if len(match_ids) != MATCHES or point_count != POINTS:
        raise ValueError(
            f"expected {MATCHES} matches and {POINTS} points, found "
            f"{len(match_ids)} matches and {point_count} points"
        )

    rows = json.loads(emissions_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("fixed emissions must be a JSON list")
    required = {"abstain", "clip", "event_type", "frame", "match_id"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("fixed emissions are not event_model_v3-compatible")
    selected = [
        row
        for row in rows
        if row.get("abstain") is not True
        and row.get("event_type") in PHYSICAL_EVENTS | {"point_end"}
    ]
    if any(row.get("abstain") is True for row in selected):
        raise AssertionError("runner event view retained an abstention")

    output_root.mkdir(parents=True)
    linked = 0
    for source in sorted(source_root.iterdir()):
        destination = output_root / source.name
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        linked += 1

    runner_events = output_root / RUNNER_EVENT_NAME
    write_json(runner_events, selected)
    source_event_manifest = json.loads(emissions_manifest_path.read_text())
    mirror_event_manifest = {
        **source_event_manifest,
        "schema": "s6_cohort_v2_runner_event_view_v1",
        "source_schema": source_event_manifest.get("schema"),
        "source_emissions": os.fspath(emissions_path),
        "source_emissions_sha256": sha256(emissions_path),
        "source_manifest": os.fspath(emissions_manifest_path),
        "source_manifest_sha256": sha256(emissions_manifest_path),
        "selection": "non-abstained contact/bounce/net_hit rows plus point_end",
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "selected_physical_rows": sum(row["event_type"] in PHYSICAL_EVENTS for row in selected),
        "selected_point_end_rows": sum(row["event_type"] == "point_end" for row in selected),
        "selected_abstained_rows": sum(row.get("abstain") is True for row in selected),
        "labels_or_reviewed_inputs": [],
    }
    write_json(runner_events.with_suffix(".manifest.json"), mirror_event_manifest)

    payload = {
        "schema": "s6_cohort_v2_mirror_v1",
        "automatic": True,
        "source_root": os.fspath(source_root),
        "source_manifest_sha256": sha256(source_root / "manifest.json"),
        "output_root": os.fspath(output_root.resolve()),
        "matches": len(match_ids),
        "points": point_count,
        "symlinks": linked,
        "event_view": {
            "path": os.fspath(runner_events.resolve()),
            "sha256": sha256(runner_events),
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "physical_rows": sum(row["event_type"] in PHYSICAL_EVENTS for row in selected),
            "point_end_rows": sum(row["event_type"] == "point_end" for row in selected),
        },
        "human_derived_inference_inputs": [],
    }
    write_json(output_root / MIRROR_MANIFEST_NAME, payload)
    return payload


def load_truth_points(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    return {str(row["clip"]) for row in payload["emissions"]}


def scope_points(report: dict[str, Any], scope: str) -> set[str]:
    points = {str(row["point"]) for row in report["points_detail"]}
    if scope == "all_46":
        return points
    held = {point for point in points if point_match(point) in HELD_OUT_BROADCASTS}
    if scope == "held_out_8":
        return held
    if scope == "other_38":
        return points - held
    raise ValueError(f"unknown scope: {scope}")


def summarize_scope(
    report: dict[str, Any],
    ledger: dict[str, Any],
    truth_points: set[str],
    scope: str,
    *,
    wall_time_seconds: float | None,
) -> dict[str, Any]:
    selected_points = scope_points(report, scope)
    details = [row for row in report["points_detail"] if row["point"] in selected_points]
    fits = [fit for point in details for fit in point.get("fits", [])]
    attempts = [attempt for point in details for attempt in point.get("flight_attempts", [])]
    ledger_rows = [row for row in ledger["rows"] if row["point"] in selected_points]
    statuses = Counter(row["status"] for row in ledger_rows)
    accepted_points = {
        str(point["point"])
        for point in details
        if point.get("complete_point_gate", {}).get("accepted") is True
    }
    gaps = [
        float(value)
        for point in details
        for value in point.get("junction_gaps_m", [])
        if math.isfinite(float(value))
    ]
    clearances = [value for fit in fits if (value := net_clearance_m(fit)) is not None]
    held_medians = [
        float(fit["held_out_reprojection_median_px"])
        for fit in fits
        if fit.get("held_out_reprojection_median_px") is not None
    ]
    held_p90s = [
        float(fit["held_out_reprojection_p90_px"])
        for fit in fits
        if fit.get("held_out_reprojection_p90_px") is not None
    ]
    selected_truth = truth_points & selected_points
    return {
        "scope": scope,
        "broadcasts": len({point_match(point) for point in selected_points}),
        "points": len(selected_points),
        "truth_points": len(selected_truth),
        "attempted_flights": len(attempts),
        "solved_flights": len(fits),
        "accepted_flights": statuses["provisional_valid"],
        "accepted_complete_points": len(accepted_points),
        "accepted_complete_points_denominator": len(selected_points),
        "accepted_complete_truth_points": len(accepted_points & selected_truth),
        "accepted_complete_truth_points_denominator": len(selected_truth),
        "held_out_flight_median_px_corpus_median_p90": percentile_pair(held_medians),
        "held_out_flight_p90_px_corpus_median_p90": percentile_pair(held_p90s),
        "junction_gap_m_median_p90": percentile_pair(gaps),
        "junctions": len(gaps),
        "net_crossings": len(clearances),
        "net_violations": sum(float(value) < -0.01 for value in clearances),
        "point_timeouts": sum("point_timeout" in point.get("reasons", []) for point in details),
        "wall_time_seconds": wall_time_seconds if scope == "all_46" else None,
        "ledger_statuses": dict(statuses),
    }


def run_reconstruction(
    mirror_root: Path,
    output_root: Path,
    truth_events: Path,
    *,
    workers: int,
    downweight_contact_adjacent: bool = False,
    shared_contact_fit: bool = False,
) -> dict[str, Any]:
    """Run one production-equivalent arm and write split summaries."""
    if workers != 8:
        raise ValueError("this package is measured with --workers 8")
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing output root: {output_root}")
    output_root.mkdir(parents=True)
    report_path = output_root / "report.json"
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.reconstruct_3d",
        "--audit-root",
        os.fspath(mirror_root),
        "--manifest",
        os.fspath(mirror_root / "manifest.json"),
        "--event-boundaries",
        os.fspath(mirror_root / RUNNER_EVENT_NAME),
        # The fixed decoder already defines its best path but records that the
        # legacy point_grammar fields are absent.  This compatibility switch
        # consumes that path; the mirror has already removed every abstention.
        "--include-dead-time-emissions",
        "--output",
        os.fspath(report_path),
        "--max-nfev",
        "20",
        "--workers",
        str(workers),
        "--math-threads",
        "1",
        "--point-timeout-seconds",
        "600",
        "--terminal-flights",
        "--anchor-bounce-geometry",
        "--whole-point-branches",
        "--branch-width",
        "3",
        "--branch-margin",
        "8",
        "--anchors-output-root",
        os.fspath(output_root / "anchors"),
    ]
    if downweight_contact_adjacent:
        command.append("--downweight-contact-adjacent")
    if shared_contact_fit:
        command.append("--shared-contact-fit")
    else:
        command.append("--independent-contacts")

    started = time.perf_counter()
    with (output_root / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)

    ledger = build_flight_ledger([report_path])
    write_json(output_root / "ledger.json", ledger)
    write_csv(output_root / "ledger.csv", ledger["rows"])
    report = json.loads(report_path.read_text())
    truth_points = load_truth_points(truth_events)
    scopes = {
        scope: summarize_scope(
            report,
            ledger,
            truth_points,
            scope,
            wall_time_seconds=elapsed,
        )
        for scope in ("all_46", "held_out_8", "other_38")
    }
    residuals = residual_diagnostics(report, ledger)
    seed_rows = []
    for row in residuals:
        if row["point"] not in SEED_20260902_POINTS:
            continue
        for worst in row["worst_frames"]:
            worst["distance_to_flight_boundary_frames"] = min(
                abs(float(worst["frame"]) - float(row[boundary]))
                for boundary in ("start_frame", "end_frame")
                if row.get(boundary) is not None
            )
        seed_rows.append(row)
    seed_payload = {
        "schema": "s6_cohort_v2_seed20_worst_frames_v1",
        "seed": 20260902,
        "points": sorted(SEED_20260902_POINTS),
        "solved_flights": len(seed_rows),
        "flights_with_worst_frame_within_1_of_boundary": sum(
            bool(row["worst_frames"])
            and float(row["worst_frames"][0]["distance_to_flight_boundary_frames"]) <= 1.0
            for row in seed_rows
        ),
        "rows": seed_rows,
    }
    write_json(output_root / "seed20_worst_frames.json", seed_payload)
    payload = {
        "schema": "s6_cohort_v2_reconstruction_run_v1",
        "automatic": True,
        "human_derived_inference_inputs": [],
        "mirror_manifest": os.fspath((mirror_root / MIRROR_MANIFEST_NAME).resolve()),
        "command": command,
        "configuration": {
            "workers": workers,
            "math_threads_per_worker": 1,
            "point_timeout_seconds": 600.0,
            "max_nfev": 20,
            "fitter": "default_anchor_first_camera_space",
            "arc_composition": True,
            "camera_lineage_rule": "effective_registered_camera_rule",
            "downweight_contact_adjacent": downweight_contact_adjacent,
            "shared_contact_fit": shared_contact_fit,
            "physics_profile": "pipeline_default",
            "event_consumer_mode": "decoder_best_path_non_abstained",
        },
        "wall_time_seconds": elapsed,
        "scopes": scopes,
    }
    write_json(output_root / "metrics.json", payload)
    write_json(output_root / "run_manifest.json", payload)
    return payload


def summarize_existing(
    report_path: Path,
    ledger_path: Path,
    truth_events: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    truth_points = load_truth_points(truth_events)
    payload = {
        "schema": "s6_cohort_v2_existing_run_summary_v1",
        "report": os.fspath(report_path.resolve()),
        "scopes": {
            scope: summarize_scope(
                report,
                ledger,
                truth_points,
                scope,
                wall_time_seconds=None,
            )
            for scope in ("all_46", "held_out_8", "other_38")
        },
    }
    write_json(output_path, payload)
    return payload


def summarize_metric_scopes(
    report_path: Path,
    metric_path: Path,
    truth_events: Path,
    output_path: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    metric = json.loads(metric_path.read_text())
    truth_points = load_truth_points(truth_events)
    rows = metric["rows"]
    summaries = {}
    for scope in ("all_46", "held_out_8", "other_38"):
        selected_points = scope_points(report, scope)
        selected_truth = selected_points & truth_points
        selected_rows = [row for row in rows if row["point"] in selected_points]
        bounce_errors = [
            float(row["bounce_error_max_m"])
            for row in selected_rows
            if row.get("bounce_error_max_m") is not None
        ]
        metric_by_point: dict[str, list[dict[str, Any]]] = {}
        for row in selected_rows:
            metric_by_point.setdefault(str(row["point"]), []).append(row)
        attempts_by_point = {
            str(point["point"]): point.get("flight_attempts", [])
            for point in report["points_detail"]
            if point["point"] in selected_points
        }
        metric_complete = {
            point
            for point, attempts in attempts_by_point.items()
            if attempts
            and sum(bool(attempt.get("terminal_end")) for attempt in attempts) == 1
            and len(metric_by_point.get(point, [])) == len(attempts)
            and all(row["metric_gate_accepted"] for row in metric_by_point[point])
        }
        summaries[scope] = {
            "scope": scope,
            "broadcasts": len({point_match(point) for point in selected_points}),
            "points": len(selected_points),
            "truth_points": len(selected_truth),
            "flights": len(selected_rows),
            "solved_flights": sum(row["solved"] for row in selected_rows),
            "flights_with_owner_bounce": sum(
                bool(row.get("owner_bounces")) for row in selected_rows
            ),
            "bounce_10cm_pass_flights": sum(
                row.get("bounce_10cm_pass") is True for row in selected_rows
            ),
            "bounce_error_max_m_median_p90": percentile_pair(bounce_errors),
            "net_crossing_required_flights": sum(
                bool(row.get("net_crossing_required")) for row in selected_rows
            ),
            "net_crossing_required_pass_flights": sum(
                bool(row.get("net_crossing_required")) and row.get("net_crossing_plausible") is True
                for row in selected_rows
            ),
            "contact_height_plausible_flights": sum(
                row.get("contact_height_plausible") is True for row in selected_rows
            ),
            "contact_reach_plausible_flights": sum(
                row.get("contact_reach_plausible") is True for row in selected_rows
            ),
            "pixel_gate_flights": sum(row["pixel_gate_accepted"] for row in selected_rows),
            "metric_gate_flights": sum(row["metric_gate_accepted"] for row in selected_rows),
            "metric_complete_points": len(metric_complete),
            "metric_complete_points_denominator": len(selected_points),
            "metric_complete_truth_points": len(metric_complete & selected_truth),
            "metric_complete_truth_points_denominator": len(selected_truth),
        }
    payload = {
        "schema": "s6_cohort_v2_metric_scope_summary_v1",
        "source": os.fspath(metric_path.resolve()),
        "thresholds": metric["thresholds"],
        "scopes": summaries,
    }
    write_json(output_path, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    root = _data_root()
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument(
        "--source-root", default=root / "processed/wk3_cohort/cohort_root_v2", type=Path
    )
    build.add_argument(
        "--emissions",
        default=root / "processed/wk3_events/cohort_v2_emissions_fixed.json",
        type=Path,
    )
    build.add_argument(
        "--emissions-manifest",
        default=root / "processed/wk3_events/cohort_v2_emissions_fixed.manifest.json",
        type=Path,
    )
    build.add_argument(
        "--output-root",
        default=root / "processed/wk3_s6v2/cohort_root_v2_events",
        type=Path,
    )

    run = subparsers.add_parser("run")
    run.add_argument("--mirror-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--truth-events", type=Path, required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--downweight-contact-adjacent", action="store_true")
    run.add_argument("--shared-contact-fit", action="store_true")

    existing = subparsers.add_parser("summarize-existing")
    existing.add_argument("--report", type=Path, required=True)
    existing.add_argument("--ledger", type=Path, required=True)
    existing.add_argument("--truth-events", type=Path, required=True)
    existing.add_argument("--output", type=Path, required=True)

    metric = subparsers.add_parser("metric")
    metric.add_argument("--report", type=Path, required=True)
    metric.add_argument("--truth-events", type=Path, required=True)
    metric.add_argument("--camera-root", type=Path, required=True)
    metric.add_argument("--output-root", type=Path, required=True)

    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        payload = build_mirror(
            args.source_root,
            args.emissions,
            args.emissions_manifest,
            args.output_root,
        )
    elif args.command == "run":
        payload = run_reconstruction(
            args.mirror_root,
            args.output_root,
            args.truth_events,
            workers=args.workers,
            downweight_contact_adjacent=args.downweight_contact_adjacent,
            shared_contact_fit=args.shared_contact_fit,
        )
    elif args.command == "summarize-existing":
        payload = summarize_existing(
            args.report,
            args.ledger,
            args.truth_events,
            args.output,
        )
    else:
        metric = run_metric_acceptance(
            args.report,
            args.truth_events,
            args.camera_root,
            args.output_root,
        )
        payload = summarize_metric_scopes(
            args.report,
            args.output_root / "metric_acceptance.json",
            args.truth_events,
            args.output_root / "scope_summary.json",
        )
        payload["all_scope_source_summary"] = metric["summary"]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
