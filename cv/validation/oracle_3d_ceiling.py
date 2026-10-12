"""Measure the unchanged Stage-6 ceiling under progressively perfect 2D inputs.

This is an evaluation-only diagnostic.  Arms B-D read owner labels and must never
be imported or invoked by automatic inference.  The fitter, physics model, and
flight acceptance gate are imported unchanged from :mod:`cv.pipeline`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from threadpoolctl import threadpool_limits

from cv.pipeline import reconstruction
from cv.pipeline.flight_ledger import build as build_flight_ledger
from cv.pipeline.flight_ledger import net_clearance_m
from cv.pipeline.flight_anchors import build_point_anchors as automatic_build_point_anchors
from cv.pipeline.reconstruction import TRACK_NAME
from cv.validation.current_standard_event_truth import authoritative_truth_path
from cv.validation.score_cross_match_event_labels_v5 import ACCEPTED, EVENT_TYPES
from cv.validation.wk3_s6_cohort import residual_diagnostics

ARTIFACT_CLASS = "oracle_diagnostic"
SCHEMA = "oracle_3d_ceiling_v1"
EXPECTED_POINTS = 198
EXPECTED_TRUTH_POINTS = 146
EXPECTED_EVENTS = 1_415
BALL_SEQUENCE_FILES = (
    "development_truth_v1.json",
    "sealed_transfer_truth_v1.json",
)
SEED_20260902_POINTS = {
    "ao2022f_w_barty_collins__pt0001",
    "ao2022f_w_barty_collins__pt0002",
    "ao2026r128_m_bublik_brooksby__pt0003",
    "cincy2024f_m_sinner_tiafoe__pt0001",
    "rg2017f_m_nadal_wawrinka__pt0002",
    "rg2024f_w_swiatek_paolini__pt0001",
    "rg2024f_w_swiatek_paolini__pt0003",
    "uso2020f_m_zverev_thiem__pt0007",
    "uso2024qf_w_pegula_swiatek__pt0001",
    "uso2024qf_w_pegula_swiatek__pt0002",
    "uso2024sf_m_fritz_tiafoe__pt0003",
    "wim2023f_m_alcaraz_djokovic__pt0005",
    "wim2023f_m_alcaraz_djokovic__pt0007",
    "wim2024f_w_krejcikova_paolini__pt0004",
    "wta_2024_540_r64_178_harriet_dart_katie_boulter__pt0001",
    "wta_2024_540_r64_178_harriet_dart_katie_boulter__pt0003",
    "wta_2026_580_r128_110_yulia_putintseva_beatriz_haddad_maia__pt0002",
    "wtaf2022f_w_garcia_sabalenka__pt0001",
    "wtaf2022f_w_garcia_sabalenka__pt0008",
    "wtaf2025f_w_rybakina_sabalenka__pt0002",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload.get("artifact_class") != ARTIFACT_CLASS:
        payload = {"artifact_class": ARTIFACT_CLASS, **payload}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def cohort_points(manifest_path: Path) -> tuple[set[str], dict[str, float]]:
    manifest = json.loads(manifest_path.read_text())
    points = {
        f"{row['id']}__pt{int(point_id):04d}"
        for row in manifest["matches"]
        for point_id in row["point_ids"]
    }
    fps = {str(row["id"]): float(row["source_fps"]) for row in manifest["matches"]}
    if len(points) != EXPECTED_POINTS:
        raise ValueError(f"expected {EXPECTED_POINTS} cohort points, found {len(points)}")
    return points, fps


def load_owner_events(
    truth_path: Path, points: set[str], fps_by_match: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Convert pinned current-standard truth into event_model_v3-shaped emissions."""
    emissions = []
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with truth_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in EVENT_TYPES or row["verdict"] not in ACCEPTED:
                continue
            point = str(row["clip"])
            if point not in points:
                continue
            frame_text = row["labeled_frame"] or row["seed_frame"]
            if not frame_text or not row["labeled_x540"] or not row["labeled_y540"]:
                raise ValueError(f"owner event lacks time/location: {row}")
            frame = float(frame_text)
            match_id = point.rsplit("__", 1)[0]
            fps = fps_by_match[match_id]
            event = {
                "schema": "event_model_v3_emission_v1",
                "artifact_class": ARTIFACT_CLASS,
                "clip": point,
                "match_id": match_id,
                "event_type": row["event_type"],
                "frame": frame,
                "candidate_frame": frame,
                "confidence": 1.0,
                "probability": 1.0,
                "class_probabilities": {
                    name: float(name == row["event_type"])
                    for name in ("contact", "bounce", "net_hit", "none")
                },
                "abstain": False,
                "decision_threshold": 1.0,
                "abstention_gap": 0.0,
                "path_marginal": 1.0,
                "gate_held": False,
                "point_gate_failure_reasons": [],
                "point_gate_verdict": "retain",
                "production_scope": True,
                "point_grammar": {
                    "schema": "oracle_truth_point_grammar_metadata_v1",
                    "in_play": True,
                    "verdict": "in_play",
                    "confidence": 1.0,
                    "truth_inputs_loaded": True,
                    "diagnostic_only": True,
                },
                "location": {
                    "source": "owner_truth",
                    "image_x": 2.0 * float(row["labeled_x540"]),
                    "image_y": 2.0 * float(row["labeled_y540"]),
                    "image_coordinate_space": "native_1920x1080",
                    "frame_subpixel": frame,
                    "time_seconds": frame / fps,
                    "fps": fps,
                    "court_x_m": None,
                    "court_y_m": None,
                },
                "provenance": "owner_truth",
                "owner_truth": {
                    "seed_id": row["seed_id"],
                    "verdict": row["verdict"],
                    "coordinate_source": "px540_scaled_exactly_2x_to_native",
                },
            }
            emissions.append(event)
            by_point[point].append(event)
    emissions.sort(key=lambda row: (row["clip"], row["frame"], row["event_type"]))
    for rows in by_point.values():
        rows.sort(key=lambda row: (row["frame"], row["event_type"]))
    if len(emissions) != EXPECTED_EVENTS or len(by_point) != EXPECTED_TRUTH_POINTS:
        raise ValueError(
            f"expected {EXPECTED_EVENTS} events/{EXPECTED_TRUTH_POINTS} points, "
            f"found {len(emissions)}/{len(by_point)}"
        )
    return emissions, dict(by_point)


def _sequence_point(case_id: str) -> str:
    return case_id.rsplit("__f", 1)[0]


def load_owner_positions(
    labels_root: Path, points: set[str]
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Load only the two owner position sources named in the package brief."""
    positions: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    sources = []
    sequence_root = labels_root / "ball_track_sequence_v1"
    sequence_rows = 0
    for filename in BALL_SEQUENCE_FILES:
        path = sequence_root / filename
        payload = json.loads(path.read_text())
        sources.append({"path": os.fspath(path.resolve()), "sha256": sha256(path)})
        for record in payload["records"]:
            point = _sequence_point(str(record["case_id"]))
            if point not in points:
                continue
            for frame_row in record["frames"]:
                if frame_row["status"] != "visible":
                    continue
                frame = int(frame_row["frame"])
                positions[point][frame] = {
                    "x1080": float(frame_row["x1080"]),
                    "y1080": float(frame_row["y1080"]),
                    "source": f"ball_track_sequence_v1/{filename}",
                }
                sequence_rows += 1

    trajectory_path = labels_root / "ball_trajectory_v1" / "ball_trajectory_labels.csv"
    sources.append(
        {"path": os.fspath(trajectory_path.resolve()), "sha256": sha256(trajectory_path)}
    )
    trajectory_rows = 0
    conflicts = []
    with trajectory_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            point = f"{row['match_id']}__{row['clip']}"
            if (
                point not in points
                or not row["frame"]
                or not row["corrected_x540"]
                or not row["corrected_y540"]
            ):
                continue
            frame = int(float(row["frame"]))
            candidate = {
                "x1080": 2.0 * float(row["corrected_x540"]),
                "y1080": 2.0 * float(row["corrected_y540"]),
                "source": "ball_trajectory_v1",
            }
            if frame in positions[point]:
                conflicts.append({"point": point, "frame": frame})
                continue
            positions[point][frame] = candidate
            trajectory_rows += 1

    metadata = {
        "artifact_class": ARTIFACT_CLASS,
        "schema": "oracle_owner_ball_positions_v1",
        "sources": sources,
        "points": len(positions),
        "frames": sum(len(rows) for rows in positions.values()),
        "sequence_frames": sequence_rows,
        "trajectory_frames": trajectory_rows,
        "source_conflicts_sequence_preferred": conflicts,
        "provenance": "owner_truth",
    }
    return {point: dict(rows) for point, rows in positions.items()}, metadata


def overlay_track_rows(
    rows: list[dict[str, str]], positions: dict[str, dict[int, dict[str, Any]]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Replace or add every owner-labelled frame while preserving all other rows."""
    output: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        frame = reconstruction.frame_number(row["frame"])
        output[(row["clip"], frame)] = {
            **row,
            "provenance": "automatic",
            "artifact_class": ARTIFACT_CLASS,
        }
    replaced = 0
    added = 0
    for point, frame_rows in positions.items():
        clip = point.rsplit("__", 1)[1]
        for frame, truth in frame_rows.items():
            key = (clip, frame)
            replaced += key in output
            added += key not in output
            output[key] = {
                "clip": clip,
                "frame": f"f_{frame:04d}.jpg",
                "x": f"{float(truth['x1080']) / 2.0:.10g}",
                "y": f"{float(truth['y1080']) / 2.0:.10g}",
                "track_id": "0",
                "score": "1.0",
                "rank": "0",
                "sources": f"owner_truth:{truth['source']}",
                "provenance": "owner_truth",
                "artifact_class": ARTIFACT_CLASS,
            }
    ordered = [output[key] for key in sorted(output, key=lambda key: (key[0], key[1]))]
    return ordered, {"replaced": replaced, "added": added, "owner_rows": replaced + added}


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def build_truth_track_root(
    source_root: Path,
    output_root: Path,
    positions: dict[str, dict[int, dict[str, Any]]],
    position_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a symlink mirror whose canonical S4 track has owner rows overlaid."""
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing oracle root: {output_root}")
    output_root.mkdir(parents=True)
    manifest = json.loads((source_root / "manifest.json").read_text())
    match_ids = {str(row["id"]) for row in manifest["matches"]}
    totals = Counter()
    modified_matches = []
    for entry in sorted(source_root.iterdir()):
        destination = output_root / entry.name
        if entry.name not in match_ids:
            _link(entry, destination)
            continue
        destination.mkdir()
        match_positions = {
            point: rows
            for point, rows in positions.items()
            if point.rsplit("__", 1)[0] == entry.name
        }
        for child in sorted(entry.iterdir()):
            child_destination = destination / child.name
            if child.name not in {TRACK_NAME, f"{TRACK_NAME}.coordinates.json"}:
                _link(child, child_destination)
                continue
            if child.name.endswith(".coordinates.json"):
                coordinate_metadata = json.loads(child.read_text())
                write_json(
                    child_destination,
                    {
                        **coordinate_metadata,
                        "provenance": "automatic_with_owner_truth_overlays",
                        "human_derived_inputs": position_metadata["sources"],
                    },
                )
                continue
            with child.open(newline="") as handle:
                source_rows = list(csv.DictReader(handle))
            overlaid, counts = overlay_track_rows(source_rows, match_positions)
            fieldnames = (
                [*source_rows[0].keys(), "provenance", "artifact_class"]
                if source_rows
                else [
                    "clip",
                    "frame",
                    "x",
                    "y",
                    "track_id",
                    "score",
                    "rank",
                    "sources",
                    "provenance",
                    "artifact_class",
                ]
            )
            with child_destination.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(overlaid)
            totals.update(counts)
            if counts["owner_rows"]:
                modified_matches.append(entry.name)
    report = {
        "artifact_class": ARTIFACT_CLASS,
        "schema": "oracle_truth_track_root_v1",
        "source_root": os.fspath(source_root.resolve()),
        "source_manifest_sha256": sha256(source_root / "manifest.json"),
        "output_root": os.fspath(output_root.resolve()),
        "track_name": TRACK_NAME,
        "provenance_columns": ["provenance", "artifact_class"],
        "modified_matches": sorted(modified_matches),
        "counts": dict(totals),
        "owner_positions": position_metadata,
        "diagnostic_only": True,
    }
    write_json(output_root / "oracle_diagnostic_manifest.json", report)
    return report


def load_truth_boundaries_with_locations(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Expose nested event_model_v3 owner pixels to the existing anchor API for arm D."""
    payload = json.loads(path.read_text())
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["emissions"]:
        location = row["location"]
        output[row["clip"]].append(
            {
                "clip": row["clip"],
                "event_type": row["event_type"],
                "frame": float(row["frame"]),
                "origin": "owner_truth",
                "probability": 1.0,
                "x1080": float(location["image_x"]),
                "y1080": float(location["image_y"]),
                "location": location,
                "provenance": "owner_truth",
            }
        )
    for rows in output.values():
        rows.sort(key=lambda row: (row["frame"], row["event_type"]))
    return dict(output)


def _augment_anchor_track(
    point_key: str,
    track: dict[int, np.ndarray],
    owner_positions: dict[str, dict[int, dict[str, Any]]],
    events: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], set[int]]:
    """Restore exact owner rows after smoothing and add event clicks for anchor fitting only."""
    augmented = {
        int(frame): np.asarray(pixel, dtype=float).copy() for frame, pixel in track.items()
    }
    owner_frames = set()
    for frame, row in owner_positions.get(point_key, {}).items():
        augmented[int(frame)] = np.array([row["x1080"], row["y1080"]], dtype=float)
        owner_frames.add(int(frame))
    for event in events:
        # Half-frame truth is the owner's -0.5-frame leading-blur convention; its
        # clicked image is on the following native frame.  This rounded sample is
        # used only by the crossing bootstrap. Bounce anchors retain exact time.
        frame = int(math.ceil(float(event["frame"])))
        if event.get("x1080") is None or event.get("y1080") is None:
            continue
        augmented[frame] = np.array([event["x1080"], event["y1080"]], dtype=float)
        owner_frames.add(frame)
    return augmented, owner_frames


@contextmanager
def diagnostic_anchor_builder(
    arm: str,
    owner_positions: dict[str, dict[int, dict[str, Any]]],
) -> Iterator[None]:
    """Decorate all anchor artifacts and make only arm D's anchors truth-derived."""
    original = reconstruction.build_point_anchors

    def build(
        point_key: str,
        flight_attempts: list[dict],
        boundary_events: list[dict],
        track: dict[int, np.ndarray],
        camera: Any,
        *,
        pixel_sigma: float = 2.0,
    ) -> dict[str, Any]:
        anchor_track = track
        owner_frames: set[int] = set()
        if arm == "D":
            anchor_track, owner_frames = _augment_anchor_track(
                point_key, track, owner_positions, boundary_events
            )
        artifact = automatic_build_point_anchors(
            point_key,
            flight_attempts,
            boundary_events,
            anchor_track,
            camera,
            pixel_sigma=pixel_sigma,
        )
        human_inputs = []
        if arm in {"B", "C", "D"}:
            human_inputs.append("current_standard_truth_v2")
        if arm in {"C", "D"}:
            human_inputs.extend(["ball_track_sequence_v1", "ball_trajectory_v1"])
        if arm == "D":
            human_inputs.append("truth_derived_bounce_and_net_anchors")
            for flight in artifact["flights"]:
                start = float(flight["start_frame"])
                end = float(flight["end_frame"])
                for anchor in flight["anchors"]:
                    if anchor["type"] == "bounce" and anchor["source"].startswith("emission:"):
                        anchor["source"] = "owner_truth_event_pixel"
                    elif anchor["type"] == "net_crossing":
                        used_truth = any(start <= frame <= end for frame in owner_frames)
                        anchor["source"] = (
                            "owner_truth_informed_track_crossing"
                            if used_truth
                            else "automatic_track_crossing_no_owner_position_overlap"
                        )
        artifact.update(
            {
                "artifact_class": ARTIFACT_CLASS,
                "diagnostic_arm": arm,
                "automatic": arm == "A",
                "human_derived_inputs": human_inputs,
            }
        )
        return artifact

    reconstruction.build_point_anchors = build
    try:
        yield
    finally:
        reconstruction.build_point_anchors = original


def percentile_pair(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=float)
    return [float(np.median(array)), float(np.percentile(array, 90))]


def summarize_arm(
    report: dict[str, Any],
    ledger: dict[str, Any],
    truth_points: set[str],
    wall_time_seconds: float,
) -> dict[str, Any]:
    fits = [fit for point in report["points_detail"] for fit in point.get("fits", [])]
    attempts = [
        attempt for point in report["points_detail"] for attempt in point.get("flight_attempts", [])
    ]
    gaps = [
        float(gap)
        for point in report["points_detail"]
        for gap in point.get("junction_gaps_m", [])
        if math.isfinite(float(gap))
    ]
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
    held_frame_errors = [
        float(row["error_px"])
        for fit in fits
        for row in fit.get("held_out_frame_errors", [])
        if math.isfinite(float(row["error_px"]))
    ]
    clearances = [value for fit in fits if (value := net_clearance_m(fit)) is not None]
    accepted_points = {
        point["point"]
        for point in report["points_detail"]
        if point.get("complete_point_gate", {}).get("accepted") is True
    }
    statuses = Counter(row["status"] for row in ledger["rows"])
    return {
        "artifact_class": ARTIFACT_CLASS,
        "schema": "oracle_3d_ceiling_arm_metrics_v1",
        "points": len(report["points_detail"]),
        "truth_event_points": len(truth_points),
        "attempted_flights": len(attempts),
        "solved_flights": len(fits),
        "accepted_flights": statuses["provisional_valid"],
        "accepted_complete_points_over_146": len(accepted_points & truth_points),
        "accepted_complete_points_over_198": len(accepted_points),
        "held_out_frame_error_px_median_p90": percentile_pair(held_frame_errors),
        "held_out_flight_median_px_corpus_median_p90": percentile_pair(held_medians),
        "held_out_flight_p90_px_corpus_median_p90": percentile_pair(held_p90s),
        "junction_gap_m_median_p90": percentile_pair(gaps),
        "junctions": len(gaps),
        "net_crossings": len(clearances),
        "net_violations": sum(float(value) < -0.01 for value in clearances),
        "wall_time_seconds": float(wall_time_seconds),
        "point_timeouts": sum(
            "point_timeout" in point.get("reasons", []) for point in report["points_detail"]
        ),
        "ledger_statuses": dict(statuses),
    }


def annotate_report(report: dict[str, Any], arm: str, human_inputs: list[dict]) -> None:
    prior_status = report.get("status")
    report.update(
        {
            "artifact_class": ARTIFACT_CLASS,
            "oracle_schema": SCHEMA,
            "diagnostic_arm": arm,
            "status": ARTIFACT_CLASS,
            "pipeline_status_before_diagnostic_annotation": prior_status,
            "automatic_result": arm == "A",
            "human_labels_used": arm != "A",
            "human_derived_inputs": human_inputs,
        }
    )
    report.setdefault("evaluation", {})["owner_truth_loaded"] = arm != "A"


def run_arm(
    arm: str,
    *,
    cohort_root: Path,
    truth_track_root: Path,
    manifest_path: Path,
    automatic_events_path: Path,
    truth_events_path: Path,
    truth_points: set[str],
    owner_positions: dict[str, dict[int, dict[str, Any]]],
    human_inputs: list[dict],
    output_root: Path,
    workers: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_root = output_root / f"arm_{arm.lower()}"
    if arm_root.exists():
        raise FileExistsError(f"refusing to replace existing arm output: {arm_root}")
    arm_root.mkdir(parents=True)
    audit_root = truth_track_root if arm in {"C", "D"} else cohort_root
    events_path = automatic_events_path if arm == "A" else truth_events_path
    if arm == "D":
        boundaries = load_truth_boundaries_with_locations(events_path)
    else:
        boundaries = reconstruction.load_event_boundaries(events_path)
    clips = reconstruction.load_automatic_point_universe(audit_root, manifest_path)
    started = time.perf_counter()
    with diagnostic_anchor_builder(arm, owner_positions), threadpool_limits(limits=1):
        report = reconstruction._run_reconstruction(
            audit_root,
            manifest_path,
            clips=clips,
            automatic_boundaries=boundaries,
            max_nfev=20,
            workers=workers,
            terminal_flights=True,
            anchor_bounce_geometry=True,
            anchor_first=True,
            shared_contact_fit=False,
            whole_point_branches=True,
            branch_width=3,
            branch_margin=8.0,
            event_boundary_source=os.fspath(events_path.resolve()),
            event_consumer_mode="point_grammar_in_play_only",
            anchors_output_root=arm_root / "anchors",
            point_timeout_seconds=timeout_seconds,
            math_threads=1,
            isolate_points=True,
        )
    elapsed = time.perf_counter() - started
    annotate_report(report, arm, human_inputs if arm != "A" else [])
    report_path = arm_root / "report.json"
    write_json(report_path, report)
    ledger = build_flight_ledger([report_path])
    ledger.update(
        {
            "artifact_class": ARTIFACT_CLASS,
            "oracle_schema": SCHEMA,
            "diagnostic_arm": arm,
            "owner_truth_loaded": arm != "A",
            "automatic_result": arm == "A",
            "human_derived_inputs": human_inputs if arm != "A" else [],
        }
    )
    write_json(arm_root / "ledger.json", ledger)
    metrics = summarize_arm(report, ledger, truth_points, elapsed)
    metrics.update(
        {
            "diagnostic_arm": arm,
            "automatic_result": arm == "A",
            "human_labels_used": arm != "A",
        }
    )
    write_json(arm_root / "metrics.json", metrics)
    write_json(
        arm_root / "run_manifest.json",
        {
            "schema": "oracle_3d_ceiling_run_manifest_v1",
            "diagnostic_arm": arm,
            "automatic_result": arm == "A",
            "human_labels_used": arm != "A",
            "human_derived_inputs": human_inputs if arm != "A" else [],
            "audit_root": os.fspath(audit_root.resolve()),
            "manifest": os.fspath(manifest_path.resolve()),
            "events": {"path": os.fspath(events_path.resolve()), "sha256": sha256(events_path)},
            "configuration": {
                "fitter": "default_anchor_first",
                "workers": workers,
                "math_threads": 1,
                "point_timeout_seconds": timeout_seconds,
                "max_nfev": 20,
                "terminal_flights": True,
                "whole_point_branches_requested": True,
                "branch_width": 3,
                "branch_margin": 8.0,
                "held_out_gate": "unchanged",
            },
            "wall_time_seconds": elapsed,
        },
    )
    return report, ledger


def _boundary_distance(row: dict[str, Any], worst: dict[str, Any]) -> float | None:
    boundaries = [row.get("start_frame"), row.get("end_frame")]
    values = [
        abs(float(worst["frame"]) - float(value)) for value in boundaries if value is not None
    ]
    return min(values) if values else None


def write_worst_frame_comparison(
    output_path: Path,
    reports: dict[str, dict[str, Any]],
    ledgers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arms = {}
    for arm in ("A", "B"):
        rows = [
            row
            for row in residual_diagnostics(reports[arm], ledgers[arm])
            if row["point"] in SEED_20260902_POINTS
        ]
        contact_adjacent = 0
        worst_errors = []
        for row in rows:
            for worst in row["worst_frames"]:
                worst["distance_to_flight_boundary_frames"] = _boundary_distance(row, worst)
            if row["worst_frames"]:
                worst_errors.append(float(row["worst_frames"][0]["error_px"]))
                distance = row["worst_frames"][0]["distance_to_flight_boundary_frames"]
                contact_adjacent += distance is not None and distance <= 1.0
        arms[arm] = {
            "solved_flights": len(rows),
            "flights_with_worst_frame_within_1_of_boundary": contact_adjacent,
            "worst_frame_error_px_median_p90": percentile_pair(worst_errors),
            "rows": rows,
        }
    indexed = {
        arm: {
            (row["point"], float(row["start_frame"]), float(row["end_frame"])): row
            for row in arms[arm]["rows"]
        }
        for arm in ("A", "B")
    }
    comparable_rows = []
    for key in sorted(indexed["A"].keys() & indexed["B"].keys()):
        row_a = indexed["A"][key]
        row_b = indexed["B"][key]
        worst_a = row_a["worst_frames"][0]
        worst_b = row_b["worst_frames"][0]
        comparable_rows.append(
            {
                "point": key[0],
                "start_frame": key[1],
                "end_frame": key[2],
                "A": worst_a,
                "B": worst_b,
                "same_worst_frame_and_error": (
                    worst_a["frame"] == worst_b["frame"]
                    and math.isclose(
                        float(worst_a["error_px"]), float(worst_b["error_px"]), abs_tol=1e-9
                    )
                ),
            }
        )
    comparable_summary = {}
    for arm in ("A", "B"):
        worst = [float(row[arm]["error_px"]) for row in comparable_rows]
        comparable_summary[arm] = {
            "worst_frame_error_px_median_p90": percentile_pair(worst),
            "worst_frame_within_1_of_boundary": sum(
                float(row[arm]["distance_to_flight_boundary_frames"]) <= 1.0
                for row in comparable_rows
            ),
        }
    payload = {
        "artifact_class": ARTIFACT_CLASS,
        "schema": "oracle_3d_ceiling_seed20_worst_frames_v1",
        "seed": 20260902,
        "points": sorted(SEED_20260902_POINTS),
        "arms": arms,
        "same_boundary_comparison": {
            "flights": len(comparable_rows),
            "same_worst_frame_and_error": sum(
                row["same_worst_frame_and_error"] for row in comparable_rows
            ),
            "summary": comparable_summary,
            "rows": comparable_rows,
        },
    }
    write_json(output_path, payload)
    return payload


def write_arm_d_non_acceptance(
    output_path: Path, report: dict[str, Any], ledger: dict[str, Any], truth_points: set[str]
) -> dict[str, Any]:
    rows = [
        {
            "flight_id": row["flight_id"],
            "point": row["point"],
            "flight_index": row["flight_index"],
            "status": row["status"],
            "solved": row["solved"],
            "reasons": row["reasons"],
        }
        for row in ledger["rows"]
        if row["status"] != "provisional_valid"
    ]
    reasons = Counter(reason for row in rows for reason in row["reasons"])
    attempted_points = {row["point"] for row in ledger["rows"]}
    point_lookup = {row["point"]: row for row in report["points_detail"]}
    zero_attempt = [
        {
            "point": point,
            "point_decision": point_lookup[point].get("decision"),
            "reasons": point_lookup[point].get("reasons", []),
        }
        for point in sorted(truth_points - attempted_points)
    ]
    payload = {
        "artifact_class": ARTIFACT_CLASS,
        "schema": "oracle_3d_ceiling_arm_d_non_acceptance_v1",
        "non_accepted_flights": len(rows),
        "reason_counts": dict(reasons.most_common()),
        "rows": rows,
        "truth_points_with_zero_attempted_flights": zero_attempt,
    }
    write_json(output_path, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    cohort_root = args.cohort_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing output root: {output_root}")
    output_root.mkdir(parents=True)
    manifest_path = cohort_root / "manifest.json"
    points, fps_by_match = cohort_points(manifest_path)
    truth_path = authoritative_truth_path()
    emissions, owner_events = load_owner_events(truth_path, points, fps_by_match)
    truth_points = set(owner_events)
    labels_root = Path(__file__).with_name("labels")
    owner_positions, position_metadata = load_owner_positions(labels_root, points)

    human_inputs = [
        {"role": "owner_event_truth", "path": os.fspath(truth_path), "sha256": sha256(truth_path)},
        *position_metadata["sources"],
    ]
    truth_events_path = output_root / "truth_events_event_model_v3.json"
    write_json(
        truth_events_path,
        {
            "schema": "oracle_event_model_v3_emissions_v1",
            "diagnostic_only": True,
            "provenance": "owner_truth",
            "source": human_inputs[0],
            "events": len(emissions),
            "points": len(truth_points),
            "emissions": emissions,
        },
    )
    write_json(output_root / "owner_positions.json", position_metadata)
    truth_track_root = output_root / "truth_track_root"
    track_report = build_truth_track_root(
        cohort_root, truth_track_root, owner_positions, position_metadata
    )

    automatic_events_path = cohort_root / "postseg_event_benchmark_v1" / "automatic_emissions.json"
    reports = {}
    ledgers = {}
    for arm in "ABCD":
        reports[arm], ledgers[arm] = run_arm(
            arm,
            cohort_root=cohort_root,
            truth_track_root=truth_track_root,
            manifest_path=manifest_path,
            automatic_events_path=automatic_events_path,
            truth_events_path=truth_events_path,
            truth_points=truth_points,
            owner_positions=owner_positions,
            human_inputs=human_inputs,
            output_root=output_root,
            workers=args.workers,
            timeout_seconds=args.point_timeout_seconds,
        )
        print(f"completed arm {arm}", flush=True)

    worst = write_worst_frame_comparison(
        output_root / "seed20_a_vs_b_worst_frames.json", reports, ledgers
    )
    non_acceptance = write_arm_d_non_acceptance(
        output_root / "arm_d" / "non_acceptance.json",
        reports["D"],
        ledgers["D"],
        truth_points,
    )
    summary = {
        "artifact_class": ARTIFACT_CLASS,
        "schema": SCHEMA,
        "diagnostic_only": True,
        "automatic_arm": "A",
        "human_label_arms": ["B", "C", "D"],
        "cohort": {
            "root": os.fspath(cohort_root),
            "points": len(points),
            "truth_event_points": len(truth_points),
            "truth_events": len(emissions),
        },
        "truth_track_overlay": track_report,
        "arms": {
            arm: json.loads((output_root / f"arm_{arm.lower()}" / "metrics.json").read_text())
            for arm in "ABCD"
        },
        "seed20_contact_spike_summary": {
            arm: {key: value for key, value in worst["arms"][arm].items() if key != "rows"}
            for arm in ("A", "B")
        },
        "seed20_same_boundary_comparison": {
            key: value for key, value in worst["same_boundary_comparison"].items() if key != "rows"
        },
        "arm_d_non_acceptance": {
            key: value for key, value in non_acceptance.items() if key not in {"rows"}
        },
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--cohort-root",
        type=Path,
        default=Path("data/processed/wk3_s6/cohort_root_v1"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/wk3_oracle/oracle_3d_ceiling_v1"),
    )
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--point-timeout-seconds", type=float, default=600.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.point_timeout_seconds <= 0:
        raise ValueError("--point-timeout-seconds must be positive")
    # The pipeline configures forkserver globally, but arm D's validation-only
    # anchor adapter is intentionally process-local and must be inherited. This
    # standalone CPU harness has not started workers yet, so selecting fork here
    # is deterministic and does not alter any pipeline module or persistent state.
    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
