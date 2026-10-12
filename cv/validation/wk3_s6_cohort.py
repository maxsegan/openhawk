"""Build and verify the week-three S6 per-frame-camera cohort mirror.

The source benchmark is immutable.  This module symlinks its unchanged artifacts into a
new root, leaves the court/camera products absent, and invokes the same three producers
used by :mod:`cv.pipeline.canonical_runner`.  No evaluation labels enter the mirror or
the regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from cv.pipeline.camera_artifacts import expand_point_cameras
from cv.pipeline.flight_ledger import net_clearance_m

FRAMES_DIR = "audit_frames_native_1080"
GENERATED_NAMES = {
    "camera_P_per_frame_v1.npz",
    "camera_P_per_point.npz",
    "court_H_per_frame_v1.npz",
    "court_H_per_point.npz",
    "court_topology_evidence_v1.json",
    "court_validation.jpg",
}
GENERATED_PREFIXES = ("camera_net_check_", "camera_projection_audit_")
MIRROR_MANIFEST = "wk3_s6_cohort_manifest.json"
QUALITY_REPORT = "wk3_s6_camera_quality.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_generated(name: str) -> bool:
    return name in GENERATED_NAMES or name.startswith(GENERATED_PREFIXES)


def _link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"wrong existing symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing to replace mirror path: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _mirror_match(source: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    links = 0
    for entry in sorted(source.iterdir()):
        if _is_generated(entry.name):
            continue
        target = destination / entry.name
        if entry.name != "run_manifests":
            _link(entry, target)
            links += 1
            continue

        # camera_cal writes a timestamped manifest here.  Keep the directory local so
        # that the benchmark source cannot be mutated through a directory symlink.
        target.mkdir(exist_ok=True)
        for manifest_entry in sorted(entry.iterdir()):
            _link(manifest_entry, target / manifest_entry.name)
            links += 1
    return links


def build_mirror(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Create an idempotent symlink mirror with court/camera products reserved."""
    source_root = source_root.resolve()
    if source_root == output_root.resolve():
        raise ValueError("source and output roots must differ")
    source_manifest = source_root / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    match_ids = [str(row["id"]) for row in manifest["matches"]]

    output_root.mkdir(parents=True, exist_ok=True)
    links = 0
    for entry in sorted(source_root.iterdir()):
        destination = output_root / entry.name
        if entry.name in match_ids:
            links += _mirror_match(entry, destination)
        elif entry.name not in {MIRROR_MANIFEST, QUALITY_REPORT}:
            _link(entry, destination)
            links += 1

    report = {
        "schema": "wk3_s6_per_frame_camera_cohort_v1",
        "automatic": True,
        "source_root": os.fspath(source_root),
        "source_manifest_sha256": _sha256(source_manifest),
        "output_root": os.fspath(output_root.resolve()),
        "matches": match_ids,
        "points": int(
            sum(
                int(row.get("point_ids") and len(row["point_ids"]) or 0)
                for row in manifest["matches"]
            )
        ),
        "symlinks": links,
        "regenerated_artifacts": sorted(GENERATED_NAMES),
        "court_configuration": {
            "anchor_mode": "first_success",
            "frame_track": True,
            "registration_stride": 5,
            "solve_scale": "default",
        },
        "camera_configuration": {
            "net_support": "singles_sticks",
            "net_frame": "court_anchor",
        },
        "human_derived_inputs": [],
    }
    (output_root / MIRROR_MANIFEST).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _run_match(match_id: str, output_root: Path, repo_root: Path) -> dict[str, Any]:
    match_root = output_root / match_id
    log_dir = output_root / "wk3_s6_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{match_id}.log"
    commands = [
        [
            sys.executable,
            "-m",
            "cv.pipeline.court_topology_runner",
            "--out",
            os.fspath(match_root),
            "--frames-dir",
            FRAMES_DIR,
            "--jobs",
            "1",
        ],
        [
            sys.executable,
            "-m",
            "cv.pipeline.camera_cal",
            "--out",
            os.fspath(match_root),
            "--frames-dir",
            FRAMES_DIR,
            "--net-support",
            "singles_sticks",
            "--net-frame",
            "court_anchor",
            "--overlay",
            "--jobs",
            "1",
        ],
    ]
    with log_path.open("w") as log:
        for command in commands:
            log.write("command: " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        expanded = expand_point_cameras(match_root, FRAMES_DIR)
        log.write(f"expanded: {expanded}\n")
    return {"match_id": match_id, "log": os.fspath(log_path)}


def regenerate(output_root: Path, *, jobs: int) -> list[dict[str, Any]]:
    """Run court, camera calibration, and camera expansion for every cohort match."""
    mirror = json.loads((output_root / MIRROR_MANIFEST).read_text())
    repo_root = Path(__file__).resolve().parents[2]
    completed = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(_run_match, match_id, output_root, repo_root): match_id
            for match_id in mirror["matches"]
        }
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"camera matches {len(completed)}/{len(futures)}", flush=True)
    return sorted(completed, key=lambda row: row["match_id"])


def verify(output_root: Path) -> dict[str, Any]:
    """Verify frame-track scope and report camera reliability for every point."""
    mirror = json.loads((output_root / MIRROR_MANIFEST).read_text())
    cohort = json.loads((output_root / "manifest.json").read_text())
    expected_points = {
        str(row["id"]): [f"pt{int(point_id):04d}" for point_id in row["point_ids"]]
        for row in cohort["matches"]
    }
    active_path = output_root / "active_play_v1.json"
    active = json.loads(active_path.read_text()) if active_path.is_file() else {}
    points = []
    for match_id in mirror["matches"]:
        match_root = output_root / match_id
        camera_path = match_root / "camera_P_per_frame_v1.npz"
        court_path = match_root / "court_H_per_frame_v1.npz"
        if not camera_path.is_file():
            raise FileNotFoundError(f"missing regenerated camera artifact for {match_id}")
        with np.load(camera_path, allow_pickle=True) as camera:
            clips = sorted(set(str(value) for value in camera["clips"]))
            if clips and not court_path.is_file():
                raise FileNotFoundError(f"missing regenerated court track for {match_id}")
            for clip in clips:
                selected = camera["clips"] == clip
                scopes = sorted(set(str(value) for value in camera["frame_scope"][selected]))
                if scopes != ["frame_track"]:
                    raise ValueError(f"{match_id}/{clip} has frame scopes {scopes}")
                frames = camera["frames"][selected]
                reliable = camera["reliable"][selected]
                active_spans = active.get(f"{match_id}/{clip}", {}).get("active_spans", [])
                active_selected = np.asarray(
                    [any(start <= frame <= end for start, end in active_spans) for frame in frames],
                    dtype=bool,
                )
                points.append(
                    {
                        "point": f"{match_id}__{clip}",
                        "match_id": match_id,
                        "clip": clip,
                        "frames": int(np.count_nonzero(selected)),
                        "frame_scope": "frame_track",
                        "reliable_frames": int(np.count_nonzero(reliable)),
                        "reliable_fraction": float(np.mean(reliable)),
                        "active_frames": int(np.count_nonzero(active_selected)),
                        "active_reliable_fraction": (
                            float(np.mean(reliable[active_selected]))
                            if np.any(active_selected)
                            else None
                        ),
                        "sources": sorted(set(str(value) for value in camera["source"][selected])),
                    }
                )
            missing_clips = sorted(set(expected_points[match_id]) - set(clips))
            for clip in missing_clips:
                points.append(
                    {
                        "point": f"{match_id}__{clip}",
                        "match_id": match_id,
                        "clip": clip,
                        "frames": 0,
                        "frame_scope": "unavailable",
                        "reliable_frames": 0,
                        "reliable_fraction": None,
                        "active_frames": 0,
                        "active_reliable_fraction": None,
                        "sources": ["court_abstained"],
                    }
                )
    fractions = np.asarray(
        [row["reliable_fraction"] for row in points if row["reliable_fraction"] is not None],
        dtype=float,
    )
    active_fractions = np.asarray(
        [
            row["active_reliable_fraction"]
            for row in points
            if row["active_reliable_fraction"] is not None
        ],
        dtype=float,
    )
    report = {
        "schema": "wk3_s6_camera_quality_v1",
        "automatic": True,
        "matches": len(mirror["matches"]),
        "points": len(points),
        "available_points": len(fractions),
        "unavailable_points": sum(row["frame_scope"] == "unavailable" for row in points),
        "all_frame_scope_frame_track": all(row["frame_scope"] == "frame_track" for row in points),
        "all_available_frame_scope_frame_track": all(
            row["frame_scope"] in {"frame_track", "unavailable"} for row in points
        ),
        "reliable_fraction": {
            "median": float(np.median(fractions)),
            "p10": float(np.percentile(fractions, 10)),
            "minimum": float(np.min(fractions)),
            "fully_reliable_points": int(np.count_nonzero(fractions == 1.0)),
        },
        "active_reliable_fraction": {
            "points": len(active_fractions),
            "median": float(np.median(active_fractions)) if len(active_fractions) else None,
            "p10": (float(np.percentile(active_fractions, 10)) if len(active_fractions) else None),
            "minimum": float(np.min(active_fractions)) if len(active_fractions) else None,
            "at_least_0_95_points": int(np.count_nonzero(active_fractions >= 0.95)),
        },
        "rows": points,
    }
    (output_root / QUALITY_REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _percentiles(values: list[float]) -> list[float | None]:
    return [float(np.median(values)), float(np.percentile(values, 90))] if values else [None, None]


def _elapsed_seconds(path: Path) -> float | None:
    match = re.search(r"Elapsed \(wall clock\) time.*: (.+)", path.read_text())
    if match is None:
        return None
    fields = [float(value) for value in match.group(1).strip().split(":")]
    return sum(value * 60**index for index, value in enumerate(reversed(fields)))


def residual_diagnostics(report: dict, ledger: dict) -> list[dict[str, Any]]:
    """Describe the checkerboard residual location without reading evaluation truth."""
    ledger_rows = {(row["point"], int(row["flight_index"])): row for row in ledger["rows"]}
    rows = []
    for point in report["points_detail"]:
        for fit in point.get("fits", []):
            index = int(fit["flight_index"])
            ledger_row = ledger_rows.get((point["point"], index), {})
            errors = sorted(
                fit.get("held_out_frame_errors", []),
                key=lambda row: float(row["error_px"]),
                reverse=True,
            )
            worst = errors[:3]
            median = fit.get("held_out_reprojection_median_px")
            p90 = fit.get("held_out_reprojection_p90_px")
            maximum = float(errors[0]["error_px"]) if errors else None
            unreliable_worst = sum(not bool(row.get("camera_reliable")) for row in worst)
            if worst and unreliable_worst >= max(1, math.ceil(len(worst) / 2)):
                diagnosis = "likely_camera_registration"
            elif (
                maximum is not None
                and median is not None
                and float(median) <= 8.0
                and maximum >= max(12.0, 2.5 * float(median))
            ):
                diagnosis = "likely_tracking_outlier"
            elif median is not None and float(median) > 8.0:
                diagnosis = "broad_model_or_anchor_residual"
            elif p90 is not None and float(p90) > 12.0:
                diagnosis = "held_out_tail"
            else:
                diagnosis = "low_residual"
            rows.append(
                {
                    "point": point["point"],
                    "flight_index": index,
                    "status": ledger_row.get("status"),
                    "start_side": ledger_row.get("start_side"),
                    "phase": ledger_row.get("start_phase"),
                    "start_frame": fit.get("start_frame"),
                    "end_frame": fit.get("end_frame"),
                    "held_out_median_px": median,
                    "held_out_p90_px": p90,
                    "worst_frames": [
                        {
                            "frame": row["frame"],
                            "error_px": row["error_px"],
                            "camera_reliable": row.get("camera_reliable"),
                            "camera_source": row.get("camera_source"),
                        }
                        for row in worst
                    ],
                    "diagnostic": diagnosis,
                }
            )
    return rows


def summarize_reconstruction(
    report_path: Path,
    ledger_path: Path,
    wall_time_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    statuses = {}
    for row in ledger["rows"]:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    fits = [fit for point in report["points_detail"] for fit in point.get("fits", [])]
    attempts = [
        attempt for point in report["points_detail"] for attempt in point.get("flight_attempts", [])
    ]
    held_medians = [
        float(fit["held_out_reprojection_median_px"])
        for fit in fits
        if fit.get("held_out_reprojection_median_px") is not None
    ]
    held_p90 = [
        float(fit["held_out_reprojection_p90_px"])
        for fit in fits
        if fit.get("held_out_reprojection_p90_px") is not None
    ]
    gaps = [
        float(gap)
        for point in report["points_detail"]
        for gap in point.get("junction_gaps_m", [])
        if math.isfinite(float(gap))
    ]
    clearances = [value for fit in fits if (value := net_clearance_m(fit)) is not None]
    shared_contacts = {
        (point["point"], int(shared["contact_index"])): float(shared["junction_gap_m"])
        for point in report["points_detail"]
        for fit in point.get("fits", [])
        for shared in fit.get("shared_contacts", [])
    }
    metrics = {
        "points": len(report["points_detail"]),
        "attempted_flights": len(attempts),
        "solved_flights": len(fits),
        "accepted_flights": statuses.get("provisional_valid", 0),
        "accepted_complete_points": sum(
            point.get("complete_point_gate", {}).get("accepted") is True
            for point in report["points_detail"]
        ),
        "held_out_median_px": _percentiles(held_medians),
        "held_out_p90_px": _percentiles(held_p90),
        "junction_gap_m": _percentiles(gaps),
        "junctions": len(gaps),
        "shared_contacts_adopted": len(shared_contacts),
        "shared_contact_gap_m": _percentiles(list(shared_contacts.values())),
        "net_crossings": len(clearances),
        "net_violations": sum(float(value) < -0.01 for value in clearances),
        "wall_time_seconds": _elapsed_seconds(wall_time_path),
        "timeouts": sum(attempt.get("skip_reason") == "point_timeout" for attempt in attempts),
        "ledger_statuses": statuses,
    }
    return metrics, residual_diagnostics(report, ledger)


def _load_anchors(root: Path) -> dict[tuple[str, int, str], np.ndarray]:
    anchors = {}
    for path in sorted(root.glob("*/pt*/anchors_v1.json")):
        match_id = path.parents[1].name
        clip = path.parent.name
        point = f"{match_id}__{clip}"
        for flight in json.loads(path.read_text())["flights"]:
            for anchor in flight["anchors"]:
                anchors[(point, int(flight["flight_index"]), str(anchor["type"]))] = np.asarray(
                    anchor["xyz"], dtype=float
                )
    return anchors


def anchor_movements(old_root: Path, new_root: Path) -> dict[str, Any]:
    old = _load_anchors(old_root)
    new = _load_anchors(new_root)
    rows = []
    for key in sorted(old.keys() & new.keys()):
        distance = float(np.linalg.norm(new[key] - old[key]))
        rows.append(
            {
                "point": key[0],
                "flight_index": key[1],
                "type": key[2],
                "distance_m": distance,
                "moved_over_10cm": distance > 0.10,
            }
        )
    by_type = {}
    for anchor_type in ("bounce", "net_crossing"):
        selected = [row for row in rows if row["type"] == anchor_type]
        by_type[anchor_type] = {
            "compared": len(selected),
            "moved_over_10cm": sum(row["moved_over_10cm"] for row in selected),
            "distance_m": _percentiles([row["distance_m"] for row in selected]),
        }
    return {"schema": "wk3_s6_anchor_movement_v1", "by_type": by_type, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-root", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--jobs", type=int, default=8)
    build_parser.add_argument("--mirror-only", action="store_true")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--report", type=Path, required=True)
    measure_parser.add_argument("--ledger", type=Path, required=True)
    measure_parser.add_argument("--wall-time", type=Path, required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--residual-output", type=Path)
    measure_parser.add_argument("--old-anchors", type=Path)
    measure_parser.add_argument("--new-anchors", type=Path)
    measure_parser.add_argument("--anchor-movement-output", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        metrics, residuals = summarize_reconstruction(args.report, args.ledger, args.wall_time)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        if args.residual_output is not None:
            args.residual_output.write_text(
                json.dumps(
                    {"schema": "wk3_s6_held_out_residuals_v1", "rows": residuals},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        if args.old_anchors is not None or args.new_anchors is not None:
            if None in (args.old_anchors, args.new_anchors, args.anchor_movement_output):
                parser.error(
                    "--old-anchors, --new-anchors, and --anchor-movement-output are required together"
                )
            movement = anchor_movements(args.old_anchors, args.new_anchors)
            args.anchor_movement_output.write_text(
                json.dumps(movement, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    mirror = build_mirror(args.source_root, args.output_root)
    result: dict[str, Any] = {"mirror": mirror}
    if not args.mirror_only:
        result["completed"] = regenerate(args.output_root, jobs=args.jobs)
        result["quality"] = verify(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
