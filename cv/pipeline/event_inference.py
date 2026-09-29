"""Generate typed-event proposals and apply a frozen model without loading labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cv.pipeline.event_decoder import PROFILES
from cv.pipeline.event_model import NUMERIC, load_bundle, predict
from cv.pipeline.event_proposals import BASE_SOURCES, _dedupe, generate_clip
from cv.pipeline.resolution import resolve_player_boxes


def _paths(
    match_dir: Path,
    fps: float,
    ball_filename: str = "ball_track_joint_native1080_arc_augmented_v2.csv",
    ball_match_dir: Path | None = None,
    player_match_dir: Path | None = None,
) -> dict[str, Path | float]:
    return {
        "fps": fps,
        "processed": match_dir,
        "camera": match_dir / "camera_P_per_frame_v1.npz",
        "ball": (ball_match_dir or match_dir) / ball_filename,
        "observed_ball": match_dir / "ball_track_joint_native1080_detour_repaired_v2.csv",
        "boxes_path": resolve_player_boxes(player_match_dir or match_dir, sided=True),
        "audio": match_dir / "contact_audio_scores_16k_native_v1.npz",
    }


def generate_proposals(
    manifest: dict,
    processed: Path,
    *,
    sources: set[str] = BASE_SOURCES,
    ball_filename: str = "ball_track_joint_native1080_arc_augmented_v2.csv",
    ball_root: Path | None = None,
    player_root: Path | None = None,
) -> list[dict]:
    rows = []
    for match in manifest["matches"]:
        match_id = match["id"]
        fps = float(match["source_fps"])
        match_dir = processed / match_id
        with np.load(match_dir / "camera_P_per_frame_v1.npz") as camera:
            camera_clips = {str(value) for value in camera["clips"]}
        point_ids = match.get("point_ids", range(1, int(manifest["points_per_match"]) + 1))
        for point in point_ids:
            local_clip = f"pt{point:04d}"
            if local_clip not in camera_clips:
                continue
            global_clip = f"{match_id}__{local_clip}"
            local_rows = generate_clip(
                local_clip,
                **_paths(
                    match_dir,
                    fps,
                    ball_filename,
                    (ball_root / match_id) if ball_root is not None else None,
                    (player_root / match_id) if player_root is not None else None,
                ),
            )
            for row in _dedupe(local_rows, sources, in_play_only=False):
                row["clip"] = global_clip
                row["candidate_id"] = f"{match_id}__{row['candidate_id']}"
                row["match_id"] = match_id
                row["source_fps"] = fps
                rows.append(row)
    add_proposal_context(rows)
    return rows


def add_proposal_context(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["clip"]].append(row)
    for clip_rows in grouped.values():
        ordered = sorted(clip_rows, key=lambda row: float(row["proposal_frame"]))
        frames = np.asarray([float(row["proposal_frame"]) for row in ordered])
        fps = float(ordered[0]["source_fps"])
        duration = max(float(frames[-1]), 1.0)
        for index, row in enumerate(ordered):
            frame = frames[index]
            court_x = float(row.get("court_x") or 0.0)
            court_y = float(row.get("court_y") or 0.0)
            row["previous_gap_seconds"] = (frame - frames[index - 1]) / fps if index else 99.0
            row["next_gap_seconds"] = (
                (frames[index + 1] - frame) / fps if index + 1 < len(frames) else 99.0
            )
            row["local_candidates_80ms"] = int(np.sum(np.abs(frames - frame) <= 0.08 * fps))
            row["local_candidates_160ms"] = int(np.sum(np.abs(frames - frame) <= 0.16 * fps))
            row["distance_to_net_m"] = abs(court_y - 11.885)
            row["distance_to_baseline_m"] = min(abs(court_y), abs(court_y - 23.77))
            row["court_outside_m"] = math.hypot(
                max(0.0, 1.37 - court_x, court_x - 9.60),
                max(0.0, -court_y, court_y - 23.77),
            )
            row["clip_progress"] = frame / duration


def apply_production_scope(rows: list[dict], active_play_path: Path, point_gate_path: Path) -> dict:
    active_play = json.loads(active_play_path.read_text())
    point_gate = json.loads(point_gate_path.read_text())
    gates = {f"{row['match_id']}__{row['clip']}": row for row in point_gate["rows"]}
    for row in rows:
        match_id, local_clip = row["clip"].rsplit("__", 1)
        decision = active_play[f"{match_id}/{local_clip}"]
        gate = gates[row["clip"]]
        spans = decision.get("event_spans", decision.get("active_spans", []))
        frame = float(row["proposal_frame"])
        fps = float(decision["fps"])
        containing = next(
            (
                (index, float(start), float(end))
                for index, (start, end) in enumerate(spans)
                if float(start) <= frame <= float(end)
            ),
            None,
        )
        # S3-07: event inference is availability-first. Decode every automatic
        # active-play point and carry the validity gate as metadata so precision
        # consumers can reproduce the retained slice exactly.
        row["production_scope"] = containing is not None
        row["point_gate_verdict"] = gate["decision"]
        row["point_gate_failure_reasons"] = list(gate.get("reasons", []))
        row["gate_held"] = gate["decision"] == "hold"
        row["event_span_count"] = float(len(spans))
        if containing is None:
            row.update(
                event_span_progress=-1.0,
                seconds_to_event_span_start=99.0,
                seconds_to_event_span_end=99.0,
                event_span_index=-1.0,
            )
        else:
            index, start, end = containing
            row.update(
                event_span_progress=(frame - start) / max(end - start, 1.0),
                seconds_to_event_span_start=(frame - start) / fps,
                seconds_to_event_span_end=(end - frame) / fps,
                event_span_index=float(index),
            )
    return {
        "points": point_gate["points"],
        "held_points": point_gate["held"],
        "retained_points": point_gate["retained"],
        "proposals": sum(bool(row["production_scope"]) for row in rows),
        "scope": "phase_event_spans_emit_with_gate_flags",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["production_scope"] = str(row.get("production_scope", "")).lower() == "true"
        for feature in (*NUMERIC, "nearest_player_edge_distance_norm"):
            try:
                value = float(row.get(feature, 0.0))
                row[feature] = value if math.isfinite(value) else 0.0
            except (TypeError, ValueError):
                row[feature] = 0.0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose = subparsers.add_parser("propose")
    propose.add_argument("--manifest", type=Path, required=True)
    propose.add_argument("--root", type=Path, required=True)
    propose.add_argument("--active-play", type=Path, required=True)
    propose.add_argument("--point-gate", type=Path, required=True)
    propose.add_argument("--output", type=Path, required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--proposals", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument(
        "--decoder-profile",
        choices=sorted(PROFILES),
        default="default",
        help="Explicit graph-decoder profile; non-default profiles remain experimental.",
    )
    args = parser.parse_args()
    if args.command == "propose":
        rows = generate_proposals(json.loads(args.manifest.read_text()), args.root)
        apply_production_scope(rows, args.active_play, args.point_gate)
        write_rows(args.output, rows)
    else:
        emissions = predict(
            load_bundle(args.model),
            read_rows(args.proposals),
            decoder_config=PROFILES[args.decoder_profile],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(emissions, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
