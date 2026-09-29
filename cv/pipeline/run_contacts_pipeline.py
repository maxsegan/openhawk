"""Run the canonical non-ball evidence substrate.

The path materializes native frames, cadence, player boxes, and cadence-aware audio evidence.
Canonical ball and event inference belong to the composed pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from cv.pipeline.artifact_cache import (
    receipt_path,
    remove_receipt,
    stage_receipt_matches,
    write_stage_receipt,
)
from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    build_provenance,
    file_record,
    portable_configuration,
)

RUNTIME_MODULE_DEPENDENCIES = (
    "cv.pipeline.players",
    "cv.pipeline.frame_cadence",
    "cv.pipeline.audio_evidence",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    outputs: list[str]
    inputs: list[str] | None = None
    gpu: bool = False
    frames: bool = False


def _path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


def _module_command(python: str, module: str, *arguments: str) -> list[str]:
    return [python, "-m", module, *arguments]


def build_stages(args: argparse.Namespace) -> list[Stage]:
    python = sys.executable
    boxes = f"player_boxes_{args.fps:g}_{args.contact_tag}.csv"
    common = ["--out", args.out]
    point_map = ["--point-map", args.point_map]
    cadence_path = _path(args.out, "frame_cadence_v1.json")

    frame_command = _module_command(
        python,
        "cv.pipeline.players",
        *common,
        "--video",
        args.video,
        "--fps",
        str(args.fps),
        "--frames-dir",
        args.frames_dir,
        *point_map,
        "--extract-only",
        "--artifact-width",
        str(args.artifact_width),
        "--artifact-height",
        str(args.artifact_height),
    )
    if args.preserve_source_fps:
        frame_command.append("--preserve-source-fps")
    cadence_command = _module_command(
        python,
        "cv.pipeline.frame_cadence",
        "--frames-root",
        _path(args.out, args.frames_dir),
        "--match-id",
        args.match,
        "--fps",
        str(args.fps),
        "--out",
        cadence_path,
        "--fail-on-timing-hold",
    )
    player_command = _module_command(
        python,
        "cv.pipeline.players",
        *common,
        "--video",
        args.video,
        "--model",
        args.player_model,
        "--device",
        str(args.device),
        "--fps",
        str(args.fps),
        "--batch",
        str(args.player_batch),
        "--frames-dir",
        args.frames_dir,
        *point_map,
        "--output",
        boxes,
        "--artifact-width",
        str(args.artifact_width),
        "--artifact-height",
        str(args.artifact_height),
    )
    if args.preserve_source_fps:
        player_command.append("--preserve-source-fps")
    audio_evidence_command = _module_command(
        python,
        "cv.pipeline.audio_evidence",
        *common,
        *point_map,
        "--video",
        args.video,
        "--frames-dir",
        args.frames_dir,
        "--audio-cache",
        args.audio_cache,
        "--seed",
        str(args.seed),
    )
    stages = [
        Stage(
            "frame_extraction",
            frame_command,
            [
                _path(args.out, args.frames_dir),
                _path(args.out, f"{args.frames_dir}.coordinates.json"),
            ],
            inputs=[args.video, _path(args.out, args.point_map)],
            frames=True,
        ),
        Stage(
            "frame_cadence",
            cadence_command,
            [cadence_path],
        ),
    ]
    if not getattr(args, "skip_player_detection", False):
        stages.append(
            Stage(
                "player_detection",
                player_command,
                [_path(args.out, boxes), _path(args.out, f"{boxes}.coordinates.json")],
                inputs=[args.player_model] if os.path.isfile(args.player_model) else [],
                gpu=True,
            )
        )
    if not args.skip_audio_evidence:
        stages.append(
            Stage(
                "audio_evidence",
                audio_evidence_command,
                [_path(args.out, args.audio_cache)],
                inputs=[args.video, _path(args.out, args.point_map)],
            )
        )
    return stages


def frame_outputs_ready(
    out_dir: str, frames_dir: str, point_map_name: str = "point_video_map.csv"
) -> bool:
    point_map = os.path.join(out_dir, point_map_name)
    root = os.path.join(out_dir, frames_dir)
    coordinate_manifest = os.path.join(out_dir, f"{frames_dir}.coordinates.json")
    if (
        not os.path.exists(point_map)
        or not os.path.isdir(root)
        or not os.path.exists(coordinate_manifest)
    ):
        return False
    with open(point_map, newline="") as handle:
        expected = {
            f"pt{int(row['pt']):04d}"
            for row in csv.DictReader(handle)
            if row.get("rally_t_start") and row.get("rally_t_end")
        }
    populated = {
        name
        for name in os.listdir(root)
        if name.startswith("pt")
        and os.path.isdir(os.path.join(root, name))
        and any(file.endswith(".jpg") for file in os.listdir(os.path.join(root, name)))
    }
    return bool(expected) and expected <= populated


def stage_outputs_ready(
    stage: Stage,
    out_dir: str,
    frames_dir: str,
    point_map_name: str = "point_video_map.csv",
) -> bool:
    if stage.frames:
        return frame_outputs_ready(out_dir, frames_dir, point_map_name)
    outputs_ready = bool(stage.outputs) and all(
        os.path.exists(path) and os.path.getsize(path) > 0 for path in stage.outputs
    )
    if not outputs_ready:
        return False
    if stage.inputs:
        if not all(os.path.exists(path) for path in stage.inputs):
            return False
        newest_input = max(os.path.getmtime(path) for path in stage.inputs)
        oldest_output = min(os.path.getmtime(path) for path in stage.outputs)
        return oldest_output >= newest_input
    return True


def plan_stages(
    stages: list[Stage],
    out_dir: str,
    frames_dir: str,
    resume: bool,
    point_map_name: str = "point_video_map.csv",
) -> list[tuple[Stage, bool]]:
    """Return ``(stage, reuse)`` pairs from content-addressed stage receipts."""
    upstream_changed = False
    plan = []
    upstream_receipts: list[str] = []
    for stage in stages:
        can_reuse = (
            resume
            and not upstream_changed
            and stage_receipt_matches(
                out_dir=out_dir,
                stage=stage.name,
                command=stage.command,
                inputs=stage.inputs or [],
                outputs=stage.outputs,
                upstream_receipts=upstream_receipts,
            )
        )
        plan.append((stage, can_reuse))
        upstream_changed = upstream_changed or not can_reuse
        upstream_receipts.append(str(receipt_path(out_dir, stage.name)))
    return plan


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_model(device: int) -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def model_records(args: argparse.Namespace) -> list[dict]:
    models = []
    if not getattr(args, "skip_player_detection", False):
        models.append(
            file_record(args.player_model, role="player_detector")
            if os.path.isfile(args.player_model)
            else {"name": args.player_model, "role": "player_detector"}
        )
    return models


def write_manifest(
    args: argparse.Namespace,
    started: float,
    started_iso: str,
    status: str,
    records: list[dict],
) -> str:
    ended = time.time()
    reused_paths = {
        output
        for record in records
        if record["status"] == "reused"
        for output in record["outputs"]
        if os.path.isfile(output)
    }
    point_map = _path(args.out, args.point_map)
    if os.path.isfile(point_map):
        reused_paths.add(point_map)
    provenance = build_provenance(
        root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        mode=AUTOMATIC_MODE,
        source_videos=[file_record(args.video, role="source_video")],
        models=model_records(args),
        configuration=vars(args),
        reused_artifacts=[
            file_record(path, role="reused_substrate_artifact") for path in sorted(reused_paths)
        ],
    )
    manifest = {
        "stage": "contacts_pipeline",
        "status": status,
        "started_at": started_iso,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(ended - started, 3),
        "gpu_devices": [args.device]
        if any(row["gpu"] and row["status"] == "executed" for row in records)
        else [],
        "gpu_models": [gpu_model(args.device)]
        if any(row["gpu"] and row["status"] == "executed" for row in records)
        else [],
        "gpu_hours": round(
            sum(
                row["wall_seconds"] for row in records if row["gpu"] and row["status"] == "executed"
            )
            / 3600.0,
            6,
        ),
        "seed": args.seed,
        "git_hash": git_hash(),
        "config": portable_configuration(vars(args)),
        "provenance": provenance,
        "outputs": {"stages": portable_configuration(records)},
    }
    directory = os.path.join(args.out, "run_manifests")
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = os.path.join(directory, f"contacts_pipeline_{stamp}.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(
        f"pipeline manifest: {path} | wall={manifest['wall_seconds']:.1f}s | "
        f"gpu_hours={manifest['gpu_hours']:.6f} | status={status}",
        flush=True,
    )
    return path


def accept_partial_timing_holds(
    args: argparse.Namespace, stage: Stage, return_code: int
) -> dict | None:
    """Allow point-scoped cadence holds while rejecting an unusable full source."""
    if stage.name != "frame_cadence" or not args.allow_partial_timing_holds or return_code != 2:
        return None
    cadence_path = _path(args.out, "frame_cadence_v1.json")
    try:
        with open(cadence_path) as handle:
            cadence = json.load(handle)
        points = int(cadence["points"])
        holds = int(cadence["timing_holds"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if points <= 0 or holds <= 0 or holds >= points:
        return None
    return {
        "timing_usable_points": points - holds,
        "timing_held_points": holds,
        "policy": "preserve_native_timestamps_and_exclude_held_points_downstream",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--frames-dir",
        default="rally_frames_50_1080",
        help="highest-resolution extracted frames; outputs retain --artifact-width/height coordinates",
    )
    parser.add_argument(
        "--point-map",
        default="point_video_map.csv",
        help="versioned aligned point map within --out",
    )
    parser.add_argument("--artifact-width", type=int, default=960)
    parser.add_argument("--artifact-height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--preserve-source-fps",
        action="store_true",
        help="do not resample video; --fps is the decoded source cadence",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--contact-tag", default="full_v1")
    parser.add_argument(
        "--ball-tag",
        "--gate-tag",
        "--subpixel",
        dest="_retired_runner_option",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--player-model", default="yolov8m.pt")
    parser.add_argument("--player-batch", type=int, default=64)
    parser.add_argument(
        "--skip-player-detection",
        action="store_true",
        help="materialize frames/cadence only so an orchestrator can overlap court and player work",
    )
    parser.add_argument("--audio-cache", default="contact_audio_scores_16k_v1.npz")
    parser.add_argument(
        "--skip-audio-evidence",
        action="store_true",
        help="motion-only profile: omit audio because no downstream stage consumes it",
    )
    parser.add_argument(
        "--allow-partial-timing-holds",
        action="store_true",
        help=(
            "continue when only a subset of points has invalid visual cadence; downstream "
            "consumers must exclude those points. A fully held source still fails closed"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only stages whose content-addressed receipt and outputs still match",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.video):
        parser.error(f"video does not exist: {args.video}")
    if not os.path.exists(_path(args.out, args.point_map)):
        parser.error(f"aligned point map {args.point_map!r} does not exist under {args.out}")
    stages = build_stages(args)
    plan = plan_stages(stages, args.out, args.frames_dir, args.resume, args.point_map)
    for stage, can_reuse in plan:
        action = "skip" if can_reuse else "run"
        print(f"[{action}] {stage.name}: {' '.join(stage.command)}", flush=True)
    if args.dry_run:
        return 0

    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    records = []
    upstream_receipts: list[str] = []
    for stage, can_reuse in plan:
        if can_reuse:
            records.append(
                {
                    "stage": stage.name,
                    "status": "reused",
                    "wall_seconds": 0.0,
                    "gpu": stage.gpu,
                    "command": stage.command,
                    "outputs": stage.outputs,
                }
            )
            upstream_receipts.append(str(receipt_path(args.out, stage.name)))
            continue
        stage_started = time.time()
        remove_receipt(args.out, stage.name)
        try:
            subprocess.run(stage.command, check=True)
        except subprocess.CalledProcessError as error:
            cadence_disposition = accept_partial_timing_holds(args, stage, error.returncode)
            if cadence_disposition is not None:
                receipt = write_stage_receipt(
                    out_dir=args.out,
                    stage=stage.name,
                    command=stage.command,
                    inputs=stage.inputs or [],
                    outputs=stage.outputs,
                    upstream_receipts=upstream_receipts,
                )
                records.append(
                    {
                        "stage": stage.name,
                        "status": "executed_with_point_holds",
                        "wall_seconds": round(time.time() - stage_started, 3),
                        "gpu": stage.gpu,
                        "command": stage.command,
                        "outputs": stage.outputs,
                        "receipt": str(receipt),
                        "disposition": cadence_disposition,
                    }
                )
                upstream_receipts.append(str(receipt))
                continue
            records.append(
                {
                    "stage": stage.name,
                    "status": "failed",
                    "wall_seconds": round(time.time() - stage_started, 3),
                    "gpu": stage.gpu,
                    "command": stage.command,
                    "outputs": stage.outputs,
                }
            )
            write_manifest(args, started, started_iso, "failed", records)
            return 1
        receipt = write_stage_receipt(
            out_dir=args.out,
            stage=stage.name,
            command=stage.command,
            inputs=stage.inputs or [],
            outputs=stage.outputs,
            upstream_receipts=upstream_receipts,
        )
        records.append(
            {
                "stage": stage.name,
                "status": "executed",
                "wall_seconds": round(time.time() - stage_started, 3),
                "gpu": stage.gpu,
                "command": stage.command,
                "outputs": stage.outputs,
                "receipt": str(receipt),
            }
        )
        upstream_receipts.append(str(receipt))
    write_manifest(args, started, started_iso, "ok", records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
