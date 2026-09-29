"""Authoritative automatic post-segmentation inference runner.

This runner owns the default implementation registry. It expects a normalized point reel and point
map, then materializes only the evidence consumed by current tracking, event, and 3D stages.
Validation code may orchestrate cohorts around it; no validation module implements these stages.

Player association is production default-on for ``keep_unique_admissible_frames``.
Rollback: ``--no-keep-unique-admissible-frames``, or set
``SelectionConfig.keep_unique_admissible_frames = False``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cv.pipeline.court_topology import (
    REGISTRATION_MASK_POLICIES,
    INTERPOLATED_CAMERA_POLICIES,
    SURFACE_WITNESS_POLICIES,
    resolve_surface_witness_policy,
)
from cv.pipeline.artifact_cache import (
    receipt_path,
    remove_receipt,
    stage_receipt_matches,
    write_stage_receipt,
)
from cv.pipeline.camera_artifacts import expand_point_cameras
from cv.pipeline.camera_bundle import bundle_match_cameras
from cv.pipeline.paths import data_root, require_paths
from cv.pipeline.provenance import AUTOMATIC_MODE, build_provenance, file_record

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
YOLO_WEIGHTS = data_root() / "models/pipeline/yolov8m.pt"
FRAMES_DIR = "audit_frames_native_1080"
REEL_NAME = "audit_reel_native_1080.mp4"
POINT_MAP_NAME = "audit_reel_point_map.csv"
CANONICAL_STAGE_REGISTRY = (
    ("source_cadence", "cv.pipeline.frame_cadence"),
    ("cadence_normalization", "cv.pipeline.cadence_normalize"),
    ("evidence_substrate", "cv.pipeline.run_contacts_pipeline"),
    ("court", "cv.pipeline.court_topology_runner"),
    ("camera", "cv.pipeline.camera_cal"),
    ("camera_expand", "cv.pipeline.camera_artifacts.expand_point_cameras"),
    ("player_side", "cv.pipeline.player_side_association"),
    ("tracking_composition", "cv.pipeline.tracking_composition"),
    ("event_features", "cv.pipeline.event_model_v2_features"),
    ("event_inference", "cv.pipeline.event_model_v3"),
    ("point_grammar", "cv.pipeline.point_grammar"),
    ("reconstruction_3d", "cv.pipeline.reconstruction"),
)
BASE_STAGE_REGISTRY = CANONICAL_STAGE_REGISTRY[:7]
RUNTIME_MODULE_DEPENDENCIES = (
    "cv.pipeline.run_contacts_pipeline",
    "cv.pipeline.court_topology_runner",
    "cv.pipeline.camera_cal",
    "cv.pipeline.player_side_association",
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _module_command(python: str, module: str, *arguments: str) -> list[str]:
    return [python, "-m", module, *arguments]


def _run_cached_command(
    *,
    match_out: Path,
    stage: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    upstream_receipts: list[Path],
) -> Path:
    if not stage_receipt_matches(
        out_dir=match_out,
        stage=stage,
        command=command,
        inputs=inputs,
        outputs=outputs,
        upstream_receipts=upstream_receipts,
    ):
        remove_receipt(match_out, stage)
        _run(command)
        return write_stage_receipt(
            out_dir=match_out,
            stage=stage,
            command=command,
            inputs=inputs,
            outputs=outputs,
            upstream_receipts=upstream_receipts,
        )
    return receipt_path(match_out, stage)


def _write_match_manifest(
    match: dict,
    match_out: Path,
    reel: Path,
    point_map: Path,
    camera: Path,
    profile: str,
    subpixel: str = "argmax",
    camera_model: str = "transport",
    court_jobs: int = 4,
    court_registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
    lost_revival_body_history: bool = False,
    overlap_fragment_recovery: bool = False,
    court_surface_witness: str = "off",
    keep_unique_admissible_frames: bool = True,
    propagate_shot_homography: bool = False,
) -> Path:
    point_camera = match_out / "camera_P_per_point.npz"
    fallbacks = []
    if point_camera.is_file():
        with np.load(point_camera, allow_pickle=True) as camera_data:
            sources = camera_data["source"] if "source" in camera_data.files else []
            ancestry = (
                camera_data["fallback_ancestry"]
                if "fallback_ancestry" in camera_data.files
                else ["[]"] * len(sources)
            )
            points = camera_data["pts"] if "pts" in camera_data.files else range(len(sources))
            fallbacks = [
                {
                    "stage": "camera_calibration",
                    "point": f"pt{int(point):04d}",
                    "type": str(source),
                    "ancestry": str(parentage),
                }
                for point, source, parentage in zip(points, sources, ancestry, strict=True)
                if str(source) != "direct"
            ]
    outputs = [
        match_out / "court_H_per_point.npz",
        match_out / "court_topology_evidence_v1.json",
        match_out / "camera_P_per_point.npz",
        camera,
        match_out / f"player_boxes_{float(match['source_fps']):g}_native_sided_v1.csv",
        match_out / "player_tracks_native_v1.csv",
        match_out / "player_tracks_native_v1.csv.coordinates.json",
    ]
    if camera_model == "bundle_v1":
        outputs.append(match_out / "camera_bundle_v1.json")
    provenance = build_provenance(
        root=PIPELINE_DIR.parents[1],
        mode=AUTOMATIC_MODE,
        source_videos=[file_record(reel, role="normalized_point_reel")],
        models=[file_record(YOLO_WEIGHTS, role="yolov8m_player_detector")],
        configuration={
            "runner": "cv.pipeline.canonical_runner",
            "match": match["id"],
            "source_fps": float(match["source_fps"]),
            "stages": list(BASE_STAGE_REGISTRY),
            "profile": profile,
            "ball_subpixel_decode": subpixel,
            "camera_model": camera_model,
            "court_jobs": court_jobs,
            "court_registration_mask": court_registration_mask,
            "court_surface_witness": court_surface_witness,
            "interpolated_camera_policy": interpolated_camera_policy,
            "lost_revival_body_history": lost_revival_body_history,
            "overlap_fragment_recovery": overlap_fragment_recovery,
            "keep_unique_admissible_frames": keep_unique_admissible_frames,
            "propagate_shot_homography": propagate_shot_homography,
        },
        reused_artifacts=[file_record(point_map, role="automatic_point_map")],
        fallbacks=fallbacks,
    )
    document = {
        "schema": "canonical_postseg_match_v1",
        "match": match["id"],
        "provenance": provenance,
        "outputs": [
            file_record(path, role="canonical_output") for path in outputs if path.is_file()
        ],
        "stage_receipts": [
            file_record(path, role="content_addressed_stage_receipt")
            for path in sorted((match_out / "run_manifests" / "stage_receipts").glob("*.json"))
        ],
    }
    path = match_out / "run_manifests" / "canonical_runner.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def resolve_court_jobs(profile: str, court_jobs: int | None = None) -> int:
    """Resolve explicit CPU scheduling without changing profile defaults."""
    if court_jobs is None:
        return 16 if profile == "motion" else 4
    if isinstance(court_jobs, bool) or not isinstance(court_jobs, int) or court_jobs < 1:
        raise ValueError("court_jobs must be a positive integer")
    return court_jobs


def run_match(
    match: dict,
    output_root: Path,
    device: int,
    profile: str = "postseg",
    subpixel: str = "argmax",
    camera_model: str = "transport",
    court_jobs: int | None = None,
    court_registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
    lost_revival_body_history: bool = False,
    overlap_fragment_recovery: bool = False,
    court_surface_witness: str = "off",
    keep_unique_admissible_frames: bool = True,
    propagate_shot_homography: bool = False,
) -> None:
    if profile not in {"postseg", "motion"}:
        raise ValueError(f"unsupported canonical profile {profile!r}")
    if subpixel not in {"argmax", "parabolic", "centroid"}:
        raise ValueError(f"unsupported ball sub-pixel decode {subpixel!r}")
    if camera_model not in {"transport", "bundle_v1"}:
        raise ValueError(f"unsupported camera model {camera_model!r}")
    if court_registration_mask not in REGISTRATION_MASK_POLICIES:
        raise ValueError(f"unknown court registration mask {court_registration_mask!r}")
    if interpolated_camera_policy not in INTERPOLATED_CAMERA_POLICIES:
        raise ValueError(f"unknown interpolated camera policy {interpolated_camera_policy!r}")
    court_surface_witness = resolve_surface_witness_policy(court_surface_witness)
    resolved_court_jobs = resolve_court_jobs(profile, court_jobs)
    match_out = output_root / match["id"]
    reel = match_out / REEL_NAME
    point_map = match_out / POINT_MAP_NAME
    fps = float(match["source_fps"])
    require_paths(
        {
            "normalized point reel": reel,
            "point map": point_map,
        }
    )
    python = sys.executable
    common = ["--out", os.fspath(match_out)]
    evidence_command = _module_command(
        python,
        "cv.pipeline.run_contacts_pipeline",
        *common,
        "--match",
        match["id"],
        "--video",
        os.fspath(reel),
        "--point-map",
        point_map.name,
        "--frames-dir",
        FRAMES_DIR,
        "--fps",
        str(fps),
        "--preserve-source-fps",
        "--ball-tag",
        "native_v1",
        "--contact-tag",
        "native_v1",
        "--gate-tag",
        "native_v1",
        "--audio-cache",
        "contact_audio_scores_16k_native_v1.npz",
        "--device",
        str(device),
        "--player-model",
        os.fspath(YOLO_WEIGHTS),
        "--subpixel",
        subpixel,
        "--resume",
    )
    evidence_command.append("--allow-partial-timing-holds")
    if profile == "motion":
        evidence_command.append("--skip-audio-evidence")
        player_evidence_command = [*evidence_command]
        evidence_command.append("--skip-player-detection")
    else:
        player_evidence_command = None
    _run(evidence_command)
    boxes = match_out / f"player_boxes_{fps:g}_native_v1.csv"
    frames = match_out / FRAMES_DIR
    frame_receipt = receipt_path(match_out, "frame_extraction")
    player_receipt = receipt_path(match_out, "player_detection")
    court_output = match_out / "court_H_per_point.npz"
    court_frame_track = match_out / "court_H_per_frame_v1.npz"
    court_evidence = match_out / "court_topology_evidence_v1.json"
    court_command = _module_command(
        python,
        "cv.pipeline.court_topology_runner",
        *common,
        "--frames-dir",
        FRAMES_DIR,
    )
    if profile == "motion" or court_jobs is not None:
        court_command.extend(["--jobs", str(resolved_court_jobs)])
    if court_surface_witness != "off":
        court_command.extend(["--court-surface-witness", court_surface_witness])
    if court_registration_mask != "off":
        court_command.extend(["--court-registration-mask", court_registration_mask])
    if interpolated_camera_policy != "off":
        court_command.extend(["--interpolated-camera-policy", interpolated_camera_policy])
    court_arguments = {
        "match_out": match_out,
        "stage": "court_geometry",
        "command": court_command,
        "inputs": [frames],
        "outputs": [court_output, court_frame_track, court_evidence],
        "upstream_receipts": [frame_receipt],
    }
    if player_evidence_command is None:
        court_receipt = _run_cached_command(**court_arguments)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            player_future = pool.submit(_run, player_evidence_command)
            court_future = pool.submit(_run_cached_command, **court_arguments)
            player_future.result()
            court_receipt = court_future.result()
    camera_output = match_out / "camera_P_per_point.npz"
    camera_command = _module_command(
        python,
        "cv.pipeline.camera_cal",
        *common,
        "--frames-dir",
        FRAMES_DIR,
        "--overlay",
    )
    camera_receipt = _run_cached_command(
        match_out=match_out,
        stage="camera_calibration",
        command=camera_command,
        inputs=[frames, court_output, court_evidence],
        outputs=[camera_output],
        upstream_receipts=[court_receipt],
    )
    camera = match_out / "camera_P_per_frame_v1.npz"
    expand_command = (
        [python, "-m", "cv.pipeline.camera_bundle", "--out", os.fspath(match_out)]
        if camera_model == "bundle_v1"
        else [
            python,
            os.fspath(PIPELINE_DIR / "camera_artifacts.py"),
            "expand_point_cameras",
        ]
    )
    expand_outputs = [camera]
    if camera_model == "bundle_v1":
        expand_outputs.append(match_out / "camera_bundle_v1.json")
    if not stage_receipt_matches(
        out_dir=match_out,
        stage="camera_expand",
        command=expand_command,
        inputs=[frames, camera_output, court_frame_track],
        outputs=expand_outputs,
        upstream_receipts=[camera_receipt],
    ):
        remove_receipt(match_out, "camera_expand")
        camera = (
            bundle_match_cameras(match_out, FRAMES_DIR)
            if camera_model == "bundle_v1"
            else expand_point_cameras(match_out, FRAMES_DIR)
        )
        write_stage_receipt(
            out_dir=match_out,
            stage="camera_expand",
            command=expand_command,
            inputs=[frames, camera_output, court_frame_track],
            outputs=expand_outputs,
            upstream_receipts=[camera_receipt],
        )
    with np.load(camera) as camera_data:
        if not len(camera_data["clips"]):
            _write_match_manifest(
                match,
                match_out,
                reel,
                point_map,
                camera,
                profile,
                subpixel,
                camera_model,
                lost_revival_body_history=lost_revival_body_history,
                overlap_fragment_recovery=overlap_fragment_recovery,
                keep_unique_admissible_frames=keep_unique_admissible_frames,
                propagate_shot_homography=propagate_shot_homography,
            )
            return
    sided_output = match_out / f"player_boxes_{fps:g}_native_sided_v1.csv"
    tracks_output = match_out / "player_tracks_native_v1.csv"
    sided_command = _module_command(
        python,
        "cv.pipeline.player_side_association",
        "--boxes",
        os.fspath(boxes),
        "--court",
        os.fspath(court_output),
        "--fps",
        str(fps),
        "--output",
        os.fspath(sided_output),
        "--tracks-output",
        os.fspath(tracks_output),
        "--court-frame-track",
        os.fspath(court_frame_track),
        *(["--lost-revival-body-history"] if lost_revival_body_history else []),
        *(["--overlap-fragment-recovery"] if overlap_fragment_recovery else []),
        *(
            ["--keep-unique-admissible-frames"]
            if keep_unique_admissible_frames
            else ["--no-keep-unique-admissible-frames"]
        ),
        *(
            [
                "--propagate-shot-homography",
                "--frames-root",
                os.fspath(frames),
            ]
            if propagate_shot_homography
            else ["--no-propagate-shot-homography"]
        ),
    )
    _run_cached_command(
        match_out=match_out,
        stage="player_side_association",
        command=sided_command,
        inputs=[
            boxes,
            boxes.with_suffix(boxes.suffix + ".coordinates.json"),
            court_output,
            court_frame_track,
        ],
        outputs=[
            sided_output,
            sided_output.with_suffix(sided_output.suffix + ".coordinates.json"),
            tracks_output,
            tracks_output.with_suffix(tracks_output.suffix + ".coordinates.json"),
        ],
        upstream_receipts=[player_receipt, court_receipt],
    )
    _write_match_manifest(
        match,
        match_out,
        reel,
        point_map,
        camera,
        profile,
        subpixel,
        camera_model,
        resolved_court_jobs,
        court_registration_mask,
        interpolated_camera_policy,
        lost_revival_body_history,
        overlap_fragment_recovery,
        court_surface_witness,
        keep_unique_admissible_frames,
        propagate_shot_homography,
    )


def run_manifest(
    manifest: dict,
    output_root: Path,
    device: int,
    profile: str = "postseg",
    subpixel: str = "argmax",
    camera_model: str = "transport",
    court_jobs: int | None = None,
    court_registration_mask: str = "off",
    interpolated_camera_policy: str = "off",
    lost_revival_body_history: bool = False,
    overlap_fragment_recovery: bool = False,
    court_surface_witness: str = "off",
    keep_unique_admissible_frames: bool = True,
    propagate_shot_homography: bool = False,
) -> None:
    for match in manifest["matches"]:
        print(f"[canonical] {match['id']}", flush=True)
        run_match(
            match,
            output_root,
            device,
            profile,
            subpixel,
            camera_model,
            court_jobs,
            court_registration_mask,
            interpolated_camera_policy,
            lost_revival_body_history,
            overlap_fragment_recovery,
            court_surface_witness,
            keep_unique_admissible_frames,
            propagate_shot_homography,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--court-jobs",
        type=int,
        help="court CPU workers; defaults stay 4 for postseg and 16 for motion",
    )
    parser.add_argument("--profile", choices=("postseg", "motion"), default="postseg")
    parser.add_argument(
        "--subpixel",
        choices=("argmax", "parabolic", "centroid"),
        default="argmax",
        help="named classified alternative for the ball heatmap decode; 'argmax' is the "
        "production default (byte-identical), 'centroid'/'parabolic' refine each peak to "
        "sub-cell position. Selected explicitly here, never through an ambient file.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("transport", "bundle_v1"),
        default="transport",
        help="camera expansion; bundle_v1 fits one fixed optical centre per match",
    )
    parser.add_argument(
        "--court-surface-witness",
        choices=SURFACE_WITNESS_POLICIES,
        default="off",
        help="illumination-aware court surface fallback after existing admission fails",
    )
    parser.add_argument(
        "--court-registration-mask", choices=REGISTRATION_MASK_POLICIES, default="off"
    )
    parser.add_argument(
        "--interpolated-camera-policy", choices=INTERPOLATED_CAMERA_POLICIES, default="off"
    )
    parser.add_argument(
        "--lost-revival-body-history",
        action="store_true",
        help="default-off lost-track-revival body-history guard in player association",
    )
    parser.add_argument(
        "--overlap-fragment-recovery",
        action="store_true",
        help="default-off measured duplicate-track fragment recovery in player association",
    )
    parser.add_argument(
        "--keep-unique-admissible-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="production default-on: keep unique frames of other admissible same-side "
        "player tracks after ByteTrack winner-take-all. Rollback: "
        "--no-keep-unique-admissible-frames.",
    )
    parser.add_argument(
        "--propagate-shot-homography",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="default-off: player association uses the nearest reliable court "
        "homography of the same camera shot on unreliable frames. A close-up "
        "is not painted. Rollback: omit or --no-propagate-shot-homography.",
    )
    args = parser.parse_args()
    if args.court_jobs is not None and args.court_jobs < 1:
        parser.error("--court-jobs must be positive")
    run_manifest(
        json.loads(args.manifest.read_text()),
        args.out,
        args.device,
        args.profile,
        args.subpixel,
        args.camera_model,
        args.court_jobs,
        args.court_registration_mask,
        args.interpolated_camera_policy,
        args.lost_revival_body_history,
        args.overlap_fragment_recovery,
        args.court_surface_witness,
        args.keep_unique_admissible_frames,
        args.propagate_shot_homography,
    )


if __name__ == "__main__":
    main()
