"""Compose full-frame, local-crop, consensus, and repair ball trackers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cv.pipeline.paths import data_root, require_paths, tracker_root
from cv.pipeline.resolution import resolve_player_boxes
from cv.pipeline.torso_lock import filter_candidate_streams, restore_lock_gaps_file

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = tracker_root()
# Subprocesses are not Python imports. Receipts follow this literal declaration and their
# transitive imports; the command-coverage test keeps the launch surface and identity together.
RUNTIME_MODULE_DEPENDENCIES = (
    "cv.pipeline.ball_neural",
    "cv.pipeline.ball_neural_batched",
    "cv.pipeline.ball_local_refine",
    "cv.pipeline.ball_local_refine_batched",
    "cv.pipeline.ball_motion_tracker",
    "cv.pipeline.guide_sequence_association",
    "cv.pipeline.ball_anchor_extension",
    "cv.pipeline.ball_track_consensus",
    "cv.pipeline.ball_track_detour_repair",
    "cv.pipeline.ball_track_local_augment",
)
MOTION_TRACKER_PATH = "cv/pipeline/ball_motion_tracker.py"
# Re-pinned for the default-off two-pass guide-detour sequence association: the extracted
# prepare_observations helper, the extracted advance_modes predictor (the single IMM
# mix-and-predict step both association passes now call), track_clip's optional per-frame
# ``admission`` pool -- honoured on the reentry, bootstrap/restart and steady-state paths
# alike -- and the run_tracker branch onto cv.pipeline.guide_sequence_association. Every
# addition is gated on MotionConfig.guide_sequence_association (and, before it,
# primary_ball_ownership), both of which default to False; with the flags off the emitted
# rows and proposals are identical to the previous pin
# (test_ball_ownership.py::test_default_off_leaves_tracker_rows_untouched,
# test_guide_sequence_association.py::test_off_is_exact_tracker_parity).
# Re-pinned 2026-09-27 for torso-lock rejection (MotionConfig.torso_lock_rejection) and the
# default-off lock_gap_carry. With --torso-lock-boxes absent the pool and rows are unchanged.
# Re-pinned 2026-09-28 for the default-off --torso-lock-keep-joined
# (MotionConfig.torso_lock_keep_joined); without it the torso filter is unchanged.
MOTION_TRACKER_SHA256 = "c927ad090238524fe22e4bf848d8454d143b89cc4a66edf36b0349385145d080"
# Artifacts of the two default-off ownership arms, named once so the launch surface, the
# reuse check and the tests cannot drift from ball_motion_tracker.ownership_artifact_path.
OWNERSHIP_ARTIFACT_NAME = "ball_track_joint_native1080_ownership_v1.csv"
PRE_ANCHOR_EXTENSION_NAME = "ball_track_joint_native1080_pre_anchor_extension_v1.csv"
ANCHOR_EXTENSION_SIDE_NAME = "ball_anchor_extension_v1.csv"
ANCHOR_EXTENSION_REPORT_NAME = "ball_anchor_extension_v1.json"
# Torso-lock ball track, default on 2026-09-27. A candidate chained inside one player's torso
# for six frames (the Halle neon shirt) is dropped from every candidate stream, the coarse lock
# is re-decoded from the filtered streams, and the motion tracker refuses torso chains too.
# Same-wave fits: fresh AL 162 -> 172 and AA 100 -> 101 on the changed attempts, panel C AL
# +1/-1, no extra accept (cv/experiments/ceiling_track_camera/FINAL_REPORT_ceiling_track_camera.md).
# The published candidate streams are kept; the filtered copies live in TORSO_LOCK_INPUTS_NAME.
# Rollback: set False (or pass --no-torso-lock).
TORSO_LOCK_BALL_TRACK = True
# Keep a small-torso chain joined at both ends to outside candidates at ball speed
# (torso_lock.KEEP_JOINED: a far player's real ball at contact). Default on 2026-09-29 together
# with TORSO_LOCK_GAP_RESTORE: same-wave fits on the changed attempts (fresh 0/A, dev, C, D, E)
# AL +7/-2, AA +8/-3 with regenerated events, no extra accept, Halle dense wrong rows 0 -> 0
# (cv/experiments/streak_centre/FINAL_REPORT_streak_centre.md).
# Rollback: set False (or pass --no-torso-lock-keep-joined).
TORSO_LOCK_KEEP_JOINED = True
TORSO_LOCK_INPUTS_NAME = "torso_lock_inputs_v1"
# Fill the torso-lock track's bracketed gaps from the same tracker run on the unfiltered pool
# (torso_lock.restore_lock_gaps: the real ball the stream filter dropped at a bounce or
# contact; near-player runs must continue the unfiltered path). Default on 2026-09-29 (see
# TORSO_LOCK_KEEP_JOINED). Rollback: set False (or pass --no-torso-lock-gap-restore).
TORSO_LOCK_GAP_RESTORE = True
TORSO_UNFILTERED_TRACK_NAME = "ball_track_joint_native1080_unfiltered_v1.csv"
TORSO_PRE_RESTORE_NAME = "ball_track_joint_native1080_torso_lock_pre_restore_v1.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_code(protocol: dict) -> dict[str, str]:
    expected = protocol["frozen_tracker"]["code_sha256"]
    observed = {relative: file_sha256(ROOT / relative) for relative in expected}
    mismatches = {
        relative: {"expected": expected[relative], "observed": digest}
        for relative, digest in observed.items()
        if digest != expected[relative]
    }
    if mismatches:
        raise RuntimeError(f"frozen tracker code changed: {json.dumps(mismatches)}")
    motion_digest = file_sha256(ROOT / MOTION_TRACKER_PATH)
    if motion_digest != MOTION_TRACKER_SHA256:
        raise RuntimeError(
            "motion tracker changed without updating its frozen transitive hash: "
            f"expected={MOTION_TRACKER_SHA256} observed={motion_digest}"
        )
    observed[MOTION_TRACKER_PATH] = motion_digest
    return observed


def verify_frozen_inputs(protocol: dict, manifest_path: Path, out_dir: Path) -> dict[str, str]:
    observed = {
        "manifest": file_sha256(manifest_path),
        "sample": file_sha256(out_dir / "audit_sample.csv"),
    }
    expected = {
        "manifest": protocol["frozen_manifest_sha256"],
        "sample": protocol["frozen_sample_sha256"],
    }
    if observed != expected:
        raise RuntimeError(f"frozen input changed: expected={expected}, observed={observed}")
    return observed


# Detectors fine-tuned on native 512x288 crops taught by the motion tracker's own retained
# arcs (cv/experiments/tracknet_finetune/build_tracker_crops.py). Measured on the sealed
# transfer split, decoded as the composed lock: 41/47/47 windows against the production
# checkpoints' 21/27/30 in the same composition. They are the crop pass default; the far-native
# tiling pass keeps the production checkpoints, where they are measurably better.
CROP_FINETUNE_TEMPLATE = "models/pipeline/ball_finetune_v1/{model}_native_crop_ep3.pth.tar"


def production_weights(model: str) -> Path:
    return EXTERNAL_ROOT / "pretrained_weights" / f"{model}_tennis_best.pth.tar"


def resolve_weights(template: str | None, model: str) -> Path:
    """Detector weights for one pass.

    ``template`` may contain ``{model}``, so a single flag selects the matching checkpoint for
    both the WASB and the TrackNetV2 arm; a relative template resolves under the data root. The
    crop pass and the far-native tiling pass each take their own template because a checkpoint
    fine-tuned on native crops is only a drop-in for the regime it was trained on.
    """
    if not template:
        return production_weights(model)
    path = Path(template.format(model=model)).expanduser()
    if not path.is_absolute():
        path = data_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"{model} weights not found: {path}")
    return path


def tracking_model_inputs(
    crop_weights: str | None = CROP_FINETUNE_TEMPLATE,
    far_native_weights: str | None = None,
    *,
    far_native: bool = True,
) -> list[Path]:
    """Actual checkpoints and external detector implementation/configuration for receipts.

    The external model package is loaded through sys.path, not repository imports. Include its
    source/configuration files, never mutable __pycache__ contents or unrelated training outputs.
    """
    selected = {production_weights(model) for model in ("wasb", "tracknetv2")}
    selected.update(resolve_weights(crop_weights, model) for model in ("wasb", "tracknetv2"))
    if far_native:
        selected.update(
            resolve_weights(far_native_weights, model) for model in ("wasb", "tracknetv2")
        )
    source = EXTERNAL_ROOT / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"external tracker source is missing: {source}")
    selected.update(path for suffix in ("*.py", "*.yaml", "*.yml") for path in source.rglob(suffix))
    return sorted(selected)


def run(command: list[str]) -> float:
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.monotonic() - started


def module_command(module: str) -> list[str]:
    return [sys.executable, "-m", f"cv.pipeline.{module}"]


def decode(
    match_out: Path,
    fps: float,
    consumer: str,
    candidates: list[tuple[str, str]],
    output: str,
) -> float:
    command = module_command("ball_track_consensus")
    for filename, source in candidates:
        command.extend(["--candidates", str(match_out / filename), "--source", source])
    command.extend(
        [
            "--output",
            str(match_out / output),
            "--fps",
            str(fps),
            "--consumer",
            consumer,
        ]
    )
    return run(command)


def run_match(
    match: dict,
    out_dir: Path,
    device: int,
    full_batch: int = 32,
    local_batch: int = 32,
    optimized_local_runtime: bool = False,
    optimized_full_runtime: bool = False,
    subpixel: str = "centroid",
    far_native: bool = True,
    crop_authoritative: bool = False,
    crop_authority_px1080: float = 4.0,
    detour_repair: bool = True,
    motion_tracker: bool = True,
    crop_weights: str | None = CROP_FINETUNE_TEMPLATE,
    far_native_weights: str | None = None,
    crop_first: bool = False,
    crop_lock_stream: bool = True,
    primary_ball_ownership: bool = False,
    anchor_extension: bool = False,
    guide_sequence_association: bool = False,
    torso_lock: bool = TORSO_LOCK_BALL_TRACK,
    torso_lock_keep_joined: bool = TORSO_LOCK_KEEP_JOINED,
    torso_lock_gap_restore: bool = TORSO_LOCK_GAP_RESTORE,
) -> dict:
    # Fail before any detector work: both options only exist inside the motion-tracker
    # composition, and the extension reads the ownership artifact the tracker writes.
    if anchor_extension and not primary_ball_ownership:
        raise ValueError("anchor_extension requires primary_ball_ownership")
    if (primary_ball_ownership or anchor_extension) and not motion_tracker:
        raise ValueError(
            "primary_ball_ownership and anchor_extension require the motion tracker stage"
        )
    if guide_sequence_association and not motion_tracker:
        raise ValueError("guide_sequence_association requires the motion tracker stage")
    torso_lock = torso_lock and motion_tracker
    match_out = out_dir / match["id"]
    frames_dir = match_out / "audit_frames_native_1080"
    require_paths(
        {
            "native frames": frames_dir,
            "WASB weights": production_weights("wasb"),
            "TrackNetV2 weights": production_weights("tracknetv2"),
            **{
                f"{model} crop weights": resolve_weights(crop_weights, model)
                for model in ("wasb", "tracknetv2")
            },
        }
    )
    fps = float(match["source_fps"])
    elapsed = {}
    for model in ("wasb", "tracknetv2"):
        elapsed[f"full_{model}"] = run(
            [
                *module_command("ball_neural_batched" if optimized_full_runtime else "ball_neural"),
                "--out",
                str(match_out),
                "--frames-dir",
                frames_dir.name,
                "--model",
                model,
                "--device",
                str(device),
                "--fps",
                str(fps),
                "--k-best",
                "5",
                "--peak-threshold",
                "0.05",
                "--nms-radius",
                "4",
                "--temporal-ensemble",
                "--subpixel",
                subpixel,
                "--batch",
                str(full_batch),
                "--output-tag",
                "native1080_sliding_k5_v1",
            ]
        )
    full_candidates = [
        ("ball_candidates_wasb_native1080_sliding_k5_v1.csv", "sliding_wasb"),
        (
            "ball_candidates_tracknetv2_native1080_sliding_k5_v1.csv",
            "sliding_tracknetv2",
        ),
    ]
    if far_native:
        # Native far-half-court tiling: two overlapping tiles cut from the 1920x1080 frame
        # at 2x the effective magnification of the full-frame pass. Far coverage is the
        # binding constraint, so these candidates join the same decoders as the sliding
        # pass rather than living in a side artifact nothing reads.
        for model in ("wasb", "tracknetv2"):
            elapsed[f"far_native_{model}"] = run(
                [
                    *module_command(
                        "ball_neural_batched" if optimized_full_runtime else "ball_neural"
                    ),
                    "--out",
                    str(match_out),
                    "--frames-dir",
                    frames_dir.name,
                    "--model",
                    model,
                    "--device",
                    str(device),
                    "--fps",
                    str(fps),
                    "--k-best",
                    "5",
                    "--peak-threshold",
                    "0.05",
                    "--nms-radius",
                    "4",
                    "--subpixel",
                    subpixel,
                    "--batch",
                    str(full_batch),
                    "--weights",
                    str(resolve_weights(far_native_weights, model)),
                    "--far-native",
                    "--native-frames-dir",
                    frames_dir.name,
                    "--camera-npz",
                    "camera_P_per_point.npz",
                    "--output-tag",
                    "native1080_far_native_v1",
                ]
            )
        full_candidates.extend(
            [
                ("ball_candidates_wasb_native1080_far_native_v1.csv", "far_native_wasb"),
                (
                    "ball_candidates_tracknetv2_native1080_far_native_v1.csv",
                    "far_native_tracknetv2",
                ),
            ]
        )
    for consumer in ("integrity", "availability"):
        elapsed[f"decode_wasb_{consumer}"] = decode(
            match_out,
            fps,
            consumer,
            [full_candidates[0]],
            f"ball_track_wasb_native1080_{consumer}_v1.csv",
        )
        elapsed[f"decode_tracknetv2_{consumer}"] = decode(
            match_out,
            fps,
            consumer,
            [full_candidates[1]],
            f"ball_track_tracknetv2_native1080_{consumer}_v1.csv",
        )
        elapsed[f"decode_joint_{consumer}"] = decode(
            match_out,
            fps,
            consumer,
            full_candidates,
            f"ball_track_joint_native1080_{consumer}_v1.csv",
        )
    player_boxes = resolve_player_boxes(match_out, sided=False)
    for model in ("wasb", "tracknetv2"):
        elapsed[f"crop_{model}"] = run(
            [
                *module_command(
                    "ball_local_refine_batched" if optimized_local_runtime else "ball_local_refine"
                ),
                "--frames-dir",
                str(frames_dir),
                "--lock-track",
                str(match_out / "ball_track_joint_native1080_integrity_v1.csv"),
                "--player-boxes",
                str(player_boxes),
                "--external-root",
                str(EXTERNAL_ROOT),
                "--weights",
                str(resolve_weights(crop_weights, model)),
                "--model",
                model,
                "--device",
                str(device),
                "--persistent-lock",
                "--fps",
                str(fps),
                "--native-crop-width",
                "512",
                "--native-crop-height",
                "288",
                "--maximum-expansion",
                "2.0",
                "--maximum-extrapolation-seconds",
                "0.6",
                "--branch-disagreement-px",
                "45.0",
                "--subpixel",
                subpixel,
                "--batch",
                str(local_batch),
                "--output",
                str(match_out / f"ball_candidates_{model}_native1080_branched_crop_v2.csv"),
            ]
        )
    crop_candidates = [
        ("ball_candidates_wasb_native1080_branched_crop_v2.csv", "branched_crop_wasb"),
        (
            "ball_candidates_tracknetv2_native1080_branched_crop_v2.csv",
            "branched_crop_tracknetv2",
        ),
    ]
    for consumer in ("integrity", "availability"):
        elapsed[f"decode_crop_{consumer}"] = decode(
            match_out,
            fps,
            consumer,
            crop_candidates,
            f"ball_track_joint_native1080_branched_crop_{consumer}_v2.csv",
        )
        elapsed[f"decode_full_crop_{consumer}"] = decode(
            match_out,
            fps,
            consumer,
            [*full_candidates, *crop_candidates],
            f"ball_track_joint_native1080_full_plus_branched_crop_{consumer}_v2.csv",
        )
    all_candidate_paths = [
        str(match_out / filename) for filename, _ in [*full_candidates, *crop_candidates]
    ]
    restored_rows = None
    if motion_tracker:
        # Crop-first composition: the native-crop decode, not the full-frame one, is the lock
        # stream the IMM follows. Measured on the ball cohort with the fine-tuned crop weights:
        # development 35/38/46 -> 37/45/48 and sealed transfer 37/44/47 -> 41/47/47, with the
        # 10 missing development estimates going to 0. With the *production* crop checkpoints
        # the same composition collapses to 22/26/35 and 21/27/30, so the two changes only work
        # together; --full-frame-lock-stream restores the old lock.
        lock_stream = (
            "ball_track_joint_native1080_branched_crop_integrity_v2.csv"
            if crop_lock_stream
            else "ball_track_joint_native1080_integrity_v1.csv"
        )
        stream_root = match_out
        torso_boxes = None
        if torso_lock:
            # Filter every stream on the pooled torso mask, then re-decode the lock from the
            # filtered streams, so the lock the IMM follows is not the shirt either.
            torso_boxes = resolve_player_boxes(match_out, sided=True)
            stream_root = match_out / TORSO_LOCK_INPUTS_NAME
            started = time.monotonic()
            filter_candidate_streams(
                match_out,
                [filename for filename, _ in [*full_candidates, *crop_candidates]],
                torso_boxes,
                stream_root,
                keep_joined=torso_lock_keep_joined,
            )
            elapsed["torso_lock_filter"] = time.monotonic() - started
            elapsed["torso_lock_relock"] = decode(
                stream_root,
                fps,
                "integrity",
                crop_candidates if crop_lock_stream else full_candidates,
                lock_stream,
            )
        motion_candidates = [
            (lock_stream, "coarse_lock"),
            *full_candidates,
            *crop_candidates,
        ]

        def motion_command_for(root: Path, output: Path, proposals: Path, torso: bool):
            command = module_command("ball_motion_tracker")
            for filename, source in motion_candidates:
                command.extend(["--candidates", str(root / filename), "--source", source])
            command.extend(
                [
                    "--output",
                    str(output),
                    "--event-proposals",
                    str(proposals),
                    "--fps",
                    str(fps),
                    "--court-homographies",
                    str(match_out / "court_H_per_frame_v1.npz"),
                    "--camera-projections",
                    str(match_out / "camera_P_per_frame_v1.npz"),
                ]
            )
            if crop_first:
                command.append("--crop-first")
                if crop_lock_stream:
                    command.append("--lock-is-crop")
            if primary_ball_ownership:
                command.append("--primary-ball-ownership")
            if guide_sequence_association:
                command.append("--guide-sequence-association")
            if torso and torso_boxes is not None:
                command.extend(["--torso-lock-boxes", str(torso_boxes)])
                if torso_lock_keep_joined:
                    command.append("--torso-lock-keep-joined")
            return command

        consumed_track = match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"
        motion_command = motion_command_for(
            stream_root, consumed_track, match_out / "ball_motion_event_proposals_v1.json", True
        )
        elapsed["motion_tracker"] = run(motion_command)
        if torso_boxes is not None and torso_lock_gap_restore:
            # The same tracker on the unfiltered pool; its rows fill only the torso-lock
            # track's bracketed gaps (torso_lock.restore_lock_gaps). The pre-restore track
            # is kept next to the consumed one.
            unfiltered = match_out / TORSO_LOCK_INPUTS_NAME / TORSO_UNFILTERED_TRACK_NAME
            elapsed["torso_lock_unfiltered_tracker"] = run(
                motion_command_for(
                    match_out,
                    unfiltered,
                    unfiltered.with_name("ball_motion_event_proposals_unfiltered_v1.json"),
                    False,
                )
            )
            restored_rows = restore_torso_lock_gaps(
                consumed_track, match_out / TORSO_PRE_RESTORE_NAME, unfiltered, torso_boxes
            )
        if anchor_extension:
            # Source-only anchor extension (cv.pipeline.ball_anchor_extension) over the same
            # explicit candidate arms. The pre-extension consumed track is preserved and the
            # extended rows replace the consumed artifact so the downstream stage reads them.
            consumed_path = match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"
            pre_extension = match_out / PRE_ANCHOR_EXTENSION_NAME
            shutil.copyfile(consumed_path, pre_extension)
            shutil.copyfile(
                consumed_path.with_name(consumed_path.name + ".coordinates.json"),
                pre_extension.with_name(pre_extension.name + ".coordinates.json"),
            )
            extension_command = [
                *module_command("ball_anchor_extension"),
                "--track",
                str(pre_extension),
                "--ownership",
                str(match_out / OWNERSHIP_ARTIFACT_NAME),
                "--output",
                str(consumed_path),
                "--side-output",
                str(match_out / ANCHOR_EXTENSION_SIDE_NAME),
                "--report",
                str(match_out / ANCHOR_EXTENSION_REPORT_NAME),
                "--fps",
                str(fps),
                "--court-homographies",
                str(match_out / "court_H_per_frame_v1.npz"),
                "--camera-projections",
                str(match_out / "camera_P_per_frame_v1.npz"),
            ]
            for filename, source in motion_candidates:
                extension_command.extend(
                    ["--candidates", str(stream_root / filename), "--source", source]
                )
            if crop_first:
                extension_command.append("--crop-first")
                if crop_lock_stream:
                    extension_command.append("--lock-is-crop")
            elapsed["anchor_extension"] = run(extension_command)
    else:
        detour_command = [
            *module_command("ball_track_detour_repair"),
            "--base",
            str(match_out / "ball_track_joint_native1080_integrity_v1.csv"),
        ]
        for path in all_candidate_paths:
            detour_command.extend(["--candidates", path])
        detour_command.extend(
            [
                "--output",
                str(match_out / "ball_track_joint_native1080_detour_repaired_v2.csv"),
                "--fps",
                str(fps),
            ]
        )
        if not detour_repair:
            detour_command.append("--passthrough")
        elapsed["detour_repair"] = run(detour_command)
        augment_command = [
            *module_command("ball_track_local_augment"),
            "--base",
            str(match_out / "ball_track_joint_native1080_detour_repaired_v2.csv"),
        ]
        for path in all_candidate_paths:
            augment_command.extend(["--candidates", path])
        augment_command.extend(
            [
                "--output",
                str(match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"),
                "--fps",
                str(fps),
                "--minimum-run",
                "2",
                "--maximum-rmse",
                "6.0",
                "--context-seconds",
                "0.2",
            ]
        )
        if crop_authoritative:
            augment_command.extend(
                [
                    "--crop-authoritative",
                    "--crop-authority-px1080",
                    str(crop_authority_px1080),
                ]
            )
        elapsed["arc_fill"] = run(augment_command)
    require_paths(
        {
            "camera": match_out / "camera_P_per_frame_v1.npz",
            "audio evidence": match_out / "contact_audio_scores_16k_native_v1.npz",
            "composed ball track": match_out / "ball_track_joint_native1080_arc_augmented_v2.csv",
        }
    )
    return {
        "match_id": match["id"],
        "fps": fps,
        "device": device,
        "full_batch": full_batch,
        "local_batch": local_batch,
        "optimized_local_runtime": optimized_local_runtime,
        "optimized_full_runtime": optimized_full_runtime,
        "subpixel": subpixel,
        "far_native": far_native,
        "crop_authoritative": crop_authoritative,
        "crop_authority_px1080": crop_authority_px1080,
        "detour_repair": detour_repair,
        "motion_tracker": motion_tracker,
        "crop_weights": {
            model: str(resolve_weights(crop_weights, model)) for model in ("wasb", "tracknetv2")
        },
        "far_native_weights": {
            model: str(resolve_weights(far_native_weights, model))
            for model in ("wasb", "tracknetv2")
        },
        "crop_first": crop_first,
        "crop_lock_stream": crop_lock_stream,
        "primary_ball_ownership": primary_ball_ownership,
        "anchor_extension": anchor_extension,
        "guide_sequence_association": guide_sequence_association,
        "torso_lock": torso_lock,
        **({"torso_lock_keep_joined": True} if torso_lock and torso_lock_keep_joined else {}),
        **(
            {"torso_lock_gap_restore": True, "torso_lock_restored_rows": restored_rows}
            if restored_rows is not None
            else {}
        ),
        "elapsed_seconds": elapsed,
        "total_elapsed_seconds": sum(elapsed.values()),
    }


def restore_torso_lock_gaps(
    consumed_track: Path, pre_restore: Path, unfiltered: Path, torso_boxes: Path
) -> int:
    """Keep the torso-lock track as ``pre_restore`` and fill its bracketed gaps in place."""
    shutil.copyfile(consumed_track, pre_restore)
    shutil.copyfile(
        consumed_track.with_name(consumed_track.name + ".coordinates.json"),
        pre_restore.with_name(pre_restore.name + ".coordinates.json"),
    )
    return restore_lock_gaps_file(pre_restore, unfiltered, consumed_track, torso_boxes)


def completed_match_result(
    match: dict,
    out_dir: Path,
    *,
    primary_ball_ownership: bool = False,
    anchor_extension: bool = False,
) -> dict | None:
    """A prior complete output for this match, or None when it must be produced again.

    The optional arms name their own side artifacts here. The consumed track alone cannot
    say which arm wrote it, so without this an ownership-off directory would be reused
    verbatim for an ownership-on request and the toggle would silently do nothing.
    """
    match_out = out_dir / match["id"]
    track = match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"
    coordinates = track.with_suffix(track.suffix + ".coordinates.json")
    required = (
        track,
        coordinates,
        match_out / "camera_P_per_frame_v1.npz",
        match_out / "contact_audio_scores_16k_native_v1.npz",
        *(
            [match_out / OWNERSHIP_ARTIFACT_NAME]
            if primary_ball_ownership or anchor_extension
            else []
        ),
        *([match_out / ANCHOR_EXTENSION_REPORT_NAME] if anchor_extension else []),
    )
    if not all(path.is_file() for path in required):
        return None
    return {
        "match_id": match["id"],
        "fps": float(match["source_fps"]),
        "reused_complete": True,
        "track_sha256": file_sha256(track),
        "coordinates_sha256": file_sha256(coordinates),
    }


def frame_count(out_dir: Path, match: dict) -> int:
    frames_dir = out_dir / match["id"] / "audit_frames_native_1080"
    return sum(1 for _ in frames_dir.glob("pt*/f_*.jpg"))


def balanced_lanes(matches: list[dict], out_dir: Path, devices: list[int]) -> list[list[dict]]:
    lanes = [[] for _ in devices]
    loads = [0 for _ in devices]
    weighted = sorted(
        ((frame_count(out_dir, match), match) for match in matches),
        key=lambda item: (-item[0], item[1]["id"]),
    )
    for weight, match in weighted:
        lane = min(range(len(devices)), key=lambda index: loads[index])
        lanes[lane].append(match)
        loads[lane] += weight
    return lanes


def run_lane(
    matches: list[dict],
    out_dir: Path,
    device: int,
    full_batch: int,
    local_batch: int,
    optimized_local_runtime: bool,
    optimized_full_runtime: bool,
    subpixel: str = "centroid",
    far_native: bool = True,
    crop_authoritative: bool = False,
    crop_authority_px1080: float = 4.0,
    detour_repair: bool = True,
    motion_tracker: bool = True,
    crop_weights: str | None = CROP_FINETUNE_TEMPLATE,
    far_native_weights: str | None = None,
    crop_first: bool = False,
    crop_lock_stream: bool = True,
    primary_ball_ownership: bool = False,
    anchor_extension: bool = False,
    guide_sequence_association: bool = False,
    torso_lock: bool = TORSO_LOCK_BALL_TRACK,
    torso_lock_keep_joined: bool = TORSO_LOCK_KEEP_JOINED,
    torso_lock_gap_restore: bool = TORSO_LOCK_GAP_RESTORE,
) -> list[dict]:
    return [
        run_match(
            match,
            out_dir,
            device,
            full_batch,
            local_batch,
            optimized_local_runtime,
            optimized_full_runtime,
            subpixel,
            far_native,
            crop_authoritative,
            crop_authority_px1080,
            detour_repair,
            motion_tracker,
            crop_weights,
            far_native_weights,
            crop_first,
            crop_lock_stream,
            primary_ball_ownership,
            anchor_extension,
            guide_sequence_association,
            torso_lock,
            torso_lock_keep_joined,
            torso_lock_gap_restore,
        )
        for match in matches
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        help="run balanced match lanes concurrently, one worker per listed GPU",
    )
    parser.add_argument("--full-batch", type=int, default=32)
    parser.add_argument("--local-batch", type=int, default=32)
    parser.add_argument("--optimized-local-runtime", action="store_true")
    parser.add_argument("--optimized-full-runtime", action="store_true")
    parser.add_argument(
        "--subpixel",
        choices=["argmax", "parabolic", "centroid"],
        default="centroid",
        help="heatmap decode refinement for the full-frame sliding candidates; 'centroid' "
        "is the default (sub-cell peak refinement, about -19%% median 2D error at zero "
        "inference cost), 'argmax' reproduces the pre-2026-09 quantised artifacts.",
    )
    far_native_group = parser.add_mutually_exclusive_group()
    far_native_group.add_argument(
        "--far-native",
        dest="far_native",
        action="store_true",
        help="run the default native far-half-court tiling pass and feed its candidates to "
        "every decoder alongside the full-frame sliding pass",
    )
    far_native_group.add_argument(
        "--no-far-native",
        dest="far_native",
        action="store_false",
        help="disable the default native far-half-court tiling pass to reduce GPU cost",
    )
    parser.set_defaults(far_native=True)
    parser.add_argument(
        "--crop-authoritative",
        action="store_true",
        help="let a native-crop detection replace the coarse lock instead of only filling "
        "its gaps. Off by default: measured on the frozen owner windows it costs accuracy, "
        "because no selection rule in this repo can pick the right one of the crop's five "
        "peaks (see WK1_REPORT.md)",
    )
    parser.add_argument(
        "--crop-authority-px1080",
        type=float,
        default=4.0,
        help="native-pixel radius around the coarse lock inside which a native-crop "
        "detection replaces it; replacements beyond about 4 native px were measured to be "
        "wrong-object jumps almost every time",
    )
    parser.add_argument(
        "--no-detour-repair",
        dest="detour_repair",
        action="store_false",
        help="skip the transient-detour repair stage (it has replaced no point on any "
        "measured cohort) and pass the lock track through unchanged",
    )
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--motion-tracker",
        dest="motion_tracker",
        action="store_true",
        help="use the default homography-aware IMM association for the composed track",
    )
    motion_group.add_argument(
        "--legacy-consensus",
        dest="motion_tracker",
        action="store_false",
        help="use the legacy beam consensus, detour repair, and polynomial gap fill",
    )
    parser.set_defaults(motion_tracker=True)
    parser.add_argument(
        "--crop-weights",
        default=CROP_FINETUNE_TEMPLATE,
        help=(
            "detector weights for the native-crop refine pass; may contain {model}, and a "
            "relative path resolves under the data root. Defaults to the native-crop fine-tune."
        ),
    )
    parser.add_argument(
        "--production-crop-weights",
        dest="crop_weights",
        action="store_const",
        const=None,
        help="run the crop pass on the untuned production tennis checkpoints",
    )
    parser.add_argument(
        "--far-native-weights",
        default=None,
        help="detector weights for the far-court native tiling pass; may contain {model}",
    )
    parser.add_argument(
        "--crop-first",
        action="store_true",
        help="motion tracker prefers native-crop observations over the coarse lock",
    )
    parser.add_argument(
        "--full-frame-lock-stream",
        dest="crop_lock_stream",
        action="store_false",
        help="feed the motion tracker the full-frame integrity decode as its lock stream",
    )
    parser.set_defaults(crop_lock_stream=True)
    parser.add_argument(
        "--primary-ball-ownership",
        action="store_true",
        help="motion tracker chains its restarts by a camera-transported position join, "
        "withholds chains proven static under a reliable camera and carries the chain id in "
        "track_id (cv.pipeline.ball_ownership); off by default, every tracked row is "
        "preserved in ball_track_joint_native1080_ownership_v1.csv",
    )
    parser.add_argument(
        "--anchor-extension",
        action="store_true",
        help="after ownership, extend every owned moving segment end in both directions over "
        "the same candidate arms with observed rows only (cv.pipeline.ball_anchor_extension); "
        "requires --primary-ball-ownership; off by default, the consumed artifact is untouched",
    )
    parser.add_argument(
        "--guide-sequence-association",
        action="store_true",
        help="two-pass association (cv.pipeline.guide_sequence_association): an unchanged "
        "first pass supplies the segment/restart/join diagnostics, short guide-opened detours "
        "that the same preceding primary immediately rejoins under reliable camera transport "
        "are re-solved over the full original measured pool, and a qualified sequence is "
        "re-associated by a second ordinary pass; off by default, and with no qualified "
        "bracket the emitted rows are the unflagged tracker's",
    )
    parser.add_argument(
        "--no-torso-lock",
        dest="torso_lock",
        action="store_false",
        help="rollback: keep candidates chained inside a player's torso and the published lock "
        "(default drops them and re-decodes the lock; see TORSO_LOCK_BALL_TRACK)",
    )
    parser.add_argument(
        "--torso-lock-keep-joined",
        action=argparse.BooleanOptionalAction,
        default=TORSO_LOCK_KEEP_JOINED,
        help="keep a far player's torso chain whose ends join outside candidates at ball "
        "speed (torso_lock.KEEP_JOINED; default on)",
    )
    parser.add_argument(
        "--torso-lock-gap-restore",
        action=argparse.BooleanOptionalAction,
        default=TORSO_LOCK_GAP_RESTORE,
        help="fill the torso-lock track's bracketed gaps from the tracker on the unfiltered "
        "pool (torso_lock.restore_lock_gaps; default on)",
    )
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.anchor_extension and not args.primary_ball_ownership:
        parser.error("--anchor-extension requires --primary-ball-ownership")
    if (args.primary_ball_ownership or args.anchor_extension) and not args.motion_tracker:
        parser.error(
            "--primary-ball-ownership and --anchor-extension need the motion tracker stage"
        )
    if args.guide_sequence_association and not args.motion_tracker:
        parser.error("--guide-sequence-association needs the motion tracker stage")
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    code_hashes = verify_frozen_code(protocol)
    input_hashes = verify_frozen_inputs(protocol, args.manifest, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        shard_paths = sorted(args.out.glob("untouched_tracking_inference_run_v1__*.json"))
        rows_by_match = {}
        for shard_path in shard_paths:
            shard = json.loads(shard_path.read_text())
            if shard["verified_code_sha256"] != code_hashes:
                raise RuntimeError(f"code hash mismatch in {shard_path}")
            if shard["verified_input_sha256"] != input_hashes:
                raise RuntimeError(f"input hash mismatch in {shard_path}")
            for row in shard["matches"]:
                rows_by_match[row["match_id"]] = row
        expected = {match["id"] for match in manifest["matches"]}
        if set(rows_by_match) != expected:
            missing = sorted(expected - set(rows_by_match))
            extra = sorted(set(rows_by_match) - expected)
            raise RuntimeError(f"incomplete shards: missing={missing}, extra={extra}")
        report = {
            "schema": "untouched_tracking_inference_run_v1",
            "labels_loaded": False,
            "manifest": str(args.manifest),
            "protocol": str(args.protocol),
            "verified_code_sha256": code_hashes,
            "verified_input_sha256": input_hashes,
            "matches": [rows_by_match[match["id"]] for match in manifest["matches"]],
        }
        output = args.out / "untouched_tracking_inference_run_v1.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output}")
        return
    selected = set(args.match_id)
    known = {match["id"] for match in manifest["matches"]}
    if selected - known:
        raise ValueError(f"unknown match IDs: {sorted(selected - known)}")
    matches = [match for match in manifest["matches"] if not selected or match["id"] in selected]
    devices = args.devices or [args.device]
    if len(set(devices)) != len(devices):
        parser.error("--devices must not contain duplicates")
    lanes = balanced_lanes(matches, args.out, devices)
    if len(devices) == 1:
        rows = run_lane(
            lanes[0],
            args.out,
            devices[0],
            args.full_batch,
            args.local_batch,
            args.optimized_local_runtime,
            args.optimized_full_runtime,
            args.subpixel,
            args.far_native,
            args.crop_authoritative,
            args.crop_authority_px1080,
            args.detour_repair,
            args.motion_tracker,
            args.crop_weights,
            args.far_native_weights,
            args.crop_first,
            args.crop_lock_stream,
            args.primary_ball_ownership,
            args.anchor_extension,
            args.guide_sequence_association,
            args.torso_lock,
            args.torso_lock_keep_joined,
            args.torso_lock_gap_restore,
        )
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    run_lane,
                    lane,
                    args.out,
                    device,
                    args.full_batch,
                    args.local_batch,
                    args.optimized_local_runtime,
                    args.optimized_full_runtime,
                    args.subpixel,
                    args.far_native,
                    args.crop_authoritative,
                    args.crop_authority_px1080,
                    args.detour_repair,
                    args.motion_tracker,
                    args.crop_weights,
                    args.far_native_weights,
                    args.crop_first,
                    args.crop_lock_stream,
                    args.primary_ball_ownership,
                    args.anchor_extension,
                    args.guide_sequence_association,
                    args.torso_lock,
                    args.torso_lock_keep_joined,
                    args.torso_lock_gap_restore,
                )
                for lane, device in zip(lanes, devices, strict=True)
            ]
            rows = [row for future in futures for row in future.result()]
        order = {match["id"]: index for index, match in enumerate(matches)}
        rows.sort(key=lambda row: order[row["match_id"]])
    report = {
        "schema": "untouched_tracking_inference_run_v1",
        "labels_loaded": False,
        "manifest": str(args.manifest),
        "protocol": str(args.protocol),
        "verified_code_sha256": code_hashes,
        "verified_input_sha256": input_hashes,
        "matches": rows,
    }
    suffix = ""
    if selected:
        suffix = "__" + "__".join(sorted(selected))
    output = args.out / f"untouched_tracking_inference_run_v1{suffix}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
