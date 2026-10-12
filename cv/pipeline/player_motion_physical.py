"""Court-register tracked player pose and emit physical racket-motion hypotheses.

This stage is automatic inference. It consumes only automatic player pose, camera, and configured
biometric artifacts. Monocular body-model outputs may be supplied as explicit, automatic priors;
their free camera and root translation are never trusted. Every selected joint remains on its
observed camera ray while anthropometric, court-root, ground, and temporal constraints resolve
depth. The output is deliberately leaky: competing height and racket hypotheses survive for S6.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from cv.pipeline import resolution as res
from cv.pipeline.pose import KEYPOINT_NAMES
from cv.pipeline.run_manifest import StageRun

MODEL_NAME = "court_ray_anthropometric_v1"
DEFAULT_POSE_NAME = "player_pose_tracked_crop_native_v1.csv"
DEFAULT_CAMERA_NAME = "camera_P_per_frame_v1.npz"
DEFAULT_OUTPUT_NAME = "player_motion_physical_v1.jsonl"

MINIMUM_MOTION_TARGET_CROP_AREA_PX2 = 64.0**2
MAXIMUM_MOTION_TARGET_TEMPORAL_CORRECTION_BOX_HEIGHTS = 0.03
MAXIMUM_MOTION_TARGET_ROOT_SPEED_MPS = 30.0
MAXIMUM_MOTION_TARGET_LOWEST_JOINT_Z_M = 0.35
MAXIMUM_MOTION_TARGET_NEIGHBOR_SECONDS = 0.081

JOINT_INDEX = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
LEFT_ELBOW = JOINT_INDEX["left_elbow"]
RIGHT_ELBOW = JOINT_INDEX["right_elbow"]
LEFT_WRIST = JOINT_INDEX["left_wrist"]
RIGHT_WRIST = JOINT_INDEX["right_wrist"]
LEFT_HIP = JOINT_INDEX["left_hip"]
RIGHT_HIP = JOINT_INDEX["right_hip"]
LEFT_KNEE = JOINT_INDEX["left_knee"]
RIGHT_KNEE = JOINT_INDEX["right_knee"]
LEFT_ANKLE = JOINT_INDEX["left_ankle"]
RIGHT_ANKLE = JOINT_INDEX["right_ankle"]

# Segment lengths are fractions of standing height. They are intentionally broad constraints,
# not claims about an individual player's exact anthropometry.
BONES = (
    (LEFT_SHOULDER, RIGHT_SHOULDER, 0.245),
    (LEFT_HIP, RIGHT_HIP, 0.185),
    (LEFT_SHOULDER, LEFT_ELBOW, 0.185),
    (RIGHT_SHOULDER, RIGHT_ELBOW, 0.185),
    (LEFT_ELBOW, LEFT_WRIST, 0.155),
    (RIGHT_ELBOW, RIGHT_WRIST, 0.155),
    (LEFT_SHOULDER, LEFT_HIP, 0.285),
    (RIGHT_SHOULDER, RIGHT_HIP, 0.285),
    (LEFT_HIP, LEFT_KNEE, 0.245),
    (RIGHT_HIP, RIGHT_KNEE, 0.245),
    (LEFT_KNEE, LEFT_ANKLE, 0.246),
    (RIGHT_KNEE, RIGHT_ANKLE, 0.246),
)

BODY_PROFILE_BONE_SCALE = {
    "neutral": (1.0,) * len(BONES),
    "female_smpl": (0.974, 1.060, 1.003, 1.003, 0.992, 0.992, 1.0, 1.0, 1.004, 1.004, 0.997, 0.997),
    "male_smpl": (1.078, 0.833, 0.964, 0.964, 1.017, 1.017, 1.0, 1.0, 0.976, 0.976, 1.018, 1.018),
}

HEIGHT_FRACTION = {
    JOINT_INDEX["nose"]: 0.945,
    LEFT_SHOULDER: 0.815,
    RIGHT_SHOULDER: 0.815,
    LEFT_ELBOW: 0.670,
    RIGHT_ELBOW: 0.670,
    LEFT_WRIST: 0.535,
    RIGHT_WRIST: 0.535,
    LEFT_HIP: 0.525,
    RIGHT_HIP: 0.525,
    LEFT_KNEE: 0.275,
    RIGHT_KNEE: 0.275,
    LEFT_ANKLE: 0.035,
    RIGHT_ANKLE: 0.035,
}


@dataclass(frozen=True)
class HeightHypothesis:
    player_id: str
    height_m: float
    prior: float
    body_profile: str = "neutral"


def frame_number(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(Path(value).stem.rsplit("_", 1)[-1])


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def camera_ray(projection: np.ndarray, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(projection[:, :3], dtype=float)
    offset = np.asarray(projection[:, 3], dtype=float)
    center = -np.linalg.solve(matrix, offset)
    direction = np.linalg.solve(matrix, np.asarray([pixel[0], pixel[1], 1.0]))
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return center, direction


def project(projection: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    image = homogeneous @ projection.T
    return image[:, :2] / image[:, 2:3]


def root_intersection_parameter(
    center: np.ndarray,
    direction: np.ndarray,
    root_xy: np.ndarray,
) -> float:
    direction_xy = np.asarray(direction[:2], dtype=float)
    denominator = float(direction_xy @ direction_xy)
    if denominator < 1e-12:
        return 10.0
    return float(direction_xy @ (np.asarray(root_xy, dtype=float) - center[:2]) / denominator)


def row_keypoints(row: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[float(row[f"{name}_x"]), float(row[f"{name}_y"])] for name in KEYPOINT_NAMES],
        dtype=float,
    )
    confidence = np.asarray(
        [float(row.get(f"{name}_confidence", 0.0)) for name in KEYPOINT_NAMES],
        dtype=float,
    )
    return points, np.clip(confidence, 0.0, 1.0)


def pose_track_alignment(row: dict) -> dict | None:
    keys = ("track_x0", "track_y0", "track_x1", "track_y1")
    if any(row.get(key, "") == "" for key in keys):
        return None
    pose_box = np.asarray([float(row[key]) for key in ("x0", "y0", "x1", "y1")])
    track_box = np.asarray([float(row[key]) for key in keys])
    intersection_width = max(0.0, min(pose_box[2], track_box[2]) - max(pose_box[0], track_box[0]))
    intersection_height = max(0.0, min(pose_box[3], track_box[3]) - max(pose_box[1], track_box[1]))
    intersection = intersection_width * intersection_height
    pose_area = max(0.0, pose_box[2] - pose_box[0]) * max(0.0, pose_box[3] - pose_box[1])
    track_area = max(0.0, track_box[2] - track_box[0]) * max(0.0, track_box[3] - track_box[1])
    union = max(pose_area + track_area - intersection, 1.0)
    pose_center = 0.5 * (pose_box[:2] + pose_box[2:])
    track_center = 0.5 * (track_box[:2] + track_box[2:])
    track_diagonal = max(float(np.linalg.norm(track_box[2:] - track_box[:2])), 1.0)
    return {
        "iou": float(intersection / union),
        "center_distance_track_diagonals": float(
            np.linalg.norm(pose_center - track_center) / track_diagonal
        ),
    }


def robust_temporal_pixels(
    rows: list[dict], radius: int = 3
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Repair isolated 2D joint jumps while preserving sustained swing motion."""
    ordered = sorted(rows, key=lambda row: frame_number(row["frame"]))
    frames = np.asarray([frame_number(row["frame"]) for row in ordered], dtype=int)
    points = np.stack([row_keypoints(row)[0] for row in ordered])
    confidence = np.stack([row_keypoints(row)[1] for row in ordered])
    corrected = points.copy()
    uncertainty = np.full(confidence.shape, np.nan)
    box_heights = np.asarray(
        [max(float(row["y1"]) - float(row["y0"]), 1.0) for row in ordered], dtype=float
    )
    for index, frame in enumerate(frames):
        neighborhood = np.flatnonzero(np.abs(frames - frame) <= radius)
        for joint in range(len(KEYPOINT_NAMES)):
            available = neighborhood[confidence[neighborhood, joint] >= 0.10]
            if len(available) < 3:
                continue
            local_frames = frames[available].astype(float)
            design = np.column_stack(
                (
                    np.ones(len(available)),
                    local_frames - frame,
                    (local_frames - frame) ** 2,
                )
            )
            weights = np.maximum(confidence[available, joint], 0.05) * np.exp(
                -0.30 * (local_frames - frame) ** 2
            )
            coefficients = np.linalg.lstsq(
                np.sqrt(weights)[:, None] * design,
                np.sqrt(weights)[:, None] * points[available, joint],
                rcond=None,
            )[0]
            fitted = coefficients[0]
            residuals = np.linalg.norm(points[available, joint] - design @ coefficients, axis=1)
            sigma = max(float(np.median(residuals)) * 1.4826, 0.5)
            displacement = float(np.linalg.norm(points[index, joint] - fitted))
            outlier = displacement > max(4.0 * sigma, 0.10 * box_heights[index])
            low_confidence = confidence[index, joint] < 0.20
            if outlier or low_confidence:
                corrected[index, joint] = fitted
            uncertainty[index, joint] = sigma
    return {
        int(frame): (corrected[index], uncertainty[index]) for index, frame in enumerate(frames)
    }


def adjacent_root_speeds(rows: list[dict], fps: float) -> dict[int, float | None]:
    """Measure the largest local court-root speed without assuming a fixed frame rate."""
    ordered = sorted(rows, key=lambda row: frame_number(row["frame"]))
    if fps <= 0.0:
        return {frame_number(row["frame"]): None for row in ordered}
    output: dict[int, float | None] = {}
    for index, row in enumerate(ordered):
        frame = frame_number(row["frame"])
        root = np.asarray([float(row["court_x"]), float(row["court_y"])], dtype=float)
        speeds = []
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            neighbor_frame = frame_number(neighbor["frame"])
            elapsed = abs(neighbor_frame - frame) / fps
            if elapsed <= 0.0 or elapsed > MAXIMUM_MOTION_TARGET_NEIGHBOR_SECONDS:
                continue
            neighbor_root = np.asarray(
                [float(neighbor["court_x"]), float(neighbor["court_y"])], dtype=float
            )
            speeds.append(float(np.linalg.norm(root - neighbor_root) / elapsed))
        output[frame] = max(speeds) if speeds else None
    return output


def motion_training_target_quality(
    row: dict,
    raw_pixels: np.ndarray,
    corrected_pixels: np.ndarray,
    diagnostics: dict,
    *,
    fps: float,
    maximum_adjacent_root_speed_mps: float | None,
) -> dict:
    """Certify a pure self-supervised target while retaining held pose as leaky evidence."""
    width = max(float(row["x1"]) - float(row["x0"]), 0.0)
    height = max(float(row["y1"]) - float(row["y0"]), 0.0)
    normalization_height = max(height, 1.0)
    maximum_correction = float(
        np.max(
            np.linalg.norm(
                np.asarray(raw_pixels, dtype=float) - np.asarray(corrected_pixels, dtype=float),
                axis=1,
            )
        )
        / normalization_height
    )
    crop_area = width * height
    reasons = []
    if fps <= 0.0:
        reasons.append("native_cadence_unavailable")
    if crop_area < MINIMUM_MOTION_TARGET_CROP_AREA_PX2:
        reasons.append("insufficient_native_pose_pixels")
    if maximum_correction > MAXIMUM_MOTION_TARGET_TEMPORAL_CORRECTION_BOX_HEIGHTS:
        reasons.append("temporal_pose_correction_excessive")
    if (
        maximum_adjacent_root_speed_mps is not None
        and maximum_adjacent_root_speed_mps > MAXIMUM_MOTION_TARGET_ROOT_SPEED_MPS
    ):
        reasons.append("impossible_player_root_speed")
    if float(diagnostics["minimum_z_m"]) > MAXIMUM_MOTION_TARGET_LOWEST_JOINT_Z_M:
        reasons.append("ungrounded_pose_target")
    return {
        "schema": "motion_training_target_quality_v1",
        "decision": "accept" if not reasons else "hold",
        "reasons": reasons,
        "evidence": {
            "native_crop_width_px": width,
            "native_crop_height_px": height,
            "native_crop_area_px2": crop_area,
            "maximum_temporal_correction_box_heights": maximum_correction,
            "maximum_adjacent_root_speed_mps": maximum_adjacent_root_speed_mps,
            "lowest_joint_z_m": float(diagnostics["minimum_z_m"]),
            "native_fps": fps if fps > 0.0 else None,
        },
        "thresholds": {
            "minimum_native_crop_area_px2": MINIMUM_MOTION_TARGET_CROP_AREA_PX2,
            "maximum_temporal_correction_box_heights": (
                MAXIMUM_MOTION_TARGET_TEMPORAL_CORRECTION_BOX_HEIGHTS
            ),
            "maximum_adjacent_root_speed_mps": MAXIMUM_MOTION_TARGET_ROOT_SPEED_MPS,
            "maximum_lowest_joint_z_m": MAXIMUM_MOTION_TARGET_LOWEST_JOINT_Z_M,
        },
    }


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def automatic_foot_contact_witness(
    rows: list[dict],
    fps: float,
    cameras: dict[tuple[str, int], np.ndarray] | None = None,
) -> dict[int, dict]:
    """Infer grounded versus airborne intervals from native-cadence foot motion.

    The lower visible ankle is used so a normal running stride does not look like a jump. A
    centered temporal baseline removes court-depth locomotion; only brief, two-foot lift excursions
    survive. The witness is deliberately soft and abstains across missing or weak pose evidence.
    """
    ordered = sorted(rows, key=lambda row: frame_number(row["frame"]))
    if not ordered or not math.isfinite(fps) or fps <= 0.0:
        return {}
    frames = np.asarray([frame_number(row["frame"]) for row in ordered], dtype=int)
    foot_y = np.empty(len(ordered), dtype=float)
    foot_confidence = np.empty(len(ordered), dtype=float)
    projected_ground_y = np.full(len(ordered), np.nan, dtype=float)
    box_height = np.asarray(
        [max(float(row["y1"]) - float(row["y0"]), 1.0) for row in ordered], dtype=float
    )
    for index, row in enumerate(ordered):
        points, confidence = row_keypoints(row)
        available = [
            ankle
            for ankle in (LEFT_ANKLE, RIGHT_ANKLE)
            if confidence[ankle] >= 0.30 and np.all(np.isfinite(points[ankle]))
        ]
        if available:
            lowest = max(available, key=lambda ankle: points[ankle, 1])
            foot_y[index] = points[lowest, 1]
            foot_confidence[index] = confidence[lowest]
        else:
            foot_y[index] = float(row["y1"])
            foot_confidence[index] = 0.10
        frame = frame_number(row["frame"])
        camera = cameras.get((row["clip"], frame)) if cameras else None
        if camera is not None and row.get("court_x", "") != "" and row.get("court_y", "") != "":
            ground = np.asarray([[float(row["court_x"]), float(row["court_y"]), 0.0]], dtype=float)
            projected_ground_y[index] = project(camera, ground)[0, 1]

    output = {
        int(frame): {
            "state": "unknown",
            "grounded_probability": 0.50,
            "airborne_probability": 0.50,
            "source": "native_pose_foot_motion_v1",
        }
        for frame in frames
    }
    maximum_gap = max(2, int(round(0.12 * fps)))
    boundaries = [0]
    boundaries.extend((np.flatnonzero(np.diff(frames) > maximum_gap) + 1).tolist())
    boundaries.append(len(frames))
    trend_radius = max(2, int(round(0.40 * fps)))
    bridge_frames = max(1, int(round(0.06 * fps)))
    maximum_air_frames = max(2, int(round(0.55 * fps)))
    for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if segment_end - segment_start < 5:
            continue
        segment_foot = foot_y[segment_start:segment_end]
        segment_ground = projected_ground_y[segment_start:segment_end]
        segment_height = box_height[segment_start:segment_end]
        segment_confidence = foot_confidence[segment_start:segment_end]
        trend = np.empty_like(segment_foot)
        court_projected = np.count_nonzero(np.isfinite(segment_ground)) >= 0.8 * len(segment_ground)
        signal = segment_ground - segment_foot if court_projected else -segment_foot
        for local_index in range(len(segment_foot)):
            left = max(0, local_index - trend_radius)
            right = min(len(segment_foot), local_index + trend_radius + 1)
            local = signal[left:right]
            local = local[np.isfinite(local)]
            trend[local_index] = np.percentile(local, 20.0) if len(local) else signal[local_index]
        lift = signal - trend
        enter = np.maximum(3.0, 0.025 * segment_height)
        candidate = (lift > enter) & (segment_confidence >= 0.25)
        for gap_start, gap_end in _boolean_runs(~candidate):
            if gap_start > 0 and gap_end < len(candidate) and gap_end - gap_start <= bridge_frames:
                candidate[gap_start:gap_end] = True
        airborne = np.zeros(len(candidate), dtype=bool)
        run_probability = np.zeros(len(candidate), dtype=float)
        for run_start, run_end in _boolean_runs(candidate):
            if run_end - run_start > maximum_air_frames:
                continue
            peak = float(np.max(lift[run_start:run_end]))
            required_peak = max(8.0, 0.055 * float(np.median(segment_height[run_start:run_end])))
            if peak < required_peak:
                continue
            airborne[run_start:run_end] = True
            probability = float(np.clip(0.55 + 0.35 * (peak / required_peak - 1.0), 0.55, 0.95))
            run_probability[run_start:run_end] = probability
        for local_index, global_index in enumerate(range(segment_start, segment_end)):
            confidence = float(segment_confidence[local_index])
            if confidence < 0.25:
                continue
            if airborne[local_index]:
                airborne_probability = run_probability[local_index]
                state = "airborne"
            else:
                airborne_probability = float(np.clip(0.10 + 0.20 * (1.0 - confidence), 0.10, 0.30))
                state = "grounded"
            output[int(frames[global_index])] = {
                "state": state,
                "grounded_probability": 1.0 - airborne_probability,
                "airborne_probability": airborne_probability,
                "lift_px": float(lift[local_index]),
                "box_height_px": float(segment_height[local_index]),
                "pose_confidence": confidence,
                "source": (
                    "court_projected_native_pose_foot_motion_v2"
                    if court_projected
                    else "native_pose_foot_motion_v1"
                ),
            }
    return output


def load_cadence_fps(path: Path | None, *, match_id: str) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if payload.get("schema") != "frame_cadence_audit_v1":
        raise ValueError(f"unsupported cadence schema in {path}")
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("match_id") in {None, match_id}
        and row.get("decision") == "timing_usable"
        and float(row.get("fps", 0.0)) > 0.0
    ]
    fps_by_clip: dict[str, float] = {}
    for row in rows:
        clip = str(row["clip"])
        fps = float(row["fps"])
        if clip in fps_by_clip and not math.isclose(fps_by_clip[clip], fps):
            raise ValueError(f"ambiguous cadence for {match_id}/{clip} in {path}")
        fps_by_clip[clip] = fps
    return fps_by_clip


def height_hypotheses(match_id: str, biometrics: dict) -> list[HeightHypothesis]:
    players = biometrics.get("matches", {}).get(match_id, [])
    hypotheses = [
        HeightHypothesis(
            player_id=player_id,
            height_m=float(biometrics["players"][player_id]["height_m"]),
            prior=1.0 / max(len(players), 1),
            body_profile=(
                "female_smpl"
                if biometrics["players"][player_id].get("sex") == "female"
                else "male_smpl"
                if biometrics["players"][player_id].get("sex") == "male"
                else "female_smpl"
                if "wtatennis.com" in biometrics["players"][player_id].get("source", "")
                else "male_smpl"
                if "atptour.com" in biometrics["players"][player_id].get("source", "")
                else "neutral"
            ),
        )
        for player_id in players
        if player_id in biometrics.get("players", {})
    ]
    if hypotheses:
        return hypotheses
    return [HeightHypothesis(player_id="unknown", height_m=1.82, prior=1.0)]


def _side_player_score(documents: list[dict], player_id: str) -> float:
    frame_scores = []
    for document in documents:
        candidates = [
            float(branch["score"])
            for branch in document["branches"]
            if branch["player_id"] == player_id
        ]
        if candidates:
            frame_scores.append(math.log(max(max(candidates), 1e-9)))
    return float(np.median(frame_scores)) if frame_scores else -math.inf


def lock_player_dimensions(
    documents: list[dict],
    heights: list[HeightHypothesis],
) -> dict[tuple[str, str], dict]:
    """Choose one immutable registered body per clip side, jointly when both sides exist."""
    players = {height.player_id: height for height in heights}
    by_clip_side: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for document in documents:
        by_clip_side[(document["clip"], document["side"])].append(document)
    locks = {}
    for clip in sorted({key[0] for key in by_clip_side}):
        sides = [side for side in ("near", "far") if (clip, side) in by_clip_side]
        assignments = []
        for ordered_players in permutations(players, min(len(sides), len(players))):
            if len(ordered_players) != len(sides):
                continue
            score = sum(
                _side_player_score(by_clip_side[(clip, side)], player_id)
                for side, player_id in zip(sides, ordered_players, strict=True)
            )
            assignments.append((score, dict(zip(sides, ordered_players, strict=True))))
        if not assignments:
            continue
        assignments.sort(key=lambda item: item[0], reverse=True)
        best_score, best = assignments[0]
        second_score = assignments[1][0] if len(assignments) > 1 else -math.inf
        margin = best_score - second_score if math.isfinite(second_score) else math.inf
        confidence = 1.0 if not math.isfinite(margin) else float(1.0 - math.exp(-max(margin, 0.0)))
        dimension_span = max(height.height_m for height in heights) - min(
            height.height_m for height in heights
        )
        identity_safe = margin >= 1.0 or len(players) == 1
        assignment_safe = confidence >= 0.05 or dimension_span <= 0.03 or len(players) == 1
        for side, player_id in best.items():
            height = players[player_id]
            locks[(clip, side)] = {
                "schema": "player_body_dimension_lock_v1",
                "scope": "clip_side",
                "source": "joint_registered_height_likelihood",
                "player_id": player_id,
                "height_m": height.height_m,
                "body_profile": height.body_profile,
                "assignment_score": best_score,
                "assignment_margin": None if not math.isfinite(margin) else margin,
                "assignment_confidence": confidence,
                "dimension_span_m": dimension_span,
                "assignment_safe": assignment_safe,
                "identity_safe": identity_safe,
            }
    return locks


def load_dimension_locks(path: Path | None, match_id: str) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    document = json.loads(path.read_text())
    if document.get("automatic_only") is not True or document.get("human_derived_inputs"):
        raise ValueError(f"body dimensions must be automatic-only: {path}")
    if document.get("match_id") != match_id:
        raise ValueError(f"body dimensions match mismatch: {path}")
    locks = {}
    for key, row in document.get("clip_side_assignments", {}).items():
        clip, side = key.rsplit("/", 1)
        locks[(clip, side)] = {"schema": "player_body_dimension_lock_v1", **row}
    return locks


def _physical_residuals(
    lambdas: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    confidence: np.ndarray,
    root_xy: np.ndarray,
    height_m: float,
    previous: np.ndarray | None,
    body_prior: np.ndarray | None,
    body_prior_pose_strength: float,
    grounded_probability: float,
    body_profile: str,
) -> np.ndarray:
    joints = centers + lambdas[:, None] * directions
    residuals = []
    profile_scale = BODY_PROFILE_BONE_SCALE[body_profile]
    for (first, second, fraction), scale in zip(BONES, profile_scale, strict=True):
        weight = math.sqrt(max(min(confidence[first], confidence[second]), 0.05))
        residuals.append(
            weight
            * (np.linalg.norm(joints[first] - joints[second]) - scale * fraction * height_m)
            / 0.035
        )
    ankles = 0.5 * (joints[LEFT_ANKLE] + joints[RIGHT_ANKLE])
    hips = 0.5 * (joints[LEFT_HIP] + joints[RIGHT_HIP])
    residuals.extend(((ankles[:2] - root_xy) / 0.10).tolist())
    residuals.append(min(ankles[2], 0.0) / 0.015)
    residuals.append(
        math.sqrt(float(np.clip(grounded_probability, 0.0, 1.0)))
        * (min(joints[LEFT_ANKLE, 2], joints[RIGHT_ANKLE, 2]) - 0.015)
        / 0.06
    )
    residuals.append((hips[2] - 0.525 * height_m) / 0.20)
    for joint, fraction in HEIGHT_FRACTION.items():
        weight = 0.30 * math.sqrt(max(confidence[joint], 0.05))
        residuals.append(weight * (joints[joint, 2] - fraction * height_m) / 0.22)
    # Broadcast views make depth weak. Keep the body compact around its court root without forcing
    # a flat plane; arms get more freedom than torso and legs.
    arm_joints = {LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST}
    for joint in range(len(KEYPOINT_NAMES)):
        depth_sigma = 0.75 if joint in arm_joints else 0.38
        residuals.append(0.25 * (joints[joint, 1] - root_xy[1]) / depth_sigma)
        residuals.append(min(joints[joint, 2], 0.0) / 0.025)
    if previous is not None and previous.shape == joints.shape:
        for joint in range(len(KEYPOINT_NAMES)):
            sigma = 0.34 if joint in arm_joints else 0.16
            residuals.extend((0.15 * (joints[joint] - previous[joint]) / sigma).tolist())
    if body_prior is not None and body_prior.shape == joints.shape:
        prior_relative = body_prior - 0.5 * (body_prior[LEFT_HIP] + body_prior[RIGHT_HIP])
        current_relative = joints - hips
        residuals.extend(
            (body_prior_pose_strength * (current_relative - prior_relative) / 0.16).ravel().tolist()
        )
    return np.asarray(residuals, dtype=float)


def residual_jacobian_sparsity(
    joint_count: int,
    previous: np.ndarray | None,
    body_prior: np.ndarray | None,
):
    rows: list[set[int]] = []
    rows.extend([{first, second} for first, second, _ in BONES])
    rows.extend([{LEFT_ANKLE, RIGHT_ANKLE}] * 4)
    rows.append({LEFT_HIP, RIGHT_HIP})
    rows.extend({joint} for joint in HEIGHT_FRACTION)
    for joint in range(joint_count):
        rows.extend(({joint}, {joint}))
    if previous is not None:
        for joint in range(joint_count):
            rows.extend(({joint}, {joint}, {joint}))
    if body_prior is not None:
        for joint in range(joint_count):
            dependencies = {joint, LEFT_HIP, RIGHT_HIP}
            rows.extend((dependencies, dependencies, dependencies))
    sparsity = lil_matrix((len(rows), joint_count), dtype=int)
    for row_index, columns in enumerate(rows):
        for column in columns:
            sparsity[row_index, column] = 1
    return sparsity.tocsr()


def align_body_prior(
    body_prior: np.ndarray,
    target: np.ndarray,
    height_m: float,
) -> np.ndarray | None:
    """Register a root-free backend skeleton to the court-oriented ray initialization."""
    result = align_body_prior_transform(body_prior, target, height_m)
    return result[0] if result is not None else None


def align_body_prior_transform(
    body_prior: np.ndarray,
    target: np.ndarray,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the registered skeleton and exact source-to-court rotation."""
    if body_prior.shape != target.shape or not np.all(np.isfinite(body_prior)):
        return None
    source_root = 0.5 * (body_prior[LEFT_HIP] + body_prior[RIGHT_HIP])
    target_root = 0.5 * (target[LEFT_HIP] + target[RIGHT_HIP])
    source = body_prior - source_root
    target_relative = target - target_root
    ankle_midpoint = 0.5 * (source[LEFT_ANKLE] + source[RIGHT_ANKLE])
    source_height = float(np.linalg.norm(source[JOINT_INDEX["nose"]] - ankle_midpoint))
    if source_height < 0.25:
        return None
    source *= (0.91 * height_m) / source_height
    core = np.asarray(
        [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        ]
    )
    covariance = source[core].T @ target_relative[core]
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    return source @ rotation.T + target_root, rotation


def solve_frame(
    projection: np.ndarray,
    pixels: np.ndarray,
    confidence: np.ndarray,
    root_xy: np.ndarray,
    height_m: float,
    previous: np.ndarray | None = None,
    body_prior: np.ndarray | None = None,
    body_prior_pose_strength: float = 0.45,
    grounded_probability: float = 1.0,
    body_profile: str = "neutral",
) -> dict:
    if body_profile not in BODY_PROFILE_BONE_SCALE:
        raise ValueError(f"unsupported body profile: {body_profile}")
    centers = []
    directions = []
    initial = []
    for pixel in pixels:
        center, direction = camera_ray(projection, pixel)
        value = root_intersection_parameter(center, direction, root_xy)
        centers.append(center)
        directions.append(direction)
        initial.append(value)
    centers_array = np.asarray(centers)
    directions_array = np.asarray(directions)
    root_plane_initial = np.asarray(initial)
    root_plane_joints = centers_array + root_plane_initial[:, None] * directions_array
    body_prior_transform = (
        align_body_prior_transform(body_prior, root_plane_joints, height_m)
        if body_prior is not None
        else None
    )
    aligned_body_prior = body_prior_transform[0] if body_prior_transform is not None else None
    constrained_body_prior = aligned_body_prior if body_prior_pose_strength > 0.0 else None
    initial_array = root_plane_initial.copy()
    if constrained_body_prior is not None:
        prior_initial = np.sum(
            (constrained_body_prior - centers_array) * directions_array,
            axis=1,
        )
        finite = np.isfinite(prior_initial)
        initial_array[finite] = np.clip(
            prior_initial[finite],
            root_plane_initial[finite] - 1.8,
            root_plane_initial[finite] + 1.8,
        )
    if previous is not None and previous.shape == (len(KEYPOINT_NAMES), 3):
        temporal_initial = np.sum((previous - centers_array) * directions_array, axis=1)
        finite = np.isfinite(temporal_initial)
        initial_array[finite] = np.clip(
            temporal_initial[finite],
            root_plane_initial[finite] - 1.8,
            root_plane_initial[finite] + 1.8,
        )
    lower = root_plane_initial - 2.0
    upper = root_plane_initial + 2.0
    fit = least_squares(
        _physical_residuals,
        initial_array,
        bounds=(lower, upper),
        args=(
            centers_array,
            directions_array,
            confidence,
            root_xy,
            height_m,
            previous,
            constrained_body_prior,
            body_prior_pose_strength,
            grounded_probability,
            body_profile,
        ),
        loss="soft_l1",
        f_scale=1.0,
        jac_sparsity=residual_jacobian_sparsity(
            len(KEYPOINT_NAMES), previous, constrained_body_prior
        ),
        tr_solver="lsmr",
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        max_nfev=45,
    )
    joints = centers_array + fit.x[:, None] * directions_array
    bone_errors = [
        abs(np.linalg.norm(joints[first] - joints[second]) - scale * fraction * height_m)
        for (first, second, fraction), scale in zip(
            BONES, BODY_PROFILE_BONE_SCALE[body_profile], strict=True
        )
    ]
    ankle_midpoint = 0.5 * (joints[LEFT_ANKLE] + joints[RIGHT_ANKLE])
    reprojection = project(projection, joints)
    reprojection_error = np.linalg.norm(reprojection - pixels, axis=1)
    diagnostics = {
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "bone_rms_m": float(np.sqrt(np.mean(np.square(bone_errors)))),
        "root_error_m": float(np.linalg.norm(ankle_midpoint[:2] - root_xy)),
        "minimum_z_m": float(np.min(joints[:, 2])),
        "maximum_z_m": float(np.max(joints[:, 2])),
        "reprojection_rms_px": float(np.sqrt(np.mean(np.square(reprojection_error)))),
    }
    diagnostics["quality"] = float(
        math.exp(-diagnostics["bone_rms_m"] / 0.08)
        * math.exp(-diagnostics["root_error_m"] / 0.25)
        * math.exp(-max(0.0, -diagnostics["minimum_z_m"]) / 0.05)
        * np.clip(float(np.median(confidence)), 0.0, 1.0)
    )
    return {
        "joints": joints,
        "diagnostics": diagnostics,
        "body_prior_rotation": (
            body_prior_transform[1] if body_prior_transform is not None else None
        ),
    }


def physical_pose_acceptable(
    diagnostics: dict,
    height_m: float,
    *,
    camera_reliable: bool = True,
    track_alignment: dict | None = None,
) -> bool:
    return bool(
        camera_reliable
        and (
            track_alignment is None
            or (
                track_alignment["iou"] >= 0.55
                and track_alignment["center_distance_track_diagonals"] <= 0.45
            )
        )
        and diagnostics["quality"] >= 0.18
        and diagnostics["bone_rms_m"] <= 0.10
        and diagnostics["root_error_m"] <= 0.45
        and diagnostics["minimum_z_m"] >= -0.05
        and diagnostics["maximum_z_m"] <= 1.30 * height_m
    )


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-7 else fallback.copy()


def racket_hypotheses(
    joints: np.ndarray,
    handedness: str | None = None,
    body_prior_rackets: list[dict] | None = None,
    body_prior_rotation: np.ndarray | None = None,
) -> list[dict]:
    """Emit grip/orientation branches around the forearm continuation."""
    torso_left = joints[LEFT_SHOULDER]
    torso_right = joints[RIGHT_SHOULDER]
    torso_axis = _unit(torso_right - torso_left, np.asarray([1.0, 0.0, 0.0]))
    vertical = np.asarray([0.0, 0.0, 1.0])
    output = []
    hands = (
        ("left", LEFT_ELBOW, LEFT_WRIST),
        ("right", RIGHT_ELBOW, RIGHT_WRIST),
    )
    for hand, elbow_index, wrist_index in hands:
        forearm = _unit(joints[wrist_index] - joints[elbow_index], torso_axis)
        lateral = _unit(np.cross(vertical, forearm), torso_axis)
        preferred = handedness is None or handedness == hand
        for grip, lateral_weight, vertical_weight in (
            ("eastern", 0.00, 0.00),
            ("semi_western_open", 0.30, 0.12),
            ("semi_western_closed", -0.30, -0.12),
            ("slice_open", 0.16, -0.34),
            ("serve", -0.10, 0.38),
        ):
            direction = _unit(
                forearm + lateral_weight * lateral + vertical_weight * vertical,
                forearm,
            )
            head_center = joints[wrist_index] + 0.50 * direction
            face_normal = _unit(np.cross(direction, lateral), vertical)
            output.append(
                {
                    "hand": hand,
                    "grip": grip,
                    "prior": round((0.20 if preferred else 0.05), 4),
                    "wrist_xyz": joints[wrist_index].tolist(),
                    "head_center_xyz": head_center.tolist(),
                    "direction_xyz": direction.tolist(),
                    "face_normal_xyz": face_normal.tolist(),
                    "head_radius_m": 0.15,
                    "source": "physical_pose_forearm_racket_branch",
                }
            )
    if body_prior_rackets and body_prior_rotation is not None:
        wrist_indices = {"left": LEFT_WRIST, "right": RIGHT_WRIST}
        for candidate in body_prior_rackets:
            hand = candidate.get("hand")
            if hand not in wrist_indices:
                continue
            direction = _unit(
                body_prior_rotation @ np.asarray(candidate["direction_xyz"], dtype=float),
                torso_axis,
            )
            wrist = joints[wrist_indices[hand]]
            head_center = wrist + 0.50 * direction
            output.append(
                {
                    "hand": hand,
                    "grip": candidate.get("grip", "model_wrist_orientation"),
                    "prior": 0.30,
                    "wrist_xyz": wrist.tolist(),
                    "head_center_xyz": head_center.tolist(),
                    "direction_xyz": direction.tolist(),
                    "face_normal_xyz": None,
                    "head_radius_m": 0.15,
                    "source": candidate.get("source", "body_model_wrist_orientation"),
                }
            )
    return output


def load_cameras(
    path: Path,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict]]:
    data = np.load(path, allow_pickle=True)
    cameras = {}
    quality = {}
    confidence = data["confidence"] if "confidence" in data.files else np.ones(len(data["frames"]))
    reliable = data["reliable"] if "reliable" in data.files else np.ones(len(data["frames"]), bool)
    source = data["source"] if "source" in data.files else np.repeat("unknown", len(data["frames"]))
    residual = (
        data["ground_residual_px"]
        if "ground_residual_px" in data.files
        else np.full(len(data["frames"]), np.nan)
    )
    for index, (clip, frame, projection) in enumerate(
        zip(data["clips"], data["frames"], data["P"], strict=True)
    ):
        key = (str(clip), int(frame))
        cameras[key] = np.asarray(projection, dtype=float)
        quality[key] = {
            "confidence": float(confidence[index]),
            "reliable": bool(reliable[index]),
            "source": str(source[index]),
            "ground_residual_px": float(residual[index]),
        }
    return cameras, quality


def load_body_priors(
    path: Path | list[Path] | tuple[Path, ...] | None,
    *,
    match_id: str | None = None,
) -> dict[tuple[str, str, int], list[dict]]:
    paths = [] if path is None else ([path] if isinstance(path, Path) else list(path))
    output = defaultdict(list)
    seen = set()
    explicit_matches = set()
    for source_path in paths:
        with source_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("schema") != "player_body_prior_coco17_v1":
                    raise ValueError(
                        f"unsupported body-prior schema at {source_path}:{line_number}"
                    )
                if row.get("automatic_only") is not True or row.get("human_derived_inputs"):
                    raise ValueError(
                        f"body prior is not automatic and label-free at {source_path}:{line_number}"
                    )
                row_match = row.get("match_id")
                if row_match:
                    explicit_matches.add(str(row_match))
                    if match_id is not None and row_match != match_id:
                        continue
                joints = np.asarray(row.get("joints_xyz", []), dtype=float)
                if joints.shape != (len(KEYPOINT_NAMES), 3):
                    continue
                key = (row["clip"], row["side"], int(row["frame"]))
                identity = (row_match, *key, row.get("backend"))
                if identity in seen:
                    raise ValueError(
                        f"duplicate body-prior state {identity} at {source_path}:{line_number}"
                    )
                seen.add(identity)
                output[key].append(
                    {
                        "joints": joints,
                        "racket_directions": row.get("racket_directions", []),
                        "backend": row.get("backend"),
                        "match_id": row_match,
                        "source_path": str(source_path),
                        "keypoints_xy_native": row.get("keypoints_xy_native"),
                        "keypoint_confidence": row.get("keypoint_confidence"),
                        "keypoint_confidence_semantics": row.get("keypoint_confidence_semantics"),
                    }
                )
    if match_id is None and len(explicit_matches) > 1:
        raise ValueError(
            "body-prior artifact spans multiple matches; load it with an explicit match_id"
        )
    return dict(output)


def body_prior_projection_reliability(
    grouped: dict[tuple[str, str], list[dict]],
    body_priors: dict[tuple[str, str, int], list[dict]],
) -> dict[tuple[str, str, str], dict]:
    """Calibrate temporal-model projections against automatic 2D overlap."""
    output = {}
    for group, rows in grouped.items():
        clip, side = group
        backends = {
            str(prior.get("backend") or "unknown")
            for (prior_clip, prior_side, _), priors in body_priors.items()
            if prior_clip == clip and prior_side == side
            for prior in priors
        }
        for backend in sorted(backends):
            frame_errors = []
            for row in rows:
                frame = frame_number(row["frame"])
                prior = next(
                    (
                        candidate
                        for candidate in body_priors.get((clip, side, frame), [])
                        if str(candidate.get("backend") or "unknown") == backend
                    ),
                    None,
                )
                if prior is None:
                    continue
                points = np.asarray(prior.get("keypoints_xy_native"), dtype=float)
                if points.shape != (len(KEYPOINT_NAMES), 2) or not np.all(np.isfinite(points)):
                    continue
                observed, confidence = row_keypoints(row)
                valid = confidence >= 0.20
                if np.count_nonzero(valid) < 5:
                    continue
                box_height = max(float(row["y1"]) - float(row["y0"]), 1.0)
                normalized = np.linalg.norm(points[valid] - observed[valid], axis=1) / box_height
                frame_errors.append(float(np.median(normalized)))
            median_error = float(np.median(frame_errors)) if frame_errors else None
            if median_error is None or len(frame_errors) < 3:
                reliability = 0.10
            else:
                reliability = float(np.clip(math.exp(-median_error / 0.10), 0.05, 0.85))
            output[(clip, side, backend)] = {
                "backend": backend,
                "overlap_frames": len(frame_errors),
                "median_normalized_keypoint_error": median_error,
                "reliability": reliability,
                "source": "automatic_2d_temporal_model_overlap",
            }
    return output


def load_track_boxes(path: Path | None) -> dict[tuple[str, str, int], dict]:
    if path is None:
        return {}
    coordinate_manifest = res.read_coordinate_manifest(path)
    if coordinate_manifest is None:
        raise ValueError(f"missing coordinate-space contract for {path}")
    artifact_size = res.manifest_artifact_size(coordinate_manifest)
    image_record = coordinate_manifest.get("image_size", {})
    image_size = res.FrameSize(
        int(image_record.get("width", 0)),
        int(image_record.get("height", 0)),
    )
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            box_columns = (
                ("track_x0", "track_y0", "track_x1", "track_y1")
                if all(
                    row.get(key, "") != ""
                    for key in ("track_x0", "track_y0", "track_x1", "track_y1")
                )
                else ("x0", "y0", "x1", "y1")
            )
            native_box = res.scale_boxes(
                [float(row[key]) for key in box_columns],
                artifact_size,
                image_size,
            )
            output[(row["clip"], row["side"], frame_number(row["frame"]))] = {
                **{
                    f"track_{key}": float(value)
                    for key, value in zip(("x0", "y0", "x1", "y1"), native_box, strict=True)
                },
                "court_x": float(row["court_x"]),
                "court_y": float(row["court_y"]),
                "conf": float(row.get("conf", 0.0)),
            }
    return output


def interpolate_track_box(
    track_boxes: dict[tuple[str, str, int], dict],
    key: tuple[str, str, int],
    maximum_gap: int = 4,
) -> dict | None:
    """Interpolate a short interior automatic player-track gap on a real frame."""
    clip, side, frame = key
    before = [
        candidate
        for candidate in track_boxes
        if candidate[0] == clip and candidate[1] == side and candidate[2] < frame
    ]
    after = [
        candidate
        for candidate in track_boxes
        if candidate[0] == clip and candidate[1] == side and candidate[2] > frame
    ]
    if not before or not after:
        return None
    left_key = max(before, key=lambda candidate: candidate[2])
    right_key = min(after, key=lambda candidate: candidate[2])
    if right_key[2] - left_key[2] > maximum_gap + 1:
        return None
    fraction = (frame - left_key[2]) / (right_key[2] - left_key[2])
    left = track_boxes[left_key]
    right = track_boxes[right_key]
    fields = ("track_x0", "track_y0", "track_x1", "track_y1", "court_x", "court_y")
    return {
        **{
            field: (1.0 - fraction) * float(left[field]) + fraction * float(right[field])
            for field in fields
        },
        "conf": min(float(left["conf"]), float(right["conf"])),
        "track_source": "interpolated_short_interior_gap",
        "track_source_frames": [left_key[2], right_key[2]],
    }


def process_match(
    match_id: str,
    pose_path: Path,
    camera_path: Path,
    biometrics_path: Path,
    output_path: Path,
    body_prior_path: Path | list[Path] | tuple[Path, ...] | None = None,
    boxes_path: Path | None = None,
    cadence_path: Path | None = None,
    body_prior_pose_strength: float = 0.45,
    use_foot_contact_witness: bool = False,
    use_sex_specific_anthropometry: bool = False,
    dimension_locks_path: Path | None = None,
) -> dict:
    biometrics = json.loads(biometrics_path.read_text())
    cameras, camera_quality = load_cameras(camera_path)
    body_priors = load_body_priors(body_prior_path, match_id=match_id)
    track_boxes = load_track_boxes(boxes_path)
    cadence_fps = load_cadence_fps(cadence_path, match_id=match_id)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with pose_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("side") in {"near", "far"}:
                key = (row["clip"], row["side"], frame_number(row["frame"]))
                if key in track_boxes and not row.get("track_x0"):
                    row.update(track_boxes[key])
                grouped[(row["clip"], row["side"])].append(row)
    prior_projection_quality = body_prior_projection_reliability(grouped, body_priors)
    for key in body_priors:
        if key not in track_boxes:
            interpolated = interpolate_track_box(track_boxes, key)
            if interpolated is not None:
                track_boxes[key] = interpolated
    existing = {
        (clip, side, frame_number(row["frame"]))
        for (clip, side), rows in grouped.items()
        for row in rows
    }
    for (clip, side, frame), priors in sorted(body_priors.items()):
        key = (clip, side, frame)
        prior = max(
            priors,
            key=lambda candidate: prior_projection_quality.get(
                (clip, side, str(candidate.get("backend") or "unknown")),
                {"reliability": 0.10},
            )["reliability"],
        )
        points = np.asarray(prior.get("keypoints_xy_native"), dtype=float)
        confidence = np.asarray(prior.get("keypoint_confidence"), dtype=float)
        track = track_boxes.get(key)
        if (
            key in existing
            or track is None
            or points.shape != (len(KEYPOINT_NAMES), 2)
            or confidence.shape != (len(KEYPOINT_NAMES),)
            or not np.all(np.isfinite(points))
        ):
            continue
        projection_quality = prior_projection_quality.get(
            (clip, side, str(prior.get("backend") or "unknown")),
            {
                "overlap_frames": 0,
                "median_normalized_keypoint_error": None,
                "reliability": 0.10,
                "source": "automatic_2d_temporal_model_overlap",
            },
        )
        if prior.get("keypoint_confidence_semantics") == "uncalibrated_model_projection":
            confidence = np.full(len(KEYPOINT_NAMES), projection_quality["reliability"])
        row = {
            "clip": clip,
            "side": side,
            "frame": f"f_{frame:04d}.jpg",
            "x0": track["track_x0"],
            "y0": track["track_y0"],
            "x1": track["track_x1"],
            "y1": track["track_y1"],
            "conf": min(track["conf"], float(projection_quality["reliability"])),
            "court_x": track["court_x"],
            "court_y": track["court_y"],
            "pose_source": f"{prior.get('backend')}_temporal_projection",
            "track_source": track.get("track_source", "observed_player_track"),
            "track_source_frames": track.get("track_source_frames", [frame]),
            **track,
        }
        for index, name in enumerate(KEYPOINT_NAMES):
            row[f"{name}_x"] = points[index, 0]
            row[f"{name}_y"] = points[index, 1]
            row[f"{name}_confidence"] = confidence[index]
        grouped[(clip, side)].append(row)
    heights = height_hypotheses(match_id, biometrics)
    registered_player_ids = {height.player_id for height in heights}
    registered_dimension_locks = load_dimension_locks(dimension_locks_path, match_id)
    for key, lock in registered_dimension_locks.items():
        if lock.get("player_id") not in registered_player_ids:
            raise ValueError(f"unregistered player in body dimension lock {key}: {lock}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path = output_path
    output_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    rows_written = 0
    accepted = 0
    by_backend = defaultdict(int)
    with output_path.open("w") as handle:
        for (clip, side), rows in sorted(grouped.items()):
            temporal_pixels = robust_temporal_pixels(rows)
            root_speeds = adjacent_root_speeds(rows, cadence_fps.get(clip, 0.0))
            foot_contact = (
                automatic_foot_contact_witness(
                    rows,
                    cadence_fps.get(clip, 0.0),
                    cameras,
                )
                if use_foot_contact_witness
                else {}
            )
            previous_by_height: dict[tuple[str, str], np.ndarray] = {}
            for row in sorted(rows, key=lambda value: frame_number(value["frame"])):
                frame = frame_number(row["frame"])
                key = (clip, frame)
                if key not in cameras:
                    continue
                raw_pixels, confidence = row_keypoints(row)
                pixels, pixel_uncertainty = temporal_pixels[frame]
                root_xy = np.asarray([float(row["court_x"]), float(row["court_y"])])
                branches = []
                for height in heights:
                    frame_priors = body_priors.get((clip, side, frame)) or [None]
                    for body_prior in frame_priors:
                        backend = str(body_prior.get("backend") if body_prior else "no_body_prior")
                        projection_quality = prior_projection_quality.get(
                            (clip, side, backend),
                            {
                                "backend": backend,
                                "overlap_frames": 0,
                                "median_normalized_keypoint_error": None,
                                "reliability": 1.0 if body_prior is None else 0.10,
                                "source": "automatic_2d_temporal_model_overlap",
                            },
                        )
                        previous_key = (height.player_id, backend)
                        result = solve_frame(
                            cameras[key],
                            pixels,
                            confidence,
                            root_xy,
                            height.height_m,
                            previous=previous_by_height.get(previous_key),
                            body_prior=body_prior["joints"] if body_prior else None,
                            body_prior_pose_strength=body_prior_pose_strength,
                            grounded_probability=float(
                                foot_contact.get(frame, {}).get("grounded_probability", 1.0)
                            ),
                            body_profile=(
                                height.body_profile if use_sex_specific_anthropometry else "neutral"
                            ),
                        )
                        stable = (
                            result["diagnostics"]["quality"] >= 0.12
                            and result["diagnostics"]["bone_rms_m"] <= 0.12
                            and result["diagnostics"]["root_error_m"] <= 0.55
                            and result["diagnostics"]["minimum_z_m"] >= -0.10
                            and result["diagnostics"]["maximum_z_m"] <= 1.35 * height.height_m
                        )
                        if stable:
                            previous_by_height[previous_key] = result["joints"]
                        score = (
                            height.prior
                            * result["diagnostics"]["quality"]
                            * max(0.25, float(projection_quality["reliability"]))
                        )
                        branches.append(
                            {
                                "backend": MODEL_NAME,
                                "body_prior_backend": (
                                    body_prior.get("backend") if body_prior else None
                                ),
                                "body_prior_source": (
                                    body_prior.get("source_path") if body_prior else None
                                ),
                                "body_prior_projection_quality": projection_quality,
                                "player_id": height.player_id,
                                "height_m": height.height_m,
                                "height_prior": height.prior,
                                "body_profile": (
                                    height.body_profile
                                    if use_sex_specific_anthropometry
                                    else "neutral"
                                ),
                                "score": score,
                                "joints_xyz": result["joints"].tolist(),
                                "diagnostics": result["diagnostics"],
                                "racket_hypotheses": racket_hypotheses(
                                    result["joints"],
                                    body_prior_rackets=(
                                        body_prior.get("racket_directions", [])
                                        if body_prior
                                        else None
                                    ),
                                    body_prior_rotation=result["body_prior_rotation"],
                                ),
                            }
                        )
                selected_index = int(np.argmax([branch["score"] for branch in branches]))
                selected = branches[selected_index]
                track_alignment = pose_track_alignment(row)
                training_target = motion_training_target_quality(
                    row,
                    raw_pixels,
                    pixels,
                    selected["diagnostics"],
                    fps=cadence_fps.get(clip, 0.0),
                    maximum_adjacent_root_speed_mps=root_speeds[frame],
                )
                decision = (
                    "accept"
                    if physical_pose_acceptable(
                        selected["diagnostics"],
                        float(selected["height_m"]),
                        camera_reliable=camera_quality[key]["reliable"],
                        track_alignment=track_alignment,
                    )
                    else "hold"
                )
                document = {
                    "schema": "player_motion_physical_v1",
                    "automatic_only": True,
                    "match_id": match_id,
                    "clip": clip,
                    "frame": frame,
                    "side": side,
                    "root_xy": root_xy.tolist(),
                    "pose_source": row.get("pose_source", "observed_tracked_crop_pose"),
                    "track_source": row.get("track_source", "observed_player_track"),
                    "track_source_frames": row.get("track_source_frames", [frame]),
                    "camera": camera_quality[key],
                    "raw_keypoints_xy": raw_pixels.tolist(),
                    "keypoints_xy": pixels.tolist(),
                    "keypoint_confidence": confidence.tolist(),
                    "temporal_uncertainty_px": pixel_uncertainty.tolist(),
                    "body_prior_sources": sorted(
                        {prior["source_path"] for prior in body_priors.get((clip, side, frame), [])}
                    ),
                    "body_prior_projection_quality": [
                        value
                        for (prior_clip, prior_side, _), value in prior_projection_quality.items()
                        if prior_clip == clip and prior_side == side
                    ],
                    "pose_track_alignment": track_alignment,
                    "foot_contact_witness": foot_contact.get(
                        frame,
                        (
                            {
                                "state": "unknown",
                                "grounded_probability": 1.0,
                                "airborne_probability": 0.0,
                                "source": "cadence_unavailable_fail_closed_grounded",
                            }
                            if use_foot_contact_witness
                            else {
                                "state": "disabled",
                                "grounded_probability": 1.0,
                                "airborne_probability": 0.0,
                                "source": "foot_contact_witness_disabled",
                            }
                        ),
                    ),
                    "branches": branches,
                    "selected_branch": selected_index,
                    "decision": decision,
                    "reasons": [] if decision == "accept" else ["physical_pose_quality"],
                    "motion_training_target": training_target,
                }
                handle.write(
                    json.dumps(json_safe(document), separators=(",", ":"), allow_nan=False) + "\n"
                )
                rows_written += 1
                accepted += int(decision == "accept")
                by_backend[selected.get("body_prior_backend") or "court_ray_baseline"] += 1
    preliminary = [json.loads(line) for line in output_path.read_text().splitlines()]
    dimension_locks = lock_player_dimensions(preliminary, heights)
    dimension_locks.update(registered_dimension_locks)
    rows_written = 0
    accepted = 0
    by_backend = defaultdict(int)
    with output_path.open("w") as handle:
        for document in preliminary:
            lock = dimension_locks.get((document["clip"], document["side"]))
            if lock is not None:
                matching = [
                    (index, branch)
                    for index, branch in enumerate(document["branches"])
                    if branch["player_id"] == lock["player_id"]
                ]
                if matching:
                    selected_index, selected = max(
                        matching,
                        key=lambda item: float(item[1]["score"]),
                    )
                    document["selected_branch"] = selected_index
                else:
                    selected = document["branches"][document["selected_branch"]]
            else:
                selected = document["branches"][document["selected_branch"]]
            document["body_dimension_lock"] = lock
            target_evidence = document["motion_training_target"]["evidence"]
            document["motion_training_target"] = motion_training_target_quality(
                {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": float(target_evidence["native_crop_width_px"]),
                    "y1": float(target_evidence["native_crop_height_px"]),
                },
                np.asarray(document["raw_keypoints_xy"], dtype=float),
                np.asarray(document["keypoints_xy"], dtype=float),
                selected["diagnostics"],
                fps=float(target_evidence.get("native_fps") or 0.0),
                maximum_adjacent_root_speed_mps=target_evidence.get(
                    "maximum_adjacent_root_speed_mps"
                ),
            )
            decision = (
                "accept"
                if (lock is None or lock["assignment_safe"])
                and physical_pose_acceptable(
                    selected["diagnostics"],
                    float(selected["height_m"]),
                    camera_reliable=bool(document["camera"]["reliable"]),
                    track_alignment=document.get("pose_track_alignment"),
                )
                else "hold"
            )
            reasons = []
            if lock is not None and not lock["assignment_safe"]:
                reasons.append("ambiguous_player_body_assignment")
            if decision == "hold" and not reasons:
                reasons.append("physical_pose_quality")
            if decision == "hold":
                document["motion_training_target"]["decision"] = "hold"
                document["motion_training_target"]["reasons"] = list(
                    dict.fromkeys(reasons + document["motion_training_target"].get("reasons", []))
                )
            document["decision"] = decision
            document["reasons"] = reasons
            handle.write(
                json.dumps(json_safe(document), separators=(",", ":"), allow_nan=False) + "\n"
            )
            rows_written += 1
            accepted += int(decision == "accept")
            by_backend[selected.get("body_prior_backend") or "court_ray_baseline"] += 1
    output_path.replace(final_output_path)
    return {
        "rows": rows_written,
        "accepted": accepted,
        "held": rows_written - accepted,
        "body_dimension_locks": len(dimension_locks),
        "selected_backends": dict(by_backend),
        "output": str(final_output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--pose-name", default=DEFAULT_POSE_NAME)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument(
        "--dimension-locks",
        type=Path,
        help="Automatic player_body_dimensions_v1 artifact from metric camera refinement.",
    )
    parser.add_argument(
        "--biometrics",
        type=Path,
        default=Path(__file__).with_name("player_biometrics.json"),
    )
    parser.add_argument(
        "--body-prior",
        type=Path,
        action="append",
        default=[],
        help="Repeat for competing automatic temporal SMPL/MHR body-prior branches.",
    )
    parser.add_argument("--body-prior-pose-strength", type=float, default=0.45)
    parser.add_argument("--boxes-name")
    parser.add_argument("--cadence", type=Path)
    parser.add_argument(
        "--foot-contact-witness",
        action="store_true",
        help="Enable the experimental native-cadence grounded/airborne soft constraint.",
    )
    parser.add_argument(
        "--sex-specific-anthropometry",
        action="store_true",
        help="Use normalized official SMPL male/female rest-shape proportions.",
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()
    pose_path = args.match_dir / args.pose_name
    camera_path = args.match_dir / args.camera_name
    output_path = args.match_dir / args.output_name
    boxes_path = args.match_dir / args.boxes_name if args.boxes_name else None
    cadence_path = args.cadence
    stage = StageRun(
        str(args.match_dir),
        "player_motion_physical",
        args,
        models=[{"name": MODEL_NAME, "role": "physical_pose_solver"}],
        reused_artifacts=[
            str(pose_path),
            str(camera_path),
            str(args.biometrics),
            *(str(path) for path in args.body_prior),
            *([str(boxes_path)] if boxes_path else []),
            *([str(cadence_path)] if cadence_path else []),
            *([str(args.dimension_locks)] if args.dimension_locks else []),
        ],
    )
    report = process_match(
        args.match_id,
        pose_path,
        camera_path,
        args.biometrics,
        output_path,
        args.body_prior,
        boxes_path,
        cadence_path,
        args.body_prior_pose_strength,
        args.foot_contact_witness,
        args.sex_specific_anthropometry,
        args.dimension_locks,
    )
    stage.finish(outputs=report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
