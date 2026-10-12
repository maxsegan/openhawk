"""Label-blind monocular 2D-to-3D player-pose lifting.

The lifter consumes one calibrated 3x4 projection, native-image COCO keypoints,
the sided-box court root, and a player stature.  Every emitted joint remains on
its observed camera ray.  Depth is selected with a small kinematic least-squares
fit using stature-scaled limb lengths, a grounded (or box-scale airborne) ankle
anchor, bilateral shoulder/hip constraints, and optional temporal continuity.

This is deliberately a geometric inference component, not a learned human-pose
model.  It does not read player truth, reviewed contacts, or any ambient file.
Callers must declare the provenance of the projection, root, and stature that
they provide.  In particular, evaluation exporters may pass reviewed cameras,
but that does not make their output automatic-inference eligible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


JOINT_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Mean adult segment lengths as fractions of stature.  They are soft priors,
# not identity-specific anthropometry.  Left/right counterparts share exactly
# the same prior, which is the fit's bilateral-consistency constraint.
LIMB_PRIORS = (
    ("nose", "left_shoulder", 0.180),
    ("nose", "right_shoulder", 0.180),
    ("left_shoulder", "left_elbow", 0.186),
    ("right_shoulder", "right_elbow", 0.186),
    ("left_elbow", "left_wrist", 0.146),
    ("right_elbow", "right_wrist", 0.146),
    ("left_hip", "left_knee", 0.245),
    ("right_hip", "right_knee", 0.245),
    ("left_knee", "left_ankle", 0.246),
    ("right_knee", "right_ankle", 0.246),
    ("left_shoulder", "left_hip", 0.288),
    ("right_shoulder", "right_hip", 0.288),
    ("left_shoulder", "right_shoulder", 0.230),
    ("left_hip", "right_hip", 0.190),
)

MIN_KEYPOINT_CONFIDENCE = 0.20
PLANTED_HEIGHT_THRESHOLD_M = 0.07
MAXIMUM_RACKET_FACE_DISTANCE_M = 0.55
RACKET_FACE_FOREARM_SCALE = 1.5
RACKET_FACE_SIGMA_PX = 75.0
MAXIMUM_FOOT_CENTRE_ROOT_DISTANCE_M = 0.90
MAXIMUM_STANCE_WIDTH_M = 1.50
MAXIMUM_ROOT_STEP_M = 0.15
MAXIMUM_INPUT_ROOT_JUMP_M = 2.0
SERVE_TORSO_FORWARD_OFFSET_M = 0.05
SERVE_TORSO_LATERAL_LIMIT_M = 0.08


@dataclass(frozen=True)
class LiftedPose:
    """One fitted pose in metric court coordinates."""

    joints: dict[str, tuple[float, float, float, float]]
    airborne_height_m: float
    support: str
    converged: bool
    cost_rms: float
    method: str = "camera_ray_stature_kinematic_v1"
    root_xy: tuple[float, float] | None = None
    root_step_m: float | None = None
    serve_phase: str | None = None
    striking_hand: str | None = None
    contact_anchor_error_m: float | None = None


@dataclass(frozen=True)
class PoseFrameInput:
    """One chronological pose observation supplied to the sequence lifter."""

    frame: int
    row: Mapping[str, Any]
    projection: np.ndarray
    root_xy: tuple[float, float]


@dataclass(frozen=True)
class ServeWindow:
    """Explicit serve geometry; provenance remains the caller's responsibility."""

    start_frame: int
    contact_frame: float
    end_frame: int
    contact_xyz: tuple[float, float, float] | None = None
    court_forward_y: float = 1.0
    anchor_source: str = "unspecified"


def discontinuous_root_frames(
    frames: Sequence[PoseFrameInput],
    maximum_jump_m: float = MAXIMUM_INPUT_ROOT_JUMP_M,
) -> dict[int, float]:
    """Return native frames whose adjacent court roots make an unsafe jump.

    This is a display fail-closed guard, not a player-track repair. Both ends of
    an over-limit one-frame transition are withheld so a skeleton is never drawn
    on either endpoint of an identity/association discontinuity.
    """
    if not np.isfinite(maximum_jump_m) or maximum_jump_m <= 0:
        raise ValueError("maximum_jump_m must be finite and positive")
    ordered = sorted(frames, key=lambda item: item.frame)
    unsafe: dict[int, float] = {}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.frame != previous.frame + 1:
            continue
        distance = float(np.linalg.norm(np.asarray(current.root_xy) - np.asarray(previous.root_xy)))
        if distance > maximum_jump_m:
            unsafe[previous.frame] = max(distance, unsafe.get(previous.frame, 0.0))
            unsafe[current.frame] = max(distance, unsafe.get(current.frame, 0.0))
    return unsafe


def project_point(projection: np.ndarray, xyz: Sequence[float]) -> tuple[float, float] | None:
    """Project one metric court point to native-image pixels."""
    matrix = np.asarray(projection, dtype=float)
    point = np.asarray([*xyz, 1.0], dtype=float)
    if matrix.shape != (3, 4) or point.shape != (4,) or not np.isfinite(matrix).all():
        return None
    homogeneous = matrix @ point
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) < 1e-10:
        return None
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


def camera_ray(
    projection: np.ndarray, pixel: Sequence[float], target_hint: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera centre and a unit ray directed toward ``target_hint``."""
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("projection must be a finite 3x4 matrix")
    calibration = matrix[:, :3]
    if abs(float(np.linalg.det(calibration))) < 1e-12:
        raise ValueError("projection has a singular left 3x3 block")
    centre = -np.linalg.solve(calibration, matrix[:, 3])
    image = np.asarray([float(pixel[0]), float(pixel[1]), 1.0])
    direction = np.linalg.solve(calibration, image)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("pixel back-projects to a degenerate ray")
    direction /= norm
    hint = np.asarray(target_hint, dtype=float)
    if float(np.dot(hint - centre, direction)) < 0:
        direction *= -1.0
    return centre, direction


def _finite(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _visible_keypoints(
    row: Mapping[str, Any], minimum_confidence: float
) -> dict[str, tuple[float, float, float]]:
    visible = {}
    for name in JOINT_NAMES:
        x = _finite(row, f"{name}_x")
        y = _finite(row, f"{name}_y")
        confidence = _finite(row, f"{name}_confidence")
        if x is None or y is None or confidence is None or confidence < minimum_confidence:
            continue
        visible[name] = (x, y, confidence)
    return visible


def estimate_airborne_height(
    row: Mapping[str, Any], stature_m: float, visible: Mapping[str, Sequence[float]]
) -> tuple[float, str]:
    """Estimate sole height from the ankle-to-box-bottom image gap.

    A gap below seven centimetres is treated as planted and snapped to z=0.
    Larger gaps retain their stature/box-height metric scale.  This is an
    automatic geometric proxy, not a qualified jump classifier.
    """
    y0 = _finite(row, "y0")
    y1 = _finite(row, "y1")
    ankles = [visible[name] for name in ("left_ankle", "right_ankle") if name in visible]
    if y0 is None or y1 is None or y1 <= y0 or not ankles:
        raise ValueError("a valid box and at least one ankle are required")
    scale = float(stature_m) / (y1 - y0)
    height = max(0.0, (y1 - max(float(point[1]) for point in ankles)) * scale)
    if height <= PLANTED_HEIGHT_THRESHOLD_M:
        return 0.0, "planted"
    return min(height, 0.45 * float(stature_m)), "airborne_box_scale"


def _target_heights(
    row: Mapping[str, Any],
    stature_m: float,
    visible: Mapping[str, Sequence[float]],
    airborne: float,
) -> dict[str, float]:
    y0 = float(row["y0"])
    y1 = float(row["y1"])
    scale = stature_m / (y1 - y0)
    ankle_y = max(
        float(visible[name][1]) for name in ("left_ankle", "right_ankle") if name in visible
    )
    return {
        name: float(np.clip(airborne + (ankle_y - point[1]) * scale, 0.0, 1.15 * stature_m))
        for name, point in visible.items()
    }


def lift_pose_frame(
    row: Mapping[str, Any],
    projection: np.ndarray,
    root_xy: Sequence[float],
    stature_m: float,
    *,
    previous_local: Mapping[str, Sequence[float]] | None = None,
    minimum_confidence: float = MIN_KEYPOINT_CONFIDENCE,
    airborne_root_continuity: bool = False,
) -> LiftedPose | None:
    """Lift one native-image pose to metric court 3D.

    ``previous_local`` contains the preceding frame's joints relative to its
    own sided-box root.  It is a soft regularizer only; missing frames should
    be represented by passing ``None`` so motion is never bridged silently.
    """
    if not np.isfinite(stature_m) or not 1.2 <= stature_m <= 2.3:
        raise ValueError("stature_m must be finite and plausible")
    root = np.asarray(root_xy, dtype=float)
    if root.shape != (2,) or not np.isfinite(root).all():
        raise ValueError("root_xy must contain two finite court coordinates")
    visible = _visible_keypoints(row, minimum_confidence)
    if len(visible) < 5 or not any(name in visible for name in ("left_ankle", "right_ankle")):
        return None
    if not any(name in visible for name in ("left_hip", "right_hip")):
        return None

    airborne, support = estimate_airborne_height(row, stature_m, visible)
    height_targets = _target_heights(row, stature_m, visible, airborne)
    names = tuple(visible)
    centres, directions, initial = [], [], []
    for name in names:
        target = np.asarray([root[0], root[1], height_targets[name]], dtype=float)
        centre, direction = camera_ray(projection, visible[name][:2], target)
        centres.append(centre)
        directions.append(direction)
        initial.append(max(0.05, float(np.dot(target - centre, direction))))
    centres_array = np.asarray(centres)
    directions_array = np.asarray(directions)
    initial_array = np.asarray(initial)
    name_to_index = {name: index for index, name in enumerate(names)}

    def positions(depths: np.ndarray) -> np.ndarray:
        return centres_array + depths[:, None] * directions_array

    def residuals(depths: np.ndarray) -> np.ndarray:
        points = positions(depths)
        residual: list[float] = []
        for first, second, ratio in LIMB_PRIORS:
            if first not in name_to_index or second not in name_to_index:
                continue
            distance = np.linalg.norm(points[name_to_index[first]] - points[name_to_index[second]])
            residual.append(float((distance - ratio * stature_m) / (0.035 * stature_m)))

        hips = [
            points[name_to_index[name]]
            for name in ("left_hip", "right_hip")
            if name in name_to_index
        ]
        hip_centre = np.mean(hips, axis=0)
        residual.extend(((hip_centre[:2] - root) / 0.16).tolist())
        residual.append(float((hip_centre[2] - 0.53 * stature_m - airborne) / 0.14))

        ankles = [
            points[name_to_index[name]]
            for name in ("left_ankle", "right_ankle")
            if name in name_to_index
        ]
        ankle_centre = np.mean(ankles, axis=0)
        residual.extend(((ankle_centre[:2] - root) / 0.18).tolist())
        for ankle in ankles:
            residual.append(
                float((ankle[2] - airborne) / (0.025 if support == "planted" else 0.06))
            )

        for name, index in name_to_index.items():
            confidence = float(visible[name][2])
            height_sigma = 0.16 + 0.14 * (1.0 - confidence)
            residual.append(float((points[index, 2] - height_targets[name]) / height_sigma))
            residual.append(float(min(0.0, points[index, 2]) / 0.025))
            residual.append(float(max(0.0, points[index, 2] - 1.25 * stature_m) / 0.05))
            if previous_local and name in previous_local:
                prior = np.asarray(previous_local[name][:3], dtype=float) + np.asarray(
                    [root[0], root[1], 0.0]
                )
                temporal_sigma = 0.24 if name.endswith("wrist") else 0.14
                residual.extend(((points[index] - prior) / temporal_sigma).tolist())
        return np.asarray(residual, dtype=float)

    solution = least_squares(
        residuals,
        initial_array,
        bounds=(np.full_like(initial_array, 0.05), np.full_like(initial_array, 120.0)),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=120,
    )
    fitted = positions(solution.x)
    fitted_ankles = [
        fitted[name_to_index[name]]
        for name in ("left_ankle", "right_ankle")
        if name in name_to_index
    ]
    foot_centre = np.mean(fitted_ankles, axis=0)
    airborne_override = airborne_root_continuity and support == "airborne_box_scale"
    if (
        np.linalg.norm(foot_centre[:2] - root) > MAXIMUM_FOOT_CENTRE_ROOT_DISTANCE_M
        and not airborne_override
    ):
        return None
    if (
        len(fitted_ankles) == 2
        and np.linalg.norm(fitted_ankles[0] - fitted_ankles[1]) > (MAXIMUM_STANCE_WIDTH_M)
        and not airborne_override
    ):
        return None
    # Ground support is a defining constraint.  The residual is intentionally
    # strong, then this exact snap removes millimetric numerical drift.  Since
    # all joints were optimized on rays, only snap when the ray-plane residual
    # is below a display-negligible 3 cm; otherwise fail closed.
    if support == "planted":
        for ankle_name in ("left_ankle", "right_ankle"):
            if ankle_name not in name_to_index:
                continue
            index = name_to_index[ankle_name]
            if abs(float(fitted[index, 2])) > 0.03:
                return None
            fitted[index, 2] = 0.0

    cost = residuals(solution.x)
    joints = {
        name: (
            float(fitted[index, 0]),
            float(fitted[index, 1]),
            float(fitted[index, 2]),
            float(visible[name][2]),
        )
        for name, index in name_to_index.items()
    }
    return LiftedPose(
        joints=joints,
        airborne_height_m=float(airborne),
        support=support,
        converged=bool(solution.success),
        cost_rms=float(np.sqrt(np.mean(cost**2))) if len(cost) else 0.0,
        root_xy=(float(root[0]), float(root[1])),
    )


def _support_for_frame(frame: PoseFrameInput, stature_m: float) -> str | None:
    visible = _visible_keypoints(frame.row, MIN_KEYPOINT_CONFIDENCE)
    if not any(name in visible for name in ("left_ankle", "right_ankle")):
        return None
    try:
        return estimate_airborne_height(frame.row, stature_m, visible)[1]
    except (KeyError, TypeError, ValueError):
        return None


def _bounded_root_path(
    frames: Sequence[PoseFrameInput],
    stature_m: float,
    serve_window: ServeWindow | None,
) -> tuple[list[np.ndarray], list[str | None], tuple[int, int] | None]:
    """Hold an airborne serve between planted endpoints and cap root motion.

    Detector-box roots remain useful while the player is planted.  During the
    airborne run containing serve contact, however, their apparent court depth
    is replaced by a smooth interpolation between the last planted observation
    before takeoff and the first planted observation after it.  The returned
    path is also rate-limited at the display contract's 0.15 m per frame.
    """
    roots = [np.asarray(frame.root_xy, dtype=float) for frame in frames]
    supports = [_support_for_frame(frame, stature_m) for frame in frames]
    if serve_window is None or not frames:
        return roots, supports, None

    contact_index = min(
        range(len(frames)), key=lambda index: abs(frames[index].frame - serve_window.contact_frame)
    )
    first_window = next(
        (index for index, frame in enumerate(frames) if frame.frame >= serve_window.start_frame),
        contact_index,
    )
    last_window = max(
        (index for index, frame in enumerate(frames) if frame.frame <= serve_window.end_frame),
        default=contact_index,
    )
    airborne = [
        index
        for index in range(first_window, last_window + 1)
        if supports[index] == "airborne_box_scale"
    ]
    if airborne:
        airborne_before = [index for index in airborne if index <= contact_index]
        takeoff = min(airborne_before or airborne)
        airborne_after = [index for index in airborne if index >= contact_index]
        airborne_end = max(airborne_after or airborne)
    else:
        takeoff = contact_index
        airborne_end = contact_index
    planted_before = [
        index for index in range(first_window, takeoff) if supports[index] == "planted"
    ]
    planted_after = [
        index for index in range(airborne_end + 1, last_window + 1) if supports[index] == "planted"
    ]
    left = planted_before[-1] if planted_before else first_window
    right = planted_after[0] if planted_after else last_window
    if right < left:
        left, right = first_window, last_window

    anchored = list(roots)
    span = max(1, right - left)
    for index in range(left, right + 1):
        fraction = (index - left) / span
        # Cubic smoothstep makes both support transitions velocity-continuous.
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        anchored[index] = roots[index].copy()
        anchored[index][1] = roots[left][1] + fraction * (roots[right][1] - roots[left][1])

    # Fail-safe rate limit over the declared serve interval, including its
    # transitions.  Outside the window the detector roots are untouched.
    for index in range(max(1, first_window), last_window + 1):
        delta = anchored[index] - anchored[index - 1]
        distance = float(np.linalg.norm(delta))
        if distance > MAXIMUM_ROOT_STEP_M:
            anchored[index] = anchored[index - 1] + delta * (MAXIMUM_ROOT_STEP_M / distance)
    for index in range(last_window - 1, first_window - 1, -1):
        delta = anchored[index] - anchored[index + 1]
        distance = float(np.linalg.norm(delta))
        if distance > MAXIMUM_ROOT_STEP_M:
            anchored[index] = anchored[index + 1] + delta * (MAXIMUM_ROOT_STEP_M / distance)
    return anchored, supports, (left, right)


def _serve_weight(frame: int, window: ServeWindow) -> float:
    contact = float(window.contact_frame)
    if frame <= contact:
        span = max(1.0, contact - window.start_frame)
        return float(np.clip((frame - window.start_frame) / span, 0.0, 1.0))
    span = max(1.0, window.end_frame - contact)
    return float(np.clip((window.end_frame - frame) / span, 0.0, 1.0))


def _complete_arm(
    joints: Mapping[str, Sequence[float]], hand: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    names = (f"{hand}_shoulder", f"{hand}_elbow", f"{hand}_wrist")
    if not all(name in joints for name in names):
        return None
    return tuple(np.asarray(joints[name][:3], dtype=float) for name in names)  # type: ignore[return-value]


def _striking_hand(
    joints: Mapping[str, Sequence[float]], contact_xyz: Sequence[float] | None
) -> str | None:
    target = None if contact_xyz is None else np.asarray(contact_xyz, dtype=float)
    candidates = []
    for hand in ("left", "right"):
        arm = _complete_arm(joints, hand)
        if arm is None:
            continue
        wrist = arm[2]
        score = float(np.linalg.norm(wrist - target)) if target is not None else -float(wrist[2])
        candidates.append((score, hand))
    return min(candidates)[1] if candidates else None


def _stabilize_torso(
    joints: Mapping[str, Sequence[float]],
    stature_m: float,
    weight: float,
    court_forward_y: float,
) -> dict[str, tuple[float, float, float, float]]:
    output = {name: tuple(float(value) for value in point) for name, point in joints.items()}
    if weight <= 0.0:
        return output
    hips = [np.asarray(output[name][:3]) for name in ("left_hip", "right_hip") if name in output]
    shoulders = [
        np.asarray(output[name][:3])
        for name in ("left_shoulder", "right_shoulder")
        if name in output
    ]
    if not hips or not shoulders:
        return output
    hip_centre = np.mean(hips, axis=0)
    shoulder_centre = np.mean(shoulders, axis=0)
    lateral_offset = float(
        np.clip(
            shoulder_centre[0] - hip_centre[0],
            -SERVE_TORSO_LATERAL_LIMIT_M,
            SERVE_TORSO_LATERAL_LIMIT_M,
        )
    )
    target_xy = np.asarray(
        [
            hip_centre[0] + lateral_offset,
            hip_centre[1] + math.copysign(SERVE_TORSO_FORWARD_OFFSET_M, court_forward_y),
        ]
    )
    shift = (target_xy - shoulder_centre[:2]) * weight
    shoulder_height_shift = (hip_centre[2] + 0.288 * stature_m - shoulder_centre[2]) * weight
    for name in ("nose", "left_shoulder", "right_shoulder"):
        if name not in output:
            continue
        point = np.asarray(output[name][:3], dtype=float)
        point[:2] += shift
        point[2] += shoulder_height_shift
        output[name] = (*point.tolist(), output[name][3])
    return output


def _anchor_contact_arm(
    joints: Mapping[str, Sequence[float]],
    stature_m: float,
    contact_xyz: Sequence[float],
    hand: str,
) -> tuple[dict[str, tuple[float, float, float, float]], float]:
    """Place a physical two-link arm and collinear racket at serve contact."""
    output = {name: tuple(float(value) for value in point) for name, point in joints.items()}
    arm = _complete_arm(output, hand)
    target = np.asarray(contact_xyz, dtype=float)
    if arm is None or target.shape != (3,) or not np.isfinite(target).all():
        return output, math.inf
    shoulder, observed_elbow, _observed_wrist = arm
    upper = 0.186 * stature_m
    forearm = 0.146 * stature_m
    shoulder_to_face = target - shoulder
    distance_to_face = float(np.linalg.norm(shoulder_to_face))
    if distance_to_face < 1e-8:
        return output, math.inf
    axis = shoulder_to_face / distance_to_face
    nominal_racket = min(MAXIMUM_RACKET_FACE_DISTANCE_M, RACKET_FACE_FOREARM_SCALE * forearm)
    minimum_racket = 0.30
    racket_length = float(
        np.clip(
            distance_to_face - 0.97 * (upper + forearm),
            minimum_racket,
            MAXIMUM_RACKET_FACE_DISTANCE_M,
        )
    )
    racket_length = max(nominal_racket, racket_length)
    wrist = target - racket_length * axis
    shoulder_to_wrist = wrist - shoulder
    wrist_distance = float(np.linalg.norm(shoulder_to_wrist))
    maximum_reach = upper + forearm
    if wrist_distance > maximum_reach:
        # A monocular shoulder can remain too far from the fitted contact. Keep
        # its court-depth coordinate fixed: moving it toward the ball in depth
        # recreates the side-dependent torso lean that this serve pass removes.
        # The minimum correction in the lateral/vertical plane normally raises
        # the shoulder along the overhead contact line.
        desired_face_distance = 0.995 * maximum_reach + racket_length
        depth_gap = float(target[1] - shoulder[1])
        available_plane_radius = math.sqrt(
            max(0.0, desired_face_distance * desired_face_distance - depth_gap * depth_gap)
        )
        plane_delta = target[[0, 2]] - shoulder[[0, 2]]
        plane_distance = float(np.linalg.norm(plane_delta))
        if plane_distance > available_plane_radius and plane_distance > 1e-8:
            shoulder[[0, 2]] = target[[0, 2]] - (
                available_plane_radius * plane_delta / plane_distance
            )
        output[f"{hand}_shoulder"] = (*shoulder.tolist(), output[f"{hand}_shoulder"][3])
        shoulder_to_face = target - shoulder
        distance_to_face = float(np.linalg.norm(shoulder_to_face))
        axis = shoulder_to_face / distance_to_face
        racket_length = float(
            np.clip(
                distance_to_face - 0.97 * maximum_reach,
                minimum_racket,
                MAXIMUM_RACKET_FACE_DISTANCE_M,
            )
        )
        racket_length = max(nominal_racket, racket_length)
        wrist = target - racket_length * axis
        shoulder_to_wrist = wrist - shoulder
        wrist_distance = float(np.linalg.norm(shoulder_to_wrist))
    wrist_distance = float(
        np.clip(wrist_distance, abs(upper - forearm) + 1e-6, maximum_reach - 1e-6)
    )
    wrist = shoulder + wrist_distance * axis

    along = (upper * upper - forearm * forearm + wrist_distance * wrist_distance) / (
        2.0 * wrist_distance
    )
    radius = math.sqrt(max(0.0, upper * upper - along * along))
    observed_offset = observed_elbow - (shoulder + np.dot(observed_elbow - shoulder, axis) * axis)
    offset_norm = float(np.linalg.norm(observed_offset))
    if offset_norm < 1e-8:
        basis = np.cross(axis, np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(basis) < 1e-8:
            basis = np.asarray([1.0, 0.0, 0.0])
        observed_offset = basis
        offset_norm = float(np.linalg.norm(observed_offset))
    elbow = shoulder + along * axis + radius * observed_offset / offset_norm
    confidence = min(
        output[f"{hand}_shoulder"][3],
        output[f"{hand}_elbow"][3],
        output[f"{hand}_wrist"][3],
    )
    output[f"{hand}_elbow"] = (*elbow.tolist(), confidence)
    output[f"{hand}_wrist"] = (*wrist.tolist(), confidence)
    output["racket_grip"] = (*wrist.tolist(), confidence)
    output["racket_head"] = (*target.tolist(), confidence)
    return output, float(np.linalg.norm(np.asarray(output["racket_head"][:3]) - target))


def lift_pose_sequence(
    frames: Sequence[PoseFrameInput],
    stature_m: float,
    *,
    serve_window: ServeWindow | None = None,
    minimum_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> dict[int, LiftedPose]:
    """Lift a chronological track with serve-specific root and arm continuity."""
    ordered = sorted(frames, key=lambda item: item.frame)
    if len({frame.frame for frame in ordered}) != len(ordered):
        raise ValueError("pose sequence contains duplicate frame numbers")
    roots, _supports, anchored_span = _bounded_root_path(ordered, stature_m, serve_window)
    lifted_by_index: dict[int, LiftedPose] = {}
    previous_local: dict[str, Sequence[float]] | None = None
    previous_frame: int | None = None
    for index, (frame_input, root) in enumerate(zip(ordered, roots, strict=True)):
        if previous_frame is None or frame_input.frame != previous_frame + 1:
            previous_local = None
        lifted = lift_pose_frame(
            frame_input.row,
            frame_input.projection,
            root,
            stature_m,
            previous_local=previous_local,
            minimum_confidence=minimum_confidence,
        )
        previous_frame = frame_input.frame
        if lifted is None:
            previous_local = None
            continue
        step = None
        if index:
            step = float(np.linalg.norm(root - roots[index - 1]))
        lifted = replace(lifted, root_step_m=step)
        lifted_by_index[index] = lifted
        previous_local = {
            name: (
                point[0] - root[0],
                point[1] - root[1],
                point[2],
                point[3],
            )
            for name, point in lifted.joints.items()
        }

    # A serve-contact exposure can fail the old stance/root gate precisely
    # because both feet are airborne.  Fill only short, bracketed gaps inside
    # the declared window from neighboring inferred poses; native frames and
    # timestamps are preserved and longer gaps remain abstentions.
    if serve_window is not None:
        for index, frame_input in enumerate(ordered):
            if index in lifted_by_index or not (
                serve_window.start_frame <= frame_input.frame <= serve_window.end_frame
            ):
                continue
            left = max(
                (candidate for candidate in lifted_by_index if candidate < index), default=-1
            )
            right = min(
                (candidate for candidate in lifted_by_index if candidate > index),
                default=len(ordered),
            )
            if left < 0 or right >= len(ordered) or right - left > 3:
                continue
            if ordered[right].frame - ordered[left].frame > 3:
                continue
            left_pose = lifted_by_index[left]
            right_pose = lifted_by_index[right]
            fraction = (frame_input.frame - ordered[left].frame) / (
                ordered[right].frame - ordered[left].frame
            )
            shared = set(left_pose.joints) & set(right_pose.joints)
            joints = {
                name: tuple(
                    float(a + fraction * (b - a))
                    for a, b in zip(left_pose.joints[name], right_pose.joints[name], strict=True)
                )
                for name in shared
            }
            if len(joints) < 5:
                continue
            step = float(np.linalg.norm(roots[index] - roots[index - 1])) if index else None
            lifted_by_index[index] = LiftedPose(
                joints=joints,
                airborne_height_m=float(
                    left_pose.airborne_height_m
                    + fraction * (right_pose.airborne_height_m - left_pose.airborne_height_m)
                ),
                support=_supports[index] or "serve_gap_interpolation",
                converged=left_pose.converged and right_pose.converged,
                cost_rms=float(
                    left_pose.cost_rms + fraction * (right_pose.cost_rms - left_pose.cost_rms)
                ),
                method="camera_ray_serve_gap_interpolation_v2",
                root_xy=(float(roots[index][0]), float(roots[index][1])),
                root_step_m=step,
            )

    if serve_window is None or not lifted_by_index:
        return {ordered[index].frame: lifted for index, lifted in lifted_by_index.items()}
    effective_window = serve_window
    if anchored_span is not None and ordered[anchored_span[1]].frame > serve_window.contact_frame:
        effective_window = replace(
            serve_window,
            end_frame=ordered[anchored_span[1]].frame,
        )
    rounded_contact = int(math.floor(serve_window.contact_frame + 0.5))
    contact_index = min(
        lifted_by_index,
        key=lambda index: (
            abs(ordered[index].frame - rounded_contact),
            -ordered[index].frame,
        ),
    )
    contact_pose = lifted_by_index[contact_index]
    hand = _striking_hand(contact_pose.joints, serve_window.contact_xyz)
    for index, lifted in list(lifted_by_index.items()):
        frame_number = ordered[index].frame
        in_window = effective_window.start_frame <= frame_number <= effective_window.end_frame
        if not in_window:
            continue
        weight = _serve_weight(frame_number, effective_window)
        if index == contact_index:
            # Contacts live on the native fractional timebase. The selected
            # native anchor frame must nevertheless receive the full posture
            # correction instead of a side-biased fraction of it.
            weight = 1.0
        joints = _stabilize_torso(lifted.joints, stature_m, weight, serve_window.court_forward_y)
        anchor_error = None
        if index == contact_index and hand is not None and serve_window.contact_xyz is not None:
            joints, anchor_error = _anchor_contact_arm(
                joints, stature_m, serve_window.contact_xyz, hand
            )
        phase = "serve"
        if anchored_span is not None:
            phase = (
                "serve_airborne_span" if anchored_span[0] <= index <= anchored_span[1] else "serve"
            )
        lifted_by_index[index] = replace(
            lifted,
            joints=joints,
            method="camera_ray_serve_continuity_kinematic_v2",
            serve_phase=phase,
            striking_hand=hand,
            contact_anchor_error_m=anchor_error,
        )
    return {ordered[index].frame: lifted for index, lifted in lifted_by_index.items()}


def torso_plane(
    joints: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return centroid and unit normal of the least-squares torso plane."""
    points = [
        np.asarray(joints[name][:3], dtype=float)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if name in joints
    ]
    if len(points) < 3:
        return None
    values = np.asarray(points)
    centre = values.mean(axis=0)
    _, _, vectors = np.linalg.svd(values - centre, full_matrices=False)
    normal = vectors[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    return centre, normal / norm


def wrist_body_plane_distance(
    joints: Mapping[str, Sequence[float]], wrist_name: str
) -> float | None:
    """Absolute metric distance of a wrist from the fitted torso plane."""
    plane = torso_plane(joints)
    if plane is None or wrist_name not in joints:
        return None
    centre, normal = plane
    return abs(float(np.dot(np.asarray(joints[wrist_name][:3]) - centre, normal)))


def add_racket_segment(
    joints: Mapping[str, Sequence[float]],
    *,
    contact_xyz: Sequence[float] | None = None,
    hand: str | None = None,
    forearm_scale: float = RACKET_FACE_FOREARM_SCALE,
    maximum_face_distance_m: float = MAXIMUM_RACKET_FACE_DISTANCE_M,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, Any]]:
    """Add a short grip-to-face segment, optionally ending at an accepted contact.

    A fitted contact is permitted only when the caller has already established
    point acceptance.  Without it, the highest-confidence complete arm is
    extended by the player ledger's calibrated 1.5-forearm model (capped at a
    physical 0.55 m wrist-to-face distance) and carries its conservative 75 px
    face uncertainty.
    """
    output = {name: tuple(float(value) for value in point) for name, point in joints.items()}
    if "racket_grip" in output and "racket_head" in output:
        selected_hand = hand or _striking_hand(output, contact_xyz)
        wrist_name = None if selected_hand is None else f"{selected_hand}_wrist"
        return output, {
            "status": "supported",
            "hand": selected_hand,
            "source": "serve_contact_arm_ik",
            "sigma_px": None,
            "wrist_body_plane_distance_m": (
                None if wrist_name is None else wrist_body_plane_distance(output, wrist_name)
            ),
        }
    candidates = []
    target = None if contact_xyz is None else np.asarray(contact_xyz, dtype=float)
    if hand not in {None, "left", "right"}:
        raise ValueError("hand must be left, right, or None")
    hands = (hand,) if hand is not None else ("left", "right")
    for candidate_hand in hands:
        wrist_name = f"{candidate_hand}_wrist"
        elbow_name = f"{candidate_hand}_elbow"
        if wrist_name not in output or elbow_name not in output:
            continue
        wrist = np.asarray(output[wrist_name][:3])
        elbow = np.asarray(output[elbow_name][:3])
        confidence = min(output[wrist_name][3], output[elbow_name][3])
        score = float(np.linalg.norm(wrist - target)) if target is not None else -confidence
        candidates.append((score, candidate_hand, wrist, elbow, confidence))
    if not candidates:
        return output, {"status": "abstain", "reason": "no_complete_arm"}
    _, hand, wrist, elbow, confidence = min(candidates, key=lambda item: item[0])
    if target is not None and target.shape == (3,) and np.isfinite(target).all():
        face = target
        source = "accepted_fitter_contact_xyz"
        sigma_px = None
    else:
        direction = wrist - elbow
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return output, {"status": "abstain", "reason": "degenerate_forearm"}
        face_distance = min(float(maximum_face_distance_m), float(forearm_scale) * norm)
        face = wrist + face_distance * direction / norm
        source = "wrist_forearm_3d_extrapolation"
        sigma_px = RACKET_FACE_SIGMA_PX
    output["racket_grip"] = (*wrist.tolist(), float(confidence))
    output["racket_head"] = (*face.tolist(), float(confidence))
    return output, {
        "status": "supported",
        "hand": hand,
        "source": source,
        "sigma_px": sigma_px,
        "wrist_body_plane_distance_m": wrist_body_plane_distance(output, f"{hand}_wrist"),
    }
