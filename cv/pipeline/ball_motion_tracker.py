"""Associate native ball candidates with a homography-aware IMM motion filter.

The tracker keeps detector observations immutable.  It clusters coincident observations,
then associates one cluster per frame with a gated innovation likelihood.  Two interacting
Kalman modes cover ordinary flight and short impulses:

* ``ballistic`` transports image position and velocity through consecutive image-to-court
  homographies.  Its image acceleration is the projection of constant 9.81 m/s² gravity.
* ``impulse`` uses the same camera transport with wider process noise and no gravity term,
  allowing a contact or bounce to change velocity without admitting arbitrary distractors.

Each emitted row contains the innovation covariance used for association, the selected mode,
and complete detector/crop provenance.  Mode changes are written separately as label-free event
proposals; they are proposals, not event types.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.track_artifact import write_track_artifact


GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    score: float
    rank: int
    sources: tuple[str, ...]
    crop_provenance: tuple[str, ...] = ()

    @property
    def detector_count(self) -> int:
        detectors = set()
        for source in self.sources:
            if "tracknet" in source.lower():
                detectors.add("tracknetv2")
            if "wasb" in source.lower():
                detectors.add("wasb")
        return len(detectors)

    @property
    def is_crop(self) -> bool:
        return any("crop" in source.lower() for source in self.sources)

    @property
    def is_guide(self) -> bool:
        return "coarse_lock" in self.sources


@dataclass
class FilterMode:
    mean: np.ndarray
    covariance: np.ndarray
    probability: float


@dataclass(frozen=True)
class MotionConfig:
    cluster_radius_native: float = 8.0
    ballistic_gate_d2: float = 16.0
    impulse_gate_d2: float = 25.0
    maximum_misses: int = 12
    minimum_restart_detectors: int = 2
    ballistic_acceleration_noise: float = 1.25
    impulse_acceleration_noise: float = 18.0
    initial_position_std: float = 8.0
    initial_velocity_std: float = 22.0
    detector_agreement_bonus: float = 2.4
    crop_prior_bonus: float = 0.25
    score_prior_weight: float = 1.2
    rank_prior_weight: float = 0.18
    guide_prior_bonus: float = 1.4
    guide_candidate_gate_native: float = 4.0
    guide_replacement_margin: float = 2.0
    ballistic_to_impulse: float = 0.035
    impulse_to_ballistic: float = 0.55
    hotspot_bin_native: float = 32.0
    hotspot_stationary_bin_native: float = 16.0
    hotspot_min_frames: int = 12
    hotspot_fraction: float = 0.03
    hotspot_consecutive_frames: int = 6
    hotspot_motion_reach: int = 24
    hotspot_motion_step_native: float = 70.0
    hotspot_escape_distance_native: float = 40.0
    # Crop-first association. Off by default: it is only correct when the crop pass runs a
    # detector whose measured accuracy actually beats the coarse lock's, which the production
    # tennis checkpoints' does not. See crop_first_config().
    crop_first: bool = False
    crop_measurement_std_native: float = 2.5
    guide_measurement_std_native: float = 3.0
    crop_candidate_gate_native: float = 4.0
    # Optional image-exit admission. A missing ball must not bootstrap on a
    # motion-rejected guide until native detector evidence supports reentry.
    qualified_image_reentry: bool = False
    reentry_boundary_margin_native: float = 32.0
    reentry_support_frames: int = 3
    # Optional primary-ball ownership (cv.pipeline.ball_ownership). Off by default: the
    # emitted rows are then byte-identical to the unflagged tracker. On, every filter
    # restart opens a segment, segments are chained by a camera-transported position join,
    # chains proven static under a reliable camera are withheld from the consumed track,
    # track_id carries the chain id, and every tracked row is preserved in the ownership
    # side artifact with its owner and join verdict.
    primary_ball_ownership: bool = False
    # Longest gap across which a segment may rejoin an earlier one. Equal to the hotspot
    # motion reach, the tracker's existing horizon for motion continuity.
    ownership_join_gap_frames: int = 24
    # Under an unreliable (held) camera the join runs in raw image coordinates with a pan
    # allowance per frame of gap; beyond three frames that allowance swamps the join.
    ownership_raw_join_gap_frames: int = 3
    # Optional two-pass guide-detour sequence association
    # (cv.pipeline.guide_sequence_association). Off by default: the emitted rows and
    # proposals are then byte-identical to the unflagged tracker. On, an unchanged first
    # association pass supplies the segment/restart/join diagnostics, short guide-opened
    # detours that the *same* preceding primary immediately rejoins are bracketed, the full
    # original measured pool inside those brackets is searched for a bounded sequence, and a
    # qualified sequence is re-associated by a second ordinary pass.
    guide_sequence_association: bool = False
    # Longest bracketed detour considered, in frames of the original cadence. Fixed, never
    # per source: a longer bypass is not a detour this mechanism can bracket.
    sequence_max_detour_frames: int = 6
    # Deterministic search bound, fixed before any data is read.
    sequence_beam_width: int = 8
    sequence_max_branch: int = 6
    # Refuse a candidate that stays inside one player's torso. Off by default:
    # the measured pool is then exactly the unflagged tracker's. On, a chain of
    # at least torso_lock_frames inside the torso is dropped and the other blob
    # is what the filter can reacquire. A one- or two-frame crossing is kept.
    torso_lock_rejection: bool = False
    torso_lock_frames: int = 6
    # Count that run in distinct frames rather than pooled rows (torso_lock.RUN_IN_FRAMES).
    torso_lock_run_in_frames: bool = False
    # Keep a small-torso chain joined at both ends to outside candidates (torso_lock.KEEP_JOINED).
    torso_lock_keep_joined: bool = False
    # Carry the filter across frames the coarse lock never reached. Off by default: the
    # composed track then exists only where the lock has a row. On, the IMM gate and the miss
    # counter bound divergence there, and a restart on such a frame needs two agreeing
    # detectors. Only change 4b of crop_first_config; sigmas, priors and gates are unchanged.
    lock_gap_carry: bool = False


# Crop-first constants, set from the measured accuracy of the native-crop fine-tune rather
# than chosen. On the sealed transfer split the fine-tuned crop-only decode has a median error
# of 1.009 native px against the composed (coarse-lock dominated) track's 1.438; 2.5 * 1.009 /
# 1.438 = 1.75 is the crop measurement sigma that ratio implies, against the guide's 3.0.
CROP_FIRST_MEASUREMENT_STD_NATIVE = 1.75
# A crop observation must be able to win the tie against the guide, so its prior bonus goes
# above guide_prior_bonus (1.4) by the same margin the guide used to hold over it.
CROP_FIRST_PRIOR_BONUS = 2.6
# The crop estimator is the more accurate one, so confining its candidates to +-4 native px of
# the *less* accurate coarse lock is backwards. 24 px is the same radius build_tracker_crops
# uses to call a candidate "rejected by the tracker".
CROP_FIRST_CANDIDATE_GATE_NATIVE = 24.0


def crop_first_config(
    base: MotionConfig | None = None,
    lock_is_crop: bool = False,
) -> MotionConfig:
    """Association tuned for a crop pass that is measurably better than the coarse lock.

    Four changes, each with a measured reason:

    1. crop measurement sigma 2.5 -> 1.75, the ratio the sealed-set medians imply;
    2. crop prior bonus 0.25 -> 2.6, above the guide's 1.4, so a crop wins a tie;
    3. guide replacement margin 2.0 -> 0.0, so a better crop is not reverted to the guide;
    4. crop candidates admitted within 24 native px of the guide instead of 4, and a frame the
       coarse lock never reached is no longer skipped.

    ``lock_is_crop`` additionally gives the guide stream the crop measurement sigma, for the
    composition that feeds the crop-only decode in as the lock.

    Everything else is untouched, including the hotspot suppression and the fail-safe restart.
    """
    config = replace(
        base or MotionConfig(),
        crop_first=True,
        crop_measurement_std_native=CROP_FIRST_MEASUREMENT_STD_NATIVE,
        crop_prior_bonus=CROP_FIRST_PRIOR_BONUS,
        guide_replacement_margin=0.0,
        crop_candidate_gate_native=CROP_FIRST_CANDIDATE_GATE_NATIVE,
    )
    if lock_is_crop:
        # The crop-first *composition* hands the tracker the crop-only decode as its lock
        # stream. That stream is a crop estimate wearing the coarse_lock name, so it must not
        # be given the coarse pass's measurement sigma.
        config = replace(config, guide_measurement_std_native=CROP_FIRST_MEASUREMENT_STD_NATIVE)
    return config


@dataclass(frozen=True)
class Geometry:
    homographies: dict[tuple[str, int], np.ndarray]
    projections: dict[tuple[str, int], np.ndarray]
    source: str
    # Per-frame registration reliability from the artifact's own flag. A frame absent from
    # this map is treated as reliable, matching artifacts that carry no flag. Ownership
    # decisions consult it; the IMM transport itself is unchanged.
    reliable: dict[tuple[str, int], bool] = field(default_factory=dict)

    def is_reliable(self, key: tuple[str, int]) -> bool:
        return key in self.homographies and self.reliable.get(key, True)


def frame_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def _normalise_homography(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    scale = value[2, 2]
    return value / scale if abs(scale) > 1e-12 else value / np.linalg.norm(value)


def ground_homography_from_projection(projection: np.ndarray) -> np.ndarray:
    """Return image-to-court H from a court-to-image 3x4 projection."""
    projection = np.asarray(projection, dtype=float)
    if projection.shape != (3, 4):
        raise ValueError("projection must be 3x4")
    return _normalise_homography(np.linalg.inv(projection[:, [0, 1, 3]]))


def load_geometry(
    homography_path: Path | None,
    projection_path: Path | None,
) -> Geometry:
    projections: dict[tuple[str, int], np.ndarray] = {}
    reliable: dict[tuple[str, int], bool] = {}
    if projection_path is not None and projection_path.is_file():
        data = np.load(projection_path, allow_pickle=False)
        projections = {
            (str(clip), int(frame)): np.asarray(projection, dtype=float)
            for clip, frame, projection in zip(
                data["clips"], data["frames"], data["P"], strict=True
            )
        }
        reliable = _reliability_flags(data)
    if homography_path is not None and homography_path.is_file():
        data = np.load(homography_path, allow_pickle=False)
        homographies = {
            (str(clip), int(frame)): _normalise_homography(homography)
            for clip, frame, homography in zip(
                data["clips"], data["frames"], data["H"], strict=True
            )
        }
        source = homography_path.name
        reliable = _reliability_flags(data) or reliable
    elif projections:
        homographies = {
            key: ground_homography_from_projection(projection)
            for key, projection in projections.items()
        }
        source = f"{projection_path.name}:derived_ground_homography"
    else:
        raise FileNotFoundError(
            "motion tracking requires court_H_per_frame_v1.npz or camera_P_per_frame_v1.npz"
        )
    return Geometry(homographies, projections, source, reliable)


def _reliability_flags(data) -> dict[tuple[str, int], bool]:
    if "reliable" not in data.files:
        return {}
    return {
        (str(clip), int(frame)): bool(flag)
        for clip, frame, flag in zip(data["clips"], data["frames"], data["reliable"], strict=True)
    }


def _artifact_scale(path: Path) -> tuple[float, float]:
    """Native scale from the artifact's own declaration; never from its name."""
    return res.coordinate_scale(path, columns=("x", "y"))


def load_observations(
    paths: list[Path],
    source_names: list[str],
    clips: set[str] | None = None,
) -> dict[str, dict[int, list[Observation]]]:
    pooled: dict[str, dict[int, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for path, source in zip(paths, source_names, strict=True):
        scale_x, scale_y = _artifact_scale(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                clip = row["clip"]
                if clips is not None and clip not in clips:
                    continue
                pooled[clip][frame_number(row["frame"])].append(
                    Observation(
                        x=float(row["x"]) * scale_x,
                        y=float(row["y"]) * scale_y,
                        score=float(row.get("score", 0.0) or 0.0),
                        rank=int(row.get("rank", 0) or 0),
                        sources=(source,),
                        crop_provenance=tuple(filter(None, (row.get("crop_provenance", ""),))),
                    )
                )
    return pooled


def merge_observations(
    observations: list[Observation],
    radius: float,
) -> list[Observation]:
    """Cluster coincident pass outputs while retaining every contributing source."""
    clusters: list[list[Observation]] = []
    for observation in sorted(observations, key=lambda item: (item.rank, -item.score)):
        nearest = None
        nearest_distance = math.inf
        for index, cluster in enumerate(clusters):
            if observation.is_guide or any(item.is_guide for item in cluster):
                continue
            center_x = sum(item.x for item in cluster) / len(cluster)
            center_y = sum(item.y for item in cluster) / len(cluster)
            distance = math.hypot(observation.x - center_x, observation.y - center_y)
            if distance <= radius and distance < nearest_distance:
                nearest = index
                nearest_distance = distance
        if nearest is None:
            clusters.append([observation])
        else:
            clusters[nearest].append(observation)
    merged = []
    for cluster in clusters:
        weights = np.asarray([max(item.score, 0.02) for item in cluster], dtype=float)
        weights /= weights.sum()
        merged.append(
            Observation(
                x=float(
                    sum(weight * item.x for weight, item in zip(weights, cluster, strict=True))
                ),
                y=float(
                    sum(weight * item.y for weight, item in zip(weights, cluster, strict=True))
                ),
                score=max(item.score for item in cluster),
                rank=min(item.rank for item in cluster),
                sources=tuple(sorted({source for item in cluster for source in item.sources})),
                crop_provenance=tuple(
                    sorted({value for item in cluster for value in item.crop_provenance})
                ),
            )
        )
    return merged


def suppress_static_hotspots(
    frames: dict[int, list[Observation]],
    config: MotionConfig,
) -> dict[int, list[Observation]]:
    """Suppress recurrent graphics while preserving supported observations that move away."""
    counts: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    stationary_counts: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for frame, observations in frames.items():
        for observation in observations:
            for source in observation.sources:
                counts[
                    source,
                    round(observation.x / config.hotspot_bin_native),
                    round(observation.y / config.hotspot_bin_native),
                ].add(frame)
                stationary_counts[
                    source,
                    round(observation.x / config.hotspot_stationary_bin_native),
                    round(observation.y / config.hotspot_stationary_bin_native),
                ].add(frame)
    threshold = max(config.hotspot_min_frames, math.ceil(len(frames) * config.hotspot_fraction))
    hotspots = {key for key, observed in counts.items() if len(observed) >= threshold}
    stationary_hotspots = set()
    for key, observed in stationary_counts.items():
        longest = current = 1
        for left, right in zip(sorted(observed), sorted(observed)[1:]):
            current = current + 1 if right == left + 1 else 1
            longest = max(longest, current)
        if longest >= config.hotspot_consecutive_frames:
            stationary_hotspots.add(key)

    def distance(left: Observation, right: Observation) -> float:
        return math.hypot(left.x - right.x, left.y - right.y)

    def has_motion_support(frame: int, observation: Observation) -> bool:
        if observation.detector_count < 2:
            return False
        for direction in (-1, 1):
            frontier = [observation]
            for step in range(1, config.hotspot_motion_reach + 1):
                neighbors = [
                    neighbor
                    for neighbor in frames.get(frame + direction * step, [])
                    if neighbor.detector_count >= 2
                    and any(
                        distance(prior, neighbor) <= config.hotspot_motion_step_native
                        for prior in frontier
                    )
                ]
                if not neighbors:
                    break
                if any(
                    distance(observation, neighbor) >= config.hotspot_escape_distance_native
                    for neighbor in neighbors
                ):
                    return True
                frontier = neighbors
        return False

    output = {}
    for frame, observations in frames.items():
        kept = []
        for observation in observations:
            if observation.is_guide:
                kept.append(observation)
                continue
            is_hotspot = any(
                (
                    source,
                    round(observation.x / config.hotspot_bin_native),
                    round(observation.y / config.hotspot_bin_native),
                )
                in hotspots
                or (
                    source,
                    round(observation.x / config.hotspot_stationary_bin_native),
                    round(observation.y / config.hotspot_stationary_bin_native),
                )
                in stationary_hotspots
                for source in observation.sources
            )
            if not is_hotspot or has_motion_support(frame, observation):
                kept.append(observation)
        output[frame] = kept
    return output


def _apply_h(homography: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = homography @ np.asarray([point[0], point[1], 1.0], dtype=float)
    if abs(value[2]) < 1e-12:
        return np.asarray(point, dtype=float)
    return value[:2] / value[2]


def _homography_jacobian(homography: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y = float(point[0]), float(point[1])
    h = homography
    denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    numerator_x = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    numerator_y = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    return np.asarray(
        [
            [
                (h[0, 0] * denominator - h[2, 0] * numerator_x) / denominator**2,
                (h[0, 1] * denominator - h[2, 1] * numerator_x) / denominator**2,
            ],
            [
                (h[1, 0] * denominator - h[2, 0] * numerator_y) / denominator**2,
                (h[1, 1] * denominator - h[2, 1] * numerator_y) / denominator**2,
            ],
        ]
    )


def camera_warp(
    previous_h: np.ndarray | None,
    current_h: np.ndarray | None,
) -> np.ndarray:
    if previous_h is None or current_h is None:
        return np.eye(3)
    return _normalise_homography(np.linalg.inv(current_h) @ previous_h)


def projected_gravity(
    position: np.ndarray,
    homography: np.ndarray | None,
    projection: np.ndarray | None,
    fps: float,
) -> np.ndarray:
    """Project constant court-space vertical gravity into native image px/frame²."""
    if homography is None or projection is None or fps <= 0:
        return np.asarray([0.0, 0.035], dtype=float)
    court_xy = _apply_h(homography, position)

    def project(z: float) -> np.ndarray:
        value = projection @ np.asarray([court_xy[0], court_xy[1], z, 1.0])
        return value[:2] / value[2]

    image_vertical_per_metre = project(1.0) - project(0.0)
    gravity = -GRAVITY_M_S2 * image_vertical_per_metre / fps**2
    magnitude = float(np.linalg.norm(gravity))
    if not np.isfinite(gravity).all() or magnitude > 3.0:
        return np.asarray([0.0, 0.035], dtype=float)
    return gravity


def _process_noise(acceleration_std: float) -> np.ndarray:
    base = np.asarray(
        [
            [0.25, 0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ]
    )
    return acceleration_std**2 * base


def predict_mode(
    mode: FilterMode,
    previous_h: np.ndarray | None,
    current_h: np.ndarray | None,
    gravity: np.ndarray,
    acceleration_std: float,
) -> FilterMode:
    transition = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    predicted = transition @ mode.mean
    predicted[:2] += 0.5 * gravity
    predicted[2:] += gravity
    covariance = transition @ mode.covariance @ transition.T + _process_noise(acceleration_std)
    warp = camera_warp(previous_h, current_h)
    position_before_warp = predicted[:2].copy()
    warped_position = _apply_h(warp, position_before_warp)
    warped_step = _apply_h(warp, position_before_warp + predicted[2:])
    jacobian = _homography_jacobian(warp, position_before_warp)
    transform = np.zeros((4, 4), dtype=float)
    transform[:2, :2] = jacobian
    transform[2:, 2:] = jacobian
    predicted[:2] = warped_position
    predicted[2:] = warped_step - warped_position
    covariance = transform @ covariance @ transform.T
    return FilterMode(predicted, covariance, mode.probability)


def measurement_covariance(
    observation: Observation,
    config: MotionConfig = MotionConfig(),
) -> np.ndarray:
    if observation.is_crop:
        standard_deviation = config.crop_measurement_std_native
    elif observation.is_guide:
        standard_deviation = config.guide_measurement_std_native
    elif any("far_native" in source for source in observation.sources):
        standard_deviation = 4.0
    else:
        standard_deviation = 6.0
    standard_deviation /= math.sqrt(max(1, observation.detector_count))
    return np.eye(2) * standard_deviation**2


def innovation(
    mode: FilterMode,
    observation: Observation,
    config: MotionConfig = MotionConfig(),
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = np.asarray([observation.x, observation.y]) - mode.mean[:2]
    covariance = mode.covariance[:2, :2] + measurement_covariance(observation, config)
    covariance = (covariance + covariance.T) / 2.0
    inverse = np.linalg.pinv(covariance)
    distance = float(residual.T @ inverse @ residual)
    return residual, covariance, distance


def update_mode(
    mode: FilterMode,
    observation: Observation,
    mode_probability: float,
    config: MotionConfig = MotionConfig(),
) -> FilterMode:
    residual, covariance, _ = innovation(mode, observation, config)
    gain = mode.covariance[:, :2] @ np.linalg.pinv(covariance)
    mean = mode.mean + gain @ residual
    updated_covariance = mode.covariance - gain @ mode.covariance[:2, :]
    updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
    return FilterMode(mean, updated_covariance, mode_probability)


def _mix_modes(modes: dict[str, FilterMode]) -> tuple[np.ndarray, np.ndarray]:
    total = sum(mode.probability for mode in modes.values()) or 1.0
    mean = sum(mode.probability * mode.mean for mode in modes.values()) / total
    covariance = np.zeros((4, 4), dtype=float)
    for mode in modes.values():
        delta = mode.mean - mean
        covariance += mode.probability * (mode.covariance + np.outer(delta, delta)) / total
    return mean, covariance


def _transition_probabilities(
    modes: dict[str, FilterMode], config: MotionConfig
) -> dict[str, float]:
    ballistic = modes["ballistic"].probability
    impulse = modes["impulse"].probability
    return {
        "ballistic": ballistic * (1.0 - config.ballistic_to_impulse)
        + impulse * config.impulse_to_ballistic,
        "impulse": ballistic * config.ballistic_to_impulse
        + impulse * (1.0 - config.impulse_to_ballistic),
    }


def advance_modes(
    modes: dict[str, FilterMode],
    previous_h: np.ndarray | None,
    current_h: np.ndarray | None,
    previous_projection,
    fps: float,
    config: MotionConfig,
) -> dict[str, FilterMode]:
    """One IMM mix-and-predict step.

    This is the single predictor of the association: ``track_clip`` calls it, and so does the
    optional second pass in cv.pipeline.guide_sequence_association, so the two passes cannot
    drift apart in the gravity projection, the mixing or the transition probabilities.
    ``previous_projection`` is the projection of the *departure* frame, the same argument
    ``track_clip`` has always passed.
    """
    mixed_mean, mixed_covariance = _mix_modes(modes)
    transition_probabilities = _transition_probabilities(modes, config)
    gravity = projected_gravity(mixed_mean[:2], previous_h, previous_projection, fps)
    shared = FilterMode(mixed_mean, mixed_covariance, 1.0)
    predicted = {
        "ballistic": predict_mode(
            shared, previous_h, current_h, gravity, config.ballistic_acceleration_noise
        ),
        "impulse": predict_mode(
            shared, previous_h, current_h, np.zeros(2), config.impulse_acceleration_noise
        ),
    }
    for name, probability in transition_probabilities.items():
        predicted[name].probability = probability
    return predicted


def initialise_modes(observation: Observation, config: MotionConfig) -> dict[str, FilterMode]:
    mean = np.asarray([observation.x, observation.y, 0.0, 0.0], dtype=float)
    covariance = np.diag(
        [
            config.initial_position_std**2,
            config.initial_position_std**2,
            config.initial_velocity_std**2,
            config.initial_velocity_std**2,
        ]
    )
    return {
        "ballistic": FilterMode(mean.copy(), covariance.copy(), 0.97),
        "impulse": FilterMode(mean.copy(), covariance.copy(), 0.03),
    }


def observation_prior(observation: Observation, config: MotionConfig) -> float:
    return (
        config.detector_agreement_bonus * max(0, observation.detector_count - 1)
        + config.crop_prior_bonus * int(observation.is_crop)
        + config.guide_prior_bonus * int(observation.is_guide)
        + config.score_prior_weight * min(max(observation.score, 0.0), 1.0)
        - config.rank_prior_weight * observation.rank
    )


def _candidate_likelihood(
    modes: dict[str, FilterMode],
    observation: Observation,
    config: MotionConfig,
) -> tuple[float, dict[str, tuple[np.ndarray, np.ndarray, float, float]]]:
    details = {}
    likelihood = 0.0
    for name, mode in modes.items():
        residual, covariance, distance = innovation(mode, observation, config)
        gate = config.ballistic_gate_d2 if name == "ballistic" else config.impulse_gate_d2
        determinant = max(float(np.linalg.det(covariance)), 1e-12)
        component = (
            mode.probability * math.exp(-0.5 * distance) / math.sqrt(determinant)
            if distance <= gate
            else 0.0
        )
        details[name] = (residual, covariance, distance, component)
        likelihood += component
    if likelihood <= 0.0:
        return -math.inf, details
    return math.log(likelihood) + observation_prior(observation, config), details


def _choose_restart(observations: list[Observation], config: MotionConfig) -> Observation | None:
    supported = [
        observation
        for observation in observations
        if observation.detector_count >= config.minimum_restart_detectors or observation.is_guide
    ]
    if not supported:
        return None
    return max(supported, key=lambda observation: observation_prior(observation, config))


def _image_edge_distances(xy: np.ndarray) -> np.ndarray:
    return np.asarray([xy[0], res.NATIVE_SIZE.width - xy[0], xy[1], res.NATIVE_SIZE.height - xy[1]])


def _predicted_image_exit(
    previous: np.ndarray, predicted: np.ndarray, config: MotionConfig
) -> bool:
    """A recent native observation and its camera-compensated next prediction exit."""
    before = _image_edge_distances(previous)
    after = _image_edge_distances(predicted)
    return bool(
        np.any((before >= 0) & (before <= config.reentry_boundary_margin_native) & (after < 0))
    )


def _supported_image_reentry(
    frame: int,
    observations: dict[int, list[Observation]],
    config: MotionConfig,
    geometry: Geometry | None = None,
    clip: str = "",
    fps: float = 25.0,
    require_boundary: bool = True,
) -> list[Observation] | None:
    """Find a short inward sequence of original multi-detector candidates.

    This offline tracker already receives the complete native clip. Looking
    ahead qualifies the first sample; it does not manufacture intermediate
    pixels. Guide-only rows cannot supply the independent detector support.
    """
    count = config.reentry_support_frames
    chains = []
    for candidate in observations.get(frame, []):
        xy = np.asarray([candidate.x, candidate.y])
        edge_distances = _image_edge_distances(xy)
        if (
            candidate.is_guide
            or candidate.detector_count < config.minimum_restart_detectors
            or np.any(edge_distances < 0)
            or (require_boundary and min(edge_distances) > config.reentry_boundary_margin_native)
        ):
            continue
        edge = int(np.argmin(edge_distances))
        chain = [candidate]
        for offset in range(1, count):
            last = chain[-1]
            viable = [
                item
                for item in observations.get(frame + offset, [])
                if not item.is_guide
                and item.detector_count >= config.minimum_restart_detectors
                and np.all(_image_edge_distances(np.asarray([item.x, item.y])) >= 0)
                and math.hypot(item.x - last.x, item.y - last.y)
                <= config.hotspot_motion_step_native
            ]
            if not viable:
                break
            predicted = np.asarray([last.x, last.y])
            if len(chain) > 1:
                predicted += np.asarray([last.x - chain[-2].x, last.y - chain[-2].y])
            chosen = min(
                viable, key=lambda item: math.hypot(item.x - predicted[0], item.y - predicted[1])
            )
            chain.append(chosen)
        if len(chain) != count:
            continue
        inward = [_image_edge_distances(np.asarray([item.x, item.y]))[edge] for item in chain]
        if require_boundary and (
            min(np.diff(inward)) <= 0 or inward[-1] - inward[0] < config.cluster_radius_native
        ):
            continue
        # A camera pan alone can move a stationary edge distractor inward.
        # Conservatively retain ambiguity until the candidate has its own
        # motion. This can also defer a real nearly stationary ball revealed
        # by a pan; it is evidence qualification, not an identity guarantee.
        final_xy = np.asarray([chain[-1].x, chain[-1].y])
        if geometry is not None:
            final_xy = _apply_h(
                camera_warp(
                    geometry.homographies.get((clip, frame + count - 1)),
                    geometry.homographies.get((clip, frame)),
                ),
                final_xy,
            )
        if (
            np.linalg.norm(final_xy - np.asarray([chain[0].x, chain[0].y]))
            < config.cluster_radius_native
        ):
            continue
        mode = initialise_modes(chain[0], config)["ballistic"]
        mode.mean[2:] = _reentry_velocity(chain, frame, geometry, clip)
        for offset, item in enumerate(chain[1:], start=1):
            previous_h = geometry.homographies.get((clip, frame + offset - 1)) if geometry else None
            current_h = geometry.homographies.get((clip, frame + offset)) if geometry else None
            projection = geometry.projections.get((clip, frame + offset - 1)) if geometry else None
            gravity = projected_gravity(mode.mean[:2], previous_h, projection, fps)
            mode = predict_mode(
                mode, previous_h, current_h, gravity, config.ballistic_acceleration_noise
            )
            if innovation(mode, item, config)[2] > config.ballistic_gate_d2:
                break
            mode = update_mode(mode, item, 1.0, config)
        else:
            chains.append(chain)
    return max(
        chains,
        key=lambda chain: sum(observation_prior(item, config) for item in chain),
        default=None,
    )


def _reentry_velocity(
    chain: list[Observation], frame: int, geometry: Geometry | None, clip: str
) -> np.ndarray:
    """Seed motion in the first exposure's image coordinates, excluding camera pan."""
    first = np.asarray([chain[0].x, chain[0].y])
    second = np.asarray([chain[1].x, chain[1].y])
    if geometry is not None:
        second = _apply_h(
            camera_warp(
                geometry.homographies.get((clip, frame + 1)),
                geometry.homographies.get((clip, frame)),
            ),
            second,
        )
    return second - first


def prepare_observations(
    frame_observations: dict[int, list[Observation]],
    config: MotionConfig,
) -> dict[int, list[Observation]]:
    """The measured pool the association actually sees: merged, then static-suppressed.

    Named so a second pass can search the same pool the first pass associated over, rather
    than a separately rebuilt one.
    """
    merged = {
        frame: merge_observations(observations, config.cluster_radius_native)
        for frame, observations in frame_observations.items()
    }
    return suppress_static_hotspots(merged, config)


def track_clip(
    clip: str,
    frame_observations: dict[int, list[Observation]],
    geometry: Geometry,
    fps: float,
    config: MotionConfig = MotionConfig(),
    admission: dict[int, tuple[Observation, ...]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Associate one clip.

    ``admission`` is the optional second-pass input of
    cv.pipeline.guide_sequence_association: on the listed frames the candidate pool is
    replaced by the qualified sequence's own choice (an empty tuple is an explicit miss) and
    the guide holds no privilege, so the object the first pass reset onto cannot recapture
    the track inside the qualified window. Every other frame is associated unchanged.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frame_observations:
        return [], []
    merged = prepare_observations(frame_observations, config)
    admission = admission or {}
    has_guide_stream = any(
        observation.is_guide for observations in merged.values() for observation in observations
    )
    rows: list[dict] = []
    proposals: list[dict] = []
    modes: dict[str, FilterMode] | None = None
    misses = 0
    previous_regime: str | None = None
    previous_h: np.ndarray | None = None
    last_observation: tuple[np.ndarray, np.ndarray | None] | None = None
    waiting_for_reentry = False
    protect_reentry_until = -1
    reacquisition_needs_guide = False
    # Ownership instrumentation: which control-flow path opened the current segment.
    segment_id = -1
    restart_kind = "bootstrap"
    pending_restart_kind = "bootstrap"
    start, end = min(merged), max(merged)
    for frame in range(start, end + 1):
        key = (clip, frame)
        current_h = geometry.homographies.get(key, previous_h)
        current_projection = geometry.projections.get(key)
        observations = merged.get(frame, [])
        guide = next((observation for observation in observations if observation.is_guide), None)
        admitted = frame in admission
        if admitted:
            # Inside a qualified window the sequence already chose, over the same measured
            # pool, under the same likelihood. The guide is not consulted on *any* path from
            # here on: not the candidate radius, not the prior, not the forced restart, not
            # the guide-supported reentry search, and not the bootstrap choice below. Applied
            # before the reentry and restart branches so a window cannot be reopened by the
            # very object the first pass reset onto.
            observations = list(admission[frame])
            guide = None
        reentry = None
        if not admitted and (waiting_for_reentry or (modes is None and reacquisition_needs_guide)):
            reentry = _supported_image_reentry(
                frame, merged, config, geometry, clip, fps, require_boundary=waiting_for_reentry
            )
            if reentry is None:
                previous_h = current_h
                continue
            waiting_for_reentry = False
            protect_reentry_until = frame + len(reentry) - 1
            reacquisition_needs_guide = True
            modes = None
            proposals.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "proposal": "qualified_image_reentry",
                    "support_frames": list(range(frame, protect_reentry_until + 1)),
                    "native_samples_invented": False,
                }
            )
        if modes is None:
            if admitted:
                # Bootstrap and post-reset restarts are inside the window too: an admitted
                # frame restarts on its own qualified observation, or stays explicitly
                # missing.  The rejected guide cannot reopen the track here either.
                chosen = observations[0] if observations else None
            else:
                chosen = (
                    reentry[0]
                    if reentry is not None
                    else (
                        guide
                        if has_guide_stream and not (config.crop_first or config.lock_gap_carry)
                        else guide or _choose_restart(observations, config)
                    )
                )
            if chosen is None:
                previous_h = current_h
                continue
            modes = initialise_modes(chosen, config)
            segment_id += 1
            restart_kind = "reentry" if reentry is not None else pending_restart_kind
            pending_restart_kind = "bootstrap"
            if reentry is not None:
                velocity = _reentry_velocity(reentry, frame, geometry, clip)
                for mode in modes.values():
                    mode.mean[2:] = velocity
            innovation_covariance = (
                measurement_covariance(chosen, config) + modes["ballistic"].covariance[:2, :2]
            )
            mahalanobis = 0.0
            regime = "ballistic"
            misses = 0
        else:
            modes = advance_modes(
                modes,
                previous_h,
                current_h,
                geometry.projections.get((clip, frame - 1), current_projection),
                fps,
                config,
            )
            if (
                guide is not None
                and frame > protect_reentry_until
                and not reacquisition_needs_guide
            ):
                observations = [
                    observation
                    for observation in observations
                    if observation.is_guide
                    or math.hypot(observation.x - guide.x, observation.y - guide.y)
                    <= (
                        config.crop_candidate_gate_native
                        if config.crop_first and observation.is_crop
                        else config.guide_candidate_gate_native
                    )
                ]
            ranked = [
                (_candidate_likelihood(modes, observation, config), observation)
                for observation in observations
            ]
            ranked = [item for item in ranked if math.isfinite(item[0][0])]
            guide_was_gated = guide is not None and not any(item[1].is_guide for item in ranked)
            if guide_was_gated and (frame <= protect_reentry_until or reacquisition_needs_guide):
                guide = None
                guide_was_gated = False
            elif guide is not None and not guide_was_gated and frame > protect_reentry_until:
                reacquisition_needs_guide = False
            if (
                config.qualified_image_reentry
                and last_observation is not None
                and (guide_was_gated or not ranked)
            ):
                # Keep native observations in their own camera frame. Exported
                # rows use a legacy coordinate space, and after a gap the last
                # observed frame need not be the previous prediction frame.
                observed_xy, observed_h = last_observation
                previous_xy = _apply_h(camera_warp(observed_h, current_h), observed_xy)
                predicted_xy, _ = _mix_modes(modes)
                if _predicted_image_exit(previous_xy, predicted_xy[:2], config):
                    waiting_for_reentry = True
                    modes = None
                    previous_regime = None
                    pending_restart_kind = "image_exit"
                    proposals.append(
                        {
                            "clip": clip,
                            "frame": frame,
                            "proposal": "missing_after_image_exit",
                            "rejected_guide": guide is not None,
                            "native_samples_invented": False,
                        }
                    )
                    previous_h = current_h
                    continue
            if guide_was_gated:
                # Forced restart: both innovation gates rejected the guide. The filter
                # state is rebuilt on the guide; whether that guide is still the same
                # object is decided by ball_ownership, not asserted here.
                modes = initialise_modes(guide, config)
                segment_id += 1
                restart_kind = "guide_gated"
                chosen = guide
                innovation_covariance = (
                    measurement_covariance(guide, config) + modes["ballistic"].covariance[:2, :2]
                )
                mahalanobis = 0.0
                regime = "ballistic"
                misses = 0
            elif not ranked:
                misses += 1
                if misses > config.maximum_misses:
                    modes = None
                    previous_regime = None
                    pending_restart_kind = "miss_reset"
                previous_h = current_h
                continue
            else:
                selected = max(ranked, key=lambda item: item[0][0])
                if guide is not None:
                    guide_item = next((item for item in ranked if item[1].is_guide), None)
                    if (
                        guide_item is not None
                        and not selected[1].is_guide
                        and selected[0][0] < guide_item[0][0] + config.guide_replacement_margin
                    ):
                        selected = guide_item
                (score, details), chosen = selected
                del score
                component_total = sum(value[3] for value in details.values()) or 1.0
                probabilities = {
                    name: details[name][3] / component_total for name in ("ballistic", "impulse")
                }
                modes = {
                    name: update_mode(mode, chosen, probabilities[name], config)
                    for name, mode in modes.items()
                }
                regime = max(probabilities, key=probabilities.get)
                mahalanobis = details[regime][2]
                innovation_covariance = details[regime][1]
                misses = 0
        if (
            has_guide_stream
            and guide is None
            and not admitted
            and not config.crop_first
            and not config.lock_gap_carry
            and frame > protect_reentry_until
            and not reacquisition_needs_guide
        ):
            # Without crop-first the composed track is defined only where the coarse lock has a
            # row, which is where its missing-estimate frames come from. Crop-first drops that
            # requirement: the IMM gate and the miss counter already bound divergence, so a
            # frame the lock never reached can still be carried by a crop observation.
            previous_h = current_h
            continue
        mixed_mean, mixed_covariance = _mix_modes(modes)
        if previous_regime is not None and regime != previous_regime:
            proposals.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "proposal": "motion_regime_switch",
                    "from_regime": previous_regime,
                    "to_regime": regime,
                    "innovation_mahalanobis": mahalanobis,
                    "detector_support": chosen.detector_count,
                    "sources": list(chosen.sources),
                }
            )
        previous_regime = regime
        source_tokens = [*chosen.sources]
        if chosen.is_crop:
            source_tokens.append("provenance:crop")
        elif any("far_native" in source for source in chosen.sources):
            source_tokens.append("provenance:far_native")
        else:
            source_tokens.append("provenance:coarse")
        source_tokens.extend(f"crop_region:{value}" for value in chosen.crop_provenance)
        confidence = 1.0 / math.sqrt(max(float(np.trace(innovation_covariance)), 1e-12))
        last_observation = (np.asarray([chosen.x, chosen.y]), current_h)
        row = {
            "clip": clip,
            "frame": f"f_{frame:04d}.jpg",
            "x": chosen.x / 2.0,
            "y": chosen.y / 2.0,
            "track_id": 0,
            "score": chosen.score,
            "rank": chosen.rank,
            "sources": "+".join(source_tokens),
            "regime": regime,
            "regime_probability": modes[regime].probability,
            "innovation_mahalanobis": mahalanobis,
            "innovation_cov_xx_native": float(innovation_covariance[0, 0]),
            "innovation_cov_xy_native": float(innovation_covariance[0, 1]),
            "innovation_cov_yy_native": float(innovation_covariance[1, 1]),
            "filter_cov_trace_native": float(np.trace(mixed_covariance[:2, :2])),
            "confidence": confidence,
            "detector_support": chosen.detector_count,
            "homography_source": geometry.source,
        }
        if config.primary_ball_ownership:
            # Private keys; ball_ownership consumes and strips them. Rows stay identical
            # to the unflagged tracker when the option is off.
            row["_segment_id"] = segment_id
            row["_restart_kind"] = restart_kind
        rows.append(row)
        previous_h = current_h
    return rows, proposals


def run_tracker(
    candidate_paths: list[Path],
    source_names: list[str],
    output: Path,
    proposals_output: Path,
    fps: float,
    geometry: Geometry,
    clips: set[str] | None = None,
    config: MotionConfig = MotionConfig(),
    player_boxes: dict | None = None,
) -> dict:
    streams = load_observations(candidate_paths, source_names, clips)
    rows = []
    proposals = []
    ownership_rows = []
    ownership_summaries = []
    pan = None
    if config.primary_ball_ownership:
        from cv.pipeline import ball_ownership

        pan = ball_ownership.pan_model(geometry)
    associate = track_clip
    if config.guide_sequence_association:
        from cv.pipeline import guide_sequence_association

        associate = guide_sequence_association.associate_clip
    for clip, observations in sorted(streams.items()):
        if config.torso_lock_rejection:
            if player_boxes is None:
                raise ValueError("torso lock rejection needs the sided player boxes")
            # A clip with no detected player has no torso to lock onto; its pool is unchanged.
            boxes = player_boxes.get(clip, {})
            from cv.pipeline.torso_lock import without_torso_locks

            observations = without_torso_locks(
                observations,
                boxes,
                min_run=config.torso_lock_frames,
                run_in_frames=config.torso_lock_run_in_frames,
                keep_joined=config.torso_lock_keep_joined,
            )
        clip_rows, clip_proposals = associate(clip, observations, geometry, fps, config)
        if config.primary_ball_ownership:
            annotated, summary = ball_ownership.assign_ownership(
                clip_rows, geometry, clip, fps, config, pan
            )
            ownership_rows.extend(annotated)
            ownership_summaries.append(summary)
            clip_rows = ball_ownership.consumed_rows(annotated)
        rows.extend(clip_rows)
        proposals.extend(clip_proposals)
    write_track_artifact(output, rows, candidate_paths)
    report = {
        "schema": "ball_motion_event_proposals_v1",
        "labels_loaded": False,
        "homography_source": geometry.source,
        "clips": len(streams),
        "emitted_frames": len(rows),
        "proposals": proposals,
    }
    if config.primary_ball_ownership:
        ownership_output = ownership_artifact_path(output)
        write_track_artifact(ownership_output, ownership_rows, candidate_paths)
        report["primary_ball_ownership"] = {
            "artifact": ownership_output.name,
            "consumed_policy": "every chain not proven static; track_id carries the chain id",
            "tracked_rows": len(ownership_rows),
            "consumed_rows": len(rows),
            "withheld_rows": len(ownership_rows) - len(rows),
            "pan_reliable_bound_native": pan.reliable_bound,
            "clips": ownership_summaries,
        }
    proposals_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def ownership_artifact_path(output: Path) -> Path:
    """Side artifact holding every tracked row with its owner and join verdict."""
    name = output.name.replace("arc_augmented_v2", "ownership_v1")
    if name == output.name:
        name = f"{output.stem}_ownership_v1{output.suffix}"
    return output.with_name(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-proposals", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--court-homographies", type=Path)
    parser.add_argument("--camera-projections", type=Path)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument(
        "--qualified-image-reentry",
        action="store_true",
        help="retain image-exit gaps until consecutive native detector candidates support inward reentry",
    )
    parser.add_argument(
        "--lock-is-crop",
        action="store_true",
        help="the coarse_lock stream is itself a native-crop decode (crop-first composition)",
    )
    parser.add_argument(
        "--guide-sequence-association",
        action="store_true",
        help="two-pass association: re-solve short guide-opened detours that the same primary "
        "immediately rejoins, over the full original measured pool",
    )
    parser.add_argument(
        "--primary-ball-ownership",
        action="store_true",
        help=(
            "chain filter segments by a camera-transported position join at every restart, "
            "withhold chains proven static under a reliable camera, carry the chain id in "
            "track_id, and preserve every tracked row in the ownership artifact"
        ),
    )
    parser.add_argument(
        "--crop-first",
        action="store_true",
        help=(
            "prefer native-crop observations over the coarse lock; only correct when the crop "
            "pass runs a detector measurably more accurate than the lock (see crop_first_config)"
        ),
    )
    parser.add_argument(
        "--torso-lock-boxes",
        type=Path,
        help="sided player boxes; with this flag, candidates that persist inside a torso are dropped",
    )
    parser.add_argument(
        "--torso-lock-run-frames",
        action="store_true",
        help="count the torso run in distinct frames, not pooled candidate rows",
    )
    parser.add_argument(
        "--torso-lock-keep-joined",
        action="store_true",
        help="keep a small-torso chain whose ends join outside candidates at ball speed",
    )
    parser.add_argument(
        "--lock-gap-carry",
        action="store_true",
        help="carry the filter across frames the coarse lock never reached (default off)",
    )
    args = parser.parse_args()
    if len(args.candidates) != len(args.source):
        parser.error("--candidates and --source counts must match")
    geometry = load_geometry(args.court_homographies, args.camera_projections)
    config = (
        crop_first_config(lock_is_crop=args.lock_is_crop) if args.crop_first else MotionConfig()
    )
    config = replace(
        config,
        qualified_image_reentry=args.qualified_image_reentry,
        primary_ball_ownership=args.primary_ball_ownership,
        guide_sequence_association=args.guide_sequence_association,
        torso_lock_rejection=args.torso_lock_boxes is not None,
        torso_lock_run_in_frames=args.torso_lock_run_frames,
        torso_lock_keep_joined=args.torso_lock_keep_joined,
        lock_gap_carry=args.lock_gap_carry,
    )
    from cv.pipeline.torso_lock import load_torso_boxes

    boxes = load_torso_boxes(args.torso_lock_boxes) if args.torso_lock_boxes else None
    report = run_tracker(
        args.candidates,
        args.source,
        args.output,
        args.event_proposals,
        args.fps,
        geometry,
        set(args.clip) or None,
        config,
        boxes,
    )
    print(
        f"wrote {report['emitted_frames']} IMM-associated observations across "
        f"{report['clips']} clips -> {args.output}"
    )


if __name__ == "__main__":
    main()
