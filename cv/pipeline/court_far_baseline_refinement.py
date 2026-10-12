"""Label-free native refinement of the far painted court baseline.

This is the production form of the S2-12 ``far_k21_r10`` arm. It starts from
an accepted topology homography, refits only the far baseline from native image
evidence, and retains the seed whenever image support or the bounded-motion
guard is insufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from cv.pipeline import court
from cv.pipeline import court_topology as topology
from cv.pipeline.court_geometry_gate import assess_court_homography


@dataclass(frozen=True)
class FarBaselineConfig:
    """Spatial constants are native 1920x1080 pixels.

    The far baseline is refined to its ITF outside edge, the one facing away from the net.
    """

    name: str = "far_k21_r10"
    kernel_px: int = 21
    search_radius_px: float = 10.0
    offset_step_px: float = 0.5
    samples: int = 96
    endpoint_trim_fraction: float = 0.05
    local_window_samples: int = 9
    minimum_prominence: float = 8.0
    maximum_band_width_px: float = 10.0
    maximum_unguarded_move_px540: float = 2.0
    minimum_support_fraction: float = 0.45
    minimum_residual_improvement_px: float = 0.35
    minimum_large_move_residual_improvement_px: float = 1.0
    refine_far_sidelines: bool = False
    far_sideline_start_y_m: float = 18.285

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FarBaselineResult:
    homography: np.ndarray
    evidence: dict[str, object]


DEFAULT_CONFIG = FarBaselineConfig()


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return topology._project_court_points(homography, np.asarray(points, dtype=float))


def _line_distance(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    normal = np.asarray(line[:2], dtype=float)
    denominator = float(np.linalg.norm(normal))
    return np.abs(np.asarray(points, dtype=float) @ normal + float(line[2])) / denominator


def _local_edge_offsets(
    response: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    config: FarBaselineConfig,
    edge_convention: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    offsets = np.arange(
        -config.search_radius_px,
        config.search_radius_px + 0.5 * config.offset_step_px,
        config.offset_step_px,
    )
    sample_x = points[:, 0][None, :] + offsets[:, None] * normal[0]
    sample_y = points[:, 1][None, :] + offsets[:, None] * normal[1]
    sampled = cv2.remap(
        response.astype(np.float32),
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    smoothing = _odd(config.local_window_samples)
    smoothed = cv2.GaussianBlur(sampled, (1, smoothing), 0)
    selected_points: list[np.ndarray] = []
    selected_offsets: list[float] = []
    evidence: list[dict[str, float]] = []
    for index in range(smoothed.shape[1]):
        profile = smoothed[:, index]
        baseline = float(np.quantile(profile, 0.20))
        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])
        prominence = peak - baseline
        if prominence < config.minimum_prominence:
            continue
        threshold = baseline + max(3.0, 0.24 * prominence)
        left = peak_index
        right = peak_index
        while left > 0 and profile[left - 1] >= threshold:
            left -= 1
        while right + 1 < len(profile) and profile[right + 1] >= threshold:
            right += 1
        width = float(offsets[right] - offsets[left])
        if width > config.maximum_band_width_px or right == len(profile) - 1:
            continue
        if edge_convention == "paint_centre":
            if left == 0:
                continue
            weights = profile[left : right + 1] - threshold
            edge_offset = float(np.sum(weights * offsets[left : right + 1]) / np.sum(weights))
        else:
            edge_offset = float(offsets[right])
        selected_points.append(points[index] + edge_offset * normal)
        selected_offsets.append(edge_offset)
        evidence.append(
            {
                "sample_index": float(index),
                "offset_px": edge_offset,
                "prominence": prominence,
                "band_width_px": width,
            }
        )
    return (
        np.asarray(selected_points, dtype=float),
        np.asarray(selected_offsets, dtype=float),
        evidence,
    )


def _fit_robust_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("insufficient far-baseline edge points")
    fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction = fit[:2]
    origin = fit[2:]
    line = np.asarray([-direction[1], direction[0], 0.0], dtype=float)
    line[2] = -float(np.dot(line[:2], origin))
    distances = _line_distance(line, points)
    cutoff = max(1.25, float(np.quantile(distances, 0.80)))
    inliers = distances <= cutoff
    if int(np.count_nonzero(inliers)) >= 2 and not bool(np.all(inliers)):
        fit = cv2.fitLine(points[inliers].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        direction = fit[:2]
        origin = fit[2:]
        line = np.asarray([-direction[1], direction[0], 0.0], dtype=float)
        line[2] = -float(np.dot(line[:2], origin))
    return line, inliers


def _refine_segment_line(
    response: np.ndarray,
    endpoints: np.ndarray,
    orientation: str,
    config: FarBaselineConfig,
    edge_convention: str,
) -> tuple[np.ndarray, dict[str, object]]:
    original = topology._segment_line(endpoints.ravel())
    normal = topology._oriented_line_normal(original, orientation, edge_convention)
    fractions = np.linspace(
        config.endpoint_trim_fraction,
        1.0 - config.endpoint_trim_fraction,
        config.samples,
    )
    sample_points = endpoints[0] + fractions[:, None] * (endpoints[1] - endpoints[0])
    edge_points, offsets, samples = _local_edge_offsets(
        response, sample_points, normal, config, edge_convention
    )
    support_fraction = float(len(edge_points) / len(sample_points))
    evidence: dict[str, object] = {
        "support_count": int(len(edge_points)),
        "sample_count": int(len(sample_points)),
        "support_fraction": support_fraction,
        "samples": samples,
    }
    if support_fraction < config.minimum_support_fraction:
        return original, {**evidence, "refined": False, "reason": "insufficient_edge_support"}
    fitted, inliers = _fit_robust_line(edge_points)
    original_residual = float(np.median(_line_distance(original, edge_points)))
    fitted_residual = float(np.median(_line_distance(fitted, edge_points)))
    improvement = original_residual - fitted_residual
    refined = improvement >= config.minimum_residual_improvement_px
    return (fitted if refined else original), {
        **evidence,
        "refined": refined,
        "reason": "refined" if refined else "insufficient_residual_improvement",
        "inlier_count": int(np.count_nonzero(inliers)),
        "median_edge_offset_px": float(np.median(offsets)),
        "original_line_residual_px": original_residual,
        "refined_line_residual_px": fitted_residual,
        "residual_improvement_px": improvement,
    }


def refine_far_baseline(
    image: np.ndarray,
    native_homography: np.ndarray,
    config: FarBaselineConfig = DEFAULT_CONFIG,
    edge_convention: str | None = None,
) -> FarBaselineResult:
    """Refine the far baseline while preserving a label-free guarded fallback."""
    edge_convention = topology.resolve_edge_convention(edge_convention)
    height, width = image.shape[:2]
    homography = np.asarray(native_homography, dtype=float)
    endpoints = _project(
        homography,
        np.asarray([(0.0, court.COURT_L), (court.COURT_W, court.COURT_L)]),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(config.kernel_px), _odd(config.kernel_px))),
    )
    original_line = topology._segment_line(endpoints.ravel())
    fitted_line, baseline_evidence = _refine_segment_line(
        response, endpoints, "far_horizontal", config, edge_convention
    )
    support_fraction = float(baseline_evidence["support_fraction"])
    base_evidence: dict[str, object] = {
        "config": config.manifest(),
        "edge_convention": edge_convention,
        "support_count": baseline_evidence["support_count"],
        "sample_count": baseline_evidence["sample_count"],
        "support_fraction": support_fraction,
        "samples": baseline_evidence["samples"],
        "far_baseline": baseline_evidence,
    }
    if not baseline_evidence["refined"]:
        return FarBaselineResult(
            homography,
            {
                **base_evidence,
                "accepted": False,
                "reason": str(baseline_evidence["reason"]),
            },
        )
    original_residual = float(baseline_evidence["original_line_residual_px"])
    fitted_residual = float(baseline_evidence["refined_line_residual_px"])
    improvement = float(baseline_evidence["residual_improvement_px"])
    offsets = np.asarray(
        [sample["offset_px"] for sample in baseline_evidence["samples"]], dtype=float
    )

    near_endpoints = _project(homography, np.asarray([(0.0, 0.0), (court.COURT_W, 0.0)]))
    left_line = topology._segment_line(
        _project(homography, np.asarray([(0.0, 0.0), (0.0, court.COURT_L)])).ravel()
    )
    right_line = topology._segment_line(
        _project(
            homography,
            np.asarray([(court.COURT_W, 0.0), (court.COURT_W, court.COURT_L)]),
        ).ravel()
    )
    side_evidence: dict[str, object] = {}
    if config.refine_far_sidelines:
        left_endpoints = _project(
            homography,
            np.asarray([(0.0, config.far_sideline_start_y_m), (0.0, court.COURT_L)]),
        )
        right_endpoints = _project(
            homography,
            np.asarray(
                [
                    (court.COURT_W, config.far_sideline_start_y_m),
                    (court.COURT_W, court.COURT_L),
                ]
            ),
        )
        left_candidate, left_evidence = _refine_segment_line(
            response, left_endpoints, "left_vertical", config, edge_convention
        )
        right_candidate, right_evidence = _refine_segment_line(
            response, right_endpoints, "right_vertical", config, edge_convention
        )
        if left_evidence["refined"]:
            left_line = left_candidate
        if right_evidence["refined"]:
            right_line = right_candidate
        side_evidence = {"left_doubles": left_evidence, "right_doubles": right_evidence}
    far_left = topology._intersection(fitted_line, left_line)
    far_right = topology._intersection(fitted_line, right_line)
    if far_left is None or far_right is None:
        return FarBaselineResult(
            homography,
            {**base_evidence, "accepted": False, "reason": "parallel_line_intersection"},
        )
    world = np.asarray(
        [
            (0.0, 0.0),
            (court.COURT_W, 0.0),
            (0.0, court.COURT_L),
            (court.COURT_W, court.COURT_L),
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(
        [near_endpoints[0], near_endpoints[1], far_left, far_right], dtype=np.float32
    )
    world_to_image = cv2.getPerspectiveTransform(world, image_points)
    candidate = np.linalg.inv(world_to_image)
    standard_world = np.asarray(topology.LANDMARKS, dtype=float)
    original_projection = _project(homography, standard_world)
    candidate_projection = _project(candidate, standard_world)
    movement_px540 = np.linalg.norm(candidate_projection - original_projection, axis=1) / 2.0
    maximum_move = float(np.max(movement_px540))

    active_improvements = [improvement]
    active_improvements.extend(
        float(row["residual_improvement_px"])
        for row in side_evidence.values()
        if row.get("refined")
    )
    large_move_witness = min(active_improvements)
    movement_guard_by_landmark = [
        bool(
            move <= config.maximum_unguarded_move_px540
            or large_move_witness >= config.minimum_large_move_residual_improvement_px
        )
        for move in movement_px540
    ]
    movement_guard_passed = all(movement_guard_by_landmark)
    geometry_valid = bool(assess_court_homography(candidate, width, height)["valid"])
    accepted = bool(
        geometry_valid
        and improvement >= config.minimum_residual_improvement_px
        and movement_guard_passed
    )
    return FarBaselineResult(
        candidate if accepted else homography,
        {
            **base_evidence,
            "accepted": accepted,
            "reason": "accepted" if accepted else "image_residual_or_motion_guard",
            "inlier_count": baseline_evidence["inlier_count"],
            "median_edge_offset_px": float(np.median(offsets)),
            "original_line_residual_px": original_residual,
            "refined_line_residual_px": fitted_residual,
            "residual_improvement_px": improvement,
            "maximum_landmark_move_px540": maximum_move,
            "movement_guard_passed": movement_guard_passed,
            "movement_guard_by_landmark": movement_guard_by_landmark,
            "large_move_residual_witness_px": large_move_witness,
            "geometry_sanity_passed": geometry_valid,
            "far_sidelines": side_evidence,
            "per_landmark_move_px540": [float(value) for value in movement_px540],
            "original_far_line": original_line.tolist(),
            "fitted_far_line": fitted_line.tolist(),
        },
    )
