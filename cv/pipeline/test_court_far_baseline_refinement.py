from __future__ import annotations

import cv2
import numpy as np

from cv.pipeline import court
from cv.pipeline.court_far_baseline_refinement import FarBaselineConfig, refine_far_baseline


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    image_points = np.asarray(
        [(180, 1000), (1740, 1000), (710, 220), (1210, 220)], dtype=np.float32
    )
    world = np.asarray(
        [(0.0, 0.0), (court.COURT_W, 0.0), (0.0, court.COURT_L), (court.COURT_W, court.COURT_L)],
        dtype=np.float32,
    )
    seed = np.linalg.inv(cv2.getPerspectiveTransform(world, image_points))
    cv2.line(image, (710, 221), (1210, 221), (245, 245, 245), 1)
    return image, seed


def test_refinement_is_label_free_bounded_and_improves_line_residual() -> None:
    image, seed = _synthetic()
    result = refine_far_baseline(
        image,
        seed,
        FarBaselineConfig(
            "test",
            minimum_support_fraction=0.25,
            minimum_residual_improvement_px=0.1,
        ),
    )
    assert result.evidence["accepted"] is True
    assert result.evidence["residual_improvement_px"] > 0
    assert result.evidence["maximum_landmark_move_px540"] <= 2.0


def test_refinement_falls_back_when_paint_is_absent() -> None:
    image, seed = _synthetic()
    image[:] = 35
    result = refine_far_baseline(image, seed)
    assert result.evidence["accepted"] is False
    np.testing.assert_allclose(result.homography, seed)


def test_large_move_requires_stronger_image_residual_witness() -> None:
    image, seed = _synthetic()
    result = refine_far_baseline(
        image,
        seed,
        FarBaselineConfig(
            "test",
            maximum_unguarded_move_px540=0.01,
            minimum_large_move_residual_improvement_px=100.0,
            minimum_support_fraction=0.25,
            minimum_residual_improvement_px=0.1,
        ),
    )
    assert result.evidence["accepted"] is False
    assert result.evidence["movement_guard_passed"] is False
    np.testing.assert_allclose(result.homography, seed)


def test_far_baseline_moves_to_opposite_edges_under_the_two_edge_conventions() -> None:
    image, seed = _synthetic()
    # A far baseline six pixels thick, so its two visible edges are distinguishable.
    cv2.line(image, (710, 221), (1210, 221), (245, 245, 245), 6)
    config = FarBaselineConfig(
        "test", minimum_support_fraction=0.25, minimum_residual_improvement_px=0.1
    )

    outside = refine_far_baseline(image, seed, config, edge_convention="itf_outside_edge")
    lower = refine_far_baseline(image, seed, config, edge_convention="image_lower_edge")
    centre = refine_far_baseline(image, seed, config, edge_convention="paint_centre")

    corners = np.asarray([(0.0, court.COURT_L), (court.COURT_W, court.COURT_L)], np.float32)

    def far_baseline_y(homography: np.ndarray) -> float:
        pixels = cv2.perspectiveTransform(
            corners.reshape(1, -1, 2), np.linalg.inv(np.asarray(homography, float))
        )[0]
        return float(np.mean(pixels[:, 1]))

    assert far_baseline_y(outside.homography) < far_baseline_y(centre.homography)
    assert far_baseline_y(centre.homography) < far_baseline_y(lower.homography)
    assert outside.evidence["edge_convention"] == "itf_outside_edge"
