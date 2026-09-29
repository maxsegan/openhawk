import cv2
import numpy as np

from cv.pipeline.court_geometry_gate import assess_court_homography


def image_to_court_homography() -> np.ndarray:
    image = np.float32(
        [[100, 900], [1820, 900], [500, 250], [1420, 250]]
    )
    court = np.float32(
        [[0, 0], [10.97, 0], [0, 23.77], [10.97, 23.77]]
    )
    return cv2.getPerspectiveTransform(image, court)


def test_accepts_full_visible_standard_court():
    result = assess_court_homography(
        image_to_court_homography(),
        1920,
        1080,
    )

    assert result["valid"]
    assert result["reasons"] == []


def test_rejects_self_similar_half_court_fit():
    image = np.float32(
        [[-800, 3000], [2700, 3000], [430, 690], [1490, 690]]
    )
    court = np.float32(
        [[0, 0], [10.97, 0], [0, 23.77], [10.97, 23.77]]
    )
    result = assess_court_homography(
        cv2.getPerspectiveTransform(image, court),
        1920,
        1080,
    )

    assert not result["valid"]
    assert "court_reference_line_offscreen" in result["reasons"]
