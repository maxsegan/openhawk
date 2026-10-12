"""Label-blind sanity checks for per-point court homographies."""

from __future__ import annotations

import math

import cv2
import numpy as np

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0


def project_court_line(
    homography: np.ndarray,
    court_y: float,
) -> np.ndarray:
    inverse = np.linalg.inv(homography)
    points = np.float32(
        [[[0.0, court_y], [COURT_WIDTH_M, court_y]]]
    )
    return cv2.perspectiveTransform(points, inverse)[0]


def assess_court_homography(
    homography: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    frame_margin: float = 0.05,
) -> dict:
    reasons = []
    lines = {}
    try:
        for name, court_y in (
            ("near_baseline", 0.0),
            ("net", NET_Y_M),
            ("far_baseline", COURT_LENGTH_M),
        ):
            lines[name] = project_court_line(homography, court_y)
    except (np.linalg.LinAlgError, cv2.error):
        return {
            "valid": False,
            "reasons": ["singular_homography"],
            "line_centers_y": {},
        }
    if not all(np.isfinite(line).all() for line in lines.values()):
        return {
            "valid": False,
            "reasons": ["nonfinite_projection"],
            "line_centers_y": {},
        }

    centers = {
        name: float(np.mean(line[:, 1]))
        for name, line in lines.items()
    }
    minimum_y = -frame_margin * image_height
    maximum_y = (1.0 + frame_margin) * image_height
    if any(not minimum_y <= value <= maximum_y for value in centers.values()):
        reasons.append("court_reference_line_offscreen")
    if not (
        centers["near_baseline"]
        > centers["net"]
        > centers["far_baseline"]
    ):
        reasons.append("court_line_order_invalid")
    if (
        centers["near_baseline"] - centers["net"] < 0.05 * image_height
        or centers["net"] - centers["far_baseline"] < 0.05 * image_height
    ):
        reasons.append("court_depth_collapsed")
    if (
        centers["near_baseline"] - centers["far_baseline"]
        < 0.25 * image_height
    ):
        reasons.append("court_depth_span_too_small")
    spans = {
        name: float(np.linalg.norm(line[1] - line[0]))
        for name, line in lines.items()
    }
    if any(
        not math.isfinite(span) or span < 0.15 * image_width
        for span in spans.values()
    ):
        reasons.append("court_width_collapsed")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "line_centers_y": centers,
        "line_spans_px": spans,
    }
