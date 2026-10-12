"""Native-pixel drawing geometry for the 3D viewer's ``flight_overlay`` layer.

Shared by the flight-review exporter (``cv.viz.portal_3d_src.export_flight_reviews``) and the
automatic production overlay (``cv.pipeline.portal_flight_overlay``). Every function here only
projects or copies values it is given: a frame without a camera gets no court lines, a missing
keypoint stays missing. Pixels are rounded to 0.1 px.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from cv.viz.point_overlay_video import COURT_LINES_M, COURT_WIDTH, NET_Y

NET_HEIGHT_M = 0.914
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
#: Keypoints below this detector confidence are not drawn.
MIN_KEYPOINT_CONFIDENCE = 0.3

Segment = tuple[tuple[float, float, float], tuple[float, float, float]]


def camera_kind(source: str, supported: bool) -> str:
    """``held`` (not supported this frame), ``intrinsic_fallback`` or ``direct``."""
    if not supported:
        return "held"
    text = (source or "").lower().replace("-", "_")
    if "intrinsic_fallback" in text or "fallback" in text:
        return "intrinsic_fallback"
    return "direct"


def court_segments() -> list[Segment]:
    """Court paint on the ground plus the net-centre line, in court metres."""
    segments = [((x1, y1, 0.0), (x2, y2, 0.0)) for (x1, y1), (x2, y2) in COURT_LINES_M]
    segments.append(((0.0, NET_Y, NET_HEIGHT_M), (COURT_WIDTH, NET_Y, NET_HEIGHT_M)))
    return segments


def project(matrix: np.ndarray | None, xyz: Iterable[float]) -> tuple[float, float] | None:
    """Native pixel of a court point, or None without a camera or in front of the lens."""
    if matrix is None:
        return None
    image = np.asarray(matrix, dtype=float) @ np.asarray([*xyz, 1.0], dtype=float)
    if not np.isfinite(image).all() or abs(float(image[2])) < 1e-12:
        return None
    u, v = float(image[0] / image[2]), float(image[1] / image[2])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v


def xy(value: tuple[float, float] | list[float] | None) -> list[float] | None:
    if value is None:
        return None
    try:
        u, v = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return [round(u, 1), round(v, 1)]


def project_segment(matrix: np.ndarray | None, segment: Segment) -> list[float] | None:
    start = project(matrix, segment[0])
    end = project(matrix, segment[1])
    if start is None or end is None:
        return None
    return [*xy(start), *xy(end)]


def court_lines(matrix: np.ndarray | None, segments: list[Segment]) -> list[list[float]]:
    lines = []
    for segment in segments:
        projected = project_segment(matrix, segment)
        if projected is not None:
            lines.append(projected)
    return lines


def box(row: dict[str, Any]) -> list[float] | None:
    """Native box ``[x0, y0, x1, y1]`` from a player-box row."""
    for keys in (
        ("x0_native", "y0_native", "x1_native", "y1_native"),
        ("x0", "y0", "x1", "y1"),
    ):
        if all(row.get(key) not in (None, "") for key in keys):
            try:
                values = [float(row[key]) for key in keys]
            except (TypeError, ValueError):
                return None
            return (
                [round(value, 1) for value in values] if all(map(math.isfinite, values)) else None
            )
    return None


def joints(row: dict[str, Any]) -> dict[str, list[float]]:
    """Confident COCO keypoints from ``<name>_x/_y/_confidence`` columns."""
    found: dict[str, list[float]] = {}
    for name in KEYPOINT_NAMES:
        x_key, y_key = f"{name}_x", f"{name}_y"
        if row.get(x_key) in (None, "") or row.get(y_key) in (None, ""):
            continue
        confidence = (
            row.get(f"{name}_confidence") or row.get(f"{name}_c") or row.get(f"{name}_conf") or "1"
        )
        try:
            if float(confidence) < MIN_KEYPOINT_CONFIDENCE:
                continue
            point = xy((row[x_key], row[y_key]))
        except (TypeError, ValueError):
            continue
        if point is not None:
            found[name] = point
    return found
