"""Which player box served: the ball at serve contact sits in the server's reach.

``serve_side_association="serve_reach"``. The existing association takes the box
whose centre is nearest the contact pixel. A near server's ball is tossed well
above her head, so on a steep camera it can sit nearer the far player's small box
than her own large one, and the whole point is then fitted from the wrong end
(fresh ``source09_case001_short_a2``). Here each box is widened by the reach a
serving body has in its own picture scale (up by ``REACH_UP_HEIGHTS`` of its
height, sideways by ``REACH_SIDE_HEIGHTS``) and the ball's distance outside that
region is divided by the box height. The smallest wins; a tie (both reach the
ball, or neither by the same margin) keeps the nearest centre. Boxes only; no
pose keypoints are required.
"""

from __future__ import annotations

import numpy as np

ASSOCIATIONS = ("centre", "serve_reach")
REACH_UP_HEIGHTS = 0.9
REACH_SIDE_HEIGHTS = 0.35


def reach_distance(row: dict, pixel: np.ndarray, scale: float = 1.0) -> float:
    """Ball distance outside the box's serving reach, in box heights."""

    x0, y0, x1, y1 = (scale * float(row[key]) for key in ("x0", "y0", "x1", "y1"))
    height = y1 - y0
    if not np.isfinite([x0, y0, x1, y1]).all() or height <= 0:
        return float("inf")
    left = x0 - REACH_SIDE_HEIGHTS * height
    right = x1 + REACH_SIDE_HEIGHTS * height
    top = y0 - REACH_UP_HEIGHTS * height
    dx = max(left - pixel[0], 0.0, pixel[0] - right)
    dy = max(top - pixel[1], 0.0, pixel[1] - y1)
    return float(np.hypot(dx, dy) / height)


def choose(rows: list[dict], pixel: np.ndarray, scale: float, association: str, centre_key):
    """The server row under ``association``; ``centre_key`` orders the legacy choice."""

    if association not in ASSOCIATIONS:
        raise ValueError("supported serve side association required")
    if association == "centre":
        return min(rows, key=centre_key)
    pixel = np.asarray(pixel, float)
    return min(rows, key=lambda row: (round(reach_distance(row, pixel, scale), 6), centre_key(row)))
