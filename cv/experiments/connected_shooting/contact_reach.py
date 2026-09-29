"""Continuous root-ray reach diagnostics, not calibrated anatomical limits.

A native box-bottom pixel defines a camera ray, not a grounded foot. Between
two assumed heights its world positions form a line segment. Distance to that
segment is a conditional lower bound on root-to-contact reach, not 3D accuracy.
"""

from __future__ import annotations

import numpy as np


def minimum_reach(contact_xyz, lower_root_xyz, upper_root_xyz) -> dict:
    """Analytic full-3D distance; no grid choice, exposure or pose interpolation."""
    contact, lower, upper = (
        np.asarray(v, float) for v in (contact_xyz, lower_root_xyz, upper_root_xyz)
    )
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (contact, lower, upper)):
        raise ValueError("finite contact and root segment endpoints required")
    if lower[2] < 0 or upper[2] < lower[2]:
        raise ValueError("ordered nonnegative assumed root heights required")
    delta = upper - lower
    length_sq = float(delta @ delta)
    if upper[2] == lower[2] and length_sq != 0:
        raise ValueError("one root ray cannot have different roots at the same height")
    fraction = (
        0.0 if length_sq == 0 else float(np.clip((contact - lower) @ delta / length_sq, 0, 1))
    )
    closest = lower + fraction * delta
    return {
        "minimum_root_to_contact_distance_m": float(np.linalg.norm(contact - closest)),
        "minimizing_assumed_root_xyz_m": closest.tolist(),
        "assumed_root_height_interval_m": [float(lower[2]), float(upper[2])],
        "minimizing_height_is_measurement": False,
        "root_segment_endpoints_m": [lower.tolist(), upper.tolist()],
    }
