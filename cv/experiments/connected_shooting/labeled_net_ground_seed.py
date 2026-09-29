"""Ballistic net-to-first-ground outgoing velocity initializer.

A default passive outgoing seed can hit the bounce cap before optimization on a
multiple-ground terminal net case. This returns a gravity-only velocity from an
estimated net point/time to an explicitly supplied first ground point/time.

It is only an initializer: not a net material law, fitted path, acceptance gate,
or truth input. Both coordinates are input estimates. There is no hard direction
or passivity requirement. Original timestamps are unchanged. Drag/Magnus fitting
follows elsewhere.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_ASSUMPTIONS = (
    "ballistic gravity-only outgoing seed; drag and Magnus fitting follow elsewhere",
    "net and first-ground coordinates are explicit input estimates, not ground truth",
    "initializer only: not a net material law, fitted path, acceptance gate, or truth input",
    "no hard direction or passivity requirement",
    "original timestamps unchanged",
)


def _finite_xyz(value, label):
    try:
        xyz = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"finite {label} coordinates required") from exc
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"finite {label} coordinates required")
    return xyz


def _finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"finite {label} required") from exc
    if not math.isfinite(number):
        raise ValueError(f"finite {label} required")
    return number


def _finite_positive(value, label):
    number = _finite_number(value, label)
    if number <= 0.0:
        raise ValueError(f"finite positive {label} required")
    return number


def _receipt(*, status, elapsed_seconds, outgoing_velocity_mps=None, unsupported_reason=None):
    payload = {
        "status": status,
        "elapsed_seconds": float(elapsed_seconds),
        "assumptions": list(_ASSUMPTIONS),
        "native_timestamps_changed": False,
    }
    if outgoing_velocity_mps is not None:
        payload["outgoing_velocity_mps"] = [float(component) for component in outgoing_velocity_mps]
    if unsupported_reason is not None:
        payload["unsupported_reason"] = unsupported_reason
    return payload


def ballistic_ground_seed(
    net_xyz_m: ArrayLike,
    net_frame: float,
    ground_xyz_m: ArrayLike,
    ground_frame: float,
    fps: float,
    *,
    gravity_mps2: float = 9.81,
    maximum_component_mps: float = 75.0,
) -> dict[str, Any]:
    """Return a ballistic outgoing seed from estimated net to first ground.

    Vxy = (groundxy - netxy) / delta and
    Vz = (groundZ - netZ + 0.5 * g * delta^2) / delta, with
    delta = (ground_frame - net_frame) / fps. Components beyond the numerical
    bound are unsupported rather than clamped.
    """
    net = _finite_xyz(net_xyz_m, "net")
    ground = _finite_xyz(ground_xyz_m, "ground")
    net_frame = _finite_number(net_frame, "net frame")
    ground_frame = _finite_number(ground_frame, "ground frame")
    fps = _finite_positive(fps, "fps")
    gravity = _finite_positive(gravity_mps2, "gravity")
    bound = _finite_positive(maximum_component_mps, "velocity bound")
    elapsed = (ground_frame - net_frame) / fps
    if elapsed <= 0.0:
        raise ValueError("positive net-to-ground duration required")
    if not net[2] > ground[2]:
        return _receipt(
            status="unsupported",
            elapsed_seconds=elapsed,
            unsupported_reason="source_net_not_above_ground",
        )
    velocity = [
        (ground[0] - net[0]) / elapsed,
        (ground[1] - net[1]) / elapsed,
        (ground[2] - net[2] + 0.5 * gravity * elapsed**2) / elapsed,
    ]
    if any(abs(component) > bound for component in velocity):
        return _receipt(
            status="unsupported",
            elapsed_seconds=elapsed,
            unsupported_reason="velocity_component_exceeds_bound",
        )
    return _receipt(status="supported", elapsed_seconds=elapsed, outgoing_velocity_mps=velocity)
