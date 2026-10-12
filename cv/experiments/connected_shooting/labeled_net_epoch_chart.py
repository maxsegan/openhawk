"""Input-event chart for a first net contact, without a flat missing-hit branch.

Experimental labeled lifting helper. Solve the incoming court-length velocity
so the original continuous measured flight reaches the net plane at a supplied
epoch inside its annotation interval. This supplies an optimization coordinate;
the original net simulator must still replay the final path and impact law.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from cv.experiments.connected_shooting import measured_dynamics


def project_net_velocity(
    theta,
    start_frame,
    net_frame,
    fps,
    surface,
    *,
    bounce_profile="nominal",
    rebound_scales=(1.0, 1.0),
    velocity_bounds=(-75.0, 75.0),
):
    """Return theta9 with only Vy changed and an explicit physical-support receipt."""
    theta = np.asarray(theta, float)
    low, high = map(float, velocity_bounds)
    if (
        theta.shape != (9,)
        or not np.isfinite(theta).all()
        or not start_frame < net_frame
        or not fps > 0
        or not low < high
        or abs(theta[1] - 11.885) < 1e-6
    ):
        raise ValueError("finite incoming state, positive duration and velocity bounds required")
    queries = np.array([start_frame, net_frame], float)
    calls = 0
    last = None

    def evaluate(vy):
        nonlocal calls, last
        candidate = theta.copy()
        candidate[4] = vy
        positions, velocities, _, impacts = measured_dynamics.simulate(
            candidate,
            start_frame,
            queries,
            fps,
            surface,
            bounce_profile=bounce_profile,
            rebound_scales=rebound_scales,
        )
        calls += 1
        last = (candidate, positions[-1], velocities[-1], impacts)
        return float(positions[-1, 1] - 11.885)

    # The original Vy and time-of-flight correction usually bracket locally.
    # A bounded bracket fallback remains explicit for a distant original path.
    initial = float(np.clip(theta[4], low, high))
    error = evaluate(initial)
    root = initial
    for _ in range(8):
        if abs(error) < 1e-8:
            break
        step = 1e-3
        probe = min(root + step, high) if root < high else max(root - step, low)
        derivative = (evaluate(probe) - error) / (probe - root)
        if abs(derivative) < 1e-8:
            break
        proposed = float(np.clip(root - error / derivative, low, high))
        if abs(proposed - root) < 1e-10:
            break
        root = proposed
        error = evaluate(root)
    if abs(error) >= 1e-8:
        root = brentq(evaluate, low, high, xtol=1e-10, maxiter=40)
    error = evaluate(root)
    candidate, xyz, velocity, impacts = last
    before = [b for b in impacts if float(b["frame"]) < net_frame - 1e-6]
    if before:
        raise ValueError("first-net chart has an earlier ground impact")
    if abs(error) > 1e-6 or (theta[1] - 11.885) * velocity[1] >= 0:
        raise ValueError("projected state does not approach the physical net plane")
    return candidate, {
        "net_frame": float(net_frame),
        "net_xyz_m": xyz.tolist(),
        "incoming_velocity_mps": velocity.tolist(),
        "plane_error_m": abs(error),
        "initial_vy_mps": float(theta[4]),
        "projected_vy_mps": float(root),
        "simulation_calls": calls,
        "prior_ground_impacts": len(before),
        "velocity_bounds_mps": [low, high],
    }
