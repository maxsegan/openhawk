"""Optional launch proposals to an existing first-ground witness.

This is a free-flight initialization chart, not an observed trajectory. It does
not cross supplied net interactions, replace event intervals, or qualify a
whole flight. The caller must retain the existing full-forward support check.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import fast_flight
from cv.pipeline.rich_ball_physics import R_BALL, spin_vector


def free_arc(start, velocity, spin, seconds):
    """Shared drag/Magnus motion without inserting an impact at the target."""
    state = tuple(np.r_[start, velocity, spin_vector(np.r_[start, velocity, spin])])
    count = max(1, int(np.ceil(seconds * 240)))
    dt = seconds / count
    positions = [np.asarray(start, float)]
    for _ in range(count):
        state = fast_flight.rk4_step(state, dt)
        if not np.isfinite(state).all():
            raise ValueError("nonfinite ground-directed propagation")
        positions.append(np.asarray(state[:3]))
    return np.asarray(positions), np.asarray(state[3:6])


def propose(start, spin, start_frame, ground_interval, ground_xy, fps, *, net_intervals=()):
    """Aim one bounded launch at the midpoint of an original ground interval.

    Intermediate root iterations may pass underground in this smooth chart;
    only a converged, descending, above-ground preimpact arc is proposed. The
    supplied target is a seed location, never a new exact ground observation.
    """
    receipt = {
        "method": "existing_ground_witness_drag_shooting",
        "status": "unavailable",
        "original_ground_interval": list(ground_interval),
        "new_exact_event_observation": False,
        "full_forward_support_required": True,
        "uses_withheld_pixels": False,
    }
    start, spin = np.asarray(start, float), np.asarray(spin, float)
    interval, xy = np.asarray(ground_interval, float), np.asarray(ground_xy, float)
    if (
        start.shape != (3,)
        or spin.shape != (3,)
        or interval.shape != (2,)
        or xy.shape != (2,)
        or not np.isfinite(np.r_[start, spin, interval, xy, start_frame, fps]).all()
        or fps <= 0
        or interval[0] > interval[1]
        or start_frame >= interval[0]
        or start[2] <= R_BALL
        or np.any(np.abs(spin) > 6)
    ):
        receipt["reason"] = "unsupported_source_ground_or_launch"
        return None, receipt
    for net in net_intervals:
        bounds = np.asarray(net, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            receipt["reason"] = "unsupported_source_net_interval"
            return None, receipt
        if bounds[1] >= start_frame and bounds[0] <= interval[1]:
            receipt["reason"] = "intervening_or_overlapping_supplied_net"
            return None, receipt
    epoch = float(np.mean(interval))
    seconds = (epoch - start_frame) / fps
    target = np.r_[xy, R_BALL]
    gravity = np.array([0.0, 0.0, -9.81])
    guess = (target - start - 0.5 * gravity * seconds**2) / seconds
    receipt.update(
        target_xyz_m=target.tolist(),
        proposal_epoch_frame=epoch,
        start_xyz_m=start.tolist(),
        gravity_velocity_mps=guess.tolist(),
    )
    if np.any(np.abs(guess) >= 75):
        receipt["reason"] = "gravity_launch_outside_existing_velocity_bounds"
        return None, receipt

    def residual(velocity):
        positions, _ = free_arc(start, velocity, spin, seconds)
        return positions[-1] - target

    try:
        solved = least_squares(
            residual, guess, bounds=(-75.0, 75.0), max_nfev=30, ftol=1e-9, xtol=1e-9, gtol=1e-9
        )
        positions, arrival = free_arc(start, solved.x, spin, seconds)
    except (ValueError, FloatingPointError, OverflowError) as error:
        receipt["reason"] = str(error)
        return None, receipt
    miss = float(np.linalg.norm(positions[-1] - target))
    receipt.update(
        optimizer_success=bool(solved.success),
        objective_evaluations=int(solved.nfev),
        endpoint_error_m=miss,
        arrival_velocity_mps=arrival.tolist(),
        minimum_preimpact_height_m=float(np.min(positions[:-1, 2])),
    )
    if not solved.success or miss > 1e-4:
        receipt["reason"] = "ground_shooting_not_converged"
        return None, receipt
    if arrival[2] >= 0 or np.min(positions[:-1, 2]) < R_BALL:
        receipt["reason"] = "ground_is_not_first_descending_impact"
        return None, receipt
    proposal = np.r_[solved.x, spin]
    receipt.update(status="proposed", velocity_spin=proposal.tolist())
    return proposal, receipt
