"""Locate a descending ground crossing on the integrator's fractional RK4 path.

This removes secant interpolation error, not uncertainty in observed contact
timing or the physical ball/court geometry. Callers retain event ownership,
rebound and dwell semantics.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from cv.experiments.connected_shooting import fast_flight
from cv.pipeline.rich_ball_physics import R_BALL


def descending_ground_root(
    state: tuple[float, ...], nxt: tuple[float, ...], dt: float
) -> tuple[float, tuple[float, ...]]:
    """Return seconds and state of the descending crossing in this RK4 step.

    ``nxt`` must be the full RK4 step from ``state`` over ``dt``. For a
    floor-level upward launch that returns within one step, start the root
    bracket at the apex: the root at zero is the launch, not a ground impact.
    """
    if not math.isfinite(dt) or dt <= 0 or not all(map(math.isfinite, (*state, *nxt))):
        raise ValueError("finite positive RK4 ground-crossing step required")
    if nxt[2] > R_BALL or nxt[5] >= 0:
        raise ValueError("descending ground crossing must be bracketed")
    if state[2] <= R_BALL and state[5] <= 0:
        raise ValueError("descending initial ground state has no supported flight")

    def advance(seconds: float) -> tuple[float, ...]:
        # Retain the exact supplied endpoint, including a root at the step end.
        return nxt if seconds == dt else fast_flight.rk4_step(state, seconds)

    low = 0.0
    if state[2] <= R_BALL:
        low = brentq(lambda t: advance(t)[5], 0.0, dt, xtol=5e-15, rtol=1e-14)
        if advance(low)[2] <= R_BALL:
            raise ValueError("upward initial ground state has no above-ground flight")
    if nxt[2] == R_BALL:
        return dt, nxt
    seconds = brentq(lambda t: advance(t)[2] - R_BALL, low, dt, xtol=5e-15, rtol=1e-14, maxiter=64)
    impact = advance(seconds)
    if impact[5] >= 0:
        raise ValueError("ground root is not descending")
    return seconds, impact


def step_segment(
    state: tuple[float, ...], nxt: tuple[float, ...], dt: float
) -> tuple[tuple[float, ...], float, bool]:
    """Return the sampling segment the measured integrator keeps for this RK4 step.

    ``(segment_end, seconds, grounded)``. Without a descending ground crossing the
    segment is the whole step. With one, it ends at the refined root knot with its
    height set to ``R_BALL`` exactly, as ``measured_dynamics.integrate`` records it
    for both an in-horizon bounce and a beyond-horizon future knot. Callers that
    locate other plane crossings by chord interpolation must use this segment so
    their crossing agrees with linear sampling of the measured trace and never
    precedes a ground impact that physically comes first. No other future knot
    shortens the segment.
    """
    if not (nxt[2] <= R_BALL and nxt[5] < 0):
        return nxt, dt, False
    seconds, impact = descending_ground_root(state, nxt, dt)
    return (impact[0], impact[1], R_BALL) + tuple(impact[3:]), seconds, True
