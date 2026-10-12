"""Optional unresolved normal-contact continuation after tiny physical rebounds.

A 100 micrometre maximum ballistic apex bounds the omitted normal excursion.
Tangential sliding uses the existing ball inertia and surface Coulomb friction;
rolling uses an explicit small resistance, not repeated bounce impulses. This
is an approximate continuation model, not a measured low-speed calibration.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from physics import impact, surface_model

RADIUS = impact.R_BALL
GRAVITY = 9.81
MAX_APEX_M = 0.0001
ROLLING_RESISTANCE = 0.02
MODE = "unresolved_normal_contact_v1"
# Resolved impacts advance by positive dwell; the finite query horizon bounds
# work. A fixed count must not invalidate otherwise physical small rebounds.
CACHE_TAG = "normal_contact_horizon_v2"
_ACTIVE = ContextVar("ground_contact_active", default=False)


def active():
    return _ACTIVE.get()


@contextmanager
def using(enabled):
    if type(enabled) is not bool:
        raise ValueError("explicit boolean ground-contact policy required")
    token = _ACTIVE.set(enabled)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def qualifies(position, velocity):
    """Only finite floor-adjacent, nonnegative sub-resolution launch velocity."""
    p, v = np.asarray(position, float), np.asarray(velocity, float)
    return bool(
        p.shape == v.shape == (3,)
        and np.isfinite(p).all()
        and np.isfinite(v).all()
        and abs(float(p[2]) - RADIUS) <= 1e-9
        and 0 <= v[2]
        and v[2] ** 2 / (2 * GRAVITY) <= MAX_APEX_M
    )


def transition_record(velocity, *, resolved_impacts=0):
    """Record total resolved grounds, including supplied continuation history."""
    v = np.asarray(velocity, float)
    return {
        "normal_contact_mode": MODE,
        "normal_contact_integration_policy": CACHE_TAG,
        "normal_contact_resolved_impacts": resolved_impacts,
        "normal_contact_omitted_apex_m": float(v[2] ** 2 / (2 * GRAVITY)),
        "normal_contact_apex_limit_m": MAX_APEX_M,
        "normal_contact_rolling_resistance": ROLLING_RESISTANCE,
        "normal_contact_model_calibrated": False,
    }


def advance(state, seconds, surface):
    """Analytic sliding-to-rolling step; dissipative and exact at slip arrest."""
    value = np.asarray(state, float)
    if value.shape != (9,) or not np.isfinite(value).all():
        raise ValueError("finite nine-coordinate ground contact state required")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("finite nonnegative contact duration required")
    if abs(float(value[2]) - RADIUS) > 1e-9:
        raise ValueError("ground contact must start on the physical floor")
    if any(
        np.any(np.abs(value[start : start + 3]) > limit)
        for start, limit in ((0, 1000), (3, 250), (6, 5000))
    ):
        raise ValueError("ground contact state outside search bounds")
    p, v, w = value[:3].copy(), value[3:6].copy(), value[6:9].copy()
    p[2], v[2] = RADIUS, 0.0
    friction = impact.SURFACES[surface_model.parse(surface).surface][1]
    slip = v[:2] + RADIUS * np.array([-w[1], w[0]])
    speed = float(np.linalg.norm(slip))
    remaining = float(seconds)
    if speed > 1e-12:
        direction = slip / speed
        acceleration = -friction * GRAVITY * direction
        angular = np.array([acceleration[1], -acceleration[0]]) / (impact.ALPHA * RADIUS)
        until_rolling = speed / (friction * GRAVITY * (1 + 1 / impact.ALPHA))
        elapsed = min(remaining, until_rolling)
        p[:2] += v[:2] * elapsed + 0.5 * acceleration * elapsed**2
        v[:2] += acceleration * elapsed
        w[:2] += angular * elapsed
        remaining -= elapsed
    if remaining > 0:
        # Sliding impulse reaches zero slip before this branch. Preserve its
        # rolling translation/spin relation while removing energy without a
        # sign reversal; spin about the normal is unmodeled and left unchanged.
        speed = float(np.linalg.norm(v[:2]))
        if speed > 0:
            deceleration = ROLLING_RESISTANCE * GRAVITY
            elapsed = min(remaining, speed / deceleration)
            direction = v[:2] / speed
            p[:2] += direction * (speed * elapsed - 0.5 * deceleration * elapsed**2)
            v[:2] = direction * max(0.0, speed - deceleration * elapsed)
            w[:2] = np.array([-v[1], v[0]]) / RADIUS
    return tuple(np.r_[p, v, w])
