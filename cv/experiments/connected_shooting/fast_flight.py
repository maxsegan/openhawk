"""Bit-identical scalar rewrite of the connected-search propagation inner loop.

Question this answers: the whole-point search spends nearly all of its time in
``physics.flight.rk4_step`` at 240 Hz, where every three-vector operation pays a
NumPy dispatch.  This module performs exactly the same double-precision
arithmetic, in exactly the same order, on Python floats.  ``test_fast_flight.py``
asserts bit-for-bit equality against the shared physics module over randomized
states, so replacing one with the other cannot change a fitted family.

It is an optimization of an existing model, not a new model: no coefficient,
step size, ordering or tolerance is changed here.
"""

from __future__ import annotations

import math

import numpy as np

from physics import flight


_P = flight.default_params
G = _P["g"]
K = 0.5 * _P["rho_air"] * np.pi * _P["R_ball"] ** 2 / _P["m_ball"]
K4 = 0.5 * _P["rho_air"] * np.pi * _P["R_ball"] ** 4 / _P["J_ball"]
C_DRAG = _P["C_drag"]
LIFT_SCALE = K * _P["C_lift"] * _P["R_ball"]
SPIN_DECAY = -_P.get("C_spin_decay", 0.025) * K4


def derivatives(vx, vy, vz, wx, wy, wz):
    """Return ``(accel, w_dot)`` components; same order of operations as physics."""
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    ax = 0.0
    ay = 0.0
    az = -G
    if vmag > 1e-12:
        drag_scale = -K * C_DRAG * vmag
        ax += drag_scale * vx
        ay += drag_scale * vy
        az += drag_scale * vz
        if wmag > 1e-12:
            ax += LIFT_SCALE * (wy * vz - wz * vy)
            ay += LIFT_SCALE * (wz * vx - wx * vz)
            az += LIFT_SCALE * (wx * vy - wy * vx)
    spin_scale = SPIN_DECAY * vmag
    return ax, ay, az, spin_scale * wx, spin_scale * wy, spin_scale * wz


def rk4_step(state, dt):
    """Advance ``(x, v, w)`` packed as nine floats by one RK4 step."""
    px, py, pz, vx, vy, vz, wx, wy, wz = state
    a1x, a1y, a1z, s1x, s1y, s1z = derivatives(vx, vy, vz, wx, wy, wz)

    m1vx = vx + a1x * dt / 2
    m1vy = vy + a1y * dt / 2
    m1vz = vz + a1z * dt / 2
    m1wx = wx + s1x * dt / 2
    m1wy = wy + s1y * dt / 2
    m1wz = wz + s1z * dt / 2
    a2x, a2y, a2z, s2x, s2y, s2z = derivatives(m1vx, m1vy, m1vz, m1wx, m1wy, m1wz)

    m2vx = vx + a2x * dt / 2
    m2vy = vy + a2y * dt / 2
    m2vz = vz + a2z * dt / 2
    m2wx = wx + s2x * dt / 2
    m2wy = wy + s2y * dt / 2
    m2wz = wz + s2z * dt / 2
    a3x, a3y, a3z, s3x, s3y, s3z = derivatives(m2vx, m2vy, m2vz, m2wx, m2wy, m2wz)

    evx = vx + a3x * dt
    evy = vy + a3y * dt
    evz = vz + a3z * dt
    ewx = wx + s3x * dt
    ewy = wy + s3y * dt
    ewz = wz + s3z * dt
    a4x, a4y, a4z, s4x, s4y, s4z = derivatives(evx, evy, evz, ewx, ewy, ewz)

    return (
        px + dt * (vx + 2 * m1vx + 2 * m2vx + evx) / 6,
        py + dt * (vy + 2 * m1vy + 2 * m2vy + evy) / 6,
        pz + dt * (vz + 2 * m1vz + 2 * m2vz + evz) / 6,
        vx + dt * (a1x + 2 * a2x + 2 * a3x + a4x) / 6,
        vy + dt * (a1y + 2 * a2y + 2 * a3y + a4y) / 6,
        vz + dt * (a1z + 2 * a2z + 2 * a3z + a4z) / 6,
        wx + dt * (s1x + 2 * s2x + 2 * s3x + s4x) / 6,
        wy + dt * (s1y + 2 * s2y + 2 * s3y + s4y) / 6,
        wz + dt * (s1z + 2 * s2z + 2 * s3z + s4z) / 6,
    )


def exceeds(px, py, pz, limit):
    """Match ``np.linalg.norm(...) > limit`` exactly at a fraction of the cost.

    The scalar sum can differ from NumPy's reduction in the last bit, so the
    rare near-threshold comparison falls back to the array reduction and the
    decision is provably the one the previous code took.
    """
    value = math.sqrt(px * px + py * py + pz * pz)
    if abs(value - limit) <= 1e-9 * limit:
        value = float(np.linalg.norm(np.array((px, py, pz))))
    return value > limit
