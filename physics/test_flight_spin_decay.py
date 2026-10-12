import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import flight


def integrate_spin(w0, v, seconds, dt=0.01):
    w = np.array([0.0, 0.0, w0], float)
    for _ in range(int(seconds / dt)):
        w = w + flight.w_dot(np.array([v, 0.0, 0.0]), w) * dt
    return np.linalg.norm(w)


def test_spin_retention_matches_real_balls():
    # 2000 rpm topspin at rally speed keeps ~95% over a 1 s flight
    w0 = 2000 * 2 * np.pi / 60
    retained = integrate_spin(w0, 30.0, 1.0) / w0
    assert 0.90 < retained < 0.985, retained


def test_zero_spin_is_stable():
    wdot = flight.w_dot(np.array([30.0, 0.0, 0.0]), np.zeros(3))
    assert np.all(np.isfinite(wdot)) and np.allclose(wdot, 0.0)


def test_decay_scales_with_spin():
    fast = np.linalg.norm(flight.w_dot(np.array([30.0, 0, 0]), np.array([0, 0, 400.0])))
    slow = np.linalg.norm(flight.w_dot(np.array([30.0, 0, 0]), np.array([0, 0, 100.0])))
    assert abs(fast / slow - 4.0) < 0.01


def test_rifle_spin_does_not_create_magnus_force():
    velocity = np.array([25.0, 0.0, 0.0])
    rifle_spin = np.array([400.0, 0.0, 0.0])
    baseline = flight.accel(velocity, np.zeros(3), lift=True)
    with_rifle_spin = flight.accel(velocity, rifle_spin, lift=True)
    np.testing.assert_allclose(with_rifle_spin, baseline)


def test_magnus_force_scales_with_perpendicular_spin_component():
    velocity = np.array([25.0, 0.0, 0.0])
    perpendicular = flight.accel(
        velocity,
        np.array([0.0, 0.0, 200.0]),
        gravity=False,
        drag=False,
    )
    tilted = flight.accel(
        velocity,
        np.array([200.0, 0.0, 200.0]),
        gravity=False,
        drag=False,
    )
    np.testing.assert_allclose(tilted, perpendicular)
