"""Regression pins for the unified court_bounce_vector (2026-07-30 fix).

The vector model must reduce EXACTLY to the calibrated scalar model in the
planar case (both regimes, including the deformation-offset D terms), and must
be continuous in roll spin — the old dispatch had a step discontinuity at
|roll_spin| = 1e-12 that sat directly inside the spin-fitting objective.
"""
import numpy as np

from physics.impact import court_bounce, court_bounce_vector


CASES = [
    (20.0, 8.0, 0.0, "clay"),
    (20.0, 8.0, 300.0, "hard"),
    (35.0, 6.0, -150.0, "grass"),   # slide regime
    (10.0, 12.0, 500.0, "hard"),
    (15.0, 5.0, 50.0, "clay"),
]


def _vector_planar(vx, vy, w, surface):
    return court_bounce_vector(
        np.array([vx, 0.0, -vy]), np.array([0.0, w, 0.0]), surface=surface)


def test_planar_parity_both_regimes():
    for vx, vy, w, surface in CASES:
        s = court_bounce(vx, vy, w, surface=surface)
        v = _vector_planar(vx, vy, w, surface)
        assert v.regime == s.regime
        np.testing.assert_allclose(v.velocity[0], s.vx2, atol=1e-9)
        np.testing.assert_allclose(v.velocity[2], s.vy2, atol=1e-9)
        np.testing.assert_allclose(v.spin[1], s.w2, atol=1e-9)


def test_continuity_in_roll_spin():
    base = _vector_planar(20.0, 8.0, 100.0, "clay")
    eps = court_bounce_vector(
        np.array([20.0, 0.0, -8.0]), np.array([1e-6, 100.0, 0.0]), surface="clay")
    assert float(np.linalg.norm(eps.velocity - base.velocity)) < 1e-4
    assert float(np.linalg.norm(eps.spin - base.spin)) < 1e-2


def test_no_rigid_body_energy_creation():
    rng = np.random.default_rng(1)
    inertia = 0.5 * 0.4 * 0.033 ** 2
    for _ in range(500):
        v = np.array([rng.uniform(1, 40), rng.uniform(-10, 10), -rng.uniform(1, 15)])
        w = rng.uniform(-300, 300, 3)
        out = court_bounce_vector(v, w, surface="hard")
        e_in = 0.5 * float(v @ v) + inertia * float(w @ w)
        e_out = (0.5 * float(out.velocity @ out.velocity)
                 + inertia * float(out.spin @ out.spin))
        assert e_out <= e_in + 1e-9
