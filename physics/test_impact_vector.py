import numpy as np

from impact import ALPHA, R_BALL, court_bounce_vector


def _energy(velocity: np.ndarray, spin: np.ndarray) -> float:
    mass = 0.057
    inertia = ALPHA * mass * R_BALL**2
    return 0.5 * mass * float(velocity @ velocity) + 0.5 * inertia * float(spin @ spin)


def test_vector_bounce_is_equivariant_under_court_rotation():
    velocity = np.array([7.0, 24.0, -8.0])
    spin = np.array([180.0, -120.0, 220.0])
    angle = 0.73
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    baseline = court_bounce_vector(velocity, spin, "clay")
    rotated = court_bounce_vector(rotation @ velocity, rotation @ spin, "clay")

    np.testing.assert_allclose(rotated.velocity, rotation @ baseline.velocity)
    np.testing.assert_allclose(rotated.spin, rotation @ baseline.spin)


def test_vector_bounce_does_not_create_rigid_body_energy():
    rng = np.random.default_rng(20260729)
    for _ in range(250):
        velocity = np.r_[
            rng.uniform(-35.0, 35.0, 2),
            -rng.uniform(1.0, 16.0),
        ]
        if np.linalg.norm(velocity[:2]) < 0.1:
            continue
        spin = rng.uniform(-500.0, 500.0, 3)
        result = court_bounce_vector(velocity, spin, "hard")
        assert _energy(result.velocity, result.spin) <= _energy(velocity, spin) + 1e-10


def test_normal_axis_spin_is_unchanged_by_point_contact():
    velocity = np.array([5.0, 23.0, -7.0])
    spin = np.array([0.0, 0.0, 320.0])
    result = court_bounce_vector(velocity, spin, "grass")
    assert result.spin[2] == spin[2]
