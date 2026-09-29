import numpy as np
import pytest

from physics.flight import (
    _batch_derivatives,
    accel,
    default_params,
    sample_signed_states,
    sample_states,
    sample_states_batch,
    w_dot,
)


def test_signed_samples_do_not_depend_on_other_queries() -> None:
    initial = (np.ones(3), np.array([3.0, 12.0, -2.0]), np.array([100.0, 20.0, 0.0]))
    times = np.array([0.17, -0.31, 0.0, -0.12, 0.17])
    batched = sample_signed_states(*initial, times)
    for index, time in enumerate(times):
        scalar = sample_signed_states(*initial, np.array([time]))
        for batch_value, scalar_value in zip(batched, scalar, strict=True):
            np.testing.assert_array_equal(batch_value[index], scalar_value[0])


def test_signed_samples_reject_nonfinite_or_nonscalar_queries() -> None:
    for times in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2))):
        with np.testing.assert_raises_regex(ValueError, "finite one-dimensional"):
            sample_signed_states(np.zeros(3), np.zeros(3), np.zeros(3), times)


def test_sample_states_preserves_query_order_and_initial_state() -> None:
    times = np.array([0.2, 0.0, 0.1])
    positions, velocities, spins = sample_states(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.zeros(3),
        times,
    )

    np.testing.assert_allclose(positions[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(velocities[1], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(spins, np.zeros((3, 3)))
    assert positions[0, 1] > positions[2, 1] > positions[1, 1]


def test_sample_states_rejects_negative_time() -> None:
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        sample_states(np.zeros(3), np.zeros(3), np.zeros(3), np.array([-0.1]))


def test_batched_samples_match_scalar_flights() -> None:
    positions = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    velocities = np.array([[4.0, 5.0, 6.0], [-4.0, 8.0, 2.0]])
    spins = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 1.0]])
    times = np.array([0.0, 0.12, 0.3])

    batched = sample_states_batch(positions, velocities, spins, times)

    for index in range(2):
        scalar = sample_states(positions[index], velocities[index], spins[index], times)
        for batched_state, scalar_state in zip(batched, scalar):
            np.testing.assert_allclose(batched_state[index], scalar_state, atol=1e-10)


@pytest.mark.parametrize("spin", [0.0, 1e-13, 1e-8, 0.5, 1.0, 1.01, 2.0, -0.5])
def test_batched_low_spin_samples_match_scalar(spin: float) -> None:
    position = np.array([5.0, 2.0, 3.0])
    velocity = np.array([3.0, 30.0, 5.0])
    angular = np.array([spin, 0.0, 0.0])
    times = np.array([1.0, 0.1, 0.0, 0.5, 0.1])
    scalar = sample_states(position, velocity, angular, times)
    batched = sample_states_batch(position[None], velocity[None], angular[None], times)
    for actual, expected in zip(batched, scalar, strict=True):
        np.testing.assert_allclose(actual[0], expected, rtol=1e-13, atol=1e-13)


def test_batched_derivatives_use_scalar_zero_speed_and_spin_guards() -> None:
    velocities = np.array([[0.0, 0.0, 0.0], [1e-13, 0.0, 0.0], [3.0, 30.0, 5.0]])
    spins = np.array([[100.0, 20.0, 0.0], [0.0, 1e12, 0.0], [0.5, 0.0, 0.0]])
    accelerations, spin_rates = _batch_derivatives(velocities, spins, default_params)
    for i, (velocity, spin) in enumerate(zip(velocities, spins, strict=True)):
        np.testing.assert_allclose(accelerations[i], accel(velocity, spin), atol=1e-14)
        np.testing.assert_allclose(spin_rates[i], w_dot(velocity, spin), atol=1e-14)


def test_batched_spin_finite_difference_matches_scalar_at_zero() -> None:
    position = np.array([5.0, 2.0, 3.0])
    velocity = np.array([3.0, 30.0, 5.0])
    times = np.array([0.1, 0.5, 1.0])
    step = 1e-4
    base = sample_states(position, velocity, np.zeros(3), times)[0]
    angular = np.eye(3) * step
    batched = sample_states_batch(
        np.repeat(position[None], 3, axis=0),
        np.repeat(velocity[None], 3, axis=0),
        angular,
        times,
    )[0]
    for axis in range(3):
        expected = (sample_states(position, velocity, angular[axis], times)[0] - base) / step
        assert np.linalg.norm(expected) > 1e-4
        np.testing.assert_allclose((batched[axis] - base) / step, expected, atol=1e-9)
