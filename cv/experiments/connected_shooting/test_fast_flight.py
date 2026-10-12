"""The scalar propagation must be the shared physics module, bit for bit."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import fast_flight, measured_dynamics, net_collision
from physics import flight


def states(count, seed):
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(-20, 20, 3), rng.uniform(-70, 70, 3), rng.uniform(-500, 500, 3))
        for _ in range(count)
    ]


@pytest.mark.parametrize("dt", [1.0 / 240.0, 1.0 / 1000.0, 0.0031])
def test_scalar_step_is_bit_identical_to_physics(dt):
    for x, v, w in states(200, 5):
        expected = np.concatenate(flight.rk4_step(x, v, w, dt))
        actual = np.array(fast_flight.rk4_step((*x, *v, *w), dt))
        assert np.array_equal(expected, actual)


def test_zero_velocity_and_spin_take_the_gravity_only_branch():
    zero = np.zeros(3)
    expected = np.concatenate(flight.rk4_step(np.array([1.0, 2.0, 3.0]), zero, zero, 1 / 240))
    actual = np.array(fast_flight.rk4_step((1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0), 1 / 240))
    assert np.array_equal(expected, actual)


def test_exceeds_matches_the_array_reduction_including_at_the_threshold():
    rng = np.random.default_rng(9)
    for _ in range(2000):
        value = rng.uniform(-1000, 1000, 3)
        assert fast_flight.exceeds(*value, 250) == bool(np.linalg.norm(value) > 250)
    on_limit = np.array([250.0, 0.0, 0.0])
    assert fast_flight.exceeds(*on_limit, 250) is False


def simulate_pair(theta, f0, queries, surface, **kwargs):
    module = net_collision if "net_frame" in kwargs else measured_dynamics
    first = module.simulate(theta, f0, queries, 25.0, surface, **kwargs)
    measured_dynamics._TRACE_CACHE.clear()
    second = module.simulate(theta, f0, queries, 25.0, surface, **kwargs)
    return first, second


def test_retained_trace_returns_the_recomputed_value():
    theta = np.array([5.0, 2.0, 2.5, -1.0, 24.0, 3.0, 2.0, 0.0, 0.0])
    queries = np.array([100.0, 103.5, 107.0, 112.0])
    for kwargs in ({}, {"net_frame": 104.0}):
        first, second = simulate_pair(theta, 100.0, queries, "hard", **kwargs)
        for left, right in zip(first[:3], second[:3], strict=True):
            assert np.array_equal(left, right)
        assert len(first[3]) == len(second[3])


def test_a_cached_trace_is_not_mutated_by_its_caller():
    theta = np.array([5.0, 2.0, 2.5, -1.0, 24.0, -5.0, 2.0, 0.0, 0.0])
    queries = np.array([100.0, 106.0, 112.0])
    measured_dynamics._TRACE_CACHE.clear()
    first = measured_dynamics.simulate(theta, 100.0, queries, 25.0, "hard")
    for bounce in first[3]:
        bounce["x"] += 100.0
        bounce["frame"] = -1.0
    second = measured_dynamics.simulate(theta, 100.0, queries, 25.0, "hard")
    assert first[3] and len(second[3]) == len(first[3])
    assert all(row["frame"] > 0 for row in second[3])


def test_a_different_query_horizon_is_a_different_trace():
    theta = np.array([5.0, 2.0, 2.5, -1.0, 24.0, 3.0, 2.0, 0.0, 0.0])
    measured_dynamics._TRACE_CACHE.clear()
    short = measured_dynamics.simulate(theta, 100.0, np.array([100.0, 104.0]), 25.0, "hard")
    long = measured_dynamics.simulate(theta, 100.0, np.array([100.0, 104.0, 118.0]), 25.0, "hard")
    assert np.array_equal(short[0][0], long[0][0])
    assert len(measured_dynamics._TRACE_CACHE) == 2
