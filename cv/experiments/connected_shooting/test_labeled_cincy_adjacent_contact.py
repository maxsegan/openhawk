"""The two local rebound overrides must remain distinct, including in the cache."""

import numpy as np

from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.experiments.connected_shooting.labeled_nonnet_terminal_fit import terminal_scales


def test_nested_local_scales_preserve_prefix_and_cache_exact_configuration():
    theta = np.array([4, 4, 1, 2, 12, -3, 0, 0, 0.0], float)
    cache = FlightCache()
    original = measured_dynamics.simulate

    def run(start):
        return cache.simulate(
            start,
            measured_dynamics.simulate,
            theta,
            start,
            np.array([start, start + 20]),
            50.0,
            "hard",
        )[0]

    baseline = {start: run(start) for start in [0, 40, 80]}
    with terminal_scales(40, (0.9, 1.1)), terminal_scales(80, (0.8, 0.95)):
        np.testing.assert_array_equal(run(0), baseline[0])
        previous = run(40)
        final = run(80)
        assert not np.array_equal(previous, baseline[40])
        assert not np.array_equal(final, baseline[80])
    with terminal_scales(40, (1.1, 0.9)), terminal_scales(80, (0.8, 0.95)):
        assert not np.array_equal(run(40), previous)
        np.testing.assert_array_equal(run(80), final)
    assert measured_dynamics.simulate is original
    for start in [0, 40, 80]:
        np.testing.assert_array_equal(run(start), baseline[start])
