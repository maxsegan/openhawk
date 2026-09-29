import numpy as np
import pytest

from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.labeled_nonnet_terminal_fit import terminal_scales


def test_terminal_scale_override_preserves_other_flights_and_restores_simulator():
    theta = np.array([4, 4, 1, 2, 12, -3, 0, 0, 0.0], float)

    def run(start):
        return measured_dynamics.simulate(theta, start, np.array([start, start + 20]), 50.0, "hard")

    before = run(0)
    last = run(40)
    original = measured_dynamics.simulate
    with terminal_scales(40, (1.0, 1.0)):
        for a, b in zip(last[:3], run(40)[:3], strict=True):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(RuntimeError):
        with terminal_scales(40, (0.8, 0.8)):
            for a, b in zip(before[:3], run(0)[:3], strict=True):
                np.testing.assert_array_equal(a, b)
            assert not np.array_equal(last[0], run(40)[0])
            raise RuntimeError("intentional restore check")
    assert measured_dynamics.simulate is original
    for a, b in zip(last[:3], run(40)[:3], strict=True):
        np.testing.assert_array_equal(a, b)
