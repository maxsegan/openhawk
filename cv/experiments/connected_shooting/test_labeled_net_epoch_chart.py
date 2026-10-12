import numpy as np
import pytest
from scipy.optimize import brentq

from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.labeled_net_epoch_chart import project_net_velocity


def test_recovers_original_incoming_velocity_at_independently_solved_epoch():
    theta = np.array([5.0, 20.0, 2.0, 1.0, -18.0, 1.0, 0.0, 0.0, 0.0])

    def y(epoch):
        positions, *_ = measured_dynamics.simulate(theta, 0.0, np.array([0.0, epoch]), 60.0, "hard")
        return positions[-1, 1] - 11.885

    epoch = brentq(y, 20.0, 40.0)
    incorrect = theta.copy()
    incorrect[4] = -10.0
    result, receipt = project_net_velocity(incorrect, 0.0, epoch, 60.0, "hard")
    np.testing.assert_allclose(result, theta, atol=1e-7)
    assert receipt["plane_error_m"] < 1e-7
    assert receipt["prior_ground_impacts"] == 0


def test_rejects_a_ground_bounce_before_the_declared_first_net():
    theta = np.array([5.0, 20.0, 0.2, 1.0, -8.0, -2.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="earlier ground impact"):
        project_net_velocity(theta, 0.0, 90.0, 60.0, "hard")
