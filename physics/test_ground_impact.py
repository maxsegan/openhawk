import math

import numpy as np
import pytest

from physics.flight import default_params, ground_impact_in_interval, sample_states


VACUUM = {**default_params, "C_drag": 0.0, "C_lift": 0.0, "C_spin_decay": 0.0}


def test_descending_root_matches_analytic_ballistic_impact_without_snapping():
    position, velocity, spin = np.array([1.0, 2.0, 1.0]), np.array([3.0, 4.0, 2.0]), np.zeros(3)
    before = [value.copy() for value in (position, velocity, spin)]
    plane = 0.0325
    expected = (2.0 + math.sqrt(4.0 + 2 * 9.81 * (1.0 - plane))) / 9.81
    impact = ground_impact_in_interval(
        position, velocity, spin, expected - 0.02, expected + 0.02, plane_z_m=plane, params=VACUUM
    )
    assert impact is not None
    assert impact.time_seconds == pytest.approx(expected, abs=1e-9)
    assert impact.position[2] == pytest.approx(plane, abs=1e-9)
    assert impact.position[:2] == pytest.approx(position[:2] + velocity[:2] * expected)
    exact = sample_states(position, velocity, spin, [impact.time_seconds], params=VACUUM)
    np.testing.assert_array_equal(impact.position, exact[0][0])
    assert impact.velocity[2] < 0
    for original, current in zip(before, (position, velocity, spin), strict=True):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize("interval", [(0.0, 0.1), (2.0, 2.1)])
def test_missing_crossing_does_not_extend_or_shift_the_observation_interval(interval):
    assert (
        ground_impact_in_interval(
            np.array([0, 0, 1]),
            np.zeros(3),
            np.zeros(3),
            *interval,
            plane_z_m=0.0325,
            params=VACUUM,
        )
        is None
    )


def test_rising_initial_plane_contact_is_not_a_second_bounce():
    assert (
        ground_impact_in_interval(
            np.array([0, 0, 0.0325]),
            np.array([0, 1, 2]),
            np.zeros(3),
            0.0,
            1.0,
            plane_z_m=0.0325,
            params=VACUUM,
        )
        is None
    )


@pytest.mark.parametrize("interval", [(math.nan, 1), (0, math.inf), (-1, 1), (1, 1), (2, 1)])
def test_invalid_intervals_fail_before_integration(interval):
    with pytest.raises(ValueError, match="interval"):
        ground_impact_in_interval(
            np.array([0, 0, 1]), np.zeros(3), np.zeros(3), *interval, plane_z_m=0.0325
        )


@pytest.mark.parametrize("bad", [[0, 1], [0, math.nan, 1], [0, 0, math.inf]])
def test_invalid_states_fail_before_integration(bad):
    with pytest.raises(ValueError, match="three-vectors"):
        ground_impact_in_interval(np.array(bad), np.zeros(3), np.zeros(3), 0, 1, plane_z_m=0.0325)
