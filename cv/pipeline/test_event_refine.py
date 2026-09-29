import numpy as np

from event_refine import refine_anchor


def test_refine_anchor_recovers_piecewise_linear_knot():
    times = np.arange(0, 21, dtype=float)
    x = np.where(times <= 10, times, 20 - times)
    y = np.where(times <= 10, 2 * times, 10 + times)

    result = refine_anchor(times, x, y, hint=9.0)

    assert result.ok
    assert abs(result.tau - 10.0) <= 0.1
    assert result.angle_change_deg > 45.0
    assert result.velocity_jump > 1.0


def test_refine_anchor_abstains_without_two_sided_support():
    times = np.arange(0, 4, dtype=float)

    result = refine_anchor(times, times, times, hint=2.0)

    assert not result.ok
    assert result.reason == "insufficient_window_support"
