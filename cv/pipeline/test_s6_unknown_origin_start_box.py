"""The unknown-origin start box is one named value, and off is the rung."""

import pytest

from cv.experiments.connected_shooting.per_flight_acceptance import Thresholds
from cv.pipeline.s6_unknown_origin_start_box import KEY, OFF, ON, overlay


def test_absent_and_off_leave_the_rung_unchanged_and_on_sets_only_the_box():
    rung = Thresholds(directional_rms_limit_px=32.0)
    assert overlay(rung, {}) == rung
    assert overlay(rung, {KEY: OFF}) == rung
    boxed = overlay(rung, {KEY: ON})
    assert boxed.unknown_origin_start_box
    assert boxed.directional_rms_limit_px == 32.0
    with pytest.raises(ValueError, match=KEY):
        overlay(rung, {KEY: "wide"})
