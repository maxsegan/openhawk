"""The 2026-09-24 owner gate loosening is one named value, and off is the rung."""

from dataclasses import replace

import pytest

from cv.experiments.connected_shooting.per_flight_acceptance import Thresholds
from cv.pipeline.s6_owner_gate_loosening import (
    KEY,
    NEAR_MISS,
    NEAR_MISS_VALUES,
    OFF,
    ROLLBACK,
    overlay,
)


def _promoted_rung() -> Thresholds:
    return Thresholds(
        bounce_ray_limit_m=ROLLBACK["bounce_ray_limit_m"],
        serve_bounce_ray_limit_m=ROLLBACK["serve_bounce_ray_limit_m"],
        directional_rms_limit_px=32.0,
        directional_time_shift_frames=1.0,
        flight_reprojection_rms_limit_px=32.0,
        bounce_uncertainty_frames=ROLLBACK["bounce_uncertainty_frames"],
        ending_passive_context_frames=ROLLBACK["ending_passive_context_frames"],
        net_clearance_sigma_m=0.10,
        grass_bounce_epoch_frames=ROLLBACK["grass_bounce_epoch_frames"],
    )


def test_absent_and_off_leave_the_production_rung_unchanged():
    rung = _promoted_rung()
    assert overlay(rung, {}) == rung
    assert overlay(rung, {KEY: OFF}) == rung


def test_the_named_value_is_the_measured_near_miss_setting():
    loosened = overlay(_promoted_rung(), {KEY: NEAR_MISS})
    for name, value in NEAR_MISS_VALUES.items():
        assert getattr(loosened, name) == value
    # The ordinary directional limit is not what the waiver changes.
    assert loosened.directional_rms_limit_px == 32.0


def test_an_unknown_value_is_refused():
    with pytest.raises(ValueError, match=KEY):
        overlay(_promoted_rung(), {KEY: "wider"})


def test_the_waiver_cannot_reach_the_windows_the_measurement_left_rejected():
    # 55 px is the first directional-only window the rescore did not accept.
    with pytest.raises(ValueError, match="48"):
        replace(_promoted_rung(), directional_only_waiver_px=55.0).validate()
    Thresholds(directional_only_waiver_px=48.0).validate()
    with pytest.raises(ValueError, match="3.5"):
        Thresholds(bounce_uncertainty_frames=4.0).validate()
