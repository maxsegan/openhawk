"""Measured-bounce reference: coefficient contract and physical behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics import bounce_reference, impact


CORPUS_SURFACES = ("clay", "hard")


def test_only_measured_surfaces_are_answerable():
    assert bounce_reference.surfaces() == ("clay", "grass", "hard")
    with pytest.raises(ValueError, match="carpet"):
        bounce_reference.restitution(16.0, "carpet")


def test_grass_is_declared_as_the_thin_broadcast_measurement_it_is():
    """Grass is not corpus evidence and must not be presented as if it were."""
    source = bounce_reference.PROVENANCE["grass_source"]
    assert source["human_derived"] is True
    assert source["spin_resolved"] is False
    assert bounce_reference.MEASURED["grass"]["n"] == source["impacts"] < 100
    # A grass court is lower and faster than a hard court.  Restitution may be
    # compared across the two instruments, because the estimator reproduces the
    # corpus there; retention may not, because the estimator reads it low on
    # both measured surfaces, so grass is compared against the estimator's own
    # hard control instead of against the corpus row.
    assert bounce_reference.restitution(16.0, "grass") < bounce_reference.restitution(16.0, "hard")
    control = source["control_on_measured_surfaces"]["hard"]
    assert (
        abs(control["estimated_restitution"] - control["corpus_restitution_at_model_spin"]) < 0.05
    )
    assert control["estimated_retention"] < control["corpus_retention_at_model_spin"]
    assert bounce_reference.horizontal_retention(16.0, "grass") > control["estimated_retention"]
    assert "reads horizontal retention low" in source["known_bias"]


def test_every_surface_carries_both_spin_conditions():
    for surface, row in bounce_reference.MEASURED.items():
        assert row["n"] > (500 if surface in CORPUS_SURFACES else 10), surface
        for variant in ("at_model_spin", "spin_free"):
            for quantity in ("restitution", "retention"):
                assert f"{quantity}_intercept_{variant}" in row
                assert f"{quantity}_slope_{variant}" in row
                assert f"{quantity}_residual_std_{variant}" in row


def test_intercept_is_the_value_at_sixteen_degrees():
    for surface in bounce_reference.surfaces():
        row = bounce_reference.MEASURED[surface]
        assert bounce_reference.restitution(16.0, surface) == pytest.approx(
            row["restitution_intercept_at_model_spin"], abs=1e-9
        )
        assert bounce_reference.horizontal_retention(16.0, surface) == pytest.approx(
            row["retention_intercept_at_model_spin"], abs=1e-9
        )


def test_spin_free_band_is_below_the_model_spin_value():
    """The corpus cannot separate rebound spin from rebound speed; both ends are published.

    Grass is excluded: a broadcast arc cannot separate them either, so its two
    columns hold the same measured ratio rather than a fabricated band.
    """
    for surface in CORPUS_SURFACES:
        assert bounce_reference.restitution(16.0, surface, spin_free=True) < (
            bounce_reference.restitution(16.0, surface)
        )
        assert bounce_reference.horizontal_retention(16.0, surface, spin_free=True) < (
            bounce_reference.horizontal_retention(16.0, surface)
        )


def test_bounce_reverses_vertical_and_keeps_horizontal_direction():
    velocity = np.array([12.0, -5.0, -8.0])
    spin = np.array([0.0, 0.0, 0.0])
    result = bounce_reference.court_bounce(velocity, spin, "hard")
    assert result.velocity[2] > 0.0
    incoming_heading = math.atan2(velocity[1], velocity[0])
    outgoing_heading = math.atan2(result.velocity[1], result.velocity[0])
    assert outgoing_heading == pytest.approx(incoming_heading, abs=1e-9)
    assert float(np.hypot(*result.velocity[:2])) < float(np.hypot(*velocity[:2]))
    assert 0.0 < result.restitution < 1.0
    assert 0.0 < result.horizontal_retention < 1.0


def test_bounce_refuses_a_rising_ball():
    with pytest.raises(ValueError, match="descending"):
        bounce_reference.court_bounce(np.array([10.0, 0.0, 3.0]), np.zeros(3), "hard")


def test_outgoing_spin_comes_from_the_cross_model_and_says_so():
    velocity = np.array([18.0, 0.0, -6.0])
    spin = np.array([0.0, 200.0, 0.0])
    result = bounce_reference.court_bounce(velocity, spin, "clay")
    expected = impact.court_bounce(18.0, 6.0, 200.0, surface="clay")
    assert float(result.spin[1]) == pytest.approx(expected.w2, rel=1e-9)
    assert "not measurable" in result.spin_source


def test_rifle_spin_is_carried_through_untouched():
    result = bounce_reference.court_bounce(
        np.array([15.0, 0.0, -6.0]), np.array([0.0, 100.0, 37.5]), "hard"
    )
    assert float(result.spin[2]) == pytest.approx(37.5)


def test_restitution_falls_with_incidence_angle():
    steep = bounce_reference.restitution(35.0, "hard")
    shallow = bounce_reference.restitution(10.0, "hard")
    assert steep < shallow


def test_dwell_time_is_the_reference_value_not_a_corpus_measurement():
    assert bounce_reference.DWELL_SECONDS == pytest.approx(0.0045)
    assert bounce_reference.PROVENANCE["outgoing_spin_identifiable"] is False


def test_production_law_never_saw_the_hawkeye_comparison_matches():
    """The default law is fitted without every Hawk-Eye match that is also a broadcast evaluation match."""
    assert bounce_reference.DEFAULT_LAW == "hawkeye_holdout"
    provenance = bounce_reference.LAWS["hawkeye_holdout"][1]
    fitted = set(provenance["hawkeye_matches_fitted"])
    excluded = set(provenance["hawkeye_matches_excluded"])
    assert len(excluded) == 51 and not fitted & excluded
    assert set(provenance["hawkeye_matches_excluded_from_fit_sample"]) <= excluded
    record = bounce_reference.law_record()
    assert record["law"] == bounce_reference.LAW_NAME


def test_holdout_law_stays_within_a_percent_of_the_full_corpus_law():
    holdout = bounce_reference.LAWS["hawkeye_holdout"][0]
    full = bounce_reference.LAWS["full_corpus"][0]
    assert holdout["grass"] == full["grass"]
    for surface in CORPUS_SURFACES:
        for key, value in full[surface].items():
            if "intercept" in key:
                assert holdout[surface][key] == pytest.approx(value, rel=0.01), (surface, key)


def test_unknown_bounce_law_is_refused(monkeypatch):
    monkeypatch.setenv(bounce_reference.LAW_VARIABLE, "nonexistent")
    with pytest.raises(ValueError, match="known bounce laws"):
        bounce_reference._selected_law()
    monkeypatch.setenv(bounce_reference.LAW_VARIABLE, "full_corpus")
    assert bounce_reference._selected_law() == "full_corpus"
