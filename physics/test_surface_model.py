"""The region- and day-aware surface is a modifier, and an unqualified surface is untouched."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics import bounce_reference, impact, surface_model


def test_a_bare_surface_name_is_fully_inert():
    for name in surface_model.surfaces():
        model = surface_model.parse(name)
        assert model.inert
        assert not model.line_switch
        assert surface_model.format_spec(model) == name


def test_an_inert_surface_reproduces_the_measured_bounce_exactly():
    velocity = np.array([1.0, -22.0, -7.0])
    spin = np.array([0.0, 0.0, 30.0])
    for surface in ("hard", "clay", "grass"):
        legacy = bounce_reference.court_bounce(velocity, spin, surface)
        # Even standing on a painted line, an unqualified name changes nothing.
        spec = bounce_reference.court_bounce(
            velocity, spin, f"{surface}@w=1.0,a=0.0,noline", position=[5.485, 10.0]
        )
        assert spec.restitution == pytest.approx(legacy.restitution, abs=1e-12)
        assert spec.horizontal_retention == pytest.approx(legacy.horizontal_retention, abs=1e-12)
        assert np.allclose(spec.velocity, legacy.velocity, atol=1e-12)
        assert np.allclose(spec.spin, legacy.spin, atol=1e-12)


def test_regions_follow_the_literature_kernel():
    assert surface_model.landing_region(5.0, 1.0) == "baseline"
    assert surface_model.landing_region(5.0, 22.8) == "baseline"
    assert surface_model.landing_region(5.0, 4.0) == "backcourt"
    assert surface_model.landing_region(5.0, 8.0) == "service_box"
    assert surface_model.landing_region(5.0, 11.5) == "near_net"
    assert surface_model.region_kernel(5.0, 1.0) == 1.0
    assert surface_model.region_kernel(5.0, 11.5) == -0.2


def test_lines_are_finite_segments():
    # The centre service line exists only between the two service lines.
    assert surface_model.line_distance_m(5.485, 10.0)[1] == "centre_service_line"
    assert surface_model.line_distance_m(5.485, 10.0)[0] == pytest.approx(0.0)
    assert surface_model.line_distance_m(5.485, 2.0)[1] != "centre_service_line"
    # A service line does not continue into the doubles alley.
    assert surface_model.line_distance_m(0.5, 5.485)[1] != "service_line"
    assert surface_model.on_line(5.0, 0.05)
    assert not surface_model.on_line(5.0, 1.0)
    # Nor do the baseline and sidelines extend beyond their physical endpoints.
    assert not surface_model.on_line(-1.0, -1.0)


def test_the_kernel_peak_reproduces_the_worn_grass_pair():
    """At the amplitude that takes fresh grass to worn grass, both coefficients arrive."""
    literature = surface_model.LITERATURE["grass"]
    amplitude = literature.worn_restitution_16deg / literature.fresh_restitution_16deg - 1.0
    model = surface_model.SurfaceModel(
        surface="grass", wear=1.0, amplitude=amplitude, base_source="literature"
    )
    state = model.state(5.0, 1.0)
    assert state.region == "baseline" and state.kernel == 1.0
    assert literature.fresh_restitution_16deg * state.restitution_factor == pytest.approx(
        literature.worn_restitution_16deg, abs=1e-6
    )
    assert literature.fresh_friction * state.friction_factor == pytest.approx(
        literature.worn_friction, abs=1e-6
    )


def test_fresh_grass_is_not_the_chart_bulk_speed_ratio():
    """The execution-confounded arm shrank grass toward 0.60; Cross's fresh pair is 0.72."""
    restitution, retention = surface_model.fresh_base("grass", 16.0)
    assert restitution == pytest.approx(0.72)
    assert 0.6 < retention < 0.8
    assert restitution > bounce_reference.restitution(16.0, "grass") + 0.10


def test_wear_raises_both_coefficients_toward_a_worn_court():
    velocity = np.array([0.0, -22.0, -7.0])
    spin = np.zeros(3)
    fresh = bounce_reference.court_bounce(
        velocity, spin, "grass@w=1.0,a=0.0,base=fresh", position=[5.0, 1.0]
    )
    worn = bounce_reference.court_bounce(
        velocity, spin, "grass@w=1.10,a=0.18,base=fresh", position=[5.0, 1.0]
    )
    assert worn.restitution > fresh.restitution
    # More friction means less horizontal speed kept.
    assert worn.horizontal_retention < fresh.horizontal_retention


def test_the_line_switch_is_a_skid_and_only_on_a_line():
    velocity = np.array([0.0, -22.0, -7.0])
    spin = np.zeros(3)
    on = bounce_reference.court_bounce(velocity, spin, "clay@w=1.0,a=0.0", position=[5.485, 10.0])
    off = bounce_reference.court_bounce(velocity, spin, "clay@w=1.0,a=0.0", position=[4.0, 10.0])
    assert on.surface_state.on_line and not off.surface_state.on_line
    assert on.horizontal_retention > off.horizontal_retention
    assert on.restitution > off.restitution


def test_the_active_model_needs_a_landing_point():
    with pytest.raises(ValueError, match="landing point"):
        bounce_reference.court_bounce(
            np.array([0.0, -22.0, -7.0]), np.zeros(3), "grass@w=1.05,a=0.15,base=fresh"
        )


def test_explicit_cross_coefficients_reproduce_the_chart_lookup():
    chart = impact.court_bounce(20.0, 8.0, 200.0, "clay")
    theta = math.degrees(math.atan2(8.0, 20.0))
    explicit = impact.court_bounce(
        20.0,
        8.0,
        200.0,
        "clay",
        coefficients=(impact.court_ey(theta, "clay"), impact.SURFACES["clay"][1]),
    )
    assert explicit.vx2 == pytest.approx(chart.vx2, abs=1e-12)
    assert explicit.vy2 == pytest.approx(chart.vy2, abs=1e-12)
    assert explicit.w2 == pytest.approx(chart.w2, abs=1e-12)
    assert explicit.regime == chart.regime


def test_a_spec_survives_a_round_trip_through_one_command_line_argument():
    model = surface_model.SurfaceModel(
        surface="grass",
        wear=1.0625,
        amplitude=0.1875,
        base_source="literature",
        provenance="spec",
    )
    assert surface_model.parse(surface_model.format_spec(model)) == model


@pytest.mark.parametrize("spec", ["clay@w=9.0", "clay@a=5.0", "clay@z=1", "not a surface"])
def test_unreadable_or_unphysical_specs_are_refused(spec):
    with pytest.raises((ValueError, KeyError)):
        surface_model.parse(spec)


def test_the_environment_selector_is_off_unless_it_is_set(monkeypatch):
    """The arm hook the sweep uses: unset, every bare name is the fitter as it stands."""
    monkeypatch.delenv(surface_model.OVERRIDE_VARIABLE, raising=False)
    assert surface_model.overrides() == {}
    assert surface_model.parse("grass").inert
    monkeypatch.setenv(
        surface_model.OVERRIDE_VARIABLE,
        '{"grass": "grass@w=1.0600,a=0.1500,base=fresh"}',
    )
    selected = surface_model.parse("grass")
    assert not selected.inert and selected.wear == pytest.approx(1.06)
    # An entry may not quietly rename the surface it stands for.
    assert surface_model.parse("clay").inert
    monkeypatch.setenv(surface_model.OVERRIDE_VARIABLE, '{"grass": "clay@w=1.05"}')
    with pytest.raises(ValueError, match="names surface"):
        surface_model.overrides()


def test_the_measured_region_map_has_the_opposite_sign_to_the_literature_kernel():
    """The corpus says the baseline strip is the low-restitution cell, not the high one."""
    for surface in ("clay", "hard"):
        measured = surface_model.MEASURED_REGION_MAP[surface]
        assert measured["restitution"]["baseline"] < measured["restitution"]["backcourt"]
        assert measured["retention"]["baseline"] < measured["retention"]["backcourt"]
    literature = surface_model.SurfaceModel(
        surface="grass", wear=1.0, amplitude=0.15, base_source="literature"
    )
    assert literature.state(5.0, 1.0).restitution_factor > 1.0


def test_the_measured_map_is_recentred_so_a_typical_bounce_is_unmoved():
    """A sample-weighted random corpus bounce sees no net shift, which is what keeps the
    existing measured base calibration intact."""
    shares = {"backcourt": 0.52, "baseline": 0.18, "service_box": 0.29, "near_net": 0.007}
    for surface in ("clay", "hard"):
        measured = surface_model.MEASURED_REGION_MAP[surface]
        for key in ("restitution", "retention"):
            weighted = sum(shares[r] * measured[key][r] for r in shares)
            assert abs(weighted) < 0.01, (surface, key, weighted)


def test_the_day_term_is_carried_but_measures_nothing():
    """21,810 corpus impacts put the per-round slope inside its own standard error on both
    surfaces, so a final and a first round come out within 0.01 of each other."""
    velocity = np.array([0.0, -22.0, -7.0])
    spin = np.zeros(3)
    for surface in ("clay", "hard"):
        measured = surface_model.MEASURED_REGION_MAP[surface]
        assert abs(measured["day_restitution_per_round"]) <= (
            2.0 * measured["day_restitution_per_round_se"]
        )
        first = bounce_reference.court_bounce(
            velocity, spin, f"{surface}@region=measured,r=1", position=[5.0, 4.0]
        )
        final = bounce_reference.court_bounce(
            velocity, spin, f"{surface}@region=measured,r=7", position=[5.0, 4.0]
        )
        assert abs(first.restitution - final.restitution) < 0.01


def test_the_measured_map_does_not_also_apply_the_literature_line_switch():
    velocity = np.array([0.0, -22.0, -7.0])
    spin = np.zeros(3)
    on = bounce_reference.court_bounce(
        velocity, spin, "clay@region=measured", position=[5.485, 10.0]
    )
    off = bounce_reference.court_bounce(
        velocity, spin, "clay@region=measured", position=[4.0, 10.0]
    )
    # The corpus line effect on clay is +0.005 restitution and +0.007 retention, both inside
    # their own standard error -- an order of magnitude smaller than the literature switch's
    # +0.045 restitution and +0.13 retention.
    assert on.restitution - off.restitution == pytest.approx(0.0053, abs=1e-3)
    assert on.horizontal_retention - off.horizontal_retention == pytest.approx(0.0074, abs=1e-3)
    literature_on = bounce_reference.court_bounce(
        velocity, spin, "clay@w=1.0,a=0.0", position=[5.485, 10.0]
    )
    literature_off = bounce_reference.court_bounce(
        velocity, spin, "clay@w=1.0,a=0.0", position=[4.0, 10.0]
    )
    assert (literature_on.horizontal_retention - literature_off.horizontal_retention) > 10.0 * (
        on.horizontal_retention - off.horizontal_retention
    )
