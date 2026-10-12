from __future__ import annotations

import numpy as np
import pytest

from physics import bounce_reference, grass_surface_model, surface_model


def test_no_broadcast_evidence_is_exactly_the_hard_reference() -> None:
    model = grass_surface_model.regional_v2_model("unseen_grass_broadcast")
    assert model.wear == 1.0
    assert model.amplitude == 0.0
    assert not model.line_switch
    velocity = np.array([2.0, -23.0, -7.0])
    spin = np.array([0.0, 0.0, 40.0])
    hard = bounce_reference.court_bounce(velocity, spin, "hard")
    # The empty-evidence prior needs no landing location because it has no regional term.
    grass = bounce_reference.court_bounce(velocity, spin, model)
    assert grass.restitution == hard.restitution
    assert grass.horizontal_retention == hard.horizontal_retention
    assert np.array_equal(grass.velocity, hard.velocity)
    assert np.array_equal(grass.spin, hard.spin)


def test_regional_v2_has_three_bounded_broadcast_scalars() -> None:
    for match_id, row in grass_surface_model.REGIONAL_V2_BROADCASTS.items():
        model = grass_surface_model.regional_v2_model(match_id)
        assert 0.96 <= model.wear <= 1.04
        assert -0.04 <= float(model.amplitude) <= 0.04
        assert model.line_switch is row["line_switch"]
        assert row["evidence_bounces"] >= 3
        assert model.base_source == "hard_reference"


def test_hard_reference_spec_round_trips() -> None:
    model = surface_model.SurfaceModel(
        surface="grass",
        wear=0.9876,
        amplitude=0.0123,
        line_switch=False,
        base_source="hard_reference",
        provenance="spec",
    )
    assert surface_model.parse(surface_model.format_spec(model)) == model
    with pytest.raises(ValueError, match="only defined for grass"):
        surface_model.SurfaceModel(surface="clay", base_source="hard_reference")


@pytest.mark.parametrize(
    ("match_id", "week"),
    [
        ("wim2025r32_w_sabalenka_raducanu", "week_1"),
        ("wim2025f_m_sinner_alcaraz", "week_2"),
        ("rg2025qf_w_boisson_andreeva", "week_2"),
        ("ao2026r128_m_bublik_brooksby", "week_1"),
        ("atp_2024_540_r16_217_taylor_fritz_alexander_zverev", "week_1"),
    ],
)
def test_tournament_week_comes_only_from_the_match_id(match_id: str, week: str) -> None:
    assert grass_surface_model.tournament_week(match_id) == week


def test_empty_fit_returns_the_exact_prior() -> None:
    assert grass_surface_model.fit_regional_v2([]) == {}


def test_line_switch_is_live_when_a_broadcast_has_stable_contrast() -> None:
    def row(surface: str, ratio: float, line_distance: float) -> dict:
        return {
            "surface": surface,
            "match_id": "grass_test",
            "landing_region": "service_box",
            "region_kernel": 0.0,
            "line_distance_m": line_distance,
            "post_pre_speed_ratio": ratio,
            "impact_ray_disagreement_m": 0.1,
            "incoming_linear_rms_px": 1.0,
            "outgoing_linear_rms_px": 1.0,
        }

    rows = [row("hard", 1.0, 1.0), row("hard", 1.1, 1.0)]
    rows += [row("grass", 2.0, 0.04), row("grass", 2.2, 0.05), row("grass", 1.0, 1.0)]

    assert grass_surface_model.fit_regional_v2(rows)["grass_test"]["line_switch"]
