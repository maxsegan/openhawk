from types import SimpleNamespace

import numpy as np

from physics_interpretation import (
    SpinPrior,
    bounce_grammar_cost,
    branch_is_decisive,
    build_point_branches,
    refined_branch_score,
    shot_spin_options,
    typical_spin_prior,
)


def test_serve_prior_trades_speed_for_spin():
    slow_center, _ = typical_spin_prior(
        gender="women",
        phase="serve",
        speed_kmh=120.0,
        profile="high_spin_serve",
    )
    fast_center, _ = typical_spin_prior(
        gender="women",
        phase="serve",
        speed_kmh=160.0,
        profile="high_spin_serve",
    )
    assert slow_center > fast_center


def test_slice_prior_is_signed_backspin():
    center, sigma = typical_spin_prior(
        gender="men",
        phase="rally",
        speed_kmh=100.0,
        profile="slice",
    )
    assert center < 0
    assert sigma > 0


def test_bounce_on_wrong_player_half_is_penalized():
    bounce = {"x": np.array([5.0, 4.0, 0.033])}
    assert bounce_grammar_cost(
        bounce,
        start_side="near",
        end_side="far",
    ) >= 12.0


def test_branch_builder_keeps_discrete_whole_point_alternatives():
    spin = [
        SpinPrior("drive", 2000.0, 900.0, 0.1),
        SpinPrior("slice", -1500.0, 900.0, 0.2),
    ]
    no_bounce = {
        "anchor": None,
        "event_cost": 0.1,
        "grammar_cost": 0.0,
        "label": "none",
    }
    bounce = {
        "anchor": {"frame": 10.0, "x": np.array([5.0, 18.0, 0.033])},
        "event_cost": 0.2,
        "grammar_cost": 0.0,
        "label": "bounce",
    }
    branches = build_point_branches(
        {
            0: {"spin": spin, "bounce": [no_bounce, bounce]},
            1: {"spin": spin, "bounce": [no_bounce]},
        },
        beam_width=4,
    )
    assert len(branches) == 4
    assert len(
        {
            (
                branch["shots"][0]["profile"],
                branch["shots"][0]["bounce_label"],
                branch["shots"][1]["profile"],
            )
            for branch in branches
        }
    ) == 4


def test_branch_margin_abstains_when_global_scores_are_close():
    decisive, margin = branch_is_decisive(
        [{"score": {"total": 10.0}}, {"score": {"total": 14.0}}],
        minimum_margin=8.0,
    )
    assert not decisive
    assert margin == 4.0


def test_serve_options_separate_flat_slice_and_kick_axes():
    flat = shot_spin_options(
        gender="men",
        phase="serve",
        speed_kmh=160.0,
        fitted_signed_rpm=1000.0,
        fitted_sidespin_rpm=0.0,
        apex_m=3.0,
        duration_s=0.7,
    )
    assert flat[0].profile == "flat_serve"

    sliced = shot_spin_options(
        gender="men",
        phase="serve",
        speed_kmh=160.0,
        fitted_signed_rpm=500.0,
        fitted_sidespin_rpm=2200.0,
        apex_m=3.0,
        duration_s=0.7,
    )
    assert sliced[0].profile == "slice_serve_right"

    kicked = shot_spin_options(
        gender="men",
        phase="serve",
        speed_kmh=160.0,
        fitted_signed_rpm=2800.0,
        fitted_sidespin_rpm=1500.0,
        apex_m=3.0,
        duration_s=0.7,
    )
    assert kicked[0].profile == "kick_serve_right"
    assert kicked[0].rifle_sigma_rpm < kicked[0].sidespin_sigma_rpm


def test_global_branch_score_penalizes_a_ball_below_the_net():
    fit = SimpleNamespace(
        theta=np.array([5.0, 5.0, 1.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
        rms_px=1.0,
        n_obs=2,
        bounces=[],
        xs_obs=np.array(
            [
                [5.0, 10.0, 0.6],
                [5.0, 13.0, 0.6],
            ]
        ),
    )
    branch = {
        "prior_cost": 0.0,
        "shots": {
            0: {
                "spin_center_rpm": 0.0,
                "spin_sigma_rpm": 1000.0,
                "spin_center_components_rpm": [0.0, 0.0, 0.0],
                "spin_sigma_components_rpm": [1000.0, 1000.0, 1000.0],
                "start_side": "near",
                "end_side": "far",
            }
        },
    }

    score = refined_branch_score(
        fits={0: fit},
        branch=branch,
        junction_gaps_m=[],
    )

    assert score["grammar"] > 10.0


def test_branch_beam_preserves_distinct_serve_families():
    spins = [
        SpinPrior("flat_serve", 1200.0, 800.0, 0.0, beam_cost=0.0),
        SpinPrior(
            "slice_serve_left",
            500.0,
            800.0,
            0.0,
            sidespin_center_rpm=-2200.0,
            beam_cost=0.1,
        ),
        SpinPrior(
            "kick_serve_right",
            2800.0,
            900.0,
            0.0,
            sidespin_center_rpm=1500.0,
            beam_cost=0.2,
        ),
    ]
    event = {
        "anchor": None,
        "event_cost": 0.0,
        "grammar_cost": 0.0,
        "label": "none",
    }
    rally_spins = [
        SpinPrior("drive", 2500.0, 900.0, 0.0, beam_cost=0.0),
        SpinPrior("slice", -1600.0, 900.0, 0.0, beam_cost=0.1),
    ]

    branches = build_point_branches(
        {
            0: {"spin": spins, "bounce": [event]},
            1: {"spin": rally_spins, "bounce": [event]},
        },
        beam_width=3,
    )

    assert {branch["shots"][0]["profile"] for branch in branches} == {
        "flat_serve",
        "slice_serve_left",
        "kick_serve_right",
    }
