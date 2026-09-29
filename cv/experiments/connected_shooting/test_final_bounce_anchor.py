"""Final-bounce anchor: picture-agreed epoch as a restart and a soft timing prior."""

from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    agent_whole_point_search as search,
    initialization,
    model,
    real_exposure_replay as exposure,
)
from cv.experiments.connected_shooting.model import R_BALL
from cv.pipeline import s6_labeled_stage as stage

FPS = 25.0
CAMERA = np.array([[1000, 0, 0, 0], [0, 0, 1000, 0], [0, 1, 0, 0]], float)


def single_flight_scene(frames: np.ndarray) -> model.Scene:
    return model.Scene(
        contact_frames=np.array([0.0, 14.0]),
        observation_frames=(frames,),
        cameras=(np.repeat(CAMERA[None], len(frames), axis=0),),
        pixels=(np.zeros((len(frames), 2)),),
        spin_parameters=np.zeros((1, 3)),
        fps=FPS,
        surface="grass",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
    )


def target(delta: float, *, source="wing_intersection", mode="two_sided_reversal", frame=10.0):
    return {
        "xyz_m": [4.0, 17.0, R_BALL],
        "construction_mode": mode,
        "impact_epoch_witness": {
            "frame": frame + delta,
            "supplied_frame": frame,
            "delta_from_supplied_frames": delta,
            "source": source,
        },
    }


def test_plan_anchors_a_two_sided_witness_within_one_frame():
    scene = single_flight_scene(np.arange(1.0, 14.0))
    plan = search.final_bounce_anchor_plan(scene, [np.array([10.0])], [[target(-0.6)]])
    assert plan["status"] == "applied"
    assert plan["seed"] == {"frame": pytest.approx(9.4), "xy_m": [4.0, 17.0]}
    assert plan["epoch_priors"] == [
        dict(flight_index=0, ordinal=0, frame=pytest.approx(9.4), sigma_frames=1.0)
    ]


@pytest.mark.parametrize(
    "row,reason",
    [
        (target(1.4), "witness_disagrees_with_supplied_bounce"),
        (target(0.2, source="supplied_epoch"), "witness_not_two_sided"),
        (target(0.2, mode="one_sided_incoming"), "witness_not_two_sided"),
        (target(float("nan")), "witness_epoch_unresolved"),
    ],
)
def test_plan_abstains_where_pictures_do_not_agree(row, reason):
    scene = single_flight_scene(np.arange(1.0, 14.0))
    plan = search.final_bounce_anchor_plan(scene, [np.array([10.0])], [[row]])
    assert plan["status"] == "abstained" and plan["seed"] is None
    assert plan["bounces"][0]["reason"] == reason


def test_plan_needs_a_final_bounce():
    scene = single_flight_scene(np.arange(1.0, 14.0))
    plan = search.final_bounce_anchor_plan(scene, [np.array([])], [[]])
    assert plan["status"] == "not_applicable"


def test_anchor_walk_restarts_the_final_flight_from_the_witness_epoch():
    start = np.array([2.0, 25.0, 1.0])
    anchor_xy = np.array([4.0, 17.0])
    launch = initialization.ground_epoch_launch(start, 9.4 / FPS, np.zeros(3), target_xy=anchor_xy)
    frames = np.arange(1.0, 14.0)
    scene = single_flight_scene(frames)
    truth = np.r_[start, launch, np.zeros(3), [1.0, 1.0]]
    axes = np.zeros((len(frames), 2))
    scene = replace(scene, pixels=(exposure.prediction(scene, truth, axes, None),))
    # A pinhole guess that heads the wrong way: the restart must not start from it.
    wrong = np.r_[start, [-9.0, 4.0, 6.0], np.zeros(3), [1.0, 1.0]]
    seed, receipt = initialization.anchor_connected_seed(
        scene,
        wrong,
        axes,
        [[10.0]],
        25.0,
        first_contact_xyz_m=start,
        contact_targets_xyz_m=[None],
        bounce_targets=[[(anchor_xy, 0.3)]],
        max_nfev=40,
        exposure_duration=None,
        terminal_ground_anchor={"frame": 9.4, "xy_m": anchor_xy.tolist()},
    )
    flight = receipt["flights"][0]
    assert flight["terminal_ground_anchor"]["frame"] == 9.4
    assert flight["training_pixel_rms"] < 1.0
    bounce = model.chain(scene, seed)[0]["bounces"][0]
    assert bounce["frame"] == pytest.approx(9.4, abs=0.5)


def test_epoch_prior_is_fixed_length_and_validated():
    scene = single_flight_scene(np.arange(1.0, 14.0))
    priors = exposure.validate_bounce_epoch_priors(
        scene, [np.array([10.0])], [dict(flight_index=0, ordinal=0, frame=9.4, sigma_frames=1.0)]
    )
    flights = [{"bounces": [{"frame": 11.4}]}]
    np.testing.assert_allclose(exposure.bounce_epoch_prior_residuals(flights, priors), [2.0])
    np.testing.assert_allclose(
        exposure.bounce_epoch_prior_residuals([{"bounces": []}], priors), [0.0]
    )
    with pytest.raises(ValueError, match="inside its flight"):
        exposure.validate_bounce_epoch_priors(
            scene, [np.array([10.0])], [dict(flight_index=0, ordinal=0, frame=15.0, sigma_frames=1)]
        )
    with pytest.raises(ValueError, match="name a supplied bounce"):
        exposure.validate_bounce_epoch_priors(
            scene, [np.array([10.0])], [dict(flight_index=0, ordinal=1, frame=9.0, sigma_frames=1)]
        )


def test_policy_switch_is_absent_by_default_and_emits_its_flag():
    assert "final_bounce_anchor" not in stage.shared_settings({})
    assert stage.shared_settings({"final_bounce_anchor": "on"})["final_bounce_anchor"] == "on"
    with pytest.raises(ValueError, match="final_bounce_anchor"):
        stage.shared_settings({"final_bounce_anchor": "yes"})
    assert "final_bounce_anchor" not in stage.PIPELINE_COMPONENT_POLICY


def test_picture_exit_partial_ending_has_no_terminal_bounce_slack():
    from cv.experiments.connected_shooting import terminal_feasibility as terminal

    scene = single_flight_scene(np.arange(1.0, 14.0))
    start = np.array([2.0, 25.0, 1.0])
    launch = initialization.ground_epoch_launch(start, 6.0 / FPS, np.zeros(3), target_xy=[4, 17])
    parameters = np.r_[start, launch, np.zeros(3), [1.0, 1.0]]
    exit_scene = replace(
        scene, supported_ending={"kind": "fov_exit", "frame": 14.0, "partial_flight": True}
    )
    np.testing.assert_allclose(
        terminal.terminal_slack(exit_scene, parameters, [np.array([6.0])], 13.0), [1, 1, 1]
    )
    # Without the partial declaration the bounce at frame 6 must be the flight end.
    ordinary = terminal.terminal_slack(scene, parameters, [np.array([6.0])], 13.0)
    assert ordinary.min() < 0


def test_exit_partial_switch_is_a_production_default():
    assert "exit_partial_endings" not in stage.shared_settings({})
    assert stage.shared_settings({"exit_partial_endings": "on"})["exit_partial_endings"] == "on"
    assert stage.PIPELINE_COMPONENT_POLICY["exit_partial_endings"] == "on"


def test_net_stop_tail_closes_three_frames_past_the_net():
    from cv.pipeline import s6_contact_components as components

    assert components._net_horizon(51.5, 100.0) == 52.5
    assert components._net_horizon(51.5, 100.0, components._NET_STOP_TAIL_FRAMES) == 54.5
    # Never past the supported window.
    assert components._net_horizon(51.5, 53.0, components._NET_STOP_TAIL_FRAMES) == 53.0 - 1e-3
    assert "net_stop_tail" not in stage.shared_settings({})
