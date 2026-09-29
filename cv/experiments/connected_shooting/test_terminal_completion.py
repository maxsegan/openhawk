from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    measured_dynamics,
    model,
    terminal_completion,
    terminal_exposure_context,
)


def fixture(end_delta=0.0):
    theta = np.array([5.0, 3.0, 2.0, 2.0, 20.0, -2.0, 2.0, 0.0, 0.0])
    _, _, _, impacts = measured_dynamics.simulate(theta, 1.0, np.array([1.0, 36.0]), 25, "hard")
    impact = impacts[0]["frame"]
    frames = np.array([1.0, 2.0])
    scene = model.Scene(
        np.array([1.0, impact + end_delta]),
        (frames,),
        (np.repeat(np.eye(3, 4)[None], len(frames), axis=0),),
        (np.zeros((len(frames), 2)),),
        np.array([[2.0, 0.0, 0.0]]),
        25.0,
        "hard",
        "measured_240hz",
    )
    return scene, theta[:6], impacts


@pytest.mark.parametrize("end_delta", [-0.4, 0.4, 0.0])
def test_completion_uses_existing_impact_without_state_or_observation_repair(end_delta):
    scene, parameters, impacts = fixture(end_delta)
    before = model.chain(scene, parameters)
    original_parameters = parameters.copy()
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["status"] == "completed"
    assert receipt["end_frame"] == impacts[0]["frame"]
    assert abs(receipt["end_xyz"][2] - model.R_BALL) < 1e-9
    assert receipt["observations_discarded"] == 0
    assert receipt["frame_delta"] == pytest.approx(-end_delta)
    assert scene.contact_frames[-1] == impacts[0]["frame"] + end_delta
    np.testing.assert_array_equal(parameters, original_parameters)
    np.testing.assert_array_equal(
        before[0]["positions"], model.chain(scene, parameters)[0]["positions"]
    )
    completed = replace(scene, contact_frames=np.array([1.0, receipt["end_frame"]]))
    np.testing.assert_array_equal(
        before[0]["positions"], model.chain(completed, parameters)[0]["positions"]
    )


def test_completion_cannot_discard_even_a_heldout_native_observation():
    scene, parameters, impacts = fixture(1.0)
    last = float(np.ceil(impacts[0]["frame"]))
    assert impacts[0]["frame"] < last <= scene.contact_frames[-1]
    assert last > impacts[0]["frame"] + impacts[0]["dwell_seconds"] * scene.fps
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=last
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == "completion_would_discard_observations_or_flight"


def test_observation_inside_existing_ground_dwell_is_preserved_without_extending_it():
    scene, parameters, impacts = fixture()
    desired_impact = float(np.ceil(impacts[0]["frame"]) - 0.02)
    shift = desired_impact - impacts[0]["frame"]
    scene = replace(
        scene,
        contact_frames=np.array([1.0 + shift, desired_impact + 0.05]),
        observation_frames=(np.array([2.0, 3.0]),),
    )
    last = float(np.ceil(desired_impact))
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=last
    )
    assert receipt["status"] == "completed"
    assert receipt["end_frame"] == last
    assert receipt["impact_frame"] == pytest.approx(desired_impact)
    assert receipt["ending_phase"] == "impact_dwell"
    assert receipt["impact_frame"] < last < receipt["impact_dwell_end_frame"]
    assert abs(receipt["end_xyz"][2] - model.R_BALL) < 1e-9
    assert receipt["observations_discarded"] == 0


@pytest.mark.parametrize("delta", [-1.01, 1.01])
def test_completion_cannot_expand_its_declared_timing_window(delta):
    scene, parameters, _ = fixture(delta)
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] in {"required_impact_not_reached", "required_impact_outside_window"}


def test_first_bounce_cannot_be_reclassified_as_second_bounce_or_camera_exit():
    scene, parameters, _ = fixture()
    for kind in ["second_bounce", "out_of_view", "net_stop"]:
        receipt = terminal_completion.complete(
            scene, parameters, kind, uncertainty_frames=1, last_observation_frame=2
        )
        assert receipt["status"] == "held"


def test_second_bounce_completes_only_at_second_impact():
    scene, parameters, impacts = fixture()
    scene = replace(scene, contact_frames=np.array([1.0, impacts[1]["frame"] - 0.2]))
    receipt = terminal_completion.complete(
        scene, parameters, "second_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["status"] == "completed"
    assert receipt["end_frame"] == impacts[1]["frame"]
    assert receipt["required_impact_ordinal"] == 2


def test_ambiguous_nearby_impacts_are_not_selected_using_truth(monkeypatch):
    scene, parameters, impacts = fixture()
    ambiguous = [impacts[0], {**impacts[0], "frame": impacts[0]["frame"] + 0.2}]
    monkeypatch.setattr(model, "chain", lambda *args, **kwargs: [{"bounces": ambiguous}])
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == "ambiguous_impacts_in_window"


@pytest.mark.parametrize(
    "uncertainty,last", [(1.01, 2), (0, 2), (float("nan"), 2), (1, 1), (1, 2.5), (1, 100)]
)
def test_invalid_evidence_refuses(uncertainty, last):
    scene, parameters, _ = fixture()
    with pytest.raises(ValueError, match="bounded timing"):
        terminal_completion.complete(
            scene,
            parameters,
            "terminal_bounce",
            uncertainty_frames=uncertainty,
            last_observation_frame=last,
        )


def test_passive_context_lets_a_point_end_at_its_impact_without_discarding_the_front():
    scene, parameters, impacts = fixture(1.0)
    last = float(np.ceil(impacts[0]["frame"]))
    dwell_end = impacts[0]["frame"] + impacts[0]["dwell_seconds"] * scene.fps
    assert last > dwell_end
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=last,
        passive_context_frames=1.0,
        native_frames=np.array([2.0, last]),
    )
    assert receipt["status"] == "completed"
    # The point ends at the ground impact; the trailing front stays a passive
    # observation at its own timestamp and the fitted state never moves.
    assert receipt["end_frame"] == pytest.approx(impacts[0]["frame"])
    assert receipt["end_frame"] < last
    assert receipt["observations_discarded"] == 0
    assert receipt["parameters_changed"] is False
    passive = receipt["passive_post_ending_context"]
    assert passive["passive_exposure_frames"] == [last]
    assert passive["creates_flight"] is False
    assert abs(receipt["end_xyz"][2] - model.R_BALL) < 1e-9


def test_a_labeled_contact_inside_the_passive_span_holds_the_completion():
    scene, parameters, impacts = fixture(1.0)
    last = float(np.ceil(impacts[0]["frame"]))
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=last,
        passive_context_frames=1.0,
        native_frames=np.array([2.0, last]),
        live_contact_frames=[1.0, last],
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == "passive_post_ending_context:live_contact_inside_passive_span"


def test_a_passive_span_beyond_the_annotator_agreement_still_holds_the_completion():
    scene, parameters, impacts = fixture(1.0)
    last = float(np.ceil(impacts[0]["frame"]))
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=last,
        passive_context_frames=terminal_exposure_context.ANNOTATOR_AGREEMENT_FRAMES,
        native_frames=np.array([2.0, last]),
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == (
        "passive_post_ending_context:passive_span_beyond_annotator_agreement"
    )


def test_a_lob_leaving_the_frame_is_still_a_gap_with_the_passive_arm_on():
    # No ground impact is reached inside the declared window, so the point does
    # not end here and passive context cannot invent an ending for it.
    scene, parameters, _ = fixture(-1.01)
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=2,
        passive_context_frames=0.5,
        native_frames=np.array([1.0, 2.0]),
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] in {"required_impact_not_reached", "required_impact_outside_window"}


def test_the_passive_arm_is_off_by_default_and_reproduces_the_promoted_refusal():
    scene, parameters, impacts = fixture(1.0)
    last = float(np.ceil(impacts[0]["frame"]))
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=last
    )
    assert receipt["passive_context_frames"] == 0.0
    assert receipt["status"] == "held"
    assert receipt["reason"] == "completion_would_discard_observations_or_flight"


@pytest.mark.parametrize("passive", [1.01, -0.1, float("nan")])
def test_passive_context_cannot_exceed_one_native_frame(passive):
    scene, parameters, _ = fixture()
    with pytest.raises(ValueError, match="bounded timing"):
        terminal_completion.complete(
            scene,
            parameters,
            "terminal_bounce",
            uncertainty_frames=1,
            last_observation_frame=2,
            passive_context_frames=passive,
        )


def _dwell_exposure_scene(miss_frames):
    """Impact ``miss_frames`` before an integer exposure that opens the declared window."""
    scene, parameters, impacts = fixture()
    exposure = float(np.ceil(impacts[0]["frame"]))
    shift = (exposure - miss_frames) - impacts[0]["frame"]
    scene = replace(
        scene,
        # supplied ending one frame after the exposure: window [exposure, exposure + 2]
        contact_frames=np.array([1.0 + shift, exposure + 1.0]),
        observation_frames=(np.array([2.0, 3.0]),),
    )
    return scene, parameters, exposure


def test_window_is_judged_on_the_reported_dwell_exposure_not_the_dwell_start():
    # source02_short_0002_a1 e07: impact 169.9976, exposure 170.0, window [170, 172].
    scene, parameters, exposure = _dwell_exposure_scene(0.0024)
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=exposure
    )
    assert receipt["bounds_frames"] == [exposure, exposure + 2.0]
    assert receipt["impact_frame"] == pytest.approx(exposure - 0.0024)
    assert receipt["status"] == "completed"
    assert receipt["end_frame"] == exposure
    assert receipt["ending_phase"] == "impact_dwell"
    assert receipt["window_test"]["impact_instant_inside"] is False
    assert receipt["window_test"]["reported_ending_inside"] is True
    strict = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=exposure,
        window_tests_reported_ending=False,
    )
    assert strict["status"] == "held"
    assert strict["reason"] == "required_impact_outside_window"


def test_same_miss_without_an_exposure_in_the_dwell_is_still_held():
    # No native exposure sees the ball on the ground inside the window, so the
    # window is not widened: the rule is evidence-tied, not a tolerance constant.
    scene, parameters, exposure = _dwell_exposure_scene(0.0024)
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=exposure - 1.0,
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == "required_impact_outside_window"


def test_exposure_after_the_dwell_does_not_admit_an_early_impact():
    # Dwell is 0.1125 frames at 25 fps; an impact 0.2 frames early has left the
    # ground by the exposure, so neither epoch is inside the window.
    scene, parameters, exposure = _dwell_exposure_scene(0.2)
    receipt = terminal_completion.complete(
        scene,
        parameters,
        "terminal_bounce",
        uncertainty_frames=1,
        last_observation_frame=exposure,
        passive_context_frames=0.5,
        native_frames=np.array([2.0, exposure]),
    )
    assert receipt["status"] == "held"
    assert receipt["reason"] == "required_impact_outside_window"


def test_window_edge_carries_the_shared_float_tolerance_only():
    scene, parameters, _ = fixture(1.0 + 5e-7)
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["status"] == "completed"
    scene, parameters, _ = fixture(1.0 + 1e-4)
    receipt = terminal_completion.complete(
        scene, parameters, "terminal_bounce", uncertainty_frames=1, last_observation_frame=2
    )
    assert receipt["reason"] == "required_impact_outside_window"
