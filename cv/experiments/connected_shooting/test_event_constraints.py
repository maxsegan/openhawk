from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import event_constraints, model


def test_runback_envelope_is_soft_beyond_professional_venue():
    inside = event_constraints.contact_xy_envelope_evidence([5.0, -7.0, 1.0])
    runback = event_constraints.contact_xy_envelope_evidence([5.0, -10.0, 1.0])
    wall = event_constraints.contact_xy_envelope_evidence([5.0, -12.01, 1.0])

    assert inside["inside_professional_runback"]
    assert inside["runback_soft_penalty"] == 0
    assert runback["inside_venue_envelope"]
    assert not runback["inside_professional_runback"]
    assert runback["professional_runback_overrun_xy_m"] == pytest.approx([0.0, 1.77])
    assert runback["runback_soft_penalty"] == pytest.approx(1.77**2)
    assert not wall["inside_venue_envelope"]


def test_event_inventory_and_uncertainty_fail_closed():
    scene, _ = model.control()
    groups = event_constraints.validate(scene, ([9], [21]), 1)
    assert [g.tolist() for g in groups] == [[9], [21]]
    for bad in [([9],), ([9, 9], [21]), ([0], [21]), ([14], [21]), ([float("nan")], [21])]:
        with pytest.raises(ValueError):
            event_constraints.validate(scene, bad, 1)
    for width in [-0.1, 2.1, float("nan")]:
        with pytest.raises(ValueError):
            event_constraints.validate(scene, groups, width)


def test_physical_residuals_are_not_robustified_with_image_outliers():
    loss = event_constraints.mixed_loss(2)
    z = np.array([0.0, 100.0, 100.0])
    rho = loss(z)
    assert rho.shape == (3, 3)
    assert rho[1, 1] < 0.1
    np.testing.assert_array_equal(rho[:, 2], [100, 1, 0])


def test_missing_bounces_supply_ground_gradient_without_inserting_impacts():
    scene, parameters = model.control()
    native_before = [p.copy() for p in scene.pixels]
    expected = event_constraints.validate(scene, ([9.25], [21.25]), 1)
    flights, residual, evidence = event_constraints.evaluate(scene, parameters, expected, 1)
    assert residual.shape == (4,)
    assert residual[0] > 0 and residual[2] > 0
    assert residual[1] == residual[3] == 0
    for a, b in zip(flights, model.chain(scene, parameters), strict=True):
        np.testing.assert_array_equal(a["positions"], b["positions"])
        assert not a["bounces"]
    for a, b in zip(scene.pixels, native_before, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not any(r["count_agrees"] for r in evidence)


def bounce_scene(*, collision_free=False):
    scene, _ = model.control()
    frames = np.arange(1.0, 27.0)
    scene = replace(
        scene,
        contact_frames=np.array([1.0, 26.0]),
        observation_frames=(frames,),
        cameras=(np.repeat(scene.cameras[0][:1], len(frames), axis=0),),
        pixels=(np.zeros((len(frames), 2)),),
        spin_parameters=np.zeros((1, 3)),
        dynamics="measured_240hz",
    )
    # The original fixture traverses the net. Ground-only fitting controls can
    # place the same dynamics outside the post instead of omitting that penalty.
    parameters = np.array([-4 if collision_free else 5, 3, 2, 2, 20, -2], float)
    projection = model.image_residual(scene, parameters).reshape(-1, 2)
    return replace(scene, pixels=(projection,)), parameters


def test_time_window_and_extra_impact_penalties_do_not_snap_the_path():
    scene, parameters = bounce_scene()
    first = model.chain(scene, parameters)[0]["bounces"][0]["frame"]
    expected = (np.array([first + 0.5]),)
    _, residual, evidence = event_constraints.evaluate(scene, parameters, expected, 1)
    np.testing.assert_array_equal(residual[:2], [0, 0])
    assert residual.shape == (3,)
    assert residual[-1] > 0  # Independent net violation in this bounce fixture.
    assert evidence[0]["modeled_bounce_frames"][0] == first
    _, residual, _ = event_constraints.evaluate(scene, parameters, (np.array([first + 2]),), 1)
    assert residual[0] == pytest.approx(-4)
    _, residual, _ = event_constraints.evaluate(scene, parameters, (np.array([]),), 1)
    assert residual[0] == pytest.approx((26 - first) / 0.25)


def test_exact_physical_fit_with_event_evidence_preserves_trajectory_and_observations():
    scene, parameters = bounce_scene(collision_free=True)
    first = model.chain(scene, parameters)[0]["bounces"][0]["frame"]
    result = model.fit(scene, parameters, bounce_frames=(np.array([first]),), max_nfev=10)
    assert result["final_pixel_rms"] < 1e-8
    assert result["bounce_window_evidence"]["final"][0]["count_agrees"]
    assert result["status"] == "synthetic_mechanism_only_not_accepted_trajectory"


def test_zero_uncertainty_uses_subframe_evidence_without_changing_the_path():
    scene, parameters = bounce_scene()
    before = model.chain(scene, parameters)[0]
    first = before["bounces"][0]["frame"]
    expected = event_constraints.validate(scene, ([first + 0.125],), 0)
    flights, residual, evidence = event_constraints.evaluate(scene, parameters, expected, 0)
    np.testing.assert_array_equal(residual[:2], [-0.5, 0])
    assert residual.shape == (3,)
    assert residual[-1] > 0
    np.testing.assert_array_equal(flights[0]["positions"], before["positions"])
    assert evidence[0]["modeled_bounce_frames"] == [first]


def test_supplied_timing_actually_guides_fit_without_retiming_native_observations():
    scene, parameters = bounce_scene(collision_free=True)
    first = model.chain(scene, parameters)[0]["bounces"][0]["frame"]
    frames_before = scene.observation_frames[0].copy()
    # Deliberately conflicting evidence checks objective plumbing, not accuracy.
    expected = (np.array([first + 2]),)
    result = model.fit(scene, parameters, bounce_frames=expected, max_nfev=40)
    fitted_time = result["flights"][0]["bounces"][0]["frame"]
    assert 0.01 < fitted_time - first < 2  # evidence changes the optimum toward its window
    np.testing.assert_array_equal(scene.observation_frames[0], frames_before)
    assert result["final_pixel_rms"] > 0  # a real tradeoff, not modified observations


def test_terminal_rebound_requires_modeled_dwell_before_first_real_picture():
    scene, parameters = bounce_scene()
    expected = (np.array([12.0]),)
    _, residual, evidence = event_constraints.evaluate(
        scene,
        parameters,
        expected,
        2.0,
        terminal_rebound_frames=np.array([13.0, 14.0]),
    )

    receipt = evidence[-1]["terminal_rebound"]
    assert residual[-2] > 0  # Rebound term precedes the collision-free net term.
    assert receipt["first_postbounce_frame"] == 13
    assert receipt["modeled_dwell_end_frame"] > 13
    assert receipt["spin_carried_through_impact"]
    assert receipt["pictures_added"] == 0


def test_terminal_rebound_refuses_non_scene_or_prebounce_rows():
    scene, parameters = bounce_scene()
    for frames in ([12.0], [13.5], []):
        with pytest.raises(ValueError, match="terminal rebound"):
            event_constraints.evaluate(
                scene,
                parameters,
                (np.array([12.0]),),
                2.0,
                terminal_rebound_frames=np.asarray(frames),
            )


def test_terminal_rebound_slack_uses_rebound_order_not_old_scene_endpoint():
    scene, parameters = bounce_scene()
    first = model.chain(scene, parameters)[0]["bounces"][0]["frame"]
    slack = event_constraints.terminal_rebound_slack(
        scene,
        parameters,
        (np.array([first]),),
        np.array([14.0, 15.0]),
    )

    assert slack[0] > 0  # bounce+dwell completes before the first rebound image
    assert slack[1] == pytest.approx(2.0)
    assert slack[2] == pytest.approx(2.0)
