from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    camera_geometry,
    initialization,
    measured_dynamics,
    model,
    postbounce_initialization as post,
)


def fixture(radial=False):
    base, _ = model.control()
    theta = np.array([3.0, 2.0, 2.0, 3.0, 22.0, -4.0, 2.0, 0.0, 0.0])
    frames = np.arange(1.0, 25.0)
    x, _, _, impacts = measured_dynamics.simulate(theta, 1.0, frames, 25.0, "hard")
    bounce = impacts[0]["frame"]
    cameras = np.repeat(base.cameras[0][:1], len(frames), axis=0)
    distortion = np.tile([2e-8, 960.0, 540.0], (len(frames), 1)) if radial else None
    pixels = camera_geometry.project(cameras, x, distortion)
    scene = replace(
        base,
        contact_frames=np.array([1.0, 24.0]),
        observation_frames=(frames,),
        cameras=(cameras,),
        pixels=(pixels,),
        spin_parameters=theta[6:][None],
        dynamics="measured_240hz",
        camera_distortion=(distortion,) if radial else None,
    )
    return scene, theta, bounce


def subset(scene, mask):
    return replace(
        scene,
        observation_frames=(scene.observation_frames[0][mask],),
        cameras=(scene.cameras[0][mask],),
        pixels=(scene.pixels[0][mask],),
        camera_distortion=None
        if scene.camera_distortion is None
        else (scene.camera_distortion[0][mask],),
    )


@pytest.mark.parametrize("radial", [False, True])
def test_outgoing_images_seed_unobserved_incoming_flight_without_xyz_input(radial):
    full, truth, bounce = fixture(radial)
    frames = full.observation_frames[0]
    scene = subset(full, (frames > bounce) & (frames.astype(int) % 3 != 0))
    before = scene.pixels[0].copy()
    with pytest.raises(ValueError, match="pre-bounce training pictures"):
        initialization.physics_backward_seed(scene, [bounce])
    seed, evidence = post.seed_flight(scene, bounce)
    assert not evidence["uses_truth_xyz_or_velocity"]
    assert not evidence["heldout_images_used"]
    assert not evidence["returned_trajectory"]
    np.testing.assert_array_equal(scene.pixels[0], before)
    # Truth is used only after the initializer has returned its seed.
    fitted = model.chain(full, seed)[0]["positions"]
    target = measured_dynamics.simulate(truth, 1.0, frames, 25.0, "hard")[0]
    assert np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))) < 0.1
    assert evidence["training_pixel_rms"] < 0.1


def test_sufficient_pre_bounce_evidence_calls_original_unchanged(monkeypatch):
    scene, _, bounce = fixture()
    expected = np.arange(9.0)

    def original(*args, **kwargs):
        assert args[0] is scene and args[1] == [bounce]
        return expected, {"old_path": True}

    monkeypatch.setattr(initialization, "physics_backward_seed", original)
    result, record = post.seed(scene, [bounce])
    assert result is expected and record == {"old_path": True}


def test_missing_outgoing_images_and_invalid_bounce_remain_abstentions():
    scene, _, bounce = fixture()
    short = subset(scene, np.arange(24) > 20)
    with pytest.raises(ValueError, match="four outgoing"):
        post.seed_flight(short, bounce)
    for invalid in [float("nan"), 1.0, 24.0, 30.0]:
        with pytest.raises(ValueError, match="interior bounce"):
            post.seed_flight(scene, invalid)


def test_knot_ground_dwell_and_original_rebound_law():
    scene, _, bounce = fixture()
    state = np.array([4.0, 8.0, 3.0, 22.0, -6.0, 2.0])
    end_dwell = bounce + post.bounce_reference.DWELL_SECONDS * scene.fps
    frames = np.array([bounce, (bounce + end_dwell) / 2, end_dwell, end_dwell + 1])
    xyz = post.knot_positions(state, frames, bounce, scene)
    np.testing.assert_array_equal(xyz[:3], np.tile([4.0, 8.0, model.R_BALL], (3, 1)))
    assert xyz[-1, 2] > model.R_BALL


def test_feasibility_recovery_is_opt_in_and_does_not_change_an_already_valid_seed():
    scene, _, bounce = fixture()
    scene = subset(scene, scene.observation_frames[0] > bounce)
    original, receipt = post.seed_flight(scene, bounce)
    recovered, new = post.seed_flight(scene, bounce, recover_launch_feasibility=True)
    np.testing.assert_array_equal(recovered, original)
    assert new.pop("launch_feasibility_recovery") is None
    assert new == receipt


def test_bad_released_spin_can_recover_a_feasible_launch_without_clipping(monkeypatch):
    scene, _, bounce = fixture()
    scene = subset(scene, scene.observation_frames[0] > bounce)
    original = post.least_squares
    calls = 0

    def injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls % 3 == 0:
            result.x[5] = 6.0  # Backward propagation exceeds the launch spin bound.
        return result

    monkeypatch.setattr(post, "least_squares", injected)
    with pytest.raises(ValueError, match="outside fitting bounds"):
        post.seed_flight(scene, bounce)
    state, receipt = post.seed_flight(scene, bounce, recover_launch_feasibility=True)
    recovery = receipt["launch_feasibility_recovery"]
    assert recovery is not None and not recovery["coordinates_clipped"]
    assert np.all(state > recovery["bounds_lo"]) and np.all(state < recovery["bounds_hi"])
    assert any(r["feasible"] for r in recovery["starts"])


def test_finite_bootstrap_uses_only_fixed_hypotheses_and_image_cost():
    guess = np.array([3.0, 7.0, 60.0, 0.0, -30.0, 2.0])
    before = guess.copy()

    def residual(state):
        if np.linalg.norm(state[2:5]) > 40:
            return np.full(8, 1e6)
        return np.array([state[2] - 24, state[4] + 12, state[5]])

    state, receipt = post.finite_bootstrap(guess, residual)
    np.testing.assert_array_equal(guess, before)
    np.testing.assert_allclose(state, [3, 7, 24, 0, -12, 0])
    assert len(receipt["candidates"]) == 10
    assert receipt["observations_discarded"] == 0
    assert not receipt["candidates"][0]["finite"]


def test_finite_bootstrap_exhaustion_stays_a_failure():
    with pytest.raises(ValueError, match="no finite scaled"):
        post.finite_bootstrap(
            np.array([3.0, 7.0, 60.0, 0.0, -30.0, 2.0]), lambda _: np.full(8, 1e6)
        )


def test_an_absent_bounce_is_allowed_and_a_nonfinite_one_is_not():
    """A volley flight has no post-bounce arc; it must not refuse the whole point."""
    scene, _, _ = fixture()
    with pytest.raises(ValueError, match="one finite or absent supplied bounce"):
        post.seed(scene, [float("nan")])
    # A bounceless flight simply keeps the ordinary image-physics seed, which
    # this single-flight fixture has enough pictures for.
    state, record = post.seed(scene, [None])
    assert record["method"] == "pre_bounce_backward_drag_magnus_ground_knot"
    assert np.isfinite(state).all()
