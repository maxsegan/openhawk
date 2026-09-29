"""Known 3D gravity serves with missing contact pictures and explicit depth hypotheses."""

from dataclasses import replace
import numpy as np
import pytest
from cv.experiments.connected_shooting import depth_conditioned_seed as seed, initialization, model
from cv.pipeline import s6_labeled_stage as stage


def fixture_scene():
    eye = np.array([5.0, -15.0, 13.0])
    target = np.array([5.0, 12.0, 0.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.vstack([right, np.cross(forward, right), forward])
    P = (
        np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1.0]])
        @ np.c_[rotation, -rotation @ eye]
    )
    frames = (np.arange(106.0, 130.0), np.arange(151.0, 170.0))
    fps = 60.0
    start = np.array([6.0, 0.0, 3.4])
    tb = 0.5
    velocity = np.array([-9.0, 35.0, (model.R_BALL - start[2] + 4.905 * tb**2) / tb])
    pixels = []
    for group in frames:
        t = (group - 100) / fps
        xyz = start + t[:, None] * velocity + 0.5 * t[:, None] ** 2 * np.array([0, 0, -9.81])
        uvw = np.c_[xyz, np.ones(len(xyz))] @ P.T
        pixels.append(uvw[:, :2] / uvw[:, 2:])
    scene = model.Scene(
        np.array([100.0, 140.0, 190.0]),
        frames,
        tuple(np.repeat(P[None], len(f), axis=0) for f in frames),
        tuple(pixels),
        np.zeros((2, 3)),
        fps,
        "clay",
    )
    initial = np.r_[6.0, 5.0, 2.0, -6.0, 25.0, -2.0, 2.0, -20.0, 4.0, np.zeros(6), 1.0, 1.0]
    return scene, initial, np.r_[start, velocity]


def test_missing_contact_pictures_recover_known_state_and_preserve_rest():
    scene, initial, truth = fixture_scene()
    before = initial.copy()
    out, receipt = seed.condition(scene, initial, ([130.0], []), 0.0)
    assert receipt["status"] == "conditioned"
    np.testing.assert_allclose(out[:6], truth, atol=1e-8)
    np.testing.assert_array_equal(out[6:], initial[6:])
    np.testing.assert_array_equal(initial, before)
    assert receipt["additional_optimizer_calls"] == 0
    assert receipt["initialization"]["flights"][0]["training_frames"][0] == 106.0


def test_depth_is_explicit_hypothesis_not_recovered_truth():
    scene, initial, _ = fixture_scene()
    out, receipt = seed.condition(scene, initial, ([130.0], []), 0.5)
    assert receipt["status"] == "conditioned" and out[1] == 0.5
    assert not np.array_equal(out[[0, 2, 3, 4, 5]], initial[[0, 2, 3, 4, 5]])


def test_labelled_net_before_ground_does_not_receive_free_flight_seed():
    scene, initial, _ = fixture_scene()
    scene = replace(
        scene, dynamics="measured_240hz", net_hit_frames=(np.array([120.0]), np.array([]))
    )
    out, r = seed.condition(scene, initial, ([130.0], []), 0.0)
    np.testing.assert_array_equal(out, initial)
    assert r["status"] == "not_applicable"


def test_sparse_input_keeps_original_seed_with_explicit_refusal():
    scene, initial, _ = fixture_scene()
    scene = replace(
        scene,
        observation_frames=(scene.observation_frames[0][:2], scene.observation_frames[1]),
        cameras=(scene.cameras[0][:2], scene.cameras[1]),
        pixels=(scene.pixels[0][:2], scene.pixels[1]),
    )
    out, r = seed.condition(scene, initial, ([130.0], []), 0.0)
    np.testing.assert_array_equal(out, initial)
    assert r["status"] == "refused_seed_retained"


def test_unknown_ground_does_not_invent_impact():
    scene, initial, _ = fixture_scene()
    out, r = seed.condition(scene, initial, ([], []), 0.0)
    np.testing.assert_array_equal(out, initial)
    assert r["status"] == "not_applicable"


@pytest.mark.parametrize("depth", [True, float("nan"), float("inf"), 45.0])
def test_bad_depth_rejected_by_initializer(depth):
    scene, _, _ = fixture_scene()
    with pytest.raises(ValueError, match="fixed first depth"):
        initialization.image_ballistic_seed(
            scene, [130.0, 180.0], ground_anchor=True, first_contact_y_m=depth
        )


def test_default_settings_unchanged_and_flag_explicit():
    assert "depth_conditioned_seed" not in stage.shared_settings({})
    assert stage.shared_settings({"depth_conditioned_seed": "on"})["depth_conditioned_seed"] == "on"
    with pytest.raises(ValueError, match="depth_conditioned_seed"):
        stage.shared_settings({"depth_conditioned_seed": "typo"})


def test_ground_beyond_first_flight_cannot_silently_unfix_depth():
    scene, initial, _ = fixture_scene()
    out, receipt = seed.condition(scene, initial, ([150.0], []), 0.0)
    np.testing.assert_array_equal(out, initial)
    assert receipt["status"] == "refused_seed_retained"
    assert "within the first flight" in receipt["reason"]


def test_missing_bounce_and_unanchored_requests_refuse_cleanly():
    scene, initial, _ = fixture_scene()
    out, receipt = seed.condition(scene, initial, (None, []), 0.0)
    np.testing.assert_array_equal(out, initial)
    assert receipt["status"] == "not_applicable"
    with pytest.raises(ValueError, match="fixed first depth"):
        initialization.image_ballistic_seed(scene, [130.0, None], first_contact_y_m=0.0)


def test_original_ground_target_and_observation_partition_preserved(monkeypatch):
    scene, initial, _ = fixture_scene()
    scene = replace(scene, dynamics="measured_240hz", ground_settling=True)
    original = initialization.image_ballistic_seed
    seen = []

    def capture(first, bounces, **options):
        seen.append((first, bounces, options))
        return original(first, bounces, **options)

    monkeypatch.setattr(initialization, "image_ballistic_seed", capture)
    out, receipt = seed.condition(
        scene, initial, ([130.0], []), 0.0, bounce_ground_targets=[[1.5, 17.5], None]
    )
    assert receipt["status"] == "conditioned"
    first, bounces, options = seen[0]
    assert first.observation_partition == scene.observation_partition
    assert first.dynamics == scene.dynamics and first.ground_settling
    assert first.pixels[0] is scene.pixels[0]
    assert options["consensus"] is True
    assert options["bounce_ground_targets"] == [[1.5, 17.5]]
