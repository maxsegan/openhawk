from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting.model import chain, control, fit


def test_shared_positions_are_exact_with_unequal_racket_velocities():
    scene, parameters = control()
    scene.validate()
    before = parameters.copy()
    flights = chain(scene, parameters)
    np.testing.assert_array_equal(flights[0]["end_xyz"], flights[1]["start_xyz"])
    assert not np.array_equal(flights[0]["velocities"][-1], flights[1]["velocities"][0])
    np.testing.assert_array_equal(parameters, before)


@pytest.mark.parametrize("times", [[1, 1, 26], [1, 13, float("nan")], [1, 27, 26]])
def test_invalid_boundary_clocks_fail(times):
    scene, _ = control()
    with pytest.raises(ValueError, match="ordered"):
        replace(scene, contact_frames=np.array(times)).validate()


def test_duplicate_or_out_of_flight_pictures_are_not_repaired():
    scene, _ = control()
    for frames in [np.array([1, 1]), np.array([0, 1]), np.array([1, 2.5])]:
        with pytest.raises(ValueError, match="unique"):
            replace(scene, observation_frames=(frames, scene.observation_frames[1])).validate()


def test_same_simulator_control_converges_without_seam_penalties():
    scene, truth = control()
    result = fit(scene, truth + np.array([0.2, -0.3, 0.1, 0.5, -0.4, 0.2, -0.3, 0.4, -0.2]))
    assert result["final_pixel_rms"] < 1e-4
    assert result["final_pixel_rms"] < result["initial_pixel_rms"]
    assert result["junction_gaps_m"] == [0.0]
    assert result["status"] == "synthetic_mechanism_only_not_accepted_trajectory"


def test_bounce_does_not_break_shared_contact_or_depend_on_observation_density():
    scene, _ = control()
    times = np.array([1.0, 26.25, 51.5])
    frames = (np.arange(1, 27, dtype=float), np.arange(27, 52, dtype=float))
    cameras = tuple(np.repeat(scene.cameras[0][:1], len(f), axis=0) for f in frames)
    scene = replace(
        scene,
        contact_frames=times,
        observation_frames=frames,
        cameras=cameras,
        pixels=tuple(np.zeros((len(f), 2)) for f in frames),
    )
    scene.validate()
    parameters = np.array([5, 3, 2, 2, 20, -2, -1, -17, 4], float)
    full = chain(scene, parameters)
    assert full[0]["bounces"]
    np.testing.assert_array_equal(full[0]["end_xyz"], full[1]["start_xyz"])
    sparse = replace(
        scene,
        observation_frames=tuple(f[::2] for f in frames),
        cameras=tuple(p[::2] for p in cameras),
        pixels=tuple(p[::2] for p in scene.pixels),
    )
    sparse.validate()
    sampled = chain(sparse, parameters)
    for a, b in zip(full, sampled, strict=True):
        np.testing.assert_array_equal(a["start_xyz"], b["start_xyz"])
        np.testing.assert_array_equal(a["end_xyz"], b["end_xyz"])
        np.testing.assert_array_equal(a["positions"][::2], b["positions"])


def test_point_rebound_parameters_preserve_fixed_control_and_require_explicit_layout():
    scene, truth = control()
    fixed = replace(scene, dynamics="measured_240hz")
    free = replace(fixed, rebound_mode="point_scales")
    a, b = chain(fixed, truth), chain(free, np.r_[truth, 1.0, 1.0])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x["positions"], y["positions"])
    with pytest.raises(ValueError, match="rebound parameters"):
        chain(free, truth)
    with pytest.raises(ValueError, match="rebound corrections"):
        replace(scene, rebound_mode="point_scales").validate()
    with pytest.raises(ValueError, match="interior seed"):
        fit(free, truth)


@pytest.mark.parametrize("width", [0, -0.02, float("nan"), float("inf")])
def test_rebound_prior_requires_finite_positive_scale(width):
    scene, truth = control()
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    with pytest.raises(ValueError, match="rebound prior"):
        fit(scene, np.r_[truth, 1, 1], rebound_prior_scale=width)


def test_rebound_prior_is_not_allowed_for_fixed_coefficients():
    scene, truth = control()
    with pytest.raises(ValueError, match="rebound prior"):
        fit(scene, truth, rebound_prior_scale=0.02)


def test_rebound_prior_is_quadratic_and_does_not_change_image_residuals(monkeypatch):
    from cv.experiments.connected_shooting import model

    scene, truth = control()
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    seed = np.r_[truth, 1.05, 0.97]
    original = model.least_squares
    captured = []

    def capture(function, initial, **kwargs):
        residual = function(initial)
        if callable(kwargs["loss"]):
            loss = kwargs["loss"](np.full(len(residual), 9.0))
            np.testing.assert_array_equal(loss[:, -2:], [[9, 9], [1, 1], [0, 0]])
            assert loss[0, 0] < 9  # image evidence remains robust
        captured.append(residual)
        return original(function, initial, **kwargs)

    monkeypatch.setattr(model, "least_squares", capture)
    baseline = fit(scene, seed, max_nfev=1)
    result = fit(scene, seed, max_nfev=1, rebound_prior_scale=0.02)
    assert baseline["rebound_prior_evidence"] is None
    np.testing.assert_array_equal(captured[0], captured[1][:-2])
    np.testing.assert_allclose(captured[1][-2:], [2.5, -1.5])
    assert result["rebound_prior_evidence"]["cost"] == pytest.approx(4.25)
    assert result["optimizer_evidence"]["cost"] - baseline["optimizer_evidence"][
        "cost"
    ] == pytest.approx(4.25)


def test_regime_retry_preserves_rebound_prior(monkeypatch):
    from cv.experiments.connected_shooting import regime_recovery

    scene, truth = control()
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    seed = np.r_[truth, np.zeros(6), 1.05, 0.97]

    def check_retry(scene, baseline, candidate):
        retried = candidate(baseline["parameters"], (0, 0, "grip"))
        assert retried["rebound_prior_evidence"] == baseline["rebound_prior_evidence"]
        return baseline

    monkeypatch.setattr(regime_recovery, "refine_fixed_branches", check_retry)
    result = fit(
        scene,
        seed,
        max_nfev=1,
        optimize_spin=True,
        recover_bounce_regimes=True,
        bounce_regime_strategy="fixed_branch",
        rebound_prior_scale=0.02,
    )
    assert result["rebound_prior_evidence"]["scale"] == 0.02
