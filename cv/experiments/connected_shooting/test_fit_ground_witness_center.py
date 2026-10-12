"""Known geometry, noisy physical-bound behavior and fitting/acceptance isolation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import fit_ground_witness_center as witness
from cv.experiments.connected_shooting import observation_operator, observation_partition
from cv.experiments.connected_shooting.real_exposure_replay import anchor_plan


def synthetic(*, translated=False, noise=0.0, seed=0):
    frames = np.r_[np.arange(-8, 0), np.arange(1, 9)]
    center = np.array([5.5, -18.0, 12.0])
    direction = np.array([5.5, 12.0, 0.0]) - center
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(direction, right), direction])
    intrinsic = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    matrices = []
    for frame in frames:
        m = (
            np.array([[1 + 0.002 * frame, 0, frame], [0, 1 + 0.002 * frame, 0], [0, 0, 1]])
            @ intrinsic
            @ rotation
        )
        c = center + (np.array([0.02 * frame, 0, 0]) if translated else 0)
        matrices.append(np.c_[m, -m @ c])
    cameras = np.array(matrices)
    truth = np.array([5.5, 18.5, witness.POLICY["ball_radius_m"]])
    dt = frames / 25.0
    velocity = np.where((dt < 0)[:, None], [3.0, 15.0, -6.0], [2.0, 10.0, 4.0])
    xyz = truth + dt[:, None] * velocity
    xyz[:, 2] -= 4.905 * dt**2
    h = np.einsum("nij,nj->ni", cameras, np.c_[xyz, np.ones(len(frames))])
    pixels = h[:, :2] / h[:, 2:]
    pixels += np.random.default_rng(seed).normal(0, noise, pixels.shape)
    return frames, pixels, np.full(len(frames), max(2.0, noise)), cameras, truth


@pytest.mark.parametrize("translated", [False, True])
def test_native_camera_known_ground_and_homogeneous_scale(translated):
    frames, pixels, sigma, cameras, truth = synthetic(translated=translated)
    original = [a.copy() for a in (frames, pixels, sigma, cameras)]
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(-0.5, 0.5), fps=25)
    assert result["status"] == "qualified", result.get("reason")
    np.testing.assert_allclose(result["target"]["xyz_m"], truth, atol=1e-7)
    assert result["best"]["epoch"] == pytest.approx(0, abs=1e-8)
    json.dumps(result, allow_nan=False)
    scaled = witness.estimate(
        frames,
        pixels,
        sigma,
        cameras * np.linspace(-4, -1, len(frames))[:, None, None],
        interval=(-0.5, 0.5),
        fps=25,
    )
    np.testing.assert_allclose(scaled["target"]["xyz_m"], truth, atol=1e-7)
    for before, after in zip(original, (frames, pixels, sigma, cameras), strict=True):
        np.testing.assert_array_equal(before, after)
    assert result["uncertainty"]["curvature_status"] == (
        "translated_camera" if translated else "qualified_model_discrepancy_only"
    )


def test_exact_source_epoch_has_no_profiled_timing_nuisance():
    frames, pixels, sigma, cameras, truth = synthetic()
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(0, 0), fps=25)
    assert result["status"] == "qualified"
    np.testing.assert_allclose(result["target"]["xyz_m"], truth, atol=1e-7)
    assert result["interval_is_exact"] and result["epoch_steps"] == 1
    assert result["epoch_at_interval_boundary"] is False
    assert result["main_physical_flag"] is None
    assert result["uncertainty"]["epoch_profile_floor_m2"] == 0
    assert result["uncertainty"]["residual_degrees_of_freedom"] == 8
    assert result["uncertainty"]["curvature_status"] == "qualified_model_discrepancy_only"


@pytest.mark.parametrize("before,after", [(3, 4), (4, 4), (4, 5), (5, 6), (6, 6)])
def test_auxiliary_availability_is_original_support_not_solver_choice(before, after):
    frames, pixels, sigma, cameras, truth = synthetic()
    take = ((frames < 0) & (frames >= -before)) | ((frames > 0) & (frames <= after))
    result = witness.estimate(
        frames[take], pixels[take], sigma[take], cameras[take], interval=(0, 0), fps=25
    )
    if min(before, after) < 4:
        assert result["reason"] == "insufficient_source_support"
        assert "target" not in result
        return
    assert result["status"] == "qualified", result.get("reason")
    np.testing.assert_allclose(result["target"]["xyz_m"], truth, atol=1e-7)
    three, six = result["auxiliaries"]
    assert three["available"] and three["status"] == "available"
    assert six["available"] == (min(before, after) >= 6)
    assert result["uncertainty"]["unavailable_model_windows"] == ([] if six["available"] else [6])


@pytest.mark.parametrize("failure", ["rank_deficient", "unconverged"])
def test_supported_but_failed_auxiliary_is_not_silently_omitted(monkeypatch, failure):
    frames, pixels, sigma, cameras, _ = synthetic()
    original = witness._profile

    def profile(*args):
        if args[-1] == 6 and failure == "rank_deficient":
            raise ValueError("rank_deficient_or_ill_conditioned")
        result = original(*args)
        if args[-1] == 6:
            result["best"]["irls_relative_change"] = 0.1
            result["best"]["active_ground_bounds"] = [7]
        return result

    monkeypatch.setattr(witness, "_profile", profile)
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(0, 0), fps=25)
    assert result["status"] == "unqualified"
    assert result["auxiliaries"][-1]["available"]
    assert result["auxiliaries"][-1]["status"] == "numerical_failure"
    assert "target" not in result


def test_noisy_vertical_bounds_are_flags_not_horizontal_rejection():
    results = []
    numerical_abstentions = []
    for seed in range(8):
        frames, pixels, sigma, cameras, truth = synthetic(noise=2, seed=seed)
        result = witness.estimate(frames, pixels, sigma, cameras, interval=(-0.5, 0.5), fps=25)
        if result["status"] != "qualified":
            assert result["reason"] in (
                "irls_unconverged",
                "unqualified_model_window:irls_unconverged",
            )
            numerical_abstentions.append(seed)
            continue
        error = np.array(result["target"]["xyz_m"][:2]) - truth[:2]
        covariance = np.array(result["uncertainty"]["total_covariance_m2"])
        results.append((result, np.linalg.norm(error)))
        assert error @ np.linalg.solve(covariance, error) < 5.9915
        assert result["uncertainty_kind"].endswith("not_calibrated_confidence")
    assert any(r["best"]["active_ground_bounds"] for r, _ in results)
    assert len(results) + len(numerical_abstentions) == 8
    assert len(results) >= 4
    assert np.sqrt(np.mean([error**2 for _, error in results])) < 0.5


def test_rank_failure_abstains_instead_of_using_constrained_rank():
    frames, pixels, sigma, cameras, _ = synthetic()
    cameras[:, 0] = cameras[:, 2] * 960
    cameras[:, 1] = cameras[:, 2] * 540
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(-0.5, 0.5), fps=25)
    assert result["status"] == "unqualified"
    assert result["reason"] == "rank_deficient_or_ill_conditioned"
    assert "target" not in result


@pytest.mark.parametrize("active,boundary", [(True, False), (False, True), (True, True)])
def test_physical_flags_cannot_mask_numerical_nonconvergence(monkeypatch, active, boundary):
    frames, pixels, sigma, cameras, _ = synthetic()
    best = dict(
        epoch=-0.5 if boundary else 0.0,
        active_ground_bounds=[7] if active else [],
        minimum_sample_height_m=0.0325,
        irls_relative_change=0.1,
    )
    monkeypatch.setattr(
        witness, "_profile", lambda *a, **k: dict(best=best, profile=[best], frames=frames.tolist())
    )
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(-0.5, 0.5), fps=25)
    assert result["status"] == "unqualified"
    assert result["reason"] == "irls_unconverged"
    assert result["main_physical_flag"] == (
        "active_ground_bound" if active else "epoch_at_interval_boundary"
    )
    assert "target" not in result


def test_incoming_ground_constraints_intersect_at_farthest_sample():
    dt = np.array([-0.2, -0.1, 0.1, 0.2])
    incoming = dt < 0
    a = np.eye(8)
    unconstrained = np.array([5, 18, 2, 10, -0.7, 2, 10, 4.0])
    z, active, _ = witness._solve(a, unconstrained, np.ones(8), dt, incoming)
    assert active == (4,)
    assert z[4] == pytest.approx(-0.981)
    assert np.min(z[4] * dt[incoming] - 4.905 * dt[incoming] ** 2) >= -1e-10


def source_fixture():
    frames, pixels, sigma, cameras, _ = synthetic()
    scene = SimpleNamespace(
        contact_frames=[-12, 12],
        observation_frames=[frames],
        fps=25,
        observation_partition=observation_partition.DEFAULT,
        bounce_witness_observation_policy=observation_partition.LEGACY_WITNESS,
    )
    event = dict(event_type="bounce", frame=0, frame_interval=[-0.5, 0.5])
    return (
        scene,
        event,
        dict(zip(frames, cameras)),
        dict(zip(frames, pixels)),
        dict(zip(frames, sigma)),
    )


@pytest.mark.parametrize("unsupported", ["front", "distortion", "physical_boundary"])
def test_unsupported_inputs_preserve_original_seed_and_acceptance(unsupported):
    scene, event, cameras, labels, radii = source_fixture()
    operator = observation_operator.declaration(
        "leading_front" if unsupported == "front" else "nominal_center"
    )
    boundary = [event]
    if unsupported == "physical_boundary":
        boundary.append(dict(event_type="net_hit", frame=2, frame_interval=[1.5, 2.5]))
    receipt = witness.source_groups(
        scene,
        [event],
        boundary,
        cameras,
        labels,
        radii,
        operator=operator,
        camera_distortion={f: np.array([0.1, 0, 0, 0]) for f in cameras}
        if unsupported == "distortion"
        else None,
    )
    original = [[dict(xyz_m=[1, 2, 0.0325], graded_circle_radius_m=0.4, nested={"a": [1]})]]
    before = deepcopy(original)
    seed = [np.array([1.0, 2.0])]
    assert receipt["qualified_count"] == 0
    assert receipt["preserved_original_count"] == 1
    assert witness.fitting_anchor_targets(original, receipt) == original == before
    np.testing.assert_array_equal(witness.fitting_seed_targets(seed, receipt), seed)


def test_fitting_copy_uses_new_weight_without_changing_acceptance():
    scene, event, cameras, labels, radii = source_fixture()
    receipt = witness.source_groups(
        scene,
        [event],
        [event],
        cameras,
        labels,
        radii,
        operator=observation_operator.declaration("nominal_center"),
    )
    original = [[dict(xyz_m=[1, 2, 0.0325], graded_circle_radius_m=0.4, nested={"a": [1]})]]
    before = deepcopy(original)
    fitting = witness.fitting_anchor_targets(original, receipt)
    assert receipt["qualified_count"] == 1
    assert original == before
    assert fitting is not original and fitting[0] is not original[0]
    assert "graded_circle_radius_m" not in fitting[0][0]
    assert fitting[0][0]["xyz_m"] != original[0][0]["xyz_m"]
    plan = anchor_plan(fitting, [None])
    assert plan["bounce"][0][0][1] == max(0.2, 2 * fitting[0][0]["uncertainty_sigma_m"])
    assert witness.fitting_seed_targets(None, receipt) is None


def test_missing_uncertainty_uses_declared_noise_assumption_with_receipt():
    scene, event, cameras, labels, radii = source_fixture()
    radii.pop(-2)
    receipt = witness.source_groups(
        scene,
        [event],
        [event],
        cameras,
        labels,
        radii,
        operator=observation_operator.declaration("nominal_center"),
    )
    row = receipt["groups"][0][0]
    assert row["status"] == "qualified"
    assert row["policy_default_pixel_sigma_input_frames"] == [-2]
    assert row["policy_default_pixel_sigma_input_count"] == 1


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan")])
def test_invalid_declared_uncertainty_does_not_become_floor(invalid):
    frames, pixels, sigma, cameras, _ = synthetic()
    sigma[3] = invalid
    result = witness.estimate(frames, pixels, sigma, cameras, interval=(0, 0), fps=25)
    assert result["status"] == "unqualified"
    assert "target" not in result
