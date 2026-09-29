from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cv.experiments.connected_shooting import model, oracle_benchmark as audit


def test_noise_default_exactly_preserves_historical_draw_and_explicit_replicates():
    import hashlib

    scene, _ = model.control()
    before = [p.copy() for p in scene.pixels]
    point = "control__pt0001"
    rng = np.random.default_rng(int(hashlib.sha256(point.encode()).hexdigest()[:8], 16))
    expected = [p + rng.normal(0, 2, p.shape) for p in scene.pixels]
    original = audit.noisy_pixels(scene, point)
    replicate = audit.noisy_pixels(scene, point, 101)
    # Other calls cannot advance a replicate's point-local stream.
    audit.noisy_pixels(scene, "another point", 101)
    repeat = audit.noisy_pixels(scene, point, 101)
    other = audit.noisy_pixels(scene, point, 102)
    for index, p in enumerate(scene.pixels):
        np.testing.assert_array_equal(p, before[index])
        np.testing.assert_array_equal(original.pixels[index], expected[index])
        np.testing.assert_array_equal(replicate.pixels[index], repeat.pixels[index])
        assert not np.array_equal(replicate.pixels[index], original.pixels[index])
        assert not np.array_equal(replicate.pixels[index], other.pixels[index])
        np.testing.assert_array_equal(
            replicate.observation_frames[index], scene.observation_frames[index]
        )
        np.testing.assert_array_equal(replicate.cameras[index], scene.cameras[index])


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "1"])
def test_invalid_noise_seed_rejected(seed):
    scene, _ = model.control()
    with pytest.raises(ValueError, match="noise seed"):
        audit.noisy_pixels(scene, "point", seed)


@pytest.mark.parametrize(
    "seed,arm", [("-1", "pixels_noise2"), (str(2**32), "pixels_noise2"), ("1", "pixels")]
)
def test_cli_invalid_noise_configuration_cannot_create_outputs(tmp_path, monkeypatch, seed, arm):
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_benchmark",
            "--truth",
            "missing.json",
            "--cameras-root",
            "missing",
            "--output",
            str(output),
            "--arms",
            arm,
            "--noise-seed",
            seed,
        ],
    )
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2
    assert not output.exists()


def fixture_point(tmp_path):
    scene, parameters = model.control()
    queries = (np.arange(1, 13.01, 0.25), np.arange(13, 26.01, 0.25))
    dense = model.chain(scene, parameters, query_frames=queries)
    point = {
        "point": "control__pt0001",
        "match_id": "control",
        "clip": "pt0001",
        "fps": 25.0,
        "surface": "hard",
        "frames": list(range(1, 27)),
        "flights": [
            {"flight_index": 0, "start_frame": 1.0, "end_frame": 13.0, "terminal": False},
            {
                "flight_index": 1,
                "start_frame": 13.0,
                "end_frame": 26.0,
                "terminal": True,
                "termination_frame": 26.0,
            },
        ],
        "trajectory_truth": {
            "schema": "simulation_trajectory_samples_v1",
            "frames": np.r_[queries[0], queries[1][1:]].tolist(),
            "positions": np.r_[dense[0]["positions"], dense[1]["positions"][1:]].tolist(),
        },
    }
    camera_path = tmp_path / "camera.npz"
    np.savez(
        camera_path,
        clips=np.array(["pt0001"] * 26),
        frames=np.arange(1, 27),
        P=np.repeat(scene.cameras[0][:1], 26, axis=0),
    )
    return point, camera_path, parameters


@pytest.mark.parametrize("warm", [False, True, "direct"])
@pytest.mark.parametrize("noise_seed", [None, 101])
def test_cli_binds_truth_adapter_and_spatial_contract_in_fit_and_audit(
    tmp_path, monkeypatch, warm, noise_seed
):
    from cv.experiments.connected_shooting import physical_audit
    from cv.pipeline.provenance import file_record

    point, camera_path, _ = fixture_point(tmp_path)
    point["termination_kind"] = "second_bounce"
    camera_dir = tmp_path / "control"
    camera_dir.mkdir()
    camera_path.rename(camera_dir / "camera_P_per_frame_v1.npz")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"points": [point]}))
    output = tmp_path / "fit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_benchmark",
            "--truth",
            str(truth),
            "--cameras-root",
            str(tmp_path),
            "--output",
            str(output),
            "--arms",
            "xyz" if noise_seed is None else "pixels_noise2",
            "--max-nfev",
            "1",
            "--jobs",
            "1",
            *(
                [
                    "--dynamics",
                    "measured_240hz",
                    "--rebound-mode",
                    "point_scales",
                    "--warm-start-rebound",
                    "--rebound-prior-scale",
                    "0.02",
                ]
                if warm
                else []
            ),
            *(["--retain-direct-rebound"] if warm == "direct" else []),
            *(["--noise-seed", str(noise_seed)] if noise_seed is not None else []),
        ],
    )
    audit.main()
    audit_output = tmp_path / "physical.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "physical_audit",
            "--report",
            str(output / "report.json"),
            "--truth",
            str(truth),
            "--cameras-root",
            str(tmp_path),
            "--output",
            str(audit_output),
        ],
    )
    physical_audit.main()
    for path in (output / "report.json", audit_output):
        report = json.loads(path.read_text())
        bindings = {b["resolved_path"]: b["record"] for b in report["input_bindings"]}
        for module in (audit.s6_point_bench, audit.owner_spatial_audit):
            source = Path(module.__file__).resolve()
            assert bindings[str(source)] == file_record(source)
        assert len(report["results"]) == 1
        assert report["results"][0]["status"] == "measured"
    report = json.loads((output / "report.json").read_text())
    row = report["results"][0]
    if noise_seed is None:
        assert "noise_seed" not in report["configuration"]
        assert "noise_seed" not in row
    else:
        assert report["configuration"]["noise_seed"] == row["noise_seed"] == noise_seed
        scene, *_ = audit.prepare(point, camera_dir / "camera_P_per_frame_v1.npz")
        scene = audit.configured_scene(scene, report["configuration"])
        noisy = audit.noisy_pixels(scene, point["point"], noise_seed)
        expected_rms = np.sqrt(np.mean(model.image_residual(noisy, row["parameters"]) ** 2))
        assert row["training_pixel_rms"] == pytest.approx(expected_rms, abs=1e-12)
    if warm:
        report = json.loads((output / "report.json").read_text())
        source = Path(audit.__file__).with_name("nested_rebound.py").resolve()
        assert any(
            b["resolved_path"] == str(source) and b["record"] == file_record(source)
            for b in report["input_bindings"]
        )
        assert report["results"][0]["nested_rebound_evidence"] is not None
        if warm == "direct":
            assert report["configuration"]["retain_direct_rebound"]
            assert report["results"][0]["nested_rebound_evidence"]["direct"]["status"] == "measured"


def test_oracle_adapter_preserves_native_train_test_disjointness(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    scene, seed, xyz, paths, heldout = audit.prepare(point, camera)
    assert seed.shape == (15,)
    for train, test, target, path in zip(
        scene.observation_frames, heldout.observation_frames, xyz, paths, strict=True
    ):
        assert not set(train) & set(test)
        assert all(train % 3 != 0) and all(test % 3 == 0)
        np.testing.assert_array_equal(target, audit.sample(path, train))
    # Alter only held-out picture truth: training targets must not change.
    altered = replace(heldout, pixels=tuple(p + 100 for p in heldout.pixels))
    assert np.array_equal(scene.pixels[0], audit.prepare(point, camera)[0].pixels[0])
    assert not np.array_equal(altered.pixels[0], heldout.pixels[0])


def test_integer_contact_exposure_has_one_outgoing_owner(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    scene, _, xyz, paths, heldout = audit.prepare(point, camera)
    assert 13 not in scene.observation_frames[0]
    assert 13 in scene.observation_frames[1]
    all_frames = np.concatenate((*scene.observation_frames, *heldout.observation_frames))
    np.testing.assert_array_equal(np.sort(all_frames), np.arange(1, 27))
    for target, frames, path in zip(xyz, scene.observation_frames, paths, strict=True):
        np.testing.assert_array_equal(target, audit.sample(path, frames))


def test_integer_contact_heldout_exposure_has_one_owner(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    point["flights"][0]["end_frame"] = 12.0
    point["flights"][1]["start_frame"] = 12.0
    scene, _, _, _, heldout = audit.prepare(point, camera)
    assert 12 not in heldout.observation_frames[0]
    assert 12 in heldout.observation_frames[1]
    assert all(12 not in frames for frames in scene.observation_frames)
    all_frames = np.concatenate((*scene.observation_frames, *heldout.observation_frames))
    np.testing.assert_array_equal(np.sort(all_frames), np.arange(1, 27))


def test_visibility_control_masks_only_first_training_arc_not_truth_or_holdout(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    point["flights"][0]["bounces"] = [{"frame": 5.5}]
    scene, _, xyz, _, heldout = audit.prepare(point, camera)
    original = copy.deepcopy(point)
    result, masks, evidence = audit.apply_training_visibility(
        scene, point, "first_flight_postbounce_only"
    )
    assert min(result.observation_frames[0]) > 5.5
    np.testing.assert_array_equal(result.pixels[1], scene.pixels[1])
    assert len(xyz[0][masks[0]]) == len(result.pixels[0])
    assert min(heldout.observation_frames[0]) < 5.5
    assert not evidence["heldout_images_changed"] and point == original
    unchanged, _, _ = audit.apply_training_visibility(scene, point, "all")
    assert unchanged is scene
    point["flights"][0]["bounces"][0]["frame"] = 12.0
    with pytest.raises(ValueError, match="four post-bounce"):
        audit.apply_training_visibility(scene, point, "first_flight_postbounce_only")


def test_bounce_objective_never_receives_future_post_ending_event_evidence(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    scene, *_ = audit.prepare(point, camera)
    point["flights"][0]["bounces"] = [{"frame": 9.5}]
    point["flights"][1]["bounces"] = [{"frame": 25.5}, {"frame": 27.5}]
    groups, excluded = audit.in_scope_bounce_evidence(point, scene)
    assert [g.tolist() for g in groups] == [[9.5], [25.5]]
    assert excluded == [{"flight_index": 1, "frame": 27.5, "reason": "outside_modeled_flight"}]
    point["flights"][1]["bounces"][1]["frame"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite source bounce"):
        audit.in_scope_bounce_evidence(point, scene)


def test_supplied_rounding_preserves_original_truth_pictures_and_terminal_time(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    scene, *_ = audit.prepare(point, camera)
    point["flights"][0]["bounces"] = [{"frame": 9.5}]
    point["flights"][1]["bounces"] = [{"frame": 20.2}, {"frame": 26.0}, {"frame": 27.2}]
    before = copy.deepcopy(point)
    pixels = [p.copy() for p in scene.pixels]
    supplied, receipt = audit.conditioning_bounces(point, scene, "nearest_native_interior")
    assert point == before
    assert [b["frame"] for f in supplied["flights"] for b in f["bounces"]] == [10, 20, 26, 27.2]
    assert receipt[0]["training_exposure_available"]
    assert max(abs(r["delta_frames"]) for r in receipt) == 0.5
    for old, new in zip(pixels, scene.pixels):
        np.testing.assert_array_equal(old, new)
    exact, _ = audit.conditioning_bounces(point, scene, "exact")
    assert exact == point
    groups, excluded = audit.in_scope_bounce_evidence(supplied, scene)
    assert [g.tolist() for g in groups] == [[10], [20, 26]]
    assert excluded[0]["frame"] == 27.2


def test_rounding_never_uses_heldout_pictures_or_repairs_topology(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    scene, *_ = audit.prepare(point, camera)
    point["flights"][0]["bounces"] = [{"frame": 9.2}]
    _, receipt = audit.conditioning_bounces(point, scene, "nearest_native_interior")
    assert receipt[0]["supplied_frame"] == 9
    assert not receipt[0]["training_exposure_available"]
    for frames in ([12.7], [8.6, 9.1]):
        point["flights"][0]["bounces"] = [{"frame": f} for f in frames]
        with pytest.raises(ValueError, match="boundary|collide"):
            audit.conditioning_bounces(point, scene, "nearest_native_interior")


def test_oracle_adapter_refuses_approximate_truth_and_missing_camera(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    point["trajectory_truth"]["schema"] = "approximation"
    with pytest.raises(ValueError, match="recorded exact"):
        audit.prepare(point, camera)
    point["trajectory_truth"]["schema"] = "simulation_trajectory_samples_v1"
    np.savez(camera, clips=np.array(["pt0001"]), frames=np.array([1]), P=np.eye(3, 4)[None])
    with pytest.raises(KeyError):
        audit.prepare(point, camera)


def test_exact_spatial_success_is_never_promoted_to_complete_point(tmp_path):
    point, camera, parameters = fixture_point(tmp_path)
    scene, _, _, paths, heldout = audit.prepare(point, camera)
    scene = replace(scene, spin_parameters=np.zeros((2, 3)))
    heldout = replace(heldout, spin_parameters=scene.spin_parameters)
    result = {
        "parameters": parameters,
        "junction_gaps_m": [0.0],
        "optimizer_success": True,
        "objective_calls": 1,
        "final_pixel_rms": 0.0,
    }
    measured = audit.measure(scene, result, paths, heldout)
    assert measured["trajectory_and_endpoint_tolerances_met"]
    assert measured["heldout_pixel_rms"] < 1e-9
    assert measured["complete_point_accepted"] is False


def test_failures_remain_in_point_and_flight_denominators():
    summary = audit.summarize(
        [
            {
                "status": "measured",
                "expected_flights": 3,
                "trajectory_and_endpoint_tolerances_met": True,
            },
            {"status": "failed", "expected_flights": 5},
        ]
    )
    assert summary["points"] == 2 and summary["expected_flights"] == 8
    assert summary["trajectory_and_endpoint_tolerances_met"] == 1
    assert summary["complete_points_accepted"] == 0


@pytest.mark.parametrize("width", [0.0, 0.25, 1.0])
def test_explicit_bounce_uncertainty_reaches_fitter_and_receipt(tmp_path, monkeypatch, width):
    point, camera, _ = fixture_point(tmp_path)
    camera_dir = tmp_path / "control"
    camera_dir.mkdir()
    camera.rename(camera_dir / "camera_P_per_frame_v1.npz")
    original_fit = model.fit
    received = []

    def captured_fit(*args, **kwargs):
        received.append(kwargs["bounce_uncertainty_frames"])
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(model, "fit", captured_fit)
    record = audit.run_point(
        point,
        str(tmp_path),
        "pixels",
        2,
        30,
        fit_bounce_windows=True,
        bounce_uncertainty_frames=width,
    )
    assert record["status"] == "measured"
    assert record["bounce_uncertainty_frames"] == width
    assert received == [width]
    assert record["bounce_window_evidence"]["uncertainty_frames"] == width


@pytest.mark.parametrize("width", ["nan", "-0.01", "1.01", "inf"])
def test_cli_rejects_invalid_uncertainty_before_creating_outputs(tmp_path, monkeypatch, width):
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_benchmark",
            "--truth",
            "missing.json",
            "--cameras-root",
            "missing",
            "--output",
            str(output),
            "--fit-bounce-windows",
            "--bounce-uncertainty-frames",
            width,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "width,mode",
    [
        ("0", "point_scales"),
        ("nan", "point_scales"),
        ("inf", "point_scales"),
        ("-1", "point_scales"),
        ("0.02", "fixed"),
    ],
)
def test_cli_rejects_invalid_rebound_prior_before_outputs(tmp_path, monkeypatch, width, mode):
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_benchmark",
            "--truth",
            "missing.json",
            "--cameras-root",
            "missing",
            "--output",
            str(output),
            "--dynamics",
            "measured_240hz",
            "--rebound-mode",
            mode,
            "--rebound-prior-scale",
            width,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_oracle_records_the_applied_rebound_prior(tmp_path):
    point, camera, _ = fixture_point(tmp_path)
    camera_dir = tmp_path / "control"
    camera_dir.mkdir()
    camera.rename(camera_dir / "camera_P_per_frame_v1.npz")
    row = audit.run_point(
        point,
        str(tmp_path),
        "pixels",
        1,
        30,
        dynamics="measured_240hz",
        rebound_mode="point_scales",
        rebound_prior_scale=0.02,
    )
    assert row["status"] == "measured"
    assert row["rebound_prior_evidence"]["scale"] == 0.02
    assert row["rebound_prior_evidence"]["cost"] == 0


def test_free_spin_and_explicit_xyz_fit_are_labeled_oracle():
    scene, truth = model.control()
    target = tuple(f["positions"] for f in model.chain(scene, truth))
    seed = np.r_[truth + 0.01, np.full(6, 0.01)]
    result = model.fit(scene, seed, optimize_spin=True, oracle_positions=target, max_nfev=20)
    assert result["conditioning"] == "oracle_native_xyz"
    assert result["optimized_spin"] is True
    assert result["final_pixel_rms"] < 0.01
    assert result["junction_gaps_m"] == [0.0]
    with pytest.raises(ValueError, match="oracle XYZ"):
        model.fit(scene, seed, optimize_spin=True, oracle_positions=(target[0],))


def test_fractional_scoring_queries_do_not_replace_native_pictures():
    scene, truth = model.control()
    original = scene.observation_frames[0].copy()
    model.chain(scene, truth, query_frames=(np.array([1.5, 12.5]), np.array([13.5, 25.5])))
    np.testing.assert_array_equal(scene.observation_frames[0], original)
    with pytest.raises(ValueError, match="trajectory queries"):
        model.chain(scene, truth, query_frames=(np.array([0.5]), np.array([14.0])))


def test_frozen_profile_replay_preserves_observations_and_refuses_unsupported_pairing():
    scene, _ = model.control()
    legacy = audit.configured_scene(scene, {})
    assert legacy.dynamics == "legacy_cross" and legacy.bounce_profile == "nominal"
    for profile in audit.BOUNCE_PROFILES:
        configured = audit.configured_scene(
            scene, {"dynamics": "measured_240hz", "bounce_profile": profile}
        )
        assert configured.bounce_profile == profile
        assert configured.pixels is scene.pixels
        assert configured.cameras is scene.cameras
        assert configured.contact_frames is scene.contact_frames
    for configuration in [
        {"bounce_profile": "retention_low"},
        {"dynamics": "measured_240hz", "bounce_profile": "unknown"},
    ]:
        with pytest.raises(ValueError, match="bounce profile"):
            audit.configured_scene(scene, configuration)
