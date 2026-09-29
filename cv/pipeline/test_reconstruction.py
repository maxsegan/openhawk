import json
from types import SimpleNamespace

import numpy as np
import pytest

import cv.pipeline.reconstruction as reconstruction_module
from cv.pipeline.contact_frame_refiner import RefinedContact

from cv.pipeline.reconstruction import (
    active_play_content_paths,
    apply_contact_frame_observation,
    apply_match_shared_priors,
    apply_shared_contact_times,
    apply_arc_junction_observation_overrides,
    append_terminal_contacts,
    bounded_contact_ray_search,
    complete_point_gate,
    contact_pose_witness,
    contact_side_evidence,
    finalize_complete_point_decision,
    joint_rally_refinement_report,
    load_automatic_point_universe,
    net_collision_anchor,
    out_of_frame_gap,
    PointCamera,
    point_gate,
    refit_exact_contact_rays,
    restrict_fits_to_attempts,
    select_reconstruction_bounce_anchor,
    spatial_fit_scope,
    terminal_residual_growth_frame,
    tracking_arc_fit_scope,
    topology_flight_key,
    upstream_hold_has_retained_tracking_arc,
    validate_artifact_cadence,
    verify_hard_bounce_anchors,
    upstream_hold_is_repairable,
)


@pytest.mark.parametrize("first", [102.0, 101.5])
def test_refiner_cannot_replace_first_contact_with_unmodeled_toss_picture(first):
    contact = {
        "frame": 102.0,
        "image_observation_override": [1035.0, 517.0],
        "image_observation_frame": 102.0,
        "image_observation_source": "contact_emission_native_xy",
    }
    refinement = RefinedContact(101, -1, (1060.0, 550.0), 10.0, False, "turn", {}, 0.0, 102)
    before = dict(contact)
    apply_contact_frame_observation(contact, refinement, first_physical_frame=first)
    assert {k: contact[k] for k in before} == before
    evidence = contact["contact_frame_refinement"]
    assert evidence["observation_applied"] is False
    assert evidence["observation_rejection_reason"] == "picture_precedes_modeled_rally"


@pytest.mark.parametrize("first", [10.0, 101.0])
def test_refiner_preserves_owned_incoming_and_exact_start_observations(first):
    contact = {"frame": 102.0}
    refinement = RefinedContact(101, -1, (1060.0, 550.0), 10.0, False, "turn", {}, 0.0, 102)
    apply_contact_frame_observation(contact, refinement, first_physical_frame=first)
    assert contact["frame"] == 102.0
    assert contact["image_observation_frame"] == 101.0
    assert contact["image_observation_override"] == [1060.0, 550.0]
    assert contact["contact_frame_refinement"]["observation_applied"] is True


def test_abstained_refinement_keeps_original_observation():
    contact = {"frame": 102.0, "image_observation_frame": 102.0}
    refinement = RefinedContact(102, 0, None, 120.0, True, "uncertain", {}, None, 102)
    apply_contact_frame_observation(contact, refinement, first_physical_frame=102.0)
    assert contact["image_observation_frame"] == 102.0
    assert contact["contact_frame_refinement"]["observation_applied"] is False


def test_shared_contact_time_is_propagated_to_adjacent_attempts() -> None:
    contacts = [{"frame": 0.0}, {"frame": 10.0}, {"frame": 20.0}]
    attempts = [
        {"flight_index": 0, "start_frame": 0.0, "end_frame": 10.0},
        {"flight_index": 1, "start_frame": 10.0, "end_frame": 20.0},
    ]
    fit = SimpleNamespace(_shared_contacts=[{"contact_index": 1, "frame": 9.0}])
    camera = SimpleNamespace(quality_at=lambda frame: {"frame": frame})

    apply_shared_contact_times(contacts, attempts, {0: fit, 1: fit}, camera, 25.0)

    assert contacts[1]["frame"] == 9.0
    assert contacts[1]["image_observation_frame"] == 10.0
    assert attempts[0]["end_frame"] == 9.0
    assert attempts[1]["start_frame"] == 9.0
    assert attempts[0]["duration_seconds"] == 9.0 / 25.0
    assert attempts[1]["shared_contact_time_adjustment"]["original_start_frame"] == 10.0


@pytest.mark.parametrize("observation,owner", [(9.0, 0), (10.0, 1), (9.4, 0)])
def test_contact_pixel_samples_its_own_side_of_racket_impact(observation, owner):
    contacts = [
        {"frame": 1.0},
        {"frame": 9.4, "image_observation_frame": observation},
        {"frame": 20.0},
    ]
    calls = []

    def state(index, frame):
        calls.append((index, frame))
        return np.array([observation, 3.0, 1.0]), np.zeros(3)

    fits = {
        i: SimpleNamespace(state=lambda frame, fps, surface, i=i: state(i, frame)) for i in (0, 1)
    }
    camera = SimpleNamespace(p_at=lambda _: np.eye(3, 4))
    result = reconstruction_module.point_contact_reprojection(
        1, contacts, fits, {observation: np.array([observation, 3.0])}, camera, {}, 25.0, "hard"
    )
    assert calls == [(owner, observation)]
    assert result["sampled_flight_index"] == owner
    assert result["position_available"] is True
    assert result["error_px"] == pytest.approx(0.0)


def test_missing_incoming_fit_cannot_be_replaced_by_outgoing_extrapolation():
    contacts = [{"frame": 1.0}, {"frame": 9.4, "image_observation_frame": 9.0}, {"frame": 20.0}]
    fits = {1: SimpleNamespace(state=lambda *args: pytest.fail("crossed racket impulse"))}
    result = reconstruction_module.point_contact_reprojection(
        1, contacts, fits, {}, None, {}, 25.0, "hard"
    )
    assert result["available"] is False
    assert result["position_available"] is False
    assert result["error_px"] is None


@pytest.mark.parametrize("target,tracked,owner", [(10.0, 9, 0), (9.0, 10, 1), (9.5, 9, 0)])
def test_contact_gate_uses_actual_picture_time_not_requested_missing_time(target, tracked, owner):
    contacts = [{"frame": 1.0}, {"frame": 9.4, "image_observation_frame": target}, {"frame": 20.0}]
    calls = []

    def state(index, frame):
        calls.append((index, frame))
        return np.array([frame, 3.0, 1.0]), np.zeros(3)

    fits = {i: SimpleNamespace(state=lambda f, fps, s, i=i: state(i, f)) for i in (0, 1)}
    camera_frames = []

    def projection(frame):
        camera_frames.append(frame)
        return np.eye(3, 4)

    result = reconstruction_module.point_contact_reprojection(
        1,
        contacts,
        fits,
        {tracked: np.array([tracked, 3.0])},
        SimpleNamespace(p_at=projection),
        {tracked: 0.4},
        25.0,
        "hard",
    )
    assert calls == [(owner, float(tracked))]
    assert camera_frames == [float(tracked)]
    assert result["observation_frame"] == tracked
    assert result["requested_observation_frame"] == target
    assert result["source_frames"] == [tracked]
    assert result["confidence"] == pytest.approx(0.4)
    assert result["error_px"] == pytest.approx(0.0)
    assert contacts[1]["image_observation_frame"] == target


def test_contact_gate_does_not_interpolate_a_pixel_across_racket_impact():
    contacts = [{"frame": 1.0}, {"frame": 9.5}, {"frame": 20.0}]
    fits = {
        0: SimpleNamespace(state=lambda frame, *args: (np.array([0.0, 2.0, 1.0]), np.zeros(3))),
        1: SimpleNamespace(state=lambda *args: pytest.fail("tie must choose earlier actual frame")),
    }
    result = reconstruction_module.point_contact_reprojection(
        1,
        contacts,
        fits,
        {9: np.array([0.0, 2.0]), 10: np.array([5.0, 2.0])},
        SimpleNamespace(p_at=lambda f: np.eye(3, 4)),
        {},
        25.0,
        "hard",
    )
    assert result["observation_frame"] == 9.0
    assert result["error_px"] == 0.0
    assert result["source_frames"] == [9]


def test_explicit_contact_pixel_time_is_not_replaced_by_nearest_track_time():
    contacts = [
        {"frame": 1.0},
        {"frame": 9.4, "image_observation_frame": 10.0, "image_observation_override": [10.0, 3.0]},
        {"frame": 20.0},
    ]
    fits = {
        0: SimpleNamespace(state=lambda *args: pytest.fail("explicit picture belongs to outgoing")),
        1: SimpleNamespace(state=lambda f, *args: (np.array([f, 3.0, 1.0]), np.zeros(3))),
    }
    result = reconstruction_module.point_contact_reprojection(
        1,
        contacts,
        fits,
        {9: np.array([9.0, 3.0])},
        SimpleNamespace(p_at=lambda f: np.eye(3, 4)),
        {},
        25.0,
        "hard",
    )
    assert result["observation_frame"] == 10.0
    assert result["error_px"] == 0.0


@pytest.mark.parametrize("xyz", [[np.nan, 0, 1], [1, 2]])
def test_contact_sample_invalid_state_abstains(xyz):
    contacts = [{"frame": 1.0}, {"frame": 9.0}]
    fits = {0: SimpleNamespace(state=lambda *args: (np.array(xyz), np.zeros(3)))}
    result = reconstruction_module.point_contact_reprojection(
        0, contacts, fits, {}, None, {}, 25.0, "hard"
    )
    assert result["reason"] == "observation_state_invalid"
    assert result["position_available"] is False


def test_missing_contact_observation_position_holds_the_point():
    decision, reasons = point_gate(
        active_valid=True,
        attempted=1,
        fits=[
            {
                "rms_px": 2.0,
                "speed_kmh": 80.0,
                "end_contact_reprojection": {"position_available": False, "error_px": None},
                "start_frame": 1.0,
                "end_frame": 10.0,
                "start_xyz": [0.0, 0.0, 1.0],
                "end_xyz": [1.0, 1.0, 1.0],
            }
        ],
        smoothing={
            "coverage": 1.0,
            "maximum_gap_seconds": 0.0,
            "long_gaps": 0,
            "repair_rate": 0.0,
            "post_heal_teleport_rate": 0.0,
        },
    )
    assert decision == "hold"
    assert reasons == ["physics_contact_observation_uncovered"]


def test_compact_fit_propagates_physics_through_observation_gap() -> None:
    fit = SimpleNamespace(
        theta=np.zeros(9),
        rms_px=1.0,
        n_obs=2,
        obs_frames=np.asarray([0, 4]),
        xs_obs=np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 16.0]]),
        vs_obs=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 8.0]]),
        bounces=[],
    )
    fit.state = lambda frame, _fps, _surface: (
        np.asarray([frame, 0.0, frame**2]),
        np.asarray([1.0, 0.0, 2.0 * frame]),
    )

    compact = reconstruction_module.compact_fit(
        fit,
        25.0,
        "hard",
        start_frame=0.0,
        end_frame=4.0,
    )

    assert [row["frame"] for row in compact["trajectory"]] == [0, 1, 2, 3, 4]
    assert compact["trajectory"][2]["xyz"] == [2.0, 0.0, 4.0]
    assert compact["observations"] == 2


def test_run_applies_contact_weighting_inside_worker_process(tmp_path, monkeypatch) -> None:
    configured = []
    monkeypatch.setattr(
        reconstruction_module,
        "configure_contact_adjacent_weighting",
        configured.append,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"matches": []}\n')

    with pytest.raises(FileNotFoundError, match="active_play"):
        reconstruction_module._run_reconstruction(
            tmp_path,
            manifest,
            clips=[],
            automatic_boundaries={},
            max_nfev=1,
            workers=1,
            downweight_contact_adjacent=True,
        )

    assert configured == [True]


def test_point_camera_uses_per_frame_homography_when_present(tmp_path) -> None:
    projection = np.eye(3, 4)
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=np.array(["pt0001", "pt0001"]),
        frames=np.array([10, 20]),
        P=np.stack([projection, projection]),
    )
    np.savez_compressed(
        tmp_path / "court_H_per_point.npz",
        pts=np.array([1]),
        H=np.stack([np.eye(3)]),
    )
    per_frame = np.stack(
        [
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ]
    )
    np.savez_compressed(
        tmp_path / "court_H_per_frame_v1.npz",
        clips=np.array(["pt0001", "pt0001"]),
        frames=np.array([10, 20]),
        H=per_frame,
    )

    camera = PointCamera(tmp_path, "pt0001")

    np.testing.assert_array_equal(camera.h_at(10), per_frame[0])
    np.testing.assert_array_equal(camera.h_at(19), per_frame[1])


def test_point_camera_does_not_override_rejected_registration(tmp_path) -> None:
    projection = np.eye(3, 4)
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=np.asarray(["pt0001", "pt0001"]),
        frames=np.asarray([10, 11]),
        P=np.stack([projection, projection]),
        reliable=np.asarray([False, False]),
        source=np.asarray(["direct+registered_interpolated", "direct+anchor_static_fallback"]),
        ground_residual_px=np.asarray([0.1, 0.2]),
        net_residual_px=np.asarray([2.0, 3.0]),
        confidence=np.asarray([0.8, 0.8]),
        fallback_ancestry=np.asarray(["[]", "[]"]),
        frame_scope=np.asarray(["frame_track", "frame_track"]),
        reference_frame=np.asarray(["f_0010.jpg", "f_0010.jpg"]),
    )
    np.savez_compressed(
        tmp_path / "court_H_per_point.npz",
        pts=np.asarray([1]),
        H=np.asarray([np.eye(3)]),
    )

    summary = PointCamera(tmp_path, "pt0001").quality_summary([[10.0, 11.0]])

    assert not summary["accepted"]
    assert summary["raw_reliable_fraction"] == 0.0
    assert summary["reliable_fraction"] == 0.0
    assert summary["lineage_recovered_frames"] == 0


def test_point_camera_falls_back_to_point_homography(tmp_path) -> None:
    projection = np.eye(3, 4)
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=np.array(["pt0001"]),
        frames=np.array([10]),
        P=np.stack([projection]),
    )
    point_h = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.savez_compressed(
        tmp_path / "court_H_per_point.npz",
        pts=np.array([1]),
        H=np.stack([point_h]),
    )

    camera = PointCamera(tmp_path, "pt0001")

    np.testing.assert_array_equal(camera.h_at(999), point_h)


@pytest.mark.parametrize(
    "extra",
    [
        {"k1": [1e-12], "dist_center": [[960, 540]]},
        {"k1": [0.0]},
        {"dist_center": [[960, 540]]},
        {"k1": [[0.0]], "dist_center": [[960, 540]]},
        {"k1": [float("nan")], "dist_center": [[960, 540]]},
        {"k1": [0.0], "dist_center": [[float("nan"), 540]]},
        {"k2": [1e-15]},
        {"k2": [float("inf")]},
    ],
)
def test_pinhole_camera_rejects_unsupported_or_malformed_distortion(tmp_path, extra):
    from cv.pipeline.reconstruction import UnsupportedCameraModelError

    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=["pt0001"],
        frames=[10],
        P=[np.eye(3, 4)],
        **extra,
    )
    with pytest.raises(UnsupportedCameraModelError):
        PointCamera(tmp_path, "pt0001")


def test_pinhole_camera_accepts_explicit_zero_distortion_and_ignores_other_clips(tmp_path):
    projection = np.eye(3, 4)
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=["pt0001", "pt0002"],
        frames=[10, 10],
        P=[projection, projection],
        k1=[0.0, 1e-8],
        k2=[0.0, 1e-15],
        dist_center=[[960.0, 540.0], [960.0, 540.0]],
    )
    np.savez_compressed(tmp_path / "court_H_per_point.npz", pts=[1], H=[np.eye(3)])
    np.testing.assert_array_equal(PointCamera(tmp_path, "pt0001").p_at(10), projection)


def test_unsupported_lens_model_is_a_point_abstention_before_track_loading(tmp_path, monkeypatch):
    match = tmp_path / "match"
    match.mkdir()
    np.savez_compressed(
        match / "camera_P_per_frame_v1.npz",
        clips=["pt0001"],
        frames=[10],
        P=[np.eye(3, 4)],
        k1=[1e-8],
        dist_center=[[960.0, 540.0]],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"matches": [{"id": "match", "source_fps": 25, "surface": "hard"}]})
    )
    (tmp_path / "active_play_v1.json").write_text(
        json.dumps({"match/pt0001": {"point_valid": True}})
    )
    (tmp_path / "untouched_tracking_point_gate_v1.json").write_text(json.dumps({"rows": []}))
    monkeypatch.setattr(reconstruction_module, "validate_artifact_cadence", lambda *_: {})
    monkeypatch.setattr(
        reconstruction_module,
        "load_track",
        lambda *_: pytest.fail("unsupported lens reached fitter inputs"),
    )
    result = reconstruction_module._run_reconstruction(
        tmp_path,
        manifest,
        clips=["match__pt0001"],
        automatic_boundaries={},
        max_nfev=1,
        workers=1,
    )
    point = result["points_detail"][0]
    assert point["decision"] == "hold"
    assert point["reasons"] == ["unsupported_camera_model"]
    assert not point["camera_calibration"]["accepted"]
    assert point["fits"] == []


def test_contact_pose_witness_estimates_height_from_player_box() -> None:
    witness = contact_pose_witness(
        {
            "near": {
                10: {
                    "x0": 100.0,
                    "y0": 100.0,
                    "x1": 200.0,
                    "y1": 300.0,
                    "conf": 0.9,
                }
            }
        },
        contact={"frame": 10.0, "side": "near"},
        ball={10: np.array([150.0, 200.0])},
        players={"near": {10: np.array([5.0, 3.0])}},
        camera=SimpleNamespace(p_at=lambda _frame: np.eye(3, 4)),
        fps=25.0,
    )

    assert witness is not None
    assert witness["apparent_ball_height_m"] == 0.925
    assert witness["apparent_height_confidence"] == 0.9
    assert witness["ball_vertical_box_fraction"] == 0.5


def test_contact_pose_witness_uses_accepted_physical_racket_branch() -> None:
    projection = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    witness = contact_pose_witness(
        {"near": {}, "far": {}},
        {
            "near": {
                10: {
                    "decision": "accept",
                    "camera": {"reliable": True},
                    "selected_branch": 0,
                    "branches": [
                        {
                            "diagnostics": {
                                "quality": 0.30,
                                "bone_rms_m": 0.05,
                                "root_error_m": 0.10,
                                "minimum_z_m": 0.0,
                            },
                            "body_prior_projection_quality": {"reliability": 0.8},
                            "racket_hypotheses": [
                                {
                                    "hand": "right",
                                    "grip": "model_wrist_orientation",
                                    "head_center_xyz": [1.0, 2.0, 1.0],
                                }
                            ],
                        },
                    ],
                }
            },
            "far": {},
        },
        contact={"frame": 10.0, "side": "near"},
        ball={10: np.array([1.0, 2.0])},
        players={"near": {10: np.array([1.0, 2.0])}},
        camera=SimpleNamespace(p_at=lambda _frame: projection),
        fps=25.0,
    )

    assert witness is not None
    assert witness["source"] == "court_registered_physical_racket_branches"
    assert witness["racket_head_hypotheses"][0]["ball_head_distance_px"] == 0.0
    assert witness["confidence"] == 0.8


def test_contact_pose_witness_keeps_calibrated_held_model_as_soft_evidence() -> None:
    projection = np.eye(3, 4)
    witness = contact_pose_witness(
        {"near": {}, "far": {}},
        {
            "near": {
                10: {
                    "decision": "hold",
                    "camera": {"reliable": True},
                    "selected_branch": 0,
                    "branches": [
                        {
                            "body_prior_backend": "gvhmr_smpl",
                            "body_prior_projection_quality": {"reliability": 0.75},
                            "diagnostics": {
                                "quality": 0.09,
                                "bone_rms_m": 0.10,
                                "root_error_m": 0.30,
                                "minimum_z_m": 0.0,
                            },
                            "racket_hypotheses": [
                                {
                                    "hand": "right",
                                    "grip": "model_wrist_orientation",
                                    "head_center_xyz": [1.0, 2.0, 1.0],
                                }
                            ],
                        },
                        {
                            "body_prior_backend": "gem_x_mhr",
                            "body_prior_projection_quality": {"reliability": 0.25},
                            "diagnostics": {
                                "quality": 0.09,
                                "bone_rms_m": 0.10,
                                "root_error_m": 0.30,
                                "minimum_z_m": 0.0,
                            },
                            "racket_hypotheses": [
                                {
                                    "hand": "right",
                                    "grip": "model_wrist_orientation",
                                    "head_center_xyz": [1.1, 2.0, 1.0],
                                }
                            ],
                        },
                    ],
                }
            },
            "far": {},
        },
        contact={"frame": 10.0, "side": "near"},
        ball={10: np.array([1.0, 2.0])},
        players={"near": {10: np.array([1.0, 2.0])}},
        camera=SimpleNamespace(p_at=lambda _frame: projection),
        fps=25.0,
    )

    assert witness is not None
    assert witness["confidence"] == 0.375
    assert {row["body_prior_backend"] for row in witness["racket_head_hypotheses"]} == {
        "gvhmr_smpl",
        "gem_x_mhr",
    }


def test_contact_pose_witness_does_not_trust_uncalibrated_model_branch() -> None:
    projection = np.eye(3, 4)
    witness = contact_pose_witness(
        {"near": {}, "far": {}},
        {
            "near": {
                10: {
                    "camera": {"reliable": True},
                    "branches": [
                        {
                            "body_prior_backend": "unscored_temporal_model",
                            "diagnostics": {
                                "quality": 0.30,
                                "bone_rms_m": 0.05,
                                "root_error_m": 0.10,
                                "minimum_z_m": 0.0,
                            },
                            "racket_hypotheses": [
                                {
                                    "hand": "right",
                                    "grip": "model_wrist_orientation",
                                    "head_center_xyz": [1.0, 2.0, 1.0],
                                }
                            ],
                        }
                    ],
                }
            },
            "far": {},
        },
        contact={"frame": 10.0, "side": "near"},
        ball={10: np.array([1.0, 2.0])},
        players={"near": {10: np.array([1.0, 2.0])}},
        camera=SimpleNamespace(p_at=lambda _frame: projection),
        fps=25.0,
    )

    assert witness is not None
    assert witness["confidence"] == 0.1


def test_body_scale_bounds_contact_ray_search() -> None:
    candidates = [
        {
            "point": np.array([0.0, float(index), 1.0]),
            "prior_score": float(index),
            "geometry_prior_score": float(19 - index),
        }
        for index in range(20)
    ]

    strong = bounded_contact_ray_search(candidates, {"apparent_height_confidence": 0.9})
    weak = bounded_contact_ray_search(candidates, {"apparent_height_confidence": 0.2})

    assert len(strong) == 8
    strong_depths = {float(candidate["point"][1]) for candidate in strong}
    assert strong_depths == {0.0, 1.0, 2.0, 3.0, 16.0, 17.0, 18.0, 19.0}
    assert len(weak) > len(strong)
    assert weak[0] is candidates[0]


def test_unique_high_confidence_bounce_can_seed_recovery_without_initial_fit() -> None:
    candidates = [
        {"frame": 20.0, "probability": 0.91, "court_xy": np.array([4.0, 16.0])},
        {"frame": 22.0, "probability": 0.18, "court_xy": np.array([4.2, 16.5])},
    ]

    selected = select_reconstruction_bounce_anchor(candidates, [], fps=25.0)

    assert selected is candidates[0]


def test_ambiguous_high_confidence_bounces_do_not_force_recovery() -> None:
    candidates = [
        {"frame": 20.0, "probability": 0.91, "court_xy": np.array([4.0, 16.0])},
        {"frame": 22.0, "probability": 0.82, "court_xy": np.array([4.2, 16.5])},
    ]

    assert select_reconstruction_bounce_anchor(candidates, [], fps=25.0) is None


def test_coherent_arc_junction_overrides_a_tracking_outlier() -> None:
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)

    class Fit:
        rms_px = 4.0

        def __init__(self, point):
            self.point = np.asarray(point, float)

        def state(self, *_args):
            return self.point, np.zeros(3)

    contacts = [
        {"frame": 10.0},
        {"frame": 20.0},
        {"frame": 30.0, "terminal": True},
    ]
    fits = {0: Fit([4.0, 5.0, 1.0]), 1: Fit([4.04, 5.02, 1.0])}

    apply_arc_junction_observation_overrides(
        contacts,
        fits,
        {20: np.array([500.0, 600.0])},
        camera,
        25.0,
        "hard",
    )

    assert contacts[1]["image_observation_override_provenance"]["arc_separation_px"] < 8.0
    np.testing.assert_allclose(contacts[1]["image_observation_override"], [402.0, 501.0])


def test_joint_ray_refit_leaves_terminal_depth_unconstrained(monkeypatch) -> None:
    contacts = [
        {"frame": 10.0, "side": "far", "phase": "serve"},
        {"frame": 20.0, "side": "near", "phase": "rally"},
        {"frame": 30.0, "side": "far", "phase": "terminal", "terminal": True},
    ]
    fits = {0: SimpleNamespace(rms_px=8.0), 1: SimpleNamespace(rms_px=8.0)}
    positions = {
        0: np.array([5.0, 24.0, 2.5]),
        1: np.array([5.0, 0.0, 1.2]),
        2: np.array([5.0, 24.0, 1.0]),
    }
    monkeypatch.setattr(reconstruction_module, "contact_pose_witness", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        reconstruction_module,
        "apply_arc_junction_observation_overrides",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        reconstruction_module,
        "contact_ray_anchor",
        lambda contact, seed, *_args, **_kwargs: (
            (None, {"available": False})
            if contact.get("terminal")
            else (np.asarray(seed), {"available": True})
        ),
    )
    monkeypatch.setattr(
        reconstruction_module,
        "contact_ray_candidates",
        lambda *_args, **_kwargs: (
            [
                {
                    "point": np.array([5.0, 0.0, 1.4]),
                    "prior_score": 0.0,
                    "player_distance_m": 0.5,
                    "wrist_distance_m": 0.7,
                }
            ],
            {"available": True},
        ),
    )
    monkeypatch.setattr(reconstruction_module, "bounce_for_shot", lambda *_args: None)
    monkeypatch.setattr(
        reconstruction_module,
        "shoot_shot",
        lambda index, *_args, **_kwargs: SimpleNamespace(rms_px=4.0 + index),
    )
    terminal_calls = []

    def terminal_fit(*args, **_kwargs):
        terminal_calls.append((args, _kwargs))
        return SimpleNamespace(rms_px=5.0, start=0, end=1)

    monkeypatch.setattr(
        reconstruction_module,
        "fit_fixed_start_terminal_segment",
        terminal_fit,
    )

    output, _, diagnostics = refit_exact_contact_rays(
        fits,
        contacts,
        positions,
        positions,
        {10: np.array([1.0, 1.0]), 20: np.array([2.0, 2.0]), 30: np.array([3.0, 3.0])},
        {"near": {20: np.array([5.0, 0.0])}, "far": {10: np.array([5.0, 24.0])}},
        SimpleNamespace(),
        25.0,
        "hard",
        20,
        [],
        {},
    )

    assert terminal_calls
    assert diagnostics[1]["joint_status"] == "two_flight_ray_solution_adopted"
    assert output[0].rms_px == 4.0
    assert output[1].rms_px == 5.0


def test_terminal_contact_extends_through_confident_same_camera_reacquisition() -> None:
    contacts = [{"frame": 20.0, "side": "near", "span": 0}]
    track = {
        **{frame: np.array([float(frame), 1.0]) for frame in range(21, 33)},
        **{frame: np.array([float(frame), 2.0]) for frame in range(43, 58)},
    }
    weights = {frame: 0.9 for frame in track}

    result = append_terminal_contacts(contacts, track, [[1, 60]], 25.0, weights)

    assert result[-1]["frame"] == 57
    assert result[-1]["source"] == "active_track_terminal_reacquired"
    assert result[-1]["reacquisition"]["runs"] == 2


def test_terminal_contact_does_not_cross_low_confidence_reacquisition() -> None:
    contacts = [{"frame": 20.0, "side": "near", "span": 0}]
    track = {
        **{frame: np.array([float(frame), 1.0]) for frame in range(21, 33)},
        **{frame: np.array([float(frame), 2.0]) for frame in range(43, 58)},
    }
    weights = {frame: (0.9 if frame < 40 else 0.2) for frame in track}

    result = append_terminal_contacts(contacts, track, [[1, 60]], 25.0, weights)

    assert result[-1]["frame"] == 32
    assert result[-1]["source"] == "active_track_terminal"


def test_terminal_contact_stops_at_second_bounce_point_end() -> None:
    contacts = [{"frame": 20.0, "side": "near", "span": 0}]
    track = {frame: np.array([float(frame), 1.0]) for frame in range(21, 81)}
    events = [
        {"event_type": "bounce", "frame": 45.0},
        {"event_type": "bounce", "frame": 70.0},
        {
            "event_type": "point_end",
            "frame": 70.0,
            "point_end": {
                "terminal_event_type": "bounce",
                "termination_kind": "second_bounce",
                "source": "test_explicit_ending",
            },
        },
    ]

    result = append_terminal_contacts(contacts, track, [[1, 90]], 25.0, boundary_events=events)

    assert result[-1]["frame"] == 70.0
    assert result[-1]["terminal_cutoff"]["source"] == "point_end"
    assert result[-1]["terminal_cutoff"]["point_end"]["frame"] == 70.0
    assert result[-1]["terminal_cutoff"]["bounce_frames"] == [45.0, 70.0]


def test_terminal_contact_without_point_end_uses_active_span_not_first_bounce() -> None:
    contacts = [{"frame": 20.0, "side": "near", "span": 0}]
    track = {frame: np.array([float(frame), 1.0]) for frame in range(21, 81)}
    events = [{"event_type": "bounce", "frame": 45.0}]

    result = append_terminal_contacts(contacts, track, [[1, 90]], 25.0, boundary_events=events)

    assert result[-1]["frame"] == 80.0
    assert result[-1]["terminal_cutoff"]["source"] == "active_span_end"


def test_terminal_contact_ignores_legacy_last_event_point_end() -> None:
    contacts = [{"frame": 20.0, "side": "near", "span": 0}]
    track = {frame: np.array([float(frame), 1.0]) for frame in range(21, 81)}
    events = [
        {
            "event_type": "point_end",
            "frame": 45.0,
            "point_end": {"source": "event_grammar_decoder_terminal_path_member"},
        }
    ]
    result = append_terminal_contacts(contacts, track, [[1, 90]], 25.0, boundary_events=events)
    assert result[-1]["frame"] == 80.0


def test_complete_point_requires_a_physical_ending_not_just_a_terminal_fit() -> None:
    accepted, reasons = complete_point_gate(
        point_decision="retain",
        attempted=1,
        solved=1,
        terminal_flights=1,
        all_shots_accepted=True,
    )
    assert not accepted
    assert "terminal_evidence_missing" in reasons


def test_terminal_residual_growth_detects_sustained_bad_tail() -> None:
    fit = SimpleNamespace(
        obs_frames=np.arange(20, dtype=float),
        _observation_errors_px=np.r_[np.ones(12), np.linspace(14.0, 30.0, 8)],
    )

    assert terminal_residual_growth_frame(fit) == 11.0


def test_high_apex_synthetic_lob_reentry_is_a_gap_not_a_termination() -> None:
    contacts = [{"frame": 10.0, "side": "near", "span": 0}]
    track = {
        **{frame: np.array([900.0 + frame, 30.0]) for frame in range(11, 24)},
        **{frame: np.array([900.0 + frame, 35.0]) for frame in range(60, 74)},
    }

    result = append_terminal_contacts(
        contacts, track, [[1, 75]], 25.0, bridge_out_of_frame_gaps=True
    )

    assert result[-1]["frame"] == 73
    assert result[-1]["reacquisition"]["out_of_frame_edges"] == ["top"]
    assert out_of_frame_gap(track[23], track[60]) == "top"


def test_in_frame_long_gap_is_not_treated_as_lob_reentry() -> None:
    assert out_of_frame_gap(np.array([900.0, 400.0]), np.array([950.0, 420.0])) is None


def test_recovery_fits_are_restricted_to_scored_attempts() -> None:
    fits = {0: object(), 2: object()}

    dropped = restrict_fits_to_attempts(fits, [{"flight_index": 0}])

    assert dropped == [2]
    assert list(fits) == [0]


def test_complete_point_gate_is_stricter_than_partial_shot_retention() -> None:
    accepted, reasons = complete_point_gate(
        point_decision="retain",
        attempted=4,
        solved=3,
        terminal_flights=1,
        all_shots_accepted=False,
        terminal_coverage={"valid": True},
    )
    assert not accepted
    assert reasons == ["incomplete_shot_solve_coverage"]

    accepted, reasons = complete_point_gate(
        point_decision="retain",
        attempted=4,
        solved=4,
        terminal_flights=1,
        all_shots_accepted=True,
        terminal_coverage={"valid": True},
    )
    assert accepted
    assert reasons == []

    accepted, reasons = complete_point_gate(
        point_decision="retain",
        attempted=4,
        solved=4,
        terminal_flights=1,
        all_shots_accepted=False,
        terminal_coverage={"valid": True},
    )
    assert not accepted
    assert reasons == ["constituent_shot_gate"]


def test_exported_point_decision_fails_closed_on_constituent_shot() -> None:
    decision, reasons = finalize_complete_point_decision(
        "retain",
        [],
        False,
        ["constituent_shot_gate"],
    )

    assert decision == "hold"
    assert reasons == ["constituent_shot_gate"]


def test_joint_rally_refinement_reports_derived_events_without_mutating_s5() -> None:
    boundaries = [
        {"event_type": "contact", "frame": 10.0, "probability": 0.9},
        {"event_type": "bounce", "frame": 20.0, "probability": 0.8},
        {"event_type": "contact", "frame": 30.0, "probability": 0.7},
    ]
    contacts = [
        {
            "frame": 10.0,
            "side": "near",
            "source": "automatic_event_boundary",
            "row": boundaries[0],
        },
        {
            "frame": 25.0,
            "side": "far",
            "source": "automatic_event_topology_insertion",
            "row": {"probability": 0.6, "origin": "two_arc_recovery"},
        },
        {
            "frame": 30.0,
            "side": "near",
            "source": "automatic_event_boundary",
            "row": boundaries[2],
        },
    ]
    topology = {
        "enabled": True,
        "selected": "insert_0_f25",
        "selection_reason": "decisive_improvement",
        "margin": 10.0,
        "scoring_stage": "post_joint_shortlist",
        "inserted": [{"frame": 25.0, "side": "far"}],
        "omitted": [],
        "retimed": [],
    }

    report = joint_rally_refinement_report(boundaries, contacts, topology)

    assert report["status"] == "selected_repair"
    assert report["upstream_artifacts_mutated"] is False
    assert [row["event_type"] for row in report["selected_events"]] == [
        "contact",
        "bounce",
        "contact",
        "contact",
    ]
    assert boundaries == [
        {"event_type": "contact", "frame": 10.0, "probability": 0.9},
        {"event_type": "bounce", "frame": 20.0, "probability": 0.8},
        {"event_type": "contact", "frame": 30.0, "probability": 0.7},
    ]


def test_joint_rally_refinement_inserts_only_physics_certified_bounces() -> None:
    boundaries = [
        {"event_type": "contact", "frame": 10.0, "probability": 0.9},
        {"event_type": "contact", "frame": 30.0, "probability": 0.8},
    ]
    contacts = [
        {"frame": 10.0, "side": "near", "row": boundaries[0]},
        {"frame": 30.0, "side": "far", "row": boundaries[1]},
    ]
    topology = {
        "enabled": True,
        "selected": "contacts_baseline",
        "selection_reason": "baseline_best",
        "margin": 0.0,
        "scoring_stage": "post_joint_shortlist",
        "inserted": [],
        "omitted": [],
        "retimed": [],
    }
    hypotheses = [
        {
            "frame": 20.0,
            "probability": 0.8,
            "source": "s5_leaky_bounce_hypothesis",
            "physics_timing_delta_frames": 0.5,
            "physics_court_delta_m": 0.4,
            "court_xy": np.array([2.0, 8.0]),
        },
        {
            "frame": 25.0,
            "probability": 0.9,
            "source": "s5_leaky_bounce_hypothesis",
            "physics_timing_delta_frames": 0.5,
            "physics_court_delta_m": 2.0,
        },
    ]

    report = joint_rally_refinement_report(boundaries, contacts, topology, hypotheses)

    assert report["modified_event_types"] == ["bounce"]
    assert [row["frame"] for row in report["operations"]["inserted_bounces"]] == [20.0]
    assert [row["event_type"] for row in report["selected_events"]] == [
        "contact",
        "bounce",
        "contact",
    ]


def test_active_play_discovery_excludes_provenance_sidecar(tmp_path) -> None:
    (tmp_path / "active_play_v1.json").write_text(
        json.dumps({"match/pt0001": {"active_spans": [[0, 10]]}})
    )
    (tmp_path / "active_play_v1.provenance.json").write_text(
        json.dumps({"schema": "pipeline_provenance_v1"})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"matches": [{"id": "match"}]}))

    assert [path.name for path in active_play_content_paths(tmp_path)] == ["active_play_v1.json"]
    assert load_automatic_point_universe(tmp_path, manifest) == ["match__pt0001"]


def test_reconstruction_rejects_mixed_cadence_artifacts(tmp_path) -> None:
    sidecar = tmp_path / "audit_frames_native_1080.coordinates.json"
    sidecar.write_text(json.dumps({"schema": "tennis.coordinate-space.v1", "fps": 60.0}))

    try:
        validate_artifact_cadence(tmp_path, 50.0)
    except ValueError as error:
        assert "manifest=50" in str(error)
        assert "frames=60" in str(error)
    else:
        raise AssertionError("mixed cadence artifacts must fail closed")


def test_hard_bounce_integrity_rejects_unanchored_final_fit() -> None:
    contacts = [{"frame": 10.0}, {"frame": 30.0}]
    anchor = {
        "frame": 20.0,
        "x": np.array([3.0, 8.0, 0.033]),
        "hard_geometry": True,
    }

    report = verify_hard_bounce_anchors({0: object()}, contacts, [anchor])

    assert report["all_satisfied"] is False
    assert report["violating_flights"] == [0]
    assert report["violations"][0]["reason"] == "hard_bounce_not_consumed"


def test_hard_bounce_integrity_accepts_only_the_exact_consumed_anchor() -> None:
    class Fit:
        pass

    contacts = [{"frame": 10.0}, {"frame": 30.0}]
    anchor = {
        "frame": 20.0,
        "x": np.array([3.0, 8.0, 0.033]),
        "hard_geometry": True,
    }
    fit = Fit()
    fit._fixed_bounce_anchor = dict(anchor)

    report = verify_hard_bounce_anchors({0: fit}, contacts, [anchor])

    assert report["all_satisfied"] is True
    assert report["satisfied_flights"] == [0]

    fit._initialization_only = True
    rejected = verify_hard_bounce_anchors({0: fit}, contacts, [anchor])
    assert rejected["all_satisfied"] is False
    assert rejected["violations"][0]["reason"] == "hard_bounce_initializer_not_refined"


def test_net_collision_anchor_pins_observed_ray_to_physical_net_plane() -> None:
    projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    class Camera:
        def p_at(self, _frame):
            return projection

    anchor, diagnostic = net_collision_anchor(
        Camera(),
        {20: np.array([5.0 / 11.885, 1.0 / 11.885])},
        20.0,
    )

    np.testing.assert_allclose(anchor, [5.0, 11.885, 1.0])
    assert diagnostic["source"] == "observed_ball_ray_physical_net_mesh_local_search"
    assert diagnostic["retiming_frames"] == 0.0


def test_net_collision_anchor_rejects_ball_above_the_net_mesh() -> None:
    projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    class Camera:
        def p_at(self, _frame):
            return projection

    anchor, diagnostic = net_collision_anchor(
        Camera(),
        {20: np.array([5.0 / 11.885, 1.2 / 11.885])},
        20.0,
    )

    assert anchor is None
    assert diagnostic["reason"] == "net_hit_outside_physical_net"


def test_s6_reconsiders_only_tracking_risk_holds() -> None:
    assert upstream_hold_is_repairable({"decision": "hold", "reasons": ["tracking_risk"]})
    assert not upstream_hold_is_repairable(
        {"decision": "hold", "reasons": ["tracking_risk", "active_play_ambiguous"]}
    )
    assert not upstream_hold_is_repairable({"decision": "hold", "reasons": ["hard_camera_failure"]})


def test_retained_tracking_arc_bypasses_only_nonhard_upstream_hold() -> None:
    tracking = {
        "arcs": [
            {"arc_id": 0, "start_frame": 10, "end_frame": 20, "decision": "retain"},
            {"arc_id": 1, "start_frame": 21, "end_frame": 23, "decision": "hold"},
        ]
    }

    assert upstream_hold_has_retained_tracking_arc(
        {
            "decision": "hold",
            "reasons": ["tracking_risk", "active_play_ambiguous"],
        },
        tracking,
    )
    assert not upstream_hold_has_retained_tracking_arc(
        {"decision": "hold", "reasons": ["tracking_risk", "hard_camera_failure"]},
        tracking,
    )


def test_tracking_arc_scope_passes_only_retained_observations() -> None:
    tracking = {
        "arcs": [
            {"arc_id": 0, "start_frame": 10, "end_frame": 14, "decision": "retain"},
            {"arc_id": 1, "start_frame": 15, "end_frame": 17, "decision": "hold"},
            {"arc_id": 2, "start_frame": 18, "end_frame": 22, "decision": "retain"},
        ]
    }

    scope = tracking_arc_fit_scope(tracking, 12.5, 20.0, set(range(10, 23)))

    assert scope["decision"] == "attempt"
    assert scope["retained_arc_ids"] == [0, 2]
    assert scope["held_arc_ids"] == [1]
    assert scope["retained_frames"] == [13, 14, 18, 19, 20]


def _box(side: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """A loader-shaped row: legacy columns plus the native pair the sidecar resolves to."""
    return {
        "side": side,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "x0_native": 2.0 * x0,
        "y0_native": 2.0 * y0,
        "x1_native": 2.0 * x1,
        "y1_native": 2.0 * y1,
    }


def test_topology_flight_key_only_reuses_unchanged_boundaries() -> None:
    start = {"frame": 10.0, "side": "near", "phase": "serve"}
    end = {"frame": 30.0, "side": "far"}

    baseline = topology_flight_key(start, end)

    assert topology_flight_key(dict(start), dict(end)) == baseline
    assert topology_flight_key({**start, "frame": 10.5}, end) != baseline
    assert topology_flight_key(start, {**end, "phase": "serve"}) != baseline
    assert topology_flight_key(start, {**end, "terminal": True}) != baseline


def test_spatial_fit_scope_withholds_only_spatial_use():
    points = {
        "match/pt0001": {
            "intervals": [
                {
                    "frame_scope": [20.0, 30.0],
                    "spatial_usable": False,
                    "timing_usable": True,
                    "reasons": ["court_support_lost"],
                }
            ]
        }
    }

    decision = spatial_fit_scope(points, "match", "pt0001", 10.0, 40.0)

    assert decision["decision"] == "withhold_spatial"
    assert decision["timing_usable"] is True


def test_spatial_fit_scope_abstains_when_artifact_has_no_point():
    decision = spatial_fit_scope({}, "match", "pt0001", 10.0, 40.0)

    assert decision["decision"] == "unknown"
    assert decision["spatial_usable"] is None


def test_contact_side_uses_temporally_near_player_boxes():
    track = {101: np.array([210.0, 170.0])}
    boxes = {
        99: [
            _box("near", 80.0, 60.0, 120.0, 100.0),
            _box("far", 5.0, 5.0, 25.0, 35.0),
        ]
    }

    evidence = contact_side_evidence(100.5, track, boxes, fps=25.0)

    assert evidence["side"] == "near"
    assert evidence["ball_frame"] == 101
    assert evidence["evidence_frames"]["near"] == 99


def test_contact_side_abstains_when_players_overlap():
    track = {100: np.array([200.0, 180.0])}
    boxes = {
        100: [
            _box("near", 80.0, 60.0, 120.0, 100.0),
            _box("far", 82.0, 60.0, 122.0, 100.0),
        ]
    }

    evidence = contact_side_evidence(100.0, track, boxes, fps=25.0)

    assert evidence["side"] == "unknown"
    assert evidence["reason"] == "players_image_overlap"


def test_point_gate_names_unresolved_player_side():
    decision, reasons = point_gate(
        active_valid=True,
        attempted=1,
        fits=[
            {
                "rms_px": 2.0,
                "speed_kmh": 80.0,
                "minimum_height_m": 0.033,
            }
        ],
        smoothing={
            "coverage": 1.0,
            "maximum_gap_seconds": 0.0,
            "long_gaps": 0,
            "repair_rate": 0.0,
            "post_heal_teleport_rate": 0.0,
        },
        unresolved_contact_sides=1,
    )

    assert decision == "hold"
    assert "player_side_unresolved" in reasons


def test_point_gate_names_bad_contact_reprojection():
    decision, reasons = point_gate(
        active_valid=True,
        attempted=1,
        fits=[
            {
                "rms_px": 2.0,
                "speed_kmh": 80.0,
                "minimum_height_m": 0.033,
                "start_contact_reprojection_px": 13.0,
                "end_contact_reprojection_px": 2.0,
            }
        ],
        smoothing={
            "coverage": 1.0,
            "maximum_gap_seconds": 0.0,
            "long_gaps": 0,
            "repair_rate": 0.0,
            "post_heal_teleport_rate": 0.0,
        },
    )

    assert decision == "hold"
    assert "physics_contact_reprojection" in reasons


def test_match_priors_only_withhold_outliers() -> None:
    points = []
    for index, height in enumerate([1.0, 1.1, 1.2, 1.3, 4.0]):
        points.append(
            {
                "match_id": "match",
                "fps": 25.0,
                "decision": "retain",
                "reasons": [],
                "complete_point_gate": {"accepted": True, "reasons": []},
                "flight_attempts": [
                    {
                        "flight_index": 0,
                        "start_side": "near",
                        "end_side": "far",
                        "start_phase": "rally",
                        "terminal_end": False,
                    }
                ],
                "fits": [
                    {
                        "flight_index": 0,
                        "status": "provisional_valid",
                        "reasons": [],
                        "start_xyz": [2.0, 2.0, height],
                        "end_xyz": [3.0, 20.0, 1.2],
                        "speed_kmh": 100.0,
                        "trajectory": [],
                    }
                ],
            }
        )

    report = apply_match_shared_priors(points)

    assert report["matches"]["match"]["fits"] == 5
    assert points[0]["fits"][0]["status"] == "provisional_valid"
    assert points[-1]["fits"][0]["status"] == "recoverable"
    assert points[-1]["decision"] == "hold"
