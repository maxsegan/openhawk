"""Scene/graphics separation must preserve the actual camera qualification contract."""

import cv2
import numpy as np
import pytest

from cv.pipeline import court, court_topology as topology


def moving_court_with_fixed_graphics():
    rng = np.random.default_rng(12)
    image = np.full((540, 960, 3), (50, 90, 45), np.uint8)
    H = cv2.getPerspectiveTransform(
        np.float32([[100, 475], [860, 475], [300, 150], [660, 150]]),
        np.float32(
            [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
        ),
    )
    for x, y in rng.integers([65, 125], [890, 445], size=(80, 2)):
        color = int(rng.integers(20, 110))
        cv2.circle(image, (int(x), int(y)), 3, (color, color, color), -1)
    for segment in topology.MODEL_SEGMENTS:
        p = cv2.perspectiveTransform(np.float32([segment]), np.linalg.inv(H))[0]
        cv2.line(image, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (255, 255, 255), 3)
    motion = np.array([[1.0, 0.0, 55.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    moved = cv2.warpPerspective(image, motion, (960, 540))
    graphic = rng.integers(0, 256, size=(110, 960, 3), dtype=np.uint8)
    image[:110] = graphic
    moved[:110] = graphic
    return image, moved, H, motion


def test_masked_sift_recovers_camera_motion_behind_screen_fixed_graphics():
    reference, target, H, motion = moving_court_with_fixed_graphics()
    with pytest.raises(ValueError):
        topology.transfer_court_homography(target, reference, H)
    for policy in ("court_observation", "court_observation_fallback"):
        fitted, evidence = topology.transfer_court_homography(
            target, reference, H, registration_mask=policy
        )
        world = np.float32(
            [[[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]]
        )
        expected = cv2.perspectiveTransform(world, motion @ np.linalg.inv(H))
        actual = cv2.perspectiveTransform(world, np.linalg.inv(fitted))
        assert np.max(np.linalg.norm(actual - expected, axis=2)) < 2.0
        assert evidence["registration_feature_mask"] == "court_observation_v1"
        assert evidence["target_topology_score"] >= 0.80
        assert bool(evidence["original_transfer_failure"]) == (
            policy == "court_observation_fallback"
        )


def test_fallback_preserves_accepted_original_without_retry(monkeypatch):
    reference, _, H, _ = moving_court_with_fixed_graphics()
    original = topology.register_static_scene
    calls = []

    def track(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(topology, "register_static_scene", track)
    off, off_evidence = topology.transfer_court_homography(reference, reference, H)
    result, evidence = topology.transfer_court_homography(
        reference, reference, H, registration_mask="court_observation_fallback"
    )
    np.testing.assert_array_equal(result, off)
    assert evidence == off_evidence
    assert calls == [{}, {}]


def test_mask_cannot_override_missing_target_paint_or_geometry(monkeypatch):
    reference, _, H, _ = moving_court_with_fixed_graphics()
    monkeypatch.setattr(
        topology,
        "register_static_scene",
        lambda *a, **k: (np.eye(3), {"inliers": 100, "inlier_ratio": 1.0}),
    )
    blank = np.zeros_like(reference)
    for policy in topology.REGISTRATION_MASK_POLICIES:
        with pytest.raises(ValueError, match="lacks target support"):
            topology.transfer_court_homography(blank, reference, H, registration_mask=policy)
        with pytest.raises(ValueError, match="geometry"):
            topology.transfer_court_homography(
                reference, reference, np.eye(3), registration_mask=policy
            )


def test_masked_registration_handles_missing_descriptors():
    blank = np.zeros((80, 100, 3), np.uint8)
    assert topology.register_static_scene(blank, blank, court_observation_mask=True) is None


def test_knn_single_neighbor_is_not_a_valid_correspondence(monkeypatch):
    class Detector:
        def detectAndCompute(self, image, mask):
            return [cv2.KeyPoint(50, 50, 3)], np.ones((1, 128), np.float32)

    monkeypatch.setattr(cv2, "SIFT_create", lambda **kwargs: Detector())
    blank = np.zeros((80, 100, 3), np.uint8)
    assert topology.register_static_scene(blank, blank, court_observation_mask=True) is None


def test_registration_policy_reaches_native_track_export(tmp_path, monkeypatch):
    import json
    from cv.pipeline import court_topology_runner as runner
    from cv.pipeline.camera_artifacts import expand_point_cameras
    from cv.pipeline.event_ground_evidence import GroundSnapshot
    from cv.pipeline.event_impulse_support import TrackSnapshot

    reference, target, H, motion = moving_court_with_fixed_graphics()
    scale = np.diag([2.0, 2.0, 1.0])
    reference = cv2.resize(reference, (1920, 1080))
    target = cv2.resize(target, (1920, 1080))
    H = H @ np.linalg.inv(scale)
    motion = scale @ motion @ np.linalg.inv(scale)
    match_out = tmp_path / "m"
    clip = match_out / "frames" / "pt0001"
    clip.mkdir(parents=True)
    # PNG-encoded native samples avoid test-only JPEG uncertainty; filenames follow the runner contract.
    for index, image in enumerate([reference, target]):
        (clip / f"f_{index:04d}.jpg").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    anchor = runner.AnchorSolution(0, "f_0000.jpg", H, "test_source_anchor", 1.0, 1.0, None, {}, {})
    monkeypatch.setattr(runner, "select_anchor", lambda *a, **k: (anchor, []))
    point, row, _, track = runner.calibrate_clip(clip, registration_mask="court_observation")
    assert row["frame_track"]["registration_mask"] == "court_observation"
    assert row["frame_track"]["samples"][1]["registration_feature_mask"] == "court_observation_v1"
    path = runner.write_frame_track(match_out, {point: track})
    result = runner.load_frame_track(path)["pt0001"][1]
    assert result[1:] == (True, "registered")
    # Assert exact serialization, and physical reprojection accuracy rather than matrix scale.
    np.testing.assert_array_equal(result[0], track["H"][1])
    world = np.float32(
        [[[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]]
    )
    expected = cv2.perspectiveTransform(world, motion @ np.linalg.inv(H))
    actual = cv2.perspectiveTransform(world, np.linalg.inv(result[0]))
    assert np.max(np.linalg.norm(actual - expected, axis=2)) < 2.0

    # Exercise the ordinary H -> P expansion and actual event camera reader. A new
    # feature-mask receipt must not create an unknown registration lineage downstream.
    ground_to_image = np.linalg.inv(H)
    projection = np.column_stack((ground_to_image[:, :2], [0.0, -20.0, 0.0], ground_to_image[:, 2]))
    np.savez_compressed(
        match_out / "camera_P_per_point.npz",
        pts=[1],
        P=[projection],
        reliable=[True],
        source=["direct"],
        ground_residual_px=[0.2],
        net_residual_px=[1.5],
        confidence=[0.8],
        fallback_ancestry=["[]"],
        frame_scope=["point_static"],
        reference_frame=["f_0000.jpg"],
    )
    expanded = expand_point_cameras(match_out, "frames")
    with np.load(expanded) as cameras:
        assert cameras["source"].tolist() == ["direct+registered", "direct+registered"]
        assert cameras["reliable"].tolist() == [True, True]
    (match_out / "audit_frames_native_1080.coordinates.json").write_text(
        json.dumps(
            {
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 1920, "height": 1080},
            }
        )
    )
    consumed = GroundSnapshot.load(tmp_path, ["m"], track=TrackSnapshot({}, [], []))
    assert consumed.cameras[("m__pt0001", 1)]["reliable"]
    assert consumed.cameras[("m__pt0001", 1)]["source"] == "direct+registered"
