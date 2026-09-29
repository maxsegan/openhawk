from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline.player_motion_physical import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    HEIGHT_FRACTION,
    HeightHypothesis,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    align_body_prior,
    align_body_prior_transform,
    automatic_foot_contact_witness,
    body_prior_projection_reliability,
    camera_ray,
    height_hypotheses,
    lock_player_dimensions,
    interpolate_track_box,
    json_safe,
    load_body_priors,
    load_cadence_fps,
    load_dimension_locks,
    load_track_boxes,
    motion_training_target_quality,
    physical_pose_acceptable,
    pose_track_alignment,
    process_match,
    project,
    racket_hypotheses,
    root_intersection_parameter,
    adjacent_root_speeds,
    residual_jacobian_sparsity,
    solve_frame,
)


def test_motion_training_target_gate_uses_native_time_and_pose_evidence() -> None:
    rows = [
        {
            "frame": f"f_{frame:04d}.jpg",
            "court_x": 0.1 * frame,
            "court_y": 2.0,
        }
        for frame in range(3)
    ]
    speeds_25 = adjacent_root_speeds(rows, 25.0)
    speeds_5994 = adjacent_root_speeds(rows, 60000 / 1001)
    assert speeds_25[1] == pytest.approx(2.5)
    assert speeds_5994[1] == pytest.approx(5.994005994)

    raw = np.zeros((17, 2), dtype=float)
    corrected = raw.copy()
    quality = motion_training_target_quality(
        {"x0": 0, "y0": 0, "x1": 64, "y1": 64},
        raw,
        corrected,
        {"minimum_z_m": 0.02},
        fps=60000 / 1001,
        maximum_adjacent_root_speed_mps=speeds_5994[1],
    )
    assert quality["decision"] == "accept"
    assert quality["evidence"]["native_fps"] == pytest.approx(60000 / 1001)


@pytest.mark.parametrize(
    ("row", "correction", "minimum_z", "root_speed", "fps", "reason"),
    [
        (
            {"x0": 0, "y0": 0, "x1": 63, "y1": 64},
            0.0,
            0.0,
            0.0,
            25.0,
            "insufficient_native_pose_pixels",
        ),
        (
            {"x0": 0, "y0": 0, "x1": 64, "y1": 64},
            2.0,
            0.0,
            0.0,
            25.0,
            "temporal_pose_correction_excessive",
        ),
        (
            {"x0": 0, "y0": 0, "x1": 64, "y1": 64},
            0.0,
            0.0,
            31.0,
            25.0,
            "impossible_player_root_speed",
        ),
        ({"x0": 0, "y0": 0, "x1": 64, "y1": 64}, 0.0, 0.36, 0.0, 25.0, "ungrounded_pose_target"),
        ({"x0": 0, "y0": 0, "x1": 64, "y1": 64}, 0.0, 0.0, 0.0, 0.0, "native_cadence_unavailable"),
    ],
)
def test_motion_training_target_gate_rejects_impure_targets(
    row: dict,
    correction: float,
    minimum_z: float,
    root_speed: float,
    fps: float,
    reason: str,
) -> None:
    raw = np.zeros((17, 2), dtype=float)
    corrected = raw.copy()
    corrected[0, 0] = correction
    quality = motion_training_target_quality(
        row,
        raw,
        corrected,
        {"minimum_z_m": minimum_z},
        fps=fps,
        maximum_adjacent_root_speed_mps=root_speed,
    )
    assert quality["decision"] == "hold"
    assert reason in quality["reasons"]


def test_foot_contact_witness_detects_brief_two_foot_jump_at_native_cadence() -> None:
    rows = []
    for frame in range(75):
        jump = 40.0 * max(0.0, 1.0 - abs(frame - 37) / 8.0)
        row = {
            "frame": f"f_{frame:04d}.jpg",
            "x0": "100",
            "y0": "100",
            "x1": "300",
            "y1": "500",
        }
        for name in (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):
            row[f"{name}_x"] = 200.0
            row[f"{name}_y"] = 480.0 - jump if "ankle" in name else 300.0
            row[f"{name}_confidence"] = 0.9
        rows.append(row)

    witness_25 = automatic_foot_contact_witness(rows, 25.0)
    witness_50 = automatic_foot_contact_witness(rows, 50.0)

    assert witness_25[37]["state"] == "airborne"
    assert witness_50[37]["state"] == "airborne"
    assert witness_25[5]["state"] == "grounded"


def test_foot_contact_witness_uses_projected_court_ground_during_player_motion() -> None:
    rows = []
    cameras = {}
    projection = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    for frame in range(50):
        court_y = 200.0 + 3.0 * frame
        jump = 35.0 * max(0.0, 1.0 - abs(frame - 25) / 6.0)
        row = {
            "clip": "pt0001",
            "frame": f"f_{frame:04d}.jpg",
            "x0": "100",
            "y0": str(court_y - 400.0),
            "x1": "300",
            "y1": str(court_y),
            "court_x": "200",
            "court_y": str(court_y),
        }
        for name in (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):
            row[f"{name}_x"] = 200.0
            row[f"{name}_y"] = court_y - jump if "ankle" in name else court_y - 200.0
            row[f"{name}_confidence"] = 0.9
        rows.append(row)
        cameras[("pt0001", frame)] = projection

    witness = automatic_foot_contact_witness(rows, 25.0, cameras)

    assert witness[25]["state"] == "airborne"
    assert witness[25]["source"] == "court_projected_native_pose_foot_motion_v2"
    assert witness[5]["state"] == "grounded"


def test_foot_contact_witness_ignores_single_foot_stride() -> None:
    rows = []
    for frame in range(50):
        row = {
            "frame": f"f_{frame:04d}.jpg",
            "x0": "100",
            "y0": "100",
            "x1": "300",
            "y1": "500",
        }
        for name in (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ):
            row[f"{name}_x"] = 200.0
            row[f"{name}_y"] = 460.0 if name == "left_ankle" and 20 <= frame <= 28 else 480.0
            row[f"{name}_confidence"] = 0.9
        rows.append(row)

    witness = automatic_foot_contact_witness(rows, 25.0)

    assert all(value["state"] != "airborne" for value in witness.values())


def test_cadence_loader_scopes_repeated_clip_ids_by_match(tmp_path: Path) -> None:
    path = tmp_path / "cadence.json"
    path.write_text(
        json.dumps(
            {
                "schema": "frame_cadence_audit_v1",
                "rows": [
                    {
                        "match_id": "match_a",
                        "clip": "pt0001",
                        "fps": 25.0,
                        "decision": "timing_usable",
                    },
                    {
                        "match_id": "match_b",
                        "clip": "pt0001",
                        "fps": 59.94,
                        "decision": "timing_usable",
                    },
                ],
            }
        )
    )

    assert load_cadence_fps(path, match_id="match_a") == {"pt0001": 25.0}
    assert load_cadence_fps(path, match_id="match_b") == {"pt0001": 59.94}


def test_pose_track_alignment_rejects_disjoint_detection() -> None:
    alignment = pose_track_alignment(
        {
            "x0": "100",
            "y0": "20",
            "x1": "140",
            "y1": "100",
            "track_x0": "10",
            "track_y0": "200",
            "track_x1": "60",
            "track_y1": "300",
        }
    )
    assert alignment is not None
    assert alignment["iou"] == 0.0
    assert not physical_pose_acceptable(
        {
            "quality": 1.0,
            "bone_rms_m": 0.0,
            "root_error_m": 0.0,
            "minimum_z_m": 0.0,
            "maximum_z_m": 1.8,
        },
        1.8,
        track_alignment=alignment,
    )


def test_physical_pose_gate_requires_strong_source_track_overlap() -> None:
    diagnostics = {
        "quality": 1.0,
        "bone_rms_m": 0.0,
        "root_error_m": 0.0,
        "minimum_z_m": 0.0,
        "maximum_z_m": 1.8,
    }

    assert not physical_pose_acceptable(
        diagnostics,
        1.8,
        track_alignment={"iou": 0.54, "center_distance_track_diagonals": 0.0},
    )
    assert physical_pose_acceptable(
        diagnostics,
        1.8,
        track_alignment={"iou": 0.55, "center_distance_track_diagonals": 0.0},
    )


def test_root_intersection_uses_stable_ground_plane_projection() -> None:
    center = np.asarray([2.0, -20.0, 8.0])
    direction = np.asarray([1e-9, 0.8, -0.6])
    direction /= np.linalg.norm(direction)
    root_xy = np.asarray([2.0, 4.0])

    parameter = root_intersection_parameter(center, direction, root_xy)
    closest = center[:2] + parameter * direction[:2]

    assert np.linalg.norm(closest - root_xy) < 1e-6
    assert parameter < 100.0


def projection() -> np.ndarray:
    return np.asarray(
        [
            [900.0, 0.0, 480.0, 0.0],
            [0.0, -180.0, 860.0, 1000.0],
            [0.0, 0.05, 1.0, 1.0],
        ]
    )


def skeleton() -> np.ndarray:
    joints = np.zeros((17, 3), dtype=float)
    joints[:, :2] = [5.4, 3.0]
    joints[0] = [5.4, 3.0, 1.72]
    joints[LEFT_SHOULDER] = [5.20, 3.0, 1.47]
    joints[RIGHT_SHOULDER] = [5.60, 3.0, 1.47]
    joints[LEFT_ELBOW] = [4.98, 3.1, 1.25]
    joints[RIGHT_ELBOW] = [5.82, 2.9, 1.25]
    joints[LEFT_WRIST] = [4.78, 3.2, 1.04]
    joints[RIGHT_WRIST] = [6.02, 2.8, 1.04]
    joints[LEFT_HIP] = [5.25, 3.0, 0.95]
    joints[RIGHT_HIP] = [5.55, 3.0, 0.95]
    joints[LEFT_KNEE] = [5.24, 3.0, 0.50]
    joints[RIGHT_KNEE] = [5.56, 3.0, 0.50]
    joints[LEFT_ANKLE] = [5.25, 3.0, 0.03]
    joints[RIGHT_ANKLE] = [5.55, 3.0, 0.03]
    return joints


def test_camera_ray_reprojects_pixel() -> None:
    pixel = np.asarray([850.0, 420.0])
    center, direction = camera_ray(projection(), pixel)
    points = np.stack((center + 2.0 * direction, center + 9.0 * direction))
    assert np.allclose(project(projection(), points), pixel, atol=1e-6)


def test_physical_solver_preserves_rays_and_ground() -> None:
    truth = skeleton()
    pixels = project(projection(), truth)
    result = solve_frame(
        projection(),
        pixels,
        np.ones(17),
        np.asarray([5.4, 3.0]),
        1.80,
    )
    fitted = result["joints"]
    assert np.max(np.linalg.norm(project(projection(), fitted) - pixels, axis=1)) < 1e-5
    assert result["diagnostics"]["minimum_z_m"] >= -0.05
    assert result["diagnostics"]["root_error_m"] < 0.15
    assert result["diagnostics"]["bone_rms_m"] < 0.12


def test_bad_temporal_prior_cannot_move_solution_far_from_court_ray() -> None:
    truth = skeleton()
    pixels = project(projection(), truth)
    bad_previous = truth + np.asarray([0.0, 30.0, 40.0])
    result = solve_frame(
        projection(),
        pixels,
        np.ones(17),
        np.asarray([5.4, 3.0]),
        1.80,
        previous=bad_previous,
    )
    assert result["diagnostics"]["maximum_z_m"] < 3.0
    assert result["diagnostics"]["minimum_z_m"] > -0.5


def test_racket_branches_cover_both_hands_and_grips() -> None:
    branches = racket_hypotheses(skeleton(), handedness="right")
    assert len(branches) == 10
    assert {row["hand"] for row in branches} == {"left", "right"}
    assert {row["grip"] for row in branches} >= {"eastern", "slice_open", "serve"}
    right = [row for row in branches if row["hand"] == "right"]
    left = [row for row in branches if row["hand"] == "left"]
    assert min(row["prior"] for row in right) > max(row["prior"] for row in left)


def test_json_safe_preserves_unknown_uncertainty_as_null() -> None:
    assert json_safe({"values": np.asarray([1.0, np.nan])}) == {"values": [1.0, None]}


def test_sparse_jacobian_matches_residual_shape() -> None:
    sparsity = residual_jacobian_sparsity(17, np.zeros((17, 3)), np.zeros((17, 3)))
    assert sparsity.shape == (12 + 4 + 1 + len(HEIGHT_FRACTION) + 34 + 51 + 51, 17)


def test_body_prior_alignment_recovers_scale_and_orientation() -> None:
    target = skeleton()
    source = (target - target[[LEFT_HIP, RIGHT_HIP]].mean(axis=0))[:, [2, 0, 1]] * 2.5
    aligned = align_body_prior(source, target, 1.80)
    assert aligned is not None
    assert (
        np.linalg.norm(
            aligned[[LEFT_HIP, RIGHT_HIP]].mean(axis=0) - target[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
        )
        < 1e-6
    )
    assert np.linalg.norm(aligned[0] - aligned[[LEFT_ANKLE, RIGHT_ANKLE]].mean(axis=0)) > 1.4


def test_body_prior_alignment_returns_exact_direction_rotation() -> None:
    target = skeleton()
    source = (target - target[[LEFT_HIP, RIGHT_HIP]].mean(axis=0))[:, [2, 0, 1]] * 2.5
    result = align_body_prior_transform(source, target, 1.80)
    assert result is not None
    aligned, rotation = result
    source_direction = source[RIGHT_WRIST] - source[RIGHT_ELBOW]
    aligned_direction = aligned[RIGHT_WRIST] - aligned[RIGHT_ELBOW]
    expected = rotation @ source_direction
    assert np.allclose(
        expected / np.linalg.norm(expected),
        aligned_direction / np.linalg.norm(aligned_direction),
        atol=1e-8,
    )


def test_body_prior_rejects_human_derived_input(tmp_path) -> None:
    path = tmp_path / "prior.jsonl"
    path.write_text(
        '{"schema":"player_body_prior_coco17_v1","automatic_only":true,'
        '"human_derived_inputs":["review"],"clip":"pt0001","side":"near",'
        '"frame":1,"joints_xyz":[]}\n'
    )

    with pytest.raises(ValueError, match="not automatic and label-free"):
        load_body_priors(path)


def test_body_prior_requires_match_scope_for_aggregate_artifact(tmp_path) -> None:
    path = tmp_path / "prior.jsonl"
    rows = []
    for match_id in ("match_a", "match_b"):
        rows.append(
            {
                "schema": "player_body_prior_coco17_v1",
                "automatic_only": True,
                "human_derived_inputs": [],
                "match_id": match_id,
                "backend": "test",
                "clip": "pt0001",
                "side": "near",
                "frame": 1,
                "joints_xyz": skeleton().tolist(),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="spans multiple matches"):
        load_body_priors(path)

    selected = load_body_priors(path, match_id="match_b")
    assert selected[("pt0001", "near", 1)][0]["match_id"] == "match_b"


def test_body_prior_projection_reliability_rewards_native_overlap() -> None:
    points = project(projection(), skeleton())
    names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    rows = []
    for frame in range(1, 5):
        row = {
            "clip": "pt0001",
            "side": "near",
            "frame": f"f_{frame:04d}.jpg",
            "x0": "100",
            "y0": "100",
            "x1": "500",
            "y1": "900",
        }
        for index, name in enumerate(names):
            row[f"{name}_x"] = points[index, 0]
            row[f"{name}_y"] = points[index, 1]
            row[f"{name}_confidence"] = 0.9
        rows.append(row)
    priors = {
        ("pt0001", "near", frame): [
            {"backend": "test_temporal_smpl", "keypoints_xy_native": points.tolist()}
        ]
        for frame in range(1, 5)
    }

    result = body_prior_projection_reliability({("pt0001", "near"): rows}, priors)

    key = ("pt0001", "near", "test_temporal_smpl")
    assert result[key]["overlap_frames"] == 4
    assert result[key]["reliability"] == pytest.approx(0.85)


def test_track_box_interpolation_requires_short_interior_gap() -> None:
    boxes = {
        ("pt0001", "near", 10): {
            "track_x0": 100.0,
            "track_y0": 200.0,
            "track_x1": 300.0,
            "track_y1": 600.0,
            "court_x": 4.0,
            "court_y": 2.0,
            "conf": 0.9,
        },
        ("pt0001", "near", 12): {
            "track_x0": 120.0,
            "track_y0": 220.0,
            "track_x1": 320.0,
            "track_y1": 620.0,
            "court_x": 4.2,
            "court_y": 2.4,
            "conf": 0.8,
        },
    }

    result = interpolate_track_box(boxes, ("pt0001", "near", 11))

    assert result is not None
    assert result["track_x0"] == 110.0
    assert result["court_y"] == pytest.approx(2.2)
    assert result["track_source_frames"] == [10, 12]
    assert interpolate_track_box(boxes, ("pt0001", "near", 9)) is None


def test_temporal_body_prior_fills_only_real_tracked_frame_gap(tmp_path) -> None:
    points = project(projection(), skeleton())
    names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    pose_path = tmp_path / "pose.csv"
    fields = ["clip", "side", "frame", "x0", "y0", "x1", "y1", "conf", "court_x", "court_y"]
    fields += [
        value for name in names for value in (f"{name}_x", f"{name}_y", f"{name}_confidence")
    ]
    with pose_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(1, 4):
            row = {
                "clip": "pt0001",
                "side": "near",
                "frame": f"f_{frame:04d}.jpg",
                "x0": 100,
                "y0": 100,
                "x1": 500,
                "y1": 900,
                "conf": 0.9,
                "court_x": 5.4,
                "court_y": 3.0,
            }
            for index, name in enumerate(names):
                row[f"{name}_x"] = points[index, 0]
                row[f"{name}_y"] = points[index, 1]
                row[f"{name}_confidence"] = 0.9
            writer.writerow(row)
    boxes_path = tmp_path / "boxes.csv"
    with boxes_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip",
                "side",
                "frame",
                "x0",
                "y0",
                "x1",
                "y1",
                "court_x",
                "court_y",
                "conf",
            ),
        )
        writer.writeheader()
        for frame in range(1, 5):
            writer.writerow(
                {
                    "clip": "pt0001",
                    "side": "near",
                    "frame": f"f_{frame:04d}.jpg",
                    "x0": 50,
                    "y0": 50,
                    "x1": 250,
                    "y1": 450,
                    "court_x": 5.4,
                    "court_y": 3.0,
                    "conf": 0.9,
                }
            )
    Path(f"{boxes_path}.coordinates.json").write_text(
        json.dumps(
            {
                "artifact_size": {"width": 960, "height": 540},
                "image_size": {"width": 1920, "height": 1080},
            }
        )
    )
    prior_path = tmp_path / "prior.jsonl"
    with prior_path.open("w") as handle:
        for frame in range(1, 5):
            handle.write(
                json.dumps(
                    {
                        "schema": "player_body_prior_coco17_v1",
                        "automatic_only": True,
                        "human_derived_inputs": [],
                        "backend": "test_temporal_smpl",
                        "clip": "pt0001",
                        "side": "near",
                        "frame": frame,
                        "joints_xyz": skeleton().tolist(),
                        "keypoints_xy_native": points.tolist(),
                        "keypoint_confidence": [0.35] * 17,
                        "keypoint_confidence_semantics": "uncalibrated_model_projection",
                    }
                )
                + "\n"
            )
    camera_path = tmp_path / "camera.npz"
    np.savez(
        camera_path,
        clips=np.asarray(["pt0001"] * 4),
        frames=np.arange(1, 5),
        P=np.stack([projection()] * 4),
        confidence=np.ones(4),
        reliable=np.ones(4, dtype=bool),
        source=np.asarray(["direct"] * 4),
        ground_residual_px=np.zeros(4),
    )
    biometrics_path = tmp_path / "biometrics.json"
    biometrics_path.write_text("{}\n")
    output_path = tmp_path / "physical.jsonl"

    report = process_match(
        "unknown_match",
        pose_path,
        camera_path,
        biometrics_path,
        output_path,
        body_prior_path=prior_path,
        boxes_path=boxes_path,
    )
    documents = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert report["rows"] == 4
    assert [document["frame"] for document in documents] == [1, 2, 3, 4]
    assert documents[-1]["body_prior_projection_quality"][0]["overlap_frames"] == 3
    assert documents[-1]["body_prior_projection_quality"][0]["reliability"] == pytest.approx(0.85)
    assert list(tmp_path.glob(".physical.jsonl.tmp-*")) == []


def test_track_boxes_use_declared_coordinate_scale(tmp_path) -> None:
    path = tmp_path / "boxes.csv"
    path.write_text(
        "clip,frame,side,x0,y0,x1,y1,court_x,court_y,conf\n"
        "pt0001,f_0010.jpg,near,100,50,500,400,4,2,0.9\n"
    )
    Path(f"{path}.coordinates.json").write_text(
        json.dumps(
            {
                "artifact_size": {"width": 1280, "height": 720},
                "image_size": {"width": 1920, "height": 1080},
            }
        )
    )

    rows = load_track_boxes(path)

    assert rows[("pt0001", "near", 10)]["track_x1"] == 750.0
    assert rows[("pt0001", "near", 10)]["track_y1"] == 600.0


def test_track_boxes_prefer_explicit_track_columns(tmp_path) -> None:
    path = tmp_path / "pose_with_track.csv"
    path.write_text(
        "clip,frame,side,x0,y0,x1,y1,track_x0,track_y0,track_x1,track_y1,"
        "court_x,court_y,conf\n"
        "pt0001,f_0010.jpg,near,100,50,500,400,120,60,520,420,4,2,0.9\n"
    )
    Path(f"{path}.coordinates.json").write_text(
        json.dumps(
            {
                "artifact_size": {"width": 1920, "height": 1080},
                "image_size": {"width": 1920, "height": 1080},
            }
        )
    )

    rows = load_track_boxes(path)

    assert rows[("pt0001", "near", 10)]["track_x0"] == 120.0
    assert rows[("pt0001", "near", 10)]["track_y1"] == 420.0


def test_transfer_biometrics_have_official_height_branches() -> None:
    biometrics = json.loads(Path(__file__).with_name("player_biometrics.json").read_text())
    matches = (
        "atpf2023sf_m_djokovic_alcaraz",
        "barcelona2019sf_m_thiem_nadal",
        "rg2025qf_w_boisson_andreeva",
        "wim2025r32_w_sabalenka_raducanu",
        "wta_2024_520_r64_165_iga_swiatek_naomi_osaka",
        "wuhan2024f_w_sabalenka_zheng",
    )

    for match_id in matches:
        players = biometrics["matches"][match_id]
        assert len(players) == 2
        for player_id in players:
            player = biometrics["players"][player_id]
            assert 1.65 <= player["height_m"] <= 2.05
            assert player["source"].startswith(
                ("https://www.atptour.com/", "https://www.wtatennis.com/")
            )


def test_body_profile_is_derived_from_official_player_registry() -> None:
    biometrics = json.loads(Path(__file__).with_name("player_biometrics.json").read_text())

    women = height_hypotheses("ao2023f_w_sabalenka_rybakina", biometrics)
    men = height_hypotheses("atpf2023sf_m_djokovic_alcaraz", biometrics)

    assert {row.body_profile for row in women} == {"female_smpl"}
    assert {row.body_profile for row in men} == {"male_smpl"}


def test_player_dimensions_are_jointly_locked_per_clip_side() -> None:
    heights = [
        HeightHypothesis("player_a", 1.70, 0.5),
        HeightHypothesis("player_b", 1.90, 0.5),
    ]
    documents = []
    for side, preferred in (("near", "player_a"), ("far", "player_b")):
        for frame in range(4):
            documents.append(
                {
                    "clip": "pt0001",
                    "side": side,
                    "frame": frame,
                    "branches": [
                        {
                            "player_id": player.player_id,
                            "score": 0.9 if player.player_id == preferred else 0.2,
                        }
                        for player in heights
                    ],
                }
            )

    locks = lock_player_dimensions(documents, heights)

    assert locks[("pt0001", "near")]["player_id"] == "player_a"
    assert locks[("pt0001", "far")]["player_id"] == "player_b"
    assert locks[("pt0001", "near")]["assignment_safe"] is True
    assert locks[("pt0001", "near")]["identity_safe"] is True
    assert locks[("pt0001", "near")]["scope"] == "clip_side"


def test_near_equal_heights_are_dimension_safe_but_not_identity_safe() -> None:
    heights = [
        HeightHypothesis("player_a", 1.96, 0.5),
        HeightHypothesis("player_b", 1.98, 0.5),
    ]
    documents = [
        {
            "clip": "pt0001",
            "side": side,
            "frame": frame,
            "branches": [{"player_id": player.player_id, "score": 0.5} for player in heights],
        }
        for side in ("near", "far")
        for frame in range(4)
    ]

    locks = lock_player_dimensions(documents, heights)

    assert locks[("pt0001", "near")]["assignment_safe"] is True
    assert locks[("pt0001", "near")]["identity_safe"] is False


def test_dimension_locks_reject_human_or_wrong_match_inputs(tmp_path: Path) -> None:
    path = tmp_path / "dimensions.json"
    path.write_text(
        json.dumps(
            {
                "automatic_only": True,
                "human_derived_inputs": [],
                "match_id": "match_a",
                "clip_side_assignments": {
                    "pt0001/near": {
                        "player_id": "player_a",
                        "height_m": 1.8,
                        "assignment_safe": True,
                    }
                },
            }
        )
    )

    assert load_dimension_locks(path, "match_a")[("pt0001", "near")]["player_id"] == "player_a"
    with pytest.raises(ValueError, match="match mismatch"):
        load_dimension_locks(path, "match_b")

    document = json.loads(path.read_text())
    document["human_derived_inputs"] = ["reviewed identity"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="automatic-only"):
        load_dimension_locks(path, "match_a")
