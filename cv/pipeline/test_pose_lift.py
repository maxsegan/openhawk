from __future__ import annotations

import numpy as np

from cv.pipeline.pose_lift import (
    MAXIMUM_ROOT_STEP_M,
    PoseFrameInput,
    ServeWindow,
    discontinuous_root_frames,
    add_racket_segment,
    camera_ray,
    lift_pose_frame,
    lift_pose_sequence,
    project_point,
    wrist_body_plane_distance,
)


def test_discontinuous_root_frames_marks_both_ends_of_one_frame_jump() -> None:
    frames = [
        PoseFrameInput(10, {}, PROJECTION, (1.0, 1.0)),
        PoseFrameInput(11, {}, PROJECTION, (1.2, 1.1)),
        PoseFrameInput(12, {}, PROJECTION, (4.0, 1.1)),
        PoseFrameInput(14, {}, PROJECTION, (8.0, 1.1)),
    ]
    assert discontinuous_root_frames(frames) == {11: 2.8, 12: 2.8}


PROJECTION = np.asarray(
    [
        [1800.0, 0.0, 960.0, -9600.0],
        [0.0, 1800.0, 540.0, 7200.0],
        [0.0, 0.0, 1.0, 10.0],
    ]
)


def _row(points: dict[str, tuple[float, float, float]], root=(5.4, 1.0)) -> dict[str, str]:
    pixels = {name: project_point(PROJECTION, point) for name, point in points.items()}
    assert all(pixel is not None for pixel in pixels.values())
    ankle_y = max(pixel[1] for name, pixel in pixels.items() if name.endswith("ankle"))
    row = {
        "x0": "760",
        "y0": "250",
        "x1": "1160",
        "y1": str(ankle_y),
        "court_x": str(root[0]),
        "court_y": str(root[1]),
    }
    for name, pixel in pixels.items():
        row[f"{name}_x"] = str(pixel[0])
        row[f"{name}_y"] = str(pixel[1])
        row[f"{name}_confidence"] = "0.95"
    return row


def _pose() -> dict[str, tuple[float, float, float]]:
    # Right wrist reaches 0.55 m toward the camera from a torso plane at y=1.
    return {
        "nose": (5.4, 1.0, 1.72),
        "left_shoulder": (5.2, 1.0, 1.48),
        "right_shoulder": (5.6, 1.0, 1.48),
        "left_elbow": (4.9, 1.0, 1.32),
        "right_elbow": (5.75, 0.72, 1.30),
        "left_wrist": (4.75, 1.0, 1.08),
        "right_wrist": (5.82, 0.44, 1.12),
        "left_hip": (5.24, 1.0, 0.94),
        "right_hip": (5.56, 1.0, 0.94),
        "left_knee": (5.22, 1.0, 0.50),
        "right_knee": (5.58, 1.0, 0.50),
        "left_ankle": (5.22, 1.0, 0.0),
        "right_ankle": (5.58, 1.0, 0.0),
    }


def test_camera_ray_reprojects_exact_pixel() -> None:
    pixel = (1100.0, 600.0)
    centre, direction = camera_ray(PROJECTION, pixel, (5.0, 1.0, 1.0))
    point = centre + 20.0 * direction
    projected = project_point(PROJECTION, point)
    assert projected is not None
    assert np.allclose(projected, pixel, atol=1e-8)


def test_lift_is_grounded_metric_and_not_a_billboard() -> None:
    result = lift_pose_frame(_row(_pose()), PROJECTION, (5.4, 1.0), 1.80)
    assert result is not None
    assert result.support == "planted"
    assert result.joints["left_ankle"][2] == 0.0
    assert result.joints["right_ankle"][2] == 0.0
    for name, truth in _pose().items():
        projected = project_point(PROJECTION, result.joints[name][:3])
        expected = project_point(PROJECTION, truth)
        assert projected is not None and expected is not None
        # Ground snap is allowed only after a sub-3 cm ray residual; the visible
        # pixel remains effectively exact at native scale.
        assert np.linalg.norm(np.asarray(projected) - expected) < 2.0
    distance = wrist_body_plane_distance(result.joints, "right_wrist")
    assert distance is not None and distance > 0.18


def test_racket_uses_accepted_contact_or_conservative_extrapolation() -> None:
    result = lift_pose_frame(_row(_pose()), PROJECTION, (5.4, 1.0), 1.80)
    assert result is not None
    contact = (5.9, 0.1, 1.25)
    accepted, accepted_receipt = add_racket_segment(result.joints, contact_xyz=contact)
    assert np.allclose(accepted["racket_head"][:3], contact)
    assert accepted_receipt["source"] == "accepted_fitter_contact_xyz"
    assert accepted_receipt["sigma_px"] is None

    extrapolated, extrapolated_receipt = add_racket_segment(result.joints)
    assert extrapolated_receipt["source"] == "wrist_forearm_3d_extrapolation"
    assert extrapolated_receipt["sigma_px"] == 75.0
    grip = np.asarray(extrapolated["racket_grip"][:3])
    head = np.asarray(extrapolated["racket_head"][:3])
    wrist = np.asarray(result.joints[f"{extrapolated_receipt['hand']}_wrist"][:3])
    elbow = np.asarray(result.joints[f"{extrapolated_receipt['hand']}_elbow"][:3])
    expected = min(0.55, 1.5 * np.linalg.norm(wrist - elbow))
    assert abs(np.linalg.norm(head - grip) - expected) < 1e-8


def test_temporal_prior_does_not_flatten_reaching_wrist() -> None:
    first = lift_pose_frame(_row(_pose()), PROJECTION, (5.4, 1.0), 1.80)
    assert first is not None
    previous = {
        name: (point[0] - 5.4, point[1] - 1.0, point[2], point[3])
        for name, point in first.joints.items()
    }
    moved = dict(_pose())
    moved["right_elbow"] = (5.76, 0.62, 1.31)
    moved["right_wrist"] = (5.86, 0.24, 1.16)
    second = lift_pose_frame(_row(moved), PROJECTION, (5.4, 1.0), 1.80, previous_local=previous)
    assert second is not None
    distance = wrist_body_plane_distance(second.joints, "right_wrist")
    assert distance is not None and distance > 0.20


def test_serve_sequence_anchors_airborne_root_between_planted_frames() -> None:
    frames = []
    raw_roots = [(5.4, 1.0), (5.4, 1.02), (5.4, 2.0), (5.4, 2.4), (5.4, 1.9), (5.4, 1.08)]
    for frame, root in enumerate(raw_roots, start=10):
        row = _row(_pose(), root=root)
        if 12 <= frame <= 14:
            row["y1"] = str(float(row["y1"]) + 30.0)
        frames.append(PoseFrameInput(frame, row, PROJECTION, root))
    result = lift_pose_sequence(
        frames,
        1.80,
        serve_window=ServeWindow(10, 13.0, 15, (5.5, 1.2, 2.45), 1.0, "test"),
    )
    assert set(result) == set(range(10, 16))
    roots = [result[frame].root_xy for frame in sorted(result)]
    assert all(root is not None for root in roots)
    steps = [
        np.linalg.norm(np.asarray(right) - np.asarray(left))
        for left, right in zip(roots, roots[1:])
    ]
    assert max(steps) <= MAXIMUM_ROOT_STEP_M + 1e-9
    assert result[13].root_xy is not None
    assert result[13].root_xy[1] < 1.2


def test_serve_contact_ik_reaches_ball_with_physical_arm_and_racket() -> None:
    frames = [PoseFrameInput(20, _row(_pose()), PROJECTION, (5.4, 1.0))]
    contact = (5.62, 0.96, 1.70)
    result = lift_pose_sequence(
        frames,
        1.80,
        serve_window=ServeWindow(19, 20.0, 21, contact, 1.0, "accepted_fit"),
    )[20]
    unanchored = lift_pose_sequence(
        frames,
        1.80,
        serve_window=ServeWindow(19, 20.0, 21, None, 1.0, "test"),
    )[20]
    assert result.striking_hand in {"left", "right"}
    hand = result.striking_hand
    assert hand is not None
    shoulder = np.asarray(result.joints[f"{hand}_shoulder"][:3])
    elbow = np.asarray(result.joints[f"{hand}_elbow"][:3])
    wrist = np.asarray(result.joints[f"{hand}_wrist"][:3])
    head = np.asarray(result.joints["racket_head"][:3])
    assert np.allclose(head, contact)
    assert abs(np.linalg.norm(shoulder - elbow) - 0.186 * 1.80) < 1e-6
    assert abs(np.linalg.norm(elbow - wrist) - 0.146 * 1.80) < 1e-6
    assert 0.30 <= np.linalg.norm(head - wrist) <= 0.55
    assert abs(shoulder[1] - unanchored.joints[f"{hand}_shoulder"][1]) < 1e-9
    assert result.contact_anchor_error_m == 0.0


def test_serve_contact_torso_has_small_forward_lean_on_both_ends() -> None:
    for forward in (-1.0, 1.0):
        result = lift_pose_sequence(
            [PoseFrameInput(20, _row(_pose()), PROJECTION, (5.4, 1.0))],
            1.80,
            serve_window=ServeWindow(19, 20.49, 21, (5.62, 0.96, 1.70), forward, "test"),
        )[20]
        shoulders = np.mean(
            [result.joints[name][:3] for name in ("left_shoulder", "right_shoulder")], axis=0
        )
        hips = np.mean([result.joints[name][:3] for name in ("left_hip", "right_hip")], axis=0)
        signed_offset = forward * (shoulders[1] - hips[1])
        assert 0.02 < signed_offset < 0.09
