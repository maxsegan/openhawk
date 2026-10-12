from __future__ import annotations

import numpy as np
import pytest

from cv.pipeline import contact_striker as striker

FPS = 25.0
PLAYER_HEIGHT_M = 1.8


def _camera() -> tuple[np.ndarray, np.ndarray]:
    """A broadcast-like end camera, with its court-plane homography."""
    eye = np.array([5.485, -14.0, 11.0])
    target = np.array([5.485, 11.885, 0.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    intrinsics = np.array([[1600.0, 0.0, 960.0], [0.0, 1600.0, 540.0], [0.0, 0.0, 1.0]])
    projection = intrinsics @ np.hstack([rotation, (-rotation @ eye).reshape(3, 1)])
    plane = projection[:, [0, 1, 3]]
    return projection, np.linalg.inv(plane)


PROJECTION, HOMOGRAPHY = _camera()


def _image(point: np.ndarray) -> np.ndarray:
    homogeneous = PROJECTION @ np.array([point[0], point[1], point[2], 1.0])
    return homogeneous[:2] / homogeneous[2]


def _person(court: tuple[float, float], height_m: float = PLAYER_HEIGHT_M) -> dict:
    """A native-coordinate person box for someone standing at ``court``."""
    feet = _image(np.array([court[0], court[1], 0.0]))
    head = _image(np.array([court[0], court[1], height_m]))
    half_width = 0.22 * abs(feet[1] - head[1])
    return {
        "x0_native": round(float(feet[0] - half_width), 2),
        "y0_native": round(float(head[1]), 2),
        "x1_native": round(float(feet[0] + half_width), 2),
        "y1_native": round(float(feet[1]), 2),
        "conf": "0.9",
    }


def _half_native(row: dict) -> dict:
    return {
        "x0": row["x0_native"] / 2,
        "y0": row["y0_native"] / 2,
        "x1": row["x1_native"] / 2,
        "y1": row["y1_native"] / 2,
        "conf": row["conf"],
    }


def _near_wrist_pixel(court: tuple[float, float], reach_m: float = 0.0) -> tuple[float, float]:
    point = _image(np.array([court[0] + reach_m, court[1], 1.6]))
    return float(point[0]), float(point[1])


def _track(near: tuple[float, float], far: tuple[float, float], frames) -> dict:
    return {
        "near": {frame: np.array(near) for frame in frames},
        "far": {frame: np.array(far) for frame in frames},
    }


def _resolve(contacts, detections, track, **kwargs):
    return striker.resolve_point_strikers(
        contacts,
        detections,
        homography_for_frame=lambda _frame: HOMOGRAPHY,
        projection_for_frame=lambda _frame: PROJECTION,
        track_by_end=track,
        fps=FPS,
        **kwargs,
    )


def test_a_net_volley_stays_with_the_near_player_it_belongs_to():
    """A body a metre past the net line is still the near player's body."""
    volleyer = (6.0, 13.2)
    receiver = (5.0, 22.0)
    contact = _near_wrist_pixel(volleyer, reach_m=0.4)
    detections = {frame: [_person(volleyer), _person(receiver)] for frame in (99, 100, 101)}
    track = _track(near=(6.0, 10.5), far=(5.0, 22.0), frames=(90, 95, 105, 110))

    (resolved,) = _resolve(
        [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}],
        detections,
        track,
    )

    assert resolved["court_y"] > striker.NET_Y  # the body really is past the net line
    assert resolved["end"] == "near"
    assert resolved["end_source"] == "point_track"
    assert resolved["status"] == "resolved"


def test_the_court_side_sign_is_used_only_when_no_track_can_claim_the_body():
    volleyer = (6.0, 13.2)
    contact = _near_wrist_pixel(volleyer, reach_m=0.4)
    detections = {100: [_person(volleyer)]}

    (resolved,) = _resolve(
        [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}],
        detections,
        {},
    )

    assert resolved["end_source"] == "court_side_fallback"
    assert resolved["end"] == "far"


def test_a_declared_half_native_box_artifact_resolves_the_same_striker():
    volleyer = (6.0, 13.2)
    receiver = (5.0, 22.0)
    contact = _near_wrist_pixel(volleyer, reach_m=0.4)
    track = _track(near=(6.0, 10.5), far=(5.0, 22.0), frames=(90, 95, 105, 110))
    request = [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}]

    native = _resolve(
        request,
        {100: [_person(volleyer), _person(receiver)]},
        track,
    )
    half = _resolve(
        request,
        {100: [_half_native(_person(volleyer)), _half_native(_person(receiver))]},
        track,
        box_scale=2.0,
    )

    assert native[0]["end"] == half[0]["end"] == "near"
    assert native[0]["box_native"] == pytest.approx(half[0]["box_native"], abs=0.02)


def test_reading_a_half_native_box_as_native_is_what_hands_the_contact_to_the_wrong_end():
    """The defect the coordinate contract removes, stated as a test."""
    volleyer = (6.0, 13.2)
    receiver = (5.0, 22.0)
    contact = _near_wrist_pixel(volleyer, reach_m=0.4)
    track = _track(near=(6.0, 10.5), far=(5.0, 22.0), frames=(90, 95, 105, 110))
    request = [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}]
    mis_scaled = [_half_native(_person(volleyer)), _half_native(_person(receiver))]

    correct = _resolve(request, {100: [_person(volleyer), _person(receiver)]}, track)
    wrong = _resolve(request, {100: mis_scaled}, track, box_scale=1.0)

    assert correct[0]["end"] == "near"
    assert wrong[0]["end"] != "near"


def test_alternation_overturns_a_marginal_wrong_end():
    """Two bodies almost equally near the ball: the rally's alternation breaks the tie."""
    near_player = (5.4, 4.0)
    far_player = (5.6, 19.0)
    serve = _near_wrist_pixel(near_player, reach_m=0.3)
    # A contact pixel that sits between the two bodies, marginally nearer the near player's
    # box in its own units, so the per-contact argmin would repeat the serving end.
    ambiguous = _near_wrist_pixel(far_player, reach_m=0.30)
    contacts = [
        {"clip": "pt0001", "frame": 60.0, "image_x": serve[0], "image_y": serve[1]},
        {"clip": "pt0001", "frame": 90.0, "image_x": ambiguous[0], "image_y": ambiguous[1]},
    ]
    detections = {frame: [_person(near_player), _person(far_player)] for frame in range(58, 93)}
    track = _track(near=near_player, far=far_player, frames=range(40, 120, 5))

    resolved = _resolve(contacts, detections, track)

    assert [row["end"] for row in resolved] == ["near", "far"]


def test_strong_evidence_beats_the_alternation_prior():
    """A second contact plainly on one body is not moved by the prior."""
    near_player = (5.4, 4.0)
    far_player = (5.6, 19.0)
    first = _near_wrist_pixel(near_player, reach_m=0.2)
    second = _near_wrist_pixel(near_player, reach_m=0.2)
    contacts = [
        {"clip": "pt0001", "frame": 60.0, "image_x": first[0], "image_y": first[1]},
        {"clip": "pt0001", "frame": 90.0, "image_x": second[0], "image_y": second[1]},
    ]
    detections = {frame: [_person(near_player), _person(far_player)] for frame in range(58, 93)}
    track = _track(near=near_player, far=far_player, frames=range(40, 120, 5))

    resolved = _resolve(contacts, detections, track)

    assert [row["end"] for row in resolved] == ["near", "near"]
    assert resolved[1]["alternation_applied"] is False


def test_no_body_within_a_racket_of_the_pixel_abstains_and_stays_in_the_denominator():
    detections = {100: [_person((5.4, 4.0))]}
    contact = _image(np.array([0.5, 22.0, 1.0]))

    (resolved,) = _resolve(
        [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}],
        detections,
        {},
    )

    assert resolved["status"] == "abstain"
    assert resolved["end"] is None
    assert resolved["abstain_reason"] == "no_admissible_body"


def test_disagreement_with_the_sided_track_is_reported_not_silently_applied():
    volleyer = (6.0, 13.2)
    receiver = (5.0, 22.0)
    contact = _near_wrist_pixel(volleyer, reach_m=0.4)
    track = _track(near=(6.0, 10.5), far=(5.0, 22.0), frames=(90, 95, 105, 110))
    sided = {100: [{**_person(volleyer), "side": "far"}]}

    (resolved,) = _resolve(
        [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}],
        {100: [_person(volleyer), _person(receiver)]},
        track,
        sided_rows_by_frame=sided,
    )

    assert resolved["end"] == "near"
    assert resolved["sided_track_end"] == "far"
    assert resolved["disagrees_with_sided_track"] is True


def test_the_wrist_reach_comes_from_the_resolved_body_not_the_other_player():
    striker_court = (5.4, 4.0)
    other_court = (5.6, 19.0)
    contact = _near_wrist_pixel(striker_court, reach_m=0.3)
    wrist = _image(np.array([striker_court[0] + 0.2, striker_court[1], 1.6]))
    pose_row = {
        **_person(striker_court),
        "x0": _person(striker_court)["x0_native"],
        "y0": _person(striker_court)["y0_native"],
        "x1": _person(striker_court)["x1_native"],
        "y1": _person(striker_court)["y1_native"],
        "right_wrist_x": float(wrist[0]),
        "right_wrist_y": float(wrist[1]),
        "right_wrist_confidence": 0.9,
        "right_elbow_x": float(wrist[0] - 10.0),
        "right_elbow_y": float(wrist[1] + 10.0),
        "right_elbow_confidence": 0.9,
        "left_wrist_x": 0.0,
        "left_wrist_y": 0.0,
        "left_wrist_confidence": 0.0,
    }

    (resolved,) = _resolve(
        [{"clip": "pt0001", "frame": 100.0, "image_x": contact[0], "image_y": contact[1]}],
        {100: [_person(striker_court), _person(other_court)]},
        _track(near=striker_court, far=other_court, frames=range(80, 120, 5)),
        pose_by_frame={100: [pose_row]},
    )

    assert resolved["pose_matched"] is True
    assert resolved["wrist_keypoint"] == "right_wrist"
    assert resolved["wrist_px"] < resolved["root_px"]
    assert resolved["racket_face_source"] == "pose_elbow_wrist_extrapolation_v1"
    assert resolved["racket_face_sigma_px"] == striker.RACKET_FACE_SIGMA_PX


def test_decode_prefers_alternation_only_inside_the_stated_penalty():
    close = [{"near": 0.10, "far": 0.40}, {"near": 0.20, "far": 0.40}]
    clear = [{"near": 0.05, "far": 0.95}, {"near": 0.05, "far": 0.95}]

    assert striker.decode_ends(close) == ["near", "far"]
    assert striker.decode_ends(clear) == ["near", "near"]


def test_the_alternation_decode_is_global_and_may_move_the_earlier_contact():
    """Stated so a caller does not read a per-contact argmin into the sequence answer."""
    costs = [{"near": 0.10, "far": 0.40}, {"near": 0.10, "far": 0.90}]

    assert striker.decode_ends(costs) == ["far", "near"]


def _write_boxes(path, rows, fieldnames, artifact_size):
    import csv as _csv
    import json as _json

    from cv.pipeline import resolution as res

    with open(path, "w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "tennis.coordinate-space.v1",
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {
            "width": artifact_size.width,
            "height": artifact_size.height,
        },
        "source": "test",
    }
    if artifact_size != res.NATIVE_SIZE:
        payload["subnative_flagged"] = True
        payload["subnative_justification"] = "test fixture legacy mirror"
    res.coordinate_manifest_path(path).write_text(_json.dumps(payload))


def test_a_match_is_read_through_its_sidecars_and_answers_the_emitted_contacts(tmp_path):
    """End to end over a half-native box artifact and a half-native ball track."""
    import json as _json

    from cv.pipeline import resolution as res

    near, far = (5.4, 4.0), (5.6, 19.0)
    contact = _near_wrist_pixel(near, reach_m=0.2)
    match_dir = tmp_path / "match"
    match_dir.mkdir()

    def _row(court, frame, side=None):
        box = _person(court)
        row = {
            "clip": "pt0001",
            "frame": f"f_{frame:04d}.jpg",
            "x0": box["x0_native"] / 2,
            "y0": box["y0_native"] / 2,
            "x1": box["x1_native"] / 2,
            "y1": box["y1_native"] / 2,
            "conf": 0.9,
        }
        if side is not None:
            row.update(side=side, court_x=court[0], court_y=court[1], track_id=side)
        return row

    frames = range(95, 106)
    _write_boxes(
        match_dir / "player_boxes_25_native_v1.csv",
        [_row(court, frame) for frame in frames for court in (near, far)],
        ["clip", "frame", "x0", "y0", "x1", "y1", "conf"],
        res.LEGACY_TRACKING_SIZE,
    )
    _write_boxes(
        match_dir / "player_boxes_25_native_sided_v1.csv",
        [
            _row(court, frame, side)
            for frame in frames
            for court, side in ((near, "near"), (far, "far"))
        ],
        ["clip", "frame", "side", "x0", "y0", "x1", "y1", "conf", "court_x", "court_y", "track_id"],
        res.LEGACY_TRACKING_SIZE,
    )
    _write_boxes(
        match_dir / "ball_track_joint_native1080_integrity_v1.csv",
        [
            {
                "clip": "pt0001",
                "frame": f"f_{frame:04d}.jpg",
                "x": contact[0] / 2,
                "y": contact[1] / 2,
            }
            for frame in frames
        ],
        ["clip", "frame", "x", "y"],
        res.LEGACY_TRACKING_SIZE,
    )
    np.savez(match_dir / "court_H_per_point.npz", pts=[1], H=[HOMOGRAPHY])
    np.savez(
        match_dir / "camera_P_per_frame_v1.npz",
        clips=np.array(["pt0001"]),
        frames=np.array([100]),
        P=np.array([PROJECTION]),
        reliable=np.array([True]),
    )
    emissions = match_dir / "event_emissions.json"
    emissions.write_text(
        _json.dumps(
            [
                {"clip": "match__pt0001", "event_type": "contact", "frame": 100.0},
                {"clip": "match__pt0001", "event_type": "bounce", "frame": 110.0},
            ]
        )
    )

    inputs = striker.MatchInputs.resolve(match_dir)
    contacts = striker.emitted_contacts(emissions, None)

    assert inputs.box_scale == 2.0
    assert inputs.fps == 25.0
    (resolved,) = inputs.resolve_clip("pt0001", contacts["pt0001"])
    assert resolved["end"] == "near"
    assert resolved["status"] == "resolved"


def test_a_clip_with_no_automatic_court_geometry_abstains_with_its_reason(tmp_path):
    from cv.pipeline import resolution as res

    match_dir = tmp_path / "match"
    match_dir.mkdir()
    _write_boxes(
        match_dir / "player_boxes_25_native_v1.csv",
        [
            {
                "clip": "pt0001",
                "frame": "f_0100.jpg",
                "x0": 1,
                "y0": 1,
                "x1": 2,
                "y1": 2,
                "conf": 0.9,
            }
        ],
        ["clip", "frame", "x0", "y0", "x1", "y1", "conf"],
        res.NATIVE_SIZE,
    )
    np.savez(match_dir / "court_H_per_point.npz", pts=[1], H=[np.full((3, 3), np.nan)])

    inputs = striker.MatchInputs.resolve(match_dir)
    (resolved,) = inputs.resolve_clip("pt0001", [{"frame": 100.0}])

    assert resolved["status"] == "abstain"
    assert resolved["abstain_reason"] == "no_automatic_court_geometry"


def test_match_inputs_preserve_fractional_native_fps_from_box_name(tmp_path):
    from cv.pipeline import resolution as res

    match_dir = tmp_path / "match"
    match_dir.mkdir()
    _write_boxes(
        match_dir / "player_boxes_59.9401_native_v1.csv",
        [],
        ["clip", "frame", "x0", "y0", "x1", "y1", "conf"],
        res.NATIVE_SIZE,
    )

    assert striker.MatchInputs.resolve(match_dir).fps == 59.9401
