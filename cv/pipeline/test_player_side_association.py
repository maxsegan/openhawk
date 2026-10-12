import cv2
import csv
import json

import numpy as np

import pytest

import player_side_association as psa
import resolution as res
from player_side_association import (
    associate_rows,
    recorded_keep_unique_admissible_frames,
    require_keep_unique_admissible_frames,
    write_rows,
)


def test_associate_rows_scales_native_homography_to_artifact_coordinates():
    native_corners = np.float32([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    court_corners = np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]])
    homography = cv2.getPerspectiveTransform(native_corners, court_corners)
    rows = []
    for frame in range(1, 31):
        rows.extend(
            [
                {
                    "clip": "pt0001",
                    "frame": f"f_{frame:04d}.jpg",
                    "x0": "420",
                    "y0": "350",
                    "x1": "540",
                    "y1": "450",
                    "conf": "0.9",
                },
                {
                    "clip": "pt0001",
                    "frame": f"f_{frame:04d}.jpg",
                    "x0": "455",
                    "y0": "115",
                    "x1": "505",
                    "y1": "155",
                    "conf": "0.9",
                },
            ]
        )
    associated = associate_rows(
        rows,
        {1: homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        tracker="greedy",
    )
    assert {row["side"] for row in associated} == {"near", "far"}
    assert len(associated) == 60


def test_empty_abstention_preserves_native_schema(tmp_path):
    output = tmp_path / "player_boxes_native_sided_v1.csv"
    write_rows(output, [], include_native=True)

    with output.open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames[-4:] == [
            "x0_native",
            "y0_native",
            "x1_native",
            "y1_native",
        ]


def test_track_export_uses_the_same_single_association_and_declares_its_limits(
    tmp_path, monkeypatch
):
    import sys

    boxes = tmp_path / "boxes.csv"
    boxes.write_text("clip,frame,x0,y0,x1,y1,conf\n")
    output = tmp_path / "sided.csv"
    tracks = tmp_path / "tracks.csv"
    calls = []
    monkeypatch.setattr(
        psa, "coordinate_sizes", lambda _: (psa.res.NATIVE_SIZE, psa.res.NATIVE_SIZE)
    )
    monkeypatch.setattr(psa, "load_homographies", lambda _: {})
    monkeypatch.setattr(psa, "load_frame_homographies", lambda *args, **kwargs: {})
    monkeypatch.setattr(psa, "track_clips", lambda *args, **kwargs: calls.append(1) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            str(boxes),
            "--court",
            str(tmp_path / "H.npz"),
            "--fps",
            "50",
            "--output",
            str(output),
            "--tracks-output",
            str(tracks),
        ],
    )
    psa.main()
    assert len(calls) == 1
    contract = json.loads(tracks.with_suffix(".csv.coordinates.json").read_text())
    assert contract["fps"] == 50.0
    assert contract["position_estimator"] == "box_bottom_center_ground_projection"
    assert contract["identity_scope"] == "point_local_side_track"
    assert contract["airborne_status_available"] is False
    assert contract["abstained"] is True


def _court_homography(native_corners):
    court_corners = np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]])
    return cv2.getPerspectiveTransform(np.float32(native_corners), court_corners)


def _rows(native: bool, frames=range(1, 31)):
    rows = []
    for frame in frames:
        for box in ((840, 700, 1080, 900), (910, 230, 1010, 310)):
            row = {
                "clip": "pt0001",
                "frame": f"f_{frame:04d}.jpg",
                "x0": str(box[0] / 2),
                "y0": str(box[1] / 2),
                "x1": str(box[2] / 2),
                "y1": str(box[3] / 2),
                "conf": "0.9",
            }
            if native:
                row |= {
                    "x0_native": str(box[0]),
                    "y0_native": str(box[1]),
                    "x1_native": str(box[2]),
                    "y1_native": str(box[3]),
                }
            rows.append(row)
    return rows


def test_native_columns_are_read_when_present():
    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])

    associated = associate_rows(
        _rows(native=True),
        {1: homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        tracker="greedy",
    )
    legacy = associate_rows(
        _rows(native=False),
        {1: homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        tracker="greedy",
    )

    assert {row["side"] for row in associated} == {"near", "far"}
    assert len(associated) == len(legacy) == 60
    # identical geometry, so the two coordinate spaces must agree to rounding
    assert (
        max(
            abs(a["court_y"] - b["court_y"])
            for a, b in zip(sorted(associated, key=str), sorted(legacy, key=str))
        )
        < 0.01
    )


def test_frame_track_homographies_override_the_point_homography():
    point_homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    panned = _court_homography([[260, 900], [1780, 900], [1440, 300], [600, 300]])
    rows = _rows(native=True)

    static = associate_rows(
        rows,
        {1: point_homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        tracker="greedy",
    )
    tracked = associate_rows(
        rows,
        {1: point_homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        frame_homographies={"pt0001": dict.fromkeys(range(1, 31), panned)},
        tracker="greedy",
    )

    assert len(tracked) == len(static) == 60
    assert any(
        abs(a["court_x"] - b["court_x"]) > 0.1
        for a, b in zip(
            sorted(static, key=lambda r: (r["frame"], r["side"])),
            sorted(tracked, key=lambda r: (r["frame"], r["side"])),
        )
    )


def test_frame_homography_loader_preserves_rejection(tmp_path):
    path = tmp_path / "court.npz"
    np.savez(
        path,
        clips=["pt0001", "pt0001"],
        frames=[1, 2],
        H=[np.eye(3), np.eye(3)],
        reliable=[False, True],
    )
    loaded = psa.load_frame_homographies(path)
    assert set(loaded["pt0001"]) == {2}


def test_shot_propagation_copies_the_nearest_reliable_homography(tmp_path):
    path = tmp_path / "court.npz"
    registered = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    fallback = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]])
    np.savez(
        path,
        clips=["pt0001", "pt0001", "pt0001"],
        frames=[1, 5, 9],
        H=[fallback, registered, fallback],
        reliable=[False, True, False],
    )
    off = psa.load_frame_homographies(path)
    assert set(off["pt0001"]) == {5}
    np.testing.assert_array_equal(off["pt0001"][5], registered)
    same_shot = {"pt0001": {1: 0, 5: 0, 9: 0}}
    on = psa.load_frame_homographies(path, propagate_shot=True, shot_ids=same_shot)
    assert set(on["pt0001"]) == {1, 5, 9}
    np.testing.assert_array_equal(on["pt0001"][1], registered)
    np.testing.assert_array_equal(on["pt0001"][9], registered)
    np.testing.assert_array_equal(on["pt0001"][5], registered)
    # No shot map is not permission to paint every finite frame.
    assert set(psa.load_frame_homographies(path, propagate_shot=True)["pt0001"]) == {5}


def test_shot_propagation_does_not_enter_another_shot(tmp_path):
    path = tmp_path / "court.npz"
    registered = np.eye(3)
    fallback = np.eye(3)
    fallback[1, 2] = 4.0
    np.savez(
        path,
        clips=["pt0001", "pt0001", "pt0001"],
        frames=[1, 5, 9],
        H=[fallback, registered, fallback],
        reliable=[False, True, False],
    )
    # Frame 1 is a different shot (the close-up). Frame 9 is the wide shot.
    on = psa.load_frame_homographies(
        path,
        propagate_shot=True,
        shot_ids={"pt0001": {1: 1, 5: 0, 9: 0}},
    )
    assert set(on["pt0001"]) == {5, 9}
    np.testing.assert_array_equal(on["pt0001"][9], registered)
    np.testing.assert_array_equal(on["pt0001"][5], registered)


def test_shot_propagation_does_not_invent_a_homography_without_a_registered_frame(tmp_path):
    path = tmp_path / "court.npz"
    np.savez(
        path,
        clips=["pt0001", "pt0001"],
        frames=[1, 2],
        H=[np.eye(3), np.eye(3)],
        reliable=[False, False],
    )
    assert psa.load_frame_homographies(path, propagate_shot=True) == {"pt0001": {}}


def test_frame_homography_without_reliability_is_not_certified(tmp_path):
    path = tmp_path / "court.npz"
    np.savez(path, clips=["pt0001"], frames=[1], H=[np.eye(3)])
    assert psa.load_frame_homographies(path) == {"pt0001": {}}


def test_absent_legacy_frame_track_is_distinct_from_abstention(tmp_path):
    assert psa.load_frame_homographies(tmp_path / "missing.npz") is None


def test_frame_homography_rejects_truthy_strings(tmp_path):
    path = tmp_path / "court.npz"
    np.savez(path, clips=["pt0001"], frames=[1], H=[np.eye(3)], reliable=["False"])
    with pytest.raises(ValueError, match="boolean"):
        psa.load_frame_homographies(path)


def test_contact_resolver_does_not_restore_rejected_frame_geometry(monkeypatch):
    monkeypatch.setattr(
        psa, "resolve_contact_striker", lambda *a, **kw: pytest.fail("rejected camera reused")
    )
    output, ledger = psa.patch_contact_strikers(
        [],
        _rows(native=True),
        [{"clip": "pt0001", "frame": 15, "image_x": 900, "image_y": 800}],
        {1: _native_homography()},
        fps=25.0,
        image_size=psa.res.NATIVE_SIZE,
        artifact_size=psa.res.NATIVE_SIZE,
        frame_homographies={},
    )
    assert output == ledger == []


@pytest.mark.parametrize("tracker", ["greedy", "bytetrack"])
@pytest.mark.parametrize("frame_track", [{}, {"pt0001": {}}])
def test_present_empty_frame_track_never_resurrects_static_camera(tracker, frame_track):
    associated = associate_rows(
        _rows(native=True),
        {1: _native_homography()},
        fps=25.0,
        image_size=psa.res.NATIVE_SIZE,
        artifact_size=psa.res.NATIVE_SIZE,
        frame_homographies=frame_track,
        tracker=tracker,
    )
    assert associated == []


def test_player_detections_only_use_witnessed_frame_geometry():
    homography = _native_homography()
    detections = psa.build_detections(
        _rows(native=True, frames=range(1, 4)),
        {1: homography},
        image_size=psa.res.NATIVE_SIZE,
        artifact_size=psa.res.NATIVE_SIZE,
        frame_homographies={"pt0001": {2: homography}},
    )
    assert {d.frame for d in detections["pt0001"]} == {2}


def test_net_band_detection_keeps_the_side_it_came_from():
    from cv.pipeline.camera_cal import NET_Y
    from cv.pipeline.player_side_association import _side_with_hysteresis

    last = {"far": (5, 5.0, NET_Y + 1.5), "near": (5, 4.0, NET_Y - 6.0)}

    # a far player who reached the net keeps the far side inside the band
    assert _side_with_hysteresis(NET_Y - 0.3, 20, 25.0, last, 5.2) == "far"
    # with no history the hard cut still applies
    assert _side_with_hysteresis(NET_Y - 0.3, 20, 25.0, {}, 5.2) == "near"
    # clearly near stays near whatever the history says
    assert _side_with_hysteresis(NET_Y - 5.0, 20, 25.0, last, 4.2) == "near"
    # an unreachable history does not drag the side across the net
    assert _side_with_hysteresis(NET_Y - 0.3, 6, 25.0, last, 5.2) == "near"


def _tracker_rows(frames=range(1, 61)):
    """Two players plus a stationary line judge outside the doubles sideline."""
    rows = []
    for frame in frames:
        drift = frame * 2
        boxes = (
            (840 + drift, 700, 1000 + drift, 900),  # near player, moving
            (910 - drift // 2, 170, 970 - drift // 2, 310),  # far player, moving
            (150, 300, 200, 380),  # line judge, still, outside the court
        )
        for box in boxes:
            rows.append(
                {
                    "clip": "pt0001",
                    "frame": f"f_{frame:04d}.jpg",
                    "x0": str(box[0] / 2),
                    "y0": str(box[1] / 2),
                    "x1": str(box[2] / 2),
                    "y1": str(box[3] / 2),
                    "x0_native": str(box[0]),
                    "y0_native": str(box[1]),
                    "x1_native": str(box[2]),
                    "y1_native": str(box[3]),
                    "conf": "0.9",
                }
            )
    return rows


def test_bytetrack_arm_returns_one_track_id_per_side():
    from collections import defaultdict

    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    rows = _tracker_rows()
    associated = associate_rows(
        rows,
        {1: homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        tracker="bytetrack",
        frame_counts={"pt0001": 60},
    )
    ids = defaultdict(set)
    for row in associated:
        ids[row["side"]].add(row["track_id"])
    assert set(ids) == {"near", "far"}
    assert all(len(seen) == 1 for seen in ids.values())


def test_bytetrack_is_the_default_tracker():
    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    associated = associate_rows(
        _tracker_rows(),
        {1: homography},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        frame_counts={"pt0001": 60},
    )

    assert {row["track_id"] for row in associated if row["side"] == "near"} == {0}


def test_unknown_tracker_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        associate_rows(
            [],
            {},
            fps=25.0,
            image_size=res.FrameSize(1920, 1080),
            artifact_size=res.FrameSize(960, 540),
            tracker="sort",
        )


def test_track_rows_carry_root_court_and_name_columns(tmp_path):
    from cv.pipeline.player_side_association import (
        PLAYER_TRACKS_FIELDS,
        build_detections,
        track_clips,
        track_gaps,
        track_rows,
        write_track_rows,
    )

    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    detections = build_detections(
        _tracker_rows(),
        {1: homography},
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
    )
    tracked = track_clips(detections, fps=25.0, frame_counts={"pt0001": 80})
    rows = track_rows(tracked, names={"pt0001": {"near": "sinner", "far": "alcaraz"}})
    assert {row["name"] for row in rows} == {"sinner", "alcaraz"}
    near = next(row for row in rows if row["side"] == "near")
    assert near["t"] == ""
    assert near["root_x_native"] == round((near["x0_native"] + near["x1_native"]) / 2, 1)
    assert near["root_y_native"] == near["y1_native"]
    assert near["occluded"] == "" and near["airborne"] == ""

    output = tmp_path / "player_tracks_native_v1.csv"
    write_track_rows(output, rows)
    with output.open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == PLAYER_TRACKS_FIELDS

    # frames 61-80 have no detection at all, so they are reported as an explicit gap
    spans = track_gaps(tracked, {"pt0001": 80})
    assert all(span["end"] == 80 for span in spans)
    assert {span["side"] for span in spans} == {"near", "far"}


def test_embeddings_are_attached_by_index():
    from cv.pipeline.player_side_association import build_detections

    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    rows = _tracker_rows(frames=range(1, 3))
    for index, row in enumerate(rows):
        row["embed_index"] = str(index)
    embeddings = np.eye(len(rows), dtype=np.float32)
    detections = build_detections(
        rows,
        {1: homography},
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(960, 540),
        embeddings=embeddings,
    )
    attached = [d for d in detections["pt0001"] if d.embedding is not None]
    assert len(attached) == len(detections["pt0001"])
    assert all(float(np.linalg.norm(d.embedding)) == 1.0 for d in attached)


def test_track_rows_take_names_only_when_the_witnesses_agree():
    from cv.pipeline.player_identity import names_by_clip
    from cv.pipeline.player_side_association import (
        build_detections,
        track_clips,
        track_rows,
    )

    agreed = {
        "schema": "tennis.player_identity.v1",
        "identities": [
            {"identity": "A", "name": "sinner", "name_source": "scoreboard_server_anchor"},
            {"identity": "B", "name": "alcaraz", "name_source": "scoreboard_server_anchor"},
        ],
        "witnesses": {"agree": True, "failures": []},
        "points": [{"clip": "pt0001", "near_name": "sinner", "far_name": "alcaraz"}],
    }
    disagreed = {
        **agreed,
        "identities": [
            {"identity": "A", "name": "", "name_source": "withheld_witness_disagreement"},
            {"identity": "B", "name": "", "name_source": "withheld_witness_disagreement"},
        ],
        "witnesses": {"agree": False, "failures": ["appearance disagrees with ends parity"]},
    }
    homography = _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])
    tracked = track_clips(
        build_detections(
            _tracker_rows(),
            {1: homography},
            image_size=res.FrameSize(1920, 1080),
            artifact_size=res.FrameSize(960, 540),
        ),
        fps=25.0,
        frame_counts={"pt0001": 80},
    )
    named = track_rows(tracked, names=names_by_clip(agreed))
    assert {row["name"] for row in named} == {"sinner", "alcaraz"}
    withheld = track_rows(tracked, names=names_by_clip(disagreed))
    assert {row["name"] for row in withheld} == {""}


# --- Per-contact striker resolution -------------------------------------------------------


def _native_homography():
    """Image (native pixels) -> court metres, near baseline at the bottom of the frame."""
    return _court_homography([[200, 900], [1720, 900], [1380, 300], [540, 300]])


def _box(x0, y0, x1, y1, conf="0.9", frame=100):
    return {
        "clip": "pt0001",
        "frame": f"f_{frame:04d}.jpg",
        "x0": str(x0),
        "y0": str(y0),
        "x1": str(x1),
        "y1": str(y1),
        "conf": conf,
    }


def test_contact_reach_is_zero_inside_the_box_and_scales_by_box_height():
    box = (100.0, 100.0, 200.0, 300.0)
    assert psa.contact_reach((150.0, 200.0), box) == 0.0
    # 100 px to the right of a 200 px tall box is half a body height.
    assert psa.contact_reach((300.0, 200.0), box) == pytest.approx(0.5)


def test_standing_height_px_reads_a_person_off_the_camera():
    projection = np.array(
        [[500.0, 0.0, 0.0, 400.0], [0.0, 0.0, -500.0, 900.0], [0.0, 0.0, 0.0, 1.0]]
    )
    assert psa.standing_height_px(projection, np.array([0.0, 0.0])) == pytest.approx(900.0)
    assert psa.standing_height_px(None, np.array([0.0, 0.0])) is None


def test_resolver_picks_the_body_the_racket_belonged_to():
    homography = _native_homography()
    striker = _box(840, 700, 1000, 900)
    bystander = _box(300, 700, 460, 900)
    detections = {100: [striker, bystander]}
    resolved = psa.resolve_contact_striker(
        (1020.0, 760.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
    )
    assert resolved is not None
    assert resolved["side"] == "near"
    assert resolved["row"] is striker
    assert resolved["reach"] < psa.CONTACT_REACH_BAND
    assert resolved["confidence"] > 0.5


def test_resolver_refuses_a_body_on_the_other_side_when_a_side_is_required():
    homography = _native_homography()
    near = _box(840, 700, 1000, 900)
    far = _box(930, 300, 990, 380)
    detections = {100: [near, far]}
    on_far = psa.resolve_contact_striker(
        (1000.0, 340.0), 100.0, detections, homography, fps=25.0, side="far", box_scale=1.0
    )
    assert on_far is not None and on_far["row"] is far
    on_near = psa.resolve_contact_striker(
        (1000.0, 340.0), 100.0, detections, homography, fps=25.0, side="near", box_scale=1.0
    )
    # The near player is metres from that pixel, so nothing on the near side qualifies.
    assert on_near is None


def test_resolver_refuses_a_box_that_is_not_a_standing_person_at_its_own_court_position():
    homography = _native_homography()
    projection = np.array(
        [[500.0, 0.0, 0.0, 400.0], [0.0, 0.0, -500.0, 900.0], [0.0, 0.0, 0.0, 1.0]]
    )
    # A 40 px box where a 900 px person belongs: a ball kid, a half-box, or a mis-detection.
    tiny = _box(840, 860, 880, 900)
    detections = {100: [tiny]}
    assert (
        psa.resolve_contact_striker(
            (890.0, 880.0),
            100.0,
            detections,
            homography,
            fps=25.0,
            projection=projection,
            box_scale=1.0,
        )
        is None
    )
    assert (
        psa.resolve_contact_striker(
            (890.0, 880.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
        )
        is not None
    )


def test_resolver_prefers_the_body_the_point_track_can_have_reached():
    homography = _native_homography()
    # A deep near player whose tall box overlaps a far player in the image: the pixel is
    # within a racket of both, and only the court positions tell them apart.
    near = _box(900, 400, 1060, 900)
    far = _box(1000, 380, 1060, 470)
    detections = {100: [near, far]}
    pixel = (1080.0, 430.0)
    unconstrained = psa.resolve_contact_striker(
        pixel, 100.0, detections, homography, fps=25.0, box_scale=1.0
    )
    assert unconstrained["row"] is near
    far_court = np.asarray(
        cv2.perspectiveTransform(np.float32([[[1030.0, 470.0]]]), homography)[0, 0], float
    )
    anchors = [(98, far_court), (102, far_court)]
    constrained = psa.resolve_contact_striker(
        pixel, 100.0, detections, homography, fps=25.0, anchors=anchors, box_scale=1.0
    )
    assert constrained["row"] is far
    assert constrained["continuity_penalty"] == 0.0


def test_resolver_returns_nothing_when_no_body_is_within_a_racket_of_the_pixel():
    homography = _native_homography()
    detections = {100: [_box(300, 700, 460, 900)]}
    assert (
        psa.resolve_contact_striker(
            (1600.0, 760.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
        )
        is None
    )


def test_resolver_confidence_falls_as_the_contact_moves_away_from_the_body():
    homography = _native_homography()
    detections = {100: [_box(840, 700, 1000, 900)]}
    close = psa.resolve_contact_striker(
        (1010.0, 800.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
    )
    far = psa.resolve_contact_striker(
        (1070.0, 800.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
    )
    assert close["confidence"] > far["confidence"] > 0.0


def _sided_row(frame, side, x0, y0, x1, y1, court_x, court_y, track_id="0"):
    return {
        "clip": "pt0001",
        "frame": f"f_{frame:04d}.jpg",
        "side": side,
        "x0": str(x0),
        "y0": str(y0),
        "x1": str(x1),
        "y1": str(y1),
        "conf": "0.9",
        "court_x": str(court_x),
        "court_y": str(court_y),
        "track_id": track_id,
    }


def _patch_inputs(striker_box, incumbent_box):
    raw = [_box(*striker_box, frame=frame) for frame in range(96, 105)] + [
        _box(*incumbent_box, frame=frame) for frame in range(96, 105)
    ]
    sided = [_sided_row(frame, "near", *incumbent_box, 2.0, 1.0) for frame in range(96, 105)] + [
        _sided_row(frame, "far", 930, 300, 990, 380, 5.0, 20.0) for frame in range(96, 105)
    ]
    contacts = [{"clip": "pt0001", "frame": 100.0, "image_x": 1020.0, "image_y": 760.0}]
    return raw, sided, contacts


def _patch(raw, sided, contacts):
    return psa.patch_contact_strikers(
        sided,
        raw,
        contacts,
        {1: _native_homography()},
        fps=25.0,
        image_size=res.FrameSize(1920, 1080),
        artifact_size=res.FrameSize(1920, 1080),
    )


def test_patch_replaces_only_the_contact_span():
    raw, sided, contacts = _patch_inputs((840, 700, 1000, 900), (300, 700, 460, 900))
    patched, ledger = _patch(raw, sided, contacts)
    assert len(patched) == len(sided)
    span = {100 - psa.CONTACT_PATCH_SPAN_FRAMES, 100, 100 + psa.CONTACT_PATCH_SPAN_FRAMES}
    for frame in span:
        row = next(
            row for row in patched if row["side"] == "near" and row["frame"] == f"f_{frame:04d}.jpg"
        )
        assert float(row["x0"]) == 840.0
    outside = next(row for row in patched if row["side"] == "near" and row["frame"] == "f_0097.jpg")
    assert float(outside["x0"]) == 300.0
    near = [entry for entry in ledger if entry["side"] == "near"]
    assert len(near) == len(span)
    assert all(entry["action"] == "replaced" for entry in near)
    assert all(entry["resolved_reach"] < entry["incumbent_reach"] for entry in near)


def test_patch_keeps_the_tracker_when_it_is_already_the_striker():
    raw, sided, contacts = _patch_inputs((840, 700, 1000, 900), (845, 700, 1005, 900))
    patched, ledger = _patch(raw, sided, contacts)
    near = [entry for entry in ledger if entry["side"] == "near"]
    assert near[0]["action"] == "none"
    contact_row = next(
        row for row in patched if row["side"] == "near" and row["frame"] == "f_0100.jpg"
    )
    assert float(contact_row["x0"]) == 845.0


def test_patch_inserts_a_striker_where_the_tracker_has_no_row_at_all():
    raw, sided, contacts = _patch_inputs((840, 700, 1000, 900), (300, 700, 460, 900))
    sided = [row for row in sided if row["side"] != "near"]
    patched, ledger = _patch(raw, sided, contacts)
    inserted = [entry for entry in ledger if entry["action"] == "inserted"]
    assert len(inserted) == 1 + 2 * psa.CONTACT_PATCH_SPAN_FRAMES
    assert any(
        row["side"] == "near" and row["frame"] == "f_0100.jpg" and float(row["x0"]) == 840.0
        for row in patched
    )


def test_contact_observations_filters_by_match_and_drops_unusable_rows(tmp_path):
    payload = {
        "emissions": [
            {
                "event_type": "contact",
                "clip": "a__pt0001",
                "match_id": "a",
                "frame": 10.0,
                "location": {"image_x": 1.0, "image_y": 2.0},
            },
            {
                "event_type": "contact",
                "clip": "b__pt0001",
                "match_id": "b",
                "frame": 11.0,
                "location": {"image_x": 1.0, "image_y": 2.0},
            },
            {
                "event_type": "bounce",
                "clip": "a__pt0001",
                "match_id": "a",
                "frame": 12.0,
                "location": {"image_x": 1.0, "image_y": 2.0},
            },
            {
                "event_type": "contact",
                "clip": "a__pt0001",
                "match_id": "a",
                "frame": 13.0,
                "abstain": True,
                "location": {"image_x": 1.0, "image_y": 2.0},
            },
            {
                "event_type": "contact",
                "clip": "a__pt0001",
                "match_id": "a",
                "frame": 14.0,
                "location": {"image_x": None, "image_y": None},
            },
        ]
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps(payload))
    rows = psa.contact_observations(path, "a")
    assert [row["frame"] for row in rows] == [10.0]
    assert rows[0]["clip"] == "pt0001"
    assert len(psa.contact_observations(path)) == 2


def test_resolver_ignores_a_partial_box_of_a_body_it_has_already_seen_whole():
    homography = _native_homography()
    whole = _box(840, 700, 1000, 900)
    sliver = _box(960, 720, 1000, 900, conf="0.7")
    detections = {100: [whole, sliver]}
    # The sliver's edge is no nearer, but its box height makes its reach smaller; without
    # suppression the resolver would take it and read the court position off its own bottom.
    resolved = psa.resolve_contact_striker(
        (1040.0, 760.0), 100.0, detections, homography, fps=25.0, box_scale=1.0
    )
    assert resolved["row"] is whole


def test_resolver_keeps_a_far_player_standing_inside_a_near_player_box():
    homography = _native_homography()
    near = _box(900, 400, 1060, 900)
    far = _box(980, 420, 1030, 490)
    detections = {100: [near, far]}
    resolved = psa.resolve_contact_striker(
        (1040.0, 450.0), 100.0, detections, homography, fps=25.0, side="far", box_scale=1.0
    )
    assert resolved is not None and resolved["row"] is far


def test_player_state_keeps_the_better_sided_box_position_and_adds_pose_hip():
    track = {
        "clip": "pt0001",
        "frame": 10,
        "side": "near",
        "name": "",
        "track_id": "n",
        "root_x_native": 4.0,
        "root_y_native": 5.0,
        "court_x": 4.0,
        "court_y": 5.0,
    }
    pose = {
        "clip": "pt0001",
        "frame": "f_0010.jpg",
        "side": "near",
        "conf": 0.9,
        "left_ankle_x": 100.0,
        "left_ankle_y": 100.0,
        "left_ankle_confidence": 0.9,
        "right_ankle_x": 120.0,
        "right_ankle_y": 100.0,
        "right_ankle_confidence": 0.9,
        "left_hip_x": 4.0,
        "left_hip_y": 3.0,
        "left_hip_confidence": 0.9,
        "right_hip_x": 4.0,
        "right_hip_y": 3.0,
        "right_hip_confidence": 0.9,
    }

    rows = psa.build_player_state_rows(
        [track],
        [pose],
        homography_for_frame=lambda _clip, _frame: np.eye(3),
        projection_for_frame=lambda _clip, _frame: None,
    )

    assert rows[0]["position_source"] == "sided_box_bottom_center"
    assert (rows[0]["court_x"], rows[0]["court_y"]) == (4.0, 5.0)
    assert rows[0]["t"] == ""
    assert (rows[0]["hip_x_native"], rows[0]["hip_y_native"]) == (4.0, 3.0)


def test_player_state_sidecar_declares_evaluation_inputs_when_present(tmp_path):
    output = tmp_path / "player_state.csv"
    truth = tmp_path / "truth.json"
    truth.write_text("{}")

    psa.write_player_state_artifact(
        output,
        [],
        sources={},
        fps=25.0,
        foot_estimator="sided_box",
        automatic=False,
        labels_or_reviewed_inputs=[str(truth)],
    )

    sidecar = json.loads(res.coordinate_manifest_path(output).read_text())
    assert sidecar["artifact_identity"] == psa.PLAYER_STATE_SCHEMA
    assert sidecar["automatic"] is False
    assert sidecar["labels_or_reviewed_inputs"] == [str(truth)]
    provenance = json.loads(output.with_suffix(".csv.provenance.json").read_text())
    assert provenance["mode"] == "diagnostic"
    assert provenance["reviewed_inputs"][0]["role"] == "player_truth"


def test_automatic_player_state_requires_parent_provenance(tmp_path):
    with pytest.raises(ValueError, match="parent automatic provenance"):
        psa.write_player_state_artifact(
            tmp_path / "player_state.csv",
            [],
            sources={},
            fps=25.0,
            foot_estimator="sided_box",
        )


def test_reliable_view_scope_separates_camera_gaps_but_not_detector_misses():
    rows = _rows(native=True, frames=range(1, 62))
    for row in rows:
        row["t"] = f"original-{row['frame']}"
    homography = _native_homography()
    support = {"pt0001": {f: homography for f in range(1, 62) if f != 31}}
    # A detector dropout inside the first view is distinct from the unsupported view31.
    rows = [r for r in rows if psa.frame_number(r["frame"]) != 15]
    detections = psa.build_detections(
        rows,
        {1: homography},
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        frame_homographies=support,
    )
    tracks = psa.track_reliable_views(detections, support, fps=25.0)
    output = psa.track_rows(tracks)
    assert {r["frame"] for r in output} == set(range(1, 62)) - {15, 31}
    assert output
    for side in {r["side"] for r in output}:
        before = {r["track_id"] for r in output if r["side"] == side and r["frame"] < 31}
        after = {r["track_id"] for r in output if r["side"] == side and r["frame"] > 31}
        assert len(before) == len(after) == 1 and before.isdisjoint(after)
    assert all(r["t"] == f"original-f_{r['frame']:04d}.jpg" for r in output)
    assert all(d.row in rows for track in tracks["pt0001"].values() for d in track.detections)


def test_reliable_scope_held_interval_cannot_win_whole_clip_identity():
    # Same plausible court location, different actors. The longer second view must
    # not suppress the first view by winning whole-point persistence ranking.
    from cv.pipeline.player_tracker import Detection

    def det(f, x):
        return Detection(
            frame=f,
            box=(x * 50, 600, x * 50 + 70, 780),
            conf=0.9,
            court=(x, 1.0),
            ground_scale=100.0,
        )

    detections = {
        "pt0001": [det(f, 1.0) for f in range(1, 31)] + [det(f, 10.0) for f in range(32, 102)]
    }
    support = {"pt0001": {f: np.eye(3) for f in list(range(1, 31)) + list(range(32, 102))}}
    # Pin winner-take-all off so this test still measures reliable-view scoping, not
    # the production keep_unique_admissible_frames default.
    ordinary = psa.track_clips(
        detections,
        fps=25.0,
        selection_config=psa.SelectionConfig(keep_unique_admissible_frames=False),
    )
    scoped = psa.track_reliable_views(detections, support, fps=25.0)
    assert set(ordinary["pt0001"]["near"].frames) == set(range(32, 102))
    assert set(scoped["pt0001"]["near"].frames) == set(range(1, 31)) | set(range(32, 102))
    assert len({r["track_id"] for r in psa.sided_rows_from_tracks(scoped)}) == 2


def test_reliable_scope_empty_and_single_contiguous_view():
    rows = _rows(native=True)
    homography = _native_homography()
    support = {"pt0001": dict.fromkeys(range(1, 31), homography)}
    detections = psa.build_detections(
        rows,
        {1: homography},
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        frame_homographies=support,
    )
    original = psa.sided_rows_from_tracks(psa.track_clips(detections, fps=25.0))
    scoped = psa.sided_rows_from_tracks(psa.track_reliable_views(detections, support, fps=25.0))
    assert [{k: v for k, v in r.items() if k != "track_id"} for r in original] == [
        {k: v for k, v in r.items() if k != "track_id"} for r in scoped
    ]
    assert psa.sided_rows_from_tracks(psa.track_reliable_views(detections, {}, fps=25.0)) == []


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--tracker", "greedy"],
        ["--sided-boxes", "old.csv"],
        ["--contact-events", "events.json"],
    ],
)
def test_reliable_scope_requires_explicit_fresh_scoped_association(monkeypatch, extra):
    import sys

    command = [
        "player_side_association",
        "--boxes",
        "missing.csv",
        "--court",
        "court.npz",
        "--fps",
        "25",
        "--output",
        "out.csv",
        "--reliable-view-scope",
    ]
    if extra:
        command += ["--court-frame-track", "frame.npz"] + extra
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as error:
        psa.main()
    assert error.value.code == 2


def test_bilateral_mode_requires_reliable_view_scope(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            "missing.csv",
            "--court",
            "c.npz",
            "--fps",
            "25",
            "--output",
            "out.csv",
            "--bilateral-overlap-recovery",
        ],
    )
    with pytest.raises(SystemExit) as error:
        psa.main()
    assert error.value.code == 2


def test_bilateral_recovery_never_receives_rows_across_held_view(monkeypatch):
    from cv.pipeline.player_tracker import Detection

    calls = []

    def select(tracks, **kwargs):
        calls.append(({d.frame for t in tracks for d in t.detections}, kwargs["bilateral_overlap"]))
        return {}

    monkeypatch.setattr(psa, "select_players", select)
    dd = [
        Detection(
            frame=f,
            box=(100.0, 100.0, 140.0, 280.0),
            court=(4.0, -1.0),
            conf=0.9,
            ground_scale=100.0,
        )
        for f in range(1, 32)
    ]
    support = {"pt0001": {f: np.eye(3) for f in range(1, 32) if f != 16}}
    psa.track_reliable_views({"pt0001": dd}, support, fps=25.0, bilateral_overlap=True)
    assert calls == [(set(range(1, 16)), True), (set(range(17, 32)), True)]


@pytest.mark.parametrize("enabled", [False, True])
def test_lost_revival_body_history_reaches_the_tracker_and_the_receipt(
    tmp_path, monkeypatch, enabled
):
    """The explicit option must travel into the tracker config and the cached receipt."""
    import sys

    boxes = tmp_path / "boxes.csv"
    boxes.write_text("clip,frame,x0,y0,x1,y1,conf\n")
    output = tmp_path / "sided.csv"
    tracks = tmp_path / "tracks.csv"
    seen = []
    monkeypatch.setattr(
        psa, "coordinate_sizes", lambda _: (psa.res.NATIVE_SIZE, psa.res.NATIVE_SIZE)
    )
    monkeypatch.setattr(psa, "load_homographies", lambda _: {})
    monkeypatch.setattr(psa, "load_frame_homographies", lambda *args, **kwargs: {})
    monkeypatch.setattr(psa, "track_clips", lambda *args, **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            str(boxes),
            "--court",
            str(tmp_path / "H.npz"),
            "--fps",
            "25",
            "--output",
            str(output),
            "--tracks-output",
            str(tracks),
            *(["--lost-revival-body-history"] if enabled else []),
        ],
    )
    psa.main()
    assert seen[0]["tracker_config"].lost_revival_body_history is enabled
    assert seen[0]["audit"] is not None
    contract = json.loads(tracks.with_suffix(".csv.coordinates.json").read_text())
    assert contract["lost_revival_body_history"] is enabled
    assert ("lost_revival_body_history_audit" in contract) is enabled
    if enabled:
        audit = contract["lost_revival_body_history_audit"]
        assert audit["schema"] == "lost_revival_body_history_audit_v1"
        assert audit["revival_refusals"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_keep_unique_admissible_frames_reaches_the_tracker_and_the_receipt(
    tmp_path, monkeypatch, enabled
):
    import sys

    boxes = tmp_path / "boxes.csv"
    boxes.write_text("clip,frame,x0,y0,x1,y1,conf\n")
    boxes.with_suffix(".csv.coordinates.json").write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 1920, "height": 1080},
            }
        )
    )
    output = tmp_path / "sided.csv"
    tracks = tmp_path / "tracks.csv"
    seen = []
    monkeypatch.setattr(
        psa, "coordinate_sizes", lambda _: (psa.res.NATIVE_SIZE, psa.res.NATIVE_SIZE)
    )
    monkeypatch.setattr(psa, "load_homographies", lambda _: {})
    monkeypatch.setattr(psa, "load_frame_homographies", lambda *args, **kwargs: {})
    monkeypatch.setattr(psa, "track_clips", lambda *args, **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            str(boxes),
            "--court",
            str(tmp_path / "H.npz"),
            "--fps",
            "25",
            "--output",
            str(output),
            "--tracks-output",
            str(tracks),
            *(
                ["--keep-unique-admissible-frames"]
                if enabled
                else ["--no-keep-unique-admissible-frames"]
            ),
        ],
    )
    psa.main()
    assert seen[0]["selection_config"].keep_unique_admissible_frames is enabled
    sided_contract = json.loads(res.coordinate_manifest_path(output).read_text())
    tracks_contract = json.loads(res.coordinate_manifest_path(tracks).read_text())
    assert sided_contract[psa.KEEP_UNIQUE_SIDECAR_KEY] is enabled
    assert tracks_contract[psa.KEEP_UNIQUE_SIDECAR_KEY] is enabled


def _sided_pose(tmp_path, *, flag=None, identity=True, name="player_boxes_25_native_sided_v1.csv"):
    path = tmp_path / name
    path.write_text("clip,frame,side,x0,y0,x1,y1,court_x,court_y\n")
    extra = {"schema": "tennis.coordinate-space.v1", "artifact": path.name}
    if identity:
        extra["artifact_identity"] = res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY
    if flag is not None:
        extra[psa.KEEP_UNIQUE_SIDECAR_KEY] = flag
    path.with_name(path.name + ".coordinates.json").write_text(json.dumps(extra))
    return path


@pytest.mark.parametrize("flag", [False, True])
def test_require_keep_unique_admissible_frames_both_states(tmp_path, flag):
    path = _sided_pose(tmp_path, flag=flag)
    require_keep_unique_admissible_frames(path, expected=flag)
    with pytest.raises(ValueError, match="does not match production default"):
        require_keep_unique_admissible_frames(path, expected=not flag)
    assert recorded_keep_unique_admissible_frames(path) is flag


def test_unflagged_sided_boxes_are_not_the_production_default(tmp_path):
    path = _sided_pose(tmp_path, flag=None)
    assert recorded_keep_unique_admissible_frames(path) is False
    require_keep_unique_admissible_frames(path, expected=False)
    with pytest.raises(ValueError, match="regenerate sided boxes"):
        require_keep_unique_admissible_frames(path, expected=True)


def test_non_sided_pose_is_not_checked(tmp_path):
    path = tmp_path / "prepared_player_pose.csv"
    path.write_text("clip,frame\n")
    path.with_name(path.name + ".coordinates.json").write_text("{}")
    assert recorded_keep_unique_admissible_frames(path) is None
    require_keep_unique_admissible_frames(path, expected=True)


def test_lost_revival_body_history_refuses_the_greedy_linker(tmp_path, monkeypatch):
    import sys

    boxes = tmp_path / "boxes.csv"
    boxes.write_text("clip,frame,x0,y0,x1,y1,conf\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            str(boxes),
            "--court",
            str(tmp_path / "H.npz"),
            "--fps",
            "25",
            "--output",
            str(tmp_path / "sided.csv"),
            "--tracker",
            "greedy",
            "--lost-revival-body-history",
        ],
    )
    with pytest.raises(SystemExit):
        psa.main()


def test_guard_refuses_preselected_sided_boxes_before_reading_or_writing(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player_side_association",
            "--boxes",
            "missing.csv",
            "--court",
            "missing.npz",
            "--fps",
            "25",
            "--output",
            str(tmp_path / "out.csv"),
            "--sided-boxes",
            "existing.csv",
            "--lost-revival-body-history",
        ],
    )
    with pytest.raises(SystemExit) as error:
        psa.main()
    assert error.value.code == 2
    assert not (tmp_path / "out.csv").exists()


def test_camera_body_scale_vouches_only_when_it_is_nearer_a_standing_body():
    from cv.pipeline.player_tracker import Detection

    # Camera 100 px per metre everywhere on the ground and 50 px per metre of height:
    # a standing 1.8 m body projects to 90 px.
    projection = np.array([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, -50.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def detection(frame, height, ground_scale):
        return Detection(
            frame=frame,
            box=(0.0, 0.0, 10.0, height),
            conf=0.9,
            court=(1.0, 1.0),
            ground_scale=ground_scale,
        )

    refused = detection(1, 100.0, 100.0)  # ground ratio 0.56, camera ratio 1.11
    inflated = detection(2, 90.0, 60.0)  # ground ratio 0.83, camera ratio 1.0: camera
    kept = detection(3, 180.0, 100.0)  # ground ratio 1.0, camera ratio 2.0: ground stays
    no_camera = detection(4, 100.0, 100.0)
    out = psa.camera_body_scale(
        {"pt1": [refused, inflated, kept, no_camera]}, {1: projection, 2: projection, 3: projection}
    )["pt1"]
    assert out[0].size_ratio == pytest.approx(100.0 / 90.0)
    assert out[1].size_ratio == pytest.approx(1.0)
    assert out[2] is kept
    assert out[3] is no_camera
