from __future__ import annotations

from contacts_v2 import classify_segments, contact_f1, point_box_distance


def _segment(frame: int = 10) -> dict:
    return {
        "clip": "pt0001",
        "track_id": "1",
        "f0": str(frame),
        "fe": str(frame + 10),
        "rms_px": "1.0",
        "speed0": "20.0",
        "x0": "1.0",
        "y0": "2.0",
        "z0": "1.0",
        "vx0": "10.0",
        "vy0": "0.0",
        "vz0": "2.0",
        "xe": "3.0",
        "ye": "2.0",
        "ze": "1.2",
        "vxe": "9.0",
        "vye": "0.0",
        "vze": "-1.0",
        "u0": "100.0",
        "v0": "100.0",
    }


def test_point_box_distance() -> None:
    assert point_box_distance(5, 5, (0, 0, 10, 10)) == 0
    assert point_box_distance(13, 14, (0, 0, 10, 10)) == 5


def test_disconnected_segment_requires_player_proximity_to_be_a_hit() -> None:
    segment = _segment()
    no_boxes = classify_segments([segment], {}, "pt0001", fps=50)
    boxes = {("pt0001", "f_0010.jpg"): [(90.0, 90.0, 110.0, 110.0)]}
    near_player = classify_segments([segment], boxes, "pt0001", fps=50)

    assert all(event["kind"] != "hit" for event in no_boxes)
    assert any(event["kind"] == "hit" for event in near_player)


def test_contact_f1_matches_within_frame_tolerance() -> None:
    predicted = [
        {"clip": "pt0001", "frame": 10, "kind": "hit"},
        {"clip": "pt0001", "frame": 30, "kind": "hit"},
    ]
    truth = [
        {"clip": "pt0001", "frame": 12},
        {"clip": "pt0001", "frame": 50},
    ]

    report = contact_f1(predicted, truth, tolerance=5)

    assert report["tp"] == 1
    assert report["precision"] == report["recall"] == report["f1"] == 0.5


def test_contact_f1_ignores_predictions_outside_annotated_clips() -> None:
    predicted = [
        {"clip": "pt0001", "frame": 10, "kind": "hit"},
        {"clip": "pt0002", "frame": 10, "kind": "hit"},
    ]
    truth = [{"clip": "pt0001", "frame": 10}]

    report = contact_f1(predicted, truth, tolerance=5)

    assert report["precision"] == report["recall"] == report["f1"] == 1.0


def test_contact_f1_explicit_scope_counts_predictions_in_truth_empty_clip() -> None:
    predicted = [
        {"clip": "pt0001", "frame": 10, "kind": "hit"},
        {"clip": "pt0002", "frame": 10, "kind": "hit"},
    ]
    truth = [{"clip": "pt0001", "frame": 10}]

    report = contact_f1(
        predicted, truth, tolerance=5, evaluation_clips={"pt0001", "pt0002"}
    )

    assert report["precision"] == 0.5
    assert report["recall"] == 1.0
    assert report["f1"] == 2 / 3
