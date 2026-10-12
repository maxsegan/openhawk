from __future__ import annotations

import json
from pathlib import Path

from cv.validation import score_contact_strikers as score


def _row(**overrides) -> dict:
    row = {
        "attempt": "a",
        "match_id": "m",
        "clip": "pt0001",
        "frame": 10.0,
        "truth_end": "near",
        "truth_provenance": "labelled_hitter",
        "legacy_half_native_end": "far",
        "sided_nearest_end": "near",
        "resolver_end": "near",
        "resolver_status": "resolved",
        "confidence": 0.8,
        "reach": 0.1,
        "reach_end_margin": 1.0,
        "end_source": "point_track",
        "scale_reference": "tracked_end_height",
        "camera_scale_ratio": 1.0,
        "alternation_applied": False,
        "disagrees_with_sided_track": False,
        "wrist_px": 40.0,
        "root_px": 150.0,
        "wrist_box_heights": 0.2,
        "root_box_heights": 0.7,
        "pose_matched": True,
        "abstain_reason": None,
    }
    row.update(overrides)
    return row


def test_a_contact_no_arm_can_answer_stays_in_the_denominator():
    rows = [
        _row(),
        score._blank_row(
            "a", "m", "pt0001", 20.0, "far", "labelled_hitter", "no_automatic_court_geometry"
        ),
    ]

    summary = score.summarise(rows)

    assert summary["contacts"] == 2
    assert summary["resolver"]["correct"] == 1
    assert summary["resolver"]["no_answer"] == 1
    assert summary["resolver"]["correct_rate"] == 0.5
    assert summary["abstain_reason"] == {"no_automatic_court_geometry": 1}


def test_a_wrong_end_and_an_abstention_are_counted_separately():
    rows = [_row(resolver_end="far"), _row(resolver_end=None, resolver_status="abstain")]

    summary = score.summarise(rows)

    assert summary["resolver"]["wrong_side"] == 1
    assert summary["resolver"]["wrong_side_rate"] == 0.5
    assert summary["resolver"]["no_answer"] == 1
    assert summary["resolver"]["correct"] == 0


def test_labelled_truth_is_read_from_the_hitter_fields_not_from_alternation(tmp_path):
    label = tmp_path / "label.json"
    label.write_text(
        json.dumps(
            {
                "match_id": "m",
                "attempt": {"clip": "pt0002"},
                "events": {
                    "records": [
                        {"event_type": "contact", "frame": 10, "hitter_end": "near"},
                        {"event_type": "bounce", "frame": 15},
                        {"event_type": "contact", "frame": 20, "hitter_end": "near"},
                    ]
                },
            }
        )
    )

    resolved = score.truth_ends([{"key": "x", "path": str(label)}])

    assert resolved["x"]["provenance"] == "labelled_hitter"
    assert resolved["x"]["ends"] == ["near", "near"]
    assert resolved["x"]["frames"] == [10.0, 20.0]


def test_derived_truth_is_only_used_when_the_labels_name_no_hitter(tmp_path):
    label = tmp_path / "label.json"
    label.write_text(
        json.dumps(
            {
                "match_id": "m",
                "attempt": {"clip": "pt0001"},
                "events": {
                    "records": [
                        {"event_type": "contact", "frame": 68},
                        {"event_type": "contact", "frame": 86},
                    ]
                },
            }
        )
    )
    topology = json.loads(Path(score.TOPOLOGY).read_text())
    key = topology["attempts"][0]["key"]

    resolved = score.truth_ends([{"key": key, "path": str(label)}])

    assert resolved[key]["provenance"] == "derived_from_review_narrative"
    assert resolved[key]["ends"][0] == topology["attempts"][0]["server_camera_half"]
    assert resolved[key]["ends"][1] != resolved[key]["ends"][0]


def test_owner_clicks_are_converted_out_of_their_declared_540_space(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "clip,event_type,seed_id,seed_frame,labeled_frame,labeled_x540,labeled_y540,"
        "verdict,note\n"
        "m__pt0001,contact,s1,10,10,100,50,confirmed,\n"
        "m__pt0001,bounce,s2,12,12,110,60,confirmed,\n"
        "m__pt0001,contact,s3,14,,,,replaced,\n"
    )

    clicks = score.owner_contact_clicks(path)

    assert clicks == {"m": {"pt0001": [{"frame": 10.0, "image_x": 200.0, "image_y": 100.0}]}}
