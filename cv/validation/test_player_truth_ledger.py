from __future__ import annotations

import json

from cv.validation import player_truth_ledger as ledger


def _player() -> dict:
    return {
        "identity": "Player",
        "in_view": True,
        "occlusion": False,
        "hip_center_px": [10, 20],
        "feet": {
            "left": {
                "ground_contact_px": [9, 30],
                "airborne": False,
                "status": "grounded",
            },
            "right": {
                "ground_contact_px": None,
                "airborne": True,
                "status": "airborne",
            },
        },
        "racket": None,
    }


def test_current_truth_schema_normalises_contacts_and_mixed_support(tmp_path):
    path = tmp_path / "attempt_players_v1.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tennis_player_racket_truth_v1",
                "match_id": "match",
                "clip": "pt0001",
                "attempt": "attempt01",
                "native_window": [10, 10],
                "contacts": [
                    {"contact_frame": 10, "player_end": "near", "stroke_type": "forehand"}
                ],
                "frames": [
                    {
                        "frame": 10,
                        "native_pts_seconds": 0.4,
                        "players": {"near": _player(), "far": _player()},
                    }
                ],
            }
        )
    )

    truth = ledger.load_truth(path)

    assert truth["contacts"] == [
        {"frame": 10.0, "end": "near", "stroke_type": "forehand", "hand": None}
    ]
    assert truth["frames"][0]["players"]["near"]["feet"][0]["ground"] == (9.0, 30.0)


def test_sequence_schema_racket_rows_are_attached_to_the_player_frame(tmp_path):
    player = {
        "player": "P",
        "visibility": "visible",
        "occluded": False,
        "hip_centre": {"px": [10, 20]},
        "feet": {
            "left": {"ground_contact_px": [9, 30], "airborne": False},
            "right": {"ground_contact_px": [11, 30], "airborne": False},
        },
    }
    path = tmp_path / "sequence_players_v1.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tennis_player_racket_sequence_labels_v1",
                "match_id": "match",
                "clip": "pt0001",
                "native_window": [10, 10],
                "contacts": [{"frame": 10, "hitter_end": "far", "stroke": "serve"}],
                "racket_frames": [
                    {
                        "frame": 10,
                        "hitter_end": "far",
                        "contact_epoch_frame": 10,
                        "face_centre_px": [15, 5],
                        "striking_hand": "right",
                        "stroke": "serve",
                    }
                ],
                "frames": [
                    {
                        "frame": 10,
                        "native_pts_seconds": 0.4,
                        "players": {"near": player, "far": player},
                    }
                ],
            }
        )
    )

    truth = ledger.load_truth(path)

    racket = truth["frames"][0]["players"]["far"]["racket"]
    assert racket["active"] is True
    assert racket["face"] == (15.0, 5.0)


def test_position_summary_keeps_missing_predictions_in_the_denominator():
    rows = [
        {
            "truth_court": (0.0, 0.0),
            "sided_box_court": (0.3, 0.4),
            "side": "near",
            "motion": "planted",
        },
        {
            "truth_court": (1.0, 1.0),
            "sided_box_court": None,
            "side": "far",
            "motion": "moving",
        },
    ]

    summary = ledger._arm_summary(rows, "sided_box")

    assert summary["eligible_player_frames"] == 2
    assert summary["answered"] == 1
    assert summary["no_answer"] == 1
    assert summary["error_m"]["median"] == 0.5


def test_racket_arm_selects_the_lowest_median_scale():
    rows = [
        {
            "selector": selector,
            "scale": scale,
            "pixel_error": error,
            "court_error": error / 10,
        }
        for selector in ("automatic_ball_pixel", "truth_hand_oracle")
        for scale, error in ((0.0, 10.0), (0.5, 8.0), (1.0, 6.0), (1.5, 4.0), (2.0, 5.0))
    ]

    summary = ledger._racket_summary(rows)

    assert summary["best_forearm_scale"] == 1.5
