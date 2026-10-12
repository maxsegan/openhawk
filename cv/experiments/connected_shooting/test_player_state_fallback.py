import csv

import pytest

from cv.experiments.connected_shooting import player_state_fallback as fallback


def artifact(tmp_path, rows):
    path = tmp_path / "player_boxes_native_sided_v1.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip", "frame", "side", "court_x", "court_y", "track_id"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(frame, side, x, y, clip="pt0001"):
    return {
        "clip": clip,
        "frame": f"f_{frame:04d}.jpg",
        "side": side,
        "court_x": x,
        "court_y": y,
        "track_id": "1",
    }


def test_the_nearest_sided_row_answers_and_declares_a_widened_sigma(tmp_path):
    path = artifact(tmp_path, [row(100, "far", 5.0, 22.0), row(120, "far", 6.0, 21.0)])
    state = fallback.contact_state(path, "pt0001", 104, "far")
    assert state["court_centre_xy_m"] == [5.0, 22.0]
    assert state["substitution"]["frame_distance"] == 4
    assert state["court_position_sigma_m"] == pytest.approx(0.65 + 0.16 * 4)
    assert state["pose_wrist_witness"]["status"] == "abstained"
    assert state["state_dof"] == 2


def test_the_required_side_is_never_swapped_for_a_closer_opponent(tmp_path):
    path = artifact(tmp_path, [row(104, "near", 5.0, 2.0), row(90, "far", 6.0, 21.0)])
    state = fallback.contact_state(path, "pt0001", 104, "far")
    assert state["side"] == "far"
    assert state["court_centre_xy_m"] == [6.0, 21.0]


def test_nothing_within_the_bounded_window_abstains(tmp_path):
    path = artifact(tmp_path, [row(10, "far", 5.0, 22.0)])
    assert fallback.contact_state(path, "pt0001", 200, "far") is None


def test_another_clip_and_a_blank_court_position_are_ignored(tmp_path):
    path = artifact(
        tmp_path,
        [
            row(104, "far", "", "", clip="pt0001"),
            row(104, "far", 9.0, 20.0, clip="pt0002"),
            row(108, "far", 5.0, 22.0),
        ],
    )
    state = fallback.contact_state(path, "pt0001", 104, "far")
    assert state["court_centre_xy_m"] == [5.0, 22.0]


def test_an_unsided_artifact_says_nothing(tmp_path):
    path = tmp_path / "unsided.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "frame", "x1", "y1"])
        writer.writeheader()
        writer.writerow({"clip": "pt0001", "frame": "f_0104.jpg", "x1": 1, "y1": 2})
    assert fallback.sided_court_rows(path, "pt0001") == {}


def test_a_side_that_is_not_near_or_far_is_refused(tmp_path):
    with pytest.raises(ValueError, match="near or far"):
        fallback.nearest_sided_position({}, 10, "left")
