"""The observation fallbacks must only ever admit, never silently change a fit."""

from __future__ import annotations

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as search


def test_an_unsupported_camera_frame_is_refused_without_the_fallback():
    attempt = {
        "events": [{"event_type": "contact", "frame": 1.0, "frame_interval": [1, 1]}],
        "owner_end_frame": 3.0,
        "point_clip": "pt0001",
        "match_id": "m",
        "owner_ball_labels": [
            {"frame": 1, "status": "visible", "x1080": 1.0, "y1080": 2.0},
            {"frame": 2, "status": "visible", "x1080": 3.0, "y1080": 4.0},
        ],
    }
    cameras = {
        "clip": "pt0001",
        "match_id": "m",
        "cameras": [
            {"frame": 1, "status": "supported"},
            {"frame": 2, "status": "held"},
        ],
    }
    with pytest.raises(ValueError, match="one supported matching camera"):
        search.prepare_attempt(attempt, cameras, "hard")


def test_the_fallback_omits_that_frame_and_records_it():
    attempt = {
        "events": [{"event_type": "contact", "frame": 1.0, "frame_interval": [1, 1]}],
        "owner_end_frame": 3.0,
        "point_clip": "pt0001",
        "match_id": "m",
        "owner_ball_labels": [
            {"frame": 1, "status": "visible", "x1080": 1.0, "y1080": 2.0},
            {"frame": 2, "status": "visible", "x1080": 3.0, "y1080": 4.0},
        ],
    }
    cameras = {
        "clip": "pt0001",
        "match_id": "m",
        "cameras": [
            {"frame": 1, "status": "supported"},
            {"frame": 2, "status": "held"},
        ],
    }
    receipt: list[dict] = []
    with pytest.raises(ValueError):
        # The scene is still degenerate here; what matters is that the refusal is
        # no longer the camera one and that the omission was recorded.
        search.prepare_attempt(
            attempt, cameras, "hard", observation_fallback=True, fallback_receipt=receipt
        )
    assert receipt[0]["fallback"] == "unsupported_camera_observations_omitted"
    assert receipt[0]["frames"] == [2]


def test_a_contact_bracket_widens_only_under_the_fallback():
    event = {"frame": 10.0, "frame_interval": [10, 10]}
    labels = {4: np.asarray([1.0, 2.0])}
    with pytest.raises(ValueError, match="nearby visible ball front"):
        search.contact_association_pixel(event, labels)
    receipt: list[dict] = []
    found = search.contact_association_pixel(
        event, labels, observation_fallback=True, fallback_receipt=receipt
    )
    assert found.tolist() == [1.0, 2.0]
    assert receipt[0]["fallback"] == "contact_association_widened_window"


def test_a_distant_front_is_taken_with_its_distance_recorded():
    """Side association is worth more than the attempt; the reach is reported."""
    event = {"frame": 10.0, "frame_interval": [10, 10]}
    receipt: list[dict] = []
    found = search.contact_association_pixel(
        event,
        {100: np.asarray([1.0, 2.0])},
        observation_fallback=True,
        fallback_receipt=receipt,
    )
    assert found.tolist() == [1.0, 2.0]
    assert receipt[0]["nearest_front_frame_distance"] == 90


def test_a_widened_window_still_refuses_when_the_attempt_has_no_front():
    event = {"frame": 10.0, "frame_interval": [10, 10]}
    with pytest.raises(ValueError, match="nearby visible ball front"):
        search.contact_association_pixel({**event}, {}, observation_fallback=True)


def test_a_bounce_bracket_with_no_endpoint_front_widens_only_under_the_fallback():
    camera = np.asarray(
        [[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, -1000.0], [0.0, 0.0, 1.0, 20.0]]
    )
    cameras = {frame: camera for frame in (8, 9, 12, 13)}
    labels = {frame: np.asarray([960.0, 700.0]) for frame in (8, 9, 12, 13)}
    event = {"frame": 10.5, "frame_interval": [10, 11]}
    with pytest.raises(ValueError, match="no visible native ground ray"):
        search.event_ground_target(event, cameras, labels)
    receipt: list[dict] = []
    target = search.event_ground_target(
        event, cameras, labels, observation_fallback=True, fallback_receipt=receipt
    )
    assert receipt[0]["fallback"] == "bounce_ground_ray_widened_bracket"
    assert np.isfinite(target["xyz_m"]).all()


def test_the_dense_supplement_never_overwrites_a_visible_base_row():
    attempt = {
        "owner_ball_labels": [
            {"frame": 1, "status": "visible", "x1080": 1.0, "y1080": 1.0},
            {"frame": 2, "status": "ambiguous", "x1080": None, "y1080": None},
        ]
    }
    dense = {
        "records": [
            {
                "frames": [
                    {"frame": 1, "front": {"status": "visible", "x1080": 9.0, "y1080": 9.0}},
                    {
                        "frame": 2,
                        "front": {
                            "status": "visible",
                            "x1080": 5.0,
                            "y1080": 6.0,
                            "uncertainty_radius_px1080": 4.0,
                        },
                    },
                ]
            }
        ]
    }
    receipt = search.supplement_with_dense_labels(attempt, dense)
    assert receipt["frames_filled"] == [2]
    assert attempt["owner_ball_labels"][0]["x1080"] == 1.0
    assert attempt["owner_ball_labels"][1]["x1080"] == 5.0
    assert attempt["owner_ball_labels"][1]["annotation_origin"] == "agent_dense_revision"


def test_the_player_ledger_skips_its_own_blank_abstentions(tmp_path):
    path = tmp_path / "state.csv"
    path.write_text(
        "clip,frame,side,court_x,court_y,court_sigma_m,position_source\n"
        "pt0001,10,near,1.5,2.5,0.65,sided_box_bottom_center\n"
        "pt0001,11,near,,,0.65,sided_box_bottom_center\n"
        "pt0002,10,near,9.9,9.9,0.65,sided_box_bottom_center\n"
    )
    rows = search.player_ledger_positions(path, "pt0001")
    assert set(rows) == {(10, "near")}
    players = [{"side": "near", "court_centre_xy_m": np.asarray([0.0, 0.0])}]
    receipt = search.apply_player_ledger(players, [{"frame": 9.5}], rows)
    assert players[0]["court_centre_xy_m"].tolist() == [1.5, 2.5]
    assert players[0]["court_position_source"] == "tennis_player_state_v1"
    assert receipt["substituted"][0]["frame"] == 10
    assert not receipt["abstained"]
