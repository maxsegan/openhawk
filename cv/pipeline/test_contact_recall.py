"""Tests for the label-free contact-recovery emitter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cv.pipeline import contact_recall as cr


# --------------------------------------------------------------------------
# a synthetic camera
# --------------------------------------------------------------------------


def camera() -> np.ndarray:
    """A simple pinhole looking down the court from behind the near baseline."""

    focal = 1200.0
    centre = (960.0, 540.0)
    # world (x, y, z) -> camera (x, -z, y) with the camera 12 m up and 6 m back
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    translation = np.asarray([0.0, 12.0, -6.0], dtype=float)
    intrinsics = np.asarray(
        [[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]], dtype=float
    )
    extrinsics = np.hstack([rotation, translation.reshape(3, 1)])
    return intrinsics @ extrinsics


def project(matrix: np.ndarray, point: tuple[float, float, float]) -> tuple[float, float]:
    homogeneous = matrix @ np.asarray([*point, 1.0], dtype=float)
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


# --------------------------------------------------------------------------
# turn profile and corners
# --------------------------------------------------------------------------


def test_turn_profile_is_zero_on_a_straight_track():
    track = {frame: (10.0 * frame, 5.0 * frame) for frame in range(10)}
    profile = cr.turn_profile(track)
    assert set(profile) == set(range(1, 9))
    assert all(turn == pytest.approx(0.0, abs=1e-6) for turn, _, _ in profile.values())


def test_turn_profile_measures_a_right_angle():
    track = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (10.0, 10.0)}
    assert cr.turn_profile(track)[1][0] == pytest.approx(90.0)


def test_turn_profile_skips_frames_with_a_missing_neighbour():
    track = {0: (0.0, 0.0), 1: (10.0, 0.0), 3: (30.0, 0.0)}
    assert cr.turn_profile(track) == {}


def test_turn_profile_ignores_a_stationary_step():
    track = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (10.0, 10.0)}
    assert 1 not in cr.turn_profile(track)


def test_corners_finds_the_reversal_and_nothing_else():
    track = {frame: (10.0 * frame, 0.0) for frame in range(6)}
    for frame in range(6, 12):
        track[frame] = (10.0 * (10 - frame), 0.0)
    found = cr.corners(track, min_deg=45.0, min_step_px=1.0)
    assert [corner.frame for corner in found] == [5]
    assert found[0].turn_deg == pytest.approx(180.0)


def test_corners_suppresses_neighbouring_maxima():
    track = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (10.0, 10.0), 3: (0.0, 10.0), 4: (0.0, 0.0)}
    assert len(cr.corners(track, min_deg=45.0, min_step_px=1.0, suppress=3)) == 1
    assert len(cr.corners(track, min_deg=45.0, min_step_px=1.0, suppress=0)) == 3


def test_corners_rejects_a_sub_pixel_turn():
    track = {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (0.5, 0.5)}
    assert cr.corners(track, min_deg=45.0, min_step_px=1.0) == []
    assert len(cr.corners(track, min_deg=45.0, min_step_px=0.1)) == 1


def test_corners_accepts_the_track_row_shapes_the_refiner_accepts():
    rows = {0: {"x": 0.0, "y": 0.0}, 1: {"x": 10.0, "y": 0.0}, 2: {"x": 10.0, "y": 10.0}}
    assert [corner.frame for corner in cr.corners(rows, min_deg=45.0, min_step_px=1.0)] == [1]


# --------------------------------------------------------------------------
# court geometry
# --------------------------------------------------------------------------


def test_court_point_at_height_inverts_the_projection():
    matrix = camera()
    for point in ((0.0, 5.0, 0.0), (-3.0, 18.0, 1.5), (4.0, 2.0, 2.5)):
        pixel = project(matrix, point)
        solved = cr.court_point_at_height(matrix, pixel, point[2])
        assert solved is not None
        assert solved[0] == pytest.approx(point[0], abs=1e-6)
        assert solved[1] == pytest.approx(point[1], abs=1e-6)


def test_court_point_at_height_refuses_a_broken_matrix():
    assert cr.court_point_at_height(np.zeros((3, 4)), (10.0, 10.0), 0.0) is None
    assert cr.court_point_at_height(np.eye(3), (10.0, 10.0), 0.0) is None


def test_height_above_court_recovers_a_contact_height():
    matrix = camera()
    ground = (1.0, 3.0)
    pixel = project(matrix, (ground[0], ground[1], 1.6))
    solved = cr.height_above_court(matrix, pixel, ground)
    assert solved is not None
    height, miss = solved
    assert height == pytest.approx(1.6, abs=1e-3)
    assert miss == pytest.approx(0.0, abs=1e-3)


def test_height_above_court_is_zero_for_a_ball_on_the_ground():
    matrix = camera()
    ground = (-2.0, 8.0)
    pixel = project(matrix, (ground[0], ground[1], 0.0))
    height, miss = cr.height_above_court(matrix, pixel, ground)
    assert height == pytest.approx(0.0, abs=1e-3)
    assert miss == pytest.approx(0.0, abs=1e-3)


def test_height_above_court_reports_the_miss_when_the_ray_is_elsewhere():
    matrix = camera()
    pixel = project(matrix, (4.0, 3.0, 1.5))
    height, miss = cr.height_above_court(matrix, pixel, (-4.0, 3.0))
    assert miss > 1.0


def test_in_court_uses_the_doubles_rectangle():
    assert cr.in_court((0.0, 12.0))
    assert cr.in_court((5.4, 0.5))
    assert not cr.in_court((9.0, 12.0))
    assert not cr.in_court((0.0, 30.0))
    assert not cr.in_court(None)


def test_court_side_splits_at_the_net():
    matrix = camera()
    assert cr.court_side(matrix, project(matrix, (0.0, 4.0, 0.0))) == "near"
    assert cr.court_side(matrix, project(matrix, (0.0, 20.0, 0.0))) == "far"
    assert cr.court_side(None, (10.0, 10.0)) == "unknown"


# --------------------------------------------------------------------------
# player proximity
# --------------------------------------------------------------------------


def player_row(x0, y0, x1, y1, court=(0.0, 3.0), wrists=()):
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "side": "near",
        "court_xy": court,
        "wrists": list(wrists),
    }


def test_nearest_player_measures_the_box_edge_and_the_wrist():
    rows = [player_row(0.0, 0.0, 100.0, 200.0, wrists=[(120.0, 40.0)])]
    witness = cr.nearest_player(rows, (140.0, 100.0))
    assert witness is not None
    assert witness.distance_px == pytest.approx(40.0)
    assert witness.wrist_px == pytest.approx(math.hypot(20.0, 60.0))


def test_nearest_player_returns_zero_inside_the_box():
    witness = cr.nearest_player([player_row(0.0, 0.0, 100.0, 200.0)], (50.0, 50.0))
    assert witness.distance_px == pytest.approx(0.0)


def test_nearest_player_takes_the_closer_of_two():
    rows = [player_row(0.0, 0.0, 10.0, 10.0), player_row(200.0, 0.0, 210.0, 10.0)]
    assert cr.nearest_player(rows, (205.0, 5.0)).distance_px == pytest.approx(0.0)


def test_nearest_player_is_none_without_rows():
    assert cr.nearest_player([], (0.0, 0.0)) is None


def test_nearest_player_skips_a_malformed_row():
    assert cr.nearest_player([{"x0": 1.0}], (0.0, 0.0)) is None


# --------------------------------------------------------------------------
# typing
# --------------------------------------------------------------------------


def contact_corner(matrix, ground, height):
    pixel = project(matrix, (ground[0], ground[1], height))
    return cr.Corner(frame=10, turn_deg=150.0, step_in_px=20.0, step_out_px=20.0, pixel=pixel)


def test_type_corner_calls_a_racket_height_ray_a_contact():
    matrix = camera()
    ground = (1.0, 4.0)
    corner = contact_corner(matrix, ground, 1.5)
    rows = [
        player_row(
            corner.pixel[0] - 40,
            corner.pixel[1] - 40,
            corner.pixel[0] + 40,
            corner.pixel[1] + 400,
            court=ground,
        )
    ]
    typed = cr.type_corner(corner, projection=matrix, player_rows=rows)
    assert typed.event_type == "contact"
    assert typed.height_m == pytest.approx(1.5, abs=1e-3)


def test_type_corner_calls_court_level_beside_a_player_a_bounce():
    matrix = camera()
    ground = (1.0, 4.0)
    corner = contact_corner(matrix, ground, 0.0)
    rows = [
        player_row(
            corner.pixel[0] - 40,
            corner.pixel[1] - 400,
            corner.pixel[0] + 40,
            corner.pixel[1] + 40,
            court=ground,
        )
    ]
    typed = cr.type_corner(corner, projection=matrix, player_rows=rows)
    assert typed.event_type == "bounce"
    assert typed.reason == "ball_at_court_level_beside_a_player"


def test_type_corner_calls_an_empty_court_corner_a_bounce():
    matrix = camera()
    corner = contact_corner(matrix, (0.0, 16.0), 0.0)
    typed = cr.type_corner(corner, projection=matrix, player_rows=[])
    assert typed.event_type == "bounce"
    assert typed.reason == "court_plane_corner_with_no_player"


def test_type_corner_refuses_a_corner_off_the_court_with_no_player():
    matrix = camera()
    corner = contact_corner(matrix, (20.0, 40.0), 0.0)
    typed = cr.type_corner(corner, projection=matrix, player_rows=[])
    assert typed.event_type is None


def test_type_corner_refuses_without_a_camera():
    corner = cr.Corner(frame=1, turn_deg=170.0, step_in_px=9.0, step_out_px=9.0, pixel=(5.0, 5.0))
    typed = cr.type_corner(corner, projection=None, player_rows=[])
    assert typed.event_type is None
    assert typed.reason == "no_camera"


def test_type_corner_carries_the_audio_onset_through():
    corner = cr.Corner(frame=1, turn_deg=170.0, step_in_px=9.0, step_out_px=9.0, pixel=(5.0, 5.0))
    assert cr.type_corner(corner, projection=None, player_rows=[], audio_z=7.5).audio_z == 7.5


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------


def scene(height_m: float = 1.5):
    """A track that reverses at frame 10 next to a player, with a camera."""

    matrix = camera()
    ground = (1.0, 4.0)
    pixel = project(matrix, (ground[0], ground[1], height_m))
    track = {}
    for offset in range(-6, 0):
        track[10 + offset] = (pixel[0] + 25.0 * offset, pixel[1] + 12.0 * offset)
    track[10] = pixel
    for offset in range(1, 7):
        track[10 + offset] = (pixel[0] - 25.0 * offset, pixel[1] - 12.0 * offset)
    box = [
        player_row(
            pixel[0] - 60.0,
            pixel[1] - 60.0,
            pixel[0] + 60.0,
            pixel[1] + 500.0,
            court=ground,
        )
    ]
    players = {frame: box for frame in track}
    projections = {frame: matrix for frame in track}
    return track, players, projections


def test_emit_reproduces_the_model_when_nothing_is_rescued():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.999}]
    rows = cr.emit(track, paths, players, {}, projections=projections)
    assert [(row.frame, row.event_type, row.source) for row in rows] == [(10.0, "contact", "model")]
    assert rows[0].witnesses[0] == "event_model"
    assert rows[0].confidence == pytest.approx(0.999)


def test_emit_drops_a_below_threshold_row_with_no_corner():
    track = {frame: (10.0 * frame, 0.0) for frame in range(20)}
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.95}]
    assert cr.emit(track, paths, {}, {}) == []


def test_emit_rescues_a_below_threshold_row_the_corner_witnesses():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.95}]
    rows = cr.emit(track, paths, players, {}, projections=projections)
    assert len(rows) == 1
    assert rows[0].source == "model+corner"
    assert rows[0].witnesses == ["event_model", "track_corner"]
    assert rows[0].reason == "corner_rescued_below_threshold"
    assert 0.95 < rows[0].confidence < cr.SHIPPED_MARGINAL


def test_emit_does_not_rescue_below_the_rescue_bar():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.10}]
    assert cr.emit(track, paths, players, {}, projections=projections) == []


def test_emit_needs_the_corner_type_to_match_unless_told_otherwise():
    track, players, projections = scene(height_m=0.0)
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.95}]
    assert cr.emit(track, paths, players, {}, projections=projections) == []
    rescued = cr.emit(
        track, paths, players, {}, projections=projections, corner_type_must_match=False
    )
    assert len(rescued) == 1


def test_emit_can_require_the_audio_onset_for_a_rescue():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.95}]
    assert (
        cr.emit(
            track,
            paths,
            players,
            {},
            projections=projections,
            require_audio_for_rescue=True,
        )
        == []
    )
    heard = cr.emit(
        track,
        paths,
        players,
        {10: 9.0},
        projections=projections,
        require_audio_for_rescue=True,
    )
    assert len(heard) == 1
    assert "audio_onset" in heard[0].witnesses


def test_emit_records_the_audio_witness_without_letting_it_decide():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.999}]
    quiet = cr.emit(track, paths, players, {10: 1.0}, projections=projections)
    loud = cr.emit(track, paths, players, {10: 9.0}, projections=projections)
    assert quiet[0].witnesses == ["event_model", "track_corner"]
    assert loud[0].witnesses == ["event_model", "track_corner", "audio_onset"]
    assert quiet[0].frame == loud[0].frame


def test_emit_can_add_corner_only_rows():
    track, players, projections = scene()
    rows = cr.emit(track, [], players, {}, projections=projections, include_corner_only=True)
    assert [(row.source, row.event_type) for row in rows] == [("corner", "contact")]
    assert rows[0].witnesses == ["track_corner"]


def test_emit_does_not_duplicate_a_corner_a_model_row_already_used():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.999}]
    rows = cr.emit(track, paths, players, {}, projections=projections, include_corner_only=True)
    assert len(rows) == 1
    assert rows[0].source == "model"


def test_emit_without_a_camera_keeps_only_the_model_rows():
    track, players, _ = scene()
    paths = [
        {"event_type": "contact", "frame": 10.0, "path_marginal": 0.95},
        {"event_type": "bounce", "frame": 30.0, "path_marginal": 0.999},
    ]
    rows = cr.emit(track, paths, players, {})
    assert [(row.frame, row.source) for row in rows] == [(30.0, "model")]


def test_emit_ignores_rows_that_are_not_physical_events():
    track, players, projections = scene()
    paths = [{"event_type": "point_end", "frame": 10.0, "path_marginal": 1.0}]
    assert cr.emit(track, paths, players, {}, projections=projections) == []


def test_emit_returns_rows_in_frame_order():
    track, players, projections = scene()
    paths = [
        {"event_type": "contact", "frame": 10.0, "path_marginal": 0.999},
        {"event_type": "bounce", "frame": 4.0, "path_marginal": 0.999},
    ]
    rows = cr.emit(track, paths, players, {}, projections=projections)
    assert [row.frame for row in rows] == [4.0, 10.0]


def test_emission_serialises():
    track, players, projections = scene()
    paths = [{"event_type": "contact", "frame": 10.0, "path_marginal": 0.999}]
    payload = cr.emit(track, paths, players, {}, projections=projections)[0].as_dict()
    assert payload["event_type"] == "contact"
    assert payload["witnesses"] == ["event_model", "track_corner"]


def test_the_fitted_constants_are_all_named():
    for name in cr.FITTED_ON_DEVELOPMENT:
        assert isinstance(getattr(cr, name), (int, float))
