"""Whole-point synthetic closed loop: geometry, corruption, contract and scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import anchor_first_fit
from cv.validation import s6_point_bench as bench


def test_generation_search_submits_only_a_bounded_batch_and_preserves_order():
    class Pool:
        def __init__(self):
            self.submitted = []

        def map(self, function, tasks, chunksize):
            assert chunksize == 1
            self.submitted.extend(tasks)
            return map(function, tasks)

    pool = Pool()
    results = bench.bounded_ordered_map(pool, lambda number: number * 2, list(range(100)), 4)
    assert next(results) == 0
    assert next(results) == 2
    results.close()
    assert pool.submitted == [0, 1, 2, 3]
    pool = Pool()
    assert list(bench.bounded_ordered_map(pool, lambda number: number, list(range(9)), 4)) == list(
        range(9)
    )
    assert pool.submitted == list(range(9))


def test_corpus_axes_map_onto_the_pipeline_court():
    """Corpus x is length with the net at zero; the pipeline's y is length with the net at
    11.885 and its x is width measured from the sideline."""
    assert np.allclose(
        bench._to_pipeline((0.0, 0.0, 1.0)), [bench.COURT_WIDTH_M / 2.0, 11.885, 1.0]
    )
    baseline = bench._to_pipeline((-11.885, -5.485, 0.0))
    assert baseline[0] == pytest.approx(0.0)
    assert baseline[1] == pytest.approx(0.0)


def test_integrate_arc_stops_one_step_past_the_court_plane():
    times, positions, velocities, _ = bench.integrate_arc(
        np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 0.0]), np.zeros(3), max_seconds=5.0
    )
    assert positions[-1][2] <= bench.BALL_RADIUS_M
    assert velocities[-1][2] < 0.0
    assert times[-1] < 1.0


def test_descending_crossing_finds_the_landing_time():
    times, positions, _, _ = bench.integrate_arc(
        np.array([0.0, 0.0, 4.905]), np.array([1.0, 0.0, 0.0]), np.zeros(3), max_seconds=3.0
    )
    ground = bench._descending_crossing(times, positions, bench.BALL_RADIUS_M)
    assert ground is not None
    # Drag is small at 1 m/s, so the free-fall time is the right order.
    assert ground == pytest.approx(0.99, abs=0.05)


def test_height_crossings_returns_both_sides_of_an_arc():
    times, positions, _, _ = bench.integrate_arc(
        np.array([0.0, 0.0, 0.0325]), np.array([8.0, 0.0, 6.0]), np.zeros(3), max_seconds=2.0
    )
    crossings = bench._height_crossings(times, positions, 1.0)
    assert len(crossings) == 2
    assert crossings[0] < crossings[1]


def test_plane_time_finds_the_net_crossing():
    times, positions, _, _ = bench.integrate_arc(
        np.array([5.0, 2.0, 2.5]), np.array([0.0, 25.0, 1.0]), np.zeros(3), max_seconds=2.0
    )
    crossing = bench._plane_time(times, positions, 1, bench.NET_Y_M)
    assert crossing is not None
    point = bench._interp_rows(crossing, times, positions)
    assert point[1] == pytest.approx(bench.NET_Y_M, abs=1e-3)


def test_solve_shot_lands_on_the_requested_bounce_spot():
    start = np.array([5.0, 1.0, 2.6])
    target = (6.2, 17.5)
    shot = bench.solve_shot(start, target, 1.4, 2000.0)
    assert shot is not None
    times, positions, _, _ = bench.integrate_arc(
        start, shot["velocity"], shot["spin"], max_seconds=3.0
    )
    ground = bench._descending_crossing(times, positions, bench.BALL_RADIUS_M)
    landing = bench._interp_rows(ground, times, positions)
    assert landing[0] == pytest.approx(target[0], abs=0.30)
    assert landing[1] == pytest.approx(target[1], abs=0.30)


def test_solve_shot_refuses_a_target_under_the_racket():
    assert bench.solve_shot(np.array([5.0, 1.0, 2.6]), (5.1, 2.0), 1.0, 2000.0) is None


def test_simulated_point_keeps_hidden_physical_end_and_applies_net_screen(monkeypatch):
    projection = np.array([[30.0, 0, 0, 700], [0, 15, -10, 300], [0, 0, 0, 1]])
    host = bench.HostPoint(
        "unit",
        "pt0001",
        "hard",
        25.0,
        (1.0, 1000.0),
        {"frames": np.arange(1, 1001), "P": np.repeat(projection[None], 1000, axis=0)},
        np.eye(3),
        {},
    )
    skeleton = bench.Skeleton(
        "unit",
        "rally",
        "hard",
        [
            bench.SkeletonStrike(
                np.array([5.0, 1.0, 2.6]),
                np.array([5.5, 11.885, 1.4]),
                np.array([6.2, 17.5, 0.0325]),
                3.0,
                2000.0,
                None,
            ),
            bench.SkeletonStrike(
                np.array([6.5, 20.0, 1.0]),
                np.array([5.5, 11.885, 1.4]),
                np.array([5.0, 4.0, 0.0325]),
                3.0,
                2000.0,
                None,
            ),
        ],
    )
    complete = bench.simulate_rally(host, skeleton, 2, np.random.default_rng(1))
    assert isinstance(complete, dict), complete
    assert complete["termination_kind"] == "second_bounce"
    # Same flight, but the camera no longer sees its last two native exposures.
    exit_frame = math.floor(complete["termination_frame"]) - 1
    host.camera_rows["P"][host.camera_rows["frames"] >= exit_frame, 0, 3] += 3000
    hidden = bench.simulate_rally(host, skeleton, 2, np.random.default_rng(1))
    assert isinstance(hidden, dict), hidden
    assert hidden["termination_kind"] == "second_bounce"
    assert hidden["termination_frame"] == complete["termination_frame"]
    assert hidden["termination_xyz"] == complete["termination_xyz"]
    assert hidden["observation_exit_frame"] == exit_frame
    assert all(f < exit_frame for f in hidden["frames"])
    assert hidden["trajectory_truth"] == complete["trajectory_truth"]
    monkeypatch.setattr(bench, "net_clearances", lambda *_: [{"ball_surface_clearance_m": -0.1}])
    assert (
        bench.simulate_rally(host, skeleton, 2, np.random.default_rng(1))
        == "net_intersection_unsupported"
    )


@pytest.mark.parametrize("hit,bounce", [(2.0, 11.606), (24.641, 12.185), (2.0, math.nan)])
def test_skeleton_cannot_play_through_a_same_half_first_bounce(hit, bounce):
    assert not bench.bounce_in_receiving_half([5.0, hit, 1.0], [5.0, bounce, 0.0325])


@pytest.mark.parametrize("hit,bounce", [(2.0, 17.0), (24.0, 5.0)])
def test_supported_skeleton_bounce_lands_in_receiving_half(hit, bounce):
    assert bench.bounce_in_receiving_half([5.0, hit, 1.0], [5.0, bounce, 0.0325])


def _point() -> dict:
    frames = list(range(100, 140))
    return {
        "point": "syn__pt0001",
        "match_id": "syn",
        "clip": "pt0001",
        "host": "syn__pt0001",
        "surface": "hard",
        "fps": 25.0,
        "shots": 2,
        "terminal_bounces": 0,
        "frames": frames,
        "track_pixels": {frame: [900.0 + frame, 400.0] for frame in frames},
        "positions": {frame: [5.0, 2.0 + 0.4 * (frame - 100), 1.0] for frame in frames},
        "contacts": [
            {"frame": 100.0, "xyz": [5.0, 2.0, 1.0], "phase": "serve", "pixel": [1000.0, 400.0]},
            {"frame": 120.0, "xyz": [5.0, 10.0, 1.0], "phase": "rally", "pixel": [1000.0, 400.0]},
        ],
        "bounces": [
            {"frame": 110.0, "xyz": [5.0, 6.0, 0.0325], "pixel": [1010.0, 400.0]},
            {"frame": 132.0, "xyz": [5.0, 25.4, 0.0325], "pixel": [1032.0, 400.0]},
        ],
        "termination_frame": 132.0,
        "termination_kind": "terminal_bounce",
        "termination_xyz": [5.0, 25.4, 0.0325],
        "termination_pixel": [1032.0, 400.0],
        "first_bounce_in_bounds": False,
        "flights": [
            {
                "flight_index": 0,
                "start_frame": 100.0,
                "end_frame": 120.0,
                "terminal": False,
                "bounces": [{"frame": 110.0, "xyz": [5.0, 6.0, 0.0325], "pixel": [1010.0, 400.0]}],
            },
            {
                "flight_index": 1,
                "start_frame": 120.0,
                "end_frame": 139.0,
                "terminal": True,
                "bounces": [{"frame": 132.0, "xyz": [5.0, 25.4, 0.0325], "pixel": [1032.0, 400.0]}],
                "termination_frame": 132.0,
                "termination_kind": "terminal_bounce",
                "termination_xyz": [5.0, 25.4, 0.0325],
                "first_bounce_in_bounds": False,
            },
        ],
    }


class _Samplers:
    """Deterministic stand-in for the s6_bench empirical samplers."""

    def sample_track_errors(self, rng, sides, contact_only=False):
        offset = 5.0 if contact_only else 1.0
        return np.full((len(sides), 2), offset), np.full(len(sides), 4.0)

    def sample_wrong_object_arcs(self, rng, count):
        return [(2, np.full((3, 2), 50.0))]

    def sample_dropout_mask(self, rng, count):
        mask = np.zeros(count, dtype=bool)
        mask[5:8] = True
        return mask

    def sample_camera_jitter(self, rng, fallback):
        return np.full((len(fallback), 2), 0.5)

    def sample_event_timing(self, rng, count):
        return np.full(count, 1.0)

    def sample_anchor_pattern(self, rng):
        return ["net_crossing"]


def test_clean_rung_is_the_exact_truth():
    point = bench.apply_rung(_point(), "clean", _Samplers(), np.random.default_rng(0))
    assert len(point["track"]) == len(point["frames"])
    assert point["track"][100][0] == pytest.approx(1000.0)
    assert [row["frame"] for row in point["emission_contacts"]] == [100.0, 120.0]
    assert [row["frame"] for row in point["emission_bounces"]] == [110.0, 132.0]


def test_track_error_rung_moves_pixels_and_smears_contacts():
    point = bench.apply_rung(_point(), "track_error", _Samplers(), np.random.default_rng(0))
    assert point["track"][105][0] == pytest.approx(1006.0)
    assert point["track"][100][0] == pytest.approx(1005.0)


def test_event_timing_rung_shifts_only_the_emissions():
    point = bench.apply_rung(_point(), "event_timing", _Samplers(), np.random.default_rng(0))
    assert [row["frame"] for row in point["emission_contacts"]] == [101.0, 121.0]
    assert [row["frame"] for row in point["emission_bounces"]] == [111.0, 133.0]
    assert point["track"][100][0] == pytest.approx(1000.0)


def test_missing_anchor_rung_removes_bounce_emissions_per_flight():
    point = bench.apply_rung(_point(), "missing_anchors", _Samplers(), np.random.default_rng(0))
    assert point["emission_bounces"] == []
    assert len(point["track"]) == len(point["frames"])


def test_realistic_rung_drops_frames():
    point = bench.apply_rung(_point(), "realistic", _Samplers(), np.random.default_rng(0))
    assert len(point["track"]) == len(point["frames"]) - 3


def test_net_crossing_frames_are_interpolated():
    point = _point()
    crossings = bench._net_crossing_frames(
        {
            **point,
            "positions": {
                frame: [5.0, 11.0 + 0.5 * (frame - 100), 1.0] for frame in point["frames"]
            },
        }
    )
    assert crossings and crossings[0] == pytest.approx(101.77, abs=0.05)


def test_nightly_argument_contract_is_the_shipped_one():
    """The nightly evaluator holds the machine-readable default 3D command.

    docs/wk1/nightly.md states it in prose, so the guard reads the evaluator's own command
    list: drift there would silently change which arm this bench measures.
    """
    nightly = (Path(bench.__file__).resolve().parent / "wk3_nightly.py").read_text()
    for flag in (
        "--include-dead-time-emissions",
        "--terminal-flights",
        "--anchor-bounce-geometry",
        "--whole-point-branches",
        "--max-nfev",
        "--branch-width",
        "--branch-margin",
    ):
        assert flag in nightly, flag
        assert flag in bench.NIGHTLY_ARGUMENTS, flag
    assert "--net-plane-constraint" not in bench.NIGHTLY_ARGUMENTS
    assert bench.ARMS["default"] == ()
    assert bench.ARMS["net_plane_constraint"] == ("--net-plane-constraint",)


def _report(**fit_overrides) -> dict:
    fit = {
        "flight_index": 0,
        "start_frame": 100.0,
        "end_frame": 120.0,
        "held_out_accepted": True,
        "held_out_reprojection_median_px": 2.0,
        "held_out_reprojection_p90_px": 3.0,
        "net_anchor_available": True,
        "start_xyz": [5.0, 2.0, 1.0],
        "end_xyz": [5.0, 10.0, 1.0],
        "bounces": [{"frame": 110.0, "x": [5.0, 6.0, 0.0325]}],
    }
    fit.update(fit_overrides)
    return {
        "points_detail": [
            {
                "point": "syn__pt0001",
                "decision": "retain",
                "reasons": [],
                "junction_gaps_m": [0.4],
                "flight_attempts": [
                    {"flight_index": 0, "start_frame": 100.0, "solved": True},
                    {"flight_index": 1, "start_frame": 120.0, "solved": False},
                ],
                "fits": [fit],
            }
        ]
    }


class _Camera:
    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def p_at(_frame):
        # Orthographic stand-in that sees only court width: x -> 200*x, image y fixed.
        # One metre of court x is 200 px, which is what the depth test relies on.
        return np.array([[200.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 400.0], [0.0, 0.0, 0.0, 1.0]])


def test_audit_scores_a_correct_flight_as_truth_good(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    truth = {"points": [_point()]}
    scored = bench.audit_report(
        truth, _report(), Path("/nonexistent"), criterion=bench.CRITERION_PIXEL_V1
    )
    row = scored["flight_rows"][0]
    assert row["solved"] and row["accepted"]
    assert row["bounce_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["truth_good"] is True
    assert scored["accepted_flights"] == 1
    assert scored["wrong_accepted_flights"] == 0
    assert scored["junction_gap_m"]["median"] == pytest.approx(0.4)


def test_audit_calls_a_depth_error_a_wrong_accept(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    truth = {"points": [_point()]}
    scored = bench.audit_report(truth, _report(end_xyz=[5.4, 10.0, 1.0]), Path("/nonexistent"))
    row = scored["flight_rows"][0]
    assert row["contact_error_px"] == pytest.approx(80.0, abs=1e-6)
    assert row["truth_good"] is False
    assert scored["wrong_accepted_flights"] == 1
    assert scored["complete_points"] == 0


def test_audit_reports_a_missed_bounce_as_infinite(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report({"points": [_point()]}, _report(bounces=[]), Path("/nonexistent"))
    assert math.isinf(scored["flight_rows"][0]["bounce_error_m"])


def test_first_cause_takes_the_earliest_blocking_stage():
    rows = [
        {"attempted": True, "solved": True, "accepted": True, "truth_good": True, "reasons": []},
        {"attempted": True, "solved": False, "accepted": False, "truth_good": False, "reasons": []},
    ]
    assert bench.first_cause({"reasons": []}, rows) == "flight_not_solved"
    assert (
        bench.first_cause({"reasons": ["camera_calibration_unreliable"]}, rows)
        == "camera_abstained"
    )
    rows[1] = {
        "attempted": False,
        "solved": False,
        "accepted": False,
        "truth_good": False,
        "reasons": [],
    }
    assert bench.first_cause({"reasons": []}, rows) == "flight_not_attempted"
    good = [
        {"attempted": True, "solved": True, "accepted": True, "truth_good": True, "reasons": []}
    ]
    assert bench.first_cause({"reasons": []}, good) == "complete"


def test_emission_carries_the_true_projected_pixel():
    point = _point()
    row = bench._emission("syn", point, "bounce", point["bounces"][0])
    assert row["clip"] == "syn__pt0001"
    assert row["abstain"] is False
    assert row["location"]["image_x"] == pytest.approx(1010.0)
    assert row["location"]["image_coordinate_space"] == "native_1920x1080"


def test_written_track_is_half_native_as_the_loader_expects(tmp_path):
    rows = [
        {
            "clip": "pt0001",
            "frame": bench._frame_name(100),
            "x": 500.0,
            "y": 200.0,
            "score": 0.9,
            "sources": "wasb+tracknetv2",
        }
    ]
    bench.write_declared_csv(tmp_path / bench.TRACK_NAME, rows, kind="track", fps=25.0)
    from cv.pipeline.reconstruction import load_track

    loaded = load_track(tmp_path, "pt0001")
    assert np.allclose(loaded[100], [1000.0, 400.0])


def test_json_default_rejects_unknown_objects(tmp_path):
    bench.write_json(tmp_path / "row.json", {"value": np.int64(3), "array": np.zeros(2)})
    assert json.loads((tmp_path / "row.json").read_text())["value"] == 3
    with pytest.raises(TypeError):
        bench._json_default(object())


def test_anchor_availability_is_keyed_by_point_and_start_frame(tmp_path):
    directory = tmp_path / "syn" / "pt0001"
    directory.mkdir(parents=True)
    (directory / "anchors_v1.json").write_text(
        json.dumps(
            {
                "point": "syn__pt0001",
                "flights": [
                    {"start_frame": 100.0, "net_anchor_available": True, "bounce_anchors": 1},
                    {"start_frame": 120.0, "net_anchor_available": False, "bounce_anchors": 2},
                ],
            }
        )
    )
    rows = bench.load_anchor_availability(tmp_path)
    assert rows["syn__pt0001"][120.0] == {"net_anchor_available": False, "bounce_anchors": 2}


def test_anchor_availability_of_a_missing_directory_is_empty(tmp_path):
    assert bench.load_anchor_availability(tmp_path / "absent") == {}


def test_audit_reports_the_anchors_of_an_unsolved_flight(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    directory = tmp_path / "syn" / "pt0001"
    directory.mkdir(parents=True)
    (directory / "anchors_v1.json").write_text(
        json.dumps(
            {
                "point": "syn__pt0001",
                "flights": [
                    {"start_frame": 100.0, "net_anchor_available": True, "bounce_anchors": 1},
                    {"start_frame": 120.0, "net_anchor_available": False, "bounce_anchors": 2},
                ],
            }
        )
    )
    scored = bench.audit_report({"points": [_point()]}, _report(), Path("/nonexistent"), tmp_path)
    assert scored["unsolved_flight_anchors"] == {"net=False,bounces=2": 1}
    assert scored["unsolved_flight_terminal_share"] == {"terminal": 1, "rally": 0}
    assert scored["by_flight_kind"]["rally"]["accepted"] == 1
    assert scored["by_flight_kind"]["terminal"]["solved"] == 0


# --------------------------------------------------------------------------------------- #
# The owner's truth definition: a flight ends at the next shot; the terminal flight ends at
# the second bounce when the first is in, at the first bounce when it is out, or at the frame
# the ball leaves the field of view.  Dead ball after that is never truth.
# --------------------------------------------------------------------------------------- #
IN_BOUNCE = {"frame": 132.0, "xyz": [5.0, 14.8, 0.0325], "pixel": [1032.0, 400.0]}
SECOND_BOUNCE = {"frame": 148.0, "xyz": [5.0, 19.0, 0.0325], "pixel": [1048.0, 400.0]}
DEAD_BOUNCE = {"frame": 156.0, "xyz": [5.0, 22.0, 0.0325], "pixel": [1056.0, 400.0]}
OUT_BOUNCE = {"frame": 132.0, "xyz": [5.0, 25.4, 0.0325], "pixel": [1032.0, 400.0]}


def _terminal_point(kind: str = "second_bounce") -> dict:
    """One point whose rally-closing flight terminates the way the owner defines it."""
    point = _point()
    point["source_match"] = "syn"
    point["mode"] = bench.MODE_RALLY
    point["lob"] = False
    point["out_of_frame_frames"] = []
    point["top_exit_frames"] = []
    terminal = point["flights"][1]
    terminal["end_frame"] = 164.0
    if kind == "second_bounce":
        rows = [IN_BOUNCE, SECOND_BOUNCE, DEAD_BOUNCE]
        termination, xyz = 148.0, SECOND_BOUNCE["xyz"]
        terminal["first_bounce_in_bounds"] = True
    elif kind == "terminal_bounce":
        rows = [OUT_BOUNCE, DEAD_BOUNCE]
        termination, xyz = 132.0, OUT_BOUNCE["xyz"]
        terminal["first_bounce_in_bounds"] = False
    else:
        rows = [DEAD_BOUNCE]
        termination, xyz = 128.0, [5.0, 13.2, 4.0]
        terminal["first_bounce_in_bounds"] = False
        point["out_of_frame_frames"] = [128, 129, 130, 131, 132, 133]
        point["top_exit_frames"] = list(point["out_of_frame_frames"])
    point["bounces"] = [point["bounces"][0], *rows]
    terminal["bounces"] = list(rows)
    terminal["termination_frame"] = termination
    terminal["termination_kind"] = kind
    terminal["termination_xyz"] = list(xyz)
    point["termination_frame"] = termination
    point["termination_kind"] = kind
    point["termination_xyz"] = list(xyz)
    point["termination_pixel"] = [1000.0 + termination - 100.0, 400.0]
    point["first_bounce_in_bounds"] = bool(terminal["first_bounce_in_bounds"])
    point["terminal_bounces"] = len(rows)
    point["dead_ball_bounces"] = sum(1 for row in rows if row["frame"] > termination)
    return point


def test_bounce_in_bounds_reads_the_singles_lines():
    assert bench.bounce_in_bounds([5.0, 12.0, 0.0325])
    assert bench.bounce_in_bounds([bench.SINGLES_X_MIN_M, 0.0, 0.0325])
    assert not bench.bounce_in_bounds([0.6, 12.0, 0.0325])  # in the doubles alley
    assert not bench.bounce_in_bounds([5.0, 24.2, 0.0325])  # past the baseline
    assert bench.bounce_in_bounds([5.0, bench.COURT_LENGTH_M, 0.0325])  # on the line


def test_an_out_first_bounce_ends_the_terminal_flight_there():
    frame, kind = bench.termination_of([OUT_BOUNCE, DEAD_BOUNCE], end_frame=164.0)
    assert (frame, kind) == (132.0, "terminal_bounce")


def test_an_in_first_bounce_ends_the_terminal_flight_at_the_second_bounce():
    frame, kind = bench.termination_of([IN_BOUNCE, SECOND_BOUNCE, DEAD_BOUNCE], end_frame=164.0)
    assert (frame, kind) == (148.0, "second_bounce")


def test_an_in_bounce_with_no_second_bounce_has_no_termination():
    frame, kind = bench.termination_of([IN_BOUNCE], end_frame=164.0)
    assert (frame, kind) == (164.0, "span_end")


def test_historical_explicit_visibility_boundary_remains_readable():
    frame, kind = bench.termination_of(
        [IN_BOUNCE, SECOND_BOUNCE],
        end_frame=164.0,
        out_of_frame_frames=range(140, 152),
    )
    assert (frame, kind) == (140.0, "out_of_view")


def test_complete_truth_keeps_ground_ending_separate_from_visibility():
    frame, kind = bench.termination_of([IN_BOUNCE, SECOND_BOUNCE], end_frame=164.0)
    exit_frame = bench._uninterrupted_exit(range(140, 152), frame)
    assert (frame, kind) == (148.0, "second_bounce")
    assert exit_frame == 140.0
    assert "out_of_view" not in bench.COMPLETE_TRUTH_DEFINITION["termination_kinds"]


@pytest.mark.parametrize("height,clear", [(0.8, False), (0.93, False), (1.3, True)])
def test_generator_net_screen_checks_ball_surface_not_only_center(height, clear):
    xyz = np.array([[5.485, 10.0, height], [5.485, 14.0, height]])
    rows = bench.net_clearances(np.array([1.0, 2.0]), xyz)
    assert len(rows) == 1
    assert (rows[0]["ball_surface_clearance_m"] >= 0) == clear


def test_generator_checks_reverse_crossings_and_exact_plane_samples():
    xyz = np.array([[5.485, 14.0, 1.3], [5.485, 11.885, 0.8], [5.485, 10.0, 1.3]])
    rows = bench.net_clearances(np.array([1.0, 2.0, 3.0]), xyz)
    assert len(rows) == 1
    assert rows[0]["frame"] == 2.0
    assert rows[0]["ball_surface_clearance_m"] < 0


def test_around_net_path_is_not_a_net_intersection():
    xyz = np.array([[-2.0, 10.0, 0.5], [-2.0, 14.0, 0.5]])
    assert bench.net_clearances(np.array([1.0, 2.0]), xyz) == []


@pytest.mark.parametrize("frames", [[2.0, 1.0], [1.0, 1.0], [1.0, math.nan]])
def test_net_truth_audit_refuses_invalid_timebase(frames):
    with pytest.raises(ValueError, match="finite ordered"):
        bench.net_clearances(np.array(frames), np.zeros((2, 3)))


def test_a_lob_that_comes_back_still_ends_at_its_bounce():
    """The ball leaves the top edge and returns before the bounce, so the bounce rule wins."""
    frame, kind = bench.termination_of(
        [IN_BOUNCE, SECOND_BOUNCE],
        end_frame=164.0,
        out_of_frame_frames=range(122, 130),
    )
    assert (frame, kind) == (148.0, "second_bounce")


def test_a_flight_that_ends_at_a_shot_keeps_its_whole_span():
    flight = {"terminal": False, "end_frame": 120.0, "bounces": [{"frame": 110.0}]}
    assert bench.terminal_scored_end(flight) == pytest.approx(120.0)


def test_the_terminal_flight_keeps_both_bounces_when_the_first_is_in():
    terminal = _terminal_point("second_bounce")["flights"][1]
    assert [row["frame"] for row in bench.scored_bounces(terminal)] == [132.0, 148.0]
    assert bench.terminal_termination_kind(terminal) == "second_bounce"


def test_an_out_first_bounce_leaves_the_terminal_flight_one_bounce():
    terminal = _terminal_point("terminal_bounce")["flights"][1]
    assert [row["frame"] for row in bench.scored_bounces(terminal)] == [132.0]


def test_a_truth_file_without_a_recorded_termination_is_re_derived():
    """A run made before this definition existed is still re-scorable from its bounces."""
    terminal = _terminal_point("second_bounce")["flights"][1]
    del terminal["termination_frame"]
    del terminal["termination_kind"]
    assert bench.terminal_scored_end(terminal) == pytest.approx(148.0)
    assert bench.terminal_termination_kind(terminal) == "second_bounce"


def _terminal_report(**terminal_overrides) -> dict:
    rally = {
        "flight_index": 0,
        "start_frame": 100.0,
        "end_frame": 120.0,
        "held_out_accepted": True,
        "held_out_reprojection_median_px": 2.0,
        "start_xyz": [5.0, 2.0, 1.0],
        "end_xyz": [5.0, 10.0, 1.0],
        "bounces": [{"frame": 110.0, "x": [5.0, 6.0, 0.0325]}],
    }
    terminal = {
        "flight_index": 1,
        "start_frame": 120.0,
        "end_frame": 148.0,
        "held_out_accepted": True,
        "held_out_reprojection_median_px": 2.0,
        "start_xyz": [5.0, 10.0, 1.0],
        "end_xyz": [5.0, 19.0, 0.0325],
        "bounces": [{"frame": 132.0, "x": [5.0, 14.8, 0.0325]}],
    }
    terminal.update(terminal_overrides)
    return {
        "points_detail": [
            {
                "point": "syn__pt0001",
                "decision": "retain",
                "reasons": [],
                "junction_gaps_m": [],
                "flight_attempts": [
                    {"flight_index": 0, "start_frame": 100.0},
                    {"flight_index": 1, "start_frame": 120.0},
                ],
                "fits": [rally, terminal],
            }
        ]
    }


def test_a_terminal_flight_is_scored_on_both_of_its_bounces(monkeypatch):
    """The first bounce is an ordinary knot; the second is the termination anchor."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    row = scored["flight_rows"][1]
    assert row["terminal"] and row["scored_bounces"] == 2 and row["dead_ball_bounces"] == 1
    assert row["termination_kind"] == "second_bounce"
    assert row["termination_anchor_source"] == "endpoint"
    assert row["bounce_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["termination_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["truth_good"] is True and row["fitted_dead_ball"] is False
    assert scored["two_bounce_terminal_flights"] == 1
    assert scored["complete_points"] == 1 and scored["wrong_accepted_flights"] == 0


def test_missing_the_first_bounce_of_a_two_bounce_terminal_flight_is_wrong(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(bounces=[{"frame": 132.0, "x": [5.0, 16.9, 0.0325]}]),
        Path("/nonexistent"),
    )
    row = scored["flight_rows"][1]
    assert row["bounce_error_m"] == pytest.approx(2.1, abs=1e-6)
    assert row["truth_good"] is False and scored["complete_points"] == 0


def test_stopping_at_the_first_bounce_of_an_in_ball_is_wrong(monkeypatch):
    """The owner's rally-closing flight does not stop at an in-bounds first bounce."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(end_frame=132.0, end_xyz=[5.0, 14.8, 0.0325]),
        Path("/nonexistent"),
    )
    row = scored["flight_rows"][1]
    assert row["termination_anchor_source"] == "missing"
    assert math.isinf(row["bounce_error_m"])
    assert row["truth_good"] is False and scored["complete_points"] == 0


def test_an_out_ball_terminal_flight_stops_at_its_only_bounce(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point("terminal_bounce")]},
        _terminal_report(
            end_frame=132.0,
            end_xyz=[5.0, 25.4, 0.0325],
            bounces=[{"frame": 132.0, "x": [5.0, 25.4, 0.0325]}],
        ),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    row = scored["flight_rows"][1]
    assert row["termination_kind"] == "terminal_bounce"
    assert row["scored_bounces"] == 1 and row["dead_ball_bounces"] == 1
    assert row["termination_anchor_source"] == "record"
    assert row["truth_good"] is True and scored["complete_points"] == 1


def test_fitting_the_dead_ball_is_a_diagnostic_not_a_penalty(monkeypatch):
    """A fit that runs past the termination and gets the dead-ball bounce wrong is still
    truth-good: nothing after the termination is part of any flight."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(
            end_frame=164.0,
            end_xyz=[5.0, 30.0, 0.4],
            bounces=[
                {"frame": 132.0, "x": [5.0, 14.8, 0.0325]},
                {"frame": 148.0, "x": [5.0, 19.0, 0.0325]},
                {"frame": 156.0, "x": [5.0, 40.0, 0.0325]},
            ],
        ),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    row = scored["flight_rows"][1]
    assert row["termination_anchor_source"] == "record"
    assert row["bounce_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["truth_good"] is True and row["fitted_dead_ball"] is True
    assert scored["terminal_flights_fitting_dead_ball"] == 1
    assert scored["complete_points"] == 1


def test_the_termination_anchor_is_scored_from_the_fitted_trajectory(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(
            end_frame=149.0,
            end_xyz=[5.0, 19.5, 0.5],
            trajectory=[
                {"frame": 147.0, "xyz": [5.0, 18.0, 0.6], "velocity": [0.0, 25.0, -14.4]},
                {"frame": 149.0, "xyz": [5.0, 19.4, 0.4]},
            ],
        ),
        Path("/nonexistent"),
    )
    row = scored["flight_rows"][1]
    assert row["termination_anchor_source"] == "endpoint"
    assert row["bounce_error_m"] == pytest.approx(0.0, abs=0.05)
    assert row["fitted_dead_ball"] is False


def test_a_terminal_exit_has_no_bounce_to_score_but_records_its_error(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point("out_of_view")]},
        _terminal_report(end_frame=128.0, end_xyz=[5.0, 13.2, 4.0], bounces=[]),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    row = scored["flight_rows"][1]
    assert row["termination_kind"] == "out_of_view"
    assert row["scored_bounces"] == 0 and row["dead_ball_bounces"] == 1
    assert row["bounce_error_m"] is None
    assert row["termination_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["truth_good"] is True


def test_a_dead_ball_bounce_is_excluded_from_the_rally_flight_too(monkeypatch):
    """Only the terminal flight is trimmed: a contact-to-contact flight keeps every bounce."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]}, _terminal_report(), Path("/nonexistent")
    )
    rally = scored["flight_rows"][0]
    assert rally["scored_bounces"] == 1 and rally["dead_ball_bounces"] == 0
    assert rally["scored_end_frame"] == pytest.approx(120.0)
    assert rally["fitted_dead_ball"] is None and rally["termination_kind"] is None


# --------------------------------------------------------------------------------------- #
# The point ending, emitted in the runner's event format
# --------------------------------------------------------------------------------------- #
def test_a_point_end_row_is_emitted_at_the_termination():
    point = bench.apply_rung(_terminal_point(), "clean", _Samplers(), np.random.default_rng(0))
    row = bench._point_end_emission("syn", point)
    assert row["event_type"] == "point_end"
    assert row["frame"] == pytest.approx(148.0)
    assert row["point_end"]["termination_kind"] == "second_bounce"
    assert row["point_end"]["terminal_event_type"] == "bounce"
    assert row["location"]["image_coordinate_space"] == "native_1920x1080"
    assert row["abstain"] is False


def test_the_point_end_row_rides_on_the_shifted_bounce_emission():
    """Event-timing noise moves the terminating bounce, and the ending moves with it."""
    point = bench.apply_rung(
        _terminal_point(), "event_timing", _Samplers(), np.random.default_rng(0)
    )
    assert point["emission_point_end"]["frame"] == pytest.approx(149.0)


def test_a_terminal_exit_ending_is_clamped_to_the_frame_edge():
    point = bench.apply_rung(
        _terminal_point("out_of_view"), "clean", _Samplers(), np.random.default_rng(0)
    )
    ending = point["emission_point_end"]
    assert ending["kind"] == "out_of_view"
    assert ending["terminal_event_type"] == "track_exit"
    assert ending["frame"] == pytest.approx(128.0)
    assert "clamped" in ending["source"]


def test_the_point_end_row_never_reaches_the_fitter():
    """``reconstruction.py`` keeps only the three physical kinds, so the ending is carried
    through the event file without changing what the fitter is handed."""
    source = (
        Path(bench.__file__).resolve().parent.parent / "pipeline" / "reconstruction.py"
    ).read_text()
    assert 'row.get("event_type") not in {"contact", "bounce", "net_hit"}' in source


# --------------------------------------------------------------------------------------- #
# Terminal exits: the ball leaves the field of view after the last shot
# --------------------------------------------------------------------------------------- #
def test_a_terminal_exit_skeleton_is_aimed_past_the_baseline():
    strike = bench.SkeletonStrike(
        hit=np.array([5.0, 1.0, 1.0]),
        net=np.array([5.0, bench.NET_Y_M, 1.2]),
        bounce=np.array([5.0, 20.0, 0.0325]),
        apex_height_m=2.0,
        spin_rpm=1800.0,
        serve_speed_ms=50.0,
    )
    long_ball = bench.as_terminal_exit(strike, np.random.default_rng(0))
    assert float(long_ball.bounce[1]) > bench.COURT_LENGTH_M
    assert not bench.bounce_in_bounds(long_ball.bounce)
    assert bench.EXIT_NET_CLEARANCE_M[0] <= long_ball.net[2] <= bench.EXIT_NET_CLEARANCE_M[1]
    assert np.allclose(long_ball.hit, strike.hit)


def test_a_terminal_exit_towards_the_near_baseline_is_aimed_the_other_way():
    strike = bench.SkeletonStrike(
        hit=np.array([5.0, 20.0, 1.0]),
        net=np.array([5.0, bench.NET_Y_M, 1.2]),
        bounce=np.array([5.0, 4.0, 0.0325]),
        apex_height_m=2.0,
        spin_rpm=1800.0,
        serve_speed_ms=None,
    )
    long_ball = bench.as_terminal_exit(strike, np.random.default_rng(1))
    assert float(long_ball.bounce[1]) < 0.0
    assert not bench.bounce_in_bounds(long_ball.bounce)


# --------------------------------------------------------------------------------------- #
# Lobs: the ball leaves the frame through the top edge and the track has no rows there
# --------------------------------------------------------------------------------------- #
def test_a_lob_skeleton_raises_the_net_clearance():
    strike = bench.SkeletonStrike(
        hit=np.array([5.0, 1.0, 1.0]),
        net=np.array([5.0, bench.NET_Y_M, 1.2]),
        bounce=np.array([5.0, 20.0, 0.0325]),
        apex_height_m=2.0,
        spin_rpm=1800.0,
        serve_speed_ms=50.0,
    )
    lobbed = bench.as_lob(strike, np.random.default_rng(0))
    assert bench.LOB_NET_CLEARANCE_M[0] <= lobbed.net[2] <= bench.LOB_NET_CLEARANCE_M[1]
    assert np.allclose(lobbed.hit, strike.hit) and np.allclose(lobbed.bounce, strike.bounce)
    assert lobbed.apex_height_m >= lobbed.net[2]
    assert lobbed.serve_speed_ms is None


def test_out_of_frame_is_the_native_frame_the_tracker_sees():
    assert bench._in_image((0.0, 0.0)) and bench._in_image((1919.0, 1079.0))
    assert not bench._in_image((960.0, -1.0))
    assert not bench._in_image((-1.0, 540.0))
    assert not bench._in_image((1920.0, 540.0))


def test_longest_run_counts_consecutive_frames():
    assert bench._longest_run([]) == 0
    assert bench._longest_run([10, 11, 12, 20, 21]) == 3
    assert bench._longest_run([5]) == 1


def test_a_top_edge_gap_is_marked_out_of_frame_in_the_truth(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    point["lob"] = True
    point["out_of_frame_frames"] = [124, 125, 126, 127]
    point["top_exit_frames"] = [124, 125, 126, 127]
    point["flights"][1]["out_of_frame_frames"] = 4
    scored = bench.audit_report(
        {"points": [point]},
        _terminal_report(),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    assert scored["flight_rows"][1]["out_of_frame_frames"] == 4
    assert scored["flights_with_out_of_frame_gap"] == 1
    assert scored["points_with_out_of_frame_gap"] == 1
    assert scored["lob_points"] == 1 and scored["complete_lob_points"] == 1


def test_a_gap_does_not_invent_a_net_crossing():
    point = _point()
    assert bench._net_crossing_frames(point) != []
    for frame in (123, 124, 125, 126, 127):
        del point["positions"][frame]
    assert bench._net_crossing_frames(point) == []


# --------------------------------------------------------------------------------------- #
# Per-match grouping
# --------------------------------------------------------------------------------------- #
def test_complete_points_are_reported_per_source_match(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report(
        {"points": [_terminal_point()]},
        _terminal_report(),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    assert scored["source_matches"] == 1
    assert scored["complete_points_by_source_match"] == {"syn": 1}
    assert scored["points_by_source_match"] == {"syn": 1}
    assert scored["source_match_rows"][0]["lob_points"] == 0
    assert scored["point_rows"][0]["source_match"] == "syn"


def test_source_match_falls_back_to_the_point_key(monkeypatch):
    """An older truth file carries no ``source_match``; the cohort broadcast is its prefix."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    del point["source_match"]
    scored = bench.audit_report(
        {"points": [point]},
        _terminal_report(),
        Path("/nonexistent"),
        criterion=bench.CRITERION_PIXEL_V1,
    )
    assert scored["complete_points_by_source_match"] == {"syn": 1}


@pytest.mark.parametrize("definition", [None, bench.COMPLETE_TRUTH_DEFINITION])
def test_audit_reads_and_writes_the_roots_it_is_given(monkeypatch, tmp_path, definition):
    """The re-scoring path: a bench owned by another package is scored without writing to it."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    truth = {"points": [_terminal_point()]}
    if definition is not None:
        truth["truth_definition"] = definition
    bench.write_json(tmp_path / "truth" / "truth.json", truth)
    bench.write_json(tmp_path / "runs" / "default__clean" / "report.json", _terminal_report())
    bench.main(
        [
            "--output-root",
            str(tmp_path / "unused"),
            "audit",
            "--truth",
            str(tmp_path / "truth" / "truth.json"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--audits-root",
            str(tmp_path / "audits"),
            "--cameras-root",
            str(tmp_path / "cameras"),
            "--criterion",
            "pixel_v1",
        ]
    )
    summary = json.loads((tmp_path / "audits" / "summary.json").read_text())
    assert summary["rows"][0]["arm"] == "default" and summary["rows"][0]["rung"] == "clean"
    assert summary["rows"][0]["complete_points"] == 1
    expected_definition = definition if definition is not None else bench.TRUTH_DEFINITION
    assert summary["truth_definition"] == expected_definition
    scored = json.loads((tmp_path / "audits" / "default__clean.json").read_text())
    assert scored["truth_definition"] == expected_definition
    assert summary["schema"] == "tennis.s6-point-bench-audit.v4"
    assert not (tmp_path / "unused").exists()
    assert (tmp_path / "audits" / "default__clean_source_matches.csv").is_file()


# --------------------------------------------------------------------------------------- #
# The development inner loop
# --------------------------------------------------------------------------------------- #
def _dev_truth() -> dict:
    """A miniature bench with enough of every shape to satisfy the quotas."""
    points = []
    plan = [
        ("second_bounce", 3, False, 0),
        ("second_bounce", 3, False, 0),
        ("second_bounce", 3, False, 0),
        ("second_bounce", 3, False, 0),
        ("terminal_bounce", 3, False, 0),
        ("terminal_bounce", 3, False, 0),
        ("out_of_view", 3, False, 0),
        ("out_of_view", 3, False, 0),
        ("second_bounce", 3, True, 40),
        ("second_bounce", 3, True, 40),
        ("second_bounce", 5, False, 0),
        ("second_bounce", 4, False, 0),
    ]
    for index, (kind, shots, lob, gap) in enumerate(plan):
        point = _terminal_point(kind if kind != "out_of_view" else "out_of_view")
        point["point"] = f"m{index:02d}__pt0001"
        point["match_id"] = f"m{index:02d}"
        point["source_match"] = f"m{index:02d}"
        point["surface"] = "hard" if index % 2 else "clay"
        point["shots"] = shots
        point["lob"] = lob
        point["mode"] = bench.MODE_LOB if lob else bench.MODE_RALLY
        point["out_of_frame_frames"] = list(range(gap))
        points.append(point)
    return {"points": points}


def test_dev_select_meets_every_quota_and_spreads_over_matches_and_surfaces():
    rows = bench.select_dev_points(_dev_truth())
    assert len(rows) == 12
    counts = {name: 0 for name, _, _ in bench.DEV_QUOTAS}
    for row in rows:
        counts[row["category"]] += 1
    assert counts == {name: wanted for name, wanted, _ in bench.DEV_QUOTAS}
    assert len({row["source_match"] for row in rows}) >= bench.DEV_MIN_SOURCE_MATCHES
    assert {row["surface"] for row in rows} == {"hard", "clay"}
    assert all(row["reason"] for row in rows)


def test_dev_select_is_deterministic():
    first = [row["point"] for row in bench.select_dev_points(_dev_truth())]
    second = [row["point"] for row in bench.select_dev_points(_dev_truth())]
    assert first == second


def test_dev_select_refuses_a_bench_that_cannot_fill_a_quota():
    truth = _dev_truth()
    truth["points"] = [
        row for row in truth["points"] if row["termination_kind"] != "terminal_bounce"
    ]
    with pytest.raises(RuntimeError, match="terminal_bounce"):
        bench.select_dev_points(truth)


def test_dev_select_writes_the_reason_for_every_point(tmp_path):
    bench.write_json(tmp_path / "truth.json", _dev_truth())
    bench.main(["--output-root", str(tmp_path), "dev-select"])
    payload = json.loads((tmp_path / bench.DEV_POINTS_NAME).read_text())
    assert payload["frozen"] is True
    assert len(payload["points"]) == 12
    assert len(payload["quotas"]) == len(bench.DEV_QUOTAS)
    assert all(row["reason"] for row in payload["points"])


def test_dev_point_keys_prefers_explicit_points(tmp_path):
    bench.write_json(tmp_path / "truth.json", _dev_truth())
    bench.main(["--output-root", str(tmp_path), "dev-select"])
    args = bench.build_parser().parse_args(
        ["--output-root", str(tmp_path), "dev-run", "--points", "a__pt0001"]
    )
    assert bench.dev_point_keys(args) == ["a__pt0001"]
    args = bench.build_parser().parse_args(["--output-root", str(tmp_path), "dev-run"])
    assert len(bench.dev_point_keys(args)) == 12


def test_run_reconstruction_passes_one_point_flag_per_development_point(monkeypatch, tmp_path):
    """The subset run is the same fitter invocation, restricted by the CLI's own flag."""
    captured: dict = {}

    class _Completed:
        returncode = 0

    def _run(command, **kwargs):
        captured["command"] = command
        return _Completed()

    monkeypatch.setattr(bench.subprocess, "run", _run)
    bench.run_reconstruction(
        tmp_path / "root", tmp_path / "out", arm="default", workers=2, points=["a__pt1", "b__pt2"]
    )
    command = captured["command"]
    assert command.count("--point") == 2
    assert "a__pt1" in command and "b__pt2" in command
    for value in bench.NIGHTLY_ARGUMENTS:
        assert value in command


def test_flight_failure_reason_names_the_mechanism():
    row = {
        "attempted": True,
        "solved": False,
        "accepted": False,
        "truth_good": False,
        "bounce_anchors": 2,
        "net_anchor_available": True,
        "bounce_ok": False,
        "contact_ok": True,
    }
    assert bench.flight_failure_reason(row, {}) == "two_bounce_anchors=2"
    row["bounce_anchors"] = 1
    row["net_anchor_available"] = False
    assert bench.flight_failure_reason(row, {}) == "no_net_anchor"
    row["net_anchor_available"] = True
    assert bench.flight_failure_reason(row, {}) == "solver_returned_none"
    row["solved"] = True
    assert bench.flight_failure_reason(row, {"reasons": ["physics_speed"]}).startswith(
        "pixel_gate:"
    )
    row["accepted"] = True
    row["failed_checks"] = "bounce"
    assert bench.flight_failure_reason(row, {}) == "wrong_accept:bounce"
    row["bounce_ok"] = True
    row["truth_good"] = True
    assert bench.flight_failure_reason(row, {}) == ""


def test_the_audit_row_carries_metric_contact_error_at_both_ends(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report({"points": [_terminal_point()]}, _terminal_report(), Path("/x"))
    row = scored["flight_rows"][0]
    assert row["start_contact_error_m"] == pytest.approx(0.0)
    assert row["end_contact_error_m"] == pytest.approx(0.0)
    assert row["failure_reason"] is not None


def test_the_audit_row_carries_the_junction_gap_on_both_sides(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    report = _terminal_report()
    report["points_detail"][0]["junction_gaps_m"] = [1.25]
    scored = bench.audit_report({"points": [_terminal_point()]}, report, Path("/x"))
    first, second = scored["flight_rows"]
    assert first["junction_gap_prev_m"] is None
    assert first["junction_gap_next_m"] == pytest.approx(1.25)
    assert second["junction_gap_prev_m"] == pytest.approx(1.25)
    assert second["junction_gap_next_m"] is None


def test_the_dev_table_prints_one_line_per_flight_and_one_per_point(monkeypatch, capsys):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    scored = bench.audit_report({"points": [_terminal_point()]}, _terminal_report(), Path("/x"))
    bench.print_dev_table(scored["flight_rows"], scored["point_rows"])
    printed = capsys.readouterr().out
    assert printed.count("syn__pt0001") == len(scored["flight_rows"]) + 1
    assert "bounce_m" in printed and "term_m" in printed and "first cause" in printed


# --------------------------------------------------------------------------------------- #
# `metric_v2`: what the owner means by right, measured in 3D everywhere
# --------------------------------------------------------------------------------------- #
def _truth_shaped_report(point: dict, *, offset=(0.0, 0.0, 0.0), gaps=(0.05,)) -> dict:
    """A report whose fitted flights ARE the truth curve, optionally displaced by ``offset``.

    Building the fit from the truth curve is the only way to test the criterion's own
    thresholds rather than the arithmetic of an invented trajectory.
    """
    paths = bench.truth_flight_paths(point, scored_only=True)
    shift = np.asarray(offset, dtype=float)
    fits = []
    for path, entry in zip(paths, point["flights"], strict=True):
        end = bench.terminal_scored_end(entry)
        frames = [
            float(frame)
            for frame in range(int(math.ceil(entry["start_frame"])), int(math.floor(end)) + 1)
        ]
        rows = []
        for frame in frames:
            position = bench.sample_path(path, frame)
            if position is None:
                continue
            rows.append({"frame": frame, "xyz": (position + shift).tolist()})
        start = bench.sample_path(path, float(entry["start_frame"]))
        finish = bench.sample_path(path, end)
        fits.append(
            {
                "flight_index": int(entry["flight_index"]),
                "start_frame": float(entry["start_frame"]),
                "end_frame": end,
                "held_out_accepted": True,
                "held_out_reprojection_median_px": 2.0,
                "start_xyz": (start + shift).tolist(),
                "end_xyz": (finish + shift).tolist(),
                "trajectory": rows,
                "bounces": [
                    {"frame": float(row["frame"]), "x": (np.asarray(row["xyz"]) + shift).tolist()}
                    for row in bench.scored_bounces(entry)
                ],
            }
        )
    return {
        "points_detail": [
            {
                "point": point["point"],
                "decision": "retain",
                "reasons": [],
                "junction_gaps_m": list(gaps),
                "flight_attempts": [
                    {"flight_index": fit["flight_index"], "start_frame": fit["start_frame"]}
                    for fit in fits
                ],
                "fits": fits,
            }
        ]
    }


def test_a_fit_that_is_the_truth_is_right_under_both_criteria(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    scored = bench.audit_report(
        {"points": [point]}, _truth_shaped_report(point), Path("/nonexistent")
    )
    assert [row["truth_good_metric_v2"] for row in scored["flight_rows"]] == [True, True]
    assert scored["by_criterion"]["metric_v2"]["complete_points"] == 1
    assert scored["by_criterion"]["pixel_v1"]["complete_points"] == 1
    assert scored["flight_rows"][0]["trajectory_rms_m"] == pytest.approx(0.0, abs=1e-6)


def test_a_flight_that_is_right_in_pixels_and_metres_wrong_in_depth_is_a_wrong_accept(monkeypatch):
    """docs/wk1/dev_loop.md's finding, as a test.

    ``_Camera`` sees only court width, so a displacement along the court length is invisible in
    the image: the flight reprojects exactly and is three metres away in the court frame.
    """
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    report = _truth_shaped_report(point, offset=(0.0, 3.0, 0.0))
    scored = bench.audit_report({"points": [point]}, report, Path("/nonexistent"))
    row = scored["flight_rows"][0]
    assert row["contact_error_px"] == pytest.approx(0.0, abs=1e-6)
    assert row["contact_error_m"] == pytest.approx(3.0, abs=1e-6)
    assert row["trajectory_rms_m"] == pytest.approx(3.0, abs=1e-6)
    assert row["truth_good_pixel_v1"] is False  # the bounce moved too
    assert row["truth_good_metric_v2"] is False
    assert "contact_m" in row["failed_checks"] and "trajectory" in row["failed_checks"]


def test_pixel_v1_accepts_the_depth_error_that_metric_v2_refuses(monkeypatch):
    """The same displacement on a flight with no bounce witness: v1 sees nothing wrong."""
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point("out_of_view")
    report = _truth_shaped_report(point, offset=(0.0, 3.0, 0.0))
    scored = bench.audit_report({"points": [point]}, report, Path("/nonexistent"))
    terminal = scored["flight_rows"][1]
    assert terminal["termination_kind"] == "out_of_view"
    assert terminal["bounce_error_m"] is None  # no bounce inside the scored span
    assert terminal["contact_error_px"] == pytest.approx(0.0, abs=1e-6)
    assert terminal["truth_good_pixel_v1"] is True
    assert terminal["truth_good_metric_v2"] is False
    assert terminal["exit_error_m"] == pytest.approx(3.0, abs=1e-6)


def test_every_metric_v2_condition_can_fail_on_its_own():
    """Each threshold is read, and only the condition it belongs to fails."""
    base = {
        "terminal": True,
        "termination_kind": "second_bounce",
        "bounce_error_m": 0.05,
        "contact_error_px": 4.0,
        "contact_error_m": 0.10,
        "termination_error_m": 0.20,
        "exit_error_m": None,
        "junction_gap_prev_m": 0.10,
        "junction_gap_next_m": 0.10,
        "trajectory_rms_m": 0.10,
        "mid_air_stop": False,
    }
    assert all(bench.metric_v2_checks(base).values())
    cases = {
        "bounce": ("bounce_error_m", bench.BOUNCE_TOLERANCE_M + 0.01),
        "contact_px": ("contact_error_px", bench.CONTACT_PIXEL_TOLERANCE_PX + 0.1),
        "contact_m": ("contact_error_m", bench.CONTACT_TOLERANCE_M + 0.01),
        "termination": ("termination_error_m", bench.TERMINATION_TOLERANCE_M + 0.01),
        "junction": ("junction_gap_next_m", bench.JUNCTION_TOLERANCE_M + 0.01),
        "trajectory": ("trajectory_rms_m", bench.TRAJECTORY_RMS_TOLERANCE_M + 0.01),
    }
    for name, (field, value) in cases.items():
        row = dict(base, **{field: value})
        checks = bench.metric_v2_checks(row)
        assert checks[name] is False, name
        assert [key for key, passed in checks.items() if not passed] == [name]


def test_a_witness_that_is_exactly_on_its_threshold_still_passes():
    row = {
        "terminal": False,
        "termination_kind": None,
        "bounce_error_m": bench.BOUNCE_TOLERANCE_M,
        "contact_error_px": bench.CONTACT_PIXEL_TOLERANCE_PX,
        "contact_error_m": bench.CONTACT_TOLERANCE_M,
        "junction_gap_next_m": bench.JUNCTION_TOLERANCE_M,
        "trajectory_rms_m": bench.TRAJECTORY_RMS_TOLERANCE_M,
    }
    assert all(bench.metric_v2_checks(row).values())


def test_an_out_of_view_ending_must_still_be_airborne_and_ballistic():
    base = {
        "terminal": True,
        "termination_kind": "out_of_view",
        "exit_error_m": 0.20,
        "exit_height_m": 3.0,
        "exit_angle_deg": 20.0,
        "mid_air_stop": False,
    }
    assert bench.metric_v2_checks(base)["airborne"] is True
    assert bench.metric_v2_checks(dict(base, exit_height_m=0.05))["airborne"] is False
    assert bench.metric_v2_checks(dict(base, exit_angle_deg=85.0))["airborne"] is False
    assert bench.metric_v2_checks(dict(base, mid_air_stop=True))["airborne"] is False
    assert bench.metric_v2_checks(dict(base, exit_height_m=None))["airborne"] is False
    # A bounce ending is not asked to be airborne: it ends on the court by definition.
    assert bench.metric_v2_checks(dict(base, termination_kind="second_bounce"))["airborne"] is True


def test_pixel_v1_asks_only_for_bounces_and_pixels():
    row = {
        "bounce_error_m": 0.05,
        "contact_error_px": 4.0,
        "contact_error_m": 9.0,
        "trajectory_rms_m": 9.0,
        "junction_gap_next_m": 9.0,
    }
    assert set(bench.pixel_v1_checks(row)) == {"bounce", "contact_px"}
    assert all(bench.pixel_v1_checks(row).values())


def test_the_last_visible_truth_frame_is_the_frame_before_the_ball_leaves():
    point = _terminal_point("out_of_view")
    assert point["out_of_frame_frames"][0] == 128
    assert bench.last_visible_truth_frame(point, point["flights"][1]) == 127.0


def test_a_flight_that_never_leaves_the_frame_has_no_exit_witness():
    point = _terminal_point()
    assert bench.last_visible_truth_frame(point, point["flights"][1]) == 148.0


def test_a_fit_that_ends_in_mid_air_before_the_flight_does_is_flagged(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    report = _truth_shaped_report(point)
    terminal = report["points_detail"][0]["fits"][1]
    terminal["trajectory"] = [row for row in terminal["trajectory"] if float(row["frame"]) <= 140.0]
    terminal["trajectory"][-1]["xyz"] = [5.0, 17.0, 2.5]
    terminal["end_frame"] = 140.0
    scored = bench.audit_report({"points": [point]}, report, Path("/nonexistent"))
    row = scored["flight_rows"][1]
    assert row["mid_air_stop"] is True
    assert row["fit_end_height_m"] == pytest.approx(2.5)
    assert scored["mid_air_stop_flights"] == 1


def test_a_frame_the_fit_never_reaches_makes_the_trajectory_rms_infinite(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    report = _truth_shaped_report(point)
    terminal = report["points_detail"][0]["fits"][1]
    terminal["trajectory"] = [row for row in terminal["trajectory"] if float(row["frame"]) <= 140.0]
    terminal["end_frame"] = 140.0
    scored = bench.audit_report({"points": [point]}, report, Path("/nonexistent"))
    row = scored["flight_rows"][1]
    assert row["trajectory_uncovered_frames"] > 0
    assert math.isinf(row["trajectory_rms_m"])
    assert row["truth_good_metric_v2"] is False


def test_the_final_segment_angle_reads_a_vertical_drop():
    fit = {
        "trajectory": [
            {"frame": 100.0, "xyz": [5.0, 10.0, 3.0]},
            {"frame": 101.0, "xyz": [5.0, 10.02, 1.0]},
        ]
    }
    final = bench.fit_final_segment(fit)
    assert final["frame"] == 101.0 and final["height_m"] == pytest.approx(1.0)
    assert final["angle_deg"] > bench.NEAR_VERTICAL_ANGLE_DEG


def test_the_fitted_direction_angle_is_read_off_the_velocity():
    fit = {
        "trajectory": [
            {"frame": 100.0, "velocity": [0.0, 20.0, 0.0]},
            {"frame": 101.0, "velocity": [0.0, 0.1, -20.0]},
        ]
    }
    assert bench.fit_direction_angle_deg(fit, 100.0) == pytest.approx(0.0, abs=1e-6)
    assert bench.fit_direction_angle_deg(fit, 101.0) > bench.NEAR_VERTICAL_ANGLE_DEG


# --------------------------------------------------------------------------------------- #
# The truth curve the criterion is measured against
# --------------------------------------------------------------------------------------- #
def test_the_re_integrated_truth_curve_hits_its_anchors():
    point = _terminal_point()
    paths = bench.truth_flight_paths(point, scored_only=True)
    for path, entry in zip(paths, point["flights"], strict=True):
        start = bench.sample_path(path, float(entry["start_frame"]))
        assert start is not None
        for bounce in bench.scored_bounces(entry):
            sampled = bench.sample_path(path, float(bounce["frame"]))
            assert sampled is not None
            assert np.linalg.norm(sampled - np.asarray(bounce["xyz"])) < 0.01


def test_recorded_truth_preserves_the_actual_non_nominal_spin_interpolant():
    start = np.array([5.0, 1.0, 2.5])
    velocity = np.array([2.0, 24.0, 5.0])
    spin = np.array([450.0, 50.0, 80.0])
    times, positions, _, _ = bench.integrate_arc(
        start, velocity, spin, max_seconds=0.8, stop_at_ground=False
    )
    piece = bench.Piece(0.0, 0.73, start, velocity, spin, times=times, positions=positions)
    recorded = bench.recorded_trajectory([piece], 100.25, 29.97)
    point = {
        "trajectory_truth": json.loads(json.dumps(recorded)),
        "flights": [
            {
                "flight_index": 0,
                "start_frame": 100.25,
                "end_frame": 100.25 + 0.73 * 29.97,
                "terminal": False,
            }
        ],
    }
    path = bench.truth_flight_paths(point)[0]
    for seconds in np.linspace(0.0, 0.73, 57):
        exact = piece.sample([seconds])[0]
        assert np.allclose(bench.sample_path(path, 100.25 + seconds * 29.97), exact, atol=1e-12)
    assert path["source"] == "generator_piece_interpolant"


def test_recorded_truth_retains_out_of_view_positions_and_clips_only_scoring():
    point = _terminal_point()
    frames = np.arange(100, 165, dtype=float)
    point["trajectory_truth"] = {
        "schema": "simulation_trajectory_samples_v1",
        "frames": frames.tolist(),
        "positions": np.column_stack([frames, frames * 2, frames * 3]).tolist(),
    }
    # Deliberately distinct from the anchors: prove no nominal-spin refit replaces these samples.
    scored = bench.truth_flight_paths(point, scored_only=True)[1]
    drawn = bench.truth_flight_paths(point, scored_only=False)[1]
    assert np.allclose(bench.sample_path(scored, 140.0), [140.0, 280.0, 420.0])
    assert scored["frames"][-1] == 148.0
    assert drawn["frames"][-1] == 164.0


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"frames": [1.0]},
        {"frames": [100.0, 100.0]},
        {"positions": [[1.0, 2.0, float("nan")]]},
    ],
)
def test_bad_recorded_truth_never_falls_back_to_an_approximation(change):
    point = _terminal_point()
    point["trajectory_truth"] = {
        "schema": "simulation_trajectory_samples_v1",
        "frames": [100.0, 156.0],
        "positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        **change,
    }
    with pytest.raises(ValueError):
        bench.truth_flight_paths(point)


def test_recorded_truth_rejects_discontinuous_shared_boundaries():
    first = bench.Piece(
        0.0, 1.0, None, None, None, times=np.array([0.0, 1.0]), positions=np.zeros((2, 3))
    )
    second = bench.Piece(
        1.0, 2.0, None, None, None, times=np.array([0.0, 1.0]), positions=np.ones((2, 3))
    )
    with pytest.raises(ValueError, match="disagree"):
        bench.recorded_trajectory([first, second], 1.0, 25.0)


def test_the_scored_truth_curve_stops_at_the_termination():
    """The dead ball is not part of the flight, so the scored curve does not draw it."""
    point = _terminal_point()
    scored_path = bench.truth_flight_paths(point, scored_only=True)[1]
    drawn_path = bench.truth_flight_paths(point, scored_only=False)[1]
    assert scored_path["frames"][-1] == pytest.approx(148.0)
    assert drawn_path["frames"][-1] == pytest.approx(156.0)


def test_the_truth_curve_cache_round_trips(tmp_path):
    truth = {"points": [_terminal_point()]}
    cache = tmp_path / "paths.npz"
    built = bench.truth_scored_paths(truth, cache_path=cache)
    assert cache.is_file()
    reloaded = bench.truth_scored_paths(truth, cache_path=cache)
    assert set(reloaded) == set(built)
    for key, paths in built.items():
        for original, restored in zip(paths, reloaded[key], strict=True):
            assert np.allclose(original["frames"], restored["frames"])
            assert np.allclose(original["positions"], restored["positions"])


def test_truth_curve_cache_is_not_identified_by_filename_alone(tmp_path):
    point = _terminal_point()
    point["trajectory_truth"] = {
        "schema": "simulation_trajectory_samples_v1",
        "frames": [100.0, 156.0],
        "positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }
    cache = tmp_path / "same_name.npz"
    truth = {"points": [point]}
    original = bench.truth_scored_paths(truth, cache_path=cache)
    point["trajectory_truth"]["positions"][0][2] = 7.0
    changed = bench.truth_scored_paths(truth, cache_path=cache)
    assert original["syn__pt0001"][0]["positions"][0][2] == 3.0
    assert changed["syn__pt0001"][0]["positions"][0][2] == 7.0
    cache.write_bytes(b"corrupt output")
    restored = bench.truth_scored_paths(truth, cache_path=cache)
    assert restored["syn__pt0001"][0]["positions"][0][2] == 7.0


def test_the_truth_curve_can_be_built_for_a_subset_of_points():
    truth = {"points": [_terminal_point()]}
    assert bench.truth_scored_paths(truth, keys=["syn__pt0001"]).keys() == {"syn__pt0001"}
    assert bench.truth_scored_paths(truth, keys=["nothing"]) == {}


# --------------------------------------------------------------------------------------- #
# Reporting both criteria
# --------------------------------------------------------------------------------------- #
def test_both_criteria_are_reported_whichever_one_is_selected(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point("out_of_view")
    report = _truth_shaped_report(point, offset=(0.0, 3.0, 0.0))
    for criterion in bench.CRITERIA:
        scored = bench.audit_report(
            {"points": [point]}, report, Path("/nonexistent"), criterion=criterion
        )
        assert scored["criterion"] == criterion
        assert set(scored["by_criterion"]) == set(bench.CRITERIA)
        assert scored["by_criterion"]["pixel_v1"]["truth_good_flights"] == 1
        assert scored["by_criterion"]["metric_v2"]["truth_good_flights"] == 0
        selected = scored["by_criterion"][criterion]
        assert scored["truth_good_flights"] == selected["truth_good_flights"]
        assert scored["wrong_accepted_flights"] == selected["wrong_accepted_flights"]
        assert scored["complete_points"] == selected["complete_points"]


def test_the_default_criterion_is_the_metric_one(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point("out_of_view")
    scored = bench.audit_report(
        {"points": [point]},
        _truth_shaped_report(point, offset=(0.0, 3.0, 0.0)),
        Path("/nonexistent"),
    )
    assert bench.DEFAULT_CRITERION == bench.CRITERION_METRIC_V2
    assert scored["criterion"] == bench.CRITERION_METRIC_V2
    assert scored["flight_rows"][1]["truth_good"] is False


def test_an_unknown_criterion_is_refused(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    with pytest.raises(ValueError, match="unknown criterion"):
        bench.audit_report(
            {"points": [_point()]}, _report(), Path("/nonexistent"), criterion="whatever"
        )


def test_historical_metric_v2_thresholds_remain_unchanged_and_are_not_owner_targets():
    """Keep historical comparisons stable without attributing old targets to the owner."""
    assert bench.BOUNCE_TOLERANCE_M == pytest.approx(0.10)
    assert bench.CONTACT_TOLERANCE_M == pytest.approx(2.5 * bench.BOUNCE_TOLERANCE_M)
    assert bench.JUNCTION_TOLERANCE_M == pytest.approx(2.5 * bench.BOUNCE_TOLERANCE_M)
    assert bench.TERMINATION_TOLERANCE_M == pytest.approx(5.0 * bench.BOUNCE_TOLERANCE_M)
    assert bench.TRAJECTORY_RMS_TOLERANCE_M == pytest.approx(3.0 * bench.BOUNCE_TOLERANCE_M)
    definition = bench.CRITERION_DEFINITION[bench.CRITERION_METRIC_V2]
    assert "historical 0.10 m" in definition["anchor"]
    assert "not the current owner's" in definition["anchor"]
    assert set(definition) - {"anchor"} == set(
        bench.metric_v2_checks({"terminal": False, "termination_kind": None, "mid_air_stop": False})
    )


def test_the_failed_check_names_go_into_the_flight_reason(monkeypatch):
    monkeypatch.setattr(bench, "PointCamera", _Camera)
    point = _terminal_point()
    scored = bench.audit_report(
        {"points": [point]}, _truth_shaped_report(point, gaps=(4.0,)), Path("/nonexistent")
    )
    assert scored["flight_rows"][0]["failure_reason"] == "wrong_accept:junction"
    assert scored["failed_check_counts"]["junction"] == 2
    assert scored["metric_v2_check_pass_counts"]["junction"] == 0
    assert scored["metric_v2_check_pass_counts"]["bounce"] == 2


# ------------------------------------------------------------------------------------------- #
# The striker witness
# ------------------------------------------------------------------------------------------- #
# A real broadcast camera from the cohort (ao2022f_w_barty_collins pt0001), so a projected body
# box and its back-projected root are measured through a projection the bench actually uses.
_STRIKER_P = np.array(
    [
        [1.27694654e02, 3.21924684e01, -4.48011735e01, 2.67205366e02],
        [8.14925934e-01, -1.25780409e01, -1.32709718e02, 8.56434495e02],
        [2.17123856e-04, 3.39452625e-02, -4.75508277e-02, 1.00000000e00],
    ]
)


def _striker_host() -> bench.HostPoint:
    frames = np.arange(90, 160)
    return bench.HostPoint(
        match_id="syn",
        clip="pt0001",
        surface="hard",
        fps=25.0,
        span=(90.0, 160.0),
        camera_rows={"frames": frames, "P": np.repeat(_STRIKER_P[None], len(frames), axis=0)},
        homography=np.eye(3),
        frame_homographies={},
    )


def test_striker_court_y_is_the_fitters_frame_and_not_the_centred_one():
    """docs/wk1/point_fit4.md section 5: the bench used to write a centred court_y."""
    rows, _, _ = bench.striker_rows(_striker_host(), _point(), rung="clean", seed=7)
    near = [row for row in rows if row["side"] == "near"]
    far = [row for row in rows if row["side"] == "far"]
    assert near and far
    assert all(row["court_y"] < bench.NET_Y_M for row in near)
    assert all(row["court_y"] > bench.NET_Y_M for row in far)
    assert min(row["court_y"] for row in rows) > -bench.COURT_LENGTH_M


def test_the_striker_moves_instead_of_taking_two_values():
    rows, _, _ = bench.striker_rows(_striker_host(), _point(), rung="clean", seed=7)
    values = {row["court_y"] for row in rows}
    assert len(values) > 20


def test_the_striker_stands_at_racket_reach_from_every_contact():
    point = _point()
    rows, _, _ = bench.striker_rows(_striker_host(), point, rung="clean", seed=7)
    by_side_frame = {(row["side"], row["frame"]): row for row in rows}
    for contact in point["contacts"]:
        side = bench.contact_side_of(contact)
        row = by_side_frame[(side, bench._frame_name(int(round(float(contact["frame"])))))]
        distance = math.hypot(
            row["court_x"] - float(contact["xyz"][0]),
            row["court_y"] - float(contact["xyz"][1]),
        )
        assert distance < bench.STRIKER_REACH_RANGE_M[1] + 0.5


def test_the_written_columns_are_the_real_cohorts():
    rows, _, _ = bench.striker_rows(_striker_host(), _point(), rung="clean", seed=7)
    assert list(rows[0]) == [
        "clip",
        "frame",
        "side",
        "x0",
        "y0",
        "x1",
        "y1",
        "conf",
        "court_x",
        "court_y",
        "track_id",
    ]


def test_the_fitters_own_loader_reads_the_written_file(tmp_path):
    from cv.pipeline.reconstruction import load_players

    point = _point()
    rows, _, _ = bench.striker_rows(_striker_host(), point, rung="clean", seed=7)
    bench.write_declared_csv(
        tmp_path / "player_boxes_25_native_sided_v1.csv", rows, kind="boxes", fps=25.0
    )
    players, boxes = load_players(tmp_path, "pt0001")
    assert players["near"] and players["far"]
    contact = point["contacts"][0]
    side = bench.contact_side_of(contact)
    frame = int(round(float(contact["frame"])))
    reach = float(np.linalg.norm(players[side][frame] - np.asarray(contact["xyz"][:2])))
    # The prior the striker feeds: anything past it makes the fitter reject the flight.
    assert reach <= anchor_first_fit.MAX_CONTACT_REACH_M


def test_the_court_column_is_the_box_root_through_the_homography():
    homography = bench.ground_homography(_STRIKER_P)
    box = bench.striker_box_native(_STRIKER_P, (4.0, 3.0))
    root = bench.box_root_court(homography, box)
    assert float(np.linalg.norm(root - np.array([4.0, 3.0]))) < 0.35


def test_a_body_box_is_taller_than_it_is_wide():
    box = bench.striker_box_native(_STRIKER_P, (5.0, 4.0))
    assert box[3] - box[1] > box[2] - box[0]


def test_the_striker_plan_never_asks_for_a_superhuman_sprint():
    keys = [
        (0.0, np.array([1.0, 1.0]), True),
        (5.0, np.array([9.0, 1.0]), False),
        (10.0, np.array([1.0, 1.0]), True),
    ]
    relaxed = bench.relax_striker_keys(keys, 25.0)
    assert np.allclose(relaxed[0][1], [1.0, 1.0])
    assert np.allclose(relaxed[2][1], [1.0, 1.0])
    reach = bench.STRIKER_MAX_SPEED_MPS * 5.0 / 25.0
    assert float(np.linalg.norm(relaxed[1][1] - relaxed[0][1])) <= reach + 1e-6


def test_box_noise_is_a_rung_parameter_and_clean_carries_none():
    host, point = _striker_host(), _point()
    clean, _, clean_diagnostic = bench.striker_rows(host, point, rung="clean", seed=7)
    noisy, _, noisy_diagnostic = bench.striker_rows(host, point, rung="realistic", seed=7)
    assert bench.STRIKER_NOISE_RUNGS == ("track_error", "realistic", "realistic_v2")
    assert clean_diagnostic["missing_frames"] == {"near": 0, "far": 0}
    assert sum(noisy_diagnostic["missing_frames"].values()) > 0
    keyed = {(row["side"], row["frame"]): row for row in clean}
    moved = [
        abs(row["court_y"] - keyed[(row["side"], row["frame"])]["court_y"])
        for row in noisy
        if (row["side"], row["frame"]) in keyed
    ]
    assert moved and max(moved) > 0.0


def test_the_planned_striker_path_is_the_same_on_every_rung():
    """Only the tracker noise is a rung parameter; the body's motion is the point's own.

    The striker draws from generators seeded on the point key alone -- ``striker_rows`` takes no
    rng -- so adding it leaves the ball-track, event-timing and anchor-pattern streams that
    ``apply_rung`` samples exactly where they were, and the generated points stay the same
    points.
    """
    import inspect

    assert "rng" not in inspect.signature(bench.striker_rows).parameters
    host, point = _striker_host(), _point()
    clean, _, _ = bench.striker_rows(host, point, rung="clean", seed=7)
    for rung in ("event_timing", "missing_anchors"):
        assert bench.striker_rows(host, point, rung=rung, seed=7)[0] == clean


def test_the_pose_file_carries_the_racket_hand_wrist_at_the_contact():
    point = _point()
    _, pose, _ = bench.striker_rows(_striker_host(), point, rung="clean", seed=7)
    by_frame = {(row["side"], row["frame"]): row for row in pose}
    contact = point["contacts"][1]
    side = bench.contact_side_of(contact)
    row = by_frame[(side, bench._frame_name(int(round(float(contact["frame"])))))]
    ball = bench.project(_STRIKER_P, contact["xyz"])
    hands = [
        math.hypot(float(row[f"{hand}_wrist_x"]) - ball[0], float(row[f"{hand}_wrist_y"]) - ball[1])
        for hand in ("left", "right")
    ]
    assert min(hands) < (row["y1"] - row["y0"])


def test_the_pose_arm_is_the_only_one_that_reads_the_pose_file():
    assert bench.ARMS["pose_witness"] == ("--pose-artifact-name", bench.POSE_NAME)
    assert "--pose-artifact-name" not in bench.NIGHTLY_ARGUMENTS


# ------------------------------------------------------------------------------------------ #
# Emissions on a frame, at the track's own pixel (docs/wk1/point_fit6.md's finding)
# ------------------------------------------------------------------------------------------ #
def _corner_track(corner: int = 110) -> dict[int, list[float]]:
    """A track that runs one way, turns a hard corner at ``corner``, and runs back."""
    track = {}
    for frame in range(corner - 6, corner + 7):
        step = frame - corner
        track[frame] = [1000.0 + 30.0 * abs(step), 400.0 - 20.0 * step]
    return track


def test_an_emission_lands_on_a_frame_and_carries_that_frame_s_track_pixel():
    track = _corner_track()
    choice = bench.emission_choice(track, 110.4, [1234.0, 400.0])
    assert choice["frame"] == 110
    assert float(choice["frame"]) == float(int(choice["frame"]))
    assert choice["pixel"] == track[110]
    assert choice["pixel_source"] == "track"
    assert choice["offset_frames"] == pytest.approx(-0.4)


def test_the_emitted_frame_is_the_track_s_own_corner():
    """The rule is cv/pipeline/contact_frame_refiner's, so the corner wins over the seed."""
    track = _corner_track(corner=112)
    choice = bench.emission_choice(track, 110.0, [0.0, 0.0])
    assert choice["frame"] == 112
    assert choice["refiner_reason"] == "turn_selected_frame"


def test_a_track_with_no_corner_keeps_the_frame_nearest_the_true_time():
    track = {frame: [1000.0 + 30.0 * frame, 400.0] for frame in range(100, 121)}
    choice = bench.emission_choice(track, 110.4, [0.0, 0.0])
    assert choice["frame"] == 110
    assert choice["refiner_reason"] == "weak_turn_keeps_emitted_frame"


def test_the_measured_timing_error_moves_the_emission_by_whole_frames():
    track = _corner_track()
    assert bench.emission_choice(track, 110.4, [0.0, 0.0], shift_frames=1.0)["frame"] == 111
    # Half a frame is most of the measured distribution; rounding it toward zero would
    # silently delete it, so it rounds away from zero and the emission stays on a frame.
    assert bench.emission_choice(track, 110.4, [0.0, 0.0], shift_frames=0.5)["frame"] == 111
    assert bench.emission_choice(track, 110.4, [0.0, 0.0], shift_frames=-0.5)["frame"] == 109
    assert bench.emission_choice(track, 110.4, [0.0, 0.0], shift_frames=0.4)["frame"] == 110
    assert bench._round_away_from_zero(-1.5) == -2


def test_an_emission_on_a_frame_the_track_lost_keeps_its_frame():
    track = _corner_track()
    del track[110]
    choice = bench.emission_choice(track, 110.4, [0.0, 0.0], fallback_pixels={110: [7.0, 8.0]})
    assert choice["frame"] == 110
    assert choice["pixel"] == [7.0, 8.0]
    assert choice["pixel_source"] == "projection_no_track_row"


def test_every_emission_of_every_rung_is_on_an_integer_frame():
    for rung in bench.RUNGS:
        point = bench.apply_rung(
            _point(), rung, _Samplers(), np.random.default_rng(0), realism=_Realism()
        )
        rows = (
            point["emission_contacts"] + point["emission_bounces"] + [point["emission_point_end"]]
        )
        for row in rows:
            frame = float(row["frame"])
            assert frame == float(int(frame)), (rung, row)


def test_the_truth_keeps_the_continuous_event_time_the_emission_rounded():
    point = _point()
    point["contacts"][1]["frame"] = 120.4
    point["flights"][0]["end_frame"] = 120.4
    point["flights"][1]["start_frame"] = 120.4
    corrupted = bench.apply_rung(point, "clean", _Samplers(), np.random.default_rng(0))
    assert corrupted["emission_contacts"][1]["frame"] == 120.0
    # The generator's own continuous time is untouched: it is what the audit scores against.
    assert corrupted["contacts"][1]["frame"] == pytest.approx(120.4)
    row = next(
        item
        for item in corrupted["emission_rows"]
        if item["event_type"] == "contact" and item["truth_frame"] == pytest.approx(120.4)
    )
    assert row["offset_frames"] == pytest.approx(-0.4)


def test_the_emission_timing_summary_reports_the_bench_s_own_offset_distribution():
    point = _point()
    point["contacts"][1]["frame"] = 120.4
    corrupted = bench.apply_rung(point, "clean", _Samplers(), np.random.default_rng(0))
    summary = bench.emission_timing_summary(corrupted["emission_rows"])
    assert summary["all"]["fractional_frames"] == 0
    assert summary["contact"]["absolute_offset_frames"]["mean"] == pytest.approx(0.2)
    assert summary["contact_serve"]["absolute_offset_frames"]["max"] == pytest.approx(0.0)


def test_the_emission_frame_types_knob_leaves_the_types_it_is_not_given_alone():
    point = _point()
    point["contacts"][1]["frame"] = 120.4
    point["bounces"][0]["frame"] = 110.4
    corrupted = bench.apply_rung(
        point, "clean", _Samplers(), np.random.default_rng(0), emission_frame_types=("contact",)
    )
    assert corrupted["emission_contacts"][1]["frame"] == 120.0
    assert corrupted["emission_bounces"][0]["frame"] == pytest.approx(110.4)
    assert corrupted["emission_bounces"][0]["pixel"] == [1010.0, 400.0]


def test_the_point_end_emission_is_on_a_frame_too():
    point = _point()
    point["termination_frame"] = 132.4
    point["termination_kind"] = "out_of_view"
    corrupted = bench.apply_rung(point, "clean", _Samplers(), np.random.default_rng(0))
    assert corrupted["emission_point_end"]["frame"] == 132.0


def test_the_contact_is_scored_where_the_fit_puts_the_ball_at_the_true_contact_time():
    """A fit whose boundary sits on a frame is not charged for the emission's rounding."""
    fit = {
        "end_frame": 121.0,
        "trajectory": [
            {"frame": 120.0, "xyz": [0.0, 0.0, 1.0], "velocity": [0.0, 25.0, 0.0]},
        ],
        "start_xyz": [0.0, 0.0, 1.0],
    }
    sampled, source = bench._fit_contact_position(fit, 120.4, 25.0, fit["start_xyz"])
    assert source == "trajectory"
    assert sampled[1] == pytest.approx(0.4)
    endpoint, source = bench._fit_contact_position(
        {"end_frame": 121.0}, 120.4, 25.0, [1.0, 2.0, 3.0]
    )
    assert source == "endpoint"
    assert list(endpoint) == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------------------------------ #
# ``realistic_v2``: the rung whose parameters are the real cohort's measured statistics
# ------------------------------------------------------------------------------------------ #
class _Realism:
    """Deterministic stand-in for ``s6_bench.EmissionRealismSamplers``."""

    def __init__(
        self,
        *,
        head=(3.0, -2.0),
        frame_offset=0.0,
        off_track=False,
        gap=0,
        drop_bounce=False,
        drop_terminal_bounce=False,
        camera_abstains=False,
        wrong_body=None,
    ):
        self.head = head
        self.frame_offset = frame_offset
        self.off_track = off_track
        self.gap = gap
        self.drop_bounce = drop_bounce
        self.drop_terminal_bounce = drop_terminal_bounce
        self._camera_abstains = camera_abstains
        self.wrong_body = wrong_body

    def sample_pixel_head(self, rng, event_type, side):
        return np.asarray(self.head, dtype=float)

    def sample_frame_offset(self, rng, event_type, side):
        return float(self.frame_offset)

    def allow_off_track_frame(self, rng, event_type):
        return bool(self.off_track)

    def sample_contact_gap(self, rng):
        return int(self.gap)

    def drop_bounce_emission(self, rng, *, terminal=False):
        return bool(self.drop_terminal_bounce if terminal else self.drop_bounce)

    def camera_abstains(self, rng):
        return bool(self._camera_abstains)

    def striker_wrong_body(self, rng):
        return None if self.wrong_body is None else float(self.wrong_body)


def _v2(point=None, **kwargs):
    return bench.apply_rung(
        point or _point(),
        bench.REALISTIC_V2,
        _Samplers(),
        np.random.default_rng(0),
        realism=_Realism(**kwargs),
    )


def test_realistic_v2_refuses_to_run_without_the_measured_real_model():
    with pytest.raises(ValueError):
        bench.apply_rung(_point(), bench.REALISTIC_V2, _Samplers(), np.random.default_rng(0))


def test_the_v2_emission_carries_the_event_model_s_xy_head_and_not_the_track_pixel():
    """docs/wk1/bench_frames.md section 8's first honest difference, as a rung parameter."""
    point = _v2(head=(9.0, -5.0))
    for row in point["emission_contacts"] + point["emission_bounces"]:
        frame = int(row["frame"])
        if frame not in point["track"]:
            continue
        assert row["pixel"][0] == pytest.approx(point["track"][frame][0] + 9.0)
        assert row["pixel"][1] == pytest.approx(point["track"][frame][1] - 5.0)
    row = next(item for item in point["emission_rows"] if item["event_type"] == "contact")
    assert row["head_offset_px"] == [9.0, -5.0]
    assert row["pixel_source"].endswith("+xy_head")


def test_the_v2_emitted_frame_lands_on_the_measured_offset_from_the_true_event_time():
    for target in (0.0, 1.0, -1.0):
        point = _v2(frame_offset=target)
        for row in point["emission_rows"]:
            if row["event_type"] == "point_end" or row["spacing_adjusted"]:
                continue
            if "snapped" in row["pixel_source"]:
                # The emission was pulled back onto a tracked frame, which is the measured
                # real behaviour and is what the next test covers.
                continue
            # The bench's truth time is continuous and the emission is an integer frame, so
            # the achieved offset is the measured target plus at most half a frame of
            # quantisation.
            assert abs(row["offset_frames"] - target) <= 0.5 + 1e-9, row


def test_a_v2_emission_snaps_onto_a_tracked_frame_at_the_measured_rate():
    point = _point()
    dropped = _v2(point=point, off_track=False)
    assert all(
        int(row["frame"]) in dropped["track"]
        for row in dropped["emission_contacts"] + dropped["emission_bounces"]
    )
    kept = _v2(point=point, off_track=True)
    assert any(
        row["pixel_source"].startswith("projection_no_track_row") for row in kept["emission_rows"]
    ) or all(int(row["frame"]) in kept["track"] for row in kept["emission_contacts"])


def test_v2_holds_the_track_gaps_near_a_contact_at_the_measured_rate():
    """The dropout sampler drops frames 105-107; the real cohort has 1.4% of contacts with a
    gap within five frames, so at rate zero those frames come back."""
    realistic = bench.apply_rung(_point(), "realistic", _Samplers(), np.random.default_rng(0))
    assert 105 not in realistic["track"]
    restored = _v2(gap=0)
    assert 105 in restored["track"]
    recut = _v2(gap=3)
    missing = [frame for frame in recut["frames"] if frame not in recut["track"]]
    # A gap the dropout sampler put away from every contact is left where it is (106, 107);
    # inside a contact's window the gap is the measured length and no longer.
    assert missing
    for contact in (100, 120):
        near = [
            frame for frame in missing if abs(frame - contact) <= bench.CONTACT_GAP_WINDOW_FRAMES
        ]
        assert 0 < len(near) <= 3
    assert 105 in missing or 122 in missing


def test_v2_deletes_bounce_emissions_at_the_measured_per_flight_rate():
    assert _v2(drop_bounce=False, drop_terminal_bounce=False)["emission_bounces"] != []
    assert _v2(drop_bounce=True, drop_terminal_bounce=True)["emission_bounces"] == []
    only_terminal = _v2(drop_bounce=False, drop_terminal_bounce=True)
    assert [
        row["truth_frame"]
        for row in only_terminal["emission_rows"]
        if row["event_type"] == "bounce"
    ] == [110.0]


def test_the_v2_point_ending_does_not_ride_a_bounce_hundreds_of_frames_away():
    """docs/wk1/bench_frames.md section 8: on the noisy rungs the ending rides the nearest
    surviving bounce, which can be a different flight's."""
    point = _v2(drop_bounce=False, drop_terminal_bounce=True)
    ending = point["emission_point_end"]
    assert abs(float(ending["frame"]) - float(point["termination_frame"])) <= 2.0
    assert ending["source"] == "s6_point_bench_termination_emitted"


def test_v2_marks_the_points_whose_camera_abstained():
    assert _v2(camera_abstains=True)["camera_abstained"] is True
    assert _v2(camera_abstains=False)["camera_abstained"] is False


def test_the_camera_artifact_drops_only_the_abstained_clip(tmp_path):
    source = tmp_path / "camera_P_per_frame_v1.npz"
    np.savez(
        source,
        clips=np.asarray(["pt0001", "pt0001", "pt0002"], dtype="<U8"),
        frames=np.asarray([1, 2, 1], dtype=np.int32),
        P=np.zeros((3, 3, 4)),
    )
    target = tmp_path / "out.npz"
    bench._write_camera_without(source, target, {"pt0001"})
    with np.load(target, allow_pickle=False) as payload:
        assert list(payload["clips"]) == ["pt0002"]
        assert payload["P"].shape == (1, 3, 4)
    points = tmp_path / "court_H_per_point.npz"
    np.savez(points, pts=np.asarray([1, 2], dtype=np.int32), H=np.zeros((2, 3, 3)))
    bench._write_camera_without(points, tmp_path / "out2.npz", {"pt0001"})
    with np.load(tmp_path / "out2.npz", allow_pickle=False) as payload:
        assert list(payload["pts"]) == [2]


def test_the_wrong_body_moves_the_striker_only_at_the_sampled_contacts():
    point = _v2(wrong_body=6.0)
    assert point["striker_wrong_body"]
    host = _striker_host()
    right, _, _ = bench.striker_rows(
        host, {**point, "striker_wrong_body": {}}, rung="clean", seed=7
    )
    wrong, _, diagnostic = bench.striker_rows(host, point, rung=bench.REALISTIC_V2, seed=7)
    moved = {
        row["frame"]
        for row in wrong
        if not any(
            other["frame"] == row["frame"]
            and other["side"] == row["side"]
            and abs(other["court_x"] - row["court_x"]) < 1.0
            for other in right
        )
    }
    assert moved
    assert diagnostic["wrong_body_frames"] >= len(point["striker_wrong_body"])
    for frame in moved:
        centre = int(str(frame).removeprefix("f_").removesuffix(".jpg"))
        assert any(abs(centre - contact) <= 1 for contact in point["striker_wrong_body"]), centre


def test_realistic_v1_is_untouched_by_the_v2_parameters():
    """v1 must stay exactly what pass 7 and pass 8 swept, so both rungs stay comparable."""
    without = bench.apply_rung(_point(), "realistic", _Samplers(), np.random.default_rng(0))
    with_model = bench.apply_rung(
        _point(), "realistic", _Samplers(), np.random.default_rng(0), realism=_Realism()
    )
    assert without["emission_rows"] == with_model["emission_rows"]
    assert sorted(without["track"]) == sorted(with_model["track"])
    assert without["camera_abstained"] is False
    assert without["striker_wrong_body"] == {}


def test_a_timed_out_attempt_does_not_break_the_audit():
    """``--point-timeout-seconds`` writes an attempt row with every field null.

    ``docs/wk1/point_fit8.md`` section 8 recorded that the audit raised ``TypeError`` on such a
    report; it matches no truth flight instead, so the flight scores as unattempted.
    """
    rows = [
        {"start_frame": None, "skip_reason": "timeout"},
        {"start_frame": 120.0, "flight_index": 1},
    ]
    assert bench._nearest(rows, 120.4, "start_frame")["flight_index"] == 1
    assert bench._nearest([rows[0]], 120.4, "start_frame") is None
