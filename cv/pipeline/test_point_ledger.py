from cv.pipeline import point_ledger
from cv.pipeline.point_ledger import build_ledger


def score_rows(*times: float) -> list[dict]:
    return [
        {
            "t_start": str(time),
            "g1": str(index),
            "g2": "0",
            "p1": "0",
            "p2": "0",
            "sets1": "0",
            "sets2": "0",
        }
        for index, time in enumerate(times)
    ]


def peaks(*times: float) -> list[dict]:
    return [{"t": str(time), "z": "10"} for time in times]


def test_ledger_uses_score_transition_and_splits_quiet_attempts() -> None:
    segments = [{"t_start": "10", "t_end": "40"}]
    scores = [
        {"t_start": "0", "g1": "0", "g2": "0", "p1": "0", "p2": "0", "sets1": "", "sets2": ""},
        {"t_start": "24", "g1": "0", "g2": "0", "p1": "15", "p2": "0", "sets1": "", "sets2": ""},
    ]
    peaks = [{"t": "12", "z": "9"}, {"t": "16", "z": "8"}, {"t": "30", "z": "10"}]

    attempts, held = build_ledger(segments, scores, peaks, peak_z=5.0, quiet_gap_seconds=6.0)

    assert len(attempts) == 2
    assert attempts[0]["confidence"] == "high"
    assert attempts[0]["pt"] == 1
    assert attempts[1]["pt"] == 2
    assert not held


def test_ledger_suppresses_uncorroborated_quiet_gap() -> None:
    attempts, held = build_ledger(
        [{"t_start": "0", "t_end": "30"}],
        score_rows(),
        peaks(5.0, 15.0, 16.0),
        peak_z=5.0,
        quiet_gap_seconds=6.0,
    )

    assert len(attempts) == 1
    assert attempts[0]["rally_t_end"] == 30.0
    assert not held


def test_ledger_score_transition_corroborates_at_inclusive_window_edge() -> None:
    attempts, held = build_ledger(
        [{"t_start": "0", "t_end": "30"}],
        score_rows(0.0, 13.0),
        peaks(5.0, 15.0, 16.0),
        peak_z=5.0,
        quiet_gap_seconds=6.0,
    )

    assert len(attempts) == 2
    assert attempts[0]["rally_t_end"] == 10.0
    assert [row["reason"] for row in held] == ["no_impact_evidence"]


def test_ledger_play_camera_boundary_corroborates_at_inclusive_window_edge() -> None:
    attempts, held = build_ledger(
        [{"t_start": "0", "t_end": "20"}],
        score_rows(),
        peaks(0.0, 6.0),
        peak_z=5.0,
        quiet_gap_seconds=6.0,
    )

    assert len(attempts) == 2
    assert attempts[0]["rally_t_end"] == 3.0
    assert not held


def test_ledger_uses_edges_from_all_play_camera_segments() -> None:
    attempts, _ = build_ledger(
        [
            {"t_start": "0", "t_end": "30"},
            {"t_start": "13", "t_end": "14"},
        ],
        score_rows(),
        peaks(5.0, 15.0, 16.0),
        peak_z=5.0,
        quiet_gap_seconds=6.0,
    )

    first_segment_attempts = [row for row in attempts if row["segment_index"] == 0]
    assert len(first_segment_attempts) == 2
    assert first_segment_attempts[0]["rally_t_end"] == 10.0


def score_state_rows(*entries: tuple[float, str, str]) -> list[dict]:
    return [
        {
            "t": str(time),
            "server": "1",
            "sets_won_1": "0",
            "sets_won_2": "0",
            "set_number": "1",
            "games_1": "2",
            "games_2": "1",
            "points_1": p1,
            "points_2": p2,
            "completed_sets": "",
        }
        for time, p1, p2 in entries
    ]


def test_serve_mode_emits_one_row_per_attempt_with_the_score_it_was_played_on() -> None:
    from cv.pipeline.point_ledger import build_scored_ledger

    attempts = [
        {"rally_t_start": "10.0", "rally_t_end": "13.0"},
        {"rally_t_start": "25.0", "rally_t_end": "40.0"},
    ]
    peaks = [{"t": "10.4", "z": "9"}, {"t": "26.0", "z": "8"}, {"t": "31.0", "z": "7"}]
    scores = score_state_rows((0.0, "0", "0"), (44.0, "15", "0"))

    rows = build_scored_ledger(attempts, peaks, scores, peak_z=5.0)

    assert [row["pt"] for row in rows] == [1, 2]
    assert rows[0]["points_1"] == "0"
    assert rows[0]["impact_count"] == 1
    assert rows[0]["confidence"] == "provisional"
    assert rows[0]["outcome_hint"] == "fault_let_or_unscored"
    assert rows[1]["confidence"] == "high"
    assert rows[1]["outcome_hint"] == "point"
    assert rows[1]["score_transition_t"] == 44.0
    assert rows[1]["games_1"] == "2"


def test_serve_mode_without_a_score_decode_still_emits_attempts() -> None:
    from cv.pipeline.point_ledger import build_scored_ledger

    rows = build_scored_ledger(
        [{"rally_t_start": "5.0", "rally_t_end": "9.0"}], [], [], peak_z=5.0
    )

    assert len(rows) == 1
    assert rows[0]["score_source"] == "none"
    assert rows[0]["server"] == ""
    assert rows[0]["first_impact_t"] == 5.0


def _decode_rows() -> list[dict]:
    rows = []
    for second in range(0, 60):
        points = "0" if second < 30 else "15"
        rows.append(
            {
                "t": str(float(second)),
                "server": "1",
                "sets_won_1": "0",
                "sets_won_2": "0",
                "set_number": "1",
                "games_1": "0",
                "games_2": "0",
                "points_1": points,
                "points_2": "0",
                "completed_sets": "",
                "score_confidence": "0.9",
                "read_age_seconds": "2.0" if second < 40 else "90.0",
            }
        )
    return rows


def test_the_attached_score_is_read_a_little_way_into_the_rally() -> None:
    """The board is redrawn after the serve, so the second of the serve can be stale."""
    from cv.pipeline.point_ledger import build_scored_ledger

    attempts = [{"rally_t_start": "28.0", "rally_t_end": "36.0"}]

    at_serve = build_scored_ledger(
        attempts, [], _decode_rows(), peak_z=5.0, attach_lookahead_seconds=0.0
    )
    into_rally = build_scored_ledger(
        attempts, [], _decode_rows(), peak_z=5.0, attach_lookahead_seconds=5.0
    )

    assert at_serve[0]["points_1"] == "0"
    assert into_rally[0]["points_1"] == "15"


def test_the_lookahead_never_runs_past_the_end_of_the_rally() -> None:
    from cv.pipeline.point_ledger import build_scored_ledger

    attempts = [{"rally_t_start": "28.0", "rally_t_end": "29.0"}]

    rows = build_scored_ledger(
        attempts, [], _decode_rows(), peak_z=5.0, attach_lookahead_seconds=20.0
    )

    assert rows[0]["points_1"] == "0"


def test_a_state_resting_on_a_stale_read_is_withheld_not_guessed() -> None:
    from cv.pipeline.point_ledger import build_scored_ledger

    attempts = [{"rally_t_start": "20.0", "rally_t_end": "24.0"},
                {"rally_t_start": "50.0", "rally_t_end": "54.0"}]

    rows = build_scored_ledger(
        attempts, [], _decode_rows(), peak_z=5.0,
        attach_lookahead_seconds=5.0, max_read_age_seconds=15.0,
    )

    assert rows[0]["score_source"] == "score_grammar_v1"
    assert rows[1]["score_source"] == "abstained"
    assert rows[1]["games_1"] == ""
    assert rows[1]["points_1"] == ""
    assert rows[1]["score_confidence"] == ""


def test_a_low_confidence_state_is_withheld_too() -> None:
    from cv.pipeline.point_ledger import build_scored_ledger

    attempts = [{"rally_t_start": "20.0", "rally_t_end": "24.0"}]

    rows = build_scored_ledger(
        attempts, [], _decode_rows(), peak_z=5.0, min_confidence=0.95
    )

    assert rows[0]["score_source"] == "abstained"


def _emission(clip: str, frame: float, event_type: str = "contact") -> dict:
    return {"clip": clip, "frame": frame, "event_type": event_type, "abstain": False}


def test_attempt_gap_matches_the_fitters_own_shot_limit() -> None:
    from cv.pipeline.reconstruction import MAX_SHOT_SECONDS

    assert point_ledger.DEFAULT_ATTEMPT_GAP_SECONDS == MAX_SHOT_SECONDS


def test_split_disabled_keeps_one_attempt_carrying_the_point_id() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001", [[0.0, 100.0], [200.0, 300.0]], [10.0, 250.0], fps=25.0, enabled=False
    )
    assert [row["attempt_id"] for row in attempts] == ["m__pt0001"]
    assert attempts[0]["parent_point"] == "m__pt0001"
    assert attempts[0]["active_spans"] == [[0.0, 100.0], [200.0, 300.0]]


def test_split_cuts_between_active_play_spans() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001", [[0.0, 100.0], [200.0, 300.0]], [10.0, 250.0], fps=25.0, enabled=True
    )
    assert [row["attempt_id"] for row in attempts] == ["m__pt0001__a01", "m__pt0001__a02"]
    assert [row["parent_point"] for row in attempts] == ["m__pt0001"] * 2
    assert [row["active_spans"] for row in attempts] == [[[0.0, 100.0]], [[200.0, 300.0]]]
    assert attempts[1]["split_reasons"] == ["span_boundary"]


def test_split_cuts_midway_between_contacts_further_apart_than_the_shot_limit() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001", [[0.0, 400.0]], [10.0, 40.0, 300.0, 330.0], fps=25.0, enabled=True
    )
    assert [row["active_spans"] for row in attempts] == [[[0.0, 170.0]], [[170.0, 400.0]]]
    assert [row["contact_frames"] for row in attempts] == [[10.0, 40.0], [300.0, 330.0]]
    assert attempts[1]["split_reasons"] == ["contact_gap"]


def test_split_leaves_an_ordinary_rally_whole() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001", [[0.0, 200.0]], [10.0, 50.0, 90.0, 130.0], fps=25.0, enabled=True
    )
    assert [row["attempt_id"] for row in attempts] == ["m__pt0001"]
    assert attempts[0]["attempt_count"] == 1


def test_split_drops_a_span_with_no_contact_rather_than_emitting_dead_time() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001",
        [[0.0, 100.0], [200.0, 300.0], [400.0, 500.0]],
        [10.0, 250.0],
        fps=25.0,
        enabled=True,
    )
    assert [row["active_spans"] for row in attempts] == [[[0.0, 100.0]], [[200.0, 300.0]]]


def test_split_without_any_contact_keeps_the_point_whole() -> None:
    attempts = point_ledger.split_point_attempts(
        "m__pt0001", [[0.0, 100.0]], [], fps=25.0, enabled=True
    )
    assert [row["attempt_id"] for row in attempts] == ["m__pt0001"]


def test_attempt_ledger_over_a_cohort_counts_attempts_not_points() -> None:
    active_play = {
        "m/pt0001": {"fps": 25.0, "active_spans": [[0.0, 100.0], [200.0, 300.0]]},
        "m/pt0002": {"fps": 25.0, "active_spans": [[0.0, 100.0]]},
    }
    emissions = [
        _emission("m__pt0001", 10.0),
        _emission("m__pt0001", 250.0),
        _emission("m__pt0001", 260.0, "bounce"),
        _emission("m__pt0002", 20.0),
    ]
    off = point_ledger.build_attempt_ledger(active_play, emissions, split_attempts=False)
    on = point_ledger.build_attempt_ledger(active_play, emissions, split_attempts=True)
    assert [row["attempt_id"] for row in off] == ["m__pt0001", "m__pt0002"]
    assert [row["attempt_id"] for row in on] == [
        "m__pt0001__a01",
        "m__pt0001__a02",
        "m__pt0002",
    ]
    assert {row["parent_point"] for row in on} == {"m__pt0001", "m__pt0002"}
    assert [row["contacts"] for row in on] == [1, 1, 1]


def test_attempt_ledger_ignores_abstained_contacts() -> None:
    active_play = {"m/pt0001": {"fps": 25.0, "active_spans": [[0.0, 400.0]]}}
    emissions = [
        _emission("m__pt0001", 10.0),
        {**_emission("m__pt0001", 300.0), "abstain": True},
    ]
    rows = point_ledger.build_attempt_ledger(active_play, emissions, split_attempts=True)
    assert [row["attempt_id"] for row in rows] == ["m__pt0001"]


def test_write_attempt_ledger_round_trips(tmp_path) -> None:
    active_play = {"m/pt0001": {"fps": 25.0, "active_spans": [[0.0, 100.0], [200.0, 300.0]]}}
    emissions = [_emission("m__pt0001", 10.0), _emission("m__pt0001", 250.0)]
    rows = point_ledger.build_attempt_ledger(active_play, emissions, split_attempts=True)
    output = tmp_path / "attempts.csv"
    point_ledger.write_attempt_ledger(output, rows)
    written = point_ledger.read_rows(output)
    assert [row["attempt_id"] for row in written] == ["m__pt0001__a01", "m__pt0001__a02"]
    assert written[1]["active_spans"] == "200:300"
    assert written[1]["split_reasons"] == "span_boundary"


def _attempt(start: float, end: float, **score) -> dict:
    row = {
        "pt": 1,
        "rally_t_start": start,
        "rally_t_end": end,
        "first_impact_t": start,
        "impact_count": 1,
        "score_transition_t": "",
        "confidence": "provisional",
        "outcome_hint": "fault_let_or_unscored",
    }
    row.update({column: "" for column in point_ledger.SCORE_COLUMNS})
    row.update(score)
    return row


def _served(start: float, end: float, points: tuple[str, str], **extra) -> dict:
    return _attempt(
        start,
        end,
        server="1",
        sets_won_1="0",
        sets_won_2="0",
        set_number="1",
        games_1="0",
        games_2="0",
        points_1=points[0],
        points_2=points[1],
        completed_sets="",
        **extra,
    )


def test_a_fault_and_its_second_serve_are_one_point() -> None:
    rows = [_served(10.0, 14.0, ("15", "0")), _served(24.0, 40.0, ("15", "0"))]
    assert point_ledger.group_attempts_into_points(rows) == [[0, 1]]


def test_a_moved_score_opens_a_new_point() -> None:
    rows = [_served(10.0, 14.0, ("15", "0")), _served(24.0, 40.0, ("30", "0"))]
    assert point_ledger.group_attempts_into_points(rows) == [[0], [1]]


def test_a_serve_after_the_continuation_gap_opens_a_new_point() -> None:
    rows = [_served(10.0, 14.0, ("15", "0")), _served(60.0, 70.0, ("15", "0"))]
    assert point_ledger.group_attempts_into_points(rows) == [[0], [1]]


def test_an_abstained_attempt_inherits_the_standing_state() -> None:
    rows = [
        _served(10.0, 14.0, ("15", "0")),
        _attempt(20.0, 30.0),
        _served(40.0, 50.0, ("30", "0")),
    ]
    assert point_ledger.group_attempts_into_points(rows) == [[0, 1], [2]]


def test_annotate_points_labels_every_row() -> None:
    rows = [_served(10.0, 14.0, ("15", "0")), _served(24.0, 40.0, ("15", "0"))]
    point_ledger.annotate_points(rows)
    assert [row["attempt_role"] for row in rows] == ["first_serve", "continuation"]
    assert [row["point_index"] for row in rows] == [1, 1]
    assert [row["attempts_in_point"] for row in rows] == [2, 2]


def test_collapse_to_points_runs_the_first_serve_to_the_last_end() -> None:
    rows = [
        _served(10.0, 14.0, ("15", "0")),
        _served(24.0, 40.0, ("15", "0"), score_transition_t=41.0),
    ]
    points = point_ledger.collapse_to_points(rows)
    assert len(points) == 1
    assert points[0]["rally_t_start"] == 10.0
    assert points[0]["rally_t_end"] == 40.0
    assert points[0]["impact_count"] == 2
    assert points[0]["confidence"] == "high"
    assert points[0]["outcome_hint"] == "point"
    assert "attempt_role" not in points[0]


def test_the_point_ledger_writes_the_point_columns(tmp_path) -> None:
    rows = [_served(10.0, 14.0, ("15", "0")), _served(24.0, 40.0, ("15", "0"))]
    point_ledger.annotate_points(rows)
    output = tmp_path / "ledger.csv"
    point_ledger.write_ledger(output, rows)
    written = point_ledger.read_rows(output)
    assert [row["point_index"] for row in written] == ["1", "1"]
    assert [row["attempt_role"] for row in written] == ["first_serve", "continuation"]


def _write_csv(path, fields, rows) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _serve_mode_inputs(tmp_path):
    attempts = tmp_path / "serve_attempts_v1.csv"
    _write_csv(
        attempts,
        ["rally_t_start", "rally_t_end"],
        [
            {"rally_t_start": 10.0, "rally_t_end": 14.0},
            {"rally_t_start": 24.0, "rally_t_end": 40.0},
            {"rally_t_start": 100.0, "rally_t_end": 120.0},
        ],
    )
    score_state = tmp_path / "score_state_v1.csv"
    _write_csv(
        score_state,
        ["t", *point_ledger.SCORE_COLUMNS],
        [
            {
                "t": 9.0,
                "server": "1",
                "sets_won_1": "0",
                "sets_won_2": "0",
                "set_number": "1",
                "games_1": "0",
                "games_2": "0",
                "points_1": "15",
                "points_2": "0",
                "completed_sets": "",
            },
            {
                "t": 95.0,
                "server": "1",
                "sets_won_1": "0",
                "sets_won_2": "0",
                "set_number": "1",
                "games_1": "0",
                "games_2": "0",
                "points_1": "30",
                "points_2": "0",
                "completed_sets": "",
            },
        ],
    )
    peaks = tmp_path / "serve_audio_peaks_v1.csv"
    _write_csv(peaks, ["t", "z"], [{"t": 11.0, "z": 9.0}, {"t": 101.0, "z": 9.0}])
    return attempts, score_state, peaks


def _run_point_ledger(monkeypatch, tmp_path, extra: list[str]) -> tuple[list[dict], dict]:
    import json
    import sys

    attempts, score_state, peaks = _serve_mode_inputs(tmp_path)
    output = tmp_path / "automatic_point_ledger_v1.csv"
    report = tmp_path / "automatic_point_ledger_v1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "point_ledger",
            "--mode",
            "serve",
            "--attempts",
            str(attempts),
            "--score-state",
            str(score_state),
            "--serve-peaks",
            str(peaks),
            "--output",
            str(output),
            "--report",
            str(report),
            *extra,
        ],
    )
    point_ledger.main()
    return point_ledger.read_rows(output), json.loads(report.read_text())


def test_the_serve_ledger_groups_attempts_into_points_by_default(monkeypatch, tmp_path) -> None:
    rows, report = _run_point_ledger(monkeypatch, tmp_path, [])

    # one row per detected serve, which is the unit every downstream clip is cut for
    assert len(rows) == 3
    assert [row["point_index"] for row in rows] == ["1", "1", "2"]
    assert [row["attempt_role"] for row in rows] == [
        "first_serve",
        "continuation",
        "first_serve",
    ]
    assert [row["attempts_in_point"] for row in rows] == ["2", "2", "1"]
    assert report["attempts"] == 3
    assert report["score_points"] == 2
    assert report["continuation_attempts"] == 1
    assert report["multi_attempt_points"] == 1
    assert report["point_continuation_gap_seconds"] == 15.0


def test_a_shorter_continuation_gap_splits_the_second_serve_off(monkeypatch, tmp_path) -> None:
    rows, report = _run_point_ledger(monkeypatch, tmp_path, ["--point-continuation-gap", "5"])

    assert [row["point_index"] for row in rows] == ["1", "2", "3"]
    assert report["score_points"] == 3
    assert report["continuation_attempts"] == 0


def test_point_rows_are_still_available_behind_the_flag(monkeypatch, tmp_path) -> None:
    rows, report = _run_point_ledger(monkeypatch, tmp_path, ["--rows", "points"])

    assert len(rows) == 2
    assert [row["rally_t_start"] for row in rows] == ["10.0", "100.0"]
    assert rows[0]["rally_t_end"] == "40.0"
    assert "attempt_role" not in rows[0]
    assert report["rows"] == "points"
