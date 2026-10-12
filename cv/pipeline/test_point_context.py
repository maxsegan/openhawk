import json

import numpy as np
import pytest

from cv.pipeline import point_context as context
from cv.pipeline import serve_speed_graphic as graphic


def segment(point: str = "pt0001", *, timeline: bool = True) -> context.Segment:
    return context.Segment("match", point, 25.0, 20.0, 100.0, source_timeline_available=timeline)


def score_read(frame: int, p1: str, p2: str, server: int | None = 1) -> graphic.ScoreOverlayReading:
    return graphic.ScoreOverlayReading(
        frame=frame,
        present=True,
        rows=(
            {"name": "ONE", "values": ["3"], "points": p1},
            {"name": "TWO", "values": ["2"], "points": p2},
        ),
        serving_row=server,
        confidence=0.9,
        abstained=False,
        abstention_reason=None,
        roi_native_xyxy=(0, 700, 700, 1080),
        crop_sha256="a" * 64,
        raw_text="ONE 2 3 15 TWO 4 2 0",
        words=(),
    )


def test_scoreboard_diff_recovers_receiver_winner() -> None:
    result = context.scoreboard_diff(score_read(1, "15", "0"), score_read(2, "15", "15"))
    assert not result["abstained"]
    assert result["winner"] == "receiver"
    assert result["winner_row"] == 2


def test_scoreboard_diff_abstains_without_server_marker() -> None:
    result = context.scoreboard_diff(score_read(1, "15", "0", None), score_read(2, "30", "0", None))
    assert result["abstained"]
    assert result["abstention_reason"] == "server_row_ambiguous"


def test_scoreboard_lookahead_uses_first_legal_read() -> None:
    unreadable = graphic.ScoreOverlayReading(
        2, False, (), None, 0.0, True, "not_readable", None, None, "", ()
    )
    result = context.scoreboard_lookahead_evidence(
        score_read(1, "15", "0"),
        [
            (segment("pt0002"), unreadable),
            (segment("pt0003"), score_read(3, "15", "15")),
        ],
    )
    assert not result["abstained"]
    assert result["winner"] == "receiver"
    assert result["lookahead_offset"] == 2
    assert len(result["lookahead_attempts"]) == 2


def test_adjacency_windows_are_broadcast_local() -> None:
    windows = context.adjacency_windows(
        [segment("pt0001"), segment("pt0002"), context.Segment("other", "pt0001", 25, 1, 2)]
    )
    assert [row["point_id"] for row in windows[("match", "pt0001")]["next"]] == ["pt0002"]
    assert windows[("other", "pt0001")]["previous"] == []


def test_sampled_cohort_does_not_treat_concatenation_as_adjacency() -> None:
    server = context.ServerPosition("near", "ad", 4.0, -1.0, 30.0, 0.8, None, {})
    result = context.segment_adjacency_evidence(
        segment(timeline=False), segment("pt0002", timeline=False), server, server, []
    )
    assert result["abstained"]
    assert result["abstention_reason"] == "sampled_audit_reel_has_no_original_gap"


def test_same_server_and_side_is_fault_or_let_evidence() -> None:
    current = context.Segment("match", "pt0001", 25.0, 1, 50, 1.0, 3.0, True)
    following = context.Segment("match", "pt0002", 25.0, 1, 50, 8.0, 10.0, True)
    server = context.ServerPosition("far", "deuce", 4.0, 24.0, 20.0, 0.9, None, {})
    result = context.segment_adjacency_evidence(
        current,
        following,
        server,
        server,
        [{"event_type": "contact", "frame": 20}, {"event_type": "net_hit", "frame": 25}],
    )
    assert result["relation"] == "same_server_same_side_fault_or_let"
    assert result["let_likelihood_within_fault_or_let"] == pytest.approx(0.75)


def test_server_end_and_side_come_from_contact_nearest_box() -> None:
    events = [
        {
            "event_type": "contact",
            "frame": 30,
            "location": {"image_x": 800.0, "image_y": 400.0},
        }
    ]
    rows = [
        {
            "clip": "pt0001",
            "frame": "f_0030.jpg",
            "x0": "380",
            "y0": "170",
            "x1": "430",
            "y1": "230",
            "court_x": "4.0",
            "court_y": "-1.0",
        },
        {
            "clip": "pt0001",
            "frame": "f_0030.jpg",
            "x0": "600",
            "y0": "30",
            "x1": "630",
            "y1": "75",
            "court_x": "7.0",
            "court_y": "24.5",
        },
    ]
    # The rows are in the legacy 960x540 box space; the loader stamps the scale it
    # resolved from the artifact's coordinate sidecar, never from the pixel magnitudes.
    for row in rows:
        row["_scale_x"] = "2.0"
        row["_scale_y"] = "2.0"
    result = context.infer_server_position(segment(), events, rows)
    assert result.end == "near"
    assert result.side == "ad"


def test_player_rows_carry_the_sidecar_declared_scale(tmp_path) -> None:
    from cv.pipeline import resolution as res

    match_dir = tmp_path
    path = match_dir / "player_boxes_yolo_native_sided_v1.csv"
    path.write_text(
        "clip,frame,x0,y0,x1,y1,court_x,court_y\npt0001,f_0030.jpg,380,170,430,230,4.0,-1.0\n"
    )
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        legacy_size=res.CANONICAL_SIZE,
        source="test",
        native_columns=("x0_native", "y0_native", "x1_native", "y1_native"),
        legacy_columns=("x0", "y0", "x1", "y1"),
        extra={"artifact": path.name},
    )
    rows = context._load_player_rows(match_dir)
    assert rows and rows[0]["_scale_x"] == "2.0" and rows[0]["_scale_y"] == "2.0"
    # No sidecar: the reader fails closed instead of guessing.
    path.with_name(path.name + ".coordinates.json").unlink()
    assert context._load_player_rows(match_dir) == []


def test_audio_call_witness_preserves_frame_ticks(tmp_path) -> None:
    rng = np.random.default_rng(3)
    samples = 0.001 * rng.standard_normal(context.AUDIO_SAMPLE_RATE * 4).astype(np.float32)
    # A voiced 700 Hz burst 0.5 seconds after the candidate.
    start = int(2.5 * context.AUDIO_SAMPLE_RATE)
    stop = int(2.75 * context.AUDIO_SAMPLE_RATE)
    times = np.arange(stop - start) / context.AUDIO_SAMPLE_RATE
    samples[start:stop] += 0.5 * np.sin(2 * np.pi * 700 * times)
    output = tmp_path / "strip.png"
    result = context.audio_call_witness(samples, fps=25.0, candidate_frame=51.0, render_path=output)
    assert not result["abstained"]
    assert result["out_call_likelihood_ratio"] > 1.0
    assert result["aligned_frame_ticks"]
    assert output.is_file()


def test_source_timeline_auto_hydrates_symlinked_raw_run(tmp_path) -> None:
    source = tmp_path / "source.mkv"
    source.touch()
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / "audit_reel_native_1080.mp4").symlink_to(source)
    (match_dir / "audit_reel_point_map.csv").write_text(
        "pt,rally_t_start,rally_t_end\n1,100.0,110.0\n"
    )
    hydrated = context._source_timeline_segments([segment()], match_dir, None)
    assert hydrated[0].source_timeline_available
    assert hydrated[0].source_start_seconds == pytest.approx(100.76)
    assert hydrated[0].source_end_seconds == pytest.approx(103.96)


def test_source_timeline_auto_keeps_concatenated_reel_fallback(tmp_path) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / "audit_reel_native_1080.mp4").touch()
    (match_dir / "audit_reel_point_map.csv").write_text(
        "pt,rally_t_start,rally_t_end\n1,100.0,110.0\n"
    )
    hydrated = context._source_timeline_segments([segment()], match_dir, None)
    assert not hydrated[0].source_timeline_available


def test_explicit_score_state_ledger_supplies_source_timeline_read(tmp_path) -> None:
    ledger = tmp_path / "match" / "score_state_v1.csv"
    ledger.parent.mkdir()
    ledger.write_text(
        "t,server,games_1,games_2,points_1,points_2,score_confidence\n10.0,1,3,2,15,0,0.9\n"
    )
    readings = context.load_score_state_ledger(ledger)
    source_segment = context.Segment(
        "match",
        "pt0001",
        30.0,
        1.0,
        120.0,
        source_start_seconds=10.1,
        source_end_seconds=14.0,
        source_timeline_available=True,
    )

    reading = context._score_at_segment_start(ledger.parent, source_segment, readings)

    assert reading.rows[0]["points"] == "15"
    assert reading.serving_row == 1
    assert reading.frame == 304
    assert reading.crop_sha256 == context._cached_file_record(str(ledger.resolve()))["sha256"]


def test_completed_second_serve_relation_preserves_ordinal() -> None:
    prior = context.build_prior(
        segment(),
        segment_index=0,
        adjacency={
            "abstained": False,
            "relation": "service_side_alternated_completed_point",
            "completed_serve_ordinal": "second",
        },
        scoreboard={"abstained": True},
        audio_call={"abstained": True},
        shipped_ending={"abstained": True},
        provenance_receipt={"human_derived_inputs": []},
    )
    assert prior["serve_ordinal"]["second"] > prior["serve_ordinal"]["first"]


def test_prior_schema_rejects_human_input() -> None:
    prior = context.build_prior(
        segment(),
        segment_index=0,
        adjacency={"abstained": True},
        scoreboard={"abstained": True},
        audio_call={"abstained": True},
        shipped_ending={"abstained": True},
        provenance_receipt={"human_derived_inputs": []},
    )
    context.validate_prior(prior)
    assert sum(prior["ending_kind"].values()) == pytest.approx(1.0)
    changed = json.loads(json.dumps(prior))
    changed["provenance"]["human_derived_inputs"] = ["truth.json"]
    with pytest.raises(ValueError, match="human-derived"):
        context.validate_prior(changed)
