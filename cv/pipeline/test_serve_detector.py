import json

import numpy as np
import pytest

from cv.pipeline import serve_detector
from cv.pipeline.serve_detector import (
    MINIMUM_START_GAP_SECONDS,
    assemble_attempts,
    grid_times,
    peaks,
    soft_labels,
)


@pytest.mark.parametrize(
    "change", ["none", "score", "motion", "audio", "views", "players", "toss_added", "output"]
)
def test_span_feature_cache_tracks_all_upstream_evidence(tmp_path, monkeypatch, change):
    np.savez(tmp_path / "shot_signals_v1.npz", motion=np.ones(100), fps=5.0, start_seconds=0.0)
    np.savez(tmp_path / "audio_flux_v1.npz", z=np.ones(100), hop_seconds=0.2, start_seconds=0.0)
    np.savez(tmp_path / "player_boxes_v1.npz", rows=np.zeros((20, 4)))
    (tmp_path / "frames_native.json").write_text(json.dumps({"fps": 1.0, "start_seconds": 0.0}))
    (tmp_path / "shot_boundaries_v1.csv").write_text("shot_index,t_start,t_end\n0,0,20\n")
    views = tmp_path / "shot_views_v1.csv"
    views.write_text("shot_index,play,replay,closeup,crowd,graphic\n0,1,0,0,0,0\n")
    score = tmp_path / "score.csv"
    score.write_text("t,p1,p2\n0,0,0\n")
    calls = []

    def build(times, **kwargs):
        calls.append(kwargs)
        return np.full((len(times), 2), float(len(calls)))

    monkeypatch.setattr(serve_detector, "build_features", build)
    serve_detector.assemble_span_features(tmp_path, score_runs=score)
    if change == "score":
        score.write_text("t,p1,p2\n0,0,0\n5,15,0\n")
    elif change == "motion":
        np.savez(
            tmp_path / "shot_signals_v1.npz", motion=np.ones(100) * 2, fps=5.0, start_seconds=0.0
        )
    elif change == "audio":
        np.savez(
            tmp_path / "audio_flux_v1.npz", z=np.ones(100) * 2, hop_seconds=0.2, start_seconds=0.0
        )
    elif change == "views":
        views.write_text("shot_index,play,replay,closeup,crowd,graphic\n0,0,1,0,0,0\n")
    elif change == "players":
        np.savez(tmp_path / "player_boxes_v1.npz", rows=np.ones((20, 4)))
    elif change == "toss_added":
        np.savez(tmp_path / "toss_windows_v1.npz", times=np.array([1.0]), values=np.ones((1, 8)))
    elif change == "output":
        (tmp_path / "serve_features_v1.npz").write_bytes(b"corrupt")
    features, _, _ = serve_detector.assemble_span_features(tmp_path, score_runs=score)
    assert len(calls) == (1 if change == "none" else 2)
    assert np.all(features == len(calls))


def test_grid_times_are_evenly_spaced_inside_the_span() -> None:
    times = grid_times(10.0, 11.0)

    assert times[0] == 10.0
    assert np.allclose(np.diff(times), 0.2)
    assert times[-1] <= 11.0


def test_soft_labels_mark_a_tolerance_window_around_each_anchor() -> None:
    times = grid_times(0.0, 4.0)

    labels = soft_labels(times, np.array([2.0]), 0.3)

    assert labels[np.argmin(np.abs(times - 2.0))] == 1
    assert labels.sum() == 3  # 1.8, 2.0, 2.2
    assert labels[0] == 0


def test_peaks_enforce_the_minimum_gap() -> None:
    times = grid_times(0.0, 20.0)
    scores = np.zeros(len(times))
    scores[np.argmin(np.abs(times - 5.0))] = 0.9
    scores[np.argmin(np.abs(times - 6.0))] = 0.8  # inside the gap, must be suppressed
    scores[np.argmin(np.abs(times - 12.0))] = 0.7

    chosen = peaks(scores, times, threshold=0.5, minimum_gap=MINIMUM_START_GAP_SECONDS)

    assert [round(times[index], 1) for index in chosen] == [5.0, 12.0]


def test_each_start_opens_one_attempt_that_closes_before_the_next() -> None:
    times = grid_times(0.0, 40.0)
    start_scores = np.zeros(len(times))
    end_scores = np.zeros(len(times))
    for moment, value in ((5.0, 0.9), (20.0, 0.9)):
        start_scores[np.argmin(np.abs(times - moment))] = value
    for moment in (9.0, 26.0):
        end_scores[np.argmin(np.abs(times - moment))] = 0.9
    shots = [{"t_start": 0.0, "t_end": 40.0}]

    attempts = assemble_attempts(
        times,
        start_scores,
        end_scores,
        start_threshold=0.5,
        end_threshold=0.5,
        shots=shots,
        score_changes=np.empty(0),
        play_mask=np.ones(len(times), dtype=bool),
    )

    assert [(round(row["rally_t_start"], 1), round(row["rally_t_end"], 1)) for row in attempts] == [
        (5.0, 9.0),
        (20.0, 26.0),
    ]


def test_a_start_outside_the_play_view_is_not_emitted() -> None:
    times = grid_times(0.0, 20.0)
    start_scores = np.zeros(len(times))
    start_scores[np.argmin(np.abs(times - 5.0))] = 0.9
    shots = [{"t_start": 0.0, "t_end": 20.0}]

    attempts = assemble_attempts(
        times,
        start_scores,
        np.zeros(len(times)),
        start_threshold=0.5,
        end_threshold=0.5,
        shots=shots,
        score_changes=np.empty(0),
        play_mask=np.zeros(len(times), dtype=bool),
    )

    assert attempts == []


@pytest.mark.parametrize("shot_end", [5.2, 5.8])
def test_short_tail_cannot_be_extended_past_its_camera_shot(shot_end: float) -> None:
    times = grid_times(0.0, 10.0)
    start_scores = np.zeros(len(times))
    start_scores[np.argmin(np.abs(times - 5.0))] = 0.9

    attempts = assemble_attempts(
        times,
        start_scores,
        np.zeros(len(times)),
        start_threshold=0.5,
        end_threshold=0.5,
        shots=[{"t_start": 0.0, "t_end": shot_end}],
        score_changes=np.empty(0),
        play_mask=times < shot_end,
        gap_fill_seconds=0.0,
    )

    # An insufficient visible tail is an abstention, not permission to invent
    # the minimum duration across a camera cut.
    assert attempts == []


def test_gap_fill_recovers_a_sub_threshold_start_next_to_an_impact() -> None:
    from cv.pipeline.serve_detector import fill_start_gaps

    times = grid_times(0.0, 120.0)
    scores = np.zeros(len(times))
    scores[np.argmin(np.abs(times - 10.0))] = 0.9
    scores[np.argmin(np.abs(times - 60.0))] = 0.4  # a serve the head under-scored
    scores[np.argmin(np.abs(times - 110.0))] = 0.9
    accepted = peaks(scores, times, threshold=0.7, minimum_gap=MINIMUM_START_GAP_SECONDS)

    filled = fill_start_gaps(
        accepted,
        scores,
        times,
        np.ones(len(times), dtype=bool),
        onsets=np.array([60.2]),
        gap_seconds=34.0,
        threshold=0.25,
    )

    assert [round(times[i], 1) for i in filled] == [10.0, 60.0, 110.0]


def test_gap_fill_needs_an_audible_impact() -> None:
    from cv.pipeline.serve_detector import fill_start_gaps

    times = grid_times(0.0, 120.0)
    scores = np.zeros(len(times))
    scores[np.argmin(np.abs(times - 10.0))] = 0.9
    scores[np.argmin(np.abs(times - 60.0))] = 0.4
    scores[np.argmin(np.abs(times - 110.0))] = 0.9
    accepted = peaks(scores, times, threshold=0.7, minimum_gap=MINIMUM_START_GAP_SECONDS)

    filled = fill_start_gaps(
        accepted,
        scores,
        times,
        np.ones(len(times), dtype=bool),
        onsets=np.array([5.0]),
        gap_seconds=34.0,
        threshold=0.25,
    )

    assert [round(times[i], 1) for i in filled] == [10.0, 110.0]


def test_the_end_falls_back_to_the_last_impact_not_the_next_boundary() -> None:
    times = grid_times(0.0, 120.0)
    start_scores = np.zeros(len(times))
    start_scores[np.argmin(np.abs(times - 10.0))] = 0.9
    shots = [{"t_start": 0.0, "t_end": 120.0}]

    attempts = assemble_attempts(
        times,
        start_scores,
        np.zeros(len(times)),  # the end head never fires
        start_threshold=0.7,
        end_threshold=0.5,
        shots=shots,
        score_changes=np.empty(0),
        play_mask=np.ones(len(times), dtype=bool),
        onsets=np.array([10.4, 12.0, 14.4]),
        last_impact_buffer=1.6,
        gap_fill_seconds=0.0,
    )

    assert len(attempts) == 1
    assert round(attempts[0]["rally_t_end"], 1) == 16.0


def test_peak_ties_use_source_time_independent_of_padding_or_row_order():
    times = np.array([10.2, 10.0, 30.0])
    scores = np.array([0.9, 0.9, 0.95])
    for order in (np.arange(3), np.array([2, 1, 0])):
        t, p = times[order], scores[order]
        selected = peaks(p, t, threshold=0.7, minimum_gap=1.0)
        assert sorted(t[selected]) == [10.0, 30.0]
        t, p = np.r_[0.0, t, 100.0], np.r_[0.01, p, 0.01]
        selected = peaks(p, t, threshold=0.7, minimum_gap=1.0)
        assert sorted(t[selected]) == [10.0, 30.0]


@pytest.mark.parametrize("sample_count", [1, 11, 100])
def test_span_features_include_available_boundary_samples(tmp_path, monkeypatch, sample_count):
    start, fps = 7.25, 5.0
    np.savez(
        tmp_path / "shot_signals_v1.npz", motion=np.ones(sample_count), fps=fps, start_seconds=start
    )
    np.savez(
        tmp_path / "audio_flux_v1.npz",
        z=np.ones(sample_count),
        hop_seconds=1 / fps,
        start_seconds=start,
    )
    np.savez(tmp_path / "player_boxes_v1.npz", rows=np.zeros((sample_count, 4)))
    (tmp_path / "frames_native.json").write_text(json.dumps({"fps": fps, "start_seconds": start}))
    (tmp_path / "shot_boundaries_v1.csv").write_text(
        f"shot_index,t_start,t_end\n0,{start},{start + sample_count / fps}\n"
    )
    (tmp_path / "shot_views_v1.csv").write_text(
        "shot_index,play,replay,closeup,crowd,graphic\n0,1,0,0,0,0\n"
    )
    seen = []

    def build(times, **kwargs):
        seen.extend(times)
        return np.zeros((len(times), 76))

    monkeypatch.setattr(serve_detector, "build_features", build)
    _, times, mask = serve_detector.assemble_span_features(tmp_path, cache=False)
    assert times[0] == start
    assert times[-1] <= start + (sample_count - 1) / fps + 1e-12
    assert len(times) == sample_count
    assert seen == list(times)
    assert mask.all()
    interior = grid_times(start + 3.0, start + sample_count / fps - 3.0)
    if len(interior):
        np.testing.assert_array_equal(times[15 : 15 + len(interior)], interior)


def test_real_boundary_features_have_finite_available_support():
    # 1.36s of native 25Hz motion, shorter audio, and no player observations:
    # neither edge has a complete multi-second feature window.
    features = serve_detector.build_features(
        np.array([0.0, 1.2]),
        signals={
            "fps": 25.0,
            "start_seconds": 0.0,
            "motion": np.linspace(0.0, 1.0, 34),
            "motion_top": np.linspace(0.0, 0.5, 34),
            "motion_bottom": np.linspace(0.5, 1.0, 34),
        },
        audio_z=np.array([0.0, 6.0, 1.0, 0.0]),
        audio_hop=0.2,
        audio_offset=0.0,
        shots=[{"t_start": 0.0, "t_end": 1.36}],
        view_probabilities={0: {"play": 1.0}},
        player_rows=np.empty((0, 7)),
        player_times=np.empty(0),
        score_changes=np.empty(0),
    )
    assert features.shape == (2, len(serve_detector.feature_names()))
    assert np.isfinite(features).all()
    assert features[0, serve_detector.feature_names().index("shot_elapsed")] == 0
    assert features[1, serve_detector.feature_names().index("shot_remaining")] > 0
