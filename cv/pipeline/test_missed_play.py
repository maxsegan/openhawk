"""Missed-play flags: score changes, play view, coverage and empty matches on synthetic timelines."""

import json

import numpy as np

from cv.pipeline import missed_play as mp


def _write_csv(path, header, rows):
    path.write_text(",".join(header) + "\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))


def _match(tmp_path, *, attempts, changes, duration=2000.0, play=None, peaks_until=None):
    """An upstream folder: ledger attempts, one decoded score state a second, views, peaks."""
    up = tmp_path / "upstream" / "m"
    up.mkdir(parents=True)
    _write_csv(up / mp.LEDGER_NAME, ["rally_t_start", "rally_t_end"], attempts)
    states, points, changed = [], 0, set(int(t) for t in changes)
    for t in range(int(duration)):
        if t in changed:
            points += 1
        age = 1e6 if t < 5 else 0.0
        states.append([t, 0, 0, 0, 0, points // 4, points % 4, 0, age])
    _write_csv(
        up / "score_state_v1.csv",
        [
            "t",
            "sets_won_1",
            "sets_won_2",
            "games_1",
            "games_2",
            "points_1",
            "points_2",
            "score_confidence",
            "read_age_seconds",
        ],
        states,
    )
    _write_csv(
        up / "shot_views_v1.csv",
        ["shot_index", "t_start", "t_end", "view"],
        [[i, a, b, v] for i, (a, b, v) in enumerate(play or [(0.0, duration, "play")])],
    )
    end = peaks_until or duration
    _write_csv(
        up / "serve_audio_peaks_v1.csv", ["t", "z"], [[t, 6.0] for t in np.arange(0, end, 3.3)]
    )
    (up / "source_integrity.json").write_text(json.dumps({"duration_seconds": duration}))
    return up


def _regular(lo, hi, step=30.0, length=8.0):
    return [[t, t + length] for t in np.arange(lo, hi, step)]


def test_score_changes_without_attempts_are_flagged_and_explained_ones_are_not(tmp_path):
    attempts = _regular(100, 1000) + _regular(1600, 1900)
    # The board moves 20 s after every attempt, and also every 30 s inside the hole.
    changes = [a[1] + 20 for a in attempts] + list(np.arange(1030, 1600, 30.0))
    document = mp.build(_match(tmp_path, attempts=attempts, changes=changes))
    # 1030 s is within the score lag of the last attempt before the hole (ends 978 s).
    assert document["counts"]["unexplained_score_changes"] == len(np.arange(1060, 1600, 30.0))
    [flag] = document["flags"]
    assert flag["gap"] == [978.0, 1600.0]
    assert flag["confidence"] == "high"
    assert flag["likely_cause"] == "serve_detection_miss"
    assert flag["long_gap"] and flag["play_view_seconds"] > 400


def test_a_clean_match_has_no_flags_and_the_first_read_is_not_a_change(tmp_path):
    attempts = _regular(100, 1900)
    changes = [5] + [a[1] + 20 for a in attempts]
    document = mp.build(_match(tmp_path, attempts=attempts, changes=changes))
    assert document["flags"] == []
    assert document["counts"]["score_changes"] == len(attempts)


def test_a_stream_that_stops_short_is_a_high_confidence_coverage_flag(tmp_path):
    attempts = _regular(100, 900)
    changes = [a[1] + 20 for a in attempts] + list(np.arange(930, 1990, 30.0))
    document = mp.build(_match(tmp_path, attempts=attempts, changes=changes, peaks_until=900))
    covered = [f for f in document["flags"] if f["coverage"]]
    assert covered and covered[0]["confidence"] == "high"
    assert covered[0]["likely_cause"] == "source_read_short:serve_audio"
    assert covered[0]["t0"] >= 897 and document["counts"]["coverage_short_streams"] == [
        "serve_audio"
    ]


def test_a_recorded_coverage_record_is_used(tmp_path):
    up = _match(tmp_path, attempts=_regular(100, 1900), changes=[])
    (up / "frames_native.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "reader": "native_view_frames",
                    "reached_seconds": 700.0,
                    "expected_seconds": 2000.0,
                    "complete": False,
                }
            }
        )
    )
    assert mp.coverage_gaps(up, 2000.0)[0]["reader"] == "native_view_frames"


def test_pre_match_score_changes_stay_low(tmp_path):
    attempts = _regular(400, 1900)
    changes = [60, 120] + [a[1] + 20 for a in attempts]
    [flag] = mp.build(_match(tmp_path, attempts=attempts, changes=changes))["flags"]
    assert flag["edge"] == "before_first_attempt" and flag["confidence"] == "low"


def test_a_match_without_attempts_on_one_long_shot_is_flagged_whole(tmp_path):
    up = _match(tmp_path, attempts=[], changes=[], play=[(0.0, 2000.0, "graphic")])
    [flag] = mp.build(up)["flags"]
    assert flag["whole_match"] and flag["likely_cause"] == "continuous_feed_view_gating"
    assert (flag["t0"], flag["t1"]) == (0.0, 2000.0)


def test_write_puts_the_record_beside_the_outcomes(tmp_path):
    _match(tmp_path, attempts=_regular(100, 1900), changes=[])
    document = mp.write(tmp_path, "m")
    assert json.loads((tmp_path / mp.FILE).read_text())["schema"] == mp.SCHEMA
    assert mp.summary(document)["flags"] == 0
