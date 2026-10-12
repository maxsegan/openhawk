"""Band abstentions released only with a track turn, a live span and an interior place."""

from __future__ import annotations

import numpy as np

from cv.pipeline import abstained_release as release
from cv.pipeline import s6_labeled_stage as stage

MATCH = "m"
CLIP = "pt0001"


def _row(frame, event_type="contact", *, marginal=0.9, abstain=True, reasons=()):
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": event_type,
        "frame": frame,
        "abstain": abstain,
        "gate_held": bool(reasons),
        "model_abstain": abstain,
        "acceptance_marginal": marginal if abstain else 0.999,
        "decision_threshold": 0.991,
        "point_gate_failure_reasons": list(reasons),
    }


def _track(pivot=50):
    """Ball moving right, then turning back left at ``pivot``."""

    track = {}
    for t in range(0, 101):
        x = 100 + 10 * t if t <= pivot else 100 + 10 * pivot - 10 * (t - pivot)
        track[t] = np.asarray([float(x), 500.0])
    return track


def _context(track=None, live=((0, 100),)):
    return {
        "track": _track() if track is None else track,
        "live_spans": [list(span) for span in live],
        "camera_spans": [[0, 100]],
        "cadence": None,
    }


def _run(rows, context, window=(0, 100)):
    return release.release_supported_abstentions(
        rows, match_id=MATCH, clip=CLIP, window=window, context=context, release=True
    )


def _rally(candidate):
    return [_row(20, abstain=False), candidate, _row(80, abstain=False)]


def test_a_supported_interior_band_contact_is_released():
    rows, census = _run(_rally(_row(50)), _context())
    assert rows[1]["abstain"] is False
    assert rows[1]["held_release_v2"]["turn_degrees"] >= 30
    assert len(census["released"]) == 1


def test_no_turn_below_floor_outside_live_or_after_the_last_contact_stays():
    straight = {t: np.asarray([100.0 + 10 * t, 500.0]) for t in range(101)}
    for rows, context in (
        (_rally(_row(50)), _context(track=straight)),
        (_rally(_row(50, marginal=0.7)), _context()),
        (_rally(_row(50)), _context(live=((0, 40),))),
        ([_row(20, abstain=False), _row(50)], _context()),
    ):
        out, census = _run(rows, context)
        assert all(row["abstain"] for row in out if row["frame"] == 50)
        assert census["released"] == []


def test_an_accepted_point_end_before_the_next_contact_blocks():
    rows = [_row(20, abstain=False), _row(50), _row(60, "point_end", abstain=False)]
    rows.append(_row(80, abstain=False))
    out, _ = _run(rows, _context())
    assert out[1]["abstain"] is True


def test_an_unreleasable_hold_stays():
    out, census = _run(_rally(_row(50, reasons=["court_geometry_failure"])), _context())
    assert out[1]["abstain"] is True
    assert census["kept"] == {"hold_not_releasable": 1}


def test_off_returns_rows_unchanged_and_the_key_is_default_off():
    rows = _rally(_row(50))
    out, _ = release.release_supported_abstentions(
        rows, match_id=MATCH, clip=CLIP, window=(0, 100), context=_context(), release=False
    )
    assert out is rows
    assert release.POLICY_KEY not in stage.shared_settings({})
