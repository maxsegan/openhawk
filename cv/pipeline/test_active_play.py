"""Tests for the composed active-play trimmer.

The load-bearing properties: a trim must never exclude a frame inside an accepted active
span (safety), interior dead time must be annotated rather than cut (phase is a label, not
a filter), and ambiguity must surface as an explicit invalid verdict, not a guess.
These use the pure span logic; the image-dependent shot gate has its own tests.
"""

from __future__ import annotations

from cv.pipeline.active_play import (
    ActivePlayResult,
    apply_leakage_trims,
    _event_spans,
    _intersect_spans_with_play_shots,
)


class _FakeSegmentation:
    def __init__(self, shots):
        self.shots = shots


def _play(start, end):
    return {"shot_id": 0, "start_frame": start, "end_frame": end, "is_play_camera": True}


def _cutaway(start, end):
    return {"shot_id": 1, "start_frame": start, "end_frame": end, "is_play_camera": False}


def test_spans_are_clipped_to_play_shots():
    seg = _FakeSegmentation([_play(1, 100), _cutaway(101, 160), _play(161, 300)])
    spans = _intersect_spans_with_play_shots([(50.0, 200.0)], seg)
    # The cutaway hole splits the rally span; nothing inside 101-160 survives.
    assert spans == [(50.0, 100.0), (161.0, 200.0)]


def test_span_entirely_inside_cutaway_is_dropped():
    seg = _FakeSegmentation([_play(1, 100), _cutaway(101, 200)])
    assert _intersect_spans_with_play_shots([(120.0, 180.0)], seg) == []


def test_no_play_shots_yields_no_spans():
    seg = _FakeSegmentation([_cutaway(1, 300)])
    assert _intersect_spans_with_play_shots([(10.0, 200.0)], seg) == []


def test_event_spans_expand_but_do_not_cross_cutaways():
    seg = _FakeSegmentation([_play(10, 100), _cutaway(101, 160), _play(161, 300)])
    spans = _event_spans([(20.0, 99.0), (165.0, 200.0)], seg, fps=25.0)
    # 0.25 s margin at 25 fps = 6.25 frames, clamped at play-shot boundaries.
    assert spans == [(13.75, 100.0), (161.0, 206.25)]


def test_contains_uses_the_trim_window():
    result = ActivePlayResult(clip="pt0001", fps=50.0, n_frames=500)
    result.trim = (100.0, 400.0)
    assert result.contains(100.0) and result.contains(400.0) and result.contains(250.0)
    assert not result.contains(99.0) and not result.contains(401.0)


def test_no_trim_contains_nothing():
    result = ActivePlayResult(clip="pt0001", fps=50.0, n_frames=500)
    assert not result.contains(10.0)


def test_as_dict_round_trips_the_essentials():
    result = ActivePlayResult(clip="pt0002", fps=25.0, n_frames=250)
    result.active_spans = [(10.0, 90.0)]
    result.event_spans = [(7.0, 93.0)]
    result.trim = (5.0, 95.0)
    result.trimmed_fraction = 0.55
    payload = result.as_dict()
    assert payload["clip"] == "pt0002"
    assert payload["trim"] == [5.0, 95.0]
    assert payload["active_spans"] == [[10.0, 90.0]]
    assert payload["event_spans"] == [[7.0, 93.0]]
    assert result.contains_event(92.0)
    assert payload["point_valid"] is True


def test_default_entry_never_declares_native_cut_continuity():
    result = ActivePlayResult(clip="pt0004", fps=25.0, n_frames=100)
    # Even a receipt left on the result stays out of an unselected entry.
    result.boundary_continuity = [{"schema": "native_shot_boundary_continuity_v1"}]
    payload = result.as_dict()
    assert "native_cut_continuity" not in payload
    assert "boundary_continuity" not in payload


def test_selected_entry_publishes_the_boundary_receipts():
    receipts = [
        {
            "schema": "native_shot_boundary_continuity_v1",
            "before_frame": 40,
            "after_frame": 44,
            "continuous": True,
        }
    ]
    result = ActivePlayResult(
        clip="pt0004",
        fps=25.0,
        n_frames=100,
        native_cut_continuity=True,
        boundary_continuity=receipts,
    )
    payload = result.as_dict()
    assert payload["native_cut_continuity"] is True
    assert payload["boundary_continuity"] == receipts


def test_leakage_proposals_split_active_and_event_spans():
    entry = {
        "clip": "pt0001",
        "fps": 25.0,
        "n_frames": 200,
        "active_spans": [[10.0, 150.0]],
        "event_spans": [[5.0, 155.0]],
        "point_valid": True,
        "reasons": [],
    }
    proposal = {
        "schema": "play_camera_leakage_v1",
        "proposed_trims": [{"start_frame": 50, "end_frame": 69, "reasons": ["closeup"]}],
    }

    result = apply_leakage_trims(entry, proposal)

    assert result["active_spans"] == [[10.0, 49.0], [70.0, 150.0]]
    assert result["event_spans"] == [[5.0, 49.0], [70.0, 155.0]]
    assert result["leakage_trim"]["decision"] == "applied"


def test_leakage_trim_below_two_seconds_holds_without_shortening_scope():
    entry = {
        "clip": "pt0001",
        "fps": 25.0,
        "n_frames": 100,
        "active_spans": [[1.0, 60.0]],
        "event_spans": [[1.0, 60.0]],
        "point_valid": True,
        "reasons": [],
    }
    proposal = {"proposed_trims": [{"start_frame": 1, "end_frame": 20, "reasons": ["closeup"]}]}

    result = apply_leakage_trims(entry, proposal)

    assert result["point_valid"] is False
    assert result["active_spans"] == [[1.0, 60.0]]
    assert result["event_spans"] == [[1.0, 60.0]]
    assert result["leakage_trim"]["active_spans_after"] == [[21.0, 60.0]]
    assert result["leakage_trim"]["decision"] == "hold"
    assert result["reasons"] == ["leakage_trim_below_2s"]
