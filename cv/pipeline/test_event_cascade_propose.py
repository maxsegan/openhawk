import pytest

from cv.pipeline import event_cascade_propose as proposer
from cv.pipeline import event_cascade_rebind as rebind
from cv.pipeline.event_cascade import Answer

WINDOW = {"id": "a__gap100", "mode": "locate", "start": 88, "end": 112, "event_frame": None}
RETIME = {"id": "a__retime100", "mode": "retime", "start": 88, "end": 112, "event_frame": 100.0}


def _event(kind, frame, status="predicted"):
    return {
        "event_type": kind,
        "frame": frame,
        "frame_interval": [frame - 1, frame + 1],
        "status": status,
        "occurrence_status": status,
        "annotation_origin": "automatic",
        "exact_epoch_observed": False,
        "timing_status": "predicted",
        "automatic_location": {"frame_subpixel": frame, "image_x": 1.0, "image_y": 2.0},
    }


def test_flash_screens_settle_none_only_when_both_say_no_contact():
    none, hit = Answer("none", None, 0.01), Answer("contact", 101.0, 0.07)
    replies = {("gemini", "ball"): none, ("gemini", "player"): none}
    walk = proposer.settle(WINDOW, lambda t, p: replies.get((t, p)))
    assert walk["action"] == "none" and len(walk["receipts"]) == 2
    replies[("gemini", "player")] = Answer(None, None, 0.01)
    assert proposer.settle(WINDOW, lambda t, p: replies.get((t, p)))["owed"] == (
        "opus",
        "ball_player",
    )
    replies[("opus", "ball_player")] = hit
    walk = proposer.settle(WINDOW, lambda t, p: replies.get((t, p)))
    assert (walk["action"], walk["frame"], walk["tier"]) == ("insert", 101.0, "opus")
    assert [r["presentation"] for r in walk["receipts"]] == ["ball", "player", "ball_player"]
    replies[("opus", "ball_player")] = Answer("contact", 140.0, 0.07)
    assert proposer.settle(WINDOW, lambda t, p: replies.get((t, p)))["action"] == "unsettled"


def test_retime_moves_only_by_at_least_two_frames():
    ask = lambda frame: proposer.settle(  # noqa: E731
        RETIME, lambda t, p: Answer("contact", frame, 0.03) if (t, p) == ("opus", "ball") else None
    )
    assert ask(101.0)["action"] == "keep"
    assert (ask(104.0)["action"], ask(104.0)["frame"]) == ("retime", 104.0)


def test_locate_windows_skip_trailing_and_accepted_spans():
    events = [_event("contact", 10.0), _event("bounce", 30.0), _event("contact", 200.0)]
    rows = proposer.locate_centres(
        events, fps=25.0, window=(0, 400), accepted=[], triggers=proposer.LOCATE_TRIGGERS
    )
    assert rows and {r["trigger"] for r in rows} == {"gap"}
    assert all(20 <= r["centre"] <= 190 for r in rows)
    covered = proposer.locate_centres(
        events, fps=25.0, window=(0, 400), accepted=[(0, 250)], triggers=proposer.LOCATE_TRIGGERS
    )
    assert covered == []


def _decision(action, frame, event_frame=None, trigger=None):
    return {
        "id": f"w{frame}",
        "window": f"w{frame}",
        "action": action,
        "frame": frame,
        "event_frame": event_frame,
        "trigger": trigger or ("retime" if action == "retime" else "gap"),
        "tier": "opus",
        "model": "anthropic/claude-opus-5.5",
        "receipts": [],
    }


def test_propose_edits_respect_cascade_decisions_and_clearance(tmp_path):
    track = tmp_path / "t.csv"
    track.write_text("clip,frame,x,y,sources,x_native,y_native\npt1,f_0060.jpg,1,1,det,50,60\n")
    inventory = [_event("contact", 10.0), _event("bounce", 40.0), _event("contact", 80.0)]
    settled = {("contact", 80.0): {"action": "reject"}}
    decisions = [
        _decision("retime", 13.0, 10.0),
        _decision("retime", 84.0, 80.0),
        _decision("insert", 41.0),
        _decision("insert", 60.0),
        _decision("insert", 79.0),
        _decision("insert", 2.0, trigger="leading"),
    ]
    edits, dropped = proposer.propose_edits(
        decisions, inventory, settled, switch="retime_locate", track_csv=track, point="pt1"
    )
    labels = sorted(proposer.edit_label(e) for e in edits)
    assert labels == [
        "insert:contact:60.0",
        "insert:contact:79.0",
        "retime:contact:10.0->13.0",
    ]
    assert {d["why"] for d in dropped} == {
        "cascade_settled_row",
        "near_existing_row",
        "trigger_not_searched",
    }
    only, _ = proposer.propose_edits(
        decisions, inventory, settled, switch="retime", track_csv=track, point="pt1"
    )
    assert [e["action"] for e in only] == ["retime"]
    inserted = next(e for e in edits if e["action"] == "insert" and e["event"]["frame"] == 60.0)
    assert inserted["event"]["automatic_location"]["image_x"] == 50.0
    assert inserted["event"]["occurrence_status"] == "predicted"

    rows = [dict(e, clip="pt1") for e in inventory] + [{"event_type": "ending", "frame": 99.0}]
    out = rebind.apply_events(rows, settled, edits)
    physical = [(e["event_type"], e["frame"]) for e in out if e["event_type"] != "ending"]
    assert physical == [("contact", 13.0), ("bounce", 40.0), ("contact", 60.0), ("contact", 79.0)]
    assert out[-1]["event_type"] == "ending"
    assert all(e.get("clip") == "pt1" for e in out if e["event_type"] != "ending")
    retimed = out[0]
    assert retimed["cascade_retimed_from"] == 10.0 and retimed["frame_interval"] == [12.0, 14.0]
    with pytest.raises(ValueError):
        rebind.apply_events(rows, {("contact", 10.0): {"action": "reject"}}, edits)
