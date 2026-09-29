"""Source-only service reset proposals: identity, event ownership and false splits."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import service_attempt_scope as scope


def event(kind, frame):
    return dict(
        event_type=kind,
        frame=frame,
        abstain=False,
        location={"frame_subpixel": frame},
        frame_interval=[frame - 1, frame + 1],
    )


def fixture(monkeypatch):
    # Existing stance helper is isolated here; actual full-source census covers
    # its real motion input. No ball/event rule is mocked.
    times = np.arange(1, 401) / 25
    monkeypatch.setattr(
        scope.segment_boundaries,
        "load_player_motion",
        lambda *a, **k: {"near": (times, np.zeros((400, 2)), np.zeros(400))},
    )
    monkeypatch.setattr(
        scope.segment_boundaries,
        "second_serve_marks",
        lambda *a, **k: [dict(seconds=10.2, impact_seconds=10.76, side="near", hold_seconds=8.0)],
    )
    obs = [dict(frame=f, status="visible", x1080=500 + 0.1 * f, y1080=400.0) for f in range(1, 111)]
    obs += [
        dict(frame=f, status="visible", x1080=1000 + 0.1 * f, y1080=600.0) for f in range(240, 401)
    ]
    return dict(
        emissions=[
            event("contact", 20),
            event("net_hit", 30),
            event("bounce", 40),
            event("contact", 270),
            event("bounce", 300),
        ],
        observations=obs,
        players=Path("declared.csv"),
        source_start=0.0,
        fps=25.0,
        native_window=(1, 400),
        cameras={"cameras": [dict(frame=f, supported=True) for f in range(1, 401)]},
    )


def test_reset_components_preserve_all_events_and_context(monkeypatch):
    args = fixture(monkeypatch)
    original = deepcopy(args["emissions"])
    result = scope.plan(**args)
    assert len(result["children"]) == 2
    a, b = result["children"]
    assert a["original_event_indices"] == [0, 1, 2] and b["original_event_indices"] == [3, 4]
    assert a["modeled_native_window"] == [1, 110]
    assert b["modeled_native_window"] == [240, 400]
    assert a["video_context_native_window"] == b["video_context_native_window"] == [1, 400]
    assert a["aftermath"]["kind"] == "identity_censored"
    assert not a["aftermath"]["physical_ending_certified"]
    assert args["emissions"] == original == result["original_emissions"]


@pytest.mark.parametrize("change", ["live_contact", "short_gap", "no_identity_break"])
def test_stance_does_not_split_ongoing_play(monkeypatch, change):
    args = fixture(monkeypatch)
    if change == "live_contact":
        args["emissions"].insert(3, event("contact", 100))
    elif change == "short_gap":
        args["emissions"].insert(3, event("bounce", 240))
    else:
        args["observations"] = [
            dict(frame=f, status="visible", x1080=500 + 0.1 * f, y1080=400.0) for f in range(1, 401)
        ]
    result = scope.plan(**args)
    assert not result["children"] and result["rejected"]


def test_casual_return_remains_original_physical_membership(monkeypatch):
    args = fixture(monkeypatch)
    args["emissions"][1:3] = [event("bounce", 30), event("contact", 45), event("bounce", 66)]
    result = scope.plan(**args)
    assert result["children"][0]["original_event_indices"] == [0, 1, 2, 3]
    assert result["original_emissions"][2] == event("contact", 45)


def test_uncertain_interval_cannot_be_clipped_at_identity_censor(monkeypatch):
    args = fixture(monkeypatch)
    args["emissions"][2]["frame_interval"] = [39, 75]
    args["observations"] = [
        r for r in args["observations"] if r["frame"] <= 70 or r["frame"] >= 240
    ]
    result = scope.plan(**args)
    assert not result["children"]
    assert (
        result["rejected"][0]["reason"]
        == "component_does_not_cover_original_event_interval_and_tail"
    )


def test_declared_player_clock_receipt_and_inconsistent_source_refusal(monkeypatch):
    args = fixture(monkeypatch)
    args["clip"] = "source_clip"
    for row in args["observations"]:
        row["native_pts_seconds"] = (row["frame"] - 1) / 25
    out = scope.plan(**args)
    assert out["player_motion_clock"]["source_player_rows_modified"] is False
    assert out["player_motion_clock"]["clip"] == "source_clip"
    args["observations"][0]["native_pts_seconds"] += 1 / 25
    with pytest.raises(ValueError, match="source window clock"):
        scope.plan(**args)
