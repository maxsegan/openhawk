"""The three default-off releases for confident events an upstream gate vetoed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cv.experiments.connected_shooting import auto_packet
from cv.pipeline import event_impulse_support as impulse
from cv.pipeline import live_shot_camera as live
from cv.pipeline import s6_contact_prefix_scope as prefix
from cv.pipeline import s6_labeled_stage as stage

MATCH = "m"
CLIP = "pt0001"


def test_a_flicker_is_not_a_wide_shot_and_a_half_second_gap_is_bridged() -> None:
    assert live.live_spans([[1, 40]], 25.0) is None
    bridged = live.live_spans([[1, 10], [20, 80]], 25.0)
    assert bridged == [(1, 80)]
    split = live.live_spans([[1, 10], [40, 120]], 25.0)
    assert split == [(1, 10), (40, 120)]


def test_only_a_shot_composition_hold_releases_and_only_inside_the_span() -> None:
    spans = [(100, 200)]
    gate = {
        "decision": "hold",
        "reasons": ["hard_camera_failure"],
        "active_play_reasons": ["phase:multiple_camera_shots"],
        "hard_timing_failure": False,
        "court_geometry_valid": True,
    }
    assert live.shot_composition_hold(gate) is True
    assert live.shot_composition_hold({**gate, "active_play_reasons": ["court_geometry_invalid"]}) is False
    assert live.shot_composition_hold({**gate, "decision": "retain"}) is False
    rows = [
        _camera("contact", 150, model_abstain=False),
        _camera("contact", 40, model_abstain=False),
        _camera("contact", 160, model_abstain=False, tracking=True),
    ]
    released, census = live.release_camera_holds(
        rows, match_id=MATCH, clip=CLIP, spans=spans, release=True, fps=25.0
    )
    assert census["released"] == 1 and census["tracking_kept"] == 1
    assert released[0]["abstain"] is False and released[0]["gate_held"] is False
    assert released[1]["abstain"] is True and "live_shot_camera" not in released[1]
    assert released[2]["gate_held"] is True and released[2]["abstain"] is True
    assert released[2]["point_gate_verdict"] == "retain"
    assert released[2]["point_gate_failure_reasons"] == ["tracking_arc_abstained"]
    untouched, census = live.release_camera_holds(
        rows, match_id=MATCH, clip=CLIP, spans=spans, release=False, fps=25.0
    )
    assert untouched == rows and census["released"] == 0


def test_a_kept_clip_reholds_a_closeup_and_a_weak_camera_reholds_nothing() -> None:
    spans = [(100, 200)]
    accepted = {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": "contact",
        "frame": 72.0,
        "abstain": False,
        "gate_held": False,
        "model_abstain": False,
        "point_gate_verdict": "retain",
        "point_gate_failure_reasons": [],
    }
    inside = {**accepted, "frame": 150.0}
    updated, census = live.rehold_closeups(
        [accepted, inside], match_id=MATCH, clip=CLIP, spans=spans, fps=50.0
    )
    assert census["reheld"] == 1
    assert updated[0]["abstain"] is True
    assert live.OUTSIDE_LIVE_SHOT in updated[0]["point_gate_failure_reasons"]
    assert updated[1] == inside
    kept, census = live.rehold_closeups(
        [accepted], match_id=MATCH, clip=CLIP, spans=None, fps=50.0
    )
    # A contact a fifth of a second before the homography locks is still the wide shot.
    near = {**accepted, "frame": 110.0}
    kept_near, census = live.rehold_closeups(
        [near], match_id=MATCH, clip=CLIP, spans=[(121, 400)], fps=50.0
    )
    assert kept_near == [near] and census["reheld"] == 0
    assert kept == [accepted] and census["reheld"] == 0


def test_a_confident_first_contact_with_an_outgoing_wing_and_a_near_box_is_a_serve() -> None:
    frame = 100.0
    event = _serve(frame)
    arcs = [
        {"arc_id": 2, "regime": "ballistic", "decision": "retain", "start_frame": 101, "end_frame": 130}
    ]
    track = {t: np.array([100.0 + 4.0 * (t - 100), 400.0 - (t - 100)], dtype=float) for t in range(101, 131)}
    near = [("near", (90.0, 390.0, 140.0, 520.0))]
    evidence = impulse.serve_outgoing_support(event, arcs, track, near, fps=50.0)
    assert evidence["supported"] is True
    assert evidence["box_distance_native_px"] == 0.0
    far = [("far", (700.0, 80.0, 760.0, 200.0))]
    refused = impulse.serve_outgoing_support(event, arcs, track, far, fps=50.0)
    assert refused["supported"] is False and refused["reason"] == "server_box_not_near_ball"
    no_wing = impulse.serve_outgoing_support(event, [], track, near, fps=50.0)
    assert no_wing["reason"] == "no_outgoing_flight_arc"


def test_only_the_earliest_model_accepted_contact_can_be_released() -> None:
    first = _serve(100.0)
    first["abstain"] = False
    first["gate_held"] = False
    first["point_gate_failure_reasons"] = []
    later = _serve(140.0)
    arcs = [
        {"arc_id": 4, "regime": "ballistic", "decision": "retain", "start_frame": 141, "end_frame": 170}
    ]
    track = {t: np.array([10.0 + t, 20.0], dtype=float) for t in range(141, 171)}
    boxes = {140: [("near", (0.0, 0.0, 30.0, 40.0))], 139: []}
    updated, census = impulse.release_serve_contacts(
        [first, later],
        match_id=MATCH,
        clip=CLIP,
        arcs=arcs,
        track=track,
        boxes_by_frame=boxes,
        fps=50.0,
    )
    assert census["released"] == 0
    assert updated[1] is later


def test_stripping_the_shot_veto_lets_the_serve_release_read_the_row() -> None:
    """A no-play clip's first contact is held for the shot and the tracking arc."""

    raw = _serve(100.0)
    raw["point_gate_verdict"] = "hold"
    raw["point_gate_failure_reasons"] = ["hard_camera_failure", "tracking_arc_abstained"]
    released, census = live.release_camera_holds(
        [raw], match_id=MATCH, clip=CLIP, spans=[(50, 200)], release=True, fps=50.0
    )
    assert census["tracking_kept"] == 1
    assert released[0]["point_gate_verdict"] == "retain"
    assert released[0]["abstain"] is True
    arcs = [
        {"arc_id": 2, "regime": "ballistic", "decision": "retain", "start_frame": 101, "end_frame": 130}
    ]
    track = {
        t: np.array([100.0 + 4.0 * (t - 100), 400.0 - (t - 100)], dtype=float) for t in range(101, 131)
    }
    boxes = {100: [("near", (90.0, 390.0, 140.0, 520.0))]}
    updated, serve = impulse.release_serve_contacts(
        released,
        match_id=MATCH,
        clip=CLIP,
        arcs=arcs,
        track=track,
        boxes_by_frame=boxes,
        fps=50.0,
    )
    assert serve["released"] == 1 and updated[0]["abstain"] is False


def test_a_prefix_before_the_first_contact_refuses_unless_isolation_is_on() -> None:
    document = [
        _plain("bounce", 20),
        _plain("contact", 40),
        _plain("bounce", 80),
        _end(120),
    ]
    with pytest.raises(ValueError, match="preceding physical row"):
        auto_packet.automatic_event_inventory(
            document,
            MATCH,
            CLIP,
            (1, 400),
            observation_scope=True,
            require_originating_contact=True,
            leading_physical_prefix=True,
        )
    events, ending = auto_packet.automatic_event_inventory(
        document,
        MATCH,
        CLIP,
        (1, 400),
        observation_scope=True,
        require_originating_contact=True,
        leading_physical_prefix=True,
        isolate_invalid_prefix=True,
    )
    assert [row["frame"] for row in events] == [40.0, 80.0]
    assert ending["isolated_invalid_prefix"] == [{"event_type": "bounce", "frame": 20.0}]
    assert prefix.isolate_invalid_prefix("off") is False
    assert prefix.isolate_invalid_prefix("on") is True


def test_a_no_play_clip_is_scoped_to_its_wide_shot_and_nothing_else_is() -> None:
    """The event model never ran on these clips because event_spans was empty."""

    refused = {
        "fps": 25.0,
        "n_frames": 500,
        "reasons": ["no_play_camera_shot"],
        "event_spans": [],
    }
    assert live.scope_event_spans(refused, [[100, 200]]) == [[100.0, 200.0]]
    # A one-second flicker is not a wide shot, so the producer gate stays in force.
    assert live.scope_event_spans(refused, [[1, 20]]) == []
    already = {**refused, "event_spans": [[10.0, 20.0]]}
    assert live.scope_event_spans(already, [[100, 200]]) == [[10.0, 20.0]]
    other = {**refused, "reasons": ["phase:multiple_camera_shots"]}
    assert live.scope_event_spans(other, [[100, 200]]) == []


def test_the_event_command_names_the_live_shot_only_when_asked() -> None:
    from cv.pipeline.broadcast_runner import event_inference_command

    default = event_inference_command("py", Path("out"), Path("ckpt.pt"))
    assert "--live-shot-camera" not in default
    selected = event_inference_command(
        "py", Path("out"), Path("ckpt.pt"), live_shot_camera=True
    )
    assert selected.count("--live-shot-camera") == 1


def test_the_three_switches_stay_absent_until_a_policy_names_them() -> None:
    assert "live_shot_camera_eligibility" not in stage.shared_settings({})
    assert "serve_outgoing_release" not in stage.shared_settings({})
    assert "preparation_prefix_isolation" not in stage.shared_settings({})
    named = stage.shared_settings(
        {
            "live_shot_camera_eligibility": "on",
            "serve_outgoing_release": "on",
            "preparation_prefix_isolation": "on",
        }
    )
    assert named["live_shot_camera_eligibility"] == "on"
    assert named["serve_outgoing_release"] == "on"
    assert named["preparation_prefix_isolation"] == "on"
    with pytest.raises(ValueError, match="unsupported global live_shot_camera_eligibility"):
        stage.shared_settings({"live_shot_camera_eligibility": "wide"})


def _camera(kind: str, frame: float, *, model_abstain: bool, tracking: bool = False) -> dict:
    reasons = ["hard_camera_failure"]
    if tracking:
        reasons.append("tracking_arc_abstained")
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": kind,
        "frame": frame,
        "abstain": True,
        "gate_held": True,
        "model_abstain": model_abstain,
        "point_gate_verdict": "hold",
        "point_gate_failure_reasons": reasons,
    }


def _serve(frame: float) -> dict:
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": "contact",
        "frame": frame,
        "abstain": True,
        "gate_held": True,
        "model_abstain": False,
        "point_gate_verdict": "retain",
        "point_gate_failure_reasons": ["tracking_arc_abstained"],
        "acceptance_marginal": 0.995,
        "decision_threshold": 0.9907,
        "location": {
            "image_x": 100.0,
            "image_y": 400.0,
            "image_coordinate_space": "native_1920x1080",
            "fps": 50.0,
        },
    }


def _plain(kind: str, frame: float) -> dict:
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": kind,
        "frame": frame,
        "location": {"frame_subpixel": float(frame)},
        "point_grammar": {"in_play": True, "verdict": "accepted"},
    }


def _end(frame: float) -> dict:
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": "point_end",
        "frame": frame,
        "location": {"frame_subpixel": float(frame)},
        "point_grammar": {"in_play": True, "verdict": "accepted"},
        "point_end": {"termination_kind": "second_ground", "terminal_event_type": "bounce"},
    }
