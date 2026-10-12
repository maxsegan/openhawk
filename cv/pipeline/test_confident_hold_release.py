"""High-confidence gate holds, and an orphan contact before the no-contact refusal."""

from __future__ import annotations

import pytest

from cv.experiments.connected_shooting import auto_packet
from cv.pipeline import confident_hold_release as release
from cv.pipeline import s6_labeled_stage as stage

MATCH = "m"
CLIP = "pt0001"
CADENCE = {
    "decision": "timing_hold",
    "reasons": ["systematic_repeated_visual_frames"],
    "duplicate_frames": [286, 289],
    "equivalence_groups": [[285, 286], [288, 289]],
}


def _row(frame, reasons, *, marginal=0.999, model_abstain=False):
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": "contact",
        "frame": frame,
        "abstain": True,
        "gate_held": True,
        "model_abstain": model_abstain,
        "acceptance_marginal": marginal,
        "point_gate_verdict": "hold",
        "point_gate_failure_reasons": list(reasons),
    }


def test_a_confident_tracking_hold_is_released_and_a_lower_one_is_not():
    rows, census = release.release_confident_holds(
        [_row(10, ["tracking_arc_abstained"]), _row(20, ["tracking_arc_abstained"], marginal=0.98)],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 100]],
        cadence=CADENCE,
        release=True,
    )
    assert census["released"] == 1
    assert rows[0]["abstain"] is False and rows[0]["gate_held"] is False
    assert rows[0]["point_gate_failure_reasons"] == []
    assert rows[1]["abstain"] is True and "held_release" not in rows[1]


def test_the_model_abstention_is_not_cleared():
    rows, census = release.release_confident_holds(
        [_row(10, ["tracking_arc_abstained"], model_abstain=True)],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 100]],
        cadence=CADENCE,
        release=True,
    )
    assert census["released"] == 0
    assert rows[0]["model_abstain"] is True and rows[0]["abstain"] is True


def test_a_camera_hold_needs_the_local_frame_and_a_repeated_frame_needs_the_cadence():
    inside, outside = _row(50, ["hard_camera_failure"]), _row(150, ["hard_camera_failure"])
    unique, duplicate = (
        _row(315, ["systematic_repeated_visual_frames"]),
        _row(286, ["systematic_repeated_visual_frames"]),
    )
    stuck = {
        **CADENCE,
        "equivalence_groups": [[280, 281, 282, 283, 284, 285]],
        "duplicate_frames": [281, 282, 283, 284, 285],
    }
    rows, census = release.release_confident_holds(
        [inside, outside, unique, duplicate],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 100]],
        cadence=CADENCE,
        release=True,
    )
    assert census["released"] == 2
    assert rows[0]["abstain"] is False
    assert rows[1]["abstain"] is True and rows[2]["abstain"] is False
    assert rows[3]["abstain"] is True
    held, census = release.release_confident_holds(
        [_row(315, ["systematic_repeated_visual_frames"])],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 400]],
        cadence=stuck,
        release=True,
    )
    assert census["released"] == 0 and held[0]["gate_held"] is True


def test_another_reason_or_the_switch_off_keeps_the_row():
    blocked = _row(10, ["tracking_arc_abstained", "outside_retained_play_interval"])
    rows, census = release.release_confident_holds(
        [blocked],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 100]],
        cadence=CADENCE,
        release=True,
    )
    assert census["released"] == 0 and rows[0] is blocked
    untouched, census = release.release_confident_holds(
        [blocked],
        match_id=MATCH,
        clip=CLIP,
        camera_spans=[[1, 100]],
        cadence=CADENCE,
        release=False,
    )
    assert untouched == [blocked] and census["released"] == 0


def test_the_switch_is_absent_until_a_policy_names_it():
    assert release.POLICY_KEY not in stage.shared_settings({})
    named = stage.shared_settings({release.POLICY_KEY: "on"})
    assert named[release.POLICY_KEY] == "on"
    with pytest.raises(ValueError, match="unsupported global held_release"):
        stage.shared_settings({release.POLICY_KEY: "wide"})


def _decoder(kind, frame, *, abstain=True, marginal=0.435):
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": kind,
        "frame": float(frame),
        "abstain": abstain,
        "model_abstain": abstain,
        "gate_held": False,
        "acceptance_marginal": marginal,
        "decision_threshold": 0.9907,
        "abstention_gap": 0.5,
        "location": {"frame_subpixel": float(frame)},
        "point_grammar": {
            "schema": "event_grammar_decoder_no_point_grammar_v1",
            "present": False,
        },
    }


def _end(frame):
    return {
        "match_id": MATCH,
        "clip": CLIP,
        "event_type": "point_end",
        "frame": float(frame),
        "location": {"frame_subpixel": float(frame)},
        "point_grammar": {"verdict": "accepted"},
        "point_end": {"termination_kind": "second_ground", "terminal_event_type": "bounce"},
    }


def test_an_orphan_contact_is_retained_only_when_the_switch_asks():
    document = [_decoder("contact", 755), _end(800)]
    with pytest.raises(ValueError, match="no contact"):
        auto_packet.automatic_event_inventory(
            document, MATCH, CLIP, (1, 900), observation_scope=False
        )
    events, ending = auto_packet.automatic_event_inventory(
        document,
        MATCH,
        CLIP,
        (1, 900),
        observation_scope=False,
        retain_orphan_contacts=True,
    )
    assert len(events) == 1
    assert events[0]["frame"] == 755.0
    assert events[0]["status"] == "ambiguous"
    assert events[0]["orphan_before_preparation"] is True
    assert ending["kind"] == "second_ground"


def test_a_repeated_picture_snaps_to_its_first_exposure_only_when_switched(monkeypatch):
    row = _row(286, ["systematic_repeated_visual_frames"])
    row["location"] = {"frame_subpixel": 286.3}
    kwargs = dict(match_id=MATCH, clip=CLIP, camera_spans=[[1, 400]], cadence=CADENCE)
    monkeypatch.delenv(release.DUPLICATE_SNAP_SWITCH, raising=False)
    rows, census = release.release_confident_holds([row], release=True, **kwargs)
    assert census["released"] == 0 and rows[0]["abstain"] is True

    monkeypatch.setenv(release.DUPLICATE_SNAP_SWITCH, "on")
    rows, census = release.release_confident_holds([row], release=True, **kwargs)
    assert census["released"] == 1 and census["duplicate_snapped"] == 1
    assert rows[0]["frame"] == 285.0 and rows[0]["location"]["frame_subpixel"] == 285.3
    assert rows[0]["duplicate_snap"] == {"from": 286, "to": 285}
    assert rows[0]["abstain"] is False
