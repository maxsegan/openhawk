from __future__ import annotations

import numpy as np
import pytest

from cv.pipeline import fit_only_bounce_rules as rules


def _detector(records, probs):
    """``probs``: {frame: p(bounce)} on clip ``pt1``."""
    frames = sorted(probs)
    p = np.zeros((len(frames), 4))
    for i, f in enumerate(frames):
        p[i, 2] = probs[f]
        p[i, 0] = 1 - probs[f]
    return rules.DetectorRecord(
        records, {"clips": ["m__pt1"] * len(frames), "frames": frames, "probabilities": p}
    )


def _result(end=100.0, dense_bounces=(98.0,), supplied=()):
    vf = {
        "start_frame": 60.0,
        "end_frame": end,
        "accepted": True,
        "bounce_witness": [{"supplied_frame": s} for s in supplied],
    }
    dense = {"start_frame": 60.0, "bounces": [{"frame": b} for b in dense_bounces]}
    return {"verdict": {"flights": [vf]}, "measurement": {"dense_flights": [dense]}}


def _decide(result, det, events=None):
    return rules.evaluate_result(result, events, det, "pt1", [(60.0, 100.0)])[(60.0, 100.0)]


def test_switch_defaults_off_and_rejects_typos():
    assert not rules.enabled({})
    assert rules.enabled({rules.SWITCH: "on"})
    with pytest.raises(ValueError):
        rules.enabled({rules.SWITCH: "yes"})


def test_unwitnessed_inside_fit_only_bounce_demotes():
    d = _decide(_result(), _detector([], {97: 0.05, 98: 0.1, 99: 0.0}))
    assert d["demote"] and d["reasons"] == [rules.UNWITNESSED]
    assert d["bounces"][0]["fit_only"] and d["bounces"][0]["position"] == "inside"


def test_witness_keeps_the_flight():
    d = _decide(_result(), _detector([], {97: 0.05, 99: 0.25}))
    assert not d["demote"] and d["bounces"][0]["witnessed"]


def test_other_type_record_breaks_the_witness_even_when_abstained():
    held = {"clip": "m__pt1", "event_type": "contact", "frame": 98.6, "abstain": True}
    d = _decide(_result(), _detector([held], {98: 0.9}))
    assert d["reasons"] == [rules.UNWITNESSED]


def test_emitted_net_hit_is_rule_2():
    net = {
        "clip": "m__pt1",
        "event_type": "net_hit",
        "frame": 97.5,
        "location": {"frame_subpixel": 98.2},
    }
    d = _decide(_result(), _detector([net], {98: 0.9}))
    assert d["reasons"] == [rules.NET_HIT] and d["reason"] == rules.NET_HIT


def test_supplied_or_packet_bounce_is_not_fit_only():
    det = _detector([], {98: 0.0})
    assert not _decide(_result(supplied=(98.3,)), det)["demote"]
    events = [{"event_type": "bounce", "frame": 98.5}]
    assert not _decide(_result(), det, events)["demote"]


def test_after_bounce_never_demotes():
    d = _decide(_result(dense_bounces=(98.0, 120.0), supplied=(98.0,)), _detector([], {}))
    assert not d["demote"]
    assert [b["position"] for b in d["bounces"]] == ["inside", "after"]
    assert d["bounces"][1]["fit_only"] and d["bounces"][1]["reason"] is None


def test_rules_json_carries_the_version():
    doc = rules.to_rules_json()
    assert doc["rules_version"] == rules.RULES_VERSION and doc["constants"]["witness_p"] == 0.2
