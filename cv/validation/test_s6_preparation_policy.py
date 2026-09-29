"""Qualification and scoping contracts for the optional cold preparation policy."""

import copy

import pytest

from cv.pipeline import s6_preparation_policy as policy


def example(status="labeled"):
    packet = {"attempts": [{"point_clip": "pt1", "owner_end_frame": 10.0}]}
    labels = {
        "events": {
            "records": [
                {
                    "clip": "pt1",
                    "event_type": "net_hit",
                    "frame": 10,
                    "frame_interval": [9, 11],
                    "status": status,
                },
                {
                    "clip": "pt2",
                    "event_type": "net_hit",
                    "frame": 20,
                    "frame_interval": [19, 21],
                    "status": "labeled",
                },
            ]
        }
    }
    return packet, labels


@pytest.mark.parametrize("setting,status", [("off", "labeled"), ("on", "ambiguous")])
def test_no_invented_net_or_control_changes(monkeypatch, setting, status):
    packet, labels = example(status)
    original = copy.deepcopy((packet, labels))

    def forbidden(*args):
        raise AssertionError("unqualified context extension")

    monkeypatch.setattr(policy.preparation, "known_terminal_context", forbidden)
    prepared, receipt = policy.prepare_packet(packet, labels, setting)
    assert prepared is packet
    assert receipt["resolved_net_events"] == []
    assert (packet, labels) == original


def test_terminal_refusal_retains_original_observations(monkeypatch):
    packet, labels = example()

    def refuse(*args):
        raise ValueError("no observed post-ending ground impact; do not invent one")

    monkeypatch.setattr(policy.preparation, "known_terminal_context", refuse)
    prepared, receipt = policy.prepare_packet(packet, labels, "on")
    assert prepared is packet
    assert receipt["terminal_context"]["status"] == "qualification_refused"
    assert [event["frame"] for event in receipt["resolved_net_events"]] == [10]


def test_net_witness_scope_and_exception_restoration(monkeypatch):
    calls = []

    def original(event, cameras, pixels, radii=None, **kwargs):
        calls.append(kwargs["eligible_frames"].tolist())
        return "witness"

    monkeypatch.setattr(policy.whole, "event_ground_target", original)
    with pytest.raises(RuntimeError):
        with policy.bounded_net_context([{"frame_interval": [9, 11]}]) as receipt:
            assert (
                policy.whole.event_ground_target(
                    {"frame": 15}, {}, {}, eligible_frames=[8, 9, 10, 11, 12, 15]
                )
                == "witness"
            )
            raise RuntimeError("exercise restoration")
    assert calls == [[12.0, 15.0]]
    assert receipt["applications"][0]["excluded_across_net_or_uncertain_frames"] == [8, 9, 10, 11]
    assert policy.whole.event_ground_target is original


def test_no_net_is_exact_function_noop():
    original = policy.whole.event_ground_target
    with policy.bounded_net_context([]) as receipt:
        assert policy.whole.event_ground_target is original
        assert receipt["applications"] == []


def test_net_scope_preserves_same_flight_context_but_excludes_after_raw_contact(monkeypatch):
    packet, labels = example()
    labels["events"]["records"].extend(
        [
            {
                "clip": "pt1",
                "event_type": "net_hit",
                "frame": 12,
                "frame_interval": [11.5, 12.5],
                "status": "labeled",
            },
            {
                "clip": "pt1",
                "event_type": "contact",
                "frame": 15,
                "frame_interval": [14, 16],
                "status": "ambiguous",
            },
            {
                "clip": "pt1",
                "event_type": "net_hit",
                "frame": 20,
                "frame_interval": [19, 21],
                "status": "labeled",
            },
        ]
    )
    monkeypatch.setattr(
        policy.preparation, "known_terminal_context", lambda packet, labels: (packet, {})
    )
    original = copy.deepcopy((packet, labels))
    _, receipt = policy.prepare_packet(packet, labels, "on")
    assert [e["frame"] for e in receipt["resolved_net_events"]] == [10, 12]
    assert [e["frame"] for e in receipt["net_continuity_scope"]["excluded_events"]] == [20]
    assert (packet, labels) == original
