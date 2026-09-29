from copy import deepcopy

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_preparation_recovery_probe as probe


def test_context_preserves_labels_events_and_original_native_rows():
    rows = [{"frame": f, "status": "visible", "x1080": f, "y1080": 4} for f in range(11, 17)]
    events = [
        {"event_type": "contact", "frame": 10.5, "status": "labeled"},
        {"event_type": "net_hit", "frame": 13.5, "status": "labeled"},
        {"event_type": "bounce", "frame": 16.5, "status": "labeled"},
    ]
    packet = {
        "attempts": [
            {
                "point_clip": "pt1",
                "owner_end_frame": 13.5,
                "events": events,
                "owner_ball_labels": rows[:3],
                "context_native_frames": [14, 15, 16],
            }
        ]
    }
    labels = {
        "attempt": {"ending_kind": "net"},
        "events": {"records": events},
        "ball": {"records": [{"clip": "pt1", "frames": rows}]},
    }
    original = deepcopy((packet, labels))
    derived, receipt = probe.known_terminal_context(packet, labels)
    assert (packet, labels) == original
    assert derived["attempts"][0]["events"] == events
    assert derived["attempts"][0]["owner_ball_labels"] == rows
    assert receipt["competitive_ending_frame"] == 13.5
    assert receipt["additional_frames"] == [14, 15, 16]
    labels["events"]["records"] = events[:2]
    with pytest.raises(ValueError, match="do not invent"):
        probe.known_terminal_context(packet, labels)


def test_partial_anchor_passes_only_observed_ground_controls(monkeypatch):
    keys = list(probe.prepare.court.GROUND)
    record = {
        "case_id": "court",
        "frames": [
            dict(
                target_id=k,
                frame=5,
                status="visible" if i else "ambiguous",
                x1080=float(i),
                y1080=float(i + 1),
            )
            for i, k in enumerate(keys)
        ],
    }
    received = {}

    def fit(xyz, pixels):
        received.update(xyz=xyz, pixels=pixels)
        return {"P": np.zeros((3, 4)).tolist()}

    monkeypatch.setattr(probe.prepare.ground, "fit_ground", fit)
    answer = probe.partial_ground_anchor(record, 0.05)
    assert len(received["xyz"]) == 7
    assert received["pixels"][:, 0].tolist() == list(range(1, 8))
    assert answer["omitted_ground_controls"] == keys[:1]
    for row in record["frames"][:3]:
        row["status"] = "ambiguous"
    with pytest.raises(ValueError, match="six observed"):
        probe.partial_ground_anchor(record, 0.05)


def test_contact_uncertainty_overlapping_observed_ground_blocks_passive_extension():
    rows = [{"frame": f, "status": "visible", "x1080": f, "y1080": 4} for f in range(11, 18)]
    events = [
        {"event_type": "contact", "frame": 10.5, "status": "labeled"},
        {"event_type": "net_hit", "frame": 13.5, "status": "labeled"},
        {"event_type": "bounce", "frame": 16.5, "frame_interval": [16, 17], "status": "labeled"},
    ]
    packet = {
        "attempts": [
            {
                "point_clip": "p",
                "owner_end_frame": 13.5,
                "events": events,
                "owner_ball_labels": rows[:3],
                "context_native_frames": [14, 15, 16, 17],
            }
        ]
    }
    labels = {
        "attempt": {"ending_kind": "net"},
        "events": {
            "records": [
                *events,
                {
                    "event_type": "contact",
                    "frame": 18,
                    "frame_interval": [16.75, 19],
                    "status": "ambiguous",
                },
            ]
        },
        "ball": {"records": [{"clip": "p", "frames": rows}]},
    }
    original = deepcopy((packet, labels))
    with pytest.raises(ValueError, match="contact interval"):
        probe.known_terminal_context(packet, labels)
    assert (packet, labels) == original
