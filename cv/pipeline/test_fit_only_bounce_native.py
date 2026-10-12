"""The native fit-only bounce rules in ``product_runner`` (assemble records, compact documents)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from cv.pipeline import fit_only_bounce_rules as rules
from cv.pipeline import product_runner as runner


def _detector(p_bounce: float):
    frames = list(range(50, 140))
    p = np.zeros((len(frames), 4))
    p[:, 2] = p_bounce
    p[:, 0] = 1 - p_bounce
    return rules.DetectorRecord(
        [], {"clips": ["pt1"] * len(frames), "frames": frames, "probabilities": p}
    )


def _result():
    """One accepted flight 60..100 whose dense state runs on to 130 with a fit-only bounce
    inside (98) and one after the verdict end (120)."""
    vf = {
        "start_frame": 60.0,
        "end_frame": 100.0,
        "accepted": True,
        "flight_index": 0,
        "bounce_witness": [],
    }
    dense = {
        "start_frame": 60.0,
        "end_frame": 130.0,
        "start_xyz": [0, 0, 1],
        "end_xyz": [0, 20, 0],
        "positions": [[0, i / 10, 1] for i in range(71)],
        "bounces": [{"frame": 98.0, "x": [0, 10, 0]}, {"frame": 120.0, "x": [0, 18, 0]}],
        "net_hits": [],
    }
    return {"verdict": {"flights": [vf]}, "measurement": {"dense_flights": [dense]}}


def _flights(result):
    flights = runner.attempt_flights(result, step=1)
    for f in flights:
        f["confirmed"] = True
    return flights


def test_unwitnessed_flight_is_demoted_and_the_continuation_cut():
    result = _result()
    flights = _flights(result)
    assert len(flights[0]["positions_m"]) == 71
    tally = runner.apply_fit_only_bounce_rules(result, flights, _detector(0.05), "pt1", None)
    f = flights[0]
    assert not f["confirmed"]
    assert f["validation"]["reasons"] == [rules.UNWITNESSED]
    assert f["fit_only_bounce_rules"]["demote"]
    assert f["continuation"]["after_bounces"] == 1 and f["continuation"]["from_frame"] == 100.0
    assert len(f["positions_m"]) == 11  # dense frames 60..100 of 60..130, every 4th kept
    assert f["bounces_xyz_m"] == [[0, 10, 0]]
    assert tally == {
        "continuation_cut": 1,
        "flights_with_fit_only_bounce": 1,
        "demoted:fit_only_bounce_unwitnessed": 1,
    }


def test_witnessed_flight_stays_validated():
    result = _result()
    flights = _flights(result)
    runner.apply_fit_only_bounce_rules(result, flights, _detector(0.5), "pt1", None)
    assert flights[0]["confirmed"] and not flights[0]["fit_only_bounce_rules"]["demote"]


def test_first_fit_packet_bounce_makes_it_supplied(tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps({"attempts": [{"events": [{"event_type": "bounce", "frame": 98.4}]}]})
    )
    result = _result()
    flights = _flights(result)
    runner.apply_fit_only_bounce_rules(result, flights, _detector(0.0), "pt1", packet)
    assert flights[0]["confirmed"]


def test_trimmed_flight_is_matched_by_its_verdict_span():
    result = _result()
    flights = _flights(result)
    flights[0]["trim"] = {"end_frame_before": 100.0, "end_frame": 90.0}
    flights[0]["end_frame"] = 90.0
    runner.apply_fit_only_bounce_rules(result, flights, _detector(0.0), "pt1", None)
    assert not flights[0]["confirmed"]


def test_compact_labels_the_documents_only_when_on(tmp_path, monkeypatch):
    data = tmp_path / "portal" / "3d" / "data"
    data.mkdir(parents=True)
    doc = {
        "frames": {
            "frame": [60, 100, 110],
            "x": [0.0, 0.0, 0.0],
            "segment": ["flight_00"] * 3,
            "acceptance": ["accepted"] * 3,
        },
        "per_flight": {
            "flights": [
                {"flight_index": 0, "start_frame": 60.0, "end_frame": 100.0, "accepted": True}
            ],
            "accepted_flight_indices": [0],
            "complete_point": True,
        },
        "bounces": [],
    }
    (data / "automatic_s6_m__pt1.json").write_text(json.dumps(doc))
    flight = {
        "start_frame": 60.0,
        "fit_only_bounce_rules": {"demote": True, "reasons": [rules.UNWITNESSED]},
    }
    (tmp_path / "MATCH_RESULT.json").write_text(
        json.dumps({"attempts": [{"key": "m__pt1", "accepted_flights": [flight]}]})
    )
    args = argparse.Namespace(out=tmp_path)
    monkeypatch.delenv(rules.SWITCH, raising=False)
    assert runner.step_fit_only_documents(args) is None
    monkeypatch.setenv(rules.SWITCH, "on")
    summary = runner.step_fit_only_documents(args)
    assert summary["status"] == "done" and summary["changed"] == 1
    out = json.loads((data / "automatic_s6_m__pt1.json").read_text())
    assert out["frames"]["acceptance"] == ["best_effort", "best_effort", rules.CONTINUATION]
    assert out["per_flight"]["accepted_flight_indices"] == []
    assert not out["per_flight"]["complete_point"]
