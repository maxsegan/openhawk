"""Optional athlete evidence: supplied evidence is exact, absence is explicit, malformed refuses."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    agent_whole_point_search as whole,
    athlete_priors,
    initialization,
    per_flight_rescore as rescore,
    real_bidirectional_search,
)

LABELS = {
    "events": {
        "records": [
            {"event_type": "contact", "frame": 10.0, "hitter": "Ann"},
            {"event_type": "contact", "frame": 40.0, "hitter": "Bo"},
        ]
    }
}
UNLABELED = {"events": {"records": [{"event_type": "contact", "frame": f} for f in (10.0, 40.0)]}}
CONTACTS = [{"frame": 10.0}, {"frame": 40.0}]


def state(name=None, stature=None):
    return {"player": name, "stature_m": stature, "court_centre_xy_m": [0.0, 0.0]}


def test_known_evidence_resolves_identically_under_both_policies():
    strict = whole.resolve_athlete_evidence(
        "stature_pose_soft", "required", ["Ann", "Bo"], ["Ann=1.8", "Bo=1.9"], LABELS, CONTACTS
    )
    optional = whole.resolve_athlete_evidence(
        "stature_pose_soft", "optional", ["Ann", "Bo"], ["Ann=1.8", "Bo=1.9"], LABELS, CONTACTS
    )
    assert strict[0] == optional[0] == ["Ann", "Bo"]
    assert strict[1] == optional[1] == {"Ann": 1.8, "Bo": 1.9}
    assert strict[2]["status"] == optional[2]["status"] == "supplied"
    assert whole.mark_athlete_evidence([state("Ann", 1.8)], optional[2]) == [state("Ann", 1.8)]


def test_original_policy_refuses_absent_evidence_and_both_refuse_malformed():
    with pytest.raises(ValueError, match="player-order cycle"):
        whole.resolve_athlete_evidence("stature_pose_soft", "required", [], [], LABELS, CONTACTS)
    for policy in ("required", "optional"):
        with pytest.raises(ValueError, match="exact statures"):
            whole.resolve_athlete_evidence(
                "stature_pose_soft", policy, ["Ann", "Bo"], ["Ann=1.8"], LABELS, CONTACTS
            )
        with pytest.raises(ValueError, match="PLAYER=METRES"):
            whole.resolve_athlete_evidence(
                "stature_pose_soft", policy, ["Ann"], ["Ann"], UNLABELED, CONTACTS
            )
    with pytest.raises(ValueError, match="without a player order"):
        whole.resolve_athlete_evidence(
            "stature_pose_soft", "optional", [], ["Ann=1.8"], LABELS, CONTACTS
        )
    with pytest.raises(ValueError, match="supported athlete-evidence policy"):
        whole.resolve_athlete_evidence("stature_pose_soft", "lenient", [], [], LABELS, CONTACTS)


def test_absent_evidence_is_declared_consistently_from_search_to_replay_and_scoring():
    names, statures, receipt = whole.resolve_athlete_evidence(
        "stature_pose_soft", "optional", [], [], UNLABELED, CONTACTS
    )
    assert names is None and statures == {} and receipt["status"] == "unavailable"
    players = whole.mark_athlete_evidence([state(), state()], receipt)
    assert all(p["athlete_evidence"]["status"] == "unavailable" for p in players)
    assert all(p["player"] is None and p["stature_m"] is None for p in players)
    with pytest.raises(ValueError, match="cannot carry a name or stature"):
        whole.mark_athlete_evidence([state("Ann", 1.8)], receipt)
    # Scoring on these states abstains explicitly instead of reading a fabricated height.
    report = athlete_priors.evaluate(np.array([[0.0, 0.0, 2.9], [1.0, 1.0, 1.0]]), players)
    assert report["athlete_evidence"]["status"] == "unavailable"
    assert report["optimization_residuals"] == [0.0, 0.0, 0.0]
    assert {row["stature_m"] for row in report["contacts"]} == {None}
    # The replay must name the same policy the search recorded and reproduce its status.
    configuration = {
        "athlete_evidence": "optional",
        "athlete_evidence_status": receipt["status"],
        "player_statures_m": statures,
    }
    replay = rescore.replay_athlete_evidence(
        argparse.Namespace(athlete_evidence="optional"), {"configuration": configuration}
    )
    assert replay["status"] == "unavailable"
    with pytest.raises(ValueError, match="must match the cached search"):
        rescore.replay_athlete_evidence(argparse.Namespace(), {"configuration": configuration})
    with pytest.raises(ValueError, match="disagrees"):
        rescore.replay_athlete_evidence(
            argparse.Namespace(athlete_evidence="optional"),
            {"configuration": configuration | {"player_statures_m": {"Ann": 1.8}}},
        )


@pytest.mark.parametrize(
    "policy, order, statures, expected",
    [
        ("required", ["Ann", "Bo"], ["Ann=1.8", "Bo=1.9"], "supplied"),
        ("optional", ["Ann", "Bo"], ["Ann=1.8", "Bo=1.9"], "supplied"),
        ("optional", [], [], "unavailable"),
    ],
)
def test_every_report_producer_writes_the_shape_the_replay_reads(policy, order, statures, expected):
    names, parsed, receipt = whole.resolve_athlete_evidence(
        "stature_pose_soft", policy, order, statures, LABELS, CONTACTS
    )
    # Both the per-topology and the final report configurations spread this one dict.
    configuration = {"player_statures_m": parsed} | whole.athlete_evidence_configuration(
        policy, receipt
    )
    assert configuration["athlete_evidence"] == policy
    assert configuration["athlete_evidence_status"] == expected
    assert configuration["athlete_evidence_receipt"] is receipt
    replay = rescore.replay_athlete_evidence(
        argparse.Namespace(athlete_evidence=policy), {"configuration": configuration}
    )
    assert replay["status"] == expected
    marked = whole.mark_athlete_evidence([state(), state()], replay)
    assert all(("athlete_evidence" in p) == (expected == "unavailable") for p in marked)
    with pytest.raises(ValueError, match="must carry the declared policy"):
        whole.athlete_evidence_configuration("required", receipt | {"policy": "optional"})


def test_replay_of_existing_strict_searches_is_unchanged():
    legacy = {"configuration": {"player_statures_m": {"Ann": 1.8, "Bo": 1.9}}}
    assert rescore.replay_athlete_evidence(argparse.Namespace(), legacy) == {
        "policy": "required",
        "status": "supplied",
    }
    with pytest.raises(ValueError, match="must match"):
        rescore.replay_athlete_evidence(argparse.Namespace(athlete_evidence="optional"), legacy)
    with pytest.raises(ValueError, match="cannot replay an unavailable"):
        rescore.replay_athlete_evidence(
            argparse.Namespace(),
            {"configuration": {"athlete_evidence_status": "unavailable"}},
        )


def test_serve_anchor_without_stature_stays_on_the_observed_ray(monkeypatch):
    monkeypatch.setattr(
        real_bidirectional_search, "ray_point_at_y", lambda *_: np.array([1.0, 5.0, 4.2])
    )
    camera, pixel = np.eye(3, 4), np.array([0.0, 0.0])
    banded, banded_receipt = initialization.serve_contact_anchor(camera, pixel, 1.8, 5.0)
    assert banded_receipt["clipped_to_band"] and banded[2] == pytest.approx(3.15)
    xyz, receipt = initialization.serve_contact_anchor(camera, pixel, None, 5.0)
    assert xyz.tolist() == [1.0, 5.0, 4.2]
    assert receipt == {
        "method": "server_contact_ray_at_depth_branch_unbanded",
        "stature_m": None,
        "height_band_m": None,
        "ray_height_at_depth_m": 4.2,
        "clipped_to_band": False,
        "contact_xyz_m": [1.0, 5.0, 4.2],
    }
