from copy import deepcopy

import pytest

from cv.pipeline import s6_optional_contacts as optional


@pytest.mark.parametrize("activated", [202, 207])
def test_producer_serialization_replays_original_score_with_activated_context(tmp_path, activated):
    """Exercise producer counts through JSON and the ordinary loader's selector."""
    import json
    from cv.experiments.connected_shooting.labeled_serve_recipe import select_candidate

    rows = [fit(4), fit(2)]
    for row in rows:
        row["measurement"]["fit"] = {"parameters": [1.0] * 11}
    hypothesis = {"added": [None], "occurrence_log_odds": 0.0}
    chosen, score = optional.best_completed({"coarse": rows, "refined": []}, hypothesis, 202)
    report = {
        "coarse_candidates": rows,
        "refined_candidates": [],
        "optional_contact_selection": {
            **optional.observation_count_receipt(202, activated),
            "added_contact_count": 1,
            "occurrence_log_odds": 0.0,
            "score": score,
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    restored = json.loads(path.read_text())
    replayed, receipt = select_candidate(restored, 1, "input-ranked-refined", None)
    assert replayed == chosen
    assert receipt["source_rank"] == score
    assert restored["optional_contact_selection"]["activated_training_observations"] == activated
    if activated != 202:
        restored["optional_contact_selection"]["training_observations"] = activated
        with pytest.raises(ValueError, match="does not reproduce its physical score"):
            select_candidate(restored, 1, "input-ranked-refined", None)


@pytest.mark.parametrize("original,activated", [(0, 5), (202, 201), (True, 207), (202, 207.0)])
def test_score_count_receipt_rejects_missing_or_invented_inventory(original, activated):
    with pytest.raises(ValueError, match="preserved activated inventory"):
        optional.observation_count_receipt(original, activated)


def events():
    return [
        {"event_type": kind, "frame": f, "frame_interval": [f - 0.2, f + 0.2]}
        for kind, f in [
            ("contact", 1),
            ("bounce", 10),
            ("bounce", 30),
            ("contact", 40),
            ("bounce", 50),
        ]
    ]


def candidate(frame=20, phase=None):
    return {
        "clip": "match__pt0001",
        "event_type": "contact",
        "frame": frame,
        "location": {"frame_subpixel": frame},
        "abstain": True,
        "class_probabilities": {"contact": 0.99, "none": 0.005},
        "phase": phase,
    }


def test_only_internal_multi_bounce_gap_existing_candidate_is_eligible():
    original = events()
    before = deepcopy(original)
    rows = optional.source_candidates(
        original,
        [candidate(), candidate(45), candidate(20, "collection")],
        "match__pt0001",
        {i: 100 + i * 0.04 for i in range(1, 60)},
    )
    assert len(rows) == 1 and rows[0]["event"]["frame"] == 20
    assert rows[0]["source_pts_interval"] == pytest.approx([100.76, 100.84])
    assert rows[0]["occurrence_log_odds"] == 1
    assert original == before
    assert not optional.source_candidates(
        original, [candidate(9)], "match__pt0001", {i: 100 + i * 0.04 for i in range(1, 60)}
    )


def test_source_hypothesis_beam_does_not_read_gate_results_and_preserves_events():
    original = events()
    rows = optional.source_candidates(
        original,
        [candidate(20), candidate(22), candidate(24)],
        "match__pt0001",
        {i: 100 + i * 0.04 for i in range(1, 60)},
    )
    for row in rows:
        row.update(supported=True, witness_support=0.8)
    doc = {
        "schema": optional.SCHEMA,
        "source_attempt_events": original,
        "gaps": optional.inconsistent_gaps(original),
        "candidates": rows,
    }
    hs = optional.hypotheses(doc, original)
    assert len(hs) == 2
    assert all(len(h["added"]) == 1 and all(e in h["events"] for e in original) for h in hs)
    rows[2]["evaluation_passes"] = True
    assert optional.hypotheses(doc, original) == hs
    with pytest.raises(ValueError, match="unchanged"):
        optional.hypotheses(doc, original[:-1])


def fit(rms, geometry=0, survived=False):
    return {
        "measurement": {"rms_px": {"training": rms}},
        "evidence": {"input_geometry_penalty": geometry, "survived": survived},
        "depth_hypothesis_m": 5,
    }


def test_physical_selection_ignores_gates_and_penalizes_added_parameters():
    bad_pass = fit(10, survived=True)
    good_hold = fit(1, survived=False)
    selected, score = optional.best_completed(
        {"coarse": [good_hold], "refined": [bad_pass]}, None, 100
    )
    assert selected is good_hold
    assert optional.physical_rank(good_hold, 1, 1, 100) > score
    assert optional.physical_rank(fit(1, 9), 0, 0, 100) - score == pytest.approx(1)


def test_bounce_membership_has_no_contact_parameter_penalty_but_retains_occurrence_prior():
    row = fit(2)
    plain = optional.physical_rank(row, 0, 0, 100)
    hypothesis = {
        "added": [{"event_type": "bounce"}],
        "added_parameter_count": 0,
        "occurrence_log_odds": 0.4,
    }
    candidate, score = optional.best_completed({"coarse": [row], "refined": []}, hypothesis, 100)
    assert candidate is row
    assert score == pytest.approx(plain - 0.4 / 100)
    assert optional.physical_rank(row, 1, 0.4, 100) > score
    with pytest.raises(ValueError, match="parameter count"):
        optional.physical_rank(row, 1, 0.4, 100, added_parameters=-1)


def test_actual_consumer_calls_same_fitter_fresh_and_shares_whole_budget():
    from types import SimpleNamespace
    import time

    original = events()
    proposals = [
        {
            "name": f"candidate_{i}",
            "events": [*deepcopy(original), {"event_type": "contact", "frame": 20 + i}],
        }
        for i in range(2)
    ]
    budget = SimpleNamespace(seconds=30, started=time.monotonic(), exhausted=False)
    calls = []

    def fitter(event_rows, horizon, name):
        calls.append((deepcopy(event_rows), horizon, name, budget.seconds))
        if name == "candidate_0":
            raise ValueError("physical initialization failure")
        return {"coarse": [fit(1)], "refined": []}

    branches, receipts = optional.fit_alternatives(fitter, proposals, 50, budget, 90)
    assert [row[3] for row in calls] == [60, 90]
    assert [row[1] for row in calls] == [50, 50]
    assert all(all(e in row[0] for e in original) for row in calls)
    assert len(branches) == 1 and receipts[0]["status"] == "no_fit"
    assert budget.seconds == 90


def test_conditioned_event_reaches_actual_physical_consumer_without_promoting_source():
    from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

    original = events()
    rows = optional.source_candidates(
        original, [candidate(20)], "match__pt0001", {i: 100 + i * 0.04 for i in range(1, 60)}
    )
    rows[0].update(supported=True, witness_support=0.8)
    doc = {
        "schema": optional.SCHEMA,
        "source_attempt_events": original,
        "gaps": optional.inconsistent_gaps(original),
        "candidates": rows,
    }
    before = deepcopy(doc)
    alternative = optional.hypotheses(doc, original)[0]
    added = alternative["added"][0]
    assert occurrence.predicted_membership(added)
    assert occurrence.resolved_membership(added)
    assert added["optional_topology_membership"]["conditioned_on_this_hypothesis"]
    assert added in alternative["events"]
    assert doc == before and rows[0]["event"]["occurrence_status"] == "optional"
    assert added not in original


def signature_inputs():
    rows = [
        {
            "frame": f,
            "status": "visible",
            "x1080": 500.0 + f,
            "y1080": 500.0,
            "native_pts_seconds": 100.0 + f / 25,
            "source_image_sha256": str(f),
        }
        for f in range(1, 17)
    ]
    attempt = {
        "match_id": "match",
        "point_clip": "pt0001",
        "fps": 25,
        "first_event_frame": 1,
        "owner_end_frame": 12,
        "owner_ball_labels": rows,
        "context_native_frames": [13, 14, 15, 16],
        "events": [
            {"event_type": "contact", "frame": 1},
            {"event_type": "bounce", "frame": 8},
        ],
    }
    cameras = {
        "match_id": "match",
        "clip": "pt0001",
        "cameras": [
            {
                "frame": f,
                "status": "supported",
                "P": [[100, 0, 0, 500], [0, 100, 0, 500], [0, 0, 1, 10]],
            }
            for f in range(1, 17)
        ],
    }
    return attempt, cameras


def test_signature_matches_actual_scene_after_preowned_context_extension():
    from cv.experiments.connected_shooting import event_recovery
    from cv.experiments.connected_shooting.agent_whole_point_search import prepare_attempt

    attempt, cameras = signature_inputs()
    original = deepcopy(attempt)
    extended = event_recovery.extend_attempt_window(
        attempt, [{"clip": "pt0001", "frames": attempt["owner_ball_labels"]}], 16
    )
    assert len(extended["owner_ball_labels"]) == 20
    signature = optional.observation_signature(extended, cameras, "fifth_frame_withheld")
    scene, *_ = prepare_attempt(extended, cameras, "hard")
    assert [r[0] for r in signature] == sorted(f for wing in scene.observation_frames for f in wing)
    assert len(signature) == 13
    assert attempt == original
    assert len(extended["owner_ball_labels"]) == 20


@pytest.mark.parametrize(
    "field,value", [("x1080", 999), ("native_pts_seconds", 999), ("source_image_sha256", "other")]
)
def test_signature_rejects_conflicting_duplicate_native_observations(field, value):
    attempt, cameras = signature_inputs()
    duplicate = deepcopy(attempt["owner_ball_labels"][0])
    duplicate[field] = value
    attempt["owner_ball_labels"].append(duplicate)
    with pytest.raises(ValueError, match="conflicting observation identity"):
        optional.observation_signature(attempt, cameras, "fifth_frame_withheld")


def test_signature_rejects_conflicting_camera_and_fractional_native_epoch():
    attempt, cameras = signature_inputs()
    duplicate = deepcopy(cameras["cameras"][0])
    duplicate["P"][0][0] += 1
    cameras["cameras"].append(duplicate)
    with pytest.raises(ValueError, match="conflicting camera identity"):
        optional.observation_signature(attempt, cameras, "fifth_frame_withheld")
    cameras["cameras"].pop()
    attempt["owner_ball_labels"][0]["frame"] += 0.1
    with pytest.raises(ValueError, match="integer native observation"):
        optional.observation_signature(attempt, cameras, "fifth_frame_withheld")


def test_original_input_domain_filters_before_physical_rank_and_never_reads_gates():
    class Forbidden(dict):
        def __getitem__(self, key):
            raise AssertionError("evaluation gate read")

    rows = [fit(rms) for rms in [1, 2, 3]]
    for row, epoch in zip(rows, [12.001, 11, 10.5]):
        row.update(epoch=epoch, verdict=Forbidden())
    branch = {"coarse": rows[:2], "refined": rows[2:]}

    def admission(row):
        return {"admissible": 10 <= row["epoch"] <= 12}

    chosen, _ = optional.best_completed(branch, None, 100, input_admission=admission)
    assert chosen is rows[1]
    assert optional.best_completed(branch, None, 100)[0] is rows[0]
    with pytest.raises(ValueError, match="no finite"):
        optional.best_completed(
            branch, None, 100, input_admission=lambda row: {"admissible": False}
        )


def test_optional_shared_replay_respects_declared_domain_selection_and_historical_contract():
    import numpy as np
    from cv.experiments.connected_shooting.labeled_serve_recipe import select_candidate

    rows = [fit(1), fit(2), fit(3)]
    for row, epoch in zip(rows, [12.001, 11, 10.5]):
        row["measurement"]["fit"] = {"parameters": np.ones(11).tolist()}
        row["epoch"] = epoch
    hypothesis = {"added": [None], "occurrence_log_odds": 1.0}
    search = {
        "coarse_candidates": rows,
        "refined_candidates": [],
        "optional_contact_selection": {
            "added_contact_count": 1,
            "occurrence_log_odds": 1.0,
            "training_observations": 100,
            "input_domain_selection": "original_physical_domain_before_rank_v1",
            "score": optional.physical_rank(rows[1], 1, 1.0, 100),
        },
    }

    def admission(row):
        return {"admissible": 10 <= row["epoch"] <= 12}

    chosen, receipt = select_candidate(
        search, 1, "input-ranked-refined", None, input_admission=admission
    )
    assert chosen is rows[1] and receipt["gate_used"] is False
    with pytest.raises(ValueError, match="no finite"):
        select_candidate(
            search,
            1,
            "input-ranked-refined",
            None,
            input_admission=lambda row: {"admissible": False},
        )
    del search["optional_contact_selection"]["input_domain_selection"]
    search["optional_contact_selection"]["score"] = optional.best_completed(
        {"coarse": rows, "refined": []}, hypothesis, 100
    )[1]
    with pytest.raises(ValueError, match="violates original"):
        select_candidate(search, 1, "input-ranked-refined", None, input_admission=admission)
