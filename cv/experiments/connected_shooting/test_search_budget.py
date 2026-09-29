"""Deadline mode retains only complete candidates and never ranks acceptance."""

import json

import numpy as np
import pytest

from cv.experiments.connected_shooting import search_budget
from cv.experiments.connected_shooting.labeled_serve_recipe import select_candidate


def candidate(score=2.0, stage="coarse", accepted=False):
    return {
        "stage": stage,
        "depth_hypothesis_m": 0.0,
        "evidence": {"input_only_rank_score": score, "survived": accepted},
        "measurement": {
            "fit": {"parameters": np.zeros(11)},
            "native_projection": [{"frame": 1, "predicted": [2, 3]}],
        },
    }


def test_deadline_keeps_immutable_complete_checkpoint(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(search_budget.time, "monotonic", lambda: clock[0])
    budget = search_budget.Budget(1.0, tmp_path / "checkpoint.json")
    row = candidate()
    budget.record(row, 1)
    row["measurement"]["fit"]["parameters"][0] = 99
    clock[0] = 1.1
    with pytest.raises(search_budget.SearchDeadline):
        budget.check()
    budget.persist()
    saved = json.loads(budget.checkpoint.read_text())
    assert saved["completed_candidates"][0]["measurement"]["fit"]["parameters"][0] == 0
    assert saved["search_budget"]["exhausted"] is True


def test_absent_deadline_preserves_original_no_checkpoint_behavior(tmp_path, monkeypatch):
    budget = search_budget.Budget(None, tmp_path / "checkpoint.json")
    monkeypatch.setattr(search_budget.time, "monotonic", lambda: float("inf"))
    budget.check()
    budget.record(candidate(), 1)
    assert not budget.checkpoint.exists()
    assert budget.completed == []


@pytest.mark.parametrize("defect", ["incomplete", "nonfinite", "no_projection", "no_score"])
def test_checkpoint_refuses_incomplete_state(tmp_path, defect):
    row = candidate()
    if defect == "incomplete":
        row["measurement"]["fit"]["parameters"] = [0.0]
    elif defect == "nonfinite":
        row["measurement"]["fit"]["parameters"][2] = np.nan
    elif defect == "no_projection":
        row["measurement"]["native_projection"] = []
    else:
        row["evidence"]["input_only_rank_score"] = np.nan
    budget = search_budget.Budget(1, tmp_path / "checkpoint.json")
    with pytest.raises(ValueError):
        budget.record(row, 1)
    assert not budget.checkpoint.exists()


def test_completed_selection_ignores_gate_and_refinement_status():
    better = candidate(score=1, accepted=False)
    worse = candidate(score=4, stage="refined", accepted=True)
    report = {"completed_candidates": [worse, better], "search_budget": {"exhausted": True}}
    selected, receipt = select_candidate(report, 1, "input-ranked-completed", None)
    assert selected is better
    assert receipt["gate_used"] is False
    report["search_budget"]["exhausted"] = False
    with pytest.raises(ValueError, match="exhausted"):
        select_candidate(report, 1, "input-ranked-completed", None)


def test_numerical_failure_fallback_replays_input_ranked_completed_candidate():
    better = candidate(score=1, accepted=False)
    worse = candidate(score=4, accepted=True)
    report = {
        "completed_candidates": [worse, better],
        "refined_candidates": [],
        "numerical_candidate_failures": [{"reason": "refine failed"}],
        "search_budget": {"exhausted": False, "retained_completed_after_numerical_failures": True},
    }
    selected, receipt = select_candidate(report, 1, "input-ranked-completed", None)
    assert selected is better
    assert receipt["gate_used"] is False
