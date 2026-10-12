"""Physical-branch retry contracts: input-only choice, deadlines and safe fallback."""

from copy import deepcopy
import numpy as np
import pytest
from cv.experiments.connected_shooting import labeled_prefix_joint_impact as prefix


def fixture(monkeypatch, boundaries):
    monkeypatch.setattr(prefix.regime_recovery, "boundary_evidence", lambda *a: boundaries)
    monkeypatch.setattr(prefix.time, "monotonic", lambda: 10.0)
    state = dict(
        best_cost=10.0,
        best_q=np.zeros(16),
        best_receipt={"bundle": {"scene": None}, "chain": []},
        calls=1,
        invalid=0,
        feasible_trials=1,
    )
    return state


def call(state, **kwargs):
    return prefix.retry_regime_boundaries(
        None,
        state,
        {"kind": "baseline"},
        np.full(16, -2.0),
        np.full(16, 2.0),
        movable=2,
        maxiter=80,
        seconds=100.0,
        started=0.0,
        solver_kwargs={"inequality_key": "input_domain_inequalities"},
        **kwargs,
    )


def test_unaffected_or_frozen_flight_does_not_start_numerical_work(monkeypatch):
    state = fixture(monkeypatch, [{"flight_index": 3}])
    monkeypatch.setattr(
        prefix.boundary_fit, "solve", lambda *a, **kw: pytest.fail("unexpected retry")
    )
    got, ended, audit = call(state)
    assert got == state and ended == {"kind": "baseline"} and audit["attempts"] == []


def test_both_original_law_signs_get_time_and_input_cost_selects(monkeypatch):
    state = fixture(monkeypatch, [{"flight_index": 0}])
    calls = []

    def solve(evaluate, seed, lo, hi, **kwargs):
        calls.append((seed.copy(), kwargs))
        out = deepcopy(state)
        out.update(
            best_cost=5 if seed[6] < 0 else 7,
            best_q=seed,
            best_receipt={"gate_accepted": False},
            calls=2,
        )
        return out, {"kind": "retried"}

    monkeypatch.setattr(prefix.boundary_fit, "solve", solve)
    got, _, audit = call(state)
    assert [row[0][6] for row in calls] == [-0.02, 0.02]
    assert [row[1]["seconds"] for row in calls] == [55.0, 100.0]
    assert all(
        row[1]["started"] == 0 and row[1]["inequality_key"] == "input_domain_inequalities"
        for row in calls
    )
    assert got["best_cost"] == 5 and got["best_receipt"]["gate_accepted"] is False
    assert audit["selected"] == "direction_-1"


def test_failed_or_worse_retry_preserves_original_feasible_output(monkeypatch):
    state = fixture(monkeypatch, [{"flight_index": 1}])
    before = deepcopy(state)

    def solve(evaluate, seed, *args, **kwargs):
        if seed[13] < 0:
            raise ValueError("no feasible trial before deadline")
        out = deepcopy(before)
        out.update(best_cost=12, best_q=seed)
        return out, {"kind": "optimizer"}

    monkeypatch.setattr(prefix.boundary_fit, "solve", solve)
    got, ended, audit = call(state)
    np.testing.assert_array_equal(got["best_q"], before["best_q"])
    assert got["best_cost"] == before["best_cost"] and ended == {"kind": "baseline"}
    assert audit["selected"] == "baseline" and audit["attempts"][0]["status"] == "execution_failed"


def test_exhausted_shared_budget_keeps_baseline(monkeypatch):
    state = fixture(monkeypatch, [{"flight_index": 0}])
    monkeypatch.setattr(prefix.time, "monotonic", lambda: 100.0)
    got, _, audit = call(state)
    assert got["best_cost"] == 10 and audit["attempts"] == []
    assert audit["stopped"] == "shared wall budget exhausted"
