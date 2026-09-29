"""Deadline incumbent retention: finite same-invocation iterate, explicit, default off."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    camera_geometry,
    model,
    real_exposure_replay,
    search_budget,
)
from cv.experiments.connected_shooting.search_budget import SearchDeadline


class CountingDeadline:
    """Cooperative deadline that fires after a fixed number of consultations."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = 0
        self.exhausted = False

    def __call__(self):
        self.calls += 1
        if self.calls > self.limit:
            self.exhausted = True
            raise SearchDeadline("test deadline")


def control_scene():
    """Same two-flight simulator control the shared-contact refine test uses."""
    scene, base = model.control()
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    truth = np.r_[base, np.zeros(6), [1.0, 1.0]]
    flights = model.chain(scene, truth)
    scene = replace(
        scene,
        pixels=tuple(
            camera_geometry.project(cameras, flight["positions"])
            for cameras, flight in zip(scene.cameras, flights, strict=True)
        ),
    )
    return scene, truth


def control_refine(deadline, retain, initial_offset=0.05, maxiter=40):
    scene, truth = control_scene()
    initial = truth.copy()
    initial[3:9] += initial_offset * np.array([1.0, -1.0, 0.4, -0.8, 0.6, -0.4])
    return real_exposure_replay.refine(
        scene,
        initial,
        (np.empty(0), np.empty(0)),
        scene.observation_frames,
        np.zeros((sum(map(len, scene.observation_frames)), 2)),
        None,
        maxiter,
        inequality_constraints=False,
        deadline_check=deadline,
        retain_deadline_incumbent=retain,
    )


def test_incumbent_keeps_only_finite_in_bounds_strict_improvements():
    incumbent = search_budget.Incumbent(10.0, np.zeros(2), np.ones(2))
    incumbent.offer([0.5, 0.5], 10.0)  # not a strict improvement
    incumbent.offer([0.5, np.nan], 1.0)  # nonfinite vector
    incumbent.offer([0.5, 0.5], np.inf)  # nonfinite cost
    incumbent.offer([1.5, 0.5], 1.0)  # outside bounds
    incumbent.offer([0.5], 1.0)  # wrong shape
    assert not incumbent.improved
    assert incumbent.evaluations == 5 and incumbent.improving_evaluations == 0
    vector = np.array([0.25, 0.75])
    incumbent.offer(vector, 4.0)
    incumbent.offer([0.5, 0.5], 6.0)  # improving but worse than incumbent
    vector[0] = 99.0
    assert incumbent.improved
    assert incumbent.vector.tolist() == [0.25, 0.75]
    assert incumbent.cost == 4.0 and incumbent.improving_evaluations == 2
    receipt = incumbent.receipt()
    assert receipt["returned_iterate"] == "search_deadline_incumbent"
    assert receipt["converged"] is False and receipt["initial_cost"] == 10.0
    with pytest.raises(ValueError):
        search_budget.Incumbent(np.nan, np.zeros(2), np.ones(2))


@pytest.mark.parametrize("retain", [False, True])
def test_deadline_before_any_improving_iterate_propagates_original_failure(retain):
    deadline = CountingDeadline(1)  # the initializer cost only; the first optimizer step fires
    with pytest.raises(SearchDeadline, match="test deadline"):
        control_refine(deadline, retain)
    assert deadline.exhausted


def test_deadline_after_finite_improvement_returns_marked_iterate_and_finishes_replay():
    deadline = CountingDeadline(60)
    fitted = control_refine(deadline, True)
    assert deadline.exhausted
    receipt = fitted["search_deadline_incumbent"]
    assert receipt["returned_iterate"] == "search_deadline_incumbent"
    assert receipt["converged"] is False
    assert receipt["improving_evaluations"] >= 1
    assert fitted["primary_optimizer"]["status"] == search_budget.SEARCH_DEADLINE_INCUMBENT_STATUS
    assert fitted["primary_optimizer"]["success"] is False
    assert fitted["primary_optimizer"]["returned_iterate"] == "search_deadline_incumbent"
    assert fitted["message"].startswith("search deadline incumbent, not a converged optimum")
    parameters = np.asarray(fitted["parameters"], float)
    assert parameters.shape == (5 + 6 * 2,) and np.isfinite(parameters).all()
    assert np.all(parameters >= [-10, -15, model.R_BALL] + [-75] * 6 + [-6] * 6 + [0.8, 0.8])
    assert np.all(parameters <= [21, 40, 12] + [75] * 6 + [6] * 6 + [1.2, 1.2])
    # The optimizer objective is reported as the retained iterate's own cost; the
    # ordinary final replay (final_cost, terminal_slack) ran after the deadline.
    assert fitted["final_cost"] < fitted["initial_cost"]
    assert fitted["final_cost"] == pytest.approx(receipt["incumbent_cost"], rel=1e-6, abs=1e-9)
    assert len(fitted["terminal_slack"]) == 3
    # The caller's deadline is not disabled: the next consultation still raises.
    with pytest.raises(SearchDeadline):
        deadline()
    json.dumps(fitted, allow_nan=False)


def test_deadline_after_improvement_without_retention_still_fails():
    deadline = CountingDeadline(60)
    with pytest.raises(SearchDeadline, match="test deadline"):
        control_refine(deadline, False)


def test_deadline_return_never_starts_a_feasibility_optimizer(monkeypatch):
    original = real_exposure_replay.minimize
    calls = []

    def primary_only(*args, **kwargs):
        calls.append(kwargs.get("method"))
        assert len(calls) == 1, "deadline replay must not launch another optimizer"
        return original(*args, **kwargs)

    def no_scalar_restart(*args, **kwargs):
        pytest.fail("deadline replay must not start a scalar feasibility search")

    monkeypatch.setattr(real_exposure_replay, "minimize", primary_only)
    monkeypatch.setattr(real_exposure_replay, "brentq", no_scalar_restart)
    fitted = control_refine(CountingDeadline(60), True)
    assert calls == ["SLSQP"]
    assert fitted["final_gate_feasibility"] is None
    assert fitted["success"] is False
    assert fitted["status"] == search_budget.SEARCH_DEADLINE_INCUMBENT_STATUS
    assert fitted["final_cost"] == pytest.approx(
        fitted["search_deadline_incumbent"]["incumbent_cost"], abs=1e-9
    )


def test_no_timeout_output_is_identical_with_retention_on_and_off():
    off = control_refine(CountingDeadline(10**9), False)
    on = control_refine(CountingDeadline(10**9), True)
    assert "search_deadline_incumbent" not in off and "search_deadline_incumbent" not in on
    assert "returned_iterate" not in on["primary_optimizer"]
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)


def test_retention_requires_single_shooting():
    scene, truth = control_scene()
    shared = replace(scene, parameterization="shared_contact_states")
    with pytest.raises(ValueError, match="single shooting"):
        real_exposure_replay.refine(
            shared,
            truth,
            (np.empty(0), np.empty(0)),
            scene.observation_frames,
            np.zeros((sum(map(len, scene.observation_frames)), 2)),
            None,
            5,
            shared_contact_states=True,
            retain_deadline_incumbent=True,
        )


def candidate(score, *, incumbent=False):
    fit = {"parameters": np.zeros(11)}
    if incumbent:
        fit["search_deadline_incumbent"] = {"returned_iterate": "search_deadline_incumbent"}
    return {
        "stage": "coarse",
        "depth_hypothesis_m": 0.0,
        "evidence": {"input_only_rank_score": score, "survived": False},
        "measurement": {"fit": fit, "native_projection": [{"frame": 1, "predicted": [2, 3]}]},
    }


def test_budget_keeps_completed_candidates_beside_incumbent_and_counts_it(tmp_path):
    budget = search_budget.Budget(5.0, tmp_path / "checkpoint.json")
    budget.retain_incumbent = True
    budget.record(candidate(3.0), 1)
    budget.record(candidate(2.0, incumbent=True), 1)
    assert len(budget.completed) == 2
    receipt = budget.receipt()
    assert receipt["deadline_incumbent_retention"] is True
    assert receipt["deadline_incumbent_candidates"] == 1
    saved = json.loads(budget.checkpoint.read_text())
    assert [row["evidence"]["input_only_rank_score"] for row in saved["completed_candidates"]] == [
        3.0,
        2.0,
    ]
    broken = candidate(1.0, incumbent=True)
    broken["measurement"]["fit"]["parameters"][0] = np.nan
    with pytest.raises(ValueError):
        budget.record(broken, 1)
    assert len(budget.completed) == 2 and budget.receipt()["deadline_incumbent_candidates"] == 1


def test_budget_default_is_off():
    budget = search_budget.Budget(1.0, Path("/nonexistent/x.json"))
    assert budget.retain_incumbent is False
    assert budget.receipt()["deadline_incumbent_retention"] is False
