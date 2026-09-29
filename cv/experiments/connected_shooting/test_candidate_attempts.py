"""Numerical failure scope, safe continuation and input-only completed fallback."""

import pytest

from cv.experiments.connected_shooting import candidate_attempts as attempt
from cv.experiments.connected_shooting.search_budget import SearchDeadline


def raises(error):
    raise error


def test_only_known_numerical_boundary_becomes_candidate_failure():
    for error in [
        ValueError("missing support"),
        FloatingPointError("bad step"),
        OverflowError("overflow"),
    ]:
        with pytest.raises(attempt.CandidateFailure):
            attempt.numerical_call("fit", raises, error)
    for error in [
        OSError("write failed"),
        RuntimeError("bug"),
        SearchDeadline("stop"),
        KeyboardInterrupt(),
        SystemExit(2),
    ]:
        with pytest.raises(type(error)):
            attempt.numerical_call("fit", raises, error)


def test_retry_scope_propagates_uncategorized_and_publication_faults():
    trials = attempt.Attempts()
    for error in [
        ValueError("checkpoint invalid"),
        OSError("disk failure"),
        SearchDeadline("stop"),
        KeyboardInterrupt(),
    ]:
        with pytest.raises(type(error)):
            trials.run({"seed": "first"}, raises, error)
    assert trials.failures == []


def test_failed_first_seed_does_not_prevent_remaining_fixed_seeds():
    trials = attempt.Attempts()
    rows = []
    for name, score in [("pinhole", None), ("anchor", 2.0), ("anchor_flat", 1.0)]:

        def solve():
            if score is None:
                return attempt.numerical_call("refine", raises, ValueError("missing support"))
            return {"score": score, "gate": score == 2.0}

        row = trials.run({"seed": name}, solve)
        if row is not None:
            rows.append(row)
    assert len(rows) == 2
    assert min(rows, key=lambda r: r["score"]) == {"score": 1.0, "gate": False}
    assert trials.failures == [
        {
            "seed": "pinhole",
            "reason": "refine: ValueError: missing support",
            "failure_kind": "numerical",
        }
    ]


def test_completed_fallback_requires_declared_failure_and_no_refined_output():
    d = {
        "search_budget": {"retained_completed_after_numerical_failures": True},
        "completed_candidates": [{}],
        "numerical_candidate_failures": [{"reason": "bad"}],
        "refined_candidates": [],
    }
    assert attempt.completed_source_allowed(d)
    assert not attempt.completed_source_allowed(d | {"numerical_candidate_failures": []})
    assert not attempt.completed_source_allowed(d | {"refined_candidates": [{}]})
    assert not attempt.completed_source_allowed(d | {"completed_candidates": []})
    assert not attempt.completed_source_allowed({"completed_candidates": [{}]})
    assert attempt.completed_source_allowed({"search_budget": {"exhausted": True}})


def test_completed_validation_is_separate_from_checkpoint_publication():
    import numpy as np

    valid = {
        "measurement": {"fit": {"parameters": [0.0] * 11}, "native_projection": [{}]},
        "evidence": {"input_only_rank_score": 1.0},
    }
    attempt.validate_completed(valid, 1)
    valid["evidence"]["input_only_rank_score"] = np.nan
    with pytest.raises(attempt.CandidateFailure, match="finite input rank"):
        attempt.validate_completed(valid, 1)
    valid["evidence"]["input_only_rank_score"] = 1.0
    valid["measurement"]["fit"]["parameters"] = [0.0]
    with pytest.raises(attempt.CandidateFailure, match="full-scene"):
        attempt.validate_completed(valid, 1)


def test_completed_output_without_deadline_keeps_chosen_topology_only():
    own = {"topology_name": "supplied", "score": 3.0}
    other = {"topology_name": "recovered", "score": 1.0}
    chosen = {
        "topology_name": "supplied",
        "coarse": [own],
        "refined": [],
        "numerical_candidate_failures": [{"reason": "bad refinement"}],
    }
    rows, receipt = attempt.completed_output(chosen, [], {"seconds": None, "exhausted": False})
    assert rows == [own]
    assert receipt["retained_completed_after_numerical_failures"]
    assert receipt["completed_candidates"] == 1
    chosen["refined"] = [own]
    rows, receipt = attempt.completed_output(chosen, [other, own], {"exhausted": True})
    assert rows == [own]
    assert not receipt["retained_completed_after_numerical_failures"]


def test_aggregate_failure_receipts_are_distinct_from_numerical_trials():
    trials = attempt.Attempts()
    trials.run(
        {"stage": "coarse"}, raises, attempt.CandidateFailure("all seeds failed", aggregate=True)
    )
    assert trials.failures[0]["failure_kind"] == "aggregate"


def test_deadline_checks_current_topology_before_retaining_candidates():
    previous = {"topology_name": "supplied"}
    with pytest.raises(SearchDeadline, match="this topology"):
        attempt.completed_at_deadline([previous], "recovered")
    assert attempt.completed_at_deadline([previous], "supplied") == [previous]


def test_common_budget_cannot_lend_present_checkpoint_to_expired_absent_branch(
    tmp_path, monkeypatch
):
    from cv.experiments.connected_shooting import search_budget

    clock = [0.0]
    monkeypatch.setattr(search_budget.time, "monotonic", lambda: clock[0])

    budget = search_budget.Budget(1.0, tmp_path / "checkpoint.json")
    present = {
        "topology_name": "supplied",
        "membership": "present",
        "stage": "coarse",
        "measurement": {
            "fit": {"parameters": [1.0] * 11, "net_response": {"outgoing_velocity_mps": [1, 2, 3]}},
            "native_projection": [{"frame": 1}],
        },
        "evidence": {"input_only_rank_score": 1.0},
    }
    budget.record(present, 1)
    # Same topology name, dimensions and global budget, but no candidate has
    # completed in the absent branch. It must fail rather than inherit a fit.
    clock[0] = 2.0
    with pytest.raises(SearchDeadline, match="this topology"):
        try:
            budget.check()
        except SearchDeadline:
            # The real deadline handler must scope the common checkpoint to the
            # currently expiring branch before deciding that anything survived.
            attempt.completed_at_deadline(budget.completed, "supplied", membership="absent")
    assert budget.exhausted
    absent = {
        **present,
        "membership": "absent",
        "measurement": {"fit": {"parameters": [2.0] * 11}, "native_projection": [{"frame": 1}]},
    }
    budget.record(absent, 1)
    assert attempt.completed_at_deadline(budget.completed, "supplied", membership="absent") == [
        absent
    ]
    chosen = {
        "topology_name": "supplied",
        "membership": "absent",
        "coarse": [absent],
        "refined": [absent],
        "numerical_candidate_failures": [],
    }
    rows, receipt = attempt.completed_output(chosen, budget.completed, budget.receipt())
    assert rows == [absent]
    assert receipt["completed_candidates"] == 1
    assert receipt["checkpointed_candidates_across_topologies"] == 2
    assert receipt["membership"] == "absent"


def test_numerical_seed_death_is_not_a_missing_camera_or_pose():
    on = {"whole_point_seed_fallback": "on"}
    assert attempt.should_fallback_to_components(
        "CandidateFailure: no completed candidate after 20 numerical failures", on
    )
    assert attempt.should_fallback_to_components(
        "ValueError: no finite full-point candidate with recorded input-only rank", on
    )
    assert attempt.should_fallback_to_components("all configured seeds failed numerically", on)
    assert not attempt.should_fallback_to_components(
        "ValueError: automatic far-side player pose is absent at contact", on
    )
    assert not attempt.should_fallback_to_components(
        "ValueError: every visible label requires one supported matching camera", on
    )
    assert not attempt.should_fallback_to_components("OSError: write failed", on)
    assert not attempt.is_fatal_attempt_failure("no completed candidate after 15 numerical failures")
    assert attempt.attempt_seed_failure_reason("no completed candidate after 20 numerical failures") == (
        "attempt_seed_failure: no completed candidate after 20 numerical failures"
    )
    assert attempt.attempt_seed_failure_reason(
        "attempt_seed_failure: already tagged"
    ) == "attempt_seed_failure: already tagged"


def test_whole_point_seed_fallback_is_off_unless_declared_on():
    from cv.pipeline import s6_labeled_stage as stage

    reason = "CandidateFailure: no completed candidate after 20 numerical failures"
    assert not attempt.should_fallback_to_components(reason, {})
    assert not attempt.should_fallback_to_components(reason, {"whole_point_seed_fallback": "off"})
    assert attempt.should_fallback_to_components(reason, {"whole_point_seed_fallback": "on"})
    assert "whole_point_seed_fallback" not in stage.shared_settings({})
    assert stage.shared_settings({"whole_point_seed_fallback": "off"})[
        "whole_point_seed_fallback"
    ] == "off"
    assert stage.shared_settings({"whole_point_seed_fallback": "on"})[
        "whole_point_seed_fallback"
    ] == "on"
    with pytest.raises(ValueError, match="whole_point_seed_fallback"):
        stage.shared_settings({"whole_point_seed_fallback": "maybe"})


def test_membership_coarse_fallback_refuses_a_foreign_response():
    present = {"topology_name": "supplied", "membership": "present"}
    chosen = {
        "topology_name": "supplied",
        "membership": "absent",
        "coarse": [present],
        "refined": [],
        "numerical_candidate_failures": [{"reason": "deadline"}],
    }
    with pytest.raises(ValueError, match="chosen topology membership"):
        attempt.completed_output(chosen, [present], {"exhausted": True})
