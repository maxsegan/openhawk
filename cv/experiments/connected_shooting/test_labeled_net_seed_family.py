from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_net_seed_family as family
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record


def context(automatic=False):
    events = [
        dict(event_type=k, frame=f, frame_interval=[f - 0.5, f + 0.5])
        for k, f in [("contact", 0.0), ("net_hit", 10.0), ("bounce", 20.0)]
    ]
    if automatic:
        for e in events:
            e.update(
                status="predicted",
                annotation_origin="automatic",
                exact_epoch_observed=False,
                occurrence_status="predicted",
            )
    return dict(
        scene=SimpleNamespace(
            contact_frames=np.array([0.0, 30.0]),
            pixels=(np.zeros((4, 2)),),
            observation_frames=(np.array([12.0, 18.0, 21.0, 25.0]),),
        ),
        heldout=SimpleNamespace(observation_frames=(np.array([19.0, 22.0]),)),
        events=events,
        bounces=[np.array([20.0])],
        targets=[
            [
                dict(
                    event_frame=20.0,
                    native_frames=[18.0, 19.0, 21.0, 22.0],
                    xyz_m=[4.0, 10.0, 0.0325],
                )
            ]
        ],
        attempt={},
    )


@pytest.mark.parametrize("automatic", [False, True])
def test_existing_occurrence_and_native_target_contract(automatic):
    c = context(automatic)
    r = family.qualify(c)
    assert r["status"] == "qualified"
    assert r["target"] == c["targets"][0][0]
    r["target"]["native_frames"].append(900)
    assert 900 not in c["targets"][0][0]["native_frames"]
    c["events"][1]["status"] = "ambiguous"
    assert family.qualify(c)["status"] == "unavailable"


@pytest.mark.parametrize(
    "change",
    [
        "foreign_frame",
        "pre_net",
        "nan",
        "duplicate_target",
        "missing_target",
        "two_grounds",
        "tail",
        "contact",
    ],
)
def test_unavailable_does_not_invent_a_target(change):
    c = context()
    if change == "foreign_frame":
        c["targets"][0][0]["native_frames"] = [18.0, 900.0]
    elif change == "pre_net":
        c["targets"][0][0]["native_frames"] = [10.0, 18.0]
    elif change == "nan":
        c["targets"][0][0]["xyz_m"][0] = np.nan
    elif change == "duplicate_target":
        c["targets"][0].append(deepcopy(c["targets"][0][0]))
    elif change == "missing_target":
        c["targets"] = [[]]
    elif change == "two_grounds":
        c["events"].append(dict(event_type="bounce", frame=25.0, frame_interval=[24.5, 25.5]))
        c["bounces"] = [np.array([20.0, 25.0])]
    elif change == "tail":
        c["scene"].terminal_net_tail = {"mode": "observed_tail"}
    else:
        c["scene"].right_boundary_kind = "original_contact"
    assert family.qualify(c)["status"] == "unavailable"


def result(cost, velocity):
    return dict(
        best=dict(
            parameters=np.arange(11, dtype=float),
            cost=cost,
            response=dict(outgoing_velocity_mps=velocity),
        ),
        termination=dict(success=True, nfev=2),
        residual_calls=9,
    )


def test_off_preserves_exact_call_and_return_even_without_qualification(monkeypatch):
    expected = result(4, [1, 2, 3])
    calls = []
    monkeypatch.setattr(family.epoch, "fit", lambda *a, **k: calls.append((a, k)) or expected)
    p = np.arange(11)
    returned = family.fit({}, p, 0.25, maxiter=7, seconds=20, free_net_velocity=True)
    assert returned is expected
    assert calls == [(({}, p, 0.25), dict(maxiter=7, seconds=20, free_net_velocity=True))]


def test_unavailable_retains_ordinary_full_budget(monkeypatch):
    calls = []
    c = context()
    c["targets"] = [[]]
    monkeypatch.setattr(
        family.epoch, "fit", lambda *a, **k: calls.append(k) or result(9, [0, 1, 2])
    )
    got = family.fit(
        c, np.arange(11), 0.25, enabled=True, maxiter=7, seconds=20, free_net_velocity=True
    )
    assert calls == [dict(maxiter=7, seconds=20, free_net_velocity=True)]
    assert got["seed_family"]["additional_runs"] == 0


@pytest.mark.parametrize(
    "costs,chosen",
    [([8.0, 2.0], "observed_first_ground"), ([2.0, 8.0], "ordinary"), ([2.0, 2.0], "ordinary")],
)
def test_two_families_share_budget_source_and_atomic_response(monkeypatch, costs, chosen):
    c = context()
    p = np.arange(11)
    clock = iter([0.0, 0.0, 4.0, 6.0])
    monkeypatch.setattr(family.time, "monotonic", lambda: next(clock))
    calls = []

    def fit(ctx, source, duration, **kw):
        assert ctx is c and source is p and duration == 0.25
        calls.append(kw)
        i = len(calls) - 1
        return result(costs[i], [i + 1, 2, 3])

    monkeypatch.setattr(family.epoch, "fit", fit)
    got = family.fit(c, p, 0.25, enabled=True, maxiter=7, seconds=20, free_net_velocity=True)
    assert [x["maxiter"] for x in calls] == [4, 3]
    assert [x["seconds"] for x in calls] == [10, 16]
    assert "first_ground_seed" not in calls[0]
    assert calls[1]["first_ground_seed"] == c["targets"][0][0]
    assert got["seed_family"]["selected"] == chosen
    assert (
        got["best"]
        is got["seed_family"]["candidates"][0 if chosen == "ordinary" else 1]["result"]["best"]
    )
    law = response_from_record(got["best"]["response"])
    np.testing.assert_array_equal(
        law.outgoing_velocity_mps, [1 if chosen == "ordinary" else 2, 2, 3]
    )
    assert got["seed_family"]["gate_selection"] is False


def test_ordinary_failure_does_not_block_additional_candidate(monkeypatch):
    calls = []

    def fit(*a, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise ValueError("measured dynamics bounce cap reached")
        return result(5, [2, 3, 4])

    monkeypatch.setattr(family.epoch, "fit", fit)
    got = family.fit(
        context(), np.arange(11), 0.25, enabled=True, maxiter=8, seconds=20, free_net_velocity=True
    )
    assert got["seed_family"]["selected"] == "observed_first_ground"
    assert got["seed_family"]["candidates"][0]["status"] == "execution_failed"


def test_deadline_preserves_completed_ordinary_and_never_starts_late(monkeypatch):
    clock = iter([0.0, 0.0, 25.0, 25.0])
    monkeypatch.setattr(family.time, "monotonic", lambda: next(clock))
    calls = []
    monkeypatch.setattr(
        family.epoch, "fit", lambda *a, **k: calls.append(k) or result(9, [1, 2, 3])
    )
    got = family.fit(
        context(), np.arange(11), 0.25, enabled=True, maxiter=8, seconds=20, free_net_velocity=True
    )
    assert len(calls) == 1
    assert got["seed_family"]["selected"] == "ordinary"
    assert got["seed_family"]["candidates"][1]["status"] == "deadline_before_start"


def test_existing_two_ground_or_joint_toss_is_not_reinitialized(monkeypatch):
    calls = []
    monkeypatch.setattr(
        family.epoch, "fit", lambda *a, **k: calls.append(k) or result(1, [1, 2, 3])
    )
    for extra in [
        dict(first_ground_seed={"existing": True}),
        dict(first_contact_toss={"incoming": True}),
    ]:
        got = family.fit(
            context(),
            np.arange(11),
            0.25,
            enabled=True,
            maxiter=8,
            seconds=20,
            free_net_velocity=True,
            **extra,
        )
        assert got["seed_family"]["additional_runs"] == 0
    assert len(calls) == 2 and all(c["maxiter"] == 8 for c in calls)


def test_default_off_shared_policy_and_invalid_nonboolean():
    from cv.pipeline.s6_shared_refinement import Policy
    from cv.pipeline.s6_labeled_stage import shared_settings

    assert Policy().net_first_ground_candidates is False
    assert shared_settings({})["net_first_ground_candidates"] == "off"
    assert (
        shared_settings({"net_first_ground_candidates": "on"})["net_first_ground_candidates"]
        == "on"
    )
    with pytest.raises(ValueError, match="explicit boolean"):
        Policy(net_first_ground_candidates="on").validate()


def test_malformed_optional_target_is_unavailable():
    c = context()
    del c["targets"]
    assert family.qualify(c)["status"] == "unavailable"
    assert family.qualify({})["status"] == "unavailable"


def test_heldout_witness_is_consumed_evidence():
    c = context()
    c["targets"][0][0]["native_frames"] = [19.0, 22.0]
    receipt = family.qualify(c)
    assert receipt["status"] == "qualified"
    assert "no independent heldout claim" in receipt["evidence_consumed_by_fit"]


def test_no_finite_visited_candidate_is_failure(monkeypatch):
    monkeypatch.setattr(family.epoch, "fit", lambda *a, **k: {"best": {"cost": float("nan")}})
    with pytest.raises(ValueError, match="no evaluable terminal-net seed family"):
        family.fit(
            context(), [0], 0.25, enabled=True, maxiter=8, seconds=20, free_net_velocity=True
        )


def test_driver_parser_to_stage_policy(tmp_path, monkeypatch):
    import json
    import sys
    from cv.validation import run_labeled_s6 as driver
    from cv.pipeline import s6_labeled_stage as stage

    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"schema": stage.SCHEMA, "rows": []}))
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_labeled_s6",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--net-first-ground-candidates",
            "on",
        ],
    )
    driver.main()
    policy = json.loads((output / "manifest.json").read_text())["policy"]
    assert stage.shared_settings(policy)["net_first_ground_candidates"] == "on"


@pytest.mark.parametrize("available", [True, False])
def test_shared_dispatch_runs_optional_family_then_scores_once(monkeypatch, available):
    from contextlib import nullcontext
    from cv.pipeline import s6_shared_refinement as shared

    c = context(automatic=True)
    eligibility = family.recipe.qualify(c)
    if not available:
        c["targets"][-1] = []
    calls, scores = [], []
    monkeypatch.setattr(shared.net.original.followup, "fit_check_copy", lambda s, h: (h, []))
    monkeypatch.setattr(shared.net, "continuous_ground", lambda _: nullcontext())
    monkeypatch.setattr(shared.serve.full.model, "chain", lambda *a: [])
    monkeypatch.setattr(shared, "_validate_state", lambda old, new, p: p)

    def fitted(*a, **kw):
        calls.append(kw)
        return result(10 - len(calls), [len(calls), 2, 3])

    monkeypatch.setattr(family.epoch, "fit", fitted)
    monkeypatch.setattr(shared, "_score", lambda *a: (scores.append(a) or {}, {}))
    output = shared._fit_net(
        c,
        np.arange(11),
        {"duration": 0.25},
        shared.Policy(net_first_ground_candidates=True, net_max_nfev=8),
        {"eligibility": eligibility},
        seconds=20,
    )
    assert len(calls) == (2 if available else 1)
    assert len(scores) == 1
    assert output["response"]["outgoing_velocity_mps"][0] == len(calls)
    assert output["receipt"]["fit"]["seed_family"]["selected"] == (
        "observed_first_ground" if available else "ordinary"
    )
