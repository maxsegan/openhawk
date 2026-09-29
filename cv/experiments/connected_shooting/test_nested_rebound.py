from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import model, nested_rebound


def setup_control():
    scene, truth = model.control()
    fixed_scene = replace(scene, dynamics="measured_240hz")
    fixed = model.fit(fixed_scene, truth + 0.05, max_nfev=1)
    scene = replace(fixed_scene, rebound_mode="point_scales")
    initial = np.r_[truth + 0.05, 1, 1]
    parameters = np.r_[fixed["parameters"], 1, 1]
    bounds = (np.full(len(parameters), -100.0), np.full(len(parameters), 100.0))
    return scene, initial, fixed, parameters, bounds


@pytest.mark.parametrize("change,selected", [(-1.0, "extended"), (0.0, "fixed"), (1.0, "fixed")])
def test_selection_uses_replayed_objective_and_retains_fixed_ties(change, selected):
    scene, initial, fixed, parameters, bounds = setup_control()
    candidate_parameters = parameters.copy()
    candidate_parameters[0] += 0.1
    cost = fixed["optimizer_evidence"]["cost"]
    calls = []

    def solve(phase, seed):
        calls.append((phase.rebound_mode, seed.copy()))
        if phase.rebound_mode == "fixed":
            return fixed
        return {
            **fixed,
            "parameters": candidate_parameters,
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": cost + change},
        }

    def objective(theta):
        return cost if np.array_equal(theta, parameters) else cost + change

    result = nested_rebound.refine(scene, initial, solve, objective, lambda _: None, bounds)
    assert [c[0] for c in calls] == ["fixed", "point_scales"]
    np.testing.assert_array_equal(calls[0][1], initial[:-2])
    np.testing.assert_array_equal(calls[1][1], parameters)
    assert result["nested_rebound_evidence"]["selected"] == selected
    np.testing.assert_array_equal(
        result["parameters"], candidate_parameters if selected == "extended" else parameters
    )
    assert result["objective_calls"] == 2 * fixed["objective_calls"]
    assert result["objective_calls_complete"]


@pytest.mark.parametrize("error", [TimeoutError, ValueError, FloatingPointError, OverflowError])
def test_failed_extension_returns_fixed_with_explicit_incomplete_work_accounting(error):
    scene, initial, fixed, parameters, bounds = setup_control()

    def solve(phase, seed):
        if phase.rebound_mode == "fixed":
            return fixed
        raise error("extension interrupted")

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda _: fixed["optimizer_evidence"]["cost"],
        lambda _: None,
        bounds,
    )
    np.testing.assert_array_equal(result["parameters"], parameters)
    assert not result["objective_calls_complete"]
    assert result["nested_rebound_evidence"]["extension"]["status"] == "held"
    assert error.__name__ in result["nested_rebound_evidence"]["extension"]["reason"]
    assert "not_extended_stationarity" in result["optimizer_evidence"]["scope"]


def test_false_lower_reported_cost_cannot_replace_fixed():
    scene, initial, fixed, parameters, bounds = setup_control()

    def solve(phase, seed):
        if phase.rebound_mode == "fixed":
            return fixed
        return {
            **fixed,
            "parameters": parameters,
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": 0.0},
        }

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda _: fixed["optimizer_evidence"]["cost"],
        lambda _: None,
        bounds,
    )
    assert result["nested_rebound_evidence"]["selected"] == "fixed"
    assert (
        result["nested_rebound_evidence"]["extension"]["reason"]
        == "extended_objective_replay_mismatch"
    )


def test_fixed_embedding_mismatch_fails_before_extension(monkeypatch):
    from cv.experiments.connected_shooting import regime_recovery

    scene, initial, fixed, _, bounds = setup_control()
    monkeypatch.setattr(regime_recovery, "exact_replay_matches", lambda *_: False)
    with pytest.raises(ValueError, match="embed exactly"):
        nested_rebound.refine(scene, initial, lambda *_: fixed, lambda _: 0, lambda _: None, bounds)


def test_bound_adjustment_applies_only_to_warm_seed_not_retained_solution():
    scene, initial, fixed, parameters, bounds = setup_control()
    bounds[1][0] = parameters[0]
    received = []

    def solve(phase, seed):
        if phase.rebound_mode == "fixed":
            return fixed
        received.append(seed.copy())
        raise ValueError("retain baseline for this control")

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda _: fixed["optimizer_evidence"]["cost"],
        lambda _: None,
        bounds,
    )
    assert received[0][0] == np.nextafter(parameters[0], -np.inf)
    np.testing.assert_array_equal(result["parameters"], parameters)
    assert result["nested_rebound_evidence"]["warm_seed_adjustment"][0] < 0


def test_real_nested_fit_records_both_phases_and_no_larger_cost():
    scene, initial, _, _, _ = setup_control()
    result = model.fit(
        scene, initial, max_nfev=3, rebound_prior_scale=0.02, warm_start_rebound=True
    )
    receipt = result["nested_rebound_evidence"]
    assert receipt["fixed_states_and_impacts_identical"]
    assert result["optimizer_evidence"]["cost"] <= receipt["embedded_objective_cost"]
    assert result["rebound_prior_evidence"]["scale"] == 0.02
    assert result["junction_gaps_m"] == [0.0]


def test_warm_start_requires_unforced_point_mode_and_unit_seed():
    scene, truth = model.control()
    with pytest.raises(ValueError, match="unforced point-scale"):
        model.fit(scene, truth, warm_start_rebound=True)
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    with pytest.raises(ValueError, match="unit initial corrections"):
        model.fit(scene, np.r_[truth, 1.01, 1], warm_start_rebound=True)
    with pytest.raises(ValueError, match="unforced point-scale"):
        model.fit(
            replace(scene, bounce_regime_override=(0, 0, "slide")),
            np.r_[truth, 1, 1],
            warm_start_rebound=True,
        )


@pytest.mark.parametrize(
    "direct_ratio,warm_ratio,selected",
    [(0.2, 0.5, "direct"), (0.5, 0.2, "extended"), (1, 1, "fixed"), (0.5, 0.5, "direct")],
)
def test_three_candidates_keep_lowest_replayed_cost_and_earliest_tie(
    direct_ratio, warm_ratio, selected
):
    scene, initial, fixed, embedded, bounds = setup_control()
    received = []
    cost = fixed["optimizer_evidence"]["cost"]
    direct, warm = embedded.copy(), embedded.copy()
    direct[0] += 0.1
    warm[0] += 0.2

    def objective(theta):
        return cost * (
            direct_ratio
            if np.array_equal(theta, direct)
            else warm_ratio
            if np.array_equal(theta, warm)
            else 1
        )

    def solve(phase, seed):
        received.append(seed.copy())
        if phase.rebound_mode == "fixed":
            return fixed
        theta = direct if len(received) == 2 else warm
        return {
            **fixed,
            "parameters": theta,
            "flights": model.chain(scene, theta),
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": objective(theta)},
        }

    result = nested_rebound.refine(
        scene, initial, solve, objective, lambda _: None, bounds, retain_direct=True
    )
    receipt = result["nested_rebound_evidence"]
    assert receipt["selected"] == selected
    assert receipt["policy"] == "fixed_direct_warm_retained_objective_v2"
    np.testing.assert_array_equal(received[1], initial)
    np.testing.assert_array_equal(received[2], embedded)
    assert result["objective_calls"] == 3 * fixed["objective_calls"]
    assert result["objective_calls_complete"]


@pytest.mark.parametrize("timeout_phase", ["direct", "extension"])
def test_point_timeout_never_starts_another_phase_and_retains_best_completed(timeout_phase):
    scene, initial, fixed, embedded, bounds = setup_control()
    count = 0
    direct = embedded.copy()
    direct[0] += 0.1
    cost = fixed["optimizer_evidence"]["cost"]

    def solve(phase, seed):
        nonlocal count
        count += 1
        if phase.rebound_mode == "fixed":
            return fixed
        if timeout_phase == "direct" or count == 3:
            raise TimeoutError("point deadline")
        return {
            **fixed,
            "parameters": direct,
            "flights": model.chain(scene, direct),
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": cost / 2},
        }

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda theta: cost if np.array_equal(theta, embedded) else cost / 2,
        lambda _: None,
        bounds,
        retain_direct=True,
    )
    receipt = result["nested_rebound_evidence"]
    assert count == (2 if timeout_phase == "direct" else 3)
    assert receipt["selected"] == ("fixed" if timeout_phase == "direct" else "direct")
    assert not result["objective_calls_complete"]
    if timeout_phase == "direct":
        assert receipt["extension"] == {"status": "not_run", "reason": "point_deadline_exhausted"}


@pytest.mark.parametrize("phase", ["fixed", "direct"])
@pytest.mark.parametrize("ratio", [0.5, 1.5])
def test_inner_recovery_deadline_propagates_without_starting_new_phases(phase, ratio):
    scene, initial, fixed, embedded, bounds = setup_control()
    calls = []
    cost = fixed["optimizer_evidence"]["cost"]
    direct = embedded.copy()
    direct[0] += 0.1

    def solve(s, seed):
        calls.append(s.rebound_mode)
        if s.rebound_mode == "fixed":
            return {
                **fixed,
                **(
                    {"point_deadline_exhausted": True, "objective_calls_complete": False}
                    if phase == "fixed"
                    else {}
                ),
            }
        assert len(calls) == 2, "a consumed one-shot deadline must not launch warm search"
        return {
            **fixed,
            "parameters": direct,
            "flights": model.chain(scene, direct),
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": cost * ratio},
            "point_deadline_exhausted": True,
            "objective_calls_complete": False,
        }

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda p: cost if np.array_equal(p, embedded) else cost * ratio,
        lambda _: None,
        bounds,
        retain_direct=True,
    )
    assert len(calls) == (1 if phase == "fixed" else 2)
    assert result["point_deadline_exhausted"] and not result["objective_calls_complete"]
    expected = "direct" if phase == "direct" and ratio < 1 else "fixed"
    assert result["nested_rebound_evidence"]["selected"] == expected
    assert result["nested_rebound_evidence"]["extension"] == {
        "status": "not_run",
        "reason": "point_deadline_exhausted",
    }


def test_inconsistent_direct_flight_states_cannot_win_on_reported_cost():
    scene, initial, fixed, embedded, bounds = setup_control()
    theta = embedded.copy()
    theta[0] += 0.1
    cost = fixed["optimizer_evidence"]["cost"]
    count = 0

    def solve(phase, seed):
        nonlocal count
        count += 1
        if phase.rebound_mode == "fixed":
            return fixed
        if count == 3:
            raise ValueError("warm attempt unavailable")
        return {
            **fixed,
            "parameters": theta,
            "optimizer_evidence": {**fixed["optimizer_evidence"], "cost": cost / 2},
        }

    result = nested_rebound.refine(
        scene,
        initial,
        solve,
        lambda p: cost if np.array_equal(p, embedded) else cost / 2,
        lambda _: None,
        bounds,
        retain_direct=True,
    )
    receipt = result["nested_rebound_evidence"]
    assert receipt["selected"] == "fixed"
    assert receipt["direct"]["reason"] == "extended_state_replay_mismatch"


def test_retained_direct_phase_exactly_reproduces_standalone_direct_fit():
    scene, initial, _, _, _ = setup_control()
    direct = model.fit(scene, initial, max_nfev=3, rebound_prior_scale=0.02)
    combined = model.fit(
        scene,
        initial,
        max_nfev=3,
        rebound_prior_scale=0.02,
        warm_start_rebound=True,
        retain_direct_rebound=True,
    )
    receipt = combined["nested_rebound_evidence"]
    np.testing.assert_array_equal(receipt["direct"]["parameters"], direct["parameters"])
    assert receipt["direct"]["optimizer"] == direct["optimizer_evidence"]
    assert combined["optimizer_evidence"]["cost"] <= min(
        receipt["embedded_objective_cost"], direct["optimizer_evidence"]["cost"]
    )
    with pytest.raises(ValueError, match="warm-start"):
        model.fit(scene, initial, retain_direct_rebound=True)
