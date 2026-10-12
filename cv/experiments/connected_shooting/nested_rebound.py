"""Research control: extend a fixed-rebound solution without losing its objective value.

No truth-based selection or physical acceptance. The fixed model must embed exactly
in the extended model; both phases use the same images and event evidence.
"""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cv.experiments.connected_shooting.model import Scene


def refine(
    scene: Scene,
    initial: np.ndarray,
    solve: Callable[[Scene, np.ndarray], dict],
    objective_cost: Callable[[np.ndarray], float],
    prior_receipt: Callable[[np.ndarray], dict | None],
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    retain_direct: bool = False,
) -> dict:
    from cv.experiments.connected_shooting import model, regime_recovery

    fixed = solve(replace(scene, rebound_mode="fixed"), initial[:-2])
    parameters = np.r_[fixed["parameters"], 1.0, 1.0]
    replay = model.chain(scene, parameters)
    if not regime_recovery.exact_replay_matches(replay, fixed["flights"]):
        raise ValueError("fixed rebound solution does not embed exactly in extended physics")
    cost = objective_cost(parameters)
    if not np.isfinite(cost) or not np.isclose(
        cost, fixed["optimizer_evidence"]["cost"], atol=1e-8, rtol=1e-10
    ):
        raise ValueError("fixed rebound objective does not embed in extended objective")
    baseline = {
        **fixed,
        "parameters": parameters,
        "flights": replay,
        "rebound_scale_factors": [1.0, 1.0],
        "rebound_prior_evidence": prior_receipt(parameters),
        "optimizer_evidence": {
            **fixed["optimizer_evidence"],
            "cost": cost,
            "scope": "fixed_rebound_subproblem_not_extended_stationarity",
        },
    }
    lower, upper = bounds
    seed = np.clip(parameters, np.nextafter(lower, upper), np.nextafter(upper, lower))
    receipt = {
        "policy": "fixed_then_extended_retained_objective_v1",
        "selection": "lower_same_objective_only_not_truth_or_physical_acceptance",
        "fixed_parameters": fixed["parameters"].tolist(),
        "fixed_optimizer": fixed["optimizer_evidence"],
        "fixed_regime_recovery": fixed["regime_recovery_evidence"],
        "embedded_parameters": parameters.tolist(),
        "embedded_objective_cost": cost,
        "extended_initial_parameters": seed.tolist(),
        "warm_seed_adjustment": (seed - parameters).tolist(),
        "fixed_states_and_impacts_identical": True,
        "selected": "fixed",
        "extension": {"status": "held"},
    }
    selected, calls = baseline, fixed["objective_calls"]
    complete_calls = fixed.get("objective_calls_complete", True)
    deadline_exhausted = bool(fixed.get("point_deadline_exhausted", False))
    phases = [("extension", "extended", seed)]
    if retain_direct:
        receipt.update(
            policy="fixed_direct_warm_retained_objective_v2",
            direct_initial_parameters=initial.tolist(),
            direct={"status": "not_run"},
        )
        receipt["extension"] = {"status": "not_run"}
        phases.insert(0, ("direct", "direct", initial.copy()))
    selected_cost = cost
    for phase, selection, phase_seed in phases:
        if deadline_exhausted:
            receipt[phase].update(status="not_run", reason="point_deadline_exhausted")
            continue
        receipt[phase]["status"] = "held"
        try:
            candidate = solve(scene, phase_seed)
            deadline_exhausted = bool(candidate.get("point_deadline_exhausted", False))
            calls += candidate["objective_calls"]
            complete_calls &= candidate.get("objective_calls_complete", True)
            candidate_cost = objective_cost(candidate["parameters"])
            receipt[phase].update(
                status="measured",
                optimizer=candidate["optimizer_evidence"],
                parameters=candidate["parameters"].tolist(),
                regime_recovery=candidate["regime_recovery_evidence"],
                replayed_objective_cost=candidate_cost,
            )
            if not np.isfinite(candidate_cost) or not np.isclose(
                candidate_cost, candidate["optimizer_evidence"]["cost"], atol=1e-8, rtol=1e-10
            ):
                receipt[phase].update(status="held", reason="extended_objective_replay_mismatch")
            elif retain_direct and not regime_recovery.exact_replay_matches(
                model.chain(scene, candidate["parameters"]), candidate["flights"]
            ):
                receipt[phase].update(status="held", reason="extended_state_replay_mismatch")
            elif candidate_cost < selected_cost:
                selected, selected_cost = candidate, candidate_cost
                receipt["selected"] = selection
        except (ValueError, FloatingPointError, OverflowError, TimeoutError) as exc:
            complete_calls = False  # interrupted solve does not return its evaluation counter
            receipt[phase].update(status="held", reason=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, TimeoutError):
                deadline_exhausted = True
                # The caller's point alarm is one-shot. Never start another
                # solve after catching its expiry; that would run unbounded.
                if phase == "direct":
                    receipt["extension"].update(reason="point_deadline_exhausted")
                break
    return {
        **selected,
        "initial_pixel_rms": fixed["initial_pixel_rms"],
        "objective_calls": calls,
        "objective_calls_complete": complete_calls,
        "nested_rebound_evidence": receipt,
        **({"point_deadline_exhausted": True} if deadline_exhausted else {}),
    }
