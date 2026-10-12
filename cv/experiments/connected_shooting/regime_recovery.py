"""Can bounded retries escape the measured bounce model's slide/grip discontinuity?

Research only. Detect transitions using the unchanged authoritative impact law;
never smooth that law, supply truth parameters, or qualify the resulting path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from physics import impact

if TYPE_CHECKING:
    from cv.experiments.connected_shooting.model import Scene

BOUNDARY_PROBE_RAD_S = 0.01
LAUNCH_RETRY_RAD_S = 2.0


def boundary_evidence(scene: Scene, flights: list[dict]) -> list[dict]:
    """Find returned rebounds whose regime changes under a tiny incoming-spin probe."""
    evidence = []
    for index, flight in enumerate(flights):
        for ordinal, bounce in enumerate(flight["bounces"]):
            if (
                bounce["frame"] + bounce.get("dwell_seconds", 0.0) * scene.fps
                >= flight["end_frame"]
            ):
                continue  # unused outgoing terminal state cannot cause this image-fit kink
            velocity = np.asarray(bounce["v_in"], float)
            spin = np.asarray(bounce["w_in"], float)
            horizontal = float(np.linalg.norm(velocity[:2]))
            downward = -float(velocity[2])
            if horizontal <= 0 or downward <= 0:
                raise ValueError("descending forward impact required for regime diagnostics")
            axis = np.array([-velocity[1], velocity[0], 0.0]) / horizontal
            topspin = float(spin @ axis)
            regimes = [
                impact.court_bounce(
                    horizontal, downward, topspin + delta, surface=scene.surface
                ).regime
                for delta in (-BOUNDARY_PROBE_RAD_S, BOUNDARY_PROBE_RAD_S)
            ]
            if regimes[0] != regimes[1]:
                evidence.append(
                    {
                        "flight_index": index,
                        "bounce_ordinal": ordinal,
                        "frame": float(bounce["frame"]),
                        "incoming_topspin_rad_s": topspin,
                        "probe_regimes": regimes,
                    }
                )
    return evidence


def refine(scene: Scene, baseline: dict, fit_candidate: Callable[[np.ndarray], dict]) -> dict:
    """At most two full-objective retries on the first detected boundary flight.

    A candidate may initially cost more across the discontinuity. Refine both
    directions before selection, retaining the baseline if neither improves it.
    Physical qualification remains a separate, unchanged requirement.
    """
    boundaries = boundary_evidence(scene, baseline["flights"])
    receipt = {
        "policy": "first_boundary_two_launch_spin_retries_v1",
        "incoming_boundary_probe_rad_s": BOUNDARY_PROBE_RAD_S,
        "launch_retry_rad_s": LAUNCH_RETRY_RAD_S,
        "boundaries": boundaries,
        "baseline_parameters": baseline["parameters"].tolist(),
        "baseline_optimizer": baseline["optimizer_evidence"],
        "selection": "strictly_lower_same_objective_cost_not_truth_or_physical_acceptance",
        "selected": "baseline",
        "attempts": [],
    }
    selected = baseline
    calls = baseline["objective_calls"]
    complete_calls = baseline.get("objective_calls_complete", True)
    deadline_exhausted = bool(baseline.get("point_deadline_exhausted", False))
    if boundaries and not deadline_exhausted:
        index = boundaries[0]["flight_index"]
        n = len(scene.contact_frames) - 1
        parameter_index = 3 + 3 * n + 3 * index  # launch topspin, hundreds of rad/s
        for direction in (-1, 1):
            seed = baseline["parameters"].copy()
            seed[parameter_index] += direction * LAUNCH_RETRY_RAD_S / 100.0
            attempt = {
                "direction": direction,
                "flight_index": index,
                "parameter_index": parameter_index,
                "initial_parameters": seed.tolist(),
                "status": "held",
            }
            if not -6 < seed[parameter_index] < 6:
                attempt["reason"] = "retry_outside_existing_spin_bounds"
            else:
                try:
                    candidate = fit_candidate(seed)
                    calls += candidate["objective_calls"]
                    complete_calls &= candidate.get("objective_calls_complete", True)
                    attempt.update(status="measured", optimizer=candidate["optimizer_evidence"])
                    cost = candidate["optimizer_evidence"]["cost"]
                    if np.isfinite(cost) and cost < selected["optimizer_evidence"]["cost"]:
                        selected = candidate
                        receipt["selected"] = f"direction_{direction}"
                except (ValueError, FloatingPointError, OverflowError, TimeoutError) as exc:
                    complete_calls = False
                    attempt["reason"] = f"{type(exc).__name__}: {exc}"
                    deadline_exhausted = isinstance(exc, TimeoutError)
            receipt["attempts"].append(attempt)
            if deadline_exhausted:
                break  # The caller's one-shot alarm has expired; never start another solve.
    return {
        **selected,
        "initial_pixel_rms": baseline["initial_pixel_rms"],
        "objective_calls": calls,
        "objective_calls_complete": complete_calls,
        "regime_recovery_evidence": receipt,
        **({"point_deadline_exhausted": True} if deadline_exhausted else {}),
    }


def exact_replay_matches(unforced: list[dict], surrogate: list[dict]) -> bool:
    """Require both native states and the full recorded impact inventory to agree."""
    if len(unforced) != len(surrogate):
        return False
    for a, b in zip(unforced, surrogate, strict=True):
        if any(
            not np.array_equal(a[key], b[key])
            for key in ("positions", "velocities", "start_xyz", "end_xyz")
        ) or len(a["bounces"]) != len(b["bounces"]):
            return False
        for left, right in zip(a["bounces"], b["bounces"], strict=True):
            if left["regime"] != right["regime"] or any(
                not np.array_equal(left[key], right[key])
                for key in ("frame", "x", "v_in", "v_out", "w_in", "w_out")
            ):
                return False
    return True


def refine_fixed_branches(
    scene: Scene, baseline: dict, fit_candidate: Callable[[np.ndarray, tuple], dict]
) -> dict:
    """Solve both smooth formula extensions; require exact unforced replay to select.

    Only the first detected transition is enumerated. A surrogate optimum that
    chooses the wrong physical regime is held, not silently returned as physics.
    No continuation of the forced model is ever exported as the selected path.
    """
    from cv.experiments.connected_shooting import model

    if scene.bounce_regime_override is not None:
        raise ValueError("unforced baseline scene required")
    boundaries = boundary_evidence(scene, baseline["flights"])
    receipt = {
        "policy": "first_boundary_two_consistent_formula_extensions_v1",
        "incoming_boundary_probe_rad_s": BOUNDARY_PROBE_RAD_S,
        "boundaries": boundaries,
        "baseline_parameters": baseline["parameters"].tolist(),
        "baseline_optimizer": baseline["optimizer_evidence"],
        "selection": "strictly_lower_objective_only_after_exact_unforced_replay",
        "selected": "baseline",
        "attempts": [],
    }
    selected = baseline
    calls = baseline["objective_calls"]
    complete_calls = baseline.get("objective_calls_complete", True)
    deadline_exhausted = bool(baseline.get("point_deadline_exhausted", False))
    if boundaries and not deadline_exhausted:
        boundary = boundaries[0]
        index, ordinal = boundary["flight_index"], boundary["bounce_ordinal"]
        for regime in ("slide", "grip"):
            override = (index, ordinal, regime)
            attempt = {
                "override": list(override),
                "initial_parameters": baseline["parameters"].tolist(),
                "status": "held",
                "consistent_with_unforced_model": False,
            }
            try:
                candidate = fit_candidate(baseline["parameters"].copy(), override)
                calls += candidate["objective_calls"]
                complete_calls &= candidate.get("objective_calls_complete", True)
                attempt.update(status="measured", optimizer=candidate["optimizer_evidence"])
                unforced = model.chain(scene, candidate["parameters"])
                bounces = unforced[index]["bounces"]
                consistent = ordinal < len(bounces) and bounces[ordinal]["regime"] == regime
                attempt["consistent_with_unforced_model"] = consistent
                if not consistent:
                    attempt["reason"] = "surrogate_optimum_outside_its_physical_regime"
                elif not exact_replay_matches(unforced, candidate["flights"]):
                    attempt["reason"] = "unforced_state_or_impact_replay_mismatch"
                else:
                    attempt["unforced_states_and_impacts_identical"] = True
                    cost = candidate["optimizer_evidence"]["cost"]
                    if np.isfinite(cost) and cost < selected["optimizer_evidence"]["cost"]:
                        selected = {
                            **candidate,
                            "flights": unforced,
                            "surrogate_bounce_override": None,
                        }
                        receipt["selected"] = regime
            except (ValueError, FloatingPointError, OverflowError, TimeoutError) as exc:
                complete_calls = False
                attempt["reason"] = f"{type(exc).__name__}: {exc}"
                deadline_exhausted = isinstance(exc, TimeoutError)
            receipt["attempts"].append(attempt)
            if deadline_exhausted:
                break
    return {
        **selected,
        "initial_pixel_rms": baseline["initial_pixel_rms"],
        "objective_calls": calls,
        "objective_calls_complete": complete_calls,
        "regime_recovery_evidence": receipt,
        **({"point_deadline_exhausted": True} if deadline_exhausted else {}),
    }
