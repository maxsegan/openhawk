"""Bounded incoming-front velocity warm start at fixed contact XYZ and epoch.

Default initialization in the common labeled prefix runner; the low-level
prefix fit retains its explicit legacy default. The caller supplies the unchanged
native front/sigma residual. Only three incoming velocity coordinates move;
there are no XYZ targets, physical priors, expanded bounds or acceptance inputs.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

VELOCITY_LIMIT_MPS = 12.0
MAX_NFEV = 60


def refine(
    raw_velocity: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    *,
    max_nfev: int = MAX_NFEV,
) -> tuple[np.ndarray, dict]:
    """Retain the best actual input likelihood within the existing velocity box."""
    began = time.monotonic()
    raw = np.asarray(raw_velocity, dtype=float)
    if raw.shape != (3,) or not np.isfinite(raw).all():
        raise ValueError("three finite raw incoming velocity coordinates required")
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("positive integer initializer evaluation budget required")
    initial = np.clip(raw, -VELOCITY_LIMIT_MPS, VELOCITY_LIMIT_MPS)
    first = np.asarray(residual(initial), dtype=float).ravel()
    if first.size == 0 or not np.isfinite(first).all():
        raise ValueError("initial incoming likelihood must be finite and nonempty")
    state = dict(velocity=initial.copy(), cost=float(first @ first), calls=1)

    def evaluate(v: np.ndarray) -> np.ndarray:
        state["calls"] += 1
        if not np.isfinite(v).all() or np.any(np.abs(v) > VELOCITY_LIMIT_MPS):
            raise ValueError("initializer trial outside original velocity box")
        value = np.asarray(residual(v), dtype=float).ravel()
        if value.shape != first.shape or not np.isfinite(value).all():
            raise ValueError("incoming residual shape/finite-value contract changed")
        cost = float(value @ value)
        if cost < state["cost"]:
            state.update(velocity=np.array(v, copy=True), cost=cost)
        return value

    try:
        solved = least_squares(
            evaluate,
            initial,
            bounds=(-VELOCITY_LIMIT_MPS, VELOCITY_LIMIT_MPS),
            x_scale="jac",
            max_nfev=max_nfev,
        )
        evaluate(solved.x)
        termination = dict(
            status="optimizer",
            success=bool(solved.success),
            message=str(solved.message),
            nfev=int(solved.nfev),
            optimality=float(solved.optimality),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        termination = dict(
            status="execution_failure_retained_valid_input_iterate", error=repr(error)
        )
    chosen = np.asarray(state["velocity"])
    repeated = np.asarray(residual(chosen), dtype=float).ravel()
    repeated_cost = float(repeated @ repeated)
    if (
        repeated.shape != first.shape
        or not np.isfinite(repeated).all()
        or abs(repeated_cost - state["cost"]) > 1e-8 * max(1.0, state["cost"])
    ):
        raise ValueError("chosen incoming initialization did not reproduce its objective")
    return chosen.copy(), dict(
        mode="bounded_front",
        raw_linear_velocity_mps=raw.tolist(),
        clipped_start_velocity_mps=initial.tolist(),
        chosen_velocity_mps=chosen.tolist(),
        initial_cost=float(first @ first),
        chosen_cost=repeated_cost,
        retained_original_clipped_seed=bool(np.array_equal(chosen, initial)),
        selection="minimum valid actual incoming front/sigma squared residual, including original clipped seed",
        bounds_mps=[-VELOCITY_LIMIT_MPS, VELOCITY_LIMIT_MPS],
        max_nfev=max_nfev,
        residual_calls=state["calls"] + 1,
        seconds=time.monotonic() - began,
        termination=termination,
        fixed_contact_xyz_and_epoch=True,
        priors_added=False,
        gate_or_truth_selection=False,
    )
