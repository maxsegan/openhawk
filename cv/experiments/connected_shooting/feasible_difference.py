"""Finite differences within an explicitly checked nonlinear residual domain.

Research helper: it does not replace final physical validation, relax bounds,
project invalid parameters, or claim that a joint optimizer step is feasible.
"""

from __future__ import annotations

import numpy as np


class UnsupportedDifference(ValueError):
    """Neither bounded side supplies a valid finite-difference coordinate."""


def feasible_jacobian(
    objective, values, lower, upper, *, rejected_errors=(ValueError,), halvings=6
):
    """Use a verified one-sided derivative or refuse the unsupported column.

    Match the ordinary two-point relative step first. On domain failure try the
    opposite direction, then halve both sides. Never differentiate an artificial
    rejection residual. The caller must expose domain failures as exceptions.
    """
    values = np.asarray(values, dtype=float)
    lower = np.broadcast_to(np.asarray(lower, dtype=float), values.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), values.shape)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < lower)
        or np.any(values > upper)
        or not isinstance(halvings, int)
        or halvings < 0
    ):
        raise ValueError("finite in-bounds vector and nonnegative integer halvings required")
    base = np.asarray(objective(values), dtype=float)
    if base.ndim != 1 or not np.isfinite(base).all():
        raise ValueError("finite residual vector required at current iterate")
    jacobian = np.empty((len(base), len(values)))
    receipt = []
    for column in range(len(values)):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(values[column]))
        if values[column] < 0:
            step = -step
        if not lower[column] <= values[column] + step <= upper[column]:
            step = -step
        attempts = []
        accepted = False
        for shrink in range(halvings + 1):
            for sign in (1, -1):
                trial = values.copy()
                trial[column] += sign * step * 0.5**shrink
                delta = trial[column] - values[column]
                if delta == 0 or not lower[column] <= trial[column] <= upper[column]:
                    continue
                try:
                    result = np.asarray(objective(trial), dtype=float)
                    if result.shape != base.shape or not np.isfinite(result).all():
                        raise ValueError("invalid finite-difference residual")
                except rejected_errors as error:
                    attempts.append({"delta": float(delta), "error": str(error)})
                    continue
                jacobian[:, column] = (result - base) / delta
                receipt.append(
                    {
                        "column": column,
                        "delta": float(delta),
                        "halvings": shrink,
                        "opposite": sign == -1,
                        "rejected_trials": attempts,
                    }
                )
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            raise UnsupportedDifference(f"no verified finite difference for coordinate {column}")
    return jacobian, receipt
