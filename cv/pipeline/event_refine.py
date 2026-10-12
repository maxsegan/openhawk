"""Refine a coarse event anchor to a continuous piecewise-polynomial knot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RefineResult:
    ok: bool
    tau: float = float("nan")
    x: float = float("nan")
    y: float = float("nan")
    cost: float = float("inf")
    rmse: float = float("inf")
    support_before: int = 0
    support_after: int = 0
    speed_before: float = float("nan")
    speed_after: float = float("nan")
    angle_change_deg: float = float("nan")
    velocity_jump: float = float("nan")
    vx_before: float = float("nan")
    vy_before: float = float("nan")
    vx_after: float = float("nan")
    vy_after: float = float("nan")
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tau": self.tau,
            "x540": self.x,
            "y540": self.y,
            "cost": self.cost,
            "rmse": self.rmse,
            "support_before": self.support_before,
            "support_after": self.support_after,
            "speed_before": self.speed_before,
            "speed_after": self.speed_after,
            "angle_change_deg": self.angle_change_deg,
            "velocity_jump": self.velocity_jump,
            "vx_before": self.vx_before,
            "vy_before": self.vy_before,
            "vx_after": self.vx_after,
            "vy_after": self.vy_after,
            "reason": self.reason,
        }


def _design(offsets: np.ndarray, degree: int) -> np.ndarray:
    before = offsets < 0
    columns = [np.ones_like(offsets)]
    for power in range(1, degree + 1):
        column = offsets**power
        columns.append(np.where(before, column, 0.0))
    for power in range(1, degree + 1):
        column = offsets**power
        columns.append(np.where(before, 0.0, column))
    return np.stack(columns, axis=1)


def fit_knot(
    times: np.ndarray,
    values_x: np.ndarray,
    values_y: np.ndarray,
    tau: float,
    degree: int = 1,
    exclude: float = 0.0,
    min_side: int = 2,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, int, int] | None:
    offsets = times - tau
    keep = np.abs(offsets) >= exclude
    if weights is not None:
        keep &= weights > 0
    before = int(np.sum(keep & (offsets < 0)))
    after = int(np.sum(keep & (offsets > 0)))
    if before < min_side or after < min_side:
        return None

    offsets_kept = offsets[keep]
    design = _design(offsets_kept, degree)
    if design.shape[0] < design.shape[1] + 1:
        return None

    target_x = values_x[keep]
    target_y = values_y[keep]
    if weights is not None:
        root = np.sqrt(weights[keep])[:, None]
        design_weighted = design * root
        target_x = target_x * root[:, 0]
        target_y = target_y * root[:, 0]
    else:
        design_weighted = design

    try:
        coefficients_x, *_ = np.linalg.lstsq(design_weighted, target_x, rcond=None)
        coefficients_y, *_ = np.linalg.lstsq(design_weighted, target_y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    residual_x = design @ coefficients_x - values_x[keep]
    residual_y = design @ coefficients_y - values_y[keep]
    cost = float(np.sum(residual_x**2 + residual_y**2))
    return cost, coefficients_x, coefficients_y, before, after


def refine_anchor(
    times: np.ndarray,
    values_x: np.ndarray,
    values_y: np.ndarray,
    hint: float,
    search: float = 3.0,
    window: float = 6.0,
    degree: int = 1,
    exclude: float = 0.0,
    min_side: int = 2,
    step: float = 0.1,
    robust: bool = True,
    huber_px: float = 4.0,
) -> RefineResult:
    """Scan knot times around a proposal while enforcing positional continuity."""
    times = np.asarray(times, dtype=float)
    values_x = np.asarray(values_x, dtype=float)
    values_y = np.asarray(values_y, dtype=float)

    local = np.abs(times - hint) <= window
    if int(np.sum(local)) < 2 * min_side + 1:
        return RefineResult(ok=False, reason="insufficient_window_support")
    times, values_x, values_y = times[local], values_x[local], values_y[local]

    order = np.argsort(times)
    times, values_x, values_y = times[order], values_x[order], values_y[order]
    grid = np.arange(hint - search, hint + search + 1e-9, step)
    weights = np.ones_like(times)

    best = None
    for _ in range(3 if robust else 1):
        best = None
        for tau in grid:
            fit = fit_knot(
                times,
                values_x,
                values_y,
                float(tau),
                degree=degree,
                exclude=exclude,
                min_side=min_side,
                weights=weights,
            )
            if fit is None:
                continue
            cost, coefficients_x, coefficients_y, before, after = fit
            normalized = cost / max(before + after - degree * 2 - 1, 1)
            if best is None or normalized < best[0]:
                best = (
                    normalized,
                    float(tau),
                    coefficients_x,
                    coefficients_y,
                    before,
                    after,
                )
        if best is None:
            return RefineResult(ok=False, reason="no_feasible_knot")
        if not robust:
            break

        _, tau, coefficients_x, coefficients_y, _, _ = best
        offsets = times - tau
        design = _design(offsets, degree)
        residual = np.hypot(
            design @ coefficients_x - values_x,
            design @ coefficients_y - values_y,
        )
        weights = np.where(
            residual <= huber_px,
            1.0,
            huber_px / np.maximum(residual, 1e-6),
        )

    normalized, tau, coefficients_x, coefficients_y, before, after = best
    vx_before = float(coefficients_x[1])
    vy_before = float(coefficients_y[1])
    vx_after = float(coefficients_x[1 + degree])
    vy_after = float(coefficients_y[1 + degree])
    speed_before = float(np.hypot(vx_before, vy_before))
    speed_after = float(np.hypot(vx_after, vy_after))
    angle_before = np.degrees(np.arctan2(vy_before, vx_before))
    angle_after = np.degrees(np.arctan2(vy_after, vx_after))
    angle_change = float(abs((angle_after - angle_before + 180.0) % 360.0 - 180.0))

    return RefineResult(
        ok=True,
        tau=tau,
        x=float(coefficients_x[0]),
        y=float(coefficients_y[0]),
        cost=float(normalized),
        rmse=float(np.sqrt(normalized)),
        support_before=int(before),
        support_after=int(after),
        speed_before=speed_before,
        speed_after=speed_after,
        angle_change_deg=angle_change,
        velocity_jump=float(np.hypot(vx_after - vx_before, vy_after - vy_before)),
        vx_before=vx_before,
        vy_before=vy_before,
        vx_after=vx_after,
        vy_after=vy_after,
        reason="ok",
    )
