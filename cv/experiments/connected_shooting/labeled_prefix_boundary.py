"""Optional local prefix objective with an exact connected boundary and net constraints.

The boundary is the current invocation's physical state, not observed XYZ truth.
Later trajectories are replayed, never spliced. This experimental optimizer uses
no scoring gates and reports infeasibility as an execution limitation.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from cv.experiments.connected_shooting.physical_compatibility import crossings

TOLERANCE_M = 1e-6


class _InvalidDerivativeProbe(Exception):
    """A numerical-domain failure, never a residual or physical constraint."""


class _DerivativeUnavailable(Exception):
    """No evaluable stencil: retain the certified incumbent without convergence."""


def _domain_jacobian(fun, x, lo, hi, statistics):
    """Retry invalid shared 1e-6 stencils at smaller absolute steps.

    Only numerical-domain failures trigger recovery. Finite infeasible physical
    constraints still participate normally. If central probes remain impossible,
    use an evaluated side and base; never invent a zero derivative.
    """
    try:
        base = fun(x)
    except _InvalidDerivativeProbe as error:
        raise _DerivativeUnavailable("optimizer base is outside the numerical domain") from error
    columns = []
    for i in range(x.size):
        if lo[i] == hi[i]:
            columns.append(np.zeros_like(base))
            continue
        one_sided = None
        for attempt, step in enumerate((1e-6, 1e-7, 1e-8, 1e-9)):
            xp, xm = x.copy(), x.copy()
            xp[i] = min(x[i] + step, hi[i])
            xm[i] = max(x[i] - step, lo[i])
            if xp[i] == xm[i]:
                continue  # No representable separation is not a fixed coordinate.
            values = []
            for q in (xp, xm):
                try:
                    values.append(fun(q))
                except _InvalidDerivativeProbe:
                    values.append(None)
            plus, minus = values
            if plus is not None and minus is not None:
                columns.append((plus - minus) / (xp[i] - xm[i]))
                statistics["smaller_stencils"] += int(attempt > 0)
                break
            if one_sided is None:
                for q, value in ((xp, plus), (xm, minus)):
                    if value is not None and q[i] != x[i]:
                        one_sided = (value - base) / (q[i] - x[i])
                        break
        else:
            if one_sided is None:
                raise _DerivativeUnavailable(f"no valid derivative for coordinate {i}")
            columns.append(one_sided)
            statistics["one_sided_stencils"] += 1
            continue
    result = np.stack(columns, axis=1)
    if not np.isfinite(result).all():
        raise _DerivativeUnavailable("nonfinite derivative")
    return result


def feasible_receipt(receipt):
    """Keep spatial tolerances separate from strict original event intervals."""
    eq = np.asarray(receipt["boundary_equality_m"])
    slack = np.asarray(receipt["clearance_slacks_m"])
    events = np.asarray(receipt.get("input_event_slacks_frames", []))
    return bool(
        all(np.isfinite(v).all() for v in (eq, slack, events))
        and np.max(np.abs(eq), initial=0) <= TOLERANCE_M
        and np.min(slack, initial=1) >= -TOLERANCE_M
        and np.min(events, initial=1) >= -float(receipt.get("input_event_tolerance_frames", 0.0))
    )


def constraints(chain, queries, movable, declared_nets, boundary):
    """Fixed-size constraints, accounting for every in-width plane crossing."""
    equality = (
        np.asarray(chain[movable - 1]["end_xyz"], float) - boundary
        if boundary is not None
        else np.empty(0)
    )
    slacks = []
    for i in range(movable):
        if len(declared_nets[i]):
            continue
        rows = crossings(queries[i], np.asarray(chain[i]["positions"], float))
        slacks.append(
            min(
                (r["ball_surface_clearance_m"] for r in rows if r["within_net_width"]),
                default=1.0,
            )
        )
    return equality, np.asarray(slacks, float)


def local_net_residuals(scene, queries, chain, movable):
    """Retain every collision component, selected by flight rather than scalar index."""
    from cv.experiments.connected_shooting import net_constraints
    from cv.experiments.connected_shooting.labeled_terminal_net_tail import collision_residuals

    values = []
    for i in range(movable):
        declared = () if scene.net_hit_frames is None else scene.net_hit_frames[i]
        if len(declared):
            values.extend(
                collision_residuals(scene, i, chain[i].get("net_hits", []), float(declared[0]))
            )
        else:
            values.append(net_constraints.penalty(queries[i], chain[i]["positions"], 0.025)[0])
    return np.asarray(values, float)


def solve(
    evaluate,
    q0,
    lo,
    hi,
    *,
    jacobian,
    maxiter,
    seconds,
    started,
    feasible=None,
    inequality_key="clearance_slacks_m",
    certify_incumbent=None,
):
    """Select input-objective minimum among feasible trials, never by score."""
    cache = OrderedDict()
    state = dict(calls=0, invalid=0, feasible_trials=0, best=None)
    derivative_statistics = dict(
        invalid_probes=0,
        fallback_jacobians=0,
        smaller_stencils=0,
        one_sided_stencils=0,
        unavailable=0,
    )
    if certify_incumbent is not None:
        state.update(incumbent_certifications=0, incumbent_certification_refusals=0)
    restoration_seed = None

    class Timeout(Exception):
        pass

    def get(q):
        if time.monotonic() - started > seconds:
            raise Timeout()
        key = np.asarray(q, float).tobytes()
        if key in cache:
            return cache[key]
        state["calls"] += 1
        try:
            r, receipt = evaluate(q)
            eq = receipt["boundary_equality_m"]
            slack = receipt[inequality_key]
            if not all(np.isfinite(v).all() for v in (r, eq, slack)):
                raise ValueError("nonfinite trial")
            cost = float(r @ r)
            admissible = feasible(receipt) if feasible is not None else feasible_receipt(receipt)
            if admissible:
                state["feasible_trials"] += 1
                if state["best"] is None or cost < state["best"][0]:
                    certified = (r, receipt)
                    if certify_incumbent is not None:
                        state["incumbent_certifications"] += 1
                        certified = certify_incumbent(q, r, receipt)
                        if certified is None:
                            state["incumbent_certification_refusals"] += 1
                    if certified is not None:
                        full_r, full_receipt = certified
                        if not np.array_equal(r, full_r):
                            raise RuntimeError("local/full incumbent objective differs")
                        state["best"] = (cost, q.copy(), full_receipt)
            item = (r, receipt, eq, slack)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            state["invalid"] += 1
            if not cache:
                raise ValueError("initial boundary chart could not be evaluated")
            r, _, eq, slack = next(iter(cache.values()))
            item = (np.full_like(r, 1e6), None, np.full_like(eq, 1e6), np.full_like(slack, -1e6))
        cache[key] = item
        if len(cache) > 48:
            cache.popitem(last=False)
        return item

    try:
        initial = get(q0)
    except Timeout as error:
        raise ValueError(
            f"boundary optimizer wall timeout before initial evaluation; seconds={seconds}"
        ) from error

    derivative_cache = OrderedDict()
    ends = np.cumsum([len(initial[i]) for i in (0, 2, 3)])
    slices = {0: slice(0, ends[0]), 2: slice(ends[0], ends[1]), 3: slice(ends[1], ends[2])}

    def derivative(index, q):
        def checked(point):
            item = get(point)
            if item[1] is None:
                derivative_statistics["invalid_probes"] += 1
                raise _InvalidDerivativeProbe()
            return np.concatenate([item[i] for i in (0, 2, 3)])

        key = np.asarray(q, float).tobytes()
        try:
            checked(q)  # Checks the wall cap and the optimizer base even on cache hits.
            if key in derivative_cache:
                return derivative_cache[key][slices[index]]
            try:
                # Concatenating residual/constraint rows preserves every original
                # stencil subtraction, and computes the common physics only once.
                result = jacobian(checked, q, lo, hi)
            except _InvalidDerivativeProbe:
                derivative_statistics["fallback_jacobians"] += 1
                result = _domain_jacobian(checked, q, lo, hi, derivative_statistics)
        except _InvalidDerivativeProbe as error:
            raise _DerivativeUnavailable(
                "optimizer base is outside the numerical domain"
            ) from error
        derivative_cache[key] = result
        if len(derivative_cache) > 2:
            derivative_cache.popitem(last=False)
        return result[slices[index]]

    cons = []
    for index, kind in ((2, "eq"), (3, "ineq")):
        if len(initial[index]):

            def fun(q, index=index):
                return get(q)[index]

            cons.append(dict(type=kind, fun=fun, jac=lambda q, index=index: derivative(index, q)))

    def residual(q):
        return get(q)[0]

    def retain_major(q):
        """Keep one objective-improving major iterate closest to boundary closure."""
        nonlocal restoration_seed
        if np.any(q < lo) or np.any(q > hi):
            return
        r, receipt, eq, _ = get(q)
        if receipt is None or not len(eq):
            return
        admissible = feasible or feasible_receipt
        if admissible(receipt) or not admissible(
            receipt | {"boundary_equality_m": np.zeros_like(eq)}
        ):
            return
        cost = float(r @ r)
        if state["best"] is not None and cost >= state["best"][0]:
            return
        rank = (float(np.max(np.abs(eq))), cost)
        if restoration_seed is None or rank < restoration_seed[0]:
            restoration_seed = (rank, np.asarray(q, float).copy())

    def restore_major():
        """Close a retained major iterate under the same wall cap and admission.

        Four bounded Newton corrections address numerical boundary closure only;
        they do not run another optimizer or change the objective/physical domain.
        Every trial still goes through get(), including full incumbent certification.
        A custom feasible predicate must read boundary_equality_m for its closure
        test; zeroing only that field leaves every other admission check intact.
        """
        evidence = dict(
            status="not_needed",
            candidate_limit=1,
            candidate_count=int(restoration_seed is not None),
            max_corrections=min(4, maxiter),
            max_backtracks=8,
            shares_original_wall_budget=True,
            initial_evaluations=state["calls"],
            initial_best_cost=None if state["best"] is None else state["best"][0],
            corrections=[],
        )
        state["major_boundary_restoration"] = evidence
        if restoration_seed is None:
            return
        rank, q = restoration_seed
        if state["best"] is not None and rank[1] >= state["best"][0]:
            evidence["status"] = "superseded"
            return
        evidence.update(status="attempted", source_boundary_error_m=rank[0], source_cost=rank[1])
        admissible = feasible or feasible_receipt
        for _ in range(min(4, maxiter)):
            _, receipt, eq, _ = get(q)
            if receipt is None:
                evidence["status"] = "outside_domain"
                return
            if admissible(receipt):
                evidence["status"] = "closed"
                return
            jac = derivative(2, q)
            frozen = set(np.flatnonzero(lo == hi))
            direction = np.zeros_like(q)
            for _ in range(len(q) + 1):
                free = [i for i in range(len(q)) if i not in frozen]
                direction[:] = 0
                direction[free] = np.linalg.lstsq(jac[:, free], -eq, rcond=1e-10)[0]
                tolerance = 1e-12 * np.maximum(1.0, np.abs(q))
                blocked = {
                    i
                    for i in free
                    if (q[i] - lo[i] <= tolerance[i] and direction[i] < 0)
                    or (hi[i] - q[i] <= tolerance[i] and direction[i] > 0)
                }
                if not blocked:
                    break
                frozen.update(blocked)
            limits = [
                (hi[i] - q[i]) / value if value > 0 else (lo[i] - q[i]) / value
                for i, value in enumerate(direction)
                if value != 0
            ]
            alpha = max(0.0, min([1.0, *limits]))
            previous_norm = float(np.linalg.norm(eq))
            accepted = False
            for _ in range(8):
                trial = np.clip(q + alpha * direction, lo, hi)
                if np.array_equal(trial, q):
                    break
                _, tested, error, _ = get(trial)
                if (
                    tested is not None
                    and admissible(tested | {"boundary_equality_m": np.zeros_like(error)})
                    and np.linalg.norm(error) < previous_norm
                ):
                    q, accepted = trial, True
                    evidence["corrections"].append(float(np.max(np.abs(error))))
                    break
                alpha *= 0.5
            if not accepted:
                evidence["status"] = "no_admissible_correction"
                return
        _, receipt, _, _ = get(q)
        evidence["status"] = (
            "closed" if receipt is not None and admissible(receipt) else "bounded_out"
        )

    objective_scale = max(float(initial[0] @ initial[0]), 1.0)
    state["objective_scale"] = objective_scale
    state["initial_boundary_error_m"] = initial[2].tolist()
    state["initial_clearance_slacks_m"] = np.asarray(initial[1]["clearance_slacks_m"]).tolist()
    state["initial_boundary_jacobian_rank"] = None
    try:
        if len(initial[2]):
            state["initial_boundary_jacobian_rank"] = int(np.linalg.matrix_rank(derivative(2, q0)))
        solution = minimize(
            lambda q: float(residual(q) @ residual(q)) / objective_scale,
            q0,
            jac=lambda q: 2.0 * derivative(0, q).T @ residual(q) / objective_scale,
            method="SLSQP",
            bounds=list(zip(lo, hi)),
            constraints=cons,
            **({"callback": retain_major} if len(initial[2]) else {}),
            options=dict(maxiter=maxiter, ftol=1e-7),
        )
        termination = dict(
            kind="optimizer",
            status=int(solution.status),
            message=str(solution.message),
            nfev=int(solution.nfev),
            nit=int(solution.nit),
        )
        if len(initial[2]):
            try:
                restore_major()
            except Timeout:
                state["major_boundary_restoration"]["status"] = "wall_timeout"
            except (
                _DerivativeUnavailable,
                ValueError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                state["major_boundary_restoration"].update(
                    status="numerical_refusal", reason=f"{type(error).__name__}: {error}"
                )
    except Timeout:
        termination = dict(kind="wall_timeout", seconds=seconds)
    except _DerivativeUnavailable as error:
        derivative_statistics["unavailable"] += 1
        termination = dict(kind="derivative_unavailable", reason=str(error))
    termination["numerical_derivatives"] = derivative_statistics
    restoration = state.get("major_boundary_restoration")
    if restoration is not None:
        restoration["evaluations"] = state["calls"] - restoration.pop("initial_evaluations")
        previous_cost = restoration.pop("initial_best_cost")
        restoration["promoted"] = bool(
            state["best"] is not None
            and (previous_cost is None or state["best"][0] < previous_cost)
        )
        termination["major_boundary_restoration"] = restoration
    if state["best"] is None:
        raise ValueError(
            f"boundary/net constrained optimizer found no feasible trial: {termination}; calls={state['calls']}, invalid={state['invalid']}"
        )
    cost, q, receipt = state.pop("best")
    return dict(best_cost=cost, best_q=q, best_receipt=receipt, **state), termination
