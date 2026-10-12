"""GPU search + CPU SLSQP certification for ``real_exposure_replay.refine`` (default off).

``TENNIS_S6_GPU_FIT=on`` replaces refine's primary SLSQP call with: the batched GPU
Levenberg-Marquardt from the recorded seed plus bounded perturbations (``..._SEEDS``, default
16, float32, ``..._ITERATIONS`` default 60), a float64 finish of the best start
(``..._FINISH64`` iterations, default 6), then, by ``..._POLISH``:

* ``all``: the unchanged SciPy SLSQP (same cost, bounds, constraints, options) started
  from the GPU point, so every returned iterate is a CPU SLSQP iterate;
* ``refined`` (default, the validated configuration): that SLSQP only for the search's refined
  stage; coarse (search-only) branches return the GPU point with a CPU-evaluated constraint check;
* ``none``: no SLSQP anywhere (research only).

``..._RESCUE=on`` (default off): if a polished point fails the CPU constraint check, the unchanged
SLSQP from refine's own start runs too and its answer is kept when feasible. On the parity
fixtures it removes the 3.2% new-infeasible class by construction; on end-to-end attempts it ran
7 times, was never kept (the CPU start was infeasible too) and cost up to ~37 CPU-s each.

Inside ``cooperative.run`` (``TENNIS_S6_GPU_FIT_BATCH``, default on with the switch, in the
whole-point search) the solves of concurrently waiting branches share one launch. Configurations the kernel does not model fall
back to the plain CPU solve (reason logged). ``..._LOG=DIR`` appends one JSON line per call to
``DIR/<pid>.jsonl``; ``..._DEVICE`` picks the card (index in PCI bus order among the visible
devices, default 0). Any GPU, build or unsupported-feature failure keeps the unchanged CPU solve.
Rollback: unset the variable (or set it to ``off``).
"""

from __future__ import annotations

import contextvars
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize as scipy_minimize

VARIABLE = "TENNIS_S6_GPU_FIT"
POLICIES = ("all", "refined", "none")
_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar("gpu_fit_stage", default=None)


def enabled() -> bool:
    value = os.environ.get(VARIABLE, "off")
    if value not in ("on", "off"):
        raise ValueError(f"{VARIABLE} must be on or off")
    return value == "on"


def batching() -> bool:
    return enabled() and os.environ.get(f"{VARIABLE}_BATCH", "on") == "on"


@contextmanager
def stage(name: str):
    """Declare the search stage (coarse/refined) of the solves inside this block."""
    token = _STAGE.set(name)
    try:
        yield
    finally:
        _STAGE.reset(token)


def _setting(name: str, default: str) -> str:
    return os.environ.get(f"{VARIABLE}_{name}", default)


def _log(row: dict) -> None:
    folder = os.environ.get(f"{VARIABLE}_LOG")
    if not folder:
        return
    path = Path(folder) / f"{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def solve_requests(requests: list[tuple]) -> list[dict]:
    """GPU answers for ``(problem, seeds)`` requests: float32 search of every start in one
    launch, then a float64 finish of each request's best start in a second launch."""
    from cv.experiments.gpu_fit import gpu_fit

    device = int(_setting("DEVICE", "0"))
    problems = np.array([problem for problem, _ in requests])
    index = np.concatenate(
        [np.full(len(seeds), k, np.int32) for k, (_, seeds) in enumerate(requests)]
    )
    starts = [seed for _, seeds in requests for seed in seeds]
    results, search_ms, _ = gpu_fit.solve(
        problems, index, starts, device=device, max_iter=int(_setting("ITERATIONS", "60"))
    )
    answers, best_rows = [], []
    for k in range(len(requests)):
        rows = results[index == k]
        usable = rows["status"] != -1
        if not usable.any():
            answers.append({"q": None})
            best_rows.append(None)
            continue
        best = int(np.argmin(np.where(usable, rows["cost_final"], np.inf)))
        answers.append({"q": rows["q"][best].copy(), "best_seed": best})
        best_rows.append(rows["q"][best].copy())
    finish = int(_setting("FINISH64", "6"))
    finish_ms = 0.0
    todo = [k for k, row in enumerate(best_rows) if row is not None]
    if finish and todo:
        finished, finish_ms, _ = gpu_fit.solve(
            problems,
            np.asarray(todo, np.int32),
            [best_rows[k] for k in todo],
            device=device,
            max_iter=0,
            max_iter64=finish,
        )
        for k, row in zip(todo, finished, strict=True):
            if row["status"] != -1:
                answers[k]["q"] = row["q"].copy()
            answers[k]["gpu_status"] = int(row["status"])
    for answer in answers:
        answer.update(batch=len(requests), search_ms=search_ms, finish_ms=finish_ms)
    return answers


def _gpu_gradients(problem, constraints, bounds, options, row):
    """SLSQP arguments whose finite-difference gradients run on the GPU (``..._GRAD``, default on).

    The objective's gradient uses SLSQP's own ``workers`` map; each constraint gets the ``jac``
    SLSQP would build (``approx_derivative`` 2-point, ``abs_step=eps``, same bounds) with a GPU
    map. Steps, bounds handling and the SLSQP iteration are SciPy's; only the P difference
    points are evaluated in one float64 launch (kernel parity ~1e-15 relative). A point the
    GPU cannot evaluate is evaluated by the CPU function, and so is every point of a launch
    that fails (a fit server that died mid-request): counted by ``gpu_fit.record_error``.

    One launch per iterate (``..._GRAD_FUSE``, default on): SLSQP asks for the objective gradient
    and then every constraint's normals at the same point, over the same free-coordinate
    stencil, so the objective's launch keeps each point's raw constraint values and the
    constraint maps read them back (bit-identical values, fewer float64 launches). Constraint
    columns of fixed variables, which ``minimize`` removes before SLSQP sees them, are zero.
    """
    if _setting("GRAD", "on") != "on" or problem is None:
        return options, constraints
    try:
        from scipy.optimize._constraints import old_bound_to_new
        from scipy.optimize._numdiff import approx_derivative
        from scipy.optimize._slsqp_py import _check_clip_x
    except ImportError:  # SciPy internals moved: keep the CPU gradients
        return options, constraints
    from cv.experiments.gpu_fit import gpu_fit

    epsilon = float(options.get("eps", np.sqrt(np.finfo(float).eps)))
    box = np.asarray(bounds, float)
    new_bounds = old_bound_to_new([tuple(pair) for pair in box])  # as _minimize_slsqp
    nraw = gpu_fit.raw_constraint_count(problem)
    stats = row.setdefault("gpu_grad", {"launches": 0, "points": 0, "cpu_points": 0})

    # scipy.optimize.minimize drops variables with equal bounds before SLSQP when it takes
    # finite differences, so the objective's map receives free-variable points.
    free = box[:, 0] != box[:, 1]

    def full(x):
        x = np.asarray(x, float)
        if len(x) == len(box):
            return x
        out = box[:, 0].copy()
        out[free] = x
        return out

    fuse = _setting("GRAD_FUSE", "on") == "on"
    stencil: dict[bytes, tuple] = {}  # last launch's points: full-point bytes -> (cost, ok, raw)

    def evaluate(points):
        points = np.array([full(x) for x in points])
        keys = [point.tobytes() for point in points]
        missing = [k for k, key in enumerate(keys) if key not in stencil] if fuse else None
        if missing is None or missing:
            todo = points if missing is None else points[missing]
            try:
                _, costs, ok, raw = gpu_fit.evaluate(problem, todo, raw=True)
            except (RuntimeError, OSError) as exc:  # fit server lost, GPU error: CPU, recorded
                gpu_fit.record_error("refine gradient", exc)
                stencil.clear()
                return (
                    np.full(len(points), np.nan),
                    np.zeros(len(points), bool),
                    np.zeros((len(points), nraw)),
                )
            stats["launches"] += 1
            stats["points"] += len(todo)
            if missing is None:
                return costs[:, 0], ok, raw
            stencil.clear()
            for k, cost, good, values in zip(missing, costs[:, 0], ok, raw, strict=True):
                stencil[keys[k]] = (cost, good, values)
        else:
            stats["fused_points"] = stats.get("fused_points", 0) + len(points)
        rows = [stencil[key] for key in keys]
        return (
            np.array([row[0] for row in rows]),
            np.array([row[1] for row in rows], bool),
            np.array([row[2] for row in rows]),
        )

    def objective_map(function, iterable):
        points = list(iterable)
        costs, ok, _ = evaluate(points)
        out = []
        for x, value, good in zip(points, costs, ok, strict=True):
            if not good:
                stats["cpu_points"] += 1
                value = function(x)
                stats["cpu_points_1e12"] = stats.get("cpu_points_1e12", 0) + int(
                    np.all(np.asarray(value) >= 1e12)
                )
            out.append(np.atleast_1d(np.asarray(value, float)))  # as approx_derivative wraps fun
        return out

    slices, offset = [], 0
    for k in range(len(constraints)):
        width = 3 if (k == 0 and problem["use_term_pen"]) else int(problem["nwin"])
        slices.append(slice(offset, offset + width))
        offset += width
    if offset != nraw:
        return {**options, "workers": objective_map}, constraints

    def constraint_jac(constraint, part):
        function = constraint["fun"]

        def mapper(wrapped, iterable):
            points = list(iterable)
            _, ok, raw = evaluate(points)
            out = []
            for x, values, good in zip(points, raw, ok, strict=True):
                if not good:
                    stats["cpu_points"] += 1
                    out.append(np.atleast_1d(wrapped(x)))
                else:
                    out.append(np.asarray(values[part], float))
            return out

        def jac(x, *args):
            x = _check_clip_x(x, new_bounds)
            if not fuse or free.all():
                return approx_derivative(
                    function,
                    x,
                    method="2-point",
                    abs_step=epsilon,
                    args=args,
                    bounds=new_bounds,
                    workers=mapper,
                )

            # Free coordinates only: the same per-coordinate steps as the full stencil
            # (approx_derivative treats coordinates independently), matching the objective's.
            def restricted(z, *args):
                point = x.copy()
                point[free] = z
                return function(point, *args)

            columns = approx_derivative(
                restricted,
                x[free],
                method="2-point",
                abs_step=epsilon,
                args=args,
                bounds=(np.asarray(new_bounds[0])[free], np.asarray(new_bounds[1])[free]),
                workers=mapper,
            )
            out = np.zeros((columns.shape[0], len(x)))
            out[:, free] = columns
            return out

        return jac

    wrapped = [
        {**c, "jac": constraint_jac(c, part)} if "jac" not in c else c
        for c, part in zip(constraints, slices, strict=True)
    ]
    return {**options, "workers": objective_map}, wrapped


def _feasible(constraints, x, tolerance=1e-9) -> bool:
    return all(np.all(np.asarray(c["fun"](x), float) >= -tolerance) for c in constraints)


def _unpolished(fun, x, constraints, answer) -> SimpleNamespace:
    """The GPU point as refine's optimizer result, judged by the CPU constraint functions."""
    feasible = all(np.all(np.asarray(c["fun"](x), float) >= 0) for c in constraints)
    converged = answer.get("gpu_status") == 0
    success = bool(feasible and converged)
    return SimpleNamespace(
        x=x,
        fun=float(fun(x)),
        success=success,
        status=0 if success else (4 if not feasible else 9),
        message=(
            "GPU Levenberg-Marquardt point, no SLSQP polish"
            + ("" if feasible else "; CPU constraint check failed")
            + ("" if converged else "; GPU iteration limit")
        ),
        nit=0,
    )


def solver(call: dict):
    """A ``scipy.optimize.minimize`` stand-in bound to one refine call's named arguments."""

    def minimize(fun, x0, *, method, bounds, constraints, options):
        from cv.experiments.gpu_fit import cooperative, gpu_fit

        if method != "SLSQP":
            raise ValueError("the GPU refine hook replaces SLSQP only")
        policy = _setting("POLISH", "refined")
        if policy not in POLICIES:
            raise ValueError(f"{VARIABLE}_POLISH must be one of {POLICIES}")
        started = time.perf_counter()
        x0 = np.asarray(x0, float)
        box = np.asarray(bounds, float)
        lo, hi = box[:, 0], box[:, 1]
        cpu = gpu_fit.CpuProblem(fun, None, list(constraints), x0, lo, hi, dict(options))
        row = {"pid": os.getpid(), "parameters": len(x0), "stage": _STAGE.get(), "policy": policy}
        start, answer = x0, None
        try:
            problem = gpu_fit.pack_call(call, {}, lambda: cpu)
        except gpu_fit.Unsupported as exc:
            row.update(route="cpu", reason=str(exc))
            problem = None
        if problem is not None:
            count = int(_setting("SEEDS", "16"))
            seeds = gpu_fit.multistart(x0, lo, hi, count, np.random.default_rng(0))
            try:
                answer = cooperative.request(
                    (problem, seeds), lambda request: solve_requests([request])[0]
                )
            except (
                RuntimeError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:  # GPU unavailable: CPU solve, counted and warned
                gpu_fit.record_error("refine search", exc)
                row.update(route="cpu", reason=f"gpu_error: {exc}"[:300])
                problem, answer = None, None
        if answer is not None:
            row.update(route="gpu", gpu_seconds=time.perf_counter() - started)
            row.update({k: v for k, v in answer.items() if k != "q"})
            if answer["q"] is not None:
                candidate = np.clip(answer["q"][: len(x0)], lo, hi)
                seed_cost, candidate_cost = float(fun(x0)), float(fun(candidate))
                if candidate_cost <= seed_cost:
                    start = candidate
                row.update(seed_cost=seed_cost, gpu_cost=candidate_cost)
        polish = (
            problem is None or policy == "all" or (policy == "refined" and _STAGE.get() != "coarse")
        )
        polish_started, cpu_started = time.perf_counter(), time.process_time()
        calls = {"n": 0}

        def counted(q):
            calls["n"] += 1
            return fun(q)

        result = None
        try:
            if polish or answer is None or answer["q"] is None:
                grad_options, grad_constraints = _gpu_gradients(
                    problem, constraints, bounds, options, row
                )
                result = scipy_minimize(
                    counted,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=grad_constraints,
                    options=grad_options,
                )
            else:
                result = _unpolished(counted, start, constraints, answer)
            rescue_on = _setting("RESCUE", "off") == "on"
            if rescue_on and polish and start is not x0 and not _feasible(constraints, result.x):
                # Feasibility rescue: a polished GPU start that ends outside the constraints (the
                # terminal slack, ~3% of solves) also gets the unchanged CPU SLSQP from refine's
                # own start. Coarse (unpolished) rows keep the search's own ranking of failures.
                grad_options, grad_constraints = _gpu_gradients(
                    problem, constraints, bounds, options, row
                )
                rescue = scipy_minimize(
                    counted,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=grad_constraints,
                    options=grad_options,
                )
                row["rescue"] = bool(_feasible(constraints, rescue.x))
                if row["rescue"]:
                    result = rescue
        finally:
            row.update(
                polished=bool(polish),
                polish_seconds=time.perf_counter() - polish_started,
                polish_cpu_seconds=time.process_time() - cpu_started,
                polish_cost_calls=calls["n"],
            )
            if result is not None:
                row.update(
                    polish_nit=int(result.nit),
                    polish_status=int(result.status),
                    final_cost=float(result.fun),
                )
            _log(row)
        return result

    return minimize
