"""GPU local solves for the anchored seed walk (``initialization.anchor_connected_seed``).

Default off; needs ``TENNIS_S6_GPU_FIT=on`` and ``TENNIS_S6_GPU_FIT_SEED=on``. Each flight of
the walk solves its velocity/spin from ~3 guesses with ``least_squares`` (soft-L1, TRF). Here
all of a flight's prepared guesses are solved in one GPU launch (float32 Levenberg-Marquardt on
the same residual, ported as kernel kind 1) and each result stands in for that guess's
``least_squares`` result; the walk's own CPU residual still scores and selects the guesses.
It is a seed for the unchanged refine/SLSQP that follows, not an accepted fit. Anything the
kernel does not model returns None and the CPU ``least_squares`` runs as before.

``walk=True`` serves ``agent_whole_point_search.structure_aware_connected_seed`` the same way
(kernel kind 1 with swept-exposure image rows, ``gpu_fit.pack_walk``); its switch is
``TENNIS_S6_GPU_FIT_WALK`` (default on with the GPU fit).
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import numpy as np

from cv.experiments.gpu_fit import refine_hook


def enabled() -> bool:
    return refine_hook.enabled() and os.environ.get("TENNIS_S6_GPU_FIT_SEED", "on") == "on"


def walk_enabled() -> bool:
    return refine_hook.enabled() and os.environ.get("TENNIS_S6_GPU_FIT_WALK", "on") == "on"


def solver(residual, lower, upper, prepared_guesses, *, walk=False):
    """``solve(index, guess)`` -> least_squares-like result or None (CPU fallback)."""
    from cv.experiments.gpu_fit import gpu_fit

    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    try:
        problem = (gpu_fit.pack_walk if walk else gpu_fit.pack_seed)(residual, lower, upper)
    except gpu_fit.Unsupported as exc:
        refine_hook._log(
            {
                "pid": os.getpid(),
                "stage": "walk" if walk else "seed",
                "route": "cpu",
                "reason": str(exc),
            }
        )
        return lambda index, guess: None
    parity_folder = os.environ.get("TENNIS_S6_GPU_FIT_SEED_PARITY")
    if parity_folder:
        _parity(parity_folder, problem, residual, prepared_guesses, walk)
    cache: dict[int, SimpleNamespace | None] = {}
    iterations = int(os.environ.get("TENNIS_S6_GPU_FIT_SEED_ITERATIONS", "40"))
    device = int(os.environ.get("TENNIS_S6_GPU_FIT_DEVICE", "0"))

    def launch(pairs):
        starts = [
            np.clip(np.asarray(g, float) / gpu_fit.SEED_SCALE, problem["lo"][:6], problem["hi"][:6])
            for _, g in pairs
        ]
        results, _, _ = gpu_fit.solve(
            problem[None],
            np.zeros(len(starts), np.int32),
            starts,
            device=device,
            max_iter=iterations,
        )
        for (index, _), row in zip(pairs, results, strict=True):
            cache[index] = (
                None
                if row["status"] == -1
                else SimpleNamespace(
                    x=np.clip(row["q"][:6] * gpu_fit.SEED_SCALE, lower, upper),
                    success=bool(row["status"] == 0),
                    status=int(row["status"]),
                    nfev=int(row["evaluations"]),
                    gpu_iterations=int(row["iterations"]),
                )
            )

    def solve(index, guess):
        if index not in cache:
            pending = [(k, g) for k, g in enumerate(prepared_guesses) if k not in cache]
            if index >= len(prepared_guesses):
                pending = [(index, guess)]
            try:
                launch(pending)
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:  # GPU unavailable
                gpu_fit.record_error("seed solve", exc)  # CPU path, counted and warned
                for k, _ in pending:
                    cache[k] = None
        return cache.get(index)

    return solve


def _parity(folder, problem, residual, guesses, walk=False) -> None:
    """Diagnostic: float64 GPU seed residual vs the CPU residual at each prepared guess."""
    import json
    from pathlib import Path

    from cv.experiments.gpu_fit import gpu_fit

    qs = np.array([np.asarray(g, float) / gpu_fit.SEED_SCALE for g in guesses])
    rows, _, ok = gpu_fit.evaluate(problem, qs, precision=64)
    out = []
    for k, guess in enumerate(guesses):
        reference = np.asarray(residual(np.asarray(guess, float)), float)
        same_len = len(reference) == int(problem["M"])
        failed = bool(np.all(reference == 1e5))
        delta = np.abs(reference - rows[k][: len(reference)]) if same_len and ok[k] else None
        diff = float(np.max(delta)) if delta is not None and not failed else None
        out.append(
            {
                "walk": walk,
                "cpu_failed": failed,
                "gpu_ok": bool(ok[k]),
                "rows": [len(reference), int(problem["M"])],
                "max_abs_diff": diff,
                "argmax_row": None if diff is None else int(np.argmax(delta)),
                "nobs": int(problem["nobs"]),
                "n_exp": int(problem["n_exp"][0]),
                "has_target": int(problem["has_init_target"]),
                "guess": np.asarray(guess, float).tolist(),
                "cpu_row": None if diff is None else float(reference[int(np.argmax(delta))]),
                "gpu_row": None if diff is None else float(rows[k][int(np.argmax(delta))]),
            }
        )
    path = Path(folder) / f"{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(out) + "\n")
