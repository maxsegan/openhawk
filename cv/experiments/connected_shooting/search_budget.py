"""Cooperative deadlines and atomic same-invocation completed-fit checkpoints."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np


class SearchDeadline(RuntimeError):
    """Stop numerical search; already completed candidates remain usable."""


# Distinct from every SLSQP exit mode (0..9): the returned vector is the best
# finite iterate the optimizer evaluated before the deadline, not an optimum.
SEARCH_DEADLINE_INCUMBENT_STATUS = 100


class Incumbent:
    """Best finite in-bounds iterate under the optimizer's own objective.

    Same invocation, same input objective, same bounds.  It reads no truth, no
    gate, no cached fit and no other point.  A vector is retained only when it
    strictly improves the initial cost, so a deadline that arrives before any
    improving evaluation leaves nothing to return.
    """

    def __init__(self, initial_cost: float, lower: np.ndarray, upper: np.ndarray):
        if not np.isfinite(initial_cost):
            raise ValueError("finite initial objective required")
        self.initial_cost = float(initial_cost)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.vector: np.ndarray | None = None
        self.cost: float | None = None
        self.evaluations = 0
        self.improving_evaluations = 0

    def offer(self, vector, cost) -> None:
        self.evaluations += 1
        values = np.asarray(vector, float)
        if (
            values.shape != self.lower.shape
            or not np.isfinite(values).all()
            or not np.isfinite(cost)
            or np.any(values < self.lower)
            or np.any(values > self.upper)
            or cost >= self.initial_cost
        ):
            return
        self.improving_evaluations += 1
        if self.cost is None or cost < self.cost:
            self.vector = values.copy()
            self.cost = float(cost)

    @property
    def improved(self) -> bool:
        return self.vector is not None

    def receipt(self) -> dict:
        return {
            "returned_iterate": "search_deadline_incumbent",
            "converged": False,
            "initial_cost": self.initial_cost,
            "incumbent_cost": self.cost,
            "objective_evaluations": self.evaluations,
            "improving_evaluations": self.improving_evaluations,
            "selection": "lowest finite in-bounds optimizer objective this invocation",
            "iterations_unknown": True,
        }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)


class Budget:
    def __init__(self, seconds: float | None, checkpoint: Path):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("positive finite search seconds required")
        self.seconds = seconds
        self.started = time.monotonic()
        self.checkpoint = checkpoint
        self.completed: list[dict] = []
        self.context: dict = {}
        self.exhausted = False
        self.retain_incumbent = False
        self.deadline_incumbents = 0

    def check(self) -> None:
        if self.seconds is not None and time.monotonic() - self.started >= self.seconds:
            self.exhausted = True
            raise SearchDeadline("shared numerical search deadline exhausted")

    def receipt(self) -> dict:
        return {
            "seconds": self.seconds,
            "elapsed_seconds": time.monotonic() - self.started,
            "exhausted": self.exhausted,
            "completed_candidates": len(self.completed),
            "selection": "input objective, depth, completion order; acceptance is never read",
            "external_fitted_seed": False,
            "deadline_incumbent_retention": self.retain_incumbent,
            "deadline_incumbent_candidates": self.deadline_incumbents,
        }

    def record(self, candidate: dict, flights: int) -> None:
        if self.seconds is None:
            return
        parameters = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
        score = candidate["evidence"]["input_only_rank_score"]
        if parameters.shape != (5 + 6 * flights,) or not np.isfinite(parameters).all():
            raise ValueError("completed checkpoint requires a finite complete-scene vector")
        if not np.isfinite(score) or not candidate["measurement"].get("native_projection"):
            raise ValueError("completed checkpoint requires input rank and native projections")
        if candidate["measurement"]["fit"].get("search_deadline_incumbent"):
            self.deadline_incumbents += 1
        self.completed.append(deepcopy(candidate))
        self.persist()

    def persist(self) -> None:
        from cv.experiments.connected_shooting import native_seed_check_pixels

        atomic_json(
            self.checkpoint,
            {
                **self.context,
                "schema": "s6_same_invocation_completed_candidates_v1",
                "completed_candidates": self.completed,
                "search_budget": self.receipt(),
                **native_seed_check_pixels.selection_fields(self.completed),
            },
        )
