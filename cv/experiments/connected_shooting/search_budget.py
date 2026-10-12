"""Cooperative deadlines and atomic same-invocation completed-fit checkpoints."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time  # noqa: F401 - the budget clock reads time.monotonic (tests patch it here)

import numpy as np
from cv.pipeline import s6_work_clock as work_clock


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, default=json_default, allow_nan=False))


# Stands in for the completed list while the rest of a checkpoint is encoded.
_COMPLETED_SLOT = "\x00s6_completed_candidates_slot\x00"


class Budget:
    def __init__(self, seconds: float | None, checkpoint: Path):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("positive finite search seconds required")
        self.seconds = seconds
        self.started = work_clock.now("search_budget")
        self.checkpoint = checkpoint
        self.completed: list[dict] = []
        self.context: dict = {}
        self.exhausted = False
        self.retain_incumbent = False
        self.deadline_incumbents = 0
        # Each recorded candidate's JSON, encoded once: every checkpoint rewrites the whole
        # list, which re-encoded all earlier candidates each time (quadratic). ``None``
        # once ``completed`` has been handed out (``persist`` from outside ``record``).
        self._encoded: list[str] | None = []

    def check(self) -> None:
        if (
            self.seconds is not None
            and work_clock.now("search_budget") - self.started >= self.seconds
        ):
            self.exhausted = True
            raise SearchDeadline("shared numerical search deadline exhausted")

    def receipt(self) -> dict:
        return {
            "seconds": self.seconds,
            "elapsed_seconds": work_clock.now("search_budget") - self.started,
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
        if self._encoded is not None:
            try:
                self._encoded.append(
                    json.dumps(self.completed[-1], default=json_default, allow_nan=False)
                )
            except (TypeError, ValueError):
                self._encoded = None  # the full encode below raises exactly as before
        self._write()

    def persist(self) -> None:
        # Callers may read and edit ``completed`` after a deadline: encode in full from now on.
        self._encoded = None
        self._write()

    def _write(self) -> None:
        from cv.experiments.connected_shooting import native_seed_check_pixels

        document = {
            **self.context,
            "schema": "s6_same_invocation_completed_candidates_v1",
            "completed_candidates": self.completed if self._encoded is None else _COMPLETED_SLOT,
            "search_budget": self.receipt(),
            **native_seed_check_pixels.selection_fields(self.completed),
        }
        if self._encoded is None:
            atomic_json(self.checkpoint, document)
            return
        # Byte-identical to ``atomic_json(document)`` with the list in place: default
        # separators join list items with ", ".
        text = json.dumps(document, default=json_default, allow_nan=False)
        slot = json.dumps(_COMPLETED_SLOT)
        if text.count(slot) != 1:
            document["completed_candidates"] = self.completed
            atomic_json(self.checkpoint, document)
            return
        _atomic_text(self.checkpoint, text.replace(slot, "[" + ", ".join(self._encoded) + "]"))
