"""Shared, torch-free contract for the event model's local timing distribution.

``EventVideoNet`` already computes a softmax over integer frame offsets inside
the crop window; only its mean and its max were ever persisted.  This module
owns the names, the validation and the JSON record of the full per-row
distribution so simple consumers (decoder, sidecars, evidence packets) can read
it without importing torch or the model.

Two facts travel with every record and must not be lost downstream:

* the scores are **uncalibrated** model outputs, not a calibrated probability
  that the event happened in a bin, and
* the offsets are relative to the row's own **native crop candidate frame**, not
  to any corrected or fitted event epoch.

The distribution is strictly additive: when a prediction artifact does not carry
it, consumers report it as unavailable.  A missing distribution is never
reconstructed from the persisted mean or max -- that would manufacture evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# The time head is a softmax over integer frame offsets inside the crop window.
# ``event_video_model`` re-exports these as ``TIME_RADIUS``/``TIME_BINS``; they
# live here so a consumer can check the exact model grid without loading torch.
TIME_RADIUS = 8
TIME_BINS = 2 * TIME_RADIUS + 1

SCHEMA = "event_time_distribution_v1"
SOURCE = "event_video_model_time_softmax"
# Row-aligned prediction-artifact arrays: ``(rows, TIME_BINS)`` each.  The grid
# is repeated per row rather than stored once so that the generic row-alignment
# and concatenation checks over prediction arrays keep working unchanged.
PMF_KEY = "time_pmf"
GRID_KEY = "time_pmf_offsets"
KEYS = (PMF_KEY, GRID_KEY)

AVAILABLE = "model_time_softmax_uncalibrated"
UNAVAILABLE = "unavailable_in_prediction_artifact"
OFFSET_REFERENCE = "native_crop_candidate_frame"
# float32 softmax rows sum to 1 well inside this; it only rejects malformed input.
SUM_TOLERANCE = 1e-5


def expected_grid() -> np.ndarray:
    """Return the exact model offset grid, in frames."""

    return np.arange(-TIME_RADIUS, TIME_RADIUS + 1, dtype=np.float32)


def present(data: Mapping[str, np.ndarray]) -> bool:
    """Return whether a prediction artifact carries the distribution.

    Presence is paired: half of the contract is a malformed artifact, not an
    older one.
    """

    found = [key for key in KEYS if key in data]
    if found and len(found) != len(KEYS):
        raise ValueError(
            f"time distribution needs both {list(KEYS)}; artifact carries only {found}"
        )
    return bool(found)


def validated(
    pmf: np.ndarray, grid: np.ndarray, *, rows: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Check shapes, row alignment, the offset grid and the normalised scores."""

    pmf = np.asarray(pmf, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if pmf.ndim != 2 or pmf.shape[1] != TIME_BINS:
        raise ValueError(f"time distribution must be (rows, {TIME_BINS}); got {pmf.shape}")
    if grid.shape != pmf.shape:
        raise ValueError("time offset grid must be row-aligned with the distribution")
    if rows is not None and len(pmf) != rows:
        raise ValueError(f"time distribution has {len(pmf)} rows, expected {rows}")
    if not np.isfinite(pmf).all() or np.any(pmf < 0.0):
        raise ValueError("time distribution must be finite and non-negative")
    if np.any(np.abs(pmf.sum(axis=1) - 1.0) > SUM_TOLERANCE):
        raise ValueError("time distribution rows must be normalised")
    if not np.isfinite(grid).all() or np.any(np.diff(grid, axis=1) <= 0.0):
        raise ValueError("time offset grid must be finite and strictly increasing")
    if len(grid) and not np.array_equal(grid, np.tile(expected_grid(), (len(grid), 1))):
        raise ValueError("time offset grid is not the exact model grid")
    return pmf, grid


def arrays(
    data: Mapping[str, np.ndarray], *, rows: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the validated ``(pmf, grid)`` pair, or ``None`` when unavailable."""

    if not present(data):
        return None
    return validated(data[PMF_KEY], data[GRID_KEY], rows=rows)


def record(pmf_row: np.ndarray, grid_row: np.ndarray) -> dict:
    """Return the JSON evidence block for one already validated row."""

    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "status": AVAILABLE,
        "calibrated": False,
        "offset_reference": OFFSET_REFERENCE,
        "offset_frames": [float(value) for value in np.asarray(grid_row)],
        "probabilities": [float(value) for value in np.asarray(pmf_row)],
    }


def status(data: Mapping[str, np.ndarray]) -> str:
    """Return the declared availability status for a prediction artifact."""

    return AVAILABLE if present(data) else UNAVAILABLE


__all__ = [
    "AVAILABLE",
    "GRID_KEY",
    "KEYS",
    "OFFSET_REFERENCE",
    "PMF_KEY",
    "SCHEMA",
    "SOURCE",
    "SUM_TOLERANCE",
    "TIME_BINS",
    "TIME_RADIUS",
    "UNAVAILABLE",
    "arrays",
    "expected_grid",
    "present",
    "record",
    "status",
    "validated",
]
