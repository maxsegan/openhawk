"""Did a whole-source reader reach the end of its source? (coverage guard)

Two silent truncations lost most of a match on 2026-10-03 (point-gap): serve audio stopped at
a 6 s audio hole 30 min into a 116 min broadcast, and NVDEC decodes stopped at
a picture presented out of order, both with exit 0. Every whole-source reader now records the
source time it reached against the time it was asked for, and a reader that ends more than
``max(30 s, 1 %)`` short is reported.

Switch ``TENNIS_COVERAGE_GUARD``: ``record`` (default) writes a loud fallback record
(``cv.pipeline.switch_fallbacks``) and the stage keeps its partial output with
``complete: false`` in its receipt; ``fail`` raises instead; ``off`` checks nothing (rollback).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SWITCH = "TENNIS_COVERAGE_GUARD"
MIN_TOLERANCE_SECONDS = 30.0
TOLERANCE_SHARE = 0.01


class CoverageError(RuntimeError):
    """A whole-source reader ended short of its source (``TENNIS_COVERAGE_GUARD=fail``)."""


def mode() -> str:
    value = os.environ.get(SWITCH, "record").strip().lower()
    if value not in ("record", "fail", "off"):
        raise ValueError(f"{SWITCH} must be record, fail or off, not {value!r}")
    return value


def source_seconds(video: str | Path) -> float:
    """The container's duration (what a whole-source read should reach)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
        + [str(video)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def tolerance(expected_seconds: float) -> float:
    return max(MIN_TOLERANCE_SECONDS, TOLERANCE_SHARE * expected_seconds)


def check(
    reader: str, reached_seconds: float, expected_seconds: float, *, where: str = ""
) -> dict | None:
    """The coverage record of one read (None when the guard is off); reports a short read."""
    if mode() == "off":
        return None
    short = expected_seconds - reached_seconds
    record = {
        "reader": reader,
        "reached_seconds": round(float(reached_seconds), 3),
        "expected_seconds": round(float(expected_seconds), 3),
        "short_seconds": round(float(short), 3),
        "tolerance_seconds": round(tolerance(expected_seconds), 3),
        "complete": short <= tolerance(expected_seconds),
    }
    if not record["complete"]:
        message = (
            f"{reader} reached {reached_seconds:.1f} s of {expected_seconds:.1f} s "
            f"({short:.1f} s short)"
        )
        if mode() == "fail":
            raise CoverageError(message)
        from cv.pipeline import switch_fallbacks

        switch_fallbacks.record(SWITCH, "coverage_short", message, where=where or reader)
    return record


def short_windows(
    scores_by_clip: dict, requested: dict, sample_rate: int, hop_samples: int
) -> list[str]:
    """Per-point audio windows whose decoded flux ends more than 1 s short of the request (an
    audio packet hole or the end of the stream). A record, never fatal: a point read is not a
    whole-source read, and the point's own gates decide."""
    return sorted(
        clip
        for clip, scores in scores_by_clip.items()
        if requested[clip][1] - requested[clip][0] - len(scores) * hop_samples / sample_rate > 1.0
    )
