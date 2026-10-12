"""Per-flight failure index (public release: disabled).

The private pipeline writes ``FAILED_FLIGHTS.json`` beside ``OUTCOMES.json`` with, for every
flight slot that did not validate, the stage where it stopped and the command that regenerates
it on the authors' batch tooling. That tooling is not part of this release, so the index is off:
``product_runner`` checks ``enabled()`` and writes nothing. Flights and every other output are
unaffected (the index is a review artifact only).
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "failed_flights_v1"
FILE = "FAILED_FLIGHTS.json"
SWITCH = "TENNIS_FAILURE_INDEX"


def enabled() -> bool:
    return False


def write(match_dir: Path, match_id: str, destination: Path | None = None, **kwargs) -> dict:
    raise RuntimeError("the failure index is not part of the public release")


def refresh(match_dir: Path, match_id: str) -> dict | None:
    return None


def summary(document: dict) -> dict:
    return document.get("counts", {})
