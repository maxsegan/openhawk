"""Start box for the first flight of an unknown-origin scene (gate review, 2026-10-04).

Without serve priors, one broadcast view fixes an unknown-origin first flight's start
depth only weakly: in the gate-review rerun, 5 of 45 such first flights started
mid-court or 7-11 m behind a baseline. ``on`` refuses that first flight unless its start
contact is inside the frozen box in ``per_flight_acceptance``
(``UNKNOWN_ORIGIN_START_*``). Absent, or ``off``, changes nothing. Measurement:
``cv/experiments/gate_review/REPORT.md``.
"""

from __future__ import annotations

from dataclasses import replace

from cv.experiments.connected_shooting.per_flight_acceptance import Thresholds

KEY = "unknown_origin_start_box_20261004"
OFF = "off"
ON = "on"


def overlay(threshold: Thresholds, settings: dict) -> Thresholds:
    """Return the rung unchanged, or with the unknown-origin start box on."""
    mode = settings.get(KEY, OFF)
    if mode in (None, OFF):
        return threshold
    if mode != ON:
        raise ValueError(f"unsupported {KEY}")
    return replace(threshold, unknown_origin_start_box=True)
