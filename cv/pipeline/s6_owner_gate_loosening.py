"""The gate loosening the owner approved on 2026-09-24.

"I'm okay with this loosening." It covers four acceptance checks and nothing
else. Absent, or ``off``, is the production rung those checks already used.
Deleting the line from ``PIPELINE_COMPONENT_POLICY`` is the rollback.

The numbers are the measured setting ``R3 + T3 + D3 + G1`` in
``cv/experiments/ceiling_census/FINAL_REPORT_gates.md``. A directional waiver
applies only when that check is the flight's only failure; that rule lives in
``per_flight_acceptance.score``.
"""

from __future__ import annotations

from dataclasses import replace

from cv.experiments.connected_shooting.per_flight_acceptance import Thresholds

KEY = "owner_gate_loosening_20260924"
OFF = "off"
NEAR_MISS = "near_miss_20260924"

#: Rollback values, which are the current production rung. Not applied: the
#: rung already carries them. Written here so the approved change and its
#: reversal are the same table.
ROLLBACK = {
    "bounce_ray_limit_m": 0.75,
    "serve_bounce_ray_limit_m": 0.50,
    "grass_bounce_epoch_frames": 0.85,
    "bounce_uncertainty_frames": 2.0,
    "ending_passive_context_frames": 0.5,
    "directional_only_waiver_px": None,
}

NEAR_MISS_VALUES = {
    "bounce_ray_limit_m": 1.25,
    "serve_bounce_ray_limit_m": 1.15,
    "grass_bounce_epoch_frames": 1.25,
    "bounce_uncertainty_frames": 3.5,
    "ending_passive_context_frames": 0.75,
    "directional_only_waiver_px": 48.0,
}


def overlay(threshold: Thresholds, settings: dict) -> Thresholds:
    """Return the rung unchanged, or the approved near-miss rung."""
    mode = settings.get(KEY, OFF)
    if mode in (None, OFF):
        return threshold
    if mode != NEAR_MISS:
        raise ValueError(f"unsupported {KEY}")
    return replace(threshold, **NEAR_MISS_VALUES)
