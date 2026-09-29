"""Soft search support for automatic following-ground epochs in the prefix fit.

The original event remains immutable. Only the adapter's explicitly generated
prediction search radius can be a soft dead zone; observed/producer intervals
remain hard. This is an uncalibrated engineering regularizer, not a timing PMF.
"""

from __future__ import annotations

import math

from cv.experiments.connected_shooting import event_constraints, labeled_event_occurrence

MODES = ("off", "prediction_hinge")
GENERATED_ORIGINS = frozenset(
    (
        "prediction_search_radius_one_native_frame_v1",
        "prediction_search_radius_one_native_frame_window_bounded_v1",
    )
)
# Same numerical interior convention as the prefix's existing input-domain
# inequalities. This is not an observed interval or physical timing tolerance.
OPTIMIZER_INTERIOR_FRAMES = 1e-5


def validate(mode: str) -> None:
    if mode not in MODES:
        raise ValueError("following_ground_timing must be off or prediction_hinge")


def plan(event: dict, start: float, end: float) -> dict:
    """Declare one already qualified no-net, single-ground following chart."""
    low, high = map(float, event["frame_interval"])
    if not all(math.isfinite(v) for v in (low, high, start, end)) or not (start < low < high < end):
        raise ValueError("following ground must retain its ordered in-flight source interval")
    origin = event.get("interval_origin")
    # Do not infer a generated radius from numeric width or predicted status.
    generated = origin in GENERATED_ORIGINS
    soft = bool(generated and labeled_event_occurrence.predicted_membership(event))
    if generated and not soft:
        raise ValueError("generated ground timing requires explicit automatic predicted provenance")
    domain = [start + OPTIMIZER_INTERIOR_FRAMES, end - OPTIMIZER_INTERIOR_FRAMES]
    if soft and not start < domain[0] < domain[1] < end:
        raise ValueError("following contact chronology has no numerical interior")
    return dict(
        mode="prediction_hinge" if soft else "protected_hard_interval",
        interval_origin=origin,
        original_interval=[low, high],
        original_epoch=float(event["frame"]),
        chart_domain=domain if soft else [low, high],
        contact_chronology=[float(start), float(end)],
        reason=(
            "generated search radius is soft timing evidence"
            if soft
            else "origin_missing_treated_as_hard"
            if origin is None
            else "human_or_producer_interval_preserved"
        ),
        optimizer_interior_frames=OPTIMIZER_INTERIOR_FRAMES if soft else 0.0,
        time_scale_frames=event_constraints.TIME_SCALE_FRAMES if soft else None,
        calibrated=False,
        source_record_changed=False,
    )


def hinge(epoch: float, interval) -> float:
    """Existing signed timing hinge, including asymmetric window-bounded support."""
    low, high = map(float, interval)
    if not all(math.isfinite(v) for v in (epoch, low, high)) or low >= high:
        raise ValueError("finite epoch and positive-width timing interval required")
    return (epoch - min(high, max(low, epoch))) / event_constraints.TIME_SCALE_FRAMES
