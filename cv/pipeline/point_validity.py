"""Compose camera/play and tracking-health evidence into a bounded point gate."""

from __future__ import annotations

import math

HARD_CAMERA_REASONS = {
    "court_geometry_invalid",
    "insufficient_input",
    "no_play_camera_shot",
    "phase:multiple_camera_shots",
}

# A broadcast cut inside a supplied clip is a shot-composition fact, not a camera
# failure: the court can still be registered and the producer can still name the
# frames it believes are live play.  ``retained_play_scope`` separates the two, and
# only for a point whose remaining hard reasons are exactly these.
SHOT_COMPOSITION_REASONS = {"phase:multiple_camera_shots"}
CAMERA_FAILURE_REASONS = HARD_CAMERA_REASONS - SHOT_COMPOSITION_REASONS
PLAY_SCOPE_DIAGNOSTIC_REASON = "multiple_camera_shots_scoped_to_play_interval"
PLAY_SCOPE_SCHEMA = "retained_play_event_scope_v1"
# Declared verbatim from cv.pipeline.camera_frame_support so the published scope
# names the transport its support evidence came from.
CAMERA_SUPPORT_TRANSPORT = {"mode": "reliable_per_frame", "missing": "hold"}


def _interval(value, *, allow_degenerate: bool = False) -> tuple[float, float] | None:
    """Return a finite ``[low, high]`` pair, or None for anything else.

    A play span must cover more than an instant, so ``low < high`` by default.  A
    per-frame camera support run is a closed set of native frame integers and may
    legitimately be one frame wide, which ``allow_degenerate`` permits.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    if high < low if allow_degenerate else not low < high:
        return None
    return low, high


def _ordered_intervals(
    values, *, bounds: tuple[float, float], allow_degenerate: bool = False
) -> list[tuple[float, float]] | None:
    """Validate a non-empty ordered, disjoint interval list inside ``bounds``."""
    if not isinstance(values, list) or not values:
        return None
    intervals: list[tuple[float, float]] = []
    for item in values:
        span = _interval(item, allow_degenerate=allow_degenerate)
        if span is None or span[0] < bounds[0] or bounds[1] < span[1]:
            return None
        if intervals and not intervals[-1][1] < span[0]:
            return None
        intervals.append(span)
    return intervals


def _frames_covered(spans: list[tuple[float, float]]) -> int:
    """Count the native frame integers a closed float interval list covers."""
    return sum(max(0, math.floor(high) - math.ceil(low) + 1) for low, high in spans)


def _intersection(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersect two ordered, disjoint closed interval lists."""
    overlap = []
    for low, high in left:
        for other_low, other_high in right:
            start, stop = max(low, other_low), min(high, other_high)
            if start <= stop:
                overlap.append((start, stop))
    return sorted(overlap)


def retained_play_scope(
    active: dict,
    active_reasons: list[str],
    geometry_valid,
    camera_support: dict | None,
) -> dict | None:
    """Certify the frames a shot-composition point may still emit events on.

    The verdict is softened only with positive evidence that the producers already
    computed:

    * an explicitly valid court geometry;
    * a usable ``trim`` inside the clip's own native frame window;
    * a non-empty set of finite retained live-play spans inside that trim.  Ordering
      and holes between spans are preserved verbatim -- a replay segment between two
      rallies stays outside the scope;
    * a non-empty per-frame reliable-camera support set (see
      :mod:`cv.pipeline.camera_frame_support`) that actually overlaps those spans.

    The last condition is separate evidence, not a restatement of the first three.
    ``is_play_camera`` is a shot-level verdict and a supplied clip can hold a close-up
    inside a span it marks live, so a span union alone never certifies the view.  The
    published support spans stay verbatim rather than being folded into the play spans:
    a consumer refuses the individual unsupported rows and keeps the supported ones,
    instead of widening or holding the whole point.

    Any other hard reason, a missing or malformed field, the absence of a retained
    interval, or a play interval with no supported frame in it returns None and leaves
    the original hold in place.
    """
    reasons = set(active_reasons)
    shot_reasons = reasons & SHOT_COMPOSITION_REASONS
    if not shot_reasons or reasons & CAMERA_FAILURE_REASONS:
        return None
    if geometry_valid is not True:
        return None
    trim = _interval(active.get("trim"))
    if trim is None:
        return None
    try:
        native_frames = float(active["n_frames"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(native_frames) or native_frames <= 0:
        return None
    if not 0.0 <= trim[0] or not trim[1] <= native_frames:
        return None
    spans = _ordered_intervals(active.get("active_spans"), bounds=trim)
    if spans is None:
        return None
    if not isinstance(camera_support, dict):
        return None
    supported = _ordered_intervals(
        camera_support.get("supported_spans"),
        bounds=(0.0, native_frames),
        allow_degenerate=True,
    )
    if supported is None:
        return None
    admitted = _intersection(spans, supported)
    if not admitted:
        return None
    return {
        "schema": PLAY_SCOPE_SCHEMA,
        "reason": PLAY_SCOPE_DIAGNOSTIC_REASON,
        "shot_composition_reasons": sorted(shot_reasons),
        "native_frames": native_frames,
        "trim": [trim[0], trim[1]],
        "spans": [[low, high] for low, high in spans],
        "camera_support": {
            "transport": CAMERA_SUPPORT_TRANSPORT,
            "supported_spans": [[low, high] for low, high in supported],
            "play_span_frames": _frames_covered(spans),
            "supported_play_span_frames": _frames_covered(admitted),
            "admitted_spans": [[low, high] for low, high in admitted],
        },
    }


def compose_point_gate(
    active_play: dict[str, dict],
    tracking_rows: list[dict],
    *,
    maximum_invalid_fraction: float = 0.2,
    court_geometry_rows: list[dict] | None = None,
    frame_cadence_rows: list[dict] | None = None,
    retained_play_scope_enabled: bool = False,
    camera_support_rows: list[dict] | None = None,
) -> dict:
    """Compose the point gate.

    ``retained_play_scope_enabled`` is default-off and reproduces the previous
    document exactly when false.  When true, a point held only because its shot
    composition is ``phase:multiple_camera_shots`` is retained *with an explicit
    play scope* -- see :func:`retained_play_scope` -- instead of being refused as a
    camera failure.  Camera, geometry and cadence failures, the tracking arcs and
    the cohort budget are untouched.

    The scope needs per-frame view evidence, so ``camera_support_rows`` is required
    whenever it is enabled: a caller that cannot supply the reliable-per-frame camera
    support document does not silently fall back to the span-only verdict.  A point
    missing from those rows keeps its original hold.
    """
    if maximum_invalid_fraction is not None and not 0.0 <= maximum_invalid_fraction <= 1.0:
        raise ValueError("maximum_invalid_fraction must be in [0, 1] or None (absolute mode)")
    if retained_play_scope_enabled and camera_support_rows is None:
        raise ValueError("the retained play scope requires reliable per-frame camera support rows")
    tracking = {f"{row['match_id']}/{row['clip']}": row for row in tracking_rows}
    court_geometry = {row["point"]: row for row in (court_geometry_rows or [])}
    frame_cadence = {f"{row['match_id']}/{row['clip']}": row for row in (frame_cadence_rows or [])}
    camera_support = {row["point"]: row for row in (camera_support_rows or [])}
    keys = sorted(set(active_play) | set(tracking) | set(court_geometry) | set(frame_cadence))
    budget = (
        None
        if maximum_invalid_fraction is None
        else math.floor(maximum_invalid_fraction * len(keys))
    )
    rows = []
    for key in keys:
        active = active_play.get(key, {})
        track = tracking.get(key, {})
        geometry = court_geometry.get(key, {})
        cadence = frame_cadence.get(key, {})
        active_reasons = list(active.get("reasons", []))
        if geometry and not geometry.get("valid", False):
            active_reasons.append("court_geometry_invalid")
        hard_camera = any(reason in HARD_CAMERA_REASONS for reason in active_reasons)
        play_scope = (
            retained_play_scope(
                active, active_reasons, geometry.get("valid"), camera_support.get(key)
            )
            if retained_play_scope_enabled and hard_camera
            else None
        )
        if play_scope is not None:
            hard_camera = False
        hard_timing = cadence.get("decision") == "timing_hold"
        tracking_arcs = list(track.get("arcs", []))
        retained_tracking_arcs = [arc for arc in tracking_arcs if arc.get("decision") == "retain"]
        held_tracking_arcs = [arc for arc in tracking_arcs if arc.get("decision") == "hold"]
        arc_gate_available = bool(tracking_arcs)
        tracking_whole_point_hold = (
            not retained_tracking_arcs if arc_gate_available else track.get("decision") == "hold"
        )
        rows.append(
            {
                "point": key,
                "match_id": key.split("/", 1)[0],
                "clip": key.split("/", 1)[1],
                "hard_camera_failure": hard_camera,
                "hard_timing_failure": hard_timing,
                "active_play_valid": bool(active.get("point_valid", False)),
                "active_play_reasons": active_reasons,
                "court_geometry_valid": geometry.get("valid"),
                "court_geometry_reasons": list(geometry.get("reasons", [])),
                "trim": active.get("trim"),
                "trimmed_fraction": active.get("trimmed_fraction"),
                "tracking_decision": track.get("decision", ""),
                "tracking_risk": float(track.get("quality_risk", float("-inf"))),
                "tracking_arc_gate_available": arc_gate_available,
                "tracking_arc_decision": (
                    "unavailable"
                    if not arc_gate_available
                    else "retain"
                    if not held_tracking_arcs
                    else "partial"
                    if retained_tracking_arcs
                    else "hold"
                ),
                "tracking_whole_point_hold": tracking_whole_point_hold,
                "tracking_whole_point_decision": (
                    "hold" if tracking_whole_point_hold else "retain"
                ),
                "tracking_arcs": tracking_arcs,
                "retained_tracking_arc_count": len(retained_tracking_arcs),
                "held_tracking_arc_count": len(held_tracking_arcs),
                "frame_cadence_decision": cadence.get("decision"),
                "active_duplicate_rate": cadence.get("active_duplicate_rate"),
                # Only a scoped point carries these; every other row keeps the
                # exact field set the previous gate wrote.
                **(
                    {
                        "play_scope_native_interval": play_scope["trim"],
                        "play_scope_native_spans": play_scope["spans"],
                        "play_scope_camera_supported_spans": play_scope["camera_support"][
                            "supported_spans"
                        ],
                        "play_scope": play_scope,
                    }
                    if play_scope is not None
                    else {}
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            not (row["hard_camera_failure"] or row["hard_timing_failure"]),
            -row["tracking_risk"],
            row["point"],
        ),
    )
    mandatory = {
        row["point"] for row in rows if row["hard_camera_failure"] or row["hard_timing_failure"]
    }
    if budget is None:
        # Absolute mode: no cohort-relative fill. Tracking abstains per arc and
        # therefore cannot veto an entire point. The old whole-point verdict is
        # retained above as an explicit diagnostic only.
        held = mandatory
    else:
        discretionary_budget = max(0, budget - len(mandatory))
        discretionary = [
            row
            for row in ranked
            if row["point"] not in mandatory
            and not (row["tracking_arc_gate_available"] and not row["tracking_whole_point_hold"])
        ]
        held = mandatory | {row["point"] for row in discretionary[:discretionary_budget]}
    for row in rows:
        row["decision"] = "hold" if row["point"] in held else "retain"
        reasons = []
        diagnostic_reasons = []
        if row["hard_camera_failure"]:
            reasons.append("hard_camera_failure")
        if row["hard_timing_failure"]:
            reasons.append("systematic_repeated_visual_frames")
        if row["tracking_whole_point_hold"]:
            diagnostic_reasons.append("tracking_risk")
        elif row["held_tracking_arc_count"]:
            diagnostic_reasons.append("tracking_arc_abstentions")
        if row.get("play_scope") is not None:
            diagnostic_reasons.append(row["play_scope"]["reason"])
        if not row["active_play_valid"] and not row["hard_camera_failure"]:
            diagnostic_reasons.append("active_play_ambiguous")
        row["reasons"] = reasons
        row["diagnostic_reasons"] = diagnostic_reasons
    return {
        "schema": "point_validity_gate_v1",
        "labels_loaded": False,
        "maximum_invalid_fraction": maximum_invalid_fraction,
        "budget": budget,
        "mandatory_holds": len(mandatory),
        "points": len(rows),
        "held": len(held),
        "retained": len(rows) - len(held),
        "partial_tracking_points": sum(row["tracking_arc_decision"] == "partial" for row in rows),
        "whole_point_tracking_diagnostic_held": sum(
            row["tracking_whole_point_hold"] for row in rows
        ),
        **(
            {
                "retained_play_scope": True,
                "play_interval_scoped_points": sum(
                    row.get("play_scope") is not None for row in rows
                ),
                # How much of the certified live-play interval the automatic camera
                # transport actually supports, so a scoped point is never read as a
                # claim that its whole span union is on the play view.
                "play_scope_camera_support": {
                    "transport": CAMERA_SUPPORT_TRANSPORT,
                    "play_span_frames": sum(
                        row["play_scope"]["camera_support"]["play_span_frames"]
                        for row in rows
                        if row.get("play_scope") is not None
                    ),
                    "supported_play_span_frames": sum(
                        row["play_scope"]["camera_support"]["supported_play_span_frames"]
                        for row in rows
                        if row.get("play_scope") is not None
                    ),
                },
            }
            if retained_play_scope_enabled
            else {}
        ),
        "rows": rows,
    }
