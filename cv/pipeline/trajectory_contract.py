"""Structural flight connections, independent of uncertainty relative to spatial truth.

A contact is one position at one time. Foot-scale reconstruction uncertainty must not allow
two adjacent flights to disagree about that state. This contract only certifies connections,
not absolute depth, event completeness, or the dynamics inside a flight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from cv.pipeline.flight_anchors import ANCHOR_TIME_BOUND_FRAMES, BALL_RADIUS_M

POSITION_NUMERICAL_TOLERANCE_M = 1e-3
TIME_NUMERICAL_TOLERANCE_FRAMES = 1e-6
PHYSICAL_TERMINATION_KINDS = frozenset({"terminal_bounce", "second_bounce", "net_stop"})
TERMINAL_WINDOW_SCHEMA = "terminal_forward_completion_window_v1"
TERMINAL_COMPLETION_SCHEMA = "terminal_forward_ground_completion_v1"


def _forward_completion_covers(cutoff: Mapping, fit: Mapping, boundary: float, end: float) -> bool:
    """Accept only a matching, explicitly bounded physical-completion receipt."""
    window = cutoff.get("forward_completion_window") or {}
    completion = fit.get("terminal_completion") or {}
    if not isinstance(window, Mapping) or not isinstance(completion, Mapping):
        return False
    try:
        lower, upper = (float(value) for value in window["bounds_frames"])
        return (
            window.get("schema") == TERMINAL_WINDOW_SCHEMA
            and completion.get("schema") == TERMINAL_COMPLETION_SCHEMA
            and completion.get("status") == "completed"
            and completion.get("window") == window
            and all(math.isfinite(value) for value in (lower, upper))
            and lower == boundary == float(window["observed_frame"])
            and lower <= end <= upper <= boundary + ANCHOR_TIME_BOUND_FRAMES
            and float(completion["observed_frame"]) == boundary
            and abs(float(completion["end_frame"]) - end) <= TIME_NUMERICAL_TOLERANCE_FRAMES
            and completion.get("fitted_observations_discarded") == 0
            and completion.get("end_xyz") == fit.get("end_xyz")
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def terminal_ground_endpoint_report(kind: str | None, fits: Sequence[Mapping[str, Any]]) -> dict:
    """A declared court impact must actually end on the fitted court plane.

    Spatial uncertainty relative to truth is not permission for the model's own
    bounce to occur in mid-air. This checks position only, not incoming velocity,
    legal point outcome, net-stop dynamics or the interior of the final flight.
    """
    reasons = []
    error = None
    if kind not in PHYSICAL_TERMINATION_KINDS:
        reasons.append("terminal_evidence_missing")
    elif kind in {"terminal_bounce", "second_bounce"}:
        terminal_fits = [fit for fit in fits if fit.get("terminal_end") is True]
        try:
            if len(terminal_fits) != 1:
                raise ValueError("expected one terminal fit")
            xyz = tuple(float(value) for value in terminal_fits[0]["end_xyz"])
            if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
                raise ValueError("invalid terminal state")
            error = abs(xyz[2] - BALL_RADIUS_M)
            if error > POSITION_NUMERICAL_TOLERANCE_M:
                reasons.append("terminal_endpoint_off_ground")
        except (KeyError, ValueError, TypeError, OverflowError):
            reasons.append("terminal_endpoint_invalid")
    return {
        "schema": "terminal_ground_endpoint_v1",
        "valid": not reasons,
        "reasons": reasons,
        "required": kind in {"terminal_bounce", "second_bounce"},
        "ground_error_m": error,
        "ball_center_plane_z_m": BALL_RADIUS_M,
        "numerical_tolerance_m": POSITION_NUMERICAL_TOLERANCE_M,
        "scope": "court-ending position consistency only; no net-stop geometry certificate",
    }


def terminal_boundary_kind(row: Mapping[str, Any]) -> str | None:
    """Read an explicit ending declaration; a last-event marker is not one.

    This validates the consumer contract, not the producer's visual evidence or
    ancestry. An out-of-view boundary can bound diagnostic fits, not complete play.
    """
    if not isinstance(row, Mapping):
        return None
    metadata = row.get("point_end")
    if not isinstance(metadata, Mapping) or row.get("event_type") != "point_end":
        return None
    if row.get("abstain") is True or row.get("gate_held") is True:
        return None
    source = metadata.get("source")
    if not source or source == "event_grammar_decoder_terminal_path_member":
        return None
    kind = metadata.get("termination_kind")
    if kind not in (*PHYSICAL_TERMINATION_KINDS, "out_of_view"):
        return None
    try:
        if not math.isfinite(float(row["frame"])):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return kind


def terminal_coverage_report(
    contacts: Sequence[Mapping[str, Any]], fits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require one explicit physical ending and a fitted final flight covering it.

    A fallback track/span end, camera exit, or residual-driven early stop remains
    a useful diagnostic boundary but cannot certify a complete scoring point.
    """
    terminals = [row for row in contacts if row.get("terminal") is True]
    terminal_fits = [row for row in fits if row.get("terminal_end") is True]
    reasons = []
    kind = None
    boundary_frame = fitted_frame = None
    forward_completed = False
    if len(terminals) != 1 or len(terminal_fits) != 1:
        reasons.append("terminal_flight_coverage")
    else:
        cutoff = terminals[0].get("terminal_cutoff") or {}
        if not isinstance(cutoff, Mapping):
            cutoff = {}
        boundary = cutoff.get("point_end") or {}
        kind = terminal_boundary_kind(boundary)
        if kind is None:
            reasons.append("terminal_evidence_missing")
        elif kind not in PHYSICAL_TERMINATION_KINDS:
            reasons.append("terminal_observation_exit")
        if kind is not None:
            boundary_frame = float(boundary["frame"])
            try:
                fitted_frame = float(terminal_fits[0]["end_frame"])
                if not math.isfinite(fitted_frame):
                    fitted_frame = None
                if kind == "second_bounce" and fitted_frame is not None:
                    forward_completed = _forward_completion_covers(
                        cutoff, terminal_fits[0], boundary_frame, fitted_frame
                    )
                if fitted_frame is None or (
                    abs(fitted_frame - boundary_frame) > TIME_NUMERICAL_TOLERANCE_FRAMES
                    and not forward_completed
                ):
                    reasons.append("terminal_end_not_covered")
            except (KeyError, TypeError, ValueError, OverflowError):
                reasons.append("terminal_end_not_covered")
    endpoint = terminal_ground_endpoint_report(kind, fits)
    if not reasons:
        reasons.extend(endpoint["reasons"])
    return {
        "schema": "terminal_coverage_v3",
        "valid": not reasons,
        "reasons": reasons,
        "termination_kind": kind,
        "boundary_frame": boundary_frame,
        "fitted_end_frame": fitted_frame,
        "ground_endpoint": endpoint,
        "ending_time_policy": "bounded_forward_completion"
        if forward_completed
        else "declared_frame",
        "scope": "declared ending time and court-plane consistency, not independent correctness",
    }


def trajectory_connection_report(fits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed on missing/nonfinite endpoints, disconnected times, or positional seams.

    Millimetre and millionth-frame tolerances accommodate numerical roundoff, not inferred
    contact error. Inputs are inspected without moving an endpoint or filling an absent flight.
    """
    reasons: list[str] = []
    normalized = []
    for fit in fits:
        try:
            start, end = float(fit["start_frame"]), float(fit["end_frame"])
            start_xyz = tuple(float(value) for value in fit["start_xyz"])
            end_xyz = tuple(float(value) for value in fit["end_xyz"])
            if (
                len(start_xyz) != 3
                or len(end_xyz) != 3
                or not all(math.isfinite(value) for value in (start, end, *start_xyz, *end_xyz))
                or end <= start
            ):
                raise ValueError("invalid flight endpoints")
        except (KeyError, TypeError, ValueError, OverflowError):
            reasons.append("physics_endpoint_invalid")
            continue
        normalized.append((start, end, start_xyz, end_xyz))

    junctions = []
    # Do not connect around an unparseable flight and thereby erase the missing evidence.
    if len(normalized) == len(fits):
        normalized.sort(key=lambda row: row[0])
        for incoming, outgoing in zip(normalized, normalized[1:]):
            time_gap = outgoing[0] - incoming[1]
            position_gap = math.dist(incoming[3], outgoing[2])
            time_connected = abs(time_gap) <= TIME_NUMERICAL_TOLERANCE_FRAMES
            position_connected = position_gap <= POSITION_NUMERICAL_TOLERANCE_M
            if not time_connected:
                reasons.append("physics_junction_time")
            if not position_connected:
                reasons.append("physics_junction")
            junctions.append(
                {
                    "incoming_end_frame": incoming[1],
                    "outgoing_start_frame": outgoing[0],
                    "time_gap_frames": time_gap if math.isfinite(time_gap) else None,
                    "position_gap_m": position_gap if math.isfinite(position_gap) else None,
                    "connected": time_connected and position_connected,
                }
            )
    if not fits:
        reasons.append("physics_endpoint_missing")
    return {
        "schema": "trajectory_connections_v1",
        "position_tolerance_m": POSITION_NUMERICAL_TOLERANCE_M,
        "time_tolerance_frames": TIME_NUMERICAL_TOLERANCE_FRAMES,
        "flight_count": len(fits),
        "expected_junctions": max(0, len(fits) - 1),
        "checked_junctions": len(junctions),
        "valid": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "junctions": junctions,
    }
