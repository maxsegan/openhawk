"""Bounded forward completion of an explicitly witnessed second-bounce ending.

Optional subframe-anchor recovery. This does not discover endings or refit a path.
It can extend an incoming physical arc to its own ground root inside a recorded
window, retaining every fitted observation. Earlier impacts require a post-impact
observation model and deliberately remain unsupported here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from cv.pipeline.flight_anchors import ANCHOR_TIME_BOUND_FRAMES, BALL_RADIUS_M
from cv.pipeline.trajectory_contract import (
    TERMINAL_COMPLETION_SCHEMA as COMPLETION_SCHEMA,
    TERMINAL_WINDOW_SCHEMA as WINDOW_SCHEMA,
    terminal_boundary_kind,
)
from physics.bounce_reference import DWELL_SECONDS, court_bounce
from physics.flight import ground_impact_in_interval


def completion_window(boundary: dict, active_end: float) -> dict | None:
    """An explicit, configuration-sized search window; never a new ending witness."""
    if terminal_boundary_kind(boundary) != "second_bounce":
        return None
    frame, limit = float(boundary["frame"]), float(active_end)
    upper = min(limit, frame + ANCHOR_TIME_BOUND_FRAMES)
    if not math.isfinite(limit) or upper <= frame:
        return None
    return {
        "schema": WINDOW_SCHEMA,
        "observed_frame": frame,
        "bounds_frames": [frame, upper],
        "maximum_shift_frames": ANCHOR_TIME_BOUND_FRAMES,
        "source": "explicit_second_bounce_with_subframe_anchor_search_policy",
        "scope": "model search interval, not independently calibrated timing confidence",
    }


def complete_second_bounce(fit, contact: dict, fps: float, surface: str) -> dict:
    """Return a proposed physical endpoint, or an explicit abstention; mutate nothing."""
    report = {"schema": COMPLETION_SCHEMA, "status": "abstain"}
    cutoff = contact.get("terminal_cutoff") or {}
    if not isinstance(cutoff, Mapping):
        return {**report, "reason": "invalid_ending_state_or_interval"}
    boundary = cutoff.get("point_end") or {}
    window = cutoff.get("forward_completion_window") or {}
    if not isinstance(window, Mapping):
        return {**report, "reason": "invalid_ending_state_or_interval"}
    if terminal_boundary_kind(boundary) != "second_bounce" or window.get("schema") != WINDOW_SCHEMA:
        return {**report, "reason": "no_explicit_second_bounce_window"}
    node = getattr(fit, "_bounce_node_split", None)
    if (
        not isinstance(node, Mapping)
        or not node
        or node.get("sampling_model") != "measured_bounce_v1"
        or getattr(fit, "_net_split", None)
    ):
        return {**report, "reason": "unsupported_incoming_physics"}
    try:
        lower, upper = (float(value) for value in window["bounds_frames"])
        declared = float(boundary["frame"])
        cadence = float(fps)
        dwell = float(node["dwell_seconds"])
        origin = float(node["frame"]) + dwell * cadence
        if not all(math.isfinite(value) for value in (lower, upper, declared, cadence, origin)):
            raise ValueError("nonfinite ending interval")
        if (
            cadence <= 0
            or not math.isfinite(dwell)
            or dwell < 0
            or float(contact["frame"]) != declared
            or float(window["observed_frame"]) != declared
            or lower != declared
            or not lower < upper <= declared + ANCHOR_TIME_BOUND_FRAMES
        ):
            raise ValueError("invalid ending interval")
        if origin >= lower:
            return {**report, "reason": "first_bounce_not_before_completion_window"}
        observations = np.asarray(fit.obs_frames, float)
        if observations.ndim != 1 or not len(observations) or not np.all(np.isfinite(observations)):
            raise ValueError("invalid fitted observation clock")
        if np.any(observations > lower):
            return {**report, "reason": "would_discard_fitted_observations"}
        impact = ground_impact_in_interval(
            node["xyz"],
            node["outgoing_velocity"],
            node["outgoing_spin"],
            (lower - origin) / cadence,
            (upper - origin) / cadence,
            plane_z_m=BALL_RADIUS_M,
        )
        if impact is None:
            return {**report, "reason": "no_descending_ground_root_in_window"}
        frame = origin + impact.time_seconds * cadence
        sampled, velocity = (
            np.asarray(value, float) for value in fit.state(frame, cadence, surface)
        )
        if (
            sampled.shape != (3,)
            or velocity.shape != (3,)
            or not np.all(np.isfinite(sampled))
            or not np.all(np.isfinite(velocity))
            or np.linalg.norm(sampled - impact.position) > 1e-6
            or abs(sampled[2] - BALL_RADIUS_M) > 1e-7
            or velocity[2] >= 0
        ):
            return {**report, "reason": "export_sampler_disagrees_with_ground_root"}
        # Additive: a bare surface name ignores the landing point (physics.surface_model).
        rebound = court_bounce(impact.velocity, impact.spin, surface, position=impact.position)
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
        return {**report, "reason": "invalid_ending_state_or_interval"}
    return {
        **report,
        "status": "completed",
        "observed_frame": declared,
        "end_frame": frame,
        "time_shift_frames": frame - declared,
        "window": dict(window),
        "end_xyz": sampled.tolist(),
        "incoming_velocity": velocity.tolist(),
        "fitted_observations_discarded": 0,
        "geometry_change": "existing_free_flight_propagation_only",
        "bounce": {
            "frame": frame,
            "x": sampled.tolist(),
            "v_in": impact.velocity.tolist(),
            "v_out": rebound.velocity.tolist(),
            "w_in": impact.spin.tolist(),
            "w_out": rebound.spin.tolist(),
            "regime": f"terminal_measured_{rebound.regime}",
            "sampling_model": "measured_bounce_v1",
            "endpoint_source": "bounded_incoming_ground_root_v1",
            "dwell_seconds": DWELL_SECONDS,
            "spin_in_rpm": float(np.linalg.norm(impact.spin) * 60 / (2 * np.pi)),
            "spin_out_rpm": float(np.linalg.norm(rebound.spin) * 60 / (2 * np.pi)),
            "termination_anchor": True,
            "scope": "incoming endpoint; outgoing impact state is diagnostic beyond active play",
        },
    }
