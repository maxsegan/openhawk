#!/usr/bin/env python3
"""Separate in-play rally spans from dead time, with an explicit per-point abstention.

Five of the owner's ten flagged tracker transitions are not tracking defects at all: ball
girls collecting balls between points, with several balls legitimately visible. "VALID
confusion, it's between points so doesn't matter." They need a PHASE label, not better
tracking. Note this is different from a cutaway: dead time is play-camera footage, so
`shot_segments.py` correctly calls it play.

WHAT DOES NOT WORK, measured, so it is not retried:

  * ball speed alone -- rally median 3.75 px/frame vs dead 2.65, heavily overlapping. During
    dead time the tracker often follows a moving distractor at rally-like apparent speed.
  * net crossings alone -- rally 0.868 vs dead 0.372 at >=1 crossing per second.

WHAT THIS DOES. A rally is a ball repeatedly traversing the court and crossing the net; dead
time is a ball held, bounced in place, or a distractor wandering locally. The score combines
vertical traversal, net crossings and speed over a one-second window, then takes contiguous
high-score spans as rallies. All temporal parameters and velocity features are expressed in
seconds, and callers must provide the source cadence explicitly.

Because none of the individual signals is clean, the output is DELIBERATELY three-valued and
carries a point-level verdict. The owner's operating point allows invalidating up to 20% of
points outright, so a point whose phase structure is ambiguous is better held out than
guessed at. The defaults are calibrated to that budget: 5/28 points invalidated = 17.9%.

MEASURED PERFORMANCE, and it is honestly mixed. On the points it accepts, rally events are
labeled in_play 212/217 = 0.977, which is the property that matters most because mislabeling
a rally event as dead would suppress a real emission. But dead-time DETECTION is weak: only
3 of 7 owner-known dead frames on accepted points are caught. The gate reliably keeps rallies
and only sometimes recognises dead time, so it should be used to ANNOTATE phase and to route
points to review, not as a filter that suppresses anything on its own.

IMPORTANT: dead time is not empty. Dead-ball bounces are real physical events -- three are
owner-adjudicated truth in this cohort -- so this labels phase and must never be used to
DELETE events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PhaseResult", "rally_activity", "detect_phase"]

WINDOW_SECONDS = 1.0
MIN_SAMPLES_PER_SECOND = 8.0
MIN_RALLY_SECONDS = 1.2
MERGE_GAP_SECONDS = 2.0
MAX_ADJACENT_GAP_SECONDS = 0.05
TRAVERSAL_SCALE_PX_PER_SECOND = 150.0
NET_CROSSINGS_PER_SECOND = 2.0
SPEED_SCALE_PX_PER_SECOND = 200.0
ACTIVITY_ON = 0.55
ACTIVITY_OFF = 0.35


@dataclass
class PhaseResult:
    frames: np.ndarray
    activity: np.ndarray
    in_play: np.ndarray
    spans: list[tuple[float, float]] = field(default_factory=list)
    point_valid: bool = True
    reason: str = "ok"

    def phase_at(self, frame: float) -> str:
        for start, end in self.spans:
            if start <= frame <= end:
                return "in_play"
        return "dead_time"


def rally_activity(
    frames: np.ndarray,
    points: np.ndarray,
    net_row: np.ndarray | None,
    *,
    fps: float,
    window_seconds: float = WINDOW_SECONDS,
) -> np.ndarray:
    """Per-sample rally-likeness in [0, 1] from traversal, net crossings and speed.

    `net_row` is the image row of the net cord beneath each sample's column, or None when
    no camera is available (the net term is then dropped rather than faked).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    frames = np.asarray(frames, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(frames) != len(points):
        raise ValueError("frames and points must have equal length")
    if len(frames) > 1 and np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be strictly increasing")
    if net_row is not None:
        net_row = np.asarray(net_row, dtype=float)
        if len(net_row) != len(frames):
            raise ValueError("net_row and frames must have equal length")
    activity = np.zeros(len(frames))
    minimum_samples = max(3, int(np.ceil(MIN_SAMPLES_PER_SECOND * window_seconds)))
    if len(frames) < minimum_samples:
        return activity

    times = frames / fps
    side = np.sign(points[:, 1] - net_row) if net_row is not None else None
    steps_seconds = np.diff(frames) / fps
    adjacent = steps_seconds <= MAX_ADJACENT_GAP_SECONDS + 1e-9
    speed = np.full(len(frames), np.nan)
    speed[1:][adjacent] = (
        np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))[adjacent] / steps_seconds[adjacent]
    )

    for index, time in enumerate(times):
        mask = np.abs(times - time) <= window_seconds / 2
        if int(mask.sum()) < minimum_samples:
            continue
        local = points[mask]

        # Vertical traversal: a rally ball runs the length of the court in view.
        extent = float(local[:, 1].max() - local[:, 1].min())
        traversal_scale = TRAVERSAL_SCALE_PX_PER_SECOND * window_seconds
        extent_term = float(np.clip(extent / traversal_scale, 0.0, 1.0))

        # Net crossings: a rally alternates sides; dead time stays put.
        if side is not None:
            crossings = int(np.sum(np.abs(np.diff(side[mask])) > 0))
            crossing_scale = NET_CROSSINGS_PER_SECOND * window_seconds
            crossing_term = float(np.clip(crossings / crossing_scale, 0.0, 1.0))
        else:
            crossing_term = 0.0

        local_speed = speed[mask]
        finite = local_speed[np.isfinite(local_speed)]
        speed_term = (
            float(np.clip(np.median(finite) / SPEED_SCALE_PX_PER_SECOND, 0.0, 1.0))
            if len(finite)
            else 0.0
        )

        weights = (0.45, 0.35, 0.20) if side is not None else (0.7, 0.0, 0.3)
        activity[index] = (
            weights[0] * extent_term + weights[1] * crossing_term + weights[2] * speed_term
        )
    return activity


def detect_phase(
    frames: np.ndarray,
    points: np.ndarray,
    net_row: np.ndarray | None = None,
    *,
    fps: float,
    window_seconds: float = WINDOW_SECONDS,
    minimum_rally_seconds: float = MIN_RALLY_SECONDS,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    shot_count: int | None = None,
    ambiguity_limit: float = 0.34,
    shot_limit: int = 3,
) -> PhaseResult:
    """Contiguous rally spans by hysteresis on the activity score, plus a point verdict."""
    frames = np.asarray(frames, dtype=float)
    minimum_samples = max(3, int(np.ceil(MIN_SAMPLES_PER_SECOND * window_seconds)))
    activity = rally_activity(
        frames,
        points,
        net_row,
        fps=fps,
        window_seconds=window_seconds,
    )
    if len(frames) < minimum_samples:
        return PhaseResult(
            frames,
            activity,
            np.zeros(len(frames), bool),
            point_valid=False,
            reason="too_few_samples",
        )

    in_play = np.zeros(len(frames), dtype=bool)
    active = False
    for index, value in enumerate(activity):
        active = value >= ACTIVITY_ON if not active else value >= ACTIVITY_OFF
        in_play[index] = active

    spans: list[tuple[float, float]] = []
    start = None
    for index, flag in enumerate(in_play):
        if flag and start is None:
            start = frames[index]
        elif not flag and start is not None:
            spans.append((start, frames[index - 1]))
            start = None
    if start is not None:
        spans.append((start, frames[-1]))

    merged: list[tuple[float, float]] = []
    for span in spans:
        gap_seconds = (span[0] - merged[-1][1]) / fps if merged else None
        if merged and gap_seconds is not None and gap_seconds <= merge_gap_seconds:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    merged = [span for span in merged if (span[1] - span[0]) / fps >= minimum_rally_seconds]
    in_play = np.zeros(len(frames), dtype=bool)
    for start, end in merged:
        in_play |= (frames >= start) & (frames <= end)

    # Point-level verdict. The owner's operating point permits invalidating up to 20% of
    # points, so ambiguity is DECLARED rather than guessed. The phase signals are
    # individually weak (speed rally 3.75 vs dead 2.65; net crossings 0.87 vs 0.37), so this
    # is the mechanism that keeps a weak detector honest rather than confidently wrong.
    sample_times = frames / fps
    sample_weights = np.gradient(sample_times) if len(sample_times) > 1 else np.ones(1)
    ambiguity_mask = (activity > ACTIVITY_OFF) & (activity < ACTIVITY_ON)
    ambiguous = float(
        np.average(
            ambiguity_mask,
            weights=np.maximum(sample_weights, np.finfo(float).eps),
        )
    )
    valid, reason = True, "ok"
    if not merged:
        valid, reason = False, "no_rally_span_found"
    elif len(merged) > 3:
        valid, reason = False, "fragmented_phase_structure"
    elif shot_count is not None and shot_count > shot_limit:
        # Camera changes are the owner's own stated hard case. A point cut across several
        # shots has no single coherent phase timeline, so hold it out.
        valid, reason = False, "multiple_camera_shots"
    elif ambiguous > ambiguity_limit:
        valid, reason = False, "activity_ambiguous"
    else:
        covered_seconds = sum((end - start) / fps for start, end in merged)
        total_seconds = (frames[-1] - frames[0] + 1) / fps
        if covered_seconds < 0.15 * total_seconds:
            valid, reason = False, "rally_span_implausibly_short"

    return PhaseResult(
        frames=frames,
        activity=activity,
        in_play=in_play,
        spans=merged,
        point_valid=valid,
        reason=reason,
    )
