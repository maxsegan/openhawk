#!/usr/bin/env python3
"""Heal ball tracks: drop jump outliers, keep genuine kinks, fill short gaps.

The decoded track is accurate for most frames and then occasionally jumps -- the tracker
latches onto a line, a logo or a shoe for a frame or two before recovering. Measured on the
audit cohort, a robust fit to each inter-event arc leaves a 4.8 px residual on inliers but
12.9 px including outliers, with a median 13.5% of samples flagged. Non-robust downstream
fits therefore measure the jumps rather than the ball, which is what made a ballistic lift
look impossible.

THE ONE THING THIS MUST NOT DO is smooth away a real event. A racket hit or a bounce is a
genuine instantaneous kink and carries the signal the whole pipeline is built on. The
discriminator is persistence, not magnitude:

    JUMP   a sample departs the trajectory and the track RETURNS to the original path.
    KINK   the track departs and STAYS on the new path.

So a sample is judged against a short linear extrapolation from the samples BEFORE it and,
separately, from the samples AFTER it. Matching either side is enough to keep it -- which is
exactly what a point on one side of a kink does -- while a jump matches neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HealResult", "heal_track"]

# Calibrated so the REMOVAL RATE matches the independently measured outlier rate (13.5%,
# from a robust-quadratic fit to inter-event arcs). An earlier 9 px value was chosen from
# the inlier residual alone and removed 22.5% -- roughly twice the real outlier population,
# discarding good samples and measurably degrading the downstream event scorer. The gate has
# to absorb linear-extrapolation error over curved, gappy arcs, not just detection noise.
# In 960x540 space; scale it if you pass native pixels.
DEFAULT_TOLERANCE_PX = 14.0
DEFAULT_SIDE_SAMPLES = 4
DEFAULT_MAX_GAP = 4
# Cadence-aware equivalents. The frame-based defaults above encode 50 fps assumptions:
# max_reach 8 frames = 160 ms of extrapolation, max_gap 4 frames = 80 ms of fill. On native
# 24/25 fps sources the same frame counts would silently DOUBLE those durations, so callers
# with a known fps should let heal_track derive the frame counts from these.
DEFAULT_MAX_REACH_SECONDS = 0.16
DEFAULT_MAX_GAP_SECONDS = 0.08
REFERENCE_FPS = 50.0


@dataclass
class HealResult:
    frames: np.ndarray
    points: np.ndarray
    kept: np.ndarray            # bool mask over the INPUT samples
    interpolated: np.ndarray    # bool mask over the OUTPUT samples
    n_removed: int
    n_filled: int

    @property
    def outlier_rate(self) -> float:
        return float(1.0 - self.kept.mean()) if len(self.kept) else 0.0


def _extrapolate(frames: np.ndarray, values: np.ndarray, target: float) -> np.ndarray | None:
    """Linear extrapolation from a short run of samples to `target`."""
    if len(frames) < 2:
        return None
    span = frames[-1] - frames[0]
    if span <= 0:
        return None
    slope = (values[-1] - values[0]) / span
    return values[-1] + slope * (target - frames[-1])


def _side_prediction(
    frames: np.ndarray,
    points: np.ndarray,
    index: int,
    direction: int,
    samples: int,
    max_reach: float,
) -> np.ndarray | None:
    """Predict sample `index` from `samples` neighbours on one side."""
    indices = []
    step = 1
    while len(indices) < samples:
        j = index + direction * step
        if j < 0 or j >= len(frames):
            break
        if abs(frames[j] - frames[index]) > max_reach:
            break
        indices.append(j)
        step += 1
    if len(indices) < 2:
        return None
    indices = sorted(indices)
    return _extrapolate(frames[indices], points[indices], frames[index])


def heal_track(
    frames: np.ndarray,
    points: np.ndarray,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    side_samples: int = DEFAULT_SIDE_SAMPLES,
    max_reach: float | None = None,
    max_gap: int | None = None,
    passes: int = 2,
    fps: float = REFERENCE_FPS,
) -> HealResult:
    """Remove jump outliers and fill short gaps by interpolation.

    `points` is (n, 2) in whatever pixel space the caller uses; `tolerance_px` is in that
    same space. Returns the healed track plus masks recording what was removed and what was
    interpolated, so a consumer can down-weight synthesised samples.
    """
    if max_reach is None:
        max_reach = max(2.0, round(DEFAULT_MAX_REACH_SECONDS * fps))
    if max_gap is None:
        max_gap = max(1, int(round(DEFAULT_MAX_GAP_SECONDS * fps)))
    frames = np.asarray(frames, dtype=float).copy()
    points = np.asarray(points, dtype=float).copy()
    if len(frames) < 5:
        return HealResult(frames, points, np.ones(len(frames), bool),
                          np.zeros(len(frames), bool), 0, 0)

    order = np.argsort(frames)
    frames, points = frames[order], points[order]
    kept = np.ones(len(frames), bool)

    for _ in range(passes):
        live_frames, live_points = frames[kept], points[kept]
        flagged: list[int] = []
        live_indices = np.flatnonzero(kept)
        for position, index in enumerate(live_indices):
            errors = []
            for direction in (-1, 1):
                prediction = _side_prediction(
                    live_frames, live_points, position, direction, side_samples, max_reach
                )
                if prediction is not None:
                    errors.append(float(np.hypot(*(live_points[position] - prediction))))
            # Endpoints have only one usable side; require that side to agree.
            if not errors:
                continue
            # A genuine kink matches the side it belongs to, so the MINIMUM decides.
            if min(errors) > tolerance_px:
                flagged.append(index)
        if not flagged:
            break
        kept[flagged] = False
        if kept.sum() < 5:
            kept[flagged] = True
            break

    healed_frames = frames[kept]
    healed_points = points[kept]

    # Fill short interior gaps so downstream fits see an evenly sampled arc. Long gaps are
    # left alone: inventing a ball across them would be fabrication, not healing.
    filled_frames: list[float] = []
    filled_points: list[np.ndarray] = []
    filled_flags: list[bool] = []
    for i in range(len(healed_frames)):
        filled_frames.append(healed_frames[i])
        filled_points.append(healed_points[i])
        filled_flags.append(False)
        if i + 1 >= len(healed_frames):
            continue
        gap = int(healed_frames[i + 1] - healed_frames[i])
        if 1 < gap <= max_gap + 1:
            for step in range(1, gap):
                ratio = step / gap
                filled_frames.append(healed_frames[i] + step)
                filled_points.append(
                    healed_points[i] * (1 - ratio) + healed_points[i + 1] * ratio
                )
                filled_flags.append(True)

    return HealResult(
        frames=np.array(filled_frames),
        points=np.array(filled_points),
        kept=kept,
        interpolated=np.array(filled_flags, dtype=bool),
        n_removed=int((~kept).sum()),
        n_filled=int(sum(filled_flags)),
    )
