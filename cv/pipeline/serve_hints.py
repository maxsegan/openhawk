"""Shared serve-launch hints for ball-event and arc pipelines."""

from __future__ import annotations

import numpy as np

SERVE_FLIGHT_MIN_PX = 150.0
SERVE_HINT_MIN_GAP_SECONDS = 3.6
SERVE_PLAYER_RADIUS_HEIGHTS = 1.35
REFERENCE_FPS = 50.0


def _smooth(values: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1 or len(values) < width:
        return values.copy()
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def _launch_near_player(
    boxes: dict,
    x: float,
    y: float,
    frame: int,
    frame_tolerance: int,
) -> str | None:
    best: tuple[float, str] | None = None
    for side in ("near", "far"):
        candidates = [
            box_frame
            for box_frame in boxes.get(side, {})
            if abs(box_frame - frame) <= frame_tolerance
        ]
        if not candidates:
            continue
        box_frame = min(candidates, key=lambda candidate: abs(candidate - frame))
        x0, y0, x1, y1 = boxes[side][box_frame]
        height = max(1.0, y1 - y0)
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        normalized_distance = float(np.hypot(x - center_x, y - center_y) / height)
        if normalized_distance <= SERVE_PLAYER_RADIUS_HEIGHTS and (
            best is None or normalized_distance < best[0]
        ):
            best = (normalized_distance, side)
    return best[1] if best else None


def auto_serve_hints(
    frames: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    boxes: dict,
    fps: float,
    *,
    max_serves: int = 2,
) -> list[float]:
    """Find at most two plausible serve launches across a point clip.

    A candidate must launch a sustained full-court flight, be near a player at the
    launch frame itself, and be separated enough from a previous serve to represent
    a fault/second-serve sequence rather than a rally contact or dead-ball relaunch.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    order = np.argsort(frames)
    sorted_frames = frames[order].astype(float)
    sorted_xs = xs[order].astype(float)
    sorted_ys = ys[order].astype(float)
    if len(sorted_frames) < 8:
        return []

    smooth_width = max(1, round(0.06 * fps))
    smooth_xs = _smooth(sorted_xs, smooth_width)
    smooth_ys = _smooth(sorted_ys, smooth_width)
    frame_delta = np.diff(sorted_frames)
    frame_delta[frame_delta == 0] = 1
    step = (
        np.hypot(np.diff(smooth_xs), np.diff(smooth_ys))
        / frame_delta
        * fps
        / REFERENCE_FPS
    )
    step = np.r_[step[0], step]

    hints: list[float] = []
    index = 3
    while index < len(sorted_frames) - 2 and len(hints) < max_serves:
        launch = (
            step[index] > 5.5
            and step[index + 1] > 5.5
            and float(np.median(step[max(0, index - 5) : index])) < 3.0
        )
        if not launch:
            index += 1
            continue

        frame = float(sorted_frames[index])
        near_at_launch = _launch_near_player(
            boxes,
            float(sorted_xs[index]),
            float(sorted_ys[index]),
            int(round(frame)),
            max(1, round(0.12 * fps)),
        )
        window = slice(index, min(len(sorted_frames), index + round(0.6 * fps)))
        travel = float(
            np.max(
                np.hypot(
                    sorted_xs[window] - sorted_xs[index],
                    sorted_ys[window] - sorted_ys[index],
                )
            )
        )
        separated = (
            not hints or frame - hints[-1] >= SERVE_HINT_MIN_GAP_SECONDS * fps
        )
        if near_at_launch and travel >= SERVE_FLIGHT_MIN_PX and separated:
            hints.append(frame)
            index += round(0.8 * fps)
            continue
        index += 1
    return hints
