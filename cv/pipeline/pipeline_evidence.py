"""Shared uncertainty evidence carried between tracking, events, and 3D fitting."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def tracking_observation_weight(
    score: float,
    sources: str | Iterable[str],
    *,
    interpolated: bool = False,
) -> float:
    """Map tracker confidence and detector agreement to a bounded fit weight."""
    if isinstance(sources, str):
        source_count = len({part for part in sources.split("+") if part})
    else:
        source_count = len(set(sources))
    calibrated = 0.20 + 0.65 * float(np.clip(score, 0.0, 1.0))
    if source_count >= 2:
        calibrated += 0.15
    if interpolated:
        calibrated *= 0.45
    return float(np.clip(calibrated, 0.10, 1.0))


def event_type_probability(row: dict, event_type: str) -> float:
    """Retain the strongest calibrated event interpretation without collapsing types."""
    values = [
        float(row.get(f"{source}_p_{event_type}", 0.0) or 0.0)
        for source in ("native", "claude", "dense", "meta")
    ]
    return float(np.clip(max(values), 0.0, 1.0))


def select_physics_compatible_bounce(
    candidates: list[dict],
    predicted_bounces: list[dict],
    *,
    fps: float,
    maximum_seconds: float = 0.20,
    maximum_court_distance_m: float = 3.0,
) -> dict | None:
    """Select a soft bounce anchor only when event and physics hypotheses overlap.

    The unconstrained physics fit is not treated as truth. It acts as a compatibility
    witness, while event probability controls the anchor uncertainty. No compatible
    candidate means no anchor rather than a forced no-bounce decision.
    """
    if not candidates or not predicted_bounces:
        return None
    ranked = []
    for candidate in candidates:
        probability = float(candidate["probability"])
        if probability < 0.10:
            continue
        for predicted in predicted_bounces:
            timing_frames = abs(float(candidate["frame"]) - float(predicted["frame"]))
            court_distance = float(
                np.linalg.norm(
                    np.asarray(candidate["court_xy"], float)
                    - np.asarray(predicted["x"][:2], float)
                )
            )
            if timing_frames > maximum_seconds * fps:
                continue
            if court_distance > maximum_court_distance_m:
                continue
            sigma_frame = 0.55 + 2.0 * (1.0 - probability)
            sigma_xy = 0.25 + 1.25 * (1.0 - probability)
            cost = (
                timing_frames / sigma_frame
                + court_distance / sigma_xy
                - math.log(max(probability, 1e-6))
            )
            ranked.append(
                (
                    cost,
                    {
                        **candidate,
                        "sigma_frame": sigma_frame,
                        "sigma_xy_m": sigma_xy,
                        "physics_timing_delta_frames": timing_frames,
                        "physics_court_delta_m": court_distance,
                    },
                )
            )
    return min(ranked, key=lambda item: item[0])[1] if ranked else None
