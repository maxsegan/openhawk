"""Bound source context independently of a tentative point-ending prediction.

These boundaries select source video to observe. They do not certify a physical
ending, a serve contact, or a suitable metric camera. No fitted state is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SCOPE_FIELDS = (
    "observation_scope",
    "predicted_rally_t_end",
    "observation_horizon_reason",
    "observation_camera_transition",
    "observation_audio_anchor",
    "observation_context_status",
)


def structural_horizon(
    begin: float,
    *,
    following: float,
    score_end: float,
    maximum_duration: float,
    shots: Sequence[Mapping],
    play_probabilities: Mapping[int, float],
    onsets: Sequence[float],
) -> dict:
    """Retain a bounded camera interval, including a witnessed imminent cut.

    A start detector can anticipate serve contact in the preceding shot. Only
    the immediately following shot may supply context, when an audio onset
    within one second of the start lands there and its play probability is at
    least 0.9. This is an extraction witness, not an accepted contact. Unknown
    view evidence and absent audio retain the original camera boundary.
    """
    index = next(
        (
            i
            for i, shot in enumerate(shots)
            if float(shot["t_start"]) <= begin < float(shot["t_end"])
        ),
        None,
    )
    if index is None:
        return dict(
            end=begin, reason="missing_containing_camera", camera_transition=None, audio_anchor=None
        )
    camera_end = float(shots[index]["t_end"])
    transition = None
    anchor = None
    nearby = [float(t) for t in onsets if math.isfinite(float(t)) and abs(float(t) - begin) <= 1]
    if index is not None and index + 1 < len(shots) and 0 < camera_end - begin <= 1:
        next_shot = shots[index + 1]
        next_start, next_end = float(next_shot["t_start"]), float(next_shot["t_end"])
        # Do not bridge missing source intervals or multiple camera transitions.
        contiguous = abs(next_start - camera_end) <= 1e-6
        # Select among admissible witnesses first: an earlier transient in the
        # old shot cannot veto independent evidence in the next shot.
        eligible = [t for t in nearby if next_start <= t < next_end]
        evidence_time = min(eligible, key=lambda t: (abs(t - begin), t)) if eligible else None
        next_id = int(next_shot.get("shot_index", index + 1))
        if (
            contiguous
            and evidence_time is not None
            and next_start <= evidence_time < next_end
            and play_probabilities.get(next_id, 0.0) >= 0.9
        ):
            transition, anchor, camera_end = next_start, evidence_time, next_end
    bounds = {
        "next_detected_serve": following,
        "camera_boundary": camera_end,
        "score_transition": score_end,
        "duration_cap": begin + maximum_duration,
    }
    horizon = min(bounds.values())
    return {
        "end": horizon,
        "reason": "|".join(k for k, v in bounds.items() if v == horizon),
        "camera_transition": transition,
        "audio_anchor": anchor,
    }
