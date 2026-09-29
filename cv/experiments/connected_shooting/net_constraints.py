"""Research net-clearance objective for explicitly collision-free flights.

Use only when the supplied event inventory excludes net contacts. Dense samples
are numerical trajectory queries, never invented video observations. The same
chord/plane intersections as the physical audit define clearance. This penalty
does not enforce a crossing, simulate a net collision, or certify a valid path.
"""

from __future__ import annotations

import numpy as np


def validate(scene, scale_m, bounce_frames):
    if (
        isinstance(scale_m, (bool, np.bool_))
        or not np.isfinite(scale_m)
        or scale_m <= 0
        or scene.dynamics != "measured_240hz"
        or bounce_frames is None
    ):
        raise ValueError(
            "net clearance requires a positive scale, measured dynamics and event inventory"
        )


def dense_queries(scene, bounce_frames):
    return tuple(
        np.unique(
            np.r_[
                native,
                events,
                np.linspace(a, b, max(2, int(np.ceil((b - a) * 240 / scene.fps)) + 1)),
            ]
        )
        for a, b, native, events in zip(
            scene.contact_frames[:-1],
            scene.contact_frames[1:],
            scene.observation_frames,
            bounce_frames,
            strict=True,
        )
    )


def penalty(frames, positions, scale_m):
    from cv.experiments.connected_shooting.physical_compatibility import crossings

    rows = crossings(frames, positions)
    deficit = max(
        (max(-r["ball_surface_clearance_m"], 0.0) for r in rows if r["within_net_width"]),
        default=0.0,
    )
    # One residual per flight keeps objective dimension fixed as crossings appear
    # or disappear. Maximum, not a sum whose scale depends on sampling density.
    return deficit / scale_m, dict(
        scale_m=scale_m,
        maximum_penetration_m=deficit,
        crossings=rows,
        sampling_hz=240,
        residual_loss="quadratic_not_robustified",
        expected_net_contacts=0,
        scope="optimization guidance only; unchanged physical audit still required",
    )
