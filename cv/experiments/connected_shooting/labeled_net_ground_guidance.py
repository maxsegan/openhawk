"""Continuous terminal timing guidance across a numerical simulation horizon.

An explicit labeled net experiment. Forecasting a physical impact beyond the
scored horizon supplies optimization guidance only; it adds no image or event
observation and does not alter the score. The former missing-height/time switch
could jump at the horizon and trap a fit before its first ground impact.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import event_constraints, net_collision


def first_impact(scene, parameters, flight, ordinal=0):
    """Find an ordered physical impact, with bounded numerical continuation if needed."""
    if len(flight["bounces"]) > ordinal:
        return flight["bounces"][ordinal], float(flight["end_frame"])
    n = len(scene.pixels)
    p = np.asarray(parameters)
    theta = np.r_[
        flight["start_xyz"], p[3 + 3 * (n - 1) : 3 + 3 * n], p[3 + 3 * n + 3 * (n - 1) : 3 + 6 * n]
    ]
    start = float(scene.contact_frames[-2])

    def simulate_to(end):
        return net_collision.simulate(
            theta,
            start,
            np.array([start, end]),
            scene.fps,
            scene.surface,
            net_frame=float(scene.net_hit_frames[-1][0]),
            bounce_profile=scene.bounce_profile,
            rebound_scales=p[-2:],
        )

    last_supported = float(scene.contact_frames[-1])
    for extension in [0.25, 0.5, 1.0, 2.0]:
        end = float(scene.contact_frames[-1] + extension * scene.fps)
        try:
            result = simulate_to(end)
        except ValueError as error:
            if str(error) != "measured dynamics bounce cap reached":
                raise
            # The requested first/second impact can precede a much later bounce
            # cascade. Shorten numerical queries rather than change that cap.
            low, high = last_supported, end
            for _ in range(24):
                middle = (low + high) / 2
                try:
                    result = simulate_to(middle)
                except ValueError as trial_error:
                    if str(trial_error) != "measured dynamics bounce cap reached":
                        raise
                    high = middle
                    continue
                if len(result[3]) > ordinal:
                    return result[3][ordinal], middle
                low = middle
            raise ValueError("terminal impact forecast cannot bracket before bounce cap") from error
        if len(result[3]) > ordinal:
            return result[3][ordinal], end
        last_supported = end
    raise ValueError("terminal timing forecast has no impact within two extra seconds")


def terminal_bounce_slot(evidence):
    """Locate the first terminal bounce residual in the existing evidence layout."""
    count = 0
    for e in evidence[:-1]:
        count += len(e["expected_bounce_frames"]) + 1
        count += sum(k in e for k in ["net_clearance", "terminal_observation", "terminal_rebound"])
        count += 4 if "net_collision" in e else int("unmodeled_net_clearance" in e)
    return count


def apply(scene, parameters, native, residuals, evidence, interval, uncertainty_frames=2.0):
    """Keep the original timing residual and interval guidance on one time scale."""
    expected = evidence[-1]["expected_bounce_frames"]
    if len(expected) not in (1, 2) or not len(scene.net_hit_frames[-1]):
        raise ValueError("one or two supplied terminal ground impacts after a net required")
    intervals = np.asarray(interval, float).reshape(-1, 2)
    if intervals.shape != (len(expected), 2) or not np.isfinite(intervals).all():
        raise ValueError("one finite original interval per ordered terminal impact required")
    if any(not lo <= frame <= hi for (lo, hi), frame in zip(intervals, expected, strict=True)):
        raise ValueError("original interval must contain its supplied impact epoch")
    r = np.array(residuals, copy=True)
    extras, timings = [], []
    slot = terminal_bounce_slot(evidence)
    for ordinal, (target, (low, high)) in enumerate(zip(expected, intervals, strict=True)):
        impact, horizon = first_impact(scene, parameters, native[-1], ordinal)
        frame = float(impact["frame"])
        delta = frame - target
        r[slot + ordinal] = (
            np.sign(delta)
            * max(abs(delta) - uncertainty_frames, 0)
            / event_constraints.TIME_SCALE_FRAMES
        )
        extras.extend([(min(frame - low, 0) + max(frame - high, 0)) / 0.1, 0.0])
        timings.append(
            dict(
                frame=frame,
                forecast_horizon=horizon,
                scored_horizon=float(scene.contact_frames[-1]),
                numerical_queries_only=True,
                original_observations_and_scoring_unchanged=True,
            )
        )
    if len(timings) == 1:
        evidence[-1]["numerical_terminal_timing"] = timings[0]
    else:
        evidence[-1]["numerical_terminal_timings"] = timings
    return native, np.r_[r, extras], evidence
