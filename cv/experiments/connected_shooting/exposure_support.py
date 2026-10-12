"""Bounded temporal support for native point observations, research only.

An explicit symmetric interval is a sensitivity hypothesis, not measured shutter
timing. Distance to a projected path is only a necessary image-support condition:
it is not a radiance/blur/leading-edge likelihood or proof of metric accuracy.
No exposure ID, original pixel, camera, contact or physical trajectory is changed.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, model


def closest_polyline(target, curve, times):
    target, curve, times = np.asarray(target), np.asarray(curve), np.asarray(times)
    if (
        target.shape != (2,)
        or curve.shape != (len(times), 2)
        or not len(times)
        or not np.isfinite(target).all()
        or not np.isfinite(curve).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("finite ordered projected support required")
    if len(times) == 1:
        return curve[0] - target, float(times[0])
    delta = np.diff(curve, axis=0)
    length2 = np.sum(delta**2, axis=1)
    alpha = np.clip(
        np.sum((target - curve[:-1]) * delta, axis=1) / np.maximum(length2, 1e-30), 0, 1
    )
    residuals = curve[:-1] + alpha[:, None] * delta - target
    index = int(np.argmin(np.sum(residuals**2, axis=1)))
    return residuals[index], float(times[index] + alpha[index] * (times[index + 1] - times[index]))


def evaluate(scene, parameters, half_width_frames, *, samples=9, simulation_cache=None):
    """Return two residual coordinates per unchanged native observation and support details.

    Both adjacent physical flights may explain a contact exposure. Contacts and
    impacts are explicit knots so the projected polyline cannot cut their corners.
    Support at point boundaries is clipped to modeled motion and disclosed. The
    source frame's camera is held fixed within its interval; rolling shutter and
    intraframe camera motion are not modeled. Apparent ball radius is not added.
    """
    scene.validate()
    if (
        isinstance(half_width_frames, (bool, np.bool_))
        or not np.isfinite(half_width_frames)
        or not 0 <= half_width_frames <= 0.5
        or type(samples) is not int
        or samples < 3
        or samples % 2 != 1
    ):
        raise ValueError("explicit half-width within [0,0.5] and odd sample count >=3 required")
    bounds = scene.contact_frames
    observations, queries = [], []
    for i, frames in enumerate(scene.observation_frames):
        for j, frame in enumerate(frames):
            lo, hi = (
                max(bounds[0], frame - half_width_frames),
                min(bounds[-1], frame + half_width_frames),
            )
            ts = np.unique(
                np.r_[np.linspace(lo, hi, samples), frame, bounds[(bounds >= lo) & (bounds <= hi)]]
            )
            observations.append((i, j, float(frame), lo, hi, ts))
            queries.extend(ts)

    def propagate(ts):
        # Include both endpoints even for an otherwise unqueried flight.
        grouped = tuple(
            np.unique(np.r_[a, ts[(ts >= a) & (ts < b)], b]) for a, b in zip(bounds, bounds[1:])
        )
        flights = model.chain(
            scene, parameters, query_frames=grouped, simulation_cache=simulation_cache
        )
        positions = {
            float(t): xyz
            for q, flight in zip(grouped, flights, strict=True)
            for t, xyz in zip(q, flight["positions"], strict=True)
        }
        return flights, positions

    all_times = np.unique(queries)
    flights, positions = propagate(all_times)
    impacts = np.array([r["frame"] for f in flights for r in f["bounces"]])
    missing = impacts[~np.isin(impacts, all_times)]
    if len(missing):
        _, positions = propagate(np.unique(np.r_[all_times, missing]))
    residuals, rows = [], []
    for i, j, frame, lo, hi, ts in observations:
        ts = np.unique(np.r_[ts, impacts[(impacts >= lo) & (impacts <= hi)]])
        xyz = np.array([positions[float(t)] for t in ts])
        radial = (
            None
            if scene.camera_distortion is None
            else np.repeat(scene.camera_distortion[i][j : j + 1], len(ts), axis=0)
        )
        projected = camera_geometry.project(
            np.repeat(scene.cameras[i][j : j + 1], len(ts), axis=0), xyz, radial
        )
        target = scene.pixels[i][j]
        residual, best_time = closest_polyline(target, projected, ts)
        nominal = projected[np.flatnonzero(ts == frame)[0]] - target
        residuals.append(residual)
        rows.append(
            dict(
                frame=int(frame),
                original_flight=i,
                support_bounds_frames=[float(lo), float(hi)],
                support_clipped=bool(
                    lo > frame - half_width_frames or hi < frame + half_width_frames
                ),
                best_support_frame=best_time,
                best_offset_frames=best_time - frame,
                nominal_error_px=float(np.linalg.norm(nominal)),
                support_error_px=float(np.linalg.norm(residual)),
                sample_count=len(ts),
                camera_fixed_within_exposure=True,
            )
        )
    return np.asarray(residuals).ravel(), rows
