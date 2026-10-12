"""Directional visible-extent support over connected physical exposure paths.

Research observation operator, not a radiance renderer or an automatic detector.
Includes both contact wings and bounce/dwell knots. Uses a linearized projected
sphere plus isotropic blur. Supplied image axes define silhouette extrema; at a
turn these need not be temporal shutter-open/close tips. Unsupported point-boundary
coverage stays explicit: no clipped exposure is returned as complete evidence.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, model, oracle_benchmark as oracle


def windows(scene, duration_frames, open_offset_frames, samples=9):
    scene.validate()
    if (
        isinstance(duration_frames, (bool, np.bool_))
        or isinstance(open_offset_frames, (bool, np.bool_))
        or not np.isfinite([duration_frames, open_offset_frames]).all()
        or not 0 < duration_frames <= 1
        or not -0.5 <= open_offset_frames <= 0.5
        or type(samples) is not int
        or samples < 3
    ):
        raise ValueError("explicit finite exposure duration/offset and >=3 samples required")
    rows, seen = [], set()
    for i, frames in enumerate(scene.observation_frames):
        for j, frame in enumerate(frames):
            if frame in seen:
                raise ValueError("native exposure cannot belong to two flight groups")
            seen.add(frame)
            lo, hi = (
                float(frame + open_offset_frames),
                float(frame + open_offset_frames + duration_frames),
            )
            covered = lo >= scene.contact_frames[0] and hi <= scene.contact_frames[-1]
            rows.append(
                dict(
                    frame=int(frame),
                    flight=i,
                    index=j,
                    bounds_frames=[lo, hi],
                    status="covered" if covered else "unsupported_point_boundary",
                    sample_frames=np.unique(
                        np.r_[
                            np.linspace(lo, hi, samples),
                            scene.contact_frames[
                                (scene.contact_frames >= lo) & (scene.contact_frames <= hi)
                            ],
                        ]
                    ),
                )
            )
    return rows


def fitted_curves(scene, parameters, duration_frames, open_offset_frames, *, samples=9, cache=None):
    rows = windows(scene, duration_frames, open_offset_frames, samples)
    active = [r for r in rows if r["status"] == "covered"]
    all_times = (
        np.unique(np.concatenate([r["sample_frames"] for r in active])) if active else np.array([])
    )

    def propagate(times):
        groups = tuple(
            np.unique(np.r_[a, times[(times >= a) & (times <= b)], b])
            for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
        )
        flights = model.chain(scene, parameters, query_frames=groups, simulation_cache=cache)
        positions = {
            float(t): x
            for g, f in zip(groups, flights, strict=True)
            for t, x in zip(g, f["positions"], strict=True)
        }
        return flights, positions

    flights, positions = propagate(all_times)
    knots = np.array(
        [
            t
            for f in flights
            for b in f["bounces"]
            for t in (b["frame"], b["frame"] + b.get("dwell_seconds", 0) * scene.fps)
        ]
    )
    knots = knots[(knots >= scene.contact_frames[0]) & (knots <= scene.contact_frames[-1])]
    if len(knots):
        _, positions = propagate(np.unique(np.r_[all_times, knots]))
    for row in rows:
        if row["status"] != "covered":
            row["positions"] = None
            continue
        lo, hi = row["bounds_frames"]
        row["sample_frames"] = np.unique(
            np.r_[row["sample_frames"], knots[(knots >= lo) & (knots <= hi)]]
        )
        row["positions"] = np.array([positions[float(t)] for t in row["sample_frames"]])
    return rows


def truth_curves(scene, paths, duration_frames, open_offset_frames, *, samples=9):
    """Read the recorded piecewise interpolant; never extrapolate or refit truth."""
    if len(paths) != len(scene.contact_frames) - 1:
        raise ValueError("truth flight inventory mismatch")
    for i, path in enumerate(paths):
        frames, xyz = np.asarray(path["frames"]), np.asarray(path["positions"])
        if (
            frames.ndim != 1
            or len(frames) < 2
            or xyz.shape != (len(frames), 3)
            or not np.isfinite(frames).all()
            or not np.isfinite(xyz).all()
            or np.any(np.diff(frames) <= 0)
            or not np.array_equal(frames[[0, -1]], scene.contact_frames[i : i + 2])
        ):
            raise ValueError("finite connected recorded truth must cover exact flight bounds")
        if i and not np.allclose(paths[i - 1]["positions"][-1], xyz[0], atol=1e-8, rtol=0):
            raise ValueError("truth contact has a position discontinuity")
    rows = windows(scene, duration_frames, open_offset_frames, samples)
    dense = np.unique(np.concatenate([p["frames"] for p in paths]))
    for row in rows:
        row["positions"] = None
        if row["status"] != "covered":
            continue
        lo, hi = row["bounds_frames"]
        ts = np.unique(np.r_[row["sample_frames"], dense[(dense >= lo) & (dense <= hi)]])
        xyz = np.empty((len(ts), 3))
        for i, path in enumerate(paths):
            a, b = scene.contact_frames[i : i + 2]
            selected = (ts >= a) & (ts <= b)
            if selected.any():
                xyz[selected] = oracle.sample(path, ts[selected])
        row.update(sample_frames=ts, positions=xyz)
    return rows


def directional_tips(
    xyz, camera, axis, *, blur_radius_px=0.0, ball_radius_m=model.R_BALL, radial=None
):
    """Return min/max support points of the sampled swept projected ellipsoids.

    The selected extrema may occur inside the exposure, not at its ends. A sharp
    physical turn is not replaced by a straight open-to-close chord. The axis is
    an explicit image measurement/condition, not inferred from fitted XYZ here.
    """
    xyz, camera, axis = np.asarray(xyz, float), np.asarray(camera, float), np.asarray(axis, float)
    if (
        xyz.ndim != 2
        or xyz.shape[1:] != (3,)
        or not len(xyz)
        or camera.shape != (3, 4)
        or axis.shape != (2,)
        or not np.isfinite(np.r_[xyz.ravel(), camera.ravel(), axis]).all()
        or np.linalg.norm(axis) < 1e-9
        or not np.isfinite([blur_radius_px, ball_radius_m]).all()
        or min(blur_radius_px, ball_radius_m) < 0
    ):
        raise ValueError(
            "finite sampled curve, pinhole camera, nonzero image axis and extents required"
        )
    direction = axis / np.linalg.norm(axis)
    cameras = np.repeat(camera[None], len(xyz), axis=0)
    centers = camera_geometry.project(cameras, xyz)
    denominator = np.c_[xyz, np.ones(len(xyz))] @ camera[2]
    if np.any(abs(denominator) <= ball_radius_m * np.linalg.norm(camera[2, :3])):
        raise ValueError("projected ball intersects camera plane")
    jacobian = (camera[None, :2, :3] - centers[:, :, None] * camera[None, 2:3, :3]) / denominator[
        :, None, None
    ]
    if radial is not None:
        radial = np.asarray(radial, float)
        if radial.shape != (3,) or not np.isfinite(radial).all():
            raise ValueError("one finite native radial-camera row per exposure required")
        if radial[0] != 0.0:
            rows = np.repeat(radial[None], len(xyz), axis=0)
            # The existing first-order sphere approximation now differentiates
            # the full native projection. Blur remains in native image pixels.
            jacobian = np.einsum(
                "nij,njk->nik", camera_geometry.distortion_jacobian(centers, rows), jacobian
            )
            centers = camera_geometry.distort(centers, rows)
    jt = np.einsum("nij,i->nj", jacobian, direction)
    offset = (
        ball_radius_m
        * np.einsum("nij,nj->ni", jacobian, jt)
        / np.maximum(np.linalg.norm(jt, axis=1)[:, None], 1e-30)
    )
    offset += blur_radius_px * direction
    minus, plus = centers - offset, centers + offset
    back, front = int(np.argmin(minus @ direction)), int(np.argmax(plus @ direction))
    return np.array([minus[back], plus[front]]), np.array([back, front])


def predict(scene, rows, axes, blur_radius_px):
    axes = np.asarray(axes, float)
    if axes.shape != (len(rows), 2):
        raise ValueError("one explicit image axis per original observation required")
    output = []
    for row, axis in zip(rows, axes, strict=True):
        if row["status"] != "covered":
            raise ValueError(
                f"unsupported full exposure at native frame {row['frame']}; no clipping"
            )
        tips, selected = directional_tips(
            row["positions"],
            scene.cameras[row["flight"]][row["index"]],
            axis,
            blur_radius_px=blur_radius_px,
            radial=None
            if scene.camera_distortion is None
            else scene.camera_distortion[row["flight"]][row["index"]],
        )
        output.append(tips)
    return np.array(output)
