"""Bound post-tape velocity from outgoing pixels. No search of the admissible set.

Ball rows (labelled or automatic) between the cord and the next event, plus a
paired streak when one is present, fix an ellipsoid of velocities. The physics
box is applied later as a clip. Missing, short, ill-conditioned, or disagreeing
evidence abstains: the caller keeps today's tape clip and does not guess.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import camera_geometry
from cv.pipeline import net_cord_response as cord
from cv.pipeline.physics_knot_solver import NET_Y, net_tape_height
from physics import flight as aero

MIN_BALL_ROWS = 3
CONDITION_LIMIT = 1e8
FRAME_MATCH = 2.0
POST_HALF_WIDTH_M = 6.4
POST_CENTRE_X_M = 5.485
GRAVITY = float(aero.default_params["g"])


def abstain(net_frame: float, reason: str, h_tol: float, **fields) -> cord.EvidenceBound:
    return cord.EvidenceBound(
        net_frame=float(net_frame),
        admitted=False,
        reason=reason,
        h_tol_m=float(h_tol),
        **fields,
    )


def _plane_contact(pixel, camera, h_tol: float):
    """Ray through one undistorted pixel, met with the net plane. None if singular."""
    pix = np.asarray(pixel, float)
    matrix = np.asarray(camera, float)
    if pix.shape != (2,) or matrix.shape != (3, 4) or not np.isfinite(np.r_[pix, matrix.ravel()]).all():
        return None
    equations = matrix[:2] - pix[:, None] * matrix[2]
    design = equations[:, [0, 2]]
    if abs(float(np.linalg.det(design))) < 1e-10:
        return None
    target = -equations[:, 1] * NET_Y - equations[:, 3]
    try:
        lateral, height = np.linalg.solve(design, target)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite([lateral, height]).all():
        return None
    if abs(float(lateral) - POST_CENTRE_X_M) > POST_HALF_WIDTH_M:
        return None
    tape = float(net_tape_height(float(lateral)))
    lo, hi = cord.tape_band(float(lateral), h_tol)
    if not lo <= float(height) <= hi:
        return None
    return np.array([float(lateral), NET_Y, tape], float)


def _rows(samples, source: str):
    return [row for row in samples if row["source"] == source]


def _linear_velocity(contact, t0, samples, fps: float):
    frames = np.asarray([row["frame"] for row in samples], float)
    times = (frames - float(t0)) / float(fps)
    if np.any(times <= 1e-4) or not np.isfinite(times).all():
        return None
    cameras = np.asarray([row["camera"] for row in samples], float)
    pixels = np.asarray([row["pixel"] for row in samples], float)
    origins = np.repeat(np.asarray(contact, float)[None], len(samples), axis=0)
    origins[:, 2] -= 0.5 * GRAVITY * times**2
    equations = cameras[:, :2] - pixels[:, :, None] * cameras[:, 2:3]
    design = (equations[:, :, :3] * times[:, None, None]).reshape(-1, 3)
    target = -np.einsum("nij,nj->ni", equations, np.c_[origins, np.ones(len(samples))]).ravel()
    try:
        velocity, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if int(rank) != 3 or singular.size < 3 or not np.isfinite(velocity).all():
        return None
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        return None
    return np.asarray(velocity, float)


def _pixel_error(contact, t0, samples, fps: float, velocity):
    frames = np.asarray([row["frame"] for row in samples], float)
    times = (frames - float(t0)) / float(fps)
    xyz = np.asarray(contact, float) + times[:, None] * np.asarray(velocity, float)
    xyz[:, 2] -= 0.5 * GRAVITY * times**2
    cameras = np.asarray([row["camera"] for row in samples], float)
    radial = None
    if any(row.get("radial") is not None for row in samples):
        radial = np.asarray(
            [
                np.zeros(3) if row.get("radial") is None else row["radial"]
                for row in samples
            ],
            float,
        )
    predicted = camera_geometry.project(cameras, xyz, radial)
    pixels = np.asarray([row["pixel"] for row in samples], float)
    return (predicted - pixels).reshape(len(samples), 2)


def _jacobian(contact, t0, samples, fps: float, velocity):
    step = 1e-3
    columns = []
    for axis in range(3):
        nudge = np.zeros(3)
        nudge[axis] = step
        forward = _pixel_error(contact, t0, samples, fps, velocity + nudge)
        backward = _pixel_error(contact, t0, samples, fps, velocity - nudge)
        columns.append(((forward - backward) / (2 * step)).ravel())
    return np.column_stack(columns)


def derive_outgoing_bound(
    *,
    net_frame: float,
    contact_pixel,
    contact_camera,
    contact_radial=None,
    outgoing: list[dict],
    fps: float,
    incoming_velocity,
    h_tol: float,
    pixel_tol_px: float = cord.PIXEL_TOL_PX,
) -> cord.EvidenceBound:
    """One linear solve and one Jacobian. Abstain rather than guess."""
    frame = float(net_frame)
    width = float(h_tol)
    rows = _rows(outgoing, "ball_row")
    if len(rows) == 0:
        return abstain(frame, "no_outgoing_observations", width)
    if len(rows) < MIN_BALL_ROWS:
        return abstain(frame, "too_few_observations", width, observation_count=len(rows))
    incoming = np.asarray(incoming_velocity, float)
    if incoming.shape != (3,) or not np.isfinite(incoming).all() or float(np.linalg.norm(incoming)) < 1e-6:
        return abstain(frame, "incoming_velocity_unavailable", width, observation_count=len(rows))
    pixel = np.asarray(contact_pixel, float)
    if contact_radial is not None:
        pixel = camera_geometry.undistort(pixel.reshape(1, 2), np.asarray(contact_radial, float).reshape(1, -1))[0]
    contact = _plane_contact(pixel, contact_camera, width)
    if contact is None:
        return abstain(frame, "no_contact_ray", width, observation_count=len(rows), sources=("ball_row",))
    prepared = []
    for row in rows:
        sample_pixel = np.asarray(row["pixel"], float)
        radial = row.get("radial")
        if radial is not None:
            sample_pixel = camera_geometry.undistort(
                sample_pixel.reshape(1, 2), np.asarray(radial, float).reshape(1, -1)
            )[0]
        prepared.append({**row, "pixel": sample_pixel, "radial": None})
    velocity = _linear_velocity(contact, frame, prepared, fps)
    if velocity is None:
        return abstain(
            frame, "ill_conditioned", width, observation_count=len(rows), sources=("ball_row",)
        )
    error = _pixel_error(contact, frame, prepared, fps, velocity)
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    if not np.isfinite(rms) or rms > float(pixel_tol_px):
        return abstain(
            frame,
            "ambiguous_residual",
            width,
            observation_count=len(rows),
            sources=("ball_row",),
            pixel_rms_px=rms,
        )
    jacobian = _jacobian(contact, frame, prepared, fps, velocity)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular.size < 3 or singular[-1] <= 0 or float(singular[0] / singular[-1]) > CONDITION_LIMIT:
        return abstain(
            frame,
            "ill_conditioned",
            width,
            observation_count=len(rows),
            sources=("ball_row",),
            pixel_rms_px=rms,
        )
    _u, values, axes = np.linalg.svd(jacobian, full_matrices=False)
    budget = float(pixel_tol_px) ** 2 * len(prepared) - float(np.sum(error**2))
    if budget <= 0:
        return abstain(
            frame,
            "ambiguous_residual",
            width,
            observation_count=len(rows),
            sources=("ball_row",),
            pixel_rms_px=rms,
        )
    bound = cord.EvidenceBound(
        net_frame=frame,
        admitted=True,
        reason="outgoing_pixels",
        h_tol_m=width,
        observation_count=len(prepared),
        sources=("ball_row",),
        pixel_rms_px=rms,
        v_center=tuple(float(v) for v in velocity),
        singular_values=tuple(float(v) for v in values),
        axes=tuple(tuple(float(v) for v in row) for row in axes),
        tau=budget,
    )
    streaks = _rows(outgoing, "streak")
    if len(streaks) >= MIN_BALL_ROWS:
        streak_rows = []
        for row in streaks:
            sample_pixel = np.asarray(row["pixel"], float)
            radial = row.get("radial")
            if radial is not None:
                sample_pixel = camera_geometry.undistort(
                    sample_pixel.reshape(1, 2), np.asarray(radial, float).reshape(1, -1)
                )[0]
            streak_rows.append({**row, "pixel": sample_pixel})
        streak_velocity = _linear_velocity(contact, frame, streak_rows, fps)
        if streak_velocity is not None and not cord.in_pixel_set(bound, streak_velocity):
            return abstain(
                frame,
                "sources_disagree",
                width,
                observation_count=len(prepared),
                sources=("ball_row", "streak"),
                pixel_rms_px=rms,
            )
        if streak_velocity is not None:
            bound = cord.EvidenceBound(
                net_frame=bound.net_frame,
                admitted=True,
                reason="outgoing_pixels",
                h_tol_m=width,
                observation_count=bound.observation_count,
                sources=("ball_row", "streak"),
                pixel_rms_px=rms,
                v_center=bound.v_center,
                singular_values=bound.singular_values,
                axes=bound.axes,
                tau=bound.tau,
            )
    boxed = cord.project_to_box(incoming, np.asarray(bound.v_center, float))
    if not cord.in_pixel_set(bound, boxed):
        return abstain(
            frame,
            "pixel_set_misses_physics_box",
            width,
            observation_count=bound.observation_count,
            sources=bound.sources,
            pixel_rms_px=rms,
        )
    return bound


def _incoming_velocity(scene, parameters, flight: int):
    count = len(scene.pixels)
    body = np.asarray(parameters, float)
    if len(body) in (5 + 3 * count, 5 + 6 * count):
        body = body[:-2]
    if len(body) not in (3 + 3 * count, 3 + 6 * count):
        return None
    velocity = body[3 + 3 * flight : 6 + 3 * flight]
    if velocity.shape != (3,):
        return None
    return velocity


def _streak_index(streaks):
    indexed = {}
    for frame, pixel in (streaks or {}).items():
        indexed[int(frame)] = np.asarray(pixel, float)
    return indexed


def bounds_from_scene(scene, bounces, parameters, streaks, h_tol: float) -> list:
    """One bound per declared net hit. No net, or no usable tail, abstains."""
    if scene.net_hit_frames is None:
        return []
    tips = _streak_index(streaks)
    bounds = []
    for flight, nets in enumerate(scene.net_hit_frames):
        if len(nets) == 0:
            continue
        net_frame = float(np.asarray(nets, float)[0])
        frames = np.asarray(scene.observation_frames[flight], float)
        later = []
        if bounces is not None and flight < len(bounces):
            later = [
                float(frame)
                for frame in np.atleast_1d(np.asarray(bounces[flight], float)).ravel()
                if float(frame) > net_frame + 1e-6
            ]
        end = float(scene.contact_frames[flight + 1])
        next_event = min(later) if later else end
        if not frames.size:
            bounds.append(abstain(net_frame, "no_outgoing_observations", h_tol))
            continue
        contact_index = int(np.argmin(np.abs(frames - net_frame)))
        if abs(float(frames[contact_index]) - net_frame) > FRAME_MATCH:
            bounds.append(abstain(net_frame, "no_contact_ray", h_tol))
            continue
        if next_event <= net_frame:
            bounds.append(abstain(net_frame, "interval_empty", h_tol))
            continue
        distortion = None if scene.camera_distortion is None else scene.camera_distortion[flight]
        outgoing = []
        for index, frame in enumerate(frames):
            if not net_frame + 1e-6 < float(frame) < next_event - 1e-6:
                continue
            radial = None if distortion is None else distortion[index]
            outgoing.append(
                {
                    "frame": float(frame),
                    "pixel": np.asarray(scene.pixels[flight][index], float),
                    "camera": np.asarray(scene.cameras[flight][index], float),
                    "radial": None if radial is None else np.asarray(radial, float),
                    "source": "ball_row",
                }
            )
            tip = tips.get(int(round(float(frame))))
            if tip is not None and np.isfinite(tip).all():
                outgoing.append(
                    {
                        "frame": float(frame),
                        "pixel": tip,
                        "camera": np.asarray(scene.cameras[flight][index], float),
                        "radial": None if radial is None else np.asarray(radial, float),
                        "source": "streak",
                    }
                )
        contact_radial = None if distortion is None else distortion[contact_index]
        incoming = _incoming_velocity(scene, parameters, flight)
        bounds.append(
            derive_outgoing_bound(
                net_frame=net_frame,
                contact_pixel=np.asarray(scene.pixels[flight][contact_index], float),
                contact_camera=np.asarray(scene.cameras[flight][contact_index], float),
                contact_radial=None if contact_radial is None else np.asarray(contact_radial, float),
                outgoing=outgoing,
                fps=float(scene.fps),
                incoming_velocity=np.zeros(3) if incoming is None else incoming,
                h_tol=float(h_tol),
            )
        )
        if incoming is None and bounds[-1].admitted:
            bounds[-1] = abstain(net_frame, "incoming_velocity_unavailable", h_tol)
    return bounds


def attach_constrained_response(scene, solved, bounds) -> dict:
    """Store the clipped outgoing velocity. Does not run the admissible-set search."""
    records = [bound.to_dict() for bound in bounds]
    updated = {**solved, "net_cord_evidence": records}
    if not any(bound.admitted for bound in bounds):
        return updated
    from cv.experiments.connected_shooting import model
    from cv.experiments.connected_shooting import observation_net_seed

    try:
        with observation_net_seed.response_context(solved):
            flights = model.chain(scene, np.asarray(solved["parameters"], float))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return updated
    for flight in flights:
        hits = flight.get("net_hits") or []
        if len(hits) != 1:
            continue
        outgoing = np.asarray(hits[0]["v_out"], float)
        if not np.isfinite(outgoing).all():
            continue
        updated["net_response"] = {
            "model": "evidence_bound_net_response_v1",
            "outgoing_velocity_mps": outgoing.tolist(),
            "searches_v_out": False,
            "net_cord_response": cord.evidence_witness(
                next(bound for bound in bounds if bound.admitted), hits[0]
            ),
        }
        break
    return updated
