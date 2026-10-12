"""Automatic geometric anchors for Stage-6 tennis flights.

Bounce centers lie on ``z = ball_radius`` and are metric 3D measurements.  For an
ordinary net crossing, the gravity-only ray lift supplies only the two observed
frames that bracket the crossing.  Its interpolated ray/plane point is retained as
bootstrap provenance, not as an observed 3D point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BALL_RADIUS_M = 0.0325
COURT_WIDTH_M = 10.97
NET_COURT_Y_M = 11.885
NET_CENTER_HEIGHT_M = 0.914
NET_POST_HEIGHT_M = 1.07
DEFAULT_PIXEL_SIGMA = 2.0
MAX_CROSSING_GAP_FRAMES = 6
# The emitted frame of an impact is an image, and the impact itself is sub-frame.  The ball
# dwells 4-5 ms on the court and on the strings (``physics.bounce_reference.DWELL_SECONDS``),
# so the frame that shows the impact shows the ball AT the impact position -- but at 25-60 fps
# the impact instant is between frames, and the emitter can only name the frame it saw.  The
# quantisation alone is uniform over one frame, whose standard deviation is 1/sqrt(12) = 0.289
# frames; ``docs/wk1/s6_bench.md`` measures the real emitter's own timing error over 518 matched
# contacts at an absolute mean of 0.307 frames with a p90 of 1.0 and a maximum of 2.0.  Combined
# in quadrature that is 0.42 frames, and one whole frame either side of the emission bounds it
# for everything but the measured 2.0-frame tail.
ANCHOR_TIME_SIGMA_FRAMES = 0.42
ANCHOR_TIME_BOUND_FRAMES = 1.0
# The floor on a witnessed impact-time prior.  ``docs/wk1/subframe_timing.md`` measures the
# corner witness claiming 0.020 frames against a robust residual spread of 0.025 on the bench
# clean rung, so a claimed sigma is right to about a factor of one; 0.05 frames is twice the
# measured spread and stops one over-confident corner from pinning the impact time hard.
SUBFRAME_TIME_PRIOR_MIN_SIGMA_FRAMES = 0.05
# The narrowest search window a witnessed prior may leave the fitter.  The prior does the work
# through the residual row; the bounds only stop the optimiser wandering a whole frame away.
SUBFRAME_TIME_PRIOR_MIN_BOUND_FRAMES = 0.25
# A prior built from the dwell witness alone is the emitted frame restated -- its mean is half a
# dwell early and its measured sigma is 0.37 frames against the uniform window's 0.42 -- so it is
# treated as an abstention and the uniform window stands.
SUBFRAME_TIME_PRIOR_TRIVIAL_WITNESSES = frozenset({"dwell"})
# How much wider than its own claim a witnessed impact-time prior has to be before it covers the
# truth 90% of the time.  Measured on the honest bench's clean rung by
# ``processed/wk3_pointfit9/analysis/timing_sigma.py``, over exactly the rows on which this
# function forms a prior -- the kinematic, kinematic_court and dwell witnesses, with a dwell-only
# answer treated as an abstention -- against the generator's own fractional impact times.
#
#   bounce, 576 witnessed priors: error median 0.006 frames, p90 0.020; claimed sigma median
#     0.020; |error| / sigma at the p90 is 0.77, so the claim is already about a third wider
#     than a 90% interval, and ``SUBFRAME_TIME_PRIOR_MIN_SIGMA_FRAMES`` widens it further -- the
#     prior's median width is the 0.05-frame floor and it covers the truth 96.7% of the time.
#     There is nothing to calibrate here and the scale stays 1.0.
#   contact, 347 witnessed priors: error median 0.067 frames, p90 0.175; claimed sigma median
#     0.092; the claim covers the truth only 79.8% of the time and |error| / sigma at the p90 is
#     1.15.  Widening by 1.15 takes the coverage to 90.2% at a median prior width of 0.105
#     frames.
SUBFRAME_TIME_PRIOR_SIGMA_SCALE = {"bounce": 1.0, "contact": 1.15}
# The floor under a court-plane corner's own positional width.  The corner's residual and its
# two sides' disagreement are the only label-free evidence about how well it is placed, and on a
# clean synthetic track both can be numerically zero, which no witness deserves.
BOUNCE_COURT_WITNESS_MIN_SIGMA_M = 0.02


def bounce_court_witness(
    event: dict,
    track: dict[int, np.ndarray],
    camera,
    fps: float | None,
) -> dict | None:
    """Where the track's own corner puts a bounce on the court plane, and how wide that is.

    This is ``cv.pipeline.subframe_timing.witness_kinematic_court``: the inbound and outbound
    ground tracks of a bounce meet *at* the bounce once the per-frame homography has taken them
    to the court plane, so their intersection is a label-free statement of the impact position
    that reads no owner click and no 3D fit.  ``docs/wk1/truth_bounce.md`` section 2 measures it
    on the bench at a median of 0.111 m from the true bounce, against 0.264 m for the ray/plane
    construction the shipped fitter used as a knot.

    The width is propagated from the two things the corner itself reports: each side's own
    residual about its fitted path, in metres on the court, and the disagreement between the two
    sides' answers for *when* the impact was, turned into metres at the sides' own speed.
    """
    if fps is None or not float(fps) > 0.0:
        return None
    if str(event.get("event_type") or "") != "bounce":
        return None
    frame = float(event["frame"])
    h_at = getattr(camera, "h_at", None)
    if h_at is None:
        return None
    rounded = int(round(frame))
    try:
        homography = {
            candidate: np.asarray(h_at(candidate), dtype=float)
            for candidate in range(rounded - 4, rounded + 5)
        }
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError):
        return None
    from cv.pipeline import subframe_timing

    try:
        witness = subframe_timing.witness_kinematic_court(
            subframe_timing.normalise_track(track), rounded, float(fps), homography
        )
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError, ZeroDivisionError):
        return None
    if witness.abstain:
        return None
    detail = witness.detail or {}
    intersection = detail.get("intersection")
    if intersection is None or len(intersection) != 2:
        return None
    xy = np.asarray(intersection, dtype=float)
    if not np.all(np.isfinite(xy)):
        return None
    residual = 0.5 * (
        float(detail.get("residual_in") or 0.0) + float(detail.get("residual_out") or 0.0)
    )
    speed = 0.5 * (float(detail.get("speed_in") or 0.0) + float(detail.get("speed_out") or 0.0))
    disagreement = float(detail.get("disagreement_frames") or 0.0)
    sigma = math.hypot(residual, 0.5 * disagreement * speed)
    if not math.isfinite(sigma):
        return None
    return {
        "xy": xy.tolist(),
        "t_subframe": float(witness.t_subframe),
        "sigma_frames": float(witness.sigma_frames),
        "sigma_m": float(max(sigma, BOUNCE_COURT_WITNESS_MIN_SIGMA_M)),
        "raw_sigma_m": float(sigma),
        "residual_m": residual,
        "speed_m_per_frame": speed,
        "disagreement_frames": disagreement,
        "source": "subframe_timing.witness_kinematic_court",
    }


def build_bounce_witnesses(
    boundary_events: list[dict],
    track: dict[int, np.ndarray],
    camera,
    fps: float | None,
) -> dict[tuple[str, float], dict]:
    """One court-plane corner witness per bounce emission of a point."""
    witnesses: dict[tuple[str, float], dict] = {}
    for event in boundary_events:
        if str(event.get("event_type") or "") != "bounce":
            continue
        key = ("bounce", float(event["frame"]))
        if key in witnesses:
            continue
        witness = bounce_court_witness(event, track, camera, fps)
        if witness is not None:
            witnesses[key] = witness
    return witnesses


def impact_time_prior(
    event: dict,
    track: dict[int, np.ndarray],
    camera,
    fps: float | None,
) -> dict | None:
    """The Gaussian prior on one impact's free sub-frame time, or ``None`` to keep the window.

    ``cv.pipeline.subframe_timing.estimate`` combines the track's image corner, the same corner
    on the court plane through the per-frame homography, and the dwell likelihood of the emitted
    frame.  ``docs/wk1/subframe_timing.md`` measures the combination at a median of 0.010 frames
    on the bench clean rung's 785 bounces against the emitted frame's own 0.333, and at 0.060
    against 0.111 on its 579 contacts.

    It never abstains, because the dwell witness answers from the emitted frame and the fps
    alone -- but a dwell-only answer *is* the emitted frame, so this returns ``None`` there and
    the fitter keeps the uniform quantisation window it has today.  The streak and audio
    witnesses are not offered: the first needs native crops decoded per frame and the second
    exists only on real broadcasts, and neither is available inside a point worker.
    """
    if fps is None or not float(fps) > 0.0:
        return None
    event_type = str(event.get("event_type") or "")
    if event_type not in ("contact", "bounce"):
        return None
    frame = float(event["frame"])
    homography = None
    h_at = getattr(camera, "h_at", None)
    if h_at is not None:
        rounded = int(round(frame))
        try:
            homography = {
                candidate: np.asarray(h_at(candidate), dtype=float)
                for candidate in range(rounded - 4, rounded + 5)
            }
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError):
            homography = None
    from cv.pipeline import subframe_timing

    try:
        estimate = subframe_timing.estimate(
            track,
            frame,
            event_type,
            float(fps),
            homography=homography,
            enabled=("kinematic", "kinematic_court", "dwell"),
        )
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError, ZeroDivisionError):
        return None
    used = tuple(estimate.used)
    if estimate.abstain or not set(used) - SUBFRAME_TIME_PRIOR_TRIVIAL_WITNESSES:
        return None
    # The witness states a sigma; whether that sigma is an interval the truth is actually inside
    # is a separate, measurable question, and the scale is what the bench answered it with.
    scale = float(SUBFRAME_TIME_PRIOR_SIGMA_SCALE.get(event_type, 1.0))
    sigma = float(
        min(
            max(scale * float(estimate.sigma_frames), SUBFRAME_TIME_PRIOR_MIN_SIGMA_FRAMES),
            ANCHOR_TIME_SIGMA_FRAMES,
        )
    )
    offset = float(
        np.clip(estimate.offset_frames, -ANCHOR_TIME_BOUND_FRAMES, ANCHOR_TIME_BOUND_FRAMES)
    )
    half = max(3.0 * sigma, SUBFRAME_TIME_PRIOR_MIN_BOUND_FRAMES)
    lower = float(max(offset - half, -ANCHOR_TIME_BOUND_FRAMES))
    upper = float(min(offset + half, ANCHOR_TIME_BOUND_FRAMES))
    return {
        "offset_frames": offset,
        "t_subframe": frame + offset,
        "sigma_frames": sigma,
        "raw_sigma_frames": float(estimate.sigma_frames),
        "sigma_scale": scale,
        "bounds_frames": [frame + lower, frame + upper],
        "witnesses_used": list(used),
        "witnesses_rejected": list(estimate.rejected),
        "source": "subframe_timing.estimate",
    }


def build_time_priors(
    boundary_events: list[dict],
    track: dict[int, np.ndarray],
    camera,
    fps: float | None,
) -> dict[tuple[str, float], dict]:
    """One prior per contact and bounce emission of a point, built once from its whole track."""
    priors: dict[tuple[str, float], dict] = {}
    for event in boundary_events:
        event_type = str(event.get("event_type") or "")
        if event_type not in ("contact", "bounce"):
            continue
        key = (event_type, float(event["frame"]))
        if key in priors:
            continue
        prior = impact_time_prior(event, track, camera, fps)
        if prior is not None:
            priors[key] = prior
    return priors


def _solve_ray_plane(
    projection: np.ndarray,
    pixel: np.ndarray,
    *,
    axis: int,
    value: float,
) -> np.ndarray | None:
    """Intersect one pinhole image ray with an axis-aligned world plane."""
    projection = np.asarray(projection, dtype=float)
    pixel = np.asarray(pixel, dtype=float)
    free = [candidate for candidate in range(3) if candidate != axis]
    u, v = pixel
    rows = []
    rhs = []
    for image_axis, coordinate in ((0, u), (1, v)):
        equation = projection[image_axis] - coordinate * projection[2]
        rows.append([equation[index] for index in free])
        rhs.append(-(equation[axis] * value + equation[3]))
    try:
        solved = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    except np.linalg.LinAlgError:
        return None
    point = np.empty(3, dtype=float)
    point[axis] = value
    point[free] = solved
    if not np.all(np.isfinite(point)):
        return None
    homogeneous = projection @ np.r_[point, 1.0]
    if abs(float(homogeneous[2])) < 1e-9:
        return None
    return point


def ray_at_height(
    projection: np.ndarray,
    pixel: np.ndarray,
    height_m: float,
) -> np.ndarray | None:
    """Return the point on an image ray at world height ``height_m``."""
    return _solve_ray_plane(projection, pixel, axis=2, value=height_m)


def ray_at_net(
    projection: np.ndarray,
    pixel: np.ndarray,
    net_y_m: float = NET_COURT_Y_M,
) -> np.ndarray | None:
    """Return the point on an image ray at the physical net plane."""
    return _solve_ray_plane(projection, pixel, axis=1, value=net_y_m)


def interpolate_track(
    track: dict[int, np.ndarray],
    frame: float,
    *,
    maximum_delta: int = 3,
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    """Interpolate an observed pixel at a possibly fractional event frame."""
    if not track:
        return None, ()
    lower_rows = [candidate for candidate in track if candidate <= frame]
    upper_rows = [candidate for candidate in track if candidate >= frame]
    lower = max(lower_rows) if lower_rows else None
    upper = min(upper_rows) if upper_rows else None
    if (
        lower is not None
        and upper is not None
        and frame - lower <= maximum_delta
        and upper - frame <= maximum_delta
    ):
        if lower == upper:
            return np.asarray(track[lower], dtype=float).copy(), (lower,)
        fraction = float((frame - lower) / (upper - lower))
        pixel = (1.0 - fraction) * np.asarray(track[lower], dtype=float) + fraction * np.asarray(
            track[upper], dtype=float
        )
        return pixel, (lower, upper)
    nearest = min(track, key=lambda candidate: abs(float(candidate) - frame))
    if abs(float(nearest) - frame) > maximum_delta:
        return None, ()
    return np.asarray(track[nearest], dtype=float).copy(), (nearest,)


def _event_pixel(
    event: dict, track: dict[int, np.ndarray]
) -> tuple[np.ndarray | None, tuple[int, ...], str]:
    location = event.get("location")
    if isinstance(location, dict):
        try:
            pixel = np.array(
                [float(location["image_x"]), float(location["image_y"])],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            pixel = np.array([np.nan, np.nan], dtype=float)
        if np.all(np.isfinite(pixel)):
            return pixel, (), "emission:location.image_x,location.image_y"
    for x_key, y_key in (("x1080", "y1080"), ("image_x", "image_y"), ("x", "y")):
        if event.get(x_key) is None or event.get(y_key) is None:
            continue
        try:
            pixel = np.array([float(event[x_key]), float(event[y_key])], dtype=float)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(pixel)):
            return pixel, (), f"emission:{x_key},{y_key}"
    pixel, source_frames = interpolate_track(track, float(event["frame"]))
    return pixel, source_frames, "track_interpolation"


def contact_ray_measurement(
    event: dict,
    track: dict[int, np.ndarray],
    camera,
    *,
    time_prior: dict | None = None,
) -> dict | None:
    """Record the directly observed ray for one racket-contact emission.

    A contact has no metric depth before the adjacent flights are fit. Keeping the
    camera centre and unit ray in the anchor artifact makes that distinction explicit:
    image position is observed, while depth and physical boundary time remain unknown.
    """
    if event.get("event_type") != "contact":
        return None
    pixel, source_frames, source = _event_pixel(event, track)
    if pixel is None:
        return None
    observation_frame = float(event.get("observation_frame", event["frame"]))
    geometry = _ray_geometry(camera.p_at(observation_frame), pixel)
    if geometry is None:
        return None
    center, direction = geometry
    quality = camera.quality_at(observation_frame)
    return {
        "type": "contact_ray",
        "event_frame": float(event["frame"]),
        "observation_frame": observation_frame,
        "boundary_offset_bounds_frames": (
            [
                float(time_prior["bounds_frames"][0]) - float(event["frame"]),
                float(time_prior["bounds_frames"][1]) - float(event["frame"]),
            ]
            if time_prior is not None
            else [-1.0, 1.0]
        ),
        "time_prior_offset_frames": (
            float(time_prior["offset_frames"]) if time_prior is not None else 0.0
        ),
        "time_prior_sigma_frames": (
            float(time_prior["sigma_frames"]) if time_prior is not None else None
        ),
        "time_prior": time_prior,
        "source_frames": [int(value) for value in source_frames],
        "image_xy": pixel.tolist(),
        "camera_center_xyz": center.tolist(),
        "ray_direction_xyz": direction.tolist(),
        "camera_reliable": bool(quality.get("reliable", False)),
        "camera_confidence": float(quality.get("confidence", 0.0) or 0.0),
        "camera_source": str(quality.get("source", "unknown")),
        "event_probability": float(event.get("probability", event.get("confidence", 0.0)) or 0.0),
        "source": source,
    }


def _camera_uncertainty_scale(camera_quality: dict) -> float:
    confidence = float(camera_quality.get("confidence", 0.0) or 0.0)
    reliable = bool(camera_quality.get("reliable", False))
    return (1.0 if reliable else 3.0) / math.sqrt(max(confidence, 0.10))


def plane_intersection_uncertainty(
    projection: np.ndarray,
    pixel: np.ndarray,
    *,
    axis: int,
    value: float,
    pixel_sigma: float,
    camera_quality: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Propagate isotropic pixel error through a ray/plane intersection Jacobian."""
    center = _solve_ray_plane(projection, pixel, axis=axis, value=value)
    if center is None:
        return None
    columns = []
    step = 0.25
    for image_axis in range(2):
        offset = np.zeros(2, dtype=float)
        offset[image_axis] = step
        upper = _solve_ray_plane(projection, pixel + offset, axis=axis, value=value)
        lower = _solve_ray_plane(projection, pixel - offset, axis=axis, value=value)
        if upper is None or lower is None:
            return None
        columns.append((upper - lower) / (2.0 * step))
    jacobian = np.column_stack(columns)
    scaled_sigma = float(pixel_sigma) * _camera_uncertainty_scale(camera_quality)
    covariance = jacobian @ (np.eye(2) * scaled_sigma**2) @ jacobian.T
    return center, covariance


def _ray_geometry(
    projection: np.ndarray, pixel: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    matrix = np.asarray(projection, dtype=float)[:, :3]
    offset = np.asarray(projection, dtype=float)[:, 3]
    try:
        center = -np.linalg.solve(matrix, offset)
        ray = np.linalg.solve(matrix, np.r_[np.asarray(pixel, dtype=float), 1.0])
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(ray))
    if norm < 1e-12 or not np.all(np.isfinite(center)) or not np.all(np.isfinite(ray)):
        return None
    ray /= norm
    return center, ray


def _ballistic_ray_lift(
    frames: list[int],
    track: dict[int, np.ndarray],
    camera,
    *,
    metric_anchors: list[dict] | None = None,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Fast gravity-only on-ray lift used only to localize the net crossing time."""
    if len(frames) < 4:
        return None
    reference = float(frames[0])
    rows = []
    targets = []
    gravity = np.array([0.0, 0.0, -9.81])
    # The time scale cancels for the crossing frame. Use frames here because the
    # point FPS is not needed until the full physical fit.
    for depth_index, frame in enumerate(frames):
        geometry = _ray_geometry(camera.p_at(frame), track[frame])
        if geometry is None:
            return None
        center, ray = geometry
        elapsed = float(frame) - reference
        for axis in range(3):
            row = np.zeros(len(frames) + 6)
            row[depth_index] = ray[axis]
            row[len(frames) + axis] = -1.0
            row[len(frames) + 3 + axis] = -elapsed
            rows.append(row)
            # Gravity is deliberately scaled per frame squared only as a weak
            # metric regularizer; its magnitude does not affect crossing order.
            targets.append(0.5 * gravity[axis] * (elapsed / 25.0) ** 2 - center[axis])
    for metric_anchor in metric_anchors or []:
        elapsed = float(metric_anchor["frame"]) - reference
        xyz = np.asarray(metric_anchor["xyz"], dtype=float)
        weight = float(metric_anchor.get("bootstrap_weight", 20.0))
        for axis in range(3):
            row = np.zeros(len(frames) + 6)
            row[len(frames) + axis] = weight
            row[len(frames) + 3 + axis] = weight * elapsed
            rows.append(row)
            targets.append(weight * (xyz[axis] - 0.5 * gravity[axis] * (elapsed / 25.0) ** 2))
    if not metric_anchors:
        # A ray sequence without any metric point retains the familiar
        # monocular scale/sign ambiguity.  A deliberately weak, human-reach
        # height prior selects the physical branch; it does not constrain the
        # net anchor height, which is measured from the crossing ray below.
        row = np.zeros(len(frames) + 6)
        row[len(frames) + 2] = 0.25
        rows.append(row)
        targets.append(0.25 * 1.5)
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
    x0 = solution[len(frames) : len(frames) + 3]
    velocity_per_frame = solution[len(frames) + 3 : len(frames) + 6]
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(velocity_per_frame)):
        return None
    return reference, x0, velocity_per_frame


def _interpolated_projection(camera, lower: int, upper: int, fraction: float) -> np.ndarray:
    if lower == upper:
        return np.asarray(camera.p_at(lower), dtype=float)
    return (1.0 - fraction) * np.asarray(camera.p_at(lower), dtype=float) + fraction * np.asarray(
        camera.p_at(upper), dtype=float
    )


def net_tape_height(x_m: float) -> float:
    lateral = min(1.0, abs(float(x_m) - COURT_WIDTH_M / 2.0) / (COURT_WIDTH_M / 2.0))
    return NET_CENTER_HEIGHT_M + lateral * (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M)


def find_net_anchor(
    track: dict[int, np.ndarray],
    camera,
    start_frame: float,
    end_frame: float,
    *,
    pixel_sigma: float = DEFAULT_PIXEL_SIGMA,
    metric_anchors: list[dict] | None = None,
) -> dict | None:
    """Bootstrap the observed-frame bracket containing the physical net crossing."""
    frames = [
        frame for frame in sorted(track) if math.ceil(start_frame) <= frame <= math.floor(end_frame)
    ]
    lift = _ballistic_ray_lift(frames, track, camera, metric_anchors=metric_anchors)
    if lift is None:
        return None
    reference, x0, velocity_per_frame = lift
    candidates = []
    for lower, upper in zip(frames, frames[1:]):
        if upper - lower > MAX_CROSSING_GAP_FRAMES:
            continue
        model0 = x0 + velocity_per_frame * (float(lower) - reference)
        model1 = x0 + velocity_per_frame * (float(upper) - reference)
        signed0 = float(model0[1] - NET_COURT_Y_M)
        signed1 = float(model1[1] - NET_COURT_Y_M)
        if signed0 == signed1 or signed0 * signed1 > 0.0:
            continue
        fraction = float(np.clip(-signed0 / (signed1 - signed0), 0.0, 1.0))
        frame = float(lower + fraction * (upper - lower))
        pixel = (1.0 - fraction) * np.asarray(track[lower], dtype=float) + fraction * np.asarray(
            track[upper], dtype=float
        )
        projection = _interpolated_projection(camera, lower, upper, fraction)
        quality = camera.quality_at(frame)
        propagated = plane_intersection_uncertainty(
            projection,
            pixel,
            axis=1,
            value=NET_COURT_Y_M,
            pixel_sigma=pixel_sigma,
            camera_quality=quality,
        )
        if propagated is None:
            continue
        xyz, covariance = propagated
        if not (-3.0 <= xyz[0] <= COURT_WIDTH_M + 3.0 and -0.5 <= xyz[2] <= 8.0):
            continue
        candidates.append(
            {
                "type": "net_crossing",
                "frame": frame,
                "crossing_frame_bounds": [float(lower), float(upper)],
                "crossing_direction_y": (1 if signed1 > signed0 else -1),
                "source_frames": [int(lower), int(upper)],
                "image_xy": pixel.tolist(),
                "xyz": xyz.tolist(),
                "pixel_sigma": float(pixel_sigma),
                "covariance_xyz_m2": covariance.tolist(),
                "sigma_m": float(math.sqrt(max(float(np.trace(covariance)), 0.0))),
                "camera_reliable": bool(quality.get("reliable", False)),
                "camera_confidence": float(quality.get("confidence", 0.0) or 0.0),
                "camera_source": str(quality.get("source", "unknown")),
                "tape_height_m": net_tape_height(float(xyz[0])),
                "clearance_m": float(xyz[2] - net_tape_height(float(xyz[0])) - BALL_RADIUS_M),
                "source": "track_ray_ballistic_crossing",
                "geometry_role": "crossing_bracket_with_bootstrap_point",
                "xyz_is_observation": False,
                "crossing_bootstrap": (
                    "gravity_ray_lift_with_metric_anchors" if metric_anchors else "gravity_ray_lift"
                ),
            }
        )
    if not candidates:
        return None
    midpoint = 0.5 * (float(start_frame) + float(end_frame))
    return min(candidates, key=lambda row: (abs(float(row["frame"]) - midpoint), row["sigma_m"]))


def bounce_anchor(
    event: dict,
    track: dict[int, np.ndarray],
    camera,
    *,
    pixel_sigma: float = DEFAULT_PIXEL_SIGMA,
    time_prior: dict | None = None,
    court_witness: dict | None = None,
) -> dict | None:
    """Describe one bounce emission as an observation, not as a knot.

    Three things are known about a bounce and they are not the same thing.  The ball is *on the
    court plane* at the impact: that is exact.  The impact *time* is sub-frame and the emitted
    frame is the image that shows it, so the time is bounded by the emission and no tighter.
    And the emitted *pixel* is an image observation of the ball on the emitted frame -- a frame
    on which the ball may already be, or still be, above the court.

    ``xyz`` is the intersection of that pixel's ray with the court plane.  It is a seed and a
    provenance record, not a measurement of the bounce: half a frame from the impact the ball is
    0.1-0.3 m up, and a broadcast camera turns that height into two to three times as much
    horizontal miss.  ``docs/wk1/bench_frames.md`` measures this row against the synthetic
    truth at a median of 0.143 m and a p90 of 0.836 m.  ``xyz_is_observation`` is therefore
    ``False`` and the fields below say what the observation actually is.
    """
    pixel, source_frames, source = _event_pixel(event, track)
    if pixel is None:
        return None
    frame = float(event["frame"])
    quality = camera.quality_at(frame)
    propagated = plane_intersection_uncertainty(
        camera.p_at(frame),
        pixel,
        axis=2,
        value=BALL_RADIUS_M,
        pixel_sigma=pixel_sigma,
        camera_quality=quality,
    )
    if propagated is None:
        return None
    xyz, covariance = propagated
    return {
        "type": "bounce",
        "frame": frame,
        "source_frames": [int(value) for value in source_frames],
        "image_xy": pixel.tolist(),
        "xyz": xyz.tolist(),
        "xyz_is_observation": False,
        "xyz_role": "ray_plane_seed",
        "plane_z_m": BALL_RADIUS_M,
        "observation_frame": frame,
        "time_bounds_frames": (
            list(time_prior["bounds_frames"])
            if time_prior is not None
            else [frame - ANCHOR_TIME_BOUND_FRAMES, frame + ANCHOR_TIME_BOUND_FRAMES]
        ),
        "time_sigma_frames": (
            float(time_prior["sigma_frames"])
            if time_prior is not None
            else ANCHOR_TIME_SIGMA_FRAMES
        ),
        "time_prior_offset_frames": (
            float(time_prior["offset_frames"]) if time_prior is not None else 0.0
        ),
        "time_prior": time_prior,
        "court_witness": court_witness,
        "pixel_sigma": float(pixel_sigma),
        "covariance_xyz_m2": covariance.tolist(),
        "sigma_m": float(math.sqrt(max(float(np.trace(covariance)), 0.0))),
        "camera_reliable": bool(quality.get("reliable", False)),
        "camera_confidence": float(quality.get("confidence", 0.0) or 0.0),
        "camera_source": str(quality.get("source", "unknown")),
        "event_probability": float(event.get("probability", event.get("confidence", 0.0)) or 0.0),
        "source": source,
    }


def build_point_anchors(
    point_key: str,
    flight_attempts: list[dict],
    boundary_events: list[dict],
    track: dict[int, np.ndarray],
    camera,
    *,
    pixel_sigma: float = DEFAULT_PIXEL_SIGMA,
    time_priors: dict[tuple[str, float], dict] | None = None,
    bounce_witnesses: dict[tuple[str, float], dict] | None = None,
) -> dict:
    """Build truth-free bounce anchors and contact-ray witnesses for a point.

    A broadcast net crossing is not an observed metric point.  Net geometry is
    therefore deliberately absent here; fitted trajectories may be checked against
    the net only after optimization.
    """
    bounce_events = [row for row in boundary_events if row.get("event_type") == "bounce"]
    contact_events = [row for row in boundary_events if row.get("event_type") == "contact"]
    priors = time_priors or {}
    witnesses = bounce_witnesses or {}
    contact_rays = [
        ray
        for event in contact_events
        if (
            ray := contact_ray_measurement(
                event,
                track,
                camera,
                time_prior=priors.get(("contact", float(event["frame"]))),
            )
        )
        is not None
    ]
    flights = []
    for attempt in flight_attempts:
        start = float(attempt["start_frame"])
        end = float(attempt["end_frame"])
        terminal_end = bool(attempt.get("terminal_end"))
        bounces = [
            anchor
            for event in bounce_events
            if start < float(event["frame"])
            if (
                float(event["frame"]) < end
                or (terminal_end and abs(float(event["frame"]) - end) <= 1e-6)
            )
            if (
                anchor := bounce_anchor(
                    event,
                    track,
                    camera,
                    pixel_sigma=pixel_sigma,
                    time_prior=priors.get(("bounce", float(event["frame"]))),
                    court_witness=witnesses.get(("bounce", float(event["frame"]))),
                )
            )
            is not None
        ]
        anchors = sorted(bounces, key=lambda row: row["frame"])
        flights.append(
            {
                "flight_index": int(attempt["flight_index"]),
                "start_frame": start,
                "end_frame": end,
                "anchors": anchors,
                "contact_rays": [
                    ray
                    for ray in contact_rays
                    if abs(float(ray["event_frame"]) - start) <= 1e-6
                    or abs(float(ray["event_frame"]) - end) <= 1e-6
                ],
                "bounce_anchors": len(bounces),
                "net_role": "post_fit_plausibility_only",
            }
        )
    return {
        "schema": "flight_anchors_v2",
        "point": point_key,
        "pixel_sigma": float(pixel_sigma),
        "impact_time_priors": len(priors),
        "bounce_court_witnesses": len(witnesses),
        "inference_inputs": ["automatic_s5_emissions", "automatic_ball_track", "automatic_camera"],
        "human_derived_inputs": [],
        "contact_rays": contact_rays,
        "flights": flights,
    }


def write_point_anchors(root: Path, match_id: str, clip: str, artifact: dict) -> Path:
    destination = Path(root) / match_id / clip / "anchors_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return destination
