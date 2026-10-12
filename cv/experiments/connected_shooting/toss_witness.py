"""Real pre-contact toss and soft serve-contact witnesses for connected search.

Evaluation-only helpers.  A toss candidate shares the fitted serve contact
position and native event time, but has its own incoming velocity because the
racket changes velocity at contact.  Native pre-contact fronts are retained;
an explicitly supplied automatic track may fill only absent label rows.  The
release constraint uses the automatic server court position and a broad hand-
height model.  It is a real single-view witness, not independent XYZ truth.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar

from cv.experiments.connected_shooting import camera_geometry, serve_side

GRAVITY = np.array([0.0, 0.0, -9.81])
FAR_BASELINE_Y_M = 23.77


@dataclass(frozen=True)
class TossConfig:
    maximum_observations: int = 10
    minimum_observations: int = 6
    label_uncertainty_floor_px: float = 2.0
    automatic_uncertainty_px: float = 6.0
    release_depth_sigma_m: float = 0.5
    release_lateral_sigma_m: float = 0.75
    contact_depth_from_feet_sigma_m: float = 0.25
    hand_height_m: float = 1.30
    hand_height_sigma_m: float = 0.35
    maximum_weighted_rms: float = 2.5
    maximum_horizontal_launch_mps: float = 2.5
    horizontal_launch_sigma_mps: float = 0.75
    contact_sigma_floor_m: tuple[float, float, float] = (0.12, 0.20, 0.12)

    def validate(self) -> None:
        values = np.asarray(
            [
                self.label_uncertainty_floor_px,
                self.automatic_uncertainty_px,
                self.release_depth_sigma_m,
                self.release_lateral_sigma_m,
                self.contact_depth_from_feet_sigma_m,
                self.hand_height_m,
                self.hand_height_sigma_m,
                self.maximum_weighted_rms,
                self.maximum_horizontal_launch_mps,
                self.horizontal_launch_sigma_mps,
                *self.contact_sigma_floor_m,
            ],
            float,
        )
        if (
            self.minimum_observations < 4
            or self.maximum_observations < self.minimum_observations
            or not np.isfinite(values).all()
            or np.any(values <= 0)
            or len(self.contact_sigma_floor_m) != 3
        ):
            raise ValueError(
                "finite positive toss configuration and at least four pictures required"
            )


CONTACT_CONSTRAINT_CONFIG = TossConfig(maximum_observations=48, minimum_observations=4)


def contact_curve_intersection(
    observations: dict,
    outgoing_frames,
    outgoing_pixels,
    outgoing_cameras,
    contact_frame_interval: tuple[float, float],
) -> dict:
    """Intersect the forward toss trace with the backward outgoing trace in image time.

    Both traces are fitted only to native pictures on their own side of contact.  The
    quadratic outgoing trace is a local bootstrap for the drag-model fit performed by
    :func:`initialization.outgoing_contact_seed`; it does not replace that physical fit.
    Projection matrices are not interpolated (their scale is arbitrary), so the nearest
    native supported camera supplies the contact ray after the sub-frame epoch is found.
    """
    rows = observations.get("rows", [])
    if observations.get("status") != "supported" or len(rows) < 4:
        raise ValueError("four supported pre-contact toss pictures required")
    toss_frames = np.asarray([row["frame"] for row in rows], float)
    toss_pixels = np.asarray([row["pixel"] for row in rows], float)
    toss_sigma = np.asarray([row["uncertainty_px"] for row in rows], float)
    serve_frames = np.asarray(outgoing_frames, float)
    serve_pixels = np.asarray(outgoing_pixels, float)
    serve_cameras = np.asarray(outgoing_cameras, float)
    interval = np.asarray(contact_frame_interval, float)
    if (
        toss_pixels.shape != (len(toss_frames), 2)
        or toss_sigma.shape != (len(toss_frames),)
        or serve_frames.ndim != 1
        or not 3 <= len(serve_frames) <= 6
        or serve_pixels.shape != (len(serve_frames), 2)
        or serve_cameras.shape != (len(serve_frames), 3, 4)
        or interval.shape != (2,)
        or not np.isfinite(
            np.r_[
                toss_frames,
                toss_pixels.ravel(),
                toss_sigma,
                serve_frames,
                serve_pixels.ravel(),
                serve_cameras.ravel(),
                interval,
            ]
        ).all()
        or np.any(toss_sigma <= 0)
        or interval[0] > interval[1]
        or np.any(toss_frames >= interval[1] + 1e-9)
        or np.any(serve_frames < interval[0] - 1e-9)
    ):
        raise ValueError("finite ordered toss/outgoing pictures and contact bracket required")

    origin = float(np.mean(interval))

    def fit_curve(frames: np.ndarray, pixels: np.ndarray, weights=None):
        degree = min(2, len(frames) - 1)
        design = np.vander(frames - origin, degree + 1, increasing=True)
        if weights is not None:
            weighted_design = design * weights[:, None]
            weighted_pixels = pixels * weights[:, None]
        else:
            weighted_design, weighted_pixels = design, pixels
        coefficients, _, rank, singular = np.linalg.lstsq(
            weighted_design, weighted_pixels, rcond=None
        )
        if rank != degree + 1 or not np.isfinite(coefficients).all():
            raise ValueError("contact-side image curve is unidentified")
        return coefficients, float(singular[0] / singular[-1])

    toss_coefficients, toss_condition = fit_curve(toss_frames, toss_pixels, 1.0 / toss_sigma)
    serve_coefficients, serve_condition = fit_curve(serve_frames, serve_pixels)

    def evaluate(coefficients: np.ndarray, frame: float) -> np.ndarray:
        powers = np.power(frame - origin, np.arange(len(coefficients)))
        return powers @ coefficients

    def separation_squared(frame: float) -> float:
        delta = evaluate(toss_coefficients, frame) - evaluate(serve_coefficients, frame)
        return float(delta @ delta)

    if interval[0] == interval[1]:
        epoch = float(interval[0])
        optimizer = None
    else:
        optimizer = minimize_scalar(
            separation_squared,
            bounds=(float(interval[0]), float(interval[1])),
            method="bounded",
            options={"xatol": 1e-5},
        )
        epoch = float(optimizer.x)
    toss_pixel = evaluate(toss_coefficients, epoch)
    serve_pixel = evaluate(serve_coefficients, epoch)
    pixel = 0.5 * (toss_pixel + serve_pixel)
    nearest = int(np.argmin(np.abs(serve_frames - epoch)))

    def tangent(coefficients: np.ndarray) -> np.ndarray:
        if len(coefficients) == 1:
            return np.zeros(2)
        powers = np.power(epoch - origin, np.arange(len(coefficients) - 1))
        return powers @ (np.arange(1, len(coefficients))[:, None] * coefficients[1:])

    toss_tangent, serve_tangent = tangent(toss_coefficients), tangent(serve_coefficients)
    denominator = np.linalg.norm(toss_tangent) * np.linalg.norm(serve_tangent)
    image_angle = (
        None
        if denominator < 1e-12
        else float(
            np.degrees(
                np.arccos(np.clip(abs(toss_tangent @ serve_tangent) / denominator, 0.0, 1.0))
            )
        )
    )
    return {
        "schema": "connected_toss_outgoing_image_intersection_v1",
        "status": "supported",
        "contact_epoch_frame": epoch,
        "contact_frame_interval": interval.tolist(),
        "contact_pixel_native": pixel.tolist(),
        "toss_pixel_native": toss_pixel.tolist(),
        "backward_outgoing_pixel_native": serve_pixel.tolist(),
        "intersection_separation_px": float(np.sqrt(separation_squared(epoch))),
        "toss_image_tangent_px_per_frame": toss_tangent.tolist(),
        "outgoing_image_tangent_px_per_frame": serve_tangent.tolist(),
        "image_crossing_angle_degrees": image_angle,
        "contact_camera": serve_cameras[nearest].tolist(),
        "contact_camera_source_frame": float(serve_frames[nearest]),
        "toss_frames": toss_frames.tolist(),
        "outgoing_frames": serve_frames.tolist(),
        "toss_curve_condition_number": toss_condition,
        "outgoing_curve_condition_number": serve_condition,
        "optimizer_success": True if optimizer is None else bool(optimizer.success),
        "native_timestamps_changed": False,
        "candidate_generator_only": True,
    }


def observations_at_contact_epoch(observations: dict, contact_frame: float) -> dict:
    """Retain only genuine pre-contact pictures for a fitted contact epoch."""
    if not np.isfinite(contact_frame):
        raise ValueError("finite fitted contact epoch required")
    shifted = copy.deepcopy(observations)
    shifted["contact_frame"] = float(contact_frame)
    shifted["rows"] = [
        row for row in shifted.get("rows", []) if float(row["frame"]) < contact_frame
    ]
    minimum = int(shifted.get("minimum_observations", 0))
    if len(shifted["rows"]) < minimum:
        shifted["status"] = "abstained"
        shifted["abstention_reason"] = "fitted_epoch_has_insufficient_precontact_toss_observations"
    shifted["fitted_contact_epoch"] = True
    shifted["native_timestamps_changed"] = False
    return shifted


def fit_contact(
    observations: dict,
    feet: dict,
    fps: float,
    *,
    contact_frame_interval: tuple[float, float] | None = None,
    contact_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    config: TossConfig = TossConfig(),
) -> dict:
    """Infer the serve contact from one ballistic toss and the server's feet.

    The seven fitted parameters are contact XYZ, incoming velocity and elapsed
    release-to-contact time.  The release is derived by propagating backward
    from the supplied native event epoch; it is never initialized from the
    outgoing serve arc.  Direct contact parameterization lets the independent
    server/court box be an optimizer bound, while the tracked feet pin the
    derived release in court depth and lateral position.  This is still a
    single-view development witness, not independent XYZ truth.

    ``contact_sigma_m`` is the local robust-fit covariance propagated through
    the ballistic intersection, with explicit floors and the labeled contact
    bracket folded in.  Those sigmas are suitable for a soft residual and a
    conservative three-sigma parameter box; they are not acceptance accuracy.
    """
    config.validate()
    rows = observations.get("rows", [])
    contact_frame = float(observations["contact_frame"])
    if observations.get("status") == "abstained":
        return {
            "schema": "connected_toss_contact_estimate_v1",
            "status": "abstained",
            "abstention_reason": observations["abstention_reason"],
            "observation_count": len(rows),
            "minimum_observations": observations["minimum_observations"],
            "contact_frame": contact_frame,
            "contact_frame_interval": [contact_frame, contact_frame],
            "contact_xyz_m": None,
            "contact_sigma_m": None,
            "automatic_inference_eligible": False,
            "independent_xyz_truth": False,
        }
    frames = np.asarray([row["frame"] for row in rows], float)
    pixels = np.asarray([row["pixel"] for row in rows], float)
    uncertainty = np.asarray([row["uncertainty_px"] for row in rows], float)
    cameras = np.asarray([row["camera"] for row in rows], float)
    feet_xy = np.asarray(feet["court_xy_m"], float)
    interval = (
        np.asarray([contact_frame, contact_frame], float)
        if contact_frame_interval is None
        else np.asarray(contact_frame_interval, float)
    )
    if (
        not np.isfinite(fps)
        or fps <= 0
        or frames.ndim != 1
        or len(frames) < config.minimum_observations
        or pixels.shape != (len(frames), 2)
        or uncertainty.shape != (len(frames),)
        or cameras.shape != (len(frames), 3, 4)
        or feet_xy.shape != (2,)
        or interval.shape != (2,)
        or not np.isfinite(
            np.r_[frames, pixels.ravel(), uncertainty, cameras.ravel(), feet_xy]
        ).all()
        or np.any(uncertainty <= 0)
        or np.any(frames >= contact_frame)
        or not np.isfinite(interval).all()
        or interval[0] > contact_frame
        or interval[1] < contact_frame
        or interval[0] > interval[1]
    ):
        raise ValueError("finite pre-contact toss, feet, cadence and containing bracket required")

    relative_seconds = (frames - contact_frame) / fps
    minimum_release_seconds = max(float(-relative_seconds[0] + 1 / fps), 0.30)
    maximum_release_seconds = max(1.60, minimum_release_seconds + 0.05)

    def states(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contact = parameters[:3]
        incoming = parameters[3:6]
        tau = parameters[6]
        xyz = (
            contact
            + relative_seconds[:, None] * incoming
            + 0.5 * relative_seconds[:, None] ** 2 * GRAVITY
        )
        release = contact - tau * incoming + 0.5 * tau**2 * GRAVITY
        launch = incoming - tau * GRAVITY
        return xyz, release, launch

    def residual(parameters: np.ndarray) -> np.ndarray:
        xyz, release, launch = states(parameters)
        contact = parameters[:3]
        upward_low = max(2.0 - float(launch[2]), 0.0)
        upward_high = max(float(launch[2]) - 12.0, 0.0)
        return np.r_[
            (
                (camera_geometry.project(cameras, xyz, camera_geometry.rows_radial(rows)) - pixels)
                / uncertainty[:, None]
            ).ravel(),
            (release[0] - feet_xy[0]) / config.release_lateral_sigma_m,
            (release[1] - feet_xy[1]) / config.release_depth_sigma_m,
            (release[2] - config.hand_height_m) / config.hand_height_sigma_m,
            (contact[1] - feet_xy[1]) / config.contact_depth_from_feet_sigma_m,
            launch[:2] / config.horizontal_launch_sigma_mps,
            upward_low,
            upward_high,
        ]

    if contact_bounds is None:
        contact_low = np.array([-10.0, -15.0, 0.5])
        contact_high = np.array([21.0, 40.0, 12.0])
    else:
        contact_low, contact_high = (np.asarray(value, float) for value in contact_bounds)
        if (
            contact_low.shape != (3,)
            or contact_high.shape != (3,)
            or not np.isfinite(np.r_[contact_low, contact_high]).all()
            or np.any(contact_low >= contact_high)
        ):
            raise ValueError("finite increasing three-axis contact bounds required")
    seed = np.array(
        [
            np.clip(feet_xy[0], contact_low[0], contact_high[0]),
            np.clip(feet_xy[1], contact_low[1], contact_high[1]),
            np.clip(2.8, contact_low[2], contact_high[2]),
            0.0,
            0.0,
            -1.0,
            min(max(0.75, minimum_release_seconds), maximum_release_seconds),
        ]
    )
    lower = np.array(
        [
            *contact_low,
            -config.maximum_horizontal_launch_mps,
            -config.maximum_horizontal_launch_mps,
            -8.0,
            minimum_release_seconds,
        ]
    )
    upper = np.array(
        [
            *contact_high,
            config.maximum_horizontal_launch_mps,
            config.maximum_horizontal_launch_mps,
            5.0,
            maximum_release_seconds,
        ]
    )
    solved = least_squares(
        residual,
        seed,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=600,
    )
    xyz, release, launch = states(solved.x)
    contact = solved.x[:3]
    incoming = solved.x[3:6]
    predicted = camera_geometry.project(cameras, xyz, camera_geometry.rows_radial(rows))
    pixel_errors = np.linalg.norm(predicted - pixels, axis=1)
    weighted = pixel_errors / uncertainty

    # The optimizer's Jacobian is with respect to normalized image and release
    # residuals.  Propagate its local covariance to contact XYZ.  A robust fit
    # may have cost below one per degree of freedom, so never claim precision
    # tighter than the stated evidence or the explicit metric floors.
    degrees = max(len(solved.fun) - len(solved.x), 1)
    residual_scale = max(2 * float(solved.cost) / degrees, 1.0)
    parameter_covariance = np.linalg.pinv(solved.jac.T @ solved.jac) * residual_scale
    tau = float(solved.x[6])
    contact_covariance = parameter_covariance[:3, :3]
    local_sigma = np.sqrt(np.maximum(np.diag(contact_covariance), 0.0))
    half_width_seconds = float(max(contact_frame - interval[0], interval[1] - contact_frame) / fps)
    bracket_sigma = np.abs(incoming) * half_width_seconds
    sigma = np.maximum(
        np.sqrt(local_sigma**2 + bracket_sigma**2),
        np.asarray(config.contact_sigma_floor_m, float),
    )
    constraints = {
        "optimizer_converged": bool(solved.success),
        "fronts_within_declared_uncertainty": bool(
            np.sqrt(np.mean(weighted**2)) <= config.maximum_weighted_rms
        ),
        "release_depth_near_tracked_feet": bool(
            abs(release[1] - feet_xy[1]) <= config.release_depth_sigma_m
        ),
        "release_lateral_near_tracked_feet": bool(
            abs(release[0] - feet_xy[0]) <= config.release_lateral_sigma_m
        ),
        "release_at_hand_height": bool(
            abs(release[2] - config.hand_height_m) <= 2 * config.hand_height_sigma_m
        ),
        "near_vertical_launch": bool(
            np.linalg.norm(launch[:2]) <= config.maximum_horizontal_launch_mps
        ),
        "upward_release": bool(2.0 <= launch[2] <= 12.0),
        "finite_local_contact_covariance": bool(np.isfinite(sigma).all()),
    }
    required = constraints
    return {
        "schema": "connected_toss_contact_estimate_v1",
        "status": "supported" if all(required.values()) else "held",
        "constraints": constraints,
        "required_constraints": list(required),
        "diagnostic_constraints": [],
        "failures": [name for name, passed in required.items() if not passed],
        "observation_count": len(rows),
        "contact_frame": contact_frame,
        "contact_frame_interval": interval.tolist(),
        "contact_xyz_m": contact.tolist(),
        "contact_sigma_m": sigma.tolist(),
        "contact_covariance_m2": contact_covariance.tolist(),
        "contact_bracket_sigma_m": bracket_sigma.tolist(),
        "incoming_contact_velocity_mps": incoming.tolist(),
        "release_seconds_before_contact": tau,
        "release_frame": float(contact_frame - tau * fps),
        "release_xyz_m": release.tolist(),
        "release_velocity_mps": launch.tolist(),
        "contact_fit_bounds_m": [contact_low.tolist(), contact_high.tolist()],
        "tracked_feet_xy_m": feet_xy.tolist(),
        "image_rms_px": float(np.sqrt(np.mean(pixel_errors**2))),
        "weighted_image_rms": float(np.sqrt(np.mean(weighted**2))),
        "native_projection": [
            {
                "frame": int(frame),
                "source": row["source"],
                "observed_ball_centre": pixel.tolist(),
                "predicted_ball_centre": estimate.tolist(),
                "error_px": float(error),
                "uncertainty_px": float(radius),
                "xyz_m": position.tolist(),
            }
            for frame, row, pixel, estimate, error, radius, position in zip(
                frames, rows, pixels, predicted, pixel_errors, uncertainty, xyz, strict=True
            )
        ],
        "ball_centre_semantics": (
            "The midpoint of two visible labeled streak endpoints is the exposure-time ball "
            "centre; a lone visible front is retained with its explicit radius. No timestamp "
            "shift or synthetic picture."
        ),
        "sigma_semantics": (
            "Local robust-fit covariance propagated to the ballistic contact, combined with "
            "the labeled event bracket and conservative per-axis metric floors."
        ),
        "gravity_mps2": GRAVITY.tolist(),
        "physical_prior": {
            "release_depth_sigma_m": config.release_depth_sigma_m,
            "release_lateral_sigma_m": config.release_lateral_sigma_m,
            "contact_depth_from_feet_sigma_m": config.contact_depth_from_feet_sigma_m,
            "hand_height_m": config.hand_height_m,
            "hand_height_sigma_m": config.hand_height_sigma_m,
            "horizontal_launch_sigma_mps": config.horizontal_launch_sigma_mps,
            "maximum_horizontal_launch_mps": config.maximum_horizontal_launch_mps,
        },
        "optimizer": {
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": str(solved.message),
            "nfev": int(solved.nfev),
        },
        "automatic_inference_eligible": False,
        "independent_xyz_truth": False,
    }


def _automatic_rows(path: Path | None, clip: str, scale: float) -> dict[int, dict]:
    if path is None:
        return {}
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive explicit automatic-track native scale required")
    output: dict[int, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip") != clip:
                continue
            frame = int(row["frame"].removeprefix("f_").removesuffix(".jpg"))
            if frame in output:
                raise ValueError("automatic toss fallback requires one track row per frame")
            output[frame] = {
                "frame": frame,
                "pixel": [scale * float(row["x"]), scale * float(row["y"])],
                "score": float(row.get("score", "nan")),
                "source": "automatic_track_fallback",
            }
    return output


AUTOMATIC_CENTER_SOURCE = "automatic_native_center"
AUTOMATIC_CENTER_SEMANTICS = "detector_heatmap_nominal_centre"


def qualified_automatic_center(row: dict) -> bool:
    """Retain observed detector centers, including coordinate-bound native guides.

    Confidence remains soft evidence, not calibrated or thresholded here. A
    coarse-lock row is observed only when the adapter traced it to an exact
    original detector coordinate and retained the guide's input binding.
    """
    if (
        row.get("annotation_origin") != "automatic"
        or row.get("observation_semantics") != AUTOMATIC_CENTER_SEMANTICS
        or row.get("status", "visible") != "visible"
        or "interpolated" in str(row.get("automatic_sources", ""))
    ):
        return False
    guide = row.get("guide_support")
    if row.get("support_class") == "guide_observation":
        if not isinstance(guide, dict):
            return False
        sources = str(guide.get("sources", ""))
        binding = guide.get("input") or {}
        if (
            not sources
            or "interpolated" in sources
            or "derived" in sources
            or not binding.get("path")
            or not binding.get("sha256")
            # A streak-centre shift keeps the traced detector centre as raw_xy.
            or not np.array_equal(
                guide.get("xy"),
                (row.get("streak_centre") or {}).get("raw_xy")
                or [row.get("x1080"), row.get("y1080")],
            )
        ):
            return False
    elif (
        row.get("support_class") != "unresolved_or_direct_tracker_observation"
        or guide is not None
        or "coarse_lock" in str(row.get("automatic_sources", ""))
    ):
        return False
    confidence = row.get("automatic_confidence")
    return confidence is None or bool(np.isfinite(float(confidence)))


def precontact_observations(
    labels: dict,
    cameras: dict,
    *,
    contact_frame: float,
    automatic_track: Path | None = None,
    automatic_track_scale: float = 1.0,
    config: TossConfig = TossConfig(),
) -> dict:
    """Resolve the last native toss fronts without replacing a visible label."""
    config.validate()
    clip = labels["attempt"]["clip"]
    records = [row for row in labels["ball"]["records"] if row["clip"] == clip]
    if len(records) != 1:
        raise ValueError("one frozen ball record required for the toss")
    camera_rows = {int(row["frame"]): row for row in cameras["cameras"]}
    if len(camera_rows) != len(cameras["cameras"]):
        raise ValueError("unique per-frame cameras required")
    labeled = {int(row["frame"]): row for row in records[0]["frames"]}
    automatic = _automatic_rows(automatic_track, clip, automatic_track_scale)
    # Native exposure times never move. At an integer contact the same-frame
    # picture belongs to the outgoing flight, so the toss ends one frame before;
    # at a half-frame contact floor(contact) remains the last incoming picture.
    last = int(np.ceil(contact_frame)) - 1
    first = last - config.maximum_observations + 1
    rows = []
    for frame in range(first, last + 1):
        camera = camera_rows.get(frame)
        if camera is None or camera.get("status") != "supported":
            continue
        label = labeled.get(frame)
        if label is not None and label.get("status") == "visible":
            if label.get("annotation_origin") == "automatic":
                if not qualified_automatic_center(label):
                    continue
                pixel = [float(label["x1080"]), float(label["y1080"])]
                if not np.isfinite(pixel).all():
                    continue
                rows.append(
                    {
                        **copy.deepcopy(label),
                        "frame": frame,
                        "pixel": pixel,
                        "uncertainty_px": config.automatic_uncertainty_px,
                        "source": AUTOMATIC_CENTER_SOURCE,
                        "incoming_operator": "center",
                        "axis": None,
                        "camera": camera["P"],
                        **camera_geometry.radial_fields(camera),
                    }
                )
                continue
            radius = label.get("uncertainty_radius_px1080")
            rows.append(
                {
                    "frame": frame,
                    "pixel": [float(label["x1080"]), float(label["y1080"])],
                    "uncertainty_px": max(
                        config.label_uncertainty_floor_px,
                        config.label_uncertainty_floor_px if radius is None else float(radius),
                    ),
                    "source": "frozen_labeled_front",
                    "camera": camera["P"],
                    **camera_geometry.radial_fields(camera),
                }
            )
        elif frame in automatic:
            rows.append(
                {
                    **automatic[frame],
                    "uncertainty_px": config.automatic_uncertainty_px,
                    "camera": camera["P"],
                    **camera_geometry.radial_fields(camera),
                }
            )
    return {
        "clip": clip,
        "contact_frame": float(contact_frame),
        "rows": rows,
        "labeled_fronts": sum(row["source"] == "frozen_labeled_front" for row in rows),
        "automatic_fallbacks": sum(row["source"] == "automatic_track_fallback" for row in rows),
        "status": "supported" if len(rows) >= config.minimum_observations else "abstained",
        "abstention_reason": None
        if len(rows) >= config.minimum_observations
        else "insufficient_camera_supported_precontact_toss_observations",
        "minimum_observations": config.minimum_observations,
        "native_timestamps_changed": False,
        "visible_labels_replaced": 0,
    }


def contact_constraint_observations(
    document: dict,
    cameras: dict,
    *,
    clip: str,
    contact_frame: float,
    contact_frame_interval: tuple[float, float],
    config: TossConfig = CONTACT_CONSTRAINT_CONFIG,
) -> dict:
    """Read toss-only or dense labeled fronts for the contact constraint arm.

    The additive ``toss_v1`` contract stores rows under ``ball.frames``; older
    dense revisions store one or more ``ball.records[*].frames``.  Only native
    frames with an explicitly supported camera and a visible labeled front are
    admitted.  The same-frame outgoing serve picture is never reused as toss.
    """
    config.validate()
    if document.get("schema") == "s6_agent_serve_toss_labels_v1":
        source_name = "toss_v1_labeled_front"
        records = [
            {"clip": document.get("clip"), "frames": document.get("ball", {}).get("frames", [])}
        ]
        labeled_contact = document.get("contact") or {}
        if labeled_contact.get("status") == "visible":
            labeled_frame = float(labeled_contact["frame"])
            labeled_interval = tuple(map(float, labeled_contact["frame_interval"]))
            if abs(labeled_frame - contact_frame) > 1.0 or not (
                labeled_interval[0] <= contact_frame <= labeled_interval[1]
            ):
                raise ValueError("toss label contact does not contain the fitted contact epoch")
            contact_frame_interval = labeled_interval
    else:
        source_name = "dense_v1_labeled_front"
        records = document.get("records") or document.get("ball", {}).get("records", [])
    camera_rows = {int(row["frame"]): row for row in cameras["cameras"]}
    if len(camera_rows) != len(cameras["cameras"]):
        raise ValueError("unique per-frame cameras required")
    labeled: dict[int, dict] = {}
    for record in records:
        if record.get("clip", clip) != clip:
            continue
        for row in record.get("frames", []):
            frame = int(row["frame"])
            if frame in labeled:
                raise ValueError("one toss label row per native frame required")
            labeled[frame] = row
    last = int(np.ceil(contact_frame)) - 1
    first = last - config.maximum_observations + 1
    release = document.get("release") or {}
    if release.get("status") == "visible" and release.get("frame_interval"):
        first = max(first, int(np.floor(float(release["frame_interval"][0]))))
    rows = []
    for frame in range(first, last + 1):
        camera = camera_rows.get(frame)
        if camera is None or not camera.get("supported", camera.get("status") == "supported"):
            continue
        row = labeled.get(frame)
        if row is None:
            continue
        front = row.get("front") or row
        if front.get("status") != "visible":
            continue
        if front.get("x1080") is None or front.get("y1080") is None:
            continue
        front_pixel = np.asarray([front["x1080"], front["y1080"]], float)
        back = row.get("back") or {}
        back_visible = (
            back.get("status") == "visible"
            and back.get("x1080") is not None
            and back.get("y1080") is not None
        )
        if back_visible:
            back_pixel = np.asarray([back["x1080"], back["y1080"]], float)
            pixel = 0.5 * (front_pixel + back_pixel)
            radii = [
                value
                for value in (
                    front.get("uncertainty_radius_px1080"),
                    back.get("uncertainty_radius_px1080"),
                )
                if value is not None
            ]
            radius = max(radii, default=config.label_uncertainty_floor_px)
            row_source = source_name.replace("_front", "_streak_midpoint")
        else:
            pixel = front_pixel
            radius = front.get("uncertainty_radius_px1080")
            row_source = source_name
        rows.append(
            {
                "frame": frame,
                "pixel": pixel.tolist(),
                "uncertainty_px": max(
                    config.label_uncertainty_floor_px,
                    config.label_uncertainty_floor_px if radius is None else float(radius),
                ),
                "source": row_source,
                "camera": camera["P"],
                **camera_geometry.radial_fields(camera),
            }
        )
    return {
        "clip": clip,
        "contact_frame": float(contact_frame),
        "contact_frame_interval": list(map(float, contact_frame_interval)),
        "rows": rows,
        "labeled_fronts": len(rows),
        "automatic_fallbacks": 0,
        "source_schema": document.get("schema"),
        "source_extension_schema": document.get("extension_schema"),
        "status": "supported" if len(rows) >= config.minimum_observations else "abstained",
        "abstention_reason": None
        if len(rows) >= config.minimum_observations
        else "insufficient_camera_supported_labeled_toss_observations",
        "minimum_observations": config.minimum_observations,
        "native_timestamps_changed": False,
        "visible_labels_replaced": 0,
    }


def labeled_server_anchor(document: dict, cameras: dict, fallback: dict) -> dict:
    """Use toss-v1 grounded release feet when present, otherwise the automatic roots.

    Contact-time shoes marked airborne are deliberately ignored.  A visible hip
    remains reported context but is not projected to the court plane as if it
    were a foot.
    """
    if document.get("schema") != "s6_agent_serve_toss_labels_v1":
        return fallback
    landmarks = next(
        (row for row in document.get("server_landmarks", []) if row.get("event") == "release"),
        None,
    )
    if landmarks is None or landmarks.get("observation_frame") is None:
        return fallback
    frame = int(landmarks["observation_frame"])
    camera_row = next(
        (
            row
            for row in cameras["cameras"]
            if int(row["frame"]) == frame and row.get("supported", row.get("status") == "supported")
        ),
        None,
    )
    if camera_row is None:
        return fallback
    camera = np.asarray(camera_row["P"], float)
    roots = []
    pixels = []
    for name in ("foot_image_left_ground", "foot_image_right_ground"):
        foot = landmarks.get(name) or {}
        if foot.get("status") != "visible":
            continue
        pixel = np.asarray([foot["x1080"], foot["y1080"]], float)
        planes = camera[:2] - pixel[:, None] * camera[2]
        augmented = np.vstack([planes, [0.0, 0.0, 1.0, 0.0]])
        _, _, right = np.linalg.svd(augmented)
        point = right[-1]
        if abs(point[3]) < 1e-12:
            continue
        xyz = point[:3] / point[3]
        if np.isfinite(xyz).all():
            roots.append(xyz[:2])
            pixels.append(pixel.tolist())
    if not roots:
        return fallback
    hip = landmarks.get("hip") or {}
    return {
        **fallback,
        "court_xy_m": np.median(np.asarray(roots), axis=0).tolist(),
        "grounded_release_foot_count": len(roots),
        "grounded_release_foot_pixels_native": pixels,
        "release_landmark_frame": frame,
        "hip_at_release_native": (
            [float(hip["x1080"]), float(hip["y1080"])] if hip.get("status") == "visible" else None
        ),
        "automatic_court_xy_fallback_m": fallback["court_xy_m"],
        "interpretation": (
            "human-labeled grounded release feet projected to the court plane; "
            "hip is context only; automatic server roots retained as fallback evidence"
        ),
    }


SHORT_HISTORY_FRAMES = 30
# A serve toss is over in about a second.  When the tracker has too few roots in
# the ordinary window the same standing player is still detected slightly
# earlier, so the window widens once before the witness abstains.
WIDE_HISTORY_FRAMES = 90
ASSOCIATION_SEARCH_RADIUS_FRAMES = 8


def server_feet(
    pose_csv: Path,
    clip: str,
    contact_frame: float,
    contact_pixel: np.ndarray,
    *,
    image_coordinate_scale: float = 1.0,
    observation_fallback: bool = False,
    nearest_detected_frame: bool = False,
    fallback_receipt: list | None = None,
    serve_side_association: str = "centre",
) -> dict:
    """Associate the server once, then robustly summarize its pre-contact court roots.

    ``serve_side_association="serve_reach"`` associates the server with the
    same rule as the contact state (``serve_side``), so both agree on the end.

    ``observation_fallback`` widens the pre-contact window and, failing that,
    accepts a short root history with its own count recorded.  The witness is a
    soft term and never gates a branch, so refusing the whole attempt because
    the tracker holds three roots instead of four measures nothing.

    ``nearest_detected_frame`` is off unless a run asks. The contact picture is
    still preferred, then the eight-frame neighbourhood. Only a hole past that
    neighbourhood may use a later or earlier detection inside the wide window.
    A pose that is present at the contact picture is unchanged.
    """
    contact_pixel = np.asarray(contact_pixel, float)
    if not np.isfinite(image_coordinate_scale) or image_coordinate_scale <= 0:
        raise ValueError("positive finite player-image coordinate scale required")
    with pose_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("clip") == clip]
    frame = int(np.ceil(contact_frame))
    at_contact = [row for row in rows if row["frame"] == f"f_{frame:04d}.jpg"]
    association_frame = frame
    if not at_contact and observation_fallback:
        radius = WIDE_HISTORY_FRAMES if nearest_detected_frame else ASSOCIATION_SEARCH_RADIUS_FRAMES
        for offset in range(1, radius + 1):
            for candidate in (frame - offset, frame + offset):
                near = [row for row in rows if row["frame"] == f"f_{candidate:04d}.jpg"]
                if near:
                    at_contact, association_frame = near, candidate
                    break
            if at_contact:
                break
    if not at_contact:
        raise ValueError("server pose is absent at the first post-contact native picture")
    for row in at_contact:
        centre = image_coordinate_scale * np.array(
            [
                (float(row["x0"]) + float(row["x1"])) / 2,
                (float(row["y0"]) + float(row["y1"])) / 2,
            ]
        )
        row["association_distance_px"] = float(np.linalg.norm(centre - contact_pixel))
    chosen = serve_side.choose(
        at_contact,
        contact_pixel,
        image_coordinate_scale,
        serve_side_association,
        lambda row: row["association_distance_px"],
    )
    track = chosen["track_id"]

    def supported_root(row: dict) -> bool:
        # A native image box can identify the player even when its camera cannot
        # provide a court root. Such rows must not count toward the feet history.
        try:
            return bool(np.isfinite([float(row["court_x"]), float(row["court_y"])]).all())
        except (KeyError, TypeError, ValueError):
            return False

    def roots_within(span: int, same_track: bool) -> list:
        start = frame - span
        return [
            row
            for row in rows
            if row["side"] == chosen["side"]
            and (not same_track or row["track_id"] == track)
            and start <= int(row["frame"][2:6]) <= frame
            and supported_root(row)
        ]

    history_span, history_status = SHORT_HISTORY_FRAMES, "track"
    history = roots_within(SHORT_HISTORY_FRAMES, True)
    if len(history) < 4:
        history, history_status = roots_within(SHORT_HISTORY_FRAMES, False), "side"
    if len(history) < 4 and observation_fallback:
        for same_track in (True, False):
            history = roots_within(WIDE_HISTORY_FRAMES, same_track)
            history_span = WIDE_HISTORY_FRAMES
            history_status = "widened_track" if same_track else "widened_side"
            if len(history) >= 4:
                break
    if len(history) < 4 and observation_fallback and history:
        history_status = "short"
    # The sided track can have a hole over the contact picture (the raw boxes
    # are still there; the exported track is not). A detection a few frames
    # later is outside the pre-contact window, so the history above is empty
    # and the old lookup refuses the component. Use that detection's own court
    # root. A picture at the contact frame never reaches this branch.
    if nearest_detected_frame and len(history) < 1 and association_frame != frame:

        def detected_roots(same_track: bool) -> list:
            return [
                row
                for row in at_contact
                if row["side"] == chosen["side"]
                and (not same_track or row["track_id"] == track)
                and supported_root(row)
            ]

        history = detected_roots(True) or detected_roots(False)
        if history:
            history_span = abs(association_frame - frame)
            history_status = "nearest_detected_frame"
    if len(history) < 1 or (len(history) < 4 and not observation_fallback):
        raise ValueError("server feet lack a short pre-contact automatic track")
    if history_status not in {"track", "side"} and fallback_receipt is not None:
        fallback_receipt.append(
            {
                "fallback": "server_feet_widened_precontact_window",
                "contact_frame": float(contact_frame),
                "association_frame": association_frame,
                "history_status": history_status,
                "history_span_frames": history_span,
                "history_count": len(history),
            }
        )
    roots = np.asarray([[float(row["court_x"]), float(row["court_y"])] for row in history])
    return {
        "side": chosen["side"],
        "track_id": track,
        "court_xy_m": np.median(roots, axis=0).tolist(),
        "history_frames": [int(row["frame"][2:6]) for row in history],
        "history_count": len(history),
        "history_status": history_status,
        "history_span_frames": history_span,
        "association_frame": association_frame,
        "court_xy_median_absolute_deviation": np.median(
            np.abs(roots - np.median(roots, axis=0)), axis=0
        ).tolist(),
        "association_distance_px": chosen["association_distance_px"],
        "image_coordinate_scale": image_coordinate_scale,
        "interpretation": "automatic player box court roots; median tracked feet proxy, not pose truth",
    }


def fit(
    contact_xyz_m: np.ndarray,
    observations: dict,
    feet: dict,
    fps: float,
    *,
    config: TossConfig = TossConfig(),
) -> dict:
    """Fit a gravity-only incoming toss joined to one fixed serve contact."""
    config.validate()
    contact = np.asarray(contact_xyz_m, float)
    rows = observations["rows"]
    if observations.get("status") == "abstained":
        return {
            "schema": "connected_real_toss_witness_v1",
            "status": "abstained",
            "abstention_reason": observations["abstention_reason"],
            "observation_count": len(rows),
            "minimum_observations": observations["minimum_observations"],
            "contact_frame": float(observations["contact_frame"]),
            "shared_contact_xyz_m": contact.tolist(),
            "shared_position_and_time_with_serve": True,
            "weighted_image_rms": None,
            "selector_penalty": 0.0,
            "hard_gate": False,
            "survived": True,
            "native_projection": [],
            "automatic_inference_eligible": False,
            "independent_xyz_truth": False,
        }
    frames = np.asarray([row["frame"] for row in rows], float)
    pixels = np.asarray([row["pixel"] for row in rows], float)
    uncertainty = np.asarray([row["uncertainty_px"] for row in rows], float)
    cameras = np.asarray([row["camera"] for row in rows], float)
    from cv.experiments.connected_shooting import player_position

    feet_xy = player_position.feet_xy(feet)
    contact_frame = float(observations["contact_frame"])
    if (
        contact.shape != (3,)
        or not np.isfinite(contact).all()
        or not np.isfinite(fps)
        or fps <= 0
        or pixels.shape != (len(frames), 2)
        or cameras.shape != (len(frames), 3, 4)
        or np.any(frames >= contact_frame)
    ):
        raise ValueError("finite contact, native toss observations, cameras and cadence required")
    times = (frames - contact_frame) / fps
    minimum_release_seconds = max(float(-times[0] + 1 / fps), 0.30)
    maximum_release_seconds = max(1.25, minimum_release_seconds + 0.05)

    def states(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        velocity = parameters[:3]
        tau = parameters[3]
        xyz = contact + times[:, None] * velocity + 0.5 * times[:, None] ** 2 * GRAVITY
        origin = contact - tau * velocity + 0.5 * tau**2 * GRAVITY
        launch_velocity = velocity - tau * GRAVITY
        return xyz, origin, launch_velocity

    def residual(parameters: np.ndarray) -> np.ndarray:
        xyz, origin, launch = states(parameters)
        projected = camera_geometry.project(cameras, xyz, camera_geometry.rows_radial(rows))
        horizontal_excess = max(
            float(np.linalg.norm(launch[:2])) - config.maximum_horizontal_launch_mps, 0.0
        )
        upward_low = max(2.0 - float(launch[2]), 0.0)
        upward_high = max(float(launch[2]) - 12.0, 0.0)
        return np.r_[
            ((projected - pixels) / uncertainty[:, None]).ravel(),
            0.0 if feet_xy is None else (origin[0] - feet_xy[0]) / config.release_lateral_sigma_m,
            0.0 if feet_xy is None else (origin[1] - feet_xy[1]) / config.release_depth_sigma_m,
            (origin[2] - config.hand_height_m) / config.hand_height_sigma_m,
            horizontal_excess / 0.5,
            upward_low,
            upward_high,
        ]

    seed = np.array(
        [0.0, 0.0, -1.0, min(max(0.75, minimum_release_seconds), maximum_release_seconds)]
    )
    solved = least_squares(
        residual,
        seed,
        bounds=([-8, -8, -8, minimum_release_seconds], [8, 8, 5, maximum_release_seconds]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=300,
    )
    xyz, origin, launch = states(solved.x)
    predicted = camera_geometry.project(cameras, xyz, camera_geometry.rows_radial(rows))
    errors = np.linalg.norm(predicted - pixels, axis=1)
    weighted = errors / uncertainty
    constraints = {
        "optimizer_converged": bool(solved.success),
        "fronts_within_declared_uncertainty": bool(
            np.sqrt(np.mean(weighted**2)) <= config.maximum_weighted_rms
        ),
        **(
            {}
            if feet_xy is None
            else {
                "release_depth_near_tracked_feet": bool(
                    abs(origin[1] - feet_xy[1]) <= config.release_depth_sigma_m
                ),
                "release_lateral_near_tracked_feet": bool(
                    abs(origin[0] - feet_xy[0]) <= config.release_lateral_sigma_m
                ),
            }
        ),
        "release_at_hand_height": bool(
            abs(origin[2] - config.hand_height_m) <= 2 * config.hand_height_sigma_m
        ),
        "near_vertical_launch": bool(
            np.linalg.norm(launch[:2]) <= config.maximum_horizontal_launch_mps
        ),
        "upward_release": bool(2.0 <= launch[2] <= 12.0),
    }
    return {
        "schema": "connected_real_toss_witness_v1",
        "status": "supported" if all(constraints.values()) else "held",
        "constraints": constraints,
        "failures": [name for name, passed in constraints.items() if not passed],
        "contact_frame": contact_frame,
        "shared_contact_xyz_m": contact.tolist(),
        "shared_position_and_time_with_serve": True,
        "racket_velocity_discontinuity_at_contact": True,
        "incoming_contact_velocity_mps": solved.x[:3].tolist(),
        "release_seconds_before_contact": float(solved.x[3]),
        "release_frame": float(contact_frame - solved.x[3] * fps),
        "release_xyz_m": origin.tolist(),
        "release_velocity_mps": launch.tolist(),
        "tracked_feet_xy_m": None if feet_xy is None else feet_xy.tolist(),
        "release_depth_error_m": None if feet_xy is None else float(origin[1] - feet_xy[1]),
        "release_lateral_error_m": None if feet_xy is None else float(origin[0] - feet_xy[0]),
        **(
            {}
            if feet_xy is not None
            else {
                "player_position_evidence": feet["court_position"],
                "abstained_constraints": [
                    "release_depth_near_tracked_feet",
                    "release_lateral_near_tracked_feet",
                ],
                "native_toss_observations_retained": True,
            }
        ),
        "image_rms_px": float(np.sqrt(np.mean(errors**2))),
        "weighted_image_rms": float(np.sqrt(np.mean(weighted**2))),
        "selector_penalty": float(np.mean(weighted**2) + np.sum(residual(solved.x)[-6:-3] ** 2)),
        "native_projection": [
            {
                "frame": int(frame),
                "source": row["source"],
                "observed_front": pixel.tolist(),
                "predicted_ball_centre": estimate.tolist(),
                "error_px": float(error),
                "uncertainty_px": float(sigma),
                "xyz_m": position.tolist(),
            }
            for frame, row, pixel, estimate, error, sigma, position in zip(
                frames, rows, pixels, predicted, errors, uncertainty, xyz, strict=True
            )
        ],
        "front_semantics": (
            "Leading visible fronts scored against projected centres with their explicit broad "
            "radii; no half-streak subtraction or timestamp shift."
        ),
        "gravity_mps2": GRAVITY.tolist(),
        "optimizer": {
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": str(solved.message),
            "nfev": int(solved.nfev),
        },
        "automatic_inference_eligible": False,
        "independent_xyz_truth": False,
    }


def serve_contact_prior(contact_y_m: float, side: str, serve_number: int) -> dict:
    """Soft inside-baseline ranking prior; it can never reject a candidate."""
    if serve_number not in {1, 2} or side not in {"near", "far"} or not np.isfinite(contact_y_m):
        raise ValueError("finite serve contact, near/far side and first/second serve required")
    inside = float(contact_y_m if side == "near" else FAR_BASELINE_Y_M - contact_y_m)
    centre = 0.25
    sigma = 0.45 if serve_number == 1 else 0.90
    return {
        "serve_number": serve_number,
        "inside_distance_m": inside,
        "centre_inside_m": centre,
        "sigma_m": sigma,
        "selector_penalty": float(0.5 * ((inside - centre) / sigma) ** 2),
        "hard_gate": False,
        "survived": True,
        "interpretation": (
            "soft contact-depth prior centred slightly inside the baseline; second-serve arm "
            "is weaker to preserve rare kick-serve alternatives"
        ),
    }
