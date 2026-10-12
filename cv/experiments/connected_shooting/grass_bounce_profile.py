"""Measure a grass bounce profile from the labeled grass broadcasts themselves.

Question this answers: the Hawk-Eye CourtVision corpus has no grass, so
``physics.bounce_reference`` refuses to answer for it and every grass attempt is
fitted on the hard profile.  Can the labeled grass attempts measure their own
restitution and horizontal retention well enough to replace that substitution?

Method, per labeled bounce
--------------------------
The ball is on the court at the labeled bounce epoch, so the observation ray
there fixes its position: that is the same centre-height ground point the
connected fitter's bounce witness already constructs.  Anchored at that point,
each side of the impact is a three-parameter free flight -- the incoming
velocity before it and the outgoing velocity after it -- and every native front
on that side supplies two image equations.  Both sides are solved by least
squares against the native fronts through the attempt's own qualified camera,
with drag and Magnus from ``physics.flight`` and a nominal spin.  Restitution is
then ``v_out_z / |v_in_z|`` and horizontal retention ``|v_out_h| / |v_in_h|``,
both read off measured velocities rather than assumed ones.

What this can and cannot do
---------------------------
* **Spin is not measured.**  The arcs are fitted at a nominal topspin, so the
  two coefficients absorb whatever the real spin was.  This is the same
  limitation the Hawk-Eye fit records, and grass is reported as one value rather
  than a spin-free/at-model-spin pair because a broadcast arc cannot separate
  them either.
* **Depth is weakly observed.**  One broadcast camera sees the along-axis
  component of a short arc poorly.  The per-bounce spread, not the intercept
  alone, is the honest statement of what this measures.
* **These are opened development labels.**  The emitted profile is a calibration
  derived from human labels; it is not automatic inference, and every attempt it
  was measured on stays declared in the artifact.

The fit is deliberately shrunk toward the chart-derived expectation with a wide
prior, so a thin or noisy measurement moves the coefficient a little and a
strong one moves it a lot.  Running the same estimator on the hard and clay
attempts, whose answer the Hawk-Eye corpus already knows, is the control that
says whether the grass number can be believed at all.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import camera_geometry, cohort, swept_exposure
from cv.experiments.connected_shooting import real_bidirectional_search as whole
from cv.pipeline import paths, provenance
from physics import flight, impact

# Declared measurement controls.  They bound which impacts are usable at all;
# none of them is a fitted quantity and every one appears in the artifact.
ARC_WINDOW_FRAMES = 12
MINIMUM_ARC_PICTURES = 3
MAXIMUM_ARC_RMS_PX = 6.0
MINIMUM_VERTICAL_SPEED_MPS = 2.0
COEFFICIENT_BOUNDS = (0.05, 1.20)
NOMINAL_TOPSPIN_RPM = 1800.0
# The frozen sweep's declared native exposure; the labeled row is the leading
# tip of the streak swept over it, not the ball centre at the exposure open.
EXPOSURE_FRAMES = 0.25
EXPOSURE_SAMPLES = 3
INCIDENCE_CENTRE_DEG = 16.0

# Wide priors on the two regression coefficients.  The centre is the chart
# model's own prediction for the surface; the width says the broadcast
# measurement is allowed to overrule it.
INTERCEPT_PRIOR_SD = 0.15
SLOPE_PRIOR_SD = 0.010


@dataclass(frozen=True)
class BounceMeasurement:
    key: str
    surface: str
    frame: float
    court_xy_m: tuple[float, float]
    incidence_deg: float
    restitution: float
    horizontal_retention: float
    incoming_velocity_mps: tuple[float, float, float]
    outgoing_velocity_mps: tuple[float, float, float]
    incoming_rms_px: float
    outgoing_rms_px: float
    incoming_pictures: int
    outgoing_pictures: int


def _spin_vector(velocity: np.ndarray, rpm: float) -> np.ndarray:
    """A nominal topspin axis for the given horizontal heading."""
    horizontal = np.array([velocity[0], velocity[1], 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-9:
        return np.zeros(3)
    direction = horizontal / norm
    axis = np.array([-direction[1], direction[0], 0.0])
    return axis * (rpm * 2.0 * math.pi / 60.0)


def _project(camera: np.ndarray, radial: np.ndarray | None, xyz: np.ndarray) -> np.ndarray:
    uvw = camera @ np.r_[xyz, 1.0]
    if abs(uvw[2]) < 1e-9:
        raise ValueError("degenerate projection")
    return camera_geometry.distort(np.asarray([uvw[:2] / uvw[2]]), radial)[0]


def _sample(anchor: np.ndarray, velocity: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Positions of one drag-and-Magnus arc at signed times about the impact.

    Negative times integrate the same equations with a negative step, which is
    the physical flight that arrives at the impact.  Reversing the velocity and
    running forward is *not* the same path once drag is present, because drag
    always opposes motion.
    """
    spin = _spin_vector(velocity, NOMINAL_TOPSPIN_RPM)
    step = 1.0 / 240.0
    positions = np.empty((len(seconds), 3), float)
    for direction in (1.0, -1.0):
        wanted = [
            (index, value) for index, value in enumerate(seconds) if np.sign(value) == direction
        ]
        if not wanted:
            continue
        wanted.sort(key=lambda row: abs(row[1]))
        x, v, w = np.asarray(anchor, float), np.asarray(velocity, float), spin
        current = 0.0
        for index, target in wanted:
            while current < abs(target) - 1e-12:
                dt = min(step, abs(target) - current)
                x, v, w = flight.rk4_step(x, v, w, direction * dt)
                current += dt
            positions[index] = x
    return positions


def _arc_residual(
    velocity: np.ndarray,
    anchor: np.ndarray,
    seconds: np.ndarray,
    cameras: list[np.ndarray],
    pixels: np.ndarray,
    radial: list[np.ndarray | None],
    fps: float,
) -> np.ndarray:
    """Native pixel residuals of one free flight anchored at a measured impact.

    The labeled row is the *leading front* of the ball's streak over the native
    exposure, not its centre, so the prediction is the same swept leading tip the
    connected fitter models.  Comparing a centre against a front biases every
    fast descending arc toward a wrong velocity, which is exactly the arc a
    bounce measurement depends on.
    """
    offsets = np.linspace(0.0, EXPOSURE_FRAMES, EXPOSURE_SAMPLES) / fps
    query = (seconds[:, None] + offsets[None, :]).ravel()
    positions = _sample(anchor, velocity, query).reshape(len(seconds), EXPOSURE_SAMPLES, 3)
    rows = []
    for sweep, camera, distortion, pixel in zip(positions, cameras, radial, pixels, strict=True):
        undistorted = camera_geometry.undistort(np.asarray([pixel]), distortion)[0]
        centres = np.asarray(
            [camera @ np.r_[point, 1.0] for point in sweep],
            float,
        )
        if np.any(np.abs(centres[:, 2]) < 1e-9):
            raise ValueError("degenerate projection")
        image = centres[:, :2] / centres[:, 2:]
        axis = image[-1] - image[0]
        if float(np.linalg.norm(axis)) < 1e-6:
            rows.append(image[0] - undistorted)
            continue
        tips, _ = swept_exposure.directional_tips(sweep, camera, axis)
        rows.append(tips[1] - undistorted)
    return np.asarray(rows, float).ravel()


def _solve_arc(
    anchor: np.ndarray,
    frames: np.ndarray,
    bounce_frame: float,
    fps: float,
    cameras: list[np.ndarray],
    pixels: np.ndarray,
    radial: list[np.ndarray | None],
    seed: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Least-squares velocity at the impact for one arc, or ``None`` if unresolved.

    The solved quantity is always the ball's velocity *at the impact instant*:
    the outgoing velocity for pictures after it, the incoming velocity for
    pictures before it.  Neither side invents an impact time or a picture.
    """
    seconds = (frames - bounce_frame) / fps
    if not len(seconds) or np.any(seconds == 0):
        return None
    try:
        solved = least_squares(
            _arc_residual,
            seed,
            args=(anchor, seconds, cameras, pixels, radial, fps),
            max_nfev=200,
            xtol=1e-10,
        )
    except (ValueError, OverflowError, FloatingPointError):
        return None
    if not solved.success and solved.status <= 0:
        return None
    residual = np.asarray(solved.fun, float).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    velocity = np.asarray(solved.x, float)
    if not np.isfinite(velocity).all() or not np.isfinite(rms):
        return None
    return velocity, rms


def measure_attempt(
    key: str,
    surface: str,
    packet: dict[str, Any],
    cameras_document: dict[str, Any],
) -> tuple[list[BounceMeasurement], list[dict[str, Any]]]:
    """Every usable bounce of one prepared attempt, plus each rejection's reason."""
    attempt = packet["attempts"][0]
    fps = float(attempt["fps"])
    rows = cameras_document["cameras"]
    cameras = {
        int(row["frame"]): np.asarray(row["P"], float)
        for row in rows
        if row.get("status") == "supported" and "P" in row
    }
    radial = {
        int(row["frame"]): np.asarray([float(row["k1"]), *row["dist_center"]], float)
        for row in rows
        if row.get("status") == "supported" and "k1" in row and "dist_center" in row
    }
    labels = {
        int(row["frame"]): np.asarray([row["x1080"], row["y1080"]], float)
        for row in attempt["owner_ball_labels"]
        if row["status"] == "visible"
    }
    events = sorted(attempt["events"], key=lambda row: float(row["frame"]))
    epochs = [float(row["frame"]) for row in events] + [float(attempt["owner_end_frame"])]
    measurements: list[BounceMeasurement] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(events):
        if row["event_type"] != "bounce":
            continue
        frame = float(row["frame"])
        reject = {"key": key, "frame": frame}
        previous = max((value for value in epochs if value < frame), default=None)
        following = min(
            (value for value in epochs if value > frame),
            # A terminal bounce has no later event, but the ball keeps flying and
            # the labels keep describing it, so the window is the pictures.
            default=(max(labels) + 1.0 if labels else None),
        )
        if previous is None or following is None or following <= frame:
            rejected.append({**reject, "reason": "bounce_is_not_bracketed_by_two_epochs"})
            continue
        anchor_frames = sorted(
            candidate
            for candidate in (
                int(np.floor(frame)),
                int(np.ceil(frame)),
            )
            if candidate in cameras and candidate in labels
        )
        if not anchor_frames:
            rejected.append({**reject, "reason": "no_native_ground_ray_at_the_impact"})
            continue
        anchor = np.mean(
            [whole.ground_point(cameras[value], labels[value]) for value in anchor_frames], axis=0
        )
        sides = {}
        for name, low, high in (
            ("incoming", max(previous, frame - ARC_WINDOW_FRAMES), frame),
            ("outgoing", frame, min(following, frame + ARC_WINDOW_FRAMES)),
        ):
            sides[name] = sorted(
                value
                for value in labels
                if value in cameras and low < value < high and abs(value - frame) >= 1
            )
        if any(len(value) < MINIMUM_ARC_PICTURES for value in sides.values()):
            rejected.append(
                {
                    **reject,
                    "reason": "too_few_native_fronts_on_one_side",
                    "pictures": {name: len(value) for name, value in sides.items()},
                }
            )
            continue
        solved = {}
        for name, chosen in sides.items():
            frames = np.asarray(chosen, float)
            seed = np.array([0.0, 15.0, 5.0 if name == "outgoing" else -5.0])
            answer = _solve_arc(
                anchor,
                frames,
                frame,
                fps,
                [cameras[int(value)] for value in chosen],
                np.asarray([labels[int(value)] for value in chosen], float),
                [radial.get(int(value)) for value in chosen],
                seed,
            )
            if answer is None:
                break
            solved[name] = answer
        if len(solved) != 2:
            rejected.append({**reject, "reason": "an_arc_did_not_resolve"})
            continue
        (incoming, incoming_rms), (outgoing, outgoing_rms) = (
            solved["incoming"],
            solved["outgoing"],
        )
        if max(incoming_rms, outgoing_rms) > MAXIMUM_ARC_RMS_PX:
            rejected.append(
                {
                    **reject,
                    "reason": "arc_image_residual_above_the_declared_bound",
                    "incoming_rms_px": incoming_rms,
                    "outgoing_rms_px": outgoing_rms,
                }
            )
            continue
        vertical_in, vertical_out = -float(incoming[2]), float(outgoing[2])
        horizontal_in = float(np.hypot(incoming[0], incoming[1]))
        horizontal_out = float(np.hypot(outgoing[0], outgoing[1]))
        if vertical_in < MINIMUM_VERTICAL_SPEED_MPS or vertical_out <= 0 or horizontal_in <= 0:
            rejected.append(
                {
                    **reject,
                    "reason": "impact_is_not_a_descending_rebound",
                    "incoming_velocity_mps": incoming.tolist(),
                    "outgoing_velocity_mps": outgoing.tolist(),
                }
            )
            continue
        restitution = vertical_out / vertical_in
        retention = horizontal_out / horizontal_in
        low, high = COEFFICIENT_BOUNDS
        if not (low <= restitution <= high and low <= retention <= high):
            rejected.append(
                {
                    **reject,
                    "reason": "coefficient_outside_the_declared_physical_bounds",
                    "restitution": restitution,
                    "horizontal_retention": retention,
                }
            )
            continue
        measurements.append(
            BounceMeasurement(
                key=key,
                surface=surface,
                frame=frame,
                court_xy_m=(float(anchor[0]), float(anchor[1])),
                incidence_deg=math.degrees(math.atan2(vertical_in, horizontal_in)),
                restitution=restitution,
                horizontal_retention=retention,
                incoming_velocity_mps=tuple(float(value) for value in incoming),
                outgoing_velocity_mps=tuple(float(value) for value in outgoing),
                incoming_rms_px=incoming_rms,
                outgoing_rms_px=outgoing_rms,
                incoming_pictures=len(sides["incoming"]),
                outgoing_pictures=len(sides["outgoing"]),
            )
        )
    return measurements, rejected


def chart_prediction(surface: str, incidence_deg: float) -> dict[str, float]:
    """The existing chart model's own answer, used only as the prior centre."""
    angle = math.radians(incidence_deg)
    horizontal, vertical = math.cos(angle) * 25.0, math.sin(angle) * 25.0
    result = impact.court_bounce(horizontal, vertical, 0.0, surface=surface)
    return {
        "restitution": float(result.vy2 / vertical),
        "horizontal_retention": float(result.vx2 / horizontal),
    }


def shrunk_regression(
    angles: np.ndarray, values: np.ndarray, prior_intercept: float, prior_slope: float
) -> dict[str, float]:
    """Ridge regression of a coefficient on incidence with an explicit wide prior.

    The prior is a proper Gaussian on (intercept, slope), so a thin measurement
    returns the chart expectation and a strong one overrules it.  The reported
    ``prior_weight`` says which happened.
    """
    design = np.column_stack([np.ones(len(angles)), angles - INCIDENCE_CENTRE_DEG])
    prior_mean = np.array([prior_intercept, prior_slope], float)
    prior_precision = np.diag([1.0 / INTERCEPT_PRIOR_SD**2, 1.0 / SLOPE_PRIOR_SD**2])
    residual_sd = float(np.std(values)) or 1e-3
    likelihood_precision = design.T @ design / residual_sd**2
    posterior_precision = likelihood_precision + prior_precision
    posterior = np.linalg.solve(
        posterior_precision,
        design.T @ values / residual_sd**2 + prior_precision @ prior_mean,
    )
    unshrunk = np.linalg.lstsq(design, values, rcond=None)[0] if len(values) > 2 else prior_mean
    return {
        "intercept": float(posterior[0]),
        "slope": float(posterior[1]),
        "residual_std": residual_sd,
        "unshrunk_intercept": float(unshrunk[0]),
        "unshrunk_slope": float(unshrunk[1]),
        "prior_intercept": prior_intercept,
        "prior_slope": prior_slope,
        "prior_weight": float(
            np.trace(np.linalg.solve(posterior_precision, prior_precision)) / 2.0
        ),
    }


def profile(measurements: list[BounceMeasurement], surface: str) -> dict[str, Any]:
    """Turn one surface's bounces into ``physics.bounce_reference`` coefficients."""
    angles = np.asarray([row.incidence_deg for row in measurements], float)
    median = float(np.median(angles))
    prior = chart_prediction(surface, median)
    restitution = shrunk_regression(
        angles,
        np.asarray([row.restitution for row in measurements], float),
        prior["restitution"],
        0.0,
    )
    retention = shrunk_regression(
        angles,
        np.asarray([row.horizontal_retention for row in measurements], float),
        prior["horizontal_retention"],
        0.0,
    )
    return {
        "surface": surface,
        "n": len(measurements),
        "attempts": sorted({row.key for row in measurements}),
        "incidence_deg_median": median,
        "chart_prior_at_median_incidence": prior,
        "restitution": restitution,
        "horizontal_retention": retention,
        "coefficients": {
            "incidence_deg_median": median,
            "n": len(measurements),
            "restitution_intercept_at_model_spin": restitution["intercept"],
            "restitution_intercept_spin_free": restitution["intercept"],
            "restitution_residual_std_at_model_spin": restitution["residual_std"],
            "restitution_residual_std_spin_free": restitution["residual_std"],
            "restitution_slope_at_model_spin": restitution["slope"],
            "restitution_slope_spin_free": restitution["slope"],
            "retention_intercept_at_model_spin": retention["intercept"],
            "retention_intercept_spin_free": retention["intercept"],
            "retention_residual_std_at_model_spin": retention["residual_std"],
            "retention_residual_std_spin_free": retention["residual_std"],
            "retention_slope_at_model_spin": retention["slope"],
            "retention_slope_spin_free": retention["slope"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preparation-root",
        type=Path,
        required=True,
        help="a prepared sweep root supplying each attempt's packet and qualified cameras",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1",
    )
    parser.add_argument("--surfaces", nargs="*", default=["grass", "hard", "clay"])
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    discovered = cohort.discover(arguments.labels_root.resolve())
    measurements: list[BounceMeasurement] = []
    rejected: list[dict[str, Any]] = []
    missing: list[str] = []
    for attempt in discovered:
        if attempt.surface not in arguments.surfaces:
            continue
        inputs = arguments.preparation_root / attempt.key / "inputs"
        if not (inputs / "packet.json").is_file() or not (inputs / "cameras.json").is_file():
            missing.append(attempt.key)
            continue
        rows, reasons = measure_attempt(
            attempt.key,
            attempt.surface,
            json.loads((inputs / "packet.json").read_text()),
            json.loads((inputs / "cameras.json").read_text()),
        )
        measurements.extend(rows)
        rejected.extend(reasons)
    profiles = {}
    for surface in sorted({row.surface for row in measurements}):
        rows = [row for row in measurements if row.surface == surface]
        profiles[surface] = profile(rows, surface)
    report = {
        "schema": "broadcast_labeled_bounce_profile_v1",
        "question": __doc__,
        "human_derived": True,
        "automatic_inference_eligible": False,
        "opened_development_data": True,
        "controls": {
            "arc_window_frames": ARC_WINDOW_FRAMES,
            "minimum_arc_pictures": MINIMUM_ARC_PICTURES,
            "maximum_arc_rms_px": MAXIMUM_ARC_RMS_PX,
            "minimum_vertical_speed_mps": MINIMUM_VERTICAL_SPEED_MPS,
            "coefficient_bounds": list(COEFFICIENT_BOUNDS),
            "nominal_topspin_rpm": NOMINAL_TOPSPIN_RPM,
            "intercept_prior_sd": INTERCEPT_PRIOR_SD,
            "slope_prior_sd": SLOPE_PRIOR_SD,
            "prior_centre": "physics.impact.court_bounce chart model at the median incidence",
        },
        "preparation_root": str(arguments.preparation_root),
        "labels_root": str(arguments.labels_root.resolve()),
        "attempts_without_preparation": missing,
        "profiles": profiles,
        "hawkeye_reference": {
            surface: {
                "restitution_intercept_spin_free": values["restitution_intercept_spin_free"],
                "retention_intercept_spin_free": values["retention_intercept_spin_free"],
                "n": values["n"],
            }
            for surface, values in __import__(
                "physics.bounce_reference", fromlist=["MEASURED"]
            ).MEASURED.items()
        },
        "measurements": [row.__dict__ for row in measurements],
        "rejected": rejected,
        "code": provenance.git_record(paths.REPO_ROOT),
    }
    path = arguments.output / "bounce_profile.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False, default=list) + "\n")
    print(
        json.dumps(
            {
                "measured": len(measurements),
                "rejected": len(rejected),
                "surfaces": {
                    surface: {
                        "n": values["n"],
                        "restitution": round(values["restitution"]["intercept"], 4),
                        "retention": round(values["horizontal_retention"]["intercept"], 4),
                    }
                    for surface, values in profiles.items()
                },
                "report": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
