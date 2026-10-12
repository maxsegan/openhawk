"""Judge the Stage-6 flight acceptance gates against owner ground truth.

This module is evaluation-only.  It reads owner bounce/contact clicks and owner-positioned
native frames *after* reconstruction has been frozen, so it must never be imported by
automatic inference.  Nothing here changes a pipeline decision; it produces a per-flight
truth ledger, a per-point primary-cause ledger, and a label-blind gate proposal.

Commands::

    python -m cv.validation.flight_gate_audit audit  --output-root ...
    python -m cv.validation.flight_gate_audit sheets --audit-root ... --output-root ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cv.pipeline import reconstruction
from cv.pipeline.flight_ledger import build as build_flight_ledger
from cv.validation.s6_cohort_v2_eval import HELD_OUT_BROADCASTS
from cv.validation.s6root_metric_acceptance import flight_metric_row

# --- truth-witness thresholds ------------------------------------------------------------
# The owner's stated location goal is 10 cm.  The pixel tolerances are the pipeline's own
# published tolerances so that a "truth-good" flight means the same thing as an accepted one,
# only measured against the owner's clicks instead of the automatic track.
BOUNCE_TRUTH_LIMIT_M = 0.10
OWNER_POSITION_MEDIAN_LIMIT_PX = 8.0
OWNER_POSITION_P90_LIMIT_PX = 20.0
OWNER_CONTACT_LIMIT_PX = 12.0
MIN_OWNER_POSITION_FRAMES = 1
CONTACT_MATCH_FRAMES = 2.0
BOUNCE_MATCH_FRAMES = 2.0
EVENT_MATCH_FRAMES = 3.0
JUNCTION_GAP_LIMIT_M = 0.30

PHYSICAL_EVENTS = ("contact", "bounce", "net_hit")

CAUSE_ORDER = (
    "camera_abstained",
    "event_missing",
    "flight_not_attempted",
    "flight_not_solved",
    "missing_bounce",
    "solved_but_rejected",
    "junction_gap",
)


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float | None:
    rows = np.asarray([float(value) for value in values], dtype=float)
    rows = rows[np.isfinite(rows)]
    return float(np.percentile(rows, q)) if rows.size else None


def point_match(point: str) -> str:
    return str(point).rsplit("__", 1)[0]


def scope_of(point: str) -> str:
    return "held_out_8" if point_match(point) in HELD_OUT_BROADCASTS else "other_38"


# --- owner witnesses ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def owner_position_index() -> dict[tuple[str, str], dict[int, tuple[float, float]]]:
    """Owner-clicked native ball positions, keyed by (match_id, clip) then frame.

    Sources are the two frozen 49-window ball-track sequence splits (98 windows) and the
    owner's corrected trajectory rows.  Only native 1920x1080 coordinates are used.
    """
    labels = Path(__file__).parent / "labels"
    index: dict[tuple[str, str], dict[int, tuple[float, float]]] = defaultdict(dict)
    sequence_root = labels / "ball_track_sequence_v1"
    for name in ("development_truth_v1.json", "sealed_transfer_truth_v1.json"):
        payload = json.loads((sequence_root / name).read_text())
        for record in payload["records"]:
            match_id, clip, *_ = str(record["case_id"]).split("__")
            for row in record["frames"]:
                if row.get("status") != "visible" or row.get("x1080") is None:
                    continue
                index[(match_id, clip)][int(row["frame"])] = (
                    float(row["x1080"]),
                    float(row["y1080"]),
                )
    for name in ("ball_trajectory_v1", "ball_trajectory_followup_v1"):
        path = labels / name / "ball_trajectory_labels.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("frame_status") != "corrected" or not row.get("frame"):
                    continue
                if not row.get("corrected_x540") or not row.get("corrected_y540"):
                    continue
                index[(row["match_id"], row["clip"])].setdefault(
                    int(row["frame"]),
                    (float(row["corrected_x540"]) * 2.0, float(row["corrected_y540"]) * 2.0),
                )
    return {key: dict(value) for key, value in index.items()}


def owner_event_index(truth_events: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Owner event clicks per point, split by physical type, with native pixels."""
    payload = json.loads(truth_events.read_text())
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {name: [] for name in PHYSICAL_EVENTS}
    )
    for row in payload["emissions"]:
        kind = str(row.get("event_type"))
        if kind not in PHYSICAL_EVENTS:
            continue
        location = row.get("location") or {}
        index[str(row["clip"])][kind].append(
            {
                "frame": float(row["frame"]),
                "fps": (None if location.get("fps") is None else float(location["fps"])),
                "image_xy": (
                    None
                    if location.get("image_x") is None
                    else (float(location["image_x"]), float(location["image_y"]))
                ),
            }
        )
    return {point: value for point, value in index.items()}


def automatic_event_index(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text())
    rows = payload["emissions"] if isinstance(payload, dict) else payload
    index: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in PHYSICAL_EVENTS}
    )
    for row in rows:
        kind = str(row.get("event_type"))
        if kind in PHYSICAL_EVENTS:
            index[str(row["clip"])][kind].append(float(row["frame"]))
    return {point: value for point, value in index.items()}


def unmatched_truth_events(
    truth_frames: Sequence[float], automatic_frames: Sequence[float], tolerance: float
) -> int:
    """Greedy nearest-first one-to-one match; return how many truth rows stay unmatched."""
    available = list(automatic_frames)
    missing = 0
    for frame in sorted(truth_frames):
        best = None
        for index, candidate in enumerate(available):
            distance = abs(candidate - frame)
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, index)
        if best is None:
            missing += 1
        else:
            available.pop(best[1])
    return missing


# --- geometry ----------------------------------------------------------------------------


def project(projection: np.ndarray, xyz: Sequence[float]) -> tuple[float, float] | None:
    homogeneous = np.asarray(projection, dtype=float) @ np.asarray(
        [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=float
    )
    if abs(float(homogeneous[2])) < 1e-9:
        return None
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


def trajectory_state(trajectory: Sequence[dict[str, Any]], frame: float) -> np.ndarray | None:
    """Linear interpolation of the fitted trajectory, clamped to at most one frame outside."""
    samples = [
        (float(row["frame"]), np.asarray(row["xyz"], dtype=float))
        for row in trajectory
        if row.get("xyz") is not None
    ]
    if not samples:
        return None
    samples.sort(key=lambda row: row[0])
    if frame <= samples[0][0]:
        return samples[0][1] if samples[0][0] - frame <= 1.0 else None
    if frame >= samples[-1][0]:
        return samples[-1][1] if frame - samples[-1][0] <= 1.0 else None
    for before, after in zip(samples, samples[1:]):
        if before[0] <= frame <= after[0]:
            span = after[0] - before[0]
            if span <= 0.0:
                return before[1]
            weight = (frame - before[0]) / span
            return before[1] + weight * (after[1] - before[1])
    return None


def reprojection_errors(
    fit: dict[str, Any],
    camera: Any,
    witnesses: Sequence[tuple[float, tuple[float, float]]],
) -> list[dict[str, Any]]:
    """Pixel error between the fitted 3D flight and owner-clicked image positions."""
    rows = []
    trajectory = fit.get("trajectory", [])
    for frame, pixel in witnesses:
        xyz = trajectory_state(trajectory, frame)
        if xyz is None:
            continue
        uv = project(camera.p_at(frame), xyz)
        if uv is None:
            continue
        rows.append(
            {
                "frame": float(frame),
                "error_px": float(math.hypot(uv[0] - pixel[0], uv[1] - pixel[1])),
            }
        )
    return rows


# --- the corrected bounce witness ----------------------------------------------------------
#
# The shipped bounce witness (`bounce_rayplane_v1`) intersects the owner's click ray with the
# court plane z = 0 on the click's own integer frame.  It is wrong twice over, and this
# package measures both errors on the synthetic bench, where the true bounce is known:
#
#   * the plane is wrong.  A ball resting on the court has its centre one radius up, and the
#     fitted bounce this witness is compared against is a ball centre.  Intersecting the
#     *true* bounce pixel at the *true* bounce time with z = 0 is already 0.124 m from the
#     true bounce; at z = R_BALL it is exact.  That is the whole of the floor.
#   * the time is wrong.  The click is on an integer frame and the bounce is not, so the ball
#     is still in the air by up to half a frame of flight.  At z = R_BALL and the click's own
#     frame the bench measures 0.143 m median -- the same number `docs/wk1/bench_frames.md`
#     measured for `flight_anchors.bounce_anchor`, which is that construction.
#
# `bounce_kinematic_v2` fixes both and stays anchored on the owner.  `cv.pipeline
# .subframe_timing`'s image-space corner supplies the sub-frame bounce time from the ball
# positions around the bounce -- owner-positioned native frames where the 98 label windows
# reach, otherwise the arc-augmented track.  The owner's click is then *transported* along
# that side's own fitted image path from the click's time to the bounce time, so the absolute
# position stays the owner's and only the sub-frame difference comes from the track.  The
# transported pixel is intersected with the plane z = R_BALL at the bounce time, which is the
# one time at which the ball really is on the court.
#
# Bench, clean rung, 576 of 785 emitted bounces (section 4 of the report): 0.264 m median for
# the shipped witness, 0.020 m for this one.

BOUNCE_WITNESS_WINDOW = 4
# Owner click radius, native px, one sigma.  docs/wk1/HARNESS_LESSONS.md item 2 measures the
# owner's own pointing at a median near 1.5 px540 and a p90 of 4.1 px540, so 2 px540 is the
# robust one-sigma spread and this is that in native pixels.  The labels carry no per-click
# radius (`uncertainty_radius_px540` is always null in docs/LABELS_DATA_DICTIONARY.md), so
# this is a stated constant.  The bench cannot calibrate it, because the bench's own "click"
# is an exact projection.
CLICK_SIGMA_PX = 4.0
# docs/wk1/HARNESS_LESSONS.md item 5 says the owner labels the leading-blur frame and a -0.5
# offset should be applied.  docs/wk1/subframe_timing.md section 1.5 measures that the frozen
# truth is inconsistent about whether that has already happened (1,160 integer labels against
# 255 half-frame ones) and that `emitted_minus_half` is worse than `emitted` on every bench
# slice.  The default is therefore to take `labeled_frame` at face value; `--click-offset-
# frames` moves it and the report states what that costs.
CLICK_OFFSET_FRAMES = 0.0
# A witness whose own one-sigma width is wider than the line it is asked to draw cannot
# resolve that line.  Reported as a separate column; it does not abstain by default.
BOUNCE_V2_SIGMA_LIMIT_M = 0.10
BOUNCE_V2_MATCH_FRAMES = 2.0

# --- the graded witness circle --------------------------------------------------------------
# The owner's instruction: his bounce clicks are on the *centre* of the ball, so the witness has
# a real width and a hard 10 cm pass/fail asks it to resolve something it cannot.  Inside a
# circle of radius max(0.20 m, 2 x the witness's own propagated sigma) the flight is fully valid
# (score 1); outside it the score decays smoothly and a flight passes the bounce line at
# score >= 0.5.  The hard 10 cm verdict is kept beside it as `bounce_hard_v1`.
BOUNCE_GRADED_FLOOR_M = 0.20
BOUNCE_GRADED_SIGMA_MULTIPLE = 2.0
GRADED_PASS_SCORE = 0.5
# The same shape for the contact witness, in pixels: fully valid inside the
# `contact_frame_refiner` radius for that contact, floored at the pipeline's own 12 px contact
# tolerance so the circle is never tighter than the shipped line.  Where the refiner abstains
# (`sigma_px` at its 120 px cap) it supplies no usable radius, so the floor is used and the
# column says so -- the lenient reading is reported separately rather than assumed.
CONTACT_GRADED_FLOOR_PX = OWNER_CONTACT_LIMIT_PX

# --- the junction witness: the one witness that spans a contact --------------------------
# Two fitted flights that share a racket contact cannot both be right if they disagree about
# where that contact was.  `docs/wk1/gate_search2.md` section 6 measured that no real witness
# ever asks: 33 of the 55 real flights the corrected witness calls right have a junction gap
# wider than the bench's own 0.25 m tolerance, three wider than 20 m and one 41.5 m.  The
# floor is that bench tolerance (`docs/wk1/bench_metric.md`: "a junction is one contact seen
# twice: the two fitted flights must agree about where the ball was to the same tolerance the
# contact itself is held to"), widened where the contact circle propagated into metres at the
# contact's own range is wider than half of it.
JUNCTION_WITNESS_FLOOR_M = 0.25
JUNCTION_WITNESS_SIGMA_MULTIPLE = 2.0


def graded_score(distance: float, radius: float) -> float:
    """1 inside the circle, ``exp(-((d - r) / r)^2)`` outside it.

    Continuous and flat at ``d = r`` (the derivative is zero there, so a flight just outside the
    circle is barely penalised), 0.5 at ``d = 1.833 r``, 0.018 at ``d = 3 r``.  It never returns
    a hard fail, only a shrinking score -- which is what "a small penalty grows with distance"
    asks for -- though beyond about 26 r it underflows to exactly 0.0 in double precision.
    """
    if not math.isfinite(distance) or not math.isfinite(radius) or radius <= 0.0:
        return 0.0
    if distance <= radius:
        return 1.0
    return float(math.exp(-(((distance - radius) / radius) ** 2)))


def graded_score_linear(distance: float, radius: float) -> float:
    """The owner's other suggested shape: a linear ramp from 1 at ``r`` to 0 at ``3 r``.

    Carried on every row so the choice of shape can be checked: it puts the 0.5 verdict at
    ``d = 2 r`` where the exponential puts it at ``d = 1.833 r``.
    """
    if not math.isfinite(distance) or not math.isfinite(radius) or radius <= 0.0:
        return 0.0
    if distance <= radius:
        return 1.0
    return float(max(0.0, 1.0 - (distance - radius) / (2.0 * radius)))


def bounce_graded_radius_m(sigma_m: float | None) -> float:
    """The bounce circle: 0.20 m, widened to 2 sigma where this witness is wider than that."""
    if sigma_m is None or not math.isfinite(float(sigma_m)):
        return BOUNCE_GRADED_FLOOR_M
    return max(BOUNCE_GRADED_FLOOR_M, BOUNCE_GRADED_SIGMA_MULTIPLE * float(sigma_m))


def contact_graded_radius_px(
    track: dict[int, Any] | None,
    frame: float,
    cache: dict[int, tuple[float, bool]] | None = None,
) -> tuple[float, bool]:
    """The contact circle: the `contact_frame_refiner` radius at this contact, floored at 12 px.

    Returns ``(radius_px, refiner_abstained)``.  The refiner is run here on the same
    arc-augmented track the fitter observed; it is the pipeline's own statement of how well the
    contact's image is known, and the owner asked for everything inside it to be fully valid.
    """
    key = int(round(float(frame)))
    if cache is not None and key in cache:
        return cache[key]
    result = (CONTACT_GRADED_FLOOR_PX, True)
    if track:
        from cv.pipeline import contact_frame_refiner

        try:
            refined = contact_frame_refiner.refine(track, float(frame))
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError):
            refined = None
        if refined is not None:
            sigma = float(refined.sigma_px)
            if refined.abstain or not math.isfinite(sigma):
                result = (CONTACT_GRADED_FLOOR_PX, True)
            else:
                result = (max(CONTACT_GRADED_FLOOR_PX, sigma), False)
    if cache is not None:
        cache[key] = result
    return result


def ray_at_height(
    projection: np.ndarray, pixel: Sequence[float], height: float
) -> np.ndarray | None:
    """Where the ray through ``pixel`` meets the horizontal plane ``z = height``, in metres."""
    matrix = np.asarray(projection, dtype=float)
    design = np.column_stack(
        [matrix[:, 0], matrix[:, 1], -np.asarray([pixel[0], pixel[1], 1.0], dtype=float)]
    )
    offset = -(matrix[:, 2] * float(height) + matrix[:, 3])
    try:
        solved = np.linalg.solve(design, offset)
    except np.linalg.LinAlgError:
        return None
    return solved[:2] if np.isfinite(solved).all() else None


def fit_image_path(
    samples: Sequence[tuple[float, tuple[float, float]]], anchor: float
) -> tuple[Any, float] | None:
    """`subframe_timing`'s own side fit, re-run here so the path can be sampled at any time.

    Quadratic in time with three or more samples so gravity and drag bend it, straight with
    two.  Returns the path and its RMS residual in pixels.
    """
    if len(samples) < 2:
        return None
    times = np.asarray([float(frame) - anchor for frame, _ in samples], dtype=float)
    points = np.asarray([point for _, point in samples], dtype=float)
    degree = 2 if len(samples) >= 3 else 1
    design = np.vander(times, degree + 1, increasing=True)
    try:
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coefficients).all():
        return None
    residual = float(np.sqrt(np.mean(np.sum((points - design @ coefficients) ** 2, axis=1))))

    def path(time: float) -> np.ndarray:
        return (np.vander([time - anchor], degree + 1, increasing=True) @ coefficients)[0]

    return path, residual


def metres_per_pixel(
    projection: np.ndarray, pixel: Sequence[float], height: float
) -> tuple[float, float, float] | None:
    """Local court-plane scale at ``pixel``: (RMS, smallest, largest) metres per pixel.

    The scale is strongly anisotropic on a broadcast camera -- a pixel of image error moves
    the court point a little sideways and a long way in depth -- so the two singular values
    are carried separately and the RMS is what the scalar sigma uses.
    """
    columns = []
    for axis in (0, 1):
        step = np.zeros(2, dtype=float)
        step[axis] = 1.0
        forward = ray_at_height(projection, np.asarray(pixel, dtype=float) + step, height)
        backward = ray_at_height(projection, np.asarray(pixel, dtype=float) - step, height)
        if forward is None or backward is None:
            return None
        columns.append((forward - backward) / 2.0)
    jacobian = np.column_stack(columns)
    if not np.isfinite(jacobian).all():
        return None
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return (
        float(np.linalg.norm(jacobian) / math.sqrt(2.0)),
        float(singular.min()),
        float(singular.max()),
    )


def camera_centre(projection: np.ndarray) -> np.ndarray | None:
    """The camera's own position in court metres, from its 3x4 projection."""
    matrix = np.asarray(projection, dtype=float)
    try:
        centre = -np.linalg.solve(matrix[:, :3], matrix[:, 3])
    except np.linalg.LinAlgError:
        return None
    return centre if np.isfinite(centre).all() else None


def transverse_metres_per_pixel(
    projection: np.ndarray, xyz: Sequence[float], *, radial: np.ndarray | None = None
) -> float | None:
    """Metres per pixel across the camera ray at ``xyz``: the scale a click radius buys.

    A click radius constrains the two directions perpendicular to the ray and says nothing
    about depth, so this is the honest conversion of a contact circle in pixels into a
    positional sigma in metres.  Measured, not assumed: the projection's own Jacobian is
    restricted to the plane through ``xyz`` perpendicular to the ray and inverted, and the
    RMS of that inverse's two singular values is returned.
    """
    matrix = np.asarray(projection, dtype=float)
    point = np.asarray(xyz, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    centre = camera_centre(matrix)
    if centre is None:
        return None
    ray = point - centre
    norm = float(np.linalg.norm(ray))
    if norm < 1e-9:
        return None
    ray = ray / norm
    # Two orthonormal directions perpendicular to the ray.
    helper = np.array([0.0, 0.0, 1.0]) if abs(ray[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(ray, helper)
    first /= float(np.linalg.norm(first))
    second = np.cross(ray, first)
    homogeneous = matrix @ np.append(point, 1.0)
    if abs(homogeneous[2]) < 1e-12:
        return None
    columns = []
    for direction in (first, second):
        derivative = matrix[:, :3] @ direction
        columns.append(
            (derivative[:2] * homogeneous[2] - homogeneous[:2] * derivative[2])
            / (homogeneous[2] ** 2)
        )
    jacobian = np.column_stack(columns)
    if radial is not None:
        from cv.experiments.connected_shooting.camera_geometry import distortion_jacobian

        jacobian = (
            distortion_jacobian(
                (homogeneous[:2] / homogeneous[2])[None], np.asarray(radial, float)[None]
            )[0]
            @ jacobian
        )
    if not np.isfinite(jacobian).all():
        return None
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular.min() < 1e-12:
        return None
    return float(math.sqrt(float(np.mean(1.0 / singular**2))))


def ray_distance_m(
    projection: np.ndarray, pixel: Sequence[float], xyz: Sequence[float]
) -> float | None:
    """Perpendicular distance in metres from ``xyz`` to the ray through ``pixel``."""
    matrix = np.asarray(projection, dtype=float)
    point = np.asarray(xyz, dtype=float)
    centre = camera_centre(matrix)
    if centre is None or point.shape != (3,):
        return None
    try:
        direction = np.linalg.solve(
            matrix[:, :3], np.asarray([pixel[0], pixel[1], 1.0], dtype=float)
        )
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12 or not np.isfinite(direction).all():
        return None
    direction = direction / norm
    offset = point - centre
    return float(np.linalg.norm(offset - float(offset @ direction) * direction))


def junction_witness(
    *,
    camera: Any,
    boundary_frame: float,
    click: dict[str, Any],
    inbound_end_xyz: Sequence[float] | None,
    outbound_start_xyz: Sequence[float] | None,
    radius_px: float,
    refiner_abstained: bool,
) -> dict[str, Any]:
    """The real witness that spans a contact, for one internal contact of a point.

    The two fitted flights that share this contact must meet each other on the owner's own
    click ray.  The witness fails **both** flights when their fitted states differ by more
    than ``max(0.25 m, 2 x the propagated contact-radius sigma)`` or when either of them lies
    further from the click ray than the contact circle allows.  Graded like the bounce
    circle: free inside, a smooth penalty outside, pass at score >= 0.5.
    """
    record: dict[str, Any] = {"formed": False, "reason": None}
    if inbound_end_xyz is None or outbound_start_xyz is None:
        record["reason"] = "missing_fitted_state"
        return record
    inbound = np.asarray(inbound_end_xyz, dtype=float)
    outbound = np.asarray(outbound_start_xyz, dtype=float)
    if inbound.shape != (3,) or outbound.shape != (3,):
        record["reason"] = "missing_fitted_state"
        return record
    if not (np.isfinite(inbound).all() and np.isfinite(outbound).all()):
        record["reason"] = "non_finite_fitted_state"
        return record
    click_frame = float(click["frame"])
    projection = camera.p_at(click_frame)
    scales = [
        value
        for state in (inbound, outbound)
        if (value := transverse_metres_per_pixel(projection, state)) is not None
    ]
    if len(scales) < 2:
        record["reason"] = "no_camera_scale"
        return record
    scale = max(scales)
    sigma_m = float(radius_px) * scale
    radius_m = max(JUNCTION_WITNESS_FLOOR_M, JUNCTION_WITNESS_SIGMA_MULTIPLE * sigma_m)
    gap_m = float(np.linalg.norm(inbound - outbound))

    rays: dict[str, float] = {}
    for name, state in (("inbound", inbound), ("outbound", outbound)):
        uv = project(projection, state)
        if uv is None:
            record["reason"] = "state_behind_camera"
            return record
        rays[f"{name}_ray_px"] = float(
            math.hypot(uv[0] - click["image_xy"][0], uv[1] - click["image_xy"][1])
        )
        metric = ray_distance_m(projection, click["image_xy"], state)
        if metric is not None:
            rays[f"{name}_ray_m"] = metric

    ray_max_px = max(rays["inbound_ray_px"], rays["outbound_ray_px"])
    position_score = graded_score(gap_m, radius_m)
    ray_score = min(
        graded_score(rays["inbound_ray_px"], float(radius_px)),
        graded_score(rays["outbound_ray_px"], float(radius_px)),
    )
    position_score_linear = graded_score_linear(gap_m, radius_m)
    ray_score_linear = min(
        graded_score_linear(rays["inbound_ray_px"], float(radius_px)),
        graded_score_linear(rays["outbound_ray_px"], float(radius_px)),
    )
    record.update(
        {
            "formed": True,
            "boundary_frame": float(boundary_frame),
            "click_frame": click_frame,
            "gap_m": gap_m,
            "metres_per_pixel": scale,
            "sigma_m": sigma_m,
            "radius_m": radius_m,
            "radius_px": float(radius_px),
            "refiner_abstained": bool(refiner_abstained),
            "ray_max_px": ray_max_px,
            "position_score": position_score,
            "ray_score": ray_score,
            "score": min(position_score, ray_score),
            "score_linear": min(position_score_linear, ray_score_linear),
            "hard_pass": bool(gap_m <= radius_m and ray_max_px <= float(radius_px)),
            **rays,
        }
    )
    record["graded_pass"] = bool(record["score"] >= GRADED_PASS_SCORE)
    record["graded_linear_pass"] = bool(record["score_linear"] >= GRADED_PASS_SCORE)
    return record


def bounce_window_track(
    click_frame: float,
    click_pixel: Sequence[float],
    track: dict[int, Any],
    owner_positions: dict[int, tuple[float, float]],
    window: int = BOUNCE_WITNESS_WINDOW,
) -> tuple[dict[int, tuple[float, float]], int]:
    """Native ball positions around the bounce, owner first, arc-augmented track second.

    The owner's own click replaces whatever sits on its frame.  `subframe_timing` excludes the
    impact frame from both side fits, so the click cannot bias the corner it supplies.
    """
    frame = int(round(float(click_frame)))
    rows: dict[int, tuple[float, float]] = {}
    owner_used = 0
    for candidate in range(frame - window, frame + window + 1):
        owned = owner_positions.get(candidate)
        if owned is not None:
            rows[candidate] = (float(owned[0]), float(owned[1]))
            owner_used += 1
            continue
        automatic = track.get(candidate)
        if automatic is not None:
            rows[candidate] = (float(automatic[0]), float(automatic[1]))
    rows[frame] = (float(click_pixel[0]), float(click_pixel[1]))
    return rows, owner_used


def bounce_witness_v2(
    *,
    camera: Any,
    click_frame: float,
    click_pixel: Sequence[float],
    track: dict[int, Any],
    owner_positions: dict[int, tuple[float, float]],
    fps: float,
    rayplane_xy: Sequence[float],
    click_offset_frames: float = CLICK_OFFSET_FRAMES,
) -> dict[str, Any]:
    """The owner's bounce as a court-plane position with a sub-frame time and a sigma.

    Returns ``formed: False`` with a reason wherever the corner cannot be built -- too few
    track frames on a side, a corner too near collinear, a ball too slow, or a corner more
    than 1.5 frames from the click.  Abstention is the honest answer there; the ray/plane
    construction would still produce a number, and that number is the wrong one.
    """
    from cv.pipeline import subframe_timing
    from cv.pipeline.rich_ball_physics import R_BALL

    frame = int(round(float(click_frame)))
    click_time = float(click_frame) + float(click_offset_frames)
    rows, owner_used = bounce_window_track(click_frame, click_pixel, track, owner_positions)
    homographies = {
        candidate: np.asarray(camera.h_at(candidate), dtype=float)
        for candidate in range(frame - BOUNCE_WITNESS_WINDOW, frame + BOUNCE_WITNESS_WINDOW + 1)
    }
    result: dict[str, Any] = {
        "formed": False,
        "reason": "not_attempted",
        "owner_frames_used": owner_used,
        "track_frames": len(rows),
        "click_time": click_time,
    }
    estimate = subframe_timing.estimate(rows, frame, "bounce", float(fps), homography=homographies)
    result["combined_t_subframe"] = float(estimate.t_subframe)
    result["combined_sigma_frames"] = float(estimate.sigma_frames)

    # The brief's literal construction, kept for comparison: the corner run on the court plane
    # through the per-frame homography.  Its time is as good as the image corner's; its
    # position is not, because every frame it extrapolates from is a ray/plane intersection of
    # a ball that is still in the air.
    court = estimate.witnesses.get("kinematic_court")
    if court is not None and not court.abstain and court.t_subframe is not None:
        intersection = np.asarray(court.detail["intersection"], dtype=float)
        if np.isfinite(intersection).all():
            result["court_xy_kinematic_court"] = [
                float(intersection[0]),
                float(intersection[1]),
            ]
            result["kinematic_court_t_subframe"] = float(court.t_subframe)
            result["kinematic_court_sigma_frames"] = float(court.sigma_frames)

    witness = estimate.witnesses.get("kinematic")
    if witness is None or witness.abstain or witness.t_subframe is None:
        result["reason"] = witness.reason if witness is not None else "no_witness"
        return result
    detail = dict(witness.detail)
    bounce_time = float(witness.t_subframe)
    projection = np.asarray(camera.p_at(bounce_time), dtype=float)

    # The track's own answer, with no click in it.  Reported as a diagnostic only: using it as
    # the truth would grade the fitter against one of its own inputs.
    corner_pixel = np.asarray(detail["intersection"], dtype=float)
    corner_xy = (
        ray_at_height(projection, corner_pixel, R_BALL) if np.isfinite(corner_pixel).all() else None
    )
    if corner_xy is not None:
        result["court_xy_track_corner"] = [float(corner_xy[0]), float(corner_xy[1])]

    # The owner's click, transported along its own side of the path to the bounce time.
    side_frames = detail["frames_in"] if click_time <= bounce_time else detail["frames_out"]
    samples = [(float(f), rows[int(f)]) for f in side_frames if int(f) in rows]
    fitted = fit_image_path(samples, float(frame))
    if fitted is None:
        result["reason"] = "no_side_path"
        return result
    path, residual_px = fitted
    transported = np.asarray(click_pixel, dtype=float) + (path(bounce_time) - path(click_time))
    if not np.isfinite(transported).all():
        result["reason"] = "non_finite_transport"
        return result
    court_xy = ray_at_height(projection, transported, R_BALL)
    scales = metres_per_pixel(projection, transported, R_BALL)
    if court_xy is None or scales is None:
        result["reason"] = "ray_does_not_meet_court"
        return result
    scale, scale_lateral, scale_depth = scales

    # The sigma, propagated: the click's own radius and the side path's residual, both in
    # pixels and both taken to metres through the local court scale, plus the ball's motion
    # over the timing uncertainty.
    speed_px = 0.5 * (float(detail["speed_in"]) + float(detail["speed_out"]))
    sigma_frames = float(witness.sigma_frames)
    sigma_click_m = scale * CLICK_SIGMA_PX
    sigma_path_m = scale * residual_px
    sigma_time_m = scale * speed_px * sigma_frames
    sigma_m = math.sqrt(sigma_click_m**2 + sigma_path_m**2 + sigma_time_m**2)
    if not math.isfinite(sigma_m):
        result["reason"] = "non_finite_sigma"
        return result
    result.update(
        {
            "formed": True,
            "reason": witness.reason,
            "t_subframe": bounce_time,
            "sigma_frames": sigma_frames,
            "offset_frames": bounce_time - click_time,
            "court_xy": [float(court_xy[0]), float(court_xy[1])],
            "transported_px": [float(transported[0]), float(transported[1])],
            "transport_px": float(np.linalg.norm(transported - np.asarray(click_pixel, float))),
            "sigma_m": float(sigma_m),
            "sigma_click_m": float(sigma_click_m),
            "sigma_path_m": float(sigma_path_m),
            "sigma_time_m": float(sigma_time_m),
            "sigma_track_only_m": float(math.hypot(sigma_path_m, sigma_time_m)),
            "sigma_click_lateral_m": float(scale_lateral * CLICK_SIGMA_PX),
            "sigma_click_depth_m": float(scale_depth * CLICK_SIGMA_PX),
            "metres_per_pixel": float(scale),
            "side": "inbound" if click_time <= bounce_time else "outbound",
            "side_frames": len(samples),
            "turn_deg": float(detail.get("turn_deg", float("nan"))),
            "rayplane_shift_m": float(
                np.linalg.norm(court_xy - np.asarray(rayplane_xy, dtype=float))
            ),
        }
    )
    return result


# --- per-flight truth ledger --------------------------------------------------------------


def flight_truth_row(
    *,
    point: dict[str, Any],
    attempt: dict[str, Any],
    fit: dict[str, Any] | None,
    camera: Any,
    owner_events: dict[str, list[dict[str, Any]]],
    owner_positions: dict[int, tuple[float, float]],
    court_bounces: list[dict[str, Any]],
    junction_prev: float | None,
    junction_next: float | None,
    pixel_accepted: bool,
    metric_row: dict[str, Any],
    track: dict[int, Any] | None = None,
    contact_radius_cache: dict[int, tuple[float, bool]] | None = None,
    junction_witnesses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    point_id = str(point["point"])
    index = int(attempt["flight_index"])
    row: dict[str, Any] = {
        "point": point_id,
        "match_id": str(point["match_id"]),
        "scope": scope_of(point_id),
        "flight_index": index,
        "flight_id": f"{point_id}__flight_{index:03d}",
        "start_frame": attempt.get("start_frame"),
        "end_frame": attempt.get("end_frame"),
        "terminal_end": bool(attempt.get("terminal_end")),
        "start_phase": attempt.get("start_phase"),
        "solved": fit is not None,
        "pixel_accepted": bool(pixel_accepted),
        "metric_accepted": bool(metric_row.get("metric_gate_accepted")),
        "metric_reasons": metric_row.get("metric_reasons"),
        "junction_gap_prev_m": junction_prev,
        "junction_gap_next_m": junction_next,
    }
    if fit is None:
        return row

    start = float(attempt["start_frame"])
    end = float(attempt["end_frame"])

    # Label-blind fit features the shippable gate may read.
    row.update(
        {
            "held_out_median_px": fit.get("held_out_reprojection_median_px"),
            "held_out_p90_px": fit.get("held_out_reprojection_p90_px"),
            "rms_px": fit.get("rms_px"),
            "anchor_satisfied": fit.get("anchor_satisfied"),
            "anchor_max_error_m": fit.get("anchor_max_error_m"),
            "net_anchor_available": fit.get("net_anchor_available"),
            "observation_coverage": fit.get("observation_coverage"),
            "observations": fit.get("observations"),
            "speed_kmh": fit.get("speed_kmh"),
            "minimum_height_m": fit.get("minimum_height_m"),
            "bounce_count": len(fit.get("bounces", [])),
            "start_contact_reprojection_px": fit.get("start_contact_reprojection_px"),
            "end_contact_reprojection_px": fit.get("end_contact_reprojection_px"),
            "contact_reprojection_max_px": max(
                [
                    float(value)
                    for key in ("start_contact_reprojection_px", "end_contact_reprojection_px")
                    if (value := fit.get(key)) is not None and math.isfinite(float(value))
                ],
                default=None,
            ),
            "contact_height_min_m": metric_row.get("contact_height_min_m"),
            "contact_height_max_m": metric_row.get("contact_height_max_m"),
            "contact_reach_max_m": metric_row.get("contact_reach_max_m"),
            "contact_height_plausible": metric_row.get("contact_height_plausible"),
            "contact_reach_plausible": metric_row.get("contact_reach_plausible"),
            "net_crossing_required": metric_row.get("net_crossing_required"),
            "net_crossing_height_m": metric_row.get("net_crossing_height_m"),
            "net_crossing_plausible": metric_row.get("net_crossing_plausible"),
        }
    )

    # Witness 1: owner bounce clicks, projected to the court plane, in metres.
    # Two constructions, reported side by side on every flight: ``bounce_rayplane_v1`` is the
    # shipped click-ray/court-plane intersection, ``bounce_kinematic_v2`` is the corrected
    # corner witness with its own sub-frame time and sigma.
    scoped_bounces = [row_ for row_ in court_bounces if start < float(row_["frame"]) < end]
    fitted_bounces = fit.get("bounces", [])
    bounce_errors: list[float] = []
    bounce_image_errors: list[float] = []
    bounce_court_y: list[float] = []
    bounce_missing = False
    for truth in scoped_bounces:
        candidates = [
            bounce
            for bounce in fitted_bounces
            if abs(float(bounce["frame"]) - float(truth["frame"])) <= BOUNCE_MATCH_FRAMES
        ]
        if not candidates:
            bounce_missing = True
            continue
        selected = min(
            candidates, key=lambda bounce: abs(float(bounce["frame"]) - float(truth["frame"]))
        )
        bounce_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(selected["x"][:2], dtype=float)
                    - np.asarray(truth["court_xy"], dtype=float)
                )
            )
        )
        bounce_court_y.append(float(truth["court_xy"][1]))
        # How far the same disagreement is in the image, so metres and pixels are comparable.
        if truth.get("image_xy") is not None:
            uv = project(camera.p_at(float(truth["frame"])), selected["x"])
            if uv is not None:
                bounce_image_errors.append(
                    float(
                        math.hypot(
                            uv[0] - truth["image_xy"][0],
                            uv[1] - truth["image_xy"][1],
                        )
                    )
                )
    row["truth_bounce_image_error_max_px"] = max(bounce_image_errors, default=None)
    row["truth_bounce_court_y_m"] = max(bounce_court_y, default=None)
    row["owner_bounce_witnesses"] = len(scoped_bounces)
    row["owner_bounce_missing"] = bounce_missing
    row["truth_bounce_error_max_m"] = max(bounce_errors, default=None)
    row["truth_bounce_pass"] = (
        None
        if not scoped_bounces
        else bool(
            not bounce_missing and all(error <= BOUNCE_TRUTH_LIMIT_M for error in bounce_errors)
        )
    )

    # Witness 2: owner-positioned native frames inside the flight span.
    witnesses = [
        (float(frame), pixel)
        for frame, pixel in sorted(owner_positions.items())
        if start <= float(frame) <= end
    ]
    errors = reprojection_errors(fit, camera, witnesses)
    row["owner_position_frames"] = len(errors)
    row["truth_reprojection_median_px"] = percentile([e["error_px"] for e in errors], 50.0)
    row["truth_reprojection_p90_px"] = percentile([e["error_px"] for e in errors], 90.0)
    if len(errors) >= MIN_OWNER_POSITION_FRAMES:
        row["truth_position_pass"] = bool(
            row["truth_reprojection_median_px"] <= OWNER_POSITION_MEDIAN_LIMIT_PX
            and row["truth_reprojection_p90_px"] <= OWNER_POSITION_P90_LIMIT_PX
        )
    else:
        row["truth_position_pass"] = None

    # Witness 3: owner contact clicks at either boundary.
    #
    # Two readings, reported side by side.  ``truth_contact_*_px`` evaluates the fitted flight
    # at the owner's own (half-frame) click time, which is what the shipped audit does; a
    # fitter whose contact sits at a different sub-frame time than the click's frame is
    # charged for the ball's motion over that gap.  ``truth_contact_*_at_fit_px`` evaluates
    # the flight at the fitter's own fitted contact time instead, so only the position is
    # compared.  Both are against the same owner click pixel.
    contact_errors: list[float] = []
    at_fit_errors: list[float] = []
    contact_scores: list[float] = []
    contact_scores_linear: list[float] = []
    at_fit_scores: list[float] = []
    contact_radii: list[float] = []
    contact_radius_abstained = False
    for boundary, key in ((start, "start"), (end, "end")):
        if key == "end" and attempt.get("terminal_end"):
            continue
        candidates = [
            click
            for click in owner_events.get("contact", [])
            if click["image_xy"] is not None
            and abs(float(click["frame"]) - boundary) <= CONTACT_MATCH_FRAMES
        ]
        if not candidates:
            row[f"truth_contact_{key}_px"] = None
            row[f"truth_contact_{key}_at_fit_px"] = None
            continue
        click = min(candidates, key=lambda row_: abs(float(row_["frame"]) - boundary))
        # The circle this contact is judged inside: the refiner's own radius for it.
        radius_px, refiner_abstained = contact_graded_radius_px(
            track, boundary, contact_radius_cache
        )
        row[f"truth_contact_{key}_radius_px"] = radius_px
        row[f"truth_contact_{key}_refiner_abstained"] = refiner_abstained
        contact_radii.append(radius_px)
        contact_radius_abstained = contact_radius_abstained or refiner_abstained
        measured = reprojection_errors(fit, camera, [(float(click["frame"]), click["image_xy"])])
        value = measured[0]["error_px"] if measured else None
        row[f"truth_contact_{key}_px"] = value
        if value is not None:
            contact_errors.append(value)
            contact_scores.append(graded_score(float(value), radius_px))
            contact_scores_linear.append(graded_score_linear(float(value), radius_px))
        # The fitter's own contact: its fitted (possibly sub-frame) time and its end state.
        fitted_time = fit.get(f"{key}_frame")
        fitted_xyz = fit.get(f"{key}_xyz")
        at_fit = None
        if fitted_time is not None and fitted_xyz is not None:
            uv = project(camera.p_at(float(fitted_time)), fitted_xyz)
            if uv is not None:
                at_fit = float(
                    math.hypot(
                        uv[0] - click["image_xy"][0],
                        uv[1] - click["image_xy"][1],
                    )
                )
        row[f"truth_contact_{key}_at_fit_px"] = at_fit
        row[f"truth_contact_{key}_time_offset_frames"] = (
            None if fitted_time is None else float(fitted_time) - float(click["frame"])
        )
        if at_fit is not None:
            at_fit_errors.append(at_fit)
            at_fit_scores.append(graded_score(float(at_fit), radius_px))
    row["truth_contact_max_px"] = max(contact_errors, default=None)
    row["truth_contact_pass"] = (
        None if not contact_errors else bool(max(contact_errors) <= OWNER_CONTACT_LIMIT_PX)
    )
    row["truth_contact_at_fit_max_px"] = max(at_fit_errors, default=None)
    row["truth_contact_at_fit_pass"] = (
        None if not at_fit_errors else bool(max(at_fit_errors) <= OWNER_CONTACT_LIMIT_PX)
    )
    # The same contact witness, graded inside the refiner's circle.  The flight takes its worst
    # boundary, exactly as the hard verdict takes its worst error.
    row["truth_contact_radius_max_px"] = max(contact_radii, default=None)
    row["truth_contact_refiner_abstained"] = contact_radius_abstained if contact_radii else None
    row["truth_contact_score"] = min(contact_scores, default=None)
    row["truth_contact_score_linear"] = min(contact_scores_linear, default=None)
    row["truth_contact_graded_pass"] = (
        None if not contact_scores else bool(min(contact_scores) >= GRADED_PASS_SCORE)
    )
    row["truth_contact_graded_linear_pass"] = (
        None if not contact_scores_linear else bool(min(contact_scores_linear) >= GRADED_PASS_SCORE)
    )
    row["truth_contact_at_fit_score"] = min(at_fit_scores, default=None)
    row["truth_contact_at_fit_graded_pass"] = (
        None if not at_fit_scores else bool(min(at_fit_scores) >= GRADED_PASS_SCORE)
    )

    # Witness 1b: the same owner bounce clicks under ``bounce_kinematic_v2``.
    v2_errors: list[float] = []
    v2_sigmas: list[float] = []
    v2_offsets: list[float] = []
    v2_transports: list[float] = []
    v2_scores: list[float] = []
    v2_scores_linear: list[float] = []
    v2_radii: list[float] = []
    v2_formed = 0
    v2_missing = False
    track_errors: list[float] = []
    for truth in scoped_bounces:
        witness = truth.get("v2") or {}
        if witness.get("court_xy_track_corner") is not None and witness.get("t_subframe"):
            track_time = float(witness["t_subframe"])
            near = [
                bounce
                for bounce in fitted_bounces
                if abs(float(bounce["frame"]) - track_time) <= BOUNCE_V2_MATCH_FRAMES
            ]
            if near:
                chosen = min(near, key=lambda bounce: abs(float(bounce["frame"]) - track_time))
                track_errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(chosen["x"][:2], dtype=float)
                            - np.asarray(witness["court_xy_track_corner"], dtype=float)
                        )
                    )
                )
        if not witness.get("formed"):
            continue
        v2_formed += 1
        v2_sigmas.append(float(witness["sigma_m"]))
        v2_transports.append(float(witness["transport_px"]))
        v2_offsets.append(float(witness["offset_frames"]))
        radius_m = bounce_graded_radius_m(float(witness["sigma_m"]))
        v2_radii.append(radius_m)
        truth_time = float(witness["t_subframe"])
        candidates = [
            bounce
            for bounce in fitted_bounces
            if abs(float(bounce["frame"]) - truth_time) <= BOUNCE_V2_MATCH_FRAMES
        ]
        if not candidates:
            v2_missing = True
            # No fitted bounce at all where the owner saw one is not a distance, it is no
            # answer.  It scores zero under the graded rule, as it fails under the hard one.
            v2_scores.append(0.0)
            v2_scores_linear.append(0.0)
            continue
        selected = min(candidates, key=lambda bounce: abs(float(bounce["frame"]) - truth_time))
        error_m = float(
            np.linalg.norm(
                np.asarray(selected["x"][:2], dtype=float)
                - np.asarray(witness["court_xy"], dtype=float)
            )
        )
        v2_errors.append(error_m)
        v2_scores.append(graded_score(error_m, radius_m))
        v2_scores_linear.append(graded_score_linear(error_m, radius_m))
    row["truth_bounce_track_error_max_m"] = max(track_errors, default=None)
    row["owner_bounce_v2_formed"] = v2_formed
    row["owner_bounce_v2_missing"] = v2_missing
    row["truth_bounce_v2_error_max_m"] = max(v2_errors, default=None)
    row["truth_bounce_v2_sigma_max_m"] = max(v2_sigmas, default=None)
    row["truth_bounce_v2_transport_max_px"] = max(v2_transports, default=None)
    row["truth_bounce_v2_offset_max_frames"] = max(v2_offsets, key=abs) if v2_offsets else None
    row["truth_bounce_v2_pass"] = (
        None
        if not v2_formed
        else bool(not v2_missing and all(error <= BOUNCE_TRUTH_LIMIT_M for error in v2_errors))
    )
    # The owner's name for the hard 10 cm verdict on the corrected witness, kept beside the
    # graded one for comparison.
    row["bounce_hard_v1"] = row["truth_bounce_v2_pass"]
    row["truth_bounce_v2_radius_max_m"] = max(v2_radii, default=None)
    row["truth_bounce_v2_score"] = min(v2_scores, default=None)
    row["truth_bounce_v2_score_linear"] = min(v2_scores_linear, default=None)
    row["truth_bounce_v2_graded_pass"] = (
        None if not v2_scores else bool(min(v2_scores) >= GRADED_PASS_SCORE)
    )
    row["truth_bounce_v2_graded_linear_pass"] = (
        None if not v2_scores_linear else bool(min(v2_scores_linear) >= GRADED_PASS_SCORE)
    )
    # The same verdict, restricted to witnesses whose own width can resolve the 0.10 m line.
    row["truth_bounce_v2_resolved_pass"] = (
        row["truth_bounce_v2_pass"]
        if v2_sigmas and max(v2_sigmas) <= BOUNCE_V2_SIGMA_LIMIT_M
        else None
    )

    # Witness 4: the junction, the one witness that spans a contact.  A flight takes its worst
    # shared contact, exactly as the other witnesses take their worst boundary.
    formed = list(junction_witnesses or [])
    row["junction_witnesses_formed"] = len(formed)
    row["truth_junction_gap_max_m"] = max((w["gap_m"] for w in formed), default=None)
    row["truth_junction_radius_max_m"] = max((w["radius_m"] for w in formed), default=None)
    row["truth_junction_sigma_max_m"] = max((w["sigma_m"] for w in formed), default=None)
    row["truth_junction_ray_max_px"] = max((w["ray_max_px"] for w in formed), default=None)
    row["truth_junction_click_offset_max_frames"] = max(
        (abs(w["click_frame"] - w["boundary_frame"]) for w in formed), default=None
    )
    row["truth_junction_radius_px"] = max((w["radius_px"] for w in formed), default=None)
    row["truth_junction_position_score"] = min((w["position_score"] for w in formed), default=None)
    row["truth_junction_ray_score"] = min((w["ray_score"] for w in formed), default=None)
    row["truth_junction_score"] = min((w["score"] for w in formed), default=None)
    row["truth_junction_score_linear"] = min((w["score_linear"] for w in formed), default=None)
    row["truth_junction_pass"] = None if not formed else bool(all(w["hard_pass"] for w in formed))
    row["truth_junction_graded_pass"] = (
        None if not formed else bool(all(w["graded_pass"] for w in formed))
    )
    row["truth_junction_graded_linear_pass"] = (
        None if not formed else bool(all(w["graded_linear_pass"] for w in formed))
    )

    checks = [row["truth_bounce_pass"], row["truth_position_pass"], row["truth_contact_pass"]]
    present = [value for value in checks if value is not None]
    row["truth_witnesses"] = len(present)
    row["truth_good"] = None if not present else bool(all(present))
    blind = [row["truth_position_pass"], row["truth_contact_pass"]]
    blind_present = [value for value in blind if value is not None]
    row["gate_blind_witnesses"] = len(blind_present)
    row["gate_blind_truth_good"] = None if not blind_present else bool(all(blind_present))

    # The same three verdicts under the corrected witnesses.  ``truth_good_v2`` and
    # ``truth_good_graded`` -- the corrected and the graded verdicts -- now require the
    # junction witness wherever it can be formed; ``*_noj`` is the same verdict without it,
    # which is what every earlier audit reported.
    for name, bounce_key, contact_key, junction_key in (
        ("truth_good_v2", "truth_bounce_v2_pass", "truth_contact_pass", "truth_junction_pass"),
        ("truth_good_v2_noj", "truth_bounce_v2_pass", "truth_contact_pass", None),
        ("truth_good_v2_at_fit", "truth_bounce_v2_pass", "truth_contact_at_fit_pass", None),
        (
            "truth_good_v2_resolved",
            "truth_bounce_v2_resolved_pass",
            "truth_contact_pass",
            None,
        ),
        (
            "truth_good_graded",
            "truth_bounce_v2_graded_pass",
            "truth_contact_graded_pass",
            "truth_junction_graded_pass",
        ),
        (
            "truth_good_graded_noj",
            "truth_bounce_v2_graded_pass",
            "truth_contact_graded_pass",
            None,
        ),
        (
            "truth_good_graded_linear",
            "truth_bounce_v2_graded_linear_pass",
            "truth_contact_graded_linear_pass",
            "truth_junction_graded_linear_pass",
        ),
        (
            "truth_good_graded_linear_noj",
            "truth_bounce_v2_graded_linear_pass",
            "truth_contact_graded_linear_pass",
            None,
        ),
        (
            "truth_good_graded_at_fit",
            "truth_bounce_v2_graded_pass",
            "truth_contact_at_fit_graded_pass",
            None,
        ),
    ):
        values = [row[bounce_key], row["truth_position_pass"], row[contact_key]]
        if junction_key is not None:
            values.append(row[junction_key])
        found = [value for value in values if value is not None]
        row[name] = None if not found else bool(all(found))
    blind_at_fit = [row["truth_position_pass"], row["truth_contact_at_fit_pass"]]
    blind_at_fit_present = [value for value in blind_at_fit if value is not None]
    row["gate_blind_truth_good_at_fit"] = (
        None if not blind_at_fit_present else bool(all(blind_at_fit_present))
    )
    blind_graded = [row["truth_position_pass"], row["truth_contact_graded_pass"]]
    blind_graded_present = [value for value in blind_graded if value is not None]
    row["gate_blind_truth_good_graded"] = (
        None if not blind_graded_present else bool(all(blind_graded_present))
    )
    return row


def junction_gap_map(point: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Per-flight incoming/outgoing junction gap, recomputed the way the pipeline does."""
    fits = sorted(
        (fit for fit in point.get("fits", []) if fit.get("start_frame") is not None),
        key=lambda fit: float(fit["start_frame"]),
    )
    gaps: dict[int, dict[str, float]] = defaultdict(dict)
    for incoming, outgoing in zip(fits, fits[1:]):
        if incoming.get("end_frame") != outgoing.get("start_frame"):
            continue
        incoming_end = incoming.get("end_xyz")
        if incoming_end is None and incoming.get("trajectory"):
            incoming_end = incoming["trajectory"][-1]["xyz"]
        outgoing_start = outgoing.get("start_xyz")
        if incoming_end is None or outgoing_start is None:
            continue
        value = float(
            np.linalg.norm(
                np.asarray(incoming_end, dtype=float) - np.asarray(outgoing_start, dtype=float)
            )
        )
        gaps[int(incoming["flight_index"])]["next"] = value
        gaps[int(outgoing["flight_index"])]["prev"] = value
    return dict(gaps)


def junction_witness_map(
    point: dict[str, Any],
    camera: Any,
    owner_events: dict[str, list[dict[str, Any]]],
    track: dict[int, Any] | None,
    contact_radius_cache: dict[int, tuple[float, bool]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Every internal contact of a point that carries an owner contact click, judged.

    One witness per shared boundary, attached to **both** flights that share it, so a
    disagreement fails the pair rather than one side of it.
    """
    witnesses: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if camera is None:
        return {}
    fits = sorted(
        (fit for fit in point.get("fits", []) if fit.get("start_frame") is not None),
        key=lambda fit: float(fit["start_frame"]),
    )
    clicks = [
        click for click in owner_events.get("contact", []) if click.get("image_xy") is not None
    ]
    for incoming, outgoing in zip(fits, fits[1:]):
        if incoming.get("end_frame") is None or incoming["end_frame"] != outgoing.get(
            "start_frame"
        ):
            continue
        boundary = float(incoming["end_frame"])
        candidates = [
            click
            for click in clicks
            if abs(float(click["frame"]) - boundary) <= CONTACT_MATCH_FRAMES
        ]
        if not candidates:
            continue
        click = min(candidates, key=lambda row_: abs(float(row_["frame"]) - boundary))
        radius_px, refiner_abstained = contact_graded_radius_px(
            track, boundary, contact_radius_cache
        )
        incoming_end = incoming.get("end_xyz")
        if incoming_end is None and incoming.get("trajectory"):
            incoming_end = incoming["trajectory"][-1]["xyz"]
        outgoing_start = outgoing.get("start_xyz")
        if outgoing_start is None and outgoing.get("trajectory"):
            outgoing_start = outgoing["trajectory"][0]["xyz"]
        record = junction_witness(
            camera=camera,
            boundary_frame=boundary,
            click=click,
            inbound_end_xyz=incoming_end,
            outbound_start_xyz=outgoing_start,
            radius_px=radius_px,
            refiner_abstained=refiner_abstained,
        )
        if not record.get("formed"):
            continue
        for fit, side in ((incoming, "end"), (outgoing, "start")):
            witnesses[int(fit["flight_index"])].append({**record, "side": side})
    return dict(witnesses)


def build_flight_rows(
    report: dict[str, Any],
    ledger: dict[str, Any],
    camera_root: Path,
    truth_events: Path,
    click_offset_frames: float = CLICK_OFFSET_FRAMES,
) -> list[dict[str, Any]]:
    pixel = {
        (str(row["point"]), int(row["flight_index"])): row["status"] == "provisional_valid"
        for row in ledger["rows"]
    }
    owner_events = owner_event_index(truth_events)
    positions = owner_position_index()
    rows: list[dict[str, Any]] = []
    for point in report["points_detail"]:
        point_id = str(point["point"])
        match_id = str(point["match_id"])
        clip = str(point["clip"])
        attempts = point.get("flight_attempts") or []
        if not attempts:
            continue
        try:
            camera = reconstruction.PointCamera(camera_root / match_id, clip)
        except ValueError:
            camera = None
        events = owner_events.get(point_id, {name: [] for name in PHYSICAL_EVENTS})
        court_bounces = []
        if camera is not None:
            for click in events.get("bounce", []):
                if click["image_xy"] is None:
                    continue
                court_bounces.append(
                    {
                        "frame": float(click["frame"]),
                        "fps": click.get("fps") or float(point.get("fps") or 25.0),
                        "image_xy": click["image_xy"],
                        "court_xy": np.asarray(
                            reconstruction.court_point(
                                camera,
                                np.asarray(click["image_xy"], dtype=float),
                                float(click["frame"]),
                            ),
                            dtype=float,
                        ),
                    }
                )
        track: dict[int, Any] = {}
        if camera is not None:
            track = reconstruction.load_track(camera_root / match_id, clip)
        if camera is not None and court_bounces:
            clip_positions = positions.get((match_id, clip), {})
            for click in court_bounces:
                click["v2"] = bounce_witness_v2(
                    camera=camera,
                    click_frame=click["frame"],
                    click_pixel=click["image_xy"],
                    track=track,
                    owner_positions=clip_positions,
                    fps=click["fps"],
                    rayplane_xy=click["court_xy"],
                    click_offset_frames=click_offset_frames,
                )
        gaps = junction_gap_map(point)
        fits = {int(fit["flight_index"]): fit for fit in point.get("fits", [])}
        owner_bounce_metric = [
            {"frame": row_["frame"], "court_xy": row_["court_xy"]} for row_ in court_bounces
        ]  # the metric scorer's expected shape, without the extra image column
        contact_radius_cache: dict[int, tuple[float, bool]] = {}
        junctions = junction_witness_map(point, camera, events, track, contact_radius_cache)
        for attempt in attempts:
            index = int(attempt["flight_index"])
            fit = fits.get(index)
            metric_row = flight_metric_row(
                point, attempt, fit, owner_bounce_metric, pixel.get((point_id, index), False)
            )
            if camera is None:
                metric_row = {**metric_row, "metric_gate_accepted": False}
            rows.append(
                flight_truth_row(
                    point=point,
                    attempt=attempt,
                    fit=fit,
                    camera=camera,
                    owner_events=events,
                    owner_positions=positions.get((match_id, clip), {}),
                    court_bounces=court_bounces,
                    junction_prev=gaps.get(index, {}).get("prev"),
                    junction_next=gaps.get(index, {}).get("next"),
                    pixel_accepted=pixel.get((point_id, index), False),
                    metric_row=metric_row,
                    track=track,
                    contact_radius_cache=contact_radius_cache,
                    junction_witnesses=junctions.get(index),
                )
            )
    return rows


# --- cross-tabulation ---------------------------------------------------------------------


def _truth_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return {
        "witnessed": len(values),
        "pass": sum(bool(value) for value in values),
        "pass_rate": (sum(bool(value) for value in values) / len(values)) if values else None,
    }


def crosstab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [row for row in rows if row.get("solved")]
    cells: dict[str, list[dict[str, Any]]] = {
        "both_accepted": [],
        "metric_only": [],
        "pixel_only": [],
        "neither": [],
    }
    for row in solved:
        pixel = bool(row["pixel_accepted"])
        metric = bool(row["metric_accepted"])
        name = (
            "both_accepted"
            if pixel and metric
            else "metric_only"
            if metric
            else "pixel_only"
            if pixel
            else "neither"
        )
        cells[name].append(row)
    summary = {}
    for name, cell in cells.items():
        summary[name] = {
            "flights": len(cell),
            "truth_bounce": _truth_summary(cell, "truth_bounce_pass"),
            "truth_position": _truth_summary(cell, "truth_position_pass"),
            "truth_contact": _truth_summary(cell, "truth_contact_pass"),
            "truth_good_all_witnesses": _truth_summary(cell, "truth_good"),
            "truth_good_graded": _truth_summary(cell, "truth_good_graded"),
            "truth_good_gate_blind_witnesses": _truth_summary(cell, "gate_blind_truth_good"),
            "bounce_error_median_p90_m": [
                percentile(
                    [
                        r["truth_bounce_error_max_m"]
                        for r in cell
                        if r.get("truth_bounce_error_max_m") is not None
                    ],
                    q,
                )
                for q in (50.0, 90.0)
            ],
            "truth_reprojection_median_p90_px": [
                percentile(
                    [
                        r["truth_reprojection_median_px"]
                        for r in cell
                        if r.get("truth_reprojection_median_px") is not None
                    ],
                    q,
                )
                for q in (50.0, 90.0)
            ],
            "truth_contact_median_p90_px": [
                percentile(
                    [
                        r["truth_contact_max_px"]
                        for r in cell
                        if r.get("truth_contact_max_px") is not None
                    ],
                    q,
                )
                for q in (50.0, 90.0)
            ],
            "junction_gap_median_p90_m": [
                percentile(
                    [
                        value
                        for r in cell
                        for value in (r.get("junction_gap_prev_m"), r.get("junction_gap_next_m"))
                        if value is not None
                    ],
                    q,
                )
                for q in (50.0, 90.0)
            ],
            "held_out_median_px_median": percentile(
                [r["held_out_median_px"] for r in cell if r.get("held_out_median_px") is not None],
                50.0,
            ),
        }
    summary["solved_flights"] = len(solved)
    return summary


# --- per-point primary cause ----------------------------------------------------------------


def point_cause(
    *,
    point: dict[str, Any],
    flight_rows: list[dict[str, Any]],
    truth_events: dict[str, list[dict[str, Any]]],
    automatic_events: dict[str, list[float]],
) -> dict[str, Any]:
    point_id = str(point["point"])
    result: dict[str, Any] = {
        "point": point_id,
        "match_id": str(point["match_id"]),
        "scope": scope_of(point_id),
        "truth_contacts": len(truth_events.get("contact", [])),
        "truth_bounces": len(truth_events.get("bounce", [])),
        "attempted_flights": len(flight_rows),
        "solved_flights": sum(bool(row["solved"]) for row in flight_rows),
        "pixel_accepted_flights": sum(bool(row["pixel_accepted"]) for row in flight_rows),
        "metric_accepted_flights": sum(bool(row["metric_accepted"]) for row in flight_rows),
    }
    camera = point.get("camera_calibration", {}) or {}
    if camera.get("accepted") is not True:
        result.update(
            {
                "cause": "camera_abstained",
                "detail": str(camera.get("reason") or "camera_calibration_unreliable"),
            }
        )
        return result

    truth_contacts = [row["frame"] for row in truth_events.get("contact", [])]
    missing_contacts = unmatched_truth_events(
        truth_contacts, automatic_events.get("contact", []), EVENT_MATCH_FRAMES
    )
    result["missing_truth_contacts"] = missing_contacts
    if missing_contacts:
        result.update(
            {
                "cause": "event_missing",
                "detail": f"contact_recall:{len(truth_contacts) - missing_contacts}"
                f"/{len(truth_contacts)}",
            }
        )
        return result

    expected = len(truth_contacts)
    terminal = sum(bool(row["terminal_end"]) for row in flight_rows)
    if len(flight_rows) < expected or terminal != 1:
        reasons = point.get("screen_reasons") or []
        detail = (
            f"attempted:{len(flight_rows)}/{expected}"
            if len(flight_rows) < expected
            else f"terminal_flights:{terminal}"
        )
        if reasons:
            detail = f"{detail};{','.join(sorted(reasons))}"
        result.update({"cause": "flight_not_attempted", "detail": detail})
        return result

    unsolved = [row for row in flight_rows if not row["solved"]]
    if unsolved:
        ledger_reasons = Counter(
            reason for row in unsolved for reason in row.get("ledger_reasons", [])
        )
        result.update(
            {
                "cause": "flight_not_solved",
                "detail": (
                    f"{len(unsolved)}/{len(flight_rows)}:"
                    f"{ledger_reasons.most_common(1)[0][0] if ledger_reasons else 'physics_fit_failed'}"
                ),
            }
        )
        return result

    if any(row.get("owner_bounce_missing") for row in flight_rows):
        count = sum(bool(row.get("owner_bounce_missing")) for row in flight_rows)
        result.update({"cause": "missing_bounce", "detail": f"{count}/{len(flight_rows)}"})
        return result

    rejected = [row for row in flight_rows if not row["pixel_accepted"]]
    if rejected:
        reasons = Counter(reason for row in rejected for reason in row.get("ledger_reasons", []))
        result.update(
            {
                "cause": "solved_but_rejected",
                "detail": (
                    f"{len(rejected)}/{len(flight_rows)}:"
                    f"{reasons.most_common(1)[0][0] if reasons else 'unknown'}"
                ),
            }
        )
        return result

    gaps = [
        value
        for row in flight_rows
        for value in (row.get("junction_gap_prev_m"), row.get("junction_gap_next_m"))
        if value is not None
    ]
    if gaps and max(gaps) > JUNCTION_GAP_LIMIT_M:
        result.update({"cause": "junction_gap", "detail": f"max_gap_m:{max(gaps):.3f}"})
        return result

    result.update({"cause": "complete", "detail": ""})
    return result


def build_point_causes(
    report: dict[str, Any],
    ledger: dict[str, Any],
    flight_rows: list[dict[str, Any]],
    truth_events: Path,
    automatic_events_path: Path,
) -> list[dict[str, Any]]:
    owner_events = owner_event_index(truth_events)
    automatic = automatic_event_index(automatic_events_path)
    ledger_reasons = {
        (str(row["point"]), int(row["flight_index"])): list(row.get("reasons", []))
        for row in ledger["rows"]
    }
    enriched: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flight_rows:
        enriched[row["point"]].append(
            {**row, "ledger_reasons": ledger_reasons.get((row["point"], row["flight_index"]), [])}
        )
    causes = []
    for point in report["points_detail"]:
        point_id = str(point["point"])
        if point_id not in owner_events:
            continue
        causes.append(
            point_cause(
                point=point,
                flight_rows=sorted(enriched.get(point_id, []), key=lambda row: row["flight_index"]),
                truth_events=owner_events[point_id],
                automatic_events=automatic.get(point_id, {name: [] for name in PHYSICAL_EVENTS}),
            )
        )
    return causes


# --- shippable gate search -------------------------------------------------------------------

CONTACT_HEIGHT_RANGE_M = (0.30, 3.50)
INF = float("inf")


def gate_accepts(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    """Evaluate one label-blind candidate gate on one solved flight row."""
    if not row.get("solved"):
        return False

    def number(key: str) -> float | None:
        value = row.get(key)
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    held_median = number("held_out_median_px")
    if held_median is None or held_median > gate["held_out_median_px"]:
        return False
    held_p90 = number("held_out_p90_px")
    if gate["held_out_p90_px"] < INF and (held_p90 is None or held_p90 > gate["held_out_p90_px"]):
        return False
    anchor_error = number("anchor_max_error_m")
    if gate["anchor_max_error_m"] < INF and (
        anchor_error is None or anchor_error > gate["anchor_max_error_m"]
    ):
        return False
    contact_px = number("contact_reprojection_max_px")
    if gate["contact_reprojection_px"] < INF and (
        contact_px is not None and contact_px > gate["contact_reprojection_px"]
    ):
        return False
    reach = number("contact_reach_max_m")
    if gate["contact_reach_m"] < INF and (reach is None or reach > gate["contact_reach_m"]):
        return False
    if gate["contact_height"]:
        low, high = number("contact_height_min_m"), number("contact_height_max_m")
        if low is None or high is None:
            return False
        if low < CONTACT_HEIGHT_RANGE_M[0] or high > CONTACT_HEIGHT_RANGE_M[1]:
            return False
    if gate["net_crossing"] and row.get("net_crossing_plausible") is not True:
        return False
    minimum_height = number("minimum_height_m")
    if minimum_height is not None and minimum_height < 0.018:
        return False
    if int(row.get("bounce_count") or 0) > 1:
        return False
    speed = number("speed_kmh")
    if speed is None or not 5.0 <= speed <= 260.0:
        return False
    coverage = number("observation_coverage")
    if coverage is None or coverage < 0.20:
        return False
    return True


def gate_grid() -> list[dict[str, Any]]:
    """Label-blind candidate gates.  Every knob is a quantity the fitter already emits."""
    grid = []
    for held_median in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0):
        for held_p90 in (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, INF):
            for anchor_error in (0.05, 0.10, 0.15, 0.25, INF):
                for contact_px in (12.0, 20.0, 40.0, INF):
                    for reach in (2.1, 2.5, INF):
                        grid.append(
                            {
                                "held_out_median_px": held_median,
                                "held_out_p90_px": held_p90,
                                "anchor_max_error_m": anchor_error,
                                "contact_reprojection_px": contact_px,
                                "contact_reach_m": reach,
                                "contact_height": True,
                                "net_crossing": True,
                            }
                        )
    return grid


def complete_points(
    rows: list[dict[str, Any]], accepted: dict[str, bool], points: set[str]
) -> set[str]:
    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["point"] in points:
            by_point[row["point"]].append(row)
    result = set()
    for point, flights in by_point.items():
        if not flights:
            continue
        if sum(bool(row["terminal_end"]) for row in flights) != 1:
            continue
        if all(accepted.get(row["flight_id"], False) for row in flights):
            result.add(point)
    return result


WITNESS_KEYS = (
    "truth_good",
    "truth_good_v2",
    "truth_good_v2_noj",
    "truth_good_graded_noj",
    "truth_junction_pass",
    "truth_junction_graded_pass",
    "truth_good_v2_at_fit",
    "truth_good_v2_resolved",
    "truth_good_graded",
    "truth_good_graded_linear",
    "truth_good_graded_at_fit",
    "gate_blind_truth_good",
    "gate_blind_truth_good_at_fit",
    "gate_blind_truth_good_graded",
    "truth_bounce_pass",
    "truth_bounce_v2_pass",
    "truth_bounce_v2_graded_pass",
    "truth_contact_pass",
    "truth_contact_graded_pass",
    "truth_contact_at_fit_pass",
    "truth_position_pass",
)


def witness_breakdown(
    scoped: list[dict[str, Any]], taken: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Accepted / witnessed / wrong and the in-scope truth-good count, per witness."""
    result: dict[str, dict[str, Any]] = {}
    for key in WITNESS_KEYS:
        witnessed = [row for row in taken if row.get(key) is not None]
        wrong = [row for row in witnessed if row[key] is False]
        result[key] = {
            "accepted_witnessed": len(witnessed),
            "wrong_accepts": len(wrong),
            "wrong_accept_rate": (len(wrong) / len(witnessed)) if witnessed else None,
            "truth_good_in_scope": sum(1 for row in scoped if row.get(key) is True),
            "witnessed_in_scope": sum(1 for row in scoped if row.get(key) is not None),
        }
    return result


def score_gate(
    rows: list[dict[str, Any]],
    accepted: dict[str, bool],
    points: set[str],
    label: str,
) -> dict[str, Any]:
    scoped = [row for row in rows if row["point"] in points]
    taken = [row for row in scoped if accepted.get(row["flight_id"], False)]
    witnessed = [row for row in taken if row.get("truth_good") is not None]
    wrong = [row for row in witnessed if row["truth_good"] is False]
    blind_witnessed = [row for row in taken if row.get("gate_blind_truth_good") is not None]
    blind_wrong = [row for row in blind_witnessed if row["gate_blind_truth_good"] is False]
    truth_good_total = sum(1 for row in scoped if row.get("truth_good") is True)
    return {
        "gate": label,
        "points": len(points),
        "flights": len(scoped),
        "solved_flights": sum(bool(row["solved"]) for row in scoped),
        "accepted_flights": len(taken),
        "accepted_witnessed": len(witnessed),
        "wrong_accepts": len(wrong),
        "wrong_accept_rate": (len(wrong) / len(witnessed)) if witnessed else None,
        "accepted_blind_witnessed": len(blind_witnessed),
        "blind_wrong_accepts": len(blind_wrong),
        "blind_wrong_accept_rate": (
            (len(blind_wrong) / len(blind_witnessed)) if blind_witnessed else None
        ),
        "truth_good_flights_in_scope": truth_good_total,
        "recall_of_truth_good": (
            sum(1 for row in witnessed if row["truth_good"] is True) / truth_good_total
            if truth_good_total
            else None
        ),
        "complete_points": len(complete_points(rows, accepted, points)),
        "wrong_accept_flight_ids": sorted(row["flight_id"] for row in wrong),
        "by_witness": witness_breakdown(scoped, taken),
    }


def acceptance_map(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, bool]:
    return {row["flight_id"]: gate_accepts(row, gate) for row in rows}


def gate_label(gate: dict[str, Any]) -> str:
    def bound(value: float, unit: str) -> str:
        return "none" if value == INF else f"{value:g}{unit}"

    return " AND ".join(
        [
            f"held_out_median<={gate['held_out_median_px']:g}px",
            f"held_out_p90<={bound(gate['held_out_p90_px'], 'px')}",
            f"anchor_max_error<={bound(gate['anchor_max_error_m'], 'm')}",
            f"contact_reprojection<={bound(gate['contact_reprojection_px'], 'px')}",
            f"contact_reach<={bound(gate['contact_reach_m'], 'm')}",
            "contact_height in [0.30,3.50]m",
            "net_crossing_plausible",
        ]
    )


TARGET_WRONG_ACCEPT_RATE = 0.10
MINIMUM_ACCEPTED_WITNESSED = 5


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the candidates no other candidate beats on both yield and wrong-accept rate."""
    scored = [
        row
        for row in candidates
        if row["development"]["accepted_witnessed"] >= MINIMUM_ACCEPTED_WITNESSED
    ]
    frontier = []
    for row in sorted(
        scored,
        key=lambda item: (
            -item["development"]["accepted_flights"],
            item["development"]["wrong_accept_rate"],
        ),
    ):
        score = row["development"]
        if any(
            existing["wrong_accept_rate"] <= score["wrong_accept_rate"] for existing in frontier
        ):
            continue
        frontier.append(
            {
                "gate": gate_label(row["gate"]),
                "accepted_flights": score["accepted_flights"],
                "accepted_witnessed": score["accepted_witnessed"],
                "wrong_accepts": score["wrong_accepts"],
                "wrong_accept_rate": score["wrong_accept_rate"],
                "blind_wrong_accept_rate": score["blind_wrong_accept_rate"],
                "complete_points": score["complete_points"],
            }
        )
    return frontier


def select_gate(
    rows: list[dict[str, Any]], development: set[str], holdout: set[str]
) -> dict[str, Any]:
    """Choose one label-blind gate on development points, then score it once on the holdout.

    Selection rule, fixed before looking at the holdout: the highest accepted-flight count whose
    development wrong-accept rate is at most ``TARGET_WRONG_ACCEPT_RATE`` over at least
    ``MINIMUM_ACCEPTED_WITNESSED`` witnessed accepted flights.  If no candidate reaches the
    target, the lowest development wrong-accept rate wins and ``target_met`` is false.
    """
    candidates = []
    for gate in gate_grid():
        accepted = acceptance_map(rows, gate)
        candidates.append(
            {"gate": gate, "development": score_gate(rows, accepted, development, gate_label(gate))}
        )
    eligible = [
        row
        for row in candidates
        if row["development"]["accepted_witnessed"] >= MINIMUM_ACCEPTED_WITNESSED
    ]
    qualified = [
        row
        for row in eligible
        if row["development"]["wrong_accept_rate"] <= TARGET_WRONG_ACCEPT_RATE
    ]
    target_met = bool(qualified)
    pool = qualified or eligible or candidates
    if target_met:
        best = max(pool, key=lambda row: row["development"]["accepted_flights"])
    else:
        best = min(
            pool,
            key=lambda row: (
                row["development"]["wrong_accept_rate"],
                -row["development"]["accepted_flights"],
            ),
        )
    accepted = acceptance_map(rows, best["gate"])
    return {
        "selection_rule": (
            f"max accepted flights with development wrong-accept rate <= "
            f"{TARGET_WRONG_ACCEPT_RATE} over >= {MINIMUM_ACCEPTED_WITNESSED} witnessed accepts; "
            "otherwise minimum development wrong-accept rate"
        ),
        "target_wrong_accept_rate": TARGET_WRONG_ACCEPT_RATE,
        "target_met": target_met,
        "selected": best["gate"],
        "selected_label": gate_label(best["gate"]),
        "candidates_searched": len(candidates),
        "development": best["development"],
        "held_out": score_gate(rows, accepted, holdout, gate_label(best["gate"])),
        "all_truth": score_gate(rows, accepted, development | holdout, gate_label(best["gate"])),
        "development_frontier": pareto_frontier(candidates),
        "accepted_flight_ids": sorted(flight_id for flight_id, value in accepted.items() if value),
    }


def reference_gates(
    rows: list[dict[str, Any]], development: set[str], holdout: set[str]
) -> list[dict[str, Any]]:
    """Score the two gates already in play, on the same scopes, for comparison."""
    pixel = {row["flight_id"]: bool(row["pixel_accepted"]) for row in rows}
    metric = {row["flight_id"]: bool(row["metric_accepted"]) for row in rows}
    blind_metric = {
        row["flight_id"]: bool(
            row.get("solved")
            and row.get("net_crossing_plausible") is True
            and row.get("contact_height_plausible") is True
            and row.get("contact_reach_plausible") is True
        )
        for row in rows
    }
    intersection = {flight_id: pixel[flight_id] and blind_metric[flight_id] for flight_id in pixel}
    result = []
    for label, accepted in (
        ("pixel (shipped default)", pixel),
        ("metric (owner-facing, reads owner bounce truth)", metric),
        ("metric minus its bounce check (label-blind subset)", blind_metric),
        ("pixel AND label-blind metric subset", intersection),
    ):
        for scope, points in (
            ("development_32", development),
            ("held_out_8", holdout),
            ("all_truth_40", development | holdout),
        ):
            score = score_gate(rows, accepted, points, label)
            score["scope"] = scope
            result.append(score)
    return result


def bounce_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metres of court error per pixel of image error at the fitted bounce, by court half."""
    result: dict[str, Any] = {}
    for name, low, high in (("near_half", 0.0, 11.885), ("far_half", 11.885, 23.77)):
        ratios, court, image = [], [], []
        for row in rows:
            metres = row.get("truth_bounce_error_max_m")
            pixels = row.get("truth_bounce_image_error_max_px")
            y = row.get("truth_bounce_court_y_m")
            if metres is None or pixels is None or y is None or not low <= float(y) < high:
                continue
            court.append(float(metres))
            image.append(float(pixels))
            if float(pixels) > 0.5:
                ratios.append(float(metres) / float(pixels))
        result[name] = {
            "flights": len(court),
            "court_error_median_p90_m": [percentile(court, 50.0), percentile(court, 90.0)],
            "image_error_median_p90_px": [percentile(image, 50.0), percentile(image, 90.0)],
            "metres_per_pixel_median": percentile(ratios, 50.0),
            "pixels_for_10cm_median": (
                0.10 / percentile(ratios, 50.0) if percentile(ratios, 50.0) else None
            ),
        }
    return result


def bounce_witness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the corrected bounce witness forms, how wide it is, and how it disagrees."""
    solved = [row for row in rows if row.get("solved")]
    formed = [row for row in solved if row.get("truth_bounce_v2_pass") is not None]
    both = [
        row
        for row in solved
        if row.get("truth_bounce_error_max_m") is not None
        and row.get("truth_bounce_v2_error_max_m") is not None
    ]

    def spread(values: list[float]) -> list[float | None]:
        return [percentile(values, q) for q in (50.0, 90.0)]

    return {
        "solved_flights": len(solved),
        "flights_with_rayplane_witness": sum(
            row.get("truth_bounce_pass") is not None for row in solved
        ),
        "flights_with_v2_witness": len(formed),
        "flights_with_both": len(both),
        "v2_sigma_median_p90_m": spread(
            [
                float(row["truth_bounce_v2_sigma_max_m"])
                for row in formed
                if row.get("truth_bounce_v2_sigma_max_m") is not None
            ]
        ),
        "v2_resolves_10cm_flights": sum(
            row.get("truth_bounce_v2_resolved_pass") is not None for row in solved
        ),
        "v2_offset_abs_median_p90_frames": spread(
            [
                abs(float(row["truth_bounce_v2_offset_max_frames"]))
                for row in formed
                if row.get("truth_bounce_v2_offset_max_frames") is not None
            ]
        ),
        "rayplane_error_median_p90_m": spread(
            [float(row["truth_bounce_error_max_m"]) for row in both]
        ),
        "v2_error_median_p90_m": spread(
            [float(row["truth_bounce_v2_error_max_m"]) for row in both]
        ),
        "rayplane_pass_on_both": sum(row.get("truth_bounce_pass") is True for row in both),
        "v2_pass_on_both": sum(row.get("truth_bounce_v2_pass") is True for row in both),
        "contact_click_time_median_p90_px": spread(
            [
                float(row["truth_contact_max_px"])
                for row in solved
                if row.get("truth_contact_max_px") is not None
            ]
        ),
        "v2_transport_px_median_p90": spread(
            [
                float(row["truth_bounce_v2_transport_max_px"])
                for row in formed
                if row.get("truth_bounce_v2_transport_max_px") is not None
            ]
        ),
        "track_corner_error_median_p90_m": spread(
            [
                float(row["truth_bounce_track_error_max_m"])
                for row in solved
                if row.get("truth_bounce_track_error_max_m") is not None
            ]
        ),
        "contact_at_fit_median_p90_px": spread(
            [
                float(row["truth_contact_at_fit_max_px"])
                for row in solved
                if row.get("truth_contact_at_fit_max_px") is not None
            ]
        ),
    }


SCORE_BINS = ((1.0, 1.01), (0.9, 1.0), (0.75, 0.9), (0.5, 0.75), (0.25, 0.5), (0.0, 0.25))


def score_histogram(values: list[float]) -> dict[str, int]:
    """How the graded scores fall.  ``1.0`` is the flights inside the circle, exactly."""
    bins: dict[str, int] = {}
    for low, high in SCORE_BINS:
        label = "1.0" if high > 1.0 else f"[{low:g},{high:g})"
        bins[label] = sum(1 for value in values if low <= value < high)
    return bins


def graded_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The graded scores: their distribution, their circles, and what they change.

    Every count here is over solved flights that have the witness in question, so the graded and
    the hard column are counted over exactly the same flights.
    """
    solved = [row for row in rows if row.get("solved")]

    def spread(values: list[float]) -> list[float | None]:
        return [percentile(values, q) for q in (50.0, 90.0)]

    def block(
        score_key: str,
        linear_key: str | None,
        hard_key: str,
        graded_key: str,
        radius_key: str,
        error_key: str,
    ) -> dict[str, Any]:
        scored = [row for row in solved if row.get(score_key) is not None]
        scores = [float(row[score_key]) for row in scored]
        radii = [float(row[radius_key]) for row in scored if row.get(radius_key) is not None]
        errors = [float(row[error_key]) for row in scored if row.get(error_key) is not None]
        hard_pass = [row for row in scored if row.get(hard_key) is True]
        graded_pass = [row for row in scored if row.get(graded_key) is True]
        gained = [
            row for row in scored if row.get(graded_key) is True and row.get(hard_key) is False
        ]
        lost = [row for row in scored if row.get(graded_key) is False and row.get(hard_key) is True]
        result = {
            "flights_scored": len(scored),
            # the low tail is the interesting end of a score, so p10 rather than p90
            "score_median_p10": [percentile(scores, 50.0), percentile(scores, 10.0)],
            "score_mean": (sum(scores) / len(scores)) if scores else None,
            "score_histogram": score_histogram(scores),
            "inside_circle_flights": sum(1 for value in scores if value >= 1.0),
            "radius_median_p90": spread(radii),
            "error_median_p90": spread(errors),
            "hard_pass": len(hard_pass),
            "graded_pass": len(graded_pass),
            "graded_only": len(gained),
            "hard_only": len(lost),
            "graded_only_error_median_p90": spread(
                [float(row[error_key]) for row in gained if row.get(error_key) is not None]
            ),
        }
        if linear_key is not None:
            linear = [
                row
                for row in scored
                if row.get(linear_key) is not None
                and (float(row[linear_key]) >= GRADED_PASS_SCORE)
                != (float(row[score_key]) >= GRADED_PASS_SCORE)
            ]
            result["linear_shape_disagreements"] = len(linear)
        return result

    contact_radii = [
        float(row["truth_contact_radius_max_px"])
        for row in solved
        if row.get("truth_contact_radius_max_px") is not None
    ]
    return {
        "bounce": block(
            "truth_bounce_v2_score",
            "truth_bounce_v2_score_linear",
            "truth_bounce_v2_pass",
            "truth_bounce_v2_graded_pass",
            "truth_bounce_v2_radius_max_m",
            "truth_bounce_v2_error_max_m",
        ),
        "contact": block(
            "truth_contact_score",
            "truth_contact_score_linear",
            "truth_contact_pass",
            "truth_contact_graded_pass",
            "truth_contact_radius_max_px",
            "truth_contact_max_px",
        ),
        "contact_radius_px_median_p90": spread(contact_radii),
        "contact_refiner_abstained_flights": sum(
            row.get("truth_contact_refiner_abstained") is True for row in solved
        ),
        "contact_refiner_radius_flights": sum(
            row.get("truth_contact_refiner_abstained") is False for row in solved
        ),
        "truth_good_hard": _truth_summary(solved, "truth_good_v2"),
        "truth_good_graded": _truth_summary(solved, "truth_good_graded"),
    }


def truth_ceiling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How many solved flights any gate could possibly accept correctly."""
    solved = [row for row in rows if row.get("solved")]
    result: dict[str, Any] = {"solved_flights": len(solved)}
    for name, key in (
        ("bounce", "truth_bounce_pass"),
        ("bounce_v2", "truth_bounce_v2_pass"),
        ("bounce_v2_graded", "truth_bounce_v2_graded_pass"),
        ("bounce_v2_resolved", "truth_bounce_v2_resolved_pass"),
        ("owner_position", "truth_position_pass"),
        ("contact", "truth_contact_pass"),
        ("contact_graded", "truth_contact_graded_pass"),
        ("contact_at_fit", "truth_contact_at_fit_pass"),
        ("all_witnesses", "truth_good"),
        ("all_witnesses_v2", "truth_good_v2"),
        ("all_witnesses_v2_noj", "truth_good_v2_noj"),
        ("all_witnesses_graded", "truth_good_graded"),
        ("all_witnesses_graded_noj", "truth_good_graded_noj"),
        ("junction", "truth_junction_pass"),
        ("junction_graded", "truth_junction_graded_pass"),
        ("all_witnesses_graded_linear", "truth_good_graded_linear"),
        ("all_witnesses_graded_at_fit", "truth_good_graded_at_fit"),
        ("all_witnesses_v2_at_fit", "truth_good_v2_at_fit"),
        ("all_witnesses_v2_resolved", "truth_good_v2_resolved"),
        ("gate_blind_witnesses", "gate_blind_truth_good"),
        ("gate_blind_witnesses_at_fit", "gate_blind_truth_good_at_fit"),
        ("gate_blind_witnesses_graded", "gate_blind_truth_good_graded"),
    ):
        result[name] = _truth_summary(solved, key)
    for name, key in (
        ("truth_good_by_scope", "truth_good"),
        ("truth_good_v2_by_scope", "truth_good_v2"),
        ("truth_good_v2_noj_by_scope", "truth_good_v2_noj"),
        ("truth_good_graded_noj_by_scope", "truth_good_graded_noj"),
        ("junction_by_scope", "truth_junction_pass"),
        ("junction_graded_by_scope", "truth_junction_graded_pass"),
        ("truth_good_v2_at_fit_by_scope", "truth_good_v2_at_fit"),
        ("truth_good_v2_resolved_by_scope", "truth_good_v2_resolved"),
        ("truth_good_graded_by_scope", "truth_good_graded"),
        ("truth_good_graded_linear_by_scope", "truth_good_graded_linear"),
        ("bounce_v2_graded_by_scope", "truth_bounce_v2_graded_pass"),
        ("contact_graded_by_scope", "truth_contact_graded_pass"),
    ):
        result[name] = {
            scope: _truth_summary([row for row in solved if row["scope"] == scope], key)
            for scope in ("other_38", "held_out_8")
        }
    return result


# --- audit driver ------------------------------------------------------------------------


def audit(
    *,
    reconstruction_root: Path,
    camera_root: Path,
    truth_events: Path,
    automatic_events: Path,
    output_root: Path,
    click_offset_frames: float = CLICK_OFFSET_FRAMES,
) -> dict[str, Any]:
    report_path = reconstruction_root / "report.json"
    report = json.loads(report_path.read_text())
    ledger = build_flight_ledger([report_path])
    rows = build_flight_rows(report, ledger, camera_root, truth_events, click_offset_frames)
    causes = build_point_causes(report, ledger, rows, truth_events, automatic_events)

    truth_points = set(owner_event_index(truth_events))
    development = {point for point in truth_points if scope_of(point) == "other_38"}
    holdout = {point for point in truth_points if scope_of(point) == "held_out_8"}

    truth_rows = [row for row in rows if row["point"] in truth_points]
    selection = select_gate(truth_rows, development, holdout)
    payload = {
        "schema": "flight_gate_audit_v1",
        "artifact_class": "offline_truth_scoring",
        "automatic_gate_changed": False,
        "sources": {
            "report": os.fspath(report_path),
            "camera_root": os.fspath(camera_root),
            "truth_events": os.fspath(truth_events),
            "automatic_events": os.fspath(automatic_events),
        },
        "thresholds": {
            "bounce_truth_limit_m": BOUNCE_TRUTH_LIMIT_M,
            "bounce_graded_floor_m": BOUNCE_GRADED_FLOOR_M,
            "bounce_graded_sigma_multiple": BOUNCE_GRADED_SIGMA_MULTIPLE,
            "contact_graded_floor_px": CONTACT_GRADED_FLOOR_PX,
            "graded_pass_score": GRADED_PASS_SCORE,
            "graded_shape": "1 inside r, exp(-((d-r)/r)^2) outside; pass at score >= 0.5",
            "bounce_v2_sigma_limit_m": BOUNCE_V2_SIGMA_LIMIT_M,
            "click_sigma_px": CLICK_SIGMA_PX,
            "bounce_witness_window_frames": BOUNCE_WITNESS_WINDOW,
            "click_offset_frames": float(click_offset_frames),
            "owner_position_median_limit_px": OWNER_POSITION_MEDIAN_LIMIT_PX,
            "owner_position_p90_limit_px": OWNER_POSITION_P90_LIMIT_PX,
            "owner_contact_limit_px": OWNER_CONTACT_LIMIT_PX,
            "minimum_owner_position_frames": MIN_OWNER_POSITION_FRAMES,
            "junction_gap_limit_m": JUNCTION_GAP_LIMIT_M,
            "junction_witness_floor_m": JUNCTION_WITNESS_FLOOR_M,
            "junction_witness_sigma_multiple": JUNCTION_WITNESS_SIGMA_MULTIPLE,
            "event_match_frames": EVENT_MATCH_FRAMES,
        },
        "witness_coverage": {
            "flights": len(rows),
            "solved_flights": sum(bool(row["solved"]) for row in rows),
            "flights_with_any_witness": sum(row.get("truth_good") is not None for row in rows),
            "flights_with_bounce_witness": sum(
                row.get("truth_bounce_pass") is not None for row in rows
            ),
            "flights_with_owner_position_witness": sum(
                row.get("truth_position_pass") is not None for row in rows
            ),
            "flights_with_contact_witness": sum(
                row.get("truth_contact_pass") is not None for row in rows
            ),
            "flights_with_gate_blind_witness": sum(
                row.get("gate_blind_truth_good") is not None for row in rows
            ),
            "flights_with_junction_witness": sum(
                row.get("truth_junction_pass") is not None for row in rows
            ),
        },
        "truth_ceiling": truth_ceiling(rows),
        "graded_summary": graded_summary(rows),
        "graded_summary_truth_40": graded_summary(truth_rows),
        "bounce_witness_summary": bounce_witness_summary(rows),
        "bounce_sensitivity": bounce_sensitivity(rows),
        "crosstab_all_46": crosstab(rows),
        "crosstab_truth_40": crosstab(truth_rows),
        "point_causes": {
            "points": len(causes),
            "counts": dict(Counter(row["cause"] for row in causes).most_common()),
            "counts_by_scope": {
                scope: dict(
                    Counter(row["cause"] for row in causes if row["scope"] == scope).most_common()
                )
                for scope in ("other_38", "held_out_8")
            },
            "detail_counts": {
                cause: dict(
                    Counter(
                        row.get("detail", "") for row in causes if row["cause"] == cause
                    ).most_common(8)
                )
                for cause in CAUSE_ORDER
            },
        },
        "reference_gates": reference_gates(truth_rows, development, holdout),
        "gate_selection": selection,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "flight_gate_audit.json", payload)
    write_csv(output_root / "flight_truth.csv", rows)
    write_csv(output_root / "point_causes.csv", causes)
    return payload


# --- native overlay contact sheets ---------------------------------------------------------


SHEET_COLUMNS = 3
SHEET_ROWS = 2
ZOOM_HALF = 120
ZOOM_SCALE = 2


def _flight_sheet_frames(start: float, end: float, count: int) -> list[int]:
    values = np.linspace(float(start), float(end), count)
    return [int(round(value)) for value in values]


def render_flight_sheet(
    *,
    camera_root: Path,
    match_id: str,
    clip: str,
    fit: dict[str, Any],
    attempt: dict[str, Any],
    owner_events: dict[str, list[dict[str, Any]]],
    owner_positions: dict[int, tuple[float, float]],
    track: dict[int, tuple[float, float]],
    caption: str,
    output_path: Path,
) -> Path:
    import cv2

    frames_dir = camera_root / match_id / "audit_frames_native_1080" / clip
    camera = reconstruction.PointCamera(camera_root / match_id, clip)
    selected = _flight_sheet_frames(
        attempt["start_frame"], attempt["end_frame"], SHEET_COLUMNS * SHEET_ROWS
    )
    tiles = []
    for frame in selected:
        path = frames_dir / f"f_{frame:04d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"missing native frame {path}")
        if image.shape[:2] != (1080, 1920):
            raise ValueError(f"{path} is not native 1920x1080")
        # Full fitted flight, reprojected through the same camera the fitter used.
        drawn = None
        for sample in fit.get("trajectory", []):
            uv = project(camera.p_at(float(sample["frame"])), sample["xyz"])
            if uv is None:
                continue
            point = (int(round(uv[0])), int(round(uv[1])))
            cv2.circle(image, point, 3, (0, 255, 0), -1)
            if int(round(float(sample["frame"]))) == frame:
                drawn = point
                cv2.circle(image, point, 14, (255, 255, 0), 2)
        for bounce in fit.get("bounces", []):
            uv = project(camera.p_at(float(bounce["frame"])), bounce["x"])
            if uv is not None:
                cv2.drawMarker(
                    image,
                    (int(round(uv[0])), int(round(uv[1]))),
                    (0, 200, 255),
                    cv2.MARKER_TILTED_CROSS,
                    28,
                    3,
                )
        for kind, color in (("bounce", (0, 140, 255)), ("contact", (0, 0, 255))):
            for click in owner_events.get(kind, []):
                if click["image_xy"] is None:
                    continue
                if abs(float(click["frame"]) - frame) > 1.0:
                    continue
                cv2.drawMarker(
                    image,
                    (int(round(click["image_xy"][0])), int(round(click["image_xy"][1]))),
                    color,
                    cv2.MARKER_CROSS,
                    36,
                    3,
                )
        if frame in owner_positions:
            pixel = owner_positions[frame]
            cv2.circle(image, (int(round(pixel[0])), int(round(pixel[1]))), 12, (255, 255, 255), 2)
        if frame in track:
            pixel = track[frame]
            cv2.circle(image, (int(round(pixel[0])), int(round(pixel[1]))), 8, (255, 0, 255), 2)
        # Native-resolution inset around the fitted ball so the eye can judge the fit.
        if drawn is not None:
            x0 = int(np.clip(drawn[0] - ZOOM_HALF, 0, 1920 - 2 * ZOOM_HALF))
            y0 = int(np.clip(drawn[1] - ZOOM_HALF, 0, 1080 - 2 * ZOOM_HALF))
            crop = image[y0 : y0 + 2 * ZOOM_HALF, x0 : x0 + 2 * ZOOM_HALF].copy()
            crop = cv2.resize(
                crop,
                (2 * ZOOM_HALF * ZOOM_SCALE, 2 * ZOOM_HALF * ZOOM_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            height, width = crop.shape[:2]
            image[1080 - height :, 1920 - width :] = crop
            cv2.rectangle(image, (1920 - width, 1080 - height), (1919, 1079), (255, 255, 255), 2)
        cv2.rectangle(image, (0, 0), (1919, 60), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"{match_id} {clip} f{frame}  {caption}",
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(image)
    sheet = np.vstack(
        [
            np.hstack(tiles[index * SHEET_COLUMNS : (index + 1) * SHEET_COLUMNS])
            for index in range(SHEET_ROWS)
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write {output_path}")
    review = cv2.resize(sheet, (sheet.shape[1] // 3, sheet.shape[0] // 3), cv2.INTER_AREA)
    cv2.imwrite(str(output_path.with_name(output_path.stem + "_review.png")), review)
    return output_path


def _load_track(camera_root: Path, match_id: str, clip: str) -> dict[int, tuple[float, float]]:
    """The composed arc-augmented track in native pixels, for the sheet overlay.

    The CSV's ``frame`` column is a file name (``f_0061.jpg``) and its ``x``/``y`` columns are
    the legacy 960x540 pair, so the canonical loader in ``reconstruction`` is used rather than
    a second parse of the same file.
    """
    path = camera_root / match_id / reconstruction.TRACK_NAME
    if not path.exists():
        return {}
    return {
        frame: (float(value[0]), float(value[1]))
        for frame, value in reconstruction.load_track(camera_root / match_id, clip).items()
    }


def select_sheet_flights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Twelve flights: 4 metric-only, 4 pixel-accepted, 4 rejected by both.

    The first two groups take the two best and the two worst truth-agreement flights, so the
    eye is shown both what each gate gets right and what it gets wrong.  The third group takes
    the four with the best truth agreement, which is what the brief asks for.
    """

    def truth_rank(row: dict[str, Any]) -> tuple:
        bounce = row.get("truth_bounce_error_max_m")
        contact = row.get("truth_contact_max_px")
        return (
            0 if row.get("truth_good") else 1,
            float(bounce) if bounce is not None else 9.99,
            float(contact) if contact is not None else 999.0,
        )

    solved = [row for row in rows if row.get("solved")]
    witnessed = [row for row in solved if row.get("truth_good") is not None]
    metric_only = sorted(
        (r for r in witnessed if r["metric_accepted"] and not r["pixel_accepted"]), key=truth_rank
    )
    pixel_side = sorted((r for r in witnessed if r["pixel_accepted"]), key=truth_rank)
    neither = sorted(
        (r for r in witnessed if not r["pixel_accepted"] and not r["metric_accepted"]),
        key=truth_rank,
    )
    selected = []
    for group, label, picks in (
        (metric_only, "metric_only", metric_only[:2] + metric_only[-2:]),
        (pixel_side, "pixel_or_both", pixel_side[:2] + pixel_side[-2:]),
        (neither, "rejected_by_both", neither[:4]),
    ):
        del group
        for row in picks:
            selected.append({**row, "sheet_group": label})
    return selected


def sheets(
    *,
    audit_root: Path,
    reconstruction_root: Path,
    camera_root: Path,
    truth_events: Path,
    output_root: Path,
) -> dict[str, Any]:
    report = json.loads((reconstruction_root / "report.json").read_text())
    with (audit_root / "flight_truth.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))

    def cast(row: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                result[key] = None
            elif value in ("True", "False"):
                result[key] = value == "True"
            else:
                try:
                    result[key] = (
                        float(value)
                        if key
                        not in (
                            "point",
                            "match_id",
                            "scope",
                            "flight_id",
                            "metric_reasons",
                            "start_phase",
                        )
                        else value
                    )
                except ValueError:
                    result[key] = value
        result["flight_index"] = int(float(row["flight_index"]))
        return result

    rows = [cast(row) for row in raw]
    selected = select_sheet_flights(rows)
    details = {str(point["point"]): point for point in report["points_detail"]}
    owner_events = owner_event_index(truth_events)
    positions = owner_position_index()
    manifest = []
    for row in selected:
        point_id = str(row["point"])
        point = details[point_id]
        match_id, clip = str(point["match_id"]), str(point["clip"])
        index = int(row["flight_index"])
        fit = next(f for f in point["fits"] if int(f["flight_index"]) == index)
        attempt = next(a for a in point["flight_attempts"] if int(a["flight_index"]) == index)
        caption = (
            f"{row['sheet_group']} pixel={bool(row['pixel_accepted'])} "
            f"metric={bool(row['metric_accepted'])} truth_good={row.get('truth_good')}"
        )
        path = output_root / f"{point_id}__flight_{index:03d}.png"
        render_flight_sheet(
            camera_root=camera_root,
            match_id=match_id,
            clip=clip,
            fit=fit,
            attempt=attempt,
            owner_events=owner_events.get(point_id, {name: [] for name in PHYSICAL_EVENTS}),
            owner_positions=positions.get((match_id, clip), {}),
            track=_load_track(camera_root, match_id, clip),
            caption=caption,
            output_path=path,
        )
        manifest.append(
            {
                "flight_id": row["flight_id"],
                "group": row["sheet_group"],
                "pixel_accepted": bool(row["pixel_accepted"]),
                "metric_accepted": bool(row["metric_accepted"]),
                "truth_good": row.get("truth_good"),
                "truth_bounce_error_max_m": row.get("truth_bounce_error_max_m"),
                "truth_contact_max_px": row.get("truth_contact_max_px"),
                "truth_reprojection_median_px": row.get("truth_reprojection_median_px"),
                "held_out_median_px": row.get("held_out_median_px"),
                "sheet": os.fspath(path),
            }
        )
    payload = {
        "schema": "flight_gate_audit_sheets_v1",
        "selection": "4 metric-only, 4 pixel-accepted, 4 rejected-by-both with best truth agreement",
        "flights": manifest,
    }
    write_json(output_root / "sheets.json", payload)
    return payload


# --- the witness's own error, measured where the truth is known ------------------------------


def bench_fitted_bounces(report_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every fitted bounce of a bench run, by point: ``{"frame": .., "xy": (x, y)}``."""
    report = json.loads(report_path.read_text())
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in report.get("points_detail", []):
        for fit in point.get("fits", []) or []:
            for bounce in fit.get("bounces", []) or []:
                position = bounce.get("x")
                if position is None or bounce.get("frame") is None:
                    continue
                result[str(point["point"])].append(
                    {
                        "frame": float(bounce["frame"]),
                        "xy": np.asarray(position[:2], dtype=float),
                        "flight_index": fit.get("flight_index"),
                    }
                )
    return result


def bench_witness(
    *,
    bench_root: Path,
    rung: str,
    output_root: Path,
    click_offset_frames: float = CLICK_OFFSET_FRAMES,
    fit_report: Path | None = None,
) -> dict[str, Any]:
    """Score both bounce witnesses against the synthetic bench's own bounce.

    The bench emits every bounce on an integer frame with an observation of the ball on that
    frame, which is exactly the shape of the owner's click, and it knows where and when the
    bounce really was.  So the same two constructions the real audit uses can be measured
    here: the click-ray/court-plane intersection (``bounce_rayplane_v1``) and the corrected
    corner (``bounce_kinematic_v2``, with and without the click fused in).
    """
    from cv.pipeline.rich_ball_physics import R_BALL as R_BALL_HEIGHT

    truth = json.loads((bench_root / "truth.json").read_text())
    root = bench_root / "roots" / rung
    emissions = json.loads((root / "event_emissions.json").read_text())
    if isinstance(emissions, dict):
        emissions = emissions["emissions"]
    truth_bounces: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for point in truth["points"]:
        truth_bounces[str(point["point"])] = list(point.get("bounces") or [])
        meta[str(point["point"])] = {
            "match_id": str(point["match_id"]),
            "clip": str(point["clip"]),
            "fps": float(point.get("fps") or 25.0),
        }
    emitted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in emissions:
        if str(row.get("event_type")) != "bounce":
            continue
        location = row.get("location") or {}
        if location.get("image_x") is None:
            continue
        emitted[str(row["clip"])].append(
            {
                "frame": float(row["frame"]),
                "image_xy": (float(location["image_x"]), float(location["image_y"])),
            }
        )

    fitted = bench_fitted_bounces(fit_report) if fit_report is not None else {}

    rows: list[dict[str, Any]] = []
    cameras: dict[tuple[str, str], Any] = {}
    tracks: dict[tuple[str, str], dict[int, Any]] = {}
    for point_id, events in sorted(emitted.items()):
        if point_id not in meta:
            continue
        match_id, clip = meta[point_id]["match_id"], meta[point_id]["clip"]
        key = (match_id, clip)
        if key not in cameras:
            try:
                cameras[key] = reconstruction.PointCamera(root / match_id, clip)
            except (ValueError, FileNotFoundError):
                cameras[key] = None
            tracks[key] = (
                reconstruction.load_track(root / match_id, clip) if cameras[key] is not None else {}
            )
        camera = cameras[key]
        if camera is None:
            continue
        available = list(truth_bounces.get(point_id, []))
        for event in sorted(events, key=lambda row_: row_["frame"]):
            if not available:
                continue
            true_bounce = min(
                available, key=lambda row_: abs(float(row_["frame"]) - event["frame"])
            )
            if abs(float(true_bounce["frame"]) - event["frame"]) > EVENT_MATCH_FRAMES:
                continue
            available.remove(true_bounce)
            true_xy = np.asarray(true_bounce["xyz"][:2], dtype=float)
            rayplane = np.asarray(
                reconstruction.court_point(
                    camera, np.asarray(event["image_xy"], dtype=float), event["frame"]
                ),
                dtype=float,
            )
            witness = bounce_witness_v2(
                camera=camera,
                click_frame=event["frame"],
                click_pixel=event["image_xy"],
                track=tracks[key],
                owner_positions={},
                fps=meta[point_id]["fps"],
                rayplane_xy=rayplane,
                click_offset_frames=click_offset_frames,
            )
            record: dict[str, Any] = {
                "point": point_id,
                "match_id": match_id,
                "emitted_frame": event["frame"],
                "true_frame": float(true_bounce["frame"]),
                "true_x_m": float(true_xy[0]),
                "true_y_m": float(true_xy[1]),
                "rayplane_error_m": float(np.linalg.norm(rayplane - true_xy)),
                "v2_formed": bool(witness.get("formed")),
                "v2_reason": witness.get("reason"),
            }
            for field, name in (
                ("court_xy_track_corner", "track_corner"),
                ("court_xy_kinematic_court", "kinematic_court"),
            ):
                if witness.get(field) is not None:
                    record[f"{name}_error_m"] = float(
                        np.linalg.norm(np.asarray(witness[field], dtype=float) - true_xy)
                    )
            if witness.get("kinematic_court_t_subframe") is not None:
                record["kinematic_court_time_error_frames"] = float(
                    witness["kinematic_court_t_subframe"]
                ) - float(true_bounce["frame"])
            if witness.get("formed"):
                record.update(
                    {
                        "v2_error_m": float(
                            np.linalg.norm(np.asarray(witness["court_xy"], dtype=float) - true_xy)
                        ),
                        "v2_sigma_m": float(witness["sigma_m"]),
                        "v2_sigma_track_only_m": float(witness["sigma_track_only_m"]),
                        "v2_sigma_click_lateral_m": float(witness["sigma_click_lateral_m"]),
                        "v2_sigma_click_depth_m": float(witness["sigma_click_depth_m"]),
                        "v2_transport_px": float(witness["transport_px"]),
                        "v2_time_error_frames": float(witness["t_subframe"])
                        - float(true_bounce["frame"]),
                        "emitted_time_error_frames": event["frame"] - float(true_bounce["frame"]),
                    }
                )
            # The same click ray at the ball's resting centre height instead of z = 0, on the
            # click's own frame: the fitter's own bounce anchor, with the plane error removed
            # and the time error left in.
            at_height = ray_at_height(
                np.asarray(camera.p_at(event["frame"]), dtype=float),
                event["image_xy"],
                R_BALL_HEIGHT,
            )
            if at_height is not None:
                record["rayplane_at_ball_height_error_m"] = float(
                    np.linalg.norm(at_height - true_xy)
                )
            # How lenient the graded circle is where the truth is known: score the run's own
            # fitted bounce against the witness, and measure the same fit against the true
            # bounce.  A fit the circle admits whose true error is large is a wrong accept the
            # leniency bought.
            if witness.get("formed"):
                radius_m = bounce_graded_radius_m(float(witness["sigma_m"]))
                witness_xy = np.asarray(witness["court_xy"], dtype=float)
                witness_time = float(witness["t_subframe"])
                near = [
                    bounce
                    for bounce in fitted.get(point_id, [])
                    if abs(bounce["frame"] - witness_time) <= BOUNCE_V2_MATCH_FRAMES
                ]
                record["graded_radius_m"] = radius_m
                record["fit_matched"] = bool(near)
                if near:
                    chosen = min(near, key=lambda bounce: abs(bounce["frame"] - witness_time))
                    to_witness = float(np.linalg.norm(chosen["xy"] - witness_xy))
                    to_true = float(np.linalg.norm(chosen["xy"] - true_xy))
                    record.update(
                        {
                            "fit_frame": chosen["frame"],
                            "fit_error_to_witness_m": to_witness,
                            "fit_error_to_true_m": to_true,
                            "fit_graded_score": graded_score(to_witness, radius_m),
                            "fit_graded_score_linear": graded_score_linear(to_witness, radius_m),
                            "fit_graded_pass": bool(
                                graded_score(to_witness, radius_m) >= GRADED_PASS_SCORE
                            ),
                            "fit_hard_pass": bool(to_witness <= BOUNCE_TRUTH_LIMIT_M),
                            "fit_true_pass": bool(to_true <= BOUNCE_TRUTH_LIMIT_M),
                        }
                    )
            rows.append(record)

    def spread(key: str, source: list[dict[str, Any]], absolute: bool = False) -> dict[str, Any]:
        values = [
            abs(float(row[key])) if absolute else float(row[key])
            for row in source
            if row.get(key) is not None
        ]
        return {
            "n": len(values),
            "median": percentile(values, 50.0),
            "p90": percentile(values, 90.0),
            "within_0.10_m": (
                sum(value <= BOUNCE_TRUTH_LIMIT_M for value in values) / len(values)
                if values and key.endswith("_m")
                else None
            ),
        }

    formed = [row for row in rows if row["v2_formed"]]

    def leniency(source: list[dict[str, Any]]) -> dict[str, Any]:
        """What the circle admits, and how wrong the admitted fits really are."""
        matched = [row for row in source if row.get("fit_error_to_true_m") is not None]

        def cell(kept: list[dict[str, Any]]) -> dict[str, Any]:
            true_errors = [float(row["fit_error_to_true_m"]) for row in kept]
            wrong = [row for row in kept if not row["fit_true_pass"]]
            return {
                "admitted": len(kept),
                "truly_right_at_0.10_m": sum(1 for row in kept if row["fit_true_pass"]),
                "truly_wrong_at_0.10_m": len(wrong),
                "true_error_median_p90_m": [
                    percentile(true_errors, 50.0),
                    percentile(true_errors, 90.0),
                ],
                "truly_wrong_error_median_p90_m": [
                    percentile([float(row["fit_error_to_true_m"]) for row in wrong], 50.0),
                    percentile([float(row["fit_error_to_true_m"]) for row in wrong], 90.0),
                ],
                "truly_wrong_beyond_0.20_m": sum(
                    1 for row in kept if float(row["fit_error_to_true_m"]) > 0.20
                ),
                "truly_wrong_beyond_0.50_m": sum(
                    1 for row in kept if float(row["fit_error_to_true_m"]) > 0.50
                ),
                "truly_wrong_beyond_1.00_m": sum(
                    1 for row in kept if float(row["fit_error_to_true_m"]) > 1.00
                ),
                "worst_true_error_m": max(true_errors, default=None),
            }

        graded = [row for row in matched if row["fit_graded_pass"]]
        hard = [row for row in matched if row["fit_hard_pass"]]
        gained = [row for row in matched if row["fit_graded_pass"] and not row["fit_hard_pass"]]
        return {
            "witnesses_with_a_fitted_bounce": len(matched),
            "witnesses_without_one": sum(1 for row in source if row.get("fit_matched") is False),
            "graded_radius_median_p90_m": [
                percentile([float(row["graded_radius_m"]) for row in source], q)
                for q in (50.0, 90.0)
            ],
            "graded_score_median_p10": [
                percentile([float(row["fit_graded_score"]) for row in matched], q)
                for q in (50.0, 10.0)
            ],
            "graded_score_histogram": score_histogram(
                [float(row["fit_graded_score"]) for row in matched]
            ),
            "graded": cell(graded),
            "hard_10cm": cell(hard),
            "graded_only": cell(gained),
            "true_10cm": {
                "flights": sum(1 for row in matched if row["fit_true_pass"]),
                "admitted_by_graded": sum(
                    1 for row in matched if row["fit_true_pass"] and row["fit_graded_pass"]
                ),
                "admitted_by_hard": sum(
                    1 for row in matched if row["fit_true_pass"] and row["fit_hard_pass"]
                ),
            },
        }

    payload = {
        "schema": "flight_gate_audit_bench_witness_v1",
        "artifact_class": "offline_truth_scoring",
        "rung": rung,
        "click_offset_frames": float(click_offset_frames),
        "sources": {"bench_root": os.fspath(bench_root), "root": os.fspath(root)},
        "emitted_bounces_matched": len(rows),
        "v2_formed": len(formed),
        "v2_abstentions": dict(
            Counter(row["v2_reason"] for row in rows if not row["v2_formed"]).most_common()
        ),
        "kinematic_court_formed": sum(
            row.get("kinematic_court_error_m") is not None for row in rows
        ),
        "all_matched": {
            "bounce_rayplane_v1_m": spread("rayplane_error_m", rows),
            "bounce_rayplane_at_ball_height_m": spread("rayplane_at_ball_height_error_m", rows),
            "bounce_kinematic_v2_m": spread("v2_error_m", rows),
        },
        "where_both_exist": {
            "bounce_rayplane_v1_m": spread("rayplane_error_m", formed),
            "bounce_kinematic_v2_m": spread("v2_error_m", formed),
            "bounce_rayplane_at_ball_height_m": spread("rayplane_at_ball_height_error_m", formed),
            "bounce_track_corner_m": spread("track_corner_error_m", formed),
            "bounce_kinematic_court_m": spread("kinematic_court_error_m", formed),
            "v2_claimed_sigma_m": spread("v2_sigma_m", formed),
            "v2_claimed_sigma_track_only_m": spread("v2_sigma_track_only_m", formed),
            "v2_claimed_click_sigma_lateral_m": spread("v2_sigma_click_lateral_m", formed),
            "v2_claimed_click_sigma_depth_m": spread("v2_sigma_click_depth_m", formed),
            "v2_transport_px": spread("v2_transport_px", formed),
            "v2_time_error_frames": spread("v2_time_error_frames", formed, absolute=True),
            "kinematic_court_time_error_frames": spread(
                "kinematic_court_time_error_frames", formed, absolute=True
            ),
            "emitted_time_error_frames": spread("emitted_time_error_frames", formed, absolute=True),
        },
    }
    if fit_report is not None:
        payload["fit_report"] = os.fspath(fit_report)
        payload["graded_leniency"] = leniency(formed)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "bench_witness.json", payload)
    write_csv(output_root / "bench_witness.csv", rows)
    return payload


def bench_junction(
    *,
    bench_root: Path,
    rung: str,
    fit_report: Path,
    output_root: Path,
) -> dict[str, Any]:
    """The same junction witness on the bench, where the true contact position is known.

    The bench emits every contact on an integer frame with a pixel, which is the shape of an
    owner contact click, and its ``truth.json`` records where that contact really was.  So the
    witness the real audit forms can be formed here and its own error measured: how often it
    passes a junction whose fitted states are far from the true contact, and how often it
    fails one that is right.  The bench's own junction tolerance is 0.25 m
    (``docs/wk1/bench_metric.md``), and that is what "truly right" means below.
    """
    truth = json.loads((bench_root / "truth.json").read_text())
    root = bench_root / "roots" / rung
    emissions = json.loads((root / "event_emissions.json").read_text())
    if isinstance(emissions, dict):
        emissions = emissions["emissions"]
    meta: dict[str, dict[str, Any]] = {}
    truth_contacts: dict[str, list[dict[str, Any]]] = {}
    for point in truth["points"]:
        point_id = str(point["point"])
        truth_contacts[point_id] = list(point.get("contacts") or [])
        meta[point_id] = {"match_id": str(point["match_id"]), "clip": str(point["clip"])}
    emitted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in emissions:
        if str(row.get("event_type")) != "contact":
            continue
        location = row.get("location") or {}
        if location.get("image_x") is None:
            continue
        emitted[str(row["clip"])].append(
            {
                "frame": float(row["frame"]),
                "image_xy": (float(location["image_x"]), float(location["image_y"])),
            }
        )

    report = json.loads(Path(fit_report).read_text())
    rows: list[dict[str, Any]] = []
    cameras: dict[tuple[str, str], Any] = {}
    tracks: dict[tuple[str, str], dict[int, Any]] = {}
    for point in report.get("points_detail", []):
        point_id = str(point["point"])
        if point_id not in meta:
            continue
        match_id, clip = meta[point_id]["match_id"], meta[point_id]["clip"]
        key = (match_id, clip)
        if key not in cameras:
            try:
                cameras[key] = reconstruction.PointCamera(root / match_id, clip)
            except (ValueError, FileNotFoundError):
                cameras[key] = None
            tracks[key] = (
                reconstruction.load_track(root / match_id, clip) if cameras[key] is not None else {}
            )
        camera = cameras[key]
        if camera is None:
            continue
        fits = sorted(
            (fit for fit in point.get("fits", []) or [] if fit.get("start_frame") is not None),
            key=lambda fit: float(fit["start_frame"]),
        )
        radius_cache: dict[int, tuple[float, bool]] = {}
        for incoming, outgoing in zip(fits, fits[1:]):
            if incoming.get("end_frame") is None or incoming["end_frame"] != outgoing.get(
                "start_frame"
            ):
                continue
            boundary = float(incoming["end_frame"])
            candidates = [
                event
                for event in emitted.get(point_id, [])
                if abs(event["frame"] - boundary) <= CONTACT_MATCH_FRAMES
            ]
            if not candidates:
                continue
            click = min(candidates, key=lambda event: abs(event["frame"] - boundary))
            radius_px, refiner_abstained = contact_graded_radius_px(
                tracks[key], boundary, radius_cache
            )
            witness = junction_witness(
                camera=camera,
                boundary_frame=boundary,
                click=click,
                inbound_end_xyz=incoming.get("end_xyz"),
                outbound_start_xyz=outgoing.get("start_xyz"),
                radius_px=radius_px,
                refiner_abstained=refiner_abstained,
            )
            record: dict[str, Any] = {
                "point": point_id,
                "match_id": match_id,
                "boundary_frame": boundary,
                "emitted_frame": click["frame"],
                "inbound_flight_index": incoming.get("flight_index"),
                "outbound_flight_index": outgoing.get("flight_index"),
                "formed": bool(witness.get("formed")),
                "reason": witness.get("reason"),
            }
            for field in (
                "gap_m",
                "sigma_m",
                "radius_m",
                "radius_px",
                "ray_max_px",
                "inbound_ray_px",
                "outbound_ray_px",
                "position_score",
                "ray_score",
                "score",
                "hard_pass",
                "graded_pass",
            ):
                if field in witness:
                    record[field] = witness[field]
            # The bench's own junction: the true contact this boundary is supposed to be.
            true_contacts = [
                contact
                for contact in truth_contacts.get(point_id, [])
                if abs(float(contact["frame"]) - boundary) <= EVENT_MATCH_FRAMES
            ]
            if true_contacts:
                true_contact = min(
                    true_contacts, key=lambda contact: abs(float(contact["frame"]) - boundary)
                )
                true_xyz = np.asarray(true_contact["xyz"], dtype=float)
                record["true_frame"] = float(true_contact["frame"])
                record["true_time_error_frames"] = boundary - float(true_contact["frame"])
                for name, fit in (("inbound", incoming), ("outbound", outgoing)):
                    state = fit.get("end_xyz") if name == "inbound" else fit.get("start_xyz")
                    if state is None:
                        continue
                    record[f"{name}_true_error_m"] = float(
                        np.linalg.norm(np.asarray(state, dtype=float) - true_xyz)
                    )
                worst = [
                    record[f"{name}_true_error_m"]
                    for name in ("inbound", "outbound")
                    if record.get(f"{name}_true_error_m") is not None
                ]
                if worst:
                    record["true_error_max_m"] = max(worst)
                    record["truly_right"] = bool(max(worst) <= JUNCTION_WITNESS_FLOOR_M)
            rows.append(record)

    formed = [row for row in rows if row["formed"] and row.get("truly_right") is not None]

    def cell(kept: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [float(row["true_error_max_m"]) for row in kept]
        return {
            "junctions": len(kept),
            "truly_right_at_0.25_m": sum(1 for row in kept if row["truly_right"]),
            "truly_wrong_at_0.25_m": sum(1 for row in kept if not row["truly_right"]),
            "true_error_median_p90_m": [percentile(errors, 50.0), percentile(errors, 90.0)],
            "beyond_1_m": sum(1 for value in errors if value > 1.0),
            "worst_true_error_m": max(errors, default=None),
        }

    def confusion(key: str) -> dict[str, int]:
        return {
            "witness_pass_truly_right": sum(1 for row in formed if row[key] and row["truly_right"]),
            "witness_pass_truly_wrong": sum(
                1 for row in formed if row[key] and not row["truly_right"]
            ),
            "witness_fail_truly_right": sum(
                1 for row in formed if not row[key] and row["truly_right"]
            ),
            "witness_fail_truly_wrong": sum(
                1 for row in formed if not row[key] and not row["truly_right"]
            ),
        }

    def spread(key: str, source: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row[key]) for row in source if row.get(key) is not None]
        return {
            "n": len(values),
            "median": percentile(values, 50.0),
            "p90": percentile(values, 90.0),
            "max": max(values, default=None),
        }

    payload = {
        "schema": "flight_gate_audit_bench_junction_v1",
        "artifact_class": "offline_truth_scoring",
        "rung": rung,
        "sources": {
            "bench_root": os.fspath(bench_root),
            "root": os.fspath(root),
            "fit_report": os.fspath(fit_report),
        },
        "thresholds": {
            "junction_witness_floor_m": JUNCTION_WITNESS_FLOOR_M,
            "junction_witness_sigma_multiple": JUNCTION_WITNESS_SIGMA_MULTIPLE,
            "contact_graded_floor_px": CONTACT_GRADED_FLOOR_PX,
            "graded_pass_score": GRADED_PASS_SCORE,
        },
        "shared_boundaries_with_an_emitted_contact": len(rows),
        "formed": sum(1 for row in rows if row["formed"]),
        "formed_with_a_true_contact": len(formed),
        "abstentions": dict(
            Counter(row["reason"] for row in rows if not row["formed"]).most_common()
        ),
        "distributions": {
            "gap_m": spread("gap_m", formed),
            "radius_m": spread("radius_m", formed),
            "sigma_m": spread("sigma_m", formed),
            "ray_max_px": spread("ray_max_px", formed),
            "radius_px": spread("radius_px", formed),
            "true_error_max_m": spread("true_error_max_m", formed),
            "true_time_error_frames": spread("true_time_error_frames", formed),
        },
        "all_formed": cell(formed),
        "admitted_hard": cell([row for row in formed if row["hard_pass"]]),
        "admitted_graded": cell([row for row in formed if row["graded_pass"]]),
        "rejected_graded": cell([row for row in formed if not row["graded_pass"]]),
        "confusion_hard": confusion("hard_pass"),
        "confusion_graded": confusion("graded_pass"),
        "position_term_only": confusion_position(formed),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "bench_junction.json", payload)
    write_csv(output_root / "bench_junction.csv", rows)
    return payload


def confusion_position(formed: list[dict[str, Any]]) -> dict[str, Any]:
    """Which half of the witness does the work on the bench: the gap or the click ray."""
    position = [row for row in formed if float(row["gap_m"]) <= float(row["radius_m"])]
    ray = [row for row in formed if float(row["ray_max_px"]) <= float(row["radius_px"])]
    return {
        "position_term_passes": len(position),
        "position_term_passes_truly_right": sum(1 for row in position if row["truly_right"]),
        "ray_term_passes": len(ray),
        "ray_term_passes_truly_right": sum(1 for row in ray if row["truly_right"]),
        "position_term_only_failures": sum(
            1
            for row in formed
            if float(row["gap_m"]) > float(row["radius_m"])
            and float(row["ray_max_px"]) <= float(row["radius_px"])
        ),
        "ray_term_only_failures": sum(
            1
            for row in formed
            if float(row["gap_m"]) <= float(row["radius_m"])
            and float(row["ray_max_px"]) > float(row["radius_px"])
        ),
    }


def parser() -> argparse.ArgumentParser:
    root = data_root()
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument(
        "--reconstruction-root", type=Path, default=root / "processed/wk3_nightly/default_3d"
    )
    audit_parser.add_argument(
        "--camera-root", type=Path, default=root / "processed/wk3_nightly/cohort_root_v3"
    )
    audit_parser.add_argument(
        "--truth-events",
        type=Path,
        default=root / "processed/wk3_oracle/oracle_3d_ceiling_v2/truth_events_event_model_v3.json",
    )
    audit_parser.add_argument(
        "--automatic-events",
        type=Path,
        default=root / "processed/wk3_nightly/cohort_root_v3/event_emissions.json",
    )
    audit_parser.add_argument("--output-root", type=Path, required=True)
    audit_parser.add_argument(
        "--click-offset-frames",
        type=float,
        default=CLICK_OFFSET_FRAMES,
        help="added to labeled_frame before the bounce witness is formed (-0.5 is the "
        "leading-blur reading of docs/wk1/HARNESS_LESSONS.md item 5)",
    )

    bench_parser = commands.add_parser("bench-witness")
    bench_parser.add_argument(
        "--bench-root", type=Path, default=root / "processed/wk3_benchframes/bench"
    )
    bench_parser.add_argument("--rung", default="clean")
    bench_parser.add_argument("--click-offset-frames", type=float, default=CLICK_OFFSET_FRAMES)
    bench_parser.add_argument(
        "--fit-report",
        type=Path,
        default=None,
        help="a bench run's report.json on the same rung; its fitted bounces are scored against "
        "the graded circle and against the generator's true bounce, so the circle's leniency "
        "is measured where the truth is known",
    )
    bench_parser.add_argument("--output-root", type=Path, required=True)

    sheet_parser = commands.add_parser("sheets")
    sheet_parser.add_argument("--audit-root", type=Path, required=True)
    sheet_parser.add_argument(
        "--reconstruction-root", type=Path, default=root / "processed/wk3_nightly/default_3d"
    )
    sheet_parser.add_argument(
        "--camera-root", type=Path, default=root / "processed/wk3_nightly/cohort_root_v3"
    )
    sheet_parser.add_argument(
        "--truth-events",
        type=Path,
        default=root / "processed/wk3_oracle/oracle_3d_ceiling_v2/truth_events_event_model_v3.json",
    )
    sheet_parser.add_argument("--output-root", type=Path, required=True)

    junction_parser = commands.add_parser("bench-junction")
    junction_parser.add_argument(
        "--bench-root", type=Path, default=root / "processed/wk3_benchframes/bench"
    )
    junction_parser.add_argument("--rung", default="clean")
    junction_parser.add_argument("--fit-report", type=Path, required=True)
    junction_parser.add_argument("--output-root", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit":
        payload = audit(
            reconstruction_root=args.reconstruction_root,
            camera_root=args.camera_root,
            truth_events=args.truth_events,
            automatic_events=args.automatic_events,
            output_root=args.output_root,
            click_offset_frames=args.click_offset_frames,
        )
        print(
            json.dumps(
                {
                    "witness_coverage": payload["witness_coverage"],
                    "point_causes": payload["point_causes"],
                    "crosstab_truth_40": payload["crosstab_truth_40"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "bench-witness":
        payload = bench_witness(
            bench_root=args.bench_root,
            rung=args.rung,
            output_root=args.output_root,
            click_offset_frames=args.click_offset_frames,
            fit_report=args.fit_report,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
    elif args.command == "bench-junction":
        payload = bench_junction(
            bench_root=args.bench_root,
            rung=args.rung,
            fit_report=args.fit_report,
            output_root=args.output_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
    else:
        payload = sheets(
            audit_root=args.audit_root,
            reconstruction_root=args.reconstruction_root,
            camera_root=args.camera_root,
            truth_events=args.truth_events,
            output_root=args.output_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
