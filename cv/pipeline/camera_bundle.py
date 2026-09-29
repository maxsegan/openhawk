"""Fit one physical broadcast-camera rig per match.

The court solver supplies image-to-court homographies at point anchor frames and a
registered homography track within each point.  A broadcast camera does not translate
between those views: it pans, tilts, and zooms about one optical centre.  This module
therefore fits one world-space camera centre and one radial-distortion coefficient from
all usable anchor frames, with only pan, tilt, and focal length varying by view.  It then
fits those three view parameters to every registered frame.

The output intentionally retains the ``camera_P_per_frame_v1.npz`` core schema used by
the current reconstruction stack.  ``P`` is the fitted pinhole camera.  The additional
``k1`` and ``dist_center`` arrays describe the shared radial lens model and are consumed
through :func:`cv.pipeline.camera_project.project_distorted` by distortion-aware clients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from cv.pipeline.camera_artifacts import RECOVERABLE_REGISTRATION_SOURCES, warp_points
from cv.pipeline.camera_cal import (
    COURT_L,
    COURT_W,
    NET_POST_X,
    NET_STICK_X,
    NET_Y,
    fixed_f_projection_from_ground,
    h_to_projection_fixed_f,
    net_height_at_x,
)
from cv.pipeline.camera_project import project_distorted

FRAME_SCOPE = "bundle_v1"
REPORT_NAME = "camera_bundle_v1.json"
COURT_LANDMARKS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [COURT_W, 0.0, 0.0],
        [0.0, COURT_L, 0.0],
        [COURT_W, COURT_L, 0.0],
        [1.37, NET_Y - 6.40, 0.0],
        [COURT_W - 1.37, NET_Y - 6.40, 0.0],
        [1.37, NET_Y + 6.40, 0.0],
        [COURT_W - 1.37, NET_Y + 6.40, 0.0],
    ],
    dtype=float,
)
NET_CURVE_X = np.unique(
    np.concatenate(
        (
            np.linspace(NET_POST_X[0], NET_POST_X[1], 41),
            np.asarray(NET_STICK_X, dtype=float),
            np.asarray([COURT_W / 2.0]),
        )
    )
)


@dataclass(frozen=True)
class AnchorObservation:
    """Automatic court and height evidence attached to one point anchor."""

    point: int
    frame: int
    homography: np.ndarray  # court plane -> distorted image
    court_pixels: np.ndarray
    net_pixels: np.ndarray | None
    yaw: float
    tilt: float
    focal: float
    center: np.ndarray


@dataclass(frozen=True)
class BundleSolution:
    """Shared rig plus the fitted pan/tilt/focal values at anchor views."""

    center: np.ndarray
    k1: float
    dist_center: np.ndarray
    view_parameters: dict[int, tuple[float, float, float]]
    success: bool
    cost: float
    optimality: float
    evaluations: int


def _frame_number(name: str) -> int:
    return int(Path(name).stem.removeprefix("f_"))


def _project_homography(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    values = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values))))
    pixels = (np.asarray(homography, dtype=float) @ homogeneous.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def projection_matrix(
    center: np.ndarray,
    yaw: float,
    tilt: float,
    focal: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Build ``K R [I|-C]`` for a level pan/tilt head with image y pointing down."""
    width, height = image_size
    forward = np.asarray(
        [
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            math.sin(tilt),
        ],
        dtype=float,
    )
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    norm = float(np.linalg.norm(right))
    if norm <= 1e-9:
        raise ValueError("camera optical axis is parallel to world up")
    right /= norm
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.stack((right, down, forward))
    intrinsic = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return intrinsic @ np.column_stack((rotation, -rotation @ np.asarray(center, dtype=float)))


def camera_center(projection: np.ndarray) -> np.ndarray:
    """Return the Euclidean null-space centre of a finite 3x4 camera."""
    matrix = np.asarray(projection, dtype=float)
    return -np.linalg.solve(matrix[:, :3], matrix[:, 3])


def _view_angles(rotation: np.ndarray) -> tuple[float, float]:
    forward = np.asarray(rotation, dtype=float)[2]
    return float(math.atan2(forward[1], forward[0])), float(math.asin(np.clip(forward[2], -1, 1)))


def _focal_from_vertical_column(
    homography: np.ndarray,
    point_projection: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Recover camera_cal's net focal from its preserved vertical column."""
    width, height = image_size

    def difference(log_focal: float) -> float:
        candidate = fixed_f_projection_from_ground(
            homography, math.exp(log_focal), w=width, h=height
        )
        return float(np.linalg.norm(candidate[:, 2] - point_projection[:, 2]))

    solved = minimize_scalar(
        difference,
        bounds=(math.log(400.0), math.log(40000.0)),
        method="bounded",
        options={"xatol": 1e-7},
    )
    return float(math.exp(solved.x))


def _view_initial_for_center(
    row: AnchorObservation,
    center: np.ndarray,
    dist_center: np.ndarray,
) -> tuple[float, float, float]:
    """Move a decomposed view initialization onto the shared-centre hypothesis."""
    image_to_court = np.linalg.inv(row.homography)
    target_xy = _project_homography(image_to_court, dist_center[None, :])[0]
    target = np.asarray([target_xy[0], target_xy[1], 0.0], dtype=float)
    forward = target - center
    forward /= np.linalg.norm(forward)
    yaw = float(math.atan2(forward[1], forward[0]))
    tilt = float(math.asin(np.clip(forward[2], -1.0, 1.0)))
    old_distance = max(float(np.linalg.norm(target - row.center)), 1e-6)
    new_distance = float(np.linalg.norm(target - center))
    focal = float(np.clip(row.focal * new_distance / old_distance, 400.0, 40000.0))
    return yaw, tilt, focal


def _net_curve(projection: np.ndarray, k1: float, dist_center: np.ndarray) -> np.ndarray:
    world = np.stack(
        (
            NET_CURVE_X,
            np.full_like(NET_CURVE_X, NET_Y),
            np.asarray([net_height_at_x(float(x)) for x in NET_CURVE_X]),
        ),
        axis=1,
    )
    return project_distorted(projection, k1, dist_center, world)


def net_residuals(
    projection: np.ndarray,
    k1: float,
    dist_center: np.ndarray,
    observed: np.ndarray | None,
) -> np.ndarray:
    """Vertical tape residuals, including explicit 1.07 m singles-stick witnesses."""
    if observed is None:
        return np.asarray([], dtype=float)
    pixels = np.asarray(observed, dtype=float).reshape(-1, 2)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if len(pixels) < 3:
        return np.asarray([], dtype=float)
    order = np.argsort(pixels[:, 0])
    pixels = pixels[order]
    curve = _net_curve(projection, k1, dist_center)
    curve = curve[np.argsort(curve[:, 0])]
    predicted_y = np.interp(pixels[:, 0], curve[:, 0], curve[:, 1])
    residual = predicted_y - pixels[:, 1]
    # Give each regulation singles-stick top an explicit residual. The tape polynomial is
    # intentionally allowed to extrapolate over the short hidden end segment.
    stick_world = np.asarray(
        [[x, NET_Y, net_height_at_x(float(x))] for x in NET_STICK_X], dtype=float
    )
    sticks = project_distorted(projection, k1, dist_center, stick_world)
    coefficients = np.polyfit(pixels[:, 0], pixels[:, 1], 2)
    stick_y = np.polyval(coefficients, sticks[:, 0])
    residual = np.concatenate((residual, sticks[:, 1] - stick_y))
    return residual


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array)))) if len(array) else float("nan")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {
        "n": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _image_size(
    match_out: Path, frames_dir: str, references: list[tuple[int, str]]
) -> tuple[int, int]:
    for point, frame in references:
        image = cv2.imread(os.fspath(match_out / frames_dir / f"pt{point:04d}" / frame))
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    return 1920, 1080


def collect_anchor_observations(
    match_out: Path,
    frames_dir: str,
) -> tuple[list[AnchorObservation], tuple[int, int], dict[int, dict[str, Any]]]:
    """Join point homographies, selected anchor frames, and saved net observations."""
    evidence = json.loads((match_out / "court_topology_evidence_v1.json").read_text())
    rows = {
        int(row["point"]): row
        for row in evidence.get("rows", [])
        if row.get("status") == "direct" and row.get("frame")
    }
    with np.load(match_out / "court_H_per_point.npz") as court_data:
        homographies = {
            int(point): np.linalg.inv(np.asarray(homography, dtype=float))
            for point, homography in zip(court_data["pts"], court_data["H"], strict=True)
            if np.isfinite(homography).all()
        }
    with np.load(match_out / "camera_P_per_point.npz", allow_pickle=True) as camera_data:
        point_camera = {
            int(point): index for index, point in enumerate(camera_data["pts"].astype(int))
        }
        references = [(point, str(row["frame"])) for point, row in rows.items()]
        image_size = _image_size(match_out, frames_dir, references)
        observations: list[AnchorObservation] = []
        for point in sorted(set(rows) & set(homographies) & set(point_camera)):
            index = point_camera[point]
            homography = homographies[point]
            point_projection = np.asarray(camera_data["P"][index], dtype=float)
            focal = _focal_from_vertical_column(homography, point_projection, image_size)
            physical = h_to_projection_fixed_f(homography, focal, w=image_size[0], h=image_size[1])
            intrinsic, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(physical)
            del intrinsic
            center = (center_h[:3] / center_h[3]).reshape(3)
            yaw, tilt = _view_angles(rotation)
            valid_cord = "net_cord_valid" in camera_data.files and bool(
                camera_data["net_cord_valid"][index]
            )
            net_pixels = (
                np.asarray(camera_data["net_cord_xy"][index], dtype=float) if valid_cord else None
            )
            observations.append(
                AnchorObservation(
                    point=point,
                    frame=_frame_number(str(rows[point]["frame"])),
                    homography=homography,
                    court_pixels=_project_homography(homography, COURT_LANDMARKS[:, :2]),
                    net_pixels=net_pixels,
                    yaw=yaw,
                    tilt=tilt,
                    focal=focal,
                    center=center,
                )
            )
    return observations, image_size, rows


def fit_bundle(
    observations: list[AnchorObservation],
    image_size: tuple[int, int],
    *,
    maximum_evaluations: int = 600,
) -> BundleSolution:
    """Robustly fit the shared rig and anchor-view parameters."""
    if not observations:
        raise ValueError("camera bundle requires at least one anchor observation")
    if not any(observation.net_pixels is not None for observation in observations):
        raise ValueError("camera bundle requires at least one observed net tape")
    dist_center = np.asarray([image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    radial_scale_sq = float(np.sum(np.square(dist_center)))
    initial_center = np.median(np.stack([row.center for row in observations]), axis=0)
    initial = [*initial_center.tolist(), 0.0]
    for row in observations:
        yaw, tilt, focal = _view_initial_for_center(row, initial_center, dist_center)
        initial.extend((yaw, tilt, math.log(focal)))
    initial_array = np.asarray(initial, dtype=float)
    lower = np.asarray(
        [-100.0, -1000.0, 0.5, -0.03] + [-np.inf, -1.50, math.log(400.0)] * len(observations),
        dtype=float,
    )
    upper = np.asarray(
        [100.0, 1000.0, 100.0, 0.03] + [np.inf, 0.50, math.log(40000.0)] * len(observations),
        dtype=float,
    )
    initial_array = np.maximum(lower + 1e-8, np.minimum(upper - 1e-8, initial_array))

    def residual(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:3]
        k1 = float(parameters[3] / radial_scale_sq)
        values: list[np.ndarray] = []
        for view_index, row in enumerate(observations):
            offset = 4 + 3 * view_index
            yaw, tilt, log_focal = parameters[offset : offset + 3]
            projection = projection_matrix(
                center, float(yaw), float(tilt), math.exp(float(log_focal)), image_size
            )
            predicted = project_distorted(projection, k1, dist_center, COURT_LANDMARKS)
            if not np.isfinite(predicted).all():
                return np.full(
                    sum(16 + (11 if r.net_pixels is not None else 0) for r in observations) + 1,
                    1e4,
                )
            values.append((predicted - row.court_pixels).reshape(-1))
            tape = net_residuals(projection, k1, dist_center, row.net_pixels)
            if len(tape):
                values.append(tape)
        # A weak zero-distortion prior prevents a one-view match from using k1 as a free warp.
        values.append(np.asarray([parameters[3] / 0.20], dtype=float))
        return np.concatenate(values)

    solved = least_squares(
        residual,
        initial_array,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=maximum_evaluations,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    views = {}
    for view_index, row in enumerate(observations):
        offset = 4 + 3 * view_index
        yaw, tilt, log_focal = solved.x[offset : offset + 3]
        views[row.point] = (float(yaw), float(tilt), float(math.exp(log_focal)))
    return BundleSolution(
        center=np.asarray(solved.x[:3], dtype=float),
        k1=float(solved.x[3] / radial_scale_sq),
        dist_center=dist_center,
        view_parameters=views,
        success=bool(solved.success and np.isfinite(solved.x).all()),
        cost=float(solved.cost),
        optimality=float(solved.optimality),
        evaluations=int(solved.nfev),
    )


def fit_frame_view(
    homography: np.ndarray,
    solution: BundleSolution,
    image_size: tuple[int, int],
    initial: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    """Fit pan, tilt, and focal length for one registered ground-plane view."""
    observed = _project_homography(homography, COURT_LANDMARKS[:, :2])

    def residual(parameters: np.ndarray) -> np.ndarray:
        yaw, tilt, log_focal = parameters
        projection = projection_matrix(
            solution.center,
            float(yaw),
            float(tilt),
            math.exp(float(log_focal)),
            image_size,
        )
        pixels = project_distorted(projection, solution.k1, solution.dist_center, COURT_LANDMARKS)
        return (pixels - observed).reshape(-1)

    initial_parameters = np.asarray([initial[0], initial[1], math.log(initial[2])])
    solved = least_squares(
        residual,
        initial_parameters,
        bounds=(
            np.asarray([-np.inf, -1.50, math.log(400.0)]),
            np.asarray([np.inf, 0.50, math.log(40000.0)]),
        ),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=100,
    )
    yaw, tilt, log_focal = solved.x
    focal = float(math.exp(log_focal))
    projection = projection_matrix(solution.center, float(yaw), float(tilt), focal, image_size)
    return projection, (float(yaw), float(tilt), focal), _rms(residual(solved.x))


def frame_or_anchor_view(
    homography: np.ndarray,
    frame: int,
    anchor: AnchorObservation,
    solution: BundleSolution,
    image_size: tuple[int, int],
    initial: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    """Preserve the joint court/net solution at its actual measured anchor exposure.

    Re-fitting that exposure against the ground alone discards its height evidence
    while leaving the old net residual attached to a different camera. Other frames
    retain the existing ground-only pan/tilt/zoom fit and explicit anchor witness.
    """
    if frame != anchor.frame:
        return fit_frame_view(homography, solution, image_size, initial)
    view = solution.view_parameters[anchor.point]
    projection = projection_matrix(solution.center, *view, image_size)
    observed = _project_homography(homography, COURT_LANDMARKS[:, :2])
    predicted = project_distorted(projection, solution.k1, solution.dist_center, COURT_LANDMARKS)
    return projection, view, _rms(predicted - observed)


def _write_visualization(
    match_out: Path,
    frames_dir: str,
    observations: list[AnchorObservation],
    solution: BundleSolution,
    image_size: tuple[int, int],
) -> str | None:
    panels = []
    for row in observations[:6]:
        frame_path = match_out / frames_dir / f"pt{row.point:04d}" / f"f_{row.frame:04d}.jpg"
        image = cv2.imread(os.fspath(frame_path))
        if image is None:
            continue
        yaw, tilt, focal = solution.view_parameters[row.point]
        projection = projection_matrix(solution.center, yaw, tilt, focal, image_size)
        court_pixels = project_distorted(
            projection, solution.k1, solution.dist_center, COURT_LANDMARKS
        )
        for pixel in court_pixels:
            cv2.circle(image, tuple(np.round(pixel).astype(int)), 6, (0, 255, 0), 2)
        curve = _net_curve(projection, solution.k1, solution.dist_center)
        cv2.polylines(image, [np.round(curve).astype(np.int32)], False, (0, 255, 255), 2)
        cv2.putText(
            image,
            f"pt{row.point:04d} f={focal:.0f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    if not panels:
        return None
    output = match_out / "camera_bundle_validation.jpg"
    cv2.imwrite(os.fspath(output), np.hstack(panels))
    return os.fspath(output)


def _write_empty_bundle(
    match_out: Path,
    image_size: tuple[int, int],
    reason: str,
) -> Path:
    """Materialize a schema-valid abstention when physical height evidence is absent."""
    output = match_out / "camera_P_per_frame_v1.npz"
    np.savez_compressed(
        output,
        clips=np.asarray([], dtype=str),
        frames=np.asarray([], dtype=np.int32),
        P=np.empty((0, 3, 4), dtype=float),
        reliable=np.asarray([], dtype=bool),
        source=np.asarray([], dtype=str),
        ground_residual_px=np.asarray([], dtype=float),
        net_residual_px=np.asarray([], dtype=float),
        confidence=np.asarray([], dtype=float),
        fallback_ancestry=np.asarray([], dtype=str),
        frame_scope=np.asarray([], dtype=str),
        reference_frame=np.asarray([], dtype=str),
        calibration_point=np.asarray([], dtype=np.int32),
        net_cord_xy=np.empty((0, 9, 2), dtype=float),
        net_cord_valid=np.asarray([], dtype=bool),
        net_cord_source=np.asarray([], dtype=str),
        k1=np.asarray([], dtype=float),
        dist_center=np.empty((0, 2), dtype=float),
        camera_center=np.empty((0, 3), dtype=float),
        pan_rad=np.asarray([], dtype=float),
        tilt_rad=np.asarray([], dtype=float),
        focal_px=np.asarray([], dtype=float),
    )
    report = {
        "schema": "camera_bundle_v1",
        "automatic": True,
        "status": "abstained",
        "reason": reason,
        "frame_scope": FRAME_SCOPE,
        "match": match_out.name,
        "image_size": list(image_size),
        "anchor_views": 0,
        "net_observed_anchor_views": 0,
        "frames": 0,
        "reliable_frames": 0,
        "visualization": None,
    }
    (match_out / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def bundle_match_cameras(match_out: Path, frames_dir: str) -> Path:
    """Fit the match rig and write a bundle-derived per-frame camera artifact."""
    match_out = Path(match_out)
    observations, image_size, evidence_rows = collect_anchor_observations(match_out, frames_dir)
    if not observations:
        return _write_empty_bundle(match_out, image_size, "no_automatic_court_anchors")
    if not any(row.net_pixels is not None for row in observations):
        return _write_empty_bundle(match_out, image_size, "no_automatic_height_witness")
    solution = fit_bundle(observations, image_size)
    if not solution.success:
        raise RuntimeError("camera bundle optimization did not converge")
    anchors = {row.point: row for row in observations}
    frame_track_path = match_out / "court_H_per_frame_v1.npz"
    if frame_track_path.is_file():
        with np.load(frame_track_path, allow_pickle=True) as track:
            clips = np.asarray(track["clips"], dtype=str)
            frames = np.asarray(track["frames"], dtype=np.int32)
            homographies = np.asarray(track["H"], dtype=float)
            registration_reliable = np.asarray(track["reliable"], dtype=bool)
            registration_sources = np.asarray(track["source"], dtype=str)
    else:
        # Calibration-only evaluation roots predate the per-frame track. Their anchor rows
        # still exercise the shared-centre and height solve without inventing other frames.
        clips = np.asarray([f"pt{row.point:04d}" for row in observations], dtype=str)
        frames = np.asarray([row.frame for row in observations], dtype=np.int32)
        homographies = np.stack([np.linalg.inv(row.homography) for row in observations])
        registration_reliable = np.ones(len(observations), dtype=bool)
        registration_sources = np.repeat("registered_anchor", len(observations))

    projections = []
    reliable = []
    sources = []
    ground_residual = []
    net_residual = []
    confidence = []
    ancestry = []
    references = []
    calibration_points = []
    net_cords = []
    net_cord_valid = []
    net_cord_source = []
    net_residual_scope = []
    view_fit_source = []
    focals = []
    pans = []
    tilts = []
    last_view: dict[int, tuple[float, float, float]] = dict(solution.view_parameters)
    match_net_values = []
    anchor_net_residual: dict[int, float] = {}
    for row in observations:
        yaw, tilt, focal = solution.view_parameters[row.point]
        projection = projection_matrix(solution.center, yaw, tilt, focal, image_size)
        value = _rms(net_residuals(projection, solution.k1, solution.dist_center, row.net_pixels))
        if np.isfinite(value):
            match_net_values.append(value)
            anchor_net_residual[row.point] = value
    shared_net_residual = float(np.median(match_net_values)) if match_net_values else float("nan")

    for clip, frame, homography, registration_ok, registration_source in zip(
        clips,
        frames,
        homographies,
        registration_reliable,
        registration_sources,
        strict=True,
    ):
        point = int(str(clip).removeprefix("pt"))
        if point not in anchors:
            raise ValueError(f"frame track point {point} has no bundle anchor")
        anchor = anchors[point]
        projection, view, ground = frame_or_anchor_view(
            np.linalg.inv(homography), int(frame), anchor, solution, image_size, last_view[point]
        )
        last_view[point] = view
        anchor_h_image_to_court = np.linalg.inv(anchor.homography)
        frame_warp = np.linalg.inv(homography) @ anchor_h_image_to_court
        cord = (
            warp_points(frame_warp, anchor.net_pixels)
            if anchor.net_pixels is not None
            else np.full((9, 2), np.nan, dtype=float)
        )
        # Net pixels exist only at the anchor. A planar registration cannot transport an
        # above-plane tape correctly, so carry the measured anchor residual as the height
        # quality witness instead of manufacturing a per-frame net observation.
        frame_net = anchor_net_residual.get(point, shared_net_residual)
        if int(frame) == anchor.frame and anchor.net_pixels is not None:
            frame_net = _rms(
                net_residuals(projection, solution.k1, solution.dist_center, anchor.net_pixels)
            )
            net_residual_scope.append("same_frame_observed_net")
        else:
            net_residual_scope.append(
                "point_anchor_observed_net"
                if point in anchor_net_residual
                else "match_anchor_median"
            )
        view_fit_source.append(
            "joint_court_net_anchor" if int(frame) == anchor.frame else "ground_only_frame_view"
        )
        lineage_ok = (
            bool(registration_ok) or str(registration_source) in RECOVERABLE_REGISTRATION_SOURCES
        )
        is_reliable = bool(
            lineage_ok
            and np.isfinite(ground)
            and ground <= 4.0
            and np.isfinite(frame_net)
            and frame_net <= 8.0
        )
        projections.append(projection)
        reliable.append(is_reliable)
        sources.append(f"bundle_v1+{registration_source}")
        ground_residual.append(ground)
        net_residual.append(frame_net)
        confidence.append(float(math.exp(-ground / 4.0 - frame_net / 8.0)) if is_reliable else 0.0)
        ancestry.append("[]")
        references.append(f"f_{anchor.frame:04d}.jpg")
        calibration_points.append(point)
        net_cords.append(cord)
        net_cord_valid.append(anchor.net_pixels is not None)
        point_camera_index = evidence_rows.get(point, {})
        net_cord_source.append(
            "observed_connected_tape_bundle_warped"
            if anchor.net_pixels is not None
            else "match_bundle_height_witness"
        )
        del point_camera_index
        pans.append(view[0])
        tilts.append(view[1])
        focals.append(view[2])

    output = match_out / "camera_P_per_frame_v1.npz"
    np.savez_compressed(
        output,
        clips=clips,
        frames=frames,
        P=np.stack(projections),
        reliable=np.asarray(reliable, dtype=bool),
        source=np.asarray(sources, dtype=str),
        ground_residual_px=np.asarray(ground_residual, dtype=float),
        net_residual_px=np.asarray(net_residual, dtype=float),
        net_residual_scope=np.asarray(net_residual_scope, dtype=str),
        view_fit_source=np.asarray(view_fit_source, dtype=str),
        confidence=np.asarray(confidence, dtype=float),
        fallback_ancestry=np.asarray(ancestry, dtype=str),
        frame_scope=np.repeat(FRAME_SCOPE, len(frames)),
        reference_frame=np.asarray(references, dtype=str),
        calibration_point=np.asarray(calibration_points, dtype=np.int32),
        net_cord_xy=np.stack(net_cords),
        net_cord_valid=np.asarray(net_cord_valid, dtype=bool),
        net_cord_source=np.asarray(net_cord_source, dtype=str),
        k1=np.repeat(solution.k1, len(frames)),
        dist_center=np.repeat(solution.dist_center[None, :], len(frames), axis=0),
        camera_center=np.repeat(solution.center[None, :], len(frames), axis=0),
        pan_rad=np.asarray(pans, dtype=float),
        tilt_rad=np.asarray(tilts, dtype=float),
        focal_px=np.asarray(focals, dtype=float),
    )
    anchor_rows = []
    for row in observations:
        yaw, tilt, focal = solution.view_parameters[row.point]
        projection = projection_matrix(solution.center, yaw, tilt, focal, image_size)
        predicted = project_distorted(
            projection, solution.k1, solution.dist_center, COURT_LANDMARKS
        )
        anchor_rows.append(
            {
                "point": row.point,
                "frame": row.frame,
                "focal_px": focal,
                "court_landmark_rms_px": _rms(predicted - row.court_pixels),
                "net_tape_rms_px": _rms(
                    net_residuals(projection, solution.k1, solution.dist_center, row.net_pixels)
                ),
                "net_observed": row.net_pixels is not None,
            }
        )
    visualization = _write_visualization(match_out, frames_dir, observations, solution, image_size)
    report = {
        "schema": "camera_bundle_v1",
        "automatic": True,
        "frame_scope": FRAME_SCOPE,
        "match": match_out.name,
        "image_size": list(image_size),
        "anchor_views": len(observations),
        "net_observed_anchor_views": sum(row.net_pixels is not None for row in observations),
        "camera_center_m": solution.center.tolist(),
        "shared_k1_pixel_inverse_squared": solution.k1,
        "distortion_center_px": solution.dist_center.tolist(),
        "optimizer": {
            "success": solution.success,
            "cost": solution.cost,
            "optimality": solution.optimality,
            "evaluations": solution.evaluations,
            "loss": "soft_l1",
        },
        "anchor_court_residual_px": _summary(
            [float(row["court_landmark_rms_px"]) for row in anchor_rows]
        ),
        "anchor_net_residual_px": _summary([float(row["net_tape_rms_px"]) for row in anchor_rows]),
        "frame_ground_residual_px": _summary(ground_residual),
        "frame_net_residual_px": _summary(net_residual),
        "reliable_frames": int(np.count_nonzero(reliable)),
        "frames": len(frames),
        "anchor_view_policy": "preserve_joint_court_net_solution_v1",
        "visualization": visualization,
        "anchors": anchor_rows,
    }
    (match_out / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="one canonical match directory")
    parser.add_argument("--frames-dir", default="audit_frames_native_1080")
    args = parser.parse_args()
    output = bundle_match_cameras(args.out, args.frames_dir)
    print(output)
    print((args.out / REPORT_NAME).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
