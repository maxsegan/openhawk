"""Upgrade a ground-plane homography to a full camera projection matrix.

The per-point homographies map the COURT PLANE (z=0) to the image — enough for player feet
and bounce points, but the ball flies above the plane, so fitting 3D flight arcs needs the
full projection P = K [R | t].

The reusable joint-camera interface in this module also fits a fixed ITF court across
many views: one optical centre and radial k1, with smooth per-frame pan/tilt/focal.
It accepts label points or automatic point-to-line observations; callers retain the
provenance boundary between those two evidence types.

Classic plane-based self-calibration: for the ground plane, H ~ K [r1 r2 t]. Assuming
square pixels, zero skew, principal point at the image center, the two constraints
(K^-1 h1) . (K^-1 h2) = 0 and |K^-1 h1| = |K^-1 h2| give the focal length; then
r1, r2, t follow and r3 = r1 x r2.

Court frame: x across (0..10.97), y along (0..23.77), z up (meters). Validation: project
the NET (a 3D feature the homography cannot express): tape at 1.07 m at the posts
(x = -0.914, 11.89), 0.914 m at the center, along y = 11.885.

    .venv/bin/python cv/pipeline/camera_cal.py --out data/processed/rg2025f --frames-dir rally_frames_25
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares

from cv.pipeline.provenance import file_record
from cv.pipeline.run_manifest import StageRun

W_IMG, H_IMG = 960, 540  # analysis frame size
COURT_W, COURT_L = 10.97, 23.77
NET_Y = COURT_L / 2
SERVICE_FROM_NET = 6.40
NET_POST_X = (-0.914, COURT_W + 0.914)  # posts sit 0.914 m outside the doubles lines
NET_H_POST, NET_H_CENTER = 1.07, 0.914
SINGLES_INSET = 1.37  # doubles alley width; singles sidelines at 1.37 and COURT_W - 1.37
# In a singles match on a doubles court the tape is carried at 1.07 m by singles sticks
# 0.914 m outside the singles sidelines, not by the doubles posts. Modelling the sag out
# to the posts puts the cord about 6 cm low where the sticks actually hold it.
NET_STICK_X = (
    SINGLES_INSET - 0.914,
    COURT_W - SINGLES_INSET + 0.914,
)
NET_SUPPORT = "singles_sticks"
TAPE_MEASUREMENTS = ("hough_top", "connected_pixels")


def net_source_is_tape_top(source: str) -> bool:
    """Whether a cord observation claims a top boundary rather than a mesh centre."""
    return source in {
        "observed_connected_tape_top_envelope",
        "observed_connected_tape_pixels_top",
        "observed_for_residual_only",
    }


def net_support_half_span(support: str = NET_SUPPORT) -> float:
    """Half-width from the centre to whatever holds the tape at ``NET_H_POST``."""
    if support == "doubles_posts":
        return COURT_W / 2.0 + abs(NET_POST_X[0])
    return COURT_W / 2.0 - NET_STICK_X[0]


def net_height_at_x(court_x: float, support: str = NET_SUPPORT) -> float:
    """Regulation tape height at a court x, sagging from the support to the centre.

    Beyond the support the cord runs level between two 1.07 m tie points, so it is held
    at ``NET_H_POST`` rather than continuing to rise.
    """
    half_span = net_support_half_span(support)
    normalized = np.clip(abs(court_x - COURT_W / 2.0) / half_span, 0.0, 1.0)
    return float(NET_H_CENTER + (NET_H_POST - NET_H_CENTER) * normalized**2)


def fit_net_top_envelope(
    candidates: list[tuple[float, float, tuple[int, int, int, int]]],
    visible_left: float,
    visible_right: float,
) -> np.ndarray | None:
    """Fit the upper supported boundary of a connected net-tape line group.

    Hough commonly emits many nearly horizontal lines across the full thickness of the tape and
    upper mesh. Averaging those lines biases the physical cord downward. The regulation height is
    measured at the top of the cord, so fit a low image-y envelope while requiring support across
    the visible span.
    """

    if visible_right <= visible_left or not candidates:
        return None

    def line_y(candidate, image_x: float) -> float:
        _mean_y, _negative_overlap, (x1, y1, x2, y2) = candidate
        ratio = (image_x - float(x1)) / max(float(x2 - x1), 1.0)
        return float(y1) + ratio * float(y2 - y1)

    probe_x = np.linspace(visible_left, visible_right, 65)
    supported_x = []
    top_y = []
    horizontal_slack = max(3.0, 0.006 * (visible_right - visible_left))
    for image_x in probe_x:
        values = [
            line_y(candidate, image_x)
            for candidate in candidates
            if float(candidate[2][0]) - horizontal_slack
            <= image_x
            <= float(candidate[2][2]) + horizontal_slack
        ]
        if values:
            supported_x.append(float(image_x))
            top_y.append(float(np.quantile(values, 0.10)))
    if len(supported_x) < 12:
        return None
    supported_x_array = np.asarray(supported_x, dtype=float)
    top_y_array = np.asarray(top_y, dtype=float)
    if np.ptp(supported_x_array) < 0.45 * (visible_right - visible_left):
        return None
    coefficients = np.polyfit(supported_x_array, top_y_array, 2)
    for _ in range(3):
        residual = np.abs(np.polyval(coefficients, supported_x_array) - top_y_array)
        cutoff = max(3.0, float(np.median(residual)) * 2.5)
        inliers = residual <= cutoff
        if int(np.count_nonzero(inliers)) < 8:
            break
        coefficients = np.polyfit(supported_x_array[inliers], top_y_array[inliers], 2)
    return coefficients


def fit_connected_net_lines(
    candidates: list[tuple[float, float, tuple[int, int, int, int]]],
) -> np.ndarray | None:
    """Legacy robust centreline fit used only when a supported top envelope is unavailable."""

    fit_x = []
    fit_y = []
    for _mean_y, _negative_overlap, (x1, y1, x2, y2) in candidates:
        segment_x = np.linspace(float(x1), float(x2), max(3, round((x2 - x1) / 30)))
        ratio = (segment_x - float(x1)) / max(float(x2 - x1), 1.0)
        segment_y = float(y1) + ratio * float(y2 - y1)
        fit_x.extend(segment_x.tolist())
        fit_y.extend(segment_y.tolist())
    if len(fit_x) < 3:
        return None
    fit_x_array = np.asarray(fit_x, dtype=float)
    fit_y_array = np.asarray(fit_y, dtype=float)
    degree = 2 if np.ptp(fit_x_array) >= 1.0 and len(candidates) >= 2 else 1
    coefficients = np.polyfit(fit_x_array, fit_y_array, degree)
    for _ in range(3):
        residual = np.abs(np.polyval(coefficients, fit_x_array) - fit_y_array)
        inliers = residual <= max(5.0, float(np.median(residual)) * 2.5)
        if int(np.count_nonzero(inliers)) < degree + 2:
            break
        coefficients = np.polyfit(fit_x_array[inliers], fit_y_array[inliers], degree)
    return coefficients


def _fit_visible_net_tape_top(
    image: np.ndarray,
    proposal: np.ndarray,
    visible_left: float,
    visible_right: float,
    service_segments: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Measure a connected white tape boundary inside a line-based proposal.

    Long mesh edges can vastly outnumber short curved tape edges in Hough output.
    Their percentile envelope locates a search region, but is not itself evidence
    of the tape. Require actual broad, horizontally connected bright pixels and
    keep only their supported top boundary. No physical height is used here.
    """
    height, width = image.shape[:2]
    span = visible_right - visible_left
    if span <= 0 or not np.isfinite(proposal).all():
        return None
    xs = np.arange(max(0, int(np.ceil(visible_left))), min(width, int(visible_right) + 1))
    if len(xs) < 12:
        return None
    proposed_y = np.polyval(proposal, xs)
    radius = max(6.0, 0.025 * height)
    y0 = max(0, int(np.floor(np.min(proposed_y) - radius)))
    y1 = min(height, int(np.ceil(np.max(proposed_y) + radius + 1)))
    if y1 <= y0:
        return None
    ys = np.arange(y0, y1)[:, None]
    pixels = image[y0:y1, xs].astype(np.float32)
    minimum = pixels.min(axis=2)
    maximum = pixels.max(axis=2)
    mask = (
        (minimum >= 100.0)
        & (maximum - minimum <= 0.25 * maximum + 15.0)
        & (np.abs(ys - proposed_y[None, :]) <= radius)
    )
    # Painted service lines can enter the search region at a shallow camera view.
    # Their ground-plane location is already observed independently of net height.
    for start, end in (service_segments[:2], service_segments[2:]):
        dx = float(end[0] - start[0])
        if abs(dx) < 1e-6:
            continue
        line_y = start[1] + (xs - start[0]) * (end[1] - start[1]) / dx
        mask &= np.abs(ys - line_y[None, :]) > 0.012 * height
    # Remove vertical centre straps, crossing sidelines and isolated bright ball
    # pixels. The tape may curve gradually or be interrupted by a player.
    horizontal = np.ones((1, max(5, round(0.01 * span))), dtype=np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, horizontal)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    components = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_WIDTH] >= 0.12 * span
        and stats[index, cv2.CC_STAT_AREA] / max(stats[index, cv2.CC_STAT_WIDTH], 1)
        >= max(1.5, 0.0015 * height)
    ]
    if not components:
        return None
    supported = np.isin(labels, components)
    columns = supported.any(axis=0)
    sample_x = xs[columns].astype(float)
    sample_y = y0 + np.argmax(supported, axis=0)[columns].astype(float) - 0.5
    minimum_support = max(12, int(np.ceil(0.45 * span)))
    if len(sample_x) < minimum_support or np.ptp(sample_x) < 0.45 * span:
        return None
    coefficients = np.polyfit(sample_x, sample_y, 2)
    for _ in range(4):
        residual = np.abs(np.polyval(coefficients, sample_x) - sample_y)
        cutoff = max(1.0, 0.0015 * height, 2.5 * float(np.median(residual)))
        inliers = residual <= cutoff
        if np.count_nonzero(inliers) < minimum_support:
            return None
        coefficients = np.polyfit(sample_x[inliers], sample_y[inliers], 2)
    residual = np.abs(np.polyval(coefficients, sample_x) - sample_y)
    inliers = residual <= max(1.0, 0.0015 * height, 2.5 * float(np.median(residual)))
    if (
        np.count_nonzero(inliers) < minimum_support
        or np.ptp(sample_x[inliers]) < 0.45 * span
        or float(np.quantile(residual[inliers], 0.9)) > max(2.0, 0.003 * height)
    ):
        return None
    return coefficients, float(sample_x[inliers].min()), float(sample_x[inliers].max())


def h_to_projection(H: np.ndarray, w: int = W_IMG, h: int = H_IMG):
    """Return (P 3x4, K, R, t, f) from a ground-plane homography, or None if degenerate."""
    cx, cy = w / 2.0, h / 2.0
    h1, h2 = H[:, 0], H[:, 1]
    # with K = diag(f, f, 1) shifted by (cx, cy):  K^-1 [u v w]^T = [(u - cx w)/f, (v - cy w)/f, w]
    a1 = np.array([h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]])
    a2 = np.array([h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]])
    # orthogonality: (a1x a2x + a1y a2y)/f^2 + a1z a2z = 0  ->  f^2 = -(a1x a2x + a1y a2y)/(a1z a2z)
    num_o = a1[0] * a2[0] + a1[1] * a2[1]
    den_o = a1[2] * a2[2]
    # equal norm: (|a1xy|^2 - |a2xy|^2)/f^2 + (a1z^2 - a2z^2) = 0
    num_e = (a1[0] ** 2 + a1[1] ** 2) - (a2[0] ** 2 + a2[1] ** 2)
    den_e = a1[2] ** 2 - a2[2] ** 2
    cands = []
    if den_o != 0 and -num_o / den_o > 0:
        cands.append(-num_o / den_o)
    if den_e != 0 and -num_e / den_e > 0:
        cands.append(-num_e / den_e)
    if not cands:
        return None
    f = float(np.sqrt(np.mean(cands)))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    r1 = Kinv @ H[:, 0]
    lam = 1.0 / np.linalg.norm(r1)
    # sign: camera must be on the positive side (t_z > 0 for points in front)
    r1 = r1 * lam
    r2 = (Kinv @ H[:, 1]) * lam
    t = (Kinv @ H[:, 2]) * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    Rm = np.stack([r1, r2, r3], axis=1)
    # orthonormalize via SVD (H noise makes R slightly non-orthogonal)
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    P = K @ np.hstack([Rm, t.reshape(3, 1)])
    return P, K, Rm, t, f


def h_to_projection_fixed_f(H: np.ndarray, f: float, w: int = W_IMG, h: int = H_IMG):
    """Build P from H with a GIVEN focal length (telephoto cameras are near-affine, so f
    is unobservable from one homography's orthogonality; we measure it via the net)."""
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    r1 = Kinv @ H[:, 0]
    lam = 1.0 / np.linalg.norm(r1)
    r1, r2, t = r1 * lam, (Kinv @ H[:, 1]) * lam, (Kinv @ H[:, 2]) * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    Rm = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    return K @ np.hstack([Rm, t.reshape(3, 1)])


def measure_net_band(
    img: np.ndarray,
    H: np.ndarray,
    *,
    prefer_top_envelope: bool = True,
    support: str = NET_SUPPORT,
    tape_measurement: str = "hough_top",
):
    """Measure visible net-top samples and map them to physical net coordinates.

    Searches above the H-projected net ground line for the bright tape. Each returned
    observation is ``(court_x, net_height_m, image_x, ground_y, top_y)``. Visible tape
    segment endpoints must not be treated as regulation post positions. The optional
    ``connected_pixels`` method measures the native white boundary inside a Hough
    proposal; the default retains the historical Hough envelope.
    """
    if tape_measurement not in TAPE_MEASUREMENTS:
        raise ValueError(f"unknown tape measurement {tape_measurement!r}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = np.float32([[NET_POST_X[0], NET_Y], [COURT_W / 2, NET_Y], [NET_POST_X[1], NET_Y]])
    gnd = cv2.perspectiveTransform(pts.reshape(1, -1, 2), H)[0]
    court = cv2.perspectiveTransform(
        np.float32([[0.0, NET_Y], [COURT_W, NET_Y]]).reshape(1, -1, 2),
        H,
    )[0]
    left, right = sorted((float(court[0, 0]), float(court[1, 0])))
    span = max(right - left, 1.0)
    ground_y = float(np.mean(court[:, 1]))
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(40, round(0.04 * span)),
        minLineLength=max(60, round(0.12 * span)),
        maxLineGap=max(12, round(0.04 * span)),
    )
    tape_candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4) if lines is not None else []:
        if x1 > x2:
            x1, x2, y1, y2 = x2, x1, y2, y1
        overlap = max(0.0, min(float(x2), right) - max(float(x1), left))
        slope = abs(float(y2 - y1)) / max(float(x2 - x1), 1.0)
        mean_y = 0.5 * (float(y1) + float(y2))
        height_fraction = (ground_y - mean_y) / img.shape[0]
        if overlap >= 0.12 * span and slope <= 0.10 and 0.035 <= height_fraction <= 0.115:
            tape_candidates.append((mean_y, -overlap, (x1, y1, x2, y2)))
    if tape_candidates:
        service_points = cv2.perspectiveTransform(
            np.float32(
                [
                    [
                        [0.0, NET_Y - SERVICE_FROM_NET],
                        [COURT_W, NET_Y - SERVICE_FROM_NET],
                        [0.0, NET_Y + SERVICE_FROM_NET],
                        [COURT_W, NET_Y + SERVICE_FROM_NET],
                    ]
                ]
            ),
            H,
        )[0]

        def service_distance(candidate) -> float:
            mean_y, _negative_overlap, (x1, _y1, x2, _y2) = candidate
            image_x = 0.5 * (x1 + x2)
            distances = []
            for start, end in (service_points[:2], service_points[2:]):
                ratio = (image_x - start[0]) / max(end[0] - start[0], 1e-6)
                distances.append(abs(mean_y - (start[1] + ratio * (end[1] - start[1]))))
            return min(distances)

        # Reject known painted court lines before selecting the upper edge of the net mesh.
        tape_candidates = [
            candidate
            for candidate in tape_candidates
            if service_distance(candidate) > 0.012 * img.shape[0]
        ]
        if not tape_candidates:
            return None

        def line_y(candidate, image_x: float) -> float:
            _mean_y, _negative_overlap, (x1, y1, x2, y2) = candidate
            ratio = (image_x - float(x1)) / max(float(x2 - x1), 1.0)
            return float(y1) + ratio * float(y2 - y1)

        def compatible(first, second) -> bool:
            first_line, second_line = first[2], second[2]
            overlap_left = max(float(first_line[0]), float(second_line[0]))
            overlap_right = min(float(first_line[2]), float(second_line[2]))
            if overlap_right >= overlap_left:
                probe = np.linspace(overlap_left, overlap_right, 3)
                delta = np.median([abs(line_y(first, x) - line_y(second, x)) for x in probe])
                return bool(delta <= 8.0)
            if first_line[2] < second_line[0]:
                gap = float(second_line[0] - first_line[2])
                boundary_x = 0.5 * float(first_line[2] + second_line[0])
            else:
                gap = float(first_line[0] - second_line[2])
                boundary_x = 0.5 * float(second_line[2] + first_line[0])
            return (
                gap <= 0.15 * span
                and abs(line_y(first, boundary_x) - line_y(second, boundary_x)) <= 12.0
            )

        groups = []
        for seed in tape_candidates:
            group = [seed]
            changed = True
            while changed:
                changed = False
                for candidate in tape_candidates:
                    if candidate in group:
                        continue
                    if any(compatible(candidate, member) for member in group):
                        group.append(candidate)
                        changed = True
            groups.append(group)

        def group_score(group) -> tuple[float, float, float]:
            covered_left = min(max(float(row[2][0]), left) for row in group)
            covered_right = max(min(float(row[2][2]), right) for row in group)
            support = sum(-float(row[1]) for row in group)
            return covered_right - covered_left, support, -float(np.mean([row[0] for row in group]))

        group_scores = [(group_score(group), group) for group in groups]
        widest = max(score[0] for score, _group in group_scores)
        near_full_coverage = [
            (score, group) for score, group in group_scores if score[0] >= 0.95 * widest
        ]
        selected = max(
            near_full_coverage,
            key=lambda item: (item[0][2], item[0][1]),
        )[1]
        visible_left = max(min(float(row[2][0]) for row in selected), left)
        visible_right = min(max(float(row[2][2]) for row in selected), right)
        if visible_right - visible_left < 0.35 * span:
            return None

        coefficients = (
            fit_net_top_envelope(selected, visible_left, visible_right)
            if prefer_top_envelope
            else None
        )
        if coefficients is None:
            coefficients = fit_connected_net_lines(selected)
        if coefficients is None:
            return None
        if prefer_top_envelope and tape_measurement == "connected_pixels":
            measured = _fit_visible_net_tape_top(
                img, coefficients, visible_left, visible_right, service_points
            )
            if measured is None:
                return None
            coefficients, visible_left, visible_right = measured
        sample_xs = np.linspace(visible_left, visible_right, 9)
        sample_ys = np.polyval(coefficients, sample_xs).tolist()
        ground_rows = np.interp(
            sample_xs,
            np.sort(court[:, 0]),
            court[np.argsort(court[:, 0]), 1],
        )
        image_to_court = np.linalg.inv(H)
        ground_pixels = [
            [sample_x, ground_row]
            for sample_x, ground_row in zip(sample_xs, ground_rows, strict=True)
        ]
        court_points = cv2.perspectiveTransform(
            np.float32([ground_pixels]),
            image_to_court,
        )[0]
        return [
            (
                float(court_point[0]),
                net_height_at_x(float(court_point[0]), support),
                float(sample_x),
                float(ground_row),
                sample_y,
            )
            for court_point, sample_x, ground_row, sample_y in zip(
                court_points,
                sample_xs,
                ground_rows,
                sample_ys,
                strict=True,
            )
        ]

    if prefer_top_envelope and tape_measurement == "connected_pixels":
        # Isolated bright columns do not establish a connected tape boundary.
        return None

    search_height = max(60, round(0.12 * img.shape[0]))
    out = []
    for (court_x, _), (gx, gy) in zip(pts, gnd, strict=True):
        xi, yi = int(round(gx)), int(round(gy))
        if not (10 <= xi < img.shape[1] - 10 and 30 <= yi < img.shape[0] - 5):
            continue
        col = gray[max(0, yi - search_height) : yi + 3, max(0, xi - 2) : xi + 3].mean(axis=1)
        # net band = bright run just above ground; find its top edge scanning upward
        bright = col > np.percentile(col, 70) + 5
        # walk up from the bottom until bright run ends
        j = len(col) - 1
        while j > 0 and not bright[j]:
            j -= 1
        top = j
        while top > 0 and bright[top]:
            top -= 1
        h_px = (len(col) - 1) - top
        if 0.008 * img.shape[0] <= h_px <= 0.12 * img.shape[0]:
            court_x = float(court_x)
            out.append(
                (
                    court_x,
                    net_height_at_x(court_x, support),
                    float(gx),
                    float(gy),
                    float(gy - h_px),
                )
            )
    return out if len(out) >= 2 else None


def solve_f_from_net(
    H: np.ndarray,
    net_obs,
    w: int = W_IMG,
    h: int = H_IMG,
    support: str = NET_SUPPORT,
) -> float | None:
    """Solve focal length by matching the physical cord curve to observed tape pixels.

    Tape pixels do not share image x with their ground-plane foot, so assigning each observed
    image column to a court x through ``H`` biases the inferred perspective. Match the two curves
    without fixed point correspondences instead.
    """

    def err(f):
        P = fixed_f_projection_from_ground(H, f, w=w, h=h)
        residuals = net_observation_residuals(P, net_obs, support=support)
        return float(np.mean(np.square(residuals))) if len(residuals) else float("inf")

    fs = np.geomspace(400, 40000, 120)
    errs = [err(f) for f in fs]
    i = int(np.argmin(errs))
    if i in (0, len(fs) - 1):
        return None
    # local refine
    lo, hi = fs[max(0, i - 1)], fs[min(len(fs) - 1, i + 1)]
    fs2 = np.linspace(lo, hi, 60)
    return float(fs2[int(np.argmin([err(f) for f in fs2]))])


def net_observation_residuals(
    P: np.ndarray,
    net_obs: list[tuple[float, float, float, float, float]],
    support: str = NET_SUPPORT,
) -> np.ndarray:
    """Symmetric image-space residuals between observed tape and a physical cord projection."""
    observed = np.asarray([[row[2], row[4]] for row in net_obs], dtype=float)
    if len(observed) < 2 or not np.isfinite(observed).all():
        return np.asarray([], dtype=float)
    court_x = np.linspace(NET_POST_X[0], NET_POST_X[1], 161)
    physical = project(
        P,
        np.stack(
            [
                court_x,
                np.full_like(court_x, NET_Y),
                [net_height_at_x(float(x), support) for x in court_x],
            ],
            axis=1,
        ),
    )
    physical = physical[np.isfinite(physical).all(axis=1)]
    if not len(physical):
        return np.asarray([], dtype=float)
    observed_to_physical = np.sqrt(
        np.min(np.sum((observed[:, None, :] - physical[None, :, :]) ** 2, axis=2), axis=1)
    )
    observed_order = np.argsort(observed[:, 0])
    observed_sorted = observed[observed_order]
    within = physical[
        (physical[:, 0] >= observed_sorted[0, 0]) & (physical[:, 0] <= observed_sorted[-1, 0])
    ]
    if not len(within):
        return observed_to_physical
    observed_y = np.interp(within[:, 0], observed_sorted[:, 0], observed_sorted[:, 1])
    physical_to_observed = np.abs(within[:, 1] - observed_y)
    stride = max(1, len(physical_to_observed) // max(len(observed), 1))
    return np.concatenate([observed_to_physical, physical_to_observed[::stride]])


def project(P: np.ndarray, X) -> np.ndarray:
    X = np.asarray(X, float).reshape(-1, 3)
    Xh = np.hstack([X, np.ones((len(X), 1))])
    uvw = (P @ Xh.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def calibration_residuals(
    P: np.ndarray,
    H: np.ndarray,
    net_obs: list[tuple[float, float, float, float, float]] | None,
    support: str = NET_SUPPORT,
) -> tuple[float, float]:
    """Return ground-plane and observed net-top RMS reprojection residuals."""
    court_points = np.array(
        [[x, y] for x in np.linspace(0.0, COURT_W, 5) for y in np.linspace(0.0, COURT_L, 7)],
        dtype=np.float32,
    )
    expected_ground = cv2.perspectiveTransform(court_points[None, ...], H)[0]
    projected_ground = project(P, np.c_[court_points, np.zeros(len(court_points))])
    ground_rms = float(np.sqrt(np.mean(np.sum((expected_ground - projected_ground) ** 2, axis=1))))
    if not net_obs:
        return ground_rms, float("nan")
    errors = net_observation_residuals(P, net_obs, support=support)
    return ground_rms, float(np.sqrt(np.mean(np.square(errors))))


def net_reprojection_check(P: np.ndarray, H: np.ndarray) -> dict:
    """Consistency numbers: ground points must match H; net top must sit above the
    net's ground line by a plausible pixel height."""
    gpts = np.array([[2, 5, 0], [9, 20, 0], [5.5, NET_Y, 0]])
    via_P = project(P, gpts)
    via_H = cv2.perspectiveTransform(gpts[:, :2].reshape(1, -1, 2).astype(np.float32), H)[0]
    ground_err = float(np.abs(via_P - via_H).max())
    net_ground = project(
        P, [[NET_POST_X[0], NET_Y, 0], [COURT_W / 2, NET_Y, 0], [NET_POST_X[1], NET_Y, 0]]
    )
    net_top = project(
        P,
        [
            [NET_POST_X[0], NET_Y, NET_H_POST],
            [COURT_W / 2, NET_Y, NET_H_CENTER],
            [NET_POST_X[1], NET_Y, NET_H_POST],
        ],
    )
    net_px = (net_ground - net_top)[:, 1]  # image-vertical extent of the net
    return {"ground_err_px": ground_err, "net_px": net_px.round(1).tolist()}


def shared_vertical_column(direct: dict[int, np.ndarray], homographies: dict[int, np.ndarray]):
    """Estimate the projection's vertical column in a homography-normalized scale.

    Broadcast play footage uses one camera family. Direct net measurements pin the otherwise
    unobservable vertical column on a subset of points; their robust median supplies that column
    for points where the net-band detector fails, while each point retains its own exact ground
    homography.
    """
    columns = []
    for pt, P in direct.items():
        H = homographies[pt]
        ground = P[:, [0, 1, 3]]
        scale = float(np.sum(H * ground) / np.sum(ground * ground))
        columns.append(scale * P[:, 2] / H[2, 2])
    if not columns:
        raise ValueError("no direct projections from which to estimate vertical column")
    return np.median(np.stack(columns), axis=0)


def projection_from_ground(H: np.ndarray, vertical_column: np.ndarray) -> np.ndarray:
    Hn = H / H[2, 2]
    return np.column_stack([Hn[:, 0], Hn[:, 1], vertical_column, Hn[:, 2]])


def intrinsic_projection_from_ground(
    H: np.ndarray,
    w: int = W_IMG,
    h: int = H_IMG,
) -> tuple[np.ndarray, float] | None:
    """Preserve ``H`` exactly while borrowing only vertical scale from self-calibration."""
    solved = h_to_projection(H, w=w, h=h)
    if solved is None:
        return None
    raw_projection, _, _, _, focal = solved
    if not 400.0 <= focal <= 40000.0:
        return None
    ground = raw_projection[:, [0, 1, 3]]
    scale = float(np.sum(H * ground) / np.sum(ground * ground))
    vertical = scale * raw_projection[:, 2] / H[2, 2]
    projection = projection_from_ground(H, vertical)
    check = net_reprojection_check(projection, H)
    if check["ground_err_px"] > 0.1:
        return None
    if not all(3.0 <= height <= 200.0 for height in check["net_px"]):
        return None
    return projection, focal


def fixed_f_projection_from_ground(
    H: np.ndarray,
    focal: float,
    w: int = W_IMG,
    h: int = H_IMG,
) -> np.ndarray:
    """Preserve the ground homography while using a measured focal for height scale."""
    raw_projection = h_to_projection_fixed_f(H, focal, w=w, h=h)
    ground = raw_projection[:, [0, 1, 3]]
    scale = float(np.sum(H * ground) / np.sum(ground * ground))
    vertical = scale * raw_projection[:, 2] / H[2, 2]
    return projection_from_ground(H, vertical)


def draw_projection_audit(img: np.ndarray, P: np.ndarray, label: str) -> np.ndarray:
    """Draw physical-height references that make bad vertical projections easy to spot."""
    out = img.copy()
    net_top = project(
        P,
        [
            [NET_POST_X[0], NET_Y, NET_H_POST],
            [COURT_W / 2, NET_Y, NET_H_CENTER],
            [NET_POST_X[1], NET_Y, NET_H_POST],
        ],
    )
    net_ground = project(
        P,
        [[NET_POST_X[0], NET_Y, 0], [COURT_W / 2, NET_Y, 0], [NET_POST_X[1], NET_Y, 0]],
    )
    for top, ground in zip(net_top, net_ground):
        cv2.line(out, tuple(np.int32(top)), tuple(np.int32(ground)), (0, 0, 255), 2)
    for a, b in ((0, 1), (1, 2)):
        cv2.line(out, tuple(np.int32(net_top[a])), tuple(np.int32(net_top[b])), (0, 255, 255), 2)

    # Two-metre poles at the service-line centres exercise the vertical column away from the net.
    for y in (NET_Y - 6.40, NET_Y + 6.40):
        pole = project(P, [[COURT_W / 2, y, 0], [COURT_W / 2, y, 2]])
        cv2.line(out, tuple(np.int32(pole[0])), tuple(np.int32(pole[1])), (255, 0, 255), 2)
    cv2.rectangle(out, (0, 0), (300, 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def write_audit_montage(
    out_dir: str,
    frames_dir: str,
    projections: dict[int, np.ndarray],
    direct_pts: set[int],
    count: int,
    seed: int,
) -> str | None:
    available = []
    for pt in sorted(projections):
        frames = sorted(glob.glob(os.path.join(out_dir, frames_dir, f"pt{pt:04d}", "f_*.jpg")))
        if frames:
            available.append((pt, frames[len(frames) // 2]))
    if not available:
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(available), size=min(count, len(available)), replace=False)
    panels = []
    for idx in chosen:
        pt, frame_path = available[int(idx)]
        img = cv2.imread(frame_path)
        source = "direct" if pt in direct_pts else "fallback"
        audited = draw_projection_audit(img, projections[pt], f"pt{pt:04d} | {source}")
        panels.append(cv2.resize(audited, (480, 270), interpolation=cv2.INTER_AREA))
    cols = 3
    blank = np.zeros_like(panels[0])
    while len(panels) % cols:
        panels.append(blank)
    montage = np.vstack([np.hstack(panels[i : i + cols]) for i in range(0, len(panels), cols)])
    path = os.path.join(out_dir, f"camera_projection_audit_n{len(chosen)}_seed{seed}.jpg")
    cv2.imwrite(path, montage)
    return path


def direct_projection_from_net(
    homography: np.ndarray,
    observations: list | None,
    focal: float | None,
    *,
    w: int,
    h: int,
    support: str = NET_SUPPORT,
) -> tuple[np.ndarray | None, dict]:
    """Apply the ordinary direct-camera qualification to observed ground and net.

    ``homography`` maps court to image. This is the same qualification used by the
    camera producer; fallback projections never qualify an automatic anchor.
    """
    if observations is None or focal is None:
        return None, {"reliable": False, "reason": "missing_net_or_focal"}
    projection = fixed_f_projection_from_ground(homography, focal, w=w, h=h)
    ground_residual, net_residual = calibration_residuals(
        projection, homography, observations, support
    )
    check = net_reprojection_check(projection, homography)
    accepted = not (
        check["ground_err_px"] > 0.5
        or ground_residual > 0.5
        or not np.isfinite(net_residual)
        or net_residual > 8.0
        or not (3 <= min(check["net_px"]))
    )
    evidence = {
        "reliable": bool(accepted),
        "reason": "direct" if accepted else "direct_calibration_rejected",
        "ground_residual_px": float(ground_residual),
        "net_residual_px": float(net_residual) if np.isfinite(net_residual) else None,
        "ground_check_px": float(check["ground_err_px"]),
        "minimum_net_height_px": float(min(check["net_px"])),
        "focal_px": float(focal),
        "net_support": support,
    }
    return (projection if accepted else None), evidence


def measure_point_net(
    job: tuple[int, np.ndarray, str | None, int, int, str],
    *,
    tape_measurement: str = "hough_top",
) -> tuple[int, str | None, list | None, float | None, str]:
    point, homography, path, image_width, image_height, support = job
    if path is None:
        return point, None, None, None, "unavailable"
    image = cv2.imread(path)
    if image is None:
        return point, None, None, None, "unavailable"
    observations = measure_net_band(
        image, homography, support=support, tape_measurement=tape_measurement
    )
    if observations is None:
        return point, os.path.basename(path), None, None, "unavailable"
    focal = solve_f_from_net(
        homography,
        observations,
        w=image_width,
        h=image_height,
    )
    source = (
        "observed_connected_tape_pixels_top"
        if tape_measurement == "connected_pixels"
        else "observed_connected_tape_top_envelope"
    )
    if focal is None:
        legacy_observations = measure_net_band(
            image,
            homography,
            prefer_top_envelope=False,
            support=support,
            tape_measurement=tape_measurement,
        )
        legacy_focal = (
            solve_f_from_net(
                homography,
                legacy_observations,
                w=image_width,
                h=image_height,
                support=support,
            )
            if legacy_observations is not None
            else None
        )
        if legacy_focal is not None:
            observations = legacy_observations
            focal = legacy_focal
            source = "hough_net_centerline_fallback"
    return point, os.path.basename(path), observations, focal, source


IMAGE_SIZE_NATIVE = (1920, 1080)
ITF_LANDMARKS = {
    "near_left_doubles": (0.0, 0.0, 0.0),
    "near_right_doubles": (COURT_W, 0.0, 0.0),
    "far_left_doubles": (0.0, COURT_L, 0.0),
    "far_right_doubles": (COURT_W, COURT_L, 0.0),
    "near_left_service": (1.37, NET_Y - SERVICE_FROM_NET, 0.0),
    "near_right_service": (COURT_W - 1.37, NET_Y - SERVICE_FROM_NET, 0.0),
    "far_left_service": (1.37, NET_Y + SERVICE_FROM_NET, 0.0),
    "far_right_service": (COURT_W - 1.37, NET_Y + SERVICE_FROM_NET, 0.0),
    "net_center_top": (COURT_W / 2.0, NET_Y, NET_H_CENTER),
    "net_left_singles_stick_top": (NET_STICK_X[0], NET_Y, NET_H_POST),
    "net_right_singles_stick_top": (NET_STICK_X[1], NET_Y, NET_H_POST),
}


NET_TAPE_THICKNESS_M = 0.05  # published white band depth; its centre sits half that low
# Additive court/net targets that the twelve-frame landmark addendum clicks. Post
# tops/bases stay out: those clicks follow a padded silhouette, not certified bare
# post geometry, so they carry an unmodeled nuisance depth.
ADDENDUM_LANDMARKS = {
    "near_left_singles": (SINGLES_INSET, 0.0, 0.0),
    "near_right_singles": (COURT_W - SINGLES_INSET, 0.0, 0.0),
    "far_left_singles": (SINGLES_INSET, COURT_L, 0.0),
    "far_right_singles": (COURT_W - SINGLES_INSET, COURT_L, 0.0),
    "near_center_mark": (COURT_W / 2.0, 0.0, 0.0),
    "far_center_mark": (COURT_W / 2.0, COURT_L, 0.0),
    "near_service_t": (COURT_W / 2.0, NET_Y - SERVICE_FROM_NET, 0.0),
    "far_service_t": (COURT_W / 2.0, NET_Y + SERVICE_FROM_NET, 0.0),
    "net_band_center": (COURT_W / 2.0, NET_Y, NET_H_CENTER),
    "net_band_left_end": (NET_POST_X[0], NET_Y, NET_H_POST),
    "net_band_right_end": (NET_POST_X[1], NET_Y, NET_H_POST),
}
COURT_LANDMARKS = ITF_LANDMARKS | ADDENDUM_LANDMARKS
# Unit court-plane directions from each line's ITF measurement edge into its paint.
# ITF measures court dimensions to the OUTSIDE of the baselines, sidelines and
# service lines, so a paint-centre click sits half a line width inward along every
# line it lies on. The centre service line and the baseline centre marks are
# measured at their own centres and therefore contribute no offset.
LANDMARK_PAINT_NORMALS = {
    "near_left_doubles": (("baseline", (0.0, 1.0)), ("doubles_sideline", (1.0, 0.0))),
    "near_right_doubles": (("baseline", (0.0, 1.0)), ("doubles_sideline", (-1.0, 0.0))),
    "far_left_doubles": (("baseline", (0.0, -1.0)), ("doubles_sideline", (1.0, 0.0))),
    "far_right_doubles": (("baseline", (0.0, -1.0)), ("doubles_sideline", (-1.0, 0.0))),
    "near_left_singles": (("baseline", (0.0, 1.0)), ("singles_sideline", (1.0, 0.0))),
    "near_right_singles": (("baseline", (0.0, 1.0)), ("singles_sideline", (-1.0, 0.0))),
    "far_left_singles": (("baseline", (0.0, -1.0)), ("singles_sideline", (1.0, 0.0))),
    "far_right_singles": (("baseline", (0.0, -1.0)), ("singles_sideline", (-1.0, 0.0))),
    "near_center_mark": (("baseline", (0.0, 1.0)),),
    "far_center_mark": (("baseline", (0.0, -1.0)),),
    "near_left_service": (("service_line", (0.0, 1.0)), ("singles_sideline", (1.0, 0.0))),
    "near_right_service": (("service_line", (0.0, 1.0)), ("singles_sideline", (-1.0, 0.0))),
    "far_left_service": (("service_line", (0.0, -1.0)), ("singles_sideline", (1.0, 0.0))),
    "far_right_service": (("service_line", (0.0, -1.0)), ("singles_sideline", (-1.0, 0.0))),
    "near_service_t": (("service_line", (0.0, 1.0)),),
    "far_service_t": (("service_line", (0.0, -1.0)),),
}
LINE_CONVENTIONS = ("itf_outside_edge_v1", "painted_line_center_v1")


def landmark_world_convention(
    landmark: str,
    convention: str,
    *,
    edge_offset_m: float = 0.025,
    service_line_extra_m: float = 0.025,
) -> np.ndarray:
    """Place one clicked target under the line convention its own file declares.

    ``painted_line_center_v1`` clicks the paint centre axis, so every measured
    line contributes ``edge_offset_m`` along its inward normal. ``itf_outside_edge_v1``
    clicks the measurement edge and contributes nothing, except that the historical
    six-frame files were always solved with the service line at its paint centre;
    ``service_line_extra_m`` retains that exact prior geometry rather than silently
    re-defining the arm this comparison is measured against.
    """
    if landmark not in COURT_LANDMARKS:
        raise ValueError(f"unknown court landmark {landmark!r}")
    if convention not in LINE_CONVENTIONS:
        raise ValueError(f"unknown line convention {convention!r}")
    for value in (edge_offset_m, service_line_extra_m):
        if not np.isfinite(value) or not -0.05 <= value <= 0.10:
            raise ValueError("finite line offsets within +-0.05..0.10 m required")
    point = np.asarray(COURT_LANDMARKS[landmark], dtype=float).copy()
    if convention == "painted_line_center_v1":
        for _, normal in LANDMARK_PAINT_NORMALS.get(landmark, ()):
            point[:2] += edge_offset_m * np.asarray(normal, dtype=float)
        if landmark.startswith("net_band_"):
            point[2] -= NET_TAPE_THICKNESS_M / 2.0
    else:
        for line, normal in LANDMARK_PAINT_NORMALS.get(landmark, ()):
            if line == "service_line":
                point[:2] += service_line_extra_m * np.asarray(normal, dtype=float)
    return point


@dataclass(frozen=True)
class CameraObservation:
    """One independent image observation of a fixed court point."""

    frame: int
    landmark: str
    world: np.ndarray
    pixel: np.ndarray
    normal: np.ndarray | None = None
    residual_transform: np.ndarray | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class JointCameraSolution:
    """One fixed rig and the pan/tilt/focal state at each observed frame."""

    center: np.ndarray
    k1: float
    dist_center: np.ndarray
    view_parameters: dict[int, tuple[float, float, float]]
    projections: dict[int, np.ndarray]
    success: bool
    cost: float
    optimality: float
    evaluations: int
    observation_count: int
    loss: str = "soft_l1"


def landmark_world(landmark: str, service_line_width_m: float = 0.0) -> np.ndarray:
    """Return a fixed ITF point, retaining the labeler's service-edge rule."""
    if landmark not in ITF_LANDMARKS:
        raise ValueError(f"unknown court landmark {landmark!r}")
    if not np.isfinite(service_line_width_m) or not 0.0 <= service_line_width_m <= 0.05:
        raise ValueError("service line width must be finite and in [0, 0.05] m")
    point = np.asarray(ITF_LANDMARKS[landmark], dtype=float).copy()
    if landmark.startswith("near_") and landmark.endswith("_service"):
        point[1] += service_line_width_m / 2.0
    elif landmark.startswith("far_") and landmark.endswith("_service"):
        point[1] -= service_line_width_m / 2.0
    return point


def joint_projection_matrix(
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


def _project_distorted(
    projection: np.ndarray, k1: float, center: np.ndarray, world: np.ndarray
) -> np.ndarray:
    # Lazy import avoids camera_project's compatibility import of this module.
    from cv.pipeline.camera_project import project_distorted

    return project_distorted(projection, k1, center, world)


def _view_from_projection(projection: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite 3x4 initial projection required")
    intrinsic, rotation, center_h, *_ = cv2.decomposeProjectionMatrix(matrix)
    center = (center_h[:3] / center_h[3]).reshape(3)
    forward = rotation[2]
    yaw = float(math.atan2(forward[1], forward[0]))
    tilt = float(math.asin(np.clip(forward[2], -1.0, 1.0)))
    focal = float(intrinsic[0, 0] / intrinsic[2, 2])
    if not np.isfinite(center).all() or center[2] <= 0.0 or not 200.0 <= focal <= 80000.0:
        raise ValueError("initial projection has no usable physical decomposition")
    return center, yaw, tilt, focal


def _initial_joint_parameters(
    frames: list[int],
    initial_projections: dict[int, np.ndarray],
    initial_solution: JointCameraSolution | None,
) -> np.ndarray:
    if initial_solution is not None:
        if not set(frames) <= set(initial_solution.view_parameters):
            raise ValueError("warm-start solution does not cover every fitted frame")
        values = [*np.asarray(initial_solution.center, dtype=float), initial_solution.k1]
        for frame in frames:
            yaw, tilt, focal = initial_solution.view_parameters[frame]
            values.extend((yaw, tilt, math.log(focal)))
        return np.asarray(values, dtype=float)
    decomposed = {frame: _view_from_projection(initial_projections[frame]) for frame in frames}
    center = np.median(np.stack([row[0] for row in decomposed.values()]), axis=0)
    values = [*center, 0.0]
    for frame in frames:
        _, yaw, tilt, focal = decomposed[frame]
        values.extend((yaw, tilt, math.log(focal)))
    return np.asarray(values, dtype=float)


def fit_joint_camera(
    observations: Iterable[CameraObservation],
    initial_projections: dict[int, np.ndarray],
    *,
    image_size: tuple[int, int] = IMAGE_SIZE_NATIVE,
    initial_solution: JointCameraSolution | None = None,
    maximum_evaluations: int = 600,
    smoothness_weight: float = 0.02,
    fixed_k1: float | None = None,
) -> JointCameraSolution:
    """Fit one rigid court, optical centre/shared k1, and smooth frame views.

    Point observations contribute two residuals. Automatic line samples with a
    ``normal`` contribute only their measured point-to-line component.
    """
    rows = list(observations)
    if not rows:
        raise ValueError("at least one camera observation required")
    frames = sorted({int(row.frame) for row in rows})
    if set(frames) != set(initial_projections):
        raise ValueError("exactly one initial projection per observed frame required")
    if not 0.0 <= smoothness_weight <= 1.0:
        raise ValueError("smoothness weight must be in [0, 1]")
    if fixed_k1 is not None and not np.isfinite(fixed_k1):
        raise ValueError("fixed k1 must be finite")
    for row in rows:
        world = np.asarray(row.world, dtype=float)
        pixel = np.asarray(row.pixel, dtype=float)
        normal = None if row.normal is None else np.asarray(row.normal, dtype=float)
        transform = (
            None
            if row.residual_transform is None
            else np.asarray(row.residual_transform, dtype=float)
        )
        if (
            world.shape != (3,)
            or pixel.shape != (2,)
            or not np.isfinite(world).all()
            or not np.isfinite(pixel).all()
            or not np.isfinite(row.weight)
            or row.weight <= 0.0
            or (normal is not None and (normal.shape != (2,) or not np.isfinite(normal).all()))
            or (
                transform is not None
                and (transform.shape != (2, 2) or not np.isfinite(transform).all())
            )
        ):
            raise ValueError("finite world/pixel observation and positive weight required")

    width, height = image_size
    dist_center = np.asarray([width / 2.0, height / 2.0], dtype=float)
    radial_scale_sq = float(np.sum(np.square(dist_center)))
    initial = _initial_joint_parameters(frames, initial_projections, initial_solution)
    initial[3] *= radial_scale_sq
    lower = np.asarray(
        [-100.0, -1000.0, 0.5, -0.12] + [-np.inf, -1.50, math.log(400.0)] * len(frames),
        dtype=float,
    )
    upper = np.asarray(
        [100.0, 1000.0, 100.0, 0.12] + [np.inf, 0.50, math.log(40000.0)] * len(frames),
        dtype=float,
    )
    initial = np.maximum(lower + 1e-9, np.minimum(upper - 1e-9, initial))
    frame_index = {frame: index for index, frame in enumerate(frames)}

    def views(parameters: np.ndarray) -> dict[int, tuple[float, float, float]]:
        return {
            frame: (
                float(parameters[4 + 3 * index]),
                float(parameters[5 + 3 * index]),
                float(math.exp(parameters[6 + 3 * index])),
            )
            for frame, index in frame_index.items()
        }

    def residual(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:3]
        k1 = float(parameters[3] / radial_scale_sq) if fixed_k1 is None else float(fixed_k1)
        view = views(parameters)
        projections = {
            frame: joint_projection_matrix(center, *state, image_size)
            for frame, state in view.items()
        }
        values: list[float] = []
        for row in rows:
            predicted = _project_distorted(
                projections[int(row.frame)],
                k1,
                dist_center,
                np.asarray(row.world, dtype=float)[None, :],
            )[0]
            difference = (predicted - np.asarray(row.pixel, dtype=float)) * math.sqrt(row.weight)
            if row.normal is None:
                if row.residual_transform is not None:
                    difference = np.asarray(row.residual_transform, dtype=float) @ difference
                values.extend(difference.tolist())
            else:
                normal = np.asarray(row.normal, dtype=float)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                values.append(float(difference @ normal))
        values.append(float(parameters[3] / (1e-6 if fixed_k1 is not None else 0.20)))
        if smoothness_weight and len(frames) >= 3:
            states = [view[frame] for frame in frames]
            for index in range(1, len(frames) - 1):
                left_dt = max(frames[index] - frames[index - 1], 1)
                right_dt = max(frames[index + 1] - frames[index], 1)
                local_focal = states[index][2]
                for component, scale in enumerate((local_focal, local_focal, 0.5 * local_focal)):
                    left = (states[index][component] - states[index - 1][component]) / left_dt
                    right = (states[index + 1][component] - states[index][component]) / right_dt
                    values.append(float((right - left) * scale * smoothness_weight))
        return np.asarray(values, dtype=float)

    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=maximum_evaluations,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    fitted_views = views(solved.x)
    fitted_center = np.asarray(solved.x[:3], dtype=float)
    projections = {
        frame: joint_projection_matrix(fitted_center, *state, image_size)
        for frame, state in fitted_views.items()
    }
    k1 = float(solved.x[3] / radial_scale_sq) if fixed_k1 is None else float(fixed_k1)
    physically_valid = bool(
        solved.success
        and np.isfinite(solved.x).all()
        and fitted_center[2] > 0.0
        and all(np.isfinite(matrix).all() for matrix in projections.values())
    )
    return JointCameraSolution(
        center=fitted_center,
        k1=k1,
        dist_center=dist_center,
        view_parameters=fitted_views,
        projections=projections,
        success=physically_valid,
        cost=float(solved.cost),
        optimality=float(solved.optimality),
        evaluations=int(solved.nfev),
        observation_count=len(rows),
    )


def project_joint_camera(
    solution: JointCameraSolution, frame: int, world: np.ndarray
) -> np.ndarray:
    """Project world points through a fitted view and its shared radial model."""
    if frame not in solution.projections:
        raise ValueError(f"joint camera has no view for frame {frame}")
    return _project_distorted(
        solution.projections[frame],
        solution.k1,
        solution.dist_center,
        np.asarray(world, dtype=float),
    )


def backproject_plane(
    projection: np.ndarray,
    pixel: np.ndarray,
    plane_height_m: float,
    *,
    k1: float = 0.0,
    dist_center: np.ndarray | None = None,
) -> np.ndarray:
    """Intersect an imaged pixel ray with a horizontal world plane."""
    from cv.pipeline.camera_project import undistort_points

    matrix = np.asarray(projection, dtype=float)
    point = np.asarray(pixel, dtype=float).reshape(1, 2)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all() or not np.isfinite(point).all():
        raise ValueError("finite projection and pixel required")
    if not np.isfinite(plane_height_m):
        raise ValueError("finite plane height required")
    if dist_center is None:
        dist_center = np.zeros(2, dtype=float)
    undistorted = undistort_points(point, k1, dist_center)[0]
    plane = matrix[:, [0, 1, 3]].copy()
    plane[:, 2] += matrix[:, 2] * plane_height_m
    world_h = np.linalg.solve(plane, np.asarray([undistorted[0], undistorted[1], 1.0]))
    if abs(world_h[2]) <= 1e-12:
        raise ValueError("image ray is parallel to requested plane")
    xy = world_h[:2] / world_h[2]
    return np.asarray([xy[0], xy[1], plane_height_m], dtype=float)


def metric_error_for_pixel_residual(
    projection: np.ndarray,
    world: np.ndarray,
    pixel_residual: np.ndarray,
    *,
    plane_height_m: float,
    k1: float = 0.0,
    dist_center: np.ndarray | None = None,
) -> float:
    """Convert an observed image residual to horizontal metres at a chosen height."""
    if dist_center is None:
        dist_center = np.zeros(2, dtype=float)
    target = np.asarray(world, dtype=float).copy()
    target[2] = plane_height_m
    predicted = _project_distorted(projection, k1, dist_center, target[None, :])[0]
    reconstructed = backproject_plane(
        projection,
        predicted + np.asarray(pixel_residual, dtype=float),
        plane_height_m,
        k1=k1,
        dist_center=dist_center,
    )
    return float(np.linalg.norm(reconstructed[:2] - target[:2]))


def pixel_to_plane_residual_transform(
    projection: np.ndarray,
    pixel: np.ndarray,
    plane_height_m: float,
    *,
    k1: float = 0.0,
    dist_center: np.ndarray | None = None,
    reference_metres_per_pixel: float = 0.02,
) -> np.ndarray:
    """Linearize pixel residuals into court-plane metres for robust fitting."""
    if not np.isfinite(reference_metres_per_pixel) or reference_metres_per_pixel <= 0.0:
        raise ValueError("positive reference metre scale required")
    point = np.asarray(pixel, dtype=float)
    columns = []
    for delta in (np.asarray([0.5, 0.0]), np.asarray([0.0, 0.5])):
        plus = backproject_plane(
            projection, point + delta, plane_height_m, k1=k1, dist_center=dist_center
        )
        minus = backproject_plane(
            projection, point - delta, plane_height_m, k1=k1, dist_center=dist_center
        )
        columns.append(plus[:2] - minus[:2])
    return np.column_stack(columns) / reference_metres_per_pixel


def joint_camera_manifest(solution: JointCameraSolution) -> dict:
    """JSON-safe rig metadata for provenance-bearing camera artifacts."""
    return {
        "model": "joint_rigid_itf_fixed_center_pan_tilt_zoom_shared_k1_v1",
        "camera_center_m": solution.center.tolist(),
        "shared_k1_pixel_inverse_squared": solution.k1,
        "distortion_center_px": solution.dist_center.tolist(),
        "views": {
            str(frame): {"pan_rad": state[0], "tilt_rad": state[1], "focal_px": state[2]}
            for frame, state in sorted(solution.view_parameters.items())
        },
        "optimizer": {
            "success": solution.success,
            "cost": solution.cost,
            "optimality": solution.optimality,
            "evaluations": solution.evaluations,
            "observations": solution.observation_count,
            "loss": solution.loss,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default="rally_frames_25")
    ap.add_argument(
        "--rep-frames",
        default="",
        help="optional JSON {ptXXXX: frame_name} of known play frames (full_court.py); "
        "net measurement uses these instead of the arbitrary clip-middle frame",
    )
    ap.add_argument(
        "--allow-reviewed-reference-frames",
        action="store_true",
        help="required acknowledgement that --rep-frames makes this a diagnostic run",
    )
    ap.add_argument(
        "--net-support",
        choices=("singles_sticks", "doubles_posts"),
        default=NET_SUPPORT,
        help="what carries the tape at 1.07 m; singles sticks is the singles-match truth",
    )
    ap.add_argument(
        "--net-frame",
        choices=("court_anchor", "clip_middle"),
        default="court_anchor",
        help="frame the net is measured on; court_anchor matches the court homography",
    )
    ap.add_argument(
        "--tape-measurement",
        choices=TAPE_MEASUREMENTS,
        default="hough_top",
        help="optional native connected-pixel tape evidence; hough_top preserves the existing method",
    )
    ap.add_argument("--overlay", action="store_true", help="write a net-overlay debug frame")
    ap.add_argument("--audit", type=int, default=0, help="write a random projection-audit montage")
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.rep_frames and not args.allow_reviewed_reference_frames:
        raise ValueError(
            "--rep-frames is reviewed input; pass --allow-reviewed-reference-frames "
            "for an explicitly diagnostic run"
        )
    stage_run = StageRun(
        args.out,
        "camera_calibration",
        args,
        mode="diagnostic" if args.rep_frames else "automatic",
        reviewed_inputs=(
            [
                {
                    **file_record(os.path.join(args.out, args.rep_frames)),
                    "type": "reviewed_camera_reference_frames",
                }
            ]
            if args.rep_frames
            else []
        ),
    )
    rep_frames = {}
    if args.rep_frames:
        import json as _json

        with open(os.path.join(args.out, args.rep_frames)) as fh:
            rep_frames = _json.load(fh)

    # The court homography comes from one anchor frame; measuring the net on a different
    # frame mixes two camera poses into one calibration, so prefer the anchor.
    anchor_frames: dict[str, str] = {}
    evidence_path = os.path.join(args.out, "court_topology_evidence_v1.json")
    if args.net_frame == "court_anchor" and os.path.exists(evidence_path):
        with open(evidence_path) as handle:
            for row in json.load(handle).get("rows", []):
                if row.get("status") == "direct" and row.get("frame"):
                    anchor_frames[f"pt{int(row['point']):04d}"] = str(row["frame"])

    def clip_frame(pt: int):
        clip = os.path.join(args.out, args.frames_dir, f"pt{pt:04d}")
        rep = rep_frames.get(f"pt{pt:04d}")
        if rep:
            path = os.path.join(clip, rep)
            if os.path.exists(path):
                return path
        anchor = anchor_frames.get(f"pt{pt:04d}")
        if anchor:
            path = os.path.join(clip, anchor)
            if os.path.exists(path):
                return path
        fps_ = sorted(glob.glob(os.path.join(clip, "f_*.jpg")))
        return fps_[len(fps_) // 2] if fps_ else None

    d = np.load(os.path.join(args.out, "court_H_per_point.npz"))
    # stored H maps IMAGE -> COURT (see court.py homography()); the plane-calibration
    # convention needs COURT -> IMAGE, so invert once here.
    finite = [
        (int(point), np.linalg.inv(homography))
        for point, homography in zip(d["pts"], d["H"], strict=True)
        if np.isfinite(homography).all()
    ]
    if not finite:
        np.savez_compressed(
            os.path.join(args.out, "camera_P_per_point.npz"),
            pts=np.asarray([], dtype=np.int32),
            P=np.empty((0, 3, 4), dtype=float),
            source=np.asarray([], dtype=str),
            reliable=np.asarray([], dtype=bool),
            ground_residual_px=np.asarray([], dtype=float),
            net_residual_px=np.asarray([], dtype=float),
            confidence=np.asarray([], dtype=float),
            fallback_ancestry=np.asarray([], dtype=str),
            frame_scope=np.asarray([], dtype=str),
            reference_frame=np.asarray([], dtype=str),
            net_cord_xy=np.empty((0, 9, 2), dtype=float),
            net_cord_valid=np.asarray([], dtype=bool),
            net_cord_source=np.asarray([], dtype=str),
        )
        stage_run.finish(outputs={"direct": 0, "total": 0, "held_missing_court": len(d["pts"])})
        print("camera calibration held every point: no automatic court homographies")
        return 0
    pts = np.asarray([point for point, _ in finite], dtype=np.int32)
    Hs = np.stack([homography for _, homography in finite])
    image_width, image_height = W_IMG, H_IMG
    for point in pts.tolist():
        sample_path = clip_frame(point)
        sample = cv2.imread(sample_path) if sample_path is not None else None
        if sample is not None:
            image_height, image_width = sample.shape[:2]
            break
    # Stage 1: measure the net in a middle frame of each clip; solve f per point
    fs = []
    focal_by_point = {}
    focal_source_points = []
    net_observations = {}
    net_observation_sources = {}
    reference_frames = {}
    jobs = [
        (pt, H, clip_frame(pt), image_width, image_height, args.net_support)
        for pt, H in zip(pts.tolist(), Hs, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        measurements = list(
            pool.map(partial(measure_point_net, tape_measurement=args.tape_measurement), jobs)
        )
    for pt, reference_frame, obs, f, observation_source in measurements:
        if obs is None:
            continue
        net_observations[pt] = obs
        net_observation_sources[pt] = observation_source
        reference_frames[pt] = reference_frame
        if f is not None:
            fs.append(f)
            focal_source_points.append(pt)
            focal_by_point[pt] = f
    if fs:
        f_med = float(np.median(fs))
        print(
            f"net-based focal: median {f_med:.0f}px over {len(fs)} points "
            f"(IQR [{np.percentile(fs, 25):.0f},{np.percentile(fs, 75):.0f}])"
        )
    else:
        print("no net-based focal solutions; trying intrinsic ground-preserving fallback")
    # Stage 2: retain only ground-preserving physical cameras whose observed net tape
    # independently reprojects. A free DLT can fit the tape only by moving known court
    # geometry, which hides bad net detections and corrupts every downstream 3D anchor.
    Ps = {}
    for pt, H in zip(pts.tolist(), Hs):
        obs = net_observations.get(pt)
        if obs is None:
            continue
        focal = focal_by_point.get(pt)
        if focal is None:
            continue
        P, _evidence = direct_projection_from_net(
            H, obs, focal, w=image_width, h=image_height, support=args.net_support
        )
        if P is None:
            continue
        Ps[pt] = P
    direct_pts = set(Ps)
    direct_count = len(direct_pts)
    homographies = dict(zip(pts.tolist(), Hs))
    sources = {pt: "direct" for pt in direct_pts}
    ancestry = {pt: [] for pt in direct_pts}
    if direct_pts:
        vertical_column = shared_vertical_column(Ps, homographies)
        for pt, H in homographies.items():
            if pt not in Ps:
                Ps[pt] = projection_from_ground(H, vertical_column)
                sources[pt] = "shared_net_fallback"
                ancestry[pt] = [
                    "shared_vertical_column",
                    *[f"pt{source_pt:04d}" for source_pt in sorted(direct_pts)],
                ]
    elif fs:
        for pt, H in homographies.items():
            Ps[pt] = fixed_f_projection_from_ground(
                H,
                f_med,
                w=image_width,
                h=image_height,
            )
            sources[pt] = "shared_net_focal_fallback"
            ancestry[pt] = [
                "median_net_focal",
                *[f"pt{source_pt:04d}" for source_pt in sorted(focal_source_points)],
            ]
        print(
            f"net-focal fallback: {len(Ps)}/{len(homographies)} projections, "
            f"median focal {f_med:.0f}px"
        )
    else:
        focal_values = []
        for pt, H in homographies.items():
            solved = intrinsic_projection_from_ground(
                H,
                w=image_width,
                h=image_height,
            )
            if solved is None:
                continue
            Ps[pt], focal = solved
            focal_values.append(focal)
            sources[pt] = "intrinsic_fallback"
            ancestry[pt] = ["ground_homography_intrinsic_solve"]
        if not Ps:
            raise SystemExit("camera calibration failed: no net or intrinsic projection")
        vertical_column = shared_vertical_column(Ps, homographies)
        for pt, H in homographies.items():
            if pt not in Ps:
                Ps[pt] = projection_from_ground(H, vertical_column)
                sources[pt] = "shared_intrinsic_fallback"
                ancestry[pt] = [
                    "shared_intrinsic_vertical_column",
                    *[
                        f"pt{source_pt:04d}"
                        for source_pt in sorted(Ps)
                        if sources.get(source_pt) == "intrinsic_fallback"
                    ],
                ]
        print(
            "intrinsic fallback: "
            f"{len(Ps)}/{len(homographies)} projections, "
            f"median focal {np.median(focal_values):.0f}px"
        )
    print(
        f"projections: {direct_count} direct net measurements + "
        f"{len(Ps) - direct_count} fallbacks = {len(Ps)}/{len(pts)}"
    )
    ordered = sorted(Ps)
    residuals = {
        point: calibration_residuals(
            Ps[point],
            homographies[point],
            net_observations.get(point),
            args.net_support,
        )
        for point in ordered
    }
    reliable = {
        point: (
            sources[point] == "direct"
            and residuals[point][0] <= 4.0
            and np.isfinite(residuals[point][1])
            and residuals[point][1] <= 8.0
        )
        for point in ordered
    }
    confidence = {
        point: (
            float(np.exp(-residuals[point][0] / 4.0 - residuals[point][1] / 8.0))
            if reliable[point]
            else 0.25
            if "fallback" in sources[point]
            else 0.0
        )
        for point in ordered
    }
    net_cord_xy = np.full((len(ordered), 9, 2), np.nan, dtype=float)
    net_cord_valid = np.zeros(len(ordered), dtype=bool)
    for index, point in enumerate(ordered):
        observations = net_observations.get(point)
        if not observations:
            continue
        observed = np.asarray([[row[2], row[4]] for row in observations], dtype=float)
        if observed.shape != (9, 2) or not np.isfinite(observed).all():
            continue
        net_cord_xy[index] = observed
        net_cord_valid[index] = net_source_is_tape_top(
            net_observation_sources.get(point, "unavailable")
        )
    np.savez_compressed(
        os.path.join(args.out, "camera_P_per_point.npz"),
        pts=np.array(ordered),
        P=np.stack([Ps[p] for p in ordered]),
        source=np.array([sources[p] for p in ordered]),
        direct_pts=np.array(sorted(direct_pts)),
        reliable=np.array([reliable[p] for p in ordered], dtype=bool),
        ground_residual_px=np.array([residuals[p][0] for p in ordered], dtype=float),
        net_residual_px=np.array([residuals[p][1] for p in ordered], dtype=float),
        confidence=np.array([confidence[p] for p in ordered], dtype=float),
        fallback_ancestry=np.array([json.dumps(ancestry[p]) for p in ordered]),
        frame_scope=np.repeat("point_static", len(ordered)),
        reference_frame=np.array([reference_frames.get(p, "") for p in ordered]),
        net_cord_xy=net_cord_xy,
        net_cord_valid=net_cord_valid,
        net_cord_source=np.array(
            [net_observation_sources.get(point, "unavailable") for point in ordered]
        ),
    )
    if args.overlay and Ps:
        mid = sorted(Ps)[len(Ps) // 2]
        clip = os.path.join(args.out, args.frames_dir, f"pt{mid:04d}")
        fps_ = sorted(glob.glob(os.path.join(clip, "f_*.jpg")))
        if fps_:
            img = cv2.imread(fps_[len(fps_) // 2])
            P = Ps[mid]
            # draw net band + a 2m-high pole at each service-T for scale
            seg = project(
                P,
                [
                    [NET_POST_X[0], NET_Y, NET_H_POST],
                    [COURT_W / 2, NET_Y, NET_H_CENTER],
                    [NET_POST_X[1], NET_Y, NET_H_POST],
                ],
            )
            gnd = project(
                P, [[NET_POST_X[0], NET_Y, 0], [COURT_W / 2, NET_Y, 0], [NET_POST_X[1], NET_Y, 0]]
            )
            for (x1, y1), (x2, y2) in zip(seg, gnd):
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            for a, b in ((0, 1), (1, 2)):
                cv2.line(img, tuple(np.int32(seg[a])), tuple(np.int32(seg[b])), (0, 255, 255), 2)
            out = os.path.join(args.out, f"camera_net_check_pt{mid:04d}.jpg")
            cv2.imwrite(out, img)
            print("overlay ->", out)
    audit_path = None
    if args.audit:
        audit_path = write_audit_montage(
            args.out, args.frames_dir, Ps, direct_pts, args.audit, args.seed
        )
        print("audit ->", audit_path)
    stage_run.fallbacks = [
        {
            "stage": "camera_calibration",
            "point": f"pt{point:04d}",
            "type": sources[point],
            "ancestry": ancestry[point],
        }
        for point in ordered
        if sources[point] != "direct"
    ]
    stage_run.finish(
        outputs={
            "projections": len(Ps),
            "direct_projections": direct_count,
            "fallback_projections": len(Ps) - direct_count,
            "requested_points": len(pts),
            "audit_samples": min(args.audit, len(Ps)),
            "audit_path": audit_path,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
