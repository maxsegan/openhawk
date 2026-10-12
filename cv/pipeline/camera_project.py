"""Distortion-aware forward projection helper for camera solve v3.

WHY: v1/v2 store a pure pinhole 3x4 projection ``P`` per frame and consumers project a
court/3D point with ``P @ [X;1]`` (see ``camera_cal.project``). The broadcast lens has
RADIAL (barrel) distortion the pinhole model cannot represent: the projection is near
perfect at image centre and systematically off at the periphery (baselines, corners).
Distortion-aware artifacts keep the pinhole ``P`` AND store a per-frame radial term
``k1`` about a distortion centre; the true image pixel is the pinhole projection with
the radial term applied on top. Every such consumer MUST project through this helper,
never ``P`` alone. Pixels may be native or downscaled, but ``P``, ``k1`` and ``centre``
must all use the same coordinate space.

PROJECTION CONVENTION (v3):
    court/3D point X (metres)
      --> pinhole:   p_u = project(P, X)            # artifact space, undistorted
      --> distort:   p_d = distort(p_u, k1, centre) # artifact space, imaged pixel
      --> (optional) * coord_scale                  # scale into a served frame

Distortion model (polynomial / Brown-Conrady radial, forward-direct in pixel space so
projection needs no root-solve; ``k2`` optional, default 0):
    d  = p_u - centre
    r2 = |d|^2
    p_d = centre + d * (1 + k1*r2 + k2*r2^2)
``centre`` is a 540-space pixel (image centre by default). Barrel vs pincushion is just
the sign of ``k1`` and is fit per frame; nothing here assumes a sign.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_cal import project  # noqa: E402


def distort_points(p_u: np.ndarray, k1: float, center, k2: float = 0.0) -> np.ndarray:
    """Apply forward radial distortion to undistorted pixels ``p_u`` (N x 2)."""
    p_u = np.asarray(p_u, float).reshape(-1, 2)
    c = np.asarray(center, float).reshape(1, 2)
    d = p_u - c
    r2 = (d**2).sum(axis=1, keepdims=True)
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    return c + d * factor


def undistort_points(
    p_d: np.ndarray, k1: float, center, k2: float = 0.0, iters: int = 8
) -> np.ndarray:
    """Invert :func:`distort_points` (distorted pixels -> undistorted) by
    fixed-point iteration. For consumers that need image->court; forward projection
    (court->image) never needs this."""
    p_d = np.asarray(p_d, float).reshape(-1, 2)
    c = np.asarray(center, float).reshape(1, 2)
    target = p_d - c
    d = target.copy()
    for _ in range(iters):
        r2 = (d**2).sum(axis=1, keepdims=True)
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        d = target / factor
    return c + d


def project_distorted(
    P: np.ndarray, k1: float, center, X, k2: float = 0.0, coord_scale: float = 1.0
) -> np.ndarray:
    """Project court/3D points ``X`` (Nx3, metres) to imaged pixels through the pinhole
    ``P`` then radial distortion, with an optional final coordinate scale.

    This is THE v3 forward projection; it reduces exactly to ``project(P, X)`` when
    ``k1 == k2 == 0`` (i.e. v1/v2 pinhole behaviour)."""
    p_u = project(P, X)
    p_d = distort_points(p_u, k1, center, k2)
    return p_d * coord_scale


def image_center(width: int = 960, height: int = 540) -> np.ndarray:
    """Return the distortion centre for the requested artifact dimensions."""
    return np.array([width / 2.0, height / 2.0], float)
