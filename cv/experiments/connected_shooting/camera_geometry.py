"""Native image projection for explicit pinhole or radial-camera observations.

Radial rows are [k1 in native px^-2, center_x, center_y]. Undistortion is
initialization-only; optimization and scoring always use original image pixels.
"""

from __future__ import annotations

import numpy as np


def radial_row(record: dict) -> np.ndarray | None:
    """Read one explicit native lens row, refusing incomplete or conflicting metadata."""
    supplied = []
    if "k1" in record or "dist_center" in record:
        if "k1" not in record or np.shape(record.get("dist_center")) != (2,):
            raise ValueError("radial camera requires k1 and a two-coordinate native centre")
        supplied.append(np.r_[float(record["k1"]), np.asarray(record["dist_center"], float)])
    if record.get("camera_distortion") is not None:
        supplied.append(np.asarray(record["camera_distortion"], float))
    if any(record.get(name) is not None for name in ("distortion", "radial_distortion")):
        raise ValueError("ambiguous radial metadata; use explicit native k1/centre or row")
    if not supplied:
        return None
    if any(row.shape != (3,) or not np.isfinite(row).all() for row in supplied):
        raise ValueError("one finite native radial-camera row required")
    if any(not np.array_equal(supplied[0], row) for row in supplied[1:]):
        raise ValueError("conflicting native radial-camera metadata")
    return supplied[0].copy()


def rows_radial(records: list[dict]) -> np.ndarray | None:
    """Keep per-exposure lens identity; do not silently fill missing radial rows."""
    rows = [radial_row(row) for row in records]
    if all(row is None for row in rows):
        return None
    if any(row is None for row in rows):
        raise ValueError("radial observation metadata must be present on every row")
    return np.asarray(rows, float)


def radial_fields(record: dict) -> dict:
    """Copy calibrated lens identity onto an observation without changing its pixels."""
    row = radial_row(record)
    return {} if row is None else {"camera_distortion": row.tolist()}


def camera_radial_map(document: dict) -> dict[int, np.ndarray] | None:
    """Bind supported native camera epochs to their declared lenses."""
    records = [r for r in document["cameras"] if "P" in r and r.get("supported", True)]
    rows = rows_radial(records)
    if rows is None:
        return None
    if any(not np.isfinite(r["frame"]) or float(r["frame"]) != int(r["frame"]) for r in records):
        raise ValueError("integer native camera epochs required")
    if len({r["frame"] for r in records}) != len(records):
        raise ValueError("unique native camera epochs required")
    return {int(r["frame"]): lens for r, lens in zip(records, rows, strict=True)}


def scene_radial_map(*scenes) -> dict[int, np.ndarray] | None:
    """Combine scene splits without losing or replacing an exposure's lens identity."""
    if all(s.camera_distortion is None for s in scenes):
        return None
    if any(s.camera_distortion is None for s in scenes):
        raise ValueError("radial metadata differs between scene splits")
    result = {}
    for scene in scenes:
        scene.validate()
        for frames, lenses in zip(scene.observation_frames, scene.camera_distortion, strict=True):
            for frame, lens in zip(frames, lenses, strict=True):
                if frame in result and not np.array_equal(result[frame], lens):
                    raise ValueError("conflicting radial metadata at native exposure")
                result[int(frame)] = lens
    return result


def radial_at_epoch(
    lenses: dict | None, epoch: float, camera_frames: list[int]
) -> np.ndarray | None:
    """Use the same native camera interpolation epochs for the declared lens."""
    if lenses is None:
        return None
    if any(frame not in lenses for frame in camera_frames):
        raise ValueError("camera interpolation lacks bound native radial metadata")
    if len(camera_frames) == 1:
        return np.asarray(lenses[camera_frames[0]], float)
    a, b = camera_frames
    weight = (epoch - a) / (b - a)
    return (1 - weight) * lenses[a] + weight * lenses[b]


def distortion_jacobian(pixels: np.ndarray, radial: np.ndarray | None) -> np.ndarray:
    """Native radial derivative with respect to ideal pinhole image coordinates."""
    pixels = np.asarray(pixels, float)
    result = np.broadcast_to(np.eye(2), (len(pixels), 2, 2)).copy()
    if radial is None:
        return result
    distort(pixels, radial)
    delta = pixels - radial[:, 1:]
    k = radial[:, 0]
    return result * (1 + k * np.sum(delta**2, axis=1))[:, None, None] + (
        2 * k[:, None, None] * delta[:, :, None] * delta[:, None, :]
    )


def distort(pixels: np.ndarray, radial: np.ndarray | None) -> np.ndarray:
    if radial is None:
        return pixels
    radial = np.asarray(radial, float)
    if radial.shape != (len(pixels), 3) or not np.isfinite(radial).all():
        raise ValueError("one finite native radial-camera row per observation required")
    if not np.any(radial[:, 0]):
        return pixels
    delta = pixels - radial[:, 1:]
    radius_sq = np.sum(delta**2, axis=1, keepdims=True)
    if np.any(1 + 3 * radial[:, :1] * radius_sq <= 0):
        raise ValueError("noninvertible radial-camera branch")
    value = radial[:, 1:] + delta * (1 + radial[:, :1] * radius_sq)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite radial projection")
    return value


def undistort(pixels: np.ndarray, radial: np.ndarray | None) -> np.ndarray:
    if radial is None:
        return pixels
    distort(pixels, radial)  # validate the complete row contract
    if not np.any(radial[:, 0]):
        return pixels
    delta = pixels - radial[:, 1:]
    estimate = delta.copy()
    for _ in range(24):
        radius_sq = np.sum(estimate**2, axis=1, keepdims=True)
        factor = 1 + radial[:, :1] * radius_sq
        if np.any(factor <= 0) or np.any(1 + 3 * radial[:, :1] * radius_sq <= 0):
            raise ValueError("noninvertible radial-camera branch")
        estimate = delta / factor
    result = radial[:, 1:] + estimate
    if np.max(np.abs(distort(result, radial) - pixels)) > 1e-6:
        raise ValueError("radial inversion did not reproduce native observations")
    return result


def project(cameras: np.ndarray, xyz: np.ndarray, radial: np.ndarray | None = None) -> np.ndarray:
    uvw = np.einsum("nij,nj->ni", cameras, np.c_[xyz, np.ones(len(xyz))])
    if not np.isfinite(uvw).all() or np.any(np.abs(uvw[:, 2]) < 1e-9):
        raise ValueError("camera projection at infinity or nonfinite")
    return distort(uvw[:, :2] / uvw[:, 2:], radial)
