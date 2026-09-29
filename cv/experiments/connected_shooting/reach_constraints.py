"""Explicit experimental contact-reach conditioning from native player root rays.

No anatomical calibration or hitter identity is claimed. Both sides and every
neighboring native exposure are required; missing/ambiguous inputs hold the arm.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import contact_geometry, contact_reach


@dataclass(frozen=True)
class RootExposure:
    frame: int
    # near/far, lower/upper assumed height, XYZ
    roots: np.ndarray


@dataclass(frozen=True)
class ReachConstraints:
    contact_frames: np.ndarray
    exposures: tuple[tuple[RootExposure, ...], ...]
    fps: float
    height_cap_m: float
    radius_m: float
    motion_allowance_mps: float
    residual_scale_m: float

    def validate(self, scene) -> None:
        if (
            not np.array_equal(self.contact_frames, scene.contact_frames[:-1])
            or len(self.exposures) != len(self.contact_frames)
            or self.fps != scene.fps
            or not np.isfinite(
                [self.height_cap_m, self.radius_m, self.motion_allowance_mps, self.residual_scale_m]
            ).all()
            or self.height_cap_m < 0
            or self.radius_m <= 0
            or self.motion_allowance_mps < 0
            or self.residual_scale_m <= 0
        ):
            raise ValueError("explicit finite matching reach configuration required")
        for frame, group in zip(self.contact_frames, self.exposures, strict=True):
            if [e.frame for e in group] != contact_geometry.neighboring_exposures(frame):
                raise ValueError("all neighboring native player exposures required")
            for exposure in group:
                roots = np.asarray(exposure.roots)
                if (
                    roots.shape != (2, 2, 3)
                    or not np.isfinite(roots).all()
                    or np.any(roots[:, 0, 2] != 0)
                    or np.any(roots[:, 1, 2] != self.height_cap_m)
                ):
                    raise ValueError("both finite near/far root-ray endpoints required")
                for lower, upper in roots:
                    contact_reach.minimum_reach(lower, lower, upper)

    def evaluate(self, flights: list[dict]) -> dict:
        if len(flights) != len(self.exposures):
            raise ValueError("one reach group per physical contact required")
        rows, residuals = [], []
        for frame, group, flight in zip(self.contact_frames, self.exposures, flights, strict=True):
            if flight["start_frame"] != frame:
                raise ValueError("reach evidence and physical contact time differ")
            for exposure in group:
                minima = [
                    contact_reach.minimum_reach(flight["start_xyz"], lower, upper)
                    for lower, upper in exposure.roots
                ]
                allowance = abs(exposure.frame - frame) / self.fps * self.motion_allowance_mps
                excess = max(
                    min(m["minimum_root_to_contact_distance_m"] for m in minima)
                    - self.radius_m
                    - allowance,
                    0.0,
                )
                residuals.append(excess / self.residual_scale_m)
                rows.append(
                    {
                        "contact_frame": float(frame),
                        "native_frame": exposure.frame,
                        "sides": dict(zip(("near", "far"), minima, strict=True)),
                        "time_offset_allowance_m": float(allowance),
                        "reach_excess_m": float(excess),
                    }
                )
        return {
            "schema": "experimental_contact_reach_conditioning_v1",
            "height_cap_m": self.height_cap_m,
            "radius_m": self.radius_m,
            "motion_allowance_mps": self.motion_allowance_mps,
            "residual_scale_m": self.residual_scale_m,
            "residual_loss": "quadratic_not_robustified",
            "residuals": residuals,
            "exposures": rows,
            "scope": "Minimum over either player and assumed airborne heights, independently per exposure. No hitter/pose estimate or calibrated error budget; not a quality gate.",
        }


def load(
    scene,
    clip: str,
    player_path: Path,
    camera_path: Path,
    *,
    height_cap_m: float,
    radius_m: float,
    motion_allowance_mps: float,
    residual_scale_m: float,
) -> ReachConstraints:
    """Read explicit native root observations only; never a frozen fitted contact."""
    metadata = json.loads(Path(str(player_path) + ".coordinates.json").read_text())
    if metadata.get("fps") != scene.fps or any(
        metadata[k] != {"width": 1920, "height": 1080} for k in ("image_size", "artifact_size")
    ):
        raise ValueError("native player coordinates and matching source cadence required")
    players = {}
    with player_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] == clip:
                players.setdefault((int(row["frame"]), row["side"]), []).append(row)
    cameras = {}
    with np.load(camera_path, allow_pickle=False) as source:
        if "k2" in source and np.any(source["k2"] != 0):
            raise ValueError("unsupported nonzero second radial coefficient")
        for i, (name, frame) in enumerate(zip(source["clips"], source["frames"], strict=True)):
            if str(name) != clip:
                continue
            if not np.isfinite(frame) or frame != int(frame) or int(frame) in cameras:
                raise ValueError("unique native camera exposure required")
            cameras[int(frame)] = (
                source["P"][i].copy(),
                np.r_[source["k1"][i], source["dist_center"][i]],
                bool(source["reliable"][i]),
            )
    groups = []
    for contact_frame in scene.contact_frames[:-1]:
        group = []
        for frame in contact_geometry.neighboring_exposures(contact_frame):
            if frame not in cameras or not cameras[frame][2]:
                raise ValueError("missing or held reach camera")
            P, radial, _ = cameras[frame]
            roots = []
            for side in ("near", "far"):
                rows = players.get((frame, side), [])
                if len(rows) != 1:
                    raise ValueError("missing or ambiguous native player reach evidence")
                pixel = np.array([float(rows[0][k]) for k in ("root_x_native", "root_y_native")])
                endpoints = np.array(
                    [
                        np.r_[contact_geometry.plane_proxy(P, radial, pixel, h), h]
                        for h in (0.0, height_cap_m)
                    ]
                )
                depths = P[2] @ np.c_[endpoints, np.ones(2)].T
                if (
                    not np.isfinite(depths).all()
                    or np.any(np.abs(depths) < 1e-9)
                    or depths[0] * depths[1] <= 0
                ):
                    raise ValueError("root-height interval crosses camera projection singularity")
                roots.append(endpoints)
            group.append(RootExposure(frame, np.asarray(roots)))
        groups.append(tuple(group))
    result = ReachConstraints(
        scene.contact_frames[:-1].copy(),
        tuple(groups),
        scene.fps,
        height_cap_m,
        radius_m,
        motion_allowance_mps,
        residual_scale_m,
    )
    result.validate(scene)
    return result
