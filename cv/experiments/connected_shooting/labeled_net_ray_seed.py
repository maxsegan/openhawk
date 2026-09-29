"""Initialize a terminal net approach from original fronts on the net plane.

This supplies an input-derived seed when the source approaches below ground.
The subsequent free-response fit imposes no net-height band. Plane intersections
near an uncertain impact are conditional initialization evidence, not XYZ truth.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import labeled_net_height_chart as chart
from cv.experiments.connected_shooting import model, camera_geometry
from cv.pipeline.physics_knot_solver import net_tape_height

NET_Y_M = 11.885


class UnsupportedSeed(ValueError):
    """Original net bracket has no usable native front for this initializer."""


def seed(context: dict, parameters, labels: dict, cameras: dict, event: dict):
    scene = context["scene"]
    n = len(scene.pixels)
    clip = context["attempt"]["point_clip"]
    camera_rows = {int(r["frame"]): r for r in cameras["cameras"]}
    low, high = map(float, event["frame_interval"])
    rays = []
    for group in labels["ball"]["records"]:
        if group["clip"] != clip:
            continue
        for row in group["frames"]:
            frame = float(row["frame"])
            if row["status"] != "visible" or not low <= frame <= high:
                continue
            camera = camera_rows.get(int(frame), {})
            if camera.get("status") != "supported" or camera.get("supported") is False:
                continue
            P = np.asarray(camera.get("P"), float)
            pixel = np.asarray([row["x1080"], row["y1080"]], float)
            if P.shape != (3, 4) or not np.isfinite(P).all() or not np.isfinite(pixel).all():
                raise ValueError("finite original net front and pinhole camera required")
            radial = camera_geometry.radial_row(camera)
            ideal = camera_geometry.undistort(
                pixel[None], None if radial is None else radial[None]
            )[0]
            equations = P[:2] - ideal[:, None] * P[2]
            xz = np.linalg.solve(equations[:, [0, 2]], -equations[:, 1] * NET_Y_M - equations[:, 3])
            if not np.isfinite(xz).all():
                raise ValueError("finite native ray/net-plane intersection required")
            rays.append(
                dict(frame=frame, original_front=pixel.tolist(), xyz_m=[xz[0], NET_Y_M, xz[1]])
            )
    if not rays:
        raise UnsupportedSeed(
            "no supported original visible front inside the supplied net interval"
        )
    position = np.mean([r["xyz_m"] for r in rays], axis=0)
    source = np.array(parameters, float)
    chain = model.chain(scene, source)
    velocity = slice(3 + 3 * (n - 1), 6 + 3 * (n - 1))
    spin = slice(3 + 3 * n + 3 * (n - 1), 6 + 3 * n + 3 * (n - 1))
    theta = np.r_[chain[-1]["start_xyz"], source[velocity], source[spin]]
    projected, projection = chart.project_net_height(
        theta,
        scene.contact_frames[-2],
        event["frame"],
        scene.fps,
        scene.surface,
        height_offset_m=float(position[2] - net_tape_height(float(position[0]))),
        rebound_scales=source[-2:],
        bounce_profile=scene.bounce_profile,
    )
    indices = [velocity.start + 1, velocity.start + 2]
    source[indices] = projected[4:6]
    return source, dict(
        policy="native_net_plane_initialization_only",
        original_native_rays=rays,
        conditional_mean_xyz_m=position.tolist(),
        changed_parameter_indices=indices,
        projection=projection,
        height_constraint_in_final_fit=False,
        interpretation="Near-impact ray intersections condition on net-plane depth; seed only, not independent truth.",
    )
