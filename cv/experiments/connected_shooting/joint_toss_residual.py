"""Original native toss evidence joined to an explicit shared serve contact.

Experimental feasibility residual only. The existing TossConfig checks and
native camera/front uncertainties are preserved; no inferred contact XYZ target.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, toss_witness


def evaluate(contact, parameters, observations, feet, fps):
    """Return image/physical residuals, seven original toss inequalities and receipt."""
    config = toss_witness.TossConfig()
    rows = observations["rows"]
    if observations.get("status") != "supported" or len(rows) < 4:
        raise ValueError("joint toss arm requires supported original precontact pictures")
    frame = float(observations["contact_frame"])
    times = (np.asarray([r["frame"] for r in rows]) - frame) / fps
    if np.any(times >= 0):
        raise ValueError("toss observations must precede shared contact")
    contact = np.asarray(contact, float)
    velocity = np.asarray(parameters[:3], float)
    tau = float(parameters[3])
    xyz = contact + times[:, None] * velocity + 0.5 * times[:, None] ** 2 * toss_witness.GRAVITY
    origin = contact - tau * velocity + 0.5 * tau**2 * toss_witness.GRAVITY
    launch = velocity - tau * toss_witness.GRAVITY
    pixels = np.asarray([r["pixel"] for r in rows])
    cameras = np.asarray([r["camera"] for r in rows])
    sigma = np.asarray([r["uncertainty_px"] for r in rows])
    predicted = camera_geometry.project(cameras, xyz, camera_geometry.rows_radial(rows))
    delta = predicted - pixels
    weighted = delta / sigma[:, None]
    offset = origin[:2] - np.asarray(feet["court_xy_m"])
    release_z = origin[2] - config.hand_height_m
    horizontal = float(np.linalg.norm(launch[:2]))
    residual = np.r_[
        weighted.ravel(),
        offset[0] / config.release_lateral_sigma_m,
        offset[1] / config.release_depth_sigma_m,
        release_z / config.hand_height_sigma_m,
        max(horizontal - config.maximum_horizontal_launch_mps, 0) / 0.5,
        max(2 - launch[2], 0),
        max(launch[2] - 12, 0),
    ]
    rms = float(np.sqrt(np.mean(np.sum(weighted**2, axis=1))))
    slack = np.array(
        [
            1 - (rms / config.maximum_weighted_rms) ** 2,
            1 - (offset[1] / config.release_depth_sigma_m) ** 2,
            1 - (offset[0] / config.release_lateral_sigma_m) ** 2,
            1 - (release_z / (2 * config.hand_height_sigma_m)) ** 2,
            1 - (horizontal / config.maximum_horizontal_launch_mps) ** 2,
            launch[2] - 2,
            12 - launch[2],
        ]
    )
    names = [
        "fronts_within_declared_uncertainty",
        "release_depth_near_tracked_feet",
        "release_lateral_near_tracked_feet",
        "release_at_hand_height",
        "near_vertical_launch",
        "upward_release_min",
        "upward_release_max",
    ]
    record = dict(
        shared_contact_xyz_m=contact.tolist(),
        shared_contact_frame=frame,
        shared_position_and_time_with_serve=True,
        incoming_contact_velocity_mps=velocity.tolist(),
        release_seconds_before_contact=tau,
        release_xyz_m=origin.tolist(),
        release_velocity_mps=launch.tolist(),
        image_rms_px=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        weighted_image_rms=rms,
        maximum_weighted_rms=config.maximum_weighted_rms,
        constraints=dict(zip(names, (slack >= 0).tolist())),
        slack=slack.tolist(),
        native_projection=[
            dict(
                frame=r["frame"],
                observed_front=r["pixel"],
                predicted_ball_centre=pred.tolist(),
                xyz_m=position.tolist(),
                error_px=float(np.linalg.norm(error)),
                uncertainty_px=float(s),
            )
            for r, pred, position, error, s in zip(rows, predicted, xyz, delta, sigma, strict=True)
        ],
        front_semantics="Original visible fronts with explicit radii against centres; no timestamp shift.",
        independent_xyz_truth=False,
    )
    # An inward optimizer margin avoids calling floating-point boundary overshoot
    # accepted. The recorded check still uses the unchanged TossConfig limit.
    search_slack = slack.copy()
    search_slack[0] = 1 - (rms / (config.maximum_weighted_rms - 1e-4)) ** 2
    record["optimizer_weighted_rms_margin"] = 1e-4
    return residual, search_slack, record
