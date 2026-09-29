from __future__ import annotations

import numpy as np

from cv.pipeline.camera_cal import NET_Y, net_height_at_x, project
from cv.pipeline.camera_metric_refine import (
    fit_clip,
    scaled_vertical_projection,
)


def projection() -> np.ndarray:
    return np.asarray(
        [
            [20.0, 0.0, 0.0, 480.0],
            [0.0, 10.0, -50.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_vertical_refinement_preserves_every_ground_point() -> None:
    camera = projection()
    points = np.asarray([[1.0, 2.0, 0.0], [5.0, 12.0, 0.0], [9.0, 22.0, 0.0]])

    assert np.allclose(
        project(camera, points), project(scaled_vertical_projection(camera, 1.12), points)
    )


def test_joint_height_assignment_recovers_scale_and_locks_players() -> None:
    camera = projection()
    true_scale = 1.08
    projections = {frame: camera for frame in range(10)}
    players = [
        {"player_id": "short", "height_m": 1.70, "body_profile": "neutral"},
        {"player_id": "tall", "height_m": 1.90, "body_profile": "neutral"},
    ]
    observations = {}
    for side, root, height in (
        ("near", np.asarray([4.0, 3.0]), 1.70),
        ("far", np.asarray([7.0, 20.0]), 1.90),
    ):
        rows = []
        for frame in range(10):
            head = project(
                scaled_vertical_projection(camera, true_scale),
                [[root[0], root[1], 0.945 * height]],
            )[0]
            rows.append(
                {
                    "frame": frame,
                    "root_xy": root,
                    "head_xy": head,
                    "confidence": 1.0,
                    "pixel_height": 80.0,
                }
            )
        observations[("pt0001", side)] = rows
    net_x = np.linspace(0.0, 10.97, 9)
    net = project(
        scaled_vertical_projection(camera, true_scale),
        [[x, NET_Y, net_height_at_x(float(x))] for x in net_x],
    )

    result = fit_clip("pt0001", observations, projections, net, players)

    assert result["accepted"] is True
    np.testing.assert_allclose(result["scale"], true_scale, atol=0.01)
    assert result["assignment"]["near"]["player_id"] == "short"
    assert result["assignment"]["far"]["player_id"] == "tall"
    assert result["identity_safe"] is True
    assert result["player_rms_after_px"] < result["player_rms_before_px"]
