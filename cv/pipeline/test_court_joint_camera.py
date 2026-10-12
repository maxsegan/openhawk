import numpy as np

from cv.pipeline.camera_bundle import projection_matrix
from cv.pipeline.camera_project import project_distorted
from cv.pipeline.camera_cal import (
    CameraObservation,
    ITF_LANDMARKS,
    backproject_plane,
    fit_joint_camera,
    metric_error_for_pixel_residual,
    project_joint_camera,
)


def synthetic_observations():
    center = np.array([5.3, -27.0, 10.0])
    dist_center = np.array([960.0, 540.0])
    k1 = 1.5e-8
    views = {10: (1.57, -0.27, 3200.0), 30: (1.575, -0.268, 3260.0), 55: (1.58, -0.265, 3300.0)}
    world = np.asarray(list(ITF_LANDMARKS.values()), dtype=float)
    names = list(ITF_LANDMARKS)
    rows = []
    initials = {}
    for frame, state in views.items():
        matrix = projection_matrix(center, *state, (1920, 1080))
        initials[frame] = matrix
        pixels = project_distorted(matrix, k1, dist_center, world)
        for name, xyz, pixel in zip(names, world, pixels, strict=True):
            rows.append(CameraObservation(frame, name, xyz, pixel))
    return rows, initials, center, k1


def test_joint_camera_recovers_shared_rig_and_views() -> None:
    rows, initials, center, k1 = synthetic_observations()
    solved = fit_joint_camera(rows, initials, maximum_evaluations=300, smoothness_weight=0.0)
    assert solved.success
    assert np.linalg.norm(solved.center - center) < 0.02
    assert abs(solved.k1 - k1) < 1e-9
    residuals = []
    for row in rows:
        residuals.append(
            np.linalg.norm(project_joint_camera(solved, row.frame, row.world[None])[0] - row.pixel)
        )
    assert max(residuals) < 0.02


def test_point_to_line_observations_fit_only_the_normal_component() -> None:
    rows, initials, _, _ = synthetic_observations()
    line_rows = [
        CameraObservation(
            row.frame,
            row.landmark,
            row.world,
            row.pixel + np.array([20.0, 0.0]),
            normal=np.array([0.0, 1.0]),
        )
        for row in rows
    ]
    control_rows = [
        CameraObservation(
            row.frame,
            row.landmark,
            row.world,
            row.pixel,
            normal=np.array([0.0, 1.0]),
        )
        for row in rows
    ]
    solved = fit_joint_camera(line_rows, initials, maximum_evaluations=100)
    control = fit_joint_camera(control_rows, initials, maximum_evaluations=100)
    assert solved.success and control.success
    assert np.allclose(solved.center, control.center, atol=1e-7)
    assert abs(solved.k1 - control.k1) < 1e-12


def test_backprojection_and_metric_conversion_are_height_aware() -> None:
    matrix = projection_matrix(np.array([5.485, -25.0, 9.0]), 1.57, -0.25, 3000.0, (1920, 1080))
    point = np.array([5.485, 23.77, 1.0])
    pixel = project_distorted(matrix, 0.0, np.array([960.0, 540.0]), point[None])[0]
    restored = backproject_plane(matrix, pixel, 1.0)
    assert np.allclose(restored, point, atol=1e-8)
    near = metric_error_for_pixel_residual(
        matrix, np.array([5.485, 0.0, 1.0]), np.array([0.0, 5.0]), plane_height_m=1.0
    )
    far = metric_error_for_pixel_residual(
        matrix, np.array([5.485, 23.77, 1.0]), np.array([0.0, 5.0]), plane_height_m=1.0
    )
    assert far > near > 0.0


def test_parent_convention_reproduces_the_prior_service_centre_geometry() -> None:
    from cv.pipeline.camera_cal import ITF_LANDMARKS, landmark_world, landmark_world_convention

    for landmark in ITF_LANDMARKS:
        assert np.allclose(
            landmark_world(landmark, 0.05),
            landmark_world_convention(landmark, "itf_outside_edge_v1"),
        )


def test_paint_centre_convention_moves_inward_along_every_measured_line() -> None:
    from cv.pipeline.camera_cal import (
        COURT_L,
        COURT_W,
        NET_H_CENTER,
        NET_TAPE_THICKNESS_M,
        SINGLES_INSET,
        landmark_world_convention,
    )

    offset = 0.025

    def centre(landmark):
        return landmark_world_convention(
            landmark, "painted_line_center_v1", edge_offset_m=offset
        )

    # Two measured lines meet at a corner, so both normals apply.
    assert np.allclose(centre("near_left_doubles"), [offset, offset, 0.0])
    assert np.allclose(centre("far_right_doubles"), [COURT_W - offset, COURT_L - offset, 0.0])
    assert np.allclose(centre("near_left_singles"), [SINGLES_INSET + offset, offset, 0.0])
    # Centre marks and the centre service line are measured at their own paint centre,
    # so only the crossing line contributes.
    assert np.allclose(centre("near_center_mark"), [COURT_W / 2.0, offset, 0.0])
    assert np.allclose(centre("far_center_mark"), [COURT_W / 2.0, COURT_L - offset, 0.0])
    assert np.allclose(centre("near_service_t"), [COURT_W / 2.0, 5.485 + offset, 0.0])
    # A tape mid-thickness click sits half the band below the modeled cord top.
    assert np.allclose(
        centre("net_band_center"), [COURT_W / 2.0, 11.885, NET_H_CENTER - NET_TAPE_THICKNESS_M / 2]
    )
    assert np.allclose(centre("net_center_top"), [COURT_W / 2.0, 11.885, NET_H_CENTER])


def test_unknown_landmark_or_convention_is_rejected() -> None:
    import pytest

    from cv.pipeline.camera_cal import landmark_world_convention

    with pytest.raises(ValueError):
        landmark_world_convention("net_left_post_top", "painted_line_center_v1")
    with pytest.raises(ValueError):
        landmark_world_convention("near_left_doubles", "outside_edge")
    with pytest.raises(ValueError):
        landmark_world_convention("near_left_doubles", "painted_line_center_v1", edge_offset_m=0.4)
