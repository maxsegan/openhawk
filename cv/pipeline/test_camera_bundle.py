from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline.camera_bundle import (
    COURT_LANDMARKS,
    NET_CURVE_X,
    AnchorObservation,
    bundle_match_cameras,
    camera_center,
    fit_bundle,
    frame_or_anchor_view,
    BundleSolution,
    projection_matrix,
)
from cv.pipeline.camera_cal import NET_Y, fixed_f_projection_from_ground, net_height_at_x
from cv.pipeline.camera_project import project_distorted

IMAGE_SIZE = (1920, 1080)
DIST_CENTER = np.asarray([960.0, 540.0])


def _net_world(indices: np.ndarray) -> np.ndarray:
    court_x = NET_CURVE_X[indices]
    return np.stack(
        (
            court_x,
            np.full_like(court_x, NET_Y),
            np.asarray([net_height_at_x(float(x)) for x in court_x]),
        ),
        axis=1,
    )


def test_projection_matrix_rotates_about_the_requested_fixed_center() -> None:
    center = np.asarray([5.4, -28.0, 8.7])
    first = projection_matrix(center, 1.50, -0.20, 4000.0, IMAGE_SIZE)
    second = projection_matrix(center, 1.58, -0.18, 4400.0, IMAGE_SIZE)

    np.testing.assert_allclose(camera_center(first), center, atol=1e-10)
    np.testing.assert_allclose(camera_center(second), center, atol=1e-10)
    assert not np.allclose(first, second)


def test_bundle_recovers_one_center_from_multiple_views() -> None:
    expected_center = np.asarray([5.3, -27.8, 8.6])
    expected_k1 = -3.0e-9
    views = [(1.48, -0.20, 3800.0), (1.55, -0.18, 4200.0), (1.62, -0.22, 4000.0)]
    net_indices = np.linspace(0, len(NET_CURVE_X) - 1, 9).round().astype(int)
    observations = []
    for index, (yaw, tilt, focal) in enumerate(views, start=1):
        projection = projection_matrix(expected_center, yaw, tilt, focal, IMAGE_SIZE)
        court_pixels = project_distorted(projection, expected_k1, DIST_CENTER, COURT_LANDMARKS)
        net_pixels = project_distorted(
            projection, expected_k1, DIST_CENTER, _net_world(net_indices)
        )
        observations.append(
            AnchorObservation(
                point=index,
                frame=100,
                homography=projection[:, [0, 1, 3]],
                court_pixels=court_pixels,
                net_pixels=net_pixels,
                yaw=yaw + 0.005,
                tilt=tilt - 0.005,
                focal=focal * 1.03,
                center=expected_center + np.asarray([0.2 * index, -0.3 * index, 0.1]),
            )
        )

    fitted = fit_bundle(observations, IMAGE_SIZE, maximum_evaluations=400)

    assert fitted.success
    np.testing.assert_allclose(fitted.center, expected_center, atol=0.35)
    assert abs(fitted.k1 - expected_k1) < 2.5e-9
    assert set(fitted.view_parameters) == {1, 2, 3}


def test_bundle_artifact_keeps_v1_fields_and_exposes_physical_rig(tmp_path: Path) -> None:
    center = np.asarray([5.4, -28.0, 8.7])
    yaw, tilt, focal = 1.52, -0.20, 4100.0
    projection = projection_matrix(center, yaw, tilt, focal, IMAGE_SIZE)
    court_to_image = projection[:, [0, 1, 3]]
    image_to_court = np.linalg.inv(court_to_image)
    point_projection = fixed_f_projection_from_ground(
        court_to_image, focal, w=IMAGE_SIZE[0], h=IMAGE_SIZE[1]
    )
    net_indices = np.linspace(0, len(NET_CURVE_X) - 1, 9).round().astype(int)
    cord = project_distorted(projection, 0.0, DIST_CENTER, _net_world(net_indices))
    np.savez_compressed(
        tmp_path / "court_H_per_point.npz",
        pts=np.asarray([1]),
        H=np.asarray([image_to_court]),
    )
    np.savez_compressed(
        tmp_path / "camera_P_per_point.npz",
        pts=np.asarray([1]),
        P=np.asarray([point_projection]),
        net_cord_valid=np.asarray([True]),
        net_cord_xy=np.asarray([cord]),
    )
    np.savez_compressed(
        tmp_path / "court_H_per_frame_v1.npz",
        clips=np.asarray(["pt0001", "pt0001"]),
        frames=np.asarray([100, 101]),
        H=np.asarray([image_to_court, image_to_court]),
        reliable=np.asarray([True, True]),
        source=np.asarray(["registered", "registered_interpolated"]),
    )
    (tmp_path / "court_topology_evidence_v1.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "point": 1,
                        "status": "direct",
                        "frame": "f_0100.jpg",
                    }
                ]
            }
        )
    )

    output = bundle_match_cameras(tmp_path, "frames")

    with np.load(output) as artifact:
        expected = {
            "clips",
            "frames",
            "P",
            "reliable",
            "source",
            "ground_residual_px",
            "net_residual_px",
            "confidence",
            "fallback_ancestry",
            "frame_scope",
            "reference_frame",
            "calibration_point",
            "net_cord_xy",
            "net_cord_valid",
            "net_cord_source",
        }
        assert expected.issubset(artifact.files)
        assert artifact["frame_scope"].tolist() == ["bundle_v1", "bundle_v1"]
        assert np.all(
            artifact["source"]
            == [
                "bundle_v1+registered",
                "bundle_v1+registered_interpolated",
            ]
        )
        assert "k1" in artifact.files
        assert "dist_center" in artifact.files
        assert "camera_center" in artifact.files
        np.testing.assert_allclose(artifact["camera_center"][0], artifact["camera_center"][1])
        np.testing.assert_allclose(camera_center(artifact["P"][0]), artifact["camera_center"][0])
    report = json.loads((tmp_path / "camera_bundle_v1.json").read_text())
    assert report["anchor_views"] == 1
    assert report["net_observed_anchor_views"] == 1
    assert report["frames"] == 2
    assert report["anchor_view_policy"] == "preserve_joint_court_net_solution_v1"
    with np.load(output) as artifact:
        assert artifact["view_fit_source"].tolist() == [
            "joint_court_net_anchor",
            "ground_only_frame_view",
        ]
        assert artifact["net_residual_scope"].tolist() == [
            "same_frame_observed_net",
            "point_anchor_observed_net",
        ]
        assert artifact["net_residual_px"][0] == pytest.approx(
            report["anchors"][0]["net_tape_rms_px"]
        )


def test_anchor_never_discards_height_evidence_to_improve_ground_fit(monkeypatch):
    from cv.pipeline import camera_bundle as bundle

    center = np.asarray([5.4, -28.0, 8.7])
    view = (1.52, -0.20, 4100.0)
    projection = projection_matrix(center, *view, IMAGE_SIZE)
    homography = projection[:, [0, 1, 3]].copy()
    # Deliberately conflicting ground evidence: the joint solution must retain
    # that residual, not erase it and inherit its former good net score.
    homography[1] += 10 * homography[2]
    anchor = AnchorObservation(
        1,
        100,
        homography,
        bundle._project_homography(homography, COURT_LANDMARKS[:, :2]),
        None,
        *view,
        center,
    )
    solution = BundleSolution(center, 0.0, DIST_CENTER, {1: view}, True, 0.0, 0.0, 1)
    calls = []

    def ground_only(*args):
        calls.append(args)
        return projection * 2, view, 0.0

    monkeypatch.setattr(bundle, "fit_frame_view", ground_only)
    actual, actual_view, ground = frame_or_anchor_view(
        homography, 100, anchor, solution, IMAGE_SIZE, (1.4, -0.3, 3000)
    )
    np.testing.assert_array_equal(actual, projection)
    assert actual_view == view
    assert ground == pytest.approx(10 / np.sqrt(2))
    assert ground > 4.0
    assert not calls
    actual, _, ground = frame_or_anchor_view(homography, 101, anchor, solution, IMAGE_SIZE, view)
    assert len(calls) == 1
    np.testing.assert_array_equal(actual, projection * 2)
    assert ground == 0.0


def test_bundle_abstains_with_an_empty_schema_when_no_anchor_solved(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "court_H_per_point.npz",
        pts=np.asarray([], dtype=np.int32),
        H=np.empty((0, 3, 3)),
    )
    np.savez_compressed(
        tmp_path / "camera_P_per_point.npz",
        pts=np.asarray([], dtype=np.int32),
        P=np.empty((0, 3, 4)),
    )
    (tmp_path / "court_topology_evidence_v1.json").write_text('{"rows": []}')

    output = bundle_match_cameras(tmp_path, "frames")

    with np.load(output) as artifact:
        assert artifact["P"].shape == (0, 3, 4)
        assert artifact["camera_center"].shape == (0, 3)
        assert artifact["frame_scope"].tolist() == []
    report = json.loads((tmp_path / "camera_bundle_v1.json").read_text())
    assert report["status"] == "abstained"
    assert report["reason"] == "no_automatic_court_anchors"
