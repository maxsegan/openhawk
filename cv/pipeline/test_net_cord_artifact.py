from pathlib import Path

import numpy as np

from cv.pipeline.bounce_detect import GroundProjector, tape_line_ud, tape_pixel_dist
from cv.pipeline.camera_artifacts import expand_point_cameras


def _point_camera(path: Path) -> np.ndarray:
    projection = np.asarray(
        [[100.0, 0.0, 0.0, 300.0], [0.0, 20.0, -80.0, 362.0], [0.0, 0.0, 0.0, 1.0]]
    )
    cord = np.stack([np.linspace(300.0, 1300.0, 9), np.linspace(510.0, 515.0, 9)], axis=1)
    np.savez_compressed(
        path,
        pts=np.asarray([1]),
        P=np.asarray([projection]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
        net_cord_source=np.asarray(["observed_connected_tape_segments"]),
    )
    return cord


def test_expand_and_projector_preserve_observed_net_cord(tmp_path: Path) -> None:
    cord = _point_camera(tmp_path / "camera_P_per_point.npz")
    frames = tmp_path / "frames" / "pt0001"
    frames.mkdir(parents=True)
    (frames / "f_0003.jpg").write_bytes(b"frame identity only")

    output = expand_point_cameras(tmp_path, "frames")
    projector = GroundProjector(str(output), "pt0001")

    np.testing.assert_allclose(tape_line_ud(projector, 3), cord)
    assert projector.net_cord_source[3] == "observed_connected_tape_segments"


def test_sparse_observed_cord_distance_uses_segments_not_vertices(tmp_path: Path) -> None:
    projection = np.asarray(
        [[100.0, 0.0, 0.0, 300.0], [0.0, 20.0, -80.0, 362.0], [0.0, 0.0, 0.0, 1.0]]
    )
    cord = np.asarray([[100.0, 250.0], [300.0, 260.0], [500.0, 250.0]])
    camera = tmp_path / "camera.npz"
    np.savez_compressed(
        camera,
        clips=np.asarray(["pt0001"]),
        frames=np.asarray([10]),
        P=np.asarray([projection]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
        net_cord_source=np.asarray(["observed"]),
    )
    projector = GroundProjector(str(camera), "pt0001")

    distance, signed_vertical = tape_pixel_dist(projector, 200.0, 255.0, 10)

    assert distance < 1e-9
    assert abs(signed_vertical) < 1e-9
