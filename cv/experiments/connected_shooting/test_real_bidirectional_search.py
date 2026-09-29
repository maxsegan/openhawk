import numpy as np
import pytest

from cv.experiments.connected_shooting import real_bidirectional_search as search
from cv.experiments.connected_shooting import camera_geometry


def camera():
    return np.array([[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]])


def test_ray_point_reprojects_and_meets_depth_plane():
    pixel = np.array([1060.0, 440.0])
    xyz = search.ray_point_at_y(camera(), pixel, 20.0)
    assert xyz[1] == pytest.approx(20.0)
    projected = camera_geometry.project(camera()[None], xyz[None])[0]
    assert np.allclose(projected, pixel)


def test_ground_point_reprojects_and_meets_ball_plane():
    pixel = np.array([960.0, 540.0])
    xyz = search.ground_point(camera(), pixel)
    assert xyz[2] == pytest.approx(search.BALL_RADIUS_M)
    assert np.allclose(camera_geometry.project(camera()[None], xyz[None])[0], pixel)


def test_search_config_rejects_more_than_one_frame_exposure():
    with pytest.raises(ValueError, match="exposure"):
        search.SearchConfig(exposure_duration_frames=1.1).validate()
