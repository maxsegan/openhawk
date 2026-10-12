import numpy as np
import pytest

from cv.experiments.connected_shooting.labeled_preparation_physical_seed import (
    ray_at_plane,
    solve_flight,
)
from cv.experiments.connected_shooting import measured_dynamics


def test_net_plane_and_contact_plane_preserve_native_ray():
    camera = np.array([[900, 0, 500, 20], [0, 900, 300, 30], [0, 0, 1, 12]], float)
    projected_source = camera @ np.array([4, 11.885, 1.5, 1])
    pixel = projected_source[:2] / projected_source[2]
    for axis, value in [(1, 11.885), (2, 1.5)]:
        xyz = ray_at_plane(camera, pixel, axis, value)
        projected = camera @ np.r_[xyz, 1]
        np.testing.assert_allclose(projected[:2] / projected[2], pixel, atol=1e-12)
        assert xyz[axis] == value
    with pytest.raises(np.linalg.LinAlgError):
        ray_at_plane(np.zeros((3, 4)), pixel, 1, 11.885)


def test_airborne_endpoint_solve_replays_and_wrong_topology_is_held():
    start, velocity = np.array([4, 4, 2.0]), np.array([2, 10, 1.0])
    target = measured_dynamics.simulate(
        np.r_[start, velocity, np.zeros(3)], 10, np.array([10, 20]), 50, "hard"
    )[0][-1]
    fitted, endpoint, receipt = solve_flight(start, velocity + 0.1, target, 10, 20, 50, "hard", 0)
    assert receipt["usable"]
    np.testing.assert_allclose(endpoint, target, atol=1e-8)
    np.testing.assert_allclose(fitted, velocity, atol=1e-7)
    _, _, wrong = solve_flight(start, velocity, target, 10, 20, 50, "hard", 1)
    assert not wrong["usable"] and wrong["actual_bounces"] == 0


def test_bounce_ray_uses_observed_epoch_and_ray_without_fixed_contact_height():
    from cv.experiments.connected_shooting.labeled_preparation_physical_seed import solve_bounce_ray

    start = np.array([4.0, 4.0, 1.0])
    velocity = np.array([1.0, 10.0, -2.0])
    first, last, fps = 10.0, 40.0, 50.0
    positions, _, _, impacts = measured_dynamics.simulate(
        np.r_[start, velocity, np.zeros(3)], first, np.array([first, last]), fps, "hard"
    )
    assert len(impacts) == 1
    camera = np.array([[900, 0, 500, 20], [0, 900, 300, 30], [0, 0, 1, 12]], float)
    projected = camera @ np.r_[positions[-1], 1]
    recovered, endpoint, receipt = solve_bounce_ray(
        start,
        velocity + 0.02,
        camera,
        projected[:2] / projected[2],
        first,
        last,
        impacts[0]["frame"],
        fps,
        "hard",
    )
    assert receipt["usable"]
    np.testing.assert_allclose(recovered, velocity, atol=1e-6)
    np.testing.assert_allclose(endpoint, positions[-1], atol=1e-6)
