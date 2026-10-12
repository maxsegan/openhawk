import numpy as np

from cv.experiments.connected_shooting import net_collision


def test_net_transition_is_continuous_and_passive() -> None:
    theta = np.array([5.5, 18.0, 1.0, 0.0, -20.0, -1.0, 0.0, 0.0, 0.0])
    positions, velocities, _spin, _bounces, hits = net_collision.simulate(
        theta, 1.0, np.array([1.0, 8.64, 12.0]), 25.0, "hard", net_frame=8.64
    )
    assert len(hits) == 1
    assert hits[0]["position_continuous"]
    assert hits[0]["x"][1] == 11.885
    assert np.linalg.norm(hits[0]["v_out"]) < np.linalg.norm(hits[0]["v_in"])
    assert velocities.shape == positions.shape


def test_net_residual_requires_plane_and_tape() -> None:
    hit = {"x": np.array([5.485, 11.885, 0.9]), "tape_height_m": 0.91}
    hit["frame"] = 10.0
    np.testing.assert_allclose(net_collision.residuals([hit], 10.0), 0.0)
