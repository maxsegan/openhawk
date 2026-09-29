import numpy as np

from camera_P_per_frame import ground_reprojection_error, transport_projection


def test_transport_preserves_vertical_geometry_and_replaces_ground_mapping() -> None:
    P0 = np.array(
        [[800.0, 0.0, 12.0, 4800.0], [0.0, 700.0, -40.0, 3500.0], [0, 0.1, 0.01, 10.0]]
    )
    warp = np.array([[1.05, 0.01, 30.0], [0.0, 0.97, -12.0], [0.0, 0.00002, 1.0]])
    ground0 = P0[:, [0, 1, 3]]
    target_court_to_image = warp @ ground0
    Hf = np.linalg.inv(target_court_to_image)

    Pf = transport_projection(P0, Hf)

    assert ground_reprojection_error(Pf, Hf) < 1e-4
    expected = warp @ P0
    expected /= expected[2, 3]
    np.testing.assert_allclose(Pf, expected, atol=1e-9)
