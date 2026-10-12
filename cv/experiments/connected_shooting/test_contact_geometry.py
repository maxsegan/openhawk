import numpy as np
import pytest

from cv.experiments.connected_shooting import camera_geometry, contact_geometry


def camera():
    return np.array([[20, 0, 0, 900], [0, 10, -50, 500], [0, 0, 0, 1]], float)


@pytest.mark.parametrize("radial", [None, np.array([1e-8, 960, 540])])
def test_root_ray_uses_camera_plane_and_declared_radial_inversion(radial):
    xyz = np.array([[3.0, 20.0, 0.0]])
    pixels = camera_geometry.project(camera()[None], xyz, None if radial is None else radial[None])
    np.testing.assert_allclose(
        contact_geometry.ground_proxy(camera(), radial, pixels[0]), xyz[0, :2], atol=1e-8
    )


@pytest.mark.parametrize("pixel", [[-1, 200], [1920, 200], [200, float("nan")]])
def test_bad_native_roots_fail(pixel):
    with pytest.raises(ValueError, match="native"):
        contact_geometry.ground_proxy(camera(), None, pixel)


def test_degenerate_plane_fails():
    with pytest.raises(ValueError, match="nondegenerate"):
        contact_geometry.ground_proxy(np.zeros((3, 4)), None, [500, 500])


def test_airborne_proxy_sensitivity_does_not_call_a_box_bottom_grounded():
    xyz = np.array([[3, 20, 0.5]])
    pixels = camera_geometry.project(camera()[None], xyz)
    np.testing.assert_allclose(
        contact_geometry.plane_proxy(camera(), None, pixels[0], 0.5), xyz[0, :2]
    )
    assert not np.allclose(contact_geometry.ground_proxy(camera(), None, pixels[0]), xyz[0, :2])


def test_fractional_contact_keeps_both_sides_and_missing_evidence_without_interpolation():
    row = {"root_x_native": "960", "root_y_native": "700", "airborne": ""}
    players = {("pt1", 62, "near"): [row], ("pt1", 63, "near"): [row]}
    cameras = {
        ("pt1", 62): {"P": camera(), "radial": None, "reliable": True},
        ("pt1", 63): {"P": camera(), "radial": None, "reliable": False},
    }
    result = contact_geometry.compare_contact(62.5, [3, 20, 3], "pt1", players, cameras)
    assert [(r["native_frame"], r["side"]) for r in result] == [
        (62, "near"),
        (62, "far"),
        (63, "near"),
        (63, "far"),
    ]
    assert result[0]["horizontal_distance_m"] == pytest.approx(0)
    assert result[0]["ground_proxy_distance_3d_m"] == pytest.approx(3)
    assert result[0]["player_row"]["airborne"] == ""
    assert result[1]["reason"] == result[3]["reason"] == "missing_player_observation"
    assert result[2]["reason"] == "missing_or_held_metric_camera"
    assert [r["frame_offset_from_contact"] for r in result] == [-0.5, -0.5, 0.5, 0.5]
    assert contact_geometry.neighboring_exposures(62) == [62]


def test_duplicate_player_rows_abstain_locally_without_selecting_one():
    rows = [
        {"root_x_native": "960", "root_y_native": "700"},
        {"root_x_native": "970", "root_y_native": "700"},
    ]
    result = contact_geometry.compare_contact(
        62, [3, 20, 3], "pt1", {("pt1", 62, "near"): rows}, {}
    )
    assert result[0]["reason"] == "ambiguous_player_observations"
    assert result[0]["conflicting_player_rows"] == rows
    assert result[0]["player_row"] is None
    assert "horizontal_distance_m" not in result[0]


@pytest.mark.parametrize("frame", [0, float("nan"), float("inf")])
def test_invalid_contact_frames_fail(frame):
    with pytest.raises(ValueError, match="positive source"):
        contact_geometry.neighboring_exposures(frame)
