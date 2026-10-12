import numpy as np
import pytest

from cv.experiments.connected_shooting import camera_geometry, contact_geometry, contact_reach


def camera():
    center = np.array([5.48, -18.0, 15.0])
    forward = np.array([5.48, 11.885, 1.0]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsics = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    return intrinsics @ np.c_[rotation, -rotation @ center]


@pytest.mark.parametrize(
    "contact,height,distance", [([1, 0, 0.375], 0.375, 1), ([0, 0, -2], 0, 2), ([0, 0, 3], 1, 2)]
)
def test_continuous_minimum_covers_interior_and_both_endpoints(contact, height, distance):
    result = contact_reach.minimum_reach(contact, [0, 0, 0], [0, 0, 1])
    assert result["minimizing_assumed_root_xyz_m"][2] == pytest.approx(height)
    assert result["minimum_root_to_contact_distance_m"] == pytest.approx(distance)
    assert not result["minimizing_height_is_measurement"]


def test_zero_height_interval_and_invalid_segments():
    assert (
        contact_reach.minimum_reach([0, 0, 3], [0, 0, 0], [0, 0, 0])[
            "minimum_root_to_contact_distance_m"
        ]
        == 3
    )
    for lower, upper in [
        ([0, 0, -1], [0, 0, 1]),
        ([0, 0, 1], [0, 0, 0]),
        ([0, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [float("nan"), 0, 1]),
    ]:
        with pytest.raises(ValueError):
            contact_reach.minimum_reach([0, 0, 3], lower, upper)


@pytest.mark.parametrize("radial", [None, np.array([1e-8, 960, 540])])
def test_known_airborne_contacts_are_never_rejected_by_their_containing_height_interval(radial):
    rng = np.random.default_rng(6013)
    P = camera()
    for _ in range(100):
        root = np.array([rng.uniform(1, 10), rng.uniform(0, 25), rng.uniform(0, 1)])
        vector = rng.normal(size=3)
        vector[2] = abs(vector[2])
        vector *= rng.uniform(0.5, 3.5) / np.linalg.norm(vector)
        contact = root + vector
        pixel = camera_geometry.project(
            P[None], root[None], None if radial is None else radial[None]
        )[0]
        lower = np.r_[contact_geometry.plane_proxy(P, radial, pixel, 0), 0]
        upper = np.r_[contact_geometry.plane_proxy(P, radial, pixel, 1), 1]
        result = contact_reach.minimum_reach(contact, lower, upper)
        assert result["minimum_root_to_contact_distance_m"] <= np.linalg.norm(vector) + 1e-8
        # The closest point is on the same source ray, not an invented exposure.
        projected = camera_geometry.project(
            P[None],
            np.array([result["minimizing_assumed_root_xyz_m"]]),
            None if radial is None else radial[None],
        )[0]
        np.testing.assert_allclose(projected, pixel, atol=1e-6)


def test_grounded_shortcut_can_falsely_reject_an_airborne_far_player():
    P = camera()
    root = np.array([6.0, 24.0, 0.8])
    contact = root + [0.0, 0.0, 2.8]
    pixel = camera_geometry.project(P[None], root[None])[0]
    lower = np.r_[contact_geometry.plane_proxy(P, None, pixel, 0), 0]
    upper = np.r_[contact_geometry.plane_proxy(P, None, pixel, 1), 1]
    assert np.linalg.norm(contact - lower) > 3.5
    assert (
        contact_reach.minimum_reach(contact, lower, upper)["minimum_root_to_contact_distance_m"]
        < 2.8
    )


def test_wrong_depth_can_be_rejected_only_conditionally_and_wider_heights_never_tighten():
    lower, upper = np.array([5.0, 0.0, 0.0]), np.array([5.0, -3.0, 1.0])
    contact = np.array([5.0, -6.0, 4.4])
    narrow = contact_reach.minimum_reach(contact, lower, upper)
    wide = contact_reach.minimum_reach(contact, lower, lower + 1.5 * (upper - lower))
    assert narrow["minimum_root_to_contact_distance_m"] > 4.5
    assert wide["minimum_root_to_contact_distance_m"] < 3.5


def test_random_analytic_minimum_dominates_grid_without_changing_inputs():
    rng = np.random.default_rng(47)
    for _ in range(100):
        lower = np.r_[rng.normal(size=2), 0.0]
        upper = lower + np.r_[rng.normal(size=2), 1.0]
        contact = rng.normal(size=3)
        before = [x.copy() for x in (contact, lower, upper)]
        result = contact_reach.minimum_reach(contact, lower, upper)
        grid = lower + np.linspace(0, 1, 1001)[:, None] * (upper - lower)
        assert (
            result["minimum_root_to_contact_distance_m"]
            <= np.linalg.norm(grid - contact, axis=1).min() + 1e-12
        )
        for a, b in zip(before, (contact, lower, upper)):
            np.testing.assert_array_equal(a, b)


def test_height_interval_crossing_camera_is_held_without_erasing_other_evidence():
    P = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.2]])
    result = contact_geometry.compare_contact(
        62,
        [3, 20, 3],
        "pt1",
        {("pt1", 62, "near"): [{"root_x_native": "960", "root_y_native": "700"}]},
        {("pt1", 62): {"P": P, "radial": None, "reliable": True}},
    )[0]
    assert result["status"] == "measured_proxy_only"
    assert all(
        r["status"] == "measured_conditional_bound"
        for r in result["continuous_reach_sensitivity"][:-1]
    )
    assert result["continuous_reach_sensitivity"][-1]["status"] == "held"
    assert "singularity" in result["continuous_reach_sensitivity"][-1]["reason"]
    page = contact_geometry.reach_table([result])
    assert "62 / near" in page and "held" in page and "1.5 m" in page
    assert "Minimum 3D root-to-contact reach" in page
