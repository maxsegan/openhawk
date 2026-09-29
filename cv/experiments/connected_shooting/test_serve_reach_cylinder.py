"""The serve reach cylinder is a hard cut, so its geometry and bounds must be exact."""

from __future__ import annotations

import numpy as np
import pytest

from cv.experiments.connected_shooting import serve_reach_cylinder as cylinder_module


def player_state(**overrides):
    state = {
        "player": "Server",
        "side": "far",
        "stature_m": 1.90,
        "court_centre_xy_m": [4.0, 24.0],
        "box_height_native_px": 95.0,
    }
    state.update(overrides)
    return state


def test_band_and_radius_scale_with_stature_and_margin():
    built = cylinder_module.build(player_state())
    assert built.radius_m == pytest.approx(1.35 * 1.90)
    assert built.height_interval_m == pytest.approx((1.40 * 1.90, 1.85 * 1.90))
    assert built.zero_penalty_height_interval_m == pytest.approx((1.50 * 1.90, 1.75 * 1.90))
    assert built.metres_per_native_pixel == pytest.approx(1.90 / 95.0)


def test_zero_margin_is_the_soft_zero_penalty_band():
    built = cylinder_module.build(
        player_state(), cylinder_module.ServeReachConfig(margin_statures=0.0)
    )
    assert built.height_interval_m == pytest.approx(built.zero_penalty_height_interval_m)


def test_toss_apex_and_tap_are_both_outside_a_real_band():
    built = cylinder_module.build(player_state())
    for height in (4.9, 1.86):
        verdict = built.evaluate([4.0, 24.0, height])
        assert not verdict["inside"]
        assert verdict["height_excess_m"] > 0
    assert built.evaluate([4.0, 24.0, 3.0])["inside"]


def test_depth_slice_is_the_cylinder_cut_at_a_pinned_court_y():
    built = cylinder_module.build(player_state())
    at_centre = built.depth_slice(24.0)
    assert at_centre["cylinder_reaches_this_depth"]
    assert at_centre["half_width_m"] == pytest.approx(built.radius_m)
    assert at_centre["x_interval_m"] == pytest.approx([4.0 - built.radius_m, 4.0 + built.radius_m])
    offset = built.depth_slice(24.0 + built.radius_m / 2)
    assert offset["half_width_m"] < at_centre["half_width_m"]
    unreachable = built.depth_slice(24.0 + 2 * built.radius_m)
    assert not unreachable["cylinder_reaches_this_depth"]
    assert unreachable["x_interval_m"] is None
    assert unreachable["z_interval_m"] == pytest.approx(list(built.height_interval_m))


def test_a_contact_outside_the_radius_is_cut_even_at_a_legal_height():
    built = cylinder_module.build(player_state())
    verdict = built.evaluate([4.0 + built.radius_m + 0.5, 24.0, 3.0])
    assert not verdict["inside"]
    assert verdict["height_excess_m"] == 0.0
    assert verdict["radial_excess_m"] == pytest.approx(0.5)


def test_missing_stature_or_bad_configuration_refuses():
    with pytest.raises(ValueError):
        cylinder_module.build(player_state(stature_m=None))
    with pytest.raises(ValueError):
        cylinder_module.ServeReachConfig(height_low_statures=2.0).validate()
    with pytest.raises(ValueError):
        cylinder_module.ServeReachConfig(radius_statures=0.0).validate()


def test_missing_box_height_keeps_the_metric_band_and_drops_the_pixel_band():
    built = cylinder_module.build(player_state(box_height_native_px=None))
    assert built.metres_per_native_pixel is None
    verdict = built.evaluate([4.0, 24.0, 3.0])
    assert verdict["inside"]
    assert verdict["band_native_px_above_box_bottom"] is None


def synthetic_camera():
    """A downward-looking camera behind the far baseline, no distortion."""
    centre = np.array([8.0, -12.0, 9.0])
    forward = np.array([0.0, 1.0, -0.28])
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.vstack([right, -up, forward])
    intrinsics = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    return intrinsics @ np.c_[rotation, -rotation @ centre]


def test_the_ray_recovers_the_world_point_that_made_its_pixel():
    camera = synthetic_camera()
    truth = np.array([5.0, 22.0, 3.1])
    pixel = camera @ np.r_[truth, 1.0]
    pixel = pixel[:2] / pixel[2]
    ray = cylinder_module.image_ray(camera, None, pixel)
    recovered = cylinder_module.ray_at_depth(ray, 22.0)
    assert recovered == pytest.approx(truth, abs=1e-6)


def test_the_ray_cut_names_the_branches_whose_height_is_impossible():
    camera = synthetic_camera()
    truth = np.array([4.0, 23.5, 3.2])
    pixel = camera @ np.r_[truth, 1.0]
    pixel = pixel[:2] / pixel[2]
    built = cylinder_module.build(player_state(court_centre_xy_m=[4.0, 24.2]))
    ray = cylinder_module.image_ray(camera, None, pixel)
    cut = cylinder_module.ray_cut(ray, built, [22.5, 23.0, 23.5, 24.0, 24.5])
    assert cut["branch_count"] == 5
    assert 23.5 in cut["depths_inside_cylinder_m"]
    # The camera sits behind the near baseline, so moving a fixed pixel deeper
    # into the far court moves it down the ray: depth and height trade off, which
    # is exactly why a depth branch can only be a serve at one height.
    heights = [row["ray_height_m"] for row in cut["branches"]]
    assert heights == sorted(heights, reverse=True)
    # Over the whole serve region the ray height moves only centimetres, so the
    # cut is not a depth pruner: this branch set is legal end to end.
    assert cut["branch_count_inside"] == 5
    assert max(heights) - min(heights) < 0.5


def test_a_toss_apex_pixel_is_cut_at_every_depth_branch_in_the_serve_region():
    camera = synthetic_camera()
    apex = np.array([4.0, 23.5, 4.6])
    pixel = camera @ np.r_[apex, 1.0]
    pixel = pixel[:2] / pixel[2]
    built = cylinder_module.build(player_state(court_centre_xy_m=[4.0, 24.2]))
    cut = cylinder_module.ray_cut(
        cylinder_module.image_ray(camera, None, pixel),
        built,
        [22.9, 23.4, 23.9, 24.4],
    )
    assert cut["depths_inside_cylinder_m"] == []
    assert all(row["height_excess_m"] > 0 for row in cut["branches"])


def test_contact_hypotheses_are_points_on_ray_and_never_bounds():
    camera = synthetic_camera()
    truth = np.array([4.2, 23.6, 3.0])
    pixel = camera @ np.r_[truth, 1.0]
    pixel = pixel[:2] / pixel[2]
    ray = cylinder_module.image_ray(camera, None, pixel)
    prior = {
        "status": "supported",
        "mode_component_index": 0,
        "components": [
            {
                "weight": 1.0,
                "mean_xyz_m": [4.1, 23.7, 3.05],
                "covariance_xyz_m2": (np.diag([0.1, 0.2, 0.08]) ** 2).tolist(),
            }
        ],
    }
    rows = cylinder_module.contact_hypotheses(ray, prior, {"court_xy_m": [4.0, 23.9]}, 1.9)
    assert len(rows) == 3
    for row in rows:
        point = np.asarray(row["contact_xyz_m"])
        projected = camera @ np.r_[point, 1.0]
        np.testing.assert_allclose(projected[:2] / projected[2], pixel, atol=1e-6)
        assert not row["optimizer_bound"]
        assert not row["optimizer_inequality"]


def test_same_player_regularization_is_serve_number_aware_and_soft():
    prior = {
        "mode_component_index": 0,
        "components": [{"covariance_xyz_m2": (np.diag([0.2, 0.3, 0.1]) ** 2).tolist()}],
    }
    history = [
        {
            "attempt_id": "a",
            "player": "P",
            "side": "far",
            "serve_number": 1,
            "accepted": True,
            "contact_xyz_m": [4.0, 23.5, 3.0],
        },
        {
            "attempt_id": "b",
            "player": "P",
            "side": "far",
            "serve_number": 2,
            "accepted": True,
            "contact_xyz_m": [8.0, 20.0, 2.0],
        },
    ]
    regularization = cylinder_module.same_player_regularization(history, "P", "far", 1, prior)
    scored = cylinder_module.evaluate_same_player_regularization([4.2, 23.5, 3.0], regularization)
    assert regularization["history_count"] == 1
    assert regularization["target_xyz_m"] == [4.0, 23.5, 3.0]
    assert scored["selector_penalty"] == pytest.approx(0.5)
    assert not scored["hard_gate"]
