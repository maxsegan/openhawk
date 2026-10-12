"""A failed residual plateau is not a solved seed; native net evidence can restore support."""

from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import initialization, model, net_collision


def scene():
    frames = np.array([10.0, 11.0])
    P = np.array([[1000, 0, 0, 0], [0, 0, 1000, 0], [0, 1, 0, 0]], float)
    xyz = np.array([5.0, 11.885, 0.9])
    uv = P @ np.r_[xyz, 1.0]
    return model.Scene(
        contact_frames=np.array([0.0, 20.0]),
        observation_frames=(frames,),
        cameras=(np.repeat(P[None], 2, axis=0),),
        pixels=(np.repeat((uv[:2] / uv[2])[None], 2, axis=0),),
        spin_parameters=np.zeros((1, 3)),
        fps=25.0,
        surface="grass",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
        net_hit_frames=(np.array([10.5]),),
    )


def test_native_net_alternative_restores_real_physical_support():
    s = scene()
    start = np.array([2.0, 25.0, 0.11])
    bad = np.r_[start, [5.0, -6.0, -1.0], np.zeros(3), [1.0, 1.0]]
    with net_collision.physical_eligibility():
        with pytest.raises(ValueError, match="bounce cap"):
            model.chain(s, bad)
        guesses, receipt = initialization._native_net_anchor_guesses(
            s, start, np.zeros(3), (1.0, 1.0), [17.0]
        )
        assert len(guesses) == 1
        result = model.chain(s, np.r_[start, guesses[0], [1.0, 1.0]])[0]
    assert len(result["net_hits"]) == 1
    assert result["net_hits"][0]["frame"] == pytest.approx(10.5, abs=1e-6)
    assert len(result["bounces"]) < 3
    assert receipt["height_constraint_in_final_fit"] is False
    np.testing.assert_allclose(receipt["conditional_mean_xyz_m"], [5, 11.885, 0.9])
    np.testing.assert_array_equal(start, bad[:3])


def test_net_seed_does_not_invent_missing_pixels_or_skip_a_prior_ground():
    s = scene()
    args = (np.array([2.0, 25.0, 0.11]), np.zeros(3), (1.0, 1.0))
    guesses, receipt = initialization._native_net_anchor_guesses(s, *args, [8.0])
    assert guesses == [] and "earlier ground" in receipt["unavailable"]
    s = replace(
        s,
        observation_frames=(s.observation_frames[0][:1],),
        cameras=(s.cameras[0][:1],),
        pixels=(s.pixels[0][:1],),
    )
    guesses, receipt = initialization._native_net_anchor_guesses(s, *args, [17.0])
    assert guesses == [] and "missing original" in receipt["unavailable"]


def test_net_seed_rejects_outside_mesh_without_clipping():
    s = scene()
    s = replace(s, pixels=(s.pixels[0] * [1, 3],))
    guesses, receipt = initialization._native_net_anchor_guesses(
        s, np.array([2.0, 25.0, 0.11]), np.zeros(3), (1.0, 1.0), [17.0]
    )
    assert guesses == [] and "outside the physical mesh" in receipt["unavailable"]


def test_constant_failure_residual_is_not_admitted_as_an_anchor(monkeypatch):
    s = replace(scene(), net_hit_frames=None)

    def unsupported(*args, **kwargs):
        raise ValueError("measured dynamics bounce cap reached")

    monkeypatch.setattr(model, "chain", unsupported)
    with pytest.raises(ValueError, match="no finite anchored seed"):
        initialization.anchor_connected_seed(
            s,
            np.r_[[2, 25, 0.11], [5, -6, -1], np.zeros(3), [1, 1]],
            np.zeros((2, 2)),
            [[]],
            25.0,
            first_contact_xyz_m=np.array([2, 25, 0.11]),
            contact_targets_xyz_m=[None],
            bounce_targets=[[]],
            max_nfev=2,
        )


def test_net_ray_initialization_uses_original_radial_camera_semantics():
    from cv.experiments.connected_shooting import camera_geometry

    s = scene()
    radial = np.repeat(np.array([[1e-7, 0.0, 0.0]]), 2, axis=0)
    distorted = replace(
        s,
        pixels=(camera_geometry.distort(s.pixels[0], radial),),
        camera_distortion=(radial,),
    )
    args = (np.array([2.0, 25.0, 0.11]), np.zeros(3), (1.0, 1.0), [17.0])
    a, _ = initialization._native_net_anchor_guesses(s, *args)
    b, receipt = initialization._native_net_anchor_guesses(distorted, *args)
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(receipt["conditional_mean_xyz_m"], [5.0, 11.885, 0.9], atol=1e-9)


def test_out_of_bound_chart_proposal_is_rejected_without_clipping(monkeypatch):
    from cv.experiments.connected_shooting import labeled_net_height_chart as chart

    def outside(theta, *args, **kwargs):
        result = np.asarray(theta).copy()
        result[3] = 75.01
        return result, {}

    monkeypatch.setattr(chart, "project_net_height", outside)
    guesses, receipt = initialization._native_net_anchor_guesses(
        scene(), np.array([2.0, 25.0, 0.11]), np.zeros(3), (1.0, 1.0), [17.0]
    )
    assert guesses == []
    assert receipt["trials"][0]["status"] == "unsupported"
    assert "original velocity or spin bounds" in receipt["trials"][0]["reason"]
