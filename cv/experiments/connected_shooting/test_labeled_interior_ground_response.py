"""Corrected integrator, explicit identity, response cache and nested composition."""

from dataclasses import replace
import numpy as np
import pytest
from cv.experiments.connected_shooting import labeled_interior_ground_response as ground
from cv.experiments.connected_shooting.model import Scene


def record(e, i=1, epoch=10):
    return dict(
        schema=ground.SCHEMA,
        routes=[
            dict(flight_index=i, native_contact_epoch=epoch, ground_ordinal=0, normal_restitution=e)
        ],
    )


def test_corrected_root_nominal_cache_identity_and_later_ground_law():
    d = ground.dynamics
    old = d.simulate
    theta = np.array([0, 8, 1, 3, -8, 1, 1, 0, 0], float)
    q = np.linspace(10, 90, 400)
    source = old(theta, 10, q, 50, "hard")
    e = source[-1][0]["applied_restitution"]
    with ground.using_response(record(e)):
        same = d.simulate(theta, 10, q, 50, "hard")
    for a, b in zip(source[:3], same[:3], strict=True):
        np.testing.assert_array_equal(a, b)
    with ground.using_response(record(e * 0.9)):
        moved = d.simulate(theta, 10, q, 50, "hard")
    assert not np.array_equal(moved[0], source[0])
    assert moved[-1][0]["frame"] == source[-1][0]["frame"]
    for name in ["x", "w_in", "w_out"]:
        np.testing.assert_array_equal(moved[-1][0][name], source[-1][0][name])
    np.testing.assert_array_equal(moved[-1][0]["v_out"][:2], source[-1][0]["v_out"][:2])
    assert moved[-1][0]["dwell_seconds"] == source[-1][0]["dwell_seconds"]
    later = moved[-1][1]
    assert "interior_ground_response" not in later
    nominal = d.bounce_reference.court_bounce(later["v_in"], later["w_in"], "hard")
    np.testing.assert_array_equal(later["v_out"], d.rebound_velocity(nominal, "nominal")[0])
    with ground.using_response(record(e)):
        restored = d.simulate(theta, 10, q, 50, "hard")
    np.testing.assert_array_equal(restored[0], source[0])
    assert d.simulate is old


def test_nested_registry_replacement_retains_all_unrelated_routes_and_restores():
    d = ground.dynamics
    old = d.simulate
    theta = np.array([0, 8, 1, 3, -8, 1, 1, 0, 0], float)
    q = np.linspace(10, 90, 400)
    prior = record(0.7)
    new = ground.merge(prior, record(0.8, i=2, epoch=100)["routes"])
    assert new["routes"][0] == ground.normalize(prior)["routes"][0]
    with ground.using_response(prior):
        expected = d.simulate(theta, 10, q, 50, "hard")
        with ground.using_response(new):
            actual = d.simulate(theta, 10, q, 50, "hard")
            other = d.simulate(theta, 100, q + 90, 50, "hard")
        np.testing.assert_array_equal(d.simulate(theta, 10, q, 50, "hard")[0], expected[0])
    np.testing.assert_array_equal(actual[0], expected[0])
    assert other[-1][0]["applied_restitution"] == 0.8 and d.simulate is old
    updated = ground.merge(new, record(0.6)["routes"])
    assert updated["routes"][1] == new["routes"][1]
    assert updated["routes"][0]["normal_restitution"] == 0.6


def test_native_identity_and_bounds_are_explicit():
    contacts = np.array([1.0, 10.0, 30.0, 50.0])
    frames = tuple(np.array([a + 1]) for a in contacts[:-1])
    scene = Scene(
        contact_frames=contacts,
        observation_frames=frames,
        cameras=tuple(np.eye(3, 4)[None] for _ in frames),
        pixels=tuple(np.zeros((1, 2)) for _ in frames),
        spin_parameters=np.zeros((3, 3)),
        fps=50,
        surface="hard",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
    )
    assert ground.normalize(record(0.7), scene)["routes"][0]["flight_index"] == 1
    for rec in [record(1.1), record(0.7, epoch=10.00001), record(0.7, i=2), record(0.7, i=True)]:
        with pytest.raises(ValueError):
            ground.normalize(rec, scene)
    net_scene = replace(scene, net_hit_frames=(np.empty(0), np.array([20.0]), np.empty(0)))
    with pytest.raises(ValueError, match="net"):
        ground.normalize(record(0.7), net_scene)
    with pytest.raises(ValueError, match="epoch"):
        ground.merge(record(0.7), record(0.7, epoch=11)["routes"])


def test_exception_restores_worker_simulator():
    old = ground.dynamics.simulate
    with pytest.raises(RuntimeError), ground.using_response(record(0.7)):
        raise RuntimeError("test")
    assert ground.dynamics.simulate is old
