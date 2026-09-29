import numpy as np
import pytest

from physics import impact
from cv.experiments.connected_shooting import (
    ground_contact as contact,
    measured_dynamics as measured,
)


def energy(state):
    return 0.5 * np.dot(state[3:6], state[3:6]) + 0.5 * impact.ALPHA * contact.RADIUS**2 * np.dot(
        state[6:9], state[6:9]
    )


@pytest.mark.parametrize(
    "velocity,spin",
    [
        ([20, 3, 0.02], [0, 0, 0]),
        ([0, 0, 0.02], [0, 100, 0]),
        ([1, 2, 0], [-2 / contact.RADIUS, 1 / contact.RADIUS, 0]),
    ],
)
def test_contact_is_dissipative_and_continuous_through_sliding_to_rolling(velocity, spin):
    state = np.r_[1.0, 2.0, contact.RADIUS, velocity, spin]
    assert contact.qualifies(state[:3], state[3:6])
    initial = state.copy()
    for _ in range(1000):
        before = energy(state)
        advanced = np.array(contact.advance(state, 1 / 240, "hard"))
        assert advanced[2] == contact.RADIUS and advanced[5] == 0
        assert energy(advanced) <= before + 1e-10
        assert np.linalg.norm(advanced[:2] - state[:2]) <= np.sqrt(2 * before) / 240 + 1e-10
        state = advanced
    assert not np.array_equal(state[:2], initial[:2])


def test_fast_tangent_is_not_clamped_by_small_normal_velocity():
    start = np.array([0, 0, contact.RADIUS, 20, 0, 0.01, 0, 0, 0.0])
    result = np.array(contact.advance(start, 0.01, "hard"))
    assert result[0] > 0.19 and result[3] > 19.9
    assert energy(result) < energy(start)


def test_contact_step_partition_does_not_change_physical_motion():
    state = np.array([0, 0, contact.RADIUS, 3, 2, 0, 0, 0, 10.0])
    one = contact.advance(state, 3.0, "clay")
    for _ in range(720):
        state = contact.advance(state, 1 / 240, "clay")
    np.testing.assert_allclose(one, state, atol=1e-10, rtol=0)


def test_large_rebounds_do_not_qualify_and_contact_cannot_start_in_air():
    assert not contact.qualifies([0, 0, contact.RADIUS], [0, 0, 0.1])
    assert not contact.qualifies([0, 0, contact.RADIUS], [0, 0, -0.001])
    with pytest.raises(ValueError, match="physical floor"):
        contact.advance([0, 0, 1, 1, 0, 0, 0, 0, 0], 1, "hard")


def test_measured_settling_keeps_floor_and_marks_actual_transition():
    theta = np.array([2, 3, 1, 1, 2, -2, 0, 0, 0.0])
    frames = np.arange(0.0, 301.0)
    with pytest.raises(ValueError, match="bounce cap"):
        measured.simulate(theta, 0.0, frames, 25.0, "hard")
    x, v, w, bounces = measured.simulate(theta, 0.0, frames, 25.0, "hard", ground_settling=True)
    assert np.isfinite(x).all() and np.min(x[:, 2]) >= contact.RADIUS - 1e-12
    assert bounces[-1]["normal_contact_mode"] == contact.MODE
    assert bounces[-1]["normal_contact_omitted_apex_m"] <= contact.MAX_APEX_M
    assert len(bounces) <= 8
    assert x[-1, 2] == contact.RADIUS and v[-1, 2] == 0
    # OFF cannot inherit the finite optional trace from either cache ordering.
    with pytest.raises(ValueError, match="bounce cap"):
        measured.simulate(theta, 0.0, frames, 25.0, "hard")


def test_nonsettling_simulation_is_exact_and_initial_fast_contact_has_no_fake_bounce():
    theta = np.array([2, 3, 1, 1, 2, 3, 0, 0, 0.0])
    frames = np.arange(0.0, 16.0)
    off = measured.simulate(theta, 0.0, frames, 25.0, "hard")
    on = measured.simulate(theta, 0.0, frames, 25.0, "hard", ground_settling=True)
    for a, b in zip(off[:3], on[:3]):
        np.testing.assert_array_equal(a, b)
    assert off[3] == on[3] == []
    theta = np.array([2, 3, contact.RADIUS, 20, 0, 0.01, 0, 0, 0.0])
    frames = np.arange(0.0, 4.0)
    x, v, w, bounces = measured.simulate(theta, 0.0, frames, 25.0, "hard", ground_settling=True)
    assert not bounces and x[-1, 0] > 4 and v[-1, 0] > 19


def test_net_settles_only_after_required_mesh_and_default_trace_stays_exact():
    from cv.experiments.connected_shooting import net_collision as net

    theta = np.array([5.5, 18, 1, 0, -20, -1, 0, 0, 0.0])
    short = np.arange(1.0, 15.0)
    off = net.simulate(theta, 1, short, 25, "hard", net_frame=8.64)
    on = net.simulate(theta, 1, short, 25, "hard", net_frame=8.64, ground_settling=True)
    for a, b in zip(off[:3], on[:3]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(off[3:], on[3:]):
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            assert ra.keys() == rb.keys()
            for key in ra:
                np.testing.assert_equal(ra[key], rb[key])
    long = np.arange(1.0, 301.0)
    with pytest.raises(ValueError, match="bounce cap"):
        net.simulate(theta, 1, long, 25, "hard", net_frame=8.64)
    x, v, _, bounces, hits = net.simulate(
        theta, 1, long, 25, "hard", net_frame=8.64, ground_settling=True
    )
    assert len(hits) == 1 and hits[0]["supplied_frame"] == 8.64
    assert hits[0]["frame"] == off[4][0]["frame"]
    assert bounces[-1]["frame"] > hits[0]["frame"]
    assert bounces[-1]["normal_contact_mode"] == contact.MODE
    assert x[-1, 2] == contact.RADIUS and v[-1, 2] == 0
    with pytest.raises(ValueError, match="bounce cap"):
        net.simulate(theta, 1, long, 25, "hard", net_frame=8.64)


@pytest.mark.parametrize("backend", ["measured", "net"])
def test_resolved_small_hops_continue_to_horizon_without_forced_settling(monkeypatch, backend):
    from cv.experiments.connected_shooting import net_collision as net
    from physics import bounce_reference

    target = measured if backend == "measured" else net
    original = target.rebound_velocity

    def large_normal(*args, **kwargs):
        velocity, receipt = original(*args, **kwargs)
        velocity[2] = 0.4  # Every apex remains 8 mm, above contact resolution.
        return velocity, receipt

    monkeypatch.setattr(target, "rebound_velocity", large_normal)
    theta = np.array([2, 3, 1, 1, 2, -2, 0, 0, 0.0])
    kwargs = {}
    if backend == "net":
        theta = np.array([5.5, 18, 1, 0, -20, -1, 0, 0, 0.0])
        kwargs = {"net_frame": 8.64}
    frames = np.arange(1.0, 301.0)
    measured._TRACE_CACHE.clear()
    try:
        with pytest.raises(ValueError, match="bounce cap"):
            target.simulate(theta, 1, frames, 25, "hard", ground_settling=False, **kwargs)
        result = target.simulate(theta, 1, frames, 25, "hard", ground_settling=True, **kwargs)
        x, v, w, bounces = result[:4]
        assert np.isfinite(x).all() and np.min(x[:, 2]) >= contact.RADIUS - 1e-12
        assert 32 < len(bounces) <= 12 / bounce_reference.DWELL_SECONDS + 1
        assert not any("normal_contact_mode" in b for b in bounces)
        assert bounces[-1]["frame"] <= frames[-1]
        assert np.all(np.diff([b["frame"] for b in bounces]) >= bounce_reference.DWELL_SECONDS * 25)
        if backend == "net":
            assert len(result[4]) == 1
            from cv.experiments.connected_shooting import model

            scene = model.Scene(
                contact_frames=np.array([1.0, 300.0]),
                observation_frames=(frames,),
                cameras=(np.zeros((len(frames), 3, 4)),),
                pixels=(np.zeros((len(frames), 2)),),
                spin_parameters=np.zeros((1, 3)),
                fps=25,
                surface="hard",
                dynamics="measured_240hz",
                ground_settling=True,
                net_hit_frames=(np.array([8.64]),),
            )
            actual = model.chain(scene, theta)[0]
            shared = model.chain(scene, model.shared_contact_seed(scene, theta))[0]
            for flight in (actual, shared):
                np.testing.assert_array_equal(flight["positions"], x)
                assert len(flight["bounces"]) == len(bounces)
                assert len(flight["net_hits"]) == 1
        with pytest.raises(ValueError, match="bounce cap"):
            target.simulate(theta, 1, frames, 25, "hard", ground_settling=False, **kwargs)
    finally:
        measured._TRACE_CACHE.clear()


@pytest.mark.parametrize("surface", ["hard", "clay", "grass"])
def test_nominal_law_can_reach_unchanged_contact_resolution_after_eighth_impact(surface):
    theta = np.array([5.0, 5.0, 0.15, 10.0, 0.0, -2.0, 0.0, 0.0, 0.0])
    result = measured.simulate(theta, 0, np.arange(201.0), 25, surface, ground_settling=True)
    bounces = result[3]
    assert len(bounces) > 8
    assert bounces[-1]["normal_contact_omitted_apex_m"] <= contact.MAX_APEX_M
    assert bounces[-1]["normal_contact_resolved_impacts"] == len(bounces)
    assert bounces[-1]["normal_contact_integration_policy"] == contact.CACHE_TAG
    prefix = np.arange(0.0, bounces[7]["frame"] - 0.001)
    off = measured.simulate(theta, 0, prefix, 25, surface, ground_settling=False)
    on = measured.simulate(theta, 0, prefix, 25, surface, ground_settling=True)
    for a, b in zip(off[:3], on[:3]):
        np.testing.assert_array_equal(a, b)


def test_settling_prior_count_and_bounce_override_are_not_limited_by_legacy_capacity():
    theta = np.array([5.0, 5.0, 0.15, 10.0, 0.0, -2.0, 0.0, 0.0, 0.0])
    options = {"initial_successive_grounds": 40, "bounce_regime_override": (9, "slide")}
    with contact.using(True):
        result = measured.simulate(theta, 0, np.arange(201.0), 25, "hard", **options)
    assert np.isfinite(result[0]).all()
    for invalid in (-1, True, 1.2):
        with pytest.raises(ValueError, match="prior successive-ground"):
            measured.simulate(
                theta,
                0,
                np.arange(10.0),
                25,
                "hard",
                ground_settling=True,
                initial_successive_grounds=invalid,
            )
    with pytest.raises(ValueError, match="prior successive-ground"):
        measured.simulate(theta, 0, np.arange(10.0), 25, "hard", ground_settling=False, **options)


def test_scene_serialization_replays_explicit_contact_and_overrides_ambient_policy():
    from dataclasses import asdict
    from cv.experiments.connected_shooting import model

    frames = np.arange(4.0)
    scene = model.Scene(
        contact_frames=np.array([0.0, 3.0]),
        observation_frames=(frames,),
        cameras=(np.zeros((4, 3, 4)),),
        pixels=(np.zeros((4, 2)),),
        spin_parameters=np.zeros((1, 3)),
        fps=25,
        surface="hard",
        dynamics="measured_240hz",
        ground_settling=True,
    )
    p = np.array([2, 3, contact.RADIUS, 20, 0, 0.01])
    first = model.chain(scene, p)[0]
    replay = model.chain(model.Scene(**asdict(scene)), p)[0]
    assert first["initial_normal_contact"]["physical_bounce_invented"] is False
    assert first["bounces"] == []
    for key in ("positions", "velocities", "end_xyz"):
        np.testing.assert_array_equal(first[key], replay[key])
    legacy = model.Scene(**(asdict(scene) | {"ground_settling": False}))
    with pytest.raises(ValueError, match="bounce cap"):
        model.chain(legacy, p)
    with contact.using(True), pytest.raises(ValueError, match="bounce cap"):
        model.chain(legacy, p)
    assert not contact.active()


def test_contact_policy_is_typed_and_cannot_bypass_original_state_bounds():
    from cv.pipeline import s6_labeled_stage as stage

    assert stage.shared_settings({})["ground_settling"] == "off"
    assert stage.shared_settings({"ground_settling": "on"})["ground_settling"] == "on"
    with pytest.raises(ValueError, match="ground_settling"):
        stage.shared_settings({"ground_settling": True})
    with pytest.raises(ValueError, match="search bounds"):
        contact.advance([0, 0, contact.RADIUS, 251, 0, 0, 0, 0, 0], 0.01, "hard")


@pytest.mark.parametrize("normal_speed", [None, 0.4])
def test_unreached_net_preserves_pre_net_guard_with_settling_enabled(monkeypatch, normal_speed):
    from cv.experiments.connected_shooting import net_collision as net

    calls = []
    original = net.rebound_velocity

    def counted(*args, **kwargs):
        calls.append(1)
        velocity, receipt = original(*args, **kwargs)
        if normal_speed is not None:
            velocity[2] = normal_speed
        return velocity, receipt

    monkeypatch.setattr(net, "rebound_velocity", counted)
    measured._TRACE_CACHE.clear()
    try:
        with pytest.raises(ValueError, match="bounce cap|ground state has no above-ground flight"):
            net.simulate(
                np.array([2, 3, 1, 1, -2, -2, 0, 0, 0.0]),
                1,
                np.arange(1.0, 301.0),
                25,
                "hard",
                net_frame=8.64,
                ground_settling=True,
            )
        assert 0 < len(calls) <= 8
        if normal_speed is not None:
            assert len(calls) == 8
    finally:
        measured._TRACE_CACHE.clear()


def test_scene_late_bounce_override_matches_settling_integrator_contract():
    from dataclasses import replace
    from cv.experiments.connected_shooting import model

    frames = np.arange(201.0)
    scene = model.Scene(
        contact_frames=np.array([0.0, 200.0]),
        observation_frames=(frames,),
        cameras=(np.zeros((len(frames), 3, 4)),),
        pixels=(np.zeros((len(frames), 2)),),
        spin_parameters=np.zeros((1, 3)),
        fps=25,
        surface="hard",
        dynamics="measured_240hz",
        ground_settling=True,
        bounce_regime_override=(0, 9, "slide"),
    )
    scene.validate()
    actual = model.chain(scene, np.array([5.0, 5.0, 0.15, 10.0, 0.0, -2.0, 0.0, 0.0, 0.0]))[0]
    assert np.isfinite(actual["positions"]).all()
    with pytest.raises(ValueError, match="regime surrogate"):
        replace(scene, ground_settling=False).validate()
