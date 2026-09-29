from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as search
from cv.experiments.connected_shooting import model


def fixture(*, long_tail=False):
    scene, p = model.control()
    scene = replace(scene, dynamics="measured_240hz", rebound_mode="point_scales")
    p = np.r_[p, np.zeros(6), [1.0, 1.0]]
    if long_tail:
        scene = replace(scene, contact_frames=np.array([1.0, 13.0, 500.0]))
        p[6:9] = [2.0, 5.0, -20.0]
    return scene, p


@pytest.mark.parametrize("override", [None, (0, 0, "slide"), (1, 0, "grip")])
def test_start_equals_original_healthy_chain_including_parameter_and_regime_slicing(override):
    scene, p = fixture()
    scene = replace(scene, bounce_regime_override=override)
    p[9:15] = [2.0, 1.0, 3.0, -2.0, 4.0, 1.0]
    p[-2:] = [0.95, 1.05]
    original = p.copy()
    expected = model.chain(scene, p)[-1]["start_xyz"]
    np.testing.assert_array_equal(search.terminal_seed_start(scene, p), expected)
    np.testing.assert_array_equal(p, original)
    assert len(scene.contact_frames) == 3


def test_finite_prefix_does_not_require_uninitialized_terminal_tail_to_survive():
    scene, p = fixture(long_tail=True)
    with pytest.raises(ValueError, match="bounce cap"):
        model.chain(scene, p)
    start = search.terminal_seed_start(scene, p)
    healthy, healthy_p = fixture()
    np.testing.assert_array_equal(start, model.chain(healthy, healthy_p)[-1]["start_xyz"])
    assert np.isfinite(start).all()


def test_single_flight_start_never_propagates_provisional_tail(monkeypatch):
    scene, p = fixture(long_tail=True)
    scene = replace(
        scene,
        contact_frames=scene.contact_frames[[0, 2]],
        observation_frames=scene.observation_frames[:1],
        cameras=scene.cameras[:1],
        pixels=scene.pixels[:1],
        spin_parameters=scene.spin_parameters[:1],
    )
    p = np.r_[p[:6], p[9:12], p[-2:]]

    def forbidden(*args, **kwargs):
        pytest.fail("Reading the initial position must not call a simulator")

    monkeypatch.setattr(model, "chain", forbidden)
    result = search.terminal_seed_start(scene, p)
    np.testing.assert_array_equal(result, p[:3])
    result[0] += 10
    assert p[0] == 5


def test_invalid_prefix_still_fails_without_lifting_physical_capacity():
    scene, p = fixture(long_tail=True)
    scene = replace(scene, contact_frames=np.array([1.0, 500.0, 510.0]))
    p[3:6] = [2.0, 5.0, -20.0]
    with pytest.raises(ValueError, match="bounce cap"):
        search.terminal_seed_start(scene, p)


def test_original_prefix_net_route_is_kept(monkeypatch):
    scene, p = fixture()
    scene = replace(scene, net_hit_frames=(np.array([8.0]), np.array([])))
    actual = model.chain
    seen = []

    def check(view, values):
        np.testing.assert_array_equal(view.net_hit_frames[0], [8.0])
        assert len(view.net_hit_frames) == 1
        seen.append(True)
        return actual(view, values)

    monkeypatch.setattr(model, "chain", check)
    assert np.isfinite(search.terminal_seed_start(scene, p)).all()
    assert seen == [True]


def test_net_stop_seed_bounce_cap_is_seed_death_and_does_not_kill_the_point(monkeypatch):
    """An automatic prefix that hits the cap while placing the net launch is seed death.

    The labelled arm reaches the search and falls back to components. The same
    cap on the automatic arm used to exit the worker before that fallback, so
    every earlier flight died with the point. A bare bounce cap still does not
    fall back.
    """
    from cv.experiments.connected_shooting import candidate_attempts as attempts
    from cv.experiments.connected_shooting import initialization
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    scene, parameters = fixture(long_tail=True)
    contract = dict(
        interval=[22.0, 23.0],
        representative=22.5,
        observation_horizon=500.0,
        kind="net_stop",
        latent_ground_epochs_supplied=False,
    )
    scene = replace(scene, terminal_net_tail=contract)

    def initial(view, *args, **kwargs):
        assert view.terminal_net_tail is None
        return parameters[:-2].copy(), {"method": "physical_fixture"}

    def boom(view, values):
        raise BounceCapacityError(1502.555864, 8)

    monkeypatch.setattr(initialization, "image_ballistic_seed", initial)
    monkeypatch.setattr(search, "terminal_seed_start", boom)
    with pytest.raises(
        ValueError, match="net-stop seed measured dynamics bounce cap reached"
    ) as caught:
        search.baseline_seed(None, scene, (np.array([]), np.array([])))
    assert type(caught.value) is ValueError
    assert "1502.555864" in str(caught.value)
    assert "after 8 impacts" in str(caught.value)
    reason = f"worker exit 1: {type(caught.value).__name__}: {caught.value}"
    on = {"whole_point_seed_fallback": "on"}
    assert attempts.should_fallback_to_components(reason, on)
    assert not attempts.should_fallback_to_components(reason, {"whole_point_seed_fallback": "off"})
    assert not attempts.should_fallback_to_components(
        "worker exit 1: BounceCapacityError: measured dynamics bounce cap reached",
        on,
    )

    other = replace(scene, terminal_net_tail={**contract, "kind": "camera_cut"})
    with pytest.raises(BounceCapacityError):
        search.baseline_seed(None, other, (np.array([]), np.array([])))


def test_baseline_constructs_net_projection_before_full_tail(monkeypatch):
    from cv.experiments.connected_shooting import initialization, labeled_net_epoch_chart

    scene, p = fixture(long_tail=True)
    contract = dict(
        interval=[22.0, 23.0],
        representative=22.5,
        observation_horizon=500.0,
        kind="net_stop",
        latent_ground_epochs_supplied=False,
    )
    scene = replace(scene, terminal_net_tail=contract)
    calls = []

    def initial(view, *args, **kwargs):
        assert view.terminal_net_tail is None
        assert max(view.observation_frames[-1]) < 22.0
        return p[:-2].copy(), {"method": "physical_fixture"}

    def project(theta, *args, **kwargs):
        calls.append(theta.copy())
        changed = theta.copy()
        changed[4] = 7.0
        return changed, {"fixture": True}

    monkeypatch.setattr(initialization, "image_ballistic_seed", initial)
    monkeypatch.setattr(labeled_net_epoch_chart, "project_net_velocity", project)
    seeded, receipt = search.baseline_seed(None, scene, (np.array([]), np.array([])))
    assert len(calls) == 1
    assert seeded[7] == 7.0
    assert receipt["terminal_net_tail_seed"]["status"] == "plane_projected"
    assert scene.contact_frames[-1] == 500.0
    assert scene.terminal_net_tail == contract


def test_fixed_rebound_prefix_uses_original_spin_without_point_scale_suffix():
    scene, parameters = fixture()
    scene = replace(scene, rebound_mode="fixed")
    parameters = parameters[:-2]
    parameters[9:15] = [2.0, 1.0, 3.0, -2.0, 4.0, 1.0]
    original = parameters.copy()
    expected = model.chain(scene, parameters)[-1]["start_xyz"]
    np.testing.assert_array_equal(search.terminal_seed_start(scene, parameters), expected)
    np.testing.assert_array_equal(parameters, original)


@pytest.mark.parametrize("kind", ["net_stop", "camera_cut"])
def test_ground_launch_prefix_keeps_the_seed(monkeypatch, kind):
    """A prefix launch already on the ground and descending keeps the unprojected seed."""
    from cv.experiments.connected_shooting import initialization

    scene, parameters = fixture(long_tail=True)
    contract = dict(
        interval=[22.0, 23.0],
        representative=22.5,
        observation_horizon=500.0,
        kind=kind,
        latent_ground_epochs_supplied=False,
    )
    scene = replace(scene, terminal_net_tail=contract)
    seed = parameters[:-2].copy()

    def initial(view, *args, **kwargs):
        return seed.copy(), {"method": "physical_fixture"}

    def boom(view, values):
        raise ValueError(search.GROUND_LAUNCH_DEATH)

    monkeypatch.setattr(initialization, "image_ballistic_seed", initial)
    monkeypatch.setattr(search, "terminal_seed_start", boom)
    seeded, evidence = search.baseline_seed(None, scene, (np.array([]), np.array([])))
    receipt = evidence["terminal_net_tail_seed"]
    assert receipt["status"] == "prefix_launch_refused_seed_retained"
    assert receipt["all_tail_retained_in_objective"]
    np.testing.assert_array_equal(seeded[: len(seed)], seed)
