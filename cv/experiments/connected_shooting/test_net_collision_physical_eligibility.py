"""Declared-net timing remains evidence rather than a physical on/off switch."""

from contextvars import copy_context
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import net_collision as net
from cv.experiments.connected_shooting import labeled_passive_tape as tape
from cv.experiments.connected_shooting import labeled_prefix_later_net as later
from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity
from cv.experiments.connected_shooting.labeled_net_normal_response import NetNormalResponse
from cv.pipeline.rich_ball_physics import R_BALL

THETA = np.array([5.5, 18.0, 1.0, 0.0, -20.0, -1.0, 0.0, 0.0, 0.0])
QUERIES = np.array([1.0, 8.64, 12.0])


def run(theta=THETA, epoch=8.64):
    return net.simulate(theta, 1.0, QUERIES, 25.0, "hard", net_frame=epoch)


def test_default_parity_and_timing_window_crossing():
    original = run()
    actual = original[-1][0]["frame"]
    inside, outside = actual - 1 + 1e-7, actual - 1 - 1e-7
    assert len(run(epoch=inside)[-1]) == 1
    assert len(run(epoch=outside)[-1]) == 0
    with net.physical_eligibility():
        a, b = run(epoch=inside), run(epoch=outside)
        assert len(a[-1]) == len(b[-1]) == 1
        for x, y in zip(a[:3], b[:3], strict=True):
            np.testing.assert_array_equal(x, y)
        # Actual event time remains observed evidence in the real later-net objective.
        scene = SimpleNamespace(net_hit_frames=[np.array([outside])])
        residual = later.net_residuals(scene, [QUERIES], [dict(net_hits=b[-1])])
        assert residual[-1] > 4.0
    np.testing.assert_array_equal(run()[0], original[0])
    assert len(run(epoch=outside)[-1]) == 0


@pytest.mark.parametrize(
    "x,z,allowed",
    [
        (5.485, 0.5, True),
        (-0.915, 0.5, True),
        (11.885, 0.5, True),
        (-0.91501, 0.5, False),
        (11.88501, 0.5, False),
        (5.485, 2.0, False),
        (5.485, R_BALL - 1e-8, False),
    ],
)
def test_mesh_span_height_and_ground_boundary(x, z, allowed):
    with net.physical_eligibility():
        assert net._collision_eligible(20.0, 10.0, (x, 11.885, z)) == allowed


def test_high_wide_and_no_crossing_are_not_collisions():
    for x, z, vy in [(5.5, 3.0, -20.0), (13.0, 1.0, -20.0), (5.5, 1.0, 20.0)]:
        theta = THETA.copy()
        theta[[0, 2, 4]] = [x, z, vy]
        with net.physical_eligibility():
            assert not run(theta)[-1]


def test_collision_continuity_and_default_noop_when_physical():
    old = run()
    with net.physical_eligibility():
        new = run()
        trace = net.integrate(THETA, 1.0, 0.44, 25.0, "hard", 8.64, "nominal", (1.0, 1.0), None)
    for a, b in zip(old[:3], new[:3], strict=True):
        np.testing.assert_array_equal(a, b)
    t, x, v = trace[:3]
    i = np.flatnonzero(np.diff(t) == 0)[0]
    np.testing.assert_array_equal(x[i], x[i + 1])
    assert not np.array_equal(v[i], v[i + 1])


@pytest.mark.parametrize(
    "law",
    [
        tape.TapeResponse(0.0, 0.1, 0.2),
        FreeNetVelocity([0.1, 3.0, -1.0]),
        NetNormalResponse(FreeNetVelocity([0.1, 3.0, -1.0]), 0.5),
    ],
)
def test_response_clones_replay_and_cache_isolation(law):
    epoch = run()[-1][0]["frame"] - 1.1
    with tape.using_response(law):
        before = run(epoch=epoch)
        with net.physical_eligibility():
            changed = run(epoch=epoch)
            repeat = run(epoch=epoch)
        restored = run(epoch=epoch)
    assert not before[-1] and len(changed[-1]) == 1 and not restored[-1]
    np.testing.assert_array_equal(before[0], restored[0])
    np.testing.assert_array_equal(changed[0], repeat[0])
    record = changed[-1][0]["experimental_response"]
    from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record

    with net.physical_eligibility(), tape.using_response(response_from_record(record)):
        np.testing.assert_array_equal(run(epoch=epoch)[0], changed[0])


def test_nested_failure_and_other_context_restore():
    untouched = copy_context()
    with pytest.raises(RuntimeError), net.physical_eligibility():
        assert net._PHYSICAL_ELIGIBILITY.get()
        assert not untouched.run(net._PHYSICAL_ELIGIBILITY.get)
        with net.physical_eligibility(False):
            assert not net._PHYSICAL_ELIGIBILITY.get()
        raise RuntimeError("test")
    assert not net._PHYSICAL_ELIGIBILITY.get()
    with pytest.raises(TypeError), net.physical_eligibility("on"):
        pass


def test_declared_topology_still_required_for_model_dispatch(monkeypatch):
    from dataclasses import replace
    from cv.experiments.connected_shooting import model

    scene, parameters = model.control()
    scene = replace(scene, dynamics="measured_240hz", net_hit_frames=None)
    baseline = model.chain(scene, parameters)

    def forbidden(*args, **kwargs):
        raise AssertionError("undeclared net flight dispatched to collision model")

    monkeypatch.setattr(net, "simulate", forbidden)
    with net.physical_eligibility():
        changed = model.chain(scene, parameters)
    for a, b in zip(baseline, changed, strict=True):
        np.testing.assert_array_equal(a["positions"], b["positions"])


def test_tape_radius_boundary_is_physical_not_timing_based():
    tape_height = net.net_tape_height(5.485)
    with net.physical_eligibility():
        assert net._collision_eligible(100, 10, (5.485, 11.885, tape_height + R_BALL))
        assert not net._collision_eligible(10, 10, (5.485, 11.885, tape_height + R_BALL + 1e-8))


def test_missing_physical_hit_remains_penalized_by_declared_event_objective():
    theta = THETA.copy()
    theta[2] = 3.0
    with net.physical_eligibility():
        result = run(theta)
        scene = SimpleNamespace(net_hit_frames=[np.array([8.64])])
        residual = later.net_residuals(scene, [QUERIES], [dict(net_hits=result[-1])])
    assert not result[-1]
    np.testing.assert_array_equal(residual, [10.0, 10.0, 10.0, 10.0])


def test_only_one_hit_even_with_outgoing_velocity_reversal():
    # A bounce-back starts the next integration step exactly on the net plane,
    # which is itself another numerical plane-crossing candidate.
    with net.physical_eligibility(), tape.using_response(FreeNetVelocity([0.0, 10.0, 0.0])):
        result = run()
    assert len(result[-1]) == 1
    hit = result[-1][0]
    assert hit["v_in"][1] < 0 < hit["v_out"][1]
