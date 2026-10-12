"""The compiled free-flight loop must reproduce the Python integration bit for bit."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import fast_flight, measured_dynamics

pytestmark = pytest.mark.skipif(fast_flight.native() is None, reason="no C compiler")


def _trace(monkeypatch, library, *args, **kwargs):
    monkeypatch.setattr(fast_flight, "native", lambda: library)
    try:
        return ("ok", measured_dynamics.integrate(*args, **kwargs))
    except (ValueError, measured_dynamics.BounceCapacityError) as error:
        return ("error", f"{type(error).__name__}: {error}")


def _same(left, right):
    assert left[0] == right[0]
    if left[0] == "error":
        assert left[1] == right[1]
        return
    for a, b in zip(left[1][:4], right[1][:4], strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)
    assert len(left[1][4]) == len(right[1][4])
    for a, b in zip(left[1][4], right[1][4], strict=True):
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


def thetas(count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        position = (rng.uniform(-6, 6), rng.uniform(-14, 14), rng.uniform(0.05, 3.2))
        velocity = (rng.uniform(-8, 8), rng.uniform(-45, 45), rng.uniform(-12, 14))
        spin = rng.uniform(-4, 4, size=3)  # hundreds of rad/s
        yield np.array([*position, *velocity, *spin])


@pytest.mark.parametrize("surface", ["hard", "clay", "grass"])
@pytest.mark.parametrize("seed", [1, 2])
def test_native_trace_equals_python_trace(monkeypatch, surface, seed):
    library = fast_flight.native()
    for n, theta in enumerate(thetas(60, seed)):
        end = [0.05, 0.4, 1.3, 2.7][n % 4]
        args = (theta, 100.0, end, 25.0, surface, "nominal", np.array([1.0, 1.0]), None)
        for kwargs in ({}, {"stop_after_first_impact": True}):
            python = _trace(monkeypatch, None, *args, **kwargs)
            native = _trace(monkeypatch, library, *args, **kwargs)
            _same(python, native)


def test_errors_and_threshold_states_take_the_python_decision(monkeypatch):
    library = fast_flight.native()
    cases = [
        np.array([0.0, 0.0, 1.0, 0.0, 250.0, 0.0, 0.0, 0.0, 0.0]),  # speed exactly at bound
        np.array([0.0, 0.0, 1.0, 0.0, 251.0, 0.0, 0.0, 0.0, 0.0]),  # speed outside bound
        np.array([0.0, 0.0, 0.0, 0.0, 10.0, -1.0, 0.0, 0.0, 0.0]),  # descending on the ground
        np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # dropped ball, many bounces
    ]
    for theta in cases:
        args = (theta, 0.0, 3.0, 25.0, "hard", "nominal", np.array([1.0, 1.0]), None)
        _same(_trace(monkeypatch, None, *args), _trace(monkeypatch, library, *args))


def test_capacity_refill_continues_the_same_trace(monkeypatch):
    library = fast_flight.native()
    original = fast_flight.native_advance

    def small(library, state, elapsed, end, step, r_ball, **kwargs):
        # Force several FF_CAPACITY returns by pretending the horizon is short.
        return original(
            library, state, elapsed, min(end, elapsed + 7 * step), step, r_ball, **kwargs
        )

    theta = np.array([0.0, -10.0, 1.0, 0.5, 25.0, 4.0, 10.0, 0.0, 0.0])
    args = (theta, 0.0, 1.5, 25.0, "hard", "nominal", np.array([1.0, 1.0]), None)
    python = _trace(monkeypatch, None, *args)
    monkeypatch.setattr(fast_flight, "native_advance", small)
    native = _trace(monkeypatch, library, *args)
    _same(python, native)


def _impacts_only(result):
    status, value = result
    return (status, value) if status == "error" else (status, value[4])


@pytest.mark.parametrize("surface", ["hard", "clay"])
@pytest.mark.parametrize("seed", [3, 4])
def test_untraced_first_impact_equals_traced(monkeypatch, surface, seed):
    """``trace=False`` (first-impact charts) keeps every impact and error, native or Python."""
    library = fast_flight.native()
    for n, theta in enumerate(thetas(60, seed)):
        end = [0.05, 0.4, 1.3, 2.7][n % 4]
        args = (theta, 100.0, end, 25.0, surface, "nominal", np.array([1.0, 1.0]), None)
        traced = _trace(monkeypatch, library, *args, stop_after_first_impact=True)
        for chosen in (library, None):
            quiet = _trace(monkeypatch, chosen, *args, stop_after_first_impact=True, trace=False)
            left, right = _impacts_only(traced), _impacts_only(quiet)
            assert left[0] == right[0]
            if left[0] == "error":
                assert left[1] == right[1]
                continue
            assert len(left[1]) == len(right[1])
            for a, b in zip(left[1], right[1], strict=True):
                assert a.keys() == b.keys()
                for key in a:
                    assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
            assert all(len(array) == 0 for array in quiet[1][:4])


def _first(fn):
    try:
        return ("ok", fn())
    except (ValueError, measured_dynamics.BounceCapacityError) as error:
        return ("error", f"{type(error).__name__}: {error}")


def _python_first(theta, end, surface, profile, scales):
    impacts = measured_dynamics.integrate(
        theta,
        100.0,
        end,
        25.0,
        surface,
        profile,
        scales,
        None,
        stop_after_first_impact=True,
        trace=False,
    )[4]
    return float(impacts[0]["frame"]) if impacts else -1.0


def _first_cases():
    rng = np.random.default_rng(7)
    for n, theta in enumerate(thetas(400, 11)):
        yield theta, [0.05, 0.4, 1.3, 2.7][n % 4]
    for _ in range(200):  # descending from low heights: crossings near step boundaries
        theta = np.array(
            [
                rng.uniform(-5, 5),
                rng.uniform(-12, 12),
                rng.uniform(0.0331, 0.2),
                rng.uniform(-3, 3),
                rng.uniform(-30, 30),
                rng.uniform(-20, -0.01),
                *rng.uniform(-4, 4, size=3),
            ]
        )
        yield theta, 1.0
    yield np.array([0.0, 0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0]), 2.0  # zero horizontal speed
    yield np.array([0.0, 0.0, 1.0, 0.0, 251.0, 0.0, 0.0, 0.0, 0.0]), 2.0  # outside the bound
    yield np.array([0.0, 0.0, 1.0, 0.0, 250.0, 0.0, 0.0, 0.0, 0.0]), 2.0  # at the bound
    yield np.array([0.0, 0.0, 0.0333, 1.0, 5.0, 2.0, 0.0, 0.0, 0.0]), 1.0  # ground launch, up
    yield np.array([0.0, 0.0, 0.0333, 1.0, 5.0, -2.0, 0.0, 0.0, 0.0]), 1.0  # ground launch, down


@pytest.mark.parametrize("surface", ["hard", "clay", "grass", "grass@w=1.06,a=0.15,base=fresh"])
def test_native_first_impact_equals_python_first_impact(surface):
    count = 0
    for theta, end in _first_cases():
        for profile, scales in (
            ("nominal", np.array([1.0, 1.0])),
            ("retention_low", np.array([0.9, 1.1])),
            ("nominal", np.array([1.3, 1.0])),
        ):
            python = _first(lambda: _python_first(theta, end, surface, profile, scales))
            native = _first(
                lambda: measured_dynamics.first_impact_frame(
                    theta, 100.0, end, 25.0, surface, profile, scales, -1.0
                )
            )
            assert python == native, (theta, end, profile, scales)
            count += python[0] == "ok" and python[1] != -1.0
    assert count > 300  # most cases do reach the ground


def test_native_brentq_matches_scipy_on_crossing_steps():
    """The ported brentq returns SciPy's root bit for bit on real crossing steps."""
    from cv.experiments.connected_shooting.ground_root import descending_ground_root
    from cv.pipeline.rich_ball_physics import R_BALL

    library = fast_flight.native()
    checked = 0
    for theta, end in _first_cases():
        w = measured_dynamics._launch_spin(theta)
        state = (*map(float, theta[:6]), *map(float, w))
        status, elapsed, seconds, impact = fast_flight.native_first_impact(
            library, state, 0.0, end, 1.0 / 240.0, R_BALL
        )
        if status != fast_flight.FF_IMPACT:
            continue
        clock = 0.0
        s = state
        while clock < elapsed:  # replay the ordinary steps in Python
            s = fast_flight.rk4_step(s, 1.0 / 240.0)
            clock += 1.0 / 240.0
        nxt = fast_flight.rk4_step(s, 1.0 / 240.0)
        expected_seconds, expected_impact = descending_ground_root(s, nxt, 1.0 / 240.0)
        assert clock == elapsed
        assert seconds == expected_seconds and impact == tuple(expected_impact)
        checked += 1
    assert checked > 300
