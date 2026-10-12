"""net_collision.integrate with the compiled ordinary steps must equal the Python loop bit for bit."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import fast_flight, measured_dynamics, net_collision

pytestmark = pytest.mark.skipif(fast_flight.native() is None, reason="no C compiler")


def _trace(monkeypatch, library, *args):
    monkeypatch.setattr(fast_flight, "native", lambda: library)
    try:
        return ("ok", net_collision.integrate(*args))
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
    for events_a, events_b in zip(left[1][4:], right[1][4:], strict=True):
        assert len(events_a) == len(events_b)
        for a, b in zip(events_a, events_b, strict=True):
            assert a.keys() == b.keys()
            for key in a:
                assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


def toward_net(count, seed):
    """Flights from either baseline region toward the net, some low enough to hit it."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        side = rng.choice([-1.0, 1.0])
        y = 11.885 + side * rng.uniform(3, 12)
        position = (rng.uniform(-2, 13), y, rng.uniform(0.2, 2.8))
        velocity = (rng.uniform(-4, 4), -side * rng.uniform(8, 40), rng.uniform(-4, 6))
        spin = rng.uniform(-4, 4, size=3)
        yield np.array([*position, *velocity, *spin])


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_native_net_trace_equals_python_trace(monkeypatch, physical, seed):
    library = fast_flight.native()
    token = net_collision._PHYSICAL_ELIGIBILITY.set(physical)
    try:
        hits = 0
        for n, theta in enumerate(toward_net(50, seed)):
            end = [0.3, 0.9, 1.6][n % 3]
            net_frame = 100.0 + (11.885 - theta[1]) / theta[4] * 25.0  # straight-line crossing
            for settling in (False, True):
                args = (
                    theta,
                    100.0,
                    end,
                    25.0,
                    "hard",
                    net_frame,
                    "nominal",
                    (1.0, 1.0),
                    None,
                    settling,
                )
                python = _trace(monkeypatch, None, *args)
                native = _trace(monkeypatch, library, *args)
                _same(python, native)
                hits += python[0] == "ok" and bool(python[1][5])
        assert hits > 0  # the corpus exercises the net event itself
    finally:
        net_collision._PHYSICAL_ELIGIBILITY.reset(token)
