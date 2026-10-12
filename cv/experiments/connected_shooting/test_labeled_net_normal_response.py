"""Direct normal law: native qualification, first-impact scope and exact replay."""

from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_passive_tape as tape, net_collision
from cv.experiments.connected_shooting.labeled_net_free_response import (
    FreeNetVelocity,
    response_from_record,
)
from cv.experiments.connected_shooting.labeled_net_normal_response import (
    NetNormalResponse,
    prior_from_observations,
)


def simulate(law, horizon=23):
    with tape.using_response(law):
        return net_collision.simulate(
            np.array([5.0, 14.0, 1.2, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
            0.0,
            np.arange(float(horizon)),
            30.0,
            "grass",
            net_frame=5.5,
        )


def test_direct_normal_cache_roundtrip_and_nominal_components():
    base = FreeNetVelocity([0.1, 3.0, -3.0])
    low, high = NetNormalResponse(base, 0.4), NetNormalResponse(base, 0.8)
    original_fn = net_collision.simulate
    nominal = simulate(base)
    a, b, again = simulate(low), simulate(high), simulate(low)
    assert net_collision.simulate is original_fn
    np.testing.assert_array_equal(a[0], again[0])
    assert not np.array_equal(a[0], b[0])
    for result, coefficient in [(a, 0.4), (b, 0.8)]:
        bounce = result[-2][0]
        baseline = nominal[-2][0]
        np.testing.assert_array_equal(bounce["x"], baseline["x"])
        np.testing.assert_array_equal(bounce["v_out"][:2], baseline["v_out"][:2])
        np.testing.assert_array_equal(bounce["w_out"], baseline["w_out"])
        assert bounce["v_out"][2] == pytest.approx(-coefficient * bounce["v_in"][2])
        assert bounce["applied_restitution"] == coefficient
        law = response_from_record(result[-1][0]["experimental_response"])
        np.testing.assert_array_equal(simulate(law)[0], result[0])
    # A later bounce uses the nominal law, rather than applying this coefficient again.
    long = simulate(low, 30)
    assert len(long[-2]) == 2
    assert "first_ground_normal_restitution" not in long[-2][1]


def context():
    scene = SimpleNamespace(
        contact_frames=[0.0, 25.0],
        observation_frames=[np.array([7.0, 8.0, 13.0, 15.0, 17.0])],
        surface="grass",
    )
    return dict(
        scene=scene,
        heldout=scene,
        events=[
            dict(event_type="net_hit", frame=5.5, frame_interval=[5.0, 6.0]),
            dict(event_type="bounce", frame=12.0, frame_interval=[11.0, 12.5]),
        ],
    )


def test_qualification_uses_observed_rebound_and_resolved_membership():
    c = context()
    prior = prior_from_observations(c, [1.0, 1.0])
    assert prior["approach_frames"] == [7.0, 8.0]
    assert prior["rebound_frames"] == [13.0, 15.0, 17.0]
    assert not prior["qualification_uses_fitted_incidence_or_gates"]
    c["scene"].observation_frames = [np.array([7.0, 8.0, 13.0, 15.0])]
    with pytest.raises(ValueError, match="three rebound"):
        prior_from_observations(c, [1.0, 1.0])
    c = context()
    c["events"][0]["status"] = "ambiguous"
    with pytest.raises(ValueError, match="resolved"):
        prior_from_observations(c, [1.0, 1.0])


@pytest.mark.parametrize("value", [0.0, 1.01, np.nan, np.inf])
def test_restitution_bounds(value):
    with pytest.raises(ValueError, match="restitution"):
        NetNormalResponse(FreeNetVelocity([0.1, 3.0, -3.0]), value)


def test_second_impact_cannot_supply_first_rebound_qualification():
    c = context()
    c["events"].append(dict(event_type="bounce", frame=16.0, frame_interval=[15.5, 16.5]))
    # Only13/15 are supported strictly between the first and second impacts.
    with pytest.raises(ValueError, match="three rebound"):
        prior_from_observations(c, [1.0, 1.0])
    c["scene"].observation_frames = [np.array([7.0, 8.0, 13.0, 14.0, 15.0, 17.0])]
    prior = prior_from_observations(c, [1.0, 1.0])
    assert prior["rebound_frames"] == [13.0, 14.0, 15.0]
