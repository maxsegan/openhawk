import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_passive_tape as tape, net_collision
from cv.experiments.connected_shooting.labeled_net_signed_response import SignedTapeResponse


def test_zero_angle_matches_original_law_exactly():
    signed = SignedTapeResponse(0, 0.25, 0.4)
    original = tape.TapeResponse(0, 0.25, 0.4)
    for incoming in (
        np.array([2.0, -30.0, -3.0]),
        np.array([-2.0, 30.0, 3.0]),
        np.array([0.5, -18.0, -1.2]),
    ):
        outgoing = signed.velocity(incoming)
        np.testing.assert_array_equal(outgoing, original.velocity(incoming))
        np.testing.assert_array_equal(outgoing, tape.net_impact_velocity(incoming))


def test_passive_energy_for_signed_approach_velocities():
    cases = (
        (-np.pi / 3, np.array([-2.0, -30.0, -3.0])),
        (-0.5, np.array([1.5, 22.0, -2.0])),
        (0.0, np.array([2.0, -30.0, -3.0])),
        (0.4, np.array([0.0, -18.0, 1.0])),
        (np.pi / 3, np.array([4.0, 25.0, -8.0])),
        (-np.pi / 2, np.array([1.0, -12.0, 4.0])),
    )
    for angle, incoming in cases:
        law = SignedTapeResponse(angle, 0.25, 0.4)
        outgoing = law.velocity(incoming)
        assert outgoing @ outgoing <= incoming @ incoming
        assert law.record(incoming, outgoing)["translational_energy_ratio"] <= 1


def test_negative_angle_can_steepen_descent_while_slowing():
    law = SignedTapeResponse(-np.arctan2(0.8, 0.6), 0.25, 0.4)
    incoming = np.array([-2.0, -30.0, -3.0])
    outgoing = law.velocity(incoming)
    assert outgoing[2] < incoming[2]
    assert outgoing @ outgoing < incoming @ incoming


def test_rejects_nonapproaching_and_invalid_inputs():
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(-np.pi / 2 - 1e-9, 0.25, 0.4)
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(np.pi / 2 + 1e-9, 0.25, 0.4)
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(0.0, -1e-12, 0.4)
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(0.0, 1.0 + 1e-12, 0.4)
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(0.0, 0.25, 0.0)
    with pytest.raises(ValueError, match="signed passive"):
        SignedTapeResponse(np.nan, 0.25, 0.4)
    law = SignedTapeResponse(-np.pi / 2, 0.25, 0.4)
    with pytest.raises(ValueError, match="approaching"):
        law.velocity(np.array([1.0, -20.0, -5.0]))
    with pytest.raises(ValueError, match="court-normal"):
        SignedTapeResponse(0.0, 0.25, 0.4).velocity(np.array([1.0, 0.0, -5.0]))
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, -20.0]))
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, np.nan, -5.0]))


def test_using_response_accepts_signed_law_and_restores():
    args = (
        np.array([5.0, 14.0, 1.2, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
        0.0,
        np.array([0.0, 5.5, 10.0]),
        30.0,
        "clay",
    )
    kwargs = dict(net_frame=5.5)
    original_fn = net_collision.simulate
    original = net_collision.simulate(*args, **kwargs)
    law = SignedTapeResponse(-np.arctan2(0.8, 0.6), 0.25, 0.4)
    with tape.using_response(law):
        signed = net_collision.simulate(*args, **kwargs)
    assert net_collision.simulate is original_fn
    assert len(signed[-1]) == len(original[-1]) == 1
    np.testing.assert_array_equal(signed[-1][0]["x"], original[-1][0]["x"])
    assert not np.array_equal(signed[-1][0]["v_out"], original[-1][0]["v_out"])
    receipt = signed[-1][0]["experimental_response"]
    assert receipt["model"] == "experimental_signed_passive_tape_v1"
    assert receipt["effective_signed_impulse"] is True
    assert receipt["independently_observed_cable_normal"] is False
    np.testing.assert_array_equal(net_collision.simulate(*args, **kwargs)[0], original[0])
    with pytest.raises(RuntimeError), tape.using_response(law):
        raise RuntimeError("probe")
    assert net_collision.simulate is original_fn
