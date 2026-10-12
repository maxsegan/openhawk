import numpy as np
import pytest

from cv.experiments.connected_shooting import labeled_passive_tape as tape, net_collision
from cv.experiments.connected_shooting.labeled_net_free_response import (
    FreeNetVelocity,
    response_from_record,
)
from cv.experiments.connected_shooting.labeled_net_signed_response import SignedTapeResponse


def test_source_mutation_cannot_change_velocity_or_key():
    source = np.array([10.0, -20.0, 3.0])
    listed = [4.0, 5.0, -6.0]
    array_law = FreeNetVelocity(source)
    list_law = FreeNetVelocity(listed)
    incoming = np.array([1.0, -30.0, -2.0])
    array_key = array_law.key
    listed_key = list_law.key
    source[:] = 0
    listed[0] = 99.0
    np.testing.assert_array_equal(array_law.velocity(incoming), [10.0, -20.0, 3.0])
    np.testing.assert_array_equal(list_law.velocity(incoming), [4.0, 5.0, -6.0])
    assert array_law.key == array_key == ("free_net_velocity_v1", 10.0, -20.0, 3.0)
    assert list_law.key == listed_key
    returned = array_law.velocity(incoming)
    returned[:] = 0
    np.testing.assert_array_equal(array_law.velocity(incoming), [10.0, -20.0, 3.0])
    assert array_law.key != (10.0, -20.0, 3.0)
    assert array_law.key != tape.TapeResponse(0.5, 0.25, 0.4).key
    hash(array_law.key)


def test_same_assigned_outgoing_for_different_incoming():
    assigned = np.array([-4.0, 12.0, 2.0])
    law = FreeNetVelocity(assigned)
    for incoming in (
        np.array([2.0, -30.0, -3.0]),
        np.array([-8.0, 18.0, 4.0]),
        np.array([1.0, 0.0, -5.0]),
        np.zeros(3),
    ):
        outgoing = law.velocity(incoming)
        np.testing.assert_array_equal(outgoing, assigned)
        assert outgoing is not assigned
        assert outgoing is not law.velocity(incoming)


def test_broad_valid_sideways_up_down_responses():
    cases = (
        np.array([40.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 30.0]),
        np.array([0.0, 0.0, -30.0]),
        np.array([-12.0, 22.0, -5.0]),
        np.array([75.0, -75.0, 75.0]),
        np.array([-75.0, 0.0, 1.0]),
    )
    incoming = np.array([2.0, -30.0, -3.0])
    for assigned in cases:
        law = FreeNetVelocity(assigned)
        np.testing.assert_array_equal(law.velocity(incoming), assigned)
        receipt = law.record(incoming, assigned)
        assert receipt["model"] == "experimental_free_net_velocity_v1"
        assert "passive" not in receipt["model"]
        assert "material law" in receipt["contact_geometry"]
        np.testing.assert_allclose(receipt["outgoing_velocity_mps"], assigned)
        assert receipt["translational_energy_ratio"] > 0
        assert np.isfinite(receipt["incoming_speed_mps"])


def test_rejects_invalid_shape_nonfinite_and_out_of_bounds():
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([1.0, -20.0])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([[1.0, -20.0, 3.0]])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([1.0, -20.0, 3.0, 0.0])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([75.0 + 1e-9, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite outgoing"):
        FreeNetVelocity([0.0, -75.0 - 1e-9, 0.0])
    law = FreeNetVelocity([3.0, -12.0, 1.0])
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, -20.0]))
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, np.nan, -5.0]))
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, np.inf, -5.0]))


def test_record_roundtrip_passive_signed_and_free():
    incoming = np.array([2.0, -30.0, -3.0])
    laws = (
        tape.TapeResponse(0.4, 0.25, 0.4),
        SignedTapeResponse(-0.4, 0.25, 0.4),
        FreeNetVelocity([3.0, -12.0, 1.5]),
    )
    for law in laws:
        outgoing = law.velocity(incoming)
        receipt = {**law.record(incoming, outgoing), "extra": "ignored", "note": 1}
        rebuilt = response_from_record(receipt)
        assert type(rebuilt) is type(law)
        np.testing.assert_array_equal(rebuilt.velocity(incoming), outgoing)
        assert rebuilt.key == law.key

    zero = np.zeros(3)
    rest = FreeNetVelocity(zero).record(zero, zero)
    created = FreeNetVelocity([1.0, 0.0, 0.0]).record(zero, np.array([1.0, 0.0, 0.0]))
    assert rest["incoming_speed_mps"] == 0.0
    assert rest["outgoing_speed_mps"] == 0.0
    assert rest["translational_energy_ratio"] == 1.0
    assert created["translational_energy_ratio"] == float("inf")
    assert not np.isnan(rest["translational_energy_ratio"])
    assert not np.isnan(created["translational_energy_ratio"])


def test_using_response_applies_chosen_outgoing_without_cache_mix():
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
    chosen = np.array([4.0, -8.0, 3.0])
    free = FreeNetVelocity(chosen)
    passive = tape.TapeResponse(0.8, 0.25, 0.4)
    colliding_passive = tape.TapeResponse(0.5, 0.25, 0.4)
    colliding_free = FreeNetVelocity((0.5, 0.25, 0.4))
    assert colliding_free.key != colliding_passive.key

    with tape.using_response(passive):
        tape_result = net_collision.simulate(*args, **kwargs)
    with tape.using_response(free):
        free_result = net_collision.simulate(*args, **kwargs)
    assert net_collision.simulate is original_fn
    assert len(free_result[-1]) == len(original[-1]) == 1
    np.testing.assert_array_equal(free_result[-1][0]["v_out"], chosen)
    np.testing.assert_array_equal(free_result[-1][0]["x"], original[-1][0]["x"])
    assert free_result[-1][0]["position_continuous"] is True
    np.testing.assert_array_equal(free_result[-1][0]["w_out"], free_result[-1][0]["w_in"])
    receipt = free_result[-1][0]["experimental_response"]
    assert receipt["model"] == "experimental_free_net_velocity_v1"
    np.testing.assert_allclose(receipt["outgoing_velocity_mps"], chosen)
    assert not np.array_equal(tape_result[-1][0]["v_out"], chosen)

    with tape.using_response(passive):
        tape_again = net_collision.simulate(*args, **kwargs)
    np.testing.assert_array_equal(tape_again[-1][0]["v_out"], tape_result[-1][0]["v_out"])
    with tape.using_response(colliding_passive):
        numeric_tape = net_collision.simulate(*args, **kwargs)
    with tape.using_response(colliding_free):
        numeric_free = net_collision.simulate(*args, **kwargs)
    np.testing.assert_array_equal(numeric_free[-1][0]["v_out"], [0.5, 0.25, 0.4])
    assert not np.array_equal(numeric_tape[-1][0]["v_out"], numeric_free[-1][0]["v_out"])
    np.testing.assert_array_equal(net_collision.simulate(*args, **kwargs)[0], original[0])
    with pytest.raises(RuntimeError), tape.using_response(free):
        raise RuntimeError("probe")
    assert net_collision.simulate is original_fn
