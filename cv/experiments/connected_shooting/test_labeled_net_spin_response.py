import json

import numpy as np
import pytest

from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity
from cv.experiments.connected_shooting.labeled_net_spin_response import FreeNetState


def test_source_mutation_cannot_change_velocity_spin_or_key():
    velocity_source = np.array([10.0, -20.0, 3.0])
    spin_listed = [40.0, -50.0, 60.0]
    law = FreeNetState(velocity_source, spin_listed)
    incoming = np.array([1.0, -30.0, -2.0])
    incoming_spin = np.array([1.0, 2.0, 3.0])
    key = law.key
    velocity_source[:] = 0
    spin_listed[0] = 99.0
    np.testing.assert_array_equal(law.velocity(incoming), [10.0, -20.0, 3.0])
    np.testing.assert_array_equal(
        law.spin(incoming_spin, incoming, law.velocity(incoming)),
        [40.0, -50.0, 60.0],
    )
    assert law.key == key == ("free_net_state_v1", 10.0, -20.0, 3.0, 40.0, -50.0, 60.0)
    returned_v = law.velocity(incoming)
    returned_w = law.spin(incoming_spin, incoming, returned_v)
    returned_v[:] = 0
    returned_w[:] = 0
    np.testing.assert_array_equal(law.velocity(incoming), [10.0, -20.0, 3.0])
    np.testing.assert_array_equal(
        law.spin(incoming_spin, incoming, law.velocity(incoming)),
        [40.0, -50.0, 60.0],
    )
    hash(law.key)


def test_key_separates_from_free_net_velocity():
    assigned = (4.0, -8.0, 3.0)
    spin = (10.0, -20.0, 30.0)
    state = FreeNetState(assigned, spin)
    free = FreeNetVelocity(assigned)
    assert state.key != free.key
    assert state.key[0] == "free_net_state_v1"
    assert free.key[0] == "free_net_velocity_v1"
    assert state.outgoing_velocity_mps == free.outgoing_velocity_mps == assigned
    assert state.key == ("free_net_state_v1",) + assigned + spin
    assert FreeNetState(assigned, (10.0, -20.0, 31.0)).key != state.key
    assert FreeNetState((4.0, -8.0, 3.1), spin).key != state.key
    hash(state.key)
    hash(free.key)


def test_ordinary_velocity_matches_free_net_velocity():
    assigned = np.array([-4.0, 12.0, 2.0])
    state = FreeNetState(assigned, (1.0, -2.0, 3.0))
    free = FreeNetVelocity(assigned)
    for incoming in (
        np.array([2.0, -30.0, -3.0]),
        np.array([-8.0, 18.0, 4.0]),
        np.array([1.0, 0.0, -5.0]),
        np.zeros(3),
    ):
        outgoing = state.velocity(incoming)
        np.testing.assert_array_equal(outgoing, assigned)
        np.testing.assert_array_equal(outgoing, free.velocity(incoming))
        assert outgoing is not assigned
        assert outgoing is not state.velocity(incoming)


def test_spin_allows_every_direction_zero_and_no_change():
    incoming_spin = np.array([12.0, -8.0, 4.0])
    incoming = np.array([2.0, -30.0, -3.0])
    outgoing = np.array([-4.0, 12.0, 2.0])
    cases = (
        np.array([100.0, 0.0, 0.0]),
        np.array([-100.0, 0.0, 0.0]),
        np.array([0.0, 200.0, 0.0]),
        np.array([0.0, -200.0, 0.0]),
        np.array([0.0, 0.0, 300.0]),
        np.array([0.0, 0.0, -300.0]),
        np.zeros(3),
        incoming_spin.copy(),
        np.array([-12.0, 8.0, -4.0]),
        np.array([1500.0, -1500.0, 1500.0]),
    )
    for assigned in cases:
        law = FreeNetState(outgoing, assigned)
        returned = law.spin(incoming_spin, incoming, outgoing)
        np.testing.assert_array_equal(returned, assigned)
        assert returned is not assigned
        assert returned is not law.spin(incoming_spin, incoming, outgoing)


def test_rejects_invalid_shape_nonfinite_and_out_of_bounds():
    valid_v = [3.0, -12.0, 1.0]
    valid_w = [10.0, -20.0, 30.0]
    with pytest.raises(ValueError, match="finite outgoing velocity"):
        FreeNetState([1.0, -20.0], valid_w)
    with pytest.raises(ValueError, match="finite outgoing velocity"):
        FreeNetState([75.0 + 1e-9, 0.0, 0.0], valid_w)
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [10.0, -20.0])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [[10.0, -20.0, 30.0]])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [10.0, -20.0, 30.0, 0.0])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [10.0, np.nan, 30.0])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [10.0, np.inf, 30.0])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [1500.0 + 1e-9, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite outgoing spin"):
        FreeNetState(valid_v, [0.0, -1500.0 - 1e-9, 0.0])
    law = FreeNetState(valid_v, valid_w)
    incoming = np.array([1.0, -20.0, -5.0])
    outgoing = np.array(valid_v)
    incoming_spin = np.array(valid_w)
    with pytest.raises(ValueError, match="finite incoming"):
        law.velocity(np.array([1.0, -20.0]))
    with pytest.raises(ValueError, match="finite incoming spin"):
        law.spin(np.array([1.0, -20.0]), incoming, outgoing)
    with pytest.raises(ValueError, match="finite incoming spin"):
        law.spin(np.array([1.0, np.nan, 3.0]), incoming, outgoing)
    with pytest.raises(ValueError, match="finite incoming velocity"):
        law.spin(incoming_spin, np.array([1.0, np.inf, -5.0]), outgoing)
    with pytest.raises(ValueError, match="finite outgoing velocity"):
        law.spin(incoming_spin, incoming, np.array([1.0, -20.0]))


def test_record_json_receipt_marks_fitted_spin_flexibility():
    incoming = np.array([2.0, -30.0, -3.0])
    assigned_v = np.array([3.0, -12.0, 1.5])
    assigned_w = np.array([-40.0, 50.0, -60.0])
    law = FreeNetState(assigned_v, assigned_w)
    outgoing = law.velocity(incoming)
    receipt = law.record(incoming, outgoing)
    assert receipt["model"] == "experimental_free_net_state_v1"
    assert receipt["spin_unchanged"] is False
    assert "measured spin" in receipt["contact_geometry"]
    assert "material law" in receipt["contact_geometry"]
    assert "fitted flexibility" in receipt["contact_geometry"]
    np.testing.assert_allclose(receipt["outgoing_velocity_mps"], assigned_v)
    np.testing.assert_allclose(receipt["outgoing_spin_rad_s"], assigned_w)
    assert receipt["outgoing_spin_rad_s"] == list(law.outgoing_spin_rad_s)
    payload = json.dumps(receipt)
    loaded = json.loads(payload)
    assert loaded["model"] == "experimental_free_net_state_v1"
    assert loaded["spin_unchanged"] is False
    assert loaded["outgoing_spin_rad_s"] == [-40.0, 50.0, -60.0]
    unchanged = FreeNetState(assigned_v, [1.0, 2.0, 3.0])
    same_spin_receipt = unchanged.record(incoming, assigned_v)
    assert same_spin_receipt["spin_unchanged"] is False
    np.testing.assert_array_equal(
        unchanged.spin(np.array([1.0, 2.0, 3.0]), incoming, assigned_v),
        [1.0, 2.0, 3.0],
    )
