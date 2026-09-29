from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import local_flight_repair_probe as probe
from cv.experiments.connected_shooting import model


def verdict(accepted, *, broken_seam=False):
    return {
        "structural_connections_valid": not broken_seam,
        "flights": [
            {"accepted": value, "checks": {"continuity_into_neighbours": not broken_seam}}
            for value in accepted
        ],
    }


def test_local_membership_improvement_cannot_trade_away_an_accepted_neighbour():
    before = verdict([True, False, True, True])
    assert probe.improvement_allowed(before, verdict([True, True, True, True]), 0.0005)
    assert not probe.improvement_allowed(before, verdict([False, True, True, True]), 0.0)
    assert not probe.improvement_allowed(before, verdict([True, False, True, True]), 0.0)


@pytest.mark.parametrize("gap", [0.001001, np.inf, np.nan])
def test_better_pixel_or_flight_count_cannot_certify_an_open_endpoint(gap):
    assert not probe.improvement_allowed(verdict([True, False, True]), verdict([True] * 3), gap)


def test_exported_chain_seams_and_inventory_are_mandatory():
    before = verdict([True, False, True])
    assert not probe.improvement_allowed(before, verdict([True] * 3, broken_seam=True), 0.0)
    assert not probe.improvement_allowed(before, verdict([True] * 2), 0.0)
    structurally_invalid = verdict([True] * 3)
    structurally_invalid["structural_connections_valid"] = False
    assert not probe.improvement_allowed(before, structurally_invalid, 0.0)


def test_local_parameter_indices_leave_contacts_rebound_and_distant_flights_unchanged():
    scene = SimpleNamespace(contact_frames=np.arange(5))
    blocks = model.shared_parameter_slices(scene)
    values = np.arange(blocks["rebound_scales"].stop, dtype=float)
    initial = values.copy()
    values[probe.local_indices(scene, 2)] += 1.0
    np.testing.assert_array_equal(values[blocks["contacts"]], initial[blocks["contacts"]])
    np.testing.assert_array_equal(
        values[blocks["rebound_scales"]], initial[blocks["rebound_scales"]]
    )
    for block in ("velocities", "spins"):
        old, new = initial[blocks[block]].reshape(-1, 3), values[blocks[block]].reshape(-1, 3)
        np.testing.assert_array_equal(old[[0, 1, 3]], new[[0, 1, 3]])


@pytest.mark.parametrize("index", [-1, 0, 3, 4])
def test_probe_refuses_serve_terminal_and_out_of_range_flights(index):
    with pytest.raises(ValueError, match="interior flight"):
        probe.local_indices(SimpleNamespace(contact_frames=np.arange(5)), index)
