"""Admissible-set response protocol and using_response restore."""

from __future__ import annotations

import numpy as np
import pytest

from cv.experiments.connected_shooting import admissible_net_response as admissible
from cv.experiments.connected_shooting import net_collision
from cv.pipeline import net_cord_response as cord
from cv.pipeline.physics_knot_solver import net_impact_velocity


def test_velocity_stays_in_the_box_even_if_the_stored_vector_is_outside():
    incoming = np.array([8.0, -20.0, 2.0])
    law = admissible.AdmissibleSetResponse((40.0, 40.0, 40.0))
    outgoing = law.velocity(incoming)
    assert cord.in_box(incoming, outgoing)
    assert float(np.linalg.norm(outgoing)) <= 0.95 * float(np.linalg.norm(incoming)) + 1e-9


def test_using_admissible_overrides_simulate_and_restores():
    args = (
        np.array([5.0, 14.0, 1.0, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
        0.0,
        np.array([0.0, 5.5, 10.0]),
        30.0,
        "clay",
    )
    kwargs = dict(net_frame=5.5)
    original_fn = net_collision.simulate
    original = net_collision.simulate(*args, **kwargs)
    incoming = original[-1][0]["v_in"]
    stored = cord.project_to_box(incoming, np.array([0.8, 2.0, 0.3]))
    law = admissible.AdmissibleSetResponse(tuple(float(v) for v in stored))
    with admissible.using_admissible(law):
        assert cord.active_mode() == cord.ADMISSIBLE_SET
        replaced = net_collision.simulate(*args, **kwargs)
    assert net_collision.simulate is original_fn
    assert cord.active_mode() == cord.TAPE_CLIP
    assert len(replaced[-1]) == 1
    np.testing.assert_allclose(replaced[-1][0]["v_out"], stored, atol=1e-6)
    np.testing.assert_array_equal(
        net_collision.simulate(*args, **kwargs)[-1][0]["v_out"],
        net_impact_velocity(incoming),
    )


def test_replay_from_verdict_reads_flight_rows_not_configuration():
    incoming = np.array([6.0, -16.0, 1.0])
    v_out = cord.project_to_box(incoming, np.array([1.0, 2.0, 0.2]))
    report = {
        "configuration": {},
        "verdict": {
            "flights": [
                {"flight_index": 0},
                {
                    "flight_index": 1,
                    "net_cord_response": {
                        **cord.receipt(cord.ADMISSIBLE_SET),
                        "v_out_mps": v_out.tolist(),
                        "outgoing_velocity_mps": v_out.tolist(),
                        "model": admissible.MODEL,
                    },
                },
            ]
        },
    }
    ctx = admissible.replay_from_verdict(report)
    with ctx:
        assert cord.active_mode() == cord.ADMISSIBLE_SET
    empty = admissible.replay_from_verdict({"verdict": {"flights": [{"flight_index": 0}]}})
    with empty:
        assert cord.active_mode() == cord.TAPE_CLIP


def test_replay_restores_the_recorded_tape_band():
    incoming = np.array([6.0, -16.0, 1.0])
    v_out = cord.project_to_box(incoming, np.array([1.0, 2.0, 0.2]))
    report = {
        "configuration": {},
        "verdict": {
            "flights": [
                {
                    "flight_index": 0,
                    "net_cord_response": {
                        **cord.receipt(cord.ADMISSIBLE_SET, h_tol_m=cord.H_TOL_WIDE_M),
                        "v_out_mps": v_out.tolist(),
                        "outgoing_velocity_mps": v_out.tolist(),
                        "model": admissible.MODEL,
                    },
                }
            ]
        },
    }
    with admissible.replay_from_verdict(report):
        assert cord.active_mode() == cord.ADMISSIBLE_SET
        assert cord.active_h_tol() == pytest.approx(cord.H_TOL_WIDE_M)
    assert cord.active_h_tol() == pytest.approx(cord.H_TOL_M)


def test_attach_flight_witness_skips_flights_without_a_net_hit():
    report = {
        "configuration": {cord.FIELD: cord.ADMISSIBLE_SET},
        "net_response": {"outgoing_velocity_mps": [1.0, 2.0, 0.3]},
    }
    flights = [
        {"flight_index": 0, "modeled_net_hits": []},
        {"flight_index": 1, "modeled_net_hits": [{"frame": 10.0}]},
    ]
    attached = admissible.attach_flight_witness(flights, report)
    assert "net_cord_response" not in attached[0]
    assert attached[1]["net_cord_response"]["outgoing_velocity_mps"] == [1.0, 2.0, 0.3]
    off = admissible.attach_flight_witness(flights, {"configuration": {}})
    assert off == flights
