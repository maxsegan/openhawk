import numpy as np

from cv.validation.s6root_metric_acceptance import (
    flight_metric_row,
    interpolate_net_crossing,
    net_height_m,
)


def test_interpolate_net_crossing() -> None:
    trajectory = [
        {"xyz": [2.0, 10.0, 1.0]},
        {"xyz": [4.0, 14.0, 2.0]},
    ]
    crossing = interpolate_net_crossing(trajectory)
    assert crossing is not None
    assert np.allclose(crossing, [2.9425, 11.885, 1.47125])


def test_net_is_taller_toward_sideline() -> None:
    assert net_height_m(0.0) > net_height_m(10.97 / 2.0)


def test_metric_gate_uses_exact_constraint_crossing() -> None:
    point = {"point": "match__pt0001"}
    attempt = {
        "flight_index": 0,
        "start_frame": 10,
        "end_frame": 30,
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
    }
    fit = {
        "start_xyz": [5.5, 4.0, 1.0],
        "end_xyz": [5.5, 18.0, 1.0],
        "start_player_distance_m": 1.0,
        "end_player_distance_m": 1.0,
        "bounces": [],
        "net_constraint": {
            "mode": "plane_crossing_height_barrier",
            "xyz": [5.5, 11.885, 1.25],
            "satisfied": True,
        },
        # This sparse output witness disagrees deliberately; the exact optimizer
        # witness is authoritative for the physical net constraint.
        "trajectory": [
            {"xyz": [5.5, 10.0, 0.8]},
            {"xyz": [5.5, 14.0, 0.8]},
        ],
    }

    row = flight_metric_row(point, attempt, fit, [], pixel_accepted=True)

    assert row["net_crossing_plausible"]
    assert row["metric_gate_accepted"]
    assert row["net_crossing_height_m"] == 1.25
