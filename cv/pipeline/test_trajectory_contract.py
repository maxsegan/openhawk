import copy
import json

import pytest

from cv.pipeline.trajectory_contract import terminal_coverage_report, trajectory_connection_report
from cv.pipeline.flight_anchors import BALL_RADIUS_M


@pytest.mark.parametrize(
    "kind,valid",
    [
        ("second_bounce", True),
        ("terminal_bounce", True),
        ("net_stop", True),
        ("out_of_view", False),
        ("span_end", False),
        (None, False),
    ],
)
def test_terminal_coverage_requires_an_explicit_physical_ending(kind, valid):
    contacts = [
        {
            "terminal": True,
            "terminal_cutoff": {
                "point_end": {
                    "event_type": "point_end",
                    "frame": 50.0,
                    "point_end": {"source": "test_witness", "termination_kind": kind},
                }
            },
        }
    ]
    report = terminal_coverage_report(
        contacts,
        [
            {
                "terminal_end": True,
                "end_frame": 50.0,
                "end_xyz": [1.0, 2.0, BALL_RADIUS_M],
            }
        ],
    )
    assert report["valid"] is valid
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("frame", [49.0, 51.0, None, float("nan"), float("inf")])
def test_terminal_coverage_rejects_early_stops_and_dead_ball_tails(frame):
    contacts = [
        {
            "terminal": True,
            "terminal_cutoff": {
                "point_end": {
                    "event_type": "point_end",
                    "frame": 50.0,
                    "point_end": {"source": "test_witness", "termination_kind": "second_bounce"},
                }
            },
        }
    ]
    report = terminal_coverage_report(contacts, [{"terminal_end": True, "end_frame": frame}])
    assert report["reasons"] == ["terminal_end_not_covered"]
    json.dumps(report, allow_nan=False)


def test_terminal_coverage_rejects_a_fallback_track_boundary():
    report = terminal_coverage_report(
        [{"terminal": True, "terminal_cutoff": {"source": "active_span_end"}}],
        [{"terminal_end": True, "end_frame": 50.0}],
    )
    assert report["reasons"] == ["terminal_evidence_missing"]


@pytest.mark.parametrize(
    "xyz,reason",
    [
        ([1.0, 2.0, 0.12382446393747704], "terminal_endpoint_off_ground"),
        ([1.0, 2.0, -0.01], "terminal_endpoint_off_ground"),
        ([1.0, 2.0, float("nan")], "terminal_endpoint_invalid"),
        ([1.0, float("inf"), BALL_RADIUS_M], "terminal_endpoint_invalid"),
        ([1.0, 2.0], "terminal_endpoint_invalid"),
        (None, "terminal_endpoint_invalid"),
    ],
)
def test_matching_ending_time_cannot_certify_an_airborne_or_invalid_bounce(xyz, reason):
    contact = {
        "terminal": True,
        "terminal_cutoff": {
            "point_end": {
                "event_type": "point_end",
                "frame": 50.0,
                "point_end": {"source": "test_witness", "termination_kind": "second_bounce"},
            }
        },
    }
    fits = [{"terminal_end": True, "end_frame": 50.0, "end_xyz": xyz}]
    report = terminal_coverage_report([contact], fits)
    assert report["valid"] is False
    assert report["reasons"] == [reason]
    json.dumps(report, allow_nan=False)


def _flights():
    return [
        {
            "start_frame": float(index * 10),
            "end_frame": float((index + 1) * 10),
            "start_xyz": [float(index), 0.0, 1.0],
            "end_xyz": [float(index + 1), 0.0, 1.0],
        }
        for index in range(4)
    ]


def test_connections_share_one_state_without_mutating_fits():
    fits = _flights()[::-1]
    before = copy.deepcopy(fits)
    report = trajectory_connection_report(fits)
    assert report["valid"]
    assert report["checked_junctions"] == 3
    assert fits == before
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("gap", [0.002, 0.30, 1.4, 10.0])
def test_one_jump_is_not_hidden_by_a_good_median(gap):
    fits = _flights()
    fits[-1]["start_xyz"][0] += gap
    report = trajectory_connection_report(fits)
    assert not report["valid"]
    assert report["reasons"] == ["physics_junction"]


@pytest.mark.parametrize("gap", [-0.1, 0.1, 10.0])
def test_nonmatching_times_cannot_skip_a_junction(gap):
    fits = _flights()
    fits[1]["start_frame"] += gap
    report = trajectory_connection_report(fits)
    assert not report["valid"]
    assert (
        "physics_junction_time" in report["reasons"]
        or "physics_endpoint_invalid" in report["reasons"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("start_frame", None),
        ("end_frame", float("nan")),
        ("end_frame", float("inf")),
        ("end_frame", -1.0),
        ("end_frame", 0.0),
        ("start_xyz", None),
        ("start_xyz", [0, 1]),
        ("end_xyz", [0, 1, float("nan")]),
        ("end_xyz", [0, 1, float("inf")]),
    ],
)
def test_invalid_endpoint_holds_without_serializing_nonfinite_numbers(key, value):
    fits = _flights()
    fits[0][key] = value
    report = trajectory_connection_report(fits)
    assert report["reasons"] == ["physics_endpoint_invalid"]
    assert report["checked_junctions"] == 0
    json.dumps(report, allow_nan=False)


def test_missing_geometry_and_empty_flights_fail_closed():
    assert not trajectory_connection_report([{}])["valid"]
    assert not trajectory_connection_report([])["valid"]


def test_only_numerical_roundoff_is_tolerated():
    fits = _flights()
    fits[1]["start_frame"] += 1e-8
    fits[1]["start_xyz"][0] += 1e-6
    assert trajectory_connection_report(fits)["valid"]


@pytest.mark.parametrize("fault", ["jump", "time_gap", "missing", "nan"])
def test_default_point_gate_rejects_invalid_connections(fault):
    from cv.pipeline.reconstruction import point_gate

    fits = [dict(fit, rms_px=2.0, speed_kmh=100.0) for fit in _flights()]
    if fault == "jump":
        fits[-1]["start_xyz"][0] += 1.4
    elif fault == "time_gap":
        fits[-1]["start_frame"] += 0.1
    elif fault == "missing":
        del fits[-1]["start_xyz"]
    else:
        fits[-1]["end_xyz"][0] = float("nan")
    decision, reasons = point_gate(
        active_valid=True,
        attempted=len(fits),
        fits=fits,
        smoothing={
            "coverage": 1.0,
            "maximum_gap_seconds": 0.0,
            "long_gaps": 0,
            "repair_rate": 0.0,
            "post_heal_teleport_rate": 0.0,
        },
    )
    assert decision == "hold"
    assert set(reasons) & {"physics_junction", "physics_junction_time", "physics_endpoint_invalid"}
