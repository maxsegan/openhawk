import copy

import pytest

from cv.validation.owner_spatial_audit import flight_checks, summarize


def _measurement():
    return {
        "point": "match__pt0001",
        "flight_index": 0,
        "solved": True,
        "accepted": True,
        "terminal": True,
        "termination_kind": "second_bounce",
        "scored_bounces": 2,
        "bounce_error_m": 0.1,
        "start_contact_error_m": 0.8,
        "trajectory_rms_m": 0.2,
        "trajectory_max_m": 0.8,
        "trajectory_uncovered_frames": 0,
        "trajectory_frames": 30,
        "termination_error_m": 0.1,
        "mid_air_stop": False,
    }


def _inputs():
    truth = {
        "points": [
            {
                "point": "match__pt0001",
                "source_match": "match",
                "flights": [{"flight_index": 0}],
                "trajectory_truth": {"schema": "simulation_trajectory_samples_v1"},
            }
        ]
    }
    report = {
        "points_detail": [
            {
                "point": "match__pt0001",
                "decision": "retain",
                "flight_attempts": [{"flight_index": 0, "terminal_end": True}],
                "terminal_coverage": {"valid": True, "termination_kind": "second_bounce"},
                "fits": [
                    {
                        "flight_index": 0,
                        "rms_px": 1.0,
                        "speed_kmh": 80.0,
                        "observation_coverage": 1.0,
                        "start_frame": 1.0,
                        "end_frame": 40.0,
                        "start_xyz": [1.0, 2.0, 3.0],
                        "end_xyz": [1.0, 3.0, 0.0325],
                        "terminal_end": True,
                    }
                ],
            }
        ]
    }
    historical = {
        "flight_rows": [_measurement()],
        "by_criterion": {"metric_v2": {"complete_points": 0}},
    }
    return truth, report, historical


def test_contacts_can_be_less_precise_than_bounces_without_allowing_trajectory_drift():
    assert all(flight_checks(_measurement()).values())
    for field in ("bounce_error_m", "trajectory_rms_m", "termination_error_m"):
        assert not all(flight_checks({**_measurement(), field: 0.31}).values())
    assert not all(flight_checks({**_measurement(), "start_contact_error_m": 0.92}).values())
    assert not all(flight_checks({**_measurement(), "trajectory_max_m": 0.92}).values())


def test_held_out_pixel_acceptance_does_not_override_a_physical_flight_hold():
    truth, report, historical = _inputs()
    report["points_detail"][0]["fits"][0]["net_clearance_m"] = -0.066
    result = summarize(truth, report, historical)
    assert result["historical_held_out_accepted"] == 1
    assert result["flight_gate_accepted"] == 0
    assert result["qualified_points"] == 0
    assert result["flight_rows"][0]["flight_gate_reasons"] == [
        "physics_net_penetration_without_collision"
    ]


def test_missing_recorded_attempt_cannot_certify_full_flight_acceptance():
    truth, report, historical = _inputs()
    del report["points_detail"][0]["flight_attempts"]
    result = summarize(truth, report, historical)
    assert result["flight_gate_accepted"] == 0
    assert result["qualified_points"] == 0
    assert result["flight_rows"][0]["flight_gate_disposition"] == "unknown"


@pytest.mark.parametrize(
    "field",
    [
        "bounce_error_m",
        "start_contact_error_m",
        "trajectory_rms_m",
        "trajectory_max_m",
        "termination_error_m",
        "trajectory_uncovered_frames",
        "trajectory_frames",
        "mid_air_stop",
    ],
)
def test_missing_required_witnesses_cannot_pass(field):
    row = _measurement()
    del row[field]
    assert not all(flight_checks(row).values())


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -1.0])
def test_nonfinite_or_invalid_spatial_error_fails(invalid):
    assert not all(flight_checks({**_measurement(), "trajectory_rms_m": invalid}).values())


def test_optional_bounce_is_not_required_for_a_true_volley():
    row = {**_measurement(), "scored_bounces": 0, "bounce_error_m": None}
    assert all(flight_checks(row).values())


def test_missing_reconstructions_stay_in_the_point_and_flight_denominators():
    truth, report, historical = _inputs()
    missing = copy.deepcopy(truth["points"][0])
    missing["point"] = "another__pt0001"
    missing["source_match"] = "another"
    truth["points"].append(missing)
    result = summarize(truth, report, historical)
    assert result["points"] == 2 and result["flights"] == 2
    assert result["qualified_points"] == 1
    assert result["by_source_match"][0] == {"source_match": "another", "points": 1, "qualified": 0}


def test_legacy_approximate_truth_cannot_certify_owner_accuracy():
    truth, report, historical = _inputs()
    del truth["points"][0]["trajectory_truth"]
    result = summarize(truth, report, historical)
    assert result["spatially_within_tolerance_points"] == 1
    assert result["qualified_points"] == 0
    assert result["failed_condition_counts"]["recorded_simulation_truth"] == 1


def test_legacy_ending_valid_flag_cannot_certify_a_bounce_above_the_court():
    truth, report, historical = _inputs()
    record = report["points_detail"][0]
    record["terminal_coverage"]["schema"] = "terminal_coverage_v1"
    record["fits"][0]["end_xyz"][2] = 0.12382446393747704
    result = summarize(truth, report, historical)
    assert result["spatially_within_tolerance_points"] == 1
    assert result["qualified_points"] == 0
    assert result["failed_condition_counts"]["court_ending_geometry_consistent"] == 1


@pytest.mark.parametrize("change", [{"decision": "hold"}, {"terminal_coverage": None}])
def test_spatial_accuracy_cannot_override_a_held_or_incomplete_point(change):
    truth, report, historical = _inputs()
    report["points_detail"][0].update(change)
    assert summarize(truth, report, historical)["qualified_points"] == 0


def test_spatial_tolerance_never_permits_a_contact_jump():
    truth, report, historical = _inputs()
    truth["points"][0]["flights"].append({"flight_index": 1})
    historical["flight_rows"].append({**_measurement(), "flight_index": 1})
    record = report["points_detail"][0]
    record["fits"].append(
        {
            "flight_index": 1,
            "start_frame": 40.0,
            "end_frame": 60.0,
            "start_xyz": [1.1, 3.0, 0.03],
            "end_xyz": [1.0, 4.0, 0.03],
        }
    )
    result = summarize(truth, report, historical)
    assert result["spatially_within_tolerance_points"] == 1
    assert result["qualified_points"] == 0
    assert result["failed_condition_counts"]["continuous_connections"] == 1


def test_duplicate_measurements_are_an_error_not_an_overwrite():
    truth, report, historical = _inputs()
    historical["flight_rows"].append(_measurement())
    with pytest.raises(ValueError, match="duplicate"):
        summarize(truth, report, historical)
