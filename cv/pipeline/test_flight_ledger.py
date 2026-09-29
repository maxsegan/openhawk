import json

from cv.pipeline.flight_ledger import build, classify_flight, net_clearance_m, point_rows
from cv.pipeline.reconstruction import _timeout_point_record


def point() -> dict:
    return {
        "point": "match__pt0001",
        "match_id": "match",
        "clip": "pt0001",
        "fps": 25.0,
        "surface": "hard",
        "active_play_valid": True,
        "smoothing": {"post_heal_teleport_rate": 0.0},
        "interpretation_branches": {"enabled": False},
    }


def test_summary_keeps_attempts_and_matches_without_flights(tmp_path):
    empty = {**point(), "fits": [], "flight_attempts": [], "decision": "hold"}
    second = {**empty, "point": "other__pt0001", "match_id": "other"}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"points_detail": [empty, second]}))
    summary = build([path])["summary"]
    assert summary["points"] == 2
    assert summary["matches"] == 2
    assert summary["flight_bearing_points"] == 0
    assert summary["points_without_flights"] == 2
    assert summary["flights"] == 0
    assert summary["complete_point_candidates"] == 0


def test_summary_does_not_merge_same_point_across_source_reports(tmp_path):
    current = {
        **point(),
        "decision": "retain",
        "flight_attempts": [{"flight_index": 0, "terminal_end": True}],
        "fits": [
            {
                "flight_index": 0,
                "rms_px": 2.0,
                "speed_kmh": 50.0,
                "observation_coverage": 1.0,
                "bounces": [],
            }
        ],
    }
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path in paths:
        path.write_text(json.dumps({"points_detail": [current]}))
    summary = build(paths)["summary"]
    assert summary["points"] == 2
    assert summary["flight_bearing_points"] == 2
    assert summary["points_without_flights"] == 0
    assert summary["complete_point_candidates"] == 2
    assert summary["matches"] == 1


def test_drop_shot_can_be_provisionally_valid():
    status, reasons = classify_flight(
        point(),
        {
            "start_side": "near",
            "end_side": "far",
            "terminal_end": False,
            "start_frame": 10,
            "end_frame": 30,
        },
        {
            "rms_px": 4.0,
            "speed_kmh": 25.0,
            "observation_coverage": 0.8,
            "bounces": [{"frame": 20}],
        },
    )

    assert status == "provisional_valid"
    assert reasons == []


def test_missing_contact_observation_position_does_not_pass_as_zero_error():
    status, reasons = classify_flight(
        point(),
        {"start_side": "near", "end_side": "far"},
        {
            "rms_px": 2.0,
            "speed_kmh": 80.0,
            "observation_coverage": 0.8,
            "start_contact_reprojection": {"position_available": False, "error_px": None},
        },
    )
    assert status != "provisional_valid"
    assert reasons == ["physics_contact_observation_uncovered"]


def test_same_side_contact_pair_routes_to_recovery():
    status, reasons = classify_flight(
        point(),
        {
            "start_side": "near",
            "end_side": "near",
            "terminal_end": False,
            "start_frame": 10,
            "end_frame": 30,
        },
        {
            "rms_px": 4.0,
            "speed_kmh": 80.0,
            "observation_coverage": 0.8,
            "bounces": [],
        },
    )

    assert status == "recoverable"
    assert reasons == ["same_side_contacts"]


def test_anchor_first_uses_independent_witness_without_net_anchor_precondition():
    fit = {
        "fit_method": "anchor_first_height_on_ray_v1",
        "rms_px": 99.0,
        "held_out_reprojection_median_px": 3.0,
        "held_out_reprojection_p90_px": 7.0,
        "net_anchor_available": True,
        "anchor_satisfied": True,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [],
    }
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
        "start_frame": 10,
        "end_frame": 30,
        "intermediate_events": [],
    }

    assert classify_flight(point(), attempt, fit) == ("provisional_valid", [])

    fit["net_anchor_available"] = False
    assert classify_flight(point(), attempt, fit) == ("provisional_valid", [])


def test_anchor_first_rejects_bad_held_out_tail_and_anchor_error():
    fit = {
        "fit_method": "anchor_first_height_on_ray_v1",
        "rms_px": 1.0,
        "held_out_reprojection_median_px": 4.0,
        "held_out_reprojection_p90_px": 18.0,
        "net_anchor_available": True,
        "anchor_satisfied": False,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [],
    }
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
        "start_frame": 10,
        "end_frame": 30,
        "intermediate_events": [],
    }

    status, reasons = classify_flight(point(), attempt, fit)

    assert status == "recoverable"
    assert reasons == ["high_held_out_reprojection_p90", "anchor_violation"]


def test_terminal_flight_with_two_bounces_is_not_rejected_as_multiple_bounces():
    fit = {
        "fit_method": "anchor_first_camera_reprojection_v5_terminal_bounce",
        "rms_px": 1.0,
        "held_out_reprojection_median_px": 3.0,
        "held_out_reprojection_p90_px": 7.0,
        "anchor_satisfied": True,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [{}, {}],
    }
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": True,
        "start_frame": 10,
        "end_frame": 60,
        "intermediate_events": [],
    }

    assert classify_flight(point(), attempt, fit) == ("provisional_valid", [])


def test_unmodeled_net_penetration_routes_to_recovery():
    fit = {
        "rms_px": 4.0,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [],
        "trajectory": [
            {"xyz": [5.5, 9.0, 1.2]},
            {"xyz": [5.5, 14.0, 0.7]},
        ],
    }
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
        "start_frame": 10,
        "end_frame": 30,
        "intermediate_events": [],
    }

    status, reasons = classify_flight(point(), attempt, fit)

    assert status == "recoverable"
    assert reasons == ["physics_net_penetration_without_collision"]
    assert net_clearance_m(fit) < 0.0


def test_net_clearance_prefers_exact_plane_constraint_witness():
    fit = {
        "net_constraint": {
            "mode": "plane_crossing_height_barrier",
            "clearance_m": 0.18,
            "satisfied": True,
        },
        # Saved trajectories are sparsely sampled and may not contain the exact
        # optimized crossing witness.
        "trajectory": [
            {"xyz": [5.5, 9.0, 1.2]},
            {"xyz": [5.5, 14.0, 0.7]},
        ],
    }

    assert net_clearance_m(fit) == 0.18


def test_explicit_net_collision_allows_low_net_crossing():
    fit = {
        "rms_px": 4.0,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [],
        "trajectory": [
            {"xyz": [5.5, 9.0, 1.2]},
            {"xyz": [5.5, 14.0, 0.7]},
        ],
    }
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
        "start_frame": 10,
        "end_frame": 30,
        "intermediate_events": [{"event_type": "net_hit", "frame": 20}],
    }

    status, reasons = classify_flight(point(), attempt, fit)

    assert status == "provisional_valid"
    assert reasons == []


def test_bounce_impact_velocity_slack_is_not_confused_with_frame_time():
    attempt = {
        "start_side": "near",
        "end_side": "far",
        "terminal_end": False,
        "start_frame": 10,
        "end_frame": 30,
    }
    fit = {
        "rms_px": 4.0,
        "speed_kmh": 80.0,
        "observation_coverage": 0.8,
        "bounces": [{}],
        "bounce_impact_velocity_slack_mps": 10.0,
    }

    status, reasons = classify_flight(point(), attempt, fit)

    assert status == "recoverable"
    assert reasons == ["physics_bounce_impact_velocity_slack"]


def test_spatial_and_bounce_defects_route_to_recovery():
    status, reasons = classify_flight(
        point(),
        {
            "start_side": "near",
            "end_side": "far",
            "terminal_end": False,
            "start_frame": 10,
            "end_frame": 30,
        },
        {
            "rms_px": 4.0,
            "speed_kmh": 100.0,
            "observation_coverage": 1.0,
            "bounces": [{}],
            "minimum_height_m": -0.2,
            "fixed_bounce_continuity_m": 0.5,
            "bounce_speed_ratio": 1.1,
            "start_player_distance_m": 3.0,
            "end_player_distance_m": 1.0,
            "start_contact_reprojection_px": 2.0,
            "end_contact_reprojection_px": 20.0,
            "end_contact_apparent_height_error_m": 0.8,
        },
    )

    assert status == "recoverable"
    assert set(reasons) == {
        "physics_underground",
        "physics_bounce_discontinuity",
        "physics_bounce_energy_gain",
        "physics_contact_reach",
        "physics_contact_reprojection",
        "physics_contact_body_scale",
    }


def test_build_preserves_failed_attempt_denominator(tmp_path):
    report = {
        "points_detail": [
            {
                **point(),
                "attempted_shots": 2,
                "fits": [
                    {
                        "start_frame": 10,
                        "end_frame": 30,
                        "rms_px": 4.0,
                        "speed_kmh": 70.0,
                        "observations": 18,
                        "bounces": [],
                    }
                ],
            }
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(__import__("json").dumps(report))

    ledger = build([path])

    assert ledger["summary"]["flights"] == 2
    assert ledger["summary"]["solved"] == 1
    assert ledger["summary"]["unsolved"] == 1


def test_timeout_point_is_held_and_visible_in_the_ledger():
    timed_out = _timeout_point_record(
        "match__pt0001",
        event_boundary_source="events.json",
        event_consumer_mode="automatic",
        point_timeout_seconds=600.0,
    )

    rows = point_rows(timed_out, "test")

    assert timed_out["decision"] == "hold"
    assert timed_out["reasons"] == ["timeout"]
    assert len(rows) == 1
    assert rows[0]["status"] == "unsolved"
    assert rows[0]["reasons"] == ["timeout"]


def test_complete_point_candidates_require_every_flight_to_pass(tmp_path):
    current_point = {
        **point(),
        "decision": "retain",
        "flight_attempts": [
            {
                "flight_index": 0,
                "start_frame": 10,
                "end_frame": 20,
                "terminal_end": False,
                "start_side": "near",
                "end_side": "far",
            },
            {
                "flight_index": 1,
                "start_frame": 20,
                "end_frame": 30,
                "terminal_end": True,
                "start_side": "far",
                "end_side": None,
            },
        ],
        "fits": [
            {
                "flight_index": 0,
                "rms_px": 4.0,
                "speed_kmh": 70.0,
                "observation_coverage": 0.8,
                "bounces": [],
            },
            {
                "flight_index": 1,
                "rms_px": 4.0,
                "speed_kmh": 50.0,
                "observation_coverage": 0.8,
                "bounces": [],
            },
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"points_detail": [current_point]}))

    assert build([path])["summary"]["complete_point_candidates"] == 1

    current_point["fits"][1]["rms_px"] = 9.0
    path.write_text(json.dumps({"points_detail": [current_point]}))
    assert build([path])["summary"]["complete_point_candidates"] == 0
