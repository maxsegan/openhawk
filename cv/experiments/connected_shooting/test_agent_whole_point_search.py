from dataclasses import replace

import numpy as np
import pytest

from cv.experiments.connected_shooting import agent_whole_point_search as search
from cv.experiments.connected_shooting import initialization, model, postbounce_initialization


def test_fixed_depth_beam_excludes_only_irrevocably_outside_branches():
    included, excluded = search.fixed_depth_beam(np.arange(-1.5, 3.01, 0.5), (-0.5, 0.5))
    assert included.tolist() == [-0.5, 0.0, 0.5]
    assert [row["depth_hypothesis_m"] for row in excluded] == [
        -1.5,
        -1.0,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    ]
    assert all(row["refit_cannot_change_fixed_depth"] for row in excluded)


def test_fixed_depth_beam_fails_when_bounds_exclude_every_branch():
    with pytest.raises(ValueError, match="excluded every"):
        search.fixed_depth_beam(np.array([20.0, 20.5]), (23.0, 24.5))


def camera():
    return np.array([[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]])


def test_terminal_rebound_fallback_omits_net_groups_only_from_seed_copy(monkeypatch):
    scene, _ = model.control()
    scene = replace(
        scene,
        dynamics="measured_240hz",
        net_hit_frames=(np.array([8.0]), np.array([])),
    )

    def refuse(*args, **kwargs):
        raise ValueError("image initializer needs two pre-bounce training pictures")

    def seed(seed_scene, bounces):
        assert seed_scene.net_hit_frames is None
        assert bounces == [None, None]
        return np.zeros(18), {"method": "test"}

    monkeypatch.setattr(initialization, "image_ballistic_seed", refuse)
    monkeypatch.setattr(postbounce_initialization, "seed", seed)
    result, receipt = search.baseline_seed(
        None,
        scene,
        (np.array([]), np.array([])),
        observation_fallback=True,
        terminal_rebound=True,
    )

    assert result.shape == (20,)
    assert receipt["terminal_rebound_seed_net_hits_omitted"] is True
    assert scene.net_hit_frames is not None


def test_observation_gaps_are_accounted_as_physics_bridges_not_endings():
    result = search.observation_accounting(
        [
            {"frame": 10, "status": "visible"},
            {"frame": 11, "status": "occluded"},
            {"frame": 12, "status": "ambiguous"},
            {"frame": 13, "status": "visible"},
            {"frame": 15, "status": "out_of_frame"},
        ]
    )
    assert result["status_counts"] == {
        "visible": 2,
        "occluded": 1,
        "ambiguous": 1,
        "out_of_frame": 1,
    }
    assert result["abstention_runs"] == [
        {"start_frame": 11, "end_frame": 12, "statuses": ["ambiguous", "occluded"]},
        {"start_frame": 15, "end_frame": 15, "statuses": ["out_of_frame"]},
    ]
    assert "never imply termination" in result["gap_policy"]


def test_family_contract_can_replace_legacy_height_gate_explicitly():
    candidate = {
        "depth_hypothesis_m": 0.0,
        "evidence": {
            "survived": True,
            "checks": {"connected_input_physics": True, "serve_depth_plausible": True},
            "contact_xyz_m": [[1.0, 0.0, 3.4]],
            "input_only_rank_score": 1.0,
        },
    }
    legacy = search.search_reporting.family(
        [candidate],
        arm="legacy",
        eligible=lambda row: row["evidence"]["survived"],
        rank_score=lambda row: row["evidence"]["input_only_rank_score"],
    )
    athlete = search.search_reporting.family(
        [candidate],
        arm="athlete",
        eligible=lambda row: row["evidence"]["survived"],
        rank_score=lambda row: row["evidence"]["input_only_rank_score"],
        required_checks=("connected_input_physics", "serve_depth_plausible"),
    )
    assert not legacy["reconstructed_on_family_basis"]
    assert athlete["reconstructed_on_family_basis"]
    assert athlete["required_checks"] == ["connected_input_physics", "serve_depth_plausible"]


def test_fractional_bounce_target_averages_bracketing_native_ground_rays():
    result = search.event_ground_target(
        {"frame": 10.5, "frame_interval": [10, 11]},
        {10: camera(), 11: camera()},
        {10: np.array([950.0, 530.0]), 11: np.array([970.0, 550.0])},
    )
    assert result["native_frames"] == [10, 11]
    assert result["xyz_m"][2] == pytest.approx(search.whole.BALL_RADIUS_M)
    assert result["bracket_radius_m"] > 0


def test_fractional_bounce_target_has_explicit_graded_subframe_arm():
    result = search.event_ground_target(
        {"frame": 10.0, "frame_interval": [9, 11]},
        {frame: camera() for frame in (8, 9, 11, 12)},
        {
            8: np.array([940.0, 680.0]),
            9: np.array([950.0, 690.0]),
            11: np.array([950.0, 690.0]),
            12: np.array([940.0, 680.0]),
        },
        {frame: 3.0 for frame in (8, 9, 11, 12)},
        mode="subframe_graded_circle",
    )
    assert result["witness_mode"] == "subframe_graded_circle"
    assert result["xyz_m"][2] == pytest.approx(search.whole.BALL_RADIUS_M)
    assert result["graded_circle_radius_m"] >= 0.20
    assert result["target_shift_from_legacy_m"] > 0
    assert result["construction_mode"] == "two_sided_reversal"
    assert result["impact_epoch_witness"]["frame"] == pytest.approx(10.0)
    np.testing.assert_allclose(result["impact_pixel_xy"], [960.0, 700.0])
    covariance = np.asarray(result["ground_covariance_xy_m2"])
    assert covariance.shape == (2, 2) and np.linalg.eigvalsh(covariance)[0] >= -1e-12


def test_two_front_chord_is_retained_only_when_neither_wing_can_be_fitted():
    result = search.event_ground_target(
        {"frame": 10.25, "frame_interval": [10, 11]},
        {10: camera(), 11: camera()},
        {10: np.array([950.0, 530.0]), 11: np.array([970.0, 550.0])},
        {10: 3.0, 11: 5.0},
        mode="subframe_graded_circle",
    )
    assert "frozen two-front chord" in result["construction"]
    assert "construction_mode" not in result
    assert result["xyz_m"] == result["legacy_subframe_witness"]["xyz_m"]


def test_graded_subframe_arm_records_fallback_for_abstained_endpoint():
    result = search.event_ground_target(
        {"frame": 10.5, "frame_interval": [9, 10]},
        {8: camera(), 9: camera()},
        {8: np.array([940.0, 520.0]), 9: np.array([950.0, 530.0])},
        {8: 3.0, 9: 3.0},
        mode="subframe_graded_circle",
    )
    assert not result["subframe_available"]
    assert result["construction_mode"] == "one_sided_incoming"
    assert result["impact_epoch_witness"]["source"] == "supplied_epoch"
    assert "extrapolated" in result["construction"]
    assert result["uncertainty_sigma_m"] > 0
    assert result["graded_circle_radius_m"] >= 0.20


def test_candidate_evidence_keeps_all_contacts_and_bounces_in_gate(monkeypatch):
    monkeypatch.setattr(
        search.whole,
        "directional_support",
        lambda *_: {"maximum_window_rms_px": 2.0},
    )
    measurement = {
        "contact_xyz": [[4.0, 0.0, 2.8], [6.0, 24.0, 1.2]],
        "dense_flights": [
            {"bounces": [{"x": [5.0, 16.0, search.whole.BALL_RADIUS_M]}]},
            {"bounces": [{"x": [8.0, 5.0, search.whole.BALL_RADIUS_M]}]},
        ],
        "physical": {"compatible": True},
        "native_projection": [],
        "rms_px": {"training": 2.0, "withheld": 3.0},
    }
    players = [
        {"court_centre_xy_m": [4.0, 0.0]},
        {"court_centre_xy_m": [6.0, 24.0]},
    ]
    targets = [
        {"xyz_m": [5.0, 16.0, search.whole.BALL_RADIUS_M]},
        {"xyz_m": [8.0, 5.0, search.whole.BALL_RADIUS_M]},
    ]
    result = search.candidate_evidence(measurement, players, targets, (-0.75, 0.9144), [])
    assert result["survived"]
    assert result["bounce_horizontal_errors_m"] == [0.0, 0.0]
    assert result["bounce_gate_errors_m"] == [0.0, 0.0]
    assert result["selector_uses_withheld_pixels"] is False


def test_graded_bounce_circle_scores_only_excess_outside_free_radius(monkeypatch):
    monkeypatch.setattr(
        search.whole,
        "directional_support",
        lambda *_: {"maximum_window_rms_px": 2.0},
    )
    measurement = {
        "contact_xyz": [[4.0, 0.0, 2.8]],
        "dense_flights": [{"bounces": [{"x": [1.0, 0.0, search.whole.BALL_RADIUS_M]}]}],
        "physical": {"compatible": True},
        "native_projection": [],
        "rms_px": {"training": 2.0, "withheld": 3.0},
    }
    result = search.candidate_evidence(
        measurement,
        [{"court_centre_xy_m": [4.0, 0.0]}],
        [{"xyz_m": [0.0, 0.0, search.whole.BALL_RADIUS_M], "graded_circle_radius_m": 0.2}],
        (-0.75, 0.9144),
        [],
    )
    assert result["bounce_horizontal_errors_m"] == [1.0]
    assert result["bounce_gate_errors_m"] == [0.8]
    assert result["survived"]


def test_contact_player_association_alternates_sides(monkeypatch):
    calls = []

    def fake_state(
        _pose,
        _clip,
        frame,
        _pixel,
        *,
        required_side=None,
        image_coordinate_scale=1.0,
        player_name=None,
        stature_m=None,
    ):
        assert image_coordinate_scale == 2.0
        calls.append(required_side)
        return {
            "frame": frame,
            "side": "near" if required_side is None else required_side,
            "court_centre_xy_m": [0.0, 0.0],
            "player": player_name,
            "stature_m": stature_m,
        }

    monkeypatch.setattr(search.single, "server_state", fake_state)
    events = [
        {"frame": value, "frame_interval": [int(value), int(value) + 1]}
        for value in (10.5, 20.5, 30.5, 40.5)
    ]
    labels = {int(value) + 1: np.array([1.0, 2.0]) for value in (10.5, 20.5, 30.5, 40.5)}
    result = search.alternating_player_states(
        None, "clip", events, labels, image_coordinate_scale=2.0
    )
    assert calls == [None, "far", "near", "far"]
    assert [row["side"] for row in result] == ["near", "far", "near", "far"]


def test_soft_athlete_arm_reports_old_caps_without_rejecting_them(monkeypatch):
    monkeypatch.setattr(
        search.whole,
        "directional_support",
        lambda *_: {"maximum_window_rms_px": 2.0},
    )
    measurement = {
        "contact_xyz": [[5.0, -0.5, 3.427]],
        "dense_flights": [{"bounces": [{"x": [2.0, 10.0, search.whole.BALL_RADIUS_M]}]}],
        "physical": {"compatible": True},
        "native_projection": [],
        "rms_px": {"training": 2.0, "withheld": 3.0},
    }
    player = {
        "player": "Zverev",
        "stature_m": 1.98,
        "court_centre_xy_m": [5.0, 0.0],
        "pose_wrist_witness": {"status": "abstained"},
    }
    result = search.candidate_evidence(
        measurement,
        [player],
        [{"xyz_m": [2.0, 10.0, search.whole.BALL_RADIUS_M], "graded_circle_radius_m": 0.2}],
        (-0.75, 0.9144),
        [],
        athlete_prior_mode="stature_pose_soft",
    )
    assert not result["legacy_global_cap_diagnostics"]["serve_height_plausible"]
    assert result["athlete_soft_priors"]["serve_height"]["residual"] == 0
    assert result["survived"]


def test_prepare_attempt_preserves_two_terminal_bounce_knots():
    labels = [
        {"frame": frame, "status": "visible", "x1080": 960.0, "y1080": 540.0}
        for frame in range(11, 31)
    ]
    attempt = {
        "point_clip": "pt0002",
        "match_id": "rg2024",
        "owner_ball_labels": labels,
        "visible_native_frames": len(labels),
        "events": [
            {"event_type": "contact", "frame": 10.5},
            {"event_type": "bounce", "frame": 20.5},
            {"event_type": "bounce", "frame": 30.5},
        ],
        "owner_end_frame": 30.5,
        "fps": 25.0,
    }
    cameras = {
        "clip": "pt0002",
        "match_id": "rg2024",
        "cameras": [
            {"frame": frame, "status": "supported", "P": camera().tolist()}
            for frame in range(11, 31)
        ],
    }
    cameras["cameras"].append({"frame": 9, "status": "unsupported", "reason": "close_up"})
    scene, heldout, bounces, native, outside = search.prepare_attempt(attempt, cameras, "clay")
    assert scene.surface == heldout.surface == "clay"
    assert bounces[-1].tolist() == [20.5, 30.5]
    assert sum(map(len, native)) == len(labels)


def test_prepare_attempt_allows_no_bounce_before_volley_contact():
    labels = [
        {"frame": frame, "status": "visible", "x1080": 960.0, "y1080": 540.0}
        for frame in range(11, 31)
    ]
    attempt = {
        "point_clip": "pt",
        "match_id": "match",
        "owner_ball_labels": labels,
        "events": [
            {"event_type": "contact", "frame": 10.5},
            {"event_type": "contact", "frame": 20.5},
            {"event_type": "bounce", "frame": 30.5},
        ],
        "owner_end_frame": 30.5,
        "fps": 25.0,
    }
    cameras = {
        "clip": "pt",
        "match_id": "match",
        "cameras": [
            {"frame": frame, "status": "supported", "P": camera().tolist()}
            for frame in range(11, 31)
        ],
    }
    _, _, bounces, _, _ = search.prepare_attempt(attempt, cameras, "hard")
    assert bounces[0].tolist() == []


def test_contact_player_association_uses_visible_bracket_neighbor():
    event = {"frame": 10.5, "frame_interval": [10, 11]}
    labels = {9: np.array([1.0, 2.0]), 10: np.array([3.0, 4.0])}
    assert search.contact_association_pixel(event, labels).tolist() == [3.0, 4.0]


def test_first_contact_epoch_prior_extends_early_only_with_picture_support():
    scene, _ = search.model.control()
    supported = search.first_contact_epoch_prior(
        {"frame": 1.0, "frame_interval": [0.5, 1.0]}, scene, {"status": "abstained"}
    )
    assert supported["bounds_frames"] == [0.0, 1.0]
    assert supported["initial_frame"] == 0.25
    assert supported["early_extension_support"] == ["first_two_postcontact_pictures"]
    # Removing the second early outgoing picture leaves only the labeled bracket.
    sparse_scene = search.replace(
        scene, observation_frames=(np.array([1.0]), scene.observation_frames[1])
    )
    unsupported = search.first_contact_epoch_prior(
        {"frame": 1.0, "frame_interval": [0.5, 1.0]},
        sparse_scene,
        {"status": "abstained"},
    )
    assert unsupported["bounds_frames"] == [0.5, 1.0]
    assert unsupported["initial_frame"] == 1.0
    assert unsupported["early_extension_support"] == []


def test_intersection_epoch_starts_inside_label_bracket_without_extension():
    scene, _ = search.model.control()
    result = search.first_contact_epoch_prior(
        {"frame": 1.0, "frame_interval": [0.5, 1.0]},
        scene,
        {"status": "supported", "rows": [{"frame": 0}]},
        initial_frame=0.63,
        stay_inside_labeled_bracket=True,
    )
    assert result["bounds_frames"] == [0.5, 1.0]
    assert result["initial_frame"] == pytest.approx(0.63)
    assert result["stays_inside_labeled_bracket"]


def test_optional_toss_adapter_failure_becomes_an_abstention():
    result = search.abstained_toss_observations(
        ValueError("contact bracket mismatch"), 95.0, (94.5, 95.5)
    )

    assert result["status"] == "abstained"
    assert result["rows"] == []
    assert result["contact_frame_interval"] == [94.5, 95.5]
    assert result["abstention_reason"] == "ValueError: contact bracket mismatch"


def test_missing_feet_history_falls_back_to_the_sided_contact_player():
    result = search.contact_player_feet_fallback(
        {
            "side": "far",
            "frame": 95,
            "court_centre_xy_m": [4.3, 23.9],
            "image_association_distance_px": 12.0,
            "image_coordinate_scale": 2.0,
        },
        ValueError("no history"),
    )

    assert result["side"] == "far"
    assert result["court_xy_m"] == [4.3, 23.9]
    assert result["history_status"] == "contact_player_state_fallback"
    assert result["fallback_reason"] == "ValueError: no history"


def test_labeled_serve_fault_requires_modeled_and_witnessed_faults():
    passed = search.serve_ending_consistency(
        "serve_fault_long", "far", [6.2, 3.0, 0.0325], [6.1, 3.2, 0.0325]
    )
    modeled_in = search.serve_ending_consistency(
        "serve_fault_long", "far", [6.2, 7.4, 0.0325], [6.1, 3.2, 0.0325]
    )
    witness_in = search.serve_ending_consistency(
        "serve_fault_long", "far", [6.2, 3.0, 0.0325], [6.1, 7.4, 0.0325]
    )

    assert passed["passed"]
    assert passed["modeled_call"]["class"] == "fault"
    assert not modeled_in["passed"]
    assert not witness_in["passed"]


def test_nonfault_ending_does_not_activate_the_serve_consistency_gate():
    result = search.serve_ending_consistency(
        "return_wide", "far", [6.2, 7.4, 0.0325], [6.1, 7.4, 0.0325]
    )

    assert not result["applicable"]
    assert result["passed"]


def test_serve_start_prior_combines_toss_and_stays_in_service_lane(monkeypatch):
    prior = {
        "status": "supported",
        "components": [
            {
                "weight": 1.0,
                "mean_xyz_m": [4.0, 23.6, 2.9],
                "covariance_xyz_m2": np.diag([0.04, 0.09, 0.01]).tolist(),
            }
        ],
        "mode_component_index": 0,
    }
    combined = {
        **prior,
        "schema": "tennis_serve_location_toss_intersection_v1",
        "toss_intersection": "supported_gaussian_product",
    }
    monkeypatch.setattr(search.serve_location, "serve_location_prior", lambda *a, **k: prior)
    monkeypatch.setattr(search.serve_location, "intersect_toss_estimate", lambda *a: combined)

    result, bounds = search.serve_start_location_prior(
        {"match_id": "match"},
        {"side": "far", "player": "P", "stature_m": 1.9},
        {"xyz_m": [7.0, 8.0, search.model.R_BALL]},
        1,
        {"status": "supported", "contact_xyz_m": [4.1, 23.8, 2.8]},
    )

    assert result["toss_intersection"] == "supported_gaussian_product"
    assert bounds["serve_side"] == "deuce"
    assert bounds["x_interval_m"][1] <= search.playstyle_pattern.CENTRE_X_M
    assert 23.17 <= bounds["y_interval_m"][0] < bounds["y_interval_m"][1] <= 24.77
    assert not bounds["optimizer_inequality_added"]


def test_serve_start_prior_ignores_a_toss_product_outside_the_service_lane(monkeypatch):
    prior = {
        "status": "supported",
        "components": [
            {
                "weight": 1.0,
                "mean_xyz_m": [4.0, 23.6, 2.9],
                "covariance_xyz_m2": np.diag([0.04, 0.09, 0.01]).tolist(),
            }
        ],
        "mode_component_index": 0,
    }
    incompatible = {
        **prior,
        "components": [
            {
                "weight": 1.0,
                "mean_xyz_m": [7.0, 20.0, 2.9],
                "covariance_xyz_m2": np.diag([0.0001, 0.0001, 0.01]).tolist(),
            }
        ],
    }
    monkeypatch.setattr(search.serve_location, "serve_location_prior", lambda *a, **k: prior)
    monkeypatch.setattr(search.serve_location, "intersect_toss_estimate", lambda *a: incompatible)

    result, bounds = search.serve_start_location_prior(
        {"match_id": "match"},
        {"side": "far", "player": "P", "stature_m": 1.9},
        {"xyz_m": [7.0, 8.0, search.model.R_BALL]},
        1,
        {"status": "supported", "contact_xyz_m": [7.0, 20.0, 2.9]},
    )

    assert result["toss_intersection"] == "abstained_toss_three_sigma_misses_service_lane"
    assert bounds["target_xyz_m"] == [4.0, 23.6, 2.9]


def test_player_state_records_sided_disagreement_and_pose_wrist(tmp_path):
    pose = tmp_path / "pose.csv"
    pose.write_text(
        "clip,frame,side,x0,y0,x1,y1,court_x,court_y,left_wrist_x,left_wrist_y,"
        "left_wrist_confidence,right_wrist_x,right_wrist_y,right_wrist_confidence\n"
        "pt,f_0010.jpg,near,0,0,100,200,5,0,45,40,0.9,80,80,0.8\n"
        "pt,f_0010.jpg,far,40,0,140,200,5,24,90,50,0.9,110,80,0.8\n"
    )
    state = search.single.server_state(
        pose,
        "pt",
        10,
        np.array([50.0, 45.0]),
        required_side="far",
        player_name="P",
        stature_m=1.8,
    )
    assert state["side"] == "far"
    assert state["pixel_nearest_side"] == "near"
    assert state["sided_file_disagrees_with_pixel_nearest"]
    assert state["pose_wrist_witness"]["status"] == "supported"
