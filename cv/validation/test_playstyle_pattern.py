"""The frozen ``playstyle_pattern_v1`` boundaries, bands and accept rules."""

from __future__ import annotations

import math

import pytest

from cv.validation import playstyle_pattern as pp


def test_frozen_boundaries_are_the_published_numbers():
    assert pp.CONTRACT["depth_boundaries_from_receiving_net_m"] == [6.40, 9.1425, 11.885]
    assert pp.CONTRACT["boundary_band_m"] == 0.5
    assert pp.CONTRACT["net_clearance_sigma_m"] == 0.10
    assert pp.CONTRACT["player_tolerance_m"] == 0.30
    assert (
        pp.CONTRACT["player_running_tolerance_m"],
        pp.CONTRACT["player_airborne_tolerance_m"],
    ) == (
        0.60,
        0.90,
    )
    assert pp.CONTRACT["timing_bracketed_frames"] == 1.0
    assert pp.CONTRACT["timing_unbracketed_frames"] == 2.0
    assert pp.SERVE_THIRD_M == pytest.approx(4.115 / 3.0)
    assert pp.CORRIDOR_BOUNDS_M == pytest.approx((4.1133333333, 6.8566666667))


def test_depth_zones_split_at_the_service_line_and_the_midpoint():
    far = "far"
    assert pp.depth_zone(5.485, pp.NET_Y_M + 3.0, far)["class"] == "short"
    assert pp.depth_zone(5.485, pp.NET_Y_M + 8.0, far)["class"] == "mid"
    assert pp.depth_zone(5.485, pp.NET_Y_M + 10.5, far)["class"] == "deep"
    assert pp.depth_zone(5.485, pp.NET_Y_M + 12.5, far)["class"] == "out_long"
    near = pp.depth_zone(5.485, pp.NET_Y_M - 8.0, "near")
    assert near["class"] == "mid" and near["depth_from_net_m"] == pytest.approx(8.0)


def test_a_bounce_inside_the_band_admits_its_neighbour_and_only_its_neighbour():
    on_line = pp.depth_zone(5.485, pp.NET_Y_M + 6.35, "far")
    assert on_line["boundary"] is True
    assert set(on_line["allowed"]) == {"short", "mid"}
    clear = pp.depth_zone(5.485, pp.NET_Y_M + 7.5, "far")
    assert clear["boundary"] is False and clear["allowed"] == ["mid"]


def test_out_long_beats_out_wide_and_a_wide_ball_inside_the_baseline_is_out_wide():
    assert pp.depth_zone(0.2, pp.NET_Y_M + 13.0, "far")["class"] == "out_long"
    assert pp.depth_zone(0.2, pp.NET_Y_M + 8.0, "far")["class"] == "out_wide"


def test_in_out_counts_the_ball_touching_the_singles_line_as_in():
    edge_x = pp.CENTRE_X_M + pp.SINGLES_HALF_WIDTH_M + pp.BALL_RADIUS_M - 1e-6
    assert pp.in_out(edge_x, pp.NET_Y_M + 8.0, "far")["class"] == "in"
    assert pp.in_out(edge_x + 0.2, pp.NET_Y_M + 8.0, "far")["class"] == "out"


def test_deuce_and_ad_follow_the_receiving_end():
    assert pp.court_side(9.0, "near")["class"] == "deuce"
    assert pp.court_side(9.0, "far")["class"] == "ad"
    assert pp.court_side(2.0, "near")["class"] == "ad"


def test_direction_is_frame_free_and_a_centre_hitter_is_not_forced_binary():
    assert pp.direction(2.0, 2.5)["class"] == "down_the_line"
    assert pp.direction(2.0, 9.0)["class"] == "cross_court"
    assert pp.direction(2.0, 5.485)["class"] == "centre"
    assert pp.direction(5.485, 9.0)["class"] == "from_centre_to_right"
    assert pp.direction(2.0, 9.0)["signed_lateral_displacement_m"] == pytest.approx(7.0)


def test_serve_thirds_run_from_the_centre_line_outward_and_a_long_serve_is_a_fault():
    # A near-end server hitting into the far player's deuce box lands at x < 5.485,
    # because the far player faces the other way.
    box, third, call = pp.serve_placement(4.9, pp.NET_Y_M + 4.0, "near")
    assert (box["class"], third["class"], call["class"]) == ("deuce", "T", "in")
    assert pp.serve_placement(6.0, pp.NET_Y_M + 4.0, "near")[0]["class"] == "ad"
    _, third, _ = pp.serve_placement(5.485 + 3.5, pp.NET_Y_M + 4.0, "near")
    assert third["class"] == "wide"
    _, _, call = pp.serve_placement(4.9, pp.NET_Y_M + 7.5, "near")
    assert call["class"] == "fault" and call["fault_direction"] == "long"


def test_volley_is_a_contact_with_no_bounce_since_the_previous_one():
    assert pp.shot_class(has_preceding_bounce=False, is_serve=False)["class"] == "volley"
    assert pp.shot_class(has_preceding_bounce=True, is_serve=False)["class"] == "groundstroke"
    assert pp.shot_class(has_preceding_bounce=False, is_serve=True)["class"] == "serve"


def test_net_outcome_is_bounded_inside_one_sigma():
    assert pp.net_outcome(0.4)["class"] == "cleared"
    assert pp.net_outcome(-0.4)["class"] == "into_net"
    tight = pp.net_outcome(0.05)
    assert tight["class"] == "net_contact"
    assert set(tight["allowed"]) == {"cleared", "net_contact"}
    assert pp.net_tape_height_m(pp.CENTRE_X_M) == pytest.approx(0.914)
    assert pp.net_tape_height_m(pp.CENTRE_X_M + pp.NET_POST_OFFSET_M) == pytest.approx(1.07)


def test_timing_tolerance_needs_a_real_bracket():
    assert pp.timing_tolerance_frames([101.0, 102.0]) == 1.0
    assert pp.timing_tolerance_frames(None) == 2.0
    assert pp.timing_tolerance_frames([101.0, 101.0]) == 2.0


def test_player_zone_and_movement():
    zone = pp.player_zone(5.485, -1.0, "near")
    assert zone["depth_band"]["class"] == "behind_baseline"
    assert pp.player_zone(5.485, 8.0, "near")["depth_band"]["class"] == "net"
    move = pp.player_movement((5.0, -1.0), (5.0, 4.0), "near")
    assert move["class"] == "approach" and move["asserted"] is True
    small = pp.player_movement((5.0, -1.0), (5.4, -1.0), "near")
    assert small["class"] == "holding"
    unasserted = pp.player_movement((5.0, -1.0), (6.5, -1.0), "near")
    assert unasserted["asserted"] is False and "holding" in unasserted["allowed"]


def test_player_position_widened_tolerance_needs_an_unchanged_pattern():
    truth, output = (5.0, 1.0), (5.0, 1.5)
    assert pp.agree_player_position(truth, output)["verdict"] == "wrong"
    assert pp.agree_player_position(truth, output, running=True)["verdict"] == "correct"
    assert (
        pp.agree_player_position(truth, output, running=True, zone_unchanged=False)["verdict"]
        == "wrong"
    )


def test_agree_uses_the_truth_band_and_never_passes_an_abstention():
    truth = pp.depth_zone(5.485, pp.NET_Y_M + 6.35, "far")
    assert pp.agree(truth, {"class": "short"})["verdict"] == "correct"
    assert pp.agree(truth, {"class": "short"})["used_band"] is False
    assert pp.agree(truth, {"class": "mid"})["used_band"] is True
    assert pp.agree(truth, {"class": "deep"})["verdict"] == "wrong"
    assert pp.agree(truth, None)["verdict"] == "no_answer"
    clear = pp.depth_zone(5.485, pp.NET_Y_M + 7.5, "far")
    assert pp.agree(clear, {"class": "short"})["verdict"] == "wrong"


def test_a_two_metre_bounce_error_inside_one_zone_still_fails_on_direction():
    truth = {
        "depth_zone": pp.depth_zone(3.0, pp.NET_Y_M + 10.0, "far"),
        "direction": pp.direction(3.0, 3.0),
    }
    output = {
        "depth_zone": pp.depth_zone(8.0, pp.NET_Y_M + 10.0, "far"),
        "direction": pp.direction(3.0, 8.0),
    }
    score = pp.score_shot(truth, output)
    assert score["fields"]["depth_zone"]["verdict"] == "correct"
    assert score["fields"]["direction"]["verdict"] == "wrong"
    assert score["pattern_correct"] is False


def test_score_shot_treats_a_missing_answer_as_a_failure_not_a_pass():
    truth = {"depth_zone": pp.depth_zone(5.485, pp.NET_Y_M + 8.0, "far")}
    score = pp.score_shot(truth, {})
    assert score["no_answer_fields"] == ["depth_zone"]
    assert score["pattern_correct"] is False


def test_score_point_partials_and_gaps():
    good = {"pattern_correct": True, "failed_fields": [], "no_answer_fields": [], "fields": {}}
    bad = {
        "pattern_correct": False,
        "failed_fields": ["direction"],
        "no_answer_fields": [],
        "fields": {},
    }
    whole = pp.score_point([good, good, good])
    assert whole["pattern_correct_point"] is True and whole["pattern_correct_shots"] == 3
    partial = pp.score_point([good, bad, good])
    assert partial["pattern_correct_point"] is False
    assert partial["partial"] is True and partial["gaps"] == [[0, 0], [2, 2]]
    assert partial["failed_fields"] == ["direction"]
    rejected = pp.score_point([good, good], accepted_flags=[True, False])
    assert rejected["pattern_correct_point"] is False and rejected["pattern_correct_shots"] == 1


def test_transition_tags():
    assert pp.transition_tag(True, True, []) == "unchanged"
    assert pp.transition_tag(False, True, ["depth_zone"]) == "boundary"
    assert pp.transition_tag(False, True, []) == "contract"


def test_distribution_table_keeps_zero_coverage_strata_and_names_the_census():
    rows = [
        {
            "rally_length_band": "3_4_short_rally",
            "surface": "hard",
            "gender": "women",
            "ending_family": "out",
            "accepted": True,
            "pattern_correct": True,
        },
        {
            "rally_length_band": "9_plus_long_rally",
            "surface": "clay",
            "gender": "men",
            "ending_family": "net",
            "accepted": True,
            "pattern_correct": False,
        },
    ]
    table = {row["slice"]: row for row in pp.distribution_table(rows, {"clay": 24})}
    assert table["lobs"]["pool"] == 0 and table["lobs"]["accepted"] == 0
    assert table["clay"]["census"] == 24 and table["hard"]["census"] is None
    assert table["net_endings"]["wrong_accepts"] == 1
    assert table["women"]["pattern_correct"] == 1


def test_binomial_bound_matches_the_published_seventeen_accept_figure():
    bound = pp.binomial_upper_bound(0, 17)
    assert 0.15 < bound < 0.17
    assert pp.binomial_upper_bound(0, 100) < 0.031
    assert math.isclose(pp.binomial_upper_bound(3, 3), 1.0)
