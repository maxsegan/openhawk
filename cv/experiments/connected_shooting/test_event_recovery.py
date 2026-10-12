import numpy as np
import pytest

from cv.experiments.connected_shooting import event_recovery as recovery


def projection(errors, split="training", start=100):
    return [
        {"frame": start + index, "split": split, "error_px": value}
        for index, value in enumerate(errors)
    ]


def test_residual_runs_finds_one_contiguous_unexplained_span():
    rows = projection([1.0, 1.2, 0.9, 30.0, 28.0, 26.0, 1.1, 1.0])
    runs = recovery.residual_runs(rows)
    assert len(runs) == 1
    assert runs[0]["onset_bracket_frames"] == [102.0, 103.0]
    assert runs[0]["pictures"] == 3
    assert runs[0]["reaches_last_training_picture"] is False


def test_residual_runs_reads_no_withheld_picture():
    rows = projection([1.0, 1.0, 1.0, 1.0]) + projection([99.0] * 4, split="withheld")
    assert recovery.residual_runs(rows) == []


def test_a_single_flagged_picture_is_not_a_run():
    assert recovery.residual_runs(projection([1.0, 1.0, 40.0, 1.0, 1.0])) == []


def test_grammar_rejects_a_missing_ending_and_a_double_interior_bounce():
    contact = {"event_type": "contact", "frame": 10.0}
    bounce = {"event_type": "bounce", "frame": 20.0}
    penalty, reasons = recovery.grammar_penalty([contact, bounce])
    assert penalty == float("inf") and reasons == ["requires_one_post_contact_ending"]

    events = [
        {"event_type": "contact", "frame": 10.0},
        {"event_type": "bounce", "frame": 15.0},
        {"event_type": "bounce", "frame": 18.0},
        {"event_type": "contact", "frame": 20.0},
        {"event_type": "bounce", "frame": 30.0},
        {"event_type": "ending", "frame": 30.0},
    ]
    penalty, reasons = recovery.grammar_penalty(events)
    assert penalty == float("inf") and reasons == ["flight_0_has_2_bounces"]


def test_grammar_charges_a_volley_but_allows_it():
    events = [
        {"event_type": "contact", "frame": 10.0},
        {"event_type": "contact", "frame": 20.0},
        {"event_type": "bounce", "frame": 30.0},
        {"event_type": "ending", "frame": 30.0},
    ]
    penalty, reasons = recovery.grammar_penalty(events)
    assert penalty == recovery.GRAMMAR_PENALTIES["volley_contact_without_preceding_bounce"]
    assert reasons == ["flight_0_is_a_volley"]


def test_grammar_accepts_the_second_bounce_terminal_state():
    events = [
        {"event_type": "contact", "frame": 10.0},
        {"event_type": "bounce", "frame": 20.0},
        {"event_type": "bounce", "frame": 28.0},
        {"event_type": "ending", "frame": 30.0},
    ]
    assert recovery.grammar_penalty(events)[0] == 0.0


def supplied():
    return [
        {"event_type": "contact", "frame": 10.0},
        {"event_type": "bounce", "frame": 20.0},
        {"event_type": "contact", "frame": 26.0},
        {"event_type": "bounce", "frame": 35.0},
        {"event_type": "ending", "frame": 40.0},
    ]


def test_topology_hypotheses_always_keep_the_supplied_branch_first():
    proposal = recovery.Proposal(
        frame=33.0,
        frame_interval=(32.0, 34.0),
        kind_likelihoods={"bounce": 0.7, "contact": 0.2, "net_hit": 0.05, "ending": 0.05},
        evidence={},
    )
    branches = recovery.topology_hypotheses(supplied(), [proposal], max_branches=4)
    assert branches[0].name == "supplied"
    assert branches[0].added == [] and branches[0].demoted == []
    names = [row.name for row in branches]
    assert "add_bounce_at_33" in names
    added = next(row for row in branches if row.name == "add_bounce_at_33")
    assert len(added.events) == len(supplied()) + 1
    assert added.added[0]["annotation_origin"] == "recovered"
    assert added.added[0]["frame_interval"] == [32.0, 34.0]


def test_topology_hypotheses_drop_branches_the_grammar_forbids():
    # A second bounce inside the already complete first flight is ungrammatical.
    proposal = recovery.Proposal(
        frame=15.0,
        frame_interval=(14.0, 16.0),
        kind_likelihoods={"bounce": 1.0, "contact": 0.0, "net_hit": 0.0, "ending": 0.0},
        evidence={},
    )
    branches = recovery.topology_hypotheses(supplied(), [proposal], max_branches=6)
    assert [row.name for row in branches] == ["supplied"]


def test_topology_hypotheses_respect_the_branch_budget():
    proposals = [
        recovery.Proposal(
            frame=float(frame),
            frame_interval=(frame - 1.0, frame + 1.0),
            kind_likelihoods={"bounce": 0.5, "contact": 0.5, "net_hit": 0.0, "ending": 0.0},
            evidence={},
        )
        for frame in (30.0, 33.0, 36.0)
    ]
    branches = recovery.topology_hypotheses(supplied(), proposals, max_branches=3)
    assert len(branches) == 3
    with pytest.raises(ValueError, match="at least one topology branch"):
        recovery.topology_hypotheses(supplied(), proposals, max_branches=0)


def test_demoting_a_supplied_event_records_its_reason():
    rows = supplied()
    branches = recovery.topology_hypotheses(rows, [], max_branches=6, demotable=[rows[1]])
    demote = next(row for row in branches if row.name.startswith("demote_"))
    assert demote.demoted[0]["demotion_reason"] == "unexplained_by_physics"
    assert all(row["event_type"] != "bounce" or row["frame"] != 20.0 for row in demote.events)


def test_ending_candidates_use_post_contact_bounces_and_the_last_picture():
    rows = [
        {"event_type": "contact", "frame": 10.0},
        {"event_type": "bounce", "frame": 20.0},
        {"event_type": "bounce", "frame": 28.0},
    ]
    proposals = recovery.ending_candidates(rows, (5.0, 33.0))
    assert [row.frame for row in proposals] == [20.0, 28.0, 33.0]
    assert all(row.kind_likelihoods["ending"] == 1.0 for row in proposals)


def test_match_recovered_events_never_credits_a_wrong_kind_or_a_far_epoch():
    removed = [
        {"event_type": "bounce", "frame": 30.0},
        {"event_type": "contact", "frame": 50.0},
    ]
    recovered = [
        {"event_type": "bounce", "frame": 31.0},
        {"event_type": "bounce", "frame": 50.0},
    ]
    result = recovery.match_recovered_events(recovered, removed)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["mean_absolute_timing_error_frames"] == 1.0


def test_image_turn_measures_a_reversal_and_abstains_without_a_secant():
    labels = {frame: np.array([float(frame), 0.0]) for frame in range(10, 17)}
    for frame in range(17, 24):
        labels[frame] = np.array([32.0 - frame, 0.0])
    angle, _ = recovery.image_turn(16.0, labels)
    assert angle is not None and angle > 90.0
    assert recovery.image_turn(2.0, labels)[0] is None


def test_modeled_height_samples_the_fitted_flight():
    flights = [
        {"start_frame": 10.0, "end_frame": 20.0, "positions": [[0, 0, 1.0], [0, 0, 3.0]]},
    ]
    assert recovery.modeled_height_m(15.0, flights) == pytest.approx(2.0)
    assert recovery.modeled_height_m(25.0, flights) is None


def test_adding_an_event_is_charged_so_the_supplied_topology_is_preferred():
    proposal = recovery.Proposal(
        frame=33.0,
        frame_interval=(32.0, 34.0),
        kind_likelihoods={"bounce": 0.7, "contact": 0.2, "net_hit": 0.05, "ending": 0.05},
        evidence={},
    )
    branches = recovery.topology_hypotheses(supplied(), [proposal], max_branches=4)
    added = next(row for row in branches if row.added)
    assert added.grammar_penalty >= recovery.GRAMMAR_PENALTIES["add_a_recovered_event"]
    assert branches[0].grammar_penalty < added.grammar_penalty


def witnessed_topology():
    return [
        {"event_type": "contact", "frame": 10.0, "frame_interval": [10.0, 10.0]},
        {"event_type": "bounce", "frame": 20.5, "frame_interval": [20.0, 21.0]},
        {"event_type": "bounce", "frame": 40.5, "frame_interval": [40.0, 41.0]},
        {"event_type": "ending", "frame": 45.0, "frame_interval": [45.0, 45.0]},
    ]


def test_an_unwitnessable_bounce_and_contact_are_named_with_their_reason():
    cameras = {frame: np.eye(3, 4) for frame in range(0, 50)}
    labels = {frame: np.array([1.0, 2.0]) for frame in range(0, 50)}
    for frame in (40, 41):
        labels.pop(frame)
    events = witnessed_topology()
    flagged = recovery.unwitnessable_supplied_events(events, cameras, labels)
    assert [(row["event_type"], row["frame"]) for row in flagged] == [("bounce", 40.5)]
    assert flagged[0]["demotion_reason"] == "bounce_interval_has_no_visible_native_ground_ray"

    for frame in (9, 10, 11):
        labels.pop(frame, None)
    flagged = recovery.unwitnessable_supplied_events(events, cameras, labels)
    kinds = {(row["event_type"], row["demotion_reason"]) for row in flagged}
    assert ("contact", "contact_bracket_has_no_nearby_visible_front") in kinds


def test_a_fully_witnessed_topology_flags_nothing():
    cameras = {frame: np.eye(3, 4) for frame in range(0, 50)}
    labels = {frame: np.array([1.0, 2.0]) for frame in range(0, 50)}
    assert recovery.unwitnessable_supplied_events(witnessed_topology(), cameras, labels) == []


def test_an_evidence_backed_demotion_is_tried_before_a_speculative_one():
    rows = supplied()
    flagged = dict(rows[1], demotion_reason="bounce_interval_has_no_visible_native_ground_ray")
    branches = recovery.topology_hypotheses(rows, [], max_branches=2, demotable=[rows[2], flagged])
    assert branches[0].name == "supplied"
    assert branches[1].demoted[0]["frame"] == rows[1]["frame"]
    assert (
        branches[1].demoted[0]["demotion_reason"]
        == "bounce_interval_has_no_visible_native_ground_ray"
    )
    assert all(
        (row["event_type"], row["frame"]) != ("bounce", rows[1]["frame"])
        for row in branches[1].events
    )


def test_a_demotion_that_would_not_change_the_topology_is_dropped():
    rows = supplied()
    absent = {"event_type": "bounce", "frame": 999.0, "demotion_reason": "not_present"}
    branches = recovery.topology_hypotheses(rows, [], max_branches=6, demotable=[absent])
    assert [row.name for row in branches] == ["supplied"]


def test_a_demotion_the_grammar_cannot_close_is_dropped():
    rows = supplied()
    terminal = dict(rows[3], demotion_reason="bounce_interval_has_no_visible_native_ground_ray")
    branches = recovery.topology_hypotheses(rows, [], max_branches=6, demotable=[terminal])
    assert [row.name for row in branches] == ["supplied"]


def test_an_ordinary_demotion_still_competes_with_the_proposals():
    """Pushing every ordinary demotion behind every proposal crowds out the
    demotion of a spurious event, which is the branch that matters most."""
    rows = supplied()
    proposals = [
        recovery.Proposal(
            frame=float(frame),
            frame_interval=(frame - 1.0, frame + 1.0),
            kind_likelihoods={"bounce": 0.5, "contact": 0.5, "net_hit": 0.0, "ending": 0.0},
            evidence={},
        )
        for frame in (30.0, 33.0, 36.0)
    ]
    branches = recovery.topology_hypotheses(rows, proposals, max_branches=6, demotable=[rows[2]])
    names = [row.name for row in branches]
    assert "demote_contact_at_26" in names
    assert names.index("demote_contact_at_26") < len(names)


def net_terminated_topology(ending=120.0):
    """A point that ends in the net: the terminal flight has no ground bounce."""
    return [
        {"event_type": "contact", "frame": 90.0, "frame_interval": [90.0, 91.0]},
        {"event_type": "bounce", "frame": 101.0, "frame_interval": [101.0, 102.0]},
        {"event_type": "contact", "frame": 108.0, "frame_interval": [108.0, 109.0]},
        {"event_type": "net_hit", "frame": ending, "frame_interval": [ending, ending]},
        {"event_type": "ending", "frame": ending, "frame_interval": [ending, ending]},
    ]


def test_a_net_ending_extends_to_its_labeled_dead_ball_bounce():
    supplied = net_terminated_topology()
    dead_ball = [{"event_type": "bounce", "frame": 127.0, "frame_interval": [127.0, 128.0]}]
    branches = recovery.net_termination_hypotheses(supplied, dead_ball, {})
    assert branches[0].name == "extend_to_dead_ball_bounce_at_127"
    events = branches[0].events
    assert [row["frame"] for row in events if row["event_type"] == "ending"] == [127.0]
    assert sum(row["event_type"] == "bounce" and row["frame"] == 127.0 for row in events) == 1
    assert not np.isinf(recovery.grammar_penalty(events)[0])


def test_a_dead_ball_bounce_already_in_the_topology_is_not_duplicated():
    supplied = [
        *net_terminated_topology(),
        {"event_type": "bounce", "frame": 127.0, "frame_interval": [127.0, 128.0]},
    ]
    dead_ball = [{"event_type": "bounce", "frame": 127.0, "frame_interval": [127.0, 128.0]}]
    branches = recovery.net_termination_hypotheses(supplied, dead_ball, {})
    events = branches[0].events
    assert sum(row["event_type"] == "bounce" and row["frame"] == 127.0 for row in events) == 1


def test_an_ambiguous_terminal_epoch_is_read_as_a_ground_impact_only_on_a_reversal():
    supplied = net_terminated_topology()
    descending_then_rising = {
        frame: np.asarray([900.0, 500.0 + (frame - 120.0) * 3.0]) for frame in range(117, 121)
    } | {frame: np.asarray([900.0, 500.0 - (frame - 120.0) * 9.0]) for frame in range(120, 124)}
    branches = recovery.net_termination_hypotheses(supplied, [], descending_then_rising)
    assert branches[0].name == "terminal_ground_impact_at_120"
    assert branches[0].added[0]["image_reversal_witness"]["reverses_downward_motion"] is True

    rising_then_descending = {
        frame: np.asarray([900.0, 500.0 - (frame - 120.0) * 3.0]) for frame in range(117, 121)
    } | {frame: np.asarray([900.0, 500.0 + (frame - 120.0) * 9.0]) for frame in range(120, 124)}
    names = [
        row.name
        for row in recovery.net_termination_hypotheses(supplied, [], rising_then_descending)
    ]
    assert not any(name.startswith("terminal_ground_impact") for name in names)


def test_the_last_resort_truncation_reports_the_events_it_drops():
    branches = recovery.net_termination_hypotheses(net_terminated_topology(), [], {})
    truncation = next(row for row in branches if row.name.startswith("truncate_to_bounce_at"))
    assert [row["frame"] for row in truncation.demoted] == [108.0, 120.0]
    assert any("partial point" in reason for reason in truncation.reasons)
    assert not np.isinf(recovery.grammar_penalty(truncation.events)[0])


def test_a_topology_with_a_terminal_bounce_is_left_alone():
    supplied = [
        {"event_type": "contact", "frame": 90.0},
        {"event_type": "bounce", "frame": 101.0},
        {"event_type": "ending", "frame": 101.0},
    ]
    assert recovery.net_termination_hypotheses(supplied, [], {}) == []


def test_extending_the_window_copies_the_frozen_rows_and_keeps_the_inventory_contiguous():
    attempt = {
        "point_clip": "pt0001",
        "owner_end_frame": 121.0,
        "owner_ball_labels": [{"frame": 121, "status": "visible"}],
        "context_native_frames": [122, 123, 124],
    }
    records = [
        {
            "clip": "pt0001",
            "frames": [
                {"frame": frame, "status": "visible" if frame != 123 else "ambiguous"}
                for frame in range(121, 125)
            ],
        }
    ]
    extended = recovery.extend_attempt_window(attempt, records, 123.0)
    assert [row["frame"] for row in extended["owner_ball_labels"]] == [121, 122, 123]
    assert extended["owner_end_frame"] == 123.0
    assert extended["visible_native_frames"] == 2
    assert extended["context_native_frames"] == [124]
    assert attempt["owner_ball_labels"] == [{"frame": 121, "status": "visible"}]
    with pytest.raises(ValueError, match="contiguous native inventory"):
        recovery.extend_attempt_window(attempt, records, 200.0)


def test_extending_to_the_same_epoch_returns_the_attempt_unchanged():
    attempt = {"point_clip": "pt0001", "owner_end_frame": 121.0}
    assert recovery.extend_attempt_window(attempt, [], 121.0) is attempt


def test_an_uncertain_bounce_is_admitted_only_as_the_truncation_boundary():
    supplied = [
        {"event_type": "contact", "frame": 90.0, "frame_interval": [90.0, 91.0]},
        {"event_type": "contact", "frame": 108.0, "frame_interval": [108.0, 109.0]},
        {"event_type": "net_hit", "frame": 120.0, "frame_interval": [120.0, 120.0]},
        {"event_type": "ending", "frame": 120.0, "frame_interval": [120.0, 120.0]},
    ]
    assert recovery.net_termination_hypotheses(supplied, [], {}) == []
    uncertain = [
        {
            "event_type": "bounce",
            "frame": 101.0,
            "frame_interval": [101.0, 102.0],
            "status": "ambiguous",
        }
    ]
    branches = recovery.net_termination_hypotheses(supplied, [], {}, uncertain_events=uncertain)
    assert [row.name for row in branches] == ["truncate_to_bounce_at_101"]
    bounce = next(row for row in branches[0].events if row["event_type"] == "bounce")
    assert bounce["supplied_status"] == "ambiguous"
    assert bounce["annotation_origin"] == "recovered"
    assert not np.isinf(recovery.grammar_penalty(branches[0].events)[0])
