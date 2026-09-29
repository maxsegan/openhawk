import numpy as np

import cv.pipeline.event_topology as event_topology_module

from cv.pipeline.event_topology import (
    build_contact_topology_branches,
    contact_candidate_evidence,
    propose_missing_contact_candidates,
    propose_missing_contact_sequences,
    propose_global_contact_paths,
    score_contact_topology_branch,
    select_contact_topology_branch,
    shortlist_contact_topology_branches,
)


def _contact(frame: float, side: str, seed_id: str) -> dict:
    return {
        "frame": frame,
        "side": side,
        "span": 0,
        "phase": "rally",
        "row": {"seed_id": seed_id, "probability": 0.9},
    }


def test_build_branches_proposes_both_sides_of_same_side_pair() -> None:
    contacts = [
        _contact(10, "far", "a"),
        _contact(30, "far", "b"),
        _contact(55, "near", "c"),
    ]
    ball = {frame: np.array([frame, frame], float) for frame in range(1, 70)}

    branches = build_contact_topology_branches(contacts, ball, 25.0)
    omissions = {row["omitted"][0]["seed_id"] for row in branches if row["omitted"]}

    assert omissions == {"a", "b"}


def _boxes(side: str, frames: range, center: tuple[float, float]) -> dict[int, list[dict]]:
    center_x, center_y = center
    return {
        frame: [
            {
                "side": side,
                "x0": str((center_x - 20) / 2),
                "y0": str((center_y - 35) / 2),
                "x1": str((center_x + 20) / 2),
                "y1": str((center_y + 35) / 2),
            }
        ]
        for frame in frames
    }


def test_missing_contact_requires_same_side_player_reach_and_kink() -> None:
    contacts = [_contact(10, "near", "a"), _contact(60, "near", "b")]
    ball = {
        frame: np.array(
            [
                100 + 8 * frame if frame <= 35 else 380 - 6 * (frame - 35),
                280 - 4 * frame if frame <= 35 else 140 + 5 * (frame - 35),
            ],
            float,
        )
        for frame in range(10, 61)
    }
    boxes = _boxes("far", range(10, 61), (380, 140))

    proposals = propose_missing_contact_candidates(contacts, ball, boxes, 25.0)

    assert proposals
    assert proposals[0]["side"] == "far"
    assert abs(proposals[0]["frame"] - 35) <= 2
    assert proposals[0]["intersection_residual_px"] <= 10

    alternating = [_contact(10, "near", "a"), _contact(60, "far", "b")]
    assert not propose_missing_contact_candidates(alternating, ball, boxes, 25.0)

    far_boxes = _boxes("far", range(10, 61), (900, 600))
    assert not propose_missing_contact_candidates(contacts, ball, far_boxes, 25.0)

    straight_ball = {
        frame: np.array([100 + 6 * frame, 280 - 3 * frame], float) for frame in range(10, 61)
    }
    assert not propose_missing_contact_candidates(contacts, straight_ball, boxes, 25.0)
    assert not propose_missing_contact_candidates(
        contacts,
        ball,
        boxes,
        25.0,
        observed_frames=set(range(10, 36)),
    )


def test_contact_candidate_evidence_names_fail_closed_reason() -> None:
    ball = {frame: np.array([100 + 6 * frame, 280 - 3 * frame], float) for frame in range(10, 61)}
    boxes = _boxes("far", range(10, 61), (380, 140))

    evidence = contact_candidate_evidence(35, "far", ball, boxes, 25.0, set(ball))

    assert not evidence["accepted"]
    assert evidence["rejection_reason"] == "insufficient_velocity_change"
    assert evidence["observations_before"] >= 4
    assert evidence["observations_after"] >= 4


def test_contact_candidate_evidence_rejects_isolated_tracker_aliases() -> None:
    ball = {
        frame: np.array(
            [
                100 + 8 * frame if frame <= 35 else 380 - 6 * (frame - 35),
                280 - 4 * frame if frame <= 35 else 140 + 5 * (frame - 35),
            ],
            float,
        )
        for frame in range(20, 51)
    }
    ball[29] = np.array([900.0, 700.0])
    ball[42] = np.array([80.0, 40.0])
    boxes = _boxes("far", range(20, 51), (380, 140))

    evidence = contact_candidate_evidence(35, "far", ball, boxes, 25.0, set(ball))

    assert evidence["accepted"]
    assert evidence["inliers_before"] < evidence["observations_before"]
    assert evidence["inliers_after"] < evidence["observations_after"]
    assert abs(evidence["frame"] - 35) <= 3


def test_missing_contact_sequences_include_three_alternating_contacts() -> None:
    contacts = [_contact(5, "near", "a"), _contact(95, "near", "b")]
    vertices = [(5, 100, 300), (28, 260, 170), (50, 120, 260), (72, 280, 160), (95, 140, 280)]
    ball = {}
    for (left_frame, left_x, left_y), (right_frame, right_x, right_y) in zip(
        vertices, vertices[1:]
    ):
        for frame in range(left_frame, right_frame + 1):
            fraction = (frame - left_frame) / (right_frame - left_frame)
            ball[frame] = np.array(
                [left_x + fraction * (right_x - left_x), left_y + fraction * (right_y - left_y)],
                float,
            )
    boxes = _boxes("far", range(5, 96), (260, 170))
    near_boxes = _boxes("near", range(5, 96), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)

    sequences = propose_missing_contact_sequences(contacts, ball, boxes, 25.0)

    assert any(
        row["contact_count"] == 3
        and [contact["side"] for contact in row["contacts"]] == ["far", "near", "far"]
        and all(
            abs(contact["frame"] - expected) <= 3
            for contact, expected in zip(row["contacts"], (28, 50, 72))
        )
        for row in sequences
    )


def test_missing_contact_sequences_include_two_between_opposite_boundaries() -> None:
    contacts = [_contact(5, "near", "a"), _contact(72, "far", "b")]
    vertices = [(5, 100, 300), (28, 260, 170), (50, 120, 260), (72, 280, 160)]
    ball = {}
    for (left_frame, left_x, left_y), (right_frame, right_x, right_y) in zip(
        vertices, vertices[1:]
    ):
        for frame in range(left_frame, right_frame + 1):
            fraction = (frame - left_frame) / (right_frame - left_frame)
            ball[frame] = np.array(
                [left_x + fraction * (right_x - left_x), left_y + fraction * (right_y - left_y)],
                float,
            )
    boxes = _boxes("far", range(5, 73), (260, 170))
    near_boxes = _boxes("near", range(5, 73), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)

    sequences = propose_missing_contact_sequences(contacts, ball, boxes, 25.0)

    assert any(
        row["contact_count"] == 2
        and [contact["side"] for contact in row["contacts"]] == ["far", "near"]
        and all(
            abs(contact["frame"] - expected) <= 3
            for contact, expected in zip(row["contacts"], (28, 50))
        )
        for row in sequences
    )


def test_global_contact_path_recovers_chain_without_s5_anchors() -> None:
    vertices = [(5, 100, 300), (28, 260, 170), (50, 120, 260), (72, 280, 160)]
    ball = {}
    for (left_frame, left_x, left_y), (right_frame, right_x, right_y) in zip(
        vertices, vertices[1:]
    ):
        for frame in range(left_frame, right_frame + 1):
            fraction = (frame - left_frame) / (right_frame - left_frame)
            ball[frame] = np.array(
                [left_x + fraction * (right_x - left_x), left_y + fraction * (right_y - left_y)],
                float,
            )
    boxes = _boxes("far", range(5, 73), (260, 170))
    near_boxes = _boxes("near", range(5, 73), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)

    paths = propose_global_contact_paths(ball, boxes, 25.0, [[5, 72]])

    assert paths
    assert any(
        len(path["contacts"]) >= 2
        and all(
            left["side"] != right["side"]
            for left, right in zip(path["contacts"], path["contacts"][1:])
        )
        for path in paths
    )


def test_global_contact_path_allows_slow_three_second_flight() -> None:
    vertices = [(5, 100, 300), (28, 260, 170), (103, 120, 260), (126, 280, 160)]
    ball = {}
    for (left_frame, left_x, left_y), (right_frame, right_x, right_y) in zip(
        vertices, vertices[1:]
    ):
        for frame in range(left_frame, right_frame + 1):
            fraction = (frame - left_frame) / (right_frame - left_frame)
            ball[frame] = np.array(
                [left_x + fraction * (right_x - left_x), left_y + fraction * (right_y - left_y)],
                float,
            )
    boxes = _boxes("far", range(5, 127), (260, 170))
    near_boxes = _boxes("near", range(5, 127), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)

    paths = propose_global_contact_paths(ball, boxes, 25.0, [[5, 126]])

    assert any(
        any(
            right["frame"] - left["frame"] > 2.2 * 25.0
            for left, right in zip(path["contacts"], path["contacts"][1:])
        )
        for path in paths
    )


def test_global_path_snaps_leaky_contact_hypothesis_to_nearby_reacquired_ball() -> None:
    ball = {
        29: np.array([260.0, 170.0]),
        54: np.array([120.0, 260.0]),
    }
    boxes = _boxes("far", range(27, 32), (260, 170))
    near_boxes = _boxes("near", range(52, 57), (120, 260))
    boxes.update(near_boxes)
    hypotheses = [
        {"event_type": "contact", "frame": 24.0, "probability": 0.30},
        {"event_type": "contact", "frame": 49.0, "probability": 0.30},
    ]

    paths = propose_global_contact_paths(
        ball,
        boxes,
        25.0,
        [[1, 70]],
        event_hypotheses=hypotheses,
    )

    assert paths
    assert any(
        [contact["frame"] for contact in path["contacts"]] == [29.0, 54.0]
        and all(
            contact["evidence_source"] == "leaky_s5_hypothesis_temporal_reach_snap"
            for contact in path["contacts"]
        )
        for path in paths
    )


def test_global_path_penalizes_foot_level_contact_when_bounce_probability_dominates() -> None:
    ball = {
        29: np.array([260.0, 170.0]),
        54: np.array([120.0, 290.0]),
        79: np.array([260.0, 170.0]),
    }
    boxes = _boxes("far", range(27, 82), (260, 170))
    near_boxes = _boxes("near", range(52, 57), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)
    hypotheses = [
        {"event_type": "contact", "frame": 29.0, "probability": 0.30},
        {"event_type": "contact", "frame": 54.0, "probability": 0.30},
        {"event_type": "bounce", "frame": 54.0, "probability": 0.80},
        {"event_type": "contact", "frame": 79.0, "probability": 0.30},
    ]

    paths = propose_global_contact_paths(
        ball,
        boxes,
        25.0,
        [[1, 90]],
        event_hypotheses=hypotheses,
    )

    contested = [
        contact
        for path in paths
        for contact in path["contacts"]
        if contact["frame"] == 54.0 and contact["side"] == "near"
    ]
    assert not contested


def test_build_branches_inserts_automatic_contact() -> None:
    contacts = [_contact(10, "near", "a"), _contact(60, "near", "b")]
    ball = {
        frame: np.array(
            [
                100 + 8 * frame if frame <= 35 else 380 - 6 * (frame - 35),
                280 - 4 * frame if frame <= 35 else 140 + 5 * (frame - 35),
            ],
            float,
        )
        for frame in range(10, 61)
    }
    boxes = _boxes("far", range(10, 61), (380, 140))

    branches = build_contact_topology_branches(
        contacts,
        ball,
        25.0,
        boxes=boxes,
        max_branches=12,
    )
    insertion = next(branch for branch in branches if branch["inserted"])

    assert [contact["side"] for contact in insertion["contacts"]] == [
        "near",
        "far",
        "near",
    ]
    assert insertion["contacts"][1]["source"] == "automatic_event_topology_insertion"


def test_build_branches_can_include_multi_contact_sequence() -> None:
    contacts = [_contact(5, "near", "a"), _contact(95, "near", "b")]
    vertices = [(5, 100, 300), (28, 260, 170), (50, 120, 260), (72, 280, 160), (95, 140, 280)]
    ball = {}
    for (left_frame, left_x, left_y), (right_frame, right_x, right_y) in zip(
        vertices, vertices[1:]
    ):
        for frame in range(left_frame, right_frame + 1):
            fraction = (frame - left_frame) / (right_frame - left_frame)
            ball[frame] = np.array(
                [left_x + fraction * (right_x - left_x), left_y + fraction * (right_y - left_y)],
                float,
            )
    boxes = _boxes("far", range(5, 96), (260, 170))
    near_boxes = _boxes("near", range(5, 96), (120, 260))
    for frame, rows in near_boxes.items():
        boxes[frame].extend(rows)

    branches = build_contact_topology_branches(
        contacts,
        ball,
        25.0,
        boxes=boxes,
        max_branches=24,
        include_sequences=True,
    )

    assert any(len(branch["inserted"]) == 3 for branch in branches)


def test_select_branch_requires_margin_and_no_solved_loss() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "score": {
            "total": 50.0,
            "attempted": 4,
            "solved": 3,
            "median_weighted_rms_px": 10.0,
        },
    }
    better = {
        "branch_id": "omit_1",
        "omitted": [{"probability": 0.5}],
        "score": {
            "total": 35.0,
            "attempted": 4,
            "solved": 3,
            "median_weighted_rms_px": 10.0,
        },
    }
    selected, decisive, margin, reason = select_contact_topology_branch([baseline, better])
    assert selected is better
    assert decisive
    assert margin == 15.0
    assert reason == "decisive_improvement"

    worse_coverage = {
        "branch_id": "omit_2",
        "score": {
            "total": 20.0,
            "attempted": 4,
            "solved": 2,
            "median_weighted_rms_px": 10.0,
        },
    }
    selected, decisive, _, reason = select_contact_topology_branch([baseline, worse_coverage])
    assert selected is baseline
    assert not decisive
    assert reason == "alternative_loses_solved_flights"


def test_select_branch_rejects_relative_win_without_absolute_evidence() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "score": {
            "total": 160.0,
            "attempted": 7,
            "solved": 1,
            "median_weighted_rms_px": 78.0,
        },
    }
    insertion = {
        "branch_id": "insert_4",
        "score": {
            "total": 145.0,
            "attempted": 9,
            "solved": 2,
            "median_weighted_rms_px": 68.0,
        },
    }

    selected, decisive, _, reason = select_contact_topology_branch([baseline, insertion])

    assert selected is baseline
    assert not decisive
    assert reason == "insufficient_global_solve_coverage"


def test_select_branch_carries_complete_reprojection_recovery_past_empty_baseline() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "omitted": [],
        "score": {
            "total": 195.0,
            "attempted": 0,
            "solved": 0,
            "median_weighted_rms_px": 100.0,
        },
    }
    recovery = {
        "branch_id": "arc_path",
        "omitted": [],
        "score": {
            "total": 70.0,
            "attempted": 2,
            "solved": 2,
            "median_weighted_rms_px": 54.0,
        },
    }

    selected, decisive, margin, reason = select_contact_topology_branch([baseline, recovery])

    assert selected is recovery
    assert not decisive
    assert margin == 125.0
    assert reason == "reprojection_recovery_candidate"


def test_select_branch_rejects_unsupported_high_confidence_omission() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "score": {
            "total": 110.0,
            "attempted": 3,
            "solved": 1,
            "median_weighted_rms_px": 45.0,
        },
    }
    omission = {
        "branch_id": "omit_2",
        "omitted": [{"probability": 0.99}],
        "score": {
            "total": 90.0,
            "attempted": 2,
            "solved": 1,
            "median_weighted_rms_px": 45.0,
        },
    }

    selected, decisive, _, reason = select_contact_topology_branch([baseline, omission])

    assert selected is baseline
    assert not decisive
    assert reason == "high_confidence_omission_unsupported"


def test_score_penalizes_same_side_and_unsolved_flights() -> None:
    branch = {
        "contacts": [_contact(10, "far", "a"), _contact(30, "far", "b")],
        "prior_cost": 0.0,
    }
    score = score_contact_topology_branch(
        branch,
        attempted=1,
        solved=0,
        weighted_rms_px=[],
        endpoint_errors_px=[],
    )
    assert score["same_side_pairs"] == 1
    assert score["components"]["unsolved"] == 45.0


def test_shortlist_keeps_baseline_and_coverage_safe_best() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "score": {
            "total": 50.0,
            "attempted": 4,
            "solved": 3,
            "median_weighted_rms_px": 10.0,
        },
    }
    unsafe = {
        "branch_id": "unsafe",
        "score": {
            "total": 10.0,
            "attempted": 4,
            "solved": 2,
            "median_weighted_rms_px": 10.0,
        },
    }
    second = {
        "branch_id": "second",
        "score": {
            "total": 30.0,
            "attempted": 4,
            "solved": 3,
            "median_weighted_rms_px": 10.0,
        },
    }
    best = {
        "branch_id": "best",
        "score": {
            "total": 20.0,
            "attempted": 4,
            "solved": 4,
            "median_weighted_rms_px": 10.0,
        },
    }

    shortlisted = shortlist_contact_topology_branches(
        [unsafe, second, baseline, best],
        width=3,
    )

    assert [row["branch_id"] for row in shortlisted] == [
        "contacts_baseline",
        "best",
        "second",
    ]


def test_shortlist_refines_coverage_gain_before_reprojection_rejection() -> None:
    baseline = {
        "branch_id": "contacts_baseline",
        "score": {
            "total": 195.0,
            "attempted": 0,
            "solved": 0,
            "median_weighted_rms_px": 100.0,
        },
    }
    recoverable = {
        "branch_id": "arc_path",
        "score": {
            "total": 93.0,
            "attempted": 2,
            "solved": 2,
            "median_weighted_rms_px": 93.0,
        },
    }

    shortlisted = shortlist_contact_topology_branches([baseline, recoverable], width=2)

    assert [row["branch_id"] for row in shortlisted] == ["contacts_baseline", "arc_path"]


def test_global_path_preserves_strong_hypothesis_timing_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        event_topology_module,
        "propose_global_contact_paths",
        lambda *_args, **_kwargs: [
            {
                "span": 0,
                "score": 1.0,
                "contacts": [
                    {
                        "frame": 210.0,
                        "hypothesis_frame": 209.0,
                        "probability": 0.98,
                        "side": "near",
                        "player_box_frame": 210,
                    },
                    {
                        "frame": 229.0,
                        "hypothesis_frame": 228.0,
                        "probability": 0.89,
                        "side": "far",
                        "player_box_frame": 229,
                    },
                ],
            }
        ],
    )

    branches = build_contact_topology_branches(
        [],
        {frame: np.array([float(frame), 100.0]) for frame in range(200, 235)},
        25.0,
        include_sequences=True,
        span_ranges=[[190.0, 240.0]],
    )

    timed = next(row for row in branches if row["branch_id"].endswith("hypothesis_timing"))
    assert [row["frame"] for row in timed["contacts"]] == [209.0, 228.0]
