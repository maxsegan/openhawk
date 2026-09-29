from __future__ import annotations

import math

import numpy as np

from cv.pipeline.event_proposals import (
    EventProposal,
    _innovation_sigma,
    _quadratic_fit,
    image_physics_departures,
    implied_bounces,
    physics_departures,
    pose_swing_proposals,
    proposal_document,
    track_corner_proposals,
)


def test_physics_departure_requires_a_sustained_change_and_names_a_bounce() -> None:
    # A descending path reverses at ground level; the next four positions keep
    # disagreeing with the pre-impact free-flight prediction.
    points = {
        1: (0.0, 0.0, 1.0),
        2: (0.1, 0.0, 0.7),
        3: (0.2, 0.0, 0.4),
        4: (0.3, 0.0, 0.1),
        5: (0.4, 0.0, 0.05),
        6: (0.5, 0.0, 0.35),
        7: (0.6, 0.0, 0.65),
        8: (0.7, 0.0, 0.9),
        9: (0.8, 0.0, 1.1),
    }

    proposals = physics_departures("match__pt0001", points, fps=25.0, residual_m=0.08)

    assert any(
        "bounce" in proposal.kinds and proposal.start_frame <= 5 <= proposal.end_frame
        for proposal in proposals
    )


def test_pose_swing_proposes_contact_at_wrist_apex() -> None:
    rows = {
        "near": {
            9: {"right_wrist_x": 10, "right_wrist_y": 80, "right_wrist_confidence": 1},
            10: {
                "right_wrist_x": 30,
                "right_wrist_y": 40,
                "right_wrist_confidence": 1,
                "right_shoulder_y": 70,
                "right_shoulder_confidence": 1,
            },
            11: {"right_wrist_x": 10, "right_wrist_y": 80, "right_wrist_confidence": 1},
        }
    }

    proposals = pose_swing_proposals("match__pt0001", rows, fps=25.0, minimum_wrist_speed_px_s=100)

    assert len(proposals) == 1
    assert proposals[0].frame == 10
    assert proposals[0].kinds == ("contact",)
    assert proposals[0].evidence["swing_cue"] == "serve_toss_overhead"


def test_implied_bounce_has_a_broad_time_range_between_contacts() -> None:
    contacts = [
        EventProposal("m__p", 10, 7, 13, ("contact",), "pose_swing", 0.8, {}),
        EventProposal("m__p", 30, 27, 33, ("contact",), "track_corner", 0.8, {}),
    ]

    proposal = implied_bounces(contacts)[0]

    assert proposal.kinds == ("bounce",)
    assert proposal.start_frame < 20 < proposal.end_frame


def test_corner_and_document_are_label_free(tmp_path) -> None:
    proposals = track_corner_proposals(
        "m__p", {1: (0, 0), 2: (5, 0), 3: (5, 5), 4: (5, 10)}, min_turn_degrees=45
    )

    automatic = tmp_path / "automatic.csv"
    automatic.write_text("clip,frame,x,y\n")
    document = proposal_document(proposals, inputs=[str(automatic)])

    assert document["labels_or_reviewed_inputs"] == []
    assert document["proposals"]
    assert document["input_records"][0]["sha256"]


def test_image_physics_names_a_contact_at_a_player_box() -> None:
    # A straight incoming path turns hard at frame 10; the two local image
    # quadratics disagree across the interval and a player box covers it.
    track = {frame: (100.0 + 20.0 * frame, 300.0) for frame in range(1, 11)}
    track.update({frame: (300.0, 300.0 + 25.0 * (frame - 10)) for frame in range(11, 21)})
    boxes = {frame: [(280.0, 280.0, 340.0, 400.0)] for frame in (10, 11)}

    proposals = image_physics_departures("m__p", track, fps=25.0, player_boxes=boxes)

    hit = [row for row in proposals if row.start_frame <= 10.5 <= row.end_frame]
    assert hit and hit[0].kinds == ("contact",)
    assert hit[0].evidence["model"] == "local_image_quadratic"
    assert hit[0].evidence["interval"] == [10, 11]
    assert hit[0].evidence["predicted_x"] is not None


def test_image_physics_speaks_across_a_track_gap() -> None:
    # The composed track loses the ball for four exposures and it reappears on
    # a different heading; the corner witness has nothing to score there.
    track = {frame: (100.0 + 20.0 * frame, 300.0) for frame in range(1, 11)}
    track.update({frame: (300.0, 300.0 + 25.0 * (frame - 10)) for frame in range(15, 25)})

    proposals = image_physics_departures("m__p", track, fps=25.0)

    covering = [row for row in proposals if row.start_frame <= 12.5 <= row.end_frame]
    assert covering, "a departure across the gap must still be proposed"
    assert covering[0].evidence["interval"] == [10, 15]


def test_image_physics_is_quiet_on_a_smooth_flight() -> None:
    track = {frame: (100.0 + 20.0 * frame, 300.0 + 0.8 * frame * frame) for frame in range(1, 30)}

    assert image_physics_departures("m__p", track, fps=25.0) == []


def test_prediction_sigma_grows_with_the_extrapolation_distance() -> None:
    # The in-sample residual of a six-point quadratic is one number; the
    # prediction interval is not, and it must widen away from the fit window.
    frames = list(range(0, 6))
    points = [np.asarray((10.0 * frame, 0.5 * frame * frame)) for frame in frames]

    fit = _quadratic_fit(frames, points, noise_sigma=1.0)

    assert fit is not None
    inside = fit.prediction_sigma([2.5])[0]
    near = fit.prediction_sigma([6.0])[0]
    far = fit.prediction_sigma([8.0])[0]
    assert inside < near < far
    assert far > 4.0 * fit.sigma


def test_innovation_sigma_reads_the_track_jitter_not_the_flight() -> None:
    # A curved but noise-free path has zero third difference; adding a known
    # per-axis jitter must be recovered to within a factor of two.
    rng = np.random.default_rng(11)
    run = list(range(0, 16))
    clean = {frame: np.asarray((8.0 * frame, 0.4 * frame * frame)) for frame in run}
    assert _innovation_sigma(clean, run) == 0.0
    noisy = {frame: value + rng.normal(0.0, 2.0, 2) for frame, value in clean.items()}
    estimate = _innovation_sigma(noisy, run)
    assert estimate is not None and 1.0 < estimate < 4.0


def test_image_physics_is_quiet_on_a_jittery_straight_track() -> None:
    # The regression the prediction interval exists to prevent: a straight
    # flight carrying the composed track's own jitter has an in-sample
    # residual that collapses, so a constant residual floor called every
    # interval a departure.
    rng = np.random.default_rng(3)
    track = {
        frame: (
            100.0 + 18.0 * frame + float(rng.normal(0.0, 1.6)),
            300.0 + 0.7 * frame * frame + float(rng.normal(0.0, 1.6)),
        )
        for frame in range(1, 60)
    }

    proposals = image_physics_departures("m__p", track, fps=25.0)

    assert proposals == []


def test_image_physics_departure_is_reported_in_prediction_sigmas() -> None:
    track = {frame: (100.0 + 20.0 * frame, 300.0) for frame in range(1, 11)}
    track.update({frame: (300.0, 300.0 + 25.0 * (frame - 10)) for frame in range(11, 21)})

    proposals = image_physics_departures("m__p", track, fps=25.0)

    hit = [row for row in proposals if row.evidence["interval"] == [10, 11]]
    assert hit
    evidence = hit[0].evidence
    assert evidence["departure_sigma"] >= evidence["departure_k"]
    forward = evidence["forward_prediction_sigma_px"]
    assert forward[0] < forward[-1], "the bar must rise with the extrapolation distance"
    assert math.isfinite(evidence["forward_noise_sigma_px"])


def test_implied_bounce_ignores_an_ambiguous_corner() -> None:
    ambiguous = [
        EventProposal("m__p", 10, 7, 13, ("contact", "bounce", "net_hit"), "track_corner", 0.8, {}),
        EventProposal(
            "m__p", 30, 27, 33, ("contact", "bounce", "net_hit"), "track_corner", 0.8, {}
        ),
    ]

    assert implied_bounces(ambiguous) == []
