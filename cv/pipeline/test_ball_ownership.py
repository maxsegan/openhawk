from __future__ import annotations

from dataclasses import replace

import numpy as np

from cv.pipeline import ball_ownership as bo
from cv.pipeline.ball_motion_tracker import Geometry, MotionConfig, Observation, track_clip

CLIP = "p"
ON = replace(MotionConfig(), primary_ball_ownership=True)


def geometry(frames, *, pan_per_frame: float = 0.0, unreliable=()) -> Geometry:
    homographies = {}
    reliable = {}
    for frame in frames:
        # court x = image x - pan * frame, so a static scene point drifts +pan px per frame.
        homographies[(CLIP, frame)] = np.asarray(
            [[1.0, 0.0, -pan_per_frame * frame], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reliable[(CLIP, frame)] = frame not in set(unreliable)
    return Geometry(homographies, {}, "fixture", reliable)


def rows(samples, segment: int, kind: str = "guide_gated") -> list[dict]:
    """Tracker-shaped rows from native (frame, x, y) samples of one segment."""
    return [
        {
            "clip": CLIP,
            "frame": f"f_{frame:04d}.jpg",
            "x": x / 2.0,
            "y": y / 2.0,
            "track_id": 0,
            "score": 0.8,
            "rank": 0,
            "sources": "coarse_lock+provenance:coarse",
            "detector_support": 0,
            bo.SEGMENT_KEY: segment,
            bo.RESTART_KEY: kind,
        }
        for frame, x, y in samples
    ]


def line(start_frame, count, x0, y0, vx, vy):
    return [(start_frame + i, x0 + vx * i, y0 + vy * i) for i in range(count)]


def outcome(pre, post, geo, config=ON):
    return bo.join_test(pre, post, geo, CLIP, 25.0, config, bo.pan_model(geo))


def owners(annotated):
    return {bo._frame(row): row["owner"] for row in annotated}


def chains(annotated):
    return {bo._frame(row): row["chain_id"] for row in annotated}


# --- join geometry -----------------------------------------------------------------------


def test_straight_continuation_joins_across_short_gaps_and_occlusion() -> None:
    geo = geometry(range(1, 40))
    pre = rows(line(1, 5, 100, 500, 40, -6), 0)
    for gap in (1, 2, 3, 6):
        # Continue the same line after the gap: x(5) = 260, next sample at 260 + 40 * gap.
        post = rows(line(5 + gap, 4, 260 + 40 * gap, 476 - 6 * gap, 40, -6), 1)
        result = outcome(pre, post, geo)
        assert result.outcome == "joined", (gap, result)
        assert result.camera_mode == "compensated"


def test_long_gap_is_uninformative_not_a_join() -> None:
    geo = geometry(range(1, 60))
    pre = rows(line(1, 5, 100, 500, 40, -6), 0)
    post = rows(line(25, 4, 100 + 40 * 24, 500 - 6 * 24, 40, -6), 1)
    result = outcome(pre, post, geo)
    assert result.outcome == "ambiguous"
    assert result.tolerance_native > ON.hotspot_motion_step_native


def test_true_velocity_reversal_at_contact_joins() -> None:
    geo = geometry(range(1, 20))
    # Down-right into the racket, then sharply up-left from the same point (source 148).
    pre = rows([(1, 1600, 780), (2, 1620, 810), (3, 1640, 840)], 0)
    post = rows([(4, 1610, 760), (5, 1570, 660), (6, 1530, 560)], 1)
    result = outcome(pre, post, geo)
    assert result.outcome == "joined", result
    assert result.distance_native < result.tolerance_native


def test_bounce_reverses_vertical_velocity_and_joins() -> None:
    geo = geometry(range(1, 20))
    pre = rows([(1, 500, 700), (2, 530, 740), (3, 560, 780)], 0)
    post = rows([(4, 590, 760), (5, 620, 725), (6, 650, 695)], 1)
    assert outcome(pre, post, geo).outcome == "joined"


def test_unrelated_jump_does_not_join() -> None:
    geo = geometry(range(1, 20))
    pre = rows(line(1, 4, 1700, 750, 64, 28), 0)
    post = rows(line(5, 4, 1600, 840, 6, -1), 1)  # a person 350 px back, walking
    result = outcome(pre, post, geo)
    assert result.outcome == "unjoined"
    assert result.distance_native > 5 * result.tolerance_native


def test_moving_wing_never_joins_a_stationary_wing() -> None:
    geo = geometry(range(1, 20))
    # Extrapolation of the ball passes exactly through the standing object.
    pre = rows(line(1, 4, 100, 500, 50, 0), 0)
    post = rows([(5, 300, 500), (6, 301, 500), (7, 300, 501)], 1)
    result = outcome(pre, post, geo)
    assert result.outcome == "post_wing_stationary"


def test_camera_pan_is_compensated_with_reliable_geometry() -> None:
    geo = geometry(range(1, 20), pan_per_frame=10.0)
    # A static scene point drifts 10 px/frame in the image under this pan.
    static = rows([(f, 300 + 10 * f, 500) for f in range(1, 6)], 0)
    assert bo.motion_state(static, geo, CLIP, bo.pan_model(geo), ON).state == "still"
    # A ball moving 40 px/frame in the scene appears at 50 px/frame; its continuation joins.
    pre = rows([(f, 100 + 50 * f, 500) for f in range(1, 5)], 0)
    post = rows([(f, 100 + 50 * f, 500) for f in range(5, 9)], 1)
    assert outcome(pre, post, geo).outcome == "joined"


def test_unreliable_bracketed_camera_uses_raw_join_with_allowance_then_ambiguous() -> None:
    frames = range(1, 30)
    geo = geometry(frames, unreliable=range(5, 20))
    pan = bo.pan_model(geo)
    assert pan.bound(CLIP, 10) == pan.reliable_bound  # bracketing frames share one camera
    pre = rows(line(6, 3, 1600, 780, 20, 30), 0)
    post = rows(line(9, 3, 1660, 870, -40, -100), 1)
    joined = bo.join_test(pre, post, geo, CLIP, 25.0, ON, pan)
    assert joined.outcome == "joined" and joined.camera_mode == "raw"
    far_post = rows(line(14, 3, 1660 + 20 * 5, 870 + 30 * 5, 20, 30), 1)
    assert bo.join_test(pre, far_post, geo, CLIP, 25.0, ON, pan).outcome == "ambiguous"


def test_unbracketed_unreliable_camera_is_uncertainty_not_static_proof() -> None:
    geo = geometry(range(1, 15), unreliable=range(1, 8))
    pan = bo.pan_model(geo)
    assert pan.bound(CLIP, 3) == float("inf")
    drifting = rows([(f, 300 + 9 * f, 500) for f in range(1, 6)], 0)
    assert bo.motion_state(drifting, geo, CLIP, pan, ON).state == "ambiguous"
    pre = rows(line(1, 3, 100, 500, 40, 0), 0)
    post = rows(line(4, 3, 220, 500, 40, 0), 1)
    assert bo.join_test(pre, post, geo, CLIP, 25.0, ON, pan).outcome == "ambiguous"
    annotated, _ = bo.assign_ownership(pre + post, geo, CLIP, 25.0, ON, pan)
    assert {row["owner"] for row in annotated} == {"ambiguous"}
    assert len(bo.consumed_rows(annotated)) == 6  # kept, never suppressed


def test_wing_with_an_interior_bounce_shrinks_to_the_samples_nearest_the_gap() -> None:
    geo = geometry(range(1, 20))
    # The pre wing contains the bounce at its first sample; a single line fits it badly.
    pre = rows([(1, 500, 780), (2, 530, 740), (3, 560, 700)], 0)
    post = rows([(4, 590, 660), (5, 620, 620), (6, 650, 580)], 1)
    result = outcome(rows([(0, 470, 740)], 0) + pre, post, geo)
    assert result.outcome == "joined"


# --- ownership semantics -----------------------------------------------------------------


def test_unjoined_moving_chain_is_a_candidate_never_primary() -> None:
    geo = geometry(range(1, 30))
    ball = rows(line(1, 8, 1400, 700, 64, 20), 0, "bootstrap")
    player = rows(line(9, 8, 1600, 840, 6, -1), 1)
    annotated, summary = bo.assign_ownership(ball + player, geo, CLIP, 25.0, ON)
    owner = owners(annotated)
    assert all(owner[f] == "primary" for f in range(1, 9))
    assert all(owner[f] == "candidate" for f in range(9, 17))
    assert chains(annotated)[8] != chains(annotated)[9]
    assert summary["primary_chain"] == 0
    # Candidates are consumed (identity is carried by track_id, not by dropping rows).
    consumed = bo.consumed_rows(annotated)
    assert len(consumed) == 16
    assert {row["track_id"] for row in consumed} == {0, 1}


def test_static_object_after_a_moving_chain_is_withheld() -> None:
    geo = geometry(range(1, 40))
    ball = rows(line(1, 8, 1400, 700, 64, 20), 0, "bootstrap")
    mark = rows([(f, 858 + (f % 2), 858) for f in range(9, 27)], 1)
    annotated, summary = bo.assign_ownership(ball + mark, geo, CLIP, 25.0, ON)
    owner = owners(annotated)
    assert all(owner[f] == "stationary" for f in range(9, 27))
    assert [bo._frame(row) for row in bo.consumed_rows(annotated)] == list(range(1, 9))
    assert summary["withheld_rows"] == 18
    assert summary["join_outcomes"] == {"post_wing_stationary": 1}


def test_ball_that_rolls_to_rest_stays_one_moving_chain() -> None:
    geo = geometry(range(1, 60))
    flight = line(1, 10, 400, 600, 30, 10)
    rest = [(f, 700, 700) for f in range(11, 50)]
    annotated, _ = bo.assign_ownership(rows(flight + rest, 0, "bootstrap"), geo, CLIP, 25.0, ON)
    assert {row["owner"] for row in annotated} == {"primary"}


def test_true_exit_then_moving_distractor_end_to_end() -> None:
    """Native pattern of source 26: ball leaves right, guide lands on a walking player."""
    frames = {}
    for f in range(1, 9):
        frames[f] = [Observation(1400 + 64 * f, 700 + 14 * f, 0.8, 0, ("coarse_lock",))]
    for f in range(9, 20):
        frames[f] = [Observation(1600 + 6 * (f - 9), 840, 0.8, 0, ("coarse_lock",))]
    geo = geometry(frames)
    plain, _ = track_clip(CLIP, frames, geo, 25.0)
    tagged, _ = track_clip(CLIP, frames, geo, 25.0, ON)
    assert [row["frame"] for row in plain] == [row["frame"] for row in tagged]
    assert bo.SEGMENT_KEY not in plain[0] and bo.SEGMENT_KEY in tagged[0]
    annotated, summary = bo.assign_ownership(tagged, geo, CLIP, 25.0, ON)
    owner = owners(annotated)
    assert owner[8] == "primary"  # the wide last ball sample survives
    assert all(owner[f] == "candidate" for f in range(9, 20))
    assert summary["join_outcomes"] == {"unjoined": 1}
    assert summary["restart_kinds"] == {"bootstrap": 1, "guide_gated": 1}
    # The existing image-exit option is the source-backed mechanism for this case and
    # composes with ownership: the distractor frames are then not tracked at all.
    combined, _ = track_clip(CLIP, frames, geo, 25.0, replace(ON, qualified_image_reentry=True))
    assert max(int(row["frame"][2:6]) for row in combined) == 8


def test_contact_reversal_end_to_end_keeps_one_chain() -> None:
    frames = {}
    for f in range(1, 9):
        frames[f] = [Observation(1500 + 20 * f, 600 + 30 * f, 0.8, 0, ("coarse_lock",))]
    for f in range(9, 16):
        frames[f] = [
            Observation(1660 - 40 * (f - 8), 840 - 100 * (f - 8), 0.8, 0, ("coarse_lock",))
        ]
    geo = geometry(frames)
    tagged, _ = track_clip(CLIP, frames, geo, 25.0, ON)
    assert len({row[bo.SEGMENT_KEY] for row in tagged}) == 2  # the IMM did restart
    annotated, summary = bo.assign_ownership(tagged, geo, CLIP, 25.0, ON)
    assert len(set(chains(annotated).values())) == 1
    assert summary["join_outcomes"] == {"joined": 1}
    assert {row["owner"] for row in annotated} == {"primary"}


# --- default-off parity --------------------------------------------------------------------


def test_default_off_leaves_tracker_rows_untouched() -> None:
    assert MotionConfig().primary_ball_ownership is False
    frames = {f: [Observation(100 + 30 * f, 400, 0.8, 0, ("coarse_lock",))] for f in range(1, 12)}
    frames[12] = [Observation(900, 900, 0.8, 0, ("coarse_lock",))]
    geo = geometry(frames)
    rows_default, proposals_default = track_clip(CLIP, frames, geo, 25.0)
    rows_explicit, proposals_explicit = track_clip(CLIP, frames, geo, 25.0, MotionConfig())
    assert rows_default == rows_explicit
    assert proposals_default == proposals_explicit
    assert all(not key.startswith("_") for row in rows_default for key in row)
    assert all(row["track_id"] == 0 for row in rows_default)


def test_geometry_without_reliability_flags_is_reliable_everywhere() -> None:
    geo = Geometry({(CLIP, 1): np.eye(3)}, {}, "legacy")
    assert geo.is_reliable((CLIP, 1))
    assert not geo.is_reliable((CLIP, 2))
