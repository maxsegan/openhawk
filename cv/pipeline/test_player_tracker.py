"""Tests for the ByteTrack/BoT-SORT style player tracker."""

from __future__ import annotations

import numpy as np
import pytest

from cv.pipeline.camera_cal import COURT_L, COURT_W, NET_Y
from dataclasses import replace

from cv.pipeline.player_tracker import (
    Detection,
    RevivalBodyHistoryAudit,
    SelectionConfig,
    Track,
    TrackerConfig,
    admissible,
    body_history_class,
    gaps,
    ground_scales,
    iou,
    keep_unique_admissible_frames,
    recover_identity_fragments,
    select_players,
    side_of,
    stitch,
    track_detections,
)

FPS = 25.0
SCALE = 100.0  # native px per court metre in the synthetic scenes


def make_detection(
    frame: int,
    court: tuple[float, float],
    *,
    conf: float = 0.9,
    height_ratio: float = 1.0,
    embedding: np.ndarray | None = None,
) -> Detection:
    """A detection whose box is consistent with a standing person at ``court``."""
    height = 1.80 * SCALE * height_ratio
    x = 100.0 + court[0] * SCALE
    y = 100.0 + court[1] * SCALE
    return Detection(
        frame=frame,
        box=(x - 0.2 * height, y - height, x + 0.2 * height, y),
        conf=conf,
        court=court,
        ground_scale=SCALE,
        embedding=embedding,
        row={"frame": f"f_{frame:04d}.jpg", "conf": conf},
    )


def unit(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=32).astype(np.float32)
    return vector / np.linalg.norm(vector)


def rally(frames: int = 60, *, noise: float = 0.0) -> list[Detection]:
    """Near and far players rallying, plus a line judge and a crouching ball kid."""
    generator = np.random.default_rng(0)
    near_look, far_look = unit(1), unit(2)
    out: list[Detection] = []
    for frame in range(1, frames + 1):
        phase = frame / frames
        jitter = generator.normal(scale=noise, size=4) if noise else np.zeros(4)
        out.append(
            make_detection(
                frame,
                (3.0 + 4.0 * phase + jitter[0], -1.0 + jitter[1]),
                embedding=near_look,
            )
        )
        out.append(
            make_detection(
                frame,
                (7.0 - 4.0 * phase + jitter[2], COURT_L + 0.8 + jitter[3]),
                embedding=far_look,
            )
        )
        # line judge: standing, still, well outside the doubles sideline
        out.append(make_detection(frame, (-2.4, 18.0), conf=0.8, embedding=unit(3)))
        # ball kid: inside the court laterally but crouched by the net post
        out.append(
            make_detection(
                frame, (0.4, NET_Y + 0.9), conf=0.7, height_ratio=0.45, embedding=unit(4)
            )
        )
    return out


def test_iou_and_ground_scale():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    homography = np.diag([1.0 / 50.0, 1.0 / 50.0, 1.0])
    scales = ground_scales(homography, [[1.0, 1.0], [4.0, 4.0]])
    assert scales == pytest.approx([50.0, 50.0])
    assert np.isnan(ground_scales(np.zeros((3, 3)), [[0.0, 0.0]])[0])


def test_tracker_gives_one_track_per_side():
    tracks = track_detections(rally(noise=0.05), fps=FPS)
    chosen = select_players(tracks, fps=FPS, n_frames=60)
    assert sorted(chosen) == ["far", "near"]
    assert side_of(chosen["near"].court_median()[1]) == "near"
    assert side_of(chosen["far"].court_median()[1]) == "far"
    # every frame of the rally is covered by exactly one track per side
    for side in ("near", "far"):
        frames = chosen[side].frames
        assert len(frames) == len(set(frames))
        assert len(frames) >= 55


def test_non_players_are_rejected_by_position_and_size_not_path_length():
    tracks = track_detections(rally(), fps=FPS)
    verdicts = {}
    for track in tracks:
        ok, reason = admissible(track, n_frames=60, selection=SelectionConfig())
        verdicts[round(track.court_median()[0], 1)] = (ok, reason)
    assert verdicts[-2.4] == (False, "outside_court_laterally")
    assert verdicts[0.4] == (False, "wrong_body_size")
    # the two players are stationary in x nowhere near as much as the judge, yet the
    # judge is rejected on position, not on how little it moved
    assert any(ok for ok, _ in verdicts.values())


def test_standing_player_who_never_moves_is_kept():
    """The v2 scorer needed path length; a server who stands still must still qualify."""
    detections = [make_detection(frame, (5.0, -1.0), embedding=unit(1)) for frame in range(1, 61)]
    detections += [
        make_detection(frame, (5.5, COURT_L + 0.5), embedding=unit(2)) for frame in range(1, 61)
    ]
    chosen = select_players(track_detections(detections, fps=FPS), fps=FPS, n_frames=60)
    assert sorted(chosen) == ["far", "near"]
    assert chosen["near"].hits == 60


def test_low_confidence_detections_extend_a_track():
    detections = []
    for frame in range(1, 41):
        conf = 0.9 if frame <= 20 else 0.2
        detections.append(
            make_detection(frame, (4.0 + 0.05 * frame, 1.0), conf=conf, embedding=unit(1))
        )
    tracks = track_detections(detections, fps=FPS, config=TrackerConfig())
    assert len(tracks) == 1
    assert tracks[0].hits == 40


def test_appearance_gate_blocks_a_wrong_association():
    """Two people crossing at the same place keep their own identity."""
    left, right = unit(5), unit(6)
    detections = []
    for frame in range(1, 41):
        detections.append(make_detection(frame, (0.2 * frame, 2.0), embedding=left))
        detections.append(make_detection(frame, (8.0 - 0.2 * frame, 2.0), embedding=right))
    tracks = [t for t in track_detections(detections, fps=FPS) if t.hits > 5]
    assert len(tracks) == 2
    for track in tracks:
        vectors = [d.embedding for d in track.detections]
        assert all(np.allclose(v, vectors[0]) for v in vectors)


def test_stitch_merges_a_detection_dropout():
    embedding = unit(7)
    first = track_detections(
        [make_detection(f, (3.0, 2.0), embedding=embedding) for f in range(1, 21)], fps=FPS
    )[0]
    second = track_detections(
        [make_detection(f, (3.4, 2.0), embedding=embedding) for f in range(30, 51)],
        fps=FPS,
    )[0]
    merged = stitch([first, second], fps=FPS, selection=SelectionConfig(), config=TrackerConfig())
    assert len(merged) == 1
    assert merged[0].hits == 41


@pytest.mark.parametrize("later_conf", [0.7, 0.9, 0.95])
def test_shared_stitch_boundary_keeps_one_original_detection_and_records_choice(later_conf):
    first = track_detections(
        [make_detection(f, (3, 2), embedding=unit(7)) for f in range(1, 21)], fps=FPS
    )[0]
    second = track_detections(
        [make_detection(f, (3.1, 2), conf=later_conf, embedding=unit(7)) for f in range(20, 41)],
        fps=FPS,
    )[0]
    first_frames, second_frames = list(first.frames), list(second.frames)
    result = stitch([second, first], fps=FPS, selection=SelectionConfig(), config=TrackerConfig())
    assert len(result) == 1
    merged = result[0]
    assert merged.frames == list(range(1, 41))
    assert merged.hits == 40
    original = second.detections[0] if later_conf > 0.9 else first.detections[-1]
    assert merged.detections[19] is original
    assert first.frames == first_frames and second.frames == second_frames
    assert merged.stitch_boundaries[0]["frame"] == 20
    assert merged.stitch_boundaries[0]["chosen_fragment"] == (
        "later" if later_conf > 0.9 else "earlier"
    )
    assert len(merged.stitch_boundaries[0]["candidates"]) == 2
    recovered = recover_identity_fragments(
        merged, [], fps=FPS, selection=SelectionConfig(), config=TrackerConfig()
    )
    assert recovered.stitch_boundaries == merged.stitch_boundaries


def test_shared_exposure_is_not_a_gap_allowing_different_boxes_to_merge():
    first = track_detections(
        [make_detection(f, (3, 2), embedding=unit(7)) for f in range(1, 21)], fps=FPS
    )[0]
    second = track_detections(
        [make_detection(f, (3.6, 2), embedding=unit(7)) for f in range(20, 41)], fps=FPS
    )[0]
    assert stitch(
        [first, second], fps=FPS, selection=SelectionConfig(), config=TrackerConfig()
    ) == [first, second]


def test_recovery_fills_anchor_gaps_without_replacing_anchor_frames():
    look = unit(7)
    anchor = track_detections(
        [make_detection(f, (3.0, 2.0), embedding=look) for f in range(1, 41)], fps=FPS
    )[0]
    fragment = track_detections(
        [make_detection(f, (3.2, 2.0), embedding=look) for f in range(30, 51)], fps=FPS
    )[0]
    recovered = recover_identity_fragments(
        anchor,
        [anchor, fragment],
        fps=FPS,
        selection=SelectionConfig(),
        config=TrackerConfig(),
    )

    assert recovered.frames == list(range(1, 51))
    assert next(d for d in recovered.detections if d.frame == 30).court == (3.0, 2.0)


def test_recovery_refuses_a_different_appearance():
    anchor = track_detections(
        [make_detection(f, (3.0, 2.0), embedding=unit(7)) for f in range(1, 31)], fps=FPS
    )[0]
    other = track_detections(
        [make_detection(f, (3.2, 2.0), embedding=unit(8)) for f in range(31, 51)], fps=FPS
    )[0]
    recovered = recover_identity_fragments(
        anchor,
        [anchor, other],
        fps=FPS,
        selection=SelectionConfig(recovery_appearance_max=0.01),
        config=TrackerConfig(),
    )

    assert recovered is anchor


def test_gaps_are_explicit():
    track = Track(track_id=0)
    track.detections = [make_detection(f, (3.0, 2.0)) for f in [1, 2, 3, 8, 9, 20]]
    assert gaps(track, first_frame=1, last_frame=25) == [(4, 7), (10, 19), (21, 25)]


def test_admissible_rejects_a_short_or_far_back_track():
    selection = SelectionConfig()
    short = Track(track_id=0)
    short.detections = [make_detection(f, (5.0, 2.0)) for f in range(1, 4)]
    short.hits = 3
    assert admissible(short, n_frames=100, selection=selection) == (False, "too_few_hits")

    deep = Track(track_id=1)
    deep.detections = [
        make_detection(f, (5.0, COURT_L + selection.depth_slack_m + 2.0)) for f in range(1, 40)
    ]
    deep.hits = 39
    assert admissible(deep, n_frames=100, selection=selection) == (
        False,
        "outside_court_in_depth",
    )

    wide = Track(track_id=2)
    wide.detections = [
        make_detection(f, (COURT_W + selection.lateral_max_m + 1.0, 2.0)) for f in range(1, 40)
    ]
    wide.hits = 39
    assert admissible(wide, n_frames=100, selection=selection) == (
        False,
        "outside_court_laterally",
    )


def bilateral_fixture():
    def track(tid, frames, ratio=1.0):
        dd = [make_detection(f, (4.0 + f * 0.005, -1.0), height_ratio=ratio) for f in frames]
        return Track(
            track_id=tid,
            detections=dd,
            hits=len(dd),
            start_frame=dd[0].frame,
            last_frame=dd[-1].frame,
        )

    return track(1, list(range(1, 11)) + list(range(26, 51))), track(2, range(10, 27))


def bilateral_recover(anchor, candidates):
    from cv.pipeline.player_tracker import recover_bilateral_fragments

    return recover_bilateral_fragments(
        anchor, candidates, n_frames=50, selection=SelectionConfig(), config=TrackerConfig()
    )


def test_bilateral_recovery_fills_only_missing_original_rows_and_preserves_anchors():
    anchor, candidate = bilateral_fixture()
    original = list(anchor.detections)
    got = bilateral_recover(anchor, [candidate])
    assert got.frames == list(range(1, 51))
    assert anchor.detections == original
    by_frame = {d.frame: d for d in got.detections}
    assert all(by_frame[d.frame] is d for d in original)
    assert all(by_frame[d.frame] is d for d in candidate.detections if 10 < d.frame < 26)
    assert got.stitch_boundaries[-1]["recovered_frames"] == list(range(11, 26))


@pytest.mark.parametrize(
    "defect",
    [
        "one_boundary",
        "different_actor",
        "competing_fragments",
        "partial_body",
        "nonfinite_boundary",
    ],
)
def test_bilateral_recovery_refuses_unsupported_identity(defect):
    from dataclasses import replace

    anchor, candidate = bilateral_fixture()
    candidates = [candidate]
    if defect == "one_boundary":
        candidate.detections = candidate.detections[:-1]
    elif defect == "different_actor":
        candidate.detections = [
            replace(d, box=tuple(v + 400 if i % 2 == 0 else v for i, v in enumerate(d.box)))
            for d in candidate.detections
        ]
    elif defect == "competing_fragments":
        candidates.append(replace(candidate, track_id=3, detections=list(candidate.detections)))
    elif defect == "partial_body":
        candidate.detections = [
            make_detection(d.frame, d.court, height_ratio=0.4) for d in candidate.detections
        ]
    else:
        candidate.detections[0] = replace(candidate.detections[0], court=(float("nan"), -1.0))
    assert bilateral_recover(anchor, candidates) is anchor


# ------------------------------------------- default-off lost-revival body history guard

GUARD = TrackerConfig(lost_revival_body_history=True)


def background_player(frames: range) -> list[Detection]:
    """A full-body person far away on every frame, so absent frames still advance the clip."""
    return [make_detection(f, (-4.0, -3.0), embedding=unit(31)) for f in frames]


def spectator_then_player(
    *, history_ratio: float = 0.40, history_frames: int = 10, gap: int = 6
) -> list[Detection]:
    """A small seated figure, a short absence, then a full-body person in the same place."""
    last = history_frames + gap + 20
    seated = [
        make_detection(f, (3.0, 2.0), height_ratio=history_ratio, embedding=unit(3))
        for f in range(1, history_frames + 1)
    ]
    upright = [
        make_detection(f, (3.0, 2.0), height_ratio=1.0, embedding=unit(3))
        for f in range(history_frames + gap, last)
    ]
    return seated + upright + background_player(range(1, last))


def test_no_established_body_history_refuses_a_full_body_revival():
    audit = RevivalBodyHistoryAudit()
    detections = spectator_then_player()
    tracks = track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    assert len(tracks) == len(track_detections(detections, fps=FPS)) + 1 == 3
    small, large = sorted(
        (t for t in tracks if t.court_median()[0] > 0), key=lambda t: t.start_frame
    )
    assert small.size_ratio_median() < SelectionConfig().size_ratio_low
    assert SelectionConfig().size_ratio_low <= large.size_ratio_median()
    assert audit.revival_refusals >= 1
    assert audit.records[0]["kind"] == "revival_refusal"
    assert audit.records[0]["history"] == "undersized"
    # Every raw row survives, split across the identities.
    assert sum(len(t.detections) for t in tracks) == len(detections)


def test_default_off_keeps_the_original_revival_and_row_behaviour():
    detections = spectator_then_player()
    default = track_detections(detections, fps=FPS)
    explicit_off = track_detections(
        detections, fps=FPS, config=TrackerConfig(), selection=SelectionConfig()
    )
    assert len(default) == 2  # the original tracker revives the identity across the gap
    assert [t.frames for t in default] == [t.frames for t in explicit_off]
    assert [t.track_id for t in default] == [t.track_id for t in explicit_off]


def test_established_body_history_survives_a_prolonged_crouch():
    """A real player measured upright keeps its identity through crouched/clipped samples."""
    upright = [
        make_detection(f, (3.0, 2.0), height_ratio=1.0, embedding=unit(5)) for f in range(1, 16)
    ]
    crouched = [
        make_detection(f, (3.0, 2.0), height_ratio=0.45, embedding=unit(5)) for f in range(16, 26)
    ]
    revival = [
        make_detection(f, (3.0, 2.0), height_ratio=1.0, embedding=unit(5)) for f in range(32, 52)
    ]
    audit = RevivalBodyHistoryAudit()
    detections = upright + crouched + revival + background_player(range(1, 52))
    tracks = track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    player = [t for t in tracks if t.court_median()[0] > 0]
    assert len(player) == 1
    assert len(player[0].detections) == len(upright + crouched + revival)
    assert audit.revival_refusals == 0
    assert audit.revival_established_history >= 1


def test_unknown_scale_and_thin_support_leave_matching_unchanged():
    """Non-finite ground scale is unknown evidence, and a short history is unsupported."""
    unknown = [
        replace(d, ground_scale=float("nan")) for d in spectator_then_player(history_frames=10)
    ]
    audit = RevivalBodyHistoryAudit()
    assert len(track_detections(unknown, fps=FPS, config=GUARD, audit=audit)) == len(
        track_detections(unknown, fps=FPS)
    )
    assert audit.revival_refusals == 0

    thin = spectator_then_player(history_frames=SelectionConfig().min_hits - 2)
    thin_audit = RevivalBodyHistoryAudit()
    assert len(track_detections(thin, fps=FPS, config=GUARD, audit=thin_audit)) == len(
        track_detections(thin, fps=FPS)
    )
    assert thin_audit.revival_refusals == 0
    assert thin_audit.revival_unsupported_scale >= 1


def test_oversized_non_player_history_also_refuses_a_full_body_revival():
    audit = RevivalBodyHistoryAudit()
    oversized = spectator_then_player(history_ratio=2.1)
    tracks = track_detections(oversized, fps=FPS, config=GUARD, audit=audit)
    assert len(tracks) == len(track_detections(oversized, fps=FPS)) + 1
    assert audit.revival_refusals >= 1
    assert audit.records[0]["history"] == "oversized"


def test_active_size_class_transition_is_audited_but_never_blocked():
    """The analogous contemporaneous transition stays matched; only revival is guarded."""
    detections = [
        make_detection(f, (3.0, 2.0), height_ratio=0.40, embedding=unit(9)) for f in range(1, 13)
    ] + [make_detection(f, (3.0, 2.0), height_ratio=1.0, embedding=unit(9)) for f in range(13, 25)]
    audit = RevivalBodyHistoryAudit()
    guarded = track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    plain = track_detections(detections, fps=FPS)
    assert [len(t.detections) for t in guarded] == [len(t.detections) for t in plain]
    assert audit.revival_refusals == 0
    assert audit.active_size_class_pairs >= 1
    # Active candidate counts must not exhaust the bounded refusal evidence buffer.
    assert audit.records == []


def test_guard_does_not_stitch_measured_valid_fragments_sharing_an_exposure():
    """Two contemporaneous full-body people stay separate under the guard, as before."""
    detections = [make_detection(f, (3.0, 2.0), embedding=unit(7)) for f in range(1, 21)] + [
        make_detection(f, (-3.0, 2.0), embedding=unit(11)) for f in range(1, 21)
    ]
    audit = RevivalBodyHistoryAudit()
    guarded = track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    plain = track_detections(detections, fps=FPS)
    assert len(guarded) == len(plain) == 2
    assert [sorted(t.frames) for t in guarded] == [sorted(t.frames) for t in plain]
    assert audit.revival_refusals == 0


def test_body_history_class_reports_its_measured_support():
    selection = SelectionConfig()
    established = Track(track_id=0, detections=[make_detection(f, (3.0, 2.0)) for f in range(10)])
    assert body_history_class(established, selection)[:1] == ("established",)
    unknown = Track(
        track_id=1,
        detections=[replace(make_detection(f, (3.0, 2.0)), ground_scale=0.0) for f in range(10)],
    )
    history, median, finite, in_band = body_history_class(unknown, selection)
    assert (history, finite, in_band) == ("unsupported", 0, 0) and np.isnan(median)


def test_run_level_audit_is_bounded_and_survives_rejected_identities():
    audit = RevivalBodyHistoryAudit(record_limit=1)
    detections: list[Detection] = []
    for index, court in enumerate([(3.0, 2.0), (-3.0, 2.0), (1.0, -3.0)]):
        detections += [
            make_detection(f, court, height_ratio=0.40, embedding=unit(20 + index))
            for f in range(1, 11)
        ]
        detections += [
            make_detection(f, court, height_ratio=1.0, embedding=unit(20 + index))
            for f in range(17, 30)
        ]
    detections += background_player(range(1, 30))
    audited = track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    summary = audit.summary()
    assert summary["revival_refusals"] >= 2
    assert len(summary["records"]) == 1 and summary["records_truncated"] >= 1
    # Refused identities are not required to be selectable players.
    assert summary["revival_refusals"] >= sum(
        1 for track in audited if track.size_ratio_median() < SelectionConfig().size_ratio_low
    )


def test_guard_audit_preserves_native_candidate_identity():
    audit = RevivalBodyHistoryAudit(clip="pt0003", observed_frame_range=(1, 30))
    detections = [make_detection(f, (3.0, 2.0), height_ratio=0.4) for f in range(1, 11)]
    detections += [make_detection(f, (3.0, 2.0)) for f in range(17, 30)]
    detections += background_player(range(1, 30))
    track_detections(detections, fps=FPS, config=GUARD, audit=audit)
    refusal = next(r for r in audit.records if r["kind"] == "revival_refusal")
    assert refusal["clip"] == "pt0003"
    assert refusal["prior_observation_frame"] == 10
    assert refusal["candidate_box_native"] == list(
        next(d.box for d in detections if d.frame == refusal["frame"])
    )
    assert refusal["candidate_court_xy_m"] == [3.0, 2.0]
    assert refusal["boundary_distance_m"] == 0.0


def _far_track(tid: int, frames, behind_m: float) -> Track:
    detections = [make_detection(f, (6.0, COURT_L + behind_m)) for f in frames]
    return Track(
        track_id=tid,
        detections=detections,
        hits=len(detections),
        start_frame=detections[0].frame,
        last_frame=detections[-1].frame,
        box=detections[-1].box,
    )


def test_keep_unique_admissible_frames_off_drops_deep_prefix():
    """Winner-take-all keeps the closer, longer identity and drops the deep prefix."""
    deep = _far_track(0, range(1, 40), behind_m=5.4)
    close = _far_track(1, range(30, 90), behind_m=2.4)
    chosen = select_players(
        [deep, close],
        fps=FPS,
        n_frames=90,
        selection=SelectionConfig(keep_unique_admissible_frames=False),
    )
    assert "far" in chosen
    assert chosen["far"].frames == list(range(30, 90))
    assert 10 not in chosen["far"].frames


def test_keep_unique_admissible_frames_default_is_on():
    assert SelectionConfig().keep_unique_admissible_frames is True


def test_keep_unique_admissible_frames_on_recovers_deep_prefix():
    deep = _far_track(0, range(1, 40), behind_m=5.4)
    close = _far_track(1, range(30, 90), behind_m=2.4)
    chosen = select_players(
        [deep, close],
        fps=FPS,
        n_frames=90,
        selection=SelectionConfig(),
    )
    frames = chosen["far"].frames
    assert list(range(1, 30)) == [f for f in frames if f < 30]
    assert list(range(30, 90)) == [f for f in frames if f >= 30]
    assert any(
        row.get("policy") == "keep_unique_admissible_frames_v1"
        for row in chosen["far"].stitch_boundaries
    )


def test_keep_unique_admissible_frames_refuses_line_judge():
    player = _far_track(1, range(30, 90), behind_m=2.4)
    judge = _far_track(2, range(1, 40), behind_m=SelectionConfig().depth_slack_m + 2.0)
    kept = keep_unique_admissible_frames(
        player,
        [player, judge],
        n_frames=90,
        selection=SelectionConfig(keep_unique_admissible_frames=True),
    )
    assert kept.frames == player.frames
    assert kept.stitch_boundaries == []
