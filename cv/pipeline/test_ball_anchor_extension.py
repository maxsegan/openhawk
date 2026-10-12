from __future__ import annotations

from dataclasses import replace

import numpy as np

from cv.pipeline import ball_anchor_extension as ax
from cv.pipeline import ball_ownership as bo
from cv.pipeline.ball_motion_tracker import Geometry, MotionConfig, Observation

CLIP = "p"
ON = replace(MotionConfig(), primary_ball_ownership=True)


def geometry(frames, *, pan_per_frame: float = 0.0) -> Geometry:
    homographies = {}
    for frame in frames:
        # court x = image x - pan * frame: a static scene point drifts +pan px per frame.
        homographies[(CLIP, frame)] = np.asarray(
            [[1.0, 0.0, -pan_per_frame * frame], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    return Geometry(homographies, {}, "fixture", {})


def owned(samples, *, chain=0, owner="primary", motion="moving", segment=0) -> list[dict]:
    """Ownership-annotated rows from native (frame, x, y) samples."""
    rows = []
    for frame, x, y in samples:
        row = {
            "clip": CLIP,
            "frame": f"f_{frame:04d}.jpg",
            "x": x / 2.0,
            "y": y / 2.0,
            "track_id": chain,
            "score": 0.8,
            "rank": 0,
            "sources": "sliding_tracknetv2+sliding_wasb",
            "regime": "ballistic",
            "confidence": 0.1,
            "detector_support": 2,
            "homography_source": "fixture",
        }
        row.update(dict.fromkeys(bo.OWNERSHIP_COLUMNS, ""))
        row.update(
            {"segment_id": segment, "chain_id": chain, "owner": owner, "chain_motion": motion}
        )
        rows.append(row)
    return rows


def obs(x, y, sources=("sliding_wasb",), score=0.3) -> Observation:
    return Observation(float(x), float(y), score, 0, tuple(sources))


def line(start, count, x0, y0, vx, vy):
    return [(start + i, x0 + vx * i, y0 + vy * i) for i in range(count)]


def pool(points, noise=()):
    frames: dict[int, list[Observation]] = {}
    for frame, x, y in points:
        frames.setdefault(frame, []).append(obs(x, y))
    for frame, x, y in noise:
        frames.setdefault(frame, []).append(obs(x, y, ("sliding_tracknetv2",)))
    return frames


def added(extended, consumed):
    before = {row["frame"] for row in consumed}
    return [row for row in extended if row["frame"] not in before]


def test_backward_continuation_from_post_contact_anchor():
    # Anchor f133-135 moves up-left; candidates f126-132 lie on that line, plus far noise.
    anchor = line(133, 3, 1140, 389, -28, -27)
    rows = owned(anchor)
    ball = [(f, 1140 + 28 * (133 - f), 389 + 27 * (133 - f)) for f in range(126, 133)]
    noise = [(f, 300 + 5 * f, 900) for f in range(126, 133)]
    geo = geometry(range(120, 140))
    extended, side, decisions = ax.extend_clip(rows, pool(ball, noise), geo, CLIP, ON)
    consumed = bo.consumed_rows(rows)
    new = added(extended, consumed)
    assert [bo._frame(r) for r in new] == list(range(126, 133))
    for row in new:
        assert "support:anchor_extension_bwd" in row["sources"]
        assert "anchor_segment:0" in row["sources"]
        assert row["track_id"] == 0
        assert row["regime"] == ax.REGIME
        assert set(row) == set(consumed[0])
    # Consumed rows are byte-for-byte the same objects, in frame order.
    assert [r for r in extended if r["frame"] in {c["frame"] for c in consumed}] == consumed
    assert [bo._frame(r) for r in extended] == sorted(bo._frame(r) for r in extended)
    assert decisions and any(
        d["verdict"] == "accepted" and d["direction"] == "bwd" for d in decisions
    )
    assert len(side) == 7 and all(s["direction"] == "bwd" for s in side)


def test_real_kink_stops_forward_extension():
    # Pre-contact anchor moving down-right; the path reverses at f126 (a contact).
    anchor = line(120, 3, 1500, 780, 13, 23)
    rows = owned(anchor)
    before = line(123, 3, 1500 + 13 * 3, 780 + 23 * 3, 13, 23)  # f123-125
    after = [(f, 1539 + 13 * 2 - 45 * (f - 125), 849 - 50 * (f - 125)) for f in range(126, 135)]
    geo = geometry(range(118, 140))
    extended, _, decisions = ax.extend_clip(rows, pool(before + after), geo, CLIP, ON)
    new = added(extended, bo.consumed_rows(rows))
    assert [bo._frame(r) for r in new] == [123, 124, 125]
    fwd = next(d for d in decisions if d["direction"] == "fwd")
    assert fwd["verdict"] == "accepted" and fwd["rows_added"] == 3


def test_competing_parallel_chains_abstain():
    anchor = line(100, 3, 500, 500, 20, 0)
    rows = owned(anchor)
    upper = [(f, 560 + 20 * (f - 103), 500 - 12 * (f - 103) + 4) for f in range(103, 110)]
    lower = [(f, 560 + 20 * (f - 103), 500 + 12 * (f - 103) - 4) for f in range(103, 110)]
    geo = geometry(range(95, 115))
    extended, side, decisions = ax.extend_clip(rows, pool(upper + lower), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(rows)) == []
    assert side == []
    fwd = next(d for d in decisions if d["direction"] == "fwd")
    assert fwd["verdict"] == "abstain" and fwd["reason"] == "rival_tie"


def test_no_row_invention_across_detection_gap():
    anchor = line(133, 3, 1140, 389, -28, -27)
    rows = owned(anchor)
    # Only f132 and f126 carry candidates; f127-131 have nothing at all.
    sparse = [(132, 1168, 416), (126, 1140 + 28 * 7, 389 + 27 * 7)]
    geo = geometry(range(120, 140))
    extended, side, decisions = ax.extend_clip(rows, pool(sparse), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(rows)) == []
    bwd = next(d for d in decisions if d["direction"] == "bwd")
    assert bwd["verdict"] == "abstain" and bwd["reason"].startswith("too_short")
    # Three observed frames then nothing: exactly three rows, never a predicted one.
    three = [(f, 1140 + 28 * (133 - f), 389 + 27 * (133 - f)) for f in (130, 131, 132)]
    extended, side, _ = ax.extend_clip(rows, pool(three), geo, CLIP, ON)
    assert [bo._frame(r) for r in added(extended, bo.consumed_rows(rows))] == [130, 131, 132]
    for row in side:
        assert float(row["innovation_px"]) <= float(row["gate_px"])


def test_camera_pan_static_clutter_is_neither_anchor_nor_extension():
    pan = 7.0
    geo = geometry(range(90, 130), pan_per_frame=pan)
    # A world-static object drifts +7 px/frame in the image: it must not anchor anything.
    static_anchor = owned([(100 + i, 800 + pan * i, 400) for i in range(3)])
    drift = [(103 + i, 800 + pan * (3 + i), 400) for i in range(10)]
    extended, _, decisions = ax.extend_clip(static_anchor, pool(drift), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(static_anchor)) == []
    assert decisions == []
    # A moving anchor whose only continuation candidates are world-static clutter abstains.
    moving = owned(line(100, 3, 800, 400, 30, 0))
    clutter = [(103 + i, 890 + pan * i, 400) for i in range(10)]  # sits still in the world
    extended, _, decisions = ax.extend_clip(moving, pool(clutter), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(moving)) == []
    assert all(d["verdict"] == "abstain" for d in decisions)
    # And a slow real ball under the same pan is still followed (no speed > pan rule).
    # 5 px/frame in the world (below the 7 px pan, above ownership's 8 px / 3-frame radius).
    slow = owned(line(100, 3, 800, 400, 5 + pan, 0))
    slow_ball = [(103 + i, 800 + (5 + pan) * (3 + i), 400) for i in range(6)]
    extended, _, _ = ax.extend_clip(slow, pool(slow_ball), geo, CLIP, ON)
    assert [bo._frame(r) for r in added(extended, bo.consumed_rows(slow))] == list(range(103, 109))


def test_extension_stops_at_existing_consumed_rows_and_withheld_frames_are_free():
    geo = geometry(range(90, 140))
    ball = owned(line(100, 3, 500, 500, 20, 0), chain=0)
    later = owned(line(110, 3, 700, 500, 20, 0), chain=1, segment=1)  # fixed rows at 110-112
    still = owned(
        [(f, 640, 300) for f in range(103, 110)],
        chain=2,
        owner="stationary",
        motion="still",
        segment=2,
    )
    rows = sorted(ball + still + later, key=bo._frame)
    candidates = [(f, 560 + 20 * (f - 103), 500) for f in range(103, 116)]
    extended, _, decisions = ax.extend_clip(rows, pool(candidates), geo, CLIP, ON)
    new = added(extended, bo.consumed_rows(rows))
    frames = [bo._frame(r) for r in new]
    assert set(frames) <= set(range(103, 110)) | set(range(113, 116))
    assert all(f not in (110, 111, 112) for f in frames)
    kept = [r for r in extended if bo._frame(r) in (110, 111, 112)]
    assert kept == bo.consumed_rows(later)


def test_camera_hold_cannot_turn_static_drift_into_a_moving_extension():
    frames = range(90, 131)
    geo = geometry(frames, pan_per_frame=4.0)
    # A bracketed camera hold gives a finite pan allowance, but cannot distinguish
    # this stationary scene object from a slow ball. The moving anchor is reliable.
    geo = Geometry(
        geo.homographies, {}, "fixture", {(CLIP, f): not 103 <= f <= 120 for f in frames}
    )
    rows = owned(line(100, 3, 500, 500, 8, 0))
    drift = [(f, 524 + 4 * (f - 103), 500) for f in range(103, 121)]
    extended, side, decisions = ax.extend_clip(rows, pool(drift), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(rows)) == []
    assert side == []
    forward = next(d for d in decisions if d["direction"] == "fwd")
    assert forward["verdict"] == "abstain"


def test_extension_cannot_hide_a_stop_inside_an_overall_moving_chain():
    geo = geometry(range(90, 125))
    rows = owned(line(100, 3, 500, 500, 15, 0))
    # Two continuation rows lead into near-stationary clutter. Overall extent is
    # still large, so only the stepwise speed discontinuity can expose the handoff.
    candidates = [(103, 545, 500), (104, 560, 500)] + [
        (f, 560 + 0.5 * (f - 104), 500) for f in range(105, 109)
    ]
    extended, side, _ = ax.extend_clip(rows, pool(candidates), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(rows)) == []
    assert side == []


def test_default_off_parity_and_guide_exclusion():
    rows = owned(line(100, 3, 500, 500, 20, 0))
    consumed = bo.consumed_rows(rows)
    geo = geometry(range(90, 120))
    assert ax.extend_clip(rows, {}, geo, CLIP, ON) == (consumed, [], [])
    guide_only = {
        f: [Observation(560 + 20 * (f - 103), 500, 1.0, 0, ("coarse_lock",))]
        for f in range(103, 110)
    }
    extended, side, _ = ax.extend_clip(rows, guide_only, geo, CLIP, ON)
    assert extended == consumed and side == []


def test_composition_flag_default_off():
    import inspect

    from cv.pipeline import tracking_composition as tc

    assert inspect.signature(tc.run_match).parameters["anchor_extension"].default is False


def test_unmeasured_camera_never_yields_an_infinite_gate():
    # Frames 103+ are unreliable with no reliable frame after them: pan is unmeasurable.
    frames = list(range(90, 130))
    geo = geometry(frames)
    geo = Geometry(geo.homographies, {}, "fixture", {(CLIP, f): f < 103 for f in frames})
    rows = owned(line(100, 3, 500, 500, 20, 0))
    far = [(f, 560 + 20 * (f - 103) + 300, 900) for f in range(103, 110)]
    extended, side, decisions = ax.extend_clip(rows, pool(far), geo, CLIP, ON)
    assert added(extended, bo.consumed_rows(rows)) == [] and side == []
    fwd = next(d for d in decisions if d["direction"] == "fwd")
    assert fwd["verdict"] == "abstain" and fwd["stop_reason"] == "camera_unmeasured"


def test_coincident_guide_cannot_delete_independent_detector_evidence():
    guide = obs(100, 100, ("coarse_lock",), score=0.99)
    measured = obs(101, 101, ("sliding_wasb",), score=0.7)
    assert guide.is_guide
    candidates = ax._observation_pool({1: [guide, measured]}, ON)[1]
    assert len(candidates) == 1
    assert not candidates[0].is_guide
    assert candidates[0].x == measured.x and candidates[0].y == measured.y
    assert candidates[0].sources == measured.sources
