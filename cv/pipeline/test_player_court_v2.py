import numpy as np

from camera_cal import COURT_L, NET_Y
from player_court_v2 import (
    Tracklet,
    build_tracklets,
    lateral_violation,
    score_tracklet,
    select_player_tracklets,
)


def make_cands(seq):
    """seq: {frame: [(cx, cy), ...]} -> candidate dict rows."""
    return {
        f: [
            dict(clip="pt0000", frame=f"f_{f:04d}.jpg", x0="0", y0="0", x1="1", y1="1",
                 conf="0.9", cx=cx, cy=cy)
            for cx, cy in rows
        ]
        for f, rows in seq.items()
    }


def test_static_line_judge_outside_court_is_rejected():
    # judge parked 2 m outside the sideline for 100 frames, no in-court candidates
    seq = {f: [(-2.0, 22.0)] for f in range(1, 101)}
    tracklets = build_tracklets(make_cands(seq), fps=50.0)
    assert len(tracklets) == 1
    assert select_player_tracklets(tracklets, "far", fps=50.0) == []


def test_moving_player_beats_larger_static_judge():
    # player rallies inside the court while a judge sits at x=-2; selection must
    # follow the player and produce no >2 m/frame discontinuities
    seq = {}
    for f in range(1, 201):
        px = 5.0 + 3.0 * np.sin(f / 25.0)
        seq[f] = [(px, 26.0 + np.cos(f / 40.0)), (-2.0, 22.0)]
    tracklets = build_tracklets(make_cands(seq), fps=50.0)
    chosen = select_player_tracklets(tracklets, "far", fps=50.0)
    assert len(chosen) == 1
    xy = np.asarray(chosen[0].xy)
    assert abs(np.median(xy[:, 0]) - 5.0) < 3.5
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert steps.max() < 2.0


def test_velocity_gate_blocks_identity_switch():
    # player detections vanish mid-clip; a ball kid 11 m away appears in the gap.
    seq = {}
    for f in range(1, 61):
        seq[f] = [(8.0, 27.0)]
    for f in range(61, 76):
        seq[f] = [(-2.3, 12.4)]  # net ball kid, inside x margin but off-court laterally
    for f in range(76, 141):
        seq[f] = [(8.2, 27.2)]
    tracklets = build_tracklets(make_cands(seq), fps=50.0)
    chosen = select_player_tracklets(tracklets, "far", fps=50.0)
    frames = sorted(f for t in chosen for f in t.frames)
    pos = {f: t.xy[i] for t in chosen for i, f in enumerate(t.frames)}
    # the kid's frames are simply absent; the two player tracklets are both kept
    assert all(pos[f][0] > 5 for f in frames)
    assert 60 in pos and 76 in pos


def test_score_prefers_in_court():
    static_judge = Tracklet(tid=0, frames=list(range(50)),
                            xy=[(-2.0, 22.0)] * 50, rows=[{}] * 50)
    player = Tracklet(tid=1, frames=list(range(50)),
                      xy=[(5.0 + 0.05 * i, 26.0) for i in range(50)], rows=[{}] * 50)
    assert score_tracklet(player, "far") > 0 > score_tracklet(static_judge, "far")
    assert lateral_violation(-2.0) == 2.0
    assert lateral_violation(5.0) == 0.0


def test_depth_split_matches_net():
    assert NET_Y == COURT_L / 2
