"""Unit tests for the net-cord (tape) interaction and the `net` classification verdict.

The net plane is a third interaction surface alongside the racket contact and the court
bounce (owner review, 2026-07-20): a ball that crosses the net line below the tape clips it, a real
mid-flight velocity discontinuity. Two pieces:

  1. `fast_simulate` can model a tape clip in its dynamics (behind NET_IN_DYNAMICS, OFF by
     default because it destabilises discovery fits — see the module note): a low crossing
     reverses and heavily damps the court-y velocity, so the ball drops near-dead at the net.
  2. `classify_knots` gives a `net` verdict to a break whose fitted contact court-y is
     depth-ghosted (so the exact net-plane position is unreliable) but whose OBSERVABLE
     signature is unmistakable: a fast incoming flight killed to near-rest at low height in
     the mid-court net band. A racket contact adds or preserves speed; the tape absorbs it.
     A `net` verdict is excluded from the hit chain and from export, kept as a labeled event
     (RICH_CHART_SPEC kind=net).
"""
import numpy as np

import physics_knot_solver as P
from physics_knot_solver import (
    NET_H_CENTER,
    NET_H_POST,
    NET_Y,
    _export_evidence,
    classify_knots,
    fast_simulate,
    net_impact_velocity,
    net_tape_height,
    select_hit_chain,
)

FPS = 50.0


class _FakeCamera:
    def p_at(self, frame):
        return np.array([[1000.0, 0.0, 480.0, 0.0],
                         [0.0, 1000.0, 270.0, -500.0],
                         [0.0, 0.0, 1.0, 0.0]])

    def h_at(self, frame):
        return np.eye(3, dtype=np.float32)


# --------------------------------------------------------------------------- #
# net geometry + impact model
# --------------------------------------------------------------------------- #
def test_tape_height_rises_from_centre_to_posts() -> None:
    assert net_tape_height(P.NET_X_CENTER) == NET_H_CENTER
    # at/beyond the posts it reaches the higher post height
    assert net_tape_height(P.NET_X_CENTER + P.NET_HALF_WIDTH) == NET_H_POST
    assert net_tape_height(-20.0) == NET_H_POST          # clamped
    # monotone in |x - centre|
    assert NET_H_CENTER < net_tape_height(P.NET_X_CENTER + 2.0) < NET_H_POST


def test_net_impact_reverses_and_damps_crossing_velocity() -> None:
    v_out = net_impact_velocity(np.array([10.0, 30.0, 4.0]))
    assert v_out[1] < 0                      # court-y (crossing) reversed
    assert abs(v_out[1]) < 30.0              # ... and heavily damped
    assert abs(v_out[0]) < 10.0              # in-plane x damped
    assert abs(v_out[2]) < 4.0               # in-plane z damped
    assert float(np.linalg.norm(v_out)) < float(np.linalg.norm([10.0, 30.0, 4.0]))


# --------------------------------------------------------------------------- #
# fast_simulate net dynamics (opt-in)
# --------------------------------------------------------------------------- #
def test_low_net_crossing_clips_the_tape_when_dynamics_on() -> None:
    # A ball crossing the net plane low (z ~ 0.4 m, below tape) should be turned back.
    theta = np.array([P.NET_X_CENTER, NET_Y - 2.0, 0.4, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0])
    saved = P.NET_IN_DYNAMICS
    try:
        P.NET_IN_DYNAMICS = True
        pos_on, _ = fast_simulate(theta, 100.0, np.arange(100.0, 140.0), FPS, "clay")
        P.NET_IN_DYNAMICS = False
        pos_off, _ = fast_simulate(theta, 100.0, np.arange(100.0, 140.0), FPS, "clay")
    finally:
        P.NET_IN_DYNAMICS = saved
    # Without the net the ball sails past y=NET_Y; with it, it is held back near the tape.
    assert pos_off[:, 1].max() > NET_Y + 1.0
    assert pos_on[:, 1].max() < pos_off[:, 1].max()


def test_high_net_crossing_passes_untouched() -> None:
    # A ball clearing the net well above the tape (z ~ 2 m) is unaffected either way.
    theta = np.array([P.NET_X_CENTER, NET_Y - 2.0, 2.0, 0.0, 25.0, 2.0, 0.0, 0.0, 0.0])
    saved = P.NET_IN_DYNAMICS
    try:
        P.NET_IN_DYNAMICS = True
        pos_on, _ = fast_simulate(theta, 100.0, np.arange(100.0, 120.0), FPS, "clay")
        P.NET_IN_DYNAMICS = False
        pos_off, _ = fast_simulate(theta, 100.0, np.arange(100.0, 120.0), FPS, "clay")
    finally:
        P.NET_IN_DYNAMICS = saved
    assert np.allclose(pos_on, pos_off, atol=1e-6)


# --------------------------------------------------------------------------- #
# classify_knots — the `net` verdict
# --------------------------------------------------------------------------- #
def _moving_ball(boundary):
    # a real sweeping arc so the dead-ball span gate never fires in these tests
    return {f: (100.0 + (f - boundary) * 6.0, 120.0 + (f - boundary) * 3.0)
            for f in range(int(boundary) - 22, int(boundary) + 23)}


def _net_segments():
    """Incoming fast toward the net; outgoing near-dead at low height at the net plane."""
    a = {"ok": True, "theta": np.array([5.0, 5.0, 1.0, 2.0, 30.0, 0.0, 0.0, 0.0, 0.0]),
         "f0": 90.0, "frames": np.arange(90.0, 100.0), "cost": 1.0}
    b = {"ok": True, "theta": np.array([8.0, NET_Y, 0.1, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0]),
         "f0": 101.0, "frames": np.arange(101.0, 111.0), "cost": 1.0}
    return [a, b]


def test_speed_collapse_at_net_is_verdict_net() -> None:
    segs = _net_segments()
    knots = classify_knots(segs, boxes={}, ball_obs=_moving_ball(100.0), fps=FPS,
                           surface="clay", camera=_FakeCamera())
    assert len(knots) == 1
    assert knots[0]["verdict"] == "net"
    assert knots[0]["speed_in"] >= P.NET_MIN_IN
    assert knots[0]["speed_out"] <= P.NET_MAX_OUT


def test_net_verdict_can_be_disabled() -> None:
    segs = _net_segments()
    saved = P.NET_VERDICT
    try:
        P.NET_VERDICT = False
        knots = classify_knots(segs, boxes={}, ball_obs=_moving_ball(100.0), fps=FPS,
                               surface="clay", camera=_FakeCamera())
    finally:
        P.NET_VERDICT = saved
    assert knots[0]["verdict"] != "net"


def test_fast_outgoing_is_not_net() -> None:
    # A struck ball leaves fast — not a tape kill, even if it crosses the net region.
    segs = _net_segments()
    segs[1]["theta"] = np.array([8.0, NET_Y, 0.5, 10.0, 25.0, 6.0, 0.0, 0.0, 0.0])
    knots = classify_knots(segs, boxes={}, ball_obs=_moving_ball(100.0), fps=FPS,
                           surface="clay", camera=_FakeCamera())
    assert knots[0]["verdict"] != "net"


def test_baseline_speed_collapse_is_not_net() -> None:
    # Same collapse but at a baseline (far from the net plane) — a drop/defensive shot, not
    # the tape. The net band must exclude it.
    segs = _net_segments()
    segs[0]["theta"] = np.array([5.0, 21.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    segs[1]["theta"] = np.array([5.0, 23.0, 0.3, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
    knots = classify_knots(segs, boxes={}, ball_obs=_moving_ball(100.0), fps=FPS,
                           surface="clay", camera=_FakeCamera())
    assert knots[0]["verdict"] != "net"


# --------------------------------------------------------------------------- #
# net is excluded from the hit chain and from export
# --------------------------------------------------------------------------- #
def test_net_survives_chain_selection_untouched() -> None:
    knots = [{"frame": 100.0, "verdict": "net", "side": "far", "near_player": None,
              "racket_feasible": True, "speed_in": 30.0, "speed_out": 2.0}]
    select_hit_chain(knots, audio_frames=[100.0], fps=FPS)
    assert knots[0]["verdict"] == "net"          # not reclassified to hit/unsupported
    assert knots[0]["score"] == 0.0              # never chain-eligible


class _CaptureWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def test_net_is_never_exported() -> None:
    # A net knot with otherwise-supporting evidence (audio + box) must still be skipped.
    knots = [{"frame": 100.0, "verdict": "net", "side": "far", "near_player": "far",
              "racket_feasible": True, "audio": True},
             {"frame": 140.0, "verdict": "hit", "side": "near", "near_player": "near",
              "racket_feasible": True, "audio": True}]
    writer = _CaptureWriter()
    _export_evidence(writer, "pt0000", knots)
    exported = [r["frame_A"] for r in writer.rows]
    assert "100.0" not in exported     # net excluded
    assert "140.0" in exported         # the real hit still exported
