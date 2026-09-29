"""Unit tests for the boundary contacts: the terminal (rally-ending) stroke and the
toss-started serve, both ONE-SIDED BY CONSTRUCTION.

Discovery defines a contact as the meeting of two fitted flights, and select_hit_chain keeps
only a legally-alternating interior chain. The two boundary contacts break that assumption:

  * the TERMINAL (last) stroke has no subsequent hit — its outgoing flight is truncated by the
    things that END points (a director cut mid-flight, the net, or a final bounce sequence
    settling into the dead-ball signature). The 2026-07-21 terminal-miss strata found ~52
    genuine last-contact losses are a player-supported break that DOES exist in-window but was
    gate-rejected (its monocular fit ghosts in altitude because the outgoing arc is truncated).
    append_terminal_contact recovers it BACKWARD from the point end, gated on a racket-feasible
    impulse + a LEGAL outgoing termination + an independent audio/box witness + a legal
    round-trip gap — NOT on height_ok (which ghosts at the boundary by construction).

  * the TOSS-started SERVE opens a window during the toss, so segments[0] is the slow rising
    toss and the serve strike is the first break (gate-rejected because the ball is above the
    server's body box). prepend_serve_contact promotes it FORWARD from the window start.

Every accept criterion is physical / observation-derived (round-trip time, court geometry,
frame bounds, the calibrated witnesses) — none tuned on the 62 dev labels.
"""
import numpy as np

import physics_knot_solver as P
from physics_knot_solver import (
    _legal_gap,
    append_terminal_contact,
    prepend_serve_contact,
    terminal_reason,
)

FPS = 50.0


def _seg(f0, f1, theta):
    return {"ok": True, "theta": np.asarray(theta, float),
            "f0": float(f0), "frames": np.arange(float(f0), float(f1)), "cost": 1.0}


def _moving_ball(lo, hi):
    return {f: (100.0 + f, 120.0 + f) for f in range(int(lo), int(hi))}


# --------------------------------------------------------------------------- #
# _legal_gap — the round-trip grammar
# --------------------------------------------------------------------------- #
def test_legal_gap_alternating_rhythm() -> None:
    assert _legal_gap(1.0, "near", "far")          # opposite side, in-rhythm
    assert not _legal_gap(1.0, "near", "near")     # same side, in-rhythm -> illegal
    assert not _legal_gap(0.2, "near", "far")      # too fast for a round trip
    assert _legal_gap(4.5, "near", "near")         # long pause resets rhythm (fault/2nd serve)


# --------------------------------------------------------------------------- #
# terminal_reason — legal terminations vs a healthy (non-terminal) flight
# --------------------------------------------------------------------------- #
def test_reason_camera_cut_when_no_outgoing_segment() -> None:
    knot = {"frame": 200.0}
    assert terminal_reason(knot, [], {180: (0, 0), 205: (0, 0)}, FPS, "clay") == "camera_cut"


def test_reason_camera_cut_when_truncated_airborne() -> None:
    # outgoing ends right at the last observation, still moving fast -> a director cut.
    out = _seg(200, 220, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    ball = _moving_ball(100, 221)  # last_obs 220 == outgoing end
    knot = {"frame": 199.0}
    assert terminal_reason(knot, [out], ball, FPS, "clay") == "camera_cut"


def test_reason_dead_ball_bounce_when_later_verdict_flags_death() -> None:
    out = _seg(200, 240, [5.0, 5.0, 1.0, 2.0, 10.0, 0, 0, 0, 0])
    ball = _moving_ball(100, 400)
    knot = {"frame": 199.0}
    assert terminal_reason(knot, [out], ball, FPS, "clay",
                           later_verdicts=("dead_ball",)) == "dead_ball_bounce"


def test_reason_none_for_healthy_midcourt_flight() -> None:
    # A fast flight ending well inside the window, not near the net, is NOT a terminal: it
    # means another contact was missed downstream.
    out = _seg(200, 230, [5.0, 3.0, 2.0, 3.0, 30.0, 4.0, 0, 0, 0])
    ball = _moving_ball(100, 400)  # last_obs 399, far from outgoing end 229
    knot = {"frame": 199.0}
    assert terminal_reason(knot, [out], ball, FPS, "clay") is None


# --------------------------------------------------------------------------- #
# append_terminal_contact
# --------------------------------------------------------------------------- #
def _base_hit(frame, side="near"):
    return {"frame": float(frame), "verdict": "hit", "side": side, "near_player": side,
            "height_ok": True, "racket_feasible": True, "audio": True}


def _terminal_knot(frame, **kw):
    k = {"frame": float(frame), "verdict": "unsupported_break", "side": "far",
         "near_player": None, "height_ok": False, "racket_feasible": True, "audio": True}
    k.update(kw)
    return k


def test_terminal_promoted_and_carries_reason() -> None:
    # last hit at 100 (near); a feasible far break at 150 (1.0 s later) whose outgoing is cut.
    out = _seg(155, 175, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    ball = _moving_ball(80, 176)
    knots = [_base_hit(100, "near"), _terminal_knot(150, side="far", near_player="far")]
    got = append_terminal_contact(knots, [out], ball, FPS, "clay")
    assert got is not None and got["frame"] == 150.0
    assert knots[1]["verdict"] == "hit"
    assert knots[1]["terminal"] is True
    assert knots[1]["terminal_reason"] == "camera_cut"


def test_terminal_ignores_height_ok_failure() -> None:
    # The terminal's fit ghosts in altitude (height_ok False) — it must still be recoverable.
    out = _seg(155, 175, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    ball = _moving_ball(80, 176)
    knots = [_base_hit(100, "near"), _terminal_knot(150, side="far", near_player="far",
                                                    height_ok=False)]
    assert append_terminal_contact(knots, [out], ball, FPS, "clay") is not None


def test_terminal_rejected_without_legal_termination() -> None:
    # outgoing is a healthy mid-court flight (reason None) -> not the last stroke.
    out = _seg(155, 185, [5.0, 3.0, 2.0, 3.0, 30.0, 4.0, 0, 0, 0])
    ball = _moving_ball(80, 400)
    knots = [_base_hit(100, "near"), _terminal_knot(150, side="far", near_player="far")]
    assert append_terminal_contact(knots, [out], ball, FPS, "clay") is None
    assert knots[1]["verdict"] == "unsupported_break"


def test_terminal_rejected_on_illegal_gap() -> None:
    out = _seg(112, 132, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    ball = _moving_ball(80, 133)
    # break only 0.2 s after the last hit -> too fast for a round trip.
    knots = [_base_hit(100, "near"), _terminal_knot(110, side="far", near_player="far")]
    assert append_terminal_contact(knots, [out], ball, FPS, "clay") is None


def test_terminal_witness_required_by_default_but_not_physics_only() -> None:
    # A dead-ball-bounce termination with NO audio and NO box: rejected by default, accepted
    # when the witness requirement is relaxed. Sited at the far baseline (y~20, clear of the
    # net band) and decaying below play speed, so the reason is dead_ball_bounce, not net.
    out = _seg(155, 200, [5.0, 20.0, 0.5, 1.0, 3.0, 0, 0, 0, 0])   # slow, far from the net
    ball = _moving_ball(80, 400)

    def mk():
        return [
            _base_hit(100, "near"),
            _terminal_knot(150, side="far", near_player=None, audio=False),
        ]

    knots = mk()
    assert append_terminal_contact(knots, [out], ball, FPS, "clay") is None   # default
    saved = P.TERMINAL_REQUIRE_WITNESS
    try:
        P.TERMINAL_REQUIRE_WITNESS = False
        knots = mk()
        got = append_terminal_contact(knots, [out], ball, FPS, "clay")
        assert got is not None and got["terminal_reason"] == "dead_ball_bounce"
    finally:
        P.TERMINAL_REQUIRE_WITNESS = saved


def test_terminal_picks_last_stroke_not_settling_break() -> None:
    # Two candidates after the last hit; the LATER one (nearer the point end) is the terminal.
    out1 = _seg(135, 155, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    out2 = _seg(175, 195, [5.0, 5.0, 3.0, 5.0, 25.0, 3.0, 0, 0, 0])
    ball = _moving_ball(80, 196)
    knots = [_base_hit(100, "near"),
             _terminal_knot(130, side="far", near_player="far"),
             _terminal_knot(170, side="far", near_player="far")]
    got = append_terminal_contact(knots, [out1, out2], ball, FPS, "clay")
    assert got is not None and got["frame"] == 170.0


def test_terminal_no_op_without_any_emitted_hit() -> None:
    knots = [_terminal_knot(150)]
    assert append_terminal_contact(knots, [], {}, FPS, "clay") is None


def test_terminal_skips_dead_ball_and_net_verdicts() -> None:
    ball = _moving_ball(80, 400)
    knots = [_base_hit(100, "near"),
             {"frame": 150.0, "verdict": "dead_ball", "side": "far", "near_player": "far",
              "height_ok": True, "racket_feasible": True, "audio": True}]
    assert append_terminal_contact(knots, [], ball, FPS, "clay") is None


# --------------------------------------------------------------------------- #
# prepend_serve_contact
# --------------------------------------------------------------------------- #
def _toss_seg():
    # slow, rising (vz>0), near-vertical (small vy): a toss.
    return _seg(0, 20, [5.0, 2.0, 2.0, 0.5, 1.0, 8.0, 0, 0, 0])


def _served_seg():
    return _seg(24, 60, [5.0, 2.0, 3.0, 1.0, 30.0, -5.0, 0, 0, 0])


def test_serve_toss_strike_promoted() -> None:
    segs = [_toss_seg(), _served_seg()]
    strike = {"frame": 22.0, "verdict": "unsupported_break", "side": "near",
              "near_player": None, "height_ok": True, "racket_feasible": True}
    got = prepend_serve_contact([strike], segs, FPS)
    assert got is not None and strike["verdict"] == "hit" and strike["serve"] is True


def test_serve_no_op_when_first_segment_is_fast() -> None:
    # segments[0] already launches at serve speed -> legacy insertion path owns it.
    segs = [_served_seg(), _seg(64, 90, [5, 15, 1, 1, 20, 2, 0, 0, 0])]
    strike = {"frame": 62.0, "verdict": "unsupported_break", "near_player": None,
              "height_ok": True, "racket_feasible": True}
    assert prepend_serve_contact([strike], segs, FPS) is None


def test_serve_no_op_when_outgoing_not_serve_speed() -> None:
    # a slow rising first flight followed by another slow flight is not a serve.
    segs = [_toss_seg(), _seg(24, 60, [5, 2, 3, 1, 8, -1, 0, 0, 0])]
    strike = {"frame": 22.0, "verdict": "unsupported_break", "near_player": None,
              "height_ok": True, "racket_feasible": True}
    assert prepend_serve_contact([strike], segs, FPS) is None
