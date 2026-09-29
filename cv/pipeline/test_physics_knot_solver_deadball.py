"""Unit tests for the dead-ball / play-state gate and the evidence-based export gate.

FIX 1 (dead-ball gate): between points the ball tracker locks onto a parked or rolling
ball; single-frame flicker fakes velocity past the energy gate and turn past the reversal
test, so neither pixel speed nor turn can flag it. The one signal that separates a parked
ball from any real flight is the WIDE-WINDOW image span (a robust 5-95 percentile range,
flicker-proof): a held ball sits in a small box while even the slowest real far-court arc
sweeps far wider. A break flagged dead_ball is excluded from the hit chain AND from export.

FIX 2 (export gate): at contact the ball is at the racket, beyond the player's body box, so
far-court serves and wide reaches legitimately leave near_player=None even when the boxes
are present. The player-box association therefore must not gate EXPORT (only the
conservative HIT verdict); an audio- or feasibility-supported, correctly-timed break is
exported so refinement can judge it.
"""
import numpy as np

from physics_knot_solver import (
    DEAD_BALL_SPAN_PX,
    _export_evidence,
    classify_knots,
    select_hit_chain,
    wide_window_span,
)


class _CaptureWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)

FPS = 50.0


# --------------------------------------------------------------------------- #
# wide_window_span — the discriminator at the heart of FIX 1
# --------------------------------------------------------------------------- #
def test_span_none_when_too_few_observations() -> None:
    obs = {100: (10.0, 10.0), 101: (11.0, 11.0)}  # < DEAD_BALL_MIN_OBS
    assert wide_window_span(obs, 100.5) is None


def test_parked_ball_has_small_span() -> None:
    # A ball sitting in one spot with a few px of jitter over the whole window.
    rng = np.random.default_rng(1)
    obs = {f: (200.0 + rng.uniform(-5, 5), 150.0 + rng.uniform(-5, 5))
           for f in range(80, 121)}
    span = wide_window_span(obs, 100.0)
    assert span is not None and span < DEAD_BALL_SPAN_PX


def test_moving_arc_has_large_span() -> None:
    # A ball sweeping across the frame — even a modest few px/frame far-court arc.
    obs = {f: (100.0 + (f - 80) * 4.0, 120.0 + (f - 80) * 2.0) for f in range(80, 121)}
    span = wide_window_span(obs, 100.0)
    assert span is not None and span > DEAD_BALL_SPAN_PX


def test_span_robust_to_single_frame_flicker() -> None:
    # A parked ball whose tracker teleports for ONE frame must still read as parked:
    # the robust percentile range rejects the lone outlier (raw max-min would not).
    obs = {f: (200.0, 150.0) for f in range(80, 121)}
    obs[100] = (900.0, 800.0)  # single-frame flicker far away
    span = wide_window_span(obs, 100.0)
    assert span is not None and span < DEAD_BALL_SPAN_PX


# --------------------------------------------------------------------------- #
# classify_knots — the gate applied to a discovered break
# --------------------------------------------------------------------------- #
class _FakeCamera:
    """Minimal camera: a fixed pinhole P and identity homography, enough for the side
    heuristic and any ground back-projection classify_knots may touch."""

    def p_at(self, frame):
        return np.array([[1000.0, 0.0, 480.0, 0.0],
                         [0.0, 1000.0, 270.0, -500.0],
                         [0.0, 0.0, 1.0, 0.0]])

    def h_at(self, frame):
        return np.eye(3, dtype=np.float32)


def _play_speed_segments():
    """Two adjacent fitted flights that meet at a break, both at play speed (so the break
    would classify as hit/unsupported, NOT low_energy) — the dead-ball gate must override."""
    # theta = [x, y, z, vx, vy, vz, wx, wy, wz]; ~20 m/s launches.
    a = {"ok": True, "theta": np.array([5.0, 5.0, 1.0, 15.0, 10.0, 3.0, 0.0, 0.0, 0.0]),
         "f0": 90.0, "frames": np.arange(90.0, 100.0), "cost": 1.0}
    b = {"ok": True, "theta": np.array([7.0, 8.0, 1.0, -12.0, -14.0, 3.0, 0.0, 0.0, 0.0]),
         "f0": 101.0, "frames": np.arange(101.0, 111.0), "cost": 1.0}
    return [a, b]


def test_classify_flags_dead_ball_when_ball_parked() -> None:
    segs = _play_speed_segments()
    parked = {f: (300.0, 200.0) for f in range(78, 123)}  # wide-window parked cluster
    knots = classify_knots(segs, boxes={}, ball_obs=parked, fps=FPS, surface="clay",
                           camera=_FakeCamera())
    assert len(knots) == 1
    assert knots[0]["verdict"] == "dead_ball"
    assert knots[0]["wide_span"] is not None and knots[0]["wide_span"] < DEAD_BALL_SPAN_PX


def test_classify_does_not_flag_dead_ball_when_ball_moves() -> None:
    segs = _play_speed_segments()
    moving = {f: (100.0 + (f - 78) * 6.0, 120.0 + (f - 78) * 3.0) for f in range(78, 123)}
    knots = classify_knots(segs, boxes={}, ball_obs=moving, fps=FPS, surface="clay",
                           camera=_FakeCamera())
    assert len(knots) == 1
    assert knots[0]["verdict"] != "dead_ball"


def test_dead_ball_gate_can_be_disabled() -> None:
    segs = _play_speed_segments()
    parked = {f: (300.0, 200.0) for f in range(78, 123)}
    knots = classify_knots(segs, boxes={}, ball_obs=parked, fps=FPS, surface="clay",
                           camera=_FakeCamera(), dead_ball_gate=False)
    assert knots[0]["verdict"] != "dead_ball"   # gate off: falls back to physics verdict


# --------------------------------------------------------------------------- #
# select_hit_chain — dead_ball must survive chain selection untouched
# --------------------------------------------------------------------------- #
def test_select_hit_chain_preserves_dead_ball() -> None:
    # A dead_ball knot sitting between two real, audio+box+feasible hits must never be
    # relabelled hit or unsupported_break, and must not join the chain.
    knots = [
        {"frame": 100.0, "side": "near", "near_player": "near",
         "racket_feasible": True, "verdict": "unsupported_break"},
        {"frame": 140.0, "side": "far", "near_player": None,
         "racket_feasible": True, "verdict": "dead_ball"},
        {"frame": 180.0, "side": "far", "near_player": "far",
         "racket_feasible": True, "verdict": "unsupported_break"},
    ]
    select_hit_chain(knots, audio_frames=[100.0, 180.0], fps=FPS)
    assert knots[1]["verdict"] == "dead_ball"           # preserved
    assert knots[0]["verdict"] == "hit"                 # real contacts still chained
    assert knots[2]["verdict"] == "hit"


def test_export_evidence_gate() -> None:
    # FIX 2 export: audio- or box-supported breaks are exported (including near_player=None
    # far-court breaks that have audio); dead_ball / low_energy are excluded; a break with
    # ONLY racket feasibility (no audio, no box) is NOT exported (feasibility floods).
    knots = [
        {"frame": 58.0, "side": "far", "near_player": None, "racket_feasible": False,
         "audio": True, "verdict": "unsupported_break"},          # audio-only far -> export
        {"frame": 100.0, "side": "near", "near_player": "near", "racket_feasible": True,
         "audio": False, "verdict": "hit"},                        # box/hit -> export
        {"frame": 150.0, "side": "far", "near_player": None, "racket_feasible": True,
         "audio": False, "verdict": "unsupported_break"},          # feasibility-only -> skip
        {"frame": 320.0, "side": "far", "near_player": None, "racket_feasible": True,
         "audio": True, "verdict": "dead_ball"},                   # dead-ball -> skip
        {"frame": 400.0, "side": "near", "near_player": None, "racket_feasible": True,
         "audio": True, "verdict": "low_energy_break"},            # pre-serve bounce -> skip
    ]
    w = _CaptureWriter()
    _export_evidence(w, "pt0001", knots)
    exported = sorted(float(r["frame_A"]) for r in w.rows)
    assert exported == [58.0, 100.0]
    assert w.rows[0]["side_phys"] == "far" and w.rows[0]["phase_A"] == "serve"
    assert w.rows[1]["side_phys"] == "near" and w.rows[1]["phase_A"] == "rally"


def test_select_hit_chain_still_classifies_non_dead() -> None:
    knots = [
        {"frame": 100.0, "side": "near", "near_player": "near",
         "racket_feasible": True, "verdict": "unsupported_break"},
        {"frame": 150.0, "side": "far", "near_player": "far",
         "racket_feasible": True, "verdict": "unsupported_break"},
    ]
    select_hit_chain(knots, audio_frames=[100.0, 150.0], fps=FPS)
    assert {k["verdict"] for k in knots} == {"hit"}
