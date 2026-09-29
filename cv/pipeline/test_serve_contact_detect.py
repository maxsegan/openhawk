"""Unit tests for serve_contact_detect.py -- the toss-impulse strike signal (gap-aware),
the toss-arc divergence, and audio snapping. No data files required (synthetic tracks)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_contact_detect as scd  # noqa: E402


def _serve_track(hint=60):
    """Synthetic far-serve track: toss rises to an apex, HANGS slow, then a STRIKE launches a
    fast descending flight. Returns (frames, xs, ys) arrays."""
    fr, x, y = [], [], []
    # rise to apex (image-y falls to a min), x roughly constant
    for i, f in enumerate(range(hint - 20, hint - 8)):
        fr.append(f)
        x.append(400.0 + 0.2 * i)
        y.append(120.0 - 4.0 * i)
    # hang near the apex (slow), image-y ~ minimum
    for f in range(hint - 8, hint):
        fr.append(f)
        x.append(402.0)
        y.append(72.0 + 0.4 * (f - (hint - 8)))
    # STRIKE at ~hint: fast descending served flight (big speed jump)
    for i, f in enumerate(range(hint + 1, hint + 20)):
        fr.append(f)
        x.append(404.0 + 7.0 * i)
        y.append(76.0 + 9.0 * i)
    return np.array(fr, float), np.array(x, float), np.array(y, float)


class TestTossImpulse(unittest.TestCase):
    def test_strike_after_hang(self):
        fr, x, y = _serve_track(60)
        r = scd.toss_impulse(fr, x, y, 60)
        self.assertIsNotNone(r)
        # apex is the highest toss point (image-y min), strike is the launch, apex < strike
        self.assertLess(r["apex_frame"], r["strike_frame"])
        self.assertTrue(56 <= r["strike_frame"] <= 63, r["strike_frame"])
        self.assertGreater(r["speed_jump"], 3.0)  # a real velocity discontinuity

    def test_strike_in_gap_uses_midpoint(self):
        # remove the samples right at the strike so the ball is untracked across the impulse
        fr, x, y = _serve_track(60)
        keep = (fr <= 58) | (fr >= 63)
        r = scd.toss_impulse(fr[keep], x[keep], y[keep], 60)
        self.assertIsNotNone(r)
        self.assertTrue(58 <= r["strike_frame"] <= 63)  # midpoint of the observation gap

    def test_ignores_earlier_impulse_in_wide_toss_window(self):
        old_f, old_x, old_y = _serve_track(20)
        new_f, new_x, new_y = _serve_track(60)
        frames = np.concatenate([old_f, new_f])
        xs = np.concatenate([old_x, new_x])
        ys = np.concatenate([old_y, new_y])
        order = np.argsort(frames)
        result = scd.toss_impulse(frames[order], xs[order], ys[order], 60)
        self.assertIsNotNone(result)
        self.assertTrue(56 <= result["strike_frame"] <= 63, result["strike_frame"])


class TestTossDivergence(unittest.TestCase):
    def test_leaves_free_fall_at_strike(self):
        fr, x, y = _serve_track(60)
        r = scd.toss_impulse(fr, x, y, 60)
        d = scd.toss_arc_divergence(fr, x, y, 60, r["apex_frame"])
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d["frame"], r["apex_frame"])


class TestAudioRefine(unittest.TestCase):
    def test_snaps_to_local_onset(self):
        scores = np.zeros(400)
        scores[2 * 61] = 88.0  # strong onset at frame 61
        r = scd.audio_refine(
            scores,
            60.0,
            fps=scd.REFERENCE_FPS,
            radius_seconds=5 / scd.REFERENCE_FPS,
        )
        self.assertIsNotNone(r)
        self.assertEqual(r["frame"], 61.0)
        self.assertGreater(r["score"], 50)


if __name__ == "__main__":
    unittest.main()
