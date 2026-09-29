"""Tests for track healing.

The load-bearing requirement is that healing removes tracker jumps WITHOUT removing real
events. A healer that quietly smooths a racket contact into a smooth curve would destroy
the signal the whole pipeline exists to find, so the kink tests matter more than the
outlier tests.
"""

from __future__ import annotations

import numpy as np
from track_heal import heal_track


def _smooth_arc(n=30, start=(200.0, 400.0), velocity=(6.0, -9.0), curvature=(0.0, 0.35)):
    frames = np.arange(n, dtype=float)
    t = frames
    points = np.stack([
        start[0] + velocity[0] * t + curvature[0] * t**2,
        start[1] + velocity[1] * t + curvature[1] * t**2,
    ], axis=1)
    return frames, points


def test_clean_arc_is_left_alone():
    frames, points = _smooth_arc()
    result = heal_track(frames, points)
    assert result.n_removed == 0
    assert np.allclose(result.points, points)


def test_single_frame_jump_is_removed():
    frames, points = _smooth_arc()
    corrupted = points.copy()
    corrupted[14] += np.array([160.0, -120.0])  # tracker latches onto a distractor
    result = heal_track(frames, corrupted)

    assert result.n_removed >= 1
    assert not result.kept[14]
    # The healed track must recover the original geometry where it kept samples.
    assert np.max(np.linalg.norm(result.points[~result.interpolated] - points[result.kept], axis=1)) < 1e-6


def test_two_frame_jump_is_removed():
    frames, points = _smooth_arc()
    corrupted = points.copy()
    corrupted[12] += np.array([150.0, 90.0])
    corrupted[13] += np.array([155.0, 95.0])
    result = heal_track(frames, corrupted)
    assert not result.kept[12] and not result.kept[13]


def test_a_real_kink_is_preserved():
    """A bounce: vertical motion reverses and STAYS reversed. Nothing may be dropped."""
    frames = np.arange(40, dtype=float)
    points = np.zeros((40, 2))
    for i, f in enumerate(frames):
        if f < 20:
            points[i] = [200 + 6 * f, 300 + 8 * f]
        else:
            points[i] = [200 + 6 * f, 300 + 8 * 20 - 9 * (f - 20)]
    result = heal_track(frames, points)

    assert result.n_removed == 0, "healing removed samples around a genuine bounce"
    assert result.kept[19] and result.kept[20] and result.kept[21]


def test_a_contact_reversal_is_preserved():
    """A racket hit: direction inverts. The kink frames must survive."""
    frames = np.arange(36, dtype=float)
    points = np.zeros((36, 2))
    for i, f in enumerate(frames):
        if f < 18:
            points[i] = [500 - 11 * f, 260 + 3 * f]
        else:
            points[i] = [500 - 11 * 18 + 12 * (f - 18), 260 + 3 * 18 + 4 * (f - 18)]
    result = heal_track(frames, points)

    assert result.n_removed == 0
    assert result.kept[17] and result.kept[18]


def test_short_gaps_are_filled_and_flagged():
    frames, points = _smooth_arc(n=30)
    keep = np.ones(30, bool)
    keep[[10, 11]] = False  # the tracker briefly loses the ball
    result = heal_track(frames[keep], points[keep])

    assert result.n_filled == 2
    assert result.interpolated.sum() == 2
    filled = result.points[result.interpolated]
    truth = points[[10, 11]]
    # Interpolation across two frames of a smooth arc should be close, not exact.
    assert np.max(np.linalg.norm(filled - truth, axis=1)) < 2.0


def test_long_gaps_are_not_invented():
    frames, points = _smooth_arc(n=40)
    keep = np.ones(40, bool)
    keep[15:25] = False
    result = heal_track(frames[keep], points[keep], max_gap=4)
    assert result.n_filled == 0


def test_jump_next_to_a_kink_is_still_caught():
    """The hard case: healing must not be fooled into keeping a jump because a kink is near."""
    frames = np.arange(40, dtype=float)
    points = np.zeros((40, 2))
    for i, f in enumerate(frames):
        if f < 20:
            points[i] = [200 + 6 * f, 300 + 8 * f]
        else:
            points[i] = [200 + 6 * f, 460 - 9 * (f - 20)]
    corrupted = points.copy()
    corrupted[26] += np.array([180.0, 140.0])
    result = heal_track(frames, corrupted)

    assert not result.kept[26]
    assert result.kept[19] and result.kept[20] and result.kept[21]


def test_short_input_is_returned_untouched():
    frames = np.arange(4, dtype=float)
    points = np.zeros((4, 2))
    result = heal_track(frames, points)
    assert result.n_removed == 0 and result.n_filled == 0
