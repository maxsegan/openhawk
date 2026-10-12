"""Unit tests for the audio-timing snap and sound-travel delay in physics_knot_solver.

snap_knots_to_audio implements PIPELINE doctrine "audio refines timing only": a
physics-discovered break within SNAP_TOL_F of an impact-audio onset is moved onto the
onset (optionally corrected for position-dependent sound travel), but a break with no
nearby onset is left where physics put it, and two breaks never collapse onto one onset.
"""
import numpy as np

from physics_knot_solver import (
    FAR_BASELINE_XYZ,
    NEAR_BASELINE_XYZ,
    SNAP_TOL_F,
    TURN_ANGLE_MIN,
    _turn_profile,
    snap_knots_to_audio,
    sound_delay_frames,
)

FPS = 50.0


def test_snap_moves_break_within_tolerance() -> None:
    knots = [{"frame": 100.5, "side": "near", "court_xy": (5.5, 0.0)}]
    snap_knots_to_audio(knots, [104.0], FPS, delay_correct=False)
    assert knots[0]["frame"] == 104.0
    assert knots[0].get("audio_snapped")


def test_snap_leaves_distant_break_alone() -> None:
    knots = [{"frame": 100.5, "side": "near", "court_xy": (5.5, 0.0)}]
    snap_knots_to_audio(knots, [100.5 + SNAP_TOL_F + 2.0], FPS, delay_correct=False)
    assert knots[0]["frame"] == 100.5
    assert not knots[0].get("audio_snapped")


def test_snap_does_not_collapse_two_breaks_onto_one_onset() -> None:
    knots = [{"frame": 200.0, "side": "near", "court_xy": (5.5, 0.0)},
             {"frame": 206.0, "side": "near", "court_xy": (5.5, 0.0)}]
    snap_knots_to_audio(knots, [203.0], FPS, delay_correct=False)
    frames = sorted(k["frame"] for k in knots)
    assert frames[0] != frames[1]
    assert 203.0 in frames


def test_snap_no_audio_is_noop() -> None:
    knots = [{"frame": 50.0, "side": "near"}]
    snap_knots_to_audio(knots, [], FPS, delay_correct=False)
    assert knots[0]["frame"] == 50.0


def test_far_hit_has_larger_sound_delay_than_near() -> None:
    near = sound_delay_frames(NEAR_BASELINE_XYZ, FPS)
    far = sound_delay_frames(FAR_BASELINE_XYZ, FPS)
    assert far > near
    assert 0.5 < near < 2.5      # near baseline ~1.6 frames
    assert 3.0 < far < 6.0       # far baseline ~5 frames


def test_delay_correction_lands_break_before_onset_by_its_delay() -> None:
    # A far-court break snapped WITH correction lands at onset - far_delay (earlier than the
    # raw onset by the larger far delay); a near break moves less.
    onset = 300.0
    far_knot = [{"frame": 297.0, "side": "far", "court_xy": (5.5, 23.77)}]
    snap_knots_to_audio(far_knot, [onset], FPS, delay_correct=True)
    far_delay = sound_delay_frames(FAR_BASELINE_XYZ, FPS)
    assert abs(far_knot[0]["frame"] - (onset - far_delay)) < 1e-6

    near_knot = [{"frame": 299.0, "side": "near", "court_xy": (5.5, 0.0)}]
    snap_knots_to_audio(near_knot, [onset], FPS, delay_correct=True)
    near_delay = sound_delay_frames(NEAR_BASELINE_XYZ, FPS)
    assert abs(near_knot[0]["frame"] - (onset - near_delay)) < 1e-6
    assert (onset - near_knot[0]["frame"]) < (onset - far_knot[0]["frame"])


def test_offcourt_fitted_position_falls_back_to_side_baseline() -> None:
    # A ghosted (off-court) fitted position must not drive the delay; side baseline is used.
    onset = 500.0
    knot = [{"frame": 498.0, "side": "far", "court_xy": (5.5, 45.0)}]  # y=45 off court
    snap_knots_to_audio(knot, [onset], FPS, delay_correct=True)
    far_delay = sound_delay_frames(FAR_BASELINE_XYZ, FPS)
    assert abs(knot[0]["frame"] - (onset - far_delay)) < 1e-6


def test_turn_profile_flat_on_straight_path() -> None:
    # A constant-velocity image path has no turn anywhere.
    uv = np.array([[float(i) * 10.0, float(i) * 5.0] for i in range(20)])
    ang = _turn_profile(uv)
    assert float(ang.max()) < 1e-6


def test_turn_profile_peaks_at_a_sharp_kink() -> None:
    # Path goes right, then sharply up-left at index 10 (a contact-like reversal).
    left = [[float(i) * 10.0, 0.0] for i in range(11)]
    right = [[100.0 - float(k) * 10.0, float(k) * 10.0] for k in range(1, 10)]
    ang = _turn_profile(np.array(left + right))
    j = int(ang.argmax())
    assert 8 <= j <= 12                      # peak localizes at the kink
    assert float(ang[j]) >= TURN_ANGLE_MIN   # and clears the proposal threshold


def test_turn_profile_ignores_near_stationary_noise() -> None:
    # Tiny jitter (sub-3px steps) must not register as a turn.
    rng = np.random.default_rng(0)
    uv = np.cumsum(rng.uniform(-1.0, 1.0, size=(30, 2)), axis=0)
    ang = _turn_profile(uv)
    assert float(ang.max()) < TURN_ANGLE_MIN
