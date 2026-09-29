"""Tests for the contact-frame refiner's rules, contract, and abstentions."""

from __future__ import annotations

import math

import pytest

from cv.pipeline.contact_frame_refiner import (
    DEFAULT_WEIGHTS,
    SIGMA_CAP_PX,
    SIGMA_FLOOR_PX,
    SIGNAL_NAMES,
    WINDOW,
    all_signals,
    audio_onset_image_frame,
    blur_streak,
    direction_change,
    disagreement_px,
    kinematic_reversal,
    normalise_track,
    racket_proximity,
    refine,
    sigma_from_disagreement,
    speed_minimum,
)


def corner_track(
    contact_frame: int,
    *,
    incoming=(40.0, -12.0),
    outgoing=(-18.0, -30.0),
    apex=(900.0, 500.0),
    span: int = 10,
    scale: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """A synthetic reversal: straight in, corner at ``contact_frame``, straight out."""

    track = {}
    for frame in range(contact_frame - span, contact_frame + span + 1):
        step = (frame - contact_frame) * scale
        velocity = incoming if frame <= contact_frame else outgoing
        track[frame] = (apex[0] + velocity[0] * step, apex[1] + velocity[1] * step)
    return track


def test_normalise_track_accepts_tuples_mappings_and_native_columns():
    track = normalise_track(
        {
            10: (1.0, 2.0),
            "11": {"x": 3.0, "y": 4.0},
            12: {"x_native": 5.0, "y_native": 6.0},
            13: {"image_x": 7.0, "image_y": 8.0},
        }
    )
    assert track == {10: (1.0, 2.0), 11: (3.0, 4.0), 12: (5.0, 6.0), 13: (7.0, 8.0)}


def test_normalise_track_drops_missing_and_non_finite_rows():
    track = normalise_track({1: None, 2: (float("nan"), 3.0), 3: (4.0, 5.0)})
    assert track == {3: (4.0, 5.0)}


@pytest.mark.parametrize("true_offset", [-2, -1, 0, 1, 2])
def test_kinematic_reversal_finds_the_corner_anywhere_in_the_window(true_offset):
    emitted = 100
    track = corner_track(emitted + true_offset)
    assert kinematic_reversal(track, emitted).offset == true_offset


def test_kinematic_reversal_by_time_agrees_with_by_pixel_on_a_clean_corner():
    track = corner_track(99)
    assert kinematic_reversal(track, 100, by_pixel=False).offset == -1
    assert kinematic_reversal(track, 100).offset == -1


def test_kinematic_reversal_needs_rows_on_both_sides():
    track = corner_track(100)
    one_sided = {frame: point for frame, point in track.items() if frame <= 100}
    signal = kinematic_reversal(one_sided, 100)
    assert signal.offset is None
    assert signal.evidence["reason"] == "not_enough_rows"


def test_kinematic_reversal_reports_parallel_lines_rather_than_guessing():
    track = {frame: (10.0 * frame, 0.0) for frame in range(90, 111)}
    signal = kinematic_reversal(track, 100)
    assert signal.offset is None
    assert signal.evidence["reason"] == "parallel_lines"


def test_kinematic_reversal_excludes_the_window_from_its_own_fit():
    """A corrupt pixel inside the window must not bend either line."""

    emitted = 100
    track = corner_track(emitted - 1)
    track[emitted] = (5.0, 5.0)
    assert kinematic_reversal(track, emitted).offset == -1


@pytest.mark.parametrize("true_offset", [-1, 0, 1])
def test_speed_minimum_lands_on_the_reversal(true_offset):
    track = corner_track(100 + true_offset)
    assert speed_minimum(track, 100).offset == true_offset


def test_speed_minimum_centered_and_step_variants_disagree_on_a_symmetric_corner():
    """The centered speed collapses at the corner; the mean of steps does not."""

    track = corner_track(101)
    assert speed_minimum(track, 100, centered=True).offset == 1
    assert speed_minimum(track, 100, centered=False).offset is not None


def test_speed_minimum_without_neighbours_answers_nothing():
    signal = speed_minimum({100: (1.0, 1.0)}, 100)
    assert signal.offset is None
    assert signal.evidence["reason"] == "no_local_speed"


def test_direction_change_picks_the_sharpest_turn():
    track = corner_track(99)
    signal = direction_change(track, 100)
    assert signal.offset == -1
    assert signal.evidence["max_turn_deg"] > 45.0


def test_direction_change_is_silent_on_a_straight_line():
    track = {frame: (10.0 * frame, 5.0 * frame) for frame in range(90, 111)}
    assert direction_change(track, 100).offset == 0
    assert direction_change(track, 100).evidence["max_turn_deg"] == pytest.approx(0.0, abs=1e-6)


def test_racket_proximity_picks_the_frame_nearest_a_wrist():
    track = {frame: (100.0 + 30.0 * abs(frame - 101), 200.0) for frame in range(98, 104)}
    pose = {frame: [{"joints": {"right_wrist": [100.0, 200.0, 0.9]}}] for frame in range(98, 104)}
    assert racket_proximity(track, 100, pose).offset == 1


def test_racket_proximity_ignores_low_confidence_keypoints():
    track = {frame: (100.0, 200.0) for frame in range(98, 104)}
    pose = {frame: [{"joints": {"right_wrist": [100.0, 200.0, 0.05]}}] for frame in range(98, 104)}
    signal = racket_proximity(track, 100, pose)
    assert signal.offset is None
    assert signal.evidence["reason"] == "no_wrist_witness"


def test_racket_proximity_ignores_a_wrist_on_the_other_side_of_the_court():
    track = {frame: (100.0, 200.0) for frame in range(98, 104)}
    pose = {frame: [{"joints": {"right_wrist": [1500.0, 900.0, 0.9]}}] for frame in range(98, 104)}
    assert racket_proximity(track, 100, pose).offset is None


def test_racket_proximity_without_pose_is_silent():
    assert racket_proximity({100: (1.0, 1.0)}, 100, None).evidence["reason"] == "no_pose"


def test_blur_streak_picks_the_shortest_streak():
    streaks = {98: 40.0, 99: 30.0, 100: 22.0, 101: 9.0, 102: 35.0}
    assert blur_streak(streaks, 100).offset == 1


def test_blur_streak_without_measurements_is_silent():
    assert blur_streak(None, 100).offset is None
    assert blur_streak({500: 3.0}, 100).evidence["reason"] == "no_streak_in_window"


@pytest.mark.parametrize(
    "onset,image",
    [(101.5, 102), (101.2, 102), (100.6, 101), (100.0, 101), (100.49, 101)],
)
def test_audio_onset_uses_the_leading_blur_convention(onset, image):
    assert audio_onset_image_frame(onset) == image


def test_audio_signal_stays_inside_the_window():
    from cv.pipeline.contact_frame_refiner import audio_reversal

    assert audio_reversal(101.2, 100).offset == 2
    assert audio_reversal(120.0, 100).offset is None
    assert audio_reversal(None, 100).offset is None


def test_all_signals_returns_every_named_rule():
    track = corner_track(100)
    signals = all_signals(track, 100)
    assert set(signals) == set(SIGNAL_NAMES)


def test_refine_returns_the_corner_frame_and_a_track_pixel():
    track = corner_track(99)
    result = refine(track, 100)
    assert result.frame == 99
    assert result.offset == -1
    assert result.pixel == track[99]
    assert result.abstain is False


def test_refine_accepts_a_float_emitted_frame():
    track = corner_track(99)
    assert refine(track, 100.0).frame == 99


def test_refine_reports_zero_disagreement_when_the_two_voters_agree():
    track = corner_track(99)
    result = refine(track, 100)
    assert result.disagreement_px == pytest.approx(0.0)
    assert result.sigma_px == SIGMA_FLOOR_PX


def test_refine_widens_sigma_when_the_two_voters_disagree():
    emitted = 100
    track = corner_track(emitted - 1)
    # Move one window pixel far away so the two voters land on different images.
    signals = all_signals(track, emitted)
    assert signals["speed_minimum"].offset == -1
    track[emitted + 1] = (track[emitted + 1][0] + 400.0, track[emitted + 1][1])
    result = refine(track, emitted)
    assert result.disagreement_px is None or result.sigma_px >= SIGMA_FLOOR_PX


def test_refine_abstains_when_no_signal_can_answer():
    result = refine({100: (5.0, 6.0)}, 100)
    assert result.abstain is True
    assert result.reason == "no_second_opinion"
    assert result.pixel == (5.0, 6.0)
    assert result.sigma_px == SIGMA_CAP_PX


def test_refine_abstains_when_there_is_no_track_at_all():
    result = refine({}, 100)
    assert result.abstain is True
    assert result.pixel is None


def test_refine_falls_back_to_the_emitted_pixel_when_the_chosen_image_has_no_row():
    track = corner_track(99)
    del track[99]
    result = refine(track, 100)
    assert result.pixel is not None
    assert result.frame in {99, 100}


def test_refine_votes_record_every_signal():
    track = corner_track(99)
    result = refine(track, 100)
    assert set(result.votes) == set(SIGNAL_NAMES)


def test_refine_never_leaves_the_window():
    track = corner_track(100)
    for emitted in range(95, 106):
        result = refine(track, emitted)
        assert result.offset in WINDOW


def test_default_weights_only_reward_the_chosen_voter():
    positive = {name for name, weight in DEFAULT_WEIGHTS.items() if weight > 0.0}
    assert positive == {"direction_change"}
    assert set(DEFAULT_WEIGHTS) == set(SIGNAL_NAMES)


def test_sigma_is_monotone_in_the_disagreement_and_capped():
    assert sigma_from_disagreement(0.0) == SIGMA_FLOOR_PX
    assert sigma_from_disagreement(50.0) > sigma_from_disagreement(10.0)
    assert sigma_from_disagreement(10_000.0) == SIGMA_CAP_PX
    assert sigma_from_disagreement(None) == SIGMA_CAP_PX


def test_disagreement_is_the_pixel_distance_between_the_two_chosen_images():
    track = corner_track(99)
    signals = all_signals(track, 100)
    value = disagreement_px(track, 100, signals)
    first = track[100 + signals["speed_minimum"].offset]
    second = track[100 + signals["kinematic_reversal"].offset]
    assert value == pytest.approx(math.hypot(first[0] - second[0], first[1] - second[1]))


def test_disagreement_is_none_when_a_voter_is_silent():
    assert disagreement_px({100: (0.0, 0.0)}, 100, all_signals({100: (0.0, 0.0)}, 100)) is None


def test_refine_is_pure_and_does_not_mutate_its_inputs():
    track = corner_track(99)
    snapshot = dict(track)
    refine(track, 100)
    assert track == snapshot


def test_weak_turn_keeps_the_emitted_frame():
    """A smooth track has no corner, so there is nothing to prefer over the model."""

    track = corner_track(99, incoming=(40.0, 0.0), outgoing=(38.0, 4.0))
    result = refine(track, 100)
    assert result.offset == 0
    assert result.reason == "weak_turn_keeps_emitted_frame"


def test_a_sharp_turn_moves_the_anchor_off_the_emitted_frame():
    track = corner_track(99)
    result = refine(track, 100)
    assert result.offset == -1
    assert result.reason == "turn_selected_frame"


def test_the_turn_gate_is_a_parameter_and_a_zero_gate_always_follows_the_turn():
    track = corner_track(99, incoming=(40.0, 0.0), outgoing=(38.0, 4.0))
    assert refine(track, 100, turn_gate_deg=0.0).offset == direction_change(track, 100).offset


def test_refine_abstains_when_the_second_opinion_is_a_long_way_from_the_choice():
    """A wide-open track whose two readings land 160+ px apart is refused."""

    emitted = 100
    track = corner_track(emitted - 1, scale=6.0)
    result = refine(
        track,
        emitted,
        streak_px={emitted + offset: (1.0 if offset == 2 else 50.0) for offset in WINDOW},
        weights={"blur_streak": 1.0},
        turn_gate_deg=0.0,
    )
    assert result.votes["blur_streak"] != result.votes["speed_minimum"]
    assert result.sigma_px == SIGMA_CAP_PX
    assert result.abstain is True
    assert result.reason == "uncertainty_beyond_gate"


def test_sigma_uses_the_chosen_image_not_the_turn_rule_when_the_gate_holds():
    track = corner_track(99, incoming=(40.0, 0.0), outgoing=(38.0, 4.0))
    result = refine(track, 100)
    assert result.offset == 0
    assert result.disagreement_px is not None


def test_pose_and_audio_do_not_change_the_shipped_answer():
    track = corner_track(99)
    plain = refine(track, 100)
    with_extras = refine(
        track,
        100,
        {100: [{"joints": {"right_wrist": [0.0, 0.0, 0.9]}}]},
        audio_onset=101.0,
        streak_px={102: 1.0},
    )
    assert with_extras.frame == plain.frame
    assert (
        with_extras.votes["audio"] is not None or with_extras.votes["racket_proximity"] is not None
    )
