"""Tests for :mod:`cv.pipeline.subframe_timing`.

The synthetic events below are built from the same model the witnesses invert --
two straight legs joined by a 4.5 ms dwell -- so a witness that recovers the
planted time is inverting its own model correctly.  Whether that model matches
tennis is what ``cv/validation/subframe_timing_benchmark.py`` measures against
the bench truth and the owner labels.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cv.pipeline import subframe_timing as st

FPS = 25.0


def synthetic_track(
    impact: float,
    velocity_in: tuple[float, float] = (40.0, 30.0),
    velocity_out: tuple[float, float] = (40.0, -22.0),
    origin: tuple[float, float] = (400.0, 300.0),
    fps: float = FPS,
    frames: range | None = None,
) -> dict[int, tuple[float, float]]:
    """Two straight legs joined by the dwell, sampled on integer frames."""

    dwell = st.dwell_frames(fps)

    def position(t: float) -> tuple[float, float]:
        if t <= impact:
            delta = t - impact
            return (origin[0] + velocity_in[0] * delta, origin[1] + velocity_in[1] * delta)
        delta = max(t - impact - dwell, 0.0)
        return (origin[0] + velocity_out[0] * delta, origin[1] + velocity_out[1] * delta)

    span = frames if frames is not None else range(int(impact) - 6, int(impact) + 7)
    return {frame: position(float(frame)) for frame in span}


# --------------------------------------------------------------------------
# the kinematic corner
# --------------------------------------------------------------------------


@pytest.mark.parametrize("impact", [10.0, 10.15, 10.3, 10.5, 10.75, 10.95])
def test_kinematic_recovers_the_planted_impact_time(impact):
    track = synthetic_track(impact)
    witness = st.witness_kinematic(track, round(impact), FPS)
    assert not witness.abstain
    assert witness.t_subframe == pytest.approx(impact, abs=0.01)


def test_kinematic_reports_both_sides_and_the_dwell_correction():
    track = synthetic_track(10.3)
    witness = st.witness_kinematic(track, 10, FPS)
    detail = witness.detail
    assert detail["t_out"] > detail["t_in"]
    assert detail["t_out_dwell_corrected"] == pytest.approx(detail["t_in"], abs=0.02)
    assert detail["dwell_frames"] == pytest.approx(st.DWELL_SECONDS * FPS)


def test_kinematic_sigma_grows_when_the_two_sides_disagree():
    clean = synthetic_track(10.3)
    noisy = dict(clean)
    noisy[8] = (noisy[8][0] + 25.0, noisy[8][1] - 25.0)
    tight = st.witness_kinematic(clean, 10, FPS)
    loose = st.witness_kinematic(noisy, 10, FPS)
    assert loose.sigma_frames > tight.sigma_frames


def test_kinematic_abstains_without_an_inbound_side():
    track = synthetic_track(10.3, frames=range(10, 17))
    witness = st.witness_kinematic(track, 10, FPS)
    assert witness.abstain
    assert witness.reason == "too_few_track_frames"


def test_kinematic_abstains_when_the_track_does_not_turn():
    straight = {frame: (10.0 * frame, 5.0 * frame) for frame in range(4, 17)}
    witness = st.witness_kinematic(straight, 10, FPS)
    assert witness.abstain
    assert witness.reason == "ill_conditioned_corner"


def test_kinematic_abstains_on_a_stationary_ball():
    still = {frame: (100.0, 100.0) for frame in range(4, 17)}
    witness = st.witness_kinematic(still, 10, FPS)
    assert witness.abstain


def test_kinematic_never_reads_the_impact_frame():
    track = synthetic_track(10.3)
    moved = dict(track)
    moved[10] = (-5000.0, -5000.0)
    assert st.witness_kinematic(track, 10, FPS).t_subframe == pytest.approx(
        st.witness_kinematic(moved, 10, FPS).t_subframe
    )


def test_kinematic_abstains_when_the_corner_is_far_from_the_emitted_frame():
    inbound = st._SideFit(10.0, (0.0, 0.0), (10.0, 10.0), 0.0, (8, 9))
    outbound = st._SideFit(10.0, (0.0, 0.0), (10.0, -10.0), 0.0, (11, 12))
    near = st._corner_witness("kinematic", inbound, outbound, 10, 0.1, st.MIN_SPEED_PX)
    assert not near.abstain
    far = st._corner_witness("kinematic", inbound, outbound, 14, 0.1, st.MIN_SPEED_PX)
    assert far.abstain
    assert far.reason == "corner_far_from_emitted_frame"


def test_a_lower_gate_answers_where_a_higher_one_abstains():
    shallow = synthetic_track(10.3, velocity_in=(40.0, 6.0), velocity_out=(40.0, -2.0))
    strict = st.witness_kinematic(shallow, 10, FPS, min_corner_sin=math.sin(math.radians(45.0)))
    assert strict.abstain
    assert strict.reason == "ill_conditioned_corner"
    assert not st.witness_kinematic(shallow, 10, FPS).abstain


def test_the_shipped_gate_is_the_swept_five_degrees():
    assert st.MIN_CORNER_SIN == pytest.approx(math.sin(math.radians(5.0)), abs=5e-4)


# --------------------------------------------------------------------------
# the court-plane corner
# --------------------------------------------------------------------------


def test_kinematic_court_matches_the_image_under_a_similarity_homography():
    track = synthetic_track(10.3)
    homography = np.array([[0.02, 0.0, -4.0], [0.0, 0.02, -3.0], [0.0, 0.0, 1.0]])
    image = st.witness_kinematic(track, 10, FPS)
    court = st.witness_kinematic_court(track, 10, FPS, homography)
    assert not court.abstain
    assert court.t_subframe == pytest.approx(image.t_subframe, abs=1e-6)


def test_kinematic_court_accepts_a_per_frame_mapping():
    track = synthetic_track(10.3)
    matrix = np.array([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 1.0]])
    per_frame = {frame: matrix for frame in track}
    assert not st.witness_kinematic_court(track, 10, FPS, per_frame).abstain


def test_kinematic_court_abstains_without_a_homography():
    track = synthetic_track(10.3)
    witness = st.witness_kinematic_court(track, 10, FPS, None)
    assert witness.abstain
    assert witness.reason == "no_homography"


def test_kinematic_court_abstains_when_the_window_has_no_matrices():
    track = synthetic_track(10.3)
    assert st.witness_kinematic_court(track, 10, FPS, {}).abstain


# --------------------------------------------------------------------------
# the blur streak
# --------------------------------------------------------------------------


def _planted_streaks(impact, velocity_in, velocity_out, frames, exposure, diameter):
    dwell = st.dwell_frames(FPS)
    return {
        frame: st._two_leg_extent(
            impact,
            frame,
            exposure,
            dwell,
            np.asarray(velocity_in, dtype=float),
            np.asarray(velocity_out, dtype=float),
            diameter,
        )
        for frame in frames
    }


@pytest.mark.parametrize("impact", [9.85, 9.9, 10.0, 10.1])
def test_streak_recovers_an_impact_inside_the_exposure(impact):
    velocity_in, velocity_out = (40.0, 30.0), (40.0, -22.0)
    track = synthetic_track(impact, velocity_in, velocity_out)
    streaks = _planted_streaks(impact, velocity_in, velocity_out, (9, 10, 11), 0.5, 10.0)
    witness = st.witness_streak(track, 10, FPS, streaks)
    assert not witness.abstain
    assert witness.t_subframe == pytest.approx(impact, abs=0.12)


def test_streak_abstains_when_the_impact_falls_in_the_shutter_gap():
    velocity_in, velocity_out = (40.0, 30.0), (40.0, -22.0)
    impact = 10.45
    track = synthetic_track(impact, velocity_in, velocity_out)
    streaks = _planted_streaks(impact, velocity_in, velocity_out, (9, 10, 11), 0.5, 10.0)
    witness = st.witness_streak(track, 10, FPS, streaks)
    assert witness.abstain
    assert witness.reason == "no_shortfall"


def test_streak_reads_a_neighbouring_frame_when_that_is_the_short_one():
    velocity_in, velocity_out = (40.0, 30.0), (40.0, -22.0)
    impact = 10.9
    track = synthetic_track(impact, velocity_in, velocity_out)
    streaks = _planted_streaks(impact, velocity_in, velocity_out, (9, 10, 11, 12), 0.5, 10.0)
    witness = st.witness_streak(track, 10, FPS, streaks)
    assert not witness.abstain
    assert witness.detail["streak_frame"] == 11


def test_streak_abstains_without_a_measurement():
    track = synthetic_track(10.1)
    assert st.witness_streak(track, 10, FPS, {}).abstain


def test_streak_abstains_when_the_ball_is_too_slow():
    slow = synthetic_track(10.1, velocity_in=(0.2, 0.1), velocity_out=(0.1, -0.2))
    witness = st.witness_streak(slow, 10, FPS, {10: 1.0})
    assert witness.abstain
    assert witness.reason == "ball_too_slow"


def test_two_leg_extent_is_the_flight_extent_when_the_impact_is_outside_the_exposure():
    velocity = np.array([40.0, 0.0])
    extent = st._two_leg_extent(20.0, 10, 0.5, 0.1, velocity, velocity, 10.0)
    assert extent == pytest.approx(0.5 * 40.0 + 10.0)


def test_two_leg_extent_is_shortest_when_the_ball_reverses_mid_exposure():
    forward = np.array([40.0, 0.0])
    backward = np.array([-40.0, 0.0])
    reversed_extent = st._two_leg_extent(10.0, 10, 0.5, 0.1, forward, backward, 10.0)
    straight_extent = st._two_leg_extent(20.0, 10, 0.5, 0.1, forward, forward, 10.0)
    assert reversed_extent < straight_extent


# --------------------------------------------------------------------------
# streak measurement from patches
# --------------------------------------------------------------------------


def test_measure_streak_finds_a_planted_bar():
    background = np.full((48, 48), 100.0)
    centre = background.copy()
    centre[24, 10:31] = 255.0
    length = st.measure_streak([background, centre, background], 1)
    assert length == pytest.approx(20.0, abs=1.0)


def test_measure_streak_is_none_without_neighbours():
    assert st.measure_streak([np.zeros((8, 8))], 0) is None


def test_measure_streak_is_none_when_nothing_moved():
    background = np.full((32, 32), 100.0)
    assert st.measure_streak([background, background.copy(), background], 1) is None


def test_calibrate_exposure_recovers_the_planted_line():
    speeds = np.linspace(5.0, 90.0, 60)
    samples = [(float(s), 0.36 * float(s) + 7.5) for s in speeds]
    exposure, diameter, count = st.calibrate_exposure(samples)
    assert exposure == pytest.approx(0.36, abs=1e-6)
    assert diameter == pytest.approx(7.5, abs=1e-6)
    assert count == 60


def test_calibrate_exposure_falls_back_on_too_few_samples():
    exposure, diameter, count = st.calibrate_exposure([(10.0, 15.0)])
    assert exposure == st.DEFAULT_EXPOSURE_FRACTION
    assert diameter == st.DEFAULT_BALL_DIAMETER_PX
    assert count == 1


def test_calibrate_exposure_refuses_a_non_physical_fit():
    samples = [(float(s), 5.0 * float(s)) for s in range(5, 60)]
    exposure, _, _ = st.calibrate_exposure(samples)
    assert exposure == st.DEFAULT_EXPOSURE_FRACTION


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def test_audio_witness_passes_the_onset_through():
    witness = st.witness_audio(10, 10.42)
    assert not witness.abstain
    assert witness.t_subframe == pytest.approx(10.42)
    assert witness.sigma_frames == st.AUDIO_SIGMA_FRAMES


def test_audio_witness_applies_the_convention_offset():
    assert st.witness_audio(10, 10.42, offset_frames=-0.5).t_subframe == pytest.approx(9.92)


def test_audio_witness_abstains_without_an_onset():
    assert st.witness_audio(10, None).abstain


def test_audio_witness_abstains_on_an_onset_from_another_event():
    witness = st.witness_audio(10, 25.0)
    assert witness.abstain
    assert witness.reason == "audio_far_from_emitted_frame"


# --------------------------------------------------------------------------
# the dwell prior
# --------------------------------------------------------------------------


def test_dwell_prior_is_a_normalised_density():
    _, _, grid, density = st.dwell_prior(10, FPS)
    assert float(np.trapezoid(density, grid)) == pytest.approx(1.0)


def test_dwell_prior_is_biased_half_a_dwell_early():
    mean, _, _, _ = st.dwell_prior(10, FPS)
    assert mean < 10.0
    assert mean == pytest.approx(10.0 - 0.5 * st.dwell_frames(FPS), abs=0.02)


def test_dwell_prior_is_tighter_than_the_uniform_baseline():
    _, sigma, _, _ = st.dwell_prior(10, FPS)
    assert sigma < 1.0 / math.sqrt(12.0)


def test_dwell_prior_widens_with_a_longer_exposure():
    _, narrow, _, _ = st.dwell_prior(10, FPS, exposure=0.2)
    _, wide, _, _ = st.dwell_prior(10, FPS, exposure=1.0)
    assert wide > narrow


def test_dwell_prior_keeps_a_floor_outside_the_exposure():
    _, _, grid, density = st.dwell_prior(10, FPS, exposure=0.2)
    outside = density[np.abs(grid - 10.0) > 0.4]
    assert float(outside.min()) > 0.0


def test_dwell_witness_never_abstains():
    assert not st.witness_dwell(10, FPS).abstain


def test_dwell_scales_with_the_frame_rate():
    assert st.dwell_frames(60.0) == pytest.approx(st.dwell_frames(25.0) * 60.0 / 25.0)


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


def _witness(name, t, sigma):
    return st.Witness(name, t, sigma, False, "test")


def test_combine_precision_weights_the_survivors():
    combined, sigma, used, rejected = st.combine(
        [_witness("kinematic", 10.0, 0.1), _witness("dwell", 10.2, 0.2)]
    )
    assert combined == pytest.approx((10.0 / 0.01 + 10.2 / 0.04) / (1 / 0.01 + 1 / 0.04))
    assert set(used) == {"kinematic", "dwell"}
    assert rejected == ()
    assert sigma > 0.0


def test_combine_drops_an_outlier():
    _, _, used, rejected = st.combine(
        [
            _witness("kinematic", 10.0, 0.05),
            _witness("audio", 10.02, 0.05),
            _witness("streak", 14.0, 0.05),
            _witness("dwell", 10.01, 0.2),
        ]
    )
    assert "streak" in rejected
    assert "streak" not in used


def test_combine_never_drops_the_kept_witness():
    _, _, used, rejected = st.combine(
        [_witness("dwell", 20.0, 0.05), _witness("kinematic", 10.0, 0.05)]
    )
    assert "dwell" in used
    assert "dwell" not in rejected


def test_combine_inflates_sigma_when_the_survivors_disagree():
    _, tight, _, _ = st.combine([_witness("kinematic", 10.0, 0.2), _witness("dwell", 10.0, 0.2)])
    _, loose, _, _ = st.combine([_witness("kinematic", 10.0, 0.2), _witness("dwell", 10.4, 0.2)])
    assert loose > tight


def test_combine_ignores_abstentions():
    combined, _, used, _ = st.combine(
        [st.Witness("streak", None, None, True, "no_crops"), _witness("dwell", 10.0, 0.2)]
    )
    assert combined == pytest.approx(10.0)
    assert used == ("dwell",)


def test_combine_returns_nothing_when_every_witness_abstains():
    combined, sigma, used, rejected = st.combine(
        [st.Witness("streak", None, None, True, "no_crops")]
    )
    assert combined is None and sigma is None and used == () and rejected == ()


# --------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------


def test_estimate_beats_the_emitted_frame_on_a_planted_impact():
    impact = 10.35
    track = synthetic_track(impact)
    result = st.estimate(track, 10, "bounce", FPS)
    assert abs(result.t_subframe - impact) < abs(10.0 - impact)
    assert not result.abstain


def test_estimate_answers_from_the_dwell_witness_alone():
    result = st.estimate({}, 10, "contact", FPS)
    assert not result.abstain
    assert result.used == ("dwell",)
    assert result.reason == "single:dwell"


def test_estimate_reports_every_witness_and_its_abstention_reason():
    result = st.estimate({}, 10, "contact", FPS)
    assert set(result.witnesses) == set(st.WITNESS_NAMES)
    assert result.witnesses["streak"].reason == "no_crops"
    assert result.witnesses["audio"].reason == "no_audio_onset"
    assert result.witnesses["kinematic_court"].reason == "not_a_bounce"


def test_estimate_only_runs_the_court_witness_on_bounces():
    track = synthetic_track(10.3)
    matrix = np.eye(3) * np.array([[0.02], [0.02], [1.0]])
    contact = st.estimate(track, 10, "contact", FPS, homography=matrix)
    bounce = st.estimate(track, 10, "bounce", FPS, homography=matrix)
    assert contact.witnesses["kinematic_court"].abstain
    assert not bounce.witnesses["kinematic_court"].abstain


def test_estimate_accepts_measured_streaks_as_floats():
    impact = 10.05
    velocity_in, velocity_out = (40.0, 30.0), (40.0, -22.0)
    track = synthetic_track(impact, velocity_in, velocity_out)
    streaks = _planted_streaks(impact, velocity_in, velocity_out, (9, 10, 11), 0.5, 10.0)
    result = st.estimate(track, 10, "contact", FPS, crops=streaks)
    assert not result.witnesses["streak"].abstain


def test_estimate_measures_streaks_from_patches():
    track = synthetic_track(10.15)
    patches = {}
    for frame in (8, 9, 10, 11, 12):
        patch = np.full((64, 64), 100.0)
        patch[32, 20 : 20 + (6 if frame == 10 else 30)] = 255.0
        patches[frame] = patch
    result = st.estimate(track, 10, "contact", FPS, crops=patches)
    assert not result.witnesses["streak"].abstain


def test_estimate_uses_the_audio_onset():
    result = st.estimate({}, 10, "contact", FPS, audio_onset=10.4)
    assert not result.witnesses["audio"].abstain
    assert result.t_subframe > 10.0


def test_estimate_accepts_the_track_row_shapes_the_pipeline_uses():
    track = synthetic_track(10.3)
    dicts = {frame: {"x": point[0], "y": point[1]} for frame, point in track.items()}
    native = {frame: {"x_native": point[0], "y_native": point[1]} for frame, point in track.items()}
    reference = st.estimate(track, 10, "bounce", FPS).t_subframe
    assert st.estimate(dicts, 10, "bounce", FPS).t_subframe == pytest.approx(reference)
    assert st.estimate(native, 10, "bounce", FPS).t_subframe == pytest.approx(reference)


def test_estimate_clips_to_one_frame_either_side():
    result = st.estimate({}, 10, "contact", FPS, audio_onset=12.5)
    assert 9.0 <= result.t_subframe <= 11.0


def test_estimate_accepts_a_float_event_frame():
    assert st.estimate({}, 10.4, "contact", FPS).event_frame == 10


def test_estimate_exposes_offset_seconds_and_bounds():
    result = st.estimate(synthetic_track(10.3), 10, "bounce", FPS)
    assert result.offset_frames == pytest.approx(result.t_subframe - 10.0)
    assert result.t_seconds == pytest.approx(result.t_subframe / FPS)
    assert result.sigma_seconds == pytest.approx(result.sigma_frames / FPS)
    low, high = result.bounds(2.0)
    assert high - low == pytest.approx(4.0 * result.sigma_frames)


def test_estimate_bounds_are_narrower_than_the_emitted_frame_window():
    result = st.estimate(synthetic_track(10.3), 10, "bounce", FPS)
    low, high = result.bounds(2.0)
    assert high - low < 1.0


def test_estimate_can_disable_a_witness():
    track = synthetic_track(10.3)
    result = st.estimate(track, 10, "bounce", FPS, enabled=("dwell",))
    assert result.witnesses["kinematic"].reason == "disabled"
    assert result.used == ("dwell",)


def test_estimate_is_serialisable():
    payload = st.estimate(synthetic_track(10.3), 10, "bounce", FPS).as_dict()
    assert payload["event_frame"] == 10
    assert set(payload["witnesses"]) == set(st.WITNESS_NAMES)
    assert isinstance(payload["used"], list)


def test_estimate_survives_a_zero_length_track_and_an_unknown_type():
    result = st.estimate({}, 0, "net_hit", 60.0)
    assert not result.abstain
    assert result.event_type == "net_hit"


# --------------------------------------------------------------------------
# the calibrated width
# --------------------------------------------------------------------------


def test_frame_ambiguity_is_large_on_a_sharp_corner_and_small_on_a_straight_run():
    sharp = st.frame_ambiguity(synthetic_track(10.0), 10)
    straight = st.frame_ambiguity(
        synthetic_track(10.0, velocity_in=(40.0, 0.0), velocity_out=(40.0, 0.0)), 10
    )
    assert sharp[0] is not None and straight[0] is not None
    assert sharp[0] > 20.0
    assert abs(straight[0]) < 1.0
    assert sharp[1] == pytest.approx(straight[1], rel=0.5)


def test_frame_ambiguity_abstains_without_neighbours():
    assert st.frame_ambiguity({10: (0.0, 0.0)}, 10) == (None, None)


def test_witness_diagnostics_are_label_free_and_name_the_corner():
    track = synthetic_track(10.3)
    witnesses = {
        "kinematic": st.witness_kinematic(track, 10, FPS),
        "dwell": st.witness_dwell(10, FPS),
    }
    diagnostics = st.witness_diagnostics(
        witnesses, event_frame=10, event_type="contact", fps=FPS, track=track, court_side="near"
    )
    assert diagnostics["kinematic"]["turn_deg"] > 5.0
    assert diagnostics["kinematic"]["is_contact"] == 1.0
    assert diagnostics["kinematic"]["is_far"] == 0.0
    assert diagnostics["kinematic"]["has_partner"] == 0.0
    assert diagnostics["kinematic"]["short_sides"] == 0.0
    assert diagnostics["dwell"]["has_turn_margin"] == 1.0
    assert set(diagnostics) == {"kinematic", "dwell"}


def test_witness_diagnostics_record_the_two_corners_agreement():
    track = synthetic_track(10.3)
    witnesses = {
        "kinematic": st.witness_kinematic(track, 10, FPS),
        "kinematic_court": st.witness_kinematic(track, 10, FPS),
        "dwell": st.witness_dwell(10, FPS),
    }
    diagnostics = st.witness_diagnostics(
        witnesses, event_frame=10, event_type="bounce", fps=FPS, track=track
    )
    assert diagnostics["kinematic"]["has_partner"] == 1.0
    assert diagnostics["kinematic"]["partner_gap_frames"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostics["kinematic"]["is_far"] == 0.5  # court side unknown


def test_sigma_feature_vector_matches_the_declared_feature_order():
    for name, features in st.SIGMA_FEATURES.items():
        row = st.sigma_feature_vector(name, {})
        assert row.shape == (len(features),)
        assert row[0] == 1.0


def test_sigma_feature_vector_refuses_an_unmodelled_witness():
    with pytest.raises(KeyError):
        st.sigma_feature_vector("streak", {})


def test_calibrated_sigma_is_none_without_a_model():
    assert st.calibrated_sigma("kinematic", {}, None) is None


def test_calibrated_sigma_grows_with_the_two_sides_disagreement():
    model = st.CALIBRATED_SIGMA_MODEL
    assert model is not None
    base = {"sigma_raw_frames": 0.02, "sin_turn": 0.5, "min_speed": 30.0, "fps": FPS}
    tight = st.calibrated_sigma("kinematic", {**base, "disagreement_frames": 0.0}, model)
    loose = st.calibrated_sigma("kinematic", {**base, "disagreement_frames": 0.8}, model)
    assert tight is not None and loose is not None
    assert loose > tight


def test_calibrated_sigma_is_clipped_to_the_model_range():
    model = st.CALIBRATED_SIGMA_MODEL
    assert model is not None
    huge = st.calibrated_sigma(
        "kinematic",
        {"sigma_raw_frames": 50.0, "sin_turn": 1e-4, "disagreement_frames": 50.0, "min_speed": 1.0},
        model,
    )
    assert huge == pytest.approx(model.ceiling)


def test_calibrated_sigma_rejects_a_model_of_the_wrong_width():
    broken = st.SigmaModel(coefficients={"dwell": (1.0, 2.0)}, scale={"dwell": 1.0})
    with pytest.raises(ValueError):
        st.calibrated_sigma("dwell", {}, broken)


def test_sigma_model_round_trips_through_a_dict():
    model = st.CALIBRATED_SIGMA_MODEL
    assert model is not None
    restored = st.SigmaModel.from_dict(model.as_dict())
    assert restored.combined_scale == pytest.approx(model.combined_scale)
    for name, row in model.coefficients.items():
        assert list(restored.coefficients[name]) == pytest.approx(list(row))


def test_shipped_model_covers_every_calibrated_witness():
    model = st.CALIBRATED_SIGMA_MODEL
    assert model is not None
    assert set(model.coefficients) == set(st.CALIBRATED_WITNESSES)
    for name, row in model.coefficients.items():
        assert len(row) == len(st.SIGMA_FEATURES[name])


def test_combine_calibrated_abstains_on_the_dwell_witness_alone():
    witnesses = {"dwell": st.witness_dwell(10, FPS)}
    combination = st.combine_calibrated(witnesses, {}, None, event_frame=10)
    assert combination.abstain
    assert combination.reason == "dwell_only"
    assert combination.t_subframe == 10.0
    assert combination.sigma_frames == pytest.approx(st.UNIFORM_SIGMA_FRAMES)


def test_combine_calibrated_abstains_when_the_witnesses_disagree():
    witnesses = {
        "kinematic": st.Witness("kinematic", 10.4, 0.02, False, "corner_intersection", {}),
        "kinematic_court": st.Witness("kinematic_court", 9.7, 0.02, False, "corner", {}),
        "dwell": st.witness_dwell(10, FPS),
    }
    combination = st.combine_calibrated(witnesses, {}, None, event_frame=10, outlier_z=1e9)
    assert combination.abstain
    assert combination.reason == "witnesses_disagree"


def test_combine_calibrated_abstains_when_it_is_no_narrower_than_the_window():
    witnesses = {
        "kinematic": st.Witness("kinematic", 10.2, 3.0, False, "corner_intersection", {}),
        "dwell": st.witness_dwell(10, FPS),
    }
    combination = st.combine_calibrated(witnesses, {}, None, event_frame=10, uniform_sigma=0.30)
    assert combination.abstain
    assert combination.reason == "not_better_than_uniform"
    assert combination.sigma_frames == pytest.approx(0.30)


def test_combine_calibrated_can_be_asked_not_to_apply_the_width_rule():
    witnesses = {
        "kinematic": st.Witness("kinematic", 10.2, 3.0, False, "corner_intersection", {}),
        "dwell": st.witness_dwell(10, FPS),
    }
    combination = st.combine_calibrated(
        witnesses, {}, None, event_frame=10, uniform_sigma=0.30, abstain_above=math.inf
    )
    assert not combination.abstain
    assert combination.reason == "combined"


def test_combine_calibrated_uses_the_fitted_width_and_not_the_claimed_one():
    track = synthetic_track(10.3)
    witnesses = {
        "kinematic": st.witness_kinematic(track, 10, FPS),
        "dwell": st.witness_dwell(10, FPS),
    }
    diagnostics = st.witness_diagnostics(
        witnesses, event_frame=10, event_type="contact", fps=FPS, track=track
    )
    combination = st.combine_calibrated(
        witnesses, diagnostics, st.CALIBRATED_SIGMA_MODEL, event_frame=10
    )
    assert combination.sigmas["kinematic"] != pytest.approx(witnesses["kinematic"].sigma_frames)
    assert combination.sigmas["dwell"] != pytest.approx(witnesses["dwell"].sigma_frames)


def test_estimate_prior_recovers_a_planted_impact_and_says_it_is_calibrated():
    prior = st.estimate_prior(synthetic_track(10.3), 10, "contact", FPS)
    assert prior.calibrated
    assert not prior.abstain
    assert prior.t_subframe == pytest.approx(10.3, abs=0.05)
    assert 0.0 < prior.sigma_frames < st.UNIFORM_SIGMA_FRAMES


def test_estimate_prior_abstains_to_the_uniform_window_without_a_track():
    prior = st.estimate_prior({}, 10, "contact", FPS)
    assert prior.abstain
    assert prior.reason == "dwell_only"
    assert prior.t_subframe == 10.0
    assert prior.sigma_frames == pytest.approx(st.UNIFORM_SIGMA_FRAMES)


def test_estimate_prior_leaves_estimate_alone():
    track = synthetic_track(10.3)
    legacy = st.estimate(track, 10, "contact", FPS)
    prior = st.estimate_prior(track, 10, "contact", FPS)
    assert legacy.sigma_frames != pytest.approx(prior.sigma_frames)
    assert legacy.witnesses["kinematic"].sigma_frames == pytest.approx(
        prior.witnesses["kinematic"].sigma_frames
    )
    assert not legacy.calibrated and prior.calibrated


def test_estimate_prior_agrees_with_combining_estimates_own_witnesses():
    track = synthetic_track(10.3)
    legacy = st.estimate(track, 10, "contact", FPS)
    diagnostics = st.witness_diagnostics(
        legacy.witnesses, event_frame=10, event_type="contact", fps=FPS, track=track
    )
    combination = st.combine_calibrated(
        legacy.witnesses, diagnostics, st.CALIBRATED_SIGMA_MODEL, event_frame=10
    )
    prior = st.estimate_prior(track, 10, "contact", FPS)
    assert prior.t_subframe == pytest.approx(combination.t_subframe)
    assert prior.sigma_frames == pytest.approx(combination.sigma_frames)
    assert prior.abstain == combination.abstain


def test_estimate_prior_reads_the_court_side_from_the_homography():
    track = synthetic_track(10.3)
    near = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -299.0], [0.0, 0.0, 1.0]])
    far = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -279.0], [0.0, 0.0, 1.0]])
    assert st._court_side(track, 10, near) == "near"
    assert st._court_side(track, 10, far) == "far"
    assert st._court_side(track, 10, None) is None
    assert st.estimate_prior(track, 10, "contact", FPS, homography=near).diagnostics["dwell"][
        "is_far"
    ] == pytest.approx(0.0)


def test_estimate_prior_is_serialisable():
    payload = st.estimate_prior(synthetic_track(10.3), 10, "contact", FPS).as_dict()
    assert payload["calibrated"] is True
    assert "diagnostics" in payload and "sigmas_calibrated" in payload
    json_safe = __import__("json").dumps(payload)
    assert "kinematic" in json_safe
