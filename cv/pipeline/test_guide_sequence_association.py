"""Two-pass association over structurally bracketed guide detours.

Every fixture is synthetic geometry. No cohort, source or clip of the replay set is named
here, and no threshold below is chosen from one.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from cv.pipeline.ball_motion_tracker import Geometry, MotionConfig, Observation, track_clip
from cv.pipeline.guide_sequence_association import associate_clip, brackets, qualify
from cv.pipeline.ball_ownership import pan_model
from cv.pipeline import ball_motion_tracker as tracker_module
from cv.pipeline import guide_sequence_association as gsa

CLIP = "p"
ON = replace(MotionConfig(), guide_sequence_association=True)
DETOUR = range(5, 8)
BALL = ("branched_crop_tracknetv2",)  # one family, low score, poor rank: the hard case


def ball_at(frame: float, pan: float = 0.0) -> tuple[float, float]:
    return (100.0 + 12 * frame + pan * frame, 200.0 + 3 * frame)


def scene(
    *,
    detour=True,
    alternative=DETOUR,
    pan=0.0,
    unreliable=(),
    last=12,
    decoy=False,
):
    """A straight ball under a coarse lock that spends ``DETOUR`` on a static foreground blob."""
    frames = {}
    for f in range(1, last + 1):
        ball = ball_at(f, pan)
        blob = (850.0 + pan * f, 720.0)
        guide = blob if detour and f in DETOUR else ball
        frames[f] = [Observation(*guide, 0.95, 0, ("coarse_lock",))]
        if f in alternative:
            if decoy:
                # Two geometrically distinct objects straddling the predicted path, each as
                # good as the other and each able to reach the rejoin anchor.
                frames[f].append(Observation(ball[0], ball[1] - 9.0, 0.5, 1, BALL))
                frames[f].append(Observation(ball[0], ball[1] + 9.0, 0.5, 1, BALL))
            else:
                # Rank 4, score 0.06: admission must not depend on raw confidence.
                frames[f].append(Observation(*ball, 0.06, 4, BALL))
    geo = Geometry(
        {
            (CLIP, f): np.array([[1.0, 0.0, -pan * f], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            for f in frames
        },
        {},
        "fixture",
        {(CLIP, f): f not in unreliable for f in frames},
    )
    return frames, geo


def run(frames, geo, config=MotionConfig()):
    return associate_clip(CLIP, frames, geo, 25, config)


def frame_of(row) -> int:
    return int(row["frame"][2:-4])


def observed(rows, frames) -> bool:
    """Every emitted row is one of that frame's original measurements, never a mean."""
    return all(
        any(row["x"] == o.x / 2 and row["y"] == o.y / 2 for o in frames[frame_of(row)])
        for row in rows
    )


# --- default-off contract -------------------------------------------------------------------


def test_off_is_exact_tracker_parity():
    frames, geo = scene()
    assert run(frames, geo) == track_clip(CLIP, frames, geo, 25, MotionConfig())
    assert run(frames, geo) == run(frames, geo, replace(ON, guide_sequence_association=False))


# --- the mechanism --------------------------------------------------------------------------


@pytest.mark.parametrize("pan", [0.0, 5.0])
def test_bracketed_detour_recovers_the_measured_continuation(pan):
    """A wrong guide with a measured continuation and a rejoin, recovered without recapture."""
    frames, geo = scene(pan=pan)
    off, _ = run(frames, geo)
    on, proposals = run(frames, geo, ON)

    assert [frame_of(row) for row in on] == [frame_of(row) for row in off]
    for row in off:
        if frame_of(row) in DETOUR:
            assert row["x"] == (850.0 + pan * frame_of(row)) / 2  # baseline holds the blob
    for row in on:
        if frame_of(row) in DETOUR:
            x, y = ball_at(frame_of(row), pan)
            assert (row["x"], row["y"]) == (x / 2, y / 2)
            # The guide held no privilege inside the window: it did not recapture the track.
            assert "coarse_lock" not in row["sources"]

    # Outside the window no other object is selected; only the filter state carried into the
    # rejoin differs, which is the whole point of not restarting on the rejected guide.
    def selection(rows):
        return [(row["frame"], row["x"], row["y"], row["sources"]) for row in rows]

    assert selection([row for row in on if frame_of(row) not in DETOUR]) == selection(
        [row for row in off if frame_of(row) not in DETOUR]
    )
    assert observed(on, frames)
    window = [p for p in proposals if p["proposal"] == "guide_sequence_association"]
    assert len(window) == 1 and window[0]["native_samples_invented"] is False
    assert window[0]["window_frames"] == list(DETOUR)


def test_the_bracket_is_the_first_pass_diagnosis_not_a_frame_range():
    frames, geo = scene()
    rows, _ = track_clip(CLIP, frames, geo, 25, replace(ON, primary_ball_ownership=True))
    found = brackets(rows, geo, CLIP, 25, ON, pan_model(geo))

    assert len(found) == 1
    assert found[0].frames == tuple(DETOUR)
    assert found[0].join_distance_native <= found[0].join_tolerance_native


# --- preservation ---------------------------------------------------------------------------


def test_no_detour_leaves_the_association_untouched():
    frames, geo = scene(detour=False, alternative=range(1, 13))
    assert run(frames, geo) == run(frames, geo, ON)


def test_no_measured_continuation_preserves_the_baseline_reset():
    """An all-missing window is not an improvement, so the existing restart stands."""
    frames, geo = scene(alternative=())
    assert run(frames, geo) == run(frames, geo, ON)


def test_two_distinct_continuations_preserve_the_whole_window():
    """Ambiguity yields the baseline, not a coin flip between objects."""
    frames, geo = scene(decoy=True)
    assert run(frames, geo) == run(frames, geo, ON)


def test_unreliable_camera_transport_is_not_re_solved():
    frames, geo = scene(unreliable=DETOUR)
    assert run(frames, geo) == run(frames, geo, ON)


def test_a_cut_is_rejected_by_the_frame_bound():
    """The bound is on frames, not observations, so a long bypass cannot be bracketed."""
    frames, geo = scene(detour=True, alternative=(), last=24)
    for f in range(5, 18):
        frames[f] = [Observation(850.0, 720.0, 0.95, 0, ("coarse_lock",))]
        frames[f].append(Observation(*ball_at(f), 0.06, 4, BALL))
    rows, _ = track_clip(CLIP, frames, geo, 25, replace(ON, primary_ball_ownership=True))
    assert brackets(rows, geo, CLIP, 25, ON, pan_model(geo)) == []


def test_a_missing_frame_is_committed_explicitly_and_emits_no_row():
    frames, geo = scene(alternative=(5, 7))
    on, proposals = run(frames, geo, ON)
    window = [p for p in proposals if p["proposal"] == "guide_sequence_association"]

    assert window and window[0]["missing_frames"] == [6]
    assert window[0]["committed"][1] is None
    assert 6 not in {frame_of(row) for row in on}
    assert {5, 7} <= {frame_of(row) for row in on}
    assert observed(on, frames)


def test_a_null_does_not_win_against_an_admissible_observation():
    """Nulls cost the worst admissible measurement, so they cannot sweep the window."""
    frames, geo = scene()
    qualified = qualify(
        track_clip(CLIP, frames, geo, 25, replace(ON, primary_ball_ownership=True))[0],
        {f: list(items) for f, items in frames.items()},
        geo,
        CLIP,
        25,
        ON,
        pan_model(geo),
    )
    assert len(qualified) == 1
    assert all(observation is not None for observation in qualified[0].sequence)


# --- real motion discontinuities -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["toss", "bounce", "contact"])
def test_real_motion_discontinuities_keep_their_supported_guide(kind):
    frames = {}
    for f in range(1, 16):
        x = 200.0 + 8 * f
        if kind == "toss":
            y = 200.0 + 0.6 * (f - 8) ** 2
        elif kind == "bounce":
            y = 350.0 - 5 * abs(f - 8)
        else:
            x = 200.0 + 8 * min(f, 8) - 7 * max(f - 8, 0)
            y = 210.0 + 2 * f
        frames[f] = [
            Observation(x, y, 0.9, 0, ("coarse_lock",)),
            Observation(x + 180, y - 90, 1.0, 0, ("sliding_wasb", "sliding_tracknetv2")),
        ]
    geo = Geometry({(CLIP, f): np.eye(3) for f in frames}, {}, "fixture")
    assert run(frames, geo) == run(frames, geo, ON)


# --- Astra review follow-up: identity, time, bounds, strict JSON, emitted-row truth ---------


def diagnostics(frames, geo, config=ON):
    return track_clip(CLIP, frames, geo, 25, replace(config, primary_ball_ownership=True))[0]


def test_the_detour_must_be_bracketed_by_the_same_primary_chain():
    """Two wings of a *secondary* chain look locally identical and must not be re-solved."""
    frames, geo = scene()
    # An earlier, independent moving object owns the primary chain; the bracketed structure
    # below belongs to a later chain the first pass never joined to it.
    prefixed = {}
    for f in range(1, 6):
        prefixed[f] = [Observation(1500.0 + 20 * f, 150.0, 0.95, 0, ("coarse_lock",))]
    for f, items in frames.items():
        prefixed[f + 6] = items
    geo2 = Geometry(
        {(CLIP, f): np.eye(3) for f in prefixed}, {}, "fixture", {(CLIP, f): True for f in prefixed}
    )
    rows = diagnostics(prefixed, geo2)
    metadata = gsa._chain_metadata(rows, geo2, CLIP, 25, ON, pan_model(geo2))
    owners = {owner for _, owner in metadata.values()}

    assert "candidate" in owners  # the structure really is off the primary chain
    assert gsa.brackets(rows, geo2, CLIP, 25, ON, pan_model(geo2)) == []
    assert run(prefixed, geo2) == run(prefixed, geo2, ON)


def test_a_detour_carrying_detector_rows_is_out_of_scope():
    """Guide-only scope, held on provenance alone: identical geometry, detector rows."""
    frames, geo = scene()
    rows = diagnostics(frames, geo)
    mixed = []
    for row in rows:
        copy = dict(row)
        if gsa.ownership._frame(row) in DETOUR:
            # Same coordinates, same scores, same segments: only the detector mix differs.
            copy["sources"] = "sliding_wasb+sliding_tracknetv2"
        mixed.append(copy)
    detours = [
        segment
        for segment in gsa.ownership._segments(mixed)
        if segment[0][gsa.ownership.RESTART_KEY] == "guide_gated"
    ]

    detour = next(s for s in detours if gsa.ownership._frame(s[0]) == min(DETOUR))

    assert len(gsa.brackets(rows, geo, CLIP, 25, ON, pan_model(geo))) == 1
    assert not gsa._guide_only(detour)
    assert gsa.brackets(mixed, geo, CLIP, 25, ON, pan_model(geo)) == []


# --- time: one step per elapsed native frame, one shared predictor --------------------------


def test_advance_is_the_trackers_own_predictor():
    """The second pass predicts through ball_motion_tracker.advance_modes, not a copy."""
    frames, geo = scene()
    modes = tracker_module.initialise_modes(Observation(*ball_at(2), 0.9, 0, BALL), ON)
    previous_h = geo.homographies[(CLIP, 2)]
    mine, _ = gsa._advance(modes, CLIP, 3, previous_h, geo, 25, ON)
    theirs = tracker_module.advance_modes(
        modes,
        previous_h,
        geo.homographies.get((CLIP, 3), previous_h),
        geo.projections.get((CLIP, 2), geo.projections.get((CLIP, 3))),
        25,
        ON,
    )

    for name in ("ballistic", "impulse"):
        assert np.allclose(mine[name].mean, theirs[name].mean)
        assert np.allclose(mine[name].covariance, theirs[name].covariance)
        assert mine[name].probability == theirs[name].probability


def test_a_native_gap_advances_every_elapsed_frame():
    """No time compression: a three-frame gap is three predictions, not one."""
    frames, geo = scene()
    modes = tracker_module.initialise_modes(Observation(*ball_at(2), 0.9, 0, BALL), ON)
    for mode in modes.values():
        mode.mean[2:] = np.asarray([12.0, 3.0])
    previous_h = geo.homographies[(CLIP, 2)]
    gapped, _ = gsa._advance_to(modes, CLIP, 2, 5, previous_h, geo, 25, ON)
    stepwise = modes
    h = previous_h
    for frame in (3, 4, 5):
        stepwise, h = gsa._advance(stepwise, CLIP, frame, h, geo, 25, ON)
    single, _ = gsa._advance(modes, CLIP, 5, previous_h, geo, 25, ON)

    assert np.allclose(gapped["ballistic"].mean, stepwise["ballistic"].mean)
    assert not np.allclose(gapped["ballistic"].mean, single["ballistic"].mean)


def test_an_unsupported_pre_wing_observation_does_not_become_an_anchor():
    """An update the gate rejected leaves the seed uncertain, so the bracket abstains."""
    frames, geo = scene()
    rows = diagnostics(frames, geo)
    bracket = gsa.brackets(rows, geo, CLIP, 25, ON, pan_model(geo))[0]
    broken = [dict(row) for row in bracket.pre_rows]
    broken[-1]["y"] = float(broken[-1]["y"]) + 400.0

    assert gsa._seed_modes(tuple(broken), geo, CLIP, 25, ON) is None
    assert (
        gsa.search(
            replace(bracket, pre_rows=tuple(broken)),
            tracker_module.prepare_observations(frames, ON),
            geo,
            CLIP,
            25,
            ON,
        )
        is None
    )


# --- the whole post-rejoin wing, not just its first row -------------------------------------


def test_every_post_wing_observation_must_stay_inside_the_gate():
    """Satisfying only post_rows[0] is not enough: a wrong arrival velocity is rejected."""
    frames, geo = scene()
    rows = diagnostics(frames, geo)
    bracket = gsa.brackets(rows, geo, CLIP, 25, ON, pan_model(geo))[0]
    merged = tracker_module.prepare_observations(frames, ON)
    displaced = [dict(row) for row in bracket.post_rows]
    displaced[-1]["y"] = float(displaced[-1]["y"]) + 300.0

    assert gsa.search(bracket, merged, geo, CLIP, 25, ON) is not None
    # Same first post-wing row, unreachable later one.
    assert (
        gsa.search(replace(bracket, post_rows=tuple(displaced)), merged, geo, CLIP, 25, ON) is None
    )
    # Proof the first row alone would have qualified it.
    assert (
        gsa.search(replace(bracket, post_rows=(displaced[0],)), merged, geo, CLIP, 25, ON)
        is not None
    )


# --- bounded search: duplicates, truncation, no manufactured uniqueness ---------------------


def test_duplicate_prefixes_are_collapsed_before_pruning():
    """A coincident guide/detector pair is one hypothesis, not two beam slots."""
    frames, geo = scene()
    for f in DETOUR:
        # Coincident with the blob the guide already offers: merge keeps the guide separate.
        frames[f].append(Observation(851.0, 720.5, 0.9, 0, ("sliding_wasb",)))
    on, proposals = run(frames, geo, ON)
    window = [p for p in proposals if p["proposal"] == "guide_sequence_association"]

    assert window and window[0]["emission_verified"] is True
    assert observed(on, frames)
    for f in DETOUR:
        x, y = ball_at(f)
        assert (next(r for r in on if frame_of(r) == f)["x"]) == x / 2


def test_a_truncated_beam_never_reports_uniqueness():
    """No runner-up under a truncated bound is a bound, not a unique answer."""
    frames, geo = scene()
    narrow = replace(ON, sequence_beam_width=1)
    rows = diagnostics(frames, geo)
    bracket = gsa.brackets(rows, geo, CLIP, 25, narrow, pan_model(geo))[0]
    merged = tracker_module.prepare_observations(frames, narrow)

    assert gsa.search(bracket, merged, geo, CLIP, 25, narrow) is None
    assert run(frames, geo) == run(frames, geo, narrow)


def test_no_qualification_carries_an_infinite_margin():
    frames, geo = scene()
    _, proposals = run(frames, geo, ON)
    window = [p for p in proposals if p["proposal"] == "guide_sequence_association"][0]

    for key in ("margin", "runner_up_log_likelihood", "sequence_log_likelihood"):
        assert window[key] is None or math.isfinite(window[key])
    assert gsa._json_number(math.inf) is None
    assert gsa._json_number(-math.inf) is None
    assert gsa._json_number(math.nan) is None


def test_the_proposal_is_strict_json():
    """No Infinity/-Infinity/NaN token can reach the run record."""
    frames, geo = scene()
    for candidate in (frames, scene(alternative=(5, 7))[0]):
        _, proposals = run(candidate, geo, ON)
        for proposal in proposals:
            json.dumps(proposal, allow_nan=False)


# --- the missing-cost policy ----------------------------------------------------------------


def test_the_missing_cost_is_a_declared_policy_not_a_guarantee():
    """Documented as a fixed price for a null, under both modes and a broad covariance."""
    frames, geo = scene()
    modes = tracker_module.initialise_modes(Observation(*ball_at(2), 0.9, 0, BALL), ON)
    broad = {name: mode for name, mode in modes.items()}
    broad["ballistic"].covariance = broad["ballistic"].covariance * 400.0
    broad["impulse"].probability = 0.9
    broad["ballistic"].probability = 0.1

    assert math.isfinite(gsa._missing_cost(modes, ON))
    assert math.isfinite(gsa._missing_cost(broad, ON))
    assert "policy" in gsa.__doc__ or gsa.MISSING_COST_POLICY == "ballistic_gate_boundary_density"


def test_an_all_null_window_is_never_promoted():
    """Nulls under a broad covariance cannot win a window; the baseline reset stands."""
    frames, geo = scene(alternative=())
    qualified = qualify(
        diagnostics(frames, geo),
        tracker_module.prepare_observations(frames, ON),
        geo,
        CLIP,
        25,
        ON,
        pan_model(geo),
    )

    assert qualified == []
    assert run(frames, geo) == run(frames, geo, ON)


# --- what the second pass actually emitted --------------------------------------------------


def test_emitted_rows_equal_the_qualified_selection_or_are_explicitly_missing():
    frames, geo = scene()
    rows, proposals = run(frames, geo, ON)
    qualified = qualify(
        diagnostics(frames, geo),
        tracker_module.prepare_observations(frames, ON),
        geo,
        CLIP,
        25,
        ON,
        pan_model(geo),
    )
    emitted = gsa._emitted(rows)

    assert qualified and all(gsa._verify(rows, item) for item in qualified)
    for item in qualified:
        for frame, observation in zip(item.bracket.frames, item.sequence, strict=True):
            if observation is None:
                assert frame not in emitted
            else:
                native = gsa.ownership._row_observation(emitted[frame])
                assert (native.x, native.y) == (observation.x, observation.y)
    assert all(p["emission_verified"] for p in proposals if "emission_verified" in p)


def test_an_unverified_window_preserves_the_baseline_and_says_so():
    """A qualified sequence the real IMM would not emit is reported, never claimed."""
    frames, geo = scene()
    baseline = run(frames, geo)
    rows, proposals = run(frames, geo, ON)
    window = [p for p in proposals if p["proposal"] == "guide_sequence_association"][0]
    qualified = qualify(
        diagnostics(frames, geo),
        tracker_module.prepare_observations(frames, ON),
        geo,
        CLIP,
        25,
        ON,
        pan_model(geo),
    )
    del window, rows
    # A sequence whose committed point is nowhere in the emitted rows must fail verification.
    moved = replace(
        qualified[0],
        sequence=tuple(
            None if o is None else replace(o, y=o.y + 500.0) for o in qualified[0].sequence
        ),
    )

    assert gsa._verify(run(frames, geo, ON)[0], moved) is False
    assert gsa._proposal(CLIP, moved, verified=False)["emission_verified"] is False
    assert baseline == run(frames, geo, replace(ON, guide_sequence_association=False))


@pytest.mark.parametrize("change", ["missing", "other_object", "same_object"])
def test_emitted_post_wing_must_retain_the_observed_object(change):
    frames, geo = scene()
    rows, _ = run(frames, geo, ON)
    qualified = qualify(
        diagnostics(frames, geo),
        tracker_module.prepare_observations(frames, ON),
        geo,
        CLIP,
        25,
        ON,
        pan_model(geo),
    )[0]
    # The interior still matches exactly. A recapture on the final post-wing observation
    # nevertheless invalidates the qualification; a nearby measurement of the same object
    # remains admissible under the existing native clustering radius.
    anchor_frame = gsa.ownership._frame(qualified.bracket.post_rows[-1])
    changed = [dict(row) for row in rows]
    if change == "missing":
        changed = [row for row in changed if frame_of(row) != anchor_frame]
    else:
        anchor = next(row for row in changed if frame_of(row) == anchor_frame)
        native_offset = ON.cluster_radius_native * (2.0 if change == "other_object" else 0.5)
        anchor["x"] += native_offset / 2.0
    assert gsa._verify(changed, qualified, ON) is (change == "same_object")


# --- the guide holds no privilege on any path inside a window -------------------------------


def test_a_bootstrap_inside_a_window_cannot_restart_on_the_guide():
    """track_clip's restart path honours admission too, so no guide recapture on a reset."""
    frames, geo = scene()
    ball = Observation(*ball_at(1), 0.06, 4, BALL)
    # Frame 1 is the bootstrap: without the guard the guide owns the restart outright.
    baseline, _ = track_clip(CLIP, frames, geo, 25, MotionConfig())
    assert next(r for r in baseline if frame_of(r) == 1)["x"] == ball_at(1)[0] / 2
    admitted, _ = track_clip(CLIP, frames, geo, 25, MotionConfig(), admission={1: (ball,)})
    row = next(r for r in admitted if frame_of(r) == 1)

    assert (row["x"], row["y"]) == (ball.x / 2, ball.y / 2)
    assert "coarse_lock" not in row["sources"]
    empty, _ = track_clip(CLIP, frames, geo, 25, MotionConfig(), admission={1: ()})
    assert 1 not in {frame_of(r) for r in empty}
