"""Two-pass sequence association over structurally bracketed guide detours.

The first association pass is unchanged.  Where a guide observation is rejected by both
motion models the tracker rebuilds its filter on that rejected guide, and whether the guide
is still the same physical object is decided afterwards by cv.pipeline.ball_ownership.  When
ownership finds the *same primary chain* rejoining immediately after such a restart, the
first pass has itself diagnosed a short detour: a bypassed run of frames bracketed on both
sides by measured rows of one object.

This module re-solves only those brackets, and only from original observed data:

1. **Bracket.**  From ``ball_ownership.assign_ownership`` -- the shared chain/owner
   classification, not local pairwise geometry -- keep runs that opened on a ``guide_gated``
   restart, whose enclosing rows are the *same* ``primary`` chain on both sides, whose own
   rows are guide-only, that are no longer than ``sequence_max_detour_frames`` frames, that
   are *measurably* rejected from the preceding primary (see ``MEASURED_REJECTIONS``), and
   that are immediately bypassed by a ``joined`` verdict between that same preceding primary
   and the following segment, under reliable camera transport across both wings and the
   whole bracket.  Anything unsure -- an ambiguous or insufficient join, a non-primary or
   mixed-chain bracket, a detour carrying detector rows, a missing wing, a held camera --
   keeps the baseline.  No per-source rule exists.

2. **Search.**  Inside the bracket, enumerate bounded sequences over the *full* original
   measured pool (after the existing merge and static-hotspot filter), not the pool the guide
   radius pruned.  Sequences are seeded by running the existing IMM over the pre-departure
   wing, scored by the existing ``_candidate_likelihood`` (existing priors, existing gates,
   both motion modes, so a real contact or bounce impulse is expressible), and required to
   stay inside the existing gate on *every* observed post-rejoin wing observation, each at
   its own original native frame.  A sequence may leave a frame explicitly missing; it may
   never invent a position, and no mean or fitted curve point can become an output.  The beam
   is bounded before any data is read, and the bound's effect is recorded rather than read as
   uniqueness.

3. **Re-associate.**  A qualified sequence is handed back to the ordinary association as a
   per-frame candidate pool (``track_clip(admission=...)``).  The second pass is the same
   code as the first; it simply cannot see the guide inside the qualified window, on any
   path, so the object the first pass reset onto cannot recapture the track there.  What the
   second pass actually emitted is then verified against what was qualified, and a window
   that did not hold is reported as such rather than claimed.

Nothing here consults a label, a downstream event, a cohort, or a per-case choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from cv.pipeline import ball_motion_tracker as tracker
from cv.pipeline import ball_ownership as ownership

# A sequence may leave a frame explicitly missing.  The cost charged for that is the
# log-density of a measurement lying exactly on the ballistic gate.
#
# This is a *declared conservative policy*, not a mathematical ordering.  It is NOT true that
# a null is worth strictly less than every admissible observation: ``_candidate_likelihood``
# mixes the impulse mode, a per-observation measurement covariance and the observation priors
# (score, rank, detector agreement, crop/guide bonuses), any of which can put an admissible
# observation below the ballistic-gate-boundary density -- a broad-covariance impulse-mode
# candidate near its own gate edge is the ordinary example.  What the policy does guarantee is
# a fixed, data-independent price for a null, set before any data is read, so that nulls
# cannot sweep a window merely because log densities are negative.  The remaining risk -- a
# window won by nulls over weak but real observations -- is bounded separately: an all-missing
# sequence is refused outright in ``qualify``.
MISSING_COST_POLICY = "ballistic_gate_boundary_density"

# ball_ownership.join_test outcomes that measurably reject the detour from the preceding
# primary. Everything else it can return -- ``joined``, ``ambiguous``, ``insufficient_wing``
# -- is either agreement or uncertainty, and keeps the baseline association.
MEASURED_REJECTIONS = frozenset({"unjoined", "post_wing_stationary"})


@dataclass(frozen=True)
class Bracket:
    """A detour the first pass diagnosed, with both of its measured wings."""

    segment_index: int
    frames: tuple[int, ...]  # bypassed interior frames, original cadence
    pre_rows: tuple[dict, ...]  # trailing rows of the preceding primary
    post_rows: tuple[dict, ...]  # leading rows of the rejoining segment
    join_distance_native: float
    join_tolerance_native: float
    chain_id: int  # the shared primary chain of both wings, from assign_ownership


@dataclass(frozen=True)
class Qualification:
    bracket: Bracket
    sequence: tuple[tracker.Observation | None, ...]
    score: float
    runner_up: float | None  # None when no geometrically distinct alternative reached
    margin: float | None  # the anchor; never an infinity
    truncated: bool  # the bounded search discarded hypotheses somewhere


def _reliable_span(geometry: tracker.Geometry, clip: str, first: int, last: int) -> bool:
    return all(geometry.is_reliable((clip, frame)) for frame in range(first, last + 1))


def _chain_metadata(
    rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
    pan: ownership.PanModel,
) -> dict[int, tuple[int, str]]:
    """``{segment_id: (chain_id, owner)}`` from the shared ownership classification.

    Keyed by the first pass' own segment identifier, so the original rows -- and therefore
    every original source coordinate, token and score -- are the ones this module goes on to
    read.  ``assign_ownership``'s public rows are used for their classification only.
    """
    ownership_rows, _ = ownership.assign_ownership(rows, geometry, clip, fps, config, pan)
    return {
        int(row["segment_id"]): (int(row["chain_id"]), str(row["owner"])) for row in ownership_rows
    }


def _guide_only(segment: list[dict]) -> bool:
    """Every row of the detour is the coarse lock's own observation."""
    return all(ownership._row_observation(row).is_guide for row in segment)


def brackets(
    rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
    pan: ownership.PanModel,
) -> list[Bracket]:
    """Short guide-only detours that the same *primary chain* immediately rejoins."""
    segments = ownership._segments(rows)
    metadata = _chain_metadata(rows, geometry, clip, fps, config, pan)
    wing = config.reentry_support_frames
    found: list[Bracket] = []
    for index in range(1, len(segments) - 1):
        detour = segments[index]
        if detour[0][ownership.RESTART_KEY] != "guide_gated":
            continue
        before, after = segments[index - 1], segments[index + 1]
        # Identity comes from the shared classification, never from local pairwise geometry:
        # both wings must be the *same* chain and that chain must be the primary one, and the
        # detour must not be part of it.  Without this a bracket of two secondary-object
        # segments, or of two different chains, would look locally identical.
        before_chain, before_owner = metadata[int(before[0][ownership.SEGMENT_KEY])]
        after_chain, after_owner = metadata[int(after[0][ownership.SEGMENT_KEY])]
        detour_chain, _ = metadata[int(detour[0][ownership.SEGMENT_KEY])]
        if before_owner != "primary" or after_owner != "primary":
            continue
        if before_chain != after_chain or detour_chain == before_chain:
            continue
        # Guide-only scope: the bypassed rows are the coarse lock's, which is what this
        # mechanism is for.  A detour carrying detector observations is an ordinary
        # association decision and is left alone.
        if not _guide_only(detour):
            continue
        first, last = ownership._frame(detour[0]), ownership._frame(detour[-1])
        if last - first + 1 > config.sequence_max_detour_frames:
            continue
        pre_end, post_start = ownership._frame(before[-1]), ownership._frame(after[0])
        # Enforced on frames, not on observations: a bracket spanning a cut or a dropped
        # region is rejected by construction, and the bypass must sit inside it.
        if post_start - pre_end - 1 > config.sequence_max_detour_frames:
            continue
        if not (pre_end < first and last < post_start):
            continue
        if post_start - pre_end > config.ownership_join_gap_frames:
            continue
        if len(before) < 2 or len(after) < 2:
            continue
        # Reliable transport across both wings and the whole bracket.  Camera pan is not
        # banned: the homography compensates it, and the join below must reach that
        # compensated mode rather than the raw pan-allowance mode.
        wing_start = ownership._frame(before[-wing:][0])
        wing_end = ownership._frame(after[:wing][-1])
        if not _reliable_span(geometry, clip, wing_start, wing_end):
            continue
        rejected = ownership.join_test(before, detour, geometry, clip, fps, config, pan)
        bypass = ownership.join_test(before, after, geometry, clip, fps, config, pan)
        # Measurably rejected: ownership's two negative verdicts. Its ``ambiguous`` and
        # ``insufficient_wing`` are the unsure ones and keep the baseline.
        if rejected.outcome not in MEASURED_REJECTIONS:
            continue
        # The bypass is the load-bearing one, so it must have been decided under compensated
        # transport rather than under the held camera's raw pan allowance.
        if bypass.outcome != "joined" or bypass.camera_mode != "compensated":
            continue
        found.append(
            Bracket(
                index,
                tuple(range(pre_end + 1, post_start)),
                tuple(before[-wing:]),
                tuple(after[:wing]),
                bypass.distance_native,
                bypass.tolerance_native,
                before_chain,
            )
        )
    return found


def _advance(
    modes: dict[str, tracker.FilterMode],
    clip: str,
    frame: int,
    previous_h: np.ndarray | None,
    geometry: tracker.Geometry,
    fps: float,
    config: tracker.MotionConfig,
) -> tuple[dict[str, tracker.FilterMode], np.ndarray | None]:
    """One IMM mix-and-predict step: the tracker's own ``advance_modes``, not a copy.

    The departure-frame projection argument is spelled exactly as ``track_clip`` spells it
    (``projections[frame - 1]`` falling back to this frame's), so the two passes share one
    predictor rather than two that have to be argued equivalent.
    """
    current_h = geometry.homographies.get((clip, frame), previous_h)
    previous_projection = geometry.projections.get(
        (clip, frame - 1), geometry.projections.get((clip, frame))
    )
    predicted = tracker.advance_modes(
        modes, previous_h, current_h, previous_projection, fps, config
    )
    return predicted, current_h


def _advance_to(
    modes: dict[str, tracker.FilterMode],
    clip: str,
    start: int,
    target: int,
    previous_h: np.ndarray | None,
    geometry: tracker.Geometry,
    fps: float,
    config: tracker.MotionConfig,
) -> tuple[dict[str, tracker.FilterMode], np.ndarray | None]:
    """Advance one step per *elapsed native frame* from ``start`` to ``target``.

    Observations are not necessarily consecutive.  Stepping once per observation would
    compress time and let a wrong arrival velocity look supported; every frame of the gap is
    predicted through, at its own homography.
    """
    for frame in range(start + 1, target + 1):
        modes, previous_h = _advance(modes, clip, frame, previous_h, geometry, fps, config)
    return modes, previous_h


def _update(
    modes: dict[str, tracker.FilterMode],
    observation: tracker.Observation,
    config: tracker.MotionConfig,
) -> tuple[dict[str, tracker.FilterMode], bool]:
    """The tracker's own update, plus whether the observation was inside either gate."""
    _, details = tracker._candidate_likelihood(modes, observation, config)
    total = sum(value[3] for value in details.values())
    if total <= 0.0:
        # Outside both gates: keep the prediction rather than pulling the state onto it, and
        # say so.  An ignored observation must not be read as a supported anchor.
        return modes, False
    return {
        name: tracker.update_mode(mode, observation, details[name][3] / total, config)
        for name, mode in modes.items()
    }, True


def _missing_cost(modes: dict[str, tracker.FilterMode], config: tracker.MotionConfig) -> float:
    """Log-density of a measurement lying exactly on the ballistic gate; see the policy note."""
    mode = modes["ballistic"]
    covariance = mode.covariance[:2, :2] + np.eye(2) * config.guide_measurement_std_native**2
    determinant = max(float(np.linalg.det(covariance)), 1e-12)
    density = mode.probability * math.exp(-0.5 * config.ballistic_gate_d2) / math.sqrt(determinant)
    return math.log(max(density, 1e-300))


def _distinct(
    left: tuple[tracker.Observation | None, ...],
    right: tuple[tracker.Observation | None, ...],
    config: tracker.MotionConfig,
) -> bool:
    """Geometric distinctness at the tracker's own same-object radius."""
    for first, second in zip(left, right, strict=True):
        if (first is None) != (second is None):
            return True
        if first is None or second is None:
            continue
        if math.hypot(first.x - second.x, first.y - second.y) > config.cluster_radius_native:
            return True
    return False


def _branches(
    modes: dict[str, tracker.FilterMode],
    observations: list[tracker.Observation],
    config: tracker.MotionConfig,
) -> tuple[list[tuple[tracker.Observation | None, float]], bool]:
    scored = [
        (observation, tracker._candidate_likelihood(modes, observation, config)[0])
        for observation in observations
    ]
    scored = [item for item in scored if math.isfinite(item[1])]
    # Deterministic order: score, then native position; never the arrival order of a stream.
    scored.sort(key=lambda item: (-item[1], item[0].x, item[0].y))
    truncated = len(scored) > config.sequence_max_branch
    return [*scored[: config.sequence_max_branch], (None, _missing_cost(modes, config))], truncated


def _collapse(hypotheses: list[tuple], config: tracker.MotionConfig) -> list[tuple]:
    """Keep the best of each geometrically equivalent prefix, before any pruning.

    The guide is deliberately a separate stream from the detector clusters near it, so a
    frame routinely offers several observations of the *same* object.  Without this the beam
    fills with duplicate prefixes, the geometrically distinct alternatives fall off the end,
    and their absence at the anchor reads as uniqueness when it is only truncation.
    """
    kept: list[tuple] = []
    for hypothesis in sorted(hypotheses, key=lambda item: -item[1]):
        if any(not _distinct(hypothesis[0], other[0], config) for other in kept):
            continue
        kept.append(hypothesis)
    return kept


def search(
    bracket: Bracket,
    merged: dict[int, list[tracker.Observation]],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
) -> Qualification | None:
    """Bounded beam over the full original measured pool inside one bracket."""
    seeded = _seed_modes(bracket.pre_rows, geometry, clip, fps, config)
    if seeded is None:
        # An unsupported pre-departure wing (an observation outside both gates) gives an
        # uncertain seed, not a strong anchor.  The bracket keeps the baseline.
        return None
    modes, previous_h = seeded
    truncated = False
    beam = [((), 0.0, modes, previous_h)]
    previous_frame = ownership._frame(bracket.pre_rows[-1])
    for frame in bracket.frames:
        nxt = []
        for sequence, score, state, last_h in beam:
            advanced, current_h = _advance_to(
                state, clip, previous_frame, frame, last_h, geometry, fps, config
            )
            branches, branch_truncated = _branches(advanced, merged.get(frame, []), config)
            truncated = truncated or branch_truncated
            for observation, step in branches:
                updated = (
                    advanced if observation is None else _update(advanced, observation, config)[0]
                )
                nxt.append(((*sequence, observation), score + step, updated, current_h))
        if not nxt:
            return None
        # Collapse duplicates *before* the bounded prune, so the beam spends its width on
        # geometrically distinct hypotheses.
        nxt = _collapse(nxt, config)
        truncated = truncated or len(nxt) > config.sequence_beam_width
        beam = nxt[: config.sequence_beam_width]
        previous_frame = frame
    # Termination is the observed post-rejoin wing: the sequence must carry the filter
    # through *every* measured row the first pass associated to the rejoining primary, each
    # at its own native frame.  Satisfying only the first row lets a wrong arrival velocity
    # qualify and be recaptured immediately after the window.
    finished = []
    for sequence, score, state, last_h in beam:
        advanced, current_h, at = state, last_h, previous_frame
        total = score
        admissible = True
        for row in bracket.post_rows:
            row_frame = ownership._frame(row)
            advanced, current_h = _advance_to(
                advanced, clip, at, row_frame, current_h, geometry, fps, config
            )
            at = row_frame
            observation = ownership._row_observation(row)
            terminal = tracker._candidate_likelihood(advanced, observation, config)[0]
            if not math.isfinite(terminal):
                admissible = False
                break
            total += terminal
            advanced, _ = _update(advanced, observation, config)
        if admissible:
            finished.append((sequence, total))
    if not finished:
        return None
    finished.sort(key=lambda item: -item[1])
    best, best_score = finished[0]
    alternative = next(
        (score for sequence, score in finished[1:] if _distinct(best, sequence, config)), None
    )
    if alternative is None:
        if truncated:
            # No distinct alternative survived, but the bounded search discarded hypotheses:
            # the absence of a runner-up is a property of the bound, not of the data, and no
            # margin can be established.  Keep the baseline.
            return None
        # Nothing was discarded, so the enumeration over the branch-bounded pool was
        # exhaustive and there genuinely is no distinct alternative.  Reported as an absent
        # runner-up, never as an infinite margin.
        return Qualification(bracket, best, best_score, None, None, truncated)
    margin = best_score - alternative
    # The existing margin a non-guide candidate must already clear to displace the guide.
    # Ties yield the baseline, never a coin flip.
    if margin < config.guide_replacement_margin:
        return None
    return Qualification(bracket, best, best_score, alternative, margin, truncated)


def _seed_modes(
    pre_rows: tuple[dict, ...],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
) -> tuple[dict[str, tracker.FilterMode], np.ndarray | None] | None:
    """Run the existing IMM over the pre-departure wing, in its own camera frame.

    One prediction step per elapsed native frame, not per observation.  ``None`` when any
    wing observation falls outside both gates: that seed is unsupported, and an ignored
    update must not be laundered into a confident anchor.
    """
    observations = [ownership._row_observation(row) for row in pre_rows]
    frames = [ownership._frame(row) for row in pre_rows]
    modes = tracker.initialise_modes(observations[0], config)
    previous_h = geometry.homographies.get((clip, frames[0]))
    previous_frame = frames[0]
    for frame, observation in zip(frames[1:], observations[1:], strict=True):
        modes, previous_h = _advance_to(
            modes, clip, previous_frame, frame, previous_h, geometry, fps, config
        )
        modes, supported = _update(modes, observation, config)
        if not supported:
            return None
        previous_frame = frame
    return modes, previous_h


def qualify(
    rows: list[dict],
    merged: dict[int, list[tracker.Observation]],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
    pan: ownership.PanModel,
) -> list[Qualification]:
    qualified = []
    for bracket in brackets(rows, geometry, clip, fps, config, pan):
        found = search(bracket, merged, geometry, clip, fps, config)
        if found is None:
            continue
        if all(observation is None for observation in found.sequence):
            # An all-missing window is not an association improvement, it is a suppressor.
            continue
        qualified.append(found)
    return qualified


def _json_number(value: float | None) -> float | None:
    """Strict-JSON scalar: a finite float or null. Never Infinity, -Infinity or NaN."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _proposal(clip: str, qualification: Qualification, verified: bool) -> dict:
    bracket = qualification.bracket
    return {
        "clip": clip,
        "frame": bracket.frames[0],
        "proposal": "guide_sequence_association",
        "window_frames": list(bracket.frames),
        "primary_chain_id": bracket.chain_id,
        "pre_wing_frames": [ownership._frame(row) for row in bracket.pre_rows],
        "post_wing_frames": [ownership._frame(row) for row in bracket.post_rows],
        "bypass_join_distance_native": _json_number(bracket.join_distance_native),
        "bypass_join_tolerance_native": _json_number(bracket.join_tolerance_native),
        "committed": [
            None if observation is None else [observation.x, observation.y]
            for observation in qualification.sequence
        ],
        "missing_frames": [
            frame
            for frame, observation in zip(bracket.frames, qualification.sequence, strict=True)
            if observation is None
        ],
        "sequence_log_likelihood": _json_number(qualification.score),
        # Null, not an infinity: no distinct alternative reached the post-rejoin wing, which
        # is an absent runner-up under a bounded search, not infinite certainty.
        "runner_up_log_likelihood": _json_number(qualification.runner_up),
        "margin": _json_number(qualification.margin),
        "runner_up_available": qualification.runner_up is not None,
        "search_truncated": qualification.truncated,
        "missing_cost_policy": MISSING_COST_POLICY,
        # What the second pass actually emitted, not what the search hoped for.
        "emission_verified": verified,
        "native_samples_invented": False,
    }


def _emitted(rows: list[dict]) -> dict[int, dict]:
    return {ownership._frame(row): row for row in rows}


def _verify(
    rows: list[dict],
    qualification: Qualification,
    config: tracker.MotionConfig = tracker.MotionConfig(),
) -> bool:
    """Did the second pass actually emit the qualified selection, frame by frame?

    The search seeds a fresh filter on the pre-departure wing, while the second pass carries
    its full history into the window, so an admitted candidate can still be rejected by the
    real IMM gate (or the frame can restart, or be dropped by the miss counter).  A proposal
    may only claim what the emitted rows show. The observed post-wing must also survive
    association as the same measured object: reaching it only in the search does not prevent
    the ordinary second pass from recapturing a nearby distractor just outside the window.
    """
    emitted = _emitted(rows)
    for frame, observation in zip(
        qualification.bracket.frames, qualification.sequence, strict=True
    ):
        row = emitted.get(frame)
        if observation is None:
            if row is not None:
                return False
            continue
        if row is None:
            return False
        native = ownership._row_observation(row)
        if not (
            math.isclose(native.x, observation.x, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(native.y, observation.y, rel_tol=0.0, abs_tol=1e-6)
        ):
            return False
    for anchor in qualification.bracket.post_rows:
        row = emitted.get(ownership._frame(anchor))
        if row is None:
            return False
        actual = ownership._row_observation(row)
        observed = ownership._row_observation(anchor)
        if math.hypot(actual.x - observed.x, actual.y - observed.y) > config.cluster_radius_native:
            return False
    return True


def associate_clip(
    clip: str,
    frame_observations: dict[int, list[tracker.Observation]],
    geometry: tracker.Geometry,
    fps: float,
    config: tracker.MotionConfig = tracker.MotionConfig(),
) -> tuple[list[dict], list[dict]]:
    """``track_clip`` with the bracketed second pass; drop-in for the one-pass call."""
    if not config.guide_sequence_association:
        return tracker.track_clip(clip, frame_observations, geometry, fps, config)
    # The segment/restart instrumentation is the only thing this borrows from the ownership
    # arm, and it adds no behaviour of its own to the first pass.
    instrumented = replace(config, primary_ball_ownership=True)
    diagnostic_rows, _ = tracker.track_clip(clip, frame_observations, geometry, fps, instrumented)
    qualified = (
        qualify(
            diagnostic_rows,
            tracker.prepare_observations(frame_observations, config),
            geometry,
            clip,
            fps,
            config,
            ownership.pan_model(geometry),
        )
        if diagnostic_rows
        else []
    )
    if not qualified:
        return tracker.track_clip(clip, frame_observations, geometry, fps, config)
    admission: dict[int, tuple[tracker.Observation, ...]] = {}
    for qualification in qualified:
        for frame, observation in zip(
            qualification.bracket.frames, qualification.sequence, strict=True
        ):
            admission[frame] = () if observation is None else (observation,)
    rows, proposals = tracker.track_clip(
        clip, frame_observations, geometry, fps, config, admission=admission
    )
    unverified = [
        qualification for qualification in qualified if not _verify(rows, qualification, config)
    ]
    if unverified:
        # At least one window did not come out of the real association as it was qualified.
        # The baseline is returned whole -- no partial commit, no fabricated row, no forced
        # zero innovation -- and the failure is reported instead of claimed as a commit.
        rows, proposals = tracker.track_clip(clip, frame_observations, geometry, fps, config)
        proposals.extend(
            _proposal(clip, qualification, verified=False) for qualification in unverified
        )
        return rows, proposals
    proposals.extend(_proposal(clip, qualification, verified=True) for qualification in qualified)
    return rows, proposals
