"""Anchor-first tennis flight fitting with exact metric event anchors.

The default objective fits one physical state directly in camera space.  A known
bounce is an exact piecewise-integration knot.  Odd frames are never in the objective
and provide the acceptance witness.  The former per-frame height-on-ray objective is
retained only as an explicit compatibility arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from cv.pipeline.flight_anchors import (
    ANCHOR_TIME_BOUND_FRAMES,
    ANCHOR_TIME_SIGMA_FRAMES,
    BALL_RADIUS_M,
    DEFAULT_PIXEL_SIGMA,
    NET_COURT_Y_M,
    interpolate_track,
    net_tape_height,
    ray_at_height,
)
from cv.pipeline.rich_ball_physics import (
    ShotFit,
    bounce_velocity,
    contact_prior,
    project_one,
    simulate_fixed_bounce_with_spin,
    simulate_hard_bounce_knot,
    spin_vector,
)
from physics.bounce_reference import DWELL_SECONDS, PROVENANCE, court_bounce
from physics.flight import sample_states, sample_states_batch
from physics.flight import sample_signed_states as _sample_signed_states

PHYSICS_NODE_SIGMA_M = 0.10
ANCHOR_SIGMA_MAX_M = 0.03
ANCHOR_SIGMA_MIN_M = 0.01
CONTACT_ANCHOR_SIGMA_M = 0.025
SHARED_CONTACT_SIGMA_M = 0.001
SOFT_CONTACT_SIGMA_MIN_M = 0.03
SOFT_CONTACT_SIGMA_MAX_M = 0.20
MAX_CONTACT_REACH_M = 2.1
# The contact row of the flight objective.  ``interpolate_track`` reads a fractional contact
# frame by drawing a chord between the frames either side of it -- across the racket impact,
# where the ball's image path has a corner.  On the synthetic bench's PERFECT track that chord
# is a median of 6.0 px from the true contact pixel with a p90 of 20.7 px, past the bench's own
# 12 px contact tolerance on 19.5% of its 579 contacts, and the objective fits it at the same
# weight as a real observation.  Two better witnesses were already in the contact row and were
# not read: the event emission's own image coordinate, and the contact-frame refiner's pixel
# with its measured sigma.  Failing both, extrapolating each side's two nearest observations to
# the contact frame and averaging the two lands a median of 1.1 px from truth, and the two
# sides' disagreement is that estimate's own error bar.
CONTACT_OBSERVATION_MIN_SIGMA_PX = 1.0
# What an image observation that states no error bar is worth.  ``reconstruction`` already turns
# a contact observation's sigma into a confidence as ``10 / sigma_px``, so ten pixels is this
# pipeline's own unit of contact-pixel doubt.
CONTACT_OBSERVATION_DEFAULT_SIGMA_PX = 10.0
# The chord's own measured error on a perfect track: a median of 6.0 px, a p90 of 20.7 px.
CONTACT_OBSERVATION_CHORD_SIGMA_PX = 8.0
CONTACT_OBSERVATION_MAX_GAP_FRAMES = 3.0
# The spin regulariser.  The shipped term is ``spin / 300`` on each Cartesian component: a prior
# centred on a ball with no spin at all and a 300 rad/s (2,865 rpm) sigma.  A real rally ball
# carries 1,500-2,500 rpm of topspin, so that term charges the truth for being physical, and it
# buys the discount back along the one direction the pixels cannot see -- the contact ray.
# ``physics/bounce_reference.py``'s provenance carries the measured incoming topspin of the
# Hawk-Eye corpus by surface, which is exactly the quantity this parameter is on a bounce
# flight: 1,870 rpm on hard and 2,184 rpm on clay over 2,616 measured impacts.  The sigma is a
# judgement, not a measurement: one unimodal prior has to cover a slice at -1,900 rpm and a lob
# at +3,200 rpm (``physics_interpretation.typical_spin_prior``), so it is wider than that
# module's per-profile 850-950 rpm.  The sidespin and rifle sigmas are that module's own.
SPIN_PRIOR_TOPSPIN_RPM = dict(PROVENANCE["incoming_spin_prior_rpm"])
SPIN_PRIOR_DEFAULT_TOPSPIN_RPM = 1900.0
SPIN_PRIOR_TOPSPIN_SIGMA_RPM = 1200.0
SPIN_PRIOR_SIDESPIN_SIGMA_RPM = 1000.0
SPIN_PRIOR_RIFLE_SIGMA_RPM = 650.0
# Striker witness.  A racket contact happens at arm-plus-racket length from the striker's body,
# and in front of it: the tracked player box gives that body a court position, so the contact is
# constrained both in how far it is from the body and on which side of the body it lies.  Both
# constraints are bands the fitter believes without argument, with the miss outside them charged
# at ``STRIKER_CONTACT_SIGMA_M`` per metre.
#
# The synthetic bench's own truth geometry, measured over its 579 contacts through
# ``reconstruction.load_players``, is: reach 0.33-1.26 m and 0.10-1.17 m in front on a rally
# contact; reach 0.05-0.48 m and 0.00-0.41 m in front on a serve.  The bands widen that on both
# ends, because a real stretched contact reaches further than any bench shot.
STRIKER_REACH_BAND_M = (0.15, 1.70)
SERVE_STRIKER_REACH_BAND_M = (0.0, 0.80)
STRIKER_BEHIND_SLACK_M = 0.35
# The witness may not argue harder than it is itself accurate.  The real tracker's player-box root
# carries about 3.5 px540 of error, and docs/wk1/gate_audit.md section 1 measures a native pixel as
# 0.0195 m of court on the near half and 0.0323 m on the far half, so 3.5 px540 is 7 native pixels
# and therefore 0.14 m near and 0.23 m far.  That, not the bench's own 0.014 m root error, is the
# number the prior is scaled by.
STRIKER_ROOT_SIGMA_M = {"near": 0.14, "far": 0.23}
STRIKER_CONTACT_SIGMA_M = 0.23
# How far the body may have walked since the last frame the tracker saw it.  The real cohort's
# per-frame court step is a median of 0.042 m and a p90 of 0.155 m; a recovering player at
# 2.5 m/s is 0.10 m per frame, and past ten frames of that the witness is a metre wide and is
# dropped rather than argued with.
STRIKER_DRIFT_M_PER_FRAME = 0.10
STRIKER_WITNESS_MAX_GAP_FRAMES = 10.0
# The single reach hinge the fitter shipped with, kept for the default arm.
SERVE_CONTACT_REACH_M = 1.6
MIN_TRAINING_FRAMES = 4
HELD_OUT_MEDIAN_LIMIT_PX = 8.0
HELD_OUT_P90_LIMIT_PX = 12.0
ANCHOR_ERROR_LIMIT_M = 0.12
JOINT_HELD_OUT_RELATIVE_TOLERANCE = 0.0
JOINT_HELD_OUT_MEDIAN_ABSOLUTE_TOLERANCE_PX = 3.0
JOINT_HELD_OUT_P90_ABSOLUTE_TOLERANCE_PX = 6.0
# About 1,700 rpm, the nominal rally topspin docs/wk1/dev_loop.md re-integrates truth with.
NOMINAL_TOPSPIN_RAD_S = 180.0
SHARED_CONTACT_HEIGHT_OFFSETS_M = (-0.3, 0.0, 0.3)
SHARED_CONTACT_TIME_OFFSETS_FRAMES = (-1.0, 0.0, 1.0)
CONTACT_BOUNDARY_TIME_LIMIT_FRAMES = 1.0
CONTACT_RAY_SEARCH_MAX_EVALUATIONS = 12
CONTACT_RAY_REFIT_CANDIDATES = 1
ONE_SIDED_CONTACT_MAX_PASSES = 2
ONE_SIDED_CONTACT_BOUNCE_FREE_PENALTY = 25.0
ONE_SIDED_CONTACT_MARGINAL_TRIGGER_M = 0.15
ONE_SIDED_CONTACT_MARGINAL_PIXEL_LIMIT_PX = 24.0
ONE_SIDED_CONTACT_OPPOSITE_ENDPOINT_MAX_MOVE_M = 0.20
ONE_SIDED_CONTACT_TRIGGER_M = 0.25
ONE_SIDED_CONTACT_TARGET_M = 0.20
SERVE_CONTACT_HEIGHT_RANGE_M = (2.5, 3.0)
SERVE_CONTACT_HEIGHT_SIGMA_M = 0.05
CONTACT_ADJACENT_OBSERVATIONS = 2
CONTACT_ADJACENT_WEIGHT = 0.1
TIMING_OFFSET_LIMIT_FRAMES = 0.5
TIMING_OFFSET_SIGMA_FRAMES = 0.25
TIMING_NUISANCE_MODES = frozenset({"none", "per_observation", "flight_constant", "field_phase"})
NET_CROSSING_MAX_HEIGHT_M = 4.5
NET_PLANE_SIGMA_M = 0.0001
NET_HEIGHT_BARRIER_SIGMA_M = 0.002
NET_BRACKET_SIGMA_M = 0.001
NET_DIRECTION_SPEED_MIN_MPS = 0.10
NET_DIRECTION_SPEED_SIGMA_MPS = 0.05
# Numerical feasibility threshold, aligned with the existing net-violation gate.
NET_CONSTRAINT_TOLERANCE_M = 0.01
DEFAULT_NET_POINT_ANCHOR = True
# Sub-frame impact anchors.  A bounce emission names a frame and a pixel.  The ball dwells
# 4-5 ms on the court, so the emitted frame really does show the ball at the impact position --
# but the impact instant is sub-frame, and half a frame either side of it the ball is 0.1-0.3 m
# above the court.  Pinning a flight through the ray/plane intersection of an integer-frame
# pixel therefore hands the fitter a knot that ``docs/wk1/bench_frames.md`` measures at a median
# of 0.143 m and a p90 of 0.836 m from the true bounce, and the objective propagates through it
# exactly.  With this arm the bounce keeps the one thing that is exact -- it is on the court
# plane -- while its horizontal position and its sub-frame time become free parameters, the
# emitted pixel becomes an ordinary ray observation of the ball on its own emitted frame, and
# the time carries the quantisation-plus-measured-timing prior from ``flight_anchors``.
#
# The bound on the horizontal move is one frame of ball travel: the objective's own speed
# ceiling is 72 m/s and the slowest supported broadcast is 25 fps, so nothing physical can put
# the impact further than about 2.9 m from the pixel that showed it.
SUBFRAME_ANCHOR_POSITION_BOUND_M = 3.0
# What the emitted impact pixel is worth as an image observation.  The tracked ball's own error
# is a median of 1.30 px (docs/wk1/s6_bench.md, 1,072 owner-positioned frames) and the real
# event model's x/y head sits a median of 4.47 px from the track on the same frame
# (docs/wk1/bench_frames.md section 2), so the emission is the weaker of the two witnesses and
# is weighted as such rather than at the tracked ball's weight.
SUBFRAME_ANCHOR_PIXEL_SIGMA_PX = DEFAULT_PIXEL_SIGMA
# How hard the fitted impact is held onto the court plane.  This is the one exact fact about a
# bounce, so it is held at the tightest anchor sigma the fitter already uses.
SUBFRAME_ANCHOR_PLANE_SIGMA_M = ANCHOR_SIGMA_MIN_M
# How many arc evaluations the seam-time search is allowed.  The gap between two arcs either
# side of an impact is smooth and close to V-shaped in time, so a bounded scalar search reaches
# a hundredth of a frame well inside this budget.
SEAM_TIME_SEARCH_MAX_EVALUATIONS = 24
SUBFRAME_TERMINAL_TIME_SEEDS_FRAMES = (0.0, -0.5, 0.5)
# What a frame the seam search cannot sample either arc on is worth.  A finite number keeps the
# bounded scalar search arithmetic real; it is far outside every tolerance the seam is judged
# against, so such a frame is never chosen.
SEAM_TIME_UNSAMPLED_GAP_M = 1.0e6
# The owner's shape for the bounce observation (owner note, 2026-09-04, and the circle
# ``docs/wk1/witness_circle.md`` section 2 already uses in the audit): the fitted impact is
# completely free inside a circle around the sub-frame court-plane witness, then pays a small
# penalty that grows with distance, and is never pinned.  The circle is
# ``max(0.20 m, 2 x sigma)`` around ``flight_anchors.bounce_court_witness``; the residual is
# ``max(0, d - r) / r``, so it is exactly zero at and inside the rim, its slope there is
# 1 / r, and a bounce a whole radius outside costs one unit against a tracked pixel's two.
BOUNCE_WITNESS_FLOOR_M = 0.20
BOUNCE_WITNESS_SIGMAS = 2.0
# How far the multistart spread is probed for evidence: every seed that converged is compared
# against the chosen solution.  This is a report, not a decision -- the fitter still takes the
# cheapest solution, exactly as before.
MULTISTART_EVIDENCE_MAX_SOLUTIONS = 12
# The junction the whole-point joint solve holds two adjacent flights to.  A fifth of
# ``metric_v2``'s 0.25 m junction tolerance, so a seam at that tolerance costs five units
# against a tracked pixel's two and the solve has a reason to close it.
WHOLE_POINT_JUNCTION_SIGMA_M = 0.05
# How many flights a point may have before the joint solve is not attempted.  The joint residual
# integrates every flight on every evaluation and the parameter count grows with the point, so
# the cost grows faster than linearly; five flights is the brief's limit and covers the bench's
# and the cohort's ordinary points.
WHOLE_POINT_MAX_FLIGHTS = 5
# The joint solve's own evaluation budget.  It is seeded at the per-flight solutions, so it is
# a refinement and not a search.
WHOLE_POINT_MAX_NFEV = 12
_DOWNWEIGHT_CONTACT_ADJACENT = False
_LEGACY_HEIGHT_ON_RAY = False


def simulate_measured_bounce_knot(
    incoming_velocity: np.ndarray,
    incoming_spin: np.ndarray,
    f0: float,
    frames: np.ndarray,
    fps: float,
    surface: str,
    bounce: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], tuple[np.ndarray, ...]]:
    """Propagate through an exact bounce using the measured rebound model."""
    bounce_frame = float(bounce["frame"])
    bounce_xyz = np.asarray(bounce["x"], float)
    query = np.asarray(frames, float)
    before = query <= bounce_frame
    positions = np.empty((len(query), 3), float)
    velocities = np.empty_like(positions)
    spins = np.empty_like(positions)
    # The pre-bounce frames and the flight's own start state are all on the incoming
    # side of the knot, so they are sampled together and share one integration.
    incoming_times = np.concatenate(
        [(query[before] - bounce_frame) / fps, [(f0 - bounce_frame) / fps]]
    )
    incoming = _sample_signed_states(bounce_xyz, incoming_velocity, incoming_spin, incoming_times)
    if np.any(before):
        positions[before], velocities[before], spins[before] = (value[:-1] for value in incoming)
    # Additive: a bare surface name ignores the landing point (physics.surface_model).
    rebound = court_bounce(incoming_velocity, incoming_spin, surface, position=bounce_xyz)
    if np.any(~before):
        elapsed = np.maximum((query[~before] - bounce_frame) / fps - DWELL_SECONDS, 0.0)
        values = sample_states(bounce_xyz, rebound.velocity, rebound.spin, elapsed)
        positions[~before], velocities[~before], spins[~before] = values
    initial = tuple(value[-1:] for value in incoming)
    record = {
        "frame": bounce_frame,
        "x": bounce_xyz,
        "v_in": np.asarray(incoming_velocity, float),
        "v_out": rebound.velocity,
        "w_in": np.asarray(incoming_spin, float),
        "w_out": rebound.spin,
        "regime": f"measured_{rebound.regime}",
        "sampling_model": "measured_bounce_v1",
        "dwell_seconds": DWELL_SECONDS,
    }
    return positions, velocities, spins, [record], tuple(value[0] for value in initial)


_STRIKER_PRIOR = False
_STRIKER_AUTHORITY = True
_CONTACT_OBSERVATION_WITNESS = False
_CONTACT_OBSERVATION_SIGMA = False
_PHYSICAL_SPIN_PRIOR = False
_SUBFRAME_ANCHORS = False
_SUBFRAME_CONTACTS = False
# The two separable halves of ``--subframe-contacts``.  ``docs/wk1/point_fit7.md`` section 12
# names splitting them as the next attribution and this package makes it: the seam-time adoption
# records the instant two already-agreeing arcs meet and, in doing so, stands the soft one-sided
# reconciliation down at that contact; the advance carries the shared contact off the emitted
# pixel's ray to the fitted instant.  Both are on inside the flag, so ``--subframe-contacts``
# alone is exactly what pass 7 measured.
_SUBFRAME_CONTACT_SEAM = True
_SUBFRAME_CONTACT_ADVANCE = True
# Whether an adopted seam time also stands the soft one-sided reconciliation down at that
# contact.  With it off the seam time is recorded and the soft pass still runs, which separates
# "the seam moved" from "the refit did not happen".
_SUBFRAME_CONTACT_SEAM_SKIPS_REFIT = True
# Whether a seam time is adopted only where an independent witness agrees with it.  Two arcs
# that are both wrong in depth still meet somewhere -- they meet along the camera ray -- so
# "the arcs already agree" is not on its own evidence that the impact was at that instant.
# With this on, the seam is adopted only when the contact carries a witnessed sub-frame time
# prior (``flight_anchors.impact_time_prior``, which needs ``--subframe-time-priors``) and the
# seam instant lies inside that prior's own window.
_SUBFRAME_CONTACT_SEAM_WITNESSED = False
# How many prior sigmas the seam instant may sit from the witnessed impact time.
SEAM_WITNESS_AGREEMENT_SIGMAS = 2.0
_SUBFRAME_PLANE_ANCHOR_ERROR = False
# Whether the fitted bounce is observed against the track's own sub-frame court-plane corner,
# in the owner's shape: free inside ``max(0.20 m, 2 sigma)`` of it, a growing penalty outside,
# never a pin.  Needs ``--subframe-anchors``, because with the shipped knot the bounce is not a
# free parameter and there is nothing for an observation to move.
_SUBFRAME_BOUNCE_WITNESS = False
# Whether the point is solved once, jointly, over every flight and every shared contact state,
# instead of by the alternating one-sided-then-soft reconciliation.  The alternating scheme
# stays as the fallback whenever the joint solve does not converge or its guard refuses it.
_WHOLE_POINT_JOINT = False
# Diagnostic hook.  ``cv/validation/s6_identifiability.py`` installs a callback here so it can
# evaluate this fitter's own objective at states the fitter did not choose -- the synthetic
# truth, above all -- without rebuilding the objective outside the fitter and hoping the copy
# stayed faithful.  Nothing in the pipeline sets it, and with it unset the fit is unchanged.
_OBJECTIVE_PROBE = None


def configure_contact_adjacent_weighting(enabled: bool) -> None:
    """Configure the explicit contact-occlusion arm before point workers fork."""
    global _DOWNWEIGHT_CONTACT_ADJACENT
    _DOWNWEIGHT_CONTACT_ADJACENT = bool(enabled)


def configure_striker_witness(*, prior: bool, authority: bool) -> None:
    """Configure the striker-witness arms before point workers fork.

    ``prior`` puts the tracked striker's court position into the flight objective as a reach and
    front-of-body band; ``authority`` lets that same witness decide which side of a bad seam is
    believed when the automatic contact pixel is the only other evidence.

    ``authority`` is on by default and ``prior`` is off, and ``WK3_REPORT.md`` section 4 is why:
    measured separately, ``authority`` rises on the clean bench and leaves the real cohort's
    accepted flights untouched to the last flight, while ``prior`` is worth seven more complete
    clean points and eight more wrong accepted flights on real video, which is the owner's
    unrecoverable error.
    """
    global _STRIKER_PRIOR, _STRIKER_AUTHORITY
    _STRIKER_PRIOR = bool(prior)
    _STRIKER_AUTHORITY = bool(authority)


def configure_contact_observation_witness(enabled: bool) -> None:
    """Configure the contact-observation arm before point workers fork.

    With it on, a flight's contact row is the contact's own image witness at that witness's own
    error bar, instead of the tracked ball interpolated across the racket impact at full weight.
    """
    global _CONTACT_OBSERVATION_WITNESS
    _CONTACT_OBSERVATION_WITNESS = bool(enabled)


def configure_contact_observation_sigma(enabled: bool) -> None:
    """Configure the weight-only arm before point workers fork.

    This is the smaller half of the contact-observation change: the row keeps the chord
    ``interpolate_track`` draws across the impact, and only stops being weighted as though that
    chord were an observation.  It leaves the fitter and the point gate looking at two different
    contact witnesses, which is what makes the gate independent evidence.
    """
    global _CONTACT_OBSERVATION_SIGMA
    _CONTACT_OBSERVATION_SIGMA = bool(enabled)


def contact_row_sigma_px_value(contact: dict, track: dict[int, np.ndarray]) -> float:
    """What the chord at a contact frame is worth, without changing which pixel it is.

    A contact that falls on an observed frame *is* an observation.  A contact between two frames
    is a chord across the racket impact, and the bench measures that chord at a median of 6.0 px
    from the true contact with a p90 of 20.7 px.
    """
    _, sources = interpolate_track(track, float(contact["frame"]))
    if len(sources) == 1:
        return CONTACT_OBSERVATION_MIN_SIGMA_PX
    return CONTACT_OBSERVATION_CHORD_SIGMA_PX


def configure_physical_spin_prior(enabled: bool) -> None:
    """Configure the spin-prior arm before point workers fork.

    With it on, the fitted spin is charged against the measured incoming topspin of the surface
    in velocity-aligned components, instead of against a ball with no spin at all.
    """
    global _PHYSICAL_SPIN_PRIOR
    _PHYSICAL_SPIN_PRIOR = bool(enabled)


def configure_subframe_anchors(enabled: bool) -> None:
    """Configure the sub-frame impact-anchor arm before point workers fork.

    With it on, a bounce anchor is a weighted observation instead of an exact knot: the impact
    stays on the court plane, its horizontal position and its sub-frame time are fitted, and the
    emitted pixel enters the objective as a ray observation of the ball on the emitted frame.
    The terminal court-plane anchor of a rally-closing flight gets the same treatment.
    """
    global _SUBFRAME_ANCHORS
    _SUBFRAME_ANCHORS = bool(enabled)


def subframe_anchor_geometry(anchor: dict, camera) -> dict | None:
    """The observation a bounce emission actually is, or ``None`` if it cannot be read.

    ``image_xy`` and ``frame`` are what the emitter said.  ``plane_z_m`` is the height the
    impact is held at.  ``seed_xy`` is the ray/plane intersection, kept only as the starting
    point of the search.
    """
    pixel = anchor.get("image_xy")
    if pixel is None:
        return None
    pixel = np.asarray(pixel, dtype=float)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        return None
    observation_frame = float(anchor.get("observation_frame", anchor["frame"]))
    plane_z = float(anchor.get("plane_z_m", BALL_RADIUS_M))
    seed = np.asarray(anchor["x"], float)
    if not np.all(np.isfinite(seed)):
        return None
    try:
        projection = np.asarray(camera.p_at(observation_frame), dtype=float)
    except (ValueError, KeyError, TypeError):
        return None
    if not np.all(np.isfinite(projection)):
        return None
    bounds = anchor.get("time_bounds_frames") or [
        float(anchor["frame"]) - ANCHOR_TIME_BOUND_FRAMES,
        float(anchor["frame"]) + ANCHOR_TIME_BOUND_FRAMES,
    ]
    return {
        "frame": float(anchor["frame"]),
        "observation_frame": observation_frame,
        "projection": projection,
        "pixel": pixel,
        "pixel_sigma_px": max(
            float(anchor.get("pixel_sigma") or SUBFRAME_ANCHOR_PIXEL_SIGMA_PX), 0.5
        ),
        "plane_z_m": plane_z,
        "seed_xy": seed[:2].copy(),
        "time_lower_frames": float(bounds[0]) - float(anchor["frame"]),
        "time_upper_frames": float(bounds[1]) - float(anchor["frame"]),
        "time_sigma_frames": max(
            float(anchor.get("time_sigma_frames") or ANCHOR_TIME_SIGMA_FRAMES), 1e-3
        ),
        # Where the prior on the free sub-frame time is centred, as an offset from the emitted
        # frame.  Zero is the uniform quantisation window the emission alone supports; a
        # witnessed prior from ``flight_anchors.impact_time_prior`` moves it.
        "time_prior_offset_frames": float(anchor.get("time_prior_offset_frames") or 0.0),
        **_court_witness_geometry(anchor.get("court_witness")),
    }


def _court_witness_geometry(witness: dict | None) -> dict:
    """The circle a court-plane corner witness draws, or an absent one."""
    if not witness:
        return {
            "court_witness_xy": None,
            "court_witness_sigma_m": None,
            "court_witness_radius_m": None,
        }
    xy = np.asarray(witness.get("xy"), dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        return {
            "court_witness_xy": None,
            "court_witness_sigma_m": None,
            "court_witness_radius_m": None,
        }
    sigma = float(witness.get("sigma_m") or 0.0)
    return {
        "court_witness_xy": xy,
        "court_witness_sigma_m": sigma,
        "court_witness_radius_m": max(BOUNCE_WITNESS_FLOOR_M, BOUNCE_WITNESS_SIGMAS * sigma),
    }


def _emission_miss(geometry: dict, position: np.ndarray) -> tuple[float, float]:
    """How far a fitted ball is from the pixel its own emission put it at, in metres and pixels.

    The metric reading is taken in the horizontal plane at the fitted ball's own height, which
    is the court displacement the emission disagrees with; the pixel reading is the residual the
    objective actually minimised.  Both are zero when the fit reprojects onto the emission.
    """
    position = np.asarray(position, float)
    projected = project_one(geometry["projection"], position)
    pixel_error = float(np.linalg.norm(projected - geometry["pixel"]))
    on_ray = ray_at_height(geometry["projection"], geometry["pixel"], float(position[2]))
    if on_ray is None or not np.all(np.isfinite(on_ray)):
        return float("nan"), pixel_error
    return float(np.linalg.norm(position[:2] - np.asarray(on_ray, float)[:2])), pixel_error


def configure_subframe_contacts(enabled: bool) -> None:
    """Configure the sub-frame shared-contact arm before point workers fork.

    A racket contact is emitted on a frame and happens between frames.  Because the ball's
    velocity reverses at the impact, two flights that are both right are separated at the
    emitted frame by roughly the velocity change times the timing error -- half a metre at
    ordinary rally speeds and a third of a frame.  With this arm on, a seam is judged at the
    sub-frame time where the two arcs actually meet, and the shared contact is carried off the
    emitted pixel's ray to that instant instead of being pinned to the ray at it.
    """
    global _SUBFRAME_CONTACTS
    _SUBFRAME_CONTACTS = bool(enabled)


def configure_subframe_contact_parts(
    *,
    seam: bool = True,
    advance: bool = True,
    seam_skips_refit: bool = True,
    seam_witnessed: bool = False,
) -> None:
    """Configure which halves of the sub-frame contact arm run, before point workers fork.

    ``seam`` adopts the sub-frame instant two adjacent arcs actually meet at; ``advance`` carries
    the soft pass's shared contact off the emitted pixel's ray to the fitted instant;
    ``seam_skips_refit`` is whether an adopted seam also stands the soft one-sided reconciliation
    down at that contact.  All three are on inside ``--subframe-contacts`` and none of them does
    anything with that flag off.
    """
    global _SUBFRAME_CONTACT_SEAM, _SUBFRAME_CONTACT_ADVANCE, _SUBFRAME_CONTACT_SEAM_SKIPS_REFIT
    global _SUBFRAME_CONTACT_SEAM_WITNESSED
    _SUBFRAME_CONTACT_SEAM = bool(seam)
    _SUBFRAME_CONTACT_ADVANCE = bool(advance)
    _SUBFRAME_CONTACT_SEAM_SKIPS_REFIT = bool(seam_skips_refit)
    _SUBFRAME_CONTACT_SEAM_WITNESSED = bool(seam_witnessed)


def configure_subframe_bounce_witness(enabled: bool) -> None:
    """Configure the court-plane bounce observation before point workers fork.

    With it on, a bounce anchor that carries a court-plane corner witness gains one residual
    row: zero inside ``max(0.20 m, 2 sigma)`` of the witness, ``(d - r) / r`` outside.  Nothing
    is pinned and no bound moves, so a fit the observation disagrees with is still reachable --
    it just costs.
    """
    global _SUBFRAME_BOUNCE_WITNESS
    _SUBFRAME_BOUNCE_WITNESS = bool(enabled)


def configure_whole_point_joint(enabled: bool) -> None:
    """Configure the simultaneous whole-point solve before point workers fork."""
    global _WHOLE_POINT_JOINT
    _WHOLE_POINT_JOINT = bool(enabled)


def bounce_witness_residual(xy: np.ndarray, geometry: dict) -> float:
    """The owner's shape, as one residual entry: flat inside the circle, growing outside."""
    witness = geometry.get("court_witness_xy")
    radius = geometry.get("court_witness_radius_m")
    if witness is None or not radius:
        return 0.0
    distance = float(np.linalg.norm(np.asarray(xy, float)[:2] - np.asarray(witness, float)))
    if not math.isfinite(distance):
        return 0.0
    return max(0.0, distance - float(radius)) / float(radius)


def configure_subframe_plane_anchor_error(enabled: bool) -> None:
    """Configure what a sub-frame impact anchor reports as its own error.

    ``anchor_max_error_m`` is a gate input, and the shipped fitter's bounce knot always reports
    0.0 because the fit is pinned through it.  With the sub-frame arm on, the default reports
    how far the fitted ball at the emitted frame is from the emission's own ray, which is real
    evidence but is not the quantity the shipped gate was tuned against.  With this arm on the
    anchor instead reports the metric assertion it actually makes -- the impact is on the court
    plane -- so the gate reads the same kind of number for both arms and the real cohort can say
    whether the *fit* changed rather than the gate's input.
    """
    global _SUBFRAME_PLANE_ANCHOR_ERROR
    _SUBFRAME_PLANE_ANCHOR_ERROR = bool(enabled)


def seam_closing_time(
    fits: list[ShotFit], contexts: list[dict], emitted_frame: float
) -> tuple[float, float]:
    """When two adjacent arcs are closest within a frame of the emitted contact, and how close.

    Two flights that are both right meet at the impact, and the impact is sub-frame.  Judging
    their seam at the emitted frame charges the fit for the emitter's quantisation: the ball's
    velocity reverses at a racket, so the two arcs pull apart on either side of the true
    instant at the rate of the velocity change.
    """
    emitted_frame = float(emitted_frame)

    def gap(frame: float) -> float:
        positions = []
        for fit, context in zip(fits, contexts, strict=True):
            try:
                positions.append(
                    np.asarray(
                        fit.state(float(frame), context["fps"], context["surface"])[0], float
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return SEAM_TIME_UNSAMPLED_GAP_M
        if len(positions) != 2 or not all(np.all(np.isfinite(row)) for row in positions):
            return SEAM_TIME_UNSAMPLED_GAP_M
        return float(np.linalg.norm(positions[0] - positions[1]))

    emitted_gap = gap(emitted_frame)
    try:
        search = minimize_scalar(
            gap,
            bounds=(
                emitted_frame - ANCHOR_TIME_BOUND_FRAMES,
                emitted_frame + ANCHOR_TIME_BOUND_FRAMES,
            ),
            method="bounded",
            options={"maxiter": SEAM_TIME_SEARCH_MAX_EVALUATIONS, "xatol": 0.01},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return emitted_frame, emitted_gap
    if not search.success or not np.isfinite(search.fun) or float(search.fun) >= emitted_gap:
        return emitted_frame, emitted_gap
    return float(search.x), float(search.fun)


def _advance_emission_point(
    ray_point: np.ndarray,
    fits: list[ShotFit],
    contexts: list[dict],
    observation_frame: float,
    boundary_frame: float,
) -> np.ndarray:
    """Carry the ball from the frame its pixel was emitted on to the impact instant.

    The emitted pixel observes the ball on an integer frame; the impact is sub-frame, so the
    two are not the same place, and pinning the shared contact to the emitted pixel's ray is
    the contact's version of the bounce knot this package removed.  Which side of the racket
    the emitted frame falls on decides whose velocity carries the ball between them: an impact
    *after* the emitted frame means that frame still shows the incoming ball.  Over less than
    one frame the displacement is the velocity times the elapsed time to within the two
    millimetres gravity adds.
    """
    ray_point = np.asarray(ray_point, float)
    elapsed_frames = float(boundary_frame) - float(observation_frame)
    if abs(elapsed_frames) < 1e-9 or len(fits) != 2:
        return ray_point
    local = 0 if elapsed_frames > 0.0 else 1
    context = contexts[local]
    try:
        velocity = np.asarray(
            fits[local].state(float(observation_frame), context["fps"], context["surface"])[1],
            float,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return ray_point
    if not np.all(np.isfinite(velocity)):
        return ray_point
    return ray_point + velocity * (elapsed_frames / float(context["fps"]))


def configure_objective_probe(callback) -> None:
    """Install (or clear with ``None``) the diagnostic objective probe.

    The callback receives, for every flight objective the fitter builds, the residual function
    itself, its bounds and seeds, the single-seed solver, the state simulator, and the converged
    parameters.  It is called after the fit is complete and cannot change it.
    """
    global _OBJECTIVE_PROBE
    _OBJECTIVE_PROBE = callback


def configure_legacy_height_on_ray(enabled: bool) -> None:
    """Accept the retired compatibility setting without changing the fitter.

    The historical height-on-ray objective also fitted a net crossing.  Net rows are
    post-fit evidence now, so even an old command line must not reactivate that path.
    """
    global _LEGACY_HEIGHT_ON_RAY
    _LEGACY_HEIGHT_ON_RAY = bool(enabled)


def contact_adjacent_weights(
    node_frames: np.ndarray,
    contact_frames: tuple[float, float],
    *,
    enabled: bool | None = None,
) -> np.ndarray:
    """Weight the two fitted observations nearest each flight contact.

    Contact-adjacent detections are often racket-head locks.  Only fitted ray nodes are
    down-weighted; the odd-frame held-out witness remains unchanged.
    """
    weights = np.ones(len(node_frames), dtype=float)
    apply = _DOWNWEIGHT_CONTACT_ADJACENT if enabled is None else enabled
    if not apply:
        return weights
    for contact_frame in contact_frames:
        nearest = np.argsort(np.abs(node_frames - float(contact_frame)), kind="stable")[
            :CONTACT_ADJACENT_OBSERVATIONS
        ]
        weights[nearest] = np.minimum(weights[nearest], CONTACT_ADJACENT_WEIGHT)
    return weights


@dataclass(frozen=True)
class RayNode:
    frame: float
    pixel: np.ndarray
    projection: np.ndarray
    derivative_xyz_per_m: np.ndarray

    def at_height(self, height: float) -> np.ndarray:
        point = ray_at_height(self.projection, self.pixel, float(height))
        if point is None:
            raise FloatingPointError("image ray is parallel to the height plane")
        return point


def _node(camera, frame: float, pixel: np.ndarray) -> RayNode | None:
    projection = np.asarray(camera.p_at(frame), dtype=float)
    zero = ray_at_height(projection, pixel, 0.0)
    one = ray_at_height(projection, pixel, 1.0)
    if zero is None or one is None:
        return None
    return RayNode(float(frame), np.asarray(pixel, float), projection, one - zero)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def _player_xy(players: dict, side: str, frame: float) -> np.ndarray | None:
    rows = players.get(side, {})
    if not rows:
        return None
    nearest_frame = min(rows, key=lambda candidate: abs(float(candidate) - frame))
    value = np.asarray(rows[nearest_frame], float)
    return value[:2] if len(value) >= 2 and np.all(np.isfinite(value[:2])) else None


def striker_witness(players: dict, side: str, frame: float) -> tuple[np.ndarray | None, float]:
    """The striker's court position at a contact, and how stale the row behind it is.

    The tracked player box misses frames -- 1.75% of them on the real cohort, in runs with a p90
    of 8 frames (docs/wk1/bench_striker.md section 2) -- and the nearest row to a contact can
    therefore be a body that has since moved.  Returning the gap lets the caller pay for that
    instead of pretending the witness is fresh.
    """
    rows = players.get(side, {})
    if not rows:
        return None, math.inf
    nearest_frame = min(rows, key=lambda candidate: abs(float(candidate) - frame))
    value = np.asarray(rows[nearest_frame], float)
    if len(value) < 2 or not np.all(np.isfinite(value[:2])):
        return None, math.inf
    return value[:2], abs(float(nearest_frame) - float(frame))


def striker_contact_sigma_m(frame_gap: float, side: str = "far") -> float | None:
    """How hard the striker witness is allowed to argue: its own error, plus how stale it is."""
    if not math.isfinite(frame_gap) or frame_gap > STRIKER_WITNESS_MAX_GAP_FRAMES:
        return None
    root = STRIKER_ROOT_SIGMA_M.get(side, STRIKER_CONTACT_SIGMA_M)
    return root + STRIKER_DRIFT_M_PER_FRAME * max(0.0, frame_gap)


def striker_contact_residuals(
    position: np.ndarray,
    player: np.ndarray | None,
    phase: str,
    side: str,
    sigma_m: float | None = STRIKER_CONTACT_SIGMA_M,
    legacy_reach_m: float = MAX_CONTACT_REACH_M,
) -> np.ndarray:
    """Charge a contact against the striker witness: reach, then front-of-body.

    ``position`` is a fitted contact in court metres and ``player`` the striker's court
    position read from the tracked player box.  The first residual is zero inside
    ``STRIKER_REACH_BAND_M`` (``SERVE_STRIKER_REACH_BAND_M`` on a serve) and grows linearly
    outside it; the second is zero unless the contact is more than ``STRIKER_BEHIND_SLACK_M``
    behind the body toward that striker's own baseline.  Both are blind to depth-free image
    evidence and so decide the contact exactly where the pixels cannot -- along the contact ray.
    """
    if player is None:
        return np.zeros(2)
    player_xy = np.asarray(player, dtype=float)[:2]
    if not np.all(np.isfinite(player_xy)):
        return np.zeros(2)
    distance = float(np.linalg.norm(np.asarray(position, dtype=float)[:2] - player_xy))
    if not _STRIKER_PRIOR:
        # The witness the fitter shipped with: one hinge at a racket's length, and nothing about
        # which side of the body the ball was struck on.
        return np.asarray([max(0.0, distance - legacy_reach_m) / 0.25, 0.0])
    if sigma_m is None or sigma_m <= 0.0:
        return np.zeros(2)
    offset = np.asarray(position, dtype=float)[:2] - player_xy
    low, high = SERVE_STRIKER_REACH_BAND_M if phase == "serve" else STRIKER_REACH_BAND_M
    toward_net = -1.0 if side == "far" else 1.0
    in_front = toward_net * float(offset[1])
    return (
        np.asarray(
            [
                max(0.0, low - distance, distance - high),
                max(0.0, -STRIKER_BEHIND_SLACK_M - in_front),
            ]
        )
        / sigma_m
    )


def _one_sided_tangent(
    track: dict[int, np.ndarray],
    frames: list[int],
    frame: float,
    *,
    after: bool,
) -> np.ndarray | None:
    """Extrapolate the two nearest observations on one side of a contact to the contact frame.

    Each side of a racket impact is its own flight, so its own two frames say where that flight
    reached the racket.  The chord between the two sides says nothing of the kind.
    """
    rows = (
        [value for value in frames if value >= frame][:2]
        if after
        else [value for value in frames if value <= frame][-2:]
    )
    if len(rows) < 2:
        return None
    low, high = float(min(rows)), float(max(rows))
    if high - low > CONTACT_OBSERVATION_MAX_GAP_FRAMES:
        return None
    base = high if after else low
    if abs(frame - base) > CONTACT_OBSERVATION_MAX_GAP_FRAMES:
        return None
    start = np.asarray(track[int(low)], float)
    end = np.asarray(track[int(high)], float)
    slope = (end - start) / (high - low)
    return (end if after else start) + slope * (frame - base)


def contact_observation(
    contact: dict, track: dict[int, np.ndarray]
) -> tuple[np.ndarray, float, float, str] | None:
    """The contact's own image witness, the frame it belongs to, and how wrong it can be.

    In order of preference: the observation the contact row already carries -- the event
    emission's image coordinate, or the contact-frame refiner's pixel and sigma; the contact
    frame's own observation when the contact falls on one; the average of the two one-sided
    tangents; and only then the chord ``interpolate_track`` draws across the impact.
    """
    override = contact.get("image_observation_override")
    if override is not None:
        pixel = np.asarray(override, dtype=float)
        if pixel.shape == (2,) and np.all(np.isfinite(pixel)):
            stated = contact.get("image_observation_sigma_px")
            sigma = (
                float(stated)
                if stated is not None and np.isfinite(float(stated))
                else CONTACT_OBSERVATION_DEFAULT_SIGMA_PX
            )
            return (
                pixel,
                float(contact.get("image_observation_frame", contact["frame"])),
                max(sigma, CONTACT_OBSERVATION_MIN_SIGMA_PX),
                str(contact.get("image_observation_source", "contact_observation_override")),
            )
    frame = float(contact["frame"])
    chord, sources = interpolate_track(track, frame)
    if chord is None:
        return None
    if len(sources) == 1:
        return np.asarray(chord, float), frame, CONTACT_OBSERVATION_MIN_SIGMA_PX, "observed_frame"
    ordered = sorted(track)
    before = _one_sided_tangent(track, ordered, frame, after=False)
    after = _one_sided_tangent(track, ordered, frame, after=True)
    if before is None or after is None:
        return (
            np.asarray(chord, float),
            frame,
            CONTACT_OBSERVATION_CHORD_SIGMA_PX,
            "track_chord",
        )
    return (
        0.5 * (before + after),
        frame,
        max(CONTACT_OBSERVATION_MIN_SIGMA_PX, float(np.linalg.norm(before - after))),
        "two_sided_tangent",
    )


def spin_prior_residuals(velocity: np.ndarray, spin: np.ndarray, surface: str) -> np.ndarray:
    """Charge a fitted spin against the measured incoming topspin of that surface.

    The components are the trajectory's own: topspin about the axis perpendicular to travel,
    then sidespin, then rifle.  Only topspin has a non-zero centre, because only topspin is
    what the Hawk-Eye corpus measured.
    """
    components = _spin_components(velocity, spin) * 100.0 * 60.0 / (2.0 * math.pi)
    center = np.asarray(
        [
            SPIN_PRIOR_TOPSPIN_RPM.get(str(surface), SPIN_PRIOR_DEFAULT_TOPSPIN_RPM),
            0.0,
            0.0,
        ]
    )
    sigma = np.asarray(
        [
            SPIN_PRIOR_TOPSPIN_SIGMA_RPM,
            SPIN_PRIOR_SIDESPIN_SIGMA_RPM,
            SPIN_PRIOR_RIFLE_SIGMA_RPM,
        ]
    )
    return (components - center) / sigma


def _simulate(
    theta: np.ndarray,
    f0: float,
    frames: np.ndarray,
    fps: float,
    surface: str,
    bounce: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    if bounce is not None:
        return simulate_fixed_bounce_with_spin(theta, f0, frames, fps, surface, bounce)
    positions, velocities, spins = sample_states(
        theta[:3],
        theta[3:6],
        spin_vector(theta),
        (np.asarray(frames, float) - f0) / fps,
    )
    return positions, velocities, spins, [], np.zeros(3)


def _anchor_record(anchor: dict) -> dict:
    return {
        **anchor,
        "x": np.asarray(anchor["xyz"], float),
    }


def _net_crossing_bounds(net: dict, f0: float, f1: float) -> tuple[float, float]:
    """Resolve the two observed frames that bracket a bootstrapped net crossing."""
    values = net.get("crossing_frame_bounds") or net.get("source_frames")
    if isinstance(values, (list, tuple)) and len(values) == 2:
        lower, upper = sorted(float(value) for value in values)
    else:
        frame = float(net["frame"])
        lower, upper = math.floor(frame), math.ceil(frame)
        if lower == upper:
            lower, upper = frame - 0.5, frame + 0.5
    lower = max(float(f0), lower)
    upper = min(float(f1), upper)
    if upper - lower < 1e-6:
        raise ValueError("net crossing bracket is empty")
    return lower, upper


def _crossing_count(y_values: np.ndarray) -> int:
    """Count net-plane sign changes without double-counting sampled zeros."""
    delta = np.asarray(y_values, dtype=float) - NET_COURT_Y_M
    if len(delta) < 2 or not np.all(np.isfinite(delta)):
        return 0
    signs = np.sign(delta)
    for index in range(1, len(signs)):
        if signs[index] == 0.0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0.0:
            signs[index] = signs[index + 1]
    if np.all(signs == 0.0):
        return 0
    return int(np.count_nonzero(signs[:-1] * signs[1:] < 0.0))


def _simulate_positions_batch(
    thetas: np.ndarray,
    f0: float,
    frames: np.ndarray,
    fps: float,
    surface: str,
    bounce: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize the ODE work shared by numerical Jacobian columns."""
    thetas = np.asarray(thetas, float)
    query = np.asarray(frames, float)
    initial_spins = np.stack([spin_vector(theta) for theta in thetas])
    if bounce is None:
        positions, _, _ = sample_states_batch(
            thetas[:, :3],
            thetas[:, 3:6],
            initial_spins,
            (query - f0) / fps,
        )
        return positions, np.zeros((len(thetas), 3))

    bounce_frame = float(bounce["frame"])
    before = query < bounce_frame
    pre_times = (query[before] - f0) / fps
    impact_time = (bounce_frame - f0) / fps
    pre_query = np.r_[pre_times, impact_time]
    pre_positions, pre_velocities, pre_spins = sample_states_batch(
        thetas[:, :3],
        thetas[:, 3:6],
        initial_spins,
        pre_query,
    )
    impact_positions = pre_positions[:, -1]
    impact_velocities = pre_velocities[:, -1]
    impact_spins = pre_spins[:, -1]
    outgoing = [
        bounce_velocity(velocity, spin, surface)[:2]
        for velocity, spin in zip(impact_velocities, impact_spins)
    ]
    outgoing_velocities = np.stack([row[0] for row in outgoing])
    outgoing_spins = np.stack([row[1] for row in outgoing])
    result = np.empty((len(thetas), len(query), 3), dtype=float)
    result[:, before] = pre_positions[:, :-1]
    if np.any(~before):
        post_positions, _, _ = sample_states_batch(
            np.repeat(np.asarray(bounce["x"], float)[None, :], len(thetas), axis=0),
            outgoing_velocities,
            outgoing_spins,
            (query[~before] - bounce_frame) / fps,
        )
        result[:, ~before] = post_positions
    return result, impact_positions - np.asarray(bounce["x"], float)


def _fit_legacy_height_on_ray_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera,
    fps: float,
    surface: str,
    max_nfev: int,
    anchors: list[dict],
    *,
    observation_weights: dict[int, float] | None = None,
    fixed_start_anchor: np.ndarray | None = None,
    fixed_end_anchor: np.ndarray | None = None,
    initial_fit: ShotFit | None = None,
    timing_nuisance: str = "none",
    timing_penalty_sigma_frames: float = TIMING_OFFSET_SIGMA_FRAMES,
    field_phase_seconds: float = 0.0,
    net_point_anchor: bool = DEFAULT_NET_POINT_ANCHOR,
) -> ShotFit | None:
    """Fit the compatibility height-on-ray objective."""
    del timing_penalty_sigma_frames, field_phase_seconds, net_point_anchor
    if timing_nuisance != "none":
        raise ValueError("timing nuisance arms require the camera-space fitter")
    start_contact = contacts[index]
    end_contact = contacts[index + 1]
    f0 = float(start_contact["frame"])
    f1 = float(end_contact["frame"])
    observed_frames = np.array(
        sorted(frame for frame in ball if math.ceil(f0) <= frame <= math.floor(f1)),
        dtype=int,
    )
    training_frames = observed_frames[observed_frames % 2 == 0]
    held_out_frames = observed_frames[observed_frames % 2 == 1]
    if len(training_frames) < MIN_TRAINING_FRAMES or len(held_out_frames) < 2:
        return None

    node_rows: list[RayNode] = []
    start_pixel, _ = interpolate_track(ball, f0)
    if start_pixel is None:
        return None
    start_node = _node(camera, f0, start_pixel)
    if start_node is None:
        return None
    node_rows.append(start_node)
    for frame in training_frames:
        if abs(float(frame) - f0) < 1e-6:
            continue
        candidate = _node(camera, float(frame), ball[int(frame)])
        if candidate is not None:
            node_rows.append(candidate)
    if len(node_rows) < MIN_TRAINING_FRAMES:
        return None

    bounce_rows = [_anchor_record(row) for row in anchors if row.get("type") == "bounce"]
    bounce = bounce_rows[0] if len(bounce_rows) == 1 else None
    net_rows = [_anchor_record(row) for row in anchors if row.get("type") == "net_crossing"]
    # The acceptance contract requires the exact vertical-plane witness. There is
    # no value spending optimizer time on a flight the ledger must later abstain.
    if not net_rows or len(bounce_rows) > 1:
        return None
    spin_identifiable = bounce is not None

    p0, _, _ = contact_prior(start_contact, ball, players, camera)
    p1, _, _ = contact_prior(end_contact, ball, players, camera)
    start_height = float(np.clip(p0[2], 0.15, 3.5))
    end_height = float(np.clip(p1[2], BALL_RADIUS_M, 3.5))
    duration = max((f1 - f0) / fps, 0.1)
    height_seed = np.array(
        [
            np.clip(
                start_height + (end_height - start_height) * (row.frame - f0) / max(f1 - f0, 1.0),
                BALL_RADIUS_M,
                4.5,
            )
            for row in node_rows
        ],
        dtype=float,
    )
    if initial_fit is not None:
        try:
            height_seed = np.asarray(
                [
                    np.clip(
                        initial_fit.state(row.frame, fps, surface)[0][2],
                        BALL_RADIUS_M,
                        4.5,
                    )
                    for row in node_rows
                ],
                dtype=float,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
    start_xyz = (
        np.asarray(fixed_start_anchor, float)
        if fixed_start_anchor is not None
        else node_rows[0].at_height(height_seed[0])
    )
    target_xyz = np.asarray(net_rows[0]["x"], float) if net_rows else np.asarray(p1, float)
    target_frame = float(net_rows[0]["frame"]) if net_rows else f1
    target_duration = max((target_frame - f0) / fps, 0.08)
    velocity_seed = (target_xyz - start_xyz) / target_duration
    velocity_seed[2] += 0.5 * 9.81 * target_duration
    if not np.all(np.isfinite(velocity_seed)) or np.linalg.norm(velocity_seed) > 75.0:
        velocity_seed = (np.asarray(p1, float) - start_xyz) / duration
        velocity_seed[2] += 0.5 * 9.81 * duration
    if initial_fit is not None:
        try:
            incumbent_velocity = initial_fit.state(f0, fps, surface)[1]
            if np.all(np.isfinite(incumbent_velocity)):
                velocity_seed = np.asarray(incumbent_velocity, dtype=float)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
    velocity_seed = np.clip(velocity_seed, [-70.0, -70.0, -35.0], [70.0, 70.0, 35.0])
    global_count = 6 if spin_identifiable else 3
    spin_seed = np.zeros(global_count - 3)
    if initial_fit is not None and spin_identifiable and len(initial_fit.theta) >= 9:
        spin_seed = np.asarray(initial_fit.theta[6:9], dtype=float)
    initial = np.r_[height_seed, velocity_seed, spin_seed]
    lower = np.r_[
        np.full(len(node_rows), BALL_RADIUS_M),
        [-70.0, -70.0, -40.0],
        ([-5.0, -5.0, -3.0] if spin_identifiable else []),
    ]
    upper = np.r_[
        np.full(len(node_rows), 6.0),
        [70.0, 70.0, 40.0],
        ([5.0, 5.0, 3.0] if spin_identifiable else []),
    ]
    if start_contact.get("phase") == "serve":
        lower[0] = 0.8

    node_frames = np.asarray([row.frame for row in node_rows], float)
    node_weights = contact_adjacent_weights(node_frames, (f0, f1))
    node_residual_scale = np.repeat(np.sqrt(node_weights), 3)
    metric_rows = [*net_rows, *bounce_rows]
    physics_query = np.unique(np.r_[node_frames, [float(row["frame"]) for row in metric_rows], f1])
    node_query_indices = np.searchsorted(physics_query, node_frames)
    anchor_query_indices = [
        int(np.searchsorted(physics_query, float(row["frame"]))) for row in metric_rows
    ]
    end_query_index = int(np.searchsorted(physics_query, f1))
    shared_contact_residual_start = 3 * len(node_rows) + 3 * len(metric_rows) + 3
    start_player = _player_xy(players, str(start_contact.get("side")), f0)
    end_player = _player_xy(players, str(end_contact.get("side")), f1)

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        heights = parameters[: len(node_rows)]
        rays = np.stack([row.at_height(height) for row, height in zip(node_rows, heights)])
        initial_xyz = (
            np.asarray(fixed_start_anchor, float) if fixed_start_anchor is not None else rays[0]
        )
        velocity = parameters[len(node_rows) : len(node_rows) + 3]
        spin = parameters[len(node_rows) + 3 :] if spin_identifiable else np.zeros(3)
        return np.r_[initial_xyz, velocity, spin], rays

    def assemble_residual(
        theta: np.ndarray,
        rays: np.ndarray,
        all_positions: np.ndarray,
        continuity: np.ndarray,
    ) -> np.ndarray:
        positions = all_positions[node_query_indices]
        pieces = [((positions - rays) / PHYSICS_NODE_SIGMA_M).ravel() * node_residual_scale]
        for anchor, query_index in zip(metric_rows, anchor_query_indices):
            anchor_position = all_positions[query_index]
            sigma = float(
                np.clip(
                    anchor.get("sigma_m", ANCHOR_SIGMA_MAX_M),
                    ANCHOR_SIGMA_MIN_M,
                    ANCHOR_SIGMA_MAX_M,
                )
            )
            pieces.append((anchor_position - anchor["x"]) / sigma)
        pieces.append(continuity / CONTACT_ANCHOR_SIGMA_M)
        end_position = all_positions[end_query_index]
        if fixed_end_anchor is not None:
            pieces.append(
                (end_position - np.asarray(fixed_end_anchor, float)) / SHARED_CONTACT_SIGMA_M
            )
        else:
            pieces.append(np.zeros(3))
        start_distance = (
            max(0.0, float(np.linalg.norm(theta[:2] - start_player)) - MAX_CONTACT_REACH_M)
            if start_player is not None
            else 0.0
        )
        end_distance = (
            max(0.0, float(np.linalg.norm(end_position[:2] - end_player)) - MAX_CONTACT_REACH_M)
            if end_player is not None and not end_contact.get("terminal")
            else 0.0
        )
        speed = float(np.linalg.norm(theta[3:6]))
        pieces.append(
            np.array(
                [
                    start_distance / 0.25,
                    end_distance / 0.25,
                    max(0.0, speed - 72.0) / 2.0,
                    max(0.0, BALL_RADIUS_M - float(np.min(positions[:, 2]))) / 0.02,
                ]
            )
        )
        if spin_identifiable:
            pieces.append(theta[6:9] * 0.03)
        return np.concatenate(pieces)

    def objective_residual(raw: np.ndarray) -> np.ndarray:
        if fixed_end_anchor is None:
            return raw
        # Encode the existing soft-L1 objective directly, except for the shared
        # terminal contact.  The terminal residual must remain quadratic or a
        # metre-scale miss is downweighted as an outlier instead of being shared.
        scale = 2.0
        root = np.sqrt(1.0 + (raw / scale) ** 2)
        transformed = np.sign(raw) * scale * np.sqrt(np.maximum(0.0, 2.0 * (root - 1.0)))
        shared = slice(shared_contact_residual_start, shared_contact_residual_start + 3)
        transformed[shared] = raw[shared]
        return transformed

    def residual(parameters: np.ndarray) -> np.ndarray:
        theta, rays = decode(parameters)
        all_positions, _, _, _, continuity = _simulate(
            theta, f0, physics_query, fps, surface, bounce
        )
        return objective_residual(assemble_residual(theta, rays, all_positions, continuity))

    # All noninitial ray-height derivatives are analytic. Only the start height
    # and the six global state variables need finite differences through the ODE.
    def jacobian(parameters: np.ndarray) -> np.ndarray:
        baseline = residual(parameters)
        raw_baseline = baseline
        if fixed_end_anchor is not None:
            baseline_theta, baseline_rays = decode(parameters)
            baseline_positions, _, _, _, baseline_continuity = _simulate(
                baseline_theta, f0, physics_query, fps, surface, bounce
            )
            raw_baseline = assemble_residual(
                baseline_theta, baseline_rays, baseline_positions, baseline_continuity
            )
        jac = np.zeros((len(baseline), len(parameters)), dtype=float)
        first_fd_height = fixed_start_anchor is None
        for column, row in enumerate(node_rows):
            if column == 0 and first_fd_height:
                continue
            selected = slice(3 * column, 3 * column + 3)
            derivative = (
                -row.derivative_xyz_per_m / PHYSICS_NODE_SIGMA_M * math.sqrt(node_weights[column])
            )
            if fixed_end_anchor is not None:
                raw = raw_baseline[selected]
                transformed = baseline[selected]
                robust_scale = np.ones(3, dtype=float)
                nonzero = np.abs(transformed) > 1e-12
                robust_scale[nonzero] = raw[nonzero] / (
                    transformed[nonzero] * np.sqrt(1.0 + (raw[nonzero] / 2.0) ** 2)
                )
                derivative = derivative * robust_scale
            jac[selected, column] = derivative
        finite_columns = [0] if first_fd_height else []
        finite_columns.extend(range(len(node_rows), len(parameters)))
        shifted_parameters = []
        actual_steps = []
        for column in finite_columns:
            step = 1e-4 * max(1.0, abs(float(parameters[column])))
            shifted = parameters.copy()
            shifted[column] = min(upper[column] - 1e-8, parameters[column] + step)
            actual_step = shifted[column] - parameters[column]
            if actual_step <= 0.0:
                shifted[column] = max(lower[column] + 1e-8, parameters[column] - step)
                actual_step = shifted[column] - parameters[column]
            shifted_parameters.append(shifted)
            actual_steps.append(actual_step)
        if shifted_parameters:
            decoded = [decode(shifted) for shifted in shifted_parameters]
            shifted_thetas = np.stack([row[0] for row in decoded])
            shifted_positions, shifted_continuities = _simulate_positions_batch(
                shifted_thetas,
                f0,
                physics_query,
                fps,
                surface,
                bounce,
            )
            for batch_index, (column, actual_step) in enumerate(zip(finite_columns, actual_steps)):
                shifted_residual = objective_residual(
                    assemble_residual(
                        shifted_thetas[batch_index],
                        decoded[batch_index][1],
                        shifted_positions[batch_index],
                        shifted_continuities[batch_index],
                    )
                )
                jac[:, column] = (shifted_residual - baseline) / actual_step
        return jac

    try:
        result = least_squares(
            residual,
            np.clip(initial, lower + 1e-6, upper - 1e-6),
            jac=jacobian,
            bounds=(lower, upper),
            loss="linear" if fixed_end_anchor is not None else "soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        theta, _ = decode(result.x)
        positions, velocities, spins, bounces, continuity = _simulate(
            theta, f0, observed_frames.astype(float), fps, surface, bounce
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None

    projected = np.array(
        [
            project_one(camera.p_at(float(frame)), xyz)
            for frame, xyz in zip(observed_frames, positions)
        ]
    )
    errors = np.linalg.norm(
        projected - np.stack([ball[int(frame)] for frame in observed_frames]), axis=1
    )
    training_mask = observed_frames % 2 == 0
    held_out_errors = errors[~training_mask]
    training_errors = errors[training_mask]
    anchor_errors = []
    metric_positions = (
        _simulate(
            theta,
            f0,
            np.asarray([float(anchor["frame"]) for anchor in metric_rows]),
            fps,
            surface,
            bounce,
        )[0]
        if metric_rows
        else []
    )
    for anchor, predicted in zip(metric_rows, metric_positions):
        anchor_errors.append(
            {
                "type": anchor["type"],
                "frame": float(anchor["frame"]),
                "error_m": float(np.linalg.norm(predicted - anchor["x"])),
            }
        )
    rms = float(np.sqrt(np.mean(training_errors**2)))
    fit = ShotFit(
        index,
        index + 1,
        theta,
        rms,
        len(observed_frames),
        positions,
        velocities,
        observed_frames,
        bounces,
        1,
    )
    object.__setattr__(fit, "_f0", f0)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", spins)
    object.__setattr__(fit, "_anchor_first", True)
    object.__setattr__(fit, "_anchor_first_free", bounce is None)
    object.__setattr__(fit, "_spin_identifiable", spin_identifiable)
    object.__setattr__(fit, "_held_out_errors_px", held_out_errors)
    object.__setattr__(fit, "_held_out_frames", observed_frames[~training_mask])
    object.__setattr__(fit, "_anchor_errors", anchor_errors)
    object.__setattr__(fit, "_net_anchor_available", bool(net_rows))
    object.__setattr__(fit, "_optimizer_nfev", int(result.nfev))
    object.__setattr__(fit, "_contact_adjacent_weighting", _DOWNWEIGHT_CONTACT_ADJACENT)
    object.__setattr__(fit, "_contact_adjacent_weight", CONTACT_ADJACENT_WEIGHT)
    object.__setattr__(
        fit,
        "_contact_adjacent_frames",
        [float(frame) for frame, weight in zip(node_frames, node_weights) if weight < 1.0],
    )
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(positions[:, 2])),
            "continuity_m": float(np.linalg.norm(continuity)) if bounce is not None else None,
            "incoming_vertical_speed_ms": (float(bounces[0]["v_in"][2]) if bounces else None),
            "speed_ratio": (
                float(
                    np.linalg.norm(bounces[0]["v_out"])
                    / max(np.linalg.norm(bounces[0]["v_in"]), 1e-9)
                )
                if bounces
                else None
            ),
        },
    )
    if bounce is not None:
        object.__setattr__(fit, "_fixed_bounce_anchor", bounce)
    object.__setattr__(
        fit,
        "_anchor_first_context",
        {
            "index": index,
            "contacts": contacts,
            "ball": ball,
            "players": players,
            "camera": camera,
            "fps": fps,
            "surface": surface,
            "max_nfev": max_nfev,
            "anchors": anchors,
            "observation_weights": observation_weights,
        },
    )
    return fit


def _spin_components(velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """Encode a Cartesian spin vector in the trajectory's velocity-aligned basis."""
    velocity = np.asarray(velocity, float)
    spin = np.asarray(spin, float)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return np.zeros(3)
    travel = velocity / speed
    topspin = np.cross(np.array([0.0, 0.0, 1.0]), travel)
    topspin_norm = float(np.linalg.norm(topspin))
    topspin = np.array([1.0, 0.0, 0.0]) if topspin_norm <= 1e-6 else topspin / topspin_norm
    sidespin = np.cross(travel, topspin)
    sidespin_norm = float(np.linalg.norm(sidespin))
    if sidespin_norm > 1e-6:
        sidespin /= sidespin_norm
    return np.asarray([spin @ topspin, spin @ sidespin, spin @ travel]) / 100.0


def _deprecated_camera_space_fit_with_net(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera,
    fps: float,
    surface: str,
    max_nfev: int,
    anchors: list[dict],
    *,
    observation_weights: dict[int, float] | None = None,
    fixed_start_anchor: np.ndarray | None = None,
    fixed_end_anchor: np.ndarray | None = None,
    initial_fit: ShotFit | None = None,
    timing_nuisance: str = "none",
    timing_penalty_sigma_frames: float = TIMING_OFFSET_SIGMA_FRAMES,
    field_phase_seconds: float = 0.0,
    net_point_anchor: bool = DEFAULT_NET_POINT_ANCHOR,
) -> ShotFit | None:
    """Fit one compact physical state by reprojection with an exact bounce knot."""
    if timing_nuisance not in TIMING_NUISANCE_MODES:
        raise ValueError(f"unknown timing nuisance mode: {timing_nuisance}")
    if timing_penalty_sigma_frames <= 0.0:
        raise ValueError("timing_penalty_sigma_frames must be positive")
    start_contact = contacts[index]
    end_contact = contacts[index + 1]
    f0 = float(start_contact["frame"])
    f1 = float(end_contact["frame"])
    observed_frames = np.asarray(
        sorted(
            frame
            for frame in ball
            if (frame > f0 if fixed_start_anchor is not None else frame >= math.ceil(f0))
            and (frame < f1 if fixed_end_anchor is not None else frame <= math.floor(f1))
        ),
        dtype=int,
    )
    training_frames = observed_frames[observed_frames % 2 == 0]
    held_out_frames = observed_frames[observed_frames % 2 == 1]
    if len(training_frames) < MIN_TRAINING_FRAMES or len(held_out_frames) < 2:
        return None

    start_pixel, _ = interpolate_track(ball, f0)
    if start_pixel is None and fixed_start_anchor is None:
        return None
    if fixed_start_anchor is None:
        fit_frames = np.asarray(
            [f0, *(float(frame) for frame in training_frames if abs(float(frame) - f0) >= 1e-6)],
            dtype=float,
        )
        fit_pixels = np.asarray(
            [
                start_pixel,
                *(ball[int(frame)] for frame in training_frames if abs(float(frame) - f0) >= 1e-6),
            ],
            dtype=float,
        )
    else:
        fit_frames = training_frames.astype(float)
        fit_pixels = np.asarray([ball[int(frame)] for frame in training_frames], dtype=float)
    bounce_rows = [_anchor_record(row) for row in anchors if row.get("type") == "bounce"]
    net_rows = [_anchor_record(row) for row in anchors if row.get("type") == "net_crossing"]
    if not net_rows or len(bounce_rows) > 1:
        return None
    bounce = bounce_rows[0] if len(bounce_rows) == 1 else None
    net = net_rows[0]
    try:
        net_frame_lower, net_frame_upper = _net_crossing_bounds(net, f0, f1)
    except (TypeError, ValueError):
        return None
    crossing_direction = int(net.get("crossing_direction_y", 0) or 0)
    if crossing_direction not in {-1, 1}:
        crossing_direction = -1 if start_contact.get("side") == "far" else 1
    spin_identifiable = bounce is not None
    start_player = _player_xy(players, str(start_contact.get("side")), f0)
    end_player = _player_xy(players, str(end_contact.get("side")), f1)
    p0, _, _ = contact_prior(start_contact, ball, players, camera)
    p1, _, _ = contact_prior(end_contact, ball, players, camera)
    start_height = float(np.clip(p0[2], 0.15, 3.5))
    start_node = _node(camera, f0, start_pixel) if start_pixel is not None else None
    if start_node is None and fixed_start_anchor is None:
        return None
    start_seed = (
        np.asarray(fixed_start_anchor, float)
        if fixed_start_anchor is not None
        else start_node.at_height(start_height)  # type: ignore[union-attr]
    )
    fit_weights = np.asarray(
        [
            1.0
            if observation_weights is None
            else float(observation_weights.get(int(round(frame)), 1.0))
            for frame in fit_frames
        ],
        dtype=float,
    )
    fit_weights *= contact_adjacent_weights(fit_frames, (f0, f1))
    fit_weights = np.sqrt(np.clip(fit_weights, 0.01, 1.0))

    if bounce is not None:
        bounce_frame = float(bounce["frame"])
        bounce_xyz = np.asarray(bounce["x"], float)
        net_frame = float(net["frame"])
        net_xyz = np.asarray(net["x"], float)
        if net_frame < bounce_frame - 1e-3:
            interval = max((bounce_frame - net_frame) / fps, 0.04)
            velocity_seed = (bounce_xyz - net_xyz) / interval
            velocity_seed[2] -= 0.5 * 9.81 * interval
        else:
            interval = max((bounce_frame - f0) / fps, 0.08)
            velocity_seed = (bounce_xyz - start_seed) / interval
            velocity_seed[2] -= 0.5 * 9.81 * interval
        spin_seed = np.zeros(3)
        if initial_fit is not None:
            try:
                if initial_fit.bounces:
                    incumbent_velocity = np.asarray(initial_fit.bounces[0]["v_in"], float)
                    incumbent_spin = np.asarray(initial_fit.bounces[0]["w_in"], float)
                else:
                    incumbent_velocity = initial_fit.state(bounce_frame, fps, surface)[1]
                    incumbent_spin = None
                if np.all(np.isfinite(incumbent_velocity)):
                    velocity_seed = incumbent_velocity
                if incumbent_spin is not None and np.all(np.isfinite(incumbent_spin)):
                    spin_seed = incumbent_spin
                incumbent_spins = np.asarray(getattr(initial_fit, "_ws_obs", []), float)
                incumbent_frames = np.asarray(getattr(initial_fit, "obs_frames", []), float)
                if (
                    incumbent_spin is None
                    and len(incumbent_spins)
                    and len(incumbent_spins) == len(incumbent_frames)
                ):
                    spin_seed = incumbent_spins[
                        int(np.argmin(np.abs(incumbent_frames - bounce_frame)))
                    ]
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
        velocity_seed = np.clip(velocity_seed, [-70.0, -70.0, -40.0], [70.0, 70.0, -0.05])
        initial = np.r_[velocity_seed, np.clip(spin_seed, -500.0, 500.0)]
        lower = np.asarray([-75.0, -75.0, -45.0, -500.0, -500.0, -500.0])
        upper = np.asarray([75.0, 75.0, -0.01, 500.0, 500.0, 500.0])
        horizontal = velocity_seed.copy()
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
        topspin_axis = (
            np.array([1.0, 0.0, 0.0])
            if horizontal_norm < 1e-6
            else np.array([-horizontal[1], horizontal[0], 0.0]) / horizontal_norm
        )
        seeds = [
            initial,
            np.r_[velocity_seed * np.array([0.92, 0.92, 1.08]), 180.0 * topspin_axis],
            np.r_[velocity_seed * np.array([1.08, 1.08, 0.92]), -180.0 * topspin_axis],
            np.r_[velocity_seed + np.array([0.0, 0.0, -2.0]), 300.0 * topspin_axis],
        ]
    else:
        target_xyz = np.asarray(net["x"], float)
        interval = max((float(net["frame"]) - f0) / fps, 0.08)
        velocity_seed = (target_xyz - start_seed) / interval
        velocity_seed[2] += 0.5 * 9.81 * interval
        if initial_fit is not None:
            try:
                initial_position, incumbent_velocity = initial_fit.state(f0, fps, surface)
                if np.all(np.isfinite(initial_position)):
                    start_seed = np.asarray(initial_position, float)
                if np.all(np.isfinite(incumbent_velocity)):
                    velocity_seed = np.asarray(incumbent_velocity, float)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
        initial = np.r_[start_seed, velocity_seed]
        lower = np.asarray([-15.0, -10.0, BALL_RADIUS_M, -75.0, -75.0, -45.0])
        upper = np.asarray([26.0, 34.0, 8.0, 75.0, 75.0, 45.0])
        if start_contact.get("phase") == "serve":
            lower[2] = 0.8
        seeds = [
            initial,
            initial + np.asarray([0.0, 0.0, 0.3, 1.0, -1.0, 1.0]),
            initial + np.asarray([0.0, 0.0, -0.3, -1.0, 1.0, -1.0]),
        ]

    dynamics_parameter_count = len(initial)
    net_crossing_parameter_index: int | None = None
    if not net_point_anchor:
        net_crossing_parameter_index = dynamics_parameter_count
        crossing_seed = float(np.clip(float(net["frame"]), net_frame_lower, net_frame_upper))
        seeds = [np.r_[seed, crossing_seed] for seed in seeds]
        lower = np.r_[lower, net_frame_lower]
        upper = np.r_[upper, net_frame_upper]
    base_parameter_count = len(seeds[0])
    timing_parameter_count = (
        len(fit_frames)
        if timing_nuisance == "per_observation"
        else (1 if timing_nuisance == "flight_constant" else 0)
    )
    if timing_parameter_count:
        seeds = [np.r_[seed, np.zeros(timing_parameter_count)] for seed in seeds]
        lower = np.r_[lower, np.full(timing_parameter_count, -TIMING_OFFSET_LIMIT_FRAMES)]
        upper = np.r_[upper, np.full(timing_parameter_count, TIMING_OFFSET_LIMIT_FRAMES)]

    physics_query = np.unique(
        np.r_[fit_frames, *([float(net["frame"])] if net_point_anchor else []), f1]
    )
    fit_indices = np.searchsorted(physics_query, fit_frames)
    net_index = (
        int(np.searchsorted(physics_query, float(net["frame"]))) if net_point_anchor else None
    )
    end_index = int(np.searchsorted(physics_query, f1))

    def simulate(parameters: np.ndarray, frames: np.ndarray):
        if bounce is not None:
            return simulate_hard_bounce_knot(
                parameters[:3],
                parameters[3:6],
                f0,
                frames,
                fps,
                surface,
                bounce,
            )
        positions, velocities, spins = sample_states(
            parameters[:3],
            parameters[3:6],
            np.zeros(3),
            (np.asarray(frames, float) - f0) / fps,
        )
        return positions, velocities, spins, [], (parameters[:3], parameters[3:6], np.zeros(3))

    def time_offsets(parameters: np.ndarray, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=float)
        if timing_nuisance == "per_observation":
            offsets = np.asarray(parameters[base_parameter_count:], dtype=float)
            if len(frames) != len(fit_frames) or not np.allclose(frames, fit_frames):
                raise ValueError("per-observation fitted offsets only apply to fit frames")
            return offsets
        if timing_nuisance == "flight_constant":
            return np.full(len(frames), float(parameters[base_parameter_count]))
        if timing_nuisance == "field_phase":
            parity = np.where(np.rint(frames).astype(int) % 2 == 0, 1.0, -1.0)
            return parity * float(field_phase_seconds) * fps
        return np.zeros(len(frames), dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            offsets = time_offsets(parameters, fit_frames)
            bracket_positions = None
            bracket_velocities = None
            if not net_point_anchor:
                crossing_frame = float(parameters[net_crossing_parameter_index])
                actual_fit_frames = fit_frames + offsets
                combined_query = np.unique(
                    np.r_[
                        actual_fit_frames,
                        net_frame_lower,
                        crossing_frame,
                        net_frame_upper,
                        f1,
                    ]
                )
                combined_positions, combined_velocities, _, _, initial_state = simulate(
                    parameters, combined_query
                )
                fit_positions = combined_positions[
                    np.searchsorted(combined_query, actual_fit_frames)
                ]
                bracket_indices = np.searchsorted(
                    combined_query,
                    [net_frame_lower, crossing_frame, net_frame_upper],
                )
                bracket_positions = combined_positions[bracket_indices]
                bracket_velocities = combined_velocities[bracket_indices]
                net_position = bracket_positions[1]
                end_position = combined_positions[int(np.searchsorted(combined_query, f1))]
                velocities = combined_velocities
            elif timing_nuisance == "none":
                positions, velocities, _, _, initial_state = simulate(parameters, physics_query)
                fit_positions = positions[fit_indices]
                net_position = positions[net_index] if net_index is not None else None
                end_position = positions[end_index]
            else:
                fit_positions, velocities, _, _, initial_state = simulate(
                    parameters, fit_frames + offsets
                )
                net_position = (
                    simulate(parameters, np.asarray([float(net["frame"])]))[0][0]
                    if net_point_anchor
                    else None
                )
                end_position = simulate(parameters, np.asarray([f1]))[0][0]
            projections = np.asarray(
                [
                    project_one(camera.p_at(frame), xyz)
                    for frame, xyz in zip(fit_frames, fit_positions, strict=True)
                ]
            )
            pieces = [((projections - fit_pixels) * fit_weights[:, None]).ravel()]
            if timing_nuisance in {"per_observation", "flight_constant"}:
                pieces.append(
                    np.asarray(parameters[base_parameter_count:], dtype=float)
                    / timing_penalty_sigma_frames
                )
            if net_point_anchor:
                net_sigma = float(
                    np.clip(net.get("sigma_m", ANCHOR_SIGMA_MIN_M), 0.005, ANCHOR_SIGMA_MIN_M)
                )
                pieces.append((net_position - np.asarray(net["x"], float)) / net_sigma)
            else:
                assert bracket_positions is not None
                assert bracket_velocities is not None
                lower_position, net_position, upper_position = bracket_positions
                crossing_velocity = bracket_velocities[1]
                minimum_height = net_tape_height(float(net_position[0])) + BALL_RADIUS_M
                directed_lower = crossing_direction * (float(lower_position[1]) - NET_COURT_Y_M)
                directed_upper = crossing_direction * (float(upper_position[1]) - NET_COURT_Y_M)
                pieces.append(
                    np.asarray(
                        [
                            (float(net_position[1]) - NET_COURT_Y_M) / NET_PLANE_SIGMA_M,
                            max(0.0, minimum_height - float(net_position[2]))
                            / NET_HEIGHT_BARRIER_SIGMA_M,
                            max(0.0, float(net_position[2]) - NET_CROSSING_MAX_HEIGHT_M)
                            / NET_HEIGHT_BARRIER_SIGMA_M,
                            max(0.0, directed_lower) / NET_BRACKET_SIGMA_M,
                            max(0.0, -directed_upper) / NET_BRACKET_SIGMA_M,
                            max(
                                0.0,
                                NET_DIRECTION_SPEED_MIN_MPS
                                - crossing_direction * float(crossing_velocity[1]),
                            )
                            / NET_DIRECTION_SPEED_SIGMA_MPS,
                        ],
                        dtype=float,
                    )
                )
            initial_position = np.asarray(initial_state[0], float)
            if fixed_start_anchor is not None:
                pieces.append(
                    (initial_position - np.asarray(fixed_start_anchor, float))
                    / SHARED_CONTACT_SIGMA_M
                )
            else:
                pieces.append(np.zeros(3))
            if fixed_end_anchor is not None:
                pieces.append(
                    (end_position - np.asarray(fixed_end_anchor, float)) / SHARED_CONTACT_SIGMA_M
                )
            else:
                pieces.append(np.zeros(3))
            start_distance = (
                max(
                    0.0,
                    float(np.linalg.norm(initial_position[:2] - start_player))
                    - MAX_CONTACT_REACH_M,
                )
                if start_player is not None
                else 0.0
            )
            end_distance = (
                max(0.0, float(np.linalg.norm(end_position[:2] - end_player)) - MAX_CONTACT_REACH_M)
                if end_player is not None and not end_contact.get("terminal")
                else 0.0
            )
            minimum_height = float(np.min(fit_positions[:, 2]))
            speed = float(np.linalg.norm(velocities[0]))
            pieces.append(
                np.asarray(
                    [
                        start_distance / 0.25,
                        end_distance / 0.25,
                        max(0.0, speed - 72.0) / 2.0,
                        max(0.0, BALL_RADIUS_M - minimum_height) / 0.002,
                    ]
                )
            )
            if bounce is not None:
                pieces.append(parameters[3:6] / 300.0)
            output = np.concatenate(pieces)
            if np.all(np.isfinite(output)):
                return output
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
            ZeroDivisionError,
        ):
            pass
        timing_residuals = timing_parameter_count
        net_residuals = 3 if net_point_anchor else 6
        return np.full(
            2 * len(fit_frames)
            + timing_residuals
            + 10
            + net_residuals
            + (3 if bounce is not None else 0),
            1e4,
        )

    results = []
    for seed in seeds:
        try:
            result = least_squares(
                residual,
                np.clip(seed, lower + 1e-6, upper - 1e-6),
                bounds=(lower, upper),
                loss="linear",
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(result.fun)):
            results.append(result)
    if not results:
        return None
    result = min(results, key=lambda candidate: float(candidate.cost))
    try:
        if timing_nuisance == "per_observation":
            fitted_offsets = time_offsets(result.x, fit_frames)
            training_offsets = {
                int(round(frame)): float(offset)
                for frame, offset in zip(fit_frames, fitted_offsets, strict=True)
                if abs(frame - round(frame)) < 1e-6
            }
            observation_offsets = []
            penalized_errors = []
            for frame in observed_frames:
                if int(frame) in training_offsets:
                    offset = training_offsets[int(frame)]
                else:
                    pixel = ball[int(frame)]

                    def held_out_objective(candidate: float) -> float:
                        predicted = simulate(result.x, np.asarray([float(frame) + candidate]))[0][0]
                        error = float(
                            np.linalg.norm(
                                project_one(camera.p_at(float(frame)), predicted) - pixel
                            )
                        )
                        return error**2 + (candidate / timing_penalty_sigma_frames) ** 2

                    trial = minimize_scalar(
                        held_out_objective,
                        bounds=(-TIMING_OFFSET_LIMIT_FRAMES, TIMING_OFFSET_LIMIT_FRAMES),
                        method="bounded",
                        options={"xatol": 1e-4},
                    )
                    offset = float(trial.x)
                observation_offsets.append(offset)
            observation_offsets = np.asarray(observation_offsets, dtype=float)
        else:
            observation_offsets = time_offsets(result.x, observed_frames.astype(float))
        positions, velocities, spins, bounces, initial_state = simulate(
            result.x, observed_frames.astype(float) + observation_offsets
        )
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None
    initial_position, initial_velocity, initial_spin = (
        np.asarray(value, float) for value in initial_state
    )
    theta = np.r_[initial_position, initial_velocity]
    if spin_identifiable:
        theta = np.r_[theta, _spin_components(initial_velocity, initial_spin)]
    else:
        theta = np.r_[theta, np.zeros(3)]
    projected = np.asarray(
        [
            project_one(camera.p_at(float(frame)), xyz)
            for frame, xyz in zip(observed_frames, positions, strict=True)
        ]
    )
    errors = np.linalg.norm(
        projected - np.stack([ball[int(frame)] for frame in observed_frames]), axis=1
    )
    timing_penalties = np.abs(observation_offsets) / timing_penalty_sigma_frames
    penalized_errors = np.sqrt(errors**2 + timing_penalties**2)
    training_mask = observed_frames % 2 == 0
    held_out_errors = errors[~training_mask]
    training_errors = errors[training_mask]
    net_constraint: dict[str, Any]
    if net_point_anchor:
        net_position = simulate(result.x, np.asarray([float(net["frame"])]))[0][0]
        net_point_error = float(np.linalg.norm(net_position - np.asarray(net["x"], float)))
        anchor_errors = [
            {
                "type": "net_crossing_point",
                "frame": float(net["frame"]),
                "error_m": net_point_error,
            }
        ]
        net_constraint = {
            "mode": "bootstrap_point_residual",
            "frame": float(net["frame"]),
            "frame_bounds": [net_frame_lower, net_frame_upper],
            "xyz": np.asarray(net_position, float).tolist(),
            "bootstrap_xyz": np.asarray(net["x"], float).tolist(),
            "bootstrap_point_error_m": net_point_error,
            "satisfied": net_point_error <= ANCHOR_ERROR_LIMIT_M,
        }
    else:
        crossing_frame = float(result.x[net_crossing_parameter_index])
        bracket_frames = np.asarray([net_frame_lower, crossing_frame, net_frame_upper], dtype=float)
        bracket_positions, bracket_velocities, *_ = simulate(result.x, bracket_frames)
        lower_position, net_position, upper_position = bracket_positions
        minimum_height = net_tape_height(float(net_position[0])) + BALL_RADIUS_M
        plane_error = abs(float(net_position[1]) - NET_COURT_Y_M)
        lower_height_deficit = max(0.0, minimum_height - float(net_position[2]))
        upper_height_excess = max(0.0, float(net_position[2]) - NET_CROSSING_MAX_HEIGHT_M)
        directed_lower = crossing_direction * (float(lower_position[1]) - NET_COURT_Y_M)
        directed_upper = crossing_direction * (float(upper_position[1]) - NET_COURT_Y_M)
        dense_frames = np.linspace(f0, f1, max(101, int(math.ceil(f1 - f0)) * 4 + 1))
        dense_positions = simulate(result.x, dense_frames)[0]
        crossing_count = _crossing_count(dense_positions[:, 1])
        direction_speed = crossing_direction * float(bracket_velocities[1][1])
        constraint_error = max(
            plane_error,
            lower_height_deficit,
            upper_height_excess,
            max(0.0, directed_lower),
            max(0.0, -directed_upper),
            (1.0 if crossing_count != 1 else 0.0),
            (1.0 if direction_speed <= 0.0 else 0.0),
        )
        constraint_satisfied = (
            plane_error <= NET_CONSTRAINT_TOLERANCE_M
            and lower_height_deficit <= NET_CONSTRAINT_TOLERANCE_M
            and upper_height_excess <= NET_CONSTRAINT_TOLERANCE_M
            and directed_lower <= NET_CONSTRAINT_TOLERANCE_M
            and directed_upper >= -NET_CONSTRAINT_TOLERANCE_M
            and crossing_count == 1
            and direction_speed > 0.0
        )
        anchor_errors = [
            {
                "type": "net_plane_constraint",
                "frame": crossing_frame,
                "error_m": constraint_error,
            }
        ]
        net_constraint = {
            "mode": "plane_crossing_height_barrier",
            "frame": crossing_frame,
            "frame_bounds": [net_frame_lower, net_frame_upper],
            "time_from_bootstrap_frames": crossing_frame - float(net["frame"]),
            "xyz": np.asarray(net_position, float).tolist(),
            "plane_error_m": plane_error,
            "minimum_ball_center_height_m": minimum_height,
            "maximum_ball_center_height_m": NET_CROSSING_MAX_HEIGHT_M,
            "constraint_tolerance_m": NET_CONSTRAINT_TOLERANCE_M,
            "clearance_m": float(net_position[2]) - minimum_height,
            "crossing_count": crossing_count,
            "crossing_direction_y": crossing_direction,
            "directed_speed_mps": direction_speed,
            "bracket_directed_lower_m": directed_lower,
            "bracket_directed_upper_m": directed_upper,
            "satisfied": constraint_satisfied,
            "bootstrap_frame": float(net["frame"]),
            "bootstrap_xyz": np.asarray(net["x"], float).tolist(),
            "bootstrap_xyz_used_as_residual": False,
        }
        if not constraint_satisfied:
            return None
    if bounce is not None:
        anchor_errors.append({"type": "bounce", "frame": float(bounce["frame"]), "error_m": 0.0})
    if fixed_start_anchor is not None:
        anchor_errors.append(
            {
                "type": "contact_ray_start",
                "frame": f0,
                "error_m": float(
                    np.linalg.norm(initial_position - np.asarray(fixed_start_anchor, float))
                ),
            }
        )
    if fixed_end_anchor is not None:
        end_position = simulate(result.x, np.asarray([f1]))[0][0]
        anchor_errors.append(
            {
                "type": "contact_ray_end",
                "frame": f1,
                "error_m": float(
                    np.linalg.norm(end_position - np.asarray(fixed_end_anchor, float))
                ),
            }
        )
    rms = float(np.sqrt(np.mean(training_errors**2)))
    fit = ShotFit(
        index,
        index + 1,
        theta,
        rms,
        len(observed_frames),
        positions,
        velocities,
        observed_frames,
        bounces,
        1,
    )
    object.__setattr__(fit, "_f0", f0)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", spins)
    object.__setattr__(fit, "_anchor_first", True)
    object.__setattr__(fit, "_anchor_first_free", bounce is None)
    object.__setattr__(fit, "_spin_identifiable", spin_identifiable)
    object.__setattr__(fit, "_held_out_errors_px", held_out_errors)
    object.__setattr__(fit, "_held_out_frames", held_out_frames)
    object.__setattr__(fit, "_anchor_errors", anchor_errors)
    object.__setattr__(fit, "_net_anchor_available", True)
    object.__setattr__(fit, "_net_constraint", net_constraint)
    object.__setattr__(fit, "_net_constraint_satisfied", bool(net_constraint["satisfied"]))
    object.__setattr__(fit, "_optimizer_nfev", sum(int(candidate.nfev) for candidate in results))
    object.__setattr__(fit, "_optimizer_starts", len(results))
    object.__setattr__(
        fit,
        "_fit_objective",
        (
            "anchor_first_camera_reprojection_v2_net_point"
            if net_point_anchor
            else "anchor_first_camera_reprojection_v3_net_plane"
        ),
    )
    object.__setattr__(fit, "_timing_nuisance", timing_nuisance)
    object.__setattr__(fit, "_timing_penalty_sigma_frames", timing_penalty_sigma_frames)
    object.__setattr__(fit, "_field_phase_seconds", float(field_phase_seconds))
    object.__setattr__(fit, "_observation_time_offsets_frames", observation_offsets)
    object.__setattr__(fit, "_penalized_observation_errors_px", penalized_errors)
    object.__setattr__(fit, "_contact_adjacent_weighting", _DOWNWEIGHT_CONTACT_ADJACENT)
    object.__setattr__(fit, "_contact_adjacent_weight", CONTACT_ADJACENT_WEIGHT)
    object.__setattr__(
        fit,
        "_contact_adjacent_frames",
        [float(frame) for frame, weight in zip(fit_frames, fit_weights) if weight < 1.0],
    )
    continuity_m = 0.0 if bounce is not None else None
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(positions[:, 2])),
            "continuity_m": continuity_m,
            "incoming_vertical_speed_ms": (float(bounces[0]["v_in"][2]) if bounces else None),
            "speed_ratio": (
                float(
                    np.linalg.norm(bounces[0]["v_out"])
                    / max(np.linalg.norm(bounces[0]["v_in"]), 1e-9)
                )
                if bounces
                else None
            ),
        },
    )
    if bounce is not None:
        object.__setattr__(fit, "_fixed_bounce_anchor", bounce)
        object.__setattr__(
            fit,
            "_bounce_node_split",
            {
                "frame": float(bounce["frame"]),
                "xyz": np.asarray(bounce["x"], float),
                "incoming_velocity": np.asarray(bounces[0]["v_in"], float),
                "outgoing_velocity": np.asarray(bounces[0]["v_out"], float),
                "incoming_spin": np.asarray(bounces[0]["w_in"], float),
                "outgoing_spin": np.asarray(bounces[0]["w_out"], float),
            },
        )
    object.__setattr__(
        fit,
        "_anchor_first_context",
        {
            "index": index,
            "contacts": contacts,
            "ball": ball,
            "players": players,
            "camera": camera,
            "fps": fps,
            "surface": surface,
            "max_nfev": max_nfev,
            "anchors": anchors,
            "observation_weights": observation_weights,
            "net_point_anchor": net_point_anchor,
        },
    )
    return fit


def _camera_ray_direction(camera, frame: float, position: np.ndarray) -> np.ndarray | None:
    """The unit direction from the camera centre to a fitted point: the blind direction."""
    try:
        projection = np.asarray(camera.p_at(float(frame)), dtype=float)
    except (ValueError, KeyError, TypeError):
        return None
    matrix = projection[:, :3]
    offset = projection[:, 3]
    try:
        centre = -np.linalg.solve(matrix, offset)
    except np.linalg.LinAlgError:
        return None
    direction = np.asarray(position, float) - centre
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return direction / norm


def _spread_evidence(
    results: list,
    simulate,
    bounce_knot,
    chosen,
    f0: float,
    f1: float,
    depth_direction: np.ndarray | None,
) -> dict[str, Any]:
    """How far apart the converged multistart solutions are, in metres and along the ray.

    Every seed the flight was started from that converged is sampled at the flight's two
    endpoints and, when the impact is a free parameter, at the impact.  This is label-free
    evidence about whether the objective had one answer or several: a flight whose seeds all
    land in the same place is identified by its own observations, and a flight whose seeds land
    metres apart along the camera ray is not.  Nothing here changes which solution is used.
    """
    evidence: dict[str, Any] = {
        "multistart_solutions": len(results),
        "multistart_start_spread_m": None,
        "multistart_end_spread_m": None,
        "multistart_bounce_spread_m": None,
        "multistart_depth_spread_m": None,
        "multistart_cost_spread": None,
    }
    if len(results) < 2:
        if len(results) == 1:
            evidence.update(
                {
                    "multistart_start_spread_m": 0.0,
                    "multistart_end_spread_m": 0.0,
                    "multistart_depth_spread_m": 0.0,
                    "multistart_cost_spread": 0.0,
                }
            )
        return evidence
    costs = [float(candidate.cost) for candidate in results]
    best = min(costs)
    evidence["multistart_cost_spread"] = float(max(costs) - best)
    try:
        chosen_states = simulate(chosen.x, np.asarray([f0, f1]))[0]
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return evidence
    chosen_bounce = (
        np.asarray(bounce_knot(chosen.x)["x"], float) if bounce_knot(chosen.x) is not None else None
    )
    starts, ends, impacts = [], [], []
    for candidate in results[:MULTISTART_EVIDENCE_MAX_SOLUTIONS]:
        try:
            states = simulate(candidate.x, np.asarray([f0, f1]))[0]
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            continue
        starts.append(np.asarray(states[0], float))
        ends.append(np.asarray(states[1], float))
        knot = bounce_knot(candidate.x)
        if knot is not None:
            impacts.append(np.asarray(knot["x"], float))
    if len(starts) < 2:
        return evidence
    evidence["multistart_start_spread_m"] = float(
        max(np.linalg.norm(row - np.asarray(chosen_states[0], float)) for row in starts)
    )
    evidence["multistart_end_spread_m"] = float(
        max(np.linalg.norm(row - np.asarray(chosen_states[1], float)) for row in ends)
    )
    if impacts and chosen_bounce is not None and len(impacts) >= 2:
        evidence["multistart_bounce_spread_m"] = float(
            max(np.linalg.norm(row - chosen_bounce) for row in impacts)
        )
    if depth_direction is not None:
        evidence["multistart_depth_spread_m"] = float(
            max(
                abs(float(np.dot(row - np.asarray(chosen_states[0], float), depth_direction)))
                for row in starts
            )
        )
    return evidence


def _hessian_evidence(
    result,
    simulate,
    lower: np.ndarray,
    upper: np.ndarray,
    f0: float,
    depth_direction: np.ndarray | None,
) -> dict[str, Any]:
    """The objective's weakest direction at the solution, and what one unit of cost buys on it.

    ``least_squares`` returns the modified Jacobian, so ``J^T J`` is the Gauss-Newton Hessian of
    the robust cost this fitter actually minimises.  Its smallest eigenvalue is how flat the
    objective is in its flattest direction; stepping along that eigenvector far enough to spend
    one unit of cost and re-sampling the flight says how many metres of the flight's own start
    that flatness is worth, and how much of the move is along the camera ray -- which is the
    direction a monocular view cannot see and the one this project's depth errors live in.
    """
    evidence: dict[str, Any] = {
        "hessian_min_curvature": None,
        "hessian_max_curvature": None,
        "hessian_curvature_ratio": None,
        "hessian_weak_start_move_m": None,
        "hessian_weak_depth_fraction": None,
    }
    jacobian = getattr(result, "jac", None)
    if jacobian is None:
        return evidence
    jacobian = np.asarray(jacobian, float)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        return evidence
    try:
        values, vectors = np.linalg.eigh(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return evidence
    minimum = float(values[0])
    maximum = float(values[-1])
    evidence["hessian_min_curvature"] = minimum
    evidence["hessian_max_curvature"] = maximum
    evidence["hessian_curvature_ratio"] = float(minimum / maximum) if maximum > 0.0 else None
    if not minimum > 1e-12:
        return evidence
    # ``least_squares`` reports cost = 0.5 * sum(rho), so a step ``s`` along an eigen-direction
    # of curvature ``lambda`` costs ``0.5 * lambda * s^2``; one unit of cost is ``s = sqrt(2/l)``.
    step = math.sqrt(2.0 / minimum)
    direction = np.asarray(vectors[:, 0], float)
    moved = np.clip(np.asarray(result.x, float) + step * direction, lower, upper)
    try:
        here = np.asarray(simulate(result.x, np.asarray([f0]))[0][0], float)
        there = np.asarray(simulate(moved, np.asarray([f0]))[0][0], float)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return evidence
    offset = there - here
    distance = float(np.linalg.norm(offset))
    if not math.isfinite(distance):
        return evidence
    evidence["hessian_weak_start_move_m"] = distance
    if depth_direction is not None and distance > 1e-9:
        evidence["hessian_weak_depth_fraction"] = float(
            abs(np.dot(offset / distance, depth_direction))
        )
    return evidence


def _camera_space_fit(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera,
    fps: float,
    surface: str,
    max_nfev: int,
    anchors: list[dict],
    *,
    observation_weights: dict[int, float] | None = None,
    fixed_start_anchor: np.ndarray | None = None,
    fixed_end_anchor: np.ndarray | None = None,
    fixed_start_sigma_m: float | None = None,
    fixed_end_sigma_m: float | None = None,
    initial_fit: ShotFit | None = None,
    parameter_seed: np.ndarray | None = None,
    timing_nuisance: str = "none",
    timing_penalty_sigma_frames: float = TIMING_OFFSET_SIGMA_FRAMES,
    field_phase_seconds: float = 0.0,
    net_point_anchor: bool = DEFAULT_NET_POINT_ANCHOR,
) -> ShotFit | None:
    """Fit one flight from image evidence and its physical bounce anchors.

    Ordinary contact-to-contact flights contain at most one bounce.  A terminal
    flight may contain a second bounce: the first bounce is the measured-model
    knot and the second is the on-court termination anchor.  Nothing is
    propagated past that second anchor.

    ``net_point_anchor`` is accepted only so frozen experiment commands remain
    parseable.  It has no effect: no net row is read and no net residual enters the
    objective.
    """
    del net_point_anchor
    if timing_nuisance not in TIMING_NUISANCE_MODES:
        raise ValueError(f"unknown timing nuisance mode: {timing_nuisance}")
    if timing_penalty_sigma_frames <= 0.0:
        raise ValueError("timing_penalty_sigma_frames must be positive")
    start_anchor_sigma = float(fixed_start_sigma_m or SHARED_CONTACT_SIGMA_M)
    end_anchor_sigma = float(fixed_end_sigma_m or SHARED_CONTACT_SIGMA_M)
    if start_anchor_sigma <= 0.0 or end_anchor_sigma <= 0.0:
        raise ValueError("contact anchor sigmas must be positive")

    start_contact = contacts[index]
    end_contact = contacts[index + 1]
    f0 = float(start_contact["frame"])
    f1 = float(end_contact["frame"])
    observed_frames = np.asarray(
        sorted(
            frame
            for frame in ball
            if (frame > f0 if fixed_start_anchor is not None else frame >= math.ceil(f0))
            and (frame < f1 if fixed_end_anchor is not None else frame <= math.floor(f1))
        ),
        dtype=int,
    )
    training_frames = observed_frames[observed_frames % 2 == 0]
    held_out_frames = observed_frames[observed_frames % 2 == 1]
    if len(training_frames) < MIN_TRAINING_FRAMES or len(held_out_frames) < 2:
        return None

    witness = contact_observation(start_contact, ball) if _CONTACT_OBSERVATION_WITNESS else None
    if witness is not None:
        start_pixel, contact_row_frame, contact_row_sigma_px, contact_row_source = witness
        if not any(row.get("type") == "bounce" for row in anchors):
            # A bounce-free flight is integrated forward from its own start, so it has no state
            # before ``f0`` to compare a witness against.  The refined contact observation sits
            # before the contact frame on 206 of the bench's 579 contacts, so this is the common
            # case and not an edge one.
            contact_row_frame = float(np.clip(contact_row_frame, f0, f1))
    else:
        start_pixel, _ = interpolate_track(ball, f0)
        contact_row_frame = f0
        contact_row_sigma_px = (
            contact_row_sigma_px_value(start_contact, ball)
            if _CONTACT_OBSERVATION_SIGMA and start_pixel is not None
            else CONTACT_OBSERVATION_MIN_SIGMA_PX
        )
        contact_row_source = "track_chord"
    if start_pixel is None and fixed_start_anchor is None:
        return None
    if fixed_start_anchor is None:
        keep = [
            float(frame)
            for frame in training_frames
            if abs(float(frame) - contact_row_frame) >= 1e-6
        ]
        fit_frames = np.asarray([contact_row_frame, *keep], dtype=float)
        fit_pixels = np.asarray(
            [start_pixel, *(ball[int(frame)] for frame in keep)],
            dtype=float,
        )
    else:
        fit_frames = training_frames.astype(float)
        fit_pixels = np.asarray([ball[int(frame)] for frame in training_frames], dtype=float)

    bounce_rows = sorted(
        (_anchor_record(row) for row in anchors if row.get("type") == "bounce"),
        key=lambda row: float(row["frame"]),
    )
    two_bounce_terminal = bool(end_contact.get("terminal")) and len(bounce_rows) == 2
    if len(bounce_rows) > 2 or (len(bounce_rows) > 1 and not two_bounce_terminal):
        return None
    bounce = bounce_rows[0] if bounce_rows else None
    terminal_bounce = bounce_rows[1] if two_bounce_terminal else None
    bounce_geometry = (
        subframe_anchor_geometry(bounce, camera)
        if _SUBFRAME_ANCHORS and bounce is not None
        else None
    )
    terminal_geometry = (
        subframe_anchor_geometry(terminal_bounce, camera)
        if _SUBFRAME_ANCHORS and terminal_bounce is not None
        else None
    )
    # A bounce flight is parameterised at its impact: velocity, spin, and -- with the arm on --
    # the impact's own horizontal position and sub-frame time.  The terminal impact adds only a
    # time, because its position is where this flight's own arc reaches the court.
    bounce_parameter_index = 6 if bounce_geometry is not None else None
    terminal_parameter_index = (
        6 + (3 if bounce_geometry is not None else 0) if terminal_geometry is not None else None
    )
    free_spin = bounce is None and (fixed_start_anchor is not None or fixed_end_anchor is not None)
    spin_identifiable = bounce is not None or free_spin
    p0, _, _ = contact_prior(start_contact, ball, players, camera)
    p1, _, _ = contact_prior(end_contact, ball, players, camera)
    start_node = _node(camera, f0, start_pixel) if start_pixel is not None else None
    if start_node is None and fixed_start_anchor is None:
        return None
    start_seed = (
        np.asarray(fixed_start_anchor, float)
        if fixed_start_anchor is not None
        else start_node.at_height(float(np.clip(p0[2], 0.15, 3.5)))  # type: ignore[union-attr]
    )
    end_seed = np.asarray(fixed_end_anchor if fixed_end_anchor is not None else p1, float)
    start_player, start_player_gap = striker_witness(players, str(start_contact.get("side")), f0)
    end_player, end_player_gap = striker_witness(players, str(end_contact.get("side")), f1)
    start_striker_sigma = striker_contact_sigma_m(start_player_gap, str(start_contact.get("side")))
    end_striker_sigma = striker_contact_sigma_m(end_player_gap, str(end_contact.get("side")))
    # A terminal endpoint is where the ball stopped, not where a racket met it, so the striker
    # witness says nothing about it.
    end_striker_player = None if end_contact.get("terminal") else end_player
    start_phase = str(start_contact.get("phase") or "rally")
    end_phase = str(end_contact.get("phase") or "rally")
    start_side = str(start_contact.get("side") or "near")
    end_side = str(end_contact.get("side") or "near")

    fit_weights = np.asarray(
        [
            1.0
            if observation_weights is None
            else float(observation_weights.get(int(round(frame)), 1.0))
            for frame in fit_frames
        ],
        dtype=float,
    )
    fit_weights *= contact_adjacent_weights(fit_frames, (f0, f1))
    if fixed_start_anchor is None and len(fit_weights):
        # The contact row is a witness with an error bar, not an observation.  Weighting it by
        # its own sigma is what stops one interpolated pixel outvoting ten tracked ones.
        fit_weights[0] *= 1.0 / max(contact_row_sigma_px, CONTACT_OBSERVATION_MIN_SIGMA_PX) ** 2
    fit_weights = np.sqrt(np.clip(fit_weights, 0.01, 1.0))

    if bounce is not None:
        bounce_frame = float(bounce["frame"])
        interval = max((bounce_frame - f0) / fps, 0.08)
        geometric_velocity_seed = (np.asarray(bounce["x"], float) - start_seed) / interval
        geometric_velocity_seed[2] -= 0.5 * 9.81 * interval
        spin_seed = np.zeros(3)
        incumbent_velocity_seed = None
        if initial_fit is not None:
            try:
                incumbent_velocity_seed = np.asarray(
                    initial_fit.state(bounce_frame, fps, surface)[1], float
                )
                if initial_fit.bounces:
                    incumbent_velocity_seed = np.asarray(initial_fit.bounces[0]["v_in"], float)
                    spin_seed = np.asarray(initial_fit.bounces[0]["w_in"], float)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
        geometric_velocity_seed = np.clip(
            geometric_velocity_seed, [-70.0, -70.0, -40.0], [70.0, 70.0, -0.05]
        )
        if incumbent_velocity_seed is not None and np.all(np.isfinite(incumbent_velocity_seed)):
            incumbent_velocity_seed = np.clip(
                incumbent_velocity_seed, [-70.0, -70.0, -40.0], [70.0, 70.0, -0.05]
            )
        else:
            incumbent_velocity_seed = None
        # With a new endpoint, start in the basin implied by that endpoint.  The
        # incumbent is still valuable as a second branch, but it must not replace
        # the only seed that is consistent with the proposed shared contact.
        velocity_seeds = [
            geometric_velocity_seed
            if fixed_start_anchor is not None
            else (
                incumbent_velocity_seed
                if incumbent_velocity_seed is not None
                else geometric_velocity_seed
            )
        ]
        if (
            fixed_start_anchor is not None
            and incumbent_velocity_seed is not None
            and not np.allclose(incumbent_velocity_seed, geometric_velocity_seed)
        ):
            velocity_seeds.append(incumbent_velocity_seed)
        lower = np.asarray([-75.0, -75.0, -45.0, -500.0, -500.0, -500.0])
        upper = np.asarray([75.0, 75.0, -0.01, 500.0, 500.0, 500.0])
        seeds = []
        for velocity_seed in velocity_seeds:
            horizontal = velocity_seed.copy()
            horizontal[2] = 0.0
            norm = float(np.linalg.norm(horizontal))
            topspin_axis = (
                np.array([1.0, 0.0, 0.0])
                if norm < 1e-6
                else np.array([-horizontal[1], horizontal[0], 0.0]) / norm
            )
            seeds.extend(
                [
                    np.r_[velocity_seed, np.clip(spin_seed, -500.0, 500.0)],
                    np.r_[
                        velocity_seed * np.array([0.92, 0.92, 1.08]),
                        180.0 * topspin_axis,
                    ],
                    np.r_[
                        velocity_seed * np.array([1.08, 1.08, 0.92]),
                        -180.0 * topspin_axis,
                    ],
                ]
            )
        if bounce_geometry is not None:
            seed_xy = np.asarray(bounce_geometry["seed_xy"], float)
            lower = np.r_[
                lower,
                seed_xy - SUBFRAME_ANCHOR_POSITION_BOUND_M,
                bounce_geometry["time_lower_frames"],
            ]
            upper = np.r_[
                upper,
                seed_xy + SUBFRAME_ANCHOR_POSITION_BOUND_M,
                bounce_geometry["time_upper_frames"],
            ]
            bounce_time_seed = float(
                np.clip(
                    bounce_geometry["time_prior_offset_frames"],
                    bounce_geometry["time_lower_frames"],
                    bounce_geometry["time_upper_frames"],
                )
            )
            seeds = [np.r_[seed, seed_xy, bounce_time_seed] for seed in seeds]
        if terminal_geometry is not None:
            lower = np.r_[lower, terminal_geometry["time_lower_frames"]]
            upper = np.r_[upper, terminal_geometry["time_upper_frames"]]
            # The terminal impact's sub-frame time needs seeding on both sides of the emitted
            # frame.  Unlike an ordinary bounce it has no observations after it -- the flight
            # ends there -- so the only rows that move with it are the court-plane row and its
            # own prior, and from a zero seed the solver converges without leaving it: measured
            # on a terminal flight whose true impact is 0.90 frames after the emitted frame,
            # a zero seed returns 0.000 at cost 13.96 while a half-frame seed reaches 0.763 at
            # cost 5.10.  The impact is uniform inside the emitted frame, so seed the middle and
            # both half-frame edges.
            # Only the primary seed is offered all three times: multiplying the whole seed list
            # tripled the solve count of every two-bounce terminal flight and timed out five of
            # the clean rung's 200 points under load, which buys nothing the extra offsets on one
            # seed do not.
            terminal_time_seed = float(
                np.clip(
                    terminal_geometry["time_prior_offset_frames"],
                    terminal_geometry["time_lower_frames"],
                    terminal_geometry["time_upper_frames"],
                )
            )
            seeds = [np.r_[seed, terminal_time_seed] for seed in seeds] + [
                np.r_[
                    seeds[0],
                    float(
                        np.clip(
                            terminal_time_seed + offset,
                            terminal_geometry["time_lower_frames"],
                            terminal_geometry["time_upper_frames"],
                        )
                    ),
                ]
                for offset in SUBFRAME_TERMINAL_TIME_SEEDS_FRAMES
                if offset != 0.0
            ]
    else:
        interval = max((f1 - f0) / fps, 0.08)
        incumbent_velocity_seed = None
        if initial_fit is not None:
            try:
                incumbent_position, incumbent_velocity = initial_fit.state(f0, fps, surface)
                if fixed_start_anchor is None:
                    start_seed = np.asarray(incumbent_position, float)
                incumbent_velocity_seed = np.asarray(incumbent_velocity, float)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass
        geometric_velocity_seed = (end_seed - start_seed) / interval
        geometric_velocity_seed[2] += 0.5 * 9.81 * interval
        endpoint_refit = free_spin
        velocity_seeds = [
            geometric_velocity_seed
            if endpoint_refit or incumbent_velocity_seed is None
            else incumbent_velocity_seed
        ]
        if (
            endpoint_refit
            and incumbent_velocity_seed is not None
            and np.all(np.isfinite(incumbent_velocity_seed))
            and not np.allclose(incumbent_velocity_seed, geometric_velocity_seed)
        ):
            velocity_seeds.append(incumbent_velocity_seed)
        lower = np.asarray([-15.0, -10.0, BALL_RADIUS_M, -75.0, -75.0, -45.0])
        upper = np.asarray([26.0, 34.0, 8.0, 75.0, 75.0, 45.0])
        # A bounce-free arc has no impact to identify spin from, so its depth is
        # normally read off a zero-spin model.  When a neighbouring flight fixes an
        # endpoint the depth is no longer free, and a zero-spin model is then the
        # reason the anchored refit cannot reproduce the observed pixels: a real
        # topspin lob simply is not a drag-only parabola.  Give spin to that case
        # only, so an unanchored first fit keeps its previous parameters.
        if free_spin:
            lower = np.r_[lower, [-500.0, -500.0, -500.0]]
            upper = np.r_[upper, [500.0, 500.0, 500.0]]
        seeds = []
        for velocity_seed in velocity_seeds:
            initial = np.r_[start_seed, velocity_seed]
            states = [
                initial,
                initial + np.asarray([0.0, 0.0, 0.3, 1.0, -1.0, 1.0]),
                initial + np.asarray([0.0, 0.0, -0.3, -1.0, 1.0, -1.0]),
            ]
            if not free_spin:
                seeds.extend(states)
                continue
            horizontal = np.asarray(velocity_seed, float).copy()
            horizontal[2] = 0.0
            norm = float(np.linalg.norm(horizontal))
            topspin_axis = (
                np.array([1.0, 0.0, 0.0])
                if norm < 1e-6
                else np.array([-horizontal[1], horizontal[0], 0.0]) / norm
            )
            seeds.extend(np.r_[state, np.zeros(3)] for state in states)
            seeds.extend(
                np.r_[initial, sign * NOMINAL_TOPSPIN_RAD_S * topspin_axis] for sign in (1.0, -1.0)
            )

    base_parameter_count = len(seeds[0])
    timing_parameter_count = (
        len(fit_frames)
        if timing_nuisance == "per_observation"
        else (1 if timing_nuisance == "flight_constant" else 0)
    )
    if timing_parameter_count:
        seeds = [np.r_[seed, np.zeros(timing_parameter_count)] for seed in seeds]
        lower = np.r_[lower, np.full(timing_parameter_count, -TIMING_OFFSET_LIMIT_FRAMES)]
        upper = np.r_[upper, np.full(timing_parameter_count, TIMING_OFFSET_LIMIT_FRAMES)]

    if parameter_seed is not None:
        # A bounce-knot parameter vector is independent of the contact-domain endpoints.
        # Preserve the joint optimizer's full velocity/spin/knot solution when rebuilding
        # a constrained flight, instead of keeping only its contact targets and discarding
        # its fitted state. Other parameterizations may change under endpoint refits.
        parameter_seed = np.asarray(parameter_seed, float)
        if (
            not _WHOLE_POINT_JOINT
            or bounce is None
            or timing_nuisance != "none"
            or parameter_seed.shape != lower.shape
            or not np.all(np.isfinite(parameter_seed))
        ):
            raise ValueError("joint parameter seed requires the same finite bounce-knot basis")
        seeds = [parameter_seed.copy(), *seeds]

    def bounce_knot(parameters: np.ndarray) -> dict:
        """The impact this parameter vector puts on the court plane, and when."""
        if bounce_geometry is None:
            return bounce
        offset = bounce_parameter_index
        return {
            **bounce,
            "frame": bounce_geometry["frame"] + float(parameters[offset + 2]),
            "x": np.asarray(
                [
                    float(parameters[offset]),
                    float(parameters[offset + 1]),
                    bounce_geometry["plane_z_m"],
                ],
                dtype=float,
            ),
        }

    def terminal_knot_frame(parameters: np.ndarray) -> float:
        if terminal_geometry is None:
            return float(terminal_bounce["frame"])
        return terminal_geometry["frame"] + float(parameters[terminal_parameter_index])

    def simulate(parameters: np.ndarray, frames: np.ndarray):
        if bounce is not None:
            return simulate_measured_bounce_knot(
                parameters[:3], parameters[3:6], f0, frames, fps, surface, bounce_knot(parameters)
            )
        launch_spin = np.asarray(parameters[6:9], float) if free_spin else np.zeros(3)
        positions, velocities, spins = sample_states(
            parameters[:3],
            parameters[3:6],
            launch_spin,
            (np.asarray(frames, float) - f0) / fps,
        )
        return positions, velocities, spins, [], (parameters[:3], parameters[3:6], launch_spin)

    def time_offsets(parameters: np.ndarray, frames: np.ndarray) -> np.ndarray:
        if timing_nuisance == "per_observation":
            if len(frames) != len(fit_frames) or not np.allclose(frames, fit_frames):
                raise ValueError("per-observation offsets apply only to fitted frames")
            return np.asarray(parameters[base_parameter_count:], float)
        if timing_nuisance == "flight_constant":
            return np.full(len(frames), float(parameters[base_parameter_count]))
        if timing_nuisance == "field_phase":
            parity = np.where(np.rint(frames).astype(int) % 2 == 0, 1.0, -1.0)
            return parity * float(field_phase_seconds) * fps
        return np.zeros(len(frames), float)

    # The default arm emits exactly the residual block the fitter shipped with: two reach hinges
    # and the two physical guards.  The striker arm adds one front-of-body entry per contact.
    striker_piece_count = 6 if _STRIKER_PRIOR else 4
    fixed_piece_count = 6 + striker_piece_count + (3 if bounce is not None or free_spin else 0)
    if terminal_bounce is not None:
        fixed_piece_count += 4
    bounce_witness_row = (
        _SUBFRAME_BOUNCE_WITNESS
        and bounce_geometry is not None
        and bounce_geometry.get("court_witness_xy") is not None
    )
    terminal_witness_row = (
        _SUBFRAME_BOUNCE_WITNESS
        and terminal_geometry is not None
        and terminal_geometry.get("court_witness_xy") is not None
    )
    if bounce_geometry is not None:
        # Two for the emitted pixel read as a ray observation, one for the sub-frame time prior.
        fixed_piece_count += 3
    if terminal_geometry is not None:
        # The exact 3D anchor becomes a plane row, a two-entry pixel row and a time prior.
        fixed_piece_count += 1
    if bounce_witness_row:
        fixed_piece_count += 1
    if terminal_witness_row:
        fixed_piece_count += 1

    def residual(parameters: np.ndarray) -> np.ndarray:
        try:
            offsets = time_offsets(parameters, fit_frames)
            positions, velocities, _, _, initial_state = simulate(parameters, fit_frames + offsets)
            end_values = simulate(parameters, np.asarray([f1]))
            end_position = end_values[0][0]
            end_velocity = end_values[1][0]
            projected = np.asarray(
                [
                    project_one(camera.p_at(frame), xyz)
                    for frame, xyz in zip(fit_frames, positions, strict=True)
                ]
            )
            pieces = [((projected - fit_pixels) * fit_weights[:, None]).ravel()]
            if timing_parameter_count:
                pieces.append(
                    np.asarray(parameters[base_parameter_count:], float)
                    / timing_penalty_sigma_frames
                )
            initial_position = np.asarray(initial_state[0], float)
            pieces.append(
                (initial_position - np.asarray(fixed_start_anchor, float)) / start_anchor_sigma
                if fixed_start_anchor is not None
                else np.zeros(3)
            )
            pieces.append(
                (end_position - np.asarray(fixed_end_anchor, float)) / end_anchor_sigma
                if fixed_end_anchor is not None
                else np.zeros(3)
            )
            start_striker = striker_contact_residuals(
                initial_position, start_player, start_phase, start_side, start_striker_sigma
            )
            end_striker = striker_contact_residuals(
                end_position, end_striker_player, end_phase, end_side, end_striker_sigma
            )
            pieces.append(
                np.r_[start_striker, end_striker]
                if _STRIKER_PRIOR
                else np.asarray([start_striker[0], end_striker[0]])
            )
            pieces.append(
                np.asarray(
                    [
                        max(0.0, float(np.linalg.norm(velocities[0])) - 72.0) / 2.0,
                        max(0.0, BALL_RADIUS_M - float(np.min(positions[:, 2]))) / 0.002,
                    ]
                )
            )
            if bounce is not None:
                pieces.append(
                    spin_prior_residuals(parameters[:3], parameters[3:6], surface)
                    if _PHYSICAL_SPIN_PRIOR
                    else parameters[3:6] / 300.0
                )
            elif free_spin:
                pieces.append(
                    spin_prior_residuals(parameters[3:6], parameters[6:9], surface)
                    if _PHYSICAL_SPIN_PRIOR
                    else parameters[6:9] / 300.0
                )
            if bounce_geometry is not None:
                # The emitted pixel is an image observation of the ball on the emitted frame --
                # a frame on which the ball is still, or already, above the court -- and not a
                # statement about where the impact was.
                observed = simulate(parameters, np.asarray([bounce_geometry["observation_frame"]]))[
                    0
                ][0]
                pieces.append(
                    (
                        project_one(bounce_geometry["projection"], observed)
                        - bounce_geometry["pixel"]
                    )
                    / bounce_geometry["pixel_sigma_px"]
                )
                pieces.append(
                    np.asarray(
                        [
                            (
                                float(parameters[bounce_parameter_index + 2])
                                - bounce_geometry["time_prior_offset_frames"]
                            )
                            / bounce_geometry["time_sigma_frames"]
                        ]
                    )
                )
                if bounce_witness_row:
                    # The owner's shape: nothing at all inside the circle the track's own
                    # court-plane corner draws, a small penalty growing outside it, no pin.
                    pieces.append(
                        np.asarray(
                            [
                                bounce_witness_residual(
                                    parameters[bounce_parameter_index : bounce_parameter_index + 2],
                                    bounce_geometry,
                                )
                            ]
                        )
                    )
            if terminal_bounce is not None:
                terminal_frame = terminal_knot_frame(parameters)
                if abs(terminal_frame - f1) <= 1e-6:
                    terminal_position = end_position
                    terminal_velocity = end_velocity
                else:
                    terminal_values = simulate(parameters, np.asarray([terminal_frame]))
                    terminal_position = terminal_values[0][0]
                    terminal_velocity = terminal_values[1][0]
                terminal_sigma = float(
                    np.clip(
                        terminal_bounce.get("sigma_m", ANCHOR_SIGMA_MAX_M),
                        ANCHOR_SIGMA_MIN_M,
                        ANCHOR_SIGMA_MAX_M,
                    )
                )
                if terminal_geometry is None:
                    pieces.append(
                        (terminal_position - np.asarray(terminal_bounce["x"], float))
                        / terminal_sigma
                    )
                else:
                    # The exact thing about a termination on the court is its height.  Where it
                    # is on the court is what this flight's own arc says, checked against the
                    # emitted pixel on the emitted frame.
                    terminal_observed = simulate(
                        parameters, np.asarray([terminal_geometry["observation_frame"]])
                    )[0][0]
                    pieces.append(
                        np.asarray(
                            [
                                (float(terminal_position[2]) - terminal_geometry["plane_z_m"])
                                / SUBFRAME_ANCHOR_PLANE_SIGMA_M
                            ]
                        )
                    )
                    pieces.append(
                        (
                            project_one(terminal_geometry["projection"], terminal_observed)
                            - terminal_geometry["pixel"]
                        )
                        / terminal_geometry["pixel_sigma_px"]
                    )
                    pieces.append(
                        np.asarray(
                            [
                                (
                                    float(parameters[terminal_parameter_index])
                                    - terminal_geometry["time_prior_offset_frames"]
                                )
                                / terminal_geometry["time_sigma_frames"]
                            ]
                        )
                    )
                    if terminal_witness_row:
                        pieces.append(
                            np.asarray(
                                [bounce_witness_residual(terminal_position[:2], terminal_geometry)]
                            )
                        )
                # A terminal court-plane anchor is an impact, not a later point
                # on an already rising rebound.
                pieces.append(np.asarray([max(0.0, float(terminal_velocity[2]) + 0.01) / 0.10]))
            output = np.concatenate(pieces)
            if np.all(np.isfinite(output)):
                return output
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
            ZeroDivisionError,
        ):
            pass
        return np.full(2 * len(fit_frames) + timing_parameter_count + fixed_piece_count, 1e4)

    def shared_contact_loss(squared_residuals: np.ndarray) -> np.ndarray:
        """Soft-L1 observations, quadratic shared-state constraints.

        SciPy supplies squared residuals after f_scale normalization. A shared
        contact is not a noisy observation that can be discarded as an outlier.
        This candidate is scoped to the already opt-in whole-point joint arm;
        the final connection contract still rejects insufficient numerical closure.
        """
        root = np.sqrt(1.0 + squared_residuals)
        rho = np.vstack((2.0 * (root - 1.0), 1.0 / root, -0.5 / root**3))
        start = 2 * len(fit_frames) + timing_parameter_count
        for offset, anchor in ((0, fixed_start_anchor), (3, fixed_end_anchor)):
            if anchor is not None:
                rows = slice(start + offset, start + offset + 3)
                rho[0, rows] = squared_residuals[rows]
                rho[1, rows] = 1.0
                rho[2, rows] = 0.0
        return rho

    constrained_loss = _WHOLE_POINT_JOINT and (
        fixed_start_anchor is not None or fixed_end_anchor is not None
    )

    def solve_from(seed: np.ndarray):
        """Run this flight's own least-squares call from one seed."""
        return least_squares(
            residual,
            np.clip(np.asarray(seed, float), lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper),
            loss=shared_contact_loss if constrained_loss else "soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=max_nfev,
        )

    results = []
    for seed in seeds:
        try:
            result = solve_from(seed)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(result.fun)):
            results.append(result)
    if not results:
        return None
    result = min(results, key=lambda candidate: float(candidate.cost))
    try:
        if timing_nuisance == "per_observation":
            training_offsets = {
                int(round(frame)): float(offset)
                for frame, offset in zip(
                    fit_frames, time_offsets(result.x, fit_frames), strict=True
                )
                if abs(frame - round(frame)) < 1e-6
            }
            observation_offsets = np.asarray(
                [training_offsets.get(int(frame), 0.0) for frame in observed_frames], float
            )
        else:
            observation_offsets = time_offsets(result.x, observed_frames.astype(float))
        positions, velocities, spins, bounces, initial_state = simulate(
            result.x, observed_frames.astype(float) + observation_offsets
        )
        end_position = simulate(result.x, np.asarray([f1]))[0][0]
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None

    initial_position, initial_velocity, initial_spin = (
        np.asarray(value, float) for value in initial_state
    )
    theta = np.r_[initial_position, initial_velocity]
    theta = np.r_[
        theta,
        _spin_components(initial_velocity, initial_spin) if spin_identifiable else np.zeros(3),
    ]
    projected = np.asarray(
        [
            project_one(camera.p_at(float(frame)), xyz)
            for frame, xyz in zip(observed_frames, positions, strict=True)
        ]
    )
    errors = np.linalg.norm(
        projected - np.stack([ball[int(frame)] for frame in observed_frames]), axis=1
    )
    training_mask = observed_frames % 2 == 0
    held_out_errors = errors[~training_mask]
    training_errors = errors[training_mask]
    anchor_errors = []
    subframe_anchor_report: dict[str, Any] = {}
    if bounce is not None:
        if bounce_geometry is None:
            anchor_errors.append(
                {"type": "bounce", "frame": float(bounce["frame"]), "error_m": 0.0}
            )
        else:
            fitted_knot = bounce_knot(result.x)
            miss_m, pixel_error_px = _emission_miss(
                bounce_geometry,
                simulate(result.x, np.asarray([bounce_geometry["observation_frame"]]))[0][0],
            )
            anchor_errors.append(
                {
                    "type": "bounce",
                    "frame": float(fitted_knot["frame"]),
                    "error_m": 0.0 if _SUBFRAME_PLANE_ANCHOR_ERROR else miss_m,
                    "emission_ray_miss_m": miss_m,
                    "emitted_frame": bounce_geometry["frame"],
                    "time_offset_frames": float(fitted_knot["frame"] - bounce_geometry["frame"]),
                    "emission_pixel_error_px": pixel_error_px,
                    "seed_distance_m": float(
                        np.linalg.norm(
                            np.asarray(fitted_knot["x"], float)[:2] - bounce_geometry["seed_xy"]
                        )
                    ),
                    "role": "court_plane_observation",
                }
            )
            subframe_anchor_report["bounce"] = anchor_errors[-1]
    if terminal_bounce is not None:
        terminal_frame = terminal_knot_frame(result.x)
        terminal_position, terminal_velocity, terminal_spin, _, _ = simulate(
            result.x, np.asarray([terminal_frame])
        )
        if terminal_geometry is None:
            terminal_error = float(
                np.linalg.norm(terminal_position[0] - np.asarray(terminal_bounce["x"], float))
            )
        else:
            terminal_miss_m, terminal_pixel_error_px = _emission_miss(
                terminal_geometry,
                simulate(result.x, np.asarray([terminal_geometry["observation_frame"]]))[0][0],
            )
            terminal_plane_miss = float(terminal_position[0][2] - terminal_geometry["plane_z_m"])
            terminal_error = (
                abs(terminal_plane_miss)
                if _SUBFRAME_PLANE_ANCHOR_ERROR
                else float(math.hypot(terminal_miss_m, terminal_plane_miss))
            )
        if float(terminal_velocity[0, 2]) >= 0.0:
            return None
        try:
            terminal_rebound = court_bounce(
                terminal_velocity[0], terminal_spin[0], surface,
                position=terminal_position[0],
            )
        except (ValueError, ZeroDivisionError):
            return None
        bounces.append(
            {
                "frame": terminal_frame,
                "x": (
                    np.asarray(terminal_bounce["x"], float)
                    if terminal_geometry is None
                    else np.asarray(terminal_position[0], float)
                ),
                "v_in": terminal_velocity[0],
                "v_out": terminal_rebound.velocity,
                "w_in": terminal_spin[0],
                "w_out": terminal_rebound.spin,
                "regime": f"terminal_measured_{terminal_rebound.regime}",
                "termination_anchor": True,
            }
        )
        anchor_errors.append(
            {
                "type": "terminal_bounce",
                "frame": terminal_frame,
                "error_m": terminal_error,
            }
        )
        if terminal_geometry is not None:
            anchor_errors[-1].update(
                {
                    "emitted_frame": terminal_geometry["frame"],
                    "time_offset_frames": float(terminal_frame - terminal_geometry["frame"]),
                    "emission_pixel_error_px": terminal_pixel_error_px,
                    "emission_ray_miss_m": terminal_miss_m,
                    "plane_miss_m": terminal_plane_miss,
                    "seed_distance_m": float(
                        np.linalg.norm(
                            np.asarray(terminal_position[0], float)[:2]
                            - terminal_geometry["seed_xy"]
                        )
                    ),
                    "role": "court_plane_observation",
                }
            )
            subframe_anchor_report["terminal_bounce"] = anchor_errors[-1]
    if fixed_start_anchor is not None:
        anchor_errors.append(
            {
                "type": "contact_start",
                "frame": f0,
                "error_m": float(np.linalg.norm(initial_position - fixed_start_anchor)),
                "sigma_m": start_anchor_sigma,
            }
        )
    if fixed_end_anchor is not None:
        anchor_errors.append(
            {
                "type": "contact_end",
                "frame": f1,
                "error_m": float(np.linalg.norm(end_position - fixed_end_anchor)),
                "sigma_m": end_anchor_sigma,
            }
        )
    depth_direction = _camera_ray_direction(camera, f0, initial_position)
    consistency_evidence: dict[str, Any] = {
        "schema": "flight_consistency_evidence_v1",
        "role": "label_free_evidence_only",
    }
    consistency_evidence.update(
        _spread_evidence(results, simulate, bounce_knot, result, f0, f1, depth_direction)
    )
    consistency_evidence.update(
        _hessian_evidence(result, simulate, lower, upper, f0, depth_direction)
    )
    consistency_evidence["objective_cost"] = float(result.cost)
    # What the impact-time priors were asked for and what the fit did with them.
    if bounce_geometry is not None:
        offset = float(result.x[bounce_parameter_index + 2])
        consistency_evidence["bounce_time_prior_residual_sigmas"] = float(
            (offset - bounce_geometry["time_prior_offset_frames"])
            / bounce_geometry["time_sigma_frames"]
        )
        consistency_evidence["bounce_time_offset_frames"] = offset
        consistency_evidence["bounce_time_prior_witnessed"] = bool((bounce or {}).get("time_prior"))
    if terminal_geometry is not None:
        terminal_offset = float(result.x[terminal_parameter_index])
        consistency_evidence["terminal_time_prior_residual_sigmas"] = float(
            (terminal_offset - terminal_geometry["time_prior_offset_frames"])
            / terminal_geometry["time_sigma_frames"]
        )
        consistency_evidence["terminal_time_offset_frames"] = terminal_offset
    # How far the fitted impacts are from the label-free court-plane corner the track draws.
    for name, geometry, fitted_xy in (
        (
            "bounce",
            bounce_geometry,
            np.asarray(bounce_knot(result.x)["x"], float)[:2]
            if bounce_geometry is not None
            else None,
        ),
        (
            "terminal",
            terminal_geometry,
            np.asarray(bounces[-1]["x"], float)[:2]
            if terminal_geometry is not None and bounces
            else None,
        ),
    ):
        if geometry is None or fitted_xy is None:
            continue
        witness_xy = geometry.get("court_witness_xy")
        if witness_xy is None:
            consistency_evidence[f"{name}_court_witness_error_m"] = None
            continue
        error = float(np.linalg.norm(fitted_xy - np.asarray(witness_xy, float)))
        consistency_evidence[f"{name}_court_witness_error_m"] = error
        consistency_evidence[f"{name}_court_witness_radius_m"] = float(
            geometry["court_witness_radius_m"]
        )
        consistency_evidence[f"{name}_court_witness_sigma_m"] = float(
            geometry["court_witness_sigma_m"]
        )
        consistency_evidence[f"{name}_court_witness_inside_circle"] = bool(
            error <= float(geometry["court_witness_radius_m"])
        )
    # How far each fitted contact is from the striker the tracker put there.
    for name, position, player, side, phase, sigma, terminal in (
        (
            "start",
            initial_position,
            start_player,
            start_side,
            start_phase,
            start_striker_sigma,
            False,
        ),
        (
            "end",
            end_position,
            end_striker_player,
            end_side,
            end_phase,
            end_striker_sigma,
            bool(end_contact.get("terminal")),
        ),
    ):
        if terminal or player is None or sigma is None:
            consistency_evidence[f"{name}_striker_distance_m"] = None
            consistency_evidence[f"{name}_striker_reach_residual"] = None
            continue
        consistency_evidence[f"{name}_striker_distance_m"] = float(
            np.linalg.norm(np.asarray(position, float)[:2] - np.asarray(player, float)[:2])
        )
        consistency_evidence[f"{name}_striker_reach_residual"] = float(
            np.linalg.norm(striker_contact_residuals(position, player, phase, side, sigma))
        )
    rms = float(np.sqrt(np.mean(training_errors**2)))
    fit = ShotFit(
        index,
        index + 1,
        theta,
        rms,
        len(observed_frames),
        positions,
        velocities,
        observed_frames,
        bounces,
        1,
    )
    metadata = {
        "_f0": f0,
        "_weighted_rms_px": rms,
        "_ws_obs": spins,
        "_anchor_first": True,
        "_anchor_first_free": bounce is None,
        "_spin_identifiable": spin_identifiable,
        "_held_out_errors_px": held_out_errors,
        "_held_out_frames": held_out_frames,
        "_observation_errors_px": errors,
        "_anchor_errors": anchor_errors,
        "_net_constraint": {"mode": "post_fit_plausibility_only", "satisfied": True},
        "_net_constraint_satisfied": True,
        "_optimizer_nfev": sum(int(candidate.nfev) for candidate in results),
        "_optimizer_starts": len(results),
        "_fit_objective": "anchor_first_camera_reprojection_v5_terminal_bounce",
        "_shared_contact_loss": "quadratic" if constrained_loss else "soft_l1",
        "_joint_parameter_seed_used": parameter_seed is not None,
        "_timing_nuisance": timing_nuisance,
        "_timing_penalty_sigma_frames": timing_penalty_sigma_frames,
        "_field_phase_seconds": float(field_phase_seconds),
        "_observation_time_offsets_frames": observation_offsets,
        "_penalized_observation_errors_px": errors,
        "_contact_observation": {
            "source": contact_row_source,
            "frame": float(contact_row_frame),
            "sigma_px": float(contact_row_sigma_px),
            "witness": _CONTACT_OBSERVATION_WITNESS,
        },
        "_spin_prior": "measured_surface_topspin" if _PHYSICAL_SPIN_PRIOR else "zero_spin",
        "_subframe_anchors": _SUBFRAME_ANCHORS,
        "_subframe_plane_anchor_error": _SUBFRAME_PLANE_ANCHOR_ERROR,
        "_subframe_anchor_report": subframe_anchor_report,
        "_consistency_evidence": consistency_evidence,
        "_subframe_bounce_witness": _SUBFRAME_BOUNCE_WITNESS,
        "_contact_observation_sigma_arm": _CONTACT_OBSERVATION_SIGMA,
        "_contact_adjacent_weighting": _DOWNWEIGHT_CONTACT_ADJACENT,
        "_contact_adjacent_weight": CONTACT_ADJACENT_WEIGHT,
        "_contact_adjacent_frames": [
            float(frame) for frame, weight in zip(fit_frames, fit_weights) if weight < 1.0
        ],
        "_physical_diagnostics": {
            "minimum_height_m": float(np.min(positions[:, 2])),
            "continuity_m": 0.0 if bounce is not None else None,
            "incoming_vertical_speed_ms": float(bounces[0]["v_in"][2]) if bounces else None,
            "speed_ratio": (
                float(
                    np.linalg.norm(bounces[0]["v_out"])
                    / max(np.linalg.norm(bounces[0]["v_in"]), 1e-9)
                )
                if bounces
                else None
            ),
            "terminal_bounce_continuity_m": (
                next(
                    (
                        float(row["error_m"])
                        for row in anchor_errors
                        if row["type"] == "terminal_bounce"
                    ),
                    None,
                )
            ),
        },
    }
    for name, value in metadata.items():
        object.__setattr__(fit, name, value)
    if bounce is not None:
        object.__setattr__(fit, "_fixed_bounce_anchor", bounce)
        object.__setattr__(
            fit,
            "_bounce_node_split",
            {
                "frame": float(bounces[0]["frame"]),
                "xyz": np.asarray(bounces[0]["x"], float),
                "sampling_model": "measured_bounce_v1",
                "dwell_seconds": DWELL_SECONDS,
                "incoming_velocity": np.asarray(bounces[0]["v_in"], float),
                "outgoing_velocity": np.asarray(bounces[0]["v_out"], float),
                "incoming_spin": np.asarray(bounces[0]["w_in"], float),
                "outgoing_spin": np.asarray(bounces[0]["w_out"], float),
            },
        )
    if terminal_bounce is not None:
        object.__setattr__(fit, "_terminal_bounce_anchor", terminal_bounce)
    object.__setattr__(
        fit,
        "_anchor_first_context",
        {
            "index": index,
            "contacts": contacts,
            "ball": ball,
            "players": players,
            "camera": camera,
            "fps": fps,
            "surface": surface,
            "max_nfev": max_nfev,
            "anchors": anchors,
            "observation_weights": observation_weights,
            "net_point_anchor": None,
        },
    )
    if _WHOLE_POINT_JOINT:
        # Everything the whole-point joint solve needs to re-evaluate this flight's own
        # objective at parameters the flight did not choose, without rebuilding it outside the
        # fitter and hoping the copy stayed faithful.  It is the same payload the diagnostic
        # probe already hands out.
        object.__setattr__(
            fit,
            "_joint_payload",
            {
                "residual": residual,
                "simulate": simulate,
                "lower": np.asarray(lower, float),
                "upper": np.asarray(upper, float),
                "parameters": np.asarray(result.x, float),
                "cost": float(result.cost),
                "f0": f0,
                "f1": f1,
                "fps": fps,
                "surface": surface,
                "parameter_basis": "bounce_knot" if bounce is not None else "contact_state",
                "timing_nuisance": timing_nuisance,
            },
        )
    if _OBJECTIVE_PROBE is not None:
        _OBJECTIVE_PROBE(
            {
                "index": index,
                "f0": f0,
                "f1": f1,
                "fps": fps,
                "surface": surface,
                "camera": camera,
                "contacts": contacts,
                "bounce": bounce,
                "terminal_bounce": terminal_bounce,
                "bounce_geometry": bounce_geometry,
                "terminal_geometry": terminal_geometry,
                "free_spin": free_spin,
                "fixed_start_anchor": fixed_start_anchor,
                "fixed_end_anchor": fixed_end_anchor,
                "residual": residual,
                "simulate": simulate,
                "solve": solve_from,
                "seeds": seeds,
                "lower": lower,
                "upper": upper,
                "parameters": np.asarray(result.x, float),
                "cost": float(result.cost),
                "fit": fit,
            }
        )
    return fit


def fit_anchor_first_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera,
    fps: float,
    surface: str,
    max_nfev: int,
    anchors: list[dict],
    *,
    observation_weights: dict[int, float] | None = None,
    fixed_start_anchor: np.ndarray | None = None,
    fixed_end_anchor: np.ndarray | None = None,
    fixed_start_sigma_m: float | None = None,
    fixed_end_sigma_m: float | None = None,
    initial_fit: ShotFit | None = None,
    parameter_seed: np.ndarray | None = None,
    timing_nuisance: str = "none",
    timing_penalty_sigma_frames: float = TIMING_OFFSET_SIGMA_FRAMES,
    field_phase_seconds: float = 0.0,
    net_point_anchor: bool = DEFAULT_NET_POINT_ANCHOR,
) -> ShotFit | None:
    """Fit one flight, defaulting to direct reprojection and an exact bounce knot."""
    kwargs = {
        "index": index,
        "contacts": contacts,
        "ball": ball,
        "players": players,
        "camera": camera,
        "fps": fps,
        "surface": surface,
        "max_nfev": max_nfev,
        "anchors": anchors,
        "observation_weights": observation_weights,
        "fixed_start_anchor": fixed_start_anchor,
        "fixed_end_anchor": fixed_end_anchor,
        "fixed_start_sigma_m": fixed_start_sigma_m,
        "fixed_end_sigma_m": fixed_end_sigma_m,
        "initial_fit": initial_fit,
        "parameter_seed": parameter_seed,
        "timing_nuisance": timing_nuisance,
        "timing_penalty_sigma_frames": timing_penalty_sigma_frames,
        "field_phase_seconds": field_phase_seconds,
        "net_point_anchor": net_point_anchor,
    }
    return _camera_space_fit(**kwargs)


def _held_out_not_worse(old: dict, new: dict) -> bool:
    old_median = old.get("held_out_reprojection_median_px")
    new_median = new.get("held_out_reprojection_median_px")
    old_p90 = old.get("held_out_reprojection_p90_px")
    new_p90 = new.get("held_out_reprojection_p90_px")
    if None in (old_median, new_median, old_p90, new_p90):
        return False
    median_safe = float(new_median) <= (
        float(old_median) * (1.0 + JOINT_HELD_OUT_RELATIVE_TOLERANCE)
        + JOINT_HELD_OUT_MEDIAN_ABSOLUTE_TOLERANCE_PX
    )
    p90_safe = float(new_p90) <= (
        float(old_p90) * (1.0 + JOINT_HELD_OUT_RELATIVE_TOLERANCE)
        + JOINT_HELD_OUT_P90_ABSOLUTE_TOLERANCE_PX
    )
    # An independently accepted arc may not lose that status in exchange for a
    # prettier junction.  Held arcs can still gain a physically shared contact when
    # their checkerboard witness does not materially worsen.
    acceptance_safe = not old.get("held_out_accepted") or new.get("held_out_accepted")
    return median_safe and p90_safe and bool(acceptance_safe)


def _append_shared_contact(fit: ShotFit, record: dict) -> None:
    contacts = list(getattr(fit, "_shared_contacts", []))
    contacts = [row for row in contacts if row["contact_index"] != record["contact_index"]]
    contacts.append(record)
    object.__setattr__(fit, "_shared_contacts", sorted(contacts, key=lambda row: row["frame"]))
    object.__setattr__(fit, "_junction_refined", True)


def _append_shared_contact_trial(fit: ShotFit, record: dict) -> None:
    trials = list(getattr(fit, "_shared_contact_trials", []))
    trials = [row for row in trials if row["contact_index"] != record["contact_index"]]
    trials.append(record)
    object.__setattr__(fit, "_shared_contact_trials", trials)


def _fixed_contact_observation(
    contact: dict, contexts: list[dict]
) -> tuple[np.ndarray, float, str] | None:
    """Freeze the visible contact pixel while its physical boundary time moves."""
    observation_frame = float(contact.get("image_observation_frame", contact["frame"]))
    override = contact.get("image_observation_override")
    if override is not None:
        pixel = np.asarray(override, dtype=float)
        if pixel.shape == (2,) and np.all(np.isfinite(pixel)):
            return (
                pixel,
                observation_frame,
                str(contact.get("image_observation_source", "event_emission_native_xy")),
            )
    event_frame = float(contact.get("event_frame", contact["frame"]))
    for context in contexts:
        pixel, _ = interpolate_track(context["ball"], event_frame)
        if pixel is not None:
            return np.asarray(pixel, dtype=float), event_frame, "track_at_contact_frame"
    return None


def _copy_contact_metadata(source: ShotFit, destination: ShotFit) -> None:
    for name in ("_shared_contacts", "_shared_contact_trials"):
        object.__setattr__(destination, name, list(getattr(source, name, [])))


def _contact_authority_score(fit: ShotFit) -> float:
    """Rank an independently fitted arc as a metric-depth contact witness.

    Checkerboard reprojection is the only label-free witness available on every
    flight.  A bounce-free arc receives a large penalty because monocular
    reprojection does not identify its depth; this is especially important for an
    out-of-view terminal flight.  Lower is better.
    """
    summary = held_out_summary(fit)
    median = summary.get("held_out_reprojection_median_px")
    p90 = summary.get("held_out_reprojection_p90_px")
    score = (float(median) if median is not None else HELD_OUT_MEDIAN_LIMIT_PX * 4.0) + 0.25 * (
        float(p90) if p90 is not None else HELD_OUT_P90_LIMIT_PX * 4.0
    )
    score += 0.50 * float(getattr(fit, "rms_px", HELD_OUT_MEDIAN_LIMIT_PX * 4.0))
    if not getattr(fit, "bounces", []):
        score += ONE_SIDED_CONTACT_BOUNCE_FREE_PENALTY
    return score


def _one_sided_candidate_safe(
    authority: ShotFit, incumbent: ShotFit, candidate: ShotFit
) -> tuple[bool, str]:
    """Guard a propagated contact without privileging a wrong-depth pixel minimum."""
    incumbent_summary = held_out_summary(incumbent)
    candidate_summary = held_out_summary(candidate)
    if _held_out_not_worse(incumbent_summary, candidate_summary):
        return True, "held_out_not_worse"
    # Reprojection can be exceptionally small on the wrong location along a
    # monocular ray.  When the other flight is the stronger witness, absolute
    # checkerboard acceptance is sufficient; requiring relative equality would
    # preserve exactly that unidentifiable incumbent depth.
    if _contact_authority_score(authority) < _contact_authority_score(
        incumbent
    ) and candidate_summary.get("held_out_accepted"):
        return True, "stronger_authority_absolute_held_out"
    if not candidate_summary.get("held_out_accepted"):
        return False, "candidate_held_out_rejected"
    return False, "authority_not_stronger"


def _trajectory_intersection_seed(
    fits: list[ShotFit], contexts: list[dict], frame: float
) -> np.ndarray | None:
    """Return the closest-point midpoint of the inbound and outbound tangents."""
    if len(fits) != 2:
        return None
    states = []
    for fit, context in zip(fits, contexts, strict=True):
        try:
            position, velocity = fit.state(frame, context["fps"], context["surface"])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None
        position = np.asarray(position, float)
        velocity = np.asarray(velocity, float)
        speed = float(np.linalg.norm(velocity))
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)) or speed < 0.1:
            return None
        states.append((position, velocity / speed))
    incoming_position, incoming_direction = states[0]
    outgoing_position, outgoing_direction = states[1]
    design = np.column_stack((incoming_direction, -outgoing_direction))
    try:
        offsets, *_ = np.linalg.lstsq(design, outgoing_position - incoming_position, rcond=None)
    except np.linalg.LinAlgError:
        return None
    inbound_point = incoming_position + float(offsets[0]) * incoming_direction
    outbound_point = outgoing_position + float(offsets[1]) * outgoing_direction
    midpoint = 0.5 * (inbound_point + outbound_point)
    return midpoint if np.all(np.isfinite(midpoint)) else None


def _contact_radius_metres(
    projection: np.ndarray, pixel: np.ndarray, height: float, sigma_px: float | None
) -> float:
    """Convert the refined contact's image-space 90% radius at a candidate depth."""
    # ``sigma_px`` is a 90% radius, while the optimizer scale is one standard
    # deviation for a Gaussian position prior.
    radius_px = max(float(sigma_px or 10.0), 1.0) / 2.146
    center = ray_at_height(projection, pixel, height)
    if center is None:
        return SOFT_CONTACT_SIGMA_MAX_M
    offsets = []
    for axis in ((radius_px, 0.0), (0.0, radius_px)):
        shifted = ray_at_height(projection, pixel + np.asarray(axis, float), height)
        if shifted is not None and np.all(np.isfinite(shifted)):
            offsets.append(float(np.linalg.norm(np.asarray(shifted, float) - center)))
    if not offsets:
        return SOFT_CONTACT_SIGMA_MAX_M
    return float(np.clip(np.median(offsets), SOFT_CONTACT_SIGMA_MIN_M, SOFT_CONTACT_SIGMA_MAX_M))


def _authority_shared_contact(
    projection: np.ndarray,
    pixel: np.ndarray,
    endpoint: np.ndarray,
    *,
    max_error_px: float = HELD_OUT_P90_LIMIT_PX,
) -> tuple[np.ndarray | None, float]:
    """Use an accepted authority state when it still satisfies the contact pixel gate."""
    projected = project_one(projection, np.asarray(endpoint, float))
    error_px = float(np.linalg.norm(projected - np.asarray(pixel, float)))
    if not np.isfinite(error_px) or error_px > max_error_px:
        return None, error_px
    return np.asarray(endpoint, float).copy(), error_px


def _adjacent_arc_gap(
    fits: dict[int, ShotFit], adjacent: list[int], contexts: list[dict], frame: float
) -> float:
    """How far apart two adjacent arcs are at one frame, or ``inf`` if either cannot be sampled."""
    positions = []
    for index, context in zip(adjacent, contexts, strict=True):
        try:
            positions.append(
                np.asarray(
                    fits[index].state(float(frame), context["fps"], context["surface"])[0], float
                )
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return math.inf
    return float(np.linalg.norm(positions[0] - positions[1]))


def _seam_time_is_witnessed(contact: dict, seam_frame: float, emitted_frame: float) -> bool:
    """Whether an independent witness puts the impact where the two arcs happen to meet.

    With ``_SUBFRAME_CONTACT_SEAM_WITNESSED`` off this is always true, which is what
    ``docs/wk1/point_fit7.md`` measured.  With it on the contact must carry a witnessed
    sub-frame time prior and the seam must land inside that prior's own window, so a seam that
    two wrongly-placed arcs agree on along the camera ray is not mistaken for the impact.
    """
    if not _SUBFRAME_CONTACT_SEAM_WITNESSED:
        return True
    prior = contact.get("time_prior") or {}
    sigma = prior.get("sigma_frames")
    if not sigma:
        return False
    offset = float(prior.get("offset_frames") or 0.0)
    return abs((float(seam_frame) - float(emitted_frame)) - offset) <= (
        SEAM_WITNESS_AGREEMENT_SIGMAS * float(sigma)
    )


SEAM_ADOPTION_SOURCE = "arcs_already_meet_inside_the_emitted_frame"


def _contact_already_reconciled(refined: dict, adjacent: list[int], contact_index: int) -> bool:
    """Whether this contact has already been reconciled, so a later pass must leave it alone.

    A seam-time adoption is a record and not a reconciliation -- nothing was refitted and no arc
    moved -- so with ``_SUBFRAME_CONTACT_SEAM_SKIPS_REFIT`` off it does not stand the soft pass
    down.  That separates "the seam moved" from "the refit did not happen", which pass 7 bundled
    behind one flag.
    """
    for index in adjacent:
        for row in getattr(refined[index], "_shared_contacts", []):
            if row.get("contact_index") != contact_index:
                continue
            if not _SUBFRAME_CONTACT_SEAM_SKIPS_REFIT and row.get("source") == SEAM_ADOPTION_SOURCE:
                continue
            return True
    return False


WHOLE_POINT_JOINT_SOURCE = "whole_point_joint_least_squares"


def whole_point_joint_solve(
    fits: dict[int, ShotFit],
    contacts: list[dict],
    diagnostic_rows: list[dict[str, Any]],
) -> dict[int, ShotFit] | None:
    """Solve every flight and every shared contact of a point in one least-squares call.

    The alternating scheme reconciles one seam at a time: it picks an authority, refits the
    other side onto it, and moves on, so a contact that is only right because of what the next
    contact does is out of its reach.  This solves the whole point at once.  The parameter
    vector is every flight's own parameters concatenated with one free sub-frame time per
    interior contact; the residual is every flight's own objective concatenated with, at each
    interior contact, the two adjacent flights' disagreement at that contact's own fitted
    instant and the contact's impact-time prior.  It is seeded at the per-flight solutions, so
    it is a refinement of them and not a fresh search, and it is bounded by
    ``WHOLE_POINT_MAX_FLIGHTS`` and ``WHOLE_POINT_MAX_NFEV``.

    The returned ``ShotFit`` objects are materialised by one constrained refit of each flight at
    the joint solution's own shared contacts, because that is what builds a flight's bounces,
    its held-out witness and its metadata.  ``None`` means the joint solve did not run, did not
    converge, or its guard refused it, and the caller falls back to the alternating scheme.
    """
    indices = sorted(fits)
    if len(indices) < 2 or len(indices) > WHOLE_POINT_MAX_FLIGHTS:
        diagnostic_rows.append(
            {"status": "joint_not_attempted", "reason": "flight_count", "flights": len(indices)}
        )
        return None
    if indices != list(range(indices[0], indices[0] + len(indices))):
        diagnostic_rows.append({"status": "joint_not_attempted", "reason": "non_contiguous"})
        return None
    payloads = [getattr(fits[index], "_joint_payload", None) for index in indices]
    contexts = [getattr(fits[index], "_anchor_first_context", None) for index in indices]
    if any(row is None for row in payloads) or any(row is None for row in contexts):
        diagnostic_rows.append({"status": "joint_not_attempted", "reason": "missing_payload"})
        return None

    seams = []
    for position, index in enumerate(indices[:-1]):
        contact_index = index + 1
        if contact_index >= len(contacts) or contacts[contact_index].get("terminal"):
            continue
        seams.append((position, contact_index, float(contacts[contact_index]["frame"])))
    if not seams:
        diagnostic_rows.append({"status": "joint_not_attempted", "reason": "no_interior_contact"})
        return None

    offsets: list[int] = []
    cursor = 0
    for payload in payloads:
        offsets.append(cursor)
        cursor += len(payload["parameters"])
    time_offset_base = cursor
    seed = np.concatenate(
        [np.asarray(payload["parameters"], float) for payload in payloads]
        + [
            np.asarray(
                [
                    float(
                        np.clip(
                            (contacts[contact_index].get("time_prior") or {}).get("offset_frames")
                            or 0.0,
                            -CONTACT_BOUNDARY_TIME_LIMIT_FRAMES,
                            CONTACT_BOUNDARY_TIME_LIMIT_FRAMES,
                        )
                    )
                    for _, contact_index, _ in seams
                ],
                dtype=float,
            )
        ]
    )
    lower = np.concatenate(
        [np.asarray(payload["lower"], float) for payload in payloads]
        + [np.full(len(seams), -CONTACT_BOUNDARY_TIME_LIMIT_FRAMES)]
    )
    upper = np.concatenate(
        [np.asarray(payload["upper"], float) for payload in payloads]
        + [np.full(len(seams), CONTACT_BOUNDARY_TIME_LIMIT_FRAMES)]
    )

    def slice_for(position: int) -> slice:
        return slice(offsets[position], offsets[position] + len(payloads[position]["parameters"]))

    def joint_residual(parameters: np.ndarray) -> np.ndarray:
        pieces = [
            payloads[position]["residual"](parameters[slice_for(position)])
            for position in range(len(payloads))
        ]
        for seam_index, (position, contact_index, emitted_frame) in enumerate(seams):
            offset = float(parameters[time_offset_base + seam_index])
            frame = emitted_frame + offset
            try:
                incoming = payloads[position]["simulate"](
                    parameters[slice_for(position)], np.asarray([frame])
                )[0][0]
                outgoing = payloads[position + 1]["simulate"](
                    parameters[slice_for(position + 1)], np.asarray([frame])
                )[0][0]
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                pieces.append(np.full(4, 1e4))
                continue
            gap = np.asarray(incoming, float) - np.asarray(outgoing, float)
            prior = contacts[contact_index].get("time_prior") or {}
            centre = float(prior.get("offset_frames") or 0.0)
            sigma = float(prior.get("sigma_frames") or ANCHOR_TIME_SIGMA_FRAMES)
            pieces.append(
                np.r_[gap / WHOLE_POINT_JUNCTION_SIGMA_M, (offset - centre) / max(sigma, 1e-3)]
            )
        output = np.concatenate(pieces)
        return (
            output
            if np.all(np.isfinite(output))
            else np.nan_to_num(output, nan=1e4, posinf=1e4, neginf=-1e4)
        )

    try:
        solution = least_squares(
            joint_residual,
            np.clip(seed, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=WHOLE_POINT_MAX_NFEV,
        )
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        diagnostic_rows.append({"status": "joint_solve_failed", "reason": "least_squares_raised"})
        return None
    if not np.all(np.isfinite(solution.fun)):
        diagnostic_rows.append({"status": "joint_solve_failed", "reason": "non_finite"})
        return None

    # Materialise each flight at the joint solution's own shared contacts.
    shared: dict[int, tuple[float, np.ndarray]] = {}
    for seam_index, (position, contact_index, emitted_frame) in enumerate(seams):
        frame = emitted_frame + float(solution.x[time_offset_base + seam_index])
        try:
            incoming = payloads[position]["simulate"](
                solution.x[slice_for(position)], np.asarray([frame])
            )[0][0]
            outgoing = payloads[position + 1]["simulate"](
                solution.x[slice_for(position + 1)], np.asarray([frame])
            )[0][0]
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            diagnostic_rows.append({"status": "joint_solve_failed", "reason": "seam_sample_failed"})
            return None
        shared[contact_index] = (
            frame,
            0.5 * (np.asarray(incoming, float) + np.asarray(outgoing, float)),
        )

    constraints: dict[int, dict[str, Any]] = {index: {} for index in indices}
    shifted_by_index: dict[int, list[dict]] = {}
    for index in indices:
        context = contexts[indices.index(index)]
        shifted = [dict(value) for value in context["contacts"]]
        for contact_index, (frame, _) in shared.items():
            if contact_index in (index, index + 1):
                row = dict(shifted[contact_index])
                row["frame"] = frame
                row.setdefault("event_frame", float(contacts[contact_index]["frame"]))
                shifted[contact_index] = row
        shifted_by_index[index] = shifted
        if index in shared:
            constraints[index]["fixed_start_anchor"] = shared[index][1]
            constraints[index]["fixed_start_sigma_m"] = SHARED_CONTACT_SIGMA_M
        if index + 1 in shared:
            constraints[index]["fixed_end_anchor"] = shared[index + 1][1]
            constraints[index]["fixed_end_sigma_m"] = SHARED_CONTACT_SIGMA_M

    candidates: dict[int, ShotFit] = {}
    for position, index in enumerate(indices):
        candidate = fit_anchor_first_shot(
            **{
                **contexts[position],
                **constraints[index],
                "contacts": shifted_by_index[index],
                "initial_fit": fits[index],
                "parameter_seed": (
                    solution.x[slice_for(position)].copy()
                    if payloads[position].get("parameter_basis") == "bounce_knot"
                    and payloads[position].get("timing_nuisance") == "none"
                    else None
                ),
            }
        )
        if candidate is None:
            diagnostic_rows.append(
                {"status": "joint_refit_failed", "flight_index": index, "reason": "fit_failed"}
            )
            return None
        if not _held_out_not_worse(held_out_summary(fits[index]), held_out_summary(candidate)):
            diagnostic_rows.append(
                {
                    "status": "joint_refit_rejected",
                    "flight_index": index,
                    "reason": "held_out_worse",
                    "incumbent_held_out": held_out_summary(fits[index]),
                    "candidate_held_out": held_out_summary(candidate),
                }
            )
            return None
        candidates[index] = candidate

    adopted = []
    for contact_index, (frame, position_xyz) in sorted(shared.items()):
        before = _adjacent_arc_gap(
            fits,
            [contact_index - 1, contact_index],
            [contexts[indices.index(contact_index - 1)], contexts[indices.index(contact_index)]],
            float(contacts[contact_index]["frame"]),
        )
        after = _adjacent_arc_gap(
            candidates,
            [contact_index - 1, contact_index],
            [contexts[indices.index(contact_index - 1)], contexts[indices.index(contact_index)]],
            frame,
        )
        adopted.append((contact_index, frame, position_xyz, before, after))
    if any(not math.isfinite(row[4]) for row in adopted):
        diagnostic_rows.append({"status": "joint_refit_rejected", "reason": "seam_not_sampled"})
        return None
    if sum(row[4] for row in adopted) >= sum(
        row[3] if math.isfinite(row[3]) else 1e6 for row in adopted
    ):
        diagnostic_rows.append(
            {
                "status": "joint_refit_rejected",
                "reason": "junction_not_closed",
                "before_m": [row[3] for row in adopted],
                "after_m": [row[4] for row in adopted],
            }
        )
        return None

    for index, candidate in candidates.items():
        _copy_contact_metadata(fits[index], candidate)
    for contact_index, frame, position_xyz, before, after in adopted:
        record = {
            "contact_index": contact_index,
            "frame": frame,
            "event_frame": float(contacts[contact_index]["frame"]),
            "observation_frame": float(
                contacts[contact_index].get(
                    "image_observation_frame", contacts[contact_index]["frame"]
                )
            ),
            "time_offset_frames": frame - float(contacts[contact_index]["frame"]),
            "xyz": np.asarray(position_xyz, float).tolist(),
            "height_m": float(position_xyz[2]),
            "junction_gap_m": after,
            "pre_refit_endpoint_gap_m": before,
            "observation_source": WHOLE_POINT_JOINT_SOURCE,
            "refitted_flight": None,
            "source": WHOLE_POINT_JOINT_SOURCE,
        }
        for index in (contact_index - 1, contact_index):
            if index in candidates:
                _append_shared_contact(candidates[index], record)
        diagnostic_rows.append(
            {
                "contact_index": contact_index,
                "status": "adopted_whole_point_joint",
                "time_offset_frames": record["time_offset_frames"],
                "junction_gap_m": after,
                "pre_refit_endpoint_gap_m": before,
            }
        )
    for index, candidate in candidates.items():
        object.__setattr__(candidate, "_fit_objective", "anchor_first_whole_point_joint_v1")
        object.__setattr__(
            candidate,
            "_whole_point_joint",
            {
                "flights": len(indices),
                "parameters": int(len(seed)),
                "seams": len(seams),
                "cost": float(solution.cost),
                "nfev": int(solution.nfev),
            },
        )
    diagnostic_rows.append(
        {
            "status": "adopted_whole_point_joint_point",
            "flights": len(indices),
            "seams": len(seams),
            "parameters": int(len(seed)),
            "cost": float(solution.cost),
            "nfev": int(solution.nfev),
        }
    )
    return candidates


def refine_anchor_first_point(
    fits: dict[int, ShotFit], diagnostics: list[dict[str, Any]] | None = None
) -> dict[int, ShotFit]:
    """Reconcile adjacent flights with a radius-weighted soft shared contact.

    The refined native-image contact and its 90% radius define a metric position prior,
    not an exact endpoint.  The inbound/outbound tangent intersection initializes ray
    depth; both flights are adopted atomically only when their independent held-out
    witnesses remain safe.  Failed reconciliation always preserves the incumbent fits.
    """
    refined = dict(fits)
    diagnostic_rows = diagnostics if diagnostics is not None else []
    if not refined:
        return refined
    all_contexts = [
        context
        for fit in refined.values()
        if (context := getattr(fit, "_anchor_first_context", None)) is not None
    ]
    if not all_contexts:
        diagnostic_rows.append({"status": "missing_anchor_first_context"})
        return refined
    contacts = max((context["contacts"] for context in all_contexts), key=len)
    if _WHOLE_POINT_JOINT:
        joint = whole_point_joint_solve(refined, contacts, diagnostic_rows)
        if joint is not None:
            return joint
    constraints: dict[int, dict[str, Any]] = {index: {} for index in fits}

    # A racket contact is emitted on a frame and happens between frames, and the ball's
    # velocity reverses at it, so two flights that are both right are separated at the emitted
    # frame by the velocity change times the timing error -- half a metre at ordinary rally
    # speeds and a third of a frame.  Where the two arcs already meet inside the emission's own
    # frame there is nothing to reconcile: that separation is the emitter's quantisation.
    # Record the instant they meet as the contact's time, so the junction is measured where the
    # impact is and both endpoints are the ball at the impact rather than the ball on a
    # neighbouring frame.  Nothing is refitted here and no arc moves.
    if _SUBFRAME_CONTACTS and _SUBFRAME_CONTACT_SEAM:
        for contact_index, original_contact in enumerate(contacts):
            if original_contact.get("terminal"):
                continue
            adjacent = [index for index in (contact_index - 1, contact_index) if index in refined]
            if len(adjacent) != 2:
                continue
            contexts = [
                getattr(refined[index], "_anchor_first_context", None) for index in adjacent
            ]
            if any(context is None for context in contexts):
                continue
            emitted_frame = float(original_contact["frame"])
            adjacent_fits = [refined[index] for index in adjacent]
            seam_frame, seam_gap = seam_closing_time(adjacent_fits, contexts, emitted_frame)
            if abs(seam_frame - emitted_frame) < 1e-6 or seam_gap > ONE_SIDED_CONTACT_TRIGGER_M:
                continue
            if not _seam_time_is_witnessed(original_contact, seam_frame, emitted_frame):
                diagnostic_rows.append(
                    {
                        "contact_index": contact_index,
                        "status": "seam_time_not_witnessed",
                        "time_offset_frames": seam_frame - emitted_frame,
                        "junction_gap_m": seam_gap,
                    }
                )
                continue
            meeting = []
            for fit, context in zip(adjacent_fits, contexts, strict=True):
                try:
                    meeting.append(
                        np.asarray(
                            fit.state(seam_frame, context["fps"], context["surface"])[0], float
                        )
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    meeting = []
                    break
            if len(meeting) != 2:
                continue
            shared = 0.5 * (meeting[0] + meeting[1])
            emitted_gap = _adjacent_arc_gap(refined, adjacent, contexts, emitted_frame)
            record = {
                "contact_index": contact_index,
                "frame": seam_frame,
                "event_frame": float(original_contact.get("event_frame", emitted_frame)),
                "observation_frame": float(
                    original_contact.get("image_observation_frame", emitted_frame)
                ),
                "time_offset_frames": seam_frame - emitted_frame,
                "xyz": shared.tolist(),
                "height_m": float(shared[2]),
                "junction_gap_m": seam_gap,
                "pre_refit_endpoint_gap_m": emitted_gap,
                "observation_source": "seam_closing_time",
                "refitted_flight": None,
                "source": SEAM_ADOPTION_SOURCE,
            }
            for index in adjacent:
                _append_shared_contact(refined[index], record)
            diagnostic_rows.append(
                {
                    "contact_index": contact_index,
                    "status": "adopted_seam_time",
                    "adjacent_flights": adjacent,
                    "time_offset_frames": record["time_offset_frames"],
                    "junction_gap_m": seam_gap,
                    "pre_refit_endpoint_gap_m": emitted_gap,
                }
            )

    # First propagate the most trustworthy metric endpoint across each bad seam.
    # This is deliberately one-sided: an exact-bounce arc with clean held-out
    # reprojection must not be pulled towards a weak monocular terminal solution.
    # Constraints accumulate across the point, so a middle flight can be refitted
    # with both neighbouring contacts in one least-squares call on the second pass.
    for pass_index in range(ONE_SIDED_CONTACT_MAX_PASSES):
        changed = False
        for contact_index, original_contact in enumerate(contacts):
            if original_contact.get("terminal"):
                continue
            adjacent = [index for index in (contact_index - 1, contact_index) if index in refined]
            if len(adjacent) != 2:
                continue
            if _contact_already_reconciled(refined, adjacent, contact_index):
                continue
            contexts = [
                getattr(refined[index], "_anchor_first_context", None) for index in adjacent
            ]
            if any(context is None for context in contexts):
                continue
            contexts = [context for context in contexts if context is not None]
            contact = dict(original_contact)
            boundary_frame = float(contact["frame"])
            fixed_observation = _fixed_contact_observation(contact, contexts)
            if fixed_observation is None:
                continue
            pixel, observation_frame, observation_source = fixed_observation
            projection = contexts[-1]["camera"].p_at(observation_frame)
            endpoints = []
            for index, context in zip(adjacent, contexts, strict=True):
                try:
                    endpoint = refined[index].state(
                        boundary_frame,
                        context["fps"],
                        context["surface"],
                    )[0]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    endpoint = refined[index].theta[:3]
                endpoints.append(np.asarray(endpoint, float))
            endpoint_gap = float(np.linalg.norm(endpoints[0] - endpoints[1]))
            endpoint_contact_errors_px = [
                float(np.linalg.norm(project_one(projection, endpoint) - pixel))
                for endpoint in endpoints
            ]
            contact_striker, contact_striker_gap = striker_witness(
                contexts[-1].get("players", {}), str(contact.get("side")), boundary_frame
            )
            contact_striker_sigma = striker_contact_sigma_m(
                contact_striker_gap, str(contact.get("side"))
            )
            endpoint_striker_residuals = [
                float(
                    np.linalg.norm(
                        striker_contact_residuals(
                            endpoint,
                            contact_striker,
                            str(contact.get("phase") or "rally"),
                            str(contact.get("side") or "near"),
                            contact_striker_sigma,
                        )
                    )
                )
                for endpoint in endpoints
            ]
            marginal_contact_disagreement = (
                ONE_SIDED_CONTACT_MARGINAL_TRIGGER_M < endpoint_gap <= ONE_SIDED_CONTACT_TRIGGER_M
                and max(endpoint_contact_errors_px) > HELD_OUT_P90_LIMIT_PX
            )
            if endpoint_gap <= ONE_SIDED_CONTACT_TRIGGER_M and not marginal_contact_disagreement:
                continue

            trials = []
            rejected_trials = []
            authority_order = sorted(
                range(2), key=lambda local: _contact_authority_score(refined[adjacent[local]])
            )
            for authority_local in authority_order:
                weak_local = 1 - authority_local
                authority_index = adjacent[authority_local]
                weak_index = adjacent[weak_local]
                authority_endpoint = endpoints[authority_local]
                height = float(np.clip(authority_endpoint[2], 0.15, 3.5))
                authority_held_out_accepted = bool(
                    held_out_summary(refined[authority_index]).get("held_out_accepted")
                )
                # `physics_contact_reprojection` is measured against the automatic track at the
                # contact frame, which docs/wk1/gate_audit.md section 4 measures as the frame the
                # automatic track is worst on: truth-good flights sit at a median of 24 px there.
                # The striker is an independent witness at exactly that place, so when it puts one
                # endpoint inside a racket's reach and the other outside, believe it over the
                # contact pixel and let the authority through the same marginal limit.
                striker_prefers_authority = (
                    _STRIKER_AUTHORITY
                    and contact_striker is not None
                    and contact_striker_sigma is not None
                    and endpoint_striker_residuals[authority_local]
                    <= 0.0
                    < endpoint_striker_residuals[weak_local]
                )
                relaxed_authority_pixel = authority_held_out_accepted and (
                    striker_prefers_authority
                    or (
                        marginal_contact_disagreement
                        and endpoint_contact_errors_px[authority_local] > HELD_OUT_P90_LIMIT_PX
                    )
                )
                authority_pixel_limit = (
                    ONE_SIDED_CONTACT_MARGINAL_PIXEL_LIMIT_PX
                    if relaxed_authority_pixel
                    else HELD_OUT_P90_LIMIT_PX
                )
                shared, authority_contact_error_px = _authority_shared_contact(
                    projection,
                    pixel,
                    authority_endpoint,
                    max_error_px=authority_pixel_limit,
                )
                if shared is None:
                    rejected_trials.append(
                        {
                            "authority_flight": authority_index,
                            "reason": "authority_contact_pixel_rejected",
                            "authority_contact_error_px": authority_contact_error_px,
                        }
                    )
                    continue
                radius_sigma_m = _contact_radius_metres(
                    projection,
                    pixel,
                    height,
                    contact.get("image_observation_sigma_px"),
                )
                # The observed radius chooses the candidate depth, while a narrow
                # endpoint residual makes the two physical states genuinely share
                # that candidate instead of stopping several decimetres apart.
                fit_sigma_m = SHARED_CONTACT_SIGMA_M
                shifted_contacts = [dict(value) for value in contexts[weak_local]["contacts"]]
                shifted_contact = dict(shifted_contacts[contact_index])
                shifted_contact.update(
                    {
                        "image_observation_override": pixel.tolist(),
                        "image_observation_frame": observation_frame,
                        "image_observation_source": observation_source,
                    }
                )
                shifted_contacts[contact_index] = shifted_contact
                endpoint_key = (
                    "fixed_end_anchor" if weak_index == contact_index - 1 else "fixed_start_anchor"
                )
                sigma_key = (
                    "fixed_end_sigma_m"
                    if weak_index == contact_index - 1
                    else "fixed_start_sigma_m"
                )
                candidate = fit_anchor_first_shot(
                    **{
                        **contexts[weak_local],
                        **constraints[weak_index],
                        endpoint_key: shared,
                        sigma_key: fit_sigma_m,
                        "contacts": shifted_contacts,
                        "initial_fit": refined[weak_index],
                    }
                )
                if candidate is None:
                    rejected_trials.append(
                        {"authority_flight": authority_index, "reason": "fit_failed"}
                    )
                    continue
                candidate_safe, guard = _one_sided_candidate_safe(
                    refined[authority_index], refined[weak_index], candidate
                )
                if not candidate_safe:
                    rejected_trials.append(
                        {
                            "authority_flight": authority_index,
                            "reason": guard,
                            "candidate_held_out": held_out_summary(candidate),
                        }
                    )
                    continue
                try:
                    candidate_endpoint = candidate.state(
                        boundary_frame,
                        contexts[weak_local]["fps"],
                        contexts[weak_local]["surface"],
                    )[0]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    rejected_trials.append(
                        {"authority_flight": authority_index, "reason": "state_failed"}
                    )
                    continue
                junction_gap = float(np.linalg.norm(authority_endpoint - candidate_endpoint))
                if junction_gap >= endpoint_gap or junction_gap > ONE_SIDED_CONTACT_TARGET_M:
                    rejected_trials.append(
                        {
                            "authority_flight": authority_index,
                            "reason": "junction_not_closed",
                            "junction_gap_m": junction_gap,
                        }
                    )
                    continue
                opposite_contact_index = (
                    contact_index - 1 if weak_index == contact_index - 1 else contact_index + 1
                )
                opposite_endpoint_move_m = 0.0
                if marginal_contact_disagreement and 0 <= opposite_contact_index < len(contacts):
                    opposite_frame = float(contacts[opposite_contact_index]["frame"])
                    try:
                        incumbent_opposite = refined[weak_index].state(
                            opposite_frame,
                            contexts[weak_local]["fps"],
                            contexts[weak_local]["surface"],
                        )[0]
                        candidate_opposite = candidate.state(
                            opposite_frame,
                            contexts[weak_local]["fps"],
                            contexts[weak_local]["surface"],
                        )[0]
                        opposite_endpoint_move_m = float(
                            np.linalg.norm(candidate_opposite - incumbent_opposite)
                        )
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        opposite_endpoint_move_m = math.inf
                    if opposite_endpoint_move_m > ONE_SIDED_CONTACT_OPPOSITE_ENDPOINT_MAX_MOVE_M:
                        rejected_trials.append(
                            {
                                "authority_flight": authority_index,
                                "reason": "opposite_endpoint_moved",
                                "opposite_endpoint_move_m": opposite_endpoint_move_m,
                            }
                        )
                        continue
                score = (
                    _contact_authority_score(refined[authority_index])
                    + _contact_authority_score(candidate)
                    + 10.0 * junction_gap
                )
                trials.append(
                    (
                        score,
                        authority_local,
                        weak_local,
                        candidate,
                        shared,
                        junction_gap,
                        radius_sigma_m,
                        fit_sigma_m,
                        guard,
                        opposite_endpoint_move_m,
                    )
                )
                # Authority candidates are ordered by label-free reliability.  The
                # first safe closure is preferable to doubling every point's solve
                # count in search of a cosmetically smaller already-passing seam.
                break
            if not trials:
                diagnostic_rows.append(
                    {
                        "contact_index": contact_index,
                        "status": "no_safe_one_sided_refit",
                        "adjacent_flights": adjacent,
                        "pre_refit_endpoint_gap_m": endpoint_gap,
                        "trials": rejected_trials,
                        "pass": pass_index,
                    }
                )
                continue
            (
                objective,
                authority_local,
                weak_local,
                candidate,
                shared,
                junction_gap,
                radius_sigma_m,
                fit_sigma_m,
                guard,
                opposite_endpoint_move_m,
            ) = min(trials, key=lambda row: row[0])
            authority_index = adjacent[authority_local]
            weak_index = adjacent[weak_local]
            previous = refined[weak_index]
            _copy_contact_metadata(previous, candidate)
            record = {
                "contact_index": contact_index,
                "frame": boundary_frame,
                "event_frame": float(contact.get("event_frame", boundary_frame)),
                "observation_frame": observation_frame,
                "time_offset_frames": 0.0,
                "xyz": shared.tolist(),
                "height_m": float(shared[2]),
                "observed_uv": pixel.tolist(),
                "observation_source": observation_source,
                "observation_sigma_px": contact.get("image_observation_sigma_px"),
                "position_prior_sigma_m": radius_sigma_m,
                "fit_endpoint_sigma_m": fit_sigma_m,
                "junction_gap_m": junction_gap,
                "pre_refit_endpoint_gap_m": endpoint_gap,
                "objective": objective,
                "authority_flight": authority_index,
                "authority_contact_error_px": authority_contact_error_px,
                "refitted_flight": weak_index,
                "selection_guard": guard,
                "marginal_contact_disagreement": marginal_contact_disagreement,
                "opposite_endpoint_move_m": opposite_endpoint_move_m,
                "pass": pass_index,
            }
            _append_shared_contact(candidate, record)
            _append_shared_contact(refined[authority_index], record)
            object.__setattr__(candidate, "_fit_objective", "anchor_first_one_sided_contact_v1")
            refined[weak_index] = candidate
            for index in adjacent:
                endpoint_key = (
                    "fixed_end_anchor" if index == contact_index - 1 else "fixed_start_anchor"
                )
                sigma_key = (
                    "fixed_end_sigma_m" if index == contact_index - 1 else "fixed_start_sigma_m"
                )
                constraints[index][endpoint_key] = shared
                constraints[index][sigma_key] = fit_sigma_m
            diagnostic_rows.append(
                {
                    "contact_index": contact_index,
                    "status": "adopted_one_sided",
                    "adjacent_flights": adjacent,
                    "authority_flight": authority_index,
                    "authority_contact_error_px": authority_contact_error_px,
                    "refitted_flight": weak_index,
                    "junction_gap_m": junction_gap,
                    "pre_refit_endpoint_gap_m": endpoint_gap,
                    "position_prior_sigma_m": radius_sigma_m,
                    "fit_endpoint_sigma_m": fit_sigma_m,
                    "observation_source": observation_source,
                    "selection_guard": guard,
                    "marginal_contact_disagreement": marginal_contact_disagreement,
                    "opposite_endpoint_move_m": opposite_endpoint_move_m,
                    "pass": pass_index,
                }
            )
            changed = True
        if not changed:
            break

    for contact_index, original_contact in enumerate(contacts):
        if original_contact.get("terminal"):
            continue
        adjacent = [index for index in (contact_index - 1, contact_index) if index in refined]
        if len(adjacent) != 2:
            continue
        if _contact_already_reconciled(refined, adjacent, contact_index):
            continue
        contexts = [getattr(refined[index], "_anchor_first_context", None) for index in adjacent]
        if any(context is None for context in contexts):
            diagnostic_rows.append(
                {
                    "contact_index": contact_index,
                    "status": "missing_adjacent_flight_context",
                }
            )
            continue
        contexts = [context for context in contexts if context is not None]
        contact = dict(original_contact)
        event_frame = float(contact.get("event_frame", contact["frame"]))
        fixed_observation = _fixed_contact_observation(contact, contexts)
        if fixed_observation is None:
            diagnostic_rows.append(
                {"contact_index": contact_index, "status": "missing_contact_image_observation"}
            )
            continue
        pixel, observation_frame, observation_source = fixed_observation
        camera = contexts[-1]["camera"]
        projection = camera.p_at(observation_frame)

        previous_frame = (
            float(contacts[contact_index - 1]["frame"]) if contact_index > 0 else -math.inf
        )
        next_frame = (
            float(contacts[contact_index + 1]["frame"])
            if contact_index + 1 < len(contacts)
            else math.inf
        )
        time_lower = max(
            -CONTACT_BOUNDARY_TIME_LIMIT_FRAMES,
            previous_frame + 0.25 - event_frame,
        )
        time_upper = min(
            CONTACT_BOUNDARY_TIME_LIMIT_FRAMES,
            next_frame - 0.25 - event_frame,
        )
        if time_lower >= time_upper:
            diagnostic_rows.append(
                {"contact_index": contact_index, "status": "empty_boundary_time_interval"}
            )
            continue
        phase = str(contact.get("phase", "rally"))
        contact_sigma_px = float(contact.get("image_observation_sigma_px") or 10.0)
        # Without a witness the contact's time prior is a proxy read off the pixel sigma.  With
        # one (``flight_anchors.impact_time_prior``) the prior is centred where the track's own
        # corner says the impact was and is as wide as that witness claims.
        contact_time_prior = contact.get("time_prior") or {}
        time_prior_center_frames = float(contact_time_prior.get("offset_frames") or 0.0)
        time_prior_sigma_frames = (
            float(contact_time_prior["sigma_frames"])
            if contact_time_prior.get("sigma_frames")
            else float(np.clip(contact_sigma_px / 40.0, 0.15, 0.75))
        )
        time_prior_center_frames = float(np.clip(time_prior_center_frames, time_lower, time_upper))
        height_lower = 1.5 if phase == "serve" else 0.15
        height_upper = 3.5

        endpoint_heights = []
        for index, context in zip(adjacent, contexts, strict=True):
            fit = refined[index]
            try:
                endpoint_heights.append(
                    float(
                        fit.state(
                            event_frame,
                            context["fps"],
                            context["surface"],
                        )[0][2]
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                endpoint_heights.append(float(fit.theta[2]))
        tangent_seed = _trajectory_intersection_seed(
            [refined[index] for index in adjacent], contexts, event_frame
        )
        height_seeds = {
            float(np.clip(np.mean(endpoint_heights), height_lower, height_upper)),
            2.75 if phase == "serve" else 1.5,
        }
        if tangent_seed is not None:
            height_seeds.add(float(np.clip(tangent_seed[2], height_lower, height_upper)))
        cache: dict[tuple[float, float], tuple[float, float, np.ndarray, float, float]] = {}

        def evaluate_geometry(
            parameters: np.ndarray,
        ) -> tuple[float, float, np.ndarray, float, float] | None:
            time_offset = float(np.clip(parameters[0], time_lower, time_upper))
            height = float(np.clip(parameters[1], height_lower, height_upper))
            key = (round(time_offset, 6), round(height, 6))
            if key in cache:
                return cache[key]
            boundary_frame = event_frame + time_offset
            shared = ray_at_height(projection, pixel, height)
            if shared is None or not np.all(np.isfinite(shared)):
                return None
            if _SUBFRAME_CONTACTS and _SUBFRAME_CONTACT_ADVANCE:
                shared = _advance_emission_point(
                    shared,
                    [refined[index] for index in adjacent],
                    contexts,
                    observation_frame,
                    boundary_frame,
                )
            prior_sigma_m = _contact_radius_metres(
                projection,
                pixel,
                height,
                contact.get("image_observation_sigma_px"),
            )
            endpoints = []
            for index, context in zip(adjacent, contexts, strict=True):
                fit = refined[index]
                try:
                    endpoint = fit.state(
                        boundary_frame,
                        context["fps"],
                        context["surface"],
                    )[0]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    endpoint = fit.theta[:3]
                endpoints.append(np.asarray(endpoint, float))
            player, player_gap = striker_witness(
                contexts[-1].get("players", {}), str(contact.get("side")), boundary_frame
            )
            reach_residual = float(
                np.linalg.norm(
                    striker_contact_residuals(
                        shared,
                        player,
                        phase,
                        str(contact.get("side")),
                        striker_contact_sigma_m(player_gap, str(contact.get("side"))),
                        SERVE_CONTACT_REACH_M if phase == "serve" else MAX_CONTACT_REACH_M,
                    )
                )
            )
            if phase == "serve":
                serve_low, serve_high = SERVE_CONTACT_HEIGHT_RANGE_M
                height_residual = (
                    max(0.0, serve_low - height, height - serve_high) / SERVE_CONTACT_HEIGHT_SIGMA_M
                )
            else:
                pose_witness = contact.get("pose_witness") or {}
                apparent_height = pose_witness.get("apparent_ball_height_m")
                apparent_confidence = float(pose_witness.get("apparent_height_confidence") or 0.0)
                height_residual = (
                    (height - float(apparent_height))
                    / max(0.30, 0.70 * (1.0 - apparent_confidence))
                    if apparent_height is not None and apparent_confidence >= 0.20
                    else (height - 1.35) / 0.75
                )
            endpoint_residuals = [
                float(np.linalg.norm(shared - endpoint)) / prior_sigma_m for endpoint in endpoints
            ]
            objective = float(
                np.sum(np.square(endpoint_residuals))
                + reach_residual**2
                + height_residual**2
                + ((time_offset - time_prior_center_frames) / time_prior_sigma_frames) ** 2
            )
            endpoint_gap = (
                float(np.linalg.norm(endpoints[0] - endpoints[1])) if len(endpoints) == 2 else 0.0
            )
            result = (objective, boundary_frame, shared, endpoint_gap, prior_sigma_m)
            cache[key] = result
            return result

        for height in height_seeds:
            evaluate_geometry(np.asarray([time_prior_center_frames, height], dtype=float))
        if cache:
            seed = np.asarray(min(cache.items(), key=lambda row: row[1][0])[0], dtype=float)
            minimize(
                lambda parameters: (
                    candidate[0]
                    if (candidate := evaluate_geometry(parameters)) is not None
                    else 1e6
                ),
                seed,
                method="Powell",
                bounds=((time_lower, time_upper), (height_lower, height_upper)),
                options={
                    "maxfev": CONTACT_RAY_SEARCH_MAX_EVALUATIONS,
                    "xtol": 0.02,
                    "ftol": 0.01,
                },
            )
        if not cache:
            diagnostic_rows.append(
                {"contact_index": contact_index, "status": "no_ray_geometry_candidate"}
            )
            continue
        refit_geometries = []
        for geometry in sorted(cache.values(), key=lambda row: row[0]):
            _, boundary_frame, shared, _, _ = geometry
            if any(
                abs(boundary_frame - selected[1]) < 0.05
                and abs(float(shared[2]) - float(selected[2][2])) < 0.05
                for selected in refit_geometries
            ):
                continue
            refit_geometries.append(geometry)
            if len(refit_geometries) == CONTACT_RAY_REFIT_CANDIDATES:
                break
        successful = []
        refit_failures = {"fit_failed": 0, "held_out_worse": 0}
        for geometry in refit_geometries:
            geometry_objective, boundary_frame, shared, endpoint_gap, prior_sigma_m = geometry
            candidates = []
            for index, context in zip(adjacent, contexts, strict=True):
                shifted_contacts = [dict(value) for value in context["contacts"]]
                shifted_contact = dict(shifted_contacts[contact_index])
                shifted_contact.update(
                    {
                        "frame": boundary_frame,
                        "event_frame": event_frame,
                        "image_observation_override": pixel.tolist(),
                        "image_observation_frame": observation_frame,
                        "image_observation_source_frames": list(
                            contact.get("image_observation_source_frames", [])
                        ),
                        "image_observation_source": observation_source,
                    }
                )
                shifted_contacts[contact_index] = shifted_contact
                endpoint = (
                    "fixed_end_anchor" if index == contact_index - 1 else "fixed_start_anchor"
                )
                sigma_key = (
                    "fixed_end_sigma_m" if index == contact_index - 1 else "fixed_start_sigma_m"
                )
                candidate = fit_anchor_first_shot(
                    **{
                        **context,
                        **constraints[index],
                        endpoint: shared,
                        sigma_key: prior_sigma_m,
                        "contacts": shifted_contacts,
                        "initial_fit": refined[index],
                    }
                )
                if candidate is None:
                    refit_failures["fit_failed"] += 1
                    candidates = []
                    break
                summary = held_out_summary(candidate)
                if not _held_out_not_worse(held_out_summary(refined[index]), summary):
                    refit_failures["held_out_worse"] += 1
                    candidates = []
                    break
                candidates.append(candidate)
            if not candidates:
                continue
            junction_gap = 0.0
            if len(candidates) == 2:
                incoming_end = candidates[0].state(
                    boundary_frame, contexts[0]["fps"], contexts[0]["surface"]
                )[0]
                outgoing_start = candidates[1].state(
                    boundary_frame, contexts[1]["fps"], contexts[1]["surface"]
                )[0]
                junction_gap = float(np.linalg.norm(incoming_end - outgoing_start))
            if junction_gap >= endpoint_gap or junction_gap > 0.20:
                refit_failures["junction_not_improved"] = (
                    refit_failures.get("junction_not_improved", 0) + 1
                )
                continue
            fit_objective = float(
                geometry_objective
                + np.sum(np.square([float(candidate.rms_px) / 4.0 for candidate in candidates]))
                + (junction_gap / prior_sigma_m) ** 2
            )
            successful.append(
                (
                    fit_objective,
                    boundary_frame,
                    shared,
                    candidates,
                    junction_gap,
                    endpoint_gap,
                    prior_sigma_m,
                )
            )
        if not successful:
            diagnostic_rows.append(
                {
                    "contact_index": contact_index,
                    "status": "no_safe_soft_anchor_refit",
                    "adjacent_flights": adjacent,
                    "geometry_evaluations": len(cache),
                    "refit_failures": refit_failures,
                }
            )
            continue
        (
            objective,
            boundary_frame,
            shared,
            candidates,
            junction_gap,
            endpoint_gap,
            prior_sigma_m,
        ) = min(successful, key=lambda row: row[0])
        record = {
            "contact_index": contact_index,
            "frame": boundary_frame,
            "event_frame": event_frame,
            "observation_frame": observation_frame,
            "time_offset_frames": boundary_frame - event_frame,
            "xyz": shared.tolist(),
            "height_m": float(shared[2]),
            "observed_uv": pixel.tolist(),
            "observation_source": observation_source,
            "observation_sigma_px": contact.get("image_observation_sigma_px"),
            "position_prior_sigma_m": prior_sigma_m,
            "time_prior_sigma_frames": time_prior_sigma_frames,
            "junction_gap_m": junction_gap,
            "pre_refit_endpoint_gap_m": endpoint_gap,
            "objective": objective,
            "evaluations": len(cache),
            "refit_candidates": len(successful),
            "serve_height_prior_m": (
                list(SERVE_CONTACT_HEIGHT_RANGE_M) if phase == "serve" else None
            ),
            "player_reach_prior_m": list(
                SERVE_STRIKER_REACH_BAND_M if phase == "serve" else STRIKER_REACH_BAND_M
            ),
            "pose_height_prior": contact.get("pose_witness"),
        }
        diagnostic_rows.append(
            {
                "contact_index": contact_index,
                "status": "adopted_soft",
                "adjacent_flights": adjacent,
                "time_offset_frames": boundary_frame - event_frame,
                "height_m": float(shared[2]),
                "junction_gap_m": junction_gap,
                "position_prior_sigma_m": prior_sigma_m,
                "observation_source": observation_source,
                "initializer": (
                    "inbound_outbound_tangent_intersection"
                    if tangent_seed is not None
                    else "endpoint_height_mean"
                ),
            }
        )
        for index, candidate in zip(adjacent, candidates, strict=True):
            previous = refined[index]
            _copy_contact_metadata(previous, candidate)
            _append_shared_contact(candidate, record)
            object.__setattr__(candidate, "_fit_objective", "anchor_first_soft_contact_ray_v2")
            refined[index] = candidate
            endpoint = "fixed_end_anchor" if index == contact_index - 1 else "fixed_start_anchor"
            sigma_key = "fixed_end_sigma_m" if index == contact_index - 1 else "fixed_start_sigma_m"
            constraints[index][endpoint] = shared
            constraints[index][sigma_key] = prior_sigma_m
    return refined


def held_out_summary(fit: ShotFit) -> dict[str, Any]:
    errors = np.asarray(getattr(fit, "_held_out_errors_px", []), float)
    anchors = getattr(fit, "_anchor_errors", [])
    maximum_anchor_error = max((float(row["error_m"]) for row in anchors), default=None)
    median = _percentile(errors, 50.0)
    p90 = _percentile(errors, 90.0)
    return {
        "held_out_observations": int(len(errors)),
        "held_out_reprojection_median_px": median,
        "held_out_reprojection_p90_px": p90,
        "anchor_max_error_m": maximum_anchor_error,
        "anchor_satisfied": maximum_anchor_error is None
        or maximum_anchor_error <= ANCHOR_ERROR_LIMIT_M,
        "net_treatment": "post_fit_plausibility_only",
        "held_out_accepted": median is not None
        and p90 is not None
        and median <= HELD_OUT_MEDIAN_LIMIT_PX
        and p90 <= HELD_OUT_P90_LIMIT_PX,
        "contact_adjacent_weighting": bool(getattr(fit, "_contact_adjacent_weighting", False)),
        "contact_adjacent_weight": getattr(fit, "_contact_adjacent_weight", None),
        "contact_adjacent_frames": getattr(fit, "_contact_adjacent_frames", []),
    }
