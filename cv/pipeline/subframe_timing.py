"""Infer the sub-frame time of a contact or a bounce, before any 3D fit.

The event model emits on an integer native frame.  The impact it names happened
somewhere inside that frame's neighbourhood, so a fitter that pins its geometry
to the emitted frame is pinning it to a time that is wrong by up to half a frame
(``docs/wk1/bench_frames.md``: the ray-plane bounce knot that follows is 0.143 m
out at the median).  The fitter's own answer is to make the impact time a free
variable; this module supplies the *prior* on that variable, so the free time
starts from an informative distribution instead of a uniform window.

Four witnesses, each returning ``(t_subframe, sigma_frames, abstain)``:

``kinematic``
    The track's corner in the image.  Fit the inbound and outbound 2D paths on
    the track (excluding the impact frame), intersect them, and read the time
    each side reaches the intersection from that side's own per-frame
    displacement.  The two answers disagree by twice the sigma.

``kinematic_court``
    The same construction after the per-frame homography has taken the track to
    the court plane.  Bounces only: on the court plane the inbound and outbound
    ground tracks meet *at* the bounce, and the corner is not foreshortened the
    way the image corner is.

``streak``
    Motion-blur geometry.  Away from an impact a frame's blur streak is
    ``exposure * speed + ball_diameter`` long; on the impact frame the ball
    spends part of the exposure going one way, dwells, and comes back, so the
    streak is *short*.  The shortfall says where inside the exposure the impact
    fell.  The exposure fraction is a shutter-angle assumption
    (:data:`DEFAULT_EXPOSURE_FRACTION`, 180 degrees) and
    :func:`calibrate_exposure` measures it from flight frames alone.

``audio``
    The per-point audio onset of :mod:`cv.pipeline.audio_contact_timing`, which
    is already in native frame units with the per-broadcast A/V offset applied.

``dwell``
    The ball is at the impact spot for :data:`physics.bounce_reference.DWELL_SECONDS`
    (4.5 ms; 0.11 frames at 25 fps, 0.27 at 60).  The frame that was emitted is
    one whose exposure overlaps that dwell, which is a likelihood on ``t`` and
    the informative version of "the emitted integer frame".  It never abstains,
    so a combined estimate always exists.

The witnesses are combined by precision weighting with leave-one-out outlier
rejection; :attr:`SubframeEstimate.witnesses` keeps every one of them so a
caller can see what each said and score them separately.

Nothing here reads a label, a truth file or a 3D fit.
``cv/validation/subframe_timing_benchmark.py`` scores it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from cv.pipeline.contact_frame_refiner import normalise_track
from physics.bounce_reference import DWELL_SECONDS

__all__ = [
    "CALIBRATED_SIGMA_MODEL",
    "DEFAULT_EXPOSURE_FRACTION",
    "DEFAULT_BALL_DIAMETER_PX",
    "SIGMA_FEATURES",
    "UNIFORM_SIGMA_FRAMES",
    "WITNESS_NAMES",
    "WINDOW",
    "SigmaModel",
    "Witness",
    "SubframeEstimate",
    "calibrate_exposure",
    "calibrated_sigma",
    "combine",
    "combine_calibrated",
    "dwell_prior",
    "estimate",
    "estimate_prior",
    "measure_streak",
    "sigma_feature_vector",
    "witness_audio",
    "witness_dwell",
    "witness_diagnostics",
    "witness_kinematic",
    "witness_kinematic_court",
    "witness_streak",
]

WINDOW = 4
"""Track frames read on each side of the emitted frame (the impact frame is excluded)."""

MIN_SIDE_FRAMES = 2
"""Fewest usable frames a side needs before it can be fitted."""

DEFAULT_EXPOSURE_FRACTION = 0.5
"""Shutter angle assumption: 180 degrees, i.e. the exposure is half the frame period.

Broadcast tennis is shot with a short shutter to keep the ball readable, so this
is an assumption and not a measurement.  :func:`calibrate_exposure` measures it
from the data, using only flight frames, and the benchmark reports what it found.
"""

DEFAULT_BALL_DIAMETER_PX = 10.0
"""Apparent ball extent added to every streak, native px; also calibrated."""

MIN_CORNER_SIN = 0.0872
"""Smallest ``|sin(turn)|`` the corner intersection will answer on: 5 degrees.

Swept on the bench clean rung (``WK3_REPORT.md``).  Between 2 and 15 degrees the
witness's accuracy is flat -- median 0.063 frames on contacts, 0.007 on bounces
-- and only its answer rate moves, because the two sides' disagreement already
prices an ill-conditioned corner into the sigma.  Five degrees is the low edge
of that plateau with a margin against a near-collinear fit whose intersection
can land anywhere.
"""

MIN_SPEED_PX = 1.5
"""Slowest per-frame image displacement the corner and streak witnesses trust."""

MIN_SPEED_M = 0.05
"""Slowest per-frame court displacement the court-plane corner witness trusts."""

MAX_OFFSET_FRAMES = 1.5
"""A witness that lands further than this from the emitted frame abstains."""

CORNER_SIGMA_FLOOR = 0.02
"""Smallest sigma the corner witness will claim, frames."""

AUDIO_SIGMA_FRAMES = 0.45
"""Sigma claimed for an audio onset, frames.

Measured on the 409 development contacts that have an onset: the residual
against the owner label has a robust spread of 0.52 frames, and deconvolving the
label's own half-frame quantisation leaves 0.43.  The bench has no audio, so
this is the one witness whose sigma could only be calibrated on real data.
"""

AUDIO_OFFSET_FRAMES = 0.0
"""Convention shift applied to the audio onset before it is used, frames.

``docs/wk1/contactframe.md`` measured that ``audio_frame`` is already in native
image-index units for this cohort, so applying the -0.5 leading-blur convention
on top of it double-counts.  Zero is that measurement, kept as a named constant
so the benchmark can sweep it.
"""

MAX_AUDIO_OFFSET_FRAMES = 3.0
"""Audio onsets further than this from the emitted frame are not this event's."""

STREAK_SIGMA_FRACTION = 0.25
"""Streak-length measurement noise, as a fraction of the predicted flight streak."""

STREAK_SIGMA_SCALE = 2.2
"""Calibration of the streak sigma against what it earns.

Measured on the 689 development events with a streak: the residual's robust
spread is 0.64 frames against a claimed 0.26, and 0.57 after the label's own
quantisation is removed.  The bench has no images, so this too could only be
calibrated on real data.
"""

KINEMATIC_SIGMA_SCALE = 1.3
"""Calibration of the corner sigma, from the bench clean rung.

The two sides' disagreement plus the fit residual claims a median of 0.020
frames; the witness's own residual against the bench's continuous truth has a
robust spread of 0.025.  The core is nearly honest already and the scale only
closes that gap -- the tail is not Gaussian and no scale fixes it, which is why
the combiner also inflates by the chi-square factor.
"""

DWELL_SIGMA_SCALE = 2.15
"""Calibration of the dwell prior's width, from the bench clean rung.

The overlap model claims 0.173 frames at 25 fps; the prior's residual against
the bench's continuous truth has a robust spread of 0.372.  The model is
over-confident because the emitted frame is chosen by the track's corner and not
by "the ball is visibly at the impact spot", so its exposure need not overlap
the dwell at all.
"""

STREAK_MIN_SHORTFALL = 0.15
"""Fractional shortfall a frame's streak needs before the witness will read it.

A shorter streak than the flight model predicts is the whole of the evidence, so
without a threshold half of all frames would look short from measurement noise
alone.
"""

DWELL_FLOOR = 0.05
"""Uniform floor mixed into the dwell likelihood, as a fraction of its peak.

Without it the dwell witness hard-excludes every time outside the emitted
frame's exposure, which would make a single wrong exposure assumption fatal.
"""

WITNESS_NAMES = ("kinematic", "kinematic_court", "streak", "audio", "dwell")

OUTLIER_Z = 3.0
"""Leave-one-out standardised residual above which a witness is dropped."""


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Witness:
    """One witness's reading of the sub-frame impact time."""

    name: str
    t_subframe: float | None
    sigma_frames: float | None
    abstain: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "t_subframe": self.t_subframe,
            "sigma_frames": self.sigma_frames,
            "abstain": self.abstain,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SubframeEstimate:
    """The combined prior on a fitter's free impact-time variable."""

    t_subframe: float
    sigma_frames: float
    abstain: bool
    reason: str
    event_frame: int
    event_type: str
    fps: float
    witnesses: dict[str, Witness]
    used: tuple[str, ...]
    rejected: tuple[str, ...]
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    sigmas_calibrated: dict[str, float] = field(default_factory=dict)
    chi2_per_dof: float | None = None
    calibrated: bool = False

    @property
    def offset_frames(self) -> float:
        """Signed sub-frame offset from the emitted integer frame."""

        return self.t_subframe - float(self.event_frame)

    @property
    def t_seconds(self) -> float:
        return self.t_subframe / self.fps

    @property
    def sigma_seconds(self) -> float:
        return self.sigma_frames / self.fps

    def bounds(self, k: float = 2.0) -> tuple[float, float]:
        """A ``k``-sigma window for the fitter's free time variable, in frames."""

        return (self.t_subframe - k * self.sigma_frames, self.t_subframe + k * self.sigma_frames)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_subframe": self.t_subframe,
            "offset_frames": self.offset_frames,
            "sigma_frames": self.sigma_frames,
            "abstain": self.abstain,
            "reason": self.reason,
            "event_frame": self.event_frame,
            "event_type": self.event_type,
            "fps": self.fps,
            "used": list(self.used),
            "rejected": list(self.rejected),
            "calibrated": self.calibrated,
            "chi2_per_dof": self.chi2_per_dof,
            "sigmas_calibrated": dict(self.sigmas_calibrated),
            "diagnostics": {name: dict(values) for name, values in self.diagnostics.items()},
            "witnesses": {name: w.as_dict() for name, w in self.witnesses.items()},
        }


def _abstained(name: str, reason: str, **detail: Any) -> Witness:
    return Witness(name, None, None, True, reason, dict(detail))


def dwell_frames(fps: float) -> float:
    """The 4.5 ms racket/court dwell expressed in native frames."""

    return DWELL_SECONDS * float(fps)


# --------------------------------------------------------------------------
# side fits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _SideFit:
    """A quadratic-in-time fit of one side's 2D path, centred on ``anchor``."""

    anchor: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    residual: float
    frames: tuple[int, ...]

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)

    def at(self, t: float) -> tuple[float, float]:
        dt = t - self.anchor
        return (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
        )


def _side_samples(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
    direction: int,
    window: int,
) -> list[tuple[int, tuple[float, float]]]:
    """Frames on one side of ``event_frame``, nearest first, impact frame excluded."""

    samples: list[tuple[int, tuple[float, float]]] = []
    for step in range(1, window + 1):
        frame = event_frame + direction * step
        point = track.get(frame)
        if point is None:
            break
        samples.append((frame, point))
    return samples


def _fit_side(
    samples: Sequence[tuple[int, tuple[float, float]]],
    anchor: float,
) -> _SideFit | None:
    """Least-squares path through ``samples``, reported as position+velocity at ``anchor``.

    Three or more samples get a quadratic in time, which lets gravity and drag
    bend the two frames nearest the impact instead of biasing the extrapolated
    velocity; two samples get a straight line.
    """

    if len(samples) < MIN_SIDE_FRAMES:
        return None
    times = np.asarray([float(frame) - anchor for frame, _ in samples], dtype=float)
    points = np.asarray([point for _, point in samples], dtype=float)
    degree = 2 if len(samples) >= 3 else 1
    design = np.vander(times, degree + 1, increasing=True)
    try:
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coefficients).all():
        return None
    predicted = design @ coefficients
    residual = float(np.sqrt(np.mean(np.sum((points - predicted) ** 2, axis=1))))
    return _SideFit(
        anchor=anchor,
        position=(float(coefficients[0][0]), float(coefficients[0][1])),
        velocity=(float(coefficients[1][0]), float(coefficients[1][1])),
        residual=residual,
        frames=tuple(int(frame) for frame, _ in samples),
    )


def _intersect(
    inbound: _SideFit,
    outbound: _SideFit,
    min_speed: float,
    min_corner_sin: float = MIN_CORNER_SIN,
) -> tuple[float, float, float, dict[str, Any]] | None:
    """Intersect the two extrapolated paths; return ``(t_in, t_out, sin_turn, detail)``.

    ``t_in`` is the time the inbound path reaches the intersection, read from the
    inbound side's own per-frame displacement; ``t_out`` is the outbound side's
    answer to the same question.  Both are in frame units because the fitted
    velocities are per frame.
    """

    if inbound.speed < min_speed or outbound.speed < min_speed:
        return None
    d_in = np.asarray(inbound.velocity, dtype=float)
    d_out = np.asarray(outbound.velocity, dtype=float)
    unit_in = d_in / inbound.speed
    unit_out = d_out / outbound.speed
    cross = float(unit_in[0] * unit_out[1] - unit_in[1] * unit_out[0])
    if abs(cross) < min_corner_sin:
        return None
    origin = np.asarray(outbound.position, dtype=float) - np.asarray(inbound.position, dtype=float)
    denominator = d_in[0] * d_out[1] - d_in[1] * d_out[0]
    if abs(denominator) < 1e-12:
        return None
    u = (origin[0] * d_out[1] - origin[1] * d_out[0]) / denominator
    w = (origin[0] * d_in[1] - origin[1] * d_in[0]) / denominator
    t_in = inbound.anchor + u
    t_out = outbound.anchor + w
    meeting = inbound.at(t_in)
    detail = {
        "t_in": t_in,
        "t_out": t_out,
        "sin_turn": abs(cross),
        "turn_deg": math.degrees(math.asin(min(1.0, abs(cross)))),
        "speed_in": inbound.speed,
        "speed_out": outbound.speed,
        "residual_in": inbound.residual,
        "residual_out": outbound.residual,
        "intersection": [meeting[0], meeting[1]],
        "frames_in": list(inbound.frames),
        "frames_out": list(outbound.frames),
    }
    return t_in, t_out, abs(cross), detail


def _corner_witness(
    name: str,
    inbound: _SideFit | None,
    outbound: _SideFit | None,
    event_frame: int,
    dwell: float,
    min_speed: float,
    min_corner_sin: float = MIN_CORNER_SIN,
) -> Witness:
    """Shared body of the image and court-plane corner witnesses."""

    if inbound is None or outbound is None:
        return _abstained(name, "too_few_track_frames")
    solved = _intersect(inbound, outbound, min_speed, min_corner_sin)
    if solved is None:
        return _abstained(
            name,
            "ill_conditioned_corner",
            speed_in=inbound.speed,
            speed_out=outbound.speed,
        )
    t_in, t_out, _, detail = solved
    # The inbound path reaches the impact spot when the impact starts; the
    # outbound path leaves it a dwell later, so the outbound reading is late by
    # exactly the dwell and is corrected before the two are averaged.
    t_out_corrected = t_out - dwell
    detail["t_out_dwell_corrected"] = t_out_corrected
    detail["dwell_frames"] = dwell
    disagreement = abs(t_in - t_out_corrected)
    t = 0.5 * (t_in + t_out_corrected)
    noise = 0.5 * (
        inbound.residual / max(inbound.speed, min_speed)
        + outbound.residual / max(outbound.speed, min_speed)
    )
    sigma_raw = math.hypot(0.5 * disagreement, noise)
    sigma = max(KINEMATIC_SIGMA_SCALE * sigma_raw, CORNER_SIGMA_FLOOR)
    detail["disagreement_frames"] = disagreement
    detail["sigma_noise_frames"] = noise
    detail["sigma_raw_frames"] = sigma_raw
    if not math.isfinite(t) or not math.isfinite(sigma):
        return _abstained(name, "non_finite", **detail)
    if abs(t - float(event_frame)) > MAX_OFFSET_FRAMES:
        return _abstained(name, "corner_far_from_emitted_frame", **detail)
    return Witness(name, t, sigma, False, "corner_intersection", detail)


# --------------------------------------------------------------------------
# witness (a): the track's corner, in the image and on the court plane
# --------------------------------------------------------------------------


def witness_kinematic(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
    fps: float,
    window: int = WINDOW,
    min_corner_sin: float = MIN_CORNER_SIN,
) -> Witness:
    """Corner intersection of the inbound and outbound image paths."""

    before = list(reversed(_side_samples(track, event_frame, -1, window)))
    after = _side_samples(track, event_frame, +1, window)
    inbound = _fit_side(before, float(event_frame)) if before else None
    outbound = _fit_side(after, float(event_frame)) if after else None
    return _corner_witness(
        "kinematic",
        inbound,
        outbound,
        event_frame,
        dwell_frames(fps),
        MIN_SPEED_PX,
        min_corner_sin,
    )


def _to_court(
    point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    projected = homography @ np.asarray([point[0], point[1], 1.0], dtype=float)
    if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-9:
        return None
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def witness_kinematic_court(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
    fps: float,
    homography: Mapping[int, Any] | np.ndarray | None,
    window: int = WINDOW,
    min_corner_sin: float = MIN_CORNER_SIN,
) -> Witness:
    """Corner intersection after the per-frame homography, on the court plane.

    ``homography`` is image-to-court, the convention of ``court_H_per_frame_v1.npz``.
    A per-frame mapping is used per frame; a single matrix is used for every
    frame in the window.
    """

    if homography is None:
        return _abstained("kinematic_court", "no_homography")

    def matrix_for(frame: int) -> np.ndarray | None:
        if isinstance(homography, np.ndarray):
            return homography
        found = homography.get(frame) if hasattr(homography, "get") else None
        if found is None:
            return None
        return np.asarray(found, dtype=float)

    def project(samples: Sequence[tuple[int, tuple[float, float]]]):
        output: list[tuple[int, tuple[float, float]]] = []
        for frame, point in samples:
            matrix = matrix_for(frame)
            if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                continue
            court = _to_court(point, matrix)
            if court is None:
                continue
            output.append((frame, court))
        return output

    before = project(list(reversed(_side_samples(track, event_frame, -1, window))))
    after = project(_side_samples(track, event_frame, +1, window))
    inbound = _fit_side(before, float(event_frame)) if before else None
    outbound = _fit_side(after, float(event_frame)) if after else None
    return _corner_witness(
        "kinematic_court",
        inbound,
        outbound,
        event_frame,
        dwell_frames(fps),
        MIN_SPEED_M,
        min_corner_sin,
    )


# --------------------------------------------------------------------------
# witness (b): motion-blur streak geometry
# --------------------------------------------------------------------------


def measure_streak(
    patches: Sequence[np.ndarray | None],
    centre_index: int,
) -> float | None:
    """Major-axis extent of the moving blob in ``patches[centre_index]``, native px.

    The neighbouring patches supply a per-pixel median background so what is
    measured is the *moving* blob and not the court line under it; the blob is
    thresholded at ``max(12, median + 6 MAD)`` of the difference image and
    measured along its own principal axis.  Returns ``None`` when there is no
    usable neighbourhood or no blob.
    """

    centre = patches[centre_index] if 0 <= centre_index < len(patches) else None
    if centre is None:
        return None
    centre_array = np.asarray(centre, dtype=float)
    neighbours = [
        np.asarray(patch, dtype=float)
        for index, patch in enumerate(patches)
        if patch is not None and index != centre_index and np.shape(patch) == centre_array.shape
    ]
    if not neighbours:
        return None
    background = np.median(np.stack(neighbours, axis=0), axis=0)
    difference = np.abs(centre_array - background)
    flat = difference.reshape(-1)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    threshold = max(12.0, median + 6.0 * mad)
    mask = difference >= threshold
    if int(mask.sum()) < 4:
        return None
    rows, columns = np.nonzero(mask)
    coordinates = np.stack([columns.astype(float), rows.astype(float)], axis=1)
    centred = coordinates - coordinates.mean(axis=0)
    if len(centred) < 2:
        return None
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ basis[0]
    return float(projection.max() - projection.min())


def _two_leg_extent(
    t: float,
    frame: int,
    exposure: float,
    dwell: float,
    velocity_in: np.ndarray,
    velocity_out: np.ndarray,
    diameter: float,
) -> float:
    """Extent of the path the ball traces during frame ``frame``'s exposure.

    The exposure is centred on the frame index and spans ``exposure`` frames.
    Before ``t`` the ball runs on ``velocity_in``; between ``t`` and ``t+dwell``
    it is still; after that it runs on ``velocity_out``.  The extent is the
    major-axis spread of that path plus the ball's own apparent diameter, which
    is what :func:`measure_streak` measures.
    """

    start = frame - 0.5 * exposure
    stop = frame + 0.5 * exposure
    knots = [start, stop]
    for knot in (t, t + dwell):
        if start < knot < stop:
            knots.append(knot)
    knots = sorted(set(knots))
    position = np.zeros(2, dtype=float)
    points = [position.copy()]
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        middle = 0.5 * (left + right)
        if middle <= t:
            velocity = velocity_in
        elif middle <= t + dwell:
            velocity = np.zeros(2, dtype=float)
        else:
            velocity = velocity_out
        position = position + velocity * (right - left)
        points.append(position.copy())
    cloud = np.stack(points, axis=0)
    centred = cloud - cloud.mean(axis=0)
    if float(np.abs(centred).max()) < 1e-9:
        return diameter
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ basis[0]
    return float(projection.max() - projection.min()) + diameter


STREAK_SCAN = 1
"""Frames either side of the emitted frame whose streak is also examined.

The emitted frame is not necessarily the frame whose *exposure* holds the
impact, so the shortest streak relative to its own predicted flight length --
the frame whose blur changed direction -- brackets the impact instead.
"""


def witness_streak(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
    fps: float,
    streaks: Mapping[int, float],
    exposure: float = DEFAULT_EXPOSURE_FRACTION,
    diameter: float = DEFAULT_BALL_DIAMETER_PX,
    window: int = WINDOW,
    scan: int = STREAK_SCAN,
) -> Witness:
    """Solve the impact time from the shortfall in the impact frame's blur streak."""

    available = {
        int(frame): float(value)
        for frame, value in streaks.items()
        if value is not None and math.isfinite(float(value))
    }
    if not available:
        return _abstained("streak", "no_streak_on_emitted_frame")
    before = list(reversed(_side_samples(track, event_frame, -1, window)))
    after = _side_samples(track, event_frame, +1, window)
    inbound = _fit_side(before, float(event_frame)) if before else None
    outbound = _fit_side(after, float(event_frame)) if after else None
    if inbound is None or outbound is None:
        return _abstained("streak", "too_few_track_frames")
    if inbound.speed < MIN_SPEED_PX or outbound.speed < MIN_SPEED_PX:
        return _abstained("streak", "ball_too_slow", speed_in=inbound.speed)
    velocity_in = np.asarray(inbound.velocity, dtype=float)
    velocity_out = np.asarray(outbound.velocity, dtype=float)
    dwell = dwell_frames(fps)
    candidates: list[tuple[float, int, float, float]] = []
    for offset in range(-scan, scan + 1):
        frame = event_frame + offset
        measured = available.get(frame)
        if measured is None:
            continue
        # A frame before the emitted one is still on the inbound leg and a frame
        # after it on the outbound leg; only the emitted frame itself could hold
        # either, so it takes the larger of the two and is the hardest to call
        # short.
        if frame < event_frame:
            speed = inbound.speed
        elif frame > event_frame:
            speed = outbound.speed
        else:
            speed = max(inbound.speed, outbound.speed)
        flight = exposure * speed + diameter
        if measured >= flight * (1.0 - STREAK_MIN_SHORTFALL):
            continue
        candidates.append((measured / flight, frame, measured, flight))
    if not candidates:
        return _abstained(
            "streak",
            "no_shortfall",
            exposure=exposure,
            diameter_px=diameter,
            speed_in=inbound.speed,
            speed_out=outbound.speed,
            scanned=sorted(available),
        )
    _, frame, measured, flight = min(candidates)
    detail: dict[str, Any] = {
        "streak_frame": frame,
        "measured_px": measured,
        "flight_extent_px": flight,
        "exposure": exposure,
        "diameter_px": diameter,
        "speed_in": inbound.speed,
        "speed_out": outbound.speed,
        "dwell_frames": dwell,
    }
    grid = np.linspace(float(frame) - 0.5 * exposure, float(frame) + 0.5 * exposure, 81)
    curve = np.asarray(
        [
            _two_leg_extent(float(t), frame, exposure, dwell, velocity_in, velocity_out, diameter)
            for t in grid
        ],
        dtype=float,
    )
    residual = curve - measured
    roots: list[float] = []
    for index in range(len(grid) - 1):
        left, right = residual[index], residual[index + 1]
        if left == 0.0:
            roots.append(float(grid[index]))
        elif left * right < 0.0:
            span = left / (left - right)
            roots.append(float(grid[index] + span * (grid[index + 1] - grid[index])))
    if not roots:
        best = int(np.argmin(np.abs(residual)))
        roots = [float(grid[best])]
        detail["root_kind"] = "nearest"
    else:
        detail["root_kind"] = "crossing"
    detail["roots"] = roots
    t = float(np.mean(roots))
    slope = max(float(np.mean(np.abs(np.gradient(curve, grid)))), 1e-6)
    measurement_sigma = STREAK_SIGMA_SCALE * STREAK_SIGMA_FRACTION * flight / slope
    spread = 0.5 * (max(roots) - min(roots))
    sigma = max(math.hypot(measurement_sigma, spread), CORNER_SIGMA_FLOOR)
    detail["sigma_measurement_frames"] = measurement_sigma
    detail["sigma_root_spread_frames"] = spread
    if abs(t - float(event_frame)) > MAX_OFFSET_FRAMES:
        return _abstained("streak", "streak_far_from_emitted_frame", **detail)
    return Witness("streak", t, sigma, False, "streak_shortfall", detail)


def calibrate_exposure(
    samples: Sequence[tuple[float, float]],
) -> tuple[float, float, int]:
    """Fit ``streak = exposure * speed + diameter`` over flight frames.

    ``samples`` are ``(speed_px_per_frame, streak_px)`` pairs taken **away** from
    any impact, so no timing truth is read.  Returns
    ``(exposure_fraction, ball_diameter_px, n)``; falls back to the defaults when
    the fit is degenerate or leaves the physical range ``0 < exposure <= 1``.
    """

    usable = [
        (float(speed), float(streak))
        for speed, streak in samples
        if math.isfinite(speed) and math.isfinite(streak) and speed > MIN_SPEED_PX and streak > 0.0
    ]
    if len(usable) < 20:
        return DEFAULT_EXPOSURE_FRACTION, DEFAULT_BALL_DIAMETER_PX, len(usable)
    speeds = np.asarray([speed for speed, _ in usable], dtype=float)
    lengths = np.asarray([streak for _, streak in usable], dtype=float)
    design = np.stack([speeds, np.ones_like(speeds)], axis=1)
    solution, *_ = np.linalg.lstsq(design, lengths, rcond=None)
    exposure, diameter = float(solution[0]), float(solution[1])
    if not (0.0 < exposure <= 1.0) or not math.isfinite(diameter):
        return DEFAULT_EXPOSURE_FRACTION, DEFAULT_BALL_DIAMETER_PX, len(usable)
    return exposure, max(diameter, 0.0), len(usable)


# --------------------------------------------------------------------------
# witness (c): the audio onset
# --------------------------------------------------------------------------


def witness_audio(
    event_frame: int,
    audio_onset: float | None,
    offset_frames: float = AUDIO_OFFSET_FRAMES,
    sigma_frames: float = AUDIO_SIGMA_FRAMES,
) -> Witness:
    """The per-point audio onset, already in native frame units."""

    if audio_onset is None or not math.isfinite(float(audio_onset)):
        return _abstained("audio", "no_audio_onset")
    t = float(audio_onset) + offset_frames
    detail = {"audio_onset": float(audio_onset), "offset_frames": offset_frames}
    if abs(t - float(event_frame)) > MAX_AUDIO_OFFSET_FRAMES:
        return _abstained("audio", "audio_far_from_emitted_frame", **detail)
    return Witness("audio", t, sigma_frames, False, "audio_onset", detail)


# --------------------------------------------------------------------------
# witness (d): the dwell prior on the emitted frame
# --------------------------------------------------------------------------


def dwell_prior(
    event_frame: int,
    fps: float,
    exposure: float = DEFAULT_EXPOSURE_FRACTION,
    floor: float = DWELL_FLOOR,
    resolution: int = 401,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Mean, sigma and density of ``p(t | emitted frame)`` under the dwell model.

    The emitted frame is the image that shows the ball at the impact spot, so
    its exposure ``[k - a/2, k + a/2]`` has to overlap the dwell ``[t, t+d]``.
    The overlap length, as a function of ``t``, is a trapezoid; ``floor`` mixes
    a little uniform density under it so a wrong exposure assumption cannot
    exclude the truth outright.  Support is one frame either side, because a
    later frame would have been emitted instead.
    """

    dwell = dwell_frames(fps)
    grid = np.linspace(float(event_frame) - 0.5 - dwell, float(event_frame) + 0.5, resolution)
    start = float(event_frame) - 0.5 * exposure
    stop = float(event_frame) + 0.5 * exposure
    overlap = np.minimum(stop, grid + dwell) - np.maximum(start, grid)
    density = np.clip(overlap, 0.0, None)
    peak = float(density.max())
    if peak <= 0.0:
        density = np.ones_like(grid)
    else:
        density = density + floor * peak
    mass = float(np.trapezoid(density, grid))
    if mass <= 0.0:
        density = np.ones_like(grid)
        mass = float(np.trapezoid(density, grid))
    density = density / mass
    mean = float(np.trapezoid(grid * density, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    return mean, math.sqrt(max(variance, 0.0)), grid, density


def witness_dwell(
    event_frame: int,
    fps: float,
    exposure: float = DEFAULT_EXPOSURE_FRACTION,
) -> Witness:
    """The informative version of "the event is on this integer frame"."""

    mean, sigma, _, _ = dwell_prior(event_frame, fps, exposure)
    return Witness(
        "dwell",
        mean,
        max(DWELL_SIGMA_SCALE * sigma, CORNER_SIGMA_FLOOR),
        False,
        "dwell_overlap",
        {"exposure": exposure, "dwell_frames": dwell_frames(fps)},
    )


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


def _precision_mean(live: Sequence[Witness]) -> tuple[float, float, float | None]:
    """Precision-weighted mean, its width inflated by chi-square, and chi2 per degree."""

    weights = np.asarray([1.0 / float(w.sigma_frames) ** 2 for w in live], dtype=float)
    values = np.asarray([float(w.t_subframe) for w in live], dtype=float)
    mean = float(np.sum(weights * values) / np.sum(weights))
    sigma = math.sqrt(1.0 / float(np.sum(weights)))
    chi2_per_dof: float | None = None
    if len(live) > 1:
        chi2_per_dof = float(np.sum(weights * (values - mean) ** 2)) / (len(live) - 1)
        sigma *= max(1.0, math.sqrt(chi2_per_dof))
    return mean, sigma, chi2_per_dof


def _reject_outliers(
    live: list[Witness],
    keep: str,
    outlier_z: float,
) -> tuple[list[Witness], list[str]]:
    """Drop, one at a time, any witness whose leave-one-out residual exceeds ``outlier_z``."""

    live = list(live)
    rejected: list[str] = []
    while len(live) > 1:
        worst_index, worst_z = -1, 0.0
        for index, candidate in enumerate(live):
            if candidate.name == keep:
                continue
            others = [w for position, w in enumerate(live) if position != index]
            mean, sigma, _ = _precision_mean(others)
            spread = math.hypot(float(candidate.sigma_frames), sigma)
            z = abs(float(candidate.t_subframe) - mean) / max(spread, 1e-9)
            if z > worst_z:
                worst_index, worst_z = index, z
        if worst_index < 0 or worst_z <= outlier_z:
            break
        rejected.append(live.pop(worst_index).name)
    return live, rejected


def combine(
    witnesses: Sequence[Witness],
    keep: str = "dwell",
    outlier_z: float = OUTLIER_Z,
) -> tuple[float | None, float | None, tuple[str, ...], tuple[str, ...]]:
    """Precision-weighted mean with leave-one-out outlier rejection.

    ``keep`` names the witness that is never dropped: the dwell prior carries
    the emitted frame itself, so removing it would let two agreeing but
    misplaced witnesses drag the answer off the frame entirely.  The returned
    sigma is inflated by the chi-square consistency factor when the survivors
    still disagree by more than their sigmas allow.
    """

    live, rejected = _reject_outliers(
        [w for w in witnesses if not w.abstain and w.t_subframe is not None and w.sigma_frames],
        keep,
        outlier_z,
    )
    if not live:
        return None, None, (), ()
    mean, sigma, _ = _precision_mean(live)
    return mean, sigma, tuple(w.name for w in live), tuple(rejected)


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------


def _streak_lengths(
    crops: Mapping[Any, Any] | None,
    event_frame: int,
    window: int,
) -> dict[int, float]:
    """Coerce ``crops`` to ``frame -> streak length in native px``.

    A float value is taken as an already-measured streak; a 2-D array is taken
    as a native grayscale patch of the same rectangle on that frame and is
    measured here against its neighbours.
    """

    if not crops:
        return {}
    lengths: dict[int, float] = {}
    patches: dict[int, np.ndarray] = {}
    for frame, value in crops.items():
        key = int(frame)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            lengths[key] = float(value)
            continue
        if isinstance(value, Mapping) and "streak_px" in value:
            candidate = value["streak_px"]
            if candidate is not None and math.isfinite(float(candidate)):
                lengths[key] = float(candidate)
            continue
        array = np.asarray(value)
        if array.ndim == 3:
            array = array.mean(axis=2)
        if array.ndim == 2:
            patches[key] = array.astype(float)
    if patches:
        frames = sorted(patches)
        ordered = [patches[frame] for frame in frames]
        for index, frame in enumerate(frames):
            if abs(frame - event_frame) > window:
                continue
            measured = measure_streak(ordered, index)
            if measured is not None:
                lengths.setdefault(frame, measured)
    return lengths


def _build_witnesses(
    track: Mapping[int, tuple[float, float]],
    frame: int,
    event_type: str,
    fps: float,
    crops: Mapping[Any, Any] | None,
    audio_onset: float | None,
    homography: Mapping[int, Any] | np.ndarray | None,
    *,
    exposure: float,
    diameter: float,
    audio_offset_frames: float,
    audio_sigma_frames: float,
    window: int,
    min_corner_sin: float,
    enabled: Sequence[str],
) -> dict[str, Witness]:
    """Run every enabled witness once; the shared body of the two entry points."""

    active = set(enabled)
    witnesses: dict[str, Witness] = {}

    if "kinematic" in active:
        witnesses["kinematic"] = witness_kinematic(track, frame, fps, window, min_corner_sin)
    else:
        witnesses["kinematic"] = _abstained("kinematic", "disabled")

    if "kinematic_court" in active and event_type == "bounce":
        witnesses["kinematic_court"] = witness_kinematic_court(
            track, frame, fps, homography, window, min_corner_sin
        )
    else:
        witnesses["kinematic_court"] = _abstained(
            "kinematic_court",
            "disabled" if "kinematic_court" not in active else "not_a_bounce",
        )

    if "streak" in active:
        lengths = _streak_lengths(crops, frame, window)
        witnesses["streak"] = (
            witness_streak(track, frame, fps, lengths, exposure, diameter, window)
            if lengths
            else _abstained("streak", "no_crops")
        )
    else:
        witnesses["streak"] = _abstained("streak", "disabled")

    if "audio" in active:
        witnesses["audio"] = witness_audio(
            frame, audio_onset, audio_offset_frames, audio_sigma_frames
        )
    else:
        witnesses["audio"] = _abstained("audio", "disabled")

    if "dwell" in active:
        witnesses["dwell"] = witness_dwell(frame, fps, exposure)
    else:
        witnesses["dwell"] = _abstained("dwell", "disabled")

    return witnesses


def estimate(
    track_rows: Mapping[Any, Any],
    event_frame: float | int,
    event_type: str,
    fps: float,
    crops: Mapping[Any, Any] | None = None,
    audio_onset: float | None = None,
    homography: Mapping[int, Any] | np.ndarray | None = None,
    *,
    exposure: float = DEFAULT_EXPOSURE_FRACTION,
    diameter: float = DEFAULT_BALL_DIAMETER_PX,
    audio_offset_frames: float = AUDIO_OFFSET_FRAMES,
    audio_sigma_frames: float = AUDIO_SIGMA_FRAMES,
    window: int = WINDOW,
    min_corner_sin: float = MIN_CORNER_SIN,
    enabled: Sequence[str] = WITNESS_NAMES,
) -> SubframeEstimate:
    """The combined sub-frame time of one emitted contact or bounce.

    ``track_rows`` maps native frame index to the composed track's position;
    the same shapes :func:`cv.pipeline.contact_frame_refiner.normalise_track`
    accepts are accepted here.  ``event_frame`` is the event model's integer
    emitted frame.  ``crops`` maps frame index to a measured streak length or a
    native grayscale patch; ``audio_onset`` is the audio contact onset in native
    frame units; ``homography`` is image-to-court, per frame or one matrix.
    All three are optional and each drops one witness when it is missing.

    The result is the prior a fitter should put on its free impact-time
    variable: ``t_subframe`` with ``sigma_frames``, or the two-sigma
    :meth:`SubframeEstimate.bounds` window.  It never abstains while an emitted
    frame and an fps exist, because the dwell witness answers from those alone.
    """

    frame = int(round(float(event_frame)))
    fps = float(fps)
    track = normalise_track(track_rows or {})
    witnesses = _build_witnesses(
        track,
        frame,
        event_type,
        fps,
        crops,
        audio_onset,
        homography,
        exposure=exposure,
        diameter=diameter,
        audio_offset_frames=audio_offset_frames,
        audio_sigma_frames=audio_sigma_frames,
        window=window,
        min_corner_sin=min_corner_sin,
        enabled=enabled,
    )

    mean, sigma, used, rejected = combine([witnesses[name] for name in WITNESS_NAMES])
    if mean is None or sigma is None:
        return SubframeEstimate(
            t_subframe=float(frame),
            sigma_frames=1.0 / math.sqrt(12.0),
            abstain=True,
            reason="no_witness",
            event_frame=frame,
            event_type=str(event_type),
            fps=fps,
            witnesses=witnesses,
            used=(),
            rejected=(),
        )
    clipped = min(max(mean, frame - 1.0), frame + 1.0)
    return SubframeEstimate(
        t_subframe=clipped,
        sigma_frames=max(sigma, CORNER_SIGMA_FLOOR),
        abstain=False,
        reason="combined" if len(used) > 1 else f"single:{used[0]}",
        event_frame=frame,
        event_type=str(event_type),
        fps=fps,
        witnesses=witnesses,
        used=used,
        rejected=rejected,
    )


# --------------------------------------------------------------------------
# the calibrated width
# --------------------------------------------------------------------------
#
# The sigma each witness claims above is a *model* of its own noise: for the
# corner witnesses the two sides' disagreement, for the dwell witness the width
# of the overlap trapezoid.  ``docs/wk1/point_fit8.md`` section 3 measured what
# happens when a fitter believes it -- clipped into [0.05, 0.42] frames it took
# the bench clean rung from 43 to 25 complete points -- and named the reason: the
# width is not calibrated to *when the witness is wrong*.  A claimed 0.05 frames
# on an event whose corner is 0.6 frames out is a confidently wrong prior, and a
# confidently wrong prior is worse than no prior.
#
# What follows replaces the claimed width with one fitted against the truth on
# the bench, as a function of the witness's own label-free diagnostics.  The
# model is
#
#     sigma(x) = clip( exp(x . beta) * s , floor, ceiling )
#
# where ``beta`` is a quantile regression of ``log |error|`` on the diagnostic
# vector ``x`` at the 68th percentile -- quantiles commute with the logarithm, so
# ``exp(x . beta)`` *is* the conditional 68th percentile of the absolute error --
# and ``s`` is one scalar per witness that repairs the tail:
# ``s = max(q68(u), q95(u) / 2)`` over the normalised residual
# ``u = |error| / exp(x . beta)``.  The first term holds the 68% coverage at one
# sigma and the second forces at least 95% at two sigma, so a heavy tail widens
# the core instead of being hidden by it.  Both are fitted on the development
# half of the bench matches and verified once on the other half
# (``cv/validation/subframe_timing_benchmark.py calibrate``).


UNIFORM_SIGMA_FRAMES = 0.42
"""The width a fitter uses when there is no witness, frames.

This is ``cv.pipeline.flight_anchors.ANCHOR_TIME_SIGMA_FRAMES``, restated here so
this module never imports the fitter: one frame of quantisation (1/sqrt(12) =
0.289) in quadrature with the measured emitter error of ``docs/wk1/s6_bench.md``.
An abstaining prior returns the emitted frame at this width, which is exactly
what the fitter already does, so an abstention costs the fitter nothing.
"""

CALIBRATED_SIGMA_MIN = 0.01
"""Floor on a calibrated width, frames.  Below this the prior pins the time."""

CALIBRATED_SIGMA_MAX = 2.0
"""Ceiling on a calibrated width, frames.

Deliberately far above the uniform window.  A witness the model says is two
frames uncertain is useless, but it has to be allowed to *say* two frames or its
coverage is a lie -- and the combination's ``not_better_than_uniform`` rule
already turns a useless width into an abstention.  Clipping here at the uniform
window instead cost the dwell witness eight points of two-sigma coverage on the
``event_timing`` rung, where the emitted frame is whole frames wrong.
"""

CHI2_ABSTAIN = 4.0
"""Chi-square per degree of freedom above which the survivors are called inconsistent.

The witnesses have already been widened to their honest one-sigma widths, so a
chi-square per degree of freedom of four means they disagree by twice those
widths.  At that point the combination's own centre is not to be trusted and the
prior abstains rather than pull the fitter to a place no witness named.
"""

SIGMA_FEATURES: dict[str, tuple[str, ...]] = {
    "kinematic": (
        "bias",
        "log_sigma_raw",
        "log_inv_sin_turn",
        "log_min_speed",
        "log1p_disagreement",
        "is_contact",
        "log_fps_ratio",
        "is_far",
        "short_sides",
        "has_partner",
        "partner_log_gap",
        "log1p_offset",
    ),
    "kinematic_court": (
        "bias",
        "log_sigma_raw",
        "log_inv_sin_turn",
        "log_min_speed",
        "log1p_disagreement",
        "log_fps_ratio",
        "is_far",
        "short_sides",
        "has_partner",
        "partner_log_gap",
        "log1p_offset",
    ),
    "dwell": (
        "bias",
        "log_dwell_frames",
        "is_contact",
        "is_far",
        "turn_margin",
        "has_turn_margin",
        "log1p_speed",
    ),
}
"""The label-free diagnostics each modelled witness's width is a function of.

``kinematic_court`` runs on bounces only, so ``is_contact`` is constant there and
is left out.  ``streak`` and ``audio`` have no entry: the bench renders no images
and carries no audio, so neither could be calibrated against a continuous truth
and both keep the claimed width of ``docs/wk1/subframe_timing.md``.
"""

CALIBRATED_WITNESSES = tuple(SIGMA_FEATURES)


@dataclass(frozen=True)
class SigmaModel:
    """A fitted width per witness: ``clip(exp(x . beta) * scale, floor, ceiling)``."""

    coefficients: Mapping[str, Sequence[float]]
    scale: Mapping[str, float]
    combined_scale: float = 1.0
    floor: float = CALIBRATED_SIGMA_MIN
    ceiling: float = CALIBRATED_SIGMA_MAX
    provenance: str = ""

    def has(self, name: str) -> bool:
        return name in self.coefficients and name in SIGMA_FEATURES

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficients": {name: list(row) for name, row in self.coefficients.items()},
            "scale": dict(self.scale),
            "combined_scale": self.combined_scale,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "provenance": self.provenance,
            "features": {name: list(SIGMA_FEATURES[name]) for name in self.coefficients},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SigmaModel":
        return cls(
            coefficients={
                str(name): tuple(float(value) for value in row)
                for name, row in payload["coefficients"].items()
            },
            scale={str(name): float(value) for name, value in payload["scale"].items()},
            combined_scale=float(payload.get("combined_scale", 1.0)),
            floor=float(payload.get("floor", CALIBRATED_SIGMA_MIN)),
            ceiling=float(payload.get("ceiling", CALIBRATED_SIGMA_MAX)),
            provenance=str(payload.get("provenance", "")),
        )


def _log(value: float, floor: float = 1e-6) -> float:
    return math.log(max(float(value), floor))


def _turn_deg(track: Mapping[int, tuple[float, float]], frame: int) -> float | None:
    """Turn angle of the track at ``frame``, degrees, or ``None`` without neighbours."""

    before, here, after = track.get(frame - 1), track.get(frame), track.get(frame + 1)
    if before is None or here is None or after is None:
        return None
    first = (here[0] - before[0], here[1] - before[1])
    second = (after[0] - here[0], after[1] - here[1])
    norms = (math.hypot(*first), math.hypot(*second))
    if min(norms) < 1e-9:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (norms[0] * norms[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def frame_ambiguity(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
) -> tuple[float | None, float | None]:
    """How clearly the emitted frame is the track's corner, and the local speed.

    The emitter picks the frame at which the track turns its sharpest corner, so
    the margin between that frame's turn angle and the sharpest of its two
    neighbours' says how nearly a different frame was emitted.  A small margin is
    a frame that could as well have been ``k-1`` or ``k+1``, which is the dwell
    witness's whole uncertainty.  Both quantities read the track alone.
    """

    here = _turn_deg(track, event_frame)
    if here is None:
        return None, None
    neighbours = [
        value
        for value in (_turn_deg(track, event_frame - 1), _turn_deg(track, event_frame + 1))
        if value is not None
    ]
    margin = here - max(neighbours) if neighbours else None
    before, after = track.get(event_frame - 1), track.get(event_frame + 1)
    speed = None
    if before is not None and after is not None:
        speed = 0.5 * math.hypot(after[0] - before[0], after[1] - before[1])
    return margin, speed


def _side_flag(court_side: str | None) -> float:
    """1 for the far court, 0 for the near court, 0.5 when it is not known."""

    if court_side == "far":
        return 1.0
    if court_side == "near":
        return 0.0
    return 0.5


def witness_diagnostics(
    witnesses: Mapping[str, Witness],
    *,
    event_frame: int,
    event_type: str,
    fps: float,
    track: Mapping[int, tuple[float, float]] | None = None,
    court_side: str | None = None,
) -> dict[str, dict[str, float]]:
    """The label-free diagnostics each modelled witness's width is conditioned on.

    Nothing here reads a truth file: the corner angle, the two step lengths, the
    two sides' disagreement, how far the answer sits from the emitted frame, the
    agreement between the image corner and the court-plane corner, the sharpness
    of the emitted frame's own corner against its neighbours, the event type, the
    frame rate and which half of the court the ball is on.
    """

    is_contact = 1.0 if str(event_type) == "contact" else 0.0
    is_far = _side_flag(court_side)
    output: dict[str, dict[str, float]] = {}
    partners = {"kinematic": "kinematic_court", "kinematic_court": "kinematic"}
    for name in ("kinematic", "kinematic_court"):
        witness = witnesses.get(name)
        if witness is None or witness.abstain or witness.t_subframe is None:
            continue
        detail = witness.detail
        partner = witnesses.get(partners[name])
        has_partner = (
            1.0 if partner is not None and not partner.abstain and partner.t_subframe else 0.0
        )
        gap = (
            abs(float(witness.t_subframe) - float(partner.t_subframe))
            if has_partner and partner is not None and partner.t_subframe is not None
            else 0.0
        )
        n_in = len(detail.get("frames_in") or ())
        n_out = len(detail.get("frames_out") or ())
        output[name] = {
            "sigma_raw_frames": float(detail.get("sigma_raw_frames") or 0.0),
            "sin_turn": float(detail.get("sin_turn") or 0.0),
            "turn_deg": float(detail.get("turn_deg") or 0.0),
            "speed_in": float(detail.get("speed_in") or 0.0),
            "speed_out": float(detail.get("speed_out") or 0.0),
            "min_speed": float(min(detail.get("speed_in") or 0.0, detail.get("speed_out") or 0.0)),
            "disagreement_frames": float(detail.get("disagreement_frames") or 0.0),
            "noise_frames": float(detail.get("sigma_noise_frames") or 0.0),
            "n_in": float(n_in),
            "n_out": float(n_out),
            "short_sides": float((n_in < 3) + (n_out < 3)),
            "offset_frames": abs(float(witness.t_subframe) - float(event_frame)),
            "has_partner": has_partner,
            "partner_gap_frames": float(gap),
            "is_contact": is_contact,
            "is_far": is_far,
            "fps": float(fps),
        }
    dwell = witnesses.get("dwell")
    if dwell is not None and not dwell.abstain and dwell.t_subframe is not None:
        margin, speed = frame_ambiguity(track or {}, event_frame)
        output["dwell"] = {
            "dwell_frames": dwell_frames(fps),
            "is_contact": is_contact,
            "is_far": is_far,
            "turn_margin_deg": 0.0 if margin is None else float(margin),
            "has_turn_margin": 0.0 if margin is None else 1.0,
            "local_speed_px": 0.0 if speed is None else float(speed),
            "fps": float(fps),
        }
    return output


def sigma_feature_vector(name: str, diagnostics: Mapping[str, float]) -> np.ndarray:
    """The design row ``x`` for one witness on one event, ordered by :data:`SIGMA_FEATURES`."""

    values: dict[str, float] = {"bias": 1.0}
    if name in ("kinematic", "kinematic_court"):
        values.update(
            {
                "log_sigma_raw": _log(diagnostics.get("sigma_raw_frames", 0.0), 1e-4),
                "log_inv_sin_turn": -_log(diagnostics.get("sin_turn", 0.0), 1e-3),
                "log_min_speed": _log(diagnostics.get("min_speed", 0.0), 1e-3),
                "log1p_disagreement": math.log1p(
                    max(0.0, diagnostics.get("disagreement_frames", 0.0))
                ),
                "is_contact": diagnostics.get("is_contact", 0.0),
                "log_fps_ratio": _log(diagnostics.get("fps", 25.0) / 25.0, 1e-3),
                "is_far": diagnostics.get("is_far", 0.5),
                "short_sides": diagnostics.get("short_sides", 0.0),
                "has_partner": diagnostics.get("has_partner", 0.0),
                "partner_log_gap": diagnostics.get("has_partner", 0.0)
                * _log(diagnostics.get("partner_gap_frames", 0.0), 1e-3),
                "log1p_offset": math.log1p(max(0.0, diagnostics.get("offset_frames", 0.0))),
            }
        )
    elif name == "dwell":
        values.update(
            {
                "log_dwell_frames": _log(diagnostics.get("dwell_frames", 0.0), 1e-3),
                "is_contact": diagnostics.get("is_contact", 0.0),
                "is_far": diagnostics.get("is_far", 0.5),
                "turn_margin": max(-2.0, min(2.0, diagnostics.get("turn_margin_deg", 0.0) / 45.0)),
                "has_turn_margin": diagnostics.get("has_turn_margin", 0.0),
                "log1p_speed": math.log1p(max(0.0, diagnostics.get("local_speed_px", 0.0))),
            }
        )
    else:
        raise KeyError(f"no width model for witness {name!r}")
    return np.asarray([values[feature] for feature in SIGMA_FEATURES[name]], dtype=float)


def calibrated_sigma(
    name: str,
    diagnostics: Mapping[str, float],
    model: SigmaModel | None,
) -> float | None:
    """The fitted width for one witness on one event, or ``None`` without a model."""

    if model is None or not model.has(name):
        return None
    row = sigma_feature_vector(name, diagnostics)
    coefficients = np.asarray(model.coefficients[name], dtype=float)
    if coefficients.shape != row.shape:
        raise ValueError(f"width model for {name!r} has {coefficients.size} of {row.size} features")
    predicted = float(np.exp(float(row @ coefficients)) * float(model.scale.get(name, 1.0)))
    if not math.isfinite(predicted):
        return None
    return float(min(max(predicted, model.floor), model.ceiling))


@dataclass(frozen=True)
class CalibratedCombination:
    """The combined prior after every width has been replaced by its fitted one."""

    t_subframe: float
    sigma_frames: float
    abstain: bool
    reason: str
    used: tuple[str, ...]
    rejected: tuple[str, ...]
    sigmas: dict[str, float]
    chi2_per_dof: float | None


def combine_calibrated(
    witnesses: Mapping[str, Witness],
    diagnostics: Mapping[str, Mapping[str, float]],
    model: SigmaModel | None,
    *,
    event_frame: int,
    keep: str = "dwell",
    outlier_z: float = OUTLIER_Z,
    chi2_abstain: float = CHI2_ABSTAIN,
    uniform_sigma: float = UNIFORM_SIGMA_FRAMES,
    abstain_above: float | None = None,
) -> CalibratedCombination:
    """Precision-weight the witnesses at their calibrated widths, or abstain.

    The prior abstains -- and hands back the emitted frame at the uniform window's
    own width, which is what the fitter uses today -- in three cases:

    ``dwell_only``
        no witness but the dwell prior answered, so the "estimate" is the emitted
        integer frame restated.  That is the weak-corner case: the corner was
        inside the turn gate, or the track had too few frames on one side.
    ``witnesses_disagree``
        the survivors of the outlier rejection still disagree by more than their
        calibrated widths allow (chi-square per degree of freedom above
        ``chi2_abstain``).
    ``not_better_than_uniform``
        the combined calibrated width is no narrower than the uniform window, so
        the prior has nothing to add.

    ``abstain_above`` is the width that last rule tests against and defaults to
    ``uniform_sigma``; the calibration pass sets it to infinity so it can measure
    the combination's own coverage before the rule removes the wide half.
    """

    if abstain_above is None:
        abstain_above = uniform_sigma

    sigmas: dict[str, float] = {}
    live: list[Witness] = []
    for name in WITNESS_NAMES:
        witness = witnesses.get(name)
        if witness is None or witness.abstain or witness.t_subframe is None:
            continue
        fitted = calibrated_sigma(name, diagnostics.get(name, {}), model)
        if fitted is None:
            fitted = float(witness.sigma_frames or uniform_sigma)
        sigmas[name] = fitted
        live.append(
            Witness(name, witness.t_subframe, fitted, False, witness.reason, witness.detail)
        )

    uniform = CalibratedCombination(
        t_subframe=float(event_frame),
        sigma_frames=float(uniform_sigma),
        abstain=True,
        reason="no_witness",
        used=(),
        rejected=(),
        sigmas=sigmas,
        chi2_per_dof=None,
    )
    if not live:
        return uniform

    survivors, rejected = _reject_outliers(live, keep, outlier_z)
    mean, sigma, chi2_per_dof = _precision_mean(survivors)
    # The witnesses' own widths are calibrated but their errors are correlated --
    # both corners read the same track -- so the precision-weighted width is
    # optimistic even when every input is honest.  One scalar, fitted the same
    # way as the per-witness tail scale, repairs the combination's own coverage.
    if model is not None:
        sigma *= float(model.combined_scale)
    used = tuple(w.name for w in survivors)
    informative = tuple(name for name in used if name != keep)

    from dataclasses import replace as _replace

    if not informative:
        return _replace(uniform, reason="dwell_only", rejected=tuple(rejected), sigmas=sigmas)
    if chi2_per_dof is not None and chi2_per_dof > chi2_abstain:
        return _replace(
            uniform,
            reason="witnesses_disagree",
            rejected=tuple(rejected),
            sigmas=sigmas,
            chi2_per_dof=chi2_per_dof,
        )
    if sigma >= abstain_above:
        return _replace(
            uniform,
            reason="not_better_than_uniform",
            rejected=tuple(rejected),
            sigmas=sigmas,
            chi2_per_dof=chi2_per_dof,
        )
    return CalibratedCombination(
        t_subframe=float(min(max(mean, event_frame - 1.0), event_frame + 1.0)),
        sigma_frames=float(max(sigma, CALIBRATED_SIGMA_MIN)),
        abstain=False,
        reason="combined" if len(used) > 1 else f"single:{used[0]}",
        used=used,
        rejected=tuple(rejected),
        sigmas=sigmas,
        chi2_per_dof=chi2_per_dof,
    )


def estimate_prior(
    track_rows: Mapping[Any, Any],
    event_frame: float | int,
    event_type: str,
    fps: float,
    crops: Mapping[Any, Any] | None = None,
    audio_onset: float | None = None,
    homography: Mapping[int, Any] | np.ndarray | None = None,
    *,
    court_side: str | None = None,
    model: SigmaModel | None = None,
    exposure: float = DEFAULT_EXPOSURE_FRACTION,
    diameter: float = DEFAULT_BALL_DIAMETER_PX,
    audio_offset_frames: float = AUDIO_OFFSET_FRAMES,
    audio_sigma_frames: float = AUDIO_SIGMA_FRAMES,
    window: int = WINDOW,
    min_corner_sin: float = MIN_CORNER_SIN,
    enabled: Sequence[str] = WITNESS_NAMES,
    chi2_abstain: float = CHI2_ABSTAIN,
    uniform_sigma: float = UNIFORM_SIGMA_FRAMES,
    abstain_above: float | None = None,
) -> SubframeEstimate:
    """:func:`estimate` with the calibrated widths and an explicit abstention.

    This is what a fitter's prior should read.  ``t_subframe`` is the combined
    time, ``sigma_frames`` its calibrated one-sigma width and ``abstain`` says
    whether to use it at all -- when it is ``True`` the returned pair *is* the
    uniform window (the emitted frame at :data:`UNIFORM_SIGMA_FRAMES`), so a
    caller that ignores the flag still gets today's behaviour.

    ``estimate`` is left exactly as it was, claimed widths and all, so nothing
    that reads it changes.  ``model`` defaults to :data:`CALIBRATED_SIGMA_MODEL`;
    passing ``None`` explicitly is not the same as leaving it out only in that a
    missing model falls back to the claimed widths.
    """

    frame = int(round(float(event_frame)))
    fps = float(fps)
    track = normalise_track(track_rows or {})
    witnesses = _build_witnesses(
        track,
        frame,
        event_type,
        fps,
        crops,
        audio_onset,
        homography,
        exposure=exposure,
        diameter=diameter,
        audio_offset_frames=audio_offset_frames,
        audio_sigma_frames=audio_sigma_frames,
        window=window,
        min_corner_sin=min_corner_sin,
        enabled=enabled,
    )
    side = court_side if court_side is not None else _court_side(track, frame, homography)
    diagnostics = witness_diagnostics(
        witnesses,
        event_frame=frame,
        event_type=str(event_type),
        fps=fps,
        track=track,
        court_side=side,
    )
    combination = combine_calibrated(
        witnesses,
        diagnostics,
        CALIBRATED_SIGMA_MODEL if model is None else model,
        event_frame=frame,
        chi2_abstain=chi2_abstain,
        uniform_sigma=uniform_sigma,
        abstain_above=abstain_above,
    )
    return SubframeEstimate(
        t_subframe=combination.t_subframe,
        sigma_frames=combination.sigma_frames,
        abstain=combination.abstain,
        reason=combination.reason,
        event_frame=frame,
        event_type=str(event_type),
        fps=fps,
        witnesses=witnesses,
        used=combination.used,
        rejected=combination.rejected,
        diagnostics=diagnostics,
        sigmas_calibrated=combination.sigmas,
        chi2_per_dof=combination.chi2_per_dof,
        calibrated=True,
    )


def _court_side(
    track: Mapping[int, tuple[float, float]],
    event_frame: int,
    homography: Mapping[int, Any] | np.ndarray | None,
) -> str | None:
    """Which half of the court the emitted pixel projects to, or ``None``.

    The homography is the court *ground* plane, so a contact struck at shoulder
    height does not project to where the racket is -- but it does project to the
    right side of the net, which is all this diagnostic is asked for.
    """

    point = track.get(event_frame)
    if point is None or homography is None:
        return None
    if isinstance(homography, np.ndarray):
        matrix = homography
    else:
        found = homography.get(event_frame) if hasattr(homography, "get") else None
        if found is None:
            return None
        matrix = np.asarray(found, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    court = _to_court(point, matrix)
    if court is None:
        return None
    return "near" if court[1] < NET_Y_M else "far"


NET_Y_M = 11.885
"""Half the court's length, metres: the net line the near/far diagnostic splits on."""


CALIBRATED_SIGMA_MODEL = SigmaModel(
    coefficients={
        "kinematic": (
            -0.977780,  # bias
            +0.584974,  # log_sigma_raw
            -0.027221,  # log_inv_sin_turn
            -0.135674,  # log_min_speed
            +0.543864,  # log1p_disagreement
            +0.246272,  # is_contact
            -0.117085,  # log_fps_ratio
            -0.174601,  # is_far
            +0.493992,  # short_sides
            +0.954433,  # has_partner
            +0.161614,  # partner_log_gap
            +0.774286,  # log1p_offset
        ),
        "kinematic_court": (
            +0.029445,  # bias
            +0.745438,  # log_sigma_raw
            +0.016371,  # log_inv_sin_turn
            -0.116596,  # log_min_speed
            +0.287073,  # log1p_disagreement
            -0.216797,  # log_fps_ratio
            +0.188652,  # is_far
            +0.798122,  # short_sides
            -0.419551,  # has_partner
            +0.131903,  # partner_log_gap
            +0.325778,  # log1p_offset
        ),
        "dwell": (
            -0.152469,  # bias
            -0.052036,  # log_dwell_frames
            -0.239819,  # is_contact
            -0.285771,  # is_far
            -0.425076,  # turn_margin
            +0.705165,  # has_turn_margin
            -0.303835,  # log1p_speed
        ),
    },
    scale={
        "kinematic": 1.828502,
        "kinematic_court": 1.524324,
        "dwell": 1.250596,
    },
    combined_scale=1.455711,
    provenance="bench clean+track_error+event_timing+realistic, development matches, 5295 events",
)
"""The shipped width model, fitted by ``subframe_timing_benchmark calibrate``.

Fitted on the fifteen development bench matches of the four rungs and verified
once on the other fifteen (``WK3_REPORT.md``).  Coverage of the truth inside one
and two calibrated sigmas, per witness, held-out half:

===============  =====  ==========  ==========
witness          rows   +/-1 sigma  +/-2 sigma
===============  =====  ==========  ==========
kinematic        1798   0.876       0.956
kinematic_court   922   0.830       0.953
dwell            2893   0.797       0.955
===============  =====  ==========  ==========

One sigma over-covers because the normalised residual is not Gaussian: holding
95% inside two sigma needs a scale of 1.83 (``kinematic``) where 68% inside one
sigma alone would take 1.00, and the wider of the two is shipped.  A prior that
is too narrow in the tail is the failure ``docs/wk1/point_fit8.md`` measured, so
the tail wins.  ``streak`` and ``audio`` have no entry: the bench renders no
images and carries no audio.
"""
