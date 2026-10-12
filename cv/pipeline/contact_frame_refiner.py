"""Choose which image in the emitted-frame window a contact ray is anchored on.

``docs/wk1/contactpix.md`` measured the near-court contact tail and found it is
a *timing* problem, not a localization problem: the arc-augmented composed
track sits 44.17 px (p90) from the owner's click at the event model's emitted
frame, 27.04 px one image earlier, and 10.75 px on the best of the three images
-- but "the best of the three" is an oracle, because it picks the image using
the owner click.

This module is the label-free replacement for that oracle.  Every rule below
reads only automatic material -- the composed track, the tracked pose, a
motion-blur streak measured from the native images, and the audio contact
onset -- and returns one image out of the
``-2..+2`` window around the emitted frame, the composed-track pixel on it, a
per-contact uncertainty radius, and an abstain flag.

Nothing here imports owner truth.  The constants that were fitted are named in
``FITTED_ON_DEVELOPMENT`` and were chosen on the 38 development broadcasts of
``processed/wk3_cohort/cohort_root_v2`` only; the 8 held-out broadcasts were
scored once, at the end, and never used to choose anything.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

WINDOW = (-2, -1, 0, 1, 2)
"""Offsets from the emitted frame that a contact ray may be re-anchored onto."""

ORACLE_WINDOW = (-1, 0, 1)
"""The three-image window ``docs/wk1/contactpix.md`` measured its oracle over."""

FIT_SPAN = 5
"""How many track rows each side of the window the reversal fit may consume."""

FIT_GAP = 3
"""First offset outside the window a reversal fit is allowed to read.

The window itself is excluded from both fits: a frame that might be the contact
must not be allowed to bend the incoming or the outgoing line towards itself.
"""

MIN_FIT_ROWS = 3
"""Rows needed on a side before its line is trusted."""

WRIST_CONFIDENCE = 0.2
"""Pose keypoint confidence below which a wrist is not a witness."""

MAX_WRIST_PX = 400.0
"""A wrist further than this from the track is a different player, not a racket."""

FITTED_ON_DEVELOPMENT = (
    "TURN_GATE_DEG",
    "SIGMA_SCALE",
    "SIGMA_FLOOR_PX",
    "SIGMA_CAP_PX",
    "ABSTAIN_SIGMA_PX",
)
"""Constants whose values were swept on the 38 development broadcasts only."""

TURN_GATE_DEG = 45.0
"""Below this turn the composed track has no corner, so the emitted frame stands.

Swept on development: under 45 degrees the emitted frame is the better anchor
(median 2.98 / p90 11.26 px against the turn rule's 8.19 / 20.31 on the 20
weakest cases), and above it the turn rule wins by a growing margin (p90 46.40
to 9.41 px on the 170 sharpest).  The plateau runs 45-60 degrees; 45 is its
low edge, which is the setting that changes the emitted frame least often.
"""

SIGMA_SCALE = 0.75
SIGMA_FLOOR_PX = 10.0
SIGMA_CAP_PX = 120.0
"""``radius = min(cap, max(floor, scale * disagreement))``, swept on development.

The sweep asks for at least 90% coverage of the owner click *overall* and at
least 80% inside every disagreement band, then takes the smallest mean radius
that satisfies both.  A flat radius also reaches 90% overall but far less
inside the widest band, which is exactly the band a fitter must not trust.
"""

ABSTAIN_SIGMA_PX = SIGMA_CAP_PX
"""At or above this radius the refiner reports no usable contact anchor."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _xy(value: Any) -> tuple[float, float] | None:
    """Accept ``(x, y)``, ``{"x":..,"y":..}`` or a native-column mapping."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for x_key, y_key in (("x", "y"), ("x_native", "y_native"), ("image_x", "image_y")):
            if value.get(x_key) not in (None, "") and value.get(y_key) not in (None, ""):
                return float(value[x_key]), float(value[y_key])
        return None
    if hasattr(value, "xy"):
        pair = value.xy
        return float(pair[0]), float(pair[1])
    x, y = value
    return float(x), float(y)


def normalise_track(track_rows: Mapping[Any, Any]) -> dict[int, tuple[float, float]]:
    """Coerce a ``frame -> position`` mapping to ``int -> (x, y)`` native pixels."""

    output: dict[int, tuple[float, float]] = {}
    for frame, value in track_rows.items():
        point = _xy(value)
        if point is None or not all(math.isfinite(component) for component in point):
            continue
        output[int(frame)] = point
    return output


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fit_line(
    samples: Sequence[tuple[float, tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Least-squares ``p(t) = intercept + slope * t`` through ``(t, (x, y))``."""

    if len(samples) < 2:
        return None
    n = float(len(samples))
    mean_t = sum(t for t, _ in samples) / n
    variance = sum((t - mean_t) ** 2 for t, _ in samples)
    if variance <= 1e-9:
        return None
    intercept: list[float] = []
    slope: list[float] = []
    for axis in (0, 1):
        mean_v = sum(point[axis] for _, point in samples) / n
        covariance = sum((t - mean_t) * (point[axis] - mean_v) for t, point in samples)
        b = covariance / variance
        slope.append(b)
        intercept.append(mean_v - b * mean_t)
    return (intercept[0], intercept[1]), (slope[0], slope[1])


def _side_samples(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    *,
    direction: int,
    gap: int,
    span: int,
) -> list[tuple[float, tuple[float, float]]]:
    samples = []
    for step in range(gap, gap + span):
        frame = emitted_frame + direction * step
        point = track.get(frame)
        if point is not None:
            samples.append((float(direction * step), point))
    return samples


# --------------------------------------------------------------------------
# the five candidate signals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One rule's answer: an offset from the emitted frame, or nothing."""

    name: str
    offset: int | None
    evidence: dict[str, Any] = field(default_factory=dict)


def kinematic_reversal(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    *,
    window: Sequence[int] = WINDOW,
    gap: int = FIT_GAP,
    span: int = FIT_SPAN,
    min_rows: int = MIN_FIT_ROWS,
    by_pixel: bool = True,
) -> Signal:
    """Intersect the incoming and outgoing 2D lines and take the nearest image.

    The ball is on the strings for about 4-5 ms, far less than a native frame,
    so the trajectory has a corner rather than a curve at contact.  Fitting the
    approach and the departure separately -- from rows *outside* the window, so
    a candidate contact frame cannot bend its own evidence -- and asking where
    the two lines meet gives a corner position and a corner time without any
    label.  ``by_pixel`` takes the window image whose track pixel is nearest the
    corner; otherwise the window image nearest the corner *time* is taken.
    """

    incoming = _side_samples(track, emitted_frame, direction=-1, gap=gap, span=span)
    outgoing = _side_samples(track, emitted_frame, direction=+1, gap=gap, span=span)
    if len(incoming) < min_rows or len(outgoing) < min_rows:
        return Signal("kinematic_reversal", None, {"reason": "not_enough_rows"})
    first = _fit_line(incoming)
    second = _fit_line(outgoing)
    if first is None or second is None:
        return Signal("kinematic_reversal", None, {"reason": "degenerate_fit"})
    (ax, ay), (bx, by) = first
    (cx, cy), (dx, dy) = second
    # p_in(t) - p_out(t) is linear in t; the corner is where it is smallest.
    offset_x, offset_y = ax - cx, ay - cy
    slope_x, slope_y = bx - dx, by - dy
    denominator = slope_x * slope_x + slope_y * slope_y
    if denominator <= 1e-9:
        return Signal("kinematic_reversal", None, {"reason": "parallel_lines"})
    t_star = -(offset_x * slope_x + offset_y * slope_y) / denominator
    limit = float(max(window)) + 1.0
    t_star = max(-limit, min(limit, t_star))
    corner = (
        0.5 * ((ax + bx * t_star) + (cx + dx * t_star)),
        0.5 * ((ay + by * t_star) + (cy + dy * t_star)),
    )
    evidence = {
        "corner_x": corner[0],
        "corner_y": corner[1],
        "corner_offset": t_star,
        "incoming_rows": len(incoming),
        "outgoing_rows": len(outgoing),
    }
    if not by_pixel:
        offset = int(max(window[0], min(window[-1], round(t_star))))
        return Signal("kinematic_reversal", offset, evidence)
    best_offset, best_distance = None, float("inf")
    for offset in window:
        point = track.get(emitted_frame + offset)
        if point is None:
            continue
        distance = _distance(point, corner)
        if distance < best_distance:
            best_offset, best_distance = offset, distance
    if best_offset is None:
        return Signal("kinematic_reversal", None, {**evidence, "reason": "no_track_in_window"})
    evidence["corner_distance_px"] = best_distance
    return Signal("kinematic_reversal", best_offset, evidence)


def speed_minimum(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    *,
    window: Sequence[int] = WINDOW,
    centered: bool = True,
) -> Signal:
    """Take the window image with the smallest local speed.

    With ``centered`` the speed at image ``f`` is ``|p(f+1) - p(f-1)| / 2``,
    which collapses towards zero exactly at a reversal because the approach and
    the departure cancel.  Otherwise it is the mean of the two adjacent steps,
    which is the honest speed but a much flatter minimum.
    """

    scores: dict[int, float] = {}
    for offset in window:
        before = track.get(emitted_frame + offset - 1)
        after = track.get(emitted_frame + offset + 1)
        if before is None or after is None:
            continue
        if centered:
            scores[offset] = 0.5 * _distance(after, before)
            continue
        here = track.get(emitted_frame + offset)
        if here is None:
            continue
        scores[offset] = 0.5 * (_distance(here, before) + _distance(after, here))
    if not scores:
        return Signal("speed_minimum", None, {"reason": "no_local_speed"})
    offset = min(scores, key=lambda key: (scores[key], abs(key), key))
    return Signal("speed_minimum", offset, {"speed_px": scores, "min_speed_px": scores[offset]})


def direction_change(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    *,
    window: Sequence[int] = WINDOW,
) -> Signal:
    """Take the window image with the largest turn between its two steps."""

    scores: dict[int, float] = {}
    for offset in window:
        before = track.get(emitted_frame + offset - 1)
        here = track.get(emitted_frame + offset)
        after = track.get(emitted_frame + offset + 1)
        if before is None or here is None or after is None:
            continue
        ux, uy = here[0] - before[0], here[1] - before[1]
        vx, vy = after[0] - here[0], after[1] - here[1]
        first = math.hypot(ux, uy)
        second = math.hypot(vx, vy)
        if first < 1e-6 or second < 1e-6:
            continue
        cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / (first * second)))
        scores[offset] = math.degrees(math.acos(cosine))
    if not scores:
        return Signal("direction_change", None, {"reason": "no_turn"})
    offset = max(scores, key=lambda key: (scores[key], -abs(key), -key))
    return Signal("direction_change", offset, {"turn_deg": scores, "max_turn_deg": scores[offset]})


def _wrists(pose_rows: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    output = []
    for pose in pose_rows or ():
        joints = pose.get("joints") or {}
        for name in ("left_wrist", "right_wrist"):
            joint = joints.get(name)
            if joint is None:
                continue
            x, y = float(joint[0]), float(joint[1])
            confidence = float(joint[2]) if len(joint) > 2 else 1.0
            if confidence >= WRIST_CONFIDENCE:
                output.append((x, y))
    return output


def racket_proximity(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    pose_rows: Mapping[int, Sequence[Mapping[str, Any]]] | None,
    *,
    window: Sequence[int] = WINDOW,
    max_px: float = MAX_WRIST_PX,
) -> Signal:
    """Take the window image where the track sits nearest a wrist keypoint.

    The racket head is not tracked, so the racket-hand wrist is the closest
    automatic stand-in: the ball reaches its minimum distance to the striking
    hand at contact and moves away on both sides.
    """

    if not pose_rows:
        return Signal("racket_proximity", None, {"reason": "no_pose"})
    scores: dict[int, float] = {}
    for offset in window:
        frame = emitted_frame + offset
        point = track.get(frame)
        if point is None:
            continue
        wrists = _wrists(pose_rows.get(frame) or ())
        if not wrists:
            continue
        nearest = min(_distance(point, wrist) for wrist in wrists)
        if nearest <= max_px:
            scores[offset] = nearest
    if not scores:
        return Signal("racket_proximity", None, {"reason": "no_wrist_witness"})
    offset = min(scores, key=lambda key: (scores[key], abs(key), key))
    return Signal("racket_proximity", offset, {"wrist_px": scores, "min_wrist_px": scores[offset]})


def blur_streak(
    streak_px: Mapping[int, float] | None,
    emitted_frame: int,
    *,
    window: Sequence[int] = WINDOW,
) -> Signal:
    """Take the window image whose ball blob is the shortest streak.

    Exposure smears the ball along its path.  During the 4-5 ms the ball is on
    the strings it barely moves, so the image that contains the dwell carries
    the shortest streak.  ``streak_px`` is the measured major-axis length of the
    moving blob at the track position, keyed by absolute frame; the measurement
    itself lives in the benchmark because it needs the native images.
    """

    if not streak_px:
        return Signal("blur_streak", None, {"reason": "no_streak_measurement"})
    scores = {
        offset: float(streak_px[emitted_frame + offset])
        for offset in window
        if streak_px.get(emitted_frame + offset) is not None
    }
    if not scores:
        return Signal("blur_streak", None, {"reason": "no_streak_in_window"})
    offset = min(scores, key=lambda key: (scores[key], abs(key), key))
    return Signal("blur_streak", offset, {"streak_px": scores, "min_streak_px": scores[offset]})


def audio_onset_image_frame(onset_frame: float) -> int:
    """The native image that shows a physical contact at ``onset_frame``.

    The owner labels the leading-blur image and the frozen truth stores that
    image minus half a frame (``docs/wk1/HARNESS_LESSONS.md`` item 5), so image
    ``k`` carries physical time ``k - 0.5`` and the inverse is ``round(t + 0.5)``.
    """

    return int(math.floor(float(onset_frame) + 1.0))


def audio_reversal(
    audio_onset: float | None,
    emitted_frame: int,
    *,
    window: Sequence[int] = WINDOW,
) -> Signal:
    """Take the image the audio contact onset lands on, if it is in the window."""

    if audio_onset is None or not math.isfinite(float(audio_onset)):
        return Signal("audio", None, {"reason": "no_audio_onset"})
    frame = audio_onset_image_frame(audio_onset)
    offset = frame - emitted_frame
    if offset < window[0] or offset > window[-1]:
        return Signal("audio", None, {"reason": "outside_window", "audio_offset": offset})
    return Signal("audio", int(offset), {"audio_frame": float(audio_onset), "audio_image": frame})


SIGNAL_NAMES = (
    "kinematic_reversal",
    "speed_minimum",
    "direction_change",
    "racket_proximity",
    "blur_streak",
    "audio",
)


def all_signals(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    pose_rows: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    audio_onset: float | None = None,
    streak_px: Mapping[int, float] | None = None,
    *,
    window: Sequence[int] = WINDOW,
) -> dict[str, Signal]:
    """Run every rule independently on one contact window."""

    return {
        "kinematic_reversal": kinematic_reversal(track, emitted_frame, window=window),
        "speed_minimum": speed_minimum(track, emitted_frame, window=window),
        "direction_change": direction_change(track, emitted_frame, window=window),
        "racket_proximity": racket_proximity(track, emitted_frame, pose_rows, window=window),
        "blur_streak": blur_streak(streak_px, emitted_frame, window=window),
        "audio": audio_reversal(audio_onset, emitted_frame, window=window),
    }


# --------------------------------------------------------------------------
# the shipped combination
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "direction_change": 1.0,
    "speed_minimum": 0.0,
    "kinematic_reversal": 0.0,
    "audio": 0.0,
    "racket_proximity": 0.0,
    "blur_streak": 0.0,
}
"""Vote weights, chosen on development broadcasts only.

Every rule was measured alone and in combination on the 456 development
contacts.  ``direction_change`` won outright -- 2.42 / 11.68 px median / p90
against 3.88 / 31.81 for the next best rule -- and *every* combination that
could outvote it was worse, so it is the only rule with a vote.  The others are
still computed and reported: ``speed_minimum`` is the second opinion the
uncertainty radius is built from, and a zero weight is a measured result, not
an untested guess.
"""

PRIORITY = ("direction_change", "speed_minimum", "kinematic_reversal", "audio", "blur_streak")
"""Tie-break order among equally weighted offsets."""

SECOND_OPINION = ("direction_change", "speed_minimum")
"""The pair whose residual disagreement becomes the per-contact radius.

The second opinion is not the second most accurate chooser -- the audio onset
is -- but the one whose disagreement makes the tightest honest radius.  Swept on
development under the same constraint the radius itself is swept under (90%
coverage overall, 80% inside every disagreement band), the speed minimum needs a
12.25 px mean radius, the blur streak 15.49 px, the kinematic reversal 16.26 px
and the audio onset 18.16 px.  The speed minimum is also silent on none of the
456 development contacts, against the reversal's 8 and audio's 27, so it never
forces an abstention of its own.
"""


@dataclass(frozen=True)
class RefinedContact:
    """What the fitter receives instead of three candidate rays."""

    frame: int
    offset: int
    pixel: tuple[float, float] | None
    sigma_px: float
    abstain: bool
    reason: str
    votes: dict[str, int | None]
    disagreement_px: float | None
    emitted_frame: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vote(
    signals: Mapping[str, Signal],
    weights: Mapping[str, float],
    window: Sequence[int],
) -> tuple[int | None, dict[int, float]]:
    tally: dict[int, float] = {}
    for name, signal in signals.items():
        weight = float(weights.get(name, 0.0))
        if signal.offset is None or weight <= 0.0:
            continue
        tally[signal.offset] = tally.get(signal.offset, 0.0) + weight
    if not tally:
        return None, tally
    best = max(tally.values())
    tied = [offset for offset, value in tally.items() if value >= best - 1e-9]
    if len(tied) == 1:
        return tied[0], tally
    for name in PRIORITY:
        signal = signals.get(name)
        if signal is not None and signal.offset in tied:
            return int(signal.offset), tally
    return min(tied, key=lambda offset: (abs(offset), offset)), tally


def disagreement_px(
    track: Mapping[int, tuple[float, float]],
    emitted_frame: int,
    signals: Mapping[str, Signal],
    names: Sequence[str] = SECOND_OPINION,
    chosen_offset: int | None = None,
) -> float | None:
    """Pixel distance between the chosen image and the second opinion's image.

    This is the honest per-contact uncertainty: when the two independent
    kinematic readings of the same window land on the same image the anchor is
    tight, and when they land two images apart the ball has moved 60-120 native
    pixels between their answers and the anchor is not tight at all.  On
    development the owner-click error p90 is 9.42 px below 10 px of
    disagreement, 11.87 px from 10 to 30 px, 6.94 px from 30 to 60 px (22
    contacts) and 69.67 px above 60 px (5 contacts).  It is the last band the
    radius exists for; the middle bands are close to flat.
    """

    points = []
    for index, name in enumerate(names):
        if index == 0 and chosen_offset is not None:
            offset: int | None = chosen_offset
        else:
            signal = signals.get(name)
            offset = None if signal is None else signal.offset
        if offset is None:
            return None
        point = track.get(emitted_frame + offset)
        if point is None:
            return None
        points.append(point)
    if len(points) < 2:
        return None
    return _distance(points[0], points[1])


def sigma_from_disagreement(value: float | None) -> float:
    """Turn the top-two disagreement into a 90% radius (calibrated on development)."""

    if value is None:
        return SIGMA_CAP_PX
    return float(min(SIGMA_CAP_PX, max(SIGMA_FLOOR_PX, SIGMA_SCALE * value)))


def refine(
    track_rows: Mapping[Any, Any],
    emitted_frame: float | int,
    pose_rows: Mapping[Any, Sequence[Mapping[str, Any]]] | None = None,
    audio_onset: float | None = None,
    *,
    streak_px: Mapping[int, float] | None = None,
    weights: Mapping[str, float] | None = None,
    window: Sequence[int] = WINDOW,
    turn_gate_deg: float = TURN_GATE_DEG,
) -> RefinedContact:
    """Pick the image a contact ray should be anchored on, and say how sure.

    ``track_rows`` maps native frame index to the arc-augmented composed track
    position; ``emitted_frame`` is the event model's contact frame; ``pose_rows``
    maps frame index to tracked pose rows; ``audio_onset`` is the audio contact
    onset in frame units (``docs/wk1/s6_audio_timing.md``) or ``None``.  Neither
    pose nor audio changes the answer under the shipped weights -- both were
    measured and lost -- but both are computed and returned in ``votes``.

    Returns the chosen frame, the composed-track pixel on it, a per-contact
    uncertainty radius calibrated to cover the owner click at least 90% of the
    time, and an abstain flag for the contacts where no rule could answer.
    """

    emitted = int(round(float(emitted_frame)))
    track = normalise_track(track_rows)
    poses = {int(frame): rows for frame, rows in (pose_rows or {}).items()}
    streaks = {int(frame): float(value) for frame, value in (streak_px or {}).items()}
    signals = all_signals(track, emitted, poses, audio_onset, streaks, window=window)
    votes = {name: signal.offset for name, signal in signals.items()}
    tally_weights = DEFAULT_WEIGHTS if weights is None else weights
    offset, _ = _vote(signals, tally_weights, window)

    turn = signals["direction_change"].evidence.get("max_turn_deg")
    weak_turn = offset is None or turn is None or float(turn) < float(turn_gate_deg)
    if weak_turn:
        # The composed track has no corner in this window, so there is nothing
        # here to prefer over the event model's own frame.
        offset = 0
    reason = "weak_turn_keeps_emitted_frame" if weak_turn else "turn_selected_frame"

    pixel = track.get(emitted + offset)
    if pixel is None:
        fallback = track.get(emitted)
        if fallback is None:
            return RefinedContact(
                frame=emitted + offset,
                offset=offset,
                pixel=None,
                sigma_px=SIGMA_CAP_PX,
                abstain=True,
                reason="no_track_pixel",
                votes=votes,
                disagreement_px=None,
                emitted_frame=emitted,
            )
        offset, pixel, reason = 0, fallback, "chosen_image_has_no_track_row"

    gap = disagreement_px(track, emitted, signals, chosen_offset=offset)
    sigma = sigma_from_disagreement(gap)
    if sigma >= ABSTAIN_SIGMA_PX:
        return RefinedContact(
            frame=emitted + offset,
            offset=offset,
            pixel=pixel,
            sigma_px=sigma,
            abstain=True,
            reason="no_second_opinion" if gap is None else "uncertainty_beyond_gate",
            votes=votes,
            disagreement_px=gap,
            emitted_frame=emitted,
        )
    return RefinedContact(
        frame=emitted + offset,
        offset=offset,
        pixel=pixel,
        sigma_px=sigma,
        abstain=False,
        reason=reason,
        votes=votes,
        disagreement_px=gap,
        emitted_frame=emitted,
    )
