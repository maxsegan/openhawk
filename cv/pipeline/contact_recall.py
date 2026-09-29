"""Recover the contacts the event model never emits, using a second witness.

``docs/wk1/gate_audit.md`` read the 146 truth points as a waterfall and found
one cause dominates every other: **77 of the 146 points are lost first to a
missing contact emission**, and 134 of the 679 owner contacts have no automatic
emission within +-3 native frames on those points.  No acceptance gate, no
fitter change and no camera change can reach the owner's 20% complete-point
target while that holds.

``docs/wk1/contactframe.md`` measured a second, independent witness for the
same events that the event model does not use: the arc-augmented composed
track has a **sharp corner** at a contact, and the image with the largest
per-frame direction change sits within 11 px (p90) of the owner's click near
court.  That corner is produced by ``ball_motion_tracker`` from the detector
candidates and the per-frame court geometry; it never reads the event model, so
it is genuinely independent evidence about the same frame.

This module turns that corner into an *emission* rather than a re-anchoring.
The shape is the brief's unanimity rule applied to two machines instead of
three labellers: a corner alone is not enough to emit, and the model's own path
marginal alone has to clear a high calibrated bar (0.9898), but a corner and a
below-threshold marginal *together* are allowed to emit at a much lower bar.

Three arms are built, and each is scored separately in
``cv.validation.contact_recall_benchmark``:

``model``
    the frozen event model at its shipped ``bounded_false_positives``
    threshold.  Unchanged; this is the baseline.
``corner``
    corner proposals with no model evidence at all, typed by
    height-from-court-plane kinematics and gated on player proximity.
``combined``
    the model, plus the corner-supported rescue of rows the model decoded but
    abstained on below the shipped threshold.

Nothing here reads a label.  Every constant in ``FITTED_ON_DEVELOPMENT`` was
swept on the 32 development broadcasts of
``processed/wk3_nightly/cohort_root_v3``; the 8 held-out broadcasts were scored
once, at the end.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cv.pipeline.contact_frame_refiner import normalise_track

# --------------------------------------------------------------------------
# court constants
# --------------------------------------------------------------------------

NET_Y_M = 11.885
"""Court ``y`` of the net; the same value ``contact_frame_benchmark`` uses."""

COURT_LENGTH_M = 23.77
COURT_HALF_WIDTH_M = 5.485
"""Doubles half-width.  ``in_court`` uses this, not the singles line."""

# --------------------------------------------------------------------------
# constants swept on development only
# --------------------------------------------------------------------------

FITTED_ON_DEVELOPMENT = (
    "CORNER_MIN_DEG",
    "CORNER_MIN_STEP_PX",
    "CORNER_SUPPRESS_FRAMES",
    "PLAYER_MAX_PX",
    "CONTACT_MIN_HEIGHT_M",
    "CONTACT_MAX_HEIGHT_M",
    "CONTACT_MAX_MISS_M",
    "RESCUE_MARGINAL",
    "RESCUE_CORNER_FRAMES",
    "AUDIO_MIN_Z",
)
"""Constants whose values were chosen on the 32 development broadcasts."""

SHIPPED_MARGINAL = 0.9898053678698258
"""The event model's shipped ``bounded_false_positives`` operating point.

Calibrated in ``docs/wk1/events_pass2.md`` on 32 development broadcasts at a
2%-false-positive budget; it is the runtime default and this module does not
move it.
"""

CORNER_MIN_DEG = 110.0
"""Turn below which the composed track has no corner worth calling an event.

Swept on the 32 development broadcasts under the precision-first rule (below).
The looser gates recover more contacts and cost precision: 45 degrees reaches
30 complete development points at 0.9833 precision, 110 degrees reaches 21 at
0.9926, and only the second clears the frozen model's own 0.9923.
"""

CORNER_MIN_STEP_PX = 1.0
"""Both track steps either side of a corner must be at least this long.

A corner between two sub-pixel steps is filter noise, not a ball changing
direction.
"""

CORNER_SUPPRESS_FRAMES = 3
"""Non-maximum suppression radius, in native frames, over the turn profile.

A real corner spreads its turn over two or three images; without suppression
one event produces three proposals and the +-3 scorer counts two of them as
false positives.
"""

PLAYER_MAX_PX = 120.0
"""Native pixels from the ball to the nearest player box a contact may be.

Measured, not assumed: the 161 owner contacts the frozen model misses sit a
median 30 px and a p90 73 px from the nearest tracked player box, so 120 px is
already well outside the population this module exists to recover.
"""

CONTACT_MIN_HEIGHT_M = 0.35
CONTACT_MAX_HEIGHT_M = 3.40
"""The height band a ball ray must pass the player's ground column in.

``height_above_court`` intersects the ball's ray with the vertical line through
the nearest player's court position.  A racket contact happens between ankle
and full stretch; a ball at court level beside a player is the bounce the
harness lessons (item 17) name as the residual confusion, and it lands below
this band.
"""

CONTACT_MAX_MISS_M = 0.80
"""How far the ball ray may pass the player's ground column and still be a contact."""

RESCUE_MARGINAL = 0.90
"""Path marginal a model row needs when a corner also witnesses it.

The shipped threshold is 0.9898 on the model alone.  A second, independent
witness lowers what the model has to carry by itself, the way the brief's
unanimity rule lets one labeller be less certain when two others agree.

0.90 is the precision-first choice on development: it is the loosest bar on the
swept grid at which the combined arm still clears the frozen model's own
development precision.  ``docs/wk1`` and this package's report also carry a
recall-first setting (0.20 with a 45-degree corner gate) that reaches 30 of 118
development complete points at 0.9833 precision.
"""

RESCUE_CORNER_FRAMES = 1
"""How far a corner may sit from a model row and still witness it."""

AUDIO_MIN_Z = 5.0
"""Onset strength at which the audio counts as a witness.

The same ``MIN_PEAK_Z`` ``cv.pipeline.audio_contact_timing`` uses.
"""

CONTACT_TYPES = ("contact", "bounce")


# --------------------------------------------------------------------------
# the turn profile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Corner:
    """One local maximum of the composed track's per-frame direction change."""

    frame: int
    turn_deg: float
    step_in_px: float
    step_out_px: float
    pixel: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def turn_profile(track: Mapping[int, tuple[float, float]]) -> dict[int, tuple[float, float, float]]:
    """Per-frame ``(turn degrees, incoming step px, outgoing step px)``.

    This is the same quantity ``contact_frame_refiner.direction_change`` scores
    inside its five-image window, evaluated over the whole track instead so a
    corner can be *proposed* rather than only used to re-anchor an existing
    emission.
    """

    profile: dict[int, tuple[float, float, float]] = {}
    for frame in sorted(track):
        before = track.get(frame - 1)
        here = track.get(frame)
        after = track.get(frame + 1)
        if before is None or here is None or after is None:
            continue
        ux, uy = here[0] - before[0], here[1] - before[1]
        vx, vy = after[0] - here[0], after[1] - here[1]
        first = math.hypot(ux, uy)
        second = math.hypot(vx, vy)
        if first < 1e-9 or second < 1e-9:
            continue
        cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / (first * second)))
        profile[frame] = (math.degrees(math.acos(cosine)), first, second)
    return profile


def corners(
    track_rows: Mapping[Any, Any],
    *,
    min_deg: float = CORNER_MIN_DEG,
    min_step_px: float = CORNER_MIN_STEP_PX,
    suppress: int = CORNER_SUPPRESS_FRAMES,
) -> list[Corner]:
    """Local maxima of the turn profile that clear the corner gate."""

    track = normalise_track(track_rows)
    profile = turn_profile(track)
    kept: list[Corner] = []
    ordered = sorted(
        (
            frame
            for frame, (turn, first, second) in profile.items()
            if turn >= min_deg and min(first, second) >= min_step_px
        ),
        key=lambda frame: (-profile[frame][0], frame),
    )
    taken: list[int] = []
    for frame in ordered:
        if any(abs(frame - other) <= suppress for other in taken):
            continue
        taken.append(frame)
        turn, first, second = profile[frame]
        kept.append(
            Corner(
                frame=int(frame),
                turn_deg=float(turn),
                step_in_px=float(first),
                step_out_px=float(second),
                pixel=track[frame],
            )
        )
    return sorted(kept, key=lambda corner: corner.frame)


# --------------------------------------------------------------------------
# height from the court plane
# --------------------------------------------------------------------------


def court_point_at_height(
    projection: np.ndarray, pixel: Sequence[float], height_m: float
) -> tuple[float, float] | None:
    """Where the ray through ``pixel`` crosses the plane ``z = height_m``.

    ``projection`` is the per-frame 3x4 world-to-image matrix from
    ``camera_P_per_frame_v1.npz``.  Two of its three rows, eliminated against
    the third, are a 2x2 linear system in ``(x, y)`` once ``z`` is fixed.
    """

    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        return None
    u, v = float(pixel[0]), float(pixel[1])
    tail = matrix[:, 2] * float(height_m) + matrix[:, 3]
    left = np.asarray(
        [
            [matrix[0, 0] - u * matrix[2, 0], matrix[0, 1] - u * matrix[2, 1]],
            [matrix[1, 0] - v * matrix[2, 0], matrix[1, 1] - v * matrix[2, 1]],
        ],
        dtype=float,
    )
    right = np.asarray([u * tail[2] - tail[0], v * tail[2] - tail[1]], dtype=float)
    determinant = float(np.linalg.det(left))
    if not math.isfinite(determinant) or abs(determinant) < 1e-9:
        return None
    solution = np.linalg.solve(left, right)
    if not np.isfinite(solution).all():
        return None
    return float(solution[0]), float(solution[1])


def height_above_court(
    projection: np.ndarray,
    pixel: Sequence[float],
    ground_xy: Sequence[float],
    *,
    max_height_m: float = 5.0,
) -> tuple[float, float] | None:
    """Height at which the ball ray passes closest to a ground point's column.

    Returns ``(height_m, miss_m)``: the height of closest approach to the
    vertical line through ``ground_xy`` -- a tracked player's own court
    position -- and how far the ray still misses that line there.  This is the
    only height estimate available from one camera and one pixel, and it is
    exactly the quantity that separates a racket contact from the ball at court
    level beside a player.
    """

    low = court_point_at_height(projection, pixel, 0.0)
    high = court_point_at_height(projection, pixel, 1.0)
    if low is None or high is None:
        return None
    origin = np.asarray(low, dtype=float)
    direction = np.asarray(high, dtype=float) - origin
    norm = float(direction @ direction)
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    target = np.asarray([float(ground_xy[0]), float(ground_xy[1])], dtype=float)
    height = float((target - origin) @ direction / norm)
    height = max(0.0, min(float(max_height_m), height))
    miss = float(np.linalg.norm(origin + height * direction - target))
    return height, miss


def in_court(point: Sequence[float] | None, *, margin_m: float = 1.0) -> bool:
    """Whether a court-plane point is inside the doubles rectangle plus a margin."""

    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    return abs(x) <= COURT_HALF_WIDTH_M + margin_m and -margin_m <= y <= COURT_LENGTH_M + margin_m


def court_side(projection: np.ndarray | None, pixel: Sequence[float]) -> str:
    """Near or far half, from the ball's own court-plane projection."""

    if projection is None:
        return "unknown"
    ground = court_point_at_height(projection, pixel, 0.0)
    if ground is None:
        return "unknown"
    return "near" if ground[1] < NET_Y_M else "far"


# --------------------------------------------------------------------------
# player proximity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerWitness:
    """The nearest tracked player to a pixel, and what geometry it supports."""

    distance_px: float
    wrist_px: float | None
    side: str | None
    court_xy: tuple[float, float] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _box_distance(box: Mapping[str, Any], pixel: Sequence[float]) -> float:
    x0, y0 = float(box["x0"]), float(box["y0"])
    x1, y1 = float(box["x1"]), float(box["y1"])
    dx = max(x0 - pixel[0], 0.0, pixel[0] - x1)
    dy = max(y0 - pixel[1], 0.0, pixel[1] - y1)
    return math.hypot(dx, dy)


def nearest_player(
    player_rows: Iterable[Mapping[str, Any]], pixel: Sequence[float]
) -> PlayerWitness | None:
    """Nearest player box to ``pixel``, in native pixels.

    Each row carries native ``x0/y0/x1/y1``, optionally a ``wrist`` list of
    native wrist positions, a ``side`` and the tracker's own ``court_xy``.
    """

    best: PlayerWitness | None = None
    for row in player_rows or ():
        try:
            distance = _box_distance(row, pixel)
        except (KeyError, TypeError, ValueError):
            continue
        if best is not None and distance >= best.distance_px:
            continue
        wrists = [
            math.hypot(float(point[0]) - pixel[0], float(point[1]) - pixel[1])
            for point in (row.get("wrists") or ())
            if point is not None
        ]
        court = row.get("court_xy")
        best = PlayerWitness(
            distance_px=float(distance),
            wrist_px=min(wrists) if wrists else None,
            side=str(row["side"]) if row.get("side") else None,
            court_xy=(float(court[0]), float(court[1])) if court is not None else None,
        )
    return best


# --------------------------------------------------------------------------
# typing a corner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedCorner:
    """A corner with a physical type and the evidence that produced it."""

    corner: Corner
    event_type: str | None
    reason: str
    player: PlayerWitness | None
    height_m: float | None
    miss_m: float | None
    ground_xy: tuple[float, float] | None
    side: str
    audio_z: float | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corner"] = self.corner.as_dict()
        return payload


def type_corner(
    corner: Corner,
    *,
    projection: np.ndarray | None,
    player_rows: Iterable[Mapping[str, Any]] | None,
    audio_z: float | None = None,
    player_max_px: float = PLAYER_MAX_PX,
    min_height_m: float = CONTACT_MIN_HEIGHT_M,
    max_height_m: float = CONTACT_MAX_HEIGHT_M,
    max_miss_m: float = CONTACT_MAX_MISS_M,
) -> TypedCorner:
    """Call one corner a contact, a bounce, or neither.

    The decision is height-from-court-plane kinematics with the audio onset as
    a tie-break: a corner is a **contact** when the ball's ray passes a tracked
    player's own ground column between ankle and stretch height, and a
    **bounce** when the same corner's court-plane projection is inside the
    court and no player column explains it.  Everything else is refused.
    """

    player = nearest_player(player_rows or (), corner.pixel)
    ground = (
        court_point_at_height(projection, corner.pixel, 0.0) if projection is not None else None
    )
    side = court_side(projection, corner.pixel) if projection is not None else "unknown"
    height: float | None = None
    miss: float | None = None
    if projection is not None and player is not None and player.court_xy is not None:
        solved = height_above_court(projection, corner.pixel, player.court_xy)
        if solved is not None:
            height, miss = solved

    near_player = player is not None and player.distance_px <= player_max_px
    if near_player and height is not None and miss is not None:
        if min_height_m <= height <= max_height_m and miss <= max_miss_m:
            return TypedCorner(
                corner=corner,
                event_type="contact",
                reason="ray_passes_player_column_at_racket_height",
                player=player,
                height_m=height,
                miss_m=miss,
                ground_xy=ground,
                side=side,
                audio_z=audio_z,
            )
        if height < min_height_m and in_court(ground):
            return TypedCorner(
                corner=corner,
                event_type="bounce",
                reason="ball_at_court_level_beside_a_player",
                player=player,
                height_m=height,
                miss_m=miss,
                ground_xy=ground,
                side=side,
                audio_z=audio_z,
            )
    if not near_player and in_court(ground):
        return TypedCorner(
            corner=corner,
            event_type="bounce",
            reason="court_plane_corner_with_no_player",
            player=player,
            height_m=height,
            miss_m=miss,
            ground_xy=ground,
            side=side,
            audio_z=audio_z,
        )
    return TypedCorner(
        corner=corner,
        event_type=None,
        reason="no_geometry_supports_a_type" if projection is not None else "no_camera",
        player=player,
        height_m=height,
        miss_m=miss,
        ground_xy=ground,
        side=side,
        audio_z=audio_z,
    )


# --------------------------------------------------------------------------
# the emission
# --------------------------------------------------------------------------


@dataclass
class Emission:
    """One augmented emission and the witnesses that produced it."""

    frame: float
    event_type: str
    confidence: float
    witnesses: list[str] = field(default_factory=list)
    source: str = "model"
    path_marginal: float | None = None
    turn_deg: float | None = None
    height_m: float | None = None
    audio_z: float | None = None
    pixel: tuple[float, float] | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_confidence(marginal: float, corner_support: bool) -> float:
    """Confidence on the emitted row.

    A row the model already clears on its own keeps its marginal.  A rescued
    row cannot claim the model's calibrated confidence, so it is reported as
    the geometric mean of what the model did say and the shipped bar it did not
    reach -- a number that is always below the shipped threshold and above the
    rescue bar, so a downstream consumer can rank rescues below native rows.
    """

    if not corner_support:
        return float(marginal)
    return float(math.sqrt(max(0.0, min(1.0, marginal)) * SHIPPED_MARGINAL))


def emit(
    track_rows: Mapping[Any, Any],
    model_paths: Iterable[Mapping[str, Any]],
    player_rows: Mapping[Any, Sequence[Mapping[str, Any]]] | None = None,
    audio_onsets: Mapping[Any, float] | None = None,
    *,
    projections: Mapping[Any, Any] | None = None,
    marginal_threshold: float = SHIPPED_MARGINAL,
    rescue_marginal: float = RESCUE_MARGINAL,
    rescue_corner_frames: int = RESCUE_CORNER_FRAMES,
    corner_min_deg: float = CORNER_MIN_DEG,
    corner_min_step_px: float = CORNER_MIN_STEP_PX,
    corner_suppress: int = CORNER_SUPPRESS_FRAMES,
    include_corner_only: bool = False,
    corner_type_must_match: bool = True,
    require_audio_for_rescue: bool = False,
    audio_min_z: float = AUDIO_MIN_Z,
    **typing_constants: float,
) -> list[Emission]:
    """Augment one point's event emissions with the composed track's corners.

    ``track_rows`` maps native frame index to the arc-augmented composed track
    position (the same input ``contact_frame_refiner.refine`` takes).
    ``model_paths`` is the event model's decoded path for the point, **including
    the rows it abstained on** -- each row needs ``frame``, ``event_type`` and
    ``path_marginal``.  ``player_rows`` maps native frame index to the tracked
    player boxes on that frame, and ``audio_onsets`` maps native frame index to
    an onset strength.  ``projections`` maps native frame index to the 3x4
    camera matrix; without it no corner can be typed and only the model's own
    rows are emitted.

    Returns the augmented emission list.  Every row carries a confidence and
    the list of witnesses that produced it, so a consumer can rank a
    two-witness rescue below a row the model cleared alone.
    """

    track = normalise_track(track_rows)
    players = {int(frame): list(rows) for frame, rows in (player_rows or {}).items()}
    onsets = {int(frame): float(value) for frame, value in (audio_onsets or {}).items()}
    cameras = {int(frame): value for frame, value in (projections or {}).items()}

    typed: list[TypedCorner] = []
    for corner in corners(
        track,
        min_deg=corner_min_deg,
        min_step_px=corner_min_step_px,
        suppress=corner_suppress,
    ):
        typed.append(
            type_corner(
                corner,
                projection=cameras.get(corner.frame),
                player_rows=players.get(corner.frame),
                audio_z=onsets.get(corner.frame),
                **typing_constants,
            )
        )

    output: list[Emission] = []
    claimed: set[int] = set()
    for row in model_paths or ():
        event_type = str(row.get("event_type"))
        if event_type not in CONTACT_TYPES and event_type != "net_hit":
            continue
        frame = float(row["frame"])
        marginal = float(row.get("path_marginal") or 0.0)
        support = [
            item
            for item in typed
            if abs(item.corner.frame - frame) <= rescue_corner_frames
            and (item.event_type == event_type or not corner_type_must_match)
        ]
        witness = min(support, key=lambda item: abs(item.corner.frame - frame), default=None)
        audio = onsets.get(int(round(frame)))
        heard = audio is not None and audio >= audio_min_z
        if marginal >= marginal_threshold:
            witnesses = ["event_model"] + (["track_corner"] if witness else [])
            reason = "model_above_shipped_threshold"
        elif (
            witness is not None
            and marginal >= rescue_marginal
            and (heard or not require_audio_for_rescue)
        ):
            witnesses = ["event_model", "track_corner"]
            reason = "corner_rescued_below_threshold"
        else:
            continue
        if witness is not None:
            claimed.add(witness.corner.frame)
        if heard:
            witnesses.append("audio_onset")
        output.append(
            Emission(
                frame=frame,
                event_type=event_type,
                confidence=_model_confidence(marginal, reason.startswith("corner")),
                witnesses=witnesses,
                source="model" if reason.startswith("model") else "model+corner",
                path_marginal=marginal,
                turn_deg=witness.corner.turn_deg if witness else None,
                height_m=witness.height_m if witness else None,
                audio_z=audio,
                pixel=track.get(int(round(frame))),
                reason=reason,
            )
        )

    if include_corner_only:
        for item in typed:
            if item.event_type is None or item.corner.frame in claimed:
                continue
            if any(
                abs(item.corner.frame - float(row.frame)) <= rescue_corner_frames
                and row.event_type == item.event_type
                for row in output
            ):
                continue
            audio = onsets.get(item.corner.frame)
            witnesses = ["track_corner"]
            if audio is not None and audio >= audio_min_z:
                witnesses.append("audio_onset")
            output.append(
                Emission(
                    frame=float(item.corner.frame),
                    event_type=item.event_type,
                    confidence=min(0.99, item.corner.turn_deg / 180.0),
                    witnesses=witnesses,
                    source="corner",
                    path_marginal=None,
                    turn_deg=item.corner.turn_deg,
                    height_m=item.height_m,
                    audio_z=audio,
                    pixel=item.corner.pixel,
                    reason=item.reason,
                )
            )

    return sorted(output, key=lambda row: (row.frame, row.event_type))
