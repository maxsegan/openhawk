"""Primary-ball ownership over IMM tracker segments (source-only, default off).

The motion tracker restarts its filter whenever both innovation gates reject the coarse
guide, and again when it bootstraps after a miss reset.  Each restart used to assert
continuity for free: Mahalanobis zero, ``track_id`` zero, whatever object the guide had
jumped to inherited the primary identity.  This module decides ownership *after* a clip
has been tracked, from the emitted samples alone:

* every restart opens a new **segment** (the tracker tags rows with the restart kind);
* consecutive segments are **joined** when constant-velocity wings fitted to each side,
  transported into one camera frame and extrapolated across the gap, pass through a
  common point at some instant inside the gap.  Velocity is never compared, so a racket
  contact or bounce joins while an unrelated jump does not;
* a moving wing never joins a stationary one, which is the same-place-wrong-object guard;
* joined segments form a **chain**, the unit of identity this module can actually stand
  behind.  ``track_id`` carries the chain id, so an identity break is visible downstream.

What the stage consumes.  The mechanism is deliberately narrow: it suppresses chains that
are *proven static* under a reliable camera (baseline marks, standing persons, shoes) and
keeps everything else, because motion alone does not establish that a moving chain is the
ball.  The chain continuous with the clip's first moving chain is labelled ``primary``;
every other moving chain is an explicit ``candidate`` (an unassigned moving object that
the source evidence available here can neither confirm nor reject).  Chains whose camera
is unreliable throughout are ``ambiguous`` and are kept: uncertainty is not proof of a
static object.  Only ``stationary`` chains are withheld from the consumed track.  A moving
unrelated object after a true image exit therefore remains a candidate chain; the existing
``qualified_image_reentry`` option is the source-backed mechanism for that case and composes
with this one.

Nothing is fabricated: no sample is moved, no gap is filled and no physical end is
inferred.  Every tracked row is preserved with its owner and join verdict in a side artifact.

Camera handling.  A broadcast camera pans and zooms about a fixed point, so the ground
homography warp between two frames is the same rotation homography for every scene plane.
It is therefore valid for an airborne ball, unlike projecting the ball onto the court.
Where the per-frame homography is flagged unreliable it is a held stale copy and is never
used.  Inside an unreliable stretch bracketed by reliable frames the test falls back to raw
image coordinates with a pan *allowance* (twice the stretch's mean shift, not a guaranteed
bound); an unbracketed stretch has no measurable pan and yields only ambiguous verdicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cv.pipeline import ball_motion_tracker as tracker

# Private row keys the tracker adds when ownership is enabled; stripped before writing.
SEGMENT_KEY = "_segment_id"
RESTART_KEY = "_restart_kind"

OWNERSHIP_COLUMNS = (
    "segment_id",
    "chain_id",
    "restart_kind",
    "owner",
    "chain_motion",
    "join_outcome",
    "join_target_segment",
    "join_gap_frames",
    "join_camera_mode",
    "join_distance_native",
    "join_tolerance_native",
    "join_mahalanobis_d2",
)


def _frame(row: dict) -> int:
    return tracker.frame_number(str(row["frame"]))


def _native_xy(row: dict) -> np.ndarray:
    # These are track_clip's internal legacy mirror rows, before artifact export.
    # Name that coordinate contract explicitly rather than embedding a scale.
    return tracker.res.scale_points(
        (float(row["x"]), float(row["y"])),
        tracker.res.LEGACY_TRACKING_SIZE,
        tracker.res.NATIVE_SIZE,
    )


def _row_observation(row: dict) -> tracker.Observation:
    tokens = tuple(
        token
        for token in str(row.get("sources", "")).split("+")
        if token and not token.startswith("provenance:") and not token.startswith("crop_region:")
    )
    x_native, y_native = _native_xy(row)
    return tracker.Observation(
        float(x_native),
        float(y_native),
        float(row.get("score", 0.0) or 0.0),
        int(row.get("rank", 0) or 0),
        tokens,
    )


def _image_shift(h_from: np.ndarray, h_to: np.ndarray) -> float:
    centre = np.asarray([tracker.res.NATIVE_SIZE.width / 2.0, tracker.res.NATIVE_SIZE.height / 2.0])
    moved = tracker._apply_h(tracker.camera_warp(h_from, h_to), centre)
    return float(np.linalg.norm(moved - centre))


@dataclass(frozen=True)
class PanModel:
    """Per-frame bound on camera image motion, from the source's own homographies."""

    reliable_bound: float
    unreliable_runs: dict[str, list[tuple[int, int, float]]] = field(default_factory=dict)

    def bound(self, clip: str, frame: int) -> float:
        for start, end, rate in self.unreliable_runs.get(clip, ()):
            if start <= frame <= end:
                return rate
        return self.reliable_bound

    def allowance(self, clip: str, frame_a: int, frame_b: int) -> float:
        """Upper bound on image motion of a static object between two frames."""
        low, high = sorted((frame_a, frame_b))
        return sum(self.bound(clip, frame) for frame in range(low + 1, high + 1))


def pan_model(geometry: tracker.Geometry) -> PanModel:
    """Measure the camera from its own homographies.

    Reliable consecutive frames give the ordinary per-frame shift bound. An unreliable run
    bracketed by reliable frames gets an *allowance* of twice its mean endpoint shift per
    frame: an object counts as moving there only when it clearly outruns the camera. This
    is not a guaranteed bound (an out-and-back pan cancels at the endpoints), which is why
    raw-coordinate verdicts never claim stillness. An unbracketed run has no measurable pan
    at all and receives an infinite allowance, so every raw verdict inside it is ambiguous.
    """
    by_clip: dict[str, list[int]] = {}
    for clip, frame in geometry.homographies:
        by_clip.setdefault(clip, []).append(frame)
    reliable_shifts = []
    runs: dict[str, list[tuple[int, int, float]]] = {}
    for clip, frames in by_clip.items():
        frames.sort()
        run_start = None
        for index, frame in enumerate(frames):
            reliable = geometry.is_reliable((clip, frame))
            previous = frames[index - 1] if index else None
            if (
                reliable
                and previous is not None
                and previous == frame - 1
                and geometry.is_reliable((clip, previous))
            ):
                reliable_shifts.append(
                    _image_shift(
                        geometry.homographies[(clip, previous)],
                        geometry.homographies[(clip, frame)],
                    )
                )
            if not reliable and run_start is None:
                run_start = frame
            if reliable and run_start is not None:
                runs.setdefault(clip, []).append((run_start, previous, frame))
                run_start = None
        if run_start is not None:
            runs.setdefault(clip, []).append((run_start, frames[-1], None))
    reliable_bound = max(reliable_shifts) if reliable_shifts else 0.0
    bounded: dict[str, list[tuple[int, int, float]]] = {}
    for clip, clip_runs in runs.items():
        for start, end, after in clip_runs:
            before = start - 1
            rate = math.inf
            if after is not None and geometry.is_reliable((clip, before)):
                span = after - before
                rate = max(
                    reliable_bound,
                    2.0
                    * _image_shift(
                        geometry.homographies[(clip, before)],
                        geometry.homographies[(clip, after)],
                    )
                    / span,
                )
            bounded.setdefault(clip, []).append((start, end, rate))
    return PanModel(reliable_bound, bounded)


def transport(
    geometry: tracker.Geometry, clip: str, frame_from: int, frame_to: int, xy: np.ndarray
) -> np.ndarray | None:
    """Move an image point between two exposures; None when either camera is unreliable."""
    if frame_from == frame_to:
        return np.asarray(xy, dtype=float)
    source = (clip, frame_from)
    target = (clip, frame_to)
    if not (geometry.is_reliable(source) and geometry.is_reliable(target)):
        return None
    return tracker._apply_h(
        tracker.camera_warp(geometry.homographies[source], geometry.homographies[target]), xy
    )


@dataclass(frozen=True)
class MotionState:
    state: str  # moving | still | ambiguous | insufficient
    extent_native: float
    camera_mode: str


def motion_state(
    rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    pan: PanModel,
    config: tracker.MotionConfig,
) -> MotionState:
    """Does this short sample window move by more than the tracker's own cluster radius?"""
    if len(rows) < 2:
        return MotionState("insufficient", 0.0, "none")
    frames = [_frame(row) for row in rows]
    points = [_native_xy(row) for row in rows]
    reference = frames[0]
    transported = [
        transport(geometry, clip, frame, reference, point)
        for frame, point in zip(frames, points, strict=True)
    ]
    compensated = [point for point in transported if point is not None]
    radius = config.cluster_radius_native
    if len(compensated) >= 2:
        extent = max(float(np.linalg.norm(point - compensated[0])) for point in compensated[1:])
        if extent >= radius:
            return MotionState("moving", extent, "compensated")
        if len(compensated) == len(rows):
            return MotionState("still", extent, "compensated")
    raw_extent = max(float(np.linalg.norm(point - points[0])) for point in points[1:])
    allowance = pan.allowance(clip, frames[0], frames[-1])
    if raw_extent >= 2.0 * allowance + radius:
        return MotionState("moving", raw_extent, "raw")
    return MotionState("ambiguous", raw_extent, "raw")


def _process_position_variance(horizon: int, acceleration_std: float) -> float:
    """Position variance the tracker's own ballistic process noise accumulates."""
    transition = np.asarray(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    covariance = np.zeros((4, 4))
    noise = tracker._process_noise(acceleration_std)
    for _ in range(max(0, horizon)):
        covariance = transition @ covariance @ transition.T + noise
    return float(covariance[0, 0])


@dataclass
class WingFit:
    """Constant velocity plus known image gravity, least squares over a short wing."""

    t_ref: float
    origin: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray
    sigma2: float
    times: np.ndarray
    residual_rms: float
    acceleration_noise: float

    def predict(self, t: float) -> tuple[np.ndarray, float]:
        """Extrapolated position and its per-axis variance.

        The variance is the regression prediction variance (measurement sigma, or the
        fitted residual when the wing disagrees with a line by more than that, since
        consecutive guide samples are not independent proof) plus the tracker's own
        ballistic process noise accumulated over the frames beyond the wing.
        """
        dt = t - self.t_ref
        position = self.origin + self.velocity * dt + 0.5 * self.gravity * dt * dt
        mean_t = float(self.times.mean())
        spread = float(((self.times - mean_t) ** 2).sum())
        leverage = 1.0 / len(self.times) + ((t - mean_t) ** 2 / spread if spread > 0 else 0.0)
        low, high = float(self.times.min()), float(self.times.max())
        beyond = 0.0 if low <= t <= high else min(abs(t - low), abs(t - high))
        sigma2 = max(self.sigma2, self.residual_rms**2)
        return position, sigma2 * leverage + _process_position_variance(
            int(math.ceil(round(beyond, 6))), self.acceleration_noise
        )


def fit_wing(
    times: list[float],
    points: list[np.ndarray],
    t_ref: float,
    gravity: np.ndarray,
    sigma2: float,
    acceleration_noise: float,
) -> WingFit:
    times_array = np.asarray(times, dtype=float)
    dts = times_array - t_ref
    corrected = np.stack(points) - 0.5 * np.outer(dts * dts, gravity)
    design = np.stack([np.ones_like(dts), dts], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, corrected, rcond=None)
    residual = corrected - design @ coefficients
    # Per-axis residual scale with the two fitted parameters removed; zero for two samples.
    dof = max(1, len(times_array) - 2)
    rms = float(np.sqrt((residual**2).sum() / (2.0 * dof))) if len(times_array) > 2 else 0.0
    return WingFit(
        t_ref,
        coefficients[0],
        coefficients[1],
        gravity,
        sigma2,
        times_array,
        rms,
        acceleration_noise,
    )


def _fit_side(
    frames: list[int],
    points: list[np.ndarray],
    t_ref: float,
    gravity: np.ndarray,
    sigma2: float,
    acceleration_noise: float,
    *,
    tail: bool,
) -> WingFit:
    """Fit the wing; if it disagrees with one line by more than the measurement sigma
    (a bounce or contact inside the wing), keep only the two samples nearest the gap."""
    fit = fit_wing(frames, points, t_ref, gravity, sigma2, acceleration_noise)
    if len(frames) > 2 and fit.residual_rms > math.sqrt(sigma2):
        keep = slice(-2, None) if tail else slice(0, 2)
        fit = fit_wing(frames[keep], points[keep], t_ref, gravity, sigma2, acceleration_noise)
    return fit


@dataclass(frozen=True)
class JoinResult:
    outcome: str  # joined | unjoined | ambiguous | post_wing_stationary | insufficient_wing
    gap_frames: int
    camera_mode: str
    distance_native: float
    tolerance_native: float
    mahalanobis_d2: float
    join_time: float
    pre_motion: str
    post_motion: str


def _wing_sigma2(rows: list[dict], config: tracker.MotionConfig) -> float:
    return float(
        np.mean(
            [tracker.measurement_covariance(_row_observation(row), config)[0, 0] for row in rows]
        )
    )


def join_test(
    pre_rows: list[dict],
    post_rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
    pan: PanModel,
) -> JoinResult:
    """Do the tail of one segment and the head of the next pass through one point?"""
    count = config.reentry_support_frames
    pre = pre_rows[-count:]
    post = post_rows[:count]
    pre_frames = [_frame(row) for row in pre]
    post_frames = [_frame(row) for row in post]
    gap = post_frames[0] - pre_frames[-1]
    pre_state = motion_state(pre, geometry, clip, pan, config)
    post_state = motion_state(post, geometry, clip, pan, config)
    if len(pre) < 2 or len(post) < 2:
        return JoinResult(
            "insufficient_wing",
            gap,
            "none",
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            pre_state.state,
            post_state.state,
        )
    if pre_state.state == "moving" and post_state.state == "still":
        return JoinResult(
            "post_wing_stationary",
            gap,
            "none",
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            pre_state.state,
            post_state.state,
        )
    reference = post_frames[0]
    pre_points = [
        transport(geometry, clip, frame, reference, _native_xy(row))
        for frame, row in zip(pre_frames, pre, strict=True)
    ]
    post_points = [
        transport(geometry, clip, frame, reference, _native_xy(row))
        for frame, row in zip(post_frames, post, strict=True)
    ]
    camera_mode = "compensated"
    pan_allowance = 0.0
    if any(point is None for point in [*pre_points, *post_points]):
        if gap > config.ownership_raw_join_gap_frames:
            return JoinResult(
                "ambiguous",
                gap,
                "raw",
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                pre_state.state,
                post_state.state,
            )
        camera_mode = "raw"
        pre_points = [_native_xy(row) for row in pre]
        post_points = [_native_xy(row) for row in post]
        pan_allowance = pan.allowance(clip, pre_frames[-1], post_frames[0])
    reference_key = (clip, reference)
    gravity = tracker.projected_gravity(
        post_points[0],
        geometry.homographies.get(reference_key) if geometry.is_reliable(reference_key) else None,
        geometry.projections.get(reference_key) if geometry.is_reliable(reference_key) else None,
        fps,
    )
    if not math.isfinite(pan_allowance):
        # Unbracketed unreliable camera: no measurable pan, so raw distances mean nothing.
        return JoinResult(
            "ambiguous",
            gap,
            "raw",
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            pre_state.state,
            post_state.state,
        )
    noise = config.ballistic_acceleration_noise
    pre_fit = _fit_side(
        pre_frames, pre_points, pre_frames[-1], gravity, _wing_sigma2(pre, config), noise, tail=True
    )
    post_fit = _fit_side(
        post_frames,
        post_points,
        post_frames[0],
        gravity,
        _wing_sigma2(post, config),
        noise,
        tail=False,
    )
    best = None
    for t in np.arange(pre_frames[-1] - 1.0, post_frames[0] + 1.0 + 1e-9, 0.1):
        pre_xy, pre_var = pre_fit.predict(t)
        post_xy, post_var = post_fit.predict(t)
        distance = float(np.linalg.norm(pre_xy - post_xy))
        variance = pre_var + post_var
        effective = max(0.0, distance - pan_allowance)
        d2 = effective * effective / variance
        # Tolerance: the tracker's impulse gate on the summed extrapolation variance, never
        # below its cluster radius (consecutive guide samples are correlated, so the
        # regression variance alone is optimistic), plus the pan allowance in raw mode.
        radius = (
            max(math.sqrt(config.impulse_gate_d2 * variance), config.cluster_radius_native)
            + pan_allowance
        )
        # Best instant: the closest pass relative to its tolerance; among equal passes the
        # tightest tolerance (a zero-distance pass must not pick the loosest extrapolation).
        key = (round(effective / (radius - pan_allowance), 6), radius)
        if best is None or key < best[4]:
            best = (d2, distance, radius, t, key)
    d2, distance, tolerance, t, _ = best
    if tolerance > config.hotspot_motion_step_native:
        # The wings cannot localise the object to within one motion step at any instant of
        # the gap; a pass inside such a radius is not evidence of identity either way.
        outcome = "ambiguous"
    else:
        outcome = "joined" if distance <= tolerance else "unjoined"
    return JoinResult(
        outcome,
        gap,
        camera_mode,
        distance,
        tolerance,
        d2,
        float(t),
        pre_state.state,
        post_state.state,
    )


def chain_motion(
    rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    pan: PanModel,
    config: tracker.MotionConfig,
) -> str:
    """``still`` only when no short window of the chain ever moved under a reliable camera.

    A chain is one object, so a ball that flies and then rolls to rest is a moving chain;
    only an object that never moved by the tracker's own cluster radius in any window is
    static.  A chain seen only under an unreliable camera is ambiguous, not static.  The
    window is the tracker's reentry sample count.
    """
    if len(rows) < 2:
        return "insufficient"
    count = max(2, config.reentry_support_frames)
    states = {
        motion_state(rows[start : start + count], geometry, clip, pan, config).state
        for start in range(0, max(1, len(rows) - count + 1))
    }
    if "moving" in states:
        return "moving"
    if "still" in states:
        return "still"
    return "ambiguous"


def _segments(rows: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current_id = None
    for row in rows:
        if row[SEGMENT_KEY] != current_id:
            segments.append([])
            current_id = row[SEGMENT_KEY]
        segments[-1].append(row)
    return segments


def assign_ownership(
    rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    fps: float,
    config: tracker.MotionConfig,
    pan: PanModel | None = None,
) -> tuple[list[dict], dict]:
    """Return ownership rows (every input row, annotated) and a per-clip summary."""
    if pan is None:
        pan = pan_model(geometry)
    segments = _segments(rows)
    chain_of_segment: dict[int, int] = {}
    joins: dict[int, JoinResult | None] = {}
    join_targets: dict[int, int | None] = {}
    next_chain = 0
    for index, segment in enumerate(segments):
        segment_id = segment[0][SEGMENT_KEY]
        if index == 0:
            chain_of_segment[segment_id] = next_chain
            next_chain += 1
            joins[segment_id] = None
            join_targets[segment_id] = None
            continue
        result = None
        target = None
        # Immediate predecessor first, then earlier segments inside the join horizon, so a
        # ball can rejoin its own chain across a short wrong-object interlude. The recorded
        # result is the first join, else the immediate predecessor's verdict.
        for candidate in reversed(segments[:index]):
            gap = _frame(segment[0]) - _frame(candidate[-1])
            if gap > config.ownership_join_gap_frames:
                break
            attempt = join_test(candidate, segment, geometry, clip, fps, config, pan)
            if result is None:
                result, target = attempt, candidate[0][SEGMENT_KEY]
            if attempt.outcome == "joined":
                result, target = attempt, candidate[0][SEGMENT_KEY]
                break
        if result is None:
            result = JoinResult(
                "unjoined",
                _frame(segment[0]) - _frame(segments[index - 1][-1]),
                "none",
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                "unknown",
                "unknown",
            )
            target = segments[index - 1][0][SEGMENT_KEY]
        joins[segment_id] = result
        join_targets[segment_id] = target
        if result.outcome == "joined":
            chain_of_segment[segment_id] = chain_of_segment[target]
        else:
            chain_of_segment[segment_id] = next_chain
            next_chain += 1
    chains: dict[int, list[dict]] = {}
    for segment in segments:
        chains.setdefault(chain_of_segment[segment[0][SEGMENT_KEY]], []).extend(segment)
    motion = {
        chain_id: chain_motion(chain_rows, geometry, clip, pan, config)
        for chain_id, chain_rows in chains.items()
    }
    # Identity is only carried by joins. The first moving chain is labelled primary; any
    # other moving chain is an explicit candidate, never promoted by motion alone.
    primary_chain = min(
        (chain_id for chain_id, state in motion.items() if state == "moving"), default=None
    )

    def owner_of(chain_id: int) -> str:
        state = motion[chain_id]
        if state == "moving":
            return "primary" if chain_id == primary_chain else "candidate"
        return {"still": "stationary", "ambiguous": "ambiguous", "insufficient": "insufficient"}[
            state
        ]

    ownership_rows = []
    for segment in segments:
        segment_id = segment[0][SEGMENT_KEY]
        chain_id = chain_of_segment[segment_id]
        result = joins[segment_id]
        for row in segment:
            public = {key: value for key, value in row.items() if not key.startswith("_")}
            public["track_id"] = chain_id
            public.update(
                {
                    "segment_id": segment_id,
                    "chain_id": chain_id,
                    "restart_kind": row[RESTART_KEY],
                    "owner": owner_of(chain_id),
                    "chain_motion": motion[chain_id],
                    "join_outcome": "initial" if result is None else result.outcome,
                    "join_target_segment": "" if result is None else join_targets[segment_id],
                    "join_gap_frames": "" if result is None else result.gap_frames,
                    "join_camera_mode": "" if result is None else result.camera_mode,
                    "join_distance_native": "" if result is None else _fmt(result.distance_native),
                    "join_tolerance_native": ""
                    if result is None
                    else _fmt(result.tolerance_native),
                    "join_mahalanobis_d2": "" if result is None else _fmt(result.mahalanobis_d2),
                }
            )
            ownership_rows.append(public)
    outcomes: dict[str, int] = {}
    for result in joins.values():
        if result is not None:
            outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    owners: dict[str, int] = {}
    for row in ownership_rows:
        owners[row["owner"]] = owners.get(row["owner"], 0) + 1
    consumed = consumed_rows(ownership_rows)
    withheld_runs = _runs([row for row in ownership_rows if row["owner"] == "stationary"])
    summary = {
        "clip": clip,
        "rows": len(rows),
        "segments": len(segments),
        "chains": len(chains),
        "primary_chain": primary_chain,
        "restart_kinds": _count(row[RESTART_KEY] for row in (segment[0] for segment in segments)),
        "join_outcomes": outcomes,
        "owner_rows": owners,
        "consumed_rows": len(consumed),
        "withheld_rows": len(rows) - len(consumed),
        "longest_withheld_run_frames": max(withheld_runs, default=0),
        "chain_motion": _count(motion.values()),
    }
    return ownership_rows, summary


def _fmt(value: float) -> str:
    return (
        "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"
    )


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _runs(rows: list[dict]) -> list[int]:
    runs = []
    previous = None
    for row in rows:
        frame = _frame(row)
        if previous is not None and frame == previous + 1:
            runs[-1] += 1
        else:
            runs.append(1)
        previous = frame
    return runs


WITHHELD_OWNERS = frozenset({"stationary"})


def consumed_rows(ownership_rows: list[dict]) -> list[dict]:
    """The rows the stage consumes: every chain not proven static, ``track_id`` = chain id.

    Primary, candidate, ambiguous and insufficient chains are all kept; identity between
    chains is carried only by the chain id, never asserted by dropping rows.
    """
    return [
        {key: value for key, value in row.items() if key not in OWNERSHIP_COLUMNS}
        for row in ownership_rows
        if row["owner"] not in WITHHELD_OWNERS
    ]
