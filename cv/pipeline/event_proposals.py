"""Label-free, high-recall event proposal generation from canonical pipeline artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cv.pipeline.ball_events import detect_events, gate_play_phase, serve_strike_frames
from cv.pipeline.bounce_detect import (
    load_boxes,
    load_track,
    projector_for_artifact,
    split_runs,
    tape_pixel_dist,
)
from cv.pipeline.resolution import coordinate_manifest_path
from cv.pipeline.provenance import file_record
from cv.pipeline.serve_hints import auto_serve_hints

SOURCE_PRIORITY = {
    "serve": 0,
    "bounce_witness": 1,
    "trajectory": 2,
    "suppressed_trajectory": 3,
    "track_gap": 4,
    "local_audio": 5,
    "sequence_audio": 6,
}
BASE_SOURCES = {
    "serve",
    "bounce_witness",
    "trajectory",
    "suppressed_trajectory",
    "track_gap",
}
ALL_SOURCES = BASE_SOURCES | {"local_audio", "sequence_audio"}
MISSING_PLAYER_DISTANCE = 99.0


# The legacy tabular candidate generator below remains the input to the old
# event_model path.  The records below are deliberately small and model-free:
# they are the interchange contract between a high-recall proposer, crop
# cutting, and the grammar.  In particular, this is not an event emission.
PROPOSAL_SCHEMA = "event_proposals_v1"
PROPOSAL_KINDS = ("contact", "bounce", "net_hit")
PROPOSAL_SOURCES = ("physics_departure", "pose_swing", "bounce_implication", "track_corner")
# These four label-free arms are the default candidate-packet inputs.  They do
# not alter the firm decoder: :mod:`cv.pipeline.event_paths` carries their
# windows forward as tentative evidence for S6 to test against physics.
DEFAULT_PROPOSAL_SOURCES = PROPOSAL_SOURCES
PROPOSAL_HALF_WINDOW = 3
IMAGE_MODEL_HISTORY = 6
IMAGE_MODEL_FUTURE = 3
# The composed native track sits on a 1.875 px lattice, so its own
# quantisation contributes a per-axis standard deviation of q/sqrt(12).  That
# is a property of the coordinate representation, not a residual floor chosen
# to keep a fit honest.
IMAGE_TRACK_LATTICE_PX = 1.875
IMAGE_NOISE_FLOOR_PX = IMAGE_TRACK_LATTICE_PX / math.sqrt(12.0)
# Consecutive exposures used for the track's own innovation estimate.  Third
# differences of a noise-free quadratic vanish, so on a consecutive run they
# are pure measurement innovation with variance 20 sigma^2.
IMAGE_NOISE_RUN = 16
THIRD_DIFFERENCE_VARIANCE = 20.0
MAD_TO_SIGMA = 0.6744897501960817
# A departure must exceed this many standard deviations of the local fit's
# *prediction* distribution at the extrapolated exposure, not of its in-sample
# residual.
IMAGE_DEPARTURE_K = 3.0
IMAGE_MAXIMUM_GAP = 8
IMAGE_SUPPRESS_FRAMES = 3.0
# The per-point homography and the sided box artifact share one court frame:
# the near baseline is y = 0, the far baseline y = 23.77 and the net y = 11.885.
# Both bands below are coarse region tests on a projected point that is
# generally above the plane, never a court position.
NET_Y_M = 11.885
NET_BAND_M = 1.5
VOLLEY_BAND_M = 4.0
PLAYER_BOX_HEIGHTS = 1.0
# A flight short enough to hold no bounce, or long enough that the mid-interval
# prior carries no information, implies nothing.  The window is capped so that
# a proposal stays a crop request rather than a whole-rally claim.
MINIMUM_RALLY_SPAN = 6.0
MAXIMUM_RALLY_SPAN = 60.0
IMPLIED_BOUNCE_MAX_HALF_WINDOW = 6.0


@dataclass(frozen=True)
class EventProposal:
    """One label-free event window offered to the video classifier.

    ``kinds`` is an explicitly non-exclusive compatibility set.  The classifier
    and grammar own the eventual type; a proposer must never turn its weak
    geometry/pose cue into an accepted event by itself.
    """

    clip: str
    frame: float
    start_frame: float
    end_frame: float
    kinds: tuple[str, ...]
    source: str
    confidence: float
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kinds"] = list(self.kinds)
        return value


def _proposal(
    clip: str,
    frame: float,
    kinds: tuple[str, ...],
    source: str,
    confidence: float,
    evidence: Mapping[str, Any],
    *,
    half_window: int = PROPOSAL_HALF_WINDOW,
) -> EventProposal:
    if source not in PROPOSAL_SOURCES:
        raise ValueError(f"unknown proposal source: {source}")
    allowed = tuple(kind for kind in kinds if kind in PROPOSAL_KINDS)
    if not allowed:
        raise ValueError("proposal needs at least one physical event kind")
    return EventProposal(
        clip=str(clip),
        frame=float(frame),
        start_frame=float(frame - half_window),
        end_frame=float(frame + half_window),
        kinds=allowed,
        source=source,
        confidence=max(0.0, min(1.0, float(confidence))),
        evidence=dict(evidence),
    )


def _contiguous_window(
    values: Mapping[int, np.ndarray], frame: int, offsets: range
) -> list[np.ndarray] | None:
    points = [values.get(frame + offset) for offset in offsets]
    return (
        None
        if any(point is None for point in points)
        else [np.asarray(point, float) for point in points]
    )


def physics_departures(
    clip: str,
    positions_m: Mapping[int, Any],
    *,
    fps: float,
    player_positions_m: Mapping[str, Mapping[int, Any]] | None = None,
    net_y_m: float = 11.885,
    residual_m: float = 0.35,
) -> list[EventProposal]:
    """Propose sustained departures from a drag/gravity flight in 3-D.

    This consumes only a metric, source-derived 3-D track.  A homography is a
    ground-plane mapping, not a licence to invent ball height, so callers with
    only image/court-plane tracks must abstain rather than pass pseudo-3-D
    points here.  Forward *and* backward predictions use ``physics.flight``'s
    signed sampler; two consecutive residuals are required before a window is
    offered.
    """

    from physics.flight import sample_signed_states

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    points: dict[int, np.ndarray] = {}
    for frame, value in positions_m.items():
        point = np.asarray(value, dtype=float)
        if point.shape == (3,) and np.isfinite(point).all():
            points[int(frame)] = point
    proposals: list[EventProposal] = []
    dt = 1.0 / fps
    for frame in sorted(points):
        before = _contiguous_window(points, frame, range(-4, 0))
        after = _contiguous_window(points, frame, range(1, 5))
        if before is None or after is None:
            continue
        velocity_before = (before[-1] - before[0]) / (3.0 * dt)
        velocity_after = (after[-1] - after[0]) / (3.0 * dt)
        forward, _, _ = sample_signed_states(
            before[-1], velocity_before, np.zeros(3), np.arange(1, 5) * dt
        )
        backward, _, _ = sample_signed_states(
            after[0], velocity_after, np.zeros(3), -np.arange(1, 5)[::-1] * dt
        )
        forward_error = np.asarray(
            [
                np.linalg.norm(actual - predicted)
                for actual, predicted in zip(after, forward, strict=True)
            ]
        )
        backward_error = np.asarray(
            [
                np.linalg.norm(actual - predicted)
                for actual, predicted in zip(before, backward[::-1], strict=True)
            ]
        )
        sustained = max(float(np.median(forward_error[:2])), float(np.median(backward_error[-2:])))
        if sustained < residual_m:
            continue
        vertical_reversal = velocity_before[2] < -0.2 and velocity_after[2] > 0.2
        near_net = abs(float(points[frame][1]) - net_y_m) <= 0.6
        kinds: tuple[str, ...]
        if vertical_reversal and points[frame][2] <= 0.35:
            kinds = ("bounce",)
        elif near_net:
            kinds = ("net_hit", "contact")
        else:
            kinds = ("contact",)
        nearest_player = math.inf
        for by_frame in (player_positions_m or {}).values():
            player = by_frame.get(frame)
            if player is not None:
                player = np.asarray(player, float)
                if player.shape == (3,) and np.isfinite(player).all():
                    nearest_player = min(
                        nearest_player, float(np.linalg.norm(points[frame][:2] - player[:2]))
                    )
        if kinds == ("contact",) and nearest_player > 2.5:
            # A departure without a plausible racket region is useful evidence,
            # but not enough to name contact over an unobserved bounce/net hit.
            kinds = ("contact", "bounce", "net_hit")
        proposals.append(
            _proposal(
                clip,
                frame,
                kinds,
                "physics_departure",
                min(1.0, sustained / max(residual_m * 3.0, 1e-6)),
                {
                    "forward_residual_m": forward_error.tolist(),
                    "backward_residual_m": backward_error.tolist(),
                    "sustained_residual_m": sustained,
                    "vertical_reversal": vertical_reversal,
                    "nearest_player_m": None
                    if not math.isfinite(nearest_player)
                    else nearest_player,
                },
            )
        )
    return union_proposals(proposals)


@dataclass(frozen=True)
class _LocalFit:
    """A local image quadratic with the pieces needed for a prediction interval.

    ``sigma`` is a per-axis noise standard deviation, ``gram`` the inverse
    Gram matrix of the design.  Together they give the variance of a
    *prediction* at an arbitrary exposure, which grows with the extrapolation
    distance; the in-sample residual does not.
    """

    coefficients: np.ndarray
    origin: float
    gram: np.ndarray
    sigma_fit: float
    sigma: float

    def predict(self, frames: list[float]) -> np.ndarray:
        return self._design(frames) @ self.coefficients

    def prediction_sigma(self, frames: list[float]) -> np.ndarray:
        """Per-axis standard deviation of a new observation at ``frames``.

        ``sigma * sqrt(1 + h)`` with ``h`` the design leverage: one unit for
        the new observation's own noise and ``h`` for the fitted mean's.
        """

        design = self._design(frames)
        leverage = np.einsum("ij,jk,ik->i", design, self.gram, design)
        return self.sigma * np.sqrt(1.0 + np.maximum(leverage, 0.0))

    def _design(self, frames: list[float]) -> np.ndarray:
        times = np.asarray(frames, dtype=float) - self.origin
        return np.column_stack((np.ones(len(times)), times, times * times))


def _quadratic_fit(
    frames: list[int], points: list[np.ndarray], *, noise_sigma: float | None = None
) -> _LocalFit | None:
    """Fit independent image-coordinate quadratics with an honest noise scale.

    The residual scale is the degrees-of-freedom corrected in-sample estimate,
    raised to the track's own innovation estimate when that is larger.  A six
    point quadratic has three residual degrees of freedom per axis, so its
    in-sample spread systematically understates the noise and, on a jittery
    track, collapses; taking the larger of the two estimates is what stops a
    sub-pixel numerical coincidence becoming a departure claim.
    """

    if len(frames) < 4:
        return None
    times = np.asarray(frames, dtype=float)
    origin = float(times[0])
    times = times - origin
    design = np.column_stack((np.ones(len(times)), times, times * times))
    values = np.asarray(points, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    dof = 2 * (len(frames) - 3)
    sigma_fit = math.sqrt(float((residuals**2).sum()) / dof) if dof > 0 else 0.0
    sigma = max(sigma_fit, float(noise_sigma or 0.0), IMAGE_NOISE_FLOOR_PX)
    return _LocalFit(
        coefficients=coefficients,
        origin=origin,
        gram=np.linalg.pinv(design.T @ design),
        sigma_fit=sigma_fit,
        sigma=sigma,
    )


def _observed_run(points: Mapping[int, np.ndarray], frame: int, offsets: range) -> list[int]:
    return [frame + offset for offset in offsets if frame + offset in points]


def _consecutive_run(points: Mapping[int, np.ndarray], frame: int, *, back: bool) -> list[int]:
    run = [frame]
    step = -1 if back else 1
    while len(run) < IMAGE_NOISE_RUN and (run[-1] + step) in points:
        run.append(run[-1] + step)
    return sorted(run)


def _innovation_sigma(points: Mapping[int, np.ndarray], run: list[int]) -> float | None:
    """Per-axis observation noise from the track's own third differences.

    A constant-acceleration image path has zero third difference, so on a run
    of consecutive exposures the third difference is the composed track's own
    innovation.  A median absolute deviation keeps a single wrong-object lock
    from inflating the estimate.  The run never crosses the candidate
    interval, so a real event cannot contaminate it.
    """

    differences = []
    for index in range(len(run) - 3):
        first, second, third, fourth = run[index : index + 4]
        if fourth - first != 3:
            continue
        differences.append(points[fourth] - 3 * points[third] + 3 * points[second] - points[first])
    if not differences:
        return None
    spread = float(np.median(np.abs(np.asarray(differences, dtype=float)).ravel()))
    return spread / (MAD_TO_SIGMA * math.sqrt(THIRD_DIFFERENCE_VARIANCE))


def image_physics_departures(
    clip: str,
    track: Mapping[int, Any],
    *,
    fps: float,
    player_boxes: Mapping[int, list[tuple[float, float, float, float]]] | None = None,
    homography: np.ndarray | None = None,
    departure_k: float = IMAGE_DEPARTURE_K,
    maximum_gap: int = IMAGE_MAXIMUM_GAP,
) -> list[EventProposal]:
    """Find two-sided 2-D ballistic departures without manufacturing 3-D.

    The model is intentionally observable: each image coordinate gets a local
    quadratic fit on the preceding and following native exposures, and the
    candidate epoch is the *interval* between two consecutive observations.
    The preceding fit is propagated forward over the interval and the following
    fit backward over it; a sustained disagreement on either side proposes an
    event window.

    The disagreement is measured against that fit's **prediction** interval at
    the extrapolated exposure, not against its in-sample residual.  The noise
    scale is the larger of the degrees-of-freedom corrected residual and the
    track's own third-difference innovation, and it is propagated through the
    design leverage, so the bar rises with the extrapolation distance instead
    of sitting on a constant pixel floor.  Without that, a six-point quadratic
    absorbs the track's own jitter, the in-sample residual collapses onto the
    floor and every interval departs.

    The interval formulation is what lets the arm speak where the composed
    track has a hole: a ball that disappears behind a player and reappears
    several exposures later is exactly the case the corner witness cannot see.
    Nothing here is a height claim and no picture or exposure is invented; the
    court plane is used only through the supplied per-point homography, and
    only for the ground/net geometry of an already-detected departure.
    """

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    points: dict[int, np.ndarray] = {}
    for frame, value in track.items():
        point = np.asarray(value, dtype=float)
        if point.shape == (2,) and np.isfinite(point).all():
            points[int(frame)] = point
    ordered = sorted(points)
    scored: list[tuple[float, EventProposal]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        gap = right - left
        if gap > maximum_gap:
            continue
        before_frames = _observed_run(points, left, range(-IMAGE_MODEL_HISTORY + 1, 1))
        after_frames = _observed_run(points, right, range(0, IMAGE_MODEL_HISTORY))
        # The innovation runs stop at the interval, so the candidate event is
        # never part of its own noise estimate.
        before_noise = _innovation_sigma(points, _consecutive_run(points, left, back=True))
        after_noise = _innovation_sigma(points, _consecutive_run(points, right, back=False))
        before_fit = _quadratic_fit(
            before_frames, [points[item] for item in before_frames], noise_sigma=before_noise
        )
        after_fit = _quadratic_fit(
            after_frames, [points[item] for item in after_frames], noise_sigma=after_noise
        )
        if before_fit is None or after_fit is None:
            continue
        forward_frames = after_frames[:IMAGE_MODEL_FUTURE]
        backward_frames = before_frames[-IMAGE_MODEL_FUTURE:]
        forward_error = np.linalg.norm(
            before_fit.predict(forward_frames)
            - np.asarray([points[item] for item in forward_frames]),
            axis=1,
        )
        backward_error = np.linalg.norm(
            after_fit.predict(backward_frames)
            - np.asarray([points[item] for item in backward_frames]),
            axis=1,
        )
        # The error is a two-dimensional norm of two independent per-axis
        # predictions, so the isotropic prediction scale is sqrt(2) sigma.
        forward_ratio = forward_error / (
            math.sqrt(2.0) * before_fit.prediction_sigma(forward_frames)
        )
        backward_ratio = backward_error / (
            math.sqrt(2.0) * after_fit.prediction_sigma(backward_frames)
        )
        forward_departure = _sustained_departure(forward_ratio)
        backward_departure = _sustained_departure(backward_ratio)
        sustained_forward = forward_departure >= departure_k
        sustained_backward = backward_departure >= departure_k
        if not (sustained_forward or sustained_backward):
            continue

        epoch = (left + right) / 2.0
        # The crossing of the two local models is the observable estimate of
        # where the ball was when its motion changed.  It is a prediction, not
        # an exposure: it is only ever used to place a crop window.
        predicted = 0.5 * (before_fit.predict([epoch])[0] + after_fit.predict([epoch])[0])
        before_velocity = points[left] - points[before_frames[-2]]
        after_velocity = points[after_frames[1]] - points[right]
        vertical_reversal = bool(before_velocity[1] * after_velocity[1] < 0.0)
        descending_before = bool(before_velocity[1] > 0.0)
        nearest_box = math.inf
        for reference in (left, right):
            for box in (player_boxes or {}).get(reference, []):
                x0, y0, x1, y1 = box
                dx = max(x0 - predicted[0], 0.0, predicted[0] - x1)
                dy = max(y0 - predicted[1], 0.0, predicted[1] - y1)
                nearest_box = min(nearest_box, math.hypot(dx, dy) / max(y1 - y0, 1.0))
        court_y = None
        near_net = False
        if homography is not None:
            projected = np.asarray(homography, float) @ np.array([predicted[0], predicted[1], 1.0])
            if np.isfinite(projected).all() and abs(projected[2]) > 1e-9:
                # The per-point homography maps a native pixel to court metres
                # with the net at y = 0; a projected ball above the plane is
                # displaced, so this is a coarse region test, never a position.
                court_y = float(projected[1] / projected[2])
                near_net = abs(court_y - NET_Y_M) <= NET_BAND_M
        # Geometry *adds* compatible kinds; it never elects one.  A ball that
        # turns over inside a player box near the net is compatible with a
        # volley, a net cord and a bounce at once, and a proposer that picked
        # one of them would be making the classifier's decision for it.
        compatible = set()
        if nearest_box <= PLAYER_BOX_HEIGHTS:
            compatible.add("contact")
        if near_net:
            compatible.update(("net_hit", "contact"))
        if vertical_reversal and descending_before:
            compatible.add("bounce")
        # Preserve uncertainty when a departure has no usable geometry; this is
        # a crop request, never an accepted event.
        kinds = tuple(kind for kind in PROPOSAL_KINDS if kind in compatible) or PROPOSAL_KINDS
        score = float(max(forward_departure, backward_departure))
        half_window = max(PROPOSAL_HALF_WINDOW, gap / 2.0 + 1.0)
        scored.append(
            (
                score,
                _proposal(
                    clip,
                    epoch,
                    kinds,
                    "physics_departure",
                    min(1.0, score / (2.0 * departure_k)),
                    {
                        "frame_space": "native_1920x1080",
                        "model": "local_image_quadratic",
                        "history_frames": len(before_frames),
                        "future_frames": len(after_frames),
                        "interval": [int(left), int(right)],
                        "forward_residual_px": forward_error.tolist(),
                        "backward_residual_px": backward_error.tolist(),
                        "forward_prediction_sigma_px": (
                            math.sqrt(2.0) * before_fit.prediction_sigma(forward_frames)
                        ).tolist(),
                        "backward_prediction_sigma_px": (
                            math.sqrt(2.0) * after_fit.prediction_sigma(backward_frames)
                        ).tolist(),
                        "forward_noise_sigma_px": before_fit.sigma,
                        "backward_noise_sigma_px": after_fit.sigma,
                        "forward_innovation_sigma_px": before_noise,
                        "backward_innovation_sigma_px": after_noise,
                        "forward_in_sample_sigma_px": before_fit.sigma_fit,
                        "backward_in_sample_sigma_px": after_fit.sigma_fit,
                        "departure_k": float(departure_k),
                        "sustained_forward": sustained_forward,
                        "sustained_backward": sustained_backward,
                        "departure_sigma": score,
                        "vertical_reversal": vertical_reversal,
                        "descending_before": descending_before,
                        "predicted_x": float(predicted[0]),
                        "predicted_y": float(predicted[1]),
                        "nearest_player_box_heights": None
                        if not math.isfinite(nearest_box)
                        else nearest_box,
                        "court_y_m": court_y,
                        "near_net": near_net,
                    },
                    half_window=half_window,
                ),
            )
        )
    return _suppressed(scored)


def _sustained_departure(ratios: np.ndarray) -> float:
    """The largest departure that at least two extrapolated exposures reach.

    One exposure can depart on a single mis-associated detection; a physical
    change of motion holds for the rest of the propagated window.
    """

    if ratios.size < 2:
        return 0.0
    return float(np.sort(ratios)[-2])


def _suppressed(
    scored: list[tuple[float, EventProposal]], *, suppress: float = IMAGE_SUPPRESS_FRAMES
) -> list[EventProposal]:
    """Keep the strongest departure in each neighbourhood, as the corner arm does."""

    kept: list[EventProposal] = []
    for _, proposal in sorted(scored, key=lambda item: (-item[0], item[1].frame)):
        if any(abs(other.frame - proposal.frame) <= suppress for other in kept):
            continue
        kept.append(proposal)
    return sorted(kept, key=lambda proposal: proposal.frame)


def pose_swing_proposals(
    clip: str,
    players: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    fps: float,
    minimum_wrist_speed_px_s: float = 180.0,
    minimum_shoulder_rotation_degrees: float = 12.0,
) -> list[EventProposal]:
    """Return wrist-arc swing apices from the corrected native pose rows.

    The detector has no learned threshold from the scoring truth.  Per player
    and per hand it wants a local maximum of native wrist speed that also shows
    either a signed reversal of the wrist arc or shoulder rotation across the
    same window.  The three swing shapes the owner named are recorded as
    provenance cues, not as separate accepted classes: a wrist that rises above
    both shoulders after a rising toss is ``serve_toss_overhead``, a short
    punch inside the net band is ``volley_punch``, and rotation with a wrist
    arc is ``groundstroke``.  All of them propose the physical ``contact``
    kind and nothing else.
    """

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    output: list[EventProposal] = []
    for player, rows in players.items():
        for hand in ("left", "right"):
            scored: list[tuple[float, EventProposal]] = []
            speeds: dict[int, float] = {}
            windows: dict[int, list[Mapping[str, Any]]] = {}
            for frame in sorted(int(item) for item in rows):
                window, wide = _pose_window(rows, frame)
                if window is None:
                    continue
                path = _wrist_path(window, hand)
                if path is None:
                    continue
                speeds[frame] = (
                    max(
                        float(np.linalg.norm(path[2] - path[0])) / 2.0,
                        float(np.linalg.norm(path[4] - path[2])) / 2.0,
                    )
                    * fps
                )
                windows[frame] = [window, wide, path]
            for frame, speed in speeds.items():
                if speed < minimum_wrist_speed_px_s:
                    continue
                neighbours = [
                    speeds[frame + offset] for offset in (-2, -1, 1, 2) if frame + offset in speeds
                ]
                if neighbours and speed < max(neighbours):
                    continue
                window, wide, path = windows[frame]
                centre = window[2]
                rotation_degrees = _shoulder_rotation_degrees(window) if wide else 0.0
                incoming, outgoing = path[2] - path[1], path[3] - path[2]
                arc_apex = float(np.dot(incoming, outgoing)) <= 0.0
                if not (arc_apex or rotation_degrees >= minimum_shoulder_rotation_degrees):
                    continue
                shoulder_y = _shoulder_y(centre)
                overhead = shoulder_y is not None and path[2][1] < shoulder_y
                cue = "groundstroke"
                if overhead and path[0][1] > path[2][1]:
                    cue = "serve_toss_overhead"
                court_y = _finite_float(centre.get("court_y"))
                punch = float(np.linalg.norm(path[4] - path[0])) < 1.5 * speed / fps
                if court_y is not None and abs(court_y - NET_Y_M) < VOLLEY_BAND_M and punch:
                    cue = "volley_punch"
                scored.append(
                    (
                        speed,
                        _proposal(
                            clip,
                            frame,
                            ("contact",),
                            "pose_swing",
                            min(
                                1.0,
                                0.65 * speed / (2.0 * minimum_wrist_speed_px_s)
                                + 0.35 * rotation_degrees / 45.0,
                            ),
                            {
                                "player": player,
                                "hand": hand,
                                "swing_cue": cue,
                                "wrist_speed_px_s": speed,
                                "arc_apex": arc_apex,
                                "shoulder_rotation_degrees": rotation_degrees,
                                "court_y_m": court_y,
                                "predicted_x": float(path[2][0]),
                                "predicted_y": float(path[2][1]),
                                "native_pose": True,
                            },
                        ),
                    )
                )
            output.extend(_suppressed(scored, suppress=2.0))
    return union_proposals(output)


def _pose_window(
    rows: Mapping[int, Mapping[str, Any]], frame: int
) -> tuple[list[Mapping[str, Any]] | None, bool]:
    """Five poses when they exist, otherwise three padded at the edges.

    Missing outer context must not become an abstention: the shoulder-rotation
    evidence is simply unavailable, and the caller is told so.
    """

    wide = [rows.get(frame + offset) for offset in (-2, -1, 0, 1, 2)]
    if all(row is not None for row in wide):
        return [row for row in wide if row is not None], True
    short = [rows.get(frame + offset) for offset in (-1, 0, 1)]
    if any(row is None for row in short):
        return None, False
    first, middle, last = (row for row in short if row is not None)
    return [first, first, middle, last, last], False


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _wrist_path(rows: list[Mapping[str, Any] | None], hand: str) -> list[np.ndarray] | None:
    path = []
    for row in rows:
        if row is None or _finite_float(row.get(f"{hand}_wrist_confidence")) is None:
            return None
        if float(row[f"{hand}_wrist_confidence"]) < 0.2:
            return None
        x = _finite_float(row.get(f"{hand}_wrist_x"))
        y = _finite_float(row.get(f"{hand}_wrist_y"))
        if x is None or y is None:
            return None
        path.append(np.asarray((x, y), dtype=float))
    return path


def _shoulder_y(row: Mapping[str, Any]) -> float | None:
    values = [
        _finite_float(row.get(f"{side}_shoulder_y"))
        for side in ("left", "right")
        if (_finite_float(row.get(f"{side}_shoulder_confidence")) or 0.0) >= 0.2
    ]
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _shoulder_rotation_degrees(rows: list[Mapping[str, Any] | None]) -> float:
    vectors = []
    for row in (rows[1], rows[3]):
        if row is None:
            return 0.0
        points = []
        for hand in ("left", "right"):
            if (_finite_float(row.get(f"{hand}_shoulder_confidence")) or 0.0) < 0.2:
                return 0.0
            x = _finite_float(row.get(f"{hand}_shoulder_x"))
            y = _finite_float(row.get(f"{hand}_shoulder_y"))
            if x is None or y is None:
                return 0.0
            points.append(np.asarray((x, y), dtype=float))
        vectors.append(points[1] - points[0])
    if min(np.linalg.norm(vector) for vector in vectors) < 1e-6:
        return 0.0
    cosine = float(
        np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def implied_bounces(contacts: list[EventProposal]) -> list[EventProposal]:
    """Offer one bounce window between successive plausible groundstrokes.

    This is deliberately broad, and only applies after two independently
    proposed contact windows.  The classifier/grammar must still observe a
    bounce; this rule merely prevents a missing pixel proposal from making that
    impossible.
    """

    by_clip: dict[str, list[EventProposal]] = {}
    for proposal in contacts:
        # Only an unambiguous contact witness implies a flight.  A corner that
        # is compatible with three kinds says nothing about which shot it is,
        # and using it here would manufacture a bounce between two bounces.
        if proposal.kinds == ("contact",):
            by_clip.setdefault(proposal.clip, []).append(proposal)
    output = []
    for clip, rows in by_clip.items():
        ordered = sorted(rows, key=lambda row: row.frame)
        for left, right in zip(ordered, ordered[1:], strict=False):
            span = right.frame - left.frame
            if span < MINIMUM_RALLY_SPAN or span > MAXIMUM_RALLY_SPAN:
                continue
            output.append(
                _proposal(
                    clip,
                    (left.frame + right.frame) / 2.0,
                    ("bounce",),
                    "bounce_implication",
                    min(left.confidence, right.confidence) * 0.5,
                    {
                        "left_contact_frame": left.frame,
                        "right_contact_frame": right.frame,
                        "span_frames": span,
                    },
                    half_window=min(
                        IMPLIED_BOUNCE_MAX_HALF_WINDOW, max(PROPOSAL_HALF_WINDOW, span / 4.0)
                    ),
                )
            )
    return union_proposals(output)


def union_proposals(
    proposals: list[EventProposal], *, merge_frames: float = 1.0
) -> list[EventProposal]:
    """Merge same-source windows only; preserve independent evidence sources."""

    output: list[EventProposal] = []
    for proposal in sorted(
        proposals, key=lambda row: (row.clip, row.source, row.frame, -row.confidence)
    ):
        existing = next(
            (
                row
                for row in reversed(output)
                if row.clip == proposal.clip
                and row.source == proposal.source
                and abs(row.frame - proposal.frame) <= merge_frames
            ),
            None,
        )
        if existing is None:
            output.append(proposal)
    return output


def track_corner_proposals(
    clip: str, track: Mapping[int, Any], *, min_turn_degrees: float = 45.0
) -> list[EventProposal]:
    """Convert the established composed-track corner witness into windows.

    This is an explicit proposer, rather than an emission rescue.  It shares
    the geometry with :mod:`contact_frame_refiner` but has no access to model
    scores or labels.
    """

    from cv.pipeline.contact_recall import corners

    output = []
    for corner in corners(track, min_deg=min_turn_degrees):
        output.append(
            _proposal(
                clip,
                corner.frame,
                ("contact", "bounce", "net_hit"),
                "track_corner",
                min(1.0, corner.turn_deg / 180.0),
                {
                    "turn_degrees": corner.turn_deg,
                    "step_in_px": corner.step_in_px,
                    "step_out_px": corner.step_out_px,
                },
            )
        )
    return output


def proposal_document(
    proposals: list[EventProposal], *, inputs: list[str] | None = None
) -> dict[str, Any]:
    """Serialize a fail-closed automatic proposal artifact."""

    rows = [proposal.as_dict() for proposal in union_proposals(proposals)]
    return {
        "schema": PROPOSAL_SCHEMA,
        "labels_or_reviewed_inputs": [],
        "inputs": list(inputs or []),
        "input_records": [
            file_record(path, role="automatic_proposal_input") for path in inputs or []
        ],
        "proposals": rows,
        "by_source": {
            source: sum(row["source"] == source for row in rows)
            for source in (
                *PROPOSAL_SOURCES,
                *sorted({r["source"] for r in rows} - set(PROPOSAL_SOURCES)),
            )
        },
        "default_packet_inputs": list(DEFAULT_PROPOSAL_SOURCES),
        "default_effect_on_firm_emissions": "none",
        "abstentions": {
            "physics_3d": "requires an explicit source-derived metric 3-D ball track",
            "pose": "requires temporally associated confident wrist keypoints",
        },
    }


def load_proposal_document(path: Path) -> list[EventProposal]:
    """Load and validate a proposal artifact; labels are rejected at the boundary."""

    document = json.loads(path.read_text())
    if document.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError(f"unsupported proposal schema: {path}")
    if document.get("labels_or_reviewed_inputs"):
        raise ValueError("automatic proposal artifact carries human-derived inputs")
    output = []
    for row in document.get("proposals", []):
        output.append(
            EventProposal(
                clip=str(row["clip"]),
                frame=float(row["frame"]),
                start_frame=float(row["start_frame"]),
                end_frame=float(row["end_frame"]),
                kinds=tuple(str(kind) for kind in row["kinds"]),
                source=str(row["source"]),
                confidence=float(row["confidence"]),
                evidence=dict(row.get("evidence") or {}),
            )
        )
    return output


def _coordinate_columns(path: Path) -> tuple[str, str, float, float]:
    """Read track columns in the sidecar-declared image space, then make native."""

    from cv.pipeline.event_crops import track_coordinate_space

    with path.open(newline="") as handle:
        columns = csv.DictReader(handle).fieldnames or []
    sidecar = coordinate_manifest_path(path)
    if sidecar is None or not sidecar.exists():
        raise FileNotFoundError(f"{path} has no coordinate sidecar")
    space = track_coordinate_space(sidecar, columns)
    return (
        str(space["x_column"]),
        str(space["y_column"]),
        float(space["scale_x"]),
        float(space["scale_y"]),
    )


def _native_track(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    x_key, y_key, scale_x, scale_y = _coordinate_columns(path)
    output: dict[str, dict[int, tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                point = (float(row[x_key]) * scale_x, float(row[y_key]) * scale_y)
                frame = _frame_number(row["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in point):
                output.setdefault(str(row["clip"]), {})[frame] = point
    return output


def _native_boxes(path: Path) -> dict[str, dict[int, list[tuple[float, float, float, float]]]]:
    """Read a sided/unsided box artifact in its declared coordinate space."""

    _, _, scale_x, scale_y = _coordinate_columns(path)
    output: dict[str, dict[int, list[tuple[float, float, float, float]]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values = (
                    float(row["x0"]) * scale_x,
                    float(row["y0"]) * scale_y,
                    float(row["x1"]) * scale_x,
                    float(row["y1"]) * scale_y,
                )
                frame = _frame_number(row["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                output.setdefault(str(row["clip"]), {}).setdefault(frame, []).append(values)
    return output


def _point_homographies(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as source:
        if "pts" not in source or "H" not in source:
            return {}
        return {
            f"pt{int(point):04d}": np.asarray(homography, dtype=float)
            for point, homography in zip(source["pts"], source["H"], strict=True)
            if np.asarray(homography).shape == (3, 3) and np.isfinite(homography).all()
        }


def _native_pose_players(path: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Load side-associated pose rows through pose.py's native sidecar contract."""

    from cv.pipeline.pose import load_native_pose_rows

    output: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for row in load_native_pose_rows(str(path)):
        try:
            clip, side, frame = str(row["clip"]), str(row["side"]), _frame_number(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if side not in {"near", "far"}:
            continue
        output.setdefault(clip, {}).setdefault(side, {})[frame] = row
    return output


def _source_fps(match_dir: Path) -> float:
    sidecar = match_dir / "audit_frames_native_1080.coordinates.json"
    try:
        fps = float(json.loads(sidecar.read_text())["fps"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing source fps sidecar: {sidecar}") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid source fps: {sidecar}")
    return fps


BOX_PATTERN = "player_boxes_*_native_sided_v1.csv"
POSE_NAMES = (
    "player_pose_tracked_crop_native_backfill_v1.csv",
    "player_pose_tracked_crop_native_v1.csv",
)


def _first_existing(match_dir: Path, names: tuple[str, ...]) -> Path | None:
    return next((match_dir / name for name in names if (match_dir / name).exists()), None)


def _sided_box_path(match_dir: Path, fps: float) -> Path | None:
    """The sided box artifact cut at this source's own cadence.

    A broadcast directory can hold several detector cadences.  Choosing by the
    declared source fps keeps one box per exposure; the sorted fallback keeps
    the choice deterministic rather than filesystem-ordered.
    """

    preferred = match_dir / f"player_boxes_{round(fps)}_native_sided_v1.csv"
    if preferred.exists():
        return preferred
    return next(iter(sorted(match_dir.glob(BOX_PATTERN))), None)


def generate_proposal_document(
    root: Path,
    output: Path,
    *,
    track_name: str,
    pose_root: Path | None = None,
    pose_roots: tuple[Path, ...] = (),
    pose_names: tuple[str, ...] = POSE_NAMES,
    corner_min_degrees: float = 110.0,
    departure_k: float = IMAGE_DEPARTURE_K,
    measured_current_track: bool = False,
) -> dict[str, Any]:
    """Generate corner, image-physics, pose, and implied-bounce proposal windows.

    ``pose_root`` is explicit because a pose refresh is an input to this stage,
    not an ambient convenience file.  A missing pose/box/homography degrades a
    single geometry cue only; it never changes the track's coordinate space or
    manufactures an observation.  Every artifact is read through its own
    coordinate sidecar, so a file named ``*_native_*`` that declares 960x540 is
    scaled rather than believed.
    """

    if pose_root is not None and pose_roots:
        raise ValueError("pass pose_root or pose_roots, not both")
    resolved_pose_roots = (pose_root,) if pose_root is not None else tuple(pose_roots)
    inputs: list[str] = []
    track_bindings: list[dict] = []
    if measured_current_track:
        from cv.pipeline import automatic_ball_track

        if track_name != automatic_ball_track.CURRENT_TRACK_NAME:
            raise ValueError(
                "measured current-track proposals require the declared S4 current track"
            )
    proposals: list[EventProposal] = []
    abstained: dict[str, list[str]] = {"pose": [], "player_boxes": [], "homography": []}
    for match_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        track_path = match_dir / track_name
        # Pose is an independent source. Missing ball observations cannot veto
        # a match or clip that has its own native, side-associated pose rows.
        if track_path.exists() and measured_current_track:
            binding = automatic_ball_track.bind_current_track(match_dir)
            by_clip = binding.observations
            track_bindings.append(binding.record())
            inputs.extend(str(path.resolve()) for path in binding.paths())
        else:
            by_clip = _native_track(track_path) if track_path.exists() else {}
        pose_path = None
        for root_candidate in resolved_pose_roots:
            pose_match = root_candidate / match_dir.name
            if pose_match.exists() and (candidate := _first_existing(pose_match, pose_names)):
                pose_path = candidate
                break
        pose_players = _native_pose_players(pose_path) if pose_path is not None else {}
        if not by_clip and not pose_players:
            continue
        fps = _source_fps(match_dir)
        inputs.append(str((match_dir / "audit_frames_native_1080.coordinates.json").resolve()))
        boxes_path = _sided_box_path(match_dir, fps)
        boxes = _native_boxes(boxes_path) if boxes_path is not None else {}
        homography_path = match_dir / "court_H_per_point.npz"
        homographies = _point_homographies(homography_path)
        for name, present in (
            ("pose", pose_path is not None),
            ("player_boxes", boxes_path is not None),
            ("homography", bool(homographies)),
        ):
            if not present:
                abstained[name].append(match_dir.name)
        for clip in sorted(set(by_clip) | set(pose_players)):
            track = by_clip.get(clip, {})
            global_clip = f"{match_dir.name}__{clip}"
            proposals.extend(
                track_corner_proposals(global_clip, track, min_turn_degrees=corner_min_degrees)
            )
            physics = image_physics_departures(
                global_clip,
                track,
                fps=fps,
                player_boxes=boxes.get(clip),
                homography=homographies.get(clip),
                departure_k=departure_k,
            )
            poses = pose_swing_proposals(global_clip, pose_players.get(clip, {}), fps=fps)
            proposals.extend(physics)
            proposals.extend(poses)
            proposals.extend(implied_bounces([*physics, *poses]))
        if track_path.exists():
            inputs.append(str(track_path.resolve()))
        for optional in (boxes_path, homography_path, pose_path):
            if optional is not None and optional.exists():
                inputs.append(str(optional.resolve()))
                coordinate_sidecar = Path(f"{optional}.coordinates.json")
                if coordinate_sidecar.exists():
                    inputs.append(str(coordinate_sidecar.resolve()))
    document = proposal_document(proposals, inputs=list(dict.fromkeys(inputs)))
    if measured_current_track:
        document["measured_current_track"] = {"bindings": track_bindings, "enabled": True}
    document["configuration"] = {
        "departure_k": float(departure_k),
        "corner_min_degrees": float(corner_min_degrees),
        "track_name": str(track_name),
    }
    document["abstentions"] = {
        "physics_3d": (
            "not used; the arm is an image-space local quadratic departure and makes no "
            "height claim, so no metric 3-D track is required or asserted"
        ),
        "pose": (
            f"{len(abstained['pose'])} broadcasts without a native pose artifact"
            if resolved_pose_roots
            else "pose root not supplied; the pose arm abstains everywhere"
        ),
        "player_boxes": f"{len(abstained['player_boxes'])} broadcasts without a sided box artifact",
        "homography": f"{len(abstained['homography'])} broadcasts without a per-point homography",
        "broadcasts": abstained,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def generate_corner_document(root: Path, output: Path, *, track_name: str) -> dict[str, Any]:
    """Compatibility spelling for the full observable proposal generator."""

    return generate_proposal_document(root, output, track_name=track_name)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalized_player_edge_distance(
    boxes: dict,
    frame: float,
    x: float,
    y: float,
    *,
    fps: float,
) -> float:
    """Distance from a ball hypothesis to the nearest player box, in box heights."""
    tolerance = max(1, round(0.12 * fps))
    distances = []
    for side in ("near", "far"):
        candidates = [
            box_frame for box_frame in boxes.get(side, {}) if abs(box_frame - frame) <= tolerance
        ]
        if not candidates:
            continue
        box_frame = min(candidates, key=lambda value: abs(value - frame))
        x0, y0, x1, y1 = boxes[side][box_frame]
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        distances.append(math.hypot(dx, dy) / max(y1 - y0, 1.0))
    return min(distances, default=MISSING_PLAYER_DISTANCE)


def _base_row(
    clip: str,
    event: dict,
    index: int,
    *,
    boxes: dict,
    fps: float,
) -> dict:
    witness = event.get("bounce_witness") or {}
    reach = event.get("reach")
    initial_type = event.get("initial_type", event["type"])
    row = {
        "clip": clip,
        "candidate_id": f"{clip}_c{index:03d}",
        "candidate_frame": float(event.get("candidate_frame", event["frame"])),
        "initial_type": initial_type,
        "final_type": event["type"],
        "initial_confidence": float(event.get("initial_confidence", event.get("confidence", 0.0))),
        "final_confidence": float(event.get("confidence", 0.0)),
        "kind": event.get("kind", ""),
        "in_play": int(bool(event.get("in_play", True))),
        "excluded": event.get("excl", ""),
        "sb": float(event.get("sb", 0.0)),
        "sa": float(event.get("sa", 0.0)),
        "speed_ratio": float(event.get("sa", 0.0)) / max(float(event.get("sb", 0.0)), 0.3),
        "angle": float(event.get("ang", 0.0)),
        "vxb": float(event.get("vxb", 0.0)),
        "vyb": float(event.get("vyb", 0.0)),
        "vxa": float(event.get("vxa", 0.0)),
        "vya": float(event.get("vya", 0.0)),
        "gap_frames": float(event.get("fF", event["frame"]))
        - float(event.get("fL", event["frame"])),
        "horiz_reverse": int(bool(event.get("horiz_rev"))),
        "vert_reverse": int(bool(event.get("vert_rev"))),
        "big_gain": int(bool(event.get("big_gain"))),
        "collapse": int(bool(event.get("collapse"))),
        "at_height": int(bool(event.get("at_height"))),
        "has_reach": int(reach is not None),
        "reach_near": int(reach == "near"),
        "reach_far": int(reach == "far"),
        "audio": float(event["audio"]) if _finite(event.get("audio")) else 0.0,
        "sequence_audio": (
            float(event["sequence_audio"]) if _finite(event.get("sequence_audio")) else 0.0
        ),
        "tape_px": float(event["tape_px"]) if _finite(event.get("tape_px")) else 999.0,
        "observed_support_px": (
            float(event["observed_support_px"])
            if _finite(event.get("observed_support_px"))
            else 999.0
        ),
        "img_x": float(event.get("img_x", 0.0)),
        "img_y": float(event.get("img_y", 0.0)),
        "court_x": float(event.get("court_x", 0.0)),
        "court_y": float(event.get("court_y", 0.0)),
        "nearest_player_edge_distance_norm": normalized_player_edge_distance(
            boxes,
            float(event.get("candidate_frame", event["frame"])),
            float(event.get("img_x", 0.0)),
            float(event.get("img_y", 0.0)),
            fps=fps,
        ),
        "bounce_witness": int(bool(witness)),
        "bounce_confidence": float(witness.get("confidence", 0.0)),
        "bounce_kink": float(witness.get("kink_strength", 0.0)),
        "review": int(bool(event.get("review"))),
        "why": "|".join(event.get("why", [])),
    }
    for direction in ("pre", "post"):
        for window_ms in (200, 400):
            prefix = f"{direction}_{window_ms}ms"
            for suffix in (
                "observations",
                "displacement_px",
                "path_px",
                "straightness",
            ):
                row_key = f"{prefix}_{suffix}"
                value = event.get(row_key, 0.0)
                row[row_key] = float(value) if _finite(value) else 0.0
    return row


def _frame_number(value: str) -> int:
    match = re.search(r"\d+", Path(value).stem)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def _observed_track_rows(path: Path | None, clip: str) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as handle:
        return [
            {
                "frame": _frame_number(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "score": float(row.get("score") or 0.0),
                "source_count": len(set(filter(None, (row.get("sources") or "").split("+")))),
            }
            for row in csv.DictReader(handle)
            if row["clip"] == clip
        ]


def _observed_net_support(
    rows: list[dict],
    frame: float,
    *,
    fps: float,
    projector,
    reference_scale: float,
) -> dict:
    radius = max(2, round(0.12 * fps))
    nearby = [row for row in rows if abs(row["frame"] - frame) <= radius]
    if not nearby:
        return {
            "observed_tape_px": 999.0,
            "observed_tape_frame_offset": 99.0,
            "observed_track_score": 0.0,
            "observed_track_source_count": 0,
        }
    measured = []
    for row in nearby:
        distance, _ = tape_pixel_dist(
            projector,
            row["x"],
            row["y"],
            int(row["frame"]),
        )
        measured.append((float(distance) * reference_scale, row))
    distance, row = min(
        measured,
        key=lambda item: (item[0], abs(item[1]["frame"] - frame)),
    )
    return {
        "observed_tape_px": distance,
        "observed_tape_frame_offset": float(row["frame"] - frame),
        "observed_track_score": row["score"],
        "observed_track_source_count": row["source_count"],
    }


def _add_anchor(rows: list[dict], base: dict, source: str, frame: float) -> None:
    row = {
        **base,
        "proposal_source": source,
        "proposal_frame": float(frame),
        "anchor_offset": float(frame) - float(base["candidate_frame"]),
    }
    for candidate_source in SOURCE_PRIORITY:
        row[f"source_{candidate_source}"] = int(source == candidate_source)
    rows.append(row)


def generate_clip(
    clip: str,
    *,
    fps: float,
    processed: Path,
    camera: Path,
    ball: Path,
    observed_ball: Path | None = None,
    boxes_path: Path,
    audio: Path,
) -> list[dict]:
    frames, xs, ys, _ = load_track(str(ball), clip)
    boxes = load_boxes(str(boxes_path), clip)
    hints = auto_serve_hints(frames, xs, ys, boxes, fps)
    pose = processed / f"player_pose_kp17_{clip}_v1.csv"
    pose_path = str(pose) if pose.exists() else None
    serves = serve_strike_frames(
        clip,
        str(camera),
        str(ball),
        str(boxes_path),
        pose_path,
        str(audio),
        hints,
        fps=fps,
    )
    events, projector = detect_events(
        clip,
        str(camera),
        str(ball),
        str(boxes_path),
        pose_path,
        str(audio),
        serves,
        fps=fps,
    )
    observed_rows = _observed_track_rows(observed_ball, clip)
    observed_projector = (
        projector_for_artifact(str(camera), clip, observed_ball)
        if observed_ball is not None and observed_ball.exists()
        else None
    )
    reference_scale = 1.0
    if observed_ball is not None and observed_ball.exists():
        sidecar = coordinate_manifest_path(observed_ball)
        if sidecar.exists():
            coordinate_space = json.loads(sidecar.read_text())
            reference_scale = 960.0 / float(coordinate_space["artifact_size"]["width"])
    for event in events:
        event["candidate_frame"] = float(event["frame"])
        event["initial_type"] = event["type"]
        event["initial_confidence"] = float(event["confidence"])
    play_lo, play_hi, _ = gate_play_phase(
        events,
        serves,
        frames,
        xs,
        ys,
        projector,
        fps=fps,
    )

    rows = []
    for index, event in enumerate(events):
        base = _base_row(clip, event, index, boxes=boxes, fps=fps)
        if observed_projector is not None:
            base.update(
                _observed_net_support(
                    observed_rows,
                    base["candidate_frame"],
                    fps=fps,
                    projector=observed_projector,
                    reference_scale=reference_scale,
                )
            )
        else:
            base.update(
                {
                    "observed_tape_px": 999.0,
                    "observed_tape_frame_offset": 99.0,
                    "observed_track_score": 0.0,
                    "observed_track_source_count": 0,
                }
            )
        _add_anchor(rows, base, "trajectory", base["candidate_frame"])
        for frame in event.get("suppressed_candidate_frames", []):
            _add_anchor(rows, base, "suppressed_trajectory", float(frame))
        proposal_frame = event.get("proposal_frame")
        if event.get("proposal_type") == "bounce" and _finite(proposal_frame):
            _add_anchor(rows, base, "bounce_witness", float(proposal_frame))
        if _finite(event.get("audio_frame")) and base["audio"] > 0.0:
            _add_anchor(rows, base, "local_audio", float(event["audio_frame"]))
        if _finite(event.get("sequence_audio_frame")) and base["sequence_audio"] > 0.0:
            _add_anchor(
                rows,
                base,
                "sequence_audio",
                float(event["sequence_audio_frame"]),
            )
    gap_index = 0
    runs = split_runs(frames, xs, ys)
    for left, right in zip(runs, runs[1:]):
        gap_lo = float(frames[left[-1]])
        gap_hi = float(frames[right[0]])
        if gap_hi - gap_lo <= 1.0:
            continue
        anchor = gap_lo
        proposal_step = max(1.0, 0.1 * fps)
        while anchor < gap_hi:
            anchor = min(anchor + proposal_step, gap_hi)
            nearest = min(
                rows,
                key=lambda row: abs(float(row["candidate_frame"]) - anchor),
                default=None,
            )
            if nearest is None:
                continue
            base = {
                key: value
                for key, value in nearest.items()
                if not key.startswith("source_")
                and key not in {"proposal_source", "proposal_frame", "anchor_offset"}
            }
            base.update(
                {
                    "candidate_id": f"{clip}_g{gap_index:03d}",
                    "candidate_frame": anchor,
                    "kind": "track_gap",
                    "nearest_player_edge_distance_norm": MISSING_PLAYER_DISTANCE,
                    "in_play": int(play_lo <= anchor <= play_hi),
                    "excluded": "" if play_lo <= anchor <= play_hi else "outside_play_window",
                    "why": "track_gap_grid",
                }
            )
            _add_anchor(rows, base, "track_gap", anchor)
            gap_index += 1
    for index, frame in enumerate(serves):
        base = {key: 0 for key in rows[0] if key not in {"clip", "candidate_id"}} if rows else {}
        base.update(
            {
                "clip": clip,
                "candidate_id": f"{clip}_s{index:02d}",
                "candidate_frame": float(frame),
                "initial_type": "serve",
                "final_type": "racket_hit",
                "kind": "serve",
                "nearest_player_edge_distance_norm": MISSING_PLAYER_DISTANCE,
                "in_play": 1,
                "excluded": "",
                "why": "serve_detector",
            }
        )
        _add_anchor(rows, base, "serve", float(frame))
    return rows


def _dedupe(rows: list[dict], sources: set[str], in_play_only: bool) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["proposal_source"] in sources and (not in_play_only or row["in_play"])
    ]
    output = []
    for row in sorted(
        selected,
        key=lambda item: (
            item["clip"],
            float(item["proposal_frame"]),
            SOURCE_PRIORITY[item["proposal_source"]],
        ),
    ):
        duplicate = next(
            (
                prior
                for prior in reversed(output)
                if prior["clip"] == row["clip"]
                and abs(float(prior["proposal_frame"]) - float(row["proposal_frame"])) <= 0.5
            ),
            None,
        )
        if duplicate is None:
            output.append(row)
    return output


def main() -> None:
    """Write a label-free proposal receipt for the supplied pipeline root."""

    import argparse

    from cv.pipeline.event_model_v2_features import TRACK_NAME

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-name", default=TRACK_NAME)
    parser.add_argument("--measured-current-track", action="store_true")
    parser.add_argument(
        "--pose-root",
        type=Path,
        action="append",
        dest="pose_roots",
        help=(
            "explicit root holding corrected native pose artifacts; repeat for partitioned "
            "cohorts; roots are searched in command-line order and never discovered ambiently"
        ),
    )
    parser.add_argument("--pose-name", action="append", dest="pose_names")
    parser.add_argument("--corner-min-degrees", type=float, default=110.0)
    parser.add_argument(
        "--departure-k",
        type=float,
        default=IMAGE_DEPARTURE_K,
        help="departure size in standard deviations of the local fit's prediction interval",
    )
    args = parser.parse_args()
    document = generate_proposal_document(
        args.root,
        args.output,
        track_name=args.track_name,
        pose_roots=tuple(args.pose_roots or ()),
        pose_names=tuple(args.pose_names) if args.pose_names else POSE_NAMES,
        corner_min_degrees=args.corner_min_degrees,
        departure_k=args.departure_k,
        measured_current_track=args.measured_current_track,
    )
    print(
        json.dumps({key: value for key, value in document.items() if key != "proposals"}, indent=2)
    )


if __name__ == "__main__":
    main()
