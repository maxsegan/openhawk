"""ByteTrack/BoT-SORT style tracking of the two on-court players.

The default path used to link boxes with the greedy nearest-neighbour matcher in
:mod:`cv.pipeline.player_court_v2`, which produced 2.5 track ids per player per point
(76% of clip-sides split) and rejected non-players mainly by how far a tracklet moved.
This module replaces both halves:

*Association* is a two-stage ByteTrack cascade with BoT-SORT appearance. Motion lives in
COURT metres, not pixels, so the per-frame homography absorbs the camera pan and no
explicit camera-motion compensation is needed. Each track carries a constant-velocity
Kalman filter; the assignment cost mixes the Mahalanobis motion distance, image-space IoU
and cosine distance between appearance embeddings, solved with the Hungarian algorithm.
High-confidence detections match first, low-confidence ones then rescue the far player
whose small box regularly scores below the detector threshold, and detections that match
nothing are offered to recently lost tracks by appearance before starting a new track.

*Non-player rejection* is by court position, body size and persistence — never by path
length. Ball kids and net-post staff sit outside the lateral gate; seated crowd and
crouching kids fail the size gate, which normalises box height by the local ground scale
of the homography and compares it against the clip's own robust median (a standing adult
at that depth); line judges behind the baseline fail the depth gate because a player's
median depth sits within a couple of metres of their own baseline. Short spurious tracks
fail the persistence gate.

The tracker then stitches surviving tracks into identities (reachable at V_MAX across the
gap AND appearance-compatible) and keeps exactly one identity per side, so a point yields
one near track and one far track with explicit gaps where nobody was detected.

``SelectionConfig.keep_unique_admissible_frames`` is production default-on: after
winner-take-all, unique frames of other admissible same-side tracks are kept. Rollback:
set that field to ``False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.pipeline.camera_cal import COURT_L, COURT_W, NET_Y

# --------------------------------------------------------------------------- geometry

COURT_X_MARGIN_M = 2.5  # detection admission gate, unchanged from v1/v2
BASELINE_MARGIN_M = 8.0
STANDING_HEIGHT_M = 1.80  # nominal adult height used only to normalise box height


def ground_scales(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Native pixels per court metre at each ground point (vectorised).

    A standing person's image height scales with the same ground scale, up to a factor
    that is near constant over a broadcast clip, so dividing box height by
    ``STANDING_HEIGHT_M * scale`` gives a depth-free body-size measure. Measured on the
    week-one sided boxes the ratio sits at 0.89-1.07 median across matches and sides,
    which is why the size gate needs no per-clip calibration.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    try:
        world_to_image = np.linalg.inv(np.asarray(homography, dtype=float))
    except np.linalg.LinAlgError:
        return np.full(len(points), np.nan)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    shifted = homogeneous.copy()
    shifted[:, 0] += 0.5
    projected = (world_to_image @ homogeneous.T).T
    projected_shift = (world_to_image @ shifted.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        base = projected[:, :2] / projected[:, 2:3]
        offset = projected_shift[:, :2] / projected_shift[:, 2:3]
    scale = np.hypot(*(offset - base).T) * 2.0
    scale[~np.isfinite(scale)] = np.nan
    return scale


def local_ground_scale(homography: np.ndarray, court_x: float, court_y: float) -> float:
    return float(ground_scales(homography, [[court_x, court_y]])[0])


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class Detection:
    """One person box on one frame, already projected to the court plane."""

    frame: int
    box: tuple[float, float, float, float]  # native pixels, x0 y0 x1 y1
    conf: float
    court: tuple[float, float]  # metres, box-bottom-centre through that frame's H
    ground_scale: float = float("nan")  # native px per court metre at the foot point
    embedding: np.ndarray | None = None
    row: dict | None = None  # the source CSV row, carried through untouched

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def size_ratio(self) -> float:
        """Box height over the height a standing adult would have at that depth."""
        if not math.isfinite(self.ground_scale) or self.ground_scale <= 0:
            return float("nan")
        return self.height / (STANDING_HEIGHT_M * self.ground_scale)


@dataclass
class TrackerConfig:
    high_conf: float = 0.45
    low_conf: float = 0.10
    sigma_measure_m: float = 0.30
    sigma_accel: float = 8.0
    v_max_ms: float = 9.0
    gate_slack_m: float = 1.00  # court-space jitter allowance on top of V_MAX * dt
    max_age_s: float = 1.5  # a lost track stays re-identifiable this long
    confirm_hits: int = 3
    weight_motion: float = 1.0
    weight_iou: float = 0.6
    weight_appearance: float = 1.2
    appearance_gate: float = 0.55  # cosine distance ceiling for an association
    appearance_momentum: float = 0.9
    match_cost_max: float = 1.4
    # Default-off guard: at lost-track revival only, refuse an identity whose measured
    # body-size history is an established different size class from the incoming
    # leftover detection. Never changes active or low-confidence matching.
    lost_revival_body_history: bool = False


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    embedding: np.ndarray | None = None
    hits: int = 0
    last_frame: int = -1
    start_frame: int = -1
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    stitch_boundaries: list[dict] = field(default_factory=list)
    # Export-only identities for independently associated reliable-view spans.
    # Empty for ordinary tracks; never used as an association or merge witness.
    association_ids: dict[int, int] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return [d.frame for d in self.detections]

    @property
    def span(self) -> int:
        return self.last_frame - self.start_frame + 1

    def court_median(self) -> tuple[float, float]:
        points = np.array([d.court for d in self.detections], dtype=float)
        return float(np.median(points[:, 0])), float(np.median(points[:, 1]))

    def size_ratio_median(self) -> float:
        ratios = [d.size_ratio for d in self.detections if math.isfinite(d.size_ratio)]
        return float(np.median(ratios)) if ratios else float("nan")

    def conf_median(self) -> float:
        return float(np.median([d.conf for d in self.detections]))

    def mean_embedding(self) -> np.ndarray | None:
        vectors = [d.embedding for d in self.detections if d.embedding is not None]
        if not vectors:
            return None
        stacked = np.asarray(vectors, dtype=np.float32)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-6 else None


# --------------------------------------------------------------------------- Kalman


def _transition(dt: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 2] = dt
    matrix[1, 3] = dt
    return matrix


def _process_noise(dt: float, sigma_accel: float) -> np.ndarray:
    g = np.array([0.5 * dt * dt, 0.5 * dt * dt, dt, dt])
    return np.diag(g * g) * sigma_accel * sigma_accel


def _predict(mean: np.ndarray, covariance: np.ndarray, dt: float, sigma_accel: float):
    transition = _transition(dt)
    mean = transition @ mean
    covariance = transition @ covariance @ transition.T + _process_noise(dt, sigma_accel)
    return mean, covariance


def _update(mean: np.ndarray, covariance: np.ndarray, measurement, sigma: float):
    observation = np.zeros((2, 4))
    observation[0, 0] = observation[1, 1] = 1.0
    innovation_cov = observation @ covariance @ observation.T + np.eye(2) * sigma * sigma
    gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
    residual = np.asarray(measurement, dtype=float) - observation @ mean
    mean = mean + gain @ residual
    covariance = (np.eye(4) - gain @ observation) @ covariance
    return mean, covariance


def _gating_distance(mean, covariance, measurement, sigma: float) -> float:
    observation = np.zeros((2, 4))
    observation[0, 0] = observation[1, 1] = 1.0
    innovation_cov = observation @ covariance @ observation.T + np.eye(2) * sigma * sigma
    residual = np.asarray(measurement, dtype=float) - observation @ mean
    return float(residual @ np.linalg.inv(innovation_cov) @ residual)


CHI2_GATE_2DOF = 9.21  # 0.99 quantile


# --------------------------------------------------------------------------- association


def _cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(1.0 - np.clip(np.dot(a.astype(np.float32), b.astype(np.float32)), -1.0, 1.0))


# ------------------------------------------------------- lost-revival body history guard


@dataclass
class RevivalBodyHistoryAudit:
    """Run-level evidence for the lost-revival body-history guard.

    Counters and a bounded record list live at run level, so a refusal survives even when
    the refused identity is never selected (or never becomes a kept track at all).
    Uninformative pairs are counted, not recorded.
    """

    enabled: bool = False
    record_limit: int = 64
    revival_pairs_considered: int = 0
    revival_refusals: int = 0
    revival_established_history: int = 0
    revival_unsupported_scale: int = 0
    active_size_class_pairs: int = 0  # eligible pairs, not necessarily assigned transitions
    clip: str | None = None
    observed_frame_range: tuple[int, int] | None = None
    records: list[dict] = field(default_factory=list)
    records_truncated: int = 0

    def note(self, **fields) -> None:
        if len(self.records) < self.record_limit:
            self.records.append(
                {"clip": self.clip, "observed_frame_range": self.observed_frame_range, **fields}
            )
        else:
            self.records_truncated += 1

    def summary(self) -> dict:
        return {
            "schema": "lost_revival_body_history_audit_v1",
            "enabled": self.enabled,
            "revival_pairs_considered": self.revival_pairs_considered,
            "revival_refusals": self.revival_refusals,
            "revival_established_history": self.revival_established_history,
            "revival_unsupported_scale": self.revival_unsupported_scale,
            "active_size_class_pairs": self.active_size_class_pairs,
            "records": list(self.records),
            "records_truncated": self.records_truncated,
        }


def _in_band(ratio: float, selection: "SelectionConfig") -> bool:
    return bool(
        math.isfinite(ratio)
        and ratio > 0
        and selection.size_ratio_low <= ratio <= selection.size_ratio_high
    )


def body_history_class(track: Track, selection: "SelectionConfig") -> tuple[str, float, int, int]:
    """Classify an identity's prior measured body history against the existing size band.

    Only finite positive perspective-normalized sizes are evidence; anything else is unknown.
    Returns ``(class, median_of_finite_samples, finite_samples, in_band_samples)``.
    """
    ratios = [
        d.size_ratio for d in track.detections if math.isfinite(d.size_ratio) and d.size_ratio > 0
    ]
    in_band = [r for r in ratios if _in_band(r, selection)]
    median = float(np.median(ratios)) if ratios else float("nan")
    if len(in_band) >= selection.min_hits:
        return "established", median, len(ratios), len(in_band)
    if len(ratios) < selection.min_hits:
        return "unsupported", median, len(ratios), len(in_band)
    if median < selection.size_ratio_low:
        return "undersized", median, len(ratios), len(in_band)
    if median > selection.size_ratio_high:
        return "oversized", median, len(ratios), len(in_band)
    return "inconclusive", median, len(ratios), len(in_band)


def revival_body_history_refusal(
    track: Track, detection: Detection, selection: "SelectionConfig"
) -> tuple[bool, str, float, int, int]:
    """Whether this lost identity refuses this leftover detection on body history."""
    history, median, finite, in_band = body_history_class(track, selection)
    refuse = history in {"undersized", "oversized"} and _in_band(detection.size_ratio, selection)
    return refuse, history, median, finite, in_band


@dataclass(frozen=True)
class _BodyHistoryGuard:
    """How one matching pass treats body history: refuse at revival, audit elsewhere."""

    selection: "SelectionConfig"
    audit: "RevivalBodyHistoryAudit | None" = None
    refuse: bool = False


def _cost(
    track: Track,
    detection: Detection,
    dt: float,
    config: TrackerConfig,
    *,
    use_appearance: bool,
    motion_gate_m: float | None = None,
) -> float:
    predicted = _predict(track.mean, track.covariance, dt, config.sigma_accel)
    reach = config.v_max_ms * dt + config.gate_slack_m
    limit = reach if motion_gate_m is None else motion_gate_m
    straight = float(np.hypot(*(np.asarray(detection.court) - track.mean[:2])))
    if straight > limit:
        return float("inf")
    mahalanobis = _gating_distance(
        predicted[0], predicted[1], detection.court, config.sigma_measure_m
    )
    if mahalanobis > CHI2_GATE_2DOF * 4:
        return float("inf")
    motion = min(1.0, math.sqrt(mahalanobis) / math.sqrt(CHI2_GATE_2DOF))
    overlap = 1.0 - iou(track.box, detection.box)
    appearance_distance = _cosine_distance(track.embedding, detection.embedding)
    total = config.weight_motion * motion + config.weight_iou * overlap
    weight = config.weight_motion + config.weight_iou
    if use_appearance and math.isfinite(appearance_distance):
        if appearance_distance > config.appearance_gate:
            return float("inf")
        total += config.weight_appearance * appearance_distance
        weight += config.weight_appearance
    return total / weight


def _match(
    tracks,
    detections,
    frame,
    config,
    *,
    use_appearance,
    motion_gate_m=None,
    fps,
    guard: "_BodyHistoryGuard | None" = None,
):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    costs = np.full((len(tracks), len(detections)), np.inf)
    for i, track in enumerate(tracks):
        dt = max(frame - track.last_frame, 1) / fps
        for j, detection in enumerate(detections):
            costs[i, j] = _cost(
                track,
                detection,
                dt,
                config,
                use_appearance=use_appearance,
                motion_gate_m=motion_gate_m,
            )
            # Body history is consulted only after the original geometric and appearance
            # gates already admitted the pair: no previously infinite cost becomes finite.
            if guard is None or not math.isfinite(costs[i, j]):
                continue
            refuse, history, median, finite, in_band = revival_body_history_refusal(
                track, detection, guard.selection
            )
            audit = guard.audit
            if not guard.refuse:
                if refuse and audit is not None:
                    # The analogous active transition: audited to show this guard's
                    # narrow reach, never blocked.
                    audit.active_size_class_pairs += 1
                continue
            if audit is not None:
                audit.revival_pairs_considered += 1
                if history == "established":
                    audit.revival_established_history += 1
                elif history == "unsupported":
                    audit.revival_unsupported_scale += 1
            if refuse:
                if audit is not None:
                    audit.revival_refusals += 1
                    audit.note(
                        kind="revival_refusal",
                        frame=int(frame),
                        track_id=int(track.track_id),
                        candidate_box_native=[float(v) for v in detection.box],
                        candidate_court_xy_m=[float(v) for v in detection.court],
                        candidate_confidence=float(detection.conf),
                        previous_box_native=[float(v) for v in track.box],
                        previous_court_xy_m=[float(v) for v in track.detections[-1].court],
                        prior_observation_frame=int(track.last_frame),
                        boundary_distance_m=float(
                            np.linalg.norm(np.asarray(detection.court) - track.detections[-1].court)
                        ),
                        track_start_frame=int(track.start_frame),
                        track_last_frame=int(track.last_frame),
                        history=history,
                        history_median_size_ratio=round(median, 4),
                        finite_size_samples=int(finite),
                        in_band_size_samples=int(in_band),
                        detection_size_ratio=round(float(detection.size_ratio), 4),
                        refused_cost=round(float(costs[i, j]), 4),
                    )
                costs[i, j] = float("inf")
    finite = np.isfinite(costs)
    if not finite.any():
        return [], list(range(len(tracks))), list(range(len(detections)))
    padded = np.where(finite, costs, 1e6)
    rows, cols = linear_sum_assignment(padded)
    matches, used_tracks, used_detections = [], set(), set()
    for i, j in zip(rows, cols, strict=True):
        if finite[i, j] and costs[i, j] <= config.match_cost_max:
            matches.append((i, j))
            used_tracks.add(i)
            used_detections.add(j)
    return (
        matches,
        [i for i in range(len(tracks)) if i not in used_tracks],
        [j for j in range(len(detections)) if j not in used_detections],
    )


def _absorb(track: Track, detection: Detection, config: TrackerConfig, fps: float) -> None:
    dt = max(detection.frame - track.last_frame, 1) / fps
    if track.hits == 0:
        track.mean = np.array([detection.court[0], detection.court[1], 0.0, 0.0])
        track.covariance = np.diag([0.5, 0.5, 4.0, 4.0])
        track.start_frame = detection.frame
    else:
        track.mean, track.covariance = _predict(
            track.mean, track.covariance, dt, config.sigma_accel
        )
        track.mean, track.covariance = _update(
            track.mean, track.covariance, detection.court, config.sigma_measure_m
        )
    track.detections.append(detection)
    track.hits += 1
    track.last_frame = detection.frame
    track.box = detection.box
    if detection.embedding is not None and detection.conf >= config.high_conf:
        if track.embedding is None:
            track.embedding = detection.embedding.astype(np.float32)
        else:
            blended = config.appearance_momentum * track.embedding + (
                1.0 - config.appearance_momentum
            ) * detection.embedding.astype(np.float32)
            norm = float(np.linalg.norm(blended))
            track.embedding = blended / norm if norm > 1e-6 else track.embedding


def track_detections(
    detections: list[Detection],
    *,
    fps: float,
    config: TrackerConfig | None = None,
    selection: "SelectionConfig | None" = None,
    audit: RevivalBodyHistoryAudit | None = None,
) -> list[Track]:
    """Run the association cascade over one clip's detections.

    ``selection`` only supplies the existing body-size band and ``min_hits`` used by the
    default-off lost-revival body-history guard; callers without one get the ordinary
    defaults. ``audit`` collects run-level guard evidence and is never read back.
    """
    config = config or TrackerConfig()
    selection = selection or SelectionConfig()
    if audit is not None:
        audit.enabled = audit.enabled or config.lost_revival_body_history
    guard = (
        _BodyHistoryGuard(selection=selection, audit=audit, refuse=True)
        if config.lost_revival_body_history
        else None
    )
    active_guard = (
        _BodyHistoryGuard(selection=selection, audit=audit, refuse=False)
        if config.lost_revival_body_history
        else None
    )
    by_frame: dict[int, list[Detection]] = {}
    for detection in detections:
        if detection.conf >= config.low_conf:
            by_frame.setdefault(detection.frame, []).append(detection)
    max_age = max(1, int(round(config.max_age_s * fps)))
    active: list[Track] = []
    lost: list[Track] = []
    finished: list[Track] = []
    next_id = 0
    for frame in sorted(by_frame):
        rows = by_frame[frame]
        high = [d for d in rows if d.conf >= config.high_conf]
        low = [d for d in rows if d.conf < config.high_conf]

        matches, unmatched_tracks, unmatched_high = _match(
            active, high, frame, config, use_appearance=True, fps=fps, guard=active_guard
        )
        for i, j in matches:
            _absorb(active[i], high[j], config, fps)

        pending = [active[i] for i in unmatched_tracks]
        confirmed = [t for t in pending if t.hits >= config.confirm_hits]
        low_matches, _, _ = _match(confirmed, low, frame, config, use_appearance=False, fps=fps)
        matched_low = set()
        for i, j in low_matches:
            _absorb(confirmed[i], low[j], config, fps)
            matched_low.add(j)

        # Re-identify: a detection nothing owns may belong to a recently lost track.
        leftovers = [high[j] for j in unmatched_high]
        revive, _, still_new = _match(
            lost,
            leftovers,
            frame,
            config,
            use_appearance=True,
            motion_gate_m=config.v_max_ms * config.max_age_s + config.gate_slack_m,
            fps=fps,
            guard=guard,
        )
        revived = set()
        for i, j in revive:
            _absorb(lost[i], leftovers[j], config, fps)
            active.append(lost[i])
            revived.add(i)
        lost = [track for index, track in enumerate(lost) if index not in revived]

        for j in still_new:
            track = Track(track_id=next_id)
            next_id += 1
            _absorb(track, leftovers[j], config, fps)
            active.append(track)

        keep: list[Track] = []
        for track in active:
            if track.last_frame == frame:
                keep.append(track)
            else:
                lost.append(track)
        active = keep
        finished.extend(t for t in lost if frame - t.last_frame > max_age)
        lost = [t for t in lost if frame - t.last_frame <= max_age]
    finished.extend(active)
    finished.extend(lost)
    unique = {id(t): t for t in finished}
    return sorted(unique.values(), key=lambda t: (t.start_frame, t.track_id))


# --------------------------------------------------------------------------- selection


@dataclass
class SelectionConfig:
    lateral_max_m: float = 1.0  # median court x must sit within this of the doubles court
    net_zone_depth_m: float = 3.5  # within this of the net a player is inside the sidelines
    net_zone_lateral_max_m: float = 0.2  # net-post ball kids and staff live just outside
    depth_slack_m: float = 6.0  # median depth may sit this far behind the own baseline
    depth_preferred_m: float = 4.0  # beyond this the score prefers someone further forward
    depth_preference_weight: float = 0.5
    net_slack_m: float = 0.5  # ... and no closer than this to the net on the wrong side
    size_ratio_low: float = 0.62  # crouching ball kid / seated crowd
    size_ratio_high: float = 1.45  # merged boxes, umpire chair, motion blur smear
    min_coverage: float = 0.06  # a real player is seen on at least this share of frames
    min_hits: int = 6
    stitch_gap_s: float = 8.0
    stitch_reach_floor_s: float = 0.4  # a stitch may always close V_MAX * this distance
    stitch_appearance_max: float = 0.60
    recovery_appearance_max: float = 0.20
    recovery_overlap_max: float = 0.75
    stitch_boundary_iou_min: float = 0.5
    # Production default-on. After winner-take-all, keep unique frames of other
    # admissible same-side tracks. A player who starts several metres behind the
    # baseline and later plays closer is often two ByteTrack identities; the
    # closer one wins and the deep prefix is dropped. Unique frames of the loser
    # are still the player. Line judges fail the existing depth/size gates and
    # are not kept. Rollback: set this field to False.
    keep_unique_admissible_frames: bool = True


def side_of(court_y: float) -> str:
    return "near" if court_y < NET_Y else "far"


def lateral_violation(court_x: float) -> float:
    return max(0.0, -court_x, court_x - COURT_W)


def depth_violation(court_y: float, side: str, selection: SelectionConfig) -> float:
    if side == "far":
        return max(
            0.0,
            court_y - (COURT_L + selection.depth_slack_m),
            (NET_Y + selection.net_slack_m) - court_y,
        )
    return max(
        0.0,
        court_y - (NET_Y - selection.net_slack_m),
        (-selection.depth_slack_m) - court_y,
    )


def baseline_excess(court_y: float, side: str) -> float:
    """How far behind their own baseline the position sits, in metres (0 inside)."""
    return max(0.0, (court_y - COURT_L) if side == "far" else -court_y)


def admissible(track: Track, *, n_frames: int, selection: SelectionConfig) -> tuple[bool, str]:
    """Court position, size and persistence gates. Path length is never consulted."""
    if track.hits < selection.min_hits:
        return False, "too_few_hits"
    if n_frames and track.hits / n_frames < selection.min_coverage:
        return False, "not_persistent"
    court_x, court_y = track.court_median()
    # Wide serves and stretched returns pull a player past the sideline, but only deep.
    # Anyone whose median position sits outside the sidelines beside the net is net-post
    # staff or a ball kid, never a player.
    lateral_limit = (
        selection.net_zone_lateral_max_m
        if abs(court_y - NET_Y) < selection.net_zone_depth_m
        else selection.lateral_max_m
    )
    if lateral_violation(court_x) > lateral_limit:
        return False, "outside_court_laterally"
    side = side_of(court_y)
    if depth_violation(court_y, side, selection) > 0:
        return False, "outside_court_in_depth"
    ratio = track.size_ratio_median()
    if math.isfinite(ratio) and not (
        selection.size_ratio_low <= ratio <= selection.size_ratio_high
    ):
        return False, "wrong_body_size"
    return True, ""


def track_score(track: Track, *, n_frames: int, selection: SelectionConfig) -> float:
    """Persistence and court centrality only — deliberately not path length."""
    court_x, court_y = track.court_median()
    side = side_of(court_y)
    coverage = track.hits / n_frames if n_frames else 0.0
    # Standing well behind the baseline all point is what a line judge does; a player
    # deep on clay is only a little behind it. This is a position preference, not a
    # movement one, so a server who never moves is not punished.
    depth_penalty = selection.depth_preference_weight * max(
        0.0, baseline_excess(court_y, side) - selection.depth_preferred_m
    )
    penalty = (
        2.0 * lateral_violation(court_x)
        + 2.0 * depth_violation(court_y, side, selection)
        + depth_penalty
    )
    return 2.0 * coverage + 0.5 * track.conf_median() - penalty


def _stitchable(
    a: Track, b: Track, *, fps: float, selection: SelectionConfig, config: TrackerConfig
) -> bool:
    first, second = (a, b) if a.last_frame <= b.start_frame else (b, a)
    if second.start_frame < first.last_frame:
        return False
    gap_frames = second.start_frame - first.last_frame
    gap_seconds = gap_frames / fps
    if gap_seconds > selection.stitch_gap_s:
        return False
    distance = float(
        np.hypot(*(np.asarray(second.detections[0].court) - np.asarray(first.detections[-1].court)))
    )
    # One shared exposure is not a temporal gap: two different people cannot
    # become one merely because the ordinary gap-reach allowance connects them.
    if gap_frames == 0 and (
        distance > config.gate_slack_m
        or iou(first.detections[-1].box, second.detections[0].box)
        < selection.stitch_boundary_iou_min
    ):
        return False
    reach = config.v_max_ms * max(gap_seconds, selection.stitch_reach_floor_s)
    if distance > reach + config.gate_slack_m:
        return False
    appearance = _cosine_distance(a.mean_embedding(), b.mean_embedding())
    if math.isfinite(appearance) and appearance > selection.stitch_appearance_max:
        return False
    return True


def stitch(
    tracks: list[Track], *, fps: float, selection: SelectionConfig, config: TrackerConfig
) -> list[Track]:
    """Merge compatible fragments; reconcile an agreeing shared boundary exactly once."""
    merged = sorted(tracks, key=lambda t: t.start_frame)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if not _stitchable(
                    merged[i], merged[j], fps=fps, selection=selection, config=config
                ):
                    continue
                first, second = merged[i], merged[j]
                if first.start_frame > second.start_frame:
                    first, second = second, first
                detections = first.detections + second.detections
                boundaries = [*first.stitch_boundaries, *second.stitch_boundaries]
                if first.last_frame == second.start_frame:
                    earlier, later = first.detections[-1], second.detections[0]
                    # Keep an original observation, never an averaged synthetic box.
                    # Equal confidence retains the earlier fragment's boundary.
                    choose_later = later.conf > earlier.conf
                    chosen = later if choose_later else earlier
                    detections = [*first.detections[:-1], chosen, *second.detections[1:]]
                    boundaries.append(
                        {
                            "frame": chosen.frame,
                            "policy": "agreeing_shared_exposure_highest_confidence_earlier_on_tie_v1",
                            "chosen_fragment": "later" if choose_later else "earlier",
                            "box_iou": iou(earlier.box, later.box),
                            "candidates": [
                                {
                                    "track_id": t.track_id,
                                    "box_native": list(d.box),
                                    "court_xy_m": list(d.court),
                                    "confidence": d.conf,
                                }
                                for t, d in ((first, earlier), (second, later))
                            ],
                        }
                    )
                combined = Track(
                    track_id=first.track_id,
                    detections=detections,
                    mean=second.mean,
                    covariance=second.covariance,
                    embedding=second.embedding if second.embedding is not None else first.embedding,
                    hits=len(detections),
                    last_frame=second.last_frame,
                    start_frame=first.start_frame,
                    box=second.box,
                    stitch_boundaries=boundaries,
                )
                merged = [t for k, t in enumerate(merged) if k not in (i, j)] + [combined]
                merged.sort(key=lambda t: t.start_frame)
                changed = True
                break
            if changed:
                break
    return merged


def recover_identity_fragments(
    anchor: Track,
    tracks: list[Track],
    *,
    fps: float,
    selection: SelectionConfig,
    config: TrackerConfig,
) -> Track:
    """Fill anchor gaps from reachable, appearance-compatible track fragments.

    The position, size and persistence gates establish the identity anchor. Applying
    those same whole-track gates to every fragment loses valid player detections when a
    noisy homography or detector dropout splits the track. A rejected fragment may fill
    frames missing from the anchor, but it never replaces an anchored frame.
    """
    anchor_frames = set(anchor.frames)
    anchor_embedding = anchor.mean_embedding()
    candidates: list[tuple[float, Track]] = []
    anchor_side = side_of(anchor.court_median()[1])
    for candidate in tracks:
        if candidate is anchor or side_of(candidate.court_median()[1]) != anchor_side:
            continue
        candidate_frames = set(candidate.frames)
        overlap = len(anchor_frames & candidate_frames) / max(
            1, min(len(anchor_frames), len(candidate_frames))
        )
        if overlap > selection.recovery_overlap_max:
            continue
        appearance = _cosine_distance(anchor_embedding, candidate.mean_embedding())
        if not math.isfinite(appearance) or appearance > selection.recovery_appearance_max:
            continue
        if candidate.start_frame > anchor.last_frame:
            gap_s = (candidate.start_frame - anchor.last_frame) / fps
            distance = float(
                np.hypot(
                    *(
                        np.asarray(candidate.detections[0].court)
                        - np.asarray(anchor.detections[-1].court)
                    )
                )
            )
        elif anchor.start_frame > candidate.last_frame:
            gap_s = (anchor.start_frame - candidate.last_frame) / fps
            distance = float(
                np.hypot(
                    *(
                        np.asarray(anchor.detections[0].court)
                        - np.asarray(candidate.detections[-1].court)
                    )
                )
            )
        else:
            gap_s = 0.0
            distance = 0.0
        if gap_s > selection.stitch_gap_s:
            continue
        reach = config.v_max_ms * max(gap_s, selection.stitch_reach_floor_s)
        if distance > reach + config.gate_slack_m:
            continue
        candidates.append((appearance, candidate))

    detections = list(anchor.detections)
    seen = set(anchor_frames)
    for _, candidate in sorted(
        candidates,
        key=lambda item: (item[0], -len(set(item[1].frames) - anchor_frames)),
    ):
        for detection in candidate.detections:
            if detection.frame not in seen:
                detections.append(detection)
                seen.add(detection.frame)
    detections.sort(key=lambda detection: detection.frame)
    if len(detections) == len(anchor.detections):
        return anchor
    return Track(
        track_id=anchor.track_id,
        detections=detections,
        mean=anchor.mean,
        covariance=anchor.covariance,
        embedding=anchor.embedding,
        hits=len(detections),
        last_frame=detections[-1].frame,
        start_frame=detections[0].frame,
        box=detections[-1].box,
        stitch_boundaries=list(anchor.stitch_boundaries),
    )


def recover_bilateral_fragments(
    anchor: Track,
    tracks: list[Track],
    *,
    n_frames: int,
    selection: SelectionConfig,
    config: TrackerConfig,
) -> Track:
    """Fill an anchored gap from one admissible, bilaterally agreeing fragment.

    The caller must restrict tracks to one reliable-view span. Both surrounding
    anchor observations must also occur in the candidate and agree spatially.
    Competing qualifying fragments abstain; source rows never replace anchors.
    New rows are not used as anchors for another recovery in this invocation.
    """
    by_frame = {d.frame: d for d in anchor.detections}
    candidates = [
        (track, {d.frame: d for d in track.detections})
        for track in tracks
        if track is not anchor
        and admissible(track, n_frames=n_frames, selection=selection)[0]
        and side_of(track.court_median()[1]) == side_of(anchor.court_median()[1])
    ]
    added: list[Detection] = []
    receipts = list(anchor.stitch_boundaries)
    frames = sorted(by_frame)
    for left, right in zip(frames, frames[1:]):
        if right - left <= 1:
            continue
        supported = []
        for candidate, observations in candidates:
            if left not in observations or right not in observations:
                continue
            evidence = []
            for frame in (left, right):
                a, b = by_frame[frame], observations[frame]
                overlap = iou(a.box, b.box)
                distance = float(np.linalg.norm(np.asarray(a.court) - b.court))
                if (
                    not math.isfinite(overlap)
                    or not math.isfinite(distance)
                    or overlap < selection.stitch_boundary_iou_min
                    or distance > config.gate_slack_m
                ):
                    break
                evidence.append(dict(frame=frame, box_iou=overlap, court_distance_m=distance))
            if len(evidence) != 2:
                continue
            interior = [d for d in candidate.detections if left < d.frame < right]
            if interior:
                supported.append((candidate, interior, evidence))
        if len(supported) != 1:
            continue
        candidate, interior, evidence = supported[0]
        added.extend(interior)
        receipts.append(
            dict(
                policy="unique_bilateral_overlap_original_anchors_v1",
                frame=left,
                end_frame=right,
                candidate_track_id=candidate.track_id,
                boundary_evidence=evidence,
                recovered_frames=[d.frame for d in interior],
            )
        )
    if not added:
        return anchor
    detections = sorted([*anchor.detections, *added], key=lambda d: d.frame)
    return replace(anchor, detections=detections, hits=len(detections), stitch_boundaries=receipts)


def recover_overlap_fragments(
    anchor: Track,
    witness: Track,
    tracks: list[Track],
    *,
    n_frames: int,
    selection: SelectionConfig,
    config: TrackerConfig,
    reliable_frames: set[int] | None,
    audit: list[dict] | None = None,
) -> Track:
    """Recover one measured identity extension from agreeing original overlap.

    This substitutes spatial evidence for unavailable appearance, never for conflicting
    appearance. Whole candidate and anchor spans must share uninterrupted reliable camera
    support. Existing recovered rows cannot become witnesses or replace anchor rows.
    """
    original = {d.frame: d for d in witness.detections}
    occupied_rows = {d.frame: d for d in anchor.detections}
    occupied = set(occupied_rows)
    supported = []
    for candidate in tracks:
        observations = {d.frame: d for d in candidate.detections}
        added = sorted(observations.keys() - occupied)
        shared = sorted(observations.keys() & original.keys())
        if not added or not shared:
            continue
        reason = ""
        eligible, why = admissible(candidate, n_frames=n_frames, selection=selection)
        appearance = _cosine_distance(witness.mean_embedding(), candidate.mean_embedding())
        lower = min(min(anchor.frames), min(candidate.frames), min(witness.frames))
        upper = max(max(anchor.frames), max(candidate.frames), max(witness.frames))
        if not eligible:
            reason = why
        elif side_of(candidate.court_median()[1]) != side_of(witness.court_median()[1]):
            reason = "different_side"
        elif math.isfinite(appearance):
            reason = "finite_appearance_uses_existing_recovery"
        elif reliable_frames is None or any(
            f not in reliable_frames for f in range(lower, upper + 1)
        ):
            reason = "not_one_reliable_camera_span"
        elif len(shared) < config.confirm_hits:
            reason = "insufficient_shared_observations"
        evidence = []
        if not reason:
            for frame in shared:
                a, b = original[frame], observations[frame]
                overlap = iou(a.box, b.box)
                distance = float(np.linalg.norm(np.asarray(a.court) - b.court))
                evidence.append(dict(frame=frame, box_iou=overlap, court_distance_m=distance))
                if (
                    not math.isfinite(overlap)
                    or not math.isfinite(distance)
                    or overlap < selection.stitch_boundary_iou_min
                    or distance > config.gate_slack_m
                ):
                    reason = "disagreeing_shared_observation"
            # Recovered rows cannot supply positive identity witnesses, but disagreement
            # with them must veto a splice that would jump between different people.
            for frame in sorted((observations.keys() & occupied) - original.keys()):
                a, b = occupied_rows[frame], observations[frame]
                overlap = iou(a.box, b.box)
                distance = float(np.linalg.norm(np.asarray(a.court) - b.court))
                evidence.append(
                    dict(frame=frame, box_iou=overlap, court_distance_m=distance, veto_only=True)
                )
                if (
                    not math.isfinite(overlap)
                    or not math.isfinite(distance)
                    or overlap < selection.stitch_boundary_iou_min
                    or distance > config.gate_slack_m
                ):
                    reason = "disagreeing_recovered_anchor_observation"
            # Inspect every overlapping observation, even after one disagreement.
        receipt = dict(
            policy="unique_original_overlap_fragment_v1",
            anchor_track_id=witness.track_id,
            candidate_track_id=candidate.track_id,
            shared_frame_count=len(shared),
            shared_frame_range=[shared[0], shared[-1]],
            boundary_evidence=evidence,
            recovered_frames=added,
            reason=reason or "qualified",
        )
        if audit is not None:
            audit.append(receipt)
        if not reason:
            supported.append((candidate, receipt))
    if len(supported) != 1:
        if len(supported) > 1:
            for _, receipt in supported:
                receipt["reason"] = "competing_qualified_fragments"
        return anchor
    candidate, receipt = supported[0]
    receipt["reason"] = "recovered"
    detections = sorted(
        [*anchor.detections, *(d for d in candidate.detections if d.frame not in occupied)],
        key=lambda d: d.frame,
    )
    return replace(
        anchor,
        detections=detections,
        hits=len(detections),
        start_frame=detections[0].frame,
        last_frame=detections[-1].frame,
        box=detections[-1].box,
        stitch_boundaries=[*anchor.stitch_boundaries, dict(receipt)],
    )


def select_players(
    tracks: list[Track],
    *,
    fps: float,
    n_frames: int,
    selection: SelectionConfig | None = None,
    config: TrackerConfig | None = None,
    bilateral_overlap: bool = False,
    overlap_recovery: bool = False,
    reliable_frames: set[int] | None = None,
    overlap_audit: list[dict] | None = None,
) -> dict[str, Track]:
    """Exactly one track per side, or a missing side when nothing is admissible."""
    selection = selection or SelectionConfig()
    config = config or TrackerConfig()
    keep = [t for t in tracks if admissible(t, n_frames=n_frames, selection=selection)[0]]
    all_by_side: dict[str, list[Track]] = {"near": [], "far": []}
    for track in tracks:
        all_by_side[side_of(track.court_median()[1])].append(track)
    by_side: dict[str, list[Track]] = {"near": [], "far": []}
    for track in keep:
        by_side[side_of(track.court_median()[1])].append(track)
    chosen: dict[str, Track] = {}
    for side, group in by_side.items():
        if not group:
            continue
        stitched = stitch(group, fps=fps, selection=selection, config=config)
        best = max(stitched, key=lambda t: track_score(t, n_frames=n_frames, selection=selection))
        original_witness = best
        best = recover_identity_fragments(
            best,
            all_by_side[side],
            fps=fps,
            selection=selection,
            config=config,
        )
        if bilateral_overlap:
            best = recover_bilateral_fragments(
                best, all_by_side[side], n_frames=n_frames, selection=selection, config=config
            )
        if overlap_recovery:
            best = recover_overlap_fragments(
                best,
                original_witness,
                all_by_side[side],
                n_frames=n_frames,
                selection=selection,
                config=config,
                reliable_frames=reliable_frames,
                audit=overlap_audit,
            )
        if selection.keep_unique_admissible_frames:
            best = keep_unique_admissible_frames(
                best,
                all_by_side[side],
                n_frames=n_frames,
                selection=selection,
            )
        best.detections.sort(key=lambda d: d.frame)
        chosen[side] = best
    return chosen


def keep_unique_admissible_frames(
    anchor: Track,
    tracks: list[Track],
    *,
    n_frames: int,
    selection: SelectionConfig,
) -> Track:
    """Append unique frames of other admissible same-side tracks; never replace.

    Production default-on via ``SelectionConfig.keep_unique_admissible_frames``.
    Winner-take-all already chose ``anchor``. A second admissible identity of the
    same side whose frames are not in the anchor is kept as extra observations on
    the same export track. Fragments that fail the ordinary position/size/persistence
    gates (line judges, ball kids, one-hit noise) are refused. Overlapping frames
    stay with the anchor. Rollback: ``SelectionConfig.keep_unique_admissible_frames = False``.
    """
    covered = set(anchor.frames)
    extra: list[Detection] = []
    extra_ids: dict[int, int] = {}
    receipts = list(anchor.stitch_boundaries)
    anchor_side = side_of(anchor.court_median()[1])
    for candidate in tracks:
        if candidate is anchor or side_of(candidate.court_median()[1]) != anchor_side:
            continue
        unique = [d for d in candidate.detections if d.frame not in covered]
        if len(unique) < selection.min_hits:
            continue
        fragment = Track(
            track_id=candidate.track_id,
            detections=unique,
            hits=len(unique),
            start_frame=unique[0].frame,
            last_frame=unique[-1].frame,
            box=unique[-1].box,
        )
        ok, _reason = admissible(fragment, n_frames=n_frames, selection=selection)
        if not ok:
            continue
        extra.extend(unique)
        extra_ids.update((d.frame, candidate.track_id) for d in unique)
        covered.update(d.frame for d in unique)
        receipts.append(
            dict(
                policy="keep_unique_admissible_frames_v1",
                candidate_track_id=candidate.track_id,
                recovered_frames=[d.frame for d in unique],
            )
        )
    if not extra:
        return anchor
    detections = sorted([*anchor.detections, *extra], key=lambda d: d.frame)
    association_ids = {**anchor.association_ids, **extra_ids}
    return Track(
        track_id=anchor.track_id,
        detections=detections,
        mean=anchor.mean,
        covariance=anchor.covariance,
        embedding=anchor.embedding,
        hits=len(detections),
        last_frame=detections[-1].frame,
        start_frame=detections[0].frame,
        box=detections[-1].box,
        stitch_boundaries=receipts,
        association_ids=association_ids,
    )


def gaps(track: Track, *, first_frame: int, last_frame: int) -> list[tuple[int, int]]:
    """Frame ranges inside the clip where the selected track has no detection."""
    seen = sorted({d.frame for d in track.detections})
    spans: list[tuple[int, int]] = []
    cursor = first_frame
    for frame in seen:
        if frame > cursor:
            spans.append((cursor, frame - 1))
        cursor = frame + 1
    if cursor <= last_frame:
        spans.append((cursor, last_frame))
    return spans
