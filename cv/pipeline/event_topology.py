"""Label-free contact-topology branches for whole-point reconstruction."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations

import numpy as np

VALID_SIDES = {"near", "far"}
MIN_TOPOLOGY_SOLVE_FRACTION = 0.50
MAX_TOPOLOGY_MEDIAN_RMS_PX = 50.0
MAX_OMISSION_EVENT_PROBABILITY = 0.85
RETIME_SEARCH_FRAMES = 3.0
RETIME_MAX_INTERSECTION_PX = 8.0
MIN_CONTACT_GAP_SECONDS = 0.18
MIN_INSERTION_SPAN_SECONDS = 0.70
# Lobs, slices, and drop-shot exchanges can exceed the old 2.2 second cap. This is
# only a proposal bound; the 3D fit and final gate remain responsible for acceptance.
MAX_INSERTION_FLIGHT_SECONDS = 3.50
INSERTION_FIT_WINDOW_SECONDS = 0.36
INSERTION_FIT_GUARD_SECONDS = 0.08
INSERTION_MAX_INTERSECTION_PX = 10.0
INSERTION_MAX_PLAYER_DISTANCE_SCALE = 0.45
INSERTION_MAX_VELOCITY_COSINE = 0.65
INSERTION_MIN_OBSERVATIONS_PER_SIDE = 4
INSERTION_DEDUP_SECONDS = 0.20
INSERTION_ALTERNATE_MIN_PROBABILITY_RATIO = 0.70
INSERTION_ALTERNATE_MIN_COSINE_GAIN = 0.25
INSERTION_MAX_SEQUENCE_CONTACTS = 5
INSERTION_MAX_SEQUENCES_PER_GAP = 8
SEQUENCE_MAX_INTERSECTION_PX = 36.0
SEQUENCE_MAX_PLAYER_DISTANCE_SCALE = 0.60
SEQUENCE_MAX_VELOCITY_COSINE = 0.85
ARC_FIT_MIN_INLIERS = 4
ARC_FIT_MAX_OFFSET_FRAMES = 3.0


def _robust_arc_fit(times: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a short image-space arc while rejecting isolated tracker aliases."""
    degree = 2 if len(times) >= 7 else 1
    keep = np.ones(len(times), dtype=bool)
    coefficients = np.zeros((2, degree + 1), dtype=float)
    residual = np.full(len(times), float("inf"))
    for _ in range(5):
        if int(keep.sum()) < max(ARC_FIT_MIN_INLIERS, degree + 2):
            break
        coefficients = np.asarray(
            [np.polyfit(times[keep], points[keep, axis], degree) for axis in range(2)],
            float,
        )
        predicted = np.column_stack([np.polyval(coefficients[axis], times) for axis in range(2)])
        residual = np.linalg.norm(points - predicted, axis=1)
        center = float(np.median(residual[keep]))
        mad = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        threshold = max(4.0, center + 4.0 * max(mad, 0.5))
        updated = residual <= threshold
        if int(updated.sum()) < max(ARC_FIT_MIN_INLIERS, degree + 2):
            break
        if np.array_equal(updated, keep):
            keep = updated
            break
        keep = updated
    rms = (
        float(np.sqrt(np.mean(np.square(residual[keep]))))
        if int(keep.sum()) >= ARC_FIT_MIN_INLIERS
        else float("inf")
    )
    return coefficients, keep, rms


def _arc_point(coefficients: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(
        [np.polyval(coefficients[axis], offset) for axis in range(2)],
        float,
    )


def _arc_velocity(coefficients: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(
        [np.polyval(np.polyder(coefficients[axis]), offset) for axis in range(2)],
        float,
    )


def _normalize_phases(contacts: list[dict]) -> list[dict]:
    output = deepcopy(contacts)
    seen_spans = set()
    for contact in output:
        if contact.get("terminal"):
            continue
        span = contact.get("span")
        contact["phase"] = "serve" if span not in seen_spans else "rally"
        seen_spans.add(span)
    return output


def _retime_contacts(contacts: list[dict], ball: dict[int, np.ndarray]) -> list[dict]:
    output = _normalize_phases(contacts)
    for index in range(1, len(output) - 1):
        contact = output[index]
        if contact.get("terminal"):
            continue
        previous = output[index - 1]
        following = output[index + 1]
        if following.get("terminal"):
            continue
        if not (previous.get("span") == contact.get("span") == following.get("span")):
            continue
        if {
            previous.get("side"),
            contact.get("side"),
            following.get("side"),
        } - VALID_SIDES:
            continue
        if previous["side"] == contact["side"] or following["side"] == contact["side"]:
            continue
        center = float(contact["frame"])
        before = sorted(frame for frame in ball if center - 14 <= frame <= center - 2)
        after = sorted(frame for frame in ball if center + 2 <= frame <= center + 14)
        if len(before) < 6 or len(after) < 6:
            continue
        pre = [
            np.polyfit(
                np.asarray(before, float) - center, [ball[frame][axis] for frame in before], 2
            )
            for axis in range(2)
        ]
        post = [
            np.polyfit(np.asarray(after, float) - center, [ball[frame][axis] for frame in after], 2)
            for axis in range(2)
        ]
        offsets = np.arange(-RETIME_SEARCH_FRAMES, RETIME_SEARCH_FRAMES + 0.25, 0.5)
        distances = np.asarray(
            [
                np.linalg.norm(
                    [
                        np.polyval(pre[axis], offset) - np.polyval(post[axis], offset)
                        for axis in range(2)
                    ]
                )
                for offset in offsets
            ],
            float,
        )
        best = int(np.argmin(distances))
        if distances[best] > RETIME_MAX_INTERSECTION_PX:
            continue
        offset = float(offsets[best])
        if abs(offset) < 0.25:
            continue
        contact["frame_detector"] = center
        contact["frame"] = center + offset
        contact["retime_delta_frames"] = offset
        contact["retime_residual_px"] = float(distances[best])
        contact["retime_source"] = "bidirectional_track_intersection"
    return output


def _event_probability(contact: dict) -> float:
    value = contact.get("row", {}).get("probability")
    try:
        return float(np.clip(float(value), 1e-4, 1.0 - 1e-4))
    except (TypeError, ValueError):
        return 0.5


def _nearest_side_box(
    boxes: dict[int, list[dict]],
    frame: float,
    side: str,
    fps: float,
) -> tuple[int, dict] | None:
    window = max(1, int(math.ceil(0.12 * fps)))
    candidates = []
    for candidate_frame, rows in boxes.items():
        if abs(candidate_frame - frame) > window:
            continue
        for row in rows:
            if row.get("side") == side:
                candidates.append((abs(candidate_frame - frame), candidate_frame, row))
    if not candidates:
        return None
    _, candidate_frame, row = min(candidates, key=lambda item: item[:2])
    return candidate_frame, row


def _distance_to_player_box(point: np.ndarray, row: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = (2.0 * float(row[key]) for key in ("x0", "y0", "x1", "y1"))
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return math.hypot(dx, dy), max(math.hypot(x1 - x0, y1 - y0), 1.0)


def _vertical_position_in_player_box(point: np.ndarray, row: dict) -> float:
    y0, y1 = (2.0 * float(row[key]) for key in ("y0", "y1"))
    return float((point[1] - y0) / max(y1 - y0, 1.0))


def contact_candidate_evidence(
    frame: int,
    side: str,
    ball: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
    observed_frames: set[int],
    *,
    max_intersection_px: float = INSERTION_MAX_INTERSECTION_PX,
    max_player_distance_scale: float = INSERTION_MAX_PLAYER_DISTANCE_SCALE,
    max_velocity_cosine: float = INSERTION_MAX_VELOCITY_COSINE,
) -> dict:
    """Return the complete local evidence and fail-closed rejection reason."""
    fit_window = max(5, int(round(INSERTION_FIT_WINDOW_SECONDS * fps)))
    fit_guard = max(1, int(round(INSERTION_FIT_GUARD_SECONDS * fps)))
    before = sorted(
        candidate
        for candidate in observed_frames
        if frame - fit_window <= candidate <= frame - fit_guard and candidate in ball
    )
    after = sorted(
        candidate
        for candidate in observed_frames
        if frame + fit_guard <= candidate <= frame + fit_window and candidate in ball
    )
    evidence = {
        "seed_frame": int(frame),
        "side": side,
        "observations_before": len(before),
        "observations_after": len(after),
        "accepted": False,
        "rejection_reason": None,
    }
    if (
        len(before) < INSERTION_MIN_OBSERVATIONS_PER_SIDE
        or len(after) < INSERTION_MIN_OBSERVATIONS_PER_SIDE
    ):
        evidence["rejection_reason"] = "insufficient_observations"
        return evidence

    pre = np.asarray([ball[candidate] for candidate in before], float)
    post = np.asarray([ball[candidate] for candidate in after], float)
    pre_time = np.asarray(before, float) - frame
    post_time = np.asarray(after, float) - frame
    pre_fit, pre_keep, pre_rms = _robust_arc_fit(pre_time, pre)
    post_fit, post_keep, post_rms = _robust_arc_fit(post_time, post)
    evidence.update(
        {
            "inliers_before": int(pre_keep.sum()),
            "inliers_after": int(post_keep.sum()),
            "pre_arc_rms_px": pre_rms,
            "post_arc_rms_px": post_rms,
            "arc_fit_degree": int(pre_fit.shape[1] - 1),
            "observed_gap_frames": int(after[0] - before[-1] - 1),
        }
    )
    if (
        int(pre_keep.sum()) < ARC_FIT_MIN_INLIERS
        or int(post_keep.sum()) < ARC_FIT_MIN_INLIERS
        or not math.isfinite(pre_rms)
        or not math.isfinite(post_rms)
    ):
        evidence["rejection_reason"] = "insufficient_arc_inliers"
        return evidence

    offsets = np.arange(
        -ARC_FIT_MAX_OFFSET_FRAMES,
        ARC_FIT_MAX_OFFSET_FRAMES + 0.25,
        0.25,
    )
    distances = np.asarray(
        [
            np.linalg.norm(_arc_point(pre_fit, offset) - _arc_point(post_fit, offset))
            for offset in offsets
        ],
        float,
    )
    best = int(np.argmin(distances))
    offset = float(offsets[best])
    velocity_before = _arc_velocity(pre_fit, offset)
    velocity_after = _arc_velocity(post_fit, offset)
    speed_product = float(np.linalg.norm(velocity_before) * np.linalg.norm(velocity_after))
    evidence["speed_product"] = speed_product
    if speed_product < 1.0:
        evidence["rejection_reason"] = "insufficient_motion"
        return evidence
    velocity_cosine = float(np.dot(velocity_before, velocity_after) / speed_product)
    evidence["velocity_cosine"] = velocity_cosine
    if velocity_cosine > max_velocity_cosine:
        evidence["rejection_reason"] = "insufficient_velocity_change"
        return evidence

    pre_point = _arc_point(pre_fit, offset)
    post_point = _arc_point(post_fit, offset)
    intersection_residual = float(distances[best])
    evidence["intersection_residual_px"] = intersection_residual
    if intersection_residual > max_intersection_px:
        evidence["rejection_reason"] = "intersection_residual"
        return evidence
    point = 0.5 * (pre_point + post_point)
    box_match = _nearest_side_box(boxes, frame + offset, side, fps)
    if box_match is None:
        evidence["rejection_reason"] = "missing_player_box"
        return evidence
    box_frame, box = box_match
    player_distance, player_scale = _distance_to_player_box(point, box)
    player_distance_scale = player_distance / player_scale
    evidence.update(
        {
            "player_distance_px": player_distance,
            "player_distance_scale": player_distance_scale,
            "player_box_frame": int(box_frame),
            "player_vertical_position_scale": _vertical_position_in_player_box(point, box),
        }
    )
    if player_distance_scale > max_player_distance_scale:
        evidence["rejection_reason"] = "outside_player_reach"
        return evidence

    intersection_score = math.exp(-intersection_residual / 5.0)
    reach_score = math.exp(-player_distance_scale / 0.25)
    fit_score = math.exp(-(pre_rms + post_rms) / 12.0)
    kink_score = float(
        np.clip((max_velocity_cosine - velocity_cosine) / (max_velocity_cosine + 1.0), 0.0, 1.0)
    )
    confidence = float(
        np.clip(
            (intersection_score * reach_score * kink_score * fit_score) ** 0.25,
            0.05,
            0.995,
        )
    )
    evidence.update(
        {
            "frame": float(frame + offset),
            "probability": confidence,
            "accepted": True,
        }
    )
    return evidence


def _local_contact_candidate(
    frame: int,
    side: str,
    ball: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
    observed_frames: set[int],
    *,
    max_intersection_px: float = INSERTION_MAX_INTERSECTION_PX,
    max_player_distance_scale: float = INSERTION_MAX_PLAYER_DISTANCE_SCALE,
    max_velocity_cosine: float = INSERTION_MAX_VELOCITY_COSINE,
) -> dict | None:
    evidence = contact_candidate_evidence(
        frame,
        side,
        ball,
        boxes,
        fps,
        observed_frames,
        max_intersection_px=max_intersection_px,
        max_player_distance_scale=max_player_distance_scale,
        max_velocity_cosine=max_velocity_cosine,
    )
    if not evidence["accepted"]:
        return None
    return {
        key: value
        for key, value in evidence.items()
        if key not in {"accepted", "rejection_reason", "seed_frame", "speed_product"}
    }


def propose_missing_contact_candidates(
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
    *,
    observed_frames: set[int] | None = None,
) -> list[dict]:
    """Propose one opposite-side contact only inside a broken same-side span."""
    observed = set(ball) if observed_frames is None else observed_frames
    proposals = []
    for gap_index, (left, right) in enumerate(zip(contacts, contacts[1:])):
        if left.get("terminal") or right.get("terminal"):
            continue
        if left.get("span") != right.get("span"):
            continue
        if left.get("side") not in VALID_SIDES or left.get("side") != right.get("side"):
            continue
        gap = float(right["frame"] - left["frame"])
        if gap < MIN_INSERTION_SPAN_SECONDS * fps:
            continue
        opposite_side = "far" if left["side"] == "near" else "near"
        first_frame = int(
            math.ceil(
                max(
                    left["frame"] + MIN_CONTACT_GAP_SECONDS * fps,
                    right["frame"] - MAX_INSERTION_FLIGHT_SECONDS * fps,
                )
            )
        )
        last_frame = int(
            math.floor(
                min(
                    right["frame"] - MIN_CONTACT_GAP_SECONDS * fps,
                    left["frame"] + MAX_INSERTION_FLIGHT_SECONDS * fps,
                )
            )
        )
        local = []
        for frame in range(first_frame, last_frame + 1):
            candidate = _local_contact_candidate(
                frame,
                opposite_side,
                ball,
                boxes,
                fps,
                observed,
            )
            if candidate is not None:
                local.append(candidate)
        local.sort(
            key=lambda row: (
                -row["probability"],
                row["intersection_residual_px"],
                row["player_distance_scale"],
            )
        )
        deduplicated = []
        for candidate in local:
            if any(
                abs(candidate["frame"] - previous["frame"]) < INSERTION_DEDUP_SECONDS * fps
                for previous in deduplicated
            ):
                continue
            deduplicated.append(candidate)
            if len(deduplicated) == 2:
                break
        selected = deduplicated[:1]
        if len(deduplicated) > 1:
            primary, alternate = deduplicated
            stronger_reversal = (
                alternate["velocity_cosine"]
                <= primary["velocity_cosine"] - INSERTION_ALTERNATE_MIN_COSINE_GAIN
            )
            competitive_probability = (
                alternate["probability"]
                >= primary["probability"] * INSERTION_ALTERNATE_MIN_PROBABILITY_RATIO
            )
            if stronger_reversal and competitive_probability:
                selected.append(alternate)
        for rank, candidate in enumerate(selected):
            proposals.append({"gap_index": gap_index, "rank": rank, **candidate})
    return proposals


def _temporal_candidate_peaks(rows: list[dict], fps: float) -> list[dict]:
    if not rows:
        return []
    merge_gap = INSERTION_DEDUP_SECONDS * fps
    selected = []
    for row in sorted(
        rows,
        key=lambda item: (
            -item["probability"],
            item["intersection_residual_px"],
            item["player_distance_scale"],
        ),
    ):
        if any(abs(row["frame"] - previous["frame"]) < merge_gap for previous in selected):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: row["frame"])


def propose_missing_contact_sequences(
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
    *,
    observed_frames: set[int] | None = None,
    max_sequence_contacts: int = INSERTION_MAX_SEQUENCE_CONTACTS,
    max_sequences_per_gap: int = INSERTION_MAX_SEQUENCES_PER_GAP,
) -> list[dict]:
    """Propose parity-correct alternating contact sequences inside broken spans."""
    observed = set(ball) if observed_frames is None else observed_frames
    proposals = []
    min_gap = MIN_CONTACT_GAP_SECONDS * fps
    max_gap = MAX_INSERTION_FLIGHT_SECONDS * fps
    for gap_index, (left, right) in enumerate(zip(contacts, contacts[1:])):
        if left.get("terminal") or right.get("terminal"):
            continue
        if left.get("span") != right.get("span"):
            continue
        if left.get("side") not in VALID_SIDES or right.get("side") not in VALID_SIDES:
            continue
        gap = float(right["frame"] - left["frame"])
        if gap < MIN_INSERTION_SPAN_SECONDS * fps:
            continue
        first_frame = int(math.ceil(left["frame"] + min_gap))
        last_frame = int(math.floor(right["frame"] - min_gap))
        peaks_by_side = {}
        for side in sorted(VALID_SIDES):
            local = []
            for frame in range(first_frame, last_frame + 1):
                candidate = _local_contact_candidate(
                    frame,
                    side,
                    ball,
                    boxes,
                    fps,
                    observed,
                    max_intersection_px=SEQUENCE_MAX_INTERSECTION_PX,
                    max_player_distance_scale=SEQUENCE_MAX_PLAYER_DISTANCE_SCALE,
                    max_velocity_cosine=SEQUENCE_MAX_VELOCITY_COSINE,
                )
                if candidate is not None:
                    local.append(candidate)
            peaks_by_side[side] = _temporal_candidate_peaks(local, fps)

        all_peaks = sorted(
            [row for rows in peaks_by_side.values() for row in rows],
            key=lambda row: row["frame"],
        )
        opposite = "far" if left["side"] == "near" else "near"
        local_sequences = []
        maximum = min(max_sequence_contacts, len(all_peaks))
        same_side = left["side"] == right["side"]
        for length in range(1 if same_side else 2, maximum + 1, 2):
            for selected in combinations(all_peaks, length):
                expected = [opposite if index % 2 == 0 else left["side"] for index in range(length)]
                if [row["side"] for row in selected] != expected:
                    continue
                frames = [
                    float(left["frame"]),
                    *(float(row["frame"]) for row in selected),
                    float(right["frame"]),
                ]
                intervals = np.diff(frames)
                if np.any(intervals < min_gap) or np.any(intervals > max_gap):
                    continue
                probability = float(
                    math.exp(
                        sum(math.log(max(float(row["probability"]), 1e-4)) for row in selected)
                        / len(selected)
                    )
                )
                local_sequences.append(
                    {
                        "gap_index": gap_index,
                        "contacts": [dict(row) for row in selected],
                        "probability": probability,
                        "max_flight_seconds": float(max(intervals) / fps),
                        "contact_count": length,
                    }
                )
        local_sequences.sort(
            key=lambda row: (
                -row["probability"],
                row["contact_count"],
                row["max_flight_seconds"],
            )
        )
        for rank, row in enumerate(local_sequences[:max_sequences_per_gap]):
            proposals.append({"rank": rank, **row})
    return proposals


def propose_global_contact_paths(
    ball: dict[int, np.ndarray],
    boxes: dict[int, list[dict]],
    fps: float,
    span_ranges: list[list[float]],
    *,
    observed_frames: set[int] | None = None,
    event_hypotheses: list[dict] | None = None,
    max_paths_per_span: int = 4,
) -> list[dict]:
    """Recover alternating contact chains without requiring existing S5 anchors."""
    observed = set(ball) if observed_frames is None else observed_frames
    paths = []
    min_gap = MIN_CONTACT_GAP_SECONDS * fps
    max_gap = MAX_INSERTION_FLIGHT_SECONDS * fps
    for span_index, (raw_start, raw_end) in enumerate(span_ranges):
        start = int(math.ceil(raw_start))
        end = int(math.floor(raw_end))
        scan_step = max(1, int(round(fps / 25.0)))
        candidates = []
        span_hypotheses = [
            row for row in (event_hypotheses or []) if start <= float(row["frame"]) <= end
        ]

        def nearest_probability(frame: float, event_type: str) -> float:
            nearby = [
                float(row.get("probability", 0.0))
                for row in span_hypotheses
                if row.get("event_type") == event_type
                and abs(float(row["frame"]) - frame) <= 0.14 * fps
            ]
            return max(nearby, default=0.0)

        for side in sorted(VALID_SIDES):
            local = []
            for frame in range(start, end + 1, scan_step):
                candidate = _local_contact_candidate(
                    frame,
                    side,
                    ball,
                    boxes,
                    fps,
                    observed,
                    max_intersection_px=18.0,
                    max_player_distance_scale=SEQUENCE_MAX_PLAYER_DISTANCE_SCALE,
                    max_velocity_cosine=SEQUENCE_MAX_VELOCITY_COSINE,
                )
                if candidate is not None:
                    local.append(candidate)
            local = _temporal_candidate_peaks(local, fps)
            for candidate in local:
                contact_probability = nearest_probability(candidate["frame"], "contact")
                bounce_probability = nearest_probability(candidate["frame"], "bounce")
                candidate["contact_hypothesis_probability"] = contact_probability
                candidate["bounce_hypothesis_probability"] = bounce_probability
                if contact_probability > 0.0:
                    candidate["probability"] = 1.0 - (
                        (1.0 - float(candidate["probability"])) * (1.0 - 0.5 * contact_probability)
                    )
                if (
                    float(candidate["player_vertical_position_scale"]) >= 0.65
                    and bounce_probability >= 0.25
                    and bounce_probability > 1.5 * max(contact_probability, 0.05)
                ):
                    candidate["probability"] *= 0.20
                    candidate["bounce_competition_penalty"] = True
            for hypothesis in span_hypotheses:
                if hypothesis.get("event_type") != "contact":
                    continue
                frame = float(hypothesis["frame"])
                search_radius = max(3, int(math.ceil(0.24 * fps)))
                nearby = []
                for candidate_frame in ball:
                    if abs(candidate_frame - frame) > search_radius:
                        continue
                    box_match = _nearest_side_box(boxes, candidate_frame, side, fps)
                    if box_match is None:
                        continue
                    box_frame, box = box_match
                    distance, scale = _distance_to_player_box(ball[candidate_frame], box)
                    distance_scale = distance / scale
                    temporal_cost = abs(candidate_frame - frame) / max(search_radius, 1)
                    nearby.append(
                        (
                            distance_scale + 0.08 * temporal_cost,
                            distance_scale,
                            abs(candidate_frame - frame),
                            candidate_frame,
                            box_frame,
                            box,
                            distance,
                        )
                    )
                if not nearby:
                    continue
                _, distance_scale, _, nearest_frame, box_frame, box, distance = min(nearby)
                if distance_scale > SEQUENCE_MAX_PLAYER_DISTANCE_SCALE:
                    continue
                probability = float(hypothesis.get("probability", 0.05))
                vertical_position_scale = _vertical_position_in_player_box(ball[nearest_frame], box)
                bounce_probability = nearest_probability(nearest_frame, "bounce")
                bounce_competition_penalty = (
                    vertical_position_scale >= 0.65
                    and bounce_probability >= 0.25
                    and bounce_probability > 1.5 * max(probability, 0.05)
                )
                if bounce_competition_penalty:
                    probability *= 0.20
                local.append(
                    {
                        "frame": float(nearest_frame),
                        "side": side,
                        "probability": probability,
                        "intersection_residual_px": 18.0,
                        "player_distance_px": distance,
                        "player_distance_scale": distance_scale,
                        "player_vertical_position_scale": vertical_position_scale,
                        "player_box_frame": int(box_frame),
                        "hypothesis_frame": frame,
                        "hypothesis_snap_delta_frames": float(nearest_frame - frame),
                        "hypothesis_origin": hypothesis.get("origin"),
                        "evidence_source": "leaky_s5_hypothesis_temporal_reach_snap",
                        "bounce_competition_penalty": bounce_competition_penalty,
                    }
                )
            local = [row for row in local if not row.get("bounce_competition_penalty")]
            candidates.extend(_temporal_candidate_peaks(local, fps))
        candidates.sort(key=lambda row: (row["frame"], row["side"]))
        if len(candidates) < 2:
            continue

        scores = []
        previous: list[int | None] = []
        lengths = []
        for index, candidate in enumerate(candidates):
            probability = float(np.clip(candidate["probability"], 1e-4, 1.0 - 1e-4))
            node_score = math.log(probability / (1.0 - probability)) + 1.0
            best_score = node_score
            best_previous = None
            best_length = 1
            for prior_index, prior in enumerate(candidates[:index]):
                gap = float(candidate["frame"] - prior["frame"])
                if prior["side"] == candidate["side"] or not min_gap <= gap <= max_gap:
                    continue
                candidate_score = scores[prior_index] + node_score + 0.35
                candidate_length = lengths[prior_index] + 1
                if (candidate_score, candidate_length) > (best_score, best_length):
                    best_score = candidate_score
                    best_previous = prior_index
                    best_length = candidate_length
            scores.append(best_score)
            previous.append(best_previous)
            lengths.append(best_length)

        endpoints = sorted(
            range(len(candidates)),
            key=lambda index: (lengths[index], scores[index]),
            reverse=True,
        )
        index_paths = []
        for endpoint in endpoints:
            if lengths[endpoint] < 2:
                continue
            indices = []
            cursor: int | None = endpoint
            while cursor is not None:
                indices.append(cursor)
                cursor = previous[cursor]
            indices.reverse()
            index_paths.append(indices)
            for position, selected_index in enumerate(indices):
                selected_candidate = candidates[selected_index]
                for alternate_index, alternate in enumerate(candidates):
                    if (
                        alternate_index in indices
                        or alternate["side"] != selected_candidate["side"]
                    ):
                        continue
                    if (
                        abs(float(alternate["frame"]) - float(selected_candidate["frame"]))
                        > 0.35 * fps
                    ):
                        continue
                    prior_frame = (
                        float(candidates[indices[position - 1]]["frame"]) if position > 0 else None
                    )
                    next_frame = (
                        float(candidates[indices[position + 1]]["frame"])
                        if position + 1 < len(indices)
                        else None
                    )
                    if (
                        prior_frame is not None
                        and not min_gap <= float(alternate["frame"]) - prior_frame <= max_gap
                    ):
                        continue
                    if (
                        next_frame is not None
                        and not min_gap <= next_frame - float(alternate["frame"]) <= max_gap
                    ):
                        continue
                    variant = list(indices)
                    variant[position] = alternate_index
                    index_paths.append(variant)

        ranked_paths = []
        signatures = set()
        for indices in index_paths:
            selected = [dict(candidates[index]) for index in indices]
            signature = tuple((round(row["frame"], 2), row["side"]) for row in selected)
            if signature in signatures:
                continue
            signatures.add(signature)
            path_score = sum(
                math.log(
                    float(np.clip(row["probability"], 1e-4, 1.0 - 1e-4))
                    / (1.0 - float(np.clip(row["probability"], 1e-4, 1.0 - 1e-4)))
                )
                + 1.0
                for row in selected
            ) + 0.35 * (len(selected) - 1)
            ranked_paths.append(
                {
                    "span": span_index,
                    "contacts": selected,
                    "score": float(path_score),
                    "minimum_probability": min(row["probability"] for row in selected),
                }
            )
        ranked_paths.sort(
            key=lambda row: (len(row["contacts"]), row["score"]),
            reverse=True,
        )
        paths.extend(ranked_paths[:max_paths_per_span])
    return paths


def _omission_indices(contacts: list[dict], fps: float) -> list[int]:
    indices = set()
    for left_index, (left, right) in enumerate(zip(contacts, contacts[1:])):
        if left.get("terminal") or right.get("terminal"):
            continue
        if left.get("span") != right.get("span"):
            continue
        same_side = left.get("side") in VALID_SIDES and left.get("side") == right.get("side")
        too_close = float(right["frame"] - left["frame"]) < MIN_CONTACT_GAP_SECONDS * fps
        if not (same_side or too_close):
            continue
        indices.add(left_index)
        indices.add(left_index + 1)
    return sorted(indices)


def build_contact_topology_branches(
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    fps: float,
    *,
    boxes: dict[int, list[dict]] | None = None,
    observed_frames: set[int] | None = None,
    max_branches: int = 8,
    include_sequences: bool = False,
    span_ranges: list[list[float]] | None = None,
    event_hypotheses: list[dict] | None = None,
) -> list[dict]:
    """Build baseline, retimed, insertion, and omission interpretations."""
    baseline = _normalize_phases(contacts)
    candidates = [
        {
            "branch_id": "contacts_baseline",
            "contacts": baseline,
            "omitted": [],
            "inserted": [],
            "retimed": [],
            "prior_cost": 0.0,
        }
    ]
    retimed = _retime_contacts(baseline, ball)
    retimed_rows = [
        {
            "index": index,
            "seed_id": contact.get("row", {}).get("seed_id"),
            "from_frame": contact.get("frame_detector"),
            "to_frame": contact["frame"],
            "residual_px": contact.get("retime_residual_px"),
        }
        for index, contact in enumerate(retimed)
        if "retime_delta_frames" in contact
    ]
    if retimed_rows:
        candidates.append(
            {
                "branch_id": "contacts_retimed",
                "contacts": retimed,
                "omitted": [],
                "inserted": [],
                "retimed": retimed_rows,
                "prior_cost": 0.25
                * sum(abs(row["to_frame"] - row["from_frame"]) for row in retimed_rows),
            }
        )

    insertion_candidates = propose_missing_contact_candidates(
        baseline,
        ball,
        boxes or {},
        fps,
        observed_frames=observed_frames,
    )
    for proposal in insertion_candidates:
        insertion_index = proposal["gap_index"] + 1
        inserted_contact = {
            "frame": proposal["frame"],
            "side": proposal["side"],
            "span": baseline[proposal["gap_index"]]["span"],
            "phase": "rally",
            "source": "automatic_event_topology_insertion",
            "side_evidence": {
                "side": proposal["side"],
                "confidence": proposal["probability"],
                "reason": "track_kink_player_reach",
                "ball_frame": int(round(proposal["frame"])),
                "evidence_frames": {proposal["side"]: proposal["player_box_frame"]},
            },
            "row": {
                "probability": proposal["probability"],
                "origin": "event_topology_track_kink_player_reach",
            },
        }
        expanded = _normalize_phases(
            baseline[:insertion_index] + [inserted_contact] + baseline[insertion_index:]
        )
        insertion = {
            key: value for key, value in proposal.items() if key not in {"gap_index", "rank"}
        }
        insertion["index"] = insertion_index
        insertion["gap_index"] = proposal["gap_index"]
        insertion["rank"] = proposal["rank"]
        prior_cost = 2.0 - math.log(max(proposal["probability"], 1e-4))
        branch_id = f"insert_{proposal['gap_index']}_f{proposal['frame']:.2f}"
        candidates.append(
            {
                "branch_id": branch_id,
                "contacts": expanded,
                "omitted": [],
                "inserted": [insertion],
                "retimed": [],
                "prior_cost": prior_cost,
            }
        )

    if include_sequences:
        for sequence in propose_missing_contact_sequences(
            baseline,
            ball,
            boxes or {},
            fps,
            observed_frames=observed_frames,
        ):
            insertion_index = sequence["gap_index"] + 1
            span = baseline[sequence["gap_index"]]["span"]
            inserted_contacts = []
            insertion_rows = []
            for offset, proposal in enumerate(sequence["contacts"]):
                inserted_contacts.append(
                    {
                        "frame": proposal["frame"],
                        "side": proposal["side"],
                        "span": span,
                        "phase": "rally",
                        "source": "automatic_event_topology_sequence_insertion",
                        "side_evidence": {
                            "side": proposal["side"],
                            "confidence": proposal["probability"],
                            "reason": "track_kink_player_reach_sequence",
                            "ball_frame": int(round(proposal["frame"])),
                            "evidence_frames": {proposal["side"]: proposal["player_box_frame"]},
                        },
                        "row": {
                            "probability": proposal["probability"],
                            "origin": "event_topology_track_kink_player_reach_sequence",
                        },
                    }
                )
                insertion_rows.append(
                    {
                        **proposal,
                        "index": insertion_index + offset,
                        "gap_index": sequence["gap_index"],
                        "rank": sequence["rank"],
                    }
                )
            expanded = _normalize_phases(
                baseline[:insertion_index] + inserted_contacts + baseline[insertion_index:]
            )
            prior_cost = sum(
                2.0 - math.log(max(float(row["probability"]), 1e-4)) for row in sequence["contacts"]
            ) + 0.5 * (len(sequence["contacts"]) - 1)
            frames = "_".join(f"{row['frame']:.2f}" for row in sequence["contacts"])
            candidates.append(
                {
                    "branch_id": f"insert_sequence_{sequence['gap_index']}_f{frames}",
                    "contacts": expanded,
                    "omitted": [],
                    "inserted": insertion_rows,
                    "retimed": [],
                    "prior_cost": prior_cost,
                }
            )

        for path_rank, path in enumerate(
            propose_global_contact_paths(
                ball,
                boxes or {},
                fps,
                span_ranges or [],
                observed_frames=observed_frames,
                event_hypotheses=event_hypotheses,
            )
        ):
            recovered = []
            insertion_rows = []
            for proposal in path["contacts"]:
                recovered.append(
                    {
                        "frame": proposal["frame"],
                        "side": proposal["side"],
                        "span": path["span"],
                        "phase": "rally",
                        "source": "automatic_arc_path_recovery",
                        "side_evidence": {
                            "side": proposal["side"],
                            "confidence": proposal["probability"],
                            "reason": "robust_two_arc_player_reach_path",
                            "ball_frame": int(round(proposal["frame"])),
                            "evidence_frames": {proposal["side"]: proposal["player_box_frame"]},
                        },
                        "row": {
                            "probability": proposal["probability"],
                            "origin": "event_topology_global_two_arc_path",
                        },
                    }
                )
                insertion_rows.append(dict(proposal))
            terminal = [
                contact
                for contact in baseline
                if contact.get("terminal") and contact.get("span") == path["span"]
            ]
            expanded = _normalize_phases(
                sorted([*recovered, *terminal], key=lambda row: float(row["frame"]))
            )
            if len(expanded) < 2:
                continue
            candidates.append(
                {
                    "branch_id": f"arc_path_{path['span']}_{path_rank}",
                    "contacts": expanded,
                    "omitted": [
                        {
                            "frame": float(contact["frame"]),
                            "side": contact.get("side"),
                            "seed_id": contact.get("row", {}).get("seed_id"),
                            "probability": _event_probability(contact),
                        }
                        for contact in baseline
                        if not contact.get("terminal") and contact.get("span") == path["span"]
                    ],
                    "inserted": insertion_rows,
                    "retimed": [],
                    "prior_cost": 0.75 * len(insertion_rows),
                    "arc_path_score": path["score"],
                }
            )
            hypothesis_timed = []
            for proposal in path["contacts"]:
                hypothesis_frame = proposal.get("hypothesis_frame")
                if (
                    hypothesis_frame is None
                    or float(proposal.get("probability", 0.0)) < 0.75
                    or abs(float(hypothesis_frame) - float(proposal["frame"])) > 2.0
                ):
                    hypothesis_timed = []
                    break
                hypothesis_timed.append(
                    {
                        **proposal,
                        "frame": float(hypothesis_frame),
                        "path_frame": float(proposal["frame"]),
                        "timing_interpretation": "s5_hypothesis_frame",
                    }
                )
            if hypothesis_timed and any(
                float(row["frame"]) != float(row["path_frame"]) for row in hypothesis_timed
            ):
                timed_contacts = []
                for proposal in hypothesis_timed:
                    timed_contacts.append(
                        {
                            "frame": proposal["frame"],
                            "side": proposal["side"],
                            "span": path["span"],
                            "phase": "rally",
                            "source": "automatic_arc_path_hypothesis_timing",
                            "side_evidence": {
                                "side": proposal["side"],
                                "confidence": proposal["probability"],
                                "reason": "leaky_s5_hypothesis_timing_branch",
                                "ball_frame": int(round(proposal["frame"])),
                                "evidence_frames": {proposal["side"]: proposal["player_box_frame"]},
                            },
                            "row": {
                                "probability": proposal["probability"],
                                "origin": "event_topology_global_two_arc_path_hypothesis_timing",
                            },
                        }
                    )
                timed_expanded = _normalize_phases(
                    sorted([*timed_contacts, *terminal], key=lambda row: float(row["frame"]))
                )
                if len(timed_expanded) >= 2:
                    candidates.append(
                        {
                            "branch_id": f"arc_path_{path['span']}_{path_rank}_hypothesis_timing",
                            "contacts": timed_expanded,
                            "omitted": [
                                {
                                    "frame": float(contact["frame"]),
                                    "side": contact.get("side"),
                                    "seed_id": contact.get("row", {}).get("seed_id"),
                                    "probability": _event_probability(contact),
                                }
                                for contact in baseline
                                if not contact.get("terminal")
                                and contact.get("span") == path["span"]
                            ],
                            "inserted": hypothesis_timed,
                            "retimed": [],
                            "prior_cost": 0.75 * len(hypothesis_timed) + 0.10,
                            "arc_path_score": path["score"],
                        }
                    )

    for index in _omission_indices(baseline, fps):
        omitted_contact = baseline[index]
        reduced = _normalize_phases(baseline[:index] + baseline[index + 1 :])
        omission_cost = max(
            0.0,
            math.log(
                _event_probability(omitted_contact) / (1.0 - _event_probability(omitted_contact))
            ),
        )
        omission = {
            "index": index,
            "frame": float(omitted_contact["frame"]),
            "side": omitted_contact.get("side"),
            "seed_id": omitted_contact.get("row", {}).get("seed_id"),
            "probability": _event_probability(omitted_contact),
        }
        branch_id = f"omit_{index}_f{float(omitted_contact['frame']):.2f}"
        candidates.append(
            {
                "branch_id": branch_id,
                "contacts": reduced,
                "omitted": [omission],
                "inserted": [],
                "retimed": [],
                "prior_cost": omission_cost,
            }
        )
        reduced_retimed = _retime_contacts(reduced, ball)
        reduced_retimed_rows = [
            {
                "index": retimed_index,
                "seed_id": contact.get("row", {}).get("seed_id"),
                "from_frame": contact.get("frame_detector"),
                "to_frame": contact["frame"],
                "residual_px": contact.get("retime_residual_px"),
            }
            for retimed_index, contact in enumerate(reduced_retimed)
            if "retime_delta_frames" in contact
        ]
        if reduced_retimed_rows:
            candidates.append(
                {
                    "branch_id": f"{branch_id}_retimed",
                    "contacts": reduced_retimed,
                    "omitted": [omission],
                    "inserted": [],
                    "retimed": reduced_retimed_rows,
                    "prior_cost": omission_cost
                    + 0.25
                    * sum(abs(row["to_frame"] - row["from_frame"]) for row in reduced_retimed_rows),
                }
            )

    unique = {}
    for branch in candidates:
        signature = tuple(
            (
                contact.get("row", {}).get("seed_id"),
                round(float(contact["frame"]), 3),
                bool(contact.get("terminal")),
            )
            for contact in branch["contacts"]
        )
        unique.setdefault(signature, branch)
    branches = list(unique.values())
    if include_sequences:
        branches.sort(
            key=lambda branch: (
                branch["branch_id"] != "contacts_baseline",
                float(branch["prior_cost"]),
                branch["branch_id"],
            )
        )
    return branches[:max_branches]


def score_contact_topology_branch(
    branch: dict,
    *,
    attempted: int,
    solved: int,
    weighted_rms_px: list[float],
    endpoint_errors_px: list[float],
) -> dict:
    contacts = branch["contacts"]
    same_side = sum(
        left.get("side") in VALID_SIDES
        and left.get("side") == right.get("side")
        and not right.get("terminal")
        for left, right in zip(contacts, contacts[1:])
    )
    unsolved_fraction = 1.0 if attempted == 0 else 1.0 - solved / attempted
    median_rms = float(np.median(weighted_rms_px)) if weighted_rms_px else 100.0
    worst_rms = max(weighted_rms_px, default=200.0)
    endpoint_median = float(np.median(endpoint_errors_px)) if endpoint_errors_px else 100.0
    components = {
        "unsolved": 45.0 * unsolved_fraction,
        "same_side": 12.0 * same_side,
        "median_reprojection": min(median_rms, 60.0),
        "worst_reprojection": 0.15 * min(worst_rms, 200.0),
        "endpoint_reprojection": 0.75 * min(endpoint_median, 80.0),
        "prior": float(branch["prior_cost"]),
    }
    return {
        "total": float(sum(components.values())),
        "components": components,
        "attempted": attempted,
        "solved": solved,
        "same_side_pairs": same_side,
        "median_weighted_rms_px": median_rms,
        "worst_weighted_rms_px": worst_rms,
        "median_endpoint_error_px": endpoint_median,
    }


def select_contact_topology_branch(
    evaluated: list[dict],
    *,
    minimum_margin: float = 8.0,
) -> tuple[dict, bool, float, str]:
    ordered = sorted(evaluated, key=lambda row: row["score"]["total"])
    baseline = next(row for row in evaluated if row["branch_id"] == "contacts_baseline")
    best = ordered[0]
    margin = float(baseline["score"]["total"] - best["score"]["total"])
    if best["branch_id"] == "contacts_baseline":
        return baseline, False, margin, "baseline_best"
    if best["score"]["solved"] < baseline["score"]["solved"]:
        return baseline, False, margin, "alternative_loses_solved_flights"
    if margin < minimum_margin:
        return baseline, False, margin, "insufficient_margin"
    rejection_reason = contact_topology_branch_rejection_reason(best)
    if (
        rejection_reason == "insufficient_global_reprojection"
        and int(baseline["score"]["attempted"]) == 0
        and int(best["score"]["attempted"]) > 0
        and int(best["score"]["solved"]) == int(best["score"]["attempted"])
    ):
        return best, False, margin, "reprojection_recovery_candidate"
    if rejection_reason is not None:
        return baseline, False, margin, rejection_reason
    return best, True, margin, "decisive_improvement"


def contact_topology_branch_rejection_reason(branch: dict) -> str | None:
    """Fail closed when a relative branch win lacks usable absolute evidence."""
    score = branch["score"]
    attempted = int(score["attempted"])
    solve_fraction = float(score["solved"] / attempted) if attempted else 0.0
    if solve_fraction < MIN_TOPOLOGY_SOLVE_FRACTION:
        return "insufficient_global_solve_coverage"
    if float(score["median_weighted_rms_px"]) > MAX_TOPOLOGY_MEDIAN_RMS_PX:
        return "insufficient_global_reprojection"
    if any(
        float(row.get("probability", 1.0)) > MAX_OMISSION_EVENT_PROBABILITY
        for row in branch.get("omitted", [])
    ):
        return "high_confidence_omission_unsupported"
    return None


def shortlist_contact_topology_branches(
    evaluated: list[dict],
    *,
    width: int = 2,
) -> list[dict]:
    """Keep baseline plus coverage-safe alternatives for expensive scoring.

    Initial reprojection comes from a deliberately cheap optimizer pass. A branch that
    increases solved-flight coverage may enter full refinement despite poor preliminary
    reprojection; the unchanged absolute gate still rejects it if refinement cannot fix it.
    Structural or solve-coverage failures remain ineligible.
    """
    if width < 1:
        raise ValueError("topology shortlist width must be at least 1")
    baseline = next(row for row in evaluated if row["branch_id"] == "contacts_baseline")
    alternatives = sorted(
        (
            row
            for row in evaluated
            if row["branch_id"] != "contacts_baseline"
            and row["score"]["solved"] >= baseline["score"]["solved"]
            and (
                contact_topology_branch_rejection_reason(row) is None
                or (
                    contact_topology_branch_rejection_reason(row)
                    == "insufficient_global_reprojection"
                    and row["score"]["solved"] > baseline["score"]["solved"]
                )
            )
        ),
        key=lambda row: row["score"]["total"],
    )
    return [baseline, *alternatives[: max(0, width - 1)]]
