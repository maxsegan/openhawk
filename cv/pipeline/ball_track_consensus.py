"""Offline local-consistency decoder for one or more ball-detector candidate streams."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cv.pipeline import resolution as res
from cv.pipeline.frame_identity import frame_number_from_name
from cv.pipeline.track_artifact import write_track_artifact


# Every radius in DecoderConfig is calibrated in the space the shipped candidate artifacts
# declare (RESOLUTION_CONTRACT.md items 4 and 7: candidates, crop geometry and thresholds
# migrate as one bundle).  Reading another space here would leave the radii behind.
CONSENSUS_PIXEL_SPACE = res.LEGACY_TRACKING_SIZE


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    score: float
    rank: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class DecoderConfig:
    beam_width: int = 160
    agreement_radius: float = 12.0
    dedup_radius: float = 6.0
    rank_cost: float = 0.24
    score_cost: float = 0.55
    agreement_bonus: float = 1.2
    speed_free: float = 55.0
    speed_scale: float = 24.0
    hard_speed_limit: float = 150.0
    prediction_radius: float = 220.0
    acceleration_scale: float = 18.0
    kink_cost: float = 1.25
    stationary_speed: float = 1.0
    stationary_cost: float = 0.8
    missing_cost: float = 0.65
    restart_cost: float = 1.4
    reacquisition_reset_gap: int = 12
    reacquisition_min_sources: int = 2
    maximum_gap: int | None = None
    interpolation_gap: int = 12
    interpolation_continuity_radius: float = 45.0
    hotspot_bin: float = 16.0
    hotspot_stationary_bin: float = 8.0
    hotspot_min_frames: int = 12
    hotspot_fraction: float = 0.03
    hotspot_consecutive_frames: int = 6
    hotspot_motion_reach: int = 24
    hotspot_motion_step: float = 35.0
    hotspot_escape_distance: float = 20.0


def config_for_consumer(
    fps: float,
    consumer: str = "integrity",
    interpolation_gap: int | None = None,
) -> DecoderConfig:
    """Return an elapsed-time-scaled association operating point."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if consumer not in {"integrity", "availability"}:
        raise ValueError(f"unknown consumer: {consumer}")

    def scaled(reference_frames: int, minimum: int = 1) -> int:
        return max(minimum, round(reference_frames * fps / 50.0))

    shared = {
        "interpolation_gap": (
            scaled(DecoderConfig().interpolation_gap)
            if interpolation_gap is None
            else interpolation_gap
        ),
        "reacquisition_reset_gap": scaled(
            DecoderConfig().reacquisition_reset_gap,
            4,
        ),
        "hotspot_min_frames": scaled(DecoderConfig().hotspot_min_frames, 4),
        "hotspot_consecutive_frames": scaled(
            DecoderConfig().hotspot_consecutive_frames,
            3,
        ),
        "hotspot_motion_reach": scaled(
            DecoderConfig().hotspot_motion_reach,
            8,
        ),
        "hotspot_motion_step": (DecoderConfig().hotspot_motion_step * 50.0 / fps),
    }
    if consumer == "availability":
        return DecoderConfig(
            missing_cost=0.9,
            restart_cost=0.6,
            **shared,
        )
    return DecoderConfig(**shared)


@dataclass
class _Path:
    cost: float
    parent: _Path | None
    choice: Candidate | None
    previous_frame: int | None
    previous: Candidate | None
    last_frame: int | None
    last: Candidate | None
    gap: int
    steps: int


def _distance(left: Candidate, right: Candidate) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def merge_candidates(
    rows: list[tuple[str, Candidate]],
    config: DecoderConfig,
) -> list[Candidate]:
    merged: list[tuple[Candidate, set[str]]] = []
    for source, candidate in sorted(rows, key=lambda item: (item[1].rank, -item[1].score)):
        nearest = next(
            (
                index
                for index, (prior, _) in enumerate(merged)
                if _distance(prior, candidate) <= config.dedup_radius
            ),
            None,
        )
        if nearest is None:
            merged.append((candidate, {source}))
            continue
        prior, sources = merged[nearest]
        sources.add(source)
        if candidate.score > prior.score:
            merged[nearest] = (candidate, sources)
    return [
        Candidate(
            candidate.x,
            candidate.y,
            candidate.score,
            candidate.rank,
            tuple(sorted(sources)),
        )
        for candidate, sources in merged
    ]


def suppress_static_hotspots(
    frames: dict[int, list[Candidate]],
    config: DecoderConfig,
) -> tuple[dict[int, list[Candidate]], set[tuple[str, int, int]]]:
    """Remove detector-specific fixed locations that recur implausibly often."""
    counts: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    stationary_counts: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for frame, candidates in frames.items():
        for candidate in candidates:
            x_bin = round(candidate.x / config.hotspot_bin)
            y_bin = round(candidate.y / config.hotspot_bin)
            stationary_x = round(candidate.x / config.hotspot_stationary_bin)
            stationary_y = round(candidate.y / config.hotspot_stationary_bin)
            for source in candidate.sources:
                counts[source, x_bin, y_bin].add(frame)
                stationary_counts[source, stationary_x, stationary_y].add(frame)
    threshold = max(
        config.hotspot_min_frames,
        math.ceil(len(frames) * config.hotspot_fraction),
    )
    hotspots = {key for key, observed_frames in counts.items() if len(observed_frames) >= threshold}
    stationary_hotspots = set()
    for key, observed_frames in stationary_counts.items():
        ordered = sorted(observed_frames)
        longest_run = current_run = 1
        for left, right in zip(ordered, ordered[1:]):
            current_run = current_run + 1 if right == left + 1 else 1
            longest_run = max(longest_run, current_run)
        if longest_run >= config.hotspot_consecutive_frames:
            stationary_hotspots.add(key)
    filtered = {}

    def has_motion_support(frame: int, candidate: Candidate) -> bool:
        if len(candidate.sources) < 2:
            return False
        for direction in (-1, 1):
            frontier = [candidate]
            for step in range(1, config.hotspot_motion_reach + 1):
                neighbors = [
                    neighbor
                    for neighbor in frames.get(frame + direction * step, [])
                    if len(neighbor.sources) >= 2
                    and any(
                        _distance(prior, neighbor) <= config.hotspot_motion_step
                        for prior in frontier
                    )
                ]
                if not neighbors:
                    break
                if any(
                    _distance(candidate, neighbor) >= config.hotspot_escape_distance
                    for neighbor in neighbors
                ):
                    return True
                frontier = neighbors
        return False

    for frame, candidates in frames.items():
        kept = []
        for candidate in candidates:
            x_bin = round(candidate.x / config.hotspot_bin)
            y_bin = round(candidate.y / config.hotspot_bin)
            stationary_x = round(candidate.x / config.hotspot_stationary_bin)
            stationary_y = round(candidate.y / config.hotspot_stationary_bin)
            is_hotspot = any(
                (source, x_bin, y_bin) in hotspots
                or (source, stationary_x, stationary_y) in stationary_hotspots
                for source in candidate.sources
            )
            if is_hotspot and not has_motion_support(frame, candidate):
                continue
            kept.append(candidate)
        filtered[frame] = kept
    return filtered, hotspots | stationary_hotspots


def _velocity(
    earlier_frame: int,
    earlier: Candidate,
    later_frame: int,
    later: Candidate,
    fps: float,
) -> tuple[float, float]:
    frame_gap = later_frame - earlier_frame
    scale = fps / (50.0 * frame_gap)
    return ((later.x - earlier.x) * scale, (later.y - earlier.y) * scale)


def _transition_cost(
    path: _Path,
    frame: int,
    candidate: Candidate,
    fps: float,
    config: DecoderConfig,
) -> float:
    unary = config.rank_cost * candidate.rank
    unary += config.score_cost * (1.0 - min(max(candidate.score, 0.0), 1.0))
    unary -= config.agreement_bonus * max(0, len(candidate.sources) - 1)
    if path.last is None or path.last_frame is None:
        return unary + (config.restart_cost if path.steps else 0.0)
    if path.gap:
        if (
            path.gap >= config.reacquisition_reset_gap
            and len(candidate.sources) >= config.reacquisition_min_sources
        ):
            return unary + config.restart_cost
        unary += config.restart_cost

    velocity = _velocity(path.last_frame, path.last, frame, candidate, fps)
    speed = math.hypot(*velocity)
    if speed > config.hard_speed_limit:
        return math.inf
    speed_cost = max(0.0, speed - config.speed_free) / config.speed_scale
    if path.previous is not None and speed < config.stationary_speed:
        speed_cost += config.stationary_cost
    if path.previous is None or path.previous_frame is None:
        return unary + speed_cost

    prior_velocity = _velocity(
        path.previous_frame,
        path.previous,
        path.last_frame,
        path.last,
        fps,
    )
    acceleration = math.hypot(
        velocity[0] - prior_velocity[0],
        velocity[1] - prior_velocity[1],
    )
    predicted_x = path.last.x + prior_velocity[0] * 50.0 / fps * (frame - path.last_frame)
    predicted_y = path.last.y + prior_velocity[1] * 50.0 / fps * (frame - path.last_frame)
    if math.hypot(candidate.x - predicted_x, candidate.y - predicted_y) > (
        config.prediction_radius * max(1.0, math.sqrt(frame - path.last_frame))
    ):
        return math.inf
    acceleration_cost = min(
        acceleration / config.acceleration_scale,
        config.kink_cost,
    )
    return unary + speed_cost + acceleration_cost


def select_consistent_path(
    frames: dict[int, list[Candidate]],
    fps: float,
    config: DecoderConfig = DecoderConfig(),
) -> list[tuple[int, Candidate | None]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frames:
        return []
    ordered_frames = list(range(min(frames), max(frames) + 1))
    beam = [
        _Path(
            cost=0.0,
            parent=None,
            choice=None,
            previous_frame=None,
            previous=None,
            last_frame=None,
            last=None,
            gap=0,
            steps=0,
        )
    ]
    for frame in ordered_frames:
        choices: list[Candidate | None] = [*frames.get(frame, []), None]
        expanded = []
        for path in beam:
            for candidate in choices:
                if candidate is None:
                    if config.maximum_gap is not None and path.gap >= config.maximum_gap:
                        continue
                    expanded.append(
                        _Path(
                            cost=path.cost + config.missing_cost,
                            parent=path,
                            choice=None,
                            previous_frame=path.previous_frame,
                            previous=path.previous,
                            last_frame=path.last_frame,
                            last=path.last,
                            gap=path.gap + 1,
                            steps=path.steps + 1,
                        )
                    )
                    continue
                expanded.append(
                    _Path(
                        cost=path.cost + _transition_cost(path, frame, candidate, fps, config),
                        parent=path,
                        choice=candidate,
                        previous_frame=path.last_frame,
                        previous=path.last,
                        last_frame=frame,
                        last=candidate,
                        gap=0,
                        steps=path.steps + 1,
                    )
                )
        if not expanded:
            raise RuntimeError(f"decoder beam exhausted at frame {frame}")
        expanded.sort(key=lambda path: path.cost)
        beam = expanded[: config.beam_width]
    selected = []
    path = beam[0]
    while path.parent is not None:
        selected.append(path.choice)
        path = path.parent
    selected.reverse()
    return list(zip(ordered_frames, selected, strict=True))


def interpolate_short_gaps(
    selected: list[tuple[int, Candidate | None]],
    maximum_gap: int,
    continuity_radius: float = DecoderConfig().interpolation_continuity_radius,
) -> list[tuple[int, Candidate | None]]:
    """Linearly fill bounded short gaps while preserving explicit provenance."""
    if maximum_gap <= 0:
        return selected
    output = list(selected)
    index = 0
    while index < len(output):
        if output[index][1] is not None:
            index += 1
            continue
        start = index
        while index < len(output) and output[index][1] is None:
            index += 1
        end = index
        gap = end - start
        if start < 2 or end + 1 >= len(output) or gap > maximum_gap:
            continue
        left_frame, left = output[start - 1]
        right_frame, right = output[end]
        prior_frame, prior = output[start - 2]
        next_frame, next_candidate = output[end + 1]
        if left is None or right is None or prior is None or next_candidate is None:
            continue
        span = right_frame - left_frame
        incoming_velocity = (
            (left.x - prior.x) / (left_frame - prior_frame),
            (left.y - prior.y) / (left_frame - prior_frame),
        )
        outgoing_velocity = (
            (next_candidate.x - right.x) / (next_frame - right_frame),
            (next_candidate.y - right.y) / (next_frame - right_frame),
        )
        predicted_right = (
            left.x + incoming_velocity[0] * span,
            left.y + incoming_velocity[1] * span,
        )
        predicted_left = (
            right.x - outgoing_velocity[0] * span,
            right.y - outgoing_velocity[1] * span,
        )
        if (
            max(
                math.dist(predicted_right, (right.x, right.y)),
                math.dist(predicted_left, (left.x, left.y)),
            )
            > continuity_radius
        ):
            continue
        for fill_index in range(start, end):
            frame = output[fill_index][0]
            fraction = (frame - left_frame) / span
            confidence = min(left.score, right.score) * (1.0 - 0.5 * abs(0.5 - fraction))
            output[fill_index] = (
                frame,
                Candidate(
                    x=left.x + fraction * (right.x - left.x),
                    y=left.y + fraction * (right.y - left.y),
                    score=confidence,
                    rank=max(left.rank, right.rank),
                    sources=("interpolated",),
                ),
            )
    return output


def load_candidate_streams(
    paths: list[Path],
    source_names: list[str],
    clips: set[str] | None,
    config: DecoderConfig,
) -> dict[str, dict[int, list[Candidate]]]:
    pooled: dict[str, dict[int, list[tuple[str, Candidate]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path, source in zip(paths, source_names, strict=True):
        res.require_artifact_space(path, CONSENSUS_PIXEL_SPACE, consumer="ball_track_consensus")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if clips is not None and row["clip"] not in clips:
                    continue
                pooled[row["clip"]][frame_number_from_name(row["frame"])].append(
                    (
                        source,
                        Candidate(
                            float(row["x"]),
                            float(row["y"]),
                            float(row["score"]),
                            int(row.get("rank", 0)),
                            (source,),
                        ),
                    )
                )
    output = {}
    for clip, frame_rows in pooled.items():
        merged = {frame: merge_candidates(rows, config) for frame, rows in frame_rows.items()}
        output[clip], _ = suppress_static_hotspots(merged, config)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument(
        "--consumer",
        choices=("integrity", "availability"),
        default="integrity",
    )
    parser.add_argument("--interpolation-gap", type=int, default=None)
    parser.add_argument("--hotspot-bin", type=float, default=DecoderConfig().hotspot_bin)
    args = parser.parse_args()
    if len(args.candidates) != len(args.source):
        parser.error("--candidates and --source counts must match")
    config = config_for_consumer(
        args.fps,
        args.consumer,
        args.interpolation_gap,
    )
    config = DecoderConfig(
        **{
            **config.__dict__,
            "hotspot_bin": args.hotspot_bin,
        }
    )
    streams = load_candidate_streams(
        args.candidates,
        args.source,
        set(args.clip) or None,
        config,
    )
    rows = []
    for clip, frames in sorted(streams.items()):
        selected = select_consistent_path(frames, args.fps, config)
        selected = interpolate_short_gaps(
            selected,
            config.interpolation_gap,
            config.interpolation_continuity_radius,
        )
        for frame, candidate in selected:
            if candidate is None:
                continue
            rows.append(
                {
                    "clip": clip,
                    "frame": f"f_{frame:04d}.jpg",
                    "x": candidate.x,
                    "y": candidate.y,
                    "track_id": 0,
                    "score": candidate.score,
                    "rank": candidate.rank,
                    "sources": "+".join(candidate.sources),
                }
            )
    write_track_artifact(args.output, rows, args.candidates)
    print(f"wrote {len(rows)} selected observations across {len(streams)} clips -> {args.output}")


if __name__ == "__main__":
    main()
