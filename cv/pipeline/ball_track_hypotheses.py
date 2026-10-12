"""Build continuous, provenance-carrying ball-track hypotheses.

The integrity track remains authoritative wherever it has a locally coherent observation.
Alternate arms may fill only frames where the integrity arm abstains. Quality is evaluated
at every observed frame, without event proposals or labels, so downstream benchmarks can
measure inference-time coverage rather than consulting truth to choose recovery windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from event_refine import refine_anchor


@dataclass(frozen=True)
class HypothesisConfig:
    half_window_seconds: float = 0.24
    search_seconds: float = 0.08
    maximum_teleport_rate: float = 0.15
    teleport_speed_px_s: float = 2250.0
    minimum_motion_px_s: float = 60.0
    strict_rmse: float = 6.0
    marginal_rmse: float = 6.5
    agreement_rmse: float = 8.0
    comparator_rmse: float = 10.0
    agreement_radius_px: float = 3.0


def assess_local_track(
    track: np.ndarray | None,
    frame: float,
    fps: float,
    *,
    require_motion: bool,
    config: HypothesisConfig = HypothesisConfig(),
) -> dict:
    if fps <= 0:
        raise ValueError("fps must be positive")
    verdict = {
        "samples": 0,
        "median_speed_px_s": None,
        "teleport_rate": None,
        "fit_rmse": None,
        "usable": False,
        "reason": "no_track",
    }
    if track is None or len(track) == 0:
        return verdict
    track = np.asarray(track, dtype=float)
    window = config.half_window_seconds * fps
    minimum_samples = max(5, round(10 * fps / 50.0))
    left = int(np.searchsorted(track[:, 0], frame - window, side="left"))
    right = int(np.searchsorted(track[:, 0], frame + window, side="right"))
    local = track[left:right]
    verdict["samples"] = int(len(local))
    if len(local) < minimum_samples:
        verdict["reason"] = "too_few_samples"
        return verdict
    frame_steps = np.diff(local[:, 0])
    maximum_adjacent_gap = max(1, round(2 * fps / 50.0))
    adjacent = frame_steps <= maximum_adjacent_gap
    if not np.any(adjacent):
        verdict["reason"] = "no_adjacent_samples"
        return verdict
    displacement = np.hypot(np.diff(local[:, 1]), np.diff(local[:, 2]))
    speeds = displacement[adjacent] * fps / np.maximum(frame_steps[adjacent], 1.0)
    verdict["median_speed_px_s"] = float(np.median(speeds))
    verdict["teleport_rate"] = float(np.mean(speeds > config.teleport_speed_px_s))
    result = refine_anchor(
        local[:, 0],
        local[:, 1],
        local[:, 2],
        hint=frame,
        search=config.search_seconds * fps,
        window=window,
        step=0.1,
    )
    verdict["fit_rmse"] = float(result.rmse) if result.ok else None
    if require_motion and verdict["median_speed_px_s"] < config.minimum_motion_px_s:
        verdict["reason"] = "static_lock"
    elif verdict["teleport_rate"] > config.maximum_teleport_rate:
        verdict["reason"] = "teleporting"
    elif not result.ok or result.rmse > config.strict_rmse:
        verdict["reason"] = "not_ballistic"
    else:
        verdict["usable"] = True
        verdict["reason"] = "ok"
    return verdict


def exact_point(track: np.ndarray | None, frame: int) -> tuple[float, float] | None:
    if track is None:
        return None
    rows = track[track[:, 0] == frame]
    if not len(rows):
        return None
    return float(rows[0, 1]), float(rows[0, 2])


def _agreement_distance(
    point: tuple[float, float] | None,
    comparators: list[tuple[tuple[float, float] | None, dict]],
    config: HypothesisConfig,
) -> float | None:
    if point is None:
        return None
    distances = [
        math.dist(point, comparator)
        for comparator, quality in comparators
        if comparator is not None
        and quality["reason"] in {"ok", "not_ballistic"}
        and quality["fit_rmse"] is not None
        and quality["fit_rmse"] <= config.comparator_rmse
    ]
    return min(distances) if distances else None


def evidence_tier(
    quality: dict,
    agreement_px: float | None,
    config: HypothesisConfig = HypothesisConfig(),
) -> str:
    if quality["usable"]:
        return "strict"
    fit_rmse = quality["fit_rmse"]
    if (
        quality["reason"] == "not_ballistic"
        and fit_rmse is not None
        and fit_rmse <= config.marginal_rmse
    ):
        return "marginal"
    if (
        quality["reason"] == "not_ballistic"
        and fit_rmse is not None
        and fit_rmse <= config.agreement_rmse
        and agreement_px is not None
        and agreement_px <= config.agreement_radius_px
    ):
        return "agreement"
    return "reject"


def build_continuous_hypotheses(
    tracks: dict[str, np.ndarray | None],
    fps: float,
    *,
    primary_arm: str,
    evidence_arm: str,
    arm_priority: tuple[str, ...],
    frame_start: int,
    frame_end: int,
    maximum_speed_px_s: float = 4000.0,
    config: HypothesisConfig = HypothesisConfig(),
) -> list[dict]:
    if primary_arm not in tracks:
        raise ValueError(f"missing primary arm: {primary_arm}")
    if evidence_arm not in tracks:
        raise ValueError(f"missing evidence arm: {evidence_arm}")
    unknown = set(arm_priority) - set(tracks)
    if unknown:
        raise ValueError(f"unknown arms in priority: {sorted(unknown)}")
    if frame_end < frame_start:
        raise ValueError("frame_end must not precede frame_start")

    point_maps = {
        arm: (
            {}
            if track is None
            else {
                int(row[0]): (float(row[1]), float(row[2]))
                for row in np.asarray(track, dtype=float)
            }
        )
        for arm, track in tracks.items()
    }
    rows = []
    for frame in range(frame_start, frame_end + 1):
        points = {arm: point_map.get(frame) for arm, point_map in point_maps.items()}
        primary_observed = points[primary_arm] is not None
        qualities = {}
        if not primary_observed:
            qualities = {
                arm: assess_local_track(
                    track,
                    frame,
                    fps,
                    require_motion=False,
                    config=config,
                )
                for arm, track in tracks.items()
                if arm != primary_arm and points[arm] is not None
            }
        tiers = {
            arm: "strict" if quality["usable"] else "reject" for arm, quality in qualities.items()
        }
        evidence_quality = qualities.get(evidence_arm)
        agreement_px = None
        if evidence_quality is not None:
            agreement_px = _agreement_distance(
                points[evidence_arm],
                [
                    (points[arm], quality)
                    for arm, quality in qualities.items()
                    if arm != evidence_arm
                ],
                config,
            )
            tiers[evidence_arm] = evidence_tier(
                evidence_quality,
                agreement_px,
                config,
            )

        source = "none"
        if primary_observed:
            source = primary_arm
            tiers[primary_arm] = "strict"
        else:
            for arm in arm_priority:
                if arm == primary_arm:
                    continue
                if points[arm] is not None and tiers.get(arm) != "reject":
                    source = arm
                    break
        selected_point = points.get(source)
        selected_quality = qualities.get(source)
        rows.append(
            {
                "frame": frame,
                "source": source,
                "quality_tier": tiers.get(source, "reject"),
                "x": selected_point[0] if selected_point is not None else None,
                "y": selected_point[1] if selected_point is not None else None,
                "fit_rmse": (
                    selected_quality["fit_rmse"] if selected_quality is not None else None
                ),
                "primary_reason": ("trusted_observation" if primary_observed else "no_observation"),
                "primary_observed": primary_observed,
                "evidence_agreement_px": agreement_px,
                "proposal_rejection": None,
            }
        )
    return enforce_speed_safety(
        rows,
        fps,
        primary_arm,
        maximum_speed_px_s=maximum_speed_px_s,
    )


def build_continuous_hypothesis_layers(
    tracks: dict[str, np.ndarray | None],
    fps: float,
    *,
    evidence_arm: str,
    arm_priority: tuple[str, ...],
    frame_start: int,
    frame_end: int,
    config: HypothesisConfig = HypothesisConfig(),
) -> list[dict]:
    """Score parallel trajectory layers at every frame without selecting a winner."""
    if evidence_arm not in tracks:
        raise ValueError(f"missing evidence arm: {evidence_arm}")
    unknown = set(arm_priority) - set(tracks)
    if unknown:
        raise ValueError(f"unknown arms in priority: {sorted(unknown)}")
    if frame_end < frame_start:
        raise ValueError("frame_end must not precede frame_start")

    point_maps = {
        arm: (
            {}
            if track is None
            else {
                int(row[0]): (float(row[1]), float(row[2]))
                for row in np.asarray(track, dtype=float)
            }
        )
        for arm, track in tracks.items()
    }
    rows = []
    for frame in range(frame_start, frame_end + 1):
        points = {arm: point_maps[arm].get(frame) for arm in arm_priority}
        qualities = {
            arm: assess_local_track(
                tracks[arm],
                frame,
                fps,
                require_motion=True,
                config=config,
            )
            for arm in arm_priority
        }
        tiers = {arm: "strict" if qualities[arm]["usable"] else "reject" for arm in arm_priority}
        evidence_point = points[evidence_arm]
        agreement_px = _agreement_distance(
            evidence_point,
            [(points[arm], qualities[arm]) for arm in arm_priority if arm != evidence_arm],
            config,
        )
        tiers[evidence_arm] = evidence_tier(
            qualities[evidence_arm],
            agreement_px,
            config,
        )
        for arm in arm_priority:
            point = points[arm]
            quality = qualities[arm]
            rows.append(
                {
                    "frame": frame,
                    "arm": arm,
                    "observed": point is not None,
                    "x": point[0] if point is not None else None,
                    "y": point[1] if point is not None else None,
                    "quality_tier": tiers[arm],
                    "samples": quality["samples"],
                    "median_speed_px_s": quality["median_speed_px_s"],
                    "teleport_rate": quality["teleport_rate"],
                    "fit_rmse": quality["fit_rmse"],
                    "reason": quality["reason"],
                    "evidence_agreement_px": (agreement_px if arm == evidence_arm else None),
                }
            )
    return rows


def enforce_speed_safety(
    rows: list[dict],
    fps: float,
    primary_arm: str,
    *,
    maximum_speed_px_s: float,
) -> list[dict]:
    output = [dict(row) for row in rows]
    selected = {row["frame"]: row for row in output if row["source"] != "none"}
    proposed_frames = sorted(
        row["frame"] for row in output if row["source"] not in {"none", primary_arm}
    )
    runs = []
    if proposed_frames:
        start = previous = proposed_frames[0]
        for frame in proposed_frames[1:]:
            if frame != previous + 1:
                runs.append((start, previous))
                start = frame
            previous = frame
        runs.append((start, previous))

    rejected = set()
    observed = sorted(selected)
    for start, end in runs:
        run = set(range(start, end + 1))
        prior = [frame for frame in observed if frame < start and frame not in run]
        following = [frame for frame in observed if frame > end and frame not in run]
        check_frames = [
            *([prior[-1]] if prior else []),
            *[frame for frame in range(start, end + 1) if frame in selected],
            *([following[0]] if following else []),
        ]
        unsafe = any(
            math.dist(
                (selected[left]["x"], selected[left]["y"]),
                (selected[right]["x"], selected[right]["y"]),
            )
            * fps
            / (right - left)
            >= maximum_speed_px_s
            for left, right in zip(check_frames, check_frames[1:])
        )
        if unsafe:
            rejected.update(run)

    for row in output:
        if row["frame"] not in rejected or row["source"] in {"none", primary_arm}:
            continue
        row.update(
            {
                "source": "none",
                "quality_tier": "reject",
                "x": None,
                "y": None,
                "fit_rmse": None,
                "proposal_rejection": "speed_unsafe",
            }
        )
    return output


def hypothesis_segments(rows: list[dict], primary_arm: str) -> list[dict]:
    segments = []
    start = None
    previous = None
    source = None
    tier = None
    for row in rows:
        proposed = row["source"] not in {"none", primary_arm}
        continues = (
            proposed
            and start is not None
            and row["frame"] == previous + 1
            and row["source"] == source
            and row["quality_tier"] == tier
        )
        if not continues and start is not None:
            segments.append(
                {
                    "start_frame": start,
                    "end_frame": previous,
                    "source": source,
                    "quality_tier": tier,
                }
            )
            start = None
        if proposed and start is None:
            start = row["frame"]
            source = row["source"]
            tier = row["quality_tier"]
        previous = row["frame"]
    if start is not None:
        segments.append(
            {
                "start_frame": start,
                "end_frame": previous,
                "source": source,
                "quality_tier": tier,
            }
        )
    return segments


def layer_segments(rows: list[dict]) -> list[dict]:
    segments = []
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, arm_rows in by_arm.items():
        start = None
        previous = None
        tier = None
        for row in arm_rows:
            accepted = row["quality_tier"] != "reject"
            continues = (
                accepted
                and start is not None
                and row["frame"] == previous + 1
                and row["quality_tier"] == tier
            )
            if not continues and start is not None:
                segments.append(
                    {
                        "start_frame": start,
                        "end_frame": previous,
                        "arm": arm,
                        "quality_tier": tier,
                    }
                )
                start = None
            if accepted and start is None:
                start = row["frame"]
                tier = row["quality_tier"]
            previous = row["frame"]
        if start is not None:
            segments.append(
                {
                    "start_frame": start,
                    "end_frame": previous,
                    "arm": arm,
                    "quality_tier": tier,
                }
            )
    return segments
