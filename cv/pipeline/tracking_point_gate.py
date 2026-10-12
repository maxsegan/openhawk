"""Compute label-free point retention decisions from tracking evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cv" / "pipeline"))

from ball_track_hypotheses import assess_local_track  # noqa: E402

from cv.pipeline import resolution as res  # noqa: E402


# Support radius, gap and arc thresholds below are calibrated in the space the shipped
# tracking artifacts declare.  The sidecar decides whether an artifact may be read here.
GATE_PIXEL_SPACE = res.LEGACY_TRACKING_SIZE

DEFAULT_FAILURE_THRESHOLDS = {
    "minimum_track_coverage_rate": 0.60,
    "maximum_track_gap_seconds": 0.60,
    "maximum_track_fragments_per_second": 1.50,
    "minimum_candidate_support_rate": 0.94,
    "maximum_nonballistic_window_rate": 0.45,
}

DEFAULT_ARC_FAILURE_THRESHOLDS = {
    "minimum_observed_frames": 4,
    "minimum_coverage_rate": 0.60,
    "minimum_candidate_support_rate": 0.94,
    "maximum_p90_innovation_mahalanobis": 12.0,
    "maximum_median_innovation_cov_trace_native": 160.0,
}


def risk_weights(gates: dict, requested_profile: str | None = None) -> tuple[str, dict[str, float]]:
    profiles = gates.get("point_quality_risk_profiles", {})
    if requested_profile is not None:
        if requested_profile not in profiles:
            raise ValueError(f"unknown tracking risk profile: {requested_profile}")
        return requested_profile, profiles[requested_profile]
    default_profile = gates.get("point_quality_risk_profile")
    if default_profile is not None:
        if default_profile not in profiles:
            raise ValueError(f"unknown default tracking risk profile: {default_profile}")
        return default_profile, profiles[default_profile]
    return "legacy_inline_weights", gates["point_quality_risk_weights"]


def frame_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact; retained for validation consumers."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=32)
def read_rows_by_clip(path: Path) -> dict[str, tuple[dict[str, str], ...]]:
    """Index a match CSV once instead of rescanning it for every point."""
    rows = read_rows(path)
    if rows:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["clip"]].append(row)
        return {clip: tuple(members) for clip, members in grouped.items()}
    return {}


def contiguous_fragments(frames: set[int]) -> int:
    ordered = sorted(frames)
    return sum(index == 0 or frame > ordered[index - 1] + 1 for index, frame in enumerate(ordered))


def integer_scope_frames(spans: list[list[float]], frame_count: int) -> set[int]:
    frames = set()
    for start, end in spans:
        first = max(1, math.ceil(float(start)))
        last = min(frame_count, math.floor(float(end)))
        frames.update(range(first, last + 1))
    return frames


def maximum_gap_frames(observed: set[int], spans: list[list[float]], frame_count: int) -> int:
    maximum = 0
    for start, end in spans:
        first = max(1, math.ceil(float(start)))
        last = min(frame_count, math.floor(float(end)))
        if last < first:
            continue
        local = sorted(frame for frame in observed if first <= frame <= last)
        boundaries = [first - 1, *local, last + 1]
        maximum = max(
            maximum,
            max((right - left - 1 for left, right in zip(boundaries, boundaries[1:])), default=0),
        )
    return maximum


def tracking_failure_reasons(
    row: dict,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    limits = DEFAULT_FAILURE_THRESHOLDS | (thresholds or {})
    reasons = []
    if row["track_coverage_rate"] < limits["minimum_track_coverage_rate"]:
        reasons.append("insufficient_live_track_coverage")
    if row["maximum_track_gap_seconds"] > limits["maximum_track_gap_seconds"]:
        reasons.append("excessive_live_track_gap")
    if row["track_fragments_per_second"] > limits["maximum_track_fragments_per_second"]:
        reasons.append("excessive_live_track_fragmentation")
    if row["candidate_support_rate"] < limits["minimum_candidate_support_rate"]:
        reasons.append("insufficient_candidate_support")
    if row["nonballistic_window_rate"] > limits["maximum_nonballistic_window_rate"]:
        reasons.append("excessive_nonballistic_windows")
    crop_limit = limits.get("maximum_crop_branch_frame_rate")
    if crop_limit is not None and row["crop_branch_frame_rate"] > crop_limit:
        reasons.append("excessive_crop_branch_dependence")
    return reasons


def candidate_support_rate(
    track_rows: list[dict],
    candidate_sources: list[list[dict]],
    radius: float = 12.0,
) -> float:
    by_source = []
    for rows in candidate_sources:
        by_frame = defaultdict(list)
        for row in rows:
            by_frame[row["clip"], frame_number(row["frame"])].append(
                (float(row["x"]), float(row["y"]))
            )
        by_source.append(by_frame)
    supported = 0
    for row in track_rows:
        key = row["clip"], frame_number(row["frame"])
        point = float(row["x"]), float(row["y"])
        if any(
            any(math.dist(point, candidate) <= radius for candidate in source.get(key, []))
            for source in by_source
        ):
            supported += 1
    return supported / max(len(track_rows), 1)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.inf
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def motion_arcs(
    track_rows: list[dict],
    scope_frames: set[int],
    candidate_sources: list[list[dict]],
    thresholds: dict[str, float] | None = None,
) -> list[dict]:
    """Split a motion track at emitted regime switches and gate each run independently."""
    if not track_rows or not any(row.get("regime") for row in track_rows):
        return []
    limits = DEFAULT_ARC_FAILURE_THRESHOLDS | (thresholds or {})
    ordered = sorted(track_rows, key=lambda row: frame_number(row["frame"]))
    runs: list[list[dict]] = []
    for row in ordered:
        if not runs or row.get("regime") != runs[-1][-1].get("regime"):
            runs.append([row])
        else:
            runs[-1].append(row)
    arcs = []
    for arc_id, rows in enumerate(runs):
        frames = {frame_number(row["frame"]) for row in rows}
        first, last = min(frames), max(frames)
        local_scope = {frame for frame in scope_frames if first <= frame <= last}
        coverage = len(frames & local_scope) / max(len(local_scope), 1)
        local_candidates = [
            [row for row in source if first <= frame_number(row["frame"]) <= last]
            for source in candidate_sources
        ]
        innovations = [
            float(row["innovation_mahalanobis"])
            for row in rows
            if row.get("innovation_mahalanobis") not in (None, "")
        ]
        covariance_traces = [
            float(row["innovation_cov_xx_native"]) + float(row["innovation_cov_yy_native"])
            for row in rows
            if row.get("innovation_cov_xx_native") not in (None, "")
            and row.get("innovation_cov_yy_native") not in (None, "")
        ]
        support = candidate_support_rate(rows, local_candidates)
        p90_innovation = _percentile(innovations, 90.0)
        median_covariance = float(np.median(covariance_traces)) if covariance_traces else math.inf
        reasons = []
        if len(rows) < limits["minimum_observed_frames"]:
            reasons.append("too_few_arc_observations")
        if coverage < limits["minimum_coverage_rate"]:
            reasons.append("insufficient_arc_coverage")
        if support < limits["minimum_candidate_support_rate"]:
            reasons.append("insufficient_arc_candidate_support")
        if p90_innovation > limits["maximum_p90_innovation_mahalanobis"]:
            reasons.append("excessive_arc_innovation")
        if median_covariance > limits["maximum_median_innovation_cov_trace_native"]:
            reasons.append("excessive_arc_innovation_covariance")
        arcs.append(
            {
                "arc_id": arc_id,
                "regime": rows[0].get("regime", "unknown"),
                "start_frame": first,
                "end_frame": last,
                "scope_frames": len(local_scope),
                "observed_frames": len(frames & local_scope),
                "coverage_rate": coverage,
                "candidate_support_rate": support,
                "p90_innovation_mahalanobis": p90_innovation,
                "median_innovation_cov_trace_native": median_covariance,
                "decision": "hold" if reasons else "retain",
                "failure_reasons": reasons,
            }
        )
    return arcs


def arc_point_decision(arcs: list[dict]) -> str:
    """Retain a point when any independently gated motion arc is usable."""
    if not arcs:
        return "unavailable"
    return "retain" if any(arc.get("decision") == "retain" for arc in arcs) else "hold"


def point_features(
    match_id: str,
    match_out: Path,
    clip: str,
    fps: float,
    spans: list[list[float]] | None = None,
    failure_thresholds: dict[str, float] | None = None,
    arc_failure_thresholds: dict[str, float] | None = None,
) -> dict:
    track_path = match_out / "ball_track_joint_native1080_arc_augmented_v2.csv"
    res.require_artifact_space(track_path, GATE_PIXEL_SPACE, consumer="tracking_point_gate")
    track_rows = list(read_rows_by_clip(track_path).get(clip, ()))
    candidate_names = [
        "ball_candidates_wasb_native1080_sliding_k5_v1.csv",
        "ball_candidates_tracknetv2_native1080_sliding_k5_v1.csv",
        "ball_candidates_wasb_native1080_branched_crop_v2.csv",
        "ball_candidates_tracknetv2_native1080_branched_crop_v2.csv",
    ]
    candidate_names.extend(
        name
        for name in (
            "ball_candidates_wasb_native1080_far_native_v1.csv",
            "ball_candidates_tracknetv2_native1080_far_native_v1.csv",
        )
        if (match_out / name).is_file()
    )
    for name in candidate_names:
        res.require_artifact_space(
            match_out / name, GATE_PIXEL_SPACE, consumer="tracking_point_gate"
        )
    candidate_sources = [
        list(read_rows_by_clip(match_out / name).get(clip, ())) for name in candidate_names
    ]
    frame_count = len(list((match_out / "audit_frames_native_1080" / clip).glob("f_*.jpg")))
    scope_source = "full_clip"
    if spans:
        scope_source = "automatic_live_spans"
    else:
        spans = [[1.0, float(frame_count)]]
    scope_frames = integer_scope_frames(spans, frame_count)
    track_rows = [row for row in track_rows if frame_number(row["frame"]) in scope_frames]
    candidate_sources = [
        [row for row in rows if frame_number(row["frame"]) in scope_frames]
        for rows in candidate_sources
    ]
    track_frames = {frame_number(row["frame"]) for row in track_rows}
    track = np.asarray(
        sorted(
            (
                frame_number(row["frame"]),
                float(row["x"]),
                float(row["y"]),
            )
            for row in track_rows
        ),
        dtype=float,
    )
    tested = failed = 0
    for start, end in spans:
        local_track = (
            track[(track[:, 0] >= float(start)) & (track[:, 0] <= float(end))]
            if len(track)
            else track
        )
        for frame in local_track[:, 0] if len(local_track) else []:
            verdict = assess_local_track(
                local_track,
                frame,
                fps,
                require_motion=True,
            )
            if verdict["reason"] in {"too_few_samples", "no_adjacent_samples"}:
                continue
            tested += 1
            failed += int(not verdict["usable"])
    crop_rows = candidate_sources[2]
    branched_frames = {
        frame_number(row["frame"])
        for row in crop_rows
        if row.get("crop_provenance", "").endswith("_branch")
    }
    scope_count = len(scope_frames)
    scope_seconds = scope_count / fps if fps > 0 else 0.0
    fragments = contiguous_fragments(track_frames)
    maximum_gap = maximum_gap_frames(track_frames, spans, frame_count)
    output = {
        "match_id": match_id,
        "clip": clip,
        "frames": frame_count,
        "scope_source": scope_source,
        "scope_spans": spans,
        "scope_frames": scope_count,
        "scope_seconds": scope_seconds,
        "track_points": len(track_rows),
        "track_fragments": fragments,
        "track_fragments_per_second": fragments / max(scope_seconds, 1.0 / fps),
        "maximum_track_gap_seconds": maximum_gap / fps,
        "track_coverage_rate": len(track_frames) / max(scope_count, 1),
        "candidate_support_rate": candidate_support_rate(track_rows, candidate_sources),
        "nonballistic_window_rate": failed / max(tested, 1),
        "crop_branch_frame_rate": len(branched_frames) / max(scope_count, 1),
        "detour_replaced_points": sum(
            row.get("sources", "").startswith("transient_detour_replacement") for row in track_rows
        ),
        "arc_filled_points": sum(
            row.get("sources", "").startswith("local_gap_fill") for row in track_rows
        ),
    }
    output["tracking_failure_reasons"] = tracking_failure_reasons(
        output,
        failure_thresholds,
    )
    output["whole_point_diagnostic_decision"] = (
        "hold" if output["tracking_failure_reasons"] else "retain"
    )
    arcs = motion_arcs(
        track_rows,
        scope_frames,
        candidate_sources,
        arc_failure_thresholds,
    )
    output["arcs"] = arcs
    if arcs:
        retained_arc_frames = {
            frame
            for arc in arcs
            if arc["decision"] == "retain"
            for frame in scope_frames
            if arc["start_frame"] <= frame <= arc["end_frame"]
        }
        held_arc_frames = scope_frames - retained_arc_frames
        held_arcs = sum(arc["decision"] == "hold" for arc in arcs)
        output.update(
            {
                "arc_count": len(arcs),
                "retained_arc_count": len(arcs) - held_arcs,
                "held_arc_count": held_arcs,
                "arc_retained_frames": len(retained_arc_frames),
                "arc_held_frames": len(held_arc_frames),
                "arc_retained_frame_rate": len(retained_arc_frames) / max(scope_count, 1),
                "arc_decision_summary": (
                    "retain" if not held_arcs else "hold" if held_arcs == len(arcs) else "partial"
                ),
            }
        )
        # Arc abstention is not a whole-point verdict.  A point remains eligible when
        # at least one motion arc is retained; downstream consumers receive the arc
        # intervals and fit only observations inside them.
        output["failure_reason_decision"] = arc_point_decision(arcs)
        if held_arc_frames:
            output["tracking_failure_reasons"] = [
                *output["tracking_failure_reasons"],
                "held_motion_arc",
            ]
    else:
        output.update(
            {
                "arc_count": 0,
                "retained_arc_count": 0,
                "held_arc_count": 0,
                "arc_retained_frames": 0,
                "arc_held_frames": scope_count,
                "arc_retained_frame_rate": 0.0,
                "arc_decision_summary": "unavailable",
            }
        )
        output["failure_reason_decision"] = (
            "hold" if output["tracking_failure_reasons"] else "retain"
        )
    return output


def assign_decisions(
    rows: list[dict],
    weights: dict[str, float],
    maximum_hold_fraction: float,
) -> list[dict]:
    means = {feature: float(np.mean([float(row[feature]) for row in rows])) for feature in weights}
    scales = {feature: float(np.std([float(row[feature]) for row in rows])) for feature in weights}
    output = []
    for row in rows:
        components = {}
        for feature, weight in weights.items():
            scale = scales[feature] or 1.0
            components[feature] = weight * (float(row[feature]) - means[feature]) / scale
        output.append(
            {
                **row,
                "quality_risk": float(sum(components.values())),
                "risk_components": components,
            }
        )
    output.sort(key=lambda row: (-row["quality_risk"], row["match_id"], row["clip"]))
    hold_count = math.floor(len(output) * maximum_hold_fraction)
    for rank, row in enumerate(output, start=1):
        row["risk_rank"] = rank
        row["decision"] = "hold" if rank <= hold_count else "retain"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--active-play", type=Path)
    parser.add_argument("--risk-profile")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    gates = protocol["gates"]
    failure_thresholds = gates.get(
        "tracking_failure_thresholds",
        DEFAULT_FAILURE_THRESHOLDS,
    )
    arc_failure_thresholds = gates.get(
        "tracking_arc_failure_thresholds",
        DEFAULT_ARC_FAILURE_THRESHOLDS,
    )
    active_play = json.loads(args.active_play.read_text()) if args.active_play else {}
    rows = []
    for match in manifest["matches"]:
        match_out = args.out / match["id"]
        point_ids = match.get(
            "point_ids",
            range(1, int(manifest["points_per_match"]) + 1),
        )
        for point in point_ids:
            clip = f"pt{point:04d}"
            phase = active_play.get(f"{match['id']}/{clip}", {})
            spans = phase.get("event_spans", phase.get("active_spans"))
            rows.append(
                point_features(
                    match["id"],
                    match_out,
                    clip,
                    float(match["source_fps"]),
                    spans=spans,
                    failure_thresholds=failure_thresholds,
                    arc_failure_thresholds=arc_failure_thresholds,
                )
            )
    risk_profile, weights = risk_weights(gates, args.risk_profile)
    scored = assign_decisions(
        rows,
        weights,
        float(gates["maximum_point_abstention_fraction"]),
    )
    # Calibrated-absolute mode: the binding decision is the per-point absolute
    # threshold verdict; the cohort-relative risk rank stays as a diagnostic only.
    # Rationale: on 89 owner tracking-quality labels the fixed-fraction rank hold
    # agreed 56% with the owner (7 false accepts, 32 good points held); absolute
    # thresholds are per-unit, interpretable, and calibrated to the same labels.
    if gates.get("tracking_decision_mode") == "absolute_calibrated_v1":
        for row in scored:
            row["decision"] = row["failure_reason_decision"]
    csv_path = args.out / "untouched_tracking_point_gate_v1.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [key for key in scored[0] if key not in {"risk_components", "arcs"}]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key not in {"risk_components", "arcs"}}
            for row in scored
        )
    arcs = [
        {"match_id": row["match_id"], "clip": row["clip"], **arc}
        for row in scored
        for arc in row["arcs"]
    ]
    report = {
        "schema": "untouched_tracking_point_gate_v1",
        "labels_loaded": False,
        "points": len(scored),
        "held": sum(row["decision"] == "hold" for row in scored),
        "retained": sum(row["decision"] == "retain" for row in scored),
        "failure_reason_held": sum(row["failure_reason_decision"] == "hold" for row in scored),
        "whole_point_diagnostic_held": sum(
            row["whole_point_diagnostic_decision"] == "hold" for row in scored
        ),
        "failure_thresholds": failure_thresholds,
        "arc_failure_thresholds": arc_failure_thresholds,
        "arc_count": len(arcs),
        "retained_arcs": sum(arc["decision"] == "retain" for arc in arcs),
        "held_arcs": sum(arc["decision"] == "hold" for arc in arcs),
        "arc_scope_frames": sum(row["scope_frames"] for row in scored),
        "arc_retained_frames": sum(row["arc_retained_frames"] for row in scored),
        "arc_held_frames": sum(row["arc_held_frames"] for row in scored),
        "arc_retained_frame_rate": sum(row["arc_retained_frames"] for row in scored)
        / max(sum(row["scope_frames"] for row in scored), 1),
        "risk_profile": risk_profile,
        "weights": weights,
        "rows": scored,
        "arcs": arcs,
    }
    json_path = args.out / "untouched_tracking_point_gate_v1.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
