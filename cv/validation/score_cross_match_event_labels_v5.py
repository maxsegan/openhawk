"""Score frozen v5 emissions against the adjudicated active-play truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "data" / "processed" / "cross_match_event_audit_v5"
DEFAULT_LABELS = (
    ROOT
    / "cv"
    / "validation"
    / "labels"
    / "cross_match_event_audit_v5"
    / "contact_labels_clean.csv"
)
DEFAULT_EMISSIONS = AUDIT_ROOT / "generalization_event_inference_v1" / "emissions.json"
DEFAULT_PROPOSALS = AUDIT_ROOT / "generalization_event_inference_review_all_v1" / "emissions.json"
DEFAULT_GATE = AUDIT_ROOT / "point_validity_gate_v1.json"
DEFAULT_CADENCE = AUDIT_ROOT / "frame_cadence_audit_v1.json"
DEFAULT_OUTPUT = AUDIT_ROOT / "frozen_v5_event_benchmark_v1.json"
ACCEPTED = {"confirmed", "adjusted", "new"}
EVENT_TYPES = {"contact", "bounce", "net_hit"}


def canonical_clip(row: dict) -> str:
    clip = row["clip"]
    return clip if "__" in clip else f"{row['match_id']}__{clip}"


def load_truth(
    path: Path, *, allow_truth_outside_spans: bool = False
) -> tuple[list[dict], dict[str, list[tuple[float, float]]], dict]:
    rows = list(csv.DictReader(path.open()))
    truth = []
    endings: dict[str, list[float]] = defaultdict(list)
    no_point = set()
    tracking_quality = {}
    completed = set()
    for row in rows:
        clip = row["clip"]
        if row["event_type"] in EVENT_TYPES and row["verdict"] in ACCEPTED:
            frame = row["labeled_frame"] or row["seed_frame"]
            if frame:
                truth.append(
                    {
                        "clip": clip,
                        "event_type": row["event_type"],
                        "frame": float(frame),
                    }
                )
        elif row["event_type"] == "point_end":
            if row["verdict"] == "no_point":
                no_point.add(clip)
            elif row["labeled_frame"]:
                endings[clip].append(float(row["labeled_frame"]))
        elif row["event_type"] == "tracking_quality":
            tracking_quality[clip] = row["verdict"]
        elif row["event_type"] == "coverage" and row["verdict"] == "complete":
            completed.add(clip)

    truth.sort(key=lambda row: (row["clip"], row["frame"], row["event_type"]))
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in truth:
        by_clip[row["clip"]].append(row)
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    truth_outside_spans = []
    for clip in completed:
        previous_end = float("-inf")
        for ending in sorted(endings.get(clip, [])):
            attempt_truth = [
                row for row in by_clip.get(clip, []) if previous_end < row["frame"] <= ending + 0.5
            ]
            if attempt_truth:
                spans[clip].append((attempt_truth[0]["frame"], ending))
            previous_end = ending
        uncovered = [
            row
            for row in by_clip.get(clip, [])
            if not any(start <= row["frame"] <= end + 0.5 for start, end in spans[clip])
        ]
        truth_outside_spans.extend(uncovered)
        if uncovered and not allow_truth_outside_spans:
            raise ValueError(f"truth outside point-ending spans for {clip}: {uncovered}")
    if missing := completed - set(endings) - no_point:
        raise ValueError(f"completed clips lack point endings: {sorted(missing)}")
    return (
        truth,
        spans,
        {
            "rows": len(rows),
            "completed": completed,
            "no_point": no_point,
            "tracking_quality": tracking_quality,
            "endings": endings,
            "truth_outside_point_end_spans": truth_outside_spans,
        },
    )


def load_predictions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload["emissions"] if isinstance(payload, dict) else payload
    return [
        {
            **row,
            "clip": canonical_clip(row),
            "frame": float(row["frame"]),
        }
        for row in rows
        if row["event_type"] in EVENT_TYPES
    ]


def scope_predictions(
    predictions: list[dict],
    spans: dict[str, list[tuple[float, float]]],
    *,
    boundary_margin_frames: float = 0.0,
    include_unscoped_clips: set[str] | None = None,
) -> list[dict]:
    include_unscoped_clips = include_unscoped_clips or set()
    return [
        row
        for row in predictions
        if row["clip"] in include_unscoped_clips
        or any(
            start - boundary_margin_frames <= row["frame"] <= end + boundary_margin_frames
            for start, end in spans.get(row["clip"], [])
        )
    ]


def match_events(
    predictions: list[dict], truth: list[dict], tolerance: float
) -> tuple[list, list, list]:
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, target in enumerate(truth):
            if (
                prediction["clip"] != target["clip"]
                or prediction["event_type"] != target["event_type"]
            ):
                continue
            delta = abs(prediction["frame"] - target["frame"])
            if delta <= tolerance:
                candidates.append((delta, prediction_index, truth_index))
    used_predictions = set()
    used_truth = set()
    matches = []
    for _, prediction_index, truth_index in sorted(candidates):
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        matches.append((predictions[prediction_index], truth[truth_index]))
    return (
        matches,
        [row for index, row in enumerate(predictions) if index not in used_predictions],
        [row for index, row in enumerate(truth) if index not in used_truth],
    )


def _counts(matches: list, false_positives: list, false_negatives: list) -> dict:
    true_positives = len(matches)
    predictions = true_positives + len(false_positives)
    truth = true_positives + len(false_negatives)
    return {
        "truth": truth,
        "predictions": predictions,
        "true_positives": true_positives,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": true_positives / predictions if predictions else None,
        "recall": true_positives / truth if truth else None,
    }


def score(predictions: list[dict], truth: list[dict], clips: set[str], tolerance: float) -> dict:
    scoped_predictions = [row for row in predictions if row["clip"] in clips]
    scoped_truth = [row for row in truth if row["clip"] in clips]
    matches, false_positives, false_negatives = match_events(
        scoped_predictions, scoped_truth, tolerance
    )
    result = _counts(matches, false_positives, false_negatives)
    result["by_type"] = {}
    for event_type in sorted(EVENT_TYPES):
        typed_matches, typed_fp, typed_fn = match_events(
            [row for row in scoped_predictions if row["event_type"] == event_type],
            [row for row in scoped_truth if row["event_type"] == event_type],
            tolerance,
        )
        result["by_type"][event_type] = _counts(typed_matches, typed_fp, typed_fn)
    per_point = {}
    for clip in sorted(clips):
        point_predictions = [row for row in scoped_predictions if row["clip"] == clip]
        point_truth = [row for row in scoped_truth if row["clip"] == clip]
        point_matches, point_fp, point_fn = match_events(point_predictions, point_truth, tolerance)
        counts = _counts(point_matches, point_fp, point_fn)
        counts["complete"] = not point_fp and not point_fn
        per_point[clip] = counts
    event_points = {row["clip"] for row in scoped_truth}
    result["points"] = len(clips)
    result["event_points"] = len(event_points)
    result["complete_event_points"] = sum(per_point[clip]["complete"] for clip in event_points)
    result["complete_event_point_rate"] = (
        result["complete_event_points"] / len(event_points) if event_points else None
    )
    result["false_positive_rows"] = false_positives
    result["false_negative_rows"] = false_negatives
    return result


def _gate_slices(path: Path) -> tuple[set[str], set[str]]:
    gate = json.loads(path.read_text())
    retained = {
        f"{row['match_id']}__{row['clip']}" for row in gate["rows"] if row["decision"] == "retain"
    }
    held = {
        f"{row['match_id']}__{row['clip']}" for row in gate["rows"] if row["decision"] != "retain"
    }
    return retained, held


def _cadence_held(path: Path) -> set[str]:
    document = json.loads(path.read_text())
    rows = document.get("rows", [])
    held = set()
    if isinstance(rows, dict):
        rows = rows.values()
    for row in rows:
        decision = row.get("decision") or row.get("timing_decision")
        if decision not in {None, "timing_usable", "pass"}:
            point = row.get("point")
            if not point and row.get("match_id") and row.get("clip"):
                point = f"{row['match_id']}__{row['clip']}"
            elif not point:
                point = row.get("clip")
            if point:
                held.add(point.replace("/", "__"))
    return held


def benchmark(labels: Path, emissions: Path, proposals: Path, gate: Path, cadence: Path) -> dict:
    truth, spans, meta = load_truth(labels)
    all_clips = set(meta["completed"])
    retained, held = _gate_slices(gate)
    cadence_held = _cadence_held(cadence)
    timing_usable = all_clips - cadence_held
    streams = {
        "frozen_emissions": scope_predictions(load_predictions(emissions), spans),
        "frozen_review_proposals": scope_predictions(load_predictions(proposals), spans),
    }
    slices = {
        "all_completed": all_clips,
        "timing_usable": timing_usable,
        "automatic_retained": all_clips & retained,
        "automatic_held": all_clips & held,
        "owner_good_tracking": {
            clip for clip, verdict in meta["tracking_quality"].items() if verdict == "good"
        },
    }
    results = {}
    for tolerance in (0.5, 1.0, 2.0):
        results[str(tolerance)] = {
            slice_name: {
                stream_name: score(rows, truth, clips, tolerance)
                for stream_name, rows in streams.items()
            }
            for slice_name, clips in slices.items()
        }
    return {
        "schema": "cross_match_event_audit_v5_benchmark_v1",
        "status": "held_out_frozen_stack_evaluation",
        "labels": {
            "path": str(labels),
            "sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            "rows": meta["rows"],
            "truth_events": len(truth),
            "truth_by_type": dict(Counter(row["event_type"] for row in truth)),
            "completed_clips": len(all_clips),
            "event_clips": len({row["clip"] for row in truth}),
            "no_point_clips": len(meta["no_point"]),
            "attempt_spans": sum(len(rows) for rows in spans.values()),
            "tracking_quality": dict(Counter(meta["tracking_quality"].values())),
            "cadence_held_clips": sorted(cadence_held),
        },
        "inputs": {
            "frozen_emissions_sha256": hashlib.sha256(emissions.read_bytes()).hexdigest(),
            "frozen_review_proposals_sha256": hashlib.sha256(proposals.read_bytes()).hexdigest(),
        },
        "slice_points": {name: sorted(clips) for name, clips in slices.items()},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--emissions", type=Path, default=DEFAULT_EMISSIONS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--cadence", type=Path, default=DEFAULT_CADENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = benchmark(args.labels, args.emissions, args.proposals, args.gate, args.cadence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
