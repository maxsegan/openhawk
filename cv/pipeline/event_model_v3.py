"""Run the frozen iteration-2 B1 event model without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from cv.pipeline.event_model_v2 import attach_gate_metadata, predict_probabilities
from cv.pipeline.event_model_v2_features import (
    AutomaticDataset,
    LEGACY_FEATURE_NAMES,
    build_automatic_dataset,
    save_dataset,
)
from cv.pipeline.point_grammar import annotate_from_root

CLASSES = ("none", "contact", "bounce", "net_hit")
MODEL_SCHEMA = "event_model_v3_iteration2_mk_v1"
MODEL_SHA256 = "304c3a8e50b9341c951d01c4c36248b9874eaa3191982e4f1f79923e8ce10031"
ORIGINAL_LOST_MODEL_SHA256 = "1b2c00db5120bcf66c349b87ebe3f046aefe996d38cfbbf2fc19fdbe398cc2ec"
ROLLBACK_MODEL_SHA256 = "d8f8bd60ccc92b17f8c17c1ed2129e8eae5d0fe3e2cc606a931c5bc1e81150bd"
NMS_RADIUS = 4
PER_TYPE_THRESHOLDS = {
    "contact": 0.6951295137405396,
    "bounce": 0.9093982577323914,
    "net_hit": 0.832057774066925,
}
HYPOTHESIS_THRESHOLDS = {
    "contact": 0.05,
    "bounce": 0.05,
    "net_hit": 0.02,
}
# Candidates that clear the floor but not the operating threshold are emitted as
# explicit abstentions instead of being dropped.
ABSTAIN_FLOORS = dict(HYPOTHESIS_THRESHOLDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_model(path: Path) -> dict:
    observed = sha256(path)
    if observed != MODEL_SHA256:
        raise ValueError(f"frozen V3 event model hash mismatch: {observed}")
    bundle = joblib.load(path)
    metadata = bundle.get("metadata", {})
    if bundle.get("architecture") != "hist_gradient_boosting":
        raise ValueError("frozen V3 event model is not histogram gradient boosting")
    if metadata.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"unsupported frozen V3 event model: {metadata.get('schema')!r}")
    if tuple(metadata.get("feature_names", ())) != LEGACY_FEATURE_NAMES:
        raise ValueError("frozen V3 event model feature schema does not match runtime")
    if bundle["model"].n_features_in_ != 25 * len(LEGACY_FEATURE_NAMES):
        raise ValueError("frozen V3 event model input width does not match runtime")
    if metadata.get("owner_review_labels") or metadata.get("reviewed_diagnostic_inputs"):
        raise ValueError("frozen V3 event model contains reviewed training inputs")
    return bundle


def decode(
    probabilities: np.ndarray,
    dataset: AutomaticDataset,
    *,
    thresholds: dict[str, float] | None = None,
    abstain_floors: dict[str, float] | None = None,
) -> list[dict]:
    """Decode LOBO or runtime probabilities into emission rows.

    Candidates at or above the per-type operating threshold are accepted.  With
    ``abstain_floors`` set, candidates between the floor and the threshold are
    retained as explicit ``abstain=True`` rows rather than dropped; consumers
    see them only through the lossless bypass.
    """

    thresholds = PER_TYPE_THRESHOLDS if thresholds is None else thresholds
    event_columns = probabilities[:, 1:]
    event_class = np.argmax(event_columns, axis=1) + 1
    event_score = event_columns[np.arange(len(probabilities)), event_class - 1]
    candidates = [
        index for index in np.argsort(-event_score) if event_score[index] > probabilities[index, 0]
    ]
    kept: list[int] = []
    for index in candidates:
        clip = str(dataset.clips[index])
        event_type = CLASSES[int(event_class[index])]
        frame = int(dataset.frames[index])
        if any(
            str(dataset.clips[prior]) == clip
            and CLASSES[int(event_class[prior])] == event_type
            and abs(int(dataset.frames[prior]) - frame) <= NMS_RADIUS
            for prior in kept
        ):
            continue
        kept.append(index)
    missing_geometry = dataset.missing_court_geometry()
    missing_coordinate = dataset.missing_court_coordinate()
    rows = []
    for index in kept:
        event_type = CLASSES[int(event_class[index])]
        score = float(event_score[index])
        threshold = float(thresholds[event_type])
        floor = float((abstain_floors or {}).get(event_type, threshold))
        if score < threshold and score < floor:
            continue
        rows.append(
            {
                "clip": str(dataset.clips[index]),
                "match_id": str(dataset.broadcasts[index]),
                "event_type": event_type,
                "frame": float(dataset.frames[index]),
                "confidence": score,
                "probability": score,
                "class_probabilities": {
                    label: float(probabilities[index, position])
                    for position, label in enumerate(CLASSES)
                },
                "abstain": score < threshold,
                "decision_threshold": threshold,
                "abstain_floor": floor if abstain_floors else None,
                "court_geometry_missing": bool(missing_geometry[index]),
                "court_coordinate_missing": bool(missing_coordinate[index]),
            }
        )
    return sorted(rows, key=lambda row: (row["clip"], row["frame"], row["event_type"]))


def decode_hypotheses(probabilities: np.ndarray, dataset: AutomaticDataset) -> list[dict]:
    """Emit a leaky per-type lattice for downstream joint reconstruction."""
    rows = []
    for clip in sorted(set(str(value) for value in dataset.clips)):
        clip_indices = np.flatnonzero(dataset.clips.astype(str) == clip)
        for class_index, event_type in enumerate(CLASSES[1:], start=1):
            threshold = HYPOTHESIS_THRESHOLDS[event_type]
            candidates = sorted(
                (
                    int(index)
                    for index in clip_indices
                    if float(probabilities[index, class_index]) >= threshold
                ),
                key=lambda index: float(probabilities[index, class_index]),
                reverse=True,
            )
            kept = []
            for index in candidates:
                frame = int(dataset.frames[index])
                if any(abs(int(dataset.frames[prior]) - frame) <= NMS_RADIUS for prior in kept):
                    continue
                kept.append(index)
            for index in kept:
                rows.append(
                    {
                        "clip": clip,
                        "match_id": str(dataset.broadcasts[index]),
                        "event_type": event_type,
                        "frame": float(dataset.frames[index]),
                        "probability": float(probabilities[index, class_index]),
                        "class_probabilities": {
                            label: float(probabilities[index, probability_index])
                            for probability_index, label in enumerate(CLASSES)
                        },
                        "origin": "frozen_event_model_v3_leaky_hypothesis",
                    }
                )
    return sorted(rows, key=lambda row: (row["clip"], row["frame"], row["event_type"]))


def run(
    root: Path,
    model_path: Path,
    point_gate: Path,
    output: Path,
    features: Path,
    hypotheses_output: Path | None = None,
) -> dict:
    dataset, feature_manifest = build_automatic_dataset(
        root, include_camera_elevation=True, court_missing="zero"
    )
    save_dataset(features, dataset, feature_manifest)
    bundle = load_frozen_model(model_path)
    probabilities = predict_probabilities(bundle["model"], dataset.windows)
    emissions = decode(probabilities, dataset, abstain_floors=ABSTAIN_FLOORS)
    hypotheses = decode_hypotheses(probabilities, dataset)
    attach_gate_metadata(emissions, point_gate)
    attach_gate_metadata(hypotheses, point_gate)
    emissions, grammar_manifest = annotate_from_root(root, emissions)
    hypotheses, _ = annotate_from_root(root, hypotheses, emit_point_end=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(emissions, indent=2, sort_keys=True) + "\n")
    hypotheses_output = hypotheses_output or output.with_name("event_hypotheses.json")
    hypotheses_output.write_text(json.dumps(hypotheses, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "automatic_event_model_v3_run_v1",
        "model": {"path": str(model_path.resolve()), "sha256": MODEL_SHA256},
        "model_recovery": {
            "original_lost_sha256": ORIGINAL_LOST_MODEL_SHA256,
            "replacement_sha256": MODEL_SHA256,
            "status": "deterministic_semantic_reproduction",
        },
        "model_training": {
            "selected_arm": bundle["metadata"]["selected_arm"],
            "geometry": bundle["metadata"]["geometry"],
            "parameters": bundle["metadata"]["parameters"],
            "training_broadcasts": bundle["metadata"]["training_broadcasts"],
            "human_derived_inputs": {
                "owner_truth_v2_sha256": bundle["metadata"]["truth_v2_sha256"],
                "blind_agent_label_sha256": bundle["metadata"]["agent_file_sha256"],
                "owner_review_labels": [],
            },
            "used_at_runtime": False,
        },
        "feature_schema": feature_manifest,
        "model_schema": MODEL_SCHEMA,
        "rollback_model": {"sha256": ROLLBACK_MODEL_SHA256, "status": "rollback_only"},
        "per_type_thresholds": PER_TYPE_THRESHOLDS,
        "abstention_floors": ABSTAIN_FLOORS,
        "nms_radius_frames": NMS_RADIUS,
        "point_grammar": grammar_manifest,
        "emission_mode": "lossless_all_emissions_with_default_in_play_consumer_view",
        "emissions": len(emissions),
        "abstained_emissions": sum(row.get("abstain") is True for row in emissions),
        "point_end_emissions": sum(row["event_type"] == "point_end" for row in emissions),
        "hypotheses": {
            "path": str(hypotheses_output),
            "rows": len(hypotheses),
            "thresholds": HYPOTHESIS_THRESHOLDS,
            "role": "leaky_downstream_evidence_not_selected_s5_events",
        },
        "gate_held_emissions": sum(row.get("gate_held") is True for row in emissions),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--point-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.root,
                args.model,
                args.point_gate,
                args.output,
                args.features,
                args.hypotheses,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
