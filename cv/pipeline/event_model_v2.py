"""Run the frozen all-development temporal GBM event model without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from cv.pipeline.event_model_v2_features import (
    AutomaticDataset,
    LEGACY_FEATURE_NAMES,
    build_automatic_dataset,
    save_dataset,
)
from cv.pipeline.point_grammar import annotate_from_root

CLASSES = ("none", "contact", "bounce", "net_hit")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
MODEL_SCHEMA = "event_model_v2_court_fix_forward_v1"
MODEL_SHA256 = "d8f8bd60ccc92b17f8c17c1ed2129e8eae5d0fe3e2cc606a931c5bc1e81150bd"
DEPRECATED_WARPED_MODEL_SHA256 = "66b133757dde660909e3ccb0ba2cfb9bcab43dc2168c54b2cfb05ddcb57212a7"
NMS_RADIUS = 2
PER_TYPE_THRESHOLDS = {
    "contact": 0.8254022002220154,
    "bounce": 0.9554616808891296,
    "net_hit": 0.899630606174469,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_model(path: Path) -> dict:
    observed = sha256(path)
    if observed != MODEL_SHA256:
        raise ValueError(f"frozen V2 event model hash mismatch: {observed}")
    bundle = joblib.load(path)
    metadata = bundle.get("metadata", {})
    if bundle.get("architecture") != "hist_gradient_boosting":
        raise ValueError("frozen V2 event model is not gradient boosting")
    if metadata.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"unsupported frozen V2 event model: {metadata.get('schema')!r}")
    if tuple(metadata.get("feature_names", ())) != LEGACY_FEATURE_NAMES:
        raise ValueError("frozen V2 event model feature schema does not match runtime")
    if bundle["model"].n_features_in_ != 25 * len(LEGACY_FEATURE_NAMES):
        raise ValueError("frozen V2 event model input width does not match runtime")
    return bundle


def predict_probabilities(model, windows: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(windows.reshape(len(windows), -1))
    output = np.zeros((len(windows), len(CLASSES)), dtype=np.float32)
    for column, label in enumerate(model.classes_):
        output[:, CLASS_TO_INDEX[str(label)]] = raw[:, column]
    return output


def decode(probabilities: np.ndarray, dataset: AutomaticDataset) -> list[dict]:
    event_columns = probabilities[:, 1:]
    event_class = np.argmax(event_columns, axis=1) + 1
    event_score = event_columns[np.arange(len(probabilities)), event_class - 1]
    candidates = [
        index
        for index in np.argsort(-event_score)
        if event_score[index] > probabilities[index, 0]
        and event_score[index] >= PER_TYPE_THRESHOLDS[CLASSES[int(event_class[index])]]
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
    return sorted(
        [
            {
                "clip": str(dataset.clips[index]),
                "match_id": str(dataset.broadcasts[index]),
                "event_type": CLASSES[int(event_class[index])],
                "frame": float(dataset.frames[index]),
                "confidence": float(event_score[index]),
                "probability": float(event_score[index]),
            }
            for index in kept
        ],
        key=lambda row: (row["clip"], row["frame"], row["event_type"]),
    )


def attach_gate_metadata(rows: list[dict], point_gate_path: Path) -> None:
    report = json.loads(point_gate_path.read_text())
    gates = {f"{row['match_id']}__{row['clip']}": row for row in report["rows"]}
    for row in rows:
        gate = gates[row["clip"]]
        row["point_gate_verdict"] = gate["decision"]
        row["point_gate_failure_reasons"] = list(gate.get("reasons", []))
        row["gate_held"] = gate["decision"] == "hold"
        row["production_scope"] = True


def run(root: Path, model_path: Path, point_gate: Path, output: Path, features: Path) -> dict:
    dataset, feature_manifest = build_automatic_dataset(
        root, include_camera_elevation=True, court_missing="zero"
    )
    save_dataset(features, dataset, feature_manifest)
    bundle = load_frozen_model(model_path)
    emissions = decode(predict_probabilities(bundle["model"], dataset.windows), dataset)
    attach_gate_metadata(emissions, point_gate)
    emissions, grammar_manifest = annotate_from_root(root, emissions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(emissions, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "automatic_event_model_v2_run_v2",
        "model": {"path": str(model_path.resolve()), "sha256": MODEL_SHA256},
        "model_training": {
            "selected_arm": bundle["metadata"]["selected_arm"],
            "parameters": bundle["metadata"]["parameters"],
            "training_broadcasts": bundle["metadata"]["training_broadcasts"],
            "human_derived_inputs": bundle["metadata"].get("reviewed_diagnostic_inputs", []),
            "used_at_runtime": False,
        },
        "feature_schema": feature_manifest,
        "model_schema": MODEL_SCHEMA,
        "deprecated_warped_model": {
            "sha256": DEPRECATED_WARPED_MODEL_SHA256,
            "status": "explicit_rollback_only_until_one_clean_rebenchmark_cycle",
        },
        "per_type_thresholds": PER_TYPE_THRESHOLDS,
        "nms_radius_frames": NMS_RADIUS,
        "point_grammar": grammar_manifest,
        "emission_mode": "lossless_all_emissions_with_default_in_play_consumer_view",
        "emissions": len(emissions),
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
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.root, args.model, args.point_gate, args.output, args.features), indent=2
        )
    )


if __name__ == "__main__":
    main()
