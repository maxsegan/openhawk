"""Frozen typed-event model training primitives and label-free inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cv.pipeline.event_decoder import (
    DEFAULT_CONFIG,
    EventDecoderConfig,
    complete_confident_contact_witnesses,
    complete_grammar_contacts,
    complete_physical_net_witnesses,
    decode_graph,
    decode_net_hits,
    refine_event_timing,
)

SCHEMA = "frozen_event_model_v1"
EVENT_TYPES = ("contact", "bounce", "net_hit")
CATEGORICAL = ("proposal_source", "kind", "initial_type", "final_type", "excluded")
NUMERIC = (
    "initial_confidence",
    "final_confidence",
    "sb",
    "sa",
    "speed_ratio",
    "angle",
    "vxb",
    "vyb",
    "vxa",
    "vya",
    "gap_frames",
    "horiz_reverse",
    "vert_reverse",
    "big_gain",
    "collapse",
    "at_height",
    "has_reach",
    "reach_near",
    "reach_far",
    "audio",
    "sequence_audio",
    "tape_px",
    "observed_support_px",
    "pre_200ms_observations",
    "pre_200ms_displacement_px",
    "pre_200ms_path_px",
    "pre_200ms_straightness",
    "pre_400ms_observations",
    "pre_400ms_displacement_px",
    "pre_400ms_path_px",
    "pre_400ms_straightness",
    "post_200ms_observations",
    "post_200ms_displacement_px",
    "post_200ms_path_px",
    "post_200ms_straightness",
    "post_400ms_observations",
    "post_400ms_displacement_px",
    "post_400ms_path_px",
    "post_400ms_straightness",
    "img_x",
    "img_y",
    "court_x",
    "court_y",
    "bounce_witness",
    "bounce_confidence",
    "bounce_kink",
    "review",
    "in_play",
    "anchor_offset",
    "source_serve",
    "source_bounce_witness",
    "source_trajectory",
    "source_suppressed_trajectory",
    "source_track_gap",
    "previous_gap_seconds",
    "next_gap_seconds",
    "local_candidates_80ms",
    "local_candidates_160ms",
    "distance_to_net_m",
    "distance_to_baseline_m",
    "court_outside_m",
    "clip_progress",
    "source_fps",
)
CONTEXT_NUMERIC = (
    "et_p_no_event",
    "et_p_contact",
    "et_p_bounce",
    "et_p_net_hit",
    "rf_p_no_event",
    "rf_p_contact",
    "rf_p_bounce",
    "rf_p_net_hit",
    "context_event_probability",
    "context_class_margin",
    "context_entropy",
    "context_previous_contact_support",
    "context_previous_bounce_support",
    "context_previous_net_hit_support",
    "context_next_contact_support",
    "context_next_bounce_support",
    "context_next_net_hit_support",
    "context_previous_contact_gap_seconds",
    "context_previous_bounce_gap_seconds",
    "context_previous_net_hit_gap_seconds",
    "context_next_contact_gap_seconds",
    "context_next_bounce_gap_seconds",
    "context_next_net_hit_gap_seconds",
    "context_contact_bridge",
    "context_bounce_bridge",
    "context_dense_events_400ms",
    "context_dense_events_1000ms",
)
PRODUCTION_ANCHOR_THRESHOLDS = {"contact": 0.83, "bounce": 0.89, "net_hit": 2.0}
PRODUCTION_BRIDGE_THRESHOLDS = {"contact": 0.80, "bounce": 0.10, "net_hit": 2.0}
PRODUCTION_MAX_GAP_SECONDS = 3.0
PRODUCTION_PLAYER_EDGE_MAX = 0.50
PRODUCTION_NET_THRESHOLD = 0.50


def build_estimator(name: str, seed: int, numeric: tuple[str, ...]) -> Pipeline:
    if name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(name)
    features = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), list(numeric)),
        ]
    )
    return Pipeline([("features", features), ("classifier", estimator)])


def aligned_probabilities(
    estimator: Pipeline,
    rows: list[dict],
    numeric: tuple[str, ...],
) -> np.ndarray:
    classes = ("no_event", *EVENT_TYPES)
    frame = pd.DataFrame(rows)
    local = estimator.predict_proba(frame[[*CATEGORICAL, *numeric]])
    observed = list(estimator.named_steps["classifier"].classes_)
    output = np.zeros((len(rows), len(classes)), dtype=float)
    for index, label in enumerate(classes):
        if label in observed:
            output[:, index] = local[:, observed.index(label)]
    return output


def _oof(rows: list[dict], name: str, seed: int, numeric: tuple[str, ...]) -> np.ndarray:
    output = np.zeros((len(rows), 1 + len(EVENT_TYPES)), dtype=float)
    for held_out in sorted({row["match_id"] for row in rows}):
        train = [row for row in rows if row["match_id"] != held_out]
        indices = [index for index, row in enumerate(rows) if row["match_id"] == held_out]
        estimator = build_estimator(name, seed, numeric)
        frame = pd.DataFrame(train)
        estimator.fit(frame[[*CATEGORICAL, *numeric]], frame["target"])
        output[indices] = aligned_probabilities(
            estimator, [rows[index] for index in indices], numeric
        )
    return output


def add_probability_context(
    rows: list[dict], extra_trees: np.ndarray, random_forest: np.ndarray
) -> None:
    classes = ("no_event", *EVENT_TYPES)
    grouped: dict[str, list[int]] = defaultdict(list)
    mean_probabilities = (extra_trees + random_forest) / 2.0
    for index, row in enumerate(rows):
        grouped[row["clip"]].append(index)
        for class_index, event_type in enumerate(classes):
            row[f"et_p_{event_type}"] = float(extra_trees[index, class_index])
            row[f"rf_p_{event_type}"] = float(random_forest[index, class_index])
        ranked = np.sort(mean_probabilities[index, 1:])
        row["context_event_probability"] = float(ranked[-1])
        row["context_class_margin"] = float(ranked[-1] - ranked[-2])
        row["context_entropy"] = float(
            -np.sum(mean_probabilities[index] * np.log(np.maximum(mean_probabilities[index], 1e-8)))
        )
    class_index = {event_type: index + 1 for index, event_type in enumerate(EVENT_TYPES)}
    for indices in grouped.values():
        indices.sort(key=lambda index: float(rows[index]["proposal_frame"]))
        frames = np.asarray([float(rows[index]["proposal_frame"]) for index in indices])
        fps = float(rows[indices[0]]["source_fps"])
        local_probabilities = mean_probabilities[np.asarray(indices)]
        for local_index, row_index in enumerate(indices):
            frame = frames[local_index]
            supports = {}
            for direction, mask in (
                ("previous", frames < frame - 0.06 * fps),
                ("next", frames > frame + 0.06 * fps),
            ):
                gaps = np.abs(frames - frame) / fps
                mask &= gaps <= 3.0
                supports[direction] = {}
                for event_type in EVENT_TYPES:
                    eligible = np.flatnonzero(mask)
                    if len(eligible):
                        values = local_probabilities[eligible, class_index[event_type]] * np.exp(
                            -np.maximum(gaps[eligible] - 1.2, 0.0) / 1.5
                        )
                        best = eligible[int(np.argmax(values))]
                        support, gap = float(values.max()), float(gaps[best])
                    else:
                        support, gap = 0.0, 3.1
                    supports[direction][event_type] = support
                    rows[row_index][f"context_{direction}_{event_type}_support"] = support
                    rows[row_index][f"context_{direction}_{event_type}_gap_seconds"] = gap
            previous, following = supports["previous"], supports["next"]
            rows[row_index]["context_bounce_bridge"] = min(
                previous["contact"], following["contact"]
            )
            rows[row_index]["context_contact_bridge"] = max(
                min(previous["bounce"], following["contact"]),
                min(previous["contact"], following["bounce"]),
                min(previous["contact"], following["contact"]),
            )
            event_probability = local_probabilities[:, 1:].max(axis=1)
            rows[row_index]["context_dense_events_400ms"] = int(
                np.sum((np.abs(frames - frame) <= 0.4 * fps) & (event_probability >= 0.2))
            )
            rows[row_index]["context_dense_events_1000ms"] = int(
                np.sum((np.abs(frames - frame) <= fps) & (event_probability >= 0.2))
            )


def fit_bundle(rows: list[dict], seed: int, training_cohorts: list[str]) -> dict:
    training = [dict(row) for row in rows]
    first_oof = {
        name: _oof(training, name, seed, NUMERIC) for name in ("extra_trees", "random_forest")
    }
    first_models = {}
    frame = pd.DataFrame(training)
    for name in first_oof:
        estimator = build_estimator(name, seed, NUMERIC)
        estimator.fit(frame[[*CATEGORICAL, *NUMERIC]], frame["target"])
        first_models[name] = estimator
    add_probability_context(training, first_oof["extra_trees"], first_oof["random_forest"])
    stacked_numeric = (*NUMERIC, *CONTEXT_NUMERIC)
    stacked = build_estimator("random_forest", seed, stacked_numeric)
    stacked_frame = pd.DataFrame(training)
    stacked.fit(stacked_frame[[*CATEGORICAL, *stacked_numeric]], stacked_frame["target"])
    binaries = {}
    for event_index, event_type in enumerate(EVENT_TYPES, start=1):
        estimator = build_estimator("random_forest", seed + event_index, stacked_numeric)
        estimator.fit(
            stacked_frame[[*CATEGORICAL, *stacked_numeric]],
            np.asarray([row["target"] == event_type for row in training], dtype=int),
        )
        binaries[event_type] = estimator
    return {
        "schema": SCHEMA,
        "seed": seed,
        "training_cohorts": list(training_cohorts),
        "training_broadcasts": sorted({row["match_id"] for row in training}),
        "training_candidates": len(training),
        "feature_schema": {
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
            "context": CONTEXT_NUMERIC,
        },
        "first_models": first_models,
        "stacked_model": stacked,
        "binary_models": binaries,
    }


def _graph_predictions(
    rows: list[dict],
    probabilities: np.ndarray,
    decoder_config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    return decode_graph(rows, probabilities, config=decoder_config, scoped_only=True)


def _binary_probabilities(bundle: dict, rows: list[dict], numeric: tuple[str, ...]) -> np.ndarray:
    output = np.zeros((len(rows), 1 + len(EVENT_TYPES)), dtype=float)
    frame = pd.DataFrame(rows)
    for index, event_type in enumerate(EVENT_TYPES, start=1):
        estimator = bundle["binary_models"][event_type]
        local = estimator.predict_proba(frame[[*CATEGORICAL, *numeric]])
        classes = list(estimator.named_steps["classifier"].classes_)
        if 1 in classes:
            output[:, index] = local[:, classes.index(1)]
    output[:, 0] = np.maximum(0.0, 1.0 - output[:, 1:].max(axis=1))
    return output


def predict(
    bundle: dict,
    rows: list[dict],
    *,
    decoder_config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    if bundle.get("schema") != SCHEMA:
        raise ValueError(f"unsupported event model: {bundle.get('schema')!r}")
    inference = [dict(row) for row in rows]
    first = {
        name: aligned_probabilities(estimator, inference, NUMERIC)
        for name, estimator in bundle["first_models"].items()
    }
    add_probability_context(inference, first["extra_trees"], first["random_forest"])
    stacked_numeric = (*NUMERIC, *CONTEXT_NUMERIC)
    probabilities = aligned_probabilities(bundle["stacked_model"], inference, stacked_numeric)
    base = refine_event_timing(
        inference,
        probabilities,
        _graph_predictions(inference, probabilities, decoder_config),
        config=decoder_config,
    )
    base = complete_confident_contact_witnesses(
        inference,
        probabilities,
        base,
        config=decoder_config,
    )
    base = complete_grammar_contacts(
        inference,
        probabilities,
        base,
        config=decoder_config,
    )
    binary = _binary_probabilities(bundle, inference, stacked_numeric)
    nets = decode_net_hits(inference, binary, config=decoder_config)
    nets = complete_physical_net_witnesses(inference, nets, config=decoder_config)
    output = list(base)
    for net in sorted(nets, key=lambda row: -row["probability"]):
        radius = max(1.0, round(0.08 * net["fps"]))
        output = [
            event
            for event in output
            if event["clip"] != net["clip"] or abs(float(event["frame"]) - net["frame"]) > radius
        ]
        output.append(net)
    return sorted(output, key=lambda row: (row["clip"], row["frame"], row["event_type"]))


def save_bundle(path: Path, bundle: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("schema") != SCHEMA:
        raise ValueError(f"unsupported event model: {bundle.get('schema')!r}")
    return bundle
