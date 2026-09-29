"""Graded, empirical-noise benchmark for the Stage-6 3D flight fitter.

The benchmark is evaluation-only.  It measures noise from frozen human-labelled and
automatic artifacts, generates trajectories with :mod:`physics.reference`, and calls
the unchanged production anchor-first fitter.  Human labels never enter automatic
inference.

Full benchmark (CPU, about 1--3 hours depending on the host)::

    TENNIS_DATA_ROOT=data \
      .venv/bin/python -m cv.validation.s6_bench run --workers 8

Quick structural smoke test::

    TENNIS_DATA_ROOT=data \
      .venv/bin/python -m cv.validation.s6_bench run --workers 8 --flights 8 \
      --skip-levers --allow-threshold-failure

The default run writes ``noise_models.json``, ``ladder.json``, ``levers.json``, and
CSV summaries below ``$TENNIS_DATA_ROOT/processed/wk3_s6bench``.  It exits non-zero
when the proposed readiness thresholds fail unless ``--allow-threshold-failure`` is
given.  Large generated artifacts remain outside the repository.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

from cv.pipeline import anchor_first_fit
from cv.pipeline.court_topology_frame_track import interpolate_track_H
from cv.validation import s6root_closed_loop
from cv.validation.physics_validation import reference_fitter_adapter
from cv.validation.s6root_common import (
    NET_Y_M,
    ORACLE_RELATIVE,
    FlightCase,
    compact_and_classify,
    fit_case,
    load_flight_cases,
    percentile,
    write_csv,
    write_json,
)

SEED = 20260902
DEFAULT_FLIGHTS = 200
DEFAULT_WORKERS = 8
TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"
TRACK_WINDOW_RELATIVE = Path("processed/wk3_s4ft/eval/ep3_B0/ball")
COHORT_RELATIVE = Path("processed/wk3_cohort/cohort_root_v2")
TIMING_RELATIVE = Path("processed/wk3_audiotime/step1_model_timing/model_timing.json")
ANCHOR_REPORT_RELATIVE = Path("processed/wk3_s6v2/default/report.json")
OUTPUT_RELATIVE = Path("processed/wk3_s6bench")
LABEL_RELATIVE = Path("cv/validation/labels/ball_track_sequence_v1")
EVENT_RELATIVE = Path("processed/wk3_oracle/oracle_3d_ceiling_v2/truth_events_event_model_v3.json")
WRONG_OBJECT_REASONS = {
    "excessive_arc_innovation",
    "excessive_arc_innovation_covariance",
    "insufficient_arc_candidate_support",
}
RUNG_NAMES = (
    "clean",
    "track_error",
    "wrong_object",
    "dropped_frames",
    "contact_adjacent",
    "camera_registration",
    "event_timing",
    "missing_anchors",
    "realistic",
)
LEVER_NAMES = (
    "single_start_ablation",
    "robust_soft_l1_f0p5",
    "robust_soft_l1_f2",
    "robust_soft_l1_f8",
    "innovation_covariance_weights",
    "drop_contact_adjacent",
    "warm_restart_multistart",
    "reference_physics_fit",
)


@dataclass(frozen=True)
class BenchInputs:
    """Explicit input roots; no ambient file silently changes benchmark behavior."""

    data_root: Path
    repository_root: Path
    track_window_root: Path
    cohort_root: Path
    timing_path: Path
    anchor_report_path: Path
    event_path: Path
    oracle_root: Path

    @classmethod
    def defaults(cls, data_root: Path, repository_root: Path) -> BenchInputs:
        return cls(
            data_root=data_root,
            repository_root=repository_root,
            track_window_root=data_root / TRACK_WINDOW_RELATIVE,
            cohort_root=data_root / COHORT_RELATIVE,
            timing_path=data_root / TIMING_RELATIVE,
            anchor_report_path=data_root / ANCHOR_REPORT_RELATIVE,
            event_path=data_root / EVENT_RELATIVE,
            oracle_root=data_root / ORACLE_RELATIVE,
        )


class _ZeroNoise:
    def normal(self, _mean: float, _sigma: float, size: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(size, dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path, *, role: str, rows: int | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "role": role,
    }
    if rows is not None:
        output["rows"] = int(rows)
    return output


def _code_provenance(inputs: BenchInputs) -> list[dict[str, Any]]:
    roles = {
        Path(__file__).resolve(): "benchmark and empirical samplers",
        inputs.repository_root / "cv/validation/s6root_closed_loop.py": "synthetic generator",
        inputs.repository_root / "cv/validation/physics_validation.py": "reference adapter",
        inputs.repository_root / "cv/pipeline/anchor_first_fit.py": "production fitter",
        inputs.repository_root / "cv/pipeline/rich_ball_physics.py": "fitted shot state",
        inputs.repository_root / "physics/reference.py": "generation/reference physics",
        inputs.repository_root / "physics/flight.py": "current flight physics",
        inputs.repository_root / "physics/impact.py": "current impact physics",
    }
    return [_source(path, role=role) for path, role in roles.items()]


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "median": percentile(finite, 50.0),
        "p90": percentile(finite, 90.0),
        "p95": percentile(finite, 95.0),
        "p99": percentile(finite, 99.0),
        "maximum": float(np.max(finite)) if len(finite) else None,
    }


def _frame_number(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(Path(value).stem.removeprefix("f_"))


def _read_track(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            native = row.get("x_native") not in (None, "")
            scale = 1.0 if native else 2.0
            rows[str(row["clip"])][_frame_number(row["frame"])] = {
                "x": float(row["x_native"] if native else row["x"]) * scale,
                "y": float(row["y_native"] if native else row["y"]) * scale,
                "cov_xx": float(row.get("innovation_cov_xx_native") or "nan"),
                "cov_xy": float(row.get("innovation_cov_xy_native") or "nan"),
                "cov_yy": float(row.get("innovation_cov_yy_native") or "nan"),
                "mahalanobis": float(row.get("innovation_mahalanobis") or "nan"),
            }
    return rows


def _court_side(projection: np.ndarray, pixel: np.ndarray) -> str:
    homography = np.linalg.inv(np.asarray(projection, dtype=float)[:, [0, 1, 3]])
    homogeneous = homography @ np.asarray([pixel[0], pixel[1], 1.0], dtype=float)
    if abs(float(homogeneous[2])) < 1e-9:
        return "unknown"
    court_y = float(homogeneous[1] / homogeneous[2])
    return "near" if court_y < NET_Y_M else "far"


def _pearson_lag_one(sequences: Sequence[Sequence[float]]) -> float | None:
    left: list[float] = []
    right: list[float] = []
    for sequence in sequences:
        for first, second in zip(sequence, sequence[1:]):
            if np.isfinite(first) and np.isfinite(second):
                left.append(float(first))
                right.append(float(second))
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _contact_index(path: Path) -> dict[tuple[str, str], list[float]]:
    payload = json.loads(path.read_text())
    output: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in payload["emissions"]:
        if row.get("event_type") != "contact":
            continue
        match_id = str(row["match_id"])
        clip_value = str(row["clip"])
        clip = clip_value.removeprefix(f"{match_id}__")
        output[(match_id, clip)].append(float(row["frame"]))
    return output


def measure_track_error(inputs: BenchInputs) -> dict[str, Any]:
    """Measure signed native-pixel residual blocks on all 98 owner windows."""
    label_root = inputs.repository_root / LABEL_RELATIVE
    contacts = _contact_index(inputs.event_path)
    samples: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    tracks: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    cameras: dict[tuple[str, str], tuple[dict[int, np.ndarray], np.ndarray]] = {}
    for split in ("development", "sealed_transfer"):
        benchmark_path = label_root / f"{split}_benchmark_v1.json"
        truth_path = label_root / f"{split}_truth_v1.json"
        benchmark = json.loads(benchmark_path.read_text())
        truth = {row["case_id"]: row for row in json.loads(truth_path.read_text())["records"]}
        sources.extend(
            [
                _source(benchmark_path, role=f"{split} 49-window case manifest"),
                _source(truth_path, role=f"{split} owner leading-edge positions"),
            ]
        )
        for case in benchmark["cases"]:
            match_id = str(case["match_id"])
            clip = str(case["clip"])
            if match_id not in tracks:
                track_path = inputs.track_window_root / match_id / TRACK_NAME
                tracks[match_id] = _read_track(track_path)
                sources.append(_source(track_path, role="shipped composed motion track"))
            camera_key = (match_id, clip)
            if camera_key not in cameras:
                camera_path = inputs.track_window_root / match_id / "camera_P_per_frame_v1.npz"
                with np.load(camera_path, allow_pickle=True) as data:
                    selected = data["clips"].astype(str) == clip
                    rows = {
                        int(frame): np.asarray(projection, dtype=float)
                        for frame, projection in zip(
                            data["frames"][selected], data["P"][selected], strict=True
                        )
                    }
                fallback = next(iter(rows.values()))
                cameras[camera_key] = (rows, fallback)
                if not any(row.get("path") == str(camera_path.resolve()) for row in sources):
                    sources.append(_source(camera_path, role="court-side attribution camera"))
            projections, fallback_projection = cameras[camera_key]
            truth_frames = {int(row["frame"]): row for row in truth[str(case["case_id"])]["frames"]}
            sequence_rows = []
            for frame_row in case["frames"]:
                frame = int(frame_row["frame"])
                owner = truth_frames[frame]
                track = tracks[match_id].get(clip, {}).get(frame)
                if owner.get("x1080") is None or owner.get("y1080") is None:
                    continue
                if track is None:
                    sequence_rows.append({"frame": frame, "missing": True})
                    continue
                owner_xy = np.asarray([owner["x1080"], owner["y1080"]], dtype=float)
                residual = np.asarray([track["x"], track["y"]], dtype=float) - owner_xy
                side = _court_side(projections.get(frame, fallback_projection), owner_xy)
                adjacent = any(
                    abs(frame - contact) <= 2.0 for contact in contacts.get(camera_key, [])
                )
                trace = float(track["cov_xx"] + track["cov_yy"])
                record = {
                    "case_id": str(case["case_id"]),
                    "match_id": match_id,
                    "clip": clip,
                    "frame": frame,
                    "side": side,
                    "contact_adjacent": adjacent,
                    "dx_px": float(residual[0]),
                    "dy_px": float(residual[1]),
                    "error_px": float(np.linalg.norm(residual)),
                    "innovation_cov_trace_native": trace,
                    "innovation_mahalanobis": float(track["mahalanobis"]),
                }
                samples.append(record)
                sequence_rows.append(record)
            sequences.append({"case_id": str(case["case_id"]), "samples": sequence_rows})
    errors = [row["error_px"] for row in samples]
    dx_sequences = [
        [row["dx_px"] for row in sequence["samples"] if not row.get("missing")]
        for sequence in sequences
    ]
    dy_sequences = [
        [row["dy_px"] for row in sequence["samples"] if not row.get("missing")]
        for sequence in sequences
    ]
    magnitude_sequences = [
        [row["error_px"] for row in sequence["samples"] if not row.get("missing")]
        for sequence in sequences
    ]
    by_side = {
        side: _stats(row["error_px"] for row in samples if row["side"] == side)
        for side in ("near", "far", "unknown")
    }
    by_contact = {
        name: _stats(
            row["error_px"] for row in samples if bool(row["contact_adjacent"]) == is_adjacent
        )
        for name, is_adjacent in (("within_2_frames", True), ("interior", False))
    }
    covariances = [
        row["innovation_cov_trace_native"]
        for row in samples
        if np.isfinite(row["innovation_cov_trace_native"])
    ]
    return {
        "provenance": {
            "sources": sources,
            "method": (
                "authoritative owner x1080/y1080 leading-edge positions minus the shipped "
                "ep3_B0 composed native track; side is the owner pixel back-projected to z=0 "
                "through that frame's real camera; contact-adjacent means <=2 owner frames"
            ),
        },
        "windows": len(sequences),
        "owner_positioned_frames": len(samples)
        + sum(row.get("missing", False) for sequence in sequences for row in sequence["samples"]),
        "matched_track_frames": len(samples),
        "missing_track_frames": sum(
            row.get("missing", False) for sequence in sequences for row in sequence["samples"]
        ),
        "error_px": _stats(errors),
        "by_side": by_side,
        "contact_adjacent": by_contact,
        "lag1_autocorrelation": {
            "dx": _pearson_lag_one(dx_sequences),
            "dy": _pearson_lag_one(dy_sequences),
            "magnitude": _pearson_lag_one(magnitude_sequences),
        },
        "innovation_cov_trace_native": _stats(covariances),
        "sequences": sequences,
    }


def _contiguous_runs(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    output = [[ordered[0]]]
    for value in ordered[1:]:
        if value == output[-1][-1] + 1:
            output[-1].append(value)
        else:
            output.append([value])
    return output


def measure_arcs_and_dropouts(inputs: BenchInputs) -> dict[str, Any]:
    """Measure held wrong-object-risk arcs and actual missing-frame runs."""
    gate_path = inputs.cohort_root / "untouched_tracking_point_gate_v1.json"
    gate = json.loads(gate_path.read_text())
    track_cache: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    wrong_arcs: list[dict[str, Any]] = []
    dropout_runs: list[int] = []
    wrong_scope_frames: set[tuple[str, str, int]] = set()
    total_scope_frames = 0
    track_sources: set[Path] = set()
    for arc in gate["arcs"]:
        match_id = str(arc["match_id"])
        clip = str(arc["clip"])
        start = int(arc["start_frame"])
        end = int(arc["end_frame"])
        total_scope_frames += end - start + 1
        if match_id not in track_cache:
            path = inputs.cohort_root / match_id / TRACK_NAME
            track_cache[match_id] = _read_track(path)
            track_sources.add(path)
        points = track_cache[match_id].get(clip, {})
        missing = [frame for frame in range(start, end + 1) if frame not in points]
        dropout_runs.extend(len(run) for run in _contiguous_runs(missing))
        reasons = set(arc.get("failure_reasons", []))
        wrong_risk = arc.get("decision") == "hold" and bool(reasons & WRONG_OBJECT_REASONS)
        if not wrong_risk:
            continue
        for frame in range(start, end + 1):
            wrong_scope_frames.add((match_id, clip, frame))
        before = max((frame for frame in points if frame < start), default=None)
        after = min((frame for frame in points if frame > end), default=None)
        offsets = []
        if before is not None and after is not None and after > before:
            left = np.asarray([points[before]["x"], points[before]["y"]], dtype=float)
            right = np.asarray([points[after]["x"], points[after]["y"]], dtype=float)
            for frame in range(start, end + 1):
                if frame not in points:
                    continue
                fraction = (frame - before) / (after - before)
                bridge = left + fraction * (right - left)
                actual = np.asarray([points[frame]["x"], points[frame]["y"]], dtype=float)
                delta = actual - bridge
                offsets.append([float(delta[0]), float(delta[1])])
        wrong_arcs.append(
            {
                "duration_frames": end - start + 1,
                "observed_frames": int(arc.get("observed_frames", 0)),
                "failure_reasons": sorted(reasons),
                "detour_offsets_px": offsets,
            }
        )
    detour_magnitudes = [
        float(np.linalg.norm(offset)) for arc in wrong_arcs for offset in arc["detour_offsets_px"]
    ]
    return {
        "provenance": {
            "sources": [
                _source(gate_path, role="46-broadcast per-arc retain/hold decisions"),
                *[
                    _source(path, role="arc availability and detour geometry")
                    for path in sorted(track_sources)
                ],
            ],
            "wrong_object_proxy": (
                "held arc with excessive innovation, excessive innovation covariance, or "
                "insufficient candidate support; detour is the emitted arc relative to the "
                "linear bridge between nearest surrounding emitted rows"
            ),
            "dropout_method": "contiguous absent CSV frames inside every declared arc scope",
        },
        "arc_count": int(gate["arc_count"]),
        "held_arcs": int(gate["held_arcs"]),
        "total_scope_frames": total_scope_frames,
        "wrong_object_proxy_arcs": len(wrong_arcs),
        "wrong_object_proxy_frame_fraction": len(wrong_scope_frames) / max(total_scope_frames, 1),
        "wrong_object_arc_start_rate_per_frame": len(wrong_arcs) / max(total_scope_frames, 1),
        "wrong_object_duration_frames": _stats(arc["duration_frames"] for arc in wrong_arcs),
        "wrong_object_detour_px": _stats(detour_magnitudes),
        "wrong_object_arcs": wrong_arcs,
        "dropout_runs": len(dropout_runs),
        "dropout_start_rate_per_frame": len(dropout_runs) / max(total_scope_frames, 1),
        "dropout_duration_frames": _stats(dropout_runs),
        "dropout_run_lengths": dropout_runs,
    }


COURT_W_M = 10.97
COURT_L_M = 23.77
CAMERA_GRID = np.asarray(
    [[x, y] for x in (0.0, COURT_W_M / 2.0, COURT_W_M) for y in (0.0, NET_Y_M, COURT_L_M)],
    dtype=float,
)


def _image_points(homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        CAMERA_GRID[None, :, :].astype(np.float64),
        np.linalg.inv(np.asarray(homography, dtype=float)),
    )[0]


def measure_camera_registration(inputs: BenchInputs) -> dict[str, Any]:
    """Measure leave-one-registration-out and static-fallback image residuals."""
    residuals: list[list[float]] = []
    fallback_residuals: list[list[float]] = []
    fallback_runs: list[int] = []
    source_counts: dict[str, int] = defaultdict(int)
    paths = sorted(inputs.cohort_root.glob("*/court_H_per_frame_v1.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            clips = data["clips"].astype(str)
            frames = data["frames"].astype(int)
            homographies = data["H"].astype(float)
            sources = data["source"].astype(str)
        for clip in sorted(set(clips)):
            selected = np.flatnonzero(clips == clip)
            clip_frames = frames[selected]
            clip_h = homographies[selected]
            clip_sources = sources[selected]
            order = np.argsort(clip_frames)
            clip_frames = clip_frames[order]
            clip_h = clip_h[order]
            clip_sources = clip_sources[order]
            for source in clip_sources:
                source_counts[str(source)] += 1
            registered = np.flatnonzero(clip_sources == "registered")
            for position in registered[1:-1]:
                left = registered[np.searchsorted(registered, position) - 1]
                right = registered[np.searchsorted(registered, position) + 1]
                predicted = interpolate_track_H(
                    np.asarray([clip_frames[left], clip_frames[right]]),
                    np.stack([clip_h[left], clip_h[right]]),
                    int(clip_frames[position]),
                )
                delta = _image_points(clip_h[position]) - _image_points(predicted)
                residuals.extend(delta.tolist())
            fallback_positions = np.flatnonzero(clip_sources == "anchor_static_fallback")
            for run in _contiguous_runs(fallback_positions.tolist()):
                fallback_runs.append(len(run))
                left = max(
                    (idx for idx in range(run[0]) if clip_sources[idx] != "anchor_static_fallback"),
                    default=None,
                )
                right = min(
                    (
                        idx
                        for idx in range(run[-1] + 1, len(clip_frames))
                        if clip_sources[idx] != "anchor_static_fallback"
                    ),
                    default=None,
                )
                if left is None or right is None:
                    continue
                for position in run:
                    predicted = interpolate_track_H(
                        np.asarray([clip_frames[left], clip_frames[right]]),
                        np.stack([clip_h[left], clip_h[right]]),
                        int(clip_frames[position]),
                    )
                    delta = _image_points(clip_h[position]) - _image_points(predicted)
                    fallback_residuals.extend(delta.tolist())
    return {
        "provenance": {
            "sources": [
                _source(path, role="automatic per-frame registered court homographies")
                for path in paths
            ],
            "method": (
                "nine-court-point native-pixel displacement: sampled registrations versus "
                "leave-one-sample-out temporal interpolation; static fallback versus the "
                "interpolation between its nearest non-fallback frames"
            ),
        },
        "source_frame_counts": dict(sorted(source_counts.items())),
        "registration_residual_px": _stats(np.linalg.norm(row) for row in residuals),
        "fallback_residual_px": _stats(np.linalg.norm(row) for row in fallback_residuals),
        "fallback_duration_frames": _stats(fallback_runs),
        "fallback_run_lengths": fallback_runs,
        "registration_residual_vectors_px": residuals,
        "fallback_residual_vectors_px": fallback_residuals,
    }


def measure_event_timing(inputs: BenchInputs) -> dict[str, Any]:
    payload = json.loads(inputs.timing_path.read_text())
    errors = [float(row["error_frames"]) for row in payload["matches"]]
    return {
        "provenance": {
            "sources": [
                _source(inputs.timing_path, role="automatic contact minus owner half-frame truth")
            ],
            "method": "bootstrap the 518 signed matched-contact offsets; no Gaussian approximation",
        },
        "signed_error_frames": errors,
        "signed": _stats(errors),
        "absolute": _stats(abs(value) for value in errors),
    }


def measure_anchor_availability(inputs: BenchInputs) -> dict[str, Any]:
    payload = json.loads(inputs.anchor_report_path.read_text())
    patterns: list[list[str]] = []
    for point in payload["points_detail"]:
        for attempt in point.get("flight_attempts", []):
            types = [
                str(value) for value in attempt.get("anchor_first", {}).get("anchor_types", [])
            ]
            patterns.append(types)
    return {
        "provenance": {
            "sources": [_source(inputs.anchor_report_path, role="551 automatic flight attempts")],
            "method": "exact automatic anchor_types multiset on each attempted cohort-v2 flight",
        },
        "flights": len(patterns),
        "without_bounce": sum("bounce" not in row for row in patterns) / max(len(patterns), 1),
        "without_net": sum("net_crossing" not in row for row in patterns) / max(len(patterns), 1),
        "without_either": sum(not row for row in patterns) / max(len(patterns), 1),
        "patterns": patterns,
    }


def measure_noise_models(inputs: BenchInputs) -> dict[str, Any]:
    """Build every empirical sampler payload and record immutable provenance."""
    track = measure_track_error(inputs)
    arcs = measure_arcs_and_dropouts(inputs)
    camera = measure_camera_registration(inputs)
    timing = measure_event_timing(inputs)
    anchors = measure_anchor_availability(inputs)
    return {
        "schema": "s6_3d_bench_empirical_noise_v1",
        "artifact_class": "evaluation_only_human_labelled_diagnostic",
        "seed": SEED,
        "code_provenance": _code_provenance(inputs),
        "track_error": track,
        "wrong_object_and_dropout": arcs,
        "camera_registration": camera,
        "event_timing": timing,
        "anchor_availability": anchors,
    }


class EmpiricalSamplers:
    """Deterministic bootstrap samplers over the measured payload and its provenance."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.track_sequences = payload["track_error"]["sequences"]
        self.wrong_arcs = payload["wrong_object_and_dropout"]["wrong_object_arcs"]
        self.dropout_lengths = payload["wrong_object_and_dropout"]["dropout_run_lengths"]
        self.registration_vectors = payload["camera_registration"][
            "registration_residual_vectors_px"
        ]
        self.fallback_vectors = payload["camera_registration"]["fallback_residual_vectors_px"]
        self.timing_errors = payload["event_timing"]["signed_error_frames"]
        self.anchor_patterns = payload["anchor_availability"]["patterns"]

    @staticmethod
    def _choice(rng: np.random.Generator, rows: Sequence[Any]) -> Any:
        if not rows:
            raise ValueError("empirical sampler has no measured rows")
        return rows[int(rng.integers(0, len(rows)))]

    def sample_track_errors(
        self,
        rng: np.random.Generator,
        sides: Sequence[str],
        *,
        contact_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Block-bootstrap signed error and covariance, preserving window autocorrelation."""
        output = np.zeros((len(sides), 2), dtype=float)
        covariances = np.full(len(sides), np.nan, dtype=float)
        cursor = 0
        while cursor < len(sides):
            target_side = sides[cursor]
            candidates = []
            for sequence in self.track_sequences:
                rows = [
                    row
                    for row in sequence["samples"]
                    if not row.get("missing")
                    and row.get("side") in {target_side, "unknown"}
                    and bool(row.get("contact_adjacent")) == contact_only
                ]
                if rows:
                    candidates.append(rows)
            block = self._choice(rng, candidates)
            start = int(rng.integers(0, len(block)))
            for row in block[start:]:
                if cursor >= len(sides) or sides[cursor] != target_side:
                    break
                output[cursor] = [row["dx_px"], row["dy_px"]]
                covariances[cursor] = float(row["innovation_cov_trace_native"])
                cursor += 1
        return output, covariances

    def sample_wrong_object_arcs(
        self, rng: np.random.Generator, frame_count: int
    ) -> list[tuple[int, np.ndarray]]:
        rate = float(
            self.payload["wrong_object_and_dropout"]["wrong_object_arc_start_rate_per_frame"]
        )
        count = int(rng.poisson(rate * frame_count))
        output = []
        measured_offsets = [
            np.asarray(arc["detour_offsets_px"], dtype=float)
            for arc in self.wrong_arcs
            if arc["detour_offsets_px"]
        ]
        if not measured_offsets:
            return output
        for _ in range(count):
            arc = self._choice(rng, self.wrong_arcs)
            duration = max(1, int(arc["duration_frames"]))
            offsets = self._choice(rng, measured_offsets)
            if len(offsets) != duration:
                indices = np.linspace(0, len(offsets) - 1, duration)
                offsets = np.vstack(
                    [
                        np.interp(indices, np.arange(len(offsets)), offsets[:, axis])
                        for axis in range(2)
                    ]
                ).T
            start = int(rng.integers(0, max(frame_count, 1)))
            output.append((start, offsets))
        return output

    def sample_dropout_mask(self, rng: np.random.Generator, frame_count: int) -> np.ndarray:
        rate = float(self.payload["wrong_object_and_dropout"]["dropout_start_rate_per_frame"])
        mask = np.zeros(frame_count, dtype=bool)
        if not self.dropout_lengths:
            return mask
        for _ in range(int(rng.poisson(rate * frame_count))):
            duration = int(self._choice(rng, self.dropout_lengths))
            start = int(rng.integers(0, max(frame_count, 1)))
            mask[start : min(frame_count, start + duration)] = True
        return mask

    def sample_camera_jitter(
        self, rng: np.random.Generator, fallback: Sequence[bool]
    ) -> np.ndarray:
        output = np.zeros((len(fallback), 2), dtype=float)
        fallback_indices = [index for index, value in enumerate(fallback) if value]
        for run in _contiguous_runs(fallback_indices):
            # A fallback holds one stale camera through a contiguous span.  Its error is
            # coherent, unlike sampled-frame registration jitter.
            value = np.asarray(
                self._choice(
                    rng,
                    self.fallback_vectors or self.registration_vectors,
                ),
                dtype=float,
            )
            output[run] = value
        for index, is_fallback in enumerate(fallback):
            if not is_fallback:
                output[index] = np.asarray(
                    self._choice(rng, self.registration_vectors), dtype=float
                )
        return output

    def sample_event_timing(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return np.asarray(
            [self._choice(rng, self.timing_errors) for _ in range(count)], dtype=float
        )

    def sample_anchor_pattern(self, rng: np.random.Generator) -> list[str]:
        return list(self._choice(rng, self.anchor_patterns))


# --------------------------------------------------------------------------------------- #
# The real chain's EMISSION conventions, measured on the cohort the nightly runs.
#
# ``docs/wk1/bench_frames.md`` section 8 lists what the frame-emitting bench still gets wrong
# or ignores about a real emission.  Every statistic below is one of those, measured here so
# a bench rung can carry it as a parameter instead of assuming it away.  Everything except
# the emitted-FRAME error is label-free: it is a property of the frozen automatic artifacts.
# The frame error needs the owner's clicks, because only the owner knows when the event was.
# --------------------------------------------------------------------------------------- #
EMISSION_COHORT_RELATIVE = Path("processed/wk3_eventthreshold/default_cohort")
STRIKER_LEDGER_RELATIVE = Path("processed/wk3_trackid/audit/contacts.csv")
EMISSION_REALISM_SCHEMA = "s6_emission_realism_v1"
EMISSION_MATCH_TOLERANCE_FRAMES = 3.0
# ``docs/wk1/contactframe.md``: the ball moves 30--60 native px per frame near court, so the
# gap that matters around a contact is a handful of frames either side.
CONTACT_GAP_WINDOW_FRAMES = 5


def _emission_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("emissions", payload) if isinstance(payload, dict) else payload
    output = []
    for row in rows:
        if row.get("abstain") is True:
            continue
        match_id = str(row.get("match_id") or str(row["clip"]).split("__")[0])
        clip = str(row["clip"])
        local = clip.removeprefix(f"{match_id}__")
        location = row.get("location") or {}
        output.append(
            {
                "match_id": match_id,
                "clip": clip,
                "local_clip": local,
                "event_type": str(row.get("event_type")),
                "frame": float(row["frame"]),
                "image_x": None if location.get("image_x") is None else float(location["image_x"]),
                "image_y": None if location.get("image_y") is None else float(location["image_y"]),
            }
        )
    return output


def _frame_homographies(match_dir: Path) -> dict[tuple[str, int], np.ndarray]:
    path = match_dir / "court_H_per_frame_v1.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {
            (str(clip), int(frame)): np.asarray(matrix, dtype=float)
            for clip, frame, matrix in zip(
                payload["clips"], payload["frames"], payload["H"], strict=True
            )
        }


def _side_from_homography(homography: np.ndarray | None, pixel: Sequence[float]) -> str:
    if homography is None:
        return "unknown"
    projected = np.asarray(homography, dtype=float) @ np.asarray(
        [float(pixel[0]), float(pixel[1]), 1.0], dtype=float
    )
    if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-9:
        return "unknown"
    return "near" if float(projected[1] / projected[2]) < NET_Y_M else "far"


def _camera_clips(match_dir: Path) -> set[str]:
    path = match_dir / "camera_P_per_frame_v1.npz"
    if not path.is_file():
        return set()
    with np.load(path, allow_pickle=False) as payload:
        return {str(clip) for clip in payload["clips"]}


def _owner_contact_and_bounce_truth() -> list[dict[str, Any]]:
    """The owner's accepted clicks, in the emission's own clip/frame space."""
    from cv.validation.current_standard_event_truth import authoritative_truth_path
    from cv.validation.score_cross_match_event_labels_v5 import ACCEPTED

    truth = []
    with authoritative_truth_path().open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in {"contact", "bounce"} or row["verdict"] not in ACCEPTED:
                continue
            frame_text = row["labeled_frame"] or row["seed_frame"]
            if not frame_text or not row["labeled_x540"] or not row["labeled_y540"]:
                continue
            truth.append(
                {
                    "clip": str(row["clip"]),
                    "event_type": str(row["event_type"]),
                    "frame": float(frame_text),
                    "owner_x": 2.0 * float(row["labeled_x540"]),
                    "owner_y": 2.0 * float(row["labeled_y540"]),
                }
            )
    return truth


def _striker_ledger(path: Path) -> dict[str, Any]:
    """The striker the fitter reads at each owner contact, from ``docs/wk1/track_id.md``."""
    if not path.is_file():
        return {"available": False, "path": str(path)}
    causes: dict[str, int] = defaultdict(int)
    ok_reach: list[float] = []
    wrong_reach: list[float] = []
    wrong_reach_m: list[float] = []
    ok_reach_m: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cause = str(row["cause"])
            causes[cause] += 1
            reach = row.get("striker_reach")
            reach_m = row.get("striker_ray_reach_m")
            if reach in (None, ""):
                continue
            if cause == "ok":
                ok_reach.append(float(reach))
                if reach_m not in (None, ""):
                    ok_reach_m.append(float(reach_m))
            elif cause in {"wrong_player_same_side", "wrong_side"}:
                wrong_reach.append(float(reach))
                if reach_m not in (None, ""):
                    wrong_reach_m.append(float(reach_m))
    # ``docs/wk1/track_id.md``'s own denominator: the contacts that have a sided box at all,
    # which excludes the 17 contacts whose point has no court homography.
    with_box = sum(
        count
        for cause, count in causes.items()
        if cause not in {"no_sided_box", "no_court_homography"}
    )
    wrong = causes.get("wrong_player_same_side", 0) + causes.get("wrong_side", 0)
    return {
        "available": True,
        "provenance": {
            "sources": [_source(path, role="striker box at each owner contact click")],
            "method": (
                "docs/wk1/track_id.md's own per-contact ledger; the wrong-body rate is over the "
                "contacts that have a sided box, and the reach is the ray-to-striker distance "
                "the fitter's reach band charges"
            ),
        },
        "causes": dict(sorted(causes.items())),
        "contacts_with_box": int(with_box),
        "wrong_body": int(wrong),
        "wrong_body_rate": wrong / with_box if with_box else 0.0,
        "wrong_side_share": (causes.get("wrong_side", 0) / wrong) if wrong else 0.0,
        "wrong_reach_body_heights": wrong_reach,
        "wrong_reach_m": wrong_reach_m,
        "ok_reach": _stats(ok_reach),
        "ok_reach_m": _stats(ok_reach_m),
        "wrong_reach": _stats(wrong_reach),
        "wrong_reach_m_stats": _stats(wrong_reach_m),
    }


def measure_emission_realism(
    cohort_root: Path,
    *,
    striker_ledger_path: Path | None = None,
    with_owner_frames: bool = True,
) -> dict[str, Any]:
    """Measure every real emission convention the frame-emitting bench does not model."""
    from cv.validation.event_pixel_accuracy import load_track
    from cv.validation.score_cross_match_event_labels_v5 import match_events

    emissions_path = cohort_root / "event_emissions.json"
    rows = _emission_rows(emissions_path)
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_match[row["match_id"]].append(row)

    head_samples: dict[str, list[list[float]]] = defaultdict(list)
    head_magnitudes: dict[str, list[float]] = defaultdict(list)
    off_track: dict[str, list[int]] = defaultdict(list)
    contact_gap_flags: list[int] = []
    contact_gap_lengths: list[int] = []
    contact_missing_at_frame = 0
    contacts_scored = 0
    flights = 0
    flights_without_bounce = 0
    terminal_spans = 0
    terminal_without_bounce = 0
    point_end_to_bounce: list[float] = []
    point_end_rows = 0
    camera_points = 0
    camera_abstained = 0
    homography_points = 0
    homography_abstained = 0
    sided: dict[str, str] = {}

    for match_id in sorted(by_match):
        match_dir = cohort_root / match_id
        track_path = match_dir / TRACK_NAME
        track = load_track(track_path) if track_path.is_file() else {}
        homographies = _frame_homographies(match_dir)
        clip_spans: dict[str, tuple[int, int]] = {}
        for clip_name, frame_number in track:
            low, high = clip_spans.get(clip_name, (frame_number, frame_number))
            clip_spans[clip_name] = (min(low, frame_number), max(high, frame_number))
        camera_clips = _camera_clips(match_dir)
        clips = sorted({row["local_clip"] for row in by_match[match_id]})
        for clip in clips:
            camera_points += 1
            if clip not in camera_clips:
                camera_abstained += 1
            homography_points += 1
            if not any(key[0] == clip for key in homographies):
                homography_abstained += 1
        for row in by_match[match_id]:
            clip = row["local_clip"]
            frame = int(round(row["frame"]))
            tracked = track.get((clip, frame))
            kind = row["event_type"]
            side = _side_from_homography(
                homographies.get((clip, frame)),
                (row["image_x"], row["image_y"])
                if row["image_x"] is not None
                else (tracked or (0.0, 0.0)),
            )
            sided[f"{row['clip']}#{frame}#{kind}"] = side
            off_track[kind].append(0 if tracked is not None else 1)
            if kind in {"contact", "bounce"} and tracked is not None and row["image_x"] is not None:
                dx = row["image_x"] - tracked[0]
                dy = row["image_y"] - tracked[1]
                key = f"{kind}|{side}"
                head_samples[key].append([float(dx), float(dy)])
                head_magnitudes[key].append(float(math.hypot(dx, dy)))
            if kind == "contact":
                contacts_scored += 1
                span = clip_spans.get(clip)
                missing = [
                    offset
                    for offset in range(-CONTACT_GAP_WINDOW_FRAMES, CONTACT_GAP_WINDOW_FRAMES + 1)
                    # Interior gaps only: a window that runs off the end of the tracked span
                    # is not a gap, and counting it would compare a bench point (a track that
                    # starts at the serve) with a broadcast clip (a track that does not).
                    if span is not None
                    and span[0] <= frame + offset <= span[1]
                    and (clip, frame + offset) not in track
                ]
                contact_gap_flags.append(1 if missing else 0)
                if missing:
                    contact_gap_lengths.append(len(missing))
                if tracked is None:
                    contact_missing_at_frame += 1
        # flights, per clip, exactly as the fitter builds them: contact to contact, then the
        # last contact to the point ending.
        for clip in clips:
            clip_rows = [row for row in by_match[match_id] if row["local_clip"] == clip]
            contacts = sorted(
                float(row["frame"]) for row in clip_rows if row["event_type"] == "contact"
            )
            bounces = sorted(
                float(row["frame"]) for row in clip_rows if row["event_type"] == "bounce"
            )
            ends = sorted(
                float(row["frame"]) for row in clip_rows if row["event_type"] == "point_end"
            )
            for start, stop in zip(contacts, contacts[1:], strict=False):
                flights += 1
                if not any(start < value < stop for value in bounces):
                    flights_without_bounce += 1
            if contacts and ends:
                terminal_spans += 1
                if not any(contacts[-1] < value <= ends[-1] + 1e-6 for value in bounces):
                    terminal_without_bounce += 1
            for end in ends:
                point_end_rows += 1
                if bounces:
                    point_end_to_bounce.append(float(min(abs(end - value) for value in bounces)))

    frame_offsets: dict[str, list[float]] = defaultdict(list)
    owner_matched = 0
    owner_truth_events = 0
    if with_owner_frames:
        truth = _owner_contact_and_bounce_truth()
        owner_truth_events = len(truth)
        predictions = [
            {
                "clip": row["clip"],
                "event_type": row["event_type"],
                "frame": row["frame"],
                "key": f"{row['clip']}#{int(round(row['frame']))}#{row['event_type']}",
            }
            for row in rows
            if row["event_type"] in {"contact", "bounce"}
        ]
        matches, _, _ = match_events(predictions, truth, EMISSION_MATCH_TOLERANCE_FRAMES)
        owner_matched = len(matches)
        for prediction, target in matches:
            side = sided.get(prediction["key"], "unknown")
            frame_offsets[f"{prediction['event_type']}|{side}"].append(
                float(prediction["frame"]) - float(target["frame"])
            )

    ledger_path = striker_ledger_path or Path()
    return {
        "schema": EMISSION_REALISM_SCHEMA,
        "artifact_class": "evaluation_only_human_labelled_diagnostic",
        "provenance": {
            "sources": [_source(emissions_path, role="frozen automatic emissions", rows=len(rows))],
            "cohort_root": str(cohort_root.resolve()),
            "method": (
                "every statistic is measured on the frozen cohort the nightly runs; only the "
                "emitted-frame offset joins the owner's accepted clicks, one-to-one within "
                "+/-3 native frames of the same event type"
            ),
        },
        "emissions": {kind: len(values) for kind, values in sorted(off_track.items())},
        "pixel_head": {
            key: {
                "count": len(values),
                "offsets_px": values,
                "magnitude": _stats(head_magnitudes[key]),
            }
            for key, values in sorted(head_samples.items())
        },
        "emitted_frame_offset": {
            key: {
                "count": len(values),
                "offsets_frames": values,
                "signed": _stats(values),
                "absolute": _stats(abs(value) for value in values),
            }
            for key, values in sorted(frame_offsets.items())
        },
        "emitted_frame_offset_matched": owner_matched,
        "owner_truth_events": owner_truth_events,
        "off_track_frame": {
            kind: {
                "count": len(values),
                "off_track": int(sum(values)),
                "rate": (sum(values) / len(values)) if values else 0.0,
            }
            for kind, values in sorted(off_track.items())
        },
        "contact_track_gap": {
            "contacts": contacts_scored,
            "window_frames": CONTACT_GAP_WINDOW_FRAMES,
            "with_gap_in_window": int(sum(contact_gap_flags)),
            "rate": (sum(contact_gap_flags) / contacts_scored) if contacts_scored else 0.0,
            "missing_at_emitted_frame": contact_missing_at_frame,
            "missing_frames_in_window": contact_gap_lengths,
            "missing_frames_in_window_stats": _stats(contact_gap_lengths),
        },
        "missing_bounce_emission": {
            "flights": flights,
            "without_bounce": flights_without_bounce,
            "rate": (flights_without_bounce / flights) if flights else 0.0,
            "terminal_spans": terminal_spans,
            "terminal_without_bounce": terminal_without_bounce,
            "terminal_rate": (
                (terminal_without_bounce / terminal_spans) if terminal_spans else 0.0
            ),
        },
        "point_end": {
            "count": point_end_rows,
            "frames_to_nearest_bounce": _stats(point_end_to_bounce),
            "within_one_frame_of_a_bounce": int(
                sum(1 for value in point_end_to_bounce if value <= 1.0)
            ),
            "scored": len(point_end_to_bounce),
        },
        "camera_abstention": {
            "points": camera_points,
            "without_camera": camera_abstained,
            "rate": (camera_abstained / camera_points) if camera_points else 0.0,
            "without_frame_homography": homography_abstained,
            "homography_rate": (
                (homography_abstained / homography_points) if homography_points else 0.0
            ),
        },
        "striker": _striker_ledger(ledger_path),
    }


class EmissionRealismSamplers:
    """Bootstrap samplers over :func:`measure_emission_realism`'s measured payload.

    Every draw is a bootstrap of a measured row, never a fitted distribution: the payload
    carries the raw per-event offsets and this class resamples them.
    """

    def __init__(self, payload: dict[str, Any]):
        if str(payload.get("schema")) != EMISSION_REALISM_SCHEMA:
            raise ValueError(f"not an emission realism payload: {payload.get('schema')}")
        self.payload = payload
        self.pixel_head = payload["pixel_head"]
        self.frame_offsets = payload["emitted_frame_offset"]
        self.off_track = payload["off_track_frame"]
        self.contact_gap = payload["contact_track_gap"]
        self.missing_bounce = payload["missing_bounce_emission"]
        self.camera = payload["camera_abstention"]
        self.striker = payload["striker"]

    @staticmethod
    def _choice(rng: np.random.Generator, rows: Sequence[Any]) -> Any:
        if not rows:
            raise ValueError("emission realism sampler has no measured rows")
        return rows[int(rng.integers(0, len(rows)))]

    def _cell(self, table: dict[str, Any], event_type: str, side: str, field: str) -> list[Any]:
        for key in (f"{event_type}|{side}", f"{event_type}|unknown", f"{event_type}|far"):
            row = table.get(key)
            if row and row[field]:
                return list(row[field])
        pooled = [
            value
            for key, row in table.items()
            if key.startswith(f"{event_type}|")
            for value in row[field]
        ]
        return pooled

    def sample_pixel_head(self, rng: np.random.Generator, event_type: str, side: str) -> np.ndarray:
        """The event model's x/y head, as an offset from the track on the emitted frame."""
        rows = self._cell(self.pixel_head, event_type, side, "offsets_px")
        return np.asarray(self._choice(rng, rows), dtype=float)

    def sample_frame_offset(self, rng: np.random.Generator, event_type: str, side: str) -> float:
        """The emitted frame minus the true event time, in frames, as the real chain errs."""
        rows = self._cell(self.frame_offsets, event_type, side, "offsets_frames")
        return float(self._choice(rng, rows))

    def allow_off_track_frame(self, rng: np.random.Generator, event_type: str) -> bool:
        row = self.off_track.get(event_type)
        rate = float(row["rate"]) if row else 0.0
        return bool(rng.random() < rate)

    def sample_contact_gap(self, rng: np.random.Generator) -> int:
        """Frames of track missing around a contact; zero when the real chain has none."""
        if rng.random() >= float(self.contact_gap["rate"]):
            return 0
        lengths = self.contact_gap["missing_frames_in_window"]
        return int(self._choice(rng, lengths)) if lengths else 0

    def drop_bounce_emission(self, rng: np.random.Generator, *, terminal: bool = False) -> bool:
        key = "terminal_rate" if terminal else "rate"
        return bool(rng.random() < float(self.missing_bounce[key]))

    def camera_abstains(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < float(self.camera["rate"]))

    def striker_wrong_body(self, rng: np.random.Generator) -> float | None:
        """Metres of extra ray-to-striker reach when the tracker holds the wrong body."""
        if not self.striker.get("available"):
            return None
        if rng.random() >= float(self.striker["wrong_body_rate"]):
            return None
        rows = self.striker["wrong_reach_m"]
        return float(self._choice(rng, rows)) if rows else None


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _generate_base(case: FlightCase) -> dict[str, Any] | None:
    try:
        with reference_fitter_adapter():
            synthetic = s6root_closed_loop.synthesize(case, _ZeroNoise())
            contact_frames = np.asarray(
                [synthetic["contacts"][0]["frame"], synthetic["contacts"][1]["frame"]],
                dtype=float,
            )
            contact_xyz = anchor_first_fit._simulate(
                synthetic["theta"],
                case.start_frame,
                contact_frames,
                case.fps,
                case.surface,
                synthetic["bounce_anchor"],
            )[0]
            frames = np.asarray(sorted(synthetic["ball"]), dtype=float)
            frame_xyz = anchor_first_fit._simulate(
                synthetic["theta"],
                case.start_frame,
                frames,
                case.fps,
                case.surface,
                synthetic["bounce_anchor"],
            )[0]
            bounce_frame = (
                float(synthetic["bounce_anchor"]["frame"])
                if synthetic["bounce_anchor"] is not None
                else None
            )
            if bounce_frame is None:
                bounce_xyz = None
                bounce_spin = None
            else:
                bounce_state = anchor_first_fit._simulate(
                    synthetic["theta"],
                    case.start_frame,
                    np.asarray([bounce_frame]),
                    case.fps,
                    case.surface,
                    synthetic["bounce_anchor"],
                )
                bounce_xyz = bounce_state[0][0]
                bounce_spin = bounce_state[2][0]
        return {
            "case": case,
            "synthetic": synthetic,
            "truth_contact_xyz": contact_xyz,
            "truth_frame_xyz": frame_xyz,
            "truth_bounce_frame": bounce_frame,
            "truth_bounce_xyz": bounce_xyz,
            "truth_bounce_spin": bounce_spin,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def build_base_cases(
    cases: Sequence[FlightCase], workers: int, target: int
) -> list[dict[str, Any]]:
    # About 90% of solved Arm-D seeds also cross and clear the net under the
    # reference generator.  Keep a deterministic reserve without paying to render
    # every remaining case during small smoke runs.
    candidate_count = min(len(cases), max(target + 20, math.ceil(target / 0.85)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        generated = list(pool.map(_generate_base, cases[:candidate_count], chunksize=1))
    selected = [row for row in generated if row is not None][:target]
    if len(selected) != target:
        raise RuntimeError(
            f"only generated {len(selected)} usable reference flights, need {target}"
        )
    return selected


@contextmanager
def _least_squares_options(
    *, robust_scale: float | None = None, single_start: bool = False
) -> Iterator[None]:
    original = anchor_first_fit.least_squares
    call_index = 0

    def wrapped(*args: Any, **kwargs: Any):
        nonlocal call_index
        call_index += 1
        if single_start and call_index > 1:
            raise ValueError("validation single-start ablation")
        if robust_scale is not None:
            kwargs["loss"] = "soft_l1"
            kwargs["f_scale"] = float(robust_scale)
        return original(*args, **kwargs)

    anchor_first_fit.least_squares = wrapped
    try:
        yield
    finally:
        anchor_first_fit.least_squares = original


_FIT_BASES: list[dict[str, Any]] = []
_FIT_NOISE: dict[str, Any] = {}


def _init_fit_workers(bases: list[dict[str, Any]], noise: dict[str, Any]) -> None:
    global _FIT_BASES, _FIT_NOISE
    _FIT_BASES = bases
    _FIT_NOISE = noise


def _apply_rung(
    base: dict[str, Any], rung: str, lever: str, rng: np.random.Generator
) -> tuple[FlightCase, dict[str, Any]]:
    samplers = EmpiricalSamplers(_FIT_NOISE)
    case = copy.copy(base["case"])
    synthetic = base["synthetic"]
    frames = np.asarray(sorted(synthetic["ball"]), dtype=int)
    ball = np.stack([synthetic["ball"][int(frame)] for frame in frames]).astype(float)
    ball += rng.normal(0.0, 1.0, ball.shape)
    contacts = copy.deepcopy(synthetic["contacts"])
    anchors = copy.deepcopy(synthetic["anchors"])
    truth_positions = np.asarray(base["truth_frame_xyz"], dtype=float)
    sides = ["near" if float(xyz[1]) < NET_Y_M else "far" for xyz in truth_positions]
    covariance = np.full(len(frames), np.nan, dtype=float)
    contact_mask = np.asarray(
        [
            any(abs(float(frame) - float(contact["frame"])) <= 2.0 for contact in contacts)
            for frame in frames
        ],
        dtype=bool,
    )

    track_errors = np.zeros_like(ball)
    if rung in {"track_error", "realistic"}:
        track_errors, covariance = samplers.sample_track_errors(rng, sides, contact_only=False)
        ball += track_errors
    if rung in {"contact_adjacent", "realistic"}:
        errors, contact_covariance = samplers.sample_track_errors(rng, sides, contact_only=True)
        if rung == "realistic":
            # Track and contact samplers are absolute conditional residuals, not
            # independent increments.  Replace the interior draw at contacts.
            ball[contact_mask] -= track_errors[contact_mask]
        ball[contact_mask] += errors[contact_mask]
        covariance[contact_mask] = contact_covariance[contact_mask]
    if rung in {"wrong_object", "realistic"}:
        for start, offsets in samplers.sample_wrong_object_arcs(rng, len(frames)):
            stop = min(len(frames), start + len(offsets))
            ball[start:stop] += offsets[: stop - start]
    drop_mask = np.zeros(len(frames), dtype=bool)
    if rung in {"dropped_frames", "realistic"}:
        drop_mask = samplers.sample_dropout_mask(rng, len(frames))
    if rung in {"camera_registration", "realistic"}:
        fallback = [
            "anchor_static_fallback" in str(case.camera.quality_at(float(frame)).get("source", ""))
            for frame in frames
        ]
        ball += samplers.sample_camera_jitter(rng, fallback)
    if rung in {"event_timing", "realistic"}:
        shifts = samplers.sample_event_timing(rng, 2)
        contacts[0]["frame"] = float(contacts[0]["frame"]) + float(shifts[0])
        contacts[1]["frame"] = float(contacts[1]["frame"]) + float(shifts[1])
        if contacts[1]["frame"] - contacts[0]["frame"] < 4.0:
            contacts[1]["frame"] = contacts[0]["frame"] + 4.0
    if rung in {"missing_anchors", "realistic"}:
        pattern = samplers.sample_anchor_pattern(rng)
        allowed = set(pattern)
        anchors = [row for row in anchors if row.get("type") in allowed]

    mapped_ball = {
        int(frame): pixel
        for frame, pixel, dropped in zip(frames, ball, drop_mask, strict=True)
        if not dropped
    }
    if lever == "drop_contact_adjacent":
        mapped_ball = {
            frame: pixel
            for frame, pixel in mapped_ball.items()
            if not any(abs(float(frame) - float(contact["frame"])) <= 2.0 for contact in contacts)
        }
    weights = {frame: 1.0 for frame in mapped_ball}
    if lever == "innovation_covariance_weights":
        finite = covariance[np.isfinite(covariance) & (covariance > 0.0)]
        reference = float(np.median(finite)) if len(finite) else 1.0
        by_frame = dict(zip(frames.tolist(), covariance.tolist(), strict=True))
        weights = {
            frame: float(np.clip(reference / by_frame.get(frame, reference), 0.01, 1.0))
            if np.isfinite(by_frame.get(frame, math.nan)) and by_frame.get(frame, 0.0) > 0.0
            else 1.0
            for frame in mapped_ball
        }
    case.ball = mapped_ball
    case.contacts = contacts
    case.anchors = anchors
    case.players = copy.deepcopy(synthetic["players"])
    case.observation_weights = weights
    case.attempt = {
        **case.attempt,
        "start_frame": float(contacts[0]["frame"]),
        "end_frame": float(contacts[1]["frame"]),
    }
    return case, {"frames": frames, "covariance": covariance}


def _fit_task(task: tuple[int, str, str, int, int]) -> dict[str, Any]:
    base_index, rung, lever, seed, max_nfev = task
    base = _FIT_BASES[base_index]
    # Every lever must see the exact corruption used by the default ladder row;
    # including the lever name here would turn a paired ablation into a cohort draw.
    rng = np.random.default_rng(
        _stable_seed(seed, rung, "default_multistart", base["case"].flight_id)
    )
    robust = {
        "robust_soft_l1_f0p5": 0.5,
        "robust_soft_l1_f2": 2.0,
        "robust_soft_l1_f8": 8.0,
    }.get(lever)
    case, _ = _apply_rung(base, rung, lever, rng)
    reference_fit = lever == "reference_physics_fit"
    physics_context = reference_fitter_adapter() if reference_fit else nullcontext()
    with (
        physics_context,
        _least_squares_options(robust_scale=robust, single_start=lever == "single_start_ablation"),
    ):
        fit = fit_case(case, max_nfev=max_nfev)
        if lever == "warm_restart_multistart" and fit is not None:
            restarted = fit_case(case, max_nfev=max_nfev, initial_fit=fit)
            if restarted is not None:
                first_error = np.median(getattr(fit, "_held_out_errors_px", [math.inf]))
                second_error = np.median(getattr(restarted, "_held_out_errors_px", [math.inf]))
                if second_error <= first_error:
                    fit = restarted
        status, reasons, compact = compact_and_classify(case, fit)
        contact_errors: list[float] = []
        start_error = None
        end_error = None
        bounce_error = None
        spin_error_rpm = None
        spin_identifiable = None
        if fit is not None:
            fitted_contacts = []
            # A timing perturbation changes the fitted flight's domain.  Score its
            # two modeled endpoints against the true contact locations; evaluating
            # at the unperturbed frame can fall just outside that domain.
            for contact in case.contacts:
                frame = float(contact["frame"])
                fitted_contacts.append(
                    np.asarray(fit.state(frame, case.fps, case.surface)[0], dtype=float)
                )
            contact_deltas = np.stack(fitted_contacts) - np.asarray(base["truth_contact_xyz"])
            start_error = contact_deltas[0]
            end_error = contact_deltas[1]
            contact_errors = np.linalg.norm(contact_deltas, axis=1).tolist()
            bounce_frame = base["truth_bounce_frame"]
            if bounce_frame is not None:
                fitted_bounce = np.asarray(
                    fit.state(float(bounce_frame), case.fps, case.surface)[0]
                )
                bounce_error = float(np.linalg.norm(fitted_bounce - base["truth_bounce_xyz"]))
                spins = np.asarray(getattr(fit, "_ws_obs", []), dtype=float)
                observed = np.asarray(getattr(fit, "obs_frames", []), dtype=float)
                if len(spins) and len(spins) == len(observed):
                    fitted_spin = spins[int(np.argmin(np.abs(observed - float(bounce_frame))))]
                    truth_spin = np.asarray(base["truth_bounce_spin"], dtype=float)
                    spin_error_rpm = float(
                        np.linalg.norm(fitted_spin - truth_spin) * 60.0 / (2.0 * np.pi)
                    )
                    denominator = max(float(np.linalg.norm(truth_spin)), 1e-9)
                    cosine = float(
                        np.dot(fitted_spin, truth_spin)
                        / max(float(np.linalg.norm(fitted_spin)) * denominator, 1e-9)
                    )
                    relative = abs(float(np.linalg.norm(fitted_spin)) - denominator) / denominator
                    spin_identifiable = cosine >= 0.8 and relative <= 0.5
    return {
        "rung": rung,
        "lever": lever,
        "flight_id": case.flight_id,
        "point": case.point,
        "match_id": case.match_id,
        "start_frame": float(base["synthetic"]["contacts"][0]["frame"]),
        "end_frame": float(base["synthetic"]["contacts"][1]["frame"]),
        "bounce_truth": base["truth_bounce_frame"] is not None,
        "net_anchor_supplied": any(row.get("type") == "net_crossing" for row in case.anchors),
        "bounce_anchor_supplied": any(row.get("type") == "bounce" for row in case.anchors),
        "observations": len(case.ball),
        "solved": fit is not None,
        "accepted": bool(compact and compact.get("held_out_accepted")),
        "composite_accepted": status == "provisional_valid",
        "status": status,
        "reasons": reasons,
        "held_out_median_px": compact.get("held_out_reprojection_median_px") if compact else None,
        "held_out_p90_px": compact.get("held_out_reprojection_p90_px") if compact else None,
        "bounce_position_error_m": bounce_error,
        "contact_position_error_m": float(np.median(contact_errors)) if contact_errors else None,
        "start_position_error_xyz": start_error.tolist() if start_error is not None else None,
        "end_position_error_xyz": end_error.tolist() if end_error is not None else None,
        "spin_error_rpm": spin_error_rpm,
        "spin_identifiable": spin_identifiable,
        "optimizer_starts": compact.get("optimizer_starts") if compact else None,
        "optimizer_nfev": compact.get("optimizer_nfev") if compact else None,
    }


def _junction_gaps(rows: list[dict[str, Any]]) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["start_position_error_xyz"] is not None
            and row["end_position_error_xyz"] is not None
        ):
            grouped[row["point"]].append(row)
    gaps = []
    for point_rows in grouped.values():
        ordered = sorted(point_rows, key=lambda row: (row["start_frame"], row["end_frame"]))
        for left, right in zip(ordered, ordered[1:]):
            if abs(float(left["end_frame"]) - float(right["start_frame"])) > 1e-6:
                continue
            # Translate both synthetic truths to the same contact.  The residual-vector
            # difference is exactly the junction gap that remains after that alignment.
            gap = np.asarray(left["end_position_error_xyz"]) - np.asarray(
                right["start_position_error_xyz"]
            )
            gaps.append(float(np.linalg.norm(gap)))
    return gaps


def summarize_rows(rung: str, lever: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [row["held_out_median_px"] for row in rows if row["held_out_median_px"] is not None]
    p90s = [row["held_out_p90_px"] for row in rows if row["held_out_p90_px"] is not None]
    bounce = [
        row["bounce_position_error_m"] for row in rows if row["bounce_position_error_m"] is not None
    ]
    contacts = [
        row["contact_position_error_m"]
        for row in rows
        if row["contact_position_error_m"] is not None
    ]
    spins = [row for row in rows if row["spin_identifiable"] is not None]
    gaps = _junction_gaps(rows)
    return {
        "rung": rung,
        "lever": lever,
        "flights": len(rows),
        "solved": sum(bool(row["solved"]) for row in rows),
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "net_anchor_supplied": sum(bool(row["net_anchor_supplied"]) for row in rows),
        "bounce_anchor_supplied": sum(bool(row["bounce_anchor_supplied"]) for row in rows),
        "held_out_median_px": percentile(medians, 50.0),
        "held_out_median_px_p90": percentile(medians, 90.0),
        "held_out_p90_px": percentile(p90s, 50.0),
        "bounce_position_error_m": percentile(bounce, 50.0),
        "bounce_position_error_m_p90": percentile(bounce, 90.0),
        "contact_position_error_m": percentile(contacts, 50.0),
        "contact_position_error_m_p90": percentile(contacts, 90.0),
        "junction_gap_m": percentile(gaps, 50.0),
        "junction_gap_m_p90": percentile(gaps, 90.0),
        "junction_pairs": len(gaps),
        "spin_identifiable": sum(bool(row["spin_identifiable"]) for row in spins),
        "spin_evaluated": len(spins),
        "spin_error_rpm": percentile(
            [row["spin_error_rpm"] for row in spins if row["spin_error_rpm"] is not None], 50.0
        ),
    }


def proposed_thresholds(flights: int) -> dict[str, dict[str, float | int]]:
    scale = flights / DEFAULT_FLIGHTS
    return {
        "clean": {
            "minimum_solved": math.ceil(198 * scale),
            "minimum_accepted": math.ceil(190 * scale),
            "maximum_held_out_median_px": 2.0,
            "maximum_held_out_p90_px": 4.0,
        },
        "isolated": {
            "minimum_solved": math.ceil(180 * scale),
            "minimum_accepted": math.ceil(180 * scale),
            "maximum_held_out_median_px": 4.0,
            "maximum_held_out_p90_px": 12.0,
            "maximum_contact_position_error_m": 0.35,
        },
        "missing_anchors": {
            "minimum_solved": math.ceil(160 * scale),
            "minimum_accepted": math.ceil(150 * scale),
            "maximum_held_out_median_px": 4.0,
            "maximum_held_out_p90_px": 12.0,
        },
        "realistic": {
            "minimum_solved": math.ceil(150 * scale),
            "minimum_accepted": math.ceil(140 * scale),
            "maximum_held_out_median_px": 6.0,
            "maximum_held_out_p90_px": 15.0,
            "maximum_contact_position_error_m": 0.50,
            "maximum_junction_gap_m": 0.75,
        },
    }


def evaluate_thresholds(table: list[dict[str, Any]], flights: int) -> dict[str, Any]:
    thresholds = proposed_thresholds(flights)
    results = []
    for row in table:
        if row["lever"] != "default_multistart":
            continue
        key = (
            row["rung"] if row["rung"] in {"clean", "missing_anchors", "realistic"} else "isolated"
        )
        limits = thresholds[key]
        failures = []
        for field, limit in limits.items():
            if field.startswith("minimum_"):
                metric = field.removeprefix("minimum_")
                if row.get(metric) is None or float(row[metric]) < float(limit):
                    failures.append(f"{metric}={row.get(metric)} < {limit}")
            elif field.startswith("maximum_"):
                metric = field.removeprefix("maximum_")
                if row.get(metric) is None or float(row[metric]) > float(limit):
                    failures.append(f"{metric}={row.get(metric)} > {limit}")
        results.append({"rung": row["rung"], "passed": not failures, "failures": failures})
    return {
        "thresholds": thresholds,
        "rungs": results,
        "passed": bool(results) and all(row["passed"] for row in results),
    }


def intolerant_rungs(table: list[dict[str, Any]]) -> list[str]:
    by_rung = {row["rung"]: row for row in table if row["lever"] == "default_multistart"}
    clean = by_rung["clean"]
    selected = []
    for rung, row in by_rung.items():
        if rung == "clean":
            continue
        accepted_drop = int(clean["accepted"]) - int(row["accepted"])
        solved_drop = int(clean["solved"]) - int(row["solved"])
        median_rise = (row["held_out_median_px"] or math.inf) - (clean["held_out_median_px"] or 0.0)
        p90_rise = (row["held_out_p90_px"] or math.inf) - (clean["held_out_p90_px"] or 0.0)
        contact_rise = (row["contact_position_error_m"] or math.inf) - (
            clean["contact_position_error_m"] or 0.0
        )
        junction_rise = (row["junction_gap_m"] or math.inf) - (clean["junction_gap_m"] or 0.0)
        if (
            rung == "realistic"
            or solved_drop >= 10
            or accepted_drop >= 10
            or median_rise >= 2.0
            or p90_rise >= 4.0
            or contact_rise >= 0.10
            or junction_rise >= 0.20
        ):
            selected.append(rung)
    return selected


def _run_tasks(
    pool: ProcessPoolExecutor,
    tasks: list[tuple[int, str, str, int, int]],
) -> list[dict[str, Any]]:
    return list(pool.map(_fit_task, tasks, chunksize=1))


SUMMARY_FIELDS = [
    "rung",
    "lever",
    "flights",
    "solved",
    "accepted",
    "net_anchor_supplied",
    "bounce_anchor_supplied",
    "held_out_median_px",
    "held_out_median_px_p90",
    "held_out_p90_px",
    "bounce_position_error_m",
    "bounce_position_error_m_p90",
    "contact_position_error_m",
    "contact_position_error_m_p90",
    "junction_gap_m",
    "junction_gap_m_p90",
    "junction_pairs",
    "spin_identifiable",
    "spin_evaluated",
    "spin_error_rpm",
]


def summary_lines(table: list[dict[str, Any]]) -> list[str]:
    lines = ["rung                 solved px-accept held-med held-p90 contact-m junction-m spin"]
    for row in table:
        spin = f"{row['spin_identifiable']}/{row['spin_evaluated']}"
        values = [
            f"{row['rung']:<20}",
            f"{row['solved']:>3}/{row['flights']:<3}",
            f"{row['accepted']:>3}/{row['flights']:<3}",
            f"{(row['held_out_median_px'] or math.nan):>8.2f}",
            f"{(row['held_out_p90_px'] or math.nan):>8.2f}",
            f"{(row['contact_position_error_m'] or math.nan):>9.3f}",
            f"{(row['junction_gap_m'] or math.nan):>10.3f}",
            f"{spin:>9}",
        ]
        lines.append(" ".join(values))
    return lines


def run_benchmark(
    inputs: BenchInputs,
    output_root: Path,
    *,
    flights: int,
    workers: int,
    max_nfev: int,
    skip_levers: bool,
) -> dict[str, Any]:
    noise = measure_noise_models(inputs)
    write_json(output_root / "noise_models.json", noise)
    cases = load_flight_cases(inputs.oracle_root, solved_only=True)
    bases = build_base_cases(cases, workers, flights)
    selection = [row["case"].flight_id for row in bases]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_fit_workers,
        initargs=(bases, noise),
    ) as pool:
        ladder_tasks = [
            (index, rung, "default_multistart", SEED, max_nfev)
            for rung in RUNG_NAMES
            for index in range(flights)
        ]
        ladder_rows = _run_tasks(pool, ladder_tasks)
        ladder_table = [
            summarize_rows(
                rung,
                "default_multistart",
                [row for row in ladder_rows if row["rung"] == rung],
            )
            for rung in RUNG_NAMES
        ]
        gate = evaluate_thresholds(ladder_table, flights)
        ladder = {
            "schema": "s6_3d_bench_ladder_v1",
            "artifact_class": "evaluation_only_human_labelled_diagnostic",
            "seed": SEED,
            "generator": "physics.reference via validation-local adapter",
            "fitter": "unchanged cv.pipeline.anchor_first_fit.fit_anchor_first_shot",
            "baseline_sensor_noise": "independent 1.0 native-pixel Gaussian on every rung",
            "pixel_gate": {
                "maximum_held_out_median_px": anchor_first_fit.HELD_OUT_MEDIAN_LIMIT_PX,
                "maximum_held_out_p90_px": anchor_first_fit.HELD_OUT_P90_LIMIT_PX,
            },
            "code_provenance": noise["code_provenance"],
            "flights_per_rung": flights,
            "selection": selection,
            "table": ladder_table,
            "threshold_gate": gate,
            "rows": ladder_rows,
        }
        write_json(output_root / "ladder.json", ladder)
        write_csv(output_root / "ladder.csv", ladder_table, SUMMARY_FIELDS)
        selected_rungs = intolerant_rungs(ladder_table)
        lever_rows: list[dict[str, Any]] = []
        lever_table: list[dict[str, Any]] = []
        if not skip_levers:
            lever_tasks = [
                (index, rung, lever, SEED, max_nfev)
                for rung in selected_rungs
                for lever in LEVER_NAMES
                for index in range(flights)
            ]
            lever_rows = _run_tasks(pool, lever_tasks)
            lever_table = [
                summarize_rows(
                    rung,
                    lever,
                    [row for row in lever_rows if row["rung"] == rung and row["lever"] == lever],
                )
                for rung in selected_rungs
                for lever in LEVER_NAMES
            ]
    levers = {
        "schema": "s6_3d_bench_robustness_levers_v1",
        "artifact_class": "evaluation_only_human_labelled_diagnostic",
        "rungs": selected_rungs,
        "call_only_contract": (
            "observation weights, observation removal, initial_fit warm restart, validation-local "
            "SciPy loss kwargs, and validation-local reference sampler substitution; no pipeline "
            "or physics source changed"
        ),
        "code_provenance": noise["code_provenance"],
        "table": lever_table,
        "rows": lever_rows,
    }
    write_json(output_root / "levers.json", levers)
    if lever_table:
        write_csv(output_root / "levers.csv", lever_table, SUMMARY_FIELDS)
    (output_root / "summary.txt").write_text("\n".join(summary_lines(ladder_table)) + "\n")
    return {"noise": noise, "ladder": ladder, "levers": levers}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure", help="measure and freeze empirical samplers")
    measure_parser.add_argument("--output-root", type=Path)
    emission_parser = subparsers.add_parser(
        "emission-model",
        help="measure the real chain's emission conventions for the point bench's rungs",
    )
    emission_parser.add_argument("--output-root", type=Path)
    emission_parser.add_argument("--cohort-root", type=Path)
    emission_parser.add_argument("--striker-ledger", type=Path)
    run_parser = subparsers.add_parser("run", help="measure, generate, fit, score, and gate")
    run_parser.add_argument("--output-root", type=Path)
    run_parser.add_argument("--flights", type=int, default=DEFAULT_FLIGHTS)
    run_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run_parser.add_argument("--max-nfev", type=int, default=20)
    run_parser.add_argument("--skip-levers", action="store_true")
    run_parser.add_argument("--allow-threshold-failure", action="store_true")
    args = parser.parse_args()
    data_root = _data_root()
    inputs = BenchInputs.defaults(data_root, _repository_root())
    output_root = (args.output_root or data_root / OUTPUT_RELATIVE).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "measure":
        payload = measure_noise_models(inputs)
        write_json(output_root / "noise_models.json", payload)
        print(
            json.dumps({key: value for key, value in payload.items() if key != "schema"}, indent=2)[
                :8000
            ]
        )
        return
    if args.command == "emission-model":
        payload = measure_emission_realism(
            args.cohort_root or data_root / EMISSION_COHORT_RELATIVE,
            striker_ledger_path=args.striker_ledger or data_root / STRIKER_LEDGER_RELATIVE,
        )
        write_json(output_root / "real_emission_model_v1.json", payload)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"pixel_head", "emitted_frame_offset", "striker"}
                },
                indent=2,
            )
        )
        return
    if args.flights <= 0 or args.workers <= 0:
        parser.error("--flights and --workers must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    result = run_benchmark(
        inputs,
        output_root,
        flights=args.flights,
        workers=min(args.workers, DEFAULT_WORKERS),
        max_nfev=args.max_nfev,
        skip_levers=args.skip_levers,
    )
    print("\n".join(summary_lines(result["ladder"]["table"])))
    gate = result["ladder"]["threshold_gate"]
    print(f"readiness_gate={'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"] and not args.allow_threshold_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
