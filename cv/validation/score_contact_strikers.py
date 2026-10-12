#!/usr/bin/env python3
"""Score per-contact striker resolution against the labelled hitters.

Evaluation-only.  Labels supply the contact times and the truth hitter end; every pixel the
resolver sees is automatic (S3 person boxes, the sided track, S4 ball track, S2 camera).
Nothing here writes into an inference root.

Three arms are scored on the same contacts, so the report separates the coordinate-contract
repair from the resolver itself:

``legacy_half_native``
    the nearest sided box to the contact pixel, read as if the box artifact were native when
    its sidecar declares a smaller space -- the defect that gave USO2020 pt0001's near-court
    volley to the far player;
``sided_nearest``
    the same nearest-box rule with the sidecar honoured;
``resolver``
    ``cv.pipeline.contact_striker`` with its side, scale, continuity and alternation gates.

Truth ends come from two provenances and are reported separately.  ``labelled`` ends are the
explicit ``hitter``/``hitter_end`` fields of the eight astralabels3 extension attempts.
``derived`` ends are read off the nine prior attempts' frozen review narratives, which name
the server's camera half and an alternating shot sequence; they are circular with respect to
the resolver's alternation prior and are never pooled with the labelled arm.

    uv run python -m cv.validation.score_contact_strikers --output <report.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from cv.pipeline import contact_striker as striker
from cv.pipeline import resolution as res
from cv.pipeline.pose import load_native_pose_rows
from cv.validation.current_standard_event_truth import authoritative_truth_path
from cv.validation.score_cross_match_event_labels_v5 import ACCEPTED

SCHEMA = "contact_striker_score_v1"
SELECTION = "processed/s6_agent_inputs/astralabels3/current_selection.json"
DEFAULT_COHORT = "processed/postseg_pipeline_benchmark_969657a"
LABEL_ROOT = Path("cv/validation/labels/s6_agent_inputs_v1")
TOPOLOGY = LABEL_ROOT / "attempt_topologies_astrareview_v1.json"
POSE_NAME = "player_pose_tracked_crop_native_v1.csv"
BALL_NAME = "ball_track_joint_native1080_integrity_v1.csv"
OTHER_END = {"near": "far", "far": "near"}
HITTER_TRUTH = LABEL_ROOT / "hitters_v1.json"
HITTER_SAMPLE = LABEL_ROOT / "astralandmarks_hitter_sample_v1.json"


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def frame_number(value: str) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        raise ValueError(f"no frame index in {value!r}")
    return int(digits)


def _sizes(path: Path) -> tuple[res.FrameSize, res.FrameSize]:
    manifest = res.read_coordinate_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"{path} has no coordinate sidecar")
    image = manifest["image_size"]
    artifact = manifest.get("legacy_artifact_size", manifest["artifact_size"])
    return (
        res.FrameSize(int(image["width"]), int(image["height"])),
        res.FrameSize(int(artifact["width"]), int(artifact["height"])),
    )


def _rows(path: Path, clip: str) -> list[dict]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("clip") == clip]


def truth_ends(selection: list[dict]) -> dict[str, dict]:
    """Per-attempt truth ends, tagged with the provenance that produced them."""
    topology = json.loads(TOPOLOGY.read_text())
    derived = {}
    for attempt in topology["attempts"]:
        half = attempt["server_camera_half"]
        derived[attempt["key"]] = {
            "ends": [
                half if index % 2 == 0 else OTHER_END[half]
                for index in range(len(attempt["contact_frames"]))
            ],
            "frames": [float(frame) for frame in attempt["contact_frames"]],
            "provenance": "derived_from_review_narrative",
        }
    out = {}
    for entry in selection:
        label = json.loads(Path(entry["path"]).read_text())
        contacts = [
            record for record in label["events"]["records"] if record.get("event_type") == "contact"
        ]
        if all(record.get("hitter_end") for record in contacts):
            out[entry["key"]] = {
                "ends": [record["hitter_end"] for record in contacts],
                "hitters": [record.get("hitter") for record in contacts],
                "frames": [float(record["frame"]) for record in contacts],
                "provenance": "labelled_hitter",
            }
            continue
        if entry["key"] in derived:
            record = dict(derived[entry["key"]])
            record["frames"] = [float(item["frame"]) for item in contacts]
            out[entry["key"]] = record
    return out


def ball_pixels(path: Path, clip: str) -> dict[int, tuple[float, float]]:
    """Native contact-pixel evidence: the automatic ball track, in its declared space."""
    image_size, artifact_size = _sizes(path)
    scale_x = res.NATIVE_SIZE.width / artifact_size.width
    scale_y = res.NATIVE_SIZE.height / artifact_size.height
    del image_size
    out: dict[int, tuple[float, float]] = {}
    for row in _rows(path, clip):
        try:
            out[frame_number(row["frame"])] = (
                float(row["x"]) * scale_x,
                float(row["y"]) * scale_y,
            )
        except (KeyError, ValueError):
            continue
    return out


def contact_pixel(track: dict[int, tuple[float, float]], frame: float):
    """The ball pixel at a possibly half-frame contact, or None when the track has no row."""
    low, high = int(np.floor(frame)), int(np.ceil(frame))
    if low == high:
        return track.get(low)
    if low in track and high in track:
        return tuple(0.5 * (np.array(track[low]) + np.array(track[high])))
    return track.get(high, track.get(low))


def load_camera(path: Path, clip: str) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    out: dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        reliable = data["reliable"] if "reliable" in data.files else None
        for index, (row_clip, frame, projection) in enumerate(
            zip(data["clips"], data["frames"], data["P"], strict=True)
        ):
            if str(row_clip) != clip:
                continue
            if reliable is not None and not bool(reliable[index]):
                continue
            matrix = np.asarray(projection, float)
            if np.isfinite(matrix).all():
                out[int(frame)] = matrix
    return out


def nearest_sided_end(pixel, frame, sided_by_frame, window, box_scale):
    return striker._nearest_sided_end(pixel, frame, sided_by_frame, window, box_scale)


def _blank_row(
    key: str,
    match_id: str,
    clip: str,
    frame: float,
    truth_end: str,
    provenance: str,
    reason: str,
) -> dict:
    """A contact no arm could answer. It stays in the denominator with its reason."""
    return {
        "attempt": key,
        "match_id": match_id,
        "clip": clip,
        "frame": float(frame),
        "truth_end": truth_end,
        "truth_provenance": provenance,
        "legacy_half_native_end": None,
        "sided_nearest_end": None,
        "resolver_end": None,
        "resolver_status": "abstain",
        "confidence": 0.0,
        "reach": None,
        "reach_end_margin": None,
        "end_source": None,
        "scale_reference": None,
        "camera_scale_ratio": None,
        "alternation_applied": False,
        "disagrees_with_sided_track": None,
        "wrist_px": None,
        "root_px": None,
        "wrist_box_heights": None,
        "root_box_heights": None,
        "pose_matched": False,
        "abstain_reason": reason,
    }


def score_attempt(key: str, label_path: Path, cohort: Path, truth: dict) -> dict:
    label = json.loads(label_path.read_text())
    match_id, clip = label["match_id"], label["attempt"]["clip"]
    match_dir = cohort / match_id
    raw_paths = sorted(match_dir.glob("player_boxes_*_native_v1.csv"))
    sided_paths = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))

    def _unscored(reason: str) -> dict:
        """Every contact of an attempt the automatic inputs cannot reach, in the denominator."""
        return {
            "attempt": key,
            "status": "abstain",
            "reason": reason,
            "match_id": match_id,
            "clip": clip,
            "contacts": [
                _blank_row(key, match_id, clip, frame, end, truth["provenance"], reason)
                for frame, end in zip(truth["frames"], truth["ends"], strict=True)
            ],
        }

    if not raw_paths:
        return _unscored("no_automatic_player_boxes")
    raw_path = raw_paths[0]
    image_size, artifact_size = _sizes(raw_path)
    box_scale = res.NATIVE_SIZE.width / artifact_size.width
    try:
        fps = float(raw_path.stem.split("_")[2])
    except (IndexError, ValueError):
        fps = 25.0

    homographies = {}
    with np.load(match_dir / "court_H_per_point.npz") as data:
        for point, homography in zip(data["pts"], data["H"], strict=True):
            matrix = np.asarray(homography, float)
            if np.isfinite(matrix).all():
                homographies[int(point)] = matrix
    point = int(clip.removeprefix("pt"))
    if point not in homographies:
        return _unscored("no_automatic_court_geometry")
    homography = res.image_to_world_homography(homographies[point], image_size, res.NATIVE_SIZE)
    projections = load_camera(match_dir / "camera_P_per_frame_v1.npz", clip)

    detections: dict[int, list[dict]] = {}
    for row in _rows(raw_path, clip):
        detections.setdefault(frame_number(row["frame"]), []).append(row)
    sided_by_frame: dict[int, list[dict]] = {}
    track_by_end: dict[str, dict[int, np.ndarray]] = {"near": {}, "far": {}}
    sided_scale = box_scale
    if sided_paths:
        _, sided_artifact = _sizes(sided_paths[0])
        sided_scale = res.NATIVE_SIZE.width / sided_artifact.width
        for row in _rows(sided_paths[0], clip):
            frame = frame_number(row["frame"])
            sided_by_frame.setdefault(frame, []).append(row)
            side = str(row.get("side") or "")
            if side in track_by_end:
                track_by_end[side][frame] = np.array([float(row["court_x"]), float(row["court_y"])])
    pose_by_frame: dict[int, list[dict]] = {}
    pose_path = match_dir / POSE_NAME
    if pose_path.exists():
        for row in load_native_pose_rows(str(pose_path)):
            if row.get("clip") == clip:
                pose_by_frame.setdefault(frame_number(row["frame"]), []).append(row)

    track = ball_pixels(match_dir / BALL_NAME, clip)
    contacts, missing_pixel = [], []
    for frame in truth["frames"]:
        pixel = contact_pixel(track, frame)
        if pixel is None:
            missing_pixel.append(frame)
            continue

        contacts.append(
            {
                "clip": clip,
                "frame": frame,
                "image_x": float(pixel[0]),
                "image_y": float(pixel[1]),
            }
        )
    truth_by_frame = dict(zip(truth["frames"], truth["ends"], strict=True))

    resolved = striker.resolve_point_strikers(
        contacts,
        detections,
        homography_for_frame=lambda _frame: homography,
        projection_for_frame=projections.get,
        track_by_end=track_by_end,
        sided_rows_by_frame=sided_by_frame,
        pose_by_frame=pose_by_frame,
        fps=fps,
        box_scale=box_scale,
    )
    window = max(1, int(np.ceil(striker.CONTACT_WINDOW_SECONDS * fps)))
    rows = []
    for record in resolved:
        frame = record["frame"]
        legacy = nearest_sided_end(
            (record["image_x"], record["image_y"]), frame, sided_by_frame, window, 1.0
        )
        honest = nearest_sided_end(
            (record["image_x"], record["image_y"]), frame, sided_by_frame, window, sided_scale
        )
        rows.append(
            {
                "attempt": key,
                "match_id": match_id,
                "clip": clip,
                "frame": frame,
                "truth_end": truth_by_frame[frame],
                "truth_provenance": truth["provenance"],
                "legacy_half_native_end": legacy["end"],
                "sided_nearest_end": honest["end"],
                "resolver_end": record["end"],
                "resolver_status": record["status"],
                "confidence": record["confidence"],
                "reach": record["reach"],
                "reach_end_margin": record["reach_end_margin"],
                "end_source": record["end_source"],
                "scale_reference": record["scale_reference"],
                "camera_scale_ratio": record["camera_scale_ratio"],
                "alternation_applied": record["alternation_applied"],
                "disagrees_with_sided_track": record.get("disagrees_with_sided_track"),
                "wrist_px": record.get("wrist_px"),
                "root_px": record.get("root_px"),
                "wrist_box_heights": record.get("wrist_box_heights"),
                "root_box_heights": record.get("root_box_heights"),
                "pose_matched": record.get("pose_matched"),
                "abstain_reason": record.get("abstain_reason"),
            }
        )
    rows.extend(
        _blank_row(
            key,
            match_id,
            clip,
            frame,
            truth_by_frame[frame],
            truth["provenance"],
            "no_automatic_ball_pixel_at_contact",
        )
        for frame in missing_pixel
    )
    rows.sort(key=lambda row: row["frame"])
    return {
        "attempt": key,
        "status": "scored",
        "match_id": match_id,
        "clip": clip,
        "fps": fps,
        "box_artifact_size": [artifact_size.width, artifact_size.height],
        "box_scale_to_native": box_scale,
        "sided_artifact_present": bool(sided_paths),
        "pose_rows_for_clip": sum(len(value) for value in pose_by_frame.values()),
        "contacts_without_ball_pixel": missing_pixel,
        "contacts": rows,
    }


def _rate(hits: int, total: int) -> float | None:
    return None if not total else round(hits / total, 4)


def summarise(rows: list[dict]) -> dict:
    """Correct-end counts per arm, abstentions kept in the denominator."""
    total = len(rows)
    arms = ("legacy_half_native_end", "sided_nearest_end", "resolver_end")
    summary = {"contacts": total}
    for arm in arms:
        correct = sum(1 for row in rows if row[arm] == row["truth_end"])
        wrong = sum(1 for row in rows if row[arm] is not None and row[arm] != row["truth_end"])
        absent = sum(1 for row in rows if row[arm] is None)
        summary[arm.removesuffix("_end")] = {
            "correct": correct,
            "wrong_side": wrong,
            "no_answer": absent,
            "correct_rate": _rate(correct, total),
            "wrong_side_rate": _rate(wrong, total),
        }
    confident = [row for row in rows if row["resolver_status"] == "resolved"]
    summary["resolver_confident"] = {
        "contacts": len(confident),
        "correct": sum(1 for row in confident if row["resolver_end"] == row["truth_end"]),
        "correct_rate": _rate(
            sum(1 for row in confident if row["resolver_end"] == row["truth_end"]),
            len(confident),
        ),
    }
    summary["resolver_status"] = dict(Counter(row["resolver_status"] for row in rows))
    summary["abstain_reason"] = dict(
        Counter(row["abstain_reason"] for row in rows if row["abstain_reason"])
    )
    summary["scale_reference"] = dict(Counter(row["scale_reference"] or "none" for row in rows))
    summary["end_source"] = dict(Counter(row["end_source"] or "none" for row in rows))
    summary["alternation_applied"] = sum(1 for row in rows if row["alternation_applied"])
    summary["disagrees_with_sided_track"] = sum(
        1 for row in rows if row["disagrees_with_sided_track"]
    )
    for name, key in (("wrist_px", "wrist_px"), ("root_px", "root_px")):
        values = [row[key] for row in rows if isinstance(row[key], (int, float))]
        summary[f"{name}_native"] = (
            None
            if not values
            else {
                "n": len(values),
                "median": round(statistics.median(values), 2),
                "p90": round(sorted(values)[max(0, int(np.ceil(0.9 * len(values))) - 1)], 2),
                "max": round(max(values), 2),
            }
        )
    return summary


def normalize_stroke_type(value: str | None) -> str | None:
    """Collapse descriptive truth strings onto the automatic coarse classes."""
    text = str(value or "").lower()
    for kind in ("serve", "overhead", "volley", "forehand", "backhand"):
        if kind in text:
            return kind
    return None


def summarise_strokes(rows: list[dict]) -> dict:
    total = len(rows)
    answered = [row for row in rows if row.get("resolver_stroke") is not None]
    correct = sum(row.get("resolver_stroke") == row.get("truth_stroke") for row in answered)
    return {
        "contacts": total,
        "correct": correct,
        "wrong_type": len(answered) - correct,
        "no_answer": total - len(answered),
        "correct_rate": _rate(correct, total),
        "answered_precision": _rate(correct, len(answered)),
        "truth_types": dict(Counter(row.get("truth_stroke") or "unknown" for row in rows)),
        "resolved_types": dict(Counter(row.get("resolver_stroke") or "abstain" for row in rows)),
    }


def score_hitter_cases(cohort: Path, cases: list[dict]) -> dict:
    """Score arbitrary labelled hitter cases without feeding them into inference."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        grouped.setdefault((case["match_id"], case["clip"]), []).append(case)
    rows = []
    skipped = {}
    for (match_id, clip), clip_cases in sorted(grouped.items()):
        clip_cases.sort(key=lambda item: float(item["frame"]))
        try:
            inputs = striker.MatchInputs.resolve(cohort / match_id)
            contacts = [
                {
                    "clip": clip,
                    "frame": float(case["frame"]),
                    "image_x": case.get("image_x"),
                    "image_y": case.get("image_y"),
                }
                for case in clip_cases
            ]
            resolved = inputs.resolve_clip(clip, contacts)
        except (FileNotFoundError, ValueError, KeyError) as error:
            skipped[f"{match_id}__{clip}"] = str(error)
            resolved = [
                {
                    "frame": float(case["frame"]),
                    "end": None,
                    "status": "abstain",
                    "abstain_reason": "missing_automatic_inputs",
                    "stroke_type": None,
                    "stroke_status": "abstain_missing_automatic_inputs",
                }
                for case in clip_cases
            ]
        by_frame = {float(record["frame"]): record for record in resolved}
        for case in clip_cases:
            record = by_frame.get(float(case["frame"]), {})
            rows.append(
                {
                    "case_id": case["case_id"],
                    "match_id": match_id,
                    "clip": clip,
                    "frame": float(case["frame"]),
                    "truth_end": case.get("truth_end"),
                    "resolver_end": record.get("end"),
                    "resolver_status": record.get("status", "abstain"),
                    "truth_stroke": normalize_stroke_type(case.get("stroke_type")),
                    "resolver_stroke": normalize_stroke_type(record.get("stroke_type")),
                    "stroke_status": record.get("stroke_status"),
                    "stroke_contact_height_box": record.get("stroke_contact_height_box"),
                    "stroke_baseline_distance_m": record.get("stroke_baseline_distance_m"),
                    "stroke_side_margin_box_heights": record.get("stroke_side_margin_box_heights"),
                    "confidence": record.get("confidence", 0.0),
                    "abstain_reason": record.get("abstain_reason"),
                    "racket_face_native": record.get("racket_face_native"),
                    "racket_face_sigma_px": record.get("racket_face_sigma_px"),
                }
            )
    end_rows = [row for row in rows if row["truth_end"] in {"near", "far"}]
    end_correct = sum(row["resolver_end"] == row["truth_end"] for row in end_rows)
    end_wrong = sum(
        row["resolver_end"] is not None and row["resolver_end"] != row["truth_end"]
        for row in end_rows
    )
    return {
        "cases": len(rows),
        "broadcasts": len({row["match_id"] for row in rows}),
        "skipped": skipped,
        "end": {
            "truth_resolved": len(end_rows),
            "correct": end_correct,
            "wrong_side": end_wrong,
            "no_answer": len(end_rows) - end_correct - end_wrong,
            "correct_rate": _rate(end_correct, len(end_rows)),
        },
        "stroke": summarise_strokes([row for row in rows if row["truth_stroke"] is not None]),
        "rows": rows,
    }


def astralandmarks_hitter_cases() -> list[dict]:
    truth = json.loads(HITTER_TRUTH.read_text())
    sample = json.loads(HITTER_SAMPLE.read_text())
    source = {row["case_id"]: row for row in sample["cases"]}
    cases = []
    for row in truth["cases"]:
        sampled = source[row["case_id"]]
        location = sampled.get("emission_location_metadata") or {}
        cases.append(
            {
                "case_id": row["case_id"],
                "match_id": row["match_id"],
                "clip": row["clip"],
                "frame": float(row["emission_frame"]),
                "image_x": location.get("image_x"),
                "image_y": location.get("image_y"),
                "truth_end": row.get("hitter_end"),
                "stroke_type": row.get("stroke_type"),
            }
        )
    return cases


# --- The 40 owner-click broadcasts ----------------------------------------------------------
#
# The owner event truth gives an accepted contact frame and a click pixel, but it names no
# hitter, so no correct-hitter rate can be computed on it.  What it does measure at scale is
# the size of the coordinate defect and the resolver's coverage: how often the legacy
# half-native read of the sided artifact would name a different end than the contract-correct
# read, how often the resolver answers at all, and how far the resolved striker's wrist and
# root sit from a real contact click.


def owner_contact_clicks(path: Path) -> dict[str, dict[str, list[dict]]]:
    """Accepted owner contact clicks in native pixels, keyed by match and clip."""
    out: dict[str, dict[str, list[dict]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] != "contact" or row["verdict"] not in ACCEPTED:
                continue
            if not row["labeled_frame"] or not row["labeled_x540"]:
                continue
            match_id, _, clip = row["clip"].partition("__")
            out.setdefault(match_id, {}).setdefault(clip, []).append(
                {
                    "frame": float(row["labeled_frame"]),
                    "image_x": float(row["labeled_x540"]) * 2.0,
                    "image_y": float(row["labeled_y540"]) * 2.0,
                }
            )
    for clips in out.values():
        for rows in clips.values():
            rows.sort(key=lambda row: row["frame"])
    return out


def score_owner_clicks(cohort: Path, truth_path: Path) -> dict:
    clicks = owner_contact_clicks(truth_path)
    rows, skipped = [], {}
    for match_id in sorted(clicks):
        match_dir = cohort / match_id
        if not match_dir.is_dir():
            skipped[match_id] = "no_automatic_match_directory"
            continue
        try:
            inputs = striker.MatchInputs.resolve(match_dir)
        except FileNotFoundError as error:
            skipped[match_id] = str(error)
            continue
        for clip in sorted(clicks[match_id]):
            contacts = clicks[match_id][clip]
            resolved = inputs.resolve_clip(clip, contacts)
            legacy = _legacy_ends(inputs, clip, contacts)
            for record in resolved:
                rows.append(
                    {
                        "match_id": match_id,
                        "clip": clip,
                        "frame": record["frame"],
                        "resolver_end": record.get("end"),
                        "resolver_status": record["status"],
                        "confidence": record.get("confidence"),
                        "reach": record.get("reach"),
                        "sided_nearest_end": record.get("sided_track_end"),
                        "legacy_half_native_end": legacy.get(record["frame"]),
                        "alternation_applied": record.get("alternation_applied", False),
                        "wrist_px": record.get("wrist_px"),
                        "root_px": record.get("root_px"),
                        "abstain_reason": record.get("abstain_reason"),
                    }
                )
    both = [row for row in rows if row["sided_nearest_end"] and row["legacy_half_native_end"]]
    answered = [row for row in rows if row["resolver_end"]]
    return {
        "contacts": len(rows),
        "broadcasts": len({row["match_id"] for row in rows}),
        "skipped_broadcasts": skipped,
        "resolver_answered": len(answered),
        "resolver_answered_rate": _rate(len(answered), len(rows)),
        "resolver_status": dict(Counter(row["resolver_status"] for row in rows)),
        "abstain_reason": dict(
            Counter(row["abstain_reason"] for row in rows if row["abstain_reason"])
        ),
        "legacy_versus_contract_end": {
            "comparable_contacts": len(both),
            "different_end": sum(
                1 for row in both if row["legacy_half_native_end"] != row["sided_nearest_end"]
            ),
            "different_end_rate": _rate(
                sum(1 for row in both if row["legacy_half_native_end"] != row["sided_nearest_end"]),
                len(both),
            ),
        },
        "resolver_versus_sided_track": {
            "comparable_contacts": sum(1 for row in answered if row["sided_nearest_end"]),
            "different_end": sum(
                1
                for row in answered
                if row["sided_nearest_end"] and row["sided_nearest_end"] != row["resolver_end"]
            ),
        },
        "alternation_applied": sum(1 for row in rows if row["alternation_applied"]),
        "reach_box_heights": _distribution([row["reach"] for row in answered]),
        "wrist_px_native": _distribution(
            [row["wrist_px"] for row in rows if isinstance(row["wrist_px"], (int, float))]
        ),
        "root_px_native": _distribution(
            [row["root_px"] for row in rows if isinstance(row["root_px"], (int, float))]
        ),
    }


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "p90": round(ordered[max(0, int(np.ceil(0.9 * len(ordered))) - 1)], 3),
        "p99": round(ordered[max(0, int(np.ceil(0.99 * len(ordered))) - 1)], 3),
        "max": round(ordered[-1], 3),
    }


def _legacy_ends(inputs, clip: str, contacts: list[dict]) -> dict[float, str | None]:
    """What the sided artifact says when its declared space is ignored."""
    if inputs.sided is None:
        return {}
    sided_by_frame: dict[int, list[dict]] = {}
    for row in striker.clip_rows(inputs.sided, clip):
        sided_by_frame.setdefault(striker.frame_number(row["frame"]), []).append(row)
    window = max(1, int(np.ceil(striker.CONTACT_WINDOW_SECONDS * inputs.fps)))
    return {
        float(contact["frame"]): striker._nearest_sided_end(
            (float(contact["image_x"]), float(contact["image_y"])),
            float(contact["frame"]),
            sided_by_frame,
            window,
            1.0,
        )["end"]
        for contact in contacts
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--owner-clicks",
        action="store_true",
        help="also probe the owner contact clicks on the 40 truth broadcasts; they name no "
        "hitter, so this measures coverage, reach and the coordinate defect's size only",
    )
    parser.add_argument(
        "--hitter-cases",
        action="store_true",
        help="also score the 40 clip-disjoint astralandmarks hitter cases",
    )
    args = parser.parse_args()
    root = data_root()
    cohort = args.cohort or (root / DEFAULT_COHORT)
    selection = json.loads((root / SELECTION).read_text())["attempts"]
    truth = truth_ends(selection)

    attempts, rows = [], []
    for entry in selection:
        key = entry["key"]
        if key not in truth:
            attempts.append({"attempt": key, "status": "abstain", "reason": "no_truth_end"})
            continue
        report = score_attempt(key, Path(entry["path"]), cohort, truth[key])
        attempts.append({k: v for k, v in report.items() if k != "contacts"})
        rows.extend(report.get("contacts", []))
    for row in rows:
        row["scorable"] = row["abstain_reason"] not in {
            "no_automatic_court_geometry",
            "no_automatic_player_boxes",
            "no_automatic_ball_pixel_at_contact",
        }

    labelled = [row for row in rows if row["truth_provenance"] == "labelled_hitter"]
    derived = [row for row in rows if row["truth_provenance"] != "labelled_hitter"]
    payload = {
        "schema": SCHEMA,
        "denominator": (
            "every labelled contact of every selected attempt, including the contacts whose "
            "automatic court geometry or ball pixel is missing upstream"
        ),
        "automatic_inference_eligible": False,
        "opened_development_data": True,
        "supplied_privileges": [
            "contact times come from the frozen labels; S5 event recall is measured elsewhere",
            "truth hitter ends come from the labels or the frozen review narratives",
        ],
        "cohort": str(cohort),
        "attempts": attempts,
        "labelled_truth": summarise(labelled),
        "labelled_truth_with_automatic_inputs": summarise(
            [row for row in labelled if row["scorable"]]
        ),
        "derived_truth": {
            **summarise(derived),
            "caveat": (
                "ends derived from the review narrative's alternating shot sequence; "
                "circular with the resolver's alternation prior, reported separately"
            ),
        },
        "contacts": rows,
    }
    if args.owner_clicks:
        payload["owner_click_probe"] = score_owner_clicks(cohort, authoritative_truth_path())
        payload["owner_click_probe"]["limits"] = (
            "owner clicks carry no hitter identity; no correct-hitter rate is claimed here"
        )
    if args.hitter_cases:
        payload["hitter_cases"] = score_hitter_cases(cohort, astralandmarks_hitter_cases())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["labelled_truth"], indent=2))
    print(json.dumps(payload["labelled_truth_with_automatic_inputs"], indent=2))
    print(json.dumps(payload["derived_truth"], indent=2))
    if args.owner_clicks:
        probe = {
            key: value
            for key, value in payload["owner_click_probe"].items()
            if key != "skipped_broadcasts"
        }
        print(json.dumps(probe, indent=2))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
