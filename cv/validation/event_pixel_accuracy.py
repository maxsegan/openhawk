"""Measure emitted event pixels against owner event clicks.

This is evaluation-only code.  Owner truth is joined after the frozen event
artifacts are loaded and is never written into an automatic-inference root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cv.validation.current_standard_event_truth import authoritative_truth_path
from cv.validation.score_cross_match_event_labels_v5 import ACCEPTED, EVENT_TYPES, match_events
from cv.validation.s6_cohort_v2_eval import HELD_OUT_BROADCASTS

SCHEMA = "event_pixel_accuracy_v1"
FATE_SCHEMA = "event_pixel_flight_fate_v1"
RECOMMENDATION_SCHEMA = "event_pixel_anchor_recommendation_v1"
EXPECTED_TRUTH_EVENTS = 1_415
NET_Y_M = 11.885
TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"
CROP_CANDIDATE_GLOB = "ball_candidates_*_native1080_branched_crop_v2.csv"

# Conservative p90 radii: the next even pixel at or above the worst pass-2
# development, cohort-v2 runtime, or cohort-v2 held-out p90 for the emitted-frame track.
# Net hits lie on the dividing plane, so their observed far/net value is also
# the fallback for the empty near cell.
TRACK_P90_UNCERTAINTY_PX = {
    ("contact", "far"): 16.0,
    ("contact", "near"): 60.0,
    ("bounce", "far"): 10.0,
    ("bounce", "near"): 30.0,
    ("net_hit", "far"): 20.0,
    ("net_hit", "near"): 20.0,
}


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_number(value: str | int | float) -> int:
    if isinstance(value, str) and value.startswith("f_"):
        return int(Path(value).stem.rsplit("_", 1)[-1])
    return int(float(value))


def owner_image_frame(frame: float) -> int:
    """Return the native image carrying an owner event click.

    Half-frame labels are the owner's -0.5-frame leading-blur convention.  The
    click was made on the following native image, as documented and used by the
    oracle Stage-6 adapter.
    """

    return int(math.ceil(frame))


def canonical_fps(value: float) -> str:
    if 59.8 < value < 59.98:
        return "59.94"
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 0.01 else f"{value:.3f}"


def _native_columns(path: Path, columns: list[str]) -> tuple[str, str, float, float]:
    sidecar = path.with_name(f"{path.name}.coordinates.json")
    sidecar_exists = sidecar.exists()
    document = json.loads(sidecar.read_text()) if sidecar_exists else {}
    declared = document.get("coordinate_columns", {}).get("native_1920x1080")
    if declared and len(declared) == 2 and all(column in columns for column in declared):
        return str(declared[0]), str(declared[1]), 1.0, 1.0
    if "x_native" in columns and "y_native" in columns:
        return "x_native", "y_native", 1.0, 1.0
    if "x" not in columns or "y" not in columns:
        raise ValueError(f"coordinate columns are absent from {path}")
    artifact = document.get("artifact_size") or {}
    image = document.get("image_size") or {}
    # Historical candidate CSVs used x/y in the detector's 960x540 space and
    # did not receive sidecars.  The regenerated contract adds x_native/y_native.
    default_width, default_height = (1920.0, 1080.0) if sidecar_exists else (960.0, 540.0)
    width = float(artifact.get("width") or image.get("width") or default_width)
    height = float(artifact.get("height") or image.get("height") or default_height)
    image_width = float(image.get("width") or 1920.0)
    image_height = float(image.get("height") or 1080.0)
    return "x", "y", image_width / width, image_height / height


def load_track(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        x_column, y_column, scale_x, scale_y = _native_columns(path, columns)
        return {
            (str(row["clip"]), frame_number(row["frame"])): (
                float(row[x_column]) * scale_x,
                float(row[y_column]) * scale_y,
            )
            for row in reader
            if row.get(x_column) not in {None, ""} and row.get(y_column) not in {None, ""}
        }


def load_crop_candidates(match_root: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(match_root.glob(CROP_CANDIDATE_GLOB)):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            x_column, y_column, scale_x, scale_y = _native_columns(path, columns)
            for row in reader:
                if row.get(x_column) in {None, ""} or row.get(y_column) in {None, ""}:
                    continue
                output[(str(row["clip"]), frame_number(row["frame"]))].append(
                    {
                        "x": float(row[x_column]) * scale_x,
                        "y": float(row[y_column]) * scale_y,
                        "score": float(row.get("score") or 0.0),
                        "rank": int(row.get("rank") or 0),
                        "detector": path.name.split("_native1080", 1)[0].removeprefix(
                            "ball_candidates_"
                        ),
                    }
                )
    return dict(output)


def best_crop_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows = list(candidates)
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row["score"]), -int(row["rank"])))


def _load_truth() -> list[dict[str, Any]]:
    truth = []
    with authoritative_truth_path().open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in EVENT_TYPES or row["verdict"] not in ACCEPTED:
                continue
            frame_text = row["labeled_frame"] or row["seed_frame"]
            if not frame_text or not row["labeled_x540"] or not row["labeled_y540"]:
                raise ValueError(f"accepted owner event lacks time/location: {row}")
            match_id, clip = str(row["clip"]).rsplit("__", 1)
            truth.append(
                {
                    "clip": str(row["clip"]),
                    "local_clip": clip,
                    "match_id": match_id,
                    "event_type": str(row["event_type"]),
                    "frame": float(frame_text),
                    "owner_x": 2.0 * float(row["labeled_x540"]),
                    "owner_y": 2.0 * float(row["labeled_y540"]),
                    "seed_id": str(row["seed_id"]),
                    "note": str(row.get("note") or ""),
                }
            )
    truth.sort(key=lambda row: (row["clip"], row["frame"], row["event_type"]))
    if len(truth) != EXPECTED_TRUTH_EVENTS:
        raise ValueError(f"expected {EXPECTED_TRUTH_EVENTS} owner events, found {len(truth)}")
    return truth


def _load_emissions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = (
        payload.get("emissions", payload.get("rows", [])) if isinstance(payload, dict) else payload
    )
    output = []
    for row in rows:
        if row.get("event_type") not in EVENT_TYPES or row.get("abstain") is True:
            continue
        canonical = dict(row)
        canonical["frame"] = float(row["frame"])
        if "__" not in str(canonical["clip"]):
            canonical["clip"] = f"{row['match_id']}__{row['clip']}"
        output.append(canonical)
    return output


def _load_homographies(path: Path) -> dict[tuple[str, int], np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {
            (str(clip), int(frame)): np.asarray(matrix, dtype=float)
            for clip, frame, matrix in zip(
                payload["clips"], payload["frames"], payload["H"], strict=True
            )
        }


def _court_side(homography: np.ndarray | None, pixel: tuple[float, float]) -> str:
    if homography is None:
        return "unknown"
    projected = homography @ np.asarray([pixel[0], pixel[1], 1.0], dtype=float)
    if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-9:
        return "unknown"
    court_y = float(projected[1] / projected[2])
    return "near" if court_y < NET_Y_M else "far"


def _distance(point: tuple[float, float] | None, owner: tuple[float, float]) -> float | None:
    if point is None:
        return None
    return math.hypot(point[0] - owner[0], point[1] - owner[1])


def select_event_anchor_pixel(
    *,
    event_type: str,
    court_side: str,
    head: tuple[float, float] | None,
    track: tuple[float, float] | None,
) -> dict[str, Any]:
    """Choose a validation-derived event pixel for a fitter owner to adopt.

    The current emitted-frame track is primary and the x/y head is a coverage
    fallback.  Raw crop peaks are deliberately excluded: their measured p90 is
    119 px overall and consensus-gated recentering did not beat the track.
    When both witnesses exist, their disagreement inflates rather than moves
    the track-centred anchor.
    """

    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event type: {event_type}")
    side = court_side if court_side in {"near", "far"} else "near"
    base = TRACK_P90_UNCERTAINTY_PX[(event_type, side)]
    if track is not None:
        disagreement = _distance(head, track) if head is not None else None
        return {
            "x": float(track[0]),
            "y": float(track[1]),
            "source": "automatic_track_at_emitted_frame",
            "uncertainty_radius_px_p90": max(base, disagreement or 0.0),
            "base_uncertainty_radius_px_p90": base,
            "head_track_disagreement_px": disagreement,
            "abstain": False,
            "reason": "track_primary_head_disagreement_inflates_uncertainty",
        }
    if head is not None:
        return {
            "x": float(head[0]),
            "y": float(head[1]),
            "source": "event_xy_head_track_missing",
            "uncertainty_radius_px_p90": base,
            "base_uncertainty_radius_px_p90": base,
            "head_track_disagreement_px": None,
            "abstain": False,
            "reason": "head_coverage_fallback",
        }
    return {
        "x": None,
        "y": None,
        "source": "none",
        "uncertainty_radius_px_p90": None,
        "base_uncertainty_radius_px_p90": base,
        "head_track_disagreement_px": None,
        "abstain": True,
        "reason": "head_and_track_missing",
    }


def consensus_crop_recenter(
    *,
    head: tuple[float, float] | None,
    track: tuple[float, float] | None,
    crop: tuple[float, float] | None,
    crop_track_gate_px: float = 4.0,
) -> tuple[tuple[float, float] | None, bool]:
    """Cheap crop-peak negative control with a strict two-witness gate."""

    baseline = track if track is not None else head
    if head is None or track is None or crop is None:
        return baseline, False
    crop_track = _distance(crop, track)
    crop_head = _distance(crop, head)
    if (
        crop_track is not None
        and crop_head is not None
        and crop_track <= crop_track_gate_px
        and crop_head <= 2.0 * crop_track_gate_px
    ):
        return crop, True
    return baseline, False


def _signed_offsets(
    point: tuple[float, float] | None,
    owner: tuple[float, float],
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    if point is None or before is None or after is None:
        return None, None
    velocity = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    norm = float(np.linalg.norm(velocity))
    if norm < 1e-6:
        return None, None
    direction = velocity / norm
    normal = np.asarray([-direction[1], direction[0]])
    delta = np.asarray(point, dtype=float) - np.asarray(owner, dtype=float)
    return float(delta @ direction), float(delta @ normal)


def _source_point(row: dict[str, Any], prefix: str) -> tuple[float, float] | None:
    x, y = row.get(f"{prefix}_x"), row.get(f"{prefix}_y")
    if x in {None, ""} or y in {None, ""}:
        return None
    return float(x), float(y)


def evaluate_artifact(
    *,
    name: str,
    emissions_path: Path,
    cohort_root: Path,
    truth: list[dict[str, Any]],
    classification_root: Path | None = None,
) -> list[dict[str, Any]]:
    emissions = _load_emissions(emissions_path)
    matches, _, _ = match_events(emissions, truth, 3.0)
    match_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for emission, target in matches:
        match_id = str(target["match_id"])
        if match_id not in match_cache:
            root = cohort_root / match_id
            geometry_root = (classification_root or cohort_root) / match_id
            match_cache[match_id] = {
                "track": load_track(root / TRACK_NAME),
                "candidates": load_crop_candidates(root),
                "homographies": _load_homographies(geometry_root / "court_H_per_frame_v1.npz"),
            }
        cached = match_cache[match_id]
        track = cached["track"]
        local_clip = str(target["local_clip"])
        emitted_frame = int(round(float(emission["frame"])))
        owner_frame = owner_image_frame(float(target["frame"]))
        owner = float(target["owner_x"]), float(target["owner_y"])
        location = emission.get("location") or {}
        head = None
        if location.get("image_x") is not None and location.get("image_y") is not None:
            head = float(location["image_x"]), float(location["image_y"])
        emitted_track = track.get((local_clip, emitted_frame))
        owner_track = track.get((local_clip, owner_frame))
        candidate = best_crop_candidate(cached["candidates"].get((local_clip, emitted_frame), []))
        crop = (float(candidate["x"]), float(candidate["y"])) if candidate else None
        before = track.get((local_clip, owner_frame - 1))
        after = track.get((local_clip, owner_frame + 1))
        homography = cached["homographies"].get((local_clip, owner_frame))
        output: dict[str, Any] = {
            "artifact": name,
            "clip": target["clip"],
            "match_id": match_id,
            "local_clip": local_clip,
            "seed_id": target["seed_id"],
            "event_type": target["event_type"],
            "truth_frame": target["frame"],
            "owner_image_frame": owner_frame,
            "emission_frame": float(emission["frame"]),
            "frame_error": float(emission["frame"]) - float(target["frame"]),
            "fps": float(location.get("fps") or emission.get("fps") or 25.0),
            "fps_group": canonical_fps(float(location.get("fps") or emission.get("fps") or 25.0)),
            "broadcast_split": (
                "held_out" if match_id in HELD_OUT_BROADCASTS else "seen_development"
            ),
            "court_side": _court_side(homography, owner),
            "owner_x": owner[0],
            "owner_y": owner[1],
            "head_x": head[0] if head else None,
            "head_y": head[1] if head else None,
            "track_emitted_x": emitted_track[0] if emitted_track else None,
            "track_emitted_y": emitted_track[1] if emitted_track else None,
            "track_owner_x": owner_track[0] if owner_track else None,
            "track_owner_y": owner_track[1] if owner_track else None,
            "crop_emitted_x": crop[0] if crop else None,
            "crop_emitted_y": crop[1] if crop else None,
            "crop_detector": candidate["detector"] if candidate else None,
            "crop_score": candidate["score"] if candidate else None,
            "crop_rank": candidate["rank"] if candidate else None,
            "owner_note": target["note"],
        }
        for prefix in ("head", "track_emitted", "track_owner", "crop_emitted"):
            point = _source_point(output, prefix)
            output[f"{prefix}_error_px"] = _distance(point, owner)
            along, lateral = _signed_offsets(point, owner, before, after)
            output[f"{prefix}_along_motion_px"] = along
            output[f"{prefix}_lateral_motion_px"] = lateral
        rows.append(output)
    return sorted(rows, key=lambda row: (row["clip"], row["truth_frame"], row["event_type"]))


def percentile(values: Iterable[float | None], q: float) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = ("head", "track_emitted", "track_owner", "crop_emitted")
    slices: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for dimension in ("event_type", "court_side", "fps_group", "broadcast_split"):
        for value in sorted({str(row[dimension]) for row in rows}):
            slices.append((dimension, value, [row for row in rows if str(row[dimension]) == value]))
    output = []
    for dimension, value, selected in slices:
        for source in sources:
            errors = [row[f"{source}_error_px"] for row in selected]
            output.append(
                {
                    "slice_dimension": dimension,
                    "slice_value": value,
                    "source": source,
                    "matched_events": len(selected),
                    "available": sum(error is not None for error in errors),
                    "median_px": percentile(errors, 50.0),
                    "p90_px": percentile(errors, 90.0),
                    "median_along_motion_px": percentile(
                        (row[f"{source}_along_motion_px"] for row in selected), 50.0
                    ),
                    "median_lateral_motion_px": percentile(
                        (row[f"{source}_lateral_motion_px"] for row in selected), 50.0
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_pixel_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "truth_frame",
        "emission_frame",
        "head_error_px",
        "track_emitted_error_px",
        "track_owner_error_px",
        "crop_emitted_error_px",
    }
    output = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in numeric:
                row[key] = float(raw[key]) if raw.get(key) not in {None, ""} else None
            output.append(row)
    return output


def _nearest_contact(
    rows: list[dict[str, Any]], frame: float, frame_field: str, tolerance: float
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["event_type"] == "contact"
        and row.get(frame_field) is not None
        and abs(float(row[frame_field]) - frame) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(float(row[frame_field]) - frame))


def link_flight_fates(
    *,
    scope: str,
    ledger_path: Path,
    pixel_rows: list[dict[str, Any]],
    pixel_artifact: str,
    boundary_field: str,
    boundary_tolerance: float,
) -> list[dict[str, Any]]:
    ledger = json.loads(ledger_path.read_text())
    flights = ledger.get("rows", ledger)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pixel_rows:
        if row["artifact"] == pixel_artifact and row["event_type"] == "contact":
            by_clip[str(row["clip"])].append(row)
    output = []
    sources = ("head", "track_emitted", "crop_emitted")
    for flight in flights:
        clip = str(flight["point"])
        contacts: list[dict[str, Any]] = []
        for boundary in ("start", "end"):
            if boundary == "end" and bool(flight.get("terminal_end")):
                continue
            frame = float(flight[f"{boundary}_frame"])
            contact = _nearest_contact(
                by_clip.get(clip, []), frame, boundary_field, boundary_tolerance
            )
            if contact is not None:
                contacts.append({"boundary": boundary, **contact})
        reasons = set(flight.get("reasons") or [])
        row: dict[str, Any] = {
            "scope": scope,
            "flight_id": flight["flight_id"],
            "point": clip,
            "match_id": flight["match_id"],
            "flight_index": flight["flight_index"],
            "status": flight["status"],
            "solved": bool(flight["solved"]),
            "accepted": flight["status"] == "provisional_valid",
            "reach_rejected": "physics_contact_reach" in reasons,
            "anchor_violated": "anchor_violation" in reasons,
            "contact_matches": len(contacts),
            "contact_seed_ids": ";".join(str(contact["seed_id"]) for contact in contacts),
            "reasons": ";".join(sorted(reasons)),
        }
        for source in sources:
            errors = [
                float(contact[f"{source}_error_px"])
                for contact in contacts
                if contact.get(f"{source}_error_px") is not None
            ]
            row[f"{source}_contact_count"] = len(errors)
            row[f"{source}_median_error_px"] = percentile(errors, 50.0)
            row[f"{source}_max_error_px"] = max(errors) if errors else None
        output.append(row)
    return output


def summarize_fates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = {
        "all_attempted": lambda row: True,
        "all_solved": lambda row: row["solved"],
        "accepted": lambda row: row["accepted"],
        "reach_rejected": lambda row: row["reach_rejected"],
        "anchor_violated": lambda row: row["anchor_violated"],
        "reach_and_anchor": lambda row: row["reach_rejected"] and row["anchor_violated"],
        "other_solved_rejection": lambda row: (
            row["solved"]
            and not row["accepted"]
            and not row["reach_rejected"]
            and not row["anchor_violated"]
        ),
    }
    output = []
    for fate, selector in selectors.items():
        selected = [row for row in rows if selector(row)]
        linked = [row for row in selected if row["head_contact_count"]]
        for source in ("head", "track_emitted", "crop_emitted"):
            available = [row for row in linked if row[f"{source}_max_error_px"] is not None]
            errors = [row[f"{source}_max_error_px"] for row in available]
            output.append(
                {
                    "scope": rows[0]["scope"] if rows else "unknown",
                    "fate": fate,
                    "source": source,
                    "flights": len(selected),
                    "linked_flights": len(linked),
                    "available_flights": len(available),
                    "contact_samples": sum(row[f"{source}_contact_count"] for row in available),
                    "flight_max_error_median_px": percentile(errors, 50.0),
                    "flight_max_error_p90_px": percentile(errors, 90.0),
                    "over_12px": sum(float(error) > 12.0 for error in errors),
                    "over_12px_fraction": (
                        sum(float(error) > 12.0 for error in errors) / len(errors)
                        if errors
                        else None
                    ),
                    "over_24px": sum(float(error) > 24.0 for error in errors),
                    "over_24px_fraction": (
                        sum(float(error) > 24.0 for error in errors) / len(errors)
                        if errors
                        else None
                    ),
                }
            )
    return output


def run_step2(step1: Path, output: Path) -> dict[str, Any]:
    root = data_root()
    pixel_rows = read_pixel_rows(step1 / "event_pixel_rows.csv")
    specifications = {
        "oracle_arm_d": {
            "ledger": root / "processed/wk3_s6fix/step4_camera_lineage/arm_d/ledger.json",
            "pixel_artifact": "pass2_lobo_development",
            "boundary_field": "truth_frame",
            "boundary_tolerance": 0.51,
            "interpretation": (
                "counterfactual automatic pass-2 head error at Arm-D owner-truth boundaries"
            ),
        },
        "cohort_v2_automatic": {
            "ledger": root / "processed/wk3_s6v2/default/ledger.json",
            "pixel_artifact": "cohort_v2_runtime",
            "boundary_field": "emission_frame",
            "boundary_tolerance": 0.01,
            "interpretation": "runtime head error at the automatic flight boundaries",
        },
    }
    rows = []
    artifacts = {}
    for scope, spec in specifications.items():
        linked = link_flight_fates(
            scope=scope,
            ledger_path=spec["ledger"],
            pixel_rows=pixel_rows,
            pixel_artifact=spec["pixel_artifact"],
            boundary_field=spec["boundary_field"],
            boundary_tolerance=spec["boundary_tolerance"],
        )
        rows.extend(linked)
        artifacts[scope] = {
            "ledger": str(spec["ledger"]),
            "ledger_sha256": sha256(spec["ledger"]),
            "flights": len(linked),
            "flights_with_matched_contact": sum(bool(row["contact_matches"]) for row in linked),
            "interpretation": spec["interpretation"],
        }
    summaries = []
    for scope in specifications:
        summaries.extend(summarize_fates([row for row in rows if row["scope"] == scope]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "flight_contact_rows.csv", rows)
    write_csv(output / "flight_fate_summary.csv", summaries)
    report = {
        "schema": FATE_SCHEMA,
        "status": "development_association_not_causal_intervention",
        "pixel_rows": str(step1 / "event_pixel_rows.csv"),
        "artifacts": artifacts,
        "fate_definitions": {
            "accepted": "ledger status == provisional_valid (the automatic pixel/physics gate)",
            "reach_rejected": "reasons contains physics_contact_reach",
            "anchor_violated": "reasons contains anchor_violation",
            "overlap": "reach_rejected and anchor_violated are non-exclusive",
        },
        "error_definition": (
            "maximum owner-click distance among non-terminal start/end contacts matched to each "
            "flight; tables retain 12 px and 24 px exceedance rates"
        ),
        "summary": summaries,
        "tables": {
            "per_flight": str(output / "flight_contact_rows.csv"),
            "summary": str(output / "flight_fate_summary.csv"),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _selector_rows(pixel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in pixel_rows:
        head = _source_point(row, "head")
        track = _source_point(row, "track_emitted")
        crop = _source_point(row, "crop_emitted")
        selected = select_event_anchor_pixel(
            event_type=str(row["event_type"]),
            court_side=str(row["court_side"]),
            head=head,
            track=track,
        )
        point = None if selected["abstain"] else (float(selected["x"]), float(selected["y"]))
        owner = float(row["owner_x"]), float(row["owner_y"])
        recentered, crop_adopted = consensus_crop_recenter(head=head, track=track, crop=crop)
        output.append(
            {
                **row,
                "selected_x": selected["x"],
                "selected_y": selected["y"],
                "selected_source": selected["source"],
                "selected_error_px": _distance(point, owner),
                "selected_uncertainty_radius_px_p90": selected["uncertainty_radius_px_p90"],
                "selected_base_uncertainty_radius_px_p90": selected[
                    "base_uncertainty_radius_px_p90"
                ],
                "head_track_disagreement_px": selected["head_track_disagreement_px"],
                "selected_abstain": selected["abstain"],
                "crop_recenter_x": recentered[0] if recentered else None,
                "crop_recenter_y": recentered[1] if recentered else None,
                "crop_recenter_error_px": _distance(recentered, owner),
                "crop_recenter_adopted": crop_adopted,
            }
        )
    return output


def _recommendation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    slices: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for dimension in ("event_type", "court_side", "fps_group", "broadcast_split"):
        for value in sorted({str(row[dimension]) for row in rows}):
            slices.append((dimension, value, [row for row in rows if str(row[dimension]) == value]))
    for dimension, value, selected in slices:
        for source in (
            "head",
            "track_emitted",
            "crop_emitted",
            "crop_recenter",
            "selected",
        ):
            errors = [row.get(f"{source}_error_px") for row in selected]
            available = [error for error in errors if error is not None]
            output.append(
                {
                    "slice_dimension": dimension,
                    "slice_value": value,
                    "source": source,
                    "matched_events": len(selected),
                    "available": len(available),
                    "median_px": percentile(available, 50.0),
                    "p90_px": percentile(available, 90.0),
                    "over_24px": sum(float(error) > 24.0 for error in available),
                    "over_24px_fraction": (
                        sum(float(error) > 24.0 for error in available) / len(available)
                        if available
                        else None
                    ),
                }
            )
    return output


def _uncertainty_rows(pixel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for event_type in sorted(EVENT_TYPES):
        for court_side in ("far", "near"):
            row: dict[str, Any] = {
                "event_type": event_type,
                "court_side": court_side,
                "recommended_track_p90_radius_px": TRACK_P90_UNCERTAINTY_PX[
                    (event_type, court_side)
                ],
            }
            for artifact, label in (
                ("pass2_lobo_development", "pass2_development"),
                ("cohort_v2_runtime", "cohort_v2_all"),
            ):
                selected = [
                    item
                    for item in pixel_rows
                    if item["artifact"] == artifact
                    and item["event_type"] == event_type
                    and item["court_side"] == court_side
                    and item.get("track_emitted_error_px") is not None
                ]
                row[f"{label}_n"] = len(selected)
                row[f"{label}_median_px"] = percentile(
                    (item["track_emitted_error_px"] for item in selected), 50.0
                )
                row[f"{label}_p90_px"] = percentile(
                    (item["track_emitted_error_px"] for item in selected), 90.0
                )
            held = [
                item
                for item in pixel_rows
                if item["artifact"] == "cohort_v2_runtime"
                and item["event_type"] == event_type
                and item["court_side"] == court_side
                and item["broadcast_split"] == "held_out"
                and item.get("track_emitted_error_px") is not None
            ]
            row["cohort_v2_held_out_n"] = len(held)
            row["cohort_v2_held_out_median_px"] = percentile(
                (item["track_emitted_error_px"] for item in held), 50.0
            )
            row["cohort_v2_held_out_p90_px"] = percentile(
                (item["track_emitted_error_px"] for item in held), 90.0
            )
            output.append(row)
    return output


def run_step3(step1: Path, output: Path) -> dict[str, Any]:
    pixel_rows = read_pixel_rows(step1 / "event_pixel_rows.csv")
    selected_rows = _selector_rows(pixel_rows)
    summaries = []
    for artifact in ("pass2_lobo_development", "cohort_v2_runtime"):
        for row in _recommendation_summary(
            [item for item in selected_rows if item["artifact"] == artifact]
        ):
            summaries.append({"artifact": artifact, **row})
    uncertainties = _uncertainty_rows(pixel_rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "selected_event_pixels.csv", selected_rows)
    write_csv(output / "source_comparison.csv", summaries)
    write_csv(output / "recommended_uncertainty.csv", uncertainties)
    report = {
        "schema": RECOMMENDATION_SCHEMA,
        "status": "validation_side_candidate_for_fitter_owner",
        "recommendation": (
            "Use the automatic arc track at the emitted frame; fall back to the x/y head only "
            "when the track is missing. Do not use an ungated detector crop peak. Keep the "
            "anchor on the selected pixel and inflate its p90 radius by head-track disagreement."
        ),
        "uncertainty_basis": (
            "next even pixel at or above the worst pass-2-development, cohort-v2-runtime, or "
            "cohort-v2 held-out emitted-track p90 in each event-type/court-side cell"
        ),
        "crop_recenter_negative_control": (
            "use a crop peak only within 4 px of track and 8 px of head; retained only as a "
            "measured negative control because it does not beat the track-primary selector"
        ),
        "uncertainty": uncertainties,
        "summary": summaries,
        "tables": {
            "selected_events": str(output / "selected_event_pixels.csv"),
            "source_comparison": str(output / "source_comparison.csv"),
            "recommended_uncertainty": str(output / "recommended_uncertainty.csv"),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_step1(output: Path) -> dict[str, Any]:
    root = data_root()
    truth = _load_truth()
    specifications = {
        "cohort_v2_runtime": (
            root / "processed/wk3_events/cohort_v2_emissions_fixed.json",
            root / "processed/wk3_cohort/cohort_root_v2",
        ),
        "pass2_lobo_development": (
            root / "processed/wk2_events/lobo_v2/emissions.json",
            root / "processed/wk1_s5/cohort_root",
        ),
    }
    rows = []
    artifacts = {}
    for name, (emissions_path, cohort_root) in specifications.items():
        emissions = _load_emissions(emissions_path)
        scoped_matches = {str(row["clip"]).rsplit("__", 1)[0] for row in emissions}
        evaluated = evaluate_artifact(
            name=name,
            emissions_path=emissions_path,
            cohort_root=cohort_root,
            truth=truth,
            classification_root=root / "processed/wk3_cohort/cohort_root_v2",
        )
        rows.extend(evaluated)
        artifacts[name] = {
            "emissions": str(emissions_path),
            "emissions_sha256": sha256(emissions_path),
            "cohort_root": str(cohort_root),
            "non_abstained_physical_emissions": len(emissions),
            "truth_events_in_emission_broadcast_scope": sum(
                row["match_id"] in scoped_matches for row in truth
            ),
            "matched_events": len(evaluated),
        }
    summaries = []
    for name in specifications:
        selected = [row for row in rows if row["artifact"] == name]
        for row in summarize(selected):
            summaries.append({"artifact": name, **row})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "event_pixel_rows.csv", rows)
    write_csv(output / "event_pixel_summary.csv", summaries)
    report = {
        "schema": SCHEMA,
        "status": "development_evaluation_owner_truth_joined_after_frozen_inference",
        "truth": {
            "path": str(authoritative_truth_path()),
            "sha256": sha256(authoritative_truth_path()),
            "events": len(truth),
            "coordinate_conversion": "owner px540 multiplied exactly by 2",
            "click_convention": (
                "motion-blurred balls use the leading edge; half-frame labels are a -0.5-frame "
                "timing convention and the click belongs to ceil(labeled_frame)"
            ),
        },
        "matching": {
            "tolerance_frames": 3.0,
            "same_event_type_required": True,
            "abstained_emissions_excluded": True,
        },
        "sources": {
            "head": "emission location.image_x/image_y",
            "track_emitted": f"{TRACK_NAME} at round(emission.frame)",
            "track_owner": f"{TRACK_NAME} at ceil(owner frame)",
            "crop_emitted": (
                "highest-score row across native branched-crop WASB and TrackNetV2 candidates "
                "at round(emission.frame)"
            ),
        },
        "artifacts": artifacts,
        "tables": {
            "per_event": str(output / "event_pixel_rows.csv"),
            "summary": str(output / "event_pixel_summary.csv"),
        },
        "summary": summaries,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("step1", "step2", "step3", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root() / "processed/wk3_eventxy/step1_pixel_accuracy",
    )
    parser.add_argument(
        "--step2-output",
        type=Path,
        default=data_root() / "processed/wk3_eventxy/step2_flight_fates",
    )
    parser.add_argument(
        "--step3-output",
        type=Path,
        default=data_root() / "processed/wk3_eventxy/step3_recommendation",
    )
    args = parser.parse_args()
    reports = {}
    if args.stage in {"step1", "all"}:
        reports["step1"] = run_step1(args.output)
    if args.stage in {"step2", "all"}:
        reports["step2"] = run_step2(args.output, args.step2_output)
    if args.stage in {"step3", "all"}:
        reports["step3"] = run_step3(args.output, args.step3_output)
    print(
        json.dumps(
            {
                name: {
                    "schema": report["schema"],
                    **(
                        {"artifacts": report["artifacts"]}
                        if "artifacts" in report
                        else {"tables": report["tables"]}
                    ),
                }
                for name, report in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
