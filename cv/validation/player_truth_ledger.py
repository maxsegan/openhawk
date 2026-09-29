#!/usr/bin/env python3
"""Measure automatic player/pose state against dense player-racket truth.

Evaluation labels are opened only here. Automatic state composition lives in
``cv.pipeline.player_side_association`` and accepts inference artifacts only.
The metre errors below use the promoted ``metric_witnesses`` per-frame camera;
the racket metre number is explicitly a court-plane-equivalent image error,
not independent airborne 3-D truth.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import contact_striker
from cv.pipeline.player_side_association import (
    build_player_state_rows,
    write_player_state_artifact,
)
from cv.pipeline.pose import (
    load_native_pose_rows,
    pose_foot_pixel,
    pose_hip_pixel,
    racket_face_from_pose,
    vertical_height_from_pixel,
)
from cv.pipeline.provenance import file_sha256
from cv.validation import score_contact_strikers

SCHEMA = "tennis_player_truth_ledger_v1"
LABEL_GLOB = "cv/validation/labels/s6_agent_inputs_v1/*_players_v1.json"
DEFAULT_COHORT = "processed/postseg_pipeline_benchmark_969657a"
DEFAULT_CAMERA_ROOT = "processed/camheight/metric_final/cameras"
RACKET_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
MOVING_SPEED_MS = 1.5


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def _xy(value) -> tuple[float, float] | None:
    if isinstance(value, dict):
        value = value.get("xy_native_px") or value.get("px")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _normalise_foot(foot: dict) -> dict:
    ground = foot.get("ground_contact_px")
    if ground is None:
        ground = foot.get("ground_contact")
    return {
        "ground": _xy(ground),
        "airborne": foot.get("airborne"),
        "occluded": bool(foot.get("occluded", False)),
        "status": str(foot.get("status") or foot.get("contact_state") or "unknown"),
    }


def _normalise_racket(racket: dict | None) -> dict | None:
    if not isinstance(racket, dict):
        return None
    face = racket.get("face_center_px", racket.get("face_centre_px"))
    if face is None:
        face = racket.get("face_center")
    return {
        "face": _xy(face),
        "active": bool(racket.get("active_striker", racket.get("is_striker", False))),
        "hand": racket.get("striking_hand"),
        "stroke": racket.get("stroke_type", racket.get("stroke")),
        "phase": racket.get("stroke_phase"),
        "contact_frame": racket.get(
            "contact_frame_reference",
            racket.get("contact_epoch", racket.get("contact_epoch_frame")),
        ),
    }


def _normalise_player(player: dict) -> dict:
    hip = player.get("hip_center_px")
    if hip is None:
        hip = player.get("hip_center", player.get("hip_centre"))
    feet = player.get("feet") or {}
    if isinstance(feet, list):
        feet = {str(index): foot for index, foot in enumerate(feet)}
    return {
        "identity": player.get("identity", player.get("player", player.get("player_id"))),
        "in_view": bool(player.get("in_view", player.get("visibility") != "out_of_view")),
        "occluded": bool(player.get("occlusion", player.get("occluded", False))),
        "hip": _xy(hip),
        "feet": [_normalise_foot(foot) for foot in feet.values()],
        "racket": _normalise_racket(player.get("racket")),
    }


def _contacts(payload: dict, frames: list[dict]) -> list[dict]:
    contacts = []
    for row in payload.get("contacts") or []:
        frame = row.get("contact_frame", row.get("frame"))
        end = row.get("player_end", row.get("hitter_end"))
        if frame is not None:
            contacts.append(
                {
                    "frame": float(frame),
                    "end": end,
                    "stroke_type": row.get("stroke_type", row.get("stroke")),
                    "hand": row.get("striking_hand", row.get("hand")),
                }
            )
    for row in payload.get("contact_windows") or []:
        contacts.append(
            {
                "frame": float(row["epoch_native_frame"]),
                "end": row.get("hitter"),
                "stroke_type": row.get("stroke"),
                "hand": row.get("striking_hand"),
            }
        )
    if contacts:
        return sorted(contacts, key=lambda row: row["frame"])
    seen = set()
    for frame in frames:
        for end, player in frame["players"].items():
            racket = player.get("racket")
            if not racket or not racket["active"] or racket["contact_frame"] is None:
                continue
            key = (float(racket["contact_frame"]), end)
            if key in seen:
                continue
            seen.add(key)
            contacts.append(
                {
                    "frame": key[0],
                    "end": end,
                    "stroke_type": racket.get("stroke"),
                    "hand": racket.get("hand"),
                }
            )
    return sorted(contacts, key=lambda row: row["frame"])


def _apply_errata(path: Path, payload: dict) -> tuple[dict, list[dict]]:
    errata_path = path.with_name(f"{path.stem}_errata.json")
    if not errata_path.is_file():
        return payload, []
    errata = json.loads(errata_path.read_text())
    if errata.get("base_sha256") != file_sha256(path):
        raise ValueError(f"errata base hash mismatch: {errata_path}")
    payload = copy.deepcopy(payload)
    for correction in errata.get("corrections") or []:
        frame = next(row for row in payload["frames"] if row["frame"] == correction["frame"])
        frame["players"][correction["end"]]["feet"][correction["foot"]] = correction["replacement"]
    return payload, [{"path": str(errata_path), "sha256": file_sha256(errata_path)}]


def load_truth(path: Path) -> dict:
    payload, errata = _apply_errata(path, json.loads(path.read_text()))
    racket_by_frame: dict[tuple[int, str], dict] = {}
    for racket in payload.get("racket_frames") or []:
        racket_by_frame[(int(racket["frame"]), str(racket["hitter_end"]))] = _normalise_racket(
            {**racket, "is_striker": True}
        )
    frames = []
    for row in payload["frames"]:
        players = {
            end: _normalise_player(player)
            for end, player in row["players"].items()
            if end in {"near", "far"}
        }
        for end, player in players.items():
            player["racket"] = racket_by_frame.get((int(row["frame"]), end), player["racket"])
        frames.append(
            {
                "frame": int(row["frame"]),
                "pts": row.get("native_pts_seconds"),
                "players": players,
            }
        )
    attempt = payload.get("attempt")
    clip = payload.get("clip") or (attempt.get("clip") if isinstance(attempt, dict) else None)
    window = payload.get("native_window")
    if window is None and isinstance(attempt, dict):
        window = attempt.get("native_window")
    return {
        "path": path,
        "schema": payload["schema"],
        "match_id": payload["match_id"],
        "clip": str(clip),
        "window": [int(value) for value in window]
        if window
        else [frames[0]["frame"], frames[-1]["frame"]],
        "frames": frames,
        "contacts": _contacts(payload, frames),
        "receipts": [{"path": str(path), "sha256": file_sha256(path)}, *errata],
    }


def _load_metric_cameras(root: Path) -> list[dict]:
    cameras = []
    for metadata_path in sorted(root.glob("*/cameras.json")):
        payload = json.loads(metadata_path.read_text())
        camera_dir = metadata_path.parent
        cameras.append(
            {
                "key": camera_dir.name,
                "match_id": payload.get("match_id"),
                "clip": payload.get("clip"),
                "window": payload.get("configuration", {}).get("native_window"),
                "H": camera_dir / "court_H_per_frame_v1.npz",
                "P": camera_dir / "camera_P_per_frame_v1.npz",
                "metadata": metadata_path,
            }
        )
    return cameras


def _camera_for_truth(truth: dict, cameras: list[dict]) -> dict | None:
    candidates = [
        camera
        for camera in cameras
        if camera["match_id"] == truth["match_id"] and camera["clip"] == truth["clip"]
    ]
    exact = [camera for camera in candidates if camera["window"] == truth["window"]]
    return exact[0] if exact else (candidates[0] if len(candidates) == 1 else None)


def _matrices(path: Path, key: str, clip: str) -> dict[int, np.ndarray]:
    output = {}
    with np.load(path, allow_pickle=True) as data:
        reliable = data.get("reliable", np.ones(len(data["frames"]), dtype=bool))
        for row_clip, frame, matrix, accepted in zip(
            data["clips"], data["frames"], data[key], reliable, strict=True
        ):
            value = np.asarray(matrix, dtype=float)
            if str(row_clip) == clip and bool(accepted) and np.isfinite(value).all():
                output[int(frame)] = value
    return output


def _project(homography: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float]:
    point = cv2.perspectiveTransform(np.float32([[[*pixel]]]), homography)[0, 0]
    return float(point[0]), float(point[1])


def _truth_position(player: dict, homography: np.ndarray) -> tuple[float, float] | None:
    points = [foot["ground"] for foot in player["feet"] if foot["ground"] is not None]
    if not points:
        return None
    court = np.asarray([_project(homography, point) for point in points], dtype=float)
    return float(court[:, 0].mean()), float(court[:, 1].mean())


def _row_box(row: dict, scale: float) -> tuple[float, float, float, float]:
    return contact_striker._box_in_native(row, scale)


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        return float(np.percentile(ordered, 100.0 * fraction))

    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 4),
        "p90": round(percentile(0.9), 4),
        "max": round(ordered[-1], 4),
    }


def _metric_error(record: dict, arm: str) -> float | None:
    truth = record.get("truth_court")
    predicted = record.get(f"{arm}_court")
    if truth is None or predicted is None:
        return None
    return float(np.linalg.norm(np.asarray(predicted) - np.asarray(truth)))


def _arm_summary(records: list[dict], arm: str) -> dict:
    eligible = [row for row in records if row["truth_court"] is not None]
    errors = [error for row in eligible if (error := _metric_error(row, arm)) is not None]

    def sliced(key: str, value: str) -> dict | None:
        return _distribution(
            [
                error
                for row in eligible
                if row.get(key) == value and (error := _metric_error(row, arm)) is not None
            ]
        )

    return {
        "eligible_player_frames": len(eligible),
        "answered": len(errors),
        "no_answer": len(eligible) - len(errors),
        "coverage": round(len(errors) / len(eligible), 4) if eligible else None,
        "error_m": _distribution(errors),
        "near_error_m": sliced("side", "near"),
        "far_error_m": sliced("side", "far"),
        "moving_error_m": sliced("motion", "moving"),
        "planted_error_m": sliced("motion", "planted"),
    }


def _motion_classes(records: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in records:
        if row["truth_court"] is not None:
            grouped[(row["attempt"], row["side"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["frame"])
        for index, row in enumerate(rows):
            neighbors = rows[max(0, index - 2) : min(len(rows), index + 3)]
            if len(neighbors) < 2:
                row["motion"] = "unknown"
                continue
            first, last = neighbors[0], neighbors[-1]
            dt = None
            if first["pts"] is not None and last["pts"] is not None:
                dt = float(last["pts"]) - float(first["pts"])
            if not dt or dt <= 0:
                dt = (last["frame"] - first["frame"]) / 25.0
            speed = float(
                np.linalg.norm(np.asarray(last["truth_court"]) - np.asarray(first["truth_court"]))
                / max(dt, 1e-6)
            )
            row["truth_speed_ms"] = round(speed, 3)
            row["motion"] = "moving" if speed >= MOVING_SPEED_MS else "planted"


def _swap_summary(records: list[dict], arm: str) -> dict:
    grouped = defaultdict(dict)
    for row in records:
        if row["truth_court"] is not None and row.get(f"{arm}_court") is not None:
            grouped[(row["attempt"], row["frame"])][row["side"]] = row
    comparable = swaps = 0
    for sides in grouped.values():
        if set(sides) != {"near", "far"}:
            continue
        comparable += 1
        near, far = sides["near"], sides["far"]
        assigned = _metric_error(near, arm) + _metric_error(far, arm)
        crossed = float(
            np.linalg.norm(np.asarray(near[f"{arm}_court"]) - np.asarray(far["truth_court"]))
            + np.linalg.norm(np.asarray(far[f"{arm}_court"]) - np.asarray(near["truth_court"]))
        )
        swaps += crossed + 0.25 < assigned
    return {
        "comparable_frames": comparable,
        "identity_side_swaps": swaps,
        "swap_rate": round(swaps / comparable, 4) if comparable else None,
        "definition": "cross-assignment beats declared near/far assignment by >0.25 m",
    }


def _behaviour_summary(records: list[dict], arm: str) -> dict:
    def coverage(rows: list[dict]) -> dict:
        emitted = sum(row.get(f"{arm}_court") is not None for row in rows)
        return {
            "player_frames": len(rows),
            "emitted": emitted,
            "emission_rate": round(emitted / len(rows), 4) if rows else None,
        }

    airborne = [row for row in records if row["support"] == "airborne"]
    mixed = [row for row in records if row["support"] == "mixed"]
    occluded = [row for row in records if row["occluded"]]
    visible = [row for row in records if row["in_view"] and not row["occluded"]]
    return {
        "airborne": coverage(airborne),
        "mixed_support": coverage(mixed),
        "occluded": coverage(occluded),
        "visible_unoccluded": coverage(visible),
        "airborne_policy": "automatic pose has no qualified airborne classifier; emitted XY carries uncertainty rather than a false grounded flag",
    }


def _automatic_inputs(truth: dict, cohort: Path) -> dict:
    match_dir = cohort / truth["match_id"]
    match_inputs = contact_striker.MatchInputs.resolve(match_dir)
    sided_path = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))[0]
    _image, artifact = contact_striker.declared_sizes(sided_path)
    scale = 1920.0 / artifact.width
    with sided_path.open(newline="") as handle:
        sided = [row for row in csv.DictReader(handle) if row.get("clip") == truth["clip"]]
    pose_path = match_dir / contact_striker.POSE_NAME
    pose = [
        row for row in load_native_pose_rows(str(pose_path)) if row.get("clip") == truth["clip"]
    ]
    ball = (
        {}
        if match_inputs.ball is None
        else contact_striker.ball_pixels(match_inputs.ball, truth["clip"])
    )
    return {
        "match_dir": match_dir,
        "sided_path": sided_path,
        "sided_scale": scale,
        "sided": sided,
        "pose_path": pose_path,
        "pose": pose,
        "ball": ball,
        "fps": match_inputs.fps,
    }


def _index_best(rows: list[dict]) -> dict[tuple[int, str], dict]:
    output = {}
    for row in rows:
        side = str(row.get("side") or "")
        if side not in {"near", "far"}:
            continue
        key = (contact_striker.frame_number(row["frame"]), side)
        if key not in output or float(row.get("conf", 0)) > float(output[key].get("conf", 0)):
            output[key] = row
    return output


def score_attempt(truth: dict, camera: dict, cohort: Path) -> dict:
    automatic = _automatic_inputs(truth, cohort)
    homographies = _matrices(camera["H"], "H", truth["clip"])
    projections = _matrices(camera["P"], "P", truth["clip"])
    sided = _index_best(automatic["sided"])
    pose = _index_best(automatic["pose"])
    records = []
    racket_rows = []
    hip_rows = []
    for frame_row in truth["frames"]:
        frame = frame_row["frame"]
        homography = homographies.get(frame)
        if homography is None:
            continue
        for side, player in frame_row["players"].items():
            feet = player["feet"]
            all_airborne = bool(feet) and all(foot["airborne"] is True for foot in feet)
            any_airborne = any(foot["airborne"] is True for foot in feet)
            support = (
                "airborne" if all_airborne else ("mixed" if any_airborne else "grounded_or_unknown")
            )
            row = {
                "attempt": camera["key"],
                "match_id": truth["match_id"],
                "clip": truth["clip"],
                "frame": frame,
                "pts": frame_row["pts"],
                "side": side,
                "identity": player["identity"],
                "in_view": player["in_view"],
                "occluded": player["occluded"] or any(foot["occluded"] for foot in feet),
                "support": support,
                "truth_court": _truth_position(player, homography),
                "sided_box_court": None,
                "pose_ankle_mean_court": None,
                "pose_lower_ankle_court": None,
            }
            sided_row = sided.get((frame, side))
            if sided_row is not None:
                box = _row_box(sided_row, automatic["sided_scale"])
                row["sided_box_court"] = _project(homography, (0.5 * (box[0] + box[2]), box[3]))
            pose_row = pose.get((frame, side))
            if pose_row is not None:
                for estimator in ("ankle_mean", "lower_ankle"):
                    pixel = pose_foot_pixel(pose_row, estimator=estimator)
                    if pixel is not None:
                        row[f"pose_{estimator}_court"] = _project(homography, pixel)
                truth_hip = player["hip"]
                automatic_hip = pose_hip_pixel(pose_row)
                if (
                    truth_hip is not None
                    and automatic_hip is not None
                    and row["truth_court"] is not None
                ):
                    truth_height = vertical_height_from_pixel(
                        projections.get(frame), row["truth_court"], truth_hip
                    )
                    automatic_height = vertical_height_from_pixel(
                        projections.get(frame),
                        row.get("pose_ankle_mean_court") or row["truth_court"],
                        automatic_hip,
                    )
                    hip_rows.append(
                        {
                            "attempt": camera["key"],
                            "side": side,
                            "pixel_error": float(
                                np.linalg.norm(np.asarray(automatic_hip) - truth_hip)
                            ),
                            "height_error": (
                                None
                                if truth_height is None or automatic_height is None
                                else abs(automatic_height - truth_height)
                            ),
                        }
                    )
                racket = player["racket"]
                if racket and racket["active"] and racket["face"] is not None:
                    truth_face = racket["face"]
                    selectors = {
                        "truth_hand_oracle": {"hand": racket.get("hand")},
                        "automatic_ball_pixel": {"contact_pixel": automatic["ball"].get(frame)},
                    }
                    for selector, selector_args in selectors.items():
                        for scale in RACKET_SCALES:
                            estimate = (
                                None
                                if selector == "automatic_ball_pixel"
                                and selector_args["contact_pixel"] is None
                                else racket_face_from_pose(
                                    pose_row,
                                    forearm_scale=scale,
                                    **selector_args,
                                )
                            )
                            racket_rows.append(
                                {
                                    "attempt": camera["key"],
                                    "frame": frame,
                                    "side": side,
                                    "selector": selector,
                                    "scale": scale,
                                    "pixel_error": (
                                        None
                                        if estimate is None
                                        else float(
                                            np.linalg.norm(
                                                np.asarray([estimate["x"], estimate["y"]])
                                                - np.asarray(truth_face)
                                            )
                                        )
                                    ),
                                    "court_error": (
                                        None
                                        if estimate is None
                                        else float(
                                            np.linalg.norm(
                                                np.asarray(
                                                    _project(
                                                        homography,
                                                        (estimate["x"], estimate["y"]),
                                                    )
                                                )
                                                - np.asarray(_project(homography, truth_face))
                                            )
                                        )
                                    ),
                                }
                            )
            records.append(row)
    return {
        "records": records,
        "racket_rows": racket_rows,
        "hip_rows": hip_rows,
        "receipts": {
            "truth": truth["receipts"],
            "sided_boxes": {
                "path": str(automatic["sided_path"]),
                "sha256": file_sha256(automatic["sided_path"]),
            },
            "pose": {
                "path": str(automatic["pose_path"]),
                "sha256": file_sha256(automatic["pose_path"]),
            },
            "metric_camera": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in (camera["metadata"], camera["H"], camera["P"])
            ],
        },
    }


def _racket_summary(rows: list[dict]) -> dict:
    selectors = {}
    for selector in ("automatic_ball_pixel", "truth_hand_oracle"):
        scales = {}
        for scale in RACKET_SCALES:
            selected = [
                row for row in rows if row["selector"] == selector and row["scale"] == scale
            ]
            scales[str(scale)] = {
                "truth_faces": len(selected),
                "native_px": _distribution(
                    [row["pixel_error"] for row in selected if row["pixel_error"] is not None]
                ),
                "court_plane_equivalent_m": _distribution(
                    [row["court_error"] for row in selected if row["court_error"] is not None]
                ),
            }
        selectors[selector] = scales
    scales = selectors["automatic_ball_pixel"]
    ranked = [
        (scale, summary) for scale, summary in scales.items() if summary["native_px"] is not None
    ]
    best = min(
        ranked, key=lambda item: (item[1]["native_px"]["median"], item[1]["native_px"]["p90"])
    )[0]
    return {
        "selectors": selectors,
        "best_forearm_scale": float(best),
        "selection": "lowest automatic-ball-pixel native-pixel median, then p90; opened development, not holdout calibration",
        "oracle_use": "truth hand isolates extrapolation geometry and never enters inference",
    }


def _hip_summary(rows: list[dict]) -> dict:
    return {
        "native_px": _distribution([row["pixel_error"] for row in rows]),
        "metric_height_proxy_m": _distribution(
            [row["height_error"] for row in rows if row["height_error"] is not None]
        ),
        "semantics": "automatic and truth hip pixels lifted on their respective vertical lines; not independent anatomical 3-D truth",
    }


def _truth_contact_cases(truths: list[dict]) -> list[dict]:
    cases = []
    for truth in truths:
        for index, contact in enumerate(truth["contacts"]):
            cases.append(
                {
                    "case_id": f"{truth['path'].stem}:{index}",
                    "match_id": truth["match_id"],
                    "clip": truth["clip"],
                    "frame": contact["frame"],
                    "truth_end": contact.get("end"),
                    "stroke_type": contact.get("stroke_type"),
                }
            )
    return cases


def _state_artifact(
    truth: dict,
    camera: dict,
    cohort: Path,
    output_root: Path,
    event_emissions: Path | None,
) -> dict:
    automatic = _automatic_inputs(truth, cohort)
    homographies = _matrices(camera["H"], "H", truth["clip"])
    projections = _matrices(camera["P"], "P", truth["clip"])
    tracks = []
    for row in automatic["sided"]:
        frame = contact_striker.frame_number(row["frame"])
        homography = homographies.get(frame)
        if homography is None:
            continue
        box = _row_box(row, automatic["sided_scale"])
        root = (0.5 * (box[0] + box[2]), box[3])
        court = _project(homography, root)
        tracks.append(
            {
                "clip": truth["clip"],
                "frame": frame,
                "t": row.get("t", ""),
                "side": row["side"],
                "name": "",
                "track_id": row.get("track_id", ""),
                "root_x_native": root[0],
                "root_y_native": root[1],
                "court_x": court[0],
                "court_y": court[1],
            }
        )
    contact_rows = []
    if event_emissions is not None and event_emissions.is_file():
        inputs = contact_striker.MatchInputs.resolve(automatic["match_dir"])
        automatic_contacts = contact_striker.emitted_contacts(
            event_emissions, truth["match_id"]
        ).get(truth["clip"], [])
        contact_rows = inputs.resolve_clip(truth["clip"], automatic_contacts)
    rows = build_player_state_rows(
        tracks,
        automatic["pose"],
        homography_for_frame=lambda _clip, frame: homographies.get(frame),
        projection_for_frame=lambda _clip, frame: projections.get(frame),
        contact_strikers=contact_rows,
        foot_estimator="sided_box",
    )
    output = output_root / truth["match_id"] / f"{truth['clip']}_player_state_v1.csv"
    write_player_state_artifact(
        output,
        rows,
        sources={
            "sided_boxes": automatic["sided_path"],
            "pose": automatic["pose_path"],
            "court": camera["H"],
            "camera": camera["P"],
            "contact_strikers": None,
            "event_emissions": event_emissions,
        },
        fps=automatic["fps"],
        foot_estimator="sided_box",
        gaps=[],
        automatic=False,
        labels_or_reviewed_inputs=[str(truth["path"])],
    )
    return {
        "attempt": camera["key"],
        "path": str(output),
        "sidecar": str(output) + ".coordinates.json",
        "provenance": str(output) + ".provenance.json",
        "rows": len(rows),
    }


def run(args) -> dict:
    root = data_root()
    cohort = args.cohort or (root / DEFAULT_COHORT)
    camera_root = args.metric_camera_root or (root / DEFAULT_CAMERA_ROOT)
    truths = [load_truth(Path(path)) for path in sorted(glob.glob(args.labels))]
    cameras = _load_metric_cameras(camera_root)
    all_records, rackets, hips, attempts = [], [], [], []
    for truth in truths:
        camera = _camera_for_truth(truth, cameras)
        if camera is None:
            attempts.append(
                {
                    "label": str(truth["path"]),
                    "status": "abstain",
                    "reason": "no_unique_promoted_metric_camera",
                }
            )
            continue
        result = score_attempt(truth, camera, cohort)
        all_records.extend(result["records"])
        rackets.extend(result["racket_rows"])
        hips.extend(result["hip_rows"])
        attempts.append(
            {
                "attempt": camera["key"],
                "label": str(truth["path"]),
                "schema": truth["schema"],
                "status": "scored",
                "player_frames": len(result["records"]),
                "contacts": len(truth["contacts"]),
                "receipts": result["receipts"],
            }
        )
    _motion_classes(all_records)
    arms = ("sided_box", "pose_ankle_mean", "pose_lower_ankle")
    summary = {
        arm: {
            **_arm_summary(all_records, arm),
            "identity": _swap_summary(all_records, arm),
            "behaviour": _behaviour_summary(all_records, arm),
        }
        for arm in arms
    }
    truth_contacts = score_contact_strikers.score_hitter_cases(cohort, _truth_contact_cases(truths))
    disjoint_hitters = score_contact_strikers.score_hitter_cases(
        cohort, score_contact_strikers.astralandmarks_hitter_cases()
    )
    states = []
    if args.states_root is not None:
        event_emissions = args.event_emissions
        for truth in truths:
            camera = _camera_for_truth(truth, cameras)
            if camera is not None:
                states.append(
                    _state_artifact(truth, camera, cohort, args.states_root, event_emissions)
                )
    return {
        "schema": SCHEMA,
        "automatic_inference_eligible": False,
        "opened_development_data": True,
        "labels_enter_inference": False,
        "denominator": {
            "label_files_discovered": len(truths),
            "attempts_scored": sum(row["status"] == "scored" for row in attempts),
            "broadcasts": len({truth["match_id"] for truth in truths}),
            "player_frames": len(all_records),
            "truth_ground_player_frames": sum(
                row["truth_court"] is not None for row in all_records
            ),
            "racket_face_scale_rows": len(rackets),
        },
        "configuration": {
            "cohort": str(cohort),
            "metric_camera_root": str(camera_root),
            "moving_threshold_ms": MOVING_SPEED_MS,
            "truth_position": "mean of visible grounded foot contact points in court metres",
            "racket_metric_semantics": "both image points projected to z=0; court-plane-equivalent error only",
        },
        "position": summary,
        "hip": _hip_summary(hips),
        "racket": _racket_summary(rackets),
        "striker_player_truth_contacts": truth_contacts,
        "striker_disjoint_40": disjoint_hitters,
        "attempts": attempts,
        "state_artifacts": states,
        "per_player_frame": all_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=LABEL_GLOB)
    parser.add_argument("--cohort", type=Path, default=None)
    parser.add_argument("--metric-camera-root", type=Path, default=None)
    parser.add_argument("--states-root", type=Path, default=None)
    parser.add_argument("--event-emissions", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["denominator"], indent=2))
    print(json.dumps(payload["position"], indent=2))
    print(json.dumps(payload["racket"], indent=2))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
