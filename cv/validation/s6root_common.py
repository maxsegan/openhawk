"""Shared, evaluation-only helpers for the WK3 Stage-6 root-cause diagnostics.

This module reads owner-derived oracle artifacts and therefore must never be imported by
automatic inference.  It deliberately calls the production camera, fitter, physics, and
flight-classification functions without modifying their implementations.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cv.pipeline import reconstruction
from cv.pipeline.anchor_first_fit import fit_anchor_first_shot
from cv.pipeline.flight_ledger import classify_flight
from cv.pipeline.rich_ball_physics import contact_image_residual, nearest

ORACLE_RELATIVE = Path("processed/wk3_oracle/oracle_3d_ceiling_v2")
COHORT_RELATIVE = Path("processed/wk3_s6/cohort_root_v1")
RESULT_RELATIVE = Path("processed/wk3_s6root")
NET_Y_M = 11.885
BALL_RADIUS_M = 0.0325


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def percentile(values: Iterable[float], q: float) -> float | None:
    rows = np.asarray(list(values), dtype=float)
    rows = rows[np.isfinite(rows)]
    return float(np.percentile(rows, q)) if len(rows) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class FlightCase:
    point: str
    match_id: str
    clip: str
    fps: float
    surface: str
    point_detail: dict[str, Any]
    attempt: dict[str, Any]
    prior_fit: dict[str, Any] | None
    anchors: list[dict[str, Any]]
    camera: reconstruction.PointCamera
    ball: dict[int, np.ndarray]
    players: dict[str, dict[int, np.ndarray]]
    observation_weights: dict[int, float]
    contacts: list[dict[str, Any]]
    owner_frames: set[int]

    @property
    def flight_id(self) -> str:
        return f"{self.point}__flight_{int(self.attempt['flight_index']):03d}"

    @property
    def start_frame(self) -> float:
        return float(self.attempt["start_frame"])

    @property
    def end_frame(self) -> float:
        return float(self.attempt["end_frame"])


@lru_cache(maxsize=1)
def _owner_frame_index() -> dict[tuple[str, str], set[int]]:
    """Load the immutable owner sources named by owner_positions.json."""
    labels = Path(__file__).parent / "labels"
    index: dict[tuple[str, str], set[int]] = {}
    sequence_root = labels / "ball_track_sequence_v1"
    for name in ("development_truth_v1.json", "sealed_transfer_truth_v1.json"):
        payload = json.loads((sequence_root / name).read_text())
        for record in payload["records"]:
            match_id, clip, *_ = str(record["case_id"]).split("__")
            selected = index.setdefault((match_id, clip), set())
            selected.update(
                int(row["frame"])
                for row in record["frames"]
                if row.get("status") == "visible" and row.get("x1080") is not None
            )
    trajectory = labels / "ball_trajectory_v1/ball_trajectory_labels.csv"
    with trajectory.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("frame_status") == "corrected" and row.get("frame"):
                index.setdefault((row["match_id"], row["clip"]), set()).add(int(row["frame"]))
    return index


def _owner_frames(track_path: Path, clip: str) -> set[int]:
    return _owner_frame_index().get((track_path.parent.name, clip), set()).copy()


def _contacts(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "frame": float(attempt["start_frame"]),
            "side": attempt.get("start_side", "unknown"),
            "phase": attempt.get("start_phase", "rally"),
            "span": int(attempt.get("span", 0)),
            "terminal": False,
        },
        {
            "frame": float(attempt["end_frame"]),
            "side": attempt.get("end_side", "unknown"),
            "phase": "terminal" if attempt.get("terminal_end") else "rally",
            "span": int(attempt.get("span", 0)),
            "terminal": bool(attempt.get("terminal_end")),
        },
    ]


def load_flight_cases(
    oracle_root: Path | None = None,
    *,
    solved_only: bool = False,
    report_path: Path | None = None,
    truth_track_root: Path | None = None,
) -> list[FlightCase]:
    """Recreate the exact per-flight inputs used by oracle Arm D."""
    root = oracle_root or data_root() / ORACLE_RELATIVE
    report = json.loads((report_path or root / "arm_d/report.json").read_text())
    truth_root = truth_track_root or root / "truth_track_root"
    cases = []
    for point in report["points_detail"]:
        if not point.get("flight_attempts"):
            continue
        match_id = str(point["match_id"])
        clip = str(point["clip"])
        match_dir = truth_root / match_id
        try:
            camera = reconstruction.PointCamera(match_dir, clip)
        except ValueError:
            continue
        raw_ball = reconstruction.load_track(match_dir, clip)
        ball, _ = reconstruction.smooth_track(
            raw_ball,
            point["active_spans"],
            float(point["fps"]),
        )
        track_weights = reconstruction.load_track_weights(match_dir, clip)
        observation_weights = {
            frame: track_weights.get(frame, 0.20) if frame in raw_ball else 0.20 for frame in ball
        }
        players, _ = reconstruction.load_players(match_dir, clip)
        fits = {int(row["flight_index"]): row for row in point.get("fits", [])}
        anchors = {
            int(row["flight_index"]): list(row["anchors"])
            for row in point.get("flight_anchors", {}).get("flights", [])
        }
        owner = _owner_frames(match_dir / reconstruction.TRACK_NAME, clip)
        for attempt in point["flight_attempts"]:
            flight_index = int(attempt["flight_index"])
            prior_fit = fits.get(flight_index)
            if solved_only and prior_fit is None:
                continue
            cases.append(
                FlightCase(
                    point=str(point["point"]),
                    match_id=match_id,
                    clip=clip,
                    fps=float(point["fps"]),
                    surface=str(point["surface"]),
                    point_detail=point,
                    attempt=attempt,
                    prior_fit=prior_fit,
                    anchors=anchors.get(flight_index, []),
                    camera=camera,
                    ball=ball,
                    players=players,
                    observation_weights=observation_weights,
                    contacts=_contacts(attempt),
                    owner_frames=owner,
                )
            )
    return sorted(cases, key=lambda row: row.flight_id)


def fit_case(
    case: FlightCase,
    *,
    camera: Any | None = None,
    ball: dict[int, np.ndarray] | None = None,
    anchors: list[dict[str, Any]] | None = None,
    fps: float | None = None,
    contacts: list[dict[str, Any]] | None = None,
    max_nfev: int = 20,
    initial_fit: Any | None = None,
) -> Any | None:
    return fit_anchor_first_shot(
        0,
        contacts or case.contacts,
        ball or case.ball,
        case.players,
        camera or case.camera,
        case.fps if fps is None else fps,
        case.surface,
        max_nfev,
        case.anchors if anchors is None else anchors,
        observation_weights=case.observation_weights,
        initial_fit=initial_fit,
    )


def compact_and_classify(
    case: FlightCase,
    fit: Any | None,
    *,
    camera: Any | None = None,
    ball: dict[int, np.ndarray] | None = None,
    contacts: list[dict[str, Any]] | None = None,
    fps: float | None = None,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Run the unchanged flight ledger classifier on a diagnostic refit."""
    if fit is None:
        status, reasons = classify_flight(case.point_detail, case.attempt, None)
        return status, reasons, None
    selected_camera = camera or case.camera
    selected_ball = ball or case.ball
    selected_contacts = contacts or case.contacts
    selected_fps = case.fps if fps is None else fps
    compact = reconstruction.compact_fit(fit, selected_fps, case.surface)
    compact.update(
        {
            "flight_index": int(case.attempt["flight_index"]),
            "start_frame": float(selected_contacts[0]["frame"]),
            "end_frame": float(selected_contacts[1]["frame"]),
            "terminal_end": bool(selected_contacts[1].get("terminal")),
            "observation_coverage": float(
                compact["observations"]
                / max(
                    1.0,
                    float(selected_contacts[1]["frame"])
                    - float(selected_contacts[0]["frame"])
                    + 1.0,
                )
            ),
        }
    )
    compact["end_xyz"] = fit.state(
        float(selected_contacts[1]["frame"]), selected_fps, case.surface
    )[0].tolist()
    for prefix, xyz_key, contact in (
        ("start", "start_xyz", selected_contacts[0]),
        ("end", "end_xyz", selected_contacts[1]),
    ):
        _, diagnostic = contact_image_residual(
            np.asarray(compact[xyz_key], dtype=float),
            contact,
            selected_ball,
            selected_camera,
            case.observation_weights,
        )
        compact[f"{prefix}_contact_reprojection_px"] = diagnostic["error_px"]
        player = nearest(
            case.players.get(str(contact.get("side")), {}),
            float(contact["frame"]),
        )
        compact[f"{prefix}_player_distance_m"] = (
            float(np.linalg.norm(np.asarray(compact[xyz_key][:2]) - player))
            if player is not None and not contact.get("terminal")
            else None
        )
    status, reasons = classify_flight(case.point_detail, case.attempt, compact)
    return status, reasons, compact


class ScaledVerticalCamera:
    """Scale only P's world-z column, leaving every z=0 court projection exact."""

    def __init__(self, base: reconstruction.PointCamera, scale: float):
        self.base = base
        self.scale = float(scale)

    def p_at(self, frame: float) -> np.ndarray:
        projection = np.asarray(self.base.p_at(frame), dtype=float).copy()
        projection[:, 2] *= self.scale
        return projection

    def h_at(self, frame: float) -> np.ndarray:
        return self.base.h_at(frame)

    def quality_at(self, frame: float) -> dict[str, Any]:
        return self.base.quality_at(frame)


class DoubledFrameCamera:
    """Expose a camera on half-frame ticks while preserving physical time."""

    def __init__(self, base: reconstruction.PointCamera):
        self.base = base

    def p_at(self, frame: float) -> np.ndarray:
        return self.base.p_at(float(frame) / 2.0)

    def h_at(self, frame: float) -> np.ndarray:
        return self.base.h_at(float(frame) / 2.0)

    def quality_at(self, frame: float) -> dict[str, Any]:
        return self.base.quality_at(float(frame) / 2.0)


@lru_cache(maxsize=None)
def video_pts(video: str) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            video,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.asarray(
        [float(line.split(",", 1)[0]) for line in completed.stdout.splitlines() if line.strip()],
        dtype=float,
    )


@lru_cache(maxsize=None)
def point_map(path: str) -> dict[str, tuple[float, float]]:
    with Path(path).open(newline="") as handle:
        return {
            f"pt{int(row['pt']):04d}": (float(row["rally_t_start"]), float(row["rally_t_end"]))
            for row in csv.DictReader(handle)
        }


def clip_pts(match_dir: Path, clip: str) -> np.ndarray:
    """Read the actual PTS of the audit-reel frames backing one extracted point."""
    video = match_dir / "audit_reel_native_1080.mp4"
    start, end = point_map(os.fspath(match_dir / "audit_reel_point_map.csv"))[clip]
    rows = video_pts(os.fspath(video.resolve()))
    selected = rows[(rows >= start - 1e-7) & (rows < end - 1e-7)]
    if not len(selected):
        raise ValueError(f"no source PTS for {match_dir.name}/{clip}")
    return selected - selected[0]


def frame_time(pts: np.ndarray, frame: float, fps: float) -> float:
    """Interpolate a possibly sub-frame event onto the source PTS timebase."""
    position = float(frame) - 1.0
    lower = int(np.clip(math.floor(position), 0, len(pts) - 1))
    upper = int(np.clip(math.ceil(position), 0, len(pts) - 1))
    if lower == upper:
        return float(pts[lower])
    fraction = position - lower
    return float((1.0 - fraction) * pts[lower] + fraction * pts[upper])
