"""Court, player boxes, 2D skeletons and fitted flights on the portal's overlay video.

Owner 2026-10-08: the match explorer's overlay video should show the court, pose and boxes as
well as the ball. The 3D viewer already draws a document's ``flight_overlay`` layer (court
lines, boxes, skeletons, automatic ball, fitted path, events); this module builds that layer for
automatic production documents from the match's own automatic outputs only:

- court lines: the per-frame camera the fitter was given (``fit/attempts/<key>/inputs/
  cameras.json``, checked against the document's ``input_bindings.cameras`` sha256). Held
  frames borrow their reference frame's camera and stay marked ``held``. Without that file the
  upstream ``camera_P_per_frame_v1.npz`` row is used and the layer says so. A frame without
  any camera gets no court lines.
- boxes and skeletons: the sided player boxes and the 2D detector keypoints kept in
  ``pose/player_pose_3d_v1.npz`` (observations, not the lifted 3D joints). Without the pose
  file, boxes only from ``player_boxes_25_native_sided_v1.csv``.
- ball: the automatic detector centre already in ``video_overlay`` (observation).
- fit: the fitter's own reprojection of each flight (``video_overlay.fitted_projection``) and
  the flight's fitted states projected through the frame's camera; flights that failed their
  gates are drawn as ``rejected``. The layer is written once per attempt: a parent document
  carries every piece's flights and the viewer's piece views reuse it.

Only frames in the attempt window (the scored span plus ``MARGIN_FRAMES``) are written, pixels
are rounded to 0.1 px and keypoints are a fixed-order list. ``TENNIS_PORTAL_FLIGHT_OVERLAY=off``
skips it (rollback). Evaluation labels are never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cv.pipeline import player_pose_3d
from cv.viz import export_connected_3d as legacy
from cv.viz.flight_overlay_geometry import (
    KEYPOINT_NAMES,
    MIN_KEYPOINT_CONFIDENCE,
    camera_kind,
    court_lines,
    court_segments,
    project,
    xy,
)

SCHEMA = "automatic_flight_overlay_v1"
ENV = "TENNIS_PORTAL_FLIGHT_OVERLAY"
MARGIN_FRAMES = 12
PATH_POINTS = 32
#: The joints the viewer's skeleton edges use, in the order each player's ``joints`` list has.
JOINT_NAMES = tuple(name for name in KEYPOINT_NAMES if not name.endswith(("_eye", "_ear")))
SIDED_BOXES = "player_boxes_25_native_sided_v1.csv"
UPSTREAM_CAMERA = "camera_P_per_frame_v1.npz"
LAYERS = {
    "court": "estimate: court paint projected through the per-frame camera the fitter used",
    "auto": "observation: automatic ball detector centre",
    "boxes": "observation: automatic sided player boxes",
    "skeleton": "observation: automatic 2D pose keypoints (confidence >= 0.3)",
    "fit": "fitted: accepted flight, the fitter's reprojection and its path",
    "rejected": "fitted: flight that failed its gates",
    "events": "automatic event emissions",
}


def enabled() -> bool:
    value = os.environ.get(ENV, "on") or "on"
    if value not in ("on", "off"):
        raise SystemExit(f"{ENV}={value!r}: expected on or off")
    return value == "on"


# --------------------------------------------------------------------------- inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet_cameras(path: Path) -> dict[int, tuple[np.ndarray | None, bool, str]]:
    """Per-frame (P, supported, source) from a fitter ``cameras.json``.

    A held row may store no P; it borrows its reference frame's camera and stays held, as the
    flight-review packs draw it.
    """
    document = json.loads(path.read_text())
    rows = document.get("cameras") or []
    by_frame = {
        int(row["frame"]): np.asarray(row["P"], float) if row.get("P") is not None else None
        for row in rows
    }
    cameras = {}
    for row in rows:
        frame = int(row["frame"])
        matrix = by_frame[frame]
        supported = bool(row.get("supported", row.get("status") == "supported"))
        if matrix is None:
            match = re.search(r"(\d+)", Path(str(row.get("reference_frame") or "")).stem)
            matrix = by_frame.get(int(match.group(1))) if match else None
            supported = False
        cameras[frame] = (matrix, supported, str(row.get("source") or ""))
    return cameras


def upstream_cameras(path: Path, clip: str) -> dict[int, tuple[np.ndarray | None, bool, str]]:
    with np.load(path, allow_pickle=False) as data:
        mask = data["clips"] == clip
        return {
            int(frame): (np.asarray(P, float), bool(reliable), str(source))
            for frame, P, reliable, source in zip(
                data["frames"][mask],
                data["P"][mask],
                data["reliable"][mask],
                data["source"][mask],
                strict=True,
            )
            if np.isfinite(P).all()
        }


class MatchInputs:
    """A match folder laid out like a production run (``fit/``, ``pose/``, ``upstream/``).

    ``root`` is the run folder itself before compaction, or a scratch folder holding the few
    files extracted from a published match's archives (``refresh_overlays``).
    """

    def __init__(self, root: Path, match_id: str, pose_npz: Path | None = None) -> None:
        self.root = root
        self.match_id = match_id
        npz = pose_npz or root / "pose" / player_pose_3d.NPZ_NAME
        self.pose_npz = npz if npz.is_file() else None
        self._pose: dict[str, dict[str, np.ndarray]] | None = None
        self._boxes: dict[tuple[str, int, str], dict] | None = None

    def camera_path(self, key: str) -> Path:
        return self.root / "fit" / "attempts" / key / "inputs" / "cameras.json"

    def cameras(
        self, document: dict[str, Any], key: str, binding: dict[str, Any] | None = None
    ) -> tuple[dict[int, tuple[np.ndarray | None, bool, str]], dict[str, Any]]:
        """The attempt's fitter cameras, checked against the bound sha256 when one is known."""
        binding = camera_binding(document) or binding
        path = self.camera_path(key)
        if binding and binding.get("path"):
            marker = f"/{self.match_id}/"
            text = str(binding["path"])
            if marker in text:
                path = self.root / text.split(marker, 1)[1]
        if path.is_file():
            receipt: dict[str, Any] = {
                "source": "fitter_packet_cameras",
                "file": str(path.relative_to(self.root)),
            }
            if binding and binding.get("sha256"):
                digest = _sha256(path)
                receipt["binding_sha256_verified"] = digest == binding["sha256"]
                if digest != binding["sha256"]:
                    return {}, {
                        **receipt,
                        "status": "abstained: camera file differs from the bound sha256",
                    }
            else:
                receipt["binding_sha256_verified"] = None
            cameras = packet_cameras(path)
            return cameras, {**receipt, "status": "ok", **_camera_counts(cameras)}
        upstream = self.root / "upstream" / self.match_id / UPSTREAM_CAMERA
        clip = str(document.get("source_clip") or "")
        if upstream.is_file() and clip:
            cameras = upstream_cameras(upstream, clip)
            return cameras, {
                "source": "upstream_per_frame_camera",
                "note": "the fitter's own camera file is absent (attempt held before fitting)",
                "status": "ok",
                **_camera_counts(cameras),
            }
        return {}, {"source": None, "status": "abstained: no camera for this attempt"}

    def _pose_by_clip(self) -> dict[str, dict[str, np.ndarray]]:
        if self._pose is None:
            pose = player_pose_3d.load_pose(self.pose_npz)
            names = [str(name) for name in pose["joint_names"]]
            order = [names.index(name) for name in JOINT_NAMES]
            clips = pose["clip"].astype(str)
            self._pose = {}
            for clip in np.unique(clips):
                mask = clips == clip
                self._pose[str(clip)] = {
                    "frame": pose["frame"][mask],
                    "side": pose["side"][mask],
                    "box": pose["box_native"][mask],
                    "keypoints": pose["keypoints_2d"][mask][:, order],
                    "confidence": pose["keypoint_conf"][mask][:, order] / 255.0,
                }
        return self._pose

    def players(self, clip: str, frames: set[int]) -> tuple[dict[int, list[dict]], str | None]:
        """Per frame ``[{side, box, joints}]`` and the source name (None when nothing is kept)."""
        found: dict[int, list[dict]] = defaultdict(list)
        if self.pose_npz is not None:
            rows = self._pose_by_clip().get(clip)
            if rows is None:
                return {}, "pose_npz"
            for i, frame in enumerate(rows["frame"]):
                frame = int(frame)
                if frame not in frames:
                    continue
                points = [
                    xy(point) if confidence >= MIN_KEYPOINT_CONFIDENCE else None
                    for point, confidence in zip(
                        rows["keypoints"][i], rows["confidence"][i], strict=True
                    )
                ]
                player = {
                    "side": player_pose_3d.SIDES[int(rows["side"][i])],
                    "box": _box(rows["box"][i]),
                }
                if any(points):
                    player["joints"] = points
                if player["box"] or "joints" in player:
                    found[frame].append(player)
            return dict(found), "pose_npz"
        boxes = self.root / "upstream" / self.match_id / SIDED_BOXES
        if not boxes.is_file():
            return {}, None
        if self._boxes is None:
            self._boxes = _sided_boxes(boxes)
        for (row_clip, frame, side), row in self._boxes.items():
            if row_clip == clip and frame in frames:
                found[frame].append({"side": side, "box": _box(row["box"])})
        return dict(found), "sided_boxes_csv"


def _sided_boxes(path: Path) -> dict[tuple[str, int, str], dict]:
    clips: set[str] = set()
    with path.open(newline="") as handle:
        clips = {row["clip"] for row in csv.DictReader(handle)}
    return player_pose_3d.load_boxes(path, {clip: (0, 10**9) for clip in clips})


def _box(values) -> list[float] | None:
    values = [float(value) for value in values]
    if not all(map(math.isfinite, values)) or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return [round(value, 1) for value in values]


def _camera_counts(cameras: dict) -> dict[str, int]:
    supported = sum(1 for _matrix, ok, _source in cameras.values() if ok)
    return {
        "frames": len(cameras),
        "supported_frames": supported,
        "held_frames": len(cameras) - supported,
    }


# --------------------------------------------------------------------------- documents


def frame_time(document: dict[str, Any], frame: float) -> float:
    timebase = document.get("native_timebase") or {}
    frames, times = timebase.get("native_frames"), timebase.get("native_times_seconds")
    if frames and times:
        return float(np.interp(frame, frames, times))
    fps = float(timebase.get("fps") or document.get("fps") or 25.0)
    return (
        float(timebase.get("origin_t") or 0.0)
        + (float(frame) - float(timebase.get("origin_frame") or 1)) / fps
    )


def attempt_window(document: dict[str, Any]) -> tuple[int, int]:
    """The scored span plus a margin, inside the document's native frame range."""
    lo, hi = (int(value) for value in document["frame_range"])
    context = document.get("review_context") or {}
    start, end = context.get("score_start_frame"), context.get("score_end_frame")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        lo = max(lo, math.floor(start) - MARGIN_FRAMES)
        hi = min(hi, math.ceil(end) + MARGIN_FRAMES)
    return lo, hi


def document_flights(
    document: dict[str, Any], piece: int | None = None, mapping: list[int] | None = None
) -> list[dict[str, Any]]:
    """Fitted flights with their states by native frame and the fitter's reprojections."""
    frames = document.get("frames") or {}
    per_flight = {
        int(row["flight_index"]): row
        for row in (document.get("per_flight") or {}).get("flights") or []
        if isinstance(row, dict) and row.get("flight_index") is not None
    }
    projections = {
        int(row["frame"]): row["fitted_projection"]
        for row in (document.get("video_overlay") or {}).get("frames") or []
        if row.get("fitted_projection")
    }
    states: dict[int, dict[int, list[float]]] = defaultdict(dict)
    segments = frames.get("segment") or []
    for i, frame in enumerate(frames.get("frame") or []):
        point = [frames.get(axis, [None] * (i + 1))[i] for axis in ("x", "y", "z")]
        match = re.fullmatch(r"flight_(\d+)", str(segments[i] if i < len(segments) else ""))
        if match and all(isinstance(value, (int, float)) for value in point):
            states[int(match.group(1))][int(frame)] = point
    flights = []
    for local, by_frame in sorted(states.items()):
        row = per_flight.get(local) or {}
        index = mapping[local] if mapping and local < len(mapping) else local
        flights.append(
            {
                "id": f"F{index}" if piece is None else f"F{index} · piece {piece}",
                "flight_index": index,
                "piece": piece,
                "role": row.get("role"),
                "accepted": bool(row.get("accepted")),
                "failed_checks": list(row.get("failures") or []),
                "start_frame": row.get("start_frame", min(by_frame)),
                "end_frame": row.get("end_frame", max(by_frame)),
                "states": by_frame,
                "projections": {
                    frame: projections[frame] for frame in by_frame if frame in projections
                },
            }
        )
    return flights


def _flight_at(frame: int, flights: list[dict[str, Any]]) -> dict[str, Any] | None:
    inside = [f for f in flights if float(f["start_frame"]) <= frame <= float(f["end_frame"])]
    near = inside or [
        f for f in flights if float(f["start_frame"]) - 2 <= frame <= float(f["end_frame"]) + 2
    ]
    return max(near, key=lambda f: float(f["start_frame"])) if near else None


def _path(matrix: np.ndarray | None, states: dict[int, list[float]]) -> list[list[float]]:
    points = [xy(project(matrix, states[frame])) for frame in sorted(states)]
    points = [point for point in points if point]
    if len(points) <= PATH_POINTS:
        return points
    last = len(points) - 1
    return [points[round(i * last / (PATH_POINTS - 1))] for i in range(PATH_POINTS)]


def frame_record(
    frame: int,
    camera: tuple[np.ndarray | None, bool, str] | None,
    auto: list[float] | None,
    players: list[dict[str, Any]],
    flights: list[dict[str, Any]],
    segments: list,
) -> dict[str, Any]:
    """One native frame; keys without evidence are left out (the viewer draws nothing)."""
    record: dict[str, Any] = {"frame": frame}
    matrix = None
    if camera is not None:
        matrix, supported, source = camera
        record["kind"] = camera_kind(source, supported)
        if matrix is not None:
            record["court"] = court_lines(matrix, segments)
    if auto:
        record["auto"] = auto
    for name, accepted in (("fit", True), ("rejected", False)):
        flight = _flight_at(frame, [f for f in flights if f["accepted"] == accepted])
        if flight is None:
            continue
        marker = flight["projections"].get(frame)
        if marker is None and frame in flight["states"]:
            marker = project(matrix, flight["states"][frame])
        if xy(marker):
            record[name] = xy(marker)
        path = _path(matrix, flight["states"]) if matrix is not None else []
        if path:
            record[f"{name}_path"] = path
    if players:
        record["players"] = players
    return record


def _events(
    document: dict[str, Any], sources: list[dict[str, Any]], auto: dict[int, list[float]]
) -> list[dict[str, Any]]:
    """Automatic event emissions of the attempt and its pieces, timed on the attempt's clock."""
    seen, rows = set(), []
    for source in sources:
        for event in source.get("observed_events") or []:
            kind, frame = event.get("event_type"), event.get("frame")
            if not kind or not isinstance(frame, (int, float)) or (kind, round(frame, 3)) in seen:
                continue
            seen.add((kind, round(frame, 3)))
            rows.append(
                {
                    "type": kind,
                    "frame": round(float(frame), 3),
                    "time_s": round(frame_time(document, frame), 3),
                    "source": "automatic",
                    "status": str(event.get("status") or event.get("timing_status") or ""),
                    "xy": auto.get(int(round(frame))),
                }
            )
    return sorted(rows, key=lambda row: row["frame"])


def build_overlay(
    document: dict[str, Any],
    window: tuple[int, int],
    cameras: dict[int, tuple[np.ndarray | None, bool, str]],
    camera_receipt: dict[str, Any],
    players: dict[int, list[dict]],
    players_source: str | None,
    flights: list[dict[str, Any]],
    event_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    overlay = document.get("video_overlay") or {}
    auto = {
        int(row["frame"]): xy(row.get("labeled_front"))
        for row in overlay.get("frames") or []
        if row.get("labeled_front") and row.get("labeled_status") != "missing"
    }
    segments = court_segments()
    frames = [
        frame_record(
            frame, cameras.get(frame), auto.get(frame), players.get(frame) or [], flights, segments
        )
        for frame in range(window[0], window[1] + 1)
    ]
    return {
        "schema": SCHEMA,
        "arm": "automatic",
        "native_size": overlay.get("native_size") or [1920, 1080],
        "window": list(window),
        "layers": LAYERS,
        "camera": camera_receipt,
        "players_source": players_source,
        "pose_keypoints": players_source == "pose_npz",
        "joint_names": list(JOINT_NAMES),
        "hold_reason": None,
        "events": _events(document, event_documents, auto),
        "flights": [
            {key: value for key, value in flight.items() if key not in ("states", "projections")}
            for flight in flights
        ],
        "frames": frames,
    }


# --------------------------------------------------------------------------- portal


def camera_binding(document: dict[str, Any]) -> dict[str, Any] | None:
    return ((document.get("provenance") or {}).get("input_bindings") or {}).get("cameras")


def _compact(document: dict[str, Any]) -> str:
    return json.dumps(legacy.point3d._clean_json(document), separators=(",", ":"), allow_nan=False)


def _attempt_key(document: dict[str, Any]) -> str:
    return str(document.get("point") or "").split("__component_", 1)[0]


def fill(portal: Path, inputs: MatchInputs, *, write: bool = True) -> dict[str, Any]:
    """Add ``flight_overlay`` to every attempt document of one match portal.

    A parent attempt's layer carries every piece's flights (tagged ``piece``); the viewer shows
    a piece through its parent's layer, so ``__component_<n>`` documents are not rewritten.
    Returns per-document sizes and sources. ``write=False`` computes everything and writes
    nothing (the refresh tool's dry run).
    """
    documents = {}
    for path in player_pose_3d.portal_documents(portal):
        document = json.loads(path.read_text())
        if document.get("source_clip") and document.get("frame_range"):
            documents[path.name] = (path, document)
    by_attempt: dict[str, list[str]] = defaultdict(list)
    for name, (_path, document) in documents.items():
        by_attempt[_attempt_key(document)].append(name)
    rows = []
    for key, names in sorted(by_attempt.items()):
        group = {name: documents[name][1] for name in names}
        for name, document in group.items():
            if "__component_" in str(document.get("point") or ""):
                continue  # a piece is shown through its attempt document's layer
            flights = document_flights(document)
            for scene in document.get("component_scenes") or []:
                child = group.get(scene.get("file"))
                if child is not None:
                    flights.extend(
                        document_flights(
                            child,
                            piece=scene.get("component_index"),
                            mapping=(scene.get("local_to_original") or {}).get("flights"),
                        )
                    )
            path = documents[name][0]
            window = attempt_window(document)
            # A parent attempt document binds no camera; its pieces bind the same attempt file.
            bound = next(filter(None, map(camera_binding, group.values())), None)
            cameras, receipt = inputs.cameras(document, key, bound)
            frames = set(range(window[0], window[1] + 1))
            players, players_source = inputs.players(str(document["source_clip"]), frames)
            document["flight_overlay"] = build_overlay(
                document,
                window,
                cameras,
                receipt,
                players,
                players_source,
                flights,
                list(group.values()),
            )
            before = path.stat().st_size
            if write:
                legacy._write_json(path, document)
                after = path.stat().st_size
            else:
                after = len(_compact(document)) + 1
            rows.append(
                {
                    "file": name,
                    "bytes_before": before,
                    "bytes_after": after,
                    "frames": window[1] - window[0] + 1,
                    "camera": receipt.get("source"),
                    "camera_status": receipt.get("status"),
                    "camera_verified": receipt.get("binding_sha256_verified"),
                    "players": players_source,
                    "flights": len(flights),
                    "accepted_flights": sum(1 for flight in flights if flight["accepted"]),
                }
            )
    return {
        "schema": SCHEMA,
        "documents": len(rows),
        "bytes_before": sum(row["bytes_before"] for row in rows),
        "bytes_after": sum(row["bytes_after"] for row in rows),
        "court_abstained": sum(1 for row in rows if row["camera_status"] != "ok"),
        "no_players": sum(1 for row in rows if row["players"] is None),
        "rows": rows,
    }
