"""Export the fixed nine-attempt connected-search evidence to the 3D portal.

This is an evaluation-only exporter.  It reads the human-derived connected-search
reports and their hash-bound label packs, never automatic pipeline inputs.  Each output
document keeps the existing ``point3d_v1`` fields and adds the selected legal-depth
family, dashed alternate members, labeled-versus-fitted event times, native-video
overlay coordinates, legality blockers, and explicit provenance.

The dense 240 Hz fitted flights are sampled only at native source exposures.  Native
JPEGs are hard-linked (copied only when hard-linking is unavailable), and every source
image named by the label pack is SHA256-verified before an attempt is exported.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from cv.pipeline import pose_lift
    from cv.viz import export_point_3d as point3d
except ModuleNotFoundError:  # direct ``python cv/viz/export_connected_3d.py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cv.pipeline import pose_lift

    import export_point_3d as point3d  # type: ignore[no-redef]


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("TENNIS_DATA_ROOT", REPO / "data"))
DEFAULT_REPORT = DEFAULT_DATA_ROOT / "processed" / "ninesweep" / "fixed_nine_sweep" / "report.json"
DEFAULT_LABEL_ROOT = REPO / "cv" / "validation" / "labels" / "s6_agent_inputs_v1"
DEFAULT_TOPOLOGIES = DEFAULT_LABEL_ROOT / "attempt_topologies_astrareview_v1.json"
DEFAULT_OUT = DEFAULT_DATA_ROOT / "processed" / "review_queue_v1" / "portal" / "3d" / "data"
DEFAULT_TOSS_LABEL_ROOT = DEFAULT_DATA_ROOT / "processed" / "astratoss_4"

OWNER_REVIEWED_SERVE_POINTS = (
    "ao2019f_pt0003_a01",
    "ao2023f_pt0001_a01",
    "ao2023f_pt0002_a01",
    "ao2023f_pt0003_a01",
    "ao2023f_pt0003_a02",
    "ao2023f_pt0004_a01",
    "grassatp2024r16_pt0001_a01",
    "miami2025f_pt0001_a01",
)
POSE_EDGES = (
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("racket_grip", "racket_head"),
)

SCHEMA = point3d.SCHEMA
INDEX_SCHEMA = point3d.INDEX_SCHEMA
EXTENSION_SCHEMA = "connected_point3d_v1"
COURT = {
    "width": point3d.COURT_W,
    "length": point3d.COURT_L,
    "net_y": point3d.NET_Y,
    "singles_inset": point3d.SINGLES_INSET,
    "service_from_net": point3d.SERVICE_FROM_NET,
    "net_center_h": point3d.NET_CENTER_H,
    "net_post_h": point3d.NET_POST_H,
}

BLOCKER_LABELS = {
    "connected_input_physics": "connected input-only physics failed",
    "serve_depth_plausible": "serve contact depth is outside the declared legal region",
    "serve_height_plausible": "serve contact height is outside the declared range",
    "all_player_reaches_plausible": "at least one fitted contact exceeds player reach",
    "all_bounce_rays_agree": "at least one fitted bounce misses its labeled camera ray",
    "bidirectional_windows_supported": "an event-neighborhood direction window exceeds 16 px RMS",
}

DISPLAY_PLAYER_HEIGHT_M = 1.82


class MissingNativeFrames(ValueError):
    """An attempt with no native frames on disk; skipped by the export with a receipt."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda _constant: None)


def _write_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            point3d._clean_json(value),
            handle,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_path(binding: dict[str, Any], data_root: Path, repo: Path) -> Path:
    base = binding.get("path_base")
    if base == "repository":
        return repo / binding["path"]
    if base == "TENNIS_DATA_ROOT":
        return data_root / binding["path"]
    path = Path(binding["path"])
    if path.is_absolute():
        return path
    raise ValueError(f"unsupported binding base {base!r}: {binding}")


def _verify_binding(binding: dict[str, Any], data_root: Path, repo: Path) -> Path:
    path = _binding_path(binding, data_root, repo)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = binding.get("sha256") or (binding.get("source") or {}).get("sha256")
    actual = _sha256(path)
    if expected and actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return path


def _round(value: Any, digits: int = 4) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _lerp(left: float, right: float, fraction: float) -> float:
    return left + (right - left) * fraction


def _lerp_vector(left: list[float], right: list[float], fraction: float) -> list[float]:
    return [_lerp(float(a), float(b), fraction) for a, b in zip(left, right, strict=True)]


def _norm(values: list[float] | None) -> float | None:
    if not values or len(values) != 3:
        return None
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _declared_native_clock(label: dict[str, Any]) -> bool:
    return label.get("schema") == "s6_labeled_validation_observation_document_v1"


def _explicit_native_pts(label: dict[str, Any]) -> bool:
    return any(
        "source_pts" in image or "source_time_base" in image
        for image in label.get("source_pack", {}).get("images", [])
    )


def _exact_native_timebase(
    label: dict[str, Any],
) -> tuple[dict[str, Any], Callable[[float], float]]:
    """Map the prepared zero-based picture inventory without regularizing its PTS."""
    pack = label["source_pack"]
    clock = pack.get("native_clock") or {}
    if (
        clock.get("original_pts_preserved") is not True
        or clock.get("pictures_resampled") is not False
    ):
        raise ValueError("declared native clock must preserve original pictures and PTS")
    images = pack.get("images", [])
    if len(images) < 2 or len({r.get("clip") for r in images}) != 1:
        raise ValueError("one complete native clip clock required")
    times = []
    for frame, row in enumerate(images):
        if type(row.get("frame")) is not int or row["frame"] != frame:
            raise ValueError("declared zero-based native image inventory has a gap or duplicate")
        pts = Fraction(str(row["source_pts"]))
        tick = Fraction(row["source_time_base"])
        if pts.denominator != 1 or tick <= 0:
            raise ValueError("invalid original native PTS/time base")
        time = pts * tick
        if not math.isclose(float(time), float(row["timestamp_seconds"]), rel_tol=0, abs_tol=1e-9):
            raise ValueError("native timestamp and source PTS conflict")
        if row.get("native_pts_seconds") is not None and not math.isclose(
            float(time), float(row["native_pts_seconds"]), rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("native timestamp aliases conflict")
        if times and time <= times[-1]:
            raise ValueError("native pictures must have strictly increasing original PTS")
        times.append(time)
    fps = float(pack["fps"])
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isclose(fps, float(Fraction(clock["modeled_fps_exact"])), rel_tol=1e-12)
    ):
        raise ValueError("declared modeled FPS aliases conflict")
    if Fraction(clock["original_start_seconds_exact"]) != times[0]:
        raise ValueError("declared clock start conflicts with original PTS")

    def native_time(frame: float) -> float:
        value = float(frame)
        if not math.isfinite(value) or value < -1e-8 or value > len(times) - 1 + 1e-8:
            raise ValueError("frame is outside the declared original native picture clock")
        value = min(len(times) - 1, max(0.0, value))
        lower = math.floor(value)
        upper = min(lower + 1, len(times) - 1)
        fraction = Fraction(str(value - lower))
        return float(times[lower] + fraction * (times[upper] - times[lower]))

    return (
        {
            "fps": fps,
            "one_based_frames": False,
            "origin_frame": 0,
            "origin_t": float(times[0]),
            "mapping": "piecewise_linear_original_native_pts",
            "native_frames": list(range(len(times))),
            "native_times_seconds": [float(t) for t in times],
            "original_clock": clock,
            "fractional_frames": "interpolated epoch between original adjacent pictures; no new picture",
        },
        native_time,
    )


def _native_timebase(label: dict[str, Any]) -> tuple[dict[str, Any], Callable[[float], float]]:
    if _declared_native_clock(label):
        return _exact_native_timebase(label)
    fps = float(label["source_pack"]["fps"])
    images = label["source_pack"].get("images", [])
    if _explicit_native_pts(label):
        # Modern automatic-addressed packets retain original integer PTS too.
        # Reuse the exact mapper without shifting pictures or rounding their epochs.
        origin = images[0].get("frame")
        if type(origin) is not int:
            raise ValueError("explicit native clock requires integer picture addresses")
        normalized = []
        for index, image in enumerate(images):
            if type(image.get("frame")) is not int or image["frame"] != origin + index:
                raise ValueError("explicit native image inventory has a gap or duplicate")
            if "source_pts" not in image or "source_time_base" not in image:
                raise ValueError("explicit original native PTS must cover every picture")
            time = Fraction(str(image["source_pts"])) * Fraction(image["source_time_base"])
            normalized.append(
                dict(
                    image,
                    frame=index,
                    timestamp_seconds=image.get("timestamp_seconds", float(time)),
                )
            )
        first = normalized[0]
        clock = dict(
            original_pts_preserved=True,
            pictures_resampled=False,
            modeled_fps_exact=str(Fraction(str(fps))),
            original_start_seconds_exact=str(
                Fraction(str(first["source_pts"])) * Fraction(first["source_time_base"])
            ),
        )
        metadata, time_at_zero_address = _exact_native_timebase(
            {"source_pack": dict(fps=fps, images=normalized, native_clock=clock)}
        )
        metadata.update(
            one_based_frames=origin == 1,
            origin_frame=origin,
            native_frames=[image["frame"] for image in images],
        )
        return metadata, lambda frame: time_at_zero_address(float(frame) - origin)
    offsets = [
        float(image["native_pts_seconds"]) - (float(image["frame"]) - 1.0) / fps
        for image in images
        if isinstance(image.get("native_pts_seconds"), (int, float))
    ]
    if not offsets:
        raise ValueError("label pack has no native PTS anchors")
    offset = sum(offsets) / len(offsets)
    maximum_residual = max(abs(value - offset) for value in offsets)
    if maximum_residual > 1e-5:
        raise ValueError(f"label pack is not regular at declared {fps} fps: {maximum_residual}")

    def native_time(frame: float) -> float:
        return offset + (float(frame) - 1.0) / fps

    return (
        {
            "fps": fps,
            "one_based_frames": True,
            "origin_frame": 1,
            "origin_t": _round(offset, 8),
            "maximum_bound_pts_residual_s": _round(maximum_residual, 9),
            "mapping": "origin_t + (frame - 1) / fps",
        },
        native_time,
    )


def _ball_record(label: dict[str, Any], source_clip: str) -> dict[str, Any]:
    for record in label.get("ball", {}).get("records", []):
        if record.get("clip") == source_clip:
            return record
    raise ValueError(f"no ball record for {source_clip}")


def _sample_flight(
    flight: dict[str, Any], frame: float
) -> tuple[list[float], list[float] | None] | None:
    start = float(flight["start_frame"])
    end = float(flight["end_frame"])
    if frame < start - 1e-8 or frame > end + 1e-8:
        return None
    positions = flight.get("positions") or []
    velocities = flight.get("velocities") or []
    if not positions:
        return None
    if end <= start or len(positions) == 1:
        velocity = velocities[0] if velocities else None
        return list(positions[0]), list(velocity) if velocity else None
    coordinate = (frame - start) / (end - start) * (len(positions) - 1)
    lower = max(0, min(len(positions) - 1, int(math.floor(coordinate))))
    upper = max(0, min(len(positions) - 1, lower + 1))
    fraction = coordinate - lower
    position = _lerp_vector(positions[lower], positions[upper], fraction)
    velocity = None
    if len(velocities) == len(positions):
        velocity = _lerp_vector(velocities[lower], velocities[upper], fraction)
    return position, velocity


def _sample_candidate(
    candidate: dict[str, Any], frame: int
) -> tuple[int, list[float], list[float] | None] | None:
    for index, flight in enumerate(candidate["measurement"].get("dense_flights", [])):
        sample = _sample_flight(flight, float(frame))
        if sample is not None:
            return index, sample[0], sample[1]
    return None


def _candidate_frames(
    candidate: dict[str, Any],
    frame_lo: int,
    frame_hi: int,
    native_time: Callable[[float], float],
    method: str,
) -> dict[str, list[Any]]:
    projection = {
        int(row["frame"]): row
        for row in candidate["measurement"].get("native_projection", [])
        if isinstance(row.get("frame"), (int, float))
    }
    keys = (
        "frame",
        "t",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "speed",
        "spin_x",
        "spin_y",
        "spin_z",
        "spin_mag",
        "conf",
        "ci_xy",
        "ci_z",
        "spin_ci",
        "rms_px",
        "segment",
        "method",
    )
    columns: dict[str, list[Any]] = {key: [] for key in keys}
    for frame in range(frame_lo, frame_hi + 1):
        sample = _sample_candidate(candidate, frame)
        row = projection.get(frame, {})
        columns["frame"].append(frame)
        columns["t"].append(_round(native_time(frame), 6))
        if sample is None:
            position: list[float | None] = [None, None, None]
            velocity: list[float | None] = [None, None, None]
            segment = None
            row_method = "native_context"
        else:
            segment_index, raw_position, raw_velocity = sample
            position = [_round(value, 4) for value in raw_position]
            velocity = [_round(value, 4) for value in (raw_velocity or [None, None, None])]
            segment = f"flight_{segment_index:02d}"
            row_method = method
        columns["x"].append(position[0])
        columns["y"].append(position[1])
        columns["z"].append(position[2])
        columns["vx"].append(velocity[0])
        columns["vy"].append(velocity[1])
        columns["vz"].append(velocity[2])
        columns["speed"].append(_round(_norm(raw_velocity if sample else None), 4))
        for key in ("spin_x", "spin_y", "spin_z", "spin_mag", "conf", "ci_xy", "ci_z", "spin_ci"):
            columns[key].append(None)
        columns["rms_px"].append(_round(row.get("error_px"), 3))
        columns["segment"].append(segment)
        columns["method"].append(row_method)
    return columns


def _candidate_for_depth(report: dict[str, Any], depth: float) -> dict[str, Any]:
    for candidate in report.get("refined_candidates", []):
        if math.isclose(float(candidate["depth_hypothesis_m"]), float(depth), abs_tol=1e-8):
            return candidate
    raise ValueError(f"no refined candidate for depth {depth}")


def _chosen_candidate(report: dict[str, Any], accepted: bool) -> dict[str, Any]:
    if not accepted:
        candidate = report.get("diagnostic_candidate")
        if not candidate:
            raise ValueError("held attempt has no diagnostic_candidate")
        return candidate
    candidate = (report.get("selected_arms") or {}).get("combined_toss_and_serve_prior")
    if not candidate:
        candidate = report.get("selected")
    if not candidate:
        raise ValueError("reconstructed attempt has no selected candidate")
    return candidate


def _sparse_report_players(
    report: dict[str, Any], topology: dict[str, Any], native_time: Callable[[float], float]
) -> dict[str, Any]:
    player_names = _player_names(topology)
    output: dict[str, Any] = {
        "names": player_names,
        "source": "connected_search_report.player_states (packet-bound)",
    }
    from cv.experiments.connected_shooting import player_position

    for side in ("near", "far"):
        # A player state whose court position is explicitly absent has no track
        # point to draw; it is omitted rather than plotted at an invented place.
        rows = sorted(
            (
                row
                for row in report.get("player_states", [])
                if row.get("side") == side and player_position.root_available(row)
            ),
            key=lambda row: row["frame"],
        )
        output[side] = {
            "frame": [int(row["frame"]) for row in rows],
            "t": [_round(native_time(row["frame"]), 6) for row in rows],
            "x": [_round(row["court_centre_xy_m"][0], 3) for row in rows],
            "y": [_round(row["court_centre_xy_m"][1], 3) for row in rows],
            "conf": [None for _row in rows],
            "track_id": ["packet_player_state" for _row in rows],
            "pose_status": ["no_pose_source" for _row in rows],
            "root_jump_m": [None for _row in rows],
        }
    return output


def _frame_number(value: Any) -> int:
    text = str(value)
    if text.startswith("f_"):
        text = text[2:].split(".", 1)[0]
    return int(round(float(text)))


def _finite_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _local_joints(
    joints: dict[str, tuple[float, float, float, float]], root_xy: tuple[float, float]
) -> dict[str, list[float]]:
    """Convert metric court joints to the player group's root-relative coordinates."""
    return {
        name: [
            _round(point[0] - root_xy[0], 4),
            _round(point[1] - root_xy[1], 4),
            _round(point[2], 4),
            _round(point[3], 4),
        ]
        for name, point in joints.items()
    }


def _statures_by_side(report: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = {"near": [], "far": []}
    for row in report.get("player_states", []):
        side = row.get("side")
        stature = row.get("stature_m")
        if side in values and isinstance(stature, (int, float)) and 1.2 <= stature <= 2.3:
            values[side].append(float(stature))
    return {
        side: statistics.median(rows) if rows else DISPLAY_PLAYER_HEIGHT_M
        for side, rows in values.items()
    }


def _contact_frames_by_side(
    contacts: list[dict[str, Any]], accepted: bool
) -> dict[tuple[int, str], dict[str, Any]]:
    if not accepted:
        return {}
    output = {}
    for contact in contacts:
        if contact.get("side") not in {"near", "far"}:
            continue
        frame = int(math.floor(float(contact["frame"]) + 0.5))
        xyz = [contact.get(axis) for axis in ("x", "y", "z")]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in xyz):
            output[(frame, contact["side"])] = contact
    return output


def _load_toss_documents(root: Path | None) -> list[tuple[dict[str, Any], Path]]:
    if root is None:
        return []
    documents = []
    for path in sorted(root.glob("*_toss_v1.json")):
        payload = _read_json(path)
        if payload.get("schema") != "s6_agent_serve_toss_labels_v1":
            raise ValueError(f"unexpected toss-label schema: {path}")
        if payload.get("automatic_inference_eligible") is not False:
            raise ValueError(f"toss label lacks evaluation-only provenance: {path}")
        documents.append((payload, path))
    return documents


def _toss_for_attempt(
    attempt: dict[str, Any], documents: list[tuple[dict[str, Any], Path]]
) -> tuple[dict[str, Any], Path] | None:
    metadata = attempt.get("metadata") or {}
    match_id = metadata.get("match_id")
    clip = metadata.get("clip")
    ordinal = "attempt02" if str(attempt.get("key", "")).endswith("_a02") else "attempt01"
    candidates = [
        item
        for item in documents
        if item[0].get("match_id") == match_id
        and item[0].get("clip") == clip
        and ordinal in str(item[0].get("attempt_id", ""))
    ]
    return candidates[0] if len(candidates) == 1 else None


def _contact_on_stature_ray(
    projection: Any,
    pixel: tuple[float, float],
    root_xy: tuple[float, float],
    stature_m: float,
) -> tuple[float, float, float] | None:
    """Turn a toss contact pixel into a conservative monocular XYZ estimate."""
    height = 1.45 * stature_m
    centre, direction = pose_lift.camera_ray(
        np.asarray(projection, dtype=float), pixel, (*root_xy, height)
    )
    if abs(float(direction[2])) < 1e-8:
        return None
    depth = (height - float(centre[2])) / float(direction[2])
    if depth <= 0:
        return None
    point = centre + depth * direction
    return tuple(float(value) for value in point)


def _serve_window_for_side(
    side: str,
    topology: dict[str, Any],
    contacts: list[dict[str, Any]],
    accepted: bool,
    side_rows: list[tuple[int, dict[str, Any]]],
    projections: dict[int, Any],
    stature_m: float,
    toss_document: tuple[dict[str, Any], Path] | None,
) -> tuple[pose_lift.ServeWindow | None, dict[str, Any] | None]:
    if side != topology.get("server_camera_half"):
        return None, None
    serve = next(
        (
            contact
            for contact in contacts
            if contact.get("phase") == "serve" and contact.get("side") == side
        ),
        None,
    )
    toss = toss_document[0] if toss_document else None
    toss_path = toss_document[1] if toss_document else None
    contact_frame = float(serve["frame"]) if serve is not None else None
    contact_xyz = None
    source = None
    if accepted and serve is not None:
        xyz = [serve.get(axis) for axis in ("x", "y", "z")]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in xyz):
            contact_xyz = tuple(float(value) for value in xyz)
            source = "accepted_fitter_contact_xyz"
    if contact_frame is None and toss is not None:
        contact_frame = _finite_float(toss.get("contact") or {}, "frame")
    if contact_xyz is None and toss is not None:
        face = toss.get("racket_face_centre_at_contact") or {}
        observation_frame = face.get("observation_frame")
        x_pixel = _finite_float(face, "x1080")
        y_pixel = _finite_float(face, "y1080")
        if (
            isinstance(observation_frame, (int, float))
            and x_pixel is not None
            and y_pixel is not None
        ):
            frame = int(observation_frame)
            pose_row = next((row for row_frame, row in side_rows if row_frame == frame), None)
            projection = projections.get(frame)
            if pose_row is not None and projection is not None:
                root_x = _finite_float(pose_row, "court_x")
                root_y = _finite_float(pose_row, "court_y")
                if root_x is not None and root_y is not None:
                    contact_xyz = _contact_on_stature_ray(
                        projection, (x_pixel, y_pixel), (root_x, root_y), stature_m
                    )
                    source = "toss_racket_face_ray_at_1.45_stature"
    if contact_frame is None:
        return None, None
    release = None if toss is None else _finite_float(toss.get("release") or {}, "frame")
    start_frame = int(math.floor(release if release is not None else contact_frame - 15.0))
    end_frame = int(math.ceil(contact_frame + 15.0))
    window = pose_lift.ServeWindow(
        start_frame=start_frame,
        contact_frame=contact_frame,
        end_frame=end_frame,
        contact_xyz=contact_xyz,
        court_forward_y=1.0 if side == "near" else -1.0,
        anchor_source=source or "timing_only_no_contact_xyz",
    )
    receipt = {
        "start_frame": start_frame,
        "contact_frame": contact_frame,
        "search_end_frame": end_frame,
        "contact_xyz_m": contact_xyz,
        "contact_anchor_source": window.anchor_source,
        "timing_source": (
            "toss_release_to_contact" if release is not None else "fit_contact_plus_minus_15"
        ),
        "toss_label": (
            None if toss_path is None else {"path": str(toss_path), "sha256": _sha256(toss_path)}
        ),
    }
    return window, receipt


def _player_source_binding(report: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            binding
            for binding in report.get("inputs", [])
            if (
                (
                    Path(str(binding.get("path", ""))).name.startswith(
                        "player_pose_tracked_crop_native"
                    )
                    and Path(str(binding.get("path", ""))).name.endswith("_v1.csv")
                )
                or (
                    Path(str(binding.get("path", ""))).name.startswith("player_boxes_")
                    and Path(str(binding.get("path", ""))).name.endswith("_native_sided_v1.csv")
                )
            )
        ),
        None,
    )


def _players(
    report: dict[str, Any],
    topology: dict[str, Any],
    native_time: Callable[[float], float],
    frame_lo: int,
    frame_hi: int,
    data_root: Path,
    repo: Path,
    projections: dict[int, Any],
    contacts: list[dict[str, Any]],
    accepted: bool,
    toss_document: tuple[dict[str, Any], Path] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    binding = _player_source_binding(report)
    if binding is None:
        players = _sparse_report_players(report, topology, native_time)
        return players, {"near": [], "far": []}, {}
    source = _verify_binding(binding, data_root, repo)
    with source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    clip = report["clip"]
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for row in source_rows:
        if row.get("clip") != clip or row.get("side") not in {"near", "far"}:
            continue
        try:
            frame = _frame_number(row.get("frame"))
        except (TypeError, ValueError):
            continue
        if not frame_lo <= frame <= frame_hi:
            continue
        confidence = _finite_float(row, "conf") or 0.0
        key = (frame, row["side"])
        if key not in best or confidence > (_finite_float(best[key], "conf") or 0.0):
            best[key] = row

    names = _player_names(topology)
    has_pose_columns = bool(source_rows and f"{pose_lift.JOINT_NAMES[0]}_x" in source_rows[0])
    players: dict[str, Any] = {
        "names": names,
        "source": f"{source.name} (hash-bound connected-search input)",
    }
    skeletons: dict[str, list[dict[str, Any]]] = {"near": [], "far": []}
    statures = _statures_by_side(report)
    contact_frames = _contact_frames_by_side(contacts, accepted)
    serve_receipts: dict[str, Any] = {}
    track_diagnostics: dict[str, Any] = {}
    for side in ("near", "far"):
        side_rows = sorted(
            ((frame, row) for (frame, row_side), row in best.items() if row_side == side),
            key=lambda item: item[0],
        )
        emitted_rows = []
        for frame, row in side_rows:
            x = _finite_float(row, "court_x")
            y = _finite_float(row, "court_y")
            if x is None or y is None:
                continue
            emitted_rows.append((frame, row, x, y))
        root_inputs = [
            pose_lift.PoseFrameInput(
                frame=frame,
                row=row,
                projection=np.asarray(projections.get(frame, []), dtype=float),
                root_xy=(x, y),
            )
            for frame, row, x, y in emitted_rows
        ]
        unsafe_root_frames = pose_lift.discontinuous_root_frames(root_inputs)
        pose_inputs = [
            item
            for item in root_inputs
            if has_pose_columns
            and item.frame in projections
            and item.frame not in unsafe_root_frames
        ]
        serve_window, serve_receipt = _serve_window_for_side(
            side,
            topology,
            contacts,
            accepted,
            side_rows,
            projections,
            statures[side],
            toss_document,
        )
        if serve_receipt is not None:
            serve_receipts[side] = serve_receipt
        lifted_frames = (
            pose_lift.lift_pose_sequence(
                pose_inputs,
                statures[side],
                serve_window=serve_window,
            )
            if pose_inputs
            else {}
        )
        pose_status = {}
        for frame, _row, _x, _y in emitted_rows:
            if frame in unsafe_root_frames:
                pose_status[frame] = "root_jump_gt_2m"
            elif not has_pose_columns:
                pose_status[frame] = "no_pose_source"
            elif frame not in projections:
                pose_status[frame] = "no_camera"
            elif frame in lifted_frames:
                pose_status[frame] = "pose"
            else:
                pose_status[frame] = "lift_rejected"
        for frame, row, x, y in emitted_rows:
            lifted = lifted_frames.get(frame)
            if lifted is not None:
                contact = contact_frames.get((frame, side))
                contact_xyz = (
                    None if contact is None else [float(contact[axis]) for axis in ("x", "y", "z")]
                )
                joints, racket = pose_lift.add_racket_segment(
                    lifted.joints,
                    contact_xyz=contact_xyz,
                    hand=lifted.striking_hand,
                )
                local_joints = _local_joints(joints, (x, y))
                skeletons[side].append(
                    {
                        "frame": frame,
                        "t": _round(native_time(frame), 6),
                        "track_id": row.get("track_id") or "",
                        "joints": local_joints,
                        "source": "native-image pose lifted on calibrated camera rays",
                        "method": lifted.method,
                        "stature_m": _round(statures[side], 3),
                        "support": lifted.support,
                        "airborne_height_m": _round(lifted.airborne_height_m, 4),
                        "kinematic_cost_rms": _round(lifted.cost_rms, 4),
                        "kinematic_converged": lifted.converged,
                        "serve_phase": lifted.serve_phase,
                        "striking_hand": lifted.striking_hand,
                        "lift_root_xy_m": [_round(value, 4) for value in lifted.root_xy]
                        if lifted.root_xy is not None
                        else None,
                        "root_step_m": _round(lifted.root_step_m, 4),
                        "contact_anchor_error_m": _round(lifted.contact_anchor_error_m, 6),
                        "racket": {
                            **racket,
                            "wrist_body_plane_distance_m": _round(
                                racket.get("wrist_body_plane_distance_m"), 4
                            ),
                        },
                    }
                )
        players[side] = {
            "frame": [row[0] for row in emitted_rows],
            "t": [_round(native_time(row[0]), 6) for row in emitted_rows],
            "x": [_round(row[2], 3) for row in emitted_rows],
            "y": [_round(row[3], 3) for row in emitted_rows],
            "conf": [_round(_finite_float(row[1], "conf"), 4) for row in emitted_rows],
            "track_id": [row[1].get("track_id") or "" for row in emitted_rows],
            "pose_status": [pose_status[row[0]] for row in emitted_rows],
            "root_jump_m": [_round(unsafe_root_frames.get(row[0]), 4) for row in emitted_rows],
        }
        statuses = list(pose_status.values())
        track_ids = [str(row[1].get("track_id") or "") for row in emitted_rows]
        track_diagnostics[side] = {
            "position_rows": len(emitted_rows),
            "frame_range": ([emitted_rows[0][0], emitted_rows[-1][0]] if emitted_rows else None),
            "track_id_counts": {
                track_id: track_ids.count(track_id) for track_id in sorted(set(track_ids))
            },
            "pose_status_counts": {
                status: statuses.count(status) for status in sorted(set(statuses))
            },
            "root_jump_limit_m_per_native_frame": pose_lift.MAXIMUM_INPUT_ROOT_JUMP_M,
            "root_jump_frames": [
                {"frame": frame, "distance_m": _round(distance, 4)}
                for frame, distance in sorted(unsafe_root_frames.items())
            ],
        }
    receipt = {
        "path": os.path.relpath(
            source, data_root if binding.get("path_base") == "TENNIS_DATA_ROOT" else repo
        ),
        "path_base": binding.get("path_base"),
        "sha256": binding.get("sha256"),
        "has_pose_keypoints": has_pose_columns,
        "lift_method": "camera_ray_serve_continuity_kinematic_v2",
        "projection_frames": len(projections),
        "statures_m": statures,
        "serve_windows": serve_receipts,
        "track_diagnostics": track_diagnostics,
        "labels_or_reviewed_inputs": [
            receipt["toss_label"]
            for receipt in serve_receipts.values()
            if receipt.get("toss_label") is not None
        ],
    }
    return players, skeletons, receipt


def _serve_pose_audit(
    skeletons: dict[str, list[dict[str, Any]]],
    players: dict[str, Any],
    projections: dict[int, Any],
    topology: dict[str, Any],
    contacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build native-overlay and side-elevation evidence for every serve frame."""
    side = topology.get("server_camera_half")
    if side not in {"near", "far"}:
        return None
    player_rows = {
        int(frame): (float(x), float(y))
        for frame, x, y in zip(
            players.get(side, {}).get("frame", []),
            players.get(side, {}).get("x", []),
            players.get(side, {}).get("y", []),
            strict=True,
        )
        if x is not None and y is not None
    }
    forward = 1.0 if side == "near" else -1.0
    frames = []
    for skeleton in skeletons.get(side, []):
        if not skeleton.get("serve_phase"):
            continue
        frame = int(skeleton["frame"])
        root = player_rows.get(frame)
        projection = projections.get(frame)
        if root is None or projection is None:
            continue
        absolute = {
            name: (float(point[0]) + root[0], float(point[1]) + root[1], float(point[2]))
            for name, point in skeleton["joints"].items()
        }
        native = {
            name: [_round(pixel[0], 2), _round(pixel[1], 2)]
            for name, point in absolute.items()
            if (pixel := pose_lift.project_point(np.asarray(projection, dtype=float), point))
            is not None
        }
        hips = [absolute[name] for name in ("left_hip", "right_hip") if name in absolute]
        if not hips:
            continue
        hip = np.mean(np.asarray(hips), axis=0)
        side_elevation = {
            name: [_round(forward * (point[1] - hip[1]), 4), _round(point[2], 4)]
            for name, point in absolute.items()
        }
        frames.append(
            {
                "frame": frame,
                "native_joints": native,
                "side_elevation_m": side_elevation,
                "root_step_m": skeleton.get("root_step_m"),
                "support": skeleton.get("support"),
                "striking_hand": skeleton.get("striking_hand"),
                "contact_anchor_error_m": skeleton.get("contact_anchor_error_m"),
            }
        )
    serve_contact = next((row for row in contacts if row.get("phase") == "serve"), None)
    contact_lean = None
    contact_frame = None
    contact_anchor_frame = None
    if serve_contact is not None:
        contact_frame = int(math.floor(float(serve_contact["frame"]) + 0.5))
        serve_skeletons = [row for row in skeletons.get(side, []) if row.get("serve_phase")]
        contact_skeleton = next(
            (
                row
                for row in serve_skeletons
                if isinstance(row.get("contact_anchor_error_m"), (int, float))
            ),
            None,
        )
        if contact_skeleton is None and serve_skeletons:
            contact_skeleton = min(
                serve_skeletons, key=lambda row: abs(int(row["frame"]) - contact_frame)
            )
        if contact_skeleton is not None:
            contact_anchor_frame = int(contact_skeleton["frame"])
            contact_lean = _torso_lean_at_contact(contact_skeleton["joints"], side)
    steps = [
        float(row["root_step_m"])
        for row in frames
        if isinstance(row.get("root_step_m"), (int, float))
    ]
    return {
        "schema": "serve_pose_audit_v1",
        "side": side,
        "contact_frame": contact_frame,
        "contact_anchor_frame": contact_anchor_frame,
        "contact_lean": contact_lean,
        "maximum_root_step_m": round(max(steps), 4) if steps else None,
        "root_step_limit_m": pose_lift.MAXIMUM_ROOT_STEP_M,
        "edges": [list(edge) for edge in POSE_EDGES],
        "frames": frames,
    }


def _player_names(topology: dict[str, Any]) -> dict[str, str | None]:
    match = str(topology.get("match_id", ""))
    pieces = match.split("_")
    candidates = [piece.title() for piece in pieces[-2:]] if len(pieces) >= 2 else []
    server = topology.get("server")
    server_side = topology.get("server_camera_half")
    if not server or server_side not in ("near", "far"):
        return {"near": None, "far": None}
    other = next((name for name in candidates if name.lower() != str(server).lower()), None)
    return {
        server_side: str(server),
        "far" if server_side == "near" else "near": other,
    }


def _nearest_player_side(report: dict[str, Any], frame: float) -> str:
    rows = report.get("player_states", [])
    if not rows:
        return "unknown"
    return min(rows, key=lambda row: abs(float(row["frame"]) - frame)).get("side", "unknown")


def _right_boundary_contact(
    candidate: dict[str, Any],
    report: dict[str, Any],
    native_time: Callable[[float], float],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """The original contact that closes a contact-prefix scene, incoming side only.

    Such a scene owns one more original contact than measured flight: the last
    contact is the modeled right boundary, and nothing after it was ever fitted.
    Its incoming position and velocity are the measured end of the final flight;
    the outgoing velocity stays explicitly absent rather than being copied from
    the incoming side or invented from a flight that does not exist.
    """
    events = [event for event in report.get("events", []) if event.get("event_type") == "contact"]
    flights = candidate["measurement"].get("dense_flights", [])
    if not flights or len(events) != len(flights) + 1:
        raise ValueError("a right-boundary contact needs one original contact past the last flight")
    event = events[-1]
    if float(event["frame"]) != float(boundary["frame"]):
        raise ValueError("right-boundary contact differs from the bound original contact")
    flight = flights[-1]
    positions = flight.get("positions") or []
    if not positions or abs(float(flight["end_frame"]) - float(event["frame"])) > 1e-6:
        raise ValueError("measured prefix must end at the original right contact")
    velocities = flight.get("velocities") or []
    velocity_in = velocities[-1] if velocities else None
    xyz = positions[-1]
    interval = event.get("frame_interval") or [None, None]
    fitted_frame = float(flight["end_frame"])
    return {
        "index": len(flights),
        "frame": _round(fitted_frame, 3),
        "fitted_frame": _round(fitted_frame, 3),
        "labeled_frame": _round(event.get("frame"), 3),
        "frame_lo": interval[0],
        "frame_hi": interval[1],
        "t": _round(native_time(fitted_frame), 6),
        "labeled_t": _round(native_time(event["frame"]), 6),
        "timing_source": "original source contact at the modeled right boundary",
        "timing_residual_px": None,
        "side": (report.get("right_contact_player") or {}).get("side", "unknown"),
        "phase": "rally",
        "status": "incoming_only",
        "fit": True,
        "boundary_kind": "original_contact",
        "outgoing_flight_modeled": False,
        "boundary_note": (
            "original contact closing the modeled prefix; the outgoing flight was never "
            "supported by enough native ball/camera rows to model and is not a point ending"
        ),
        "x": _round(xyz[0], 4),
        "y": _round(xyz[1], 4),
        "z": _round(xyz[2], 4),
        "ci_x": None,
        "ci_z": None,
        "speed_in": _round(_norm(velocity_in), 3),
        "speed_out": None,
        "v_in": [_round(value, 4) for value in (velocity_in or [None, None, None])],
        "v_out": [None, None, None],
        "velocity_ci": None,
        "racket_normal": [None, None, None],
        "racket_normal_speed": None,
        "racket_face_yaw_deg": None,
        "racket_face_pitch_deg": None,
        "racket_normal_ci_deg": None,
        "racket_speed_ci": None,
        "rms_in_px": None,
        "rms_out_px": None,
        "spin_out": {"mag": None, "x": None, "y": None, "z": None, "ci": None, "frame": None},
        "label_note": event.get("note"),
        "annotation_origin": event.get("annotation_origin"),
    }


def _contacts(
    candidate: dict[str, Any],
    report: dict[str, Any],
    native_time: Callable[[float], float],
    *,
    right_boundary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = [event for event in report.get("events", []) if event.get("event_type") == "contact"]
    flights = candidate["measurement"].get("dense_flights", [])
    xyz_rows = candidate["measurement"].get("contact_xyz", [])
    output = []
    for index, event in enumerate(events):
        if index >= len(flights) or index >= len(xyz_rows):
            break
        fitted_frame = float(flights[index]["start_frame"])
        xyz = xyz_rows[index]
        velocity_out = (flights[index].get("velocities") or [None])[0]
        velocity_in = None
        if index:
            prior = flights[index - 1].get("velocities") or []
            velocity_in = prior[-1] if prior else None
        interval = event.get("frame_interval") or [None, None]
        output.append(
            {
                "index": index,
                "frame": _round(fitted_frame, 3),
                "fitted_frame": _round(fitted_frame, 3),
                "labeled_frame": _round(event.get("frame"), 3),
                "frame_lo": interval[0],
                "frame_hi": interval[1],
                "t": _round(native_time(fitted_frame), 6),
                "labeled_t": _round(native_time(event["frame"]), 6),
                "timing_source": "connected fit; label retained separately",
                "timing_residual_px": None,
                "side": _nearest_player_side(report, float(event["frame"])),
                "phase": "serve" if index == 0 else "rally",
                "status": "fit",
                "fit": True,
                "x": _round(xyz[0], 4),
                "y": _round(xyz[1], 4),
                "z": _round(xyz[2], 4),
                "ci_x": None,
                "ci_z": None,
                "speed_in": _round(_norm(velocity_in), 3),
                "speed_out": _round(_norm(velocity_out), 3),
                "v_in": [_round(value, 4) for value in (velocity_in or [None, None, None])],
                "v_out": [_round(value, 4) for value in (velocity_out or [None, None, None])],
                "velocity_ci": None,
                "racket_normal": [None, None, None],
                "racket_normal_speed": None,
                "racket_face_yaw_deg": None,
                "racket_face_pitch_deg": None,
                "racket_normal_ci_deg": None,
                "racket_speed_ci": None,
                "rms_in_px": None,
                "rms_out_px": None,
                "spin_out": {
                    "mag": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "ci": None,
                    "frame": None,
                },
                "label_note": event.get("note"),
                "annotation_origin": event.get("annotation_origin"),
            }
        )
    if right_boundary is not None:
        output.append(_right_boundary_contact(candidate, report, native_time, right_boundary))
    return output


def _modeled_bounces(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    flights = candidate["measurement"].get("dense_flights", [])
    evidence_by_flight = candidate.get("evidence", {}).get("bounce_xyz_m", [])
    physical = candidate["measurement"].get("physical") or {}
    terminal = physical.get("terminal_completion") or {}
    output = []
    terminal_xyz = terminal.get("end_xyz")
    # The diagnostic fit may bounce several physically impossible times within one flight.
    # ``evidence.bounce_xyz_m`` is the search report's explicit fitted-event selection, one
    # list per labeled flight; choose those positions rather than displaying incidental later
    # impacts as if they were labels.  A terminal completion can lie just beyond the dense
    # display horizon (AO1 after its net hit), so retain its separately fitted impact.
    if evidence_by_flight:
        for flight_index, expected_rows in enumerate(evidence_by_flight):
            dense = flights[flight_index].get("bounces", []) if flight_index < len(flights) else []
            for expected in expected_rows:
                source = None
                if dense:
                    source = min(
                        dense,
                        key=lambda row: math.dist(
                            [float(value) for value in row.get("x", [0, 0, 0])],
                            [float(value) for value in expected],
                        ),
                    )
                    if math.dist(source["x"], expected) > 1e-4:
                        source = None
                if source is not None:
                    output.append({**source, "flight_index": flight_index})
                elif terminal_xyz and math.dist(terminal_xyz, expected) <= 1e-4:
                    output.append(
                        {
                            "frame": terminal["impact_frame"],
                            "x": terminal_xyz,
                            "regime": "terminal",
                            "flight_index": flight_index,
                        }
                    )
        return sorted(output, key=lambda row: row["frame"])
    return sorted(
        (
            {**bounce, "flight_index": flight_index}
            for flight_index, flight in enumerate(flights)
            for bounce in flight.get("bounces", [])
        ),
        key=lambda row: row["frame"],
    )


def _bounces(
    candidate: dict[str, Any], report: dict[str, Any], native_time: Callable[[float], float]
) -> list[dict[str, Any]]:
    labels = [event for event in report.get("events", []) if event.get("event_type") == "bounce"]
    output = []
    for index, bounce in enumerate(_modeled_bounces(candidate)):
        xyz = bounce.get("x") or [None, None, None]
        label = labels[index] if index < len(labels) else {}
        interval = label.get("frame_interval") or [None, None]
        x, y = xyz[:2]
        output.append(
            {
                "segment": f"flight_{bounce.get('flight_index', index):02d}"
                if bounce.get("flight_index") is not None
                else None,
                "frame": _round(bounce.get("frame"), 4),
                "fitted_frame": _round(bounce.get("frame"), 4),
                "labeled_frame": _round(label.get("frame"), 3),
                "frame_lo": interval[0],
                "frame_hi": interval[1],
                "t": _round(native_time(bounce["frame"]), 6),
                "labeled_t": _round(native_time(label["frame"]), 6) if label else None,
                "x": _round(x, 4),
                "y": _round(y, 4),
                "z": _round(xyz[2], 4),
                "v_in": [_round(value, 4) for value in bounce.get("v_in", [None, None, None])],
                "v_out": [_round(value, 4) for value in bounce.get("v_out", [None, None, None])],
                "regime": bounce.get("regime"),
                "rms_px": None,
                "in_court": (
                    isinstance(x, (int, float))
                    and isinstance(y, (int, float))
                    and -0.3 <= x <= point3d.COURT_W + 0.3
                    and -0.3 <= y <= point3d.COURT_L + 0.3
                ),
                "label_note": label.get("note"),
                "annotation_origin": label.get("annotation_origin"),
            }
        )
    return output


def _net_collisions(
    candidate: dict[str, Any], native_time: Callable[[float], float]
) -> list[dict[str, Any]]:
    output = []
    for flight in candidate["measurement"].get("dense_flights", []):
        for collision in flight.get("net_hits", []):
            xyz = collision.get("x") or [None, None, None]
            output.append(
                {
                    "frame": _round(collision.get("frame"), 4),
                    "labeled_frame": _round(collision.get("supplied_frame"), 3),
                    "t": _round(native_time(collision["frame"]), 6),
                    "x": _round(xyz[0], 4),
                    "y": _round(xyz[1], 4),
                    "z": _round(xyz[2], 4),
                    "position_continuous": collision.get("position_continuous"),
                }
            )
    return output


def _blocker_text(attempt: dict[str, Any]) -> str:
    blockers = attempt.get("blocking_bounds") or []
    if not blockers:
        return "The whole surviving family is complete, continuous, and input-only legal."
    details = [BLOCKER_LABELS.get(blocker, blocker.replace("_", " ")) for blocker in blockers]
    if "all_bounce_rays_agree" in blockers and isinstance(
        attempt.get("bounce_ray_margin_m"), (int, float)
    ):
        excess = max(0.0, -float(attempt["bounce_ray_margin_m"]))
        details.append(f"worst bounce-ray excess {excess:.3f} m")
    if "all_player_reaches_plausible" in blockers:
        margins = attempt.get("player_reach_margins_m") or []
        if margins:
            details.append(f"worst player-reach excess {max(0.0, -min(margins)):.3f} m")
    if "bidirectional_windows_supported" in blockers and isinstance(
        attempt.get("directional_window_margin_px"), (int, float)
    ):
        details.append(
            f"directional-window excess {max(0.0, -float(attempt['directional_window_margin_px'])):.3f} px"
        )
    return "; ".join(details) + "."


def _camera_summary(
    report: dict[str, Any], data_root: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[int, Any]]:
    binding = next(
        (
            item
            for item in report.get("inputs", [])
            if str(item.get("path", "")).endswith("/cameras.json")
        ),
        None,
    )
    if not binding:
        return {}, {}, {}
    path = _verify_binding(binding, data_root, repo)
    camera = _read_json(path)
    anchor = camera.get("anchor") or {}
    fit = anchor.get("fit") or {}
    summary = {
        "schema": camera.get("schema"),
        "transport_reference_policy": camera.get("transport_reference_policy"),
        "supported_frames": camera.get("supported"),
        "total_frames": camera.get("total"),
        "anchor_frame": anchor.get("frame"),
        "focal_native_px": _round(fit.get("focal_native_px"), 3),
        "camera_center_m": [_round(value, 4) for value in fit.get("camera_center_m", [])],
        "ground_rms_px": _round(fit.get("native_rms_px"), 3),
        "airborne_metric_accuracy_certified": camera.get(
            "airborne_metric_accuracy_certified", False
        ),
    }
    receipt = {
        "path": os.path.relpath(path, data_root),
        "path_base": "TENNIS_DATA_ROOT",
        "sha256": binding.get("sha256"),
    }
    projections = {
        int(row["frame"]): row["P"]
        for row in camera.get("cameras", [])
        if row.get("status") == "supported"
        and isinstance(row.get("frame"), (int, float))
        and isinstance(row.get("P"), list)
    }
    anchor_projection = fit.get("P")
    anchor_frame = anchor.get("frame")
    if isinstance(anchor_frame, (int, float)) and isinstance(anchor_projection, list):
        projections.setdefault(int(anchor_frame), anchor_projection)
    summary["projection_frames"] = len(projections)
    return summary, receipt, projections


def _link_declared_native_frames(
    key: str,
    label: dict[str, Any],
    source_clip: str,
    frame_lo: int,
    frame_hi: int,
    portal_root: Path,
    data_root: Path,
    native_time: Callable[[float], float],
) -> tuple[dict[str, Any], str]:
    """Copy exact nested source bindings to viewer names; never infer source filenames."""
    images = [r for r in label["source_pack"]["images"] if r.get("clip") == source_clip]
    bindings = {r["frame"]: r for r in images}
    if len(bindings) != len(images) or not set(range(frame_lo, frame_hi + 1)) <= bindings.keys():
        raise ValueError("requested playback lacks declared original native images")
    paths = {}
    for frame, image in bindings.items():
        binding = image.get("source") or {}
        absolute_shared = Path(binding.get("path", "")).is_absolute() and Path(
            binding["path"]
        ).resolve().is_relative_to(data_root.resolve())
        if (
            binding.get("path_base") != "TENNIS_DATA_ROOT" and not absolute_shared
        ) or not binding.get("sha256"):
            raise ValueError("native image requires an exact shared source binding")
        path = _verify_binding(binding, data_root, REPO)
        if (
            image.get("image_url") is not None
            and (data_root / image["image_url"]).resolve() != path.resolve()
        ):
            raise ValueError("native image source path aliases conflict")
        paths[frame] = path
    suffixes = {p.suffix for p in paths.values()}
    if len(suffixes) != 1:
        raise ValueError("one native image format required per clip")
    suffix = next(iter(suffixes))
    destination = portal_root / "frames" / key
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(frame_lo, frame_hi + 1):
        image, source = bindings[frame], paths[frame]
        target = destination / f"f_{frame:04d}{suffix}"
        if target.exists() or target.is_symlink():
            if not target.exists() or not os.path.samefile(target, source):
                target.unlink()
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        rows.append(
            {
                "frame": frame,
                "native_pts_seconds": native_time(frame),
                "source_pts": image["source_pts"],
                "source_time_base": image["source_time_base"],
                "file": target.name,
                "source": os.path.relpath(source, data_root),
                "sha256": image["source"]["sha256"],
                "label_bound": True,
                "label_bound_sha256": image["source"]["sha256"],
            }
        )
    manifest = {
        "schema": "connected_native_frame_manifest_v1",
        "attempt_key": key,
        "source_clip": source_clip,
        "frame_range": [frame_lo, frame_hi],
        "count": len(rows),
        "all_label_pack_images_verified": True,
        "frames": rows,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest, pretty=True)
    return manifest, _sha256(manifest_path)


def _link_frames(
    key: str,
    label: dict[str, Any],
    source_clip: str,
    frame_lo: int,
    frame_hi: int,
    portal_root: Path,
    data_root: Path,
    native_time: Callable[[float], float],
) -> tuple[dict[str, Any], str]:
    if _declared_native_clock(label) or _explicit_native_pts(label):
        return _link_declared_native_frames(
            key, label, source_clip, frame_lo, frame_hi, portal_root, data_root, native_time
        )
    image_bindings = {
        int(image["frame"]): image
        for image in label["source_pack"].get("images", [])
        if image.get("clip") == source_clip
    }
    for image in image_bindings.values():
        path = data_root / image["image_url"]
        expected = (image.get("source") or {}).get("sha256")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source image binding failed: {path}")
    ball_images = [
        image for image in image_bindings.values() if frame_lo <= int(image["frame"]) <= frame_hi
    ]
    if not ball_images:
        raise ValueError(f"no source images for {key}")
    source_directory = (data_root / ball_images[0]["image_url"]).parent
    suffixes = {Path(image["image_url"]).suffix for image in image_bindings.values()}
    if len(suffixes) != 1:
        raise ValueError("one native image format required per clip")
    suffix = next(iter(suffixes))
    destination = portal_root / "frames" / key
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(frame_lo, frame_hi + 1):
        source = source_directory / f"f_{frame:04d}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / source.name
        if target.exists() or target.is_symlink():
            if target.stat().st_ino != source.stat().st_ino:
                target.unlink()
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        actual_sha = _sha256(source)
        binding = image_bindings.get(frame)
        rows.append(
            {
                "frame": frame,
                "native_pts_seconds": _round(native_time(frame), 6),
                "file": source.name,
                "source": os.path.relpath(source, data_root),
                "sha256": actual_sha,
                "label_bound": binding is not None,
                "label_bound_sha256": (binding.get("source") or {}).get("sha256")
                if binding
                else None,
            }
        )
    manifest = {
        "schema": "connected_native_frame_manifest_v1",
        "attempt_key": key,
        "source_clip": source_clip,
        "frame_range": [frame_lo, frame_hi],
        "count": len(rows),
        "all_label_pack_images_verified": True,
        "frames": rows,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest, pretty=True)
    return manifest, _sha256(manifest_path)


def _video_overlay(
    label: dict[str, Any],
    candidate: dict[str, Any],
    source_clip: str,
    frame_lo: int,
    frame_hi: int,
) -> dict[str, Any]:
    record = _ball_record(label, source_clip)
    labels = {int(row["frame"]): row for row in record.get("frames", [])}
    predictions = {
        int(row["frame"]): row
        for row in candidate["measurement"].get("native_projection", [])
        if isinstance(row.get("frame"), (int, float))
    }
    frames = []
    for frame in range(frame_lo, frame_hi + 1):
        observed = labels.get(frame, {})
        predicted = predictions.get(frame, {})
        visible = observed.get("status") == "visible"
        frames.append(
            {
                "frame": frame,
                "labeled_front": [
                    _round(observed.get("x1080"), 2),
                    _round(observed.get("y1080"), 2),
                ]
                if visible
                else None,
                "labeled_status": observed.get("status"),
                "labeled_radius_px": observed.get("uncertainty_radius_px1080"),
                "fitted_projection": [
                    _round(predicted.get("predicted", [None, None])[0], 2),
                    _round(predicted.get("predicted", [None, None])[1], 2),
                ]
                if predicted.get("predicted")
                else None,
                "fit_error_px": _round(predicted.get("error_px"), 3),
                "fit_split": predicted.get("split"),
            }
        )
    size = label.get("native_size") or label["source_pack"].get("native_size") or [1920, 1080]
    if isinstance(size, dict):
        size = [size.get("width"), size.get("height")]
    # Controlled input swaps retain human-derived clip/evaluation provenance,
    # even when their ball stream comes from a detector. Describe that stream
    # without declaring the whole run automatic or calling centres human fronts.
    ball_origin = label.get("stream_origins", {}).get("ball")
    ball_convention = label.get("ball_convention")
    marker = "blue dot: frozen human leading visible front"
    if ball_origin == "automatic":
        marker = "blue dot: automatic observed native detector center"
    elif ball_convention is not None:
        marker = "blue dot: supplied observed native ball position"
    return {
        "schema": "connected_video_ball_overlay_v1",
        "native_size": size,
        "labeled_marker": marker,
        "fitted_marker": "yellow dot: reprojected fitted ball centre",
        "frames": frames,
    }


def _timeline_ticks(
    report: dict[str, Any],
    contacts: list[dict[str, Any]],
    native_time: Callable[[float], float],
) -> list[dict[str, Any]]:
    """Expose labeled contact/bounce epochs and fitted contacts on one native axis."""
    ticks = [
        {
            "frame": _round(event["frame"], 3),
            "t": _round(native_time(event["frame"]), 6),
            "kind": event["event_type"],
            "source": "label",
        }
        for event in report.get("events", [])
        if event.get("event_type") in {"contact", "bounce"}
        and isinstance(event.get("frame"), (int, float))
    ]
    ticks.extend(
        {
            "frame": _round(contact["fitted_frame"], 3),
            "t": _round(native_time(contact["fitted_frame"]), 6),
            "kind": "contact",
            "source": "fit",
        }
        for contact in contacts
        if isinstance(contact.get("fitted_frame"), (int, float))
    )
    return sorted(ticks, key=lambda row: (row["frame"], row["source"], row["kind"]))


def _player_state_ledger(
    report: dict[str, Any],
    data_root: Path,
    native_time: Callable[[float], float],
    frame_lo: int,
    frame_hi: int,
) -> dict[str, Any] | None:
    """Read this attempt's ``tennis_player_state_v1`` sidecar for the viewer.

    Its own sidecar contract asks the viewer to render the position sigma and
    the contact-only racket observations, and to treat a blank as an abstention.
    A blank row is therefore skipped rather than drawn at the origin.
    """
    match_id = report.get("match_id") or report["attempt_id"].split("__", 1)[-1].split("_pt", 1)[0]
    clip = report["clip"]
    path = (
        data_root
        / "processed/playerledger/ledger_v1/states"
        / match_id
        / f"{clip}_player_state_v1.csv"
    )
    if not path.is_file():
        return None
    positions: dict[str, list[dict[str, Any]]] = {"near": [], "far": []}
    rackets: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip") != clip or row.get("side") not in positions:
                continue
            try:
                frame = int(float(row["frame"]))
            except (TypeError, ValueError):
                continue
            if not frame_lo <= frame <= frame_hi:
                continue
            if row.get("court_x") and row.get("court_y"):
                positions[row["side"]].append(
                    {
                        "frame": frame,
                        "t": _round(native_time(frame), 4),
                        "court_xy_m": [
                            _round(float(row["court_x"]), 3),
                            _round(float(row["court_y"]), 3),
                        ],
                        "sigma_m": _round(
                            float(row["court_sigma_m"]) if row.get("court_sigma_m") else None, 3
                        ),
                        "source": row.get("position_source"),
                        "hip_height_proxy_m": _round(
                            float(row["hip_height_proxy_m"])
                            if row.get("hip_height_proxy_m")
                            else None,
                            3,
                        ),
                    }
                )
            if row.get("racket_face_court_x") and row.get("racket_face_court_y"):
                rackets.append(
                    {
                        "frame": frame,
                        "t": _round(native_time(frame), 4),
                        "side": row["side"],
                        "court_xy_m": [
                            _round(float(row["racket_face_court_x"]), 3),
                            _round(float(row["racket_face_court_y"]), 3),
                        ],
                        "image_xy_native": [
                            _round(float(row["racket_face_x_native"]), 1)
                            if row.get("racket_face_x_native")
                            else None,
                            _round(float(row["racket_face_y_native"]), 1)
                            if row.get("racket_face_y_native")
                            else None,
                        ],
                        "sigma_px": _round(
                            float(row["racket_face_sigma_px"])
                            if row.get("racket_face_sigma_px")
                            else None,
                            1,
                        ),
                        "stroke_type": row.get("stroke_type") or None,
                        "source": row.get("racket_face_source"),
                    }
                )
    return {
        "schema": "tennis_player_state_v1",
        "source": os.path.relpath(path, data_root),
        "source_sha256": _sha256(path),
        "positions": positions,
        "contact_rackets": rackets,
        "semantics": (
            "court positions from sided boxes with a stated sigma; a blank row is the "
            "ledger's abstention and is omitted; the racket face is a contact-only "
            "elbow-wrist extrapolation, not a measured face normal"
        ),
    }


def export_attempt(
    attempt: dict[str, Any],
    aggregate: dict[str, Any],
    topology: dict[str, Any],
    report_path: Path,
    label_path: Path,
    out_dir: Path,
    data_root: Path,
    repo: Path,
    context_frames: int,
    per_flight: dict[str, Any] | None = None,
    toss_document: tuple[dict[str, Any], Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = attempt["key"]
    report = _read_json(report_path)
    label = _read_json(label_path)
    accepted = bool(attempt["reconstructed"])
    candidate = _chosen_candidate(report, accepted)
    selected_depth = float(candidate["depth_hypothesis_m"])
    if accepted and not math.isclose(
        selected_depth, float(attempt["family"]["selected_depth_y_m"])
    ):
        raise ValueError(f"selected depth mismatch for {key}")

    source_clip = report["clip"]
    ball_record = _ball_record(label, source_clip)
    labeled_frames = [int(row["frame"]) for row in ball_record.get("frames", [])]
    label_window_frames = [
        int(image["frame"])
        for image in label["source_pack"].get("images", [])
        if image.get("clip") == source_clip and isinstance(image.get("frame"), (int, float))
    ]
    source_path = data_root / label["source_pack"]["images"][0]["image_url"]
    available = sorted(
        int(path.stem.split("_")[-1])
        for path in source_path.parent.glob("f_*.jpg")
        if path.stem.split("_")[-1].isdigit()
    )
    if not available or not labeled_frames or not label_window_frames:
        # No native frames on disk for this attempt (or no ball rows): the wing cannot
        # scrub it.  Refuse this attempt with a reason instead of aborting the export.
        raise MissingNativeFrames(
            f"{key}: {len(available)} native frames under {source_path.parent}, "
            f"{len(labeled_frames)} labeled ball frames"
        )
    # The source-pack image span is the labeled review window. It includes label
    # abstentions and the passive pictures after the terminal event, so it—not
    # the shorter fitted-flight span—is the scrubbable contract. Add only real
    # decoded pictures around it; native time and cadence remain unchanged.
    label_window = [min(label_window_frames), max(label_window_frames)]
    frame_lo = max(min(available), label_window[0] - context_frames)
    frame_hi = min(max(available), label_window[1] + context_frames)
    timebase, native_time = _native_timebase(label)

    portal_root = out_dir.parent
    frame_manifest, frame_manifest_sha = _link_frames(
        key, label, source_clip, frame_lo, frame_hi, portal_root, data_root, native_time
    )
    primary_frames = _candidate_frames(
        candidate,
        frame_lo,
        frame_hi,
        native_time,
        "connected_fit" if accepted else "diagnostic_fit_not_accepted",
    )
    _mark_frame_acceptance(primary_frames, per_flight)
    acceptance = "not_scored_per_flight"
    if per_flight is not None:
        acceptance = (
            "complete"
            if per_flight["complete_point"]
            else "partial"
            if per_flight["partial_point"]
            else "no_accepted_flight"
        )

    alternates = []
    if accepted:
        for depth in attempt["family"].get("member_depths_m", []):
            if math.isclose(float(depth), selected_depth, abs_tol=1e-8):
                continue
            alternate = _candidate_for_depth(report, float(depth))
            alternates.append(
                {
                    "label": f"serve depth {float(depth):.1f} m",
                    "serve_depth_m": float(depth),
                    "selected": False,
                    "frames": _candidate_frames(
                        alternate, frame_lo, frame_hi, native_time, "family_alternate"
                    ),
                }
            )

    contacts = _contacts(candidate, report, native_time)
    bounces = _bounces(candidate, report, native_time)
    net_collisions = _net_collisions(candidate, native_time)
    camera, camera_receipt, projections = _camera_summary(report, data_root, repo)
    players, skeletons, player_source_receipt = _players(
        report,
        topology,
        native_time,
        frame_lo,
        frame_hi,
        data_root,
        repo,
        projections,
        contacts,
        accepted,
        toss_document,
    )
    serve_pose_audit = _serve_pose_audit(skeletons, players, projections, topology, contacts)
    blockers = attempt.get("blocking_bounds") or []
    blocker_text = _blocker_text(attempt)
    report_sha = _sha256(report_path)
    aggregate_sha = _sha256(Path(aggregate["_path"]))
    label_sha = _sha256(label_path)

    checks = attempt.get("selected_or_diagnostic_legal_checks") or {}
    # Only the frozen nine were reviewed by the owner; a discovered cohort's
    # attempt has a legal family and nothing more.  Do not label it reviewed.
    reviewed = (
        bool((topology or {}).get("topology_in_words"))
        and (topology or {}).get("source")
        != "synthesized from the sweep's explicit hitter order and the fitted toss side"
    )
    verdict = (
        ("reconstructed, owner-reviewed" if reviewed else "legal connected family")
        if accepted
        else "not accepted"
    )
    quality = {
        "accepted": accepted,
        "gate_accepted_overall": accepted,
        "per_flight_acceptance": acceptance,
        "accepted_flights": None if per_flight is None else per_flight["accepted_flight_count"],
        "flight_count": None if per_flight is None else per_flight["flight_count"],
        "tier1_labels_consumed": True,
        "clip": {
            "accepted": accepted,
            "rms_training_px": _round((attempt.get("rms_px") or {}).get("training"), 3),
            "rms_withheld_px": _round((attempt.get("rms_px") or {}).get("withheld"), 3),
            "maximum_directional_window_rms_px": _round(
                attempt.get("maximum_directional_window_rms_px"), 3
            ),
            "bounce_ray_margin_m": _round(attempt.get("bounce_ray_margin_m"), 4),
        },
        "gate": checks,
        "verdict": verdict,
        "trajectory_role": "selected_family_member" if accepted else "best_diagnostic_not_accepted",
        "diagnostic_depth_m": None if accepted else selected_depth,
        "blockers": blockers,
        "blocker_text": blocker_text,
        "not_independently_xyz_certified": True,
        "footnote": (
            "Owner review of this human-derived surviving family does not promote the method "
            "to production and is not independent XYZ certification."
        ),
    }

    doc = {
        "schema": SCHEMA,
        "extension_schema": EXTENSION_SCHEMA,
        "match": report["attempt_id"].split("__", 1)[-1].split("_pt", 1)[0],
        "clip": key,
        "source_clip": source_clip,
        "point": key,
        "attempt_id": report["attempt_id"],
        "tag": aggregate.get("cohort", "labeled_attempt_sweep"),
        "fps": timebase["fps"],
        "native_timebase": timebase,
        "force_follow": True,
        "trail": {"recent": 0.55, "ghost": 0.015, "fade": 0.25, "full": 0.95},
        "frames_base": "/portal/3d",
        "frames_dir": "frames",
        "frames_pattern": "f_%04d.jpg",
        "frames_available": True,
        "frame_range": [frame_lo, frame_hi],
        "review_frame_window": {
            "labeled": label_window,
            "passive_context": [frame_lo, frame_hi],
            "context_frames_each_side": context_frames,
            "semantics": (
                "full source-pack label window plus decoded native passive context; "
                "not truncated to fitted flights"
            ),
        },
        "court": COURT,
        "quality": quality,
        "family": {**attempt["family"], "selected_label": f"serve depth {selected_depth:.1f} m"},
        "per_flight": per_flight,
        "alternates": alternates,
        "topology": {
            "summary": topology.get("topology_in_words"),
            "shot_count": topology.get("shot_count"),
            "server": topology.get("server"),
            "serve_number": topology.get("serve_number"),
            "physical_event_sequence": topology.get("physical_event_sequence"),
        },
        "camera": camera,
        "counts": {
            "frames": sum(value is not None for value in primary_frames["x"]),
            "native_video_frames": frame_hi - frame_lo + 1,
            "contacts": len(contacts),
            "contacts_fit": len(contacts),
            "bounces": len(bounces),
            "bounces_in_court": sum(bool(bounce["in_court"]) for bounce in bounces),
            "net_collisions": len(net_collisions),
            "family_members": attempt["family"].get("count", 0),
            "alternate_trajectories": len(alternates),
            "skeleton_frames": {
                "near": len(skeletons["near"]),
                "far": len(skeletons["far"]),
            },
        },
        "spin_present": False,
        "confidence_status": "No calibrated per-frame 3D confidence; null is emitted rather than invented.",
        "speed_caveat": "Fitted physics speed; not an independently calibrated broadcast measurement.",
        "skeleton_schema": "metric_camera_ray_player_pose_v2",
        "skeleton_caveat": (
            "Tracked native-frame keypoints are lifted along calibrated camera rays with "
            "stature-scaled limb priors. During the serve, airborne root depth is interpolated "
            "between planted endpoints, torso lean is stabilized, and a physical striking-arm "
            "chain ends at the contact racket face. Accepted contacts use fitted contact XYZ; "
            "explicit toss labels can supply evaluation-only release/contact evidence. This monocular 3D is "
            "directionally useful, not independently certified anatomical truth; missing "
            "pose or camera frames stay missing."
        ),
        "frames": primary_frames,
        "contacts": contacts,
        "bounces": bounces,
        "net_collisions": net_collisions,
        "labeled_events": report.get("events", []),
        "timeline_ticks": _timeline_ticks(report, contacts, native_time),
        "players": players,
        "skeletons": skeletons,
        "serve_pose_audit": serve_pose_audit,
        "player_state_ledger": _player_state_ledger(
            report, data_root, native_time, frame_lo, frame_hi
        ),
        "video_overlay": {
            **_video_overlay(label, candidate, source_clip, frame_lo, frame_hi),
            "serve_pose_frames": [] if serve_pose_audit is None else serve_pose_audit["frames"],
            "serve_pose_edges": [list(edge) for edge in POSE_EDGES],
        },
        "provenance": {
            "human_derived": bool(aggregate.get("human_derived")),
            "automatic_inference_eligible": bool(aggregate.get("automatic_inference_eligible")),
            "independent_xyz_truth_available": bool(
                aggregate.get("independent_xyz_truth_available")
            ),
            "aggregate_report": {
                "path": os.path.relpath(aggregate["_path"], data_root),
                "sha256": aggregate_sha,
            },
            "attempt_report": {
                "path": os.path.relpath(report_path, data_root),
                "sha256": report_sha,
            },
            "fixed_configuration_sha256": aggregate["fixed_configuration_sha256"],
            "label_pack": {
                "path": os.path.relpath(label_path, repo),
                "sha256": label_sha,
                "annotation_origin": label.get("annotation_origin"),
            },
            "camera": camera_receipt,
            "player_tracking": player_source_receipt,
            "frame_manifest": {
                "path": f"frames/{key}/manifest.json",
                "sha256": frame_manifest_sha,
                "all_label_pack_images_verified": frame_manifest["all_label_pack_images_verified"],
            },
        },
    }
    filename = f"connected_{key}.json"
    _write_json(out_dir / filename, doc)
    index_entry = {
        "clip": key,
        "point": key,
        "match": doc["match"],
        "file": filename,
        "tag": doc["tag"],
        "collection": f"connected_{aggregate.get('cohort', 'sweep')}_sweep",
        "label": (
            f"{key.upper()} · {verdict}"
            if per_flight is None
            else f"{key.upper()} · {verdict} · "
            f"{acceptance} {per_flight['accepted_flight_count']}/{per_flight['flight_count']} flights"
        ),
        "verdict": verdict,
        "accepted": accepted,
        "blocker_text": blocker_text,
        "per_flight_acceptance": acceptance,
        "accepted_flights": None if per_flight is None else per_flight["accepted_flight_count"],
        "flight_count": None if per_flight is None else per_flight["flight_count"],
        "family_count": attempt["family"].get("count", 0),
        "family_range_m": attempt["family"].get("depth_y_range_m"),
        "family_width_m": attempt["family"].get("width_m"),
        "family_midpoint_m": attempt["family"].get("midpoint_m"),
        "selected_depth_m": selected_depth,
        "n_frames": doc["counts"]["frames"],
        "n_contacts": len(contacts),
        "n_contacts_fit": len(contacts),
        "n_bounces": len(bounces),
        "spin_present": False,
        "duration_s": _round(native_time(frame_hi) - native_time(frame_lo), 3),
        "frames_available": True,
        "skeleton_frames": len(skeletons["near"]) + len(skeletons["far"]),
        "serve_contact_lean_signed_toward_net_deg": (
            None
            if serve_pose_audit is None or serve_pose_audit.get("contact_lean") is None
            else serve_pose_audit["contact_lean"]["signed_toward_net_deg"]
        ),
        "serve_side": None if serve_pose_audit is None else serve_pose_audit["side"],
        "serve_maximum_root_step_m": (
            None if serve_pose_audit is None else serve_pose_audit["maximum_root_step_m"]
        ),
        "source_group": "curated_real_attempts",
    }
    return doc, index_entry


def _synthesized_topology(attempt: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Name the server and its camera half when no reviewed topology file exists.

    The reviewed nine carry a hand-written topology document.  A discovered
    cohort does not, so the two facts the viewer actually needs are taken from
    the sweep's own explicit hitter order and from the fitter's toss-server
    side.  Nothing here invents a shot description.
    """
    metadata = attempt.get("metadata") or {}
    statures = metadata.get("player_statures_m") or []
    server = str(statures[0][0]) if statures else None
    side = (report.get("toss_server_feet") or {}).get("side")
    return {
        "key": attempt["key"],
        "match_id": metadata.get("match_id", ""),
        "clip": metadata.get("clip"),
        "server": server,
        "server_camera_half": side if side in ("near", "far") else None,
        "topology_in_words": (metadata.get("discovery") or {}).get("topology_in_words"),
        "source": "synthesized from the sweep's explicit hitter order and the fitted toss side",
    }


def _per_flight_verdict(document: Path | None, rung: str) -> dict[str, Any] | None:
    """One attempt's per-flight verdict at one declared acceptance rung."""
    if document is None or not document.is_file():
        return None
    payload = _read_json(document)
    entry = next((row for row in payload["rungs"] if row["rung"] == rung), None)
    if entry is None:
        raise ValueError(f"{document} carries no rung named {rung}")
    verdict = entry["verdict"]
    return {
        "rung": rung,
        "thresholds": entry["thresholds"],
        "selected_depth_m": entry["selected_depth_m"],
        "flight_count": verdict["flight_count"],
        "accepted_flight_count": verdict["accepted_flight_count"],
        "accepted_flight_indices": verdict["accepted_flight_indices"],
        "rejected_flight_indices": verdict["rejected_flight_indices"],
        "complete_point": verdict["complete_point"],
        "partial_point": verdict["partial_point"],
        "maximum_junction_gap_m": verdict.get("maximum_junction_gap_m", 0.0),
        "gaps": verdict["gaps"],
        "flights": [
            {
                "flight_index": row["flight_index"],
                "role": row["role"],
                "accepted": row["accepted"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "failures": row["failures"],
                "reprojection_rms_px": _round(row["reprojection_rms_px"], 3),
                "maximum_window_rms_px": _round(row["maximum_window_rms_px"], 3),
                "worst_bounce_gate_distance_m": _round(
                    max(
                        (
                            entry["gate_distance_m"]
                            for entry in row["bounce_witness"]
                            if entry["gate_distance_m"] is not None
                        ),
                        default=None,
                    ),
                    4,
                ),
            }
            for row in verdict["flights"]
        ],
        "scope": (
            "opened human-derived development evidence; an accepted flight is not an "
            "independently certified 3D shot"
        ),
    }


def _mark_frame_acceptance(frames: dict[str, list[Any]], per_flight: dict[str, Any] | None) -> None:
    """Label every fitted exposure with the verdict of the flight it belongs to."""
    accepted = None if per_flight is None else set(per_flight["accepted_flight_indices"])
    column: list[Any] = []
    for segment in frames["segment"]:
        if segment is None or accepted is None:
            column.append(None)
            continue
        column.append("accepted" if int(str(segment).split("_")[-1]) in accepted else "rejected")
    frames["acceptance"] = column


def _ground_from_pixel(projection: Any, pixel: tuple[float, float]) -> tuple[float, float] | None:
    """Back-project a truth foot pixel to the court plane for evaluation."""
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        return None
    homography = matrix[:, [0, 1, 3]]
    if abs(float(np.linalg.det(homography))) < 1e-12:
        return None
    court = np.linalg.solve(homography, np.asarray([pixel[0], pixel[1], 1.0]))
    if abs(float(court[2])) < 1e-12:
        return None
    return float(court[0] / court[2]), float(court[1] / court[2])


def _distribution(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(np.max(array)), 4),
    }


def _torso_lean_at_contact(
    joints: Mapping[str, Sequence[float]], side: str
) -> dict[str, float] | None:
    shoulders = [
        np.asarray(joints[name][:3], dtype=float)
        for name in ("left_shoulder", "right_shoulder")
        if name in joints
    ]
    hips = [
        np.asarray(joints[name][:3], dtype=float)
        for name in ("left_hip", "right_hip")
        if name in joints
    ]
    if len(shoulders) != 2 or len(hips) != 2:
        return None
    delta = np.mean(shoulders, axis=0) - np.mean(hips, axis=0)
    if delta[2] <= 1e-8:
        return None
    forward = 1.0 if side == "near" else -1.0
    return {
        "signed_toward_net_deg": round(math.degrees(math.atan2(forward * delta[1], delta[2])), 3),
        "total_from_vertical_deg": round(
            math.degrees(math.atan2(float(np.linalg.norm(delta[:2])), delta[2])), 3
        ),
    }


def _paired_foot_errors(
    predicted: list[tuple[float, float]], truth: list[tuple[float, float]]
) -> list[float]:
    """Pair up to two feet by minimum total court distance.

    Truth schema generations do not share a reliable anatomical left/right
    convention, so the evaluation uses the best unordered physical pairing.
    """
    if not predicted or not truth:
        return []
    if len(predicted) == 1 or len(truth) == 1:
        return [
            min(
                math.dist(predicted_point, truth_point)
                for predicted_point in predicted
                for truth_point in truth
            )
        ]
    direct = [math.dist(predicted[0], truth[0]), math.dist(predicted[1], truth[1])]
    crossed = [math.dist(predicted[0], truth[1]), math.dist(predicted[1], truth[0])]
    return direct if sum(direct) <= sum(crossed) else crossed


def _truth_attempt(truth: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        attempt
        for attempt in aggregate.get("attempts", [])
        if (attempt.get("metadata") or {}).get("match_id") == truth["match_id"]
        and (attempt.get("metadata") or {}).get("clip") == truth["clip"]
    ]
    if len(candidates) == 1:
        return candidates[0]
    ordinal = "a02" if "attempt02" in truth["path"].stem else "a01"
    exact = [attempt for attempt in candidates if attempt["key"].endswith(ordinal)]
    return exact[0] if len(exact) == 1 else None


def score_player_truth(
    report_path: Path,
    labels_glob: str,
    output: Path,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    repo: Path = REPO,
    reach_threshold_m: float = 0.10,
    toss_label_root: Path | None = None,
) -> dict[str, Any]:
    """Score the lifted skeleton on dense evaluation-only player truth.

    The truth files are opened only inside this exporter-side scorer.  Neither
    ``pose_lift`` nor any automatic inference component receives a truth value.
    """
    from cv.validation import player_truth_ledger

    aggregate = _read_json(report_path)
    toss_documents = _load_toss_documents(toss_label_root)
    truths = [player_truth_ledger.load_truth(Path(path)) for path in sorted(glob.glob(labels_glob))]
    feet_errors: list[float] = []
    hip_errors: list[float] = []
    racket_errors: list[float] = []
    racket_by_source: dict[str, list[float]] = {}
    reach_distances: list[float] = []
    root_steps: list[float] = []
    lean_by_side: dict[str, list[float]] = {"near": [], "far": []}
    paired_errors: dict[str, tuple[list[float], list[float]]] = {
        "feet_error_m": ([], []),
        "hip_reprojection_error_px": ([], []),
        "racket_face_error_px": ([], []),
    }
    attempts = []
    receipts = []

    for truth in truths:
        attempt = _truth_attempt(truth, aggregate)
        if attempt is None or not attempt.get("report"):
            attempts.append(
                {
                    "label": str(truth["path"]),
                    "status": "abstain",
                    "reason": "no_unique_allref_attempt_report_and_camera",
                }
            )
            continue
        attempt_report_path = Path(attempt["report"])
        report = _read_json(attempt_report_path)
        _camera, camera_receipt, projections = _camera_summary(report, data_root, repo)
        player_binding = _player_source_binding(report)
        if not projections or player_binding is None:
            attempts.append(
                {
                    "attempt": attempt["key"],
                    "label": str(truth["path"]),
                    "status": "abstain",
                    "reason": "missing_projection_or_pose_binding",
                }
            )
            continue
        pose_path = _verify_binding(player_binding, data_root, repo)
        with pose_path.open(newline="", encoding="utf-8") as handle:
            pose_rows = list(csv.DictReader(handle))
        pose_index: dict[tuple[int, str], dict[str, Any]] = {}
        for row in pose_rows:
            if row.get("clip") != truth["clip"] or row.get("side") not in {"near", "far"}:
                continue
            frame = _frame_number(row.get("frame"))
            key = frame, row["side"]
            if key not in pose_index or (_finite_float(row, "conf") or 0.0) > (
                _finite_float(pose_index[key], "conf") or 0.0
            ):
                pose_index[key] = row

        accepted = bool(attempt.get("reconstructed"))
        candidate = _chosen_candidate(report, accepted)
        contacts = _contacts(candidate, report, lambda frame: float(frame))
        contact_frames = _contact_frames_by_side(contacts, accepted)
        statures = _statures_by_side(report)
        frame_results: dict[tuple[int, str], tuple[pose_lift.LiftedPose, dict[str, Any], dict]] = {}
        attempt_counts = {"feet": 0, "hips": 0, "rackets": 0, "reaches": 0}
        topology = _synthesized_topology(attempt, report)
        toss_document = _toss_for_attempt(attempt, toss_documents)

        for side in ("near", "far"):
            truth_by_frame = {row["frame"]: row["players"].get(side) for row in truth["frames"]}
            legacy_frames: dict[
                int, tuple[pose_lift.LiftedPose, dict[str, Any], dict[str, Any]]
            ] = {}
            legacy_previous_local = None
            legacy_previous_frame = None
            for frame in sorted(truth_by_frame):
                truth_player = truth_by_frame[frame]
                pose_row = pose_index.get((frame, side))
                projection = projections.get(frame)
                if truth_player is None or pose_row is None or projection is None:
                    legacy_previous_local = None
                    legacy_previous_frame = frame
                    continue
                x = _finite_float(pose_row, "court_x")
                y = _finite_float(pose_row, "court_y")
                if x is None or y is None:
                    legacy_previous_local = None
                    legacy_previous_frame = frame
                    continue
                if legacy_previous_frame is None or frame != legacy_previous_frame + 1:
                    legacy_previous_local = None
                legacy = pose_lift.lift_pose_frame(
                    pose_row,
                    projection,
                    (x, y),
                    statures[side],
                    previous_local=legacy_previous_local,
                )
                legacy_previous_frame = frame
                if legacy is None:
                    legacy_previous_local = None
                    continue
                legacy_contact = contact_frames.get((frame, side))
                legacy_contact_xyz = (
                    None
                    if legacy_contact is None
                    else [float(legacy_contact[axis]) for axis in ("x", "y", "z")]
                )
                legacy_joints, legacy_racket = pose_lift.add_racket_segment(
                    legacy.joints, contact_xyz=legacy_contact_xyz
                )
                legacy_frames[frame] = (legacy, legacy_joints, legacy_racket)
                legacy_previous_local = _local_joints(legacy.joints, (x, y))
            side_rows = sorted(
                (
                    (frame, row)
                    for (frame, row_side), row in pose_index.items()
                    if row_side == side
                    and frame in truth_by_frame
                    and truth_by_frame[frame] is not None
                ),
                key=lambda item: item[0],
            )
            pose_inputs = []
            for frame, pose_row in side_rows:
                projection = projections.get(frame)
                x = _finite_float(pose_row, "court_x")
                y = _finite_float(pose_row, "court_y")
                if projection is None or x is None or y is None:
                    continue
                pose_inputs.append(
                    pose_lift.PoseFrameInput(
                        frame=frame,
                        row=pose_row,
                        projection=np.asarray(projection, dtype=float),
                        root_xy=(x, y),
                    )
                )
            serve_window, _serve_receipt = _serve_window_for_side(
                side,
                topology,
                contacts,
                accepted,
                side_rows,
                projections,
                statures[side],
                toss_document,
            )
            lifted_frames = pose_lift.lift_pose_sequence(
                pose_inputs,
                statures[side],
                serve_window=serve_window,
            )
            for frame in sorted(truth_by_frame):
                truth_player = truth_by_frame[frame]
                pose_row = pose_index.get((frame, side))
                projection = projections.get(frame)
                if truth_player is None or pose_row is None or projection is None:
                    continue
                lifted = lifted_frames.get(frame)
                if lifted is None:
                    continue
                if lifted.serve_phase and lifted.root_step_m is not None:
                    root_steps.append(float(lifted.root_step_m))
                contact = contact_frames.get((frame, side))
                contact_xyz = (
                    None if contact is None else [float(contact[axis]) for axis in ("x", "y", "z")]
                )
                racket_joints, racket_receipt = pose_lift.add_racket_segment(
                    lifted.joints,
                    contact_xyz=contact_xyz,
                    hand=lifted.striking_hand,
                )
                frame_results[(frame, side)] = (lifted, racket_joints, racket_receipt)

                truth_feet = [
                    court
                    for foot in truth_player["feet"]
                    if foot["ground"] is not None
                    and (court := _ground_from_pixel(projection, foot["ground"])) is not None
                ]
                predicted_feet = [
                    tuple(lifted.joints[name][:2])
                    for name in ("left_ankle", "right_ankle")
                    if name in lifted.joints
                ]
                errors = _paired_foot_errors(predicted_feet, truth_feet)
                feet_errors.extend(errors)
                attempt_counts["feet"] += len(errors)
                legacy_result = legacy_frames.get(frame)
                if legacy_result is not None:
                    legacy_feet = [
                        tuple(legacy_result[0].joints[name][:2])
                        for name in ("left_ankle", "right_ankle")
                        if name in legacy_result[0].joints
                    ]
                    legacy_errors = _paired_foot_errors(legacy_feet, truth_feet)
                    if len(legacy_errors) == len(errors):
                        paired_errors["feet_error_m"][0].extend(legacy_errors)
                        paired_errors["feet_error_m"][1].extend(errors)

                truth_hip = truth_player.get("hip")
                hips = [
                    lifted.joints[name][:3]
                    for name in ("left_hip", "right_hip")
                    if name in lifted.joints
                ]
                if truth_hip is not None and hips:
                    predicted_hip = np.mean(np.asarray(hips), axis=0)
                    hip_pixel = pose_lift.project_point(projection, predicted_hip)
                    if hip_pixel is not None:
                        hip_error = math.dist(hip_pixel, truth_hip)
                        hip_errors.append(hip_error)
                        attempt_counts["hips"] += 1
                        if legacy_result is not None:
                            legacy_hips = [
                                legacy_result[0].joints[name][:3]
                                for name in ("left_hip", "right_hip")
                                if name in legacy_result[0].joints
                            ]
                            if legacy_hips:
                                legacy_pixel = pose_lift.project_point(
                                    projection, np.mean(np.asarray(legacy_hips), axis=0)
                                )
                                if legacy_pixel is not None:
                                    paired_errors["hip_reprojection_error_px"][0].append(
                                        math.dist(legacy_pixel, truth_hip)
                                    )
                                    paired_errors["hip_reprojection_error_px"][1].append(hip_error)

                truth_racket = truth_player.get("racket")
                if truth_racket and truth_racket["active"] and truth_racket["face"] is not None:
                    racket_head = racket_joints.get("racket_head")
                    racket_pixel = (
                        None
                        if racket_head is None
                        else pose_lift.project_point(projection, racket_head[:3])
                    )
                    if racket_pixel is not None:
                        error = math.dist(racket_pixel, truth_racket["face"])
                        racket_errors.append(error)
                        source = str(racket_receipt.get("source"))
                        racket_by_source.setdefault(source, []).append(error)
                        attempt_counts["rackets"] += 1
                        if legacy_result is not None:
                            legacy_head = legacy_result[1].get("racket_head")
                            legacy_pixel = (
                                None
                                if legacy_head is None
                                else pose_lift.project_point(projection, legacy_head[:3])
                            )
                            if legacy_pixel is not None:
                                paired_errors["racket_face_error_px"][0].append(
                                    math.dist(legacy_pixel, truth_racket["face"])
                                )
                                paired_errors["racket_face_error_px"][1].append(error)

        for truth_contact in truth["contacts"]:
            side = truth_contact.get("end")
            if side not in {"near", "far"}:
                continue
            frame = int(math.floor(float(truth_contact["frame"]) + 0.5))
            result = frame_results.get((frame, side))
            if result is None:
                continue
            distance = result[2].get("wrist_body_plane_distance_m")
            if isinstance(distance, (int, float)) and math.isfinite(distance):
                reach_distances.append(float(distance))
                attempt_counts["reaches"] += 1

        serve_lean = None
        if attempt["key"] in OWNER_REVIEWED_SERVE_POINTS:
            serve_contact = next(
                (contact for contact in contacts if contact.get("phase") == "serve"), None
            )
            if serve_contact is not None and serve_contact.get("side") in {"near", "far"}:
                serve_side = str(serve_contact["side"])
                serve_frame = int(math.floor(float(serve_contact["frame"]) + 0.5))
                serve_candidates = [
                    (frame, result)
                    for (frame, side), result in frame_results.items()
                    if side == serve_side and result[0].serve_phase
                ]
                anchored = [
                    item
                    for item in serve_candidates
                    if isinstance(item[1][0].contact_anchor_error_m, (int, float))
                ]
                serve_result = (
                    anchored[0][1]
                    if anchored
                    else min(serve_candidates, key=lambda item: abs(item[0] - serve_frame))[1]
                    if serve_candidates
                    else None
                )
                if serve_result is not None:
                    serve_lean = _torso_lean_at_contact(serve_result[1], serve_side)
                    if serve_lean is not None:
                        lean_by_side[serve_side].append(serve_lean["signed_toward_net_deg"])

        attempts.append(
            {
                "attempt": attempt["key"],
                "label": str(truth["path"]),
                "status": "scored",
                "accepted_fitter_contact_policy": accepted,
                "counts": attempt_counts,
                "owner_reviewed_serve_lean": serve_lean,
            }
        )
        receipts.append(
            {
                "attempt": attempt["key"],
                "truth": truth["receipts"],
                "attempt_report": {
                    "path": str(attempt_report_path),
                    "sha256": _sha256(attempt_report_path),
                },
                "camera": camera_receipt,
                "pose": {"path": str(pose_path), "sha256": _sha256(pose_path)},
            }
        )

    beyond = sum(distance > reach_threshold_m for distance in reach_distances)
    payload = {
        "schema": "skeleton3d_player_truth_score_v2",
        "automatic_inference_eligible": False,
        "opened_development_data": True,
        "player_truth_labels_enter_pose_lift": False,
        "serve_toss_labels_enter_pose_lift": toss_label_root is not None,
        "configuration": {
            "pose_lift": "camera_ray_serve_continuity_kinematic_v2",
            "serve_root": "planted-endpoint smoothstep with 0.15 m/frame maximum step",
            "serve_arm": "stature-scaled two-link IK with racket face at explicit contact XYZ",
            "feet_pairing": "minimum-distance unordered assignment of grounded truth feet",
            "hip": "mean lifted hip projected through the per-frame camera to native pixels",
            "racket": (
                "accepted fitter contact XYZ only on its rounded native contact frame; "
                "otherwise the ledger-calibrated 1.5-forearm wrist extrapolation, capped at "
                "0.55 m, with 75 px ledger sigma"
            ),
            "body_plane": "least-squares plane through visible shoulders and hips",
            "beyond_body_plane_threshold_m": reach_threshold_m,
            "toss_label_root": None if toss_label_root is None else str(toss_label_root),
        },
        "denominator": {
            "truth_files_discovered": len(truths),
            "attempts_scored": sum(row["status"] == "scored" for row in attempts),
            "attempts_abstained": sum(row["status"] == "abstain" for row in attempts),
            "feet_scored": len(feet_errors),
            "hips_scored": len(hip_errors),
            "racket_faces_scored": len(racket_errors),
            "contacts_with_wrist_plane_distance": len(reach_distances),
        },
        "metrics": {
            "feet_error_m": _distribution(feet_errors),
            "hip_reprojection_error_px": _distribution(hip_errors),
            "racket_face_error_px": _distribution(racket_errors),
            "racket_face_error_by_source_px": {
                source: _distribution(values) for source, values in sorted(racket_by_source.items())
            },
            "wrist_body_plane_distance_m": _distribution(reach_distances),
            "wrist_beyond_body_plane": {
                "count": beyond,
                "denominator": len(reach_distances),
                "rate": round(beyond / len(reach_distances), 4) if reach_distances else None,
            },
            "serve_root_step_m": _distribution(root_steps),
            "owner_reviewed_serve_lean_signed_toward_net_deg": {
                side: _distribution(values) for side, values in lean_by_side.items()
            },
            "paired_before_after": {
                name: {
                    "before": _distribution(values[0]),
                    "after": _distribution(values[1]),
                }
                for name, values in paired_errors.items()
            },
        },
        "attempts": attempts,
        "receipts": receipts,
    }
    _write_json(output, payload, pretty=True)
    return payload


def export_all(
    report_path: Path,
    out_dir: Path,
    label_root: Path = DEFAULT_LABEL_ROOT,
    topologies_path: Path = DEFAULT_TOPOLOGIES,
    data_root: Path = DEFAULT_DATA_ROOT,
    repo: Path = REPO,
    context_frames: int = 10,
    per_flight_root: Path | None = None,
    per_flight_rung: str = "base",
    merge_index: bool = False,
    toss_label_root: Path | None = None,
) -> dict[str, Any]:
    aggregate = _read_json(report_path)
    aggregate["_path"] = str(report_path.resolve())
    if aggregate.get("schema") != "connected_labeled_attempt_sweep_v2":
        raise ValueError("connected export requires a labeled connected-family sweep report")
    if len(aggregate.get("attempts", [])) != aggregate.get("attempt_denominator"):
        raise ValueError("the report must carry one row per attempt in its denominator")
    if not aggregate.get("human_derived") or aggregate.get("automatic_inference_eligible"):
        raise ValueError("unexpected inference provenance on the connected sweep report")
    # The configuration lives beside its own report, whatever cohort produced it.
    config_path = report_path.parent / "fixed_configuration.json"
    config_binding = {
        "path": os.path.relpath(config_path, data_root),
        "path_base": "TENNIS_DATA_ROOT",
        "sha256": aggregate["fixed_configuration_sha256"],
    }
    _verify_binding(config_binding, data_root, repo)
    topologies = {item["key"]: item for item in _read_json(topologies_path).get("attempts", [])}
    toss_documents = _load_toss_documents(toss_label_root)
    input_by_name = {Path(binding["path"]).name: binding for binding in aggregate.get("inputs", [])}
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_attempts: list[dict[str, Any]] = []
    entries = []
    held_before_fitting = []
    for attempt in aggregate["attempts"]:
        key = attempt["key"]
        if not attempt.get("report"):
            # An attempt held on an input precondition never reached a depth
            # branch, so it has no fitted trajectory to scrub.  It stays in the
            # index's denominator with its blocker instead of disappearing.
            held_before_fitting.append(
                {
                    "key": key,
                    "blocker": attempt.get("input_precondition_blocker"),
                    "status": attempt.get("status"),
                }
            )
            continue
        report = Path(attempt["report"])
        attempt_report = _read_json(report)
        label_binding = next(
            item
            for item in attempt_report.get("inputs", [])
            if item.get("path_base") == "repository"
            and str(item.get("path", "")).startswith("cv/validation/labels/")
        )
        label_path = _verify_binding(label_binding, data_root, repo)
        aggregate_binding = input_by_name.get(label_path.name)
        if not aggregate_binding:
            raise ValueError(f"aggregate report does not bind {label_path.name}")
        _verify_binding(aggregate_binding, data_root, repo)
        if label_path.parent != label_root.resolve():
            raise ValueError(f"label escaped expected root: {label_path}")
        try:
            _doc, entry = export_attempt(
                attempt,
                aggregate,
                topologies.get(key) or _synthesized_topology(attempt, attempt_report),
                report,
                label_path,
                out_dir,
                data_root,
                repo,
                context_frames,
                _per_flight_verdict(
                    None if per_flight_root is None else per_flight_root / key / "per_flight.json",
                    per_flight_rung,
                ),
                _toss_for_attempt(attempt, toss_documents),
            )
        except MissingNativeFrames as error:
            skipped_attempts.append({"key": attempt.get("key"), "reason": str(error)})
            continue
        entries.append(entry)
        print(
            f"-> {entry['file']}  {entry['verdict']}  "
            f"family={entry['family_count']} frames={entry['n_frames']}"
        )
    index = {
        "schema": INDEX_SCHEMA,
        "extension_schema": "connected_point3d_index_v1",
        "collection": f"connected_{aggregate.get('cohort', 'sweep')}_sweep",
        "count": len(entries),
        "per_flight_rung": None if per_flight_root is None else per_flight_rung,
        "complete_point_count": sum(
            entry.get("per_flight_acceptance") == "complete" for entry in entries
        ),
        "partial_point_count": sum(
            entry.get("per_flight_acceptance") == "partial" for entry in entries
        ),
        "accepted_flight_count": sum(entry.get("accepted_flights") or 0 for entry in entries),
        "accepted_count": sum(entry["accepted"] for entry in entries),
        "held_count": (sum(not entry["accepted"] for entry in entries) + len(held_before_fitting)),
        "human_derived": True,
        "automatic_inference_eligible": False,
        "attempt_denominator": aggregate["attempt_denominator"],
        "owner_reviewed_serve_points": list(OWNER_REVIEWED_SERVE_POINTS),
        "owner_reviewed_serve_lean_signed_toward_net_deg": {
            side: _distribution(
                [
                    float(entry["serve_contact_lean_signed_toward_net_deg"])
                    for entry in entries
                    if entry.get("point") in OWNER_REVIEWED_SERVE_POINTS
                    and entry.get("serve_side") == side
                    and isinstance(
                        entry.get("serve_contact_lean_signed_toward_net_deg"), (int, float)
                    )
                ]
            )
            for side in ("near", "far")
        },
        "owner_reviewed_serve_maximum_root_step_m": max(
            (
                float(entry["serve_maximum_root_step_m"])
                for entry in entries
                if entry.get("point") in OWNER_REVIEWED_SERVE_POINTS
                and isinstance(entry.get("serve_maximum_root_step_m"), (int, float))
            ),
            default=None,
        ),
        "held_before_fitting": held_before_fitting,
        "points": entries,
    }
    if merge_index and (out_dir / "index.json").is_file():
        # The live wing carries one canonical index built from several
        # collections.  Replace only the rows this export owns, so another
        # lane's points keep their entries instead of disappearing.
        canonical = _read_json(out_dir / "index.json")
        if canonical.get("collection") == "canonical_3d_viewer":
            replacement = {entry["file"]: entry for entry in entries}
            merged = []
            for row in canonical.get("points", []):
                update = replacement.pop(row.get("file"), None)
                merged.append({**row, **update} if update else row)
            merged.extend(replacement.values())
            canonical["points"] = merged
            canonical["count"] = len(merged)
            _write_json(out_dir / "index.json", canonical, pretty=True)
        else:
            _write_json(out_dir / "index.json", index, pretty=True)
        _write_json(out_dir / "connected_index.json", index, pretty=True)
    else:
        _write_json(out_dir / "index.json", index, pretty=True)
    export_manifest = {
        "schema": "connected_3d_export_manifest_v1",
        "skipped_attempts": skipped_attempts,
        "report": os.path.relpath(report_path, data_root),
        "report_sha256": _sha256(report_path),
        "fixed_configuration_sha256": aggregate["fixed_configuration_sha256"],
        "attempt_count": len(entries),
        "accepted_count": index["accepted_count"],
        "held_count": index["held_count"],
        "context_frames_each_side": context_frames,
        "toss_label_root": None if toss_label_root is None else str(toss_label_root),
        "documents": [
            {"file": entry["file"], "sha256": _sha256(out_dir / entry["file"])} for entry in entries
        ],
    }
    _write_json(out_dir.parent / "export_manifest.json", export_manifest, pretty=True)
    print(f"-> index.json ({len(entries)} attempts; {index['accepted_count']} reconstructed)")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--topologies", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--context-frames", type=int, default=10)
    parser.add_argument(
        "--per-flight",
        type=Path,
        help="per-flight acceptance sweep output; tags each attempt complete/partial",
    )
    parser.add_argument("--per-flight-rung", default="base")
    parser.add_argument(
        "--toss-label-root",
        type=Path,
        help=(
            "explicit evaluation-only toss-label directory; omitted means no toss labels "
            "enter pose export"
        ),
    )
    parser.add_argument(
        "--merge-index",
        action="store_true",
        help="update this export's rows inside an existing canonical index instead of replacing it",
    )
    parser.add_argument(
        "--score-player-truth",
        type=Path,
        help="also write the evaluation-only dense player-truth score to this JSON path",
    )
    parser.add_argument(
        "--player-truth-labels",
        default=str(DEFAULT_LABEL_ROOT / "*_players_v1.json"),
        help="glob used only by --score-player-truth",
    )
    args = parser.parse_args()
    if args.context_frames < 0:
        parser.error("--context-frames must be nonnegative")
    export_all(
        args.report.resolve(),
        args.out.resolve(),
        args.label_root.resolve(),
        args.topologies.resolve(),
        args.data_root.resolve(),
        REPO,
        args.context_frames,
        None if args.per_flight is None else args.per_flight.resolve(),
        args.per_flight_rung,
        args.merge_index,
        None if args.toss_label_root is None else args.toss_label_root.resolve(),
    )
    if args.score_player_truth is not None:
        score = score_player_truth(
            args.report.resolve(),
            args.player_truth_labels,
            args.score_player_truth.resolve(),
            data_root=args.data_root.resolve(),
            repo=REPO,
            toss_label_root=(
                None if args.toss_label_root is None else args.toss_label_root.resolve()
            ),
        )
        print(json.dumps(score["denominator"], indent=2))
        print(json.dumps(score["metrics"], indent=2))
        print(f"-> {args.score_player_truth.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
