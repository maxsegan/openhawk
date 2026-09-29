"""Camera artifact schema transformations shared by canonical inference stages."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.camera_P_per_frame import transport_projection

FRAME_TRACK_ARTIFACT = "court_H_per_frame_v1.npz"
RECOVERABLE_REGISTRATION_SOURCES = {
    "registered_interpolated",
}
MAXIMUM_GROUND_RESIDUAL_PX = 4.0
MAXIMUM_NET_RESIDUAL_PX = 8.0
# Point-camera prefixes that are not a direct net solve. Tracked court
# registration still sits on the suffix after "+".
FALLBACK_POINT_CAMERA_PREFIXES = frozenset(
    {
        "intrinsic_fallback",
        "shared_intrinsic_fallback",
        "shared_net_fallback",
        "shared_net_focal_fallback",
    }
)
TRACKED_REGISTRATION_SUFFIXES = frozenset(
    {"registered", "registered_interpolated", "line_model_registered"}
)
STATIC_SHOT_REGISTRATION_SUFFIXES = frozenset({"anchor_static_fallback"})


def split_camera_source(source: str) -> tuple[str, str]:
    """Return (point prefix, registration suffix); suffix is empty when untagged."""
    prefix, separator, suffix = str(source).partition("+")
    return (prefix, suffix) if separator == "+" else (str(source), "")


def fallback_point_camera_source(source: str) -> bool:
    prefix, _suffix = split_camera_source(source)
    return prefix in FALLBACK_POINT_CAMERA_PREFIXES


MAXIMUM_RIG_PRIOR_ANCHORS = 24
RECONSTRUCTION_TRACK_ARTIFACT = "ball_track_joint_native1080_arc_augmented_v2.csv"
RECONSTRUCTION_CAMERA_ARTIFACT = "camera_P_per_frame_v1.npz"
RECONSTRUCTION_POINT_HOMOGRAPHY_ARTIFACT = "court_H_per_point.npz"
RECONSTRUCTION_FRAME_HOMOGRAPHY_ARTIFACT = "court_H_per_frame_v1.npz"
RECONSTRUCTION_CADENCE_SIDECAR = "audit_frames_native_1080.coordinates.json"


def registered_frame_reliable(
    *,
    point_reliable: bool,
    point_source: str,
    registration_reliable: bool,
    registration_source: str,
    ground_residual_px: float,
    net_residual_px: float,
) -> bool:
    """Recognize measured registration lineage with numerically valid geometry."""
    if not point_reliable or not str(point_source).startswith("direct"):
        return False
    if registration_reliable:
        return True
    return (
        registration_source in RECOVERABLE_REGISTRATION_SOURCES
        and np.isfinite(float(ground_residual_px))
        and float(ground_residual_px) <= MAXIMUM_GROUND_RESIDUAL_PX
        and np.isfinite(float(net_residual_px))
        and float(net_residual_px) <= MAXIMUM_NET_RESIDUAL_PX
    )


@dataclass(frozen=True)
class MatchRigPrior:
    """Automatic shared-centre rig used only where point height calibration abstained."""

    solution: Any
    image_size: tuple[int, int]
    net_residual_px: float
    source_points: tuple[int, ...]


def fit_match_rig_prior(
    match_out: Path,
    frames_dir: str,
    point_records: dict[int, dict[str, Any]],
) -> MatchRigPrior | None:
    """Fit camera_bundle on direct automatic anchors, preserving the transport default."""
    from cv.pipeline.camera_bundle import (
        collect_anchor_observations,
        fit_bundle,
        net_residuals,
        projection_matrix,
    )

    if not any(not bool(record["reliable"]) for record in point_records.values()):
        return None
    required = (
        match_out / "court_topology_evidence_v1.json",
        match_out / "court_H_per_point.npz",
        match_out / "camera_P_per_point.npz",
    )
    if not all(path.is_file() for path in required):
        return None
    observations, image_size, _ = collect_anchor_observations(match_out, frames_dir)
    observations = [
        row
        for row in observations
        if row.point in point_records
        and bool(point_records[row.point]["reliable"])
        and str(point_records[row.point]["source"]).startswith("direct")
        and row.net_pixels is not None
    ]
    if not observations:
        return None
    if len(observations) > MAXIMUM_RIG_PRIOR_ANCHORS:
        selected = np.linspace(0, len(observations) - 1, MAXIMUM_RIG_PRIOR_ANCHORS, dtype=np.int32)
        observations = [observations[int(index)] for index in selected]
    try:
        solution = fit_bundle(observations, image_size)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not solution.success:
        return None
    residuals = []
    for row in observations:
        yaw, tilt, focal = solution.view_parameters[row.point]
        projection = projection_matrix(solution.center, yaw, tilt, focal, image_size)
        values = net_residuals(projection, solution.k1, solution.dist_center, row.net_pixels)
        if len(values):
            residuals.append(float(np.sqrt(np.mean(np.square(values)))))
    if not residuals:
        return None
    net_residual = float(np.median(residuals))
    if not np.isfinite(net_residual) or net_residual > MAXIMUM_NET_RESIDUAL_PX:
        return None
    return MatchRigPrior(
        solution=solution,
        image_size=image_size,
        net_residual_px=net_residual,
        source_points=tuple(sorted(row.point for row in observations)),
    )


def rig_prior_projection(
    prior: MatchRigPrior,
    frame_homography: np.ndarray,
    point: int,
) -> tuple[np.ndarray, float] | None:
    """Fit one view to a target-supported ground plane under the shared-centre prior."""
    from cv.pipeline.camera_bundle import fit_frame_view

    if not np.isfinite(frame_homography).all() or abs(np.linalg.det(frame_homography)) < 1e-12:
        return None
    reference_point = min(prior.source_points, key=lambda value: abs(value - point))
    initial = prior.solution.view_parameters[reference_point]
    try:
        projection, _, ground = fit_frame_view(
            np.linalg.inv(frame_homography), prior.solution, prior.image_size, initial
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(projection).all() or not np.isfinite(ground):
        return None
    return projection, float(ground)


def compose_frame_projection(
    point_projection: np.ndarray,
    frame_homography: np.ndarray,
) -> np.ndarray | None:
    """Move a point-anchored projection onto one frame's registered ground plane.

    Registration models a camera that pans, tilts and zooms about a fixed centre, so the
    anchor and the frame differ by one image homography. ``transport_projection`` applies
    it to the whole projection, which keeps the net-calibrated vertical geometry instead of
    re-solving monocular height per frame. ``frame_homography`` is image-to-court, the
    convention of ``court_H_per_frame_v1.npz``.
    """
    projection = np.asarray(point_projection, dtype=float)
    homography = np.asarray(frame_homography, dtype=float)
    if not np.isfinite(projection).all() or not np.isfinite(homography).all():
        return None
    if abs(np.linalg.det(homography)) < 1e-12:
        return None
    if abs(np.linalg.det(projection[:, [0, 1, 3]])) < 1e-12:
        return None
    composed = transport_projection(projection, homography)
    return composed if np.isfinite(composed).all() else None


def anchor_to_frame_warp(
    point_projection: np.ndarray,
    frame_homography: np.ndarray,
) -> np.ndarray:
    """Image homography taking anchor-frame pixels to ``frame_homography``'s frame."""
    anchor = np.linalg.inv(np.asarray(point_projection, dtype=float)[:, [0, 1, 3]])
    return np.linalg.inv(np.asarray(frame_homography, dtype=float)) @ anchor


def warp_points(warp: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply an image homography to (N, 2) pixels, preserving NaN rows."""
    values = np.asarray(points, dtype=float)
    finite = np.isfinite(values).all(axis=1)
    out = np.full_like(values, np.nan)
    if not finite.any():
        return out
    homogeneous = np.column_stack((values[finite], np.ones(int(finite.sum()))))
    projected = (warp @ homogeneous.T).T
    denominators = projected[:, 2]
    usable = np.abs(denominators) > 1e-9
    rows = np.flatnonzero(finite)[usable]
    out[rows] = projected[usable, :2] / denominators[usable, None]
    return out


def _load_frame_track(match_out: Path) -> dict[str, dict[int, tuple[np.ndarray, bool, str]]]:
    path = match_out / FRAME_TRACK_ARTIFACT
    if not path.exists():
        return {}
    track: dict[str, dict[int, tuple[np.ndarray, bool, str]]] = {}
    with np.load(path) as data:
        rows = len(data["frames"])
        # court_H_per_frame.py, the human-assisted writer of the same artifact, carries
        # kind/support instead of reliable/source; treat its rows as reliable.
        reliable = data["reliable"] if "reliable" in data.files else np.ones(rows, dtype=bool)
        source = (
            data["source"]
            if "source" in data.files
            else np.asarray(["frame_track"] * rows, dtype=str)
        )
        for clip, frame, homography, is_reliable, row_source in zip(
            data["clips"], data["frames"], data["H"], reliable, source, strict=True
        ):
            track.setdefault(str(clip), {})[int(frame)] = (
                np.asarray(homography, dtype=float),
                bool(is_reliable),
                str(row_source),
            )
    return track


def expand_point_cameras(match_out: Path, frames_dir: str) -> Path:
    """Move each point camera onto every frame, following the registered court track.

    Without ``court_H_per_frame_v1.npz`` this falls back to the previous behaviour: the
    point camera repeated verbatim with ``frame_scope="point_static"``.
    """
    point_camera = np.load(match_out / "camera_P_per_point.npz", allow_pickle=True)
    frame_track = _load_frame_track(match_out)
    optional_defaults = {
        "reliable": False,
        "source": "legacy_missing_provenance",
        "ground_residual_px": float("nan"),
        "net_residual_px": float("nan"),
        "confidence": 0.0,
        "fallback_ancestry": "[]",
        "frame_scope": "unknown",
        "reference_frame": "",
    }
    by_point = {}
    for index, (point, projection) in enumerate(
        zip(point_camera["pts"], point_camera["P"], strict=True)
    ):
        by_point[int(point)] = {
            "P": projection,
            "net_cord_xy": (
                point_camera["net_cord_xy"][index]
                if "net_cord_xy" in point_camera.files
                else np.full((9, 2), np.nan, dtype=float)
            ),
            "net_cord_valid": (
                bool(point_camera["net_cord_valid"][index])
                if "net_cord_valid" in point_camera.files
                else False
            ),
            "net_cord_source": (
                str(point_camera["net_cord_source"][index])
                if "net_cord_source" in point_camera.files
                else "legacy_missing_net_cord"
            ),
            **{
                field: point_camera[field][index] if field in point_camera.files else default
                for field, default in optional_defaults.items()
            },
        }
    rig_prior = fit_match_rig_prior(match_out, frames_dir, by_point)
    rig_fallback_frames = 0
    rig_fallback_points: set[int] = set()
    clips = []
    frames = []
    projections = []
    metadata = {field: [] for field in optional_defaults}
    calibration_points = []
    net_cords = []
    net_cord_valid = []
    net_cord_source = []
    for point, record in sorted(by_point.items()):
        clip = f"pt{point:04d}"
        clip_track = frame_track.get(clip, {})
        for frame_path in sorted((match_out / frames_dir / clip).glob("f_*.jpg")):
            frame = int(frame_path.stem.removeprefix("f_"))
            projection = record["P"]
            net_cord = record["net_cord_xy"]
            frame_scope = "point_static"
            frame_reliable = bool(record["reliable"])
            source = str(record["source"])
            ground_residual = float(record["ground_residual_px"])
            net_residual = float(record["net_residual_px"])
            confidence = float(record["confidence"])
            fallback_ancestry = str(record["fallback_ancestry"])
            reference_frame = str(record["reference_frame"])
            tracked = clip_track.get(frame)
            if tracked is not None:
                homography, registered, registration_source = tracked
                composed = compose_frame_projection(record["P"], homography)
                if composed is not None:
                    projection = composed
                    # the observed tape was measured on the anchor frame, so it has to
                    # travel with the camera or it stops marking the net after a pan
                    net_cord = warp_points(anchor_to_frame_warp(record["P"], homography), net_cord)
                    frame_scope = "frame_track"
                    frame_reliable = registered_frame_reliable(
                        point_reliable=frame_reliable,
                        point_source=str(record["source"]),
                        registration_reliable=registered,
                        registration_source=registration_source,
                        ground_residual_px=float(record["ground_residual_px"]),
                        net_residual_px=float(record["net_residual_px"]),
                    )
                    source = f"{source}+{registration_source}"
                if not bool(record["reliable"]) and registered and rig_prior is not None:
                    rig_projection = rig_prior_projection(rig_prior, homography, point)
                    if rig_projection is not None:
                        candidate, rig_ground = rig_projection
                        if rig_ground <= MAXIMUM_GROUND_RESIDUAL_PX:
                            projection = candidate
                            frame_scope = "rig_prior_fallback"
                            frame_reliable = True
                            source = f"rig_prior+{registration_source}"
                            ground_residual = rig_ground
                            net_residual = rig_prior.net_residual_px
                            confidence = float(
                                math.exp(
                                    -rig_ground / MAXIMUM_GROUND_RESIDUAL_PX
                                    - rig_prior.net_residual_px / MAXIMUM_NET_RESIDUAL_PX
                                )
                            )
                            # camera_bundle's rig is fully automatic.  Its source points
                            # remain explicit here without introducing human-derived ancestry.
                            fallback_ancestry = "[]"
                            reference_frame = "match_rig:" + ",".join(
                                f"pt{value:04d}" for value in rig_prior.source_points
                            )
                            rig_fallback_frames += 1
                            rig_fallback_points.add(point)
            clips.append(clip)
            frames.append(frame)
            projections.append(projection)
            net_cords.append(net_cord)
            net_cord_valid.append(record["net_cord_valid"])
            net_cord_source.append(record["net_cord_source"])
            calibration_points.append(point)
            for field in metadata:
                if field == "frame_scope":
                    metadata[field].append(frame_scope)
                elif field == "reliable":
                    metadata[field].append(frame_reliable)
                elif field == "source":
                    metadata[field].append(source)
                elif field == "ground_residual_px":
                    metadata[field].append(ground_residual)
                elif field == "net_residual_px":
                    metadata[field].append(net_residual)
                elif field == "confidence":
                    metadata[field].append(confidence)
                elif field == "fallback_ancestry":
                    metadata[field].append(fallback_ancestry)
                elif field == "reference_frame":
                    metadata[field].append(reference_frame)
                else:
                    metadata[field].append(record[field])
    output = match_out / "camera_P_per_frame_v1.npz"
    if not projections:
        np.savez_compressed(
            output,
            clips=np.asarray([], dtype=str),
            frames=np.asarray([], dtype=np.int32),
            P=np.empty((0, 3, 4), dtype=float),
            reliable=np.asarray([], dtype=bool),
            source=np.asarray([], dtype=str),
            ground_residual_px=np.asarray([], dtype=float),
            net_residual_px=np.asarray([], dtype=float),
            confidence=np.asarray([], dtype=float),
            fallback_ancestry=np.asarray([], dtype=str),
            frame_scope=np.asarray([], dtype=str),
            reference_frame=np.asarray([], dtype=str),
            calibration_point=np.asarray([], dtype=np.int32),
            net_cord_xy=np.empty((0, 9, 2), dtype=float),
            net_cord_valid=np.asarray([], dtype=bool),
            net_cord_source=np.asarray([], dtype=str),
        )
        return output
    np.savez_compressed(
        output,
        clips=np.asarray(clips),
        frames=np.asarray(frames, dtype=np.int32),
        P=np.stack(projections),
        reliable=np.asarray(metadata["reliable"], dtype=bool),
        source=np.asarray(metadata["source"], dtype=str),
        ground_residual_px=np.asarray(metadata["ground_residual_px"], dtype=float),
        net_residual_px=np.asarray(metadata["net_residual_px"], dtype=float),
        confidence=np.asarray(metadata["confidence"], dtype=float),
        fallback_ancestry=np.asarray(metadata["fallback_ancestry"], dtype=str),
        frame_scope=np.asarray(metadata["frame_scope"], dtype=str),
        reference_frame=np.asarray(metadata["reference_frame"], dtype=str),
        calibration_point=np.asarray(calibration_points, dtype=np.int32),
        net_cord_xy=np.stack(net_cords),
        net_cord_valid=np.asarray(net_cord_valid, dtype=bool),
        net_cord_source=np.asarray(net_cord_source, dtype=str),
    )
    (match_out / "camera_rig_fallback_v1.json").write_text(
        json.dumps(
            {
                "schema": "camera_rig_fallback_v1",
                "automatic": True,
                "status": "available" if rig_prior is not None else "unavailable",
                "source_points": list(rig_prior.source_points) if rig_prior is not None else [],
                "shared_net_residual_px": (
                    rig_prior.net_residual_px if rig_prior is not None else None
                ),
                "recovered_frames": rig_fallback_frames,
                "recovered_points": sorted(rig_fallback_points),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def _compact_strings(values: np.ndarray) -> np.ndarray:
    """Drop unused fixed-width Unicode capacity from a point-local array."""
    if values.dtype.kind not in {"S", "U"}:
        return values
    # Match-wide camera artifacts can inherit one exceptional provenance string and
    # consequently allocate several kilobytes for every row.  NumPy selection keeps
    # that dtype width.  Rebuilding from Python values lets each point carry only the
    # characters it actually uses.
    return np.asarray(values.tolist())


def _slice_npz(
    source: Path,
    destinations: dict[str, Path],
    *,
    selector: str,
    selector_value,
) -> dict[str, int]:
    """Read one match-wide NPZ once and write compact point-local NPZ files."""
    counts: dict[str, int] = {}
    with np.load(source, allow_pickle=True) as data:
        # Materialize each compressed member once.  Re-indexing ``NpzFile`` for
        # every point would silently decompress its match-wide member every time.
        arrays = {name: np.asarray(data[name]) for name in data.files}
    selectors = arrays[selector]
    row_count = len(selectors)
    for clip, destination in destinations.items():
        selected = selectors == selector_value(clip)
        counts[clip] = int(np.count_nonzero(selected))
        payload: dict[str, np.ndarray] = {}
        for name, values in arrays.items():
            point_values = values[selected] if values.ndim and len(values) == row_count else values
            payload[name] = _compact_strings(point_values)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
    return counts


def _copy_coordinate_sidecar(source: Path, destination: Path) -> None:
    """Carry a sliced artifact's coordinate declaration into its point slice.

    The point reconstructor resolves every pixel column through the artifact's
    ``.coordinates.json`` and fails closed without it; a slice that dropped the
    sidecar could not be read.  A file is never coordinate evidence, so only an
    existing sidecar is copied and its absence is left to the reader to reject.
    """
    sidecar = res.coordinate_manifest_path(source)
    if sidecar.is_file():
        shutil.copyfile(sidecar, res.coordinate_manifest_path(destination))


def _slice_csv(
    source: Path,
    destinations: dict[str, Path],
) -> dict[str, int]:
    """Scan a match CSV once and preserve its schema in every point slice."""
    rows_by_clip: dict[str, list[dict[str, str]]] = {clip: [] for clip in destinations}
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is absent: {source}")
        fieldnames = reader.fieldnames
        for row in reader:
            clip = str(row.get("clip", ""))
            if clip in rows_by_clip:
                rows_by_clip[clip].append(row)
    for clip, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_by_clip[clip])
        _copy_coordinate_sidecar(source, destination)
    return {clip: len(rows) for clip, rows in rows_by_clip.items()}


def _slice_jsonl(source: Path, destinations: dict[str, Path]) -> dict[str, int]:
    rows_by_clip: dict[str, list[str]] = {clip: [] for clip in destinations}
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            clip = str(row.get("clip", ""))
            if clip in rows_by_clip:
                rows_by_clip[clip].append(line if line.endswith("\n") else f"{line}\n")
    for clip, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(rows_by_clip[clip]))
        _copy_coordinate_sidecar(source, destination)
    return {clip: len(rows) for clip, rows in rows_by_clip.items()}


def slice_reconstruction_match_artifacts(
    match_dir: Path,
    point_match_dirs: dict[str, Path],
    *,
    camera_artifact_name: str = RECONSTRUCTION_CAMERA_ARTIFACT,
    pose_artifact_name: str | None = None,
    physical_motion_name: str | None = None,
) -> dict[str, Any]:
    """Materialize the exact match-dir inputs S6 reads as point-local files.

    Reconstruction deliberately isolates point fitting in child processes.  Feeding
    those children match-wide arrays makes every child inflate and scan the same data.
    This helper performs that selection once before reconstruction and emits the same
    filenames and schemas used by the small multi-broadcast cohort roots.
    """
    if not point_match_dirs:
        raise ValueError("at least one point slice is required")
    for clip in point_match_dirs:
        if not clip.startswith("pt"):
            raise ValueError(f"invalid point clip: {clip}")

    required = (
        match_dir / camera_artifact_name,
        match_dir / RECONSTRUCTION_POINT_HOMOGRAPHY_ARTIFACT,
        match_dir / RECONSTRUCTION_TRACK_ARTIFACT,
        match_dir / RECONSTRUCTION_CADENCE_SIDECAR,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing reconstruction inputs: {missing}")

    artifacts: dict[str, dict[str, int]] = {}

    def destinations(name: str) -> dict[str, Path]:
        return {clip: point_dir / name for clip, point_dir in point_match_dirs.items()}

    artifacts[camera_artifact_name] = _slice_npz(
        match_dir / camera_artifact_name,
        destinations(camera_artifact_name),
        selector="clips",
        selector_value=str,
    )
    artifacts[RECONSTRUCTION_POINT_HOMOGRAPHY_ARTIFACT] = _slice_npz(
        match_dir / RECONSTRUCTION_POINT_HOMOGRAPHY_ARTIFACT,
        destinations(RECONSTRUCTION_POINT_HOMOGRAPHY_ARTIFACT),
        selector="pts",
        selector_value=lambda clip: int(clip.removeprefix("pt")),
    )
    frame_homography = match_dir / RECONSTRUCTION_FRAME_HOMOGRAPHY_ARTIFACT
    if frame_homography.is_file():
        artifacts[RECONSTRUCTION_FRAME_HOMOGRAPHY_ARTIFACT] = _slice_npz(
            frame_homography,
            destinations(RECONSTRUCTION_FRAME_HOMOGRAPHY_ARTIFACT),
            selector="clips",
            selector_value=str,
        )
    artifacts[RECONSTRUCTION_TRACK_ARTIFACT] = _slice_csv(
        match_dir / RECONSTRUCTION_TRACK_ARTIFACT,
        destinations(RECONSTRUCTION_TRACK_ARTIFACT),
    )
    for source in sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv")):
        artifacts[source.name] = _slice_csv(source, destinations(source.name))
    if pose_artifact_name:
        source = match_dir / pose_artifact_name
        if source.is_file():
            artifacts[source.name] = _slice_csv(source, destinations(source.name))
    if physical_motion_name:
        source = match_dir / physical_motion_name
        if source.is_file():
            artifacts[source.name] = _slice_jsonl(source, destinations(source.name))

    for point_dir in point_match_dirs.values():
        point_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            match_dir / RECONSTRUCTION_CADENCE_SIDECAR,
            point_dir / RECONSTRUCTION_CADENCE_SIDECAR,
        )

    return {
        "schema": "reconstruction_point_artifact_slices_v1",
        "source_match_dir": str(match_dir),
        "points": sorted(point_match_dirs),
        "artifacts": artifacts,
    }


# --------------------------------------------------------------- attempt-window camera

WINDOW_CAMERA_SCHEMA = "s6_owner_camera_transport_v1"
# One attempt window is a few hundred frames, so every frame carries its own registration
# and its own line-support witness instead of a stride-five sample plus interpolation. On
# the seventeen labeled attempts that removes the last two landmark-frame abstentions, which
# were frames adjacent to a close-up that no sample covered, without changing the landmark
# error and without accepting any of the 23 genuine close-up frames. The match-scale stage
# keeps its stride: it registers whole matches, not one window.
DEFAULT_WINDOW_REGISTRATION_STRIDE = 1
WINDOW_HEIGHT_SOURCES = ("metric_witnesses", "net_tape", "ground_intrinsic")
DEFAULT_WINDOW_HEIGHT_SOURCE = "metric_witnesses"
# camera_cal.main's per-point acceptance ladder, applied to one attempt window.
MAXIMUM_DIRECT_GROUND_RESIDUAL_PX = 0.5
MAXIMUM_DIRECT_GROUND_CONSISTENCY_PX = 0.5
MINIMUM_NET_IMAGE_HEIGHT_PX = 3.0


@dataclass(frozen=True)
class WindowCamera:
    """One automatic camera per native frame of a single attempt window.

    ``projections`` are court-to-image 3x4 matrices; ``homographies`` are the
    image-to-court ground transforms of ``court_H_per_frame_v1.npz``.  Frames whose
    registration is unsupported keep the anchor geometry and are reported unreliable,
    exactly as ``court_topology_runner.register_clip`` decides it.
    """

    frames: tuple[int, ...]
    projections: np.ndarray
    homographies: np.ndarray
    reliable: tuple[bool, ...]
    source: tuple[str, ...]
    anchor: dict[str, Any]
    point: dict[str, Any]
    registration: dict[str, Any]


def solve_window_camera(
    frame_paths: list[Path],
    frame_ids: list[int],
    *,
    stride: int = DEFAULT_WINDOW_REGISTRATION_STRIDE,
    anchor_mode: str | None = None,
    edge_convention: str | None = None,
    height_source: str = DEFAULT_WINDOW_HEIGHT_SOURCE,
    player_height_observations: list[dict[str, Any]] | None = None,
    tape_measurement: str = "hough_top",
) -> WindowCamera | dict[str, Any]:
    """Run the automatic S2 anchor, registration and height calibration on one window.

    Inputs are source frames and configuration only.  Returns a status dictionary rather
    than a camera when the automatic path abstains, so the abstention stays in the caller's
    denominator instead of being replaced by a fallback of unknown ancestry.
    """
    import cv2

    from cv.pipeline import camera_cal, court_topology, court_topology_runner

    if len(frame_paths) != len(frame_ids) or not frame_paths:
        raise ValueError("matching nonempty frame paths and native frame ids required")
    if tape_measurement not in {"hough_top", "connected_pixels"}:
        raise ValueError("unknown tape measurement")
    tape_options = {} if tape_measurement == "hough_top" else {"tape_measurement": tape_measurement}
    mode = anchor_mode or court_topology_runner.DEFAULT_ANCHOR_MODE
    anchor, attempts = court_topology_runner.select_anchor(frame_paths, mode)
    if anchor is None:
        return {
            "status": "abstained",
            "reason": "anchor_abstained",
            "anchor_failures": attempts,
        }
    per_frame, reliable, source, evidence = court_topology_runner.register_clip(
        frame_paths, anchor, stride
    )
    convention, convention_evidence = court_topology.edge_convention_world_transform(
        edge_convention or court_topology_runner.ARTIFACT_EDGE_CONVENTION
    )
    per_frame = {index: convention @ value for index, value in per_frame.items()}
    anchor_image = cv2.imread(os.fspath(frame_paths[anchor.index]))
    if anchor_image is None:
        return {"status": "abstained", "reason": "unreadable_anchor_frame"}
    height, width = anchor_image.shape[:2]
    court_to_image = np.linalg.inv(convention @ anchor.homography)
    point = _calibrate_window_height(
        anchor_image,
        court_to_image,
        width,
        height,
        height_source=height_source,
        **tape_options,
    )
    if point is None:
        return {
            "status": "abstained",
            "reason": "no_net_or_intrinsic_height_solution",
            "anchor_frame": frame_paths[anchor.index].name,
        }
    projection = point.pop("P")
    # A calibrated ground homography already determines focal, tilt and camera height under
    # the declared square-pixel/centred-principal-point model.  The old gate nevertheless
    # required that solution to agree with a nearly flat tape curve within eight pixels.  On
    # the opened camera cohort a sub-pixel improvement in that weak tape objective can move
    # focal by several thousand pixels and camera height by metres.  Keep the tape as an
    # explicit witness, but do not let it veto the independently observable metric solution.
    metric_ground_solution = point["height_source"] in {
        "metric_witnesses",
        "ground_intrinsic",
    }
    point_reliable = bool(
        point["source"] == "direct"
        and point["ground_residual_px"] <= MAXIMUM_GROUND_RESIDUAL_PX
        and (
            metric_ground_solution
            or (
                np.isfinite(point["net_residual_px"])
                and point["net_residual_px"] <= MAXIMUM_NET_RESIDUAL_PX
            )
        )
    )
    point["reliable"] = point_reliable
    projections, homographies, frame_reliable, frame_source = [], [], [], []
    for index in range(len(frame_paths)):
        homography = np.asarray(per_frame[index], dtype=float)
        composed = compose_frame_projection(projection, homography)
        registration_source = str(source[index])
        if composed is None:
            projections.append(np.full((3, 4), np.nan))
            homographies.append(homography)
            frame_reliable.append(False)
            frame_source.append(f"{point['source']}+{registration_source}+uncomposable")
            continue
        projections.append(composed)
        homographies.append(homography)
        frame_reliable.append(
            registered_frame_reliable(
                point_reliable=point_reliable,
                point_source=point["source"],
                registration_reliable=bool(reliable[index]),
                registration_source=registration_source,
                ground_residual_px=point["ground_residual_px"],
                net_residual_px=point["net_residual_px"],
            )
        )
        frame_source.append(f"{point['source']}+{registration_source}")
    if height_source == "metric_witnesses":
        bundled = _fit_metric_window_bundle(
            frame_ids,
            homographies,
            projections,
            anchor_index=anchor.index,
            net_observations=camera_cal.measure_net_band(
                anchor_image, court_to_image, **tape_options
            ),
            player_height_observations=player_height_observations or [],
            image_size=(width, height),
        )
        projections, frame_reliable, frame_source = _admit_window_bundle(
            bundled,
            projections,
            homographies,
            frame_ids,
            frame_reliable,
            frame_source,
            point,
            anchor_index=anchor.index,
        )
    return WindowCamera(
        frames=tuple(int(value) for value in frame_ids),
        projections=np.stack(projections),
        homographies=np.stack(homographies),
        reliable=tuple(frame_reliable),
        source=tuple(frame_source),
        anchor={
            "frame": int(frame_ids[anchor.index]),
            "image": frame_paths[anchor.index].name,
            "source": anchor.source,
            "topology_score": float(anchor.topology_score),
            "surface_score": float(anchor.surface_score),
            "visible_edge_refinement": anchor.refinement_evidence,
            "graded_band_climb": anchor.far_evidence,
            "anchor_mode": mode,
            "anchor_failures": attempts,
            "edge_convention": convention_evidence,
        },
        point=point,
        registration=evidence,
    )


def _emitted_window_diagnostics(projections, homographies, frame_ids, net_cord, anchor_index):
    """Residuals and transported witnesses for the actual emitted projections."""
    from cv.pipeline import camera_cal

    diagnostics = []
    for frame, projection, homography in zip(frame_ids, projections, homographies, strict=True):
        cord = None
        if net_cord is not None:
            warp = np.linalg.inv(homography) @ homographies[anchor_index]
            cord = warp_points(warp, np.asarray(net_cord, dtype=float))
        observations = None if cord is None else [(0.0, 0.0, x, 0.0, y) for x, y in cord]
        try:
            ground, net = camera_cal.calibration_residuals(
                np.asarray(projection), np.linalg.inv(homography), observations
            )
        except (ValueError, np.linalg.LinAlgError):
            ground, net = float("inf"), float("nan")
        diagnostics.append(
            dict(
                frame=int(frame),
                ground_residual_px=float(ground) if np.isfinite(ground) else None,
                net_residual_px=float(net) if np.isfinite(net) else None,
                net_cord_xy=None if cord is None else cord.tolist(),
                metric_ground_qualified=bool(
                    np.isfinite(projection).all()
                    and np.isfinite(ground)
                    and ground <= MAXIMUM_GROUND_RESIDUAL_PX
                ),
            )
        )
    return diagnostics


def _admit_window_bundle(
    bundled, original, homographies, frame_ids, reliability, sources, point, *, anchor_index
):
    """An optimizer success is not camera admission; preserve an honest fallback."""
    from cv.pipeline import camera_cal

    point["initial_calibration"] = {
        key: (
            None
            if isinstance(point.get(key), float) and not np.isfinite(point[key])
            else point.get(key)
        )
        for key in (
            "source",
            "height_model",
            "focal_native_px",
            "ground_residual_px",
            "net_residual_px",
            "reliable",
        )
    }
    selected = original
    candidate_diagnostics = None
    if bundled is not None:
        candidate_diagnostics = _emitted_window_diagnostics(
            bundled["projections"], homographies, frame_ids, point.get("net_cord_xy"), anchor_index
        )
    accepted = candidate_diagnostics is not None and all(
        row["metric_ground_qualified"] for row in candidate_diagnostics
    )
    record = {} if bundled is None else {k: v for k, v in bundled.items() if k != "projections"}
    record.update(
        status="accepted" if accepted else "rejected",
        admission="finite emitted camera and original <=4px ground residual rule on every frame",
        candidate_frame_diagnostics=candidate_diagnostics,
    )
    if accepted:
        selected = bundled["projections"]
        sources = [f"metric_bundle+{value}" for value in sources]
        point["height_model"] = "joint_square_pixel_metric_bundle"
        point["focal_native_px"] = camera_cal._view_from_projection(selected[anchor_index])[3]
    else:
        record["reason"] = (
            "bundle unavailable"
            if bundled is None
            else "emitted bundle violates original ground reliability"
        )
        record["retained"] = (
            "original intrinsic projection; original and emitted qualifications still apply"
        )
        point["fallback_ancestry"] = [*point.get("fallback_ancestry", []), "metric_bundle_rejected"]
    point["metric_bundle"] = record
    diagnostics = _emitted_window_diagnostics(
        selected, homographies, frame_ids, point.get("net_cord_xy"), anchor_index
    )
    reliable = [
        bool(old and row["metric_ground_qualified"])
        for old, row in zip(reliability, diagnostics, strict=True)
    ]
    for row, okay in zip(diagnostics, reliable, strict=True):
        row["reliable"] = okay
        row["net_cord_source"] = (
            (
                point.get("net_observation_source", "unavailable")
                if row["frame"] == int(frame_ids[anchor_index])
                else "registered_anchor_transport"
            )
            if row["net_cord_xy"] is not None
            else "unavailable"
        )
        row["confidence"] = (
            float(
                math.exp(
                    -row["ground_residual_px"] / MAXIMUM_GROUND_RESIDUAL_PX
                    - row["net_residual_px"] / MAXIMUM_NET_RESIDUAL_PX
                )
            )
            if okay and row["net_residual_px"] is not None
            else 0.0
        )
    point["emitted_frame_diagnostics"] = diagnostics
    anchor = diagnostics[anchor_index]
    point["ground_residual_px"] = anchor["ground_residual_px"]
    point["net_residual_px"] = anchor["net_residual_px"]
    point["reliable"] = reliable[anchor_index]
    point["height_witnesses"] = {
        **point.get("height_witnesses", {}),
        "status": (
            "unavailable"
            if anchor["net_residual_px"] is None
            else "corroborates"
            if anchor["net_residual_px"] <= MAXIMUM_NET_RESIDUAL_PX
            else "disagrees_not_used_as_focal"
        ),
    }
    return selected, reliable, sources


def _fit_metric_window_bundle(
    frame_ids: list[int],
    homographies: list[np.ndarray],
    initial_projections: list[np.ndarray],
    *,
    anchor_index: int,
    net_observations: list | None,
    player_height_observations: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    """Fit one physical centre and smooth per-frame views from automatic witnesses only.

    Registered ground homographies provide the strong focal/tilt/height constraints.  The
    measured cord curve and grounded ankle-to-head pose lengths are weak metric witnesses;
    neither is allowed to replace the court evidence on its own.  At most 24 temporal knots
    enter the shared-centre solve, after which every registered frame is fitted to its own
    ground plane with that centre and median-smoothed view parameters.
    """
    from types import SimpleNamespace

    import cv2

    from cv.pipeline import camera_bundle, camera_cal

    if not frame_ids or len(frame_ids) != len(homographies):
        return None
    ground_world = camera_bundle.COURT_LANDMARKS
    stride = max(1, math.ceil(len(frame_ids) / 24))
    knot_indices = set(range(0, len(frame_ids), stride)) | {anchor_index, len(frame_ids) - 1}
    frame_index = {int(frame): index for index, frame in enumerate(frame_ids)}
    usable_players = [
        row for row in player_height_observations if int(row.get("frame", -1)) in frame_index
    ]
    # The near player is both larger in the image and less affected by pose truncation. Select
    # the tallest-looking grounded poses by a declared, label-free image-space rule.
    usable_players = [row for row in usable_players if row.get("side") == "near"]
    usable_players.sort(key=lambda row: float(row.get("pixel_height", 0.0)), reverse=True)
    usable_players = usable_players[:8]
    knot_indices.update(frame_index[int(row["frame"])] for row in usable_players)
    knot_indices = sorted(knot_indices)
    observations: list[camera_cal.CameraObservation] = []
    initial: dict[int, np.ndarray] = {}
    for index in knot_indices:
        frame = int(frame_ids[index])
        court_to_image = np.linalg.inv(np.asarray(homographies[index], dtype=float))
        pixels = cv2.perspectiveTransform(
            ground_world[:, :2].astype(np.float32)[None, ...], court_to_image
        )[0]
        # The retained ground-preserving P is a projective hybrid, not a
        # centred square-pixel rig. Decomposing it can move the principal point
        # hundreds of pixels; rebuilding only its yaw/tilt/focal is not a
        # coordinate-preserving initialization. Use the existing rigid ground
        # self-calibration for initialization only; observations stay unchanged.
        rigid = camera_cal.h_to_projection(court_to_image, w=image_size[0], h=image_size[1])
        if rigid is None:
            return None
        initial[frame] = rigid[0]
        observations.extend(
            camera_cal.CameraObservation(
                frame=frame,
                landmark=f"automatic_ground_{landmark}",
                world=np.asarray(world, dtype=float),
                pixel=np.asarray(pixel, dtype=float),
                weight=1.0,
            )
            for landmark, (world, pixel) in enumerate(zip(ground_world, pixels, strict=True))
        )
    anchor_frame = int(frame_ids[anchor_index])
    if net_observations:
        observations.extend(
            camera_cal.CameraObservation(
                frame=anchor_frame,
                landmark="automatic_net_cord",
                world=np.asarray([court_x, camera_cal.NET_Y, net_height], dtype=float),
                pixel=np.asarray([image_x, top_y], dtype=float),
                # The tape detector has uncertain x correspondence and is deliberately weaker
                # than one painted-court point.
                weight=0.08,
            )
            for court_x, net_height, image_x, _ground_y, top_y in net_observations
        )
    for row in usable_players:
        observations.append(
            camera_cal.CameraObservation(
                frame=int(row["frame"]),
                landmark="automatic_player_stature",
                world=np.asarray(
                    [*row["root_xy"], float(row["height_m"]) * float(row["head_fraction"])],
                    dtype=float,
                ),
                pixel=np.asarray(row["head_xy"], dtype=float),
                # Population-height uncertainty is carried explicitly and keeps this a weak
                # witness relative to the rigid court.
                weight=float(row["confidence"]) * 0.04,
            )
        )
    try:
        solution = camera_cal.fit_joint_camera(
            observations,
            initial,
            image_size=image_size,
            maximum_evaluations=300,
            smoothness_weight=0.03,
            fixed_k1=0.0,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not solution.success:
        return None
    rig = SimpleNamespace(
        center=solution.center,
        k1=solution.k1,
        dist_center=solution.dist_center,
    )
    knot_views = solution.view_parameters
    fitted: list[np.ndarray] = []
    states: list[tuple[float, float, float]] = []
    ground_rms: list[float] = []
    for frame, homography in zip(frame_ids, homographies, strict=True):
        nearest = min(knot_views, key=lambda value: abs(value - int(frame)))
        try:
            projection, state, residual = camera_bundle.fit_frame_view(
                np.linalg.inv(np.asarray(homography, dtype=float)),
                rig,
                image_size,
                knot_views[nearest],
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        fitted.append(projection)
        states.append(state)
        ground_rms.append(float(residual))
    state_array = np.asarray(states, dtype=float)
    smoothed = state_array.copy()
    if len(states) >= 3:
        padded = np.pad(state_array, ((2, 2), (0, 0)), mode="edge")
        for index in range(len(states)):
            smoothed[index] = np.median(padded[index : index + 5], axis=0)
        candidates = [
            camera_bundle.projection_matrix(solution.center, *state, image_size)
            for state in smoothed
        ]
        # Fail closed per frame if smoothing no longer represents its registered court.
        for index, (candidate, homography) in enumerate(zip(candidates, homographies, strict=True)):
            expected = cv2.perspectiveTransform(
                ground_world[:, :2].astype(np.float32)[None, ...],
                np.linalg.inv(np.asarray(homography, dtype=float)),
            )[0]
            predicted = camera_cal.project(candidate, ground_world)
            candidate_rms = float(np.sqrt(np.mean(np.square(predicted - expected))))
            if candidate_rms <= max(2.0, ground_rms[index] + 0.5):
                fitted[index] = candidate
                ground_rms[index] = candidate_rms
            else:
                smoothed[index] = state_array[index]
    player_sources = sorted(
        {str(row.get("height_source", "population_prior")) for row in usable_players}
    )
    return {
        "projections": fitted,
        "schema": "automatic_metric_window_bundle_v1",
        "initialization": "rigid_ground_self_calibration; original projective cameras retained separately",
        "camera_center_m": solution.center.tolist(),
        "shared_k1_pixel_inverse_squared": solution.k1,
        "knot_frames": [int(frame_ids[index]) for index in knot_indices],
        "ground_observations": len(knot_indices) * len(ground_world),
        "net_cord_observations": 0 if not net_observations else len(net_observations),
        "player_stature_observations": len(usable_players),
        "player_height_sources": player_sources,
        "population_prior_sigma_m": (
            None if not usable_players else float(usable_players[0].get("height_sigma_m", 0.08))
        ),
        "ground_rms_px_median": float(np.median(ground_rms)),
        "optimizer_cost": solution.cost,
        "optimizer_evaluations": solution.evaluations,
        "temporal_smoothing": "five-frame median with per-frame <=2px ground guard",
    }


def _calibrate_window_height(
    anchor_image: np.ndarray,
    court_to_image: np.ndarray,
    width: int,
    height: int,
    *,
    height_source: str = DEFAULT_WINDOW_HEIGHT_SOURCE,
    tape_measurement: str = "hough_top",
) -> dict[str, Any] | None:
    """Pin the projection's vertical column from the anchor frame's own evidence.

    ``net_tape`` is ``camera_cal.main``'s per-point ladder restricted to one window: an
    observed net-tape focal first, then the ground-preserving intrinsic solve.  A window has
    no sibling points, so the shared-column fallbacks of the match-scoped stage do not apply
    and their absence is reported rather than approximated.

    ``metric_witnesses`` uses the calibrated ground solution and records the regulation net
    curve (0.914 m centre, 1.07 m supports) as an independent residual.  This is the default:
    the ground constraints jointly determine focal, tilt and camera height, whereas the nearly
    flat tape curve alone is ill-conditioned.  ``ground_intrinsic`` is retained as an alias for
    reproducing the pre-promotion experiment and ``net_tape`` as its negative control.
    """
    from cv.pipeline import camera_cal

    if height_source not in WINDOW_HEIGHT_SOURCES:
        raise ValueError(f"unknown height source {height_source!r}")
    if tape_measurement not in {"hough_top", "connected_pixels"}:
        raise ValueError("unknown tape measurement")
    tape_options = {} if tape_measurement == "hough_top" else {"tape_measurement": tape_measurement}
    observations = camera_cal.measure_net_band(anchor_image, court_to_image, **tape_options)
    top_source = (
        "observed_connected_tape_pixels_top"
        if tape_measurement == "connected_pixels"
        else "observed_connected_tape_top_envelope"
    )
    if height_source in {"metric_witnesses", "ground_intrinsic"}:
        solved = camera_cal.intrinsic_projection_from_ground(court_to_image, w=width, h=height)
        if solved is not None:
            projection, focal = solved
            ground_residual, net_residual = camera_cal.calibration_residuals(
                projection, court_to_image, observations
            )
            return {
                "P": projection,
                "source": "direct",
                "height_source": height_source,
                "height_model": (
                    "joint_square_pixel_ground_intrinsics_with_physical_height_witnesses"
                    if height_source == "metric_witnesses"
                    else "joint_square_pixel_ground_intrinsics"
                ),
                "height_witnesses": {
                    "net_centre_height_m": camera_cal.NET_H_CENTER,
                    "net_support_height_m": camera_cal.NET_H_POST,
                    "net_support": camera_cal.NET_SUPPORT,
                    "singles_stick_x_m": list(camera_cal.NET_STICK_X),
                    "doubles_post_x_m": list(camera_cal.NET_POST_X),
                    "observed_tape_samples": 0 if observations is None else len(observations),
                    "status": (
                        "unavailable"
                        if observations is None
                        else "corroborates"
                        if np.isfinite(net_residual) and net_residual <= MAXIMUM_NET_RESIDUAL_PX
                        else "disagrees_not_used_as_focal"
                    ),
                },
                "focal_native_px": float(focal),
                "net_observation_source": (
                    "unavailable"
                    if observations is None
                    else top_source
                    if tape_measurement == "connected_pixels"
                    else "observed_for_residual_only"
                ),
                "net_observations": 0 if observations is None else len(observations),
                "ground_residual_px": float(ground_residual),
                "net_residual_px": float(net_residual),
                "fallback_ancestry": [],
                "net_cord_xy": _net_cord_pixels(observations),
                "net_cord_valid": observations is not None,
            }
    focal = (
        camera_cal.solve_f_from_net(court_to_image, observations, w=width, h=height)
        if observations is not None
        else None
    )
    observation_source = top_source
    if focal is None:
        legacy = camera_cal.measure_net_band(
            anchor_image, court_to_image, prefer_top_envelope=False
        )
        legacy_focal = (
            camera_cal.solve_f_from_net(court_to_image, legacy, w=width, h=height)
            if legacy is not None
            else None
        )
        if legacy_focal is not None:
            observations, focal = legacy, legacy_focal
            observation_source = "hough_net_centerline_fallback"
    if focal is not None:
        projection = camera_cal.fixed_f_projection_from_ground(
            court_to_image, focal, w=width, h=height
        )
        ground_residual, net_residual = camera_cal.calibration_residuals(
            projection, court_to_image, observations
        )
        check = camera_cal.net_reprojection_check(projection, court_to_image)
        if (
            check["ground_err_px"] <= MAXIMUM_DIRECT_GROUND_CONSISTENCY_PX
            and ground_residual <= MAXIMUM_DIRECT_GROUND_RESIDUAL_PX
            and np.isfinite(net_residual)
            and net_residual <= MAXIMUM_NET_RESIDUAL_PX
            and min(check["net_px"]) >= MINIMUM_NET_IMAGE_HEIGHT_PX
        ):
            return {
                "P": projection,
                "source": "direct",
                "focal_native_px": float(focal),
                "height_source": "net_tape",
                "height_model": "ground_homography_with_net_derived_focal_height_column",
                "net_observation_source": observation_source,
                "net_cord_valid": camera_cal.net_source_is_tape_top(observation_source),
                "net_observations": len(observations),
                "ground_residual_px": float(ground_residual),
                "net_residual_px": float(net_residual),
                "fallback_ancestry": [],
                "net_cord_xy": _net_cord_pixels(observations),
            }
    solved = camera_cal.intrinsic_projection_from_ground(court_to_image, w=width, h=height)
    if solved is None:
        return None
    projection, intrinsic_focal = solved
    ground_residual, net_residual = camera_cal.calibration_residuals(
        projection, court_to_image, observations
    )
    return {
        "P": projection,
        "source": "intrinsic_fallback",
        "height_source": "ground_intrinsic",
        "height_model": "joint_square_pixel_ground_intrinsics",
        "focal_native_px": float(intrinsic_focal),
        "net_observation_source": (
            observation_source if observations is not None else "unavailable"
        ),
        "net_cord_valid": observations is not None
        and camera_cal.net_source_is_tape_top(observation_source),
        "net_observations": 0 if observations is None else len(observations),
        "ground_residual_px": float(ground_residual),
        "net_residual_px": float(net_residual),
        "fallback_ancestry": ["ground_homography_intrinsic_solve"],
        "net_cord_xy": _net_cord_pixels(observations),
    }


def _net_cord_pixels(observations: list | None) -> list[list[float]] | None:
    """The nine observed tape pixels, or nothing when the tape was not measured."""
    if not observations:
        return None
    pixels = [[float(row[2]), float(row[4])] for row in observations]
    return pixels if len(pixels) == 9 else None


def _window_confidence(camera: WindowCamera) -> float:
    if not camera.point["reliable"]:
        return 0.25 if camera.point["fallback_ancestry"] else 0.0
    return float(
        math.exp(
            -camera.point["ground_residual_px"] / MAXIMUM_GROUND_RESIDUAL_PX
            - camera.point["net_residual_px"] / MAXIMUM_NET_RESIDUAL_PX
        )
    )


def write_window_camera(
    out_dir: Path,
    match_id: str,
    clip: str,
    camera: WindowCamera,
    *,
    inputs: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Write one attempt window's automatic camera in the layouts the fitter packet reads.

    ``cameras.json`` carries the per-frame rows in the ``s6_owner_camera_transport_v1``
    layout the connected fitter's packet consumes, with ``annotation_origin`` set to
    ``automatic`` and ``human_derived`` false: these rows are pipeline output and must never
    be read as agent coordinates. The two npz files are the ordinary per-frame pipeline
    artifacts restricted to this window.
    """
    from cv.pipeline import paths, provenance

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(camera.frames, dtype=np.int32)
    clips = np.asarray([clip] * len(frames), dtype=str)
    reliable = np.asarray(camera.reliable, dtype=bool)
    source = np.asarray(camera.source, dtype=str)
    point = int(clip.removeprefix("pt")) if clip.startswith("pt") else -1
    net_cord = camera.point.get("net_cord_xy")
    frame_diagnostics = camera.point.get("emitted_frame_diagnostics")
    net_cord_xy = np.repeat(
        np.asarray(net_cord, dtype=float)[None, ...]
        if net_cord is not None
        else np.full((1, 9, 2), np.nan),
        len(frames),
        axis=0,
    )
    if frame_diagnostics is not None:
        net_cord_xy = np.asarray(
            [
                row["net_cord_xy"] if row["net_cord_xy"] is not None else np.full((9, 2), np.nan)
                for row in frame_diagnostics
            ],
            dtype=float,
        )

    def emitted_array(name):
        return (
            np.asarray(
                [row[name] if row[name] is not None else np.nan for row in frame_diagnostics]
            )
            if frame_diagnostics is not None
            else np.full(
                len(frames),
                _window_confidence(camera) if name == "confidence" else camera.point[name],
            )
        )

    np.savez_compressed(
        out_dir / FRAME_TRACK_ARTIFACT,
        clips=clips,
        frames=frames,
        H=camera.homographies,
        reliable=reliable,
        source=source,
    )
    np.savez_compressed(
        out_dir / RECONSTRUCTION_CAMERA_ARTIFACT,
        clips=clips,
        frames=frames,
        P=camera.projections,
        reliable=reliable,
        source=source,
        ground_residual_px=emitted_array("ground_residual_px"),
        net_residual_px=emitted_array("net_residual_px"),
        confidence=emitted_array("confidence"),
        fallback_ancestry=np.asarray(
            [json.dumps(camera.point["fallback_ancestry"])] * len(frames), dtype=str
        ),
        frame_scope=np.asarray(["frame_track"] * len(frames), dtype=str),
        reference_frame=np.asarray([camera.anchor["image"]] * len(frames), dtype=str),
        calibration_point=np.full(len(frames), point, dtype=np.int32),
        net_cord_xy=net_cord_xy,
        net_cord_valid=np.full(
            len(frames),
            net_cord is not None and camera.point.get("net_cord_valid", True),
            dtype=bool,
        ),
        net_cord_source=np.asarray(
            [row["net_cord_source"] for row in frame_diagnostics]
            if frame_diagnostics is not None
            else [camera.point["net_observation_source"]] * len(frames),
            dtype=str,
        ),
    )
    rows = window_camera_rows(camera)
    cameras = {
        "schema": WINDOW_CAMERA_SCHEMA,
        "scope": (
            "Automatic S2 camera for one attempt window: source frames and configuration "
            "only. The row layout matches the connected fitter's packet; the provenance "
            "does not, and must not be read as human evidence."
        ),
        "human_derived": False,
        "annotation_origin": "automatic",
        "camera_stream": "automatic_s2",
        "inputs": inputs,
        "code": provenance.git_record(paths.REPO_ROOT),
        "configuration": configuration,
        "match_id": match_id,
        "clip": clip,
        "cameras": rows,
        "anchor": camera.anchor,
        "point_calibration": {
            key: value for key, value in camera.point.items() if key != "net_cord_xy"
        },
        "registration": camera.registration,
        "supported": sum(row["status"] == "supported" for row in rows),
        "total": len(rows),
        "automatic_inference_eligible": True,
        "airborne_metric_accuracy_certified": False,
    }
    (out_dir / "cameras.json").write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n")
    return cameras


def window_camera_rows(camera: WindowCamera) -> list[dict[str, Any]]:
    """Per-frame rows in the layout the connected fitter's packet reads."""
    rows = []
    for index, frame in enumerate(camera.frames):
        projection = camera.projections[index]
        supported = bool(camera.reliable[index]) and bool(np.isfinite(projection).all())
        row: dict[str, Any] = {
            "frame": int(frame),
            "status": "supported" if supported else "held",
            "source": camera.source[index],
            "parameterization": "automatic_ground_homography_with_net_vertical_column",
        }
        diagnostics = camera.point.get("emitted_frame_diagnostics")
        if diagnostics is not None:
            current = diagnostics[index]
            row.update(
                {
                    key: current[key]
                    for key in ("ground_residual_px", "net_residual_px", "confidence")
                }
            )
            row["fallback_ancestry"] = camera.point["fallback_ancestry"]
            row["parameterization"] = camera.point.get("height_model", row["parameterization"])
        if supported:
            row["P"] = projection.tolist()
        else:
            row["reason"] = camera.source[index]
        rows.append(row)
    return rows
