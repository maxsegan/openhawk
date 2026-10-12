"""Detect sustained nonstandard-camera spans inside automatic active play."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.native_actor_continuity import ActorInputs, load_actor_inputs, witnesses
from cv.pipeline.resolution import (
    FrameSize,
    coordinate_columns_and_size,
    coordinate_manifest_path,
    scale_boxes,
)
from cv.pipeline.provenance import AUTOMATIC_MODE, build_provenance, file_record

SCHEMA = "play_camera_leakage_v1"
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0
COURT_LINES = [
    ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),
    ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),
    ((0.0, 0.0), (0.0, COURT_LENGTH_M)),
    ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),
    ((1.372, 0.0), (1.372, COURT_LENGTH_M)),
    ((COURT_WIDTH_M - 1.372, 0.0), (COURT_WIDTH_M - 1.372, COURT_LENGTH_M)),
    ((1.372, 5.485), (COURT_WIDTH_M - 1.372, 5.485)),
    ((1.372, COURT_LENGTH_M - 5.485), (COURT_WIDTH_M - 1.372, COURT_LENGTH_M - 5.485)),
    ((COURT_WIDTH_M / 2.0, 5.485), (COURT_WIDTH_M / 2.0, COURT_LENGTH_M - 5.485)),
]
LINE_SAMPLE_STEP_M = 0.5
LINE_MARGIN_PX = 8
LINE_CONTRAST = 18.0
LINE_PERP_PX = (4, 6)
SHIFT_RANGE_PX = 96
SHIFT_STEP_PX = 16
REFERENCE_SUPPORT_MINIMUM = 0.25
RELATIVE_SUPPORT_DROP = 0.35
HEIGHT_REFERENCE_BAND_M = (1.0, 2.6)
GIANT_APPARENT_HEIGHT_M = 3.2
MINIMUM_LEAK_RUN_SECONDS = 0.25
LEAK_RUN_DENSITY = 0.7
LEAK_GAP_BRIDGE_FRAMES = 4
SHOT_SNAP_FRAMES = 6
MINIMUM_COURT_SAMPLES = 20

THRESHOLDS = {
    "line_contrast": LINE_CONTRAST,
    "line_sample_step_m": LINE_SAMPLE_STEP_M,
    "reference_support_minimum": REFERENCE_SUPPORT_MINIMUM,
    "relative_support_drop": RELATIVE_SUPPORT_DROP,
    "height_reference_band_m": list(HEIGHT_REFERENCE_BAND_M),
    "giant_apparent_height_m": GIANT_APPARENT_HEIGHT_M,
    "minimum_leak_run_seconds": MINIMUM_LEAK_RUN_SECONDS,
    "leak_run_density": LEAK_RUN_DENSITY,
    "leak_gap_bridge_frames": LEAK_GAP_BRIDGE_FRAMES,
    "shot_snap_frames": SHOT_SNAP_FRAMES,
    "minimum_court_samples": MINIMUM_COURT_SAMPLES,
}


@dataclass
class BoxObservation:
    box: np.ndarray
    confidence: float


@dataclass
class CameraEvidence:
    homography: np.ndarray | None = None
    sources: set[str] = field(default_factory=set)
    reliable_frames: set[int] = field(default_factory=set)


@dataclass
class MatchLeakageInputs:
    """Explicit, run-local snapshots; never an ambient cross-run cache."""

    match_dir: Path
    boxes: dict[str, dict[str, dict[int, BoxObservation]]]
    cameras: dict[str, CameraEvidence]
    records: list[tuple[Path, dict | None]]
    box_path: Path | None
    actor_inputs: ActorInputs | None = None

    def assert_unchanged(self) -> None:
        candidates = sorted(self.match_dir.glob("player_boxes_*_native_sided_v1.csv"))
        selected = candidates[0] if candidates else None
        if selected != self.box_path:
            raise ValueError("selected player artifact changed during leakage evaluation")
        for path, expected in self.records:
            actual = file_record(path) if path.exists() else None
            if actual != expected:
                raise ValueError(f"leakage input changed during evaluation: {path}")


@dataclass
class FrameEvidence:
    frame: int
    court_support: float | None
    court_samples: int
    frame_readable: bool
    apparent_near_m: float | None
    apparent_far_m: float | None
    near_present: bool
    far_present: bool
    side_conflict: bool


@dataclass
class PointReference:
    support: float | None
    apparent_height_m: float | None

    @property
    def court_arm_enabled(self) -> bool:
        return self.support is not None and self.support >= REFERENCE_SUPPORT_MINIMUM

    @property
    def box_arm_enabled(self) -> bool:
        low, high = HEIGHT_REFERENCE_BAND_M
        return self.apparent_height_m is not None and low <= self.apparent_height_m <= high


def frame_number(value: str) -> int:
    match = re.search(r"\d+", Path(str(value)).stem)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def apply_h(homography: np.ndarray, point) -> np.ndarray:
    value = homography @ np.array([point[0], point[1], 1.0])
    return value[:2] / value[2]


def _load_boxes(path: Path, clip: str) -> dict[str, dict[int, BoxObservation]]:
    return _load_boxes_by_clip(path, selected_clip=clip).get(clip, {"near": {}, "far": {}})


def _load_boxes_by_clip(
    path: Path, *, selected_clip: str | None = None
) -> dict[str, dict[str, dict[int, BoxObservation]]]:
    sidecar = json.loads(coordinate_manifest_path(path).read_text())
    image = sidecar["image_size"]
    image_size = FrameSize(int(image["width"]), int(image["height"]))
    output: dict[str, dict[str, dict[int, BoxObservation]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns, artifact_size = coordinate_columns_and_size(
            reader.fieldnames or (),
            sidecar,
            native_columns=("x0_native", "y0_native", "x1_native", "y1_native"),
            legacy_columns=("x0", "y0", "x1", "y1"),
        )
        for row in reader:
            side = row.get("side")
            clip = row.get("clip")
            if (
                not clip
                or (selected_clip is not None and clip != selected_clip)
                or side not in {"near", "far"}
            ):
                continue
            point_boxes = output.setdefault(clip, {"near": {}, "far": {}})
            box = scale_boxes(
                [[float(row[column]) for column in columns]],
                artifact_size,
                image_size,
            )[0]
            frame = frame_number(row["frame"])
            observation = BoxObservation(box, float(row.get("conf", 0.0)))
            current = point_boxes[side].get(frame)
            if current is None or observation.confidence > current.confidence:
                point_boxes[side][frame] = observation
    return output


def _load_camera(match_dir: Path, clip: str) -> CameraEvidence:
    camera = CameraEvidence()
    homographies = match_dir / "court_H_per_point.npz"
    if homographies.exists():
        with np.load(homographies, allow_pickle=False) as data:
            points = [int(value) for value in data["pts"]]
            point = int(clip.replace("pt", "").lstrip("0") or 0)
            if point in points:
                camera.homography = data["H"][points.index(point)]
    projections = match_dir / "camera_P_per_frame_v1.npz"
    if projections.exists():
        with np.load(projections, allow_pickle=False) as data:
            mask = data["clips"] == clip
            camera.sources = {str(value) for value in data["source"][mask]}
            camera.reliable_frames = {
                int(frame)
                for frame, reliable in zip(data["frames"][mask], data["reliable"][mask])
                if reliable
            }
    return camera


def load_match_inputs(
    match_dir: Path, *, native_actor_continuity: bool = False
) -> MatchLeakageInputs:
    """Read the match-wide player/camera tables once, retaining per-point semantics."""
    candidates = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))
    box_path = candidates[0] if candidates else None
    homographies = match_dir / "court_H_per_point.npz"
    projections = match_dir / "camera_P_per_frame_v1.npz"
    paths = [homographies, projections]
    if box_path is not None:
        paths += [box_path, coordinate_manifest_path(box_path)]
    records = [(path, file_record(path) if path.exists() else None) for path in paths]
    boxes = _load_boxes_by_clip(box_path) if box_path is not None else {}
    cameras = {}
    if homographies.exists():
        with np.load(homographies, allow_pickle=False) as data:
            for point, H in zip(data["pts"], data["H"], strict=True):
                cameras.setdefault(f"pt{int(point):04d}", CameraEvidence(homography=H))
    if projections.exists():
        with np.load(projections, allow_pickle=False) as data:
            for clip, frame, reliable, source in zip(
                data["clips"], data["frames"], data["reliable"], data["source"], strict=True
            ):
                camera = cameras.setdefault(str(clip), CameraEvidence())
                camera.sources.add(str(source))
                if reliable:
                    camera.reliable_frames.add(int(frame))
    actor_inputs = None
    if native_actor_continuity:
        if box_path is None:
            raise ValueError("native actor continuity requires original sided anchors")
        actor_inputs = load_actor_inputs(match_dir, box_path)
        for path in actor_inputs.paths:
            if path not in paths:
                records.append((path, file_record(path)))
    result = MatchLeakageInputs(match_dir, boxes, cameras, records, box_path, actor_inputs)
    result.assert_unchanged()
    return result


def ground_scale_px_per_m(homography: np.ndarray, court_point) -> float | None:
    if abs(np.linalg.det(homography)) < 1e-12:
        return None
    inverse = np.linalg.inv(homography)
    left = apply_h(inverse, (court_point[0] - 0.5, court_point[1]))
    right = apply_h(inverse, (court_point[0] + 0.5, court_point[1]))
    scale = float(np.linalg.norm(right - left))
    return scale if scale > 1e-6 else None


def sample_court_points(
    homography: np.ndarray, width: int, height: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if abs(np.linalg.det(homography)) < 1e-12:
        return []
    inverse = np.linalg.inv(homography)
    samples = []
    for (x0, y0), (x1, y1) in COURT_LINES:
        length = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(int(length / LINE_SAMPLE_STEP_M), 1)
        for index in range(steps + 1):
            fraction = index / steps
            ahead_fraction = min(fraction + 0.02, 1.0)
            point = apply_h(
                inverse,
                (x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction),
            )
            ahead = apply_h(
                inverse,
                (x0 + (x1 - x0) * ahead_fraction, y0 + (y1 - y0) * ahead_fraction),
            )
            direction = ahead - point
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            perpendicular = np.array([-direction[1], direction[0]]) / norm
            if (
                LINE_MARGIN_PX <= point[0] < width - LINE_MARGIN_PX
                and LINE_MARGIN_PX <= point[1] < height - LINE_MARGIN_PX
            ):
                samples.append((point, perpendicular))
    return samples


def _support_at_shift(
    gray: np.ndarray,
    points: np.ndarray,
    offsets: np.ndarray,
    shift: np.ndarray,
) -> float:
    height, width = gray.shape[:2]
    probes = points[:, None, :] + offsets + shift[None, None, :]
    coordinates = np.rint(probes).astype(int)
    valid = (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] < width)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] < height)
    ).all(axis=1)
    if valid.sum() < max(int(0.5 * len(points)), 1):
        return 0.0
    coordinates = coordinates[valid]
    values = gray[coordinates[..., 1], coordinates[..., 0]].astype(np.float32)
    line = values[:, 0]
    background = np.median(values[:, 1:], axis=1)
    return float(np.mean(line - background >= LINE_CONTRAST))


def court_support(gray: np.ndarray, samples) -> tuple[float, int]:
    if not samples:
        return 0.0, 0
    points = np.asarray([point for point, _ in samples], dtype=np.float64)
    normals = np.asarray([normal for _, normal in samples], dtype=np.float64)
    offsets = np.zeros((len(samples), 1 + 2 * len(LINE_PERP_PX), 2))
    for index, distance in enumerate(LINE_PERP_PX):
        offsets[:, 1 + 2 * index, :] = normals * distance
        offsets[:, 2 + 2 * index, :] = -normals * distance
    best = 0.0
    shifts = np.arange(-SHIFT_RANGE_PX, SHIFT_RANGE_PX + 1, SHIFT_STEP_PX)
    for vertical in shifts:
        for horizontal in shifts:
            best = max(
                best,
                _support_at_shift(gray, points, offsets, np.array([horizontal, vertical])),
            )
    return best, len(samples)


def _box_evidence(
    boxes: dict[str, dict[int, BoxObservation]],
    homography: np.ndarray | None,
    frame: int,
) -> tuple[float | None, float | None, bool, bool, bool]:
    if homography is None:
        return None, None, False, False, False
    apparent = {"near": None, "far": None}
    present = {"near": False, "far": False}
    conflict = False
    for side in ("near", "far"):
        observation = boxes[side].get(frame)
        if observation is None:
            continue
        present[side] = True
        root = np.array(
            [
                (observation.box[0] + observation.box[2]) / 2.0,
                observation.box[3],
            ]
        )
        court = apply_h(homography, root)
        scale = ground_scale_px_per_m(homography, court)
        if scale is not None:
            apparent[side] = float(observation.box[3] - observation.box[1]) / scale
        conflict |= side == "near" and court[1] > NET_Y_M + 2.0
        conflict |= side == "far" and court[1] < NET_Y_M - 2.0
    return apparent["near"], apparent["far"], present["near"], present["far"], conflict


def evidence_for_clip(
    match_dir: Path,
    clip: str,
    frames: list[int],
    boxes: dict[str, dict[int, BoxObservation]],
    camera: CameraEvidence,
    *,
    measure_court_lines: bool = True,
) -> list[FrameEvidence]:
    samples = None
    output = []
    for frame in frames:
        readable = False
        support = None
        sample_count = 0
        path = match_dir / "audit_frames_native_1080" / clip / f"f_{frame:04d}.jpg"
        if camera.homography is not None and path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                readable = True
                if samples is None and measure_court_lines:
                    samples = sample_court_points(
                        camera.homography,
                        image.shape[1],
                        image.shape[0],
                    )
                if measure_court_lines:
                    support, sample_count = court_support(image, samples)
        near, far, near_present, far_present, side_conflict = _box_evidence(
            boxes,
            camera.homography,
            frame,
        )
        output.append(
            FrameEvidence(
                frame,
                support,
                sample_count,
                readable,
                near,
                far,
                near_present,
                far_present,
                side_conflict,
            )
        )
    return output


def point_reference(evidence: list[FrameEvidence]) -> PointReference:
    support = [
        row.court_support
        for row in evidence
        if row.frame_readable
        and row.court_support is not None
        and row.court_samples >= MINIMUM_COURT_SAMPLES
    ]
    heights = [
        value
        for row in evidence
        for value in (row.apparent_near_m, row.apparent_far_m)
        if value is not None
    ]
    return PointReference(
        float(np.median(support)) if support else None,
        float(np.median(heights)) if heights else None,
    )


def leak_reasons(row: FrameEvidence, reference: PointReference) -> list[str]:
    reasons = []
    if (
        reference.court_arm_enabled
        and row.frame_readable
        and row.court_support is not None
        and row.court_samples >= MINIMUM_COURT_SAMPLES
        and row.court_support <= RELATIVE_SUPPORT_DROP * reference.support
    ):
        reasons.append("court_support_lost")
    if reference.box_arm_enabled and any(
        value is not None and value > GIANT_APPARENT_HEIGHT_M
        for value in (row.apparent_near_m, row.apparent_far_m)
    ):
        reasons.append("giant_player_box")
    return reasons


def leak_runs(
    evidence: list[FrameEvidence],
    fps: float,
    reference: PointReference,
) -> list[dict]:
    flags = [(row.frame, bool(leak_reasons(row, reference))) for row in evidence]
    minimum_run = max(int(round(MINIMUM_LEAK_RUN_SECONDS * fps)), 2)
    candidates = []
    start = None
    last_leak = None
    for index, (_, leak) in enumerate(flags):
        if leak:
            start = index if start is None else start
            last_leak = index
        elif start is not None and index - last_leak > LEAK_GAP_BRIDGE_FRAMES:
            candidates.append((start, last_leak))
            start = None
    if start is not None:
        candidates.append((start, last_leak))
    output = []
    for start_index, end_index in candidates:
        local = flags[start_index : end_index + 1]
        density = sum(leak for _, leak in local) / len(local)
        if len(local) < minimum_run or density < LEAK_RUN_DENSITY:
            continue
        reasons = sorted(
            {
                reason
                for row in evidence[start_index : end_index + 1]
                for reason in leak_reasons(row, reference)
            }
        )
        output.append(
            {
                "start_frame": local[0][0],
                "end_frame": local[-1][0],
                "frames": len(local),
                "density": round(density, 3),
                "reasons": reasons,
            }
        )
    return output


def snap_to_shots(run: dict, shots: list[dict]) -> dict:
    output = dict(run)
    for boundary_field, boundaries in (
        ("start_frame", [int(row["start_frame"]) for row in shots]),
        ("end_frame", [int(row["end_frame"]) for row in shots]),
    ):
        nearby = [
            value for value in boundaries if abs(value - run[boundary_field]) <= SHOT_SNAP_FRAMES
        ]
        if nearby:
            output[boundary_field] = min(nearby, key=lambda value: abs(value - run[boundary_field]))
            output.setdefault("snapped", []).append(boundary_field)
    output["frames"] = int(output["end_frame"] - output["start_frame"] + 1)
    return output


def visual_play_support(
    run: dict, evidence: list[FrameEvidence], *, actor_witnesses: list[dict] | None = None
) -> dict:
    """Describe visual actor support, independently of static court-line contrast.

    This is evidence that pictures can still contain ordinary play, not a claim
    of reliable metric calibration or competitive event membership.
    """
    local = [row for row in evidence if run["start_frame"] <= row.frame <= run["end_frame"]]
    expected = int(run["end_frame"] - run["start_frame"] + 1)
    low, high = HEIGHT_REFERENCE_BAND_M
    continuity_frames = {
        obs["frame"]
        for witness in actor_witnesses or []
        if witness["supported"]
        for obs in witness["observations"]
    }
    supported = sum(
        row.frame_readable
        and (
            row.frame in continuity_frames
            or (
                row.near_present
                and row.far_present
                and not row.side_conflict
                and all(
                    value is not None and low <= value <= high
                    for value in (row.apparent_near_m, row.apparent_far_m)
                )
            )
        )
        for row in local
    )
    original_supported = sum(
        row.frame_readable
        and row.near_present
        and row.far_present
        and not row.side_conflict
        and all(
            value is not None and low <= value <= high
            for value in (row.apparent_near_m, row.apparent_far_m)
        )
        for row in local
    )
    conflicting = any(
        row.side_conflict
        or any(
            value is not None and value > GIANT_APPARENT_HEIGHT_M
            for value in (row.apparent_near_m, row.apparent_far_m)
        )
        for row in local
    )
    fraction = supported / expected if expected > 0 else 0.0
    return {
        "schema": "native_two_player_visual_support_v1",
        "start_frame": run["start_frame"],
        "end_frame": run["end_frame"],
        "expected_frames": expected,
        "observed_frames": len({row.frame for row in local}),
        "normal_two_player_frames": supported,
        "normal_two_player_fraction": fraction,
        "minimum_fraction": 0.8,
        "native_actor_continuity_frames": len(continuity_frames & {row.frame for row in local}),
        "metric_scaled_actor_support_frames": original_supported,
        "native_actor_witnesses": actor_witnesses or [],
        "conflicting_actor_evidence": conflicting,
        "supported": len({row.frame for row in local}) == expected
        and fraction >= 0.8
        and not conflicting,
        "camera_reliability_upgraded": False,
        "competitive_phase_inferred": False,
    }


def enrich_visual_support(
    match_dir: Path,
    clip: str,
    proposal: dict,
    *,
    inputs: MatchLeakageInputs,
    entry: dict | None = None,
) -> dict:
    """Reuse original court-loss proposals, measuring fresh independent actor support.

    Reused proposals cannot supply their own positive visual verdict. This avoids
    repeating the costly unchanged static-line sweep while reading actual native
    pictures and bound automatic sided boxes for every proposed run.
    """
    if inputs.match_dir.resolve() != match_dir.resolve():
        raise ValueError("visual support inputs belong to a different match")
    runs = proposal.get("proposed_trims", [])
    frames = sorted(
        {
            frame
            for run in runs
            for frame in range(int(run["start_frame"]), int(run["end_frame"]) + 1)
        }
    )
    evidence = evidence_for_clip(
        match_dir,
        clip,
        frames,
        inputs.boxes.get(clip, {"near": {}, "far": {}}),
        inputs.cameras.get(clip, CameraEvidence()),
        measure_court_lines=False,
    )
    actor_witnesses = []
    if inputs.actor_inputs is not None:
        if entry is None:
            raise ValueError("native actor continuity requires original shot and cadence scope")
        actor_witnesses = witnesses(
            inputs.actor_inputs,
            clip,
            set(frames),
            shots=entry.get("shots") or [],
            fps=float(entry["fps"]),
        )
    return proposal | {
        "proposed_trims": [
            run
            | {
                "visual_play_support": visual_play_support(
                    run, evidence, actor_witnesses=actor_witnesses
                )
            }
            for run in runs
        ]
    }


def propose_for_point(
    match_dir: Path, clip: str, entry: dict, *, inputs: MatchLeakageInputs | None = None
) -> tuple[list[FrameEvidence], dict]:
    fps = float(entry.get("fps") or 25.0)
    active_spans = entry.get("active_spans") or []
    frames = sorted(
        {
            frame
            for start, end in active_spans
            for frame in range(math.ceil(start), math.floor(end) + 1)
        }
    )
    if inputs is None:
        box_paths = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))
        boxes = _load_boxes(box_paths[0], clip) if box_paths else {"near": {}, "far": {}}
        camera = _load_camera(match_dir, clip)
    else:
        if inputs.match_dir.resolve() != match_dir.resolve():
            raise ValueError("leakage inputs belong to a different match")
        boxes = inputs.boxes.get(clip, {"near": {}, "far": {}})
        camera = inputs.cameras.get(clip, CameraEvidence())
    evidence = evidence_for_clip(match_dir, clip, frames, boxes, camera)
    reference = point_reference(evidence)
    runs = [
        snap_to_shots(run, entry.get("shots") or []) for run in leak_runs(evidence, fps, reference)
    ]
    for run in runs:
        run["visual_play_support"] = visual_play_support(run, evidence)
    sources = sorted(camera.sources)
    return evidence, {
        "clip": clip,
        "fps": fps,
        "active_spans": active_spans,
        "proposed_trims": runs,
        "active_frames": len(frames),
        "trimmed_frames": sum(run["frames"] for run in runs),
        "frames_unreadable": sum(not row.frame_readable for row in evidence),
        "homography_missing": camera.homography is None,
        "support_ref": reference.support,
        "height_ref": reference.apparent_height_m,
        "court_arm_enabled": reference.court_arm_enabled,
        "box_arm_enabled": reference.box_arm_enabled,
        "calibration_provenance": {
            "source": sources or ["court_H_per_point"],
            "residuals": None,
            "frame_scope": active_spans,
            "fallback_ancestry": [source for source in sources if "fallback" in source],
            "reliable_frame_fraction": (
                len(camera.reliable_frames & set(frames)) / len(frames) if frames else 0.0
            ),
        },
    }


def run_match(match_dir: Path, match_id: str, active: dict, points: set[str] | None) -> dict:
    report = {
        "schema": SCHEMA,
        "match_id": match_id,
        "thresholds": THRESHOLDS,
        "labels_loaded": False,
        "points": {},
    }
    inputs = load_match_inputs(match_dir)
    for key, entry in sorted(active.items()):
        if not key.startswith(f"{match_id}/"):
            continue
        clip = entry["clip"]
        if points and clip not in points:
            continue
        _, proposal = propose_for_point(match_dir, clip, entry, inputs=inputs)
        report["points"][clip] = proposal
    inputs.assert_unchanged()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Match artifact directory")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--active-play", type=Path, required=True)
    parser.add_argument("--point", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_match(
        args.out,
        args.match_id,
        json.loads(args.active_play.read_text()),
        set(args.point) or None,
    )
    reused_paths = [args.active_play]
    reused_paths.extend(
        path
        for path in (
            args.out / "court_H_per_point.npz",
            args.out / "camera_P_per_frame_v1.npz",
            *sorted(args.out.glob("player_boxes_*_native_sided_v1.csv")),
        )
        if path.is_file()
    )
    report["provenance"] = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=AUTOMATIC_MODE,
        configuration={
            "stage": "play_camera_leakage",
            "match_id": args.match_id,
            "points": sorted(args.point),
            "thresholds": THRESHOLDS,
            "labels_loaded": False,
        },
        reused_artifacts=[file_record(path, role="stage_input") for path in reused_paths],
    )
    output = args.output or args.out / "play_camera_leakage_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
