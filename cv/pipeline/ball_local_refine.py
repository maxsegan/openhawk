"""Recover missing ball detections with trajectory- and player-centered local crops.

Full-frame ball detectors occasionally lose a visible ball because it occupies only a few
pixels or because a stronger response wins elsewhere. This stage revisits only gaps in a
high-confidence lock track. Short-range trajectory extrapolations provide one crop family;
the two likely tennis players provide another for longer, curved gaps near racket contact.

The output contains detector observations with explicit crop provenance. It never fills a
frame by interpolation. Downstream association must still decide whether each observation
belongs to the ball, and genuinely occluded gaps remain missing.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.track_artifact import rows_with_native_mirror, write_track_artifact
from cv.pipeline.ball_neural import (
    INPUT_WH,
    MEAN,
    STD,
    build_model,
    subpixel_refine,
    topk_peaks,
)


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    x: float
    y: float
    score: float = 1.0
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerBox:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class RecoveryRegion:
    clip: str
    frame: int
    center_x: float
    center_y: float
    width: float
    height: float
    provenance: str


def frame_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def load_track(path: Path) -> dict[str, list[TrackPoint]]:
    output: dict[str, list[TrackPoint]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[row["clip"]].append(
                TrackPoint(
                    frame_number(row["frame"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row.get("score", 1.0) or 0.0),
                    tuple(filter(None, row.get("sources", "").split("+"))),
                )
            )
    return {clip: sorted(points, key=lambda point: point.frame) for clip, points in output.items()}


def load_player_boxes(path: Path) -> dict[tuple[str, int], list[PlayerBox]]:
    output: dict[tuple[str, int], list[PlayerBox]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[row["clip"], frame_number(row["frame"])].append(
                PlayerBox(
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row.get("conf", 1.0) or 0.0),
                )
            )
    return output


def likely_players(
    boxes: list[PlayerBox],
    half_split: float = 270.0,
    minimum_confidence: float = 0.5,
) -> list[tuple[str, PlayerBox]]:
    """Choose the largest credible person in each image half.

    Broadcast tennis players dominate box area within their respective near/far image
    halves. This intentionally remains a proposal mechanism: downstream trajectory
    association rejects line judges or spectators when that approximation fails.
    """
    credible = [box for box in boxes if box.confidence >= minimum_confidence]
    near = [box for box in credible if box.center[1] >= half_split]
    far = [box for box in credible if box.center[1] < half_split]
    selected = []
    if near:
        selected.append(("player_near", max(near, key=lambda box: box.area)))
    if far:
        selected.append(("player_far", max(far, key=lambda box: box.area)))
    return selected


def _side_fit(points: list[TrackPoint], side: str, samples: int) -> tuple[np.ndarray, np.ndarray]:
    chosen = points[-samples:] if side == "left" else points[:samples]
    frames = np.array([point.frame for point in chosen], dtype=np.float64)
    coordinates = np.array([[point.x, point.y] for point in chosen], dtype=np.float64)
    if len(np.unique(frames)) < 2:
        raise ValueError("at least two unique frames are required")
    design = np.column_stack([frames, np.ones_like(frames)])
    coefficients, _, _, _ = np.linalg.lstsq(design, coordinates, rcond=None)
    return coefficients[0], coefficients[1]


def _predict(frame: int, fit: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    slope, intercept = fit
    point = slope * frame + intercept
    return float(point[0]), float(point[1])


def build_recovery_regions(
    clip: str,
    frame_count: int,
    track: list[TrackPoint],
    player_boxes: dict[tuple[str, int], list[PlayerBox]],
    maximum_gap: int = 40,
    trajectory_horizon: int = 14,
    side_samples: int = 5,
    trajectory_crop: tuple[float, float] = (384.0, 216.0),
    bidirectional_crop: tuple[float, float] = (192.0, 108.0),
    player_crop: tuple[float, float] = (320.0, 180.0),
) -> list[RecoveryRegion]:
    """Create local detector crops inside bounded gaps of the lock track."""
    if len(track) < 2:
        return []
    by_frame = {point.frame: point for point in track}
    ordered = sorted(by_frame)
    regions = []
    for left_frame, right_frame in zip(ordered, ordered[1:]):
        gap = right_frame - left_frame - 1
        if gap <= 0 or gap > maximum_gap:
            continue
        left_points = [
            by_frame[frame]
            for frame in ordered
            if left_frame - trajectory_horizon <= frame <= left_frame
        ]
        right_points = [
            by_frame[frame]
            for frame in ordered
            if right_frame <= frame <= right_frame + trajectory_horizon
        ]
        fits = []
        if len(left_points) >= 2:
            fits.append(("trajectory_forward", _side_fit(left_points, "left", side_samples)))
        if len(right_points) >= 2:
            fits.append(("trajectory_backward", _side_fit(right_points, "right", side_samples)))
        for frame in range(left_frame + 1, right_frame):
            if not 1 <= frame <= frame_count:
                continue
            for provenance, fit in fits:
                distance = (
                    frame - left_frame if provenance.endswith("forward") else right_frame - frame
                )
                if distance > trajectory_horizon:
                    continue
                center_x, center_y = _predict(frame, fit)
                regions.append(
                    RecoveryRegion(
                        clip,
                        frame,
                        center_x,
                        center_y,
                        *trajectory_crop,
                        provenance,
                    )
                )
            predictions = {
                provenance: _predict(frame, fit)
                for provenance, fit in fits
                if (frame - left_frame if provenance.endswith("forward") else right_frame - frame)
                <= trajectory_horizon
            }
            if {"trajectory_forward", "trajectory_backward"} <= predictions.keys():
                forward = predictions["trajectory_forward"]
                backward = predictions["trajectory_backward"]
                disagreement_x = abs(forward[0] - backward[0])
                disagreement_y = abs(forward[1] - backward[1])
                regions.append(
                    RecoveryRegion(
                        clip,
                        frame,
                        (forward[0] + backward[0]) / 2.0,
                        (forward[1] + backward[1]) / 2.0,
                        min(
                            trajectory_crop[0],
                            bidirectional_crop[0] + disagreement_x,
                        ),
                        min(
                            trajectory_crop[1],
                            bidirectional_crop[1] + disagreement_y,
                        ),
                        "trajectory_bidirectional",
                    )
                )
            for provenance, box in likely_players(player_boxes.get((clip, frame), [])):
                center_x, center_y = box.center
                crop_width = max(player_crop[0], (box.x1 - box.x0) * 2.5)
                crop_height = max(player_crop[1], (box.y1 - box.y0) * 1.5)
                regions.append(
                    RecoveryRegion(
                        clip,
                        frame,
                        center_x,
                        center_y,
                        crop_width,
                        crop_height,
                        provenance,
                    )
                )
    deduplicated = {}
    for region in regions:
        key = (
            region.clip,
            region.frame,
            round(region.center_x / 12.0),
            round(region.center_y / 12.0),
            region.provenance,
        )
        deduplicated[key] = region
    return list(deduplicated.values())


def build_persistent_lock_regions(
    clip: str,
    frame_count: int,
    track: list[TrackPoint],
    fps: float,
    native_size: res.FrameSize,
    artifact_size: res.FrameSize,
    native_crop: tuple[float, float] = (512.0, 288.0),
    maximum_expansion: float = 2.0,
    maximum_extrapolation_seconds: float = 0.6,
    branch_disagreement_px: float = 45.0,
    side_samples: int = 5,
) -> list[RecoveryRegion]:
    """Center one true-native crop per frame from past/future lock extrapolations."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(track) < 2:
        return []
    ordered = sorted(track, key=lambda point: point.frame)
    frames = [point.frame for point in ordered]
    by_frame = {point.frame: point for point in ordered}
    maximum_extrapolation = max(1, round(maximum_extrapolation_seconds * fps))
    fit_horizon = max(2, round(0.28 * fps))
    base_crop = res.scale_points(
        np.asarray(native_crop, dtype=float),
        native_size,
        artifact_size,
    )
    regions = []

    def add_region(
        frame: int,
        center: tuple[float, float],
        nearest_distance: int,
        provenance: str,
    ) -> None:
        expansion = min(
            maximum_expansion,
            1.0 + nearest_distance / maximum_extrapolation,
        )
        regions.append(
            RecoveryRegion(
                clip,
                frame,
                center[0],
                center[1],
                float(base_crop[0] * expansion),
                float(base_crop[1] * expansion),
                provenance,
            )
        )

    for frame in range(1, frame_count + 1):
        exact = by_frame.get(frame)
        if exact is not None:
            add_region(
                frame,
                (exact.x, exact.y),
                0,
                "persistent_lock_observed",
            )
            continue
        else:
            insertion = bisect.bisect_left(frames, frame)
            left = ordered[max(0, insertion - side_samples) : insertion]
            right = ordered[insertion : insertion + side_samples]
            left = [point for point in left if frame - point.frame <= fit_horizon]
            right = [point for point in right if point.frame - frame <= fit_horizon]
            predictions = {}
            if len(left) >= 2 and frame - left[-1].frame <= maximum_extrapolation:
                predictions["forward"] = _predict(
                    frame,
                    _side_fit(left, "left", side_samples),
                )
            if len(right) >= 2 and right[0].frame - frame <= maximum_extrapolation:
                predictions["backward"] = _predict(
                    frame,
                    _side_fit(right, "right", side_samples),
                )
            if not predictions:
                continue
            if {"forward", "backward"} <= predictions.keys():
                left_distance = frame - left[-1].frame
                right_distance = right[0].frame - frame
                if (
                    math.dist(predictions["forward"], predictions["backward"])
                    > branch_disagreement_px
                ):
                    add_region(
                        frame,
                        predictions["forward"],
                        left_distance,
                        "persistent_lock_forward_branch",
                    )
                    add_region(
                        frame,
                        predictions["backward"],
                        right_distance,
                        "persistent_lock_backward_branch",
                    )
                    continue
                span = max(1, left_distance + right_distance)
                forward_weight = right_distance / span
                backward_weight = left_distance / span
                center = (
                    predictions["forward"][0] * forward_weight
                    + predictions["backward"][0] * backward_weight,
                    predictions["forward"][1] * forward_weight
                    + predictions["backward"][1] * backward_weight,
                )
                nearest_distance = min(left_distance, right_distance)
                provenance = "persistent_lock_bidirectional"
            elif "forward" in predictions:
                center = predictions["forward"]
                nearest_distance = frame - left[-1].frame
                provenance = "persistent_lock_forward"
            else:
                center = predictions["backward"]
                nearest_distance = right[0].frame - frame
                provenance = "persistent_lock_backward"
        add_region(frame, center, nearest_distance, provenance)
    return regions


def _crop_tensor(
    frame: np.ndarray,
    region: RecoveryRegion,
    artifact_size: res.FrameSize,
) -> np.ndarray:
    image_size = res.FrameSize(frame.shape[1], frame.shape[0])
    center = res.scale_points(
        np.array([region.center_x, region.center_y]),
        artifact_size,
        image_size,
    )
    crop_size = res.scale_points(
        np.array([region.width, region.height]),
        artifact_size,
        image_size,
    )
    left = center[0] - crop_size[0] / 2.0
    top = center[1] - crop_size[1] / 2.0
    transform = np.array(
        [
            [crop_size[0] / INPUT_WH[0], 0.0, left],
            [0.0, crop_size[1] / INPUT_WH[1], top],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        frame,
        transform,
        INPUT_WH,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return normalized.transpose(2, 0, 1)


def heatmap_to_artifact(
    x: float,
    y: float,
    heatmap_width: int,
    heatmap_height: int,
    region: RecoveryRegion,
) -> tuple[float, float]:
    left = region.center_x - region.width / 2.0
    top = region.center_y - region.height / 2.0
    return (
        left + (x + 0.5) * region.width / heatmap_width,
        top + (y + 0.5) * region.height / heatmap_height,
    )


def refined_peaks(
    heatmap: np.ndarray,
    k_best: int,
    peak_threshold: float,
    nms_radius: int,
    subpixel: str = "centroid",
) -> list[tuple[float, float, float]]:
    """Return NMS peaks refined within their own heatmap neighbourhood."""
    return [
        (*subpixel_refine(heatmap, x, y, subpixel), score)
        for x, y, score in topk_peaks(heatmap, k_best, peak_threshold, nms_radius)
    ]


def supported_peaks(
    heatmap: np.ndarray,
    region: RecoveryRegion,
    artifact_size: res.FrameSize,
    k_best: int,
    peak_threshold: float,
    nms_radius: int,
    subpixel: str,
    audit: dict | None = None,
) -> list[tuple[float, float, float]]:
    """Mask reflected padding before NMS; refine only native-supported signal."""
    height, width = heatmap.shape
    ys, xs = np.indices(heatmap.shape)
    image_x, image_y = heatmap_to_artifact(xs, ys, width, height, region)
    supported = res.points_inside_image(np.stack([image_x, image_y], axis=-1), artifact_size)
    supported &= np.isfinite(heatmap)
    if audit is not None:
        audit["regions"] = audit.get("regions", 0) + 1
        audit["unsupported_heatmap_cells"] = audit.get("unsupported_heatmap_cells", 0) + int(
            (~supported).sum()
        )
        audit["fully_unsupported_regions"] = audit.get("fully_unsupported_regions", 0) + int(
            not supported.any()
        )
    if not supported.any():
        return []
    # Exact old arithmetic/order for entirely in-picture heatmaps.
    if supported.all():
        refined = refined_peaks(heatmap, k_best, peak_threshold, nms_radius, subpixel)
    else:
        ranked = np.where(supported, heatmap, -np.inf)
        # Padding must not bias the local centroid/parabola of a genuine border peak.
        measured = np.where(supported, heatmap, 0.0)
        refined = [
            (*subpixel_refine(measured, x, y, subpixel), score)
            for x, y, score in topk_peaks(ranked, k_best, peak_threshold, nms_radius)
        ]
    peaks = []
    for px, py, score in refined:
        point = heatmap_to_artifact(px, py, width, height, region)
        if not res.points_inside_image(point, artifact_size):
            if audit is not None:
                audit["refined_centres_outside_image"] = (
                    audit.get("refined_centres_outside_image", 0) + 1
                )
            continue
        peaks.append((px, py, score))
    return peaks


def ensemble_target_heatmaps(
    outputs: np.ndarray,
    target_channels: tuple[int, ...],
) -> np.ndarray:
    if outputs.ndim != 5:
        raise ValueError("outputs must be regions x contexts x channels x height x width")
    if outputs.shape[1] != len(target_channels):
        raise ValueError("target channel count must match context count")
    selected = np.stack(
        [outputs[:, index, channel] for index, channel in enumerate(target_channels)],
        axis=1,
    )
    return np.mean(selected, axis=1)


def infer_regions(
    model,
    frames_dir: Path,
    regions: list[RecoveryRegion],
    device: int,
    artifact_size: res.FrameSize,
    batch_size: int = 32,
    k_best: int = 5,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = True,
    subpixel: str = "centroid",
    native_support_audit: dict | None = None,
) -> list[dict]:
    import torch

    frame_cache: dict[tuple[str, int], np.ndarray] = {}
    frame_counts = {
        clip: len(list((frames_dir / clip).glob("f_*.jpg")))
        for clip in {region.clip for region in regions}
    }

    def read_frame(clip: str, frame: int) -> np.ndarray:
        key = (clip, frame)
        if key not in frame_cache:
            path = frames_dir / clip / f"f_{frame:04d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"cannot read {path}")
            frame_cache[key] = image
        return frame_cache[key]

    contexts = (
        (
            ((-2, -1, 0), 2),
            ((-1, 0, 1), 1),
            ((0, 1, 2), 0),
        )
        if temporal_ensemble
        else (((-1, 0, 1), 1),)
    )
    regions_per_batch = max(1, batch_size // len(contexts))
    rows = []
    for offset in range(0, len(regions), regions_per_batch):
        chunk = regions[offset : offset + regions_per_batch]
        samples = []
        for region in chunk:
            for offsets, _ in contexts:
                context = []
                for frame_offset in offsets:
                    frame = region.frame + frame_offset
                    bounded = max(1, min(frame, frame_counts[region.clip]))
                    context.append(
                        _crop_tensor(read_frame(region.clip, bounded), region, artifact_size)
                    )
                samples.append(np.concatenate(context, axis=0))
        inputs = torch.from_numpy(np.stack(samples)).to(f"cuda:{device}")
        with torch.inference_mode():
            outputs = model(inputs)[0].sigmoid().cpu().numpy()
        outputs = outputs.reshape(len(chunk), len(contexts), *outputs.shape[1:])
        heatmaps = ensemble_target_heatmaps(
            outputs,
            tuple(channel for _, channel in contexts),
        )
        for region, sample in zip(chunk, heatmaps, strict=True):
            heatmap = sample
            for rank, (x, y, score) in enumerate(
                supported_peaks(
                    heatmap,
                    region,
                    artifact_size,
                    k_best,
                    peak_threshold,
                    nms_radius,
                    subpixel,
                    native_support_audit,
                )
            ):
                artifact_x, artifact_y = heatmap_to_artifact(
                    x,
                    y,
                    heatmap.shape[1],
                    heatmap.shape[0],
                    region,
                )
                rows.append(
                    {
                        "clip": region.clip,
                        "frame": f"f_{region.frame:04d}.jpg",
                        "x": artifact_x,
                        "y": artifact_y,
                        "score": score,
                        "on_court": True,
                        "rank": rank,
                        "crop_provenance": region.provenance,
                        "crop_center_x": region.center_x,
                        "crop_center_y": region.center_y,
                        "crop_width": region.width,
                        "crop_height": region.height,
                        "temporal_context": (
                            "past_center_future" if temporal_ensemble else "centered"
                        ),
                    }
                )
    return rows


def write_candidate_artifact(path: Path, rows: list[dict], coordinate_source: Path) -> None:
    """Write local candidates with the same dual-coordinate contract as their lock track."""
    write_track_artifact(path, rows, [coordinate_source])


def write_candidate_artifact_batches(
    path: Path,
    batches: Iterable[list[dict]],
    coordinate_source: Path,
) -> int:
    """Append ordered candidate batches without retaining the match-wide row table."""
    manifest = res.read_coordinate_manifest(coordinate_source)
    if manifest is None:
        raise ValueError(f"coordinate manifest missing for track source: {coordinate_source}")
    image_size = res.FrameSize(
        int(manifest["image_size"]["width"]),
        int(manifest["image_size"]["height"]),
    )
    legacy_size = res.manifest_artifact_size(manifest, legacy_columns=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = None
    with path.open("w", newline="") as handle:
        for rows in batches:
            if not rows:
                continue
            if writer is None:
                fieldnames = [*list(rows[0]), "x_native", "y_native"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            mirrored = rows_with_native_mirror(rows, legacy_size)
            writer.writerows(mirrored)
            count += len(rows)
        if writer is None:
            writer = csv.DictWriter(
                handle,
                fieldnames=["clip", "frame", "x", "y", "track_id", "x_native", "y_native"],
            )
            writer.writeheader()
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=image_size,
        legacy_size=legacy_size,
        source=os.fspath(coordinate_source),
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
        extra={"artifact": path.name},
    )
    return count


def main(*, infer_region_batches=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--lock-track", type=Path, required=True)
    parser.add_argument("--player-boxes", type=Path, required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=("wasb", "tracknetv2"), default="wasb")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--maximum-gap", type=int, default=40)
    parser.add_argument("--trajectory-horizon", type=int, default=14)
    parser.add_argument("--centered-only", action="store_true")
    parser.add_argument(
        "--subpixel",
        choices=("argmax", "parabolic", "centroid"),
        default="centroid",
        help="refine each crop heatmap peak below one cell; defaults to centroid",
    )
    parser.add_argument("--persistent-lock", action="store_true")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--native-crop-width", type=float, default=512.0)
    parser.add_argument("--native-crop-height", type=float, default=288.0)
    parser.add_argument("--maximum-expansion", type=float, default=2.0)
    parser.add_argument("--maximum-extrapolation-seconds", type=float, default=0.6)
    parser.add_argument("--branch-disagreement-px", type=float, default=45.0)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-width", type=int, default=960)
    parser.add_argument("--artifact-height", type=int, default=540)
    parser.add_argument(
        "--max-resident-gb",
        type=float,
        default=32.0,
        help="hard host-RSS ceiling for the streaming batched runtime (default: 32)",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="bounded prepared-batch queue depth for the streaming batched runtime",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="JPEG decode/crop worker count for the streaming batched runtime",
    )
    parser.add_argument(
        "--legacy-buffered",
        action="store_true",
        help="use the legacy unbounded decoded-frame cache instead of streaming",
    )
    args = parser.parse_args()

    if args.max_resident_gb <= 0:
        parser.error("--max-resident-gb must be positive")
    if args.prefetch_batches < 1:
        parser.error("--prefetch-batches must be at least one")
    if args.decode_workers < 1:
        parser.error("--decode-workers must be at least one")

    artifact_size = res.FrameSize(args.artifact_width, args.artifact_height)
    track = load_track(args.lock_track)
    # Persistent-lock refinement never consults player boxes. The match-wide RG table has
    # 3.27 million rows, so parsing it here used gigabytes before the first crop was decoded.
    boxes = {} if args.persistent_lock else load_player_boxes(args.player_boxes)
    clip_filter = set(args.clip)
    regions = []
    for clip, points in track.items():
        if clip_filter and clip not in clip_filter:
            continue
        frame_count = len(list((args.frames_dir / clip).glob("f_*.jpg")))
        if args.persistent_lock:
            if args.fps is None:
                parser.error("--persistent-lock requires --fps")
            probe = cv2.imread(str(args.frames_dir / clip / "f_0001.jpg"))
            if probe is None:
                raise OSError(f"cannot read first frame for {clip}")
            regions.extend(
                build_persistent_lock_regions(
                    clip,
                    frame_count,
                    points,
                    args.fps,
                    res.FrameSize(probe.shape[1], probe.shape[0]),
                    artifact_size,
                    native_crop=(args.native_crop_width, args.native_crop_height),
                    maximum_expansion=args.maximum_expansion,
                    maximum_extrapolation_seconds=args.maximum_extrapolation_seconds,
                    branch_disagreement_px=args.branch_disagreement_px,
                )
            )
        else:
            regions.extend(
                build_recovery_regions(
                    clip,
                    frame_count,
                    points,
                    boxes,
                    maximum_gap=args.maximum_gap,
                    trajectory_horizon=args.trajectory_horizon,
                )
            )
    if not regions:
        raise SystemExit("no bounded lock gaps produced recovery regions")
    model = build_model(args.external_root, args.model, args.weights, args.device)
    native_support_audit = {
        "policy": "native_image_point_support_v1",
        "padding_is_observation": False,
        "physical_court_membership_measured": False,
        "artifact_size": {"width": artifact_size.width, "height": artifact_size.height},
    }
    common = {
        "native_support_audit": native_support_audit,
        "batch_size": args.batch,
        "temporal_ensemble": not args.centered_only,
        "subpixel": args.subpixel,
    }
    if infer_region_batches is not None and not args.legacy_buffered:
        batches = infer_region_batches(
            model,
            args.frames_dir,
            regions,
            args.device,
            artifact_size,
            max_resident_gb=args.max_resident_gb,
            prefetch_batches=args.prefetch_batches,
            decode_workers=args.decode_workers,
            **common,
        )
        row_count = write_candidate_artifact_batches(args.output, batches, args.lock_track)
    else:
        rows = infer_regions(
            model,
            args.frames_dir,
            regions,
            args.device,
            artifact_size,
            **common,
        )
        write_candidate_artifact(args.output, rows, args.lock_track)
        row_count = len(rows)
    sidecar = res.coordinate_manifest_path(args.output)
    metadata = json.loads(sidecar.read_text())
    metadata["native_image_support"] = native_support_audit
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote {row_count} local candidates from {len(regions)} crops "
        f"across {len({region.clip for region in regions})} clips -> {args.output}"
    )


if __name__ == "__main__":
    main()
