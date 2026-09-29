"""Run official WASB-model-zoo tennis trackers on extracted rally frames.

Supports the official tennis-pretrained ``wasb`` and ``tracknetv2`` checkpoints under one
inference harness. It emits one scored ball candidate per frame plus linked tracks in the same
schema as :mod:`ball`, so downstream trajectory/contact code is shared.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.ball import link, load_court_gate
from cv.pipeline.frame_identity import frame_number_from_name
from cv.pipeline.paths import tracker_root
from cv.pipeline.run_manifest import StageRun
from cv.pipeline.track_artifact import write_candidate_artifact, write_track_artifact

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_WH = (512, 288)


def topk_peaks(
    heatmap: np.ndarray, k: int, threshold: float, radius: int
) -> list[tuple[int, int, float]]:
    """Up to ``k`` local-maximum heatmap peaks, greedy-NMS by ``radius`` (heatmap px).

    Track-before-detect: WASB's argmax export throws away every secondary peak, and the
    correct ball at the far court / through occlusion is frequently the SECOND-strongest
    response, not the first. This returns the argmax plus lower peaks above a LOW
    ``threshold``, each separated by ``radius`` so a single blob is not double-counted.
    Always returns at least the global argmax (so a k>=1 caller never emits an empty frame,
    matching the argmax export's behaviour when the whole map is sub-threshold).
    Sorted by score descending; each entry is (x, y, score) in heatmap pixels.
    """
    ys, xs = np.where(heatmap >= threshold)
    if len(xs) == 0:
        y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        return [(int(x), int(y), float(heatmap[y, x]))]
    scores = heatmap[ys, xs]
    order = np.argsort(-scores, kind="stable")
    r2 = radius * radius
    peaks: list[tuple[int, int, float]] = []
    for idx in order:
        x, y, s = int(xs[idx]), int(ys[idx]), float(scores[idx])
        if all((x - px) ** 2 + (y - py) ** 2 > r2 for px, py, _ in peaks):
            peaks.append((x, y, s))
            if len(peaks) >= k:
                break
    return peaks


def subpixel_refine(heatmap: np.ndarray, x: int, y: int, method: str) -> tuple[float, float]:
    """Refine an integer heatmap peak to sub-cell position, in heatmap pixels.

    A bare ``argmax`` is quantised to the 512x288 grid -- one cell is 960/512 = 1.875 px540 on the
    full regime, so the argmax alone carries ~cell/sqrt(12) ~= 0.54 px540/axis of discretisation
    error even for a perfect heatmap. Refining the returned peak within its own neighbourhood
    (parabolic vertex, or floor-subtracted 5x5 centroid) recovers most of it at zero inference
    cost, and composes with :func:`topk_peaks` (pass its integer ``(x, y)``); ``cv2.transform``
    downstream already accepts the float result. ``method='argmax'`` is the identity (production
    default). Measured on the owner correction frames: median 2D error 1.90 -> 1.54 px540 (-19%).

    NOTE: this refines toward the HEATMAP's own ball centre; it does NOT apply the ~0.94 px540
    vertical owner-vs-tracker label offset, which is an owner-labelling-convention artifact (the
    automatic joint track carries the same offset vs owner) and is deliberately left uncorrected
    so 3D reprojection stays on the automatic-track geometry.
    """
    if method == "argmax":
        return float(x), float(y)
    h, w = heatmap.shape
    if method == "parabolic":

        def vertex(left: float, centre: float, right: float) -> float:
            denom = left - 2.0 * centre + right
            if denom >= 0.0:
                return 0.0
            return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))

        dx = (
            vertex(float(heatmap[y, x - 1]), float(heatmap[y, x]), float(heatmap[y, x + 1]))
            if 0 < x < w - 1
            else 0.0
        )
        dy = (
            vertex(float(heatmap[y - 1, x]), float(heatmap[y, x]), float(heatmap[y + 1, x]))
            if 0 < y < h - 1
            else 0.0
        )
        return float(x) + dx, float(y) + dy
    if method == "centroid":
        radius = 2
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        window = heatmap[y0:y1, x0:x1].astype(np.float64)
        window = window - window.min()
        total = window.sum()
        if total <= 0.0:
            return float(x), float(y)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        return float((window * xs).sum() / total), float((window * ys).sum() / total)
    raise ValueError(f"unknown sub-pixel decode method {method!r}")


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def attribute_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: attribute_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [attribute_dict(item) for item in value]
    return value


def scale_court_gates(
    gates: dict[int, np.ndarray],
    native_size: res.FrameSize,
    artifact_size: res.FrameSize,
) -> dict[int, np.ndarray]:
    return {
        point: res.scale_points(quad, native_size, artifact_size).astype(np.float32)
        for point, quad in gates.items()
    }


def court_gates_in_artifact_space(
    out_dir: str,
    frames_dir: str,
    artifact_size: res.FrameSize,
) -> dict[int, np.ndarray]:
    """Per-point court polygons in the same space the exported candidates live in.

    :func:`cv.pipeline.ball.load_court_gate` returns polygons in NATIVE image space, but
    candidate coordinates are scaled to ``artifact_size`` before the point-in-polygon test.
    The gate has to be scaled exactly once, here, so both sides of the test share a
    coordinate space. Scaling it twice (which the inlined predecessor of this function did)
    contracts the polygon toward the origin by the square of the ratio and rejects most of
    the genuinely on-court candidates -- above all the far court, where coverage is already
    the binding constraint.
    """
    gates = load_court_gate(out_dir)
    if not gates:
        return {}
    probe_frames = sorted(glob.glob(os.path.join(out_dir, frames_dir, "pt*", "f_*.jpg")))
    if not probe_frames:
        return gates
    native_size = res.frame_size_for_path(probe_frames[0])
    return scale_court_gates(gates, native_size, artifact_size)


def scale_projection(
    P: np.ndarray,
    native_size: res.FrameSize,
    artifact_size: res.FrameSize,
) -> np.ndarray:
    """Rescale a world->native-pixel camera projection to world->artifact-pixel."""
    scale = np.diag(
        [
            artifact_size.width / native_size.width,
            artifact_size.height / native_size.height,
            1.0,
        ]
    )
    return scale @ P


def affine_for_frame(width: int, height: int, inverse: bool = False) -> np.ndarray:
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(width, height))
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, -scale / 2.0], dtype=np.float32)
    src[2] = src[1] + np.array([scale / 2.0, 0.0], dtype=np.float32)
    dst[0] = np.array([INPUT_WH[0] / 2.0, INPUT_WH[1] / 2.0], dtype=np.float32)
    dst[1] = dst[0] + np.array([0.0, -INPUT_WH[0] / 2.0], dtype=np.float32)
    dst[2] = dst[1] + np.array([INPUT_WH[0] / 2.0, 0.0], dtype=np.float32)
    return cv2.getAffineTransform(dst, src) if inverse else cv2.getAffineTransform(src, dst)


def preprocess(path: str) -> tuple[np.ndarray, np.ndarray, res.FrameSize]:
    bgr = cv2.imread(path)
    if bgr is None:
        raise OSError(f"cannot read {path}")
    inverse = affine_for_frame(bgr.shape[1], bgr.shape[0], inverse=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    warped = cv2.warpAffine(rgb, affine_for_frame(bgr.shape[1], bgr.shape[0]), INPUT_WH)
    normalized = (warped.astype(np.float32) / 255.0 - MEAN) / STD
    return normalized.transpose(2, 0, 1), inverse, res.FrameSize(bgr.shape[1], bgr.shape[0])


def temporal_contexts(target: int, frame_count: int) -> list[tuple[list[int], int]]:
    """Place one target frame in past-only, centered, and future-only triplets."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    contexts = []
    for target_channel in range(3):
        start = target - target_channel
        indices = [min(max(start + offset, 0), frame_count - 1) for offset in range(3)]
        contexts.append((indices, target_channel))
    return contexts


def build_model(external_root: str, model_name: str, weights: str, device: int):
    import torch
    import yaml

    src = os.path.join(external_root, "src")
    sys.path.insert(0, src)
    from models import build_model as wasb_build_model

    config_name = "wasb.yaml" if model_name == "wasb" else "tracknetv2.yaml"
    with open(os.path.join(src, "configs", "model", config_name)) as f:
        model_config = yaml.safe_load(f)
    model = wasb_build_model({"model": attribute_dict(model_config)})
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval().to(torch.device(f"cuda:{device}"))


def infer_clip(
    model,
    frame_paths: list[str],
    batch_size: int,
    device: int,
    artifact_size: res.FrameSize = res.CANONICAL_SIZE,
    k_best: int = 1,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = False,
    subpixel: str = "argmax",
) -> list[dict]:
    import torch

    prepared = [preprocess(path) for path in frame_paths]
    rows = []

    def append_prediction(index: int, heatmap: np.ndarray) -> None:
        frame_name = os.path.basename(frame_paths[index])
        frame_id = frame_number_from_name(frame_name)
        if k_best <= 1:
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peaks = [(int(x), int(y), float(heatmap[y, x]))]
        else:
            peaks = topk_peaks(heatmap, k_best, peak_threshold, nms_radius)
        for rank, (x, y, score) in enumerate(peaks):
            rx, ry = subpixel_refine(heatmap, int(x), int(y), subpixel)
            original = cv2.transform(np.float32([[[rx, ry]]]), prepared[index][1])[0, 0]
            artifact = res.scale_points(original, prepared[index][2], artifact_size)
            row = {
                "frame": frame_name,
                "frame_id": frame_id,
                "x": float(artifact[0]),
                "y": float(artifact[1]),
                "score": float(score),
            }
            if k_best > 1:
                row["rank"] = rank
            rows.append(row)

    if temporal_ensemble:
        targets_per_batch = max(1, batch_size // 3)
        for target_offset in range(0, len(frame_paths), targets_per_batch):
            targets = list(
                range(target_offset, min(target_offset + targets_per_batch, len(frame_paths)))
            )
            samples = []
            for target in targets:
                for indices, target_channel in temporal_contexts(target, len(frame_paths)):
                    tensor = np.concatenate([prepared[index][0] for index in indices], axis=0)
                    samples.append((target, target_channel, tensor))
            inputs = torch.from_numpy(np.stack([item[2] for item in samples])).to(f"cuda:{device}")
            with torch.inference_mode():
                heatmaps = model(inputs)[0].sigmoid().cpu().numpy()
            grouped: dict[int, list[np.ndarray]] = {target: [] for target in targets}
            for (target, target_channel, _), sample in zip(samples, heatmaps, strict=True):
                grouped[target].append(sample[target_channel])
            for target in targets:
                append_prediction(target, np.mean(grouped[target], axis=0))
        return rows

    triplets = []
    for start in range(0, len(frame_paths), 3):
        indices = list(range(start, min(start + 3, len(frame_paths))))
        indices += [indices[-1]] * (3 - len(indices))
        tensor = np.concatenate([prepared[index][0] for index in indices], axis=0)
        triplets.append((indices, tensor))

    for offset in range(0, len(triplets), batch_size):
        chunk = triplets[offset : offset + batch_size]
        inputs = torch.from_numpy(np.stack([item[1] for item in chunk])).to(f"cuda:{device}")
        with torch.inference_mode():
            heatmaps = model(inputs)[0].sigmoid().cpu().numpy()
        for (indices, _), sample in zip(chunk, heatmaps):
            for channel, index in enumerate(indices):
                if index >= len(frame_paths) or (channel > 0 and index == indices[channel - 1]):
                    continue
                append_prediction(index, sample[channel])
    return rows


def infer_clip_far_native(
    model,
    frame_paths: list[str],
    native_root: str,
    P: np.ndarray,
    img_w: int,
    img_h: int,
    batch_size: int,
    k_best: int,
    peak_threshold: float,
    nms_radius: int,
    subpixel: str = "argmax",
) -> list[dict]:
    """k-best WASB peaks over the NATIVE-resolution far half-court crop of each frame.

    Reuses :mod:`ball_far2x` cropping geometry (far-court volume -> image bbox -> two
    overlapping tiles) but cuts the tiles from the paired 1920x1080 frame (real sensor
    pixels, not a Lanczos upscale — the prior far2x measured 2.2x far coverage from fake-2x;
    real pixels beat that) and extracts the top-k peaks per tile instead of a single argmax.
    Peaks are mapped back to the 960x540 coordinate convention and merged across the two
    tiles by native-pixel NMS. Output rows carry a ``rank`` column, same as the k-best
    full-frame path.

    ``P``, ``img_w`` and ``img_h`` must already be in the ARTIFACT (960x540) space the
    candidate CSV is written in: :mod:`ball_far2x` cuts the paired native frame at exactly
    ``UPSCALE`` times the tile it computed and maps detections back by dividing by the same
    factor. Handing it a native-space box makes it read ``UPSCALE`` times out of bounds and
    emit coordinates in the wrong space, so the caller rescales the camera projection with
    :func:`scale_projection` first.
    """
    import torch

    from cv.pipeline.ball_far2x import UPSCALE, far_crop_box, preprocess_frame, tiles_for_box

    box = far_crop_box(P, img_w, img_h)
    tiles = tiles_for_box(box)
    prepared = [
        preprocess_frame(path, tiles, os.path.join(native_root, os.path.basename(path)))
        for path in frame_paths
    ]
    device = next(model.parameters()).device
    per_frame: dict[int, list[tuple[float, float, float]]] = {
        i: [] for i in range(len(frame_paths))
    }
    for t_index, tile in enumerate(tiles):
        x0, y0 = tile[0], tile[1]
        up_w = int((tile[2] - tile[0]) * UPSCALE)
        up_h = int((tile[3] - tile[1]) * UPSCALE)
        inverse = affine_for_frame(up_w, up_h, inverse=True)
        stream = [prepared[i][t_index] for i in range(len(frame_paths))]
        triplets = []
        for start in range(0, len(stream), 3):
            idx = list(range(start, min(start + 3, len(stream))))
            idx += [idx[-1]] * (3 - len(idx))
            triplets.append((idx, np.concatenate([stream[i] for i in idx], axis=0)))
        for offset in range(0, len(triplets), batch_size):
            chunk = triplets[offset : offset + batch_size]
            inputs = torch.from_numpy(np.stack([t for _, t in chunk])).to(device)
            with torch.inference_mode():
                heatmaps = model(inputs)[0].sigmoid().cpu().numpy()
            for (indices, _), sample in zip(chunk, heatmaps):
                for channel, index in enumerate(indices):
                    if index >= len(stream) or (channel > 0 and index == indices[channel - 1]):
                        continue
                    for hx, hy, score in topk_peaks(
                        sample[channel], k_best, peak_threshold, nms_radius
                    ):
                        rx, ry = subpixel_refine(sample[channel], int(hx), int(hy), subpixel)
                        up_xy = cv2.transform(np.float32([[[rx, ry]]]), inverse)[0, 0]
                        per_frame[index].append(
                            (
                                float(up_xy[0] / UPSCALE + x0),
                                float(up_xy[1] / UPSCALE + y0),
                                float(score),
                            )
                        )
    rows = []
    r2 = nms_radius * nms_radius
    for index, path in enumerate(frame_paths):
        frame_name = os.path.basename(path)
        frame_id = frame_number_from_name(frame_name)
        merged: list[tuple[float, float, float]] = []
        for x, y, score in sorted(per_frame[index], key=lambda d: -d[2]):
            if all((x - mx) ** 2 + (y - my) ** 2 > r2 for mx, my, _ in merged):
                merged.append((x, y, score))
                if len(merged) >= k_best:
                    break
        for rank, (x, y, score) in enumerate(merged):
            rows.append(
                {
                    "frame": frame_name,
                    "frame_id": frame_id,
                    "x": x,
                    "y": y,
                    "score": score,
                    "rank": rank,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir", default="rally_frames_50_benchmark")
    parser.add_argument("--model", choices=["wasb", "tracknetv2"], required=True)
    parser.add_argument("--external-root", default=os.fspath(tracker_root()))
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-tag", default="", help="suffix for versioned candidate outputs")
    parser.add_argument("--artifact-width", type=int, default=960)
    parser.add_argument("--artifact-height", type=int, default=540)
    parser.add_argument(
        "--k-best",
        type=int,
        default=1,
        help="export up to K heatmap peaks per frame (default 1 = the "
        "historical single-argmax export; K>1 adds a 'rank' column and "
        "is the multi-hypothesis / track-before-detect substrate)",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.05,
        help="LOW heatmap floor for secondary k-best peaks (only used when "
        "--k-best>1; the argmax is always kept regardless)",
    )
    parser.add_argument(
        "--nms-radius", type=int, default=4, help="min separation (heatmap px) between k-best peaks"
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="average past-only, centered, and future-only three-frame predictions",
    )
    parser.add_argument(
        "--subpixel",
        choices=["argmax", "parabolic", "centroid"],
        default="centroid",
        help="heatmap decode: 'centroid' (default) refines the returned peak to sub-cell "
        "position with a floor-subtracted 5x5 centroid, worth about -19%% median 2D error on "
        "owner frames at zero inference cost (EXPERIMENTS.md Investigation 1); 'argmax' is the "
        "pre-2026-09 bare-integer-peak default, kept to reproduce frozen artifacts.",
    )
    parser.add_argument(
        "--far-native",
        action="store_true",
        help="run k-best over the NATIVE-resolution far half-court crop "
        "(reuses ball_far2x geometry; needs --native-frames-dir and a "
        "per-point camera P). Far coverage is the binding constraint.",
    )
    parser.add_argument(
        "--native-frames-dir",
        default="rally_frames_50_1080",
        help="paired 1920x1080 frame dir (1:1 filenames) for --far-native",
    )
    parser.add_argument(
        "--camera-npz",
        default="camera_P_per_point.npz",
        help="per-point camera projections for the far-crop box (--far-native)",
    )
    args = parser.parse_args()
    if not args.weights:
        args.weights = os.path.join(
            args.external_root,
            "pretrained_weights",
            f"{args.model}_tennis_best.pth.tar",
        )
    stage_run = StageRun(args.out, f"ball_tracking_{args.model}", args, gpu_devices=[args.device])
    model = build_model(args.external_root, args.model, args.weights, args.device)
    artifact_size = res.FrameSize(args.artifact_width, args.artifact_height)
    gates = court_gates_in_artifact_space(args.out, args.frames_dir, artifact_size)

    k_best = max(1, args.k_best)
    ranked = k_best > 1
    cams = None
    if args.far_native:
        loaded = np.load(os.path.join(args.out, args.camera_npz))
        cams = {int(p): P for p, P in zip(loaded["pts"], loaded["P"])}

    candidate_rows, track_rows = [], []
    input_frames = processed_frames = candidate_frames = processed_clips = skipped_camera_clips = 0
    clip_dirs = sorted(glob.glob(os.path.join(args.out, args.frames_dir, "pt*")))
    for clip_index, clip_dir in enumerate(clip_dirs):
        clip = os.path.basename(clip_dir)
        frame_paths = sorted(
            glob.glob(os.path.join(clip_dir, "f_*.jpg")), key=frame_number_from_name
        )
        input_frames += len(frame_paths)
        if args.far_native:
            P = cams.get(int(clip[2:]))
            if P is None:
                skipped_camera_clips += 1
                print(f"{args.model} far-native: {clip} has no camera P; skipped", flush=True)
                continue
            probe = cv2.imread(frame_paths[0])
            native_size = res.FrameSize(probe.shape[1], probe.shape[0])
            from cv.pipeline.ball_far2x import UPSCALE

            if native_size.width != round(
                artifact_size.width * UPSCALE
            ) or native_size.height != round(artifact_size.height * UPSCALE):
                raise ValueError(
                    f"--far-native needs native frames at exactly {UPSCALE}x the artifact "
                    f"space: {native_size.label} vs {artifact_size.label}"
                )
            predictions = infer_clip_far_native(
                model,
                frame_paths,
                os.path.join(args.out, args.native_frames_dir, clip),
                scale_projection(P, native_size, artifact_size),
                artifact_size.width,
                artifact_size.height,
                args.batch,
                k_best,
                args.peak_threshold,
                args.nms_radius,
                args.subpixel,
            )
        else:
            predictions = infer_clip(
                model,
                frame_paths,
                args.batch,
                args.device,
                artifact_size,
                k_best,
                args.peak_threshold,
                args.nms_radius,
                args.temporal_ensemble,
                args.subpixel,
            )
        processed_frames += len(frame_paths)
        processed_clips += 1
        candidate_frames += len({row["frame_id"] for row in predictions})
        gate = gates.get(int(clip[2:]))
        by_frame: dict[int, list[tuple[float, float, float]]] = {}
        for row in predictions:
            on_court = gate is None or cv2.pointPolygonTest(gate, (row["x"], row["y"]), False) >= 0
            cand = [clip, row["frame"], row["x"], row["y"], row["score"], on_court]
            if ranked:
                cand.append(row.get("rank", 0))
            candidate_rows.append(tuple(cand))
            if on_court and row["score"] >= args.score_threshold:
                by_frame.setdefault(row["frame_id"], []).append((row["x"], row["y"], 1))
        linked_input = [
            (fid, by_frame.get(fid, [])) for fid in sorted({row["frame_id"] for row in predictions})
        ]
        for track_id, track in enumerate(link(linked_input, args.fps)):
            for frame_id, x, y in track:
                track_rows.append((clip, f"f_{frame_id:04d}.jpg", x, y, track_id))
        print(
            f"{args.model}: {clip_index + 1}/{len(clip_dirs)} clips, "
            f"{processed_frames} processed frames, {len(candidate_rows)} candidate rows, "
            f"{len(track_rows)} linked points",
            flush=True,
        )

    output_name = args.model + (f"_{args.output_tag}" if args.output_tag else "")
    candidates_path = os.path.join(args.out, f"ball_candidates_{output_name}.csv")
    header = ["clip", "frame", "x", "y", "score", "on_court"]
    if ranked:
        header.append("rank")
    tracks_path = os.path.join(args.out, f"ball_track_{output_name}.csv")
    # The sidecar's image size is the size of the frames the detector read, whether or not
    # any candidate survived (a far-native pass that skips every clip must not declare the
    # artifact space as its image size: the consensus stage compares sources on it).
    frame_files = sorted(glob.glob(os.path.join(args.out, args.frames_dir, "pt*", "f_*.jpg")))
    image_size = res.frame_size_for_path(frame_files[0]) if frame_files else artifact_size
    write_candidate_artifact(
        Path(candidates_path),
        header,
        [dict(zip(header, cand)) for cand in candidate_rows],
        image_size=image_size,
        legacy_size=artifact_size,
        source=os.path.join(args.out, args.frames_dir),
    )
    write_track_artifact(
        Path(tracks_path),
        [
            {"clip": clip, "frame": frame, "x": x, "y": y, "track_id": track_id}
            for clip, frame, x, y, track_id in track_rows
        ],
        [Path(candidates_path)],
    )
    stage_run.finish(
        outputs={
            "clips": len(clip_dirs),
            "count_contract": "tracker_frame_counts_v2",
            "processed_clips": processed_clips,
            "skipped_camera_clips": skipped_camera_clips,
            "input_frames": input_frames,
            "processed_frames": processed_frames,
            "candidate_frames": candidate_frames,
            "candidate_rows": len(candidate_rows),
            "linked_points": len(track_rows),
            "candidates_csv": candidates_path,
            "tracks_csv": tracks_path,
            "image_size": [image_size.width, image_size.height],
            "artifact_size": [artifact_size.width, artifact_size.height],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
