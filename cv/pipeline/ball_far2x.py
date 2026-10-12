"""Far-court magnified WASB re-tracking (substrate enhancement, far2x_v1).

Motivation: the observability study (data/processed/bakeoff/observability_study/report.json)
measured far-court ball SNR (~2-4 px/frame motion vs ~1 px detection noise) as the binding
constraint, and 5 of the forensic Tier-1 misses were track-absent at the far court. This stage
re-runs the official WASB tennis tracker on magnified crops of the far half-court for every
aligned rally window, and emits a NEW versioned candidates/track pair in the standard schema
with coordinates mapped back to native 960x540 frame pixels. Existing artifacts are untouched.

Geometry: each point's camera P (camera_P_per_point.npz) projects the far half-court volume
(x in [-3, W+3], y in [net-0.5, baseline+7] m, z in [0, 4] m) to an image-space crop box
(median box fallback for points without P). Because the WASB harness letterboxes any input to
512x288, a single 2x-upscaled crop of the full ~686 px-wide far box would only reach ~1.4x
effective magnification; the box is therefore split into two overlapping 480 px-wide tiles,
each Lanczos-2x upscaled, giving exactly 2.0x effective magnification vs the full-frame run
(rule-17 HOW deviation, logged). Per-frame detections are merged across tiles by score.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python cv/pipeline/ball_far2x.py \
        --out data/processed/rg2025f --frames-dir rally_frames_50_contact_v2 \
        --output-tag far2x_v1 --batch 16
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball import link, load_court_gate
from ball_neural import INPUT_WH, MEAN, STD, affine_for_frame, build_model
from paths import tracker_root
from run_manifest import StageRun

COURT_W, COURT_L = 10.97, 23.77
NET_Y = COURT_L / 2
UPSCALE = 2.0
TILE_W = 480  # native px; 480 * 2 (Lanczos) -> 960 -> letterbox 512 => 2.0x full-frame scale


def project(P: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    homog = np.concatenate([pts3, np.ones((len(pts3), 1))], axis=1)
    uv = (P @ homog.T).T
    return uv[:, :2] / uv[:, 2:3]


def far_crop_box(P: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Image bbox of the far half-court volume (with out-margins and lob height)."""
    corners = np.array(
        [
            [x, y, z]
            for x in (-3.0, COURT_W / 2, COURT_W + 3.0)
            for y in (NET_Y - 0.5, COURT_L, COURT_L + 7.0)
            for z in (0.0, 4.0)
        ]
    )
    uv = project(P, corners)
    x0 = int(max(0, math.floor(uv[:, 0].min())))
    y0 = int(max(0, math.floor(uv[:, 1].min())))
    x1 = int(min(img_w, math.ceil(uv[:, 0].max())))
    y1 = int(min(img_h, math.ceil(uv[:, 1].max())))
    return x0, y0, x1, y1


def tiles_for_box(box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Two overlapping TILE_W-wide tiles spanning the crop box (one if it fits)."""
    x0, y0, x1, y1 = box
    height = y1 - y0
    # ensure the letterboxed (2x-upscaled) tile keeps its full height inside 512x288
    tile_w = max(TILE_W, math.ceil(height * INPUT_WH[0] / INPUT_WH[1]))
    tile_w = min(tile_w, x1 - x0)
    starts = [x0] if x1 - x0 <= tile_w else [x0, x1 - tile_w]
    return [(sx, y0, sx + tile_w, y1) for sx in dict.fromkeys(starts)]


def preprocess_frame(
    path: str, tiles: list[tuple[int, int, int, int]], native_path: str | None = None
) -> list[np.ndarray]:
    """Tile tensors for one frame.

    Default mode: crop the 540p frame at ``tiles`` (540p px) and Lanczos-2x upscale.
    Native mode (``native_path`` set): crop the paired 1920x1080 frame at ``2*tiles`` —
    the crop then has exactly the same pixel dimensions as the upscaled 540p crop, but
    made of real sensor pixels instead of interpolation. Downstream mapping (inverse
    affine, /UPSCALE, +tile origin) is identical, so outputs stay in the 960x540
    coordinate convention either way.
    """
    bgr = cv2.imread(native_path or path)
    if bgr is None:
        raise OSError(f"cannot read {native_path or path}")
    out = []
    for x0, y0, x1, y1 in tiles:
        if native_path:
            s = int(UPSCALE)
            up = cv2.cvtColor(bgr[s * y0 : s * y1, s * x0 : s * x1], cv2.COLOR_BGR2RGB)
        else:
            crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            up = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LANCZOS4)
        warped = cv2.warpAffine(up, affine_for_frame(up.shape[1], up.shape[0]), INPUT_WH)
        out.append(((warped.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
    return out


def infer_tile_stream(model, tensors: list[np.ndarray], inverse: np.ndarray, batch: int, tile):
    """Run WASB triplets over one tile's frame stream; return per-frame (x, y, score)
    in NATIVE frame pixels."""
    import torch

    x0, y0 = tile[0], tile[1]
    triplets = []
    for start in range(0, len(tensors), 3):
        indices = list(range(start, min(start + 3, len(tensors))))
        indices += [indices[-1]] * (3 - len(indices))
        triplets.append((indices, np.concatenate([tensors[i] for i in indices], axis=0)))

    results: dict[int, tuple[float, float, float]] = {}
    for offset in range(0, len(triplets), batch):
        chunk = triplets[offset : offset + batch]
        inputs = torch.from_numpy(np.stack([t for _, t in chunk])).to(model_device(model))
        with torch.inference_mode():
            heatmaps = model(inputs)[0].sigmoid().cpu().numpy()
        for (indices, _), sample in zip(chunk, heatmaps):
            for channel, index in enumerate(indices):
                if index >= len(tensors) or (channel > 0 and index == indices[channel - 1]):
                    continue
                heatmap = sample[channel]
                hy, hx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                up_xy = cv2.transform(np.float32([[[hx, hy]]]), inverse)[0, 0]
                results[index] = (
                    float(up_xy[0] / UPSCALE + x0),
                    float(up_xy[1] / UPSCALE + y0),
                    float(heatmap[hy, hx]),
                )
    return results


def model_device(model):
    return next(model.parameters()).device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir", default="rally_frames_50_contact_v2")
    parser.add_argument(
        "--native-frames-dir",
        default="",
        help="paired 1920x1080 frame dir (filenames 1:1 with --frames-dir); crops are cut "
        "from these REAL pixels instead of Lanczos-upscaling the 540p frames. Outputs stay "
        "in the 960x540 coordinate convention.",
    )
    parser.add_argument("--external-root", default=os.fspath(tracker_root()))
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", type=int, default=0, help="CUDA index inside the visible set")
    parser.add_argument("--physical-gpu", type=int, default=2, help="telemetry: nvidia-smi index")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--output-tag", default="far2x_v1")
    parser.add_argument("--io-threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit-clips", type=int, default=0, help="debug: only first N clips")
    args = parser.parse_args()
    if not args.weights:
        args.weights = os.path.join(
            args.external_root, "pretrained_weights", "wasb_tennis_best.pth.tar"
        )
    stage_run = StageRun(
        args.out, f"ball_tracking_wasb_{args.output_tag}", args, gpu_devices=[args.physical_gpu]
    )
    model = build_model(args.external_root, "wasb", args.weights, args.device)
    gates = load_court_gate(args.out)

    cam = np.load(os.path.join(args.out, "camera_P_per_point.npz"))
    Ps = {int(p): P for p, P in zip(cam["pts"], cam["P"])}

    clip_dirs = sorted(glob.glob(os.path.join(args.out, args.frames_dir, "pt*")))
    if args.limit_clips:
        clip_dirs = clip_dirs[: args.limit_clips]
    probe = cv2.imread(sorted(glob.glob(os.path.join(clip_dirs[0], "f_*.jpg")))[0])
    img_h, img_w = probe.shape[:2]
    boxes = {p: far_crop_box(P, img_w, img_h) for p, P in Ps.items()}
    median_box = tuple(int(v) for v in np.median(np.array(list(boxes.values())), axis=0))
    fallback_clips = 0

    candidate_rows, track_rows = [], []
    pool = ThreadPoolExecutor(args.io_threads)
    for clip_index, clip_dir in enumerate(clip_dirs):
        clip = os.path.basename(clip_dir)
        pid = int(clip[2:])
        box = boxes.get(pid)
        if box is None:
            box, fallback_clips = median_box, fallback_clips + 1
        tiles = tiles_for_box(box)
        frame_paths = sorted(glob.glob(os.path.join(clip_dir, "f_*.jpg")))
        if args.native_frames_dir:
            native_root = os.path.join(args.out, args.native_frames_dir, clip)
            prepared = list(
                pool.map(
                    lambda p: preprocess_frame(
                        p, tiles, os.path.join(native_root, os.path.basename(p))
                    ),
                    frame_paths,
                )
            )
        else:
            prepared = list(pool.map(lambda p: preprocess_frame(p, tiles), frame_paths))

        merged: dict[int, tuple[float, float, float]] = {}
        for t_index, tile in enumerate(tiles):
            up_w = int((tile[2] - tile[0]) * UPSCALE)
            up_h = int((tile[3] - tile[1]) * UPSCALE)
            inverse = affine_for_frame(up_w, up_h, inverse=True)
            stream = [prepared[i][t_index] for i in range(len(frame_paths))]
            for index, det in infer_tile_stream(model, stream, inverse, args.batch, tile).items():
                if index not in merged or det[2] > merged[index][2]:
                    merged[index] = det
        prepared = None

        gate = gates.get(pid)
        linked_input = []
        for index, path in enumerate(frame_paths):
            frame_name = os.path.basename(path)
            x, y, score = merged[index]
            on_court = gate is None or cv2.pointPolygonTest(gate, (x, y), False) >= 0
            candidate_rows.append((clip, frame_name, x, y, score, on_court))
            detections = [(x, y, 1)] if on_court and score >= args.score_threshold else []
            linked_input.append((int(frame_name[2:6]), detections))
        for track_id, track in enumerate(link(linked_input, args.fps)):
            for frame_id, tx, ty in track:
                track_rows.append((clip, f"f_{frame_id:04d}.jpg", tx, ty, track_id))
        print(
            f"far2x wasb: {clip_index + 1}/{len(clip_dirs)} clips ({clip}, {len(tiles)} tiles), "
            f"{len(candidate_rows)} frames, {len(track_rows)} linked points",
            flush=True,
        )
    pool.shutdown()

    candidates_path = os.path.join(args.out, f"ball_candidates_wasb_{args.output_tag}.csv")
    with open(candidates_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame", "x", "y", "score", "on_court"])
        writer.writerows(candidate_rows)
    tracks_path = os.path.join(args.out, f"ball_track_wasb_{args.output_tag}.csv")
    with open(tracks_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame", "x", "y", "track_id"])
        writer.writerows(track_rows)
    stage_run.finish(
        outputs={
            "clips": len(clip_dirs),
            "clips_median_box_fallback": fallback_clips,
            "candidate_frames": len(candidate_rows),
            "linked_points": len(track_rows),
            "candidates_csv": candidates_path,
            "tracks_csv": tracks_path,
            "median_far_box": list(median_box),
            "effective_magnification_vs_fullframe": (
                "2.0x (two 480px tiles, native 1080p pixels; output coords are 960x540-based)"
                if args.native_frames_dir
                else "2.0x (two 480px tiles, Lanczos 2x)"
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
