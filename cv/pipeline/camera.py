"""Play-camera detection via frame clustering — Phase A of the CV pipeline.

The behind-baseline "play camera" is the most common, near-static shot in any tennis
broadcast (the camera barely moves between points). So instead of hand-tuning a per-surface
court detector, we: extract small frames at low fps, cluster their downscaled signatures
with k-means, select court-like clusters automatically or in an explicit diagnostic mode,
median-smooth the per-frame labels, and emit contiguous play runs as candidate point segments.

Outputs (to --out):
- ``frames_small/``      320x180 JPEGs at --fps (kept for later stages)
- ``segments.csv``       t_start,t_end,n_frames per contiguous play-camera run
- ``debug_clusters/``    a few sample frames per cluster for eyeballing
- ``labels.npz``         per-frame cluster id + play flag + timestamps

    .venv/bin/python cv/pipeline/camera.py --video MATCH.mp4 \
        --out data/processed/rg2025f --fps 1
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FEAT_W, FEAT_H = 32, 18
ROOT = Path(__file__).resolve().parents[2]

# How many frames of a cluster carry its admission evidence.  Every witness in
# ``_select_play_clusters`` is a median or a mean over this sample, so the sample size is
# the resolution of the decision: at the original seven frames ``wide_view_fraction`` could
# only take the eight values j/7, and a 1,732-frame cluster was admitted or discarded on
# one YOLO prediction either side of the 0.40 threshold.  Measured over 83 broadcasts'
# already-clustered frames (cv/experiments/play_camera_gate/MEASUREMENT.md): resampling the
# seven-frame verdict disagrees with the 63-frame verdict on 2.5% of clusters and 24% of
# broadcasts, and four genuine broadcasts lost every play segment they had to it.  Raising the
# count costs only person-model inference over frames already extracted and decoded -- about
# twenty seconds a broadcast on a GPU, against a stage whose video decode dominates.
PLAY_CLUSTER_SAMPLE_FRAMES = 31


def extract_small(
    video: str,
    out_dir: str,
    fps: float,
    source_frames: str | None = None,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))
    sources = sorted(glob.glob(os.path.join(source_frames, "*.jpg"))) if source_frames else []
    if existing and (not sources or len(existing) == len(sources)):
        print(f"reusing {len(existing)} extracted frames in {out_dir}")
        return existing
    if sources:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        def resize(item: tuple[int, str]) -> str:
            index, source = item
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"unreadable source frame: {source}")
            output = os.path.join(out_dir, f"f_{index:06d}.jpg")
            image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(output, image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise OSError(f"could not write frame: {output}")
            return output

        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(resize, enumerate(sources, start=1)))
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        video,
        "-vf",
        f"fps={fps},scale=320:180",
        "-q:v",
        "4",
        os.path.join(out_dir, "f_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))


def features(paths: list[str]) -> np.ndarray:
    X = np.empty((len(paths), FEAT_W * FEAT_H * 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        img = cv2.resize(img, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
        X[i] = img.reshape(-1).astype(np.float32) / 255.0
    return X


def cluster(X: np.ndarray, k: int, seed: int = 0):
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=5)
    labels = km.fit_predict(X)
    return labels, km


def _courtness(path: str) -> int:
    """Broadcast-agnostic court-view score: number of long straight bright lines in the
    lower 3/4 of the frame (wide court views have many; closeups/crowd shots few).

    This is an un-normalised count of Hough segments on a fixed 320x180 frame, so it is not
    comparable across line detectors or frame sizes, and ``_select_play_clusters`` bands it
    from both sides rather than taking "more is better".  Measured over 83 broadcasts, accepted
    play clusters sit at 6-49 segments while the high counts belong to wide scenic venue shots
    (rain delays, establishing shots, full-stadium views) and to the graphics chrome of sources
    that are not match video; see cv/experiments/play_camera_gate/MEASUREMENT.md.
    """
    img = cv2.imread(path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    )
    mask = (th > 40).astype(np.uint8) * 255
    mask[: mask.shape[0] // 4] = 0
    segs = cv2.HoughLinesP(
        mask, 1, np.pi / 180, threshold=40, minLineLength=mask.shape[1] // 6, maxLineGap=12
    )
    return 0 if segs is None else len(segs)


def _color_coherence(path: str) -> float:
    """Fraction in the dominant coarse HSV bin; full-court views are surface-dominated."""
    image = cv2.imread(path)
    if image is None:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    quantized = hsv // np.array([15, 48, 48], dtype=np.uint8)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    return float(counts.max() / len(quantized))


def _select_play_clusters(
    scores: dict[int, tuple[float, float, float]],
    wide_view_fraction: dict[int, float] | None,
) -> set[int]:
    """Admit play clusters on wide-player, line and surface evidence.

    Each pass runs only when the one before it admits nothing, so a broadcast that already has
    a play cluster keeps exactly that one.  The band's upper line bound is measured, not
    decorative: across 83 broadcasts every accepted play cluster sat between 6 and 49 segments,
    and of the nine clusters the ceiling alone holds out, seven are a non-broadcast stream's
    chrome or scenic venue shots that carry court lines *and* distant people
    (cv/experiments/play_camera_gate/MEASUREMENT.md).
    """
    if wide_view_fraction is not None:
        selected = {
            cluster_id
            for cluster_id, (_, line_score, coherence) in scores.items()
            if wide_view_fraction.get(cluster_id, 0.0) >= 0.40
            and 8.0 <= line_score <= 50.0
            and coherence >= 0.15
        }
        if selected:
            return selected
        fallback = {
            cluster_id
            for cluster_id, (_, line_score, coherence) in scores.items()
            if wide_view_fraction.get(cluster_id, 0.0) >= 0.75
            and line_score >= 3.0
            and coherence >= 0.20
        }
        if fallback:
            top_coherence = max(scores[cluster_id][2] for cluster_id in fallback)
            return {
                cluster_id
                for cluster_id in fallback
                if scores[cluster_id][2] >= max(0.20, 0.80 * top_coherence)
            }
        # Both established passes have left this broadcast with no play cluster at all, which
        # costs every one of its points.  Repeat the strict band without its upper line bound
        # and at the fallback's raised surface bar.  This can only turn nothing into something:
        # a broadcast either pass admits never reaches here, so no existing selection moves.
        # The ceiling is otherwise load-bearing -- it refuses scenic venue shots and line-heavy
        # stream chrome that carry wide-player support -- and what still refuses those here is
        # coherence >= 0.20: measured over 83 broadcasts, lifting the ceiling at this point
        # recovers one genuine grass play camera (59 segments, 0.208 coherence) and admits
        # nothing else, while the non-broadcast sources stay refused.
        return {
            cluster_id
            for cluster_id, (_, line_score, coherence) in scores.items()
            if wide_view_fraction.get(cluster_id, 0.0) >= 0.40
            and line_score >= 8.0
            and coherence >= 0.20
        }
    top_combined = max(score[0] for score in scores.values())
    top_coherence = max(score[2] for score in scores.values())
    return {
        cluster_id
        for cluster_id, (combined, line_score, coherence) in scores.items()
        if combined >= 0.72 * top_combined
        and line_score >= 8.0
        and coherence >= max(0.15, 0.60 * top_coherence)
    }


def pick_play_clusters(
    labels: np.ndarray,
    k: int,
    frame_paths: list[str],
    *,
    reference_paths: list[str] | None = None,
    person_model: str | None = None,
    device: str = "cpu",
    sample_frames: int = PLAY_CLUSTER_SAMPLE_FRAMES,
) -> set[int]:
    """Select surface-dominated clusters with consistent court-line evidence.

    ``sample_frames`` is the per-cluster evidence sample; see
    :data:`PLAY_CLUSTER_SAMPLE_FRAMES` for why it is not seven.
    """
    rng = np.random.default_rng(0)
    scores: dict[int, tuple[float, float, float]] = {}
    sampled_indices: dict[int, np.ndarray] = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) < 10:
            continue
        samples = rng.choice(idx, size=min(sample_frames, len(idx)), replace=False)
        sampled_indices[c] = samples
        line_score = float(np.median([_courtness(frame_paths[j]) for j in samples]))
        coherence = float(np.median([_color_coherence(frame_paths[j]) for j in samples]))
        scores[c] = (line_score * coherence, line_score, coherence)
    if not scores:
        return set()
    wide_view_fraction = None
    if person_model:
        from ultralytics import YOLO

        references = reference_paths or frame_paths
        selected = [
            (cluster_id, index)
            for cluster_id, indices in sampled_indices.items()
            for index in indices
        ]
        predictions = YOLO(person_model).predict(
            [references[index] for _, index in selected],
            classes=[0],
            imgsz=960,
            batch=64,
            device=device,
            verbose=False,
        )
        witnesses: dict[int, list[bool]] = {cluster_id: [] for cluster_id in scores}
        for (cluster_id, _), prediction in zip(selected, predictions, strict=True):
            boxes = prediction.boxes.xyxy.cpu().numpy()
            heights = (
                (boxes[:, 3] - boxes[:, 1]) / prediction.orig_shape[0]
                if len(boxes)
                else np.empty(0)
            )
            witnesses[cluster_id].append(len(boxes) >= 2 and float(heights.max()) <= 0.38)
        wide_view_fraction = {
            cluster_id: float(np.mean(values)) for cluster_id, values in witnesses.items()
        }
        print(
            "play-cluster evidence:",
            {
                cluster_id: {
                    "line": round(scores[cluster_id][1], 2),
                    "surface": round(scores[cluster_id][2], 3),
                    "wide_player": round(wide_view_fraction[cluster_id], 3),
                    "frames": len(sampled_indices[cluster_id]),
                }
                for cluster_id in scores
            },
        )
    return _select_play_clusters(scores, wide_view_fraction)


def smooth(flags: np.ndarray, width: int = 5) -> np.ndarray:
    pad = width // 2
    padded = np.pad(flags.astype(np.int8), pad, mode="edge")
    return np.array([int(np.median(padded[i : i + width])) for i in range(len(flags))], dtype=bool)


def segments_from_flags(flags: np.ndarray, fps: float, min_len_s: float = 4.0):
    segs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if (i - start) / fps >= min_len_s:
                segs.append((start / fps, i / fps, i - start))
            start = None
    if start is not None and (len(flags) - start) / fps >= min_len_s:
        segs.append((start / fps, len(flags) / fps, len(flags) - start))
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--min-seg", type=float, default=4.0, help="min play-segment seconds")
    ap.add_argument(
        "--source-frames",
        type=Path,
        help="existing full-frame observations at --fps; avoids decoding the video twice",
    )
    ap.add_argument("--person-model", type=Path, default=ROOT / "yolov8m.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--play-clusters",
        help="comma-separated reviewed cluster ids for diagnostic/curated runs",
    )
    ap.add_argument(
        "--allow-reviewed-play-clusters",
        action="store_true",
        help="required acknowledgement that --play-clusters makes this run human-assisted",
    )
    args = ap.parse_args()
    if args.play_clusters and not args.allow_reviewed_play_clusters:
        raise ValueError(
            "--play-clusters is human-assisted; pass --allow-reviewed-play-clusters "
            "for an explicitly curated diagnostic run"
        )
    video = sorted(glob.glob(args.video))[0] if any(c in args.video for c in "*?[") else args.video
    from cv.pipeline.run_manifest import StageRun

    stage_run = StageRun(
        args.out,
        "camera_play_segments",
        args,
        mode="diagnostic" if args.play_clusters else "automatic",
        reviewed_inputs=(
            [{"type": "reviewed_play_clusters", "value": args.play_clusters}]
            if args.play_clusters
            else []
        ),
    )

    frames = extract_small(
        video,
        os.path.join(args.out, "frames_small"),
        args.fps,
        os.fspath(args.source_frames) if args.source_frames else None,
    )
    print(f"{len(frames)} frames @ {args.fps}fps")
    X = features(frames)
    labels, km = cluster(X, args.k)
    explicit_clusters = (
        {int(value) for value in args.play_clusters.split(",") if value.strip()}
        if args.play_clusters
        else None
    )
    references = (
        sorted(glob.glob(os.path.join(args.source_frames, "*.jpg")))
        if args.source_frames
        else frames
    )
    if len(references) != len(frames):
        raise ValueError(
            f"camera reference-frame mismatch: references={len(references)} frames={len(frames)}"
        )
    play_cs = explicit_clusters or pick_play_clusters(
        labels,
        args.k,
        frames,
        reference_paths=references,
        person_model=os.fspath(args.person_model),
        device=args.device,
    )
    if not play_cs <= set(range(args.k)):
        raise ValueError(f"invalid play clusters for k={args.k}: {sorted(play_cs)}")
    sizes = np.bincount(labels, minlength=args.k)
    print("cluster sizes:", dict(enumerate(sizes.tolist())), "| play clusters:", sorted(play_cs))

    flags = smooth(np.isin(labels, list(play_cs)))
    segs = segments_from_flags(flags, args.fps, args.min_seg)
    play_frac = flags.mean()
    print(f"play-camera fraction: {play_frac:.2%} | segments >= {args.min_seg}s: {len(segs)}")

    np.savez_compressed(
        os.path.join(args.out, "labels.npz"),
        labels=labels,
        play=flags,
        fps=args.fps,
        play_clusters=np.array(sorted(play_cs)),
    )
    with open(os.path.join(args.out, "segments.csv"), "w") as f:
        f.write("t_start,t_end,n_frames\n")
        for a, b, n in segs:
            f.write(f"{a:.1f},{b:.1f},{n}\n")

    dbg = os.path.join(args.out, "debug_clusters")
    shutil.rmtree(dbg, ignore_errors=True)
    os.makedirs(dbg)
    rng = np.random.default_rng(0)
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        for j in rng.choice(idx, size=min(3, len(idx)), replace=False):
            shutil.copy(frames[j], os.path.join(dbg, f"c{c:02d}_{os.path.basename(frames[j])}"))
    print(f"debug samples -> {dbg}")
    stage_run.finish(
        outputs={
            "frames": len(frames),
            "play_clusters": sorted(play_cs),
            "play_fraction": round(float(play_frac), 4),
            "segments": len(segs),
            "selection_source": "reviewed_diagnostic" if explicit_clusters else "automatic",
            "reviewed_play_clusters_loaded": explicit_clusters is not None,
            "play_cluster_sample_frames": PLAY_CLUSTER_SAMPLE_FRAMES,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
