"""Native-cadence shot-boundary detection for broadcast video.

Stage 1 needs cuts at the source frame rate: a fault and its second serve are usually one
continuous play-camera shot, while replays, close-ups and graphics arrive as separate shots.
The 1 fps clustering in ``camera.py`` cannot see either boundary.

No learned shot-boundary model (TransNetV2 or equivalent) is available offline in this
environment, so this is the sanctioned fallback: a robust grid-histogram cut detector run on
every decoded frame.  One ``ffmpeg`` process streams the whole source as small grayscale
frames; per frame we keep a 3x3 grid of 16-bin intensity histograms and the mean absolute
difference against the previous frame.  A cut is a sharp local maximum of the histogram
distance relative to its own neighbourhood, which tolerates the slow drift of a panning
play camera and the global brightness swings of an outdoor broadcast.

The per-frame motion energy is retained alongside the shots: slow-motion replay is separable
from live play by inter-frame displacement, and broadcast slow motion also repeats frames,
so both the motion median and the near-duplicate fraction are reported per shot.

    .venv/bin/python -m cv.pipeline.shot_boundaries --video match.mp4 --out out_dir
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIGNAL_WIDTH = 96
SIGNAL_HEIGHT = 54
GRID = 3
BINS = 16
CHUNK_FRAMES = 4096
DUPLICATE_MOTION = 0.06  # mean |delta| below this is a repeated broadcast frame


@dataclass(frozen=True)
class Signals:
    """Per-frame native-cadence signals for one decoded span."""

    fps: float
    start_seconds: float
    distance: np.ndarray
    motion: np.ndarray
    motion_top: np.ndarray
    motion_bottom: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return self.start_seconds + np.arange(len(self.distance), dtype=np.float64) / self.fps


def _grid_histograms(frames: np.ndarray) -> np.ndarray:
    """(n, H, W) uint8 -> (n, GRID*GRID*BINS) float32 L1-normalised per cell."""
    count, height, width = frames.shape
    rows = np.minimum((np.arange(height) * GRID) // height, GRID - 1)
    columns = np.minimum((np.arange(width) * GRID) // width, GRID - 1)
    cell = (rows[:, None] * GRID + columns[None, :]).astype(np.int64)
    quantised = (frames >> (8 - int(np.log2(BINS)))).astype(np.int64)
    index = (
        np.arange(count, dtype=np.int64)[:, None, None] * (GRID * GRID * BINS)
        + cell[None, :, :] * BINS
        + quantised
    )
    counts = np.bincount(index.ravel(), minlength=count * GRID * GRID * BINS)
    counts = counts.reshape(count, GRID * GRID, BINS).astype(np.float32)
    return (counts / np.maximum(counts.sum(axis=2, keepdims=True), 1.0)).reshape(count, -1)


def stream_signals(
    video: Path | str,
    fps: float,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> Signals:
    """Decode the span at native cadence and return the per-frame cut/motion signals."""
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_seconds:
        command += ["-ss", f"{start_seconds:.6f}"]
    command += ["-i", str(video)]
    if duration_seconds is not None:
        command += ["-t", f"{duration_seconds:.6f}"]
    command += [
        "-map",
        "0:v:0",
        "-vf",
        f"scale={SIGNAL_WIDTH}:{SIGNAL_HEIGHT},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    frame_bytes = SIGNAL_WIDTH * SIGNAL_HEIGHT
    half = SIGNAL_HEIGHT // 2
    distances: list[np.ndarray] = []
    motions: list[np.ndarray] = []
    tops: list[np.ndarray] = []
    bottoms: list[np.ndarray] = []
    previous_histogram: np.ndarray | None = None
    previous_frame: np.ndarray | None = None
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            payload = process.stdout.read(frame_bytes * CHUNK_FRAMES)
            if not payload:
                break
            usable = len(payload) - len(payload) % frame_bytes
            frames = np.frombuffer(payload[:usable], dtype=np.uint8).reshape(
                -1, SIGNAL_HEIGHT, SIGNAL_WIDTH
            )
            histograms = _grid_histograms(frames)
            if previous_histogram is not None:
                histograms = np.vstack([previous_histogram[None, :], histograms])
                frames = np.vstack([previous_frame[None, :, :], frames])
                offset = 0
            else:
                offset = 1
            difference = np.abs(np.diff(histograms, axis=0)).sum(axis=1) / (2.0 * GRID * GRID)
            deltas = np.abs(np.diff(frames.astype(np.int16), axis=0)).astype(np.float32)
            if offset:
                distances.append(np.zeros(1, dtype=np.float32))
                motions.append(np.zeros(1, dtype=np.float32))
                tops.append(np.zeros(1, dtype=np.float32))
                bottoms.append(np.zeros(1, dtype=np.float32))
            distances.append(difference.astype(np.float32))
            motions.append(deltas.mean(axis=(1, 2)))
            tops.append(deltas[:, :half, :].mean(axis=(1, 2)))
            bottoms.append(deltas[:, half:, :].mean(axis=(1, 2)))
            previous_histogram = histograms[-1]
            previous_frame = frames[-1]
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {stderr.decode()[:2000]}")
    if not distances:
        raise RuntimeError(f"no frames decoded from {video}")
    return Signals(
        fps=fps,
        start_seconds=start_seconds,
        distance=np.concatenate(distances),
        motion=np.concatenate(motions),
        motion_top=np.concatenate(tops),
        motion_bottom=np.concatenate(bottoms),
    )


def _rolling_median(values: np.ndarray, half_width: int) -> np.ndarray:
    """Median over a centred window, computed on a strided view (window is small)."""
    padded = np.pad(values, half_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_width + 1)
    return np.median(windows, axis=1)


def detect_cuts(
    signals: Signals,
    *,
    absolute_threshold: float = 0.22,
    relative_factor: float = 5.0,
    neighbour_ratio: float = 1.8,
    context_seconds: float = 2.0,
    blend_seconds: float = 0.12,
    guard_seconds: float = 0.35,
    minimum_shot_seconds: float = 0.4,
) -> np.ndarray:
    """Frame indices whose incoming boundary is a hard cut.

    A cut can smear across two or three decoded frames when the source was rate-converted or
    the encoder blended the transition, so candidates are grouped first and the group peak is
    compared against the rest of its neighbourhood rather than against its own smear.
    """
    distance = signals.distance
    if len(distance) < 5:
        return np.empty(0, dtype=np.int64)
    half_width = max(2, int(round(context_seconds * signals.fps)))
    background = _rolling_median(distance, half_width)
    candidate = np.where(
        (distance >= absolute_threshold)
        & (distance >= relative_factor * np.maximum(background, 1e-4))
    )[0]
    candidate = candidate[candidate > 0]
    if not len(candidate):
        return np.empty(0, dtype=np.int64)
    blend = max(1, int(round(blend_seconds * signals.fps)))
    groups: list[list[int]] = [[int(candidate[0])]]
    for index in candidate[1:]:
        if index - groups[-1][-1] <= blend:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    guard = max(blend + 1, int(round(guard_seconds * signals.fps)))
    accepted: list[int] = []
    for group in groups:
        peak = int(group[int(np.argmax(distance[group]))])
        low = max(0, group[0] - guard)
        high = min(len(distance), group[-1] + guard + 1)
        outside = np.concatenate([distance[low : group[0]], distance[group[-1] + 1 : high]])
        if len(outside) and distance[peak] < neighbour_ratio * outside.max():
            continue
        accepted.append(peak)
    if not accepted:
        return np.empty(0, dtype=np.int64)
    minimum_gap = max(1, int(round(minimum_shot_seconds * signals.fps)))
    kept = [accepted[0]]
    for index in accepted[1:]:
        if index - kept[-1] < minimum_gap:
            if distance[index] > distance[kept[-1]]:
                kept[-1] = index
            continue
        kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def shots_from_cuts(signals: Signals, cuts: np.ndarray) -> list[dict]:
    """Contiguous shots with their motion statistics, in absolute source seconds."""
    edges = [0, *cuts.tolist(), len(signals.distance)]
    shots = []
    for shot_index, (left, right) in enumerate(zip(edges, edges[1:])):
        if right <= left:
            continue
        motion = signals.motion[left:right]
        interior = motion[1:] if len(motion) > 1 else motion
        shots.append(
            {
                "shot_index": shot_index,
                "t_start": signals.start_seconds + left / signals.fps,
                "t_end": signals.start_seconds + right / signals.fps,
                "frames": int(right - left),
                "motion_median": float(np.median(interior)) if len(interior) else 0.0,
                "motion_p90": float(np.percentile(interior, 90)) if len(interior) else 0.0,
                "motion_top_median": float(np.median(signals.motion_top[left:right])),
                "motion_bottom_median": float(np.median(signals.motion_bottom[left:right])),
                "duplicate_fraction": (
                    float(np.mean(interior < DUPLICATE_MOTION)) if len(interior) else 0.0
                ),
                "cut_distance_in": float(signals.distance[left]) if left else 0.0,
            }
        )
    return shots


def write_shots(out_dir: Path, shots: list[dict], signals: Signals, meta: dict) -> Path:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shot_boundaries_v1.csv"
    fields = list(shots[0]) if shots else ["shot_index", "t_start", "t_end", "frames"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shots)
    np.savez_compressed(
        out_dir / "shot_signals_v1.npz",
        distance=signals.distance,
        motion=signals.motion,
        motion_top=signals.motion_top,
        motion_bottom=signals.motion_bottom,
        fps=signals.fps,
        start_seconds=signals.start_seconds,
    )
    (out_dir / "shot_boundaries_v1.json").write_text(
        json.dumps(
            {
                "schema": "shot_boundaries_v1",
                "detector": "grid_histogram_native_cadence",
                "learned_model": None,
                "signal_resolution": [SIGNAL_WIDTH, SIGNAL_HEIGHT],
                "shots": len(shots),
                "frames": int(len(signals.distance)),
                **meta,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, help="source cadence; probed when omitted")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    fps = args.fps
    if fps is None:
        from cv.pipeline.broadcast_source import probe

        fps = float(probe(args.video)["fps"])
    signals = stream_signals(
        args.video, fps, start_seconds=args.start, duration_seconds=args.duration
    )
    cuts = detect_cuts(signals)
    shots = shots_from_cuts(signals, cuts)
    path = write_shots(
        args.out,
        shots,
        signals,
        {
            "video": args.video.name,
            "fps": fps,
            "start_seconds": args.start,
            "duration_seconds": args.duration,
        },
    )
    print(f"{len(signals.distance)} frames, {len(shots)} shots -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
