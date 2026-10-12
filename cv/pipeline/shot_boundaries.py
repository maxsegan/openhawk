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

``--gpu-shrink`` (default from ``TENNIS_SHOT_GPU_SHRINK=1``; off by default) is OUTPUT-CHANGING,
slightly: NVDEC decodes and the card shrinks each frame to the 96x54 grey signal, so only 5 KB
per frame leaves the GPU instead of a 3 MB picture piped through ffmpeg (Nadal-Kyrgios: 14.5 min
and 1,611 CPU-s with NVDEC feeding ffmpeg -> about 4 min and 100 CPU-s). The grey follows
ffmpeg 8's ``scale=96:54,format=gray`` (BT.709 decode to clipped RGB, BT.601 weights) with a
bicubic shrink, within one grey level on 87% of pixels. On four full matches 94-96% of cuts and
90-96% of serve attempts are unchanged, as close as the same code run with ffmpeg 6.1.1
(upstream-dive report section 4.1). The GPU used is ``TENNIS_VIDEO_DECODE=nvdec:<gpu>``'s, else
``TENNIS_NVDEC_GPU``, else 0. Spans (``--start``/``--duration``), sources other than
limited-range BT.709 8-bit 4:2:0, a missing PyNvVideoCodec or a card with less than 2 GB free
take the ffmpeg path, reported on stderr. ``shot_boundaries_v1.json`` names the path
(``signal_path``) only when the GPU produced the signal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv.pipeline import nvdec_video, source_coverage

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
    accumulator = _SignalAccumulator()
    process = nvdec_video.popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            payload = process.stdout.read(frame_bytes * CHUNK_FRAMES)
            if not payload:
                break
            usable = len(payload) - len(payload) % frame_bytes
            accumulator.add(
                np.frombuffer(payload[:usable], dtype=np.uint8).reshape(
                    -1, SIGNAL_HEIGHT, SIGNAL_WIDTH
                )
            )
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {stderr.decode()[:2000]}")
    if not accumulator.distances:
        raise RuntimeError(f"no frames decoded from {video}")
    return accumulator.signals(fps, start_seconds)


class _SignalAccumulator:
    """Per-frame cut and motion signals from consecutive chunks of 96x54 grey frames."""

    def __init__(self) -> None:
        self.distances: list[np.ndarray] = []
        self.motions: list[np.ndarray] = []
        self.tops: list[np.ndarray] = []
        self.bottoms: list[np.ndarray] = []
        self.previous_histogram: np.ndarray | None = None
        self.previous_frame: np.ndarray | None = None

    def add(self, frames: np.ndarray) -> None:
        half = SIGNAL_HEIGHT // 2
        histograms = _grid_histograms(frames)
        if self.previous_histogram is not None:
            histograms = np.vstack([self.previous_histogram[None, :], histograms])
            frames = np.vstack([self.previous_frame[None, :, :], frames])
        else:
            for values in (self.distances, self.motions, self.tops, self.bottoms):
                values.append(np.zeros(1, dtype=np.float32))
        difference = np.abs(np.diff(histograms, axis=0)).sum(axis=1) / (2.0 * GRID * GRID)
        deltas = np.abs(np.diff(frames.astype(np.int16), axis=0)).astype(np.float32)
        self.distances.append(difference.astype(np.float32))
        self.motions.append(deltas.mean(axis=(1, 2)))
        self.tops.append(deltas[:, :half, :].mean(axis=(1, 2)))
        self.bottoms.append(deltas[:, half:, :].mean(axis=(1, 2)))
        self.previous_histogram = histograms[-1]
        self.previous_frame = frames[-1]

    def signals(self, fps: float, start_seconds: float) -> Signals:
        return Signals(
            fps=fps,
            start_seconds=start_seconds,
            distance=np.concatenate(self.distances),
            motion=np.concatenate(self.motions),
            motion_top=np.concatenate(self.tops),
            motion_bottom=np.concatenate(self.bottoms),
        )


GPU_BATCH = 256


def gpu_grey(nv12, height: int, width: int):
    """(B, 3H/2, W) uint8 NV12 tensor -> (B, 54, 96) uint8 grey, ffmpeg 8's
    ``scale=96:54,format=gray`` of a limited-range BT.709 source: planes shrunk (bicubic,
    antialiased), decoded to RGB with BT.709 and clipped, mixed with BT.601 weights. (Measured on
    synthetic colour patches, ffmpeg's own grey path also darkens by ~7 levels at this 20x
    shrink; imitating that changed no cut or attempt, so it is left out.)"""
    import torch.nn.functional as F

    size = (SIGNAL_HEIGHT, SIGNAL_WIDTH)

    def shrink(planes):
        return F.interpolate(planes, size=size, mode="bicubic", antialias=True, align_corners=False)

    luma = shrink(nv12[:, None, :height].float())
    chroma = nv12[:, height : height + height // 2].reshape(-1, height // 2, width // 2, 2)
    chroma = shrink(chroma.permute(0, 3, 1, 2).float())
    y = (luma - 16.0) * (255.0 / 219.0)
    u = (chroma[:, :1] - 128.0) * (255.0 / 224.0)
    v = (chroma[:, 1:] - 128.0) * (255.0 / 224.0)
    r = (y + 1.5748 * v).clamp_(0, 255)
    g = (y - 0.1873 * u - 0.4681 * v).clamp_(0, 255)
    b = (y + 1.8556 * u).clamp_(0, 255)
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return grey.round_().clamp_(0, 255).to(dtype=nv12.dtype).squeeze(1)


def _gpu_note(message: str) -> None:
    print(f"shot_boundaries: {message}; ffmpeg path", file=sys.stderr, flush=True)


def shrink_gpu() -> int:
    """The GPU for ``--gpu-shrink``: NVDEC's when that switch names one."""
    gpu = nvdec_video.requested_gpu()
    return int(os.environ.get("TENNIS_NVDEC_GPU", "0")) if gpu is None else gpu


def gpu_signals(
    video: Path | str,
    fps: float,
    gpu: int,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> Signals | None:
    """Whole-video signals from NVDEC frames shrunk on the card, or None where this path cannot
    stand in (the caller then takes the ffmpeg path)."""
    if start_seconds or duration_seconds is not None:
        _gpu_note("--gpu-shrink covers whole videos only")
        return None
    try:
        import PyNvVideoCodec  # noqa: F401
        import torch

        info = nvdec_video.probe(os.fspath(video))
        if info is None:
            _gpu_note(f"{video} is not 8-bit 4:2:0")
            return None
        colour = (info.color.get("color_range"), info.color.get("colorspace"))
        if colour != (1, 1):  # AVCOL_RANGE_MPEG, AVCOL_SPC_BT709
            _gpu_note(f"{video} is not limited-range BT.709 ({info.color})")
            return None
        free_mib = torch.cuda.mem_get_info(gpu)[0] >> 20
        if free_mib < nvdec_video.MIN_FREE_MIB:
            _gpu_note(f"cuda:{gpu} has {free_mib} MiB free")
            return None
        decoder = nvdec_video._decoder(os.fspath(video), gpu, info)
    except Exception as error:  # no library, or the decoder refused this codec/GPU
        _gpu_note(f"GPU shrink unavailable ({error})")
        return None
    height, width = info.height, info.width
    accumulator = _SignalAccumulator()
    with torch.cuda.device(decoder.device):
        batch = torch.empty(
            (GPU_BATCH, height * 3 // 2, width), dtype=torch.uint8, device=decoder.device
        )
        host = torch.empty((GPU_BATCH, SIGNAL_HEIGHT, SIGNAL_WIDTH), dtype=torch.uint8).pin_memory()
        count = 0
        for _, nv12 in decoder.frames_from(info.start_pts):
            batch[count].copy_(nv12[: height * 3 // 2])
            count += 1
            if count == GPU_BATCH:
                host.copy_(gpu_grey(batch, height, width))
                accumulator.add(host.numpy().copy())
                count = 0
        if count:
            host[:count].copy_(gpu_grey(batch[:count], height, width))
            accumulator.add(host[:count].numpy().copy())
    if not accumulator.distances:
        raise RuntimeError(f"no frames decoded from {video}")
    return accumulator.signals(fps, start_seconds)


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
    parser.add_argument(
        "--gpu-shrink",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TENNIS_SHOT_GPU_SHRINK", "0") == "1",
        help="NVDEC + shrink on the GPU (OUTPUT-CHANGING, slightly; see the module docstring)",
    )
    args = parser.parse_args()
    fps = args.fps
    if fps is None:
        from cv.pipeline.broadcast_source import probe

        fps = float(probe(args.video)["fps"])
    signals = None
    if args.gpu_shrink:
        signals = gpu_signals(
            args.video,
            fps,
            shrink_gpu(),
            start_seconds=args.start,
            duration_seconds=args.duration,
        )
    meta = {} if signals is None else {"signal_path": "gpu_shrink_bt709_v1"}
    if signals is None:
        signals = stream_signals(
            args.video, fps, start_seconds=args.start, duration_seconds=args.duration
        )
    expected = args.duration
    if expected is None:
        expected = source_coverage.source_seconds(args.video) - args.start
    coverage = source_coverage.check("shot_signals", len(signals.distance) / fps, expected)
    if coverage:
        meta["coverage"] = coverage
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
            **meta,
        },
    )
    print(f"{len(signals.distance)} frames, {len(shots)} shots -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
