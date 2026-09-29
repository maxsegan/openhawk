"""Label-free short-window audio features shared by inference and diagnostics."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

SAMPLE_RATE = 16_000
FFT_SAMPLES = 320
HOP_SAMPLES = 160
AUDIO_CACHE_SCHEMA = 1


def read_audio(video: str, start: float, duration: float, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        str(start),
        "-i",
        video,
        "-t",
        str(duration),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def onset_scores(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    fft_samples: int = FFT_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> np.ndarray:
    """Return robust-z positive spectral flux in the impact-frequency band."""
    if len(samples) < 2 * fft_samples:
        return np.empty(0, dtype=np.float32)
    window = np.hanning(fft_samples)
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(samples[index : index + fft_samples] * window))
            for index in range(0, len(samples) - fft_samples + 1, hop_samples)
        ]
    )
    frequencies = np.fft.rfftfreq(fft_samples, 1.0 / sample_rate)
    band = (frequencies >= 1800.0) & (frequencies <= 7000.0)
    flux = np.maximum(np.diff(spectra[:, band], axis=0), 0.0).sum(axis=1)
    flux = np.r_[0.0, flux]
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    return ((flux - median) / max(1.4826 * mad, 1e-9)).astype(np.float32)


def _audio_cache_identity(video: str, sample_rate: int) -> dict:
    path = Path(video).resolve()
    stat = path.stat()
    return {
        "schema": AUDIO_CACHE_SCHEMA,
        "video": str(path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "sample_rate": sample_rate,
        "fft_samples": FFT_SAMPLES,
        "hop_samples": HOP_SAMPLES,
    }


def _load_audio_cache(path: str, identity: dict) -> tuple[dict[str, np.ndarray], dict]:
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if any(metadata.get(key) != value for key, value in identity.items()):
                return {}, {}
            scores = {
                clip: cached[f"scores_{clip}"].astype(np.float32, copy=False)
                for clip in metadata.get("windows", {})
                if f"scores_{clip}" in cached
            }
            return scores, metadata.get("windows", {})
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}, {}


def _write_audio_cache(
    path: str,
    identity: dict,
    scores_by_clip: dict[str, np.ndarray],
    windows: dict[str, list[float]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    metadata = {**identity, "windows": windows}
    with tempfile.NamedTemporaryFile(
        prefix=".contact_audio_",
        suffix=".npz",
        dir=os.path.dirname(path) or ".",
        delete=False,
    ) as handle:
        temporary = handle.name
    try:
        np.savez_compressed(
            temporary,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            **{f"scores_{clip}": scores for clip, scores in scores_by_clip.items()},
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_compute_audio_scores(
    video: str,
    windows: dict[str, tuple[float, float]],
    sample_rate: int,
    cache_path: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, int | str]]:
    """Load matching per-window spectral flux, computing absent or stale entries."""
    identity = _audio_cache_identity(video, sample_rate)
    cached_scores, cached_windows = _load_audio_cache(cache_path, identity)
    requested = {clip: [float(start), float(end)] for clip, (start, end) in windows.items()}
    scores_by_clip: dict[str, np.ndarray] = {}
    hits = 0
    for clip, bounds in requested.items():
        if cached_windows.get(clip) == bounds and clip in cached_scores:
            scores_by_clip[clip] = cached_scores[clip]
            hits += 1
            continue
        start, end = bounds
        scores_by_clip[clip] = onset_scores(read_audio(video, start, end - start, sample_rate))

    if cache_path:
        merged_scores = dict(cached_scores)
        merged_scores.update(scores_by_clip)
        merged_windows = dict(cached_windows)
        merged_windows.update(requested)
        _write_audio_cache(cache_path, identity, merged_scores, merged_windows)
    return scores_by_clip, {
        "cache_path": cache_path,
        "cache_hits": hits,
        "cache_misses": len(windows) - hits,
    }


def onset_frames(
    scores: np.ndarray,
    threshold: float,
    fps: float,
    sample_rate: int,
    fft_samples: int,
    hop_samples: int,
    min_gap_seconds: float,
    audio_delay_seconds: float,
) -> list[tuple[int, float]]:
    distance = max(1, round(min_gap_seconds * sample_rate / hop_samples))
    peaks, properties = find_peaks(scores, height=threshold, distance=distance, prominence=2.0)
    events = []
    for peak, score in zip(peaks, properties["peak_heights"]):
        seconds = (peak * hop_samples + 0.5 * fft_samples) / sample_rate
        frame = max(1, round((seconds - audio_delay_seconds) * fps) + 1)
        events.append((frame, float(score)))
    return events


def evaluation_points(
    out_dir: str, point_map: str, frames_dir: str, points_file: str = ""
) -> list[int]:
    if points_file:
        with open(points_file) as handle:
            return [
                int(line) for raw in handle if (line := raw.strip()) and not line.startswith("#")
            ]
    path = os.path.join(out_dir, point_map)
    if os.path.exists(path):
        with open(path, newline="") as handle:
            return [
                int(row["pt"])
                for row in csv.DictReader(handle)
                if row.get("rally_t_start") and row.get("rally_t_end")
            ]
    root = os.path.join(out_dir, frames_dir)
    return sorted(int(name[2:]) for name in os.listdir(root) if name.startswith("pt"))


def load_windows(out_dir: str, point_map: str, points: set[int]) -> dict[str, tuple[float, float]]:
    windows = {}
    with open(os.path.join(out_dir, point_map), newline="") as handle:
        for row in csv.DictReader(handle):
            point = int(row["pt"])
            if point in points and row.get("rally_t_start") and row.get("rally_t_end"):
                windows[f"pt{point:04d}"] = (
                    float(row["rally_t_start"]),
                    float(row["rally_t_end"]),
                )
    return windows
