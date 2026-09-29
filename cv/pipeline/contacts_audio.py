"""Detect sharp broadcast-audio onsets as contact-timing candidates.

This is an independent timing feature for the ball-physics contact chain, not a replacement
for hit/bounce classification. Racket impacts tend to produce sharp high-frequency onsets;
the source has a stable audio/video delay, exposed as ``--audio-delay-frames``. Every emitted
event is located on the video timeline and decorated with the interpolated WASB position so a
12-frame visual audit can reject crowd, shoe, and court-bounce sounds before fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contacts_v2 import (  # noqa: E402
    contact_f1,
    count_report,
    evaluation_points,
    load_boxes,
    load_contact_ground_truth,
    player_distance,
    split_ground_truth,
    write_event_audit,
)
from run_manifest import StageRun  # noqa: E402

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
    """Robust-z positive spectral flux in the racket-impact frequency band."""
    if len(samples) < 2 * fft_samples:
        return np.empty(0, dtype=np.float32)
    window = np.hanning(fft_samples)
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(samples[i : i + fft_samples] * window))
            for i in range(0, len(samples) - fft_samples + 1, hop_samples)
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
    """Load matching per-window spectral flux, computing only absent or stale entries."""
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
        # Preserve valid entries from another point subset so fixed-slice runs do not evict
        # the full-match cache (or vice versa).
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


def load_windows(out_dir: str, point_map: str, points: set[int]) -> dict[str, tuple[float, float]]:
    windows = {}
    with open(os.path.join(out_dir, point_map), newline="") as handle:
        for row in csv.DictReader(handle):
            pt = int(row["pt"])
            if pt in points and row.get("rally_t_start") and row.get("rally_t_end"):
                windows[f"pt{pt:04d}"] = (
                    float(row["rally_t_start"]),
                    float(row["rally_t_end"]),
                )
    return windows


def load_ball_positions(out_dir: str, name: str, score_threshold: float):
    rows = defaultdict(dict)
    with open(os.path.join(out_dir, name), newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["score"]) >= score_threshold:
                frame = int(row["frame"][2:6])
                rows[row["clip"]][frame] = (float(row["x"]), float(row["y"]))
    return rows


def interpolated_position(positions: dict[int, tuple[float, float]], frame: int):
    if not positions:
        return None
    frames = np.array(sorted(positions))
    if frame < frames[0] or frame > frames[-1]:
        return None
    xy = np.array([positions[int(index)] for index in frames])
    return tuple(float(np.interp(frame, frames, xy[:, axis])) for axis in (0, 1))


def build_events(
    scores_by_clip: dict[str, np.ndarray],
    positions,
    boxes,
    threshold: float,
    fps: float,
    sample_rate: int,
    min_gap_seconds: float,
    audio_delay_seconds: float,
) -> dict[str, list[dict]]:
    by_clip = {}
    for clip, scores in scores_by_clip.items():
        events = []
        for frame, score in onset_frames(
            scores,
            threshold,
            fps,
            sample_rate,
            FFT_SAMPLES,
            HOP_SAMPLES,
            min_gap_seconds,
            audio_delay_seconds,
        ):
            xy = interpolated_position(positions.get(clip, {}), frame)
            if xy is None:
                continue
            u, v = xy
            events.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "kind": "hit",
                    "track_id": -1,
                    "u": u,
                    "v": v,
                    "player_distance_px": player_distance(boxes, clip, frame, u, v),
                    "rms_px": 0.0,
                    "confidence": score,
                }
            )
        by_clip[clip] = events
    return by_clip


def write_events(out_dir: str, events: list[dict], output_tag: str = "") -> None:
    fields = [
        "clip",
        "frame",
        "kind",
        "track_id",
        "u",
        "v",
        "player_distance_px",
        "rms_px",
        "confidence",
    ]
    suffix = f"_{output_tag}" if output_tag else ""
    with open(
        os.path.join(out_dir, f"contact_events_audio{suffix}.csv"), "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: event[field] for field in fields} for event in events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--point-map", default="point_video_map.csv")
    parser.add_argument("--points-file", default="", help="optional fixed evaluation point IDs")
    parser.add_argument("--frames-dir", default="rally_frames_50_contact_v2")
    parser.add_argument("--candidates", default="ball_candidates_wasb_contact_v2.csv")
    parser.add_argument("--boxes", default="player_boxes_50_contact_v2.csv")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--audio-cache",
        default="contact_audio_scores_16k_v1.npz",
        help="shared spectral-flux cache (relative paths are resolved under --out; empty disables)",
    )
    parser.add_argument("--score-threshold", type=float, default=10.0)
    parser.add_argument("--thresholds", default="3,5,8,10,15,20,30,40")
    parser.add_argument("--ball-score-threshold", type=float, default=0.3)
    parser.add_argument("--min-gap-seconds", type=float, default=0.35)
    parser.add_argument(
        "--audio-delay-seconds",
        type=float,
        default=0.08,
        help="audio-to-video correction in seconds; cadence invariant",
    )
    parser.add_argument("--contact-ground-truth", default="")
    parser.add_argument("--tolerance-frames", type=int, default=5)
    parser.add_argument("--audit", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-mcp-counts", action="store_true")
    args = parser.parse_args()
    stage_run = StageRun(args.out, "contacts_audio", args, seed=args.seed)

    eval_points = set(
        evaluation_points(args.out, args.point_map, args.frames_dir, args.points_file)
    )
    windows = load_windows(args.out, args.point_map, eval_points)
    cache_path = args.audio_cache
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(args.out, cache_path)
    scores_by_clip, cache_report = load_or_compute_audio_scores(
        args.video, windows, args.sample_rate, cache_path
    )
    print(
        f"audio cache: {cache_report['cache_hits']} hit / "
        f"{cache_report['cache_misses']} miss -> {cache_report['cache_path'] or 'disabled'}"
    )
    positions = load_ball_positions(args.out, args.candidates, args.ball_score_threshold)
    boxes = load_boxes(args.out, args.boxes, args.fps, args.fps)
    gt_points = split_ground_truth(args.match) if args.report_mcp_counts else []
    truth = (
        load_contact_ground_truth(args.contact_ground_truth) if args.contact_ground_truth else []
    )

    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    if args.score_threshold not in thresholds:
        thresholds.append(args.score_threshold)
    selected_by_clip, selected_summary, selected_score = {}, "", None
    for threshold in sorted(set(thresholds)):
        by_clip = build_events(
            scores_by_clip,
            positions,
            boxes,
            threshold,
            args.fps,
            args.sample_rate,
            args.min_gap_seconds,
            args.audio_delay_seconds,
        )
        _, summary = count_report(by_clip, sorted(eval_points), gt_points)
        flat = [event for events in by_clip.values() for event in events]
        line = f"audio_z={threshold:g} | {summary}"
        score = None
        if truth:
            score = contact_f1(flat, truth, args.tolerance_frames)
            line += (
                f" | contact_f1 P={score['precision']:.3f} R={score['recall']:.3f} "
                f"F1={score['f1']:.3f} TP={score['tp']} pred={score['n_pred']} "
                f"gt={score['n_truth']}"
            )
        print(line)
        if threshold == args.score_threshold:
            selected_by_clip, selected_summary, selected_score = by_clip, summary, score

    selected = [event for events in selected_by_clip.values() for event in events]
    write_events(args.out, selected, args.output_tag)
    audit_path = write_event_audit(
        args.out,
        args.frames_dir,
        selected,
        boxes,
        args.audit,
        args.seed,
        tag="audio" + (f"_{args.output_tag}" if args.output_tag else ""),
    )
    if audit_path:
        print(f"audit -> {audit_path}")
    stage_run.finish(
        outputs={
            "evaluation_points": len(eval_points),
            "events": len(selected),
            "summary": selected_summary,
            "frame_score": (
                {key: value for key, value in selected_score.items() if key != "matches"}
                if selected_score
                else None
            ),
            "audit_samples": min(args.audit, len(selected)),
            "audit_path": audit_path,
            "audio_cache": cache_report,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
