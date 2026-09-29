#!/usr/bin/env python3
"""Detect repeated visual frames before timing-sensitive event evaluation.

Some nominal 59.94 fps broadcasts contain lower-cadence pictures repeated into a CFR stream.
The timestamps are real, but adjacent frame indices are not independent observations. Treating
them as independent biases velocity, creates false trajectory kinks, and overstates event timing
precision.

This audit runs FFmpeg's blockwise duplicate detector on extracted native frames, records every
duplicate-equivalence group, and emits a label-free timing verdict. It never deletes or renumbers
frames. Downstream stages can zero-weight repeated observations, while annotation tools can store
an event interval spanning visually equivalent frames instead of an arbitrary member of the run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

SHOWINFO_TIME = re.compile(r"pts_time:([0-9.]+)")
MPDECIMATE = "mpdecimate=hi=1280:lo=640:frac=0.5,showinfo"
MONOTONIC_SOURCE_FILTER = "select=isnan(prev_selected_pts)+gt(pts\\,prev_selected_pts)"
SYSTEMATIC_DUPLICATE_RATE = 0.05
MINIMUM_SYSTEMATIC_DUPLICATES = 5


@dataclass(frozen=True)
class CadenceResult:
    match_id: str
    clip: str
    fps: float
    frame_count: int
    duplicate_frames: list[int]
    duplicate_pair_starts: list[int]
    equivalence_groups: list[list[int]]
    active_frame_count: int
    active_duplicate_frames: list[int]
    active_duplicate_rate: float
    isolated_duplicate_fraction: float
    effective_unique_fps: float
    decision: str
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def kept_frames_from_showinfo(stderr: str, fps: float) -> list[int]:
    """Map retained FFmpeg PTS values back to one-based input frame indices."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return [round(float(match.group(1)) * fps) + 1 for match in SHOWINFO_TIME.finditer(stderr)]


def equivalence_groups(duplicate_frames: list[int]) -> list[list[int]]:
    """Group a retained canonical frame with the repeated frames that follow it."""
    groups: list[list[int]] = []
    for duplicate in sorted(set(duplicate_frames)):
        if groups and duplicate == groups[-1][-1] + 1:
            groups[-1].append(duplicate)
        else:
            groups.append([duplicate - 1, duplicate])
    return groups


def _inside_spans(frame: int, spans: list[list[float]]) -> bool:
    return any(float(start) <= frame <= float(end) for start, end in spans)


def classify_cadence(
    *,
    match_id: str,
    clip: str,
    fps: float,
    frame_count: int,
    kept_frames: list[int],
    active_spans: list[list[float]],
    minimum_systematic_duplicates: int = MINIMUM_SYSTEMATIC_DUPLICATES,
    systematic_duplicate_rate: float = SYSTEMATIC_DUPLICATE_RATE,
    minimum_isolated_fraction: float | None = None,
) -> CadenceResult:
    duplicates = sorted(set(range(1, frame_count + 1)) - set(kept_frames))
    groups = equivalence_groups(duplicates)
    active_frames = [
        frame for frame in range(1, frame_count + 1) if _inside_spans(frame, active_spans)
    ]
    active_duplicates = [frame for frame in duplicates if _inside_spans(frame, active_spans)]
    active_groups = [
        group for group in groups if any(frame in active_duplicates for frame in group)
    ]
    isolated = sum(len(group) == 2 for group in active_groups)
    isolated_fraction = isolated / max(1, len(active_groups))
    duplicate_rate = len(active_duplicates) / max(1, len(active_frames))
    # A sustained static picture can look like a very large equivalence group, but
    # it is not an exposure-rate conversion.  Standard broadcast conversions have
    # short holds: up through five frames for 23.976 -> 29.97 4:5 pulldown.
    # Keep the compatibility argument above for callers that used the former
    # heuristic; regularity is determined later by cadence_action().
    del minimum_isolated_fraction
    largest_group = max((len(group) for group in active_groups), default=0)
    systematic = (
        len(active_duplicates) >= minimum_systematic_duplicates
        and duplicate_rate >= systematic_duplicate_rate
        and largest_group <= 5
    )
    reasons = ["systematic_repeated_visual_frames"] if systematic else []
    duration_seconds = frame_count / fps
    return CadenceResult(
        match_id=match_id,
        clip=clip,
        fps=fps,
        frame_count=frame_count,
        duplicate_frames=duplicates,
        duplicate_pair_starts=[frame - 1 for frame in duplicates],
        equivalence_groups=groups,
        active_frame_count=len(active_frames),
        active_duplicate_frames=active_duplicates,
        active_duplicate_rate=round(duplicate_rate, 6),
        isolated_duplicate_fraction=round(isolated_fraction, 6),
        effective_unique_fps=round((frame_count - len(duplicates)) / duration_seconds, 6),
        decision="timing_hold" if systematic else "timing_usable",
        reasons=reasons,
    )


def audit_clip(
    frames_dir: Path,
    *,
    match_id: str,
    clip: str,
    fps: float,
    active_spans: list[list[float]],
) -> CadenceResult:
    frame_count = len(list(frames_dir.glob("f_*.jpg")))
    if frame_count == 0:
        return CadenceResult(
            match_id=match_id,
            clip=clip,
            fps=fps,
            frame_count=0,
            duplicate_frames=[],
            duplicate_pair_starts=[],
            equivalence_groups=[],
            active_frame_count=0,
            active_duplicate_frames=[],
            active_duplicate_rate=0.0,
            isolated_duplicate_fraction=0.0,
            effective_unique_fps=0.0,
            decision="timing_hold",
            reasons=["no_extracted_frames"],
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "f_%04d.jpg"),
        "-vf",
        MPDECIMATE,
        "-an",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    kept = kept_frames_from_showinfo(completed.stderr, fps)
    return classify_cadence(
        match_id=match_id,
        clip=clip,
        fps=fps,
        frame_count=frame_count,
        kept_frames=kept,
        active_spans=active_spans,
    )


def audit_source_clip(
    source: Path,
    *,
    match_id: str,
    clip: str,
    fps: float,
    frame_count: int,
    active_spans: list[list[float]],
    decoder: str | None = None,
    minimum_systematic_duplicates: int = MINIMUM_SYSTEMATIC_DUPLICATES,
    systematic_duplicate_rate: float = SYSTEMATIC_DUPLICATE_RATE,
) -> CadenceResult:
    """Audit cadence before image serialization, preserving only increasing source PTS."""

    command = ["ffmpeg", "-hide_banner", "-nostdin", "-filter_threads", "1"]
    if decoder is not None:
        command.extend(["-c:v", decoder, "-threads", "1"])
        if decoder == "libdav1d":
            command.extend(["-tilethreads", "1", "-framethreads", "1", "-filmgrain", "0"])
    command.extend(
        [
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"{MONOTONIC_SOURCE_FILTER},{MPDECIMATE},showinfo",
            "-vsync",
            "0",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    kept = kept_frames_from_showinfo(completed.stderr, fps)
    return classify_cadence(
        match_id=match_id,
        clip=clip,
        fps=fps,
        frame_count=frame_count,
        kept_frames=kept,
        active_spans=active_spans,
        minimum_systematic_duplicates=minimum_systematic_duplicates,
        systematic_duplicate_rate=systematic_duplicate_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--active-play", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--match-id")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--fail-on-timing-hold", action="store_true")
    args = parser.parse_args()

    jobs = []
    if args.frames_root:
        if not args.match_id or not args.fps:
            parser.error("--frames-root requires --match-id and --fps")
        for clip_dir in sorted(args.frames_root.glob("pt*")):
            frame_count = len(list(clip_dir.glob("f_*.jpg")))
            if frame_count:
                jobs.append(
                    (
                        clip_dir,
                        args.match_id,
                        clip_dir.name,
                        args.fps,
                        [[1.0, float(frame_count)]],
                    )
                )
    else:
        if not args.audit_root or not args.active_play:
            parser.error("provide --frames-root or both --audit-root and --active-play")
        active_play = json.loads(args.active_play.read_text())
        for key, decision in sorted(active_play.items()):
            match_id, clip = key.split("/", 1)
            jobs.append(
                (
                    args.audit_root / match_id / "audit_frames_native_1080" / clip,
                    match_id,
                    clip,
                    float(decision["fps"]),
                    decision.get("active_spans", []),
                )
            )
    if not jobs:
        parser.error("no point frame directories found")

    def run(job: tuple[Path, str, str, float, list[list[float]]]) -> CadenceResult:
        frames_dir, match_id, clip, fps, spans = job
        return audit_clip(
            frames_dir,
            match_id=match_id,
            clip=clip,
            fps=fps,
            active_spans=spans,
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, jobs))
    held = [row for row in rows if row.decision == "timing_hold"]
    held_matches = sorted({row.match_id for row in held})
    payload = {
        "schema": "frame_cadence_audit_v1",
        "detector": MPDECIMATE,
        "points": len(rows),
        "timing_holds": len(held),
        "timing_hold_matches": held_matches,
        "rows": [row.as_dict() for row in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{len(rows)} points audited; {len(held)} timing holds across {len(held_matches)} matches"
    )
    for row in held:
        print(
            f"  {row.match_id}/{row.clip}: {len(row.active_duplicate_frames)}/"
            f"{row.active_frame_count} active frames repeated "
            f"({row.active_duplicate_rate:.1%})"
        )
    print(f"wrote {args.out}")
    if held and args.fail_on_timing_hold:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
