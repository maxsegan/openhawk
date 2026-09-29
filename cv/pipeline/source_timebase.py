"""Audit every decoded source-frame PTS without changing pictures or retaining a huge dump."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from cv.pipeline.provenance import AUTOMATIC_MODE, build_provenance, file_record


def frame_pts(lines: Iterable[str]) -> Iterator[float | None]:
    """Parse FFprobe compact frame records, never substituting a best-effort timestamp."""
    for line in lines:
        fields = line.strip().split("|")
        if fields[0] != "frame":
            continue
        values = [field.split("=", 1)[1] for field in fields[1:] if field.startswith("pts_time=")]
        try:
            yield float(values[0]) if len(values) == 1 else None
        except ValueError:
            yield None


def audit_timestamps(timestamps: Iterable[float | None], fps: float) -> dict:
    """Bounded-memory global cadence check; local regularity cannot conceal accumulated drift."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source fps must be finite and positive")
    tolerance = max(0.0011, 0.02 / fps)
    first = previous = last = None
    count = invalid_count = 0
    maximum_phase = 0.0
    min_interval = max_interval = None
    examples = []
    for index, value in enumerate(timestamps):
        count += 1
        reasons = []
        if value is None or not math.isfinite(value):
            reasons.append("missing_or_nonfinite_pts")
        else:
            last = value
            if index == 0:
                first = value
            if first is not None:
                phase = abs(value - first - index / fps)
                maximum_phase = max(maximum_phase, phase)
                if phase > tolerance:
                    reasons.append("nominal_phase_error")
            if previous is not None:
                interval = value - previous
                min_interval = interval if min_interval is None else min(min_interval, interval)
                max_interval = interval if max_interval is None else max(max_interval, interval)
                if interval <= 0:
                    reasons.append("nonincreasing_pts")
                if abs(interval - 1.0 / fps) > tolerance:
                    reasons.append("nominal_interval_error")
        if reasons:
            invalid_count += 1
            if len(examples) < 32:
                examples.append({"frame_index": index + 1, "reasons": reasons})
        previous = value if value is not None and math.isfinite(value) else None
    return {
        "schema": "source_full_timebase_v1",
        "timestamp_field": "frame.pts_time",
        "scope": "every_decoded_video_frame",
        "frame_index_origin": 1,
        "frames": count,
        "fps": fps,
        "first_pts_seconds": first,
        "last_pts_seconds": last,
        "pts_tolerance_seconds": tolerance,
        "maximum_nominal_phase_error_seconds": maximum_phase if first is not None else None,
        "min_interval_seconds": min_interval,
        "max_interval_seconds": max_interval,
        "invalid_frames": invalid_count,
        "invalid_examples": examples,
        "nominal_timeline_valid": count >= 2 and invalid_count == 0,
    }


def audit_source_timebase(video: Path, fps: float) -> dict:
    """Stream decoded PTS with bounded memory and one CPU decoder thread.

    A nonzero decoder exit or error output fails closed, including tail corruption
    that could otherwise leave a regular but incomplete prefix. The enclosing
    source-stage receipt binds this report to the source bytes and code revision.

    Disable unrelated stream-info decoding: FFprobe 8.0.1's Opus parser can emit
    an error on its empty EOF flush even for a newly generated valid MKV. Selecting
    v:0 alone still initializes that audio parser. This is a video-clock audit,
    not an audio integrity certificate; no emitted decoder error is suppressed.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-nofind_stream_info",
        "-threads",
        "1",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "compact=p=1:nk=0",
        str(video),
    ]
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+t") as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True) as process:
            assert process.stdout is not None
            try:
                report = audit_timestamps(frame_pts(process.stdout), fps)
            except BaseException:
                process.kill()
                raise
            return_code = process.wait()
        errors.seek(0)
        error_text = errors.read(4096).strip()
    report.update(
        {
            "command": command,
            "decoder_return_code": return_code,
            "decoder_error_excerpt": error_text,
            "elapsed_seconds": time.monotonic() - started,
            "completed": return_code == 0 and not error_text,
            "stream_information_discovery": False,
            "audio_integrity": "not_audited_by_video_clock_scan",
        }
    )
    report["nominal_timeline_valid"] &= report["completed"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = file_record(args.video, role="source_video")
    provenance = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=AUTOMATIC_MODE,
        source_videos=[source],
        configuration={"fps": args.fps, "decoder_threads": 1},
    )
    report = audit_source_timebase(args.video, args.fps)
    if file_record(args.video, role="source_video") != source:
        raise ValueError("source video changed during timestamp audit")
    report["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["nominal_timeline_valid"]:
        raise SystemExit("source timebase rejected; see report")


if __name__ == "__main__":
    main()
