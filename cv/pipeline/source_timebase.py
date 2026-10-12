"""Audit every decoded source-frame PTS without changing pictures or retaining a huge dump."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from cv.pipeline.provenance import AUTOMATIC_MODE, build_provenance, file_record


MAX_LOCAL_DEFECT_SPANS = 256


class NativeClockIrregularError(ValueError):
    """One point's original pictures do not sit on a regular, authenticated source clock.

    Raised for a per-point timing defect (a source PTS gap, skip or splice inside the point's
    window, or extracted picture epochs that are not contiguous source frames). Consumers that
    work point by point (portal export) hold that point with this reason instead of failing
    the whole match; everything else still sees an ordinary ``ValueError``.
    """


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
    # Local defects (a missing, repeated or gapped timestamp) are the frames a later
    # point-level check can hold; a phase error alone only says an earlier gap moved the
    # nominal clock, so it is counted but not located.
    local_count = 0
    local_spans: list[list] = []
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
        if any(reason != "nominal_phase_error" for reason in reasons):
            local_count += 1
            pts = value if value is not None and math.isfinite(value) else None
            if local_spans and local_spans[-1][1] == index:
                local_spans[-1][1] = index + 1
                local_spans[-1][3] = pts if pts is not None else local_spans[-1][3]
            elif len(local_spans) < MAX_LOCAL_DEFECT_SPANS:
                local_spans.append([index + 1, index + 1, previous, pts])
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
        "local_defect_frames": local_count,
        # [first frame, last frame, pts before the span, last pts in it] (1-based frames)
        "local_defect_spans": local_spans,
        "local_defect_spans_truncated": local_count > 0
        and len(local_spans) >= MAX_LOCAL_DEFECT_SPANS,
        "nominal_timeline_valid": count >= 2 and invalid_count == 0,
    }


def _audit_on_nvdec(video: Path, fps: float, inventory: Path | None = None) -> dict | None:
    """The same audit from NVDEC's decoded-frame PTS (``TENNIS_VIDEO_DECODE=nvdec``), or None.

    Only a clean decode is reported: a decoder exception, or any packet without its decoded
    frame, returns None so the CPU audit below decides, keeping its fail-closed verdicts.
    ``inventory`` receives the PTS as FFprobe's compact lines (``%f`` of the same double).
    """
    from cv.pipeline import nvdec_video

    found = nvdec_video.decoded_pts_times(str(video))
    if found is None:
        return None
    times, status = found
    started = time.monotonic()
    pending = None if inventory is None else inventory.with_name(inventory.name + ".pending")
    if pending is None:
        report = audit_timestamps(times, fps)
    else:
        with pending.open("w") as handle:
            report = audit_timestamps(_tee_lines(times, handle), fps)
    if status["error"] or status["frames"] != status["packets"]:
        print(f"source_timebase: NVDEC audit not clean ({status}); CPU audit", flush=True)
        if pending is not None:
            pending.unlink(missing_ok=True)
        return None
    if pending is not None:
        pending.replace(inventory)
    report.update(
        {
            "command": [status["decoder"], status["library"], "decoded frame pts", str(video)],
            "decoder_return_code": 0,
            "decoder_error_excerpt": "",
            "elapsed_seconds": time.monotonic() - started,
            "completed": True,
            "stream_information_discovery": False,
            "audio_integrity": "not_audited_by_video_clock_scan",
        }
    )
    return report


def _tee_lines(times: Iterable[float], handle) -> Iterator[float]:
    for value in times:
        handle.write(PTS_LINE % value)
        yield value


#: One FFprobe ``-show_frames -show_entries frame=pts_time -of compact=p=1:nk=0`` record.
PTS_LINE = "frame|pts_time=%f\n"


def audit_source_timebase(video: Path, fps: float, inventory: Path | None = None) -> dict:
    """Stream decoded PTS with bounded memory and one CPU decoder thread.

    A nonzero decoder exit or error output fails closed, including tail corruption
    that could otherwise leave a regular but incomplete prefix. The enclosing
    source-stage receipt binds this report to the source bytes and code revision.

    Disable unrelated stream-info decoding: FFprobe 8.0.1's Opus parser can emit
    an error on its empty EOF flush even for a newly generated valid MKV. Selecting
    v:0 alone still initializes that audio parser. This is a video-clock audit,
    not an audio integrity certificate; no emitted decoder error is suppressed.

    ``inventory``: also write every frame's PTS line there (FFprobe's own stdout, or NVDEC's
    identical lines), published only for a completed scan, so the S6 source clock
    (``s6_broadcast_backend.source_pts_cache``) need not decode the whole source again.
    """
    nvdec = _audit_on_nvdec(video, fps, inventory)
    if nvdec is not None:
        return nvdec
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
    pending = None if inventory is None else inventory.with_name(inventory.name + ".pending")
    with (
        tempfile.TemporaryFile(mode="w+t") as errors,
        open(os.devnull, "w") if pending is None else pending.open("w") as copy,
    ):
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True) as process:
            assert process.stdout is not None
            try:
                report = audit_timestamps(frame_pts(_copied(process.stdout, copy)), fps)
            except BaseException:
                process.kill()
                raise
            return_code = process.wait()
        errors.seek(0)
        error_text = errors.read(4096).strip()
    if pending is not None:
        if return_code == 0 and not error_text:
            pending.replace(inventory)
        else:
            pending.unlink(missing_ok=True)
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


def _copied(lines: Iterable[str], handle) -> Iterator[str]:
    for line in lines:
        handle.write(line)
        yield line


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
