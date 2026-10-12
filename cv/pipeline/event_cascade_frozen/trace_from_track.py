"""Pass-1 minting evidence rendered from the pipeline's own ball track.

`trace.py` builds its candidates from a three-frame difference, which is what a fresh broadcast
has before the upstream runs.  That was measured at 60% nomination recall and collapsed on clay,
where crowd motion swamps a six-pixel ball (`minting_calibration/RESULT.md`).

This renders the same trajectory map from `ball_track_joint_native1080_arc_augmented_v2.csv`
instead -- the label-free track the upstream already produces -- so the question "does a real
track fix minting recall?" can be answered on clips that have owner truth, BEFORE spending the
GPU-hours to produce tracks for the new holdout.

The track is label-free: it is a detector plus a motion filter and knows nothing about events.
Using it here does not leak truth into the nominations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


from . import trace as tr  # noqa: E402

FRAME_RE = re.compile(r"f_(\d+)")


def load_track(path: Path, clip: str) -> dict[int, list[tuple[float, float, float]]]:
    """(frame -> [(x, y, area)]) in the 960x540 detection space `trace` draws in."""
    points: dict[int, list[tuple[float, float, float]]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None and "x_native" not in reader.fieldnames:
            # Older audit roots (cross_match_event_audit_v1..v4) wrote only the 960x540
            # `x`/`y` columns. Fail here rather than raising a bare KeyError per row.
            raise SystemExit(
                f"{path} has no x_native/y_native columns; it predates the native track "
                "columns. Use a root produced by the current tracking_composition."
            )
        for row in reader:
            if row["clip"] != clip:
                continue
            match = FRAME_RE.search(row["frame"])
            if not match:
                continue
            try:
                x = float(row["x_native"]) * tr.SCALE
                y = float(row["y_native"]) * tr.SCALE
            except (TypeError, ValueError):
                continue
            points.setdefault(int(match.group(1)), []).append((x, y, 12.0))
    return points


def render(
    frames_dir: Path,
    track_csv: Path,
    clip: str,
    output: Path,
    *,
    window: int = 60,
    overlap: int = 10,
) -> dict:
    frames = sorted(frames_dir.glob("f_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {frames_dir}")
    points = load_track(track_csv, clip)
    output.mkdir(parents=True, exist_ok=True)
    windows = []
    start = 1
    while start < len(frames):
        end = min(len(frames), start + window - 1)
        path = output / f"trace_{start:04d}_{end:04d}.jpg"
        # No dense-cell suppression: the track has one point per frame, so there is no player
        # clutter to remove and suppressing would delete a genuinely stationary ball.
        _render_window_no_suppress(frames, points, start, end, path)
        windows.append({"start": start, "end": end, "path": str(path)})
        if end >= len(frames):
            break
        start = end - overlap + 1
    covered = sum(1 for f in range(1, len(frames) + 1) if points.get(f))
    report = {
        "clip": clip,
        "frames": len(frames),
        "tracked_frames": covered,
        "track_coverage": covered / len(frames),
        "windows": windows,
    }
    (output / "track_evidence.json").write_text(json.dumps(report, indent=1))
    return report


def _render_window_no_suppress(frames, points, start, end, path):
    import cv2
    import numpy as np

    base = cv2.resize(tr._read(frames[(start + end) // 2 - 1]), None, fx=tr.SCALE, fy=tr.SCALE)
    canvas = (base * 0.35).astype(np.uint8)
    span = max(1, end - start)
    for frame in range(start, end + 1):
        for x, y, _area in points.get(frame, []):
            colour = tr._colour((frame - start) / span)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, colour, -1, cv2.LINE_AA)
        if frame % 5 == 0 and points.get(frame):
            x, y, _ = points[frame][0]
            cv2.putText(
                canvas, str(frame), (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
    cv2.putText(
        canvas, f"frames {start}-{end}   blue=early  red=late",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--clip", required=True, help="the clip column value, e.g. pt0003")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(render(args.frames, args.track, args.clip, args.output, window=args.window)))


if __name__ == "__main__":
    main()
