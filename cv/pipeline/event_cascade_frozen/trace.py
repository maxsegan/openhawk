"""Label-free ball-motion trace evidence for truth minting.

Astra cannot run a shell in this environment (codex's bubblewrap sandbox cannot create user
namespaces), so every piece of evidence has to be rendered here and attached to the prompt.
A whole 1920x1080 frame downscaled to fit a prompt turns a tennis ball into two pixels, so a
frame-by-frame sweep is hopeless.  What survives downscaling is the ball's *arc*: this module
draws it.

Method, deliberately classical and label-free:
  * three-frame minimum difference  d_i = min(|f_i - f_{i-1}|, |f_i - f_{i+1}|)  suppresses the
    ghost trail a plain pairwise difference leaves behind and tolerates slow camera pan;
  * threshold, open, and keep small round bright blobs (a ball, not a player or a line);
  * plot every surviving candidate over a window on one dimmed frame, coloured by time.

The output is a trajectory map, not a track: no association, no filter, no truth.  Direction
reversals in the map are where impacts are, which is all pass 1 needs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SCALE = 0.5  # 1920x1080 -> 960x540 for detection
MIN_AREA = 3
MAX_AREA = 260
MAX_PER_FRAME = 6
DIFF_THRESHOLD = 16


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def candidates(frames: list[Path]) -> dict[int, list[tuple[float, float, float]]]:
    """Per 1-based frame index, up to MAX_PER_FRAME (x, y, area) in the detection space."""
    small = [
        cv2.cvtColor(cv2.resize(_read(p), None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
        for p in frames
    ]
    out: dict[int, list[tuple[float, float, float]]] = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for index in range(1, len(small) - 1):
        previous = cv2.absdiff(small[index], small[index - 1])
        following = cv2.absdiff(small[index], small[index + 1])
        moving = cv2.min(previous, following)
        _, mask = cv2.threshold(moving, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rows = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            if max(width, height) > 26 or min(width, height) < 1:
                continue
            if max(width, height) / max(1.0, min(width, height)) > 4.0:
                continue
            fill = area / max(1.0, width * height)
            if fill < 0.35:
                continue
            x, y = centroids[label]
            rows.append((float(x), float(y), area))
        rows.sort(key=lambda row: -row[2])
        out[index + 1] = rows[:MAX_PER_FRAME]
    return out


def suppress_dense_cells(
    points: dict[int, list[tuple[float, float, float]]],
    start: int,
    end: int,
    *,
    cell: int = 28,
    keep_ratio: float = 0.12,
) -> dict[int, list[tuple[float, float, float]]]:
    """Drop candidates that sit where candidates keep appearing.

    A player's limbs fire the difference detector in the same few cells for most of a window;
    a ball passes through a cell once or twice.  Counting candidates per spatial cell over the
    window and dropping the crowded cells therefore removes players without any model of what
    a player looks like.  The threshold is a fraction of the window length, so it scales with
    how long the window is rather than being a magic pixel count.
    """
    counts: dict[tuple[int, int], int] = {}
    for frame in range(start, end + 1):
        for x, y, _area in points.get(frame, []):
            counts[(int(x) // cell, int(y) // cell)] = (
                counts.get((int(x) // cell, int(y) // cell), 0) + 1
            )
    limit = max(3, int(keep_ratio * (end - start + 1)))
    kept: dict[int, list[tuple[float, float, float]]] = {}
    for frame in range(start, end + 1):
        rows = [
            row
            for row in points.get(frame, [])
            if counts.get((int(row[0]) // cell, int(row[1]) // cell), 0) <= limit
        ]
        kept[frame] = rows
    return kept


def _colour(fraction: float) -> tuple[int, int, int]:
    """Blue (early) -> red (late), so time order is readable without labels."""
    value = int(np.clip(fraction, 0.0, 1.0) * 255)
    bgr = cv2.applyColorMap(np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_window(
    frames: list[Path],
    points: dict[int, list[tuple[float, float, float]]],
    start: int,
    end: int,
    output: Path,
    *,
    label_every: int = 5,
) -> Path:
    base = cv2.resize(_read(frames[(start + end) // 2 - 1]), None, fx=SCALE, fy=SCALE)
    canvas = (base * 0.35).astype(np.uint8)
    points = suppress_dense_cells(points, start, end)
    span = max(1, end - start)
    for frame in range(start, end + 1):
        for x, y, _area in points.get(frame, []):
            colour = _colour((frame - start) / span)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, colour, -1, cv2.LINE_AA)
        if frame % label_every == 0 and points.get(frame):
            x, y, _ = points[frame][0]
            cv2.putText(
                canvas,
                str(frame),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"frames {start}-{end}   blue=early  red=late",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output


def render_strip(frames: list[Path], centres: list[tuple[int, float, float]], output: Path) -> Path:
    """Native crops at 4x around a given centre, one panel per frame, left to right."""
    panels = []
    half = 96
    for frame, x, y in centres:
        image = _read(frames[frame - 1])
        cx, cy = int(x / SCALE), int(y / SCALE)
        left = int(np.clip(cx - half, 0, image.shape[1] - 2 * half))
        top = int(np.clip(cy - half, 0, image.shape[0] - 2 * half))
        crop = image[top : top + 2 * half, left : left + 2 * half]
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            crop, str(frame), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
        )
        panels.append(crop)
    canvas = cv2.hconcat(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=10)
    args = parser.parse_args()

    frames = sorted(args.frames.glob("f_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {args.frames}")
    points = candidates(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    windows = []
    start = 1
    while start < len(frames):
        end = min(len(frames), start + args.window - 1)
        path = args.output / f"trace_{start:04d}_{end:04d}.jpg"
        render_window(frames, points, start, end, path)
        windows.append({"start": start, "end": end, "path": str(path)})
        if end >= len(frames):
            break
        start = end - args.overlap + 1
    (args.output / "candidates.json").write_text(
        json.dumps(
            {
                "frames": len(frames),
                "detection_scale": SCALE,
                "windows": windows,
                "candidates": {str(k): v for k, v in points.items()},
            }
        )
    )
    print(json.dumps({"frames": len(frames), "windows": len(windows)}))


if __name__ == "__main__":
    main()


def follow_centres(
    points: dict[int, list[tuple[float, float, float]]],
    frame: int,
    x: float,
    y: float,
    *,
    radius: int = 6,
    gate: float = 110.0,
) -> list[tuple[int, float, float]]:
    """Centres for a +/-radius window, following the nearest candidate to a moving prediction.

    The nomination gives one (frame, x, y).  A fixed crop at that point loses a fast ball
    within two frames, so each step re-centres on the nearest surviving candidate inside
    `gate` pixels and carries the implied velocity forward when no candidate is available.
    No association model, no filter: just enough to keep the ball inside the crop.
    """
    centres = {frame: (float(x), float(y))}
    for direction in (1, -1):
        cx, cy = float(x), float(y)
        vx = vy = 0.0
        for step in range(1, radius + 1):
            index = frame + direction * step
            px, py = cx + vx, cy + vy
            best = None
            for candidate_x, candidate_y, _area in points.get(index, []):
                distance = ((candidate_x - px) ** 2 + (candidate_y - py) ** 2) ** 0.5
                if distance <= gate and (best is None or distance < best[0]):
                    best = (distance, candidate_x, candidate_y)
            if best is not None:
                vx, vy = best[1] - cx, best[2] - cy
                cx, cy = best[1], best[2]
            else:
                cx, cy = px, py
            centres[index] = (cx, cy)
    return [(f, centres[f][0], centres[f][1]) for f in sorted(centres)]


def render_refine(
    frames: list[Path],
    centres: list[tuple[int, float, float]],
    output: Path,
    *,
    crop_native: int = 128,
    zoom: int = 6,
    columns: int = 5,
) -> Path:
    """Native crops, nearest-neighbour magnified, laid out as a numbered storyboard.

    Native pixels and nearest-neighbour magnification are the project's measured evidence
    shape (`c96_z8_trail_s12_storyboard_full540_probabilities`); the point is that the
    decision is made on real sensor pixels rather than on a smoothed downscale.
    """
    half = crop_native // 2
    panels = []
    for frame, x, y in centres:
        if not 1 <= frame <= len(frames):
            continue
        image = _read(frames[frame - 1])
        cx, cy = int(round(x / SCALE)), int(round(y / SCALE))
        left = int(np.clip(cx - half, 0, image.shape[1] - crop_native))
        top = int(np.clip(cy - half, 0, image.shape[0] - crop_native))
        crop = image[top : top + crop_native, left : left + crop_native]
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (60, 60, 60), 2)
        cv2.putText(
            crop, str(frame), (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA
        )
        panels.append(crop)
    if not panels:
        raise ValueError("no panels rendered")
    rows = []
    for index in range(0, len(panels), columns):
        row = panels[index : index + columns]
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(cv2.hconcat(row))
    canvas = cv2.vconcat(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output
