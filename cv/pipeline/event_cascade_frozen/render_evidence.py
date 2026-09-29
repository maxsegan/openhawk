"""Render fair native crops for the 278 held-back nominations.

Clean crops first. The trail presentation adds a second copy with a faded
orange-before / cyan-after track, and leaves the nominated frame clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .fairlib import (  # noqa: E402
    COLUMNS, CROP_NATIVE, RADIUS, SCRIPTS, ZOOM, art_root, frames_dir,
    nominations, track_csv,
)

from . import trace as tr  # noqa: E402
from . import trace_from_track as tft  # noqa: E402

# Owner portal colours: orange #ff9d42 before, cyan #55d8ff after. OpenCV is BGR.
ORANGE = (66, 157, 255)
CYAN = (255, 216, 85)


def render_panel(
    image: np.ndarray,
    center_xy: tuple[float, float],
    frame: int,
    *,
    trail: bool,
    centres_native: list[tuple[int, float, float]],
    target_frame: int,
    crop_native: int = CROP_NATIVE,
    zoom: int = ZOOM,
) -> np.ndarray:
    """One numbered crop. ``trail`` draws other-frame track dots, never on ``target_frame``."""
    height, width = image.shape[:2]
    half = crop_native // 2
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    left = int(np.clip(cx - half, 0, max(0, width - crop_native)))
    top = int(np.clip(cy - half, 0, max(0, height - crop_native)))
    crop = image[top:top + crop_native, left:left + crop_native]
    if crop.shape[0] != crop_native or crop.shape[1] != crop_native:
        raise ValueError(f"frame {frame} crop {crop.shape} from {width}x{height}")
    panel = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    if trail and frame != target_frame:
        for other, ox, oy in centres_native:
            if other == frame:
                continue
            px = int(round((ox - left) * zoom))
            py = int(round((oy - top) * zoom))
            if not (0 <= px < panel.shape[1] and 0 <= py < panel.shape[0]):
                continue
            dist = abs(other - frame)
            color = ORANGE if other < frame else CYAN
            radius = max(4, 14 - dist)
            alpha = max(0.18, 0.75 - 0.045 * dist)
            stamp = panel.copy()
            cv2.circle(stamp, (px, py), radius, color, -1, cv2.LINE_8)
            mask = np.zeros(panel.shape[:2], np.uint8)
            cv2.circle(mask, (px, py), radius, 255, -1, cv2.LINE_8)
            blended = (alpha * stamp + (1.0 - alpha) * panel).astype(np.uint8)
            panel[mask > 0] = blended[mask > 0]
    # Frame index sits in the corner, off the ball.
    cv2.rectangle(panel, (0, 0), (168, 52), (0, 0, 0), -1)
    cv2.putText(
        panel, str(frame), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (40, 40, 40), 2)
    return panel


def _centres(points: dict, frame: int) -> list[tuple[int, float, float]]:
    if points.get(frame):
        x, y, _area = points[frame][0]
    else:
        # Carry the nearest tracked frame rather than inventing a court centre.
        known = [f for f in points if points[f]]
        if not known:
            raise RuntimeError(f"no track points near frame {frame}")
        nearest = min(known, key=lambda f: abs(f - frame))
        x, y, _area = points[nearest][0]
    centres = tr.follow_centres(points, frame, x, y, radius=RADIUS)
    # follow_centres returns detection-space (960x540) coordinates. Native is / SCALE.
    native = []
    for index, dx, dy in centres:
        native.append((index, dx / tr.SCALE, dy / tr.SCALE))
    return native


def _read_window(folder: Path, frames: list[int]) -> dict[int, np.ndarray]:
    out = {}
    for frame in frames:
        path = folder / f"f_{frame:04d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        out[frame] = image
    return out


def _stack(panels: list[np.ndarray], columns: int) -> np.ndarray:
    rows = []
    blank = np.zeros_like(panels[0])
    for index in range(0, len(panels), columns):
        row = panels[index:index + columns]
        while len(row) < columns:
            row.append(blank)
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def _write(path: Path, image: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "sha256": digest,
    }


def render_nomination(
    nom: dict,
    points: dict,
    folder: Path,
    out_dir: Path,
    *,
    layout: str,
    frames_per_image: int,
) -> dict:
    native = _centres(points, nom["nominated_frame"])
    native = [row for row in native if (folder / f"f_{row[0]:04d}.jpg").is_file()]
    if not native:
        raise RuntimeError(f"{nom['clip']} {nom['id']} has no frames in ±{RADIUS}")
    images = _read_window(folder, [row[0] for row in native])
    target = nom["nominated_frame"]

    def panels(trail: bool) -> list[tuple[int, np.ndarray]]:
        built = []
        for frame, (x, y) in ((row[0], (row[1], row[2])) for row in native):
            built.append((frame, render_panel(
                images[frame], (x, y), frame,
                trail=trail, centres_native=native, target_frame=target,
            )))
        return built

    record = {
        "clip": nom["clip"],
        "id": nom["id"],
        "nominated_frame": nom["nominated_frame"],
        "nominated_type": nom["nominated_type"],
        "frames": [row[0] for row in native],
        "crop_native": CROP_NATIVE,
        "zoom": ZOOM,
        "layout": layout,
        "frames_per_image": frames_per_image if layout == "several" else len(native),
        "images": [],
    }
    clean = panels(False)
    trail = panels(True)
    # The nominated frame's trail copy is the clean crop. Keep the file anyway so
    # the second presentation has the same frame order, and check the bytes match.
    kinds = (("clean", clean), ("trail", trail))
    for kind, built in kinds:
        if layout == "one":
            canvas = _stack([panel for _frame, panel in built], COLUMNS)
            meta = _write(out_dir / kind / f"{nom['id']}.jpg", canvas)
            meta["kind"] = kind
            meta["frames"] = [frame for frame, _panel in built]
            record["images"].append(meta)
        else:
            group = max(1, frames_per_image)
            for start in range(0, len(built), group):
                chunk = built[start:start + group]
                if group == 1:
                    image = chunk[0][1]
                else:
                    image = _stack([panel for _frame, panel in chunk], len(chunk))
                frame0 = chunk[0][0]
                frame1 = chunk[-1][0]
                meta = _write(out_dir / kind / f"{nom['id']}_f{frame0:04d}_{frame1:04d}.jpg", image)
                meta["kind"] = kind
                meta["frames"] = [frame for frame, _panel in chunk]
                record["images"].append(meta)
    return record


def render_clips(clips: list[str], layout: str, frames_per_image: int) -> Path:
    rows = nominations(clips)
    by_clip: dict[str, list] = {}
    for row in rows:
        by_clip.setdefault(row["clip"], []).append(row)
    root = art_root() / "evidence" / layout / f"per{frames_per_image if layout == 'several' else 'grid'}"
    for clip, noms in by_clip.items():
        match, point = clip.split("__", 1)
        folder = frames_dir(match, point)
        points = tft.load_track(track_csv(match), point)
        out = root / clip
        manifest_path = out / "manifest.json"
        done = {}
        if manifest_path.is_file():
            done = {row["id"]: row for row in json.loads(manifest_path.read_text())["nominations"]}
        built = []
        for nom in noms:
            if nom["id"] in done and done[nom["id"]].get("images"):
                built.append(done[nom["id"]])
                continue
            built.append(render_nomination(nom, points, folder, out, layout=layout,
                                           frames_per_image=frames_per_image))
            print(f"rendered {clip} {nom['id']} frames={len(built[-1]['frames'])}", flush=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "clip": clip, "layout": layout, "frames_per_image": frames_per_image,
            "crop_native": CROP_NATIVE, "zoom": ZOOM, "radius": RADIUS,
            "nominations": built,
        }, indent=1))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--layout", choices=("several", "one"), required=True)
    parser.add_argument("--frames-per-image", type=int, default=1)
    args = parser.parse_args()
    from fairlib import SLICE_CLIPS, load_truth
    clips = args.clips or list(SLICE_CLIPS)
    if clips == ["all"]:
        clips = sorted(load_truth())
    path = render_clips(clips, args.layout, args.frames_per_image)
    print(path)


if __name__ == "__main__":
    main()
