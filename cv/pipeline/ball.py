"""Ball tracking on rally clips — classical motion differencing + trajectory linking.

The ball is the only small, fast, ballistic object on a near-static play camera. Per clip:

1. Motion mask per frame: ``min(|f_t - f_{t-1}|, |f_t - f_{t+1}|)`` (two-sided difference
   suppresses ghosting), threshold, connected components.
2. Candidate filter: small area, roundish, not inside a player box (pose CSV), not in the
   top crowd band.
3. Greedy trajectory linking with a velocity gate; keep segments >= 4 frames.

Outputs ``ball_track.csv`` (clip, frame, x, y, track_id) + optional debug overlays.
Contact events (direction changes near players) are derived in ``contacts.py``.

    .venv/bin/python cv/pipeline/ball.py --out data/processed/uso2025f \
        --frames-dir rally_frames_25 --pose player_pose_25.csv --debug 3
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

try:
    from cv.pipeline.frame_identity import frame_number_from_name
except ModuleNotFoundError:  # Preserve the documented standalone script entrypoint.
    from frame_identity import frame_number_from_name

AREA_MIN, AREA_MAX = 2, 150
V_MAX_PXS = 1500.0  # px/second velocity gate (960x540 analysis frames)
MIN_SPEED_PXS = 125.0  # px/second median linked speed
MIN_TRACK = 4


def load_court_gate(out_dir: str):
    """Per-point image-space court polygon (expanded) from the homographies npz."""
    import numpy as _np

    path = os.path.join(out_dir, "court_H_per_point.npz")
    if not os.path.exists(path):
        return {}
    d = _np.load(path)
    gates = {}
    quad_court = _np.float32([[-2, -5], [12.97, -5], [12.97, 28.77], [-2, 28.77]])  # +margins
    for pt, H in zip(d["pts"].tolist(), d["H"]):
        Hi = _np.linalg.inv(H)
        img_quad = cv2.perspectiveTransform(quad_court.reshape(1, -1, 2), Hi)[0]
        gates[pt] = img_quad.astype(_np.float32)
    return gates


def load_pose_boxes(out_dir: str, pose_name: str, pose_fps: float = 25.0, track_fps: float = 25.0):
    """Boxes keyed by (clip, frame-name at track_fps); pose may be at a lower fps —
    each track frame maps to the nearest pose frame."""
    raw = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(out_dir, pose_name), newline="")):
        raw[(r["clip"], frame_number_from_name(r["frame"]))].append(
            (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
        )
    if abs(pose_fps - track_fps) < 1e-6:
        return {(c, f"f_{k:04d}.jpg"): v for (c, k), v in raw.items()}
    boxes = {}
    per_clip = defaultdict(dict)
    for (c, k), v in raw.items():
        per_clip[c][k] = v
    for c, d in per_clip.items():
        keys = sorted(d)
        # map every plausible track-frame index to nearest pose frame
        max_track = int(max(keys) * track_fps / pose_fps) + 3
        for tk in range(1, max_track + 1):
            pk = min(keys, key=lambda q: abs(q - 1 - (tk - 1) * pose_fps / track_fps))
            boxes[(c, f"f_{tk:04d}.jpg")] = d[pk]
    return boxes


def candidates(prev, cur, nxt, player_boxes, gate=None):
    d1 = cv2.absdiff(cur, prev)
    d2 = cv2.absdiff(cur, nxt)
    motion = cv2.min(d1, d2)
    _, m = cv2.threshold(cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY), 18, 255, cv2.THRESH_BINARY)
    m[: m.shape[0] // 5] = 0  # crowd band
    m[int(m.shape[0] * 0.80) :, : int(m.shape[1] * 0.40)] = 0  # scoreboard/ticker block
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (AREA_MIN <= a <= AREA_MAX):
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) > 4 * max(1, min(w, h)):  # elongated streaks ok-ish at 25fps; cap 4:1
            continue
        x, y = cent[i]
        inside = any(
            bx0 - 6 <= x <= bx1 + 6 and by0 - 6 <= y <= by1 + 6
            for bx0, by0, bx1, by1 in player_boxes
        )
        if inside:
            continue
        if gate is not None and cv2.pointPolygonTest(gate, (float(x), float(y)), False) < 0:
            continue
        out.append((float(x), float(y), int(a)))
    return out


def link(
    dets_per_frame: list[tuple[int, list[tuple]]],
    fps: float,
    min_track: int = MIN_TRACK,
    min_speed_pxs: float = MIN_SPEED_PXS,
    min_straightness: float = 0.5,
):
    """Greedy nearest-neighbor linking using original frame indices and per-second gates.

    The classical motion detector uses speed and straightness rejection. Neural detectors have
    already learned appearance, so their decoder can disable those filters while retaining the
    same temporal association and track identifiers.
    """
    tracks, active = [], []  # active: [ (track:list[(fi,x,y)], vx, vy) ]
    for fi, dets in dets_per_frame:
        used = set()
        nxt_active = []
        for tr, vx, vy in active:
            lfi, lx, ly = tr[-1]
            gap = fi - lfi
            if gap > 3:
                tracks.append(tr)
                continue
            px, py = lx + vx * gap, ly + vy * gap
            best, best_d = None, V_MAX_PXS / fps * gap + 20
            for j, (x, y, a) in enumerate(dets):
                if j in used:
                    continue
                d = np.hypot(x - px, y - py)
                if d < best_d:
                    best, best_d = j, d
            if best is None:
                nxt_active.append((tr, vx, vy))
                continue
            x, y, a = dets[best]
            used.add(best)
            nvx, nvy = (x - lx) / gap, (y - ly) / gap
            tr.append((fi, x, y))
            nxt_active.append((tr, nvx, nvy))
        for j, (x, y, a) in enumerate(dets):
            if j not in used:
                nxt_active.append(([(fi, x, y)], 0.0, 0.0))
        active = nxt_active
    tracks.extend(tr for tr, _, _ in active)
    good = []
    for t in tracks:
        if len(t) < min_track:
            continue
        xs = np.array([(x, y) for _, x, y in t])
        steps = np.linalg.norm(np.diff(xs, axis=0), axis=1)
        path = float(steps.sum())
        disp = float(np.linalg.norm(xs[-1] - xs[0]))
        frame_steps = np.diff(np.array([fi for fi, _, _ in t], dtype=float))
        speed = float(np.median(steps / frame_steps) * fps) if len(steps) else 0.0
        straight_enough = min_straightness <= 0 or (path > 0 and disp / path >= min_straightness)
        if speed >= min_speed_pxs and straight_enough:
            good.append(t)
    return good


def process_clip(clip_dir: str, clip: str, pose_boxes, gate=None, fps: float = 25.0):
    frames = sorted(glob.glob(os.path.join(clip_dir, "f_*.jpg")), key=frame_number_from_name)
    if len(frames) < 5:
        return []
    # drop cut/closeup frames (MAD outliers vs the clip's median signature) — crowd pans
    # otherwise produce long, plausible-looking noise tracks
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shots_from_video as _sv

    keep = _sv.play_frames(clip_dir, [os.path.basename(f) for f in frames])
    frames = [f for f in frames if os.path.basename(f) in keep]
    if len(frames) < 5:
        return []
    imgs = [cv2.imread(f) for f in frames]
    frame_ids = [frame_number_from_name(f) for f in frames]
    dets = [(frame_ids[0], [])]
    for i in range(1, len(imgs) - 1):
        consecutive = frame_ids[i] - frame_ids[i - 1] == frame_ids[i + 1] - frame_ids[i] == 1
        if not consecutive or imgs[i - 1] is None or imgs[i] is None or imgs[i + 1] is None:
            dets.append((frame_ids[i], []))
            continue
        pb = []
        for off in (-2, -1, 0, 1, 2):
            j = min(max(i + off, 0), len(frames) - 1)
            pb.extend(pose_boxes.get((clip, os.path.basename(frames[j])), []))
        dets.append((frame_ids[i], candidates(imgs[i - 1], imgs[i], imgs[i + 1], pb, gate)))
    dets.append((frame_ids[-1], []))
    frame_path = dict(zip(frame_ids, frames))
    rows = []
    for tid, tr in enumerate(link(dets, fps)):
        for fi, x, y in tr:
            rows.append((clip, os.path.basename(frame_path[fi]), round(x, 1), round(y, 1), tid))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default="rally_frames_25")
    ap.add_argument("--pose", default="player_pose_25.csv")
    ap.add_argument("--debug", type=int, default=0, help="write overlay JPEGs for N clips")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--fps", type=float, default=25.0, help="frames-dir fps")
    ap.add_argument("--pose-fps", type=float, default=25.0)
    ap.add_argument("--output", default="ball_track.csv")
    args = ap.parse_args()
    from run_manifest import StageRun

    stage_run = StageRun(args.out, "ball_tracking", args)

    pose_boxes = load_pose_boxes(args.out, args.pose, args.pose_fps, args.fps)
    gates = load_court_gate(args.out)
    clips = sorted(glob.glob(os.path.join(args.out, args.frames_dir, "pt*")))
    if args.max_clips:
        clips = clips[: args.max_clips]
    all_rows = []
    for ci, clip_dir in enumerate(clips):
        clip = os.path.basename(clip_dir)
        rows = process_clip(clip_dir, clip, pose_boxes, gates.get(int(clip[2:])), args.fps)
        all_rows.extend(rows)
        if ci % 25 == 0:
            print(f"{ci}/{len(clips)} clips, {len(all_rows)} track points", flush=True)
        if args.debug and ci < args.debug and rows:
            debug_frames = sorted(
                glob.glob(os.path.join(clip_dir, "f_*.jpg")), key=frame_number_from_name
            )
            img = cv2.imread(debug_frames[len(debug_frames) // 2])
            by_track = defaultdict(list)
            for _, fr, x, y, tid in rows:
                by_track[tid].append((x, y))
            for tid, pts in by_track.items():
                col = tuple(int(c) for c in np.random.default_rng(tid).integers(80, 255, 3))
                for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.imwrite(os.path.join(args.out, f"ball_debug_{clip}.jpg"), img)
    out_csv = os.path.join(args.out, args.output)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame", "x", "y", "track_id"])
        w.writerows(all_rows)
    print(f"{len(all_rows)} track points, {len(clips)} clips -> {out_csv}")
    stage_run.finish(outputs={"track_points": len(all_rows), "clips": len(clips), "csv": out_csv})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
