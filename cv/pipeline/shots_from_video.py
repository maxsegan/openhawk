"""First end-to-end video->tokens emitter (no ball tracking yet) + honest scoring.

Per aligned point: filter pose detections to the two on-court players (foot point through
the per-point homography), split near/far, detect swings as wrist-speed peaks, type them
fh/bh from wrist-side geometry (both players right-handed here; near player faces away from
camera so sides flip), and emit a ChartShot stream:

    [serve by the baseline-center player] [shot ...]   (direction/depth/outcome not emitted)

Then score against the MCP chart with cv/validation/scoring.py — detection P/R/F1 and
shot-type accuracy on matched shots. This is the honest floor for the supervised model to
beat; direction/depth need ball tracking (queued).

    .venv/bin/python cv/pipeline/shots_from_video.py --out data/processed/rg2025f \
        --match 20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "charting"))

from scoring import ChartShot, score  # noqa: E402
from ground_truth import ground_truth_stream  # noqa: E402

import cv2  # noqa: E402  (perspectiveTransform)

COURT_W, COURT_L = 10.97, 23.77
NET_Y = COURT_L / 2
FPS = 8.0                      # players.py extraction rate (override --fps)
SWING_MIN_GAP = 0.55           # s between swings of the same player
SWING_SPEED_PXS = 500.0        # wrist speed threshold in px/SECOND (540p); F1-swept on RG


def load_pose(out_dir: str, name: str = "player_pose.csv"):
    per_clip = defaultdict(list)
    with open(os.path.join(out_dir, name), newline="") as f:
        for r in csv.DictReader(f):
            per_clip[r["clip"]].append(r)
    return per_clip


def load_H(out_dir: str):
    d = np.load(os.path.join(out_dir, "court_H_per_point.npz"))
    return dict(zip(d["pts"].tolist(), d["H"]))


def play_frames(clip_dir: str, frames: list[str]) -> set[str]:
    """Frames belonging to the play camera within a rally clip.

    The clip is mostly play-camera; cuts/replays/closeups deviate hard from the clip's
    median signature. Returns the frame names within 3 MADs of the median."""
    sigs = {}
    for fr in frames:
        img = cv2.imread(os.path.join(clip_dir, fr))
        if img is None:
            continue
        sigs[fr] = cv2.resize(img, (16, 9)).astype(np.float32).ravel()
    if len(sigs) < 5:
        return set(frames)
    arr = np.stack(list(sigs.values()))
    med = np.median(arr, axis=0)
    dist = np.linalg.norm(arr - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
    keep = dist <= np.median(dist) + 3 * 1.4826 * mad
    kept = {fr for fr, k in zip(sigs.keys(), keep) if k}
    return kept if len(kept) >= 5 else set(frames)


def court_xy(H, x_img, y_img):
    p = cv2.perspectiveTransform(np.float32([[[x_img, y_img]]]), H)[0, 0]
    return float(p[0]), float(p[1])


def track_players(rows, H):
    """frame -> {'near': det, 'far': det} keeping the two detections on court."""
    # scale: pose ran on 960x540 frames (players.py extraction size)
    by_frame = defaultdict(list)
    for r in rows:
        x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
        foot = ((x0 + x1) / 2, y1)
        cx, cy = court_xy(H, *foot)
        if -2.5 <= cx <= COURT_W + 2.5 and -4.0 <= cy <= COURT_L + 4.0:
            by_frame[r["frame"]].append({**r, "cx": cx, "cy": cy})
    out = {}
    for fr, dets in by_frame.items():
        near = [d for d in dets if d["cy"] < NET_Y]
        far = [d for d in dets if d["cy"] >= NET_Y]
        pick = {}
        if near:
            pick["near"] = max(near, key=lambda d: float(d["conf"]))
        if far:
            pick["far"] = max(far, key=lambda d: float(d["conf"]))
        out[fr] = pick
    return out


def wrist_speeds(frames_sorted, tracks, side):
    """Per-frame max wrist speed (px/frame) for the near/far player."""
    speeds, prev = [], None
    for fr in frames_sorted:
        det = tracks.get(fr, {}).get(side)
        if det is None:
            speeds.append(0.0)
            prev = None
            continue
        lw = (float(det["lw_x"]), float(det["lw_y"]))
        rw = (float(det["rw_x"]), float(det["rw_y"]))
        if prev is None:
            speeds.append(0.0)
        else:
            plw, prw = prev
            v = max(np.hypot(lw[0] - plw[0], lw[1] - plw[1]),
                    np.hypot(rw[0] - prw[0], rw[1] - prw[1]))
            speeds.append(float(v))
        prev = (lw, rw)
    return np.array(speeds)


def swing_peaks(speeds, min_gap_frames):
    peaks, last = [], -10 * min_gap_frames
    for i in range(1, len(speeds) - 1):
        thr = SWING_SPEED_PXS / FPS
        if (speeds[i] >= thr and speeds[i] >= speeds[i - 1]
                and speeds[i] >= speeds[i + 1] and i - last >= min_gap_frames):
            peaks.append(i)
            last = i
    return peaks


def fh_bh(det, side):
    """Right-handed heuristic: active wrist right of body center = forehand for the far
    player (faces camera); flipped for the near player (faces away)."""
    cx_body = (float(det["x0"]) + float(det["x1"])) / 2
    # active wrist = the faster-moving one is unknown here; use the wrist farther from body
    lw, rw = float(det["lw_x"]), float(det["rw_x"])
    wrist = lw if abs(lw - cx_body) > abs(rw - cx_body) else rw
    right_of_body = wrist > cx_body
    if side == "far":
        return "forehand" if right_of_body else "backhand"
    return "forehand" if not right_of_body else "backhand"


def emit_point(rows, H, clip_dir=None):
    frames = sorted({r["frame"] for r in rows})
    if clip_dir:
        keep = play_frames(clip_dir, frames)
        frames = [f for f in frames if f in keep]
        rows = [r for r in rows if r["frame"] in keep]
    tracks = track_players(rows, H)
    events = []  # (frame_idx, side, det)
    for side in ("near", "far"):
        sp = wrist_speeds(frames, tracks, side)
        for i in swing_peaks(sp, int(SWING_MIN_GAP * FPS)):
            det = tracks.get(frames[i], {}).get(side)
            if det is not None:
                events.append((i, side, det))
    events.sort()
    if not events:
        return []
    shots = [ChartShot("serve", shot_type="serve", role="server")]
    for k, (i, side, det) in enumerate(events[1:]):  # first swing ~ the serve itself
        shots.append(ChartShot("shot", fh_bh(det, side), None, None,
                               "in_play", None))
    return shots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--pose", default="player_pose.csv")
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()
    global FPS
    FPS = args.fps
    pose = load_pose(args.out, args.pose)
    Hs = load_H(args.out)
    # ground truth per point (ordered), restricted to points we have clips for
    gt_stream, _ = ground_truth_stream(args.match)
    # split gt stream back into per-point lists
    gt_points, cur = [], []
    for s in gt_stream:
        if s.kind == "<point>":
            gt_points.append(cur)
            cur = []
        else:
            cur.append(s)
    gt_points.append(cur)

    pred_all, gt_all = [], []
    n_scored = 0
    for clip, rows in sorted(pose.items()):
        pt = int(clip[2:])
        if pt not in Hs or pt - 1 >= len(gt_points):
            continue
        pred = emit_point(rows, Hs[pt])
        if not pred:
            continue
        if pred_all:
            pred_all.append(ChartShot("<point>"))
            gt_all.append(ChartShot("<point>"))
        pred_all.extend(pred)
        gt_all.extend(gt_points[pt - 1])
        n_scored += 1

    s = score(pred_all, gt_all)
    print(f"points scored: {n_scored}")
    print("END-TO-END (pose-swing emitter, no ball):")
    print("  " + s.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
