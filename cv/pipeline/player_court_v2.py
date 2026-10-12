"""Player association v2 — REPLAY-BUG fix 1 (far-player dot jumping to non-players).

The replay renderer's fallback (`largest_box_per_side`, mirroring contacts_v2.load_boxes)
picks the largest-area person per court half inside the +/-2.5 m x / +/-8 m baseline
margin. A 12-frame overlay audit on rg2025f pt0092 shows that selection flip-flopping
between the actual far player (foot point x ~ 8-10 m, y ~ 26-28 m) and closer-to-camera
non-players inside the margin — the far-left line judge (x ~ -2.0, y ~ 22) and a ball kid
crouched by the net post (x ~ -2.3, y ~ 12.4) — because pixel area at those depths is
comparable and the pose export is near-player-biased (pt0092: 1819 near rows, 0 far).

v2 replaces per-frame largest-area with per-side track selection:

1. Candidates are gated exactly like v1 (court-margin box, near/far split at NET_Y).
2. Tracklets are grown per side with a max-velocity gate (V_MAX 10 m/s in court frame,
   gaps up to GAP_MAX_S coasted) — temporal consistency.
3. Tracklets are scored with a court-position prior: the player plays from inside the
   court's lateral extent (line judges / ball kids hug positions ~2 m outside the
   sidelines and are static); movement adds evidence, lateral/depth violations subtract.
4. Player-eligible tracklets (score > 0, lateral violation < LATERAL_MAX) are accepted
   greedily by score, without frame overlap, and adjacent accepted tracklets must be
   linkable at <= V_MAX across the gap — identity persistence. Frames with no accepted
   tracklet emit nothing (honest absence beats a wrong dot).

Writes a NEW versioned artifact next to the frozen input (never modifies it):
    <boxes stem minus _v1>_v2.csv  with columns
    clip,frame,side,x0,y0,x1,y1,conf,court_x,court_y,track_id

Usage:
  .venv/bin/python cv/pipeline/player_court_v2.py --match rg2025f
  .venv/bin/python cv/pipeline/player_court_v2.py --match uso2025f
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from cv.pipeline.camera_cal import COURT_L, COURT_W, NET_Y

COURT_X_MARGIN_M = 2.5  # v1 gate, unchanged
BASELINE_MARGIN_M = 8.0  # v1 gate, unchanged
V_MAX_MS = 10.0  # max plausible player speed for track continuity
GAP_MAX_S = 0.6  # tracklet coasts unmatched this long before closing
LINK_SLACK_M = 0.35  # detection jitter allowance on the per-frame gate
LATERAL_MAX_M = 1.0  # tracklets living further outside [0, COURT_W] are non-players
DEPTH_SLACK_M = 7.0  # players stand up to ~7 m behind their baseline
ADJ_GAP_MAX_S = 2.0  # adjacent accepted tracklets closer than this must be linkable

MATCHES = {
    "rg2025f": dict(boxes="player_boxes_50_full_v1.csv", fps=50.0),
    "uso2025f": dict(boxes="player_boxes_59.9401_full_v1.csv", fps=59.94005994005994),
    # wim2025f clips are 60s CHUNKS of the whole broadcast (full_frames.py), not rally
    # windows; court_H_per_point.npz is chunk-keyed to match. Non-play chunks have no H
    # and are skipped (honest absence).
    "wim2025f": dict(boxes="player_boxes_25_fullmatch_v1.csv", fps=25.0),
    # Window-map v2 shadow substrates (2026-07-17 WINDOW-V2 RERUN): same box artifact
    # names, resolved under data/processed/<match>_w2/ (map there IS point_video_map_v2).
    "rg2025f_w2": dict(boxes="player_boxes_50_full_v1.csv", fps=50.0),
    "uso2025f_w2": dict(boxes="player_boxes_59.9401_full_v1.csv", fps=59.94005994005994),
}


@dataclass
class Tracklet:
    tid: int
    frames: list[int] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)

    @property
    def f0(self) -> int:
        return self.frames[0]

    @property
    def fe(self) -> int:
        return self.frames[-1]

    def last_xy(self) -> np.ndarray:
        return np.asarray(self.xy[-1])


def frame_num(raw: str) -> int:
    raw = raw.strip()
    if raw.startswith("f_"):
        return int(raw[2:6])
    return int(float(raw))


def lateral_violation(cx: float) -> float:
    return max(0.0, -cx, cx - COURT_W)


def depth_violation(cy: float, side: str) -> float:
    if side == "far":
        return max(0.0, cy - (COURT_L + DEPTH_SLACK_M), (NET_Y + 0.5) - cy)
    return max(0.0, cy - (NET_Y - 0.5), -DEPTH_SLACK_M - cy)


def score_tracklet(t: Tracklet, side: str) -> float:
    xy = np.asarray(t.xy)
    cx_med, cy_med = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
    lv = lateral_violation(cx_med)
    dv = depth_violation(cy_med, side)
    return 0.5 * np.log1p(len(t.frames)) + 1.0 * np.log1p(path) - 3.0 * lv - 1.5 * dv


def build_tracklets(cands: dict[int, list[dict]], fps: float) -> list[Tracklet]:
    """Greedy nearest-neighbour linking in court frame under the max-velocity gate."""
    tracklets: list[Tracklet] = []
    active: list[Tracklet] = []
    next_id = 0
    gap_max_f = max(1, int(round(GAP_MAX_S * fps)))
    for f in sorted(cands):
        still_active = [t for t in active if f - t.fe <= gap_max_f]
        closed = [t for t in active if f - t.fe > gap_max_f]
        tracklets.extend(closed)
        active = still_active
        rows = cands[f]
        taken: set[int] = set()
        # match existing tracklets first (nearest candidate under the velocity gate)
        for t in sorted(active, key=lambda t: -len(t.frames)):
            dt_s = (f - t.fe) / fps
            gate = V_MAX_MS * dt_s + LINK_SLACK_M
            best_i, best_d = None, gate
            for i, row in enumerate(rows):
                if i in taken:
                    continue
                d = float(np.hypot(row["cx"] - t.last_xy()[0], row["cy"] - t.last_xy()[1]))
                if d <= best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                row = rows[best_i]
                taken.add(best_i)
                t.frames.append(f)
                t.rows.append(row)
                t.xy.append((row["cx"], row["cy"]))
        for i, row in enumerate(rows):
            if i in taken:
                continue
            t = Tracklet(tid=next_id)
            next_id += 1
            t.frames.append(f)
            t.rows.append(row)
            t.xy.append((row["cx"], row["cy"]))
            active.append(t)
    tracklets.extend(active)
    return tracklets


def select_player_tracklets(tracklets: list[Tracklet], side: str, fps: float) -> list[Tracklet]:
    scored = [(score_tracklet(t, side), t) for t in tracklets]
    eligible = [
        (s, t)
        for s, t in scored
        if s > 0 and lateral_violation(float(np.median(np.asarray(t.xy)[:, 0]))) < LATERAL_MAX_M
    ]
    accepted: list[Tracklet] = []
    for s, t in sorted(eligible, key=lambda item: -item[0]):
        covered = set()
        for a in accepted:
            covered.update(a.frames)
        if covered.intersection(t.frames):
            continue
        # identity persistence: a nearby accepted tracklet must be reachable at V_MAX
        ok = True
        for a in accepted:
            if t.f0 >= a.fe:
                gap_s = (t.f0 - a.fe) / fps
                d = float(np.hypot(*(np.asarray(t.xy[0]) - a.last_xy())))
            else:
                gap_s = (a.f0 - t.fe) / fps
                d = float(np.hypot(*(t.last_xy() - np.asarray(a.xy[0]))))
            if 0 <= gap_s <= ADJ_GAP_MAX_S and d > V_MAX_MS * max(gap_s, 1.0 / fps) + LINK_SLACK_M:
                ok = False
                break
        if ok:
            accepted.append(t)
    return accepted


def process_clip(rows: list[dict], H: np.ndarray, fps: float) -> list[dict]:
    pts = np.float32([[[(float(r["x0"]) + float(r["x1"])) / 2, float(r["y1"])]] for r in rows])
    court = cv2.perspectiveTransform(pts, H)[:, 0, :]
    cands: dict[str, dict[int, list[dict]]] = {"near": defaultdict(list), "far": defaultdict(list)}
    for r, (cx, cy) in zip(rows, court):
        cx, cy = float(cx), float(cy)
        if not (
            -COURT_X_MARGIN_M <= cx <= COURT_W + COURT_X_MARGIN_M
            and -BASELINE_MARGIN_M <= cy <= COURT_L + BASELINE_MARGIN_M
        ):
            continue
        side = "near" if cy < NET_Y else "far"
        cands[side][frame_num(r["frame"])].append({**r, "cx": cx, "cy": cy})
    out = []
    for side in ("near", "far"):
        tracklets = build_tracklets(cands[side], fps)
        for t in select_player_tracklets(tracklets, side, fps):
            for f, row in zip(t.frames, t.rows):
                out.append(
                    dict(
                        clip=row["clip"],
                        frame=row["frame"],
                        side=side,
                        x0=row["x0"],
                        y0=row["y0"],
                        x1=row["x1"],
                        y1=row["y1"],
                        conf=row["conf"],
                        court_x=round(row["cx"], 3),
                        court_y=round(row["cy"], 3),
                        track_id=t.tid,
                    )
                )
    out.sort(key=lambda r: (frame_num(r["frame"]), r["side"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, choices=sorted(MATCHES))
    parser.add_argument("--clips", default="", help="optional comma list, e.g. pt0092,pt0006")
    args = parser.parse_args()
    spec = MATCHES[args.match]
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo, "data", "processed", args.match)
    boxes_path = os.path.join(out_dir, spec["boxes"])
    out_path = os.path.join(out_dir, spec["boxes"].replace("_v1.csv", "_v2.csv"))
    assert out_path != boxes_path

    data = np.load(os.path.join(out_dir, "court_H_per_point.npz"))
    homographies = dict(zip(data["pts"].tolist(), data["H"]))

    per_clip: dict[str, list[dict]] = defaultdict(list)
    with open(boxes_path, newline="") as handle:
        for row in csv.DictReader(handle):
            per_clip[row["clip"]].append(row)
    only = set(filter(None, args.clips.split(",")))

    n_in = n_out = n_clips = 0
    fields = [
        "clip",
        "frame",
        "side",
        "x0",
        "y0",
        "x1",
        "y1",
        "conf",
        "court_x",
        "court_y",
        "track_id",
    ]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in sorted(per_clip):
            if only and clip not in only:
                continue
            H = homographies.get(int(clip[2:]))
            if H is None:
                continue
            rows = per_clip[clip]
            n_in += len(rows)
            out_rows = process_clip(rows, H, spec["fps"])
            for r in out_rows:
                writer.writerow(r)
            n_out += len(out_rows)
            n_clips += 1
    print(f"{args.match}: {n_clips} clips, {n_in} candidate boxes -> {n_out} player rows")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
