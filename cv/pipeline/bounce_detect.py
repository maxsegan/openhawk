"""Programmatic ball-bounce detector (Stage-2 rebuild, Q0b step 1).

WHY. A bounce is the ONE ball event with exact monocular 3D: the ball is on the ground
plane (z=0), so the v3 ground-plane homography turns its image pixel directly into court
metres. Every OTHER 3D quantity in the pipeline is a depth-ambiguous fit output; the current
fitted bounces are >1 m off because they are fit OUTPUTS, not detections. This module detects
bounce frames from the decoded 2D track's image-space signature and projects them through the
ground plane to get exact court positions -- hard anchors for the flight-first fitter.

SIGNATURE OF A BOUNCE (image space). The ball's WORLD vertical velocity reverses sharply
(down -> up) over ~1 frame while its horizontal velocity stays continuous. In image space
this is a downward->upward kink in y_img:
  - NEAR-court bounces: a large, obvious local MAXIMUM in y_img (ball plunges to the clay
    then rebounds; tens of px of prominence).
  - FAR-court bounces: the ball is also receding from the camera, so the vertical reversal
    is superimposed on a monotone y_img trend and shows up as a SUBTLE kink (a few px) --
    detected as a curvature/slope-break, not a clean local max.
An APEX (top of a flight arc) is a smooth local MINIMUM in y_img and is never a candidate.

EXCLUSIONS (what is NOT a ground bounce):
  - Racket contacts: also downward->upward kinks, but they sit inside/near a player box in
    IMAGE space (depth-robust) and usually reverse the horizontal direction. Excluded by
    expanded-box proximity.
  - Net collisions: kinks at the net plane where the ball rebounds back toward the side it
    came from. Demoted by net-plane proximity + horizontal reversal.

PROJECTION (image -> exact court). undistort the bounce pixel (per-frame k1 about
dist_center, v3), then intersect with the ground plane via inv(P[:, [0,1,3]]) -- the pinhole
ground homography self-consistent with the v3 P. The ball CENTER sits one ball radius
(~3.3 cm) above the clay at contact; that is negligible (< a few cm in court metres) versus
the >1 m fit errors this module replaces, and is NOT corrected here (noted).

OUTPUT. CSV rows: clip, frame (subframe-interpolated), img_x, img_y (540-space, the actual
decoded/distorted pixel), court_x, court_y (metres), confidence, kink_strength, plus flags.

This is a NEW module and adopts nothing; run it with a v3 camera npz + the decoded track.
Usage:
  .venv/bin/python cv/pipeline/bounce_detect.py --match rg2025f --clip pt0021 \
      --camera-file <v3 npz abs path> --out-csv <path>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from cv.pipeline import resolution as res
    from cv.pipeline.frame_identity import frame_number_from_name
except ModuleNotFoundError:  # Standalone legacy-compatible entrypoint.
    import resolution as res
    from frame_identity import frame_number_from_name

# Court geometry (metres; same frame the solver uses).
NET_Y = 11.885
BASELINE_NEAR, BASELINE_FAR = 0.0, 23.77
DOUBLES_X = (0.0, 10.97)
SINGLES_X = (1.37, 9.60)

# Net tape geometry (the WHITE TAPE along the top of the net; sags at centre, rises at posts).
# Same numbers as physics_knot_solver (kept local so this detector imports no solver).
NET_H_CENTER = 0.914  # tape height at the centre (m)
NET_H_POST = 1.07  # tape height at the singles/doubles posts (m)
NET_X_CENTER = 5.485  # court x of the net centre (m)
NET_HALF_WIDTH = 6.4  # centre-to-post span for the tape-height ramp (doubles + posts)


def net_tape_height(x: float) -> float:
    """Height (m) of the white tape at court-x: NET_H_CENTER at centre rising to NET_H_POST at posts."""
    frac = min(abs(float(x) - NET_X_CENTER) / NET_HALF_WIDTH, 1.0)
    return NET_H_CENTER + (NET_H_POST - NET_H_CENTER) * frac


def frame_num(raw: str) -> int:
    raw = raw.strip()
    return frame_number_from_name(raw) if raw.startswith("f_") else int(round(float(raw)))


# --------------------------------------------------------------------------- camera / ground
class GroundProjector:
    """Distortion-aware image->ground-plane projection from a v3 (or v1) camera npz.

    v3 files carry per-frame ``k1`` + ``dist_center``; a bounce pixel is undistorted first,
    then intersected with z=0 through the pinhole ground homography ``inv(P[:, [0,1,3]])``
    (self-consistent with the stored pinhole ``P``). v1/v2 files (no k1) reduce to a plain
    pinhole ground intersect.
    """

    def __init__(
        self,
        camera_file: str,
        clip: str,
        image_size: res.FrameSize | None = None,
        artifact_size: res.FrameSize | None = None,
    ):
        d = np.load(camera_file, allow_pickle=True)
        m = d["clips"] == clip
        if not m.any():
            raise ValueError("clip %s not in %s" % (clip, camera_file))
        frames = d["frames"][m].astype(int)
        projections = d["P"][m]
        if image_size is not None and artifact_size is not None:
            projections = [
                res.world_to_image_projection(P, image_size, artifact_size) for P in projections
            ]
        self.P = {int(f): P for f, P in zip(frames, projections)}
        self.has_k1 = "k1" in d
        if self.has_k1:
            scale = 1.0
            if image_size is not None and artifact_size is not None:
                scale = artifact_size.width / image_size.width
                if not np.isclose(
                    scale,
                    artifact_size.height / image_size.height,
                ):
                    raise ValueError("distortion scaling requires equal aspect ratio")
            self.k1 = {int(f): float(k) / (scale * scale) for f, k in zip(frames, d["k1"][m])}
            self.center = {
                int(f): np.asarray(c, float) * scale for f, c in zip(frames, d["dist_center"][m])
            }
        self.reliable = (
            {int(f): bool(r) for f, r in zip(frames, d["reliable"][m])} if "reliable" in d else {}
        )
        self.net_cord = {}
        self.net_cord_source = {}
        if "net_cord_xy" in d and "net_cord_valid" in d:
            cords = np.asarray(d["net_cord_xy"][m], dtype=float)
            valid = np.asarray(d["net_cord_valid"][m], dtype=bool)
            if image_size is not None and artifact_size is not None:
                cords = cords * np.asarray(
                    [
                        artifact_size.width / image_size.width,
                        artifact_size.height / image_size.height,
                    ]
                )
            sources = (
                d["net_cord_source"][m].astype(str)
                if "net_cord_source" in d
                else np.repeat("observed", len(frames))
            )
            self.net_cord = {
                int(frame): cord
                for frame, cord, is_valid in zip(frames, cords, valid, strict=True)
                if is_valid and np.isfinite(cord).all()
            }
            self.net_cord_source = {
                int(frame): str(source)
                for frame, source, is_valid in zip(frames, sources, valid, strict=True)
                if is_valid
            }
        self._fk = np.array(sorted(self.P))

    def _nearest(self, f: int) -> int:
        return int(self._fk[np.argmin(np.abs(self._fk - f))])

    def undistort(self, u: float, v: float, f: int) -> np.ndarray:
        """Distorted 540-space pixel -> undistorted 540-space pixel (fixed-point)."""
        if not self.has_k1:
            return np.array([u, v], float)
        k1 = self.k1.get(f) if f in self.k1 else self.k1[self._nearest(f)]
        c = self.center.get(f) if f in self.center else self.center[self._nearest(f)]
        target = np.array([u, v], float) - c
        d = target.copy()
        for _ in range(8):
            r2 = float((d * d).sum())
            d = target / (1.0 + k1 * r2)
        return c + d

    def to_court(self, u: float, v: float, f: int) -> tuple[float, float]:
        """Distorted 540-space pixel at frame f -> court (X, Y) metres on z=0."""
        fi = int(round(f))
        P = self.P.get(fi) if fi in self.P else self.P[self._nearest(fi)]
        pu = self.undistort(u, v, fi)
        Hp = P[:, [0, 1, 3]]  # columns for X, Y, homogeneous-1 (Z=0)
        w = np.linalg.solve(Hp, np.array([pu[0], pu[1], 1.0]))
        return float(w[0] / w[2]), float(w[1] / w[2])

    def is_reliable(self, f: int) -> bool:
        fi = int(round(f))
        if not self.reliable:
            return True
        return self.reliable.get(fi, self.reliable.get(self._nearest(fi), True))


def projector_for_artifact(
    camera_file: str,
    clip: str,
    artifact_file: str | os.PathLike,
) -> GroundProjector:
    manifest = res.read_coordinate_manifest(artifact_file)
    if manifest is None:
        return GroundProjector(camera_file, clip)
    image = manifest["image_size"]
    artifact = manifest["artifact_size"]
    return GroundProjector(
        camera_file,
        clip,
        image_size=res.FrameSize(int(image["width"]), int(image["height"])),
        artifact_size=res.FrameSize(
            int(artifact["width"]),
            int(artifact["height"]),
        ),
    )


# --------------------------------------------------------------------------- data loading
def load_track(ball_file: str, clip: str):
    """Return sorted arrays (frame, x_img, y_img, track_id) for the clip (540-space)."""
    rows = []
    with open(ball_file, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["clip"] != clip:
                continue
            rows.append(
                (frame_num(r["frame"]), float(r["x"]), float(r["y"]), int(r.get("track_id", -1)))
            )
    rows.sort()
    if not rows:
        return (np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0, int))
    a = np.array(rows, float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3].astype(int)


def load_boxes(box_file: str, clip: str):
    """{side -> {frame -> (x0,y0,x1,y1)}} in 540-space (same as the ball track)."""
    boxes = {"near": {}, "far": {}}
    if not box_file or not os.path.exists(box_file):
        return boxes
    with open(box_file, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("clip") != clip:
                continue
            side = r.get("side")
            if side not in boxes:
                continue
            boxes[side][frame_num(r["frame"])] = (
                float(r["x0"]),
                float(r["y0"]),
                float(r["x1"]),
                float(r["y1"]),
            )
    return boxes


def near_player_box(
    boxes, u: float, v: float, f: int, expand: float = 0.35, tol_f: int = 6
) -> str | None:
    """Which side's expanded box contains pixel (u,v) at ~frame f, if any (depth-robust)."""
    for side in ("near", "far"):
        cand = [g for g in boxes.get(side, {}) if abs(g - f) <= tol_f]
        if not cand:
            continue
        x0, y0, x1, y1 = boxes[side][min(cand, key=lambda q: abs(q - f))]
        dx, dy = (x1 - x0) * expand, (y1 - y0) * expand
        if x0 - dx <= u <= x1 + dx and y0 - dy <= v <= y1 + dy:
            return side
    return None


# ------------------------------------------------------------ net-tape image projection (RULE 1a)
def tape_line_ud(proj: "GroundProjector", f: int, nx: int = 80) -> np.ndarray:
    """Project the WHITE TAPE (top of the net, y=NET_Y, z=net_tape_height(x)) into the image at
    frame f, returned as an undistorted 540-space polyline [[u,v], ...] across x in [0, 10.97].
    The tape is a specific line in the image; the mesh BELOW it has its own (lower) trajectory and
    is deliberately excluded (a ball into the mesh dies -- handled by the post-kink test, RULE 1c)."""
    fi = int(round(f))
    if proj.net_cord:
        cord_frame = fi if fi in proj.net_cord else proj._nearest(fi)
        observed = proj.net_cord.get(cord_frame)
        if observed is not None:
            if proj.has_k1:
                return np.stack(
                    [
                        proj.undistort(float(point[0]), float(point[1]), cord_frame)
                        for point in observed
                    ]
                )
            return observed.copy()
    P = proj.P.get(fi) if fi in proj.P else proj.P[proj._nearest(fi)]
    pts = np.empty((nx, 2))
    for i, X in enumerate(np.linspace(DOUBLES_X[0], DOUBLES_X[1], nx)):
        w = P @ np.array([X, NET_Y, net_tape_height(X), 1.0])
        pts[i] = (w[0] / w[2], w[1] / w[2])
    return pts


def tape_pixel_dist(proj: "GroundProjector", u: float, v: float, f: int):
    """Min pixel distance (undistorted 540-space) from the ball pixel (u,v) to the projected tape
    line at frame f, and the signed vertical offset (ball above the tape reads negative). A ball that
    CLIPPED the tape sits within a few px of this line; a ground bounce projects tens of px away."""
    pu = proj.undistort(u, v, int(round(f)))
    line = tape_line_ud(proj, f)
    starts = line[:-1]
    vectors = line[1:] - starts
    lengths_squared = np.sum(vectors * vectors, axis=1)
    valid = lengths_squared > 1e-12
    if not valid.any():
        distance = float(np.hypot(*(line[0] - pu)))
        return distance, float(pu[1] - line[0, 1])
    starts = starts[valid]
    vectors = vectors[valid]
    lengths_squared = lengths_squared[valid]
    fractions = np.clip(
        np.sum((pu - starts) * vectors, axis=1) / lengths_squared,
        0.0,
        1.0,
    )
    closest = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(closest - pu, axis=1)
    nearest = int(np.argmin(distances))
    return float(distances[nearest]), float(pu[1] - closest[nearest, 1])


# net-vs-bounce disambiguation knobs (RULE 1; owner 2026-07-23). Calibrated on rg2025f where net
# events sit <=25 px from the projected tape and ground bounces >=41 px -- a clean gap.
NET_TAPE_BAND_PX = 33.0  # a kink whose ball is within this of the tape line -> net is a LIVE
# hypothesis (RULE 1a). Below the bounce floor (41 px), above the net
# ceiling (25 px): the tape band gates the flip so far bounces never enter.
NET_DEAD_SP = 5.0  # px/frame: post-kink image speed at/below this = the ball died at the net
NET_DEAD_FRAC = 0.55  # AND speed fell to <= this fraction of the incoming speed (speed collapse)
NET_BOUNCE_VX = 2.5  # px/frame: a TRUE bounce keeps at least this horizontal speed, same sign
NET_SERVE_LOOKBACK = 60  # frames: a candidate this soon after a serve strike is in the serve flight
# -> the hitter's side is the server's side (same-side authority, RULE 1b)


# Player-reach (pose-hand) gate constants — the pt0026 baseline leak: a racket contact at
# arm's reach just OUTSIDE the box (f149) read as a ground bounce. A real ground bounce sheds
# vertical speed (restitution < 1: |vy_out| < |vy_in|); a racket contact on a controlled ball
# ADDS vertical energy (|vy_out| >> |vy_in|). So a kink within a player's hand reach that
# leaves markedly faster upward than it arrived is a HIT, not a bounce.
SWING_REACH_FLOOR = 42.0  # px (540-space) minimum hand-reach radius
SWING_REACH_FRAC = 0.55  # fraction of the player's box height = hand-reach radius
ENERGY_GAIN_RATIO = 2.0  # |vy_after| must exceed this * |vy_before| to read as a strike
VY_OUT_MIN = 8.0  # and exceed this absolute upward image speed (px/frame)
NET_HIT_Y_TOL = 1.6  # a kink within this court-y of the net plane is a NET event


def player_reach(boxes, pose, u: float, v: float, f: int, tol_f: int = 3):
    """Nearest player whose HAND (pose wrist; box-bottom fallback) is within reach of the
    ball pixel (u,v) at ~frame f. Returns (side, wrist_dist_px, reach_radius_px) or None.
    Reach radius scales with the player's on-screen size so near/far are handled uniformly."""
    best = None
    for side in ("near", "far"):
        bcand = [g for g in boxes.get(side, {}) if abs(g - f) <= tol_f]
        if not bcand:
            continue
        x0, y0, x1, y1 = boxes[side][min(bcand, key=lambda q: abs(q - f))]
        R = max(SWING_REACH_FLOOR, SWING_REACH_FRAC * (y1 - y0))
        wd = None
        pc = [g for g in pose.get(side, {}) if abs(g - f) <= tol_f] if pose else []
        if pc:
            g = min(pc, key=lambda q: abs(q - f))
            for w in ("left_wrist", "right_wrist"):
                val = pose[side][g].get(w)
                if val and val[2] > 0.3:
                    dd = float(np.hypot(u - val[0], v - val[1]))
                    wd = dd if wd is None else min(wd, dd)
        if wd is None:
            wd = float(np.hypot(u - (x0 + x1) / 2.0, v - y1))
        if wd <= R and (best is None or wd < best[1]):
            best = (side, wd, R)
    return best


# --------------------------------------------------------------------------- run splitting
def split_runs(frames, xs, ys, gap_max: int = 3, jump_px: float = 55.0):
    """Contiguous arcs: split on a frame gap > gap_max OR a pixel jump > jump_px (the
    distractor-flicker that shatters the decoded track into wrong-object segments)."""
    runs, cur = [], [0]
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        jump = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        if gap > gap_max or jump > jump_px:
            runs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    runs.append(cur)
    return runs


def _smooth(a: np.ndarray, w: int = 3) -> np.ndarray:
    if len(a) < w:
        return a.astype(float)
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def _arc_len(y: np.ndarray, j: int, min_vy: float = 1.2) -> int:
    """Longest run of consistent vertical image motion (either descending-into or
    rising-away-from) that touches index j -- the ballistic-flight bracket. Real play
    bounces are bracketed by a long smooth arc (serve/rally shot); ballboy hand-bounces and
    dead-ball jitter are short. Direction-agnostic: a far bounce's incoming leg reads as
    rising (ball receding) while a near bounce's incoming reads as descending."""
    n = len(y)

    def run(step: int, sign: int) -> int:
        cnt, k = 0, j
        while 0 <= k + step < n:
            if sign * (y[k + step] - y[k]) >= min_vy:
                cnt += 1
                k += step
            else:
                break
        return cnt

    return max(run(+1, -1), run(+1, +1), run(-1, -1), run(-1, +1))


# --------------------------------------------------------------------------- detection core
def detect_in_run(
    idx, frames, xs, ys, proj, boxes, *, min_len=12, kink_min=3.0, arc_min=9, edge_trim=4
):
    """Detect bounce candidates within one contiguous run.

    A bounce is the sharp VERTICAL-DECELERATION spike: the ball's image-y velocity drops
    (kink = vy_before - vy_after > 0) as its world vertical velocity flips from down to up.
    This unifies near-court reversals (vy: large+ -> negative), kick-serve plateaus
    (vy: large+ -> ~0) and subtle far-court bounces (vy: ~0 -> negative) under one signal --
    an image-y LOCAL MAX misses the plateau and far cases. An APEX has kink < 0 (vy rises)
    and is never a candidate. Candidates are kink local-maxima above threshold, edge-trimmed
    (track-break-adjacent points are unreliable), and each carries a ballistic-arc length.
    """
    if len(idx) < min_len:
        return []
    fr = frames[idx].astype(float)
    x = xs[idx]
    y = ys[idx]
    ys_s = _smooth(y, 3)
    xs_s = _smooth(x, 3)
    n = len(idx)

    def kink_at(jj: int) -> float:
        sp = min(5, jj, n - 1 - jj)
        if sp < 3:
            return -1e9
        vyb = (ys_s[jj] - ys_s[jj - sp]) / sp
        vya = (ys_s[jj + sp] - ys_s[jj]) / sp
        return vyb - vya

    out = []
    for j in range(edge_trim, n - edge_trim):
        kink = kink_at(j)
        if kink < kink_min:
            continue
        if not (kink >= kink_at(j - 1) and kink >= kink_at(j + 1)):
            continue
        sp = min(5, j, n - 1 - j)
        vy_before = (ys_s[j] - ys_s[j - sp]) / sp
        vy_after = (ys_s[j + sp] - ys_s[j]) / sp
        vx_before = (xs_s[j] - xs_s[j - sp]) / sp
        vx_after = (xs_s[j + sp] - xs_s[j]) / sp
        horiz_reverse = vx_before * vx_after < 0 and min(abs(vx_before), abs(vx_after)) > 0.8
        arc = _arc_len(y, j)
        # subframe vertex of a parabola through (fr, y) near the kink.
        lo, hi = max(0, j - 2), min(n, j + 3)
        pf, py = fr[lo:hi], y[lo:hi]
        sub_f, sub_y = fr[j], y[j]
        if len(pf) >= 3 and np.ptp(pf) > 0:
            a2, b2, c2 = np.polyfit(pf - pf[0], py, 2)
            if a2 < 0:
                vtx = -b2 / (2 * a2) + pf[0]
                if pf[0] - 1 <= vtx <= pf[-1] + 1:
                    sub_f = float(vtx)
                    sub_y = float(np.polyval([a2, b2, c2], sub_f - pf[0]))
        sub_x = float(np.interp(sub_f, fr, x))
        cx, cy = proj.to_court(sub_x, sub_y, int(round(sub_f)))
        sp_before = float(np.hypot(vx_before, vy_before))
        sp_after = float(np.hypot(vx_after, vy_after))
        out.append(
            {
                "run_j": j,
                "frame": sub_f,
                "img_x": sub_x,
                "img_y": sub_y,
                "court_x": cx,
                "court_y": cy,
                "kink": float(kink),
                "prom": float(kink),
                "arc": int(arc),
                "vy_before": float(vy_before),
                "vy_after": float(vy_after),
                "vx_before": float(vx_before),
                "vx_after": float(vx_after),
                "sp_before": sp_before,
                "sp_after": sp_after,
                "horiz_reverse": bool(horiz_reverse),
                "far_court": bool(cy > NET_Y + 3.0),
            }
        )
    return out


def classify_and_score(cand, boxes, proj, arc_min=9, pose=None):
    """Tag each candidate (bounce / racket / net / off-court / short-arc) and score it.

    A candidate is a genuine GROUND BOUNCE only if it clears every exclusion:
      - not inside/near a player box (image space)  -> else racket_contact,
      - not within a player's HAND REACH with a vertical-energy GAIN (a strike, not a
        passive bounce)                              -> else racket_contact (swing gate),
      - not a net-plane event (a kink at the net plane) -> else net,
      - a plausible on-court ground position          -> else reject_offcourt,
      - bracketed by a real ballistic arc (>= arc_min frames) -> else reject_shortarc
        (kills ballboy hand-bounces and dead-ball jitter).
    """
    f = cand["frame"]
    u, v = cand["img_x"], cand["img_y"]
    cx, cy = cand["court_x"], cand["court_y"]
    side = near_player_box(boxes, u, v, int(round(f)))
    # net-plane event: a KINK whose ground projection is within NET_HIT_Y_TOL of the net line
    # is a ball that DIED at the net (a passing ball has no kink there), regardless of a
    # horizontal reversal (the pt0026 final barely-net dribble had none) -> net, not bounce.
    near_net = abs(cy - NET_Y) < NET_HIT_Y_TOL
    in_court_x = 0.3 <= cx <= DOUBLES_X[1] + 0.6
    in_court_y = BASELINE_NEAR - 1.5 <= cy <= BASELINE_FAR + 1.5
    plausible_ground = in_court_x and in_court_y
    long_arc = cand["arc"] >= arc_min
    # swing gate: within a player's hand reach AND the ball leaves markedly faster UPWARD than it
    # arrived = a racket strike ADDING vertical energy, not a passive bounce (fixes the pt0026
    # f149 leak where a hit at arm's reach just outside the box read as a baseline bounce).
    reach = player_reach(boxes, pose, u, v, int(round(f)))
    swing_gain = bool(
        reach is not None
        and cand["vy_after"] < 0
        and abs(cand["vy_after"]) > VY_OUT_MIN
        and abs(cand["vy_after"]) > ENERGY_GAIN_RATIO * max(abs(cand["vy_before"]), 1.0)
    )
    # a ball that FALLS fast into a plausible ground spot and DECELERATES (restitution < 1, no
    # energy added) is a GROUND BOUNCE even when it lands at a player's feet inside his box --
    # the return-bounce case the crude in-box rule used to hide (pt0026 f139, Alcaraz's return
    # bouncing at Sinner's feet). Only an energy-GAIN event there is a strike.
    falling_bounce = bool(
        cand["vy_before"] > 6.0
        and abs(cand["vy_after"]) < cand["vy_before"]
        and plausible_ground
        and long_arc
        and not near_net
    )

    flags = []
    if side is not None:
        flags.append("in_%s_box" % side)
    if near_net:
        flags.append("net_plane")
    if swing_gain:
        flags.append("swing_energy_gain_%s" % reach[0])
    if falling_bounce and side is not None:
        flags.append("bounce_at_feet")
    if not plausible_ground:
        flags.append("court_implausible")
    if not long_arc:
        flags.append("short_arc")

    if swing_gain:
        verdict = "racket_contact"  # energy-gain strike (in box or at reach)
    elif near_net and side is None:  # a real net event is away from players
        verdict = "net"  # (a toss ball high up also projects near
        # the net line -- exclude it via the box)
    elif side is not None and not falling_bounce:
        verdict = "racket_contact"  # in-box, not a falling ground bounce
    elif not plausible_ground:
        verdict = "reject_offcourt"
    elif not long_arc:
        verdict = "reject_shortarc"
    else:
        verdict = "bounce"  # incl. a ground bounce at a player's feet

    # confidence: kink strength (down-weighted for the intrinsically subtle far-court kink)
    # + arc support + camera reliability; penalised near the net.
    kink_ev = np.clip(cand["kink"] / 12.0, 0.0, 1.0)
    arc_ev = np.clip(cand["arc"] / 20.0, 0.0, 1.0)
    conf = float(np.clip(0.2 + 0.55 * kink_ev + 0.25 * arc_ev, 0.2, 0.95))
    if cand["far_court"] and cand["kink"] < 6:
        conf = float(min(conf, 0.55))  # subtle far kink: cap confidence
    if near_net:
        conf *= 0.6
    if not proj.is_reliable(int(round(f))):
        conf *= 0.5
    cand.update(
        {
            "verdict": verdict,
            "flags": flags,
            "confidence": round(conf, 3),
            "kink_strength": round(cand["kink"], 3),
            "box_side": side,
        }
    )
    return cand


def _serve_side_for(f: float, serves):
    """The server's side if candidate frame `f` falls inside a serve's flight (0..NET_SERVE_LOOKBACK
    frames after a serve strike), else None. A serve flight to the net is well under a second."""
    if not serves:
        return None
    best = None
    for s in serves:
        sf = s.get("frame")
        side = s.get("side")
        if sf is None or side is None:
            continue
        if sf - 1.0 <= f <= sf + NET_SERVE_LOOKBACK:
            if best is None or sf > best[0]:
                best = (sf, side)
    return best[1] if best else None


def disambiguate_net_bounce(cand, proj, serves=None):
    """RULE 1 (owner 2026-07-23): decide net-tape hit vs ground bounce for a candidate whose base
    verdict is 'bounce'. Three signals, combined so no single px threshold decides alone:

      (a) TAPE PROXIMITY -- the ball projects within NET_TAPE_BAND_PX of the WHITE TAPE line in the
          image. Only then is a net hit a live hypothesis; a bounce tens of px from the tape is left
          untouched (this gate is what keeps far-court bounces from ever flipping).
      (b) SAME-SIDE IMPOSSIBILITY -- the candidate's ground position sits on the HITTER'S OWN side of
          the net before any net crossing. A same-side pre-net bounce essentially never happens in
          tennis, so this strongly implies the ball hit the net. The hitter's side is the server's
          side when the candidate is in a serve flight (authority); otherwise it is inferred from the
          incoming vertical image motion (this broadcast: far court is up-image, so a ball descending
          the image -- vy_before>0 -- is arriving from the far side).
      (c) POST-KINK ARC -- a tape clip / mesh death sheds nearly all speed and the horizontal motion
          collapses or reverses (the ball drops); a TRUE bounce rebounds with speed shed but keeps
          real horizontal motion in the SAME direction. Classify by that signature.

    Reclassifies to 'net' when, inside the tape band, either same-side-impossibility holds or the
    post-kink arc shows the ball dying -- unless the post-kink arc is a clean continuing bounce.
    Mutates and returns `cand`.
    """
    if cand["verdict"] != "bounce":
        return cand
    u, v, f = cand["img_x"], cand["img_y"], cand["frame"]
    td, _vsigned = tape_pixel_dist(proj, u, v, int(round(f)))
    cand["tape_px"] = round(td, 1)
    if td > NET_TAPE_BAND_PX:  # (a) not at the tape -> a genuine ground bounce
        return cand
    # (b) same-side impossibility
    origin = _serve_side_for(f, serves)
    if origin is None:
        vyb = cand.get("vy_before", 0.0)
        origin = "far" if vyb > 1.0 else ("near" if vyb < -1.0 else None)
    cur_side = "far" if cand["court_y"] > NET_Y else "near"
    same_side = origin is not None and origin == cur_side
    # (c) post-kink arc signature
    spb, spa = cand.get("sp_before", 0.0), cand.get("sp_after", 0.0)
    vxb, vxa = cand.get("vx_before", 0.0), cand.get("vx_after", 0.0)
    dead = spa <= NET_DEAD_SP and spa <= NET_DEAD_FRAC * max(spb, 1e-6)
    continuing_bounce = vxb * vxa > 0 and abs(vxa) >= NET_BOUNCE_VX and spa > NET_DEAD_SP
    if (same_side or dead) and not continuing_bounce:
        cand["verdict"] = "net"
        why = []
        if same_side:
            why.append("same_side_%s" % cur_side)
        if dead:
            why.append("post_kink_dead")
        cand["flags"].append("net_tape_reclass:" + "+".join(why))
        cand["confidence"] = round(min(0.9, max(cand["confidence"], 0.5)), 3)
    return cand


def _load_pose_for_gate(pose_file, clip):
    """{side -> {frame -> {kp_name -> (x,y,conf)}}} for the swing gate; empty if absent."""
    pose = {"near": {}, "far": {}}
    if not pose_file or not os.path.exists(pose_file):
        return pose
    with open(pose_file) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("clip,"):
                continue
            p = line.rstrip("\n").split(",")
            if len(p) != 7 or p[0] != clip:
                continue
            fr = p[1].strip()
            fnum = frame_num(fr)
            pose.setdefault(p[2], {}).setdefault(fnum, {})[p[3]] = (
                float(p[4]),
                float(p[5]),
                float(p[6]),
            )
    return pose


def detect(
    match: str,
    clip: str,
    camera_file: str,
    ball_file: str,
    box_file: str,
    serve_frame: float | None = None,
    pose_file: str | None = None,
    serves=None,
):
    frames, xs, ys, tid = load_track(ball_file, clip)
    proj = projector_for_artifact(camera_file, clip, ball_file)
    boxes = load_boxes(box_file, clip)
    # pose (for the hand-reach swing gate); auto-discover the per-clip CSV if not given
    if pose_file is None:
        cand = os.path.join(os.path.dirname(ball_file), "player_pose_kp17_%s_v1.csv" % clip)
        pose_file = cand if os.path.exists(cand) else None
    pose = _load_pose_for_gate(pose_file, clip)
    runs = split_runs(frames, xs, ys)
    raw = []
    for run in runs:
        raw.extend(detect_in_run(np.array(run), frames, xs, ys, proj, boxes))
    for c in raw:
        classify_and_score(c, boxes, proj, pose=pose)
    # RULE 1: net-tape hit vs ground bounce disambiguation (owner 2026-07-23). A serve context
    # (frame+side) supplies the same-side authority; if absent it is inferred from image motion.
    if serves is None and serve_frame is not None:
        serves = [{"frame": serve_frame, "side": None}]
    for c in raw:
        disambiguate_net_bounce(c, proj, serves=serves)
    # gate pre-serve ball bouncing (before the first serve strike) as non-play.
    if serve_frame is not None:
        for c in raw:
            if c["frame"] < serve_frame - 4:
                if c["verdict"] == "bounce":
                    c["verdict"] = "pre_serve"
                c["flags"].append("pre_serve")
    # de-duplicate near-identical detections (same event caught in overlapping windows).
    raw.sort(key=lambda c: c["frame"])
    dedup = []
    for c in raw:
        if (
            dedup
            and abs(c["frame"] - dedup[-1]["frame"]) <= 3
            and c["verdict"] == dedup[-1]["verdict"]
        ):
            if c["confidence"] > dedup[-1]["confidence"]:
                dedup[-1] = c
            continue
        dedup.append(c)
    return dedup, proj


COLS = [
    "clip",
    "frame",
    "img_x",
    "img_y",
    "court_x",
    "court_y",
    "confidence",
    "kink_strength",
    "verdict",
    "flags",
    "box_side",
    "arc",
    "vy_before",
    "vy_after",
    "tape_px",
]


def write_csv(path: str, clip: str, dets: list[dict]):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for c in dets:
            w.writerow(
                [
                    clip,
                    round(c["frame"], 2),
                    round(c["img_x"], 2),
                    round(c["img_y"], 2),
                    round(c["court_x"], 3),
                    round(c["court_y"], 3),
                    c["confidence"],
                    c["kink_strength"],
                    c["verdict"],
                    "|".join(c["flags"]),
                    c.get("box_side") or "",
                    c["arc"],
                    round(c["vy_before"], 2),
                    round(c["vy_after"], 2),
                    c.get("tape_px", ""),
                ]
            )
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="rg2025f")
    ap.add_argument("--clip", default="pt0021")
    ap.add_argument(
        "--camera-file", required=True, help="per-frame camera npz (v3: P+k1+dist_center)"
    )
    ap.add_argument("--ball-file", default=None, help="decoded ball track csv (abs path)")
    ap.add_argument("--box-file", default=None, help="player boxes v2 csv (abs path)")
    ap.add_argument(
        "--serve-frame",
        type=float,
        default=None,
        help="first serve-strike frame; bounces before it are tagged pre_serve",
    )
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    proc = os.path.join(HERE, "..", "..", "data", "processed", args.match)
    ball = args.ball_file or os.path.join(proc, "ball_track_wasb_full_v1_decoded.csv")
    box = args.box_file or os.path.join(proc, "player_boxes_50_full_v2.csv")
    dets, _ = detect(args.match, args.clip, args.camera_file, ball, box, args.serve_frame)
    write_csv(args.out_csv, args.clip, dets)
    bounces = [d for d in dets if d["verdict"] == "bounce"]
    print("wrote %d detections (%d bounce) -> %s" % (len(dets), len(bounces), args.out_csv))
    for d in dets:
        print(
            "  f%-8.2f %-14s conf=%.2f kink=%5.2f court=(%6.2f,%6.2f) img=(%5.0f,%5.0f) %s"
            % (
                d["frame"],
                d["verdict"],
                d["confidence"],
                d["kink_strength"],
                d["court_x"],
                d["court_y"],
                d["img_x"],
                d["img_y"],
                "|".join(d["flags"]),
            )
        )


if __name__ == "__main__":
    main()
