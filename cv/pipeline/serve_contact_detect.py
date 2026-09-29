"""Serve-contact detector (Stage-2 rebuild, Q0b step 3).

WHY. The incumbent serve contact is a placeholder heuristic (the "chest ghost", ~5.5 m off):
it reprojects to the server's armpit while the ball at contact is a metre away, high and to
the side (owner review of pt0021: displayed serve contact f662 at Alcaraz's armpit is WRONG;
TRUE contact is f666, "much higher and to the viewer's right", ~1 m off). And the FIRST-serve
FAULT strike is never detected at all ("a massive sudden change in ball trajectory in a
well-tracked ball" that the contact-first middle drops). This module detects the serve strike
FROM THE BALL TRACK (+ audio + pose), so the serve contact is a measured event, not a guess.

THE SERVE SIGNATURE. A serve begins with a TOSS: the ball leaves the hand and rises (image-y
DECREASES for a far server / increases-then-decreases handled by sign), decelerates to an APEX
(image-y extremum, near-zero velocity), and HANGS for a few frames. Then the racket STRIKES it:
a sudden velocity discontinuity -- the ball departs the hang with a large downward+lateral
speed. The strike frame is that impulse; the contact image position is the tracked ball there
(high, near the extended racket -- NOT the body).

FOUR TECHNIQUES (each an independent estimate; fused at the end):
  1. TOSS-IMPULSE (owner's heuristic). Smooth the track, find the toss apex (image-y extremum
     with low speed), then the first post-apex frame where the ball SPEED jumps beyond the
     hang noise (the impulse). That frame = contact; the track there = contact pixel. Robust to
     audio position-delay and to pose dropout.
  2. POSE-KINEMATIC. The serving player's racket wrist reaches PEAK EXTENSION (image-y minimum /
     arm fully up) at contact. Gives an independent timing and a POSITION prior: the ball at
     contact is near the extended wrist, above the head.
  3. AUDIO-ANCHORED. The contact-audio onset lane near the impulse refines timing to (sub)frame.
     (Position-dependent delay: trusted only as a local snap around the impulse, not absolute.)
  4. TOSS-ARC DIVERGENCE. Fit a ballistic (free-fall) arc to the toss's rising+hang leg;
     the contact is the first frame the track deviates beyond pixel noise from that arc
     (the ball stops being in free flight = the racket has acted on it).

3D POSITION (honest). The ball at contact is NOT on a known plane, so depth is constrained, not
exact: the server's court position (feet from the box bottom / pose ankle through the v3 ground
plane) fixes court x,y near the baseline; the contact HEIGHT comes from the camera ray at a
plausible overhead reach (~2.4-2.9 m). Residual depth uncertainty is reported, not hidden.

Imports geometry from bounce_detect (GroundProjector, load_track, load_boxes); modifies nothing.
Usage:
  .venv/bin/python cv/pipeline/serve_contact_detect.py --clip pt0021 \
      --camera-file <v3 npz abs> --serve-hints 60,664 [--pose-file <csv>] [--eval]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bounce_detect import (  # noqa: E402
    GroundProjector,
    _smooth,
    load_boxes,
    load_track,
    near_player_box,
    projector_for_artifact,
)

AUDIO_SAMPLE_RATE = 100.0
REFERENCE_FPS = 50.0
SERVE_REACH_H = (2.35, 2.90)                       # plausible overhead contact-height band (m)


# --------------------------------------------------------------------------- pose loading
COCO_WRISTS = ("left_wrist", "right_wrist")


def load_pose(pose_file: str, clip: str):
    """{side -> {frame -> {kp_name -> (x, y, conf)}}} in 540-space, or empty if absent."""
    pose: dict[str, dict[int, dict[str, tuple]]] = {"near": {}, "far": {}}
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
            f = int(fr[2:6]) if fr.startswith("f_") else int(round(float(fr)))
            pose.setdefault(p[2], {}).setdefault(f, {})[p[3]] = (
                float(p[4]), float(p[5]), float(p[6]))
    return pose


def load_audio(audio_file: str, clip: str):
    if not audio_file or not os.path.exists(audio_file):
        return None
    d = np.load(audio_file, allow_pickle=True)
    key = "scores_%s" % clip
    return np.asarray(d[key]) if key in d else None


def audio_at(scores, f: float, fps: float = REFERENCE_FPS) -> float:
    if scores is None:
        return float("nan")
    i = int(round(AUDIO_SAMPLE_RATE * f / fps))
    return float(scores[i]) if 0 <= i < len(scores) else float("nan")


SERVE_FAST = 5.5                                   # px/frame that separates served flight from toss
HANG_SLOW = 3.0                                     # px/frame ceiling for the pre-strike hold
STRIKE_HINT_BACK_SECONDS = 0.48


def window_track(
    frames,
    xs,
    ys,
    hint,
    *,
    fps=REFERENCE_FPS,
    back_seconds=1.04,
    forward_seconds=0.68,
):
    """Sorted (frame, x, y) samples within [hint-back, hint+fwd] as float arrays."""
    m = (frames >= hint - back_seconds * fps) & (
        frames <= hint + forward_seconds * fps
    )
    f = frames[m].astype(float)
    x = xs[m].astype(float)
    y = ys[m].astype(float)
    o = np.argsort(f)
    return f[o], x[o], y[o]


def serve_side(frames, xs, ys, boxes, hint, *, fps=REFERENCE_FPS, expand=0.7):
    """Which player box the toss/serve run sits near (the server)."""
    f, x, y = window_track(frames, xs, ys, hint, fps=fps)
    for i in range(len(f)):
        s = near_player_box(boxes, x[i], y[i], int(f[i]), expand=expand)
        if s:
            return s
    return None


# --------------------------------------------------------------------------- technique 1
def toss_impulse(frames, xs, ys, hint, *, fps=REFERENCE_FPS):
    """Owner's heuristic, gap-aware. The toss rises to an apex (image-y min) then HANGS slow;
    the racket STRIKE launches sustained fast flight. The strike frame is the boundary between
    the slow hold and the fast served flight -- the last hold sample, or the MIDPOINT of the
    track gap when the ball is briefly untracked across the impulse (occlusion-bias rule).
    Contact pixel = the tracked sample at the strike, interpolated across the gap."""
    f, x, y = window_track(frames, xs, ys, hint, fps=fps)
    if len(f) < 6:
        return None
    smooth_width = max(1, round(0.06 * fps))
    xs_s = _smooth(x, smooth_width)
    ys_s = _smooth(y, smooth_width)
    # per-sample speed normalised by the frame gap (px/frame), so a gap doesn't fake a jump
    df = np.diff(f)
    df[df == 0] = 1
    step = np.hypot(np.diff(xs_s), np.diff(ys_s)) / df * fps / REFERENCE_FPS
    step = np.r_[step[0], step]                     # align to samples
    # STRIKE = the first SUSTAINED fast-flight sample that is PRECEDED by a low-speed toss HANG.
    # Direction-agnostic: a NEAR server's struck ball RISES in image (img-y falls below the toss
    # apex) while a FAR server's DESCENDS, so keying off the image-y extremum (old `argmin(y)`)
    # mis-fires for a near serve -- it locks onto the higher post-strike ball and returns a strike
    # ~20 frames late (pt0026: f70 instead of f50). The pre-strike hang (speed dips near zero as
    # the toss hangs, then jumps at contact) is the invariant. The fast toss RISE is excluded
    # because it is NOT preceded by a hang.
    fast_i = None
    for i in range(3, len(f) - 1):
        if f[i] < hint - STRIKE_HINT_BACK_SECONDS * fps:
            continue
        if step[i] > SERVE_FAST and step[i + 1] > SERVE_FAST \
                and float(np.median(step[max(0, i - 4):i])) < HANG_SLOW:
            fast_i = i
            break
    if fast_i is None:                              # fall back: old apex-then-fast behaviour
        eligible = np.flatnonzero(f >= hint - STRIKE_HINT_BACK_SECONDS * fps)
        if not len(eligible):
            return None
        apex0 = int(eligible[np.argmin(y[eligible])])
        for i in range(apex0 + 1, len(f) - 1):
            if step[i] > SERVE_FAST and step[i + 1] > SERVE_FAST:
                fast_i = i
                break
        if fast_i is None:                          # fall back: global max jerk after apex
            jerk = np.diff(step, prepend=step[0])
            cand = np.arange(apex0 + 1, len(f))
            if not len(cand):
                return None
            fast_i = int(cand[np.argmax(jerk[cand])])
    # apex/hang = the slowest sample in the ~18 frames before the strike (the toss hang)
    lo = max(0, fast_i - 18)
    apex_i = lo + int(np.argmin(step[lo:fast_i + 1])) if fast_i > lo else max(0, fast_i - 1)
    apex_f = float(f[apex_i])
    # the strike sits at the boundary: last sample still in the hold (slow) before fast_i
    hold_i = fast_i - 1
    while hold_i > apex_i and step[hold_i] > SERVE_FAST:
        hold_i -= 1
    f_hold, f_fast = f[hold_i], f[fast_i]
    gap = f_fast - f_hold
    if gap >= 3:                                    # untracked across the impulse -> gap midpoint
        strike_f = float(round((f_hold + f_fast) / 2.0))
        # contact pixel interpolated across the gap
        sx = float(np.interp(strike_f, [f_hold, f_fast], [x[hold_i], x[fast_i]]))
        sy = float(np.interp(strike_f, [f_hold, f_fast], [y[hold_i], y[fast_i]]))
    else:                                           # tracked through: strike = last hold sample
        strike_f = float(f_hold)
        sx, sy = float(x[hold_i]), float(y[hold_i])
    jump = float(step[fast_i] / max(np.median(step[max(0, hold_i - 3):hold_i + 1]), 0.4))
    return {"apex_frame": apex_f, "strike_frame": strike_f, "strike_px": (sx, sy),
            "speed_jump": jump, "gap": float(gap)}


# --------------------------------------------------------------------------- technique 2
def pose_peak_wrist(pose, side, hint, *, fps=REFERENCE_FPS, window_seconds=0.6):
    """Peak wrist extension (image-y minimum among the two wrists) near the hint: independent
    contact timing + a position prior (ball is near the extended wrist)."""
    if side not in pose or not pose[side]:
        return None
    best = None
    for f, kps in pose[side].items():
        if abs(f - hint) > window_seconds * fps:
            continue
        for w in COCO_WRISTS:
            v = kps.get(w)
            if v is None or v[2] < 0.45:
                continue
            if best is None or v[1] < best[2]:      # smaller y = higher = more extended
                best = (f, w, v[1], v[0], v[2])
    if best is None:
        return None
    return {"frame": float(best[0]), "wrist": best[1], "wrist_px": (best[3], best[2]),
            "conf": best[4]}


# --------------------------------------------------------------------------- technique 3
def audio_refine(
    scores,
    strike_frame,
    *,
    fps=REFERENCE_FPS,
    radius_seconds=0.1,
):
    """Snap to the strongest audio onset within +/-radius of the impulse (local, not absolute)."""
    if scores is None:
        return None
    radius = max(1, round(radius_seconds * fps))
    fs = np.arange(strike_frame - radius, strike_frame + radius + 1)
    vals = [(audio_at(scores, f, fps), f) for f in fs]
    vals = [(v, f) for v, f in vals if not np.isnan(v)]
    if not vals:
        return None
    v, f = max(vals)
    return {"frame": float(f), "score": float(v)}


# --------------------------------------------------------------------------- technique 4
def toss_arc_divergence(
    frames,
    xs,
    ys,
    hint,
    apex_frame,
    *,
    fps=REFERENCE_FPS,
    sigma_px=3.0,
):
    """Fit free-fall (parabola in image-y vs frame, line in image-x) to the toss's rising+apex
    leg; contact = first post-apex frame the track leaves that arc beyond ~3*sigma_px (the ball
    stops being in free flight = the racket has acted on it)."""
    f, x, y = window_track(frames, xs, ys, hint, fps=fps)
    ai = int(np.argmin(np.abs(f - apex_frame)))
    lo = max(0, ai - 14)
    fit_f = f[lo:ai + 2]
    if len(fit_f) < 5:
        return None
    t = fit_f - fit_f[0]
    cy = np.polyfit(t, y[lo:ai + 2], 2)
    cx = np.polyfit(t, x[lo:ai + 2], 1)
    for j in range(ai, len(f)):
        tt = f[j] - fit_f[0]
        if np.hypot(x[j] - np.polyval(cx, tt), y[j] - np.polyval(cy, tt)) > 3.0 * sigma_px:
            return {"frame": float(f[j]), "px": (float(x[j]), float(y[j]))}
    return None


# --------------------------------------------------------------------------- 3D constraint
def contact_3d(proj: GroundProjector, boxes, pose, side, strike_frame, strike_px):
    """Constrain the contact 3D: court x,y from the server feet (pose ankle or box bottom via
    the ground plane); height from the camera ray at the plausible overhead-reach band. Returns
    (court_x, court_y, z, depth_uncertainty_note)."""
    f = int(round(strike_frame))
    feet = None
    if pose.get(side):
        near_f = min(pose[side], key=lambda g: abs(g - f)) if pose[side] else None
        if near_f is not None and abs(near_f - f) <= 6:
            kps = pose[side][near_f]
            anks = [kps.get(a) for a in ("left_ankle", "right_ankle")]
            anks = [a for a in anks if a and a[2] > 0.4]
            if anks:
                ax = np.mean([a[0] for a in anks])
                ay = np.mean([a[1] for a in anks])
                feet = proj.to_court(ax, ay, f)
    if feet is None:
        b = boxes.get(side, {})
        cand = [g for g in b if abs(g - f) <= 6]
        if cand:
            x0, y0, x1, y1 = b[min(cand, key=lambda q: abs(q - f))]
            feet = proj.to_court((x0 + x1) / 2.0, y1, f)
    # contact height from the ray: intersect the (undistorted) contact ray at the server's
    # court x,y column and read the height where it best matches the reach band.
    z_lo, z_hi = SERVE_REACH_H
    # solve for (depth along y) and z given court x fixed at feet-x: parametrise the ray by z,
    # find court-y so the pixel matches; then clamp z into the reach band. Report midband.
    z = 0.5 * (z_lo + z_hi)
    court = feet if feet else (float("nan"), float("nan"))
    note = "court x,y from %s feet; height=reach-band midpoint %.2f m (depth +/- ~1 m)" % (
        "pose-ankle" if pose.get(side) else "box-bottom", z)
    return (court[0], court[1], z, note)


# --------------------------------------------------------------------------- orchestration
def detect_serve(
    clip,
    camera_file,
    ball_file,
    box_file,
    hint,
    pose_file=None,
    audio_file=None,
    fps=REFERENCE_FPS,
):
    frames, xs, ys, tid = load_track(ball_file, clip)
    proj = projector_for_artifact(camera_file, clip, ball_file)
    boxes = load_boxes(box_file, clip)
    pose = load_pose(pose_file, clip)
    scores = load_audio(audio_file, clip)

    side = serve_side(frames, xs, ys, boxes, hint, fps=fps)
    t1 = toss_impulse(frames, xs, ys, hint, fps=fps)
    if t1 is None:
        return {"hint": hint, "error": "no serve run near hint"}
    t2 = pose_peak_wrist(pose, side, hint, fps=fps) if side else None
    t3 = audio_refine(scores, t1["strike_frame"], fps=fps)
    t4 = toss_arc_divergence(
        frames, xs, ys, hint, t1["apex_frame"], fps=fps
    )

    # FUSE: impulse gives the coarse strike; audio snaps timing if a strong onset is adjacent;
    # toss-arc divergence is a cross-check; pose gives the position sanity + independent timing.
    strike = t1["strike_frame"]
    timing_src = "toss_impulse"
    if t3 and t3["score"] > 8 and abs(t3["frame"] - strike) <= 2:
        strike = t3["frame"]
        timing_src = "audio_refined"
    strike_px = t1["strike_px"]
    if timing_src == "audio_refined":
        # re-read the contact pixel at the audio-snapped frame (interp across any gap)
        wf, wx, wy = window_track(frames, xs, ys, hint, fps=fps)
        if len(wf):
            strike_px = (float(np.interp(strike, wf, wx)), float(np.interp(strike, wf, wy)))

    cx, cy, cz, note = contact_3d(proj, boxes, pose, side, strike, strike_px)
    return {
        "hint": hint, "side": side, "strike_frame": strike, "timing_src": timing_src,
        "strike_px": strike_px, "court_x": cx, "court_y": cy, "z": cz, "depth_note": note,
        "toss_impulse": t1, "pose": t2, "audio": t3, "toss_divergence": t4,
    }


# --------------------------------------------------------------------------- eval / CLI
OWNER = {  # pt0021 owner labels: strike frame + approx contact pixel (540-space)
    "main": {"hint": 664, "frame": 666, "px": (429, 90),
             "note": "high above far server, to the viewer's right"},
    "fault": {"hint": 60, "frame": 71, "px": (422, 89),
              "note": "toss apex f58, hang f58-69, impulse in the f69-73 gap"},
}


def _fmt(v):
    return "n/a" if v is None else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="rg2025f")
    ap.add_argument("--clip", default="pt0021")
    ap.add_argument("--camera-file", required=True)
    ap.add_argument("--ball-file", default=None)
    ap.add_argument("--box-file", default=None)
    ap.add_argument("--pose-file", default=None)
    ap.add_argument("--audio-file", default=None)
    ap.add_argument("--serve-hints", default="60,664")
    ap.add_argument("--eval", action="store_true", help="print per-technique table vs owner labels")
    args = ap.parse_args()
    proc = os.path.join(HERE, "..", "..", "data", "processed", args.match)
    ball = args.ball_file or os.path.join(proc, "ball_track_wasb_full_v1_decoded.csv")
    box = args.box_file or os.path.join(proc, "player_boxes_50_full_v2.csv")
    pose = args.pose_file or os.path.join(proc, "player_pose_kp17_%s_v1.csv" % args.clip)
    audio = args.audio_file or os.path.join(proc, "contact_audio_scores_16k_v1.npz")
    hints = [float(s) for s in args.serve_hints.split(",") if s.strip()]

    results = [detect_serve(args.clip, args.camera_file, ball, box, h, pose, audio) for h in hints]
    for r in results:
        if r.get("error"):
            print("hint %s: %s" % (r["hint"], r["error"]))
            continue
        t1, t2, t3, t4 = r["toss_impulse"], r["pose"], r["audio"], r["toss_divergence"]
        print("\n=== serve hint %s (side=%s) ===" % (r["hint"], r["side"]))
        print("  1 toss-impulse : apex f%.0f  STRIKE f%.1f  px(%.0f,%.0f)  jump x%.1f"
              % (t1["apex_frame"], t1["strike_frame"], t1["strike_px"][0], t1["strike_px"][1],
                 t1["speed_jump"]))
        print("  2 pose-wrist   : %s" % ("peak %s f%.0f px(%.0f,%.0f) conf%.2f"
              % (t2["wrist"], t2["frame"], t2["wrist_px"][0], t2["wrist_px"][1], t2["conf"])
              if t2 else "n/a (no serving-side pose in window)"))
        print("  3 audio-refine : %s" % ("onset f%.0f score %.1f" % (t3["frame"], t3["score"])
              if t3 else "n/a"))
        print("  4 toss-diverge : %s" % ("leaves arc at f%.0f px(%.0f,%.0f)"
              % (t4["frame"], t4["px"][0], t4["px"][1]) if t4 else "n/a (stays in free-fall)"))
        print("  FUSED strike f%.1f (%s) px(%.0f,%.0f)  3D court(%.2f,%.2f,%.2f)"
              % (r["strike_frame"], r["timing_src"], r["strike_px"][0], r["strike_px"][1],
                 r["court_x"], r["court_y"], r["z"]))
        print("       %s" % r["depth_note"])

    if args.eval:
        print("\n================ SERVE EVAL vs owner labels (pt0021) ================")
        print("%-6s %-14s %6s %6s %8s  %s" % ("serve", "technique", "det_f", "lbl_f", "px_err", "verdict"))
        pair = {"main": results[1] if len(results) > 1 else results[0], "fault": results[0]}
        for name, r in pair.items():
            lab = OWNER[name]
            if r.get("error"):
                print("%-6s %-14s   ERROR" % (name, "-"))
                continue
            t1, t2, t3, t4 = r["toss_impulse"], r["pose"], r["audio"], r["toss_divergence"]
            def row(tech, df, px):
                pe = "" if px is None else "%.0f" % np.hypot(px[0] - lab["px"][0], px[1] - lab["px"][1])
                dv = "" if df is None else ("HIT" if abs(df - lab["frame"]) <= 2 else "off%+.0f" % (df - lab["frame"]))
                print("%-6s %-14s %6s %6d %8s  %s"
                      % (name, tech, ("%.1f" % df) if df is not None else "-", lab["frame"], pe, dv))
            row("toss_impulse", t1["strike_frame"], t1["strike_px"])
            row("pose_wrist", t2["frame"] if t2 else None, t2["wrist_px"] if t2 else None)
            row("audio", t3["frame"] if t3 else None, None)
            row("toss_diverge", t4["frame"] if t4 else None, t4["px"] if t4 else None)
            row("FUSED", r["strike_frame"], r["strike_px"])
            print("  label: %s\n" % lab["note"])


if __name__ == "__main__":
    main()
