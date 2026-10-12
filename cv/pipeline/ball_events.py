"""Unified ball-event layer (Q0e revamp, 2026-07-24).

WHY. The per-type detectors (bounce_detect, serve_contact_detect, net_hit_detect) each owned a
piece of the classification decision, and the owner's ground-truth audit (contact_labels.csv)
convicted the split: five bounces were typed as racket contacts (pt0026 f206/f408/f481/f602/f614),
a net hit was typed as a contact (pt0026 f720), a real contact was mis-handled as a bounce
(pt0026 f217), contact timing landed a frame or two late (pt0026 f697), the sub-frame serve
contact was never constructed (pt0026 f50.5), a gap-bridged fault strike was dropped (pt0021 f72),
and a terminal contact was missed (pt0021 f759).

THIS MODULE owns the DECISION. Every track discontinuity/kink (in-run velocity changepoint OR a
gap between two arcs that brackets an event) becomes ONE candidate, classified into exactly one of
{racket_hit, bounce, net_hit, tracking_artifact} by physically-grounded rules, with sub-frame
timing and an incoming-authoritative position. It IMPORTS the primitives (GroundProjector,
track/box/pose loading, tape geometry, net-plane intersection) from the per-type detectors and
supersedes their classification duty; they remain as measurement libraries.

THE RULES (owner's directive, made mechanical):
  * ENERGY/SPEED REGIME. A racket hit ADDS speed (impulse: outgoing >> incoming, or a clean
    horizontal reversal); a bounce SHEDS vertical speed while keeping horizontal motion; a net hit
    COLLAPSES speed (the ball dies on the cord).
  * DIRECTION GEOMETRY. A bounce CONTINUES the same horizontal direction (ball keeps travelling
    the way it came, just up instead of down); a hit REVERSES or REDIRECTS; a net hit KILLS motion.
  * PLAYER PROXIMITY (image space). Wrist reach (pose skeleton, box fallback) is the precise cue:
    a soft/high contact sits within a player's wrist reach even when the ball box-projection to the
    ground is beyond the court (the ball is at racket height, not on the clay); a ground bounce at a
    player's FEET is outside wrist reach.
  * NET TAPE BAND + SAME-SIDE-IMPOSSIBILITY. A net event's ball pixel sits within a few px of the
    projected white-tape line and its speed collapses; a far-court bounce projects tens of px away.
  * AUDIO ONSET agrees (confirms/flags, never outvotes the track).
  * TWO CONSTRUCTIONS. Sub-frame timing + position at a hit come from INTERSECTING the incoming and
    outgoing local trajectories (serves: toss-line meets serve-line; f697-class: incoming meets
    outgoing); when the ball is briefly UNTRACKED at the event (gap-bridging: fault f72, f217) the
    same intersection is taken across the gap. Position is INCOMING-AUTHORITATIVE (the incoming arc
    sampled the ball down to the event; the outgoing traces forward on its own).

Usage:
  .venv/bin/python cv/pipeline/ball_events.py --clip pt0026 \
      --camera-file <v3 npz abs> [--serve-frames 50,...] [--out-csv <path>] [--eval]
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bounce_detect import (  # noqa: E402
    classify_and_score as classify_bounce_witness,
    detect_in_run as detect_bounce_witnesses,
    disambiguate_net_bounce,
    load_track,
    load_boxes,
    projector_for_artifact,
    split_runs,
    near_player_box,
    player_reach,
    tape_pixel_dist,
    _load_pose_for_gate,
    NET_Y,
    BASELINE_NEAR,
    BASELINE_FAR,
)
from serve_contact_detect import load_audio, audio_at  # noqa: E402
from serve_hints import auto_serve_hints  # noqa: E402

# ------------------------------------------------------------------ classification knobs
VX_MIN = 1.6  # px/frame: a horizontal component this small is "no clear horizontal motion"
VERT_MIN = 1.6  # px/frame: same for vertical
BIG_GAIN_RATIO = 1.8  # outgoing speed >= this * incoming AND ...
BIG_GAIN_ABS = 8.0  # ... >= this absolute px/frame => a racket impulse (adds speed)
COLLAPSE_FRAC = 0.6  # net: outgoing <= this * incoming AND ...
COLLAPSE_ABS = 5.5  # ... <= this absolute px/frame => the ball died (speed collapse)
NET_COLLAPSE_MIN_INCOMING = 3.0  # slower "collapses" are tracker quantization, not impacts
TAPE_BAND_PX = 33.0  # ball pixel within this of the projected tape line => net is a live hypothesis
NET_OBSERVED_SUPPORT_PX = 25.0  # a net claim needs a REAL observation this close to the claimed point
DEPART_SLOW = 2.2  # px/frame: a "rest" incoming speed (held/hanging ball before a strike)
DEPART_FAST = 3.0  # px/frame: motion after rest (rest->motion departure = a strike)
ARREST_FAST = 4.0  # px/frame: motion before ...
ARREST_SLOW = 2.6  # ... rest after (motion->rest arrest, e.g. a soft high touch that stalls)
ANGLE_MIN = 33.0  # deg: turning angle that counts as a redirect
BASELINE_SKIM_MIN_ANGLE = 18.0
GROUND_Y_LO = BASELINE_NEAR - 2.0  # court-y band (z=0 projection) that is a PLAUSIBLE ground spot
GROUND_Y_HI = BASELINE_FAR + 2.0  # beyond it, the ball is at HEIGHT (contact), not on the clay
AUDIO_HIT = 40.0  # audio onset score that CONFIRMS a contact (never rejects on its own)
AUDIO_REDIRECT_MIN = 30.0
AUDIO_RADIUS = 3  # frames each side searched for the onset peak
SEQUENCE_AUDIO_RADIUS = 5
SEQUENCE_WRIST_AUDIO_MIN = 5.0
GAP_MAX = 9  # frames: bridge short occlusions; long bridges require track-strong evidence
SUPPRESS = 4  # frames: non-max-suppression radius for duplicate candidates
FIT_WIN = 6  # samples per side used to fit the local incoming/outgoing trajectory lines
KINK_WIN = 4  # samples per side used for the regime velocity vectors
REFERENCE_FPS = 50.0

_EVENT_FRAME_FIELDS = (
    "frame",
    "fL",
    "fF",
    "audio_frame",
    "sequence_audio_frame",
    "proposal_frame",
    "candidate_frame",
)


def _reference_scale(fps):
    if fps <= 0:
        raise ValueError("fps must be positive")
    return REFERENCE_FPS / float(fps)


def _reference_track(frames, xs, ys, fps):
    """Resample native observations onto the detector's calibrated 50 Hz clock."""
    frames = np.asarray(frames)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if not len(frames):
        return frames.astype(int), xs, ys, np.zeros((0, 3), float)
    scale = _reference_scale(fps)
    if abs(scale - 1.0) < 1e-9:
        return (
            frames.astype(int),
            xs.copy(),
            ys.copy(),
            np.column_stack([frames.astype(float), xs, ys]),
        )
    native_gap_max = max(1, int(np.ceil(3.0 / scale)))
    native_jump_px = 55.0 * scale
    reference_frames = []
    reference_xs = []
    reference_ys = []
    for run in split_runs(
        frames,
        xs,
        ys,
        gap_max=native_gap_max,
        jump_px=native_jump_px,
    ):
        indices = np.asarray(run, int)
        native_frames = frames[indices].astype(float)
        scaled_frames = native_frames * scale
        if len(indices) == 1:
            grid = np.asarray([int(round(scaled_frames[0]))])
        else:
            grid = np.arange(
                int(round(scaled_frames[0])),
                int(round(scaled_frames[-1])) + 1,
            )
        reference_frames.extend(grid.tolist())
        reference_xs.extend(np.interp(grid, scaled_frames, xs[indices]).tolist())
        reference_ys.extend(np.interp(grid, scaled_frames, ys[indices]).tolist())
    observed = np.column_stack([frames.astype(float) * scale, xs, ys])
    return (
        np.asarray(reference_frames, int),
        np.asarray(reference_xs, float),
        np.asarray(reference_ys, float),
        observed,
    )


def _reference_projector(projector, fps):
    scale = _reference_scale(fps)
    if abs(scale - 1.0) < 1e-9:
        return projector
    output = copy.copy(projector)
    for attribute in ("P", "k1", "center", "reliable"):
        if not hasattr(projector, attribute):
            continue
        values = getattr(projector, attribute)
        setattr(
            output,
            attribute,
            {int(round(frame * scale)): value for frame, value in values.items()},
        )
    output._fk = np.asarray(sorted(output.P))
    return output


def _reference_frame_map(values, fps):
    scale = _reference_scale(fps)
    if abs(scale - 1.0) < 1e-9:
        return values
    return {
        side: {int(round(frame * scale)): value for frame, value in frames.items()}
        for side, frames in values.items()
    }


def _scale_event_frames(events, factor):
    for event in events:
        for field in _EVENT_FRAME_FIELDS:
            if event.get(field) is not None:
                event[field] = float(event[field]) * factor
        if event.get("suppressed_candidate_frames"):
            event["suppressed_candidate_frames"] = [
                float(frame) * factor
                for frame in event["suppressed_candidate_frames"]
            ]
        witness = event.get("bounce_witness")
        if witness is not None and witness.get("frame") is not None:
            witness["frame"] = float(witness["frame"]) * factor


# ------------------------------------------------------------------ local kinematics
def _median_vel(fr, x, y):
    """Median per-frame velocity (vx, vy) over a short ordered sample block (gap-normalised)."""
    if len(fr) < 2:
        return 0.0, 0.0
    df = np.diff(fr).astype(float)
    df[df == 0] = 1.0
    vx = np.median(np.diff(x) / df)
    vy = np.median(np.diff(y) / df)
    return float(vx), float(vy)


def _line_fit(fr, x, y):
    """Least-squares line x(f), y(f) over the samples; returns (cx, cy) numpy poly1 coeffs."""
    fr = np.asarray(fr, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(fr) & np.isfinite(x) & np.isfinite(y)
    fr, x, y = fr[finite], x[finite], y[finite]
    if not len(fr):
        return np.zeros(2), np.zeros(2), 0.0
    unique_frames = np.unique(fr)
    if len(unique_frames) != len(fr):
        x = np.asarray([np.median(x[fr == frame]) for frame in unique_frames])
        y = np.asarray([np.median(y[fr == frame]) for frame in unique_frames])
        fr = unique_frames
    if len(fr) < 2:
        return (np.array([0.0, x[0]]), np.array([0.0, y[0]]), float(fr[0]))
    t = fr - fr[0]
    cx = np.polyfit(t, x, 1)
    cy = np.polyfit(t, y, 1)
    return cx, cy, float(fr[0])


def _intersect(linL, linR):
    """Frame tb where the incoming line L and outgoing line R best coincide (LSQ over x & y),
    and the INCOMING position there. linL/linR = (cx, cy, f0). Solves for the single frame t that
    minimises |P_L(t) - P_R(t)|^2 in image space (2 equations, 1 unknown)."""
    (cxL, cyL, f0L), (cxR, cyR, f0R) = linL, linR
    # P_L(f) - P_R(f) = 0 -> build A t = b from the two linear coords
    # x: cxL[0]*(f-f0L)+cxL[1] = cxR[0]*(f-f0R)+cxR[1]
    ax = cxL[0] - cxR[0]
    bx = cxR[0] * (-f0R) + cxR[1] - (cxL[0] * (-f0L) + cxL[1])
    ay = cyL[0] - cyR[0]
    by = cyR[0] * (-f0R) + cyR[1] - (cyL[0] * (-f0L) + cyL[1])
    A = np.array([ax, ay])
    B = np.array([bx, by])
    denom = float(A @ A)
    if denom < 1e-9:
        return None
    tb = float((A @ B) / denom)
    px = float(cxL[0] * (tb - f0L) + cxL[1])
    py = float(cyL[0] * (tb - f0L) + cyL[1])
    return tb, px, py


def sign_same(a, b):
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def _angle(vxb, vyb, vxa, vya):
    a = np.array([vxb, vyb])
    b = np.array([vxa, vya])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    c = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _refine_boundary(sp, vsx, vsy, j, n):
    """Pin a candidate boundary to the LAST-SLOW | FIRST-FAST frame (owner's pt0026 f698 rule:
    the contact is between the last slow frame and the first jumped frame, NEVER on/after the first
    fast frame -- a further speedup is already post-hit). Within a small window around j:
      * SPEED-JUMP mode: the incoming baseline speed sets a threshold; the FIRST frame F whose speed
        crosses it (and stays) is the first-fast frame -> boundary L = F-1. Deterministic first-cross,
        not the max acceleration (which drifts to the second, larger jump).
      * REVERSAL mode (no speed jump -- a redirect/soft touch at ~constant speed): the boundary is the
        sharpest single-frame direction turn.
    Returns the corrected boundary index L."""
    lo, hi = max(2, j - 4), min(n - 2, j + 4)
    # A literal component reversal is the strongest timing witness: the event lies between the
    # final sample moving one way and the first sample moving the other way. Resolve that before
    # speed-jump placement, which otherwise drifts to a later acceleration on the outgoing arc.
    reversals = []
    for b in range(lo, hi):
        horiz = vsx[b] * vsx[b + 1] < 0 and min(abs(vsx[b]), abs(vsx[b + 1])) >= VX_MIN
        vert = vsy[b] * vsy[b + 1] < 0 and min(abs(vsy[b]), abs(vsy[b + 1])) >= 1.0
        if (horiz or vert) and min(sp[b], sp[b + 1]) >= 2.5:
            reversals.append((_angle(vsx[b], vsy[b], vsx[b + 1], vsy[b + 1]), b))
    if reversals:
        return max(reversals)[1]

    base_win = sp[max(1, j - 6) : max(2, j - 1)]
    base = float(np.median(base_win)) if base_win.size else float(sp[max(1, j)])
    thr = max(BIG_GAIN_RATIO * base, base + 2.5, DEPART_FAST)
    for F in range(lo, hi + 1):
        nxt = sp[F : min(n, F + 2)]
        if sp[F] >= thr and nxt.size and float(np.median(nxt)) >= thr:
            return F - 1
    # ARREST mode: incoming fast, outgoing dies (a soft touch / stall that sheds nearly all speed).
    # Symmetric to first-fast: the FIRST frame whose speed drops below the collapse threshold is the
    # first-slow frame -> boundary L = F-1 (the last frame the ball was still moving in).
    if base > 3.0:
        drop = COLLAPSE_FRAC * base
        for F in range(lo, hi + 1):
            nxt = sp[F : min(n, F + 2)]
            if sp[F] <= drop and nxt.size and float(np.median(nxt)) <= drop:
                return F - 1
    best, best_ang = j, -1.0
    for b in range(lo, hi):
        a = _angle(vsx[b], vsy[b], vsx[b + 1], vsy[b + 1])
        if a > best_ang:
            best_ang, best = a, b
    return best


# ------------------------------------------------------------------ candidate generation
def _candidates(frames, xs, ys):
    """Every track discontinuity as a candidate boundary. Two kinds:
      * IN-RUN changepoint: an interior index where the incoming and outgoing regime velocities
        differ (turn angle, speed jump/collapse, rest<->motion). Non-max-suppressed by strength.
      * GAP bridge: a gap of 2..GAP_MAX frames between two runs -> one candidate spanning the gap.
    Returns list of dicts with the raw index geometry; timing/type are decided later."""
    runs = split_runs(frames, xs, ys)
    cand = []
    # in-run changepoints
    for run in runs:
        r = np.array(run)
        rf = frames[r].astype(float)
        rx = xs[r].astype(float)
        ry = ys[r].astype(float)
        n = len(r)
        if n < 5:
            continue
        # per-sample single-frame step velocity (gap-normalised): vstep[i] = (P[i]-P[i-1])/df.
        df = np.diff(rf)
        df[df == 0] = 1.0
        vsx = np.r_[0.0, np.diff(rx) / df]
        vsy = np.r_[0.0, np.diff(ry) / df]
        scored = []
        for j in range(2, n - 2):
            # boundary between sample j (last incoming) and j+1 (first outgoing). Robust regime
            # velocities: incoming steps that do NOT cross the boundary; outgoing steps from it.
            lb = slice(max(1, j - KINK_WIN + 1), j + 1)  # steps into j-W+1 .. j
            la = slice(j + 1, min(n, j + KINK_WIN + 1))  # steps j->j+1 .. (first = the jump)
            vxb, vyb = float(np.median(vsx[lb])), float(np.median(vsy[lb]))
            vxa, vya = float(np.median(vsx[la])), float(np.median(vsy[la]))
            sb = float(np.hypot(vxb, vyb))
            sa = float(np.hypot(vxa, vya))
            ang = _angle(vxb, vyb, vxa, vya)
            ratio = sa / max(sb, 0.3)
            departure = sb < DEPART_SLOW and sa > DEPART_FAST
            arrest = sb > ARREST_FAST and sa < ARREST_SLOW
            is_cand = (
                ang >= ANGLE_MIN
                or ratio >= BIG_GAIN_RATIO
                or ratio <= COLLAPSE_FRAC
                or departure
                or arrest
            )
            if not is_cand:
                continue
            # PLACEMENT weight = single-frame acceleration ACROSS this boundary (the jump instant).
            # This pins the boundary to the true last-slow|first-fast frame; the robust regime
            # medians (above) are ambiguous by +/-1 because a lone jump inside the window is
            # absorbed. Non-max suppression then keeps the sharpest boundary of each cluster.
            accel = float(np.hypot(vsx[j + 1] - vsx[j], vsy[j + 1] - vsy[j]))
            scored.append((accel, j, rf[j], vxb, vyb, vxa, vya, sb, sa, ang))
        scored.sort(key=lambda s: -s[0])
        taken = []
        for s in scored:
            if all(abs(s[2] - t[2]) > SUPPRESS for t in taken):
                taken.append(s)
        sp = np.hypot(vsx, vsy)
        for s in taken:
            _, j, fj, vxb, vyb, vxa, vya, sb, sa, ang = s
            j = _refine_boundary(sp, vsx, vsy, j, n)  # pin to last-slow|first-fast (owner rule)
            # recompute regime medians at the refined boundary
            lbb = slice(max(1, j - KINK_WIN + 1), j + 1)
            laa = slice(j + 1, min(n, j + KINK_WIN + 1))
            vxb, vyb = float(np.median(vsx[lbb])), float(np.median(vsy[lbb]))
            vxa, vya = float(np.median(vsx[laa])), float(np.median(vsy[laa]))
            sb, sa = float(np.hypot(vxb, vyb)), float(np.hypot(vxa, vya))
            ang = _angle(vxb, vyb, vxa, vya)
            lb0 = max(0, j - FIT_WIN + 1)
            la1 = min(n, j + 1 + FIT_WIN)
            cand.append(
                {
                    "kind": "kink",
                    "run": r,
                    "j": j,
                    "fL": float(rf[j]),
                    "fF": float(rf[j + 1]),
                    "px_at": (float(rx[j]), float(ry[j])),
                    "linL": _line_fit(rf[lb0 : j + 1], rx[lb0 : j + 1], ry[lb0 : j + 1]),
                    "linR": _line_fit(rf[j + 1 : la1], rx[j + 1 : la1], ry[j + 1 : la1]),
                    "vxb": vxb,
                    "vyb": vyb,
                    "vxa": vxa,
                    "vya": vya,
                    "sb": sb,
                    "sa": sa,
                    "ang": ang,
                }
            )
    # gap bridges
    for i in range(len(runs) - 1):
        rA = np.array(runs[i])
        rB = np.array(runs[i + 1])
        feA = frames[rA[-1]]
        fsB = frames[rB[0]]
        gap = fsB - feA
        if gap < 2 or gap > GAP_MAX:
            continue
        aslc = rA[-FIT_WIN:]
        bslc = rB[:FIT_WIN]
        fA = frames[aslc].astype(float)
        xA = xs[aslc].astype(float)
        yA = ys[aslc].astype(float)
        fB = frames[bslc].astype(float)
        xB = xs[bslc].astype(float)
        yB = ys[bslc].astype(float)
        vxb, vyb = _median_vel(fA, xA, yA)
        vxa, vya = _median_vel(fB, xB, yB)
        cand.append(
            {
                "kind": "gap",
                "run": None,
                "j": None,
                "fL": float(feA),
                "fF": float(fsB),
                "px_at": (float(xA[-1]), float(yA[-1])),
                "linL": _line_fit(fA, xA, yA),
                "linR": _line_fit(fB, xB, yB),
                "vxb": float(vxb),
                "vyb": float(vyb),
                "vxa": float(vxa),
                "vya": float(vya),
                "sb": float(np.hypot(vxb, vyb)),
                "sa": float(np.hypot(vxa, vya)),
                "ang": _angle(vxb, vyb, vxa, vya),
            }
        )
    return cand


# ------------------------------------------------------------------ classification
def _classify(c, proj, boxes, pose, scores, track=None):
    """Assign one type + sub-frame timing + incoming-authoritative position + witnesses."""
    vxb, vyb, vxa, vya = c["vxb"], c["vyb"], c["vxa"], c["vya"]
    sb, sa = c["sb"], c["sa"]
    # sub-frame timing + incoming position from the trajectory intersection (both constructions).
    inter = _intersect(c["linL"], c["linR"])
    if inter is not None and c["fL"] <= inter[0] <= c["fF"]:
        # the two trajectories cross WITHIN the last-incoming..first-outgoing bracket -> that
        # crossing is the sub-frame contact instant, position incoming-authoritative.
        tb, px, py = inter
        pos_src = "traj_intersect" + ("_gap" if c["kind"] == "gap" else "")
    else:
        # crossing falls outside the bracket (a sharp reversal whose extrapolated lines meet
        # early/late): the contact is inside the bracket -> take the gap midpoint, and read the
        # incoming line THERE (incoming-authoritative), not the raw dot.
        tb = 0.5 * (c["fL"] + c["fF"])
        cxL, cyL, f0L = c["linL"]
        px = float(cxL[0] * (tb - f0L) + cxL[1])
        py = float(cyL[0] * (tb - f0L) + cyL[1])
        pos_src = "bracket_mid"
    fi = int(round(tb))
    cx, cy = proj.to_court(px, py, fi)
    tape_px, _v = tape_pixel_dist(proj, px, py, fi)
    # Distance from the CLAIMED event point to the nearest OBSERVED track sample within
    # +-2 frames. The claim above is an extrapolated intersection (or a bracket-mid read of
    # the incoming line); nothing so far guarantees a real ball was ever seen there.
    observed_support_px = float("nan")
    if track is not None and len(track):
        near = track[np.abs(track[:, 0] - tb) <= 2.0]
        if len(near):
            observed_support_px = float(
                np.min(np.hypot(near[:, 1] - px, near[:, 2] - py))
            )
    track_context = _local_track_context(track, tb)
    reach = player_reach(boxes, pose, px, py, fi, tol_f=4)
    box = near_player_box(boxes, px, py, fi, expand=0.6)
    aud, aud_frame = _audio_peak(scores, tb)
    sequence_aud, sequence_aud_frame = _audio_peak(scores, tb, SEQUENCE_AUDIO_RADIUS)
    cxL, cyL, f0L = c["linL"]
    audio_px = float(cxL[0] * (aud_frame - f0L) + cxL[1])
    audio_py = float(cyL[0] * (aud_frame - f0L) + cyL[1])
    audio_cx, audio_cy = proj.to_court(audio_px, audio_py, aud_frame)
    audio_tape_px, _ = tape_pixel_dist(proj, audio_px, audio_py, aud_frame)
    sequence_px = float(cxL[0] * (sequence_aud_frame - f0L) + cxL[1])
    sequence_py = float(cyL[0] * (sequence_aud_frame - f0L) + cyL[1])
    sequence_cx, sequence_cy = proj.to_court(
        sequence_px, sequence_py, sequence_aud_frame
    )
    sequence_tape_px, _ = tape_pixel_dist(
        proj, sequence_px, sequence_py, sequence_aud_frame
    )
    player_box = None
    if box:
        box_frames = boxes.get(box, {})
        nearby = [frame for frame in box_frames if abs(frame - fi) <= 2]
        if nearby:
            player_box = box_frames[min(nearby, key=lambda frame: abs(frame - fi))]

    horiz_rev = (vxb * vxa < 0) and min(abs(vxb), abs(vxa)) > VX_MIN
    vert_rev = (vyb * vya < 0) and min(abs(vyb), abs(vya)) > VERT_MIN
    big_gain = sa >= BIG_GAIN_RATIO * sb and sa >= BIG_GAIN_ABS
    collapse = sa <= COLLAPSE_FRAC * sb and sa <= COLLAPSE_ABS
    departure = sb < DEPART_SLOW and sa > DEPART_FAST
    arrest = sb > ARREST_FAST and sa < ARREST_SLOW
    on_ground = GROUND_Y_LO <= cy <= GROUND_Y_HI
    at_height = not on_ground
    tape_near = tape_px <= TAPE_BAND_PX
    audio_hit = aud >= AUDIO_HIT
    dir_change = horiz_rev or vert_rev or departure or arrest or c["ang"] >= ANGLE_MIN
    shallow_ground_rev = (
        on_ground
        and vyb * vya < 0
        and abs(vyb) >= 0.75
        and abs(vya) >= 2.5
        and sign_same(vxb, vxa)
        and c["ang"] >= 45.0
    )
    baseline_ground_arrest = (
        on_ground
        and (cy <= BASELINE_NEAR or cy >= BASELINE_FAR)
        and vyb * vya < 0
        and abs(vyb) >= 5.0
        and abs(vya) >= 0.5
        and collapse
        and sign_same(vxb, vxa)
        and c["ang"] >= 60.0
        and (np.isnan(aud) or aud < 15.0)
    )
    feet_bounce = (
        player_box is not None
        and py >= player_box[3] - 2.0
        and vert_rev
        and not horiz_rev
        and (np.isnan(aud) or aud < 15.0)
    )
    long_gap = c["kind"] == "gap" and c["fF"] - c["fL"] > 6

    why = []
    net_die = (collapse and sb >= NET_COLLAPSE_MIN_INCOMING) or (
        vert_rev and sa < sb
    )  # speed collapses, or the ball drops off the tape
    # A net collision is strictly dissipative: the ball cannot leave faster than it arrived.
    # (Same physics gate that cut a review queue of net candidates from 78 to 33.)
    net_dissipates = sa <= 0.85 * sb
    # The tape test above runs on an INFERRED point. Untouched-cohort review found net_hit
    # proposed with the visible ball nowhere near the net: a wrong track identity or a bad
    # arc fit can place the extrapolated intersection on the tape. Require that a real
    # observed sample sits near the claimed point before the tape claim counts.
    net_observed = (
        np.isfinite(observed_support_px) and observed_support_px <= NET_OBSERVED_SUPPORT_PX
    )
    # NET: ball pixel at the tape band, its motion dies there (speed collapse OR a vertical reversal
    # that sheds speed = the ball hit the cord and dropped), and it is not a player redirect / volley.
    if tape_near and net_die and net_dissipates and net_observed and reach is None and not big_gain:
        vtype = "net_hit"
        why.append("tape%.0fpx" % tape_px)
        why.append("observed%.0fpx" % observed_support_px)
        why.append("collapse" if collapse else "drop_off_tape")
    # RACKET (clean impulse): a horizontal reversal or an added-speed impulse -- the track alone
    # is decisive, no proximity needed.
    elif horiz_rev or big_gain:
        vtype = "racket_hit"
        why.append("horiz_reverse" if horiz_rev else "speed_gain%.1fx" % (sa / max(sb, 0.3)))
    # RACKET (occluded redirect): a gap-bridged sharp turn tightly inside wrist reach is a contact
    # even when monocular ground projection places the low ball on the court plane.
    elif (
        c["kind"] == "gap"
        and not long_gap
        and reach is not None
        and reach[1] <= 0.85 * reach[2]
        and c["ang"] >= 75.0
    ):
        vtype = "racket_hit"
        why.append("wrist_reach_%s" % reach[0])
        why.append("occluded_redirect")
    elif feet_bounce:
        vtype = "bounce"
        why.append("feet_vertical_reversal")
    elif aud >= AUDIO_REDIRECT_MIN and c["ang"] >= 75.0 and at_height:
        vtype = "racket_hit"
        why.append("audio_redirect")
        tb = float(aud_frame)
        cxL, cyL, f0L = c["linL"]
        px = float(cxL[0] * (tb - f0L) + cxL[1])
        py = float(cyL[0] * (tb - f0L) + cyL[1])
        fi = int(round(tb))
        cx, cy = proj.to_court(px, py, fi)
        tape_px, _v = tape_pixel_dist(proj, px, py, fi)
        pos_src = "audio_onset_incoming"
    # RACKET (soft/high): within a player's WRIST reach with a velocity direction change, and the
    # ball is either at racket height (z=0 projection off court) or an audio onset confirms it --
    # NOT a clean ground bounce. Catches the terminal touch / high far-court contacts.
    elif reach is not None and not long_gap and dir_change and (at_height or audio_hit):
        vtype = "racket_hit"
        why.append("wrist_reach_%s" % reach[0])
        why.append("at_height" if at_height else "audio%.0f" % aud)
    # RACKET (audio-confirmed): a strong contact onset AT a track regime change, where the ball is
    # not on a clean ground-bounce signature. Catches the far serve whose wrist sits above the head
    # (out of the 42 px reach radius) but whose strike is unmistakable in the audio + the track.
    elif audio_hit and dir_change and not (on_ground and vert_rev and not horiz_rev and sb > 3.0):
        vtype = "racket_hit"
        why.append("audio_onset%.0f" % aud)
    # BOUNCE: on plausible ground, vertical reverses (world down->up), horizontal continues.
    elif on_ground and (vert_rev or shallow_ground_rev or baseline_ground_arrest) and not horiz_rev:
        vtype = "bounce"
        if vert_rev:
            why.append("ground_vreverse")
        elif baseline_ground_arrest:
            why.append("baseline_ground_arrest")
        else:
            why.append("shallow_ground_vreverse")
    # BOUNCE (far-baseline skim): a bounce right AT a baseline where the ball recedes nearly
    # parallel to the camera -- its world vertical reversal is < 1 px in the image (below the track's
    # ~1.9 px quantization), so it reads flat. Typed a bounce when it sits within a bounce-radius of a
    # baseline, keeps its horizontal direction, is away from any wrist, and is quiet (no contact
    # onset) -- tightly gated so it can never fire on a contact.
    elif (
        min(abs(cy - BASELINE_FAR), abs(cy - BASELINE_NEAR)) <= 1.6
        and sign_same(vxb, vxa)
        and abs(vxb) > VX_MIN
        and not horiz_rev
        and c["ang"] >= BASELINE_SKIM_MIN_ANGLE
        and reach is None
        and (np.isnan(aud) or aud < 15.0)
    ):
        vtype = "bounce"
        why.append("baseline_skim")
    else:
        vtype = "tracking_artifact"
        why.append("no_regime")

    if vtype == "bounce" and baseline_ground_arrest:
        tb = float(c["fL"])
        px, py = c["px_at"]
        fi = int(round(tb))
        cx, cy = proj.to_court(px, py, fi)
        tape_px, _v = tape_pixel_dist(proj, px, py, fi)
        pos_src = "baseline_ground_observation"

    if (
        vtype == "racket_hit"
        and any(reason.startswith("audio_onset") for reason in why)
        and abs(float(aud_frame) - tb) <= AUDIO_RADIUS
    ):
        tb = float(aud_frame)
        cxL, cyL, f0L = c["linL"]
        px = float(cxL[0] * (tb - f0L) + cxL[1])
        py = float(cyL[0] * (tb - f0L) + cyL[1])
        fi = int(round(tb))
        cx, cy = proj.to_court(px, py, fi)
        tape_px, _v = tape_pixel_dist(proj, px, py, fi)
        pos_src = "audio_onset_incoming"

    c.update(
        {
            "type": vtype,
            "frame": tb,
            "img_x": px,
            "img_y": py,
            "court_x": cx,
            "court_y": cy,
            "pos_src": pos_src,
            "audio": aud,
            "audio_frame": aud_frame,
            "audio_x": audio_px,
            "audio_y": audio_py,
            "audio_court_x": audio_cx,
            "audio_court_y": audio_cy,
            "audio_tape_px": audio_tape_px,
            "sequence_audio": sequence_aud,
            "sequence_audio_frame": sequence_aud_frame,
            "sequence_audio_x": sequence_px,
            "sequence_audio_y": sequence_py,
            "sequence_audio_court_x": sequence_cx,
            "sequence_audio_court_y": sequence_cy,
            "sequence_audio_tape_px": sequence_tape_px,
            "tape_px": tape_px,
            "observed_support_px": observed_support_px,
            **track_context,
            "reach": None if reach is None else reach[0],
            "box": box,
            "horiz_rev": horiz_rev,
            "vert_rev": vert_rev,
            "big_gain": big_gain,
            "collapse": collapse,
            "at_height": at_height,
            "why": why,
        }
    )
    c["confidence"] = _confidence(vtype, c, aud, tape_px, reach)
    return c


def _local_track_context(track, frame):
    """Summarize whether observed motion persists smoothly around a candidate."""

    def summarize(samples):
        if len(samples) < 2:
            return 0.0, 0.0, 0.0
        path = float(np.sum(np.hypot(np.diff(samples[:, 1]), np.diff(samples[:, 2]))))
        displacement = float(
            np.hypot(
                samples[-1, 1] - samples[0, 1],
                samples[-1, 2] - samples[0, 2],
            )
        )
        return displacement, path, displacement / max(path, 1e-6)

    output = {}
    if track is None or not len(track):
        for direction in ("pre", "post"):
            for window_ms in (200, 400):
                prefix = f"{direction}_{window_ms}ms"
                output[f"{prefix}_observations"] = 0
                output[f"{prefix}_displacement_px"] = 0.0
                output[f"{prefix}_path_px"] = 0.0
                output[f"{prefix}_straightness"] = 0.0
        return output
    for direction, sign in (("pre", -1.0), ("post", 1.0)):
        for window_ms, radius in ((200, 10.0), (400, 20.0)):
            delta = sign * (track[:, 0] - frame)
            samples = track[(delta >= 0.0) & (delta <= radius)]
            samples = samples[np.argsort(samples[:, 0])]
            displacement, path, straightness = summarize(samples)
            prefix = f"{direction}_{window_ms}ms"
            output[f"{prefix}_observations"] = int(len(samples))
            output[f"{prefix}_displacement_px"] = displacement
            output[f"{prefix}_path_px"] = path
            output[f"{prefix}_straightness"] = straightness
    return output


def _audio_peak(scores, tb, radius=AUDIO_RADIUS):
    if scores is None:
        return float("nan"), int(round(tb))
    values = [
        (audio_at(scores, frame), frame)
        for frame in range(int(round(tb)) - radius, int(round(tb)) + radius + 1)
    ]
    values = [(value, frame) for value, frame in values if not np.isnan(value)]
    if not values:
        return float("nan"), int(round(tb))
    value, frame = max(values)
    return float(value), int(frame)


def _confidence(vtype, c, aud, tape_px, reach):
    if vtype == "tracking_artifact":
        return 0.2
    conf = 0.5
    if vtype == "racket_hit":
        if (
            c["big_gain"]
            or c["horiz_rev"]
            or "occluded_redirect" in c["why"]
            or "audio_redirect" in c["why"]
        ):
            conf = 0.8
        if not np.isnan(aud) and aud >= AUDIO_HIT:
            conf = min(0.95, conf + 0.15)
        if reach is not None:
            conf = min(0.95, conf + 0.05)
    elif vtype == "bounce":
        conf = 0.7
        if not np.isnan(aud) and aud < 15:
            conf = min(0.9, conf + 0.1)  # quiet = consistent with a bounce
    elif vtype == "net_hit":
        conf = 0.75
    return round(float(conf), 3)


def _specialized_bounce_witnesses(frames, xs, ys, proj, boxes, pose):
    """Propose ground bounces from the event-specific vertical-deceleration measurement."""
    witnesses = []
    for run in split_runs(frames, xs, ys):
        witnesses.extend(
            detect_bounce_witnesses(np.asarray(run), frames, xs, ys, proj, boxes)
        )
    for witness in witnesses:
        classify_bounce_witness(witness, boxes, proj, pose=pose)
        disambiguate_net_bounce(witness, proj)
    return [witness for witness in witnesses if witness["verdict"] == "bounce"]


def _attach_bounce_witness(candidates, witness, scores):
    """Attach a specialized proposal to the nearest unified candidate, or preserve it alone."""
    target = min(
        candidates,
        key=lambda candidate: abs(candidate["frame"] - witness["frame"]),
        default=None,
    )
    if target is None or abs(target["frame"] - witness["frame"]) > SUPPRESS:
        audio, _ = _audio_peak(scores, witness["frame"])
        reach = None
        target = {
            "kind": "bounce_witness",
            "fL": witness["frame"],
            "fF": witness["frame"],
            "vxb": witness["vx_before"],
            "vyb": witness["vy_before"],
            "vxa": witness["vx_after"],
            "vya": witness["vy_after"],
            "sb": witness["sp_before"],
            "sa": witness["sp_after"],
            "ang": _angle(
                witness["vx_before"],
                witness["vy_before"],
                witness["vx_after"],
                witness["vy_after"],
            ),
            "type": "tracking_artifact",
            "frame": witness["frame"],
            "img_x": witness["img_x"],
            "img_y": witness["img_y"],
            "court_x": witness["court_x"],
            "court_y": witness["court_y"],
            "confidence": 0.2,
            "pos_src": "bounce_parabola_proposal",
            "audio": audio,
            "tape_px": witness.get("tape_px", float("nan")),
            "reach": reach,
            "box": witness.get("box_side"),
            "horiz_rev": bool(witness["horiz_reverse"]),
            "vert_rev": witness["vy_before"] * witness["vy_after"] < 0,
            "big_gain": False,
            "collapse": witness["sp_after"] <= COLLAPSE_FRAC * witness["sp_before"],
            "at_height": False,
            "why": ["bounce_witness_pending"],
        }
        candidates.append(target)
    current = target.get("bounce_witness")
    if current is None or witness["confidence"] > current["confidence"]:
        target["bounce_witness"] = witness
        target["proposal_type"] = "bounce"
        target["proposal_frame"] = witness["frame"]
        target["proposal_x"] = witness["img_x"]
        target["proposal_y"] = witness["img_y"]
        target["proposal_confidence"] = witness["confidence"]
        target["proposal_reason"] = "vertical_deceleration_arc%d" % witness["arc"]


# ------------------------------------------------------------------ orchestration
def detect_events(
    clip,
    camera_file,
    ball_file,
    box_file,
    pose_file=None,
    audio_file=None,
    serve_frames=None,
    fps=REFERENCE_FPS,
):
    native_frames, native_xs, native_ys, _ = load_track(ball_file, clip)
    native_proj = projector_for_artifact(camera_file, clip, ball_file)
    native_boxes = load_boxes(box_file, clip)
    if pose_file is None:
        cand = os.path.join(os.path.dirname(ball_file), "player_pose_kp17_%s_v1.csv" % clip)
        pose_file = cand if os.path.exists(cand) else None
    native_pose = _load_pose_for_gate(pose_file, clip)
    scores = load_audio(audio_file, clip)
    scale = _reference_scale(fps)
    frames, xs, ys, observed = _reference_track(
        native_frames,
        native_xs,
        native_ys,
        fps,
    )
    proj = _reference_projector(native_proj, fps)
    boxes = _reference_frame_map(native_boxes, fps)
    pose = _reference_frame_map(native_pose, fps)
    serve_frames = [float(frame) * scale for frame in serve_frames or []]

    cands = _candidates(frames, xs, ys)
    for c in cands:
        _classify(c, proj, boxes, pose, scores, track=observed)

    # de-duplicate: same type within SUPPRESS frames -> keep the higher-confidence one.
    cands.sort(key=lambda c: c["frame"])
    dedup = []
    for c in cands:
        if (
            dedup
            and abs(c["frame"] - dedup[-1]["frame"]) <= SUPPRESS
            and c["type"] == dedup[-1]["type"]
        ):
            if c["confidence"] > dedup[-1]["confidence"]:
                c["suppressed_candidate_frames"] = [
                    *dedup[-1].get("suppressed_candidate_frames", []),
                    dedup[-1]["frame"],
                ]
                dedup[-1] = c
            else:
                dedup[-1].setdefault("suppressed_candidate_frames", []).append(
                    c["frame"]
                )
            continue
        dedup.append(c)
    for witness in _specialized_bounce_witnesses(frames, xs, ys, proj, boxes, pose):
        _attach_bounce_witness(dedup, witness, scores)
    dedup.sort(key=lambda candidate: candidate["frame"])
    # strip private numpy payloads before returning
    for c in dedup:
        c.pop("run", None)
        c.pop("linL", None)
        c.pop("linR", None)
        c.pop("px_at", None)
    if abs(scale - 1.0) >= 1e-9:
        _scale_event_frames(dedup, 1.0 / scale)
    return dedup, native_proj


# ------------------------------------------------------------------ phase gating (live play only)
# Events only COUNT inside live play. A dribble push or a dead-ball dribble locally looks like a
# racket_hit (each push adds speed) -- which is exactly why phase gating exists. The serve strikes and
# the terminal event DEFINE the phases: pre-first-serve, between-serves dribble (near-stationary
# oscillation behind a baseline near the bouncing server, no net crossing), and post-terminal
# (the ball dying/being cleared after the point-ending event) are all EXCLUDED. This mirrors the
# gates track_arcs already applied to the old event source; they are reapplied ON TOP of the unified
# layer so both track_arcs and the contact-seed export consume the same in-play set.
DRIBBLE_EXTENT_PX = 42.0  # image bbox extent below which a pre-serve run is a stationary dribble
DRIBBLE_BASELINE_M = 1.0  # a run whose median court-y is within this of / beyond a baseline
BASELINE_CONTACT_M = 6.0  # a racket_hit this close to a baseline is a genuine cross-court shot
# (vs a dead-ball dribble stuck near the net) -- used to find the terminal
NET_AFTER_BOUNCE_FRAMES = 30.0  # a terminal bounce cannot reach the net without another strike
FAR_SPEED_GAIN_MARGIN_M = 8.0  # unsupported speed-only events this far beyond a baseline abstain
FAR_SPEED_GAIN_AUDIO_MIN = 3.0
SAME_SIDE_DUP_MAX_FRAMES = 20.0
SERVE_FIRST_BOUNCE_MAX_FRAMES = 32.0


def _event_side(event):
    return event.get("reach") or ("near" if event["court_y"] < NET_Y else "far")


def _promote_bounce_witness(event, reason):
    witness = event["bounce_witness"]
    frame = witness["frame"]
    pos_src = "bounce_parabola_vertex"
    if witness["confidence"] < 0.8 and abs(frame - event["frame"]) > 1.0:
        frame = 0.5 * (frame + event["frame"])
        pos_src = "bounce_kink_parabola_consensus"
    event.update(
        {
            "type": "bounce",
            "frame": frame,
            "img_x": witness["img_x"],
            "img_y": witness["img_y"],
            "court_x": witness["court_x"],
            "court_y": witness["court_y"],
            "confidence": max(0.8, witness["confidence"]),
            "pos_src": pos_src,
            "tape_px": witness.get("tape_px", event.get("tape_px", float("nan"))),
            "reach": None,
            "at_height": False,
            "why": [
                reason,
                "vertical_deceleration%.1f" % witness["kink_strength"],
                "ballistic_arc%d" % witness["arc"],
            ],
            "review": False,
        }
    )


def _abstain_as_proposal(event, proposal_type, reason):
    event.update(
        {
            "proposal_type": proposal_type,
            "proposal_frame": event["frame"],
            "proposal_x": event["img_x"],
            "proposal_y": event["img_y"],
            "proposal_confidence": event["confidence"],
            "proposal_reason": reason,
            "type": "tracking_artifact",
            "confidence": 0.2,
            "why": [*event.get("why", []), reason],
            "review": True,
            "review_reason": reason,
        }
    )


def _validate_baseline_skims(events):
    """Require an independent ballistic witness for quantization-limited baseline bounces."""
    for event in events:
        if event["type"] != "bounce" or "baseline_skim" not in event.get("why", []):
            continue
        if event.get("bounce_witness") is not None:
            _promote_bounce_witness(event, "baseline_skim_witnessed")
        else:
            _abstain_as_proposal(event, "bounce", "unsupported_baseline_skim")


def _validate_net_hits(events):
    """A tape-near slowdown must also show a directional response or independent evidence."""
    for event in events:
        if event["type"] != "net_hit":
            continue
        reasons = event.get("why", [])
        audio = event.get("audio", float("nan"))
        supported = (
            "drop_off_tape" in reasons
            or event.get("ang", 0.0) >= 25.0
            or (np.isfinite(audio) and audio >= AUDIO_HIT)
        )
        if not supported:
            _abstain_as_proposal(event, "net_hit", "unsupported_net_slowdown")


def _is_weak_wrist_contact(event):
    return event["type"] == "racket_hit" and any(
        reason.startswith("wrist_reach") for reason in event.get("why", [])
    )


def _validate_weak_wrist_contacts(events, serve_frames):
    """Emit wrist-only redirects only when the surrounding rally sequence requires a hit."""
    strong_physical = sorted(
        (
            event
            for event in events
            if event["type"] in ("racket_hit", "bounce", "net_hit")
            and not _is_weak_wrist_contact(event)
        ),
        key=lambda event: event["frame"],
    )
    for event in sorted(
        (item for item in events if _is_weak_wrist_contact(item)),
        key=lambda item: item["frame"],
    ):
        if any(abs(event["frame"] - frame) <= 4.0 for frame in serve_frames):
            continue
        prior = next(
            (
                item
                for item in reversed(strong_physical)
                if item["frame"] < event["frame"] - 2.0
            ),
            None,
        )
        following = next(
            (
                item
                for item in strong_physical
                if item["frame"] > event["frame"] + 2.0
            ),
            None,
        )
        supported = (
            prior is not None
            and prior["type"] == "bounce"
            and _event_side(prior) == _event_side(event)
            and following is not None
            and _event_side(following) != _event_side(event)
        )
        if supported:
            sequence_audio = event.get("sequence_audio", float("nan"))
            if (
                event.get("ang", float("inf")) < ANGLE_MIN
                and np.isfinite(sequence_audio)
                and sequence_audio >= SEQUENCE_WRIST_AUDIO_MIN
                and abs(event["sequence_audio_frame"] - event["frame"])
                <= SEQUENCE_AUDIO_RADIUS
            ):
                event.update(
                    {
                        "frame": float(event["sequence_audio_frame"]),
                        "img_x": event["sequence_audio_x"],
                        "img_y": event["sequence_audio_y"],
                        "court_x": event["sequence_audio_court_x"],
                        "court_y": event["sequence_audio_court_y"],
                        "tape_px": event["sequence_audio_tape_px"],
                        "pos_src": "sequence_audio_onset_incoming",
                    }
                )
            event["why"] = [*event.get("why", []), "sequence_supported_wrist"]
            event["confidence"] = max(0.65, event["confidence"])
        else:
            _abstain_as_proposal(event, "contact", "unsupported_wrist_contact")


def _sequence_contact(event, serve_frames):
    if event["type"] != "racket_hit":
        return False
    serve = any(abs(event["frame"] - frame) <= 4.0 for frame in serve_frames)
    physical = any(
        reason == "horiz_reverse"
        or reason == "occluded_redirect"
        or reason == "audio_redirect"
        or reason.startswith("speed_gain")
        or reason.startswith("wrist_reach")
        or reason == "sequence_required_contact"
        for reason in event.get("why", [])
    )
    spatial = (
        event.get("reach") is not None
        or event.get("at_height")
        or min(abs(event["court_y"] - BASELINE_NEAR), abs(event["court_y"] - BASELINE_FAR))
        < BASELINE_CONTACT_M
    )
    return serve or (
        event["confidence"] >= 0.6
        and physical
        and spatial
    )


def _decode_bounce_sequence(events, serve_frames):
    """Promote specialized bounce proposals only when a struck flight requires that bounce."""
    contacts = sorted(
        (event for event in events if _sequence_contact(event, serve_frames)),
        key=lambda event: event["frame"],
    )
    for contact in contacts:
        if contact["type"] != "racket_hit":
            continue
        following_contacts = [
            event
            for event in contacts
            if event["type"] == "racket_hit" and event["frame"] > contact["frame"] + 2.0
        ]
        next_contact = following_contacts[0] if following_contacts else None
        flight_hi = min(
            contact["frame"] + 90.0,
            next_contact["frame"] - 2.0 if next_contact else float("inf"),
        )
        accepted = [
            event
            for event in events
            if event["type"] in ("bounce", "net_hit")
            and contact["frame"] + 2.0 < event["frame"] <= flight_hi
        ]
        if accepted:
            continue
        expected_side = "far" if _event_side(contact) == "near" else "near"
        proposals = [
            event
            for event in events
            if event.get("bounce_witness") is not None
            and event["type"] != "bounce"
            and contact["frame"] + 2.0 < event["proposal_frame"] <= flight_hi
            and (
                "near"
                if event["bounce_witness"]["court_y"] < NET_Y
                else "far"
            )
            == expected_side
        ]
        if not proposals:
            continue
        proposal = min(proposals, key=lambda event: event["proposal_frame"])
        reason = (
            "sequence_serve_bounce"
            if any(abs(contact["frame"] - frame) <= 4.0 for frame in serve_frames)
            else "sequence_flight_bounce"
        )
        _promote_bounce_witness(proposal, reason)


def _enforce_serve_bounce(events, serve_frames, frames, xs, ys, proj):
    """A legal serve must bounce before the receiver can strike it."""
    for serve_frame in serve_frames:
        serve_events = [
            event
            for event in events
            if event["type"] == "racket_hit" and abs(event["frame"] - serve_frame) <= 4.0
        ]
        if not serve_events:
            continue
        serve = min(serve_events, key=lambda event: abs(event["frame"] - serve_frame))
        following = sorted(
            (
                event
                for event in events
                if event["frame"] > serve["frame"] + 2.0
                and event["type"] in ("racket_hit", "bounce", "net_hit")
            ),
            key=lambda event: event["frame"],
        )
        if not following:
            continue
        event = following[0]
        if (
            event["type"] != "racket_hit"
            or event["frame"] - serve["frame"] > SERVE_FIRST_BOUNCE_MAX_FRAMES
            or event.get("at_height")
            or _event_side(event) == _event_side(serve)
        ):
            continue
        window = (frames >= event["frame"] - 5.0) & (frames <= event["frame"])
        indices = np.flatnonzero(window)
        if indices.size:
            max_y = float(np.max(ys[indices]))
            candidates = indices[ys[indices] >= max_y - 0.2]
            track_index = int(candidates[-1])
            event["frame"] = float(frames[track_index])
            event["img_x"] = float(xs[track_index])
            event["img_y"] = float(ys[track_index])
            event["court_x"], event["court_y"] = proj.to_court(
                event["img_x"], event["img_y"], int(round(event["frame"]))
            )
            event["tape_px"], _ = tape_pixel_dist(
                proj, event["img_x"], event["img_y"], int(round(event["frame"]))
            )
            event["pos_src"] = "serve_bounce_track_apex"
        event["type"] = "bounce"
        event["confidence"] = 0.8
        event["why"] = ["serve_must_bounce"]


def _suppress_same_side_duplicates(events):
    """A player cannot strike twice before the opponent has touched the ball.

    Keep this as a narrow duplicate gate: only adjacent hit proposals from the same inferred
    player within a physically implausible 20-frame interval are compared. The stronger proposal
    wins; an exact tie keeps the earlier event.
    """
    kept = []
    for event in sorted(
        (item for item in events if item["type"] == "racket_hit"),
        key=lambda item: item["frame"],
    ):
        if (
            kept
            and event["frame"] - kept[-1]["frame"] <= SAME_SIDE_DUP_MAX_FRAMES
            and _event_side(event) == _event_side(kept[-1])
        ):
            prior = kept[-1]
            winner, loser = (
                (event, prior) if event["confidence"] > prior["confidence"] else (prior, event)
            )
            loser["type"] = "tracking_artifact"
            loser["confidence"] = 0.2
            loser["why"] = [
                *loser.get("why", []),
                "same_side_duplicate",
            ]
            if winner is event:
                kept[-1] = event
            continue
        kept.append(event)


def _repair_missing_contacts_between_bounces(events, emit=True):
    """Recover a required strike from an incomplete but otherwise physical rally sequence.

    Without a serve anchor, preserve the same sequence evidence as a review proposal rather than
    emitting it: the clip may include dead-ball motion that cannot be separated from live play.
    """

    def candidate_score(event):
        local_audio = event.get("audio", float("nan"))
        wide_audio = event.get("sequence_audio", float("nan"))
        audio = max(
            0.0 if not np.isfinite(local_audio) else local_audio,
            0.0 if not np.isfinite(wide_audio) else wide_audio,
        )
        return (
            (15.0 if event.get("reach") is not None else 0.0)
            + (20.0 if event.get("at_height") else 0.0)
            + min(audio, 40.0)
            + min(event.get("ang", 0.0), 180.0) / 4.5
        )

    for _ in range(3):
        repaired = False
        physical = sorted(
            (
                event
                for event in events
                if event.get("in_play")
                and event["type"] in ("racket_hit", "bounce", "net_hit")
            ),
            key=lambda event: event["frame"],
        )
        for before, after in zip(physical, physical[1:]):
            before_side = _event_side(before)
            after_side = _event_side(after)
            required_side = None
            if before["type"] == "bounce" and after_side != before_side:
                required_side = before_side
            elif (
                before["type"] == "racket_hit"
                and after["type"] in ("bounce", "net_hit")
                and after_side == before_side
            ):
                required_side = "far" if before_side == "near" else "near"
            if required_side is None:
                continue
            candidates = [
                event
                for event in events
                if event.get("in_play")
                and event["type"] == "tracking_artifact"
                and before["frame"] + 3.0 < event["frame"] < after["frame"] - 3.0
                and _event_side(event) == required_side
                and "unsupported_baseline_skim" not in event.get("why", [])
            ]
            if not candidates:
                continue
            event = max(candidates, key=candidate_score)
            score = candidate_score(event)
            wide_audio = event.get("sequence_audio", float("nan"))
            if score < 25.0 and (not np.isfinite(wide_audio) or wide_audio < 10.0):
                continue
            if not emit:
                if event.get("proposal_reason") != "unanchored_sequence_contact":
                    _abstain_as_proposal(
                        event, "contact", "unanchored_sequence_contact"
                    )
                repaired = True
                continue
            local_audio = event.get("audio", float("nan"))
            use_wide_audio = (
                np.isfinite(wide_audio)
                and wide_audio >= 5.0
                and (
                    not np.isfinite(local_audio)
                    or wide_audio >= local_audio + 2.0
                )
            )
            prefix = "sequence_audio" if use_wide_audio else "audio"
            anchor_audio = wide_audio if use_wide_audio else local_audio
            if (
                event.get("ang", float("inf")) < ANGLE_MIN
                and np.isfinite(anchor_audio)
                and anchor_audio >= 5.0
            ):
                event.update(
                    {
                        "frame": float(event[f"{prefix}_frame"]),
                        "img_x": event[f"{prefix}_x"],
                        "img_y": event[f"{prefix}_y"],
                        "court_x": event[f"{prefix}_court_x"],
                        "court_y": event[f"{prefix}_court_y"],
                        "tape_px": event[f"{prefix}_tape_px"],
                        "pos_src": "sequence_required_audio_incoming",
                    }
                )
            event["type"] = "racket_hit"
            event["confidence"] = min(0.85, 0.6 + score / 200.0)
            event["why"] = ["sequence_required_contact", f"evidence_score{score:.0f}"]
            event["review"] = False
            repaired = True
        if not emit:
            break
        if not repaired:
            break


def _dribble_runs(frames, xs, ys, proj, serve_frames):
    """Between-serves dribble run spans [(f0,f1),...]: a near-stationary oscillation behind a baseline
    (small image extent, median court-y beyond a baseline) that still has a serve strike AHEAD of it
    (the server bouncing the ball before the next serve). Reuses track_arcs.pre_serve_dribble's logic."""
    out = []
    for run in split_runs(frames, xs, ys):
        r = np.array(run)
        rf = frames[r].astype(float)
        rx = xs[r]
        ry = ys[r]
        # A near-stationary oscillation BEHIND a baseline is never live play (the ball is always in
        # flight during a rally) -- it is the server bouncing the ball before a serve OR the ball
        # dying after the point. Flag it regardless of whether a serve strike is detected ahead
        # (robust to serve-detection gaps: the between-serves dribble is caught even if the following
        # serve was missed).
        ext = float(max(np.ptp(rx), np.ptp(ry)))
        if ext >= DRIBBLE_EXTENT_PX:
            continue
        cys = [proj.to_court(float(rx[i]), float(ry[i]), int(rf[i]))[1] for i in range(len(rf))]
        cy_med = float(np.median(cys))
        if (
            cy_med > BASELINE_FAR - DRIBBLE_BASELINE_M
            or cy_med < BASELINE_NEAR + DRIBBLE_BASELINE_M
        ):
            out.append((float(rf[0]), float(rf[-1])))
    return out


def _gate_play_phase_reference(events, serve_frames, frames, xs, ys, proj):
    """Annotate each event with in_play (bool) + excl (reason) and return (play_lo, play_hi, terminal).

    LIVE play is the LAST serve's point: [first serve strike, the terminal event]. The terminal is the
    earliest net/bounce after the last serve with NO genuine cross-court contact after it (a genuine
    contact is a racket_hit near a baseline -- the dead-ball nets that follow the point sit stuck near
    the net, so they are NOT terminal candidates and are correctly excluded). Serve strikes are always
    kept (incl. a fault strike the owner labels); everything before the last serve that is not a serve
    strike is pre-serve / between-serves dribble / fault junk; everything after the terminal is dead."""
    for event in events:
        reasons = event.get("why", [])
        speed_only = any(reason.startswith("speed_gain") for reason in reasons) and not any(
            reason == "horiz_reverse"
            or reason == "occluded_redirect"
            or reason.startswith("wrist_reach")
            for reason in reasons
        )
        far_outside = (
            event["court_y"] < BASELINE_NEAR - FAR_SPEED_GAIN_MARGIN_M
            or event["court_y"] > BASELINE_FAR + FAR_SPEED_GAIN_MARGIN_M
        )
        audio = event.get("audio", float("nan"))
        quiet = not np.isfinite(audio) or audio < FAR_SPEED_GAIN_AUDIO_MIN
        serve_supported = any(abs(event["frame"] - frame) <= 4.0 for frame in serve_frames)
        if (
            event["type"] == "racket_hit"
            and speed_only
            and event.get("reach") is None
            and far_outside
            and quiet
            and not serve_supported
        ):
            event["type"] = "tracking_artifact"
            event["confidence"] = 0.2
            event["why"] = [*reasons, "unsupported_far_speed_gain"]

        unsupported_held_ball = (
            event["type"] == "racket_hit"
            and event.get("kind") == "gap"
            and event.get("sb", float("inf")) < DEPART_SLOW
            and event.get("reach") is None
            and quiet
            and not serve_supported
        )
        if unsupported_held_ball:
            event["type"] = "tracking_artifact"
            event["confidence"] = 0.2
            event["why"] = [*event.get("why", []), "unsupported_held_ball_departure"]

    _enforce_serve_bounce(events, serve_frames, frames, xs, ys, proj)
    _validate_baseline_skims(events)
    _validate_net_hits(events)
    _decode_bounce_sequence(events, serve_frames)
    _validate_weak_wrist_contacts(events, serve_frames)
    _decode_bounce_sequence(events, serve_frames)
    _suppress_same_side_duplicates(events)

    if not serve_frames:
        for e in events:
            e["in_play"] = True
            e["excl"] = ""
            if (
                e.get("proposal_type") == "bounce"
                and e["type"] == "tracking_artifact"
            ):
                e["review"] = True
                e["review_reason"] = "unresolved_bounce_witness"
        lo = float(frames[0]) if len(frames) else 0.0
        hi = float(frames[-1]) if len(frames) else 0.0
        _repair_missing_contacts_between_bounces(events, emit=False)
        _decode_bounce_sequence(events, serve_frames)
        _repair_missing_contacts_between_bounces(events, emit=False)
        _decode_bounce_sequence(events, serve_frames)
        return lo, hi, None
    first_serve, last_serve = min(serve_frames), max(serve_frames)
    dribbles = _dribble_runs(frames, xs, ys, proj, serve_frames)

    def genuine_contact(e):
        physical = any(
            reason == "horiz_reverse"
            or reason == "occluded_redirect"
            or reason == "audio_redirect"
            or reason.startswith("speed_gain")
            or reason.startswith("wrist_reach")
            or reason == "sequence_required_contact"
            for reason in e.get("why", [])
        )
        recent_bounce = any(
            prior["type"] == "bounce" and 0.0 < e["frame"] - prior["frame"] <= 40.0
            for prior in events
        )
        spatial = (
            e.get("reach") is not None
            or (e.get("at_height") and recent_bounce)
            or min(abs(e["court_y"] - BASELINE_NEAR), abs(e["court_y"] - BASELINE_FAR))
            < BASELINE_CONTACT_M
        )
        return e["type"] == "racket_hit" and e["confidence"] >= 0.6 and physical and spatial

    terminal = None
    planes = sorted(
        (
            e
            for e in events
            if e["type"] in ("net_hit", "bounce") and e["frame"] >= last_serve - 3.0
        ),
        key=lambda e: e["frame"],
    )
    for e in planes:
        if e["type"] == "bounce" and any(
            later["type"] == "net_hit"
            and e["frame"] < later["frame"] <= e["frame"] + NET_AFTER_BOUNCE_FRAMES
            for later in planes
        ):
            continue
        if not any(g["frame"] > e["frame"] + 2.0 and genuine_contact(g) for g in events):
            terminal = e["frame"]
            break
    if terminal is None:
        gs = [e["frame"] for e in events if genuine_contact(e) and e["frame"] >= last_serve - 3.0]
        terminal = max(gs) if gs else float(frames[-1])
    play_lo, play_hi = first_serve - 3.0, terminal + 2.0
    for e in events:
        f = e["frame"]
        if any(abs(f - sf) <= 3.0 for sf in serve_frames):
            reason = ""  # a serve strike is always live
        elif f < first_serve - 3.0:
            reason = "pre_serve"
        elif f > terminal + 2.0:
            reason = "post_terminal"
        elif any(a <= f <= b for a, b in dribbles):
            reason = "dribble"
        elif f < last_serve - 3.0:
            reason = "between_serves"
        else:
            reason = ""
        e["in_play"] = reason == ""
        e["excl"] = reason
        if (
            e.get("proposal_type") == "bounce"
            and e["type"] == "tracking_artifact"
            and last_serve - 3.0 <= e.get("proposal_frame", e["frame"]) <= terminal + 2.0
            and reason not in ("pre_serve", "between_serves", "dribble")
        ):
            e["review"] = True
            e["review_reason"] = "unresolved_bounce_witness"
    _repair_missing_contacts_between_bounces(events)
    _decode_bounce_sequence(events, serve_frames)
    _repair_missing_contacts_between_bounces(events)
    _decode_bounce_sequence(events, serve_frames)
    return play_lo, play_hi, terminal


def gate_play_phase(
    events,
    serve_frames,
    frames,
    xs,
    ys,
    proj,
    fps=REFERENCE_FPS,
):
    """Run phase decoding in calibrated 50 Hz units and return native frames."""
    scale = _reference_scale(fps)
    if abs(scale - 1.0) < 1e-9:
        return _gate_play_phase_reference(
            events,
            serve_frames,
            frames,
            xs,
            ys,
            proj,
        )
    reference_frames, reference_xs, reference_ys, _ = _reference_track(
        frames,
        xs,
        ys,
        fps,
    )
    reference_projector = _reference_projector(proj, fps)
    reference_serves = [float(frame) * scale for frame in serve_frames]
    _scale_event_frames(events, scale)
    try:
        play_lo, play_hi, terminal = _gate_play_phase_reference(
            events,
            reference_serves,
            reference_frames,
            reference_xs,
            reference_ys,
            reference_projector,
        )
    finally:
        _scale_event_frames(events, 1.0 / scale)
    return (
        play_lo / scale,
        play_hi / scale,
        None if terminal is None else terminal / scale,
    )


COLS = [
    "clip",
    "frame",
    "type",
    "img_x",
    "img_y",
    "court_x",
    "court_y",
    "confidence",
    "pos_src",
    "audio",
    "tape_px",
    "reach",
    "in_play",
    "excl",
    "why",
    "proposal_type",
    "proposal_frame",
    "proposal_x",
    "proposal_y",
    "proposal_confidence",
    "proposal_reason",
    "review",
    "review_reason",
]


def write_csv(path, clip, events):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for c in events:
            w.writerow(
                [
                    clip,
                    round(c["frame"], 2),
                    c["type"],
                    round(c["img_x"], 1),
                    round(c["img_y"], 1),
                    round(c["court_x"], 2),
                    round(c["court_y"], 2),
                    c["confidence"],
                    c["pos_src"],
                    "" if np.isnan(c["audio"]) else round(c["audio"], 1),
                    round(c["tape_px"], 1),
                    c["reach"] or "",
                    int(c.get("in_play", True)),
                    c.get("excl", ""),
                    "|".join(c["why"]),
                    c.get("proposal_type", ""),
                    ""
                    if c.get("proposal_frame") is None
                    else round(c["proposal_frame"], 2),
                    "" if c.get("proposal_x") is None else round(c["proposal_x"], 1),
                    "" if c.get("proposal_y") is None else round(c["proposal_y"], 1),
                    c.get("proposal_confidence", ""),
                    c.get("proposal_reason", ""),
                    int(c.get("review", False)),
                    c.get("review_reason", ""),
                ]
            )
    os.replace(tmp, path)


def serve_strike_frames(
    clip,
    camera_file,
    ball_file,
    box_file,
    pose_file,
    audio_file,
    serve_hints,
    fps=50.0,
):
    """Serve strike frames from serve_contact_detect (used to tag which racket_hits are serves)."""
    import serve_contact_detect as scd

    out = []
    for h in serve_hints or []:
        r = scd.detect_serve(
            clip,
            camera_file,
            ball_file,
            box_file,
            h,
            pose_file,
            audio_file,
            fps=fps,
        )
        if not r.get("error"):
            out.append(float(r["strike_frame"]))
    return sorted(out)


def export_contacts_witnessed(match, clip, events, out_path, serve_frames=None):
    """Rewrite contacts_witnessed_<clip>.json from the unified racket_hits (Q0e). This is the CONTACT
    seed source the labeler + review pages read; regenerating it from ball_events is what drops the
    five ex-false-contacts and the net event from the contact list and re-times/re-locates the rest,
    so the seed list itself becomes the visible proof of the revamp.

    HONEST SCOPE. This revamp fixes DETECTION + TYPING + 2D placement + sub-frame timing. The 3D lift
    (base_xyz/fused_xyz/height) is a downstream stage (Q0d) not re-run here, so those fields are null
    and tier='abstained' with a reason -- the ray witness (incoming-authoritative contact pixel) and
    the audio witness are populated; the review card shows the 2D truth honestly."""
    import json

    serve_frames = serve_frames or []
    # PHASE GATE (applied on top of ball_events): only in-play racket_hits become contact seeds.
    # Callers must have run gate_play_phase(...) to set e["in_play"]; if absent (no serve info),
    # everything is treated as in-play.
    contacts = []
    for e in events:
        if e["type"] != "racket_hit":
            continue
        if not e.get("in_play", True):
            continue
        f = e["frame"]
        is_serve = any(abs(f - sf) <= 3.0 for sf in serve_frames)
        hitter = e.get("reach")
        if hitter is None:
            hitter = "near" if e["court_y"] < NET_Y else "far"
        aud = e.get("audio")
        contacts.append(
            {
                "frame": round(f, 2),
                "anchor_frame": int(round(f)),
                "phase": "serve" if is_serve else "rally",
                "hitter": hitter,
                "base_xyz": None,
                "base_flags": ["q0e_2d_event"],
                "fused_xyz": None,
                "tier": "abstained",
                "confidence": e["confidence"],
                "depth_source": "none",
                "height_m": None,
                "court_xy": None,
                "horiz_offset_m": None,
                "window_min_offset_m": None,
                "other_side_residual_m": None,
                "witnesses": {
                    "ray": {
                        "px540": [round(e["img_x"], 1), round(e["img_y"], 1)],
                        "reproj_px": 0.0,
                        "note": "incoming-authoritative contact pixel (%s); 3D lies on this ray"
                        % e["pos_src"],
                    },
                    "audio": (
                        {
                            "present": True,
                            "onset_frame": round(f, 1),
                            "peak_score": round(aud, 1),
                            "confirms": bool(aud >= AUDIO_HIT),
                            "note": "audio onset peak %.0f %s the strike"
                            % (aud, "CONFIRMS" if aud >= AUDIO_HIT else "is weak at"),
                        }
                        if aud is not None and not np.isnan(aud)
                        else {"present": False}
                    ),
                },
                "agreements": {},
                "reason": "Q0e detection revamp: 2D event typed (%s) + sub-frame "
                "timed; 3D lift pending re-run" % "|".join(e["why"]),
            }
        )
    counts = {"pinned": 0, "bounded": 0, "abstained": len(contacts)}
    doc = {"match": match, "clip": clip, "counts": counts, "contacts": contacts}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, out_path)
    return len(contacts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="rg2025f")
    ap.add_argument("--clip", default="pt0026")
    ap.add_argument("--camera-file", required=True)
    ap.add_argument("--ball-file", default=None)
    ap.add_argument("--box-file", default=None)
    ap.add_argument("--pose-file", default=None)
    ap.add_argument("--audio-file", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--serve-hints", default=None, help="comma serve hint frames for serve tagging")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument(
        "--export-witnessed",
        action="store_true",
        help="rewrite contacts_witnessed_<clip>.json from the racket_hits",
    )
    ap.add_argument(
        "--witnessed-output",
        default=None,
        help="explicit contacts_witnessed JSON path; defaults beside --out-csv",
    )
    args = ap.parse_args()
    proc = os.path.join(HERE, "..", "..", "data", "processed", args.match)
    ball = args.ball_file or os.path.join(proc, "ball_track_wasb_full_v1_decoded.csv")
    box = args.box_file or os.path.join(proc, "player_boxes_50_full_v2.csv")
    pose = args.pose_file or os.path.join(proc, "player_pose_kp17_%s_v1.csv" % args.clip)
    audio = args.audio_file or os.path.join(proc, "contact_audio_scores_16k_v1.npz")
    events, proj = detect_events(
        args.clip,
        args.camera_file,
        ball,
        box,
        pose if os.path.exists(pose) else None,
        audio if os.path.exists(audio) else None,
        fps=args.fps,
    )
    # phase gate: needs the serve strikes + the raw track. Serve hints -> strike frames.
    frames, xs, ys, _tid = load_track(ball, args.clip)
    boxes = load_boxes(box, args.clip)
    hints = (
        [float(s) for s in args.serve_hints.split(",")]
        if args.serve_hints
        else auto_serve_hints(frames, xs, ys, boxes, args.fps)
    )
    sfr = serve_strike_frames(
        args.clip,
        args.camera_file,
        ball,
        box,
        pose if os.path.exists(pose) else None,
        audio if os.path.exists(audio) else None,
        hints,
        fps=args.fps,
    )
    play_lo, play_hi, terminal = gate_play_phase(
        events,
        sfr,
        frames,
        xs,
        ys,
        proj,
        fps=args.fps,
    )
    print("play_window=[%.1f, %.1f] terminal=%s serves=%s" % (play_lo, play_hi, terminal, sfr))
    if args.out_csv:
        write_csv(args.out_csv, args.clip, events)
    if args.export_witnessed:
        outp = args.witnessed_output
        if outp is None and args.out_csv:
            outp = os.path.join(
                os.path.dirname(os.path.abspath(args.out_csv)),
                "contacts_witnessed_%s.json" % args.clip,
            )
        if outp is None:
            outp = os.path.join(proc, "contacts_witnessed_%s.json" % args.clip)
        n = export_contacts_witnessed(args.match, args.clip, events, outp, sfr)
        print("wrote %d in-play contacts -> %s" % (n, outp))
    print(
        "clip %s: %d events (%d in-play)"
        % (args.clip, len(events), sum(e.get("in_play", True) for e in events))
    )
    for c in events:
        print(
            "  f%-8.2f %-16s conf=%.2f court=(%6.2f,%6.2f) img=(%5.0f,%5.0f) aud=%5s tape=%5.1f %s"
            % (
                c["frame"],
                c["type"],
                c["confidence"],
                c["court_x"],
                c["court_y"],
                c["img_x"],
                c["img_y"],
                "n/a" if np.isnan(c["audio"]) else "%.0f" % c["audio"],
                c["tape_px"],
                "|".join(c["why"]),
            )
        )


if __name__ == "__main__":
    main()
