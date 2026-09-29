"""Approach B (bake-off W5): global trajectory fit-and-segment contact detector.

Fits piecewise 3D ballistic arcs (physics/flight.py, drag on; optional Magnus) to the
per-frame WASB ball candidates through each point's calibrated camera projection P.
Knots — the only places velocity may change discontinuously — are proposed from the
track itself (either-axis direction flips, acceleration spikes, tracking gaps; this is
the SEED-FREE primary mode) or additionally from the W1 candidate union (--seed-events,
the hedged secondary mode). Segments are refined by residual-driven splitting and
flight-consistency merging, so the final arcs satisfy the flight model globally.

Each knot is classified with the validated impact physics (physics/impact.py):
  bounce — at the court plane, vertical velocity reversal within restitution range;
  hit    — near a player box, redirection toward the opponent side, racket-feasible
           speed change (impact.racket_hit_feasible);
  noise  — neither.
Orphan arc starts (no feasible predecessor) that leave a player at speed are hits too;
this is what detects serves (toss -> drive) without any longitudinal reversal.

Outputs (standard event schema plus knot-feature and bounce-position tables):
  events csv:  clip, frame, kind(hit|bounce), track_id, u, v, player_distance_px,
               rms_px, confidence, source
  knots csv:   per-knot physics features (for Phase-3 fusion)
  bounces csv: bounce positions in court coordinates (P2 groundwork)

Example (seed-free, dev clips):
    .venv/bin/python cv/pipeline/physics_events.py \
        --out data/processed/rg2025f --outdir data/processed/bakeoff/B \
        --clips-file dev_clips.txt --tag devfree_v1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "physics"))

import flight  # noqa: E402
import impact  # noqa: E402
from camera_cal import project  # noqa: E402

FPS = 50.0
NET_Y = 23.77 / 2

# --- observation selection ---
SCORE_MIN = 0.45          # WASB per-frame candidate score gate
STATIC_SPEED = 0.6        # px/frame below which the ball is "held" (pre-serve, dead time)
STATIC_RUN = 25           # frames of sustained sub-static speed that get discarded

# --- segmentation ---
MIN_SEG = 8               # min frames per arc at 50 fps
GAP_BREAK = 20            # frames: force a segment break across longer tracking gaps
FLIP_SPEED = 0.8          # px/frame: min |d| on an axis for a direction flip to count
ACCEL_SPIKE = 1.6         # px/frame^2 on smoothed derivative to propose a knot
                          # (1.2 was dev-tested for far-court kinks: no measurable gain)
SPLIT_RMS = 2.2           # px: residual above which an arc is searched for a hidden knot
NOISE_RMS = 6.0           # px: arcs above this never produce events
MERGE_DV = 3.0            # m/s: junction below this is a candidate for arc merging
MAX_JUNCTION_GAP = 30     # frames: arcs further apart are not joined by a knot
JOIN_DIST = 4.0           # m: extrapolated arc endpoints must meet within this
MAX_NFEV = 60

# --- knot classification (court frame, meters, m/s) ---
BOUNCE_Z_MAX = 0.7        # fitted height tolerance for a court-plane bounce
BOUNCE_E_RANGE = (0.25, 1.35)   # observed -vz_out/vz_in vs model e_y ~0.6-0.85 (noisy)
BOUNCE_H_RATIO = (0.25, 1.10)   # horizontal speed ratio through the bounce
HIT_DV_MIN = 5.0          # m/s minimum impulse for a racket hit
HIT_SPEED_OUT_MIN = 9.0   # m/s: real shots leave the racket faster than prep bounces
HIT_SPEED_IN_MIN = 6.0    # m/s: incoming ball at a junction hit is a live ball
NMS_FRAMES = 8            # de-duplicate hit events closer than this
CHAIN_SPEED_MIN = 12.0    # m/s: a live-play chain contains at least one fast arc
CHAIN_OBS_MIN = 20        # frames: minimum observation support for a live chain
HIT_DIST_PX = 80.0        # image-space distance to the nearest player box
HIT_Z_MAX = 4.2           # racket contacts happen below this height
SERVE_SPEED_MIN = 10.0    # m/s: orphan-start arcs slower than this are not contacts
SERVE_DIST_PX = 120.0     # serve contact is above the box; allow a looser gate


def parse_frame(name: str) -> int:
    return int(name[2:6])


# ------------------------------- flight integration --------------------------------------


def integrate_states(x0, v0, w0, n: int, dt: float):
    """(n,3) positions and velocities at consecutive frame steps; dt may be negative."""
    xs = np.zeros((n, 3))
    vs = np.zeros((n, 3))
    x, v = np.array(x0, float), np.array(v0, float)
    w = np.array(w0, float)
    lift = bool(np.linalg.norm(w) > 1)
    xs[0], vs[0] = x, v
    for i in range(1, n):
        x, v, w = flight.rk4_step(x, v, w, dt, lift=lift)
        xs[i], vs[i] = x, v
    return xs, vs


def _ray_point(P, u, v, z):
    """Back-project pixel (u, v) to the world plane Z=z."""
    A = np.array(
        [
            [P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
            [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]],
        ]
    )
    b = -np.array(
        [
            P[0, 2] * z + P[0, 3] - u * (P[2, 2] * z + P[2, 3]),
            P[1, 2] * z + P[1, 3] - v * (P[2, 2] * z + P[2, 3]),
        ]
    )
    try:
        xy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([xy[0], xy[1], z])


class Arc:
    """A fitted flight arc over observation frames [f0, fe] (1-based clip frames)."""

    __slots__ = ("f0", "fe", "x0", "v0", "w", "rms", "n_obs", "inlier_frac", "uv0", "uve")

    def __init__(self, f0, fe, x0, v0, w, rms, n_obs, inlier_frac, uv0, uve):
        self.f0, self.fe = int(f0), int(fe)
        self.x0, self.v0, self.w = x0, v0, w
        self.rms = float(rms)
        self.n_obs = int(n_obs)
        self.inlier_frac = float(inlier_frac)
        self.uv0, self.uve = uv0, uve

    def state_at(self, f: float, fps: float | None = None):
        fps = FPS if fps is None else fps
        """3D position/velocity at clip frame f (extrapolates outside [f0, fe])."""
        steps = f - self.f0
        if abs(steps) < 1e-9:
            return np.array(self.x0), np.array(self.v0)
        n = int(abs(round(steps))) + 1
        dt = math.copysign(1.0 / fps, steps)
        xs, vs = integrate_states(self.x0, self.v0, self.w, n, dt)
        return xs[-1], vs[-1]


INIT_HEIGHTS = (0.1, 1.2, 2.5)   # multi-start ray heights (single-view depth ambiguity)

ANCHOR_W = 8.0    # px per meter: weight of a junction position anchor in the residual
BOUNCE_ANCHOR_W = 10.0   # bounce hypothesis: the pixel IS on the court plane (strong)
HIT_ANCHOR_W = 2.0       # hit hypothesis: near the player's box foot (weak: reach ~2 m)
HIT_ANCHOR_WZ = 0.3      # even weaker vertically: contact height varies 0.3-3 m
ANCHOR_RMS_TOL = 0.6     # px: max pixel-rms increase an anchored refit may cost
VOLUME_W = 6.0    # px per meter outside the plausible play volume (soft prior)
UNDER_W = 40.0    # px per meter of underground excursion: the ball stays above court
SPEED_W = 2.0     # px per m/s above the fastest plausible serve speed
SPEED_MAX = 62.0  # m/s (~223 km/h)
# Occam prior: the depth-degenerate rms valley is flat along the view ray, so among
# projectively equivalent arcs prefer LOW and SLOW (tosses/prep would otherwise be
# read as high-altitude fast balls braking on drag)
SOFT_Z_W = 0.5    # px per meter of trajectory apex above SOFT_Z_MAX
SOFT_Z_MAX = 5.0  # m: rally balls and tosses stay below; only rare lobs exceed
SOFT_V_W = 0.02   # px per m/s of initial speed
SPIN_W = 0.15     # px per 100 rad/s of fitted topspin (mild shrinkage)
Z_PLAY_MAX = 8.0  # m: lobs stay under this at RG
Y_PLAY = (-6.0, 30.0)
SPIN_MAX = 5.0    # scaled: 500 rad/s

# theta = [x0, y0, z0, vx0, vy0, vz0, s] with topspin w = 100*s * unit(z_hat x v_h).
# Spinless fits are structurally biased on clay: heavy topspin adds ~0.3-0.5 g of Magnus
# downdraft, which a spinless fit fakes by pulling the arc toward the camera (depth
# ghost). The volume/spin priors break the residual spin-depth degeneracy.
FIT_LO = np.array([-15, -10, 0.0, -90, -90, -60, -SPIN_MAX], float)
FIT_HI = np.array([26, 34, 14.0, 90, 90, 60, SPIN_MAX], float)


def spin_vector(theta: np.ndarray) -> np.ndarray:
    """Launch spin in a velocity-aligned top/side/rifle basis.

    Seven-parameter legacy fits contain topspin only. Nine-parameter fits add a
    sidespin axis perpendicular to both travel and topspin, plus rifle spin along
    travel. Spin parameters are in hundreds of radians per second.
    """
    velocity = np.asarray(theta[3:6], float)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return np.zeros(3)
    travel_axis = velocity / speed
    topspin_axis = np.cross(np.array([0.0, 0.0, 1.0]), travel_axis)
    topspin_norm = float(np.linalg.norm(topspin_axis))
    if topspin_norm <= 1e-6:
        topspin_axis = np.array([1.0, 0.0, 0.0])
    else:
        topspin_axis /= topspin_norm
    sidespin_axis = np.cross(travel_axis, topspin_axis)
    sidespin_norm = float(np.linalg.norm(sidespin_axis))
    if sidespin_norm > 1e-6:
        sidespin_axis /= sidespin_norm
    components = np.zeros(3)
    components[: min(3, max(0, len(theta) - 6))] = theta[6:9]
    return 100.0 * (
        components[0] * topspin_axis
        + components[1] * sidespin_axis
        + components[2] * travel_axis
    )


def fit_arc(P, uv, fidx, fps: float | None = None,
            max_nfev: int = MAX_NFEV, seed_state=None, anchor=None,
            seed_only: bool = False) -> Arc | None:
    """Robust 3D flight fit (drag + single-axis Magnus) matching the observed pixels.

    Multi-start (ray back-projections at several heights, plus an optional continuity
    seed from the previous arc) guards the depth-ambiguous local minima of short slow
    arcs. `anchor=(frame, xyz[, per-axis weights])` adds a soft position term tying the
    arc's extrapolated state at that frame to a known 3D point (a neighbor arc's state,
    the court plane at a bounce pixel, a player box at a hit) — the global-consistency
    constraint that resolves the telephoto depth ambiguity. One reweighting round: drop
    observations beyond max(3 px, 3*median residual), refit. Reported rms is over pixel
    residuals only.
    """
    from scipy.optimize import least_squares

    fps = FPS if fps is None else fps
    fidx = np.asarray(fidx, int)
    uv = np.asarray(uv, float)
    if len(uv) < 5:
        return None
    dt = 1.0 / fps
    span_t = (fidx[-1] - fidx[0]) * dt
    if span_t <= 0:
        return None
    lo, hi = FIT_LO, FIT_HI
    # anchor: one (frame, xyz[, weights]) tuple or a list of them
    anchors = []
    if anchor is not None:
        raw = anchor if isinstance(anchor, list) else [anchor]
        for item in raw:
            if item is None:
                continue
            w = np.asarray(item[2], float) if len(item) > 2 else np.full(3, ANCHOR_W)
            anchors.append((int(item[0]), np.asarray(item[1], float), w))
    inits = []
    if not seed_only:
        for z in INIT_HEIGHTS:
            p0 = _ray_point(P, *uv[0], z)
            p1 = _ray_point(P, *uv[-1], z)
            if p0 is None or p1 is None:
                continue
            theta0 = np.concatenate([p0, (p1 - p0) / span_t, [0.0]])
            inits.append(("ray", np.clip(theta0, lo + 1e-6, hi - 1e-6)))
    if seed_state is not None:
        x0s, v0s = seed_state
        theta0 = np.concatenate([x0s, v0s, [0.0]])
        inits.append(("seed", np.clip(theta0, lo + 1e-6, hi - 1e-6)))
    if not inits:
        return None

    def make_resid(fidx_, uv_):
        """theta = state at fidx_[0]; pixel residuals + anchors + soft priors."""
        f_base = fidx_[0]
        rel_ = fidx_ - f_base
        n_fwd = int(rel_[-1]) + 1
        a_rels = [(int(f - f_base), xyz, w) for f, xyz, w in anchors]
        n_int = max([n_fwd] + [r + 1 for r, _, _ in a_rels if r > 0])
        n_back = max([0] + [-r for r, _, _ in a_rels if r < 0])

        def resid(theta):
            w = spin_vector(theta)
            xs, _ = integrate_states(theta[:3], theta[3:6], w, n_int, dt)
            obs_xs = xs[:n_fwd]
            parts = [(project(P, xs[rel_]) - uv_).ravel()]
            xs_back = None
            if n_back:
                xs_back, _ = integrate_states(theta[:3], theta[3:6], w, n_back + 1, -dt)
            for a_rel, xyz, wvec in a_rels:
                xa = xs[a_rel] if a_rel >= 0 else xs_back[-a_rel]
                parts.append(wvec * (xa - xyz))
            parts.append(np.array([
                VOLUME_W * max(0.0, float(obs_xs[:, 2].max()) - Z_PLAY_MAX),
                VOLUME_W * max(0.0, Y_PLAY[0] - float(obs_xs[:, 1].min())),
                VOLUME_W * max(0.0, float(obs_xs[:, 1].max()) - Y_PLAY[1]),
                UNDER_W * max(0.0, -float(obs_xs[:, 2].min())),
                SPEED_W * max(0.0, float(np.linalg.norm(theta[3:6])) - SPEED_MAX),
                SOFT_Z_W * max(0.0, float(obs_xs[:, 2].max()) - SOFT_Z_MAX),
                SOFT_V_W * float(np.linalg.norm(theta[3:6])),
                SPIN_W * abs(theta[6]),
            ]))
            return np.concatenate(parts)
        return resid

    def pix_stats(sol, n_obs):
        res = np.linalg.norm(sol.fun[: 2 * n_obs].reshape(-1, 2), axis=1)
        cut = max(3.0, 3.0 * float(np.median(res)))
        keep = res <= cut
        rms_in = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.any() else math.inf
        return res, keep, rms_in

    full_resid = make_resid(fidx, uv)
    solutions = []
    for kind, theta0 in inits:
        try:
            sol = least_squares(full_resid, theta0, bounds=(lo, hi),
                                loss="soft_l1", f_scale=3.0, max_nfev=max_nfev)
        except Exception:  # noqa: BLE001
            continue
        res, keep, rms_in = pix_stats(sol, len(uv))
        # basin score: inlier rms, penalized for explaining fewer points and for
        # leaving the play volume (ghost basins hug the camera)
        # anchors and priors count against a basin: a ghost that violates a 3 m anchor
        # must lose to a truthful fit even at slightly higher pixel rms
        prior_pen = float(np.linalg.norm(sol.fun[2 * len(uv):])) / max(len(uv), 1) ** 0.5
        score = rms_in + 2.0 * (1.0 - float(keep.mean())) + prior_pen
        solutions.append((kind, score, sol, res, keep))
    if not solutions:
        return None
    best_score = min(s[1] for s in solutions)
    # near-ties resolve toward the continuity seed — but only when no anchor is in
    # play (anchored fits pick the true best; the seed may be a ghost)
    seeded = [s for s in solutions if s[0] == "seed" and s[1] <= best_score + 0.2]
    if anchors:
        seeded = []
    _, _, sol, res, keep = seeded[0] if seeded else min(solutions, key=lambda s: s[1])

    # trim outlier runs at the segment ends (cross-knot contamination), keep interior mask
    idx = np.where(keep)[0]
    if len(idx) < 5:
        return None
    keep_trim = np.zeros(len(uv), bool)
    keep_trim[idx[0]: idx[-1] + 1] = keep[idx[0]: idx[-1] + 1]
    inlier_frac = float(keep_trim[idx[0]: idx[-1] + 1].mean())
    fidx_t, uv_t = fidx[keep_trim], uv[keep_trim]
    if keep_trim.sum() < len(uv):
        # advance the init state to the trimmed first frame before refitting
        theta_init = sol.x.copy()
        shift = int(fidx_t[0] - fidx[0])
        if shift > 0:
            xs_s, vs_s = integrate_states(sol.x[:3], sol.x[3:6], spin_vector(sol.x),
                                          shift + 1, dt)
            theta_init[:3], theta_init[3:6] = xs_s[-1], vs_s[-1]
            theta_init = np.clip(theta_init, lo + 1e-6, hi - 1e-6)
        try:
            sol = least_squares(make_resid(fidx_t, uv_t), theta_init, bounds=(lo, hi),
                                loss="soft_l1", f_scale=3.0, max_nfev=max_nfev)
            res_in = np.linalg.norm(sol.fun[: 2 * len(uv_t)].reshape(-1, 2), axis=1)
        except Exception:  # noqa: BLE001
            # untrimmed fallback: keep the original frame span so x0/v0 stay aligned
            rms0 = float(np.sqrt(np.mean(res[keep] ** 2)))
            return Arc(fidx[0], fidx[-1], sol.x[:3], sol.x[3:6], spin_vector(sol.x),
                       rms0, len(uv), float(keep.mean()), uv[0], uv[-1])
    else:
        res_in = res
    rms = float(np.sqrt(np.mean(res_in**2)))
    return Arc(fidx_t[0], fidx_t[-1], sol.x[:3], sol.x[3:6], spin_vector(sol.x), rms,
               int(keep_trim.sum()), inlier_frac, uv_t[0], uv_t[-1])


# ------------------------------- knot proposal --------------------------------------------


def drop_static_runs(fidx, uv, score):
    """Remove sustained near-stationary runs (held ball, dead time)."""
    if len(fidx) < STATIC_RUN:
        return fidx, uv, score
    disp = np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.maximum(np.diff(fidx), 1)
    disp = np.concatenate([[disp[0]], disp])
    k = np.ones(9) / 9
    smooth = np.convolve(disp, k, mode="same")
    static = smooth < STATIC_SPEED
    keep = np.ones(len(fidx), bool)
    i = 0
    while i < len(fidx):
        if static[i]:
            j = i
            while j < len(fidx) and static[j]:
                j += 1
            if j - i >= STATIC_RUN:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return fidx[keep], uv[keep], score[keep]


def propose_boundaries(fidx, uv, seed_frames=()) -> list[tuple[int, bool]]:
    """(index, is_gap) pairs where a new segment starts.

    Non-gap cuts (flips, spikes, seeds) are refined to the local acceleration peak; the
    caller trims one frame on each side of them so junction-straddling observations do
    not contaminate either arc's fit.
    """
    n = len(fidx)
    if n == 0:
        return []
    gap_cuts = {0}
    for i in range(1, n):
        if fidx[i] - fidx[i - 1] > GAP_BREAK:
            gap_cuts.add(i)
    k3 = np.ones(3) / 3
    du = np.gradient(np.convolve(uv[:, 0], k3, mode="same"))
    dv = np.gradient(np.convolve(uv[:, 1], k3, mode="same"))
    accel = np.zeros(n)
    accel[1:-1] = np.hypot(du[2:] - du[:-2], dv[2:] - dv[:-2]) / 2
    knot_cuts = set()
    for i in range(2, n - 2):
        flip_u = np.sign(du[i - 1]) != np.sign(du[i + 1]) and abs(du[i - 1]) + abs(du[i + 1]) > 2 * FLIP_SPEED
        flip_v = np.sign(dv[i - 1]) != np.sign(dv[i + 1]) and abs(dv[i - 1]) + abs(dv[i + 1]) > 2 * FLIP_SPEED
        spike = accel[i] > ACCEL_SPIKE
        if flip_u or flip_v or spike:
            lo, hi = max(1, i - 2), min(n - 1, i + 3)
            knot_cuts.add(int(lo + np.argmax(accel[lo:hi])))
    for f in seed_frames:
        i = int(np.argmin(np.abs(fidx - f)))
        if abs(fidx[i] - f) <= 3 and 0 < i < n:
            knot_cuts.add(i)
    # thin: keep boundaries at least MIN_SEG apart (first wins; gap cuts always kept)
    ordered = sorted(gap_cuts | knot_cuts)
    thinned: list[tuple[int, bool]] = []
    for c in ordered:
        if c in gap_cuts or not thinned or c - thinned[-1][0] >= MIN_SEG:
            thinned.append((c, c in gap_cuts))
    return thinned


# ------------------------------- split / merge refinement ---------------------------------


def split_arc(P, uv, fidx, fps, depth: int = 2) -> list[Arc]:
    """Fit; if the residual betrays a hidden knot, split at the best interior point."""
    arc = fit_arc(P, uv, fidx, fps)
    if arc is None:
        return []
    if arc.rms <= SPLIT_RMS or len(uv) < 2 * MIN_SEG or depth == 0:
        return [arc]
    best = [arc]
    best_cost = arc.rms
    # candidate split points: interior residual peak and midpoint
    xs, _ = integrate_states(arc.x0, arc.v0, arc.w, int(fidx[-1] - fidx[0]) + 1, 1.0 / fps)
    res = np.linalg.norm(project(P, xs[(fidx - fidx[0]).astype(int)]) - uv, axis=1)
    interior = np.arange(MIN_SEG, len(uv) - MIN_SEG)
    if len(interior) == 0:
        return [arc]
    peaks = [int(interior[np.argmax(res[interior])]), len(uv) // 2]
    for s in dict.fromkeys(peaks):
        left = split_arc(P, uv[:s], fidx[:s], fps, depth - 1)
        right = split_arc(P, uv[s:], fidx[s:], fps, depth - 1)
        if not left or not right:
            continue
        cost = max(a.rms for a in left + right)
        if cost < best_cost - 0.3:
            best, best_cost = left + right, cost
    return best


def merge_arcs(P, arcs, obs_by_frame, fps) -> list[Arc]:
    """Join adjacent arcs whose junction shows no real velocity discontinuity."""
    arcs = sorted(arcs, key=lambda a: a.f0)
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(arcs):
            a = arcs[i]
            if i + 1 < len(arcs):
                b = arcs[i + 1]
                gap = b.f0 - a.fe
                if 0 < gap <= MAX_JUNCTION_GAP:
                    fj = (a.fe + b.f0) / 2
                    _, va = a.state_at(fj, fps)
                    _, vb = b.state_at(fj, fps)
                    if float(np.linalg.norm(vb - va)) < MERGE_DV:
                        frames = sorted(
                            f for f in obs_by_frame if a.f0 <= f <= b.fe
                        )
                        uv = np.array([obs_by_frame[f] for f in frames])
                        joint = fit_arc(P, uv, frames, fps)
                        if joint is not None and joint.rms <= max(a.rms, b.rms) + 0.3:
                            out.append(joint)
                            i += 2
                            changed = True
                            continue
            out.append(a)
            i += 1
        arcs = out
    return arcs


# ------------------------------- knot classification --------------------------------------


def junction_frame(a: Arc, b: Arc, fps: float | None = None) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    fps = FPS if fps is None else fps
    """Contact frame between arcs: where forward- and backward-extrapolations meet."""
    lo, hi = a.fe, b.f0
    if hi <= lo:
        f = int(round((a.fe + b.f0) / 2))
        xa, va = a.state_at(f, fps)
        _, vb = b.state_at(f, fps)
        return f, xa, va, vb
    frames = list(range(lo, hi + 1))
    best, best_d = frames[len(frames) // 2], np.inf
    for f in frames:
        xa, _ = a.state_at(f, fps)
        xb, _ = b.state_at(f, fps)
        d = float(np.linalg.norm(xa - xb))
        if d < best_d:
            best, best_d = f, d
    xa, va = a.state_at(best, fps)
    xb, vb = b.state_at(best, fps)
    return best, (xa + xb) / 2, va, vb


def bounce_residual(v_in: np.ndarray, v_out: np.ndarray, surface: str = "clay") -> float:
    """Relative disagreement with the Cross bounce model, minimized over topspin."""
    vh_in = float(np.hypot(v_in[0], v_in[1]))
    if vh_in < 1e-3 or v_in[2] >= 0:
        return math.inf
    vh_out = float(np.hypot(v_out[0], v_out[1]))
    best = math.inf
    for w1 in (0.0, 150.0, 300.0, 450.0):
        try:
            b = impact.court_bounce(vh_in, -float(v_in[2]), w1, surface)
        except ValueError:
            continue
        err = math.hypot(
            (b.vx2 - vh_out) / max(vh_in, 1.0),
            (b.vy2 - float(v_out[2])) / max(-float(v_in[2]), 1.0),
        )
        best = min(best, err)
    return best


def _heading_change_deg(v_in: np.ndarray, v_out: np.ndarray) -> float:
    vh_in = np.array([v_in[0], v_in[1]])
    vh_out = np.array([v_out[0], v_out[1]])
    nh_in, nh_out = np.linalg.norm(vh_in), np.linalg.norm(vh_out)
    if nh_in < 1e-3 or nh_out < 1e-3:
        return 0.0
    cos_h = float(vh_in @ vh_out / (nh_in * nh_out))
    return math.degrees(math.acos(max(-1, min(1, cos_h))))


def knot_features(x: np.ndarray, v_in: np.ndarray, v_out: np.ndarray,
                  dist_px: float) -> dict:
    """Physics residual features of a junction (fusion inputs; scoring-free)."""
    dv = float(np.linalg.norm(v_out - v_in))
    vh_in = np.array([v_in[0], v_in[1]])
    vh_out = np.array([v_out[0], v_out[1]])
    nh_in, nh_out = np.linalg.norm(vh_in), np.linalg.norm(vh_out)
    cos_h = float(vh_in @ vh_out / (nh_in * nh_out)) if nh_in > 1e-3 and nh_out > 1e-3 else 0.0
    e_vert = (-float(v_out[2]) / float(v_in[2])) if v_in[2] < -1e-3 else math.nan
    h_ratio = float(nh_out / nh_in) if nh_in > 1e-3 else math.nan
    ang = math.degrees(math.acos(max(-1, min(1, cos_h))))
    b_res = bounce_residual(v_in, v_out)
    return {
        "dv": round(dv, 2), "z": round(float(x[2]), 2),
        "e_vert": round(e_vert, 3) if not math.isnan(e_vert) else "",
        "h_ratio": round(h_ratio, 3) if not math.isnan(h_ratio) else "",
        "cos_h": round(cos_h, 3), "angle_h_deg": round(ang, 1),
        "bounce_resid": round(b_res, 3) if math.isfinite(b_res) else "",
        "dist_px": round(dist_px, 1) if math.isfinite(dist_px) else "",
        "speed_in": round(float(np.linalg.norm(v_in)), 1),
        "speed_out": round(float(np.linalg.norm(v_out)), 1),
        "racket_feasible": int(impact.racket_hit_feasible(v_in, v_out)),
    }


def classify_knot(x: np.ndarray, v_in: np.ndarray, v_out: np.ndarray,
                  dist_px: float) -> tuple[str, dict]:
    """bounce / hit / noise from free-fit junction states (fallback classifier)."""
    feats = knot_features(x, v_in, v_out, dist_px)
    e_vert = feats["e_vert"] if feats["e_vert"] != "" else math.nan
    h_ratio = feats["h_ratio"] if feats["h_ratio"] != "" else math.nan
    cos_h = feats["cos_h"]
    dv = feats["dv"]
    ang = feats["angle_h_deg"]
    y_flip = bool(np.sign(v_out[1]) != np.sign(v_in[1])) if abs(v_in[1]) > 0.5 else False
    is_bounce = (
        float(x[2]) <= BOUNCE_Z_MAX
        and v_in[2] < -1.0
        and v_out[2] > 0.2
        and not math.isnan(e_vert)
        and BOUNCE_E_RANGE[0] <= e_vert <= BOUNCE_E_RANGE[1]
        and cos_h > 0.0
        and BOUNCE_H_RATIO[0] <= h_ratio <= BOUNCE_H_RATIO[1]
    )
    if is_bounce:
        return "bounce", feats
    feasible = impact.racket_hit_feasible(v_in, v_out)
    feats["racket_feasible"] = int(feasible)
    is_hit = (
        dv >= HIT_DV_MIN
        and feasible
        and dist_px <= HIT_DIST_PX
        and float(x[2]) <= HIT_Z_MAX
        and (y_flip or ang >= 45.0)
    )
    if is_hit:
        return "hit", feats
    return "noise", feats


# ------------------------------- per-clip pipeline ----------------------------------------


def filter_player_boxes(P: np.ndarray, boxes_by_frame: dict) -> dict:
    """Keep only the actual players: feet on/near the court, largest box per side.

    The raw person boxes include ball kids and line judges; a bounce landing near one
    must not read as a racket hit (contacts_v2.load_boxes applies the same idea)."""
    out = {}
    for frame, cands in boxes_by_frame.items():
        sides: dict[str, list] = {}
        for x0, y0, x1, y1 in cands:
            foot = _ray_point(P, (x0 + x1) / 2, y1, 0.0)
            if foot is None:
                continue
            if not (-3.0 <= foot[0] <= 13.97 and -8.0 <= foot[1] <= 31.77):
                continue
            side = "near" if foot[1] < NET_Y else "far"
            sides.setdefault(side, []).append((x0, y0, x1, y1))
        chosen = [
            max(group, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            for group in sides.values()
        ]
        if chosen:
            out[frame] = chosen
    return out


def process_clip(clip: str, P: np.ndarray, obs: tuple, boxes_by_frame: dict,
                 seed_frames=(), fps: float | None = None, emit_soft: bool = False):
    """Full fit-and-segment for one rally window. Returns (events, knots, arcs_meta)."""
    fps = FPS if fps is None else fps
    fidx, uv, score = obs
    keep = score >= SCORE_MIN
    fidx, uv, score = fidx[keep], uv[keep], score[keep]
    fidx, uv, score = drop_static_runs(fidx, uv, score)
    if len(fidx) < MIN_SEG:
        return [], [], []
    boxes_by_frame = filter_player_boxes(P, boxes_by_frame)
    bounds = propose_boundaries(fidx, uv, seed_frames)
    bounds.append((len(fidx), True))
    arcs = []
    for (a, a_gap), (b, b_gap) in zip(bounds, bounds[1:]):
        a_eff = a if a_gap else a + 1       # trim junction-straddling frames
        b_eff = b if b_gap else b - 1
        if b_eff - a_eff < MIN_SEG:
            continue
        seg_uv, seg_f = uv[a_eff:b_eff], fidx[a_eff:b_eff]
        if len(seg_uv) > 200:   # cost cap; hidden knots recovered by split pass
            sub = np.arange(0, len(seg_uv), 2)
            seg_uv, seg_f = seg_uv[sub], seg_f[sub]
        arcs.extend(split_arc(P, seg_uv, seg_f, fps))
    arcs = [a for a in arcs if a.rms <= NOISE_RMS and a.inlier_frac >= 0.5]
    obs_map = {int(f): uv[i] for i, f in enumerate(fidx)}
    arcs = merge_arcs(P, arcs, obs_map, fps)
    arcs = sorted(arcs, key=lambda a: a.f0)

    def nearest_box_dist(frame: int, u: float, v: float) -> float:
        cands = boxes_by_frame.get(frame, [])
        if not cands:
            return math.inf
        return min(
            math.hypot(max(x0 - u, 0, u - x1), max(y0 - v, 0, v - y1))
            for x0, y0, x1, y1 in cands
        )

    def arc_obs(a):
        frames = sorted(f for f in obs_map if a.f0 <= f <= a.fe)
        return np.array([obs_map[f] for f in frames]), np.array(frames, int)

    def nearest_box_foot(frame: int, u: float, v: float):
        """(distance_px, court-plane foot position) of the nearest player box."""
        cands = boxes_by_frame.get(frame, [])
        best = (math.inf, None)
        for x0, y0, x1, y1 in cands:
            dpx = math.hypot(max(x0 - u, 0, u - x1), max(y0 - v, 0, v - y1))
            if dpx < best[0]:
                foot = _ray_point(P, (x0 + x1) / 2, y1, 0.0)
                best = (dpx, foot)
        return best

    def anchor_ok(refit, orig):
        # ghosts overfit, so a truthful anchored solution may cost some rms; accept
        # anything within tolerance of the original or under the tracker noise floor
        return refit.rms <= max(orig.rms + ANCHOR_RMS_TOL, 2.2)

    def anchored_pair(a, b, fj, target, weights, prev_anchor=None):
        """Refit both arcs anchored to `target` at frame fj; None if either fails.

        `prev_anchor` is the incoming arc's already-decided start anchor (previous
        junction): with both ends pinned, the arc's depth-rate is finally determined."""
        uv_a, fr_a = arc_obs(a)
        uv_b, fr_b = arc_obs(b)
        ra = fit_arc(P, uv_a, fr_a, fps,
                     anchor=[prev_anchor, (fj, target, weights)],
                     seed_state=(np.asarray(a.x0), np.asarray(a.v0)))
        rb = fit_arc(P, uv_b, fr_b, fps, anchor=(fj, target, weights),
                     seed_state=(np.asarray(b.x0), np.asarray(b.v0)))
        if ra is None or rb is None:
            return None
        if not (anchor_ok(ra, a) and anchor_ok(rb, b)):
            return None
        return ra, rb

    # live-play chains: arcs connected through valid junctions form one continuous
    # ball flight. Serve-prep taps and ball handling form short slow chains that no
    # globally consistent rally trajectory contains — the chart-free conservation cut.
    # ---- pass A: anchored junction hypothesis testing over ALL adjacent pairs.
    # A bounce pixel back-projects through the court plane to an exact 3D point; a hit
    # is near a player box. Refitting both arcs against each hypothesis anchor resolves
    # the telephoto depth ambiguity and lets the impact physics decide; winning refits
    # replace the arcs so anchors chain along the rally.
    junctions = []
    start_anchor: dict[int, tuple] = {}   # arc index -> its decided start anchor
    i = 0
    while i < len(arcs) - 1:
        a, b = arcs[i], arcs[i + 1]
        i += 1
        gap = b.f0 - a.fe
        if gap > MAX_JUNCTION_GAP:
            continue
        fj_free, xj_free, _, _ = junction_frame(a, b, fps)
        fj_cands = sorted({a.fe, (a.fe + b.f0) // 2, b.f0})
        prev_anchor = start_anchor.get(i - 1)

        def junction_pixel(fj):
            near_obs = [f for f in range(fj - 2, fj + 3) if f in obs_map]
            if near_obs:
                return np.asarray(obs_map[min(near_obs, key=lambda f: abs(f - fj))])
            return project(P, xj_free.reshape(1, 3))[0]

        def side_px_speed(f_lo, f_hi):
            frames_ = sorted(f for f in obs_map if f_lo <= f <= f_hi)
            if len(frames_) < 3:
                return 0.0
            pts = np.array([obs_map[f] for f in frames_])
            span = max(frames_[-1] - frames_[0], 1)
            return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) / span

        # a contact changes motion; a junction inside a near-static blob (toss apex,
        # held ball) is never a contact
        moving = (side_px_speed(a.fe - 6, a.fe) >= 1.5
                  or side_px_speed(b.f0, b.f0 + 6) >= 1.5)

        hyps = []   # (klass, cost, ra, rb, fj, uvj, d_px, xj, anchor)
        for fj in fj_cands if moving else []:
            uvj = junction_pixel(fj)
            d_px, foot = nearest_box_foot(fj, *uvj)
            ground = _ray_point(P, uvj[0], uvj[1], 0.0)
            if ground is not None and -2 <= ground[0] <= 13 and -2 <= ground[1] <= 26:
                pair = anchored_pair(a, b, fj, [ground[0], ground[1], 0.05],
                                     (BOUNCE_ANCHOR_W, BOUNCE_ANCHOR_W, BOUNCE_ANCHOR_W),
                                     prev_anchor)
                if pair is not None:
                    ra, rb = pair
                    _, v_in = ra.state_at(fj, fps)
                    _, v_out = rb.state_at(fj, fps)
                    e_vert = (-float(v_out[2]) / float(v_in[2])) if v_in[2] < -0.3 else math.nan
                    ok = (
                        not math.isnan(e_vert)
                        and v_out[2] > 0.0
                        and BOUNCE_E_RANGE[0] <= e_vert <= BOUNCE_E_RANGE[1]
                    )
                    cost = (max(0.0, ra.rms - a.rms) + max(0.0, rb.rms - b.rms)
                            + bounce_residual(v_in, v_out))
                    if ok:
                        xj = np.array([ground[0], ground[1], 0.0])
                        anc = (fj, [ground[0], ground[1], 0.05],
                               (BOUNCE_ANCHOR_W, BOUNCE_ANCHOR_W, BOUNCE_ANCHOR_W))
                        hyps.append(("bounce", cost, ra, rb, fj, uvj, d_px, xj, anc))
            if foot is not None and d_px <= SERVE_DIST_PX:
                target = [foot[0], foot[1], 1.3]
                pair = anchored_pair(a, b, fj, target, (HIT_ANCHOR_W, HIT_ANCHOR_W, HIT_ANCHOR_WZ),
                                     prev_anchor)
                if pair is not None:
                    ra, rb = pair
                    xj, v_in = ra.state_at(fj, fps)
                    _, v_out = rb.state_at(fj, fps)
                    dv = float(np.linalg.norm(v_out - v_in))
                    y_flip = (bool(np.sign(v_out[1]) != np.sign(v_in[1]))
                              if abs(v_in[1]) > 0.5 else False)
                    speed_out = float(np.linalg.norm(v_out))
                    speed_in = float(np.linalg.norm(v_in))
                    # serve: slow toss in, fast drive out, contact overhead
                    serve_like = (speed_out >= 25.0 and speed_in <= 8.0
                                  and float(xj[2]) >= 1.7)
                    ok = (
                        dv >= HIT_DV_MIN
                        and impact.racket_hit_feasible(v_in, v_out)
                        and float(xj[2]) <= HIT_Z_MAX
                        and speed_out >= HIT_SPEED_OUT_MIN
                        and (d_px <= HIT_DIST_PX or serve_like)
                        and (speed_in >= HIT_SPEED_IN_MIN or serve_like)
                        # every rally hit sends the ball back toward the opponent
                        and (y_flip or serve_like)
                    )
                    cost = (max(0.0, ra.rms - a.rms) + max(0.0, rb.rms - b.rms)
                            + max(0.0, (HIT_DV_MIN + 2.0 - dv) * 0.1))
                    if ok:
                        anc = (fj, target, (HIT_ANCHOR_W, HIT_ANCHOR_W, HIT_ANCHOR_WZ))
                        hyps.append(("hit", cost, ra, rb, fj, uvj, d_px, xj, anc))
        if hyps:
            klass, cost, ra, rb, fj, uvj, d_px, xj, anc = min(hyps, key=lambda h: h[1])
            arcs[i - 1], arcs[i] = ra, rb
            a, b = ra, rb
            start_anchor[i] = anc
            _, v_in = a.state_at(fj, fps)
            _, v_out = b.state_at(fj, fps)
            # refine the emitted contact frame: where the two anchored arcs meet
            if gap > 1:
                fj = min(range(a.fe, b.f0 + 1),
                         key=lambda f: float(np.linalg.norm(
                             a.state_at(f, fps)[0] - b.state_at(f, fps)[0])))
        else:
            klass, cost = "noise", math.inf
            fj = fj_free
            uvj = junction_pixel(fj)
            d_px, _ = nearest_box_foot(fj, *uvj)
            _, xj, v_in, v_out = junction_frame(a, b, fps)
        costs = {k: min(h[1] for h in hyps if h[0] == k) for k in ("bounce", "hit")
                 if any(h[0] == k for h in hyps)}
        junctions.append(dict(idx=i - 1, fj=fj, klass=klass, cost=cost, uvj=uvj,
                              d_px=d_px, xj=xj, v_in=v_in, v_out=v_out, costs=costs,
                              gap=gap))

    # ---- pass B: live-play chains from the anchored arcs. Arcs connected through
    # valid junctions form one continuous ball flight; serve-prep taps and handling
    # form short slow chains that no globally consistent rally trajectory contains —
    # the chart-free conservation cut.
    # velocity may only change discontinuously at a contact: a noise-classified
    # junction with a large unexplained velocity jump is NOT a physical link, so it
    # breaks the chain (and the downstream arc may become an orphan-start serve/hit)
    def is_link(j):
        if j["klass"] in ("hit", "bounce"):
            return True
        dv = float(np.linalg.norm(np.asarray(j["v_out"]) - np.asarray(j["v_in"])))
        return dv <= 2 * MERGE_DV

    junction_link = {j["idx"] + 1 for j in junctions if is_link(j)}
    chains, cur = [], []
    for i, _a in enumerate(arcs):
        if cur and i in junction_link and cur[-1] == i - 1:
            cur.append(i)
        else:
            if cur:
                chains.append(cur)
            cur = [i]
    if cur:
        chains.append(cur)
    live: set[int] = set()
    for chain in chains:
        top_speed = max(float(np.linalg.norm(arcs[j].v0)) for j in chain)
        support = sum(arcs[j].n_obs for j in chain)
        if top_speed >= CHAIN_SPEED_MIN and support >= CHAIN_OBS_MIN:
            live.update(chain)
    events, knots = [], []
    has_predecessor = junction_link
    # ---- pass C: orphan-start serves/hits: live-chain starts with no joinable
    # predecessor. Anchor the arc's start to the nearest player box (depth prior).
    for i, a in enumerate(arcs):
        if i not in live or i in has_predecessor:
            continue
        d_px, foot = nearest_box_foot(a.f0, *a.uv0)
        if foot is not None and d_px <= SERVE_DIST_PX:
            uv_a, fr_a = arc_obs(a)
            ra = fit_arc(P, uv_a, fr_a, fps,
                         anchor=(a.f0, [foot[0], foot[1], 1.8], (3.0, 3.0, 0.6)),
                         seed_state=(np.asarray(a.x0), np.asarray(a.v0)))
            if ra is not None and ra.rms <= a.rms + ANCHOR_RMS_TOL:
                arcs[i] = a = ra
        speed0 = float(np.linalg.norm(a.v0))
        # an orphan-start contact must leave toward the opponent's side
        toward_opponent = (a.v0[1] >= 8.0) if a.x0[1] < NET_Y else (a.v0[1] <= -8.0)
        if speed0 >= SERVE_SPEED_MIN and d_px <= SERVE_DIST_PX and toward_opponent:
            # the contact precedes the first tracked frame (occlusion near the player):
            # back-extrapolate to the frame nearest the player box
            f_hit, uv_hit, d_hit = a.f0, a.uv0, d_px
            for f in range(a.f0 - 1, max(a.f0 - 15, 0), -1):
                x_b, _ = a.state_at(f, fps)
                if x_b[2] < 0.0:
                    break
                uv_b = project(P, x_b.reshape(1, 3))[0]
                d_b, _ = nearest_box_foot(f, *uv_b)
                if d_b < d_hit - 1e-6:
                    f_hit, uv_hit, d_hit = f, uv_b, d_b
            side = "near" if a.x0[1] < NET_Y else "far"
            conf = min(1.0, speed0 / 40.0) * math.exp(-a.rms / 6.0)
            events.append(
                dict(clip=clip, frame=f_hit, kind="hit", track_id=-3,
                     u=round(uv_hit[0], 1), v=round(uv_hit[1], 1),
                     player_distance_px=round(d_hit, 1), rms_px=round(a.rms, 2),
                     confidence=round(conf, 3), source="physics_orphan_start")
            )
            knots.append(
                dict(clip=clip, frame=f_hit, klass="hit", u=round(uv_hit[0], 1),
                     v=round(uv_hit[1], 1), x=round(a.x0[0], 2), y=round(a.x0[1], 2),
                     dv="", z=round(a.x0[2], 2), e_vert="", h_ratio="", cos_h="",
                     angle_h_deg="", bounce_resid="", dist_px=round(d_hit, 1),
                     speed_in=0.0, speed_out=round(speed0, 1), racket_feasible="",
                     cost_bounce="", cost_hit="",
                     rms_in="", rms_out=round(a.rms, 2), gap=0, side=side,
                     orphan=1, confidence=round(conf, 3))
            )
    # ---- pass D: emit junction knots; events only from live junctions
    for j in junctions:
        a, b = arcs[j["idx"]], arcs[j["idx"] + 1]
        is_live = j["idx"] in live and (j["idx"] + 1) in live
        klass = j["klass"] if is_live else "noise"
        feats = knot_features(j["xj"], j["v_in"], j["v_out"], j["d_px"])
        side = "near" if j["xj"][1] < NET_Y else "far"
        rms_pair = max(a.rms, b.rms)
        conf = math.exp(-max(0.0, j["cost"])) if klass != "noise" else 0.0
        costs = j["costs"]
        knots.append(
            dict(clip=clip, frame=j["fj"], klass=klass,
                 u=round(float(j["uvj"][0]), 1), v=round(float(j["uvj"][1]), 1),
                 x=round(float(j["xj"][0]), 2), y=round(float(j["xj"][1]), 2), **feats,
                 cost_bounce=round(costs["bounce"], 3) if "bounce" in costs else "",
                 cost_hit=round(costs["hit"], 3) if "hit" in costs else "",
                 rms_in=round(a.rms, 2), rms_out=round(b.rms, 2), gap=j["gap"],
                 side=side, orphan=0, confidence=round(conf, 3))
        )
        if klass in ("hit", "bounce"):
            events.append(
                dict(clip=clip, frame=j["fj"], kind=klass, track_id=-4,
                     u=round(float(j["uvj"][0]), 1), v=round(float(j["uvj"][1]), 1),
                     player_distance_px=round(j["d_px"], 1) if math.isfinite(j["d_px"]) else math.inf,
                     rms_px=round(rms_pair, 2), confidence=round(conf, 3),
                     source="physics_knot")
            )
        elif (emit_soft and is_live and math.isfinite(j["d_px"])
                and j["d_px"] <= HIT_DIST_PX
                and feats["dv"] >= 3.0 and feats["speed_out"] >= 7.0):
            # soft candidate: failed the hard gates but plausible; the conservation
            # decode may select it to repair a same-side gap (missed-hit hole).
            # Dev-tested 2026-07-16: no measurable gain; off by default.
            events.append(
                dict(clip=clip, frame=j["fj"], kind="hit", track_id=-4,
                     u=round(float(j["uvj"][0]), 1), v=round(float(j["uvj"][1]), 1),
                     player_distance_px=round(j["d_px"], 1),
                     rms_px=round(rms_pair, 2), confidence=0.05,
                     source="physics_soft")
            )
    # NMS: de-duplicate hit events (orphan + junction can fire on the same contact)
    hits = sorted((e for e in events if e["kind"] == "hit"),
                  key=lambda e: -e["confidence"])
    kept: list[dict] = []
    for e in hits:
        if all(abs(e["frame"] - k["frame"]) >= NMS_FRAMES for k in kept):
            kept.append(e)
    events = sorted(
        kept + [e for e in events if e["kind"] != "hit"], key=lambda e: e["frame"]
    )
    arcs_meta = [
        dict(clip=clip, f0=a.f0, fe=a.fe, rms=round(a.rms, 2), n=a.n_obs,
             speed0=round(float(np.linalg.norm(a.v0)), 1),
             x0=round(a.x0[0], 2), y0=round(a.x0[1], 2), z0=round(a.x0[2], 2),
             wspin=round(float(np.linalg.norm(a.w)), 0))
        for a in arcs
    ]
    return events, knots, arcs_meta


def _worker(args):
    clip, P, obs, boxes, seeds = args
    try:
        return clip, process_clip(clip, P, obs, boxes, seeds)
    except Exception as exc:  # noqa: BLE001
        return clip, ("ERROR", repr(exc), [])


# ------------------------------- io + main -------------------------------------------------


def load_observations(path: str, clips: set[str]):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["clip"] in clips:
                out[r["clip"]].append(
                    (parse_frame(r["frame"]), float(r["x"]), float(r["y"]), float(r["score"]))
                )
    packed = {}
    for clip, rows in out.items():
        rows.sort()
        a = np.array(rows)
        packed[clip] = (a[:, 0].astype(int), a[:, 1:3], a[:, 3])
    return packed


def load_player_boxes(path: str, clips: set[str]):
    out = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["clip"] in clips:
                out[r["clip"]][parse_frame(r["frame"])].append(
                    (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
                )
    return {c: dict(d) for c, d in out.items()}


def load_seed_events(path: str, clips: set[str]):
    out = defaultdict(list)
    if not path:
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["clip"] in clips:
                out[r["clip"]].append(int(r["frame"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="substrate dir (read-only)")
    ap.add_argument("--outdir", required=True, help="artifact dir (bakeoff/B)")
    ap.add_argument("--candidates", default="ball_candidates_wasb_full_v1.csv")
    ap.add_argument("--boxes", default="player_boxes_50_full_v1.csv")
    ap.add_argument("--clips-file", default="", help="one clip id (ptNNNN) per line")
    ap.add_argument("--seed-events", default="", help="event csv for seeded knot proposal")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    d = np.load(os.path.join(args.out, "camera_P_per_point.npz"))
    Ps = {int(p): P for p, P in zip(d["pts"], d["P"])}

    frames_root = os.path.join(args.out, "rally_frames_50_contact_v2")
    all_clips = sorted(c for c in os.listdir(frames_root) if c.startswith("pt"))
    if args.clips_file:
        with open(args.clips_file) as f:
            wanted = [line.strip() for line in f if line.strip()]
    else:
        wanted = all_clips
    have_P = [c for c in wanted if int(c[2:]) in Ps]
    print(f"{len(wanted)} clips requested, {len(have_P)} with camera projection "
          f"(coverage slice: {len(wanted) - len(have_P)} missing P/H)")

    clipset = set(have_P)
    obs = load_observations(os.path.join(args.out, args.candidates), clipset)
    boxes = load_player_boxes(os.path.join(args.out, args.boxes), clipset)
    seeds = load_seed_events(
        os.path.join(args.out, args.seed_events) if args.seed_events else "", clipset
    )

    tasks = [
        (c, Ps[int(c[2:])], obs[c], boxes.get(c, {}), tuple(seeds.get(c, ())))
        for c in have_P if c in obs
    ]
    print(f"{len(tasks)} clips with observations; workers={args.workers}")
    all_events, all_knots, all_arcs, errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for clip, result in ex.map(_worker, tasks, chunksize=2):
            if result[0] == "ERROR":
                errors.append((clip, result[1]))
                continue
            ev, kn, arcs = result
            all_events.extend(ev)
            all_knots.extend(kn)
            all_arcs.extend(arcs)
    if errors:
        print(f"{len(errors)} clip errors, first: {errors[:3]}")

    os.makedirs(args.outdir, exist_ok=True)

    def write(name, rows, fields):
        path = os.path.join(args.outdir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows):5d} rows -> {path}")

    ev_fields = ["clip", "frame", "kind", "track_id", "u", "v", "player_distance_px",
                 "rms_px", "confidence", "source"]
    kn_fields = ["clip", "frame", "klass", "u", "v", "x", "y", "z", "dv", "e_vert",
                 "cost_bounce", "cost_hit",
                 "h_ratio", "cos_h", "angle_h_deg", "bounce_resid", "dist_px", "speed_in",
                 "speed_out", "racket_feasible", "rms_in", "rms_out", "gap", "side",
                 "orphan", "confidence"]
    arc_fields = ["clip", "f0", "fe", "rms", "n", "speed0", "x0", "y0", "z0", "wspin"]
    all_events.sort(key=lambda e: (e["clip"], e["frame"]))
    all_knots.sort(key=lambda e: (e["clip"], e["frame"]))
    write(f"events_{args.tag}.csv", all_events, ev_fields)
    write(f"knots_{args.tag}.csv", all_knots, kn_fields)
    write(f"arcs_{args.tag}.csv", all_arcs, arc_fields)
    n_hit = sum(e["kind"] == "hit" for e in all_events)
    n_b = sum(e["kind"] == "bounce" for e in all_events)
    print(f"events: {n_hit} hits, {n_b} bounces over {len({e['clip'] for e in all_events})} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
