"""Discover racket hits as breaks in a globally fitted ball flight — no detector input.

Question (Q3, physics-first): given only native ball observations, per-frame cameras, and
the flight model (gravity + drag + Magnus, bounces integrated in the dynamics), can a
split-and-merge fit recover the hit times a human labeled — without consuming any contact
detector's output?

Method: within a point window, observations are partitioned into hit-to-hit segments.
Each segment is one simulated flight (bounces are part of the dynamics, so a segment
legally spans hit -> bounce -> next hit). A segment whose robust reprojection cost is bad
gets split at its worst residual; a split survives only if it beats the parent by a fixed
per-knot penalty (an extra hit must earn its existence). Adjacent segments that fit as
one flight are merged back. Discovered knots = the segment boundaries = claimed hits.

This module deliberately reuses rich_ball_physics's primitives (FrameCamera, simulate,
project_one) so its fits are comparable with the refinement pass. Nothing here reads
Tier-1 labels; scoring lives in cv/validation/score_knot_solver.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rich_ball_physics import (  # noqa: E402
    FrameCamera,
    load_ball,
    match_spec,
)

PIXEL_SIGMA = 2.0        # fit-optimiser residual scale (soft_l1 f_scale); NOT the model-
                         # selection noise — that is calibrated per clip/region (see below).
STREAK_MIN_ELONG = 2.0   # only confident streaks constrain direction
STREAK_ANGLE_SIGMA_DEG = 15.0
MIN_SEG_OBS = 8          # a flight needs this many observations to be fittable
BLOCK_OBS = 20           # bottom-up initial block size
MAX_NFEV = 60
GAP_BREAK_F = 30         # unobserved run longer than this forces a segment boundary

# ---- noise-calibrated one-vs-two-flight model selection (GRIC / MDL family) ---------------
# Every split/merge decision here is one-vs-two flights. The incumbent used FIXED chi^2
# margins (SPLIT_PENALTY=60 to keep a merge, AUDIO/PHYS_SPLIT_PENALTY=8 to keep a split,
# FORCE_MERGE_FACTOR/MERGE_LOCAL_PX for infeasible breaks) — constants that implicitly assume
# a single 2 px observation noise everywhere and were hand-tuned on one broadcast. That is the
# broadcast-fragile tuning the advisor flags. Replaced with a criterion whose only free scale
# is the MEASURED observation noise: an extra flight (its 9 params + 1 break time) is justified
# only when the chi^2 improvement, computed under a per-region calibrated sigma, exceeds what
# those added parameters would absorb by chance. Sigma is calibrated per clip and court region
# (near/far). MEASURED (RG dev-9, 2026-07-21): REPROJECTION-pixel noise is ~1.3-2.0 px and, to
# our surprise, roughly homogeneous near vs far (far-court balls are smaller/blurrier but also
# move fewer px/frame, so per-pixel localisation is comparable) — the "far >> near" intuition
# holds in metric court units, not in the pixel space the fit lives in. It is also broadcast-
# stable (USO measured similar). So calibration mainly corrects the ABSOLUTE scale (measured
# ~1.3-2.0 px vs the incumbent's assumed 2.0) and adapts per clip; the criterion is defined in
# dimensionless chi^2 units, the portable form the tuned pixel margins were not.
FLIGHT_PARAMS = 9        # p0(3) + v0(3) + spin(3): the parameters an added flight introduces
BREAK_PARAMS = 1         # the break time
GRIC_LAMBDA = 0.8        # chi^2 cost per added parameter for a TRIGGERED split (a proposal
                         # backed by a physical trigger — an image-trajectory turn or an
                         # impact-audio onset). A triggered split must clear Delta-chi2 >
                         # GRIC_LAMBDA*(FLIGHT_PARAMS+BREAK_PARAMS) = 8 chi^2. Chosen so the
                         # criterion in calibrated-noise units reduces to the incumbent's tuned
                         # split margin (8) at the floored 2 px noise — see EXPERIMENTS
                         # 2026-07-21; a pure AIC(=2)/BIC penalty over-merged fast exchanges.
MERGE_FACTOR = 7.5       # the bar to KEEP an UNTRIGGERED split (arbitrary bottom-up block
                         # boundaries have no evidence a contact sits there) is this multiple of
                         # the triggered-split penalty — the Occam / two-part-MDL prior that an
                         # extra flight without a trigger is expensive to encode. Bottom-up
                         # merging is aggressive; the trigger-gated split passes then re-expose
                         # real contacts. (Legacy encoded this as merge 60 vs split 8 = 7.5x.)
INFEASIBLE_BREAK_FACTOR = 2.0 # a physically UNSUPPORTED break (no feasible impulse / no player)
                              # must clear this further multiple at force-merge — a prior against
                              # a break with no physical support (replaces FORCE_MERGE_FACTOR +
                              # the separate MERGE_LOCAL_PX pixel gate).
ROBUST_CLIP_SIGMA = 6.0  # a residual beyond this many sigma contributes a constant (robust chi^2)
DEFAULT_SIGMA_PX = 2.0   # per-region fallback when too few clean observations to calibrate
CALIB_MIN_OBS = 40       # a region with fewer clean residuals than this falls back to default
CALIB_MAX_MED_PX = 8.0   # a segment whose median residual exceeds this is not a clean single arc
AUDIO_EVIDENCE_CREDIT = 0.0   # extra chi^2 credit for an audio-proposed split beyond the trigger
                              # penalty itself (0: the onset trigger already sets the low bar)

# ---- calibrated noise model + GRIC criterion ---------------------------------------------
NET_Y_COURT = 11.885     # court-y of the net plane; the near/far region boundary


class NoiseModel:
    """Per-region observation-noise sigma (pixels), region 0 = near court (camera side, y <
    net), 1 = far. Measured reprojection-pixel noise is roughly homogeneous near vs far (see
    the section note), so the two regions usually calibrate close; the split is kept because it
    is cheap, lets a genuinely noisier region down-weight itself, and is where a broadcast with
    real far-court blur would show up."""

    __slots__ = ("sigma_near", "sigma_far", "n_near", "n_far")

    def __init__(self, sigma_near: float, sigma_far: float,
                 n_near: int = 0, n_far: int = 0) -> None:
        self.sigma_near = float(sigma_near)
        self.sigma_far = float(sigma_far)
        self.n_near = int(n_near)
        self.n_far = int(n_far)

    def sigma_array(self, regions: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(regions) == 1, self.sigma_far, self.sigma_near)

    def __repr__(self) -> str:
        return (f"NoiseModel(near={self.sigma_near:.3f}px/n={self.n_near}, "
                f"far={self.sigma_far:.3f}px/n={self.n_far})")


DEFAULT_NOISE = NoiseModel(DEFAULT_SIGMA_PX, DEFAULT_SIGMA_PX)


def region_of(uv: np.ndarray, camera: FrameCamera, frames: np.ndarray) -> np.ndarray:
    """Near(0)/far(1) court region per observation via homography back-projection (court-y vs
    the net). Image-space depth-robust: it only needs which half of the court the ball is in."""
    import cv2
    reg = np.empty(len(frames), dtype=int)
    for i, f in enumerate(frames):
        xy = cv2.perspectiveTransform(np.float32([[uv[i]]]), camera.h_at(int(f)))[0, 0]
        reg[i] = 1 if xy[1] >= NET_Y_COURT else 0
    return reg


def seg_chi2(fit: dict, noise: NoiseModel) -> float:
    """Robust chi^2 of a fitted flight under the calibrated per-region noise. Replaces the
    fixed-sigma robust `cost`: residuals are scaled by the region's measured sigma (not a
    constant 2 px) and clipped at ROBUST_CLIP_SIGMA sigma. This is the quantity every one-vs-
    two decision compares, so the split/merge thresholds no longer depend on a broadcast's
    absolute pixel scale."""
    if not fit.get("ok"):
        return math.inf
    r = fit.get("per_frame_px")
    reg = fit.get("regions")
    if r is None or reg is None:      # a hand-built fit (tests) without residual arrays
        return float(fit.get("cost", math.inf))
    z = np.minimum(np.asarray(r) / noise.sigma_array(reg), ROBUST_CLIP_SIGMA)
    return float(np.sum(z * z))


def calibrate_noise(segments: list[dict],
                    default_px: float = DEFAULT_SIGMA_PX,
                    floor_px: float = DEFAULT_SIGMA_PX) -> NoiseModel:
    """Fit an observation-noise sigma per region from UNAMBIGUOUS single-flight stretches:
    long, low-residual fitted segments away from any break. Their reprojection residual is
    observation noise (the flight model explains the signal). sigma = median|r| / 1.1774
    (median of a 2-D isotropic Gaussian's magnitude = 1.1774 sigma), robust to the odd
    contaminated frame. Regions with too few clean observations fall back to `default_px`.

    The clean-arc residual is a LOWER BOUND on the decision-relevant noise (in-sample residuals
    of a 9-parameter fit shrink; only the best-fitting frames enter; frames near a contact are
    noisier). We therefore FLOOR sigma at `floor_px` = the incumbent's empirically-validated
    2 px — never trusting the observations MORE than the tuned system did, but raising sigma
    (distrusting more) wherever a region or broadcast measures NOISIER than that. This is what
    makes the criterion (a) reproduce the tuned dev-9 result where measured noise <= 2 px and
    (b) still adapt on a genuinely noisier broadcast, without a per-broadcast pixel margin. The
    RAW measured sigmas are reported (n_near/n_far let a reader see the pre-floor values)."""
    near: list[float] = []
    far: list[float] = []
    for s in segments:
        if not s.get("ok"):
            continue
        r = s.get("per_frame_px")
        reg = s.get("regions")
        if r is None or reg is None or len(r) < 2 * MIN_SEG_OBS:
            continue
        if float(np.median(r)) > CALIB_MAX_MED_PX:   # not a clean arc: skip
            continue
        for i in range(2, len(r) - 2):               # trim ends (a contact may sit just outside)
            (far if reg[i] == 1 else near).append(float(r[i]))

    def sigma(res: list[float]) -> float:
        if len(res) < CALIB_MIN_OBS:
            return default_px
        return max(float(np.median(res)) / 1.1774, floor_px)

    return NoiseModel(sigma(near), sigma(far), n_near=len(near), n_far=len(far))


def gric_penalty(n_obs: int, n_extra_params: int = FLIGHT_PARAMS + BREAK_PARAMS,
                 lam: float = GRIC_LAMBDA) -> float:
    """Chi^2 a split must earn for its extra parameters (GRIC family). lam>0 is a fixed
    per-parameter cost (AIC=2); lam<=0 selects the BIC form ln(N_obs)*params, which scales
    the penalty with sample size. The manifold-dimension term of full GRIC cancels between
    one- and two-flight models (same observations, same per-datum curve dimension), leaving
    only this parameter term."""
    per_param = lam if lam > 0 else math.log(max(n_obs, 2))
    return per_param * n_extra_params


def split_beats_merge(chi2_one: float, chi2_two: float, n_obs: int,
                      lam: float = GRIC_LAMBDA, credit: float = 0.0,
                      factor: float = 1.0) -> float:
    """Signed margin by which two flights beat one: (chi2 improvement) - factor*(parameter
    penalty) + (evidence credit). >0 keeps the split; <=0 merges. The single criterion behind
    block merge, physics/audio split, boundary refinement and force-merge — the noise-calibrated
    chi^2 is common; only `factor` (the Occam prior: 1 for a triggered split, MERGE_FACTOR for
    an untriggered block boundary, more for a physically unsupported break) and `credit` vary,
    never a per-broadcast chi^2 margin."""
    return (chi2_one - chi2_two) - factor * gric_penalty(n_obs, lam=lam) + credit


# ---- position-dependent audio sound-travel delay -----------------------------------------
# The broadcast mic is effectively at the main camera behind the near baseline, elevated.
# Sound from a far-baseline contact reaches it 2-5 frames (50 fps) after the racket strike;
# near contacts ~1 frame. So an impact-audio onset time = true-contact time + travel delay,
# and the delay is a function of WHERE on court the hit happened. Correcting for it de-biases
# timing (near vs far) before Tier-1 scoring and un-shifts the observation-gap alignment.
SOUND_MPS = 343.0
# Court frame is metres: baselines at y=0 (near, camera side) and y=23.77 (far), net y=11.885,
# width centred near x=5.5. Camera-centre note: the per-frame P decomposition puts the centre
# at (6.49, -34.11, 33.89) m, which implies a ~6.9-frame delay even at the NEAR baseline
# (z~34 m) — physically incompatible with the measured ~1-frame near delay, so the per-frame
# P absolute scale is not trustworthy for sound travel. We use a nominal camera centre that
# reproduces the known near~1.6 / far~5.0 frame delays instead (side-based fallback position).
NOMINAL_CAM = np.array([5.5, -10.0, 6.0])
NEAR_BASELINE_XYZ = np.array([5.5, 0.0, 1.0])
FAR_BASELINE_XYZ = np.array([5.5, 23.77, 1.0])


def sound_delay_frames(court_xyz, fps: float) -> float:
    """Frames the racket sound takes to reach the mic from a contact at court_xyz."""
    dist = float(np.linalg.norm(np.asarray(court_xyz, float) - NOMINAL_CAM))
    return dist / SOUND_MPS * fps


def _hitter_xyz(knot: dict) -> np.ndarray:
    """Best court position of the hitter for the delay: the fitted contact position when
    it is on-court, else the hitter's baseline by side (fitted depth ghosts off-court often
    enough that the side-based nominal is the robust default)."""
    xy = knot.get("court_xy")
    if xy is not None and -3.0 <= xy[0] <= 14.0 and -3.0 <= xy[1] <= 27.0:
        return np.array([xy[0], xy[1], 1.0])
    return FAR_BASELINE_XYZ if knot.get("side") == "far" else NEAR_BASELINE_XYZ

# Fast flight integration: identical physics to rich_ball_physics.simulate (gravity,
# drag, Magnus lift, same constants, same bounce model via bounce_velocity) but scalar
# math and RK2 — the original allocates numpy arrays per RK4 substep and is ~100x too
# slow for discovery, where every candidate split/merge needs a fresh fit.
from rich_ball_physics import bounce_velocity  # noqa: E402
import flight as _flight  # noqa: E402  (physics/ is on sys.path via rich_ball_physics)

_P = _flight.default_params
_G = _P["g"]
_K = 0.5 * _P["rho_air"] * math.pi * _P["R_ball"] ** 2 / _P["m_ball"]
_CD = _P["C_drag"]
_CL = _P["C_lift"]
_RB = _P["R_ball"]
FAST_SUBSTEPS = 2

# ---- net-cord (tape) interaction --------------------------------------------------------
# The net plane is a third interaction surface alongside the racket and the court bounce
# (owner review, 2026-07-20): a ball that crosses the net line below the tape clips it — a real
# mid-flight velocity discontinuity. The tape absorbs most of the closing (court-y) speed
# and reverses what remains, the in-plane components are damped, and the ball either dribbles
# over or falls back (cf. pt0121: a netted backhand that bounces up without crossing, and
# pt0119 f262: the ball dying on the tape). Modeled here the way bounce_velocity models the
# court so a genuinely-netted flight fits as ONE flight instead of forcing a phantom split.
NET_Y = 11.885            # court y of the net plane (m)
NET_H_CENTER = 0.91       # tape height at the centre (m)
NET_H_POST = 1.07         # tape height at the singles/doubles posts (m)
NET_X_CENTER = 5.485      # court x of the net centre (m)
NET_HALF_WIDTH = 6.4      # centre-to-post span used for the tape-height ramp (doubles + posts)
E_NET = 0.25              # tape restitution on the crossing (court-y) velocity
NET_TANG = 0.4            # in-plane (x, z) velocity retained through a tape clip
NET_IN_DYNAMICS = False   # tape clips inside fast_simulate: OFF by default. Measured to
                          # REGRESS discovery on the dev-9 human labels (recall .677->.500,
                          # precision .724->.593): a net impact firing during least_squares
                          # iterations perturbs the cost landscape of every flight that dips
                          # near the tape, destabilising fits far from any real net event, and
                          # it diverges from rich_ball_physics.simulate (refinement can't model
                          # it, so a netted flight abstains anyway). The net VERDICT below
                          # captures net events from the observable speed-collapse signature
                          # without touching the fit. `--net-dynamics` re-enables for A/B.


def net_tape_height(x: float) -> float:
    """Tape height at court-x: NET_H_CENTER at the centre rising to NET_H_POST at the posts."""
    frac = min(abs(float(x) - NET_X_CENTER) / NET_HALF_WIDTH, 1.0)
    return NET_H_CENTER + (NET_H_POST - NET_H_CENTER) * frac


def net_impact_velocity(v_in: np.ndarray) -> np.ndarray:
    """Velocity after a tape clip: the crossing (court-y) component is reversed and heavily
    damped (E_NET), the in-plane (x, z) components damped (NET_TANG). Most closing energy is
    absorbed — the ball drops near-dead at the net, the signature classify_knots keys on."""
    vx, vy, vz = float(v_in[0]), float(v_in[1]), float(v_in[2])
    return np.array([NET_TANG * vx, -E_NET * vy, NET_TANG * vz])


def _accel(vx, vy, vz, wx, wy, wz):
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz) or 1e-9
    ax, ay, az = -_K * _CD * vmag * vx, -_K * _CD * vmag * vy, -_K * _CD * vmag * vz - _G
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    if wmag > 1.0:
        cx, cy, cz = wy * vz - wz * vy, wz * vx - wx * vz, wx * vy - wy * vx
        cmag = math.sqrt(cx * cx + cy * cy + cz * cz)
        if cmag > 1e-9:
            s = _CL * (wmag * _RB / vmag) * _K * vmag * vmag / cmag
            ax, ay, az = ax + s * cx, ay + s * cy, az + s * cz
    return ax, ay, az


def fast_simulate(theta, f0: float, query_frames, fps: float, surface: str,
                  return_velocity: bool = False):
    """Positions at query frames + bounce list; physics-equivalent to simulate()."""
    query = np.asarray(query_frames, float)
    dt = 1.0 / (fps * FAST_SUBSTEPS)
    steps = int(math.ceil((float(query.max()) - f0) * FAST_SUBSTEPS))
    x, y, z = float(theta[0]), float(theta[1]), float(theta[2])
    vx, vy, vz = float(theta[3]), float(theta[4]), float(theta[5])
    wx, wy, wz = float(theta[6]), float(theta[7]), float(theta[8])
    xs = np.empty((steps + 1, 3))
    vs_arr = np.empty((steps + 1, 3))
    xs[0] = (x, y, z)
    vs_arr[0] = (vx, vy, vz)
    bounces = []
    nets = []
    for step in range(steps):
        ax, ay, az = _accel(vx, vy, vz, wx, wy, wz)
        mvx, mvy, mvz = vx + ax * dt / 2, vy + ay * dt / 2, vz + az * dt / 2
        ax, ay, az = _accel(mvx, mvy, mvz, wx, wy, wz)
        nx, ny, nz = x + mvx * dt, y + mvy * dt, z + mvz * dt
        nvx, nvy, nvz = vx + ax * dt, vy + ay * dt, vz + az * dt
        if nz < _RB and nvz < 0 and len(bounces) < 2:
            frac = min(max((z - _RB) / max(z - nz, 1e-9), 0.0), 1.0)
            bx, by, bz = x + frac * (nx - x), y + frac * (ny - y), _RB
            vb = np.array([vx + frac * (nvx - vx), vy + frac * (nvy - vy),
                           vz + frac * (nvz - vz)])
            v_after, w_after, regime = bounce_velocity(vb, np.array([wx, wy, wz]), surface)
            bounces.append({"frame": f0 + (step + frac) / FAST_SUBSTEPS,
                            "x": np.array([bx, by, bz]), "v_in": vb,
                            "v_out": v_after, "regime": regime})
            nvx, nvy, nvz = float(v_after[0]), float(v_after[1]), float(v_after[2])
            wx, wy, wz = float(w_after[0]), float(w_after[1]), float(w_after[2])
            nx, ny, nz = bx + nvx * dt * (1 - frac), by + nvy * dt * (1 - frac), \
                bz + nvz * dt * (1 - frac)
        elif (NET_IN_DYNAMICS and (y - NET_Y) * (ny - NET_Y) < 0 and len(nets) < 1):
            # The ball crosses the net plane this step; clip the tape if it does so below
            # tape height and within the court width. Modeled like a bounce: land on the
            # plane, apply net_impact_velocity, re-integrate the remaining fraction.
            frac = min(max((NET_Y - y) / (ny - y), 0.0), 1.0)
            cx = x + frac * (nx - x)
            cz = z + frac * (nz - z)
            if cz < net_tape_height(cx) + _RB and -0.5 <= cx <= 2 * NET_X_CENTER + 0.5:
                vn = np.array([vx + frac * (nvx - vx), vy + frac * (nvy - vy),
                               vz + frac * (nvz - vz)])
                v_after = net_impact_velocity(vn)
                nets.append({"frame": f0 + (step + frac) / FAST_SUBSTEPS,
                             "x": np.array([cx, NET_Y, cz]), "v_in": vn, "v_out": v_after})
                nvx, nvy, nvz = float(v_after[0]), float(v_after[1]), float(v_after[2])
                nx, ny, nz = cx + nvx * dt * (1 - frac), NET_Y + nvy * dt * (1 - frac), \
                    cz + nvz * dt * (1 - frac)
        x, y, z, vx, vy, vz = nx, ny, nz, nvx, nvy, nvz
        xs[step + 1] = (x, y, z)
        vs_arr[step + 1] = (vx, vy, vz)
    q = np.clip((query - f0) * FAST_SUBSTEPS, 0, steps)
    lo = np.floor(q).astype(int)
    hi = np.minimum(lo + 1, steps)
    alpha = (q - lo)[:, None]
    positions = (1 - alpha) * xs[lo] + alpha * xs[hi]
    velocities = (1 - alpha) * vs_arr[lo] + alpha * vs_arr[hi]
    if return_velocity:
        return positions, velocities, bounces
    return positions, bounces


def fit_segment(frames: np.ndarray, uv: np.ndarray, camera: FrameCamera, fps: float,
                surface: str, seed: np.ndarray | None = None,
                streaks: dict | None = None) -> dict:
    """Fit one flight (theta = p0, v0, spin) to the observations; robust cost + residuals."""
    f0 = float(frames[0])
    P_stack = np.stack([camera.p_at(int(f)) for f in frames])
    # Seed: ground-plane back-projection of the endpoints at 1 m height.
    import cv2
    h0, h1 = camera.h_at(int(frames[0])), camera.h_at(int(frames[-1]))
    xy0 = cv2.perspectiveTransform(np.float32([[uv[0]]]), h0)[0, 0]
    xy1 = cv2.perspectiveTransform(np.float32([[uv[-1]]]), h1)[0, 0]
    duration = max((frames[-1] - frames[0]) / fps, 0.2)
    p0 = np.array([xy0[0], xy0[1], 1.0])
    v0 = np.array([(xy1[0] - xy0[0]) / duration, (xy1[1] - xy0[1]) / duration,
                   4.905 * duration / 2])
    theta0 = np.r_[p0, np.clip(v0, -55, 55), 0.0, 0.0, 0.0]
    if seed is not None:
        theta0 = seed
    # Streak-direction observations: the motion-blur axis at frame i must match the
    # projected flight direction there (mod 180 — a streak has no arrow). Direction
    # needs no exposure calibration, unlike streak length.
    streak_idx = ([i for i, f in enumerate(frames) if int(f) in streaks]
                  if streaks else [])
    streak_angles = np.array([streaks[int(frames[i])][0] for i in streak_idx]) \
        if streak_idx else np.zeros(0)

    def residual(theta):
        xs, bounces = fast_simulate(theta, f0, frames, fps, surface)
        xyz1 = np.hstack([xs, np.ones((len(xs), 1))])
        proj = np.einsum("nij,nj->ni", P_stack, xyz1)
        pixel = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
        pix_res = ((pixel - uv) / PIXEL_SIGMA).ravel()
        # Physical anchors — without them a monocular fit drifts up the camera ray
        # (observed: fitted flights at 20-30 m altitude). Soft, fixed-length residuals:
        # a flight starts at racket height, stays below lob altitude, and its bounces
        # land on or near the court.
        start_z = (theta[2] - 1.5) / 1.5
        apex = max(0.0, float(xs[:, 2].max()) - 12.0) / 0.5
        bounce_pen = 0.0
        for bounce in bounces:
            bx, by = float(bounce["x"][0]), float(bounce["x"][1])
            bounce_pen += (max(0.0, -2.0 - bx) + max(0.0, bx - 12.97)
                           + max(0.0, -3.0 - by) + max(0.0, by - 26.77))
        parts = [pix_res, [start_z, apex, bounce_pen / 0.5]]
        if streak_idx:
            lo = np.maximum(np.array(streak_idx) - 1, 0)
            hi = np.minimum(np.array(streak_idx) + 1, len(frames) - 1)
            deltas = pixel[hi] - pixel[lo]
            predicted = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 180.0
            diff = np.abs(predicted - streak_angles)
            diff = np.minimum(diff, 180.0 - diff)
            parts.append(diff / STREAK_ANGLE_SIGMA_DEG)
        return np.concatenate([np.ravel(part) for part in parts])

    try:
        sol = least_squares(residual, theta0, loss="soft_l1", f_scale=3.0,
                            max_nfev=MAX_NFEV, x_scale=[5, 10, 2, 20, 20, 10, 300, 300, 300])
    except (ValueError, RuntimeError):
        return {"ok": False, "cost": np.inf, "frames": frames}
    res = residual(sol.x)[:len(frames) * 2].reshape(-1, 2)
    per_frame = np.linalg.norm(res, axis=1)
    robust = float(np.sum(np.minimum(per_frame, 6.0) ** 2))
    # Per-observation pixel residual + court region: the raw material for the calibrated noise
    # model and every noise-scaled chi^2 comparison (seg_chi2). `cost` is kept for callers that
    # want the legacy fixed-sigma robust cost (and hand-built test fits without residual arrays).
    return {"ok": True, "cost": robust, "theta": sol.x, "frames": frames, "uv": uv,
            "per_frame": per_frame, "per_frame_px": per_frame * PIXEL_SIGMA,
            "regions": region_of(uv, camera, frames), "f0": f0}


def split_point(fit: dict) -> int | None:
    """Index to split at: the peak of the smoothed residual, away from the edges."""
    per = fit["per_frame"]
    if len(per) < 2 * MIN_SEG_OBS:
        return None
    kernel = np.ones(3) / 3
    smooth = np.convolve(per, kernel, mode="same")
    lo, hi = MIN_SEG_OBS, len(per) - MIN_SEG_OBS
    if lo >= hi:
        return None
    return int(lo + np.argmax(smooth[lo:hi]))


def _onset_true_time(seg: dict, fa: float, fps: float, surface: str,
                     delay_correct: bool) -> float:
    """Sound-arrival onset -> true-contact time, using the flight's fitted position at the
    onset for the sound-travel delay (clamped on-court so a ghosted depth can't inflate it)."""
    if not delay_correct:
        return fa
    # Evaluate the flight position at the onset, clamped to the segment's observed span so
    # an onset outside it (belongs to another flight) can't drive the integrator past its
    # fitted range.
    q = min(max(fa, float(seg["frames"][0])), float(seg["frames"][-1]))
    xs, _ = fast_simulate(seg["theta"], seg["f0"], np.array([q]), fps, surface)
    x, y = float(xs[0][0]), float(xs[0][1])
    x = min(max(x, 0.0), 10.97)
    y = min(max(y, 0.0), 23.77)
    return fa - sound_delay_frames([x, y, 1.0], fps)


TURN_ANGLE_MIN = 40.0      # deg; interior image-trajectory turn below this proposes nothing
TURN_WIN = 3               # observations each side used to measure the local turn
BOUNCE_EXCLUDE_F = 5.0     # a bounce also kinks the image path sharply; skip turn peaks near
                           # the fitted bounce so physics_split proposes only contact kinks


def _turn_profile(uv: np.ndarray) -> np.ndarray:
    """Per-observation turn angle (deg) of the image trajectory: the angle between the mean
    incoming and outgoing pixel-velocity over a TURN_WIN window. A racket contact kinks the
    observed path sharply even when a flexible single-flight fit bends smoothly through it;
    a bounce also kinks it, but the two-flight veto rejects a bounce (one flight already
    models it). Endpoints and near-stationary stretches read 0."""
    n = len(uv)
    ang = np.zeros(n)
    for i in range(TURN_WIN, n - TURN_WIN):
        v1 = uv[i] - uv[i - TURN_WIN]
        v2 = uv[i + TURN_WIN] - uv[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 3.0 or n2 < 3.0:      # too slow to have a trustworthy direction
            continue
        cos = float(np.dot(v1, v2) / (n1 * n2))
        ang[i] = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return ang


def physics_split(done: list[dict], camera: FrameCamera, fps: float, surface: str,
                  streaks=None, noise: NoiseModel = DEFAULT_NOISE) -> list[dict]:
    """Audio-free interior split: propose splits at local maxima of the observed-trajectory
    turn angle (a velocity-discontinuity signal), and keep one only if two flights beat one
    under the noise-calibrated GRIC criterion (split_beats_merge). This finds soft/far contacts
    by physics alone — no audio, no detector — and the criterion rejects bounces (a single
    flight with the bounce in its dynamics already explains them, so the split earns no
    parameter-adjusted chi^2 gain)."""
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(done):
            if not seg.get("ok") or len(seg["frames"]) < 2 * MIN_SEG_OBS:
                continue
            fr, uv = seg["frames"], seg["uv"]
            ang = _turn_profile(uv)
            lo, hi = MIN_SEG_OBS, len(fr) - MIN_SEG_OBS
            # A bounce kinks the image path as sharply as a contact; exclude turn peaks near
            # the flight's own fitted bounce so only contact kinks are proposed.
            _, bounces = fast_simulate(seg["theta"], seg["f0"], np.array([fr[-1]]), fps,
                                       surface)
            bframes = [b["frame"] for b in bounces]

            def near_bounce(frame):
                return any(abs(frame - bounce_frame) <= BOUNCE_EXCLUDE_F
                           for bounce_frame in bframes)
            # local maxima above threshold, strongest first (the veto still rejects any that
            # slip through)
            cand = [j for j in range(lo, hi) if ang[j] >= TURN_ANGLE_MIN
                    and ang[j] >= ang[j - 1] and ang[j] >= ang[j + 1]
                    and not near_bounce(float(fr[j]))]
            cand.sort(key=lambda j: ang[j], reverse=True)
            best = None
            for idx in cand[:6]:
                left = fit_segment(fr[:idx], uv[:idx], camera, fps, surface,
                                   seed=seg.get("theta"), streaks=streaks)
                right = fit_segment(fr[idx:], uv[idx:], camera, fps, surface,
                                    streaks=streaks)
                if not (left["ok"] and right["ok"]):
                    continue
                gain = split_beats_merge(seg_chi2(seg, noise),
                                         seg_chi2(left, noise) + seg_chi2(right, noise),
                                         len(fr), lam=GRIC_LAMBDA)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, left, right)
            if best is not None:
                done[i:i + 1] = [best[1], best[2]]
                changed = True
                break
    return done


def audio_split(done: list[dict], audio_frames, camera: FrameCamera, fps: float,
                surface: str, streaks=None, delay_correct: bool = True,
                noise: NoiseModel = DEFAULT_NOISE) -> list[dict]:
    """Audio proposes, physics disposes: at each audio onset interior to a fitted flight
    with no boundary nearby, try splitting the flight at the observation closest to the
    onset. Keep the split only if two flights genuinely fit better than one (minus a small
    penalty) — a spurious onset inside a real single arc cannot earn it. Recovers soft
    contacts a single 9-parameter fit bends through with low position residual. The onset is
    first corrected to true-contact time (sound-travel delay) so the split lands on the
    occlusion gap around the strike, not the later sound arrival."""
    if not audio_frames:
        return done
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(done):
            if not seg.get("ok") or len(seg["frames"]) < 2 * MIN_SEG_OBS:
                continue
            fr, uv = seg["frames"], seg["uv"]
            f0, f1 = float(fr[0]), float(fr[-1])
            best = None
            for fa_raw in audio_frames:
                fa = _onset_true_time(seg, fa_raw, fps, surface, delay_correct)
                if not (f0 + 2 < fa < f1 - 2):
                    continue
                idx = int(np.argmin(np.abs(fr - fa)))
                if idx < MIN_SEG_OBS or len(fr) - idx < MIN_SEG_OBS:
                    continue
                boundary = (float(fr[idx - 1]) + float(fr[idx])) / 2.0
                if abs(boundary - fa) > 12:   # nearest obs is far from the onset
                    continue
                left = fit_segment(fr[:idx], uv[:idx], camera, fps, surface,
                                   seed=seg.get("theta"), streaks=streaks)
                right = fit_segment(fr[idx:], uv[idx:], camera, fps, surface,
                                    streaks=streaks)
                if not (left["ok"] and right["ok"]):
                    continue
                # Audio onset is independent contact evidence: it enters as a prior credit that
                # lowers the chi^2 bar (same criterion, not a separate smaller margin).
                gain = split_beats_merge(seg_chi2(seg, noise),
                                         seg_chi2(left, noise) + seg_chi2(right, noise),
                                         len(fr), lam=GRIC_LAMBDA,
                                         credit=AUDIO_EVIDENCE_CREDIT)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, left, right)
            if best is not None:
                done[i:i + 1] = [best[1], best[2]]
                changed = True
                break
    return done


def _block_merge(frames_all: np.ndarray, uv_all: np.ndarray, lo: int, hi: int,
                 camera: FrameCamera, fps: float, surface: str, streaks,
                 noise: NoiseModel) -> list[dict]:
    """Bottom-up: fit small blocks, then greedily merge neighbours that share one flight.
    Merging is cheap and well-posed; fitting one flight to a whole rally (top-down splitting)
    is neither. A merge (one flight) is kept unless splitting beats it under the calibrated
    GRIC criterion — the same one-vs-two test used everywhere else."""
    blocks = []
    for b0 in range(lo, hi, BLOCK_OBS):
        b1 = min(b0 + BLOCK_OBS, hi)
        if b1 - b0 < MIN_SEG_OBS and blocks:
            blocks[-1] = (blocks[-1][0], b1)
        else:
            blocks.append((b0, b1))
    done = [fit_segment(frames_all[b0:b1], uv_all[b0:b1], camera, fps, surface,
                        streaks=streaks)
            for b0, b1 in blocks]
    merged = True
    while merged and len(done) > 1:
        merged = False
        best = None
        for i in range(len(done) - 1):
            a, b = done[i], done[i + 1]
            jf = np.r_[a["frames"], b["frames"]]
            juv = np.vstack([a["uv"], b["uv"]])
            # Seed the joint from the left flight's solution: continuing an already fitted
            # flight through the bounce is a far better start than a straight line between
            # distant endpoints.
            seeded = fit_segment(jf, juv, camera, fps, surface,
                                 seed=a.get("theta"), streaks=streaks)
            fresh = fit_segment(jf, juv, camera, fps, surface, streaks=streaks)
            joint = seeded if seg_chi2(seeded, noise) <= seg_chi2(fresh, noise) else fresh
            # Untriggered block boundary: keep it split only if two flights beat one by the
            # blind-merge (MERGE_FACTOR) bar; otherwise MERGE (gain <= 0). Aggressive merging
            # first; triggered splits re-expose the real contacts afterwards.
            gain = split_beats_merge(seg_chi2(joint, noise),
                                     seg_chi2(a, noise) + seg_chi2(b, noise),
                                     len(jf), lam=GRIC_LAMBDA, factor=MERGE_FACTOR)
            if joint["ok"] and gain <= 0 and (best is None or gain < best[0]):
                best = (gain, i, joint)
        if best is not None:
            _, i, joint = best
            done[i:i + 2] = [joint]
            merged = True
    return done


def solve_window(frames_all: np.ndarray, uv_all: np.ndarray, camera: FrameCamera,
                 fps: float, surface: str, boxes=None, ball_obs=None,
                 streaks=None, audio_frames=None,
                 delay_correct: bool = True, use_physics_split: bool = True,
                 use_audio_split: bool = True,
                 dead_ball_gate: bool = True,
                 noise: NoiseModel | None = None
                 ) -> tuple[list[dict], list[dict], NoiseModel]:
    """Split-and-merge over the whole window. Returns (segments, knots, noise_model).

    Two passes: (1) a bottom-up+merge pass under the default noise exposes the long, clean
    single-flight arcs, from which a per-region observation-noise sigma is calibrated; (2) the
    full split-and-merge runs under that calibrated model, so every one-vs-two-flight decision
    (merge, physics/audio split, boundary refinement, force-merge) shares one noise-derived
    GRIC criterion instead of hand-tuned chi^2 margins. Pass a `noise` to skip calibration."""
    # Natural boundaries at long observation gaps: the fit cannot bridge unobserved play.
    pieces = []
    start = 0
    for i in range(1, len(frames_all)):
        if frames_all[i] - frames_all[i - 1] > GAP_BREAK_F:
            pieces.append((start, i))
            start = i
    pieces.append((start, len(frames_all)))
    live = [(lo, hi) for lo, hi in pieces if hi - lo >= MIN_SEG_OBS]

    # Pass 1 — calibrate the noise model from clean single-flight arcs (unless supplied).
    if noise is None:
        prelim: list[dict] = []
        for lo, hi in live:
            prelim += _block_merge(frames_all, uv_all, lo, hi, camera, fps, surface,
                                   streaks, DEFAULT_NOISE)
        noise = calibrate_noise(prelim)

    # Pass 2 — full solve under the calibrated model.
    segments: list[dict] = []
    for lo, hi in live:
        done = _block_merge(frames_all, uv_all, lo, hi, camera, fps, surface, streaks, noise)
        # Physics-only interior split first: a velocity-discontinuity (image-trajectory turn)
        # exposes soft/far contacts the flexible single-flight fit bends through, WITHOUT any
        # audio — so discovery does not depend on the audio channel. The GRIC criterion rejects
        # bounces (one flight already models them, so the split earns no parameter-adjusted gain).
        if use_physics_split:
            done = physics_split(done, camera, fps, surface, streaks=streaks, noise=noise)
            done.sort(key=lambda s: s["frames"][0])
        # Audio-guided split then fills remaining soft contacts that leave no image turn
        # (near-collinear in/out direction): the impact-audio onset proposes the location and
        # the same criterion (with an evidence credit) keeps only genuine improvements.
        if use_audio_split:
            done = audio_split(done, audio_frames, camera, fps, surface, streaks=streaks,
                               delay_correct=delay_correct, noise=noise)
        # Boundary refinement: block merging leaves knots on block edges; slide each boundary
        # locally to the index that minimises the pair's total calibrated chi^2 (pure boundary
        # location — the split already exists, so no parameter penalty applies here).
        done.sort(key=lambda s: s["frames"][0])
        for i in range(len(done) - 1):
            a, b = done[i], done[i + 1]
            if float(b["frames"][0]) - float(a["frames"][-1]) > GAP_BREAK_F:
                continue
            jf = np.r_[a["frames"], b["frames"]]
            juv = np.vstack([a["uv"], b["uv"]])
            base_split = len(a["frames"])
            best_pair = (seg_chi2(a, noise) + seg_chi2(b, noise), a, b)
            for delta in (-8, -4, -2, 2, 4, 8):
                split = base_split + delta
                if split < MIN_SEG_OBS or len(jf) - split < MIN_SEG_OBS:
                    continue
                left = fit_segment(jf[:split], juv[:split], camera, fps, surface,
                                   seed=a.get("theta"), streaks=streaks)
                right = fit_segment(jf[split:], juv[split:], camera, fps, surface,
                                    seed=b.get("theta"), streaks=streaks)
                pair_chi2 = seg_chi2(left, noise) + seg_chi2(right, noise)
                if left["ok"] and right["ok"] and pair_chi2 < best_pair[0]:
                    best_pair = (pair_chi2, left, right)
            done[i], done[i + 1] = best_pair[1], best_pair[2]
        done = force_merge_unsupported(done, boxes, ball_obs, camera, fps, surface,
                                       streaks=streaks, noise=noise)
        segments.extend(done)

    knots = classify_knots(segments, boxes or {}, ball_obs or {}, fps, surface, camera,
                           dead_ball_gate=dead_ball_gate)
    return segments, knots, noise


def force_merge_unsupported(done: list[dict], boxes, ball_obs, camera: FrameCamera,
                            fps: float, surface: str, streaks=None,
                            noise: NoiseModel = DEFAULT_NOISE) -> list[dict]:
    """A break that is not a feasible hit has no physical support, so it carries a prior
    against existence: it survives only if the split clears INFEASIBLE_BREAK_FACTOR times the
    GRIC parameter penalty (a stronger bar than a normal split). This replaces the old fixed
    FORCE_MERGE_FACTOR*SPLIT_PENALTY margin and the separate MERGE_LOCAL_PX pixel gate — the
    calibrated chi^2 already measures local fit quality in noise units."""
    changed = True
    while changed and len(done) > 1:
        changed = False
        knots = classify_knots(done, boxes or {}, ball_obs or {}, fps, surface, camera)
        by_frame = {k['frame']: k for k in knots}
        for i in range(len(done) - 1):
            knot = by_frame.get(float(done[i + 1]['frames'][0]))
            if knot is None or knot['verdict'] in ('hit', 'net'):
                continue  # a real stroke or a labeled net-cord event: do not merge away
            a, b = done[i], done[i + 1]
            jf = np.r_[a['frames'], b['frames']]
            juv = np.vstack([a['uv'], b['uv']])
            seeded = fit_segment(jf, juv, camera, fps, surface, seed=a.get('theta'),
                                 streaks=streaks)
            fresh = fit_segment(jf, juv, camera, fps, surface, streaks=streaks)
            joint = seeded if seg_chi2(seeded, noise) <= seg_chi2(fresh, noise) else fresh
            if not joint['ok']:
                continue
            # An infeasible break has no physical support: it must clear the blind-merge bar
            # times a further INFEASIBLE_BREAK_FACTOR to stand. gain>0 => keep split; else merge.
            gain = split_beats_merge(seg_chi2(joint, noise),
                                     seg_chi2(a, noise) + seg_chi2(b, noise), len(jf),
                                     lam=GRIC_LAMBDA, factor=MERGE_FACTOR * INFEASIBLE_BREAK_FACTOR)
            if gain <= 0:
                done[i:i + 2] = [joint]
                changed = True
                break
    return done


HIT_Z_MIN, HIT_Z_MAX = -0.2, 4.5   # sanity only: fitted z is depth-coupled and noisy
MIN_PLAY_SPEED = 8.0               # pre-serve ball bouncing is 2-5 m/s; play is 15-60
BOX_EXPAND = 0.4                    # player-box expansion for image-space proximity

# Net-cord verdict. A discovered break where a fast incoming flight is killed to near-rest at
# low height and mid-court is the ball dying on the net tape, not a racket contact (owner review,
# 2026-07-20). The fitted contact court-y is depth-ghosted and unreliable at the net, so the
# discriminator is the observable one: a large speed collapse (racket contacts add or preserve
# speed; the tape absorbs it) at low fitted height, with the incoming flight in the mid-court
# net band rather than at either baseline where a struck ball leaves. Excluded from the hit
# chain and from export, kept as a labeled `net` event (RICH_CHART_SPEC kind=net). Measured on
# dev-9: this signature flags 4 breaks (incl. the pt0119 f262 tape reference), none a real
# contact and none reaching emission — a representation fix, aggregate-neutral by construction.
NET_MIN_IN = 15.0        # incoming must be a real flight, not a pre-serve dribble
NET_MAX_OUT = 8.0        # outgoing near-dead: a struck ball leaves faster than this
NET_Z_MAX = 1.4          # fitted contact height near the tape (generous for depth noise)
NET_Y_BAND = 8.0         # |fitted court-y - net plane| band: excludes baseline racket hits
NET_VERDICT = True       # emit the net-cord classification verdict (A/B: set False to disable)

# Dead-ball / play-state gate. Between points the tracker locks onto a parked or slowly
# rolling ball; single-frame flicker on that lock fakes velocity past the energy gate and a
# turn angle past the reversal test, so neither pixel SPEED nor turn can flag it (a real
# far-court arc also moves only a few px/frame). The one signal that separates a parked ball
# from any real flight is the WIDE-WINDOW image span: a held ball's observations sit within a
# small box over ~+/-22 frames (measured <55px on the dev set) while even the slowest real
# far-court arc sweeps >100px through the same window. Measured on the consolidated dev-9 run:
# the 3 parked-ball emits span 29-34px (5-95 pctile) vs a real-contact minimum of 72px.
DEAD_BALL_SPAN_PX = 55.0     # robust wide-window image span below which the ball is parked
DEAD_BALL_WIN_F = 22         # half-window (frames) the span is measured over
DEAD_BALL_MIN_OBS = 6        # too few observations to trust the span; leave the break alone


def wide_window_span(ball_obs: dict, boundary: float) -> float | None:
    """Robust image-space bounding-box span of the ball over +/-DEAD_BALL_WIN_F frames.
    Uses a 5-95 percentile range per axis, not raw max-min, so a single-frame tracker
    flicker (which fakes speed and turn) cannot inflate a parked ball's span past the gate.
    Returns None when there are too few observations to judge."""
    fr_i = int(round(boundary))
    wide = [f for f in ball_obs if abs(f - fr_i) <= DEAD_BALL_WIN_F]
    if len(wide) < DEAD_BALL_MIN_OBS:
        return None
    xs = np.array([ball_obs[f][0] for f in wide])
    ys = np.array([ball_obs[f][1] for f in wide])
    return float((np.percentile(xs, 95) - np.percentile(xs, 5))
                 + (np.percentile(ys, 95) - np.percentile(ys, 5)))


def load_streaks(path: str, clip: str) -> dict:
    """frame -> (angle_deg mod 180, elongation) for confident streak measurements."""
    import csv as _csv
    from rich_ball_physics import frame_num
    streaks = {}
    with open(path, newline="") as handle:
        for row in _csv.DictReader(handle):
            if row["clip"] != clip:
                continue
            if float(row["elongation"]) < STREAK_MIN_ELONG:
                continue
            streaks[frame_num(row["frame"])] = (
                float(row["streak_angle_deg"]), float(row["elongation"]))
    return streaks


def load_boxes(path: str, clip: str) -> dict:
    """Raw image-space player boxes per side per frame (540-line convention, same as the
    ball track). Image-space proximity is depth-robust; the fitted knot position is not."""
    import csv as _csv
    from rich_ball_physics import frame_num
    boxes: dict[str, dict[int, tuple]] = {"near": {}, "far": {}}
    with open(path, newline="") as handle:
        for row in _csv.DictReader(handle):
            if row.get("clip") != clip:
                continue
            boxes[row["side"]][frame_num(row["frame"])] = (
                float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
    return boxes


def near_box_side(boxes: dict, ball_obs: dict, boundary: float) -> str | None:
    """Which player's (expanded) box contains the ball observation nearest the break?"""
    frames = [f for f in ball_obs if abs(f - boundary) <= 6]
    if not frames:
        return None
    f = min(frames, key=lambda q: abs(q - boundary))
    u, v = ball_obs[f]
    for side in ("near", "far"):
        cand = [g for g in boxes.get(side, {}) if abs(g - f) <= 8]
        if not cand:
            continue
        x0, y0, x1, y1 = boxes[side][min(cand, key=lambda q: abs(q - f))]
        dx, dy = (x1 - x0) * BOX_EXPAND, (y1 - y0) * BOX_EXPAND
        if x0 - dx <= u <= x1 + dx and y0 - dy <= v <= y1 + dy:
            return side
    return None


def classify_knots(segments: list[dict], boxes: dict, ball_obs: dict, fps: float,
                   surface: str, camera: FrameCamera | None = None,
                   dead_ball_gate: bool = True) -> list[dict]:
    """A discovered break is a HIT only if physics allows it: a player at the ball in
    image space, a racket-feasible velocity change, and a sane fitted height. Everything
    else is an unsupported break (kept for diagnostics, never claimed as a contact)."""
    from impact import racket_hit_feasible
    knots = []
    for a, b in zip(segments, segments[1:]):
        # Contact happens inside the unobserved gap between the incoming flight's last
        # observation and the outgoing flight's first (the racket occludes the ball);
        # the gap midpoint is the unbiased time estimate.
        boundary = (float(a["frames"][-1]) + float(b["frames"][0])) / 2.0
        if float(b["frames"][0]) - float(a["frames"][-1]) > GAP_BREAK_F:
            continue  # separated by an observation gap, not by a discovered break
        if not (a.get("ok") and b.get("ok")):
            continue
        # incoming state: continue the left flight to the boundary
        x_in_arr, v_in_arr, _ = fast_simulate(a["theta"], a["f0"], np.array([boundary]),
                                              fps, surface, return_velocity=True)
        v_in = v_in_arr[0]
        v_out = b["theta"][3:6]
        z = float(b["theta"][2])
        # Fitted contact court position (for the sound-delay estimate): average the incoming
        # flight extrapolated to the boundary with the outgoing flight's launch position
        # (both meet at the contact). Depth-noisy — _hitter_xyz falls back to the side
        # baseline when it lands off-court.
        court_xy = ((x_in_arr[0][:2] + b["theta"][0:2]) / 2.0)
        near_player = near_box_side(boxes, ball_obs, boundary) if boxes else None
        # Hitter side from the OUTGOING flight's screen direction: a near-court hit
        # moves the ball up-screen (pixel y decreasing), a far-court hit down-screen.
        # Robust to depth error, unlike ball position at the break.
        side = None
        if camera is not None:
            xs_dir, _ = fast_simulate(b["theta"], b["f0"],
                                      np.array([b["f0"], b["f0"] + 3.0]), fps, surface)
            P0 = camera.p_at(int(b["f0"]))
            pix = []
            for xyz in xs_dir:
                proj = P0 @ np.r_[xyz, 1.0]
                pix.append(proj[:2] / max(proj[2], 1e-9))
            dy = pix[1][1] - pix[0][1]
            if abs(dy) > 0.5:
                side = "near" if dy < 0 else "far"
        if side is None:
            side = near_player
        if side is None and ball_obs:
            import cv2
            near = [f for f in ball_obs if abs(f - boundary) <= 6]
            if near:
                f = min(near, key=lambda q: abs(q - boundary))
                xy = cv2.perspectiveTransform(
                    np.float32([[ball_obs[f]]]), camera.h_at(f))[0, 0]
                side = "near" if xy[1] > 11.885 else "far"
        feasible = bool(racket_hit_feasible(v_in, v_out))
        height_ok = HIT_Z_MIN < z < HIT_Z_MAX
        speed_in_v = float(np.linalg.norm(v_in))
        speed_out_v = float(np.linalg.norm(v_out))
        if max(speed_in_v, speed_out_v) < MIN_PLAY_SPEED:
            # The server bouncing the ball pre-serve makes real low-energy flights with
            # real bounces near a player — every evidence gate passes. Energy does not.
            verdict = "low_energy_break"
        else:
            verdict = "hit" if (height_ok and feasible and near_player) else "unsupported_break"
        # Net-cord override: a fast incoming flight killed to near-rest at low height in the
        # mid-court net band is the ball dying on the tape, not a racket contact. Runs before
        # the dead-ball gate: it is a real (energetic) flight, so the energy/turn gates pass,
        # but the outgoing speed collapse and net-band position are the tape signature. -0.2 <=
        # z guards against ghosted-negative fits reading as net at a real high contact.
        net_band = abs(float(court_xy[1]) - NET_Y) <= NET_Y_BAND
        if (NET_VERDICT and speed_in_v >= NET_MIN_IN and speed_out_v <= NET_MAX_OUT
                and HIT_Z_MIN < z < NET_Z_MAX and net_band):
            verdict = "net"
        # Dead-ball override: a break whose ball is parked across the wide window is a
        # between-points tracker lock, not a stroke — excluded from hits AND from export.
        # Runs last because a parked ball can otherwise pass the energy/feasibility gates on
        # flicker. A real reversal sweeps out and back, so it clears the span test on its own.
        span = wide_window_span(ball_obs, boundary)
        if dead_ball_gate and span is not None and span < DEAD_BALL_SPAN_PX:
            verdict = "dead_ball"
        knots.append({"frame": boundary, "z": z, "side": side,
                      "court_xy": (float(court_xy[0]), float(court_xy[1])),
                      "speed_in": float(np.linalg.norm(v_in)),
                      "speed_out": float(np.linalg.norm(v_out)),
                      "near_player": near_player, "height_ok": height_ok,
                      "racket_feasible": feasible, "wide_span": span,
                      "verdict": verdict})
    return knots


MIN_HIT_GAP_S = 0.5
MAX_HIT_GAP_S = 3.0      # a rally exchange never takes longer than this...
RESET_GAP_S = 4.0        # ...but after a longer pause (fault, second serve) rhythm resets
MIN_KNOT_SCORE = 2.0     # a chain member needs real evidence, not just feasibility
W_AUDIO, W_BOX, W_FEAS = 2.0, 1.5, 1.0
AUDIO_TOL_F = 5.0


def select_hit_chain(knots: list[dict], audio_frames: list[float], fps: float) -> None:
    """Tennis is a rhythm: hits alternate sides with travel-time gaps. Pick the legal
    alternating chain that collects the most independent evidence (audio onset, player
    box, racket feasibility); chain members become the claimed hits. Pure priors + DP."""
    for knot in knots:
        audio = any(abs(knot["frame"] - a) <= AUDIO_TOL_F for a in audio_frames)
        knot["audio"] = bool(audio)
        # A dead-ball or net-cord break is not a stroke; keep it off the chain (score below
        # the eligibility floor) and preserve its verdict rather than reclassifying it below.
        if knot.get("verdict") in ("dead_ball", "net"):
            knot["score"] = 0.0
            continue
        knot["score"] = (W_AUDIO * audio + W_BOX * (knot["near_player"] is not None)
                         + W_FEAS * knot["racket_feasible"])
    order = sorted(range(len(knots)), key=lambda i: knots[i]["frame"])
    best_at: list[float] = [0.0] * len(order)
    parent: list[int | None] = [None] * len(order)
    for oi, i in enumerate(order):
        best_at[oi] = knots[i]["score"]
        if knots[i]["score"] < MIN_KNOT_SCORE:
            best_at[oi] = -1e9  # not chain-eligible
            continue
        for oj in range(oi):
            j = order[oj]
            if best_at[oj] < 0:
                continue
            gap = (knots[i]["frame"] - knots[j]["frame"]) / fps
            if gap < MIN_HIT_GAP_S:
                continue
            rhythm_ok = gap <= MAX_HIT_GAP_S and not (
                knots[i]["side"] is not None and knots[j]["side"] is not None
                and knots[i]["side"] == knots[j]["side"])
            reset_ok = gap >= RESET_GAP_S  # pause: fault serve, second serve, let
            if not (rhythm_ok or reset_ok):
                continue
            candidate = best_at[oj] + knots[i]["score"]
            if candidate > best_at[oi]:
                best_at[oi] = candidate
                parent[oi] = oj
    if not order:
        return
    end = int(np.argmax(best_at))
    chain = set()
    oi: int | None = end
    while oi is not None:
        chain.add(order[oi])
        oi = parent[oi]
    for idx, knot in enumerate(knots):
        if knot.get("verdict") in ("dead_ball", "net"):
            continue   # gated out upstream; not chain-eligible, verdict preserved
        knot["verdict"] = "hit" if idx in chain else "unsupported_break"


# ---- boundary contacts: terminal (rally-ending) + serve, one-sided by construction --------
# Discovery defines a contact as the MEETING OF TWO fitted flights, and select_hit_chain keeps
# only a legally-alternating interior chain. Both boundary contacts break that assumption:
#   * the SERVE opens the point with no incoming flight (handled at window start);
#   * the TERMINAL (last) stroke closes it with no subsequent hit — its outgoing flight is
#     truncated by exactly the things that END points (a director cut mid-flight, the net, or a
#     final bounce sequence settling into the dead-ball signature), so it either never forms a
#     fittable outgoing segment or, when it does, the alternating chain drops it as a dangling
#     leaf. The 2026-07-21 terminal-miss strata confirmed this: ~52 of the genuine last-contact
#     losses are a player-supported break that DOES exist in-window but was gate-rejected
#     ("player_break_unfit"), plus ~24 camera-clipped strokes with no outgoing arc at all.
# This module makes both first-class. Acceptance reuses the SAME physical witnesses every other
# emission uses — a racket-feasible impulse, a plausible height, an independent audio/box
# witness, and a legal round-trip gap from the neighbouring emitted hit — so NO constant is
# tuned on the 62 dev labels (the dev9b lesson). `terminal_reason` is DESCRIPTIVE only (how the
# outgoing flight ends); the accept gate deliberately does NOT key on the outcome class, because
# the strata showed terminal misses are flat across net / wide / long / winner.
TERMINAL_TRUNC_F = 15.0   # an outgoing flight whose last observation sits within this many
                          # frames of the window's last observation, still airborne, is a
                          # truncation (director cut / frame exit) — observation-derived, not a
                          # margin fitted to labels.
SERVE_LAUNCH_MIN = 15.0   # a served ball leaves the racket well above rally-idle speed; reused
                          # from the legacy serve-insertion threshold (physical, not label-fit).
TERMINAL_REQUIRE_WITNESS = True  # gate the terminal on an independent audio/box witness in
                                 # ADDITION to a legal outgoing termination. On = precision-safe
                                 # default; --terminal-physics-only sets it False for the A/B.


def _outgoing_segment(knot: dict, segments: list[dict]) -> dict | None:
    """The fitted flight launched by this break: the segment starting nearest AFTER the knot
    time (a knot is the midpoint of the gap between its two flights)."""
    F = float(knot["frame"])
    after = [s for s in segments if s.get("ok") and float(s["frames"][0]) >= F - 2.0]
    if not after:
        return None
    return min(after, key=lambda s: float(s["frames"][0]))


def terminal_reason(knot: dict, segments: list[dict], ball_obs: dict, fps: float,
                    surface: str, later_verdicts: tuple = ()) -> str | None:
    """How the terminal stroke's OUTGOING flight ends — and, as a GATE, WHETHER it ends the
    point at all. Returns one of the legal terminations a final contact needs no subsequent hit
    to justify, or None when the outgoing is a healthy play-speed flight that does NOT terminate
    (meaning another contact must follow — so this break is not the last stroke). The three
    legal terminations mirror the physical ways points end:
      * camera_cut   — observations stop while the ball is still airborne (director cut / the
                       arc leaves the frame); includes the case with no fittable outgoing arc;
      * net          — the outgoing dies slow and low in the mid-court net band (tape);
      * dead_ball_bounce — the outgoing decays below play speed / a dead-ball or net verdict
                       break follows (the final bounce sequence settling out).
    Purely physical / observation-derived; no label-fit constant."""
    if any(v in ("dead_ball", "net") for v in later_verdicts):
        return "dead_ball_bounce"    # the ball's death was independently flagged downstream
    out = _outgoing_segment(knot, segments)
    last_obs = max(ball_obs) if ball_obs else None
    if out is None:
        return "camera_cut"          # no fittable outgoing arc: the frames end at a cut
    end_f = float(out["frames"][-1])
    xs, vs, _ = fast_simulate(out["theta"], out["f0"], np.array([end_f]), fps, surface,
                              return_velocity=True)
    end_speed = float(np.linalg.norm(vs[0]))
    end_y = float(xs[0][1])
    if last_obs is not None and end_f >= last_obs - TERMINAL_TRUNC_F and end_speed > MIN_PLAY_SPEED:
        return "camera_cut"          # observations stop while still airborne and moving
    if end_speed <= NET_MAX_OUT and abs(end_y - NET_Y) <= NET_Y_BAND:
        return "net"                 # dies slow and low in the mid-court net band
    if end_speed < MIN_PLAY_SPEED:
        return "dead_ball_bounce"    # the flight decays below play speed: settling out
    return None                      # a healthy flight that does not end the point


def _legal_gap(gap_s: float, side_a, side_b) -> bool:
    """A legal round-trip between two consecutive strokes: a rhythm exchange to the OTHER side,
    or a >=RESET_GAP_S pause (fault / second serve / let). Pure physical grammar (round-trip
    time), the same test select_hit_chain uses; no label-fit constant."""
    if gap_s < MIN_HIT_GAP_S:
        return False
    alt_ok = gap_s <= MAX_HIT_GAP_S and not (
        side_a in ("near", "far") and side_a == side_b)
    return alt_ok or gap_s >= RESET_GAP_S


def append_terminal_contact(knots: list[dict], segments: list[dict], ball_obs: dict,
                            fps: float, surface: str) -> dict | None:
    """Promote AT MOST ONE terminal (rally-ending) contact per window: the LAST real stroke
    after the last emitted hit, whose outgoing flight legally terminates. Searched BACKWARD
    from the point end so the ball-settling breaks that trail a winner/error (bounces, roll)
    are passed over and the actual final stroke is taken. One-sided by construction, so it is
    gated on stroke witnesses (audio onset OR player box) + racket feasibility + plausible
    height + a legal gap from the previous emitted hit — never on the presence of an outgoing
    flight, which is exactly what a terminal contact lacks. Returns the promoted knot or None."""
    hits = [k for k in knots if k.get("verdict") == "hit"]
    if not hits:
        return None
    last_hit = max(hits, key=lambda k: k["frame"])
    after = sorted((k for k in knots if k["frame"] > last_hit["frame"] + 1e-6),
                   key=lambda k: k["frame"], reverse=True)
    for pos, k in enumerate(after):
        if k.get("verdict") in ("dead_ball", "net"):
            continue                 # ball-death markers are not strokes
        # NOTE: height_ok is deliberately NOT required here. A terminal contact's outgoing arc
        # is truncated by construction, so its monocular fit ghosts in altitude (z out of the
        # HIT_Z sanity band) — the very signature of the "player_break_unfit" class. Requiring
        # height_ok would systematically reject exactly the strokes this recovery targets. The
        # racket-feasible impulse, the legal outgoing termination, the independent witness and
        # the round-trip gap carry the physical load instead.
        if not k.get("racket_feasible"):
            continue
        gap_s = (k["frame"] - last_hit["frame"]) / fps
        # Alternation is judged on the SAME `side` field the rally grammar / chart-scale
        # partition use, NOT on near_player: a terminal whose side repeats the previous
        # stroke's is a grammar violation (no rally plays two consecutive same-side strokes),
        # and admitting it is the dominant terminal false-positive mode (measured 2026-07-21:
        # near_player-vs-side disagreement let same-side phantoms through). Enforcing the
        # grammar here — the pipeline's core precision mechanism — costs some recall on
        # side-mislabelled true terminals (correctly abstained, per the precision-first rule).
        if not _legal_gap(gap_s, k.get("side"), last_hit.get("side")):
            continue
        # The outgoing flight must LEGALLY TERMINATE (net / camera-cut / bounce-decay) — this
        # is the one-sided emission's core justification and rejects a healthy flight that only
        # means another contact was missed. `after` is sorted latest-first, so the breaks that
        # come after this one (nearer the point end) are the settling tail whose verdicts
        # corroborate a bounce-decay termination.
        later = tuple(x.get("verdict") for x in after[:pos])
        reason = terminal_reason(k, segments, ball_obs, fps, surface, later_verdicts=later)
        if reason is None:
            continue
        # Independent witness (audio onset or player box) on top of the physical termination:
        # ON by default (precision-safe); a camera-cut/net termination is self-evidencing (the
        # point demonstrably ended there) so it stands without one.
        has_witness = bool(k.get("audio")) or (k.get("near_player") is not None)
        if (TERMINAL_REQUIRE_WITNESS and not has_witness
                and reason not in ("camera_cut", "net")):
            continue
        k["verdict"] = "hit"
        k["terminal"] = True
        k["terminal_reason"] = reason
        return k
    return None


def prepend_serve_contact(knots: list[dict], segments: list[dict], fps: float) -> dict | None:
    """Recover a TOSS-started serve (forward mirror of append_terminal_contact). When the
    window opens during the toss, segments[0] is the slow, RISING, near-vertical toss (which
    may even leave the frame top) rather than the serve strike; the strike is then the break
    between the toss and the served flight (segments[1]), and the alternating chain drops it
    because at the strike the ball is above the server's body box (near_player fails). The
    toss->strike signature — a slow rising ball becoming a serve-speed descending one — is
    itself serve-specific and unambiguous (no bounce or rally stroke produces it), so it is
    gated on that kinematics + racket feasibility + plausible height, no label-fit constant.
    Returns the promoted knot or None."""
    if len(segments) < 2 or not (segments[0].get("ok") and segments[1].get("ok")):
        return None
    v0 = segments[0]["theta"][3:6]
    is_toss = (float(np.linalg.norm(v0)) <= SERVE_LAUNCH_MIN and float(v0[2]) > 0
               and abs(float(v0[1])) < MIN_PLAY_SPEED)
    if not is_toss:
        return None
    if float(np.linalg.norm(segments[1]["theta"][3:6])) <= SERVE_LAUNCH_MIN:
        return None
    boundary = (float(segments[0]["frames"][-1]) + float(segments[1]["frames"][0])) / 2.0
    cand = [k for k in knots if abs(k["frame"] - boundary) <= GAP_BREAK_F]
    if not cand:
        return None
    strike = min(cand, key=lambda k: abs(k["frame"] - boundary))
    if strike.get("verdict") in ("hit", "dead_ball", "net"):
        return None
    if not (strike.get("racket_feasible") and strike.get("height_ok")):
        return None
    strike["verdict"] = "hit"
    strike["serve"] = True
    return strike


SNAP_TOL_F = 8.0    # a physics break this close to an impact-audio onset is the same
                    # contact; the onset is the less biased time estimate (the gap
                    # midpoint still carries occlusion-window error).


def snap_knots_to_audio(knots: list[dict], audio_frames: list[float], fps: float,
                        delay_correct: bool = True) -> None:
    """Audio refines timing only (PIPELINE doctrine): move each discovered break onto the
    nearest impact-audio onset within SNAP_TOL_F. Existence stays physics-discovered — this
    only corrects the observation-gap-midpoint time. If two breaks would snap to the same
    onset, the farther one is left where physics put it (they are distinct events).

    delay_correct: subtract each hit's position-dependent sound-travel time from the onset
    before snapping, so the break lands on the TRUE contact instant rather than the delayed
    sound-arrival instant. The delay is larger for far-court hits (~5 frames) than near
    (~1.6), so this removes a systematic near/far timing bias. Note: the fused diagnostic
    reference is itself at the (uncorrected) audio onset, so measured against it the
    corrected snap shows the expected per-side negative offset — the gain is realised at
    true-contact / Tier-1 scoring, not against this reference."""
    if not audio_frames:
        return
    claimed: dict[float, float] = {}   # onset -> abs distance of the break that took it
    delays = [sound_delay_frames(_hitter_xyz(k), fps) if delay_correct else 0.0
              for k in knots]

    def corrected(onset: float, i: int) -> float:
        return onset - delays[i]

    order = sorted(range(len(knots)),
                   key=lambda i: min((abs(corrected(a, i) - knots[i]["frame"])
                                      for a in audio_frames), default=1e9))
    for i in order:
        f = knots[i]["frame"]
        near = [a for a in audio_frames if abs(corrected(a, i) - f) <= SNAP_TOL_F]
        near = [a for a in near if a not in claimed or abs(corrected(a, i) - f) < claimed[a]]
        if not near:
            continue
        onset = min(near, key=lambda a: abs(corrected(a, i) - f))
        claimed[onset] = abs(corrected(onset, i) - f)
        knots[i]["frame"] = float(corrected(onset, i))
        knots[i]["audio_snapped"] = True
        knots[i]["audio_delay_frames"] = round(float(delays[i]), 2)


def load_audio_frames(out_dir: str, clip: str) -> list[float]:
    import csv as _csv
    path = os.path.join(out_dir, "contact_events_audio_full_v1.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return [float(r["frame"]) for r in _csv.DictReader(handle) if r["clip"] == clip]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="rg2025f", choices=["rg2025f", "uso2025f"])
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--surface", default="clay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-contacts", default="",
                        help="also write discovered hits as a features-style contacts CSV")
    parser.add_argument("--streaks-file", default="",
                        help="ball_streaks CSV; adds streak-direction residuals to fits")
    parser.add_argument("--ball-file", default="",
                        help="override the spec ball track (e.g. ball_track_wasb_full_v1.csv)")
    parser.add_argument("--camera-file", default="camera_P_per_frame_v1.npz",
                        help="per-frame projection npz (default v1; pass "
                             "camera_P_per_frame_v2.npz for the drift-free v2 solve — the "
                             "matching court_H_per_frame_v*.npz is loaded automatically)")
    parser.add_argument("--audio-delay", dest="no_audio_delay", action="store_false",
                        help="re-enable sound-travel delay correction (OFF by default: "
                             "the 2026-07-20 sweep showed it overcorrects vs human truth "
                             "— broadcast effects mics sit courtside, not at the camera)")
    parser.add_argument("--no-audio-delay", action="store_true", default=True,
                        help="disable the position-dependent sound-travel delay correction "
                             "when snapping/splitting on audio onsets (A/B diagnostic)")
    parser.add_argument("--no-audio-split", action="store_true",
                        help="disable audio-proposed interior splits (physics-only discovery)")
    parser.add_argument("--audio-snap", dest="no_audio_snap", action="store_false",
                        help="re-enable snapping break times to audio onsets (OFF by "
                             "default: physics gap-midpoint timing beat the snap on human "
                             "labels — sweep 2026-07-20)")
    parser.add_argument("--no-audio-snap", action="store_true", default=True,
                        help="disable snapping break times onto audio onsets (independent "
                             "physics timing)")
    parser.add_argument("--no-physics-split", action="store_true",
                        help="disable the velocity-discontinuity interior split (A/B)")
    parser.add_argument("--no-dead-ball-gate", action="store_true",
                        help="disable the wide-window dead-ball / play-state gate (A/B)")
    parser.add_argument("--net-dynamics", action="store_true",
                        help="enable net-cord tape clips inside fast_simulate (A/B; OFF by "
                             "default — measured to regress discovery). The net verdict in "
                             "classify_knots is independent of this.")
    parser.add_argument("--no-net-verdict", action="store_true",
                        help="disable the net-cord classification verdict (A/B)")
    parser.add_argument("--gric-lambda", type=float, default=0.8,
                        help="chi^2 cost per added parameter for a triggered split "
                             "(pass <=0 for the BIC form ln(N)*params)")
    parser.add_argument("--merge-factor", type=float, default=7.5,
                        help="blind-merge Occam multiple on the split penalty (bar to keep an "
                             "untriggered bottom-up block boundary as two flights)")
    parser.add_argument("--sigma-near", type=float, default=None,
                        help="override the calibrated near-court observation-noise sigma (px)")
    parser.add_argument("--sigma-far", type=float, default=None,
                        help="override the calibrated far-court observation-noise sigma (px)")
    parser.add_argument("--no-terminal", action="store_true",
                        help="disable the terminal (rally-ending) contact recovery (A/B)")
    parser.add_argument("--terminal-physics-only", action="store_true",
                        help="accept a terminal contact on the legal-outgoing-termination gate "
                             "alone, WITHOUT requiring an independent audio/box witness (A/B; "
                             "default requires the witness for precision)")
    parser.add_argument("--no-serve-toss", action="store_true",
                        help="disable the toss-started serve-strike recovery (A/B)")
    parser.add_argument("--export-evidence", action="store_true",
                        help="export every audio- or box-supported break to refinement "
                             "(FIX 2). Diagnosed and correct at the export stage, but measured "
                             "to REGRESS end-to-end recall on the dev-9 human labels: the extra "
                             "contacts destabilise the joint refinement, drifting neighbouring "
                             "real contacts' fitted timing outside +/-5. Off by default until "
                             "the refinement tolerates additional contacts.")
    args = parser.parse_args()
    delay_correct = not args.no_audio_delay
    global NET_IN_DYNAMICS, NET_VERDICT, GRIC_LAMBDA, MERGE_FACTOR, TERMINAL_REQUIRE_WITNESS
    NET_IN_DYNAMICS = args.net_dynamics
    NET_VERDICT = not args.no_net_verdict
    GRIC_LAMBDA = args.gric_lambda
    MERGE_FACTOR = args.merge_factor
    TERMINAL_REQUIRE_WITNESS = not args.terminal_physics_only
    forced_noise = (NoiseModel(args.sigma_near, args.sigma_far)
                    if args.sigma_near is not None and args.sigma_far is not None else None)
    spec = match_spec(args.match)
    out = {"match": args.match, "clips": {}}
    for clip in args.clips:
        camera = FrameCamera(spec.out, clip, args.camera_file)
        if not camera.available():
            out["clips"][clip] = {"error": "no per-frame camera"}
            continue
        ball_path = (os.path.join(spec.out, args.ball_file) if args.ball_file
                     else spec.ball)
        ball = load_ball(ball_path, clip)
        frames = np.array(sorted(ball), float)
        if len(frames) < MIN_SEG_OBS:
            out["clips"][clip] = {"error": "too few ball observations"}
            continue
        uv = np.stack([ball[f] for f in sorted(ball)])
        boxes = load_boxes(spec.players, clip)
        streaks = (load_streaks(os.path.join(spec.out, args.streaks_file), clip)
                   if args.streaks_file else None)
        audio_frames = load_audio_frames(spec.out, clip)
        segments, knots, noise = solve_window(frames, uv, camera, args.fps, args.surface,
                                              boxes=boxes, ball_obs=ball, streaks=streaks,
                                              audio_frames=audio_frames,
                                              delay_correct=delay_correct,
                                              use_physics_split=not args.no_physics_split,
                                              use_audio_split=not args.no_audio_split,
                                              dead_ball_gate=not args.no_dead_ball_gate,
                                              noise=forced_noise)
        # The serve: windows often START at the serve, so its break has no left flight
        # and split-and-merge cannot discover it. The first fitted flight's launch IS the
        # serve when it leaves at play speed.
        if segments and segments[0].get("ok"):
            launch_speed = float(np.linalg.norm(segments[0]["theta"][3:6]))
            first_frame = float(segments[0]["frames"][0])
            already = knots and abs(knots[0]["frame"] - first_frame) < 5
            if launch_speed > SERVE_LAUNCH_MIN and not already:
                knots.insert(0, {
                    "frame": first_frame, "z": float(segments[0]["theta"][2]),
                    "side": "near" if segments[0]["theta"][4] < 0 else "far",
                    "speed_in": 0.0, "speed_out": launch_speed,
                    "near_player": None, "height_ok": True,
                    "racket_feasible": True, "verdict": "hit"})
        if not args.no_audio_snap:
            snap_knots_to_audio(knots, audio_frames, args.fps, delay_correct=delay_correct)
        knots.sort(key=lambda k: k["frame"])
        select_hit_chain(knots, audio_frames, args.fps)
        # Boundary contacts the alternating chain cannot carry (both one-sided by construction),
        # applied AFTER the chain so its from-scratch reclassification does not erase them:
        #   * TERMINAL (last) stroke — no subsequent hit; searched BACKWARD from the point end;
        #   * TOSS-started SERVE — when segments[0] is the slow rising toss (not the strike), the
        #     serve strike is the first break, gate-rejected because the ball is above the
        #     server's body box; promote it FORWARD from the window start.
        if not args.no_terminal:
            append_terminal_contact(knots, segments, ball or {}, args.fps, args.surface)
        if not args.no_serve_toss:
            prepend_serve_contact(knots, segments, args.fps)
        knots.sort(key=lambda k: k["frame"])
        hits = [k for k in knots if k["verdict"] == "hit"]
        out["clips"][clip] = {
            "observations": len(frames),
            "segments": len(segments),
            "hits": len(hits),
            "unsupported_breaks": len(knots) - len(hits),
            "median_cost": float(np.median([s["cost"] for s in segments if s["ok"]]))
            if segments else None,
            "noise": {"sigma_near_px": round(noise.sigma_near, 3),
                      "sigma_far_px": round(noise.sigma_far, 3),
                      "n_near": noise.n_near, "n_far": noise.n_far,
                      "gric_lambda": GRIC_LAMBDA, "merge_factor": MERGE_FACTOR},
            "knots": knots,
        }
        print(clip, f"noise={noise}",
              json.dumps({k: v for k, v in out["clips"][clip].items() if k != "knots"}))
    with open(args.output, "w") as handle:
        json.dump(out, handle, indent=2)
    if args.export_contacts:
        import csv as _csv
        with open(args.export_contacts, "w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=[
                "clip", "frame_A", "side_phys", "phase_A", "source_A", "veto"])
            writer.writeheader()
            for clip, data in out["clips"].items():
                if "knots" not in data:
                    continue
                if args.export_evidence:
                    _export_evidence(writer, clip, data["knots"])
                else:
                    _export_legacy(writer, clip, data["knots"], spec, args.camera_file)
    return 0


def _export_evidence(writer, clip: str, knots: list[dict]) -> None:
    """FIX 2 export gate (opt-in via --export-evidence; OFF by default). Diagnosis: the 9
    audit "discovered-but-not-exported" misses were correctly-timed, audio-confirmed breaks
    that died with near_player=None — NOT a box-coverage gap (the boxes are present at those
    frames) but because at contact the ball is at the racket, beyond the player's body box
    (far-court serves and wide reaches), so image-box containment legitimately fails; the
    legacy path then dropped them when the STALE spec ball track had no nearby observation.
    This gate instead exports any break with independent contact evidence — an impact-audio
    onset or a box association — resolving side from the depth-robust outgoing-flight screen
    direction, never from the stale track. Racket feasibility is deliberately NOT an export
    signal on its own (the monocular fitted velocities make it a weak test many phantom
    splits pass). The hit VERDICT stays conservative (box still required). Correct at this
    stage, but measured to regress end-to-end recall on the human labels (see module note /
    EXPERIMENTS) because the joint refinement is destabilised by the extra contacts."""
    first = True
    for knot in knots:
        if knot["verdict"] in ("low_energy_break", "dead_ball", "net"):
            continue  # pre-play bouncing / parked ball / net-cord event, not a candidate hit
        if not (knot.get("audio") or knot["near_player"] is not None):
            continue  # no independent contact evidence — stays out of refinement
        side = knot["near_player"] or knot.get("side") or "unknown"
        writer.writerow({"clip": clip, "frame_A": f"{knot['frame']:.1f}", "side_phys": side,
                         "phase_A": "serve" if first else "rally",
                         "source_A": "knot_solver", "veto": "0"})
        first = False


def _export_legacy(writer, clip: str, knots: list[dict], spec,
                   camera_file: str = "camera_P_per_frame_v1.npz") -> None:
    """Default export. A non-hit break with no box association has its side resolved from the
    nearest STALE spec-ball observation and is DROPPED when that (sparse) track has no
    observation nearby — the behaviour FIX 2 diagnosed as silently losing correctly-timed
    audio-confirmed far-court contacts. It is kept as the default because, end-to-end, it
    scores BETTER on the human labels than exporting those breaks does (the extra contacts
    destabilise the joint refinement, drifting neighbouring real contacts outside +/-5). FIX 1
    (dead_ball) knots are excluded here too, so the two fixes toggle independently."""
    import cv2
    camera = FrameCamera(spec.out, clip, camera_file)
    ball = load_ball(spec.ball, clip)
    first = True
    for knot in knots:
        if knot["verdict"] in ("dead_ball", "net"):
            continue
        if knot["verdict"] != "hit" and knot["near_player"] is None:
            near = [f for f in ball if abs(f - knot["frame"]) <= 6]
            if not near:
                continue
            f = min(near, key=lambda q: abs(q - knot["frame"]))
            xy = cv2.perspectiveTransform(np.float32([[ball[f]]]), camera.h_at(f))[0, 0]
            side = "near" if xy[1] > 11.885 else "far"
        else:
            side = knot["near_player"] or "unknown"
        if knot["verdict"] == "low_energy_break":
            continue
        if not (knot.get("audio") or knot["near_player"] is not None
                or knot["racket_feasible"]):
            continue
        writer.writerow({"clip": clip, "frame_A": f"{knot['frame']:.1f}", "side_phys": side,
                         "phase_A": "serve" if first else "rally",
                         "source_A": "knot_solver", "veto": "0"})
        first = False


if __name__ == "__main__":
    raise SystemExit(main())
