"""Ground the flight model's aerodynamic constants in real Hawk-Eye landmark geometry.

``physics/flight.py`` ships textbook constants (``C_drag=0.55``, ``C_lift=0.6``,
``C_aerodrag=0.01``) that were never fitted to tennis data.  The public ``ryurko/hawkeye``
corpus (701k shot trajectories, 2019-2021 Roland Garros + Australian Open) supplies, per
strike, a small set of *physical landmarks* -- the hit (contact) point, the incoming apex
(``peak``), the net crossing, the bounce (landing), and for a subset a post-bounce apex.
There are **no frame timestamps**, so the corpus constrains arc *geometry*, not velocity
dynamics directly.  The play-by-play tables add two ground-truth anchors the geometry lacks:
radar ``serve_speed_kph`` and last-shot ``spin_rpm``.

Read-only w.r.t. the physics package (it re-implements the *same* dynamics as
``flight.accel``/``accel_cross``/``w_dot`` and asserts fidelity against ``flight.RK4``).

--mode v1 (original): zero-spin serve anchoring + geometry probes.  Superseded -- kept for
provenance -- because forcing serve spin to zero let drag absorb the neglected Magnus term.

--mode v2 (2026-07-20 mandate): SPIN IS IN THE INVERSE PROBLEM.  A full 3D Magnus integrator
carries topspin (vertical dip) and sidespin (lateral curve); the bounce test feeds the
measured incoming spin to ``physics.impact.court_bounce``.  Ran on the OLD (broken) spin
decay -- superseded by v3.

--mode v3 (default, 2026-07-20 round-3 mandate, on the FIXED spin decay commit 79f034d):
  * Exp A -- KILLER VALIDATION: infer per-shot spin from landmark geometry ALONE (spin free,
    measurement held out) and compare to the measured ``spin_rpm``; a synthetic no-noise
    reference isolates geometric degeneracy from measurement noise.
  * Exp B -- hierarchical fit: per-shot (v0, launch, spin) nuisance vs global C_lift and
    C_spin_decay, with MEASURED spin; does the v2 +2.4cm apex bias survive the decay fix?
  * Exp C -- bounce recommendations re-derived on the corrected flight model.

It writes a report JSON and prints a summary.  It NEVER edits ``physics/*.py``; the
integration decision is left to review.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

# Ensure the repo root is importable (so `physics` resolves in spawned worker processes too).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:                                        # scipy is available in the project venv
    from scipy.optimize import least_squares
except Exception as exc:                    # pragma: no cover - environment guard
    raise SystemExit("hawkeye_flight_fit requires scipy (project .venv)") from exc


# --------------------------------------------------------------------------------------- #
# Physical constants (mirror physics/flight.py; NOT imported so the fitter is standalone,
# but the test asserts these dynamics match flight.RK4 for an in-plane launch).
# --------------------------------------------------------------------------------------- #
G = 9.81
M_BALL = 0.057
R_BALL = 0.0325
J_BALL = 3.155e-5
RHO_AIR = 1.205
K_FORCE = 0.5 * RHO_AIR * math.pi * R_BALL ** 2 / M_BALL      # drag/lift accel constant
K_SPIN4 = 0.5 * RHO_AIR * math.pi * R_BALL ** 4 / J_BALL      # skin-friction torque constant

# Spin decay (physics/flight.py w_dot, fixed 2026-07-20 commit 79f034d):
#   w_dot = -C_spin_decay * k4 * |v| * w   (exponential in w, ~95% retention over 1s @30m/s)
TEXTBOOK = {"C_drag": 0.55, "C_lift": 0.6, "C_spin_decay": 0.025}

RPM_PER_RADSEC = 60.0 / (2.0 * math.pi)
RADSEC_PER_RPM = 1.0 / RPM_PER_RADSEC

# Plausible-range gates for the "reproducible for SOME v0/spin" test (Experiment A).
PLAUSIBLE = {
    "serve": {"v0": (30.0, 70.0), "spin_rpm": (0.0, 5000.0), "theta_deg": (-25.0, 20.0)},
    "rally": {"v0": (8.0, 55.0), "spin_rpm": (0.0, 6000.0), "theta_deg": (-30.0, 45.0)},
}

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


# --------------------------------------------------------------------------------------- #
# Planar flight integrator (vertical incidence plane): state (u, z, vu, vz, w).
#   u  -- horizontal ground distance from the hit (m), monotonically increasing
#   z  -- height (m)
#   vu -- horizontal speed (m/s, >0 forward)
#   vz -- vertical speed (m/s, + up)
#   w  -- spin about the lateral axis (rad/s, + = topspin -> pushes ball down)
# Dynamics identical to flight.accel_cross / flight.w_dot with C_drag, C_lift, C_aerodrag.
# --------------------------------------------------------------------------------------- #
def _accel(vu, vz, w, cdrag, clift):
    vmag = math.hypot(vu, vz)
    if vmag < 1e-9:
        return 0.0, -G
    s = w * R_BALL / vmag                       # spin ratio S (signed)
    cl = clift * s
    # accel_cross: vxdot = -k vmag (C_D vx - C_L vz); vzdot = -g - k vmag (C_L vx + C_D vz)
    au = -K_FORCE * vmag * (cdrag * vu - cl * vz)
    az = -G - K_FORCE * vmag * (cl * vu + cdrag * vz)
    return au, az


def _wdot(vu, vz, w, cspindecay):
    # fixed flight.py form: exponential decay proportional to |v| and w
    vmag = math.hypot(vu, vz)
    return -cspindecay * K_SPIN4 * vmag * w


def integrate(z0, vu0, vz0, w0, cdrag, clift, cspindecay=TEXTBOOK["C_spin_decay"],
              dt=0.004, max_t=4.0, stop_at_ground=True):
    """RK4 flight from (u=0, z=z0) with launch (vu0, vz0) and spin w0.

    Returns arrays (u, z) sampled each step plus the interpolated ground-crossing.  Horizontal
    speed stays positive for realistic launches, so ``u`` is monotone and ``z(u)`` is a
    function suitable for landmark interpolation.
    """
    us = [0.0]
    zs = [z0]
    u, z, vu, vz, w = 0.0, z0, vu0, vz0, w0
    n = int(max_t / dt)
    for _ in range(n):
        au1, az1 = _accel(vu, vz, w, cdrag, clift)
        wd1 = _wdot(vu, vz, w, cspindecay)
        au2, az2 = _accel(vu + au1 * dt / 2, vz + az1 * dt / 2, w + wd1 * dt / 2, cdrag, clift)
        wd2 = _wdot(vu + au1 * dt / 2, vz + az1 * dt / 2, w + wd1 * dt / 2, cspindecay)
        au3, az3 = _accel(vu + au2 * dt / 2, vz + az2 * dt / 2, w + wd2 * dt / 2, cdrag, clift)
        wd3 = _wdot(vu + au2 * dt / 2, vz + az2 * dt / 2, w + wd2 * dt / 2, cspindecay)
        au4, az4 = _accel(vu + au3 * dt, vz + az3 * dt, w + wd3 * dt, cdrag, clift)
        wd4 = _wdot(vu + au3 * dt, vz + az3 * dt, w + wd3 * dt, cspindecay)

        du = dt / 6 * (vu + 2 * (vu + au1 * dt / 2) + 2 * (vu + au2 * dt / 2) + (vu + au3 * dt))
        dz = dt / 6 * (vz + 2 * (vz + az1 * dt / 2) + 2 * (vz + az2 * dt / 2) + (vz + az3 * dt))
        vu = vu + dt / 6 * (au1 + 2 * au2 + 2 * au3 + au4)
        vz = vz + dt / 6 * (az1 + 2 * az2 + 2 * az3 + az4)
        w = w + dt / 6 * (wd1 + 2 * wd2 + 2 * wd3 + wd4)
        u_new, z_new = u + du, z + dz

        if stop_at_ground and z_new < 0.0 and u_new > 0.0:
            frac = z / (z - z_new) if (z - z_new) != 0 else 1.0
            us.append(u + (u_new - u) * frac)
            zs.append(0.0)
            break
        if vu <= 0.0:                            # unphysical: ball reversed -> abort
            us.append(u_new)
            zs.append(z_new)
            break
        u, z = u_new, z_new
        us.append(u)
        zs.append(z)
    return np.asarray(us), np.asarray(zs), (vu, vz, w)


@dataclass
class SimSummary:
    reached_ground: bool
    U: float            # ground range to z=0 (m)
    z_net: float        # height at the net's ground distance (m)
    u_peak: float       # ground distance to apex (m)
    z_peak: float       # apex height (m)
    v_ground: tuple     # (vu, vz, w) at ground (for restitution)


def simulate_landmarks(z0, v0, theta, w, cdrag, clift, u_net, **kw) -> SimSummary:
    vu0, vz0 = v0 * math.cos(theta), v0 * math.sin(theta)
    us, zs, vground = integrate(z0, vu0, vz0, w, cdrag, clift, **kw)
    reached = zs[-1] <= 1e-6 and us[-1] > 0
    U = float(us[-1])
    ipk = int(np.argmax(zs))
    # Parabolic sub-step refinement of the apex: z is flat near the top, so a raw argmax
    # gives a noisy horizontal apex location.  Fit a parabola through the 3 samples.
    u_pk, z_pk = float(us[ipk]), float(zs[ipk])
    if 0 < ipk < len(zs) - 1:
        z0_, z1_, z2_ = zs[ipk - 1], zs[ipk], zs[ipk + 1]
        denom = (z0_ - 2 * z1_ + z2_)
        if denom < -1e-9:                       # concave (real maximum)
            delta = 0.5 * (z0_ - z2_) / denom
            delta = max(-1.0, min(1.0, delta))
            u_pk = float(us[ipk] + delta * (us[ipk + 1] - us[ipk] if delta >= 0
                                            else us[ipk] - us[ipk - 1]))
            z_pk = float(z1_ - 0.25 * (z0_ - z2_) * delta)
    z_net = float(np.interp(u_net, us, zs)) if u_net <= us[-1] else float(zs[-1])
    return SimSummary(reached, U, z_net, u_pk, z_pk, vground)


# --------------------------------------------------------------------------------------- #
# Full 3D flight integrator with Magnus (v2 -- spin IN the state).
#
# Faithful re-implementation of physics.flight.accel / w_dot in court coordinates
# (x = length, net at x=0; y = lateral; z = up).  Validated against flight.RK4 in the test
# for a spinning launch.  Spin is a real 3-vector so topspin (dip) AND sidespin (lateral
# curve) both act.  Launch is parameterised in a heading frame:
#   heading azimuth alpha (court x-y plane), elevation theta, speed v0
#   spin magnitude |w| fixed (from radar spin_rpm), split by tilt psi:
#     w = |w| ( cos(psi) * L + sin(psi) * zhat ),  L = zhat x heading  (topspin axis)
#   psi=0 -> pure topspin (downward Magnus); psi=+/-90deg -> pure sidespin (lateral curve).
# --------------------------------------------------------------------------------------- #
def _accel3d(v, w, cdrag, clift):
    vx, vy, vz = v
    wx, wy, wz = w
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vmag < 1e-9:
        return (0.0, 0.0, -G)
    ax = -K_FORCE * cdrag * vmag * vx
    ay = -K_FORCE * cdrag * vmag * vy
    az = -G - K_FORCE * cdrag * vmag * vz
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    if wmag > 1e-9:
        # w x v
        cx = wy * vz - wz * vy
        cy = wz * vx - wx * vz
        cz = wx * vy - wy * vx
        cmag = math.sqrt(cx * cx + cy * cy + cz * cz)
        if cmag > 1e-9:
            S = wmag * R_BALL / vmag
            cl = clift * S
            scale = K_FORCE * cl * vmag * vmag / cmag
            ax += scale * cx
            ay += scale * cy
            az += scale * cz
    return (ax, ay, az)


def _wdot3d(v, w, cspindecay):
    # fixed flight.py form: w_dot = -C_spin_decay * k4 * |v| * w (exponential, no 1/|w|)
    vmag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    c = -cspindecay * K_SPIN4 * vmag
    return (c * w[0], c * w[1], c * w[2])


def integrate3d(x0, v0, w0, cdrag, clift, cspindecay=TEXTBOOK["C_spin_decay"],
                dt=0.004, max_t=4.0):
    """RK4 in 3D from position x0 with velocity v0 and spin w0.  Stops at z<=0 (ground) or
    when horizontal speed reverses.  Returns list of (x,y,z) and the ground-contact velocity
    + spin (for the bounce model)."""
    x = list(x0)
    v = list(v0)
    w = list(w0)
    pts = [tuple(x)]
    n = int(max_t / dt)

    def add(a, b, s):
        return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)

    vx_sign = 1.0 if v[0] >= 0 else -1.0
    for _ in range(n):
        a1 = _accel3d(v, w, cdrag, clift)
        d1v = _wdot3d(v, w, cspindecay)
        v2 = add(v, a1, dt / 2); w2 = add(w, d1v, dt / 2)
        a2 = _accel3d(v2, w2, cdrag, clift); d2v = _wdot3d(v2, w2, cspindecay)
        v3 = add(v, a2, dt / 2); w3 = add(w, d2v, dt / 2)
        a3 = _accel3d(v3, w3, cdrag, clift); d3v = _wdot3d(v3, w3, cspindecay)
        v4 = add(v, a3, dt); w4 = add(w, d3v, dt)
        a4 = _accel3d(v4, w4, cdrag, clift); d4v = _wdot3d(v4, w4, cspindecay)

        vx_mid = (v[0] + 2 * v2[0] + 2 * v3[0] + v4[0]) / 6
        vy_mid = (v[1] + 2 * v2[1] + 2 * v3[1] + v4[1]) / 6
        vz_mid = (v[2] + 2 * v2[2] + 2 * v3[2] + v4[2]) / 6
        xn = (x[0] + vx_mid * dt, x[1] + vy_mid * dt, x[2] + vz_mid * dt)
        v = [v[0] + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
             v[1] + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]),
             v[2] + dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])]
        w = [w[0] + dt / 6 * (d1v[0] + 2 * d2v[0] + 2 * d3v[0] + d4v[0]),
             w[1] + dt / 6 * (d1v[1] + 2 * d2v[1] + 2 * d3v[1] + d4v[1]),
             w[2] + dt / 6 * (d1v[2] + 2 * d2v[2] + 2 * d3v[2] + d4v[2])]
        if xn[2] < 0.0:
            frac = x[2] / (x[2] - xn[2]) if (x[2] - xn[2]) != 0 else 1.0
            xg = (x[0] + (xn[0] - x[0]) * frac, x[1] + (xn[1] - x[1]) * frac, 0.0)
            pts.append(xg)
            return pts, (v[0], v[1], v[2]), (w[0], w[1], w[2])
        if v[0] * vx_sign <= 0:                 # horizontal reversal -> abort
            pts.append(xn)
            break
        x = list(xn)
        pts.append(xn)
    return pts, (v[0], v[1], v[2]), (w[0], w[1], w[2])


@dataclass
class Land3D:
    reached: bool
    net_y: float
    net_z: float
    bounce_x: float
    bounce_y: float
    apex_z: float
    v_ground: tuple
    w_ground: tuple
    apex_x: float = float("nan")
    apex_y: float = float("nan")


def _interp_at_x(pts, xtarget):
    """Linear interpolation of (y,z) where the path crosses court-x = xtarget."""
    for i in range(1, len(pts)):
        x0, x1 = pts[i - 1][0], pts[i][0]
        if (x0 - xtarget) * (x1 - xtarget) <= 0 and x0 != x1:
            f = (xtarget - x0) / (x1 - x0)
            return (pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]),
                    pts[i - 1][2] + f * (pts[i][2] - pts[i - 1][2]))
    return None


def simulate3d(hit, v0, alpha, theta, spin_mag, psi, cdrag, clift,
               cspindecay=TEXTBOOK["C_spin_decay"], dt=0.004):
    """Integrate a shot and return court-frame landmarks.  ``hit`` is (x,y,z); heading
    ``alpha`` (rad, court x-y); elevation ``theta``; spin magnitude ``spin_mag`` (rad/s) tilt
    ``psi`` (rad, 0=topspin)."""
    ch, sh = math.cos(alpha), math.sin(alpha)
    vh = v0 * math.cos(theta)
    vv = v0 * math.sin(theta)
    vel = (vh * ch, vh * sh, vv)
    # heading unit h=(ch,sh,0); topspin axis L = zhat x h = (-sh, ch, 0); side axis = zhat
    cw, sw = math.cos(psi), math.sin(psi)
    wvec = (spin_mag * cw * (-sh), spin_mag * cw * ch, spin_mag * sw)
    pts, vg, wg = integrate3d(hit, vel, wvec, cdrag, clift, cspindecay=cspindecay, dt=dt)
    reached = pts[-1][2] <= 1e-6
    net = _interp_at_x(pts, 0.0)
    ipk = max(range(len(pts)), key=lambda i: pts[i][2])
    apex_z = pts[ipk][2]
    apex_x, apex_y = pts[ipk][0], pts[ipk][1]
    bx, by = pts[-1][0], pts[-1][1]
    return Land3D(reached, net[0] if net else float("nan"),
                  net[1] if net else float("nan"), bx, by, apex_z, vg, wg,
                  apex_x, apex_y)


# --------------------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------------------- #
@dataclass
class Strike:
    key: tuple
    tour: str            # atp / wta
    tournament: str      # roland_garros / australian_open
    year: int
    surface: str         # clay / hard
    phase: str           # serve / rally
    strike_index: int
    is_last: bool
    hit: tuple
    peak_in: tuple | None
    net: tuple | None
    bounce: tuple | None
    peak_out: tuple | None
    serve_speed_ms: float | None = None
    serve_type: str | None = None
    spin_rpm: float | None = None

    # derived landmark geometry (filled in build)
    U_obs: float = 0.0
    u_net_obs: float = 0.0
    u_peak_obs: float = 0.0
    z0: float = 0.0
    z_net_obs: float = float("nan")
    z_peak_obs: float = float("nan")


def _tournament_meta(fname: str):
    base = os.path.basename(fname)
    tour = "wta" if base.startswith("wta_") else "atp"
    tournament = "roland_garros" if "roland_garros" in base else "australian_open"
    surface = "clay" if tournament == "roland_garros" else "hard"
    m = None
    for y in (2019, 2020, 2021):
        if f"_{y}" in base:
            m = y
            break
    return tour, tournament, (m or 0), surface


def _horiz(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_pbp(pbp_path: str):
    """Return {(point_ID, serve_num): (serve_speed_ms, serve_type, spin_rpm, is_fault)}."""
    out = {}
    with open(pbp_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("point_ID"), str(row.get("serve_num", "")).strip())
            speed = None
            raw = str(row.get("serve_speed_kph", "")).upper().replace("KPH", "").strip()
            try:
                kph = float(raw)
                if kph > 40.0:                 # 0 KPH = missing / faulted radar
                    speed = kph / 3.6
            except ValueError:
                pass
            spin = None
            try:
                s = float(str(row.get("spin_rpm", "")).strip())
                if s > 0:
                    spin = s
            except ValueError:
                pass
            out[key] = (speed, row.get("serve_type"), spin,
                        str(row.get("is_fault", "")).strip())
    return out


def load_strikes(traj_path: str, pbp_dir: str):
    """Parse one match's ball_trajectory into fully-derived Strike records."""
    tour, tournament, year, surface = _tournament_meta(traj_path)
    pbp_path = os.path.join(
        pbp_dir, os.path.basename(traj_path).replace("_ball_trajectory.csv", "_pbp.csv"))
    pbp = load_pbp(pbp_path) if os.path.exists(pbp_path) else {}

    seq = defaultdict(list)
    with open(traj_path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pt = row["point_ID"]
                sv = str(row["serve_num"]).strip()
                si = int(float(row["strike_index"]))
                pos = row["position"].strip().lower()
                x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            except (KeyError, TypeError, ValueError):
                continue
            seq[(pt, sv, si)].append((pos, (x, y, z)))

    # which strike index is the last one of each (point, serve)?
    last_si = {}
    for (pt, sv, si) in seq:
        last_si[(pt, sv)] = max(last_si.get((pt, sv), 0), si)

    strikes = []
    for (pt, sv, si), rows in seq.items():
        hit = peak_in = net = bounce = peak_out = None
        seen_bounce = False
        for pos, xyz in rows:
            if pos == "hit" and hit is None:
                hit = xyz
            elif pos == "net" and net is None:
                net = xyz
            elif pos == "bounce" and bounce is None:
                bounce = xyz
                seen_bounce = True
            elif pos == "peak":
                if not seen_bounce and peak_in is None:
                    peak_in = xyz
                elif seen_bounce and peak_out is None:
                    peak_out = xyz
        if hit is None or bounce is None:
            continue

        phase = "serve" if si == 1 else "rally"
        speed = stype = spin = None
        meta = pbp.get((pt, sv))
        if meta:
            if phase == "serve":
                speed, stype = meta[0], meta[1]
            if si == last_si[(pt, sv)]:
                spin = meta[2]

        s = Strike(
            key=(os.path.basename(traj_path), pt, sv, si), tour=tour,
            tournament=tournament, year=year, surface=surface, phase=phase,
            strike_index=si, is_last=(si == last_si[(pt, sv)]),
            hit=hit, peak_in=peak_in, net=net, bounce=bounce, peak_out=peak_out,
            serve_speed_ms=speed, serve_type=stype, spin_rpm=spin)

        s.z0 = hit[2]
        s.U_obs = _horiz(hit, bounce)
        s.u_net_obs = _horiz(hit, net) if net else float("nan")
        s.z_net_obs = net[2] if net else float("nan")
        if peak_in is not None:
            s.u_peak_obs = _horiz(hit, peak_in)
            s.z_peak_obs = peak_in[2]
        else:                                   # peak == hit (typical flat serve)
            s.u_peak_obs = 0.0
            s.z_peak_obs = hit[2]

        # Basic sanity: forward-travelling, plausible geometry
        if s.U_obs < 1.0 or s.U_obs > 40.0:
            continue
        if s.z0 < 0.2 or s.z0 > 4.0:
            continue
        strikes.append(s)
    return strikes


# --------------------------------------------------------------------------------------- #
# Per-strike inverse solvers
# --------------------------------------------------------------------------------------- #
def _guess_launch(s: Strike):
    z0, zpk, U = s.z0, s.z_peak_obs, s.U_obs
    rise = max(zpk - z0, 0.0)
    t_up = math.sqrt(2 * rise / G) if rise > 0 else 0.0
    t_down = math.sqrt(2 * max(zpk, 0.05) / G)
    T = max(t_up + t_down, 0.25)
    vu = U / T
    vz = G * t_up if t_up > 0 else -0.5
    v0 = math.hypot(vu, vz)
    theta = math.atan2(vz, vu)
    return v0, theta


def solve_free(s: Strike, cdrag, clift, dt=0.004):
    """Experiment A: fit (v0, theta, w) to the full landmark set; return residual + state.

    Targets (m): range U, net height, apex distance u_peak, apex height z_peak.  4 targets,
    3 free params -> the residual measures whether the model can reproduce the *arc shape*
    for some launch state under the given constants.
    """
    v0g, thg = _guess_launch(s)
    has_net = math.isfinite(s.z_net_obs) and math.isfinite(s.u_net_obs)
    u_net = s.u_net_obs if has_net else s.U_obs * 0.5

    def resid(p):
        v0, theta, w = p
        sim = simulate_landmarks(s.z0, v0, theta, w, cdrag, clift, u_net, dt=dt)
        r = [sim.U - s.U_obs, sim.u_peak - s.u_peak_obs, sim.z_peak - s.z_peak_obs]
        r.append((sim.z_net - s.z_net_obs) if has_net else 0.0)
        return r

    # Projectile guess underestimates v0 (ignores drag) and gives no spin sign; the landmark
    # objective is multimodal in spin, so multi-start and keep the lowest-RMSE basin.
    lo = [5.0, math.radians(-40), -700.0]
    hi = [75.0, math.radians(60), 700.0]
    starts = []
    for vscale in (1.15, 1.5):
        for w0 in (0.0, 250.0, -250.0):        # rad/s: flat, topspin, backspin
            starts.append([min(max(v0g * vscale, 6.0), 74.0), thg, w0])
    best = None
    for x0 in starts:
        try:
            res = least_squares(resid, x0, bounds=(lo, hi),
                                xtol=1e-3, ftol=1e-3, max_nfev=40)
        except Exception:
            continue
        rr = float(np.sqrt(np.mean(np.asarray(resid(res.x)) ** 2)))
        if best is None or rr < best[0]:
            best = (rr, res.x)
        if rr < 0.02:                          # good enough, stop early
            break
    if best is None:
        return None
    v0, theta, w = best[1]
    r = np.asarray(resid(best[1]))
    return {
        "v0": float(v0), "theta_deg": float(math.degrees(theta)),
        "spin_rpm": float(w * RPM_PER_RADSEC),
        "rmse": float(math.sqrt(np.mean(r ** 2))),
        "r_range": float(r[0]), "r_upeak": float(r[1]),
        "r_zpeak": float(r[2]), "r_znet": float(r[3]),
    }


def solve_serve_anchored(s: Strike, cdrag, clift, dt=0.004):
    """Experiment B: serve speed fixed to radar; w=0 (flat); fit theta to the *range*, then
    the net-height mismatch is the over-determined residual that discriminates C_drag."""
    if s.serve_speed_ms is None or not math.isfinite(s.z_net_obs):
        return None
    v0 = s.serve_speed_ms
    u_net = s.u_net_obs

    def resid(p):
        (theta,) = p
        sim = simulate_landmarks(s.z0, v0, theta, 0.0, cdrag, clift, u_net, dt=dt)
        return [sim.U - s.U_obs]

    try:
        res = least_squares(resid, [math.radians(-4.0)],
                            bounds=([math.radians(-30)], [math.radians(20)]),
                            xtol=1e-4, ftol=1e-4, max_nfev=40)
    except Exception:
        return None
    theta = float(res.x[0])
    sim = simulate_landmarks(s.z0, v0, theta, 0.0, cdrag, clift, u_net, dt=dt)
    if not sim.reached_ground or abs(sim.U - s.U_obs) > 0.75:
        return None                            # could not match landing at this speed
    return {"theta_deg": math.degrees(theta),
            "r_znet": float(sim.z_net - s.z_net_obs),
            "z_net_obs": s.z_net_obs, "z_net_sim": sim.z_net}


def solve_spin_anchored(s: Strike, cdrag, clift, dt=0.004):
    """Experiment C: spin fixed to spin_rpm (assumed topspin); fit (v0, theta) to range+net;
    the apex-height mismatch is the residual that discriminates C_lift."""
    if s.spin_rpm is None or not math.isfinite(s.z_net_obs) or s.peak_in is None:
        return None
    w = s.spin_rpm * RADSEC_PER_RPM            # topspin (+) assumption
    u_net = s.u_net_obs
    v0g, thg = _guess_launch(s)

    def resid(p):
        v0, theta = p
        sim = simulate_landmarks(s.z0, v0, theta, w, cdrag, clift, u_net, dt=dt)
        return [sim.U - s.U_obs, sim.z_net - s.z_net_obs]

    try:
        res = least_squares(resid, [max(v0g, 8.0), thg],
                            bounds=([5.0, math.radians(-30)], [60.0, math.radians(55)]),
                            xtol=1e-3, ftol=1e-3, max_nfev=50)
    except Exception:
        return None
    v0, theta = res.x
    sim = simulate_landmarks(s.z0, v0, theta, w, cdrag, clift, u_net, dt=dt)
    if not sim.reached_ground or abs(sim.U - s.U_obs) > 0.75 \
            or abs(sim.z_net - s.z_net_obs) > 0.5:
        return None
    return {"v0": float(v0), "theta_deg": math.degrees(theta),
            "r_zpeak": float(sim.z_peak - s.z_peak_obs),
            "z_peak_obs": s.z_peak_obs, "z_peak_sim": sim.z_peak}


def measure_restitution(s: Strike, cdrag, clift, dt=0.004):
    """Experiment D: recover incoming velocity at the bounce from the free arc fit, then use
    the post-bounce apex height to get the vertical COR e_y = vz_out / |vz_in|."""
    if s.peak_out is None:
        return None
    fit = solve_free(s, cdrag, clift, dt=dt)
    if fit is None or fit["rmse"] > 0.35:
        return None
    v0 = fit["v0"]
    theta = math.radians(fit["theta_deg"])
    w = fit["spin_rpm"] * RADSEC_PER_RPM
    vu0, vz0 = v0 * math.cos(theta), v0 * math.sin(theta)
    _, _, (vu_g, vz_g, _) = integrate(s.z0, vu0, vz0, w, cdrag, clift, dt=dt)
    vz_in = abs(vz_g)
    if vz_in < 1.0:
        return None
    h_out = s.peak_out[2]                       # apex height above court after bounce
    if h_out <= R_BALL or h_out > 4.0:
        return None
    vz_out = math.sqrt(2 * G * (h_out - R_BALL))   # ballistic (drag on short hop negligible)
    e_y = vz_out / vz_in
    th1 = math.degrees(math.atan2(vz_in, abs(vu_g)))
    if not (0.05 < e_y < 1.2):
        return None
    return {"e_y": e_y, "theta1_deg": th1, "vz_in": vz_in, "vz_out": vz_out}


# ======================================================================================= #
# v2: spin IN the inverse problem (mandate 2026-07-20)
# ======================================================================================= #
def _heading(s: Strike):
    return math.atan2(s.bounce[1] - s.hit[1], s.bounce[0] - s.hit[0])


def _serve_targets(s: Strike):
    """Observed 3D landmark targets for a serve/shot: net (y,z) and bounce (x,y)."""
    if s.net is None:
        return None
    return (s.net[1], s.net[2], s.bounce[0], s.bounce[1])


def solve_serve_joint(s: Strike, cdrag, clift, spin_mag, dt=0.004, free_spin=False):
    """Serve inverse solve with spin IN the state (known speed = radar).

    Free params: elevation theta, yaw offset dalpha, spin tilt psi (+ spin_mag if free_spin).
    Targets: net (y,z), bounce (x,y).  With speed AND spin fixed this is over-determined by
    one, so the residual discriminates (C_drag, C_lift).  If ``free_spin`` the spin magnitude
    becomes a nuisance param -> the system is exactly determined and the residual collapses,
    which is itself the finding for spin-less serves.
    """
    tg = _serve_targets(s)
    if tg is None or s.serve_speed_ms is None:
        return None
    v0 = s.serve_speed_ms
    a0 = _heading(s)

    def resid(p):
        theta, da, psi = p[0], p[1], p[2]
        sm = p[3] if free_spin else spin_mag
        L = simulate3d(s.hit, v0, a0 + da, theta, sm, psi, cdrag, clift, dt=dt)
        if not math.isfinite(L.net_y) or not L.reached:
            return [10.0, 10.0, 10.0, 10.0]
        return [L.net_y - tg[0], L.net_z - tg[1], L.bounce_x - tg[2], L.bounce_y - tg[3]]

    x0 = [math.radians(-6.0), 0.0, 0.0]
    lo = [math.radians(-30), math.radians(-25), math.radians(-100)]
    hi = [math.radians(20), math.radians(25), math.radians(100)]
    if free_spin:
        x0.append(spin_mag if spin_mag > 0 else 1500 * RADSEC_PER_RPM)
        lo.append(300 * RADSEC_PER_RPM)
        hi.append(4000 * RADSEC_PER_RPM)
    try:
        res = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-4, ftol=1e-4, max_nfev=60)
    except Exception:
        return None
    r = np.asarray(resid(res.x))
    return {"rmse": float(np.sqrt(np.mean(r ** 2))),
            "theta_deg": math.degrees(res.x[0]), "dalpha_deg": math.degrees(res.x[1]),
            "psi_deg": math.degrees(res.x[2]),
            "spin_rpm": (res.x[3] * RPM_PER_RADSEC) if free_spin else spin_mag * RPM_PER_RADSEC}


# ======================================================================================= #
# v3: spin as a free per-shot nuisance on the FIXED-decay model (mandate 2026-07-20 round 3)
# ======================================================================================= #
def infer_spin(s: Strike, cdrag, clift, cspindecay=TEXTBOOK["C_spin_decay"], dt=0.004):
    """KILLER VALIDATION: infer per-shot spin from LANDMARK GEOMETRY ALONE (measurement not
    used).  Fit full launch state (v0, elevation, yaw, spin magnitude, spin tilt psi) to the
    SIX landmark targets net(y,z), bounce(x,y), apex(x,z).  5 free params, 6 targets ->
    over-determined by 1.  Returns the inferred spin magnitude (rpm) for comparison against
    the held-out measured spin_rpm.  This is exactly what a broadcast solver must do."""
    if s.net is None or s.peak_in is None or not math.isfinite(s.z_peak_obs):
        return None
    a0 = _heading(s)
    v0g, _ = _guess_launch(s)
    ax_obs, az_obs = s.peak_in[0], s.z_peak_obs
    tg = (s.net[1], s.net[2], s.bounce[0], s.bounce[1])

    def resid(p):
        v0, theta, da, sm, psi = p
        L = simulate3d(s.hit, v0, a0 + da, theta, sm, psi, cdrag, clift, dt=dt)
        if not math.isfinite(L.net_y) or not L.reached:
            return [10.0] * 6
        return [L.net_y - tg[0], L.net_z - tg[1], L.bounce_x - tg[2], L.bounce_y - tg[3],
                L.apex_z - az_obs, L.apex_x - ax_obs]

    lo = [6.0, math.radians(-25), math.radians(-30), -6000 * RADSEC_PER_RPM, math.radians(-90)]
    hi = [55.0, math.radians(55), math.radians(30), 6000 * RADSEC_PER_RPM, math.radians(90)]
    best = None
    # Multi-start over speed, elevation AND spin: the landmark objective is multi-modal and
    # near-degenerate in spin, so give the optimiser every chance to find the true basin.
    for w0_rpm in (500.0, 1500.0, 2500.0, 3500.0):
        for vscale in (0.9, 1.3):
            for th0 in (8.0, 16.0):
                x0 = [min(max(v0g * vscale, 8.0), 54.0), math.radians(th0), 0.0,
                      w0_rpm * RADSEC_PER_RPM, 0.0]
                try:
                    res = least_squares(resid, x0, bounds=(lo, hi),
                                        xtol=1e-3, ftol=1e-3, max_nfev=50)
                except Exception:
                    continue
                rr = float(np.sqrt(np.mean(np.asarray(resid(res.x)) ** 2)))
                if best is None or rr < best[0]:
                    best = (rr, res.x)
        if best is not None and best[0] < 0.01:
            break
    if best is None:
        return None
    rmse, x = best
    if rmse > 0.20:                              # could not reproduce the arc -> drop
        return None
    return {"rmse": rmse, "v0": float(x[0]), "theta_deg": math.degrees(x[1]),
            "inferred_spin_rpm": float(x[3] * RPM_PER_RADSEC),
            "inferred_topspin_rpm": float(x[3] * RPM_PER_RADSEC * math.cos(x[4])),
            "psi_deg": math.degrees(x[4]),
            "measured_spin_rpm": s.spin_rpm}


def solve_rally_spin(s: Strike, cdrag, clift, spin_mag, dt=0.004, free_spin=False,
                     psi_free=False, cspindecay=TEXTBOOK["C_spin_decay"]):
    """Rally inverse solve, spin IN the state, topspin-dominant axis (psi=0 fixed).

    No radar speed for rally shots, so speed v0 is free.  Free: v0, theta, dalpha
    (+ spin_mag if free_spin).  Targets: net(y,z), bounce(x,y), apex_z (5).  With measured
    spin and psi=0 this is over-determined by 2 -> discriminates the constants (apex height
    is the Magnus/lift signature)."""
    tg = _serve_targets(s)
    if tg is None or s.peak_in is None or not math.isfinite(s.z_peak_obs):
        return None
    a0 = _heading(s)
    v0g, _ = _guess_launch(s)

    # Fit v0, theta, yaw (+psi if psi_free, +spin if free_spin) to the FOUR non-apex
    # landmarks (net y,z; bounce x,y).  The apex height is held out: its residual is the
    # clean Magnus/lift signature (over-determined by 1).  With v0 free, range is matched at
    # essentially any C_drag, so C_drag is not constrained by rally geometry -- only C_lift is
    # (via the apex).  ``psi_free`` allows a sidespin component: if the apex bias vanishes
    # when psi is free, the low-C_lift preference was a spin-axis confound, not real.
    i_psi = 3
    i_spin = 3 + int(psi_free)

    def unpack(p):
        psi = p[i_psi] if psi_free else 0.0
        sm = p[i_spin] if free_spin else spin_mag
        return p[0], p[1], p[2], psi, sm

    def resid(p):
        v0, theta, da, psi, sm = unpack(p)
        L = simulate3d(s.hit, v0, a0 + da, theta, sm, psi, cdrag, clift,
                       cspindecay=cspindecay, dt=dt)
        if not math.isfinite(L.net_y) or not L.reached:
            return [10.0] * 4
        return [L.net_y - tg[0], L.net_z - tg[1], L.bounce_x - tg[2], L.bounce_y - tg[3]]

    x0 = [max(v0g, 10.0), math.radians(10.0), 0.0]
    lo = [6.0, math.radians(-25), math.radians(-25)]
    hi = [55.0, math.radians(55), math.radians(25)]
    if psi_free:
        x0.append(0.0); lo.append(math.radians(-80)); hi.append(math.radians(80))
    if free_spin:
        x0.append(spin_mag if spin_mag > 0 else 1800 * RADSEC_PER_RPM)
        lo.append(0.0); hi.append(5000 * RADSEC_PER_RPM)
    try:
        res = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-3, ftol=1e-3, max_nfev=70)
    except Exception:
        return None
    r = np.asarray(resid(res.x))
    fit_rmse = float(np.sqrt(np.mean(r ** 2)))
    if fit_rmse > 0.30:
        return None
    v0, theta, da, psi, sm = unpack(res.x)
    L = simulate3d(s.hit, v0, a0 + da, theta, sm, psi, cdrag, clift,
                   cspindecay=cspindecay, dt=dt)
    return {"fit_rmse": fit_rmse, "v0": float(v0), "theta_deg": math.degrees(theta),
            "psi_deg": math.degrees(psi), "r_apex": float(L.apex_z - s.z_peak_obs),
            "spin_rpm": sm * RPM_PER_RADSEC}


def spin_dependent_bounce(s: Strike, cdrag, clift, spin_mag, dt=0.004):
    """Recover incoming velocity+spin at the bounce (spin IN the state), then compare the
    MEASURED post-bounce rebound to physics.impact.court_bounce's prediction, conditioned on
    the incoming topspin.  ``spin_mag`` is the measured incoming spin (rad/s)."""
    from physics import impact
    if s.peak_out is None or s.net is None:
        return None
    a0 = _heading(s)
    v0g, _ = _guess_launch(s)
    tg = _serve_targets(s)

    # recover launch (topspin axis) reproducing incoming landmarks
    def resid(p):
        v0, theta, da = p
        L = simulate3d(s.hit, v0, a0 + da, theta, spin_mag, 0.0, cdrag, clift, dt=dt)
        if not math.isfinite(L.net_y) or not L.reached:
            return [10.0] * 5
        return [L.net_y - tg[0], L.net_z - tg[1], L.bounce_x - tg[2],
                L.bounce_y - tg[3], L.apex_z - s.z_peak_obs]

    try:
        res = least_squares(resid, [max(v0g, 10.0), math.radians(10.0), 0.0],
                            bounds=([6.0, math.radians(-25), math.radians(-25)],
                                    [55.0, math.radians(55), math.radians(25)]),
                            xtol=1e-3, ftol=1e-3, max_nfev=60)
    except Exception:
        return None
    r = float(np.sqrt(np.mean(np.asarray(resid(res.x)) ** 2)))
    if r > 0.35:
        return None
    v0, theta, da = res.x
    ch, sh = math.cos(a0 + da), math.sin(a0 + da)
    vh, vv = v0 * math.cos(theta), v0 * math.sin(theta)
    wvec = (spin_mag * (-sh), spin_mag * ch, 0.0)          # pure topspin about L
    pts, vg, wg = integrate3d(s.hit, (vh * ch, vh * sh, vv), wvec, cdrag, clift, dt=dt)
    vgx, vgy, vgz = vg
    vh_in = math.hypot(vgx, vgy)
    vz_in = abs(vgz)
    if vz_in < 1.0 or vh_in < 2.0:
        return None
    # Incoming topspin for the bounce model: use the MEASURED contact spin (physical spin
    # decay over a <1s flight is a few %), NOT flight.py's integrated spin, which decays
    # unphysically fast under C_aerodrag=0.01 (see spin_decay diagnostic).  w_flight_frac
    # records how much of the launch spin flight.py claims survives to the bounce.
    w_top = spin_mag                                       # rad/s topspin at bounce (measured)
    w_flight = math.sqrt(wg[0] ** 2 + wg[1] ** 2 + wg[2] ** 2)
    w_flight_frac = w_flight / spin_mag if spin_mag > 0 else float("nan")
    th1 = math.degrees(math.atan2(vz_in, vh_in))

    # measured rebound from post-bounce apex height
    h_out = s.peak_out[2]
    if h_out <= R_BALL or h_out > 4.0:
        return None
    vz_out_meas = math.sqrt(2 * G * (h_out - R_BALL))
    ey_meas = vz_out_meas / vz_in

    # impact.py prediction (Cross 2020) with the SAME incoming state incl. spin
    try:
        b = impact.court_bounce(vh_in, vz_in, w_top, surface=s.surface)
    except Exception:
        return None
    ey_pred = b.vy2 / vz_in

    out = {"theta1_deg": th1, "w_top_radps": w_top, "w_top_rpm": w_top * RPM_PER_RADSEC,
           "vh_in": vh_in, "vz_in": vz_in, "w_flight_frac": w_flight_frac,
           "ey_meas": ey_meas, "ey_pred": ey_pred, "regime": b.regime,
           "surface": s.surface}

    # horizontal speed retention: measured from bounce -> post-bounce apex geometry
    # time to apex t = vz_out/g; horizontal distance bounce->apex gives vh_out
    t_up = vz_out_meas / G
    if t_up > 1e-3:
        dxo = s.peak_out[0] - s.bounce[0]
        dyo = s.peak_out[1] - s.bounce[1]
        vh_out_meas = math.hypot(dxo, dyo) / t_up
        out["vh_out_meas"] = vh_out_meas
        out["vh_ret_meas"] = vh_out_meas / vh_in
        out["vh_ret_pred"] = b.vx2 / vh_in
    return out


# --------------------------------------------------------------------------------------- #
# Parallel workers (module-level so ProcessPoolExecutor can pickle them)
# --------------------------------------------------------------------------------------- #
def _worker_free(args):
    s, cd, cl, dt = args
    fit = solve_free(s, cd, cl, dt=dt)
    return (s.phase, fit)


def _worker_serve_grid(args):
    """Return (group, [r_znet-per-cd]) for one serve across the whole C_drag grid."""
    s, grid, cl, dt = args
    out = []
    for cd in grid:
        r = solve_serve_anchored(s, cd, cl, dt=dt)
        out.append(r["r_znet"] if r is not None else None)
    return (group_of(s), out)


def _worker_spin_grid(args):
    s, grid, cd, dt = args
    out = []
    for cl in grid:
        r = solve_spin_anchored(s, cd, cl, dt=dt)
        out.append(r["r_zpeak"] if r is not None else None)
    return (group_of(s), out)


def _worker_restitution(args):
    s, cd, cl, dt = args
    r = measure_restitution(s, cd, cl, dt=dt)
    return (s.surface, r)


# ---- v2 spin-aware workers ------------------------------------------------------------- #
def _worker_serve_joint_grid(args):
    """(group, [rmse per (cd,cl)]) for a gold serve (known speed+spin)."""
    s, cdcl, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    out = []
    for cd, cl in cdcl:
        r = solve_serve_joint(s, cd, cl, sm, dt=dt)
        out.append(r["rmse"] if r is not None else None)
    return (group_of(s), out)


def _worker_rally_apex_grid(args):
    """(group, [r_apex per cl]) for a rally shot with measured spin (psi=0 topspin)."""
    s, cl_grid, cd, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    out = []
    for cl in cl_grid:
        r = solve_rally_spin(s, cd, cl, sm, dt=dt)
        out.append(r["r_apex"] if r is not None else None)
    return (group_of(s), out)


def _worker_rally_nuisance_grid(args):
    """(group, [r_apex per cl]) with spin as a FREE nuisance (measurement ignored) -> shows
    identifiability loss when spin is unknown."""
    s, cl_grid, cd, dt = args
    sm = (s.spin_rpm or 1800) * RADSEC_PER_RPM
    out = []
    for cl in cl_grid:
        r = solve_rally_spin(s, cd, cl, sm, dt=dt, free_spin=True)
        out.append(r["r_apex"] if r is not None else None)
    return (group_of(s), out)


def _worker_sidespin_check(args):
    """(|apex|_psi0, |apex|_psifree, |psi_deg|) at fixed constants -> sidespin confound test."""
    s, cd, cl, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    r0 = solve_rally_spin(s, cd, cl, sm, dt=dt)
    rf = solve_rally_spin(s, cd, cl, sm, dt=dt, psi_free=True)
    if r0 is None or rf is None:
        return None
    return (abs(r0["r_apex"]), abs(rf["r_apex"]), abs(rf["psi_deg"]))


def _worker_bounce_v2(args):
    s, cd, cl, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    r = spin_dependent_bounce(s, cd, cl, sm, dt=dt)
    return (s.surface, r)


# ---- v3 workers (fixed-decay model) ---------------------------------------------------- #
def _worker_infer_spin(args):
    s, cd, cl, dt = args
    r = infer_spin(s, cd, cl, dt=dt)
    return (group_of(s), r)


def _worker_rally_apex_measured(args):
    """(group, [r_apex per cl]) at fixed decay, measured spin -> C_lift discriminator."""
    s, cl_grid, cd, decay, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    out = []
    for cl in cl_grid:
        r = solve_rally_spin(s, cd, cl, sm, dt=dt, cspindecay=decay)
        out.append(r["r_apex"] if r is not None else None)
    return (group_of(s), out)


def _worker_decay_grid(args):
    """(group, [r_apex per C_spin_decay]) at textbook C_lift, measured spin."""
    s, decay_grid, cd, cl, dt = args
    sm = s.spin_rpm * RADSEC_PER_RPM
    out = []
    for decay in decay_grid:
        r = solve_rally_spin(s, cd, cl, sm, dt=dt, cspindecay=decay)
        out.append(r["r_apex"] if r is not None else None)
    return (group_of(s), out)


def pmap(fn, items, workers, chunksize=8):
    if workers <= 1 or len(items) < 64:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


# --------------------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------------------- #
def qsummary(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()),
            "q": {f"q{q:g}": float(x) for q, x in zip(QUANTILES, np.quantile(v, QUANTILES))}}


def in_range(val, lo, hi):
    return lo <= val <= hi


# --------------------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------------------- #
def stratified_sample(strikes, max_n, rng):
    """Stratify by phase (serve/rally) and apex-height tercile so the fit is not dominated
    by one regime."""
    if len(strikes) <= max_n:
        return list(strikes)
    heights = sorted(s.z_peak_obs for s in strikes if math.isfinite(s.z_peak_obs))
    if heights:
        t1 = heights[len(heights) // 3]
        t2 = heights[2 * len(heights) // 3]
    else:
        t1 = t2 = 1.0

    def band(s):
        h = s.z_peak_obs
        return 0 if h <= t1 else (1 if h <= t2 else 2)

    buckets = defaultdict(list)
    for s in strikes:
        buckets[(s.phase, band(s))].append(s)
    per = max(1, max_n // max(len(buckets), 1))
    out = []
    for key, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:max_n]


# --------------------------------------------------------------------------------------- #
# Constant grid fitting with train/test split
# --------------------------------------------------------------------------------------- #
def _grid_fit(pool_rows, grid, textbook_val, resid_key_abs, resid_key_signed,
              train_grp, test_grp):
    """Shared train/test grid selection.  ``pool_rows`` = [(group, [resid-per-grid-value])].

    Fits the grid value minimising median |residual| on TRAIN; reports abs+signed residual
    summaries on TRAIN and TEST at both the textbook value and the fitted value.
    """
    ti = min(range(len(grid)), key=lambda i: abs(grid[i] - textbook_val))  # textbook col

    def col_absmed(rows, gi):
        vals = [abs(r[gi]) for grp, r in rows if r[gi] is not None]
        return (float(np.median(vals)) if vals else float("nan")), len(vals)

    train_rows = [row for row in pool_rows if row[0] == train_grp]
    test_rows = [row for row in pool_rows if row[0] == test_grp]

    curve = []
    for gi, gv in enumerate(grid):
        med, n = col_absmed(train_rows, gi)
        curve.append((gv, med, n))
    valid = [c for c in curve if math.isfinite(c[1])]
    star = min(valid, key=lambda c: c[1])[0] if valid else textbook_val
    si = grid.index(star)

    def stats(rows, gi):
        absv = [abs(r[gi]) for grp, r in rows if r[gi] is not None]
        signed = [r[gi] for grp, r in rows if r[gi] is not None]
        return {resid_key_abs: qsummary(absv), resid_key_signed: qsummary(signed)}

    star_key = "C_drag_star" if "net" in resid_key_abs else "C_lift_star"
    curve_key = "C_drag" if "net" in resid_key_abs else "C_lift"
    med_key = "median_abs_net_m" if "net" in resid_key_abs else "median_abs_peak_m"
    return {
        "grid_curve": [{curve_key: c[0], med_key: c[1], "n": c[2]} for c in curve],
        star_key: star,
        "train_textbook": stats(train_rows, ti),
        "train_fitted": stats(train_rows, si),
        "test_textbook": stats(test_rows, ti),
        "test_fitted": stats(test_rows, si),
    }


# --------------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------------- #
def group_of(s: Strike):
    return f"{s.tournament}_{s.year}"


def run(args):
    rng = random.Random(args.seed)
    traj_files = sorted(glob.glob(os.path.join(args.hawkeye_root, "ball_trajectory", "*.csv")))
    if args.max_files:
        rng.shuffle(traj_files)
        traj_files = sorted(traj_files[:args.max_files])
    pbp_dir = os.path.join(args.hawkeye_root, "play_by_play")

    all_strikes = []
    for path in traj_files:
        all_strikes.extend(load_strikes(path, pbp_dir))
    rng.shuffle(all_strikes)

    sample = stratified_sample(all_strikes, args.max_strikes, rng)
    cd0, cl0 = TEXTBOOK["C_drag"], TEXTBOOK["C_lift"]
    W = args.workers

    # ---- Experiment A: free-launch arc reproducibility + identifiability -------------- #
    free_rmse, recov = [], defaultdict(list)
    implausible = defaultdict(int)
    counts = defaultdict(int)
    a_results = pmap(_worker_free, [(s, cd0, cl0, args.dt) for s in sample], W)
    for phase, fit in a_results:
        if fit is None:
            continue
        s = type("P", (), {"phase": phase})()
        counts[s.phase] += 1
        free_rmse.append(fit["rmse"])
        pl = PLAUSIBLE[s.phase]
        if not in_range(fit["v0"], *pl["v0"]):
            implausible[f"{s.phase}/v0"] += 1
        if not in_range(abs(fit["spin_rpm"]), *pl["spin_rpm"]):
            implausible[f"{s.phase}/spin"] += 1
        if not in_range(fit["theta_deg"], *pl["theta_deg"]):
            implausible[f"{s.phase}/theta"] += 1
        recov[f"{s.phase}/v0"].append(fit["v0"])
        recov[f"{s.phase}/spin_rpm"].append(fit["spin_rpm"])
        recov[f"{s.phase}/theta_deg"].append(fit["theta_deg"])

    exp_a = {
        "n_fit": {k: v for k, v in counts.items()},
        "landmark_rmse_m": qsummary(free_rmse),
        "recovered": {k: qsummary(v) for k, v in sorted(recov.items())},
        "implausible_recovery_counts": dict(implausible),
        "note": ("4 landmarks vs 3 free params.  Low RMSE => the model reproduces the arc "
                 "for some launch state, so landmark geometry ALONE does not identify the "
                 "constants; identifiability comes from the serve-speed / spin anchors "
                 "below and from plausibility of the recovered launch state."),
    }

    # ---- Experiment B: C_drag from radar-anchored flat serves ------------------------- #
    flat = [s for s in all_strikes
            if s.phase == "serve" and s.serve_speed_ms is not None
            and math.isfinite(s.z_net_obs)
            and (s.serve_type or "").strip().lower() in ("flat", "pronated")]
    rng.shuffle(flat)
    flat = flat[:args.max_serves]
    by_grp = defaultdict(list)
    for s in flat:
        by_grp[group_of(s)].append(s)
    train_grp = args.train_group
    test_grp = args.test_group
    cd_grid = [round(float(x), 3) for x in np.arange(0.30, 1.31, 0.05)]
    exp_b = {
        "anchor": "radar serve_speed_kph; flat/pronated serves; w=0; fit theta to range",
        "discriminator": "net-height residual (m)",
        "n_flat_serves": len(flat),
        "group_counts": {k: len(v) for k, v in sorted(by_grp.items())},
        "train_group": train_grp, "test_group": test_grp,
        "cdrag_grid": cd_grid,
    }
    if by_grp.get(train_grp) and by_grp.get(test_grp):
        rows_b = pmap(_worker_serve_grid,
                      [(s, cd_grid, cl0, args.dt) for s in flat], W)
        exp_b.update(_grid_fit(rows_b, cd_grid, TEXTBOOK["C_drag"],
                               "abs_net_residual", "signed_net_residual",
                               train_grp, test_grp))

    # ---- Experiment C: C_lift from spin-anchored last shots ---------------------------- #
    spinshots = [s for s in all_strikes
                 if s.spin_rpm is not None and s.phase == "rally"
                 and s.peak_in is not None and math.isfinite(s.z_net_obs)]
    rng.shuffle(spinshots)
    spinshots = spinshots[:args.max_spin]
    by_grp_c = defaultdict(list)
    for s in spinshots:
        by_grp_c[group_of(s)].append(s)
    cl_grid = [round(float(x), 3) for x in np.arange(0.05, 1.55, 0.10)]
    exp_c = {
        "anchor": "spin_rpm (assumed topspin); fit v0,theta to range+net",
        "discriminator": "apex-height residual (m)",
        "assumption_caveat": ("spin_rpm is a magnitude with unknown axis; treated as pure "
                              "topspin. Slice/sidespin shots violate this and add noise."),
        "n_spin_shots": len(spinshots),
        "group_counts": {k: len(v) for k, v in sorted(by_grp_c.items())},
        "train_group": train_grp, "test_group": test_grp,
        "clift_grid": cl_grid,
    }
    if by_grp_c.get(train_grp) and by_grp_c.get(test_grp):
        rows_c = pmap(_worker_spin_grid,
                      [(s, cl_grid, cd0, args.dt) for s in spinshots], W)
        exp_c.update(_grid_fit(rows_c, cl_grid, TEXTBOOK["C_lift"],
                               "abs_peak_residual", "signed_peak_residual",
                               train_grp, test_grp))

    # ---- Experiment D: court restitution e_y from bounce-out apex ---------------------- #
    rest = defaultdict(lambda: defaultdict(list))
    bounce_sample = [s for s in all_strikes if s.peak_out is not None]
    rng.shuffle(bounce_sample)
    bounce_sample = bounce_sample[:args.max_bounce]
    n_bounce = 0
    d_results = pmap(_worker_restitution,
                     [(s, cd0, cl0, args.dt) for s in bounce_sample], W)
    for surf, r in d_results:
        if r is None:
            continue
        n_bounce += 1
        rest[surf]["e_y"].append(r["e_y"])
        rest[surf]["theta1_deg"].append(r["theta1_deg"])

    def angle_bins(theta, ey):
        """Median e_y in 10-30/30-45 deg incidence bins + OLS slope/intercept vs incidence."""
        theta = np.asarray(theta, float)
        ey = np.asarray(ey, float)
        bins = {}
        for lo, hi in ((5, 15), (15, 25), (25, 40)):
            m = (theta >= lo) & (theta < hi)
            if m.sum() >= 20:
                bins[f"{lo}-{hi}deg"] = {"n": int(m.sum()),
                                         "e_y_median": float(np.median(ey[m]))}
        fit = None
        if len(theta) >= 50:
            A = np.vstack([np.ones_like(theta), theta]).T
            coef, *_ = np.linalg.lstsq(A, ey, rcond=None)
            fit = {"intercept": float(coef[0]), "slope_per_deg": float(coef[1]),
                   "cross_intercept": 0.95, "cross_slope_per_deg": -0.005}
        return bins, fit

    # Cross-2020 model prediction (impact.py) at the observed mean incidence per surface.
    def cross_ey(theta1_deg, surface):
        base = 0.95 - 0.005 * theta1_deg
        scale = {"hard": 1.0, "clay": 0.85 / 0.83}[surface]
        return max(0.1, min(0.95, base * scale))

    exp_d = {
        "method": ("incoming vertical speed from free arc fit; outgoing from post-bounce apex "
                   "height (ballistic); e_y = vz_out/|vz_in|. Surface: hard=AO, clay=RG."),
        "n_measured": n_bounce,
        "by_surface": {},
    }
    for surf, d in rest.items():
        ey = d["e_y"]
        th = d["theta1_deg"]
        mean_th = float(np.mean(th)) if th else float("nan")
        bins, linfit = angle_bins(th, ey)
        exp_d["by_surface"][surf] = {
            "e_y_measured": qsummary(ey),
            "mean_incidence_deg": mean_th,
            "cross_model_ey_at_mean_incidence": (
                cross_ey(mean_th, surf) if math.isfinite(mean_th) else None),
            "e_y_by_incidence_bin": bins,
            "e_y_vs_incidence_ols": linfit,
        }

    payload = {
        "schema": "tennis.hawkeye-flight-fit.v1",
        "source": os.path.abspath(args.hawkeye_root),
        "params_config": {"dt": args.dt, "seed": args.seed,
                          "max_strikes": args.max_strikes, "max_serves": args.max_serves,
                          "max_spin": args.max_spin, "max_bounce": args.max_bounce,
                          "max_files": args.max_files},
        "textbook_constants": TEXTBOOK,
        "n_strikes_loaded": len(all_strikes),
        "n_sampled_free_fit": len(sample),
        "limitations": [
            "sparse landmarks only; no frame timestamps -> geometry, not dynamics",
            "per-strike free fit has >= as many DOF as landmarks -> constants only identified "
            "via radar-speed (C_drag) and spin (C_lift) anchors",
            "spin axis unknown; C_lift fit assumes pure topspin on groundstrokes",
            "restitution assumes negligible drag on the short bounce-out hop",
            "tournaments 2019-2021 RG (clay) + AO (hard)",
        ],
        "experiment_A_free_arc_reproducibility": exp_a,
        "experiment_B_cdrag_serve_anchored": exp_b,
        "experiment_C_clift_spin_anchored": exp_c,
        "experiment_D_restitution": exp_d,
    }
    return payload


# --------------------------------------------------------------------------------------- #
# v2 driver: spin IN the inverse problem
# --------------------------------------------------------------------------------------- #
def _grid_star_2d(rows, cdcl):
    """rows = [(group,[rmse per (cd,cl)])] restricted to one group -> argmin cell + surface."""
    surface = []
    best = None
    for gi, (cd, cl) in enumerate(cdcl):
        vals = [r[gi] for _, r in rows if r[gi] is not None]
        med = float(np.median(vals)) if vals else float("nan")
        surface.append({"C_drag": cd, "C_lift": cl, "median_rmse_m": med, "n": len(vals)})
        if vals and (best is None or med < best[1]):
            best = ((cd, cl), med)
    return surface, (best[0] if best else (None, None))


def _apex_grid_summary(rows, cl_grid, group):
    grows = [r for g, r in rows if g == group]
    curve = []
    for gi, cl in enumerate(cl_grid):
        absv = [abs(r[gi]) for r in grows if r[gi] is not None]
        signed = [r[gi] for r in grows if r[gi] is not None]
        curve.append({"C_lift": cl, "median_abs_apex_m": float(np.median(absv)) if absv else None,
                      "median_signed_apex_m": float(np.median(signed)) if signed else None,
                      "n": len(absv)})
    valid = [c for c in curve if c["median_abs_apex_m"] is not None]
    star = min(valid, key=lambda c: c["median_abs_apex_m"])["C_lift"] if valid else None
    return curve, star


def run_v2(args):
    rng = random.Random(args.seed)
    traj_files = sorted(glob.glob(os.path.join(args.hawkeye_root, "ball_trajectory", "*.csv")))
    if args.max_files:
        rng.shuffle(traj_files)
        traj_files = sorted(traj_files[:args.max_files])
    pbp_dir = os.path.join(args.hawkeye_root, "play_by_play")
    all_strikes = []
    for path in traj_files:
        all_strikes.extend(load_strikes(path, pbp_dir))
    rng.shuffle(all_strikes)
    W = args.workers
    cd0, cl0 = TEXTBOOK["C_drag"], TEXTBOOK["C_lift"]
    tr, te = args.train_group, args.test_group

    # ---- Exp A: spin coverage audit -------------------------------------------------- #
    cov = defaultdict(lambda: defaultdict(int))
    serve_gold, rally_gold, bounce_gold = [], [], []
    for s in all_strikes:
        g = group_of(s)
        cov[g]["strikes"] += 1
        if s.phase == "serve" and s.is_last and s.spin_rpm and s.serve_speed_ms and s.net:
            cov[g]["serve_gold"] += 1
            serve_gold.append(s)
        if s.phase == "rally" and s.is_last and s.spin_rpm and s.net and s.peak_in \
                and math.isfinite(s.z_peak_obs):
            cov[g]["rally_gold"] += 1
            rally_gold.append(s)
            if s.peak_out is not None:
                cov[g]["rally_gold_bounceout"] += 1
                bounce_gold.append(s)
    audit = {
        "structural_finding": (
            "spin_rpm is the LAST TRACKED shot's spin, not a per-shot field and NOT a serve "
            "field.  Even nominal rally_length==1 points have the return tracked, so the serve "
            "is almost never the last strike -> serve spin is available only for genuinely "
            "unreturned serves (small, flat-biased).  The large clean measured-spin set is "
            "rally last-shots."),
        "serve_gold_speed_spin_net": len(serve_gold),
        "rally_gold_spin_net_apex": len(rally_gold),
        "rally_gold_with_bounceout": len(bounce_gold),
        "serve_gold_spin_rpm": qsummary([s.spin_rpm for s in serve_gold]),
        "rally_gold_spin_rpm": qsummary([s.spin_rpm for s in rally_gold]),
        "by_group": {g: dict(d) for g, d in sorted(cov.items())},
    }

    # ---- Exp B: serve joint C_drag/C_lift with spin (gold set) ------------------------ #
    rng.shuffle(serve_gold)
    sg = serve_gold[:args.max_serves]
    cdcl = [(cd, cl) for cd in (0.45, 0.55, 0.65, 0.75, 0.85)
            for cl in (0.2, 0.6, 1.0)]
    exp_b = {"design": ("terminal (unreturned) serves with radar speed AND spin; 3D fit of "
                        "elevation, yaw, spin-tilt psi to net(y,z)+bounce(x,y); over-det by 1"),
             "n": len(sg), "grid": [{"C_drag": a, "C_lift": b} for a, b in cdcl]}
    if len(sg) >= 40:
        rows_b = pmap(_worker_serve_joint_grid, [(s, cdcl, args.dt) for s in sg], W)
        surface, star = _grid_star_2d(rows_b, cdcl)
        exp_b["objective_surface"] = surface
        exp_b["argmin_cd_cl"] = list(star)
        rr = [c["median_rmse_m"] for c in surface if math.isfinite(c["median_rmse_m"])]
        exp_b["surface_min"], exp_b["surface_max"] = (float(min(rr)), float(max(rr))) if rr \
            else (None, None)

    # ---- Exp C: rally C_lift from apex, spin IN the model ----------------------------- #
    rng.shuffle(rally_gold)
    rg = rally_gold[:args.max_rally]
    cl_grid = [round(float(x), 3) for x in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)]
    rows_c = pmap(_worker_rally_apex_grid,
                  [(s, cl_grid, cd0, args.dt) for s in rg], W)
    curve_tr, star_tr = _apex_grid_summary(rows_c, cl_grid, tr)
    curve_te, star_te = _apex_grid_summary(rows_c, cl_grid, te)
    exp_c = {
        "design": ("rally last-shots with MEASURED spin as topspin (psi=0); fit v0,theta,yaw "
                   "to net+bounce, hold out apex height as the C_lift discriminator; C_drag "
                   "unconstrained because v0 is free"),
        "cl_grid": cl_grid, "train_group": tr, "test_group": te,
        "train_curve": curve_tr, "train_cl_star": star_tr,
        "test_curve": curve_te, "test_cl_star": star_te,
        "group_counts": {g: sum(1 for s in rg if group_of(s) == g) for g in {tr, te}},
    }

    # sidespin-confound check + nuisance-spin identifiability (subsamples)
    sub = rg[:min(len(rg), args.max_aux)]
    ss = [x for x in pmap(_worker_sidespin_check,
                          [(s, cd0, cl0, args.dt) for s in sub], W) if x is not None]
    if ss:
        exp_c["sidespin_confound_check_at_textbook"] = {
            "median_abs_apex_psi0_m": float(np.median([a for a, _, _ in ss])),
            "median_abs_apex_psifree_m": float(np.median([b for _, b, _ in ss])),
            "median_abs_psi_deg": float(np.median([p for _, _, p in ss])),
            "n": len(ss),
            "reading": ("if psi-free does NOT shrink the apex bias, the low-C_lift preference "
                        "is a real vertical signal, not a sidespin artifact"),
        }
    rows_nu = pmap(_worker_rally_nuisance_grid,
                   [(s, cl_grid, cd0, args.dt) for s in sub], W)
    curve_nu, star_nu = _apex_grid_summary([(tr, r) for _, r in rows_nu], cl_grid, tr)
    exp_c["nuisance_spin_curve"] = curve_nu
    exp_c["nuisance_spin_note"] = (
        "spin freed as a nuisance param (measurement ignored): if the apex curve flattens "
        "vs the measured-spin curve, C_lift is only identifiable WITH measured spin")

    # ---- Exp D: spin-dependent bounce vs impact.py ------------------------------------ #
    rng.shuffle(bounce_gold)
    bg = bounce_gold[:args.max_bounce]
    d_rows = pmap(_worker_bounce_v2, [(s, cd0, cl0, args.dt) for s in bg], W)
    per_surf = defaultdict(list)
    wfrac = []
    for surf, r in d_rows:
        if r is None:
            continue
        per_surf[surf].append(r)
        if math.isfinite(r.get("w_flight_frac", float("nan"))):
            wfrac.append(r["w_flight_frac"])
    exp_d = {
        "design": ("incoming velocity from the spin-in-state arc fit; incoming topspin = "
                   "MEASURED spin (physical decay over <1s is small); compare measured "
                   "rebound (post-bounce apex) to physics.impact.court_bounce, bucketed by "
                   "incoming topspin.  hard=AO, clay=RG."),
        "n_measured": sum(len(v) for v in per_surf.values()),
        "flight_spin_survival_frac_at_bounce": qsummary(wfrac),
        "flight_spin_survival_note": (
            "fraction of launch spin that flight.py's C_aerodrag=0.01 leaves at the bounce; "
            "<<1 means the Magnus effect is largely gone by landing -- a flight-model concern"),
        "by_surface": {},
    }
    tsp_buckets = [(-200, 800), (800, 1600), (1600, 2600), (2600, 6000)]
    for surf, rs in per_surf.items():
        wt = np.array([r["w_top_rpm"] for r in rs])
        eym = np.array([r["ey_meas"] for r in rs])
        eyp = np.array([r["ey_pred"] for r in rs])
        vhm = np.array([r.get("vh_ret_meas", np.nan) for r in rs])
        vhp = np.array([r.get("vh_ret_pred", np.nan) for r in rs])
        buckets = {}
        for lo, hi in tsp_buckets:
            m = (wt >= lo) & (wt < hi)
            if m.sum() >= 15:
                buckets[f"{lo}-{hi}rpm"] = {
                    "n": int(m.sum()),
                    "ey_meas_median": float(np.median(eym[m])),
                    "ey_pred_median": float(np.median(eyp[m])),
                    "ey_gap_median": float(np.median(eym[m] - eyp[m])),
                    "vh_ret_meas_median": float(np.nanmedian(vhm[m])) if np.isfinite(vhm[m]).any() else None,
                    "vh_ret_pred_median": float(np.nanmedian(vhp[m])) if np.isfinite(vhp[m]).any() else None,
                }
        exp_d["by_surface"][surf] = {
            "n": len(rs),
            "ey_meas": qsummary(eym), "ey_pred_impact_py": qsummary(eyp),
            "incoming_topspin_rpm": qsummary(wt),
            "by_incoming_topspin": buckets,
        }

    payload = {
        "schema": "tennis.hawkeye-flight-fit.v2",
        "source": os.path.abspath(args.hawkeye_root),
        "params_config": {k: getattr(args, k) for k in
                          ("dt", "seed", "max_files", "max_serves", "max_rally",
                           "max_bounce", "max_aux", "train_group", "test_group")},
        "textbook_constants": TEXTBOOK,
        "n_strikes_loaded": len(all_strikes),
        "headline": (
            "Spin is now IN the inverse problem (3D Magnus flight + spin-aware bounce).  "
            "Serve spin is essentially unavailable in this corpus (spin_rpm = last-tracked "
            "shot), so the v1 serve-drag signal cannot be cleanly reproduced; the clean "
            "calibration set is rally last-shots with measured spin."),
        "limitations": [
            "serve spin unavailable -> serve joint fit is small and flat-serve biased",
            "spin AXIS unknown; rally uses topspin-dominant assumption (tested via psi-free)",
            "no timestamps: incoming velocity is fit, not measured; bounce-out apex is ballistic",
            "flight.py C_aerodrag=0.01 decays spin very fast (see survival frac)",
            "tournaments 2019-2021 RG (clay) + AO (hard)",
        ],
        "experiment_A_spin_audit": audit,
        "experiment_B_serve_joint_with_spin": exp_b,
        "experiment_C_rally_clift_with_spin": exp_c,
        "experiment_D_spin_dependent_bounce": exp_d,
    }
    return payload


# --------------------------------------------------------------------------------------- #
# v3 driver: hierarchical fit on the FIXED-decay flight model (mandate round 3)
# --------------------------------------------------------------------------------------- #
def _make_synthetic_strikes(rng, cdrag, clift, dt, n):
    """Generate ON-MODEL shots (no noise) with known spin, as Strike records whose landmarks
    are exactly reproducible -> isolates geometric degeneracy from real measurement noise."""
    out = []
    while len(out) < n:
        hit = (-11.0 + rng.uniform(-1, 1), rng.uniform(-3, 3), rng.uniform(0.6, 1.2))
        v0 = rng.uniform(16, 34)
        th = math.radians(rng.uniform(5, 22))
        yaw = math.radians(rng.uniform(-9, 9))
        spin = rng.uniform(600, 3400) * RADSEC_PER_RPM
        L = simulate3d(hit, v0, yaw, th, spin, 0.0, cdrag, clift, dt=dt)
        if not L.reached or not math.isfinite(L.net_y) or L.bounce_x - hit[0] < 4:
            continue
        s = Strike(key=("syn",), tour="atp", tournament="australian_open", year=2020,
                   surface="hard", phase="rally", strike_index=3, is_last=True, hit=hit,
                   peak_in=(L.apex_x, L.apex_y, L.apex_z), net=(0.0, L.net_y, L.net_z),
                   bounce=(L.bounce_x, L.bounce_y, 0.03), peak_out=None,
                   spin_rpm=spin * RPM_PER_RADSEC)
        s.z0 = hit[2]
        s.z_peak_obs = L.apex_z
        s.U_obs = math.hypot(L.bounce_x - hit[0], L.bounce_y - hit[1])
        out.append(s)
    return out


def _infer_report(rows):
    pairs = [(abs(r["inferred_spin_rpm"]), r["measured_spin_rpm"], r["rmse"])
             for _, r in rows if r is not None and r.get("measured_spin_rpm")]
    if len(pairs) < 5:
        return {"n": len(pairs)}
    inf = np.array([p[0] for p in pairs])
    meas = np.array([p[1] for p in pairs])
    err = inf - meas
    out = {
        "n": len(pairs),
        "fit_rmse_m_median": float(np.median([p[2] for p in pairs])),
        "measured_spin_rpm_median": float(np.median(meas)),
        "inferred_spin_rpm_median": float(np.median(inf)),
        "median_abs_error_rpm": float(np.median(np.abs(err))),
        "median_signed_error_rpm": float(np.median(err)),
        "pearson_r": float(np.corrcoef(inf, meas)[0, 1]),
        "spearman_r": float(np.corrcoef(np.argsort(np.argsort(inf)),
                                        np.argsort(np.argsort(meas)))[0, 1]),
        "by_measured_tercile": [],
    }
    order = np.argsort(meas)
    t = len(order) // 3
    for i, (lo, hi) in enumerate([(0, t), (t, 2 * t), (2 * t, len(order))]):
        idx = order[lo:hi]
        out["by_measured_tercile"].append({
            "tercile": i + 1,
            "measured_rpm_range": [float(meas[idx].min()), float(meas[idx].max())],
            "n": int(len(idx)),
            "median_abs_error_rpm": float(np.median(np.abs(err[idx]))),
            "median_inferred_rpm": float(np.median(inf[idx])),
            "median_measured_rpm": float(np.median(meas[idx])),
        })
    return out


def run_v3(args):
    rng = random.Random(args.seed)
    traj_files = sorted(glob.glob(os.path.join(args.hawkeye_root, "ball_trajectory", "*.csv")))
    if args.max_files:
        rng.shuffle(traj_files)
        traj_files = sorted(traj_files[:args.max_files])
    pbp_dir = os.path.join(args.hawkeye_root, "play_by_play")
    all_strikes = []
    for path in traj_files:
        all_strikes.extend(load_strikes(path, pbp_dir))
    rng.shuffle(all_strikes)
    W = args.workers
    cd0, cl0, dec0 = TEXTBOOK["C_drag"], TEXTBOOK["C_lift"], TEXTBOOK["C_spin_decay"]
    tr, te = args.train_group, args.test_group

    rally_gold = [s for s in all_strikes if s.phase == "rally" and s.is_last and s.spin_rpm
                  and s.net and s.peak_in and math.isfinite(s.z_peak_obs)]
    bounce_gold = [s for s in rally_gold if s.peak_out is not None]

    # ---- Exp A: KILLER VALIDATION -- infer spin from geometry, compare to measured ---- #
    rng.shuffle(rally_gold)
    infer_pool = rally_gold[:args.max_infer]
    rows_inf = pmap(_worker_infer_spin, [(s, cd0, cl0, args.dt) for s in infer_pool], W)
    syn_strikes = _make_synthetic_strikes(random.Random(1), cd0, cl0, args.dt, args.max_aux)
    rows_syn = pmap(_worker_infer_spin, [(s, cd0, cl0, args.dt) for s in syn_strikes], W)
    syn_rep = _infer_report(rows_syn)
    syn_rep["note"] = "on-model, no noise -> any error here is pure geometric degeneracy"
    exp_a = {
        "design": ("fit full per-shot launch state (v0, elevation, yaw, spin magnitude, spin "
                   "tilt) to 6 landmarks net(y,z)+bounce(x,y)+apex(x,z) WITHOUT using the "
                   "measured spin; compare inferred spin magnitude to held-out spin_rpm"),
        "real_data": _infer_report(rows_inf),
        "synthetic_best_case": syn_rep,
        "reading": ("if inferred vs measured is uncorrelated even on synthetic no-noise data, "
                    "spin is geometrically UNIDENTIFIABLE from sparse landmarks -- a hard "
                    "limit for any broadcast solver relying on hit/peak/net/bounce alone"),
    }

    # ---- Exp B: C_lift and C_spin_decay with MEASURED spin (fixed-decay model) --------- #
    rg = rally_gold[:args.max_rally]
    cl_grid = [round(float(x), 3) for x in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)]
    rows_cl = pmap(_worker_rally_apex_measured,
                   [(s, cl_grid, cd0, dec0, args.dt) for s in rg], W)
    ctr, star_tr = _apex_grid_summary(rows_cl, cl_grid, tr)
    cte, star_te = _apex_grid_summary(rows_cl, cl_grid, te)

    decay_grid = [round(float(x), 4) for x in (0.0, 0.0125, 0.025, 0.05, 0.1, 0.2)]
    rows_dec = pmap(_worker_decay_grid,
                    [(s, decay_grid, cd0, cl0, args.dt) for s in rg[:args.max_aux2]], W)
    dtr, _ = _apex_grid_summary(rows_dec, decay_grid, tr)
    decay_curve = [{"C_spin_decay": decay_grid[i], "median_abs_apex_m": x["median_abs_apex_m"],
                    "n": x["n"]} for i, x in enumerate(dtr)]

    exp_b = {
        "design": ("MEASURED spin fixed as topspin; fit v0,theta,yaw to net+bounce, hold out "
                   "apex height. On the FIXED-decay model, does the v2 +2.4cm apex bias at "
                   "textbook C_lift survive?  C_drag stays unconstrained (v0 free)."),
        "cl_grid": cl_grid, "train_group": tr, "test_group": te,
        "train_curve": ctr, "train_cl_star": star_tr,
        "test_curve": cte, "test_cl_star": star_te,
        "clift_entanglement_note": (
            "with per-shot spin FREE the Magnus force ~ C_lift*|w| is degenerate (Exp A shows "
            "|w| itself is not recoverable), so C_lift is only identifiable with measured spin"),
        "cspin_decay_curve_train": decay_curve,
        "cspin_decay_note": ("apex residual vs C_spin_decay at textbook C_lift, measured spin; "
                             "flat => the corpus does not constrain the decay constant"),
        "group_counts": {g: sum(1 for s in rg if group_of(s) == g) for g in {tr, te}},
    }

    # ---- Exp C: bounce refit on the FIXED-decay model --------------------------------- #
    rng.shuffle(bounce_gold)
    bg = bounce_gold[:args.max_bounce]
    d_rows = pmap(_worker_bounce_v2, [(s, cd0, cl0, args.dt) for s in bg], W)
    per_surf = defaultdict(list)
    wfrac = []
    for surf, r in d_rows:
        if r is None:
            continue
        per_surf[surf].append(r)
        if math.isfinite(r.get("w_flight_frac", float("nan"))):
            wfrac.append(r["w_flight_frac"])

    def cross_ey(theta1_deg, surface):
        base = 0.95 - 0.005 * theta1_deg
        scale = {"hard": 1.0, "clay": 0.85 / 0.83}[surface]
        return max(0.1, min(0.95, base * scale))

    exp_c = {
        "design": ("incoming velocity from the FIXED-decay arc fit; incoming topspin = measured "
                   "spin; compare measured rebound to impact.court_bounce, bucketed by incoming "
                   "topspin.  Re-derives the v2 bounce recommendations on the corrected model."),
        "n_measured": sum(len(v) for v in per_surf.values()),
        "flight_spin_survival_frac_at_bounce": qsummary(wfrac),
        "by_surface": {},
    }
    tsp_buckets = [(-200, 800), (800, 1600), (1600, 2600), (2600, 6000)]
    for surf, rs in per_surf.items():
        wt = np.array([r["w_top_rpm"] for r in rs])
        eym = np.array([r["ey_meas"] for r in rs])
        eyp = np.array([r["ey_pred"] for r in rs])
        th = np.array([r["theta1_deg"] for r in rs])
        vhm = np.array([r.get("vh_ret_meas", np.nan) for r in rs])
        vhp = np.array([r.get("vh_ret_pred", np.nan) for r in rs])
        buckets = {}
        for lo, hi in tsp_buckets:
            m = (wt >= lo) & (wt < hi)
            if m.sum() >= 15:
                buckets[f"{lo}-{hi}rpm"] = {
                    "n": int(m.sum()),
                    "ey_meas_median": float(np.median(eym[m])),
                    "ey_pred_median": float(np.median(eyp[m])),
                    "ey_gap_median": float(np.median(eym[m] - eyp[m])),
                    "mean_incidence_deg": float(np.mean(th[m])),
                    "vh_ret_meas_median": float(np.nanmedian(vhm[m])) if np.isfinite(vhm[m]).any() else None,
                    "vh_ret_pred_median": float(np.nanmedian(vhp[m])) if np.isfinite(vhp[m]).any() else None,
                }
        # simple e_y intercept recommendation: measured e_y regressed to zero-topspin & mean angle
        exp_c["by_surface"][surf] = {
            "n": len(rs),
            "ey_meas": qsummary(eym), "ey_pred_impact_py": qsummary(eyp),
            "ey_gap_median": float(np.median(eym - eyp)),
            "mean_incidence_deg": float(np.mean(th)),
            "cross_model_ey_at_mean_incidence": cross_ey(float(np.mean(th)), surf),
            "incoming_topspin_rpm": qsummary(wt),
            "by_incoming_topspin": buckets,
        }

    payload = {
        "schema": "tennis.hawkeye-flight-fit.v3",
        "source": os.path.abspath(args.hawkeye_root),
        "params_config": {k: getattr(args, k) for k in
                          ("dt", "seed", "max_files", "max_infer", "max_rally", "max_bounce",
                           "max_aux", "max_aux2", "train_group", "test_group")},
        "textbook_constants": TEXTBOOK,
        "flight_model": "FIXED spin decay (commit 79f034d): w_dot = -C_spin_decay*k4*|v|*w",
        "n_strikes_loaded": len(all_strikes),
        "n_rally_gold": len(rally_gold), "n_bounce_gold": len(bounce_gold),
        "headline": (
            "On the corrected flight model: (A) per-shot spin is NOT inferable from sparse "
            "landmarks (degenerate even on synthetic no-noise data) -> a broadcast solver "
            "cannot recover spin from hit/peak/net/bounce; (B) with MEASURED spin, C_lift is "
            "identifiable and the apex bias is re-measured; (C) bounce recommendations "
            "re-derived on the fixed model."),
        "limitations": [
            "spin AXIS unknown; topspin-dominant assumption for rally (Exp B)",
            "no timestamps: incoming velocity is fit, not clocked; bounce-out apex is ballistic",
            "serve spin unavailable (see v2 audit) -> C_drag remains unconstrained here",
            "tournaments 2019-2021 RG (clay) + AO (hard)",
        ],
        "experiment_A_killer_spin_inference": exp_a,
        "experiment_B_clift_cspindecay_measured_spin": exp_b,
        "experiment_C_bounce_refit_fixed_model": exp_c,
    }
    return payload


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hawkeye-root", default="data/external/ryurko-hawkeye")
    p.add_argument("--output")
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--max-files", type=int, default=0, help="0 = all matches")
    p.add_argument("--max-strikes", type=int, default=4000, help="Experiment A sample")
    p.add_argument("--max-serves", type=int, default=8000, help="Experiment B pool")
    p.add_argument("--max-spin", type=int, default=6000, help="Experiment C pool")
    p.add_argument("--max-bounce", type=int, default=4000, help="Experiment D pool")
    p.add_argument("--train-group", default="australian_open_2020")
    p.add_argument("--test-group", default="australian_open_2021")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    p.add_argument("--mode", choices=("v1", "v2", "v3"), default="v3",
                   help="v1 = zero-spin serves; v2 = spin in state (old decay); "
                        "v3 = fixed-decay hierarchical fit + spin-inference validation")
    # v2/v3 sample sizes
    p.add_argument("--max-rally", type=int, default=6000, help="rally-gold pool (C_lift)")
    p.add_argument("--max-aux", type=int, default=1500, help="sidespin / synthetic subsample")
    p.add_argument("--max-infer", type=int, default=3000, help="v3 spin-inference pool")
    p.add_argument("--max-aux2", type=int, default=1500, help="v3 C_spin_decay subsample")
    return p


def _summary_v2(payload):
    a = payload["experiment_A_spin_audit"]
    print(f"[A] serve_gold={a['serve_gold_speed_spin_net']} "
          f"rally_gold={a['rally_gold_spin_net_apex']} "
          f"bounce_gold={a['rally_gold_with_bounceout']}")
    b = payload["experiment_B_serve_joint_with_spin"]
    if "argmin_cd_cl" in b:
        print(f"[B] serve joint argmin (cd,cl)={b['argmin_cd_cl']} "
              f"surface rmse {b.get('surface_min')}-{b.get('surface_max')} m (n={b['n']})")
    c = payload["experiment_C_rally_clift_with_spin"]
    tc = {x["C_lift"]: x["median_abs_apex_m"] for x in c["test_curve"]}
    print(f"[C] rally C_lift* test={c['test_cl_star']} (textbook 0.6); "
          f"|apex| @0.05={tc.get(0.05)} @0.6={tc.get(0.6)}")
    sc = c.get("sidespin_confound_check_at_textbook")
    if sc:
        print(f"    sidespin check: |apex| psi0={sc['median_abs_apex_psi0_m']:.4f} "
              f"psifree={sc['median_abs_apex_psifree_m']:.4f}")
    d = payload["experiment_D_spin_dependent_bounce"]
    print(f"[D] spin survival to bounce (flight.py) median="
          f"{d['flight_spin_survival_frac_at_bounce'].get('median')}")
    for surf, v in d["by_surface"].items():
        print(f"    {surf}: ey_meas={v['ey_meas'].get('median')} "
              f"ey_pred(impact.py)={v['ey_pred_impact_py'].get('median')} (n={v['n']})")


def _summary_v3(payload):
    a = payload["experiment_A_killer_spin_inference"]
    r = a["real_data"]
    syn = a["synthetic_best_case"]
    print(f"[A] KILLER spin inference (n={r.get('n')}): median abs err={r.get('median_abs_error_rpm')} rpm, "
          f"Pearson r={r.get('pearson_r')}; synthetic best-case r={syn.get('pearson_r')} "
          f"(err {syn.get('median_abs_error_rpm')} rpm)")
    b = payload["experiment_B_clift_cspindecay_measured_spin"]
    tc = {x["C_lift"]: (x["median_abs_apex_m"], x["median_signed_apex_m"]) for x in b["test_curve"]}
    print(f"[B] C_lift* test={b['test_cl_star']} (textbook 0.6); "
          f"|apex|@0.6={tc.get(0.6, (None,))[0]} signed@0.6={tc.get(0.6,(None,None))[1]} @0.05={tc.get(0.05,(None,))[0]}")
    c = payload["experiment_C_bounce_refit_fixed_model"]
    print(f"[C] bounce spin survival={c['flight_spin_survival_frac_at_bounce'].get('median')}")
    for surf, v in c["by_surface"].items():
        print(f"    {surf}: ey_meas={round(v['ey_meas']['median'],3)} "
              f"ey_pred={round(v['ey_pred_impact_py']['median'],3)} gap={round(v['ey_gap_median'],3)} (n={v['n']})")


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = {"v1": run, "v2": run_v2, "v3": run_v3}[args.mode](args)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(rendered)
        print(f"wrote {args.output}")
    if args.mode == "v3":
        _summary_v3(payload)
    elif args.mode == "v2":
        _summary_v2(payload)
    else:
        a = payload["experiment_A_free_arc_reproducibility"]["landmark_rmse_m"]
        print(f"[A] free-arc landmark RMSE median={a.get('median')} m (n={a.get('n')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
