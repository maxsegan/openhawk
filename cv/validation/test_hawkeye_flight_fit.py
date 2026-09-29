"""Tests for hawkeye_flight_fit.

Two things matter for trust: (1) the standalone planar integrator must reproduce the SAME
dynamics as ``physics/flight.py`` (so a fitted constant means the same thing there), and
(2) the inverse solver must recover a known launch state from synthetic landmarks.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import hawkeye_flight_fit as hff
from physics import flight


def test_integrator_matches_flight_rk4_inplane():
    """In-plane launch: our planar RK4 must match flight.RK4 (full 3D) to cm precision."""
    z0, v0, theta, w = 1.2, 28.0, math.radians(12.0), 200.0  # topspin
    vu0, vz0 = v0 * math.cos(theta), v0 * math.sin(theta)

    params = dict(flight.default_params)
    params["C_drag"], params["C_lift"], params["C_spin_decay"] = 0.55, 0.6, 0.025
    x0 = np.array([0.0, 0.0, z0])
    vvec = np.array([vu0, 0.0, vz0])
    # flight.py: +w[1] with v in +x gives Magnus with z-component -k*C_L*R*w*vx (down) -> topspin
    wvec = np.array([0.0, w, 0.0])
    _, xarr, _, _ = flight.RK4(0.001, 20000, x0, vvec, wvec, params=params, verbose=False)
    ref_range = xarr[-1, 0]
    ref_peak = xarr[:, 2].max()

    us, zs, _ = hff.integrate(z0, vu0, vz0, w, 0.55, 0.6, cspindecay=0.025, dt=0.001)
    assert abs(us[-1] - ref_range) < 0.05, (us[-1], ref_range)
    assert abs(zs.max() - ref_peak) < 0.02, (zs.max(), ref_peak)


def test_integrator_no_spin_matches():
    z0, v0, theta = 2.8, 50.0, math.radians(-6.0)     # flat serve-like
    vu0, vz0 = v0 * math.cos(theta), v0 * math.sin(theta)
    params = dict(flight.default_params)
    params["C_lift"] = 0.6
    x0 = np.array([0.0, 0.0, z0])
    # flight.RK4 divides by |w| and |w x v|; with w=0 disable lift/aerodrag (no effect when
    # spin is zero anyway).  Our integrator guards these divisions internally.
    _, xarr, _, _ = flight.RK4(0.001, 20000, x0, np.array([vu0, 0.0, vz0]),
                               np.array([0.0, 0.0, 0.0]), lift=False, aerodrag=False,
                               params=params, verbose=False)
    us, zs, _ = hff.integrate(z0, vu0, vz0, 0.0, 0.55, 0.6, dt=0.001)
    assert abs(us[-1] - xarr[-1, 0]) < 0.05


def test_solve_free_recovers_synthetic():
    """Generate landmarks from a known launch, then check the free solver recovers it."""
    z0, v0_true, th_true, w_true = 1.0, 30.0, math.radians(15.0), 150.0
    u_net = 6.0
    sim = hff.simulate_landmarks(z0, v0_true, th_true, w_true, 0.55, 0.6, u_net, dt=0.002)

    s = hff.Strike(key=("t",), tour="atp", tournament="australian_open", year=2020,
                   surface="hard", phase="rally", strike_index=3, is_last=False,
                   hit=(0, 0, z0), peak_in=(sim.u_peak, 0, sim.z_peak),
                   net=(0, 0, sim.z_net), bounce=(sim.U, 0, 0.03), peak_out=None)
    s.z0 = z0
    s.U_obs = sim.U
    s.u_net_obs = u_net
    s.z_net_obs = sim.z_net
    s.u_peak_obs = sim.u_peak
    s.z_peak_obs = sim.z_peak

    fit = hff.solve_free(s, 0.55, 0.6, dt=0.002)
    assert fit is not None
    assert fit["rmse"] < 0.05, fit
    assert abs(fit["v0"] - v0_true) < 3.0, fit
    assert abs(fit["theta_deg"] - math.degrees(th_true)) < 4.0, fit


def test_serve_anchored_discriminates_cdrag():
    """A serve arc generated at C_drag=0.55 should have a smaller net-height residual when
    scored at 0.55 than at a very different drag -- i.e. the anchor is discriminating."""
    z0, v0, th, w = 2.85, 52.0, math.radians(-5.0), 0.0
    u_net = 11.9
    sim = hff.simulate_landmarks(z0, v0, th, w, 0.55, 0.6, u_net, dt=0.002)

    def make(cd_true_sim):
        s = hff.Strike(key=("t",), tour="atp", tournament="australian_open", year=2020,
                       surface="hard", phase="serve", strike_index=1, is_last=False,
                       hit=(0, 0, z0), peak_in=None, net=(0, 0, sim.z_net),
                       bounce=(sim.U, 0, 0.03), peak_out=None,
                       serve_speed_ms=v0, serve_type="Flat")
        s.z0 = z0
        s.U_obs = sim.U
        s.u_net_obs = u_net
        s.z_net_obs = sim.z_net
        s.u_peak_obs = 0.0
        s.z_peak_obs = z0
        return s

    s = make(0.55)
    r_true = hff.solve_serve_anchored(s, 0.55, 0.6, dt=0.002)
    r_wrong = hff.solve_serve_anchored(s, 0.80, 0.6, dt=0.002)
    assert r_true is not None
    assert abs(r_true["r_znet"]) < 0.03, r_true
    if r_wrong is not None:                    # wrong drag => larger net-height mismatch
        assert abs(r_wrong["r_znet"]) > abs(r_true["r_znet"]), (r_true, r_wrong)


def test_integrate3d_matches_flight_rk4_with_spin():
    """Full 3D integrator with a general spin vector must match flight.RK4 (the reference
    3D integrator) to cm precision -- this is what makes a fitted constant meaningful in
    physics/flight.py."""
    x0 = np.array([-11.0, 0.5, 2.8])
    v0 = np.array([38.0, -1.5, -1.0])        # serve-ish, travelling +x
    w0 = np.array([30.0, 250.0, 120.0])       # mixed topspin + sidespin (rad/s)
    params = dict(flight.default_params)
    params["C_drag"], params["C_lift"], params["C_spin_decay"] = 0.55, 0.6, 0.025

    _, xarr, _, _ = flight.RK4(0.0005, 40000, x0, v0, w0, params=params, verbose=False)
    pts, _, _ = hff.integrate3d(tuple(x0), tuple(v0), tuple(w0), 0.55, 0.6,
                                cspindecay=0.025, dt=0.0005)
    ours = np.asarray(pts)
    # compare final (ground) position and lateral excursion
    assert abs(ours[-1, 0] - xarr[-1, 0]) < 0.05, (ours[-1], xarr[-1])
    assert abs(ours[-1, 1] - xarr[-1, 1]) < 0.05, (ours[-1], xarr[-1])
    assert abs(ours[:, 2].max() - xarr[:, 2].max()) < 0.02


def test_simulate3d_sidespin_curves_laterally():
    """Pure sidespin (psi=90deg) must deflect the ball laterally vs no spin."""
    hit = (-11.0, 0.0, 2.8)
    straight = hff.simulate3d(hit, 45.0, 0.0, math.radians(-5), 0.0, 0.0, 0.55, 0.6, dt=0.002)
    sidespin = hff.simulate3d(hit, 45.0, 0.0, math.radians(-5), 300.0, math.radians(90),
                              0.55, 0.6, dt=0.002)
    assert abs(sidespin.bounce_y - straight.bounce_y) > 0.3, (straight.bounce_y,
                                                              sidespin.bounce_y)


def test_solve_rally_spin_recovers_apex():
    """Generate a shot on-model with known topspin, then the rally solver (given the true
    spin) must reproduce the apex height (residual ~0)."""
    hit = (-11.0, 0.5, 1.0)
    spin = 1800 * hff.RADSEC_PER_RPM
    L = hff.simulate3d(hit, 26.0, math.atan2(-0.3, 11.0), math.radians(14), spin, 0.0,
                       0.55, 0.6, dt=0.002)
    s = hff.Strike(key=("t",), tour="atp", tournament="australian_open", year=2020,
                   surface="hard", phase="rally", strike_index=3, is_last=True,
                   hit=hit, peak_in=(0, 0, L.apex_z), net=(0.0, L.net_y, L.net_z),
                   bounce=(L.bounce_x, L.bounce_y, 0.03), peak_out=None,
                   spin_rpm=1800.0)
    s.z0 = hit[2]
    s.z_peak_obs = L.apex_z
    s.U_obs = math.hypot(L.bounce_x - hit[0], L.bounce_y - hit[1])
    fit = hff.solve_rally_spin(s, 0.55, 0.6, spin, dt=0.002)
    assert fit is not None
    assert abs(fit["r_apex"]) < 0.03, fit


def test_flight_spin_decay_is_physical():
    """The FIXED flight.py spin decay (C_spin_decay=0.025, exponential in w) must retain the
    large majority of a groundstroke's spin to the bounce (~90%+ over a ~1s flight)."""
    hit = (-11.0, 0.0, 1.0)
    spin = 1900 * hff.RADSEC_PER_RPM
    v0, th = 25.0, math.radians(12)
    pts, vg, wg = hff.integrate3d(hit, (v0 * math.cos(th), 0.0, v0 * math.sin(th)),
                                  (0.0, spin, 0.0), 0.55, 0.6,
                                  cspindecay=0.025, dt=0.001)
    frac = math.sqrt(wg[0] ** 2 + wg[1] ** 2 + wg[2] ** 2) / spin
    assert frac > 0.90, frac        # spin now persists through the flight


def test_load_strikes_smoke(tmp_path):
    traj = tmp_path / "ball_trajectory"
    traj.mkdir()
    pbp = tmp_path / "play_by_play"
    pbp.mkdir()
    f = traj / "atp_australian_open_2020MS999_ball_trajectory.csv"
    f.write_text(
        "point_ID,set_num,game_num,point_num,serve_num,strike_index,position,x,y,z\n"
        "1_1_1_1,1,1,1,1,1,hit,-11.4,-0.3,2.82\n"
        "1_1_1_1,1,1,1,1,1,peak,-11.4,-0.3,2.82\n"
        "1_1_1_1,1,1,1,1,1,net,0.0,0.6,1.27\n"
        "1_1_1_1,1,1,1,1,1,bounce,7.0,1.1,0.033\n"
    )
    p = pbp / "atp_australian_open_2020MS999_pbp.csv"
    p.write_text(
        "point_ID,serve_num,serve_speed_kph,serve_type,spin_rpm\n"
        "1_1_1_1,1,180 KPH,Flat,2500\n"
    )
    strikes = hff.load_strikes(str(f), str(pbp))
    assert len(strikes) == 1
    s = strikes[0]
    assert s.phase == "serve"
    assert abs(s.serve_speed_ms - 50.0) < 0.1        # 180 kph
    assert s.surface == "hard"
    assert abs(s.U_obs - math.hypot(18.4, 1.4)) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
