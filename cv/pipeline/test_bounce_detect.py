"""Unit tests for bounce_detect.py -- the vertical-deceleration bounce signal, the
ballistic-arc bracket, and the ground-plane projection math. No data files required."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounce_detect as bd  # noqa: E402


class _FakeProj:
    """Ground projector stub: identity-ish court map + always reliable."""
    def to_court(self, u, v, f):
        return (u / 60.0, v / 20.0)

    def is_reliable(self, f):
        return True


class TestArcLen(unittest.TestCase):
    def test_long_ballistic_arc(self):
        # a steadily descending image-y run: one long consistent arc touching the midpoint
        y = np.arange(0, 40, 2.0)                    # +2 px/frame, 20 frames
        self.assertGreaterEqual(bd._arc_len(y, 10), 9)

    def test_static_run_has_no_arc(self):
        y = np.full(20, 100.0)                       # dead ball: no consistent motion
        self.assertLess(bd._arc_len(y, 10), 9)


class TestKinkDetection(unittest.TestCase):
    def _synth_bounce(self):
        # descending then ascending image-y (a near-court bounce = local max / big kink),
        # with a horizontal sweep so it is bracketed by a real arc.
        f = np.arange(60, 100)
        down = np.linspace(100, 300, 20)             # falling (vy>0)
        up = np.linspace(300, 160, 20)               # rebounding (vy<0)
        y = np.concatenate([down, up])
        x = np.linspace(200, 520, 40)                # continuous horizontal sweep
        return f.astype(int), x, y

    def test_detects_bounce_kink(self):
        f, x, y = self._synth_bounce()
        cand = bd.detect_in_run(np.arange(len(f)), f, x, y, _FakeProj(), {"near": {}, "far": {}})
        self.assertTrue(cand, "a clear down->up reversal must yield a kink candidate")
        # the kink must sit near the reversal frame (index ~19-20 -> frame ~79-80)
        self.assertTrue(any(78 <= c["frame"] <= 82 for c in cand))
        self.assertTrue(all(c["kink"] > 0 for c in cand))

    def test_apex_is_not_a_bounce(self):
        # rising then falling image-y (an APEX: vy<0 then vy>0 -> kink<0) yields no candidate
        f = np.arange(60, 100).astype(int)
        y = np.concatenate([np.linspace(300, 120, 20), np.linspace(120, 300, 20)])
        x = np.linspace(200, 520, 40)
        cand = bd.detect_in_run(np.arange(len(f)), f, x, y, _FakeProj(), {"near": {}, "far": {}})
        self.assertFalse(cand, "a smooth apex (up-then-down) must NOT be a bounce candidate")


class TestGroundProjection(unittest.TestCase):
    def test_pinhole_ground_roundtrip(self):
        # a simple pinhole P: court (X,Y,0) -> pixel; invert via inv(P[:,[0,1,3]]) round-trips.
        rng = np.random.default_rng(0)
        clip = "ptTEST"
        P = np.array([[800.0, 0, 480, 100.0],
                      [0, 800.0, 270, 900.0],
                      [0, 0, 1, 12.0]])
        tmp = os.path.join(os.path.dirname(__file__), "_bd_cam_tmp.npz")
        np.savez(tmp, clips=np.array([clip]), frames=np.array([5], dtype=np.int32),
                 P=P[None], k1=np.array([0.0]), dist_center=np.array([[480.0, 270.0]]),
                 reliable=np.array([True]))
        try:
            proj = bd.GroundProjector(tmp, clip)
            for _ in range(5):
                X, Y = rng.uniform(-2, 12), rng.uniform(0, 24)
                uvw = P @ np.array([X, Y, 0, 1.0])
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
                cx, cy = proj.to_court(u, v, 5)
                self.assertAlmostEqual(cx, X, places=4)
                self.assertAlmostEqual(cy, Y, places=4)
        finally:
            os.remove(tmp)

    def test_projection_scales_to_artifact_coordinates(self):
        clip = "ptTEST"
        projection = np.array(
            [
                [1600.0, 0.0, 960.0, 200.0],
                [0.0, 1600.0, 540.0, 1800.0],
                [0.0, 0.0, 1.0, 12.0],
            ]
        )
        tmp = os.path.join(os.path.dirname(__file__), "_bd_scaled_cam_tmp.npz")
        np.savez(
            tmp,
            clips=np.array([clip]),
            frames=np.array([5], dtype=np.int32),
            P=projection[None],
            reliable=np.array([True]),
        )
        try:
            proj = bd.GroundProjector(
                tmp,
                clip,
                image_size=bd.res.FrameSize(1920, 1080),
                artifact_size=bd.res.FrameSize(960, 540),
            )
            world = np.array([4.0, 17.0, 0.0, 1.0])
            native = projection @ world
            artifact = 0.5 * native[:2] / native[2]
            court_x, court_y = proj.to_court(*artifact, 5)
            self.assertAlmostEqual(court_x, world[0], places=4)
            self.assertAlmostEqual(court_y, world[1], places=4)
        finally:
            os.remove(tmp)


def _make_proj(P, frame=5):
    """A GroundProjector over a single-frame pinhole P (k1=0 -> undistort is identity)."""
    clip = "ptTEST"
    tmp = os.path.join(os.path.dirname(__file__), "_bd_net_tmp.npz")
    np.savez(tmp, clips=np.array([clip]), frames=np.array([frame], dtype=np.int32),
             P=np.asarray(P)[None], k1=np.array([0.0]),
             dist_center=np.array([[480.0, 270.0]]), reliable=np.array([True]))
    proj = bd.GroundProjector(tmp, clip)
    os.remove(tmp)
    return proj, clip, frame


# a pinhole looking down the court: [X,Y,Z,1] -> pixel; Z (height) shifts the image point up.
_P_NET = np.array([[800.0, 0.0, 60.0, 100.0],
                   [0.0, 800.0, -300.0, 900.0],
                   [0.0, 0.0, 1.0, 40.0]])


class TestNetTapeGeometry(unittest.TestCase):
    def test_tape_height_ramp(self):
        self.assertAlmostEqual(bd.net_tape_height(bd.NET_X_CENTER), bd.NET_H_CENTER, places=6)
        self.assertAlmostEqual(bd.net_tape_height(0.0),
                               bd.NET_H_CENTER + (bd.NET_H_POST - bd.NET_H_CENTER)
                               * (bd.NET_X_CENTER / bd.NET_HALF_WIDTH), places=6)
        # clamps at the posts (never exceeds NET_H_POST)
        self.assertAlmostEqual(bd.net_tape_height(-50.0), bd.NET_H_POST, places=6)

    def test_ball_on_tape_is_near_the_line(self):
        proj, _clip, f = _make_proj(_P_NET)
        # a world point ON the tape projects onto the tape line -> distance ~0
        w = _P_NET @ np.array([bd.NET_X_CENTER, bd.NET_Y, bd.net_tape_height(bd.NET_X_CENTER), 1.0])
        u, v = w[0] / w[2], w[1] / w[2]
        d, _sv = bd.tape_pixel_dist(proj, u, v, f)
        self.assertLess(d, 2.5)                             # ~0 up to tape-line sampling step
        # a ground point well inside the far court projects tens of px from the tape line
        wg = _P_NET @ np.array([bd.NET_X_CENTER, 20.0, 0.0, 1.0])
        ug, vg = wg[0] / wg[2], wg[1] / wg[2]
        dg, _ = bd.tape_pixel_dist(proj, ug, vg, f)
        self.assertGreater(dg, 20.0)


class TestNetVsBounce(unittest.TestCase):
    def _cand(self, proj, f, world_pt, vy_before, vx_before, vx_after, vy_after):
        w = np.asarray(proj.P[f]) @ np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
        u, v = w[0] / w[2], w[1] / w[2]
        cx, cy = proj.to_court(u, v, f)
        return {"verdict": "bounce", "img_x": u, "img_y": v, "frame": float(f),
                "court_x": cx, "court_y": cy, "confidence": 0.6, "flags": [],
                "vy_before": vy_before, "vx_before": vx_before,
                "vx_after": vx_after, "vy_after": vy_after,
                "sp_before": float(np.hypot(vx_before, vy_before)),
                "sp_after": float(np.hypot(vx_after, vy_after))}

    def test_same_side_impossible_bounce_flips_to_net(self):
        # SAME-SIDE isolated from the death test: the candidate sits on the tape and rebounds with
        # real horizontal speed (NOT dead, horizontal REVERSES so it is not a continuing bounce
        # either). The only thing that flips it is that the serve came from the same side its ground
        # projection lands on -- an impossible pre-net same-side bounce -> NET.
        proj, _clip, f = _make_proj(_P_NET)
        c = self._cand(proj, f, (bd.NET_X_CENTER, bd.NET_Y, bd.net_tape_height(bd.NET_X_CENTER)),
                       vy_before=1.0, vx_before=11.0, vx_after=-6.0, vy_after=-4.0)
        server_side = "far" if c["court_y"] > bd.NET_Y else "near"   # the side its ground pos lands
        bd.disambiguate_net_bounce(c, proj, serves=[{"frame": f - 12, "side": server_side}])
        self.assertEqual(c["verdict"], "net")
        self.assertTrue(any("same_side" in fl for fl in c["flags"]))

    def test_same_side_opposite_serve_side_does_not_flip_alone(self):
        # same candidate but the serve came from the OTHER side (the ball has crossed) and it is not
        # dead: no same-side impossibility, so it is NOT flipped by RULE 1b alone.
        proj, _clip, f = _make_proj(_P_NET)
        c = self._cand(proj, f, (bd.NET_X_CENTER, bd.NET_Y, bd.net_tape_height(bd.NET_X_CENTER)),
                       vy_before=1.0, vx_before=11.0, vx_after=-6.0, vy_after=-4.0)
        other = "near" if c["court_y"] > bd.NET_Y else "far"
        bd.disambiguate_net_bounce(c, proj, serves=[{"frame": f - 12, "side": other}])
        self.assertEqual(c["verdict"], "bounce")

    def test_far_court_ground_bounce_stays_bounce(self):
        # a genuine ground bounce well inside the court projects far from the tape -> the tape band
        # gate never opens; it stays a bounce regardless of the other signals.
        proj, _clip, f = _make_proj(_P_NET)
        c = self._cand(proj, f, (bd.NET_X_CENTER, 20.0, 0.0),
                       vy_before=3.0, vx_before=2.0, vx_after=1.0, vy_after=-6.0)
        bd.disambiguate_net_bounce(c, proj, serves=None)
        self.assertEqual(c["verdict"], "bounce")
        self.assertGreater(c["tape_px"], bd.NET_TAPE_BAND_PX)

    def test_post_kink_dead_flips_to_net(self):
        # near the tape, the ball sheds nearly all speed and the horizontal motion collapses
        # (mesh death / tape drop) -> NET, even without serve context.
        proj, _clip, f = _make_proj(_P_NET)
        c = self._cand(proj, f, (bd.NET_X_CENTER, bd.NET_Y, bd.net_tape_height(bd.NET_X_CENTER)),
                       vy_before=10.0, vx_before=9.0, vx_after=0.8, vy_after=-1.0)
        bd.disambiguate_net_bounce(c, proj, serves=None)
        self.assertEqual(c["verdict"], "net")

    def test_continuing_bounce_near_tape_stays_bounce(self):
        # a ball that happens to be near the tape line but REBOUNDS keeping real horizontal speed
        # in the SAME direction (a true bounce, not a dead net clip) is NOT flipped. Origin near
        # (vy_before<0) so it is not same-side either.
        proj, _clip, f = _make_proj(_P_NET)
        c = self._cand(proj, f, (bd.NET_X_CENTER, bd.NET_Y, bd.net_tape_height(bd.NET_X_CENTER)),
                       vy_before=-8.0, vx_before=9.0, vx_after=8.0, vy_after=-7.0)
        bd.disambiguate_net_bounce(c, proj, serves=None)
        self.assertEqual(c["verdict"], "bounce")


if __name__ == "__main__":
    unittest.main()
