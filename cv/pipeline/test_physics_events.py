"""Self-tests for the physics fit-and-segment detector (Approach B).

Synthetic end-to-end check: simulate serve -> bounce -> return hit with the validated
flight/impact models, project through a synthetic broadcast camera, add pixel noise,
and require the pipeline to recover the knots at the right frames with the right classes
and no spurious mid-flight events.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "physics"))

import impact  # noqa: E402
from camera_cal import project  # noqa: E402
from physics_events import (  # noqa: E402
    FPS,
    classify_knot,
    fit_arc,
    integrate_states,
    process_clip,
)


def synthetic_P():
    """Broadcast-ish camera: behind the near baseline, elevated, looking down-court."""
    C = np.array([5.485, -25.0, 12.0])
    look = np.array([5.485, 14.0, 0.0]) - C
    zc = look / np.linalg.norm(look)
    xc = np.cross(zc, np.array([0.0, 0.0, 1.0]))
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    f = 1400.0
    K = np.array([[f, 0, 480.0], [0, f, 270.0], [0, 0, 1.0]])
    return K @ np.hstack([R, (-R @ C).reshape(3, 1)])


def simulate_rally(fps: float = FPS):
    """serve (near) -> bounce (far service box) -> return hit (far player).

    Returns (frames, world_positions, truth) with truth frame indices (1-based).
    """
    dt = 1.0 / fps
    xs = []
    x = np.array([6.2, 0.3, 2.9])
    v = np.array([-1.0, 33.0, -4.0])
    w = np.zeros(3)
    bounce_i = hit_i = None
    for i in range(200):
        xs.append(x.copy())
        xn, vn = integrate_states(x, v, w, 2, dt)
        x, v = xn[1], vn[1]
        if x[2] < impact.R_BALL and v[2] < 0 and bounce_i is None:
            bounce_i = len(xs)
            vh = math.hypot(v[0], v[1])
            u = np.array([v[0], v[1]]) / vh
            b = impact.court_bounce(vh, -v[2], 150.0, "clay")
            v = np.array([b.vx2 * u[0], b.vx2 * u[1], b.vy2])
            x[2] = impact.R_BALL
        if bounce_i is not None and hit_i is None and len(xs) - bounce_i >= 22:
            hit_i = len(xs)
            v = np.array([2.0, -26.0, 6.5])  # return toward the near side
        if bounce_i is not None and hit_i is not None and len(xs) - hit_i >= 55:
            break
    frames = np.arange(1, len(xs) + 1)
    return frames, np.array(xs), {"serve": 1, "bounce": bounce_i, "hit": hit_i}


def test_fit_arc_recovers_flight():
    P = synthetic_P()
    dt = 1.0 / FPS
    xs, vs = integrate_states([6.0, 1.0, 2.5], [-1.0, 30.0, 2.0], [0, 0, 0], 40, dt)
    uv = project(P, xs) + np.random.default_rng(0).normal(0, 0.4, (40, 2))
    arc = fit_arc(P, uv, np.arange(1, 41))
    assert arc is not None and arc.rms < 1.5, (arc and arc.rms)
    x_end, v_end = arc.state_at(40)
    assert np.linalg.norm(x_end - xs[-1]) < 1.0, x_end
    assert np.linalg.norm(v_end - vs[-1]) < 3.0, (v_end, vs[-1])


def test_classify_knot_bounce_and_hit():
    x_b = np.array([4.0, 17.0, 0.05])
    v_in = np.array([-0.8, 24.0, -6.0])
    b = impact.court_bounce(float(np.hypot(v_in[0], v_in[1])), 6.0, 200.0, "clay")
    u = v_in[:2] / np.linalg.norm(v_in[:2])
    v_out = np.array([b.vx2 * u[0], b.vx2 * u[1], b.vy2])
    klass, feats = classify_knot(x_b, v_in, v_out, dist_px=400.0)
    assert klass == "bounce", (klass, feats)

    x_h = np.array([5.0, 21.5, 1.1])
    klass, feats = classify_knot(
        x_h, np.array([-0.5, 18.0, -2.0]), np.array([1.5, -24.0, 5.0]), dist_px=25.0
    )
    assert klass == "hit", (klass, feats)

    # tiny impulse far from any player is noise
    klass, _ = classify_knot(
        np.array([5.0, 12.0, 3.0]), np.array([0.0, 20.0, -1.0]),
        np.array([0.3, 19.2, -1.4]), dist_px=math.inf,
    )
    assert klass == "noise", klass


def test_pipeline_end_to_end():
    P = synthetic_P()
    frames, xs, truth = simulate_rally()
    rng = np.random.default_rng(7)
    uv = project(P, xs) + rng.normal(0, 0.5, (len(xs), 2))
    score = np.full(len(frames), 0.9)
    # player boxes: feet on the ground under the serve / hit positions (as in real
    # detections), spanning up to roughly head height
    def player_box(x, y):
        foot = project(P, [[x, y, 0.0]])[0]
        head = project(P, [[x, y, 1.9]])[0]
        return (foot[0] - 30, head[1], foot[0] + 30, foot[1])

    box_serve = player_box(xs[0][0], xs[0][1])
    box_hit = player_box(xs[truth["hit"] - 1][0], xs[truth["hit"] - 1][1])
    boxes = {int(f): [box_serve, box_hit] for f in frames}
    events, knots, arcs = process_clip("pt9999", P, (frames, uv, score), boxes)
    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]
    assert any(abs(e["frame"] - truth["hit"]) <= 2 for e in hits), (
        [e["frame"] for e in hits], truth)
    assert any(abs(e["frame"] - truth["bounce"]) <= 2 for e in bounces), (
        [e["frame"] for e in bounces], truth)
    assert any(abs(e["frame"] - truth["serve"]) <= 3 for e in hits), (
        [e["frame"] for e in hits], truth)
    # no spurious hit events away from the true contacts
    for e in hits:
        assert min(abs(e["frame"] - truth["serve"]), abs(e["frame"] - truth["hit"])) <= 3, e


if __name__ == "__main__":
    test_fit_arc_recovers_flight()
    test_classify_knot_bounce_and_hit()
    test_pipeline_end_to_end()
    print("physics_events self-tests OK")
