"""Post-fit checks: one synthetic fitted point per check (pass, fail, abstain)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import post_fit_checks as pfc

NET = pfc.NET_Y_M


def dense(p0, p1, *, start=0.0, end=60.0, bounces=(), nets=(), z_mid=None, v0=None, n=61):
    """Straight-line (plus optional parabola peak ``z_mid``) compact dense flight."""
    t = np.linspace(0.0, 1.0, n)
    p = np.outer(1 - t, p0) + np.outer(t, p1)
    if z_mid is not None:
        p[:, 2] += 4 * (z_mid - (p0[2] + p1[2]) / 2) * t * (1 - t)
    return {
        "start_frame": start,
        "end_frame": end,
        "frames": list(start + (end - start) * t),
        "positions": p.tolist(),
        "v_start": list(v0) if v0 is not None else [0.0, 25.0, 2.0],
        "speed_max_mps": 30.0,
        "fps": 50.0,
        "bounces": [{"frame": f, "x": list(x), "v_in": None, "v_out": None} for f, x in bounces],
        "net_hits": [{"frame": f} for f in nets],
    }


def flight(d, *, role="mid_rally", start_contact=True, end_contact=False, accepted=True, xyz0=None):
    contacts = []
    if start_contact:
        contacts.append(
            {
                "end": "start",
                "kind": "contact",
                "frame": d["start_frame"],
                "xyz": xyz0 or d["positions"][0],
                "root_to_contact_m": 0.8,
                "foot_reach": {"reach_m": 0.9, "sigma_m": 0.1},
            }
        )
    if end_contact:
        contacts.append(
            {
                "end": "end",
                "kind": "contact",
                "frame": d["end_frame"],
                "xyz": d["positions"][-1],
                "root_to_contact_m": None,
                "foot_reach": None,
            }
        )
    return {
        "start_frame": d["start_frame"],
        "end_frame": d["end_frame"],
        "role": role,
        "accepted": accepted,
        "contacts": contacts,
        "dense": d,
    }


def point(*flights, score=None, **extra):
    return {
        "flights": list(flights),
        "observations": [],
        "cameras": {},
        "used_frames": [],
        "score": score or {},
        **extra,
    }


def status(check, pt, i=0, constants=pfc.CONSTANTS):
    return pfc.DECIDE[check](pfc.MEASURE[check](pt, i), constants)


def test_grammar_own_side_landing_fails_and_net_first_passes():
    d = dense([5, 2, 1.0], [5, 8, 0.2], bounces=[(50.0, [5, 8, 0.03])])
    assert status("grammar", point(flight(d))) == ("fail", ["own_side_landing"])
    d = dense([5, 2, 1.0], [5, 8, 0.2], bounces=[(50.0, [5, 8, 0.03])], nets=[30.0])
    assert status("grammar", point(flight(d))) == ("fail", ["net_then_own_side_bounce"])
    d = dense([5, 2, 1.0], [5, 18, 0.03], z_mid=2.0, bounces=[(60.0, [5, 18, 0.03])])
    assert status("grammar", point(flight(d))) == ("pass", [])


def test_grammar_contact_ending_rules():
    d = dense([5, 2, 1.0], [5, 6, 1.0], z_mid=2.0)
    assert "no_crossing_before_contact" in status("grammar", point(flight(d, end_contact=True)))[1]
    d = dense(
        [5, 2, 1.0],
        [5, 22, 1.0],
        z_mid=2.0,
        bounces=[(30.0, [5, 15, 0.03]), (45.0, [5, 19, 0.03])],
    )
    assert "two_bounces_before_contact" in status("grammar", point(flight(d, end_contact=True)))[1]


def test_grammar_serve_rules():
    serve = dense([7.0, -0.5, 2.8], [3.0, 22.0, 1.0], bounces=[(20.0, [3.0, 20.7, 0.03])])
    ret = dense([3.0, 22.0, 1.0], [5.0, 2.0, 1.0], start=60.0, end=120.0)
    # Landing 2.4 m past the service line, then a return: a fault cannot be returned.
    pt = point(flight(serve, role="serve", end_contact=True), flight(ret))
    assert "serve_out_rally_continued" in status("grammar", pt)[1]
    good = dense([7.0, -0.5, 2.8], [3.0, 22.0, 1.0], bounces=[(20.0, [3.0, 15.0, 0.03])])
    pt = point(
        flight(good, role="serve", end_contact=True), flight(ret), score={"deuce_half": True}
    )
    # Near server at x=7 (> centre) is on the deuce half: passes; ad half: fails.
    assert status("grammar", pt) == ("pass", [])
    pt["score"] = {"deuce_half": False}
    assert status("grammar", pt)[1] == ["serve_wrong_half"]


def test_net_clearance():
    low = dense([5, 2, 1.0], [5, 20, 0.6])  # crosses at ~0.8 m, below the 0.914 m tape
    assert status("net_clearance", point(flight(low))) == ("fail", ["below_tape"])
    assert (
        status("net_clearance", point(flight(dense([5, 2, 1.0], [5, 20, 0.6], nets=[33.0]))))[0]
        == "abstain"
    )
    high = dense([5, 2, 1.0], [5, 20, 1.0], z_mid=2.5)
    assert status("net_clearance", point(flight(high))) == ("pass", [])
    short = dense([5, 2, 1.0], [5, 8, 0.2])
    assert status("net_clearance", point(flight(short)))[0] == "abstain"


def _score(winner, server_row=1, server_side="near", attempts=1):
    state = {"confidence": 1.0}
    return {
        "state": state,
        "next_state": state,
        "winner_row": winner,
        "server_row": server_row,
        "server_side": server_side,
        "last_attempt_of_point": True,
        "attempts_in_point": attempts,
    }


def test_scoreboard():
    # Near player (row 1, the server) hits a ball that lands in: row 1 wins.
    d = dense([5, 2, 1.0], [5, 18, 0.03], z_mid=2.0, bounces=[(60.0, [5, 18, 0.03])])
    f = flight(d, role="final")
    assert status("scoreboard", point(f, score=_score(1))) == ("pass", [])
    assert status("scoreboard", point(f, score=_score(2))) == ("fail", ["winner_disagrees"])
    # Out long: row 2 wins.
    d = dense([5, 2, 1.0], [5, 26, 0.03], z_mid=2.0, bounces=[(60.0, [5, 26, 0.03])])
    assert status("scoreboard", point(flight(d, role="final"), score=_score(2)))[0] == "pass"
    # On the line: abstain. Unknown score: abstain.
    d = dense([5, 2, 1.0], [5, 23.7, 0.03], z_mid=2.0, bounces=[(60.0, [5, 23.7, 0.03])])
    assert status("scoreboard", point(flight(d, role="final"), score=_score(1)))[0] == "abstain"
    assert status("scoreboard", point(f))[0] == "abstain"


def test_point_winner_and_deuce_half():
    s = dict(sets_won_1="0", sets_won_2="0", set_number="1", games_1="2", games_2="1", server="1")
    assert (
        pfc.point_winner(
            s | dict(points_1="15", points_2="0"), s | dict(points_1="30", points_2="0")
        )
        == 1
    )
    assert (
        pfc.point_winner(
            s | dict(points_1="40", points_2="AD"), s | dict(points_1="40", points_2="40")
        )
        == 1
    )
    assert (
        pfc.point_winner(
            s | dict(points_1="40", points_2="30"),
            s | dict(games_1="3", points_1="0", points_2="0"),
        )
        == 1
    )
    assert (
        pfc.point_winner(
            s | dict(points_1="15", points_2="0"), s | dict(points_1="40", points_2="0")
        )
        is None
    )
    assert pfc._deuce_half(s | dict(points_1="15", points_2="15")) is True
    assert pfc._deuce_half(s | dict(points_1="40", points_2="AD")) is False


def test_score_context_winner_from_next_point():
    base = dict(
        score_source="score_grammar_v1",
        server="1",
        sets_won_1="0",
        sets_won_2="0",
        set_number="1",
        games_1="0",
        games_2="0",
        score_confidence="1.0",
    )
    rows = [
        ("a1", base | dict(point_index="1", points_1="0", points_2="0", rally_t_start="1")),
        ("a2", base | dict(point_index="1", points_1="0", points_2="0", rally_t_start="5")),
        ("a3", base | dict(point_index="2", points_1="0", points_2="15", rally_t_start="30")),
    ]
    ctx = pfc.score_context(rows)
    assert ctx["a2"]["winner_row"] == 2 and ctx["a2"]["last_attempt_of_point"]
    assert not ctx["a1"]["last_attempt_of_point"]
    assert ctx["a3"]["winner_row"] is None


def test_hitter_reach():
    d = dense([5, -0.5, 2.8], [5, 18, 0.03], bounces=[(50.0, [5, 18, 0.03])])
    assert status("hitter_reach", point(flight(d, role="serve"))) == ("pass", [])
    d = dense([5, -0.5, 3.9], [5, 18, 0.03])
    assert status("hitter_reach", point(flight(d, role="serve")))[1] == ["serve_height"]
    d = dense([5, 2, 1.0], [5, 18, 0.03])
    f = flight(d)
    f["contacts"][0]["foot_reach"] = {"reach_m": 4.0, "sigma_m": 0.2}
    assert status("hitter_reach", point(f))[1] == ["out_of_reach"]
    assert status("hitter_reach", point(flight(d, start_contact=False)))[0] == "abstain"


def test_unused_residual():
    d = dense([5, 2, 1.0], [5, 20, 1.0], z_mid=2.0, end=60.0, n=121)
    d["frames"] = list(np.linspace(0.0, 72.0, 121))  # dense continuation past the end
    P = [[100.0, 0, 0, 0], [0, 100.0, 0, 0], [0, 0, 0, 1.0]]
    cams = {str(k): P for k in range(80)}

    def obs(frames, off):
        out = []
        for k in frames:
            x = pfc.position_at(d, float(k))
            u, v = pfc.project(P, x)
            out.append([k, u + off, v])
        return out

    f = flight(d)
    pt = point(f, cameras=cams, observations=obs([10, 20, 30], 5.0))
    assert status("unused_residual", pt) == ("pass", [])
    pt = point(f, cameras=cams, observations=obs([10, 20, 30], 80.0))
    assert status("unused_residual", pt)[1] == ["unused_observations_missed"]
    pt = point(f, cameras=cams, observations=obs([62, 64, 66], 80.0))
    assert status("unused_residual", pt)[1] == ["after_end_observations_missed"]
    pt = point(f, cameras=cams, observations=obs([10, 20, 30], 80.0), used_frames=[10, 20, 30])
    assert status("unused_residual", pt)[0] == "abstain"


def test_physics():
    d = dense([5, 2, 1.0], [5, 20, 1.0], z_mid=2.0, v0=[0, 66.0, 0])
    assert status("physics", point(flight(d)))[1] == ["too_fast"]
    assert status("physics", point(flight(d, role="serve")))[0] == "pass"
    d = dense([5, 2, 1.0], [5, 20, 1.0], z_mid=14.0)
    assert status("physics", point(flight(d)))[1] == ["apex_too_high"]
    d = dense([5, 2, 1.0], [5, 20, 0.3], bounces=[(59.0, [5, 20, 0.03])])
    assert status("physics", point(flight(d, end_contact=True)))[1] == ["bounce_to_hit_too_short"]


def test_decide_records_every_check_and_demotes_on_active_only():
    d = dense([5, 2, 1.0], [5, 8, 0.2], bounces=[(50.0, [5, 8, 0.03])])
    pt = point(flight(d))
    out = pfc.evaluate(pt, [pfc.span_key(0.0, 60.0)])
    dec = out[pfc.span_key(0.0, 60.0)]
    assert set(dec["checks"]) == set(pfc.CHECKS)
    assert dec["checks"]["grammar"]["status"] == "fail"
    assert dec["demote"] == ("grammar:own_side_landing" in pfc.ACTIVE)
    m = pfc.measure_point(pt, [pfc.span_key(0.0, 60.0)])[pfc.span_key(0.0, 60.0)]
    assert not pfc.decide(m, active=())["demote"]
    assert pfc.decide(m, active=pfc.ALL_REASONS)["reasons"] == ["post_fit_grammar:own_side_landing"]
    assert pfc.evaluate(pt, [pfc.span_key(1.0, 60.0)]) == {}


def test_checks_json_matches_module():
    frozen = Path(__file__).resolve().parents[1] / "experiments" / "post_fit_checks" / "CHECKS.json"
    if not frozen.is_file():
        pytest.skip("CHECKS.json not frozen yet")
    assert json.loads(frozen.read_text()) == json.loads(json.dumps(pfc.to_checks_json()))


def test_switch():
    assert not pfc.enabled({})
    assert pfc.enabled({pfc.SWITCH: "on"})
    with pytest.raises(ValueError):
        pfc.enabled({pfc.SWITCH: "yes"})


def test_recorded_diagnostics_do_not_demote():
    d = dense([5, 2, 1.0], [5, 10, 0.03], bounces=[(50.0, [5, 10.5, 0.03])], nets=[40.0])
    status_, why = status("grammar", point(flight(d)))
    assert "net_then_own_side_bounce" in why
    serve = dense(
        [7.0, -0.5, 2.8], [3.0, 15.0, 0.03], end=30.0, bounces=[(30.0, [3.0, 15.0, 0.03])]
    )
    ret = dense(
        [3.0, 15.0, 0.03], [5, 18, 0.03], start=30.0, z_mid=2.0, bounces=[(60.0, [5, 18, 0.03])]
    )
    pt = point(
        flight(serve, role="serve", end_contact=True),
        flight(ret, role="final"),
        score=_score(2),
    )
    assert pfc.complete_point(pt["flights"])
    assert status("scoreboard", pt, 1)[1] == ["winner_disagrees", "winner_disagrees_complete_point"]
    m = pfc.measure_point(pt, [pfc.span_key(30.0, 60.0)])[pfc.span_key(30.0, 60.0)]
    assert not pfc.decide(m)["demote"]
