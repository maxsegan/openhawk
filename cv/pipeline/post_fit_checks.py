"""Post-fit checks v1: re-validate what the fit added or changed before a flight counts as validated.

Why (owner, 2026-10-10): the fit can add a bounce the detector never recorded, move events,
choose a terminal ending and draw a continuation. The fit-only bounce audit
(``cv/experiments/fit_only_bounce_audit/REPORT.md``) found such flights wrong 60/100, many of
them balls into the net. Rules v1 (``fit_only_bounce_rules``) already demote unwitnessed
fit-only bounces; these checks look at the FITTED point as a whole, next to them, and never
repair: a failing flight leaves the validated tier for the best-effort tier with its reasons.

Each check is a pure function over one fitted point (``point_inputs``: the final S6 result's
flights with their dense states and contacts, the attempt's ball observations and cameras, the
player-box reach of each contact, and the score context). It is split in two:

* ``measure_<check>(point, i)``: numbers about flight ``i`` (no thresholds beyond the frozen
  definitions of what is measured);
* ``decide_<check>(measures, constants)``: ``pass`` / ``fail`` (with reasons) / ``abstain``.

Checks (the brief's a-f; constants in ``CONSTANTS``, frozen as ``CHECKS.json``):

* ``grammar``: on the fitted sequence. A struck flight's first fitted bounce is on the
  opponent's side unless a fitted net hit comes first (``own_side_landing``); a flight ending
  at a racket contact crossed the net (``no_crossing_before_contact``) and bounced at most once
  (``two_bounces_before_contact``); a serve whose fitted landing is outside the diagonal service
  box is a fault, so no struck flight may follow it in the attempt (``serve_out_rally_continued``);
  the server stands on the deuce/ad half the score implies (``serve_wrong_half``, score known).
* ``net_clearance``: every crossing of the net plane inside the flight's span is above the tape
  (``camera_cal.net_height_at_x``) by at least ``-net_margin_m`` (ball centre), unless a fitted net
  hit is at the crossing (``below_tape``).
* ``scoreboard``: on the attempt's last flight when the fit calls it the point's final flight:
  the winner implied by the fitted ending (in / out / net / own side, who struck last) matches
  the scoreboard's next point (``winner_disagrees``). Abstains when the score, the server's end
  or the ending is unknown, or the line call is within ``line_margin_m``.
* ``hitter_reach``: a serve contact between ``serve_z_m``; any launching contact within
  ``reach_max_m`` (horizontal, after the box sigma) of a player's box foot point
  (``serve_height``, ``out_of_reach``). 3D pose is computed after this step, so it is not used.
* ``unused_residual``: the fitted path projected on frames where the detector's confident
  (``guide_observation``) ball was not used by the fit, inside the span
  (``unused_observations_missed``) and on the frames just after the flight's end where the
  fitted continuation exists (``after_end_observations_missed``).
* ``physics``: launch speed (serve vs rally), apex height, and the gap between the last bounce
  and the racket contact that ends the flight (``too_fast``, ``apex_too_high``,
  ``bounce_to_hit_too_short``). The speed kept at a bounce is set by the fit's bounce law, so it
  is recorded only.

Switch ``TENNIS_POST_FIT_CHECKS`` (``off`` default = today's output, the rollback; ``on``).
Every check reads only automatic outputs; no evaluation label.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np

from cv.pipeline.camera_cal import net_height_at_x

CHECKS_VERSION = "post_fit_checks_v1"
SWITCH = "TENNIS_POST_FIT_CHECKS"
CHECKS = ("grammar", "net_clearance", "scoreboard", "hitter_reach", "unused_residual", "physics")

NET_Y_M = 11.885
COURT_LENGTH_M = 23.77
CENTRE_X_M = 5.485
SINGLES_HALF_WIDTH_M = 4.115
SERVICE_DEPTH_M = 6.40
BALL_RADIUS_M = 0.0325
#: Dense fitted states are sampled at 240 Hz; inputs keep every ``DENSE_STEP``-th (120 Hz).
DENSE_HZ = 240.0
DENSE_STEP = 2
#: A fitted serve starts at least this high (``point_end_tag.SERVE_MIN_HEIGHT_M``).
SERVE_MIN_HEIGHT_M = 2.0

CONSTANTS: dict[str, float] = {
    # grammar
    "own_side_margin_m": 0.5,
    "crossing_margin_m": 0.5,
    "serve_box_margin_m": 0.6,
    "centre_margin_m": 0.3,
    # net clearance (ball centre minus tape height)
    "net_margin_m": 0.10,
    "net_hit_frames": 2.0,
    # scoreboard
    "line_margin_m": 0.3,
    "score_min_confidence": 0.5,
    # hitter reach
    "serve_z_min_m": 2.2,
    "serve_z_max_m": 3.6,
    "reach_max_m": 3.0,
    # unused residual
    "residual_px": 40.0,
    "residual_min_frames": 3,
    "after_end_frames": 12,
    # physics
    "serve_speed_max_mps": 75.0,
    "rally_speed_max_mps": 60.0,
    "apex_max_m": 12.0,
    "bounce_to_hit_min_s": 0.05,
}


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(SWITCH, "off")
    if value not in ("on", "off"):
        raise ValueError(f"{SWITCH} must be on or off, not {value!r}")
    return value == "on"


def to_checks_json(active: tuple[str, ...] | None = None) -> dict:
    """The frozen definition as data (``CHECKS.json``)."""
    return {
        "checks_version": CHECKS_VERSION,
        "switch": SWITCH,
        "checks": list(CHECKS),
        "active": list(ACTIVE if active is None else active),
        "constants": dict(CONSTANTS),
        "definition": __doc__,
    }


# ---------------------------------------------------------------- inputs


def _dense_by_key(result: dict) -> dict[tuple, dict]:
    """``product_runner.dense_flights``: dense states keyed by (component, flight index)."""
    found = {}
    for component in result.get("component_results") or []:
        measurement = (component.get("result") or {}).get("measurement") or {}
        for index, flight in enumerate(measurement.get("dense_flights") or []):
            found[(component["component_index"], index)] = flight
    for index, flight in enumerate((result.get("measurement") or {}).get("dense_flights") or []):
        found[(None, index)] = flight
    return found


def _used_frames(result: dict) -> set[int]:
    """Native frames whose ball observation the fit used (training or withheld)."""
    used: set[int] = set()
    holders = [(c.get("result") or {}) for c in result.get("component_results") or []]
    for res in holders + [result]:
        for row in (res.get("measurement") or {}).get("native_projection") or []:
            used.add(int(row["frame"]))
    return used


def _r(values, digits: int = 4) -> list:
    return [round(float(v), digits) for v in values]


def _compact_dense(d: dict) -> dict:
    positions = d["positions"]
    n = len(positions)
    start, end = float(d["start_frame"]), float(d["end_frame"])
    keep = list(range(0, n, DENSE_STEP))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    frames = [start + (end - start) * i / (n - 1) if n > 1 else start for i in keep]
    speeds = [float(np.linalg.norm(v)) for v in d.get("velocities") or []]
    return {
        "start_frame": start,
        "end_frame": end,
        "frames": _r(frames, 3),
        "positions": [_r(positions[i]) for i in keep],
        "v_start": _r(d["velocities"][0], 3) if d.get("velocities") else None,
        "speed_max_mps": round(max(speeds), 3) if speeds else None,
        "fps": round((end - start) * DENSE_HZ / (n - 1), 4) if n > 1 and end > start else None,
        "bounces": [
            {
                "frame": round(float(b["frame"]), 4),
                "x": _r(b["x"]),
                "v_in": _r(b["v_in"], 3) if b.get("v_in") else None,
                "v_out": _r(b["v_out"], 3) if b.get("v_out") else None,
            }
            for b in d.get("bounces") or []
            if "x" in b
        ],
        "net_hits": [
            {"frame": round(float(h["frame"]), 4), **({"x": _r(h["x"])} if "x" in h else {})}
            for h in d.get("net_hits") or []
        ],
    }


def point_inputs(
    result: dict,
    *,
    cameras: dict | None = None,
    observations: dict | None = None,
    reach=None,
) -> dict:
    """The compact fitted point the checks read (JSON-able).

    ``cameras``: the attempt's ``cameras.json``; ``observations``: its ``observations.json``
    (the fit's ``labels`` input); ``reach``: optional ``(frame, xyz) -> foot_reach row``
    (``flight_validation.foot_reach`` over the attempt's player boxes).
    """
    dense = _dense_by_key(result)
    flights = []
    for f in sorted(
        (result.get("verdict") or {}).get("flights") or [], key=lambda f: float(f["start_frame"])
    ):
        d = dense.get((f.get("component_index"), f.get("flight_index")))
        if d is not None and abs(float(d["start_frame"]) - float(f["start_frame"])) > 1:
            d = None
        contacts = []
        for c in f.get("contacts") or []:
            xyz = c.get("contact_xyz_m")
            frame = float(f["start_frame"] if c.get("end") == "start" else f["end_frame"])
            contacts.append(
                {
                    "end": c.get("end"),
                    "kind": c.get("kind"),
                    "frame": frame,
                    "xyz": _r(xyz) if xyz is not None else None,
                    "root_to_contact_m": c.get("root_to_contact_m"),
                    "foot_reach": (
                        reach(frame, xyz)
                        if reach is not None and xyz is not None and c.get("kind") == "contact"
                        else None
                    ),
                }
            )
        flights.append(
            {
                "start_frame": float(f["start_frame"]),
                "end_frame": float(f["end_frame"]),
                "role": f.get("role"),
                "accepted": bool(f.get("accepted")),
                "contacts": contacts,
                "dense": _compact_dense(d) if d is not None and len(d["positions"]) > 1 else None,
            }
        )
    used = _used_frames(result)
    obs = []
    for record in ((observations or {}).get("ball") or {}).get("records") or []:
        for row in record.get("frames") or []:
            if row.get("status") == "visible" and row.get("support_class") == "guide_observation":
                frame = int(row["frame"])
                obs.append([frame, round(float(row["x1080"]), 2), round(float(row["y1080"]), 2)])
    lo = min((fl["start_frame"] for fl in flights), default=0.0)
    hi = (
        max(
            (fl["dense"]["end_frame"] for fl in flights if fl["dense"]),
            default=max((fl["end_frame"] for fl in flights), default=0.0),
        )
        + CONSTANTS["after_end_frames"]
        + 1
    )
    obs = [o for o in obs if lo - 1 <= o[0] <= hi]
    wanted = {o[0] for o in obs}
    cams = {}
    for cam in (cameras or {}).get("cameras") or []:
        k = int(cam["frame"])
        if k in wanted and cam.get("supported") and cam.get("P") is not None:
            cams[str(k)] = [_r(row, 6) for row in cam["P"]]
    return {
        "schema": "post_fit_point_inputs_v1",
        "flights": flights,
        "used_frames": sorted(f for f in used if lo - 1 <= f <= hi),
        "observations": obs,
        "cameras": cams,
    }


# ---------------------------------------------------------------- score context

_POINT_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}
SCORE_FIELDS = (
    "server",
    "sets_won_1",
    "sets_won_2",
    "set_number",
    "games_1",
    "games_2",
    "points_1",
    "points_2",
)


def _state(ledger: Mapping) -> dict | None:
    if not ledger or ledger.get("score_source") != "score_grammar_v1" or not ledger.get("server"):
        return None
    if any(str(ledger.get(k, "")) == "" for k in SCORE_FIELDS):
        return None
    try:
        confidence = float(ledger.get("score_confidence") or 0.0)
    except ValueError:
        confidence = 0.0
    return {**{k: str(ledger[k]) for k in SCORE_FIELDS}, "confidence": confidence}


def point_winner(before: Mapping, after: Mapping) -> int | None:
    """Score row (1 or 2) that won the point taking ``before`` to ``after``, else None."""
    sets = [int(after[f"sets_won_{k}"]) - int(before[f"sets_won_{k}"]) for k in (1, 2)]
    if sets != [0, 0]:
        return 1 if sets == [1, 0] else 2 if sets == [0, 1] else None
    if before["set_number"] != after["set_number"]:
        return None
    games = [int(after[f"games_{k}"]) - int(before[f"games_{k}"]) for k in (1, 2)]
    if games != [0, 0]:
        return 1 if games == [1, 0] else 2 if games == [0, 1] else None
    p0 = (before["points_1"], before["points_2"])
    p1 = (after["points_1"], after["points_2"])
    if all(v.isdigit() and v not in ("15", "30", "40") for v in p0 + p1):  # tiebreak
        d = [int(p1[k]) - int(p0[k]) for k in (0, 1)]
        return 1 if d == [1, 0] else 2 if d == [0, 1] else None
    if not all(v in _POINT_ORDER for v in p0 + p1):
        return None
    a = [_POINT_ORDER[v] for v in p0]
    b = [_POINT_ORDER[v] for v in p1]
    if a == [3, 4] and b == [3, 3]:
        return 1
    if a == [4, 3] and b == [3, 3]:
        return 2
    d = [b[k] - a[k] for k in (0, 1)]
    return 1 if d == [1, 0] else 2 if d == [0, 1] else None


def _deuce_half(state: Mapping) -> bool | None:
    """True when the score puts the server on the deuce half (even points played)."""
    p = (state["points_1"], state["points_2"])
    if all(v.isdigit() and v not in ("15", "30", "40") for v in p):
        return (int(p[0]) + int(p[1])) % 2 == 0
    if not all(v in _POINT_ORDER for v in p):
        return None
    a = [_POINT_ORDER[v] for v in p]
    return sum(a) % 2 == 0


def game_key(state: Mapping) -> tuple:
    return tuple(state[k] for k in ("set_number", "games_1", "games_2", "server"))


def score_context(attempts: list[tuple[str, Mapping]]) -> dict[str, dict]:
    """Per attempt key: the ledger's score state, the point's winner from the next point's state,
    whether the attempt is its point's last, and the expected deuce/ad half.

    ``attempts``: (key, ledger row) for every ledger attempt of the match, any order."""
    rows = sorted(
        attempts,
        key=lambda kv: float(kv[1].get("rally_t_start") or 0.0) if kv[1] else 0.0,
    )
    by_point: dict[int, list[tuple[str, Mapping]]] = defaultdict(list)
    for key, ledger in rows:
        try:
            by_point[int(ledger.get("point_index"))].append((key, ledger))
        except (TypeError, ValueError):
            continue
    point_state = {}
    for index, members in by_point.items():
        states = [s for s in (_state(ledger) for _, ledger in members) if s is not None]
        point_state[index] = states[0] if states else None
    out = {}
    for index, members in by_point.items():
        state = point_state[index]
        after = point_state.get(index + 1)
        winner = None
        if state is not None and after is not None:
            winner = point_winner(state, after)
        for position, (key, _) in enumerate(members):
            out[key] = {
                "state": state,
                "next_state": after,
                "winner_row": winner,
                "server_row": int(state["server"]) if state else None,
                "deuce_half": _deuce_half(state) if state else None,
                "last_attempt_of_point": position == len(members) - 1,
                "attempts_in_point": len(members),
                "game": list(game_key(state)) if state else None,
            }
    return out


def serve_side(result: dict) -> str | None:
    """Near/far end of the attempt's fitted serve (first accepted serve flight, high start)."""
    dense = _dense_by_key(result)
    for f in sorted(
        (result.get("verdict") or {}).get("flights") or [], key=lambda f: float(f["start_frame"])
    ):
        if not f.get("accepted") or f.get("role") != "serve":
            continue
        d = dense.get((f.get("component_index"), f.get("flight_index")))
        if d is None or float(d["start_xyz"][2]) < SERVE_MIN_HEIGHT_M:
            continue
        return "near" if float(d["start_xyz"][1]) < NET_Y_M else "far"
    return None


def server_sides(sides: list[tuple[list | None, str | None]]) -> dict[tuple, str]:
    """Server's end per game: the majority of the fitted serve ends in that game (unanimous
    games only; a changeover inside a game never happens)."""
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for game, side in sides:
        if game is not None and side is not None:
            votes[tuple(game)][side] += 1
    return {g: next(iter(c)) for g, c in votes.items() if len(c) == 1}


# ---------------------------------------------------------------- geometry helpers


def _side(y: float) -> int:
    return -1 if y < NET_Y_M else 1


def _launch(f: dict) -> tuple[list | None, bool]:
    """Launch point of a flight and whether a racket struck it (contact or serve)."""
    contact = next((c for c in f["contacts"] if c["end"] == "start"), None)
    struck = (contact is not None and contact["kind"] == "contact") or f.get("role") == "serve"
    xyz = (contact or {}).get("xyz") or ((f.get("dense") or {}).get("positions") or [None])[0]
    return xyz, struck


def _ends_at_contact(f: dict) -> bool:
    return any(c["end"] == "end" and c["kind"] == "contact" for c in f["contacts"])


def _inside(f: dict, frame: float) -> bool:
    return float(frame) < f["end_frame"] + 1e-3


def position_at(dense: dict, frame: float) -> np.ndarray | None:
    frames = dense["frames"]
    if not frames or frame < frames[0] - 1e-6 or frame > frames[-1] + 1e-6:
        return None
    p = np.asarray(dense["positions"], float)
    return np.array([np.interp(frame, frames, p[:, k]) for k in range(3)])


def project(P, xyz) -> np.ndarray:
    h = np.asarray(P, float) @ np.append(np.asarray(xyz, float), 1.0)
    return h[:2] / h[2]


def _box_out(xy, launch) -> tuple[float, bool]:
    """Distance outside the diagonal service box (negative inside) and whether the landing is
    on the receiver's half."""
    x, y = float(xy[0]), float(xy[1])
    receiver = _side(y) != _side(float(launch[1]))
    long = abs(y - NET_Y_M) - SERVICE_DEPTH_M
    # Cross court: the box is on the other side of the centre line from the server.
    across = (x - CENTRE_X_M) * (float(launch[0]) - CENTRE_X_M) <= 0
    wide = abs(x - CENTRE_X_M) - SINGLES_HALF_WIDTH_M if across else abs(x - CENTRE_X_M)
    return max(long, wide), receiver


def _court_out(xy) -> float:
    """Distance outside the singles court (negative inside)."""
    x, y = float(xy[0]), float(xy[1])
    return max(abs(x - CENTRE_X_M) - SINGLES_HALF_WIDTH_M, -y, y - COURT_LENGTH_M)


# ---------------------------------------------------------------- a. grammar


def measure_grammar(point: dict, i: int) -> dict:
    flights = point["flights"]
    f = flights[i]
    d = f["dense"]
    if d is None:
        return {"measured": False}
    xyz, struck = _launch(f)
    side0 = _side(float(xyz[1]))
    bounces = [b for b in d["bounces"] if _inside(f, b["frame"])]
    nets = [h["frame"] for h in d["net_hits"] if _inside(f, h["frame"])]
    m: dict = {
        "measured": True,
        "struck": struck,
        "role": f["role"],
        "inside_bounces": len(bounces),
    }
    if bounces:
        first = bounces[0]
        m["first_bounce_own_side_m"] = round((float(first["x"][1]) - NET_Y_M) * side0, 3)
        m["net_before_first_bounce"] = any(h < first["frame"] for h in nets)
    m["ends_at_contact"] = _ends_at_contact(f)
    end = next((c["xyz"] for c in f["contacts"] if c["end"] == "end" and c["xyz"]), None)
    if end is None and d["positions"]:
        end = d["positions"][-1]
    m["end_own_side_m"] = round((float(end[1]) - NET_Y_M) * side0, 3)
    m["net_hits_inside"] = len(nets)
    if nets:
        after = [b for b in bounces if b["frame"] > nets[0]]
        m["net_then_own_side_bounce"] = any(_side(float(b["x"][1])) == side0 for b in after)
    if f["role"] == "serve" and float(xyz[2]) >= SERVE_MIN_HEIGHT_M:
        m["serve"] = True
        m["serve_launch_x_m"] = round(float(xyz[0]), 3)
        m["serve_launch_side"] = "near" if side0 < 0 else "far"
        if bounces:
            out, receiver = _box_out(bounces[0]["x"], xyz)
            m["serve_box_out_m"] = round(out, 3)
            m["serve_landing_receiver_half"] = receiver
        later = [
            g
            for g in flights[i + 1 :]
            if g["accepted"] and abs(g["start_frame"] - f["end_frame"]) <= 1.0
        ]
        m["struck_flight_follows"] = any(_launch(g)[1] for g in later)
        score = point.get("score") or {}
        if score.get("deuce_half") is not None:
            # Near server faces +y: the deuce half is +x; far server: -x.
            deuce_x = 1 if side0 < 0 else -1
            offset = (float(xyz[0]) - CENTRE_X_M) * deuce_x
            m["serve_deuce_offset_m"] = round(offset, 3)
            m["score_deuce_half"] = bool(score["deuce_half"])
    return m


def decide_grammar(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured"):
        return "abstain", []
    reasons = []
    own = m.get("first_bounce_own_side_m")
    if m["struck"] and own is not None and own > c["own_side_margin_m"]:
        if not m.get("net_before_first_bounce"):
            reasons.append("own_side_landing")
    if m["struck"] and m["ends_at_contact"] and m["end_own_side_m"] > c["crossing_margin_m"]:
        reasons.append("no_crossing_before_contact")
    if m["ends_at_contact"] and m["inside_bounces"] >= 2:
        reasons.append("two_bounces_before_contact")
    if m.get("net_then_own_side_bounce"):
        # Coordinator 2026-10-10: recorded diagnostic (8 wrong vs 18 right flights rules v1 keep).
        reasons.append("net_then_own_side_bounce")
    if (
        m.get("serve")
        and m.get("serve_box_out_m") is not None
        and m["serve_box_out_m"] > c["serve_box_margin_m"]
        and m.get("struck_flight_follows")
    ):
        reasons.append("serve_out_rally_continued")
    offset = m.get("serve_deuce_offset_m")
    if offset is not None:
        wrong = -offset if m["score_deuce_half"] else offset
        if wrong > c["centre_margin_m"]:
            reasons.append("serve_wrong_half")
    return ("fail" if reasons else "pass"), reasons


# ---------------------------------------------------------------- b. net clearance


def measure_net_clearance(point: dict, i: int) -> dict:
    f = point["flights"][i]
    d = f["dense"]
    if d is None:
        return {"measured": False}
    frames = np.asarray(d["frames"], float)
    p = np.asarray(d["positions"], float)
    nets = [h["frame"] for h in d["net_hits"]]
    crossings = []
    for k in range(len(p) - 1):
        if frames[k] > f["end_frame"] + 1e-3:
            break
        y0, y1 = p[k, 1] - NET_Y_M, p[k + 1, 1] - NET_Y_M
        if y0 == 0 or y0 * y1 >= 0:
            continue
        s = y0 / (y0 - y1)
        x = p[k, 0] + s * (p[k + 1, 0] - p[k, 0])
        z = p[k, 2] + s * (p[k + 1, 2] - p[k, 2])
        frame = frames[k] + s * (frames[k + 1] - frames[k])
        if not -0.914 <= x <= 10.97 + 0.914:
            continue
        tape = net_height_at_x(float(x))
        crossings.append(
            {
                "frame": round(float(frame), 3),
                "x_m": round(float(x), 3),
                "tape_m": round(tape, 3),
                "clearance_m": round(float(z - tape), 3),
                "net_hit_near": any(abs(h - frame) <= CONSTANTS["net_hit_frames"] for h in nets),
            }
        )
    free = [x["clearance_m"] for x in crossings if not x["net_hit_near"]]
    return {
        "measured": True,
        "crossings": crossings,
        "min_clearance_m": min(free) if free else None,
    }


def decide_net_clearance(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured") or m["min_clearance_m"] is None:
        return "abstain", []
    if m["min_clearance_m"] < -c["net_margin_m"]:
        return "fail", ["below_tape"]
    return "pass", []


# ---------------------------------------------------------------- c. scoreboard


def _ending(f: dict, c: Mapping) -> tuple[str | None, dict]:
    """The fitted ending of a point's final flight: ``in`` / ``out`` / ``net`` / ``own_side`` /
    ``line`` (too close to call) or None (no fitted ending)."""
    d = f["dense"]
    xyz, _ = _launch(f)
    side0 = _side(float(xyz[1]))
    bounces = [b for b in d["bounces"] if _inside(f, b["frame"])]
    nets = [h["frame"] for h in d["net_hits"] if _inside(f, h["frame"])]
    info: dict = {}
    if bounces:
        first = bounces[0]
        info["first_bounce_xy_m"] = first["x"][:2]
        if _side(float(first["x"][1])) == side0:
            return ("net" if nets and nets[0] < first["frame"] else "own_side"), info
        if f["role"] == "serve" and float(xyz[2]) >= SERVE_MIN_HEIGHT_M:
            out, _ = _box_out(first["x"], xyz)
        else:
            out = _court_out(first["x"])
        info["out_m"] = round(out, 3)
        if abs(out) < c["line_margin_m"]:
            return "line", info
        return ("out" if out > 0 else "in"), info
    if nets:
        end = d["positions"][-1]
        if _side(float(end[1])) == side0:
            return "net", info
    return None, info


def complete_point(flights: list[dict]) -> bool:
    """One connected fit from a serve to the end: every fitted flight accepted, the first a
    serve, each starting where the previous ended, the last not ending at a racket contact."""
    if not flights or flights[0]["role"] != "serve" or not all(f["accepted"] for f in flights):
        return False
    joined = all(
        abs(b["start_frame"] - a["end_frame"]) <= 1.0 for a, b in zip(flights, flights[1:])
    )
    return joined and not _ends_at_contact(flights[-1])


def measure_scoreboard(point: dict, i: int) -> dict:
    flights = point["flights"]
    f = flights[i]
    score = point.get("score") or {}
    m: dict = {"measured": False}
    if f["dense"] is None:
        return m | {"why": "no_dense"}
    if i != len(flights) - 1 or f["role"] != "final" and f["role"] != "serve":
        return m | {"why": "not_final_flight"}
    if _ends_at_contact(f):
        return m | {"why": "ends_at_contact"}
    if not score.get("state"):
        return m | {"why": "score_unknown"}
    serve = f["role"] == "serve"
    if not score.get("last_attempt_of_point"):
        return m | {"why": "not_last_attempt_of_point"}
    if score.get("winner_row") is None:
        return m | {"why": "winner_unknown"}
    if score.get("server_side") is None:
        return m | {"why": "server_end_unknown"}
    ending, info = _ending(f, CONSTANTS)
    xyz, _ = _launch(f)
    hitter_side = "near" if float(xyz[1]) < NET_Y_M else "far"
    server_row = int(score["server_row"])
    hitter_row = server_row if hitter_side == score["server_side"] else 3 - server_row
    if ending in ("in",):
        implied = hitter_row
    elif ending in ("out", "net", "own_side"):
        implied = 3 - hitter_row
        if serve and score.get("attempts_in_point", 1) < 2:
            # A first serve that misses is a fault, not the point (unless the ledger saw a
            # second attempt, this attempt is the point's last and only one).
            implied = None
    else:
        implied = None
    return {
        "measured": True,
        "ending": ending,
        **info,
        "hitter_side": hitter_side,
        "hitter_row": hitter_row,
        "implied_winner_row": implied,
        "score_winner_row": score["winner_row"],
        "score_confidence": (score.get("state") or {}).get("confidence"),
        "next_confidence": (score.get("next_state") or {}).get("confidence"),
        "complete_point": complete_point(flights),
    }


def decide_scoreboard(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured") or m.get("implied_winner_row") is None:
        return "abstain", []
    conf = min(m.get("score_confidence") or 0.0, m.get("next_confidence") or 0.0)
    if conf < c["score_min_confidence"]:
        return "abstain", []
    if m["implied_winner_row"] != m["score_winner_row"]:
        # On a complete point (one connected fit, serve to the end) the fit's final flight is
        # the point's last shot; recorded separately as a diagnostic (coordinator 2026-10-10).
        extra = ["winner_disagrees_complete_point"] if m.get("complete_point") else []
        return "fail", ["winner_disagrees", *extra]
    return "pass", []


# ---------------------------------------------------------------- d. hitter reach


def measure_hitter_reach(point: dict, i: int) -> dict:
    f = point["flights"][i]
    contact = next((c for c in f["contacts"] if c["end"] == "start"), None)
    if contact is None or contact["kind"] != "contact" or contact["xyz"] is None:
        if f["role"] == "serve" and f["dense"] is not None:
            xyz = f["dense"]["positions"][0]
            return {"measured": True, "serve": True, "launch_z_m": round(float(xyz[2]), 3)}
        return {"measured": False}
    reach = contact.get("foot_reach") or {}
    return {
        "measured": True,
        "serve": f["role"] == "serve",
        "launch_z_m": round(float(contact["xyz"][2]), 3),
        "foot_reach_m": reach.get("reach_m"),
        "foot_sigma_m": reach.get("sigma_m"),
        "root_to_contact_m": contact.get("root_to_contact_m"),
    }


def decide_hitter_reach(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured"):
        return "abstain", []
    reasons = []
    z = m["launch_z_m"]
    if (
        m["serve"]
        and z >= SERVE_MIN_HEIGHT_M - 0.5
        and not c["serve_z_min_m"] <= z <= c["serve_z_max_m"]
    ):
        reasons.append("serve_height")
    if m.get("foot_reach_m") is not None:
        if m["foot_reach_m"] - (m.get("foot_sigma_m") or 0.0) > c["reach_max_m"]:
            reasons.append("out_of_reach")
    elif not reasons and not m["serve"]:
        return "abstain", []
    return ("fail" if reasons else "pass"), reasons


# ---------------------------------------------------------------- e. unused residual


def measure_unused_residual(point: dict, i: int) -> dict:
    f = point["flights"][i]
    d = f["dense"]
    if d is None:
        return {"measured": False}
    used = set(point.get("used_frames") or [])
    cams = point.get("cameras") or {}
    end = f["end_frame"]
    inside, after = [], []
    for frame, u, v in point.get("observations") or []:
        if frame in used or str(frame) not in cams:
            continue
        if f["start_frame"] < frame < end:
            target = inside
        elif end < frame <= end + CONSTANTS["after_end_frames"]:
            target = after
        else:
            continue
        xyz = position_at(d, float(frame))
        if xyz is None:
            continue
        px = project(cams[str(frame)], xyz)
        target.append([frame, round(float(np.hypot(px[0] - u, px[1] - v)), 1)])
    return {"measured": True, "inside": inside, "after": after}


def decide_unused_residual(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured") or not (m["inside"] or m["after"]):
        return "abstain", []
    reasons = []
    if sum(r > c["residual_px"] for _, r in m["inside"]) >= c["residual_min_frames"]:
        reasons.append("unused_observations_missed")
    if sum(r > c["residual_px"] for _, r in m["after"]) >= c["residual_min_frames"]:
        reasons.append("after_end_observations_missed")
    return ("fail" if reasons else "pass"), reasons


# ---------------------------------------------------------------- f. physics


def measure_physics(point: dict, i: int) -> dict:
    f = point["flights"][i]
    d = f["dense"]
    if d is None:
        return {"measured": False}
    p = np.asarray(d["positions"], float)
    inside = p[np.asarray(d["frames"]) <= f["end_frame"] + 1e-3]
    m: dict = {
        "measured": True,
        "serve": f["role"] == "serve",
        "launch_speed_mps": round(float(np.linalg.norm(d["v_start"])), 2) if d["v_start"] else None,
        "apex_z_m": round(float(inside[:, 2].max()), 3) if len(inside) else None,
    }
    bounces = [b for b in d["bounces"] if _inside(f, b["frame"])]
    if bounces and _ends_at_contact(f) and d.get("fps"):
        m["bounce_to_hit_s"] = round((f["end_frame"] - bounces[-1]["frame"]) / d["fps"], 4)
    ratios = []
    for b in bounces:
        if b.get("v_in") and b.get("v_out"):
            vin, vout = np.asarray(b["v_in"]), np.asarray(b["v_out"])
            if np.linalg.norm(vin[:2]) > 1e-6:
                ratios.append(round(float(np.linalg.norm(vout[:2]) / np.linalg.norm(vin[:2])), 3))
    m["bounce_horizontal_speed_kept"] = ratios
    return m


def decide_physics(m: dict, c: Mapping = CONSTANTS) -> tuple[str, list[str]]:
    if not m.get("measured"):
        return "abstain", []
    reasons = []
    cap = c["serve_speed_max_mps"] if m["serve"] else c["rally_speed_max_mps"]
    if m["launch_speed_mps"] is not None and m["launch_speed_mps"] > cap:
        reasons.append("too_fast")
    if m["apex_z_m"] is not None and m["apex_z_m"] > c["apex_max_m"]:
        reasons.append("apex_too_high")
    gap = m.get("bounce_to_hit_s")
    if gap is not None and gap < c["bounce_to_hit_min_s"]:
        reasons.append("bounce_to_hit_too_short")
    return ("fail" if reasons else "pass"), reasons


# ---------------------------------------------------------------- evaluation

MEASURE = {
    "grammar": measure_grammar,
    "net_clearance": measure_net_clearance,
    "scoreboard": measure_scoreboard,
    "hitter_reach": measure_hitter_reach,
    "unused_residual": measure_unused_residual,
    "physics": measure_physics,
}
DECIDE = {
    "grammar": decide_grammar,
    "net_clearance": decide_net_clearance,
    "scoreboard": decide_scoreboard,
    "hitter_reach": decide_hitter_reach,
    "unused_residual": decide_unused_residual,
    "physics": decide_physics,
}
#: The failing reasons that demote, as ``check:reason`` (every other reason is recorded only).
#: Chosen on development data (lane step 3, ``cv/experiments/post_fit_checks/REPORT.md``): the
#: physically impossible fitted outputs. Own-side landings, serve-box and deuce/ad calls, the
#: scoreboard, reach, serve height, unused observations and the bounce-to-hit gap flagged too
#: many right flights (or were unmeasurable) on the known answers.
ACTIVE: tuple[str, ...] = (
    "grammar:no_crossing_before_contact",
    "grammar:two_bounces_before_contact",
    "net_clearance:below_tape",
    "physics:too_fast",
    "physics:apex_too_high",
)
ALL_REASONS: tuple[str, ...] = (
    "grammar:own_side_landing",
    "grammar:no_crossing_before_contact",
    "grammar:two_bounces_before_contact",
    "grammar:serve_out_rally_continued",
    "grammar:serve_wrong_half",
    "grammar:net_then_own_side_bounce",
    "net_clearance:below_tape",
    "scoreboard:winner_disagrees",
    "scoreboard:winner_disagrees_complete_point",
    "hitter_reach:serve_height",
    "hitter_reach:out_of_reach",
    "unused_residual:unused_observations_missed",
    "unused_residual:after_end_observations_missed",
    "physics:too_fast",
    "physics:apex_too_high",
    "physics:bounce_to_hit_too_short",
)


def span_key(start: float, end: float) -> tuple[float, float]:
    return (round(float(start), 4), round(float(end), 4))


def measure_point(point: dict, spans: list[tuple[float, float]]) -> dict[tuple, dict]:
    """Measures of every check for each validated flight span (verdict start, untrimmed end);
    spans without an accepted verdict flight are omitted."""
    out = {}
    wanted = set(spans)
    for i, f in enumerate(point["flights"]):
        span = span_key(f["start_frame"], f["end_frame"])
        if not f["accepted"] or span not in wanted:
            continue
        out[span] = {name: MEASURE[name](point, i) for name in CHECKS}
    return out


def decide(
    measures: Mapping[str, dict],
    constants: Mapping = CONSTANTS,
    active: tuple[str, ...] | None = None,
) -> dict:
    """Per-flight decision record from the measures of :func:`measure_point`."""
    active = ACTIVE if active is None else active
    checks, reasons = {}, []
    for name in CHECKS:
        status, why = DECIDE[name](measures[name], constants)
        checks[name] = {"status": status, "reasons": why, "measures": measures[name]}
        if status == "fail":
            reasons += [f"post_fit_{name}:{r}" for r in why if f"{name}:{r}" in active]
    return {
        "checks_version": CHECKS_VERSION,
        "active": list(active),
        "demote": bool(reasons),
        "reasons": reasons,
        "checks": checks,
    }


def evaluate(point: dict, spans: list[tuple[float, float]]) -> dict[tuple, dict]:
    return {span: decide(m) for span, m in measure_point(point, spans).items()}
