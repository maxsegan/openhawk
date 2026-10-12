"""Post-processing validation of accepted S6 flights: physically implausible ones are tagged.

The fitter accepts a flight when its own gates pass; some accepted flights are still not a
racket-launched ball flight (a contact placed on a bounce, the serve toss, a real flight split
at a fake contact). This reads only the fit result (fitted 3D, the automatic event rows the fit
used) and, optionally, the players' court-plane foot points, never an evaluation label. Each
accepted flight gets ``{"ok", "reasons", "measures"}``; nothing is deleted, consumers drop or
keep tagged flights (the product counts only ``ok`` flights as confirmed output).

Reasons:

- ``launch_height``: the launching contact is fitted outside ``LAUNCH_Z_M`` (a racket contact
  is off the ground and within a jumping reach).
- ``low_launch`` (soft): fitted between ``LAUNCH_Z_M[0]`` and ``LOW_LAUNCH_M``. Low volleys and
  pick-ups are real, so it is only a tag; it joins the reasons (and removes the flight) only when
  another reason agrees.
- ``far_from_player``: the launching contact is horizontally more than ``REACH_M`` + the foot
  point's sigma + ``REACH_MARGIN_M`` from every player's foot point at that time (measured
  only when both players' boxes are present).
- ``starts_at_bounce``: the flight launches from a contact fitted below ``BOUNCE_LAUNCH_Z_M``
  within ``BOUNCE_NEAR`` frames of an automatic bounce row or of the previous accepted flight's
  fitted bounce, unless the ball reverses its court-length direction at the junction with the
  previous flight (a half volley is struck just after the bounce and goes back).
- ``toss_like``: the attempt's first accepted flight travels less than ``TOSS_TRAVEL_M``
  horizontally (a toss goes up and down at the server; any real flight crosses metres).
- ``away_from_net`` (soft): a contact-launched flight ends farther from the net than it started
  and on the same side. Tosses, knocks toward the back fence and balls bounced after the point
  do this (7 opened-panel beam extras, all post-point; the product smoke's 2), but so does a
  real mishit that ends the point (1 correct opened-panel flight), so like ``low_launch`` it is
  a tag that removes the flight only when another reason agrees.
- ``own_side_bounce``: the flight's first fitted bounce is on the launching contact's side of
  the net, with no net hit, between ``OWN_BOUNCE_MIN`` frames and ``OWN_BOUNCE_S`` seconds after
  the launch: a struck ball crosses the net before it lands (a bounce at the launch itself is a
  half volley's incoming bounce; later own-side landings are net-short balls).
- ``mid_flight_split``: the flight starts where the previous accepted flight ends and the
  ball keeps its direction across the junction (same court-length direction, horizontal
  heading within ``SPLIT_ANGLE_DEG``): a contact always sends the ball back.

Switch ``TENNIS_FLIGHT_PLAUSIBILITY`` (``off`` default, ``on``) adds two reasons:

- ``mean_speed``: the fitted path averages more than ``MEAN_SPEED_MAX_MPS`` (the best-effort
  tier's cap; a 230 km/h first serve averages ~45 m/s over its flight).
- ``serve_short_own_side``: a serve with no fitted net hit whose first fitted bounce is on the
  server's side more than ``SERVE_OWN_SIDE_M`` from the net, at any time (``own_side_bounce``
  covers 0.2 s only). Typically a serve into the net fitted as landing short.

Contact-gap lane 2026-10-04 blind audit on dry run 4: of 19,024 confirmed flights 116 fail one
(91 mean speed, 24 own-side serve); 24 sampled were 22 wrong, 2 doubtful, 0 ok, against 11 of
12 ok for mixed-in confirmed flights.

Switch ``TENNIS_FLIGHT_TRIM`` (``off`` default, ``on``) adds the wrong-flight shapes of the
dry-run-5 blind audits (``cv/experiments/flight_trim/REPORT.md``; thresholds frozen on the
audited development flights before any fresh sheet was drawn). Rejections:

- ``serve_bounce_before_net``: a serve's first fitted bounce is on the server's side, more than
  ``SERVE_BEFORE_NET_M`` from the net, before any fitted net hit; or a serve with neither a
  bounce nor a net hit never reaches the net and ends on the server's side.
- ``doubles_back``: the ball retreats more than ``REVERSAL_M`` along the court with no fitted
  net hit before it (only a racket sends it back).
- ``never_descends``: no bounce or net hit, the apex in the last ``RISING_APEX_SHARE`` of the
  path above ``RISING_END_Z_M``, and the ball either more than ``RISING_PAST_NET_M`` past the
  net or going away from it (a lob still rising when it leaves the picture before the net is
  kept).
- ``away_high_or_deep``: a contact-launched flight going away from the net on its own side
  (``away_from_net``) that climbs above ``AWAY_APEX_M`` or ends more than ``AWAY_BEHIND_M``
  behind a baseline: a ball knocked away after the point.
- ``unknown_origin_start``: the first flight of an unknown origin
  (``unknown_origin_flight``) starts more than ``UNKNOWN_START_BEHIND_M`` behind a baseline or
  higher than ``UNKNOWN_START_Z_M`` (not the "inside" limit, which removed good volleys).

Trims (the flight stays confirmed, cut short, with ``trim`` in its row; a rejection wins):

- ``net_rebound``: after a fitted net hit the ball retreats more than ``NET_RETREAT_M`` from the
  net; the path after the net hit is cut.
- ``past_back_wall``: the path is cut where it runs more than ``BACK_WALL_M`` behind a baseline.

A play-view check for single-serve attempts was not buildable from these inputs: the wrong
side-angle serves carry the clip's one propagated main-camera registration and shots the
classifier called play-camera (report section 2).
"""

from __future__ import annotations

import os

import numpy as np

from cv.pipeline import point_end_tag

SCHEMA = "flight_validation_v1"
#: 3.5 m (owner example) tagged two correct opened-panel serves fitted at 3.57-3.60 m. The floor is
#: the ball on the ground: 4 correct opened-panel launches were fitted at 0.05-0.11 m.
LAUNCH_Z_M = (0.02, 3.7)
LOW_LAUNCH_M = 0.15
REACH_M = 2.5
#: Wider than the beam's pre-acceptance margin (1.5 m): validation also reads every baseline
#: flight, and at 1.5 m it tagged 5 correct opened-panel contacts (foot reach 4.9-6.6 m).
REACH_MARGIN_M = 4.0
FOOT_SIGMA = 0.1  # box heights
NEAR_FRAMES = 3
BOUNCE_NEAR = 2.0
#: A launch next to a bounce is the bounce only if fitted at the ground (a 2.1 m contact next to
#: an automatic bounce row was a correct contact).
BOUNCE_LAUNCH_Z_M = 0.5
TOSS_TRAVEL_M = 5.0
NET_Y_M = 11.885
OWN_BOUNCE_MIN = 1.0
#: The one opened-panel case (a contact placed 3.2 frames before an emitted bounce, launch at
#: 1.4 m) landed 0.11 s after launch; the nearest correct own-side landing was 0.8 s after.
OWN_BOUNCE_S = 0.2
DENSE_HZ = 240.0
SPLIT_JOIN = 1.0
SPLIT_ANGLE_DEG = 25.0
PLAUSIBILITY_SWITCH = "TENNIS_FLIGHT_PLAUSIBILITY"
MEAN_SPEED_MAX_MPS = 60.0
SERVE_OWN_SIDE_M = 1.5


TRIM_SWITCH = "TENNIS_FLIGHT_TRIM"
COURT_LENGTH_M = 23.77
COURT_CENTRE_X_M = 5.485
SERVE_BEFORE_NET_M = 0.3
#: Serves into the net whose observations stop 2.2-2.6 m short of it were judged right.
SERVE_SHORT_END_M = 3.0
REVERSAL_M = 0.3
#: A judged-right net fault rolled 1.9 m back; the wrong rebounds ran 2.7-4.8 m.
NET_RETREAT_M = 2.0
RISING_APEX_SHARE = 0.9
RISING_END_Z_M = 2.5
RISING_PAST_NET_M = 3.0
AWAY_APEX_M = 4.0
AWAY_BEHIND_M = 6.0
#: Returners stand up to ~8 m behind on clay; 6.2-8.5 m starts were mostly judged doubtful,
#: not wrong. The height limit is the launch-height reach (``LAUNCH_Z_M``).
UNKNOWN_START_BEHIND_M = 9.0
UNKNOWN_START_Z_M = 3.7
#: Run-off behind a baseline is 6.4-11 m; a fitted path farther back is past the back wall.
BACK_WALL_M = 10.0
UNKNOWN_ORIGIN_ROLE = "unknown_origin_flight"


def plausibility_enabled(environ=None) -> bool:
    return (environ or os.environ).get(PLAUSIBILITY_SWITCH, "off") == "on"


def trim_enabled(environ=None) -> bool:
    return (environ or os.environ).get(TRIM_SWITCH, "off") == "on"


def _behind(y: float) -> float:
    return max(-y, y - COURT_LENGTH_M, 0.0)


def trim_checks(
    d: dict,
    f: dict,
    *,
    index: int,
    count: int,
    launch_xyz,
    is_contact: bool,
) -> tuple[list[str], dict | None, dict]:
    """``TENNIS_FLIGHT_TRIM`` reasons, an optional trim and the measures, for one flight.

    ``d``: the fitted dense flight (240 Hz positions over its start..end frames, bounces and
    net hits with frames); ``f``: the verdict flight; ``index``/``count``: its place among the
    attempt's accepted flights."""
    path = np.asarray(d["positions"], float)
    start, end = float(d["start_frame"]), float(d["end_frame"])
    n = len(path)

    def at(frame: float) -> int:
        share = (float(frame) - start) / (end - start) if end > start else 0.0
        return int(np.clip(round(share * (n - 1)), 0, n - 1))

    y, z = path[:, 1], path[:, 2]
    p0 = np.asarray(launch_xyz if launch_xyz is not None else d["start_xyz"], float)
    side = np.sign(p0[1] - NET_Y_M)
    bounces = sorted(d.get("bounces") or [], key=lambda b: float(b["frame"]))
    nets = sorted(float(h["frame"]) for h in d.get("net_hits") or [])
    role = f.get("role")
    reasons, trim = [], None
    apex = int(np.argmax(z))
    measures = {
        "apex_z_m": round(float(z[apex]), 3),
        "apex_share": round(apex / (n - 1), 3),
        "end_behind_m": round(_behind(float(y[-1])), 3),
    }
    if role == "serve" and side != 0:
        first = bounces[0] if bounces else None
        if first is not None and "x" in first:
            yb = float(first["x"][1]) - NET_Y_M
            before_net = not nets or nets[0] > float(first["frame"])
            measures["serve_first_bounce_net_m"] = round(abs(yb), 3)
            if np.sign(yb) == side and abs(yb) > SERVE_BEFORE_NET_M and before_net:
                reasons.append("serve_bounce_before_net")
        elif first is None and not nets:
            reached = bool(np.any((y - NET_Y_M) * side <= 0))
            short = abs(float(y[-1]) - NET_Y_M)
            measures["serve_end_net_m"] = round(short, 3)
            if not reached and short > SERVE_SHORT_END_M:
                reasons.append("serve_bounce_before_net")
    # Progress along the court in the launch direction (first 0.1 s), and the largest retreat.
    lead = min(n - 1, 24)
    direction = np.sign(y[lead] - y[0])
    if direction != 0 and n > 2:
        progress = (y - y[0]) * direction
        best = np.maximum.accumulate(progress)
        retreat = best - progress
        worst = int(np.argmax(retreat))
        measures["retreat_m"] = round(float(retreat[worst]), 3)
        if retreat[worst] > REVERSAL_M:
            # Where the retreat starts: the last furthest point before the worst retreat.
            turn = int(np.argmax(progress[: worst + 1]))
            net_before = [h for h in nets if at(h) <= turn + 1]
            if not net_before:
                reasons.append("doubles_back")
            else:
                from_net = float(np.max(np.abs(y[turn:] - NET_Y_M)))
                measures["net_retreat_m"] = round(from_net, 3)
                if from_net > NET_RETREAT_M:
                    trim = {"end_frame": round(net_before[0], 3), "reason": "net_rebound"}
    away = (
        is_contact
        and side != 0
        and (
            (float(y[-1]) - NET_Y_M) * side > 0
            and abs(float(y[-1]) - NET_Y_M) > abs(p0[1] - NET_Y_M)
        )
    )
    if (
        not bounces
        and not nets
        and apex >= RISING_APEX_SHARE * (n - 1)
        and z[apex] > RISING_END_Z_M
        and (away or (float(y[-1]) - NET_Y_M) * side < -RISING_PAST_NET_M)
    ):
        reasons.append("never_descends")
    if away:
        lands = [_behind(float(b["x"][1])) for b in bounces if "x" in b]
        deep = max([measures["end_behind_m"]] + lands)
        if z[apex] > AWAY_APEX_M or deep > AWAY_BEHIND_M:
            reasons.append("away_high_or_deep")
    if index == 0 and role == UNKNOWN_ORIGIN_ROLE:
        behind = _behind(float(p0[1]))
        measures["start_behind_m"] = round(behind, 3)
        if behind > UNKNOWN_START_BEHIND_M or p0[2] > UNKNOWN_START_Z_M:
            reasons.append("unknown_origin_start")
    past = np.flatnonzero(np.maximum(-y, y - COURT_LENGTH_M) > BACK_WALL_M)
    if len(past) and past[0] > 0:
        frame = round(start + (past[0] - 1) / (n - 1) * (end - start), 3)
        if trim is None or frame < trim["end_frame"]:
            trim = {"end_frame": frame, "reason": "past_back_wall"}
    if reasons:
        trim = None
    return reasons, trim, measures


def _ray_at_ground(P: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    eq = P[:2] - pixel[:, None] * P[2]
    return np.linalg.solve(eq[:, :2], -eq[:, 3])


def foot_reach(cameras: dict, boxes: dict, frame: float, xyz) -> dict | None:
    """Nearest player foot point (court plane) to the contact, horizontally, with a sigma.

    ``cameras``: ``{"cameras": [{"frame", "supported", "P"}]}``; the nearest frame within
    ``NEAR_FRAMES`` with a supported camera and a box is used. None when unmeasured."""
    if xyz is None:
        return None
    by_frame = {int(c["frame"]): c for c in cameras.get("cameras") or []}
    centre = int(round(frame))
    for d in sorted(range(-NEAR_FRAMES, NEAR_FRAMES + 1), key=abs):
        k = centre + d
        cam = by_frame.get(k)
        if cam is None or not cam.get("supported") or not boxes.get(k):
            continue
        P = np.asarray(cam["P"], float)
        # A missing player box makes the other player the "nearest" (correct contacts read
        # 10-30 m): measure only when both a near and a far box are present.
        if {row[0] for row in boxes[k]} < {"near", "far"}:
            continue
        feet = []
        for side, (x0, y0, x1, y1) in boxes[k]:
            h = max(y1 - y0, 1.0)
            bottom = np.array([(x0 + x1) / 2, y1])
            try:
                g = _ray_at_ground(P, bottom)
                lo = _ray_at_ground(P, bottom + [0, FOOT_SIGMA * h])
                hi = _ray_at_ground(P, bottom - [0, FOOT_SIGMA * h])
            except np.linalg.LinAlgError:
                continue
            sigma = max(np.linalg.norm(lo - g), np.linalg.norm(hi - g))
            feet.append((side, g, float(sigma)))
        if not feet:
            continue
        xy = np.asarray(xyz, float)[:2]
        side, g, sigma = min(feet, key=lambda f: np.linalg.norm(xy - f[1]) - f[2])
        dist = float(np.linalg.norm(xy - g))
        return {
            "reach_m": round(dist, 3),
            "sigma_m": round(sigma, 3),
            "side": side,
            "frame": k,
            "excess_m": round(max(0.0, dist - REACH_M - sigma), 3),
        }
    return None


def _dense(result: dict) -> dict:
    found = {}
    for component in result.get("component_results") or []:
        measurement = (component.get("result") or {}).get("measurement") or {}
        for flight in measurement.get("dense_flights") or []:
            found.setdefault(round(float(flight["start_frame"])), flight)
    for flight in (result.get("measurement") or {}).get("dense_flights") or []:
        found.setdefault(round(float(flight["start_frame"])), flight)
    return found


def _velocity(dense: dict | None, at: str) -> np.ndarray | None:
    if dense is None or not dense.get("velocities"):
        return None
    rows = dense["velocities"]
    return np.asarray(rows[0] if at == "start" else rows[-1], float)


def trim_rows(result: dict) -> dict[float, tuple]:
    """:func:`trim_checks` for every accepted verdict flight with a dense state, keyed by start
    frame (reads only the fit result)."""
    flights = [f for f in (result.get("verdict") or {}).get("flights") or [] if f.get("accepted")]
    flights.sort(key=lambda f: float(f["start_frame"]))
    dense = _dense(result)
    out = {}
    for index, f in enumerate(flights):
        start = float(f["start_frame"])
        d = dense.get(round(start))
        if d is None or len(d.get("positions") or []) < 2:
            continue
        launch = next((c for c in f.get("contacts") or [] if c.get("end") == "start"), None)
        is_contact = launch is not None and launch.get("kind") == "contact"
        xyz = (launch or {}).get("contact_xyz_m") or d.get("start_xyz")
        out[start] = trim_checks(
            d,
            f,
            index=index,
            count=len(flights),
            launch_xyz=xyz,
            is_contact=is_contact,
        )
    return out


def validate(
    result: dict,
    reach=None,
    plausibility: bool | None = None,
    trim: bool | None = None,
) -> list[dict]:
    """One row per accepted verdict flight, in frame order.

    ``reach``: optional ``(frame, xyz) -> foot_reach row | None``. ``plausibility``: the
    ``TENNIS_FLIGHT_PLAUSIBILITY`` reasons (default: the switch). ``trim``: the
    ``TENNIS_FLIGHT_TRIM`` reasons and trims (default: the switch)."""
    if plausibility is None:
        plausibility = plausibility_enabled()
    if trim is None:
        trim = trim_enabled()
    flights = [f for f in (result.get("verdict") or {}).get("flights") or [] if f.get("accepted")]
    flights.sort(key=lambda f: float(f["start_frame"]))
    dense = _dense(result)
    bounces = [
        float(e["frame"])
        for e in point_end_tag.result_events(result)
        if e.get("event_type") == "bounce"
    ]
    trimmed = trim_rows(result) if trim else {}
    rows, previous = [], None
    for index, f in enumerate(flights):
        start, end = float(f["start_frame"]), float(f["end_frame"])
        d = dense.get(round(start))
        launch = next((c for c in f.get("contacts") or [] if c.get("end") == "start"), None)
        is_contact = launch is not None and launch.get("kind") == "contact"
        xyz = (launch or {}).get("contact_xyz_m") or (d or {}).get("start_xyz")
        reasons, tags, measures, cut = [], [], {}, None
        if is_contact and xyz is not None:
            z = float(xyz[2])
            measures["launch_z_m"] = round(z, 3)
            if not LAUNCH_Z_M[0] <= z <= LAUNCH_Z_M[1]:
                reasons.append("launch_height")
            elif z < LOW_LAUNCH_M:
                tags.append("low_launch")
            r = reach(start, xyz) if reach is not None else None
            measures["reach"] = r
            if r is not None and r["excess_m"] > REACH_MARGIN_M:
                reasons.append("far_from_player")
        reverses = split = False
        if previous is not None and abs(float(previous["end_frame"]) - start) <= SPLIT_JOIN:
            v0 = _velocity(dense.get(round(float(previous["start_frame"]))), "end")
            v1 = _velocity(d, "start")
            if v0 is not None and v1 is not None:
                h0, h1 = v0[:2], v1[:2]
                n0, n1 = np.linalg.norm(h0), np.linalg.norm(h1)
                if n0 > 1e-6 and n1 > 1e-6:
                    angle = float(np.degrees(np.arccos(np.clip(h0 @ h1 / (n0 * n1), -1, 1))))
                    measures["junction_angle_deg"] = round(angle, 1)
                    reverses = bool(np.sign(h0[1]) != np.sign(h1[1]))
                    split = not reverses and angle <= SPLIT_ANGLE_DEG
        if is_contact:
            near = [b for b in bounces if abs(b - start) <= BOUNCE_NEAR]
            prev_d = None if previous is None else dense.get(round(float(previous["start_frame"])))
            fitted = [
                float(b["frame"])
                for b in (prev_d or {}).get("bounces") or []
                if abs(float(b["frame"]) - start) <= BOUNCE_NEAR
            ]
            z = None if xyz is None else float(xyz[2])
            if (near or fitted) and z is not None and z < BOUNCE_LAUNCH_Z_M and not reverses:
                measures["bounce_near"] = sorted(near + fitted)
                reasons.append("starts_at_bounce")
        if (
            is_contact
            and d is not None
            and d.get("bounces")
            and "x" in d["bounces"][0]
            and d.get("positions")
            and not d.get("net_hits")
        ):
            first = d["bounces"][0]
            y0 = float((xyz if xyz is not None else d["start_xyz"])[1])
            dt = float(first["frame"]) - start
            span = float(d["end_frame"]) - float(d["start_frame"])
            seconds = dt * (len(d["positions"]) - 1) / DENSE_HZ / span if span > 0 else None
            if (
                (y0 - NET_Y_M) * (float(first["x"][1]) - NET_Y_M) > 0
                and dt >= OWN_BOUNCE_MIN
                and seconds is not None
                and seconds <= OWN_BOUNCE_S
            ):
                measures["own_bounce_s"] = round(seconds, 3)
                reasons.append("own_side_bounce")
        if is_contact and d is not None:
            y_start = float(np.asarray(d["start_xyz"], float)[1]) - NET_Y_M
            y_end = float(np.asarray(d["end_xyz"], float)[1]) - NET_Y_M
            if y_start * y_end > 0 and abs(y_end) > abs(y_start):
                measures["away_from_net_m"] = round(abs(y_end) - abs(y_start), 3)
                tags.append("away_from_net")
        if index == 0 and d is not None:
            travel = float(
                np.linalg.norm(
                    np.asarray(d["end_xyz"], float)[:2] - np.asarray(d["start_xyz"], float)[:2]
                )
            )
            measures["travel_m"] = round(travel, 3)
            if travel < TOSS_TRAVEL_M:
                reasons.append("toss_like")
        if split:
            reasons.append("mid_flight_split")
        if plausibility and d is not None and len(d.get("positions") or []) > 1:
            path = np.asarray(d["positions"], float)
            seconds = (len(path) - 1) / DENSE_HZ
            speed = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) / seconds
            measures["mean_speed_mps"] = round(speed, 2)
            if speed > MEAN_SPEED_MAX_MPS:
                reasons.append("mean_speed")
            first = (d.get("bounces") or [{}])[0]
            if f.get("role") == "serve" and "x" in first and not d.get("net_hits"):
                y0 = float((xyz if xyz is not None else d["start_xyz"])[1]) - NET_Y_M
                y1 = float(first["x"][1]) - NET_Y_M
                if y0 * y1 > 0 and abs(y1) > SERVE_OWN_SIDE_M:
                    measures["serve_own_side_m"] = round(abs(y1), 3)
                    reasons.append("serve_short_own_side")
        if start in trimmed:
            extra, cut, more = trimmed[start]
            reasons += extra
            measures.update(more)
        if reasons:
            reasons += tags
            cut = None
        rows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "role": f.get("role"),
                "ok": not reasons,
                "reasons": reasons,
                "tags": tags,
                "measures": measures,
                **({"trim": cut} if cut else {}),
            }
        )
        if not reasons:
            # A removed flight is not the ball's path: the next junction is judged against
            # the last flight that passed.
            previous = f
    return rows


def rejected_starts(rows: list[dict]) -> list[float]:
    return [r["start_frame"] for r in rows if not r["ok"]]


def apply(result: dict, rows: list[dict]) -> dict:
    """A copy of ``result`` whose tagged flights are no longer accepted (kept, with reasons)."""
    import copy

    out = copy.deepcopy(result)
    bad = {round(r["start_frame"], 3): r["reasons"] for r in rows if not r["ok"]}
    verdict = out.get("verdict") or {}
    kept = []
    for i, f in enumerate(verdict.get("flights") or []):
        reasons = bad.get(round(float(f["start_frame"]), 3))
        if f.get("accepted") and reasons:
            f["accepted"] = False
            f["post_validation"] = {"schema": SCHEMA, "rejected": reasons}
        if f.get("accepted"):
            kept.append(f.get("flight_index", i))
    if "accepted_flight_count" in verdict:
        verdict["accepted_flight_count"] = len(kept)
    if "accepted_flight_indices" in verdict:
        verdict["accepted_flight_indices"] = kept
    out["post_validation"] = {"schema": SCHEMA, "rows": rows}
    return out
