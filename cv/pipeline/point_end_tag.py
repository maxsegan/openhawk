"""Post-processing tag for accepted flights fitted after the point is over.

A player knocking back a fault serve or a let is a real ball flight, and S6 may fit
and accept it. For the point-winner consumer it is a no-op, so fitting and planning
are unchanged: this only marks each emitted flight ``after_point_end`` with the
automatic call that decided it, and consumers drop or keep it.

The call reads only what the fit itself read (its automatic event rows) and the
fitted flights, never an evaluation label. The first call in time wins:

- ``net_serve`` / ``serve_long`` / ``serve_wide``: the serve's own landing row, read
  on the court camera (see ``dead_serve_call``).
- ``fitted_serve_fault``: the accepted serve flight's first fitted bounce lies
  outside the receiver's service box by more than ``FITTED_SERVE_FAULT_M``.

A serve call is dropped when the fitted flight from that contact starts below
``SERVE_MIN_HEIGHT_M``: it is a groundstroke, not a serve.
- ``out_bounce``: the first supported bounce outside the singles court by more than
  the shared out-bounce margin (``s6_contact_components.out_bounce_call``).

Every flight starting after the call is tagged, up to a contact that follows
``RESTART_S`` without any supported row (the next serve). The call and the serve
flight themselves stay untagged. Development evidence, 2026-09-28:
``cv/experiments/dead_ball/FINAL_REPORT_dead_ball.md``.
"""

from __future__ import annotations

from cv.pipeline import s6_contact_components as components
from cv.pipeline import s6_first_contact_role as role

SCHEMA = "point_end_tag_v1"
#: A serve lands within this of its contact; a groundstroke takes longer.
SERVE_WINDOW_S = 0.7
#: Serve landing rows on the far half read far more coarsely: good far-half serves
#: read 0.3 m and 0.8 m long in the 2026-09-28 census.
SERVE_FAULT_MARGIN_NEAR_M = 0.25
SERVE_FAULT_MARGIN_FAR_M = 1.0
#: Fitted serve landings of real rallies reach 0.55 m outside the box on the
#: opened panels; the one fitted fault knock-back read 0.87 m.
FITTED_SERVE_FAULT_M = 0.6
#: A fitted flight starting lower is not a serve, whatever its role: the first flight
#: of an attempt that starts mid-rally is named ``serve`` too (Paul-Vukic product
#: run: 0.1-0.7 m). Opened-panel serves start at 2.06 m and up except two.
SERVE_MIN_HEIGHT_M = 2.0
RESTART_S = 4.0

_NET_Y_M = 0.5 * components._COURT_LENGTH_M
_CENTRE_X_M = 0.5 * components._COURT_WIDTH_M


def _fps(events: list[dict]) -> float | None:
    for event in events:
        for holder in (event, role.source_record(event)):
            location = holder.get("automatic_location") or holder.get("location")
            if isinstance(location, dict) and location.get("fps"):
                return float(location["fps"])
    return None


def result_events(result: dict) -> list[dict]:
    """The event rows one S6 result was fitted on, in frame order."""
    plan = result.get("source_plan") or {}
    events = plan.get("original_events")
    if events is None:
        events = ((result.get("evaluation_context") or {}).get("attempt") or {}).get("events")
    return sorted(events or [], key=lambda row: float(row["frame"]))


def _box_out(xy: tuple[float, float], serve_y: float) -> float | None:
    """Distance outside the receiver's service box, or None on the server's half."""
    x, y = xy
    if (serve_y < _NET_Y_M) == (y < _NET_Y_M):
        return None
    long = abs(y - _NET_Y_M) - components._SERVICE_DEPTH_M
    wide = abs(x - _CENTRE_X_M) - components._SINGLES_HALF_WIDTH_M
    return max(long, wide)


def dead_serve_call(events: list[dict], fps: float | None = None) -> dict | None:
    """The serve's landing row when the serve is already dead, as a receipt, else None.

    The serve is the first supported contact; its landing is the first supported
    bounce after it, before any other contact and within ``SERVE_WINDOW_S``. A net
    row before the landing calls the serve (let or net fault). Otherwise the
    landing must be on the receiver's half and outside the box by more than the
    half's margin. An unread court position or no frame rate is not judged.
    """
    rows = sorted(events, key=lambda row: float(row["frame"]))
    fps = _fps(rows) or fps
    contacts = [e for e in rows if e.get("event_type") == "contact" and role.supported(e)]
    if not contacts or fps is None:
        return None
    start = float(contacts[0]["frame"])
    horizon = start + SERVE_WINDOW_S * fps
    if len(contacts) > 1:
        horizon = min(horizon, float(contacts[1]["frame"]))
    inside = [
        e
        for e in rows
        if start < float(e["frame"]) < horizon
        and e.get("event_type") in ("bounce", "net_hit")
        and role.supported(e)
    ]
    bounces = [e for e in inside if e["event_type"] == "bounce"]
    if not bounces:
        return None
    landing = bounces[0]
    frame = float(landing["frame"])
    receipt = dict(serve_frame=start, frame=frame)
    nets = [
        float(e["frame"])
        for e in inside
        if e["event_type"] == "net_hit" and float(e["frame"]) < frame
    ]
    if nets:
        return receipt | dict(evidence="net_serve", net_frames=nets)
    server = components._court_metres(contacts[0])
    ground = components._court_metres(landing)
    if server is None or ground is None:
        return None
    out = _box_out(ground, server[1])
    if out is None:
        return None
    near = ground[1] < _NET_Y_M
    margin = SERVE_FAULT_MARGIN_NEAR_M if near else SERVE_FAULT_MARGIN_FAR_M
    if out <= margin:
        return None
    long = abs(ground[1] - _NET_Y_M) - components._SERVICE_DEPTH_M
    return receipt | dict(
        evidence="serve_long" if long >= out else "serve_wide",
        court_xy_m=list(ground),
        out_by_m=out,
        margin_m=margin,
        landing_half="near" if near else "far",
    )


def fitted_serve_call(flights: list[dict]) -> dict | None:
    """The accepted serve flight's fitted landing when it is outside the box."""
    serves = [f for f in flights if f.get("role") == "serve" and f.get("bounces_xyz_m")]
    if not serves:
        return None
    serve = min(serves, key=lambda f: float(f["start_frame"]))
    if serve["start_xyz_m"][2] < SERVE_MIN_HEIGHT_M:
        return None
    landing = serve["bounces_xyz_m"][0]
    out = _box_out((landing[0], landing[1]), serve["start_xyz_m"][1])
    if out is None or out <= FITTED_SERVE_FAULT_M:
        return None
    return dict(
        evidence="fitted_serve_fault",
        serve_frame=float(serve["start_frame"]),
        # Every later flight starts after the serve flight.
        frame=float(serve["start_frame"]),
        court_xy_m=[landing[0], landing[1]],
        out_by_m=out,
        margin_m=FITTED_SERVE_FAULT_M,
    )


def point_end_call(
    events: list[dict], flights: list[dict], fps: float | None = None
) -> dict | None:
    """The earliest automatic point-end call of one attempt, or None."""
    serve = dead_serve_call(events, fps)
    if serve is not None and any(
        abs(float(f["start_frame"]) - serve["serve_frame"]) <= 3
        and f.get("start_xyz_m")
        and f["start_xyz_m"][2] < SERVE_MIN_HEIGHT_M
        for f in flights
    ):
        serve = None  # the fit says the first contact is no serve
    calls = [serve, fitted_serve_call(flights)]
    out = components.out_bounce_call(events)
    if out is not None:
        calls.append(dict(out, evidence=f"out_bounce:{out['evidence']}"))
    calls = [c for c in calls if c is not None]
    return min(calls, key=lambda c: float(c["frame"])) if calls else None


def _restart_frame(events: list[dict], call: dict, fps: float | None) -> float | None:
    """The first supported contact after ``RESTART_S`` of nothing, after the call."""
    fps = _fps(events) or fps
    if fps is None:
        return None
    previous = float(call["frame"])
    for event in events:
        frame = float(event["frame"])
        if frame <= float(call["frame"]) or not role.supported(event):
            continue
        if event.get("event_type") == "contact" and (frame - previous) / fps >= RESTART_S:
            return frame
        previous = frame
    return None


def tag(events: list[dict], flights: list[dict], fps: float | None = None) -> dict | None:
    """Mark each flight record ``after_point_end`` in place; return the attempt's call."""
    call = point_end_call(events, flights, fps)
    restart = _restart_frame(events, call, fps) if call is not None else None
    for flight in flights:
        start = float(flight["start_frame"])
        after = (
            call is not None
            and start > float(call["frame"])
            and (restart is None or start < restart - 0.5)
        )
        flight["after_point_end"] = bool(after)
        flight["point_end_signal"] = call["evidence"] if after else None
    if call is not None:
        call = dict(call, schema=SCHEMA, restart_frame=restart)
    return call


def tag_result(result: dict, flights: list[dict]) -> dict | None:
    """``tag`` on the event rows of the S6 result the flight records came from."""
    events = result_events(result)
    fps = ((result.get("evaluation_context") or {}).get("attempt") or {}).get("fps")
    return tag(events, flights, fps)
