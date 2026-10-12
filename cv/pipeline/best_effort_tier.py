"""Best-effort flights for the flight slots the validated tier leaves empty (post-S6, CPU only).

Owner 2026-10-03: alongside the validated flights (S6 gate-accepted, as ``assemble`` reports
them), offer a clearly labelled best-effort layer for every other original flight slot, and
count both yields separately. Nothing here changes S6, its gates or the validated flights; it
reads the final S6 result of an attempt and that attempt's own automatic inputs.

Tiers, best first (one per original flight slot):

- ``validated``: the flight ``assemble`` already emits (S6 gates passed). Not duplicated here.
- ``fitted_unvalidated``: the fitter's own final state for a slot whose flight failed one or
  more gates, kept only when it is plausible (:func:`path_checks`), follows the observed
  ball (median reprojection <= ``FITTED_MEDIAN_PX``) and did not fail a ``DEMOTED_GATES``
  gate; otherwise the slot is demoted to the anchored attempt below. Carries the failed gates, the fitter's residuals and the
  reprojection against the observed ball track. Fitted states, not observations.
- ``anchored``: no fitted state; a gravity-only (no drag, no spin) arc through the slot's 3D
  anchors in time order: contacts and bounces. Anchors come from a neighbouring fitted state
  that shares the contact (``fitted_neighbour``), a bounce pixel on the court plane
  (``court_plane_ray``) or a contact pixel's camera ray at the striker's foot position
  (``striker_ray``); a neighbour lends its contact only when it is validated or a kept fitted
  state. Interpolated states between anchors, not observations; kept only when plausible, not
  turning back at a bounce and following the observed ball at all (``ANCHORED_MEDIAN_PX``).
- ``none``: fewer than two usable anchors, an implausible or unsupported arc, or a slot after
  the point's out-bounce call (abstention; stays in the denominator).

Switch ``TENNIS_BEST_EFFORT_TIER`` (``off`` default, ``on``): when off, ``assemble`` writes
exactly what it wrote before. When on, it adds ``best_effort_flights`` to each attempt row and a
``best_effort`` block (tier counts and both yields) to ``MATCH_RESULT.json`` and ``OUTCOMES.json``;
every existing field is unchanged. No evaluation label is read.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "best_effort_tier_v3"
SWITCH = "TENNIS_BEST_EFFORT_TIER"
TIERS = ("validated", "fitted_unvalidated", "anchored", "none")
#: Tiers at or above each tier, for "every flight of the point in the tier or better".
AT_LEAST = {
    "validated": ("validated",),
    "extended": ("validated", "fitted_unvalidated", "anchored"),
}
GRAVITY = 9.81
BALL_RADIUS_M = 0.0325  # fitted bounce height of the S6 states
DENSE_HZ = 240.0
POSITION_STEP = 4  # every 4th 240 Hz sample, as assemble keeps validated flights
#: A camera within this many frames of an anchor's (sub-frame) epoch is used for its ray.
CAMERA_NEAR = 2
#: Box for 3D anchors (the S6 venue envelope).
VENUE_X_M = (-6.0, 16.97)
VENUE_Y_M = (-12.0, 35.77)
CONTACT_Z_M = (0.05, 3.8)
#: Plausibility of a best-effort path (v2, frozen 2026-10-03 after the first blind audit, whose
#: wrong flights started 15 m behind a baseline at 4-8 m, flew to 33-48 m, lasted 0.03 s, ran
#: at 145 m/s, passed under the tape or reversed at a bounce): every sample inside a 9 m
#: run-off envelope and under the apex ceiling, a launch at racket height, a minimum duration,
#: a mean speed range, and the tape cleared.
PATH_X_M = (-5.0, 15.97)
PATH_Y_M = (-9.0, 32.77)
APEX_MAX_M = 8.0
MIN_DURATION_S = 0.2
MEAN_SPEED_MPS = (2.0, 60.0)
NET_Y_M = 11.885
NET_CLEAR_M = 0.914
#: Agreement with the observed ball (median reprojection over at least MIN_COMPARED pictures).
#: Fitted states: 2/16 wrong at <= 10 px, 8/20 at 10-30 px, 19/23 above 30 px in the first audit.
#: Anchored arcs only need to follow the ball at all (several good ones sat at 30-60 px).
MIN_COMPARED = 5
FITTED_MEDIAN_PX = 15.0
ANCHORED_MEDIAN_PX = 50.0
#: v3 (2026-10-03, second blind audit): fitted states that failed these S6 gates are demoted to
#: the anchored attempt even when plausible (5/16 wrong there against 3/44 for the other gates;
#: 15/24 and 10/15 wrong in the first audit).
DEMOTED_GATES = ("bounce_rays_agree", "flight_reprojection_supported")
#: A contact ray anchored at a foot point farther than this horizontally is not used.
STRIKER_REACH_M = 3.0
#: Slot plan reasons that place the slot after the point ended (S6 out-bounce call).
AFTER_POINT_END = ("post_out_bounce_origin",)
NOTE = (
    "Best-effort states are not physics-validated and are not observations: "
    "fitted_unvalidated = the fitter's own state for a slot that failed its gates; "
    "anchored = a gravity-only arc interpolated between 3D anchors."
)


def enabled(environ=None) -> bool:
    return (environ or os.environ).get(SWITCH, "off") == "on"


# --------------------------------------------------------------------------- geometry


def _cameras(cameras: dict) -> dict[int, np.ndarray]:
    return {
        int(c["frame"]): np.asarray(c["P"], float)
        for c in cameras.get("cameras") or []
        if c.get("supported") and c.get("P")
    }


def _camera_near(by_frame: dict[int, np.ndarray], frame: float) -> np.ndarray | None:
    centre = int(round(frame))
    for d in sorted(range(-CAMERA_NEAR, CAMERA_NEAR + 1), key=abs):
        if centre + d in by_frame:
            return by_frame[centre + d]
    return None


def _ray(P: np.ndarray, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and unit direction of the ray through ``pixel``."""
    M, p4 = P[:, :3], P[:, 3]
    centre = -np.linalg.solve(M, p4)
    direction = np.linalg.solve(M, np.array([pixel[0], pixel[1], 1.0]))
    return centre, direction / np.linalg.norm(direction)


def ray_at_height(P: np.ndarray, pixel, z: float) -> np.ndarray | None:
    centre, d = _ray(P, pixel)
    if abs(d[2]) < 1e-9:
        return None
    s = (z - centre[2]) / d[2]
    return centre + s * d if s > 0 else None


def ray_at_foot(P: np.ndarray, pixel, foot_xy) -> tuple[np.ndarray, float] | None:
    """The ray point horizontally nearest a foot point, and that horizontal distance."""
    centre, d = _ray(P, pixel)
    dxy = d[:2]
    if float(dxy @ dxy) < 1e-12:
        return None
    s = float((np.asarray(foot_xy) - centre[:2]) @ dxy / (dxy @ dxy))
    if s <= 0:
        return None
    point = centre + s * d
    return point, float(np.linalg.norm(point[:2] - np.asarray(foot_xy)))


def project(P: np.ndarray, xyz) -> np.ndarray:
    h = P @ np.append(np.asarray(xyz, float), 1.0)
    return h[:2] / h[2]


def _inside_venue(xyz) -> bool:
    x, y, _ = xyz
    return VENUE_X_M[0] <= x <= VENUE_X_M[1] and VENUE_Y_M[0] <= y <= VENUE_Y_M[1]


def ballistic(anchors: list[dict], fps: float) -> tuple[list[list[float]], list[float]] | None:
    """Gravity-only arcs through consecutive anchors at ``DENSE_HZ``; positions and frames."""
    positions, frames = [], []
    for a, b in zip(anchors, anchors[1:]):
        T = (b["frame"] - a["frame"]) / fps
        if T <= 0:
            return None
        p0, p1 = np.asarray(a["xyz_m"], float), np.asarray(b["xyz_m"], float)
        g = np.array([0.0, 0.0, -GRAVITY])
        v0 = (p1 - p0) / T - 0.5 * g * T
        n = max(int(np.ceil(T * DENSE_HZ)), 1)
        last = b is anchors[-1]
        for k in range(n + (1 if last else 0)):
            t = min(k / DENSE_HZ, T)
            positions.append((p0 + v0 * t + 0.5 * g * t * t).tolist())
            frames.append(a["frame"] + t * fps)
    return positions, frames


def path_checks(positions: list, frames: list, fps: float) -> list[str]:
    """Plausibility failures of a best-effort 3D path (empty when plausible)."""
    xyz = np.asarray(positions, float)
    frames = np.asarray(frames, float)
    failures = []
    if (
        xyz[:, 0].min() < PATH_X_M[0]
        or xyz[:, 0].max() > PATH_X_M[1]
        or xyz[:, 1].min() < PATH_Y_M[0]
        or xyz[:, 1].max() > PATH_Y_M[1]
    ):
        failures.append("outside_envelope")
    if xyz[:, 2].max() > APEX_MAX_M:
        failures.append("apex")
    if not CONTACT_Z_M[0] <= xyz[0, 2] <= CONTACT_Z_M[1]:
        failures.append("launch_height")
    duration = (frames[-1] - frames[0]) / fps
    if duration < MIN_DURATION_S:
        failures.append("fragment")
    else:
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() / duration
        if not MEAN_SPEED_MPS[0] <= speed <= MEAN_SPEED_MPS[1]:
            failures.append("speed")
    side = np.sign(xyz[:, 1] - NET_Y_M)
    for k in np.nonzero(side[:-1] * side[1:] < 0)[0]:
        w = (NET_Y_M - xyz[k, 1]) / (xyz[k + 1, 1] - xyz[k, 1])
        if xyz[k, 2] + w * (xyz[k + 1, 2] - xyz[k, 2]) < NET_CLEAR_M:
            failures.append("through_net")
            break
    return failures


def reverses_at_bounce(anchors: list[dict]) -> bool:
    """A bounce anchor across which the ball's horizontal travel turns back (no contact)."""
    for a, b, c in zip(anchors, anchors[1:], anchors[2:]):
        if b["kind"] != "bounce":
            continue
        before = np.subtract(b["xyz_m"], a["xyz_m"])[:2]
        after = np.subtract(c["xyz_m"], b["xyz_m"])[:2]
        if float(before @ after) < 0:
            return True
    return False


def support_checks(support: dict, limit_px: float) -> list[str]:
    if support.get("compared", 0) < MIN_COMPARED:
        return ["unsupported"]
    return ["residual"] if support["median_px"] > limit_px else []


# --------------------------------------------------------------------------- inputs


def observed_ball(labels: dict) -> dict[int, np.ndarray]:
    """Observed (visible) automatic ball centres by native frame."""
    seen = {}
    for record in (labels.get("ball") or {}).get("records") or []:
        for row in record.get("frames") or []:
            if row.get("status") == "visible" and row.get("x1080") is not None:
                seen[int(row["frame"])] = np.array([row["x1080"], row["y1080"]], float)
    return seen


def _foot_points(P: np.ndarray, rows: list) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(side, box, ground foot point) per player box at one frame."""
    feet = []
    for side, box in rows:
        bottom = np.array([(box[0] + box[2]) / 2, box[3]])
        ground = ray_at_height(P, bottom, 0.0)
        if ground is not None:
            feet.append((side, np.asarray(box, float), ground[:2]))
    return feet


def _event_pixel(event: dict | None) -> np.ndarray | None:
    location = (event or {}).get("automatic_location") or (event or {}).get("location") or {}
    if location.get("image_x") is None:
        return None
    return np.array([location["image_x"], location["image_y"]], float)


def _pixel_at(frame: float, event: dict | None, ball: dict[int, np.ndarray]):
    pixel = _event_pixel(event)
    if pixel is not None:
        return pixel, "event_location"
    for d in (0, -1, 1):
        if int(round(frame)) + d in ball:
            return ball[int(round(frame)) + d], "observed_ball"
    return None, None


def _event_at(events: list[dict], kind: str, frame: float) -> dict | None:
    near = [
        e for e in events if e.get("event_type") == kind and abs(float(e["frame"]) - frame) < 0.5
    ]
    return min(near, key=lambda e: abs(float(e["frame"]) - frame)) if near else None


def contact_anchor(frame, events, ball, cams, boxes) -> dict | None:
    """A contact's 3D anchor from its pixel ray at the nearest player's foot position."""
    P = _camera_near(cams, frame)
    pixel, pixel_source = _pixel_at(frame, _event_at(events, "contact", frame), ball)
    if P is None or pixel is None:
        return None
    rows = boxes.get(int(round(frame))) or []
    best = None
    for side, box, foot in _foot_points(P, rows):
        found = ray_at_foot(P, pixel, foot)
        if found is None:
            continue
        point, reach = found
        if best is None or reach < best[1]:
            best = (point, reach, side)
    if best is None:
        return None
    point, reach, side = best
    if reach > STRIKER_REACH_M or not (CONTACT_Z_M[0] <= point[2] <= CONTACT_Z_M[1]):
        return None
    if not _inside_venue(point):
        return None
    return {
        "kind": "contact",
        "frame": float(frame),
        "xyz_m": [round(float(v), 4) for v in point],
        "source": "striker_ray",
        "pixel_source": pixel_source,
        "striker_side": side,
        "ray_to_foot_m": round(reach, 3),
    }


def bounce_anchor(frame, events, ball, cams) -> dict | None:
    P = _camera_near(cams, frame)
    pixel, pixel_source = _pixel_at(frame, _event_at(events, "bounce", frame), ball)
    if P is None or pixel is None:
        return None
    point = ray_at_height(P, pixel, BALL_RADIUS_M)
    if point is None or not _inside_venue(point):
        return None
    return {
        "kind": "bounce",
        "frame": float(frame),
        "xyz_m": [round(float(v), 4) for v in point],
        "source": "court_plane_ray",
        "pixel_source": pixel_source,
    }


# --------------------------------------------------------------------------- per slot


def dense_by_slot(result: dict) -> dict[int, dict]:
    """The fitter's final dense state of each original slot that has one."""
    dense = {}
    for component in result.get("component_results") or []:
        measurement = (component.get("result") or {}).get("measurement") or {}
        for index, flight in enumerate(measurement.get("dense_flights") or []):
            dense[(component.get("component_index"), index)] = flight
    for index, flight in enumerate((result.get("measurement") or {}).get("dense_flights") or []):
        dense[(None, index)] = flight
    found = {}
    for flight in (result.get("verdict") or {}).get("flights") or []:
        state = dense.get((flight.get("component_index"), flight.get("flight_index")))
        if state is None or abs(float(state["start_frame"]) - float(flight["start_frame"])) > 1:
            continue
        found[flight.get("original_flight_index", flight.get("flight_index"))] = state
    return found


def _state_frames(state: dict) -> list[float]:
    n = len(state["positions"])
    start, end = float(state["start_frame"]), float(state["end_frame"])
    return list(np.linspace(start, end, n)) if n > 1 else [start]


def reprojection(positions, frames, ball, cams, start, end) -> dict:
    """Pixel residual of a 3D path against the observed ball inside [start, end]."""
    frames = np.asarray(frames, float)
    xyz = np.asarray(positions, float)
    inside = [f for f in sorted(ball) if start <= f <= end]
    errors = []
    for f in inside:
        P = cams.get(f)
        if P is None:
            continue
        k = int(np.argmin(np.abs(frames - f)))
        if abs(frames[k] - f) > 0.5:
            continue
        errors.append(float(np.linalg.norm(project(P, xyz[k]) - ball[f])))
    slot_frames = max(int(np.floor(end)) - int(np.ceil(start)) + 1, 0)
    out = {"slot_frames": slot_frames, "observed_frames": len(inside), "compared": len(errors)}
    if errors:
        e = np.asarray(errors)
        out.update(
            median_px=round(float(np.median(e)), 2),
            p90_px=round(float(np.percentile(e, 90)), 2),
            within_20px_share=round(float(np.mean(e <= 20.0)), 3),
        )
    return out


def _residuals(flight: dict) -> dict:
    return {
        k: round(float(flight[k]), 3) if isinstance(flight.get(k), (int, float)) else flight.get(k)
        for k in ("reprojection_rms_px", "withheld_rms_px", "maximum_window_rms_px")
    }


def attempt_layer(
    result: dict,
    accepted: list[dict],
    cameras: dict,
    labels: dict,
    boxes: dict[int, list],
    fps: float,
) -> list[dict]:
    """One row per original flight slot: its tier, and for best-effort tiers the 3D states."""
    verdict = result.get("verdict") or {}
    plan = result.get("source_plan") or {}
    slots_plan = {s.get("original_flight_index"): s for s in plan.get("original_slots") or []}
    events = plan.get("original_events") or []
    validated = {round(float(f["start_frame"]), 1) for f in accepted}
    dense = dense_by_slot(result)
    cams = _cameras(cameras)
    ball = observed_ball(labels)
    flights = sorted(
        verdict.get("flights") or [],
        key=lambda f: f.get("original_flight_index", f.get("flight_index")) or 0,
    )
    # Fitted states that pass the plausibility and support checks; only these (and validated
    # states) lend their contacts to an anchored neighbour.
    checked = {}
    for flight in flights:
        index = flight.get("original_flight_index", flight.get("flight_index"))
        state = dense.get(index)
        if state is None or round(float(flight["start_frame"]), 1) in validated:
            continue
        frames = _state_frames(state)
        support = reprojection(
            state["positions"],
            frames,
            ball,
            cams,
            float(flight["start_frame"]),
            float(flight["end_frame"]),
        )
        failures = path_checks(state["positions"], frames, fps) + support_checks(
            support, FITTED_MEDIAN_PX
        )
        failures += [f"gate:{g}" for g in flight.get("failures") or [] if g in DEMOTED_GATES]
        checked[index] = (frames, support, failures)
    trusted = {
        index: state
        for index, state in dense.items()
        if index not in checked or not checked[index][2]
    }
    rows = []
    for position, flight in enumerate(flights):
        index = flight.get("original_flight_index", flight.get("flight_index"))
        slot = slots_plan.get(index) or {}
        start, end = float(flight["start_frame"]), float(flight["end_frame"])
        plan_reasons = list(flight.get("reasons") or slot.get("reasons") or [])
        row = {
            "original_flight_index": index,
            "start_frame": start,
            "end_frame": end,
            "role": flight.get("role"),
            "after_point_end": any(r in AFTER_POINT_END for r in plan_reasons),
        }
        state = dense.get(index)
        if round(start, 1) in validated:
            # Positions stay in the validated flight; only its reprojection, as a reference.
            support = (
                reprojection(state["positions"], _state_frames(state), ball, cams, start, end)
                if state is not None
                else None
            )
            rows.append({**row, "tier": "validated", "support": support})
            continue
        gates = list(flight.get("failures") or [])
        demoted = None
        if state is not None:
            frames, support, failures = checked[index]
            if not failures:
                rows.append(
                    {
                        **row,
                        "tier": "fitted_unvalidated",
                        "state_kind": "fitted",
                        "gates_failed": gates,
                        "plan_reasons": plan_reasons,
                        "s6_accepted": bool(flight.get("accepted")),
                        "residuals": _residuals(flight),
                        "support": support,
                        "start_xyz_m": state["start_xyz"],
                        "end_xyz_m": state["end_xyz"],
                        "bounces_xyz_m": [b["x"] for b in state.get("bounces") or []],
                        "positions_m": state["positions"][::POSITION_STEP],
                        "frames": [round(float(f), 3) for f in frames[::POSITION_STEP]],
                        "positions_note": f"fitted dense samples, every {POSITION_STEP}th kept",
                    }
                )
                continue
            # The fitter's state is implausible or does not follow the ball: try anchors.
            demoted = {"checks_failed": failures, "support": support}
        if row["after_point_end"]:
            rows.append(
                {
                    **row,
                    "tier": "none",
                    "reason": "after_point_end_slot",
                    "plan_reasons": plan_reasons,
                    "demoted_fitted_state": demoted,
                }
            )
            continue
        anchors, missing = [], []
        previous = (
            trusted.get(flights[position - 1].get("original_flight_index")) if position else None
        )
        following = (
            trusted.get(flights[position + 1].get("original_flight_index"))
            if position + 1 < len(flights)
            else None
        )
        if previous is not None and abs(float(previous["end_frame"]) - start) < 0.5:
            anchors.append(
                {
                    "kind": "contact",
                    "frame": start,
                    "xyz_m": previous["end_xyz"],
                    "source": "fitted_neighbour",
                }
            )
        else:
            found = contact_anchor(start, events, ball, cams, boxes)
            anchors.append(found) if found else missing.append("start_contact")
        for frame in slot.get("source_bounce_frames") or []:
            if start < float(frame) < end:
                found = bounce_anchor(float(frame), events, ball, cams)
                anchors.append(found) if found else missing.append("bounce")
        next_contact = slot.get("known_next_contact_index") is not None or (
            slot.get("endpoint_semantics") == "confirmed_original_contact"
        )
        if following is not None and abs(float(following["start_frame"]) - end) < 0.5:
            anchors.append(
                {
                    "kind": "contact",
                    "frame": end,
                    "xyz_m": following["start_xyz"],
                    "source": "fitted_neighbour",
                }
            )
        elif next_contact:
            found = contact_anchor(end, events, ball, cams, boxes)
            anchors.append(found) if found else missing.append("end_contact")
        anchors.sort(key=lambda a: a["frame"])
        base = {
            **row,
            "plan_reasons": plan_reasons,
            "gates_failed": gates,
            "demoted_fitted_state": demoted,
            "anchors": anchors,
            "anchors_missing": missing,
        }
        if len(anchors) < 2 or anchors[0]["kind"] != "contact":
            rows.append({**base, "tier": "none", "reason": "too_few_anchors"})
            continue
        arc = ballistic(anchors, fps)
        if arc is None:
            rows.append({**base, "tier": "none", "reason": "implausible_arc:timing"})
            continue
        positions, frames = arc
        support = reprojection(positions, frames, ball, cams, start, end)
        failures = path_checks(positions, frames, fps)
        if reverses_at_bounce(anchors):
            failures.append("reverses_at_bounce")
        failures += support_checks(support, ANCHORED_MEDIAN_PX)
        if failures:
            rows.append(
                {
                    **base,
                    "tier": "none",
                    "reason": f"implausible_arc:{failures[0]}",
                    "checks_failed": failures,
                    "support": support,
                }
            )
            continue
        sources = sorted({a["source"] for a in anchors})
        rows.append(
            {
                **base,
                "tier": "anchored",
                "state_kind": "interpolated",
                "sub_tier": "fitted_neighbour"
                if sources == ["fitted_neighbour"]
                else ("mixed" if "fitted_neighbour" in sources else "rays_only"),
                "covers_slot_end": abs(anchors[-1]["frame"] - end) < 0.5,
                "support": support,
                "start_xyz_m": positions[0],
                "end_xyz_m": positions[-1],
                "bounces_xyz_m": [a["xyz_m"] for a in anchors if a["kind"] == "bounce"],
                "positions_m": positions[::POSITION_STEP],
                "frames": [round(float(f), 3) for f in frames[::POSITION_STEP]],
                "positions_note": f"gravity-only arc between anchors, {DENSE_HZ:.0f} Hz, "
                f"every {POSITION_STEP}th kept",
            }
        )
    return rows


def attempt_rows(result: dict, accepted: list[dict], row: dict) -> list[dict]:
    """:func:`attempt_layer` on the automatic inputs an attempt's fit row names."""
    from cv.pipeline import event_beam

    labels = _load(event_beam._resolve(row["labels"]))
    return attempt_layer(
        result,
        accepted,
        _load(event_beam._resolve(row["cameras"])),
        labels,
        event_beam.player_boxes(row),
        float(labels["source_pack"]["fps"]),
    )


# --------------------------------------------------------------------------- summary


def _complete(slots: list[dict], tiers, *, after_point_end: bool = False) -> bool:
    live = [s for s in slots if after_point_end or not s["after_point_end"]]
    return bool(live) and all(s["tier"] in tiers for s in live)


def summary(
    attempt_slots: list[list[dict]], attempts: int, s6_complete: list[bool] | None = None
) -> dict:
    """Tier counts and both yields over a match (``attempts``: every ledger attempt).

    ``s6_complete``: per attempt, S6's own ``complete_point`` verdict (all flights accepted and
    a classified physical ending; never true in the unresolved-ending component mode), kept
    as the validated tier's complete-point count "as now". Best-effort states cannot change
    that verdict, so the extended tier reports the slot rules only.
    """
    tiers = Counter(s["tier"] for slots in attempt_slots for s in slots)
    sub = Counter(
        f"{s['tier']}:{s.get('sub_tier') or (s.get('gates_failed') or ['-'])[0]}"
        for slots in attempt_slots
        for s in slots
        if s["tier"] in ("fitted_unvalidated", "anchored")
    )
    none = Counter(s.get("reason") for slots in attempt_slots for s in slots if s["tier"] == "none")
    slots_total = sum(tiers.values())
    yields = {}
    for name, accepted in AT_LEAST.items():
        yields[name] = {
            "tiers": list(accepted),
            "flights": sum(tiers[t] for t in accepted),
            "flight_slots": slots_total,
            "attempts_with_flights": sum(
                any(s["tier"] in accepted for s in slots) for slots in attempt_slots
            ),
            "complete_points_every_slot": sum(
                _complete(slots, accepted, after_point_end=True) for slots in attempt_slots
            ),
            "complete_points": sum(_complete(slots, accepted) for slots in attempt_slots),
            "attempts": attempts,
        }
    if s6_complete is not None:
        yields["validated"]["complete_points_s6_rule"] = sum(map(bool, s6_complete))
    return {
        "schema": SCHEMA,
        "switch": "on",
        "slots_by_tier": {t: tiers[t] for t in TIERS},
        "best_effort_by_sub_tier": dict(sorted(sub.items())),
        "none_by_reason": dict(none.most_common()),
        "yields": yields,
        "complete_point_rules": {
            "complete_points": "every original flight slot of the attempt is in the tier or "
            "better, slots after an S6 out-bounce call excluded; at least one slot",
            "complete_points_every_slot": "the same with every slot, including those after "
            "an out-bounce call",
            "complete_points_s6_rule": "S6's own complete_point (validated only; as reported "
            "before this tier existed)",
        },
        "note": NOTE,
    }


def portal_layer(match_result: dict) -> dict:
    """``portal/3d/data/best_effort.json``: the best-effort flights per attempt for the viewer.

    A separate file, so the validated point documents and the index stay byte-identical."""
    points = {}
    for attempt in match_result.get("attempts") or []:
        flights = []
        for slot in attempt.get("best_effort_flights") or []:
            if slot["tier"] not in ("fitted_unvalidated", "anchored"):
                continue
            if slot["tier"] == "anchored":
                support = slot.get("support") or {}
                label = f"BEST EFFORT · anchored arc ({support.get('observed_frames', 0)} ball pictures)"
            else:
                label = f"BEST EFFORT · fitted, failed {', '.join(slot['gates_failed']) or 'gates'}"
            flights.append(
                {
                    k: slot.get(k)
                    for k in (
                        "tier",
                        "sub_tier",
                        "original_flight_index",
                        "start_frame",
                        "end_frame",
                        "frames",
                        "positions_m",
                        "anchors",
                        "gates_failed",
                        "support",
                    )
                }
                | {"label": label}
            )
        if flights:
            points[attempt["key"]] = flights
    return {
        "schema": f"{SCHEMA}_portal",
        "match_id": match_result.get("match_id"),
        "note": NOTE,
        "points": points,
    }


def _load(path) -> dict:
    return json.loads(Path(path).read_text())
