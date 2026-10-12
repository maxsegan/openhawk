"""Release a band abstention the ball track and the rally both support.

The second tier after ``held_release_20260926``. A contact or bounce the decoder
abstained on with an acceptance marginal in ``[FLOOR, threshold)`` becomes an
accepted row only when all of these hold, none of which reads a label:

* every gate hold on the row is one ``confident_hold_release`` would clear
  (tracking arc, camera-supported frame, clean-cadence repeated frame);
* the frame is inside a live play-camera span;
* the observed image track turns by at least ``MIN_TURN_DEGREES`` at a pivot
  within ``RADIUS`` frames of the row;
* the row is interior to the attempt window's accepted rally: an accepted
  physical row comes before it and an accepted contact after it, with no
  accepted ``point_end`` in between.

The last rule is what keeps pre-serve dribbles, dead balls and aftermath taps
out; the 2026-09-27 census on opened panels 0/A/C/D and development found no
non-competitive row that satisfies it. Candidates are judged against the
accepted rows before this tier runs, so one release never supports another.
"""

from __future__ import annotations

import math

from cv.pipeline.confident_hold_release import (
    RELEASABLE,
    _covers,
    _marginal,
    _reason_passes,
    _same_clip,
)

POLICY_KEY = "held_release_v2_20260927"
SCHEMA = "supported_band_abstention_release_v1"
FLOOR = 0.8
RELEASED_TYPES = frozenset({"contact", "bounce"})
PHYSICAL = frozenset({"contact", "bounce", "net_hit"})
MIN_TURN_DEGREES = 30.0
RADIUS = 2  # pivot frames either side of the row
LAG = 3  # furthest observed sample used for each displacement
MIN_STEP_PX = 1.0
INTERIOR_GAP = 1.0  # an accepted neighbour must be more than this many frames away


def track_turn(track: dict, frame: float, radius: int = RADIUS, lag: int = LAG) -> float | None:
    """Largest direction change of the observed image track near ``frame``, in degrees."""

    best = None
    centre = int(round(float(frame)))
    for t in range(centre - radius, centre + radius + 1):
        here = track.get(t)
        before = next((track[t - k] for k in range(lag, 0, -1) if (t - k) in track), None)
        after = next((track[t + k] for k in range(lag, 0, -1) if (t + k) in track), None)
        if here is None or before is None or after is None:
            continue
        v_in = [float(here[0] - before[0]), float(here[1] - before[1])]
        v_out = [float(after[0] - here[0]), float(after[1] - here[1])]
        n_in, n_out = math.hypot(*v_in), math.hypot(*v_out)
        if n_in < MIN_STEP_PX or n_out < MIN_STEP_PX:
            continue
        cosine = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (n_in * n_out)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        best = angle if best is None else max(best, angle)
    return best


def _threshold(row: dict) -> float | None:
    value = row.get("decision_threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _interior(frame: float, accepted: list[dict]) -> bool:
    before = [r for r in accepted if float(r["frame"]) < frame - INTERIOR_GAP]
    after = [r for r in accepted if float(r["frame"]) > frame + INTERIOR_GAP]
    if not any(r["event_type"] in PHYSICAL for r in before):
        return False
    later = sorted(after, key=lambda r: float(r["frame"]))
    for row in later:
        if row["event_type"] == "point_end":
            return False
        if row["event_type"] == "contact":
            return True
    return False


def release_supported_abstentions(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    window: tuple[float, float],
    context: dict | None,
    release: bool,
) -> tuple[list[dict], dict]:
    """Clear eligible band abstentions. ``release`` False returns the rows unchanged."""

    census = {"schema": SCHEMA, "released": [], "kept": {}}
    if not release or context is None:
        return rows, census
    track = context.get("track") or {}
    live = context.get("live_spans")
    lo, hi = float(window[0]), float(window[1])
    accepted = [
        row
        for row in rows
        if _same_clip(row, match_id, clip)
        and not row.get("abstain", False)
        and isinstance(row.get("frame"), (int, float))
        and lo <= float(row["frame"]) <= hi
    ]

    def keep(why: str) -> None:
        census["kept"][why] = census["kept"].get(why, 0) + 1

    output = []
    for source in rows:
        frame = source.get("frame")
        marginal = _marginal(source)
        threshold = _threshold(source)
        reasons = list(source.get("point_gate_failure_reasons") or [])
        if (
            not _same_clip(source, match_id, clip)
            or source.get("event_type") not in RELEASED_TYPES
            or source.get("abstain") is not True
            or source.get("model_abstain") is not True
            or not isinstance(frame, (int, float))
            or not lo <= float(frame) <= hi
            or marginal is None
            or threshold is None
            or not FLOOR <= marginal < threshold
        ):
            output.append(source)
            continue
        frame = float(frame)
        why = None
        if any(reason not in RELEASABLE for reason in reasons):
            why = "hold_not_releasable"
        else:
            for reason in reasons:
                ok, detail = _reason_passes(
                    reason, frame, context.get("camera_spans"), context.get("cadence")
                )
                if not ok:
                    why = detail
                    break
        if why is None and not _covers(live, frame):
            why = "outside_live_span"
        turn = None if why else track_turn(track, frame)
        if why is None and (turn is None or turn < MIN_TURN_DEGREES):
            why = "no_track_turn"
        if why is None and not _interior(frame, accepted):
            why = "not_interior_to_accepted_rally"
        if why is not None:
            keep(why)
            output.append(source)
            continue
        output.append(
            {
                **source,
                "abstain": False,
                "gate_held": False,
                "point_gate_verdict": "retain",
                "point_gate_failure_reasons": [],
                "held_release_v2": {
                    "schema": SCHEMA,
                    "floor": FLOOR,
                    "marginal": marginal,
                    "turn_degrees": round(turn, 1),
                    "cleared": reasons,
                },
            }
        )
        census["released"].append(
            {"event_type": source["event_type"], "frame": frame, "marginal": marginal}
        )
    return output, census
