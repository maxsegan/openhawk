"""``playstyle_pattern_v1``: the frozen play-style correctness contract and its scorer.

The product downstream of the 3D fit is play-style analysis: shot direction,
depth zone, bounce side, serve placement, approach/volley/lob, player zone and
movement.  Fine metric precision is not the product; a wrong *pattern* on an
accepted point still is the unrecoverable error.  This module freezes the field
list, the category boundaries, the bounded-boundary bands and the accept rules
so that a truth plate and a fitter output can be scored against the same
definitions.  ``docs/PLAYSTYLE_PATTERN_V1.md`` is the prose copy of what is here;
the numbers in this file are the normative ones.

Pure geometry and comparison only: no I/O, no labels, no fitting.  Court frame is
the ITF frame the rest of the repository uses (``cv.pipeline.camera_cal``):
``x`` across the doubles court in ``[0, 10.97]``, ``y`` along the court in
``[0, 23.77]``, ``z`` up, net at ``y = 11.885``.  The frame is right handed, so a
player standing at the near end faces ``+y`` and their right hand is at ``+x``.
"""

from __future__ import annotations

import math

VERSION = "playstyle_pattern_v1"
SCHEMA = "playstyle_pattern_contract_v1"

# --- court geometry, ITF, metres -------------------------------------------
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0  # 11.885
CENTRE_X_M = COURT_WIDTH_M / 2.0  # 5.485
SINGLES_INSET_M = 1.37
SINGLES_HALF_WIDTH_M = CENTRE_X_M - SINGLES_INSET_M  # 4.115
SERVICE_LINE_FROM_NET_M = 6.40
BASELINE_FROM_NET_M = NET_Y_M  # 11.885
#: The mid/deep split: halfway from the service line to the baseline.
MID_DEEP_FROM_NET_M = (SERVICE_LINE_FROM_NET_M + BASELINE_FROM_NET_M) / 2.0  # 9.1425
BALL_RADIUS_M = 0.0325
NET_HEIGHT_CENTRE_M = 0.914
NET_HEIGHT_POST_M = 1.07
NET_POST_OFFSET_M = 6.399  # posts 0.914 m outside the doubles sidelines

#: Lateral corridors: the singles width split into equal thirds.
CORRIDOR_WIDTH_M = 2.0 * SINGLES_HALF_WIDTH_M / 3.0  # 2.743333...
CORRIDOR_BOUNDS_M = (
    CENTRE_X_M - SINGLES_HALF_WIDTH_M + CORRIDOR_WIDTH_M,  # 4.113333
    CENTRE_X_M - SINGLES_HALF_WIDTH_M + 2.0 * CORRIDOR_WIDTH_M,  # 6.856667
)
#: Serve box thirds, measured from the centre service line out to the singles line.
SERVE_THIRD_M = SINGLES_HALF_WIDTH_M / 3.0  # 1.371667

# --- tolerances -------------------------------------------------------------
#: One band for every spatial category boundary in the contract.
BOUNDARY_BAND_M = 0.5
#: Player court position: nominal, and the wider allowance at a running or
#: airborne contact, which only applies when the zone and the movement pattern
#: are unchanged.
PLAYER_TOLERANCE_M = 0.30
PLAYER_RUNNING_TOLERANCE_M = 0.60
PLAYER_AIRBORNE_TOLERANCE_M = 0.90
#: Below this displacement a player's movement direction is not asserted.
MOVEMENT_ASSERTED_M = 2.0
MOVEMENT_HOLDING_M = 1.0
#: Recovery toward the centre mark.
RECOVERY_TOLERANCE_M = 1.0
#: Event epochs: one frame where the label carries a bracket, two otherwise.
TIMING_BRACKETED_FRAMES = 1.0
TIMING_UNBRACKETED_FRAMES = 2.0
#: Bounce epochs are judged at two frames whether or not the label carries a
#: bracket (owner, 2026-09-08: "even +/-2 frames passes my eye test just fine");
#: contact epochs keep the one-frame bracketed tolerance because a serve
#: modelled one frame late is visibly wrong (the serve starts too low).
BOUNCE_TIMING_FRAMES = 2.0
#: Net clearance geometry sigma; inside one sigma either net verdict is allowed.
NET_CLEARANCE_SIGMA_M = 0.10
#: Free, when every "must" field holds and the pattern is stable.
SPEED_FREE_FRACTION = 0.15
HEIGHT_FREE_M = 0.5

#: The fields a shot must get right for the point to be pattern-correct.
MUST_FIELDS = (
    "striker",
    "striker_end",
    "shot_class",
    "bounce_half",
    "bounce_side",
    "depth_zone",
    "direction",
    "in_out",
    "serve_box",
    "serve_third",
    "net_outcome",
    "contact_epoch",
    "bounce_epoch",
    "player_zone",
    "player_movement",
    "ending",
)
#: Reported, never a "must": these may vary among plausible solutions.
FREE_FIELDS = ("airborne_height", "speed", "spin", "lob_apex", "exact_clearance")

CONTRACT = {
    "schema": SCHEMA,
    "version": VERSION,
    "court_frame": (
        "ITF, x across the doubles court 0..10.97 m, y along the court 0..23.77 m, z up, "
        "net at y = 11.885 m; right handed, so a near-end player faces +y with their right at +x"
    ),
    "depth_boundaries_from_receiving_net_m": [
        SERVICE_LINE_FROM_NET_M,
        MID_DEEP_FROM_NET_M,
        BASELINE_FROM_NET_M,
    ],
    "lateral_corridor_bounds_m": list(CORRIDOR_BOUNDS_M),
    "serve_third_width_m": SERVE_THIRD_M,
    "boundary_band_m": BOUNDARY_BAND_M,
    "player_tolerance_m": PLAYER_TOLERANCE_M,
    "player_running_tolerance_m": PLAYER_RUNNING_TOLERANCE_M,
    "player_airborne_tolerance_m": PLAYER_AIRBORNE_TOLERANCE_M,
    "timing_bracketed_frames": TIMING_BRACKETED_FRAMES,
    "timing_unbracketed_frames": TIMING_UNBRACKETED_FRAMES,
    "bounce_timing_frames": BOUNCE_TIMING_FRAMES,
    "net_clearance_sigma_m": NET_CLEARANCE_SIGMA_M,
    "must_fields": list(MUST_FIELDS),
    "free_fields": list(FREE_FIELDS),
    "note": (
        "A 2 m bounce error is not excused by both ends being deep: direction, side, "
        "neighbouring shots and player movement must also hold. An explicit bounded boundary "
        "category is allowed only where the evidence straddles that boundary; an unrestricted "
        "unknown is not a free pass."
    ),
}


# --- banded classification --------------------------------------------------
def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def banded(value: float, bounds, labels, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """Classify ``value`` on an ordered axis and declare its boundary neighbours.

    ``bounds`` are the ascending cut points, ``labels`` the ``len(bounds) + 1``
    class names.  The result carries the class, the distance to the nearest cut,
    and every class the evidence is allowed to be within ``band_m`` of the cut.
    """
    if len(labels) != len(bounds) + 1:
        raise ValueError("one more label than boundary is required")
    if not _finite(value):
        return {
            "class": None,
            "margin_m": None,
            "boundary": False,
            "allowed": [],
            "status": "no_answer",
        }
    value = float(value)
    index = 0
    for cut in bounds:
        if value >= cut:
            index += 1
    margin = min(abs(value - cut) for cut in bounds) if bounds else float("inf")
    allowed = {labels[index]}
    for position, cut in enumerate(bounds):
        if abs(value - cut) <= band_m:
            allowed.add(labels[position])
            allowed.add(labels[position + 1])
    return {
        "class": labels[index],
        "margin_m": None if math.isinf(margin) else float(margin),
        "boundary": bool(margin <= band_m),
        "allowed": sorted(allowed),
        "status": "resolved",
    }


def corridor(x: float, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """The lateral corridor of a court x, in absolute court coordinates."""
    return banded(x, CORRIDOR_BOUNDS_M, ("left", "centre", "right"), band_m=band_m)


def receiving_end(striker_end: str) -> str:
    """The end the ball is travelling to."""
    if striker_end not in {"near", "far"}:
        raise ValueError("striker end must be near or far")
    return "far" if striker_end == "near" else "near"


def distance_from_net_m(y: float, end: str) -> float:
    """Signed distance from the net into ``end``'s half; negative is the far side of the net."""
    if end not in {"near", "far"}:
        raise ValueError("end must be near or far")
    return float(y - NET_Y_M) if end == "far" else float(NET_Y_M - y)


def court_half(y: float, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """Which half of the court a ground point is in."""
    return banded(y, (NET_Y_M,), ("near", "far"), band_m=band_m)


def court_side(x: float, end: str, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """Deuce or ad half, from the point of view of the player at ``end``.

    A near-end player faces +y with their right at +x, so their deuce court is
    ``x > 5.485``; the far-end player faces the other way and theirs is ``x < 5.485``.
    """
    labels = ("ad", "deuce") if end == "near" else ("deuce", "ad")
    return banded(x, (CENTRE_X_M,), labels, band_m=band_m)


def depth_zone(x: float, y: float, end: str, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """Short / mid / deep / out long / out wide for a ground point in ``end``'s half.

    Depth is measured from the receiving net.  ``out_long`` takes precedence over
    ``out_wide`` because a ball past the baseline is long however wide it is.
    """
    depth = distance_from_net_m(y, end)
    zone = banded(
        depth,
        (0.0, SERVICE_LINE_FROM_NET_M, MID_DEEP_FROM_NET_M, BASELINE_FROM_NET_M),
        ("own_half", "short", "mid", "deep", "out_long"),
        band_m=band_m,
    )
    lateral = banded(
        abs(float(x) - CENTRE_X_M),
        (SINGLES_HALF_WIDTH_M + BALL_RADIUS_M,),
        ("inside_singles", "out_wide"),
        band_m=band_m,
    )
    combined = dict(zone)
    combined["depth_from_net_m"] = float(depth)
    combined["lateral"] = lateral
    if zone["class"] not in {"out_long", "own_half"} and lateral["class"] == "out_wide":
        combined["class"] = "out_wide"
        combined["allowed"] = sorted(set(zone["allowed"]) | {"out_wide"})
        combined["margin_m"] = lateral["margin_m"]
        combined["boundary"] = bool(zone["boundary"] or lateral["boundary"])
    elif lateral["boundary"]:
        combined["allowed"] = sorted(set(combined["allowed"]) | {"out_wide"})
        combined["boundary"] = True
    return combined


def in_out(x: float, y: float, end: str, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """In or out against the singles lines, with the ball touching the line counted in."""
    depth = distance_from_net_m(y, end)
    long_margin = BASELINE_FROM_NET_M + BALL_RADIUS_M - depth
    wide_margin = SINGLES_HALF_WIDTH_M + BALL_RADIUS_M - abs(float(x) - CENTRE_X_M)
    margin = min(long_margin, wide_margin)
    allowed = {"in" if margin >= 0 else "out"}
    if abs(margin) <= band_m:
        allowed |= {"in", "out"}
    return {
        "class": "in" if margin >= 0 else "out",
        "margin_m": float(margin),
        "boundary": bool(abs(margin) <= band_m),
        "allowed": sorted(allowed),
        "status": "resolved",
    }


def direction(hitter_x: float, land_x: float, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """Cross-court, down the line or centre, relative to the hitter's own corridor.

    Both halves share the court ``x`` axis, so the corridors compare directly and
    no facing convention is needed.  A hitter standing in the centre third is not
    forced into the binary label; the result names the landing corridor instead.
    """
    hit, land = corridor(hitter_x, band_m=band_m), corridor(land_x, band_m=band_m)
    if hit["class"] is None or land["class"] is None:
        return {"class": None, "status": "no_answer", "allowed": [], "boundary": False}

    def label(hitter: str, landing: str) -> str:
        if landing == "centre":
            return "centre"
        if hitter == "centre":
            return f"from_centre_to_{landing}"
        return "down_the_line" if hitter == landing else "cross_court"

    allowed = {label(a, b) for a in hit["allowed"] for b in land["allowed"]}
    return {
        "class": label(hit["class"], land["class"]),
        "allowed": sorted(allowed),
        "boundary": bool(hit["boundary"] or land["boundary"]),
        "margin_m": min(v for v in (hit["margin_m"], land["margin_m"]) if v is not None),
        "hitter_corridor": hit["class"],
        "landing_corridor": land["class"],
        "signed_lateral_displacement_m": float(land_x) - float(hitter_x),
        "status": "resolved",
    }


def serve_placement(x: float, y: float, striker_end: str, *, band_m: float = BOUNDARY_BAND_M):
    """Target service box and the T / body / wide third inside it.

    The thirds are equal in lateral extent from the centre service line out to
    the singles sideline.  A landing that is not inside the receiver's service
    box is a fault, and the fault direction is reported instead of a third.
    """
    end = receiving_end(striker_end)
    depth = distance_from_net_m(y, end)
    box = court_side(x, end, band_m=band_m)
    third = banded(
        abs(float(x) - CENTRE_X_M),
        (SERVE_THIRD_M, 2.0 * SERVE_THIRD_M),
        ("T", "body", "wide"),
        band_m=band_m,
    )
    long_margin = SERVICE_LINE_FROM_NET_M + BALL_RADIUS_M - depth
    wide_margin = SINGLES_HALF_WIDTH_M + BALL_RADIUS_M - abs(float(x) - CENTRE_X_M)
    good = depth > 0 and long_margin >= 0 and wide_margin >= 0
    margin = min(long_margin, wide_margin, depth)
    call = {
        "class": "in" if good else "fault",
        "margin_m": float(margin),
        "boundary": bool(abs(margin) <= band_m),
        "allowed": sorted(
            {"in", "fault"} if abs(margin) <= band_m else {"in" if good else "fault"}
        ),
        "fault_direction": None
        if good
        else ("long" if long_margin < 0 else "wide" if wide_margin < 0 else "own_half"),
        "status": "resolved",
    }
    return box, third, call


def shot_class(has_preceding_bounce: bool, is_serve: bool, apex_height_m=None) -> dict:
    """Serve, groundstroke or volley, from the label event sequence alone.

    A contact with no bounce on the striker's side since the previous contact is
    an interception: a volley or half-volley.  Overhead and lob subtypes need
    trajectory or context support and are reported separately, never inferred
    from an apex threshold alone.
    """
    if is_serve:
        return {"class": "serve", "status": "resolved", "allowed": ["serve"], "boundary": False}
    label = "groundstroke" if has_preceding_bounce else "volley"
    return {"class": label, "status": "resolved", "allowed": [label], "boundary": False}


def net_tape_height_m(x: float) -> float:
    """The net band top at a court x, linear from the centre strap to the posts."""
    offset = min(abs(float(x) - CENTRE_X_M), NET_POST_OFFSET_M)
    span = offset / NET_POST_OFFSET_M
    return NET_HEIGHT_CENTRE_M + span * (NET_HEIGHT_POST_M - NET_HEIGHT_CENTRE_M)


def net_outcome(clearance_m, *, sigma_m: float = NET_CLEARANCE_SIGMA_M) -> dict:
    """Cleared, net contact or into the net, from the crossing clearance margin.

    ``clearance_m`` is the ball surface above the tape at the modelled crossing.
    Inside one geometry sigma the verdict is bounded and either neighbour counts.
    """
    if not _finite(clearance_m):
        return {"class": None, "status": "no_answer", "allowed": [], "boundary": False}
    value = float(clearance_m)
    label = "cleared" if value > sigma_m else "into_net" if value < -sigma_m else "net_contact"
    allowed = {label}
    if abs(value - sigma_m) <= sigma_m:
        allowed |= {"cleared", "net_contact"}
    if abs(value + sigma_m) <= sigma_m:
        allowed |= {"into_net", "net_contact"}
    return {
        "class": label,
        "clearance_m": value,
        "sigma_m": float(sigma_m),
        "boundary": bool(abs(value) <= 2.0 * sigma_m),
        "allowed": sorted(allowed),
        "status": "resolved",
    }


def player_zone(x: float, y: float, end: str, *, band_m: float = BOUNDARY_BAND_M) -> dict:
    """A player's occupancy: depth band at their own end, and lateral corridor."""
    depth = distance_from_net_m(y, end)
    band = banded(
        depth,
        (SERVICE_LINE_FROM_NET_M, BASELINE_FROM_NET_M),
        ("net", "midcourt", "behind_baseline"),
        band_m=band_m,
    )
    lateral = corridor(x, band_m=band_m)
    return {
        "depth_band": band,
        "lateral": lateral,
        "class": None if band["class"] is None else f"{band['class']}|{lateral['class']}",
        "allowed": sorted(f"{a}|{b}" for a in band["allowed"] for b in lateral["allowed"]),
        "boundary": bool(band["boundary"] or lateral["boundary"]),
        "status": band["status"],
    }


def player_movement(previous_xy, current_xy, end: str) -> dict:
    """Direction of travel between two contact epochs of the same player.

    The label is only asserted once the displacement reaches
    ``MOVEMENT_ASSERTED_M``; below ``MOVEMENT_HOLDING_M`` the player is holding
    position, and in between the direction is reported but bounded.
    """
    if previous_xy is None or current_xy is None:
        return {"class": None, "status": "no_answer", "allowed": [], "asserted": False}
    dx = float(current_xy[0]) - float(previous_xy[0])
    dnet = distance_from_net_m(float(current_xy[1]), end) - distance_from_net_m(
        float(previous_xy[1]), end
    )
    distance = math.hypot(dx, dnet)
    if distance < MOVEMENT_HOLDING_M:
        label = "holding"
    elif abs(dnet) >= abs(dx):
        label = "approach" if dnet < 0 else "retreat"
    else:
        label = "lateral_plus_x" if dx > 0 else "lateral_minus_x"
    asserted = distance >= MOVEMENT_ASSERTED_M
    return {
        "class": label,
        "displacement_m": float(distance),
        "asserted": bool(asserted),
        "allowed": [label] if asserted else sorted({label, "holding"}),
        "boundary": not asserted,
        "status": "resolved",
    }


def recovery(contact_xy, next_opponent_epoch_xy) -> dict:
    """Whether the player moved back toward the centre mark before the next contact."""
    if contact_xy is None or next_opponent_epoch_xy is None:
        return {"class": None, "status": "no_answer", "allowed": [], "boundary": False}
    before = abs(float(contact_xy[0]) - CENTRE_X_M)
    after = abs(float(next_opponent_epoch_xy[0]) - CENTRE_X_M)
    label = "recovered" if after < before - RECOVERY_TOLERANCE_M else "did_not_recover"
    boundary = abs((before - after) - RECOVERY_TOLERANCE_M) <= RECOVERY_TOLERANCE_M
    return {
        "class": label,
        "centre_gain_m": float(before - after),
        "boundary": bool(boundary),
        "allowed": sorted({"recovered", "did_not_recover"}) if boundary else [label],
        "status": "resolved",
    }


def timing_tolerance_frames(frame_interval) -> float:
    """One frame where the label carries a bracket, two where it does not."""
    try:
        low, high = float(frame_interval[0]), float(frame_interval[1])
    except (TypeError, ValueError, IndexError):
        return TIMING_UNBRACKETED_FRAMES
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return TIMING_UNBRACKETED_FRAMES
    return TIMING_BRACKETED_FRAMES


def bounce_timing_tolerance_frames(frame_interval) -> float:
    """Two frames for a bounce epoch, never tighter than the contact rule."""
    return max(timing_tolerance_frames(frame_interval), BOUNCE_TIMING_FRAMES)


#: Free-text ending labels collapse to these terminal classes.
ENDING_CLASSES = ("first_out_bounce", "second_bounce", "into_net", "exit", "wall", "other")

_ENDING_FAMILY_TO_CLASS = {
    "out": "first_out_bounce",
    "serve_fault": "first_out_bounce",
    "net": "into_net",
    "second_bounce": "second_bounce",
    "unreturned_or_winner": "second_bounce",
    "other": "other",
}


def ending_class(family: str | None, kind: str | None = None) -> dict:
    """Map a normalised ending family (and its raw kind) to a terminal class.

    ``unreturned_or_winner`` is a scoring outcome, not a physical ending; unless
    the raw kind names the physical event the ball still ends at its second
    bounce, so that is the default and the raw kind is retained beside it.
    """
    raw = (kind or "").lower()
    if "net" in raw and "unreturned" not in raw:
        resolved = "into_net"
    elif "second_bounce" in raw or "double_bounce" in raw:
        resolved = "second_bounce"
    elif any(token in raw for token in ("out", "long", "wide")) and "unreturned" not in raw:
        resolved = "first_out_bounce"
    else:
        resolved = _ENDING_FAMILY_TO_CLASS.get(family or "", "other")
    return {
        "class": resolved,
        "family": family,
        "kind": kind,
        "allowed": [resolved],
        "boundary": False,
        "status": "resolved" if resolved != "other" else "no_answer",
    }


# --- comparison -------------------------------------------------------------
def agree(truth: dict | None, output: dict | None) -> dict:
    """Compare one banded field: the output must sit inside the truth's allowed set.

    The band is the *truth's*: where the evidence itself straddles a boundary the
    neighbour counts, and where it does not, only the exact class counts.  An
    output that abstains is a no-answer, never a pass.
    """
    if truth is None or truth.get("status") == "no_answer" or truth.get("class") is None:
        return {"verdict": "no_truth", "truth": None, "output": (output or {}).get("class")}
    if output is None or output.get("class") is None:
        return {
            "verdict": "no_answer",
            "truth": truth["class"],
            "output": None,
            "boundary": bool(truth.get("boundary")),
        }
    allowed = set(truth.get("allowed") or [truth["class"]])
    ok = output["class"] in allowed
    exact = output["class"] == truth["class"]
    return {
        "verdict": "correct" if ok else "wrong",
        "exact": bool(exact),
        "used_band": bool(ok and not exact),
        "truth": truth["class"],
        "output": output["class"],
        "boundary": bool(truth.get("boundary")),
        "margin_m": truth.get("margin_m"),
    }


def agree_epoch(truth_frame, output_frame, tolerance_frames: float) -> dict:
    """Compare one event epoch against its contract tolerance."""
    if not _finite(truth_frame):
        return {"verdict": "no_truth", "truth": None, "output": output_frame}
    if not _finite(output_frame):
        return {"verdict": "no_answer", "truth": float(truth_frame), "output": None}
    delta = abs(float(output_frame) - float(truth_frame))
    return {
        "verdict": "correct" if delta <= tolerance_frames + 1e-9 else "wrong",
        "truth": float(truth_frame),
        "output": float(output_frame),
        "delta_frames": float(delta),
        "tolerance_frames": float(tolerance_frames),
    }


def agree_player_position(
    truth_xy,
    output_xy,
    *,
    running: bool = False,
    airborne: bool = False,
    zone_unchanged: bool = True,
    movement_unchanged: bool = True,
) -> dict:
    """Compare a player court position under the nominal and the widened tolerance.

    The wider running/airborne allowance is only available when the zone and the
    movement pattern are unchanged; otherwise the nominal 0.30 m applies.
    """
    if truth_xy is None or output_xy is None:
        return {"verdict": "no_answer", "error_m": None}
    error = math.dist(
        (float(truth_xy[0]), float(truth_xy[1])), (float(output_xy[0]), float(output_xy[1]))
    )
    tolerance = PLAYER_TOLERANCE_M
    widened = None
    if (running or airborne) and zone_unchanged and movement_unchanged:
        widened = PLAYER_AIRBORNE_TOLERANCE_M if airborne else PLAYER_RUNNING_TOLERANCE_M
        tolerance = widened
    return {
        "verdict": "correct" if error <= tolerance + 1e-9 else "wrong",
        "error_m": float(error),
        "tolerance_m": float(tolerance),
        "widened_tolerance_m": widened,
    }


def score_shot(truth: dict, output: dict) -> dict:
    """Score one flight's pattern vector against the plate's.

    ``truth`` and ``output`` are field name to banded-result mappings, plus the
    epoch entries ``contact_epoch``/``bounce_epoch`` carrying ``frame`` and
    ``tolerance_frames``.  A shot is pattern-correct when every must field that
    the plate can speak to is correct.
    """
    fields, failures, no_answer, banded_rescues = {}, [], [], []
    for name in MUST_FIELDS:
        if name in {"contact_epoch", "bounce_epoch"}:
            entry = truth.get(name) or {}
            result = agree_epoch(
                entry.get("frame"),
                (output.get(name) or {}).get("frame"),
                float(entry.get("tolerance_frames") or TIMING_UNBRACKETED_FRAMES),
            )
        else:
            result = agree(truth.get(name), output.get(name))
        fields[name] = result
        if result["verdict"] == "wrong":
            failures.append(name)
        elif result["verdict"] == "no_answer":
            no_answer.append(name)
        elif result.get("used_band"):
            banded_rescues.append(name)
    return {
        "pattern_correct": not failures and not no_answer,
        "failed_fields": failures,
        "no_answer_fields": no_answer,
        "banded_fields": banded_rescues,
        "fields": fields,
    }


def score_point(shot_scores: list[dict], *, accepted_flags: list[bool] | None = None) -> dict:
    """Roll shot verdicts up to the point.

    Pattern-correct requires every shot to hold.  A partial is a prefix or an
    interior run that holds with the rest declared as gaps.  ``accepted_flags``
    names which flights the fitter actually accepted, so a point that only
    accepted some of its flights can never be pattern-correct as a whole point.
    """
    if not shot_scores:
        return {
            "pattern_correct_point": False,
            "pattern_correct_shots": 0,
            "shot_count": 0,
            "partial": False,
            "failed_shot_indices": [],
            "failed_fields": [],
            "gaps": [],
        }
    flags = accepted_flags or [True] * len(shot_scores)
    good = [bool(s["pattern_correct"]) and bool(a) for s, a in zip(shot_scores, flags, strict=True)]
    failed = [i for i, ok in enumerate(good) if not ok]
    gaps, run = [], None
    for index, ok in enumerate(good):
        if ok and run is None:
            run = index
        elif not ok and run is not None:
            gaps.append([run, index - 1])
            run = None
    if run is not None:
        gaps.append([run, len(good) - 1])
    fields = sorted({f for s, ok in zip(shot_scores, good) if not ok for f in s["failed_fields"]})
    return {
        "pattern_correct_point": bool(good) and all(good),
        "pattern_correct_shots": int(sum(good)),
        "shot_count": len(good),
        "partial": bool(gaps) and not all(good),
        "failed_shot_indices": failed,
        "failed_fields": fields,
        "gaps": gaps,
    }


def transition_tag(strict_accepted: bool, pattern_correct: bool, banded_fields: list[str]) -> str:
    """Tag a strict-to-pattern status change ``refit``, ``boundary`` or ``contract``.

    Nothing is refitted by the scorer, so ``refit`` is never produced here; it is
    reserved for a run that changes the fit.  A change carried by a bounded
    boundary category is ``boundary``; one carried by the contract itself being
    looser than the strict spatial metric is ``contract``.
    """
    if strict_accepted == pattern_correct:
        return "unchanged"
    return "boundary" if banded_fields else "contract"


# --- distribution -----------------------------------------------------------
#: The slices published beside every accepted set, from the joint contract R2.4.
SLICES = (
    ("1_serve_only", lambda row: row.get("rally_length_band") == "1_serve_only"),
    ("2_serve_plus_return", lambda row: row.get("rally_length_band") == "2_serve_plus_return"),
    ("3_4_short_rally", lambda row: row.get("rally_length_band") == "3_4_short_rally"),
    ("5_8_medium_rally", lambda row: row.get("rally_length_band") == "5_8_medium_rally"),
    ("9_plus_long_rally", lambda row: row.get("rally_length_band") == "9_plus_long_rally"),
    ("hard", lambda row: row.get("surface") == "hard"),
    ("clay", lambda row: row.get("surface") == "clay"),
    ("grass", lambda row: row.get("surface") == "grass"),
    ("net_endings", lambda row: row.get("ending_family") == "net"),
    ("approach_or_volley", lambda row: bool(row.get("has_volley"))),
    ("lobs", lambda row: bool(row.get("has_lob"))),
    ("unreturned_winners", lambda row: row.get("ending_family") == "unreturned_or_winner"),
    (
        "second_serves_resolved",
        lambda row: bool(row.get("serve_number_resolved")) and row.get("serve_number") == 2,
    ),
    ("women", lambda row: row.get("gender") == "women"),
    ("men", lambda row: row.get("gender") == "men"),
)


def distribution_table(rows: list[dict], census: dict[str, int] | None = None) -> list[dict]:
    """The accepted-set distribution table every run prints.

    ``rows`` are one per attempt with the slice keys plus ``accepted`` and
    ``pattern_correct``.  ``census`` optionally supplies the population count for
    each slice name; a slice the census does not carry reports ``None`` rather
    than a guess.  Zero-coverage strata stay in the table at zero.
    """
    census = census or {}
    table = []
    for name, predicate in SLICES:
        pool = [row for row in rows if predicate(row)]
        table.append(
            {
                "slice": name,
                "pool": len(pool),
                "accepted": sum(1 for row in pool if row.get("accepted")),
                "pattern_correct": sum(1 for row in pool if row.get("pattern_correct")),
                "wrong_accepts": sum(
                    1 for row in pool if row.get("accepted") and not row.get("pattern_correct")
                ),
                "census": census.get(name),
            }
        )
    return table


def binomial_upper_bound(failures: int, trials: int, confidence: float = 0.95) -> float | None:
    """One-sided Clopper-Pearson upper bound on the wrong-accept rate.

    Zero wrong accepts in a small accepted set is weak evidence, and correlated
    broadcasts make it weaker still; publish this beside the count, never a
    claim of a zero rate.
    """
    if trials <= 0 or failures < 0 or failures > trials:
        return None
    if failures == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        tail = sum(
            math.comb(trials, k) * mid**k * (1.0 - mid) ** (trials - k)
            for k in range(0, failures + 1)
        )
        if tail > 1.0 - confidence:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)
