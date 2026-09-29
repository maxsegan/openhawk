"""Bounded optional interior-contact scope, default off.

The existing front door only inspects an accepted-contact gap that already
carries more than one supplied ground, and only admits a candidate that
separates a ground before it from a ground after it. A genuine one-ground or
zero-ground interior return -- a volley, or a rally ball whose neighbouring
ground the upstream stage correctly admits and separates -- therefore has no
expressible alternative at all: it is refused at qualification, before any
evidence is consulted.

This helper adds the missing qualification path and nothing else. It opens the
remaining interior gaps, which the existing required-repair branch never owns,
and it proposes candidates there under the unchanged source-held contact rule.
No manual source, clip or frame rule appears here. Every refusal is an explicit
receipt tagged ``Q`` (qualification: structural, deterministic, independent of
classifier calibration) or ``W`` (witness: evidential).

Nothing here fits, gates, retimes or ranks by residual. The added alternatives
are singletons, capped against the existing beam without evicting any of it,
and the selector's acceptance and complexity score stay exactly as they are.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import math
from pathlib import Path

import numpy as np

from cv.pipeline import event_topology
from cv.pipeline.s6_optional_contacts import INTERIOR_OPTIONAL

SCHEMA = "s6_witnessed_interior_contact_v1"
SCOPE = "witnessed_interior_contacts_v1"
OFF = "off"
ON = "on"
# The declared kind of the gaps this scope adds. The existing interior and final
# kinds are untouched, so every legacy consumer keeps its own gap selection.
GAP_KIND = INTERIOR_OPTIONAL
# Fixed total alternatives: max(2, existing branches). Two new singletons when
# the existing beam is empty, one when it holds a single branch, none beyond.
MAX_TOTAL_ALTERNATIVES = 2
# The one ball-side reach bound, reused unchanged from the existing insertion
# evidence, expressed as a fraction of the actor box diagonal.
MAX_PLAYER_DISTANCE_SCALE = event_topology.INSERTION_MAX_PLAYER_DISTANCE_SCALE
QUALIFICATION = "Q"
WITNESS = "W"


def policy(document: dict) -> str:
    """The witness's own interior-scope declaration; absence is off, not a default."""
    if "optional_interior_contacts" not in document:
        return OFF
    declaration = document["optional_interior_contacts"]
    if not isinstance(declaration, dict) or declaration.get("policy") not in {OFF, ON}:
        raise ValueError("unknown optional interior contact scope declaration")
    return declaration["policy"]


def _interval(event: dict) -> tuple[float, float]:
    a, b = event.get("frame_interval", [event["frame"], event["frame"]])
    return float(a), float(b)


def _rank_key(row: dict) -> tuple:
    # Identical ordering to ``s6_optional_contacts._ranked``: source witness
    # strength first, then the source class log-odds. No fitted quantity.
    return (-row["witness_support"], -row["occurrence_log_odds"], row["event"]["frame"], row["id"])


def scope_gaps(events: list[dict]) -> list[dict]:
    """Accepted contact-to-contact gaps the existing required repair never owns.

    A gap carrying more than one supplied ground is exactly the existing
    mandatory branch's gap and is skipped here, so no legacy gap can gain a new
    interval. Leading and trailing gaps are not contact-to-contact gaps and are
    never built, which leaves the final-contact scope untouched.
    """
    from cv.pipeline import s6_optional_contacts as optional

    claimed = {tuple(gap["interval"]) for gap in optional.inconsistent_gaps(events)}
    contacts = sorted((e for e in events if e["event_type"] == "contact"), key=lambda e: e["frame"])
    result = []
    for left, right in zip(contacts, contacts[1:]):
        bounces = [
            e
            for e in events
            if e["event_type"] == "bounce" and left["frame"] < e["frame"] < right["frame"]
        ]
        if (float(left["frame"]), float(right["frame"])) in claimed:
            continue
        result.append(
            {
                "interval": [float(left["frame"]), float(right["frame"])],
                "kind": GAP_KIND,
                "bounces": deepcopy(bounces),
                "left_contact_interval": list(_interval(left)),
                "right_contact_interval": list(_interval(right)),
            }
        )
    return result


def actor_boxes(players: Path, clip: str) -> tuple[dict[int, list[list[float]]], dict]:
    """The original declared actor boxes in their already validated native contract.

    Absent or undeclared coordinates abstain: the returned inventory is empty
    and every candidate then fails the ball-side witness with its own reason.
    No box is scaled by an assumed convention and none is invented.
    """
    from cv.pipeline import resolution
    from cv.pipeline.pose_player_crop import frame_number, native_box_for_row

    manifest = resolution.read_coordinate_manifest(players)
    if manifest is None:
        return {}, {"available": False, "reason": "actor_boxes_declare_no_coordinate_contract"}
    size = resolution.manifest_artifact_size(manifest)
    boxes: dict[int, list[list[float]]] = {}
    rows = 0
    total_rows = 0
    invalid_rows = 0
    with players.open(newline="") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            if row.get("clip") != clip:
                continue
            rows += 1
            try:
                confidence = float(row.get("conf", ""))
                if not math.isfinite(confidence) or confidence <= 0:
                    invalid_rows += 1
                    continue
                box = [
                    float(value) for value in native_box_for_row(row, size, resolution.NATIVE_SIZE)
                ]
                frame = frame_number(row["frame"])
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue
            if (
                not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                invalid_rows += 1
                continue
            boxes.setdefault(frame, []).append(box)
    return boxes, {
        "available": bool(boxes),
        "source": "original declared automatic actor boxes; inputs.players unchanged",
        "rows": rows,
        "total_rows": total_rows,
        "requested_clip": clip,
        "reason": "no_matching_clip_rows" if total_rows and not rows else None,
        "invalid_or_unconfident_rows": invalid_rows,
        "frames": len(boxes),
        "coordinate_space": "verified_native",
    }


def observed_points(
    observations: list[dict] | None, supported_frames: set[int]
) -> tuple[dict[int, tuple[float, float]], dict]:
    """Original visible non-interpolated native ball rows on supported frames.

    Admission is the existing composition helper's, unchanged: interpolated
    estimates, non-finite points and conflicting duplicates are not
    observations. Frames whose camera or declared view is not supported are
    then dropped, because an unsupported epoch cannot witness anything.
    """
    from cv.pipeline import s6_contact_composition as composition

    admitted, receipt = composition.admitted_observations(observations or [])
    keep = {frame for frame in admitted if frame in supported_frames}
    points: dict[int, tuple[float, float]] = {}
    for row in observations or []:
        # Frame membership alone cannot identify the admitted measured row when
        # a derived or invisible duplicate precedes it in the original inventory.
        if (
            row.get("status") != composition.VISIBLE
            or row.get("support_class") == composition.DERIVED_SUPPORT
        ):
            continue
        frame = row.get("frame")
        if (
            not isinstance(frame, (int, float))
            or isinstance(frame, bool)
            or not math.isfinite(frame)
            or float(frame) != int(frame)
        ):
            continue
        frame = int(frame)
        if frame not in keep or frame in points:
            continue
        try:
            point = (float(row["x1080"]), float(row["y1080"]))
            if all(math.isfinite(value) for value in point):
                points[frame] = point
        except (KeyError, TypeError, ValueError):
            continue
    return points, {
        **receipt,
        "admitted_frames_on_supported_views": len(points),
        "admitted_frames_on_unsupported_views": len(admitted) - len(keep),
    }


def _actor_witness(
    points: dict[int, tuple[float, float]],
    boxes: dict[int, list[list[float]]],
    low: float,
    high: float,
) -> tuple[dict | None, str]:
    """An actually measured ball row inside the candidate interval, at an actor.

    The ball row and the actor box are the SAME native frame: no predicted or
    fitted position, no head or model XY, no extrapolated bracket, and no
    tolerance derived from a fit error. Absent evidence abstains with its own
    reason rather than being replaced by a nearby guess.
    """
    inside = [frame for frame in sorted(points) if low <= frame <= high]
    if not inside:
        return None, "no_original_observation_inside_candidate_interval"
    measured = []
    for frame in inside:
        point = np.asarray(points[frame], float)
        for box in boxes.get(frame, []):
            # ``_distance_to_player_box`` doubles the box columns it reads, i.e.
            # it expects the original half-native tracking convention. Hand it
            # the verified native box in that same convention so the distance
            # and the box scale it returns are native pixels.
            row = {
                name: value / 2.0 for name, value in zip(("x0", "y0", "x1", "y1"), box, strict=True)
            }
            distance, scale = event_topology._distance_to_player_box(point, row)
            measured.append((distance / scale, distance, scale, frame))
    if not measured:
        # The ball is observed but no actor box shares its native frame: the
        # actor is occluded, untracked or outside the declared inventory.
        return None, "no_actor_box_on_the_observed_native_frame"
    distance_scale, distance, scale, frame = min(measured)
    witness = {
        "observation_frame": frame,
        "distance_px": distance,
        "actor_box_scale_px": scale,
        "distance_scale": distance_scale,
        "maximum_distance_scale": MAX_PLAYER_DISTANCE_SCALE,
        "source": "original_native_ball_observation_and_declared_actor_box",
        "same_native_frame": True,
        "predicted_position_used": False,
    }
    if distance_scale > MAX_PLAYER_DISTANCE_SCALE:
        return witness, "observed_ball_not_within_actor_reach"
    return witness, ""


def candidates(
    events: list[dict],
    emissions: list[dict],
    clip: str,
    pts: dict[int, float],
    gaps: list[dict],
    first_gap_index: int,
    *,
    points: dict[int, tuple[float, float]],
    boxes: dict[int, list[list[float]]],
    cuts: set[int],
    supported_frames: set[int],
) -> tuple[list[dict], list[dict]]:
    """Source-held contacts strictly inside one optional interior gap.

    Occurrence stays the source emission's own unchanged held-contact rule and
    the generated event epoch and interval are untouched. Everything added is
    grammar and witness: where the row lies relative to the accepted stream,
    whether both proposed sub-gaps keep enough original measured native ball
    rows on a continuous supported view, and whether a measured ball row inside
    the candidate interval is actually at an actor.
    """
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event
    from cv.pipeline import s6_contact_composition as composition

    frames = sorted(pts)
    if len(frames) < 2 or any(b - a != 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("complete ordered original native picture cadence required")
    window = (int(frames[0]), int(frames[-1]))
    blocking = [_interval(e) for e in events if e.get("event_type") in composition.BLOCKING_EVENTS]
    result: list[dict] = []
    exclusions: list[dict] = []

    def refuse(index, gap_index, event, reason, failure_class, **extra):
        exclusions.append(
            {
                "id": f"emission_{index}",
                "gap_index": gap_index,
                "gap_kind": GAP_KIND,
                "source_emission_index": index,
                "frame_interval": None if event is None else list(_interval(event)),
                "reason": reason,
                "failure_class": failure_class,
                **extra,
            }
        )

    for offset, gap in enumerate(gaps):
        gap_index = first_gap_index + offset
        low, high = (float(value) for value in gap["interval"])
        left_edge = float(gap["left_contact_interval"][1])
        right_edge = float(gap["right_contact_interval"][0])
        grounds = [float(e["frame"]) for e in gap["bounces"]]
        for index, row in enumerate(emissions):
            from cv.pipeline import s6_optional_contacts as optional

            held = optional._held_source_contact(row, clip)
            if held is None:
                continue
            contact, none = held
            event = automatic_physical_event(row, prediction_window=window)
            a, b = (float(value) for value in event["frame_interval"])
            if not low < a <= float(event["frame"]) <= b < high:
                continue
            if any(start <= b and a <= end for start, end in blocking):
                refuse(
                    index,
                    gap_index,
                    event,
                    "candidate_overlaps_a_supplied_event_interval",
                    QUALIFICATION,
                )
                continue
            span = range(int(math.floor(left_edge)), int(math.ceil(right_edge)) + 1)
            if any(frame in cuts for frame in span):
                refuse(
                    index,
                    gap_index,
                    event,
                    "native_view_cut_inside_a_proposed_subgap",
                    QUALIFICATION,
                )
                continue
            interval_frames = range(int(math.floor(a)), int(math.ceil(b)) + 1)
            if any(frame not in pts for frame in interval_frames):
                refuse(
                    index,
                    gap_index,
                    event,
                    "candidate_interval_outside_the_original_native_pictures",
                    QUALIFICATION,
                )
                continue
            if any(frame not in supported_frames for frame in interval_frames):
                refuse(
                    index,
                    gap_index,
                    event,
                    "unsupported_camera_or_view_in_candidate_interval",
                    QUALIFICATION,
                )
                continue
            if any(frame not in supported_frames for frame in span):
                refuse(
                    index,
                    gap_index,
                    event,
                    "unsupported_camera_or_view_inside_a_proposed_subgap",
                    QUALIFICATION,
                )
                continue
            subgaps = [
                {
                    "name": name,
                    "interval": [start, end],
                    "observation_count": sum(1 for f in points if start < f < end),
                    "supplied_bounce_count": sum(1 for f in grounds if start < f < end),
                }
                for name, start, end in (
                    ("left_contact_to_candidate", left_edge, a),
                    ("candidate_to_right_contact", b, right_edge),
                )
            ]
            if any(entry["supplied_bounce_count"] > 1 for entry in subgaps):
                refuse(
                    index,
                    gap_index,
                    event,
                    "more_than_one_supplied_bounce_in_one_new_subgap",
                    QUALIFICATION,
                    subgaps=subgaps,
                )
                continue
            if sum(entry["supplied_bounce_count"] for entry in subgaps) != len(grounds):
                refuse(
                    index,
                    gap_index,
                    event,
                    "supplied_bounce_lies_in_no_new_subgap",
                    QUALIFICATION,
                    subgaps=subgaps,
                )
                continue
            if any(entry["observation_count"] < composition.MIN_OBSERVATIONS for entry in subgaps):
                refuse(
                    index,
                    gap_index,
                    event,
                    "new_subgap_below_the_original_observation_floor",
                    QUALIFICATION,
                    subgaps=subgaps,
                    minimum_observations=composition.MIN_OBSERVATIONS,
                )
                continue
            actor, reason = _actor_witness(points, boxes, a, b)
            if reason:
                refuse(
                    index,
                    gap_index,
                    event,
                    reason,
                    WITNESS,
                    **({"actor_witness": actor} if actor is not None else {}),
                )
                continue
            event.update(
                note="optional classifier contact; independently witnessed occurrence",
                occurrence_status="optional",
                status="predicted",
            )
            result.append(
                {
                    "id": f"emission_{index}",
                    "gap_index": gap_index,
                    "gap_interval": gap["interval"],
                    "gap_kind": GAP_KIND,
                    "event": event,
                    "source_emission_index": index,
                    "source_emission": row,
                    "source_pts_interval": [
                        float(value)
                        for value in np.interp(
                            [a, b], np.asarray(frames, float), [pts[f] for f in frames]
                        )
                    ],
                    "occurrence_log_odds": float(
                        np.clip(math.log(contact / max(none, 1e-9)), -1, 1)
                    ),
                    "actor_witness": actor,
                    "subgaps": subgaps,
                }
            )
    return result, exclusions


def _conditioned_event(candidate: dict) -> dict:
    event = deepcopy(candidate["event"])
    event["occurrence_status"] = "predicted"
    event["optional_gap_kind"] = GAP_KIND
    event["optional_topology_membership"] = {
        "candidate_id": candidate["id"],
        "conditioned_on_this_hypothesis": True,
        "accepted_source_stream_changed": False,
    }
    return event


def expansion_receipt(document: dict, legacy: list[dict]) -> dict:
    """Record the fixed cap even when it admits no new alternative.

    Pure source-only diagnostic; never modifies the bound witness or existing
    alternatives. Preparation stores it before hashing the completed witness.
    """
    supported = [
        row
        for row in document.get("candidates", [])
        if row.get("gap_kind") == GAP_KIND and row.get("supported")
    ]
    total = max(MAX_TOTAL_ALTERNATIVES, len(legacy))
    allowed = max(0, total - len(legacy))
    ordered = sorted(supported, key=_rank_key)
    capped = ordered[allowed:]
    return {
        "schema": SCHEMA,
        "policy": ON,
        "scope": SCOPE,
        "existing_alternatives": len(legacy),
        "maximum_total_alternatives": total,
        "added_alternatives_allowed": allowed,
        "supported_candidate_ids": [row["id"] for row in ordered],
        "cap_excluded": [
            {"id": row["id"], "reason": "fixed_total_alternative_cap_reached"} for row in capped
        ],
        "singletons_only": True,
        "existing_branches_evicted": False,
        "source_evidence_rank_only": True,
        "fitted_or_gated": False,
        "runtime_model_calls": 0,
    }


def extend_hypotheses(document: dict, events: list[dict], legacy: list[dict]) -> list[dict]:
    """Append at most a bounded number of singleton interior alternatives.

    The existing branches are returned first, unchanged in order and content;
    nothing is ever evicted. A new branch is one existing branch -- the source
    best, when there is one -- plus exactly ONE new interior candidate, so two
    new interior candidates are never paired and an added contact never opens
    its own sub-gaps. The cap is a fixed total, and the candidates it leaves out
    are recorded rather than silently dropped.
    """
    if policy(document) == OFF:
        return legacy
    receipt = expansion_receipt(document, legacy)
    supported = [
        row
        for row in document.get("candidates", [])
        if row.get("gap_kind") == GAP_KIND and row.get("supported")
    ]
    chosen = sorted(supported, key=_rank_key)[: receipt["added_alternatives_allowed"]]
    if not chosen:
        return legacy
    # The optional interior gaps are exactly the gaps the existing required
    # repair does not own, so a new singleton never shares a gap with an
    # existing branch's additions and cannot change their composition.
    base = legacy[0] if legacy else None
    result = list(legacy)
    for position, row in enumerate(chosen, 1):
        event = _conditioned_event(row)
        original = deepcopy(base["events"]) if base is not None else deepcopy(events)
        added = ([*deepcopy(base["added"])] if base is not None else []) + [event]
        result.append(
            {
                "name": f"optional_contacts_interior_{position}",
                "events": sorted([*original, event], key=lambda e: e["frame"]),
                "added": added,
                "candidate_ids": [
                    *(base["candidate_ids"] if base is not None else []),
                    row["id"],
                ],
                "occurrence_log_odds": (
                    (base["occurrence_log_odds"] if base is not None else 0.0)
                    + row["occurrence_log_odds"]
                ),
                "witnessed_interior_contacts": {
                    **receipt,
                    "base_alternative": None if base is None else base["name"],
                    "added_candidate_id": row["id"],
                    "actor_witness": deepcopy(row["actor_witness"]),
                },
            }
        )
    return result
