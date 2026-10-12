"""Release a high-confidence event a gate held and the model did not refuse.

The three reasons this may clear are the ones the 2026-09-26 audit measured:
``tracking_arc_abstained``, ``hard_camera_failure``, and
``systematic_repeated_visual_frames``. A row the model itself abstained on
stays abstained. Camera holds clear only when this frame is inside the
reliable per-frame camera support. Repeated-frame holds clear only when the
native cadence audit shows short duplicate groups and this frame is not one
of the repeated pictures. Anything else, including a close-up rehold, stays.
"""

from __future__ import annotations

import math
import os

POLICY_KEY = "held_release_20260926"
SCHEMA = "confident_residual_hold_release_v1"
FLOOR = 0.99
PHYSICAL = frozenset({"contact", "bounce", "net_hit"})
TRACKING = "tracking_arc_abstained"
CAMERA = "hard_camera_failure"
REPEATED = "systematic_repeated_visual_frames"
RELEASABLE = frozenset({TRACKING, CAMERA, REPEATED})
# Same bound the cadence audit uses for a broadcast pulldown, not a stuck picture.
MAX_EQUIVALENCE_GROUP = 5
#: Default off. ``on``: a confident row on a repeated picture of a short-hold cadence is
#: released at the first exposure of its picture group (the picture it actually shows),
#: instead of staying held. cv/experiments/contact_gap/REPORT.md.
DUPLICATE_SNAP_SWITCH = "TENNIS_HELD_RELEASE_DUPLICATE_SNAP"


def duplicate_snap(cadence: dict | None, frame: float) -> int | None:
    """The first exposure of the repeated picture at ``frame``, when that one is releasable."""

    if os.environ.get(DUPLICATE_SNAP_SWITCH, "off") != "on" or not isinstance(cadence, dict):
        return None
    index = int(round(float(frame)))
    for group in cadence.get("equivalence_groups") or []:
        members = [int(item) for item in group]
        if index in members and members[0] != index:
            first = members[0]
            return first if cadence_allows(cadence, first)[0] else None
    return None


def _marginal(row: dict) -> float | None:
    value = row.get("acceptance_marginal")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return score


def _same_clip(row: dict, match_id: str, clip: str) -> bool:
    if row.get("match_id") != match_id:
        return False
    name = str(row.get("clip", ""))
    return name == clip or name.endswith("__" + clip)


def _covers(spans, frame: float) -> bool:
    return any(float(low) <= frame <= float(high) for low, high in spans or [])


def cadence_allows(cadence: dict | None, frame: float) -> tuple[bool, str]:
    """True when ``frame`` is a unique exposure on a diagnosed short-hold cadence."""

    if not isinstance(cadence, dict):
        return False, "no_cadence_audit"
    if cadence.get("decision") != "timing_hold":
        return False, "cadence_not_a_timing_hold"
    if REPEATED not in (cadence.get("reasons") or []):
        return False, "cadence_reason_missing"
    groups = cadence.get("equivalence_groups") or []
    if not groups:
        return False, "no_equivalence_groups"
    if any(len(group) > MAX_EQUIVALENCE_GROUP for group in groups):
        return False, "stuck_frame_group"
    duplicates = {int(item) for item in cadence.get("duplicate_frames") or []}
    if int(round(float(frame))) in duplicates:
        return False, "frame_is_a_repeated_picture"
    return True, "unique_exposure_on_short_holds"


def camera_allows(spans, frame: float) -> tuple[bool, str]:
    """True when the local per-frame camera check supports ``frame``."""

    if not spans:
        return False, "no_per_frame_camera_support"
    if _covers(spans, float(frame)):
        return True, "frame_in_supported_span"
    return False, "frame_outside_supported_span"


def _reason_passes(reason: str, frame: float, camera_spans, cadence) -> tuple[bool, str]:
    if reason == TRACKING:
        return True, "high_confidence_tracking_hold"
    if reason == CAMERA:
        return camera_allows(camera_spans, frame)
    if reason == REPEATED:
        return cadence_allows(cadence, frame)
    return False, "not_releasable"


def _snap_row(row: dict, first: int) -> dict:
    """Move a row from a repeated picture to its group's first exposure, kept on record."""

    shift = float(first) - float(row["frame"])
    moved = {**row, "frame": float(first), "duplicate_snap": {"from": row["frame"], "to": first}}
    location = row.get("location")
    if isinstance(location, dict) and location.get("frame_subpixel") is not None:
        moved["location"] = {
            **location,
            "frame_subpixel": float(location["frame_subpixel"]) + shift,
        }
    return moved


def release_confident_holds(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    camera_spans,
    cadence: dict | None,
    release: bool,
) -> tuple[list[dict], dict]:
    """Clear eligible holds. ``release`` False returns the rows unchanged."""

    census = {"schema": SCHEMA, "released": 0, "kept": 0}
    if not release:
        return rows, census
    output = []
    for source in rows:
        reasons = list(source.get("point_gate_failure_reasons") or [])
        marginal = _marginal(source)
        frame = source.get("frame")
        if (
            not _same_clip(source, match_id, clip)
            or source.get("event_type") not in PHYSICAL
            or source.get("model_abstain") is not False
            or source.get("gate_held") is not True
            or source.get("abstain") is not True
            or marginal is None
            or marginal < FLOOR
            or not reasons
            or any(reason not in RELEASABLE for reason in reasons)
            or not isinstance(frame, (int, float))
        ):
            output.append(source)
            continue
        blocked = None
        for reason in reasons:
            ok, why = _reason_passes(reason, float(frame), camera_spans, cadence)
            if not ok:
                blocked = why
                break
        snapped = None
        if blocked == "frame_is_a_repeated_picture":
            first = duplicate_snap(cadence, float(frame))
            if first is not None and all(
                _reason_passes(reason, float(first), camera_spans, cadence)[0] for reason in reasons
            ):
                snapped, blocked = first, None
        if blocked is not None:
            census["kept"] += 1
            output.append(source)
            continue
        if snapped is not None:
            source = _snap_row(source, snapped)
            census["duplicate_snapped"] = census.get("duplicate_snapped", 0) + 1
        output.append(
            {
                **source,
                "abstain": False,
                "gate_held": False,
                "point_gate_verdict": "retain",
                "point_gate_failure_reasons": [],
                "held_release": {
                    "schema": SCHEMA,
                    "floor": FLOOR,
                    "marginal": marginal,
                    "cleared": reasons,
                },
            }
        )
        census["released"] += 1
    return output, census
