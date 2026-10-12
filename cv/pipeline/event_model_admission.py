"""Optional admission of model-accepted events whose only veto is tracking-arc quality.

Disabled unless an event consumer explicitly asks for it.  The decoder already
accepted these rows at its own calibrated threshold; the emission gate then held
them because the ball-tracking arc covering the emitted frame did not pass the
tracking gate's own fit-quality checks.  This policy admits such a row as an
*uncertain observation* and records why, instead of erasing it.

What it does not do.  No threshold moves, no probability is recalibrated, no epoch
or pixel is touched, no event is invented or reclassified, and nothing here reads a
label, a picture or a reviewed choice.  An admitted row is not independent evidence:
the same ball track and crops fed the decoder that emitted it and the arc gate that
held it.  It certifies neither a physical ending nor a competitive flight, so the
shared ending rule treats it as identity-only.

Every other veto still applies.  Model abstention, a point the validity gate holds,
a frame outside the retained play span or without reliable camera support, invalid
native timing, a missing or ambiguous arc, and arcs held for coverage or candidate
support -- an actual observation gap rather than fit quality -- all remain refusals.
"""

from __future__ import annotations

import math

import numpy as np

from cv.pipeline.event_impulse_support import ALLOWED_REASONS, model_identity_admission

SCHEMA = "model_accepted_tracking_held_admission_v1"
IDENTITY_STATUS = "admitted_model_event_without_arc_support"
HOLD_REASON = "tracking_arc_abstained"
ARC_GATE_DECISION = "model_accepted_admission_only"
CONFIG = {
    "policy": "model_accepted_tracking_held_v1",
    "event_types": ["contact", "bounce"],
    "required_point_gate_verdict": "retain",
    "required_point_gate_failure_reasons": [HOLD_REASON],
    "identity_admission": "original_decoder_decision",
    # The arc failure reasons this policy is willing to treat as fit quality.  The
    # tracking gate's coverage and candidate-support failures stay vetoes.
    "tracking_quality_failure_reasons": sorted(ALLOWED_REASONS),
    "threshold_changed": False,
    "epochs_changed": False,
    "track_frames_restored": False,
    "independent_of_video_and_track_signals": False,
    "certifies_physical_ending": False,
    "certifies_competitive_flight": False,
}


def _refused(reason: str, **extra) -> dict:
    return {
        "schema": SCHEMA,
        "policy": CONFIG["policy"],
        "admitted": False,
        "reason": reason,
        **extra,
    }


def admission(event: dict, arcs: list[dict]) -> dict:
    """Return this row's admission evidence, admitted or refused, with its reason."""

    if (
        event.get("model_abstain") is not False
        or event.get("gate_held") is not True
        or event.get("point_gate_verdict") != CONFIG["required_point_gate_verdict"]
        or event.get("point_gate_failure_reasons") != CONFIG["required_point_gate_failure_reasons"]
        or event.get("event_type") not in CONFIG["event_types"]
    ):
        return _refused("not_exclusively_tracking_held_physical_event")
    identity = model_identity_admission(event)
    if identity is None:
        return _refused("original_decoder_admission_missing_or_inconsistent")
    frame = float(event["frame"])
    location = event.get("location") or {}
    fps = float(location.get("fps", 0))
    probability = float(event.get("probability", 0))
    pixel = np.asarray([location.get("image_x"), location.get("image_y")], dtype=float)
    if (
        not math.isfinite(frame)
        or not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
        or location.get("image_coordinate_space") != "native_1920x1080"
        or not np.isfinite(pixel).all()
    ):
        return _refused("missing_native_event_evidence")
    gate = event.get("tracking_arc_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("available") is not True
        or gate.get("decision") != "hold"
    ):
        return _refused("tracking_arc_gate_unavailable_or_not_held")
    matching = [arc for arc in arcs if arc.get("arc_id") in (gate.get("matching_arc_ids") or [])]
    if len(matching) != 1 or len(gate.get("matching_arc_ids") or []) != 1:
        # No arc at all, or more than one, is a different barrier: there is then no
        # single fit whose quality the veto can be attributed to.
        return _refused("missing_or_overlapping_arc")
    arc = matching[0]
    reasons = arc.get("failure_reasons") or []
    if arc.get("decision") != "hold" or not reasons or not set(reasons) <= ALLOWED_REASONS:
        return _refused("arc_held_for_more_than_fit_quality", arc_failure_reasons=sorted(reasons))
    return {
        "schema": SCHEMA,
        "policy": CONFIG["policy"],
        "admitted": True,
        "evidence_status": "model_accepted_tracking_held_observation",
        "identity": identity,
        "tracking_hold": {
            "retained": False,
            "arc_id": arc.get("arc_id"),
            "regime": arc.get("regime"),
            "decision": arc.get("decision"),
            "failure_reasons": sorted(reasons),
            "start_frame": arc.get("start_frame"),
            "end_frame": arc.get("end_frame"),
            "coverage_rate": arc.get("coverage_rate"),
            "candidate_support_rate": arc.get("candidate_support_rate"),
        },
        "observed_impulse_support": False,
        "independent_of_video_and_track_signals": False,
        "certifies_physical_ending": False,
        "certifies_competitive_flight": False,
    }


def admit_model_accepted_events(rows: list[dict], tracking_gate: dict) -> list[dict]:
    """Admit the exclusively tracking-held model-accepted rows of ``rows``.

    Every physical row carries its own admission evidence, admitted or refused, so
    the counts are auditable from the emissions alone.  An admitted row keeps its
    original abstention, hold and arc-gate decisions under ``pre_admission_*`` and
    declares its degraded status through ``event_identity_support``; the shared
    ending rule reads that block and refuses to certify an ending from it.
    """

    if not isinstance(tracking_gate, dict):
        raise ValueError("model-accepted admission requires the original tracking arc gate")
    arcs = {f"{r['match_id']}__{r['clip']}": r.get("arcs", []) for r in tracking_gate["rows"]}
    output = []
    for source in rows:
        if source.get("event_type") == "point_end":
            raise ValueError("model-accepted admission runs before the ending rule")
        prefix = f"{source['match_id']}__"
        clip = source["clip"]
        key = clip if clip.startswith(prefix) else prefix + clip
        evidence = admission(source, arcs.get(key, []))
        row = {**source, "model_accepted_admission": evidence}
        if evidence["admitted"]:
            row.update(
                pre_admission_abstain=source["abstain"],
                pre_admission_gate_held=source["gate_held"],
                pre_admission_point_gate_failure_reasons=list(source["point_gate_failure_reasons"]),
                pre_admission_tracking_arc_gate=dict(source["tracking_arc_gate"]),
                abstain=False,
                gate_held=False,
                point_gate_failure_reasons=[],
            )
            row["tracking_arc_gate"] = {
                **source["tracking_arc_gate"],
                "decision": ARC_GATE_DECISION,
                "original_decision": source["tracking_arc_gate"]["decision"],
                "track_frames_restored": False,
            }
            row["event_identity_support"] = {
                "status": IDENTITY_STATUS,
                "original_model_abstain": False,
                "localization_status": "model_emission_only",
                "evidence_status": "tracking_arc_held",
                "certifies_physical_ending": False,
            }
        output.append(row)
    return output


def admission_record(rows: list[dict], *, enabled: bool) -> dict:
    """Summarize what the policy did, for a run manifest."""

    evidence = [row.get("model_accepted_admission") for row in rows]
    refusals: dict[str, int] = {}
    for item in evidence:
        if item is not None and not item["admitted"]:
            refusals[item["reason"]] = refusals.get(item["reason"], 0) + 1
    return {
        "schema": SCHEMA,
        "enabled": enabled,
        "configuration": CONFIG if enabled else None,
        "rows_considered": sum(item is not None for item in evidence),
        "admitted_events": sum(bool(item and item["admitted"]) for item in evidence),
        "refusals": dict(sorted(refusals.items())),
        "scope": "event claims only; track frames, endings and terminal evidence are not restored",
    }


__all__ = [
    "ARC_GATE_DECISION",
    "CONFIG",
    "IDENTITY_STATUS",
    "SCHEMA",
    "admission",
    "admission_record",
    "admit_model_accepted_events",
]
