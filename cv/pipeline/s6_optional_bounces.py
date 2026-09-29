"""Source-only optional bounce proposals, enabled only by explicit shared policy.

Retain uncertain occurrence separately from the producer's timing interval.
Classifier and grouped ground-motion support are fixed engineering ablations,
not independent calibrated probabilities. No fitted states or scoring inputs.

The explicit ``classifier_interior`` mode keeps the v1 source inventory and
ranking; its caller enables existing interior scope and composition.

The explicit ``classifier_v2`` mode holds a row only when the source prefers no
event at all, and carries a competing impact type into the occurrence score
instead of waiving the row. It changes which source rows are offered, never the
producer's event type and never its timing: the epoch and the one-frame search
interval stay exactly what the producer emitted, so a source row whose interval
does not reach the real event remains a source-timing limitation of its own.
"""

from __future__ import annotations

from copy import deepcopy
import math

from cv.experiments.connected_shooting.auto_packet import automatic_physical_event
from cv.pipeline.s6_optional_contacts import excluded_phase

SCHEMA = "s6_optional_bounce_candidates_v1"
SCHEMA_V2 = "s6_optional_bounce_candidates_v2"
MODE_V2 = "classifier_v2"
MODE_INTERIOR = "classifier_interior"
# These modes share scope/composition, but only v2 changes candidate admission.
INTERIOR_MODES = (MODE_V2, MODE_INTERIOR)
MODES = {"classifier", "classifier_ground", *INTERIOR_MODES}
SCHEMAS = {SCHEMA, SCHEMA_V2}
MAX_ALTERNATIVES = 2


def held_bounce_row(row: dict, clip: str) -> bool:
    """This arm's own v1 held-ground rule, exposed for ground-encounter reads.

    No candidate is built and no timing is produced here; other scopes only ask
    whether the source itself held a ground encounter on a row.
    """
    if row.get("clip") != clip or row.get("event_type") != "bounce" or excluded_phase(row):
        return False
    if not (row.get("abstain") or row.get("gate_held")):
        return False
    probability = row.get("class_probabilities") or {}
    values = [probability.get(k) for k in ("none", "contact", "bounce", "net_hit")]
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values
    ):
        return False
    return probability["bounce"] > 0 and probability["bounce"] >= max(
        probability[k] for k in ("none", "contact", "net_hit")
    )


def inventory_schema(mode: str) -> str:
    """v2 inventories are their own schema; v1 documents stay byte-identical."""
    if mode not in MODES:
        raise ValueError("explicit supported bounce candidate mode required")
    return SCHEMA_V2 if mode == MODE_V2 else SCHEMA


def source_candidates(
    events: list[dict],
    emissions: list[dict],
    clip: str,
    window: tuple[int, int],
    *,
    end_frame: float,
    ground_evidence: list[dict] = (),
    mode: str = "classifier",
) -> dict:
    """Inventory held source bounces inside original contact/ending boundaries.

    At most one additional bounce per currently bounce-free contact interval.
    A terminal observation horizon bounds scope; it never becomes a point ending.
    Competing crop proposals remain alternatives, not multiple impacts.
    """
    schema = inventory_schema(mode)
    competition = mode == MODE_V2
    low, high = window
    if not all(math.isfinite(v) for v in (low, high, end_frame)) or not low < end_frame <= high:
        raise ValueError("ordered original window and bounded ending/horizon required")
    contacts = sorted((e for e in events if e["event_type"] == "contact"), key=lambda e: e["frame"])
    boundaries = [e["frame"] for e in contacts] + [end_frame]
    gaps = []
    for left, right in zip(boundaries, boundaries[1:]):
        if low <= left < right <= high and not any(
            e["event_type"] == "bounce" and left <= e["frame"] <= right for e in events
        ):
            gaps.append([left, right])
    ground = {}
    for row in ground_evidence:
        if row.get("event_type") == "bounce" and row.get("clip") == clip:
            key = row["candidate_frame"]
            group = ground.setdefault(key, [])
            if row not in group:
                group.append(row)
    candidates, held = [], []
    for index, row in enumerate(emissions):
        if row.get("clip") != clip or row.get("event_type") != "bounce":
            continue
        reason = None
        if not (row.get("abstain") or row.get("gate_held")):
            continue
        if excluded_phase(row):
            reason = "excluded_source_phase"
        probability = row.get("class_probabilities") or {}
        values = [probability.get(k) for k in ("none", "contact", "bounce", "net_hit")]
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        ):
            reason = reason or "missing_or_invalid_source_probabilities"
        elif probability["bounce"] <= 0 or (
            # v2 holds only when the source prefers no event at all. A competing
            # impact type is carried into the occurrence term below instead of
            # waiving the row; the producer's own event type is never rewritten.
            probability["bounce"] <= probability["none"]
            if competition
            else probability["bounce"] < max(probability[k] for k in ("none", "contact", "net_hit"))
        ):
            reason = reason or "source_prefers_none_or_other_impact"
        if reason:
            held.append({"source_emission_index": index, "reason": reason})
            continue
        try:
            event = automatic_physical_event(row, prediction_window=window)
        except (ValueError, TypeError, KeyError) as error:
            held.append(
                {"source_emission_index": index, "reason": f"invalid_native_interval: {error}"}
            )
            continue
        a, b = event["frame_interval"]
        owners = [i for i, (left, right) in enumerate(gaps) if left < a <= b < right]
        overlap = any(
            e["event_type"] == "contact"
            and max(a, e.get("frame_interval", [e["frame"], e["frame"]])[0])
            <= min(b, e.get("frame_interval", [e["frame"], e["frame"]])[1])
            for e in contacts
        )
        if len(owners) != 1 or overlap:
            held.append(
                {
                    "source_emission_index": index,
                    "reason": "no_empty_contact_gap_or_contact_interval_overlap",
                }
            )
            continue
        event.update(note="optional original classifier bounce", occurrence_status="optional")
        witnesses = [
            witness
            for witness in ground.get(row.get("candidate_frame"), [])
            if witness.get("original_classifier_probabilities") == probability
            and witness.get("original_model_epoch") == event["frame"]
        ]
        # Support only directly aligned model/emission epochs. Any future
        # decoder timing transport needs its own explicit producer binding.
        # The caller must bind artifact/source ancestry; this checks prediction
        # identity only, not video or producer identity.
        witness = witnesses[0] if len(witnesses) == 1 else None
        valid_ground = bool(
            witness is not None
            and witness.get("original_classifier_probabilities") == probability
            and witness.get("supported")
        )
        contribution = float(witness.get("prior_nats", 0)) if valid_ground else 0.0
        if not math.isfinite(contribution) or not 0 <= contribution <= 1:
            raise ValueError("source ground contribution must retain its declared zero-to-one cap")
        candidates.append(
            {
                "id": f"emission_{index}",
                "source_emission_index": index,
                "source_emission": deepcopy(row),
                "gap_index": owners[0],
                "event": event,
                "classifier_log_odds": max(
                    -1.0,
                    min(
                        1.0,
                        math.log(
                            probability["bounce"]
                            / max(
                                sum(probability[k] for k in ("none", "contact", "net_hit"))
                                if competition
                                else probability["none"],
                                1e-9,
                            )
                        ),
                    ),
                ),
                "ground_prior_nats": contribution,
                "ground_evidence": deepcopy(witness),
                "ground_prediction_identity_supported": valid_ground,
                # v2 keeps the producer's decoded type and the raw pre-grammar
                # head preference as separate recorded facts. The consumer never
                # re-types the event; the categorical is an engineering ablation
                # score, not a calibrated probability of occurrence.
                **(
                    {
                        "decoder_event_type": row.get("event_type"),
                        # Exact ties resolve in the fixed declared class order,
                        # which never prefers bounce.
                        "classifier_argmax": max(
                            ("none", "contact", "bounce", "net_hit"),
                            key=lambda k: probability[k],
                        ),
                        "class_probabilities": {
                            k: float(probability[k])
                            for k in ("none", "contact", "bounce", "net_hit")
                        },
                        "type_competition_carried_in_occurrence": True,
                    }
                    if competition
                    else {}
                ),
            }
        )
    return {
        "schema": schema,
        "clip": clip,
        "source_attempt_events": deepcopy(events),
        "original_window": list(window),
        "end_frame": end_frame,
        "gaps": gaps,
        "candidates": candidates,
        "held_candidates": held,
        "default_enabled": False,
        "timing_distribution_invented": False,
        **(
            {
                "candidate_mode": mode,
                "type_competition": "decoder_type_retained_source_categorical_in_occurrence",
                "calibrated_probability_claimed": False,
            }
            if competition
            else {}
        ),
    }


def validate_interior_policy_binding(document: dict, mode: str) -> None:
    """Do not declare changed v2 admission as the conservative interior mode.

    Historical policy/witness handling is unchanged outside this additive mode.
    Witness loading separately regenerates inventory and hypotheses from source.
    """
    actual = document.get("mode")
    if MODE_INTERIOR in (mode, actual) and actual != mode:
        raise ValueError("optional bounce witness mode differs from shared policy")


def hypotheses(document: dict, events: list[dict], mode: str) -> list[dict]:
    """Return two bounded alternatives; caller retains the no-addition branch.

    Use the strongest source candidate per gap, then omit the weakest gap in
    the second alternative. With only one gap, use its two strongest timing
    candidates instead. This is not exhaustive omission/combination coverage;
    all original candidate evidence remains in the inventory.
    """
    if (
        mode not in MODES
        or document.get("schema") != inventory_schema(mode)
        or document["source_attempt_events"] != events
        or (document.get("candidate_mode") == MODE_V2) != (mode == MODE_V2)
    ):
        raise ValueError("supported mode and unchanged source events required")
    groups = {}
    for row in document["candidates"]:
        prior = row["classifier_log_odds"] + (
            row["ground_prior_nats"] if mode == "classifier_ground" else 0
        )
        groups.setdefault(row["gap_index"], []).append((row, prior))
    groups = {
        key: sorted(rows, key=lambda v: (-v[1], v[0]["event"]["frame"], v[0]["id"]))
        for key, rows in groups.items()
    }
    if not groups:
        return []
    if len(groups) == 1:
        beam = [((row,), prior) for row, prior in next(iter(groups.values()))[:MAX_ALTERNATIVES]]
    else:
        strongest = [groups[key][0] for key in sorted(groups)]
        weakest = min(range(len(strongest)), key=lambda i: (strongest[i][1], strongest[i][0]["id"]))
        variants = [strongest, [row for i, row in enumerate(strongest) if i != weakest]]
        beam = [
            (tuple(row for row, _ in variant), sum(prior for _, prior in variant))
            for variant in variants
        ]
    result = []
    for rows, prior in beam:
        if not rows:
            continue
        added = []
        for row in rows:
            event = deepcopy(row["event"])
            event.update(
                occurrence_status="predicted",
                optional_topology_membership={
                    "candidate_id": row["id"],
                    "conditioned_on_this_hypothesis": True,
                    "accepted_source_stream_changed": False,
                },
            )
            added.append(event)
        result.append(
            {
                "name": f"optional_bounces_{len(result) + 1}",
                "events": sorted([*deepcopy(events), *added], key=lambda e: e["frame"]),
                "added": added,
                "candidate_ids": [r["id"] for r in rows],
                "occurrence_log_odds": prior,
                "evidence_mode": mode,
                # Flight velocity/spin and shared contact-state dimensions depend
                # on racket contacts, not supplied ground-event membership. Ground
                # epochs are outcomes of the existing physical integration; these
                # proposals add constraints/occurrence hypotheses, not variables.
                "added_parameter_count": 0,
            }
        )
        if len(result) == MAX_ALTERNATIVES:
            break
    return result
