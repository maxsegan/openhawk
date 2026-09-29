"""Explicit occurrence support is separate from uncertain event timing.

Labeled research only. No note parsing, fitted inputs, or blanket ambiguous-event
promotion. Reconcile stale packets by insertion; preserve every existing event,
observation and camera. New contact membership requires a versioned inventory.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


def record_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def predicted_membership(row: dict) -> bool:
    """Accepted producer evidence is usable without claiming observed occurrence.

    The input adapter owns model acceptance. This only validates its explicit
    representation; it neither thresholds confidence again nor adds an event.
    Source records and their uncertainty/probability fields remain unchanged.
    """
    if row.get("status") != "predicted":
        return False
    if (
        row.get("annotation_origin") != "automatic"
        or row.get("exact_epoch_observed") is not False
        or row.get("occurrence_status", "predicted") != "predicted"
        or row.get("timing_status", "predicted") != "predicted"
    ):
        raise ValueError("predicted event requires explicit automatic uncertain provenance")
    try:
        low, high = map(float, row["frame_interval"])
        frame = float(row["frame"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("predicted event requires its finite native interval") from error
    if not all(math.isfinite(v) for v in (low, frame, high)) or not low <= frame <= high:
        raise ValueError("predicted event epoch must lie inside its finite native interval")
    return True


def resolved_membership(row: dict, *, default: str | None = "labeled") -> bool:
    """Supplied resolved topology includes model predictions, not human promotion."""
    if row.get("occurrence_status") in ("unsupported", "ambiguous", "absent"):
        return False
    if "occurrence_admission_rule" in row or "representative_epoch_policy" in row:
        rule = timing_admission(row)
        if rule is None:
            raise ValueError("prepared occurrence lost its timing-uncertain source contract")
        return rule == "explicit_source_bound_occurrence"
    return (
        predicted_membership(row)
        or row.get("status", default) == "labeled"
        or timing_admission(row) == "explicit_source_bound_occurrence"
    )


def _prepared_metadata(rule: str) -> dict:
    return {
        "annotation_origin": "agent",
        "exact_epoch_observed": False,
        "occurrence_admission_rule": rule,
        "representative_epoch_policy": "original_representative_not_exact_observation",
    }


def timing_admission(row: dict) -> str | None:
    """Return the explicit admission rule, validating its original interval."""
    if predicted_membership(row):
        return None  # Already supplied model topology; never insert as a reviewed event.
    if "occurrence_status" in row and row["occurrence_status"] not in (
        "supported",
        "unsupported",
        "ambiguous",
        "absent",
    ):
        raise ValueError("unknown occurrence status")
    if row.get("occurrence_status") in ("unsupported", "ambiguous", "absent"):
        return None
    legacy = (
        row.get("event_type") == "bounce"
        and row.get("status") == "ambiguous"
        and row.get("timing_status") == "abstained_exact_epoch"
    )
    explicit = (
        row.get("event_type") in ("contact", "bounce", "net_hit")
        and row.get("status") == "ambiguous"
        and row.get("occurrence_status") == "supported"
        and row.get("timing_status") in ("abstained", "abstained_exact_epoch")
    )
    if not (legacy or explicit):
        return None
    if explicit:
        evidence = row.get("occurrence_evidence", {})
        original = {
            k: v for k, v in row.items() if k not in ("occurrence_status", "occurrence_evidence")
        }
        raw = evidence.get("original_event")
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("original event evidence must be an object")
            expected = deepcopy(raw)
            # A newly explicit timing abstention is review metadata, not a
            # change to the original epoch, interval, type or source note.
            if "timing_status" not in expected:
                expected["timing_status"] = row["timing_status"]
            prepared = {**expected, **_prepared_metadata("explicit_source_bound_occurrence")}
            if original != expected and original != prepared:
                raise ValueError("occurrence qualification changed the original event")
            original = raw
        if (
            not evidence.get("reviewer")
            or not evidence.get("basis")
            or not isinstance(evidence.get("source_label_sha256"), str)
            or len(evidence["source_label_sha256"]) != 64
            or evidence.get("original_event_sha256") != record_hash(original)
        ):
            raise ValueError(
                "occurrence qualification requires matching original event hash and evidence"
            )
    try:
        low, high = map(float, row["frame_interval"])
        frame = float(row["frame"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "timing-uncertain bounce/contact/net requires a finite interval and representative"
        ) from error
    if (
        not all(math.isfinite(x) for x in (low, frame, high))
        or not low <= frame <= high
        or low == high
    ):
        raise ValueError(
            "timing-uncertain bounce/contact/net requires a nonzero interval containing its frame"
        )
    return (
        "explicit_source_bound_occurrence" if explicit else "legacy_bounce_exact_epoch_abstention"
    )


def physical_record(row: dict, rule: str) -> dict:
    if (
        rule
        not in (
            "explicit_source_bound_occurrence",
            "legacy_bounce_exact_epoch_abstention",
        )
        or timing_admission(row) != rule
    ):
        raise ValueError("physical event requires its validated occurrence admission rule")
    record = deepcopy(row)
    if rule == "explicit_source_bound_occurrence":
        evidence = record["occurrence_evidence"]
        if "original_event" not in evidence:
            # Preserve source fields before adding mechanical packet metadata.
            # This also retains an original human/automatic annotation origin.
            evidence["original_event"] = {
                k: deepcopy(v)
                for k, v in row.items()
                if k not in ("occurrence_status", "occurrence_evidence")
            }
    return {
        **record,
        **_prepared_metadata(rule),
    }


def terminal_interval_end(events: list[dict], ending: float, clip: str) -> float:
    """Validation horizon of an explicitly qualified ending-coincident impact.

    The ending representative remains unchanged. Its uncertainty interval can
    contain the physical impact after that representative; this is not a new
    event or permission to cross the next competitive contact.
    """
    endings = [
        r
        for r in events
        if r.get("clip", clip) == clip
        and r["event_type"] == "ending"
        and float(r["frame"]) == ending
    ]
    if len(endings) != 1:
        return ending
    end_low, end_high = map(float, endings[0].get("frame_interval", [ending, ending]))
    if (
        not math.isfinite(end_low)
        or not math.isfinite(end_high)
        or not end_low <= ending <= end_high
    ):
        raise ValueError("invalid source ending interval")
    matches = [
        r
        for r in events
        if r.get("clip", clip) == clip
        and r["event_type"] == "bounce"
        and float(r["frame"]) == ending
        and timing_admission(r)
    ]
    if len(matches) != 1:
        return ending
    low, high = map(float, matches[0]["frame_interval"])
    if not end_low <= low <= ending <= high <= end_high:
        raise ValueError("terminal bounce interval exceeds its source ending interval")
    return high


def reconcile_packet(packet: dict, labels: dict) -> tuple[dict, dict]:
    """Insert qualified missing events, never overwrite an existing packet edit."""
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one original packet attempt required")
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    source = [r for r in labels["events"]["records"] if r.get("clip", clip) == clip]
    qualified = [(r, timing_admission(r)) for r in source]
    qualified = [(r, rule) for r, rule in qualified if rule]
    receipt = {
        "schema": "labeled_event_occurrence_reconciliation_v1",
        "status": "unchanged",
        "source_events_sha256": record_hash(source),
        "clip": clip,
        "insertions": [],
        "observations_changed": False,
        "cameras_changed": False,
        "native_timestamps_changed": False,
        "existing_packet_events_overwritten": False,
        "contact_epochs_fitted_freely": False,
    }
    if not qualified:
        return packet, receipt
    ids = [r["id"] for r in source if r.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate original event identifiers")
    original_events = attempt["events"]
    events = deepcopy(original_events)
    agent_rows = deepcopy(attempt.get("agent_event_rows", []))
    start, ending = float(attempt["first_event_frame"]), float(attempt["owner_end_frame"])
    interval_end = terminal_interval_end(source, ending, clip)
    for row, rule in qualified:
        frame = float(row["frame"])
        terminal_match = row["event_type"] == "bounce" and frame == ending and interval_end > ending
        if row["event_type"] == "net_hit" and frame == ending:
            terminal_match = any(
                e["event_type"] == "ending"
                and float(e["frame"]) == ending
                and "net" in e.get("ending_kind", "")
                and e.get("frame_interval") == row.get("frame_interval")
                for e in source
            )
        if not (start < frame < ending or (start < frame and terminal_match)):
            raise ValueError(
                "qualified uncertain event must lie inside existing attempt boundaries"
            )
        if row["event_type"] == "net_hit":
            low, high = map(float, row["frame_interval"])
            contacts = sorted(float(r["frame"]) for r in events if r["event_type"] == "contact")
            net_horizon = high if terminal_match else ending
            bounds = [*contacts, net_horizon]
            if sum(a < low and high <= b for a, b in zip(bounds, bounds[1:])) != 1:
                raise ValueError("timing-uncertain net interval must belong to one flight")
        same_id = [r for r in events if row.get("id") and r.get("id") == row["id"]]
        matches = [
            r for r in events if r["event_type"] == row["event_type"] and float(r["frame"]) == frame
        ]
        if len(matches) > 1 or (same_id and same_id != matches):
            raise ValueError("qualified source conflicts with existing packet event identity")
        if matches:
            if list(matches[0].get("frame_interval", [])) != list(row["frame_interval"]):
                raise ValueError("qualified source conflicts with existing packet interval")
            continue
        low, high = map(float, row["frame_interval"])
        if row["event_type"] == "contact":
            if not start < low <= high < ending:
                raise ValueError("uncertain event interval crosses an attempt boundary")
            for other in events:
                a, b = map(float, other.get("frame_interval", [other["frame"], other["frame"]]))
                if not (high < a or low > b):
                    raise ValueError(
                        "uncertain contact bracket overlaps an existing physical transition"
                    )
            # Existing observed side metadata is a cross-check, never invented.
            contacts = sorted(
                [r for r in agent_rows if r["event_type"] == "contact"], key=lambda r: r["frame"]
            )
            before = [r for r in contacts if float(r["frame"]) < frame]
            after = [r for r in contacts if float(r["frame"]) > frame]
            side = row.get("hitter_end")
            if side in ("near", "far") and any(
                r.get("hitter_end") == side for r in before[-1:] + after[:1]
            ):
                raise ValueError("qualified contact contradicts neighboring observed hitter ends")
        events.append(physical_record(row, rule))
        existing_rows = [
            r
            for r in agent_rows
            if (row.get("id") and r.get("id") == row["id"])
            or (r["event_type"] == row["event_type"] and float(r["frame"]) == frame)
        ]
        if existing_rows and existing_rows != [row]:
            raise ValueError("qualified source conflicts with existing agent event metadata")
        if not existing_rows:
            agent_rows.append(deepcopy(row))
        receipt["insertions"].append(
            {
                "original_event": deepcopy(row),
                "event_sha256": record_hash(row),
                "admission_rule": rule,
            }
        )
    if not receipt["insertions"]:
        return packet, receipt
    events.sort(key=lambda r: float(r["frame"]))
    pairs = [(r["event_type"], float(r["frame"])) for r in events]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate prepared physical event")
    contacts = [float(r["frame"]) for r in events if r["event_type"] == "contact"]
    if contacts != sorted(set(contacts)):
        raise ValueError("contact epochs must be unique and ordered")
    bounds = [*contacts, ending]
    interval_bounds = [*contacts, interval_end]
    for row, _ in qualified:
        if row["event_type"] == "bounce":
            low, high = map(float, row["frame_interval"])
            if (
                sum(a < low and high <= b for a, b in zip(interval_bounds, interval_bounds[1:]))
                != 1
            ):
                raise ValueError("timing-uncertain bounce interval must belong to one flight")
    counts = [
        sum(r["event_type"] == "bounce" and a < float(r["frame"]) <= b for r in events)
        for a, b in zip(bounds, bounds[1:])
    ]
    updated = deepcopy(attempt)
    updated.update(
        events=events,
        agent_event_rows=sorted(agent_rows, key=lambda r: float(r["frame"])),
        contact_count=len(contacts),
        terminal_bounce_count=counts[-1],
        structurally_ground_replayable=all(n == 1 for n in counts),
    )
    mutable = {
        "events",
        "agent_event_rows",
        "contact_count",
        "terminal_bounce_count",
        "structurally_ground_replayable",
    }
    if {k: v for k, v in updated.items() if k not in mutable} != {
        k: v for k, v in attempt.items() if k not in mutable
    }:
        raise ValueError("event reconciliation altered unrelated attempt metadata")
    receipt.update(
        status="inserted_original_occurrences",
        before_contact_count=attempt["contact_count"],
        after_contact_count=len(contacts),
        bounce_counts_by_flight=counts,
        terminal_interval_end=interval_end,
        source_ending_representative_unchanged=ending,
        declared_flight_inventory_revision_required=len(contacts) != attempt["contact_count"],
    )
    result = deepcopy(packet)
    result["attempts"][0] = updated
    result["event_occurrence_reconciliation"] = receipt
    return result, receipt
