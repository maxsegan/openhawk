"""Apply settled cascade decisions to an automatic S6 packet before the second fit.

A confirm makes the emitted row a supported claim and names the tier and
model that settled it. A reject removes the row. An unsettled decision is not
an edit. A bound contact prefix is re-qualified from the edited inventory with
the production qualifier, so the fitter replays it; an unresolved inventory is
edited in place with its boundaries recomputed. The observations document is
the packet's automatic consumer document and gets the same edits; it is not
read as an answer.

Optional PROPOSE/RETIME edits (``event_cascade_propose``, runner switch ``--propose``)
travel as ``proposals``: a ``retime`` replaces an automatic contact with the same row at
the model's frame, an ``insert`` adds a new supported automatic contact. Without them
nothing is inserted, as in FREEZE v2.

Ported from the cascade-rerun branch (``event_cascade.apply_receipts``,
``build_dev_wave._rebind/_write_cascade``, ``fresh_wave.settling``).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from cv.pipeline import provenance
from cv.pipeline import s6_contact_prefix_scope as prefix
from cv.pipeline import s6_first_contact_role as role
from cv.pipeline.event_cascade import FITTER_SOURCE, PHYSICAL, TIER_BY_NAME
from cv.pipeline.s6_contact_prefix_runtime import labels_for_scope

CONFIRM_NOTE = "cascade confirm of a retained abstention"
EDITABLE_SCOPES = ("s6_observed_contact_prefix_v1", "s6_unresolved_ending_input_v1")


def event_key(event: dict) -> tuple[str, float]:
    return (event["event_type"], round(float(event["frame"]), 5))


def settling(decisions: list[dict]) -> dict[tuple[str, float], dict]:
    """Confirms and rejects of one attempt. An unsettled row does not edit the packet."""
    found = {}
    for row in decisions:
        if row.get("action") == "unsettled":
            continue
        if row.get("action") not in ("confirm", "reject"):
            raise ValueError(f"decision action {row.get('action')}")
        key = event_key(row)
        if key in found:
            raise ValueError(f"two settled decisions for {key}")
        found[key] = row
    return found


def require_automatic(packet: dict) -> None:
    """The cascade edits automatic events and reads the automatic ball track only."""
    origins = packet.get("stream_origins") or {}
    if origins.get("events") != "automatic" or origins.get("ball") != "automatic":
        raise ValueError(f"cascade needs automatic ball and events, got {origins}")


def confirm_event(event: dict, decision: dict) -> dict:
    """The same epoch, now a claim. The receipt names the settling tier and model.

    A fitter-adjudicated confirm (no model asked) carries tier 0, source
    ``fitter_adjudicated`` and no model; S6 treats it exactly as a paid confirm.
    """
    if decision.get("action") != "confirm" or not decision.get("tier"):
        raise ValueError("confirm needs a settling tier")
    if role.supported(event):
        raise ValueError("a supported row is not a cascade confirm")
    if decision["tier"] == FITTER_SOURCE:
        number, source, model = 0, FITTER_SOURCE, None
    else:
        tier = TIER_BY_NAME[decision["tier"]]
        number, source, model = tier.number, tier.name, tier.model
    clip = event.get("clip")
    row = {
        key: value for key, value in event.items() if key not in ("automatic_abstention", "clip")
    }
    row["status"] = "predicted"
    row["occurrence_status"] = "predicted"
    row["note"] = CONFIRM_NOTE
    row["cascade_tier"] = number
    row["cascade_source"] = source
    row["cascade_model"] = model
    if clip is not None:
        row["clip"] = clip
    if not role.supported(row):
        raise ValueError("confirmed row is still unsupported")
    return row


def apply_events(
    events: list[dict],
    decisions: dict[tuple[str, float], dict],
    proposals: list[dict] | None = None,
) -> list[dict]:
    """A reject drops the row, a confirm supports it. A key that matches nothing raises.

    ``proposals`` (``propose_edits``): a ``retime`` replaces the row at its ``key`` with its
    ``event``; an ``insert`` adds its ``event``, carrying the inventory's ``clip`` if the rows
    have one. Rows stay in frame order.
    """
    retimes = {tuple(p["key"]): p for p in proposals or [] if p["action"] == "retime"}
    inserts = [p for p in proposals or [] if p["action"] == "insert"]
    if set(retimes) & set(decisions):
        raise ValueError("a retimed row also has a settled decision")
    used: set[tuple[str, float]] = set()
    kept = []
    for event in events:
        if event.get("event_type") not in PHYSICAL:
            kept.append(deepcopy(event))
            continue
        key = event_key(event)
        if key in retimes:
            if key in used:
                raise ValueError(f"two events share {key}")
            used.add(key)
            row = deepcopy(retimes[key]["event"])
            if event.get("clip") is not None:
                row["clip"] = event["clip"]
            kept.append(row)
            continue
        decision = decisions.get(key)
        if decision is None:
            kept.append(deepcopy(event))
            continue
        if key in used:
            raise ValueError(f"two events share {key}")
        used.add(key)
        if decision["action"] == "reject":
            continue
        if decision["action"] == "confirm":
            kept.append(confirm_event(event, decision))
            continue
        kept.append(deepcopy(event))
    missing = (set(decisions) | set(retimes)) - used
    if missing:
        raise ValueError(f"decision matched no event: {sorted(missing)[:3]}")
    if not inserts and not retimes:
        return kept
    clip = next((e["clip"] for e in events if e.get("clip") is not None), None)
    physical = [row for row in kept if row.get("event_type") in PHYSICAL]
    for proposal in inserts:
        row = deepcopy(proposal["event"])
        if clip is not None:
            row["clip"] = clip
        physical.append(row)
    physical.sort(key=lambda row: float(row["frame"]))
    # Other rows keep their places; physical rows fill theirs in frame order, extras
    # right after the last physical slot.
    slots = [i for i, row in enumerate(kept) if row.get("event_type") in PHYSICAL]
    last = slots[-1] if slots else len(kept) - 1
    ordered = iter(physical)
    out = []
    for index, row in enumerate(kept):
        out.append(next(ordered) if row.get("event_type") in PHYSICAL else row)
        if index == last:
            out.extend(ordered)
    out.extend(ordered)
    return out


def _declared_boundaries(events: list[dict], intervals: list[list[float]]) -> list[list[float]]:
    """Boundaries that are not an event's own interval. Those are rebuilt."""
    pool = [
        tuple(float(value) for value in event["frame_interval"])
        for event in events
        if event.get("event_type") in PHYSICAL
    ]
    declared = []
    for interval in intervals:
        key = (float(interval[0]), float(interval[1]))
        hit = next(
            (
                i
                for i, item in enumerate(pool)
                if abs(item[0] - key[0]) < 1e-6 and abs(item[1] - key[1]) < 1e-6
            ),
            None,
        )
        if hit is None:
            declared.append([key[0], key[1]])
            continue
        pool.pop(hit)
    return declared


def _refresh_scope(scope: dict, events: list[dict], previous: list[dict]) -> dict:
    if scope.get("schema") != prefix.UNRESOLVED_INPUT_SCHEMA:
        raise ValueError(f"unresolved inventory required, got {scope.get('schema')}")
    declared = _declared_boundaries(previous, scope.get("unresolved_boundary_intervals") or [])
    refreshed = deepcopy(scope)
    refreshed["original_event_observations"] = deepcopy(events)
    refreshed["unresolved_boundary_intervals"] = prefix.unresolved_boundaries(
        events, boundaries=declared
    )
    if "leading_physical_prefix" in refreshed:
        # The declaration names the rows before the first contact. An inserted or
        # retimed contact can change or empty that prefix; re-declare it from the
        # edited inventory, as the preparer would have, instead of contradicting it.
        if role.leading_physical_prefix(events):
            refreshed["leading_physical_prefix"] = prefix.leading_prefix_declaration(
                events, refreshed["leading_physical_prefix"]["source_refusal"]
            )
        else:
            refreshed.pop("leading_physical_prefix")
    return refreshed


def _as_unresolved(attempt: dict, events: list[dict], scope: dict) -> dict:
    """A bound prefix whose edited inventory will not bind, as a retained inventory."""
    bound = deepcopy(attempt)
    bound["events"] = events
    bound["observation_scope"] = scope
    bound.pop("original_observation_scope", None)
    bound["owner_end_frame"] = scope["observation_horizon"]
    bound["owner_end_frame_semantics"] = "observation_horizon_not_physical_event"
    bound["ending_supplied"] = False
    bound["point_end"] = None
    clip = attempt.get("point_clip")
    if clip is not None:
        bound["agent_event_rows"] = [{**event, "clip": clip} for event in events]
    bound["contact_count"] = sum(event["event_type"] == "contact" for event in events)
    prefix.validate_unresolved_input(bound)
    return bound


def rebind_attempt(
    attempt: dict, decisions: dict, cameras: dict, labels: dict, proposals: list | None = None
) -> dict:
    """A bound contact prefix, re-qualified from the edited original inventory."""
    if not decisions and not proposals:
        return deepcopy(attempt)
    scope = attempt.get("observation_scope") or {}
    if scope.get("schema") != prefix.SCHEMA:
        raise ValueError(f"bound contact prefix required, got {scope.get('schema')}")
    previous = attempt["original_physical_events"]
    original = apply_events(previous, decisions, proposals)
    if not original:
        raise ValueError("cascade removed every physical event")
    unresolved = _refresh_scope(attempt["original_observation_scope"], original, previous)
    source = dict(
        attempt,
        events=original,
        owner_end_frame=scope["original_owner_end_frame"],
        owner_end_frame_semantics=scope["original_owner_end_frame_semantics"],
        observation_scope=unresolved,
        point_end=scope.get("original_point_end"),
    )
    window = scope["native_window"] if scope.get("native_window_origin") == "passed" else None
    plan = prefix.qualify(
        source,
        cameras,
        observation_partition=scope["observation_partition"],
        observation_fallback=scope["observation_fallback"],
        native_window=window,
        mode=scope["mode"],
        labels=labels,
    )
    if plan.get("status") != "qualified":
        # The production preparer keeps the unresolved inventory when the prefix will not bind.
        return _as_unresolved(attempt, original, unresolved)
    bound = prefix.bind_attempt(source, plan)
    clip = attempt["point_clip"]
    bound["agent_event_rows"] = [{**event, "clip": clip} for event in original]
    bound["contact_count"] = sum(event["event_type"] == "contact" for event in original)
    prefix.validate(
        bound, cameras, observation_partition=scope["observation_partition"], labels=labels
    )
    return bound


def apply_unresolved(attempt: dict, decisions: dict, proposals: list | None = None) -> dict:
    """Edit a retained unresolved inventory in place. No prefix is rebuilt."""
    if not decisions and not proposals:
        return deepcopy(attempt)
    scope = attempt.get("observation_scope") or {}
    if scope.get("schema") != prefix.UNRESOLVED_INPUT_SCHEMA:
        raise ValueError(f"unresolved inventory required, got {scope.get('schema')}")
    previous = attempt["events"]
    events = apply_events(previous, decisions, proposals)
    if not events:
        raise ValueError("cascade removed every physical event")
    bound = deepcopy(attempt)
    bound["events"] = events
    bound["observation_scope"] = _refresh_scope(scope, events, previous)
    clip = attempt.get("point_clip")
    if clip is not None and "agent_event_rows" in bound:
        bound["agent_event_rows"] = [{**event, "clip": clip} for event in events]
    bound["contact_count"] = sum(event["event_type"] == "contact" for event in events)
    prefix.validate_unresolved_input(bound)
    return bound


def rewrite_labels(labels: dict, decisions: dict, proposals: list | None = None) -> dict:
    """The consumer document, same edits as the packet."""
    document = deepcopy(labels)
    document["events"]["records"] = apply_events(
        document["events"]["records"], decisions, proposals
    )
    return document


def rebind(
    packet: dict, labels: dict, cameras: dict, decisions: dict, proposals: list | None = None
) -> tuple[dict, dict]:
    """Edited packet and consumer document for the second fit."""
    from cv.experiments.connected_shooting import observation_scope

    require_automatic(packet)
    source_attempt = packet["attempts"][0]
    schema = (source_attempt.get("observation_scope") or {}).get("schema")
    if schema == prefix.SCHEMA:
        bound = rebind_attempt(source_attempt, decisions, cameras, labels, proposals)
    elif schema == prefix.UNRESOLVED_INPUT_SCHEMA:
        bound = apply_unresolved(source_attempt, decisions, proposals)
    else:
        raise ValueError(f"cascade cannot edit scope {schema!r}")
    new_labels = rewrite_labels(labels, decisions, proposals)
    meta = new_labels["attempt"]
    meta["observation_scope"] = bound["observation_scope"]
    meta["ending_kind"] = "unresolved"
    meta.pop("ending_frame", None)
    observation_scope.validate(bound, labels_for_scope(new_labels, {"attempts": [bound]}))
    new_packet = dict(packet)
    new_packet["attempts"] = [bound]
    return new_packet, new_labels


def _identity(record: dict) -> tuple:
    return tuple(record.get(key) for key in ("path_base", "path", "sha256"))


def automatic_producer(
    row: dict, folder: Path, decisions: dict, proposals: list | None = None
) -> Path:
    """Provenance of an edited automatic packet: the original producer plus the cascade.

    The automatic S6 boundary accepts only a packet and consumer document its producer
    binds. The edit is recorded as a new automatic producer whose parent is the original
    one: same source video, the original artifacts except the replaced packet and
    document, the edited pair, the settled decisions (model, tokens, dollars per call)
    and the cascade models. No human or reviewed input is added.
    """
    from cv.pipeline.s6_input_origin import derived_producer
    from cv.pipeline.s6_labeled_stage import resolve

    settled = sorted(decisions.values(), key=lambda d: (float(d["frame"]), d["event_type"]))
    decisions_path = folder / "cascade_decisions.json"
    edits = [p["decision"] for p in proposals or []]
    document = {"schema": "event_cascade_attempt_edits_v1", "decisions": settled}
    if edits:
        document["proposals"] = edits
    decisions_path.write_text(json.dumps(document, indent=2) + "\n")
    tier_names = {tier.number: tier.name for tier in TIER_BY_NAME.values()}
    tiers = sorted(
        {
            (receipt["tier"], receipt["model"])
            for d in settled
            for receipt in d.get("receipts") or []
        }
    )
    proposers = sorted(
        {(receipt["tier"], receipt["model"]) for d in edits for receipt in d.get("receipts") or []}
    )
    return derived_producer(
        row,
        folder / "automatic_provenance.json",
        [
            provenance.file_record(
                folder / "observations.json", role="automatic_observation_document"
            ),
            provenance.file_record(folder / "packet.json", role="automatic_observation_packet"),
            provenance.file_record(decisions_path, role="event_cascade_decisions"),
        ],
        {
            "producer": (
                "fitter_adjudication_rebind"
                if settled and all(d.get("tier") == FITTER_SOURCE for d in settled) and not edits
                else "event_cascade_freeze_v2_rebind"
            ),
            "confirms": sum(d["action"] == "confirm" for d in settled),
            "rejects": sum(d["action"] == "reject" for d in settled),
            **(
                {
                    "retimes": sum(d["action"] == "retime" for d in edits),
                    "inserts": sum(d["action"] == "insert" for d in edits),
                }
                if edits
                else {}
            ),
        },
        resolve,
        models=[
            {"name": model, "role": f"event_cascade_{tier_names.get(tier, tier)}"}
            for tier, model in tiers
        ]
        + [{"name": model, "role": f"event_cascade_propose_{tier}"} for tier, model in proposers],
    )


def write_manifest(
    manifest: dict,
    packet: dict,
    labels: dict,
    folder: Path,
    decisions: dict | None = None,
    proposals: list | None = None,
) -> Path:
    """Write the edited packet, its consumer document and a manifest that binds them.

    An automatic row also gets a new producer provenance (``automatic_producer``) that
    binds the edited pair, so the automatic S6 boundary accepts it; ``decisions`` are the
    attempt's settled decisions and ``proposals`` its PROPOSE/RETIME edits.
    """
    folder.mkdir(parents=True, exist_ok=True)
    labels_path = folder / "observations.json"
    packet_path = folder / "packet.json"
    manifest_path = folder / "manifest.json"
    labels_path.write_text(json.dumps(labels))
    packet = dict(packet)
    packet["external_label_binding"] = {
        "record": provenance.file_record(labels_path),
        "semantics": packet.get("external_label_binding", {}).get(
            "semantics", "automatic consumer document; legacy interface name only"
        ),
    }
    packet_path.write_text(json.dumps(packet))
    document = json.loads(json.dumps(manifest))
    row = document["rows"][0]
    row["labels"] = provenance.file_record(labels_path)
    row["packet"] = provenance.file_record(packet_path)
    if row.get("observation_origin") == "automatic":
        if not decisions and not proposals:
            raise ValueError("an edited automatic packet needs its settled decisions")
        original = json.loads(json.dumps(manifest))["rows"][0]
        row["automatic_provenance"] = provenance.file_record(
            automatic_producer(original, folder, decisions or {}, proposals)
        )
    manifest_path.write_text(json.dumps(document, indent=2) + "\n")
    return manifest_path
