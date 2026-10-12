"""Explicit source-only contact proposals entering the existing bounded S6 search.

Cue support is retained, never multiplied as independent probabilities. This first
scope offers contact alternatives strictly between original contacts; other kinds
and leading/trailing windows stay in the receipt with explicit exclusions.
"""

from copy import deepcopy
import math

from cv.pipeline import event_paths, provenance

KEY = "independent_event_proposals"
SCHEMA = "s6_independent_event_proposals_v1"


def policy(document: dict) -> str:
    declaration = document.get(KEY)
    if declaration is None:
        return "off"
    if not isinstance(declaration, dict) or declaration.get("policy") not in {"on", "off"}:
        raise ValueError("explicit independent event proposal policy required")
    return declaration["policy"]


def prepare(inputs, attempt: dict, images: list[dict], cameras: dict) -> dict:
    from cv.pipeline import s6_contact_composition as composition
    from cv.pipeline import s6_optional_contacts as optional
    from cv.pipeline import s6_witnessed_interior_contacts as interior

    path = inputs.independent_event_candidates
    if path is None:
        raise ValueError("independent event proposals require an explicit bound sidecar")
    clip = f"{inputs.match_id}__{inputs.clip}"
    packet = event_paths.load_evidence_packet(path, inputs.events, clip)
    clock = {int(row["frame"]): float(row["native_pts_seconds"]) for row in images}
    optional._native_clock(clock)
    supported = {
        int(row["frame"]) for row in cameras["cameras"] if row.get("status") == "supported"
    }
    points, observation_receipt = interior.observed_points(
        attempt.get("owner_ball_labels"), supported
    )
    cuts = set()
    if getattr(inputs, "optional_contact_views", None) is not None:
        cuts, unsupported, _ = optional.pose_view_barriers(inputs.optional_contact_views, clock)
        supported -= unsupported
    events = attempt["events"]
    contacts = sorted((e for e in events if e["event_type"] == "contact"), key=lambda e: e["frame"])
    inventory = deepcopy(packet.get("proposal_windows", []))
    admitted, excluded = [], []
    for index, row in enumerate(inventory):
        reason = None
        a, frame, b = float(row["start_frame"]), float(row["frame"]), float(row["end_frame"])
        interval = [a, b]
        if "contact" not in row["kinds"]:
            reason = "no_contact_compatibility_in_current_scope"
        elif not all(math.isfinite(v) for v in (a, frame, b)) or not a <= frame <= b:
            raise ValueError("ordered finite independent proposal timing required")
        pair = next(
            (
                (left, right)
                for left, right in zip(contacts, contacts[1:])
                if optional._interval(left)[1] < a <= frame <= b < optional._interval(right)[0]
            ),
            None,
        )
        if reason is None and pair is None:
            reason = "outside_original_contact_to_contact_scope"
        if reason is None and any(
            optional._interval(e)[0] <= b and a <= optional._interval(e)[1] for e in events
        ):
            reason = "overlaps_original_event_interval"
        subgaps = []
        if reason is None:
            left, right = pair
            low, high = optional._interval(left)[1], optional._interval(right)[0]
            span = range(math.floor(low), math.ceil(high) + 1)
            if any(f not in clock or f not in supported for f in span) or any(
                f in cuts for f in span
            ):
                reason = "unsupported_original_camera_clock_or_declared_view_in_subgap"
            grounds = [
                float(e["frame"])
                for e in events
                if e["event_type"] == "bounce" and low < e["frame"] < high
            ]
            subgaps = [
                dict(
                    interval=[start, end],
                    observations=sum(start < f < end for f in points),
                    supplied_bounces=sum(start < f < end for f in grounds),
                )
                for start, end in ((low, a), (b, high))
            ]
            if reason is None and any(
                g["observations"] < composition.MIN_OBSERVATIONS for g in subgaps
            ):
                reason = "new_subgap_below_original_observation_floor"
            if reason is None and (
                any(g["supplied_bounces"] > 1 for g in subgaps)
                or sum(g["supplied_bounces"] for g in subgaps) != len(grounds)
            ):
                reason = "original_bounce_inventory_not_preserved_by_contact_split"
        if reason is not None:
            excluded.append(
                dict(source_index=index, reason=reason, frame_interval=interval, subgaps=subgaps)
            )
            continue
        event = dict(
            event_type="contact",
            frame=frame,
            frame_interval=interval,
            status="predicted",
            annotation_origin="automatic",
            occurrence_status="optional",
            interval_origin="independent_source_proposal_window",
            exact_epoch_observed=False,
            timing_status="predicted",
            note="Independent source proposal; physical occurrence unconfirmed",
        )
        admitted.append(
            dict(
                id=f"independent_proposal_{index}",
                source_index=index,
                source=row["source"],
                event=event,
                subgaps=subgaps,
                source_proposal=deepcopy(row),
            )
        )
    return dict(
        schema=SCHEMA,
        policy="on",
        source_sidecar=provenance.file_record(path),
        source_attempt_events=deepcopy(events),
        inventory=inventory,
        candidates=admitted,
        exclusions=excluded,
        observations=observation_receipt,
        occurrence_log_odds=0.0,
        support_is_independent_probability=False,
        ranking="native epoch then source then original index; no fitted or label evidence",
        scope="original_contact_to_contact_only",
        native_timestamps_changed=False,
    )


def extend_hypotheses(document: dict, events: list[dict], existing: list[dict]) -> list[dict]:
    if policy(document) == "off":
        return existing
    from cv.pipeline.s6_optional_contacts import MAX_ADDED_HYPOTHESES

    declaration = document[KEY]
    if declaration.get("schema") != SCHEMA or declaration.get("source_attempt_events") != events:
        raise ValueError("independent proposal witness differs from original source events")
    rows = sorted(
        declaration["candidates"],
        key=lambda r: (r["event"]["frame"], r["source"], r["source_index"]),
    )
    result = list(existing)
    remaining = max(0, MAX_ADDED_HYPOTHESES - len(existing))
    grouped = {}
    for row in rows:
        key = (row["event"]["frame"], *row["event"]["frame_interval"])
        grouped.setdefault(key, []).append(row)
    for group in list(grouped.values())[:remaining]:
        row = group[0]
        event = deepcopy(row["event"])
        result.append(
            dict(
                name=row["id"],
                events=sorted([*deepcopy(events), event], key=lambda e: e["frame"]),
                added=[event],
                candidate_ids=[row["id"]],
                occurrence_log_odds=0.0,
                independent_event_proposal=dict(
                    source_sidecar=declaration["source_sidecar"],
                    source_index=row["source_index"],
                    source_proposal=deepcopy(row["source_proposal"]),
                    correlated_support=[deepcopy(r["source_proposal"]) for r in group],
                    support_is_independent_probability=False,
                ),
            )
        )
    return result


RESEARCH_INPUT_KEY = "source_proposal_research_input"
RESEARCH_INPUT_SCHEMA = "s6_source_proposal_research_input_v1"


def bind_research_packet(original_path, witness_path) -> dict:
    """Bind an explicit assisted-input ablation without promoting it to automatic mode.

    The complete original packet is retained byte-for-byte as a source artifact;
    only the two witness bindings are added to its copied document. Channels may
    be labeled, while proposal generation remains independently source-only.
    """
    import json

    packet = json.loads(original_path.read_text())
    if (
        packet.get("human_derived") is not True
        or packet.get("automatic_inference_eligible") is not False
    ):
        raise ValueError("research proposal handoff requires explicit retained assistance")
    if len(packet.get("attempts", [])) != 1 or RESEARCH_INPUT_KEY in packet:
        raise ValueError("one original research attempt required")
    if "optional_contact_witness" in packet or "optional_contact_witness" in packet["attempts"][0]:
        raise ValueError("research proposal handoff cannot replace an existing witness")
    witness = json.loads(witness_path.read_text())
    if witness["source_attempt_events"] != packet["attempts"][0]["events"]:
        raise ValueError("research witness must preserve all original firm events")
    binding = provenance.file_record(witness_path)
    packet["optional_contact_witness"] = binding
    packet["attempts"][0]["optional_contact_witness"] = binding
    packet[RESEARCH_INPUT_KEY] = dict(
        schema=RESEARCH_INPUT_SCHEMA,
        original_packet=provenance.file_record(original_path),
        assistance=packet.get("assistance"),
        automatic_inference_eligible=False,
        source_proposal_generation_used_reference_labels=False,
    )
    return packet


def validate_research_packet(packet: dict, witness: dict, resolve) -> None:
    """Verify exact frozen input preservation for the opt-in research handoff."""
    import json

    declaration = packet.get(RESEARCH_INPUT_KEY, {})
    if declaration.get("schema") != RESEARCH_INPUT_SCHEMA:
        raise ValueError("optional proposals on assisted inputs require explicit research handoff")
    original_path = resolve(declaration["original_packet"])
    if provenance.file_sha256(original_path) != declaration["original_packet"]["sha256"]:
        raise ValueError("original research packet changed")
    original = json.loads(original_path.read_text())
    rebuilt = bind_research_packet(original_path, resolve(packet["optional_contact_witness"]))
    if packet != rebuilt or witness["source_attempt_events"] != original["attempts"][0]["events"]:
        raise ValueError("research proposal handoff changed original input channels")
    if policy(witness) == "on":
        if witness[KEY]["source_attempt_events"] != original["attempts"][0]["events"]:
            raise ValueError("independent research proposal witness changed original events")
