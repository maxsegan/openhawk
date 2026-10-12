"""Source-bound optional bounce scopes, before numerical scene qualification.

An empty accepted ground tail remains an explicit null hold. Candidate ground
impacts never supply a physical point ending or change the native horizon.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from cv.pipeline import provenance

SCHEMA = "s6_optional_bounce_scope_v1"
WITNESS_SCHEMA = "s6_optional_bounce_scope_witness_v1"
MODES = ("classifier", "classifier_v2", "classifier_interior")
KNOWN_ENDING_ELIGIBILITY = "known_physical_ending_interior_v2"


def _wire(rows) -> str:
    # Native guide metadata contains tuple XY; JSON persists the same values as
    # arrays. Compare wire identity without rewriting any observation field.
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(record: dict) -> list | dict:
    from cv.pipeline.s6_labeled_stage import resolve

    path = resolve(record)
    if provenance.file_record(path) != record:
        raise ValueError("optional bounce source binding changed")
    return json.loads(path.read_text())


def _qualify(
    events,
    observations,
    window,
    segmentation_source,
    allow_terminal_net,
    allow_observed_horizon=False,
):
    from cv.experiments.connected_shooting import observation_scope

    try:
        scope = observation_scope.qualify(
            events,
            window,
            segmentation_source=segmentation_source,
            allow_terminal_net=allow_terminal_net,
            allow_observed_horizon=allow_observed_horizon,
        )
        # One shared support rule: a rebound row after a supplied ground, an
        # observed flight after the contact when the ground count is unknown.
        if not observation_scope.supported_tail(events, observations, scope):
            raise ValueError("observed ground scope requires visible native rebound support")
        return scope, None
    except ValueError as error:
        return None, str(error)


def build_witness(
    *,
    emissions: list[dict],
    events: list[dict],
    observations: list[dict],
    window: tuple[int, int],
    match_id: str,
    clip: str,
    source_events: dict,
    segmentation_source: dict,
    mode: str = "classifier",
    allow_terminal_net: bool = False,
    allow_observed_horizon: bool = False,
    physical_ending: dict | None = None,
    absent_membership: dict | None = None,
) -> dict:
    from cv.experiments.connected_shooting import auto_packet
    from cv.pipeline import s6_optional_bounces

    if mode not in MODES:
        raise ValueError("only explicit classifier optional bounce evidence is enabled")
    absent = []
    if absent_membership is not None:
        # The shared resolver's validated authorization, and nothing else, opens
        # this branch: exactly one original automatic predicted terminal net,
        # bound by the hash of its whole record. It is validated before any
        # inventory is built, so an independent caller cannot reach a partially
        # rebuilt witness with an arbitrary event.
        from cv.pipeline import s6_terminal_net_membership as membership

        absent = membership.authorized_events(absent_membership)
    if [r["frame"] for r in observations] != list(range(window[0], window[1] + 1)):
        raise ValueError("complete original native observation inventory required")
    original, ending = auto_packet.automatic_event_inventory(
        emissions,
        match_id,
        clip,
        window,
        observation_scope=True,
        require_originating_contact=True,
    )
    scoped = ending.pop("unqualified_observation_scope", False)
    if absent:
        # The original emissions, clip, window and mode are unchanged; only the
        # accepted partition and the declared ending move to the branch this
        # hypothesis owns, so a held ground blocked solely by that ending
        # requalifies under its own original probability and interval.
        if scoped or physical_ending is not None:
            raise ValueError("absent membership witness declares no physical ending")
        if float(ending["frame"]) != float(absent[0]["frame"]):
            raise ValueError("absent membership must remove the original ending-coincident event")
        retained = [e for e in original if e != absent[0]]
        if len(original) - len(retained) != 1 or events != retained:
            raise ValueError("optional bounce original accepted events changed")
        scoped = True
    elif events != original:
        raise ValueError("optional bounce original accepted events changed")
    if not absent and (
        (scoped and physical_ending is not None) or (not scoped and physical_ending != ending)
    ):
        raise ValueError("optional bounce original physical ending changed")
    end = float(window[1]) if scoped else ending["frame"]
    # Preserve producer clip aliases. The helper's exact clip match is applied
    # separately to each declared alias, without rewriting any emission.
    aliases = {e.get("clip") for e in emissions}
    clip_key = f"{match_id}__{clip}" if f"{match_id}__{clip}" in aliases else clip
    if any(e.get("clip") not in {clip, f"{match_id}__{clip}"} for e in emissions):
        raise ValueError("optional bounce witness requires one original source selection")
    if len(aliases) > 1:
        raise ValueError("mixed producer clip aliases require explicit common source normalization")
    inventory = s6_optional_bounces.source_candidates(
        events,
        emissions,
        clip_key,
        window,
        end_frame=end,
        mode=mode,
    )
    unchanged, null_hold = (None, None)
    alternatives = []
    interior = not scoped and mode in s6_optional_bounces.INTERIOR_MODES
    if interior:
        # A terminal net has no accepted bounce to occupy its final gap. Do not
        # accidentally treat that gap as an interior contact-to-contact interval.
        last_contact = max(e["frame"] for e in events if e["event_type"] == "contact")
        retained = []
        for candidate in inventory["candidates"]:
            if inventory["gaps"][candidate["gap_index"]][1] <= last_contact:
                retained.append(candidate)
            else:
                inventory["held_candidates"].append(
                    {
                        "source_emission_index": candidate["source_emission_index"],
                        "reason": "known_ending_alternatives_require_two_bounding_contacts",
                    }
                )
        inventory["candidates"] = retained
        # The ending is known and unchanged, so there is no rebound tail to
        # qualify: these alternatives only add interior bounces between two
        # accepted contacts. `_qualify` stays the unknown-ending check and the
        # scope container below still refuses a known ending.
        alternatives = [
            {**h, "scope": None, "eligibility": KNOWN_ENDING_ELIGIBILITY}
            for h in s6_optional_bounces.hypotheses(inventory, events, mode)
        ]
    elif scoped:
        # The unchanged source events may themselves qualify now: under the
        # explicit policy an unresolved visible tail is an unknown ground count.
        # Every original hypothesis is still qualified from the same witnesses.
        unchanged, null_hold = _qualify(
            events,
            observations,
            window,
            segmentation_source,
            allow_terminal_net,
            allow_observed_horizon,
        )
        for h in s6_optional_bounces.hypotheses(inventory, events, mode):
            scope, hold = _qualify(
                h["events"],
                observations,
                window,
                segmentation_source,
                allow_terminal_net,
                allow_observed_horizon,
            )
            alternatives.append({**h, "scope": scope, "hold": hold})
    return {
        "schema": WITNESS_SCHEMA,
        "mode": mode,
        "match_id": match_id,
        "clip": clip,
        "native_window": list(window),
        "source_events": deepcopy(source_events),
        "original_events": deepcopy(events),
        "original_emissions": deepcopy(emissions),
        "original_observations": deepcopy(observations),
        "segmentation_source": deepcopy(segmentation_source),
        "allow_terminal_net": allow_terminal_net,
        # Absent unless an explicit membership hypothesis declares one, so every
        # existing witness document stays byte-identical.
        **(
            {
                "absent_membership_events": deepcopy(absent),
                "absent_membership_ending": deepcopy(ending),
                "absent_membership_authorization_sha256": hashlib.sha256(
                    json.dumps(absent_membership, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            if absent
            else {}
        ),
        # Absent unless enabled, so every existing witness document, and every
        # old report that binds one by hash, stays byte-identical.
        **({"allow_observed_horizon": True} if allow_observed_horizon else {}),
        "observation_horizon": float(window[1]),
        "physical_ending": deepcopy(physical_ending),
        "inventory": inventory,
        "hypotheses": alternatives,
        "unchanged_scope": unchanged,
        "null_hold": null_hold,
        "not_applicable": None if scoped or interior else "known_physical_ending_scope_not_enabled",
        "runtime_model_calls": 0,
        "ground_evidence_files_read": False,
    }


def eligible_hypotheses(document: dict) -> list[dict]:
    """The one eligibility rule both ordinary readers use.

    An unknown-ending alternative is eligible once its rebound scope qualified.
    A known-ending interior alternative is eligible on its explicit tag; that
    tag is only honoured in an interior mode with a known ending, so
    no row can borrow unknown-ending scope semantics it was not built with.
    """
    from cv.pipeline import s6_optional_bounces

    interior = (
        document.get("physical_ending") is not None
        and document.get("mode") in s6_optional_bounces.INTERIOR_MODES
    )
    rows = []
    for row in document["hypotheses"]:
        tagged = row.get("eligibility") == KNOWN_ENDING_ELIGIBILITY
        if tagged and (not interior or row.get("scope") is not None):
            raise ValueError("known-ending interior eligibility outside its source witness mode")
        if row.get("scope") is not None or tagged:
            rows.append(row)
    return rows


def _owner_inventory_matches(
    attempt: dict, document: dict, original: list[dict], *, ending_frame: float | None = None
) -> bool:
    """Admit existing explicit native aftermath transport, never a new ending.

    ``ending_frame`` is the *original* declared ending, so a branch that declares
    that ending absent still transports exactly the rows the source authorized.
    """
    owner = attempt["owner_ball_labels"]
    if _wire(owner) == _wire(original):
        return True
    ending = document["physical_ending"] if ending_frame is None else {"frame": ending_frame}
    extension = attempt.get("modeled_window_extension")
    if ending is None or not isinstance(extension, dict):
        return False
    start, end = extension.get("from_frame"), extension.get("to_frame")
    if (
        not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start != ending["frame"]
        or not start < end <= document["native_window"][1]
        or extension.get("source") != "frozen label document; rows copied verbatim"
    ):
        return False
    wanted = list(range(math.floor(start) + 1, math.floor(end) + 1))
    if extension.get("native_frames_added") != wanted:
        return False
    source = {r["frame"]: r for r in document["original_observations"]}
    if any(frame not in source for frame in wanted):
        return False
    expected = [*original, *(source[frame] for frame in wanted)]
    return _wire(owner) == _wire(expected)


def load_witness(attempt: dict) -> dict:
    record = attempt.get("optional_bounce_witness")
    if not isinstance(record, dict):
        raise ValueError("bound optional bounce witness required")
    document = _read(record)
    if document.get("schema") != WITNESS_SCHEMA:
        raise ValueError("unsupported optional bounce witness")
    emissions = _read(document["source_events"])
    expected = build_witness(
        emissions=emissions,
        events=document["original_events"],
        observations=document["original_observations"],
        window=tuple(document["native_window"]),
        match_id=document["match_id"],
        clip=document["clip"],
        source_events=document["source_events"],
        segmentation_source=document["segmentation_source"],
        mode=document["mode"],
        allow_terminal_net=document["allow_terminal_net"],
        allow_observed_horizon=document.get("allow_observed_horizon", False),
        physical_ending=document["physical_ending"],
    )
    if document != expected:
        raise ValueError("optional bounce inventory or hypotheses differ from original source")
    # The declared ending stays the original one for every binding check below,
    # even when an explicit membership hypothesis solves its absent branch.
    declared_ending = document["physical_ending"]
    ending_frame = declared_ending["frame"] if declared_ending else document["observation_horizon"]
    from cv.pipeline import s6_terminal_net_membership as membership

    document = membership.bounce_witness(attempt, document)
    if attempt["events"] != document["original_events"]:
        raise ValueError("optional bounce attempt replaced original accepted events")
    original = [
        r
        for r in document["original_observations"]
        if attempt["first_event_frame"] <= r["frame"] <= attempt["owner_end_frame"]
    ]
    if _wire(attempt["ball_observations"]) != _wire(original) or not _owner_inventory_matches(
        attempt,
        document,
        original,
        ending_frame=declared_ending["frame"] if declared_ending else None,
    ):
        raise ValueError("optional bounce original native observations changed")
    if attempt["owner_end_frame"] != ending_frame:
        raise ValueError("optional bounce original ending or horizon changed")
    return document


def scope_container(witness_record: dict, document: dict) -> dict:
    if document["physical_ending"] is not None or document["null_hold"] is None:
        raise ValueError("optional scope container requires an unchanged unknown-ending hold")
    qualified = [h for h in document["hypotheses"] if h["scope"] is not None]
    if not qualified:
        raise ValueError("no source-supported qualified optional bounce scope")
    return {
        "schema": SCHEMA,
        "native_window": document["native_window"],
        "observation_horizon": document["observation_horizon"],
        "physical_ending": None,
        "ending_semantics": "unresolved",
        "segmentation_source": document["segmentation_source"],
        "optional_bounce_witness": deepcopy(witness_record),
        "null_hold": document["null_hold"],
        "qualified_hypotheses": [{"name": h["name"], "scope": h["scope"]} for h in qualified],
    }


def validate_report_events(attempt: dict, report: dict) -> bool:
    selection = report.get("optional_contact_selection")
    if not selection or selection.get("policy") != "source_witness_optional_bounce_v1":
        return False
    if selection.get("witness_record") != attempt.get("optional_bounce_witness"):
        raise ValueError("optional bounce selected witness differs from packet")
    document = load_witness(attempt)
    eligible = eligible_hypotheses(document)
    if document["null_hold"] is None:
        eligible = [{"name": "supplied", "events": attempt["events"]}, *eligible]
    if not any(
        h["name"] == selection.get("selected_topology") and h["events"] == report.get("events")
        for h in eligible
    ):
        raise ValueError("selected events are not a qualified source bounce hypothesis")
    return True


def validate(attempt: dict, labels: dict | None = None, report: dict | None = None) -> dict:
    document = load_witness(attempt)
    contract = attempt.get("observation_scope")
    if contract != scope_container(attempt["optional_bounce_witness"], document):
        raise ValueError("optional bounce scope differs from its source hypotheses")
    if attempt.get("ending_supplied") is not False or attempt.get("point_end") is not None:
        raise ValueError("optional bounce scope cannot supply physical point death")
    if attempt.get("owner_end_frame_semantics") != "observation_horizon_not_physical_event":
        raise ValueError("optional bounce scope must preserve unknown-ending semantics")
    if attempt.get("segmentation_source") != document["segmentation_source"]:
        raise ValueError("optional bounce segmentation source changed")
    if labels is not None:
        meta = labels["attempt"]
        if (
            meta.get("observation_scope") != contract
            or "ending_frame" in meta
            or meta.get("ending_kind") != "unresolved"
            or meta.get("segmentation_source") != document["segmentation_source"]
        ):
            raise ValueError("optional bounce consumer scope changed")
        source = [{k: v for k, v in e.items() if k != "clip"} for e in labels["events"]["records"]]
        if source != attempt["events"] or _wire(labels["ball"]["records"][0]["frames"]) != _wire(
            document["original_observations"]
        ):
            raise ValueError("optional bounce consumer original inventory changed")
    if report is not None and (
        report.get("configuration", {}).get("observation_scope") != contract
        or not validate_report_events(attempt, report)
    ):
        raise ValueError("optional bounce report does not preserve its source scope")
    return contract
