"""Optional source-derived physical prefix, with no claimed competitive ending.

This only bounds a shared S6 solve before a subsequent original racket event.
It does not decide whether that event is a return or dead-ball recovery, infer a
serve role, change observations, or use any fitted state.
"""

from copy import deepcopy
import math

SCHEMA = "s6_observed_first_flight_scope_v1"
PHYSICAL = {"contact", "bounce", "net_hit"}


def physical_inventory(emissions: list[dict]) -> list[dict]:
    from cv.experiments.connected_shooting.auto_packet import automatic_physical_event

    records = []
    for index, raw in enumerate(emissions):
        if raw.get("event_type") not in PHYSICAL:
            continue
        admitted = not raw.get("abstain", False)
        held = not admitted and raw.get("model_abstain") is False
        event = automatic_physical_event(raw)
        event.update(
            original_emission_index=index,
            status="predicted" if admitted else "ambiguous" if held else "abstained",
            occurrence_status="predicted" if admitted else "uncertain" if held else "absent",
        )
        records.append(event)
    return sorted(records, key=lambda e: (e["frame"], e["event_type"]))


def qualify(original: list[dict], window, operator: dict, segmentation_source: dict) -> dict:
    from cv.experiments.connected_shooting.observation_scope import (
        _segmentation_source,
        event_digest,
    )

    source = _segmentation_source(segmentation_source)
    if source is None:
        raise ValueError("automatic ledger source required for physical prefix")
    if len(window) != 2 or any(type(f) is not int for f in window) or window[0] >= window[1]:
        raise ValueError("original ordered native window required")
    if operator != {
        "schema": "ball_observation_operator_v1",
        "kind": "nominal_center",
        "exposure_duration_frames": None,
    }:
        raise ValueError("prefix currently supports declared nominal centers only")
    if not original or original != sorted(original, key=lambda e: (e["frame"], e["event_type"])):
        raise ValueError("ordered original physical inventory required")
    for e in original:
        interval = e.get("frame_interval", [])
        if (
            e.get("event_type") not in PHYSICAL
            or len(interval) != 2
            or not all(math.isfinite(float(f)) for f in [e["frame"], *interval])
            or not window[0] <= interval[0] <= e["frame"] <= interval[1] <= window[1]
            or e.get("annotation_origin") != "automatic"
        ):
            raise ValueError("original automatic physical intervals must lie in native window")
    admitted = [e for e in original if e.get("status") == "predicted"]
    if len(admitted) < 3 or [e["event_type"] for e in admitted[:3]] != [
        "contact",
        "bounce",
        "contact",
    ]:
        raise ValueError("original contact-ground-contact first flight required")
    first, ground, barrier = admitted[:3]
    if first["frame_interval"][1] >= ground["frame_interval"][0]:
        raise ValueError("first contact and ground intervals overlap")
    # A model-supported held physical event is an unresolved barrier, not permission
    # to remove it. The initial implementation abstains on such intervening events.
    if any(e.get("status") == "ambiguous" and e["frame"] <= barrier["frame"] for e in original):
        raise ValueError("uncertain physical event intervenes in first-flight prefix")
    if len({(e["frame"], e["event_type"]) for e in admitted}) != len(admitted):
        raise ValueError("duplicate admitted physical events")
    horizon = math.ceil(barrier["frame_interval"][0]) - 1
    if ground["frame_interval"][1] >= horizon:
        raise ValueError("no native rebound horizon before next contact interval")
    modeled = [deepcopy(first), deepcopy(ground)]
    return dict(
        schema=SCHEMA,
        native_window=list(window),
        modeled_native_window=[window[0], horizon],
        observation_horizon=horizon,
        physical_ending=None,
        ending_semantics="unresolved",
        supplied_ground_count=1,
        original_events_sha256=event_digest(original),
        original_event_count=len(original),
        modeled_events_sha256=event_digest(modeled),
        modeled_event_count=2,
        original_next_physical_contact=deepcopy(barrier),
        source="original automatic first-flight events and next contact interval",
        segmentation_assistance=None,
        segmentation_source=source,
        ball_observation_operator=deepcopy(operator),
        first_contact_role="unspecified",
        complete_original_attempt=False,
    )


def prepare(emissions, observations, window, operator, segmentation_source):
    """Return an explicit hold or a source-bound partial scope, never a point ending."""
    original = physical_inventory(emissions)
    result = dict(schema=SCHEMA, supported=False, original_physical_events=original)
    try:
        contract = qualify(original, window, operator, segmentation_source)
        admitted = [e for e in original if e["status"] == "predicted"]
        events = admitted[:2]
        first, ground = events
        visible = [r for r in observations if r.get("status") == "visible"]
        if not any(
            ground["frame_interval"][1] < r["frame"] <= contract["observation_horizon"]
            for r in visible
        ):
            raise ValueError("no observed native rebound before next contact interval")
        if (
            sum(first["frame"] <= r["frame"] <= contract["observation_horizon"] for r in visible)
            < 4
        ):
            raise ValueError("fewer than four native first-flight observations")
        return result | dict(supported=True, contract=contract, modeled_events=events)
    except ValueError as error:
        return result | dict(reason=str(error))


def role_policy(packet: dict, requested: str) -> dict | None:
    """A physical prefix cannot inherit the cohort-wide first-event-is-serve assumption."""
    from cv.experiments.connected_shooting import observation_scope

    attempt = packet["attempts"][0]
    if attempt.get("observation_scope", {}).get("schema") != SCHEMA:
        return None
    observation_scope.validate(attempt)
    return dict(
        requested_role=requested,
        effective_role="unspecified",
        reason="source-derived physical prefix supplies no serve-role evidence",
        separately_qualified_origin_supplied=False,
    )
