"""Carry an already source-validated search topology into in-memory refinement.

The loader owns file/witness validation. Refinement consumes this receipt without
opening sources again or rewriting original packet/label events.
"""

from copy import deepcopy

SCHEMA = "s6_selected_source_topology_v1"


def _source_bindings(selection: dict) -> dict:
    return {
        name: deepcopy(selection[name])
        for name in ("witness_record", "witness_records", "source_bindings")
        if name in selection
    }


def admit_selected_events(attempt: dict, search: dict) -> dict | None:
    """Validate through the existing optional-family reader, at the loader boundary."""
    if search.get("optional_contact_selection") is None:
        return None
    from cv.pipeline import s6_optional_event_union
    from cv.pipeline import s6_terminal_net_membership as net_membership

    selection = search["optional_contact_selection"]
    # Refinement continues on the branch scope the search actually selected.
    attempt = net_membership.replay_attempt(attempt, selection)
    if selection.get("policy") == net_membership.SELECTION_POLICY:
        if search["events"] != attempt["events"]:
            raise ValueError("selected refinement topology lacks source-witness admission")
    elif not s6_optional_event_union.validate_report_events(attempt, search):
        raise ValueError("selected refinement topology lacks source-witness admission")
    return dict(
        schema=SCHEMA,
        selected_topology=search["optional_contact_selection"]["selected_topology"],
        source_policy=search["optional_contact_selection"]["policy"],
        source_bindings=_source_bindings(search["optional_contact_selection"]),
        original_events=deepcopy(attempt["events"]),
        selected_events=deepcopy(search["events"]),
        original_inputs_changed=False,
        physical_ending_inferred=False,
    )


def require_selected_contacts(bundle: dict) -> dict:
    """Permit added contacts only under an unchanged validated source transaction."""
    from cv.pipeline import s6_terminal_net_membership as net_membership

    receipt = bundle.get("selected_event_admission")
    search = bundle["search"]
    selection = search.get("optional_contact_selection") or {}
    attempt = net_membership.replay_attempt(bundle["packet"]["attempts"][0], selection)
    if (
        not receipt
        or receipt.get("schema") != SCHEMA
        or receipt.get("original_inputs_changed") is not False
        or receipt.get("physical_ending_inferred") is not False
        or receipt.get("original_events") != attempt["events"]
        or receipt.get("selected_events") != search["events"]
        or receipt.get("selected_topology") != selection.get("selected_topology")
        or receipt.get("source_policy") != selection.get("policy")
        or receipt.get("source_bindings") != _source_bindings(selection)
    ):
        raise ValueError("changed contact topology requires matching source admission")
    original = sorted(
        (e for e in attempt["events"] if e["event_type"] == "contact"),
        key=lambda event: float(event["frame"]),
    )
    selected = sorted(
        (e for e in search["events"] if e["event_type"] == "contact"),
        key=lambda event: float(event["frame"]),
    )
    if not original or not selected or selected[0] != original[0]:
        raise ValueError("selected topology must preserve the original first contact")
    if any(sum(e == row for row in selected) != 1 for e in original):
        raise ValueError("selected topology must preserve every original contact")
    # Existing first-contact profiling records a fitted epoch separately from
    # source membership. It may move within the original interval, but it may
    # not replace a contact, its timing support or its role.
    active = deepcopy(bundle["context"]["events"])
    first = min(
        (e for e in active if e["event_type"] == "contact"),
        key=lambda event: float(event["frame"]),
    )
    if first.get("profiled_within_original_interval") is True:
        low, high = first["frame_interval"]
        if not float(low) <= float(first["frame"]) <= float(high):
            raise ValueError("profiled first contact escaped its original interval")
        if "original_representative_frame" not in first:
            raise ValueError("profiled first contact lacks its original representative")
        first["frame"] = first.pop("original_representative_frame")
        first.pop("profiled_within_original_interval")
    if active != search["events"]:
        raise ValueError("refinement context differs from its admitted source events")
    return dict(
        schema=SCHEMA,
        selected_topology=receipt["selected_topology"],
        original_contact_count=len(original),
        selected_contact_count=len(selected),
        original_first_contact_preserved=True,
        original_inputs_changed=False,
    )
