"""Shared runtime application of an input-qualified original-contact prefix.

The requested recipe remains recorded. Only mechanisms requiring an unavailable
physical ending or a changed topology abstain, identically for every such scope.
"""

from copy import deepcopy
from cv.pipeline import s6_contact_prefix_scope as scope

# Terminal mechanisms cannot infer that a cropped contact ends the point.
NOT_APPLICABLE_SETTINGS = (
    "independent_event_proposals",
    "optional_contacts",
    "optional_final_contacts",
    "optional_interior_contacts",
    "optional_contact_timing",
    "optional_contact_composition",
    "optional_bounces",
    "optional_conditional_grounds",
    "terminal_context_ownership",
    "terminal_net_tail",
    "terminal_net_membership",
    "observed_horizon_tail",
    "terminal_impact_intervals",
    "terminal_ground_normal",
    "terminal_net_coupling",
    "terminal_ground_coupling",
    "terminal_net_seed",
    "observation_net_seed",
    "ground_settling",
    "terminal_ground_sparse_wings",
    "net_first_contact_toss",
    "prefix_net_revisit",
)
#: Scope-local interior composition, admitted only under ``terminal_identity``.
#: An interior contact strictly inside the retained span needs no ending; every
#: terminal family and every bounce family above still abstains.
PREFIX_LOCAL_INTERIOR_SETTINGS = (
    "optional_contacts",
    "optional_interior_contacts",
    "optional_contact_composition",
)


def active(attempt):
    return attempt.get("observation_scope", {}).get("schema") == scope.SCHEMA


def _policy_error(mode):
    """The bound scope's own policy demand; coverage keeps its original wording."""
    if mode == scope.COVERAGE:
        return "contact-prefix inputs require explicit shared coverage policy"
    return f"contact-prefix inputs require explicit shared {mode} policy"


def bound_mode(attempt):
    """The mode the bound prefix contract itself declares, never the request."""
    return attempt.get("observation_scope", {}).get("mode")


def prepare_packet(packet, labels, cameras, mode, partition):
    scope.validate_mode(mode)
    attempt = packet["attempts"][0]
    if active(attempt):
        if mode == "off" or mode != bound_mode(attempt):
            raise ValueError(_policy_error(bound_mode(attempt)))
        contract = scope.validate(attempt, cameras, observation_partition=partition, labels=labels)
        return packet, {
            "status": "qualified",
            "contract": contract,
            "already_prepared": True,
            **{
                name: contract[name]
                for name in (
                    "mode",
                    "observation_partition",
                    "observation_fallback",
                    "first_coverage_failure",
                )
            },
            **(
                {"terminal_identity_cut": contract["terminal_identity_cut"]}
                if mode == scope.TERMINAL_IDENTITY
                else {}
            ),
        }
    if mode == "off":
        return packet, {"status": "policy_disabled"}
    source = dict(attempt, owner_ending_kind=labels.get("attempt", {}).get("ending_kind"))
    qualification = scope.qualify(
        source,
        cameras,
        observation_partition=partition,
        observation_fallback=True,
        native_window=labels.get("attempt", {}).get("native_window"),
        mode=mode,
        labels=labels,
    )
    if qualification["status"] != "qualified":
        return packet, qualification
    prepared = deepcopy(packet)
    prepared["attempts"] = [scope.bind_attempt(source, qualification)]
    return prepared, qualification


def applied_policy(policy, packet):
    attempt = packet["attempts"][0]
    if not active(attempt):
        return policy, None
    mode = bound_mode(attempt)
    if policy.get("contact_prefix_scope", "off") != mode:
        raise ValueError(_policy_error(mode))
    local = mode == scope.TERMINAL_IDENTITY
    result = dict(policy)
    changes = {}
    retained = {}
    for name in NOT_APPLICABLE_SETTINGS:
        if local and name in PREFIX_LOCAL_INTERIOR_SETTINGS:
            if result.get(name, "off") != "off":
                retained[name] = {
                    "requested": result[name],
                    "applied": result[name],
                    "reason": "interior_contact_strictly_inside_retained_prefix",
                }
            continue
        if result.get(name, "off") != "off":
            changes[name] = {
                "requested": result[name],
                "applied": "off",
                "reason": "original_contact_right_boundary",
            }
        result[name] = "off"
    result["observation_scope"] = "on"
    applicability = {
        "schema": "s6_contact_prefix_applicability_v1",
        "changes": changes,
        "topology": "original_retained_events",
        "right_boundary_kind": "original_contact",
    }
    if local:
        # Published explicitly so a reader cannot mistake a prefix-local interior
        # proposal for the full topology search: terminal families stay off.
        applicability |= {
            "mode": mode,
            "interior_composition": scope.INTERIOR_COMPOSITION,
            "interior_composition_scope": "strictly_inside_retained_prefix",
            "terminal_families": "off",
            "scope_local_settings": retained,
        }
    return result, applicability


def labels_for_scope(labels, packet):
    attempt = packet["attempts"][0]
    if not active(attempt):
        return labels
    result = deepcopy(labels)
    meta = result["attempt"]
    meta.setdefault("original_source_attempt", deepcopy(meta))
    meta.pop("ending_frame", None)
    meta.update(observation_scope=deepcopy(attempt["observation_scope"]), ending_kind="unresolved")
    return result


def validate_labels(attempt, labels):
    if labels is None:
        return
    original = attempt["original_physical_events"]
    if any(e.get("clip", attempt["point_clip"]) != attempt["point_clip"] for e in original):
        raise ValueError("contact-prefix original events differ from bound consumer clip")
    # Automatic packets omit this redundant address; labeled packets may retain
    # it. Compare the same physical record after checking the enclosing clip.
    original = [{k: v for k, v in e.items() if k != "clip"} for e in original]
    events = [
        {k: v for k, v in e.items() if k != "clip"}
        for e in labels["events"]["records"]
        if e.get("clip", attempt["point_clip"]) == attempt["point_clip"]
        and e.get("event_type") in scope.PHYSICAL
    ]
    # Labeled contexts may include later physical events; original packet events
    # must still occur verbatim and in order. No label makes a new runtime cut.
    if [e for e in events if e in original] != original:
        raise ValueError("contact-prefix original events differ from bound consumer")
    meta = labels.get("attempt", {})
    declared = meta.get("observation_scope")
    if declared is not None and declared not in (
        attempt["observation_scope"],
        attempt.get("original_observation_scope"),
    ):
        raise ValueError("consumer scope differs from original or qualified prefix")
