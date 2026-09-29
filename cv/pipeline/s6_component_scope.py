"""Executable original-contact components, separately bound to a frozen source plan.

This is not a prefix contract. Every component keeps original records and native
epochs; the closing contact is an ordinary contact with possibly absent evidence.
Embedded original documents allow standalone stage replay without cached fits.
"""

from copy import deepcopy

from cv.experiments.connected_shooting.labeled_event_occurrence import record_hash
from cv.pipeline import s6_contact_components as preparation
from cv.pipeline import s6_contact_prefix_runtime as prefix_runtime
from cv.pipeline import s6_first_contact_role as role

SCHEMA = "s6_original_contact_component_v1"
SOURCE_FIELD = "contact_component_source"


def active(attempt: dict) -> bool:
    return attempt.get("observation_scope", {}).get("schema") == SCHEMA


def track_terminal(attempt: dict) -> dict | None:
    """Ball-track ending on a bound component, or None for a contact boundary.

    Ordinary components stay contact-bounded. A terminal component is still a
    component: the search uses the supplied ending and does not turn on the
    horizon, recovery, or net-tail mechanisms.
    """
    if not active(attempt):
        return None
    scope = attempt["observation_scope"]
    receipt = scope.get(preparation.TERMINAL_TRACK_FIELD)
    if receipt is None:
        return None
    if scope.get("right_boundary_kind") != "supplied_end" or not isinstance(receipt, dict):
        raise ValueError("ball-track terminal requires a supplied ending")
    if receipt.get("kind") not in preparation.TERMINAL_KINDS:
        raise ValueError("explicit ball-track terminal kind required")
    if scope.get("endpoint_semantics") != receipt["kind"]:
        raise ValueError("endpoint semantics differ from the ball-track terminal")
    return receipt


def _split_meta(component: dict) -> dict:
    return dict(
        parent_component_index=component["split_parent_component_index"],
        local_flight_index=component["split_local_flight_index"],
        original_flight_index=component["original_flight_indices"][0],
    )


def _contract_from(plan: dict, component: dict) -> dict:
    return dict(
        schema=SCHEMA,
        mode=preparation.MODE,
        plan_sha256=plan["plan_sha256"],
        source_bindings=deepcopy(plan["source_bindings"]),
        component=deepcopy(component),
        first_contact_role=component["first_contact_role"],
        native_window=list(plan["original_native_window"]),
        modeled_horizon=component["end_frame"],
        modeled_frame_interval=[component["start_frame"], component["end_frame"]],
        observation_horizon=component["end_frame"],
        original_observation_horizon=plan["original_owner_end_frame"],
        observation_partition=plan["observation_partition"],
        observation_fallback=plan["observation_fallback"],
        original_slot_count=plan["original_slot_count"],
        retained_flight_indices=list(component["original_flight_indices"]),
        unresolved_original_slots=deepcopy(
            [s for s in plan["original_slots"] if s["status"] == "unresolved"]
        ),
        right_boundary_kind=(
            "supplied_end"
            if component.get(preparation.TERMINAL_TRACK_FIELD)
            else "original_contact"
        ),
        right_boundary_membership="half_open_original_interior",
        ending_semantics="unresolved",
        physical_ending=None,
        # Present only on a ball-track terminal, so an ordinary contact-bounded
        # component contract stays byte-identical.
        **(
            {
                "endpoint_semantics": component["endpoint_semantics"],
                preparation.TERMINAL_TRACK_FIELD: deepcopy(
                    component[preparation.TERMINAL_TRACK_FIELD]
                ),
            }
            if component.get(preparation.TERMINAL_TRACK_FIELD)
            else {}
        ),
        complete_original_source=False,
        continuity_with_other_components=False,
        **(
            {preparation.SPLIT_RECEIPT_FIELD: _split_meta(component)}
            if "split_local_flight_index" in component
            else {}
        ),
        # Present only where the original source carried leading evidence, so
        # existing component contracts and their replays are unchanged.
        **(
            {
                "leading_physical_prefix": deepcopy(plan["leading_physical_prefix"]),
                "leading_prefix_modelled": False,
            }
            if plan.get("leading_physical_prefix")
            else {}
        ),
    )


def _labels(labels: dict, contract: dict) -> dict:
    result = deepcopy(labels)
    meta = result["attempt"]
    meta.setdefault("original_source_attempt", deepcopy(meta))
    meta.pop("ending_frame", None)
    meta.update(
        first_contact_frame=contract["component"]["start_frame"],
        ending_kind="unresolved",
        observation_scope=deepcopy(contract),
    )
    return result


def _contract(plan: dict, index: int) -> dict:
    return _contract_from(plan, plan["components"][index])


def _split_contract(plan: dict, parent_index: int, local_flight: int, component: dict) -> dict:
    if (
        component.get("split_parent_component_index") != parent_index
        or component.get("split_local_flight_index") != local_flight
        or len(component.get("original_flight_indices") or []) != 1
    ):
        raise ValueError("split component identity differs from the parent flight")
    return _contract_from(plan, component)


def _expected_contract(plan: dict, attempt_contract: dict, packet, labels, cameras) -> dict:
    split = attempt_contract.get(preparation.SPLIT_RECEIPT_FIELD)
    if split is None:
        return _contract(plan, attempt_contract["component"]["component_index"])
    source = preparation._source(
        packet,
        labels,
        cameras,
        plan["observation_partition"],
        plan.get("leading_event_components", preparation.LEADING_DEFAULT),
    )
    component = preparation.split_component(
        plan,
        split["parent_component_index"],
        split["local_flight_index"],
        source=source,
        labels=labels,
    )
    return _split_contract(
        plan, split["parent_component_index"], split["local_flight_index"], component
    )


def _bind_attempt(
    plan: dict, packet: dict, labels: dict, cameras: dict, contract: dict
) -> tuple[dict, dict]:
    source = preparation._source(
        packet,
        labels,
        cameras,
        plan["observation_partition"],
        plan.get("leading_event_components", preparation.LEADING_DEFAULT),
    )
    component = contract["component"]
    attempt = deepcopy(source)
    # Remove only former prefix declarations and terminal grammars. Their exact
    # original packet remains in SOURCE_FIELD; physical records are never edited.
    for name in (
        "original_physical_events",
        "original_owner_end_frame",
        "original_owner_end_frame_semantics",
        "original_observation_scope",
        "terminal_net_tail",
        "observed_horizon_tail",
    ):
        attempt.pop(name, None)
    attempt.update(
        events=deepcopy(component["events"]),
        owner_end_frame=component["end_frame"],
        first_event_frame=component["start_frame"],
        contact_count=len(component["original_contact_indices"]),
        owner_end_frame_semantics=(
            "ball_track_terminal_not_declared_point_end"
            if component.get(preparation.TERMINAL_TRACK_FIELD)
            else "original_contact_right_boundary_not_physical_ending"
        ),
        observation_scope=contract,
    )
    if component["first_contact_role"] in role.BOUND_ROLES:
        binder = (
            role.bind_rally if component["first_contact_role"] == "rally" else role.bind_unknown
        )
        attempt = binder(
            attempt,
            plan["original_events"],
            derivation=component["first_contact_role_evidence"]["derivation"],
        )
        if attempt[role.FIELD] != component["first_contact_role_evidence"]:
            raise ValueError("component role differs from original prepared role")
    attempt[SOURCE_FIELD] = deepcopy(
        dict(
            plan=plan,
            packet=packet,
            labels=labels,
            cameras=cameras,
        )
    )
    result = deepcopy(packet)
    result["attempts"] = [attempt]
    # The caller updates this raw-file binding after writing unchanged derived
    # labels. Keeping a stale parent label hash would misrepresent stage input.
    result.pop("external_label_binding", None)
    return result, _labels(labels, contract)


def bind(plan: dict, packet: dict, labels: dict, cameras: dict, index: int) -> tuple[dict, dict]:
    """Bind one member of the completely validated input-only roster."""
    preparation.validate(
        plan,
        packet,
        labels,
        cameras,
        observation_partition=plan["observation_partition"],
        observation_fallback=plan["observation_fallback"],
    )
    if type(index) is not int or not 0 <= index < len(plan["components"]):
        raise ValueError("original prepared component index required")
    return _bind_attempt(plan, packet, labels, cameras, _contract(plan, index))


def bind_split(
    plan: dict, packet: dict, labels: dict, cameras: dict, parent_index: int, local_flight: int
) -> tuple[dict, dict]:
    """Bind one original flight of a failed joined component, without rewriting the plan."""
    preparation.validate(
        plan,
        packet,
        labels,
        cameras,
        observation_partition=plan["observation_partition"],
        observation_fallback=plan["observation_fallback"],
    )
    source = preparation._source(
        packet,
        labels,
        cameras,
        plan["observation_partition"],
        plan.get("leading_event_components", preparation.LEADING_DEFAULT),
    )
    component = preparation.split_component(
        plan, parent_index, local_flight, source=source, labels=labels
    )
    contract = _split_contract(plan, parent_index, local_flight, component)
    return _bind_attempt(plan, packet, labels, cameras, contract)


def validate(attempt: dict, labels: dict | None = None, report: dict | None = None) -> dict:
    """Cheap structural replay; complete source qualification runs before fits."""
    if not active(attempt):
        raise ValueError("original-contact component contract required")
    receipt = attempt.get(SOURCE_FIELD, {})
    plan = receipt.get("plan", {})
    if record_hash({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get(
        "plan_sha256"
    ):
        raise ValueError("component source plan hash changed")
    contract = attempt["observation_scope"]
    expected = _expected_contract(
        plan, contract, receipt["packet"], receipt["labels"], receipt["cameras"]
    )
    if contract != expected:
        raise ValueError("component differs from original source plan")
    for name in ("packet", "labels", "cameras"):
        if record_hash(receipt[name]) != plan["source_bindings"][name]:
            raise ValueError("component original source document changed")
    component = contract["component"]
    if (
        attempt["events"] != component["events"]
        or attempt["owner_end_frame"] != component["end_frame"]
    ):
        raise ValueError("component original events or contact endpoint changed")
    if attempt["owner_ball_labels"] != receipt["packet"]["attempts"][0]["owner_ball_labels"]:
        raise ValueError("component changed original native observations")
    if any(attempt.get(k) is not None for k in ("terminal_net_tail", "observed_horizon_tail")):
        raise ValueError("component contact boundary cannot carry a terminal grammar")
    if labels is not None and labels != _labels(receipt["labels"], contract):
        raise ValueError("component labels differ from original observations or scope")
    role.validate(attempt, labels, requested=component["first_contact_role"])
    if report is not None and (
        report.get("configuration", {}).get("observation_scope") != contract
        or report.get("events") != attempt["events"]
        or report.get("configuration", {}).get("contact_components") != preparation.MODE
        or report.get("configuration", {}).get("right_boundary_kind")
        != contract["right_boundary_kind"]
    ):
        raise ValueError("search differs from bound original-contact component")
    return contract


def validate_inputs(attempt: dict, labels: dict, cameras: dict, partition: str) -> dict:
    contract = validate(attempt, labels)
    source = attempt[SOURCE_FIELD]
    if cameras != source["cameras"]:
        # The plan stays bound to the camera it was prepared from. A near-baseline
        # refinement may replace that document for projection only, and only by the
        # fields the refit itself writes. Rollback of the refit leaves this false.
        from cv.pipeline.court_near_baseline_refinement import same_component_source
        from cv.pipeline.court_paint_refinement import strip_refinement

        # A paint refit keeps the projection it replaced; strip it first, then the
        # near-baseline rule decides whether what remains is the bound source.
        base = strip_refinement(cameras)
        if base is None or not (
            base == source["cameras"] or same_component_source(base, source["cameras"])
        ):
            raise ValueError("component cameras differ from original source")
    preparation.validate(
        source["plan"],
        source["packet"],
        source["labels"],
        source["cameras"],
        observation_partition=partition,
        observation_fallback=contract["observation_fallback"],
    )
    split = contract.get(preparation.SPLIT_RECEIPT_FIELD)
    if split is None:
        expected, _ = bind(
            source["plan"],
            source["packet"],
            source["labels"],
            source["cameras"],
            contract["component"]["component_index"],
        )
    else:
        expected, _ = bind_split(
            source["plan"],
            source["packet"],
            source["labels"],
            source["cameras"],
            split["parent_component_index"],
            split["local_flight_index"],
        )
    if attempt != expected["attempts"][0]:
        raise ValueError("component attempt differs from original input-only binding")
    return contract


def applied_policy(policy: dict, packet: dict) -> tuple[dict, dict | None]:
    attempt = packet["attempts"][0]
    if not active(attempt):
        return policy, None
    if policy.get("contact_components", "off") != preparation.MODE:
        raise ValueError("component requires explicit unresolved_ending component policy")
    contract = validate(attempt)
    result = dict(policy)
    changes = {}
    for key in (*prefix_runtime.NOT_APPLICABLE_SETTINGS, "contact_prefix_scope"):
        if result.get(key, "off") != "off":
            changes[key] = dict(
                requested=result[key], applied="off", reason="original_contact_boundary"
            )
        result[key] = "off"
    result.update(
        observation_scope="on", first_contact_role=contract["component"]["first_contact_role"]
    )
    return result, dict(
        schema="s6_contact_component_applicability_v1",
        changes=changes,
        plan_sha256=contract["plan_sha256"],
        component_index=contract["component"]["component_index"],
        right_boundary_kind=contract["right_boundary_kind"],
        topology="original_component_events",
    )


def refinement_policy(base, policy):
    from cv.pipeline.s6_component_orchestration import refinement_policy as allocated

    return allocated(base, policy)
