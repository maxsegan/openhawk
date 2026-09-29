"""Default-off cold S6 component execution under one fixed source budget.

Invoked by s6_attempt_execution.execute. The entire roster and budget are written before
any numerical worker starts. Components never borrow unused work or retry, and
original unresolved slots remain explicit in the aggregate verdict.
"""

from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from dataclasses import asdict, replace
import json
import math
import time

from cv.experiments.connected_shooting.labeled_event_occurrence import record_hash
from cv.pipeline import provenance, s6_labeled_stage as stage
from cv.pipeline import s6_component_scope as scope
from cv.pipeline import s6_contact_components as preparation

SCHEMA = "labeled_s6_contact_component_source_v1"
ALLOCATION = "contact_component_allocation"


def _integers(total: int, weights: list[int]) -> list[int]:
    if type(total) is not int or total < 1:
        raise ValueError("positive source integer iteration budget required")
    denominator = sum(weights)
    shares = [total * w // denominator for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denominator), i))
    for i in order[: total - sum(shares)]:
        shares[i] += 1
    return shares


def _share_seconds(total: float, n: int) -> list[float]:
    if n < 1:
        return []
    milliseconds = max(0, int(round(float(total) * 1000)))
    if milliseconds == 0:
        return [0.0] * n
    return [part / 1000.0 for part in _integers(milliseconds, [1] * n)]


def _released_components(plan: dict) -> set[int]:
    """Components that exist only because ``ambiguous_interior_ground`` released them.

    Their flight was held with the switch off, so the component must not take
    search from the components that ran then.
    """
    return {
        i
        for i, component in enumerate(plan["components"])
        if component.get(preparation.AMBIGUOUS_GROUND_FIELD)
        == preparation.AMBIGUOUS_GROUND_SUPPLIED
    }


def _boundary_added(plan: dict) -> set[int]:
    """Components whose flights all close on a labelled boundary (boundary_endings on)."""
    if plan.get(preparation.BOUNDARY_ENDINGS_FIELD, preparation.BOUNDARY_ENDINGS_OFF) != (
        preparation.BOUNDARY_ENDINGS_ON
    ):
        return set()
    return {
        i
        for i, component in enumerate(plan["components"])
        if ((component.get("terminal_track_endpoint") or {}).get("evidence")) == "labelled_boundary"
        and len(component["original_flight_indices"]) == 1
    }


def allocate(plan: dict, policy: dict, timeout: float) -> dict:
    """Largest-remainder integer caps; continuous caps use flight-count shares.

    These are per-call optimizer caps, as in the ordinary stage, not a claim
    about summed iterations over its existing multistart algorithm. Hard wall
    shares bound total component work including all refinement calls.
    """
    from cv.pipeline.s6_shared_refinement import Policy

    if ALLOCATION in policy:
        raise ValueError("original source policy cannot contain a child allocation")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite total source wall budget required")
    search = policy.get("search_seconds")
    if (
        not isinstance(search, (int, float))
        or not math.isfinite(search)
        or not 0 < search < timeout
    ):
        raise ValueError(
            "component mode requires explicit search_seconds below total source timeout"
        )
    weights = [len(c["original_flight_indices"]) for c in plan["components"]]
    boundary_added = _boundary_added(plan)
    released = _released_components(plan)
    added = boundary_added | released
    # Reserve a fixed fraction for source preparation/export bookkeeping. Unused
    # reserve and fast-component time are never added to a later fit.
    reserve = timeout * 0.05
    if search >= timeout - reserve:
        raise ValueError("search budget must leave the fixed source preparation reserve")
    base = asdict(Policy())
    base["interior_seconds"] = stage.shared_settings(policy)["interior_block_seconds"]
    caps = {
        k: v
        for k, v in base.items()
        if k.endswith(("_maxiter", "_max_nfev", "_seconds")) or k == "seconds_per_stage"
    }
    # Components that exist only because boundary_endings closed a held flight, or
    # ambiguous_interior_ground released one, keep their flight-count share, but on
    # top of the others: the remaining components are split exactly as without the
    # switch. The source deadline still binds.
    base = [0 if i in added else w for i, w in enumerate(weights)]
    if not any(base):
        base = weights

    def split(total: int) -> list[int]:
        own = _integers(total, base)
        whole = _integers(total, weights)
        return [whole[i] if i in added else own[i] for i in range(len(weights))]

    integer_caps = {k: split(v) for k, v in caps.items() if type(v) is int} if weights else {}
    search_caps = (
        {k: split(policy[k]) for k in ("coarse_iterations", "refine_iterations")} if weights else {}
    )
    allocations = []
    for i, w in enumerate(weights):
        fraction = w / sum(weights if i in added else base)
        refinement = {
            k: (integer_caps[k][i] if type(v) is int else v * fraction) for k, v in caps.items()
        }
        iterations = {k: v[i] for k, v in search_caps.items()}
        executable = all(v > 0 for v in iterations.values()) and (
            policy.get("refinement", "off") == "off" or all(v > 0 for v in refinement.values())
        )
        allocations.append(
            dict(
                component_index=i,
                plan_sha256=plan["plan_sha256"],
                flight_weight=w,
                total_flight_weight=sum(weights),
                fraction=fraction,
                wall_seconds=(timeout - reserve) * fraction,
                search_seconds=search * fraction,
                **iterations,
                refinement_caps=refinement,
                executable=executable,
                reason=None if executable else "zero_allocated_integer_cap",
            )
        )
    document = dict(
        schema="s6_contact_component_budget_v1",
        source_policy=deepcopy(policy),
        source_wall_seconds=timeout,
        source_preparation_reserve_seconds=reserve,
        allocation_rule="original prepared flight counts; largest remainder integer caps",
        **(
            {
                k: sorted(v)
                for k, v in (
                    ("boundary_added_components", boundary_added),
                    ("released_components", released),
                )
                if v
            }
            if added and any(i not in added for i in range(len(weights)))
            else {}
        ),
        optimizer_cap_semantics="per solver call, preserving existing multistart procedure",
        borrowing=False,
        sequential_retries=False,
        components=allocations,
    )
    return document | dict(sha256=record_hash(document))


def child_policy(source_policy: dict, child_packet: dict, budget: dict, index: int) -> dict:
    applied, _ = scope.applied_policy(source_policy, child_packet)
    allocation = budget["components"][index]
    return (
        applied
        | {k: allocation[k] for k in ("coarse_iterations", "refine_iterations", "search_seconds")}
        | {
            ALLOCATION: dict(budget=deepcopy(budget), allocation=deepcopy(allocation)),
        }
    )


_FAILURE_STATUSES = frozenset(
    {
        "execution_failed",
        "timeout",
        "source_budget_exhausted",
        "invalid_component_emission",
        "component_source_failed",
    }
)


def _settings(policy: dict) -> dict:
    return stage.shared_settings(policy)


def _search_exhausted(record: dict) -> bool:
    budget = (record.get("consumed") or {}).get("search_budget") or {}
    if isinstance(budget, dict) and budget.get("exhausted") is True:
        return True
    inner = (record.get("result") or {}).get("search_budget") or {}
    return isinstance(inner, dict) and inner.get("exhausted") is True


def _accepted_originals(plan: dict, index: int, result: dict) -> list[int]:
    try:
        flights = checked_emissions(plan, index, result)
    except (ValueError, KeyError, TypeError):
        return []
    originals = plan["components"][index]["original_flight_indices"]
    return [originals[local] for local, flight in enumerate(flights) if flight.get("accepted")]


def _unaccepted_locals(plan: dict, index: int, result: dict) -> list[int]:
    originals = plan["components"][index]["original_flight_indices"]
    accepted = set(_accepted_originals(plan, index, result))
    return [local for local, original in enumerate(originals) if original not in accepted]


def split_child_policy(
    source_policy: dict, child_packet: dict, budget: dict, parent_index: int, split_allocation: dict
) -> dict:
    applied, _ = scope.applied_policy(source_policy, child_packet)
    parent = budget["components"][parent_index]
    return (
        applied
        | {
            k: split_allocation[k]
            for k in ("coarse_iterations", "refine_iterations", "search_seconds")
        }
        | {
            ALLOCATION: dict(
                budget=deepcopy(budget),
                allocation=deepcopy(parent),
                split_fallback=deepcopy(split_allocation),
            )
        }
    )


def retry_child_policy(
    source_policy: dict, child_packet: dict, budget: dict, index: int, retry_allocation: dict
) -> dict:
    applied, _ = scope.applied_policy(source_policy, child_packet)
    parent = budget["components"][index]
    return (
        applied
        | {
            k: retry_allocation[k]
            for k in ("coarse_iterations", "refine_iterations", "search_seconds")
        }
        | {
            ALLOCATION: dict(
                budget=deepcopy(budget),
                allocation=deepcopy(parent),
                exhausted_retry=deepcopy(retry_allocation),
            )
        }
    )


def validate_policy(policy: dict, attempt: dict) -> dict:
    contract = scope.validate(attempt)
    binding = policy.get(ALLOCATION)
    if binding is None:
        raise ValueError("component stage requires its frozen source work allocation")
    budget = binding["budget"]
    expected = allocate(
        attempt[scope.SOURCE_FIELD]["plan"], budget["source_policy"], budget["source_wall_seconds"]
    )
    index = contract["component"]["component_index"]
    if budget != expected or binding["allocation"] != expected["components"][index]:
        raise ValueError("component allocation differs from original source budget")
    split = contract.get(preparation.SPLIT_RECEIPT_FIELD)
    if split is not None:
        child = binding.get("split_fallback")
        if not isinstance(child, dict):
            raise ValueError("split fallback requires its frozen child allocation")
        if (
            child.get("parent_component_index") != split["parent_component_index"]
            or child.get("local_flight_index") != split["local_flight_index"]
            or child.get("original_flight_index") != split["original_flight_index"]
        ):
            raise ValueError("split allocation differs from the bound original flight")
        expected_policy = split_child_policy(
            budget["source_policy"], dict(attempts=[attempt]), budget, index, child
        )
        if policy != expected_policy:
            raise ValueError("component numerical recipe differs from its frozen source allocation")
        if not child.get("executable"):
            raise ValueError("zero component allocation cannot become a solver default")
        return child
    retry = binding.get("exhausted_retry")
    if retry is not None:
        if not isinstance(retry, dict) or retry.get("parent_component_index") != index:
            raise ValueError("exhausted retry requires its frozen parent allocation")
        expected_policy = retry_child_policy(
            budget["source_policy"], dict(attempts=[attempt]), budget, index, retry
        )
        if policy != expected_policy:
            raise ValueError("component numerical recipe differs from its frozen source allocation")
        if not retry.get("executable"):
            raise ValueError("zero component allocation cannot become a solver default")
        return retry
    if not binding["allocation"]["executable"]:
        raise ValueError("zero component allocation cannot become a solver default")
    expected_policy = child_policy(budget["source_policy"], dict(attempts=[attempt]), budget, index)
    if policy != expected_policy:
        raise ValueError("component numerical recipe differs from its frozen source allocation")
    return binding["allocation"]


def refinement_policy(base, policy):
    """Apply all allocated caps explicitly; ordinary mode returns the same object."""
    if ALLOCATION not in policy:
        return base
    binding = policy[ALLOCATION]
    if "split_fallback" in binding:
        return replace(base, **binding["split_fallback"]["refinement_caps"])
    if "exhausted_retry" in binding:
        return replace(base, **binding["exhausted_retry"]["refinement_caps"])
    return replace(base, **binding["allocation"]["refinement_caps"])


def checked_emissions(plan: dict, index: int, result: dict) -> list[dict]:
    component = plan["components"][index]
    verdict = result.get("verdict", {})
    flights = verdict.get("flights", [])
    if not flights and not result.get("measurement"):
        return []
    contract = scope._contract(plan, index)
    reconstruction = verdict.get("contact_component_reconstruction", {})
    if (
        len(flights) != len(component["original_flight_indices"])
        or verdict.get("complete_point") is not False
        or reconstruction.get("scope") != contract
        or reconstruction.get("local_to_original") != component["local_to_original"]
        or reconstruction.get("complete_original_source") is not False
        or result.get("preparation", {}).get("contact_component_scope") != contract
    ):
        raise ValueError(
            "component emissions differ from original source binding or flight mapping"
        )
    return flights


def checked_split_emissions(plan: dict, contract: dict, result: dict) -> list[dict]:
    component = contract["component"]
    verdict = result.get("verdict", {})
    flights = verdict.get("flights", [])
    if not flights and not result.get("measurement"):
        return []
    reconstruction = verdict.get("contact_component_reconstruction", {})
    if (
        len(flights) != 1
        or len(component["original_flight_indices"]) != 1
        or verdict.get("complete_point") is not False
        or reconstruction.get("scope") != contract
        or reconstruction.get("local_to_original") != component["local_to_original"]
        or reconstruction.get("complete_original_source") is not False
        or result.get("preparation", {}).get("contact_component_scope") != contract
    ):
        raise ValueError(
            "split emissions differ from the original flight binding or flight mapping"
        )
    return flights


def aggregate(
    plan: dict,
    records: list[dict],
    *,
    seed_failure: str | None = None,
    split_records: list[dict] | None = None,
) -> dict:
    """One emission at most per original origin; duplicate identity is an error."""
    emissions = {}
    for record in records:
        index = record["component_index"]
        component = plan["components"][index]
        flights = checked_emissions(plan, index, record["result"])
        if not flights:
            continue
        if len(flights) != len(component["original_flight_indices"]):
            raise ValueError("component emission count differs from original flight mapping")
        for local, flight in enumerate(flights):
            original = component["original_flight_indices"][local]
            if original in emissions:
                raise ValueError("duplicate component emission for original flight")
            slot = plan["original_slots"][original]
            # Fitted epochs can move inside supplied brackets. Identity comes
            # from the frozen map; do not relabel a fit by nearest reference.
            if not math.isfinite(float(flight["start_frame"])):
                raise ValueError("invalid component flight origin")
            emissions[original] = deepcopy(flight) | dict(
                original_flight_index=original,
                original_contact_index=slot["original_contact_index"],
                original_origin_frame=slot["start_frame"],
                component_index=index,
                local_flight_index=local,
            )
    for record in split_records or []:
        contract = record["contract"]
        original = contract[preparation.SPLIT_RECEIPT_FIELD]["original_flight_index"]
        incumbent = emissions.get(original)
        if incumbent is not None and incumbent.get("accepted"):
            continue
        flights = checked_split_emissions(plan, contract, record["result"])
        if not flights or not flights[0].get("accepted"):
            continue
        slot = plan["original_slots"][original]
        if not math.isfinite(float(flights[0]["start_frame"])):
            raise ValueError("invalid split flight origin")
        emissions[original] = deepcopy(flights[0]) | dict(
            original_flight_index=original,
            original_contact_index=slot["original_contact_index"],
            original_origin_frame=slot["start_frame"],
            component_index=contract["component"]["component_index"],
            local_flight_index=contract["component"]["split_local_flight_index"],
            split_fallback=True,
        )
    slots = []
    for original, source_slot in enumerate(plan["original_slots"]):
        if original in emissions:
            slots.append(emissions[original] | dict(emitted=True))
        else:
            reasons = deepcopy(source_slot["reasons"]) or ["component_execution_did_not_emit"]
            if seed_failure:
                from cv.experiments.connected_shooting.candidate_attempts import (
                    attempt_seed_failure_reason,
                )

                tag = attempt_seed_failure_reason(seed_failure)
                if tag not in reasons:
                    reasons = [tag, *reasons]
            slots.append(
                dict(
                    original_flight_index=original,
                    original_contact_index=source_slot["original_contact_index"],
                    original_origin_frame=source_slot["start_frame"],
                    start_frame=source_slot["start_frame"],
                    end_frame=source_slot["end_frame"],
                    accepted=False,
                    emitted=False,
                    reasons=reasons,
                    component_index=source_slot["component_index"],
                )
            )
    accepted = [i for i, f in enumerate(slots) if f.get("accepted") and f["emitted"]]
    return dict(
        complete_point=False,
        complete_original_source=False,
        flight_count=len(emissions),
        original_slot_count=len(slots),
        flights=slots,
        accepted_flight_count=len(accepted),
        accepted_flight_indices=accepted,
        gaps=[i for i, f in enumerate(slots) if not f["emitted"]],
        slot_semantics="all original origins; emitted false entries are explicit missing outputs",
    )


def _record(path: Path) -> dict:
    record = provenance.file_record(path)
    stage.resolve(record)  # Reject temporary paths outside declared provenance roots.
    return record


def _child_producer(row: dict, input_dir: Path, plan: Path, configuration: dict) -> dict:
    """An automatic child row needs a producer that binds its rewritten labels and packet."""
    if row.get("observation_origin") != "automatic":
        return {}
    from cv.pipeline.s6_input_origin import derived_producer

    path = derived_producer(
        row,
        input_dir / "automatic_provenance.json",
        [
            provenance.file_record(
                input_dir / "labels.json", role="automatic_observation_document"
            ),
            provenance.file_record(input_dir / "packet.json", role="automatic_observation_packet"),
            provenance.file_record(plan, role="contact_component_plan"),
        ],
        {"producer": "s6_contact_component_scope", **configuration},
        stage.resolve,
    )
    return dict(automatic_provenance=_record(path))


def run_source(
    row: dict,
    output: Path,
    policy: dict,
    timeout: float,
    *,
    execute: Callable[[dict, Path, dict, float], dict],
    started: float | None = None,
    source_packet: dict | None = None,
    seed_failure: str | None = None,
    whole_point: dict | None = None,
) -> dict:
    """Dispatch only an unresolved source; resolved sources keep ordinary behavior."""
    from cv.pipeline import s6_contact_prefix_scope as prefix
    from cv.experiments.connected_shooting import candidate_attempts as attempts

    started = time.monotonic() if started is None else started
    paths = stage.validate_row(row)
    packet, labels, cameras = (
        json.loads(paths[n].read_text()) for n in ("packet", "labels", "cameras")
    )
    if source_packet is not None:
        packet = source_packet
    original = packet["attempts"][0]
    declaration = original.get("original_observation_scope", original.get("observation_scope", {}))
    if declaration.get("schema") != prefix.UNRESOLVED_INPUT_SCHEMA:
        if seed_failure is not None:
            raise ValueError("seed-failure fallback still lacks an unresolved inventory")
        ordinary = execute(row, output, {**policy, "contact_components": "off"}, timeout)
        if ordinary.get("seed_failure_fallback"):
            stage.save(output / "cases" / row["key"] / "result.json", ordinary)
            return ordinary
        if attempts.should_fallback_to_components(ordinary.get("reason", ""), policy):
            destination = output / "cases" / row["key"]
            stash = output / "cases" / f"{row['key']}__whole_point_seed_failure"
            if destination.exists():
                if stash.exists():
                    raise ValueError("seed-failure stash already present")
                destination.rename(stash)
            remaining = max(1.0, timeout - (time.monotonic() - started))
            return run_source(
                row,
                output,
                policy,
                remaining,
                execute=execute,
                started=started,
                source_packet=preparation.as_unresolved_source(packet),
                seed_failure=str(ordinary.get("reason") or "no completed candidate"),
                whole_point=ordinary,
            )
        ordinary["contact_components"] = dict(
            status="not_applicable_source_scope", requested_mode=preparation.MODE
        )
        stage.save(output / "cases" / row["key"] / "result.json", ordinary)
        return ordinary
    destination = output / "cases" / row["key"]
    if destination.exists():
        raise ValueError("fresh component source output required")
    plan = preparation.prepare(
        packet,
        labels,
        cameras,
        mode=preparation.MODE,
        observation_partition=stage.shared_settings(policy)["observation_partition"],
        leading_event_components=stage.shared_settings(policy).get(
            "leading_event_components", preparation.LEADING_DEFAULT
        ),
        contact_component_routing=stage.shared_settings(policy).get(
            preparation.ROUTING_FIELD, preparation.ROUTING_DEFAULT
        ),
        terminal_track_endpoint=stage.shared_settings(policy).get(
            preparation.TERMINAL_TRACK_FIELD, preparation.TERMINAL_TRACK_OFF
        ),
        supported_flight_endings=stage.shared_settings(policy).get(
            preparation.SUPPORTED_ENDINGS_FIELD, preparation.SUPPORTED_ENDINGS_OFF
        ),
        ending_ownership=stage.shared_settings(policy).get(
            preparation.ENDING_OWNERSHIP_FIELD, preparation.ENDING_OWNERSHIP_OFF
        ),
        out_bounce_ending=stage.shared_settings(policy).get(
            preparation.OUT_BOUNCE_ENDING_FIELD, preparation.OUT_BOUNCE_ENDING_OFF
        ),
        boundary_endings=stage.shared_settings(policy).get(
            preparation.BOUNDARY_ENDINGS_FIELD, preparation.BOUNDARY_ENDINGS_OFF
        ),
        ambiguous_interior_ground=stage.shared_settings(policy).get(
            preparation.AMBIGUOUS_GROUND_FIELD, preparation.AMBIGUOUS_GROUND_OFF
        ),
        net_stop_tail=stage.shared_settings(policy).get(
            preparation.NET_STOP_TAIL_FIELD, preparation.NET_STOP_TAIL_OFF
        ),
    )
    budget = allocate(plan, policy, timeout)
    stage.save(destination / "component_plan.json", plan)
    stage.save(destination / "component_budget.json", budget)
    if seed_failure is not None:
        stage.save(destination / "seed_failure_source_packet.json", packet)
    jobs = []
    for index, component in enumerate(plan["components"]):
        child_packet, child_labels = scope.bind(plan, packet, labels, cameras, index)
        input_dir = destination / "component_inputs" / str(index)
        stage.save(input_dir / "labels.json", child_labels)
        label_record = _record(input_dir / "labels.json")
        child_packet["external_label_binding"] = dict(record=label_record)
        stage.save(input_dir / "packet.json", child_packet)
        child_row = (
            deepcopy(row)
            | dict(
                key=f"{row['key']}__component_{index}",
                labels=label_record,
                packet=_record(input_dir / "packet.json"),
                declared_flights=None
                if row.get("observation_origin") == "automatic"
                else len(component["original_flight_indices"]),
            )
            | _child_producer(
                row, input_dir, destination / "component_plan.json", {"component_index": index}
            )
        )
        recipe = child_policy(policy, child_packet, budget, index)
        manifest = dict(
            schema="labeled_s6_observations_v1",
            rows=[child_row],
            policy=recipe,
            source_row=deepcopy(row),
            source_plan=_record(destination / "component_plan.json"),
        )
        manifest_path = destination / "components" / str(index) / "manifest.json"
        stage.save(manifest_path, manifest)
        jobs.append(
            dict(
                component_index=index,
                row=child_row,
                policy=recipe,
                manifest=_record(manifest_path),
                source_scope_applicability=scope.applied_policy(policy, child_packet)[1],
            )
        )
    frozen = dict(
        schema="s6_contact_component_execution_v1",
        row=deepcopy(row),
        policy=deepcopy(policy),
        plan=_record(destination / "component_plan.json"),
        budget=_record(destination / "component_budget.json"),
        jobs=deepcopy(jobs),
        code=provenance.git_record(stage.paths.REPO_ROOT),
        cold_search=True,
        fitted_input_states_reused=False,
        runtime_model_calls=0,
        **(
            {"seed_failure_source_packet": _record(destination / "seed_failure_source_packet.json")}
            if seed_failure is not None
            else {}
        ),
    )
    stage.save(destination / "component_execution.json", frozen)
    paint_cache = stage.warm_paint_cache(
        labels, cameras, policy, destination / stage.PAINT_CACHE_DIR
    )
    preparation_seconds = time.monotonic() - started
    records = []
    split_records = []
    settings = _settings(policy)
    for job in jobs:
        index = job["component_index"]
        allocation = budget["components"][index]
        remaining = timeout - (time.monotonic() - started)
        cap = min(allocation["wall_seconds"], remaining)
        before = time.monotonic()
        if not allocation["executable"] or cap <= 0:
            result = dict(
                status="source_budget_exhausted",
                verdict={},
                reason=allocation["reason"] or "source_deadline",
            )
        else:
            try:
                result = execute(
                    job["row"], destination / "components" / str(index), job["policy"], cap
                )
            except Exception as error:
                result = dict(
                    status="execution_failed", verdict={}, reason=f"{type(error).__name__}: {error}"
                )
        consumed = time.monotonic() - before
        result.setdefault("key", job["row"]["key"])
        result.setdefault("surface", job["row"]["surface"])
        result.setdefault("declared_flights", job["row"]["declared_flights"])
        result_path = (
            destination / "components" / str(index) / "cases" / job["row"]["key"] / "result.json"
        )
        rejected_binding = None
        try:
            if result["key"] != job["row"]["key"]:
                raise ValueError("component result key differs from its execution job")
            checked_emissions(plan, index, result)
        except (ValueError, KeyError, TypeError) as error:
            rejected_path = result_path.with_name("invalid_emission_result.json")
            stage.save(rejected_path, result)
            rejected_binding = _record(rejected_path)
            result = dict(
                key=job["row"]["key"],
                surface=job["row"]["surface"],
                declared_flights=job["row"]["declared_flights"],
                status="invalid_component_emission",
                reason=str(error),
                verdict=dict(complete_point=False, accepted_flight_count=0, flight_count=0),
            )
        if rejected_binding is not None or not result_path.exists():
            stage.save(result_path, result)
        search_path = result_path.parent / "search/report.json"
        try:
            search_budget = (
                json.loads(search_path.read_text()).get("search_budget")
                if search_path.is_file()
                else None
            )
        except (OSError, ValueError) as error:
            search_budget = dict(status="unavailable_incomplete_report", reason=str(error))
        records.append(
            dict(
                component_index=index,
                local_to_original=plan["components"][index]["local_to_original"],
                result=result,
                result_binding=_record(result_path),
                manifest=job["manifest"],
                allocated=allocation,
                executed_timeout_seconds=max(0.0, cap),
                rejected_emission=rejected_binding,
                source_scope_applicability=job["source_scope_applicability"],
                consumed=dict(
                    wall_seconds=consumed,
                    search_budget=search_budget,
                    total_optimizer_iterations=None,
                    iteration_count_status="not reported by every solver branch",
                ),
            )
        )
        record = records[-1]
        remaining_source = timeout - (time.monotonic() - started)
        if (
            settings.get(preparation.SEARCH_BUDGET_FIELD) == preparation.SEARCH_BUDGET_RETRY
            and _search_exhausted(record)
            and remaining_source > 1
            and allocation["executable"]
        ):
            extra_search = min(remaining_source, max(float(allocation["search_seconds"]), 1.0))
            extra_wall = min(remaining_source, max(float(allocation["wall_seconds"]), extra_search))
            retry_allocation = dict(
                parent_component_index=index,
                flight_weight=allocation["flight_weight"],
                wall_seconds=extra_wall,
                search_seconds=extra_search,
                coarse_iterations=allocation["coarse_iterations"],
                refine_iterations=allocation["refine_iterations"],
                refinement_caps=deepcopy(allocation["refinement_caps"]),
                executable=extra_search > 0 and extra_wall > 0,
                reason=None
                if extra_search > 0 and extra_wall > 0
                else "zero_exhausted_retry_budget",
            )
            if retry_allocation["executable"]:
                child_packet, _ = scope.bind(plan, packet, labels, cameras, index)
                retry_recipe = retry_child_policy(
                    policy, child_packet, budget, index, retry_allocation
                )
                retry_out = destination / "components" / f"{index}__exhausted_retry"
                retry_before = time.monotonic()
                try:
                    retry_result = execute(job["row"], retry_out, retry_recipe, extra_wall)
                except Exception as error:
                    retry_result = dict(
                        status="execution_failed",
                        verdict={},
                        reason=f"{type(error).__name__}: {error}",
                    )
                retry_result.setdefault("key", job["row"]["key"])
                retry_result.setdefault("surface", job["row"]["surface"])
                retry_result.setdefault("declared_flights", job["row"]["declared_flights"])
                retry_path = retry_out / "cases" / job["row"]["key"] / "result.json"
                retry_rejected = None
                try:
                    if retry_result["key"] != job["row"]["key"]:
                        raise ValueError("component result key differs from its execution job")
                    checked_emissions(plan, index, retry_result)
                except (ValueError, KeyError, TypeError) as error:
                    rejected_path = retry_path.with_name("invalid_emission_result.json")
                    stage.save(rejected_path, retry_result)
                    retry_rejected = _record(rejected_path)
                    retry_result = dict(
                        key=job["row"]["key"],
                        surface=job["row"]["surface"],
                        declared_flights=job["row"]["declared_flights"],
                        status="invalid_component_emission",
                        reason=str(error),
                        verdict=dict(complete_point=False, accepted_flight_count=0, flight_count=0),
                    )
                if retry_rejected is not None or not retry_path.exists():
                    stage.save(retry_path, retry_result)
                retry_search = None
                search_path = retry_path.parent / "search/report.json"
                try:
                    retry_search = (
                        json.loads(search_path.read_text()).get("search_budget")
                        if search_path.is_file()
                        else None
                    )
                except (OSError, ValueError) as error:
                    retry_search = dict(status="unavailable_incomplete_report", reason=str(error))
                first_accepted = _accepted_originals(plan, index, record["result"])
                retry_accepted = _accepted_originals(plan, index, retry_result)
                record["exhausted_retry"] = dict(
                    result_binding=_record(retry_path),
                    allocated=retry_allocation,
                    executed_timeout_seconds=extra_wall,
                    consumed=dict(
                        wall_seconds=time.monotonic() - retry_before,
                        search_budget=retry_search,
                    ),
                    rejected_emission=retry_rejected,
                    adopted=len(retry_accepted) > len(first_accepted),
                )
                if len(retry_accepted) > len(first_accepted):
                    record["exhausted_first_result"] = deepcopy(record["result"])
                    record["result"] = retry_result
                    record["result_binding"] = _record(retry_path)
                    record["rejected_emission"] = retry_rejected
                    record["executed_timeout_seconds"] = extra_wall
                    record["consumed"] = dict(
                        wall_seconds=record["consumed"]["wall_seconds"]
                        + (time.monotonic() - retry_before),
                        search_budget=retry_search,
                        total_optimizer_iterations=None,
                        iteration_count_status="not reported by every solver branch",
                    )
        unaccepted = _unaccepted_locals(plan, index, record["result"])
        if (
            settings.get(preparation.SPLIT_FIELD) == preparation.SPLIT_ON
            and len(plan["components"][index]["original_flight_indices"]) >= 2
            and unaccepted
        ):
            remaining_source = timeout - (time.monotonic() - started)
            remaining_wall = max(
                0.0,
                min(
                    float(allocation["wall_seconds"]) - float(record["consumed"]["wall_seconds"]),
                    remaining_source,
                ),
            )
            if _search_exhausted(record):
                remaining_search = (
                    min(remaining_source, float(allocation["search_seconds"]))
                    if settings.get(preparation.SEARCH_BUDGET_FIELD)
                    in (preparation.SEARCH_BUDGET_RETRY, preparation.SEARCH_BUDGET_SPLIT)
                    else 0.0
                )
            else:
                remaining_search = min(remaining_wall, float(allocation["search_seconds"]))
            remaining_wall = max(remaining_wall, remaining_search)
            if remaining_wall > 0 and remaining_search > 0:
                walls = _share_seconds(remaining_wall, len(unaccepted))
                searches = _share_seconds(remaining_search, len(unaccepted))
                freeze_jobs = []
                for local, wall_share, search_share in zip(unaccepted, walls, searches):
                    split_allocation = dict(
                        parent_component_index=index,
                        local_flight_index=local,
                        original_flight_index=plan["components"][index]["original_flight_indices"][
                            local
                        ],
                        flight_weight=1,
                        wall_seconds=wall_share,
                        search_seconds=search_share,
                        coarse_iterations=allocation["coarse_iterations"],
                        refine_iterations=allocation["refine_iterations"],
                        refinement_caps=deepcopy(allocation["refinement_caps"]),
                        executable=(
                            search_share > 0 and wall_share > 0 and allocation["executable"]
                        ),
                        reason=None
                        if search_share > 0 and wall_share > 0
                        else "zero_split_fallback_budget",
                    )
                    child_packet, child_labels = scope.bind_split(
                        plan, packet, labels, cameras, index, local
                    )
                    input_dir = destination / "split_fallback" / f"{index}_{local}" / "inputs"
                    stage.save(input_dir / "labels.json", child_labels)
                    label_record = _record(input_dir / "labels.json")
                    child_packet["external_label_binding"] = dict(record=label_record)
                    stage.save(input_dir / "packet.json", child_packet)
                    child_row = (
                        deepcopy(row)
                        | dict(
                            key=f"{row['key']}__component_{index}__split_{local}",
                            labels=label_record,
                            packet=_record(input_dir / "packet.json"),
                            declared_flights=None
                            if row.get("observation_origin") == "automatic"
                            else 1,
                        )
                        | _child_producer(
                            row,
                            input_dir,
                            destination / "component_plan.json",
                            {"component_index": index, "split_local_flight_index": local},
                        )
                    )
                    recipe = split_child_policy(
                        policy, child_packet, budget, index, split_allocation
                    )
                    manifest_path = (
                        destination / "split_fallback" / f"{index}_{local}" / "manifest.json"
                    )
                    stage.save(
                        manifest_path,
                        dict(
                            schema="labeled_s6_observations_v1",
                            rows=[child_row],
                            policy=recipe,
                            source_row=deepcopy(row),
                            source_plan=_record(destination / "component_plan.json"),
                        ),
                    )
                    freeze_jobs.append(
                        dict(
                            row=child_row,
                            policy=recipe,
                            manifest=_record(manifest_path),
                            allocated=split_allocation,
                            contract=child_packet["attempts"][0]["observation_scope"],
                        )
                    )
                split_freeze = dict(
                    schema="s6_failed_component_split_execution_v1",
                    parent_component_index=index,
                    trigger=(
                        record["result"].get("status")
                        if record["result"].get("status") in _FAILURE_STATUSES
                        else ("search_exhausted" if _search_exhausted(record) else "rejection")
                    ),
                    remaining_wall_seconds=remaining_wall,
                    remaining_search_seconds=remaining_search,
                    unaccepted_local_flights=list(unaccepted),
                    jobs=freeze_jobs,
                )
                freeze_path = destination / f"split_fallback_{index}.json"
                stage.save(freeze_path, split_freeze)
                for split_job in freeze_jobs:
                    local = split_job["allocated"]["local_flight_index"]
                    cap = min(
                        split_job["allocated"]["wall_seconds"],
                        timeout - (time.monotonic() - started),
                    )
                    split_before = time.monotonic()
                    split_out = destination / "split_fallback" / f"{index}_{local}"
                    if not split_job["allocated"]["executable"] or cap <= 0:
                        split_result = dict(
                            status="source_budget_exhausted",
                            verdict={},
                            reason=split_job["allocated"]["reason"] or "source_deadline",
                        )
                    else:
                        try:
                            split_result = execute(
                                split_job["row"], split_out, split_job["policy"], cap
                            )
                        except Exception as error:
                            split_result = dict(
                                status="execution_failed",
                                verdict={},
                                reason=f"{type(error).__name__}: {error}",
                            )
                    split_result.setdefault("key", split_job["row"]["key"])
                    split_result.setdefault("surface", split_job["row"]["surface"])
                    split_result.setdefault(
                        "declared_flights", split_job["row"]["declared_flights"]
                    )
                    split_path = split_out / "cases" / split_job["row"]["key"] / "result.json"
                    split_rejected = None
                    try:
                        if split_result["key"] != split_job["row"]["key"]:
                            raise ValueError("split result key differs from its execution job")
                        checked_split_emissions(plan, split_job["contract"], split_result)
                    except (ValueError, KeyError, TypeError) as error:
                        rejected_path = split_path.with_name("invalid_emission_result.json")
                        stage.save(rejected_path, split_result)
                        split_rejected = _record(rejected_path)
                        split_result = dict(
                            key=split_job["row"]["key"],
                            surface=split_job["row"]["surface"],
                            declared_flights=split_job["row"]["declared_flights"],
                            status="invalid_component_emission",
                            reason=str(error),
                            verdict=dict(
                                complete_point=False, accepted_flight_count=0, flight_count=0
                            ),
                        )
                    if split_rejected is not None or not split_path.exists():
                        stage.save(split_path, split_result)
                    search_path = split_path.parent / "search/report.json"
                    try:
                        split_search = (
                            json.loads(search_path.read_text()).get("search_budget")
                            if search_path.is_file()
                            else None
                        )
                    except (OSError, ValueError) as error:
                        split_search = dict(
                            status="unavailable_incomplete_report", reason=str(error)
                        )
                    split_records.append(
                        dict(
                            component_index=index,
                            local_flight_index=local,
                            original_flight_index=split_job["allocated"]["original_flight_index"],
                            contract=split_job["contract"],
                            result=split_result,
                            result_binding=_record(split_path),
                            manifest=split_job["manifest"],
                            allocated=split_job["allocated"],
                            executed_timeout_seconds=max(0.0, cap),
                            rejected_emission=split_rejected,
                            freeze=_record(freeze_path),
                            consumed=dict(
                                wall_seconds=time.monotonic() - split_before,
                                search_budget=split_search,
                                total_optimizer_iterations=None,
                                iteration_count_status="not reported by every solver branch",
                            ),
                        )
                    )
    result = dict(
        schema=SCHEMA,
        key=row["key"],
        surface=row["surface"],
        status="components_measured_requires_native_review",
        declared_flights=row["declared_flights"],
        cold_search=True,
        runtime_model_calls=0,
        source_plan=plan,
        source_execution=_record(destination / "component_execution.json"),
        component_results=records,
        verdict=aggregate(plan, records, seed_failure=seed_failure, split_records=split_records),
        **(
            {
                "split_fallback": dict(
                    schema="s6_failed_component_split_v1",
                    records=split_records,
                )
            }
            if split_records
            else {}
        ),
        source_budget=budget,
        preparation_wall_seconds=preparation_seconds,
        **({"court_paint_cache": paint_cache} if paint_cache is not None else {}),
        wall_seconds=time.monotonic() - started,
        wall_accounting="source validation through aggregation; final JSON serialization excluded",
        complete_original_source=False,
        continuity_between_components=False,
        reference_rows_consulted=False,
        **(
            {
                "seed_failure_fallback": True,
                "attempt_seed_failure": attempts.attempt_seed_failure_reason(seed_failure),
                "whole_point_seed_failure": deepcopy(whole_point),
            }
            if seed_failure is not None
            else {}
        ),
    )
    stage.save(destination / "result.json", result)
    return result


def validate_result(
    result: dict, *, row: dict | None = None, policy: dict | None = None, verify_files: bool = True
) -> dict:
    """Rebuild source plan, allocation, roster and aggregate without fitting."""
    if result.get("schema") != SCHEMA:
        raise ValueError("component source aggregate required")
    plan = result["source_plan"]
    records = result["component_results"]
    if [r["component_index"] for r in records] != list(range(len(plan["components"]))):
        raise ValueError("exactly one result slot per original prepared component required")
    split_records = (result.get("split_fallback") or {}).get("records") or []
    if (
        result["verdict"]
        != aggregate(
            plan,
            records,
            seed_failure=result.get("attempt_seed_failure"),
            split_records=split_records,
        )
        or result.get("complete_original_source") is not False
    ):
        raise ValueError("aggregate differs from original component emissions")
    if result.get("continuity_between_components") is not False:
        raise ValueError("disconnected components cannot assert continuity")
    if not verify_files:
        return result

    def read(binding):
        path = stage.resolve(binding)
        if provenance.file_sha256(path) != binding["sha256"]:
            raise ValueError("component execution artifact changed")
        return json.loads(path.read_text())

    execution = read(result["source_execution"])
    row = execution["row"] if row is None else row
    policy = execution["policy"] if policy is None else policy
    if (
        execution["row"] != row
        or execution["policy"] != policy
        or result["declared_flights"] != row["declared_flights"]
    ):
        raise ValueError("component aggregate differs from original source manifest")
    paths = stage.validate_row(row)
    originals = [json.loads(paths[k].read_text()) for k in ("packet", "labels", "cameras")]
    if result.get("seed_failure_fallback"):
        originals[0] = read(execution["seed_failure_source_packet"])
    preparation.validate(
        plan,
        *originals,
        observation_partition=stage.shared_settings(policy)["observation_partition"],
        contact_component_routing=stage.shared_settings(policy).get(
            preparation.ROUTING_FIELD, preparation.ROUTING_DEFAULT
        ),
    )
    if read(execution["plan"]) != plan or read(execution["budget"]) != result["source_budget"]:
        raise ValueError("source plan or budget differs from before-fit freeze")
    expected_budget = allocate(plan, policy, result["source_budget"]["source_wall_seconds"])
    if result["source_budget"] != expected_budget or len(execution["jobs"]) != len(records):
        raise ValueError("component budget or job roster changed")
    for index, (job, record) in enumerate(zip(execution["jobs"], records)):
        child_packet, child_labels = scope.bind(plan, *originals, index)
        if job["component_index"] != index or job["policy"] != child_policy(
            policy, child_packet, expected_budget, index
        ):
            raise ValueError("component recipe differs from frozen source work allocation")
        component = plan["components"][index]
        cap = record["executed_timeout_seconds"]
        expected_applicability = scope.applied_policy(policy, child_packet)[1]
        wall_cap = expected_budget["components"][index]["wall_seconds"]
        if record.get("exhausted_retry"):
            wall_cap = expected_budget["source_wall_seconds"]
        if (
            record["local_to_original"] != component["local_to_original"]
            or record["result"].get("key") != job["row"]["key"]
            or not math.isfinite(cap)
            or not 0 <= cap <= wall_cap
            or record["result"].get("execution_timeout_seconds", cap) != cap
            or record["source_scope_applicability"] != expected_applicability
            or job["source_scope_applicability"] != expected_applicability
        ):
            raise ValueError(
                "component result identity or executed work differs from frozen source"
            )
        if record["rejected_emission"] is not None:
            rejected = read(record["rejected_emission"])
            try:
                if rejected.get("key") != job["row"]["key"]:
                    raise ValueError("wrong component result key")
                checked_emissions(plan, index, rejected)
            except (ValueError, KeyError, TypeError):
                pass
            else:
                raise ValueError("valid component emission cannot be discarded after fitting")
        manifest = read(record["manifest"])
        if (
            record["manifest"] != job["manifest"]
            or manifest["rows"] != [job["row"]]
            or manifest["policy"] != job["policy"]
        ):
            raise ValueError("component manifest differs from original execution roster")
        inputs = stage.validate_row(job["row"])
        if json.loads(inputs["labels"].read_text()) != child_labels:
            raise ValueError("component label derivation changed")
        child_packet["external_label_binding"] = dict(record=job["row"]["labels"])
        if json.loads(inputs["packet"].read_text()) != child_packet:
            raise ValueError("component packet derivation changed")
        if (
            read(record["result_binding"]) != record["result"]
            or record["allocated"] != expected_budget["components"][index]
        ):
            raise ValueError("component result or allocation differs from original emission")
    if split_records:
        if stage.shared_settings(policy).get(preparation.SPLIT_FIELD) != preparation.SPLIT_ON:
            raise ValueError("split fallback records require the declared split policy")
        for record in split_records:
            expected_packet, _ = scope.bind_split(
                plan,
                *originals,
                record["component_index"],
                record["local_flight_index"],
            )
            if record["contract"] != expected_packet["attempts"][0]["observation_scope"]:
                raise ValueError("split contract differs from original flight binding")
            if read(record["result_binding"]) != record["result"]:
                raise ValueError("split result differs from original emission")
    return result
