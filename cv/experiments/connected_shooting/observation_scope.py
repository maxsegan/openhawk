"""Explicit native observation horizon without a claimed physical point ending.

The optional adapter supplies this contract from declared clip scope and automatic
physical events. It is not a line call, missing-event repair, or completeness label.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

#: An unresolved tail must be an observed flight, not a couple of stray fronts.
MINIMUM_VISIBLE_TAIL_ROWS = 5
SCHEMA = "s6_observed_ground_scope_v1"
AUTOMATIC_SCHEMA = "s6_observed_ground_scope_v2"
PHYSICAL = {"contact", "bounce", "net_hit"}


class UnsupportedTailTopology(ValueError):
    """Valid original events need a different explicitly enabled scope family."""


class UnsupportedOriginTopology(ValueError):
    """The original stream opens on physical evidence that is not its first contact.

    A separate class from the tail refusal: the leading rows are valid original
    observations, so a caller with an explicit policy can own them as an
    unresolved input inventory instead of dropping them. It stays a ``ValueError``
    so every existing handler keeps refusing exactly as before.
    """


def event_digest(events: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _segmentation_source(source: dict | None) -> dict | None:
    """Bind scope identity; automatic ancestry is checked by the producer adapter."""
    if source is None:
        return None
    if not isinstance(source, dict) or set(source) != {"kind", "record"}:
        raise ValueError("explicit automatic point-ledger binding required")
    record = source["record"]
    if source["kind"] != "automatic_point_ledger" or not isinstance(record, dict):
        raise ValueError("automatic point-ledger scope origin required")
    if set(record) - {"path_base", "path", "sha256", "bytes"}:
        raise ValueError("unsupported segmentation source fields")
    path = record.get("path")
    if (
        record.get("path_base") != "TENNIS_DATA_ROOT"
        or not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        or ("bytes" in record and (type(record["bytes"]) is not int or record["bytes"] <= 0))
    ):
        raise ValueError("bound shared-data segmentation source required")
    return deepcopy(source)


def qualify(
    events: list[dict],
    window: tuple[int, int] | list[int],
    *,
    segmentation_source: dict | None = None,
    allow_terminal_net: bool = False,
    allow_observed_horizon: bool = False,
    event_origin: str | None = None,
) -> dict:
    """Qualify an observed ground tail; never derive an event from the horizon."""
    segmentation_source = _segmentation_source(segmentation_source)
    if event_origin not in {None, "label", "automatic"}:
        raise ValueError("explicit supported physical event origin required")
    if len(window) != 2 or any(type(f) is not int for f in window) or window[0] >= window[1]:
        raise ValueError("ordered native observation window required")
    if not events or any(e.get("event_type") not in PHYSICAL for e in events):
        raise ValueError("observation scope requires original physical events")
    pairs = [(e["frame"], e["event_type"]) for e in events]
    if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
        raise ValueError("unique ordered original physical events required")
    for e in events:
        lo, hi = e["frame_interval"]
        if (
            not all(math.isfinite(float(x)) for x in (e["frame"], lo, hi))
            or not window[0] <= lo <= e["frame"] <= hi <= window[1]
        ):
            raise ValueError("entire original event interval must lie in observation scope")
    contacts = [e for e in events if e["event_type"] == "contact"]
    if not contacts or events[0]["event_type"] != "contact":
        raise UnsupportedOriginTopology(
            "observation scope needs an originating contact; prior events cannot be dropped"
        )
    tail = [e for e in events if e["frame"] > contacts[-1]["frame"]]
    net_tail = bool(allow_terminal_net and tail and tail[0]["event_type"] == "net_hit")
    ground_tail = tail[1:] if net_tail else tail
    # An unresolved visible tail has an UNKNOWN ground count, not zero. The
    # count is only admitted as unknown under the explicit opt-in; without it
    # the original "one or two supplied grounds" contract is unchanged.
    unresolved = bool(allow_observed_horizon and not ground_tail)
    if not unresolved and (
        len(ground_tail) not in (1, 2) or any(e["event_type"] != "bounce" for e in ground_tail)
    ):
        raise UnsupportedTailTopology(
            "observation scope needs one or two supplied ground events after last contact"
        )
    if tail and tail[-1]["frame_interval"][1] >= window[1]:
        raise UnsupportedTailTopology(
            "observation scope requires a horizon after the entire ground interval"
        )
    if unresolved and contacts[-1]["frame_interval"][1] >= window[1]:
        raise UnsupportedTailTopology(
            "observation scope requires a horizon after the entire contact interval"
        )
    contract = dict(
        schema=SCHEMA,
        native_window=list(window),
        observation_horizon=float(window[1]),
        physical_ending=None,
        ending_semantics="unresolved",
        supplied_ground_count=len(ground_tail),
        **(
            {
                "terminal_kind": "observed_horizon",
                "terminal_ground_count": "unknown",
                "latent_ground_origin": "model_latent_not_source_event",
            }
            if unresolved
            else {}
        ),
        **({"terminal_net_preserved": True} if net_tail else {}),
        original_events_sha256=event_digest(events),
        original_event_count=len(events),
        source="explicit supplied clip scope and automatic physical events",
        segmentation_assistance="supplied development clip window",
    )
    if segmentation_source is not None:
        contract.update(
            schema=AUTOMATIC_SCHEMA,
            source="automatic pipeline clip scope and automatic physical events",
            segmentation_assistance=None,
            segmentation_source=segmentation_source,
        )
    if event_origin is not None:
        contract.update(
            event_origin=event_origin,
            source=(
                "automatic pipeline clip scope and declared physical events"
                if segmentation_source is not None
                else "explicit supplied clip scope and declared physical events"
            ),
        )
    return contract


def supported_tail(events: list[dict], rows: list[dict], contract: dict) -> bool:
    """The one visible-tail support rule every scope reader applies.

    With supplied grounds this is rebound support after the ground interval.
    With an unknown terminal ground count the last original event is the
    contact, so the same rule reads as visible support after the contact
    interval, and the tail must be an observed flight rather than a stray front.
    """
    final_event = events[-1]
    horizon = contract["observation_horizon"]
    support = [
        r
        for r in rows
        if r["status"] == "visible" and final_event["frame_interval"][1] < r["frame"] <= horizon
    ]
    required = (
        MINIMUM_VISIBLE_TAIL_ROWS if contract.get("terminal_kind") == "observed_horizon" else 1
    )
    return len(support) >= required


def validate(attempt: dict, labels: dict | None = None, report: dict | None = None) -> dict | None:
    contract = attempt.get("observation_scope")
    if contract is None:
        if (
            report is not None
            and report.get("configuration", {}).get("observation_scope") is not None
        ):
            raise ValueError("search scope has no original packet contract")
        return None
    from cv.pipeline import s6_contact_prefix_scope, s6_contact_prefix_runtime
    from cv.pipeline import s6_component_scope

    if s6_component_scope.active(attempt):
        return s6_component_scope.validate(attempt, labels, report)

    if contract.get("schema") == s6_contact_prefix_scope.UNRESOLVED_INPUT_SCHEMA:
        # A retained original inventory that no contract owns yet. It replays
        # structurally, but it declares no topology, so a search report can
        # never be validated against it.
        if report is not None:
            raise ValueError("a retained unresolved inventory cannot own a search report")
        return s6_contact_prefix_scope.validate_unresolved_input(attempt, labels)

    if contract.get("schema") == s6_contact_prefix_scope.SCHEMA:
        checked = s6_contact_prefix_scope.validate(
            attempt, observation_partition=contract["observation_partition"]
        )
        s6_contact_prefix_runtime.validate_labels(attempt, labels)
        if report is not None:
            if report.get("configuration", {}).get("observation_scope") != checked:
                raise ValueError("search differs from original contact-prefix scope")
            if checked.get("mode") == s6_contact_prefix_scope.TERMINAL_IDENTITY:
                from cv.pipeline import s6_optional_contacts

                selected = s6_optional_contacts.validate_report_events(attempt, report)
                if report.get("events") != attempt["events"] and not selected:
                    raise ValueError("search differs from original contact-prefix scope")
                s6_contact_prefix_scope.checked_interior_events(
                    checked, attempt["events"], report["events"]
                )
            elif report.get("events") != attempt["events"]:
                raise ValueError("search differs from original contact-prefix scope")
        return checked
    from cv.pipeline import s6_optional_bounce_scope

    if contract.get("schema") == s6_optional_bounce_scope.SCHEMA:
        return s6_optional_bounce_scope.validate(attempt, labels, report)
    from cv.pipeline import s6_first_flight_scope as prefix

    is_prefix = contract.get("schema") == prefix.SCHEMA
    if is_prefix:
        original = attempt.get("original_physical_events", [])
        expected = prefix.qualify(
            original,
            contract.get("native_window", []),
            attempt.get("ball_observation_operator"),
            attempt.get("segmentation_source"),
        )
        modeled = [e for e in original if e.get("status") == "predicted"][:2]
        if attempt["events"] != modeled:
            raise ValueError("modeled prefix differs from original physical membership")
    else:
        expected = qualify(
            attempt["events"],
            contract.get("native_window", []),
            segmentation_source=attempt.get("segmentation_source"),
            allow_terminal_net=contract.get("terminal_net_preserved") is True,
            allow_observed_horizon=contract.get("terminal_kind") == "observed_horizon",
            event_origin=contract.get("event_origin"),
        )
        if contract.get("event_origin") is not None:
            if attempt.get("stream_origins", {}).get("events") != contract["event_origin"]:
                raise ValueError("scope event origin differs from packet stream origin")
            if (
                labels is not None
                and labels.get("stream_origins", {}).get("events") != contract["event_origin"]
            ):
                raise ValueError("scope event origin differs from consumer stream origin")
    if contract != expected:
        raise ValueError("observation scope differs from its original physical events/window")
    if attempt.get("ending_supplied") is not False or attempt.get("point_end") is not None:
        raise ValueError("observation horizon cannot assert a physical ending")
    if (
        attempt.get("owner_end_frame") != contract["observation_horizon"]
        or attempt.get("owner_end_frame_semantics") != "observation_horizon_not_physical_event"
    ):
        raise ValueError("legacy end-frame alias must explicitly denote observation horizon")
    if labels is not None:
        meta = labels["attempt"]
        if meta.get("segmentation_source") != attempt.get("segmentation_source"):
            raise ValueError("consumer segmentation source differs from packet")
        if (
            meta.get("observation_scope") != contract
            or "ending_frame" in meta
            or meta.get("ending_kind") != "unresolved"
        ):
            raise ValueError("consumer document must retain unresolved scope, not an ending epoch")
        source_events = labels["events"]["records"]
        expected_events = original if is_prefix else attempt["events"]
        if any(
            event.get("clip", attempt["point_clip"]) != attempt["point_clip"]
            for event in [*source_events, *expected_events]
        ):
            raise ValueError("scope physical events differ from their bound clip")
        source = [{k: v for k, v in e.items() if k != "clip"} for e in source_events]
        expected_source = [{k: v for k, v in e.items() if k != "clip"} for e in expected_events]
        if source != expected_source:
            raise ValueError("original physical event inventory changed across scope adapter")
        rows = labels["ball"]["records"][0]["frames"]
        if [r["frame"] for r in rows] != list(
            range(contract["native_window"][0], contract["native_window"][1] + 1)
        ):
            raise ValueError(
                "complete native inventory, including absent ball observations, required"
            )
        if not supported_tail(attempt["events"], rows, contract):
            raise ValueError("observed ground scope requires visible native rebound support")
    if report is not None:
        from cv.pipeline import s6_optional_event_union

        optional_events = s6_optional_event_union.validate_report_events(attempt, report)
        if report.get("configuration", {}).get("observation_scope") != contract or (
            report.get("events") != attempt["events"] and not optional_events
        ):
            raise ValueError(
                "search must retain original observation scope and every physical event"
            )
    return contract


def horizon(meta: dict) -> float:
    contract = meta.get("observation_scope")
    return float(
        contract.get("modeled_horizon", contract.get("observation_horizon"))
        if contract
        else meta["ending_frame"]
    )


def _finite_prediction(row: dict) -> bool:
    try:
        prediction = row.get("predicted", [])
        return len(prediction) == 2 and all(math.isfinite(float(x)) for x in prediction)
    except (TypeError, ValueError):
        return False


def mark_verdict(
    verdict: dict,
    contract: dict | None,
    *,
    context: dict | None = None,
    measurement: dict | None = None,
) -> dict:
    """Keep old gates intact but do not call an unknown ending a legacy complete point."""
    if contract is None:
        return verdict
    result = deepcopy(verdict)
    from cv.pipeline import s6_contact_prefix_scope

    from cv.pipeline import s6_component_scope

    if contract.get("schema") == s6_component_scope.SCHEMA:
        result["contact_component_reconstruction"] = dict(
            scope=deepcopy(contract),
            geometry_gates_passed=bool(verdict.get("complete_point")),
            complete_original_source=False,
            local_to_original=deepcopy(contract["component"]["local_to_original"]),
            reference_flight_coverage="not_evaluated_here",
        )
        result["complete_point"] = False
        return result
    if contract.get("schema") == s6_contact_prefix_scope.SCHEMA:
        result["contact_prefix_reconstruction"] = {
            "scope": deepcopy(contract),
            "geometry_gates_passed": bool(verdict.get("complete_point")),
            "complete_original_source": False,
            "retained_original_flight_indices": contract["retained_flight_indices"],
            "unresolved_original_slots": deepcopy(contract["unresolved_original_slots"]),
            "reference_flight_coverage": "not_evaluated_here",
        }
        result["complete_point"] = False
        return result
    result["observed_scope_reconstruction"] = {
        "schema": contract["schema"],
        "scope": deepcopy(contract),
        "geometry_gates_passed": bool(verdict.get("complete_point")),
        "physical_ending_classified": False,
        "reference_flight_coverage": "not_evaluated_here",
        "complete_original_attempt": None,
    }
    if context is not None and measurement is not None:
        expected = sorted(
            r["frame"] for r in context["attempt"]["owner_ball_labels"] if r["status"] == "visible"
        )
        actual = sorted(
            {r["frame"] for r in measurement.get("native_projection", []) if _finite_prediction(r)}
        )
        missing = sorted(set(expected) - set(actual))
        scene = context["scene"]
        from cv.experiments.connected_shooting.observed_horizon_tail import modeled_horizon

        horizon_retained = float(scene.contact_frames[-1]) == modeled_horizon(
            context["attempt"], contract
        )
        result["observed_scope_reconstruction"]["native_coverage"] = {
            "required_visible_frames": expected,
            "projected_native_frames": actual,
            "missing_visible_frames": missing,
            "observation_horizon_retained": horizon_retained,
            "leading_context_is_not_scored_geometry": True,
            "all_modeled_visible_rows_projected": not missing and horizon_retained,
        }
    result["complete_point"] = False
    result["partial_point"] = bool(result.get("accepted_flight_count", 0))
    result["ending_semantics"] = "unresolved"
    return result
