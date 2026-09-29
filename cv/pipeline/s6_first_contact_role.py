"""Explicit original rally origin; absent/unspecified retains legacy serve semantics.

This module does not split attempts. The default binding requires an explicit
source-typed rally contact. The separate single-attempt route checks structured
original membership and an earlier original serve; cross-attempt index alone is
insufficient. Neither route rewrites the original event records.

A third bound role, ``unknown``, states that the source cannot resolve the
origin of this contact, so it is neither a demonstrated serve nor a demonstrated
rally stroke. It disables the serve-specific priors explicitly and keeps the
widest venue contact envelope -- the union of both alternatives -- rather than
asserting either one. Two source facts derive it: leading physical evidence
before the first observed contact, or a later original contact in a source that
never declares a serve, where the single-attempt rally chain cannot be walked.
"""

from argparse import Namespace
from copy import deepcopy
import math

SCHEMA = "s6_original_rally_contact_role_v1"
UNKNOWN_SCHEMA = "s6_original_unknown_contact_role_v1"
ROLES = ("unspecified", "serve", "rally", "unknown")
#: Roles derived from original observations and carried by a replayable receipt.
BOUND_ROLES = ("rally", "unknown")
FIELD = "first_contact_role_evidence"
EXPLICIT_RALLY = "explicit_source_rally_role_at_later_original_contact"
SINGLE_ATTEMPT = "single_original_attempt_after_original_serve"
LEADING_PREFIX = "leading_original_physical_evidence_before_first_observed_contact"
#: A later original contact in a source that never declares a serve. The chain
#: back to the first observed contact cannot be walked, so neither the serve nor
#: the rally route applies and the origin stays unknown.
UNDECLARED_SERVE_ORIGIN = "later_original_contact_in_a_source_without_a_declared_serve"
UNKNOWN_DERIVATIONS = (LEADING_PREFIX, UNDECLARED_SERVE_ORIGIN)
#: The closed source-derived alternative set for an unobserved origin. It is
#: read from the row types already present, never from a per-point judgement.
UNKNOWN_ORIGIN_HYPOTHESES = (
    "unobserved_contact_before_the_leading_physical_evidence",
    "attempt_origin_outside_the_supplied_native_window",
    "leading_physical_evidence_is_a_producer_false_positive",
)
#: The closed alternative set for a later original contact whose own origin is
#: unresolved because no original contact declares a serve.
UNDECLARED_SERVE_HYPOTHESES = (
    "unobserved_contact_inside_the_preceding_original_span",
    "a_preceding_original_physical_row_is_a_producer_false_positive",
    "the_preceding_original_span_ends_one_attempt_and_this_contact_opens_another",
)
SERVE_SETTINGS = (
    "toss_player_prior",
    "independent_toss_horizontal_prior",
    "serve_ground_normal",
    "joint_toss_requalification",
    "earlier_toss_support",
    "net_first_contact_toss",
    "prefix_net_revisit",
)
SERVE_SEARCH_SETTINGS = (
    "serve_reach_cut",
    "serve_toss_constraint",
    "serve_start_prior",
    "serve_contact_hypotheses",
    "serve_contact_epoch",
    "serve_ending_consistency",
)


def _events(events: list[dict]) -> list[dict]:
    return [{k: deepcopy(v) for k, v in e.items() if k != "clip"} for e in events]


def _first(attempt: dict) -> dict:
    first = next((e for e in attempt["events"] if e["event_type"] == "contact"), None)
    if first is None:
        raise ValueError("original contact identity required")
    return first


def source_record(event: dict) -> dict:
    """One declared producer wrapper, not recursive note/ancestry inference."""
    original = event.get("original_event", {}).get("original")
    if original is None:
        return event
    if not isinstance(original, dict) or original.get("event_type") != event["event_type"]:
        raise ValueError("original event wrapper must preserve physical type")
    return original


def supported(event: dict) -> bool:
    """Use existing occurrence admission; uncertainty about timing is separate."""
    from cv.experiments.connected_shooting.labeled_event_occurrence import resolved_membership

    if event.get("occurrence_status") in ("uncertain", "ambiguous", "absent", "unsupported"):
        return False
    if event.get("status") == "confirmed":
        return event.get("occurrence_status") in (None, "supported", "confirmed") and event.get(
            "occurrence_supported"
        ) not in (False, "partial")
    return resolved_membership(event, default=None)


def typed_serve(event: dict) -> bool:
    """Only structured source declarations qualify; free-text notes are ignored."""
    records = [event, source_record(event)]
    roles = [
        str(e[k]).lower()
        for e in records
        for k in ("role", "stroke", "shot_type")
        if e.get(k) is not None
    ]
    declared = (
        any(
            e.get("serve_number") in (1, 2)
            or e.get("phase") in ("serve", "serve_attempt_1", "serve_attempt_2")
            for e in records
        )
        or "serve" in roles
    )
    if declared and any(value != "serve" for value in roles):
        raise ValueError("contradictory original contact role declarations")
    return declared


def single_attempt_membership(original: list[dict]) -> dict:
    """Every original physical record must explicitly identify the same delivery."""
    identities = []
    for event in original:
        if event.get("event_type") not in ("contact", "bounce", "net_hit"):
            raise ValueError("original physical events required for single-attempt role")
        source = source_record(event)
        identity = source.get("attempt_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("explicit original attempt membership required for every event")
        if event.get("attempt_id", identity) != identity:
            raise ValueError("event wrapper contradicts original attempt membership")
        identities.append(identity)
    if not identities or len(set(identities)) != 1:
        raise ValueError("one original attempt required; cannot infer rally across attempts")
    return {"original_attempt_id": identities[0], "original_event_count": len(original)}


def bind_rally(
    attempt: dict, original_events: list[dict], *, derivation: str = EXPLICIT_RALLY
) -> dict:
    """Bind a confirmed later original contact, without inference or fit inputs."""
    from cv.experiments.connected_shooting.observation_scope import event_digest

    original = _events(original_events)
    first = _events([_first(attempt)])[0]
    contacts = [e for e in original if e["event_type"] == "contact"]
    indices = [i for i, e in enumerate(contacts) if e == first]
    if len(indices) != 1 or indices[0] <= 0:
        raise ValueError("rally role requires a later original contact identity")
    membership = None
    if derivation == EXPLICIT_RALLY:
        typed = [first[k] for k in ("role", "stroke", "shot_type") if first.get(k) is not None]
        if (
            not typed
            or any(value != "rally" for value in typed)
            or first.get("serve_number") is not None
        ):
            raise ValueError(
                "rally origin requires explicit original rally role and no conflicting serve declaration"
            )
        typed_serve(first)  # Refuse a contradictory structured original serve wrapper.
    elif derivation == SINGLE_ATTEMPT:
        membership = single_attempt_membership(original)
        if not supported(contacts[0]) or not typed_serve(contacts[0]):
            raise ValueError(
                "single-attempt rally role requires an occurrence-supported original serve"
            )
        if any(typed_serve(e) for e in contacts[1 : indices[0] + 1]):
            raise ValueError("later typed serve is a barrier to original rally-role derivation")
    else:
        raise ValueError("supported original rally-role derivation required")
    if not supported(first):
        raise ValueError("rally origin requires an occurrence-supported source contact")
    frames = [float(e["frame"]) for e in original]
    if not frames or not all(math.isfinite(f) for f in frames) or frames != sorted(frames):
        raise ValueError("ordered finite original events required for rally role")
    if len({(e["frame"], e["event_type"]) for e in original}) != len(original):
        raise ValueError("unique original physical events required")
    for e in original:
        interval = e.get("frame_interval", [])
        if (
            len(interval) != 2
            or not all(math.isfinite(float(f)) for f in interval)
            or not interval[0] <= e["frame"] <= interval[1]
        ):
            raise ValueError("original event requires its finite timing interval")
    result = deepcopy(attempt)
    result["first_contact_role"] = "rally"
    result[FIELD] = dict(
        schema=SCHEMA,
        role="rally",
        original_contact_index=indices[0],
        original_events=original,
        original_events_sha256=event_digest(original),
        first_contact=first,
        derivation=derivation,
        **({"single_attempt_membership": membership} if membership is not None else {}),
        match_id=attempt["match_id"],
        clip=attempt["point_clip"],
        fitted_state_consulted=False,
    )
    return result


def leading_physical_prefix(original: list[dict]) -> list[dict]:
    """Original physical rows that precede the first observed contact, verbatim."""
    first = next((e for e in original if e["event_type"] == "contact"), None)
    if first is None:
        raise ValueError("original observed contact required")
    return [e for e in original if e["frame"] < first["frame"]]


def preceding_original_span(original: list[dict], first: dict) -> dict | None:
    """Describe the original span that closes on this contact, from rows alone.

    Returns ``None`` for the first original contact. The description is derived
    from the retained physical rows, so a receipt replays from them; it is not a
    judgement about which row is wrong.
    """
    contacts = [e for e in original if e["event_type"] == "contact"]
    indices = [i for i, e in enumerate(contacts) if e == first]
    if len(indices) != 1:
        raise ValueError("one original observed contact identity required")
    if indices[0] == 0:
        return None
    previous = contacts[indices[0] - 1]
    between = [
        e
        for e in original
        if float(previous["frame"]) < float(e["frame"]) < float(first["frame"])
        and e["event_type"] in ("bounce", "net_hit")
    ]
    bounces = [e for e in between if e["event_type"] == "bounce"]
    return dict(
        previous_original_contact_frame=float(previous["frame"]),
        previous_original_contact_occurrence_supported=bool(supported(previous)),
        ground_event_frames=[float(e["frame"]) for e in between],
        ground_event_count=len(between),
        supplied_bounce_frames=[float(e["frame"]) for e in bounces],
        # The unchanged single-flight rule: at most one supplied ground impact.
        # A net cord alongside one bounce is an ordinary competitive flight.
        single_competitive_flight_representable=len(bounces) <= 1,
    )


def bind_unknown(
    attempt: dict, original_events: list[dict], *, derivation: str = LEADING_PREFIX
) -> dict:
    """Bind an unknown origin proved by original rows, not by a guess."""
    from cv.experiments.connected_shooting.observation_scope import event_digest

    if derivation not in UNKNOWN_DERIVATIONS:
        raise ValueError("supported original unknown-role derivation required")
    original = _events(original_events)
    first = _events([_first(attempt)])[0]
    contacts = [e for e in original if e["event_type"] == "contact"]
    indices = [i for i, e in enumerate(contacts) if e == first]
    if len(indices) != 1:
        raise ValueError("unknown role requires one original observed contact identity")
    prefix = leading_physical_prefix(original)
    span = None
    if derivation == LEADING_PREFIX:
        if not prefix:
            raise ValueError("unknown role requires leading original physical evidence")
        if not any(supported(e) for e in prefix):
            # Unresolved leading rows are an occurrence barrier for the slot that
            # owns them, not evidence that an earlier contact was played.
            raise ValueError(
                "unknown role requires one occurrence-supported leading physical event"
            )
    else:
        if indices[0] == 0:
            raise ValueError("an undeclared-serve unknown origin requires a later original contact")
        if any(typed_serve(e) for e in contacts):
            # Some original contact declares a serve, so the declared serve and
            # single-attempt rally routes own this source instead.
            raise ValueError(
                "a declared original serve resolves this source; unknown role does not apply"
            )
        span = preceding_original_span(original, first)
    if typed_serve(contacts[0]):
        # An observed original serve resolves the attempt origin; the ordinary
        # single-attempt rally derivation owns every later contact.
        raise ValueError("an original typed serve resolves the origin; unknown role does not apply")
    typed = [first[k] for k in ("role", "stroke", "shot_type") if first.get(k) is not None]
    if typed or first.get("serve_number") is not None:
        raise ValueError(
            "a typed original contact is not unknown; use its declared serve/rally route"
        )
    if not supported(first):
        raise ValueError("unknown origin requires an occurrence-supported source contact")
    frames = [float(e["frame"]) for e in original]
    if not frames or not all(math.isfinite(f) for f in frames) or frames != sorted(frames):
        raise ValueError("ordered finite original events required for unknown role")
    if len({(e["frame"], e["event_type"]) for e in original}) != len(original):
        raise ValueError("unique original physical events required")
    for e in original:
        interval = e.get("frame_interval", [])
        if (
            len(interval) != 2
            or not all(math.isfinite(float(f)) for f in interval)
            or not interval[0] <= e["frame"] <= interval[1]
        ):
            raise ValueError("original event requires its finite timing interval")
    result = deepcopy(attempt)
    result["first_contact_role"] = "unknown"
    result[FIELD] = dict(
        schema=UNKNOWN_SCHEMA,
        role="unknown",
        role_candidates=["serve", "rally"],
        origin_hypotheses=list(
            UNKNOWN_ORIGIN_HYPOTHESES
            if derivation == LEADING_PREFIX
            else UNDECLARED_SERVE_HYPOTHESES
        ),
        origin_resolved=False,
        serve_priors_applicable=False,
        original_contact_index=indices[0],
        leading_physical_prefix=deepcopy(prefix),
        leading_physical_prefix_count=len(prefix),
        **({"preceding_original_span": span} if span is not None else {}),
        original_events=original,
        original_events_sha256=event_digest(original),
        first_contact=first,
        derivation=derivation,
        match_id=attempt["match_id"],
        clip=attempt["point_clip"],
        fitted_state_consulted=False,
    )
    return result


def _bound(attempt: dict, receipt: dict) -> dict:
    """Recompute the declared receipt from its own original records."""
    role = receipt.get("role")
    if role == "rally":
        return bind_rally(
            attempt, receipt.get("original_events", []), derivation=receipt.get("derivation")
        )[FIELD]
    if role == "unknown":
        return bind_unknown(
            attempt, receipt.get("original_events", []), derivation=receipt.get("derivation")
        )[FIELD]
    raise ValueError("explicit supported bound first-contact role required")


def validate(
    attempt: dict,
    labels: dict | None = None,
    report: dict | None = None,
    requested: str | None = None,
) -> dict | None:
    """Recompute role at consumer boundaries. Without labels, check shape/digest only.

    The stage and file replay/export must also pass the original label document;
    in-memory state validation alone cannot establish source provenance.
    """
    role = attempt.get("first_contact_role", "unspecified")
    if role is None:
        role = "unspecified"
    if role not in ROLES or requested is not None and requested not in ROLES:
        raise ValueError("explicit supported first-contact role required")
    receipt = attempt.get(FIELD)
    if role not in BOUND_ROLES:
        if receipt is not None or requested in BOUND_ROLES:
            raise ValueError("rally policy requires bound original rally-contact evidence")
        if report is not None and report.get("configuration", {}).get(FIELD) is not None:
            raise ValueError("search rally role has no original input binding")
        return None
    if requested is not None and requested != role:
        raise ValueError("bound rally origin requires explicit rally policy")
    if not isinstance(receipt, dict):
        raise ValueError("rally origin requires original contact role evidence")
    if receipt.get("role") != role:
        raise ValueError("bound role evidence differs from the declared first-contact role")
    expected = _bound(attempt, receipt)
    if receipt != expected:
        raise ValueError("rally contact role does not replay from original events")
    if labels is not None:
        source = _events(
            [
                e
                for e in labels["events"]["records"]
                if e.get("clip", attempt["point_clip"]) == attempt["point_clip"]
            ]
        )
        original = receipt["original_events"]
        low, high = original[0]["frame"], original[-1]["frame"]
        if [
            e
            for e in source
            if e.get("event_type") in ("contact", "bounce", "net_hit") and low <= e["frame"] <= high
        ] != original:
            raise ValueError("rally original inventory differs from source labels")
    if report is not None:
        if report.get("configuration", {}).get(FIELD) != receipt:
            raise ValueError("search differs from original rally role")
        if _events([_first(report)])[0] != receipt["first_contact"]:
            raise ValueError("search changed original rally-contact identity")
    return receipt


def apply_players(players: list[dict], receipt: dict | None) -> list[dict]:
    if receipt is not None:
        players[0][FIELD] = deepcopy(receipt)
    return players


def rally(players: list[dict] | None) -> bool:
    return bool(players and (players[0].get(FIELD) or {}).get("role") == "rally")


def bound_role(players: list[dict] | None) -> str:
    """The role a bound receipt declares; an unbound origin stays unspecified."""
    role = bool(players) and (players[0].get(FIELD) or {}).get("role")
    return role if role in BOUND_ROLES else "unspecified"


def effective_role(receipt: dict | None) -> str:
    role = (receipt or {}).get("role")
    return role if role in BOUND_ROLES else "unspecified"


def serve_priors_applicable(players: list[dict] | None) -> bool:
    """A demonstrated rally origin and an unknown origin both refuse serve priors."""
    return bound_role(players) == "unspecified"


def abstention_reason(role: str) -> str:
    """Name the actual origin evidence; an unknown origin is not a rally claim."""
    if role == "rally":
        return "original_rally_contact"
    if role == "unknown":
        return "unknown_original_first_contact_origin"
    raise ValueError("explicit supported bound first-contact role required")


def limits(player: dict) -> tuple[float, float]:
    """Rally uses the existing ordinary contact venue envelope, never serve depth.

    An unknown origin uses the same envelope because it is the union of the
    serve and rally alternatives, not because the contact is known to be a rally.
    """
    from cv.experiments.connected_shooting import agent_single_flight_search as single
    from cv.experiments.connected_shooting.event_constraints import CONTACT_XY_ENVELOPE_M

    if serve_priors_applicable([player]):
        return single.limits(player["side"])[:2]
    return CONTACT_XY_ENVELOPE_M[1]


def applied_policy(
    policy: dict, attempt: dict, labels: dict | None = None
) -> tuple[dict, dict | None]:
    if attempt.get("first_contact_role") in BOUND_ROLES and labels is None:
        raise ValueError("original source labels required to apply rally policy")
    receipt = validate(attempt, labels, requested=policy.get("first_contact_role", "unspecified"))
    if receipt is None:
        return policy, None
    role = receipt["role"]
    result = dict(policy)
    changes = {}
    for name in (*SERVE_SETTINGS, *SERVE_SEARCH_SETTINGS):
        if result.get(name, "off") != "off":
            changes[name] = dict(
                requested=result[name], applied="off", reason=abstention_reason(role)
            )
        result[name] = "off"
    return result, dict(requested_role=role, effective_role=role, source=receipt, changes=changes)


def check_search_options(args: Namespace) -> None:
    """Both bound roles refuse serve-only mechanisms and serve-only evidence."""
    if any(getattr(args, name, "off") != "off" for name in SERVE_SEARCH_SETTINGS):
        raise ValueError("original rally contact cannot consume serve-only numerical mechanisms")
    if any(
        getattr(args, name, None) is not None
        for name in ("serve_speed", "serve_contact_history", "toss_labels")
    ):
        raise ValueError("original rally contact cannot consume serve-only external evidence")
    if getattr(args, "depth_hypothesis_m", None):
        raise ValueError(
            "rally initialization uses original observations, not caller depth choices"
        )


def mark_result(result: dict, receipt: dict | None) -> dict:
    if receipt is not None:
        result[FIELD] = deepcopy(receipt)
    return result
