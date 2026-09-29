"""Explicit membership of an automatically predicted terminal net interaction.

Default off. Under the explicit ``predicted`` policy the source's own automatic
terminal net prediction becomes an uncertain *model* hypothesis with two declared
branches over the same original inputs:

``present``
    the existing original-interval net ending, exactly as ``terminal_net_tail``
    already qualifies it;
``absent``
    the same attempt with that one source-bound net event, and the ending it
    derives, omitted from the *effective event view*. Every interior event, every
    native row, camera and epoch is retained; the terminal ground count becomes
    ``unknown`` under the existing observed-horizon contract and the physical
    ending is ``None``.

Nothing here edits a source record. The original net event, its derived ending,
the full original event stream digest and the original observation inventory are
bound into one receipt, so the omission stays a declared hypothesis that can be
recovered, replayed and refused rather than a repaired annotation.

Only an *automatic predicted* occurrence with ordinary automatic ancestry is
eligible. A human exact or reviewed occurrence is never demoted, and ambiguous
events in general are not ignored: exactly the one disputed source-bound net and
its coincident ending are authorized, by hash, in the absent branch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

SCHEMA = "s6_terminal_net_membership_v1"
#: The single authorization a lower-level helper may act on. It names exactly one
#: original automatic predicted terminal net and its coincident derived ending,
#: both bound by hash to the qualified contract they came from.
AUTHORIZATION_SCHEMA = "s6_terminal_net_membership_authorization_v1"
#: Attempt/report key carrying the validated derivation receipt.
RECEIPT_KEY = "terminal_net_membership"
#: Ordinary search receipt of the declared membership enumeration, written for
#: every membership run including the all-branches-failed path.
RECEIPT_FILE = "terminal_net_membership_search.json"
POLICIES = ("off", "predicted")
PRESENT = "present"
ABSENT = "absent"
MEMBERSHIPS = (PRESENT, ABSENT)
#: Declared selection family when no ordinary optional witness is prepared.
SELECTION_POLICY = "source_predicted_terminal_net_membership_v1"
#: Only this occurrence support may be treated as uncertain membership.
ELIGIBLE_OCCURRENCE_SUPPORT = "source_predicted"
#: Continuous parameters one additional contact adds, as the optional families count them.
PARAMETERS_PER_CONTACT = 6
#: The ordinary automatic producer consumer document. Status text alone cannot
#: authorize membership uncertainty: the whole attempt must be automatic.
AUTOMATIC_DOCUMENT_SCHEMA = "s6_automatic_observation_document_v1"
#: The one existing search-time net response producer, exactly as
#: ``observation_net_seed.response_context`` already validates it.
NATIVE_RESPONSE_METHOD = "three_consecutive_native_rays_gravity_seed_v1"
#: Declared fitted response state and the native-estimated scalars it spends. A
#: record naming anything else has no counted construction receipt here.
RESPONSE_STATE_DIMENSIONS = {"outgoing_velocity_mps": 3}


def policy(value: str | None) -> str:
    """Validate the explicit shared policy; anything unknown fails closed."""
    resolved = "off" if value is None else value
    if resolved not in POLICIES:
        raise ValueError("unsupported terminal net membership policy")
    return resolved


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def receipt_of(attempt: dict) -> dict | None:
    """The bound membership receipt an effective attempt carries, if any."""
    receipt = attempt.get(RECEIPT_KEY)
    if receipt is None:
        return None
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported terminal net membership receipt")
    return receipt


def membership_of(attempt: dict) -> str | None:
    receipt = receipt_of(attempt)
    if receipt is None:
        return None
    selected = receipt.get("membership")
    if selected not in MEMBERSHIPS:
        raise ValueError("membership receipt must declare its selected branch")
    return selected


def index_of(membership: str | None) -> int:
    """Fixed enumeration order: net present first, then net absent."""
    if membership is None:
        return 0
    return MEMBERSHIPS.index(membership)


def _net_event_match(row: dict, event: dict) -> bool:
    return row["event_type"] == event["event_type"] and float(row["frame"]) == float(event["frame"])


def _ancestry(packet: dict, labels: dict) -> str | None:
    """Name why this attempt is not an ordinary automatic producer attempt.

    This is the enforcing boundary for "automatic", and it is deliberately the
    whole document pair rather than one event's status text: a human-authored
    packet that merely writes ``status: predicted`` on a net record cannot reach
    membership uncertainty. The stage's own origin validator
    (``s6_input_origin.validate_automatic``) checks the producer provenance of
    these same two documents before the stage runs; this repeats the document
    half of that contract at the point of demotion, where it is decided.
    """
    if labels.get("schema") != AUTOMATIC_DOCUMENT_SCHEMA:
        return "terminal net membership requires the automatic producer consumer document"
    for name, document in (("consumer", labels), ("packet", packet)):
        if document.get("annotation_origin") != "automatic":
            return f"terminal net membership requires an automatic {name} document"
        if document.get("human_derived") or document.get("human_derived_inputs"):
            return f"automatic {name} document declares human-derived evidence"
        if document.get("automatic_inference_eligible") is not True:
            return f"automatic {name} document is not inference eligible"
    origins = labels.get("stream_origins")
    if not isinstance(origins, dict) or origins.get("events") != "automatic":
        return "terminal net membership requires an automatic event stream"
    return None


def _eligibility(contract: dict, labels: dict) -> str | None:
    """Name why this original net contract cannot carry membership uncertainty."""
    from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

    support = contract.get("occurrence_support")
    if support != ELIGIBLE_OCCURRENCE_SUPPORT:
        return (
            "human exact or reviewed terminal net occurrence is never demoted: "
            f"occurrence support {support!r}"
        )
    if contract.get("predicted_occurrence_admitted") is not True:
        return "original contract did not admit a predicted terminal net occurrence"
    if contract.get("occurrence_confirmed") is not False:
        return "confirmed terminal net occurrence cannot carry membership uncertainty"
    event = contract["original_net_event"]
    ending = contract["original_ending"]
    if not occurrence.predicted_membership(event):
        return "terminal net membership requires an automatic predicted occurrence"
    if (
        event.get("annotation_origin") != "automatic"
        or ending.get("annotation_origin") != "automatic"
    ):
        return "terminal net membership requires ordinary automatic ancestry on event and ending"
    if ending.get("status") != "predicted" or ending.get("exact_epoch_observed") is not False:
        return "derived ending must be an automatic prediction without an observed epoch"
    if float(ending["frame"]) != float(event["frame"]) or list(
        map(float, ending["frame_interval"])
    ) != list(map(float, event["frame_interval"])):
        return "derived ending must be coincident with its original net interval"
    if labels.get("attempt", {}).get("ending_kind") != ending.get("ending_kind"):
        return "consumer ending kind differs from the original derived ending"
    return None


def qualify(prepared: dict, source: dict, labels: dict, setting: str) -> tuple[dict, dict | None]:
    """Qualify both branches from the original packet; never edit a source record.

    ``prepared`` is the packet after ``labeled_terminal_net_tail`` qualification;
    ``source`` is the same packet as it entered that qualification. Both branches
    are derived from those original inputs alone: no fit state, evaluation label
    or reference event is read. A branch pair that cannot share one source
    observation horizon is recorded as an explicit no-branch condition.
    """
    if policy(setting) == "off":
        return prepared, None
    if len(prepared.get("attempts", [])) != 1 or len(source.get("attempts", [])) != 1:
        raise ValueError("one original attempt required")
    attempt = prepared["attempts"][0]
    contract = attempt.get("terminal_net_tail")
    if contract is None:
        return prepared, {
            "status": "not_applicable",
            "reason": "no qualified original terminal net contract",
        }
    reason = _ancestry(prepared, labels) or _eligibility(contract, labels)
    if reason is not None:
        return prepared, {"status": "not_applicable", "reason": reason}
    event = contract["original_net_event"]
    matches = [index for index, row in enumerate(attempt["events"]) if _net_event_match(row, event)]
    if len(matches) != 1:
        return prepared, {
            "status": "not_applicable",
            "reason": "one original packet net event must bind the qualified contract",
        }
    index = matches[0]
    packet_event = attempt["events"][index]
    if list(map(float, packet_event["frame_interval"])) != list(
        map(float, event["frame_interval"])
    ):
        return prepared, {
            "status": "not_applicable",
            "reason": "packet net interval differs from the qualified source interval",
        }
    absent_events = [row for i, row in enumerate(attempt["events"]) if i != index]
    source_attempt = source["attempts"][0]
    if source_attempt["events"] != attempt["events"]:
        return prepared, {
            "status": "not_applicable",
            "reason": "terminal net qualification changed the original event inventory",
        }
    from cv.experiments.connected_shooting import observed_horizon_tail as horizon_tail

    authorization = _authorization(contract, packet_event)
    derived = deepcopy(source)
    derived["attempts"][0] = {**deepcopy(source_attempt), "events": deepcopy(absent_events)}
    try:
        _, horizon_contract = horizon_tail.qualify(derived, labels, absent_membership=authorization)
    except (ValueError, KeyError, TypeError) as error:
        return prepared, {
            "status": "not_applicable",
            "reason": f"net-absent observed horizon did not qualify: {error}",
        }
    horizon = float(contract["observation_horizon"])
    if float(horizon_contract["observation_horizon"]) != horizon:
        return prepared, {
            "status": "not_applicable",
            "reason": "memberships do not share one source observation horizon",
        }
    aftermath = set(contract["native_aftermath_frames"])
    if not aftermath <= set(horizon_contract["native_tail_frames"]):
        return prepared, {
            "status": "not_applicable",
            "reason": "net-absent tail does not retain the original aftermath rows",
        }
    contacts = sum(row["event_type"] == "contact" for row in attempt["events"])
    absent_contacts = sum(row["event_type"] == "contact" for row in absent_events)
    reference = min(contacts, absent_contacts)
    receipt = {
        "schema": SCHEMA,
        "status": "qualified",
        "policy": "predicted",
        "memberships": list(MEMBERSHIPS),
        "enumeration_order": list(MEMBERSHIPS),
        "occurrence_support": ELIGIBLE_OCCURRENCE_SUPPORT,
        "automatic_probability": event.get("automatic_probability"),
        "original_net_event": deepcopy(event),
        "original_net_event_sha256": contract["original_event_sha256"],
        "original_ending": deepcopy(contract["original_ending"]),
        "original_ending_sha256": digest(contract["original_ending"]),
        "packet_net_event_index": index,
        "packet_net_event_sha256": digest(packet_event),
        "original_events_sha256": digest(attempt["events"]),
        "original_event_count": len(attempt["events"]),
        "absent_events_sha256": digest(absent_events),
        "absent_event_count": len(absent_events),
        "removed_event_count": 1,
        "original_observations_sha256": digest(attempt["owner_ball_labels"]),
        "observation_horizon": horizon,
        "shared_observation_horizon": True,
        "original_records_retained": True,
        "source_records_edited": False,
        "automatic_ancestry": "automatic producer consumer/packet documents and event stream",
        "authorization": deepcopy(authorization),
        "present": {"kind": contract["kind"], "contract": deepcopy(contract)},
        "absent": {"kind": horizon_contract["kind"], "contract": deepcopy(horizon_contract)},
        # The complete derived contract of each branch, not only the net/original
        # event hashes: an absent contract edited under an unchanged horizon must
        # not replay.
        "present_contract_sha256": digest(contract),
        "absent_contract_sha256": digest(horizon_contract),
        # Both branches solve the same flight inventory, so neither adds a
        # continuous dimension over the other. The offset is derived here from
        # the actual contact inventory and checked against the fitted vectors.
        "added_parameter_count_offset": {
            PRESENT: PARAMETERS_PER_CONTACT * (contacts - reference),
            ABSENT: PARAMETERS_PER_CONTACT * (absent_contacts - reference),
        },
        "reference_contact_count": reference,
        "contact_count": {PRESENT: contacts, ABSENT: absent_contacts},
        "parameter_complexity_rule": (
            "actual continuous parameters on a common flight-inventory reference; "
            "no per-event penalty and no latent ground counted as a supplied event"
        ),
    }
    result = deepcopy(prepared)
    result["attempts"][0] = {**deepcopy(attempt), RECEIPT_KEY: receipt}
    return result, receipt


def _authorization(contract: dict, packet_event: dict) -> dict:
    """Bind the one omission a branch may declare, by full original record."""
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "membership": ABSENT,
        "occurrence_support": contract["occurrence_support"],
        "event": deepcopy(contract["original_net_event"]),
        "event_sha256": contract["original_event_sha256"],
        "packet_event": deepcopy(packet_event),
        "packet_event_sha256": digest(packet_event),
        "ending": deepcopy(contract["original_ending"]),
        "ending_sha256": digest(contract["original_ending"]),
        "origin": "declared model hypothesis, not a repaired annotation",
    }


def validated_authorization(value) -> dict:
    """The contract a lower-level absent-event helper must be handed.

    This is the whole permission: exactly one original automatic *predicted*
    terminal ``net_hit`` and the coincident automatic derived ending, each bound
    by the hash of its full original record. A human, exact, reviewed or
    arbitrary event, an extra event, or a record whose hash no longer matches its
    own bytes is refused here, so no independent caller can turn these helpers
    into a generic event-deletion API.
    """
    from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

    if not isinstance(value, dict) or value.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("declared absent events require their membership authorization")
    if value.get("membership") != ABSENT:
        raise ValueError("only the net-absent branch may declare a source event absent")
    if value.get("occurrence_support") != ELIGIBLE_OCCURRENCE_SUPPORT:
        raise ValueError("membership authorization requires a source-predicted occurrence")
    event, ending = value.get("event"), value.get("ending")
    packet_event = value.get("packet_event")
    if not isinstance(event, dict) or not isinstance(ending, dict):
        raise ValueError("membership authorization requires its original net and ending records")
    if digest(event) != value.get("event_sha256") or digest(ending) != value.get("ending_sha256"):
        raise ValueError("membership authorization record differs from its bound hash")
    if not isinstance(packet_event, dict) or digest(packet_event) != value.get(
        "packet_event_sha256"
    ):
        raise ValueError("membership authorization packet record differs from its bound hash")
    if event.get("event_type") != "net_hit" or not occurrence.predicted_membership(event):
        raise ValueError("membership authorization requires an automatic predicted terminal net")
    if not _net_event_match(packet_event, event) or list(
        map(float, packet_event["frame_interval"])
    ) != list(map(float, event["frame_interval"])):
        raise ValueError("membership authorization packet record is not the same net event")
    if (
        ending.get("event_type") != "ending"
        or "net" not in ending.get("ending_kind", "")
        or ending.get("annotation_origin") != "automatic"
        or ending.get("status") != "predicted"
        or ending.get("exact_epoch_observed") is not False
    ):
        raise ValueError("membership authorization requires its automatic predicted ending")
    if float(ending["frame"]) != float(event["frame"]) or list(
        map(float, ending["frame_interval"])
    ) != list(map(float, event["frame_interval"])):
        raise ValueError("membership authorization ending is not coincident with its net")
    return value


def authorized_events(value) -> list[dict]:
    """The single packet-shaped record a validated authorization declares absent."""
    return [deepcopy(validated_authorization(value)["packet_event"])]


def response_dimensions(fit: dict | None) -> int | None:
    """Native-estimated response parameters one *candidate* actually spent.

    ``0`` for a fixed law, which spends nothing new; the declared fitted state of
    a native response otherwise, which is three for the one existing search-time
    producer. ``None`` means the candidate carries a response whose construction
    receipt cannot be substantiated: it is unrankable, not silently zero and not
    silently penalized. The validation is exactly the binding
    ``observation_net_seed.response_context`` already requires before replaying a
    response, so no second producer schema is invented here.
    """
    if fit is None:
        return 0
    response = fit.get("net_response")
    receipt = fit.get("net_response_initialization")
    if response is None and receipt is None:
        return 0
    if (
        not isinstance(response, dict)
        or not isinstance(receipt, dict)
        or receipt.get("method") != NATIVE_RESPONSE_METHOD
        or receipt.get("selection_reads_acceptance") is not False
        or receipt.get("ground_epoch_supplied") is not False
        or response != receipt.get("response")
    ):
        return None
    total = 0
    for name, value in response.items():
        if name not in RESPONSE_STATE_DIMENSIONS:
            return None
        try:
            values = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if len(values) != RESPONSE_STATE_DIMENSIONS[name] or not all(map(math.isfinite, values)):
            return None
        total += RESPONSE_STATE_DIMENSIONS[name]
    return total or None


def occurrence_prior(attempt: dict) -> dict:
    """Bounded source occurrence preference, using the existing optional score scale."""
    branch = membership_of(attempt)
    return source_occurrence_prior(attempt[RECEIPT_KEY]["automatic_probability"], branch)


def source_occurrence_prior(probability, branch: str) -> dict:
    """Pure score convention; callers retain the original event/provenance binding."""
    if branch not in MEMBERSHIPS:
        raise ValueError("an effective membership branch is required for its occurrence prior")
    if probability is None:
        odds = 0.0
    else:
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("finite source occurrence probability in [0, 1] required")
        # Same [-1, 1] engineering bound used by optional classifier events.
        # This is an odds preference, not a calibrated posterior or confidence.
        odds = max(-1.0, min(1.0, math.log(max(probability, 1e-9) / max(1 - probability, 1e-9))))
    return {
        "source_probability": probability,
        "log_odds": odds if branch == PRESENT else 0.0,
        "absent_baseline": 0.0,
        "bounds_nats": [-1.0, 1.0],
        "calibrated_probability": False,
        "reason": "source_probability_unavailable"
        if probability is None
        else "source_classifier_odds",
    }


def replay_occurrence_prior(declared: dict) -> float:
    event = declared["predicted_net_occurrence"]["original_net_event"]
    prior = source_occurrence_prior(
        event.get("automatic_probability"), declared["selected_membership"]
    )
    if prior != declared.get("occurrence_prior"):
        raise ValueError("replayed membership occurrence prior differs from its source event")
    return prior["log_odds"]


def candidate_dimensions(membership: str | None, row: dict) -> int | None:
    """Added continuous parameters this one completed candidate spent.

    Counted from the candidate's own construction receipt, never from a family
    penalty. The net-absent branch has no net at all, so a response there would
    be an inherited fitted state rather than a dimension to count: that is a hard
    failure, not a score.
    """
    fit = row.get("measurement", {}).get("fit")
    dimensions = response_dimensions(fit)
    if membership == ABSENT and dimensions != 0:
        raise ValueError("net-absent candidate inherited a terminal net response")
    return dimensions


def _validated(attempt: dict) -> dict:
    """Bind one original prepared attempt to its receipt; refuse any tampering."""
    receipt = receipt_of(attempt)
    if receipt is None or receipt.get("status") != "qualified":
        raise ValueError("terminal net membership requires its qualified original receipt")
    if receipt.get("membership") is not None:
        raise ValueError("an effective membership attempt cannot be re-derived")
    contract = attempt.get("terminal_net_tail")
    if contract != receipt["present"]["contract"]:
        raise ValueError("original terminal net contract differs from its membership receipt")
    if receipt["automatic_probability"] != contract["original_net_event"].get(
        "automatic_probability"
    ):
        raise ValueError("membership occurrence probability differs from the original net")
    # The complete derived contract of each branch is bound, so an absent
    # contract edited while its horizon is retained cannot replay.
    if (
        digest(contract) != receipt["present_contract_sha256"]
        or digest(receipt["absent"]["contract"]) != receipt["absent_contract_sha256"]
    ):
        raise ValueError("membership branch contract differs from its bound derivation hash")
    validated_authorization(receipt["authorization"])
    if digest(attempt["events"]) != receipt["original_events_sha256"]:
        raise ValueError("original event inventory differs from its membership receipt")
    if len(attempt["events"]) != receipt["original_event_count"]:
        raise ValueError("original event count differs from its membership receipt")
    if digest(attempt["owner_ball_labels"]) != receipt["original_observations_sha256"]:
        raise ValueError("original native observations differ from their membership receipt")
    index = receipt["packet_net_event_index"]
    if not isinstance(index, int) or not 0 <= index < len(attempt["events"]):
        raise ValueError("membership receipt lost its original net event position")
    if digest(attempt["events"][index]) != receipt["packet_net_event_sha256"]:
        raise ValueError("bound original net event differs from its membership receipt")
    if float(contract["observation_horizon"]) != float(receipt["observation_horizon"]) or float(
        receipt["absent"]["contract"]["observation_horizon"]
    ) != float(receipt["observation_horizon"]):
        raise ValueError("membership branches no longer share one source observation horizon")
    return receipt


def effective_attempt(attempt: dict, membership: str) -> dict:
    """The pure branch view of one original prepared attempt.

    Present keeps the original net packet exactly. Absent removes exactly the one
    bound source net event from the effective event view, drops its derived
    ending semantics and declares the existing unresolved observed-horizon
    contract. Every native row, camera, epoch and interior event is retained and
    the original records stay recoverable through the receipt.
    """
    if membership not in MEMBERSHIPS:
        raise ValueError("explicit terminal net membership branch required")
    receipt = _validated(attempt)
    bound = {**deepcopy(receipt), "membership": membership}
    if membership == PRESENT:
        return {**deepcopy(attempt), RECEIPT_KEY: bound}
    index = receipt["packet_net_event_index"]
    events = [row for i, row in enumerate(attempt["events"]) if i != index]
    if digest(events) != receipt["absent_events_sha256"]:
        raise ValueError("net-absent event view differs from its membership receipt")
    result = {
        key: deepcopy(value)
        for key, value in attempt.items()
        if key not in ("terminal_net_tail", "terminal_net_hypothesis_event", RECEIPT_KEY)
    }
    result["events"] = deepcopy(events)
    result["observed_horizon_tail"] = deepcopy(receipt["absent"]["contract"])
    result[RECEIPT_KEY] = bound
    return result


def branches(attempt: dict) -> list[tuple[str, dict]]:
    """Both effective attempts in the fixed enumeration order."""
    return [(name, effective_attempt(attempt, name)) for name in MEMBERSHIPS]


def source_events(attempt: dict) -> list[dict]:
    """The original accepted event stream every source witness binds to.

    Recovered from the effective attempt plus its receipt, then verified by the
    original digest, so a witness can never be bound to an edited stream.
    """
    membership = membership_of(attempt)
    if membership in (None, PRESENT):
        return attempt["events"]
    receipt = attempt[RECEIPT_KEY]
    index = receipt["packet_net_event_index"]
    events = list(attempt["events"])
    # The packet stream carries the producer's record as the packet holds it,
    # which is not always byte-identical to the consumer document's own copy.
    events.insert(index, deepcopy(receipt["authorization"]["packet_event"]))
    if digest(events) != receipt["original_events_sha256"]:
        raise ValueError("original event stream cannot be recovered from the membership receipt")
    return events


def effective_events(attempt: dict, events: list[dict]) -> list[dict]:
    """Apply the authorized omission to a source-qualified contact alternative.

    Candidate qualification still reads the original accepted events. Its view
    may add witnessed contacts, but must retain exactly the one original net
    record until this resolver removes it; no other event can be dropped here.
    """
    if membership_of(attempt) != ABSENT:
        return events
    source_events(attempt)  # Verify original accepted-stream ancestry first.
    net = validated_authorization(attempt[RECEIPT_KEY]["authorization"])["packet_event"]
    matches = [i for i, event in enumerate(events) if event == net]
    if len(matches) != 1:
        raise ValueError("contact alternative must retain exactly the authorized original net")
    return [deepcopy(event) for i, event in enumerate(events) if i != matches[0]]


def source_ending_frame(attempt: dict) -> float | None:
    """The original derived ending epoch a branch declares absent."""
    membership = membership_of(attempt)
    if membership in (None, PRESENT):
        return None
    return float(attempt[RECEIPT_KEY]["original_ending"]["frame"])


def bounce_witness(attempt: dict, document: dict) -> dict:
    """Requalify the original bounce witness under the effective branch scope.

    The same frozen witness builder runs on the same original emissions, clip,
    native window and candidate mode. Only the accepted event partition and the
    declared ending change, to exactly the branch this membership declares, so a
    source candidate blocked solely by the disputed net ending can requalify with
    its original probability, interval and ordering untouched.
    """
    membership = membership_of(attempt)
    if membership in (None, PRESENT):
        return document
    receipt = attempt[RECEIPT_KEY]
    ending = document.get("physical_ending")
    if ending is None:
        raise ValueError("net-absent witness requires an original physical ending to remove")
    if float(ending["frame"]) != float(receipt["original_ending"]["frame"]):
        raise ValueError("optional bounce witness ending differs from the declared absent ending")
    from cv.pipeline import s6_optional_bounce_scope as scope

    return scope.build_witness(
        emissions=document["original_emissions"],
        events=attempt["events"],
        observations=document["original_observations"],
        window=tuple(document["native_window"]),
        match_id=document["match_id"],
        clip=document["clip"],
        source_events=document["source_events"],
        segmentation_source=document["segmentation_source"],
        mode=document["mode"],
        allow_terminal_net=document["allow_terminal_net"],
        allow_observed_horizon=True,
        physical_ending=None,
        absent_membership=deepcopy(receipt["authorization"]),
    )


def added_parameter_count(attempt: dict, base: int) -> int:
    """Actual added continuous parameters on the common membership reference."""
    membership = membership_of(attempt)
    if membership is None:
        return base
    offset = attempt[RECEIPT_KEY]["added_parameter_count_offset"][membership]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("nonnegative integer membership parameter offset required")
    return base + offset


def derivation_fields(attempt: dict) -> dict:
    """The bound derivation of an original prepared attempt, without a branch.

    Used by the membership search receipt, which must record what was enumerated
    even when no branch completed and no selection exists.
    """
    receipt = receipt_of(attempt)
    if receipt is None:
        return {}
    return {
        "schema": SCHEMA,
        "occurrence_support": receipt["occurrence_support"],
        "automatic_probability": receipt["automatic_probability"],
        "original_net_event_sha256": receipt["original_net_event_sha256"],
        "original_ending_sha256": receipt["original_ending_sha256"],
        "original_events_sha256": receipt["original_events_sha256"],
        "present_contract_sha256": receipt["present_contract_sha256"],
        "absent_contract_sha256": receipt["absent_contract_sha256"],
        "observation_horizon": receipt["observation_horizon"],
        "added_parameter_count_offset": deepcopy(receipt["added_parameter_count_offset"]),
        "original_records_retained": True,
        "source_records_edited": False,
    }


def selection_fields(attempt: dict) -> dict:
    """Provenance of the selected branch, keeping the rejected net visible."""
    membership = membership_of(attempt)
    if membership is None:
        return {}
    receipt = attempt[RECEIPT_KEY]
    return {
        RECEIPT_KEY: {
            "schema": SCHEMA,
            "policy": "predicted",
            "selected_membership": membership,
            "enumeration_order": list(MEMBERSHIPS),
            "occurrence_support": receipt["occurrence_support"],
            "occurrence_prior": occurrence_prior(attempt),
            "original_net_event_sha256": receipt["original_net_event_sha256"],
            "original_ending_sha256": receipt["original_ending_sha256"],
            "original_events_sha256": receipt["original_events_sha256"],
            "present_contract_sha256": receipt["present_contract_sha256"],
            "absent_contract_sha256": receipt["absent_contract_sha256"],
            "observation_horizon": receipt["observation_horizon"],
            "rejected_occurrence_hypothesis": (
                None
                if membership == ABSENT
                else {"membership": ABSENT, "kind": receipt["absent"]["kind"]}
            ),
            "predicted_net_occurrence": (
                {
                    "status": "rejected_model_hypothesis",
                    "original_net_event": deepcopy(receipt["original_net_event"]),
                    "original_ending": deepcopy(receipt["original_ending"]),
                    "physical_ending": None,
                    "terminal_ground_count": "unknown",
                }
                if membership == ABSENT
                else {
                    "status": "selected_model_hypothesis",
                    "original_net_event": deepcopy(receipt["original_net_event"]),
                }
            ),
            "added_parameter_count_offset": receipt["added_parameter_count_offset"][membership],
            "source_records_edited": False,
            "original_records_retained": True,
            "selection_uses_gates": False,
            "selection_uses_reference": False,
        },
        "selected_membership": membership,
    }


def absent_source_nets(configuration: dict) -> list[dict]:
    """Source net events the selected branch declares absent, for shared helpers.

    Replay, refinement and export narrow the shared bounded-net preparation
    helper with exactly this list, so no continuation inherits a captured global
    net the selected branch does not own.
    """
    declared = configuration.get(RECEIPT_KEY)
    if declared is None or declared.get("selected_membership") != ABSENT:
        return []
    return [deepcopy(declared["predicted_net_occurrence"]["original_net_event"])]


def replay_attempt(attempt: dict, configuration: dict) -> dict:
    """Rebuild the effective attempt a frozen report was actually solved on.

    Takes the original packet attempt plus the report's own membership block and
    returns the same validated branch view the search used. A tampered receipt,
    branch name or original hash is refused rather than replayed.
    """
    declared = configuration.get(RECEIPT_KEY)
    if declared is None:
        return attempt
    if not isinstance(declared, dict) or declared.get("schema") != SCHEMA:
        raise ValueError("unsupported replayed terminal net membership block")
    membership = declared.get("selected_membership")
    if membership not in MEMBERSHIPS:
        raise ValueError("replayed membership must name one declared branch")
    receipt = _validated(attempt)
    for name in (
        "original_net_event_sha256",
        "original_ending_sha256",
        "original_events_sha256",
        "present_contract_sha256",
        "absent_contract_sha256",
    ):
        if declared.get(name) != receipt[name]:
            raise ValueError(f"replayed membership {name} differs from the original packet")
    if float(declared.get("observation_horizon", float("nan"))) != float(
        receipt["observation_horizon"]
    ):
        raise ValueError("replayed membership horizon differs from the original packet")
    if declared.get("occurrence_support") != receipt["occurrence_support"]:
        raise ValueError("replayed membership occurrence support differs from the original packet")
    effective = effective_attempt(attempt, membership)
    if selection_fields(effective)[RECEIPT_KEY] != declared:
        raise ValueError("replayed membership provenance differs from the original packet")
    return effective
