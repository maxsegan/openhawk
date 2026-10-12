"""Opt-in terminal flight whose ground count is unknown, not zero.

The source has no accepted physical ending for the visible ball tail. That tail
has an observation horizon and no witnessed physical point ending: its ground
count is ``unknown``. Impacts the solver finds after the last supplied contact
interval stay *latent modelled* events. They are never source events, never a
point ending, and never a ground witness; they are reported in their own column.

Everything explicit is preserved: every supplied interior bounce/contact/net
residual, the original event inventory, the source observation cadence, and the
competitive serve and earlier contacts. ``complete_point`` stays false.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from cv.experiments.connected_shooting import observation_operator

KIND = "observed_horizon"
#: Same numerical comparison tolerance the net tail uses at an event boundary.
EVENT_TOLERANCE_FRAMES = 1e-6
#: A tail shorter than this is not an observed flight, it is a couple of stray
#: fronts after a contact. Shared across source cases.
MINIMUM_VISIBLE_TAIL_ROWS = 5
#: Native broadcast frame used by the automatic tracker rows.
NATIVE_IMAGE_SIZE = (1920.0, 1080.0)
IMAGE_BORDER_MARGIN_PX = 20.0
HORIZON_REASONS = ("image_border", "window_end", "track_end")
UNRESOLVED_STATUS = ("absent", "unsupported", "ambiguous")


def _finite(*values) -> bool:
    return bool(np.isfinite(np.asarray(values, float)).all())


def _same_record(left: dict, right: dict) -> bool:
    """Whole-record identity, so a shared type and epoch is never enough."""
    return left == right


def _tail_rows(rows, low: float, high: float) -> list[dict]:
    return [row for row in rows if low < float(row["frame"]) <= high]


def _horizon_reason(rows, last_visible: dict, window_end: float) -> str:
    """Name why the source stopped supporting the flight; never invent an end."""
    x, y = float(last_visible["x1080"]), float(last_visible["y1080"])
    width, height = NATIVE_IMAGE_SIZE
    if (
        x <= IMAGE_BORDER_MARGIN_PX
        or y <= IMAGE_BORDER_MARGIN_PX
        or x >= width - IMAGE_BORDER_MARGIN_PX
        or y >= height - IMAGE_BORDER_MARGIN_PX
    ):
        return "image_border"
    if float(last_visible["frame"]) >= window_end:
        return "window_end"
    return "track_end"


def qualify(
    packet: dict, labels: dict, *, absent_membership: dict | None = None
) -> tuple[dict, dict]:
    """Qualify an unresolved visible tail from source observations alone.

    Refuses a malformed, unobserved or ambiguous-event tail. Reads only the
    supplied automatic events and the frozen source observation inventory: no
    evaluation label, saved fit state, manual selection or frontier call.

    ``absent_membership`` is the shared resolver's validated authorization, and
    nothing else opens this door: it names exactly one original automatic
    predicted terminal net and its coincident ending, each bound by the hash of
    its full original record. That one record may be missing from the accepted
    inventory and passed over in the tail's ambiguity check, matched by its whole
    record rather than by type and epoch. Every other held source opinion still
    refuses the tail, and the declared record is preserved verbatim here.
    """
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one original attempt required")
    attempt = packet["attempts"][0]
    clip = attempt["point_clip"]
    events = attempt["events"]
    if not events or any(e["event_type"] not in {"contact", "bounce", "net_hit"} for e in events):
        raise ValueError("original physical events required")
    contacts = [e for e in events if e["event_type"] == "contact"]
    if not contacts:
        raise ValueError("original contact required")
    last_contact = contacts[-1]
    low, high = (float(v) for v in last_contact["frame_interval"])
    if (
        not _finite(low, high, last_contact["frame"])
        or not low <= float(last_contact["frame"]) <= high
    ):
        raise ValueError("malformed original terminal contact interval")
    if last_contact.get("occurrence_status") in UNRESOLVED_STATUS:
        raise ValueError("unresolved originating contact cannot open an observed tail")
    if any(float(e["frame"]) > float(last_contact["frame"]) for e in events):
        raise ValueError("existing supplied terminal event path remains authoritative")
    absent, absent_record = [], None
    if absent_membership is not None:
        from cv.pipeline import s6_terminal_net_membership as membership

        authorization = membership.validated_authorization(absent_membership)
        absent = [deepcopy(authorization["packet_event"])]
        absent_record = deepcopy(authorization["event"])
        if any(_same_record(e, absent[0]) for e in events):
            raise ValueError("declared absent membership event is still an accepted event")
        if float(absent[0]["frame"]) <= float(last_contact["frame"]):
            raise ValueError("only a terminal source event may be declared absent")

    records = [r for r in labels["ball"]["records"] if r["clip"] == clip]
    if len(records) != 1:
        raise ValueError("one matching source observation window required")
    rows = records[0]["frames"]
    frames = [int(r["frame"]) for r in rows]
    if not frames or frames != sorted(frames) or len(frames) != len(set(frames)):
        raise ValueError("ordered unique source observation inventory required")
    window_end = float(frames[-1])
    tail = _tail_rows(rows, high, window_end)
    visible = [r for r in tail if r["status"] == "visible"]
    if len(visible) < MINIMUM_VISIBLE_TAIL_ROWS:
        raise ValueError("observed horizon requires a supported visible tail after the contact")
    contiguous = 1
    best = 1
    for previous, row in zip(visible, visible[1:]):
        contiguous = contiguous + 1 if int(row["frame"]) == int(previous["frame"]) + 1 else 1
        best = max(best, contiguous)
    if best < MINIMUM_VISIBLE_TAIL_ROWS:
        raise ValueError("observed horizon requires contiguous visible tail rows")
    # automatic_ball_rows preserves the producer's ID as automatic_track_id.
    # Reading only track_id silently accepts any number of unrelated objects.
    identities = [r.get("automatic_track_id", r.get("track_id")) for r in visible]
    tracks = {str(value) for value in identities if value not in (None, "")}
    if len(tracks) != 1 or any(value in (None, "") for value in identities):
        raise ValueError("one automatic track identity required across the observed tail")
    if any(not _finite(r["x1080"], r["y1080"]) for r in visible):
        raise ValueError("finite native source observations required across the observed tail")

    # A held-but-not-abstaining physical event inside the tail means the source
    # has an opinion about this tail that the accepted inventory does not carry.
    # That is ambiguity, not an unresolved horizon: refuse.
    for record in labels.get("events", {}).get("records", []):
        if record.get("clip", clip) != clip or record["event_type"] not in {
            "contact",
            "bounce",
            "net_hit",
        }:
            continue
        frame = float(record["frame"])
        # Only the whole declared record is passed over. A different source
        # opinion that merely shares this type and epoch still refuses the tail.
        if absent_record is not None and _same_record(record, absent_record):
            continue
        if high < frame <= window_end and record.get("model_abstain") is False:
            raise ValueError("ambiguous source physical event inside the unresolved tail")

    last_visible = visible[-1]
    duration = observation_operator.from_packet(packet)["exposure_duration_frames"]
    horizon = float(last_visible["frame"]) + observation_operator.support_span(duration)
    if not _finite(horizon) or horizon <= high or horizon > window_end + 1.0:
        raise ValueError(
            "observed horizon must follow the contact and stay inside the source window"
        )
    reason = _horizon_reason(rows, last_visible, window_end)
    if reason not in HORIZON_REASONS:
        raise ValueError("explicit source horizon reason required")

    from cv.experiments.connected_shooting.event_recovery import extend_attempt_window

    prepared = deepcopy(packet)
    active = extend_attempt_window(prepared["attempts"][0], labels["ball"]["records"], horizon)
    raw_hash = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract = dict(
        kind=KIND,
        interval=[high, horizon],
        representative=float(last_contact["frame"]),
        observation_horizon=horizon,
        horizon_reason=reason,
        horizon_frame=float(last_visible["frame"]),
        native_tail_frames=[int(r["frame"]) for r in visible],
        supplied_ground_count=0,
        terminal_ground_count="unknown",
        physical_ending=None,
        ending_semantics="unresolved",
        latent_ground_epochs_supplied=False,
        latent_ground_origin="model_latent_not_source_event",
        original_events_sha256=raw_hash,
        original_event_count=len(events),
        original_competitive_contact_frame=float(last_contact["frame"]),
        source="automatic source observations and automatic physical events",
        # Absent unless an explicit membership hypothesis declares one, so every
        # existing contract and the receipts that bind one stay byte-identical.
        **(
            {
                "absent_membership_events": absent,
                "absent_membership_source_records": [absent_record],
                "absent_membership_events_sha256": hashlib.sha256(
                    json.dumps(absent, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "absent_membership_authorization_sha256": hashlib.sha256(
                    json.dumps(absent_membership, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "absent_membership_origin": "declared model hypothesis, not a repaired annotation",
            }
            if absent
            else {}
        ),
    )
    active["observed_horizon_tail"] = contract
    prepared["attempts"][0] = active
    return prepared, contract


def for_events(contract: dict, events: list[dict]) -> dict:
    """Rebase only the unresolved suffix of a declared source-event topology.

    Alternative contacts and bounce witnesses keep their ordinary constraints.
    Original source identity stays unchanged; the branch still has no known end.
    """
    contacts = [event for event in events if event["event_type"] == "contact"]
    last = max(contacts, key=lambda event: float(event["frame"]))
    low = float(last["frame_interval"][1])
    high = float(contract["observation_horizon"])
    if not float(last["frame"]) <= low < high:
        raise ValueError("alternative contact requires an observed suffix inside the horizon")
    if any(float(event["frame"]) > high for event in events):
        raise ValueError("alternative physical event lies beyond the supported observation horizon")
    native_tail = [frame for frame in contract["native_tail_frames"] if frame > low]
    # Original qualification establishes one track identity for this entire
    # suffix. A later contact must retain its own contiguous visible support.
    if not any(
        native_tail[index + MINIMUM_VISIBLE_TAIL_ROWS - 1] - frame == MINIMUM_VISIBLE_TAIL_ROWS - 1
        for index, frame in enumerate(native_tail[: -(MINIMUM_VISIBLE_TAIL_ROWS - 1)])
    ):
        raise ValueError("alternative contact requires five contiguous visible tail rows")
    supplied = [
        float(event["frame"])
        for event in events
        if event["event_type"] == "bounce" and float(last["frame"]) < float(event["frame"]) <= high
    ]
    return {
        **contract,
        "interval": [low, high],
        "representative": float(last["frame"]),
        "supplied_ground_count": len(supplied),
        "supplied_ground_frames": supplied,
        "native_tail_frames": native_tail,
    }


def modeled_horizon(attempt: dict, scope: dict) -> float:
    """Keep the original source window distinct from its supported ball horizon."""
    tail = attempt.get("observed_horizon_tail")
    if tail is None:
        return float(scope["observation_horizon"])
    return float(tail["observation_horizon"])


def interval(scene):
    value = getattr(scene, "observed_horizon_tail", None)
    return None if value is None else tuple(map(float, value["interval"]))


def latent_grounds(scene, flights) -> list[float]:
    """Modelled impacts on the final flight after the last supplied contact."""
    bounds = interval(scene)
    if bounds is None:
        return []
    return [
        float(hit["frame"])
        for hit in flights[-1].get("bounces", [])[
            scene.observed_horizon_tail.get("supplied_ground_count", 0) :
        ]
        if float(hit["frame"]) >= bounds[0] - EVENT_TOLERANCE_FRAMES
    ]


def competitive_grounds(scene, flight_index, impacts, supplied_count=None):
    """Latent tail impacts are not extra supplied impacts on the last flight.

    Interior flights keep every explicit supplied-count constraint, including
    the extra-impact push. Only the unresolved tail stops being told how many
    times the ball may touch the ground, which is the truth about the input.
    """
    bounds = interval(scene)
    if bounds is None or flight_index != len(scene.pixels) - 1:
        return impacts
    count = (
        scene.observed_horizon_tail.get("supplied_ground_count", 0)
        if supplied_count is None
        else supplied_count
    )
    return [
        hit
        for index, hit in enumerate(impacts)
        if index < count or float(hit["frame"]) < bounds[0] - EVENT_TOLERANCE_FRAMES
    ]


def terminal_slack(scene, parameters, native_last, *, cache=None, duration: float | None = 0.25):
    """Horizon feasibility only; never a demand that the tail reach the ground."""
    from cv.experiments.connected_shooting import model

    bounds = interval(scene)
    if bounds is None:
        raise ValueError("original observed horizon contract required")
    flights = model.chain(scene, parameters, simulation_cache=cache)
    horizon_slack = (
        1.0
        if scene.contact_frames[-1] >= native_last + observation_operator.support_span(duration)
        else -1e6
    )
    latent = latent_grounds(scene, flights)
    # A latent impact may sit anywhere inside the observed tail. It may not be
    # pushed past the horizon, and it is never required to exist.
    beyond = 1.0 if not latent else float(bounds[1] - max(latent))
    return np.array([horizon_slack, beyond, 1.0])


def completion(scene, parameters, native, *, duration: float | None = 0.25) -> dict:
    """A receipt for an unresolved ending: supported horizon, never a ground end."""
    from cv.experiments.connected_shooting import model

    bounds = interval(scene)
    if bounds is None:
        raise ValueError("original observed horizon contract required")
    flights = model.chain(scene, parameters)
    fit = flights[-1]
    latent = latent_grounds(scene, flights)
    checks = dict(
        native_tail_retained=max(native) + observation_operator.support_span(duration)
        <= float(scene.contact_frames[-1]) + 1e-8,
        original_tail_retained=set(scene.observed_horizon_tail["native_tail_frames"]).issubset(
            set(map(int, native))
        ),
        latent_grounds_inside_horizon=all(
            value <= bounds[1] + EVENT_TOLERANCE_FRAMES for value in latent
        ),
    )
    valid = all(checks.values())
    return dict(
        status="held",
        reason=None if valid else "observed_horizon_tail_not_supported",
        kind=KIND,
        required=False,
        checks=checks,
        valid=valid,
        end_frame=float(scene.contact_frames[-1]),
        end_xyz=np.asarray(fit["end_xyz"]).tolist(),
        parameters_changed=False,
        observations_discarded=0,
        physical_ending_classified=False,
        ending_semantics="unresolved",
        terminal_ground_count="unknown",
        horizon_reason=scene.observed_horizon_tail["horizon_reason"],
        latent_terminal_ground_frames=latent,
        latent_ground_origin="model_latent_not_source_event",
        original_contract=scene.observed_horizon_tail,
    )


def mark_verdict(verdict, contract):
    """Separate a supported partial flight from a witnessed complete point."""
    if contract is None:
        return verdict
    result = deepcopy(verdict)
    final = result["flights"][-1] if result.get("flights") else {}
    receipt = final.get("observed_horizon_tail") or {}
    result["observed_horizon_tail"] = dict(
        kind=KIND,
        horizon_reason=contract["horizon_reason"],
        observation_horizon=contract["observation_horizon"],
        terminal_ground_count="unknown",
        supplied_ground_count=0,
        physical_ending=None,
        ending_semantics="unresolved",
        latent_terminal_ground_frames=receipt.get("latent_terminal_ground_frames", []),
        latent_ground_origin="model_latent_not_source_event",
        latent_ground_is_source_event=False,
        latent_ground_is_point_ending=False,
        original_events_sha256=contract["original_events_sha256"],
        supported_partial_tail=bool(final.get("accepted")),
        witnessed_complete_point=False,
    )
    result["complete_point"] = False
    result["complete_original_point"] = False
    result["partial_point"] = bool(result.get("accepted_flight_count", 0))
    result["ending_semantics"] = "unresolved"
    return result
