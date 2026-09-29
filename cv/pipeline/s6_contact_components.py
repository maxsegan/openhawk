"""Input-only maximal supported contact components for unresolved source endings.

``prepare(packet, labels, cameras, mode="unresolved_ending")`` returns a plan,
not executable stage packets. No solver, scoring, reference or exported state is
read. Every original contact-origin slot remains in the plan, including uncertain
origins and unknown endpoints. Components have separate identities and no shared
state across gaps. Current runtime preparation is deliberately not connected.
"""

from __future__ import annotations

from copy import deepcopy
import math

from cv.experiments.connected_shooting import source_flight_coverage as coverage
from cv.experiments.connected_shooting.labeled_event_occurrence import record_hash
from cv.pipeline import s6_contact_prefix_scope as prefix
from cv.pipeline import s6_first_contact_role as role

SCHEMA = "s6_supported_contact_components_v1"
MODE = "unresolved_ending"
DEFAULT_MODE = "off"
MODES = (DEFAULT_MODE, MODE)
#: Default-off admission of an original source whose first physical record is
#: not its first contact. The leading rows stay evidence; no contact is invented.
LEADING_DEFAULT = "off"
LEADING_PREFIX = "declared_prefix"
LEADING_MODES = (LEADING_DEFAULT, LEADING_PREFIX)
LEADING_FIELD = "leading_physical_prefix"
#: Default-off admission of a source whose original topology contradicts one
#: competitive flight somewhere. The contradictory span stays an explicit
#: unsupported original slot; the supported spans around it become components,
#: and a component origin the source cannot resolve is bound unknown.
ROUTING_DEFAULT = "off"
ROUTING_UNSUPPORTED_SPAN = "unsupported_original_span"
ROUTING_MODES = (ROUTING_DEFAULT, ROUTING_UNSUPPORTED_SPAN)
ROUTING_SCHEMA = "s6_contact_component_source_routing_v1"
ROUTING_FIELD = "contact_component_routing"
#: Default off until a same-wave measurement promotes it. ``ball_track`` makes a
#: flight searchable when the ball's own events or track show the ending, with
#: no known next contact. ``ball_track_last_sample`` does that and, only on the
#: point's last original flight, ends at the last visible ball sample when those
#: four endings are absent. Explicit ``off`` is the rollback and leaves every
#: existing plan unchanged. Neither mode is the production default here.
TERMINAL_TRACK_OFF = "off"
TERMINAL_TRACK_BALL = "ball_track"
TERMINAL_TRACK_LAST_SAMPLE = "ball_track_last_sample"
TERMINAL_TRACK_MODES = (TERMINAL_TRACK_OFF, TERMINAL_TRACK_BALL, TERMINAL_TRACK_LAST_SAMPLE)
TERMINAL_TRACK_FIELD = "terminal_track_endpoint"
#: Default off. ``on`` adds a net-stop terminal and trims each flight at the
#: first camera cut, held camera, point end or dead-ball boundary. It does not
#: invent a bounce. Explicit ``off`` leaves every existing plan unchanged.
SUPPORTED_ENDINGS_OFF = "off"
SUPPORTED_ENDINGS_ON = "on"
SUPPORTED_ENDINGS_MODES = (SUPPORTED_ENDINGS_OFF, SUPPORTED_ENDINGS_ON)
SUPPORTED_ENDINGS_FIELD = "supported_flight_endings"
#: Default off. ``on`` keeps an aftermath row as a dead-ball boundary, drops an
#: automatic bounce more than 1.5 s after the previous live bounce, and stops a
#: last-flight tail at the first picture exit or track gap. Explicit ``off``
#: leaves every existing plan unchanged. Not a production default here.
ENDING_OWNERSHIP_OFF = "off"
ENDING_OWNERSHIP_ON = "on"
ENDING_OWNERSHIP_MODES = (ENDING_OWNERSHIP_OFF, ENDING_OWNERSHIP_ON)
ENDING_OWNERSHIP_FIELD = "ending_ownership"
#: Default off. ``on`` ends live play at the first supported bounce whose own
#: court reading is outside the singles court by more than ``OUT_BOUNCE_MARGIN_M``.
#: The serve box is not judged (a missed serve bounce is common). A later
#: contact is the dead ball and opens no flight; the flight that owns the out
#: bounce ends on it. A bounce with no reliable per-frame court reading is not
#: judged. Explicit ``off`` leaves every existing plan unchanged.
OUT_BOUNCE_ENDING_OFF = "off"
OUT_BOUNCE_ENDING_ON = "on"
OUT_BOUNCE_ENDING_MODES = (OUT_BOUNCE_ENDING_OFF, OUT_BOUNCE_ENDING_ON)
OUT_BOUNCE_ENDING_FIELD = "out_bounce_ending"
OUT_BOUNCE_MARGIN_M = 0.75
#: Default off. ``supplied`` releases an interior contact-to-contact flight held
#: only by one labelled ``ambiguous`` bounce inside it: that bounce's unresolved
#: occurrence and the source boundary built from the same interval stop being
#: barriers, both contacts must be supported, and nothing else may cross the span.
#: The bounce then reaches the fitter as a supplied ground whose ray and timing
#: witnesses come from the ball track, so a flight that did not bounce fails its
#: checks and stays unaccepted. The released flight is its own component. Explicit
#: ``off`` leaves every existing plan unchanged.
AMBIGUOUS_GROUND_OFF = "off"
AMBIGUOUS_GROUND_SUPPLIED = "supplied"
AMBIGUOUS_GROUND_MODES = (AMBIGUOUS_GROUND_OFF, AMBIGUOUS_GROUND_SUPPLIED)
AMBIGUOUS_GROUND_FIELD = "ambiguous_interior_ground"
_COURT_WIDTH_M = 10.97
_COURT_LENGTH_M = 23.77
_SINGLES_HALF_WIDTH_M = 4.115
_SERVICE_DEPTH_M = 6.40
#: Default off. ``on`` closes a held flight just before the first labelled
#: boundary row after its contact (a receiver touch, a continuation landing, an
#: uncertain row), or at its net stop when that row is the net. A dead-ball close
#: at the last live bounce keeps up to four ball samples after that bounce, so the
#: fitted flight contains the bounce. Explicit ``off`` leaves every plan unchanged.
BOUNDARY_ENDINGS_OFF = "off"
BOUNDARY_ENDINGS_ON = "on"
BOUNDARY_ENDINGS_MODES = (BOUNDARY_ENDINGS_OFF, BOUNDARY_ENDINGS_ON)
BOUNDARY_ENDINGS_FIELD = "boundary_endings"
#: ``net_stop_tail="on"`` closes a net stop three frames past the net event instead
#: of one, so the fitted tape hit can sit inside the flight (panel D
#: source06_short_0001_a2 A2_serve closed 4 cm short of the net plane). The net
#: check still needs a modelled hit in the tape band. ``off`` leaves plans unchanged.
NET_STOP_TAIL_OFF = "off"
NET_STOP_TAIL_ON = "on"
NET_STOP_TAIL_MODES = (NET_STOP_TAIL_OFF, NET_STOP_TAIL_ON)
NET_STOP_TAIL_FIELD = "net_stop_tail"
_NET_STOP_TAIL_FRAMES = 3.0
#: Post-bounce samples a dead-ball close keeps: the bounce witness uses up to
#: four fronts per side within twelve frames.
_BOUNCE_TAIL_SAMPLES = 4
#: A labelled-boundary close this near an abstained bounce or net row is taken
#: to be arriving at it (see ``boundary_row_event``).
_ROW_ARRIVAL_FRAMES = 3.0
_BOUNCE_TAIL_FRAMES = 12.0
#: A later automatic bounce this far after the previous live bounce is the dead
#: ball, not a second competitive landing. The first bounce of the flight is
#: never dropped: a slow lob can be in the air longer than this.
_POST_PLAY_BOUNCE_S = 1.5
#: A hole longer than this, after the last live bounce, is the end of the track.
#: Shorter dropouts stay in the flight.
_TRACK_GAP_S = 0.40
_BORDER_RUN_MIN = 3
_AFTERMATH_SCOPES = frozenset(
    {
        "visible_aftermath",
        "observed_aftermath",
        "aftermath",
        "post_ending",
        "post_delivery_aftermath",
        "post_winner_aftermath",
    }
)
#: Span limits, not physical endings. A flight closed on one of these is partial.
SPAN_ENDPOINT_KINDS = (
    "camera_cut",
    "held_camera",
    "point_end",
    "dead_ball",
)
#: Kinds this switch attaches to the scene so the fitter can represent them.
SUPPORTED_ENDING_KINDS = ("net_stop", *SPAN_ENDPOINT_KINDS)
TERMINAL_KINDS = (
    "second_bounce",
    "terminal_bounce",
    "fov_exit",
    "wall",
    "last_visible_sample",
    "net_stop",
    *SPAN_ENDPOINT_KINDS,
)
#: Default-off. A joined multi-flight component that dies or rejects is refit
#: as one-flight children; every child that passes is kept. Explicit ``off``
#: and an absent key leave the maximal joined roster unchanged.
SPLIT_DEFAULT = "off"
SPLIT_ON = "on"
SPLIT_MODES = (SPLIT_DEFAULT, SPLIT_ON)
SPLIT_FIELD = "failed_component_split_fallback"
SPLIT_RECEIPT_FIELD = "failed_component_split"
#: Default-off. A component that exhausts its search is retried once with one
#: extra equal search share from remaining source wall. ``seed_restarts`` stays
#: the frozen numerical ``three``; this key does not add families.
#: ``exhausted_split`` does not retry the joined component. It gives the split
#: fallback's one-flight children the same search share the retry would have
#: used. Off and ``exhausted_retry`` are unchanged; with plain ``off`` the
#: children of an exhausted parent get zero search and never run.
SEARCH_BUDGET_DEFAULT = "off"
SEARCH_BUDGET_RETRY = "exhausted_retry"
SEARCH_BUDGET_SPLIT = "exhausted_split"
SEARCH_BUDGET_MODES = (SEARCH_BUDGET_DEFAULT, SEARCH_BUDGET_RETRY, SEARCH_BUDGET_SPLIT)
SEARCH_BUDGET_FIELD = "component_search_budget"
#: Label phases that already say whether a bounce is the ending. A phase that
#: does not say is not guessed: an in-court bounce still needs a second bounce,
#: an image-border exit, or a wall.
_OUT_BOUNCE_PHASES = frozenset(
    {
        "terminal_long_landing",
        "terminal_out_bounce",
        "terminal_sideline_bounce",
        "failed_serve_landing",
        "failed_serve_grounding",
        "fault_ending",
    }
)
_IN_BOUNCE_PHASES = frozenset(
    {
        "serve_landing",
        "serve_service_court_bounce",
        "service_bounce",
        "serve_bounce",
        "return_landing",
        "return_bounce",
        "return_first_bounce",
        "rally_bounce",
        "rally_landing",
        "unreturned_shot_first_bounce",
        "last_shot_first_bounce",
    }
)
_SECOND_BOUNCE_PHASES = frozenset({"unreturned_second_bounce", "terminal_second_bounce"})
_FOV_MARGIN_PX = 20.0
_FOV_IMAGE = (1920.0, 1080.0)
_FOV_MIN_VISIBLE_ROWS = 5
#: These three holds are the unknown-next rule itself. A ball-track terminal
#: may clear them. An unsupported origin, a failed coverage check, or a role
#: failure may not.
_TERMINAL_CLEARABLE_REASONS = frozenset(
    {
        "unknown_next_contact_endpoint",
        "source_boundary_barrier",
        "unsupported_multiple_ground_events",
    }
)
#: A supported-span close may also step over a dead-ball row that sits at the
#: boundary it just honoured. An overlapping row still blocks the close.
_SUPPORTED_CLEARABLE_REASONS = _TERMINAL_CLEARABLE_REASONS | frozenset(
    {"noncompetitive_source_event"}
)
_DEAD_BALL_PHASES = frozenset(
    {
        "dead_ball",
        "ball_collection",
        "collection",
        "walking",
        "players_walking",
        "ball_kid",
        "ball_selection",
        "towelling",
        # A disposal touch and the bounce after it are the dead ball, not a
        # later competitive origin. The labelled phase is the same boundary.
        "dead_ball_racket_disposal",
        "dead_ball_disposal_bounce",
    }
)
#: Written by the cascade when a confirm supports a row the decoder abstained.
#: A span trim may not use that bounce to open a flight. The string is the
#: confirm mark in ``event_cascade.apply_receipts``.
_CASCADE_CONFIRM_NOTE = "cascade confirm of a retained abstention"
_HELD_CAMERA_STATUS = frozenset(
    {"held", "held_camera", "close_up", "non_play", "cut", "camera_cut"}
)
_PLAY_CAMERA_STATUS = frozenset({"supported", "play", "play_camera", "ok"})
_NET_Y_M = 11.885


def _physical(records: list[dict], clip: str) -> list[dict]:
    return [
        {k: deepcopy(v) for k, v in e.items() if k != "clip"}
        for e in records
        if e.get("clip", clip) == clip and e.get("event_type") in prefix.PHYSICAL
    ]


def _declared_prefix(source: dict, expected: list[dict], leading: str) -> list[dict] | None:
    """Check the declared leading evidence against the original physical rows.

    Returns the leading rows when the source declares them, ``None`` when the
    source opens on its first contact. Nothing is removed or reordered here.
    """
    if leading not in LEADING_MODES:
        raise ValueError("explicit supported leading-evidence preparation mode required")
    declaration = source.get("observation_scope", {}).get(LEADING_FIELD)
    prefix = role.leading_physical_prefix(expected)
    if not prefix:
        if declaration is not None:
            raise ValueError("declared leading evidence contradicts an original contact origin")
        return None
    if leading != LEADING_PREFIX:
        raise ValueError(
            "original component source must open on its first observed contact; "
            "leading physical evidence requires the explicit declared-prefix policy"
        )
    if not isinstance(declaration, dict) or declaration.get("fabricated_events") is not False:
        raise ValueError("original source must declare its leading physical evidence explicitly")
    declared = [
        {k: deepcopy(v) for k, v in e.items() if k != "clip"} for e in declaration.get("events", [])
    ]
    if declared != prefix or declaration.get("event_count") != len(prefix):
        raise ValueError("declared leading evidence differs from the original physical records")
    return prefix


def _source(packet: dict, labels: dict, cameras: dict, partition: str, leading: str) -> dict:
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one original attempt packet required for component preparation")
    source = deepcopy(packet["attempts"][0])
    if source.get(role.FIELD) is not None or source.get("first_contact_role") == "rally":
        raise ValueError(
            "original unsplit source required; rally-scoped inputs cannot be split again"
        )
    if source.get("observation_scope", {}).get("schema") == prefix.SCHEMA:
        # Reuse the existing inverse source contract, never infer a larger source
        # from the currently retained prefix or any successful numerical output.
        prefix.validate(source, cameras, observation_partition=partition, labels=labels)
        source.update(
            events=deepcopy(source["original_physical_events"]),
            owner_end_frame=source["original_owner_end_frame"],
            owner_end_frame_semantics=source["original_owner_end_frame_semantics"],
            observation_scope=deepcopy(source["original_observation_scope"]),
        )
    declaration = source.get("observation_scope", {})
    if declaration.get("schema") != prefix.UNRESOLVED_INPUT_SCHEMA:
        raise ValueError("original unresolved-ending source inventory required")
    if (
        declaration.get("ending_semantics") != "unresolved"
        or declaration.get("physical_ending") is not None
    ):
        raise ValueError("component source cannot assert a physical ending")
    clip = source["point_clip"]
    meta = labels.get("attempt", {})
    if (
        meta.get("clip", clip) != clip
        or labels.get("match_id", source["match_id"]) != source["match_id"]
    ):
        raise ValueError("component source label identity differs from packet")
    window = declaration.get("native_window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(v) is not int for v in window)
        or window[0] >= window[1]
    ):
        raise ValueError("ordered original native window required")
    prefix._checked_events(source["events"], window)
    expected = _physical(source["events"], clip)
    if len(expected) != len(source["events"]):
        raise ValueError("all original physical events must belong to the source clip")
    original = _physical(labels["events"]["records"], clip)
    # The source packet owns its declared physical inventory. Labels may retain
    # later noncompetitive context; keep it in the plan without promoting those
    # rows to new competitive origins or a physical endpoint.
    low, high = expected[0]["frame"], expected[-1]["frame"]
    if [e for e in original if low <= e["frame"] <= high] != expected:
        raise ValueError("original component inventory differs from source labels")
    _declared_prefix(source, expected, leading)
    for event in original:
        if event["event_type"] != "contact" or low <= event["frame"] <= high:
            continue
        if window[0] <= event["frame"] <= source["owner_end_frame"] and not (
            event.get("competitive_event") is False
            or role.source_record(event).get("competitive_origin") is False
        ):
            raise ValueError(
                "competitive source contact outside original inventory cannot be omitted"
            )
    horizon = float(source["owner_end_frame"])
    if (
        not math.isfinite(horizon)
        or not window[0] <= horizon <= window[1]
        or declaration.get("observation_horizon") != horizon
    ):
        raise ValueError("original native horizon must remain unchanged")
    frames = [r["frame"] for r in source["owner_ball_labels"]]
    if any(
        type(f) is not int or not window[0] <= f <= window[1] for f in frames
    ) or frames != sorted(set(frames)):
        raise ValueError("ordered unique original native ball rows required")
    ball_records = [r for r in labels["ball"]["records"] if r.get("clip") == clip]
    if (
        not frames
        or len(ball_records) != 1
        or [row for row in ball_records[0]["frames"] if frames[0] <= row["frame"] <= frames[-1]]
        != source["owner_ball_labels"]
    ):
        raise ValueError("original component observations differ from source labels")
    return source


def _unsupported_spans(source: dict) -> list[dict]:
    """Original contact-to-contact spans no single competitive flight can own.

    Derived from the retained physical rows alone: more than one supplied bounce
    between two original contacts. A net hit alongside one bounce is allowed.
    The span is named, never repaired and never removed; its origin
    slot stays unresolved and its reference origins stay in the denominator.
    """
    contacts = [e for e in source["events"] if e["event_type"] == "contact"]
    spans = []
    for index, contact in enumerate(contacts[1:], 1):
        # Match the exact original row. Removing clip from only this side breaks
        # identity for valid labeled observations that retain clip on each event.
        span = role.preceding_original_span(source["events"], contact)
        if span["single_competitive_flight_representable"]:
            continue
        spans.append(
            dict(
                original_contact_index=index,
                start_frame=span["previous_original_contact_frame"],
                end_frame=float(contact["frame"]),
                **{k: v for k, v in span.items() if k != "previous_original_contact_frame"},
                reason="more than one supplied ground event between two original contacts",
                interpretation=(
                    "at least one original row is a missing contact, a false positive or the "
                    "end of the attempt; the source does not say which"
                ),
            )
        )
    return spans


def source_routing_receipt(attempt: dict, qualification: dict, *, mode: str) -> dict:
    """Receipt for retaining the original inventory after a refused prefix cut.

    Nothing is modelled here: the source keeps its own unresolved-ending
    declaration, every original row, its native window and its horizon, and the
    component partition decides ownership later from the same inputs.
    """
    if mode != ROUTING_UNSUPPORTED_SPAN:
        raise ValueError("explicit supported component source routing mode required")
    if qualification.get("status") == "qualified":
        raise ValueError("a qualified contact prefix owns its source")
    declaration = attempt.get("observation_scope", {})
    if declaration.get("schema") != prefix.UNRESOLVED_INPUT_SCHEMA:
        raise ValueError("component source routing requires the unresolved-ending inventory")
    if (
        declaration.get("ending_semantics") != "unresolved"
        or declaration.get("physical_ending") is not None
    ):
        raise ValueError("a routed component source cannot assert a physical ending")
    events = attempt["events"]
    if declaration.get("original_event_observations") != events:
        raise ValueError("routed component source differs from its declared original inventory")
    contacts = [e for e in events if e["event_type"] == "contact"]
    if not contacts:
        raise ValueError("a routed component source requires an original contact")
    spans = _unsupported_spans(attempt)
    return dict(
        schema=ROUTING_SCHEMA,
        mode=mode,
        status="original_inventory_retained",
        contact_prefix_mode=qualification.get("mode"),
        contact_prefix_status=qualification.get("status"),
        contact_prefix_refusal=qualification.get("reason"),
        contact_prefix_bound=False,
        source_contract=prefix.UNRESOLVED_INPUT_SCHEMA,
        owner="contact components partition the original inventory before any fit",
        original_physical_event_count=len(events),
        original_contact_count=len(contacts),
        original_contact_frames=[float(e["frame"]) for e in contacts],
        original_events_sha256=record_hash(events),
        native_window=list(declaration["native_window"]),
        observation_horizon=declaration["observation_horizon"],
        unsupported_original_spans=spans,
        unsupported_original_span_count=len(spans),
        original_rows_dropped=0,
        original_rows_inserted=0,
        native_epochs_modified=False,
        complete_original_source=False,
        physical_ending=None,
        ending_semantics="unresolved",
        reference_denominator_changed=False,
        fitted_states_consulted=False,
        reference_rows_consulted=False,
    )


def route_refused_prefix(attempt: dict, qualification: dict, settings: dict) -> dict | None:
    """Who owns a source whose contact-prefix cut refused: components, or nobody.

    Returns the routing receipt under the explicit shared policy, and ``None``
    when the original refusal stands. Shared by every preparation entrypoint so
    the supplied-clip runtime and the automatic runtime route identically.
    """
    if not prefix.component_source_routing(
        settings.get(ROUTING_FIELD, ROUTING_DEFAULT),
        settings["contact_prefix_scope"],
        settings.get("contact_components", DEFAULT_MODE),
    ):
        return None
    return source_routing_receipt(attempt, qualification, mode=settings[ROUTING_FIELD])


def _automatic_net_ending(event: dict) -> bool:
    """Consumer marker the automatic prepare writes beside a terminal net hit.

    It names the same net. It is not a point ending and not a second boundary.
    """
    return (
        event.get("event_type") == "ending"
        and event.get("annotation_origin") == "automatic"
        and event.get("status") == "predicted"
        and event.get("exact_epoch_observed") is False
        and "net" in str(event.get("ending_kind") or "")
    )


def _automatic_net_ending_marker(event: dict, labels: dict) -> bool:
    """True when that marker sits on the same frame as a net hit."""
    if not _automatic_net_ending(event):
        return False
    try:
        frame = float(event["frame"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(frame):
        return False
    for row in labels["events"]["records"]:
        if row is event or row.get("event_type") != "net_hit":
            continue
        try:
            other = float(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(other) and other == frame:
            return True
    return False


def _barriers(source: dict, labels: dict) -> list[dict]:
    barriers = []

    def add(kind: str, interval: list, **identity) -> None:
        if (
            len(interval) != 2
            or not all(math.isfinite(float(f)) for f in interval)
            or interval[0] > interval[1]
        ):
            raise ValueError("finite ordered source barrier interval required")
        barriers.append(dict(kind=kind, frame_interval=list(interval), **identity))

    previous_identity = None
    for index, event in enumerate(source["events"]):
        if not role.supported(event):
            add("unresolved_occurrence", event["frame_interval"], original_event_index=index)
        original = role.source_record(event)
        identity = original.get("attempt_id")
        if (
            previous_identity is not None and identity is not None and identity != previous_identity
        ) or event.get("attempt_id", identity) != identity:
            add(
                "original_attempt_membership_conflict",
                event["frame_interval"],
                original_event_index=index,
            )
        if identity is not None:
            previous_identity = identity
        if event.get("competitive_event") is False or (
            event["event_type"] == "contact" and original.get("competitive_origin") is False
        ):
            add("noncompetitive_source_event", event["frame_interval"], original_event_index=index)
    for index, interval in enumerate(
        source["observation_scope"].get("unresolved_boundary_intervals", [])
    ):
        add("unresolved_source_boundary", interval, source_boundary_index=index)
    for frame in source.get("unannotated_context_frames", []):
        add("unannotated_context", [frame, frame])
    for index, event in enumerate(labels["events"]["records"]):
        if event.get("clip", source["point_clip"]) == source["point_clip"] and event.get(
            "event_type"
        ) in ("source_cut", "ending"):
            if event["event_type"] == "ending" and role.supported(event):
                # An automatic net ending is the consumer marker beside the
                # net_hit. The endings switch already closes that flight. It is
                # not a resolved point ending, and it must not veto the
                # component partition when the whole-point net-stop seed dies.
                if not _automatic_net_ending_marker(event, labels):
                    raise ValueError(
                        "resolved source ending contradicts unresolved component preparation"
                    )
                continue
            add(
                "declared_source_boundary", event.get("frame_interval", []), label_event_index=index
            )
    contacts = [e for e in source["events"] if e["event_type"] == "contact"]
    for index, event in enumerate(contacts[1:], 1):
        if role.typed_serve(event):
            add("later_typed_serve", event["frame_interval"], original_contact_index=index)
    return barriers


def _origin_role(
    source: dict, contact: dict, index: int, labels: dict, routing: str = ROUTING_DEFAULT
) -> tuple[str, dict | None]:
    # This temporary identity carrier is not a cropped executable attempt.
    identity = dict(match_id=source["match_id"], point_clip=source["point_clip"], events=[contact])
    contacts = [e for e in source["events"] if e["event_type"] == "contact"]
    untyped = (
        all(contact.get(k) is None for k in ("role", "stroke", "shot_type"))
        and contact.get("serve_number") is None
    )
    unresolved_origin = untyped and not role.typed_serve(contacts[0])
    if unresolved_origin and role.leading_physical_prefix(source["events"]):
        # Leading evidence proves the original origin was not observed. It does
        # not prove a rally: the clip may hold a fault and then its serve, so
        # every untyped origin here stays unknown and serve priors go off.
        bound = role.bind_unknown(identity, source["events"])
        return "unknown", role.validate(bound, labels, requested="unknown")
    if unresolved_origin and index > 0 and routing == ROUTING_UNSUPPORTED_SPAN:
        # No original contact declares a serve, so the single-attempt rally
        # chain cannot be walked back to one. A later origin is then unresolved
        # rather than a demonstrated rally stroke: the preceding span may hide a
        # contact, a false positive or the end of the earlier attempt.
        bound = role.bind_unknown(
            identity, source["events"], derivation=role.UNDECLARED_SERVE_ORIGIN
        )
        return "unknown", role.validate(bound, labels, requested="unknown")
    if index == 0:
        if any(
            str(e.get(k, "serve")).lower() != "serve"
            for e in (contact, role.source_record(contact))
            for k in ("role", "stroke", "shot_type")
            if e.get(k) is not None
        ):
            raise ValueError(
                "original first contact declares a nonserve; cannot apply legacy serve semantics"
            )
        return ("serve" if role.typed_serve(contact) else "unspecified"), None
    typed = [contact[k] for k in ("role", "stroke", "shot_type") if contact.get(k) is not None]
    derivation = (
        role.EXPLICIT_RALLY if typed and all(v == "rally" for v in typed) else role.SINGLE_ATTEMPT
    )
    bound = role.bind_rally(identity, source["events"], derivation=derivation)
    return "rally", role.validate(bound, labels, requested="rally")


def _inventory_window(attempt: dict) -> list[int]:
    """Integer native window covering every original event interval and the horizon."""
    events = attempt["events"]
    if not events:
        raise ValueError("original physical events required")
    lo = min(int(math.floor(float(e["frame_interval"][0]))) for e in events)
    hi = max(int(math.ceil(float(e["frame_interval"][1]))) for e in events)
    hi = max(hi, int(math.ceil(float(attempt["owner_end_frame"]))))
    for row in attempt.get("owner_ball_labels") or ():
        frame = int(row["frame"])
        lo, hi = min(lo, frame), max(hi, frame)
    declared = (attempt.get("observation_scope") or {}).get("native_window")
    if isinstance(declared, list) and len(declared) == 2 and all(type(v) is int for v in declared):
        lo, hi = min(lo, declared[0]), max(hi, declared[1])
    if lo >= hi:
        raise ValueError("ordered original native window required")
    return [lo, hi]


def as_unresolved_source(packet: dict) -> dict:
    """View an observed-ground unresolved-horizon packet as a component inventory.

    Whole-point numerical seed death is the only caller. Ordinary searches and
    already-unresolved inventories are unchanged. Missing cameras and poses are
    not repaired here.
    """
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one original attempt packet required for component preparation")
    result = deepcopy(packet)
    attempt = result["attempts"][0]
    scope = attempt.get("observation_scope") or {}
    if scope.get("schema") == prefix.UNRESOLVED_INPUT_SCHEMA:
        return result
    from cv.pipeline import s6_component_scope

    if s6_component_scope.active(attempt):
        raise ValueError("a bound component cannot be re-inventoried as a source")
    if scope.get("physical_ending") is not None:
        raise ValueError("a resolved physical ending cannot become a seed-failure inventory")
    window = _inventory_window(attempt)
    contract = prefix.unresolved_input_contract(
        attempt["events"],
        window,
        event_origin=scope.get("event_origin") or "label",
        boundaries=scope.get("unresolved_boundary_intervals") or [],
        refusal="attempt_seed_failure",
        segmentation_source=scope.get("segmentation_source"),
    )
    contract["observation_horizon"] = float(attempt["owner_end_frame"])
    attempt["observation_scope"] = contract
    return result


def terminal_bounce_allowance(kind: str) -> frozenset[int]:
    """How many post-contact bounces the named ball-track ending contains."""
    if kind == "second_bounce":
        return frozenset({2})
    if kind == "terminal_bounce":
        return frozenset({1})
    if kind in {"fov_exit", "wall"}:
        return frozenset({0, 1})
    # The net is the ending. At most one bounce before it is an anchor; a second
    # bounce would already have closed the flight.
    if kind == "net_stop":
        return frozenset({0, 1})
    # The last visible sample is not a bounce count. Every bounce that actually
    # sits before that sample stays in the flight; the acceptance checks judge it.
    # A camera cut, held camera, point end or dead-ball close is the same: the
    # span stops, and the bounces that remain are judged rather than counted.
    if kind == "last_visible_sample" or kind in SPAN_ENDPOINT_KINDS:
        return frozenset(range(12))
    raise ValueError("explicit ball-track terminal kind required")


def _event_phase(event: dict) -> str | None:
    phase = role.source_record(event).get("phase")
    if phase is None:
        phase = event.get("phase")
    return str(phase) if phase is not None else None


def _bounce_in_court(event: dict) -> bool | None:
    """True in, False out, None when the event does not say.

    A missing court reading is not an out. The second-bounce rule needs a
    known in bounce; a single bounce is terminal only when it is known out.
    """
    phase = _event_phase(event)
    if phase in _OUT_BOUNCE_PHASES:
        return False
    if phase in _IN_BOUNCE_PHASES or phase in _SECOND_BOUNCE_PHASES:
        return True
    location = event.get("location") or role.source_record(event).get("location") or {}
    try:
        x = float(location["court_x_fraction"])
        y = float(location["court_y_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    low, high = -0.02, 1.02
    return low <= x <= high and low <= y <= high


def _court_metres(event: dict) -> tuple[float, float] | None:
    """Ground position of a bounce on a reliable per-frame camera, else None.

    Automatic rows carry ``automatic_location``. A held, static-fallback or
    missing court transport is not a reading.
    """
    record = role.source_record(event)
    location = None
    for holder in (event, record):
        for key in ("automatic_location", "location"):
            value = holder.get(key)
            if isinstance(value, dict):
                location = value
                break
        if location is not None:
            break
    if location is None:
        return None
    transport = location.get("court_transport")
    if not isinstance(transport, dict):
        return None
    if transport.get("reliable_per_frame") is not True or transport.get("static_fallback"):
        return None
    if transport.get("held"):
        return None
    try:
        x = float(location["court_x_fraction"]) * _COURT_WIDTH_M
        y = float(location["court_y_fraction"]) * _COURT_LENGTH_M
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def out_bounce_call(events: list[dict], margin_m: float = OUT_BOUNCE_MARGIN_M) -> dict | None:
    """First supported bounce that is out by more than ``margin_m``, as a receipt.

    Out means outside the singles court, for every bounce including the serve's
    landing: the serve box is not judged, because a missed serve bounce would
    make the return's deep landing look like a long serve. A labelled out phase
    is out. A bounce with no reliable per-frame court reading is not judged.
    """
    for event in sorted(events, key=lambda row: float(row["frame"])):
        if event.get("event_type") != "bounce" or not role.supported(event):
            continue
        frame = float(event["frame"])
        if _event_phase(event) in _OUT_BOUNCE_PHASES:
            return dict(frame=frame, evidence="labelled_out_phase")
        metres = _court_metres(event)
        if metres is None:
            continue
        x, y = metres
        wide = abs(x - 0.5 * _COURT_WIDTH_M) - _SINGLES_HALF_WIDTH_M
        deep = max(-y, y - _COURT_LENGTH_M)
        distance = max(wide, deep)
        if distance > margin_m:
            return dict(
                frame=frame,
                evidence="court_reading_out",
                court_xy_m=[x, y],
                out_by_m=distance,
                margin_m=margin_m,
            )
    return None


def _is_wall(event: dict) -> bool:
    if event.get("event_type") == "wall":
        return True
    phase = _event_phase(event) or ""
    return "wall" in phase.lower()


def _on_image_border(x: float, y: float) -> bool:
    width, height = _FOV_IMAGE
    return (
        x <= _FOV_MARGIN_PX
        or y <= _FOV_MARGIN_PX
        or x >= width - _FOV_MARGIN_PX
        or y >= height - _FOV_MARGIN_PX
    )


def _terminal_receipt(
    kind: str, frame: float, *, evidence: str, bounce_frames: list, **extra
) -> dict:
    receipt = {
        "kind": kind,
        "frame": float(frame),
        "evidence": evidence,
        "bounce_frames": [float(value) for value in bounce_frames],
        "requires_next_contact": False,
    }
    receipt.update(extra)
    return receipt


def _fov_exit_frame(
    source: dict, start: float, next_frame: float | None, limit: float | None = None
) -> float | None:
    """Last in-picture frame when the ball leaves at the border and does not return.

    A track that stops in the interior, or that stops because a later contact
    exists, is not an exit. Those are the open-ending failure: fitting a dead
    ball or a rally that still has a hitter.
    """
    if next_frame is not None:
        return None
    rows = []
    for row in source.get("owner_ball_labels") or ():
        if row.get("status") != "visible":
            continue
        frame = float(row["frame"])
        if frame <= start or (limit is not None and frame >= float(limit)):
            continue
        try:
            x = float(row["x1080"])
            y = float(row["y1080"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        rows.append((frame, x, y))
    if len(rows) < _FOV_MIN_VISIBLE_ROWS:
        return None
    rows.sort()
    if not _on_image_border(rows[-1][1], rows[-1][2]):
        return None
    if not any(not _on_image_border(x, y) for _, x, y in rows[:-1]):
        return None
    return rows[-1][0]


def _last_visible_ball_frame(
    source: dict, start: float, horizon: float, limit: float | None = None
) -> float | None:
    """Last visible ball sample after the contact, never past the declared horizon.

    The frame is an observed row. A missing tail is not extended, and a sample
    before or at the contact is not an ending.
    """
    last = None
    cap = float(horizon) if limit is None else min(float(horizon), float(limit) - 1e-9)
    for row in source.get("owner_ball_labels") or ():
        if row.get("status") != "visible":
            continue
        frame = float(row["frame"])
        if frame <= start or frame > cap:
            continue
        try:
            x = float(row["x1080"])
            y = float(row["y1080"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if last is None or frame > last:
            last = frame
    return last


def _ball_track_terminal(
    source: dict, contact: dict, next_frame: float | None, limit: float | None = None
) -> dict | None:
    """Ending evidenced by this flight's own bounces, border exit, or wall.

    Stops at the terminal. A later contact, bounce, or picture is post-ending
    and is not part of the receipt. Nothing here invents an event.
    """
    start = float(contact["frame"])
    later = []
    for event in source["events"]:
        frame = float(event["frame"])
        if frame <= start:
            continue
        if next_frame is not None and frame >= float(next_frame):
            continue
        if limit is not None and frame >= float(limit):
            continue
        later.append(event)
    bounces = [
        event for event in later if event["event_type"] == "bounce" and role.supported(event)
    ]
    if bounces:
        first_in = _bounce_in_court(bounces[0])
        if first_in is False:
            return _terminal_receipt(
                "terminal_bounce",
                bounces[0]["frame"],
                evidence="first_bounce_out",
                bounce_frames=[bounces[0]["frame"]],
            )
        if len(bounces) >= 2 and (
            first_in is True or _event_phase(bounces[1]) in _SECOND_BOUNCE_PHASES
        ):
            return _terminal_receipt(
                "second_bounce",
                bounces[1]["frame"],
                evidence="second_bounce_after_in_bounce",
                bounce_frames=[bounces[0]["frame"], bounces[1]["frame"]],
            )
        if len(bounces) >= 2:
            return None
    walls = [event for event in later if _is_wall(event) and role.supported(event)]
    if walls and not bounces:
        return _terminal_receipt("wall", walls[0]["frame"], evidence="wall_event", bounce_frames=[])
    exit_frame = _fov_exit_frame(source, start, next_frame, limit)
    if exit_frame is None:
        return None
    kept = [float(event["frame"]) for event in bounces if float(event["frame"]) <= exit_frame]
    if len(kept) > 1:
        return None
    return _terminal_receipt(
        "fov_exit", exit_frame, evidence="image_border_exit", bounce_frames=kept
    )


def _boundary_kind(event: dict) -> str | None:
    """Camera cut, held camera, point end or dead ball. Anything else is not a span limit."""
    event_type = str(event.get("event_type") or "")
    phase = str(_event_phase(event) or "").lower()
    if event_type in {"source_cut", "camera_cut"} or phase in {
        "camera_cut",
        "shot_boundary",
        "cut",
    }:
        return "camera_cut"
    if event_type == "held_camera" or phase in {"held_camera", "close_up"}:
        return "held_camera"
    if event_type in {"ending", "point_end"}:
        if _automatic_net_ending(event):
            return None
        return "point_end"
    if event_type == "collection" or phase in _DEAD_BALL_PHASES:
        return "dead_ball"
    return None


def _boundary_frame(event: dict) -> float | None:
    interval = event.get("frame_interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        try:
            frame = float(interval[0])
        except (TypeError, ValueError):
            frame = None
        else:
            if math.isfinite(frame):
                return frame
    try:
        frame = float(event["frame"])
    except (KeyError, TypeError, ValueError):
        return None
    return frame if math.isfinite(frame) else None


def _first_boundary(
    source: dict, labels: dict, cameras: dict, start: float, horizon: float
) -> tuple[float, str] | None:
    """Earliest span limit strictly after this contact and before ``horizon``.

    A point-end that is the declared horizon is not an interior boundary.
    A held camera counts only when this flight was on a play camera and then left it.
    """
    best: tuple[float, str] | None = None
    owner_end = float(source["owner_end_frame"])

    def offer(frame: float, kind: str) -> None:
        nonlocal best
        if kind == "point_end" and frame >= owner_end - 1e-3:
            return
        if not start < frame < horizon:
            return
        if best is None or frame < best[0]:
            best = (frame, kind)

    clip = source["point_clip"]
    seen = list(source["events"])
    for event in labels.get("events", {}).get("records", []):
        if event.get("clip", clip) == clip:
            seen.append(event)
    for event in seen:
        kind = _boundary_kind(event)
        frame = _boundary_frame(event) if kind is not None else None
        if kind is not None and frame is not None:
            offer(frame, kind)
    rows = []
    for row in (cameras or {}).get("cameras") or ():
        try:
            frame = float(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        status = str(row.get("status") or "")
        if math.isfinite(frame) and status:
            rows.append((frame, status))
    rows.sort()
    if rows:
        at_contact = [status for frame, status in rows if frame <= start]
        armed = bool(at_contact) and at_contact[-1] in _PLAY_CAMERA_STATUS
        for frame, status in rows:
            if frame <= start:
                continue
            if status in _PLAY_CAMERA_STATUS:
                armed = True
                continue
            if armed and status in _HELD_CAMERA_STATUS:
                offer(frame, "camera_cut" if status in {"cut", "camera_cut"} else "held_camera")
                break
    return best


def _event_scope(event: dict) -> str | None:
    scope = event.get("original_scope")
    if scope is None:
        scope = role.source_record(event).get("scope")
    return str(scope) if scope is not None else None


def _is_aftermath_event(event: dict) -> bool:
    """A labelled dead-ball row. An observed continuation is not this.

    ``observed_continuation`` still names a landing the point may own. Only an
    aftermath scope, or a phase that already says the ball is dead, is the
    boundary after live play.
    """
    if _event_scope(event) in _AFTERMATH_SCOPES:
        return True
    phase = str(_event_phase(event) or "").lower()
    return phase in _DEAD_BALL_PHASES or phase.startswith("dead_ball")


def _labeled_competitive(event: dict) -> bool:
    """A supplied label, not an automatic or cascade row.

    The 1.5 s rule must not drop a labelled second bounce. Those stay live
    even when they land slowly. Aftermath labels are not competitive.
    """
    if _is_aftermath_event(event):
        return False
    return event.get("status") == "labeled" or event.get("annotation_origin") == "agent"


def _extra_label_events(source: dict, labels: dict) -> list[dict]:
    """Label rows past the inventory. The source list itself is not repeated."""
    clip = source["point_clip"]
    owned = {(event["event_type"], float(event["frame"])) for event in source["events"]}
    extra = []
    for event in labels.get("events", {}).get("records", []):
        if event.get("clip", clip) != clip:
            continue
        try:
            frame = float(event["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(frame):
            continue
        if (event.get("event_type"), frame) in owned:
            continue
        extra.append(event)
    return extra


def _ownership_split(
    source: dict, labels: dict, start: float, window: float
) -> tuple[float | None, list[dict]]:
    """Live bounces before the first post-play row, and that row's frame.

    A labelled aftermath boundary ends live play. So does an automatic bounce
    more than 1.5 s after the previous live bounce. The flight's first bounce
    is kept: time from the contact is not this test.
    """
    fps = float(source.get("fps") or 25.0)
    rows = []
    for event in list(source["events"]) + _extra_label_events(source, labels):
        try:
            frame = float(event["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < frame < window and math.isfinite(frame):
            rows.append(event)
    rows.sort(key=lambda event: float(event["frame"]))
    live: list[dict] = []
    for event in rows:
        frame = float(event["frame"])
        if _is_aftermath_event(event):
            return frame, live
        if event.get("event_type") != "bounce" or not role.supported(event):
            continue
        if (
            live
            and not _labeled_competitive(event)
            and frame - float(live[-1]["frame"]) > _POST_PLAY_BOUNCE_S * fps
        ):
            return frame, live
        live.append(event)
    return None, live


def _visible_tail(source: dict, after: float, horizon: float) -> list[tuple[float, float, float]]:
    rows = []
    for row in source.get("owner_ball_labels") or ():
        if row.get("status") != "visible":
            continue
        try:
            frame = float(row["frame"])
            x = float(row["x1080"])
            y = float(row["y1080"])
        except (KeyError, TypeError, ValueError):
            continue
        if not after < frame < horizon:
            continue
        if not (math.isfinite(frame) and math.isfinite(x) and math.isfinite(y)):
            continue
        rows.append((frame, x, y))
    rows.sort()
    return rows


def _tail_cap(source: dict, after: float, horizon: float) -> tuple[float, str] | None:
    """First picture exit or pre-gap frame strictly after ``after``.

    A one-frame clip of the border is not an exit. A hole shorter than
    ``_TRACK_GAP_S`` is not a new track. The frame is the start of the border
    run, or the last sample before the hole.
    """
    fps = float(source.get("fps") or 25.0)
    rows = _visible_tail(source, after, horizon)
    if not rows:
        return None
    gap_frame = None
    gap_at = _TRACK_GAP_S * fps
    for left, right in zip(rows, rows[1:]):
        if right[0] - left[0] > gap_at:
            gap_frame = left[0]
            break
    border_frame = None
    run = 0
    run_start = None
    for frame, x, y in rows:
        if gap_frame is not None and frame > gap_frame:
            break
        if _on_image_border(x, y):
            if run == 0:
                run_start = frame
            run += 1
            if run >= _BORDER_RUN_MIN:
                border_frame = run_start
                break
        else:
            run = 0
            run_start = None
    options = []
    if gap_frame is not None:
        options.append((gap_frame, "track_gap"))
    if border_frame is not None:
        options.append((border_frame, "picture_exit"))
    if not options:
        return None
    return min(options)


def _ownership_candidate(
    source: dict,
    start: float,
    live: list[dict],
    limit: float,
    horizon: float,
    boundary_endings: str = BOUNDARY_ENDINGS_OFF,
    tail_ceiling: float | None = None,
) -> dict | None:
    """Close at the tail cap, or at the last live bounce, once play has ended.

    One live bounce and a later post-play row. A picture exit or a track gap
    between them keeps that tail. Otherwise the bounce is the endpoint and the
    dead-ball arc after it is not fitted.
    """
    if len(live) != 1:
        return None
    bounce = float(live[0]["frame"])
    window = min(float(horizon), float(limit))
    capped = _tail_cap(source, bounce, window)
    if capped is not None and bounce < capped[0] < window:
        frame, reason = capped
        kind = "fov_exit" if reason == "picture_exit" else "last_visible_sample"
        return _terminal_receipt(
            kind,
            frame,
            evidence="image_border_exit" if reason == "picture_exit" else "track_gap",
            bounce_frames=[bounce],
        )
    if not math.isfinite(bounce) or bounce <= start:
        return None
    end = bounce
    extra = {}
    if boundary_endings == BOUNDARY_ENDINGS_ON:
        # The tail stops before the next labelled boundary row, so the close
        # never crosses the row that ended live play.
        ceiling = window if tail_ceiling is None else min(window, float(tail_ceiling))
        tail = _bounce_tail(source, bounce, ceiling)
        if tail is not None:
            end = tail
            extra = {"post_bounce_tail_frame": tail}
    return _terminal_receipt(
        "dead_ball",
        end,
        evidence="last_live_bounce_before_post_play",
        bounce_frames=[bounce],
        partial_flight=True,
        boundary_frame=float(limit),
        **extra,
    )


def _bounce_tail(source: dict, bounce: float, window: float) -> float | None:
    """Last of up to four visible samples just after the bounce, before ``window``.

    Closing exactly on the bounce leaves the fitted ground impact outside the
    flight. Two samples at least, or the close stays on the bounce.
    """
    horizon = min(float(window), float(bounce) + _BOUNCE_TAIL_FRAMES + 1e-6)
    rows = _visible_tail(source, float(bounce), horizon)[:_BOUNCE_TAIL_SAMPLES]
    if len(rows) < 2:
        return None
    return float(rows[-1][0])


def _first_barrier_start(barriers: list[dict], crossing: list[int], start: float) -> float | None:
    """Earliest crossing barrier that begins after the contact."""
    starts = [
        float(barriers[index]["frame_interval"][0])
        for index in crossing
        if float(barriers[index]["frame_interval"][0]) > start
    ]
    return min(starts) if starts else None


def _labelled_boundary_terminal(
    source: dict,
    contact: dict,
    barriers: list[dict],
    crossing: list[int],
    next_frame: float | None,
    limit: float | None,
    net_stop_tail: str = NET_STOP_TAIL_OFF,
) -> dict | None:
    """Close before the first labelled boundary row after this contact.

    The row itself is not modelled and no event is invented. A net stop is kept
    when every row it crosses contains the net frame (the row is that net). A
    row that already covers the contact leaves the slot held.
    """
    start = float(contact["frame"])
    net = _net_stop_terminal(source, contact, next_frame, limit, net_stop_tail)
    if net is not None and len(net["bounce_frames"]) <= 1:
        net_frame = float(net["net_frame"])
        blocking = [
            index
            for index in crossing
            if barriers[index]["frame_interval"][0] <= float(net["frame"])
            and barriers[index]["frame_interval"][1] >= start
            and not (
                barriers[index]["frame_interval"][0]
                <= net_frame
                <= barriers[index]["frame_interval"][1]
            )
        ]
        if not blocking:
            return net
    if any(
        barriers[index]["frame_interval"][0] <= start <= barriers[index]["frame_interval"][1]
        for index in crossing
    ):
        return None
    first = _first_barrier_start(barriers, crossing, start)
    if first is None:
        return None
    if limit is not None:
        first = min(first, float(limit))
    candidate = _span_terminal(source, contact, (first, "point_end"))
    if candidate is None:
        return None
    candidate["evidence"] = "labelled_boundary"
    # An automatic abstention right after the close is a likely landing or net
    # touch the detector would not commit to. Acceptance checks the partial
    # flight is actually arriving at it.
    row_events = sorted(
        {
            source["events"][barriers[index]["original_event_index"]]["event_type"]
            for index in crossing
            if "original_event_index" in barriers[index]
            and float(barriers[index]["frame_interval"][0]) == first
            and "automatic_abstention" in source["events"][barriers[index]["original_event_index"]]
        }
        & {"bounce", "net_hit"}
    )
    if len(row_events) == 1 and first - float(candidate["frame"]) <= _ROW_ARRIVAL_FRAMES:
        candidate["boundary_row_event"] = row_events[0]
    return _accept_terminal(candidate, barriers, crossing, start)


def _pull_tail(terminal: dict, source: dict, start: float, horizon: float) -> dict:
    """Stop a last-sample or span tail at the first exit or track gap.

    A net stop, a bounce ending and an exit that is already the terminal stay
    where they are. Pulling back cannot move the end onto or before the last
    live bounce.
    """
    if terminal["kind"] not in {"last_visible_sample", *SPAN_ENDPOINT_KINDS}:
        return terminal
    if terminal.get("evidence") in {
        "track_gap",
        "image_border_exit",
        "last_live_bounce_before_post_play",
    }:
        return terminal
    after = float(start)
    if terminal["bounce_frames"]:
        after = max(float(frame) for frame in terminal["bounce_frames"])
    capped = _tail_cap(source, after, min(float(horizon), float(terminal["frame"]) + 1e-6))
    if capped is None:
        return terminal
    frame, reason = capped
    if not after < frame < float(terminal["frame"]) - 1e-6:
        return terminal
    updated = dict(terminal)
    updated["frame"] = float(frame)
    updated["bounce_frames"] = [
        float(value) for value in terminal["bounce_frames"] if float(value) <= frame + 1e-9
    ]
    updated["tail_cap"] = reason
    return updated


def _is_net_stop_event(event: dict) -> bool:
    """A net that ends the flight. A cord that continues is not this event."""
    if event.get("event_type") == "contact":
        return False
    if event.get("event_type") == "net_hit":
        return True
    phase = str(_event_phase(event) or "").lower()
    return phase in {"net", "net_hit", "into_net", "net_stop", "serve_net_fault"}


def _net_horizon(net_frame: float, window: float, frames: float = 1.0) -> float | None:
    """One frame past the net (``frames``), still strictly inside the supported window.

    The net hit has to sit inside the flight, not on its endpoint. The extra
    frame is the short drop, not a licence to fit the rest of the point.
    """
    end = min(float(net_frame) + float(frames), float(window) - 1e-3)
    if not math.isfinite(end) or end <= float(net_frame):
        return None
    return end


def _net_stop_terminal(
    source: dict,
    contact: dict,
    next_frame: float | None,
    limit: float | None,
    net_stop_tail: str = NET_STOP_TAIL_OFF,
) -> dict | None:
    """Net stop when a supported net event ends the flight.

    A later contact, or an in-court bounce after the net, is a cord or a rally
    that continues. Those keep the ordinary ending. Post-tape velocity of a cord
    stays the admissible-set response; this terminal is the ball that stops.
    Monocular pixels are not a net: a ray through the net line is not a position.
    """
    start = float(contact["frame"])
    window = float(source["owner_end_frame"]) + 1.0
    if next_frame is not None:
        window = min(window, float(next_frame))
    if limit is not None:
        window = min(window, float(limit))
    # The next contact is inside this window: the flight continues to a hitter.
    if next_frame is not None and float(next_frame) < window + 1e-9:
        return None
    nets = []
    bounces_before = []
    ordered = sorted(source["events"], key=lambda event: float(event["frame"]))
    for event in ordered:
        frame = float(event["frame"])
        if frame <= start or frame >= window:
            continue
        if not role.supported(event):
            continue
        if _is_net_stop_event(event):
            nets.append(event)
        elif event["event_type"] == "bounce" and not nets:
            bounces_before.append(event)
        elif event["event_type"] == "bounce" and nets:
            # A named competitive landing, or a later in-court bounce, means the
            # ball continued (a cord). A short drop beside the net does not.
            phase = _event_phase(event)
            tail = 0.6 * float(source.get("fps") or 25.0)
            if (
                phase in _IN_BOUNCE_PHASES
                or phase in _SECOND_BOUNCE_PHASES
                or (_bounce_in_court(event) is True and frame > float(nets[0]["frame"]) + tail)
            ):
                return None
    if not nets or len(bounces_before) > 1:
        return None
    net_frame = float(nets[0]["frame"])
    end = _net_horizon(
        net_frame, window, _NET_STOP_TAIL_FRAMES if net_stop_tail == NET_STOP_TAIL_ON else 1.0
    )
    if end is None:
        return None
    return _terminal_receipt(
        "net_stop",
        end,
        evidence="net_event",
        bounce_frames=[float(event["frame"]) for event in bounces_before],
        net_frame=net_frame,
        partial_flight=False,
    )


def _cascade_confirmed_abstention(event: dict) -> bool:
    """A row the decoder abstained and a later confirm supported."""
    return event.get("note") == _CASCADE_CONFIRM_NOTE


def _span_uses_confirmed_abstention(
    source: dict, contact: dict, boundary: tuple[float, str]
) -> bool:
    """A span close whose interior bounce is only a confirmed abstention.

    The held camera, the cut, the point end and the dead-ball mark still trim
    a flight the decoder already accepted. They do not open one on the contact
    before a bounce the decoder withheld.
    """
    start = float(contact["frame"])
    frame = float(boundary[0])
    for event in source["events"]:
        if event.get("event_type") != "bounce" or not _cascade_confirmed_abstention(event):
            continue
        if not role.supported(event):
            continue
        bounce = float(event["frame"])
        if start < bounce < frame:
            return True
    return False


def _span_terminal(source: dict, contact: dict, boundary: tuple[float, str]) -> dict | None:
    """Close the flight at the boundary. The kind is the boundary, not a bounce."""
    start = float(contact["frame"])
    frame, kind = boundary
    end = None
    for row in source.get("owner_ball_labels") or ():
        if row.get("status") != "visible":
            continue
        sample = float(row["frame"])
        if start < sample < float(frame) and (end is None or sample > end):
            end = sample
    if end is None:
        end = float(frame) - 1.0
    if not math.isfinite(end) or end <= start:
        return None
    bounces = [
        float(event["frame"])
        for event in source["events"]
        if event["event_type"] == "bounce"
        and role.supported(event)
        and start < float(event["frame"]) <= end
    ]
    if len(bounces) > 1:
        return None
    return _terminal_receipt(
        kind,
        end,
        evidence="supported_span",
        bounce_frames=bounces,
        partial_flight=True,
        boundary_frame=float(frame),
    )


def _accept_terminal(candidate: dict | None, barriers, crossing, start: float) -> dict | None:
    if candidate is None:
        return None
    if len(candidate["bounce_frames"]) > 1 and candidate["kind"] in {
        "net_stop",
        *SPAN_ENDPOINT_KINDS,
    }:
        return None
    if _barrier_overlaps(barriers, crossing, start, float(candidate["frame"])):
        return None
    return candidate


def _ambiguous_ground_barriers(
    source: dict, barriers: list[dict], crossing: list[int], left: dict, right: dict
) -> list[int]:
    """Barriers that are exactly one ambiguous labelled bounce inside this flight.

    Returns the bounce's ``unresolved_occurrence`` barrier and every
    ``unresolved_source_boundary`` with the same interval, or nothing when any
    other barrier crosses the span or the span holds more than one such bounce.
    """
    occurrences = []
    for i in crossing:
        barrier = barriers[i]
        if barrier["kind"] != "unresolved_occurrence":
            continue
        event = source["events"][barrier["original_event_index"]]
        if (
            event["event_type"] == "bounce"
            and event.get("status") == "ambiguous"
            and float(left["frame"]) < float(event["frame"]) < float(right["frame"])
        ):
            occurrences.append(i)
    if len(occurrences) != 1:
        return []
    interval = barriers[occurrences[0]]["frame_interval"]
    duplicates = [
        i
        for i in crossing
        if barriers[i]["kind"] == "unresolved_source_boundary"
        and barriers[i]["frame_interval"] == interval
    ]
    excused = sorted(occurrences + duplicates)
    if set(crossing) - set(excused):
        return []
    return excused


def _barrier_overlaps(barriers: list[dict], crossing: list[int], start: float, end: float) -> bool:
    return any(
        barriers[index]["frame_interval"][0] <= end
        and barriers[index]["frame_interval"][1] >= start
        for index in crossing
    )


def split_component(
    plan: dict,
    parent_index: int,
    local_flight: int,
    *,
    source: dict | None = None,
    labels: dict | None = None,
) -> dict:
    """One original flight of a joined component, still bound to that parent.

    The frozen maximal plan is not rewritten. The child keeps the parent's
    component index and maps back through ``original_flight_indices``. Later
    flights re-derive their origin role from the original source; the first
    child inherits the parent's role.
    """
    if plan.get("schema") != SCHEMA or plan.get("mode") != MODE:
        raise ValueError("explicit supported contact component plan required")
    if type(parent_index) is not int or not 0 <= parent_index < len(plan["components"]):
        raise ValueError("split parent component index required")
    parent = plan["components"][parent_index]
    flights = list(parent["original_flight_indices"])
    if len(flights) < 2:
        raise ValueError("split fallback applies only to a joined multi-flight component")
    if type(local_flight) is not int or not 0 <= local_flight < len(flights):
        raise ValueError("split local flight index required")
    original = flights[local_flight]
    contact_indices = list(parent["original_contact_indices"])
    left = contact_indices[local_flight]
    right = contact_indices[local_flight + 1]
    original_events = plan["original_events"]
    contacts = [event for event in original_events if event["event_type"] == "contact"]
    start = float(contacts[left]["frame"])
    end = float(contacts[right]["frame"])
    event_indices = [
        index
        for index, event in enumerate(original_events)
        if start <= float(event["frame"]) <= end
    ]
    slot = plan["original_slots"][original]
    if local_flight == 0:
        first_role = parent["first_contact_role"]
        evidence = deepcopy(parent["first_contact_role_evidence"])
    else:
        if source is None or labels is None:
            raise ValueError("later split flight requires the original source and labels")
        first_role, evidence = _origin_role(
            source,
            contacts[left],
            left,
            labels,
            plan.get(ROUTING_FIELD, ROUTING_DEFAULT),
        )
    component = dict(
        component_index=parent_index,
        original_flight_indices=[original],
        original_contact_indices=[left, right],
        first_contact_role=first_role,
        first_contact_role_evidence=evidence,
        left_contact_supported_native_frames=list(slot["origin_bracket_supported_native_frames"]),
        right_contact_supported_native_frames=list(
            slot["next_contact_bracket_supported_native_frames"]
        ),
        right_boundary_kind="original_contact",
        right_boundary_membership="half_open_original_interior",
        complete_original_source=False,
        physical_ending=None,
        continuity_with_other_components=False,
        start_frame=start,
        end_frame=end,
        original_event_indices=event_indices,
        events=[deepcopy(original_events[index]) for index in event_indices],
        local_to_original=dict(flights=[original], contacts=[left, right], events=event_indices),
        split_parent_component_index=parent_index,
        split_local_flight_index=local_flight,
    )
    return component | {"component_sha256": record_hash(component)}


def prepare(
    packet: dict,
    labels: dict,
    cameras: dict,
    *,
    mode: str = DEFAULT_MODE,
    observation_partition: str = "fifth_frame_withheld",
    observation_fallback: bool = True,
    leading_event_components: str = LEADING_DEFAULT,
    contact_component_routing: str = ROUTING_DEFAULT,
    terminal_track_endpoint: str = TERMINAL_TRACK_OFF,
    supported_flight_endings: str = SUPPORTED_ENDINGS_OFF,
    ending_ownership: str = ENDING_OWNERSHIP_OFF,
    out_bounce_ending: str = OUT_BOUNCE_ENDING_OFF,
    boundary_endings: str = BOUNDARY_ENDINGS_OFF,
    ambiguous_interior_ground: str = AMBIGUOUS_GROUND_OFF,
    net_stop_tail: str = NET_STOP_TAIL_OFF,
) -> dict:
    """Freeze maximal eligible adjacent spans and every omitted original slot.

    Four training/one check exposures are the existing coverage requirements.
    Endpoint bracket availability is evidence, never fabricated or required in
    addition to ordinary interior-contact evidence. Uncertain occurrences and
    declared cuts break adjacency; no later contact can replace an unknown end.
    ``terminal_track_endpoint="ball_track"`` is the exception: a flight whose own
    bounce, image-border exit, or wall shows the ending is searchable without a
    next contact, and the component stops at that terminal.
    ``ball_track_last_sample`` keeps that rule and, only when this slot is the
    point's last original flight and none of those four endings is present,
    stops at the last visible ball sample in the slot. An interior flight stays
    held. Default ``off`` keeps the unknown-next hold.
    ``supported_flight_endings="on"`` adds a net stop and, separately, closes a
    flight at the first camera cut, held camera, point end or dead-ball boundary.
    The boundary kind is kept on the receipt and the flight is marked partial.
    No bounce is invented.
    ``ending_ownership="on"`` keeps a labelled aftermath row as a dead-ball
    boundary, drops an automatic bounce more than 1.5 s after the previous live
    bounce, and stops a last-flight tail at the first picture exit or track gap.
    A labelled second bounce is not dropped. A net stop is not moved. Default
    ``off`` leaves published plans unchanged. Counts here describe supplied
    runtime origins, not a reference denominator.
    ``out_bounce_ending="on"`` ends live play at the first bounce read outside
    the singles court by more than ``OUT_BOUNCE_MARGIN_M``: a contact after it
    opens no flight. The flight that owns the out bounce is unchanged.
    ``boundary_endings="on"`` closes a still-held flight before its first
    labelled boundary row (or at a net stop that row names), and keeps a few
    samples after a dead-ball bounce. Default ``off`` leaves plans unchanged.
    ``ambiguous_interior_ground="supplied"`` releases an interior flight held only
    by one labelled ambiguous bounce inside it; the bounce is fitted as supplied
    and the flight is its own component (see ``AMBIGUOUS_GROUND_SUPPLIED``).
    """
    if mode not in MODES:
        raise ValueError("explicit supported component preparation mode required")
    if leading_event_components not in LEADING_MODES:
        raise ValueError("explicit supported leading-evidence preparation mode required")
    if contact_component_routing not in ROUTING_MODES:
        raise ValueError("explicit supported component source routing mode required")
    if terminal_track_endpoint not in TERMINAL_TRACK_MODES:
        raise ValueError("explicit ball-track terminal endpoint mode required")
    if supported_flight_endings not in SUPPORTED_ENDINGS_MODES:
        raise ValueError("explicit supported flight-endings mode required")
    if ending_ownership not in ENDING_OWNERSHIP_MODES:
        raise ValueError("explicit ending-ownership mode required")
    if out_bounce_ending not in OUT_BOUNCE_ENDING_MODES:
        raise ValueError("explicit out-bounce ending mode required")
    if boundary_endings not in BOUNDARY_ENDINGS_MODES:
        raise ValueError("explicit boundary-endings mode required")
    if ambiguous_interior_ground not in AMBIGUOUS_GROUND_MODES:
        raise ValueError("explicit ambiguous interior ground mode required")
    if net_stop_tail not in NET_STOP_TAIL_MODES:
        raise ValueError("explicit net-stop tail mode required")
    if mode == DEFAULT_MODE:
        return dict(schema=SCHEMA, mode=mode, status="policy_disabled")
    if type(observation_fallback) is not bool:
        raise ValueError("explicit camera fallback boolean required")
    source = _source(packet, labels, cameras, observation_partition, leading_event_components)
    leading_prefix = role.leading_physical_prefix(source["events"])
    inventory = coverage.inventory(
        source,
        cameras,
        observation_partition=observation_partition,
        observation_fallback=observation_fallback,
    )
    _, visible = coverage.visible_inputs(source, cameras, observation_fallback=observation_fallback)
    contacts = [e for e in source["events"] if e["event_type"] == "contact"]
    if len(inventory) != len(contacts) or any(
        row["original_flight_index"] != i or row["start_frame"] != contacts[i]["frame"]
        for i, row in enumerate(inventory)
    ):
        raise ValueError("coverage inventory must preserve every original contact index")
    supported = [role.supported(e) for e in contacts]
    out_call = (
        out_bounce_call(source["events"]) if out_bounce_ending == OUT_BOUNCE_ENDING_ON else None
    )
    dead = [out_call is not None and float(e["frame"]) > float(out_call["frame"]) for e in contacts]
    barriers = _barriers(source, labels)
    endpoint_frames = [
        sorted(f for f in visible if e["frame_interval"][0] <= f <= e["frame_interval"][1])
        for e in contacts
    ]
    slots, components = [], []
    active = None
    for index, row in enumerate(inventory):
        end = row["end_frame"]
        # The full supplied contact brackets bound possible adjacency, while
        # observation row ownership remains the original half-open frame domain.
        low = contacts[index]["frame_interval"][0]
        high = contacts[index + 1]["frame_interval"][1] if index + 1 < len(contacts) else end
        crossing = [
            i
            for i, b in enumerate(barriers)
            if b["frame_interval"][0] <= high and b["frame_interval"][1] >= low
        ]
        right = index + 1 if index + 1 < len(contacts) else None
        released = []
        if (
            ambiguous_interior_ground == AMBIGUOUS_GROUND_SUPPLIED
            and crossing
            and right is not None
            and supported[index]
            and supported[right]
        ):
            released = _ambiguous_ground_barriers(
                source, barriers, crossing, contacts[index], contacts[right]
            )
            crossing = [i for i in crossing if i not in released]
        endpoint_barriers = [
            i
            for i in crossing
            if barriers[i]["kind"] != "unresolved_occurrence"
            or source["events"][barriers[i]["original_event_index"]]["event_type"] == "contact"
        ]
        known_right = (
            right if right is not None and supported[right] and not endpoint_barriers else None
        )
        # The dead-ball contact still closes the flight that owns the out bounce:
        # the ball did travel to that racket. Only its own flight is refused.
        span_limit = None
        if supported_flight_endings == SUPPORTED_ENDINGS_ON:
            horizon = (
                float(contacts[right]["frame"])
                if right is not None
                else float(source["owner_end_frame"]) + 1.0
            )
            span_limit = _first_boundary(
                source, labels, cameras, float(contacts[index]["frame"]), horizon
            )
            # A known next contact past the boundary is not this flight's end.
            # Closing on it would fit the cut, the close-up or the dead ball.
            if (
                span_limit is not None
                and known_right is not None
                and span_limit[0] < float(contacts[known_right]["frame"])
            ):
                known_right = None
        reasons = []
        if not supported[index]:
            reasons.append("unresolved_origin_occurrence")
        if dead[index]:
            reasons.append("post_out_bounce_origin")
        # A disposal contact is the dead-ball phase, not the next flight.
        if (
            supported_flight_endings == SUPPORTED_ENDINGS_ON
            and _boundary_kind(contacts[index]) == "dead_ball"
        ):
            reasons.append("dead_ball_phase_origin")
        if known_right is None:
            reasons.append("unknown_next_contact_endpoint")
        if crossing:
            reasons.append("source_boundary_barrier")
        if not row["coverage_qualified"]:
            reasons.append("unsupported_original_flight_coverage")
        if len(row["source_bounce_frames"]) > 1:
            reasons.append("unsupported_multiple_ground_events")
        # Last flight only. An interior flight still ends on its next contact.
        ownership_limit = None
        live_bounces: list[dict] = []
        if ending_ownership == ENDING_OWNERSHIP_ON and right is None:
            ownership_limit, live_bounces = _ownership_split(
                source,
                labels,
                float(contacts[index]["frame"]),
                float(source["owner_end_frame"]) + 1.0,
            )
        terminal = None
        if (
            terminal_track_endpoint in (TERMINAL_TRACK_BALL, TERMINAL_TRACK_LAST_SAMPLE)
            and "unknown_next_contact_endpoint" in reasons
            and not any(reason not in _TERMINAL_CLEARABLE_REASONS for reason in reasons)
        ):
            next_frame = float(contacts[right]["frame"]) if right is not None else None
            limit = span_limit[0] if span_limit is not None else None
            track_limit = limit
            if ownership_limit is not None:
                track_limit = (
                    ownership_limit if track_limit is None else min(track_limit, ownership_limit)
                )
            candidate = _ball_track_terminal(source, contacts[index], next_frame, track_limit)
            # Last original flight only. A later contact, even an unsupported
            # one, means this slot is interior and the last-sample ending is
            # refused. The four ball-track endings above still apply there.
            if (
                candidate is None
                and terminal_track_endpoint == TERMINAL_TRACK_LAST_SAMPLE
                and right is None
            ):
                sample = _last_visible_ball_frame(
                    source,
                    float(contacts[index]["frame"]),
                    float(source["owner_end_frame"]),
                    limit,
                )
                # A sample past the post-play row is the dead ball, not this flight.
                if sample is not None and ownership_limit is not None and sample >= ownership_limit:
                    sample = None
                if sample is not None:
                    candidate = _terminal_receipt(
                        "last_visible_sample",
                        sample,
                        evidence="last_visible_ball_sample",
                        bounce_frames=[
                            float(event["frame"])
                            for event in source["events"]
                            if event["event_type"] == "bounce"
                            and role.supported(event)
                            and float(contacts[index]["frame"]) < float(event["frame"]) <= sample
                            and (ownership_limit is None or float(event["frame"]) < ownership_limit)
                        ],
                    )
            # A sample that exists only because the boundary cut the tail is not
            # an ending of its own. The boundary kind is the endpoint.
            if (
                candidate is not None
                and candidate.get("kind") == "last_visible_sample"
                and span_limit is not None
            ):
                candidate = None
            if (
                candidate is None
                and ending_ownership == ENDING_OWNERSHIP_ON
                and ownership_limit is not None
            ):
                candidate = _ownership_candidate(
                    source,
                    float(contacts[index]["frame"]),
                    live_bounces,
                    ownership_limit,
                    float(source["owner_end_frame"]) + 1.0,
                    boundary_endings,
                    _first_barrier_start(barriers, crossing, float(contacts[index]["frame"])),
                )
            if candidate is not None and not _barrier_overlaps(
                barriers,
                crossing,
                float(contacts[index]["frame"]),
                float(candidate["frame"]),
            ):
                terminal = candidate
                reasons = [
                    reason for reason in reasons if reason not in _TERMINAL_CLEARABLE_REASONS
                ]
        if (
            terminal is None
            and supported_flight_endings == SUPPORTED_ENDINGS_ON
            and "unknown_next_contact_endpoint" in reasons
            and not any(reason not in _SUPPORTED_CLEARABLE_REASONS for reason in reasons)
        ):
            next_frame = float(contacts[right]["frame"]) if right is not None else None
            limit = span_limit[0] if span_limit is not None else None
            candidate = _accept_terminal(
                _net_stop_terminal(source, contacts[index], next_frame, limit, net_stop_tail),
                barriers,
                crossing,
                float(contacts[index]["frame"]),
            )
            if (
                candidate is None
                and span_limit is not None
                and not _span_uses_confirmed_abstention(source, contacts[index], span_limit)
            ):
                candidate = _accept_terminal(
                    _span_terminal(source, contacts[index], span_limit),
                    barriers,
                    crossing,
                    float(contacts[index]["frame"]),
                )
            if candidate is not None:
                terminal = candidate
                reasons = [
                    reason for reason in reasons if reason not in _SUPPORTED_CLEARABLE_REASONS
                ]
        if (
            terminal is None
            and boundary_endings == BOUNDARY_ENDINGS_ON
            and "unknown_next_contact_endpoint" in reasons
            and "source_boundary_barrier" in reasons
            and not any(reason not in _SUPPORTED_CLEARABLE_REASONS for reason in reasons)
        ):
            candidate = _labelled_boundary_terminal(
                source,
                contacts[index],
                barriers,
                crossing,
                float(contacts[right]["frame"]) if right is not None else None,
                span_limit[0] if span_limit is not None else None,
                net_stop_tail,
            )
            if candidate is not None:
                terminal = candidate
                reasons = [
                    reason for reason in reasons if reason not in _SUPPORTED_CLEARABLE_REASONS
                ]
        if (
            supported_flight_endings == SUPPORTED_ENDINGS_ON
            and terminal is not None
            and terminal.get("kind") != "net_stop"
        ):
            # Ownership can close on the bounce or the gap after the tape.
            # That row is the dead ball. The net stop, when it is earlier,
            # stays the ending.
            next_frame = float(contacts[right]["frame"]) if right is not None else None
            limit = span_limit[0] if span_limit is not None else None
            net = _accept_terminal(
                _net_stop_terminal(source, contacts[index], next_frame, limit, net_stop_tail),
                barriers,
                crossing,
                float(contacts[index]["frame"]),
            )
            if net is not None and float(net["frame"]) < float(terminal["frame"]) - 1e-6:
                terminal = net
        if ending_ownership == ENDING_OWNERSHIP_ON and terminal is not None:
            terminal = _pull_tail(
                terminal,
                source,
                float(contacts[index]["frame"]),
                float(source["owner_end_frame"]) + 1.0,
            )
        slot = dict(
            **deepcopy(row),
            original_contact_index=index,
            origin_occurrence_supported=supported[index],
            supplied_next_contact_index=right,
            known_next_contact_index=known_right,
            endpoint_semantics="confirmed_original_contact"
            if known_right is not None
            else "unknown",
            origin_bracket_supported_native_frames=endpoint_frames[index],
            next_contact_bracket_supported_native_frames=(
                endpoint_frames[right] if right is not None else []
            ),
            barrier_indices=crossing,
            reasons=reasons,
            component_index=None,
            local_flight_index=None,
            **(
                {AMBIGUOUS_GROUND_FIELD: {"released_barrier_indices": released}} if released else {}
            ),
        )
        # A terminal flight is its own component. Extending the previous
        # contact-to-contact component would refit flights that already close
        # on a known contact.
        if terminal is not None or released:
            active = None
        if reasons:
            active = None
        elif active is None:
            try:
                first_role, evidence = _origin_role(
                    source, contacts[index], index, labels, contact_component_routing
                )
            except ValueError as error:
                reasons.append("component_origin_role_unavailable")
                slot["role_blocker"] = str(error)
            else:
                active = dict(
                    component_index=len(components),
                    original_flight_indices=[],
                    original_contact_indices=[index],
                    first_contact_role=first_role,
                    first_contact_role_evidence=evidence,
                    left_contact_supported_native_frames=endpoint_frames[index],
                    right_boundary_kind="supplied_end"
                    if terminal is not None
                    else "original_contact",
                    right_boundary_membership="half_open_original_interior",
                    complete_original_source=False,
                    physical_ending=None,
                    continuity_with_other_components=False,
                    **({AMBIGUOUS_GROUND_FIELD: AMBIGUOUS_GROUND_SUPPLIED} if released else {}),
                    **(
                        {
                            "endpoint_semantics": terminal["kind"],
                            TERMINAL_TRACK_FIELD: deepcopy(terminal),
                        }
                        if terminal is not None
                        else {}
                    ),
                )
                components.append(active)
        if not reasons:
            slot.update(
                component_index=active["component_index"],
                local_flight_index=len(active["original_flight_indices"]),
            )
            active["original_flight_indices"].append(index)
            if terminal is not None:
                slot["endpoint_semantics"] = terminal["kind"]
                slot[TERMINAL_TRACK_FIELD] = deepcopy(terminal)
                active["right_contact_supported_native_frames"] = []
                active = None
            else:
                active["original_contact_indices"].append(right)
                active["right_contact_supported_native_frames"] = endpoint_frames[right]
                if released:
                    active = None
        slot["status"] = "prepared" if not reasons else "unresolved"
        slots.append(slot)
    for component in components:
        indices = component["original_contact_indices"]
        left = contacts[indices[0]]["frame"]
        if component.get(TERMINAL_TRACK_FIELD):
            right = float(component[TERMINAL_TRACK_FIELD]["frame"])
        else:
            right = contacts[indices[-1]]["frame"]
        component.update(
            start_frame=left,
            end_frame=right,
            original_event_indices=[
                i for i, e in enumerate(source["events"]) if left <= e["frame"] <= right
            ],
        )
        component["events"] = [
            deepcopy(source["events"][i]) for i in component["original_event_indices"]
        ]
        component["local_to_original"] = dict(
            flights=list(component["original_flight_indices"]),
            contacts=list(indices),
            events=list(component["original_event_indices"]),
        )
        component["component_sha256"] = record_hash(component)
    plan = dict(
        schema=SCHEMA,
        mode=mode,
        status="prepared" if components else "held",
        observation_partition=observation_partition,
        observation_fallback=observation_fallback,
        # Emitted only where leading evidence exists, so an ordinary plan and
        # every already published component output stay byte-identical.
        **(
            dict(
                leading_event_components=leading_event_components,
                leading_physical_prefix=deepcopy(leading_prefix),
                leading_physical_prefix_count=len(leading_prefix),
                leading_prefix_semantics=(
                    "original evidence before the first observed contact; "
                    "retained as input, never modelled and never removed"
                ),
            )
            if leading_prefix
            else {}
        ),
        # Emitted only under the explicit routing policy, so an ordinary plan
        # and every already published component output stay byte-identical.
        **(
            dict(
                contact_component_routing=contact_component_routing,
                unsupported_original_spans=_unsupported_spans(source),
            )
            if contact_component_routing != ROUTING_DEFAULT
            else {}
        ),
        # Emitted only when a run asks. Default off keeps published plans identical.
        **(
            {TERMINAL_TRACK_FIELD: terminal_track_endpoint}
            if terminal_track_endpoint != TERMINAL_TRACK_OFF
            else {}
        ),
        **(
            {SUPPORTED_ENDINGS_FIELD: supported_flight_endings}
            if supported_flight_endings != SUPPORTED_ENDINGS_OFF
            else {}
        ),
        **(
            {ENDING_OWNERSHIP_FIELD: ending_ownership}
            if ending_ownership != ENDING_OWNERSHIP_OFF
            else {}
        ),
        **(
            {OUT_BOUNCE_ENDING_FIELD: out_bounce_ending, "out_bounce_call": deepcopy(out_call)}
            if out_bounce_ending != OUT_BOUNCE_ENDING_OFF
            else {}
        ),
        **(
            {BOUNDARY_ENDINGS_FIELD: boundary_endings}
            if boundary_endings != BOUNDARY_ENDINGS_OFF
            else {}
        ),
        **(
            {AMBIGUOUS_GROUND_FIELD: ambiguous_interior_ground}
            if ambiguous_interior_ground != AMBIGUOUS_GROUND_OFF
            else {}
        ),
        **({NET_STOP_TAIL_FIELD: net_stop_tail} if net_stop_tail != NET_STOP_TAIL_OFF else {}),
        source_bindings={
            k: record_hash(v)
            for k, v in dict(packet=packet, labels=labels, cameras=cameras).items()
        },
        match_id=source["match_id"],
        clip=source["point_clip"],
        attempt_id=source["attempt_id"],
        original_native_window=list(source["observation_scope"]["native_window"]),
        original_owner_end_frame=source["owner_end_frame"],
        original_events=deepcopy(source["events"]),
        original_events_sha256=record_hash(source["events"]),
        later_source_context_events=[
            deepcopy(e)
            for e in labels["events"]["records"]
            if e.get("clip", source["point_clip"]) == source["point_clip"]
            and e.get("event_type") in prefix.PHYSICAL
            and source["events"][-1]["frame"] < e["frame"] <= source["owner_end_frame"]
        ],
        original_slot_count=len(slots),
        slot_count_semantics="all_original_contact_origins_including_uncertain_occurrence",
        occurrence_supported_origin_indices=[i for i, yes in enumerate(supported) if yes],
        original_slots=slots,
        unresolved_original_slot_indices=[
            i for i, s in enumerate(slots) if s["status"] == "unresolved"
        ],
        barriers=barriers,
        components=components,
        complete_original_source=False,
        physical_ending=None,
        fitted_states_consulted=False,
        reference_rows_consulted=False,
        reference_denominator_policy="preserve_every_original_reference_row_by_original_origin_identity",
        runtime_orchestration=False,
    )
    return plan | {"plan_sha256": record_hash(plan)}


def validate(
    plan: dict,
    packet: dict,
    labels: dict,
    cameras: dict,
    *,
    mode: str = MODE,
    observation_partition: str = "fifth_frame_withheld",
    observation_fallback: bool = True,
    leading_event_components: str | None = None,
    contact_component_routing: str | None = None,
    terminal_track_endpoint: str | None = None,
    supported_flight_endings: str | None = None,
    ending_ownership: str | None = None,
    out_bounce_ending: str | None = None,
    boundary_endings: str | None = None,
    ambiguous_interior_ground: str | None = None,
    net_stop_tail: str | None = None,
) -> dict:
    """Recompute the entire input-only decision, including maximality and gaps.

    The leading-evidence and routing modes are read from the plan when the caller
    does not state one; the plan hash covers both, so a relaxed plan cannot be
    smuggled in.
    """
    if plan.get("schema") != SCHEMA or plan.get("mode") != MODE:
        raise ValueError("explicit supported contact component plan required")
    if leading_event_components is None:
        leading_event_components = plan.get("leading_event_components", LEADING_DEFAULT)
    if contact_component_routing is None:
        contact_component_routing = plan.get(ROUTING_FIELD, ROUTING_DEFAULT)
    if terminal_track_endpoint is None:
        terminal_track_endpoint = plan.get(TERMINAL_TRACK_FIELD, TERMINAL_TRACK_OFF)
    if supported_flight_endings is None:
        supported_flight_endings = plan.get(SUPPORTED_ENDINGS_FIELD, SUPPORTED_ENDINGS_OFF)
    if ending_ownership is None:
        ending_ownership = plan.get(ENDING_OWNERSHIP_FIELD, ENDING_OWNERSHIP_OFF)
    if out_bounce_ending is None:
        out_bounce_ending = plan.get(OUT_BOUNCE_ENDING_FIELD, OUT_BOUNCE_ENDING_OFF)
    if boundary_endings is None:
        boundary_endings = plan.get(BOUNDARY_ENDINGS_FIELD, BOUNDARY_ENDINGS_OFF)
    if ambiguous_interior_ground is None:
        ambiguous_interior_ground = plan.get(AMBIGUOUS_GROUND_FIELD, AMBIGUOUS_GROUND_OFF)
    if net_stop_tail is None:
        net_stop_tail = plan.get(NET_STOP_TAIL_FIELD, NET_STOP_TAIL_OFF)
    if (plan["mode"], plan["observation_partition"], plan["observation_fallback"]) != (
        mode,
        observation_partition,
        observation_fallback,
    ):
        raise ValueError("contact component plan differs from requested preparation policy")
    expected = prepare(
        packet,
        labels,
        cameras,
        mode=mode,
        observation_partition=observation_partition,
        observation_fallback=observation_fallback,
        leading_event_components=leading_event_components,
        contact_component_routing=contact_component_routing,
        terminal_track_endpoint=terminal_track_endpoint,
        supported_flight_endings=supported_flight_endings,
        ending_ownership=ending_ownership,
        out_bounce_ending=out_bounce_ending,
        boundary_endings=boundary_endings,
        ambiguous_interior_ground=ambiguous_interior_ground,
        net_stop_tail=net_stop_tail,
    )
    if plan != expected:
        raise ValueError("contact components differ from original source preparation")
    return plan
