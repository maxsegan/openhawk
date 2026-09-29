"""Default-off original-contact prefix scope, cut from source inputs alone.

Two explicit modes decide the cut, both before any candidate exists and both
from supplied events, ball rows and cameras only, so each is an input-derived
scope rather than a per-case repair choice:

``coverage``
    The cut is the first original flight that fails the unchanged four-training
    / one-check coverage after the existing camera qualification.

``terminal_identity``
    The cut is the final original contact flight, and only when the source's own
    observed tail after that contact is owned by more than one automatic
    identity, no supplied terminal event is authoritative about that tail, and
    at least one earlier original contact span is supported.  The trigger is
    recomputed from source packets here and in ``validate``; no fit, acceptance
    score, reason string or stage receipt is consulted.

The right boundary is an original contact, not a ground horizon and not a point
ending.  Nothing here invents an ending, a bounce, a root position, a serve
role or an observation row, and the original inventory, owner end frame, native
window and ending declaration are preserved verbatim in the receipt so replay
reconstructs the source exactly.
"""

from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = "s6_observed_contact_prefix_v1"
COVERAGE = "coverage"
TERMINAL_IDENTITY = "terminal_identity"
UNRESOLVED_ENDING = "unresolved_ending"
UNRESOLVED_INPUT_SCHEMA = "s6_unresolved_ending_input_v1"
#: A held or false opening (a bounce before any admitted contact, after the
#: serve contact was vetoed) must not refuse the rest of the rally. Default off.
#: ``on`` drops that prefix from the competitive inventory and keeps the first
#: contact onward. It does not retimed anything and it does not invent a contact.
PREFIX_ISOLATION_FIELD = "preparation_prefix_isolation"
PREFIX_ISOLATION_ON = "on"


def isolate_invalid_prefix(mode: str) -> bool:
    """Whether a physical prefix before the first contact is removed instead of refusing."""

    if mode in (None, "off", False):
        return False
    if mode in (PREFIX_ISOLATION_ON, True):
        return True
    raise ValueError("preparation_prefix_isolation must be off or on")
MODES = ("off", COVERAGE, TERMINAL_IDENTITY, UNRESOLVED_ENDING)
DEFAULT_MODE = "off"
RIGHT_BOUNDARY_KIND = "original_contact"
RIGHT_BOUNDARY_MEMBERSHIP = "half_open_original_interior"
RIGHT_BOUNDARY_SEMANTICS = "original_contact_right_boundary_not_physical_ending"
UNRESOLVED_REASON = "coverage_cut"
#: Per-mode unresolved-slot reason.  ``coverage`` keeps its original string.
UNRESOLVED_REASONS = {
    COVERAGE: UNRESOLVED_REASON,
    TERMINAL_IDENTITY: "terminal_identity_cut",
    UNRESOLVED_ENDING: "unresolved_ending_cut",
}
IDENTITY_SCHEMA = "s6_terminal_identity_refusal_v1"
INTERIOR_SCOPE_SCHEMA = "s6_contact_prefix_interior_scope_v1"
#: Interior composition is scope-local and exists only under the new mode.
INTERIOR_COMPOSITION = "prefix_local"
PHYSICAL = {"contact", "bounce", "net_hit"}
#: Fixed, published, identical for every prefix scene.  Enforcement lives in the
#: stage/backend and ``prepare_attempt``; this list is the contract's copy.
NOT_APPLICABLE_FEATURES = (
    "terminal_completion",
    "terminal_rebound_seeds",
    "terminal_net_seeds",
    "terminal_incoming_training_axis",
    "inactive_ending_priors",
    "ground_settling",
    "optional_event_union",
    "optional_bounce_scope",
    "event_recovery",
    "first_flight_scope",
)
#: Tail grammars of the original scene; a prefix scene carries neither.
DROPPED_TAIL_KEYS = ("terminal_net_tail", "observed_horizon_tail")


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("explicit supported contact prefix mode required")
    return mode


def applicable(mode: str) -> bool:
    return validate_mode(mode) != "off"


def terminal_identity_refusal(attempt: dict, labels: dict) -> dict:
    """Recompute the observed-tail identity refusal from source packets alone.

    This is the whole trigger input for ``terminal_identity``.  ``refused`` is
    true only when the source itself shows a genuinely observed tail after the
    final supplied contact that more than one automatic identity owns, while no
    supplied terminal event is authoritative about that tail.  A short, absent,
    broken or ambiguous tail holds. Identity changes after a real occlusion can
    retain an earlier prefix, but never join the occluded tail. Reads the frozen source
    observation inventory: no fit, acceptance score, reason string or evaluation label.
    """
    from cv.experiments.connected_shooting import observed_horizon_tail as tail

    receipt = {
        "schema": IDENTITY_SCHEMA,
        "refused": False,
        "reason": None,
        "source": "automatic source observations and automatic physical events",
    }

    def held(reason: str) -> dict:
        return receipt | {"reason": reason}

    events = attempt.get("events")
    if not events or any(event.get("event_type") not in PHYSICAL for event in events):
        return held("original physical events required")
    contacts = [event for event in events if event["event_type"] == "contact"]
    if not contacts:
        return held("original contact required")
    last = contacts[-1]
    interval = [float(value) for value in last.get("frame_interval", [])]
    frame = float(last["frame"])
    if (
        len(interval) != 2
        or not all(math.isfinite(value) for value in [*interval, frame])
        or not interval[0] <= frame <= interval[1]
    ):
        return held("malformed original terminal contact interval")
    if last.get("occurrence_status") in tail.UNRESOLVED_STATUS:
        return held("unresolved originating contact cannot open an observed tail")
    if any(float(event["frame"]) > frame for event in events):
        return held("existing supplied terminal event path remains authoritative")
    clip = attempt["point_clip"]
    records = [row for row in labels.get("ball", {}).get("records", []) if row["clip"] == clip]
    if len(records) != 1:
        return held("one matching source observation window required")
    rows = records[0]["frames"]
    frames = [int(row["frame"]) for row in rows]
    if not frames or frames != sorted(frames) or len(frames) != len(set(frames)):
        return held("ordered unique source observation inventory required")
    window_end = float(frames[-1])
    visible = [
        row
        for row in rows
        if interval[1] < float(row["frame"]) <= window_end and row["status"] == "visible"
    ]
    receipt = receipt | {
        "last_contact_frame": frame,
        "last_contact_interval": interval,
        "supplied_terminal_event_count": 0,
        "native_window_end": window_end,
        "visible_tail_frames": [int(row["frame"]) for row in visible],
    }
    if len(visible) < tail.MINIMUM_VISIBLE_TAIL_ROWS:
        return held("observed horizon requires a supported visible tail after the contact")
    contiguous = best = 1
    for previous, row in zip(visible, visible[1:]):
        contiguous = contiguous + 1 if int(row["frame"]) == int(previous["frame"]) + 1 else 1
        best = max(best, contiguous)
    if best < tail.MINIMUM_VISIBLE_TAIL_ROWS:
        return held("observed horizon requires contiguous visible tail rows")
    # Same reading the tail contract uses: the producer's own identity, never a
    # bare ``track_id`` that silently accepts unrelated objects.
    identities = [row.get("automatic_track_id", row.get("track_id")) for row in visible]
    if any(value in (None, "") for value in identities):
        return held("incomplete automatic identity across the observed tail")
    distinct = sorted({str(value) for value in identities})
    receipt = receipt | {"tail_identities": distinct, "tail_identity_count": len(distinct)}
    if len(distinct) < 2:
        return held("one automatic track identity owns the observed tail")
    if any(
        not all(math.isfinite(float(row[key])) for key in ("x1080", "y1080")) for row in visible
    ):
        return held("finite native source observations required across the observed tail")
    for record in labels.get("events", {}).get("records", []):
        if record.get("clip", clip) != clip or record["event_type"] not in PHYSICAL:
            continue
        if interval[1] < float(record["frame"]) <= window_end and record.get("model_abstain") is (
            False
        ):
            return held("ambiguous source physical event inside the unresolved tail")
    return receipt | {
        "refused": True,
        "reason": "more than one automatic track identity owns the observed tail",
    }


def _native_window(attempt: dict, native_window, scope: dict | None) -> tuple[list[int], str]:
    """Original root window; never shrunk, never invented past observed rows."""
    if native_window is not None:
        window, origin = list(native_window), "passed"
    elif isinstance(scope, dict) and scope.get("native_window") is not None:
        window, origin = list(scope["native_window"]), "original_observation_scope"
    else:
        frames = [int(row["frame"]) for row in attempt["owner_ball_labels"]]
        if not frames:
            raise ValueError("original native window unavailable from labels or source scope")
        window, origin = [min(frames), max(frames)], "owner_ball_label_extent"
    if len(window) != 2 or any(type(f) is not int for f in window) or window[0] >= window[1]:
        raise ValueError("original ordered native window required")
    return window, origin


LEADING_PREFIX_SCHEMA = "s6_leading_physical_prefix_v1"


def leading_prefix_declaration(events: list[dict], refusal: str) -> dict:
    """Declare the retained leading rows and the closed unobserved-origin set.

    Shared by every producer adapter, so the supplied-clip swap arms and the
    automatic runtime cannot disagree about what was retained or why.
    """
    from cv.pipeline import s6_first_contact_role as role

    prefix_events = role.leading_physical_prefix(events)
    if not prefix_events:
        raise ValueError("declared leading evidence requires original preceding physical rows")
    first = next(e for e in events if e["event_type"] == "contact")
    return dict(
        schema=LEADING_PREFIX_SCHEMA,
        events=deepcopy(prefix_events),
        event_count=len(prefix_events),
        first_observed_contact=deepcopy(first),
        origin_status="unknown",
        role_candidates=["serve", "rally"],
        origin_hypotheses=list(role.UNKNOWN_ORIGIN_HYPOTHESES),
        modelled=False,
        fabricated_events=False,
        dropped_original_rows=0,
        native_epochs_modified=False,
        source_refusal=refusal,
    )


def unresolved_input_contract(
    events: list[dict],
    window,
    *,
    event_origin: str,
    boundaries,
    refusal: str,
    leading_prefix: dict | None = None,
    segmentation_source: dict | None = None,
) -> dict:
    """The shared unresolved-input inventory; every original row is retained."""
    return dict(
        schema=UNRESOLVED_INPUT_SCHEMA,
        ending_semantics="unresolved",
        physical_ending=None,
        native_window=list(window),
        observation_horizon=float(window[1]),
        horizon_semantics="native_inventory_limit_not_physical_ending",
        original_event_observations=deepcopy(events),
        original_inventory=dict(native_window=list(window), event_origin=event_origin),
        unresolved_boundary_intervals=deepcopy(list(boundaries)),
        preceding_scope_refusal=refusal,
        **({"leading_physical_prefix": leading_prefix} if leading_prefix is not None else {}),
        **(
            {"segmentation_source": deepcopy(segmentation_source)}
            if segmentation_source is not None
            else {}
        ),
    )


def retained_inventory(
    error: ValueError,
    events: list[dict],
    window,
    *,
    event_origin: str,
    boundaries,
    mode: str,
    leading_admitted: bool = False,
    segmentation_source: dict | None = None,
) -> dict:
    """Own one observation-scope refusal as the retained original inventory.

    Every preparation entrypoint routes its refusal here, so the supplied-input
    runner and the automatic runtime cannot disagree about which refusal a
    declared policy owns or what the retained inventory then contains.

    An origin refusal needs the declared leading-evidence admission and carries
    the leading-prefix declaration. A tail refusal needs the unresolved-ending
    prefix policy and declares no leading prefix: a stream that opens on a row
    other than its first contact is refused for its origin first, so a tail
    refusal has no preceding rows to declare. Anything no declared policy owns
    is raised unchanged.

    This decides ownership only. It is pure and source-only: no contact, ground
    event, ending, cut or native epoch is created, and no original row is
    dropped, reordered or retimed.
    """
    from cv.experiments.connected_shooting import observation_scope

    if isinstance(error, observation_scope.UnsupportedOriginTopology):
        if not leading_admitted:
            raise error
        leading_prefix = leading_prefix_declaration(events, str(error))
    elif isinstance(error, observation_scope.UnsupportedTailTopology):
        if mode != UNRESOLVED_ENDING:
            raise error
        leading_prefix = None
    else:
        raise error
    return unresolved_input_contract(
        events,
        window,
        event_origin=event_origin,
        boundaries=boundaries,
        refusal=str(error),
        leading_prefix=leading_prefix,
        segmentation_source=segmentation_source,
    )


def unresolved_boundaries(events: list[dict], *, boundaries=()) -> list[list[float]]:
    """Every declared unresolved bound: supplied ones first, then the refused rows.

    ONE ordering, shared by the supplied-input harness and the automatic runtime,
    because the boundary list is part of the retained inventory contract and a
    different order is a different contract over the same evidence.
    """
    return [
        *deepcopy(list(boundaries)),
        *[
            deepcopy(event["frame_interval"])
            for event in events
            if event.get("status") in {"ambiguous", "abstained", "unsupported"}
        ],
    ]


def qualify_or_retain(
    events: list[dict],
    window,
    *,
    event_origin: str | None,
    boundaries=(),
    mode: str,
    leading_admitted: bool = False,
    allow_terminal_net: bool = False,
    allow_observed_horizon: bool = False,
    segmentation_source: dict | None = None,
) -> tuple[dict, bool]:
    """Qualify an observation scope, or own the refusal as the retained inventory.

    ONE decision, reached by the supplied-input preparation harness and the automatic
    runtime through the same code, so the two cannot disagree about which refusal a
    declared policy owns or what the retained inventory then contains. Each caller
    still supplies its own truthful provenance -- the runtime names the automatic
    point ledger it segmented from, the harness names the declared event origin of a
    supplied clip -- because the two really do have different ancestry and a shared
    decision must not pretend otherwise.

    A declared unresolved boundary that reaches into the observed span forces the
    refusal even when the admitted rows would qualify on their own: a source that
    refuses one of its own occurrences is not describing an ordinary ground tail,
    and reading it as one would model straight through the refusal.

    Returns ``(contract, retained)``. Pure: no event, ending, cut or native epoch is
    created, and no original row is dropped, reordered or retimed.
    """
    from cv.experiments.connected_shooting import observation_scope

    checked = []
    for pair in boundaries:
        # Validated before conversion: a malformed bound must refuse as a bad
        # boundary, never as an unsupported topology the prefix policy would own.
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)
            or not window[0] <= pair[0] <= pair[1] <= window[1]
        ):
            raise ValueError("ordered native unresolved boundary interval required")
        checked.append(deepcopy(list(pair)))
    boundaries = checked
    try:
        contract = observation_scope.qualify(
            events,
            window,
            segmentation_source=segmentation_source,
            allow_terminal_net=allow_terminal_net,
            allow_observed_horizon=allow_observed_horizon,
            event_origin=event_origin,
        )
        if any(pair[1] >= events[0]["frame_interval"][0] for pair in boundaries):
            raise observation_scope.UnsupportedTailTopology(
                "declared unresolved source boundary interrupts the observation horizon"
            )
    except (
        observation_scope.UnsupportedOriginTopology,
        observation_scope.UnsupportedTailTopology,
    ) as error:
        return (
            retained_inventory(
                error,
                events,
                window,
                event_origin=event_origin if event_origin is not None else "automatic",
                boundaries=boundaries,
                mode=mode,
                leading_admitted=leading_admitted,
                segmentation_source=segmentation_source,
            ),
            True,
        )
    return contract, False


def leading_admission(
    leading_event_components: str, contact_prefix_scope: str, contact_components: str
) -> bool:
    """Default-off admission of original evidence before the first observed contact.

    It needs the unresolved-input prefix contract to hold the inventory and the
    component policy to model the supported contact-to-contact spans; without
    both, the original refusal stands rather than a half-owned attempt.
    """
    from cv.pipeline import s6_contact_components as components

    if leading_event_components == components.LEADING_DEFAULT:
        return False
    if leading_event_components != components.LEADING_PREFIX:
        raise ValueError("explicit supported leading-evidence policy required")
    if contact_prefix_scope != UNRESOLVED_ENDING or contact_components != components.MODE:
        raise ValueError(
            "leading physical evidence requires the unresolved-ending prefix and "
            "contact-component policies; it cannot run as a bare admission"
        )
    return True


def component_source_routing(
    contact_component_routing: str, contact_prefix_scope: str, contact_components: str
) -> bool:
    """Default-off admission of the component partition as the source contract.

    A refused prefix cut is not a refused source. This mode owns one contiguous
    run of one-ground-event flights starting at the first original contact, so a
    contradictory span anywhere inside that run refuses the whole attempt, while
    the component grammar partitions the same inventory and keeps every
    unsupported span explicit. Nothing here relaxes a prefix check, changes a
    cut or drops an original row; it decides only which contract owns the source.
    """
    from cv.pipeline import s6_contact_components as components

    if contact_component_routing == components.ROUTING_DEFAULT:
        return False
    if contact_component_routing != components.ROUTING_UNSUPPORTED_SPAN:
        raise ValueError("explicit supported component source routing policy required")
    if contact_prefix_scope != UNRESOLVED_ENDING or contact_components != components.MODE:
        raise ValueError(
            "component source routing requires the unresolved-ending prefix and "
            "contact-component policies; it cannot run as a bare admission"
        )
    return True


#: Default-off admission of an original source that declares a physical ending
#: and also refuses one of its own interior physical occurrences.
UNCERTAIN_DEFAULT = "off"
UNCERTAIN_RETAINED = "retained_inventory"
UNCERTAIN_MODES = (UNCERTAIN_DEFAULT, UNCERTAIN_RETAINED)
UNCERTAIN_FIELD = "uncertain_original_occurrence"


def uncertain_occurrence_admission(
    uncertain_original_occurrence: str, contact_prefix_scope: str, contact_components: str
) -> bool:
    """Default-off admission of a declared ending beside a refused occurrence.

    A source that refuses one of its own interior occurrences cannot be prepared
    as one resolved competitive topology: the refused row may be a real impact
    or nothing at all, so every span containing it is unrepresentable. It is the
    same shape as a contradictory span, and it needs the same two contracts --
    the unresolved-input inventory to hold every original row and the component
    policy to model the supported contact-to-contact spans around the refusal.

    Admission decides ownership only. The declared ending is neither asserted
    nor removed: it stays a bound original boundary, exactly as an unresolved
    endpoint declaration does, and the refused row stays in the inventory.
    """
    from cv.pipeline import s6_contact_components as components

    if uncertain_original_occurrence == UNCERTAIN_DEFAULT:
        return False
    if uncertain_original_occurrence != UNCERTAIN_RETAINED:
        raise ValueError("explicit supported uncertain-occurrence policy required")
    if contact_prefix_scope != UNRESOLVED_ENDING or contact_components != components.MODE:
        raise ValueError(
            "an uncertain original occurrence requires the unresolved-ending prefix and "
            "contact-component policies; it cannot run as a bare admission"
        )
    return True


#: Default-off admission of the automatic producer's own recorded abstentions as
#: uncertain occurrences of the physical events it declined to claim.
ABSTAINED_DEFAULT = "off"
ABSTAINED_DECLARED_ENDING = "declared_ending"
ABSTAINED_RETAINED = "retained_inventory"
ABSTAINED_MODES = (ABSTAINED_DEFAULT, ABSTAINED_DECLARED_ENDING, ABSTAINED_RETAINED)
ABSTAINED_FIELD = "automatic_abstained_occurrence"


def abstained_occurrence_admission(
    automatic_abstained_occurrence: str,
    uncertain_original_occurrence: str,
    contact_prefix_scope: str,
    contact_components: str,
) -> str | None:
    """Default-off admission of an automatic abstention as an uncertain occurrence.

    The automatic producer records both what it claims and what it declined to
    claim, but only the claims reach an inventory, and every converted row is
    stamped ``predicted``. So an automatic source can state a resolved ending
    beside an occurrence it actually refused, and the refusal is simply gone:
    the attempt is prepared as one resolved competitive topology that the source
    never asserted. That is the same shape as a refused *labeled* occurrence,
    and it needs the same owner -- which is why this is not a policy of its own
    but a scope on `uncertain_occurrence_admission`, and raises without it.

    ``declared_ending`` admits the abstention only where the automatic stream
    declares a resolved ending, which is the shape the labeled admission owns.
    ``retained_inventory`` admits it wherever the inventory retains it, so an
    unresolved inventory also carries its own refused rows as boundaries.
    Returns the admitted mode, or ``None``.
    """
    if automatic_abstained_occurrence == ABSTAINED_DEFAULT:
        return None
    if automatic_abstained_occurrence not in ABSTAINED_MODES:
        raise ValueError("explicit supported abstained-occurrence policy required")
    if not uncertain_occurrence_admission(
        uncertain_original_occurrence, contact_prefix_scope, contact_components
    ):
        raise ValueError(
            "a retained automatic abstention requires the uncertain-occurrence admission; "
            "it cannot run as a bare admission"
        )
    return automatic_abstained_occurrence


def unresolved_input(attempt: dict) -> bool:
    """A retained, still unqualified original inventory: no contract owns it yet."""
    return attempt.get("observation_scope", {}).get("schema") == UNRESOLVED_INPUT_SCHEMA


def validate_unresolved_input(attempt: dict, labels: dict | None = None) -> dict:
    """Structural replay of a retained unresolved-ending inventory.

    This is the pending source declaration, not a modelled scope: it asserts no
    ending, no topology and no owned flight, so a search may not consume it.
    Only the contact-prefix binding or the component partition may own it.
    """
    contract = attempt.get("observation_scope") or {}
    if contract.get("schema") != UNRESOLVED_INPUT_SCHEMA:
        raise ValueError("original unresolved-ending inventory required")
    inventory = contract.get("original_inventory") or {}
    expected = unresolved_input_contract(
        attempt["events"],
        contract.get("native_window", []),
        event_origin=inventory.get("event_origin"),
        boundaries=contract.get("unresolved_boundary_intervals", []),
        refusal=contract.get("preceding_scope_refusal"),
        leading_prefix=contract.get("leading_physical_prefix"),
        segmentation_source=contract.get("segmentation_source"),
    )
    if contract != expected:
        raise ValueError("unresolved inventory differs from its original physical events/window")
    _checked_events(attempt["events"], contract["native_window"])
    if attempt.get("ending_supplied") is not False or attempt.get("point_end") is not None:
        raise ValueError("observation horizon cannot assert a physical ending")
    if (
        attempt.get("owner_end_frame") != contract["observation_horizon"]
        or attempt.get("owner_end_frame_semantics") != "observation_horizon_not_physical_event"
    ):
        raise ValueError("legacy end-frame alias must explicitly denote observation horizon")
    if any(attempt.get(name) is not None for name in DROPPED_TAIL_KEYS):
        raise ValueError("a retained unresolved inventory cannot carry a terminal grammar")
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
        source = [
            {k: v for k, v in event.items() if k != "clip"}
            for event in labels["events"]["records"]
            if event.get("clip", attempt["point_clip"]) == attempt["point_clip"]
            and event.get("event_type") in PHYSICAL
        ]
        expected_source = [
            {k: v for k, v in event.items() if k != "clip"} for event in attempt["events"]
        ]
        if source != expected_source:
            raise ValueError("original physical event inventory changed across scope adapter")
        rows = labels["ball"]["records"][0]["frames"]
        if [row["frame"] for row in rows] != list(
            range(contract["native_window"][0], contract["native_window"][1] + 1)
        ):
            raise ValueError(
                "complete native inventory, including absent ball observations, required"
            )
    return contract


def _checked_events(events: list[dict], window: list[int]) -> None:
    if not events:
        raise ValueError("original physical events required")
    pairs = [(float(e.get("frame")), e.get("event_type")) for e in events]
    if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
        raise ValueError("unique ordered original physical events required")
    for event in events:
        interval = event.get("frame_interval", [])
        if (
            event.get("event_type") not in PHYSICAL
            or len(interval) != 2
            or not all(math.isfinite(float(f)) for f in [event["frame"], *interval])
            or not window[0] <= interval[0] <= float(event["frame"]) <= interval[1] <= window[1]
        ):
            raise ValueError("entire original event interval must lie in the native window")


def _ending_kind(scope: dict | None) -> str:
    """An already bound unresolved horizon stays unresolved; nothing is promoted."""
    if (
        isinstance(scope, dict)
        and scope.get("physical_ending") is None
        and scope.get("ending_semantics") == "unresolved"
    ):
        return "unresolved_horizon"
    return "original_physical_ending"


def _contract(
    attempt: dict,
    cameras_document: dict,
    rows: list[dict],
    cut: int,
    observation_partition: str,
    observation_fallback: bool,
    native_window,
    mode: str = COVERAGE,
    identity: dict | None = None,
) -> tuple[dict, list[dict]]:
    from cv.experiments.connected_shooting import source_flight_coverage as coverage
    from cv.experiments.connected_shooting.observation_scope import event_digest

    events, bounds = coverage.domain(attempt)
    scope = attempt.get("observation_scope")
    window, window_origin = _native_window(attempt, native_window, scope)
    _checked_events(events, window)
    _, labels = coverage.visible_inputs(
        attempt, cameras_document, observation_fallback=observation_fallback
    )
    for row in rows[:cut]:
        if len(row["source_bounce_frames"]) not in (0, 1):
            raise ValueError("retained prefix flight needs zero or one supplied ground event")
    right = [
        event
        for event in events
        if event["event_type"] == "contact" and float(event["frame"]) == bounds[cut]
    ]
    if len(right) != 1:
        raise ValueError("exactly one original contact must open the failing flight")
    right = right[0]
    interval = right["frame_interval"]
    support = sorted(frame for frame in labels if interval[0] <= frame <= interval[1])
    if not support:
        raise ValueError("right original contact lacks supported native ball/camera evidence")
    retained_flights = []
    for row in rows[:cut]:
        native, train, check = coverage.frame_partition(
            labels,
            row["start_frame"],
            row["end_frame"],
            inclusive_end=False,
            observation_partition=observation_partition,
        )
        retained = {
            **{key: row[key] for key in ("original_flight_index", "start_frame", "end_frame")},
            "inclusive_end": False,
            "native_frames": [int(frame) for frame in native],
            "train_frames": [int(frame) for frame in train],
            "check_frames": [int(frame) for frame in check],
            "native_count": int(len(native)),
            "train_count": int(len(train)),
            "check_count": int(len(check)),
            "source_bounce_frames": list(row["source_bounce_frames"]),
        }
        if [retained[key] for key in ("native_frames", "train_frames", "check_frames")] != [
            row[key] for key in ("native_frames", "train_frames", "check_frames")
        ]:
            raise ValueError("retained membership differs from original interior membership")
        retained_flights.append(retained)
    retained_events = [
        deepcopy(event) for event in events if float(event["frame"]) <= float(right["frame"])
    ]
    if sum(event == right for event in retained_events) != 1:
        raise ValueError("retained prefix must contain the right original contact exactly once")
    contract = dict(
        schema=SCHEMA,
        mode=mode,
        observation_partition=observation_partition,
        observation_fallback=bool(observation_fallback),
        original_events_sha256=event_digest(events),
        original_event_count=len(events),
        original_inventory=deepcopy(rows),
        original_ball_rows_sha256=event_digest(attempt["owner_ball_labels"]),
        original_ball_frames=[int(row["frame"]) for row in attempt["owner_ball_labels"]],
        original_owner_end_frame=float(attempt["owner_end_frame"]),
        original_owner_end_frame_semantics=attempt.get("owner_end_frame_semantics"),
        original_ending_kind=attempt.get("owner_ending_kind", _ending_kind(scope)),
        original_ending_frame=float(attempt["owner_end_frame"]),
        original_point_end=deepcopy(attempt.get("point_end")),
        original_observation_scope=deepcopy(scope),
        native_window=list(window),
        native_window_origin=window_origin,
        modeled_native_window=[window[0], float(right["frame"])],
        retained_flight_indices=list(range(cut)),
        retained_flights=retained_flights,
        retained_membership="original_interior_half_open_unchanged",
        retained_events_sha256=event_digest(retained_events),
        retained_event_count=len(retained_events),
        unresolved_original_slots=[
            {
                **{
                    key: row[key]
                    for key in (
                        "original_flight_index",
                        "start_frame",
                        "end_frame",
                        "inclusive_end",
                        "native_count",
                        "train_count",
                        "check_count",
                    )
                },
                "reason": UNRESOLVED_REASONS[mode],
            }
            for row in rows[cut:]
        ],
        unresolved_original_flight_count=len(rows) - cut,
        first_coverage_failure=cut if mode == COVERAGE else None,
        right_contact=deepcopy(right),
        right_boundary_kind=RIGHT_BOUNDARY_KIND,
        right_boundary_membership=RIGHT_BOUNDARY_MEMBERSHIP,
        right_contact_supported_native_frames=[int(frame) for frame in support],
        modeled_horizon=float(right["frame"]),
        observation_horizon=float(right["frame"]),
        ending_semantics="unresolved",
        complete_original_source=False,
        physical_ending=None,
        first_contact_role=deepcopy(attempt.get("first_contact_role")),
        first_contact_role_origin="supplied_unchanged",
        not_applicable_features=list(NOT_APPLICABLE_FEATURES),
        reference_flight_count="not opened or inferred",
        source="original supplied contacts, ball rows and cameras only",
    )
    if mode == TERMINAL_IDENTITY:
        # Additive, mode-local fields only: a ``coverage`` contract stays byte
        # identical to the one this module has always published.
        if identity is None or identity.get("refused") is not True:
            raise ValueError("terminal-identity prefix requires its recomputed source refusal")
        if cut != len(rows) - 1:
            raise ValueError("terminal-identity cut must be the final original contact flight")
        contract.update(
            terminal_identity_cut=cut,
            terminal_identity_refusal=deepcopy(identity),
            supplied_terminal_ground_count=0,
            interior_composition=INTERIOR_COMPOSITION,
            interior_composition_scope="strictly_inside_retained_prefix",
            terminal_families="off",
            original_slot_count=len(rows),
            original_slot_count_semantics="runtime original slots, not the reference flight count",
        )
    if mode == UNRESOLVED_ENDING:
        contract.update(unresolved_ending_cut=cut, original_slot_count=len(rows))
    return contract, retained_events


def _unresolved_cut(attempt: dict, rows: list[dict]) -> int:
    """Same source-only boundary derivation for qualification and replay checks."""
    from cv.experiments.connected_shooting import source_flight_coverage as coverage

    failure = coverage.first_coverage_failure(rows)
    cut = min(len(rows) - 1, failure if failure is not None else len(rows) - 1)
    blocked = [
        [float(v) for v in event["frame_interval"]]
        for event in attempt["events"]
        if event.get("status") in {"ambiguous", "abstained", "unsupported"}
    ]
    blocked.extend([frame, frame] for frame in attempt.get("unannotated_context_frames", []))
    blocked.extend(attempt.get("observation_scope", {}).get("unresolved_boundary_intervals", []))
    for low, high in blocked:
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError("ordered finite unresolved source interval required")
        for row in rows[:cut]:
            if low <= row["end_frame"] and high >= row["start_frame"]:
                cut = min(cut, row["original_flight_index"])
    return cut


def qualify(
    attempt: dict,
    cameras_document: dict,
    *,
    observation_partition: str = "fifth_frame_withheld",
    observation_fallback: bool = True,
    native_window=None,
    mode: str = COVERAGE,
    labels: dict | None = None,
) -> dict:
    """Status receipt: ``qualified``, ``not_applicable`` or an explicit hold.

    Under ``coverage``, ``not_applicable`` means every original flight is
    covered, so the prepared scene must be identical to today.  A first-flight
    failure still holds: there is no earlier complete span to retain.

    Under ``terminal_identity`` the cut is the final original contact flight and
    the trigger is the recomputed source identity refusal; a first-flight final
    contact, an unsupported earlier span or any absent refusal holds.
    """
    from cv.experiments.connected_shooting import observation_partition as partition
    from cv.experiments.connected_shooting import source_flight_coverage as coverage

    validate_mode(mode)
    if mode == "off":
        raise ValueError("explicit applicable contact prefix mode required")
    partition.validate(observation_partition)
    receipt = dict(
        schema=SCHEMA,
        mode=mode,
        status="held",
        observation_partition=observation_partition,
        observation_fallback=bool(observation_fallback),
    )
    identity = None
    if mode == UNRESOLVED_ENDING:
        declaration = attempt.get("observation_scope", {})
        if declaration.get("schema") != UNRESOLVED_INPUT_SCHEMA:
            return receipt | {
                "status": "not_applicable",
                "reason": "no supplied unresolved ending inventory",
            }
        if (
            declaration.get("ending_semantics") != "unresolved"
            or declaration.get("physical_ending") is not None
        ):
            return receipt | {"reason": "unresolved ending source cannot assert a physical ending"}
    if mode == TERMINAL_IDENTITY:
        if labels is None:
            raise ValueError("terminal-identity qualification requires the source observations")
        try:
            identity = terminal_identity_refusal(attempt, labels)
        except (KeyError, TypeError, ValueError) as error:
            return receipt | {"reason": str(error)}
        receipt["terminal_identity_refusal"] = identity
        if not identity["refused"]:
            return receipt | {
                "status": "not_applicable",
                "reason": identity["reason"],
            }
    try:
        rows = coverage.inventory(
            attempt,
            cameras_document,
            observation_partition=observation_partition,
            observation_fallback=observation_fallback,
        )
    except (KeyError, TypeError, ValueError) as error:
        return receipt | {"reason": str(error)}
    receipt["inventory"] = rows
    failure = coverage.first_coverage_failure(rows)
    receipt["first_coverage_failure"] = failure
    if mode == COVERAGE:
        cut = failure
        if cut is None:
            return receipt | {
                "status": "not_applicable",
                "reason": "every original flight meets the unchanged train/check coverage",
            }
        if cut == 0:
            return receipt | {
                "reason": "first original flight fails coverage; no earlier complete span to retain"
            }
    elif mode == UNRESOLVED_ENDING:
        try:
            cut = _unresolved_cut(attempt, rows)
        except (KeyError, TypeError, ValueError) as error:
            return receipt | {"reason": str(error)}
        receipt["unresolved_ending_cut"] = cut
        if cut == 0:
            return receipt | {
                "reason": "no supported original contact-to-contact prefix before unresolved input"
            }
    else:
        cut = len(rows) - 1
        receipt["terminal_identity_cut"] = cut
        if cut == 0:
            return receipt | {
                "reason": "final original contact opens the first flight; no earlier span to retain"
            }
        if any(row.get("coverage_qualified") is not True for row in rows[:cut]):
            return receipt | {
                "reason": "terminal-identity prefix requires supported earlier original spans"
            }
    try:
        contract, retained = _contract(
            attempt,
            cameras_document,
            rows,
            cut,
            observation_partition,
            bool(observation_fallback),
            native_window,
            mode=mode,
            identity=identity,
        )
    except (KeyError, TypeError, ValueError) as error:
        return receipt | {"reason": str(error)}
    return receipt | {"status": "qualified", "contract": contract, "retained_events": retained}


def _inside(contract: dict, values, *, closed: bool = False) -> bool:
    """Strictly inside the retained prefix: after its start, before its contact.

    ``closed`` admits the right original contact itself, which is the only epoch
    a source gap may be bounded by; an inserted epoch never reaches it.
    """
    start = float(contract["native_window"][0])
    right = float(contract["modeled_horizon"])
    return all(
        math.isfinite(float(value))
        and start <= float(value)
        and (float(value) < right or closed and float(value) == right)
        for value in values
    )


def interior_admission(
    contract: dict, candidate: dict, *, source_pts_window: tuple[float, float] | None = None
) -> dict:
    """Whether one witnessed interior candidate is scope-local to this prefix.

    Admission is a source-domain question only: the candidate's whole event
    interval, its proposed sub-gaps and its observed actor witness must lie
    strictly inside the retained prefix, and its observation frame must be one
    of the original native rows the contract already published.  Nothing here
    reads a fit, an OFF-selected topology, a chosen epoch or a label.
    """
    from cv.pipeline import s6_optional_contacts as optional

    if contract.get("mode") != TERMINAL_IDENTITY:
        raise ValueError("scope-local interior composition requires the terminal-identity prefix")
    result = {"id": candidate.get("id"), "admitted": False, "reason": None}
    if candidate.get("gap_kind") not in (optional.INTERIOR, optional.INTERIOR_OPTIONAL):
        return result | {"reason": "not_an_interior_contact_proposal"}
    event = candidate.get("event") or {}
    interval = [float(value) for value in event.get("frame_interval", [])]
    if len(interval) != 2 or not _inside(contract, [*interval, event.get("frame", math.nan)]):
        return result | {"reason": "proposal_interval_outside_the_retained_prefix"}
    result["interval"] = interval
    if not _inside(contract, candidate.get("gap_interval", []), closed=True):
        return result | {"reason": "proposal_gap_outside_the_retained_prefix"}
    for entry in candidate.get("subgaps", []):
        if not _inside(contract, entry.get("interval", []), closed=True):
            return result | {"reason": "proposal_support_outside_the_retained_prefix"}
    witness = candidate.get("actor_witness") or {}
    frame = witness.get("observation_frame")
    if frame is None or not _inside(contract, [frame]):
        return result | {"reason": "actor_witness_outside_the_retained_prefix"}
    if int(frame) not in contract.get("original_ball_frames", []):
        return result | {"reason": "actor_witness_is_not_an_original_native_observation"}
    result["actor_witness_frame"] = int(frame)
    for row in (candidate.get("pose") or {}).get("rows", []):
        observed = row.get("observed_frames") or []
        if not observed or not _inside(contract, [row.get("frame", math.nan), *observed]):
            return result | {"reason": "pose_support_outside_the_retained_prefix"}
    audio = candidate.get("audio") or {}
    if audio.get("supported"):
        from cv.pipeline.event_contact_witness import FFT_SAMPLES, HOP_SAMPLES, SAMPLE_RATE

        transient = audio.get("transient") or {}
        epoch = float(transient.get("source_pts_seconds", math.nan))
        width = float(transient.get("width_seconds", math.nan))
        # The registered width spans spectral-flux sample centres. Each sample
        # reads this FFT window and the preceding hop, even for a narrow peak.
        left_support = width / 2 + (FFT_SAMPLES / 2 + HOP_SAMPLES) / SAMPLE_RATE
        right_support = width / 2 + FFT_SAMPLES / 2 / SAMPLE_RATE
        if (
            source_pts_window is None
            or not all(math.isfinite(value) for value in (epoch, width))
            or width < 0
            or not source_pts_window[0] <= epoch - left_support
            or not epoch + right_support < source_pts_window[1]
        ):
            return result | {"reason": "audio_support_outside_the_retained_prefix"}
    return result | {"admitted": True}


def checked_interior_events(contract: dict, original: list[dict], events: list[dict]) -> list[dict]:
    """Admit only scope-local optional contacts on top of the retained events.

    Every original retained event stays verbatim and in order, each addition is
    an optional contact conditioned on its own hypothesis whose entire interval
    is strictly inside the retained prefix, and the right original contact stays
    the last contact, so nothing is inserted at or after the right boundary and
    no ending, bounce or endpoint parameter is created.
    """
    if not isinstance(contract, dict) or contract.get("mode") != TERMINAL_IDENTITY:
        raise ValueError("contact prefix retains its original events and right boundary")
    frames = [float(event["frame"]) for event in events]
    if frames != sorted(frames):
        raise ValueError("ordered prefix-local event topology required")
    added = [event for event in events if event not in original]
    if [event for event in events if event in original] != original:
        raise ValueError("prefix-local composition changed the retained original events")
    right = float(contract["modeled_horizon"])
    for event in added:
        membership = event.get("optional_topology_membership") or {}
        interval = [float(value) for value in event.get("frame_interval", [])]
        if (
            event.get("event_type") != "contact"
            or event.get("occurrence_status") not in ("optional", "predicted")
            or membership.get("conditioned_on_this_hypothesis") is not True
            or membership.get("accepted_source_stream_changed") is not False
            or len(interval) != 2
            or not _inside(contract, [*interval, event["frame"]])
        ):
            raise ValueError("prefix-local addition must be an interior optional source contact")
    contacts = [float(event["frame"]) for event in events if event["event_type"] == "contact"]
    if max(contacts) != right:
        raise ValueError("prefix-local composition moved the right original contact")
    return added


def prefix_local_witness(attempt: dict, document: dict) -> tuple[dict, dict]:
    """Requalify an existing source witness against this prefix, nothing more.

    The returned witness is the same prepared document with every candidate that
    is not scope-local marked unsupported and recorded.  Candidate order, ids,
    gap indices, the existing total alternative cap and the existing
    deterministic input rank are untouched, so no per-flight winner is created
    and no branch is evicted for a reason other than leaving the prefix.
    """
    from cv.experiments.connected_shooting.observation_scope import event_digest
    from cv.pipeline import s6_optional_contacts as optional

    contract = attempt.get("observation_scope")
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ValueError("bound contact-prefix attempt required")
    if contract.get("mode") != TERMINAL_IDENTITY:
        raise ValueError("scope-local interior composition requires the terminal-identity prefix")
    if document.get("schema") not in optional.SCHEMAS or optional.declared_final_scope(document):
        raise ValueError("a prefix-local witness cannot declare the final-contact scope")
    if (
        event_digest(document.get("source_attempt_events", []))
        != contract["original_events_sha256"]
    ):
        raise ValueError("optional contact witness does not bind the original source events")
    result = deepcopy(document)
    # Native source epochs qualify acoustic evidence; no FPS assumption or
    # extrapolation can move a transient across the original contact boundary.
    clock = [
        (float(row["frame"]), float(row["native_pts_seconds"]))
        for row in attempt["owner_ball_labels"]
        if row.get("native_pts_seconds") is not None
    ]
    source_pts_window = None
    if clock:
        import numpy as np

        frames, epochs = np.asarray(clock, float).T
        right = float(contract["modeled_horizon"])
        if (
            np.isfinite(frames).all()
            and np.isfinite(epochs).all()
            and np.all(np.diff(frames) > 0)
            and np.all(np.diff(epochs) > 0)
            and frames[0] < right <= frames[-1]
        ):
            source_pts_window = (float(epochs[0]), float(np.interp(right, frames, epochs)))
    admissions = [
        interior_admission(contract, row, source_pts_window=source_pts_window)
        for row in result.get("candidates", [])
    ]
    for row, admission in zip(result.get("candidates", []), admissions):
        if admission["admitted"]:
            continue
        row["supported"] = False
        row["prefix_scope_refusal"] = admission["reason"]
    receipt = {
        "schema": INTERIOR_SCOPE_SCHEMA,
        "mode": TERMINAL_IDENTITY,
        "interior_composition": INTERIOR_COMPOSITION,
        "retained_window": [
            float(contract["native_window"][0]),
            float(contract["modeled_horizon"]),
        ],
        "right_boundary_kind": RIGHT_BOUNDARY_KIND,
        "admitted_ids": [row["id"] for row in admissions if row["admitted"]],
        "refused": [
            {"id": row["id"], "reason": row["reason"]} for row in admissions if not row["admitted"]
        ],
        "proposal_count": len(admissions),
        "terminal_families": "off",
        "imported_fitted_frames": [],
        "imported_selected_topology": None,
        "evaluation_labels_consulted": False,
        "existing_rank_and_cap_unchanged": True,
        "runtime_model_calls": 0,
    }
    return result, receipt


def bind_attempt(attempt: dict, qualification: dict) -> dict:
    """Bind the prefix scene while preserving every original source declaration."""
    if qualification.get("schema") != SCHEMA or qualification.get("status") != "qualified":
        raise ValueError("qualified contact-prefix receipt required")
    contract = deepcopy(qualification["contract"])
    bound = deepcopy(attempt)
    bound["original_physical_events"] = deepcopy(attempt["events"])
    bound["original_owner_end_frame"] = contract["original_owner_end_frame"]
    bound["original_owner_end_frame_semantics"] = contract["original_owner_end_frame_semantics"]
    bound["original_observation_scope"] = deepcopy(contract["original_observation_scope"])
    bound["original_native_window"] = list(contract["native_window"])
    bound["observation_scope"] = contract
    bound["events"] = deepcopy(qualification["retained_events"])
    bound["owner_end_frame"] = contract["modeled_horizon"]
    bound["owner_end_frame_semantics"] = RIGHT_BOUNDARY_SEMANTICS
    # The prefix scene has no tail grammar; the original tail lives only in the
    # contract.  No terminal bounce, ending or root position is created.
    for key in DROPPED_TAIL_KEYS:
        bound.pop(key, None)
    if "point_end" in bound:
        bound["point_end"] = None
    if "ending_supplied" in bound:
        bound["ending_supplied"] = False
    return bound


def validate(
    attempt: dict,
    cameras_document: dict | None = None,
    *,
    observation_partition: str = "fifth_frame_withheld",
    labels: dict | None = None,
) -> dict | None:
    """Reconstruct the source from a bound prefix attempt, or return ``None``.

    Without cameras this is a shape, prefix-membership and digest check.  With
    cameras the qualification is recomputed from the original inventory, the
    original endpoint declaration and the same observed rows; a tampered cut,
    count, event, horizon or shrunken window cannot pass.
    """
    from cv.experiments.connected_shooting import observation_partition as partition
    from cv.experiments.connected_shooting.observation_scope import event_digest

    contract = attempt.get("observation_scope")
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        return None
    partition.validate(observation_partition)
    if contract.get("observation_partition") != observation_partition:
        raise ValueError("contact prefix cut was computed under a different observation partition")
    original = attempt.get("original_physical_events")
    events = attempt.get("events")
    if not isinstance(original, list) or not isinstance(events, list):
        raise ValueError("bound prefix attempt must retain its original physical inventory")
    right = contract.get("right_contact")
    inventory_rows = contract.get("original_inventory")
    mode = contract.get("mode")
    if mode not in (COVERAGE, TERMINAL_IDENTITY, UNRESOLVED_ENDING):
        raise ValueError("contact prefix contract declares no supported mode")
    if (
        contract.get("right_boundary_kind") != RIGHT_BOUNDARY_KIND
        or contract.get("right_boundary_membership") != RIGHT_BOUNDARY_MEMBERSHIP
        or contract.get("complete_original_source") is not False
        or contract.get("physical_ending") is not None
        or not isinstance(right, dict)
        or right.get("event_type") != "contact"
        or not isinstance(inventory_rows, list)
        or not inventory_rows
    ):
        raise ValueError("contact prefix contract shape invalid")
    cut = contract.get(
        {
            COVERAGE: "first_coverage_failure",
            TERMINAL_IDENTITY: "terminal_identity_cut",
            UNRESOLVED_ENDING: "unresolved_ending_cut",
        }[mode]
    )
    if (
        type(cut) is not int
        or not 1 <= cut < len(inventory_rows)
        or contract.get("retained_flight_indices") != list(range(cut))
        or len(contract.get("retained_flights", [])) != cut
        or [slot["original_flight_index"] for slot in contract.get("unresolved_original_slots", [])]
        != list(range(cut, len(inventory_rows)))
        or contract.get("unresolved_original_flight_count") != len(inventory_rows) - cut
    ):
        raise ValueError("retained/unresolved original slot roster invalid")
    if any(row.get("coverage_qualified") is not True for row in inventory_rows[:cut]):
        raise ValueError("retained original spans are not all supported")
    if mode == COVERAGE:
        if inventory_rows[cut].get("coverage_qualified") is not False:
            raise ValueError(
                "bound cut is not the first coverage failure of the original inventory"
            )
    elif mode == UNRESOLVED_ENDING:
        original_scope = contract.get("original_observation_scope", {})
        if (
            original_scope.get("schema") != UNRESOLVED_INPUT_SCHEMA
            or original_scope.get("ending_semantics") != "unresolved"
            or original_scope.get("physical_ending") is not None
            or contract.get("original_slot_count") != len(inventory_rows)
        ):
            raise ValueError("unresolved ending prefix lost original source inventory")
        source = dict(attempt, events=original, observation_scope=original_scope)
        if _unresolved_cut(source, inventory_rows) != cut:
            raise ValueError("unresolved ending cut crosses unsupported original input")
    elif (
        cut != len(inventory_rows) - 1
        or contract.get("first_coverage_failure") is not None
        or contract.get("interior_composition") != INTERIOR_COMPOSITION
        or contract.get("terminal_families") != "off"
        or contract.get("supplied_terminal_ground_count") != 0
        or contract.get("original_slot_count") != len(inventory_rows)
        or not isinstance(contract.get("terminal_identity_refusal"), dict)
        or contract["terminal_identity_refusal"].get("schema") != IDENTITY_SCHEMA
        or contract["terminal_identity_refusal"].get("refused") is not True
    ):
        raise ValueError("terminal-identity prefix contract shape invalid")
    horizon = contract.get("modeled_horizon")
    if contract.get("observation_horizon") != horizon:
        raise ValueError("prefix horizon aliases must denote the original contact")
    if float(right["frame"]) != horizon or float(inventory_rows[cut]["start_frame"]) != horizon:
        raise ValueError("modeled right boundary is not the failing flight's original contact")
    if (
        event_digest(original) != contract.get("original_events_sha256")
        or len(original) != contract.get("original_event_count")
        or sum(event == right for event in original) != 1
    ):
        raise ValueError("original physical event inventory changed under the prefix contract")
    expected = [event for event in original if float(event["frame"]) <= horizon]
    if (
        events != expected
        or event_digest(events) != contract.get("retained_events_sha256")
        or len(events) != contract.get("retained_event_count")
    ):
        raise ValueError("modeled prefix differs from original physical membership")
    if (
        float(attempt.get("owner_end_frame")) != horizon
        or attempt.get("owner_end_frame_semantics") != RIGHT_BOUNDARY_SEMANTICS
        or attempt.get("original_owner_end_frame") != contract.get("original_owner_end_frame")
        or attempt.get("original_native_window") != contract.get("native_window")
        or attempt.get("original_observation_scope") != contract.get("original_observation_scope")
    ):
        raise ValueError("bound prefix attempt lost an original source declaration")
    if (
        attempt.get("point_end") is not None
        or attempt.get("ending_supplied") is True
        or any(key in attempt for key in DROPPED_TAIL_KEYS)
    ):
        raise ValueError("a contact prefix cannot assert a physical ending or tail grammar")
    frames = [int(row["frame"]) for row in attempt["owner_ball_labels"]]
    window = contract["native_window"]
    if frames != contract.get("original_ball_frames") or event_digest(
        attempt["owner_ball_labels"]
    ) != contract.get("original_ball_rows_sha256"):
        raise ValueError("original native observation window must not be shrunk or changed")
    if cameras_document is not None:
        if mode == TERMINAL_IDENTITY and labels is None:
            raise ValueError("terminal-identity prefix must be replayed against its source rows")
        source = dict(
            attempt,
            events=original,
            owner_end_frame=contract["original_owner_end_frame"],
            owner_end_frame_semantics=contract["original_owner_end_frame_semantics"],
            observation_scope=contract["original_observation_scope"],
            point_end=contract["original_point_end"],
        )
        recomputed = qualify(
            source,
            cameras_document,
            observation_partition=contract["observation_partition"],
            observation_fallback=contract["observation_fallback"],
            native_window=window if contract["native_window_origin"] == "passed" else None,
            mode=mode,
            labels=labels,
        )
        if (
            recomputed["status"] != "qualified"
            or recomputed["contract"] != contract
            or recomputed["retained_events"] != events
        ):
            raise ValueError("contact prefix does not replay from the original source rows")
    return contract
