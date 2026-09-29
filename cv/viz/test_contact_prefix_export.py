"""Contact-prefix export: right-boundary contact, unresolved receipt and source replay.

Every fixture here is a small manufactured *source* document run through the
actual shared helpers (``s6_contact_prefix_scope`` / ``s6_contact_prefix_runtime``);
no contract is hand-written and no fitted state is read.
"""

from copy import deepcopy
import json

import pytest

from cv.pipeline import s6_contact_prefix_runtime as runtime
from cv.viz import export_connected_3d as legacy
from cv.viz import export_local_s6 as export

CLIP = "pt0046"
MATCH = "wta_2020_580_f_226"
#: Three original contacts; the last original span (50 -> 52) has too few
#: training rows, so the cut retains two complete contact-to-contact flights.
CONTACTS = [10.0, 30.0, 50.0]
HORIZON = 52.0
WINDOW = (1, 52)


def native_time(frame):
    return 100.0 + (float(frame) - 1.0) / 25.0


def event(kind, frame):
    return dict(
        event_type=kind,
        frame=frame,
        frame_interval=[frame - 1.0, frame + 1.0],
        status="predicted",
        annotation_origin="automatic",
    )


def source_inputs(contacts=None, end=HORIZON, window=WINDOW):
    contacts = CONTACTS if contacts is None else contacts
    attempt = dict(
        events=[event("contact", frame) for frame in contacts],
        owner_end_frame=end,
        owner_end_frame_semantics="observation_horizon_not_physical_event",
        owner_ball_labels=[
            dict(frame=frame, status="visible", x1080=float(frame), y1080=2.0 * frame)
            for frame in range(window[0], window[1] + 1)
        ],
        point_clip=CLIP,
        match_id=MATCH,
        fps=25.0,
        ending_supplied=False,
        point_end=None,
        observation_scope=dict(
            schema="s6_observed_ground_scope_v2",
            native_window=list(window),
            observation_horizon=float(end),
            physical_ending=None,
            ending_semantics="unresolved",
            supplied_ground_count=0,
            terminal_kind="observed_horizon",
        ),
    )
    cameras = dict(
        clip=CLIP,
        match_id=MATCH,
        cameras=[
            dict(frame=frame, status="supported", P=[[1.0] * 4] * 3)
            for frame in range(window[0], window[1] + 1)
        ],
    )
    return attempt, cameras


def bound_source(mode="coverage", **kwargs):
    """Original packet, cameras and the packet the shared helper actually binds."""
    attempt, cameras = source_inputs(**kwargs)
    if mode == "unresolved_ending":
        from cv.pipeline.s6_contact_prefix_scope import UNRESOLVED_INPUT_SCHEMA

        attempt["observation_scope"]["schema"] = UNRESOLVED_INPUT_SCHEMA
    packet = {"attempts": [deepcopy(attempt)]}
    prepared, receipt = runtime.prepare_packet(
        deepcopy(packet), source_labels(), cameras, mode, "fifth_frame_withheld"
    )
    return packet, cameras, prepared, receipt


def source_labels(contacts=None):
    contacts = CONTACTS if contacts is None else contacts
    return {
        "attempt": {"clip": CLIP, "ending_kind": "unresolved"},
        "source_pack": {"fps": 25.0, "native_size": [1920, 1080], "images": []},
        "events": {"records": [{**event("contact", frame), "clip": CLIP} for frame in contacts]},
        "ball": {
            "records": [
                {
                    "clip": CLIP,
                    "frames": [
                        {
                            "frame": frame,
                            "status": "visible",
                            "x1080": float(frame),
                            "y1080": 2.0 * frame,
                        }
                        for frame in range(WINDOW[0], WINDOW[1] + 1)
                    ],
                }
            ]
        },
    }


def measured_candidate():
    """Two measured flights closing exactly on the third original contact."""
    flights = [
        {
            "start_frame": 10.0,
            "end_frame": 30.0,
            "positions": [[1.0, 2.0, 1.5], [5.0, 8.0, 2.0], [9.0, 14.0, 0.9]],
            "velocities": [[4.0, 6.0, 1.0], [4.0, 6.0, 0.0], [4.0, 6.0, -1.0]],
            "bounces": [],
            "net_hits": [],
        },
        {
            "start_frame": 30.0,
            "end_frame": 50.0,
            "positions": [[9.0, 14.0, 0.9], [6.0, 9.0, 2.2], [3.0, 4.0, 1.1]],
            "velocities": [[-3.0, -5.0, 2.0], [-3.0, -5.0, 0.0], [-3.0, -5.0, -2.0]],
            "bounces": [],
            "net_hits": [],
        },
    ]
    return {
        "measurement": {
            "dense_flights": flights,
            "contact_xyz": [flights[0]["positions"][0], flights[1]["positions"][0]],
            "native_projection": [
                {"frame": frame, "predicted": [float(frame), 2.0 * frame], "error_px": 1.0}
                for frame in range(10, 51)
            ],
        }
    }


def prefix_result(contract):
    """Minimal actual-stage shaped result for a bound prefix scene."""
    candidate = measured_candidate()
    verdict = {
        "complete_point": False,
        "flight_count": 2,
        "accepted_flight_count": 2,
        "flights": [
            {"start_frame": 10.0, "end_frame": 30.0, "role": "serve", "accepted": True},
            {"start_frame": 30.0, "end_frame": 50.0, "role": "rally", "accepted": True},
        ],
        "failure_counts": {},
        "contact_prefix_reconstruction": {
            "scope": deepcopy(contract),
            "geometry_gates_passed": True,
            "complete_original_source": False,
            "retained_original_flight_indices": contract["retained_flight_indices"],
            "unresolved_original_slots": deepcopy(contract["unresolved_original_slots"]),
            "reference_flight_coverage": "not_evaluated_here",
        },
    }
    return {
        "key": "point1",
        "verdict": verdict,
        "measurement": candidate["measurement"],
        "evaluation_context": {"scene": {"contact_frames": list(CONTACTS)}},
        "scorer": {"rung": "cold"},
    }


def prefix_search(packet, contract):
    return {
        "clip": CLIP,
        "attempt_id": f"{CLIP}__{MATCH}_pt0046",
        "events": deepcopy(packet["attempts"][0]["events"]),
        "configuration": {
            "observation_scope": deepcopy(contract),
            "contact_prefix_scope": contract["mode"],
            "right_boundary_kind": "original_contact",
            "surface": "hard",
            "exposure_duration_frames": 0.25,
        },
        "player_states": [],
        "inputs": [],
    }


# --------------------------------------------------------------------------
# The extra original contact
# --------------------------------------------------------------------------


def test_last_original_contact_is_incoming_only_and_invents_no_outgoing_flight():
    _, _, prepared, receipt = bound_source()
    contract = receipt["contract"]
    candidate = measured_candidate()
    search = prefix_search(prepared, contract)
    ordinary = legacy._contacts(candidate, search, native_time)
    assert [row["index"] for row in ordinary] == [0, 1]

    rows = legacy._contacts(
        candidate, search, native_time, right_boundary=contract["right_contact"]
    )
    assert ordinary == rows[:-1], "retained contacts keep their exact original output"
    last = rows[-1]
    assert last["index"] == 2 and last["labeled_frame"] == last["fitted_frame"] == 50.0
    assert last["boundary_kind"] == "original_contact"
    assert last["outgoing_flight_modeled"] is False
    # Incoming geometry is the measured end of the final flight, nothing else.
    assert [last["x"], last["y"], last["z"]] == [3.0, 4.0, 1.1]
    assert last["v_in"] == [-3.0, -5.0, -2.0] and last["speed_in"] is not None
    assert last["v_out"] == [None, None, None] and last["speed_out"] is None
    assert last["spin_out"]["mag"] is None and last["racket_normal"] == [None, None, None]
    # k measured flights against k + 1 original contacts.
    assert len(candidate["measurement"]["dense_flights"]) + 1 == len(rows)


def test_right_boundary_refuses_a_contact_the_measurement_does_not_end_on():
    _, _, prepared, receipt = bound_source()
    contract = receipt["contract"]
    candidate = measured_candidate()
    search = prefix_search(prepared, contract)

    moved = {**deepcopy(contract["right_contact"]), "frame": 49.0}
    with pytest.raises(ValueError, match="differs from the bound original contact"):
        legacy._contacts(candidate, search, native_time, right_boundary=moved)

    short = deepcopy(candidate)
    short["measurement"]["dense_flights"][-1]["end_frame"] = 49.0
    with pytest.raises(ValueError, match="must end at the original right contact"):
        legacy._contacts(short, search, native_time, right_boundary=contract["right_contact"])

    padded = deepcopy(search)
    padded["events"] = [*padded["events"], event("contact", 51.0)]
    with pytest.raises(ValueError, match="one original contact past the last flight"):
        legacy._contacts(candidate, padded, native_time, right_boundary=contract["right_contact"])


def test_ordinary_contacts_are_untouched_without_a_declared_right_boundary():
    _, _, prepared, receipt = bound_source()
    candidate = measured_candidate()
    search = prefix_search(prepared, receipt["contract"])
    assert legacy._contacts(candidate, search, native_time) == legacy._contacts(
        candidate, search, native_time, right_boundary=None
    )


# --------------------------------------------------------------------------
# Output claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["coverage", "unresolved_ending"])
def test_prefix_output_binds_the_retained_roster_and_refuses_a_complete_point(mode):
    _, _, prepared, receipt = bound_source(mode=mode)
    contract = receipt["contract"]
    result = prefix_result(contract)
    search = prefix_search(prepared, contract)
    export.checked_contact_prefix_output(result, search, contract)

    for mutation, error in (
        (lambda r: r["verdict"].update(complete_point=True), "complete-point"),
        (lambda r: r["verdict"].update(complete_original_point=True), "complete-point"),
        (lambda r: r["verdict"].pop("contact_prefix_reconstruction"), "complete-point"),
        (
            lambda r: r["verdict"]["contact_prefix_reconstruction"].update(
                complete_original_source=True
            ),
            "complete-point",
        ),
        (
            lambda r: r["verdict"]["contact_prefix_reconstruction"].update(
                unresolved_original_slots=[]
            ),
            "complete-point",
        ),
        (
            lambda r: r["verdict"].update(observed_scope_reconstruction={"scope": {}}),
            "observed-horizon tail",
        ),
        (
            lambda r: r["evaluation_context"]["scene"].update(contact_frames=[10.0, 30.0, 49.0]),
            "original right contact",
        ),
        # Padding the measured roster to the original source size is refused.
        (
            lambda r: r["verdict"].update(flight_count=3),
            "retained original flights",
        ),
        (
            lambda r: r["evaluation_context"]["scene"].update(
                contact_frames=[10.0, 30.0, 40.0, 50.0]
            ),
            "retained original flights",
        ),
    ):
        broken = prefix_result(contract)
        mutation(broken)
        with pytest.raises(ValueError, match=error):
            export.checked_contact_prefix_output(broken, search, contract)


def test_prefix_output_requires_its_explicit_shared_search_policy():
    _, _, prepared, receipt = bound_source()
    contract = receipt["contract"]
    result = prefix_result(contract)
    for name in ("contact_prefix_scope", "right_boundary_kind"):
        search = prefix_search(prepared, contract)
        search["configuration"].pop(name)
        with pytest.raises(ValueError, match="explicit shared search policy"):
            export.checked_contact_prefix_output(result, search, contract)


def test_checked_observation_scope_routes_a_bound_prefix_packet():
    _, _, prepared, receipt = bound_source()
    contract = receipt["contract"]
    result = prefix_result(contract)
    search = prefix_search(prepared, contract)
    labels = source_labels()
    original = deepcopy(prepared)
    export.checked_observation_scope(result, search, prepared, labels)
    assert prepared == original, "the reader never rewrites the source packet"

    tampered = deepcopy(prepared)
    tampered["attempts"][0]["observation_scope"]["complete_original_source"] = True
    with pytest.raises(ValueError, match="contact prefix contract shape invalid"):
        export.checked_observation_scope(result, search, tampered, labels)

    dropped = deepcopy(labels)
    dropped["events"]["records"] = dropped["events"]["records"][:2]
    with pytest.raises(ValueError, match="differ from bound consumer"):
        export.checked_observation_scope(result, search, prepared, dropped)


# --------------------------------------------------------------------------
# Review context and the original pictures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["coverage", "unresolved_ending"])
def test_review_context_stops_modeled_xyz_at_the_contact_and_keeps_later_video(mode):
    _, _, prepared, receipt = bound_source(mode=mode)
    contract = receipt["contract"]
    result = prefix_result(contract)
    original = deepcopy(result)
    lo, hi = WINDOW
    context = export.competitive_context(result, native_time, lo, hi)
    assert result == original
    assert context["kind"] == "local_s6_observation_scope"
    assert context["right_boundary_kind"] == "original_contact"
    assert context["modeled_horizon_frame"] == context["score_end_frame"] == 50.0
    assert context["display_end_frame"] == 50.0
    assert context["physical_ending_frame"] is None
    assert context["complete_original_source"] is False
    assert context["unresolved_original_flight_count"] == 1
    assert context["source_ending_frame"] == HORIZON
    # The whole original window stays playable past the right contact.
    assert context["view_end_t"] == native_time(hi) > context["score_end_t"]
    assert context["source_native_window"] == [lo, hi]

    frames = legacy._candidate_frames(measured_candidate(), lo, hi, native_time, "cold")
    legacy._mark_frame_acceptance(frames, {**result["verdict"], "accepted_flight_indices": [0, 1]})
    export.mark_context_frames(frames, context)
    after = [index for index, frame in enumerate(frames["frame"]) if frame > 50]
    assert after, "original frames beyond the cut are retained"
    assert all(frames["x"][index] is None for index in after)
    assert all(frames["acceptance"][index] == "context" for index in after)
    assert all(frames["t"][index] == native_time(frames["frame"][index]) for index in after)
    # And the source video overlay still covers them.
    overlay = legacy._video_overlay(source_labels(), measured_candidate(), CLIP, lo, hi)
    assert [row["frame"] for row in overlay["frames"]] == list(range(lo, hi + 1))
    assert overlay["frames"][-1]["labeled_front"] == [52.0, 104.0]
    assert overlay["frames"][-1]["fitted_projection"] is None


def test_review_context_refuses_a_horizon_outside_the_measured_support():
    _, _, prepared, receipt = bound_source()
    result = prefix_result(receipt["contract"])
    result["verdict"]["contact_prefix_reconstruction"]["scope"]["modeled_horizon"] = 49.0
    with pytest.raises(ValueError, match="prefix horizon aliases"):
        export.competitive_context(result, native_time, *WINDOW)
    result = prefix_result(receipt["contract"])
    result["evaluation_context"]["scene"]["contact_frames"][-1] = 49.0
    with pytest.raises(ValueError, match="modeled support"):
        export.competitive_context(result, native_time, *WINDOW)
    result = prefix_result(receipt["contract"])
    result["verdict"]["observed_scope_reconstruction"] = {"scope": {"observation_horizon": 50.0}}
    with pytest.raises(ValueError, match="one right-boundary reconstruction"):
        export.competitive_context(result, native_time, *WINDOW)


# --------------------------------------------------------------------------
# The separate coverage receipt
# --------------------------------------------------------------------------


def test_coverage_receipt_lists_original_ordinals_and_unresolved_slots():
    _, _, _, receipt = bound_source()
    contract = receipt["contract"]
    coverage = export.contact_prefix_coverage(contract)
    assert coverage["retained_original_flight_indices"] == [0, 1]
    assert coverage["measured_flight_count"] == 2
    assert coverage["original_flight_slot_count"] == 3
    assert coverage["unresolved_original_flight_count"] == 1
    assert [slot["original_flight_index"] for slot in coverage["unresolved_original_slots"]] == [2]
    slot = coverage["unresolved_original_slots"][0]
    assert slot["start_frame"] == 50.0 and slot["end_frame"] == HORIZON
    assert slot["reason"] == "coverage_cut" and "positions" not in slot
    assert coverage["complete_original_source"] is False
    assert coverage["original_native_window"] == [1, 52]
    assert coverage["modeled_native_window"] == [1, 50.0]
    # The original endpoint declaration is forwarded, never reinterpreted.
    assert coverage["original_ending_frame"] == HORIZON
    assert coverage["original_ending_kind"] == contract["original_ending_kind"]
    assert coverage["reference_flight_count"] == "not opened or inferred"
    # A receipt, not a measured array: no trajectory field can appear here.
    assert "dense_flights" not in json.dumps(coverage)


# --------------------------------------------------------------------------
# Source-binding replay
# --------------------------------------------------------------------------


def stage_invocation(packet, cameras, policy=None):
    """Reproduce the stage's own prefix preparation and recorded receipts."""
    from cv.pipeline import s6_labeled_stage as stage

    policy = {"contact_prefix_scope": "coverage"} if policy is None else policy
    labels = source_labels()
    prepared, receipt = runtime.prepare_packet(
        deepcopy(packet),
        labels,
        cameras,
        stage.shared_settings(policy)["contact_prefix_scope"],
        stage.shared_settings(policy)["observation_partition"],
    )
    applied, applicability = runtime.applied_policy(policy, prepared)
    invocation = {
        "policy": policy,
        "shared_settings": stage.shared_settings(applied),
        "scope_applicability": applicability,
        "preparation": {"contact_prefix_scope": receipt},
    }
    return json.loads(json.dumps(invocation)), json.loads(json.dumps(prepared))


def test_replay_rebuilds_the_prefix_from_the_original_inputs():
    packet, cameras, prepared, _ = bound_source()
    invocation, executed = stage_invocation(packet, cameras)
    replayed, receipt = export.replayed_contact_prefix(
        invocation, deepcopy(packet), source_labels(), cameras
    )
    assert json.loads(json.dumps(replayed)) == executed
    assert receipt == invocation["preparation"]["contact_prefix_scope"]
    assert receipt["contract"]["first_coverage_failure"] == 2
    assert prepared["attempts"][0]["observation_scope"] == receipt["contract"]


def test_replay_refuses_a_re_enabled_terminal_setting_or_a_changed_applicability():
    packet, cameras, _, _ = bound_source()
    policy = {"contact_prefix_scope": "coverage", "terminal_ground_normal": "on"}
    invocation, _ = stage_invocation(packet, cameras, policy)
    # The recorded abstention is what actually ran.
    assert invocation["shared_settings"]["terminal_ground_normal"] == "off"
    assert invocation["scope_applicability"]["changes"]["terminal_ground_normal"] == {
        "requested": "on",
        "applied": "off",
        "reason": "original_contact_right_boundary",
    }
    export.replayed_contact_prefix(invocation, deepcopy(packet), source_labels(), cameras)

    reenabled = deepcopy(invocation)
    reenabled["shared_settings"]["terminal_ground_normal"] = "on"
    with pytest.raises(ValueError, match="replayed prefix abstention"):
        export.replayed_contact_prefix(reenabled, deepcopy(packet), source_labels(), cameras)

    forged = deepcopy(invocation)
    forged["scope_applicability"]["changes"] = {}
    with pytest.raises(ValueError, match="scope applicability"):
        export.replayed_contact_prefix(forged, deepcopy(packet), source_labels(), cameras)


@pytest.mark.parametrize("fully_covered", [False, True])
def test_replay_accepts_an_implicit_default_added_after_the_producer(fully_covered):
    options = {"end": 80.0, "window": (1, 80)} if fully_covered else {}
    packet, cameras, _, _ = bound_source(**options)
    invocation, executed = stage_invocation(packet, cameras)
    del invocation["shared_settings"]["terminal_net_membership"]
    replayed, _ = export.replayed_contact_prefix(
        invocation, deepcopy(packet), source_labels(), cameras
    )
    assert json.loads(json.dumps(replayed)) == executed


@pytest.mark.parametrize("selection", ["off", "predicted"])
def test_replay_still_requires_receipts_for_explicit_settings(selection):
    packet, cameras, _, _ = bound_source()
    policy = {"contact_prefix_scope": "coverage", "terminal_net_membership": selection}
    invocation, _ = stage_invocation(packet, cameras, policy)
    del invocation["shared_settings"]["terminal_net_membership"]
    with pytest.raises(ValueError, match="replayed prefix abstention"):
        export.replayed_contact_prefix(invocation, deepcopy(packet), source_labels(), cameras)


def test_replay_refuses_prefix_inputs_without_the_explicit_shared_policy():
    packet, cameras, prepared, _ = bound_source()
    invocation, _ = stage_invocation(packet, cameras)
    with pytest.raises(ValueError, match="explicit shared coverage policy"):
        export.replayed_contact_prefix(
            {**invocation, "policy": {"contact_prefix_scope": "off"}},
            deepcopy(prepared),
            source_labels(),
            cameras,
        )


def test_first_flight_coverage_failure_still_holds_on_replay():
    # No earlier complete span exists, so the source is prepared as it is today.
    from cv.pipeline import s6_labeled_stage as stage

    contacts = [10.0, 13.0, 30.0, 50.0]
    short, short_cameras = source_inputs(contacts=contacts)
    policy = {"contact_prefix_scope": "coverage"}
    invocation = {
        "policy": policy,
        "shared_settings": stage.shared_settings(policy),
        "scope_applicability": None,
        "preparation": {},
    }
    replayed, receipt = export.replayed_contact_prefix(
        invocation, {"attempts": [deepcopy(short)]}, source_labels(contacts), short_cameras
    )
    assert receipt["status"] == "held" and "contract" not in receipt
    assert receipt["first_coverage_failure"] == 0
    assert replayed["attempts"][0] == short


def test_fully_covered_source_keeps_todays_prepared_scene():
    packet, cameras, prepared, receipt = bound_source(end=80.0, window=(1, 80))
    assert receipt["status"] == "not_applicable"
    assert prepared == packet, "a covered source is prepared exactly as it is today"
    assert export.contact_prefix_scope({"verdict": {}}) is None
