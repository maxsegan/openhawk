"""Original-contact prefix scope: cut policy, source contract and replay refusal."""

from copy import deepcopy
from pathlib import Path

import pytest

from cv.experiments.connected_shooting import source_flight_coverage as coverage
from cv.pipeline import s6_contact_prefix_scope as scope

CONTACTS = [45.09228515625, 67.91796875, 99.1689453125, 132.666015625]
HORIZON = 135.0


def event(kind, frame):
    return dict(
        event_type=kind,
        frame=frame,
        frame_interval=[frame - 1.0, frame + 1.0],
        status="predicted",
        annotation_origin="automatic",
    )


def ground_scope(window, horizon):
    return dict(
        schema="s6_observed_ground_scope_v2",
        native_window=list(window),
        observation_horizon=float(horizon),
        physical_ending=None,
        ending_semantics="unresolved",
        supplied_ground_count=0,
        terminal_kind="observed_horizon",
    )


def inputs(
    contacts=None, end=HORIZON, bounces=(), unsupported=(), window=(1, 135), with_scope=True
):
    contacts = CONTACTS if contacts is None else contacts
    events = sorted(
        [event("contact", frame) for frame in contacts]
        + [event("bounce", frame) for frame in bounces],
        key=lambda e: (e["frame"], e["event_type"]),
    )
    attempt = dict(
        events=events,
        owner_end_frame=end,
        owner_end_frame_semantics="observation_horizon_not_physical_event",
        owner_ball_labels=[
            dict(frame=frame, status="visible", x1080=float(frame), y1080=2.0 * frame)
            for frame in range(window[0], window[1] + 1)
        ],
        point_clip="pt0046",
        match_id="wta_2020_580_f_226",
        fps=25.0,
        ending_supplied=False,
        point_end=None,
        observed_horizon_tail={"mode": "observed_horizon"},
    )
    if with_scope:
        attempt["observation_scope"] = ground_scope(window, end)
    cameras = dict(
        clip="pt0046",
        match_id="wta_2020_580_f_226",
        cameras=[
            dict(
                frame=frame,
                status="unsupported" if frame in unsupported else "supported",
                P=[[1.0] * 4] * 3,
            )
            for frame in range(window[0], window[1] + 1)
        ],
    )
    return attempt, cameras


def test_mode_is_default_off_and_explicit():
    assert scope.DEFAULT_MODE == "off" and scope.MODES == (
        "off",
        "coverage",
        "terminal_identity",
        "unresolved_ending",
    )
    assert not scope.applicable("off")
    assert scope.applicable("coverage") and scope.applicable("terminal_identity")
    with pytest.raises(ValueError, match="contact prefix mode"):
        scope.validate_mode("on")


def test_fully_covered_source_is_not_applicable():
    attempt, cameras = inputs(end=160.0, window=(1, 160))
    result = scope.qualify(attempt, cameras)
    assert result["status"] == "not_applicable" and result["first_coverage_failure"] is None
    assert "contract" not in result
    assert [row["coverage_qualified"] for row in result["inventory"]] == [True] * 4
    with pytest.raises(ValueError, match="qualified contact-prefix receipt"):
        scope.bind_attempt(attempt, result)


def test_first_flight_coverage_failure_still_holds():
    attempt, cameras = inputs(contacts=[45.09228515625, 48.0, 67.91796875])
    result = scope.qualify(attempt, cameras)
    assert result["status"] == "held" and result["first_coverage_failure"] == 0
    assert "no earlier complete span" in result["reason"] and "contract" not in result


def test_terminal_gap_retains_the_source46_prefix_without_an_ending():
    attempt, cameras = inputs()
    result = scope.qualify(attempt, cameras)
    assert result["status"] == "qualified" and result["first_coverage_failure"] == 3
    contract = result["contract"]
    assert contract["schema"] == "s6_observed_contact_prefix_v1"
    assert contract["retained_flight_indices"] == [0, 1, 2]
    assert contract["right_contact"] == attempt["events"][3]
    assert contract["right_boundary_kind"] == "original_contact"
    assert contract["modeled_horizon"] == CONTACTS[3]
    assert contract["complete_original_source"] is False
    assert contract["physical_ending"] is None
    assert contract["original_ending_kind"] == "unresolved_horizon"
    assert contract["original_owner_end_frame"] == HORIZON
    assert contract["original_observation_scope"] == ground_scope((1, 135), HORIZON)
    assert contract["native_window"] == [1, 135]
    assert contract["native_window_origin"] == "original_observation_scope"
    assert contract["modeled_native_window"] == [1, CONTACTS[3]]
    assert contract["original_event_count"] == 4
    assert contract["first_contact_role"] is None
    assert [(row["train_count"], row["check_count"]) for row in contract["original_inventory"]] == [
        (18, 4),
        (26, 6),
        (26, 7),
        (2, 1),
    ]
    assert contract["unresolved_original_slots"] == [
        {
            "original_flight_index": 3,
            "start_frame": CONTACTS[3],
            "end_frame": HORIZON,
            "inclusive_end": True,
            "native_count": 3,
            "train_count": 2,
            "check_count": 1,
            "reason": "coverage_cut",
        }
    ]
    assert contract["right_contact_supported_native_frames"] == [132, 133]
    assert result["retained_events"] == attempt["events"][:4]


def test_middle_gap_keeps_only_the_earlier_qualified_span():
    attempt, cameras = inputs(contacts=[45.09228515625, 67.91796875, 70.0, 99.1689453125])
    result = scope.qualify(attempt, cameras)
    assert result["status"] == "qualified" and result["contract"]["first_coverage_failure"] == 1
    contract = result["contract"]
    assert contract["retained_flight_indices"] == [0]
    assert contract["modeled_horizon"] == 67.91796875
    assert [slot["original_flight_index"] for slot in contract["unresolved_original_slots"]] == [
        1,
        2,
        3,
    ]
    assert contract["unresolved_original_flight_count"] == 3
    assert len(result["retained_events"]) == 2


def test_retained_membership_is_the_original_interior_membership():
    attempt, cameras = inputs()
    contract = scope.qualify(attempt, cameras)["contract"]
    original = coverage.inventory(attempt, cameras)
    for retained, row in zip(contract["retained_flights"], original):
        assert retained["native_frames"] == row["native_frames"]
        assert retained["train_frames"] == row["train_frames"]
        assert retained["check_frames"] == row["check_frames"]
        assert retained["inclusive_end"] is False
    assert contract["retained_membership"] == "original_interior_half_open_unchanged"
    assert contract["right_boundary_membership"] == "half_open_original_interior"


def test_integer_right_contact_on_a_fifth_frame_stays_in_the_dropped_flight():
    attempt, cameras = inputs(contacts=[45.09228515625, 67.91796875, 130.0], end=133.0)
    result = scope.qualify(attempt, cameras)
    contract = result["contract"]
    assert contract["first_coverage_failure"] == 2 and contract["modeled_horizon"] == 130.0
    last = contract["retained_flights"][-1]
    assert 130 not in last["native_frames"] and 130 not in last["check_frames"]
    assert last["native_frames"][-1] == 129
    assert contract["original_inventory"][2]["native_frames"] == [130, 131, 132, 133]
    assert contract["right_contact_supported_native_frames"] == [129, 130, 131]


@pytest.mark.parametrize("count,status", [(0, "qualified"), (1, "qualified"), (2, "held")])
def test_retained_prefix_flight_allows_zero_or_one_supplied_ground(count, status):
    attempt, cameras = inputs(bounces=[57.5, 60.25][:count])
    result = scope.qualify(attempt, cameras)
    assert result["status"] == status
    if status == "held":
        assert "zero or one supplied ground event" in result["reason"]
    else:
        assert result["contract"]["retained_flights"][0]["source_bounce_frames"] == [57.5][:count]


def test_wrong_camera_identity_or_support_holds():
    attempt, cameras = inputs()
    held = scope.qualify(attempt, dict(cameras, clip="pt0047"))
    assert held["status"] == "held" and "one matching camera document" in held["reason"]
    attempt, cameras = inputs(unsupported=(50,))
    strict = scope.qualify(attempt, cameras, observation_fallback=False)
    assert strict["status"] == "held" and "supported matching camera" in strict["reason"]
    assert scope.qualify(attempt, cameras)["status"] == "qualified"
    attempt, cameras = inputs(unsupported=(132, 133))
    unsupported_right = scope.qualify(attempt, cameras)
    assert unsupported_right["status"] == "held"
    assert "lacks supported native ball/camera evidence" in unsupported_right["reason"]


def test_bind_attempt_preserves_the_original_source_and_creates_no_ending():
    attempt, cameras = inputs()
    before = deepcopy(attempt)
    qualification = scope.qualify(attempt, cameras)
    bound = scope.bind_attempt(attempt, qualification)
    assert attempt == before
    assert bound["events"] == attempt["events"][:4]
    assert bound["original_physical_events"] == attempt["events"]
    assert bound["owner_end_frame"] == CONTACTS[3]
    assert bound["owner_end_frame_semantics"] == scope.RIGHT_BOUNDARY_SEMANTICS
    assert bound["original_owner_end_frame"] == HORIZON
    assert bound["original_owner_end_frame_semantics"] == ("observation_horizon_not_physical_event")
    assert bound["original_native_window"] == [1, 135]
    assert bound["original_observation_scope"] == attempt["observation_scope"]
    assert "observed_horizon_tail" not in bound and "terminal_net_tail" not in bound
    assert bound["point_end"] is None and bound["ending_supplied"] is False
    assert bound["owner_ball_labels"] == attempt["owner_ball_labels"]
    assert not [e for e in bound["events"] if e["event_type"] == "bounce"]
    assert bound["observation_scope"] == qualification["contract"]


def test_validate_replays_the_cut_from_the_original_source_rows():
    attempt, cameras = inputs()
    bound = scope.bind_attempt(attempt, scope.qualify(attempt, cameras))
    assert scope.validate(bound)["first_coverage_failure"] == 3
    assert scope.validate(bound, cameras)["modeled_horizon"] == CONTACTS[3]
    assert scope.validate(attempt, cameras) is None
    assert scope.validate({"observation_scope": None}) is None
    with pytest.raises(ValueError, match="different observation partition"):
        scope.validate(bound, cameras, observation_partition="all_native")


def test_validate_accepts_a_label_extent_window_without_a_source_scope():
    attempt, cameras = inputs(with_scope=False)
    qualification = scope.qualify(attempt, cameras)
    contract = qualification["contract"]
    assert contract["native_window_origin"] == "owner_ball_label_extent"
    assert contract["original_ending_kind"] == "original_physical_ending"
    assert contract["original_observation_scope"] is None
    bound = scope.bind_attempt(attempt, qualification)
    assert scope.validate(bound, cameras) == contract
    passed = scope.qualify(attempt, cameras, native_window=[1, 135])["contract"]
    assert passed["native_window_origin"] == "passed"
    assert (
        scope.validate(
            scope.bind_attempt(
                attempt,
                dict(
                    schema=scope.SCHEMA,
                    status="qualified",
                    contract=passed,
                    retained_events=[e for e in attempt["events"] if e["frame"] <= CONTACTS[3]],
                ),
            ),
            cameras,
        )
        == passed
    )


def tampered(change):
    attempt, cameras = inputs()
    bound = scope.bind_attempt(attempt, scope.qualify(attempt, cameras))
    contract = bound["observation_scope"]
    if change == "original_inventory":
        contract["original_inventory"][0]["train_count"] = 99
    elif change == "cut":
        contract["first_coverage_failure"] = 2
    elif change == "retained_roster":
        contract["retained_flight_indices"] = [0, 1]
    elif change == "right_contact":
        contract["right_contact"] = event("contact", 120.0)
    elif change == "right_contact_support":
        contract["right_contact_supported_native_frames"] = [131, 132, 133]
    elif change == "event_count":
        contract["original_event_count"] = 3
    elif change == "dropped_original_event":
        bound["original_physical_events"] = bound["original_physical_events"][:3]
    elif change == "added_post_cut_event":
        bound["events"] = [*bound["events"], event("bounce", 134.0)]
    elif change == "horizon":
        contract["original_owner_end_frame"] = 160.0
        contract["original_ending_frame"] = 160.0
        bound["original_owner_end_frame"] = 160.0
    elif change == "modeled_horizon":
        contract["modeled_horizon"] = 133.0
    elif change == "complete":
        contract["complete_original_source"] = True
    elif change == "boundary_kind":
        contract["right_boundary_kind"] = "observed_horizon"
    elif change == "invented_ending":
        bound["point_end"] = dict(frame=HORIZON)
    elif change == "reinstated_tail":
        bound["observed_horizon_tail"] = {"mode": "observed_horizon"}
    elif change == "shrunk_window":
        bound["owner_ball_labels"] = [
            row for row in bound["owner_ball_labels"] if row["frame"] <= 133
        ]
    elif change == "retained_membership":
        contract["retained_flights"][0]["train_frames"].append(132)
    return bound, cameras


@pytest.mark.parametrize(
    "change",
    [
        "original_inventory",
        "cut",
        "retained_roster",
        "right_contact",
        "right_contact_support",
        "event_count",
        "dropped_original_event",
        "added_post_cut_event",
        "horizon",
        "modeled_horizon",
        "complete",
        "boundary_kind",
        "invented_ending",
        "reinstated_tail",
        "shrunk_window",
        "retained_membership",
    ],
)
def test_tampered_prefix_contract_is_rejected(change):
    bound, cameras = tampered(change)
    with pytest.raises(ValueError):
        scope.validate(bound, cameras)


def forged_earlier_cut():
    """A self-consistent contract whose cut is not the first coverage failure."""
    from cv.experiments.connected_shooting.observation_scope import event_digest

    attempt, cameras = inputs()
    bound = scope.bind_attempt(attempt, scope.qualify(attempt, cameras))
    contract = bound["observation_scope"]
    retained = [e for e in bound["original_physical_events"] if e["frame"] <= CONTACTS[2]]
    contract.update(
        first_coverage_failure=2,
        retained_flight_indices=[0, 1],
        retained_flights=contract["retained_flights"][:2],
        unresolved_original_slots=[
            dict(slot, original_flight_index=index)
            for index, slot in zip(
                (2, 3), [contract["unresolved_original_slots"][0]] * 2, strict=True
            )
        ],
        unresolved_original_flight_count=2,
        right_contact=deepcopy(bound["original_physical_events"][2]),
        modeled_horizon=CONTACTS[2],
        modeled_native_window=[1, CONTACTS[2]],
        retained_events_sha256=event_digest(retained),
        retained_event_count=len(retained),
    )
    contract["unresolved_original_slots"][0]["start_frame"] = CONTACTS[2]
    contract["unresolved_original_slots"][1]["start_frame"] = CONTACTS[3]
    bound.update(events=retained, owner_end_frame=CONTACTS[2])
    return bound, cameras


def test_consistently_forged_earlier_cut_is_rejected_with_and_without_cameras():
    bound, cameras = forged_earlier_cut()
    with pytest.raises(ValueError, match="first coverage failure"):
        scope.validate(bound)
    with pytest.raises(ValueError, match="first coverage failure"):
        scope.validate(bound, cameras)


def test_prefix_helper_reads_no_fitted_input():
    source = Path(scope.__file__).read_text()
    assert not any(
        token in source
        for token in ("result.json", "verdict", "candidate_", "scipy", "optimi", "gate")
    )
    assert "fps" not in source and "spin" not in source
