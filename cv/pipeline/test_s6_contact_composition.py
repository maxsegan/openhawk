"""Bounded two-contact composition: source compatibility, receipts, refusals."""

from copy import deepcopy
import pytest
from cv.pipeline import s6_contact_composition as composition, s6_optional_contacts as optional


class Tripwire(list):
    def __iter__(self):
        raise AssertionError("observation record read while the policy is off")


class Forbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("evaluation gate read")


def contact(frame, interval):
    return {"event_type": "contact", "frame": frame, "frame_interval": list(interval)}


def bounce(frame, interval):
    return {"event_type": "bounce", "frame": frame, "frame_interval": list(interval)}


def source_events():
    """The shape of the native source-14 failure: one gap, two supplied grounds."""
    return [
        contact(91.7, [91.2, 92.2]),
        bounce(110.0, [109.5, 110.5]),
        bounce(174.0, [173.5, 174.5]),
        contact(188.25, [187.75, 188.75]),
    ]


def candidate(name, frame, interval, *, witness=0.9, odds=0.8, gap_index=0, gap=(91.7, 188.25)):
    return {
        "id": name,
        "gap_index": gap_index,
        "gap_interval": list(gap),
        "event": {
            "event_type": "contact",
            "frame": frame,
            "frame_interval": list(interval),
            "occurrence_status": "optional",
            "status": "predicted",
        },
        "witness_support": witness,
        "occurrence_log_odds": odds,
        "supported": True,
    }


def observations(frames=range(90, 191), replaced=None):
    rows = []
    for frame in frames:
        row = {
            "frame": float(frame),
            "status": "visible",
            "x1080": 500.0 + frame,
            "y1080": 400.0,
            "support_class": "unresolved_or_direct_tracker_observation",
        }
        row.update((replaced or {}).get(int(frame), {}))
        rows.append(row)
    return rows


def witness(events, candidates, policy=composition.POLICY):
    document = {
        "schema": optional.SCHEMA,
        "source_attempt_events": deepcopy(events),
        "gaps": optional.inconsistent_gaps(events),
        "candidates": candidates,
    }
    if policy is not None:
        document["contact_composition"] = {"policy": policy}
    return document


def legacy_beam(groups, interior):
    """The unchanged single-addition beam; Astra's caller supplies this in place."""
    beam = [()]
    for gap_index in interior:
        beam = sorted(
            ((*a, b) for a in beam for b in optional._ranked(groups[gap_index])),
            key=lambda rows: (
                -sum(r["witness_support"] for r in rows),
                -sum(r["occurrence_log_odds"] for r in rows),
                tuple(r["id"] for r in rows),
            ),
        )[: optional.MAX_ADDED_HYPOTHESES]
    return beam


def source_case(rows=None, policy=composition.POLICY, frames=range(90, 191), replaced=None):
    events = source_events()
    rows = rows or [
        candidate("emission_123", 123.0, [122.0, 124.0]),
        candidate("emission_158", 158.0, [157.0, 159.0], witness=0.85),
    ]
    groups = {0: rows}
    return {
        "document": witness(events, rows, policy),
        "events": events,
        "legacy_beam": legacy_beam(groups, [0]),
        "groups": groups,
        "interior": [0],
        "observations": observations(frames, replaced),
    }


def extend(case):
    return composition.extend_interior_beam(
        case["document"],
        case["events"],
        case["legacy_beam"],
        case["groups"],
        case["interior"],
        case["observations"],
    )


def eligible_pairs(receipt):
    return [row for row in receipt["pairs"] if row["eligible"]]


def sole_refusal(receipt):
    assert len(receipt["pairs"]) == 1
    return receipt["pairs"][0]["reason"]


def added_hypothesis(rows, retimed=None):
    added = []
    for row in rows:
        event = deepcopy(row["event"])
        event.update((retimed or {}).get(row["id"], {}))
        event["optional_topology_membership"] = {
            "candidate_id": row["id"],
            "conditioned_on_this_hypothesis": True,
            "accepted_source_stream_changed": False,
        }
        added.append(event)
    return {"name": "optional_contacts_2", "added": added}


def test_declared_policy_is_off_unless_the_witness_declares_it():
    document = {"gaps": []}
    assert composition.policy(document) == "off"
    assert composition.policy({**document, "contact_composition": {"policy": "off"}}) == "off"
    on = {**document, "contact_composition": {"policy": composition.POLICY}}
    before = deepcopy(on)
    assert composition.policy(on) == composition.POLICY
    assert on == before
    for malformed in ("on", "bounded pairs", None, True):
        with pytest.raises(ValueError, match="contact composition declaration"):
            composition.policy({**document, "contact_composition": {"policy": malformed}})
    for malformed in ([composition.POLICY], composition.POLICY, {}, None):
        with pytest.raises(ValueError, match="contact composition declaration"):
            composition.policy({**document, "contact_composition": malformed})


def test_off_returns_the_original_beam_object_without_reading_observations():
    case = source_case(policy=None)
    case["observations"] = Tripwire()
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"] and receipt is None
    case["document"]["contact_composition"] = {"policy": "off"}
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"] and receipt is None
    assert composition.admissibility(
        added_hypothesis(case["groups"][0]), case["document"], case["events"], Tripwire()
    ) == {
        "schema": composition.SCHEMA,
        "policy": "off",
        "eligible": True,
        "pairs": [],
        "checked_gaps": [],
        "candidate_ids": [],
        "status": "policy_off",
    }


def test_source_pool_order_matches_the_unchanged_single_addition_sort():
    rows = [
        candidate("emission_a", 140.0, [139.0, 141.0], witness=0.5, odds=0.9),
        candidate("emission_b", 123.0, [122.0, 124.0], witness=0.9, odds=0.2),
        candidate("emission_c", 158.0, [157.0, 159.0], witness=0.9, odds=0.2),
        candidate("emission_d", 150.0, [149.0, 151.0], witness=0.1, odds=0.1),
    ]
    pool = composition.ranked_pool(rows)
    assert [row["id"] for row in pool] == ["emission_b", "emission_c", "emission_a"]
    assert pool[: optional.MAX_ADDED_HYPOTHESES] == optional._ranked(rows)
    assert len(pool) == composition.MAX_POOL


def test_held_source_pair_qualifies_with_a_zero_ground_volley_subgap():
    case = source_case()
    beam, receipt = extend(case)
    assert len(beam) == composition.MAX_ALTERNATIVES
    assert beam[0] is case["legacy_beam"][0]
    assert [row["id"] for row in beam[0]] == ["emission_123"]
    assert [row["id"] for row in beam[1]] == ["emission_123", "emission_158"]
    assert receipt["status"] == "paired_alternative_added"
    assert receipt["chosen_paired_candidate_ids"] == ["emission_123", "emission_158"]
    assert receipt["pools"] == [
        {
            "gap_index": 0,
            "supported_candidates": 2,
            "pool_candidate_ids": ["emission_123", "emission_158"],
        }
    ]
    pair = receipt["pairs"][0]
    assert pair["eligible"] and pair["reason"] is None
    assert [row["name"] for row in pair["subgaps"]] == [
        "left_contact_to_first",
        "first_to_second",
        "second_to_right_contact",
    ]
    # The middle return is a volley: no supplied ground, and that is not a veto.
    assert [row["supplied_bounce_frames"] for row in pair["subgaps"]] == [[110.0], [], [174.0]]
    assert [row["observation_count"] for row in pair["subgaps"]] == [29, 32, 28]
    assert all(row["observation_count"] >= composition.MIN_OBSERVATIONS for row in pair["subgaps"])


def test_paired_branch_keeps_six_parameters_per_contact_outside_the_helper():
    beam, _ = extend(source_case())
    added = len(beam[1])
    row = {
        "evidence": {"input_geometry_penalty": 0.5},
        "measurement": {"rms_px": {"training": 4.0}},
    }
    assert added == 2
    assert optional.physical_rank(row, added, 0.5, 40) == optional.physical_rank(
        row, 0, 0.5, 40, added_parameters=6 * added
    )
    assert not any(
        hasattr(composition, name)
        for name in ("physical_rank", "best_completed", "fit_alternatives", "hypotheses")
    )


def test_subgap_boundaries_are_the_actual_original_contact_intervals():
    case = source_case()
    _, receipt = extend(case)
    pair = receipt["pairs"][0]
    left, middle, right = pair["subgaps"]
    assert left["interval"] == [92.2, 122.0] and right["interval"] == [159.0, 187.75]
    assert middle["interval"] == [124.0, 157.0]
    # Boundaries are the supplied contact intervals, not their frames and not a
    # fabricated centre: frames inside an accepted contact support no subgap.
    assert 92 not in left["observation_frames"] and left["observation_frames"][0] == 93
    assert 188 not in right["observation_frames"] and right["observation_frames"][-1] == 187
    assert min(middle["observation_frames"]) == 125
    assert composition.gap_boundaries(case["document"], case["events"], 7) == (
        None,
        "gap_index_outside_the_witness",
    )
    # Only an original interior gap has two supplied contacts to bound it.
    case["document"]["gaps"][0]["kind"] = optional.FINAL
    assert composition.gap_boundaries(case["document"], case["events"], 0) == (
        None,
        "gap_is_not_an_original_interior_gap",
    )


def test_missing_gap_boundary_contact_refuses_instead_of_inventing_one():
    case = source_case()
    case["events"][0]["frame"] = 91.6
    _, receipt = extend(case)
    assert sole_refusal(receipt) == "gap_boundary_is_not_one_supplied_contact"
    assert receipt["status"] == "no_eligible_pair"
    case = source_case()
    case["events"].append(contact(91.7, [91.0, 92.0]))
    _, receipt = extend(case)
    assert sole_refusal(receipt) == "gap_boundary_is_not_one_supplied_contact"


@pytest.mark.parametrize(
    "frame,interval,reason",
    [
        (123.5, [123.0, 124.5], "pair_intervals_overlap_or_are_not_strictly_ordered"),
        (123.0, [122.0, 124.0], "pair_intervals_overlap_or_are_not_strictly_ordered"),
        (124.0, [124.0, 126.0], "pair_intervals_overlap_or_are_not_strictly_ordered"),
        (110.2, [109.6, 110.8], "pair_overlaps_a_supplied_event_interval"),
        (92.1, [91.8, 92.6], "pair_overlaps_a_supplied_event_interval"),
        (190.0, [189.0, 191.0], "pair_is_not_wholly_inside_the_original_gap"),
        (float("nan"), [122.0, 124.0], "non_finite_or_unordered_candidate_interval"),
        (123.0, [124.0, 122.0], "non_finite_or_unordered_candidate_interval"),
    ],
)
def test_incompatible_second_contact_is_refused_with_an_explicit_reason(frame, interval, reason):
    rows = [
        candidate("emission_123", 123.0, [122.0, 124.0]),
        candidate("emission_x", frame, interval, witness=0.85),
    ]
    case = source_case(rows)
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    assert sole_refusal(receipt) == reason and not eligible_pairs(receipt)


@pytest.mark.parametrize(
    "replaced,reason",
    [
        ({"status": "derived_estimate"}, "status_is_not_visible"),
        ({"support_class": "interpolated_estimate"}, "interpolated_estimate_is_not_an_observation"),
        ({"x1080": float("nan")}, "non_finite_native_point"),
        ({"y1080": None}, "unreadable_native_point"),
        ({"frame": 130.5}, "non_integer_native_frame"),
    ],
)
def test_derived_or_unusable_rows_are_not_observations_and_starve_the_subgap(replaced, reason):
    case = source_case(replaced={frame: replaced for frame in range(125, 157)})
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    pair = receipt["pairs"][0]
    assert pair["reason"] == "new_subgap_below_the_original_observation_floor"
    assert pair["subgaps"][1]["observation_count"] < composition.MIN_OBSERVATIONS
    assert [row["observation_count"] for row in pair["subgaps"]] == [29, 0, 28]
    assert {row["reason"] for row in receipt["observations"]["excluded"]} == {reason}
    assert receipt["observations"]["interpolated_or_derived_used"] is False


def test_subgap_just_below_the_observation_floor_is_refused_and_counted():
    frames = [*range(90, 125), 130, 140, 150, *range(157, 191)]
    beam, receipt = extend(source_case(frames=frames))
    pair = receipt["pairs"][0]
    assert pair["subgaps"][1]["observation_frames"] == [130, 140, 150]
    assert pair["reason"] == "new_subgap_below_the_original_observation_floor"
    beam, receipt = extend(source_case(frames=[*frames, 145]))
    assert [row["id"] for row in beam[1]] == ["emission_123", "emission_158"]
    assert receipt["pairs"][0]["subgaps"][1]["observation_frames"] == [130, 140, 145, 150]
    empty = source_case(frames=[])
    beam, receipt = extend(empty)
    assert beam is empty["legacy_beam"]
    assert receipt["observations"]["admitted_frames"] == 0
    assert receipt["pairs"][0]["subgaps"][0]["observation_frames"] == []


def test_conflicting_duplicate_native_frame_supports_no_subgap():
    case = source_case(frames=[*range(90, 191), *range(125, 157)])
    duplicates = [row for row in case["observations"] if 125 <= row["frame"] < 157][32:]
    for row in duplicates:
        row["x1080"] += 3.0
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    assert receipt["pairs"][0]["subgaps"][1]["observation_count"] == 0
    assert {row["reason"] for row in receipt["observations"]["excluded"]} == {
        "conflicting_duplicate_native_frame"
    }


def test_supplied_bounces_keep_their_intervals_and_refuse_a_crowded_subgap():
    rows = [
        candidate("emission_178", 178.0, [177.0, 179.0]),
        candidate("emission_182", 182.5, [181.5, 183.5], witness=0.85),
    ]
    case = source_case(rows)
    original = deepcopy(case["events"])
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    pair = receipt["pairs"][0]
    assert pair["reason"] == "more_than_one_supplied_bounce_in_one_new_subgap"
    assert pair["subgaps"][0]["supplied_bounce_frames"] == [110.0, 174.0]
    assert case["events"] == original


def test_top_three_pool_permits_a_compatible_lower_ranked_pair():
    rows = [
        candidate("emission_140", 140.0, [122.5, 158.5], witness=1.0),
        candidate("emission_123", 123.0, [122.0, 124.0], witness=0.9),
        candidate("emission_158", 158.0, [157.0, 159.0], witness=0.85),
        candidate("emission_low", 150.0, [149.0, 151.0], witness=0.05),
    ]
    case = source_case(rows)
    beam, receipt = extend(case)
    assert receipt["pools"][0]["pool_candidate_ids"] == [
        "emission_140",
        "emission_123",
        "emission_158",
    ]
    assert receipt["pools"][0]["supported_candidates"] == 4
    assert len(receipt["pairs"]) == composition.MAX_PAIRS
    assert [row["candidate_ids"] for row in eligible_pairs(receipt)] == [
        ["emission_123", "emission_158"]
    ]
    assert [row["id"] for row in beam[0]] == ["emission_140"]
    assert [row["id"] for row in beam[1]] == ["emission_123", "emission_158"]


def test_reversed_candidate_order_gives_the_same_beam_and_receipt():
    rows = [
        candidate("emission_123", 123.0, [122.0, 124.0]),
        candidate("emission_158", 158.0, [157.0, 159.0]),
    ]
    forward = source_case(rows)
    reverse = source_case(list(reversed(deepcopy(rows))))
    first, one = extend(forward)
    second, two = extend(reverse)
    assert one == two
    assert [[row["id"] for row in branch] for branch in first] == [
        [row["id"] for row in branch] for branch in second
    ]
    assert one["chosen_paired_candidate_ids"] == ["emission_123", "emission_158"]


def two_gap_case(second_pairable=True):
    events = [
        *source_events(),
        bounce(210.0, [209.5, 210.5]),
        bounce(280.0, [279.5, 280.5]),
        contact(300.0, [299.5, 300.5]),
    ]
    first = [
        candidate("emission_123", 123.0, [122.0, 124.0]),
        candidate("emission_158", 158.0, [157.0, 159.0], witness=0.85),
    ]
    later = [231.0, 233.0] if second_pairable else [231.0, 261.0]
    second = [
        candidate("emission_232", 232.0, later, gap_index=1, gap=(188.25, 300.0), witness=0.7),
        candidate(
            "emission_260",
            260.0,
            [259.0, 261.0],
            gap_index=1,
            gap=(188.25, 300.0),
            witness=0.6,
        ),
    ]
    groups = {0: first, 1: second}
    return {
        "document": witness(events, first + second),
        "events": events,
        "legacy_beam": legacy_beam(groups, [0, 1]),
        "groups": groups,
        "interior": [0, 1],
        "observations": observations(range(90, 305)),
    }


def test_two_interior_gaps_keep_support_under_the_bounded_two_state_search():
    case = two_gap_case()
    beam, receipt = extend(case)
    assert len(beam) == composition.MAX_ALTERNATIVES
    assert [row["id"] for row in beam[0]] == ["emission_123", "emission_232"]
    assert receipt["chosen_paired_candidate_ids"] == [
        "emission_123",
        "emission_158",
        "emission_232",
        "emission_260",
    ]
    assert [row["gap_index"] for row in receipt["pools"]] == [0, 1]
    single = two_gap_case(second_pairable=False)
    beam, receipt = extend(single)
    assert receipt["chosen_paired_candidate_ids"] == [
        "emission_123",
        "emission_158",
        "emission_232",
    ]
    assert len(beam) == composition.MAX_ALTERNATIVES
    assert [row["eligible"] for row in receipt["pairs"]] == [True, False]


def test_unsupported_interior_gap_keeps_the_original_beam():
    case = two_gap_case()
    case["groups"][1] = []
    case["legacy_beam"] = []
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    assert receipt["status"] == "interior_gap_without_supported_candidate"
    case = source_case()
    case["legacy_beam"] = []
    beam, receipt = extend(case)
    assert beam is case["legacy_beam"]
    assert receipt["status"] == "no_original_branch_to_accompany"
    assert eligible_pairs(receipt)


def test_no_input_is_mutated_by_preparation_or_recheck():
    case = source_case()
    before = deepcopy(case)
    beam, receipt = extend(case)
    hypothesis = added_hypothesis(beam[1])
    frozen = deepcopy(hypothesis)
    composition.admissibility(hypothesis, case["document"], case["events"], case["observations"])
    assert case == before and hypothesis == frozen
    assert beam[1][0] is case["groups"][0][0]


def test_recheck_uses_the_actual_hypothesis_events_after_retiming():
    case = source_case()
    beam, _ = extend(case)
    receipt = composition.admissibility(
        added_hypothesis(beam[1]), case["document"], case["events"], case["observations"]
    )
    assert receipt["eligible"] and receipt["checked_gaps"] == [0]
    assert receipt["status"] == "rechecked"
    assert receipt["pairs"][0]["candidate_ids"] == ["emission_123", "emission_158"]
    collided = added_hypothesis(
        beam[1], {"emission_123": {"frame": 157.5, "frame_interval": [156.5, 158.5]}}
    )
    receipt = composition.admissibility(
        collided, case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"]
    assert receipt["pairs"][0]["reason"] == "pair_intervals_overlap_or_are_not_strictly_ordered"
    starved = added_hypothesis(
        beam[1], {"emission_123": {"frame": 96.0, "frame_interval": [95.0, 97.0]}}
    )
    receipt = composition.admissibility(
        starved, case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"]
    assert receipt["pairs"][0]["reason"] == "new_subgap_below_the_original_observation_floor"
    # The refreshed receipt reports the retimed pair's actual observed support.
    assert receipt["pairs"][0]["subgaps"][0]["observation_frames"] == [93, 94]


def test_recheck_admits_a_single_addition_and_reads_no_gate():
    case = source_case()
    for row in case["document"]["candidates"]:
        row["verdict"] = Forbidden()
    for event in case["events"]:
        event["verdict"] = Forbidden()
    receipt = composition.admissibility(
        added_hypothesis(case["groups"][0][:1]),
        case["document"],
        case["events"],
        case["observations"],
    )
    assert receipt["eligible"] and receipt["pairs"] == [] and receipt["checked_gaps"] == []
    assert receipt["status"] == "at_most_one_addition_per_gap"
    assert receipt["candidate_ids"] == ["emission_123"]
    _, pair_receipt = extend(case)
    assert eligible_pairs(pair_receipt)


def test_repeated_and_unknown_candidate_identities_are_refused():
    case = source_case()
    rows = case["groups"][0]
    repeated = added_hypothesis([rows[0], rows[0]])
    receipt = composition.admissibility(
        repeated, case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"] and receipt["status"] == "repeated_added_candidate_id"
    assert receipt["candidate_id"] == "emission_123"
    unknown = added_hypothesis(rows)
    unknown["added"][1]["optional_topology_membership"]["candidate_id"] = "emission_other"
    receipt = composition.admissibility(
        unknown, case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"] and receipt["status"] == "unknown_added_candidate_id"
    missing = added_hypothesis(rows)
    missing["added"][0].pop("optional_topology_membership")
    receipt = composition.admissibility(
        missing, case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"] and receipt["status"] == "unknown_added_candidate_id"
    boundaries, _ = composition.gap_boundaries(case["document"], case["events"], 0)
    duplicated = composition.qualify_pair(
        [("emission_123", rows[0]["event"])] * 2, boundaries, case["events"], [], gap_index=0
    )
    assert duplicated["reason"] == "repeated_candidate_id" and not duplicated["eligible"]


def test_more_than_two_added_contacts_in_one_gap_are_refused():
    rows = [
        candidate("emission_123", 123.0, [122.0, 124.0]),
        candidate("emission_140", 140.0, [139.0, 141.0], witness=0.88),
        candidate("emission_158", 158.0, [157.0, 159.0], witness=0.85),
    ]
    case = source_case(rows)
    receipt = composition.admissibility(
        added_hypothesis(rows), case["document"], case["events"], case["observations"]
    )
    assert not receipt["eligible"]
    assert receipt["pairs"][0]["reason"] == "more_than_two_added_contacts_in_one_gap"
    beam, pair_receipt = extend(case)
    assert len(beam[1]) == composition.MAX_ADDED_PER_GAP
