"""Final-gap optional contact scope: source-shaped synthetic fixtures only.

No source5 constant, clip identity or frame threshold appears here: every
fixture is a small generic accepted stream plus generic held emission rows.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from cv.pipeline import s6_optional_bounces, s6_optional_contacts as optional
from cv.pipeline import s6_optional_event_union as union

CLIP = "match__pt0001"
PTS = {i: 100 + i * 0.04 for i in range(1, 200)}


def accepted(*rows):
    return [
        {"event_type": kind, "frame": float(f), "frame_interval": [f - 0.2, f + 0.2]}
        for kind, f in rows
    ]


def held_contact(frame, phase=None, contact=0.9, none=0.05):
    return {
        "clip": CLIP,
        "event_type": "contact",
        "frame": frame,
        "location": {"frame_subpixel": frame},
        "abstain": True,
        "class_probabilities": {"contact": contact, "none": none},
        "phase": phase,
    }


def held_bounce(frame, bounce=0.8):
    return {
        "clip": CLIP,
        "event_type": "bounce",
        "frame": frame,
        "location": {"frame_subpixel": frame},
        "abstain": True,
        "class_probabilities": {
            "bounce": bounce,
            "none": 1 - bounce,
            "contact": 0.0,
            "net_hit": 0.0,
        },
    }


def build(
    events,
    emissions,
    end_boundary,
    *,
    cuts=frozenset(),
    tail_contract=None,
    ending_semantics=optional.SUPPLIED_ENDING,
):
    gap = optional.final_gap(events, end_boundary)
    assert gap is not None
    return optional.final_gap_candidates(
        events,
        emissions,
        CLIP,
        PTS,
        gap,
        0,
        cuts=set(cuts),
        tail_contract=tail_contract,
        ending_semantics=ending_semantics,
    )


# --- proposal shape -------------------------------------------------------


def test_final_gap_runs_from_last_accepted_contact_to_the_declared_ending():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40), ("bounce", 90))
    gap = optional.final_gap(events, 100)
    assert gap["interval"] == [40.0, 100.0] and gap["kind"] == optional.FINAL
    assert [e["frame"] for e in gap["bounces"]] == [90.0]
    assert gap["last_contact_interval"] == [39.8, 40.2]
    # No contact, no declared ending, or an ending at/before the last contact.
    assert optional.final_gap(accepted(("bounce", 5)), 100) is None
    assert optional.final_gap(events, None) is None
    assert optional.final_gap(events, 40) is None


def test_supplied_ending_ground_stays_ordered_after_the_added_contact():
    events = accepted(("contact", 40), ("bounce", 50), ("bounce", 90))
    rows, excluded = build(events, [held_contact(60), held_contact(95)], 100)
    assert [r["event"]["frame"] for r in rows] == [60]
    assert rows[0]["preceding_ground"] == "accepted"
    assert [r["reason"] for r in excluded] == ["accepted_terminal_ground_precedes_candidate"]


def test_only_an_actual_supplied_ending_makes_a_final_gap_ground_terminal():
    # One accepted ground, then a candidate return, then live observed suffix.
    events = accepted(("contact", 40), ("bounce", 50))
    emissions = [held_contact(60)]
    # The automatic observation horizon is not a physical ending, so the accepted
    # ground is an ordinary bounce and the later return is proposed.
    rows, excluded = build(events, emissions, 100, ending_semantics=optional.SCOPE_HORIZON_ENDING)
    assert [r["event"]["frame"] for r in rows] == [60] and not excluded
    assert rows[0]["preceding_ground"] == "accepted"
    # Same for an unresolved tail, whose own suffix contract still has to support it.
    contract = {"observation_horizon": 110.0, "native_tail_frames": list(range(41, 110))}
    rows, excluded = build(
        events,
        emissions,
        110,
        tail_contract=contract,
        ending_semantics=optional.OBSERVED_HORIZON_ENDING,
    )
    assert [r["event"]["frame"] for r in rows] == [60] and not excluded
    rows, excluded = build(
        events,
        emissions,
        110,
        tail_contract={**contract, "native_tail_frames": [41, 42]},
        ending_semantics=optional.OBSERVED_HORIZON_ENDING,
    )
    assert not rows and [r["reason"] for r in excluded] == ["no_original_native_suffix_support"]
    # An actual supplied ending keeps the terminal ordering requirement.
    rows, excluded = build(events, emissions, 100, ending_semantics=optional.SUPPLIED_ENDING)
    assert not rows
    assert [r["reason"] for r in excluded] == ["accepted_terminal_ground_precedes_candidate"]
    # Two accepted grounds are still a dead ball under every ending.
    dead = accepted(("contact", 40), ("bounce", 45), ("bounce", 50))
    rows, excluded = build(dead, emissions, 100, ending_semantics=optional.SCOPE_HORIZON_ENDING)
    assert not rows and [r["reason"] for r in excluded] == ["dead_ball_before_candidate"]


def test_final_gap_candidates_require_explicit_ending_semantics():
    events = accepted(("contact", 40), ("bounce", 50))
    with pytest.raises(ValueError, match="explicit declared final-gap ending semantics"):
        build(events, [held_contact(60)], 100, ending_semantics="unresolved")


def test_declared_end_boundary_keeps_a_nonphysical_horizon_nonphysical():
    supplied = {"owner_end_frame": 100.0}
    assert optional.declared_end_boundary(supplied) == (100.0, None, optional.SUPPLIED_ENDING)
    # The automatic packet declares ending_supplied=False and states that its legacy
    # owner_end_frame alias is an observation horizon; neither is a supplied ending.
    scoped = {
        "owner_end_frame": 100.0,
        "ending_supplied": False,
        "owner_end_frame_semantics": optional.SCOPE_HORIZON_ENDING,
        "observation_scope": {"observation_horizon": 100.0},
    }
    assert optional.declared_end_boundary(scoped) == (100.0, None, optional.SCOPE_HORIZON_ENDING)
    assert optional.declared_end_boundary({**scoped, "owner_end_frame_semantics": None}) == (
        100.0,
        None,
        optional.SCOPE_HORIZON_ENDING,
    )
    tail = {"observation_horizon": 90.0}
    assert optional.declared_end_boundary({**scoped, "observed_horizon_tail": tail}) == (
        90.0,
        tail,
        optional.OBSERVED_HORIZON_ENDING,
    )
    assert optional.declared_end_boundary({}) == (None, None, optional.NO_ENDING)


def test_zero_ground_volley_is_proposed_and_records_its_missing_ground():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    rows, excluded = build(events, [held_contact(60)], 100)
    assert len(rows) == 1 and rows[0]["preceding_ground"] == "none" and not excluded


def test_held_ground_before_a_candidate_is_recorded_not_counted():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    rows, _ = build(events, [held_bounce(50), held_contact(60)], 100)
    assert [r["preceding_ground"] for r in rows] == ["held"]
    # The bounce arm's own source rule decides what counts as a held ground.
    assert s6_optional_bounces.held_bounce_row(held_bounce(50), CLIP)
    assert not s6_optional_bounces.held_bounce_row(held_bounce(50, bounce=0.2), CLIP)


def test_two_accepted_grounds_before_a_candidate_and_collection_are_excluded():
    events = accepted(("contact", 40), ("bounce", 50), ("bounce", 60), ("bounce", 90))
    rows, excluded = build(events, [held_contact(70), held_contact(75, "collection")], 100)
    reasons = [r["reason"] for r in excluded]
    assert not rows and reasons == ["dead_ball_before_candidate"]


def test_candidate_overlapping_the_last_accepted_contact_is_excluded():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    # A held row whose generated interval still covers the accepted contact.
    rows, excluded = build(events, [held_contact(41.1)], 100)
    assert not rows
    assert [r["reason"] for r in excluded] == ["candidate_overlaps_the_last_accepted_contact"]


def test_native_view_cut_inside_the_candidate_interval_is_excluded():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    rows, excluded = build(events, [held_contact(60)], 100, cuts={60})
    assert not rows
    assert [r["reason"] for r in excluded] == ["native_view_cut_inside_candidate_interval"]


def test_no_original_native_suffix_support_is_excluded_with_its_reason():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    contract = {
        "observation_horizon": 100.0,
        "native_tail_frames": list(range(41, 62)),  # no five contiguous rows after 61
    }
    supported = {**contract, "native_tail_frames": list(range(41, 100))}
    semantics = optional.OBSERVED_HORIZON_ENDING
    rows, excluded = build(
        events, [held_contact(60)], 100, tail_contract=contract, ending_semantics=semantics
    )
    assert not rows
    assert [r["reason"] for r in excluded] == ["no_original_native_suffix_support"]
    assert excluded[0]["detail"].startswith("ValueError")
    rows, excluded = build(
        events, [held_contact(60)], 100, tail_contract=supported, ending_semantics=semantics
    )
    assert len(rows) == 1 and not excluded


def test_source_emitted_occurrence_rule_and_generated_interval_are_unchanged():
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    rows, _ = build(
        events,
        [
            held_contact(60),
            {**held_contact(62), "abstain": False, "gate_held": False},
            held_contact(64, contact=0.2, none=0.7),
        ],
        100,
    )
    assert [r["event"]["frame"] for r in rows] == [60]
    interior = optional.source_candidates(
        accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40)),
        [held_contact(25)],
        CLIP,
        PTS,
    )
    assert rows[0]["event"]["frame_interval"] == [59.0, 61.0]
    assert interior[0]["event"]["frame_interval"] == [24.0, 26.0]


# --- beam ------------------------------------------------------------------


def witness(events, interior_rows, final_rows, *, scoped=True):
    gaps = [{**g, "kind": optional.INTERIOR} for g in optional.inconsistent_gaps(events)]
    candidates = []
    for index, row in enumerate(interior_rows):
        candidates.append({**row, "gap_index": 0, "gap_kind": optional.INTERIOR})
    if final_rows:
        gaps.append({"interval": [40.0, 100.0], "kind": optional.FINAL, "bounces": []})
    for row in final_rows:
        candidates.append({**row, "gap_index": len(gaps) - 1, "gap_kind": optional.FINAL})
    document = {
        "schema": optional.SCHEMA_FINAL if scoped else optional.SCHEMA,
        "source_attempt_events": events,
        "gaps": gaps if scoped else [{k: v for k, v in g.items() if k != "kind"} for g in gaps],
        "candidates": candidates,
    }
    if scoped:
        document["scope"] = optional.FINAL_SCOPE
    return document


def supported_row(identifier, frame, support=0.8):
    return {
        "id": identifier,
        "supported": True,
        "witness_support": support,
        "occurrence_log_odds": 1.0,
        "event": {
            "event_type": "contact",
            "frame": float(frame),
            "frame_interval": [frame - 1.0, frame + 1.0],
        },
    }


def test_off_parity_of_the_interior_beam():
    events = accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40))
    rows = [supported_row("a", 25), supported_row("b", 27, 0.7)]
    legacy = optional.hypotheses(witness(events, rows, [], scoped=False), events)
    assert [h["name"] for h in legacy] == ["optional_contacts_1", "optional_contacts_2"]
    assert all("beam_composition" not in h for h in legacy)
    assert all("optional_gap_kind" not in e for h in legacy for e in h["added"])
    scoped = optional.hypotheses(witness(events, rows, []), events)
    assert [h["candidate_ids"] for h in scoped] == [h["candidate_ids"] for h in legacy]
    assert [h["beam_composition"] for h in scoped] == ["interior_only", "interior_only"]


def test_final_addition_does_not_require_an_interior_candidate():
    # No interior gap exists at all: the final addition stands on its own.
    events = accepted(("contact", 10), ("bounce", 20), ("contact", 40))
    document = witness(events, [], [supported_row("f1", 60), supported_row("f2", 70, 0.6)])
    assert optional.scoped_gaps(document, optional.INTERIOR) == []
    beam = optional.hypotheses(document, events)
    assert [h["beam_composition"] for h in beam] == ["final_only", "final_only"]
    assert [h["candidate_ids"] for h in beam] == [["f1"], ["f2"]]
    assert all(e["optional_gap_kind"] == "final" for h in beam for e in h["added"])
    assert all(all(e in h["events"] for e in events) for h in beam)


def test_interior_alternatives_are_retained_with_one_final_addition():
    events = accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40))
    document = witness(events, [supported_row("a", 25)], [supported_row("f1", 60)])
    beam = optional.hypotheses(document, events)
    assert [h["candidate_ids"] for h in beam] == [["a"], ["a", "f1"]]
    assert [h["beam_composition"] for h in beam] == ["interior_only", "interior_and_final"]
    assert len(beam) <= optional.MAX_ADDED_HYPOTHESES


def test_unsupported_interior_gap_still_blocks_an_interior_repair():
    events = accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40))
    document = witness(events, [], [])
    assert optional.hypotheses(document, events) == []


def test_v1_witness_may_not_carry_a_final_gap_or_a_scope():
    events = accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40))
    document = witness(events, [supported_row("a", 25)], [supported_row("f1", 60)], scoped=False)
    document["gaps"][1] = {"interval": [40.0, 100.0], "kind": optional.FINAL, "bounces": []}
    with pytest.raises(ValueError, match="undeclared final"):
        optional.hypotheses(document, events)
    with pytest.raises(ValueError, match="unsupported optional contact witness scope"):
        optional.declared_final_scope({"schema": optional.SCHEMA, "scope": optional.FINAL_SCOPE})
    assert not optional.declared_final_scope({"schema": optional.SCHEMA})


def test_hypotheses_still_bind_the_unchanged_accepted_events():
    events = accepted(("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40))
    document = witness(events, [supported_row("a", 25)], [supported_row("f1", 60)])
    before = deepcopy(events)
    with pytest.raises(ValueError, match="unchanged"):
        optional.hypotheses(document, events[:-1])
    assert events == before


# --- union composition -----------------------------------------------------


def contact_hypothesis(name, frame, kind="final"):
    return {
        "name": name,
        "added": [
            {
                "event_type": "contact",
                "frame": float(frame),
                "frame_interval": [frame - 1.0, frame + 1.0],
                "optional_gap_kind": kind,
            }
        ],
        "candidate_ids": [f"c_{frame}"],
        "occurrence_log_odds": 1.0,
        "added_parameter_count": 6,
    }


def bounce_hypothesis(name, frame, candidate="b1"):
    return {
        "name": name,
        "added": [
            {
                "event_type": "bounce",
                "frame": float(frame),
                "frame_interval": [frame - 0.5, frame + 0.5],
            }
        ],
        "candidate_ids": [candidate],
        "occurrence_log_odds": 0.5,
    }


def documents(mode="classifier", scoped=True):
    contact = {"schema": optional.SCHEMA_FINAL if scoped else optional.SCHEMA}
    if scoped:
        contact["scope"] = optional.FINAL_SCOPE
    bounce = {
        "mode": mode,
        "inventory": {
            "gaps": [[40.0, 100.0]],
            "candidates": [{"id": "b1", "gap_index": 0}],
        },
    }
    return {"contact": contact, "bounce": bounce}


def test_final_contact_composes_with_a_v1_bounce_witness_without_enabling_v2():
    docs = documents()
    assert union.composes(docs) and not union.composes_v2(docs)
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 70)],
        "bounce": [bounce_hypothesis("optional_bounces_1", 50)],
    }
    joint, receipts = union.composed(docs, {"events": accepted(("contact", 40))}, rows)
    assert [r["status"] for r in receipts] == ["composed"]
    assert joint[0]["added_contact_count"] == 1 and joint[0]["added_bounce_count"] == 1
    assert joint[0]["added_parameter_count"] == 6
    assert [e["frame"] for e in joint[0]["events"]] == [40.0, 50.0, 70.0]


def test_interior_only_composition_stays_v2_only():
    docs = documents()
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 25, kind="interior")],
        "bounce": [bounce_hypothesis("optional_bounces_1", 50)],
    }
    _, receipts = union.composed(docs, {"events": []}, rows)
    assert receipts[0]["reason"] == "interior_only_composition_requires_the_v2_bounce_witness"


def test_overlapping_added_intervals_are_refused():
    docs = documents()
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 50)],
        "bounce": [bounce_hypothesis("optional_bounces_1", 50)],
    }
    joint, receipts = union.composed(docs, {"events": []}, rows)
    assert not joint
    assert receipts[0]["reason"] == "added_contact_and_bounce_intervals_overlap"


def test_a_separated_ground_composes_before_and_after_a_final_contact():
    docs = documents()
    events = accepted(("contact", 40))
    for frame, order in ((50, [40.0, 50.0, 70.0]), (80, [40.0, 70.0, 80.0])):
        rows = {
            "contact": [contact_hypothesis("optional_contacts_1", 70)],
            "bounce": [bounce_hypothesis("optional_bounces_1", frame)],
        }
        joint, receipts = union.composed(docs, {"events": events}, rows)
        assert [r["status"] for r in receipts] == ["composed"]
        assert [e["frame"] for e in joint[0]["events"]] == order
        assert joint[0]["added_contact_count"] == 1 and joint[0]["added_bounce_count"] == 1
    # An interior contact keeps the existing same-gap refusal exactly.
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 70, kind="interior")],
        "bounce": [bounce_hypothesis("optional_bounces_1", 80)],
    }
    _, receipts = union.composed(
        {**docs, "bounce": {**docs["bounce"], "mode": s6_optional_bounces.MODE_V2}},
        {"events": events},
        rows,
    )
    assert receipts[0]["reason"] == "added_contact_splits_the_gap_that_owns_a_bounce_candidate"


def test_every_original_bounce_candidate_is_validated_against_its_source_inventory():
    docs = documents()
    events = accepted(("contact", 40))
    for frame in (50, 80):
        rows = {
            "contact": [contact_hypothesis("optional_contacts_1", 70)],
            "bounce": [bounce_hypothesis("optional_bounces_1", frame, candidate="other")],
        }
        joint, receipts = union.composed(docs, {"events": events}, rows)
        assert not joint
        assert receipts[0]["reason"] == "bounce_candidate_outside_its_own_source_inventory"
    # A candidate bound to a gap index its own inventory does not carry.
    broken = deepcopy(docs)
    broken["bounce"]["inventory"]["candidates"][0]["gap_index"] = 7
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 70)],
        "bounce": [bounce_hypothesis("optional_bounces_1", 80)],
    }
    _, receipts = union.composed(broken, {"events": events}, rows)
    assert receipts[0]["reason"] == "bounce_candidate_owner_gap_missing_from_its_source_inventory"


def test_a_declared_terminal_net_tail_refuses_only_the_ground_after_the_contact():
    docs = documents()
    events = accepted(("contact", 40))
    attempt = {
        "events": events,
        "terminal_net_tail": {"kind": "terminal_net", "representative": 45},
    }
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 70)],
        "bounce": [bounce_hypothesis("optional_bounces_1", 80)],
    }
    joint, receipts = union.composed(docs, attempt, rows)
    assert not joint
    assert (
        receipts[0]["reason"]
        == "declared_terminal_net_tail_admits_no_ground_after_an_added_contact"
    )
    # The same scope still composes a ground ordered before the added contact.
    rows["bounce"] = [bounce_hypothesis("optional_bounces_1", 50)]
    joint, receipts = union.composed(docs, attempt, rows)
    assert [r["status"] for r in receipts] == ["composed"] and len(joint) == 1


def test_joint_pairs_are_bounded_and_cover_each_final_contact_first():
    docs = documents()
    rows = {
        "contact": [
            contact_hypothesis("optional_contacts_1", 70),
            contact_hypothesis("optional_contacts_2", 75),
        ],
        "bounce": [
            bounce_hypothesis("optional_bounces_1", 50),
            bounce_hypothesis("optional_bounces_2", 55, candidate="b1"),
        ],
    }
    joint, receipts = union.composed(docs, {"events": []}, rows)
    assert len(joint) <= union.MAX_JOINT
    assert [(r["contact_hypothesis"], r["bounce_hypothesis"]) for r in receipts] == [
        ("optional_contacts_1", "optional_bounces_1"),
        ("optional_contacts_2", "optional_bounces_1"),
    ]
    assert union.pair_order(rows) == union.FINAL_PAIR_ORDER
    interior = {**rows, "contact": [contact_hypothesis("optional_contacts_1", 25, "interior")]}
    assert union.pair_order(interior) == union.INTERIOR_PAIR_ORDER


# --- actual preparation of a composed final topology -----------------------


def scene_inputs(events, end_frame=60.0, **extra):
    """A small ordinary attempt/camera pair for the real preparation adapter."""
    rows = [
        {
            "frame": f,
            "status": "visible",
            "x1080": 400.0 + 2 * f,
            "y1080": 400.0 + f,
            "native_pts_seconds": 100.0 + f / 25,
        }
        for f in range(1, int(end_frame) + 1)
    ]
    attempt = {
        "match_id": "match",
        "clip": "pt0001",
        "point_clip": "pt0001",
        "fps": 25,
        "first_event_frame": 1,
        "owner_end_frame": end_frame,
        "events": events,
        "owner_ball_labels": rows,
        **extra,
    }
    camera_rows = [
        {
            "frame": f,
            "status": "supported",
            "P": [[100, 0, 0, 500], [0, 100, 0, 500], [0, 0, 1, 10]],
        }
        for f in range(1, int(end_frame) + 1)
    ]
    return attempt, {"match_id": "match", "clip": "pt0001", "cameras": camera_rows}


def test_a_composed_after_contact_ground_reaches_the_actual_prepared_scene():
    from cv.experiments.connected_shooting.agent_whole_point_search import prepare_attempt

    # Supplied ending: one accepted terminal ground, then the added contact, then the
    # independently source-ranked ground the union composed after it.
    events = accepted(("contact", 1), ("bounce", 20))
    docs = documents()
    rows = {
        "contact": [contact_hypothesis("optional_contacts_1", 30)],
        "bounce": [bounce_hypothesis("optional_bounces_1", 45)],
    }
    joint, receipts = union.composed(docs, {"events": events}, rows)
    assert [r["status"] for r in receipts] == ["composed"]
    attempt, camera_document = scene_inputs(events)
    scene, _held, bounces, _native, _outside = prepare_attempt(
        attempt, camera_document, "hard", joint[0]["events"], attempt["owner_end_frame"]
    )
    assert list(scene.contact_frames) == [1.0, 30.0, 60.0]
    assert [list(map(float, group)) for group in bounces] == [[20.0], [45.0]]
    # Without that composed ground the same supplied-ending scope has no terminal
    # ground at all, so ordinary preparation refuses the contact-only topology.
    contact_only = sorted([*events, *rows["contact"][0]["added"]], key=lambda e: e["frame"])
    with pytest.raises(ValueError, match="terminal bounces required"):
        prepare_attempt(attempt, camera_document, "hard", contact_only, attempt["owner_end_frame"])


def test_an_unresolved_tail_prepares_a_final_contact_with_or_without_a_ground():
    from cv.experiments.connected_shooting.agent_whole_point_search import prepare_attempt

    events = accepted(("contact", 1))
    contract = {
        "kind": "observed_horizon",
        "interval": [1.2, 60.0],
        "observation_horizon": 60.0,
        "native_tail_frames": list(range(2, 61)),
        "supplied_ground_count": 0,
        "horizon_reason": "window_end",
        "physical_ending": None,
        "terminal_ground_count": "unknown",
        "latent_ground_epochs_supplied": False,
    }
    attempt, camera_document = scene_inputs(events, observed_horizon_tail=contract)
    added = contact_hypothesis("optional_contacts_1", 30)["added"]
    scene, _held, bounces, _native, _outside = prepare_attempt(
        attempt, camera_document, "hard", [*events, *added], attempt["owner_end_frame"]
    )
    assert list(scene.contact_frames) == [1.0, 30.0, 60.0]
    assert [list(map(float, group)) for group in bounces] == [[], []]
    assert scene.observed_horizon_tail["supplied_ground_count"] == 0
    # The composed ground after the added contact is a supplied ground of the
    # rebased suffix, which the same unresolved contract recounts.
    composed = sorted(
        [*events, *added, *bounce_hypothesis("optional_bounces_1", 45)["added"]],
        key=lambda e: e["frame"],
    )
    scene, _held, bounces, _native, _outside = prepare_attempt(
        attempt, camera_document, "hard", composed, attempt["owner_end_frame"]
    )
    assert [list(map(float, group)) for group in bounces] == [[], [45.0]]
    assert scene.observed_horizon_tail["supplied_ground_count"] == 1


# --- prepared witness ------------------------------------------------------


def stage_inputs(tmp_path, emissions):
    """A small ordinary automatic input set; no video and no pose are bound."""
    frames = tmp_path / "native" / "pt0001"
    frames.mkdir(parents=True)
    events_path = tmp_path / "automatic_events.json"
    events_path.write_text(json.dumps(emissions))
    for name in ("source.mp4", "players.csv", "source_pts.csv"):
        (tmp_path / name).write_text("clip\n")
    return SimpleNamespace(
        match_id="match",
        clip="pt0001",
        events=events_path,
        source_video=tmp_path / "source.mp4",
        players=tmp_path / "players.csv",
        source_pts=tmp_path / "source_pts.csv",
        frames_directory=frames,
        optional_contact_pose=None,
        optional_contact_views=None,
    )


def stage_attempt(end_frame, horizon=None):
    attempt = {
        "events": accepted(("contact", 10), ("bounce", 20), ("contact", 40), ("bounce", 90)),
        "owner_end_frame": float(end_frame),
    }
    if horizon is not None:
        attempt["observed_horizon_tail"] = {
            "observation_horizon": float(horizon),
            "native_tail_frames": list(range(41, int(horizon))),
        }
    return attempt


def images():
    return [{"frame": f, "native_pts_seconds": PTS[f]} for f in sorted(PTS)]


def cameras():
    return {"cameras": [{"frame": f, "status": "supported"} for f in sorted(PTS)]}


def test_off_witness_is_unchanged_and_declares_no_scope(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(60)])
    attempt = stage_attempt(100)
    off = optional.build_witness(inputs, attempt, images(), cameras())
    assert off["schema"] == optional.SCHEMA and "scope" not in off
    assert off["status"] == "no_source_candidate" and off["gaps"] == [] and off["candidates"] == []
    assert not optional.declared_final_scope(off)
    assert all("kind" not in gap for gap in off["gaps"])
    with pytest.raises(ValueError, match="explicit optional final contact policy"):
        optional.build_witness(inputs, attempt, images(), cameras(), optional_final_contacts="yes")


def test_on_witness_declares_its_scope_ending_and_unavailable_pose(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(60)])
    attempt = stage_attempt(100)
    report = optional.build_witness(
        inputs, attempt, images(), cameras(), optional_final_contacts="on"
    )
    assert report["schema"] == optional.SCHEMA_FINAL
    assert optional.declared_final_scope(report)
    assert report["final_gap"] == {
        "end_boundary": 100.0,
        "ending_semantics": "supplied_attempt_ending",
        "final_gap_built": True,
    }
    assert [gap["kind"] for gap in report["gaps"]] == [optional.FINAL]
    candidate = report["candidates"][0]
    assert candidate["gap_kind"] == optional.FINAL and candidate["event"]["frame"] == 60
    # No local pose is bound, so the modality is unavailable, never negative.
    assert candidate["pose"] == {
        "available": False,
        "supported": False,
        "rows": [],
        "reason": "bound_player_input_has_no_native_skeleton",
    }
    assert candidate["audio"]["supported"] is False
    assert candidate["supported"] is False
    # The original accepted stream and the ordinary receipts are preserved.
    assert report["source_attempt_events"] == attempt["events"]
    assert report["native_timestamps_changed"] is False


def test_observation_scoped_attempt_declares_a_nonphysical_horizon_ending(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(95)])
    attempt = stage_attempt(100)
    supplied = optional.build_witness(
        inputs, attempt, images(), cameras(), optional_final_contacts="on"
    )
    assert supplied["final_gap"]["ending_semantics"] == optional.SUPPLIED_ENDING
    assert not supplied["candidates"]
    assert [r["reason"] for r in supplied["final_gap_exclusions"]] == [
        "accepted_terminal_ground_precedes_candidate"
    ]
    # The same attempt as ordinary automatic preparation actually builds it: the
    # horizon is not a physical ending, so the later candidate is proposed.
    scoped = optional.build_witness(
        inputs,
        {
            **attempt,
            "ending_supplied": False,
            "owner_end_frame_semantics": optional.SCOPE_HORIZON_ENDING,
        },
        images(),
        cameras(),
        optional_final_contacts="on",
    )
    assert scoped["final_gap"]["ending_semantics"] == optional.SCOPE_HORIZON_ENDING
    assert [r["event"]["frame"] for r in scoped["candidates"]] == [95]
    assert not scoped["final_gap_exclusions"]


def test_unresolved_tail_ends_at_the_original_observation_horizon(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(60)])
    attempt = stage_attempt(120, horizon=110)
    report = optional.build_witness(
        inputs, attempt, images(), cameras(), optional_final_contacts="on"
    )
    assert report["final_gap"]["ending_semantics"] == "original_observation_horizon"
    assert report["gaps"][0]["interval"] == [40.0, 110.0]
    # The horizon contract is applied, so a candidate with no contiguous visible
    # suffix is excluded with its own reason rather than proposed.
    attempt["observed_horizon_tail"]["native_tail_frames"] = [41, 42]
    report = optional.build_witness(
        inputs, attempt, images(), cameras(), optional_final_contacts="on"
    )
    assert not report["candidates"]
    assert report["final_gap_exclusions"][0]["reason"] == "no_original_native_suffix_support"
