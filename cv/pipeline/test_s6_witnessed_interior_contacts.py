"""Optional interior contact scope: source-shaped synthetic fixtures only.

No source identity, clip or frame constant of any real point appears here. The
one-ground volley fixture is a small generic accepted stream plus generic held
emission rows; it is a regression shape, never a production selector.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from cv.pipeline import s6_optional_contacts as optional
from cv.pipeline import s6_witnessed_interior_contacts as interior

CLIP = "match__pt0001"
PTS = {i: 100 + i * 0.04 for i in range(1, 200)}
# One native actor box, in the verified native contract, wide enough that a ball
# a few pixels away is inside its reach and one a long way off is not.
NEAR_BOX = (900.0, 400.0, 1000.0, 620.0)


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


def observation(frame, x=950.0, y=500.0, status="visible", support_class=None):
    row = {"frame": frame, "status": status, "x1080": x, "y1080": y}
    if support_class is not None:
        row["support_class"] = support_class
    return row


def ball_rows(frames, **kwargs):
    return [observation(frame, **kwargs) for frame in frames]


def images():
    return [{"frame": f, "native_pts_seconds": PTS[f]} for f in sorted(PTS)]


def cameras(unsupported=()):
    return {
        "cameras": [
            {"frame": f, "status": "unsupported" if f in set(unsupported) else "supported"}
            for f in sorted(PTS)
        ]
    }


def write_players(tmp_path, frames, box=NEAR_BOX):
    path = tmp_path / "players.csv"
    lines = ["clip,frame,side,track_id,x0_native,y0_native,x1_native,y1_native,conf"]
    for frame in frames:
        lines.append("pt0001,%d,near,1,%g,%g,%g,%g,0.9" % (frame, *box))
    path.write_text("\n".join(lines) + "\n")
    (tmp_path / "players.csv.coordinates.json").write_text(
        json.dumps({"artifact_size": {"width": 1920, "height": 1080}})
    )
    return path


def stage_inputs(tmp_path, emissions, *, player_frames=range(1, 200), box=NEAR_BOX):
    frames = tmp_path / "native" / "pt0001"
    frames.mkdir(parents=True, exist_ok=True)
    events_path = tmp_path / "automatic_events.json"
    events_path.write_text(json.dumps(emissions))
    for name in ("source.mp4", "source_pts.csv"):
        (tmp_path / name).write_text("clip\n")
    players = write_players(tmp_path, player_frames, box)
    views = tmp_path / "views.csv"
    if not views.exists():
        views.write_text("t_start,t_end,shot_index\n100,110,1\n")
    return SimpleNamespace(
        match_id="match",
        clip="pt0001",
        events=events_path,
        source_video=tmp_path / "source.mp4",
        players=players,
        source_pts=tmp_path / "source_pts.csv",
        frames_directory=frames,
        optional_contact_pose=None,
        optional_contact_views=views,
    )


def volley_attempt(observations=None):
    """One accepted contact-to-contact gap carrying a single supplied ground.

    The upstream stage already admits and separates that ground, so the existing
    front door -- which requires more than one ground in the gap -- can express
    no alternative here at all.
    """
    return {
        "events": accepted(("contact", 10), ("contact", 40), ("bounce", 50), ("contact", 90)),
        "owner_end_frame": 190.0,
        "owner_ball_labels": (
            observations
            if observations is not None
            else ball_rows([12, 14, 16, 18, 20, 22, 25, 30, 32, 34, 36, 38])
        ),
    }


def build(inputs, attempt, **kwargs):
    return optional.build_witness(inputs, attempt, images(), cameras(), **kwargs)


# --- default off -----------------------------------------------------------


def test_default_off_witness_and_hypotheses_are_exactly_the_old_ones(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    attempt = volley_attempt()
    off = build(inputs, attempt)
    assert "optional_interior_contacts" not in off
    assert interior.policy(off) == "off"
    assert off["status"] == "no_source_candidate"
    assert off["gaps"] == [] and off["candidates"] == []
    assert optional.hypotheses(off, attempt["events"]) == []
    # The kwarg itself is explicit; an unknown value is refused.
    with pytest.raises(ValueError, match="explicit optional interior contact policy"):
        build(inputs, attempt, optional_interior_contacts="yes")
    assert build(inputs, attempt, optional_interior_contacts="off") == off


def test_default_off_keeps_the_existing_mandatory_branch_byte_for_byte(tmp_path):
    # A >1-ground gap: the existing required repair, which this scope never owns.
    events = accepted(
        ("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40), ("contact", 90)
    )
    attempt = {
        "events": events,
        "owner_end_frame": 190.0,
        "owner_ball_labels": ball_rows([12, 14, 16, 18, 22, 32, 34, 36, 38]),
    }
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    off = build(inputs, attempt)
    on = build(inputs, attempt, optional_interior_contacts="on")
    assert [gap["interval"] for gap in off["gaps"]] == [[10.0, 40.0]]
    legacy_off = [row for row in off["candidates"]]
    legacy_on = [row for row in on["candidates"] if row.get("gap_kind") != interior.GAP_KIND]
    assert legacy_on == legacy_off
    # The mandatory gap is not reopened by the optional scope.
    assert [gap["interval"] for gap in interior.scope_gaps(events)] == [[40.0, 90.0]]
    assert optional._legacy_hypotheses(on, events) == optional._legacy_hypotheses(off, events)


# --- the qualification front door -----------------------------------------


def witness(tmp_path, emissions, attempt=None, **kwargs):
    attempt = volley_attempt() if attempt is None else attempt
    inputs = stage_inputs(tmp_path, emissions, **kwargs)
    return optional.build_witness(
        inputs, attempt, images(), cameras(), optional_interior_contacts="on"
    )


def reasons(report):
    return [row["reason"] for row in report["optional_interior_contacts"]["exclusions"]]


def test_one_ground_volley_gap_reaches_the_optional_branch(tmp_path):
    report = witness(tmp_path, [held_contact(25)])
    assert interior.policy(report) == "on"
    assert report["optional_interior_contacts"]["scope"] == interior.SCOPE
    rows = [r for r in report["candidates"] if r["gap_kind"] == interior.GAP_KIND]
    assert [r["event"]["frame"] for r in rows] == [25]
    assert not reasons(report)
    # Both proposed sub-gaps keep their original measured native rows, the one
    # supplied ground lies in exactly one of them, and the ball is at an actor.
    assert [row["supplied_bounce_count"] for row in rows[0]["subgaps"]] == [0, 0]
    assert all(row["observation_count"] >= 4 for row in rows[0]["subgaps"])
    assert rows[0]["actor_witness"]["same_native_frame"] is True
    assert rows[0]["actor_witness"]["distance_scale"] <= interior.MAX_PLAYER_DISTANCE_SCALE
    # No other accepted gap is forced to carry a candidate.
    assert {r["gap_index"] for r in rows} == {0}


def test_zero_ground_and_one_ground_subgaps_are_legal_but_two_are_not(tmp_path):
    # Candidate after the supplied ground: one ground left, none right.
    attempt = volley_attempt(ball_rows([42, 44, 46, 48, 60, 62, 64, 66, 68, 70]))
    attempt["events"] = accepted(("contact", 40), ("bounce", 50), ("contact", 80))
    report = witness(tmp_path, [held_contact(60)], attempt)
    rows = [r for r in report["candidates"] if r["gap_kind"] == interior.GAP_KIND]
    assert [row["supplied_bounce_count"] for row in rows[0]["subgaps"]] == [1, 0]
    # A second ground in the same gap makes it the existing mandatory gap, which
    # this scope never opens, so nothing is proposed there at all.
    attempt["events"] = accepted(("contact", 40), ("bounce", 50), ("bounce", 55), ("contact", 80))
    report = witness(tmp_path, [held_contact(60)], attempt)
    assert not [r for r in report["candidates"] if r["gap_kind"] == interior.GAP_KIND]


def test_no_held_classifier_means_no_candidate(tmp_path):
    accepted_row = {**held_contact(25), "abstain": False, "gate_held": False}
    assert not [
        r
        for r in witness(tmp_path, [accepted_row])["candidates"]
        if r["gap_kind"] == interior.GAP_KIND
    ]
    # Contact probability below none is not a held contact either.
    low = held_contact(25, contact=0.1, none=0.8)
    assert not [
        r for r in witness(tmp_path, [low])["candidates"] if r["gap_kind"] == interior.GAP_KIND
    ]


def test_declared_aftermath_is_never_a_candidate(tmp_path):
    report = witness(tmp_path, [held_contact(25, phase="post_winner_aftermath")])
    assert not [r for r in report["candidates"] if r["gap_kind"] == interior.GAP_KIND]


def test_overlapping_supplied_impact_net_or_bounce_is_refused(tmp_path):
    attempt = volley_attempt()
    attempt["events"] = [*attempt["events"], {"event_type": "net_hit", "frame": 25.0}]
    report = witness(tmp_path, [held_contact(25)], attempt)
    assert reasons(report) == ["candidate_overlaps_a_supplied_event_interval"]
    assert report["optional_interior_contacts"]["exclusions"][0]["failure_class"] == "Q"
    # A candidate on a supplied ground epoch is refused by the same rule.
    attempt = volley_attempt(ball_rows([42, 44, 46, 48, 50, 52, 54, 56, 58]))
    attempt["events"] = accepted(("contact", 40), ("bounce", 50), ("contact", 80))
    report = witness(tmp_path, [held_contact(50)], attempt)
    assert reasons(report) == ["candidate_overlaps_a_supplied_event_interval"]


def test_interpolated_or_sparse_native_rows_abstain(tmp_path):
    sparse = ball_rows([12, 14, 30, 32, 34, 36])
    report = witness(tmp_path, [held_contact(25)], volley_attempt(sparse))
    assert reasons(report) == ["new_subgap_below_the_original_observation_floor"]
    # Enough rows, but the left sub-gap's are interpolated estimates.
    mixed = ball_rows([12, 14, 16, 18], support_class="interpolated_estimate") + ball_rows(
        [30, 32, 34, 36]
    )
    report = witness(tmp_path, [held_contact(25)], volley_attempt(mixed))
    assert reasons(report) == ["new_subgap_below_the_original_observation_floor"]
    excluded = report["optional_interior_contacts"]["observations"]["excluded"]
    assert all(
        row["reason"] == "interpolated_estimate_is_not_an_observation"
        for row in excluded
        if row["frame"] in {12, 14, 16, 18}
    )


def test_unsupported_camera_view_inside_the_candidate_interval_abstains(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    report = optional.build_witness(
        inputs,
        volley_attempt(),
        images(),
        cameras(unsupported=range(24, 28)),
        optional_interior_contacts="on",
    )
    assert reasons(report) == ["unsupported_camera_or_view_in_candidate_interval"]


def test_a_declared_view_cut_across_a_proposed_subgap_abstains(tmp_path):
    views = tmp_path / "views.csv"
    # Two declared source-time shots; their boundary falls inside the gap.
    views.write_text(
        "shot_index,t_start,t_end\n0,100.0,101.0\n1,101.0,110.0\n",
    )
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    inputs.optional_contact_views = views
    report = optional.build_witness(
        inputs, volley_attempt(), images(), cameras(), optional_interior_contacts="on"
    )
    assert reasons(report) == ["native_view_cut_inside_a_proposed_subgap"]


# --- the ball-side witness -------------------------------------------------


def test_ball_not_near_an_actor_is_an_evidential_refusal(tmp_path):
    report = witness(tmp_path, [held_contact(25)], box=(50.0, 50.0, 120.0, 200.0))
    assert reasons(report) == ["observed_ball_not_within_actor_reach"]
    excluded = report["optional_interior_contacts"]["exclusions"][0]
    assert excluded["failure_class"] == "W"
    assert excluded["actor_witness"]["distance_scale"] > interior.MAX_PLAYER_DISTANCE_SCALE


def test_the_actor_box_must_share_the_observed_native_frame(tmp_path):
    # Boxes everywhere except the candidate interval: an occluded or untracked
    # actor abstains with its own reason instead of borrowing a nearby frame.
    frames = [f for f in range(1, 200) if not 24 <= f <= 26]
    report = witness(tmp_path, [held_contact(25)], player_frames=frames)
    assert reasons(report) == ["no_actor_box_on_the_observed_native_frame"]


def test_no_measured_ball_row_inside_the_candidate_interval_abstains(tmp_path):
    attempt = volley_attempt(ball_rows([12, 14, 16, 18, 30, 32, 34, 36]))
    report = witness(tmp_path, [held_contact(25)], attempt)
    assert reasons(report) == ["no_original_observation_inside_candidate_interval"]
    assert report["optional_interior_contacts"]["exclusions"][0]["failure_class"] == "W"


def test_missing_actor_coordinate_contract_abstains(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    (tmp_path / "players.csv.coordinates.json").unlink()
    report = optional.build_witness(
        inputs, volley_attempt(), images(), cameras(), optional_interior_contacts="on"
    )
    assert reasons(report) == ["no_actor_box_on_the_observed_native_frame"]
    assert report["optional_interior_contacts"]["actor_boxes"]["available"] is False


def test_audio_or_pose_support_is_still_required_for_a_hypothesis(tmp_path):
    report = witness(tmp_path, [held_contact(25)])
    row = [r for r in report["candidates"] if r["gap_kind"] == interior.GAP_KIND][0]
    # No audio and no bound skeleton: the candidate qualifies but is unsupported,
    # so it never reaches the alternatives.
    assert row["supported"] is False
    assert optional.hypotheses(report, report["source_attempt_events"]) == []


# --- bounded alternatives --------------------------------------------------


def supported_witness(tmp_path, emissions, attempt=None):
    report = witness(tmp_path, emissions, attempt)
    for row in report["candidates"]:
        if row["gap_kind"] == interior.GAP_KIND:
            row["supported"] = True
    return report


def test_two_candidates_become_two_singleton_alternatives_never_a_pair(tmp_path):
    attempt = volley_attempt(ball_rows([12, 14, 16, 18, 20, 22, 28, 30, 32, 34, 36, 38]))
    report = supported_witness(tmp_path, [held_contact(20), held_contact(30)], attempt)
    result = optional.hypotheses(report, attempt["events"])
    assert [h["name"] for h in result] == [
        "optional_contacts_interior_1",
        "optional_contacts_interior_2",
    ]
    assert all(len(h["added"]) == 1 for h in result)
    assert [h["candidate_ids"] for h in result] == [["emission_0"], ["emission_1"]]
    assert all(h["events"] != attempt["events"] for h in result)
    # The accepted stream itself is unchanged in every alternative.
    for h in result:
        assert [e for e in h["events"] if "optional_gap_kind" not in e] == attempt["events"]
        assert h["added"][0]["optional_gap_kind"] == interior.GAP_KIND
        assert h["witnessed_interior_contacts"]["singletons_only"] is True


def test_a_third_candidate_is_cap_excluded_with_its_receipt(tmp_path):
    attempt = volley_attempt(ball_rows([12, 14, 16, 18, 20, 22, 25, 28, 30, 32, 34, 36, 38]))
    report = supported_witness(
        tmp_path, [held_contact(20), held_contact(25), held_contact(30)], attempt
    )
    result = optional.hypotheses(report, attempt["events"])
    assert len(result) == interior.MAX_TOTAL_ALTERNATIVES
    receipt = result[0]["witnessed_interior_contacts"]
    assert len(receipt["cap_excluded"]) == 1
    assert receipt["cap_excluded"][0]["reason"] == "fixed_total_alternative_cap_reached"
    assert receipt["existing_branches_evicted"] is False


def test_existing_branches_are_retained_and_bound_the_new_ones(tmp_path):
    legacy = [
        {
            "name": "optional_contacts_1",
            "events": deepcopy(accepted(("contact", 10), ("contact", 40))),
            "added": [],
            "candidate_ids": ["legacy_a"],
            "occurrence_log_odds": 0.5,
        }
    ]
    attempt = volley_attempt()
    report = supported_witness(tmp_path, [held_contact(25)], attempt)
    result = interior.extend_hypotheses(report, attempt["events"], deepcopy(legacy))
    # One existing branch leaves room for exactly one new alternative, built on
    # that branch rather than replacing it.
    assert [h["name"] for h in result] == ["optional_contacts_1", "optional_contacts_interior_1"]
    assert result[0] == legacy[0]
    assert result[1]["candidate_ids"] == ["legacy_a", "emission_0"]
    assert result[1]["witnessed_interior_contacts"]["base_alternative"] == "optional_contacts_1"
    assert result[1]["occurrence_log_odds"] > 0.5
    # Two existing branches leave no room at all.
    two = [*deepcopy(legacy), {**deepcopy(legacy[0]), "name": "optional_contacts_2"}]
    assert interior.extend_hypotheses(report, attempt["events"], two) == two


def test_off_policy_never_extends_even_with_prepared_candidates(tmp_path):
    attempt = volley_attempt()
    report = supported_witness(tmp_path, [held_contact(25)], attempt)
    report.pop("optional_interior_contacts")
    assert interior.extend_hypotheses(report, attempt["events"], []) == []
    with pytest.raises(ValueError, match="unknown optional interior contact scope"):
        interior.policy({"optional_interior_contacts": {"policy": "maybe"}})


def test_added_events_carry_no_predicted_geometry(tmp_path):
    attempt = volley_attempt()
    report = supported_witness(tmp_path, [held_contact(25)], attempt)
    added = optional.hypotheses(report, attempt["events"])[0]["added"][0]
    assert not {"x", "y", "z", "x1080", "y1080", "position_m"} & set(added)
    assert added["optional_topology_membership"]["accepted_source_stream_changed"] is False


# --- shared settings and adapter forwarding --------------------------------


def test_shared_settings_declare_the_scope_and_require_optional_contacts():
    from cv.pipeline import s6_labeled_stage

    assert s6_labeled_stage.shared_settings({})["optional_interior_contacts"] == "off"
    assert (
        s6_labeled_stage.shared_settings({"optional_interior_contacts": "on"})[
            "optional_interior_contacts"
        ]
        == "on"
    )
    with pytest.raises(ValueError, match="unsupported global optional_interior_contacts"):
        s6_labeled_stage.shared_settings({"optional_interior_contacts": "yes"})


def test_preparation_adapter_requires_the_explicit_policy(tmp_path):
    from cv.pipeline import s6_automatic_observations

    inputs = SimpleNamespace(optional_contact_pose=None)
    with pytest.raises(ValueError, match="explicit optional interior contact policy"):
        s6_automatic_observations.build_observations(
            inputs, tmp_path / "out", optional_interior_contacts="yes"
        )
    with pytest.raises(ValueError, match="optional interior contacts require"):
        s6_automatic_observations.build_observations(
            inputs, tmp_path / "out", optional_interior_contacts="on"
        )


def test_selection_scope_must_match_its_source_witness(tmp_path):
    report = supported_witness(tmp_path, [held_contact(25)])
    assert interior.policy(report) == "on"
    report.pop("optional_interior_contacts")
    assert interior.policy(report) == "off"


@pytest.mark.parametrize(
    "bad",
    [
        observation(25, x=1, support_class="interpolated_estimate"),
        observation(25, x=1, status="occluded"),
        observation(25, x=float("nan")),
        observation(25.8, x=1),
        observation("25", x=1),
        observation(True, x=1),
    ],
)
def test_only_the_actual_admitted_duplicate_row_supplies_native_pixels(bad):
    points, _ = interior.observed_points([bad, observation(25)], set(PTS))
    assert points == {25: (950.0, 500.0)}


def test_conflicting_measured_duplicates_cannot_supply_actor_reach():
    points, receipt = interior.observed_points([observation(25, x=1), observation(25)], set(PTS))
    assert points == {}
    assert any(r["reason"] == "conflicting_duplicate_native_frame" for r in receipt["excluded"])


def test_missing_view_or_missing_camera_inventory_abstains(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    inputs.optional_contact_views = None
    report = build(inputs, volley_attempt(), optional_interior_contacts="on")
    assert not report["candidates"]
    inputs.optional_contact_views = tmp_path / "views.csv"
    report = optional.build_witness(
        inputs, volley_attempt(), images(), {"cameras": []}, optional_interior_contacts="on"
    )
    assert not report["candidates"]


def test_missing_camera_in_wing_does_not_establish_continuity(tmp_path):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    camera = cameras()
    camera["cameras"] = [row for row in camera["cameras"] if row["frame"] != 19]
    report = optional.build_witness(
        inputs, volley_attempt(), images(), camera, optional_interior_contacts="on"
    )
    assert reasons(report) == ["unsupported_camera_or_view_inside_a_proposed_subgap"]


@pytest.mark.parametrize("confidence", ["", "0", "-1", "nan", "inf"])
def test_actor_rows_require_real_positive_confidence(tmp_path, confidence):
    inputs = stage_inputs(tmp_path, [held_contact(25)], player_frames=[25])
    inputs.players.write_text(inputs.players.read_text().replace(",0.9\n", f",{confidence}\n"))
    report = build(inputs, volley_attempt(), optional_interior_contacts="on")
    assert reasons(report) == ["no_actor_box_on_the_observed_native_frame"]
    assert report["optional_interior_contacts"]["actor_boxes"]["invalid_or_unconfident_rows"] == 1


def test_candidate_free_new_gaps_do_not_expand_audio_decoding(tmp_path, monkeypatch):
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    calls = []

    def unavailable(video, low, high):
        calls.append((low, high))
        raise OSError("fixture audio unavailable")

    monkeypatch.setattr(optional.witness, "decode_clocked_audio", unavailable)
    report = build(inputs, volley_attempt(), optional_interior_contacts="on")
    assert len(report["gaps"]) == 2
    assert calls == [(PTS[10] - 0.25, PTS[40] + 0.25)]
    assert report["optional_interior_contacts"]["expansion"]["supported_candidate_ids"] == []


def test_cap_receipt_survives_no_new_branch_and_does_not_mutate_evidence(tmp_path):
    attempt = volley_attempt()
    report = supported_witness(tmp_path, [held_contact(25)], attempt)
    before = deepcopy(report)
    legacy = [{"name": "optional_contacts_1"}, {"name": "optional_contacts_2"}]
    receipt = interior.expansion_receipt(report, legacy)
    assert receipt["added_alternatives_allowed"] == 0
    assert receipt["cap_excluded"] == [
        {"id": "emission_0", "reason": "fixed_total_alternative_cap_reached"}
    ]
    assert interior.extend_hypotheses(report, attempt["events"], legacy) == legacy
    assert report == before


def test_union_selection_identifies_and_binds_the_new_contact_family(tmp_path):
    from cv.pipeline import s6_optional_event_union as union

    attempt = volley_attempt()
    report = supported_witness(tmp_path, [held_contact(25)], attempt)
    alternative = optional.hypotheses(report, attempt["events"])[0]
    records = {"contact": {"sha256": "contact_receipt"}, "bounce": {"sha256": "bounce_receipt"}}
    selected = union.selection_fields(alternative["name"], alternative, [], {}, records)
    assert selected["source_family"] == "contact"
    assert selected["witness_record"] == records["contact"]
    assert selected["added_contact_count"] == 1
    assert selected["added_bounce_count"] == 0


def source_audio(monkeypatch, frames):
    calls = []

    def decode(video, low, high):
        calls.append((low, high))
        return SimpleNamespace(low=low, high=high, receipt={"command": ["fixture"]})

    def peaks(audio):
        return [
            {"source_pts_seconds": PTS[f] + 0.03, "support": 0.8}
            for f in frames
            if audio.low <= PTS[f] <= audio.high
        ]

    monkeypatch.setattr(optional.witness, "decode_clocked_audio", decode)
    monkeypatch.setattr(optional.witness, "audio_peaks", peaks)
    return calls


def test_mixed_required_and_optional_gaps_keep_legacy_composition_and_replay(tmp_path, monkeypatch):
    source_audio(monkeypatch, [25, 60])
    inputs = stage_inputs(tmp_path, [held_contact(25), held_contact(60)])
    attempt = volley_attempt(ball_rows(sorted(set([*range(12, 90, 2), 25]))))
    attempt["events"] = accepted(
        ("contact", 10), ("bounce", 20), ("bounce", 30), ("contact", 40), ("contact", 90)
    )
    kwargs = {
        "optional_contact_composition": "bounded_pairs",
        "optional_contact_timing": "pmf_peaks",
    }
    off = build(inputs, attempt, **kwargs)
    on = build(inputs, attempt, optional_interior_contacts="on", **kwargs)
    legacy = optional.hypotheses(off, attempt["events"], observations=attempt["owner_ball_labels"])
    replay = optional.search_hypotheses(on, attempt)
    assert len(legacy) == 1 and len(replay) == 2
    assert replay[0] == legacy[0]
    assert replay[1]["candidate_ids"] == ["emission_0", "emission_1"]
    assert on["candidates"][0]["contact_timing"]["status"] == "unavailable_in_source_emission"
    assert "contact_timing" not in on["candidates"][1]
    stored = on["optional_interior_contacts"]["expansion"]
    assert stored == interior.expansion_receipt(on, legacy)
    assert all(replay[1]["witnessed_interior_contacts"][k] == v for k, v in stored.items())
    assert replay[1]["events"] == sorted(
        [*attempt["events"], *replay[1]["added"]], key=lambda e: e["frame"]
    )


def test_audio_support_is_source_timed_after_an_empty_first_gap(tmp_path, monkeypatch):
    calls = source_audio(monkeypatch, [60])
    inputs = stage_inputs(tmp_path, [held_contact(60)])
    attempt = volley_attempt(ball_rows(range(42, 80, 2)))
    report = build(inputs, attempt, optional_interior_contacts="on")
    row = report["candidates"][0]
    assert row["gap_index"] == 1 and row["audio"]["supported"]
    assert row["audio"]["transient"]["source_pts_seconds"] == PTS[60] + 0.03
    assert calls == [(PTS[40] - 0.25, PTS[90] + 0.25)]
    result = optional.search_hypotheses(report, attempt)
    assert len(result) == 1
    assert report["optional_interior_contacts"]["expansion"]["supported_candidate_ids"] == [
        row["id"]
    ]


def test_actor_near_boundary_distance_is_native_pixels_exactly():
    receipt, reason = interior._actor_witness({25: (1100.0, 500.0)}, {25: [list(NEAR_BOX)]}, 25, 25)
    assert not reason
    assert receipt["distance_px"] == 100.0
    assert receipt["actor_box_scale_px"] == pytest.approx((100**2 + 220**2) ** 0.5)


def test_mismatched_clip_is_reported_as_missing_membership(tmp_path):
    path = write_players(tmp_path, [25])
    path.write_text(path.read_text().replace("pt0001,", "other__pt0001,"))
    boxes, receipt = interior.actor_boxes(path, "pt0001")
    assert not boxes and receipt["reason"] == "no_matching_clip_rows"
    assert receipt["total_rows"] == 1 and receipt["rows"] == 0


def test_gap_scope_partitions_the_shared_legacy_rule():
    import random

    rng = random.Random(31)
    for _ in range(40):
        events = accepted(*[("contact", f) for f in (10, 40, 70, 100)])
        for low in (10, 40, 70):
            events.extend(
                accepted(*[("bounce", low + f) for f in rng.sample(range(2, 28), rng.randrange(4))])
            )
        old = {tuple(g["interval"]) for g in optional.inconsistent_gaps(events)}
        new = {tuple(g["interval"]) for g in interior.scope_gaps(events)}
        assert not old & new
        assert old | new == {(10.0, 40.0), (40.0, 70.0), (70.0, 100.0)}


def test_empty_candidate_expansion_is_explicit(tmp_path):
    report = witness(tmp_path, [])
    assert report["optional_interior_contacts"]["expansion"]["supported_candidate_ids"] == []
    assert report["optional_interior_contacts"]["expansion"]["cap_excluded"] == []


def test_union_replays_persisted_interior_contact_without_a_bounce_alternative(
    tmp_path, monkeypatch
):
    import hashlib
    from cv.pipeline import s6_optional_bounce_scope as bounce_scope
    from cv.pipeline import s6_optional_event_union as union

    source_audio(monkeypatch, [25])
    inputs = stage_inputs(tmp_path, [held_contact(25)])
    attempt = volley_attempt()
    document = build(inputs, attempt, optional_interior_contacts="on")
    path = tmp_path / "contact_witness.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))
    contact_record = {
        "path": path.name,
        "path_base": "TENNIS_DATA_ROOT",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    bounce = {
        "schema": bounce_scope.WITNESS_SCHEMA,
        "mode": "classifier",
        "hypotheses": [],
        "null_hold": None,
    }
    # Known-ending bounce arm has no conditional branch. Its loader is already
    # tested independently; exercise the real contact/union search and replay.
    monkeypatch.setattr(bounce_scope, "load_witness", lambda _: bounce)
    attempt["optional_bounce_witness"] = {"path": "separate_bound_bounce"}
    rows, receipts = union.hypotheses({"contact": document, "bounce": bounce}, attempt)
    assert len(rows) == 1 and rows[0]["source_family"] == "contact"
    selected = union.selection_fields(
        rows[0]["name"],
        rows[0],
        receipts,
        {},
        {"contact": contact_record, "bounce": attempt["optional_bounce_witness"]},
    )
    selected.update(selected_topology=rows[0]["name"], optional_interior_contacts="on")
    report = {"events": rows[0]["events"], "optional_contact_selection": selected}
    assert union.validate_report_events(attempt, report)
    selected["optional_interior_contacts"] = "off"
    with pytest.raises(ValueError, match="interior contact selection differs"):
        union.validate_report_events(attempt, report)


def test_malformed_declared_actor_coordinates_fail_the_source_contract(tmp_path):
    path = write_players(tmp_path, [25])
    path.with_suffix(".csv.coordinates.json").write_text(
        json.dumps({"artifact_size": {"width": 0, "height": 1080}})
    )
    with pytest.raises(ValueError):
        interior.actor_boxes(path, "pt0001")


def test_new_gap_audio_padding_cannot_duplicate_a_legacy_contact_transient(tmp_path, monkeypatch):
    source_audio(monkeypatch, [34, 60])
    inputs = stage_inputs(tmp_path, [held_contact(34), held_contact(60)])
    attempt = volley_attempt(ball_rows(range(12, 90, 2)))
    attempt["events"] = accepted(
        ("contact", 10), ("bounce", 20), ("bounce", 37), ("contact", 40), ("contact", 90)
    )
    off = build(inputs, attempt)
    on = build(inputs, attempt, optional_interior_contacts="on")
    assert off["candidates"][0]["audio"]["supported"]
    # The new [40,90] gap's0.25s padding also contains the old f34 transient.
    # It must not change the legacy unique-transient rule or confidence.
    assert on["candidates"][0] == off["candidates"][0]
    assert on["candidates"][1]["audio"]["supported"]


def test_active_pairs_and_pmf_preserve_the_full_legacy_beam(tmp_path, monkeypatch):
    from cv.pipeline import s6_contact_timing as timing

    source_audio(monkeypatch, [34, 60, 110, 130])
    inputs = stage_inputs(tmp_path, [held_contact(f) for f in (34, 60, 110, 130)])
    attempt = volley_attempt(ball_rows(range(12, 150, 2)))
    attempt["events"] = accepted(
        ("contact", 10), ("bounce", 20), ("bounce", 80), ("contact", 90), ("contact", 150)
    )

    def prepared_peak(candidate, *_args):
        event = deepcopy(candidate["event"])
        event["frame"] += 0.25
        event["frame_interval"] = [event["frame"] - 1, event["frame"] + 1]
        return {"alternatives": [{"event": event, "peak": {"mass": 0.7}}]}

    monkeypatch.setattr(timing, "prepare", prepared_peak)
    kwargs = {
        "optional_contact_composition": "bounded_pairs",
        "optional_contact_timing": "pmf_peaks",
    }
    off = build(inputs, attempt, **kwargs)
    on = build(inputs, attempt, optional_interior_contacts="on", **kwargs)
    old = optional.search_hypotheses(off, attempt)
    new_legacy = optional._legacy_hypotheses(
        on, attempt["events"], observations=attempt["owner_ball_labels"]
    )
    assert new_legacy == old
    assert any(len(h["added"]) == 2 for h in old)
    assert any("contact_timing_alternative" in h for h in old)
    assert len(old) > 2
    # Actual active PMF/pair branches already fill the unchanged beam. Both
    # new supported candidates are explicitly capped; neither evicts a branch.
    assert optional.search_hypotheses(on, attempt) == old
    receipt = on["optional_interior_contacts"]["expansion"]
    assert receipt["added_alternatives_allowed"] == 0
    assert len(receipt["supported_candidate_ids"]) == len(receipt["cap_excluded"]) == 2


def test_final_and_interior_scopes_keep_legacy_audio_and_gap_offsets(tmp_path, monkeypatch):
    source_audio(monkeypatch, [34, 60, 130])
    inputs = stage_inputs(tmp_path, [held_contact(f) for f in (34, 60, 130)])
    attempt = volley_attempt(ball_rows(range(12, 190, 2)))
    attempt["events"] = accepted(
        ("contact", 10),
        ("bounce", 20),
        ("bounce", 37),
        ("contact", 40),
        ("contact", 90),
        ("bounce", 100),
        ("bounce", 180),
    )
    off = build(inputs, attempt, optional_final_contacts="on")
    on = build(inputs, attempt, optional_final_contacts="on", optional_interior_contacts="on")
    assert optional.INTERIOR != optional.INTERIOR_OPTIONAL != optional.FINAL
    assert [g["kind"] for g in on["gaps"]] == [optional.INTERIOR, optional.FINAL, interior.GAP_KIND]
    assert on["gaps"][2]["interval"] == [40.0, 90.0]
    assert on["gaps"][1]["interval"] == [90.0, 190.0]
    old_candidates = [r for r in on["candidates"] if r.get("gap_kind") != interior.GAP_KIND]
    assert old_candidates == off["candidates"]
    new = next(r for r in on["candidates"] if r.get("gap_kind") == interior.GAP_KIND)
    assert new["gap_index"] == 2 and new["supported"]
    assert on["audio_clocks"][-1]["gap_index"] == 2
    assert on["audio_clocks"][-1]["gap_kind"] == interior.GAP_KIND
    assert optional._legacy_hypotheses(on, attempt["events"]) == optional.hypotheses(
        off, attempt["events"]
    )


def test_empty_receipt_matches_actual_replay_and_native_actor_loader(tmp_path):
    report = witness(tmp_path, [])
    legacy = optional._legacy_hypotheses(report, report["source_attempt_events"])
    assert legacy == []
    assert report["optional_interior_contacts"]["expansion"] == interior.expansion_receipt(
        report, legacy
    )
    boxes, _ = interior.actor_boxes(tmp_path / "players.csv", "pt0001")
    receipt, reason = interior._actor_witness({25: (1100.0, 500.0)}, boxes, 25, 25)
    assert not reason and receipt["distance_px"] == 100.0
