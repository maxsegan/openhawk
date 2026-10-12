import pytest

from cv.pipeline.point_grammar import (
    LOCATION_MAX_OFFSET_FRAMES,
    AutomaticEvidence,
    ChainMembership,
    annotate_abstentions,
    chain_memberships,
    point_end_emissions,
    pre_serve_double_bounce_guard,
    pre_serve_missing_serve_guard,
    pre_serve_transfer_guard,
    select_consumer_emissions,
    toss_ascent_signature,
)


def _annotated(frame: float, event_type: str, chain_id: str | None) -> dict:
    return {
        "clip": "m__pt0001",
        "match_id": "m",
        "event_type": event_type,
        "frame": frame,
        "abstain": False,
        "location": {
            "image_x": 100.0,
            "image_y": 50.0,
            "court_x_fraction": 0.5,
            "court_y_fraction": 0.5,
        },
        "point_grammar": {
            "in_play": True,
            "confidence": 0.9,
            "structure_span_id": "m__pt0001::span_00",
            "chain_evidence": {"member": chain_id is not None, "chain_id": chain_id},
        },
    }


def test_chain_membership_requires_contact_and_terminal() -> None:
    rows = [
        {"frame": 10.0, "event_type": "contact"},
        {"frame": 25.0, "event_type": "bounce"},
        {"frame": 39.0, "event_type": "contact"},
        {"frame": 200.0, "event_type": "bounce"},
    ]
    memberships = chain_memberships(rows, 25.0)
    assert [row.member for row in memberships] == [True, True, True, False]
    assert memberships[0].chain_length == 3
    assert memberships[-1].distance_reference_frames == 161.0


def test_toss_signature_uses_motion_evidence() -> None:
    weak = {
        "pre_400ms_observations": "10",
        "pre_400ms_displacement_px": "26",
        "pre_400ms_straightness": "0.60",
        "post_400ms_displacement_px": "140",
    }
    assert not toss_ascent_signature(weak)
    assert toss_ascent_signature({**weak, "post_400ms_displacement_px": "220"})


def test_consumer_defaults_to_in_play_and_has_lossless_bypass() -> None:
    rows = [
        {"frame": 1, "point_grammar": {"in_play": True}},
        {"frame": 2, "point_grammar": {"in_play": False}},
    ]
    assert select_consumer_emissions(rows) == [rows[0]]
    assert select_consumer_emissions(rows, include_dead_time=True) == rows


def test_consumer_fails_closed_when_metadata_is_missing() -> None:
    with pytest.raises(ValueError, match="metadata missing"):
        select_consumer_emissions([{"frame": 1}])


def test_pre_serve_transfer_guard_requires_all_three_live_witnesses() -> None:
    membership = ChainMembership("chain_01", 2, True, 0.0)
    track = {"continuous_through_emission": True}
    reasons = ["s3_phase_dead_plus_stationary_players"]
    assert pre_serve_transfer_guard({"excluded": "pre_serve"}, membership, track, True, reasons)
    assert not pre_serve_transfer_guard(
        {"excluded": "post_terminal"}, membership, track, True, reasons
    )
    assert not pre_serve_transfer_guard(
        {"excluded": "pre_serve"}, membership, track, False, reasons
    )


def test_double_bounce_guard_requires_continuous_toss_evidence() -> None:
    reasons = ["pre_serve_unbridged_second_bounce"]
    assert pre_serve_double_bounce_guard({"continuous_through_emission": True}, True, reasons)
    assert not pre_serve_double_bounce_guard({"continuous_through_emission": False}, True, reasons)


def test_missing_serve_guard_requires_coherent_pre_serve_chain() -> None:
    membership = ChainMembership("chain_01", 2, True, 0.0)
    track = {"continuous_through_emission": True}
    reasons = ["s3_phase_dead_plus_stationary_players"]
    assert pre_serve_missing_serve_guard({"excluded": "pre_serve"}, membership, track, [], reasons)
    assert not pre_serve_missing_serve_guard(
        {"excluded": "pre_serve"}, membership, track, [10.0], reasons
    )


def test_consumer_view_never_returns_abstentions() -> None:
    rows = [
        {"frame": 1, "abstain": False, "point_grammar": {"in_play": True}},
        {"frame": 2, "abstain": True, "point_grammar": {"in_play": False}},
    ]
    assert select_consumer_emissions(rows) == [rows[0]]
    assert select_consumer_emissions(rows, include_dead_time=True) == [rows[0]]
    assert select_consumer_emissions(rows, include_abstained=True, include_dead_time=True) == rows


def test_point_end_is_emitted_once_per_terminated_chain() -> None:
    rows = [
        _annotated(10.0, "contact", "chain_01"),
        _annotated(25.0, "bounce", "chain_01"),
        _annotated(35.0, "bounce", "chain_01"),
        _annotated(200.0, "contact", "chain_02"),
        _annotated(215.0, "bounce", "chain_02"),
        _annotated(225.0, "bounce", "chain_02"),
        _annotated(400.0, "bounce", None),
    ]
    for row in (rows[2], rows[5]):
        row["terminal_evidence"] = {
            "termination_kind": "second_bounce",
            "preceding_bounce_candidate_frame": row["frame"] - 10.0,
        }
    endings = point_end_emissions(rows)
    assert [(row["frame"], row["point_end"]["chain_id"]) for row in endings] == [
        (35.0, "chain_01"),
        (225.0, "chain_02"),
    ]
    assert endings[0]["event_type"] == "point_end"
    assert endings[0]["point_end"]["terminal_event_type"] == "bounce"
    assert endings[0]["location"]["image_x"] == 100.0
    assert endings[0]["location"]["court_x_fraction"] == 0.5


def test_point_end_ignores_abstentions_and_dead_time() -> None:
    abstained = _annotated(30.0, "bounce", "chain_01")
    abstained["abstain"] = True
    dead = _annotated(31.0, "bounce", "chain_01")
    dead["point_grammar"]["in_play"] = False
    rows = [_annotated(10.0, "contact", "chain_01"), abstained, dead]
    assert point_end_emissions(rows) == []


def test_abstentions_are_located_but_not_graded_by_the_grammar(tmp_path) -> None:
    rows = [{"clip": "m__pt0001", "match_id": "m", "event_type": "contact", "frame": 12.0}]
    proposals = [
        {
            "clip": "m__pt0001",
            "proposal_frame": 12.5,
            "img_x": 480.0,
            "img_y": 270.0,
            "court_x": 3.0,
            "court_y": 8.0,
            "source_fps": 25.0,
        }
    ]
    annotated = annotate_abstentions(rows, tmp_path, proposals, {})
    location = annotated[0]["location"]
    assert annotated[0]["abstain"] is True
    assert annotated[0]["point_grammar"]["grammar_evaluated"] is False
    assert select_consumer_emissions(annotated) == []
    # No coordinate sidecar under tmp_path, so coordinates stay in artifact space.
    assert location["image_x"] == 480.0
    assert location["frame_subpixel"] == 12.5
    assert location["time_seconds"] == pytest.approx(0.5)


def test_location_is_dropped_when_no_proposal_is_close_enough(tmp_path) -> None:
    proposals = [{"clip": "m__pt0001", "proposal_frame": 100.0, "img_x": 1.0, "img_y": 2.0}]
    evidence = AutomaticEvidence(tmp_path, proposals, {})
    far = evidence.location("m__pt0001", 12.0, proposals[0], 25.0)
    near = evidence.location("m__pt0001", 100.0 - LOCATION_MAX_OFFSET_FRAMES, proposals[0], 25.0)
    assert far["image_x"] is None
    assert far["source"].startswith("no_proposal_within")
    assert near["image_x"] == 1.0
