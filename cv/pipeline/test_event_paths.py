from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline.event_grammar_decoder import GrammarConfig, TRANSITION_LOG_PRIOR
from cv.pipeline.event_paths import (
    _grammar_document,
    _load_grammar,
    apply_ending_context,
    build_packets,
    diverse_paths,
    packet_ending_selection,
    tentative_events,
)
from cv.pipeline import point_context


def _rows(values: list[dict[str, float]]) -> np.ndarray:
    output = np.full((len(values), 4), 0.001, dtype=np.float64)
    for row, values_by_kind in enumerate(values):
        for kind, value in values_by_kind.items():
            output[row, ("none", "contact", "bounce", "net_hit").index(kind)] = value
        output[row, 0] = max(0.001, 1.0 - sum(values_by_kind.values()))
    return output


def test_diverse_paths_does_not_spend_slots_on_timing_shifts() -> None:
    probabilities = _rows(
        [
            {"contact": 0.99},
            {"bounce": 0.80},
            {"bounce": 0.79},
            {"contact": 0.99},
            {"net_hit": 0.90},
        ]
    )
    frames = np.asarray([10, 20, 21, 30, 40])
    data = {
        "probabilities": probabilities,
        "frames": frames,
        "court_y": np.asarray([0.2, 0.8, 0.8, 0.8, 0.5]),
        "time_offset": np.zeros(5),
    }

    paths, _nodes, _marginals = diverse_paths(
        probabilities,
        frames,
        data["court_y"],
        np.arange(5),
        "match__pt0001",
        data,
        config=GrammarConfig(local_maximum_radius=0),
        count=2,
        timing_tolerance=1.0,
    )

    assert len(paths) == 2
    assert [event["event_type"] for event in paths[0]["events"]] == [
        "contact",
        "bounce",
        "contact",
        "net_hit",
    ]
    assert [event["event_type"] for event in paths[1]["events"]] != [
        "contact",
        "bounce",
        "contact",
        "net_hit",
    ]
    assert paths[0]["log_score_gap"] == 0.0


def test_relaxed_contact_transition_carries_a_different_event_set() -> None:
    probabilities = _rows([{"contact": 0.95}, {"contact": 0.60}, {"bounce": 0.80}])
    frames = np.asarray([10, 20, 30])
    data = {
        "probabilities": probabilities,
        "frames": frames,
        "court_y": np.asarray([0.2, 0.8, 0.2]),
        "time_offset": np.zeros(3),
    }
    relaxed = GrammarConfig(
        transition_log_prior={**TRANSITION_LOG_PRIOR, ("contact", "contact"): 0.0},
        side_violation_penalty=0.0,
    )

    paths, *_ = diverse_paths(
        probabilities,
        frames,
        data["court_y"],
        np.arange(3),
        "match__pt0001",
        data,
        config=relaxed,
        count=2,
    )

    assert any(
        [event["event_type"] for event in path["events"]] == ["contact", "contact", "bounce"]
        for path in paths
    )


def test_tentative_rows_relax_local_maximum_without_changing_firm_nodes() -> None:
    probabilities = _rows([{"contact": 0.8}, {"contact": 0.6}])
    frames = np.asarray([10, 11])
    data = {
        "probabilities": probabilities,
        "frames": frames,
        "court_y": np.asarray([0.2, 0.2]),
        "time_offset": np.zeros(2),
    }
    paths, nodes, marginals = diverse_paths(
        probabilities,
        frames,
        data["court_y"],
        np.arange(2),
        "match__pt0001",
        data,
        count=1,
    )

    tentative = tentative_events(
        "match__pt0001",
        paths,
        nodes,
        marginals,
        np.arange(2),
        data,
        [],
        marginal_floor=0.3,
        probability_floor=0.05,
        timing_tolerance=1.0,
    )

    assert [(node.frame, node.event_type) for node in nodes] == [(10, "contact")]
    assert any(
        row["frame"] == 11
        and row["event_type"] == "contact"
        and row["source"] == "classifier_probability"
        for row in tentative
    )


def test_tentative_keeps_native_row_when_firm_time_head_moves_it() -> None:
    probabilities = _rows([{"contact": 0.8}])
    frames = np.asarray([10])
    data = {
        "probabilities": probabilities,
        "frames": frames,
        "court_y": np.asarray([0.2]),
        "time_offset": np.asarray([3.0]),
    }
    paths, nodes, marginals = diverse_paths(
        probabilities,
        frames,
        data["court_y"],
        np.arange(1),
        "match__pt0001",
        data,
        count=1,
    )

    tentative = tentative_events(
        "match__pt0001",
        paths,
        nodes,
        marginals,
        np.arange(1),
        data,
        [],
        marginal_floor=0.3,
        probability_floor=0.05,
        timing_tolerance=1.0,
    )

    assert paths[0]["events"][0]["frame"] == 13
    assert any(
        row["frame"] == 10
        and row["event_type"] == "contact"
        and row["source"] == "classifier_probability"
        for row in tentative
    )


def test_point_context_softly_selects_packet_ending_without_changing_path_score() -> None:
    context = {
        "schema": point_context.SCHEMA,
        "automatic": True,
        "match_id": "match",
        "point_id": "pt0001",
        "serve_ordinal": {"first": 0.25, "second": 0.25, "fault": 0.25, "let": 0.25},
        "ending_kind": {"out": 0.8, "second_bounce": 0.1, "net": 0.05, "fov_exit": 0.05},
        "winner": "receiver",
        "winner_likelihoods": {"server": 0.025, "receiver": 0.9, "unknown": 0.075},
        "evidence_receipts": {"audio_call": {}, "scoreboard": {}},
        "abstentions": [],
        "provenance": {"human_derived_inputs": []},
    }
    paths = [
        {
            "rank": 1,
            "log_score": -2.0,
            "log_score_gap": 0.0,
            "ending_kind": "second_bounce",
            "events": [
                {"event_type": "contact", "frame": 10.0, "marginal": 0.9},
                {"event_type": "bounce", "frame": 20.0, "marginal": 0.8},
                {"event_type": "bounce", "frame": 30.0, "marginal": 0.8},
            ],
            "endings": [
                {"kind": "first_bounce_out", "frame": 20.0, "source_event_index": 1},
                {"kind": "second_bounce", "frame": 30.0, "source_event_index": 2},
            ],
        }
    ]

    selected = apply_ending_context(paths, context)[0]

    assert selected["ending_kind"] == "first_bounce_out"
    assert selected["decoder_ending_kind"] == "second_bounce"
    assert selected["log_score"] == -2.0
    assert selected["ending_context_witness"]["firm_path_or_score_changed"] is False


def test_packet_context_selects_semantics_only_at_ranked_terminal_picture() -> None:
    context = {
        "schema": point_context.SCHEMA,
        "automatic": True,
        "match_id": "match",
        "point_id": "pt0001",
        "serve_ordinal": {"first": 0.25, "second": 0.25, "fault": 0.25, "let": 0.25},
        "ending_kind": {"out": 0.8, "second_bounce": 0.1, "net": 0.05, "fov_exit": 0.05},
        "winner": "unknown",
        "winner_likelihoods": {"server": 0.0, "receiver": 0.0, "unknown": 1.0},
        "evidence_receipts": {"audio_call": {}},
        "abstentions": ["scoreboard_winner"],
        "provenance": {"human_derived_inputs": []},
    }
    paths = [
        {
            "rank": 1,
            "log_score": -1.0,
            "log_score_gap": 0.0,
            "ending_kind": "second_bounce",
            "events": [
                {"event_type": "contact", "frame": 10.0, "marginal": 0.9},
                {"event_type": "bounce", "frame": 30.0, "marginal": 0.8},
            ],
            "endings": [{"kind": "second_bounce", "frame": 30.0, "source_event_index": 1}],
        },
        {
            "rank": 2,
            "log_score": -4.0,
            "log_score_gap": 3.0,
            "ending_kind": "first_bounce_out",
            "events": [
                {"event_type": "contact", "frame": 10.0, "marginal": 0.9},
                {"event_type": "bounce", "frame": 30.0, "marginal": 0.8},
            ],
            "endings": [{"kind": "first_bounce_out", "frame": 30.0, "source_event_index": 1}],
        },
    ]

    scored = apply_ending_context(paths, context)
    selection = packet_ending_selection(scored, 1.0)

    assert selection["decoder_anchor_frame"] == 30.0
    assert selection["decoder_ending_kind"] == "second_bounce"
    assert selection["ending_kind"] == "first_bounce_out"
    assert selection["firm_path_or_score_changed"] is False


def test_packet_keeps_proposals_tentative_and_firm_unchanged(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    np.savez(
        predictions,
        probabilities=_rows([{"contact": 0.99}]),
        frames=np.asarray([10]),
        clips=np.asarray(["match__pt0001"]),
        broadcasts=np.asarray(["match"]),
        court_y=np.asarray([0.2]),
        time_offset=np.zeros(1),
    )
    firm = tmp_path / "firm.json"
    firm_rows = [{"clip": "match__pt0001", "event_type": "contact", "frame": 10, "abstain": False}]
    firm.write_text(json.dumps(firm_rows))
    proposals = tmp_path / "proposals.json"
    proposals.write_text(
        json.dumps(
            {
                "schema": "event_proposals_v1",
                "labels_or_reviewed_inputs": [],
                "proposals": [
                    {
                        "clip": "match__pt0001",
                        "frame": 30,
                        "start_frame": 27,
                        "end_frame": 33,
                        "kinds": ["bounce"],
                        "source": "bounce_implication",
                        "confidence": 0.5,
                        "evidence": {},
                    }
                ],
            }
        )
    )
    active = tmp_path / "active_play.json"
    active.write_text(
        json.dumps(
            {
                "match/pt0001": {
                    "active_spans": [[8, 20]],
                    "event_spans": [[5, 23]],
                    "gate_held": False,
                }
            }
        )
    )

    build_packets(predictions, firm, proposals, tmp_path / "packets", count=1, active_play=active)
    packet = json.loads((tmp_path / "packets/match__pt0001.event_candidates.json").read_text())

    assert packet["firm"] == firm_rows
    assert packet["configuration"]["proposal_effect_on_firm"] == "none"
    assert packet["schema"] == "event_candidate_packet_v2"
    assert packet["configuration"]["tentative_probability_floor"] == 0.05
    assert packet["tentative"][0]["source"] == "bounce_implication"
    assert packet["tentative"][0]["likelihood_source"].startswith("uncalibrated")
    assert packet["lattice"][0]["event_type"] == "contact"
    assert packet["proposal_windows"][0]["kinds"] == ["bounce"]
    assert packet["active_play"]["event_spans"] == [[5, 23]]
    provenance = json.loads((tmp_path / "packets/provenance.json").read_text())
    assert provenance["mode"] == "automatic"
    assert all(row["sha256"] for row in provenance["reused_artifacts"])


def test_proposal_peak_is_typed_even_below_tentative_floor(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    np.savez(
        predictions,
        probabilities=_rows([{"bounce": 0.002}, {"contact": 0.9, "bounce": 0.004}]),
        frames=np.asarray([29, 30]),
        clips=np.asarray(["match__pt0001", "match__pt0001"]),
        broadcasts=np.asarray(["match", "match"]),
        court_y=np.asarray([0.2, 0.2]),
        time_offset=np.zeros(2),
    )
    firm = tmp_path / "firm.json"
    firm.write_text("[]")
    proposals = tmp_path / "proposals.json"
    proposals.write_text(
        json.dumps(
            {
                "schema": "event_proposals_v1",
                "labels_or_reviewed_inputs": [],
                "proposals": [
                    {
                        "clip": "match__pt0001",
                        "frame": 29,
                        "start_frame": 28,
                        "end_frame": 31,
                        "kinds": ["contact", "bounce"],
                        "source": "bounce_implication",
                        "confidence": 0.5,
                        "evidence": {},
                    }
                ],
            }
        )
    )

    build_packets(predictions, firm, proposals, tmp_path / "packets", count=1)
    packet = json.loads((tmp_path / "packets/match__pt0001.event_candidates.json").read_text())
    peak = next(row for row in packet["tentative"] if row["source"] == "proposal_classifier_peak")

    assert peak["event_type"] == "bounce"
    assert peak["frame"] == 30
    assert peak["classifier_probability"] == pytest.approx(0.004)
    assert peak["timing_range"] == [28, 31]
    assert not any(
        row.get("source") == "proposal_classifier_peak" and row.get("event_type") == "contact"
        for row in packet["tentative"]
    )


def test_packet_rejects_human_derived_proposal_artifact(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    np.savez(
        predictions,
        probabilities=_rows([{"contact": 0.99}]),
        frames=np.asarray([10]),
        clips=np.asarray(["match__pt0001"]),
        broadcasts=np.asarray(["match"]),
        court_y=np.asarray([0.2]),
    )
    firm = tmp_path / "firm.json"
    firm.write_text("[]")
    proposals = tmp_path / "proposals.json"
    proposals.write_text(
        json.dumps(
            {
                "schema": "event_proposals_v1",
                "labels_or_reviewed_inputs": ["owner labels"],
                "proposals": [],
            }
        )
    )

    try:
        build_packets(predictions, firm, proposals, tmp_path / "packets")
    except ValueError as error:
        assert "human-derived" in str(error)
    else:
        raise AssertionError("reviewed inputs must fail closed")


def test_grammar_round_trip_keeps_exact_per_point_side_maps(tmp_path: Path) -> None:
    source = tmp_path / "grammar.json"
    source.write_text(
        json.dumps(
            {
                "schema": "event_path_grammar_v1",
                "labels_or_reviewed_inputs": [],
                "grammar": {
                    "net_line": {"match__pt0001": 1.25},
                    "net_line_default": 1.0,
                    "side_dead_band": {"match__pt0001": 0.2},
                    "transition_log_prior": {"contact->bounce": -0.25},
                    "minimum_gap": {"contact->bounce": 4},
                    "terminal_second_bounce": True,
                },
            }
        )
    )

    document = _grammar_document(_load_grammar(source))

    assert document["net_line"] == {"match__pt0001": 1.25}
    assert document["side_dead_band"] == {"match__pt0001": 0.2}
    assert document["transition_log_prior"]["contact->bounce"] == -0.25
    assert document["terminal_second_bounce"] is True


def _timing_columns():
    """A newer prediction artifact's local timing softmax, one row per crop row."""

    from cv.pipeline import event_time_distribution as timing

    pmf = np.full((2, timing.TIME_BINS), 0.02, dtype=np.float32)
    pmf[0, timing.TIME_RADIUS] += 1.0
    pmf[1, timing.TIME_RADIUS + 3] += 1.0
    pmf /= pmf.sum(axis=1, keepdims=True)
    return {
        timing.PMF_KEY: pmf,
        timing.GRID_KEY: np.tile(timing.expected_grid(), (2, 1)),
    }


def _evidence_fixture(tmp_path, distribution=False):
    probabilities = np.asarray([[0.001, 0.001, 0.997, 0.001], [0.70, 0.05, 0.249, 0.001]])
    predictions = tmp_path / "source.npz"
    np.savez(
        predictions,
        probabilities=probabilities,
        frames=np.asarray([139, 215]),
        clips=np.asarray(["match__pt0001"] * 2),
        broadcasts=np.asarray(["match"] * 2),
        court_y=np.asarray([0.2, 0.2]),
        time_offset=np.asarray([-0.015, 0.1]),
        time_confidence=np.asarray([0.51, 0.29]),
        **(_timing_columns() if distribution else {}),
    )
    rows = []
    for i, frame in enumerate([139, 215]):
        rows.append(
            dict(
                clip="match__pt0001",
                match_id="match",
                candidate_frame=frame,
                frame=frame,
                event_type="bounce",
                abstain=True,
                model_abstain=i == 1,
                gate_held=i == 0,
                class_probabilities=dict(
                    zip(("none", "contact", "bounce", "net_hit"), probabilities[i])
                ),
                impulse_support=dict(
                    supported=False,
                    reason="wing_not_locally_consistent" if i == 0 else "model_hold",
                ),
            )
        )
    emissions = tmp_path / "emissions.json"
    emissions.write_text(json.dumps(rows))
    return predictions, emissions, rows


def test_evidence_only_retains_refusals_and_does_not_decode_or_invent_time(tmp_path, monkeypatch):
    from cv.pipeline import event_paths

    predictions, emissions, rows = _evidence_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("evidence-only export must not decode alternative paths")

    monkeypatch.setattr(event_paths, "diverse_paths", forbidden)
    event_paths.build_packets(predictions, emissions, None, tmp_path / "out", evidence_only=True)
    path = tmp_path / "out/match__pt0001.event_candidates.json"
    packet = event_paths.load_evidence_packet(path, emissions, "match__pt0001")
    assert packet["firm"] == [] and packet["paths"] == []
    assert packet["raw_emissions"] == rows
    assert len(packet["tentative"]) >= 2
    assert packet["source_evidence"][0]["raw_emission_indices"] == [0]
    assert packet["source_evidence"][1]["class_probabilities"]["none"] == 0.70
    assert all(
        r["time_distribution_status"] == "unavailable_in_prediction_artifact"
        for r in packet["source_evidence"]
    )
    assert all(not r["time_distribution_is_calibrated"] for r in packet["source_evidence"])


@pytest.mark.parametrize(
    "tamper", ["firm", "refusal", "scores", "clip", "source", "candidate_time", "path"]
)
def test_candidate_sidecar_rejects_changed_source_or_evidence(tmp_path, tamper):
    from cv.pipeline import event_paths

    predictions, emissions, rows = _evidence_fixture(tmp_path)
    event_paths.build_packets(predictions, emissions, None, tmp_path / "out", evidence_only=True)
    path = tmp_path / "out/match__pt0001.event_candidates.json"
    packet = json.loads(path.read_text())
    if tamper == "firm":
        packet["firm"] = [rows[0]]
    elif tamper == "refusal":
        packet["raw_emissions"][0]["gate_held"] = False
    elif tamper == "scores":
        packet["source_evidence"][0]["class_probabilities"]["bounce"] = 1.0
    elif tamper == "candidate_time":
        packet["tentative"][0]["frame"] += 1
    elif tamper == "path":
        packet["paths"] = [{"events": [rows[0]]}]
    elif tamper == "clip":
        packet["clip"] = "other__pt0001"
    else:
        emissions.write_text("[]")
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        event_paths.load_evidence_packet(path, emissions, "match__pt0001")


def test_a_sidecar_round_trips_the_local_timing_distribution_when_it_exists(tmp_path):
    from cv.pipeline import event_paths, event_time_distribution as timing

    predictions, emissions, _rows = _evidence_fixture(tmp_path, distribution=True)
    event_paths.build_packets(predictions, emissions, None, tmp_path / "out", evidence_only=True)
    path = tmp_path / "out/match__pt0001.event_candidates.json"
    packet = event_paths.load_evidence_packet(path, emissions, "match__pt0001")
    expected = _timing_columns()[timing.PMF_KEY]
    for index, record in enumerate(packet["source_evidence"]):
        assert record["time_distribution_status"] == timing.AVAILABLE
        assert record["time_distribution_is_calibrated"] is False
        assert record["time_distribution"]["calibrated"] is False
        assert record["time_distribution"]["offset_frames"] == [float(v) for v in range(-8, 9)]
        assert record["time_distribution"]["probabilities"] == pytest.approx(
            expected[index].tolist(), abs=1e-9
        )
    # The distribution is a checked part of the sidecar, not free-text metadata.
    tampered = json.loads(path.read_text())
    tampered["source_evidence"][0]["time_distribution"]["probabilities"][0] += 0.5
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="changed source classifier evidence"):
        event_paths.load_evidence_packet(path, emissions, "match__pt0001")


def test_an_evidence_sidecar_never_invents_a_distribution_for_an_old_artifact(tmp_path):
    from cv.pipeline import event_paths

    predictions, emissions, _rows = _evidence_fixture(tmp_path)
    event_paths.build_packets(predictions, emissions, None, tmp_path / "out", evidence_only=True)
    packet = json.loads((tmp_path / "out/match__pt0001.event_candidates.json").read_text())
    assert all(
        "time_distribution" not in record
        and record["time_distribution_status"] == "unavailable_in_prediction_artifact"
        for record in packet["source_evidence"]
    )


@pytest.mark.parametrize("evidence_only", [False, True])
def test_all_packet_modes_refuse_malformed_optional_timing_evidence(tmp_path, evidence_only):
    from cv.pipeline import event_paths, event_time_distribution as timing

    predictions, emissions, _ = _evidence_fixture(tmp_path, distribution=True)
    with np.load(predictions, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data[timing.PMF_KEY][0, 0] += 0.01
    np.savez(predictions, **data)
    with pytest.raises(ValueError, match="time distribution rows must be normalised"):
        event_paths.build_packets(
            predictions, emissions, None, tmp_path / "out", evidence_only=evidence_only
        )


def test_evidence_sidecar_retains_independent_cues_without_classifier_admission(tmp_path):
    from cv.pipeline import event_paths
    from cv.pipeline.event_proposals import EventProposal, proposal_document

    predictions, emissions, original = _evidence_fixture(tmp_path)
    evidence = {
        "cue_family": "audio",
        "correlation_group": "same_waveform",
        "time_distribution": {"offset_seconds": [-0.02, 0, 0.02], "mass": [0.2, 0.6, 0.2]},
        "timing_is_calibrated": False,
    }
    # No classifier crop at either candidate epoch, and no decoded contact.
    proposals = [
        EventProposal("match__pt0001", 80, 78, 82, ("contact", "bounce"), source, 0.7, evidence)
        for source in ["audio_transient", "audio_embedding"]
    ]
    source = tmp_path / "proposals.json"
    source.write_text(json.dumps(proposal_document(proposals)))
    event_paths.build_packets(predictions, emissions, source, tmp_path / "out", evidence_only=True)
    path = tmp_path / "out/match__pt0001.event_candidates.json"
    packet = event_paths.load_evidence_packet(path, emissions, "match__pt0001")
    assert packet["proposal_windows"] == [
        p.as_dict() for p in sorted(proposals, key=lambda p: p.source)
    ]
    assert packet["raw_emissions"] == original
    assert packet["firm"] == packet["paths"] == packet["lattice"] == []
    assert all(row["frame"] != 80 for row in packet["tentative"])
    assert (
        len(packet["proposal_windows"]) == 2
    )  # correlated cues remain separate evidence, not votes
    packet["proposal_windows"][0]["evidence"]["time_distribution"]["mass"] = [0, 1, 0]
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="changed independent source proposals"):
        event_paths.load_evidence_packet(path, emissions, "match__pt0001")


def test_evidence_sidecar_checks_independent_cue_source_digest(tmp_path):
    from cv.pipeline import event_paths
    from cv.pipeline.event_proposals import proposal_document

    predictions, emissions, _ = _evidence_fixture(tmp_path)
    source = tmp_path / "proposals.json"
    source.write_text(json.dumps(proposal_document([])))
    event_paths.build_packets(predictions, emissions, source, tmp_path / "out", evidence_only=True)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="independent proposal ancestry changed"):
        event_paths.load_evidence_packet(
            tmp_path / "out/match__pt0001.event_candidates.json", emissions, "match__pt0001"
        )
