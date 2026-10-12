import numpy as np

from cv.pipeline.event_decoder import (
    BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG,
    BOUNCE_RECALL_SHADOW_CONFIG,
    CONTACT_NET_RECALL_SHADOW_CONFIG,
    CONTACT_WITNESS_RECALL_SHADOW_CONFIG,
    DEFAULT_CONFIG,
    EVENT_RECALL_SHADOW_CONFIG,
    GRAMMAR_RECALL_SHADOW_CONFIG,
    PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    SERVE_CONTACT_RECALL_SHADOW_CONFIG,
    complete_confident_contact_witnesses,
    complete_grammar_contacts,
    complete_physical_net_witnesses,
    decode_graph,
    decode_net_hits,
    refine_event_timing,
    suppress_same_type_duplicates,
)


def test_bounce_anchor_shadow_changes_only_bounce_anchor() -> None:
    assert BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.anchor_thresholds == {
        **DEFAULT_CONFIG.anchor_thresholds,
        "bounce": 0.80,
    }
    assert BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.bridge_thresholds == DEFAULT_CONFIG.bridge_thresholds
    assert BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.max_gap_seconds == DEFAULT_CONFIG.max_gap_seconds
    assert (
        BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.contact_witness_completion_threshold
        == DEFAULT_CONFIG.contact_witness_completion_threshold
    )


def _row(candidate_id: str, frame: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "clip": "match__pt0001",
        "match_id": "match",
        "proposal_frame": frame,
        "source_fps": 25.0,
        "nearest_player_edge_distance_norm": 0.2,
        "proposal_source": "trajectory",
        "production_scope": True,
    }


def test_default_profile_recovers_lower_confidence_bounce() -> None:
    assert DEFAULT_CONFIG.anchor_thresholds["bounce"] == 0.75
    assert DEFAULT_CONFIG.bridge_thresholds["contact"] == 0.60

    rows = [_row("contact", 10.0), _row("bounce", 30.0)]
    probabilities = np.asarray(
        [
            [0.01, 0.95, 0.01, 0.0],
            [0.10, 0.01, 0.85, 0.0],
        ]
    )

    default = decode_graph(rows, probabilities, config=DEFAULT_CONFIG)
    shadow = decode_graph(rows, probabilities, config=BOUNCE_RECALL_SHADOW_CONFIG)

    assert [row["event_type"] for row in default] == ["contact", "bounce"]
    assert [row["event_type"] for row in shadow] == ["contact", "bounce"]


def test_serve_shadow_promotes_only_dedicated_serve_candidate() -> None:
    rows = [_row("serve", 10.0), _row("trajectory", 30.0)]
    rows[0]["proposal_source"] = "serve"
    probabilities = np.asarray(
        [
            [0.20, 0.79, 0.01, 0.0],
            [0.20, 0.79, 0.01, 0.0],
        ]
    )

    default = decode_graph(rows, probabilities, config=DEFAULT_CONFIG)
    shadow = decode_graph(rows, probabilities, config=SERVE_CONTACT_RECALL_SHADOW_CONFIG)

    assert default == []
    assert [(row["candidate_id"], row["event_type"]) for row in shadow] == [("serve", "contact")]


def test_contact_witness_completion_respects_scope_source_and_exclusion() -> None:
    rows = [
        _row("accepted", 20.0),
        _row("held", 25.0),
        _row("near_existing", 31.0),
        _row("wrong_source", 50.0),
    ]
    for row in rows[:3]:
        row["proposal_source"] = "bounce_witness"
    rows[1]["production_scope"] = False
    probabilities = np.asarray(
        [
            [0.01, 0.74, 0.20, 0.0],
            [0.01, 0.95, 0.01, 0.0],
            [0.01, 0.95, 0.01, 0.0],
            [0.01, 0.95, 0.01, 0.0],
        ]
    )
    predicted = [
        {
            "candidate_id": "existing",
            "clip": "match__pt0001",
            "match_id": "match",
            "event_type": "bounce",
            "frame": 30.0,
            "probability": 0.99,
            "fps": 25.0,
            "production_scope": True,
        }
    ]

    completed = complete_confident_contact_witnesses(
        rows,
        probabilities,
        predicted,
        config=CONTACT_WITNESS_RECALL_SHADOW_CONFIG,
    )

    assert [(row["candidate_id"], row["event_type"]) for row in completed] == [
        ("accepted", "contact"),
        ("existing", "bounce"),
    ]


def test_decoder_rejects_distant_low_confidence_contact_bridge() -> None:
    rows = [_row("left", 10.0), _row("bridge", 20.0), _row("right", 30.0)]
    rows[1]["nearest_player_edge_distance_norm"] = 2.0
    probabilities = np.asarray(
        [
            [0.01, 0.01, 0.98, 0.0],
            [0.18, 0.81, 0.01, 0.0],
            [0.01, 0.01, 0.98, 0.0],
        ]
    )

    predicted = decode_graph(rows, probabilities)

    assert all(row["candidate_id"] != "bridge" for row in predicted)


def test_grammar_shadow_completes_player_local_contact_between_bounces() -> None:
    rows = [_row("candidate", 20.0), _row("companion", 26.0)]
    rows[0]["nearest_player_edge_distance_norm"] = 0.35
    rows[1]["nearest_player_edge_distance_norm"] = 0.05
    probabilities = np.asarray(
        [
            [0.10, 0.70, 0.05, 0.0],
            [0.70, 0.15, 0.05, 0.0],
        ]
    )
    predicted = [
        {
            "candidate_id": "left",
            "clip": "match__pt0001",
            "match_id": "match",
            "event_type": "bounce",
            "frame": 10.0,
            "probability": 0.95,
            "fps": 25.0,
            "production_scope": True,
        },
        {
            "candidate_id": "right",
            "clip": "match__pt0001",
            "match_id": "match",
            "event_type": "bounce",
            "frame": 40.0,
            "probability": 0.95,
            "fps": 25.0,
            "production_scope": True,
        },
    ]

    completed = complete_grammar_contacts(
        rows,
        probabilities,
        predicted,
        config=GRAMMAR_RECALL_SHADOW_CONFIG,
    )

    assert [event["event_type"] for event in completed] == ["bounce", "contact", "bounce"]
    assert completed[1]["frame"] == 23.0
    assert completed[1]["timing_refinement"] == "contact_candidate_pair_midpoint"


def test_timing_refinement_suppresses_same_type_duplicates() -> None:
    events = [
        {
            "candidate_id": "strong",
            "clip": "match__pt0001",
            "event_type": "contact",
            "frame": 20.0,
            "probability": 0.90,
            "fps": 25.0,
        },
        {
            "candidate_id": "bridge",
            "clip": "match__pt0001",
            "event_type": "contact",
            "frame": 21.5,
            "probability": 0.76,
            "fps": 25.0,
        },
    ]

    kept = suppress_same_type_duplicates(events)

    assert [event["candidate_id"] for event in kept] == ["strong"]


def test_refinement_merges_hypotheses_that_converge_on_one_contact() -> None:
    rows = [_row("strong", 20.0), _row("bridge", 21.5)]
    for row in rows:
        row["proposal_source"] = "trajectory"
    probabilities = np.asarray(
        [
            [0.10, 0.82, 0.01, 0.0],
            [0.10, 0.81, 0.01, 0.0],
        ]
    )
    predicted = [
        {
            "candidate_id": row["candidate_id"],
            "clip": row["clip"],
            "match_id": row["match_id"],
            "event_type": "contact",
            "frame": row["proposal_frame"],
            "probability": probabilities[index, 1],
            "fps": row["source_fps"],
            "production_scope": True,
        }
        for index, row in enumerate(rows)
    ]

    refined = refine_event_timing(rows, probabilities, predicted)

    assert len(refined) == 1
    assert refined[0]["candidate_id"] == "strong"
    assert 20.0 < refined[0]["frame"] < 21.5


def test_refinement_does_not_cross_production_scope() -> None:
    rows = [_row("outside", 10.0), _row("serve", 12.0)]
    rows[0]["production_scope"] = False
    rows[1]["proposal_source"] = "serve"
    probabilities = np.asarray(
        [
            [0.05, 0.95, 0.0, 0.0],
            [0.05, 0.95, 0.0, 0.0],
        ]
    )
    predicted = [
        {
            "candidate_id": "serve",
            "clip": "match__pt0001",
            "match_id": "match",
            "event_type": "contact",
            "frame": 12.0,
            "probability": 0.95,
            "fps": 25.0,
            "production_scope": True,
        }
    ]

    refined = refine_event_timing(rows, probabilities, predicted)

    assert refined[0]["frame"] == 12.0
    assert "timing_refinement" not in refined[0]


def test_combined_shadow_requires_net_tape_support() -> None:
    rows = [_row("supported", 20.0), _row("distant", 40.0)]
    rows[0]["tape_px"] = 12.0
    rows[1]["tape_px"] = 30.0
    probabilities = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.44],
            [0.5, 0.0, 0.0, 0.45],
        ]
    )

    predicted = decode_net_hits(
        rows,
        probabilities,
        config=EVENT_RECALL_SHADOW_CONFIG,
    )

    assert [event["candidate_id"] for event in predicted] == ["supported"]


def test_contact_net_shadow_does_not_relax_bounce_thresholds() -> None:
    assert CONTACT_NET_RECALL_SHADOW_CONFIG.anchor_thresholds["bounce"] == 0.89
    assert CONTACT_NET_RECALL_SHADOW_CONFIG.bridge_thresholds["bounce"] == 0.10


def test_physical_net_shadow_preserves_calibrated_contact_configuration() -> None:
    assert PHYSICAL_NET_WITNESS_SHADOW_CONFIG.bridge_thresholds["contact"] == 0.75
    assert (
        PHYSICAL_NET_WITNESS_SHADOW_CONFIG.bridge_thresholds["bounce"]
        == (DEFAULT_CONFIG.bridge_thresholds["bounce"])
    )
    assert (
        PHYSICAL_NET_WITNESS_SHADOW_CONFIG.contact_witness_completion_threshold
        == DEFAULT_CONFIG.contact_witness_completion_threshold
    )
    assert PHYSICAL_NET_WITNESS_SHADOW_CONFIG.net_threshold == DEFAULT_CONFIG.net_threshold


def test_physical_net_shadow_accepts_demoted_initial_net_with_direct_support() -> None:
    row = _physical_net_row(final_type="tracking_artifact", collapse=True)
    row["initial_type"] = "net_hit"
    row["distance_to_net_m"] = 2.0
    row["tape_px"] = 8.0
    row["observed_tape_px"] = 2.0
    row["observed_track_score"] = 0.3
    row["observed_track_source_count"] = 2

    predicted = complete_physical_net_witnesses(
        [row],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert len(predicted) == 1
    assert predicted[0]["event_type"] == "net_hit"


def test_physical_net_shadow_rejects_fitted_only_classified_collapse() -> None:
    row = _physical_net_row(final_type="tracking_artifact", collapse=True)
    row["initial_type"] = "net_hit"
    row["observed_tape_px"] = float("inf")
    row["observed_track_score"] = 0.0
    row["observed_track_source_count"] = 0

    predicted = complete_physical_net_witnesses(
        [row],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert predicted == []


def _physical_net_row(**overrides: object) -> dict:
    row = _row("physical", 20.0)
    row.update(
        {
            "tape_px": 20.0,
            "observed_support_px": 2.0,
            "observed_tape_px": 10.0,
            "observed_track_score": 0.5,
            "observed_track_source_count": 2,
            "sb": 8.0,
            "speed_ratio": 0.8,
            "distance_to_net_m": 2.0,
            "audio": 2.0,
            "collapse": False,
            "bounce_witness": False,
            "final_type": "tracking_artifact",
            "big_gain": False,
            "has_reach": False,
        }
    )
    row.update(overrides)
    return row


def test_physical_net_witness_rejects_ordinary_supported_crossing() -> None:
    completed = complete_physical_net_witnesses(
        [_physical_net_row()], [], config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG
    )

    assert completed == []


def test_physical_net_witness_rejects_bounce_conflict() -> None:
    completed = complete_physical_net_witnesses(
        [_physical_net_row(final_type="net_hit", collapse=True, bounce_witness=True)],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert completed == []


def test_physical_net_witness_accepts_classified_collapse_branch() -> None:
    completed = complete_physical_net_witnesses(
        [_physical_net_row(final_type="net_hit", collapse=True)],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert len(completed) == 1
    assert "classified_collapse" in completed[0]["completion_reason"]


def test_physical_net_witness_accepts_dual_geometry_branch() -> None:
    completed = complete_physical_net_witnesses(
        [_physical_net_row(distance_to_net_m=0.25)],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert len(completed) == 1
    assert "dual_geometry" in completed[0]["completion_reason"]


def test_physical_net_witness_accepts_localized_audio_branch() -> None:
    completed = complete_physical_net_witnesses(
        [_physical_net_row(audio=120.0, observed_tape_px=0.5)],
        [],
        config=PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    )

    assert len(completed) == 1
    assert "localized_audio_impulse" in completed[0]["completion_reason"]
