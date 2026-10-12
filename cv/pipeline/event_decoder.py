"""Shared label-free graph decoding for typed event probabilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cv.pipeline.event_sequence_graph import EventNode, complete_bridges

EVENT_TYPES = ("contact", "bounce", "net_hit")


@dataclass(frozen=True)
class EventDecoderConfig:
    anchor_thresholds: Mapping[str, float]
    bridge_thresholds: Mapping[str, float]
    max_gap_seconds: float
    contact_player_edge_max: float
    net_threshold: float = 0.50
    net_tape_max_px: float = float("inf")
    contact_witness_completion_threshold: float | None = None
    physical_net_witness: bool = False
    serve_contact_anchor_threshold: float | None = None
    grammar_contact_completion_threshold: float | None = None
    grammar_contact_pair_midpoint: bool = False


# Adopted 2026-08-05 (owner-directed): the measured precision-preserving slice —
# contact bridge 0.80 -> 0.75 with witness completion 0.72, plus the binary net arm
# (0.425 with tape <= 25 px). Benchmarked on 1,406 current-standard events:
# 507/10/899 -> 522/10/884 (+15 truths, +0 FP); V5 holdout 241/11/302 -> 249/11/294.
# A composed 822-event ablation selected bounce 0.75: 454 TP / 20 FP is the highest-recall
# setting at or above the owner-directed 95.75% precision floor. S5-05 then selected the
# 0.60 contact bridge: 466 TP / 24 FP on 822 historical events and 145 / 6 on 266 untouched
# events, with the bounce anchor fixed at 0.75.
# The prior default is preserved below as LEGACY_DEFAULT_CONFIG.
DEFAULT_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.75, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.60, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=3.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
    contact_witness_completion_threshold=0.72,
)

LEGACY_DEFAULT_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.89, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.80, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=3.0,
    contact_player_edge_max=0.50,
)

BOUNCE_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.80, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.80, "bounce": 0.05, "net_hit": 2.0},
    max_gap_seconds=4.0,
    contact_player_edge_max=0.50,
)

CONTACT_NET_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.89, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.75, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=3.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
)

CONTACT_WITNESS_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.89, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.75, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=3.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
    contact_witness_completion_threshold=0.72,
)

BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={**DEFAULT_CONFIG.anchor_thresholds, "bounce": 0.80},
    bridge_thresholds=DEFAULT_CONFIG.bridge_thresholds,
    max_gap_seconds=DEFAULT_CONFIG.max_gap_seconds,
    contact_player_edge_max=DEFAULT_CONFIG.contact_player_edge_max,
    net_threshold=DEFAULT_CONFIG.net_threshold,
    net_tape_max_px=DEFAULT_CONFIG.net_tape_max_px,
    contact_witness_completion_threshold=DEFAULT_CONFIG.contact_witness_completion_threshold,
)

EVENT_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.80, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.75, "bounce": 0.05, "net_hit": 2.0},
    max_gap_seconds=4.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
)

GUARDED_EVENT_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.80, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.75, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=4.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
)

PHYSICAL_NET_WITNESS_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds={"contact": 0.83, "bounce": 0.89, "net_hit": 2.0},
    bridge_thresholds={"contact": 0.75, "bounce": 0.10, "net_hit": 2.0},
    max_gap_seconds=3.0,
    contact_player_edge_max=0.50,
    net_threshold=0.425,
    net_tape_max_px=25.0,
    contact_witness_completion_threshold=0.72,
    physical_net_witness=True,
)

SERVE_CONTACT_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds=DEFAULT_CONFIG.anchor_thresholds,
    bridge_thresholds=DEFAULT_CONFIG.bridge_thresholds,
    max_gap_seconds=DEFAULT_CONFIG.max_gap_seconds,
    contact_player_edge_max=DEFAULT_CONFIG.contact_player_edge_max,
    net_threshold=DEFAULT_CONFIG.net_threshold,
    net_tape_max_px=DEFAULT_CONFIG.net_tape_max_px,
    contact_witness_completion_threshold=DEFAULT_CONFIG.contact_witness_completion_threshold,
    serve_contact_anchor_threshold=0.75,
)

BOUNCE_SERVE_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.anchor_thresholds,
    bridge_thresholds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.bridge_thresholds,
    max_gap_seconds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.max_gap_seconds,
    contact_player_edge_max=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.contact_player_edge_max,
    net_threshold=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.net_threshold,
    net_tape_max_px=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.net_tape_max_px,
    contact_witness_completion_threshold=(
        BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.contact_witness_completion_threshold
    ),
    serve_contact_anchor_threshold=0.75,
)

BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.anchor_thresholds,
    bridge_thresholds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.bridge_thresholds,
    max_gap_seconds=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.max_gap_seconds,
    contact_player_edge_max=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.contact_player_edge_max,
    net_threshold=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.net_threshold,
    net_tape_max_px=BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.net_tape_max_px,
    contact_witness_completion_threshold=(
        BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG.contact_witness_completion_threshold
    ),
    physical_net_witness=True,
    serve_contact_anchor_threshold=0.75,
)

GRAMMAR_RECALL_SHADOW_CONFIG = EventDecoderConfig(
    anchor_thresholds=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.anchor_thresholds,
    bridge_thresholds=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.bridge_thresholds,
    max_gap_seconds=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.max_gap_seconds,
    contact_player_edge_max=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.contact_player_edge_max,
    net_threshold=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.net_threshold,
    net_tape_max_px=BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.net_tape_max_px,
    contact_witness_completion_threshold=(
        BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG.contact_witness_completion_threshold
    ),
    physical_net_witness=True,
    serve_contact_anchor_threshold=0.75,
    grammar_contact_completion_threshold=0.65,
    grammar_contact_pair_midpoint=True,
)

PROFILES = {
    "default": DEFAULT_CONFIG,
    "bounce_recall_shadow": BOUNCE_RECALL_SHADOW_CONFIG,
    "contact_net_recall_shadow": CONTACT_NET_RECALL_SHADOW_CONFIG,
    "contact_witness_recall_shadow": CONTACT_WITNESS_RECALL_SHADOW_CONFIG,
    "bounce_anchor_recall_shadow": BOUNCE_ANCHOR_RECALL_SHADOW_CONFIG,
    "event_recall_shadow": EVENT_RECALL_SHADOW_CONFIG,
    "guarded_event_recall_shadow": GUARDED_EVENT_RECALL_SHADOW_CONFIG,
    "physical_net_witness_shadow": PHYSICAL_NET_WITNESS_SHADOW_CONFIG,
    "serve_contact_recall_shadow": SERVE_CONTACT_RECALL_SHADOW_CONFIG,
    "bounce_serve_recall_shadow": BOUNCE_SERVE_RECALL_SHADOW_CONFIG,
    "bounce_serve_physical_recall_shadow": BOUNCE_SERVE_PHYSICAL_RECALL_SHADOW_CONFIG,
    "grammar_recall_shadow": GRAMMAR_RECALL_SHADOW_CONFIG,
}


def decode_graph(
    rows: list[dict],
    probabilities: np.ndarray,
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
    scoped_only: bool = True,
) -> list[dict]:
    """Decode anchors and conservative interior bridges from typed probabilities."""
    class_index = {event_type: index + 1 for index, event_type in enumerate(EVENT_TYPES)}
    grouped: dict[str, list[tuple[int, str, float]]] = {}
    for row_index, row in enumerate(rows):
        if scoped_only and not row.get("production_scope"):
            continue
        for event_type in EVENT_TYPES:
            probability = float(probabilities[row_index, class_index[event_type]])
            if probability < config.bridge_thresholds[event_type]:
                continue
            serve_anchor = (
                event_type == "contact"
                and row.get("proposal_source") == "serve"
                and config.serve_contact_anchor_threshold is not None
                and probability >= config.serve_contact_anchor_threshold
            )
            if (
                event_type == "contact"
                and probability < config.anchor_thresholds[event_type]
                and float(row["nearest_player_edge_distance_norm"]) > config.contact_player_edge_max
                and not serve_anchor
            ):
                continue
            grouped.setdefault(row["clip"], []).append((row_index, event_type, probability))

    output = []
    for clip, hypotheses in grouped.items():
        fps = float(rows[hypotheses[0][0]]["source_fps"])
        ranked_anchors = sorted(
            (
                hypothesis
                for hypothesis in hypotheses
                if hypothesis[2] >= config.anchor_thresholds[hypothesis[1]]
                or (
                    hypothesis[1] == "contact"
                    and rows[hypothesis[0]].get("proposal_source") == "serve"
                    and config.serve_contact_anchor_threshold is not None
                    and hypothesis[2] >= config.serve_contact_anchor_threshold
                )
            ),
            key=lambda hypothesis: hypothesis[2],
            reverse=True,
        )
        anchors = []
        for row_index, event_type, probability in ranked_anchors:
            frame = float(rows[row_index]["proposal_frame"])
            if any(abs(node.frame - frame) <= 0.08 * fps for node in anchors):
                continue
            anchors.append(
                EventNode(
                    f"{row_index}:{event_type}",
                    frame,
                    event_type,
                    probability,
                    fps=fps,
                )
            )
        bridges = [
            EventNode(
                f"{row_index}:{event_type}",
                float(rows[row_index]["proposal_frame"]),
                event_type,
                probability,
                fps=fps,
            )
            for row_index, event_type, probability in hypotheses
            if event_type != "net_hit" and probability < config.anchor_thresholds[event_type]
        ]
        selected = complete_bridges(
            anchors,
            bridges,
            max_gap_seconds=config.max_gap_seconds,
        )
        for node in selected:
            row = rows[int(node.node_id.split(":", 1)[0])]
            output.append(
                {
                    "candidate_id": row["candidate_id"],
                    "clip": clip,
                    "match_id": row["match_id"],
                    "event_type": node.event_type,
                    "frame": node.frame,
                    "probability": node.unary,
                    "fps": fps,
                    "production_scope": bool(row.get("production_scope")),
                }
            )
    return sorted(output, key=lambda row: (row["clip"], row["frame"]))


def suppress_same_type_duplicates(
    events: list[dict], *, exclusion_seconds: float = 0.08
) -> list[dict]:
    """Remove duplicate same-type emissions introduced by timing refinement."""
    kept = []
    for event in sorted(events, key=lambda row: -float(row["probability"])):
        radius = exclusion_seconds * float(event["fps"])
        if any(
            prior["clip"] == event["clip"]
            and prior["event_type"] == event["event_type"]
            and abs(float(prior["frame"]) - float(event["frame"])) <= radius
            for prior in kept
        ):
            continue
        kept.append(event)
    return sorted(kept, key=lambda row: (row["clip"], row["frame"], row["event_type"]))


def complete_confident_contact_witnesses(
    rows: list[dict],
    probabilities: np.ndarray,
    predicted: list[dict],
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Add isolated contact classifications from strong bounce-kink witnesses."""
    threshold = config.contact_witness_completion_threshold
    if threshold is None:
        return predicted
    contact_index = EVENT_TYPES.index("contact") + 1
    additions = []
    for row_index, row in enumerate(rows):
        probability = float(probabilities[row_index, contact_index])
        if (
            not row.get("production_scope")
            or row.get("proposal_source") != "bounce_witness"
            or probability < threshold
        ):
            continue
        frame = float(row["proposal_frame"])
        fps = float(row["source_fps"])
        radius = 0.08 * fps
        if any(
            event["clip"] == row["clip"] and abs(float(event["frame"]) - frame) <= radius
            for event in predicted
        ):
            continue
        additions.append(
            {
                "candidate_id": row["candidate_id"],
                "clip": row["clip"],
                "match_id": row["match_id"],
                "event_type": "contact",
                "frame": frame,
                "probability": probability,
                "fps": fps,
                "production_scope": True,
                "completion_reason": "bounce_kink_contact_witness",
            }
        )
    kept = []
    for addition in sorted(additions, key=lambda event: -float(event["probability"])):
        radius = 0.08 * float(addition["fps"])
        if any(
            prior["clip"] == addition["clip"]
            and abs(float(prior["frame"]) - float(addition["frame"])) <= radius
            for prior in kept
        ):
            continue
        kept.append(addition)
    return sorted([*predicted, *kept], key=lambda row: (row["clip"], row["frame"]))


def decode_net_hits(
    rows: list[dict],
    binary_probabilities: np.ndarray,
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Decode net collisions with probability, tape support, and temporal NMS."""
    net_index = EVENT_TYPES.index("net_hit") + 1
    selected = []
    for row_index, row in enumerate(rows):
        probability = float(binary_probabilities[row_index, net_index])
        if not row.get("production_scope") or probability < config.net_threshold:
            continue
        if float(row.get("tape_px", float("inf"))) > config.net_tape_max_px:
            continue
        selected.append(
            {
                "candidate_id": row["candidate_id"],
                "clip": row["clip"],
                "match_id": row["match_id"],
                "event_type": "net_hit",
                "frame": float(row["proposal_frame"]),
                "probability": probability,
                "fps": float(row["source_fps"]),
                "production_scope": True,
            }
        )
    return suppress_same_type_duplicates(selected)


def complete_physical_net_witnesses(
    rows: list[dict],
    predicted: list[dict],
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Add conservative label-free net collisions from measured-cord trajectory evidence."""
    if not config.physical_net_witness:
        return predicted
    additions = []
    for row in rows:
        tape_px = float(row.get("tape_px", float("inf")))
        fitted_support_px = float(row.get("observed_support_px", float("inf")))
        observed_tape_px = float(row.get("observed_tape_px", float("inf")))
        observed_track_score = float(row.get("observed_track_score", 0.0))
        observed_source_count = int(float(row.get("observed_track_source_count", 0)))
        incoming_speed = float(row.get("sb", 0.0))
        speed_ratio = float(row.get("speed_ratio", float("inf")))
        distance_to_net_m = float(row.get("distance_to_net_m", float("inf")))
        audio = float(row.get("audio", 0.0))
        collapse = bool(row.get("collapse"))
        bounce_conflict = bool(row.get("bounce_witness")) or row.get("final_type") == "bounce"
        fitted_support = tape_px <= 3.0 and fitted_support_px <= 6.0
        direct_support = (
            observed_tape_px <= 25.0 and observed_track_score >= 0.10 and observed_source_count >= 2
        )
        classified_collapse = (
            (row.get("initial_type") == "net_hit" or row.get("final_type") == "net_hit")
            and collapse
            and distance_to_net_m <= 4.0
            and direct_support
        )
        dual_geometry = distance_to_net_m <= 0.5 and fitted_support_px <= 4.0
        localized_audio_impulse = audio >= 100.0 and observed_tape_px <= 1.0
        if (
            not row.get("production_scope")
            or not (fitted_support or direct_support)
            or incoming_speed < 4.0
            or speed_ratio > 1.65
            or bounce_conflict
            or bool(row.get("big_gain"))
            or bool(row.get("has_reach"))
            or not (classified_collapse or dual_geometry or localized_audio_impulse)
        ):
            continue
        branches = []
        if classified_collapse:
            branches.append("classified_collapse")
        if dual_geometry:
            branches.append("dual_geometry")
        if localized_audio_impulse:
            branches.append("localized_audio_impulse")
        additions.append(
            {
                "candidate_id": row["candidate_id"],
                "clip": row["clip"],
                "match_id": row["match_id"],
                "event_type": "net_hit",
                "frame": float(row["proposal_frame"]),
                "probability": 0.90,
                "fps": float(row["source_fps"]),
                "production_scope": True,
                "completion_reason": "measured_cord_trajectory_witness:" + "+".join(branches),
            }
        )
    return suppress_same_type_duplicates([*predicted, *additions])


def complete_grammar_contacts(
    rows: list[dict],
    probabilities: np.ndarray,
    predicted: list[dict],
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Fill one player-local contact between consecutive emitted bounces."""
    threshold = config.grammar_contact_completion_threshold
    if threshold is None:
        return predicted
    contact_index = EVENT_TYPES.index("contact") + 1
    rows_by_clip: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    events_by_clip: dict[str, list[dict]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        if row.get("production_scope"):
            rows_by_clip[row["clip"]].append((row, float(probabilities[row_index, contact_index])))
    for event in predicted:
        events_by_clip[event["clip"]].append(event)

    additions = []
    for clip, events in events_by_clip.items():
        ordered = sorted(events, key=lambda event: float(event["frame"]))
        for left, right in zip(ordered, ordered[1:]):
            if left["event_type"] != "bounce" or right["event_type"] != "bounce":
                continue
            fps = float(left["fps"])
            left_frame = float(left["frame"])
            right_frame = float(right["frame"])
            gap_seconds = (right_frame - left_frame) / fps
            if gap_seconds < 0.16 or gap_seconds > config.max_gap_seconds:
                continue
            margin = 0.08 * fps
            candidates = [
                (probability, row)
                for row, probability in rows_by_clip.get(clip, [])
                if left_frame + margin < float(row["proposal_frame"]) < right_frame - margin
                and probability >= threshold
                and float(row["nearest_player_edge_distance_norm"])
                <= config.contact_player_edge_max
            ]
            if not candidates:
                continue
            probability, row = max(candidates, key=lambda item: item[0])
            frame = float(row["proposal_frame"])
            timing_refinement = None
            if config.grammar_contact_pair_midpoint:
                companions = [
                    candidate
                    for candidate, candidate_probability in rows_by_clip.get(clip, [])
                    if candidate["proposal_source"]
                    in {"trajectory", "suppressed_trajectory", "bounce_witness"}
                    and abs(float(candidate["proposal_frame"]) - frame) <= 0.24 * fps
                    and candidate_probability >= 0.10
                    and float(candidate["nearest_player_edge_distance_norm"])
                    <= config.contact_player_edge_max
                ]
                closest_player = min(
                    companions,
                    key=lambda candidate: float(candidate["nearest_player_edge_distance_norm"]),
                    default=None,
                )
                if closest_player is not None:
                    closest_frame = float(closest_player["proposal_frame"])
                    if abs(closest_frame - frame) >= 0.08 * fps and float(
                        closest_player["nearest_player_edge_distance_norm"]
                    ) + 0.20 <= float(row["nearest_player_edge_distance_norm"]):
                        frame = 0.5 * (frame + closest_frame)
                        timing_refinement = "contact_candidate_pair_midpoint"
            additions.append(
                {
                    "candidate_id": row["candidate_id"],
                    "clip": clip,
                    "match_id": row["match_id"],
                    "event_type": "contact",
                    "frame": frame,
                    "probability": probability,
                    "fps": fps,
                    "production_scope": True,
                    "completion_reason": "same_type_candidate_grammar_shadow",
                    **({"timing_refinement": timing_refinement} if timing_refinement else {}),
                }
            )
    return suppress_same_type_duplicates([*predicted, *additions])


def refine_event_timing(
    rows: list[dict],
    probabilities: np.ndarray,
    predicted: list[dict],
    *,
    config: EventDecoderConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """Fuse nearby label-free hypotheses and suppress refinement-created duplicates."""
    class_index = {event_type: index + 1 for index, event_type in enumerate(EVENT_TYPES)}
    rows_by_clip: dict[str, list[int]] = defaultdict(list)
    rows_by_candidate: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        rows_by_clip[row["clip"]].append(row_index)
        rows_by_candidate[row["candidate_id"]].append(row_index)

    def selected_row_index(event: dict) -> int:
        return min(
            rows_by_candidate[event["candidate_id"]],
            key=lambda row_index: abs(
                float(rows[row_index]["proposal_frame"]) - float(event["frame"])
            ),
        )

    def hypotheses(
        event: dict,
        *,
        radius_seconds: float,
        minimum_probability: float,
        relative_probability: float = 0.0,
        player_edge_max: float | None = None,
        sources: set[str] | None = None,
    ) -> list[tuple[float, float]]:
        probability_index = class_index[event["event_type"]]
        by_frame: dict[float, float] = {}
        for row_index in rows_by_clip[event["clip"]]:
            row = rows[row_index]
            if not row.get("production_scope"):
                continue
            frame = float(row["proposal_frame"])
            probability = float(probabilities[row_index, probability_index])
            if abs(frame - float(event["frame"])) > radius_seconds * event["fps"]:
                continue
            if probability < minimum_probability:
                continue
            if probability < relative_probability * float(event["probability"]):
                continue
            if sources is not None and row["proposal_source"] not in sources:
                continue
            if (
                player_edge_max is not None
                and float(row["nearest_player_edge_distance_norm"]) > player_edge_max
            ):
                continue
            rounded_frame = round(frame, 6)
            by_frame[rounded_frame] = max(probability, by_frame.get(rounded_frame, 0.0))
        return sorted(by_frame.items())

    output = []
    for event in predicted:
        refined = dict(event)
        selected_row = rows[selected_row_index(event)]
        refinements = []
        if event["event_type"] == "contact" and selected_row["proposal_source"] == "serve":
            nearby = hypotheses(
                event,
                radius_seconds=0.24,
                minimum_probability=0.60,
                sources={"trajectory", "bounce_witness"},
            )
            if nearby:
                frame, _ = max(
                    nearby,
                    key=lambda item: (item[1], -abs(item[0] - float(event["frame"]))),
                )
                refined["frame"] = frame
                refinements.append("serve_trajectory")

        if selected_row["proposal_source"] != "serve" and float(event["probability"]) < 0.85:
            nearby = hypotheses(
                refined,
                radius_seconds=0.08,
                minimum_probability=0.80,
                relative_probability=0.90,
            )
            if len(nearby) >= 2 and nearby[-1][0] - nearby[0][0] >= 1.5:
                frames = np.asarray([item[0] for item in nearby])
                weights = np.asarray([item[1] ** 4 for item in nearby])
                refined["frame"] = float(np.average(frames, weights=weights))
                refinements.append("close_consensus")

        if (
            event["event_type"] == "contact"
            and float(event["probability"]) < config.anchor_thresholds["contact"]
        ):
            nearby = hypotheses(
                refined,
                radius_seconds=0.12,
                minimum_probability=0.20,
                player_edge_max=config.contact_player_edge_max,
            )
            if len(nearby) >= 2:
                refined["frame"] = float(np.mean([item[0] for item in nearby]))
                refinements.append("player_local_consensus")

        if event["event_type"] == "contact" and selected_row["proposal_source"] != "serve":
            nearby = hypotheses(
                refined,
                radius_seconds=0.14,
                minimum_probability=0.75,
                relative_probability=0.75,
                player_edge_max=config.contact_player_edge_max,
                sources={"trajectory", "bounce_witness"},
            )
            if len(nearby) >= 2 and nearby[-1][0] - nearby[0][0] >= 1.5:
                refined["frame"] = float(np.mean([item[0] for item in nearby]))
                refinements.append("contact_arc_consensus")

        if event["event_type"] == "bounce":
            by_frame = {}
            sources = set()
            for row_index in rows_by_clip[event["clip"]]:
                row = rows[row_index]
                if not row.get("production_scope"):
                    continue
                if row["proposal_source"] not in {"trajectory", "bounce_witness"}:
                    continue
                frame = float(row["proposal_frame"])
                probability = float(probabilities[row_index, class_index["bounce"]])
                if abs(frame - float(refined["frame"])) > 0.10 * event["fps"]:
                    continue
                if probability < 0.60:
                    continue
                rounded_frame = round(frame, 6)
                by_frame[rounded_frame] = max(probability, by_frame.get(rounded_frame, 0.0))
                sources.add(row["proposal_source"])
            if len(by_frame) >= 2 and sources == {"trajectory", "bounce_witness"}:
                frames = np.asarray(list(by_frame))
                weights = np.asarray([by_frame[frame] ** 2 for frame in frames])
                refined["frame"] = float(np.average(frames, weights=weights))
                refinements.append("bounce_witness_consensus")

        if refinements and abs(float(refined["frame"]) - float(event["frame"])) > 1e-6:
            refined["raw_frame"] = float(event["frame"])
            refined["timing_refinement"] = "|".join(refinements)
        output.append(refined)
    return suppress_same_type_duplicates(output)
