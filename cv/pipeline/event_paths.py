"""Export diverse whole event paths and tentative evidence for downstream S6.

The firm event stream is copied from an existing automatic emission receipt.
Alternative paths are decoded only from frozen classifier rows and grammar
configuration; label files are rejected at this boundary.  Proposal windows
remain tentative and never change the firm path or lattice scores.

Two ranked paths are duplicates when they have the same ordered physical kinds,
contact sides, and ending-kind family, and every corresponding event is within
``timing_tolerance`` frames.  Candidates are considered best-score first, so a
timing-shifted duplicate never consumes one of the K topology slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cv.pipeline.event_grammar_decoder import (
    CLASSES,
    EVENT_TYPES,
    GrammarConfig,
    Node,
    build_nodes,
    nbest_paths,
    path_score,
    viterbi,
    viterbi_exact_side,
)
from cv.pipeline.event_proposals import DEFAULT_PROPOSAL_SOURCES, load_proposal_document
from cv.pipeline import event_time_distribution as timing
from cv.pipeline import event_time_neighborhood as neighborhood
from cv.pipeline import point_context
from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    build_provenance,
    file_record,
    write_provenance,
)

PACKET_SCHEMA = "event_candidate_packet_v2"
INDEX_SCHEMA = "event_candidate_packet_index_v2"
DEFAULT_K = 8
MAX_K = 8
DEFAULT_TIMING_TOLERANCE = 1.0
DEFAULT_TENTATIVE_MARGINAL_FLOOR = 0.3
DEFAULT_TENTATIVE_PROBABILITY_FLOOR = 0.05
DEFAULT_RAW_PATH_LIMIT = 4096
COURT_BOUNDARY_TOLERANCE = 0.02
MAXIMUM_ENDING_CONTEXT_LOG_EVIDENCE = 1.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side_name(side: int) -> str:
    return "far" if side > 0 else "near" if side < 0 else "unknown"


def _same_topology(left: list[dict], right: list[dict], tolerance: float) -> bool:
    """Return whether two ranked event sets differ only by tolerated timing."""

    return len(left) == len(right) and all(
        a["event_type"] == b["event_type"]
        and a["side"] == b["side"]
        and a.get("terminal_kind") == b.get("terminal_kind")
        and abs(float(a["candidate_frame"]) - float(b["candidate_frame"])) <= tolerance
        for a, b in zip(left, right, strict=True)
    )


def _node_marginals(
    nodes: list[Node], config: GrammarConfig
) -> tuple[list[int], np.ndarray, float]:
    if config.exact_side_state:
        return viterbi_exact_side(nodes, config)
    return viterbi(nodes, config)


def _grouped_marginal(nodes: list[Node], marginals: np.ndarray, position: int) -> float:
    node = nodes[position]
    return min(
        1.0,
        sum(
            float(marginals[index])
            for index, other in enumerate(nodes)
            if other.index == node.index and other.event_type == node.event_type
        ),
    )


def _corrected_frame(data: Mapping[str, np.ndarray], row: int) -> float:
    offsets = data.get("time_offset")
    if offsets is None:
        return float(data["frames"][row])
    corrected = float(data["frames"][row]) + float(np.clip(offsets[row], -3.0, 3.0))
    # Match the shipped native-exposure contract: the time head may select a
    # neighbouring exposure, but it never invents a fractional picture.
    return float(round(corrected))


def _event_record(
    node: Node,
    position: int,
    nodes: list[Node],
    marginals: np.ndarray,
    data: Mapping[str, np.ndarray],
    global_rows: np.ndarray,
    timing_tolerance: float,
    court_x: np.ndarray | None,
) -> dict[str, Any]:
    row = int(global_rows[node.index])
    frame = _corrected_frame(data, row)
    probabilities = data["probabilities"][row]
    event = {
        "event_type": node.event_type,
        "candidate_frame": float(node.frame),
        "frame": frame,
        "timing_range": [frame - timing_tolerance, frame + timing_tolerance],
        "side": _side_name(node.side),
        "marginal": _grouped_marginal(nodes, marginals, position),
        "kind_likelihoods": {
            name: float(probabilities[index]) for index, name in enumerate(CLASSES[1:], start=1)
        },
        "classifier_none_likelihood": float(probabilities[0]),
        "source_row": row,
    }
    if court_x is not None:
        x = float(court_x[row])
        event["court_x_fraction"] = x if math.isfinite(x) else None
    y = float(data["court_y"][row])
    event["court_y_fraction"] = y if math.isfinite(y) else None
    if node.terminal:
        event["terminal_kind"] = "second_bounce"
    return event


def _ending_candidates(events: list[dict]) -> list[dict]:
    """Derive conservative ending-kind families from one automatic path."""

    output: list[dict] = []
    for index, event in enumerate(events):
        kind = None
        if event.get("terminal_kind") == "second_bounce":
            kind = "second_bounce"
        elif event["event_type"] == "bounce":
            since_contact = []
            for prior in reversed(events[:index]):
                if prior["event_type"] == "contact":
                    break
                since_contact.append(prior)
            if any(prior["event_type"] == "bounce" for prior in since_contact):
                kind = "second_bounce"
            elif any(prior["event_type"] == "net_hit" for prior in since_contact):
                kind = "ground_after_net_hit"
            else:
                x, y = event.get("court_x_fraction"), event.get("court_y_fraction")
                if x is not None and y is not None:
                    low, high = -COURT_BOUNDARY_TOLERANCE, 1.0 + COURT_BOUNDARY_TOLERANCE
                    if not (low <= x <= high and low <= y <= high):
                        kind = "first_bounce_out"
        elif event["event_type"] == "net_hit" and index == len(events) - 1:
            kind = "net_hit"
        if kind is not None:
            output.append(
                {
                    "kind": kind,
                    "frame": event["frame"],
                    "timing_range": list(event["timing_range"]),
                    "source_event_index": index,
                }
            )
    return output


def _ending_family(kind: str) -> str | None:
    normalized = point_context.normalize_consumer_ending_kind(kind)
    return {
        "out": "first_bounce_out",
        "second_bounce": "second_bounce",
        "net": "net_hit",
        "fov_exit": "fov_exit",
    }.get(normalized)


def apply_ending_context(paths: list[dict], context: Mapping[str, Any] | None) -> list[dict]:
    """Attach bounded soft context evidence without changing firm paths or scores."""

    if context is None:
        return paths
    output = []
    for source in paths:
        row = dict(source)
        try:
            witness = point_context.ending_witness(context, row["events"])
        except (KeyError, TypeError, ValueError) as error:
            row["ending_context_witness"] = {
                "available": False,
                "abstention_reason": f"invalid_point_context:{type(error).__name__}:{error}",
                "hard_gate": False,
            }
            output.append(row)
            continue
        scored = []
        for index, ending in enumerate(row["endings"]):
            event = row["events"][int(ending["source_event_index"])]
            semantic_kinds = [str(ending["kind"])]
            score_or_restart = bool(
                witness.get("scoreboard_winner_used") or witness.get("same_side_serve_restart_used")
            )
            if event["event_type"] == "bounce" and score_or_restart:
                # One observed impact can still be a first out bounce or a
                # second in-court bounce. Keep both semantic hypotheses at the
                # identical native picture; context never invents an event.
                semantic_kinds.extend(["first_bounce_out", "second_bounce"])
            for semantic_kind in dict.fromkeys(semantic_kinds):
                family = _ending_family(semantic_kind)
                likelihood = float(witness["ending_likelihoods"].get(family, 0.0))
                best = max(witness["ending_likelihoods"].values())
                context_penalty = min(
                    MAXIMUM_ENDING_CONTEXT_LOG_EVIDENCE,
                    max(0.0, math.log(max(best, 1e-6) / max(likelihood, 1e-6))),
                )
                fit_log_evidence = math.log(max(float(event.get("marginal", 0.0)), 1e-6))
                scored.append(
                    {
                        **ending,
                        "kind": semantic_kind,
                        "decoder_kind": ending["kind"],
                        "semantic_alternative": semantic_kind != ending["kind"],
                        "context_family": family,
                        "context_likelihood": likelihood,
                        "context_log_penalty": context_penalty,
                        "event_log_evidence": fit_log_evidence,
                        "combined_log_evidence": fit_log_evidence - context_penalty,
                        "hard_gate": False,
                        "source_ending_index": index,
                    }
                )
        selected = (
            max(scored, key=lambda item: (item["combined_log_evidence"], item["frame"]))
            if witness.get("available") and scored
            else None
        )
        row["decoder_ending_kind"] = row["ending_kind"]
        if selected is not None:
            row["ending_kind"] = selected["kind"]
        row["ending_context_witness"] = {
            **witness,
            "maximum_log_evidence": MAXIMUM_ENDING_CONTEXT_LOG_EVIDENCE,
            "hypotheses": scored,
            "selected": selected,
            "firm_path_or_score_changed": False,
        }
        output.append(row)
    return output


def packet_ending_selection(paths: list[dict], timing_tolerance: float) -> dict[str, Any]:
    """Choose a tentative physical ending from observed packet candidates."""

    baseline = next((path for path in paths if path.get("endings")), None)
    if baseline is None:
        return {
            "status": "abstained",
            "abstention_reason": "no_resolved_ending_candidate",
            "hard_gate": False,
            "firm_path_or_score_changed": False,
        }
    anchor = float(baseline["endings"][-1]["frame"])
    strong_context = any(
        (path.get("ending_context_witness") or {}).get("scoreboard_winner_used")
        or (path.get("ending_context_witness") or {}).get("same_side_serve_restart_used")
        for path in paths
    )
    candidates = []
    for path in paths:
        witness = path.get("ending_context_witness") or {}
        for hypothesis in witness.get("hypotheses", []):
            if strong_context or abs(float(hypothesis["frame"]) - anchor) <= timing_tolerance:
                candidates.append({**hypothesis, "path_rank": int(path["rank"])})
    selected = (
        max(candidates, key=lambda row: (row["combined_log_evidence"], -row["path_rank"]))
        if candidates
        else None
    )
    return {
        "status": "selected" if selected is not None else "context_unavailable",
        "decoder_anchor_frame": anchor,
        "anchor_source_path_rank": int(baseline["rank"]),
        "decoder_ending_kind": baseline.get("decoder_ending_kind", baseline["ending_kind"]),
        "ending_kind": (selected["kind"] if selected is not None else baseline.get("ending_kind")),
        "selected": selected,
        "candidates": sorted(
            candidates, key=lambda row: (-row["combined_log_evidence"], row["path_rank"])
        ),
        "hard_gate": False,
        "firm_path_or_score_changed": False,
        "native_timing_invented": False,
        "timing_tolerance_frames": timing_tolerance,
        "score_or_restart_can_move_terminal_picture": strong_context,
    }


def load_point_contexts(root: Path | None) -> dict[str, dict]:
    """Load explicit per-point priors; malformed rows remain unavailable."""

    if root is None:
        return {}
    contexts: dict[str, dict] = {}
    for path in sorted((root / "points").glob("*/*.json")):
        try:
            document = json.loads(path.read_text())
            point_context.validate_prior(document)
            clip = f"{document['match_id']}__{document['point_id']}"
            document = {**document, "consumer_source": file_record(path, role="point_context")}
            contexts[clip] = document
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return contexts


def diverse_paths(
    probabilities: np.ndarray,
    frames: np.ndarray,
    court_y: np.ndarray,
    global_rows: np.ndarray,
    clip: str,
    data: Mapping[str, np.ndarray],
    *,
    config: GrammarConfig | None = None,
    count: int = DEFAULT_K,
    timing_tolerance: float = DEFAULT_TIMING_TOLERANCE,
    raw_path_limit: int = DEFAULT_RAW_PATH_LIMIT,
    court_x: np.ndarray | None = None,
) -> tuple[list[dict], list[Node], np.ndarray]:
    """Return top-K score-ranked paths after topology/timing de-duplication."""

    if not 1 <= count <= MAX_K:
        raise ValueError(f"path count must be between 1 and {MAX_K}")
    if timing_tolerance < 0:
        raise ValueError("timing tolerance must be non-negative")
    resolved = (config or GrammarConfig()).for_clip(clip)
    nodes = build_nodes(probabilities, frames, court_y, config=resolved, clip=clip)
    best_path, marginals, best_score = _node_marginals(nodes, resolved)
    best_start = nodes[best_path[0]].frame if best_path else None
    best_end = nodes[best_path[-1]].frame if best_path else None
    requested = min(max(64, count * 8), raw_path_limit)
    accepted: list[dict] = []
    while True:
        pool = {tuple(path): score for score, path in nbest_paths(nodes, resolved, count=requested)}
        # A raw list-Viterbi beam can spend thousands of entries on small
        # combinations of the same strong nodes before it reaches a materially
        # different event.  Add the best path forced through every off-best
        # node with nontrivial posterior, and the best path with each best-path
        # observation removed.  All candidates are then re-scored under the
        # unmodified grammar, so this broadens topology coverage without giving
        # a proposal or diversity heuristic hidden score.
        chosen = set(best_path)
        boost = 1_000.0
        off_path = []
        seen_observations = set()
        for target in sorted(
            range(len(nodes)), key=lambda position: float(marginals[position]), reverse=True
        ):
            node = nodes[target]
            observation = (node.index, node.event_type)
            if target in chosen or node.terminal or observation in seen_observations:
                continue
            seen_observations.add(observation)
            off_path.append(target)
            if len(off_path) == 12:
                break
        for target in off_path:
            forced_nodes = list(nodes)
            forced_nodes[target] = replace(nodes[target], gain=nodes[target].gain + boost)
            forced, _marginals, _score = _node_marginals(forced_nodes, resolved)
            if target in forced:
                pool[tuple(forced)] = path_score(nodes, forced, resolved)
            for omitted in chosen:
                if omitted == target:
                    continue
                altered = [
                    replace(node, gain=node.gain + boost) if position == target else node
                    for position, node in enumerate(nodes)
                    if position != omitted
                ]
                original = [position for position in range(len(nodes)) if position != omitted]
                candidate, _marginals, _score = _node_marginals(altered, resolved)
                mapped = [original[position] for position in candidate]
                if target in mapped:
                    pool[tuple(mapped)] = path_score(nodes, mapped, resolved)
        for left_index, left in enumerate(off_path):
            for right in off_path[left_index + 1 :]:
                forced_nodes = [
                    replace(node, gain=node.gain + boost) if position in {left, right} else node
                    for position, node in enumerate(nodes)
                ]
                forced, _marginals, _score = _node_marginals(forced_nodes, resolved)
                if left in forced and right in forced:
                    pool[tuple(forced)] = path_score(nodes, forced, resolved)
        for omitted in chosen:
            remaining = [position for position in range(len(nodes)) if position != omitted]
            reduced = [nodes[position] for position in remaining]
            alternative, _marginals, _score = _node_marginals(reduced, resolved)
            mapped = [remaining[position] for position in alternative]
            if mapped:
                pool[tuple(mapped)] = path_score(nodes, mapped, resolved)
        # Ending presence is part of topology. Materialise every bounce/net
        # prefix of the candidate pool so an earlier physical ending competes
        # as its own whole sequence instead of being hidden inside a longer
        # profitable continuation.
        for candidate in tuple(pool):
            for offset, position in enumerate(candidate, start=1):
                if nodes[position].event_type not in {"bounce", "net_hit"}:
                    continue
                prefix = candidate[:offset]
                pool.setdefault(prefix, path_score(nodes, prefix, resolved))
        raw = sorted(((score, list(path)) for path, score in pool.items()), reverse=True)
        accepted = []
        for score, positions in raw:
            # These are whole candidate streams, not arbitrary profitable
            # suffixes. Every path must cover the automatic stream's start. A
            # path may end earlier only on a physical bounce/net candidate;
            # that is precisely the ending-kind alternative the endpoint prior
            # is allowed to price.
            if best_start is not None and (
                nodes[positions[0]].frame > best_start + timing_tolerance
                or (
                    nodes[positions[-1]].frame < best_end - timing_tolerance
                    and nodes[positions[-1]].event_type == "contact"
                )
            ):
                continue
            events = [
                _event_record(
                    nodes[position],
                    position,
                    nodes,
                    marginals,
                    data,
                    global_rows,
                    timing_tolerance,
                    court_x,
                )
                for position in positions
            ]
            # Terminal and ordinary bounce states are one physical observation.
            collapsed: list[dict] = []
            for event in events:
                if (
                    collapsed
                    and collapsed[-1]["event_type"] == event["event_type"]
                    and collapsed[-1]["source_row"] == event["source_row"]
                ):
                    if event.get("terminal_kind"):
                        collapsed[-1]["terminal_kind"] = event["terminal_kind"]
                    continue
                collapsed.append(event)
            if any(
                _same_topology(collapsed, item["events"], timing_tolerance) for item in accepted
            ):
                continue
            endings = _ending_candidates(collapsed)
            accepted.append(
                {
                    "rank": len(accepted) + 1,
                    "log_score": float(score),
                    "log_score_gap": 0.0,
                    "events": collapsed,
                    "endings": endings,
                    "ending_kind": endings[-1]["kind"] if endings else "unresolved",
                }
            )
            if len(accepted) == count:
                break
        if accepted:
            # ``raw`` also contains profitable suffixes rejected by the
            # whole-stream coverage contract. Rank and price only admissible
            # paths, so rank one is the zero-gap reference by construction.
            reference_score = float(accepted[0]["log_score"])
            for item in accepted:
                item["log_score_gap"] = reference_score - float(item["log_score"])
        if len(accepted) >= count or len(raw) < requested or requested >= raw_path_limit:
            break
        requested = min(raw_path_limit, requested * 2)
    return accepted, nodes, marginals


def _proposal_likelihoods(
    proposal,
    mask: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], str]:
    local = [
        int(row)
        for row in mask
        if proposal.start_frame <= float(data["frames"][row]) <= proposal.end_frame
    ]
    if local:
        return (
            {
                kind: float(
                    max(data["probabilities"][row, EVENT_TYPES.index(kind) + 1] for row in local)
                )
                if kind in proposal.kinds
                else 0.0
                for kind in EVENT_TYPES
            },
            "maximum_frozen_classifier_likelihood_in_proposal_window",
        )
    share = proposal.confidence / len(proposal.kinds)
    return (
        {kind: float(share if kind in proposal.kinds else 0.0) for kind in EVENT_TYPES},
        "uncalibrated_proposal_compatibility_share_no_classifier_row",
    )


def _classifier_event_record(
    row: int,
    event_type: str,
    data: Mapping[str, np.ndarray],
    *,
    timing_range: list[float] | None = None,
    source: str,
    reason: str,
    court_x: np.ndarray | None = None,
) -> dict[str, Any]:
    """Expose one frozen classifier row without promoting it into the lattice."""

    frame = float(data["frames"][row])
    probabilities = data["probabilities"][row]
    event = {
        "event_type": event_type,
        "candidate_frame": frame,
        # This tier is explicitly the observed crop row.  The time head remains
        # available as evidence, but it must not move or invent that exposure.
        "frame": frame,
        "timing_range": timing_range or [frame, frame],
        "side": "unknown",
        "classifier_probability": float(probabilities[CLASSES.index(event_type)]),
        "classifier_time_offset": (
            float(data["time_offset"][row]) if "time_offset" in data else None
        ),
        "kind_likelihoods": {
            name: float(probabilities[index]) for index, name in enumerate(CLASSES[1:], start=1)
        },
        "classifier_none_likelihood": float(probabilities[0]),
        "source_row": int(row),
        "source": source,
        "reason": reason,
    }
    if court_x is not None:
        x = float(court_x[row])
        event["court_x_fraction"] = x if math.isfinite(x) else None
    y = float(data["court_y"][row])
    event["court_y_fraction"] = y if math.isfinite(y) else None
    return event


def tentative_events(
    clip: str,
    paths: list[dict],
    nodes: list[Node],
    marginals: np.ndarray,
    global_rows: np.ndarray,
    data: Mapping[str, np.ndarray],
    proposals: list,
    *,
    marginal_floor: float,
    probability_floor: float = DEFAULT_TENTATIVE_PROBABILITY_FLOOR,
    timing_tolerance: float,
    grammar: GrammarConfig | None = None,
    court_x: np.ndarray | None = None,
) -> list[dict]:
    """Return label-free evidence that is never inserted into the firm lattice."""

    if not 0.0 <= marginal_floor <= 1.0:
        raise ValueError("tentative marginal floor must lie in [0, 1]")
    if not 0.0 <= probability_floor <= 1.0:
        raise ValueError("tentative probability floor must lie in [0, 1]")
    path_rows = {
        (event["source_row"], event["event_type"]) for path in paths for event in path["events"]
    }
    output = []
    native_candidate_keys = {
        (int(event["source_row"]), str(event["event_type"]), float(event["frame"]))
        for path in paths
        for event in path["events"]
    }
    grouped: set[tuple[int, str]] = set()
    for position, node in enumerate(nodes):
        global_key = (int(global_rows[node.index]), node.event_type)
        key = global_key
        marginal = _grouped_marginal(nodes, marginals, position)
        if key in grouped or global_key in path_rows or marginal < marginal_floor:
            continue
        grouped.add(key)
        event = _event_record(
            node,
            position,
            nodes,
            marginals,
            data,
            global_rows,
            timing_tolerance,
            court_x,
        )
        event.update(
            {
                "source": "decoder_lattice",
                "reason": "off_path_node_above_marginal_floor",
            }
        )
        output.append(event)
        native_candidate_keys.add(
            (int(event["source_row"]), str(event["event_type"]), float(event["frame"]))
        )

    # The shipped default deliberately relaxes local-maximum suppression only
    # in the tentative tier.  Firm nodes, paths, marginals and scores above are
    # already frozen.  A row enters on its own classifier probability and keeps
    # the native crop frame rather than applying the time-head correction.
    # Candidate membership depends only on the explicit floor and relaxed
    # local-maximum flag; the nodes never enter a scored path.
    tentative_nodes = build_nodes(
        data["probabilities"][global_rows],
        data["frames"][global_rows],
        data["court_y"][global_rows],
        config=replace(grammar or GrammarConfig(), terminal_second_bounce=False),
        clip=clip,
        tentative_probability_floor=probability_floor,
        relax_local_maximum=True,
    )
    for node in tentative_nodes:
        row = int(global_rows[node.index])
        native_frame = float(data["frames"][row])
        if (row, node.event_type, native_frame) in native_candidate_keys:
            continue
        output.append(
            _classifier_event_record(
                row,
                node.event_type,
                data,
                source="classifier_probability",
                reason="per_row_probability_above_tentative_floor_local_maximum_relaxed",
                court_x=court_x,
            )
        )
        native_candidate_keys.add((row, node.event_type, native_frame))

    mask = global_rows
    for proposal in proposals:
        if proposal.source not in DEFAULT_PROPOSAL_SOURCES:
            continue
        uncovered_kinds = [
            kind
            for kind in proposal.kinds
            if not any(
                event["event_type"] == kind
                and event["timing_range"][0] <= proposal.end_frame
                and proposal.start_frame <= event["timing_range"][1]
                for path in paths
                for event in path["events"]
            )
        ]
        if not uncovered_kinds:
            continue
        likelihoods, source = _proposal_likelihoods(proposal, mask, data)
        local = [
            int(row)
            for row in mask
            if proposal.start_frame <= float(data["frames"][row]) <= proposal.end_frame
        ]
        if local:
            for kind in uncovered_kinds:
                row = max(
                    local,
                    key=lambda item: float(data["probabilities"][item, CLASSES.index(kind)]),
                )
                existing = next(
                    (
                        item
                        for item in output
                        if item.get("source") == "proposal_classifier_peak"
                        and item.get("source_row") == row
                        and item.get("event_type") == kind
                    ),
                    None,
                )
                if existing is not None:
                    existing["timing_range"] = [
                        min(float(existing["timing_range"][0]), proposal.start_frame),
                        max(float(existing["timing_range"][1]), proposal.end_frame),
                    ]
                    existing.setdefault("proposal_sources", []).append(proposal.source)
                    continue
                event = _classifier_event_record(
                    row,
                    kind,
                    data,
                    timing_range=[proposal.start_frame, proposal.end_frame],
                    source="proposal_classifier_peak",
                    reason="classifier_kind_peak_inside_label_free_proposal_window",
                    court_x=court_x,
                )
                event.update(
                    {
                        "proposal_confidence": proposal.confidence,
                        "proposal_source": proposal.source,
                        "proposal_sources": [proposal.source],
                        "likelihood_source": source,
                        "evidence": proposal.evidence,
                    }
                )
                output.append(event)
            continue
        likelihoods = {
            kind: probability if kind in uncovered_kinds else 0.0
            for kind, probability in likelihoods.items()
        }
        output.append(
            {
                "event_type": None,
                "candidate_frame": proposal.frame,
                "frame": proposal.frame,
                "timing_range": [proposal.start_frame, proposal.end_frame],
                "side": "unknown",
                "marginal": max(likelihoods.values(), default=0.0),
                "proposal_confidence": proposal.confidence,
                "kind_likelihoods": likelihoods,
                "likelihood_source": source,
                "source": proposal.source,
                "reason": f"label_free_{proposal.source}",
                "evidence": proposal.evidence,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            float(row["timing_range"][0]),
            str(row.get("event_type")),
            str(row["source"]),
        ),
    )


def lattice_events(
    nodes: list[Node],
    marginals: np.ndarray,
    global_rows: np.ndarray,
    data: Mapping[str, np.ndarray],
    *,
    timing_tolerance: float,
    court_x: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Expose every unique physical lattice node for downstream miss attribution."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for position, node in enumerate(nodes):
        key = (int(global_rows[node.index]), node.event_type)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            _event_record(
                node,
                position,
                nodes,
                marginals,
                data,
                global_rows,
                timing_tolerance,
                court_x,
            )
        )
    return sorted(output, key=lambda row: (float(row["frame"]), row["event_type"]))


def _load_rows(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    rows = document.get("emissions") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("firm emission artifact has no emission list")
    return rows


def _load_grammar(path: Path | None) -> GrammarConfig:
    if path is None:
        return GrammarConfig()
    source = json.loads(path.read_text())
    if source.get("labels_or_reviewed_inputs"):
        raise ValueError("automatic grammar configuration carries human-derived inputs")
    values = dict(source.get("grammar", source))
    transition = values.get("transition_log_prior")
    if transition:
        values["transition_log_prior"] = {
            tuple(key.split("->", 1)): float(value) for key, value in transition.items()
        }
    minimum_gap = values.get("minimum_gap")
    if minimum_gap:
        values["minimum_gap"] = {
            tuple(key.split("->", 1)): int(value) for key, value in minimum_gap.items()
        }
    return GrammarConfig(**values)


def _grammar_document(config: GrammarConfig) -> dict[str, Any]:
    """Serialize every score-affecting value, including per-point side maps."""

    resolved = config.resolved()
    document = resolved.as_dict()
    if isinstance(resolved.net_line, Mapping):
        document["net_line"] = dict(resolved.net_line)
    if isinstance(resolved.side_dead_band, Mapping):
        document["side_dead_band"] = dict(resolved.side_dead_band)
    return document


def classifier_source_evidence(
    data: Mapping[str, np.ndarray], rows: np.ndarray, emissions: list[dict]
) -> list[dict]:
    """Preserve row scores and exact decoded refusals, without inferring admission.

    A crop row's none score is not a no-event probability for a whole span.
    Overlapping rows remain correlated evidence, never independent votes.

    The local timing softmax is preserved verbatim when the prediction artifact
    carries it.  Older artifacts only ever stored its mean and its max, and are
    reported as unavailable rather than having a distribution invented for them.
    """
    distribution = timing.arrays(data, rows=len(data["frames"]))
    result = []
    for source_row in rows:
        i = int(source_row)
        values = np.asarray(data["probabilities"][i], dtype=float)
        if (
            values.shape != (len(CLASSES),)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError("finite four-class prediction scores required")
        scores = dict(zip(CLASSES, values.tolist(), strict=True))
        frame = float(data["frames"][i])
        matches = [
            j
            for j, emission in enumerate(emissions)
            if emission.get("candidate_frame") == frame
            and emission.get("class_probabilities") == scores
        ]
        record = dict(
            source_row=i,
            native_crop_frame=frame,
            class_probabilities=scores,
            raw_emission_indices=matches,
            decoder_evidence_status="available" if matches else "not_decoded_or_unmatched",
            time_distribution_status=timing.status(data),
            time_distribution_is_calibrated=False,
        )
        if distribution is not None:
            record["time_distribution"] = timing.record(distribution[0][i], distribution[1][i])
        for field in (
            "time_offset",
            "time_confidence",
            "offset",
            "xy",
            "centres",
            "track_observed",
            "source",
        ):
            if field in data:
                record[field] = np.asarray(data[field][i]).tolist()
        result.append(record)
    return result


def check_emission_neighborhoods(rows: list[dict], clip: str, data: Mapping[str, np.ndarray]):
    """Validate any timing neighborhood an emission row carries, against its source.

    Older producers wrote no neighborhood at all and stay readable; one that is
    present has to be the clip's own available rows around the row's original
    candidate frame, on the exact model timing grid.
    """

    for row in rows:
        block = row.get(neighborhood.KEY)
        if block is None:
            continue
        neighborhood.validate(
            block,
            clip=clip,
            candidate_frame=float(row["candidate_frame"]),
            classes=CLASSES,
            clips=data.get("clips"),
            frames=data.get("frames"),
        )


def load_evidence_packet(path: Path, emissions_path: Path, clip: str) -> dict:
    """Check a v2 sidecar's ancestry and firm stream; never change events."""
    packet = json.loads(path.read_text())
    if packet.get("schema") != PACKET_SCHEMA or packet.get("clip") != clip:
        raise ValueError("candidate sidecar schema or clip mismatch")
    if packet.get("labels_or_reviewed_inputs") or not packet.get("configuration", {}).get(
        "evidence_only"
    ):
        raise ValueError("automatic evidence-only candidate packet required")
    inputs = packet.get("source_inputs", {})
    for name in ("predictions", "emissions"):
        source = inputs.get(name, {})
        resolved = Path(source.get("resolved_path", ""))
        if not resolved.is_file() or _sha256(resolved) != source.get("record", {}).get("sha256"):
            raise ValueError(f"candidate {name} ancestry changed")
    if inputs["emissions"]["record"]["sha256"] != _sha256(emissions_path):
        raise ValueError("candidate sidecar binds a different automatic event stream")
    source_rows = [
        row
        for row in _load_rows(emissions_path)
        if (
            str(row.get("clip"))
            if "__" in str(row.get("clip"))
            else f"{row.get('match_id')}__{row.get('clip')}"
        )
        == clip
    ]
    firm = sorted(
        (row for row in source_rows if not row.get("abstain", False)),
        key=lambda row: (float(row["frame"]), str(row.get("event_type"))),
    )
    if packet.get("raw_emissions") != source_rows or packet.get("firm") != firm:
        raise ValueError("candidate sidecar changed firm events or original refusals")
    with np.load(inputs["predictions"]["resolved_path"], allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    mask = np.flatnonzero(data["clips"] == clip)
    if packet.get("source_evidence") != classifier_source_evidence(data, mask, source_rows):
        raise ValueError("candidate sidecar changed source classifier evidence")
    check_emission_neighborhoods(source_rows, clip, data)
    if any(packet.get(field) for field in ("paths", "lattice")):
        raise ValueError("evidence-only sidecar cannot carry decoded or fitted proposals")
    expected_proposals = []
    if "proposals" in inputs:
        source = inputs["proposals"]
        resolved = Path(source.get("resolved_path", ""))
        if not resolved.is_file() or _sha256(resolved) != source.get("record", {}).get("sha256"):
            raise ValueError("candidate independent proposal ancestry changed")
        expected_proposals = [
            row.as_dict() for row in load_proposal_document(resolved) if row.clip == clip
        ]
    if packet.get("proposal_windows", []) != expected_proposals:
        raise ValueError("candidate changed independent source proposals")
    source_rows_by_index = set(map(int, mask))
    for candidate in packet.get("tentative", []):
        row = candidate.get("source_row")
        kind = candidate.get("event_type")
        if row not in source_rows_by_index or kind not in EVENT_TYPES:
            raise ValueError("candidate has no matching classifier source row")
        expected = _classifier_event_record(
            row,
            kind,
            data,
            source="classifier_probability",
            reason="per_row_probability_above_tentative_floor_local_maximum_relaxed",
        )
        if candidate != expected:
            raise ValueError("candidate changed native classifier evidence")
    return packet


def build_packets(
    predictions_path: Path,
    firm_emissions_path: Path,
    proposals_path: Path | None,
    output_dir: Path,
    *,
    grammar: GrammarConfig | None = None,
    grammar_source: Path | None = None,
    crop_manifest: Path | None = None,
    active_play: Path | None = None,
    count: int = DEFAULT_K,
    timing_tolerance: float = DEFAULT_TIMING_TOLERANCE,
    tentative_marginal_floor: float = DEFAULT_TENTATIVE_MARGINAL_FLOOR,
    tentative_probability_floor: float = DEFAULT_TENTATIVE_PROBABILITY_FLOOR,
    raw_path_limit: int = DEFAULT_RAW_PATH_LIMIT,
    point_context_root: Path | None = None,
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Write one automatic candidate packet per point plus a provenance index."""

    if evidence_only and any(
        source is not None for source in (crop_manifest, active_play, point_context_root)
    ):
        raise ValueError(
            "evidence-only export accepts classifier, emission and independent proposal sources only"
        )
    loaded = np.load(predictions_path, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    required = {"probabilities", "frames", "clips", "broadcasts", "court_y"}
    if missing := required - set(data):
        raise ValueError(f"prediction artifact misses {sorted(missing)}")
    rows = len(data["frames"])
    # Validate optional timing evidence for every export mode, even if no
    # physical candidate will be emitted from this artifact.
    timing.arrays(data, rows=rows)
    if any(len(data[name]) != rows for name in required if name != "probabilities"):
        raise ValueError("prediction arrays are not row-aligned")
    court_x = None
    if crop_manifest is not None:
        manifest = json.loads(crop_manifest.read_text())
        index = manifest.get("rows_index")
        if not isinstance(index, list) or len(index) != rows:
            raise ValueError("crop manifest rows do not align with frozen predictions")
        court_x = np.asarray(
            [math.nan if row.get("court_x") is None else float(row["court_x"]) for row in index]
        )
    firm = _load_rows(firm_emissions_path)
    if any(row.get("labels_or_reviewed_inputs") for row in firm):
        raise ValueError("firm emission row carries reviewed inputs")
    if proposals_path is None and not evidence_only:
        raise ValueError("proposal input required unless exporting classifier evidence only")
    proposals = load_proposal_document(proposals_path) if proposals_path is not None else []
    proposals_by_clip: dict[str, list] = defaultdict(list)
    for proposal in proposals:
        proposals_by_clip[proposal.clip].append(proposal)
    firm_by_clip: dict[str, list[dict]] = defaultdict(list)
    raw_by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in firm:
        clip = str(row.get("clip"))
        if "__" not in clip and row.get("match_id"):
            clip = f"{row['match_id']}__{clip}"
        raw_by_clip[clip].append(row)
        if not row.get("abstain", False):
            firm_by_clip[clip].append(row)
    active_by_clip: dict[str, dict[str, Any]] = {}
    if active_play is not None:
        active_document = json.loads(active_play.read_text())
        if active_document.get("labels_or_reviewed_inputs"):
            raise ValueError("active-play artifact carries reviewed inputs")
        rows_document = active_document.get("rows", active_document)
        if not isinstance(rows_document, dict):
            raise ValueError("active-play artifact has no point mapping")
        for key, value in rows_document.items():
            if not isinstance(value, dict):
                continue
            match_id, _, point = str(key).partition("/")
            active_by_clip[f"{match_id}__{point}"] = value
    contexts_by_clip = load_point_contexts(point_context_root)
    clips = sorted(set(map(str, data["clips"])) | set(proposals_by_clip) | set(firm_by_clip))
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_records = []
    resolved_grammar = grammar or GrammarConfig()
    for clip in clips:
        mask = np.flatnonzero(data["clips"] == clip)
        paths: list[dict] = []
        nodes: list[Node] = []
        marginals = np.zeros(0)
        if len(mask) and not evidence_only:
            paths, nodes, marginals = diverse_paths(
                data["probabilities"][mask],
                data["frames"][mask],
                data["court_y"][mask],
                mask,
                clip,
                data,
                config=resolved_grammar,
                count=count,
                timing_tolerance=timing_tolerance,
                raw_path_limit=raw_path_limit,
                court_x=court_x,
            )
        tentative = tentative_events(
            clip,
            paths,
            nodes,
            marginals,
            mask,
            data,
            [] if evidence_only else proposals_by_clip.get(clip, []),
            marginal_floor=tentative_marginal_floor,
            probability_floor=tentative_probability_floor,
            timing_tolerance=timing_tolerance,
            grammar=resolved_grammar,
            court_x=court_x,
        )
        paths = apply_ending_context(paths, contexts_by_clip.get(clip))
        ending_selection = packet_ending_selection(paths, timing_tolerance)
        lattice = lattice_events(
            nodes,
            marginals,
            mask,
            data,
            timing_tolerance=timing_tolerance,
            court_x=court_x,
        )
        match_id, _, point = clip.rpartition("__")
        packet = {
            "schema": PACKET_SCHEMA,
            "clip": clip,
            "match_id": match_id,
            "point": point,
            "labels_or_reviewed_inputs": [],
            "firm": sorted(
                firm_by_clip.get(clip, []),
                key=lambda row: (float(row["frame"]), str(row.get("event_type"))),
            ),
            "paths": paths,
            "tentative": tentative,
            "lattice": lattice,
            "proposal_windows": [
                proposal.as_dict()
                for proposal in proposals_by_clip.get(clip, [])
                if evidence_only or proposal.source in DEFAULT_PROPOSAL_SOURCES
            ],
            "point_context": contexts_by_clip.get(clip),
            "tentative_ending_witnesses": [
                {
                    "path_rank": path["rank"],
                    "ending_kind": path["ending_kind"],
                    "witness": path.get("ending_context_witness"),
                }
                for path in paths
                if path.get("ending_context_witness") is not None
            ],
            "tentative_ending_selection": ending_selection,
            "active_play": {
                "active_spans": active_by_clip.get(clip, {}).get("active_spans", []),
                "event_spans": active_by_clip.get(clip, {}).get("event_spans", []),
                "gate_held": active_by_clip.get(clip, {}).get("gate_held"),
            },
            "configuration": {
                "k": count,
                "timing_tolerance_frames": timing_tolerance,
                "tentative_marginal_floor": tentative_marginal_floor,
                "tentative_probability_floor": tentative_probability_floor,
                "tentative_local_maximum_suppression": "relaxed",
                "proposal_classifier_peaks": True,
                "proposal_sources": list(DEFAULT_PROPOSAL_SOURCES),
                "proposal_effect_on_firm": "none",
                "dedupe_rule": (
                    "best-score-first; same ordered kind, contact/court side and ending kind, "
                    "with every corresponding candidate frame within timing_tolerance"
                ),
            },
        }
        if evidence_only:
            packet["source_evidence"] = classifier_source_evidence(
                data, mask, raw_by_clip.get(clip, [])
            )
            packet["raw_emissions"] = raw_by_clip.get(clip, [])
            check_emission_neighborhoods(packet["raw_emissions"], clip, data)
            packet["source_inputs"] = {
                name: {"record": file_record(source), "resolved_path": str(source.resolve())}
                for name, source in (
                    ("predictions", predictions_path),
                    ("emissions", firm_emissions_path),
                )
            }
            if proposals_path is not None:
                packet["source_inputs"]["proposals"] = {
                    "record": file_record(proposals_path),
                    "resolved_path": str(proposals_path.resolve()),
                }
            packet["configuration"].update(
                evidence_only=True,
                decoded_alternative_paths=False,
                candidate_status="retained_hypotheses_not_physical_occurrence",
                timing_scores="predictive_scores_not_calibrated_confidence",
            )
        name = f"{clip}.event_candidates.json"
        path = output_dir / name
        path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        packet_records.append(
            {
                "clip": clip,
                "path": name,
                "paths": len(paths),
                "tentative": len(tentative),
                "lattice": len(lattice),
                "proposal_windows": len(packet["proposal_windows"]),
                "firm": len(packet["firm"]),
                "sha256": _sha256(path),
            }
        )
    index = {
        "schema": INDEX_SCHEMA,
        "labels_or_reviewed_inputs": [],
        "inputs": {
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "firm_emissions": str(firm_emissions_path.resolve()),
            "firm_emissions_sha256": _sha256(firm_emissions_path),
            "proposals": str(proposals_path.resolve()) if proposals_path else None,
            "proposals_sha256": _sha256(proposals_path) if proposals_path else None,
            "crop_manifest": str(crop_manifest.resolve()) if crop_manifest else None,
            "crop_manifest_sha256": _sha256(crop_manifest) if crop_manifest else None,
            "grammar": str(grammar_source.resolve()) if grammar_source else None,
            "grammar_sha256": _sha256(grammar_source) if grammar_source else None,
            "active_play": str(active_play.resolve()) if active_play else None,
            "active_play_sha256": _sha256(active_play) if active_play else None,
            "point_context_manifest": (
                str((point_context_root / "manifest.json").resolve())
                if point_context_root is not None
                else None
            ),
            "point_context_manifest_sha256": (
                _sha256(point_context_root / "manifest.json")
                if point_context_root is not None
                and (point_context_root / "manifest.json").is_file()
                else None
            ),
        },
        "grammar": _grammar_document(resolved_grammar),
        "packets": packet_records,
    }
    index_path = output_dir / "manifest.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    reused = [
        file_record(predictions_path, role="classifier_predictions"),
        file_record(firm_emissions_path, role="firm_event_emissions"),
    ]
    if proposals_path is not None:
        reused.append(file_record(proposals_path, role="label_free_event_proposals"))
    for path, role in (
        (crop_manifest, "event_crop_manifest"),
        (grammar_source, "event_grammar"),
        (active_play, "active_play_spans"),
        (
            point_context_root / "manifest.json"
            if point_context_root is not None and (point_context_root / "manifest.json").is_file()
            else None,
            "point_context_run",
        ),
    ):
        if path is not None:
            reused.append(file_record(path, role=role))
    models = []
    firm_manifest_path = firm_emissions_path.with_suffix(".manifest.json")
    if firm_manifest_path.exists():
        firm_manifest = json.loads(firm_manifest_path.read_text())
        checkpoint_record = (firm_manifest.get("calibration_identity") or {}).get("checkpoint")
        if checkpoint_record and checkpoint_record.get("sha256"):
            models.append(checkpoint_record)
        reused.append(file_record(firm_manifest_path, role="firm_event_run_receipt"))
    provenance = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=AUTOMATIC_MODE,
        models=models,
        configuration={
            "schema": PACKET_SCHEMA,
            "k": count,
            "timing_tolerance_frames": timing_tolerance,
            "tentative_marginal_floor": tentative_marginal_floor,
            "tentative_probability_floor": tentative_probability_floor,
            "tentative_local_maximum_suppression": "relaxed",
            "proposal_classifier_peaks": True,
            "raw_path_limit": raw_path_limit,
            "proposal_sources": list(DEFAULT_PROPOSAL_SOURCES),
            "grammar": _grammar_document(resolved_grammar),
            "point_context": {
                "supplied": point_context_root is not None,
                "usable_points": len(contexts_by_clip),
                "maximum_ending_log_evidence": MAXIMUM_ENDING_CONTEXT_LOG_EVIDENCE,
                "effect": "tentative ending selection only; firm paths and scores unchanged",
            },
        },
        reused_artifacts=reused,
        fallbacks=[
            {"name": "missing_classifier_rows", "action": "proposal-only tentative packet"},
            {
                "name": "proposal_without_classifier_row",
                "action": "uncalibrated compatibility share; never firm",
            },
            {"name": "unknown_ending", "action": "unresolved; downstream must abstain"},
        ],
    )
    provenance["outputs"] = {
        "packet_manifest": file_record(index_path, role="event_candidate_packet_index"),
        "packets": len(packet_records),
        "paths": sum(row["paths"] for row in packet_records),
    }
    write_provenance(output_dir / "provenance.json", provenance)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--firm-emissions", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="retain classifier alternatives and original refusals without decoding new paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grammar", type=Path)
    parser.add_argument("--crop-manifest", type=Path)
    parser.add_argument("--active-play", type=Path)
    parser.add_argument("--point-context-root", type=Path)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--timing-tolerance", type=float, default=DEFAULT_TIMING_TOLERANCE)
    parser.add_argument(
        "--tentative-marginal-floor", type=float, default=DEFAULT_TENTATIVE_MARGINAL_FLOOR
    )
    parser.add_argument(
        "--tentative-probability-floor",
        type=float,
        default=DEFAULT_TENTATIVE_PROBABILITY_FLOOR,
        help="per-kind classifier floor for relaxed tentative rows; firm paths are unchanged",
    )
    parser.add_argument("--raw-path-limit", type=int, default=DEFAULT_RAW_PATH_LIMIT)
    args = parser.parse_args()
    result = build_packets(
        args.predictions,
        args.firm_emissions,
        args.proposals,
        args.output_dir,
        grammar=_load_grammar(args.grammar),
        grammar_source=args.grammar,
        crop_manifest=args.crop_manifest,
        active_play=args.active_play,
        count=args.k,
        timing_tolerance=args.timing_tolerance,
        tentative_marginal_floor=args.tentative_marginal_floor,
        tentative_probability_floor=args.tentative_probability_floor,
        raw_path_limit=args.raw_path_limit,
        point_context_root=args.point_context_root,
        evidence_only=args.evidence_only,
    )
    print(json.dumps({"packets": len(result["packets"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
