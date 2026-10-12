"""Soft tennis-sequence support for typed ball-event hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

EVENT_TYPES = ("contact", "bounce")


@dataclass(frozen=True)
class EventNode:
    node_id: str
    frame: float
    event_type: str
    unary: float
    side: str | None = None
    fps: float = 50.0


def _reference_gap(left: EventNode, right: EventNode) -> float:
    fps = max((left.fps + right.fps) / 2.0, 1e-6)
    return (right.frame - left.frame) * 50.0 / fps


def transition_score(left: EventNode, right: EventNode) -> float:
    """Score a possible event transition without making rally grammar a hard rule."""
    gap = _reference_gap(left, right)
    if gap <= 2.0 or gap > 180.0:
        return -4.0
    if gap < 5.0:
        timing = -1.0
    elif gap <= 100.0:
        timing = 0.25
    else:
        timing = -0.75 * (gap - 100.0) / 80.0

    type_score = {
        ("contact", "bounce"): 0.55,
        ("bounce", "contact"): 0.65,
        ("contact", "contact"): 0.15,
        ("bounce", "bounce"): -0.35,
    }[(left.event_type, right.event_type)]
    side_score = 0.0
    if (
        left.event_type == right.event_type == "contact"
        and left.side is not None
        and right.side is not None
    ):
        side_score = 0.3 if left.side != right.side else -0.3
    return timing + type_score + side_score


def _transition_allowed(left: EventNode, right: EventNode) -> bool:
    return (left.event_type, right.event_type) in {
        ("contact", "bounce"),
        ("bounce", "contact"),
        ("contact", "contact"),
    }


def complete_bridges(
    anchors: list[EventNode],
    candidates: list[EventNode],
    *,
    max_gap_seconds: float = 2.5,
    exclusion_seconds: float = 0.08,
) -> list[EventNode]:
    """Add only candidates that make both adjacent rally transitions plausible."""
    selected = list(anchors)
    for candidate in sorted(candidates, key=lambda node: node.unary, reverse=True):
        radius = exclusion_seconds * candidate.fps
        if any(abs(node.frame - candidate.frame) <= radius for node in selected):
            continue
        ordered = sorted(selected, key=lambda node: node.frame)
        left = next(
            (node for node in reversed(ordered) if node.frame < candidate.frame),
            None,
        )
        right = next(
            (node for node in ordered if node.frame > candidate.frame),
            None,
        )
        if left is None or right is None:
            continue
        if (
            (candidate.frame - left.frame) / candidate.fps > max_gap_seconds
            or (right.frame - candidate.frame) / candidate.fps > max_gap_seconds
        ):
            continue
        if _transition_allowed(left, candidate) and _transition_allowed(candidate, right):
            selected.append(candidate)
    return sorted(selected, key=lambda node: node.frame)


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return log(probability / (1.0 - probability))


def sequence_features(nodes: list[EventNode]) -> dict[str, dict[str, float]]:
    """Return local graph support features for every hypothesis."""
    ordered = sorted(nodes, key=lambda node: (node.frame, node.event_type))
    output: dict[str, dict[str, float]] = {}
    for index, node in enumerate(ordered):
        features: dict[str, float] = {}
        for direction, neighbors in (
            ("previous", reversed(ordered[:index])),
            ("next", ordered[index + 1 :]),
        ):
            best_by_type = {event_type: -20.0 for event_type in EVENT_TYPES}
            nearest_by_type = {event_type: 181.0 for event_type in EVENT_TYPES}
            for neighbor in neighbors:
                gap = abs(neighbor.frame - node.frame)
                if gap > 180.0:
                    if direction == "previous":
                        break
                    continue
                if gap <= 2.0:
                    continue
                edge = (
                    transition_score(neighbor, node)
                    if direction == "previous"
                    else transition_score(node, neighbor)
                )
                support = _logit(neighbor.unary) + edge
                best_by_type[neighbor.event_type] = max(
                    best_by_type[neighbor.event_type],
                    support,
                )
                nearest_by_type[neighbor.event_type] = min(
                    nearest_by_type[neighbor.event_type],
                    gap,
                )
            for event_type in EVENT_TYPES:
                features[f"{direction}_{event_type}_support"] = best_by_type[event_type]
                features[f"{direction}_{event_type}_gap"] = nearest_by_type[event_type]

        if node.event_type == "bounce":
            bridge = min(
                features["previous_contact_support"],
                features["next_contact_support"],
            )
        else:
            bridge = max(
                min(
                    features["previous_contact_support"],
                    features["next_bounce_support"],
                ),
                min(
                    features["previous_bounce_support"],
                    features["next_contact_support"],
                ),
            )
        features["bridge_support"] = bridge
        output[node.node_id] = features
    return output


def decode_soft_path(
    nodes: list[EventNode],
    *,
    minimum_probability: float = 0.05,
    event_cost: float = 1.5,
    exclusion_radius: float = 2.0,
) -> list[EventNode]:
    """Decode one plausible path while allowing unsupported regions to restart."""
    ordered = sorted(
        (node for node in nodes if node.unary >= minimum_probability),
        key=lambda node: (node.frame, node.event_type),
    )
    if not ordered:
        return []

    scores = [_logit(node.unary) - event_cost for node in ordered]
    previous: list[int | None] = [None] * len(ordered)
    for index, node in enumerate(ordered):
        best_score = scores[index]
        best_previous = None
        for prior_index, prior in enumerate(ordered[:index]):
            if node.frame - prior.frame <= exclusion_radius:
                continue
            candidate = (
                scores[prior_index]
                + transition_score(prior, node)
                + _logit(node.unary)
                - event_cost
            )
            if candidate > best_score:
                best_score = candidate
                best_previous = prior_index
        scores[index] = best_score
        previous[index] = best_previous

    end = max(range(len(ordered)), key=scores.__getitem__)
    if scores[end] <= 0.0:
        return []
    selected = []
    while end is not None:
        selected.append(ordered[end])
        end = previous[end]
    return list(reversed(selected))
