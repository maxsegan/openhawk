"""Soft point-context evidence for choosing an S6 physical ending hypothesis.

This module changes only ending selection. It never changes the fit envelope,
event windows, terminal rebound, serve model, or any acceptance gate. A context
likelihood contributes at most ``MAXIMUM_LOG_PENALTY`` nats beside the fitted
hypothesis's own input-only score, so contradictory fit evidence can always win.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cv.pipeline import point_context


SCHEMA = "connected_ending_witness_v1"
MAXIMUM_LOG_PENALTY = 1.25
LIKELIHOOD_FLOOR = 1e-6


def hypothesis_kind(events: Sequence[Mapping[str, Any]], termination_kind: str) -> str | None:
    """Name the physical ending family represented by one fitted topology."""

    contacts = [row for row in events if row.get("event_type") == "contact"]
    if not contacts:
        return None
    last_contact = max(float(row["frame"]) for row in contacts)
    terminal = [row for row in events if float(row.get("frame", -math.inf)) > last_contact]
    if any(row.get("event_type") == "net_hit" for row in terminal):
        return "net_hit"
    normalized = point_context.normalize_consumer_ending_kind(termination_kind)
    if normalized == "out":
        return "first_bounce_out"
    if normalized == "net":
        return "net_hit"
    if normalized in {"second_bounce", "fov_exit"}:
        return normalized
    bounce_count = sum(row.get("event_type") == "bounce" for row in terminal)
    return (
        "second_bounce" if bounce_count >= 2 else "first_bounce_out" if bounce_count == 1 else None
    )


def build(
    context: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and convert one optional context prior into an S6 receipt."""

    if context is None:
        return {
            "schema": SCHEMA,
            "available": False,
            "abstention_reason": "point_context_not_supplied",
            "hard_gate": False,
            "maximum_log_penalty": MAXIMUM_LOG_PENALTY,
        }
    try:
        witness = point_context.ending_witness(context, events)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema": SCHEMA,
            "available": False,
            "abstention_reason": f"invalid_point_context:{type(error).__name__}:{error}",
            "hard_gate": False,
            "maximum_log_penalty": MAXIMUM_LOG_PENALTY,
        }
    return {
        **witness,
        "schema": SCHEMA,
        "maximum_log_penalty": MAXIMUM_LOG_PENALTY,
        "selection_rule": (
            "fit input-only rank score plus capped negative-log context likelihood; "
            "context is never a gate"
        ),
    }


def penalty(witness: Mapping[str, Any], kind: str | None) -> float:
    """Return a bounded non-negative cost; unavailable evidence is exactly zero."""

    if not witness.get("available") or kind is None:
        return 0.0
    likelihoods = witness.get("ending_likelihoods")
    if not isinstance(likelihoods, Mapping) or kind not in likelihoods:
        return 0.0
    values = [float(value) for value in likelihoods.values()]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        return 0.0
    best = max(values)
    selected = float(likelihoods[kind])
    raw = math.log(max(best, LIKELIHOOD_FLOOR) / max(selected, LIKELIHOOD_FLOOR))
    return float(min(MAXIMUM_LOG_PENALTY, max(0.0, raw)))


def score(
    context: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
    termination_kind: str,
) -> dict[str, Any]:
    """Return the auditable soft cost for one fitted ending hypothesis."""

    witness = build(context, events)
    kind = hypothesis_kind(events, termination_kind)
    return {
        "kind": kind,
        "penalty": penalty(witness, kind),
        "witness": witness,
        "hard_gate": False,
    }
