"""Isolate known numerical candidate failures without hiding runtime or I/O faults."""

from __future__ import annotations

import numpy as np


class CandidateFailure(ValueError):
    """A specific numerical fit/measurement could not produce a complete candidate."""

    def __init__(self, message, *, aggregate=False):
        super().__init__(message)
        self.aggregate = aggregate


def numerical_call(phase, function, *args, **kwargs):
    """Wrap only a numerical boundary, never reporting, checkpoints or source checks."""
    try:
        return function(*args, **kwargs)
    except (ValueError, FloatingPointError, OverflowError) as error:
        raise CandidateFailure(f"{phase}: {type(error).__name__}: {error}") from error


class Attempts:
    def __init__(self):
        self.failures = []

    def run(self, metadata, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except CandidateFailure as error:
            self.failures.append(
                {
                    **metadata,
                    "reason": str(error),
                    "failure_kind": "aggregate" if error.aggregate else "numerical",
                }
            )
            return None


def completed_source_allowed(search):
    receipt = search.get("search_budget", {})
    return bool(receipt.get("exhausted")) or bool(
        receipt.get("retained_completed_after_numerical_failures") is True
        and search.get("numerical_candidate_failures")
        and search.get("completed_candidates")
        and not search.get("refined_candidates")
    )


def validate_completed(candidate, flights):
    """Validate numerical payload before separately publishing its checkpoint."""
    parameters = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
    score = candidate["evidence"]["input_only_rank_score"]
    if parameters.shape != (5 + 6 * flights,) or not np.isfinite(parameters).all():
        raise CandidateFailure("completed candidate needs a finite full-scene vector")
    if not np.isfinite(score) or not candidate["measurement"].get("native_projection"):
        raise CandidateFailure(
            "completed candidate needs a finite input rank and native projection"
        )


def topology_completed(candidates, topology_name, *, membership=None):
    """A current invocation's completed candidates must match its selected topology."""
    return [
        candidate
        for candidate in candidates
        if candidate.get("topology_name") == topology_name
        and candidate.get("membership") == membership
    ]


def completed_output(chosen, completed, receipt):
    """Retain numerical-failure coarse fallback even without a wall deadline."""
    membership = chosen.get("membership")
    scoped = topology_completed(completed, chosen["topology_name"], membership=membership)
    fallback = bool(
        not chosen["refined"] and chosen["coarse"] and chosen["numerical_candidate_failures"]
    )
    if fallback:
        scoped = list(chosen["coarse"])
        if membership is not None and len(scoped) != len(
            topology_completed(scoped, chosen["topology_name"], membership=membership)
        ):
            raise ValueError("coarse fallback differs from the chosen topology membership")
    return scoped, {
        **receipt,
        "topology_name": chosen["topology_name"],
        "checkpointed_candidates_across_topologies": receipt.get("completed_candidates", 0),
        "completed_candidates": len(scoped),
        "retained_completed_after_numerical_failures": fallback,
        **({"membership": membership} if membership is not None else {}),
    }


def completed_at_deadline(candidates, topology_name, *, membership=None):
    from cv.experiments.connected_shooting.search_budget import SearchDeadline

    scoped = topology_completed(candidates, topology_name, membership=membership)
    if not scoped:
        raise SearchDeadline("shared deadline exhausted before this topology completed a candidate")
    return scoped


#: Whole-point seed death that must degrade to the component partition.
#: The net-stop line is only the prefix-chain failure raised by ``baseline_seed``
#: before any candidate exists. A bare bounce cap is not this death.
NUMERICAL_SEED_DEATH_MARKERS = (
    "no completed candidate after",
    "no finite full-point candidate",
    "all configured seeds failed numerically",
    "net-stop seed measured dynamics bounce cap reached",
)
#: Missing camera or pose: the attempt cannot invent those inputs.
FATAL_ATTEMPT_MARKERS = (
    "player pose is absent",
    "every visible label requires one supported matching camera",
    "one matching camera document per attempt required",
    "no camera",
)


def attempt_seed_failure_reason(message):
    """Receipt on an abstained original slot after whole-point seed death."""
    text = str(message).strip()
    prefix = "attempt_seed_failure: "
    return text if text.startswith(prefix) else f"{prefix}{text}"


def is_fatal_attempt_failure(reason):
    """True when the attempt is missing a camera or required pose input."""
    text = str(reason).lower()
    return any(marker.lower() in text for marker in FATAL_ATTEMPT_MARKERS)


def should_fallback_to_components(reason, policy=None):
    """Numerical whole-point seed death, not a missing camera or pose.

    Absent or off `whole_point_seed_fallback` keeps the attempt dead. On
    degrades to the component partition with attempt_seed_failure receipts.
    Short-attempt seed death uses the same key.
    """
    from cv.pipeline.s6_labeled_stage import shared_settings

    if shared_settings(policy or {}).get("whole_point_seed_fallback", "off") != "on":
        return False
    if not reason or is_fatal_attempt_failure(reason):
        return False
    text = str(reason)
    return any(marker in text for marker in NUMERICAL_SEED_DEATH_MARKERS)
