"""Re-read the automatic decoder's own acceptance decision at a lower operating point.

The trained ``r3d_18`` event decoder writes three numbers on every row it decodes:
``acceptance_marginal``, the ``decision_threshold`` it was judged against, and the
resulting ``abstain``.  The threshold is a *deployment* choice, not a property of the
model, and the sealed 2026-09-19 event benchmark measured what changes when it moves:
emitting the decoder's whole marginal band instead of the calibrated operating point is
worth +10.7 recall points for -5.1 precision, at zero cost, and beats every VLM
adjudicator arm on both axes
(``cv/experiments/event_adjudication/BENCHMARK.md``).

That benchmark cannot say whether S6 *wants* the trade.  More true events mean more
complete points and more reconstructible flights; more false events can corrupt a flight
grammar that would otherwise have solved.  This module is the default-off lever that lets
the question be measured on the panel, and it is deliberately the same operation the
benchmark's own band-floor arm performed: a re-threshold of rows the decoder already
wrote, not a re-decode.  Nothing here runs a model, moves an epoch, invents a row or
edits a frozen artifact.

Two refusals are structural, because both would answer a different question:

* **Only a threshold refusal may be restated.**  A row abstains either because its
  acceptance marginal fell short (``model_abstain``) or because a gate held it --
  ``tracking_arc_abstained`` and the point-validity gate both write ``abstain`` without
  ``model_abstain``.  Promoting a gate-held row is the separate, already-measured
  ``model_accepted_admission`` mechanism, which was a measured net negative.  This
  module leaves every gate refusal exactly where the producer left it.
* **The operating point may only go DOWN.**  A floor above the producer's own declared
  threshold would discard claims the producer actually made, which is a different
  experiment with a different failure mode.  It raises instead.
"""

from __future__ import annotations

import math
from typing import Any

#: The policy key, its off value, and the event types a restatement may touch.
OPERATING_POINT_FIELD = "automatic_event_operating_point"
OPERATING_POINT_DEFAULT = "off"
RESTATABLE_EVENT_TYPES = ("contact", "bounce", "net_hit")

#: Written on every row this module restates, so a prepared attempt carries the whole
#: arithmetic of the change rather than an unexplained accepted row.
RESTATEMENT_FIELD = "automatic_operating_point"
RESTATEMENT_SCHEMA = "automatic_event_operating_point_restatement_v1"


def event_operating_point(value: Any) -> float | None:
    """The declared consumer-side acceptance floor, or ``None`` when off.

    ``"off"`` is the default and the absent value.  Anything else must be a finite
    probability in ``[0, 1]``; a bool is not a probability.
    """

    if value is None or value == OPERATING_POINT_DEFAULT:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "automatic_event_operating_point must be 'off' or a finite probability"
        )
    floor = float(value)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError(
            "automatic_event_operating_point must be 'off' or a finite probability"
        )
    return floor


def _acceptance_score(row: dict) -> float | None:
    """The score the producer's own ``abstain`` was decided on.

    ``acceptance_marginal`` is that field.  It equals ``path_marginal`` on every row but
    a second-bounce support state, whose acceptance score is the counterfactual
    ordinary-grammar marginal instead; the fallback keeps an older stream readable.
    """

    for name in ("acceptance_marginal", "path_marginal"):
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
    return None


def restatable(row: dict) -> bool:
    """Whether this row is a THRESHOLD refusal of a physical event, and nothing else."""

    return (
        row.get("abstain") is True
        and row.get("model_abstain") is True
        and row.get("event_type") in RESTATABLE_EVENT_TYPES
        and _acceptance_score(row) is not None
        and isinstance(row.get("decision_threshold"), (int, float))
        and not isinstance(row.get("decision_threshold"), bool)
        and math.isfinite(float(row["decision_threshold"]))
    )


def restate(document: list[dict], floor: float | None) -> tuple[list[dict], dict]:
    """Return the emission document read at ``floor``, with the census of what moved.

    With ``floor`` ``None`` the document is returned unchanged and the census records
    that the producer's own operating point stands.  Rows are never dropped, reordered,
    retimed or added: a restated row is the producer's row with ``abstain`` cleared and
    the arithmetic of the change attached.
    """

    if not isinstance(document, list):
        raise ValueError("automatic event artifact must be a JSON list")
    census = {
        "schema": RESTATEMENT_SCHEMA,
        "declared_floor": floor,
        "rows": len(document),
        "restatable_rows": 0,
        "restated_rows": 0,
        "producer_thresholds": [],
        "restated": [],
    }
    if floor is None:
        return document, census
    thresholds: set[float] = set()
    output: list[dict] = []
    for row in document:
        if not restatable(row):
            output.append(row)
            continue
        census["restatable_rows"] += 1
        producer_threshold = float(row["decision_threshold"])
        thresholds.add(producer_threshold)
        if floor > producer_threshold:
            raise ValueError(
                "a declared automatic event operating point may only LOWER the producer's "
                f"own threshold: {floor!r} is above its declared {producer_threshold!r}"
            )
        score = _acceptance_score(row)
        if score < floor:
            output.append(row)
            continue
        census["restated_rows"] += 1
        census["restated"].append(
            {
                "clip": row.get("clip"),
                "event_type": row.get("event_type"),
                "frame": row.get("frame"),
                "acceptance_marginal": score,
            }
        )
        output.append(
            {
                **row,
                "abstain": False,
                "model_abstain": False,
                RESTATEMENT_FIELD: {
                    "schema": RESTATEMENT_SCHEMA,
                    "restated": True,
                    "declared_floor": floor,
                    "producer_decision_threshold": producer_threshold,
                    "acceptance_marginal": score,
                    "producer_abstained": True,
                    "basis": (
                        "the producer's own recorded acceptance decision, re-read at a "
                        "lower consumer operating point; no model ran and no epoch moved"
                    ),
                },
            }
        )
    census["producer_thresholds"] = sorted(thresholds)
    return output, census
