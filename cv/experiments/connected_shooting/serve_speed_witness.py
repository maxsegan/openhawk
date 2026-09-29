"""Soft-score connected depth branches against a label-free radar reading.

The displayed value is compared with the fitted ball-centre velocity at the
first contact instant.  Contact-time uncertainty, display quantisation, radar
angle, and drag-model error are represented by a speed-scaled engineering
allowance.  This witness ranks an already-legal family and never rejects one of
its members.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

UNIT_TO_MPS = {"km/h": 1 / 3.6, "mph": 0.44704}


@dataclass(frozen=True)
class RadarRelation:
    bias_mps: float = 0.0
    absolute_sigma_mps: float = 1.0
    relative_sigma: float = 0.04

    def validate(self) -> None:
        if (
            not np.isfinite([self.bias_mps, self.absolute_sigma_mps, self.relative_sigma]).all()
            or self.absolute_sigma_mps <= 0
            or self.relative_sigma <= 0
        ):
            raise ValueError("finite radar relation and positive uncertainties required")

    def sigma_mps(self, speed_mps: float) -> float:
        """Quadrature floor plus a proportional angle/model allowance."""
        if not math.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("finite positive reference speed required")
        return float(math.hypot(self.absolute_sigma_mps, self.relative_sigma * speed_mps))


def launch_speed(candidate: dict[str, Any]) -> float:
    """Return the fitted post-contact launch speed of the first flight."""
    parameters = np.asarray(candidate["measurement"]["fit"]["parameters"], float)
    if parameters.ndim != 1 or len(parameters) < 6 or not np.isfinite(parameters[:6]).all():
        raise ValueError("candidate lacks one finite launch state")
    return float(np.linalg.norm(parameters[3:6]))


def from_label_document(labels: dict[str, Any]) -> dict[str, Any]:
    """Normalize a frozen source-bound speed witness embedded in agent labels."""
    evidence = labels.get("serve_speed_evidence")
    if not evidence or evidence.get("status") != "visible":
        reading = {
            "abstained": True,
            "abstention_reason": "no_visible_frozen_serve_speed_evidence",
            "visible_refresh_observed": False,
        }
    else:
        unit = evidence.get("unit")
        value = evidence.get("value")
        if unit not in UNIT_TO_MPS or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("visible serve-speed evidence requires a finite supported unit")
        current = [row for row in evidence["observations"] if row["role"] == "current_serve"]
        previous = [
            row
            for row in evidence["observations"]
            if row["role"] == "preceding_display"
            and row.get("value") is not None
            and math.isfinite(float(row["value"]))
        ]
        refresh = bool(
            current
            and previous
            and max(row["frame"] for row in previous) < min(row["frame"] for row in current)
            and any(float(row["value"]) != float(value) for row in previous)
        )
        reading = {
            "abstained": False,
            "abstention_reason": None,
            "speed_mps": float(value) * UNIT_TO_MPS[unit],
            "display_value": float(value),
            "display_unit": unit,
            "visible_refresh_observed": refresh,
            "previous_stable_value": None if not previous else float(previous[-1]["value"]),
            "current_frames": [int(row["frame"]) for row in current],
            "contact_event_id": evidence.get("contact_event_id"),
            "annotation_origin": evidence.get("annotation_origin"),
        }
    return {
        "schema": "connected_frozen_label_serve_speed_v1",
        "reading": reading,
        "source_schema": None if evidence is None else evidence.get("schema"),
        "interpretation": (
            "source-bound frozen graphic evidence; display refresh is required for scoring, "
            "and the resulting witness is ranking-only"
        ),
    }


def evaluate(
    candidate: dict[str, Any],
    graphic_reading: dict[str, Any],
    relation: RadarRelation = RadarRelation(),
) -> dict[str, Any]:
    """Evaluate one already-fit branch without consulting withheld pixels."""
    relation.validate()
    fitted = launch_speed(candidate)
    radar = graphic_reading.get("speed_mps")
    if graphic_reading.get("abstained") or radar is None or not math.isfinite(float(radar)):
        return {
            "available": False,
            "abstained": True,
            "abstention_reason": graphic_reading.get("abstention_reason", "no_radar_speed"),
            "launch_speed_mps": fitted,
            "radar_speed_mps": None,
            "survived": True,
            "hard_gate": False,
            "selector_penalty": 0.0,
        }
    if not graphic_reading.get("visible_refresh_observed"):
        return {
            "available": False,
            "abstained": True,
            "abstention_reason": "no_readable_precontact_value_or_visible_refresh",
            "launch_speed_mps": fitted,
            "radar_speed_mps": float(radar),
            "survived": True,
            "hard_gate": False,
            "selector_penalty": 0.0,
            "graphic_confidence": graphic_reading.get("confidence"),
        }
    expected = float(radar) + relation.bias_mps
    delta = fitted - expected
    sigma = relation.sigma_mps(expected)
    z = delta / sigma
    return {
        "available": True,
        "abstained": False,
        "abstention_reason": None,
        "launch_speed_mps": fitted,
        "radar_speed_mps": float(radar),
        "radar_minus_launch_mps": -delta,
        "standardized_residual": z,
        "survived": True,
        "hard_gate": False,
        "selector_penalty": float(0.5 * z * z),
        "relation": {
            "assumption": (
                "displayed radar speed is a noisy witness of ball-centre speed at the "
                "fitted first-contact instant"
            ),
            "bias_mps": relation.bias_mps,
            "sigma_formula": "sqrt(absolute_sigma_mps^2 + (relative_sigma * speed_mps)^2)",
            "absolute_sigma_mps": relation.absolute_sigma_mps,
            "relative_sigma": relation.relative_sigma,
            "resolved_sigma_mps": sigma,
            "uncertainty_status": "engineering allowance, not calibrated uncertainty",
        },
        "selector_uses_withheld_pixels": False,
    }


def add_to_candidate(
    candidate: dict[str, Any],
    graphic_reading: dict[str, Any],
    relation: RadarRelation = RadarRelation(),
) -> dict[str, Any]:
    """Copy the witness into candidate evidence and update truth-free rank only."""
    witness = evaluate(candidate, graphic_reading, relation)
    evidence = candidate["evidence"]
    evidence["serve_speed_witness"] = witness
    evidence.setdefault("checks", {})["serve_speed_scored"] = witness["available"]
    evidence["input_only_rank_score"] = float(
        evidence["input_only_rank_score"] + witness["selector_penalty"]
    )
    return candidate
