"""Soft per-player service-contact position prior, split by deuce/ad side.

Research helper for the owner's serve idea: a small number of contact-position
clusters per player and service side, in absolute court coordinates and relative
to the server's grounded feet.  The caller supplies explicit reviewed
qualification per record; nothing here infers trust from fitted fields (old
fitted depths were often pinned at the baseline and automatic player boxes can
be metres off).  Evaluation returns a smooth soft energy for an optimizer; it is
never a hard gate, an acceptance certificate, or an automatic-inference input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
SERVICE_SIDES = ("deuce", "ad")
SERVER_ENDS = ("near", "far")
QUALIFICATION_FIELDS = ("contact_qualified", "identity_qualified", "ground_qualified")
SCHEMA = "connected_player_serve_contact_prior_v1"


@dataclass(frozen=True)
class ServePriorConfig:
    accepted_contact_sources: tuple[str, ...] = ("reviewed_label", "astra_reviewed")
    accepted_identity_sources: tuple[str, ...] = ("reviewed_label", "astra_reviewed")
    accepted_ground_sources: tuple[str, ...] = ("reviewed_label", "astra_reviewed")
    # Per-axis sigma floors (x, depth, height) in the canonical frame.  These are
    # conservative settings for a first version, not measured accuracy.
    sigma_floor_m: tuple[float, float, float] = (0.5, 0.75, 0.3)
    minimum_split_examples: int = 6
    minimum_cluster_examples: int = 3
    split_separation_m: float = 1.0
    split_separation_ratio: float = 2.0
    shared_branch_weight: float = 0.5

    def validate(self) -> None:
        floors = np.asarray(self.sigma_floor_m, float)
        if (
            floors.shape != (3,)
            or not np.isfinite(floors).all()
            or np.any(floors <= 0)
            or self.minimum_cluster_examples < 2
            or self.minimum_split_examples < 2 * self.minimum_cluster_examples
            or not 0 < self.shared_branch_weight <= 1
            or self.split_separation_m <= 0
            or self.split_separation_ratio <= 0
        ):
            raise ValueError("finite positive serve-prior configuration required")


def canonical(values: Any, server_end: str) -> np.ndarray:
    """Rotate a far-end court XY(Z) by 180 degrees so every server serves from the near end."""
    out = np.array(values, float)
    if server_end not in SERVER_ENDS or out.shape not in ((2,), (3,)) or not np.isfinite(out).all():
        raise ValueError("finite court XY or XYZ and a near/far server end required")
    if server_end == "far":
        out[0], out[1] = COURT_WIDTH_M - out[0], COURT_LENGTH_M - out[1]
    return out


def _relative(contact: np.ndarray, feet: np.ndarray) -> np.ndarray:
    return np.array([contact[0] - feet[0], contact[1] - feet[1], contact[2]])


def _rejections(record: dict[str, Any], config: ServePriorConfig) -> tuple[list[str], list[str]]:
    """Explicit reviewed qualification only; never inferred from fitted or gate fields."""
    if any(not isinstance(record.get(field), bool) for field in QUALIFICATION_FIELDS):
        raise ValueError(f"explicit boolean {QUALIFICATION_FIELDS} required per record")
    sources = record.get("sources", {})
    court = [
        reason
        for reason, failed in (
            ("contact_not_qualified", not record["contact_qualified"]),
            ("identity_not_qualified", not record["identity_qualified"]),
            ("contact_source", sources.get("contact") not in config.accepted_contact_sources),
            ("identity_source", sources.get("identity") not in config.accepted_identity_sources),
        )
        if failed
    ]
    relative = court + [
        reason
        for reason, failed in (
            ("ground_not_qualified", not record["ground_qualified"]),
            ("ground_source", sources.get("ground") not in config.accepted_ground_sources),
            ("feet_missing", record.get("grounded_feet_xy_m") is None),
        )
        if failed
    ]
    return court, relative


def _deterministic_split(xy: np.ndarray, config: ServePriorConfig) -> list[np.ndarray] | None:
    """Split along the principal XY axis only when two groups are clearly separated."""
    centred = xy - xy.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    order = np.argsort(centred @ axis, kind="stable")
    best = None
    for cut in range(
        config.minimum_cluster_examples, len(xy) - config.minimum_cluster_examples + 1
    ):
        left, right = order[:cut], order[cut:]
        separation = float(np.linalg.norm(np.median(xy[left], 0) - np.median(xy[right], 0)))
        spread = float(np.sqrt(max(np.mean(np.var(xy[left], 0)), np.mean(np.var(xy[right], 0)))))
        margin = separation - config.split_separation_ratio * spread
        if best is None or margin > best[0]:
            best = (margin, separation, left, right)
    if best is not None and best[0] >= 0 and best[1] >= config.split_separation_m:
        return [best[2], best[3]]
    return None


def _fit_branch(points: list[np.ndarray], ids: list[str], config: ServePriorConfig) -> dict:
    if not points:
        return {"status": "abstained", "mode": None, "clusters": [], "source_attempt_ids": []}
    array = np.asarray(points, float)
    floors = np.asarray(config.sigma_floor_m, float) ** 2
    groups, mode = [np.arange(len(array))], "single_robust_centre"
    if len(array) >= config.minimum_split_examples:
        split = _deterministic_split(array[:, :2], config)
        if split is not None:
            groups, mode = split, "two_cluster"
    clusters = []
    for index in groups:
        subset = array[index]
        variance = np.maximum(np.var(subset, axis=0), floors) if len(subset) > 1 else floors
        clusters.append(
            {
                "centre_m": np.median(subset, axis=0).tolist(),
                "sigma_m": np.sqrt(variance).tolist(),
                "weight": len(subset) / len(array),
                "count": int(len(subset)),
                "attempt_ids": [ids[i] for i in index],
            }
        )
    return {"status": "supported", "mode": mode, "clusters": clusters, "source_attempt_ids": ids}


def fit_prior(
    records: list[dict[str, Any]],
    player_id: str,
    service_side: str,
    target_attempt_id: str | None = None,
    config: ServePriorConfig = ServePriorConfig(),
) -> dict[str, Any]:
    """Fit the explicit player+side prior from other qualified attempts (no pooling)."""
    config.validate()
    if service_side not in SERVICE_SIDES:
        raise ValueError("service_side must be the caller's explicit 'deuce' or 'ad' label")
    ids = [record["attempt_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate attempt_id in serve-prior records")
    absolute, absolute_ids, relative, relative_ids, rejected = [], [], [], [], []
    target_seen = False
    for record in records:
        if record["player_id"] != player_id or record["service_side"] != service_side:
            continue
        if record["attempt_id"] == target_attempt_id:
            target_seen = True
            continue
        court_reasons, relative_reasons = _rejections(record, config)
        if not court_reasons:
            contact = canonical(record["contact_xyz_m"], record["server_end"])
            absolute.append(contact)
            absolute_ids.append(record["attempt_id"])
            if not relative_reasons:
                feet = canonical(record["grounded_feet_xy_m"], record["server_end"])
                relative.append(_relative(contact, feet))
                relative_ids.append(record["attempt_id"])
        if court_reasons or relative_reasons:
            rejected.append(
                {
                    "attempt_id": record["attempt_id"],
                    "absolute_rejections": court_reasons,
                    "relative_rejections": relative_reasons,
                }
            )
    return {
        "schema": SCHEMA,
        "player_id": player_id,
        "service_side": service_side,
        "target_attempt_id": target_attempt_id,
        "target_excluded": target_seen,
        "canonical_frame": "near-end server; far-end records rotated 180 degrees about court centre",
        "absolute": _fit_branch(absolute, absolute_ids, config),
        "relative": _fit_branch(relative, relative_ids, config),
        "rejected": rejected,
        "config": asdict(config),
        "hard_gate": False,
        "automatic_inference_eligible": False,
        "interpretation": (
            "soft per-player/per-side contact clusters from explicitly qualified other attempts; "
            "unnormalized kernel energy, not a calibrated probability density; "
            "sigma floors are settings, not accuracy certificates"
        ),
    }


def _branch_energy(branch: dict, point: np.ndarray) -> tuple[float, list[float]]:
    clusters = branch["clusters"]
    centres = np.asarray([row["centre_m"] for row in clusters], float)
    sigmas = np.asarray([row["sigma_m"] for row in clusters], float)
    weights = np.asarray([row["weight"] for row in clusters], float)
    mahalanobis2 = np.sum(((point - centres) / sigmas) ** 2, axis=1)
    energy = -2.0 * logsumexp(-0.5 * mahalanobis2, b=weights)
    return max(float(energy), 0.0), mahalanobis2.tolist()


def evaluate(
    prior: dict[str, Any],
    contact_xyz_m: Any,
    server_end: str,
    grounded_feet_xy_m: Any = None,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Smooth soft energy -2 log(sum_k w_k exp(-0.5 d_k^2)) per available branch."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("finite non-negative caller weight required")
    contact = canonical(contact_xyz_m, server_end)
    feet = None if grounded_feet_xy_m is None else canonical(grounded_feet_xy_m, server_end)
    points = {"absolute": contact}
    if feet is not None:
        points["relative"] = _relative(contact, feet)
    active = [name for name, point in points.items() if prior[name]["status"] == "supported"]
    branch_weight = prior["config"]["shared_branch_weight"] if len(active) == 2 else 1.0
    branches, residuals = {}, []
    for name in active:
        energy, mahalanobis2 = _branch_energy(prior[name], points[name])
        branches[name] = {
            "canonical_point_m": points[name].tolist(),
            "energy": energy,
            "cluster_mahalanobis2": mahalanobis2,
            "branch_weight": branch_weight,
            "weighted_energy": float(weight * branch_weight * energy),
        }
        residuals.append(np.sqrt(weight * branch_weight * energy))
    total = float(sum(row["weighted_energy"] for row in branches.values()))
    return {
        "schema": SCHEMA + "_evaluation",
        "player_id": prior["player_id"],
        "service_side": prior["service_side"],
        "branch_availability": {
            "absolute": prior["absolute"]["status"] == "supported",
            "relative": prior["relative"]["status"] == "supported" and feet is not None,
            "ground_supplied": feet is not None,
            "note": "absolute and relative evidence are correlated, not independent certificates",
        },
        "branches": branches,
        "caller_weight": float(weight),
        "energy": total,
        "optimization_residuals": [float(value) for value in residuals],
        "selector_penalty": 0.5 * total,
        "hard_gate": False,
        "automatic_inference_eligible": False,
    }


def optimization_residuals(prior, contact_xyz_m, server_end, grounded_feet_xy_m=None, weight=1.0):
    """Residuals whose half-sum-of-squares equals selector_penalty (half the energy)."""
    result = evaluate(prior, contact_xyz_m, server_end, grounded_feet_xy_m, weight)
    return np.asarray(result["optimization_residuals"], float)
