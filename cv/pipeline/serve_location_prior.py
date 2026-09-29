"""Serve-contact mixture prior learned from external Hawk-Eye landmarks.

The artifact contains no repository evaluation labels.  A query prefers current-match automatic
contacts, then an artifact match cell, then the player across other matches, and finally a global
deuce/ad prior.  With ``end=None`` the returned XYZ is serve-local: side-normalized lateral offset,
inward baseline depth, height.  Supplying ``near`` or ``far`` returns the pipeline court frame.

This module only ranks/regularizes contact hypotheses.  It exposes uncertainty and an abstain path;
it does not hard-accept a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cv.pipeline import paths

SCHEMA = "tennis_serve_location_prior_v1"
COURT_LENGTH_M = 23.77
COURT_WIDTH_M = 10.97
HALF_WIDTH_M = COURT_WIDTH_M / 2.0
MIN_MATCH_SERVES = 8
MIN_STATURE_SAMPLES = 8
COVARIANCE_FLOOR_M2 = np.diag([0.04**2, 0.04**2, 0.04**2])


def _normalize_name(name: str | None) -> str:
    return "" if name is None else "".join(c for c in name.casefold() if c.isalnum())


def _key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_covariance(value: Any) -> np.ndarray:
    covariance = np.asarray(value, float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("a finite 3x3 covariance is required")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0:
        covariance += np.eye(3) * (1e-6 - eigenvalues.min())
    return covariance


def _local_to_pipeline(
    mean: np.ndarray, covariance: np.ndarray, side: str, end: str
) -> tuple[np.ndarray, np.ndarray]:
    side_sign = 1.0 if side == "deuce" else -1.0
    end_sign = 1.0 if end == "far" else -1.0
    lateral_sign = -end_sign * side_sign
    depth_sign = -1.0 if end == "far" else 1.0
    output = np.asarray(
        [
            HALF_WIDTH_M + lateral_sign * mean[0],
            COURT_LENGTH_M - mean[1] if end == "far" else mean[1],
            mean[2],
        ]
    )
    transform = np.diag([lateral_sign, depth_sign, 1.0])
    return output, transform @ covariance @ transform.T


def _pipeline_to_local(xyz: Sequence[float], side: str, end: str) -> np.ndarray:
    point = np.asarray(xyz, float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("each match contact must contain three finite coordinates")
    side_sign = 1.0 if side == "deuce" else -1.0
    end_sign = 1.0 if end == "far" else -1.0
    return np.asarray(
        [
            -(point[0] - HALF_WIDTH_M) * end_sign * side_sign,
            COURT_LENGTH_M - point[1] if end == "far" else point[1],
            point[2],
        ]
    )


def _component_peak(component: dict[str, Any]) -> float:
    covariance = _validate_covariance(
        component.get("covariance_xyz_m2", component.get("covariance_local_m2"))
    )
    return float(component["weight"] / math.sqrt(max(np.linalg.det(covariance), 1e-15)))


def _render_component(
    component: dict[str, Any], stature_m: float | None, side: str, end: str | None
) -> dict[str, Any]:
    mean = np.asarray(component["mean_local_m"], float)
    covariance = _validate_covariance(component["covariance_local_m2"])
    ratio = component.get("height_stature_ratio")
    stature_scaled = False
    if (
        stature_m is not None
        and ratio is not None
        and int(ratio.get("n", 0)) >= MIN_STATURE_SAMPLES
    ):
        target_mean = float(ratio["mean"]) * stature_m
        target_sigma = max(float(ratio["sigma"]) * stature_m, 0.04)
        old_sigma = math.sqrt(max(float(covariance[2, 2]), 1e-12))
        scale = target_sigma / old_sigma
        covariance[2, :] *= scale
        covariance[:, 2] *= scale
        mean[2] = target_mean
        stature_scaled = True
    if end is None:
        output_mean, output_covariance = mean, covariance
    else:
        output_mean, output_covariance = _local_to_pipeline(mean, covariance, side, end)
    return {
        "weight": float(component["weight"]),
        "mean_xyz_m": output_mean.tolist(),
        "sigma_xyz_m": np.sqrt(np.maximum(np.diag(output_covariance), 0.0)).tolist(),
        "covariance_xyz_m2": output_covariance.tolist(),
        "n_assigned": int(component.get("n_assigned", 0)),
        "height_scaled_by_stature": stature_scaled,
        "height_stature_ratio": ratio,
    }


def _normal_logpdf(delta: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("positive-definite covariance required")
    return float(
        -0.5 * (3 * math.log(2 * math.pi) + logdet + delta @ np.linalg.solve(covariance, delta))
    )


def _empirical_match_mixture(rows: Sequence[dict[str, Any]], side: str, end: str) -> dict[str, Any]:
    """Fit a bounded one-to-three-component mixture without a runtime sklearn dependency."""
    local = []
    for row in rows:
        if row.get("human_derived") is not False:
            raise ValueError(
                "automatic current-match prior rejects human-derived/undeclared contacts"
            )
        xyz = row.get("xyz_m")
        local.append(
            np.asarray(xyz, float)
            if row.get("coordinate_frame") == "serve_local"
            else _pipeline_to_local(xyz, side, end)
        )
    values = np.asarray(local, float)
    if len(values) < MIN_MATCH_SERVES:
        raise ValueError(f"current-match prior requires at least {MIN_MATCH_SERVES} contacts")
    best: tuple[float, list[dict[str, Any]]] | None = None
    for count in range(1, min(3, len(values) // MIN_MATCH_SERVES) + 1):
        order = np.argsort(values[:, 0])
        seeds = values[order[np.linspace(0, len(values) - 1, count).astype(int)]].copy()
        assignment = np.zeros(len(values), dtype=int)
        for _ in range(40):
            updated = np.argmin(
                np.linalg.norm(values[:, None, :] - seeds[None, :, :], axis=2), axis=1
            )
            if np.array_equal(updated, assignment) and _:
                break
            assignment = updated
            for index in range(count):
                if np.any(assignment == index):
                    seeds[index] = np.mean(values[assignment == index], axis=0)
        components = []
        valid = True
        log_likelihood = 0.0
        for index in range(count):
            members = values[assignment == index]
            if len(members) < MIN_MATCH_SERVES:
                valid = False
                break
            covariance = np.cov(members, rowvar=False, ddof=0) + COVARIANCE_FLOOR_M2
            component = {
                "weight": len(members) / len(values),
                "n_assigned": len(members),
                "mean_local_m": np.mean(members, axis=0).tolist(),
                "covariance_local_m2": covariance.tolist(),
                "height_stature_ratio": None,
            }
            components.append(component)
        if not valid:
            continue
        for value in values:
            terms = [
                math.log(component["weight"])
                + _normal_logpdf(
                    value - np.asarray(component["mean_local_m"]),
                    _validate_covariance(component["covariance_local_m2"]),
                )
                for component in components
            ]
            maximum = max(terms)
            log_likelihood += maximum + math.log(sum(math.exp(term - maximum) for term in terms))
        parameter_count = count * 9 + count - 1
        bic = parameter_count * math.log(len(values)) - 2 * log_likelihood
        if best is None or bic < best[0]:
            best = (bic, components)
    if best is None:
        raise ValueError("current-match contacts could not form a supported mixture")
    return {"n": len(values), "components": best[1], "bic": best[0]}


@dataclass(frozen=True)
class ServeLocationPrior:
    artifact: dict[str, Any]
    artifact_path: Path
    artifact_sha256: str

    @classmethod
    def load(cls, artifact_path: Path | None = None) -> "ServeLocationPrior":
        path = (
            paths.data_root() / "processed/serve_prior/serve_location_prior_v1.json"
            if artifact_path is None
            else artifact_path
        ).resolve()
        artifact = json.loads(path.read_text())
        if artifact.get("schema") != SCHEMA:
            raise ValueError(f"unsupported serve-prior schema: {artifact.get('schema')}")
        provenance = artifact.get("training_provenance") or {}
        if provenance.get("evaluation_labels_used"):
            raise ValueError("runtime serve prior rejects artifacts trained on evaluation labels")
        if not provenance.get("source_receipt_sha256"):
            raise ValueError("runtime serve prior requires a source receipt binding")
        return cls(artifact=artifact, artifact_path=path, artifact_sha256=_sha256(path))

    def _players(self, player_name: str | None) -> list[str]:
        if not player_name or _normalize_name(player_name) in {"", "unknown"}:
            return []
        return self.artifact["player_aliases"].get(_normalize_name(player_name), [])

    def mixture(
        self,
        player_name: str | None,
        side: str,
        serve_number: int | None,
        stature_m: float | None,
        *,
        end: str | None = None,
        match_id: str | None = None,
        current_match_contacts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a soft mixture in serve-local or absolute pipeline court coordinates."""
        if side not in {"deuce", "ad"}:
            raise ValueError("side must be deuce or ad")
        if end not in {None, "near", "far"}:
            raise ValueError("end must be near, far, or omitted for serve-local coordinates")
        if serve_number not in {None, 1, 2}:
            raise ValueError("serve_number must be 1, 2, or unknown")
        if stature_m is not None and (not math.isfinite(stature_m) or stature_m <= 0):
            raise ValueError("stature must be finite and positive when supplied")
        players = self._players(player_name)
        groups: list[dict[str, Any]] = []
        level = "abstained"
        selector = None
        if current_match_contacts is not None:
            if end is None:
                raise ValueError("current-match absolute contacts require an explicit court end")
            if len(current_match_contacts) >= MIN_MATCH_SERVES:
                groups = [_empirical_match_mixture(current_match_contacts, side, end)]
                level = "current_match"
                selector = "explicit_automatic_current_match_contacts"
        numbers = (serve_number,) if serve_number in {1, 2} else (1, 2)
        if not groups and match_id and end is not None:
            for player in players:
                for number in numbers:
                    value = self.artifact["match_priors"].get(
                        _key(match_id, player, side, end, number)
                    )
                    if value:
                        groups.append(value)
            if groups:
                level, selector = "known_match", "artifact_match_player_side_end_serve"
        if not groups and end is not None:
            for player in players:
                for number in numbers:
                    value = self.artifact["player_priors"].get(_key(player, side, end, number))
                    if value:
                        groups.append(value)
            if groups:
                level, selector = "player_across_matches", "artifact_player_side_end_serve"
        if not groups:
            for number in numbers:
                value = self.artifact["global_priors"].get(_key(side, number))
                if value:
                    groups.append(value)
            if groups:
                level, selector = "global_per_side", "artifact_global_side_serve"
        if not groups:
            return {
                "schema": "tennis_serve_location_mixture_v1",
                "status": "abstained",
                "abstention_reason": "no_supported_prior_cell",
                "components": [],
                "artifact_sha256": self.artifact_sha256,
            }
        group_total = sum(int(group["n"]) for group in groups)
        components = []
        for group in groups:
            group_weight = int(group["n"]) / group_total
            for source in group["components"]:
                rendered = _render_component(source, stature_m, side, end)
                rendered["weight"] *= group_weight
                components.append(rendered)
        total_weight = sum(component["weight"] for component in components)
        for component in components:
            component["weight"] /= total_weight
        mode_index = max(
            range(len(components)), key=lambda index: _component_peak(components[index])
        )
        return {
            "schema": "tennis_serve_location_mixture_v1",
            "status": "supported",
            "soft_prior_only": True,
            "selection_level": level,
            "selection_key": selector,
            "player_query": player_name or "unknown",
            "player_aliases": players,
            "side": side,
            "end": end,
            "serve_number": serve_number,
            "stature_m": stature_m,
            "coordinate_frame": "serve_local" if end is None else "pipeline_court",
            "coordinate_axes": (
                ["side_normalized_lateral", "inward_baseline_depth", "height"]
                if end is None
                else ["court_x", "court_y", "height"]
            ),
            "components": components,
            "mode_component_index": mode_index,
            "mode_xyz_m": components[mode_index]["mean_xyz_m"],
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "source_receipt_sha256": self.artifact["training_provenance"]["source_receipt_sha256"],
            "automatic_current_match_contacts": len(current_match_contacts or []),
            "evaluation_labels_used": [],
        }


def intersect_toss_estimate(prior: dict[str, Any], toss_estimate: dict[str, Any]) -> dict[str, Any]:
    """Multiply every prior Gaussian by a toss-contact Gaussian and renormalize weights."""
    if prior.get("status") != "supported":
        return {**prior, "toss_intersection": "abstained_prior_unavailable"}
    if toss_estimate.get("status") != "supported" or toss_estimate.get("contact_xyz_m") is None:
        return {**prior, "toss_intersection": "abstained_toss_unavailable"}
    toss_mean = np.asarray(toss_estimate["contact_xyz_m"], float)
    if toss_mean.shape != (3,) or not np.isfinite(toss_mean).all():
        raise ValueError("finite three-axis toss contact estimate required")
    if toss_estimate.get("contact_covariance_m2") is not None:
        toss_covariance = _validate_covariance(toss_estimate["contact_covariance_m2"])
    else:
        sigma = np.asarray(toss_estimate.get("contact_sigma_m"), float)
        if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("toss estimate requires covariance or positive per-axis sigma")
        toss_covariance = np.diag(sigma**2)
    components = []
    log_weights = []
    for source in prior["components"]:
        source_mean = np.asarray(source["mean_xyz_m"], float)
        source_covariance = _validate_covariance(source["covariance_xyz_m2"])
        precision = np.linalg.inv(source_covariance) + np.linalg.inv(toss_covariance)
        covariance = np.linalg.inv(precision)
        mean = covariance @ (
            np.linalg.solve(source_covariance, source_mean)
            + np.linalg.solve(toss_covariance, toss_mean)
        )
        log_weights.append(
            math.log(source["weight"])
            + _normal_logpdf(toss_mean - source_mean, source_covariance + toss_covariance)
        )
        components.append(
            {
                **source,
                "mean_xyz_m": mean.tolist(),
                "sigma_xyz_m": np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist(),
                "covariance_xyz_m2": covariance.tolist(),
            }
        )
    maximum = max(log_weights)
    weights = np.exp(np.asarray(log_weights) - maximum)
    weights /= weights.sum()
    for component, weight in zip(components, weights, strict=True):
        component["weight"] = float(weight)
    mode_index = max(range(len(components)), key=lambda index: _component_peak(components[index]))
    return {
        **prior,
        "schema": "tennis_serve_location_toss_intersection_v1",
        "components": components,
        "mode_component_index": mode_index,
        "mode_xyz_m": components[mode_index]["mean_xyz_m"],
        "toss_intersection": "supported_gaussian_product",
        "toss_contact_xyz_m": toss_mean.tolist(),
        "toss_contact_sigma_m": np.sqrt(np.diag(toss_covariance)).tolist(),
    }


def serve_location_prior(
    player_name: str | None,
    side: str,
    serve_number: int | None,
    stature_m: float | None,
    *,
    end: str | None = None,
    match_id: str | None = None,
    current_match_contacts: Sequence[dict[str, Any]] | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Convenience entrypoint for callers that do not retain a loaded prior object."""
    return ServeLocationPrior.load(artifact_path).mixture(
        player_name,
        side,
        serve_number,
        stature_m,
        end=end,
        match_id=match_id,
        current_match_contacts=current_match_contacts,
    )
