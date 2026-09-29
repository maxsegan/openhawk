from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import serve_location_prior as prior


def _component(mean, sigma=(0.2, 0.2, 0.1), ratio=None):
    return {
        "weight": 1.0,
        "n_assigned": 30,
        "mean_local_m": list(mean),
        "sigma_local_m": list(sigma),
        "covariance_local_m2": np.diag(np.square(sigma)).tolist(),
        "height_stature_ratio": ratio,
    }


def _artifact(path: Path) -> prior.ServeLocationPrior:
    payload = {
        "schema": prior.SCHEMA,
        "training_provenance": {
            "source_receipt_sha256": "abc",
            "evaluation_labels_used": [],
        },
        "player_aliases": {"player": ["P.PLAYER"]},
        "global_priors": {
            "deuce|1": {"n": 100, "components": [_component((0.8, 0.5, 2.8))]},
            "deuce|2": {"n": 50, "components": [_component((1.2, 0.3, 2.9))]},
        },
        "player_priors": {
            "P.PLAYER|deuce|near|1": {
                "n": 30,
                "components": [
                    _component((1.0, 0.2, 2.7), ratio={"n": 30, "mean": 1.5, "sigma": 0.05})
                ],
            }
        },
        "match_priors": {
            "m|P.PLAYER|deuce|near|1": {
                "n": 10,
                "components": [_component((1.1, 0.1, 2.75))],
            }
        },
    }
    path.write_text(json.dumps(payload))
    return prior.ServeLocationPrior(payload, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_query_prefers_match_then_player_and_mirrors_court_end(tmp_path: Path) -> None:
    model = _artifact(tmp_path / "prior.json")
    match = model.mixture("Player", "deuce", 1, 1.9, end="near", match_id="m")
    assert match["selection_level"] == "known_match"
    assert match["mode_xyz_m"] == pytest.approx([6.585, 0.1, 2.75])
    player = model.mixture("Player", "deuce", 1, 1.9, end="near")
    assert player["selection_level"] == "player_across_matches"
    assert player["mode_xyz_m"] == pytest.approx([6.485, 0.2, 2.85])
    far = model.mixture("Player", "deuce", 1, 1.9, end="far")
    assert far["selection_level"] == "global_per_side"
    assert far["mode_xyz_m"] == pytest.approx([4.685, 23.27, 2.8])


def test_unknown_player_and_serve_return_weighted_global_mixture(tmp_path: Path) -> None:
    model = _artifact(tmp_path / "prior.json")
    result = model.mixture(None, "deuce", None, None)
    assert result["selection_level"] == "global_per_side"
    assert result["coordinate_frame"] == "serve_local"
    assert [component["weight"] for component in result["components"]] == pytest.approx(
        [2 / 3, 1 / 3]
    )


def test_current_match_contacts_require_automatic_provenance(tmp_path: Path) -> None:
    model = _artifact(tmp_path / "prior.json")
    contacts = [
        {"xyz_m": [6.5 + 0.01 * index, 0.2, 2.9], "human_derived": False} for index in range(8)
    ]
    result = model.mixture("Player", "deuce", 1, 1.9, end="near", current_match_contacts=contacts)
    assert result["selection_level"] == "current_match"
    contacts[0]["human_derived"] = True
    with pytest.raises(ValueError, match="rejects human-derived"):
        model.mixture("Player", "deuce", 1, 1.9, end="near", current_match_contacts=contacts)


def test_toss_intersection_reduces_uncertainty_and_moves_mode(tmp_path: Path) -> None:
    model = _artifact(tmp_path / "prior.json")
    base = model.mixture("Player", "deuce", 1, None, end="near")
    combined = prior.intersect_toss_estimate(
        base,
        {
            "status": "supported",
            "contact_xyz_m": [6.2, 0.4, 3.0],
            "contact_sigma_m": [0.1, 0.1, 0.1],
        },
    )
    assert combined["toss_intersection"] == "supported_gaussian_product"
    assert combined["mode_xyz_m"][0] < base["mode_xyz_m"][0]
    assert combined["components"][0]["sigma_xyz_m"][0] < base["components"][0]["sigma_xyz_m"][0]


def test_loader_rejects_evaluation_trained_artifact(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "schema": prior.SCHEMA,
                "training_provenance": {
                    "source_receipt_sha256": "abc",
                    "evaluation_labels_used": ["truth.json"],
                },
            }
        )
    )
    with pytest.raises(ValueError, match="evaluation labels"):
        prior.ServeLocationPrior.load(path)
