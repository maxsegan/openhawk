import numpy as np
import pytest

from cv.experiments.connected_shooting import athlete_priors


def player(name, stature, root, wrist_status="abstained"):
    return {
        "player": name,
        "stature_m": stature,
        "court_centre_xy_m": root,
        "pose_wrist_witness": {
            "status": wrist_status,
            "wrist_to_ball_proxy_m": 0.6,
        },
    }


def test_zverev_serve_height_is_inside_stature_scaled_soft_band():
    players = [player("Zverev", 1.98, [5.0, 0.0])]
    result = athlete_priors.evaluate(np.array([[5.0, -0.5, 3.427]]), players)
    assert result["hard_gate"] is False
    assert result["serve_height"]["zero_penalty_interval_m"] == pytest.approx([2.97, 3.465])
    assert result["serve_height"]["residual"] == 0
    assert result["contacts"][0]["root_reach_residual"] == 0


def test_stature_reach_is_soft_and_pose_wrist_is_reported_separately():
    players = [player("Paolini", 1.63, [0.0, 0.0], "supported")]
    result = athlete_priors.evaluate(np.array([[2.3, 0.0, 2.6]]), players)
    row = result["contacts"][0]
    assert row["zero_penalty_root_reach_m"] == pytest.approx(2.2005)
    assert row["root_reach_residual"] > 0
    assert row["root_reach_selector_penalty"] > 0
    assert row["pose_wrist_witness"]["status"] == "supported"
    assert row["wrist_selector_penalty"] == 0


@pytest.mark.parametrize("stature", [0, -1, np.nan])
def test_invalid_stature_fails_closed(stature):
    with pytest.raises(ValueError):
        athlete_priors.optimization_residuals(
            np.array([[0.0, 0.0, 3.0]]), [player("P", stature, [0.0, 0.0])]
        )


def unavailable_player(root):
    return {
        "player": None,
        "stature_m": None,
        "court_centre_xy_m": root,
        "athlete_evidence": athlete_priors.unavailable_evidence(),
        "pose_wrist_witness": {
            "status": "abstained",
            "abstention_reason": "missing_positive_stature_or_box_scale",
        },
    }


def test_explicitly_unavailable_evidence_abstains_with_fixed_shape_zero_residuals():
    contacts = np.array([[5.0, -0.5, 4.5], [2.3, 23.0, 1.0]])
    supplied = [player("Server", 1.98, [5.0, 0.0]), player("Receiver", 1.63, [0.0, 23.0])]
    reference = athlete_priors.optimization_residuals(contacts, supplied)
    mixed = [unavailable_player([5.0, 0.0]), supplied[1]]
    residuals = athlete_priors.optimization_residuals(contacts, mixed)
    assert residuals.shape == reference.shape == (3,)
    assert residuals[0] == 0.0 and reference[0] > 0
    assert residuals[1] == 0.0
    assert residuals[2] == reference[2] > 0
    report = athlete_priors.evaluate(contacts, mixed)
    assert report["serve_height"]["stature_m"] is None
    assert report["serve_height"]["zero_penalty_interval_m"] is None
    assert report["serve_height"]["residual"] == 0.0
    assert report["serve_height"]["athlete_evidence"]["status"] == "unavailable"
    assert report["contacts"][0]["zero_penalty_root_reach_m"] is None
    assert report["contacts"][0]["pose_wrist_witness"]["status"] == "abstained"
    assert report["contacts"][1]["stature_m"] == 1.63
    assert report["athlete_evidence"] == {
        "status": "partial",
        "abstained_terms": ["serve_height", "root_reach[0]"],
        "abstained_residuals_fixed_zero": True,
    }
    assert (
        report["selector_penalty"]
        == athlete_priors.evaluate(contacts, supplied)["contacts"][1]["root_reach_selector_penalty"]
    )
    everyone = athlete_priors.evaluate(contacts, [unavailable_player(r) for r in ([5, 0], [0, 23])])
    assert everyone["athlete_evidence"]["status"] == "unavailable"
    assert everyone["selector_penalty"] == 0.0
    assert athlete_priors.evaluate(contacts, supplied)["athlete_evidence"]["status"] == "supplied"


def test_missing_stature_without_declaration_and_declared_stature_both_fail():
    contacts = np.array([[5.0, -0.5, 3.0]])
    silent = {"player": None, "stature_m": None, "court_centre_xy_m": [5.0, 0.0]}
    with pytest.raises(ValueError, match="without an explicit unavailable"):
        athlete_priors.optimization_residuals(contacts, [silent])
    contradictory = player("P", 1.8, [5.0, 0.0]) | {
        "athlete_evidence": athlete_priors.unavailable_evidence()
    }
    with pytest.raises(ValueError, match="cannot also declare"):
        athlete_priors.optimization_residuals(contacts, [contradictory])
