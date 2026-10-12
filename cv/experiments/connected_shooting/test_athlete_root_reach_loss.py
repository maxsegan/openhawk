"""A wrong actor cannot exert unbounded reach influence; replay preserves one policy."""

from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cv.experiments.connected_shooting import (
    athlete_priors as priors,
    agent_single_flight_search as single,
    labeled_interior_normal as interior,
    player_state_fallback as fallback,
)
from cv.pipeline import s6_labeled_stage as stage


def player(root=(0.0, 0.0)):
    return dict(player="P", stature_m=1.8, court_centre_xy_m=list(root))


def test_quadratic_default_retains_original_states_residuals_and_report():
    players = [player(), player((0.0, 23.0))]
    original = deepcopy(players)
    points = np.array([[4.0, 0.0, 2.0], [5.0, 23.0, 1.0]])
    expected = np.r_[(2.7 - 2.0) / 0.144, (4.0 - 2.43) / 0.18, (5.0 - 2.43) / 0.18]
    before = priors.evaluate(points, players)
    assert priors.configure_root_reach(players, "quadratic") == original
    np.testing.assert_allclose(priors.optimization_residuals(points, players), expected)
    assert priors.evaluate(points, players) == before
    assert priors.root_reach_configuration("quadratic") == {}
    assert "root_reach_loss" not in before["contacts"][0]


def test_cauchy_objective_report_and_weighted_interior_residual_agree():
    players = [player(), player((0.0, 23.0))]
    points = np.array([[4.0, 0.0, 2.0], [5.0, 23.0, 1.0]])
    raw = priors.optimization_residuals(points, players)
    priors.configure_root_reach(players, "cauchy")
    c = priors.ROOT_REACH_CAUCHY_SCALE
    transformed = priors.optimization_residuals(points, players)
    assert transformed[0] == raw[0]  # serve-height evidence is unchanged
    np.testing.assert_allclose(transformed[1:] ** 2, c * c * np.log1p((raw[1:] / c) ** 2))
    report = priors.evaluate(points, players)
    assert report["optimization_residuals"] == transformed.tolist()
    assert report["selector_penalty"] == 0.5 * float(transformed @ transformed)
    for index, row in enumerate(report["contacts"]):
        assert row["root_reach_raw_residual"] == raw[index + 1]
        assert row["root_reach_selector_penalty"] == 0.5 * transformed[index + 1] ** 2
        local = interior.indexed_player_residuals(points, players, index)
        np.testing.assert_array_equal(local, 4 * transformed[index + 1 : index + 2])
        assert float(local @ local) == pytest.approx(32 * row["root_reach_selector_penalty"])


def test_cauchy_has_quadratic_local_curvature_and_bounded_declining_influence():
    players = priors.configure_root_reach([player()], "cauchy")
    c = priors.ROOT_REACH_CAUCHY_SCALE

    def cost(r):
        contact = np.array([[2.43 + 0.18 * r, 0, 3.0]])
        return priors.evaluate(contact, players)["contacts"][0]["root_reach_selector_penalty"]

    assert cost(1e-3) / (0.5e-6) == pytest.approx(1, rel=1e-6)
    gradients = []
    for r in (1, 4, 10, 100):
        gradient = (cost(r + 1e-4) - cost(r - 1e-4)) / 2e-4
        assert gradient == pytest.approx(r / (1 + (r / c) ** 2), rel=1e-6)
        assert gradient <= c / 2 + 1e-8
        gradients.append(gradient)
    assert gradients[-1] < gradients[-2] < gradients[1]


def test_zero_hinge_and_unavailable_roster_are_exact_zero_under_both_modes():
    for mode in priors.ROOT_REACH_LOSSES:
        players = priors.configure_root_reach([player()], mode)
        np.testing.assert_array_equal(
            priors.optimization_residuals(np.array([[0, 0, 3.0]]), players), [0, 0]
        )
        unavailable = [
            dict(
                player=None,
                stature_m=None,
                court_centre_xy_m=[0, 0],
                athlete_evidence=priors.unavailable_evidence(),
            )
        ]
        priors.configure_root_reach(unavailable, mode)
        report = priors.evaluate(np.array([[100, 100, 10]]), unavailable)
        assert report["optimization_residuals"] == [0.0, 0.0]
        assert report["selector_penalty"] == 0


def test_mixed_or_invalid_policy_refuses_and_replay_requires_explicit_matching_policy():
    players = [player(), player() | {"athlete_root_reach_loss": "cauchy"}]
    with pytest.raises(ValueError, match="uniform"):
        priors.optimization_residuals(np.ones((2, 3)), players)
    with pytest.raises(ValueError, match="differs"):
        priors.configure_root_reach(players, "quadratic")
    with pytest.raises(ValueError, match="supported"):
        priors.configure_root_reach([player()], "wins")
    report = {"configuration": priors.root_reach_configuration("cauchy")}
    with pytest.raises(ValueError, match="match"):
        priors.replay_root_reach(SimpleNamespace(), report)
    mode = priors.replay_root_reach(SimpleNamespace(athlete_root_reach_loss="cauchy"), report)
    players = priors.configure_root_reach([player()], mode)
    # The runtime's serialized player context preserves policy without ambient global state.
    restored = json.loads(json.dumps(players))
    points = np.array([[9.0, 0.0, 3.0]])
    assert priors.evaluate(points, restored) == priors.evaluate(points, players)
    assert priors.replay_root_reach(SimpleNamespace(), {"configuration": {}}) == "quadratic"
    assert stage.shared_settings({})["athlete_root_reach_loss"] == "quadratic"
    assert (
        stage.shared_settings({"athlete_root_reach_loss": "cauchy"})["athlete_root_reach_loss"]
        == "cauchy"
    )
    with pytest.raises(ValueError, match="unsupported global athlete_root_reach_loss"):
        stage.shared_settings({"athlete_root_reach_loss": "wins"})


def test_original_root_disagreement_survives_exact_and_fallback_contact_state(tmp_path):
    row = dict(
        clip="pt0001",
        frame="f_0010.jpg",
        side="far",
        track_id="7",
        court_x="5.5",
        court_y="31.5",
        x0="930",
        x1="980",
        y0="180",
        y1="260",
        s6_original_court_x="5.4",
        s6_original_court_y="23",
        s6_root_status="derived",
        s6_root_proxy="original_tracker_box_native",
    )
    path = tmp_path / "players.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    exact = single.server_state(
        path, "pt0001", 10, np.array([950, 250]), player_name="P", stature_m=1.8
    )
    near = fallback.contact_state(path, "pt0001", 11, "far", player_name="P", stature_m=1.8)
    assert exact["court_centre_xy_m"] == near["court_centre_xy_m"] == [5.5, 31.5]
    witness = exact["root_observation"]
    assert near["root_observation"] == witness
    assert witness["source_frame"] == "f_0010.jpg"
    assert witness["original_court_xy_m"] == [5.4, 23.0]
    assert witness["original_derived_disagreement_m"] == pytest.approx(np.hypot(0.1, 8.5))
    assert (
        priors.evaluate(np.array([[5.5, 31.5, 3]]), [exact])["contacts"][0]["root_observation"]
        == witness
    )
    assert priors.root_observation({}) == {}
    assert (
        priors.root_observation(row | {"s6_original_court_x": ""})["root_observation"][
            "original_court_xy_m"
        ]
        is None
    )
