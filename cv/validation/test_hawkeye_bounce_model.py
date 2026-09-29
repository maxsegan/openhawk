"""Hawk-Eye bounce measurement: geometry helpers, round trip and aggregation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cv.validation import hawkeye_bounce_model as model


def test_topspin_axis_gives_a_downward_magnus_force():
    velocity = np.array([20.0, 0.0, 0.0])
    axis = model.topspin_axis(velocity)
    force = np.cross(axis, velocity)
    assert force[2] < 0.0
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0)


def test_topspin_axis_of_a_vertical_velocity_is_zero():
    assert np.allclose(model.topspin_axis(np.array([0.0, 0.0, 5.0])), np.zeros(3))


def test_apex_is_the_vertical_turning_point():
    times, positions, velocities, _ = model._dense(
        [0.0, 0.0, 0.5], [10.0, 0.0, 6.0], np.zeros(3), max_seconds=1.5
    )
    apex = model._apex(positions, velocities)
    assert apex is not None
    assert apex[2] == pytest.approx(float(positions[:, 2].max()), abs=1e-3)
    del times


def test_horizontal_distance_lookup_matches_the_path():
    _, positions, _, _ = model._dense(
        [1.0, 2.0, 0.5], [10.0, 0.0, 5.0], np.zeros(3), max_seconds=1.0
    )
    point = model._at_horizontal_distance(positions, [1.0, 2.0, 0.5], 4.0)
    assert point is not None
    assert math.hypot(point[0] - 1.0, point[1] - 2.0) == pytest.approx(4.0, abs=1e-2)


def test_horizontal_distance_lookup_returns_none_beyond_the_path():
    _, positions, _, _ = model._dense(
        [0.0, 0.0, 0.5], [5.0, 0.0, 1.0], np.zeros(3), max_seconds=0.3
    )
    assert model._at_horizontal_distance(positions, [0.0, 0.0, 0.5], 50.0) is None


def test_outgoing_solver_recovers_a_generated_rebound():
    """Generate a rebound, read off its landmarks, and refit the launch velocity."""
    bounce = np.array([2.0, 3.0, model.AERO_PARAMS["R_ball"]])
    velocity = np.array([9.0, 4.0, 6.0])
    _, positions, velocities, _ = model._dense(bounce, velocity, np.zeros(3), max_seconds=1.6)
    apex = model._apex(positions, velocities)
    arrival = model._at_horizontal_distance(positions, bounce, 8.0)
    pair = model.BouncePair(
        match_file="synthetic.csv",
        point_id="1_1_1_1",
        serve_num="1",
        strike_index=2,
        surface="hard",
        tour="atp",
        population="pair",
        hit=(0.0, 0.0, 1.0),
        peak_in=None,
        net=None,
        bounce=tuple(bounce),
        peak_out=tuple(apex),
        next_hit=tuple(arrival),
        measured_spin_rpm=None,
    )
    solved = model.solve_outgoing(pair, spin_rpm=0.0)
    assert solved is not None
    assert np.allclose(solved["outgoing_velocity"], velocity, atol=0.05)


def test_cross_prediction_matches_the_shipped_impact_model():
    from physics import impact

    prediction = model.cross_prediction(20.0, 6.0, 250.0, "hard")
    expected = impact.court_bounce(20.0, 6.0, 250.0, surface="hard")
    assert prediction["e_y_shipped"] == pytest.approx(expected.vy2 / 6.0)
    assert prediction["regime_shipped"] == expected.regime
    assert prediction["incidence_deg"] == pytest.approx(math.degrees(math.atan2(6.0, 20.0)))


def test_bucket_labels_use_the_configured_edges():
    assert model._bucket(22.0, model.SPEED_EDGES_MS) == "20-25"
    assert model._bucket(45.0, model.SPEED_EDGES_MS) == "30-inf"
    assert model._bucket(-1.0, model.ANGLE_EDGES_DEG) == "out_of_range"


def _row(surface: str, incidence: float, restitution: float, retention: float) -> dict:
    return {
        "surface": surface,
        "population": "pair",
        "incidence_deg": incidence,
        "speed_in_ms": 22.0,
        "vertical_in_ms": 6.0,
        "restitution": restitution,
        "horizontal_retention": retention,
        "restitution_spin_free": restitution - 0.15,
        "horizontal_retention_spin_free": retention - 0.11,
        "deflection_deg": 0.2,
        "apex_out_height_m": 1.1,
        "cross_e_y_shipped": 0.87,
        "cross_horizontal_retention_shipped": 0.67,
        "cross_regime_shipped": "grip",
    }


def test_surface_model_recovers_a_planted_slope():
    rows = [
        _row("hard", angle, 0.90 - 0.01 * (angle - 16.0), 0.66 - 0.005 * (angle - 16.0))
        for angle in np.linspace(8.0, 30.0, 60)
    ]
    fitted = model.fit_surface_model(rows)["hard"]
    assert fitted["at_model_spin"]["restitution_intercept"] == pytest.approx(0.90, abs=1e-6)
    assert fitted["at_model_spin"]["restitution_slope_per_deg"] == pytest.approx(-0.01, abs=1e-6)
    assert fitted["spin_free"]["restitution_intercept"] == pytest.approx(0.75, abs=1e-6)


def test_surface_model_skips_a_thin_surface():
    assert model.fit_surface_model([_row("clay", 16.0, 0.9, 0.66)] * 10) == {}


def test_summary_reports_the_gap_against_cross():
    rows = [_row("hard", 16.0, 0.75, 0.56) for _ in range(4)]
    summary = model.summarize(rows)
    assert summary["all"]["n"] == 4
    assert summary["all"]["restitution_gap_vs_cross"]["median"] == pytest.approx(0.75 - 0.87)
    assert summary["by_surface"]["hard"]["cross_regime_counts"] == {"grip": 4}


def test_corpus_files_are_balanced_and_deterministic(tmp_path):
    trajectory = tmp_path / "ball_trajectory"
    trajectory.mkdir()
    for index in range(4):
        (trajectory / f"atp_roland_garros_2019SM{index:03d}_ball_trajectory.csv").touch()
        (trajectory / f"atp_australian_open_2020MS{index:03d}_ball_trajectory.csv").touch()
    selected = model.corpus_files(tmp_path, 4)
    assert len(selected) == 4
    assert sum("roland_garros" in name for name in selected) == 2
    assert selected == model.corpus_files(tmp_path, 4)
