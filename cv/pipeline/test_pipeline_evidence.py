from __future__ import annotations

import pytest

from cv.pipeline.pipeline_evidence import (
    event_type_probability,
    select_physics_compatible_bounce,
    tracking_observation_weight,
)


def test_tracking_weight_rewards_consensus_and_marks_interpolation() -> None:
    single = tracking_observation_weight(0.8, "wasb")
    consensus = tracking_observation_weight(0.8, "wasb+tracknetv2")
    interpolated = tracking_observation_weight(
        0.8,
        "wasb+tracknetv2",
        interpolated=True,
    )
    assert consensus > single > interpolated
    assert 0.1 <= interpolated <= 1.0


def test_event_probability_preserves_strongest_type_hypothesis() -> None:
    row = {
        "native_p_bounce": 0.4,
        "claude_p_bounce": 0.8,
        "dense_p_bounce": 0.7,
        "meta_p_bounce": 0.9,
    }
    assert event_type_probability(row, "bounce") == pytest.approx(0.9)


def test_bounce_selection_requires_event_physics_overlap() -> None:
    candidates = [
        {"frame": 100.0, "court_xy": [4.0, 18.0], "probability": 0.9},
        {"frame": 150.0, "court_xy": [9.0, 2.0], "probability": 0.99},
    ]
    predicted = [{"frame": 101.0, "x": [4.2, 18.3, 0.033]}]
    selected = select_physics_compatible_bounce(candidates, predicted, fps=25.0)
    assert selected is not None
    assert selected["frame"] == 100.0
    assert selected["sigma_xy_m"] < 0.5


def test_bounce_selection_abstains_without_physics_witness() -> None:
    candidate = {"frame": 100.0, "court_xy": [4.0, 18.0], "probability": 0.99}
    assert select_physics_compatible_bounce([candidate], [], fps=25.0) is None
