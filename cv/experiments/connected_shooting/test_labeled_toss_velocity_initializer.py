"""Bounded coupled-likelihood initialization, retention and failure contracts."""

import numpy as np
import pytest
from cv.experiments.connected_shooting import labeled_toss_velocity_initializer as initializer


def test_refine_repairs_coupled_solution_without_widening_box():
    raw = np.array([1.0, 30.0, -9.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 0.1]])
    target = matrix @ raw

    def residual(velocity):
        return matrix @ velocity - target

    result, receipt = initializer.refine(raw, residual)
    assert np.all(abs(result) <= 12)
    assert receipt["chosen_cost"] < receipt["initial_cost"] * 0.01
    assert np.array_equal(raw, [1.0, 30.0, -9.0])
    assert receipt["fixed_contact_xyz_and_epoch"]
    assert np.isclose(receipt["chosen_cost"], np.sum(residual(result) ** 2))


def test_exact_initial_state_is_retained_without_drift():
    raw = np.array([0.3, -0.2, -3.0])
    result, receipt = initializer.refine(raw, lambda v: v - raw)
    assert np.array_equal(result, raw)
    assert receipt["retained_original_clipped_seed"]
    assert receipt["chosen_cost"] == 0


def test_invalid_initial_likelihood_is_failure_not_zero_error():
    with pytest.raises(ValueError, match="finite and nonempty"):
        initializer.refine(np.zeros(3), lambda v: np.array([np.nan]))


def test_optimizer_failure_retains_explicit_valid_start(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("controlled failure")

    monkeypatch.setattr(initializer, "least_squares", fail)
    chosen, receipt = initializer.refine(
        np.array([0.0, 13.0, -2.0]), lambda v: v - np.array([0.0, 2.0, -1.0])
    )
    assert np.array_equal(chosen, [0.0, 12.0, -2.0])
    assert receipt["termination"]["status"] == "execution_failure_retained_valid_input_iterate"
