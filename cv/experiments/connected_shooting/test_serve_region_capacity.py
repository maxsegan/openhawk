"""Accounting/geometry contracts; actual frozen cohort replay is a separate experiment."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import serve_region_capacity as capacity


def test_sphere_is_a_volume_and_does_not_snap_contact_to_surface():
    center = np.array([5.0, 24.0, 2.8])
    assert capacity.sphere_slack(center, center, 0.6) == 0.6
    assert capacity.sphere_slack(center + [0.3, 0.4, 0], center, 0.6) == pytest.approx(0.1)
    assert capacity.sphere_slack(center + [0, -2, 0], center, 0.6) == pytest.approx(-1.4)
    np.testing.assert_array_equal(center, [5, 24, 2.8])


@pytest.mark.parametrize("radius", [0, -1, True, np.inf, np.nan])
def test_invalid_radius_rejected(radius):
    with pytest.raises(ValueError):
        capacity.sphere_slack([0, 0, 0], [0, 0, 0], radius)


def test_failed_repairs_retained_in_denominator_and_not_counted_as_baseline_success():
    rows = [
        dict(point="held", status="held"),
        dict(
            point="retained",
            status="measured",
            repair_triggered=False,
            baseline=dict(synthetic_correct=True),
        ),
    ]
    rows.append(
        dict(
            point="lost",
            status="measured",
            repair_triggered=True,
            baseline=dict(synthetic_correct=True),
            arms={a: dict(status="held") for a in ("terminal_only", "contact_sphere")},
        )
    )
    summary = capacity.summarize(rows)
    assert summary["points"] == 3
    assert summary["baseline_correct"] == 2
    for arm in summary["arms"].values():
        assert arm["synthetic_correct"] == 1
        assert arm["lost"] == ["lost"]
        assert arm["held"] == 2


def test_optimizer_success_does_not_replace_spatial_truth_or_sphere_membership():
    arms = {
        a: dict(
            status="measured",
            sphere_slack_m=-0.1,
            score=dict(synthetic_correct=True),
            fit=dict(optimizer=dict(success=True)),
        )
        for a in ("terminal_only", "contact_sphere")
    }
    summary = capacity.summarize(
        [
            dict(
                point="p",
                status="measured",
                repair_triggered=True,
                baseline=dict(synthetic_correct=False),
                arms=arms,
            )
        ]
    )
    assert summary["arms"]["terminal_only"]["synthetic_correct"] == 1
    assert summary["arms"]["contact_sphere"]["synthetic_correct"] == 0
    assert summary["arms"]["contact_sphere"]["optimizer_converged"] == 1
