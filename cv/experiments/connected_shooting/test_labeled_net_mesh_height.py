"""Full-mesh coordinates replay on both court sides without moving the incoming state."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.labeled_net_height_chart import project_net_height
from cv.pipeline.physics_knot_solver import net_tape_height


@pytest.mark.parametrize("start_y, vy", [(20.0, -14.0), (4.0, 14.0)])
@pytest.mark.parametrize("fraction", [1e-5, 0.08, 0.5, 1.0])
def test_mesh_coordinate_replays_full_physical_height_range(start_y, vy, fraction):
    source = np.array([5.0, start_y, 1.7, 1.0, vy, -8.0, 0.0, 0.0, 0.0])
    recovered, receipt = project_net_height(source, 0.0, 25.0, 60.0, "hard", mesh_fraction=fraction)
    xyz, _, _, impacts = measured_dynamics.simulate(
        recovered, 0.0, np.array([0.0, 25.0]), 60.0, "hard"
    )
    expected_height = measured_dynamics.R_BALL + fraction * net_tape_height(xyz[-1, 0])
    assert abs(xyz[-1, 1] - 11.885) < 1e-6
    assert abs(xyz[-1, 2] - expected_height) < 1e-6
    assert not impacts
    np.testing.assert_array_equal(recovered[[0, 1, 2, 3, 6, 7, 8]], source[[0, 1, 2, 3, 6, 7, 8]])
    assert receipt["height_support_is_explicit_reviewed_input"] is False
    assert receipt["height_support"] == "full_physical_mesh"


@pytest.mark.parametrize("fraction", [0.08, 0.5, 1.0])
def test_mesh_chart_handles_strong_curvature_without_earlier_ground(fraction):
    source = np.array([2.737, 25.718, 0.948, 6.474, -14.927, 3.867, 5.998, 1.444, -5.965])
    recovered, receipt = project_net_height(
        source,
        634.5,
        647.0,
        25.0,
        "hard",
        mesh_fraction=fraction,
        rebound_scales=(1.0132, 0.9861),
    )
    xyz, _, _, impacts = measured_dynamics.simulate(
        recovered,
        634.5,
        np.array([634.5, 647.0]),
        25.0,
        "hard",
        rebound_scales=(1.0132, 0.9861),
    )
    assert not impacts
    assert (
        abs(xyz[-1, 2] - (measured_dynamics.R_BALL + fraction * net_tape_height(xyz[-1, 0]))) < 1e-6
    )
    assert receipt["plane_error_m"] < 1e-6


def test_mesh_coordinate_rejects_simultaneous_ground_and_conflicting_review_override():
    source = np.array([5.0, 20.0, 1.7, 1.0, -14.0, 0.0, 0.0, 0.0, 0.0])
    for fraction in [0.0, -0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="mesh fraction"):
            project_net_height(source, 0.0, 25.0, 60.0, "hard", mesh_fraction=fraction)
    with pytest.raises(ValueError, match="mesh fraction"):
        project_net_height(source, 0.0, 25.0, 60.0, "hard", 0.0, mesh_fraction=0.5)
