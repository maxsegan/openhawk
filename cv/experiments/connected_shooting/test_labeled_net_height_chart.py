import numpy as np
import pytest
from scipy.optimize import brentq

from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.labeled_net_height_chart import project_net_height
from cv.pipeline.physics_knot_solver import net_tape_height


@pytest.mark.parametrize("initial_vz", [0.5, -15.0])
def test_recovers_two_velocity_components_from_independent_crossing_state(initial_vz):
    source = np.array([5.0, 20.0, 2.0, 1.0, -18.0, 1.0, 0.0, 0.0, 0.0])

    def state(frame):
        return measured_dynamics.simulate(source, 0.0, np.array([0.0, frame]), 60.0, "hard")[0][-1]

    frame = brentq(lambda t: state(t)[1] - 11.885, 20.0, 40.0)
    xyz = state(frame)
    offset = xyz[2] - net_tape_height(xyz[0])
    wrong = source.copy()
    wrong[4:6] = [-12.0, initial_vz]
    recovered, receipt = project_net_height(wrong, 0.0, frame, 60.0, "hard", offset)
    np.testing.assert_allclose(recovered, source, atol=1e-6)
    assert receipt["height_error_m"] < 1e-7
    assert receipt["plane_error_m"] < 1e-7


def test_aerodynamic_seed_failure_does_not_reject_reachable_tape_height():
    # AO2022 point2's original incoming state: strong spin makes the gravity-only
    # height initializer touch ground before reaching the requested net plane.
    theta = np.array(
        [
            2.7367059509445024,
            25.7181185178242,
            0.9478576226062198,
            6.474354031170952,
            -14.92689658496174,
            3.86663388726928,
            5.997887397500267,
            1.4444863360210862,
            -5.9654439916911315,
        ]
    )
    scales = (1.0132168660248229, 0.9860589407404869)
    recovered, receipt = project_net_height(
        theta, 634.5, 647.0, 25.0, "hard", 0.0, rebound_scales=scales
    )
    assert receipt["initial_seed_failures"]
    assert receipt["supported_airborne_vz_seed_mps"] > receipt["initial_airborne_vz_seed_mps"]
    xyz = measured_dynamics.simulate(
        recovered, 634.5, np.array([634.5, 647.0]), 25.0, "hard", rebound_scales=scales
    )[0][-1]
    assert abs(xyz[1] - 11.885) < 1e-6
    assert abs(xyz[2] - net_tape_height(xyz[0])) < 1e-6
    np.testing.assert_array_equal(recovered[[0, 1, 2, 3, 6, 7, 8]], theta[[0, 1, 2, 3, 6, 7, 8]])


def test_newton_ground_trial_backtracks_to_reachable_lower_mesh():
    # Numerical regression from a supplied-input cold state: a valid seed's
    # Newton step crossed the first-ground boundary, aborting before fitting.
    theta = np.array(
        [
            2.396482782976493,
            -1.4139026077800967,
            0.7447899385907546,
            -3.871386853411394,
            16.9399323795801,
            6.316950406931244,
            5.999990603394208,
            -5.999991940776708,
            -5.250169079286552,
        ]
    )
    scales = (0.9943296482926702, 0.9529811076343955)
    recovered, receipt = project_net_height(
        theta,
        189.0,
        201.5,
        25.0,
        "hard",
        mesh_fraction=1.0001e-5,
        rebound_scales=scales,
    )
    assert any(row["stage"] == "newton" for row in receipt["airborne_domain_backtracks"])
    xyz, _, _, impacts = measured_dynamics.simulate(
        recovered, 189.0, np.array([189.0, 201.5]), 25.0, "hard", rebound_scales=scales
    )
    assert not impacts
    assert receipt["prior_ground_impacts"] == 0
    assert abs(xyz[-1, 1] - 11.885) < 1e-6
    assert abs(xyz[-1, 2] - receipt["required_net_height_m"]) < 1e-6
    np.testing.assert_array_equal(recovered[[0, 1, 2, 3, 6, 7, 8]], theta[[0, 1, 2, 3, 6, 7, 8]])


def test_below_ground_requested_net_endpoint_remains_unsupported():
    theta = np.array([5.0, 20.0, 2.0, 1.0, -18.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="(airborne trial|did not converge|does not approach)"):
        project_net_height(theta, 0.0, 30.0, 60.0, "hard", height_offset_m=-2.0)


def test_unrelated_projector_error_is_not_treated_as_airborne_boundary(monkeypatch):
    from cv.experiments.connected_shooting import labeled_net_height_chart as chart

    def broken(*args, **kwargs):
        raise ValueError("projected state does not approach the physical net plane")

    monkeypatch.setattr(chart, "project_net_velocity", broken)
    theta = np.array([5.0, 20.0, 2.0, 1.0, -18.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="does not approach"):
        chart.project_net_height(theta, 0.0, 30.0, 60.0, "hard", mesh_fraction=0.5)
