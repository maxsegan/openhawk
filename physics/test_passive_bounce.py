"""Tests for the passive coupled rebound approximation.

These check the model's own invariants -- the normal law the caller asked for, the Coulomb
bound, slip kinematics, rotation covariance and energy dissipation -- not agreement with any
calibrated bounce measurement. ``physics.passive_bounce`` is explicitly uncalibrated, so
asserting a particular speed-retention number here would invent evidence it does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics import impact
from physics.passive_bounce import APPROXIMATIONS, MODEL_ID, coupled_rebound

SURFACES = sorted(impact.SURFACES)


def _energy(velocity: np.ndarray, spin: np.ndarray) -> float:
    translational = 0.5 * impact.M_BALL * float(velocity @ velocity)
    rotational = 0.5 * impact.J_BALL * float(spin @ spin)
    return translational + rotational


def _slip(velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
    return velocity[:2] + impact.R_BALL * np.array([-spin[1], spin[0]])


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _rolling_state(spin: np.ndarray, descent: float) -> np.ndarray:
    """Velocity whose bottom point is exactly at rest for ``spin`` (zero slip, bit for bit)."""
    return np.array([impact.R_BALL * spin[1], -impact.R_BALL * spin[0], -descent])


def _random_cases(count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, str, float, float]]:
    """Wide sweep: every surface, near-zero through fast velocities, small through large spin."""
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        # log-uniform magnitudes reach 1e-6 m/s and 45 m/s within the same sweep
        horizontal = 10.0 ** rng.uniform(-6.0, 1.65)
        descent = 10.0 ** rng.uniform(-6.0, 1.5)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        velocity = np.array([horizontal * np.cos(heading), horizontal * np.sin(heading), -descent])
        spin = rng.uniform(-1.0, 1.0, size=3) * 10.0 ** rng.uniform(-2.0, 3.0)
        cases.append(
            (
                velocity,
                spin,
                SURFACES[index % len(SURFACES)],
                float(rng.uniform(0.0, 1.0)),
                float(rng.uniform(0.0, 1.0)),
            )
        )
    return cases


def test_rolling_contact_takes_no_tangential_impulse():
    spin = np.array([-210.0, 140.0, 75.0])
    velocity = _rolling_state(spin, descent=9.0)
    assert _slip(velocity, spin).tolist() == [0.0, 0.0]

    result = coupled_rebound(velocity, spin, restitution=0.8, surface="hard")

    assert result.receipt["tangential_impulse_limited_by"] == "no_slip"
    assert result.receipt["tangential_impulse_per_mass_mps"] == [0.0, 0.0]
    np.testing.assert_array_equal(result.velocity[:2], velocity[:2])
    np.testing.assert_array_equal(result.spin, spin)
    assert result.velocity[2] == pytest.approx(0.8 * 9.0)
    # only the normal channel dissipated
    assert result.receipt["kinetic_energy_ratio"] < 1.0
    assert result.receipt["horizontal_speed_retained_ratio"] == pytest.approx(1.0)


def test_overspin_gains_forward_speed_while_total_energy_falls():
    velocity = np.array([5.0, 0.0, -8.0])
    spin = np.array([0.0, 3.0 * 5.0 / impact.R_BALL, 0.0])  # bottom point runs backwards
    assert _slip(velocity, spin)[0] < 0.0

    result = coupled_rebound(velocity, spin, restitution=0.85, surface="clay")

    # friction pushes forward, so translation gains at the expense of spin
    assert result.velocity[0] > velocity[0]
    assert result.spin[1] < spin[1]
    assert _energy(result.velocity, result.spin) < _energy(velocity, spin)
    assert result.receipt["kinetic_energy_ratio"] < 1.0
    assert result.receipt["horizontal_speed_retained_ratio"] > 1.0


def test_pure_vertical_drop_with_spin_gets_a_finite_horizontal_response():
    velocity = np.array([0.0, 0.0, -6.0])
    spin = np.array([0.0, 300.0, 0.0])

    result = coupled_rebound(velocity, spin, restitution=0.7, surface="hard")

    assert np.isfinite(result.velocity).all() and np.isfinite(result.spin).all()
    # slip points along -x, so the ball is kicked along +x; nothing is clipped away
    assert result.velocity[0] > 0.0
    assert result.velocity[1] == pytest.approx(0.0, abs=1e-15)
    assert result.spin[1] < spin[1]
    assert _energy(result.velocity, result.spin) < _energy(velocity, spin)
    # no incoming horizontal speed means there is no retention ratio to report
    assert result.receipt["horizontal_speed_in_mps"] == 0.0
    assert result.receipt["horizontal_speed_retained_ratio"] is None


def test_zero_horizontal_and_zero_spin_stays_purely_vertical():
    velocity = np.array([0.0, 0.0, -6.0])
    spin = np.zeros(3)

    result = coupled_rebound(velocity, spin, restitution=0.6, surface="grass")

    assert result.velocity[0] == 0.0 and result.velocity[1] == 0.0
    assert result.velocity[2] == pytest.approx(3.6)
    np.testing.assert_array_equal(result.spin, np.zeros(3))
    assert result.receipt["horizontal_speed_retained_ratio"] is None


def test_rebound_is_equivariant_under_court_rotation():
    velocity = np.array([6.0, -19.0, -7.5])
    spin = np.array([160.0, -240.0, 190.0])
    rotation = _rotation_z(0.91)

    baseline = coupled_rebound(velocity, spin, restitution=0.72, surface="clay")
    rotated = coupled_rebound(
        rotation @ velocity, rotation @ spin, restitution=0.72, surface="clay"
    )

    np.testing.assert_allclose(rotated.velocity, rotation @ baseline.velocity, atol=1e-12)
    np.testing.assert_allclose(rotated.spin, rotation @ baseline.spin, atol=1e-9)
    assert rotated.receipt["kinetic_energy_ratio"] == pytest.approx(
        baseline.receipt["kinetic_energy_ratio"]
    )


@pytest.mark.parametrize("surface", SURFACES)
def test_total_energy_never_increases_on_any_surface(surface):
    for velocity, spin, _sweep_surface, restitution, tangential in _random_cases(240, seed=11):
        result = coupled_rebound(
            velocity,
            spin,
            restitution=restitution,
            surface=surface,
            tangential_restitution=tangential,
        )
        energy_in = _energy(velocity, spin)
        energy_out = _energy(result.velocity, result.spin)
        assert energy_out <= energy_in * (1.0 + 1e-12), (velocity, spin, surface)
        assert result.receipt["kinetic_energy_ratio"] <= 1.0 + 1e-12
        assert np.isfinite(result.velocity).all() and np.isfinite(result.spin).all()


def test_tangential_impulse_stays_inside_the_coulomb_and_grip_budgets():
    for velocity, spin, surface, restitution, tangential in _random_cases(240, seed=23):
        result = coupled_rebound(
            velocity,
            spin,
            restitution=restitution,
            surface=surface,
            tangential_restitution=tangential,
        )
        receipt = result.receipt
        magnitude = receipt["tangential_impulse_magnitude_mps"]
        friction_budget = impact.SURFACES[surface][1] * (1.0 + restitution) * -velocity[2]
        assert receipt["friction_budget_per_mass_mps"] == pytest.approx(friction_budget)
        assert magnitude <= friction_budget * (1.0 + 1e-12)
        assert magnitude <= receipt["grip_budget_per_mass_mps"] * (1.0 + 1e-12)
        # the impulse opposes the incoming slip and points nowhere else
        impulse = np.asarray(receipt["tangential_impulse_per_mass_mps"])
        slip_in = np.asarray(receipt["slip_in_mps"])
        cross = float(impulse[0] * slip_in[1] - impulse[1] * slip_in[0])
        assert float(impulse @ slip_in) <= 0.0
        assert abs(cross) <= 1e-12 * magnitude * receipt["slip_speed_in_mps"] + 1e-300


def test_normal_law_and_normal_spin_are_untouched_by_friction():
    for velocity, spin, surface, restitution, tangential in _random_cases(240, seed=37):
        result = coupled_rebound(
            velocity,
            spin,
            restitution=restitution,
            surface=surface,
            tangential_restitution=tangential,
        )
        assert result.velocity[2] == -restitution * velocity[2]
        assert result.spin[2] == spin[2]


def test_gripping_contact_reverses_slip_by_the_tangential_restitution():
    velocity = np.array([1.0, 0.0, -20.0])  # steep: the friction budget far exceeds the slip
    spin = np.zeros(3)

    result = coupled_rebound(
        velocity, spin, restitution=0.8, surface="hard", tangential_restitution=0.25
    )

    assert result.receipt["tangential_impulse_limited_by"] == "tangential_grip"
    expected = -0.25 * np.asarray(result.receipt["slip_in_mps"])
    np.testing.assert_allclose(result.receipt["slip_out_mps"], expected, atol=1e-12)
    np.testing.assert_allclose(_slip(result.velocity, result.spin), expected, atol=1e-12)


def test_sliding_contact_is_bounded_by_the_per_surface_friction_chart():
    velocity = np.array([30.0, 0.0, -2.0])  # shallow: the Coulomb budget binds
    spin = np.zeros(3)

    outgoing = {}
    for surface in SURFACES:
        result = coupled_rebound(velocity, spin, restitution=0.5, surface=surface)
        mu = impact.SURFACES[surface][1]
        assert result.receipt["tangential_impulse_limited_by"] == "coulomb_friction"
        assert result.receipt["friction_coefficient"] == mu
        assert result.velocity[0] == pytest.approx(30.0 - mu * 1.5 * 2.0)
        assert float(np.linalg.norm(result.receipt["slip_out_mps"])) > 0.0  # still sliding
        outgoing[surface] = result.velocity[0]

    # a rougher chart entry must brake more: clay (0.80) > hard (0.70) > grass (0.60)
    assert outgoing["clay"] < outgoing["hard"] < outgoing["grass"]


def test_response_is_continuous_through_vanishing_slip():
    spin = np.array([-90.0, 130.0, 40.0])
    velocity = _rolling_state(spin, descent=7.0)
    reference = coupled_rebound(velocity, spin, restitution=0.75, surface="clay")
    direction = np.array([0.6, -0.8, 0.0])

    previous = None
    for epsilon in (1e-2, 1e-4, 1e-6, 1e-8):
        nudged = coupled_rebound(
            velocity + epsilon * direction, spin, restitution=0.75, surface="clay"
        )
        assert nudged.receipt["tangential_impulse_limited_by"] == "tangential_grip"
        velocity_gap = float(np.linalg.norm(nudged.velocity - reference.velocity))
        spin_gap = float(np.linalg.norm(nudged.spin - reference.spin)) * impact.R_BALL
        deviation = velocity_gap + spin_gap
        # the gap shrinks with the perturbation instead of jumping at the no-slip boundary
        assert deviation <= 10.0 * epsilon
        if previous is not None:
            assert deviation < previous
        previous = deviation


def test_float_array_inputs_are_not_mutated():
    velocity = np.array([9.0, -3.0, -11.0])
    spin = np.array([120.0, 450.0, -80.0])
    velocity_copy = velocity.copy()
    spin_copy = spin.copy()

    result = coupled_rebound(velocity, spin, restitution=0.9, surface="grass")

    np.testing.assert_array_equal(velocity, velocity_copy)
    np.testing.assert_array_equal(spin, spin_copy)
    assert not np.shares_memory(result.velocity, velocity)
    assert not np.shares_memory(result.spin, spin)


def test_receipt_records_the_model_identity_and_its_unmeasured_physics():
    velocity = np.array([12.0, 4.0, -6.0])
    spin = np.array([50.0, 260.0, -30.0])

    result = coupled_rebound(
        velocity, spin, restitution=0.83, surface="hard", tangential_restitution=0.1
    )
    receipt = result.receipt

    assert receipt["model"] == MODEL_ID
    assert "NOT the Cross 2020 deformation law" in receipt["calibration"]
    assert receipt["approximations"] == list(APPROXIMATIONS)
    assert receipt["surface"] == "hard"
    assert receipt["friction_coefficient"] == impact.SURFACES["hard"][1]
    assert receipt["normal_restitution"] == 0.83
    assert receipt["tangential_restitution"] == 0.1
    assert receipt["normal_impulse_per_mass_mps"] == pytest.approx(1.83 * 6.0)
    assert receipt["ball_constants"] == {
        "mass_kg": impact.M_BALL,
        "radius_m": impact.R_BALL,
        "inertia_kg_m2": impact.J_BALL,
        "alpha": impact.ALPHA,
    }
    np.testing.assert_allclose(receipt["slip_in_mps"], _slip(velocity, spin), atol=1e-12)
    np.testing.assert_allclose(
        receipt["slip_out_mps"], _slip(result.velocity, result.spin), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(receipt["tangential_impulse_per_mass_mps"]),
        result.velocity[:2] - velocity[:2],
        atol=1e-12,
    )
    assert receipt["kinetic_energy_in_j"] == pytest.approx(_energy(velocity, spin))
    assert receipt["kinetic_energy_out_j"] == pytest.approx(_energy(result.velocity, result.spin))
    assert receipt["kinetic_energy_ratio"] == pytest.approx(
        receipt["kinetic_energy_out_j"] / receipt["kinetic_energy_in_j"]
    )
    assert receipt["horizontal_speed_retained_ratio"] == pytest.approx(
        float(np.linalg.norm(result.velocity[:2])) / float(np.linalg.norm(velocity[:2]))
    )


def test_result_is_frozen():
    result = coupled_rebound(
        np.array([5.0, 0.0, -5.0]), np.zeros(3), restitution=0.8, surface="hard"
    )
    with pytest.raises(AttributeError):
        result.velocity = np.zeros(3)


@pytest.mark.parametrize(
    "override",
    [
        {"velocity": np.array([np.nan, 0.0, -5.0])},
        {"velocity": np.array([1.0, np.inf, -5.0])},
        {"velocity": np.array([1.0, 0.0, -np.inf])},
        {"spin": np.array([np.nan, 0.0, 0.0])},
        {"spin": np.array([0.0, np.inf, 0.0])},
        {"velocity": np.array([1.0, 0.0])},
        {"velocity": np.zeros((3, 3))},
        {"spin": np.array([0.0, 0.0, 0.0, 0.0])},
        {"velocity": np.array([1.0, 0.0, 5.0])},  # ascending
        {"velocity": np.array([1.0, 0.0, 0.0])},  # grazing: no normal impulse
        {"restitution": -0.01},
        {"restitution": 1.01},
        {"restitution": np.nan},
        {"tangential_restitution": -0.01},
        {"tangential_restitution": 1.5},
        {"tangential_restitution": np.inf},
        {"surface": "carpet"},
        {"surface": "Hard"},
    ],
)
def test_malformed_inputs_are_rejected(override):
    call = {
        "velocity": np.array([8.0, 2.0, -6.0]),
        "spin": np.array([10.0, 200.0, -5.0]),
        "restitution": 0.8,
        "surface": "hard",
        **override,
    }
    incoming_velocity = call.pop("velocity")
    incoming_spin = call.pop("spin")
    with pytest.raises(ValueError):
        coupled_rebound(incoming_velocity, incoming_spin, **call)
