from __future__ import annotations

import math

import numpy as np
import pytest

from physics import reference


def _launch(speed: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([speed * math.cos(angle), 0.0, speed * math.sin(angle)])


@pytest.mark.parametrize(
    ("rpm", "angle_deg", "range_m", "incidence_deg"),
    [(0.0, 7.4, 22.77, 15.2), (4000.0, 15.2, 22.77, 24.1)],
)
def test_cross_2020_figure_6_trajectory_examples(
    rpm: float,
    angle_deg: float,
    range_m: float,
    incidence_deg: float,
) -> None:
    _, positions, velocities, _ = reference.RK4(
        0.0005,
        10_000,
        np.array([0.0, 0.0, 1.0]),
        _launch(30.0, angle_deg),
        np.array([0.0, rpm * 2.0 * math.pi / 60.0, 0.0]),
        params=reference.GROUNDSTROKE_PARAMS,
        verbose=False,
    )
    incidence = math.degrees(math.atan2(-velocities[-1, 2], velocities[-1, 0]))
    # Figure values are plotted/read to roughly the nearest 0.1 degree and the paper
    # does not publish its integration step or every physical constant.
    assert positions[-1, 0] == pytest.approx(range_m, abs=0.45)
    assert incidence == pytest.approx(incidence_deg, abs=0.5)


def test_cross_2020_zero_spin_zero_offset_grip_example() -> None:
    bounce = reference.court_bounce(
        np.array([1.0, 0.0, -2.5]),
        np.zeros(3),
        normal_cor=0.75,
        mu=0.73,
        e_x=0.0,
        deformation_offset_m=0.0,
    )
    assert bounce.regime == "grip"
    assert bounce.velocity[0] == pytest.approx(1.0 / (1.0 + 0.55), abs=1e-12)


@pytest.mark.parametrize(
    ("racket_angle_deg", "racket_motion_deg", "speed", "angle", "spin"),
    [
        (0.0, 0.0, 34.9, 6.5, 160.0),
        (0.0, 30.0, 31.8, 13.6, -16.7),
        (-5.0, 0.0, 34.8, 1.1, 106.0),
        (-5.0, 30.0, 30.4, 8.6, -65.8),
    ],
)
def test_cross_2005_figure_f_examples(
    racket_angle_deg: float,
    racket_motion_deg: float,
    speed: float,
    angle: float,
    spin: float,
) -> None:
    racket_motion = math.radians(racket_motion_deg)
    racket_velocity = np.array([-20.0 * math.cos(racket_motion), 20.0 * math.sin(racket_motion)])
    face = math.radians(racket_angle_deg)
    outward_normal = np.array([-math.cos(face), math.sin(face)])
    result = reference.planar_racket_impact(
        np.array([15.0, 0.0]), racket_velocity, outward_normal, 400.0
    )
    result_angle = math.degrees(math.atan2(result.velocity[1], -result.velocity[0]))
    assert np.linalg.norm(result.velocity) == pytest.approx(speed, abs=0.15)
    assert result_angle == pytest.approx(angle, abs=0.15)
    assert result.spin == pytest.approx(spin, abs=0.2)


def test_variable_tension_numeric_examples() -> None:
    coefficients = reference.racket_coefficients(242.667)
    assert coefficients == pytest.approx(
        {"e_n": 0.42, "e_t": 0.11, "a": 0.648, "b": 0.3, "c": 0.4, "d": 0.583},
        abs=5e-5,
    )
    tension = 55.0 * 4.4497375
    assert reference.effective_tension(tension, 28.0) / 4.4497375 == pytest.approx(55.0)
    assert reference.effective_tension(tension, 40.0) / 4.4497375 == pytest.approx(60.5)


def test_empirical_coefficients_saturate_and_recover_zero_spin_limit() -> None:
    assert reference.drag_coefficient(0.0) == pytest.approx(0.508)
    assert reference.lift_coefficient(0.0) == 0.0
    assert reference.lift_coefficient(1e6) == pytest.approx(1.0 / 2.022, rel=2e-6)
    assert reference.drag_coefficient(1e6) == pytest.approx(0.508 + 22.503**-0.4, rel=2e-6)


def test_measured_spin_decay_retains_98_percent_over_6_4_metres() -> None:
    speed = 30.0
    _, _, _, spins = reference.RK4(
        0.0002,
        2000,
        np.zeros(3),
        np.array([speed, 0.0, 0.0]),
        np.array([0.0, 300.0, 0.0]),
        gravity=False,
        drag=False,
        lift=False,
        params=reference.default_params,
        verbose=False,
    )
    index = round((6.4 / speed) / 0.0002)
    assert spins[index, 1] / spins[0, 1] == pytest.approx(0.98, rel=2e-5)


def test_surface_chart_values_are_transcribed_exactly() -> None:
    assert reference.SURFACES == {
        "grass": {"cor": 0.60, "cof": 0.60, "bounce_height_fraction": 0.36},
        "hard": {"cor": 0.83, "cof": 0.70, "bounce_height_fraction": 0.69},
        "clay": {"cor": 0.85, "cof": 0.80, "bounce_height_fraction": 0.72},
    }


@pytest.mark.parametrize("surface", ["grass", "hard", "clay"])
def test_bounce_does_not_add_rigid_body_energy(surface: str) -> None:
    rng = np.random.default_rng(20260902)
    for _ in range(250):
        direction = rng.uniform(-math.pi, math.pi)
        horizontal_speed = rng.uniform(3.0, 50.0)
        velocity = np.array(
            [
                horizontal_speed * math.cos(direction),
                horizontal_speed * math.sin(direction),
                -rng.uniform(0.5, 25.0),
            ]
        )
        spin = rng.uniform(-650.0, 650.0, 3)
        result = reference.court_bounce(velocity, spin, surface)
        assert reference.rigid_body_energy(
            result.velocity, result.spin
        ) <= reference.rigid_body_energy(velocity, spin) * (1.0 + 1e-12)


def test_court_bounce_is_symmetric_under_mirror() -> None:
    # Polar vectors use Q under reflection; angular velocity is an axial vector and
    # therefore uses det(Q)Q.
    q = np.diag([1.0, -1.0, 1.0])
    axial_q = -q
    velocity = np.array([22.0, 3.0, -8.0])
    spin = np.array([80.0, 310.0, 45.0])
    original = reference.court_bounce(velocity, spin, "clay")
    mirrored = reference.court_bounce(q @ velocity, axial_q @ spin, "clay")
    assert mirrored.regime == original.regime
    assert mirrored.velocity == pytest.approx(q @ original.velocity, abs=1e-12)
    assert mirrored.spin == pytest.approx(axial_q @ original.spin, abs=1e-12)


def test_rifle_spin_generates_neither_lift_nor_spin_ratio() -> None:
    velocity = np.array([30.0, -4.0, 2.0])
    spin = 20.0 * velocity
    assert reference.spin_ratio(velocity, spin) == pytest.approx(0.0, abs=1e-14)
    with_lift = reference.accel(velocity, spin)
    without_lift = reference.accel(velocity, spin, lift=False)
    assert with_lift == pytest.approx(without_lift, abs=1e-12)
