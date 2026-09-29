"""Literature-backed tennis-ball flight and impact reference model.

The default aerodynamic profile uses Stepanek's empirical spin-ratio curves as
reproduced by Cross (2000), with the saturation retained.  ``GROUNDSTROKE_PARAMS``
selects the simpler constant-drag, linear-lift, constant-spin assumptions used for
the numerical examples in Cross (2020).  The public RK4 and sampling interfaces
mirror :mod:`physics.flight` so validation code can substitute this module without
changing the production fitter.

Coordinates are right handed with court normal ``+z``.  Magnus acceleration points
along ``spin x velocity``.  A positive court-bounce topspin component is therefore
along ``cross(+z, horizontal_velocity)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

G = 9.81
M_BALL = 0.057
R_BALL = 0.033
ALPHA = 0.55
J_BALL = ALPHA * M_BALL * R_BALL**2
RHO_AIR = 1.21

# Cross and Lindsey (2014) observed about 2% spin loss over a 6.4 m flight.
SPIN_DECAY_PER_M = -math.log(0.98) / 6.4

# Tennis Industry chart.  COR is the chart's bulk speed ratio, not a measured
# normal COR; see court_normal_cor for the explicit mapping used here.
SURFACES = {
    "grass": {"cor": 0.60, "cof": 0.60, "bounce_height_fraction": 0.36},
    "hard": {"cor": 0.83, "cof": 0.70, "bounce_height_fraction": 0.69},
    "clay": {"cor": 0.85, "cof": 0.80, "bounce_height_fraction": 0.72},
}

default_params: dict[str, float | str] = {
    "g": G,
    "m_ball": M_BALL,
    "R_ball": R_BALL,
    "J_ball": J_BALL,
    "rho_air": RHO_AIR,
    "coefficient_model": "stepanek",
    "spin_decay_per_m": SPIN_DECAY_PER_M,
}

GROUNDSTROKE_PARAMS: dict[str, float | str] = {
    **default_params,
    "coefficient_model": "cross_2020",
    "C_drag": 0.55,
    "C_lift_slope": 0.6,
    "spin_decay_per_m": 0.0,
}


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def spin_ratio(velocity: np.ndarray, spin: np.ndarray, radius: float = R_BALL) -> float:
    """Return ``R*|omega_perpendicular|/|v|`` for the Magnus-active spin.

    The cited two-dimensional correlations were measured with spin perpendicular
    to the flight direction.  Excluding rifle spin is the coordinate-free extension
    that preserves their experiment and prevents axial spin from producing lift.
    """

    velocity = _finite_vector(velocity, "velocity")
    spin = _finite_vector(spin, "spin")
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-12:
        return 0.0
    perpendicular = spin - np.dot(spin, velocity / speed) * velocity / speed
    return max(0.0, float(radius) * float(np.linalg.norm(perpendicular)) / speed)


def drag_coefficient(spin_parameter: float, model: str = "stepanek") -> float:
    """Return the empirical drag coefficient for a non-negative spin ratio.

    Stepanek/Cross: ``Cd = .508 + [22.503 + 4.196*S^(-5/2)]^(-2/5)``.
    The limiting value at zero spin is evaluated analytically.
    """

    ratio = abs(float(spin_parameter))
    if model == "cross_2020":
        return 0.55
    if model != "stepanek":
        raise ValueError(f"unknown aerodynamic coefficient model: {model}")
    if ratio <= 1e-12:
        return 0.508
    return 0.508 + (22.503 + 4.196 * ratio**-2.5) ** -0.4


def lift_coefficient(spin_parameter: float, model: str = "stepanek") -> float:
    """Return lift magnitude, including the empirical high-spin saturation.

    Stepanek/Cross: ``Cl = 1/[2.022 + 0.981/S]``.  Its asymptote is
    ``1/2.022 = 0.4946``.  Cross (2020) instead uses ``Cl = 0.6*S``.
    """

    ratio = abs(float(spin_parameter))
    if ratio <= 1e-12:
        return 0.0
    if model == "cross_2020":
        return 0.6 * ratio
    if model != "stepanek":
        raise ValueError(f"unknown aerodynamic coefficient model: {model}")
    return 1.0 / (2.022 + 0.981 / ratio)


def reynolds_number(
    speed: float,
    *,
    radius: float = R_BALL,
    rho_air: float = RHO_AIR,
    dynamic_viscosity: float = 1.81e-5,
) -> float:
    """Return ``rho*v*(2R)/mu`` for reporting the correlation's validity range."""

    return float(rho_air) * abs(float(speed)) * 2.0 * float(radius) / float(dynamic_viscosity)


def accel(
    v: np.ndarray,
    w: np.ndarray,
    gravity: bool = True,
    drag: bool = True,
    lift: bool = True,
    params: Mapping[str, float | str] = default_params,
) -> np.ndarray:
    """Compute gravity, quadratic drag, and vector Magnus acceleration."""

    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    ax = 0.0
    ay = 0.0
    az = -float(params["g"]) if gravity else 0.0
    if speed <= 1e-12:
        return np.array([ax, ay, az])
    radius = float(params["R_ball"])
    force_scale = 0.5 * float(params["rho_air"]) * math.pi * radius**2 / float(params["m_ball"])
    model = str(params.get("coefficient_model", "stepanek"))
    drag_model = str(params.get("drag_model", model))
    lift_model = str(params.get("lift_model", model))
    cross_x = wy * vz - wz * vy
    cross_y = wz * vx - wx * vz
    cross_z = wx * vy - wy * vx
    cross_norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    ratio = radius * cross_norm / (speed * speed)
    if drag:
        cd = float(params.get("C_drag", drag_coefficient(ratio, drag_model)))
        drag_scale = -force_scale * cd * speed
        ax += drag_scale * vx
        ay += drag_scale * vy
        az += drag_scale * vz
    if lift and cross_norm > 1e-12:
        cl = lift_coefficient(ratio, lift_model)
        if lift_model == "cross_2020":
            cl = float(params.get("C_lift_slope", 0.6)) * ratio
        lift_scale = force_scale * cl * speed**2 / cross_norm
        ax += lift_scale * cross_x
        ay += lift_scale * cross_y
        az += lift_scale * cross_z
    return np.array([ax, ay, az])


def w_dot(
    v: np.ndarray,
    w: np.ndarray,
    params: Mapping[str, float | str] = default_params,
) -> np.ndarray:
    """Distance-exponential spin decay fitted to the reported 2% loss over 6.4 m."""

    speed = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)
    scale = -float(params.get("spin_decay_per_m", SPIN_DECAY_PER_M)) * speed
    return scale * np.asarray(w, dtype=float)


def rk4_step(
    x: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dt: float,
    gravity: bool = True,
    drag: bool = True,
    lift: bool = True,
    aerodrag: bool = True,
    params: Mapping[str, float | str] = default_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance the coupled state by one fourth-order Runge--Kutta step."""

    position = _finite_vector(x, "position")
    velocity = _finite_vector(v, "velocity")
    spin = _finite_vector(w, "spin")

    def derivatives(vel: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = accel(vel, omega, gravity=gravity, drag=drag, lift=lift, params=params)
        dw = w_dot(vel, omega, params) if aerodrag else np.zeros(3)
        return a, dw

    a1, w1 = derivatives(velocity, spin)
    v2 = velocity + 0.5 * dt * a1
    o2 = spin + 0.5 * dt * w1
    a2, w2 = derivatives(v2, o2)
    v3 = velocity + 0.5 * dt * a2
    o3 = spin + 0.5 * dt * w2
    a3, w3 = derivatives(v3, o3)
    v4 = velocity + dt * a3
    o4 = spin + dt * w3
    a4, w4 = derivatives(v4, o4)
    return (
        position + dt * (velocity + 2.0 * v2 + 2.0 * v3 + v4) / 6.0,
        velocity + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0,
        spin + dt * (w1 + 2.0 * w2 + 2.0 * w3 + w4) / 6.0,
    )


def sample_states(
    x0: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    times_s: np.ndarray,
    *,
    substeps_per_second: float = 100.0,
    params: Mapping[str, float | str] = default_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a bounce-free flight at non-negative elapsed times."""

    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or np.any(times < -1e-12):
        raise ValueError("flight sample times must be a one-dimensional nonnegative array")
    output = tuple(np.empty((len(times), 3), dtype=float) for _ in range(3))
    if not len(times):
        return output
    order = np.argsort(times)
    position = _finite_vector(x0, "position").copy()
    velocity = _finite_vector(v0, "velocity").copy()
    spin = _finite_vector(w0, "spin").copy()
    current = 0.0
    maximum_step = 1.0 / float(substeps_per_second)
    for output_index in order:
        target = float(times[output_index])
        while current < target - 1e-12:
            step = min(maximum_step, target - current)
            position, velocity, spin = rk4_step(position, velocity, spin, step, params=params)
            current += step
        output[0][output_index] = position
        output[1][output_index] = velocity
        output[2][output_index] = spin
    return output


def sample_states_batch(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    initial_spins: np.ndarray,
    times_s: np.ndarray,
    *,
    substeps_per_second: float = 100.0,
    params: Mapping[str, float | str] = default_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched sibling of :func:`sample_states`, matching ``physics.flight``."""

    positions = np.asarray(initial_positions, dtype=float)
    velocities = np.asarray(initial_velocities, dtype=float)
    spins = np.asarray(initial_spins, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("batched flight positions must have shape (batch, 3)")
    if velocities.shape != positions.shape or spins.shape != positions.shape:
        raise ValueError("batched flight states must share shape")
    rows = [
        sample_states(
            position,
            velocity,
            spin,
            times_s,
            substeps_per_second=substeps_per_second,
            params=params,
        )
        for position, velocity, spin in zip(positions, velocities, spins, strict=True)
    ]
    if not rows:
        shape = (0, len(np.asarray(times_s)), 3)
        return tuple(np.empty(shape, dtype=float) for _ in range(3))
    return tuple(np.stack([row[index] for row in rows]) for index in range(3))


def RK4(
    dt: float,
    steps: int,
    x0: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    gravity: bool = True,
    drag: bool = True,
    lift: bool = True,
    aerodrag: bool = True,
    params: Mapping[str, float | str] = default_params,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``physics.flight.RK4`` and stop at the interpolated ``z=0`` crossing."""

    times = [0.0]
    positions = [_finite_vector(x0, "position").copy()]
    velocities = [_finite_vector(v0, "velocity").copy()]
    spins = [_finite_vector(w0, "spin").copy()]
    for index in range(int(steps)):
        next_position, next_velocity, next_spin = rk4_step(
            positions[-1],
            velocities[-1],
            spins[-1],
            dt,
            gravity=gravity,
            drag=drag,
            lift=lift,
            aerodrag=aerodrag,
            params=params,
        )
        next_time = (index + 1) * dt
        if next_position[2] < 0.0:
            fraction = positions[-1][2] / (positions[-1][2] - next_position[2])
            next_time = times[-1] + dt * fraction
            next_position = positions[-1] + fraction * (next_position - positions[-1])
            next_velocity = velocities[-1] + fraction * (next_velocity - velocities[-1])
            next_spin = spins[-1] + fraction * (next_spin - spins[-1])
            positions.append(next_position)
            velocities.append(next_velocity)
            spins.append(next_spin)
            times.append(next_time)
            if verbose:
                print(
                    f"Ball hits ground at x={next_position[0]:.3f} meters after {index + 1} steps."
                )
            break
        positions.append(next_position)
        velocities.append(next_velocity)
        spins.append(next_spin)
        times.append(next_time)
    return tuple(np.asarray(values) for values in (times, positions, velocities, spins))


@dataclass(frozen=True)
class BounceResult:
    velocity: np.ndarray
    spin: np.ndarray
    regime: str
    normal_cor: float
    friction: float


def court_normal_cor(theta1_deg: float, surface: str = "hard") -> float:
    """Angle-dependent Cross normal COR, scaled by the chart's bulk surface COR.

    Cross (2020) gives ``ey=.95-.005*theta`` for modern hard court.  The chart
    does not give normal COR, so its bulk COR is used only as a relative surface
    scale.  This assumption is intentionally visible rather than silently treating
    bulk speed COR as a normal restitution measurement.
    """

    key = surface.lower()
    if key not in SURFACES:
        raise ValueError(f"unknown court surface: {surface}")
    base = 0.95 - 0.005 * abs(float(theta1_deg))
    return float(np.clip(base * SURFACES[key]["cor"] / SURFACES["hard"]["cor"], 0.0, 0.95))


def court_bounce(
    velocity: np.ndarray,
    spin: np.ndarray,
    surface: str = "hard",
    *,
    e_x: float = 0.1,
    mu: float | None = None,
    normal_cor: float | None = None,
    deformation_offset_m: float | None = None,
    radius: float = R_BALL,
    alpha: float = ALPHA,
) -> BounceResult:
    """Apply Cross's sliding or gripping court-bounce equations in three axes."""

    incoming = _finite_vector(velocity, "velocity")
    omega = _finite_vector(spin, "spin")
    horizontal_speed = float(np.linalg.norm(incoming[:2]))
    downward_speed = -float(incoming[2])
    if horizontal_speed <= 1e-12 or downward_speed <= 0.0:
        raise ValueError("court impact requires horizontal travel and downward velocity")
    key = surface.lower()
    if key not in SURFACES:
        raise ValueError(f"unknown court surface: {surface}")

    forward = np.array([incoming[0], incoming[1], 0.0]) / horizontal_speed
    normal = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(normal, forward)
    theta = math.degrees(math.atan2(downward_speed, horizontal_speed))
    ey = court_normal_cor(theta, key) if normal_cor is None else float(normal_cor)
    friction = SURFACES[key]["cof"] if mu is None else float(mu)
    speed = math.hypot(horizontal_speed, downward_speed)
    offset = 3e-4 * speed if deformation_offset_m is None else float(deformation_offset_m)
    normal_impulse_per_mass = (1.0 + ey) * downward_speed

    # Contact-point slip: u = v_t + R(n x omega).
    tangential_velocity = np.array([incoming[0], incoming[1], 0.0])
    slip = tangential_velocity + radius * np.cross(normal, omega)
    slip_norm = float(np.linalg.norm(slip))
    slide_direction = slip / slip_norm if slip_norm > 1e-12 else np.zeros(3)
    slide_impulse = -friction * normal_impulse_per_mass * slide_direction

    def apply_impulse(tangent_delta_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        velocity_out = tangential_velocity + tangent_delta_v
        spin_out = omega - np.cross(normal, tangent_delta_v) / (alpha * radius)
        # Cross's forward offset D of the normal-force line reduces topspin.
        spin_out -= (normal_impulse_per_mass * offset / (alpha * radius**2)) * lateral
        contact_slip_out = velocity_out + radius * np.cross(normal, spin_out)
        return velocity_out, spin_out, contact_slip_out

    slide_velocity, slide_spin, slide_slip = apply_impulse(slide_impulse)
    slides_throughout = slip_norm > 1e-12 and float(np.dot(slide_slip, slide_direction)) > 0.0
    if slides_throughout:
        tangential_out, spin_out, regime = slide_velocity, slide_spin, "slide"
    else:
        # Eq. 12's additional D term acts only along the incoming horizontal path.
        grip_impulse = -alpha * (1.0 + e_x) / (1.0 + alpha) * slip
        grip_impulse -= normal_impulse_per_mass * offset / ((1.0 + alpha) * radius) * forward
        tangential_out, spin_out, _ = apply_impulse(grip_impulse)
        regime = "grip"
    velocity_out = tangential_out + ey * downward_speed * normal
    return BounceResult(velocity_out, spin_out, regime, ey, friction)


def rigid_body_energy(
    velocity: np.ndarray,
    spin: np.ndarray,
    *,
    mass: float = M_BALL,
    inertia: float = J_BALL,
) -> float:
    """Return translational plus rotational kinetic energy in joules."""

    return 0.5 * mass * float(np.dot(velocity, velocity)) + 0.5 * inertia * float(
        np.dot(spin, spin)
    )


@dataclass(frozen=True)
class RacketImpactResult:
    tangential_velocity: float
    normal_velocity: float
    spin: float


def racket_impact(
    tangential_velocity: float,
    normal_velocity: float,
    spin: float,
    *,
    radius: float = R_BALL,
    a: float = 0.648,
    b: float = 0.300,
    c: float = 0.400,
    d: float = 0.583,
    e_n: float = 0.420,
) -> RacketImpactResult:
    """Cross (2005) apparent racket-frame impact, equations 5--7."""

    return RacketImpactResult(
        a * tangential_velocity + b * radius * spin,
        e_n * normal_velocity,
        c * spin + d * tangential_velocity / radius,
    )


@dataclass(frozen=True)
class PlanarRacketResult:
    velocity: np.ndarray
    spin: float


def planar_racket_impact(
    ball_velocity: np.ndarray,
    racket_velocity: np.ndarray,
    outward_normal: np.ndarray,
    spin: float,
) -> PlanarRacketResult:
    """Apply the Cross racket law in a two-dimensional court-view plane."""

    ball = np.asarray(ball_velocity, dtype=float)
    racket = np.asarray(racket_velocity, dtype=float)
    normal = np.asarray(outward_normal, dtype=float)
    if ball.shape != (2,) or racket.shape != (2,) or normal.shape != (2,):
        raise ValueError("planar velocities and normal must be two-vectors")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("racket normal must be nonzero")
    normal = normal / normal_norm
    tangent = np.array([normal[1], -normal[0]])
    relative = ball - racket
    normal_in = -float(np.dot(relative, normal))
    if normal_in <= 0.0:
        raise ValueError("ball must approach the racket face")
    result = racket_impact(float(np.dot(relative, tangent)), normal_in, spin)
    relative_out = result.normal_velocity * normal + result.tangential_velocity * tangent
    return PlanarRacketResult(racket + relative_out, result.spin)


def racket_coefficients(tension_n: float, *, clamp_below_224_n: bool = False) -> dict[str, float]:
    """Return the variable-tension note's ``eN,eT,a,b,c,d`` coefficients."""

    tension = max(float(tension_n), 224.0) if clamp_below_224_n else float(tension_n)
    e_n = 0.43 - 0.03 / 56.0 * (tension - 224.0)
    e_t = 60.0 / 18667.0 * tension - 250133.0 / 373340.0
    alpha = ALPHA
    return {
        "e_n": e_n,
        "e_t": e_t,
        "a": (1.0 + alpha * e_t) / (1.0 + alpha) * 0.648 / 0.68419,
        "b": (1.0 - e_t) / (1.0 + alpha) * alpha * 0.300 / 0.31581,
        "c": (alpha + e_t) / (1.0 + alpha) * 0.400 / 0.42581,
        "d": (1.0 - e_t) / (1.0 + alpha) * 0.583 / 0.57419,
    }


def effective_tension(tension_n: float, normal_speed: float, n: float = 0.1) -> float:
    """Variable-tension note's quadratic speed correction, clamped to 10--40 m/s."""

    tension = float(tension_n)
    speed = float(np.clip(normal_speed, 10.0, 40.0))
    p = n * tension / 1080.0
    q = 11.0 * n * tension / 540.0
    r = (27.0 - 35.0 * n) * tension / 27.0
    return p * speed**2 + q * speed + r
