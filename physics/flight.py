import math
from dataclasses import dataclass

import numpy as np


# The S6 stack uses this measured ball set everywhere: 57 g, 32.5 mm radius,
# and 3.155e-5 kg m^2 inertia.  Keep impact.py and rich_ball_physics.py aligned.
default_params = {
    "g": 9.81,
    "m_ball": 0.057,
    "R_ball": 0.0325,
    "J_ball": 3.155e-5,
    "rho_air": 1.205,
    "C_drag": 0.55,
    "C_lift": 0.6,
    "C_spin_decay": 0.025,
}


@dataclass(frozen=True)
class GroundImpactState:
    """Incoming state at a solved ground crossing, before any impact response."""

    time_seconds: float
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray


def ground_impact_in_interval(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    lower_seconds: float,
    upper_seconds: float,
    *,
    plane_z_m: float,
    params: dict = default_params,
) -> GroundImpactState | None:
    """Solve a descending free-flight crossing inside an explicit observation interval.

    This never invents a witness, widens the interval, snaps a position to the
    plane, or models the outgoing rebound. A missing descending bracket abstains.
    Callers must supply a bounce-free incoming arc and own the interval's evidence.
    Times use exactly the forward sampler's origin and integration convention.
    """
    from scipy.optimize import brentq

    initial = tuple(np.asarray(value, float) for value in (position, velocity, spin))
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in initial):
        raise ValueError("ground-crossing state must contain finite three-vectors")
    lower, upper, plane = float(lower_seconds), float(upper_seconds), float(plane_z_m)
    if not all(math.isfinite(value) for value in (lower, upper, plane)) or not 0 <= lower < upper:
        raise ValueError("ground-crossing interval must be finite, increasing and nonnegative")

    def state(time_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = tuple(value[0] for value in sample_states(*initial, [time_seconds], params=params))
        if any(not np.all(np.isfinite(value)) for value in values):
            raise ValueError("nonfinite ground-crossing propagation")
        return values

    first, last = state(lower), state(upper)
    first_height, last_height = float(first[0][2] - plane), float(last[0][2] - plane)
    if first_height < 0.0 or last_height > 0.0:
        return None
    # A just-launched rebound on the plane is not the later descending impact.
    if first_height == 0.0 and float(first[1][2]) >= 0.0:
        return None
    root = float(brentq(lambda time: float(state(time)[0][2] - plane), lower, upper, xtol=1e-10))
    impact_position, impact_velocity, impact_spin = state(root)
    if float(impact_velocity[2]) >= 0.0 or abs(float(impact_position[2] - plane)) > 1e-7:
        return None
    return GroundImpactState(root, impact_position, impact_velocity, impact_spin)


def sample_signed_states(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample signed elapsed seconds on a shared 0.01 s RK4 grid.

    Targets share whole integration steps but retain their own final partial
    step. This is the same clock used by the measured-impact fitting objective
    and exported trajectory states; query order does not change the trajectory.
    """
    query = np.asarray(times, float)
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("signed sample times must be a finite one-dimensional array")
    count = len(query)
    positions = np.empty((count, 3), float)
    velocities = np.empty((count, 3), float)
    spins = np.empty((count, 3), float)
    anchor = tuple(np.asarray(value, float).copy() for value in (position, velocity, spin))
    step_seconds = 0.01
    for direction in (1.0, -1.0):
        selected = sorted(
            (
                index
                for index in range(count)
                if abs(query[index]) > 1e-12 and math.copysign(1.0, query[index]) == direction
            ),
            key=lambda index: abs(query[index]),
        )
        if not selected:
            continue
        grid = [anchor]
        whole_step = math.copysign(step_seconds, direction)
        for index in selected:
            remaining, whole, final_step = float(query[index]), 0, 0.0
            # Preserve the original arithmetic, including floating-point step edges.
            while abs(remaining) > 1e-12:
                magnitude = min(step_seconds, abs(remaining))
                step = math.copysign(magnitude, remaining)
                if magnitude < step_seconds:
                    final_step = step
                    break
                whole += 1
                remaining -= step
            while len(grid) <= whole:
                grid.append(rk4_step(*grid[-1], whole_step))
            state = grid[whole]
            if final_step:
                state = rk4_step(*state, final_step)
            positions[index], velocities[index], spins[index] = state
    for index in range(count):
        if abs(query[index]) <= 1e-12:
            positions[index], velocities[index], spins[index] = anchor
    return positions, velocities, spins


def sample_states(x0, v0, w0, times_s, *, substeps_per_second=100.0, params=default_params):
    """Sample a drag-plus-Magnus free flight at nonnegative elapsed times.

    This small forward sampler is intentionally bounce-free.  Callers that own an
    observed court impact can split the arc at its exact plane anchor and apply the
    surface impact law there, rather than letting an integrator invent impact timing.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or np.any(times < -1e-12):
        raise ValueError("flight sample times must be a one-dimensional nonnegative array")
    if not len(times):
        empty = np.empty((0, 3), dtype=float)
        return empty, empty.copy(), empty.copy()
    order = np.argsort(times)
    sorted_times = times[order]
    positions = np.empty((len(times), 3), dtype=float)
    velocities = np.empty((len(times), 3), dtype=float)
    spins = np.empty((len(times), 3), dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    w = np.asarray(w0, dtype=float).copy()
    current = 0.0
    maximum_step = 1.0 / float(substeps_per_second)
    for output_index, target in zip(order, sorted_times):
        while current < target - 1e-12:
            step = min(maximum_step, float(target - current))
            x, v, w = rk4_step(x, v, w, step, params=params)
            current += step
        positions[output_index] = x
        velocities[output_index] = v
        spins[output_index] = w
    return positions, velocities, spins


def _batch_derivatives(velocities, spins, params):
    speeds = np.linalg.norm(velocities, axis=1)
    k = 0.5 * params["rho_air"] * np.pi * params["R_ball"] ** 2 / params["m_ball"]
    accelerations = np.zeros_like(velocities)
    accelerations[:, 2] = -params["g"]
    # Match accel exactly: these batches supply finite-difference perturbations
    # of a scalar baseline. A different low-spin cutoff corrupts that Jacobian.
    moving = speeds > 1e-12
    accelerations += (-k * params["C_drag"] * speeds * moving)[:, None] * velocities
    lift_active = moving & (np.linalg.norm(spins, axis=1) > 1e-12)
    accelerations += (
        k * params["C_lift"] * params["R_ball"] * lift_active[:, None] * np.cross(spins, velocities)
    )
    spin_k = 0.5 * params["rho_air"] * np.pi * params["R_ball"] ** 4 / params["J_ball"]
    spin_rates = (-params.get("C_spin_decay", 0.025) * spin_k * speeds)[:, None] * spins
    return accelerations, spin_rates


def _rk4_step_batch(positions, velocities, spins, dt, params):
    v1 = velocities
    a1, w1 = _batch_derivatives(velocities, spins, params)
    v_mid1 = velocities + a1 * dt / 2.0
    w_mid1 = spins + w1 * dt / 2.0
    a2, w2 = _batch_derivatives(v_mid1, w_mid1, params)
    v_mid2 = velocities + a2 * dt / 2.0
    w_mid2 = spins + w2 * dt / 2.0
    a3, w3 = _batch_derivatives(v_mid2, w_mid2, params)
    v_end = velocities + a3 * dt
    w_end = spins + w3 * dt
    a4, w4 = _batch_derivatives(v_end, w_end, params)
    return (
        positions + dt * (v1 + 2.0 * v_mid1 + 2.0 * v_mid2 + v_end) / 6.0,
        velocities + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0,
        spins + dt * (w1 + 2.0 * w2 + 2.0 * w3 + w4) / 6.0,
    )


def sample_states_batch(
    initial_positions,
    initial_velocities,
    initial_spins,
    times_s,
    *,
    substeps_per_second=100.0,
    params=default_params,
):
    """Vectorized sibling of :func:`sample_states` for finite-difference columns."""
    times = np.asarray(times_s, dtype=float)
    positions0 = np.asarray(initial_positions, dtype=float)
    velocities0 = np.asarray(initial_velocities, dtype=float)
    spins0 = np.asarray(initial_spins, dtype=float)
    if positions0.ndim != 2 or positions0.shape[1] != 3:
        raise ValueError("batched flight positions must have shape (batch, 3)")
    if velocities0.shape != positions0.shape or spins0.shape != positions0.shape:
        raise ValueError("batched flight states must share shape")
    if times.ndim != 1 or np.any(times < -1e-12):
        raise ValueError("flight sample times must be a one-dimensional nonnegative array")
    batch = len(positions0)
    output_positions = np.empty((batch, len(times), 3), dtype=float)
    output_velocities = np.empty_like(output_positions)
    output_spins = np.empty_like(output_positions)
    if not len(times):
        return output_positions, output_velocities, output_spins
    order = np.argsort(times)
    x = positions0.copy()
    v = velocities0.copy()
    w = spins0.copy()
    current = 0.0
    maximum_step = 1.0 / float(substeps_per_second)
    for output_index, target in zip(order, times[order]):
        while current < target - 1e-12:
            step = min(maximum_step, float(target - current))
            x, v, w = _rk4_step_batch(x, v, w, step, params)
            current += step
        output_positions[:, output_index] = x
        output_velocities[:, output_index] = v
        output_spins[:, output_index] = w
    return output_positions, output_velocities, output_spins


def accel(v, w, gravity=True, drag=True, lift=True, params=default_params):
    """
    Compute the acceleration vector as a function of the velocity and spin vectors

    Parameters
    ----------
    v : numpy array
        velocity vector of the ball (in m/s)
    w : numpy array
        spin vector of the ball (in rad/s)
    gravity : bool (optional, default: True)
        Include gravitational force
    drag : bool (optional, default: True)
        Include drag force
    lift: bool (optionl, default: True)
        Include lift force
    parms: dict (optionl, default: default_params dict)
        dict of constants and parameters

    Returns
    -------
    a : numpy array
        acceleration vector of the ball (in m/s^2)
    """

    # This is on the finite-difference hot path.  Avoid np.cross and
    # np.linalg.norm dispatches for tiny fixed-size vectors.
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    ax = 0.0
    ay = 0.0
    az = -params["g"] if gravity else 0.0

    # Compute drag and lift constant
    k = 0.5 * params["rho_air"] * np.pi * params["R_ball"] ** 2 / params["m_ball"]

    # Compute drag acceleration vector
    if drag and vmag > 1e-12:
        drag_scale = -k * params["C_drag"] * vmag
        ax += drag_scale * vx
        ay += drag_scale * vy
        az += drag_scale * vz

    # Compute lift acceleration vector
    if lift and vmag > 1e-12 and wmag > 1e-12:
        lift_scale = k * params["C_lift"] * params["R_ball"]
        ax += lift_scale * (wy * vz - wz * vy)
        ay += lift_scale * (wz * vx - wx * vz)
        az += lift_scale * (wx * vy - wy * vx)

    return np.array((ax, ay, az))


def w_dot(v, w, params=default_params):
    """
    Compute the rate of change of the ball's spin

    Parameters
    ----------
    v : numpy array
        velocity vector of the ball (in m/s)
    w : numpy array
        spin vector of the ball (in rad/s)
    parms: dict (optionl, default: default_params dict)
        dict of constants and parameters

    Returns
    -------
    wdot : numpy array
        acceleration vector of the ball (in m/s^2)
    """

    # Skin-friction torque: proportional to airflow speed AND to the spin itself, so
    # spin decays roughly exponentially with a long time constant. The previous form
    # (-k*C*v^2 * unit(w)) decayed at a rate independent of |w|, draining any spin
    # linearly to zero in ~0.5 s of flight (real balls retain ~95% to the bounce; the
    # Hawkeye v2 calibration measured the old model at 0.09% retention), and divided by
    # zero at |w| = 0. C_spin_decay = 0.025 gives ~95% retention over a 1 s, 30 m/s
    # flight.
    vmag = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)
    k4 = 0.5 * params["rho_air"] * np.pi * params["R_ball"] ** 4 / params["J_ball"]
    decay = params.get("C_spin_decay", 0.025)
    wdot = -decay * k4 * vmag * w
    return wdot


def rk4_step(
    x,
    v,
    w,
    dt,
    gravity=True,
    drag=True,
    lift=True,
    aerodrag=True,
    params=default_params,
):
    """Advance coupled position, velocity, and spin by one RK4 step."""

    def derivatives(velocity, spin):
        acceleration = accel(
            velocity,
            spin,
            gravity=gravity,
            drag=drag,
            lift=lift,
            params=params,
        )
        spin_rate = w_dot(velocity, spin, params) if aerodrag else np.zeros(3)
        return acceleration, spin_rate

    x1 = v
    v1, w1 = derivatives(v, w)

    v_mid1 = v + v1 * dt / 2
    w_mid1 = w + w1 * dt / 2
    x2 = v_mid1
    v2, w2 = derivatives(v_mid1, w_mid1)

    v_mid2 = v + v2 * dt / 2
    w_mid2 = w + w2 * dt / 2
    x3 = v_mid2
    v3, w3 = derivatives(v_mid2, w_mid2)

    v_end = v + v3 * dt
    w_end = w + w3 * dt
    x4 = v_end
    v4, w4 = derivatives(v_end, w_end)
    return (
        x + dt * (x1 + 2 * x2 + 2 * x3 + x4) / 6,
        v + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6,
        w + dt * (w1 + 2 * w2 + 2 * w3 + w4) / 6,
    )


def accel_cross(v, w, params=default_params):
    # a simpler version of the function above (works only in 2D)
    vmag = np.linalg.norm(v)
    k = 0.5 * params["rho_air"] * np.pi * params["R_ball"] ** 2 / params["m_ball"]
    S = w[1] * params["R_ball"] / vmag
    C_L = params["C_lift"] * S
    C_D = params["C_drag"]
    vxdot = -k * vmag * (C_D * v[0] - C_L * v[2])
    vzdot = -params["g"] - k * vmag * (C_L * v[0] + C_D * v[2])

    a = np.array([vxdot, 0.0, vzdot])
    return a


def RK4(
    dt,
    steps,
    x0,
    v0,
    w0,
    gravity=True,
    drag=True,
    lift=True,
    aerodrag=True,
    params=default_params,
    verbose=True,
):
    """
    Compute the flightpath of a tennis ball using a 4th order Runge-Kutta routine, given the ball's
    initial conditions (3D position, velocity, and spin), until it reaches the ground

    Parameters
    ----------
    dt : numeric
        length of timestep (in seconds)
    steps : int
        maximum number of timesteps
    x0 : numpy array
        initial 3D position of the ball (in m)
    v0 : numpy array
        initial velocity vector of the ball (in m/s)
    w0 : numpy array
        initial spin vector of the ball (in rad/s)
    gravity : bool (optional, default: True)
        Include gravitational force
    drag : bool (optional, default: True)
        Include drag force
    lift: bool (optionl, default: True)
        Include lift force
    aerodrag : bool (optional, default: True)
        Include aerodrag force (induces change in spin)
    parms: dict (optionl, default: default_params dict)
        dict of constants and parameters
    verbose : bool (optional, default: True)
        Print progress and output messages

    Returns
    -------
    t_array : numpy array
        array designating the time at each step
    x_array : numpy array
        ball position at each timestep
    v_array : numpy array
        ball velocity at each timestep
    w_array : numpy array
        ball spin at each timestep
    """

    # Initialize arrays for storing time, position, velocity, and spin at each step
    t_array = np.zeros(steps + 1)
    x_array = np.zeros((steps + 1, 3))
    v_array = np.zeros((steps + 1, 3))
    w_array = np.zeros((steps + 1, 3))

    # Set first element of each array to provided initial conditions
    x_array[0] = x0
    v_array[0] = v0
    w_array[0] = w0

    if not aerodrag:
        w_array[:] = w0

        # Set up RK4 iterations
        for i in range(steps):
            # Calculate RK4 parameters
            k1 = v_array[i]
            vk1 = accel(k1, w0, gravity, drag, lift, params)

            k2 = v_array[i] + vk1 * dt / 2
            vk2 = accel(k2, w0, gravity, drag, lift, params)

            k3 = v_array[i] + vk2 * dt / 2
            vk3 = accel(k3, w0, gravity, drag, lift, params)

            k4 = v_array[i] + vk3 * dt
            vk4 = accel(k4, w0, gravity, drag, lift, params)

            # Update arrays with the next timstep
            t_array[i + 1] = (i + 1) * dt
            x_array[i + 1] = x_array[i] + 1.0 / 6.0 * dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            v_array[i + 1] = v_array[i] + 1.0 / 6.0 * dt * (vk1 + 2.0 * vk2 + 2.0 * vk3 + vk4)

            # Evaluate position where ball hits the ground
            if x_array[i + 1, 2] < 0:
                imax = i + 1
                zrat = x_array[i, 2] / (x_array[i, 2] - x_array[i + 1, 2])
                t_array[i + 1] = t_array[i] + dt * zrat
                x_array[i + 1] = x_array[i] + (x_array[i + 1] - x_array[i]) * zrat
                v_array[i + 1] = v_array[i] + (v_array[i + 1] - v_array[i]) * zrat
                if verbose:
                    print(
                        "Ball hits ground at x="
                        + str(round(x_array[i + 1, 0], 3))
                        + " meters after "
                        + str(imax)
                        + " steps."
                    )
                break
        else:
            imax = i + 1

    else:
        # Set up RK4 iterations
        for i in range(steps):
            # Calculate RK4 parameters
            k1 = v_array[i]
            vk1 = accel(k1, w_array[i], gravity, drag, lift, params)
            wk1 = w_dot(k1, w_array[i], params)

            k2 = v_array[i] + vk1 * dt / 2
            vk2 = accel(k2, w_array[i] + wk1 * dt / 2, gravity, drag, lift, params)
            wk2 = w_dot(k2, w_array[i] + wk1 * dt / 2, params)

            k3 = v_array[i] + vk2 * dt / 2
            vk3 = accel(k3, w_array[i] + wk2 * dt / 2, gravity, drag, lift, params)
            wk3 = w_dot(k3, w_array[i] + wk2 * dt / 2, params)

            k4 = v_array[i] + vk3 * dt
            vk4 = accel(k4, w_array[i] + wk3 * dt, gravity, drag, lift, params)
            wk4 = w_dot(k4, w_array[i] + wk3 * dt, params)

            # Update arrays with the next timstep
            t_array[i + 1] = (i + 1) * dt
            x_array[i + 1] = x_array[i] + 1.0 / 6.0 * dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            v_array[i + 1] = v_array[i] + 1.0 / 6.0 * dt * (vk1 + 2.0 * vk2 + 2.0 * vk3 + vk4)
            w_array[i + 1] = w_array[i] + 1.0 / 6.0 * dt * (wk1 + 2.0 * wk2 + 2.0 * wk3 + wk4)

            # Evaluate position where ball hits the ground
            if x_array[i + 1, 2] < 0:
                imax = i + 1
                zrat = x_array[i, 2] / (x_array[i, 2] - x_array[i + 1, 2])
                t_array[i + 1] = t_array[i] + dt * zrat
                x_array[i + 1] = x_array[i] + (x_array[i + 1] - x_array[i]) * zrat
                v_array[i + 1] = v_array[i] + (v_array[i + 1] - v_array[i]) * zrat
                w_array[i + 1] = w_array[i] + (w_array[i + 1] - w_array[i]) * zrat
                if verbose:
                    print(
                        "Ball hits ground at x="
                        + str(round(x_array[i + 1, 0], 3))
                        + " meters after "
                        + str(imax)
                        + " steps."
                    )
                break
            else:
                imax = i + 1

    return t_array[: imax + 1], x_array[: imax + 1], v_array[: imax + 1], w_array[: imax + 1]
