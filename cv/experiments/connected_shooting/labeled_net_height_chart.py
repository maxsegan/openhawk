"""Optional native-reviewed upper-net height coordinate for labeled lifting.

The caller must explicitly justify the permitted height offsets from native
event evidence. This is an incoming continuous-state constraint, not a measured
cable law. No net position or physical state is snapped after propagation.
"""

from __future__ import annotations

import numpy as np

from cv.experiments.connected_shooting.labeled_net_epoch_chart import project_net_velocity
from cv.pipeline.physics_knot_solver import net_tape_height
from cv.experiments.connected_shooting.measured_dynamics import R_BALL


def project_net_height(
    theta,
    start_frame,
    net_frame,
    fps,
    surface,
    height_offset_m=None,
    *,
    mesh_fraction=None,
    **kwargs,
):
    """Solve Vy and Vz so the incoming flight reaches plane and relative tape height."""
    initial = np.asarray(theta, float)
    if mesh_fraction is None:
        if height_offset_m is None or not np.isfinite(height_offset_m):
            raise ValueError("finite explicit net-height offset required")
    elif (
        height_offset_m is not None or not np.isfinite(mesh_fraction) or not 0 < mesh_fraction <= 1
    ):
        raise ValueError("mesh fraction must be in (0, 1] without a tape-offset override")
    if initial.shape != (9,) or not np.isfinite(initial).all():
        raise ValueError("finite incoming theta9 required")

    def desired_height(x):
        tape = net_tape_height(float(x))
        return (
            tape + float(height_offset_m)
            if mesh_fraction is None
            else R_BALL + float(mesh_fraction) * tape
        )

    calls = 0
    last = None

    def evaluate(vz):
        nonlocal calls, last
        trial = initial.copy()
        trial[5] = vz
        result, receipt = project_net_velocity(
            trial, start_frame, net_frame, fps, surface, **kwargs
        )
        calls += receipt["simulation_calls"]
        xyz = receipt["net_xyz_m"]
        desired = desired_height(xyz[0])
        last = result, receipt, desired
        return float(xyz[2] - desired)

    # Seed the root from its requested airborne endpoint. The supplied Vz can
    # hit the ground before the net even when the requested height is feasible.
    # Rejecting that intermediate seed would create a false support boundary.
    seconds = (net_frame - start_frame) / fps
    if seconds <= 0:
        raise ValueError("positive incoming net duration required")
    approximate_x = initial[0] + seconds * initial[3]
    approximate_height = desired_height(approximate_x)
    vz = float((approximate_height - initial[2]) / seconds + 0.5 * 9.81 * seconds)
    initial_height_seed = vz
    seed_failures = []
    for lift in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        vz = initial_height_seed + lift
        if not -75 <= vz <= 75:
            continue
        try:
            error = evaluate(vz)
            break
        except ValueError as exc:
            if str(exc) != "first-net chart has an earlier ground impact":
                raise
            seed_failures.append({"vz_mps": vz, "reason": str(exc)})
    else:
        raise ValueError("upper-net initialization has no supported airborne seed")
    supported_height_seed = vz
    domain_failures = []

    def supported_step(origin, step, stage):
        # Preserve the last airborne iterate. A Newton/derivative trial can
        # leave this branch even though its requested endpoint is reachable.
        for _ in range(24):
            trial = origin + step
            if trial == origin:
                break
            try:
                return trial, evaluate(trial)
            except ValueError as exc:
                if str(exc) != "first-net chart has an earlier ground impact":
                    raise
                domain_failures.append({"stage": stage, "vz_mps": trial})
                step *= 0.5
        raise ValueError("upper-net height step has no supported airborne trial")

    for _ in range(12):
        if abs(error) < 1e-8:
            break
        probe, probe_error = supported_step(vz, 1e-3, "derivative")
        derivative = (probe_error - error) / (probe - vz)
        if not np.isfinite(derivative) or abs(derivative) < 1e-8:
            raise ValueError("upper-net height coordinate is locally unsupported")
        proposed = vz - error / derivative
        if not -75 <= proposed <= 75:
            raise ValueError("upper-net height requires unsupported incoming Vz")
        vz, error = supported_step(vz, proposed - vz, "newton")
    error = evaluate(vz)
    result, receipt, desired = last
    if abs(error) > 1e-6:
        raise ValueError("upper-net height coordinate did not converge")
    return result, receipt | {
        "height_offset_from_tape_m": float(height_offset_m)
        if mesh_fraction is None
        else desired - net_tape_height(float(receipt["net_xyz_m"][0])),
        "required_net_height_m": desired,
        "height_error_m": abs(error),
        "projected_vz_mps": float(result[5]),
        "initial_airborne_vz_seed_mps": initial_height_seed,
        "supported_airborne_vz_seed_mps": supported_height_seed,
        "initial_seed_failures": seed_failures,
        "height_chart_simulation_calls": calls,
        "airborne_domain_backtracks": domain_failures,
        "height_support_is_explicit_reviewed_input": mesh_fraction is None,
        **(
            {"mesh_height_fraction": float(mesh_fraction), "height_support": "full_physical_mesh"}
            if mesh_fraction is not None
            else {}
        ),
    }
