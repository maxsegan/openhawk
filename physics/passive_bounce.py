"""Passive coupled ground rebound: a rigid-sphere Coulomb point-impulse approximation.

This is deliberately NOT the calibrated Cross 2020 deformation law used by
``physics.impact.court_bounce`` / ``court_bounce_vector``. Those carry the measured
angle-dependent normal restitution ``e_y`` and the normal-force offset ``D`` that brakes the
forward velocity and torques topspin. Here the normal restitution is whatever the caller
supplies -- deciding where that number comes from is the caller's job -- and the tangential
response is a single isotropic Coulomb impulse acting on the full planar contact slip of an
ideal point contact. The result is a passive, energy-dissipating rebound with no fitted
parameter of its own: the only chart consulted is the existing per-surface COF in
``impact.SURFACES``.

Model, per unit mass, with the court normal along +z and the ball descending (``v_z < 0``):

  normal impulse    Jn = (1 + e) * (-v_z)          outgoing   v_z' = -e * v_z
  contact slip      s  = v[:2] + R * (-w_y, w_x)   (velocity of the ball's bottom point)
  tangential        j  = -min(mu * Jn,  a/(1+a) * (1 + e_t) * |s|) * s / |s|    (j = 0 if s = 0)
  updates           v[:2] += j      w[:2] += (j_y, -j_x) / (a * R)      w_z unchanged

where ``a = ALPHA = J/(m R^2)``. The second term of the friction cap is the impulse that would
leave the contact point with a reversed slip ``-e_t * s`` (sticking / tangential restitution);
taking the smaller of the two is the usual sliding-versus-gripping switch, written as one
continuous formula rather than two regimes. Both branches stay at or below ``2a/(1+a) * |s|``
for ``e_t <= 1``, which is exactly the point where the tangential channel would start injecting
energy, and ``e in [0, 1]`` makes the normal channel non-increasing, so total kinetic energy is
always conserved or dissipated.

Nothing is clipped, floored or forced on top of that. There is no post-impulse horizontal
rescaling, no energy clipping, no minimum speed, no incidence threshold and no forced rolling.
A rebound with zero horizontal speed, or one that is purely vertical, is a legitimate output.

Every call returns a receipt naming the model, both slip vectors, the impulse actually applied,
the coefficients used and the realized energy ratios, together with an explicit list of the
physics this approximation asserts without measuring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from physics import impact

MODEL_ID = "passive_coupled_bounce_v1"

# Removing slip beyond a/(1+a)*2*|s| would add tangential energy; the gripping branch of the
# friction cap is a/(1+a)*(1+e_t)*|s|, which stays at or below that bound for e_t <= 1.
_GRIP_COEFFICIENT = impact.ALPHA / (1.0 + impact.ALPHA)

# What this approximation asserts without measuring it. Carried in every receipt so a consumer
# never mistakes a coupled_rebound state for a calibrated bounce.
APPROXIMATIONS = (
    "ideal point contact: no contact patch, no normal-force offset D, no deformation torque",
    "normal restitution is caller-supplied; this helper does not independently model its dependencies",
    "tangential restitution is a nominal constant, not fitted to this impact",
    "friction is the existing per-surface COF chart in impact.SURFACES, constant through contact",
    "spin about the court normal is untouched: no drilling friction at a point contact",
    "rigid sphere: contact duration, ball compression and surface compliance are not modelled",
    "not calibrated against the Cross 2020 court-bounce measurements",
)


@dataclass(frozen=True)
class CoupledRebound:
    """Outgoing state of one passive coupled rebound, with its provenance receipt."""

    velocity: np.ndarray
    spin: np.ndarray
    receipt: dict[str, Any]


def _validated_vector(value: Any, name: str) -> np.ndarray:
    """Copy ``value`` into a fresh finite float three-vector, so callers' arrays never mutate."""
    vector = np.array(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a three-vector, got shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def coupled_rebound(
    velocity: np.ndarray,
    spin: np.ndarray,
    *,
    restitution: float,
    surface: str,
    tangential_restitution: float = impact.E_X_COURT,
) -> CoupledRebound:
    """Rebound a descending ball off the court under a Coulomb point impulse.

    ``velocity`` (m/s) and ``spin`` (rad/s) are court-frame three-vectors; the court plane is
    z = 0 and the ball must be descending (``velocity[2] < 0``). ``restitution`` is the normal
    coefficient the caller wants preserved: the outgoing vertical velocity is exactly
    ``-restitution * velocity[2]``, whatever the tangential channel does. ``surface`` selects
    the friction coefficient from the existing ``impact.SURFACES`` chart and nothing else --
    the surface does not set the normal restitution here. ``tangential_restitution`` is the
    slip-reversal coefficient of the gripping branch.

    This is an approximation, not the calibrated Cross model; see the module docstring.
    """
    velocity_in = _validated_vector(velocity, "velocity")
    spin_in = _validated_vector(spin, "spin")

    restitution = float(restitution)
    tangential_restitution = float(tangential_restitution)
    if not np.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be a finite coefficient in [0, 1]")
    if not np.isfinite(tangential_restitution) or not 0.0 <= tangential_restitution <= 1.0:
        raise ValueError("tangential_restitution must be a finite coefficient in [0, 1]")
    if surface not in impact.SURFACES:
        raise ValueError(f"surface must be one of {sorted(impact.SURFACES)}, got {surface!r}")
    descending_speed = -float(velocity_in[2])
    if not descending_speed > 0.0:
        raise ValueError("court rebound requires a descending ball (velocity[2] < 0)")

    mu = impact.SURFACES[surface][1]
    radius = impact.R_BALL

    # Normal channel: the caller's restitution, preserved exactly.
    normal_impulse_per_mass = (1.0 + restitution) * descending_speed

    # Tangential channel: one isotropic Coulomb impulse opposing the bottom-point slip.
    slip_in = velocity_in[:2] + radius * np.array([-spin_in[1], spin_in[0]])
    slip_speed_in = float(np.linalg.norm(slip_in))
    grip_scale = _GRIP_COEFFICIENT * (1.0 + tangential_restitution)
    if slip_speed_in > 0.0:
        # Scale form of min(mu*Jn, grip_scale*|s|) * (-s/|s|): identical arithmetic, but it
        # never divides by a vanishing slip speed, so the limit s -> 0 stays continuous.
        friction_scale = mu * normal_impulse_per_mass / slip_speed_in
        scale = min(friction_scale, grip_scale)
        limited_by = "coulomb_friction" if friction_scale < grip_scale else "tangential_grip"
    else:
        scale = 0.0
        limited_by = "no_slip"
    impulse = -scale * slip_in

    velocity_out = np.empty(3)
    velocity_out[:2] = velocity_in[:2] + impulse
    velocity_out[2] = -restitution * velocity_in[2]

    spin_out = np.empty(3)
    spin_out[:2] = spin_in[:2] + np.array([impulse[1], -impulse[0]]) / (impact.ALPHA * radius)
    spin_out[2] = spin_in[2]

    slip_out = velocity_out[:2] + radius * np.array([-spin_out[1], spin_out[0]])

    def kinetic_energy(v: np.ndarray, w: np.ndarray) -> float:
        return 0.5 * impact.M_BALL * float(v @ v) + 0.5 * impact.J_BALL * float(w @ w)

    # A descending ball always carries kinetic energy, so this ratio is always defined.
    energy_in = kinetic_energy(velocity_in, spin_in)
    energy_out = kinetic_energy(velocity_out, spin_out)

    horizontal_speed_in = float(np.linalg.norm(velocity_in[:2]))
    horizontal_speed_out = float(np.linalg.norm(velocity_out[:2]))
    horizontal_retained = (
        horizontal_speed_out / horizontal_speed_in if horizontal_speed_in > 0.0 else None
    )

    receipt: dict[str, Any] = {
        "model": MODEL_ID,
        "model_family": "dissipative rigid-sphere Coulomb point impulse",
        "calibration": "uncalibrated approximation; NOT the Cross 2020 deformation law",
        "approximations": list(APPROXIMATIONS),
        "surface": surface,
        "friction_coefficient": mu,
        "friction_coefficient_source": "impact.SURFACES chart COF (unchanged, not refitted)",
        "normal_restitution": restitution,
        "normal_restitution_source": "caller-supplied; preserved exactly in the outgoing z",
        "tangential_restitution": tangential_restitution,
        "ball_constants": {
            "mass_kg": impact.M_BALL,
            "radius_m": impact.R_BALL,
            "inertia_kg_m2": impact.J_BALL,
            "alpha": impact.ALPHA,
        },
        "slip_in_mps": slip_in.tolist(),
        "slip_out_mps": slip_out.tolist(),
        "slip_speed_in_mps": slip_speed_in,
        "slip_speed_out_mps": float(np.linalg.norm(slip_out)),
        "normal_impulse_per_mass_mps": normal_impulse_per_mass,
        "tangential_impulse_per_mass_mps": impulse.tolist(),
        "tangential_impulse_magnitude_mps": float(np.linalg.norm(impulse)),
        "friction_budget_per_mass_mps": mu * normal_impulse_per_mass,
        "grip_budget_per_mass_mps": grip_scale * slip_speed_in,
        "tangential_impulse_limited_by": limited_by,
        "kinetic_energy_in_j": energy_in,
        "kinetic_energy_out_j": energy_out,
        "kinetic_energy_ratio": energy_out / energy_in,
        "horizontal_speed_in_mps": horizontal_speed_in,
        "horizontal_speed_out_mps": horizontal_speed_out,
        "horizontal_speed_retained_ratio": horizontal_retained,
    }
    return CoupledRebound(velocity=velocity_out, spin=spin_out, receipt=receipt)
