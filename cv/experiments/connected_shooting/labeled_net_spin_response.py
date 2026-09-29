"""Independently fitted outgoing velocity and world spin at the nominal net plane.

A net hit can alter spin. Forcing pre-net spin onto a long post-net two-bounce
trajectory may distort bounce timing and location. FreeNetState holds the same
continuous change of velocity as FreeNetVelocity plus an explicit outgoing
world spin at the observed plane crossing. It is a helper, not a yield claim.

It implements the existing response protocol so a parent simulator or fitter
can accept it for velocity, key and record. spin() returns the assigned
outgoing spin and allows every direction, zero and no-change. Position and
time continuity stay with the parent; this helper does not invent them.

There is no material law and the stored spin is fitted flexibility, not a
measurement. Broad component bounds are numerical (|v_i| <= 75 m/s,
|w_i| <= 1500 rad/s), not a direction or sign restriction. There is no
clamping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity

_MAX_SPIN_COMPONENT_RAD_S = 1500.0


def _finite3(value, label):
    try:
        values = np.asarray(value, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"finite {label} required") from exc
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"finite {label} required")
    return values


@dataclass(frozen=True)
class FreeNetState(FreeNetVelocity):
    """Assigned outgoing velocity and world spin at the observed net-plane crossing."""

    outgoing_spin_rad_s: tuple[float, float, float]

    def __post_init__(self):
        super().__post_init__()
        values = _finite3(self.outgoing_spin_rad_s, "outgoing spin")
        if np.any(np.abs(values) > _MAX_SPIN_COMPONENT_RAD_S):
            raise ValueError("finite outgoing spin required")
        object.__setattr__(
            self,
            "outgoing_spin_rad_s",
            (float(values[0]), float(values[1]), float(values[2])),
        )

    @property
    def key(self):
        return ("free_net_state_v1",) + self.outgoing_velocity_mps + self.outgoing_spin_rad_s

    def spin(self, incoming_spin, incoming_velocity, outgoing_velocity):
        _finite3(incoming_spin, "incoming spin")
        _finite3(incoming_velocity, "incoming velocity")
        _finite3(outgoing_velocity, "outgoing velocity")
        return np.array(self.outgoing_spin_rad_s, float)

    def record(self, incoming, outgoing):
        receipt = super().record(incoming, outgoing)
        receipt.update(
            model="experimental_free_net_state_v1",
            spin_unchanged=False,
            outgoing_spin_rad_s=list(self.outgoing_spin_rad_s),
            contact_geometry=(
                "independently fitted outgoing velocity and spin at nominal net plane; "
                "fitted flexibility, not measured spin or a material law"
            ),
        )
        return receipt
