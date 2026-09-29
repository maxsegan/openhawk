"""Independently fitted outgoing velocity at the repository's nominal net plane.

Net hits have many behaviors and are not feasibly narrowed to one material law.
FreeNetVelocity holds a continuous change of velocity at the observed plane
crossing, inferred from pixels. Material laws can initialize or control a fit.
It implements the existing response protocol so labeled_passive_tape.simulator
and using_response accept it unchanged. Position and spin stay continuous.

There is no hard passivity requirement. A parent fitter may use a weak excess-
speed prior; Astra reviews physical plausibility. Broad component bounds are
numerical (|v_i| <= 75 m/s), not a direction or sign restriction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv.experiments.connected_shooting.labeled_net_signed_response import SignedTapeResponse
from cv.experiments.connected_shooting.labeled_passive_tape import TapeResponse

_MAX_COMPONENT_MPS = 75.0


def _finite3(value, label):
    try:
        values = np.asarray(value, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"finite {label} velocity required") from exc
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"finite {label} velocity required")
    return values


def _translational_energy_ratio(incoming, outgoing):
    incoming_ke = float(incoming @ incoming)
    outgoing_ke = float(outgoing @ outgoing)
    if incoming_ke > 0.0:
        return outgoing_ke / incoming_ke
    return 1.0 if outgoing_ke == 0.0 else float("inf")


@dataclass(frozen=True)
class FreeNetVelocity:
    """Assigned outgoing velocity at the observed net-plane crossing."""

    outgoing_velocity_mps: tuple[float, float, float]

    def __post_init__(self):
        values = _finite3(self.outgoing_velocity_mps, "outgoing")
        if np.any(np.abs(values) > _MAX_COMPONENT_MPS):
            raise ValueError("finite outgoing velocity required")
        object.__setattr__(
            self,
            "outgoing_velocity_mps",
            (float(values[0]), float(values[1]), float(values[2])),
        )

    @property
    def key(self):
        return ("free_net_velocity_v1",) + self.outgoing_velocity_mps

    def velocity(self, incoming):
        _finite3(incoming, "incoming")
        return np.array(self.outgoing_velocity_mps, float)

    def record(self, incoming, outgoing):
        v = np.asarray(incoming, float)
        w = np.asarray(outgoing, float)
        return dict(
            model="experimental_free_net_velocity_v1",
            outgoing_velocity_mps=list(self.outgoing_velocity_mps),
            incoming_speed_mps=float(np.linalg.norm(v)),
            outgoing_speed_mps=float(np.linalg.norm(w)),
            translational_energy_ratio=_translational_energy_ratio(v, w),
            spin_unchanged=True,
            contact_geometry=(
                "independently fitted outgoing velocity at nominal net plane; "
                "not a passive material law"
            ),
        )


def response_from_record(record):
    """Rebuild a net response from a receipt, ignoring extra metadata."""
    if "first_ground_normal_restitution" in record:
        from cv.experiments.connected_shooting.labeled_net_normal_response import NetNormalResponse

        base = {k: v for k, v in record.items() if k != "first_ground_normal_restitution"}
        return NetNormalResponse(
            response_from_record(base),
            record["first_ground_normal_restitution"],
            horizontal_mode=record.get("first_ground_horizontal_mode", "off"),
            tangential_multiplier=record.get("first_ground_tangential_multiplier", 1.0),
            transverse_velocity_mps=record.get("first_ground_transverse_velocity_mps", 0.0),
        )
    if "outgoing_spin_rad_s" in record:
        from cv.experiments.connected_shooting.labeled_net_spin_response import FreeNetState

        return FreeNetState(record["outgoing_velocity_mps"], record["outgoing_spin_rad_s"])
    if record.get("model") == "admissible_net_response_v1" or "v_out_mps" in record:
        from cv.experiments.connected_shooting.admissible_net_response import from_record

        payload = record if "outgoing_velocity_mps" in record else {
            "outgoing_velocity_mps": record["v_out_mps"]
        }
        return from_record(payload)
    if "outgoing_velocity_mps" in record:
        return FreeNetVelocity(record["outgoing_velocity_mps"])
    angle = record["normal_angle"]
    args = (angle, record["restitution"], record["tangential_retention"])
    return SignedTapeResponse(*args) if angle < 0 else TapeResponse(*args)
