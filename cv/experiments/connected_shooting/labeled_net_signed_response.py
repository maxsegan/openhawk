"""Experimental signed effective tape impulse for reviewed terminal net drop.

TapeResponse confines the effective normal angle to [0, pi/2], which can send
incoming horizontal energy upward. A reviewed terminal drop may need the
opposite signed tilt. SignedTapeResponse allows angle in [-pi/2, pi/2] with
the same passive law n = [0, -sign(vy) cos(a), sin(a)] and
w = mu (v - (v·n) n) - e (v·n) n. It does not change TapeResponse or any
simulator default. Existing simulator(response) / using_response(response)
accept this class because they dispatch on velocity, key and record.

A negative angle is not a claim that every clip supports it. The caller must
choose and review the signed angle explicitly. Angle 0 with e = 0.25, mu = 0.4
is the original nested tape law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv.experiments.connected_shooting.labeled_passive_tape import TapeResponse


@dataclass(frozen=True)
class SignedTapeResponse(TapeResponse):
    """Passive tape response whose effective normal angle may point down."""

    def __post_init__(self):
        values = np.array([self.normal_angle, self.restitution, self.tangential_retention])
        if not np.isfinite(values).all() or not (
            -np.pi / 2 <= values[0] <= np.pi / 2 and 0 <= values[1] <= 1 and 0 < values[2] <= 1
        ):
            raise ValueError("finite signed passive response parameters required")

    def record(self, incoming, outgoing):
        receipt = super().record(incoming, outgoing)
        receipt.update(
            model="experimental_signed_passive_tape_v1",
            contact_geometry=(
                "effective signed impulse at nominal net plane; "
                "not an independently observed cable normal"
            ),
            effective_signed_impulse=True,
            independently_observed_cable_normal=False,
        )
        return receipt
