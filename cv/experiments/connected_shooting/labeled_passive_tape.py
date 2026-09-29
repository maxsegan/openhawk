"""Experimental passive tape deflection at the repository's nominal net plane.

A tilted effective impulse normal can redirect a descending ball upward while
it continues into the opposite court. This is not a resolved cylindrical cable
contact or a measured material response. Original event association, integration,
ground response, position continuity and spin are preserved. No default changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import FunctionType, SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import measured_dynamics, net_collision
from cv.pipeline.physics_knot_solver import E_NET, NET_TANG, net_impact_velocity


@dataclass(frozen=True)
class TapeResponse:
    normal_angle: float
    restitution: float
    tangential_retention: float

    def __post_init__(self):
        values = np.array([self.normal_angle, self.restitution, self.tangential_retention])
        if not np.isfinite(values).all() or not (
            0 <= values[0] <= np.pi / 2 and 0 <= values[1] <= 1 and 0 < values[2] <= 1
        ):
            raise ValueError("finite passive response parameters required")

    @property
    def key(self):
        return (float(self.normal_angle), float(self.restitution), float(self.tangential_retention))

    def normal(self, velocity):
        vy = float(np.asarray(velocity)[1])
        if vy == 0:
            raise ValueError("incoming court-normal direction unresolved")
        return np.array([0.0, -np.sign(vy) * np.cos(self.normal_angle), np.sin(self.normal_angle)])

    def velocity(self, incoming):
        v = np.asarray(incoming, float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError("finite incoming velocity required")
        n = self.normal(v)
        approach = float(v @ n)
        if approach >= 0:
            raise ValueError("incoming ball is not approaching effective tape normal")
        if self.key == (0.0, E_NET, NET_TANG):
            return net_impact_velocity(v)
        return self.tangential_retention * (v - approach * n) - self.restitution * approach * n

    def record(self, incoming, outgoing):
        v = np.asarray(incoming)
        w = np.asarray(outgoing)
        return dict(
            model="passive_inclined_tape_v1",
            normal_angle=self.normal_angle,
            restitution=self.restitution,
            tangential_retention=self.tangential_retention,
            effective_normal=self.normal(v).tolist(),
            translational_energy_ratio=float(w @ w / (v @ v)),
            spin_unchanged=True,
            contact_geometry="effective impulse at nominal net plane; cable normal not independently observed",
        )


def simulator(response):
    """Clone numerical functions with immutable law and law-specific trace cache."""
    from cv.experiments.connected_shooting.labeled_net_normal_response import NetNormalResponse

    if isinstance(response, NetNormalResponse):
        from cv.experiments.connected_shooting.labeled_net_normal_response import (
            simulator as normal_simulator,
        )

        return normal_simulator(response)
    # Nested response scopes must clone the numerical function, not the outer
    # response wrapper (whose closure and response-specific cache belong to it).
    original = getattr(net_collision.simulate, "_base_net_simulator", net_collision.simulate)
    integrate = FunctionType(
        net_collision.integrate.__code__,
        {
            **net_collision.integrate.__globals__,
            "net_impact_velocity": response.velocity,
            "net_impact_spin": getattr(response, "spin", net_collision.net_impact_spin),
        },
        name="passive_tape_integrate",
        argdefs=net_collision.integrate.__defaults__,
    )

    def cached(key, function, *args):
        return measured_dynamics.cached_integration(
            ("passive_tape_v1", response.key, *key), function, *args
        )

    simulated = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "integrate": integrate,
            "measured_dynamics": SimpleNamespace(cached_integration=cached),
        },
        name="passive_tape_simulate",
        argdefs=original.__defaults__,
    )
    simulated.__kwdefaults__ = original.__kwdefaults__

    def run(*args, **kwargs):
        result = simulated(*args, **kwargs)
        for hit in result[-1]:
            hit["experimental_response"] = response.record(hit["v_in"], hit["v_out"])
        return result

    run._base_net_simulator = original
    return run


@contextmanager
def using_response(response):
    """Worker-local override, restored even on a failed simulation."""
    prior = net_collision.simulate
    replacement = simulator(response)
    net_collision.simulate = replacement
    try:
        yield
    finally:
        net_collision.simulate = prior
