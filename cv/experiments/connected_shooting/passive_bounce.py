"""Default-off coupled response after uninterrupted successive ground impacts.

The first ground after a racket/net impulse stays nominal. Subsequent grounds
keep the active normal law but use one Coulomb translation/spin impulse. The
nominal horizontal multiplier does not apply to those coupled impacts. This is
an approximate rigid-sphere response family, not a measured passive-ball law.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import math

import numpy as np

from physics import bounce_reference, surface_model
from physics.passive_bounce import coupled_rebound

MODES = ("off", "coupled_slip")
# Numerical capacity, not a different settling criterion or an impact prior.
# Preserved tangential motion creates more resolved, shallow subsequent hops.
COUPLED_IMPACT_CAPACITY = 32
_ACTIVE = ContextVar("passive_bounce_response", default="off")


def validate(value):
    if value not in MODES:
        raise ValueError("passive_bounce_response must be off or coupled_slip")
    return value


def active():
    return _ACTIVE.get()


@contextmanager
def using(mode):
    token = _ACTIVE.set(validate(mode))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def impact_capacity(legacy_limit):
    return legacy_limit if active() == "off" else max(legacy_limit, COUPLED_IMPACT_CAPACITY)


def cache_prefix():
    return (
        ()
        if active() == "off"
        else ("subsequent_coupled_slip_v1", "impact_capacity", COUPLED_IMPACT_CAPACITY)
    )


def nominal_state(velocity, spin, surface, *, position, regime, successive_grounds):
    """Keep nominal arithmetic; permit a vertical subsequent coupled impact."""
    if active() == "off" or successive_grounds == 0 or np.linalg.norm(velocity[:2]) > 0:
        return bounce_reference.court_bounce(
            velocity, spin, surface, spin_regime_override=regime, position=position
        )
    # The nominal routine rejects zero horizontal speed. Only its normal law is
    # needed here: no tangent direction or fictitious horizontal speed is added.
    model = surface_model.parse(surface)
    normal, retention = bounce_reference.base_coefficients(90.0, model)
    if not model.inert:
        normal, retention, state = model.coefficients(
            float(position[0]), float(position[1]), normal, retention
        )
    else:
        state = None
    return bounce_reference.MeasuredBounceResult(
        velocity=np.array([0.0, 0.0, -velocity[2] * normal]),
        spin=np.asarray(spin, float).copy(),
        restitution=normal,
        horizontal_retention=retention,
        regime="vertical_normal_reference",
        spin_source="incoming placeholder; coupled impulse supplies outgoing spin",
        surface_state=state,
    )


def apply(velocity, spin, rebound, outgoing, receipt, surface, successive_grounds):
    """Preserve active normal overrides and their counters before coupling."""
    if active() == "off" or successive_grounds == 0:
        return rebound, outgoing, receipt
    if receipt.get("horizontal_response_kind") == "phenomenological_translation_only":
        raise ValueError("coupled passive response cannot apply a separate horizontal override")
    coefficient = float(outgoing[2] / -velocity[2])
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError("coupled passive response requires physical active normal restitution")
    response = coupled_rebound(
        velocity, spin, restitution=coefficient, surface=surface_model.parse(surface).surface
    )
    # Keep the normal component bit-identical to the already-applied response.
    # The coefficient division/multiplication above can otherwise differ by an ulp.
    response.velocity[2] = outgoing[2]
    record = {
        **receipt,
        "passive_bounce_response": "coupled_slip",
        "passive_bounce_impact_capacity": COUPLED_IMPACT_CAPACITY,
        "successive_ground_index": successive_grounds,
        "nominal_applied_horizontal_retention": receipt["applied_horizontal_retention"],
        "nominal_requested_horizontal_retention": receipt["requested_horizontal_retention"],
        "requested_horizontal_retention": None,
        "coefficient_clipped": receipt["requested_restitution"] != receipt["applied_restitution"],
        "horizontal_point_scale_applied": False,
        "coupled_impulse": response.receipt,
        "applied_horizontal_retention": response.receipt["horizontal_speed_retained_ratio"],
        "horizontal_response_kind": "coupled_coulomb_impulse",
        "measured_surface_law": False,
    }
    return (
        replace(
            rebound,
            velocity=response.velocity,
            spin=response.spin,
            horizontal_retention=response.receipt["horizontal_speed_retained_ratio"],
            restitution=coefficient,
            regime="coupled_slip",
            spin_source="coupled Coulomb point impulse; not a spin measurement",
        ),
        response.velocity,
        record,
    )
