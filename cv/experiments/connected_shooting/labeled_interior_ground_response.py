"""Persistent ground normal response for original native flight identities.

Worker-local, never a measured surface law. Compose this scope with any net
response; ground-only original flights cannot alias the net integrator's impacts.
Each route names one original flight and one original ground ordinal: the first
ground (ordinal 0) or, for a flight whose original events supply two grounds,
the second ground (ordinal 1). Later simulated grounds keep the nominal law.
Records holding only first-ground routes are unchanged in content and order.
"""

from contextlib import contextmanager
from copy import deepcopy
from types import FunctionType

import numpy as np

from cv.experiments.connected_shooting import measured_dynamics as dynamics

SCHEMA = "interior_ground_response_v1"
ORDINALS = (0, 1)


def normalize(record, scene=None):
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("explicit interior ground-response schema required")
    rows = []
    for row in record.get("routes", []):
        i, epoch, ordinal, coefficient = (
            row["flight_index"],
            row["native_contact_epoch"],
            row["ground_ordinal"],
            row["normal_restitution"],
        )
        if (
            type(i) is not int
            or i < 0
            or type(ordinal) is not int
            or ordinal not in ORDINALS
            or not np.isfinite([epoch, coefficient]).all()
            or not 0.05 <= coefficient <= 1
        ):
            raise ValueError("original flight identity, ordinal 0 or 1 and COR [.05,1] required")
        if scene is not None:
            if i >= len(scene.pixels) or float(scene.contact_frames[i]) != float(epoch):
                raise ValueError("ground response differs from original native flight identity")
            if scene.net_hit_frames is not None and len(scene.net_hit_frames[i]):
                raise ValueError("interior ground response excludes declared net flight")
        normalized = dict(
            flight_index=i,
            native_contact_epoch=float(epoch),
            ground_ordinal=ordinal,
            normal_restitution=float(coefficient),
        )
        if "horizontal_retention" in row:
            retention = row["horizontal_retention"]
            if not np.isfinite(retention) or not 0.05 <= retention <= 1:
                raise ValueError("absolute horizontal retention in [.05,1] required")
            normalized["horizontal_retention"] = float(retention)
        rows.append(normalized)
    if len({(r["flight_index"], r["ground_ordinal"]) for r in rows}) != len(rows):
        raise ValueError("unique original flight response routes required")
    epochs = {}
    for r in rows:
        epochs.setdefault(r["flight_index"], set()).add(r["native_contact_epoch"])
    if any(len(v) != 1 for v in epochs.values()) or len(
        {e for v in epochs.values() for e in v}
    ) != len(epochs):
        raise ValueError("one native contact epoch per original flight route required")
    return dict(
        schema=SCHEMA,
        routes=sorted(rows, key=lambda r: (r["flight_index"], r["ground_ordinal"])),
    )


def merge(record, routes, scene=None):
    """Replace only explicitly named original routes; retain unrelated responses.

    A route is named by its flight and ground ordinal, so a first-ground update
    never discards a retained second-ground response of the same flight.
    """
    old = normalize(record, scene)
    incoming = normalize(dict(schema=SCHEMA, routes=routes), scene)
    rows = {(r["flight_index"], r["ground_ordinal"]): r for r in old["routes"]} if old else {}
    for row in incoming["routes"]:
        for (i, _), previous in rows.items():
            if i == row["flight_index"] and (
                previous["native_contact_epoch"] != row["native_contact_epoch"]
            ):
                raise ValueError("cannot change an existing original response epoch")
        key = (row["flight_index"], row["ground_ordinal"])
        # A normal-only refinement must not silently erase an earlier optional
        # horizontal response on the same original impact. Absent fields on a
        # new route remain nominal; existing records without them are unchanged.
        rows[key] = {**rows.get(key, {}), **row}
    return normalize(dict(schema=SCHEMA, routes=list(rows.values())), scene)


def simulator(record, scene=None):
    record = normalize(record, scene)
    original = dynamics.simulate
    if record is None or not record["routes"]:
        return original
    original = getattr(original, "_interior_ground_base_simulate", original)
    mapped = {}
    for row in record["routes"]:
        mapped.setdefault(row["native_contact_epoch"], {})[row["ground_ordinal"]] = row

    def run(theta, first, queries, fps, surface, **kwargs):
        rows = mapped.get(float(first))
        if rows is None:
            return original(theta, first, queries, fps, surface, **kwargs)
        flight = next(iter(rows.values()))["flight_index"]
        key = (
            SCHEMA,
            flight,
            float(first),
            tuple(
                (ordinal, rows[ordinal]["normal_restitution"])
                + (
                    (rows[ordinal]["horizontal_retention"],)
                    if "horizontal_retention" in rows[ordinal]
                    else ()
                )
                for ordinal in sorted(rows)
            ),
        )

        def integrate(*args):
            ordinal = 0

            def rebound(state, profile, scales=(1.0, 1.0)):
                nonlocal ordinal
                velocity, receipt = dynamics.rebound_velocity(state, profile, scales)
                row = rows.get(ordinal)
                if row is not None:
                    coefficient = row["normal_restitution"]
                    nominal = receipt["applied_restitution"]
                    if coefficient != nominal:
                        velocity = velocity.copy()
                        velocity[2] *= coefficient / nominal
                    receipt = {
                        **receipt,
                        "interior_ground_response": deepcopy(row),
                        "nominal_applied_restitution": nominal,
                        "applied_restitution": coefficient,
                        "measured_surface_law": False,
                    }
                    if "horizontal_retention" in row:
                        retention = row["horizontal_retention"]
                        nominal_retention = receipt["applied_horizontal_retention"]
                        if retention != nominal_retention:
                            velocity = velocity.copy()
                            velocity[:2] *= retention / nominal_retention
                        receipt = {
                            **receipt,
                            "nominal_applied_horizontal_retention": nominal_retention,
                            "applied_horizontal_retention": retention,
                            "horizontal_response_kind": "phenomenological_translation_only",
                        }
                ordinal += 1
                return velocity, receipt

            function = FunctionType(
                dynamics.integrate.__code__,
                {**dynamics.integrate.__globals__, "rebound_velocity": rebound},
                argdefs=dynamics.integrate.__defaults__,
            )
            return function(*args)

        def cached(trace_key, function, *args):
            return dynamics.cached_integration((key, *trace_key), function, *args)

        changed = FunctionType(
            original.__code__,
            {**original.__globals__, "integrate": integrate, "cached_integration": cached},
            argdefs=original.__defaults__,
        )
        changed.__kwdefaults__ = original.__kwdefaults__
        return changed(theta, first, queries, fps, surface, **kwargs)

    run._interior_ground_base_simulate = original
    return run


@contextmanager
def using_response(record=None, scene=None):
    """Complete registry applies throughout one worker evaluation, then restores."""
    previous = dynamics.simulate
    dynamics.simulate = simulator(record, scene)
    try:
        yield
    finally:
        dynamics.simulate = previous
