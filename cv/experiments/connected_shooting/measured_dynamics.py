"""Research model-matched capacity control: measured rebound and explicit dwell.

Uses physics modules, never validation truth. Impacts are detected from flight,
not inserted from labels. Matching a generating model is not independent validation.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np

from physics import bounce_reference
from cv.experiments.connected_shooting import passive_bounce
from cv.experiments.connected_shooting import fast_flight
from cv.experiments.connected_shooting.ground_root import descending_ground_root
from cv.pipeline.rich_ball_physics import MAX_SIMULATED_BOUNCES, R_BALL, spin_vector


class BounceCapacityError(ValueError):
    """The unchanged finite-impact guard refused further propagation."""

    def __init__(self, frame: float, resolved_impacts: int):
        super().__init__("measured dynamics bounce cap reached")
        self.frame = float(frame)
        self.resolved_impacts = int(resolved_impacts)

    def __reduce__(self):
        return type(self), (self.frame, self.resolved_impacts)


# One process-wide bounded exact-value store.  Keys carry every input that can
# change a trace, so a hit returns the value the integrator would have produced.
TRACE_CACHE_ENTRIES = 96
_TRACE_CACHE: OrderedDict = OrderedDict()


# Fixed sensitivity controls, not calibrated uncertainty or fitted parameters.
# Entries are (vertical restitution multiplier, horizontal retention multiplier).
BOUNCE_PROFILES = {
    "nominal": (1.0, 1.0),
    "restitution_low": (0.95, 1.0),
    "restitution_high": (1.05, 1.0),
    "retention_low": (1.0, 0.95),
    "retention_high": (1.0, 1.05),
}


def validate_rebound_scales(scales) -> np.ndarray:
    scales = np.asarray(scales, float)
    if (
        scales.shape != (2,)
        or not np.isfinite(scales).all()
        or np.any(scales < 0.8)
        or np.any(scales > 1.2)
    ):
        raise ValueError("two finite rebound corrections in [0.8, 1.2] required")
    return scales


def rebound_velocity(rebound, profile: str, scales=(1.0, 1.0)) -> tuple[np.ndarray, dict]:
    """Perturb only rebound translation, with explicit coefficient saturation.

    Spin and dwell stay nominal. This is a phenomenological sensitivity test,
    not a coupled friction/spin impact model or an energy certificate.
    """
    if profile not in BOUNCE_PROFILES:
        raise ValueError("explicit supported bounce profile required")
    scales = validate_rebound_scales(scales)
    vertical, horizontal = np.asarray(BOUNCE_PROFILES[profile]) * scales
    requested = np.array(
        [rebound.restitution * vertical, rebound.horizontal_retention * horizontal]
    )
    applied = np.clip(requested, 0.05, 1.0)
    velocity = rebound.velocity.copy()
    if profile != "nominal" or np.any(scales != 1.0):
        velocity[:2] *= applied[1] / rebound.horizontal_retention
        velocity[2] *= applied[0] / rebound.restitution
    return velocity, {
        "bounce_profile": profile,
        "requested_restitution": float(requested[0]),
        "requested_horizontal_retention": float(requested[1]),
        "applied_restitution": float(applied[0]),
        "applied_horizontal_retention": float(applied[1]),
        "coefficient_clipped": bool(np.any(requested != applied)),
    }


def _trace_key(theta, f0, end, fps, surface, bounce_profile, rebound_scales, override):
    return (
        np.asarray(theta, float).tobytes(),
        float(f0),
        float(end),
        float(fps),
        surface,
        bounce_profile,
        np.asarray(rebound_scales, float).tobytes(),
        override,
    )


def integrate(
    theta,
    f0,
    end,
    fps,
    surface,
    bounce_profile,
    rebound_scales,
    override,
    stop_after_first_impact=False,
    ground_settling=False,
    initial_successive_grounds=0,
):
    """Run the 240 Hz forward propagation once and return its raw trace.

    Split out of :func:`simulate` because the connected search asks for the same
    physical flight at several query sets inside one objective evaluation: the
    exposure sweep re-queries a flight after its bounce knots are known, and the
    physical checks re-query it densely.  Only the sampling differs, so the
    integration is memoized on every input that can change it.  ``end`` is part
    of the key because it terminates the loop. The uncached first-impact-only
    mode is for root bracketing; it must never supply a complete flight trace.
    """
    x = np.array(theta[:3], float)
    v = np.array(theta[3:6], float)
    w = spin_vector(theta)
    state = (
        float(x[0]),
        float(x[1]),
        float(x[2]),
        float(v[0]),
        float(v[1]),
        float(v[2]),
        float(w[0]),
        float(w[1]),
        float(w[2]),
    )
    times = [0.0]
    positions = [state[0:3]]
    velocities = [state[3:6]]
    spins = [state[6:9]]
    elapsed = 0.0
    bounces = []
    successive_grounds = initial_successive_grounds
    from cv.experiments.connected_shooting import ground_contact

    settled = ground_settling and ground_contact.qualifies(state[:3], state[3:6])
    if settled:
        state = ground_contact.advance(state, 0.0, surface)
        velocities[0] = state[3:6]
    step = 1.0 / 240.0
    while elapsed < end:
        if settled:
            state = ground_contact.advance(state, step, surface)
            elapsed += step
            times.append(elapsed)
            positions.append(state[:3])
            velocities.append(state[3:6])
            spins.append(state[6:9])
            continue
        px, py, pz, vx, vy, vz, wx, wy, wz = state
        if (
            fast_flight.exceeds(px, py, pz, 1000)
            or fast_flight.exceeds(vx, vy, vz, 250)
            or fast_flight.exceeds(wx, wy, wz, 5000)
        ):
            raise ValueError("measured forward state outside search bounds")
        nxt = fast_flight.rk4_step(state, step)
        if not all(map(math.isfinite, nxt)):
            raise ValueError("nonfinite measured forward state")
        if nxt[2] <= R_BALL and nxt[5] < 0:
            if pz <= R_BALL and vz <= 0:
                raise ValueError("descending initial ground state has no supported flight")
            impact_seconds, impact = descending_ground_root(state, nxt, step)
            impact_time = elapsed + impact_seconds
            if impact_time > end:
                # Keep the same incoming interpolation segment as a longer
                # query that includes this root. The full RK4 endpoint is
                # below ground and is not collinear with the refined root.
                # This future knot supports sampling, not an exported event.
                times.append(impact_time)
                positions.append((impact[0], impact[1], R_BALL))
                velocities.append(impact[3:6])
                spins.append(impact[6:9])
                break
            xb = np.array((impact[0], impact[1], R_BALL))
            vb = np.array(impact[3:6])
            wb = np.array(impact[6:9])
            regime = override[1] if override is not None and len(bounces) == override[0] else None
            # The landing point is additive: a bare surface name ignores it and behaves
            # exactly as before, and a region-aware surface model needs it (physics.surface_model).
            rebound = passive_bounce.nominal_state(
                vb,
                wb,
                surface,
                regime=regime,
                position=xb,
                successive_grounds=successive_grounds,
            )
            outgoing, rebound_record = rebound_velocity(rebound, bounce_profile, rebound_scales)
            rebound, outgoing, rebound_record = passive_bounce.apply(
                vb, wb, rebound, outgoing, rebound_record, surface, successive_grounds
            )
            successive_grounds += 1
            bounces.append(
                {
                    "frame": f0 + impact_time * fps,
                    "x": xb.copy(),
                    "v_in": vb.copy(),
                    "v_out": outgoing.copy(),
                    "w_in": wb.copy(),
                    "w_out": rebound.spin.copy(),
                    "regime": rebound.regime,
                    "dwell_seconds": bounce_reference.DWELL_SECONDS,
                    **rebound_record,
                }
            )
            if stop_after_first_impact:
                times.append(impact_time)
                positions.append(tuple(xb))
                velocities.append(tuple(vb))
                spins.append(tuple(wb))
                break
            settled = ground_settling and ground_contact.qualifies(xb, outgoing)
            if settled:
                bounces[-1].update(
                    ground_contact.transition_record(
                        outgoing, resolved_impacts=len(bounces) + initial_successive_grounds
                    )
                )
            if not ground_settling and len(
                bounces
            ) + initial_successive_grounds >= passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES):
                raise BounceCapacityError(f0 + impact_time * fps, len(bounces))
            elapsed = impact_time + bounce_reference.DWELL_SECONDS
            ground = (float(xb[0]), float(xb[1]), float(xb[2]))
            out_v = (float(outgoing[0]), float(outgoing[1]), float(outgoing[2]))
            out_w = (float(rebound.spin[0]), float(rebound.spin[1]), float(rebound.spin[2]))
            times.extend([impact_time, elapsed])
            positions.extend([ground, ground])
            velocities.extend([impact[3:6], out_v])
            spins.extend([impact[6:9], out_w])
            state = ground + out_v + out_w
            if settled:
                state = ground_contact.advance(state, 0.0, surface)
                velocities[-1] = state[3:6]
        else:
            elapsed += step
            times.append(elapsed)
            positions.append(nxt[0:3])
            velocities.append(nxt[3:6])
            spins.append(nxt[6:9])
            state = nxt
    return (
        np.asarray(times),
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(spins),
        bounces,
    )


def cached_integration(key, function, *args):
    """Return ``function(*args)`` for a key that already carries every input."""
    rows = _TRACE_CACHE
    hit = rows.get(key)
    if hit is not None:
        rows.move_to_end(key)
        return hit
    value = function(*args)
    rows[key] = value
    if len(rows) > TRACE_CACHE_ENTRIES:
        rows.popitem(last=False)
    return value


def resample(trace, queries, f0, fps):
    """Sample a retained trace; bounce dwell stays an unresolved impact."""
    times, positions, velocities, spins, bounces = trace
    seconds = (queries - f0) / fps
    sampled = [
        np.column_stack([np.interp(seconds, times, values[:, k]) for k in range(3)])
        for values in (positions, velocities, spins)
    ]
    for bounce in bounces:
        low = (bounce["frame"] - f0) / fps
        inside = (seconds > low) & (seconds < low + bounce_reference.DWELL_SECONDS)
        sampled[1][inside] = 0.0
    return *sampled, [
        {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in row.items()} for row in bounces
    ]


def simulate(
    theta: np.ndarray,
    f0: float,
    queries: np.ndarray,
    fps: float,
    surface: str,
    *,
    bounce_profile: str = "nominal",
    rebound_scales=(1.0, 1.0),
    bounce_regime_override: tuple[int, str] | None = None,
    ground_settling: bool | None = None,
    initial_successive_grounds: int = 0,
):
    if bounce_profile not in BOUNCE_PROFILES:
        raise ValueError("explicit supported bounce profile required")
    from cv.experiments.connected_shooting import ground_contact

    if ground_settling is None:
        ground_settling = ground_contact.active()
    if type(ground_settling) is not bool:
        raise ValueError("explicit boolean ground-settling policy required")
    # Settling resolves every above-threshold rebound through the query horizon.
    # The prior count affects the response law, not numerical admissibility.
    if (
        type(initial_successive_grounds) is not int
        or initial_successive_grounds < 0
        or (
            not ground_settling
            and initial_successive_grounds >= passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES)
        )
    ):
        raise ValueError("finite prior successive-ground count within the bounce cap required")
    rebound_scales = validate_rebound_scales(rebound_scales)
    if bounce_regime_override is not None and (
        len(bounce_regime_override) != 2
        or not isinstance(bounce_regime_override[0], int)
        or bounce_regime_override[0] < 0
        or (
            not ground_settling
            and bounce_regime_override[0] >= passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES)
        )
        or bounce_regime_override[1] not in {"slide", "grip"}
    ):
        raise ValueError("one explicit bounce ordinal/regime surrogate required")
    queries = np.asarray(queries, float)
    if (
        queries.ndim != 1
        or not len(queries)
        or not np.isfinite(queries).all()
        or np.any(np.diff(queries) <= 0)
        or not np.isfinite(f0)
        or not np.isfinite(fps)
        or fps <= 0
        or queries[0] < f0
        or np.shape(theta) != (9,)
        or not np.isfinite(theta).all()
        or theta[2] < R_BALL - 1e-9
    ):
        raise ValueError("finite forward measured-dynamics initial state and queries required")
    if surface not in bounce_reference.surfaces():
        raise ValueError("measured dynamics requires a measured surface")
    end = float((queries[-1] - f0) / fps)
    override = None if bounce_regime_override is None else tuple(bounce_regime_override)
    trace = cached_integration(
        passive_bounce.cache_prefix()
        + (("prior_grounds", initial_successive_grounds) if initial_successive_grounds else ())
        + ((ground_contact.CACHE_TAG,) if ground_settling else ())
        + _trace_key(theta, f0, end, fps, surface, bounce_profile, rebound_scales, override),
        integrate,
        theta,
        f0,
        end,
        fps,
        surface,
        bounce_profile,
        rebound_scales,
        override,
        *(
            (False, ground_settling, initial_successive_grounds)
            if initial_successive_grounds
            else ((False, True) if ground_settling else ())
        ),
    )
    return resample(trace, queries, f0, fps)
