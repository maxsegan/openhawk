"""Measured-dynamics propagation with an explicit, supplied net-hit transition.

The event time is evidence, not a fitted picture or a snapped position.  The
ball position remains continuous; the default retains spin and changes velocity
according to the existing tape-impact law. Explicit research responses can also
change spin at the same contact. Separate residuals require the
continuous state to reach the net plane below the measured tape height.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from cv.experiments.connected_shooting import fast_flight, measured_dynamics
from cv.experiments.connected_shooting.measured_dynamics import (
    BOUNCE_PROFILES,
    rebound_velocity,
    validate_rebound_scales,
)
from cv.pipeline.physics_knot_solver import net_impact_velocity, net_tape_height
from cv.pipeline.rich_ball_physics import MAX_SIMULATED_BOUNCES, R_BALL, spin_vector
from cv.experiments.connected_shooting.ground_root import descending_ground_root, step_segment
from physics import bounce_reference
from cv.experiments.connected_shooting import passive_bounce


_PHYSICAL_ELIGIBILITY = ContextVar("declared_net_physical_eligibility", default=False)


@contextmanager
def physical_eligibility(enabled: bool = True):
    """Optional declared-net physics; event-time agreement stays a residual.

    Only callers already dispatching a supplied net flight enter this simulator.
    This mode does not add net topology to collision-free flights. Context-local
    state is read by cloned response simulators and included in their trace keys.
    """
    if not isinstance(enabled, bool):
        raise TypeError("physical net eligibility requires a boolean")
    token = _PHYSICAL_ELIGIBILITY.set(enabled)
    try:
        yield
    finally:
        _PHYSICAL_ELIGIBILITY.reset(token)


def active_policy() -> str:
    """Serializable identity required to replay an optional physical-mode result."""
    return "physical_mesh_v1" if _PHYSICAL_ELIGIBILITY.get() else "timing_window_v1"


def require_policy(recorded: str) -> None:
    """Reject replay under different collision semantics, including absent scope."""
    if recorded != active_policy():
        raise ValueError("recorded net collision policy differs from active replay policy")


def _collision_eligible(hit_frame, supplied_frame, hit):
    if not _PHYSICAL_ELIGIBILITY.get():
        return abs(hit_frame - supplied_frame) <= 1.0
    from cv.pipeline import net_cord_response as cord

    within_posts = abs(float(hit[0]) - 5.485) <= 6.4
    if cord.uses_tape_band(float(supplied_frame)):
        # Tape band, not the whole mesh: a cord clip is at the tape.
        return within_posts and cord.in_tape_band((hit[0], 11.885, hit[2]))
    # Match the repository's measured net span and tape/radius convention.
    # A mathematical net-plane crossing above/wide of the mesh is not a hit.
    return (
        within_posts and R_BALL <= float(hit[2]) <= net_tape_height(float(hit[0])) + R_BALL
    )


def _eligibility_cache_key(key):
    return ("physical_mesh_v1", *key) if _PHYSICAL_ELIGIBILITY.get() else key


def net_impact_spin(incoming_spin, incoming_velocity, outgoing_velocity):
    """Original net model preserves spin; explicit research responses may override."""
    return np.asarray(incoming_spin, float).copy()


def integrate(
    theta,
    f0,
    end,
    fps,
    surface,
    net_frame,
    bounce_profile,
    rebound_scales,
    override,
    ground_settling=False,
):
    """Run the net-transition propagation once; see measured_dynamics.integrate."""
    spin = spin_vector(theta)
    state = (
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
        float(theta[3]),
        float(theta[4]),
        float(theta[5]),
        float(spin[0]),
        float(spin[1]),
        float(spin[2]),
    )
    elapsed = 0.0
    times = [0.0]
    positions = [state[0:3]]
    velocities = [state[3:6]]
    spins = [state[6:9]]
    bounces: list[dict] = []
    successive_grounds = 0
    net_hits: list[dict] = []
    from cv.experiments.connected_shooting import ground_contact

    settled = ground_settling and bool(net_hits) and ground_contact.qualifies(state[:3], state[3:6])
    if settled:
        state = ground_contact.advance(state, 0.0, surface)
        velocities[0] = state[3:6]
    step = 1.0 / 240.0
    while elapsed < end - 1e-12:
        if settled:
            state = ground_contact.advance(state, step, surface)
            elapsed += step
            times.append(elapsed)
            positions.append(state[:3])
            velocities.append(state[3:6])
            spins.append(state[6:9])
            continue
        # Keep the 240 Hz grid independent of the final query, as the ordinary
        # measured simulator does. This makes terminal-impact replay identical
        # when the caller shortens the domain to the detected impact.
        dt = step
        px, py, pz, vx, vy, vz = state[:6]
        nxt = fast_flight.rk4_step(state, dt)
        ground_state, ground_elapsed, ground_dt = state, elapsed, dt
        # The plane crossing is located on the segment the measured sampler keeps
        # for this step: the whole step, or the chord to the refined ground root
        # when the step also holds a descending ground crossing. The RK4 endpoint
        # below the floor is not collinear with that root, so a whole-step chord
        # would place a near-floor hit below R_BALL and could order a crossing
        # before a ground impact that physically precedes it.
        segment_end, segment_dt, grounded = step_segment(state, nxt, dt)
        crossed_net = (py - 11.885) * (segment_end[1] - 11.885) <= 0 and py != segment_end[1]
        if not net_hits and crossed_net:
            fraction = float(np.clip((11.885 - py) / (segment_end[1] - py), 0.0, 1.0))
            hit_seconds = elapsed + fraction * segment_dt
            hit_frame = f0 + hit_seconds * fps
            hit = tuple(a + fraction * (b - a) for a, b in zip(state, segment_end))
            if _collision_eligible(hit_frame, net_frame, hit):
                if hit_seconds > end:
                    # The queried domain does not own this contact. Keep the
                    # incoming knot so samples up to ``end`` equal those of a
                    # longer query that owns it, but record no event and apply
                    # no impulse, as the ordinary measured integrator does for
                    # a ground root beyond its horizon.
                    times.append(hit_seconds)
                    positions.append((hit[0], 11.885, hit[2]))
                    velocities.append(hit[3:6])
                    spins.append(hit[6:9])
                    break
                if grounded and fraction >= 1.0:
                    # The mesh contact coincides with the ground root: no ordered,
                    # positive-duration net-to-ground flight exists.
                    raise ValueError(
                        "simultaneous net and ground contact has no supported ordering"
                    )
                xh = np.array((hit[0], 11.885, hit[2]))
                vh = np.array(hit[3:6])
                wh = np.array(hit[6:9])
                successive_grounds = 0
                outgoing = net_impact_velocity(vh)
                from cv.pipeline import net_cord_response as cord

                outgoing = cord.constrain_outgoing(vh, outgoing, float(net_frame))
                outgoing_spin = net_impact_spin(wh, vh, outgoing)
                net_hits.append(
                    {
                        "frame": float(hit_frame),
                        "supplied_frame": float(net_frame),
                        "x": xh.copy(),
                        "v_in": vh.copy(),
                        "v_out": outgoing.copy(),
                        "w_in": wh.copy(),
                        "w_out": outgoing_spin.copy(),
                        "tape_height_m": net_tape_height(float(xh[0])),
                        "position_continuous": True,
                    }
                )
                tape = (float(xh[0]), 11.885, float(xh[2]))
                out_v = (float(outgoing[0]), float(outgoing[1]), float(outgoing[2]))
                out_w = tuple(float(w) for w in outgoing_spin)
                times.extend([hit_seconds, hit_seconds])
                positions.extend([tape, tape])
                velocities.extend([hit[3:6], out_v])
                spins.extend([hit[6:9], out_w])
                remaining = dt - fraction * segment_dt
                ground_state = tape + out_v + out_w
                ground_elapsed, ground_dt = hit_seconds, remaining
                nxt = fast_flight.rk4_step(ground_state, remaining)
        if nxt[2] <= R_BALL and nxt[5] < 0:
            if ground_dt == 0:
                # A net impulse exactly at this step's floor endpoint does not
                # supply an ordered, positive-duration net-to-ground flight.
                raise ValueError("simultaneous net and ground contact has no supported ordering")
            if ground_state[2] <= R_BALL and ground_state[5] <= 0:
                raise ValueError("descending initial ground state has no supported flight")
            impact_seconds, impact = descending_ground_root(ground_state, nxt, ground_dt)
            impact_time = ground_elapsed + impact_seconds
            if impact_time > end:
                # Same future-knot contract as measured_dynamics.integrate: the
                # refined root supports sampling up to ``end``; the impact is
                # not this query's event and no rebound is applied.
                times.append(impact_time)
                positions.append((impact[0], impact[1], R_BALL))
                velocities.append(impact[3:6])
                spins.append(impact[6:9])
                break
            xb = np.array((impact[0], impact[1], R_BALL))
            vb = np.array(impact[3:6])
            wb = np.array(impact[6:9])
            regime = override[1] if override is not None and len(bounces) == override[0] else None
            # Additive: a bare surface name ignores the landing point (physics.surface_model).
            rebound = passive_bounce.nominal_state(
                vb,
                wb,
                surface,
                regime=regime,
                position=xb,
                successive_grounds=successive_grounds,
            )
            outgoing, record = rebound_velocity(rebound, bounce_profile, rebound_scales)
            rebound, outgoing, record = passive_bounce.apply(
                vb, wb, rebound, outgoing, record, surface, successive_grounds
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
                    **record,
                }
            )
            settled = ground_settling and bool(net_hits) and ground_contact.qualifies(xb, outgoing)
            if settled:
                bounces[-1].update(
                    ground_contact.transition_record(outgoing, resolved_impacts=len(bounces))
                )
            # Normal-contact continuation requires the declared net impulse.
            # Preserve the original pre-net invalid-candidate guard.
            if not (ground_settling and net_hits) and len(
                bounces
            ) >= passive_bounce.impact_capacity(MAX_SIMULATED_BOUNCES):
                raise measured_dynamics.BounceCapacityError(f0 + impact_time * fps, len(bounces))
            elapsed = impact_time + bounce_reference.DWELL_SECONDS
            ground = (float(xb[0]), float(xb[1]), float(xb[2]))
            out_v = (float(outgoing[0]), float(outgoing[1]), float(outgoing[2]))
            out_w = (float(rebound.spin[0]), float(rebound.spin[1]), float(rebound.spin[2]))
            state = ground + out_v + out_w
            if settled:
                state = ground_contact.advance(state, 0.0, surface)
                out_v = state[3:6]
            times.extend([impact_time, elapsed])
            positions.extend([ground, ground])
            velocities.extend([impact[3:6], out_v])
            spins.extend([impact[6:9], out_w])
            continue
        elapsed += dt
        state = nxt
        times.append(elapsed)
        positions.append(state[0:3])
        velocities.append(state[3:6])
        spins.append(state[6:9])
    return (
        np.asarray(times),
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(spins),
        bounces,
        net_hits,
    )


def _copy_events(rows):
    return [
        {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in row.items()} for row in rows
    ]


def resample(trace, queries, f0, fps):
    times, positions, velocities, spins, bounces, net_hits = trace
    seconds = (queries - f0) / fps
    sampled = [
        np.column_stack([np.interp(seconds, times, values[:, k]) for k in range(3)])
        for values in (positions, velocities, spins)
    ]
    for bounce in bounces:
        exact = np.isclose(queries, float(bounce["frame"]), atol=1e-8, rtol=0)
        sampled[0][exact] = bounce["x"]
        sampled[1][exact] = bounce["v_in"]
        sampled[2][exact] = bounce["w_in"]
    return *sampled, _copy_events(bounces), _copy_events(net_hits)


def simulate(
    theta: np.ndarray,
    f0: float,
    queries: np.ndarray,
    fps: float,
    surface: str,
    *,
    net_frame: float,
    bounce_profile: str = "nominal",
    rebound_scales=(1.0, 1.0),
    bounce_regime_override: tuple[int, str] | None = None,
    ground_settling: bool | None = None,
):
    """Propagate a declared-net flight; supplied epoch remains timing evidence."""
    from cv.experiments.connected_shooting import ground_contact

    if ground_settling is None:
        ground_settling = ground_contact.active()
    if type(ground_settling) is not bool:
        raise ValueError("explicit boolean ground-settling policy required")
    if bounce_profile not in BOUNCE_PROFILES:
        raise ValueError("explicit supported bounce profile required")
    rebound_scales = validate_rebound_scales(rebound_scales)
    queries = np.asarray(queries, float)
    if (
        queries.ndim != 1
        or not len(queries)
        or np.any(np.diff(queries) <= 0)
        or not np.isfinite(queries).all()
        or not f0 < net_frame <= queries[-1]
        or queries[0] < f0
        or np.shape(theta) != (9,)
        or not np.isfinite(theta).all()
        or theta[2] < R_BALL - 1e-9
    ):
        raise ValueError("finite ordered net-transition flight required")
    end = (float(queries[-1]) - f0) / fps
    override = None if bounce_regime_override is None else tuple(bounce_regime_override)
    key = (
        np.asarray(theta, float).tobytes(),
        float(f0),
        float(end),
        float(fps),
        surface,
        float(net_frame),
        bounce_profile,
        np.asarray(rebound_scales, float).tobytes(),
        override,
    )
    trace = measured_dynamics.cached_integration(
        passive_bounce.cache_prefix()
        + _eligibility_cache_key(
            (ground_contact.CACHE_TAG, "net", *key) if ground_settling else ("net", *key)
        ),
        integrate,
        theta,
        f0,
        end,
        fps,
        surface,
        net_frame,
        bounce_profile,
        rebound_scales,
        override,
        *((True,) if ground_settling else ()),
    )
    return resample(trace, queries, f0, fps)


def residuals(
    net_hits: list[dict], supplied_frame: float, *, plane_scale_m: float = 0.03
) -> np.ndarray:
    """Require a physically located tape contact without moving the state."""
    if len(net_hits) != 1:
        return np.array([10.0, 10.0, 10.0, 10.0])
    hit = net_hits[0]
    xyz = np.asarray(hit["x"], float)
    half_width = 6.4
    centre_x = 5.485
    horizontal_excess = max(abs(float(xyz[0]) - centre_x) - half_width, 0.0)
    from cv.pipeline import net_cord_response as cord

    if cord.uses_tape_band(float(hit["frame"])):
        lo, hi = cord.tape_band(float(xyz[0]))
        band_excess = max(lo - float(xyz[2]), 0.0) + max(float(xyz[2]) - hi, 0.0)
        height_excess = band_excess
    else:
        height_excess = max(float(xyz[2]) - float(hit["tape_height_m"]) - R_BALL, 0.0)
    return np.array(
        [
            (float(xyz[1]) - 11.885) / plane_scale_m,
            height_excess / plane_scale_m,
            horizontal_excess / plane_scale_m,
            (float(hit["frame"]) - float(supplied_frame)) / 0.25,
        ]
    )
