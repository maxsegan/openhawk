"""Optional connected preceding-ground / terminal-net contact refinement.

Accepts only the current invocation's observed context, vector and response
registries. The complete original scene and clock survive. A single quadratic
input objective ranks candidates; acceptance gates and saved fits are not read.
Call ``fit(..., enabled=True)`` explicitly after the interior sweep. Default OFF.
The final passive epoch stays fixed while the shared racket contact XYZ may move.
"""

from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
from copy import deepcopy
import time
import numpy as np
from scipy.optimize import least_squares
from cv.experiments.connected_shooting import (
    model,
    labeled_interior_ground_response as ground,
    labeled_net_recipe as net,
    labeled_interior_normal as interior,
    labeled_net_free_response as free,
    labeled_passive_tape as tape,
    measured_dynamics as dynamics,
    labeled_net_normal_response as normal,
    labeled_net_height_chart as height,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache

NUMERICAL_ERRORS = interior.NUMERICAL_ERRORS


def qualify(context, duration):
    """Require original racket boundaries; a terminal passive endpoint is allowed."""
    scene = context["scene"]
    rows = interior._flight_rows(context, duration)
    n = len(rows)
    if n < 3 or rows[-2]["reason"]:
        raise ValueError("supported original preceding non-serve ground flight required")
    eligibility = net.qualify(context)
    contacts = [e for e in context["events"] if e["event_type"] == "contact"]
    for epoch in scene.contact_frames[-3:-1]:
        matches = [e for e in contacts if abs(float(e["frame"]) - float(epoch)) < 1e-8]
        if len(matches) != 1:
            raise ValueError("preceding start and shared boundary must be original racket contacts")
        lo, hi = map(float, matches[0]["frame_interval"])
        if not lo <= epoch <= hi:
            raise ValueError("original racket epoch must remain inside its own interval")
    first, shared, end = map(float, scene.contact_frames[-3:])
    if any(
        first < float(e["frame"]) <= end and abs(float(e["frame"]) - shared) > 1e-8
        for e in contacts
    ):
        raise ValueError("no additional racket contact within coupled terminal pair")
    for interval in (rows[-2]["interval"], eligibility["net"]["frame_interval"]):
        if not float(interval[0]) < float(interval[1]):
            raise ValueError("positive-width original ground and net intervals required")
    if any(float(e["frame_interval"][1]) > end for e in eligibility["ground_events"]):
        raise ValueError("original terminal ground interval extends beyond passive endpoint")
    return dict(preceding=deepcopy(rows[-2]), terminal=deepcopy(eligibility))


def _candidate_topology(chain, spec):
    """Keep original impact membership and intervals; never add/drop a ground."""
    preceding, terminal = chain[-2:]
    if preceding["net_hits"] or len(preceding["bounces"]) != 1:
        raise ValueError("preceding flight left its original ground-only branch")
    checks = [(preceding["bounces"][0], spec["preceding"]["interval"])]
    if len(terminal["net_hits"]) != 1:
        raise ValueError("terminal flight must retain exactly one original net")
    checks.append((terminal["net_hits"][0], spec["terminal"]["net"]["frame_interval"]))
    events = spec["terminal"]["ground_events"]
    if len(terminal["bounces"]) != len(events):
        raise ValueError("terminal ground count differs from original inventory")
    checks.extend(
        (b, e["frame_interval"]) for b, e in zip(terminal["bounces"], events, strict=True)
    )
    for impact, (lo, hi) in checks:
        epoch = float(impact["frame"])
        if not np.isfinite(epoch) or epoch < lo - 1e-7 or epoch > hi + 1e-7:
            raise ValueError("candidate impact outside original event interval")


def _preceding_endpoint(scene, parameters, flight, start_xyz):
    """Propagate only physical preceding slots, never unexpanded terminal slots.

    The active complete ground registry applies by the ORIGINAL native contact
    epoch. No truncated scene, rebased clock or manually fitted contact is used.
    """
    n = len(scene.pixels)
    v, w = 3 + 3 * flight, 3 + 3 * n + 3 * flight
    theta = np.r_[start_xyz, parameters[v : v + 3], parameters[w : w + 3]]
    first, last = scene.contact_frames[flight : flight + 2]
    xyz, _, _, impacts = dynamics.simulate(
        theta,
        float(first),
        np.array([first, last]),
        scene.fps,
        scene.surface,
        bounce_profile=scene.bounce_profile,
        rebound_scales=parameters[-2:],
    )
    if len(impacts) != 1:
        raise ValueError("preceding flight left original single-ground branch")
    return xyz[-1].copy()


def fit(
    context,
    source,
    duration,
    *,
    ground_response=None,
    net_response=None,
    enabled=False,
    seconds=240.0,
    maxiter=120,
):
    """Return unchanged context and a complete current-state refinement receipt.

    Caller owns the existing collision-policy scope. This helper owns its ground
    and net response scopes; callers must not wrap it in an active net response.
    Unsupported topology, numerical failures and exhausted budget retain the exact
    incumbent. Failed initialization is disclosed, never counted as improvement.
    """
    started = time.monotonic()
    if not np.isfinite(seconds) or seconds <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError("positive common budgets required")
    source = np.asarray(source, float).copy()
    n = len(context["scene"].pixels)
    if source.shape != (5 + 6 * n,) or not np.isfinite(source).all():
        raise ValueError("finite full single-shooting source vector required")
    registry = ground.normalize(deepcopy(ground_response), context["scene"])
    initial_net = deepcopy(net_response)
    details = dict(
        full_vector=source.tolist(),
        initial_source=source.tolist(),
        ground_response=deepcopy(registry),
        initial_ground_response=deepcopy(registry),
        net_response=deepcopy(initial_net),
        initial_net_response=deepcopy(initial_net),
        status="disabled" if not enabled else "source_retained",
        gate_selection=False,
        external_fitted_inputs_used=False,
        policy=dict(
            enabled=bool(enabled),
            seconds=float(seconds),
            maxiter=maxiter,
            objective="all native image + physical residuals + weak response/source priors",
            contact_epochs_fixed=True,
            prefix_fixed=True,
            native_rows_unchanged=True,
        ),
    )
    if not enabled:
        details["wall_seconds"] = time.monotonic() - started
        return context, details
    try:
        spec = qualify(context, duration)
        if initial_net is not None:
            law = free.response_from_record(initial_net)
            underlying = getattr(law, "net_response", law)
            if hasattr(underlying, "outgoing_spin_rad_s"):
                raise ValueError(
                    "terminal coupling preserves nominal net spin; supplied spin route unsupported"
                )
        details["inventory"] = spec
        result = _fit(
            context,
            source,
            duration,
            registry,
            initial_net,
            seconds=seconds,
            maxiter=maxiter,
            start_time=started,
            spec=spec,
        )
        best = result["best"]
        details.update(
            status=best["status"],
            full_vector=np.asarray(best["parameters"]).tolist(),
            ground_response=deepcopy(best["ground_response"]),
            net_response=deepcopy(best["net_response"]),
            fit=result,
        )
    except NUMERICAL_ERRORS + (TimeoutError, IndexError) as error:
        details["reason"] = f"{type(error).__name__}: {error}"
    details["wall_seconds"] = time.monotonic() - started
    return context, details


def _fit(ctx, source, duration, source_ground, source_net, *, seconds, maxiter, start_time, spec):
    eligibility = spec["terminal"]
    checks, consumed = net.original.followup.fit_check_copy(ctx["scene"], ctx["heldout"])
    scene, axes, added = net.epoch.full.merge_scene(
        ctx["scene"], checks, ctx["bounces"], ctx["axes"]
    )
    n = len(scene.pixels)
    a = n - 2
    b = n - 1
    rows = interior._flight_rows(ctx, duration)
    if n < 3 or rows[a]["reason"]:
        raise ValueError("unsupported original preceding ground flight")
    normal_prior = normal.prior_from_observations(ctx, source)
    source = np.asarray(source, float)
    registry = ground.normalize(source_ground, scene)
    law = free.response_from_record(source_net) if source_net else None

    def laws(record, response):
        s = ExitStack()
        s.enter_context(ground.using_response(record, scene))
        if response:
            s.enter_context(tape.using_response(response))
        return s

    with laws(registry, law):
        original = model.chain(scene, source)
    projector = interior.BlockProjector(scene, source, a, [rows[a]], original)
    old_bounce = original[a]["bounces"][0]
    ground_seed = float(np.clip(old_bounce["frame"], *rows[a]["interval"]))
    e_seed = old_bounce["applied_restitution"]
    hit = original[b]["net_hits"][0]
    net_seed = float(np.clip(hit["frame"], *eligibility["net"]["frame_interval"]))
    fraction = float(
        np.clip((hit["x"][2] - model.R_BALL) / height.net_tape_height(hit["x"][0]), 1e-5, 1)
    )
    v0, v1 = 3 + 3 * a, 3 + 3 * b
    w0, w1 = 3 + 3 * n + 3 * a, 3 + 3 * n + 3 * b
    outgoing = np.array((source_net or {}).get("outgoing_velocity_mps", hit["v_out"]))
    record = source_net or {}
    post_normal = (
        record["first_ground_normal_restitution"]
        if "first_ground_normal_restitution" in record
        else original[b]["bounces"][0]["applied_restitution"]
    )
    q = np.r_[
        source[v0 : v0 + 2],
        ground_seed,
        source[w0 : w0 + 3],
        e_seed,
        source[v1],
        net_seed,
        fraction,
        source[w1 : w1 + 3],
        outgoing,
        post_normal,
        record.get("first_ground_tangential_multiplier", 1.0),
        record.get("first_ground_transverse_velocity_mps", 0.0),
    ]
    lo = np.r_[
        [-75, -75, rows[a]["interval"][0]],
        [-6] * 3,
        0.05,
        -75,
        eligibility["net"]["frame_interval"][0],
        1e-5,
        [-6] * 3,
        [-75] * 3,
        0.05,
        0,
        -20,
    ]
    hi = np.r_[
        [75, 75, rows[a]["interval"][1]],
        [6] * 3,
        1.0,
        75,
        eligibility["net"]["frame_interval"][1],
        1.0,
        [6] * 3,
        [75] * 3,
        1.0,
        4,
        20,
    ]
    scale = np.r_[30, 30, 1, [3] * 3, 1, 30, 1, 1, [3] * 3, [15] * 3, 1, 1, 5]
    q = np.clip(q, lo + 1e-9, hi - 1e-9)
    cache = FlightCache(entries_per_flight=64)
    target = np.concatenate(scene.pixels)
    selected = np.r_[
        np.arange(v0, v0 + 3), np.arange(v1, v1 + 3), np.arange(w0, w0 + 3), np.arange(w1, w1 + 3)
    ]
    calls = 0
    feasible_calls = 0
    bad = Counter()
    best = None
    size = None
    active_e = None

    def expand(x):
        nonlocal active_e
        z = x * scale
        p = source.copy()
        p[v0 : v0 + 2] = z[:2]
        p[w0 : w0 + 3] = z[3:6]
        p[v1] = z[7]
        p[w1 : w1 + 3] = z[10:13]
        record = ground.merge(
            registry,
            [
                dict(
                    flight_index=a,
                    native_contact_epoch=float(scene.contact_frames[a]),
                    ground_ordinal=0,
                    normal_restitution=float(z[6]),
                )
            ],
            scene,
        )
        response = normal.NetNormalResponse(
            free.FreeNetVelocity(z[13:16]),
            z[16],
            horizontal_mode="heading",
            tangential_multiplier=z[17],
            transverse_velocity_mps=z[18],
        )
        if active_e != float(z[6]):
            projector.clear()
            active_e = float(z[6])
        with laws(record, None):
            p, roots = projector.project(p, [z[2]])
            endpoint = _preceding_endpoint(scene, p, a, original[a]["start_xyz"])
            last = scene.contact_frames[b]
            theta = np.r_[endpoint, p[v1 : v1 + 3], p[w1 : w1 + 3]]
            theta, net_receipt = height.project_net_height(
                theta,
                float(last),
                z[8],
                scene.fps,
                scene.surface,
                mesh_fraction=z[9],
                bounce_profile=scene.bounce_profile,
                rebound_scales=p[-2:],
            )
            p[v1 : v1 + 3] = theta[3:6]
        return p, record, response, dict(ground=roots, net=net_receipt)

    def evaluate(p, record, response):
        with laws(record, response):
            chain = model.chain(scene, p, simulation_cache=cache)
            _, physics, _ = net.epoch.interval.block.event_constraints.evaluate(
                scene,
                p,
                ctx["bounces"],
                2.0,
                simulation_cache=cache,
                net_clearance_scale_m=0.1,
                terminal_rebound_frames=ctx.get("terminal_rebound_frames"),
            )
            image = (
                net.epoch.full.exposure.prediction(
                    scene, p, axes, duration, cache, termination_kind=ctx["termination_kind"]
                )
                - target
            )
        prior = (p[selected] - source[selected]) / np.r_[[30.0] * 6, [6.0] * 6]
        bounce = chain[a]["bounces"][0]
        nominal = bounce.get("nominal_applied_restitution", bounce["applied_restitution"])
        prior = np.r_[prior, 4 * (bounce["applied_restitution"] - nominal) / 0.2]
        h = chain[b]["net_hits"][0]
        prior = np.r_[prior, max(np.linalg.norm(h["v_out"]) - np.linalg.norm(h["v_in"]), 0.0) / 10]
        if response and hasattr(response, "first_ground_normal_restitution"):
            prior = np.r_[
                prior,
                (response.first_ground_normal_restitution - normal_prior["center"])
                / normal_prior["sigma"],
                (response.tangential_multiplier - 1) / 0.5,
                response.transverse_velocity_mps / 5,
            ]
        else:
            post = chain[b]["bounces"][0]
            prior = np.r_[
                prior,
                (post["applied_restitution"] - normal_prior["center"]) / normal_prior["sigma"],
                0.0,
                0.0,
            ]
        residual = np.r_[image.ravel(), physics, prior]
        detail = dict(
            cost=float(0.5 * (residual @ residual)),
            image_sse=float(np.sum(image**2)),
            physics_sse=float(physics @ physics),
            prior_sse=float(prior @ prior),
            all_native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
            preceding_normal_restitution=float(bounce["applied_restitution"]),
            preceding_normal_at_bound=bool(
                min(abs(bounce["applied_restitution"] - edge) for edge in (0.05, 1.0)) < 1e-6
            ),
        )
        return residual, detail, chain

    with net.continuous_ground(ctx):
        sr, source_metrics, source_chain = evaluate(source, registry, law)
        if not np.isfinite(sr).all():
            raise ValueError("source objective is nonfinite")
        size = len(sr)
        best = dict(
            parameters=source.copy(),
            ground_response=registry,
            net_response=source_net,
            metrics=source_metrics,
            status="source_retained",
            chart=None,
        )

        def residual(x):
            nonlocal calls, feasible_calls, best
            if time.monotonic() - start_time > seconds:
                raise TimeoutError("common terminal coupling wall budget exhausted")
            calls += 1
            try:
                p, record, response, chart = expand(x)
                rr, metrics, chain = evaluate(p, record, response)
                _candidate_topology(chain, spec)
                if len(rr) != size or not np.isfinite(rr).all():
                    raise ValueError("candidate residual must retain finite source objective shape")
                feasible_calls += 1
                if metrics["cost"] < best["metrics"]["cost"] * (1 - 1e-9):
                    best = dict(
                        parameters=p.copy(),
                        ground_response=record,
                        net_response=response.record(
                            chain[b]["net_hits"][0]["v_in"], chain[b]["net_hits"][0]["v_out"]
                        ),
                        metrics=metrics,
                        status="improved",
                        chart=chart,
                    )
                return rr
            except NUMERICAL_ERRORS as error:
                bad[str(error)] += 1
                return np.full(size, 1e5)

        def jac(x):
            cols = []
            for j in range(len(x)):
                low, high = x.copy(), x.copy()
                low[j] = max(lo[j] / scale[j], x[j] - 1e-6)
                high[j] = min(hi[j] / scale[j], x[j] + 1e-6)
                cols.append((residual(high) - residual(low)) / (high[j] - low[j]))
            return np.column_stack(cols)

        try:
            solved = least_squares(
                residual,
                q / scale,
                jac=jac,
                bounds=(lo / scale, hi / scale),
                max_nfev=maxiter,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            termination = dict(
                success=bool(solved.success),
                message=solved.message,
                nfev=solved.nfev,
                njev=solved.njev,
            )
        except NUMERICAL_ERRORS + (TimeoutError,) as error:
            termination = dict(success=False, message=f"{type(error).__name__}: {error}")
    return dict(
        best=best,
        source_metrics=source_metrics,
        seconds=time.monotonic() - start_time,
        calls=calls,
        feasible_calls=feasible_calls,
        invalid_reasons=dict(bad),
        termination=termination,
        policy=dict(
            maxiter=maxiter,
            seconds=seconds,
            changed_flights=[a, b],
            ground_chart=True,
            one_direct_preceding_normal=True,
            original_ground_interval=rows[a]["interval"],
            original_net_interval=eligibility["net"]["frame_interval"],
            contact_epochs_fixed=True,
            prefix_fixed=True,
            passive_endpoint_fixed_epoch_only=True,
            native_rows_unchanged=True,
            source_incumbent=True,
            input_quadratic_cost=True,
            gate_ranking=False,
            external_fitted_inputs_used=False,
        ),
        consumed=consumed,
    )
