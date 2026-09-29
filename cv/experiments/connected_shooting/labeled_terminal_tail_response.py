"""Input-qualified terminal-net response without a fabricated ground epoch.

Reuse the shared physical model, full native exposure objective and bounded
constrained optimizer. Only the final launch and outgoing net velocity vary;
original contacts, observations, intervals and earlier flights stay fixed.
This net-present hypothesis does not resolve ambiguous event occurrence.
"""

import time

import numpy as np

from cv.experiments.connected_shooting import event_constraints, full_native_continuation as full
from cv.experiments.connected_shooting import labeled_prefix_boundary as boundary
from cv.experiments.connected_shooting import labeled_terminal_net_tail as tail
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity
from cv.experiments.connected_shooting.labeled_passive_tape import using_response
from cv.experiments.connected_shooting.labeled_prefix_joint_impact import _jacobian


def qualify(context):
    scene = context["scene"]
    if tail.interval(scene) is None or len(context["bounces"][-1]):
        raise ValueError("original terminal interaction without supplied ground required")
    if sum(len(rows) for rows in scene.net_hit_frames) != 1 or len(scene.net_hit_frames[-1]) != 1:
        raise ValueError("single original terminal net hypothesis required")
    return dict(terminal_tail=True, ground_events=[], terminal_flight=len(scene.pixels) - 1)


def fit(context, source, duration, *, maxiter, seconds):
    """Retain the input-objective minimum within the original interaction domain."""
    qualify(context)
    if type(maxiter) is not int or maxiter < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive shared operation and wall budgets required")
    started = time.monotonic()
    scene, axes, added = full.merge_scene(
        context["scene"], context["heldout"], context["bounces"], context["axes"]
    )
    source = np.asarray(source, float)
    initial_chain = full.model.chain(scene, source)
    tail.require_event_domain(scene, initial_chain)
    tail.require_mesh_response(scene, initial_chain)
    hit = initial_chain[-1]["net_hits"][0]
    n, last = len(scene.pixels), len(scene.pixels) - 1
    selected = np.r_[
        np.arange(3 + 3 * last, 6 + 3 * last), np.arange(3 + 3 * n + 3 * last, 6 + 3 * n + 3 * last)
    ]
    scale = np.array([30.0] * 3 + [3.0] * 3 + [15.0] * 3)
    lo = np.array([-75.0] * 3 + [-6.0] * 3 + [-75.0] * 3) / scale
    hi = -lo
    initial = np.r_[source[selected], hit["v_out"]] / scale
    if np.any(initial < lo) or np.any(initial > hi):
        raise ValueError("source outside shared numerical response box")
    target = np.concatenate(scene.pixels)
    cache = FlightCache(entries_per_flight=64)

    def evaluate(q):
        values = np.asarray(q) * scale
        p = source.copy()
        p[selected] = values[:6]
        response = FreeNetVelocity(values[6:])
        with using_response(response):
            chain = full.model.chain(scene, p, simulation_cache=cache)
            slacks = tail.original_domain_slacks_frames(scene, chain)
            mesh_slacks = tail.mesh_response_slacks_m(scene, chain)
            _, physics, _ = event_constraints.evaluate(
                scene,
                p,
                context["bounces"],
                2.0,
                simulation_cache=cache,
                net_clearance_scale_m=0.1,
                terminal_rebound_frames=context.get("terminal_rebound_frames"),
            )
            image = (
                full.exposure.prediction(
                    scene, p, axes, duration, cache, termination_kind=context["termination_kind"]
                )
                - target
            )
        hits = chain[-1].get("net_hits", [])
        incoming = np.linalg.norm(hits[0]["v_in"]) if len(hits) == 1 else 0.0
        prior = np.r_[
            (p[selected] - source[selected]) / np.array([30.0] * 3 + [6.0] * 3),
            max(np.linalg.norm(values[6:]) - incoming, 0.0) / 10.0,
        ]
        r = np.r_[image.ravel(), physics, prior]
        return r, dict(
            boundary_equality_m=np.empty(0),
            clearance_slacks_m=mesh_slacks,
            input_event_slacks_frames=slacks,
            input_event_tolerance_frames=tail.EVENT_TOLERANCE_FRAMES,
            input_domain_inequalities=np.r_[
                mesh_slacks, slacks + tail.EVENT_TOLERANCE_FRAMES - 1e-5
            ],
            parameters=p,
            response=dict(outgoing_velocity_mps=values[6:].tolist()),
            all_native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
        )

    state, termination = boundary.solve(
        evaluate,
        initial,
        lo,
        hi,
        jacobian=_jacobian,
        maxiter=maxiter,
        seconds=seconds,
        started=started,
        inequality_key="input_domain_inequalities",
    )
    best = state["best_receipt"]
    response = FreeNetVelocity(best["response"]["outgoing_velocity_mps"])
    # Final replay is uncached, under exactly the response exported with parameters.
    with using_response(response):
        replay = full.model.chain(scene, best["parameters"])
        slacks = tail.require_event_domain(scene, replay)
        mesh_slacks = tail.require_mesh_response(scene, replay)
    return dict(
        best=dict(
            parameters=best["parameters"].tolist(),
            response=best["response"],
            cost=state["best_cost"],
            all_native_rms_px=best["all_native_rms_px"],
        ),
        initial_source=source.tolist(),
        initial_response=dict(outgoing_velocity_mps=hit["v_out"]),
        termination=termination,
        calls=state["calls"],
        invalid_trials=state["invalid"],
        feasible_trials=state["feasible_trials"],
        input_event_slacks_frames=slacks.tolist(),
        event_comparison_tolerance_frames=tail.EVENT_TOLERANCE_FRAMES,
        mesh_response_slacks_m=mesh_slacks.tolist(),
        near_cord_geometric_uncertainty_m=0.03,
        full_native_rows_added=added,
        elapsed_seconds=time.monotonic() - started,
        unforced_replay_verified=True,
        occurrence_inferred=False,
        net_present="assumed",
        net_absent="not_evaluated",
        objective="unchanged quadratic native images + event physics + broad launch/response priors",
    )
