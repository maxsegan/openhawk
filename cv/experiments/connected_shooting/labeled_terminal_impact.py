"""Optional original-interval terminal refinement of an in-memory connected point.

Question: can a first-impact coordinate fix a late terminal bounce without moving
an already fitted prefix? No saved fitted inputs, event edits, ground-XYZ targets,
new response law, or gate-selected rollback. The first version supports one
original terminal ground event and real observations on both sides of it.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import observation_operator

from contextlib import nullcontext
from copy import deepcopy
import time

import numpy as np

from cv.experiments.connected_shooting import (
    interior_contact_epochs,
    labeled_prefix_boundary as boundary,
    labeled_preparation_net_followup as followup,
    labeled_prefix_joint_impact as prefix,
    net_constraints,
)
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response


def qualify(context, duration):
    scene = context["scene"]
    scene.validate()
    context["heldout"].validate()
    if (
        len(scene.pixels) < 2
        or scene.parameterization != "single_shooting"
        or scene.dynamics != "measured_240hz"
        or scene.rebound_mode != "point_scales"
        or scene.bounce_regime_override is not None
    ):
        raise ValueError("measured connected point-scale final non-serve flight required")
    span = observation_operator.support_span(duration)
    last = len(scene.pixels) - 1
    start, end = map(float, scene.contact_frames[-2:])
    if not np.array_equal(scene.contact_frames, context["heldout"].contact_frames):
        raise ValueError("training and withheld contact topology disagree")
    original, source_events = interior_contact_epochs.original_contact_inventory(context, duration)
    events = [e for e in source_events if original[-2] < float(e["frame"]) <= original[-1]]
    if any(e["event_type"] == "net_hit" for e in events) or any(
        s.net_hit_frames is not None and len(s.net_hit_frames[-1])
        for s in (scene, context["heldout"])
    ):
        raise ValueError("declared terminal net belongs to the separate net adapter")
    if any(e["event_type"] == "contact" for e in events):
        raise ValueError("terminal context contains another physical contact")
    grounds = [e for e in events if e["event_type"] == "bounce"]
    if len(grounds) != 1:
        raise ValueError(
            "first version requires exactly one original terminal ground; no ordinal discarded"
        )
    event = grounds[0]
    lo, hi = prefix._interval(event["frame_interval"], "terminal ground")
    if not start < lo < hi < end:
        raise ValueError("original ground interval must be inside supported terminal context")
    original = np.asarray(context["bounces"][-1], float)
    if original.shape != (1,) or abs(original[0] - float(event["frame"])) > 1e-8:
        raise ValueError("original event and physical bounce inventory disagree")
    frames = np.unique(
        np.r_[scene.observation_frames[-1], context["heldout"].observation_frames[-1]]
    )
    before = frames[frames + span < lo]
    after = frames[frames > hi]
    if len(before) < 2 or len(after) < 2:
        raise ValueError("two original native exposures on each impact wing required")
    return dict(
        flight=last,
        interval=[lo, hi],
        initial_epoch=float(np.clip(event["frame"], lo, hi)),
        original_event=deepcopy(event),
        incoming_frames=before.tolist(),
        rebound_frames=after.tolist(),
    )


def fit(context, source, duration, *, max_nfev=80, seconds=300.0, net_response=None):
    """Return the same context and a full connected vector with final Vz charted."""
    started = time.monotonic()
    spec = qualify(context, duration)
    if type(max_nfev) is not int or max_nfev < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive optimizer budgets required")
    last = spec["flight"]
    n = last + 1
    src = np.asarray(source, float).copy()
    if src.shape != (5 + 6 * n,) or not np.isfinite(src).all():
        raise ValueError("full finite connected source vector required")
    checks, _ = followup.fit_check_copy(context["scene"], context["heldout"])
    # merge_scene is the existing duplicate-safe full-native operator; no native
    # row is removed from context or scoring by the local objective selection.
    scene, axes, added = prefix.full.merge_scene(
        context["scene"], checks, context["bounces"], context["axes"]
    )
    queries = net_constraints.dense_queries(scene, context["bounces"])
    projector = prefix.ConnectedFollowingProjector(scene, last, spec["interval"], src[5 + 3 * last])
    velocity = 3 + 3 * last
    spin = 3 + 3 * n + 3 * last
    q0 = np.r_[src[velocity : velocity + 2], spec["initial_epoch"], src[spin : spin + 3]]
    lo = np.array([-75.0, -75.0, spec["interval"][0], -6.0, -6.0, -6.0])
    hi = np.array([75.0, 75.0, spec["interval"][1], 6.0, 6.0, 6.0])
    if np.any(q0 < lo) or np.any(q0 > hi):
        raise ValueError("source terminal launch outside original bounds")
    offset = sum(len(p) for p in scene.pixels[:-1])
    response = response_from_record(net_response) if net_response is not None else None

    def physical_scope():
        return using_response(response) if response is not None else nullcontext()

    with physical_scope():
        original_chain = prefix.full.model.chain(scene, src, query_frames=queries)

    def evaluate(q):
        p = src.copy()
        p[velocity : velocity + 2] = q[:2]
        p[spin : spin + 3] = q[3:]
        with physical_scope():
            p, root = projector.project(p, float(q[2]))
            chain = prefix.full.model.chain(scene, p, query_frames=queries)
            predicted = np.asarray(
                prefix.full.exposure.prediction(
                    scene, p, axes, duration, termination_kind=context["termination_kind"]
                )
            )
        image = predicted[offset:] - scene.pixels[-1]
        eq, slack = boundary.constraints(chain[-1:], queries[-1:], 1, [[]], None)
        residual = np.r_[
            image.ravel(), (q[3:] - src[spin : spin + 3]) / 6.0, np.minimum(slack, 0) / 0.025
        ]
        return residual, dict(
            p=p,
            chain=chain,
            root=root,
            boundary_equality_m=eq,
            clearance_slacks_m=slack,
            native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
        )

    initial_residual, initial = evaluate(q0)
    state, termination = boundary.solve(
        evaluate,
        q0,
        lo,
        hi,
        jacobian=prefix._jacobian,
        maxiter=max_nfev,
        seconds=seconds,
        started=started,
    )
    best = state["best_receipt"]
    p = best["p"]
    epoch = float(state["best_q"][2])
    with physical_scope():
        replayed = prefix.full.model.chain(scene, p, query_frames=queries)
    impact = float(replayed[-1]["bounces"][0]["frame"])
    if abs(impact - epoch) > 1e-7:
        raise ValueError("original-law replay does not reproduce terminal impact chart")
    _, slack = boundary.constraints(replayed[-1:], queries[-1:], 1, [[]], None)
    if np.min(slack, initial=1) < -boundary.TOLERANCE_M:
        raise ValueError("original terminal replay violates net feasibility")
    prefix_error = max(
        (
            float(np.max(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"]))))
            for a, b in zip(replayed[:-1], original_chain[:-1])
        ),
        default=0.0,
    )
    if prefix_error != 0.0:
        raise ValueError("terminal fit changed earlier physical positions")
    frozen = np.ones(len(src), bool)
    frozen[velocity : velocity + 3] = False
    frozen[spin : spin + 3] = False
    if not np.array_equal(p[frozen], src[frozen]):
        raise ValueError("terminal fit changed a frozen parameter")
    return context, dict(
        automatic_inference_eligible=False,
        full_vector=p.tolist(),
        initial_source=src.tolist(),
        q=state["best_q"].tolist(),
        initial_q=q0.tolist(),
        cost=state["best_cost"],
        initial_chart_cost=float(initial_residual @ initial_residual),
        native_rms_px=best["native_rms_px"],
        initial_chart_native_rms_px=initial["native_rms_px"],
        termination=termination,
        calls=state["calls"],
        invalid_trials=state["invalid"],
        original_interval=spec,
        impact_epoch=epoch,
        unforced_impact_epoch=impact,
        root=best["root"],
        prefix_max_position_error_m=prefix_error,
        clearance_slacks_m=slack.tolist(),
        retained_net_response=deepcopy(net_response),
        fit_policy=dict(
            mode="original terminal first-impact interval",
            objective="all original terminal/native rebound context pixels, weak spin stay-close, physical nonnet clearance",
            new_fitted_degrees_of_freedom=0,
            ground_xyz_target_used=False,
            gate_selection=False,
            fixed_prefix=True,
            fixed_rebound_scalars=src[-2:].tolist(),
            duration=duration,
            max_nfev=max_nfev,
            seconds=seconds,
            optimizer="SLSQP absolute central FD1e-6; input-only feasible selection",
            added_native_rows=added,
        ),
    )
