"""Terminal first-ground normal response on the current connected state.

Generalizes the original-interval terminal adapter: the final non-serve flight's
first-ground normal restitution may move inside [0.05, 1] through the existing
persistent ground response registry, and one or two ORIGINAL terminal grounds are
supported with every ordinal represented. The first impact is charted on its
original interval; a second impact must occur inside its original interval; a
missing original impact is invalid. Subsequent simulated grounds may continue
through the original native tail, without creating observed events or dropping pixels.
Fixed-normal and free-normal arms share one objective, context and budgets. With
``second_normal``, exactly two original grounds and two distinct native exposures
after the second interval, the second original ground's
normal restitution is one more regularized coordinate on the same registry
(ordinal 1), with its own soft prior in the source and candidate objectives; one
original ground leaves that option inactive. No case key, file, cached fit,
gate, acceptance bit or XYZ ground target is read.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import observation_operator

from contextlib import ExitStack
from copy import deepcopy
import time

import numpy as np

from cv.experiments.connected_shooting import (
    interior_contact_epochs,
    labeled_interior_ground_response as ground,
    labeled_prefix_boundary as boundary,
    labeled_prefix_joint_impact as prefix,
    labeled_preparation_net_followup as followup,
    net_constraints,
    regime_recovery,
)
from cv.experiments.connected_shooting.labeled_interior_normal import (
    NUMERICAL_ERRORS,
    _feasible as _boundary_feasible,
    _restore,
)
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response
from physics import bounce_reference

NORMAL_BOUNDS = (0.05, 1.0)
NORMAL_SIGMA_FLOOR = 0.15
SECOND_REBOUND_MIN_EXPOSURES = 2
SPIN_PRIOR_SCALE = 6.0
CLEARANCE_SCALE_M = 0.025
EPOCH_TOLERANCE_FRAMES = 1e-7
INCUMBENT_RELATIVE_TOLERANCE = 1e-9
BOUND_TOLERANCE = 1e-9
COORDINATES = ("vx_mps", "vy_mps", "first_impact_frame", "spin_0", "spin_1", "spin_2")
FIRST_NORMAL = "first_ground_normal_restitution"
SECOND_NORMAL = "second_ground_normal_restitution"
OBJECTIVE = dict(
    image="every native final-flight pixel (training and merged check rows), px",
    spin="(spin - source spin) / 6 per launch spin coordinate",
    clearance="min(net clearance slack, 0) / 0.025 m on the final flight",
    normal="(applied first-ground restitution - e0) / sigma_e, dimensionless",
    normal_center="nominal law at the source terminal first impact x point vertical scale, clipped",
    normal_sigma="max(0.15, surface restitution residual std at model spin x vertical scale)",
    prior_is_calibrated_uncertainty=False,
    ground_xyz_target_used=False,
    gate_selection=False,
    selection="lowest input objective among feasible trials; source incumbent when representable",
)
SECOND_OBJECTIVE = dict(
    second_normal="(applied second-original-ground restitution - e0_2) / sigma_e2, dimensionless",
    second_normal_center="nominal law at the source terminal second impact x point vertical scale, clipped",
    second_normal_sigma="max(0.15, surface restitution residual std at model spin x vertical scale)",
    second_normal_in_source_objective=True,
    second_ground_xyz_target_used=False,
)


def _feasible(receipt):
    """Keep spatial clearance and original impact epochs in their own units."""
    return bool(
        _boundary_feasible(receipt)
        and np.isfinite(receipt["impact_slacks_frames"]).all()
        and np.min(receipt["impact_slacks_frames"], initial=1) >= -EPOCH_TOLERANCE_FRAMES
    )


def _impact_slacks(impacts, intervals):
    if len(impacts) != len(intervals):
        raise ValueError("impact count differs from original grounds")
    return np.asarray(
        [
            slack
            for impact, (lo, hi) in zip(impacts, intervals, strict=True)
            for slack in (float(impact["frame"]) - lo, hi - float(impact["frame"]))
        ]
    )


def _original_impacts(impacts, spec):
    """Match every original ordinal; later physics remains unlabelled continuation.

    Exact total counts create a disconnected optimizer domain when a poor normal
    response brings the next bounce into the observed tail. Its image residuals,
    including every original rebound picture, must remain available to move it out.
    This does not authorize skipping an early impact to match a later event.
    """
    count = spec["ground_count"]
    if len(impacts) < count:
        raise ValueError("terminal flight is missing an original ground ordinal")
    extra = impacts[count:]
    if extra and not spec["continuation"]["enabled"]:
        raise ValueError("extra ground requires original terminal continuation observations")
    if any(
        not spec["intervals"][-1][1]
        < float(b["frame"])
        <= spec["continuation"]["end_frame"] + EPOCH_TOLERANCE_FRAMES
        for b in extra
    ):
        raise ValueError("extra ground is not after all original intervals inside terminal context")
    return impacts[:count]


def qualify(context, duration):
    """Original topology only: one or two terminal grounds, no nets or contacts inside."""
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
    grounds = sorted(
        (e for e in events if e["event_type"] == "bounce"), key=lambda e: float(e["frame"])
    )
    if len(grounds) not in (1, 2):
        raise ValueError(
            f"one or two original terminal grounds required, found {len(grounds)}; "
            "no ordinal discarded"
        )
    intervals = [prefix._interval(e["frame_interval"], "terminal ground") for e in grounds]
    edges = [start, *(v for lo_hi in intervals for v in lo_hi), end]
    if any(a >= b for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("original ground intervals must be ordered inside the terminal context")
    original = np.asarray(context["bounces"][-1], float)
    if original.shape != (len(grounds),) or np.any(
        np.abs(original - np.asarray([float(e["frame"]) for e in grounds])) > 1e-8
    ):
        raise ValueError("original events and physical bounce inventory disagree")
    frames = np.unique(
        np.r_[scene.observation_frames[-1], context["heldout"].observation_frames[-1]]
    )
    before = frames[frames + span < intervals[0][0]]
    after = frames[frames > intervals[0][1]]
    if len(before) < 2 or len(after) < 2:
        raise ValueError("two original native exposures on each first-impact wing required")
    wings = dict(incoming_frames=before.tolist(), rebound_frames=after.tolist())
    if len(grounds) == 2:
        wings["between_ground_frames"] = after[after + span < intervals[1][0]].tolist()
        wings["second_rebound_frames"] = frames[frames > intervals[1][1]].tolist()
    return dict(
        flight=last,
        ground_count=len(grounds),
        intervals=[list(v) for v in intervals],
        interval=list(intervals[0]),
        event_epochs=[float(e["frame"]) for e in grounds],
        initial_epoch=float(np.clip(grounds[0]["frame"], *intervals[0])),
        original_events=deepcopy(grounds),
        continuation=dict(
            enabled=context.get("termination_kind") in {"terminal_bounce", "second_bounce"}
            and bool(np.any(frames > intervals[-1][1])),
            after_frame=float(intervals[-1][1]),
            end_frame=end,
            native_frames=frames[frames > intervals[-1][1]].tolist(),
            policy="later simulated grounds are physics, not additional observed events",
        ),
        **wings,
    )


def _scope(net, record, scene):
    stack = ExitStack()
    if net is not None:
        stack.enter_context(using_response(net))
    stack.enter_context(ground.using_response(record, scene))
    return stack


def _impact_row(b):
    return dict(
        frame=float(b["frame"]),
        xy_m=np.asarray(b["x"], float)[:2].tolist(),
        v_in_mps=np.asarray(b["v_in"], float).tolist(),
        v_out_mps=np.asarray(b["v_out"], float).tolist(),
        incidence_deg=float(
            np.degrees(np.arctan2(-b["v_in"][2], float(np.hypot(b["v_in"][0], b["v_in"][1]))))
        ),
        applied_restitution=float(b["applied_restitution"]),
        nominal_applied_restitution=float(
            b.get("nominal_applied_restitution", b["applied_restitution"])
        ),
        applied_horizontal_retention=float(b["applied_horizontal_retention"]),
        regime=b["regime"],
        coefficient_clipped=bool(b["coefficient_clipped"]),
    )


def fit(
    context,
    source,
    duration,
    *,
    ground_response=None,
    net_response=None,
    free_normal=True,
    second_normal=False,
    max_nfev=80,
    seconds=300.0,
):
    """Return the same context and one terminal candidate with explicit selection receipts."""
    started = time.monotonic()
    spec = qualify(context, duration)
    if type(free_normal) is not bool or type(second_normal) is not bool:
        raise ValueError("free_normal and second_normal must be explicit booleans")
    if second_normal and not free_normal:
        raise ValueError("second_normal requires the free first-ground normal arm")
    if type(max_nfev) is not int or max_nfev < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive optimizer budgets required")
    last, n = spec["flight"], spec["flight"] + 1
    src = np.asarray(source, float).copy()
    if src.shape != (5 + 6 * n,) or not np.isfinite(src).all():
        raise ValueError("full finite connected source vector required")
    initial_record = ground.normalize(deepcopy(ground_response), context["scene"])
    net = response_from_record(net_response) if net_response is not None else None
    checks, consumed = followup.fit_check_copy(context["scene"], context["heldout"])
    scene, axes, added = prefix.full.merge_scene(
        context["scene"], checks, context["bounces"], context["axes"]
    )
    queries = net_constraints.dense_queries(scene, context["bounces"])
    epoch = float(scene.contact_frames[last])
    velocity, spin = 3 + 3 * last, 3 + 3 * n + 3 * last
    offset = sum(len(p) for p in scene.pixels[:-1])
    fps = float(scene.fps)
    grounds = spec["ground_count"]
    second_support = dict(
        minimum_distinct_exposures=SECOND_REBOUND_MIN_EXPOSURES,
        native_frames=spec.get("second_rebound_frames", []),
        distinct_exposure_count=len(spec.get("second_rebound_frames", [])),
        original_second_interval=spec["intervals"][1] if grounds == 2 else None,
        policy="distinct original training/check exposures strictly after the second interval",
    )
    second_active = bool(
        second_normal
        and grounds == 2
        and second_support["distinct_exposure_count"] >= SECOND_REBOUND_MIN_EXPOSURES
    )
    existing, existing_second = (
        next(
            (
                r
                for r in (initial_record["routes"] if initial_record else [])
                if r["flight_index"] == last and r["ground_ordinal"] == ordinal
            ),
            None,
        )
        for ordinal in (0, 1)
    )

    # Source chain under the current complete registries; the prefix reference.
    with _scope(net, initial_record, scene):
        original_chain = prefix.full.model.chain(scene, src, query_frames=queries)
        source_prediction = np.asarray(
            prefix.full.exposure.prediction(
                scene, src, axes, duration, termination_kind=context["termination_kind"]
            )
        )
    source_impacts = original_chain[-1]["bounces"]
    scale = float(src[-2])
    surface = str(scene.surface)
    row = bounce_reference.MEASURED[surface]
    if source_impacts:
        b = source_impacts[0]
        nominal_center = float(b.get("nominal_applied_restitution", b["applied_restitution"]))
        center_origin = "nominal law at source terminal first impact x vertical scale"
    else:
        nominal_center = float(
            np.clip(
                bounce_reference.restitution(row["incidence_deg_median"], surface) * scale,
                *NORMAL_BOUNDS,
            )
        )
        center_origin = "source has no terminal impact; surface median incidence x vertical scale"
    e0 = float(np.clip(nominal_center, *NORMAL_BOUNDS))
    sigma_e = float(max(NORMAL_SIGMA_FLOOR, row["restitution_residual_std_at_model_spin"] * scale))
    prior = dict(
        center=e0,
        sigma=sigma_e,
        bounds=list(NORMAL_BOUNDS),
        center_origin=center_origin,
        vertical_scale=scale,
        surface=surface,
        prior_is_calibrated_uncertainty=False,
    )
    if second_active:
        # Same soft convention as the first ground: the surface law at the source's
        # own second impact, never an observed XY landing target. Uncertainty is the
        # same surface residual floor; it is disclosed, not calibrated.
        if len(source_impacts) >= 2:
            b = source_impacts[1]
            second_center = float(b.get("nominal_applied_restitution", b["applied_restitution"]))
            second_origin = "nominal law at source terminal second impact x vertical scale"
        else:
            second_center = float(
                np.clip(
                    bounce_reference.restitution(row["incidence_deg_median"], surface) * scale,
                    *NORMAL_BOUNDS,
                )
            )
            second_origin = (
                "source has no terminal second impact; surface median incidence x vertical scale"
            )
        e0_2 = float(np.clip(second_center, *NORMAL_BOUNDS))
        sigma_e2 = sigma_e
        prior["second"] = dict(
            center=e0_2,
            sigma=sigma_e2,
            bounds=list(NORMAL_BOUNDS),
            center_origin=second_origin,
            vertical_scale=scale,
            surface=surface,
            prior_is_calibrated_uncertainty=False,
        )
    else:
        e0_2 = sigma_e2 = None

    def route(e, e2=None):
        rows = [
            dict(
                flight_index=last,
                native_contact_epoch=epoch,
                ground_ordinal=0,
                normal_restitution=float(e),
            )
        ]
        if e2 is not None:
            rows.append(
                dict(
                    flight_index=last,
                    native_contact_epoch=epoch,
                    ground_ordinal=1,
                    normal_restitution=float(e2),
                )
            )
        return rows

    def terms(p, chain, predicted):
        image = predicted[offset:] - scene.pixels[-1]
        spin_term = (p[spin : spin + 3] - src[spin : spin + 3]) / SPIN_PRIOR_SCALE
        impacts = chain[-1]["bounces"]
        original_impacts = _original_impacts(impacts, spec)
        applied = float(impacts[0]["applied_restitution"])
        normal = np.asarray([(applied - e0) / sigma_e])
        if second_active:
            # Original ordinal 1 exists here: _original_impacts demanded both grounds.
            applied_second = float(original_impacts[1]["applied_restitution"])
            normal = np.r_[normal, (applied_second - e0_2) / sigma_e2]
        eq, clearance = boundary.constraints(chain[-1:], queries[-1:], 1, [[]], None)
        impact_slacks = _impact_slacks(original_impacts, spec["intervals"])
        # Seconds condition the optimizer's epoch inequalities; feasibility never
        # compares them with the metre tolerance. The first epoch is chart-bounded.
        optimizer_slacks = np.r_[clearance, impact_slacks[2:] / fps]
        residual = np.r_[
            image.ravel(), spin_term, np.minimum(clearance, 0) / CLEARANCE_SCALE_M, normal
        ]
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite terminal residual")
        components = dict(
            image=float(np.sum(image**2)),
            spin=float(spin_term @ spin_term),
            clearance=float(np.sum((np.minimum(clearance, 0) / CLEARANCE_SCALE_M) ** 2)),
            normal=float(normal[0] ** 2),
        )
        if second_active:
            components["second_normal"] = float(normal[1] ** 2)
        return residual, dict(
            boundary_equality_m=eq,
            clearance_slacks_m=clearance,
            impact_slacks_frames=impact_slacks,
            optimizer_slacks=optimizer_slacks,
            components=components,
            native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
            impacts=[_impact_row(b) for b in impacts],
            continuation_impacts=[_impact_row(b) for b in impacts[grounds:]],
        )

    source_row = dict(
        impact_count=len(source_impacts),
        impacts=[_impact_row(b) for b in source_impacts],
        inside_intervals=[
            bool(lo - EPOCH_TOLERANCE_FRAMES <= float(b["frame"]) <= hi + EPOCH_TOLERANCE_FRAMES)
            for b, (lo, hi) in zip(source_impacts, spec["intervals"])
        ],
        representable=False,
        cost=None,
        components=None,
        native_rms_px=None,
        feasible=None,
    )
    try:
        source_residual, source_receipt = terms(src, original_chain, source_prediction)
        source_row.update(
            cost=float(source_residual @ source_residual),
            components=source_receipt["components"],
            native_rms_px=source_receipt["native_rms_px"],
            feasible=bool(_feasible(source_receipt)),
            representable=len(source_impacts) >= grounds
            and all(source_row["inside_intervals"])
            and len(source_row["inside_intervals"]) == grounds,
        )
    except NUMERICAL_ERRORS as error:
        source_row["reason"] = f"{type(error).__name__}: {error}"

    projector = prefix.ConnectedFollowingProjector(scene, last, spec["interval"], src[velocity + 2])
    if source_impacts and source_row["inside_intervals"][0]:
        # The source root may sit a rounding error outside its bound; clip it back.
        tau0 = float(np.clip(source_impacts[0]["frame"], *spec["interval"]))
        tau_origin = "source_impact"
    else:
        tau0, tau_origin = spec["initial_epoch"], "clipped_original_event"
    q0 = np.r_[src[velocity : velocity + 2], tau0, src[spin : spin + 3]]
    lo = np.array([-75.0, -75.0, spec["interval"][0], -6.0, -6.0, -6.0])
    hi = np.array([75.0, 75.0, spec["interval"][1], 6.0, 6.0, 6.0])
    names = list(COORDINATES)
    if free_normal:
        seed_e = existing["normal_restitution"] if existing else e0
        q0 = np.r_[q0, float(np.clip(seed_e, *NORMAL_BOUNDS))]
        lo, hi = np.r_[lo, NORMAL_BOUNDS[0]], np.r_[hi, NORMAL_BOUNDS[1]]
        names.append(FIRST_NORMAL)
    if second_active:
        seed_e2 = existing_second["normal_restitution"] if existing_second else e0_2
        q0 = np.r_[q0, float(np.clip(seed_e2, *NORMAL_BOUNDS))]
        lo, hi = np.r_[lo, NORMAL_BOUNDS[0]], np.r_[hi, NORMAL_BOUNDS[1]]
        names.append(SECOND_NORMAL)
    if np.any(q0 < lo) or np.any(q0 > hi):
        raise ValueError("source terminal launch outside original bounds")
    frozen = np.ones(len(src), bool)
    frozen[velocity : velocity + 3] = False
    frozen[spin : spin + 3] = False
    constraint_size = 1 + 2 * (grounds - 1)

    counters = dict(evaluations=0, invalid_trials=0, feasible_trials=0)
    completed = None
    active = dict(e=None)
    second_index = names.index(SECOND_NORMAL) if second_active else None

    def evaluate(q):
        nonlocal completed
        if time.monotonic() - started > seconds:
            raise TimeoutError("terminal ground response wall budget reached")
        counters["evaluations"] += 1
        p = src.copy()
        p[velocity : velocity + 2] = q[:2]
        p[spin : spin + 3] = q[3:6]
        if free_normal:
            e = float(q[6])
            e2 = float(q[second_index]) if second_active else None
            record = ground.merge(initial_record, route(e, e2), scene)
            if active["e"] != (e, e2):
                # Chart receipts (second impact) depend on the response; never reuse them.
                projector.chart.cache.clear()
                active["e"] = (e, e2)
        else:
            record = initial_record
        try:
            with _scope(net, record, scene):
                p, root = projector.project(p, float(q[2]))
                chain = prefix.full.model.chain(scene, p, query_frames=queries)
                predicted = np.asarray(
                    prefix.full.exposure.prediction(
                        scene, p, axes, duration, termination_kind=context["termination_kind"]
                    )
                )
            residual, receipt = terms(p, chain, predicted)
        except NUMERICAL_ERRORS:
            counters["invalid_trials"] += 1
            raise
        receipt.update(p=p, chain=chain, root=root, response=record, q=np.asarray(q, float).copy())
        cost = float(residual @ residual)
        if _feasible(receipt):
            counters["feasible_trials"] += 1
            if completed is None or cost < completed[0]:
                completed = (cost, np.asarray(q, float).copy(), receipt)
        return residual, receipt

    initial_chart = None
    restoration = dict(status="not_started")
    termination, solver = None, None
    regime_retries = dict(
        policy="terminal_boundary_two_launch_spin_retries_v1",
        boundary_probe_rad_s=regime_recovery.BOUNDARY_PROBE_RAD_S,
        launch_retry_rad_s=regime_recovery.LAUNCH_RETRY_RAD_S,
        boundaries=[],
        attempts=[],
        selection="lowest feasible unchanged input objective; original physics in every trial",
        shared_wall_budget_seconds=seconds,
    )
    try:
        try:
            initial_residual, initial_receipt = evaluate(q0)
            initial_chart = dict(
                cost=float(initial_residual @ initial_residual),
                components=initial_receipt["components"],
                native_rms_px=initial_receipt["native_rms_px"],
                feasible=bool(_feasible(initial_receipt)),
                minimum_clearance_slack_m=float(np.min(initial_receipt["clearance_slacks_m"])),
                minimum_impact_slack_frames=float(np.min(initial_receipt["impact_slacks_frames"])),
                impacts=initial_receipt["impacts"],
            )
        except NUMERICAL_ERRORS as error:
            initial_chart = dict(
                status="chart_not_entered", reason=f"{type(error).__name__}: {error}"
            )
        restored, restoration = _restore(
            evaluate,
            q0,
            lo,
            hi,
            started,
            seconds,
            max_nfev,
            constraint_size,
            feasible=_feasible,
            inequality_key="optimizer_slacks",
        )
        state, termination = boundary.solve(
            evaluate,
            restored,
            lo,
            hi,
            jacobian=prefix._jacobian,
            maxiter=max_nfev,
            seconds=seconds,
            started=started,
            feasible=_feasible,
            inequality_key="optimizer_slacks",
        )
        solver = {k: state.get(k) for k in ("calls", "invalid", "feasible_trials")}
        # A central difference across the Cross spin-regime jump can make SLSQP
        # declare success at its starting point. Preserve the ordinary solve,
        # then try both sides of a detected terminal boundary with the unchanged
        # law. Neither a saved fit nor an acceptance result chooses a retry.
        regime_retries["boundaries"] = [
            row
            for row in regime_recovery.boundary_evidence(scene, completed[2]["chain"])
            if row["flight_index"] == last
        ]
        retry_origin = np.asarray(completed[1], float).copy()
        if regime_retries["boundaries"]:
            for direction in (-1, 1):
                if time.monotonic() - started >= seconds:
                    regime_retries["stopped"] = "shared wall budget exhausted"
                    break
                seed = retry_origin.copy()
                seed[3] += direction * regime_recovery.LAUNCH_RETRY_RAD_S / 100.0
                attempt = dict(direction=direction, initial_q=seed.tolist(), status="held")
                regime_retries["attempts"].append(attempt)
                if np.any(seed < lo) or np.any(seed > hi):
                    attempt["reason"] = "retry outside original launch bounds"
                    continue
                try:
                    restored_seed, retry_restoration = _restore(
                        evaluate,
                        seed,
                        lo,
                        hi,
                        started,
                        seconds,
                        max_nfev,
                        constraint_size,
                        feasible=_feasible,
                        inequality_key="optimizer_slacks",
                    )
                    retried, retry_termination = boundary.solve(
                        evaluate,
                        restored_seed,
                        lo,
                        hi,
                        jacobian=prefix._jacobian,
                        maxiter=max_nfev,
                        seconds=seconds,
                        started=started,
                        feasible=_feasible,
                        inequality_key="optimizer_slacks",
                    )
                    attempt.update(
                        status="measured",
                        restoration=retry_restoration,
                        termination=retry_termination,
                        cost=retried["best_cost"],
                        q=retried["best_q"].tolist(),
                    )
                except (*NUMERICAL_ERRORS, TimeoutError) as error:
                    attempt["reason"] = f"{type(error).__name__}: {error}"
    except (*NUMERICAL_ERRORS, TimeoutError) as error:
        reason = f"{type(error).__name__}: {error}"
        if restoration.get("status") == "not_started":
            restoration = dict(status="failed", reason=reason)
        termination = dict(
            kind="completed_candidate_retained"
            if completed is not None
            else "numerical_preparation_or_budget_failure",
            reason=reason,
        )

    details = dict(
        automatic_inference_eligible=False,
        gate_selection=False,
        external_fitted_inputs_used=False,
        arm="free_normal" if free_normal else "fixed_normal",
        free_normal=free_normal,
        status="source_retained",
        feasible_output=bool(source_row["representable"] and source_row["feasible"]),
        full_vector=src.tolist(),
        initial_source=src.tolist(),
        ground_response=deepcopy(initial_record),
        initial_ground_response=deepcopy(initial_record),
        retained_net_response=deepcopy(net_response),
        original_interval=spec,
        coordinates=names,
        initial_q=q0.tolist(),
        initial_q_origin=dict(
            first_impact=tau_origin,
            normal="existing_route" if existing else "prior_center",
            **(
                dict(second_normal="existing_route" if existing_second else "prior_center")
                if second_active
                else {}
            ),
        ),
        bounds=dict(lo=lo.tolist(), hi=hi.tolist()),
        normal_prior=prior,
        source=source_row,
        initial_chart=initial_chart,
        feasibility=dict(restoration=restoration, solver=solver, **counters),
        termination=termination,
        regime_retries=regime_retries,
        consumed_check_rows=consumed,
        fit_policy=dict(
            mode="original terminal first-impact interval with persistent first-ground normal route"
            if free_normal
            else "original terminal first-impact interval, nominal normal law",
            objective={**OBJECTIVE, **SECOND_OBJECTIVE} if second_active else OBJECTIVE,
            ground_count=grounds,
            second_normal=dict(
                requested=second_normal,
                active=second_active,
                reason=None
                if second_active
                else (
                    "not requested"
                    if not second_normal
                    else (
                        "one original terminal ground; second response stays nominal"
                        if grounds == 1
                        else "two distinct native exposures after the original second interval required"
                    )
                ),
                registry_ordinal=1 if second_active else None,
                **(dict(native_support=deepcopy(second_support)) if second_normal else {}),
            ),
            continuation=deepcopy(spec["continuation"]),
            second_ground_constraint="original interval slack in seconds" if grounds == 2 else None,
            impact_epoch_tolerance_frames=EPOCH_TOLERANCE_FRAMES,
            feasibility_units="clearance metres; original impact epochs frames",
            new_fitted_degrees_of_freedom=(1 if free_normal else 0) + (1 if second_active else 0),
            fixed_prefix=True,
            fixed_rebound_scalars=src[-2:].tolist(),
            duration=duration,
            max_nfev=max_nfev,
            seconds=seconds,
            optimizer="feasibility restoration then SLSQP, absolute central FD 1e-6; input-only feasible selection",
            added_native_rows=added,
        ),
        wall_seconds=time.monotonic() - started,
    )
    if completed is None:
        details.update(
            reason="no feasible terminal trial",
            feasible_output=bool(source_row["representable"] and source_row["feasible"]),
        )
        return context, details
    cost, q, best = completed
    if source_row["representable"] and source_row["feasible"]:
        improved = cost < source_row["cost"] - INCUMBENT_RELATIVE_TOLERANCE * max(
            1.0, source_row["cost"]
        )
        if not improved:
            details.update(
                reason="feasible candidate not better than the representable source incumbent",
                candidate_cost=cost,
                candidate_components=best["components"],
            )
            return context, details
    p, record = best["p"], best["response"]
    with _scope(net, record, scene):
        replayed = prefix.full.model.chain(scene, p, query_frames=queries)
    impacts = replayed[-1]["bounces"]
    verification = []
    try:
        original_impacts = _original_impacts(impacts, spec)
    except ValueError as error:
        verification.append(str(error))
    else:
        if abs(float(impacts[0]["frame"]) - float(q[2])) > EPOCH_TOLERANCE_FRAMES:
            verification.append("original-law replay does not reproduce terminal impact chart")
        if np.min(_impact_slacks(original_impacts, spec["intervals"])) < -EPOCH_TOLERANCE_FRAMES:
            verification.append("replay impact outside original interval")
        if free_normal and impacts[0].get("applied_restitution") != float(q[6]):
            verification.append("replay did not apply the fitted normal route")
        if second_active and original_impacts[1].get("applied_restitution") != float(
            q[second_index]
        ):
            verification.append("replay did not apply the fitted second-ground normal route")
    _, slack = boundary.constraints(replayed[-1:], queries[-1:], 1, [[]], None)
    if np.min(slack, initial=1) < -boundary.TOLERANCE_M:
        verification.append("original terminal replay violates net feasibility")
    prefix_error = max(
        (
            float(np.max(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"]))))
            for a, b in zip(replayed[:-1], original_chain[:-1])
        ),
        default=0.0,
    )
    if prefix_error != 0.0:
        verification.append("terminal fit changed earlier physical positions")
    if not np.array_equal(p[frozen], src[frozen]):
        verification.append("terminal fit changed a frozen parameter")
    if verification:
        details.update(
            reason="candidate replay verification failed: " + "; ".join(verification),
            candidate_cost=cost,
        )
        return context, details
    at_bounds = [
        name
        for name, value, a, b in zip(names, q, lo, hi, strict=True)
        if abs(value - a) <= BOUND_TOLERANCE or abs(value - b) <= BOUND_TOLERANCE
    ]
    details.update(
        status="refined",
        feasible_output=True,
        full_vector=p.tolist(),
        ground_response=deepcopy(record),
        q=q.tolist(),
        cost=cost,
        components=best["components"],
        native_rms_px=best["native_rms_px"],
        impacts=[_impact_row(b) for b in impacts],
        continuation_impacts=[_impact_row(b) for b in impacts[grounds:]],
        impact_epoch=float(q[2]),
        unforced_impact_epoch=float(impacts[0]["frame"]),
        root={k: v for k, v in best["root"].items()},
        normal=dict(
            fitted=float(q[6]) if free_normal else None,
            applied=float(impacts[0]["applied_restitution"]),
            nominal_at_fitted_incoming=float(
                impacts[0].get("nominal_applied_restitution", impacts[0]["applied_restitution"])
            ),
            prior_center=e0,
            prior_sigma=sigma_e,
            **(
                dict(
                    second=dict(
                        fitted=float(q[second_index]),
                        applied=float(impacts[1]["applied_restitution"]),
                        nominal_at_fitted_incoming=float(
                            impacts[1].get(
                                "nominal_applied_restitution", impacts[1]["applied_restitution"]
                            )
                        ),
                        prior_center=e0_2,
                        prior_sigma=sigma_e2,
                    )
                )
                if second_active
                else {}
            ),
        ),
        bounds_reached=at_bounds,
        prefix_max_position_error_m=prefix_error,
        clearance_slacks_m=slack.tolist(),
        wall_seconds=time.monotonic() - started,
    )
    return context, details
