"""Fit a terminal net failure in its original event interval on the physical plane.

Opened labeled experiment, fixed prefix. Default retains the original net law;
--mesh-response explicitly fits passive restitution/tangential retention. The final incoming
velocity/spin and net epoch vary; optional first_contact_toss jointly fits the single-serve contact
XYZ/epoch and original incoming toss with a free net response. It is default-off,
uses one fixed native objective and never changes observations. --joint also allows its preceding flight to move; an implicit Vy coordinate prevents the missing
net branch from supplying a flat residual. Run --case KEY --output DIR.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import labeled_interval_block_fit as interval
from cv.experiments.connected_shooting import labeled_net_epoch_chart as chart
from cv.experiments.connected_shooting import labeled_net_height_chart as height_chart
from cv.experiments.connected_shooting import labeled_passive_tape as tape
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import provenance

full = interval.full


def qualify_first_contact_toss(
    context, observations, duration, *, joint_toss_requalification=False
):
    """Original single serve/net/ground topology and operator-aware native toss rows."""
    from cv.experiments.connected_shooting import labeled_prefix_joint_impact as prefix
    from cv.experiments.connected_shooting import serve_timing_profile as profile

    scene = context["scene"]
    if len(scene.pixels) != 1 or scene.parameterization != "single_shooting":
        raise ValueError("net serve toss requires one single-shooting flight")
    role = context.get("net_serve_evidence", {})
    if role.get("role") != "serve":
        raise ValueError("net serve toss requires qualified original serve role")
    events = context["events"]
    groups = [[e for e in events if e["event_type"] == k] for k in ("contact", "net_hit", "bounce")]
    if any(len(g) != 1 for g in groups):
        raise ValueError("net serve toss requires exactly one original contact, net and ground")
    contact, net, ground = [g[0] for g in groups]
    ci = np.asarray(profile.contact_epoch_bounds(context), float)
    ni, gi = [np.asarray(e["frame_interval"], float) for e in (net, ground)]
    if any(a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1] for a in (ci, ni, gi)):
        raise ValueError("finite ordered original net serve intervals required")
    if not ci[1] < ni[0] <= ni[1] < gi[0] <= gi[1] < scene.contact_frames[-1]:
        raise ValueError("net serve requires ordered contact, net, ground and native context")
    if scene.net_hit_frames is None or len(scene.net_hit_frames[0]) != 1:
        raise ValueError("net serve scene must retain its original collision")
    if float(scene.net_hit_frames[0][0]) != float(net["frame"]):
        raise ValueError("original net event and scene identity disagree")
    if float(scene.contact_frames[0]) != float(contact["frame"]):
        raise ValueError("original serve event and scene identity disagree")
    rows = prefix._prefix_rows(
        observations, float(ci[0]), duration, joint_toss_requalification=joint_toss_requalification
    )
    for epoch in ci:
        profile.profile_context(context, float(epoch))
    return dict(rows=rows, contact_interval=ci, contact=contact, net=net, ground=ground)


def fit(
    context,
    source,
    duration,
    *,
    maxiter,
    seconds,
    mesh_response=False,
    joint=False,
    upper_net=False,
    mesh_height=False,
    jacobian_step=1e-4,
    quadratic_images=False,
    inclination_bounds=None,
    response_type=tape.TapeResponse,
    free_net_velocity=False,
    first_ground_seed=None,
    free_net_spin=False,
    first_ground_normal_prior=None,
    first_ground_horizontal_mode="off",
    first_contact_toss=None,
    joint_toss_requalification=False,
    initial_net_response=None,
    player_anchor=None,
    toss_horizontal_sigma_mps=None,
    toss_weight=4.0,
):
    if first_ground_horizontal_mode not in ("off", "retention", "heading"):
        raise ValueError("explicit supported first-ground horizontal mode required")
    if first_ground_horizontal_mode != "off" and first_ground_normal_prior is None:
        raise ValueError("horizontal response requires qualified direct-normal ground evidence")
    if type(mesh_height) is not bool or (mesh_height and upper_net):
        raise ValueError(
            "mesh height must be an explicit boolean and cannot combine with upper net"
        )
    if first_ground_normal_prior is not None:
        if not free_net_velocity or free_net_spin or mesh_response:
            raise ValueError(
                "direct ground normal coefficient requires preserved-spin free net velocity"
            )
        center = float(first_ground_normal_prior["center"])
        sigma = float(first_ground_normal_prior["sigma"])
        if not np.isfinite([center, sigma]).all() or not 0.05 <= center <= 1 or sigma <= 0:
            raise ValueError("finite bounded direct-normal prior required")
    if free_net_spin and not free_net_velocity:
        raise ValueError("independent net spin requires explicit free net velocity")
    if first_ground_seed is not None and not free_net_velocity:
        raise ValueError("first-ground initialization requires explicit free net velocity")
    if free_net_velocity and (mesh_response or inclination_bounds is not None):
        raise ValueError(
            "free net velocity and material-response coordinates are separate controls"
        )
    if inclination_bounds is not None:
        bounds = np.asarray(inclination_bounds, float)
        if (
            not mesh_response
            or bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
        ):
            raise ValueError("finite increasing inclination bounds require explicit mesh response")
        inclination_bounds = tuple(float(x) for x in bounds)
    if initial_net_response is not None and first_contact_toss is None:
        raise ValueError("current net-response initialization requires explicit joint toss mode")
    toss_support = None
    if first_contact_toss is not None:
        if joint or not free_net_velocity or upper_net or not mesh_height:
            raise ValueError(
                "joint net serve toss requires single-flight full mesh and free response"
            )
        if not np.isfinite(toss_weight) or toss_weight <= 0:
            raise ValueError("positive shared toss weight required")
        toss_support = qualify_first_contact_toss(
            context,
            first_contact_toss,
            duration,
            joint_toss_requalification=joint_toss_requalification,
        )
    started = time.monotonic()
    scene, axes, added = full.merge_scene(
        context["scene"], context["heldout"], context["bounces"], context["axes"]
    )
    n = len(scene.pixels)
    last = n - 1
    source = np.asarray(source)
    # The fixed-contact source need not integrate under a new net response.
    # Its prior net state is only an initializer; the joint arm starts from toss rays.
    physical = (
        [{"start_xyz": source[:3], "net_hits": []}]
        if toss_support is not None
        else full.model.chain(scene, source)
    )
    event = next(
        e
        for e in context["events"]
        if e["event_type"] == "net_hit"
        and scene.contact_frames[-2] < e["frame"] < scene.contact_frames[-1]
    )
    changed = np.arange(last - int(joint), n)
    selected = np.r_[
        np.concatenate([np.arange(3 + 3 * i, 6 + 3 * i) for i in changed]),
        np.concatenate([np.arange(3 + 3 * n + 3 * i, 6 + 3 * n + 3 * i) for i in changed]),
    ]
    width = len(selected)
    y_index = 3 * (len(changed) - 1) + 1
    mid = float(event["frame"])
    scale = np.r_[[30.0] * (3 * len(changed)), [3.0] * (3 * len(changed))]
    lo = np.r_[[-75.0] * (3 * len(changed)), [-6.0] * (3 * len(changed))]
    hi = -lo
    lo[y_index], hi[y_index] = np.asarray(event["frame_interval"]) - mid
    scale[y_index] = 1.0
    initial = source[selected].copy()
    initial[y_index] = 0.0
    if upper_net:
        lo[y_index + 1], hi[y_index + 1] = -0.10, full.model.R_BALL + 0.03
        scale[y_index + 1] = 1.0
        initial[y_index + 1] = 0.0
    if mesh_height:
        # A numerical margin above simultaneous ground/net contact; about 10um,
        # not an observed exclusion of lower mesh contacts.
        lo[y_index + 1], hi[y_index + 1] = 1e-5, 1.0
        scale[y_index + 1] = 1.0
        hits = physical[last].get("net_hits", [])
        if hits:
            xyz = hits[0]["x"]
            initial[y_index + 1] = (
                float(xyz[2]) - full.model.R_BALL
            ) / height_chart.net_tape_height(float(xyz[0]))
        else:
            initial[y_index + 1] = 0.5
    if mesh_response:
        initial = np.r_[initial, 0.25, 0.4]
        scale = np.r_[scale, 1.0, 1.0]
        lo, hi = np.r_[lo, 0.01, 0.01], np.r_[hi, 1.0, 1.0]
        if inclination_bounds is not None:
            initial = np.r_[initial, 0.0]
            scale = np.r_[scale, 1.0]
            lo, hi = np.r_[lo, inclination_bounds[0]], np.r_[hi, inclination_bounds[1]]
    if free_net_velocity:
        initial = np.r_[initial, np.zeros(3)]
        scale = np.r_[scale, [15.0, 15.0, 15.0]]
        lo, hi = np.r_[lo, [-75.0] * 3], np.r_[hi, [75.0] * 3]
    if free_net_spin:
        initial = np.r_[initial, np.zeros(3)]
        scale = np.r_[scale, [300.0] * 3]
        lo, hi = np.r_[lo, [-1500.0] * 3], np.r_[hi, [1500.0] * 3]
    if first_ground_normal_prior is not None:
        initial = np.r_[initial, center]
        scale, lo, hi = np.r_[scale, 1.0], np.r_[lo, 0.05], np.r_[hi, 1.0]
    if first_ground_horizontal_mode != "off":
        initial, scale = np.r_[initial, 1.0], np.r_[scale, 1.0]
        lo, hi = np.r_[lo, 0.0], np.r_[hi, 4.0]
    if first_ground_horizontal_mode == "heading":
        initial, scale = np.r_[initial, 0.0], np.r_[scale, 5.0]
        lo, hi = np.r_[lo, -20.0], np.r_[hi, 20.0]
    toss_offset = len(initial)
    toss_seed_receipt = None
    if toss_support is not None:
        from cv.experiments.connected_shooting import (
            labeled_isolated_serve_fit as isolated,
            labeled_prefix_joint_impact as prefix,
            labeled_toss_player_anchor as anchors,
            athlete_priors,
        )

        ci = toss_support["contact_interval"]
        contact_mid = float(np.mean(ci))
        source_epoch = float(scene.contact_frames[0])
        toss_rows = toss_support["rows"]
        toss_frames = np.array([r["frame"] for r in toss_rows])
        toss_cameras = np.array([r["camera"] for r in toss_rows])
        toss_pixels = np.array([r["pixel"] for r in toss_rows])
        toss_sigmas = np.array([r["uncertainty_px"] for r in toss_rows])
        anchors.require_rows(player_anchor, toss_frames)
        isolated.velocity_residual(np.zeros(3), toss_horizontal_sigma_mps)
        depth = float(np.clip(source[1], *context["depth_bounds"]))
        xyz, incoming, toss_seed_receipt = isolated.toss_ray_seed(
            toss_rows, source_epoch, scene.fps, depth
        )
        initial = np.r_[initial, xyz, incoming, source_epoch - contact_mid]
        scale = np.r_[scale, [1.0] * 3, [5.0] * 3, 1.0]
        lo = np.r_[lo, 0.0, context["depth_bounds"][0], 2.0, [-12.0] * 3, ci[0] - contact_mid]
        hi = np.r_[hi, 10.97, context["depth_bounds"][1], 4.0, [12.0] * 3, ci[1] - contact_mid]
        if ci[0] == ci[1]:
            # Fixed original singleton epochs do not need a solver coordinate.
            initial, scale, lo, hi = initial[:-1], scale[:-1], lo[:-1], hi[:-1]
    initial = np.clip(initial, lo + 1e-9, hi - 1e-9) / scale
    target = np.concatenate(scene.pixels)
    cache = FlightCache(entries_per_flight=64)
    loss = interval.block.event_constraints.mixed_loss(0 if quadratic_images else 2 * len(target))
    best = None
    calls = bad = 0
    invalid_reasons = Counter()
    residual_count = None

    def expand(x):
        p = source.copy()
        values = x * scale
        p[selected] = values[:width]
        active_scene = scene
        contact_epoch = float(scene.contact_frames[0])
        if toss_support is not None:
            contact_epoch = contact_mid + (values[toss_offset + 6] if ci[0] != ci[1] else 0.0)
            active_scene = replace(
                scene, contact_frames=np.r_[contact_epoch, scene.contact_frames[1:]]
            )
            p[:3] = values[toss_offset : toss_offset + 3]
        epoch = mid + values[y_index]
        p[selected[y_index]] = source[selected[y_index]]
        start = (
            p[:3]
            if toss_support is not None
            else full.model.chain(scene, p)[last - 1]["end_xyz"]
            if joint
            else physical[-1]["start_xyz"]
        )
        theta = np.r_[
            start, p[3 + 3 * last : 6 + 3 * last], p[3 + 3 * n + 3 * last : 6 + 3 * n + 3 * last]
        ]
        if upper_net or mesh_height:
            theta[5] = source[selected[y_index + 1]]
        projector = (
            height_chart.project_net_height
            if upper_net or mesh_height
            else chart.project_net_velocity
        )
        projected, receipt = projector(
            theta,
            active_scene.contact_frames[-2],
            epoch,
            scene.fps,
            scene.surface,
            bounce_profile=scene.bounce_profile,
            rebound_scales=p[-2:],
            **(
                {"mesh_fraction": values[y_index + 1]}
                if mesh_height
                else {"height_offset_m": values[y_index + 1]}
                if upper_net
                else {}
            ),
        )
        p[selected[y_index]] = projected[4]
        if upper_net or mesh_height:
            p[selected[y_index + 1]] = projected[5]
        response = (
            response_type(
                0.0 if inclination_bounds is None else values[width + 2],
                values[width],
                values[width + 1],
            )
            if mesh_response
            else None
        )
        if free_net_velocity:
            from cv.experiments.connected_shooting.labeled_net_free_response import FreeNetVelocity

            response = FreeNetVelocity(values[width : width + 3])
        if free_net_spin:
            from cv.experiments.connected_shooting.labeled_net_spin_response import FreeNetState

            response = FreeNetState(values[width : width + 3], values[width + 3 : width + 6])
        if first_ground_normal_prior is not None:
            from cv.experiments.connected_shooting.labeled_net_normal_response import (
                NetNormalResponse,
            )

            response = NetNormalResponse(
                response,
                values[width + 3],
                horizontal_mode=first_ground_horizontal_mode,
                tangential_multiplier=values[width + 4]
                if first_ground_horizontal_mode != "off"
                else 1.0,
                transverse_velocity_mps=values[width + 5]
                if first_ground_horizontal_mode == "heading"
                else 0.0,
            )
        return p, receipt, response, active_scene, contact_epoch

    def residual(x):
        nonlocal best, calls, bad, residual_count
        if time.monotonic() - started > seconds:
            raise TimeoutError("bounded net-epoch fit exhausted its wall time")
        calls += 1
        try:
            p, receipt, response, active_scene, contact_epoch = expand(x)
            with tape.using_response(response) if response else nullcontext():
                _, physics, _ = interval.block.event_constraints.evaluate(
                    active_scene,
                    p,
                    context["bounces"],
                    2.0,
                    simulation_cache=cache,
                    net_clearance_scale_m=None if toss_support is not None else 0.1,
                    terminal_rebound_frames=context.get("terminal_rebound_frames"),
                )
                image = (
                    full.exposure.prediction(
                        active_scene,
                        p,
                        axes,
                        duration,
                        cache,
                        termination_kind=context["termination_kind"],
                    )
                    - target
                )
            prior = (p[selected] - source[selected]) / np.r_[
                [30.0] * (3 * len(changed)), [6.0] * (3 * len(changed))
            ]
            if free_net_velocity:
                incoming_speed = np.linalg.norm(receipt["incoming_velocity_mps"])
                outgoing_speed = np.linalg.norm(response.outgoing_velocity_mps)
                # Broad soft plausibility term, not a fixed material law or hard passivity gate.
                prior = np.r_[prior, max(outgoing_speed - incoming_speed, 0.0) / 10.0]
            if free_net_spin:
                prior = np.r_[prior, np.asarray(response.outgoing_spin_rad_s) / 600.0]
            if first_ground_normal_prior is not None:
                prior = np.r_[prior, (response.first_ground_normal_restitution - center) / sigma]
            if first_ground_horizontal_mode != "off":
                prior = np.r_[prior, (response.tangential_multiplier - 1.0) / 0.5]
            if first_ground_horizontal_mode == "heading":
                prior = np.r_[prior, response.transverse_velocity_mps / 5.0]
            toss_detail = {}
            if toss_support is not None:
                incoming = (x * scale)[toss_offset + 3 : toss_offset + 6]
                prediction = prefix._toss_prediction(
                    toss_rows,
                    toss_cameras,
                    toss_frames,
                    p[:3],
                    incoming,
                    contact_epoch,
                    scene.fps,
                    duration,
                )
                delta = prediction - toss_pixels
                toss_residual = (toss_weight * delta / toss_sigmas[:, None]).ravel()
                athlete = athlete_priors.optimization_residuals(
                    np.array([p[:3]]), context["players"]
                )
                dt = (toss_frames - contact_epoch) / scene.fps
                xyz = p[:3] + dt[:, None] * incoming + 0.5 * dt[:, None] ** 2 * isolated.GRAVITY
                anchor = anchors.residuals(player_anchor, xyz[:, :2])
                velocity = isolated.velocity_residual(incoming, toss_horizontal_sigma_mps)
                prior = np.r_[prior, toss_residual, 4 * athlete, anchor, velocity]
                toss_detail = dict(
                    contact_epoch=contact_epoch,
                    toss_incoming_velocity_mps=incoming.copy(),
                    toss_rms_px=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                    toss_cost=float(toss_residual @ toss_residual),
                    athlete_cost=float(np.sum((4 * athlete) ** 2)),
                    player_relative_toss=anchors.report(
                        player_anchor, p[:3], incoming, toss_frames, contact_epoch, scene.fps
                    ),
                    toss_projections=[
                        dict(
                            frame=r["frame"],
                            observed=r["pixel"],
                            predicted=pred.tolist(),
                            incoming_operator=r.get("incoming_operator"),
                            uncertainty_px=r["uncertainty_px"],
                        )
                        for r, pred in zip(toss_rows, prediction, strict=True)
                    ],
                )
            r = np.r_[image.ravel(), physics, prior]
            residual_count = len(r)
            cost = float(2 * np.sum(loss((r / 2) ** 2)[0]))
            if best is None or cost < best["cost"]:
                best = dict(
                    parameters=p.copy(),
                    cost=cost,
                    chart=receipt,
                    response=None
                    if response is None
                    else dict(
                        outgoing_velocity_mps=list(response.outgoing_velocity_mps),
                        outgoing_spin_rad_s=list(response.outgoing_spin_rad_s),
                    )
                    if free_net_spin
                    else dict(outgoing_velocity_mps=list(response.outgoing_velocity_mps))
                    if free_net_velocity
                    else dict(
                        normal_angle=response.normal_angle,
                        restitution=response.restitution,
                        tangential_retention=response.tangential_retention,
                    ),
                    all_native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
                    calls=calls,
                    **toss_detail,
                )
                if first_ground_normal_prior is not None:
                    best["response"]["first_ground_normal_restitution"] = (
                        response.first_ground_normal_restitution
                    )
                if first_ground_horizontal_mode != "off":
                    best["response"].update(
                        first_ground_horizontal_mode=first_ground_horizontal_mode,
                        first_ground_tangential_multiplier=response.tangential_multiplier,
                        first_ground_transverse_velocity_mps=response.transverse_velocity_mps,
                    )
            return r
        except ValueError as error:
            bad += 1
            invalid_reasons[str(error)] += 1
            if residual_count is None:
                raise
            return np.full(residual_count, 1e5)

    initialization_receipt = None
    if free_net_velocity:
        _, initial_receipt, _, _, _ = expand(initial)
        initial_outgoing = tape.TapeResponse(0.0, 0.25, 0.4).velocity(
            np.asarray(initial_receipt["incoming_velocity_mps"])
        )
        if first_ground_seed is not None:
            from cv.experiments.connected_shooting.labeled_net_ground_seed import (
                ballistic_ground_seed,
            )

            initialization_receipt = ballistic_ground_seed(
                initial_receipt["net_xyz_m"],
                initial_receipt["net_frame"],
                first_ground_seed["xyz_m"],
                first_ground_seed["event_frame"],
                scene.fps,
            )
            if initialization_receipt["status"] != "supported":
                raise ValueError(f"unsupported first-ground initializer: {initialization_receipt}")
            initialization_receipt["input_ground_target"] = first_ground_seed
            initialization_receipt["input_net_chart"] = initial_receipt
            initial_outgoing = np.asarray(initialization_receipt["outgoing_velocity_mps"])
        if toss_support is not None and first_ground_seed is None:
            # Preserve the current invocation's basin while moving its contact.
            # Do not integrate the prior complete tail under a substituted law.
            if initial_net_response is not None:
                initial_outgoing = np.asarray(initial_net_response["outgoing_velocity_mps"], float)
                if (
                    initial_outgoing.shape != (3,)
                    or not np.isfinite(initial_outgoing).all()
                    or np.any(abs(initial_outgoing) > 75)
                ):
                    raise ValueError(
                        "finite bounded same-invocation outgoing net velocity required"
                    )
                initialization_receipt = dict(
                    mode="same_invocation_current_net_response", response=initial_net_response
                )
                if (
                    first_ground_normal_prior is not None
                    and "first_ground_normal_restitution" in initial_net_response
                ):
                    index = width + 3
                    coefficient = float(initial_net_response["first_ground_normal_restitution"])
                    if not np.isfinite(coefficient) or not lo[index] <= coefficient <= hi[index]:
                        raise ValueError("current net ground coefficient outside declared bounds")
                    initial[index] = coefficient / scale[index]
                for name, offset, enabled in (
                    (
                        "first_ground_tangential_multiplier",
                        4,
                        first_ground_horizontal_mode != "off",
                    ),
                    (
                        "first_ground_transverse_velocity_mps",
                        5,
                        first_ground_horizontal_mode == "heading",
                    ),
                ):
                    if enabled and name in initial_net_response:
                        index = width + offset
                        value = float(initial_net_response[name])
                        if not np.isfinite(value) or not lo[index] <= value <= hi[index]:
                            raise ValueError(
                                "current net horizontal response outside declared bounds"
                            )
                        initial[index] = value / scale[index]
            else:
                try:
                    _, prior_chart = chart.project_net_velocity(
                        source[:9],
                        scene.contact_frames[0],
                        mid,
                        scene.fps,
                        scene.surface,
                        bounce_profile=scene.bounce_profile,
                        rebound_scales=source[-2:],
                    )
                    initial_outgoing = tape.TapeResponse(0.0, 0.25, 0.4).velocity(
                        np.asarray(prior_chart["incoming_velocity_mps"])
                    )
                    initialization_receipt = dict(
                        mode="same_invocation_incoming_net_chart_passive_response",
                        source_chart=prior_chart,
                    )
                except ValueError as error:
                    initialization_receipt = dict(
                        mode="new_toss_chart_passive_response",
                        previous_source_chart_unsupported=str(error),
                    )
        initial[width : width + 3] = initial_outgoing / scale[width : width + 3]
    if toss_support is None:
        residual(initial)
    else:
        # A nearly settled original tail can cross the numerical bounce cap after
        # the contact moves. Search only for an integrable initial response, then
        # optimize the unchanged full native objective; no gate is consulted.
        trials = []
        multipliers = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
        original_vz = float(initial[width + 2] * scale[width + 2])
        for multiplier in multipliers:
            candidate = initial.copy()
            vz = original_vz * multiplier
            trial = dict(multiplier=multiplier, outgoing_vz_mps=vz)
            if not lo[width + 2] <= vz <= hi[width + 2]:
                trials.append(trial | {"status": "outside_existing_velocity_bounds"})
                continue
            candidate[width + 2] = vz / scale[width + 2]
            try:
                residual(candidate)
            except ValueError as error:
                if str(error) != "measured dynamics bounce cap reached":
                    raise
                trials.append(trial | {"status": "execution_failed", "reason": str(error)})
            else:
                trials.append(trial | {"status": "integrable"})
                initial = candidate
                break
        else:
            raise ValueError(
                f"net serve initialization exhausted bounce-cap response schedule: {trials}"
            )
        initialization_receipt = (initialization_receipt or {}) | {
            "bounce_cap_feasibility": {
                "ordered_vz_multipliers": list(multipliers),
                "trials": trials,
                "original_outgoing_vz_mps": original_vz,
                "selection": "first integrable full native objective, no acceptance gates",
                "gate_used": False,
                "native_rows_dropped": False,
            }
        }

    def jac(x):
        columns = []
        for j in range(len(x)):
            a, b = x.copy(), x.copy()
            a[j] = max(lo[j] / scale[j], x[j] - jacobian_step)
            b[j] = min(hi[j] / scale[j], x[j] + jacobian_step)
            columns.append((residual(b) - residual(a)) / (b[j] - a[j]))
        return np.column_stack(columns)

    try:
        solved = least_squares(
            residual,
            initial,
            jac=jac,
            bounds=(lo / scale, hi / scale),
            loss=loss,
            f_scale=2.0,
            max_nfev=maxiter,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        termination = dict(
            success=bool(solved.success), message=solved.message, nfev=solved.nfev, njev=solved.njev
        )
    except TimeoutError as error:
        termination = dict(success=False, message=str(error))
    return dict(
        best=best,
        termination=termination,
        residual_calls=calls,
        invalid_calls=bad,
        invalid_reasons=dict(invalid_reasons),
        initialization=initialization_receipt,
        seconds=time.monotonic() - started,
        policy=dict(
            image_loss="quadratic" if quadratic_images else "soft_l1_scale2px",
            jacobian_step=jacobian_step,
            changed_flights=changed.tolist(),
            first_contact_and_earlier_prefix_fixed=toss_support is None,
            **(
                {
                    "first_contact_toss": True,
                    "contact_interval": ci.tolist(),
                    "qualified_toss_frames": toss_frames.tolist(),
                    "toss_seed": toss_seed_receipt,
                    "toss_weight": toss_weight,
                    "contact_height_bounds_m": [2.0, 4.0],
                    "operator_aware_support": True,
                    "ordinary_net_clearance_penalty": False,
                }
                if toss_support is not None
                else {}
            ),
            joint=joint,
            original_net_interval=event["frame_interval"],
            **(
                {
                    "mesh_height_coordinate": True,
                    "mesh_fraction_bounds": [1e-5, 1.0],
                    "mesh_height_initialization": "same_invocation_net_state_or_mesh_midpoint",
                }
                if mesh_height
                else {}
            ),
            activated_frames=added,
            consumed_old_check_pixels=True,
            physics_changed=mesh_response or free_net_velocity,
            original_net_simulator_used=not (mesh_response or free_net_velocity),
            independent_outgoing_net_velocity=free_net_velocity,
            independent_outgoing_net_spin=free_net_spin,
            first_ground_normal_prior=first_ground_normal_prior,
            **(
                {
                    "first_ground_horizontal_mode": first_ground_horizontal_mode,
                    "horizontal_multiplier_prior": {
                        "center": 1.0,
                        "sigma": 0.5,
                        "bounds": [0.0, 4.0],
                    },
                    "horizontal_transverse_prior_mps": {
                        "center": 0.0,
                        "sigma": 5.0,
                        "bounds": [-20.0, 20.0],
                    }
                    if first_ground_horizontal_mode == "heading"
                    else None,
                    "horizontal_prior_is_calibrated": False,
                    "horizontal_prior_uses_native_residual_diagnosis": False,
                }
                if first_ground_horizontal_mode != "off"
                else {}
            ),
            outgoing_spin_soft_scale_rad_s=600.0 if free_net_spin else None,
            outgoing_spin_bounds_rad_s=[-1500.0, 1500.0] if free_net_spin else None,
            outgoing_spin_initialization="zero_world_spin"
            if free_net_spin
            else "preserve_incoming",
            excess_net_speed_soft_scale_mps=10.0 if free_net_velocity else None,
            outgoing_net_velocity_bounds_mps=[-75.0, 75.0] if free_net_velocity else None,
            passive_mesh_response_fitted=mesh_response,
            inclination_bounds_rad=inclination_bounds,
            response_type="cv.experiments.connected_shooting.labeled_net_spin_response.FreeNetState"
            if free_net_spin
            else "cv.experiments.connected_shooting.labeled_net_free_response.FreeNetVelocity"
            if free_net_velocity
            else f"{response_type.__module__}.{response_type.__qualname__}",
            explicit_reviewed_upper_net_height_input=upper_net,
            upper_net_offset_bounds_m=[-0.10, full.model.R_BALL + 0.03] if upper_net else None,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-response",
        action="store_true",
        help="Fit explicit passive mesh restitution/tangential retention within [.01,1]",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Allow the previous flight to move the final contact continuously",
    )
    parser.add_argument(
        "--upper-net",
        action="store_true",
        help="Explicit reviewed upper-net height band; requires supporting native evidence",
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(
        (full.paths.data_root() / "processed/ownerfix/adopted_perflight3/report.json").read_text()
    )
    item = next(r for r in base["attempts"] if r["key"] == args.case)
    rung = "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f"
    c, p, _, _, _, threshold, duration, loaded = full.load_case(dict(item=item, rung=rung))
    before, original = interval.census.score(
        c,
        p,
        threshold,
        duration,
        item["metadata"]["ending_kind"],
        scorer=interval.scope.LOCAL_SCORE,
    )
    report = dict(
        key=args.case,
        before=before,
        source_parameters=p,
        sources=[
            *loaded["sources"],
            provenance.file_record(__file__),
            provenance.file_record(chart.__file__),
            provenance.file_record(height_chart.__file__),
            provenance.file_record(tape.__file__),
        ],
        human_derived=True,
        automatic_inference_eligible=False,
        original_contact_frames=c["scene"].contact_frames,
        original_events=c["events"],
    )
    for module in [__file__, chart.__file__, height_chart.__file__, tape.__file__]:
        shutil.copy2(module, args.output / Path(module).name)
    (args.output / "start.json").write_text(
        json.dumps(full.profile.jsonable(report), indent=2) + "\n"
    )
    try:
        fitted = fit(
            c,
            p,
            duration,
            maxiter=args.maxiter,
            seconds=args.seconds,
            mesh_response=args.mesh_response,
            joint=args.joint,
            upper_net=args.upper_net,
        )
        selected = fitted["best"]["parameters"]
        original_chain = full.model.chain(c["scene"], p)
        response = fitted["best"]["response"]
        with tape.using_response(tape.TapeResponse(**response)) if response else nullcontext():
            verdict, measurement = interval.census.score(
                c,
                selected,
                threshold,
                duration,
                item["metadata"]["ending_kind"],
                scorer=interval.scope.LOCAL_SCORE,
            )
            final_chain = full.model.chain(c["scene"], selected)
        prefix_drift = max(
            float(np.max(abs(a["positions"] - b["positions"])))
            for a, b in zip(
                original_chain[: -(1 + int(args.joint))], final_chain[: -(1 + int(args.joint))]
            )
        )
        report.update(
            status="measured",
            fit=fitted,
            after=verdict,
            measurement=measurement,
            prefix_maximum_drift_m=prefix_drift,
            final_net_hits=final_chain[-1]["net_hits"],
        )
    except ValueError as error:
        report.update(status="unsupported_initial_chart", reason=str(error))
    (args.output / "report.json").write_text(
        json.dumps(full.profile.jsonable(report), indent=2) + "\n"
    )
    print(
        args.case,
        report["status"],
        report.get("after", {}).get("accepted_flight_count"),
        flush=True,
    )


if __name__ == "__main__":
    main()
