"""Reusable, opt-in joint toss/outgoing fit for an isolated labeled serve.

Status: experimental. This API receives an explicit prepared context and seed;
it never reads a cohort key, reviewed winner, label path or airborne XYZ truth.
Only one-contact, one-ground, no-net attempts qualify. Contact and first impact
remain bounded by their actual original intervals. See labeled_isolated_serve
for a CLI using ordinary search-report/packet/camera inputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import least_squares
from cv.experiments.connected_shooting import (
    full_native_continuation as full,
    camera_geometry,
    athlete_priors,
    labeled_toss_front as toss_front,
    labeled_toss_player_anchor as player_anchor_terms,
    net_constraints,
)
from cv.experiments.connected_shooting.labeled_serve_contact_bounce_fit import FirstFlightProjector
from cv.experiments.connected_shooting.labeled_preparation_net_followup import fit_check_copy

GRAVITY = np.array([0.0, 0.0, -9.81])


def interval(event: dict, name: str) -> np.ndarray:
    values = np.asarray(event.get("frame_interval", []), float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError(f"{name} requires an original finite nonzero interval")
    if not values[0] <= float(event["frame"]) <= values[1]:
        raise ValueError(f"{name} representative epoch lies outside its interval")
    return values


def qualify(context: dict, observations: dict, duration: float | None) -> dict:
    """Input-only eligibility; refuse resets/net topology rather than invent events."""
    scene = context["scene"]
    if len(scene.pixels) != 1 or len(scene.contact_frames) != 2:
        raise ValueError("isolated serve requires exactly one connected flight")
    contacts = [e for e in context["events"] if e["event_type"] == "contact"]
    bounces = [e for e in context["events"] if e["event_type"] == "bounce"]
    if len(contacts) != 1 or len(bounces) != 1:
        raise ValueError("exactly one supplied contact and one ground event required")
    role = context.get("isolated_serve_evidence", {}).get("role")
    explicit_role = next(
        (contacts[0][k] for k in ("stroke", "shot_type", "role") if k in contacts[0]), None
    )
    if explicit_role is not None and str(explicit_role).lower() != "serve":
        raise ValueError("original contact is explicitly not a serve")
    if (
        role != "serve"
        and explicit_role != "serve"
        and contacts[0].get("serve_number") not in (1, 2)
    ):
        raise ValueError("explicit original or qualified first-contact serve evidence required")
    for supplied in (context["scene"], context["heldout"]):
        supplied.validate()
    camera_geometry.scene_radial_map(context["scene"], context["heldout"])
    if any(e["event_type"] == "net_hit" for e in context["events"]) or (
        scene.net_hit_frames is not None and any(len(g) for g in scene.net_hit_frames)
    ):
        raise ValueError("declared net interaction requires a different mechanism")
    if scene.parameterization != "single_shooting" or scene.rebound_mode != "point_scales":
        raise ValueError("single-shooting point-scale parameterization required")
    ci, bi = interval(contacts[0], "contact"), interval(bounces[0], "bounce")
    if ci[1] >= bi[0] or bi[1] >= scene.contact_frames[-1]:
        raise ValueError("separate contact, impact interval and observed rebound context required")
    if observations.get("status") != "supported":
        raise ValueError("original toss observations are unsupported")
    span = 0.0 if duration is None else float(duration)
    if not np.isfinite(span) or span < 0:
        raise ValueError("nonnegative finite native exposure duration required")
    rows, excluded, seen = [], [], set()
    for raw in observations["rows"]:
        row = deepcopy(raw)
        frame = float(row["frame"])
        if not np.isfinite(frame) or frame in seen:
            raise ValueError("finite unique native toss frames required")
        seen.add(frame)
        if frame + span >= ci[0]:
            excluded.append(frame)
            continue
        P, pixel = np.asarray(row["camera"], float), np.asarray(row["pixel"], float)
        sigma = float(row["uncertainty_px"])
        if (
            P.shape != (3, 4)
            or pixel.shape != (2,)
            or not np.isfinite(P).all()
            or not np.isfinite(pixel).all()
            or not np.isfinite(sigma)
            or sigma <= 0
            or row.get("supported") is False
            or row.get("status", "supported") != "supported"
        ):
            raise ValueError(
                "supported finite native toss cameras/fronts with positive uncertainty required"
            )
        camera_geometry.radial_row(row)
        rows.append(row)
    rows.sort(key=lambda r: r["frame"])
    camera_geometry.rows_radial(rows)
    if len(rows) < 4:
        raise ValueError(
            "at least four toss fronts safely before the entire contact interval required"
        )
    return dict(
        status="eligible",
        contact=contacts[0],
        bounce=bounces[0],
        contact_interval=ci,
        bounce_interval=bi,
        rows=rows,
        excluded_toss_frames=excluded,
        release_constraint="none: release not measured",
    )


def toss_ray_seed(rows: list[dict], contact_epoch: float, fps: float, depth: float):
    """Conditional ballistic contact seed using each native camera, scale-invariant.

    Solve contact X/Z and incoming velocity with contact Y fixed to the supplied
    numerical seed. This is an initializer, not identified depth or a height prior.
    No averaging of projective camera matrices or frozen-camera pixel extrapolation.
    """
    design, rhs = [], []
    for row in rows:
        P = np.asarray(row["camera"], float)
        scale = np.linalg.norm(P[2, :3])
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("degenerate toss projection camera")
        P = P / scale
        radial = camera_geometry.radial_row(row)
        pixel = camera_geometry.undistort(
            np.asarray(row["pixel"], float)[None], None if radial is None else radial[None]
        )[0]
        A = P[:2] - pixel[:, None] * P[2]
        dt = (float(row["frame"]) - contact_epoch) / fps
        fixed = np.array([0.0, depth, 0.0]) + 0.5 * dt**2 * GRAVITY
        sigma = float(row["uncertainty_px"])
        design.append(np.c_[A[:, 0], A[:, 2], dt * A[:, :3]] / sigma)
        rhs.append((-A[:, :3] @ fixed - A[:, 3]) / sigma)
    matrix, target = np.vstack(design), np.concatenate(rhs)
    value, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != 5 or not np.isfinite(value).all():
        raise ValueError("rank-deficient conditional toss initialization")
    return (
        np.array([value[0], depth, value[1]]),
        value[2:],
        dict(
            mode="native-camera conditional ballistic toss seed",
            rank=int(rank),
            singular_values=singular,
            contact_depth_fixed_to_numerical_seed=float(depth),
            initializer_only=True,
        ),
    )


def velocity_residual(velocity, sigma_mps):
    """One velocity scale in either axis costs 16, like one supplied toss uncertainty."""
    if sigma_mps is None:
        return np.empty(0)
    if not np.isfinite(sigma_mps) or sigma_mps <= 0:
        raise ValueError("positive finite soft horizontal velocity scale required")
    return 4 * np.asarray(velocity, float)[:2] / sigma_mps


def fit(
    context: dict,
    source: np.ndarray,
    observations: dict,
    duration: float | None,
    *,
    fixed_contact_epoch: float | None = None,
    free_rebound_scales: bool = False,
    height_bounds: tuple[float, float] = (2.0, 4.0),
    maxiter: int = 80,
    seconds: float = 300.0,
    toss_horizontal_sigma_mps: float | None = None,
    incoming_toss_model: str = "center",
    player_anchor: dict | None = None,
):
    """Fit one explicit serve; return profiled evaluation context and complete candidate.

    There is no gate-based selection. The retained iterate minimizes one fixed
    native/toss/soft-physical objective. Scoring remains outside this function.
    incoming_toss_model default center keeps the original native-frame projection;
    paired_swept_front overlays original paired leading fronts only.
    """
    velocity_residual(np.zeros(3), toss_horizontal_sigma_mps)
    toss_front.assert_incoming_model(incoming_toss_model, duration)
    began = time.monotonic()
    eligible = qualify(context, observations, duration)
    if incoming_toss_model != toss_front.CENTER:
        toss_front.require_enriched(eligible["rows"])
    if incoming_toss_model == toss_front.PAIRED_OR_MOTION_SWEPT_FRONT:
        eligible["rows"] = toss_front.qualify_motion_rows(eligible["rows"])
    ctx = context
    source = np.asarray(source, float)
    if source.shape != (11,) or not np.isfinite(source).all():
        raise ValueError("finite eleven-parameter one-flight seed required")
    if maxiter < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive bounded fit budget required")
    if (
        len(height_bounds) != 2
        or not np.isfinite(height_bounds).all()
        or height_bounds[0] >= height_bounds[1]
    ):
        raise ValueError("finite ordered contact height search bounds required")
    contact = eligible["contact"]
    ci, bi, obs = eligible["contact_interval"], eligible["bounce_interval"], eligible["rows"]
    if fixed_contact_epoch is not None and not ci[0] <= fixed_contact_epoch <= ci[1]:
        raise ValueError("fixed contact must lie inside original interval")
    # Check the entire continuous profile before starting, rather than discovering
    # membership reassignment during a numerical derivative.
    for epoch in ci if fixed_contact_epoch is None else [fixed_contact_epoch]:
        full.profile.profile_context(ctx, float(epoch))
    arm = "continuous" if fixed_contact_epoch is None else "fixed"
    free_scalars = free_rebound_scales
    bundles = {}

    def bundle(epoch):
        if epoch not in bundles:
            if len(bundles) > 48:
                bundles.clear()
            c = full.profile.profile_context(ctx, epoch)
            checks, duplicate_receipt = fit_check_copy(c["scene"], c["heldout"])
            scene, axes, activated = full.merge_scene(c["scene"], checks, c["bounces"], c["axes"])
            bundles[epoch] = (
                c,
                scene,
                axes,
                activated,
                FirstFlightProjector(scene, bi, source[5]),
                net_constraints.dense_queries(scene, c["bounces"]),
            )
        return bundles[epoch]

    epochs = np.array([r["frame"] for r in obs])
    player_anchor_terms.require_rows(player_anchor, epochs)
    pixels = np.array([r["pixel"] for r in obs])
    contact0 = float(contact["frame"] if fixed_contact_epoch is None else fixed_contact_epoch)
    seed_depth = float(np.clip(source[1], *ctx["depth_bounds"]))
    xyz0, incoming0, seed_receipt = toss_ray_seed(obs, contact0, ctx["scene"].fps, seed_depth)
    seed_receipt["source_depth_clipped_m"] = seed_depth - float(source[1])
    p0 = source.copy()
    p0[:3] = xyz0
    p0[2] = np.clip(p0[2], height_bounds[0] + 1e-6, height_bounds[1] - 1e-6)
    # q: xyz, Vx,Vy, latent bounce offset, spin3, incoming toss velocity3, optional contact offset.
    q0 = np.r_[p0[:5], 0.0, p0[6:9], incoming0]
    lo = np.r_[
        0.0,
        ctx["depth_bounds"][0],
        height_bounds[0],
        -75.0,
        -75.0,
        -(bi[1] - bi[0]) / 2,
        [-6.0] * 3,
        [-12.0] * 3,
    ]
    hi = np.r_[
        10.97,
        ctx["depth_bounds"][1],
        height_bounds[1],
        75.0,
        75.0,
        (bi[1] - bi[0]) / 2,
        [6.0] * 3,
        [12.0] * 3,
    ]
    if arm == "continuous":
        q0 = np.r_[q0, contact0 - np.mean(ci)]
        lo = np.r_[lo, -(ci[1] - ci[0]) / 2]
        hi = np.r_[hi, (ci[1] - ci[0]) / 2]
    if free_scalars:
        q0 = np.r_[q0, source[-2:]]
        lo = np.r_[lo, 0.8, 0.8]
        hi = np.r_[hi, 1.2, 1.2]
    inset = np.minimum(1e-7, 0.01 * (hi - lo))
    q0 = np.clip(q0, lo + inset, hi - inset)
    calls = invalid = ambiguous_chord_calls = 0
    best = None

    def evaluate(q):
        epoch = float(np.mean(ci) + q[12]) if arm == "continuous" else contact0
        c, s, axes, activated, chart, queries = bundle(epoch)
        p = source.copy()
        p[:5] = q[:5]
        p[6:9] = q[6:9]
        if free_scalars:
            p[-2:] = q[-2:]
        p, cr = chart.project(p, float(np.mean(bi) + q[5]))
        pred = full.exposure.prediction(
            s, p, axes, duration, termination_kind=c["termination_kind"]
        )
        image = pred - np.concatenate(s.pixels)
        dt = (epochs - epoch) / s.fps
        xyz = p[:3] + dt[:, None] * q[9:12] + 0.5 * dt[:, None] ** 2 * np.array([0, 0, -9.81])
        tosspred = camera_geometry.project(
            np.array([r["camera"] for r in obs]), xyz, camera_geometry.rows_radial(obs)
        )
        if incoming_toss_model == toss_front.PAIRED_SWEPT_FRONT:
            tosspred = toss_front.overlay_paired_leading(
                obs, tosspred, p[:3], q[9:12], epoch, s.fps, duration
            )
        elif incoming_toss_model == toss_front.PAIRED_OR_MOTION_SWEPT_FRONT:
            tosspred = toss_front.overlay_motion_leading(
                obs, tosspred, p[:3], q[9:12], epoch, s.fps, duration
            )
        delta = tosspred - pixels
        sigma = np.array([r["uncertainty_px"] for r in obs])
        chain = full.model.chain(s, p, query_frames=queries)
        net, netreceipt = net_constraints.penalty(queries[0], chain[0]["positions"], 0.025)
        athlete = athlete_priors.optimization_residuals(np.array([p[:3]]), c["players"])
        anchor_part = player_anchor_terms.residuals(player_anchor, xyz[:, :2])
        residual = np.r_[
            image.ravel(),
            (4 * delta / sigma[:, None]).ravel(),
            net,
            4 * athlete,
            (p[6:9] - source[6:9]) / 6,
            velocity_residual(q[9:12], toss_horizontal_sigma_mps),
            anchor_part,
        ]
        detail = dict(
            parameters=p.copy(),
            contact_epoch=epoch,
            bounce_epoch=cr["epoch"],
            bounce_receipt=cr,
            toss_incoming_velocity_mps=q[9:12].copy(),
            toss_velocity_prior_cost=float(
                np.sum(velocity_residual(q[9:12], toss_horizontal_sigma_mps) ** 2)
            ),
            player_relative_toss=player_anchor_terms.report(
                player_anchor, p[:3], q[9:12], epochs, epoch, s.fps
            ),
            incoming_diagnostics=player_anchor_terms.incoming_diagnostics(
                p[:3], q[9:12], epochs, epoch, s.fps
            ),
            objective_terms=dict(
                native=float(np.sum(image**2)),
                toss=float(np.sum((4 * delta / sigma[:, None]) ** 2)),
                nets=float(np.sum(np.asarray(net) ** 2)),
                athletes=float(np.sum((4 * athlete) ** 2)),
                spin=float(np.sum(((p[6:9] - source[6:9]) / 6) ** 2)),
                horizontal_velocity=float(
                    np.sum(velocity_residual(q[9:12], toss_horizontal_sigma_mps) ** 2)
                ),
                player_anchor=float(anchor_part @ anchor_part),
            ),
            native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
            toss_rms_px=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
            toss_projections=[
                dict(
                    frame=r["frame"],
                    observed=r["pixel"],
                    predicted=pr,
                    error_px=float(np.linalg.norm(dd)),
                    uncertainty_px=r["uncertainty_px"],
                    incoming_operator=toss_front.operator_identity(r, incoming_toss_model),
                )
                for r, pr, dd in zip(obs, tosspred, delta, strict=True)
            ],
            net=netreceipt,
        )
        return residual, detail

    length = len(evaluate(q0)[0])

    def residual(q):
        nonlocal calls, invalid, ambiguous_chord_calls, best
        if time.monotonic() - began > seconds:
            raise TimeoutError("bounded isolated-serve wall budget")
        calls += 1
        try:
            r, d = evaluate(q)
        except ValueError as error:
            if "zero candidate exposure chord" in str(error):
                ambiguous_chord_calls += 1
            invalid += 1
            return np.full(length, 1e4)
        cost = float(r @ r)
        if not np.isfinite(cost):
            invalid += 1
            return np.full(length, 1e4)
        if best is None or cost < best["cost"]:
            best = dict(cost=cost, **d)
        return r

    def jac(q):
        cols = []
        for j in range(len(q)):
            a = q.copy()
            b = q.copy()
            a[j] = max(lo[j], q[j] - 1e-6)
            b[j] = min(hi[j], q[j] + 1e-6)
            cols.append((residual(b) - residual(a)) / (b[j] - a[j]))
        return np.column_stack(cols)

    try:
        solved = least_squares(
            residual,
            q0,
            bounds=(lo, hi),
            jac=jac,
            x_scale="jac",
            max_nfev=maxiter,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        termination = dict(
            success=bool(solved.success), message=solved.message, nfev=solved.nfev, njev=solved.njev
        )
    except TimeoutError as e:
        termination = dict(success=False, message=str(e))
    if best is None:
        raise ValueError("no finite isolated-serve candidate")
    c, s, axes, activated, chart, queries = bundle(best["contact_epoch"])
    chain = full.model.chain(c["scene"], best["parameters"])
    error = abs(chain[0]["bounces"][0]["frame"] - best["bounce_epoch"])
    if error >= 1e-7:
        raise ValueError("exported original chain did not reproduce latent first impact")
    return c, dict(
        best=best,
        initial_parameters=p0,
        seed=seed_receipt,
        termination=termination,
        calls=calls,
        invalid_calls=invalid,
        ambiguous_candidate_chord_calls=ambiguous_chord_calls,
        seconds=time.monotonic() - began,
        activated_frames=activated,
        eligibility=eligible,
        fitting_scene=asdict(s),
        export_bounce_error_frames=error,
        policy=dict(
            fixed_contact_epoch=fixed_contact_epoch,
            free_rebound_scales=free_scalars,
            rebound_scale_bounds=[0.8, 1.2] if free_scalars else None,
            spin_bounds=[-6.0, 6.0],
            contact_height_search_bounds=height_bounds,
            contact_interval=ci,
            bounce_interval=bi,
            depth_bounds=ctx["depth_bounds"],
            max_nfev=maxiter,
            wall_budget_seconds=seconds,
            loss="quadratic native fronts; toss4/sigma; athlete4; net.025m; spin/6",
            toss_horizontal_sigma_mps=toss_horizontal_sigma_mps,
            toss_velocity_regularization="4 * horizontal velocity / sigma; preference only, not observed evidence",
            player_anchor=player_anchor,
            player_anchor_prior=(
                "off: exact unchanged objective"
                if player_anchor is None
                else "shared dead-zoned whitened horizontal offset from the pooled same-camera "
                "player anchor at every qualified toss row; declared uncalibrated sigmas"
            ),
            release_prior_used=False,
            ground_position_objective_used=False,
            formerly_withheld_pixels_consumed=True,
            automatic_inference_eligible=False,
            **toss_front.policy(incoming_toss_model, duration),
        ),
    )
