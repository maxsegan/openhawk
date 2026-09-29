"""Bounded joint toss/contact/first-impact fit for labeled connected points.

Explicit experiment API; no automatic inference or gate changes. Original
contact and first-ground intervals become bounded coordinates. First Vz is
recovered by the unchanged physical impact chart. Later launch velocities,
spins and point rebound scales remain source-exact; connected later positions
may move and must be reviewed. An explicit following_launches option frees
up to two neighboring launch velocity/spin blocks; the default remains zero. The objective consumes every original native
pixel plus optional paired incoming fronts, existing player ties and net
clearance. Incoming blur is conditional modeling, not a shutter measurement.
An explicit free_first_rebound option adds only first-flight response scales;
the exported point scalars stay unchanged and replay requires the named override.
An explicit following_bounce_intervals option replaces eligible movable following
Vz coordinates with latent epochs inside their supplied bounce intervals. Each
root uses the current connected contact; the disabled path is unchanged.
An optional joint_toss_requalification lets the joint fitter's four-exposure
requirement recheck only the named upstream toss-count abstention, preserving
original observations/status and every native support check.
An optional direct_first_ground_normal adds an absolute first-ground normal COR
in [.05,1] for original ground-only points. The response follows the latent serve
epoch and is exported using the shared persistent ground-response registry.
Horizontal, spin, dwell and subsequent impact laws remain unchanged.
An optional local_boundary objective uses only movable-prefix native/player terms,
pins its internal end contact to the same invocation source, and enforces
non-net surface clearance. The full continuous tail is replayed, never spliced.
An optional bounded_front incoming initializer refines only the clipped velocity
against the actual front likelihood; the default linear_clipped path is unchanged.
An optional declared_net_chart (with preserve_later_nets) gives a movable following
flight with one whole declared net interval the net stage's epoch/mesh-height chart
coordinates in place of its Vy/Vz, so trials stay on the replayed collision branch.
Optional player_anchor / toss_horizontal_sigma_mps add the shared soft player-relative
toss conditioning (labeled_toss_player_anchor) and the existing soft horizontal
incoming-velocity residual; both default to None, the exact unchanged objective.
An optional pixel_loss="soft_l1" reweights only the native image residual block through
prefix_pixel_loss (soft-L1 at C=2 native pixels); every physical evidence term stays
quadratic and the default "off" is the exact unchanged objective and receipt.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import observation_operator, toss_witness

import time
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    athlete_priors,
    camera_geometry,
    full_native_continuation as full,
    labeled_nonnet_terminal_fit as local_rebound,
    labeled_interior_ground_response as direct_ground,
    labeled_prefix_boundary as boundary_fit,
    labeled_prefix_net_chart as net_chart,
    labeled_toss_front as toss_front,
    labeled_toss_player_anchor as player_anchor_terms,
    net_collision,
    net_constraints,
    prefix_pixel_loss,
    prefix_following_ground_timing as following_timing,
    regime_recovery,
)
from cv.experiments.connected_shooting.labeled_isolated_serve_fit import velocity_residual
from cv.experiments.connected_shooting.labeled_serve_contact_bounce_fit import FirstFlightProjector
from cv.experiments.connected_shooting.model import R_BALL
from cv.experiments.connected_shooting.labeled_interval_block_fit import FirstImpactChart

GRAVITY = np.array([0.0, 0.0, -9.81])
INVALID_RESIDUAL = 1.0e6
CACHE_LIMIT = 48
DERIVATIVE_STEP = 1.0e-6
SPARSE_TOSS_REASON = "insufficient_camera_supported_precontact_toss_observations"
Q_PHYSICAL = 10  # XYZ(3) Vx Vy(2) bounce offset(1) spin(3) contact offset(1)


class _WallTimeout(Exception):
    pass


def _interval(values, name):
    lo, hi = map(float, values)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError(f"finite positive-width original {name} interval required")
    return lo, hi


def _check_camera(camera, where):
    p = np.asarray(camera, float)
    if p.shape[-2:] != (3, 4) or not np.isfinite(p).all():
        raise ValueError(f"{where}: finite 3x4 cameras required")
    return p


def qualify(context, *, preserve_later_nets=False):
    """Read actual Scene/events topology, never infer a serve from vector length."""
    scene = context["scene"]
    n = len(scene.pixels)
    if (
        n < 2
        or scene.parameterization != "single_shooting"
        or scene.dynamics != "measured_240hz"
        or scene.rebound_mode != "point_scales"
        or scene.bounce_regime_override is not None
    ):
        raise ValueError("measured point-scale single-shooting point with n >= 2 required")
    events = context["events"]
    contacts = sorted((e for e in events if e["event_type"] == "contact"), key=lambda e: e["frame"])
    if not contacts or abs(float(contacts[0]["frame"]) - scene.contact_frames[0]) > 1e-8:
        raise ValueError("original first contact and scene epoch disagree")
    event = contacts[0]
    explicit_roles = [event[k] for k in ("stroke", "role") if event.get(k) is not None]
    if any(str(role).lower() != "serve" for role in explicit_roles):
        raise ValueError("explicit non-serve contact cannot receive a serve adapter")
    qualified_role = (context.get("prefix_serve_evidence") or {}).get("role")
    if not explicit_roles and qualified_role != "serve":
        raise ValueError("original or explicitly qualified serve evidence required")
    contact = _interval(event["frame_interval"], "contact")
    epoch = float(scene.contact_frames[0])
    if not contact[0] <= epoch <= contact[1]:
        raise ValueError("source contact outside original interval")
    if type(preserve_later_nets) is not bool:
        raise ValueError("preserve_later_nets must be an explicit boolean")
    has_nets = any(e["event_type"] == "net_hit" for e in events) or (
        scene.net_hit_frames is not None and any(len(v) for v in scene.net_hit_frames)
    )
    if has_nets and not preserve_later_nets:
        raise ValueError("declared nets are unsupported in this adapter")
    if preserve_later_nets:
        from cv.experiments.connected_shooting.labeled_prefix_later_net import qualify_later_nets

        qualify_later_nets(context)
    ground_events = [
        e
        for e in events
        if e["event_type"] == "bounce" and epoch < e["frame"] < scene.contact_frames[1]
    ]
    if len(ground_events) != 1:
        raise ValueError("exactly one original first-flight ground event required")
    ground = _interval(ground_events[0]["frame_interval"], "ground")
    if not contact[1] < ground[0] < ground[1] < scene.contact_frames[1]:
        raise ValueError("ground interval must lie strictly between contact intervals")
    depth = _interval(context["depth_bounds"], "depth bounds")
    for name in ("scene", "heldout"):
        original = context[name]
        # Consumed check rows may be empty; the fitting scene remains nonempty.
        original.validate(allow_empty_observations=name == "heldout")
        for cameras in original.cameras:
            _check_camera(cameras, name)
        # Scene.validate binds finite lens rows to each native exposure. The
        # shared swept operator projects its sphere support in native pixels.
    camera_geometry.scene_radial_map(context["scene"], context["heldout"])
    return dict(
        n=n,
        fps=float(scene.fps),
        contact_epoch=epoch,
        contact=contact,
        ground=ground,
        depth=depth,
        ground_rep=float(ground_events[0]["frame"]),
    )


def _prefix_rows(observations, contact_lo, duration, *, joint_toss_requalification=False):
    """Qualify actual incoming rows; upstream count abstentions remain recorded."""
    if type(joint_toss_requalification) is not bool:
        raise ValueError("joint_toss_requalification must be an explicit boolean")
    requalifying = observations.get("status") != "supported"
    if requalifying and not (
        joint_toss_requalification
        and observations.get("status") == "abstained"
        and observations.get("abstention_reason") == SPARSE_TOSS_REASON
    ):
        raise ValueError("supported prefix observations required")
    rows, seen = [], set()
    for original in observations["rows"]:
        row = deepcopy(original)
        if requalifying and row.get("source") != "frozen_labeled_front":
            if not (
                row.get("source") == toss_witness.AUTOMATIC_CENTER_SOURCE
                and toss_witness.qualified_automatic_center(row)
                and row.get("incoming_operator") == toss_front.CENTER
            ):
                raise ValueError(
                    "joint toss requalification requires original labeled fronts "
                    "or qualified native detector centers"
                )
        frame = float(row["frame"])
        if not np.isfinite(frame) or frame in seen:
            raise ValueError("finite unique incoming native frames required")
        seen.add(frame)
        if row.get("supported") is False or row.get("status", "supported") not in (
            "supported",
            "visible",
        ):
            raise ValueError("incoming row is not supported")
        _check_camera(row["camera"], f"prefix {frame}")
        camera_geometry.radial_row(row)
        pixel = np.asarray(row["pixel"], float)
        sigma = float(row["uncertainty_px"])
        if (
            pixel.shape != (2,)
            or not np.isfinite(pixel).all()
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("finite incoming pixel and positive uncertainty required")
        if frame + observation_operator.support_span(duration) < contact_lo:
            rows.append(row)
    if len(rows) < 4:
        raise ValueError("at least four wholly precontact exposures required")
    rows.sort(key=lambda r: r["frame"])
    camera_geometry.rows_radial(rows)
    toss_front.require_enriched(rows)
    return rows


def conditional_velocity(
    rows: Sequence[Dict[str, Any]], contact_xyz: np.ndarray, epoch: float, fps: float
) -> np.ndarray:
    """Linear camera-ray least squares for the incoming velocity at fixed contact XYZ/epoch."""
    A, b = [], []
    for row in rows:
        dt = (float(row["frame"]) - epoch) / fps
        P = np.asarray(row["camera"], dtype=float)
        radial = camera_geometry.radial_row(row)
        ideal = camera_geometry.undistort(
            np.asarray(row["pixel"], float)[None], None if radial is None else radial[None]
        )[0]
        u, v = map(float, ideal)
        anchor = np.asarray(contact_xyz, dtype=float) + 0.5 * GRAVITY * dt * dt
        for line in (P[0] - u * P[2], P[1] - v * P[2]):
            A.append(line[:3] * dt)
            b.append(-(line[:3] @ anchor + line[3]))
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def pack_params(
    source: np.ndarray, q: np.ndarray, n: int, following_launches: int = 0
) -> np.ndarray:
    """Copy the source vector with the physical q written into the first-flight slots."""
    p = np.array(source, dtype=float)
    dimensions = 10 + 6 * following_launches
    if p.shape != (5 + 6 * n,) or np.shape(q) not in ((dimensions,), (dimensions + 3,)):
        raise ValueError("full single-shooting source and 10/13 active coordinates required")
    p[0:3] = q[0:3]
    p[3:5] = q[3:5]
    p[3 + 3 * n : 6 + 3 * n] = q[6:9]
    for j in range(following_launches):
        flight, offset = j + 1, 10 + 6 * j
        p[3 + 3 * flight : 6 + 3 * flight] = q[offset : offset + 3]
        p[3 + 3 * n + 3 * flight : 6 + 3 * n + 3 * flight] = q[offset + 3 : offset + 6]
    return p


class _Bundles:
    """Bounded cache of per-epoch profiled contexts keyed by the exact epoch float."""

    def __init__(self, context: Dict[str, Any], seed_vz: float, ground: Tuple[float, float]):
        self.context, self.seed_vz, self.ground = context, float(seed_vz), ground
        self._cache: "OrderedDict[float, Dict[str, Any]]" = OrderedDict()

    def get(self, epoch: float) -> Dict[str, Any]:
        key = float(epoch)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ctx = full.profile.profile_context(self.context, key)
        scene, axes, activated = full.merge_scene(
            ctx["scene"], ctx["heldout"], ctx["bounces"], ctx["axes"]
        )
        bundle = dict(
            context=ctx,
            scene=scene,
            axes=axes,
            activated_frames=activated,
            projector=FirstFlightProjector(scene, self.ground, self.seed_vz),
            queries=net_constraints.dense_queries(scene, ctx["bounces"]),
        )
        self._cache[key] = bundle
        if len(self._cache) > CACHE_LIMIT:
            self._cache.popitem(last=False)
        return bundle


def local_evaluation_view(bundle, parameters, movable, duration):
    """Exact initial flight view when every consumed exposure stays inside it.

    A crossing exposure needs the following launch, and a terminal-net domain
    carries suffix inequalities. These modes retain the original full evaluator.
    The view is computational only: it never changes event/observation inputs.
    """
    scene = bundle["scene"]
    n = len(scene.pixels)
    if movable >= n or scene.terminal_net_tail is not None:
        return None
    end = float(scene.contact_frames[movable])
    span = observation_operator.support_span(duration)
    if any(
        len(frames) and np.max(frames) + span > end for frames in scene.observation_frames[:movable]
    ):
        return None
    fields = {
        name: getattr(scene, name)[:movable]
        for name in ("pixels", "cameras", "observation_frames", "spin_parameters")
    }
    for name in ("camera_distortion", "net_hit_frames"):
        if getattr(scene, name) is not None:
            fields[name] = getattr(scene, name)[:movable]
    view = replace(scene, contact_frames=scene.contact_frames[: movable + 1], **fields)
    p = np.asarray(parameters)
    values = np.r_[p[:3], p[3 : 3 + 3 * movable], p[3 + 3 * n : 3 + 3 * n + 3 * movable], p[-2:]]
    count = sum(len(v) for v in view.pixels)
    return dict(
        scene=view,
        parameters=values,
        axes=bundle["axes"][:count],
        queries=bundle["queries"][:movable],
    )


def _toss_prediction(
    rows: Sequence[Dict[str, Any]],
    cams: np.ndarray,
    frames: np.ndarray,
    contact_xyz: np.ndarray,
    incoming: np.ndarray,
    epoch: float,
    fps: float,
    duration: float,
) -> np.ndarray:
    dts = (frames - epoch) / fps
    xyz = contact_xyz + incoming * dts[:, None] + 0.5 * GRAVITY * dts[:, None] ** 2
    centers = camera_geometry.project(cams, xyz, camera_geometry.rows_radial(list(rows)))
    if duration is None:
        return centers
    return np.asarray(
        toss_front.overlay_paired_leading(
            rows, centers, contact_xyz, incoming, epoch, fps, duration
        ),
        float,
    )


def _jacobian(
    fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Absolute-step finite differences: central inside, one-sided when clipped by a bound."""
    columns = []
    for i in range(x.size):
        if lo[i] == hi[i]:
            columns.append(np.zeros_like(fun(x)))
            continue
        xp, xm = x.copy(), x.copy()
        xp[i] = min(x[i] + DERIVATIVE_STEP, hi[i])
        xm[i] = max(x[i] - DERIVATIVE_STEP, lo[i])
        columns.append((fun(xp) - fun(xm)) / (xp[i] - xm[i]))
    return np.stack(columns, axis=1)


def first_ground_record(scene, epoch, normal):
    """One original first ground, rebound-only; latent contact must match its active scene."""
    return direct_ground.normalize(
        dict(
            schema=direct_ground.SCHEMA,
            routes=[
                dict(
                    flight_index=0,
                    native_contact_epoch=float(epoch),
                    ground_ordinal=0,
                    normal_restitution=float(normal),
                )
            ],
        ),
        scene,
    )


def using_first_rebound(epoch: float, scales: Sequence[float] | None, *, normal=None, scene=None):
    """Worker-local explicit first-flight law; use for scoring and replay too."""
    if normal is not None:
        if scales is not None or scene is None:
            raise ValueError("direct first normal requires its active scene and no scale override")
        return direct_ground.using_response(first_ground_record(scene, epoch, normal), scene)
    return nullcontext() if scales is None else local_rebound.terminal_scales(epoch, scales)


def _project_with_rebound(projector, parameters, tau, scales):
    """Include local response in the root cache without changing point scalars."""
    if scales is None:
        return projector.project(parameters, tau)
    local = parameters.copy()
    local[-2:] = scales
    projected, receipt = projector.project(local, tau)
    projected[-2:] = parameters[-2:]
    return projected, receipt


def following_intervals(context, count, *, timing_policy="off"):
    """Inventory every movable following flight from original events, never score bits."""
    following_timing.validate(timing_policy)
    scene = context["scene"]
    inventory = []
    for flight in range(1, count + 1):
        start, end = map(float, scene.contact_frames[flight : flight + 2])
        events = [e for e in context["events"] if start < float(e["frame"]) < end]
        grounds = [e for e in events if e["event_type"] == "bounce"]
        reason = None
        if any(e["event_type"] == "net_hit" for e in events) or (
            scene.net_hit_frames is not None and len(scene.net_hit_frames[flight])
        ):
            reason = "declared net in following flight"
        elif len(grounds) != 1:
            reason = "requires exactly one supplied bounce in following flight"
        else:
            try:
                interval = _interval(grounds[0]["frame_interval"], "following bounce")
                if not start < interval[0] < interval[1] < end:
                    raise ValueError("following interval overlaps contact")
            except (KeyError, TypeError, ValueError) as exc:
                reason = str(exc)
        row = dict(flight=flight, supported=reason is None, reason=reason)
        if reason is None:
            row.update(
                interval=list(interval), center=float(np.clip(grounds[0]["frame"], *interval))
            )
            if timing_policy != "off":
                timing = following_timing.plan(grounds[0], start, end)
                row.update(interval=timing["chart_domain"], timing_policy=timing)
        inventory.append(row)
    return inventory


class ConnectedFollowingProjector:
    """Solve following Vz at the current connected start, preserving every other slot."""

    def __init__(self, scene, flight, interval, reference_vz):
        self.scene = scene
        self.shared = replace(scene, parameterization="shared_contact_states")
        self.chart = FirstImpactChart(self.shared, flight, interval)
        self.flight = flight
        self.reference_vz = float(reference_vz)

    def project(self, parameters, epoch):
        p = np.asarray(parameters, float).copy()
        # The seed constructor runs the actual single-shooting prefix. No frozen
        # contact XYZ is injected, and the chart cache includes that changed XYZ.
        shared = full.model.shared_contact_seed(self.shared, p)
        shared[self.chart.vz] = self.reference_vz
        projected, receipt = self.chart.project(shared, epoch)
        p[5 + 3 * self.flight] = projected[self.chart.vz]
        return p, dict(
            receipt,
            flight=self.flight,
            connected_start_xyz_m=shared[3 * self.flight : 3 * self.flight + 3].tolist(),
        )


def source_impact_seed(q, chain, first_interval, first_center, following):
    """Map same-invocation physical impacts into active interval coordinates.

    An out-of-interval/missing/extra impact cannot be an unchanged feasible
    incumbent. Retain ordinary chart initialization for that coordinate and
    disclose why; never clip its physical epoch and call that the source.
    """
    seed = np.asarray(q, float).copy()
    targets = [dict(flight=0, coordinate=5, interval=first_interval, center=first_center)]
    targets += [dict(row, coordinate=12 + 6 * (row["flight"] - 1)) for row in following]
    records = []
    for row in targets:
        impacts = chain[row["flight"]]["bounces"]
        epoch = float(impacts[0]["frame"]) if len(impacts) == 1 else None
        lo, hi = map(float, row["interval"])
        supported = epoch is not None and np.isfinite(epoch) and lo <= epoch <= hi
        if supported:
            seed[row["coordinate"]] = epoch - row["center"]
        records.append(
            dict(
                flight=row["flight"],
                interval=[lo, hi],
                source_epoch=epoch,
                physical_impact_count=len(impacts),
                source_epoch_used=bool(supported),
                reason=None
                if supported
                else "source does not have exactly one impact inside supplied interval",
            )
        )
        if "timing_policy" in row:
            timing = row["timing_policy"]
            original_low, original_high = timing["original_interval"]
            records[-1].update(
                interval_role="effective_chart_domain",
                original_interval=timing["original_interval"],
                source_inside_original_interval=bool(
                    epoch is not None and original_low <= epoch <= original_high
                ),
                source_inside_chart_domain=bool(supported),
            )
            if not supported:
                records[-1]["reason"] = (
                    "source does not have exactly one impact inside effective chart domain"
                )
    return seed, dict(
        mode="same_invocation_source_impacts",
        impacts=records,
        source_representable=all(row["source_epoch_used"] for row in records),
        source_is_observed_truth=False,
        gate_selection=False,
    )


def choose_initialization(evaluate, legacy_q, source_q, lo, hi, *, local_boundary):
    """Use the lowest input-cost feasible initial seed, without scoring gates.

    Both candidates see the same residual and constraints. If neither is
    feasible, preserve the original chart's deterministic restoration start.
    The best feasible initial seed dominates other initial incumbents under
    this fixed objective and enters the existing feasible-trial tracker.
    """
    values, records = [], []
    for name, q in (("legacy_chart", legacy_q), ("source_impact_chart", source_q)):
        q = np.asarray(q, float)
        record = dict(name=name, valid=False, feasible=False, cost=None, reason=None)
        value = None
        try:
            if not np.isfinite(q).all() or np.any(q < lo) or np.any(q > hi):
                raise ValueError("initial coordinates outside existing numerical bounds")
            residual, receipt = evaluate(q)
            if not np.isfinite(residual).all():
                raise ValueError("nonfinite initial residual")
            equality = np.asarray(receipt["boundary_equality_m"]) if local_boundary else np.empty(0)
            slacks = np.asarray(receipt["clearance_slacks_m"]) if local_boundary else np.empty(0)
            if not np.isfinite(equality).all() or not np.isfinite(slacks).all():
                raise ValueError("nonfinite initial physical constraints")
            feasible = boundary_fit.feasible_receipt(receipt) if local_boundary else True
            cost = float(residual @ residual)
            record.update(
                valid=True,
                feasible=bool(feasible),
                cost=cost,
                boundary_equality_m=equality.tolist(),
                clearance_slacks_m=slacks.tolist(),
            )
            value = (q.copy(), residual, receipt)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        values.append(value)
        records.append(record)
    feasible = [i for i, r in enumerate(records) if r["feasible"]]
    if feasible:
        selected = min(feasible, key=lambda i: (records[i]["cost"], i))
        rule = (
            "lowest same observed objective among feasible initial seeds; stable legacy-first ties"
        )
    else:
        selected = 0
        rule = "neither feasible; retain original legacy chart restoration start"
        if values[0] is None:
            raise ValueError(f"no feasible initialization and legacy chart invalid: {records}")
    return (
        *values[selected],
        dict(
            candidates=records, selected=records[selected]["name"], rule=rule, gate_selection=False
        ),
        values[1],
    )


def terminal_mesh_slacks(scene, chain, *, enabled):
    """Carry a fixed response's physical domain while prefix launches move."""
    if not enabled:
        return np.empty(0)
    from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
        mesh_response_slacks_m,
    )

    values = mesh_response_slacks_m(scene, chain, include_supplied_final=True)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("terminal mesh response has no finite physical support")
    return values


def retry_regime_boundaries(
    evaluate, state, termination, lo, hi, *, movable, maxiter, seconds, started, solver_kwargs
):
    """Try both sides of one observed numerical branch kink under the original law.

    The ordinary feasible incumbent always survives. All attempts share its wall
    deadline, physical constraints and observed objective; no gates select a retry.
    """
    best = state["best_receipt"]
    audit = dict(
        policy="first_movable_boundary_two_launch_spin_retries_v1",
        attempts=[],
        original_termination=termination,
        selected="baseline",
        launch_retry_rad_s=regime_recovery.LAUNCH_RETRY_RAD_S,
        same_wall_deadline=True,
        physical_law_unchanged=True,
        selection="strictly lower feasible original input objective",
    )
    try:
        boundaries = [
            r
            for r in regime_recovery.boundary_evidence(best["bundle"]["scene"], best["chain"])
            if r["flight_index"] < movable
        ]
    except (ValueError, KeyError) as error:
        audit.update(boundaries=[], reason=str(error))
        return state, termination, audit
    audit["boundaries"] = boundaries
    if not boundaries:
        return state, termination, audit
    flight = boundaries[0]["flight_index"]
    coordinate = 6 if flight == 0 else 13 + 6 * (flight - 1)
    origin = state["best_q"].copy()
    for ordinal, direction in enumerate((-1, 1)):
        elapsed = time.monotonic() - started
        if elapsed >= seconds:
            audit["stopped"] = "shared wall budget exhausted"
            break
        deadline = elapsed + (seconds - elapsed) / (2 - ordinal)
        seed = origin.copy()
        seed[coordinate] += direction * regime_recovery.LAUNCH_RETRY_RAD_S / 100.0
        row = dict(
            direction=direction,
            flight=flight,
            initial_q=seed.tolist(),
            status="held",
            deadline_seconds_from_stage_start=deadline,
        )
        audit["attempts"].append(row)
        if np.any(seed < lo) or np.any(seed > hi):
            row["reason"] = "retry outside original launch bounds"
            continue
        try:
            candidate, ended = boundary_fit.solve(
                evaluate,
                seed,
                lo,
                hi,
                jacobian=_jacobian,
                maxiter=maxiter,
                seconds=deadline,
                started=started,
                **solver_kwargs,
            )
        except (
            ValueError,
            FloatingPointError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
            TimeoutError,
        ) as error:
            row.update(reason=str(error), status="execution_failed")
            continue
        row.update(
            status="measured",
            cost=candidate["best_cost"],
            termination=ended,
            feasible_trials=candidate["feasible_trials"],
        )
        for name in ("calls", "invalid", "feasible_trials"):
            state[name] += candidate[name]
        for name in ("incumbent_certifications", "incumbent_certification_refusals"):
            if name in candidate:
                state[name] = state.get(name, 0) + candidate[name]
        if candidate["best_cost"] < state["best_cost"]:
            state.update(
                {name: candidate[name] for name in ("best_q", "best_cost", "best_receipt")}
            )
            termination = ended
            audit["selected"] = f"direction_{direction}"
    return state, termination, audit


def fit(
    context: Dict[str, Any],
    source: Sequence[float],
    observations: Dict[str, Any],
    *,
    toss_weight: float = 4.0,
    duration: float | None = 0.25,
    max_nfev: int = 80,
    seconds: float = 300.0,
    following_launches: int = 0,
    incoming_initializer: str = "linear_clipped",
    free_first_rebound: bool = False,
    direct_first_ground_normal: bool = False,
    first_rebound_initial_scales: Sequence[float] | None = None,
    preserve_later_nets: bool = False,
    following_bounce_intervals: bool = False,
    joint_toss_requalification: bool = False,
    local_boundary: bool = False,
    source_incumbent: bool = False,
    declared_net_chart: bool = False,
    terminal_mesh_response_domain: bool = False,
    player_anchor: Dict[str, Any] | None = None,
    toss_horizontal_sigma_mps: float | None = None,
    local_evaluation: bool = True,
    pixel_loss: str = "off",
    following_ground_timing: str = "off",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    began = time.monotonic()
    prefix_pixel_loss.validate(pixel_loss)
    following_timing.validate(following_ground_timing)
    robust_pixels = pixel_loss != "off"
    if type(local_evaluation) is not bool:
        raise ValueError("local_evaluation must be an explicit boolean")
    observation_operator.support_span(duration)
    velocity_residual(np.zeros(3), toss_horizontal_sigma_mps)
    if type(source_incumbent) is not bool:
        raise ValueError("source_incumbent must be an explicit boolean")
    if (
        not np.isfinite([toss_weight, seconds]).all()
        or toss_weight < 0
        or seconds <= 0
        or isinstance(max_nfev, bool)
        or int(max_nfev) != max_nfev
        or max_nfev < 1
    ):
        raise ValueError("nonnegative weight, positive duration/budget required")
    if type(declared_net_chart) is not bool:
        raise ValueError("declared_net_chart must be an explicit boolean")
    if declared_net_chart and not preserve_later_nets:
        raise ValueError("declared-net chart coordinates require preserve_later_nets")
    spec = qualify(context, preserve_later_nets=preserve_later_nets)
    n, fps, epoch0 = spec["n"], spec["fps"], spec["contact_epoch"]
    if type(following_launches) is not int or not 0 <= following_launches <= min(2, n - 1):
        raise ValueError("following_launches must be 0, 1 or 2 within original topology")
    if type(following_bounce_intervals) is not bool:
        raise ValueError("following_bounce_intervals must be an explicit boolean")
    interval_inventory = following_intervals(
        context,
        following_launches,
        timing_policy=following_ground_timing if following_bounce_intervals else "off",
    )
    interval_specs = (
        [r for r in interval_inventory if r["supported"]] if following_bounce_intervals else []
    )
    if type(local_boundary) is not bool:
        raise ValueError("local_boundary must be an explicit boolean")
    original_net_domain = context["scene"].terminal_net_tail is not None
    if type(terminal_mesh_response_domain) is not bool:
        raise ValueError("terminal_mesh_response_domain must be an explicit boolean")
    from cv.experiments.connected_shooting.observation_net_seed import single_final_net

    if terminal_mesh_response_domain and not (
        (original_net_domain or single_final_net(context["scene"]))
        and local_boundary
        and preserve_later_nets
    ):
        raise ValueError("terminal mesh response domain requires a constrained preserved net tail")
    if original_net_domain and not local_boundary:
        raise ValueError("original terminal net interval requires constrained prefix fitting")
    movable = 1 + following_launches
    net_inventory = net_chart.specs(context, movable) if declared_net_chart else []
    net_specs = [row for row in net_inventory if row["supported"]]
    incoming_offset = 10 + 6 * following_launches
    if type(free_first_rebound) is not bool:
        raise ValueError("free_first_rebound must be an explicit boolean")
    if type(direct_first_ground_normal) is not bool:
        raise ValueError("direct_first_ground_normal must be an explicit boolean")
    if direct_first_ground_normal and free_first_rebound:
        raise ValueError(
            "direct first normal and multiplicative first rebound are mutually exclusive"
        )
    if direct_first_ground_normal and (
        any(e["event_type"] == "net_hit" for e in context["events"])
        or original_net_domain
        or (
            context["scene"].net_hit_frames is not None
            and any(len(v) for v in context["scene"].net_hit_frames)
        )
    ):
        raise ValueError("direct first normal currently requires original ground-only topology")
    if first_rebound_initial_scales is not None and not free_first_rebound:
        raise ValueError("first rebound initialization requires free_first_rebound")
    src = np.array(source, dtype=float)  # private copy; the caller's array is never mutated
    if src.shape != (5 + 6 * n,) or not np.isfinite(src).all():
        raise ValueError(f"source must have length {5 + 6 * n}, got {src.shape}")
    boundary = None
    source_chain = None
    if local_boundary:
        if movable < n:
            source_chain = full.model.chain(context["scene"], src)
            boundary = np.asarray(source_chain[movable - 1]["end_xyz"], float)
    contact_lo, contact_hi = spec["contact"]
    domain_lo, domain_hi = full.profile.contact_epoch_bounds(context)
    ground_lo, ground_hi = spec["ground"]
    tau0 = float(np.clip(spec["ground_rep"], ground_lo, ground_hi))
    spin_slice = slice(3 + 3 * n, 6 + 3 * n)
    active = toss_weight > 0
    rows = _prefix_rows(
        observations,
        contact_lo,
        duration,
        joint_toss_requalification=joint_toss_requalification,
    )
    frames = np.array([float(r["frame"]) for r in rows])
    player_anchor_terms.require_rows(player_anchor, frames)
    player_terms = player_anchor is not None or toss_horizontal_sigma_mps is not None
    if player_terms and not active:
        raise ValueError("player-relative toss terms require an active toss objective")
    pixels = np.array([r["pixel"] for r in rows], dtype=float)
    sigmas = np.array([float(r["uncertainty_px"]) for r in rows])[:, None]
    cams = np.stack([np.asarray(r["camera"], dtype=float) for r in rows])
    bundles = _Bundles(context, src[5], (ground_lo, ground_hi))
    fast_local = bool(
        local_evaluation
        and local_boundary
        and not original_net_domain
        and not terminal_mesh_response_domain
        and local_evaluation_view(bundles.get(epoch0), src, movable, duration) is not None
    )

    lo = np.array(
        [
            -10.0,
            spec["depth"][0],
            R_BALL,
            -75.0,
            -75.0,
            ground_lo - tau0,
            -6.0,
            -6.0,
            -6.0,
            domain_lo - epoch0,
        ]
    )
    hi = np.array(
        [
            21.0,
            spec["depth"][1],
            12.0,
            75.0,
            75.0,
            ground_hi - tau0,
            6.0,
            6.0,
            6.0,
            domain_hi - epoch0,
        ]
    )
    q0 = np.concatenate([src[0:3], src[3:5], [0.0], src[spin_slice], [0.0]])
    for j in range(following_launches):
        flight = j + 1
        q0 = np.r_[
            q0,
            src[3 + 3 * flight : 6 + 3 * flight],
            src[3 + 3 * n + 3 * flight : 6 + 3 * n + 3 * flight],
        ]
        lo = np.r_[lo, [-75.0] * 3, [-6.0] * 3]
        hi = np.r_[hi, [75.0] * 3, [6.0] * 3]
    for row in interval_specs:
        coordinate = 12 + 6 * (row["flight"] - 1)
        q0[coordinate] = 0.0
        lo[coordinate], hi[coordinate] = np.asarray(row["interval"]) - row["center"]
    net_seed_records = []
    if net_specs:
        # Same-invocation physical net state, restored into the chart bounds and disclosed.
        seed_chain = full.model.chain(context["scene"], src)
        for row in net_specs:
            record = net_chart.seed(seed_chain, row)
            c = row["coordinate"]
            q0[c], q0[c + 1] = record["epoch_offset"], record["mesh_fraction"]
            lo[c], hi[c] = np.asarray(row["interval"], float) - row["frame"]
            lo[c + 1], hi[c + 1] = net_chart.FRACTION_BOUNDS
            net_seed_records.append(record)
    if incoming_initializer not in ("linear_clipped", "bounded_front"):
        raise ValueError("incoming_initializer must be linear_clipped or bounded_front")
    if incoming_initializer != "linear_clipped" and not active:
        raise ValueError("bounded incoming initialization requires an active toss objective")
    conditional_initial = conditional_velocity(rows, src[0:3], epoch0, fps)
    incoming_initialization = None
    initial_incoming = np.clip(conditional_initial, -12.0, 12.0)
    if incoming_initializer == "bounded_front":
        from cv.experiments.connected_shooting import labeled_toss_velocity_initializer

        initial_incoming, incoming_initialization = labeled_toss_velocity_initializer.refine(
            conditional_initial,
            lambda velocity: (
                (
                    _toss_prediction(rows, cams, frames, src[:3], velocity, epoch0, fps, duration)
                    - pixels
                )
                / sigmas
            ).ravel(),
        )
    if active:
        lo, hi = np.append(lo, [-12.0] * 3), np.append(hi, [12.0] * 3)
        q0 = np.append(q0, initial_incoming)
    rebound_offset = len(q0)
    if free_first_rebound:
        lo, hi = np.r_[lo, [0.8, 0.8]], np.r_[hi, [1.2, 1.2]]
        initial_scales = np.asarray(
            src[-2:] if first_rebound_initial_scales is None else first_rebound_initial_scales,
            float,
        )
        if initial_scales.shape != (2,) or not np.isfinite(initial_scales).all():
            raise ValueError("two finite initial first-flight rebound scales required")
        q0 = np.r_[q0, initial_scales]
    if direct_first_ground_normal:
        nominal_chain = full.model.chain(context["scene"], src)
        if not nominal_chain[0]["bounces"]:
            raise ValueError("direct first normal requires a source first ground")
        initial_normal = float(nominal_chain[0]["bounces"][0]["applied_restitution"])
        # Same physical domain as the existing interior-normal response.
        if not 0.05 <= initial_normal <= 1.0:
            raise ValueError("source normal response is outside the direct physical domain")
        q0, lo, hi = np.r_[q0, initial_normal], np.r_[lo, 0.05], np.r_[hi, 1.0]
    legacy_q0 = q0.copy()
    source_seed_receipt = None
    if source_incumbent:
        if free_first_rebound and not np.array_equal(initial_scales, src[-2:]):
            raise ValueError("source incumbent requires source rebound initialization")
        q0, source_seed_receipt = source_impact_seed(
            q0,
            full.model.chain(context["scene"], src),
            (ground_lo, ground_hi),
            tau0,
            interval_specs,
        )
        if net_specs:
            source_seed_receipt["net_hits"] = net_seed_records
            source_seed_receipt["source_representable"] = bool(
                source_seed_receipt["source_representable"]
                and all(row["source_representable"] for row in net_seed_records)
            )
    if np.any(lo > hi) or np.any(np.delete(lo >= hi, 9)):
        raise ValueError("bounds must be strictly ordered except fixed contact timing")
    if np.any(q0 < lo) or np.any(q0 > hi):
        raise ValueError(f"initial q lies outside the physical bounds: {q0}")

    source_q0 = q0.copy()

    def evaluate(q: np.ndarray, *, full_diagnostics=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        epoch, tau = epoch0 + float(q[9]), tau0 + float(q[5])
        b = bundles.get(epoch)
        scales = q[rebound_offset : rebound_offset + 2] if free_first_rebound else None
        normal = float(q[rebound_offset]) if direct_first_ground_normal else None
        physical_q = q[:rebound_offset].copy()
        for row in interval_specs:
            physical_q[12 + 6 * (row["flight"] - 1)] = src[5 + 3 * row["flight"]]
        for row in net_specs:
            c, flight = row["coordinate"], row["flight"]
            physical_q[c] = src[4 + 3 * flight]
            physical_q[c + 1] = src[5 + 3 * flight]
        p, projector_receipt = _project_with_rebound(
            b["projector"], pack_params(src, physical_q, n, following_launches), tau, scales
        )
        following_receipts = []
        net_receipts = []
        interval_by_flight = {row["flight"]: row for row in interval_specs}
        net_by_flight = {row["flight"]: row for row in net_specs}
        with using_first_rebound(epoch, scales, normal=normal, scene=b["scene"]):
            # Ascending flights: each chart starts from the current connected state.
            for flight in range(1, movable):
                if flight in interval_by_flight:
                    row = interval_by_flight[flight]
                    projectors = b.setdefault("following_projectors", {})
                    if flight not in projectors:
                        projectors[flight] = ConnectedFollowingProjector(
                            b["scene"], flight, row["interval"], src[5 + 3 * flight]
                        )
                    target = row["center"] + float(q[12 + 6 * (flight - 1)])
                    p, following_receipt = projectors[flight].project(p, target)
                    following_receipts.append(following_receipt)
                elif flight in net_by_flight:
                    row = net_by_flight[flight]
                    charts = b.setdefault("net_charts", {})
                    if flight not in charts:
                        charts[flight] = net_chart.DeclaredNetChart(b["scene"], row)
                    c = row["coordinate"]
                    p, net_receipt = charts[flight].project(p, float(q[c]), float(q[c + 1]))
                    net_receipts.append(net_receipt)
            # The verified feasible seed is the actual source, not a root's
            # numerically nearby reconstruction. Nuisance toss velocity may vary.
            source_coordinates = (
                source_seed_receipt is not None
                and source_seed_receipt["source_representable"]
                and np.array_equal(q[:incoming_offset], source_q0[:incoming_offset])
                and (scales is None or np.array_equal(scales, src[-2:]))
                and (not direct_first_ground_normal or normal == initial_normal)
            )
            if source_coordinates:
                delta = float(np.max(np.abs(p - src)))
                if delta > 1e-7:
                    raise ValueError("source impact chart failed unchanged-vector replay")
                p = src.copy()
            view = (
                local_evaluation_view(b, p, movable, duration)
                if fast_local and not full_diagnostics
                else None
            )
            active_scene = b["scene"] if view is None else view["scene"]
            active_p = p if view is None else view["parameters"]
            active_queries = b["queries"] if view is None else view["queries"]
            chain = full.model.chain(active_scene, active_p, query_frames=active_queries)
            # The unchanged collision simulator decides the branch; a missed net is invalid.
            replayed_nets = [
                net_chart.replay_check(chain, net_by_flight[r["flight"]], r["net_epoch"])
                for r in net_receipts
            ]
            pred = np.asarray(
                full.exposure.prediction(
                    active_scene,
                    active_p,
                    b["axes"] if view is None else view["axes"],
                    duration,
                    # The computational cut is not a passive ground ending.
                    # Qualified local exposures already have complete support.
                    termination_kind=b["context"]["termination_kind"]
                    if view is None
                    else "net_stop",
                ),
                float,
            )
        image = pred - np.concatenate(active_scene.pixels)
        local_rows = sum(len(v) for v in b["scene"].pixels[:movable])
        # `image` stays the raw native pixel error: it carries the reported native RMS,
        # so only this objective copy is ever reweighted.
        raw_native = image[:local_rows].ravel() if local_boundary else image.ravel()
        native = prefix_pixel_loss.transform(raw_native) if robust_pixels else raw_native
        if preserve_later_nets:
            from cv.experiments.connected_shooting.labeled_prefix_later_net import net_residuals

            nets = net_residuals(active_scene, active_queries, chain)
        else:
            nets = np.array(
                [
                    net_constraints.penalty(qf, fl["positions"], 0.025)[0]
                    for qf, fl in zip(active_queries, chain)
                ],
                float,
            )
        starts = np.array([f["start_xyz"] for f in chain], dtype=float)
        athletes = (
            4.0
            * np.asarray(
                athlete_priors.optimization_residuals(
                    starts[:movable] if local_boundary else starts,
                    b["context"]["players"][:movable]
                    if local_boundary
                    else b["context"]["players"],
                ),
                float,
            ).ravel()
        )
        spin = (q[6:9] - src[spin_slice]) / 6.0
        for j in range(following_launches):
            flight, offset = j + 1, 10 + 6 * j
            spin = np.r_[
                spin,
                (q[offset + 3 : offset + 6] - src[3 + 3 * n + 3 * flight : 6 + 3 * n + 3 * flight])
                / 6.0,
            ]
        objective_nets = (
            boundary_fit.local_net_residuals(b["scene"], b["queries"], chain, movable)
            if local_boundary
            else nets
        )
        parts = [native, objective_nets, athletes, spin]
        if free_first_rebound:
            parts.append((scales - src[-2:]) / 0.2)
        normal_prior = np.empty(0)
        if direct_first_ground_normal:
            nominal = float(chain[0]["bounces"][0]["nominal_applied_restitution"])
            normal_prior = np.array([4.0 * (normal - nominal) / 0.2])
            parts.append(normal_prior)
        toss_rms = None
        toss_raw_rms = None
        anchor_part = np.empty(0)
        velocity_part = np.empty(0)
        if active:
            toss = (
                _toss_prediction(
                    rows,
                    cams,
                    frames,
                    q[0:3],
                    q[incoming_offset : incoming_offset + 3],
                    epoch,
                    fps,
                    duration,
                )
                - pixels
            ) / sigmas
            parts.append(toss_weight * toss.ravel())
            toss_rms = float(np.sqrt(np.mean(np.sum(toss**2, axis=1))))
            toss_raw_rms = float(np.sqrt(np.mean(np.sum((toss * sigmas) ** 2, axis=1))))
            if player_terms:
                # Shared soft player-relative conditioning on the same contact-referenced
                # ballistic samples; the exact OFF objective appends nothing.
                incoming_q = q[incoming_offset : incoming_offset + 3]
                anchor_part = player_anchor_terms.residuals(
                    player_anchor,
                    player_anchor_terms.ballistic_xy(q[0:3], incoming_q, frames, epoch, fps),
                )
                velocity_part = velocity_residual(incoming_q, toss_horizontal_sigma_mps)
                parts.append(anchor_part)
                parts.append(velocity_part)
        timing_rows = []
        if following_ground_timing != "off":
            for row in interval_specs:
                timing = row["timing_policy"]
                if timing["mode"] != "prediction_hinge":
                    continue
                impact_epoch = row["center"] + float(q[12 + 6 * (row["flight"] - 1)])
                value = following_timing.hinge(impact_epoch, timing["original_interval"])
                timing_rows.append(dict(flight=row["flight"], epoch=impact_epoch, residual=value))
            # Appended after all legacy parts: response-prior indexing is unchanged.
            parts.append(np.asarray([r["residual"] for r in timing_rows], float))
        objective_terms = dict(
            native=float(native @ native),
            nets=float(objective_nets @ objective_nets),
            athletes=float(athletes @ athletes),
            spin=float(spin @ spin),
            first_rebound=float(parts[4] @ parts[4]) if free_first_rebound else 0.0,
            **(
                {"first_ground_normal": float(normal_prior @ normal_prior)}
                if direct_first_ground_normal
                else {}
            ),
            toss=float(toss_weight**2 * np.sum(toss**2)) if active else 0.0,
            player_anchor=float(anchor_part @ anchor_part),
            horizontal_velocity=float(velocity_part @ velocity_part),
            **(
                {"following_ground_timing": float(sum(r["residual"] ** 2 for r in timing_rows))}
                if following_ground_timing != "off"
                else {}
            ),
        )
        receipt = dict(
            objective_terms=objective_terms,
            p=p,
            tau=tau,
            epoch=epoch,
            chain=chain,
            bundle=b,
            projector_receipt=projector_receipt,
            following_receipts=following_receipts,
            declared_net_receipts=net_receipts,
            declared_net_replay=replayed_nets,
            native_rms=(
                float(np.sqrt(np.mean(np.sum(image**2, axis=1)))) if view is None else None
            ),
            toss_rms=toss_rms,
            toss_raw_rms=toss_raw_rms,
            net_residuals=nets.tolist() if view is None else None,
            local_evaluation_view=view is not None,
            **(
                {"following_ground_timing": timing_rows} if following_ground_timing != "off" else {}
            ),
            # On-only: objective_terms["native"] is the reweighted block cost, so the
            # unweighted sum of squares is reported separately rather than implied.
            **(
                {
                    "pixel_loss": dict(
                        mode=pixel_loss,
                        scale_px=prefix_pixel_loss.SCALE_PX,
                        raw_native_squared_error=float(raw_native @ raw_native),
                        objective_native_rows=int(raw_native.size // 2),
                    )
                }
                if robust_pixels
                else {}
            ),
        )
        if local_boundary:
            eq, slacks = boundary_fit.constraints(
                chain,
                b["queries"],
                movable,
                b["scene"].net_hit_frames if b["scene"].net_hit_frames is not None else [()] * n,
                boundary,
            )
            slacks = np.r_[
                slacks,
                [r["width_slack_m"] for r in net_receipts],
                terminal_mesh_slacks(b["scene"], chain, enabled=terminal_mesh_response_domain),
            ]
            receipt.update(boundary_equality_m=eq, clearance_slacks_m=slacks)
            if original_net_domain:
                from cv.experiments.connected_shooting.labeled_terminal_net_tail import (
                    original_domain_slacks_frames,
                    EVENT_TOLERANCE_FRAMES,
                )

                event_slacks = original_domain_slacks_frames(b["scene"], chain)
                receipt["input_event_slacks_frames"] = event_slacks
                receipt["input_event_tolerance_frames"] = EVENT_TOLERANCE_FRAMES
                # Numerical interior is only an optimizer margin, never a wider
                # observation interval or a metre-valued timing tolerance.
                receipt["input_domain_inequalities"] = np.r_[
                    slacks, event_slacks + EVENT_TOLERANCE_FRAMES - 1e-5
                ]
        return np.concatenate(parts), receipt

    def full_evaluate(q):
        return evaluate(q, full_diagnostics=True)

    def certify(q, residual, receipt):
        try:
            full_residual, full_receipt = full_evaluate(q)
            if not boundary_fit.feasible_receipt(full_receipt):
                return None
            if source_chain is not None:
                scales = q[rebound_offset : rebound_offset + 2] if free_first_rebound else None
                with using_first_rebound(
                    full_receipt["epoch"],
                    scales,
                    normal=float(q[rebound_offset]) if direct_first_ground_normal else None,
                    scene=full_receipt["bundle"]["context"]["scene"],
                ):
                    replay = full.model.chain(
                        full_receipt["bundle"]["context"]["scene"], full_receipt["p"]
                    )
                if (
                    max(
                        (
                            float(
                                np.max(
                                    np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"]))
                                )
                            )
                            for a, b in zip(replay[movable:], source_chain[movable:])
                        ),
                        default=0.0,
                    )
                    > 1e-4
                ):
                    return None
            return full_residual, full_receipt
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            return None

    if source_seed_receipt is None:
        r0, receipt0 = full_evaluate(q0)
    else:
        q0, r0, receipt0, initializations, source_value = choose_initialization(
            full_evaluate,
            legacy_q0,
            source_q0,
            lo,
            hi,
            local_boundary=local_boundary,
        )
        represented = source_seed_receipt["source_representable"]
        source_record = initializations["candidates"][1]
        source_seed_receipt.update(
            source_vector_replayed_exactly=bool(
                represented
                and source_value is not None
                and np.array_equal(source_value[2]["p"], src)
            ),
            source_feasible_incumbent=bool(represented and source_record["feasible"]),
            unchanged_source_objective_cost=source_record["cost"] if represented else None,
            selection="minimum same observed objective among valid feasible trials; no score/gate access",
            initializations=initializations,
        )
    if not np.all(np.isfinite(r0)):
        raise ValueError(
            "initial root evaluates to a non-finite objective (diagnostic, not a penalty win)"
        )
    state: Dict[str, Any] = dict(
        best_q=q0.copy(),
        best_cost=float(r0 @ r0),
        best_receipt=receipt0,
        calls=1,
        invalid=0,
        started=began,
    )

    def guarded(q: np.ndarray) -> np.ndarray:
        if time.monotonic() - state["started"] > seconds:
            raise _WallTimeout()
        state["calls"] += 1
        try:
            r, receipt = evaluate(q)
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            r, receipt = None, None
        if r is None or not np.all(np.isfinite(r)):
            state["invalid"] += 1  # evaluation guard, not a physics negative
            return np.full(r0.shape, INVALID_RESIDUAL)
        cost = float(r @ r)
        if cost < state["best_cost"]:
            state.update(best_q=q.copy(), best_cost=cost, best_receipt=receipt)
        return r

    termination: Dict[str, Any]
    regime_retry = None
    if local_boundary:
        solver_kwargs = {
            **({"inequality_key": "input_domain_inequalities"} if original_net_domain else {}),
            **({"certify_incumbent": certify} if fast_local else {}),
        }
        state, termination = boundary_fit.solve(
            evaluate,
            q0,
            lo,
            hi,
            jacobian=_jacobian,
            maxiter=max_nfev,
            seconds=seconds,
            started=began,
            **solver_kwargs,
        )
        state, termination, regime_retry = retry_regime_boundaries(
            evaluate,
            state,
            termination,
            lo,
            hi,
            movable=movable,
            maxiter=max_nfev,
            seconds=seconds,
            started=began,
            solver_kwargs=solver_kwargs,
        )
    else:
        try:
            # TRF requires positive-width bounds. Keep a singleton contact domain
            # fixed while optimizing the remaining physical coordinates.
            moving = lo < hi

            def expand(q):
                full_q = q0.copy()
                full_q[moving] = q
                return full_q

            sol = least_squares(
                lambda q: guarded(expand(q)),
                q0[moving],
                jac=lambda q: _jacobian(guarded, expand(q), lo, hi)[:, moving],
                bounds=(lo[moving], hi[moving]),
                method="trf",
                max_nfev=max_nfev,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            termination = dict(
                kind="optimizer",
                status=int(sol.status),
                message=str(sol.message),
                nfev=int(sol.nfev),
            )
        except _WallTimeout:
            termination = dict(kind="wall_timeout", seconds=seconds)
    best_q, best = state["best_q"], state["best_receipt"]
    p_best, bundle = best["p"], best["bundle"]
    local_scales = best_q[rebound_offset : rebound_offset + 2] if free_first_rebound else None
    with using_first_rebound(
        best["epoch"],
        local_scales,
        normal=float(best_q[rebound_offset]) if direct_first_ground_normal else None,
        scene=bundle["context"]["scene"],
    ):
        free_chain = full.model.chain(
            bundle["context"]["scene"], p_best
        )  # original evaluation scene, natural impact regime
    free_tau = float(free_chain[0]["bounces"][0]["frame"])
    if original_net_domain:
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import require_event_domain

        require_event_domain(bundle["context"]["scene"], free_chain)
    if abs(free_tau - best["tau"]) > 1e-7:
        raise ValueError(
            f"unforced chain first bounce {free_tau} does not reproduce fitted tau {best['tau']}"
        )
    following_readback = []
    for row in best["following_receipts"]:
        impacts = free_chain[row["flight"]]["bounces"]
        actual = float(impacts[0]["frame"]) if impacts else None
        if actual is None or abs(actual - row["epoch"]) > 1e-7:
            raise ValueError("unforced following flight does not reproduce its interval chart")
        following_readback.append(
            dict(flight=row["flight"], epoch=actual, error_frames=actual - row["epoch"])
        )
    net_readback = []
    for row, receipt in zip(net_specs, best["declared_net_receipts"], strict=True):
        check = net_chart.replay_check(free_chain, row, receipt["net_epoch"])
        flight_end = float(context["scene"].contact_frames[row["flight"] + 1])
        declared_grounds = sorted(
            (
                e
                for e in context["events"]
                if e["event_type"] == "bounce" and row["frame"] < float(e["frame"]) < flight_end
            ),
            key=lambda e: float(e["frame"]),
        )
        check["declared_following_grounds"] = len(declared_grounds)
        if declared_grounds and check["following_ground_count"] == 0:
            raise ValueError("declared-net chart replay has no ground impact after the net")
        if declared_grounds and check["first_following_ground_frame"] is not None:
            low, high = map(float, declared_grounds[0]["frame_interval"])
            check["first_following_ground_inside_declared_interval"] = bool(
                low <= check["first_following_ground_frame"] <= high
            )
        net_readback.append(check)
    if not active:
        incoming = conditional_velocity(rows, best_q[0:3], best["epoch"], fps)
        toss = (
            _toss_prediction(
                rows, cams, frames, best_q[0:3], incoming, best["epoch"], fps, duration
            )
            - pixels
        ) / sigmas
        best["toss_rms"] = float(np.sqrt(np.mean(np.sum(toss**2, axis=1))))
        best["toss_raw_rms"] = float(np.sqrt(np.mean(np.sum((toss * sigmas) ** 2, axis=1))))
    else:
        incoming = best_q[incoming_offset : incoming_offset + 3]
    original_chain = full.model.chain(context["scene"], src)
    shifts = [
        dict(
            flight=i,
            start=(np.asarray(f["start_xyz"]) - np.asarray(g["start_xyz"])).tolist(),
            end=(np.asarray(f["end_xyz"]) - np.asarray(g["end_xyz"])).tolist(),
        )
        for i, (f, g) in enumerate(zip(free_chain, original_chain))
    ]
    later_same = bool(
        np.array_equal(
            p_best[6 + 3 * following_launches : 3 + 3 * n],
            src[6 + 3 * following_launches : 3 + 3 * n],
        )
        and np.array_equal(
            p_best[6 + 3 * n + 3 * following_launches :], src[6 + 3 * n + 3 * following_launches :]
        )
    )
    if local_boundary:
        with using_first_rebound(
            best["epoch"],
            local_scales,
            normal=float(best_q[rebound_offset]) if direct_first_ground_normal else None,
            scene=bundle["scene"],
        ):
            dense_readback = full.model.chain(
                bundle["scene"], p_best, query_frames=bundle["queries"]
            )
        _, replay_slacks = boundary_fit.constraints(
            dense_readback,
            bundle["queries"],
            movable,
            bundle["scene"].net_hit_frames
            if bundle["scene"].net_hit_frames is not None
            else [()] * n,
            boundary,
        )
        replay_slacks = np.r_[
            replay_slacks,
            terminal_mesh_slacks(
                bundle["scene"], dense_readback, enabled=terminal_mesh_response_domain
            ),
        ]
        if np.min(replay_slacks, initial=1.0) < -boundary_fit.TOLERANCE_M:
            raise ValueError("original dense replay violates physical net feasibility")
        boundary_error = (
            0.0
            if boundary is None
            else float(np.max(np.abs(np.asarray(free_chain[movable - 1]["end_xyz"]) - boundary)))
        )
        if boundary_error > boundary_fit.TOLERANCE_M:
            raise ValueError("original physical replay violates the fitted boundary")
        tail_errors = [
            float(np.max(np.abs(np.asarray(f["positions"]) - np.asarray(g["positions"]))))
            for f, g in zip(free_chain[movable:], original_chain[movable:])
        ]
        if max(tail_errors, default=0.0) > 1e-4:
            raise ValueError("fixed tail changed materially after constrained physical replay")
    if not later_same:
        raise RuntimeError("adapter changed a frozen later launch or rebound scalar")
    result = dict(
        automatic_inference_eligible=False,
        conditional_initial_incoming_velocity=conditional_initial.tolist(),
        incoming_initial_clipped_to_numerical_box=bool(
            active and np.any(abs(conditional_initial) > 12)
        ),
        incoming_frames=frames.tolist(),
        incoming_qualification=dict(
            policy_enabled=joint_toss_requalification,
            upstream_status=observations.get("status"),
            upstream_abstention_reason=observations.get("abstention_reason"),
            upstream_minimum_observations=observations.get("minimum_observations"),
            upstream_observation_count=len(observations.get("rows", [])),
            joint_minimum_wholly_precontact_exposures=4,
            qualified_frames=frames.tolist(),
            requalified=observations.get("status") != "supported",
            original_observations_and_status_unchanged=True,
            scope="named upstream observation-count abstention only; all original row/exposure checks retained",
        ),
        incoming_prediction_px=_toss_prediction(
            rows, cams, frames, best_q[:3], incoming, best["epoch"], fps, duration
        ).tolist(),
        fit_policy=dict(
            blur="model conditional, not measured shutter",
            heldout="all native heldout consumed by merge_scene",
            inference="none automatic",
            gate_selection="none",
            ground_xy_prior="none",
            release_prior="none",
            intervals=dict(
                first_contact=[contact_lo, contact_hi],
                optimizer_contact_domain=[domain_lo, domain_hi],
                first_ground=[ground_lo, ground_hi],
                depth=spec["depth"],
            ),
            active_indices=dict(
                xyz=[0, 1, 2],
                first_vx_vy=[3, 4],
                first_spin=list(range(3 + 3 * n, 6 + 3 * n)),
                first_vz_via_projector=[5],
                incoming=(
                    list(range(incoming_offset, incoming_offset + 3)) if active else "inactive"
                ),
                following_flights=list(range(1, following_launches + 1)),
            ),
            following_launches=following_launches,
            frozen_later_velocities_spins_scalars_bit_identical=later_same,
            later_velocities_spins_scalars_bit_identical=bool(
                np.array_equal(p_best[6 : 3 + 3 * n], src[6 : 3 + 3 * n])
                and np.array_equal(p_best[6 + 3 * n :], src[6 + 3 * n :])
            ),
            bounds=dict(lo=lo.tolist(), hi=hi.tolist()),
            toss_weight=toss_weight,
            duration=duration,
            player_anchor_prior=(
                "off: exact unchanged objective"
                if player_anchor is None
                else "shared dead-zoned whitened horizontal offset from the pooled same-camera "
                "player anchor at every qualified toss row; declared uncalibrated sigmas"
            ),
            toss_horizontal_sigma_mps=toss_horizontal_sigma_mps,
            local_evaluation=dict(
                enabled=local_evaluation,
                qualified=fast_local,
                scope="movable prefix only when all consumed exposures stay inside it; no terminal domain",
                retained_state_certification="full objective, constraints and original fixed-tail replay",
                incumbent_certifications=state.get("incumbent_certifications", 0),
                incumbent_certification_refusals=state.get("incumbent_certification_refusals", 0),
                diagnostic_scope="native RMS and net residuals retain full-chain meanings",
            ),
            optimizer="least_squares trf, absolute central finite differences 1e-6",
            budget="max_nfev limits optimizer residual evaluations, excluding numerical Jacobian calls; wall cap also applies",
            max_nfev=max_nfev,
            seconds=seconds,
        ),
        termination=termination,
        **({"regime_retries": regime_retry} if regime_retry is not None else {}),
        calls=state["calls"],
        invalid_trials=state["invalid"],
        invalid_note="invalid trials are evaluation guards, never scored as best and never physics negatives",
        q=best_q.tolist(),
        full_vector=p_best.tolist(),
        initial_source=src.tolist(),
        initial_q=q0.tolist(),
        contact_epoch=best["epoch"],
        contact_epoch_original=epoch0,
        tau=best["tau"],
        tau_original=tau0,
        roots_receipt=dict(
            projector=best["projector_receipt"],
            unforced_first_bounce=free_tau,
            activated_frames=bundle["activated_frames"],
        ),
        cost=state["best_cost"],
        initial_chart_cost=float(r0 @ r0),
        native_rms_px=best["native_rms"],
        toss_weighted_radial_rms=best["toss_rms"],
        toss_raw_radial_rms_px=best["toss_raw_rms"],
        net_residuals=best["net_residuals"],
        incoming_velocity=np.asarray(incoming, float).tolist(),
        incoming_source=("fitted" if active else "conditional after fit"),
        objective_terms=best["objective_terms"],
        incoming_diagnostics=player_anchor_terms.incoming_diagnostics(
            best_q[0:3], incoming, frames, best["epoch"], fps
        ),
        player_relative_toss=player_anchor_terms.report(
            player_anchor, best_q[0:3], incoming, frames, best["epoch"], fps
        )
        | dict(
            anchor=player_anchor,
            toss_horizontal_sigma_mps=toss_horizontal_sigma_mps,
            horizontal_velocity_cost=best["objective_terms"]["horizontal_velocity"],
        ),
        endpoint_shifts_from_original_source=shifts,
        initial_chart_native_rms_px=receipt0["native_rms"],
        initial_chart_toss_weighted_radial_rms=receipt0["toss_rms"],
    )
    if local_boundary:
        result["fit_policy"]["local_boundary"] = dict(
            enabled=True,
            movable_flights=list(range(movable)),
            boundary_source="same invocation current physical state; regularization, not truth",
            boundary_xyz_m=None if boundary is None else boundary.tolist(),
            boundary_error_m=boundary_error,
            tail_max_position_error_m=max(tail_errors, default=0.0),
            objective="local native/toss/player/spin residuals; all downstream paths replayed",
            physical_constraints="all in-width non-net plane crossings, ball surface >= 0",
            clearance_slacks_m=best["clearance_slacks_m"].tolist(),
            feasible_trials=state["feasible_trials"],
            initial_boundary_jacobian_rank=state["initial_boundary_jacobian_rank"],
            initial_boundary_error_m=state["initial_boundary_error_m"],
            initial_clearance_slacks_m=state["initial_clearance_slacks_m"],
            objective_scale=state["objective_scale"],
        )
        result["fit_policy"]["heldout"] = (
            "movable-prefix native heldout consumed; fixed tail retained for full replay and scoring"
        )
        result["fit_policy"]["optimizer"] = "SLSQP constrained local prefix; central FD 1e-6"
        result["fit_policy"]["budget"] = "max_nfev is primary SLSQP iteration cap; same wall cap"
    result["fit_policy"]["following_bounce_intervals"] = dict(
        enabled=following_bounce_intervals,
        inventory=interval_inventory,
        scope="original supplied epochs replace Vz; objective and original law unchanged",
        unsupported="original velocity coordinate retained, no invented event",
    )
    if following_ground_timing != "off":
        result["fit_policy"]["following_ground_timing"] = dict(
            mode=following_ground_timing,
            status="applied" if best["following_ground_timing"] else "not_applicable",
            following_charts_enabled=following_bounce_intervals,
            scope="charted following grounds in this prefix stage only; first ground unchanged",
            inventory=interval_inventory,
            fitted=best["following_ground_timing"],
            remaining_hard_domains="first ground and other refinement stages unchanged",
            original_source_events_unchanged=True,
            gates_unchanged=True,
            observations_unchanged=True,
        )
        if best["following_ground_timing"]:
            result["fit_policy"]["following_bounce_intervals"]["scope"] = (
                "supplied ground occurrence replaces Vz; generated prediction timing uses "
                "contact chronology plus the existing soft interval hinge"
            )
    if robust_pixels:
        result["fit_policy"]["pixel_loss"] = dict(
            enabled=True,
            mode=pixel_loss,
            upstream_search_rank="unchanged",
            scale_px=prefix_pixel_loss.SCALE_PX,
            scope="native image residual block only; toss, player, net, athlete, spin and "
            "rebound/normal evidence stay quadratic",
            paths="one transformed residual vector feeds initialization, source incumbent, "
            "regime retries, SLSQP and local/full certification",
            reporting="objective_terms.native and every cost field are the reweighted objective; "
            "native RMS, projections and gates keep raw pixel meanings",
        )
        result["pixel_loss"] = dict(
            **best["pixel_loss"],
            initial_chart_raw_native_squared_error=receipt0["pixel_loss"][
                "raw_native_squared_error"
            ],
        )
    result["roots_receipt"]["following"] = best["following_receipts"]
    result["roots_receipt"]["following_unforced_readback"] = following_readback
    if terminal_mesh_response_domain:
        result["fit_policy"]["terminal_mesh_response_domain"] = dict(
            enabled=True,
            policy="existing_mesh_response_slacks_m",
            final_dense_replay_slacks_m=terminal_mesh_slacks(
                bundle["scene"], dense_readback, enabled=True
            ).tolist(),
            near_tape_flexibility_unchanged=True,
        )
    if declared_net_chart:
        result["fit_policy"]["declared_net_chart"] = dict(
            enabled=True,
            inventory=net_inventory,
            active_flights=[row["flight"] for row in net_specs],
            coordinates="following-launch Vy/Vz slots hold net epoch offset and mesh fraction",
            fraction_bounds=list(net_chart.FRACTION_BOUNDS),
            seeds=net_seed_records,
            replay=net_readback,
            collision_policy=net_collision.active_policy(),
            branch="unchanged collision simulator replays every trial; no hit at the chart epoch is an invalid trial",
            width_inequality="net half-width slack appended to clearance constraints under local_boundary",
            exported_vector="physical projected Vy/Vz; replay needs no chart",
        )
        result["roots_receipt"]["declared_net"] = best["declared_net_receipts"]
    if preserve_later_nets:
        result["fit_policy"]["preserve_later_nets"] = dict(
            enabled=True,
            original_events_and_native_rows_unchanged=True,
            declared_net_residuals="existing net_collision.residuals; no mesh-clearance penalty on declared net flights",
            first_flight_net_supported=False,
        )
        if original_net_domain:
            result["fit_policy"]["preserve_later_nets"].update(
                declared_net_residuals="original interval hinge; unchanged collision geometry",
                original_interval_hard_constraint=True,
                input_event_slacks_frames=best["input_event_slacks_frames"].tolist(),
                optimizer_interior_frames=1e-5,
                unforced_event_domain_verified=True,
                occurrence_inferred=False,
                net_present="assumed",
                net_absent="not_evaluated",
            )
    if direct_first_ground_normal:
        result["initial_ground_response"] = None
        result["ground_response"] = first_ground_record(
            bundle["context"]["scene"], best["epoch"], best_q[rebound_offset]
        )
        result["fit_policy"]["direct_first_ground_normal"] = dict(
            enabled=True,
            bounds=[0.05, 1.0],
            initial_normal=initial_normal,
            prior="4*(normal-current incoming nominal)/.2; same as interior-normal fitting",
            scope="original first ground only; horizontal/spin/dwell/later grounds unchanged",
            original_epoch=epoch0,
            active_epoch=best["epoch"],
            source="same invocation source first impact, no external fitted input",
            replay="persistent ground_response under original flight/ground identity",
        )
    if free_first_rebound:
        result["first_flight_rebound_scales"] = local_scales.tolist()
        result["fit_policy"]["first_flight_rebound"] = dict(
            enabled=True,
            bounds=[0.8, 1.2],
            prior_sigma=0.2,
            prior_center=src[-2:].tolist(),
            scope="all court impacts within first flight only; later response uses original point scales",
            replay="using_first_rebound(contact_epoch, first_flight_rebound_scales)",
            root_cache="local scales included in projector key; exported point scales restored",
        )
        if first_rebound_initial_scales is not None:
            result["fit_policy"]["first_flight_rebound"]["explicit_initial_scales"] = (
                initial_scales.tolist()
            )
    if source_seed_receipt is not None:
        source_seed_receipt.update(
            source_retained=bool(
                np.array_equal(p_best, src)
                and (not direct_first_ground_normal or best_q[rebound_offset] == initial_normal)
            ),
            selected_objective_cost=float(state["best_cost"]),
        )
        if source_seed_receipt["source_feasible_incumbent"] and (
            state["best_cost"] > source_seed_receipt["unchanged_source_objective_cost"] + 1e-7
        ):
            raise RuntimeError("feasible source incumbent was lost by input-objective selection")
        result["source_incumbent"] = source_seed_receipt
    if incoming_initialization is not None:
        result["incoming_initialization"] = incoming_initialization
        result["fit_policy"]["incoming_initializer"] = (
            "bounded_front; fixed contact conditional warm start"
        )
    return bundle["context"], result
