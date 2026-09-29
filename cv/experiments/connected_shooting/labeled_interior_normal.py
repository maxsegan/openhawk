"""Bounded interior first-ground normal-response block on the current cold state.

One common input-qualified policy: inventory every original three-flight ground
triple in native chronology, fit only the first eligible triple, and defer the
rest explicitly. The block keeps the current fixed left start and right endpoint,
propagates its two inner contacts continuously, frees per-flight Vx/Vy, the
first-ground epoch chart and launch spin, and frees the three first-ground normal
CORs directly in [0.05, 1]. An opt-in translation-only retention coordinate uses
original two-wing support and the same current-incoming nominal prior for every
trial. Spin, heading, dwell and later-ground laws remain unchanged.
When the nominal seed cannot enter the original single-ground branch (a block
flight grounds twice before its next contact), one deterministic per-flight
nominal-to-upper normal ladder raises only that flight's seed, in chronology,
until every block flight grounds once; boundary restoration then starts there.
An input-admissible source remains the incumbent under the same input objective;
a feasible candidate must improve its cost. An inadmissible source is retained
only when no replay-valid feasible candidate is available. No gates, case keys,
fitted external inputs, file IO or model calls are consulted at runtime.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import observation_operator

from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import time

import numpy as np
from physics import bounce_reference
from scipy.optimize import minimize

from cv.experiments.connected_shooting import (
    athlete_priors,
    interior_contact_epochs,
    player_position,
    labeled_interior_ground_response as ground,
    labeled_prefix_boundary as boundary,
    labeled_prefix_joint_impact as prefix,
    labeled_preparation_net_followup as followup,
    measured_dynamics,
    net_constraints,
)
from cv.experiments.connected_shooting.labeled_interval_block_fit import FirstImpactChart
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response

NUMERICAL_ERRORS = (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError)
BLOCK = 3
COORDINATES_PER_FLIGHT = 6  # Vx Vy first-ground epoch, three launch spin coordinates.
NORMAL_BOUNDS = (0.05, 1.0)
NORMAL_PRIOR_WEIGHT = 4.0
NORMAL_PRIOR_SIGMA = 0.2
SPIN_PRIOR_SCALE = 6.0
FALLBACK_NORMAL_SEED = 0.75  # Seed only; used when no chart trial can report a nominal.
LADDER_RUNGS = 4  # Fixed seed rungs from each flight's own seed up to the upper bound.
INCUMBENT_RELATIVE_TOLERANCE = 1e-9
BOUNDARY_EQUALITY_TOLERANCE_M = 1e-5  # Numerical closure; the full chain is always replayed.
SUFFIX_TOLERANCE_M = 1e-4
EPOCH_TOLERANCE_FRAMES = 1e-7
POLICY = dict(
    selection="first eligible original ground triple in native chronology",
    block_flights=BLOCK,
    direct_normal_bounds=list(NORMAL_BOUNDS),
    normal_prior="4*(e - nominal_applied_at_current_incoming)/0.2 on every trial",
    horizontal_spin_response=False,
    fitted_vector_ranking=False,
    budget_escalation=False,
    seed_restoration=(
        "deterministic per-flight nominal-to-upper normal ladder, native chronology, "
        "only while a block flight grounds twice; then boundary feasibility restoration"
    ),
    ladder_rungs=LADDER_RUNGS,
    iteration_budget="maxiter per phase: feasibility restoration, then image fitting",
    wall_budget="one shared deadline including preparation and both solver phases",
    new_labels=False,
    boundary_equality_tolerance_m=BOUNDARY_EQUALITY_TOLERANCE_M,
    clearance_tolerance_m=boundary.TOLERANCE_M,
    propagated_suffix_tolerance_m=SUFFIX_TOLERANCE_M,
    file_io=False,
    model_calls=0,
)


# ----------------------------------------------------------------------------
# Original observed inventory


def _flight_rows(context, duration):
    span = observation_operator.support_span(duration)
    scene, heldout = context["scene"], context["heldout"]
    scene.validate()
    heldout.validate()
    if (
        scene.parameterization != "single_shooting"
        or scene.dynamics != "measured_240hz"
        or scene.rebound_mode != "point_scales"
        or scene.bounce_regime_override is not None
    ):
        raise ValueError("measured single-shooting point scales and native exposure required")
    if not np.array_equal(scene.contact_frames, heldout.contact_frames):
        raise ValueError("original training/check contact epochs differ")
    if len(context["bounces"]) != len(scene.pixels):
        raise ValueError("one original physical ground inventory per flight required")
    original, events = interior_contact_epochs.original_contact_inventory(context, duration)
    rows = []
    for i, (start, end) in enumerate(zip(scene.contact_frames[:-1], scene.contact_frames[1:])):
        inside = [e for e in events if original[i] < float(e["frame"]) < original[i + 1]]
        grounds = [e for e in inside if e["event_type"] == "bounce"]
        reason, interval, epoch = None, None, None
        incoming, rebound = [], []
        if any(e["event_type"] in ("net_hit", "contact") for e in inside) or any(
            s.net_hit_frames is not None and len(s.net_hit_frames[i]) for s in (scene, heldout)
        ):
            reason = "original net/contact inside ground-only flight"
        elif len(grounds) != 1:
            reason = "exactly one original ground interval required"
        else:
            try:
                interval = prefix._interval(grounds[0]["frame_interval"], "interior ground")
                if not start < interval[0] < interval[1] < end:
                    raise ValueError("ground interval overlaps original contact")
                epoch = float(grounds[0]["frame"])
                declared = np.asarray(context["bounces"][i], float)
                if declared.shape != (1,) or abs(declared[0] - epoch) > 1e-8:
                    raise ValueError("original event and physical ground inventory disagree")
                frames = np.unique(
                    np.r_[scene.observation_frames[i], heldout.observation_frames[i]]
                )
                incoming = frames[frames + span < interval[0]].tolist()
                rebound = frames[frames > interval[1]].tolist()
                if len(incoming) < 2 or len(rebound) < 2:
                    reason = "fewer than two original native exposures on one ground wing"
            except (KeyError, TypeError, ValueError) as error:
                reason, interval, epoch = str(error), None, None
        rows.append(
            dict(
                flight=i,
                reason=reason,
                interval=None if interval is None else list(interval),
                event_epoch=epoch,
                initial_epoch=None if interval is None else float(np.clip(epoch, *interval)),
                incoming_frames=incoming,
                rebound_frames=rebound,
            )
        )
    return rows


def inventory(context, duration, *, allow_pairs=False, allow_sparse_wings=False):
    """Original triples, optionally isolated ground pairs; no fit-based scheduling."""
    if type(allow_pairs) is not bool:
        raise ValueError("allow_pairs must be an explicit boolean")
    if type(allow_sparse_wings) is not bool:
        raise ValueError("allow_sparse_wings must be an explicit boolean")
    rows = _flight_rows(context, duration)
    original, events = interior_contact_epochs.original_contact_inventory(context, duration)
    contacts = [float(e["frame"]) for e in events if e["event_type"] == "contact"]

    def block_record(first, size):
        block = rows[first : first + size]
        sparse_reason = "fewer than two original native exposures on one ground wing"
        sparse = [r for r in block if r["reason"] == sparse_reason]
        supported = [r for r in block if r["reason"] is None]
        # The other two ground flights and fixed outer XYZ endpoints constrain
        # the shared contact geometry. No observation or event is synthesized.
        sparse_admitted = bool(
            allow_sparse_wings
            and size == BLOCK
            and len(sparse) == 1
            and len(supported) == BLOCK - 1
            and len(set(sparse[0]["incoming_frames"] + sparse[0]["rebound_frames"])) >= 4
        )
        reasons = [
            f"flight {r['flight']}: {r['reason']}"
            for r in block
            if r["reason"] and not (sparse_admitted and r["reason"] == sparse_reason)
        ]
        for epoch in original[first : first + size + 1]:
            if not any(abs(c - float(epoch)) < 1e-8 for c in contacts):
                reasons.append(f"boundary epoch {float(epoch)} is not an original racket contact")
        record = dict(
            first=first,
            flights=[first + j for j in range(size)],
            eligible=not reasons,
            reasons=reasons,
            grounds=block,
        )
        if allow_sparse_wings:
            record["sparse_wing_admission"] = dict(
                qualified=sparse_admitted and not reasons,
                flights=[r["flight"] for r in sparse] if sparse_admitted else [],
                evidence=[
                    dict(
                        flight=r["flight"],
                        position_in_block=r["flight"] - first,
                        incoming_native_epochs=r["incoming_frames"],
                        rebound_native_epochs=r["rebound_frames"],
                        unique_objective_wing_epochs=len(
                            set(r["incoming_frames"] + r["rebound_frames"])
                        ),
                    )
                    for r in sparse
                ],
                policy="one sparse ground wing in a triple with two fully supported flights; at least four original wing exposures in that flight",
                uncertainty="shared geometry and bounded nominal response regularize missing evidence; no independent depth or bounce-response certificate",
            )
        return record

    triples = [block_record(first, BLOCK) for first in range(1, len(rows) - BLOCK + 1)]
    if allow_pairs:
        covered = {i for row in triples if row["eligible"] for i in row["flights"]}
        for first in range(1, len(rows) - 1):
            if covered.intersection((first, first + 1)):
                continue
            pair = block_record(first, 2)
            if pair["eligible"]:
                triples.append(pair)
        triples.sort(key=lambda row: (row["first"], -len(row["flights"])))
    return triples


# ----------------------------------------------------------------------------
# Soft priors


def horizontal_wing_support(grounds, window):
    """Only original unique observations on both sides can free translation."""
    rows = []
    for i, g in zip(window, grounds, strict=True):
        before = sorted(set(map(float, g["incoming_frames"])))
        after = sorted(set(map(float, g["rebound_frames"])))
        rows.append(
            dict(
                flight=i,
                incoming_native_frames=before,
                rebound_native_frames=after,
                supported=len(before) >= 2 and len(after) >= 2,
            )
        )
    return rows


def horizontal_residuals(chain, support, surface, point_scale):
    """Same residual for incumbent and trial; identically zero under nominal law."""
    sigma = max(
        0.10,
        bounce_reference.MEASURED[surface]["retention_residual_std_at_model_spin"]
        * float(point_scale),
    )
    residuals = []
    for row in support:
        if not row["supported"]:
            continue
        impacts = chain[row["flight"]]["bounces"]
        if len(impacts) != 1:
            residuals.append(0.0)
            continue
        impact = impacts[0]
        applied = float(impact["applied_horizontal_retention"])
        nominal = float(impact.get("nominal_applied_horizontal_retention", applied))
        residuals.append((applied - nominal) / sigma)
    return np.asarray(residuals, float)


def indexed_player_residuals(starts, players, moving_contact):
    """Keep original serve identity; never reinterpret a rally slice as contact0."""
    residuals = athlete_priors.optimization_residuals(np.asarray(starts), players)
    return 4.0 * residuals[1 + moving_contact : 2 + moving_contact]


def contact_cues(context, labels, cameras, rows, scale):
    """Broad observed-player root axial depth cue; abstains without native support."""
    players = context.get("players", [])
    if labels is None or cameras is None or rows is None:
        return [
            dict(index=i, status="abstained", reason="no observed player-depth inputs supplied")
            for i in range(len(players))
        ]
    clip = cameras["clip"]
    cm = {int(r["frame"]): r for r in cameras["cameras"]}
    ball = {
        int(r["frame"]): r
        for g in labels["ball"]["records"]
        if g["clip"] == clip
        for r in g["frames"]
    }
    out = []
    original_contacts = None
    for index, player in enumerate(players):
        root = player_position.root_xy(player)
        if root is None:
            out.append(
                dict(index=index, status="abstained", reason="player position explicitly absent")
            )
            continue
        if original_contacts is None:
            duration = context.get("observation_operator", {}).get("exposure_duration_frames")
            original_contacts, original_events = interior_contact_epochs.original_contact_inventory(
                context, duration
            )
        # Timing changes the active physical state, not the native event bracket
        # used to associate existing camera/ball/player evidence with this contact.
        epoch = float(original_contacts[index])
        candidates = [
            e
            for e in original_events
            if e["event_type"] == "contact"
            and float(e["frame"]) == epoch
            and e.get("frame_interval") is not None
        ]
        if len(candidates) != 1 or player.get("side") is None:
            out.append(dict(index=index, status="abstained", reason="no unique original contact"))
            continue
        lo, hi = map(float, candidates[0]["frame_interval"])
        evidence = []
        for f in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            matches = [
                r
                for r in rows
                if r["clip"] == clip
                and r["frame"] == f"f_{f:04d}.jpg"
                and r.get("side") == player["side"]
            ]
            cam = cm.get(f, {})
            if (
                len(matches) != 1
                or ball.get(f, {}).get("status") != "visible"
                or cam.get("status") != "supported"
            ):
                continue
            r = matches[0]
            if not np.isfinite(float(r["conf"])) or float(r["conf"]) < 0.25:
                continue
            p = np.asarray(cam["P"], float)
            center = -np.linalg.solve(p[:, :3], p[:, 3])
            axis = root - center[:2]
            axis /= np.linalg.norm(axis)
            evidence.append(
                dict(
                    frame=f,
                    camera_center=center.tolist(),
                    axial_direction=axis.tolist(),
                    box=[float(r[k]) * scale for k in ("x0", "y0", "x1", "y1")],
                    ball_front=[ball[f]["x1080"], ball[f]["y1080"]],
                )
            )
        if not evidence:
            out.append(
                dict(
                    index=index,
                    status="abstained",
                    reason="no same-bracket visible front and supported sided box/camera",
                )
            )
            continue
        axis = np.mean([e["axial_direction"] for e in evidence], axis=0)
        axis /= np.linalg.norm(axis)
        out.append(
            dict(
                index=index,
                status="supported",
                root_xy=root.tolist(),
                axis=axis.tolist(),
                sigma_m=1.0,
                pixel_weight=12.0,
                evidence=evidence,
                wrist_used=False,
                height_observation=False,
            )
        )
    return out


def depth_residuals(starts, cues, indices):
    values = []
    for i in indices:
        cue = cues[i] if i < len(cues) else dict(status="abstained")
        if cue["status"] != "supported":
            continue
        u = float(np.dot(starts[i, :2] - cue["root_xy"], cue["axis"])) / cue["sigma_m"]
        # Square equals standard twice-Huber loss, quadratic inside one sigma.
        loss = u * u if abs(u) <= 1 else 2 * abs(u) - 1
        values.append(cue["pixel_weight"] * np.sign(u) * np.sqrt(loss))
    return np.asarray(values, float)


def native_objective_rows(scene, first, duration, *, block_size=BLOCK):
    """Block exposures plus preceding fronts whose actual exposure crosses the contact."""
    offsets = np.cumsum([0, *[len(g) for g in scene.pixels]])
    frames = np.concatenate(scene.observation_frames)
    start = float(scene.contact_frames[first])
    mask = np.zeros(len(frames), bool)
    mask[offsets[first] : offsets[first + block_size]] = True
    mask |= (frames < start) & (frames + observation_operator.support_span(duration) > start)
    return np.flatnonzero(mask)


# ----------------------------------------------------------------------------
# Connected block chart


class BlockProjector:
    """Sequential original-law roots; each following start is the actual propagated end."""

    def __init__(self, scene, source, first, grounds, original_chain):
        self.scene, self.first = scene, first
        self.n = len(scene.pixels)
        self.shared_scene = replace(scene, parameterization="shared_contact_states")
        self.slices = prefix.full.model.shared_parameter_slices(self.shared_scene)
        self.fixed_start = np.asarray(original_chain[first]["start_xyz"], float).copy()
        self.reference_vz = [float(source[5 + 3 * (first + j)]) for j in range(len(grounds))]
        self.charts = [
            FirstImpactChart(self.shared_scene, first + j, row["interval"])
            for j, row in enumerate(grounds)
        ]

    def clear(self):
        """Chart receipts (second impact) depend on the response; never reuse them."""
        for chart in self.charts:
            chart.cache.clear()

    def project(self, parameters, epochs):
        p = np.asarray(parameters, float).copy()
        shared = np.zeros(self.slices["rebound_scales"].stop)
        shared[self.slices["velocities"]] = p[3 : 3 + 3 * self.n]
        shared[self.slices["spins"]] = p[3 + 3 * self.n : -2]
        shared[self.slices["rebound_scales"]] = p[-2:]
        start, receipts = self.fixed_start.copy(), []
        for j, chart in enumerate(self.charts):
            i = self.first + j
            shared[3 * i : 3 * i + 3], shared[chart.vz] = start, self.reference_vz[j]
            projected, receipt = chart.project(shared, float(epochs[j]))
            p[5 + 3 * i] = projected[chart.vz]
            receipts.append(dict(receipt, flight=i, connected_start_xyz_m=start.tolist()))
            if j + 1 < len(self.charts):
                a, b = self.scene.contact_frames[i : i + 2]
                state = np.r_[
                    start,
                    p[3 + 3 * i : 6 + 3 * i],
                    p[3 + 3 * self.n + 3 * i : 6 + 3 * self.n + 3 * i],
                ]
                xyz, *_ = measured_dynamics.simulate(
                    state,
                    float(a),
                    np.asarray([a, b]),
                    self.scene.fps,
                    self.scene.surface,
                    bounce_profile=self.scene.bounce_profile,
                    rebound_scales=p[-2:],
                )
                start = xyz[-1].copy()
        return p, receipts


class BranchError(ValueError):
    """A charted block flight grounded more than once before its next contact."""

    def __init__(self, flight, block_position, grounds):
        super().__init__(
            f"block flight {flight} left the original single-ground branch: "
            f"{grounds} grounds before its next contact"
        )
        self.flight, self.block_position, self.grounds = flight, block_position, grounds


def _rungs(seed):
    """Fixed rungs strictly above one flight's seed, ending at the upper normal bound."""
    upper = NORMAL_BOUNDS[1]
    return [seed + k * (upper - seed) / LADDER_RUNGS for k in range(1, LADDER_RUNGS + 1)]


def _seed_ladder(evaluate, q0, physical, window, record):
    """Raise only a twice-grounding flight's normal seed until the chart is entered.

    Deterministic and bounded: each step moves one flight one rung up its own
    ladder, earlier flights first, so at most BLOCK*LADDER_RUNGS extra trials.
    Any other chart error is left to boundary restoration unchanged. ``record``
    is filled in place so an exhausted ladder still reports its steps.
    """
    q = np.asarray(q0, float).copy()
    steps = record["steps"] = []
    record.update(status="running", entry_error=None)
    for _ in range(len(window) * LADDER_RUNGS + 1):
        try:
            evaluate(q)
        except BranchError as error:
            if error.grounds <= 1:
                record.update(
                    status="chart_not_entered",
                    entry_error=f"{type(error).__name__}: {error}",
                )
                break
            j = error.block_position
            higher = [r for r in _rungs(float(q0[physical + j])) if r > q[physical + j] + 1e-12]
            if not higher:
                record.update(status="exhausted", entry_error=str(error))
                raise ValueError(
                    f"seed ladder exhausted at the upper normal bound; {error}"
                ) from error
            q[physical + j] = higher[0]
            steps.append(
                dict(flight=int(window[j]), grounds=int(error.grounds), normal=float(higher[0]))
            )
            record["seed_normals"] = q[physical : physical + len(window)].tolist()
            continue
        except TimeoutError as error:
            record.update(
                status="timed_out",
                entry_error=str(error),
                seed_normals=q[physical : physical + len(window)].tolist(),
            )
            raise
        except NUMERICAL_ERRORS as error:
            record.update(
                status="chart_not_entered", entry_error=f"{type(error).__name__}: {error}"
            )
            break
        record["status"] = "entered" if steps else "not_needed"
        break
    else:
        record["status"] = "step_bound_exceeded"
        raise ValueError("seed ladder exceeded its fixed step bound")
    record["seed_normals"] = q[physical : physical + len(window)].tolist()
    return q


def _feasible(receipt):
    return (
        np.max(np.abs(receipt["boundary_equality_m"]), initial=0) <= BOUNDARY_EQUALITY_TOLERANCE_M
        and np.min(receipt["clearance_slacks_m"], initial=1) >= -boundary.TOLERANCE_M
    )


def _block_constraints(chain, queries, first, grounds, target, fps):
    """Identical input constraints for the original source and fitted trials."""
    block_size = len(grounds)
    eq = np.asarray(chain[first + block_size - 1]["end_xyz"], float) - target
    _, clearance = boundary.constraints(
        chain[first : first + block_size],
        queries[first : first + block_size],
        block_size,
        [[] for _ in range(block_size)],
        None,
    )
    intervals = []
    for j, g in enumerate(grounds):
        impacts = chain[first + j]["bounces"]
        if len(impacts) > 1:
            raise BranchError(first + j, j, len(impacts))
        if not impacts:
            raise ValueError("charted block flight has no ground impact")
        tau = float(impacts[0]["frame"])
        intervals.extend([tau - g["interval"][0], g["interval"][1] - tau])
    # Preserve the existing trial constraint units and tolerances. Source
    # admissibility must not use the stricter seed-epoch `inside` bookkeeping.
    return dict(
        boundary_equality_m=eq,
        clearance_slacks_m=np.r_[clearance, np.asarray(intervals) / fps],
    )


def _restore(
    evaluate,
    q0,
    lo,
    hi,
    started,
    seconds,
    maxiter,
    size,
    *,
    feasible=_feasible,
    inequality_key="clearance_slacks_m",
):
    """Generic boundary feasibility restoration; stops at the first feasible trial."""
    calls, hits, invalid = 0, 0, 0
    entry_error = None
    try:
        initial = evaluate(q0)[1]
    except NUMERICAL_ERRORS as error:
        initial, entry_error = None, f"{type(error).__name__}: {error}"
    if initial is not None and feasible(initial):
        return q0, dict(status="initial_feasible", calls=0, cache_hits=0, invalid_trials=0)
    cache = OrderedDict()
    best = None

    class Feasible(Exception):
        pass

    def residual(q):
        nonlocal calls, hits, best, invalid
        if time.monotonic() - started > seconds:
            raise TimeoutError("interior normal feasibility wall budget reached")
        key = np.asarray(q, float).tobytes()
        if key in cache:
            hits += 1
            return cache[key]
        calls += 1
        try:
            _, receipt = evaluate(q)
            result = np.r_[receipt["boundary_equality_m"], np.minimum(receipt[inequality_key], 0)]
            if feasible(receipt):
                best = np.asarray(q, float).copy()
                raise Feasible()
        except NUMERICAL_ERRORS:
            invalid += 1
            result = np.full(size, 1e6)
        cache[key] = result
        if len(cache) > 48:
            cache.popitem(last=False)
        return result

    try:
        solution = minimize(
            lambda q: float(residual(q) @ residual(q)),
            q0,
            jac=lambda q: 2 * prefix._jacobian(residual, q, lo, hi).T @ residual(q),
            method="SLSQP",
            bounds=list(zip(lo, hi)),
            options=dict(maxiter=maxiter, ftol=1e-14),
        )
        residual(solution.x)
    except Feasible:
        return best, dict(
            status="first_feasible_trial",
            calls=calls,
            cache_hits=hits,
            invalid_trials=invalid,
            entry_error=entry_error,
        )
    raise ValueError(
        f"no feasible interior restoration; {solution.message}; calls={calls}, "
        f"cache_hits={hits}, invalid={invalid}, entry_error={entry_error}"
    )


def _scope(net, record, scene):
    """Compose the unrelated net response with the complete ground registry."""
    stack = ExitStack()
    if net is not None:
        stack.enter_context(using_response(net))
    stack.enter_context(ground.using_response(record, scene))
    return stack


def _replay_errors(replay, original):
    return [
        float(np.max(np.abs(a["positions"] - b["positions"])))
        for a, b in zip(replay, original, strict=True)
    ]


# ----------------------------------------------------------------------------
# Fit


def fit(
    context,
    source,
    duration,
    *,
    ground_response=None,
    net_response=None,
    labels=None,
    cameras=None,
    pose_rows=None,
    pose_image_scale=1.0,
    seconds=180.0,
    maxiter=60,
    _first=None,
    allow_pairs=False,
    allow_sparse_wings=False,
    restoration_geometry_only=False,
    horizontal_retention=False,
):
    """Refine the first eligible triple or isolated pair, retaining the source on failure.

    ``maxiter`` bounds each solver phase; ``seconds`` is shared across preparation,
    feasibility restoration and image fitting. Neither phase extends that deadline.
    ``restoration_geometry_only`` skips image/priors only for infeasible restoration
    trials; every feasible candidate still evaluates the unchanged full objective.
    """
    started = time.monotonic()
    if type(horizontal_retention) is not bool:
        raise ValueError("horizontal_retention must be an explicit boolean")
    if type(restoration_geometry_only) is not bool:
        raise ValueError("restoration_geometry_only must be an explicit boolean")
    if type(allow_pairs) is not bool:
        raise ValueError("allow_pairs must be an explicit boolean")
    if not np.isfinite(seconds) or seconds <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError("positive common budgets required")
    if not np.isfinite(pose_image_scale) or pose_image_scale <= 0:
        raise ValueError("positive pose image scale required")
    if type(allow_sparse_wings) is not bool:
        raise ValueError("allow_sparse_wings must be an explicit boolean")
    options = {"allow_pairs": True} if allow_pairs else {}
    if allow_sparse_wings:
        options["allow_sparse_wings"] = True
    rows = inventory(context, duration, **options)
    n = len(context["scene"].pixels)
    src = np.asarray(source, float).copy()
    if src.shape != (5 + 6 * n,) or not np.isfinite(src).all():
        raise ValueError("finite full single-shooting source vector required")
    initial_record = ground.normalize(deepcopy(ground_response), context["scene"])
    net = response_from_record(net_response) if net_response is not None else None
    eligible = [r for r in rows if r["eligible"]]
    details = dict(
        full_vector=src.tolist(),
        initial_source=src.tolist(),
        ground_response=deepcopy(initial_record),
        initial_ground_response=deepcopy(initial_record),
        retained_net_response=deepcopy(net_response),
        status="source_retained",
        inventory=rows,
        selected_first=None,
        deferred_eligible_firsts=[r["first"] for r in eligible[1:]],
        gate_selection=False,
        external_fitted_inputs_used=False,
        policy=dict(
            POLICY,
            seconds=float(seconds),
            maxiter=maxiter,
            restoration_geometry_only=restoration_geometry_only,
        ),
    )
    if allow_sparse_wings:
        details["policy"]["allow_sparse_wings"] = True
    if not eligible:
        if _first is not None:
            raise ValueError("requested block is not an original eligible triple")
        details.update(
            reason="no eligible original ground triple", wall_seconds=time.monotonic() - started
        )
        return context, details
    if _first is None:
        spec = next((row for row in eligible if len(row["flights"]) == BLOCK), eligible[0])
        details["deferred_eligible_firsts"] = [
            row["first"] for row in eligible if row["first"] != spec["first"]
        ]
    else:
        if type(_first) is not int:
            raise ValueError("original eligible block index required")
        matches = [row for row in eligible if row["first"] == _first]
        if len(matches) != 1:
            raise ValueError("requested block is not an original eligible triple")
        spec = matches[0]
        details["deferred_eligible_firsts"] = [
            row["first"] for row in eligible if row["first"] != _first
        ]
        details["policy"]["selection"] = "original eligible index supplied by chronological sweep"
    first, grounds = spec["first"], spec["grounds"]
    block_size = len(grounds)
    window = list(range(first, first + block_size))
    inner = list(range(first + 1, first + block_size))
    if block_size != BLOCK:
        details["policy"].update(
            block_flights=block_size, selection="original isolated ground pair"
        )
    details["selected_first"] = first
    details["selected_flights"] = window
    cues = contact_cues(context, labels, cameras, pose_rows, pose_image_scale)
    details["cues"] = [{k: v for k, v in c.items() if k != "evidence"} for c in cues]
    players = context["players"]
    if len(players) != n:
        raise ValueError("one original player root per flight required")

    checks, consumed = followup.fit_check_copy(context["scene"], context["heldout"])
    scene, axes, _ = prefix.full.merge_scene(
        context["scene"], checks, context["bounces"], context["axes"]
    )
    queries = net_constraints.dense_queries(scene, context["bounces"])
    image_rows = native_objective_rows(scene, first, duration, block_size=block_size)
    image_target = np.concatenate(scene.pixels)[image_rows]
    horizontal_support = horizontal_wing_support(grounds, window) if horizontal_retention else []
    if horizontal_retention:
        details["policy"].update(
            horizontal_translation_response="absolute_retention_0.05_to_1",
            horizontal_prior="(actual - nominal at current incoming)/surface sigma with 0.10 floor",
            contact_epochs_changed=False,
        )
        details["horizontal_support"] = horizontal_support
    existing = {r["flight_index"]: r for r in (initial_record["routes"] if initial_record else [])}

    def block_residual(p, chain, pred):
        image = np.asarray(pred)[image_rows] - image_target
        starts = np.asarray([f["start_xyz"] for f in chain])
        reach = np.concatenate([indexed_player_residuals(starts, players, i) for i in inner])
        depth = depth_residuals(starts, cues, inner)
        spin = np.concatenate(
            [
                (
                    p[3 + 3 * n + 3 * i : 6 + 3 * n + 3 * i]
                    - src[3 + 3 * n + 3 * i : 6 + 3 * n + 3 * i]
                )
                / SPIN_PRIOR_SCALE
                for i in window
            ]
        )
        normal = []
        for i in window:
            impacts = chain[i]["bounces"]
            if len(impacts) != 1:
                normal.append(0.0)
                continue
            b = impacts[0]
            nominal = b.get("nominal_applied_restitution", b["applied_restitution"])
            normal.append(
                NORMAL_PRIOR_WEIGHT * (b["applied_restitution"] - nominal) / NORMAL_PRIOR_SIGMA
            )
        normal = np.asarray(normal, float)
        horizontal = horizontal_residuals(chain, horizontal_support, scene.surface, src[-1])
        residual = np.r_[image.ravel(), reach, depth, spin, normal, horizontal]
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite interior normal block residual")
        components = dict(
            image=float(np.sum(image**2)),
            reach=float(reach @ reach),
            depth=float(depth @ depth),
            spin=float(spin @ spin),
            normal=float(normal @ normal),
        )
        if horizontal_retention:
            components["horizontal"] = float(horizontal @ horizontal)
        return residual, components, starts

    # Score the unchanged source under exactly the same objective. Only a source
    # satisfying the trial's input constraints may veto a feasible replacement
    # on cost; an infeasible source remains the explicit fallback on failure.
    with _scope(net, initial_record, scene):
        original = prefix.full.model.chain(scene, src, query_frames=queries)
        source_prediction = prefix.full.exposure.prediction(
            scene, src, axes, duration, termination_kind=context["termination_kind"]
        )
    source_residual, source_components, _ = block_residual(src, original, source_prediction)
    source_cost = float(source_residual @ source_residual)
    target = np.asarray(original[first + block_size - 1]["end_xyz"], float)
    source_admissibility = dict(admissible=False)
    try:
        source_constraints = _block_constraints(
            original, queries, first, grounds, target, scene.fps
        )
        source_admissibility.update(
            admissible=bool(_feasible(source_constraints)),
            boundary_error_m=source_constraints["boundary_equality_m"].tolist(),
            constraint_slacks=source_constraints["clearance_slacks_m"].tolist(),
            minimum_slack=float(np.min(source_constraints["clearance_slacks_m"])),
        )
        if not source_admissibility["admissible"]:
            source_admissibility["reason"] = "source violates original block physical constraints"
    except NUMERICAL_ERRORS as error:
        source_admissibility["reason"] = f"{type(error).__name__}: {error}"
    details["source_admissibility"] = source_admissibility
    details["feasible_output"] = source_admissibility["admissible"]
    projector = BlockProjector(scene, src, first, grounds, original)

    seeds, inside = [], []
    for j, g in enumerate(grounds):
        impacts = original[first + j]["bounces"]
        lo, hi = g["interval"]
        current = float(impacts[0]["frame"]) if len(impacts) == 1 else None
        ok = current is not None and lo <= current <= hi
        inside.append(bool(ok))
        seeds.append(current if ok else float(np.clip(g["event_epoch"], lo, hi)))
    physical = block_size * COORDINATES_PER_FLIGHT
    q0, lo, hi = [], [], []
    frozen = np.ones(len(src), bool)
    for j, i in enumerate(window):
        v, w = 3 + 3 * i, 3 + 3 * n + 3 * i
        q0.extend(np.r_[src[v : v + 2], seeds[j], src[w : w + 3]])
        lo.extend([-75.0, -75.0, grounds[j]["interval"][0], -6.0, -6.0, -6.0])
        hi.extend([75.0, 75.0, grounds[j]["interval"][1], 6.0, 6.0, 6.0])
        frozen[v : v + 3] = frozen[w : w + 3] = False
    q0, lo, hi = map(np.asarray, (q0, lo, hi))
    if np.any(q0 < lo) or np.any(q0 > hi):
        raise ValueError("source interior coordinates outside common bounds")

    def routes(cors, retentions=None):
        return [
            dict(
                flight_index=i,
                native_contact_epoch=float(scene.contact_frames[i]),
                ground_ordinal=0,
                normal_restitution=float(c),
                **(
                    {"horizontal_retention": float(retentions[i])}
                    if retentions and i in retentions
                    else {}
                ),
            )
            for i, c in zip(window, cors, strict=True)
        ]

    def physical_vector(q):
        p = src.copy()
        for j, i in enumerate(window):
            v, w = 3 + 3 * i, 3 + 3 * n + 3 * i
            p[v : v + 2] = q[6 * j : 6 * j + 2]
            p[w : w + 3] = q[6 * j + 3 : 6 * j + 6]
        return p

    # Seed normals: existing declared routes keep their values; otherwise the
    # nominal applied at the current incoming, read from the source chain or
    # from one nominal-law projection of the seed when the source lacks a single
    # ground in a block flight.
    normal_seed, seed_kinds = [], []
    seed_chain = None
    for j, i in enumerate(window):
        if i in existing:
            normal_seed.append(existing[i]["normal_restitution"])
            seed_kinds.append("existing_route")
            continue
        impacts = original[i]["bounces"]
        if len(impacts) == 1:
            b = impacts[0]
            normal_seed.append(
                float(b.get("nominal_applied_restitution", b["applied_restitution"]))
            )
            seed_kinds.append("source_nominal")
            continue
        if seed_chain is None:
            try:
                projector.clear()
                with _scope(net, initial_record, scene):
                    seeded, _ = projector.project(physical_vector(q0), seeds)
                    seed_chain = prefix.full.model.chain(scene, seeded, query_frames=queries)
            except (*NUMERICAL_ERRORS, TimeoutError) as error:
                seed_chain = f"{type(error).__name__}: {error}"
        impacts = seed_chain[i]["bounces"] if isinstance(seed_chain, list) else []
        if len(impacts) == 1:
            b = impacts[0]
            normal_seed.append(
                float(b.get("nominal_applied_restitution", b["applied_restitution"]))
            )
            seed_kinds.append("seed_projection_nominal")
        else:
            normal_seed.append(FALLBACK_NORMAL_SEED)
            seed_kinds.append("fallback_placeholder_seed")
    normal_seed = np.clip(np.asarray(normal_seed, float), *NORMAL_BOUNDS)
    q0 = np.r_[q0, normal_seed]
    lo = np.r_[lo, [NORMAL_BOUNDS[0]] * block_size]
    hi = np.r_[hi, [NORMAL_BOUNDS[1]] * block_size]
    horizontal_flights, horizontal_seeds = [], []
    for row in horizontal_support:
        if not row["supported"]:
            row["active"] = False
            row["reason"] = "insufficient original two-wing evidence"
            continue
        impacts = original[row["flight"]]["bounces"]
        if not impacts and isinstance(seed_chain, list):
            impacts = seed_chain[row["flight"]]["bounces"]
        seed = float(impacts[0]["applied_horizontal_retention"]) if impacts else None
        if seed is None or not np.isfinite(seed) or not 0.05 <= seed <= 1:
            row.update(
                active=False, reason="no bounded physical initial retention", source_retention=seed
            )
            continue
        row.update(active=True, source_retention=seed)
        horizontal_flights.append(row["flight"])
        horizontal_seeds.append(seed)
    if horizontal_retention:
        q0 = np.r_[q0, horizontal_seeds]
        lo = np.r_[lo, [0.05] * len(horizontal_seeds)]
        hi = np.r_[hi, [1.0] * len(horizontal_seeds)]
        details["horizontal_parameter_count"] = len(horizontal_seeds)
        details["horizontal_active_flights"] = horizontal_flights
    constraint_size = 3 + block_size + 2 * block_size

    completed = None
    counters = dict(
        evaluations=0,
        invalid_trials=0,
        feasible_trials=0,
        trial_projection_evaluations=0,
        infeasible_restoration_projection_skips=0,
    )
    active_key = None

    def evaluate(q, *, restoration_only=False):
        nonlocal completed, active_key
        if time.monotonic() - started > seconds:
            raise TimeoutError("interior normal block wall budget reached")
        counters["evaluations"] += 1
        cors = [float(v) for v in q[physical : physical + block_size]]
        retentions = dict(zip(horizontal_flights, q[physical + block_size :], strict=True))
        record = ground.merge(initial_record, routes(cors, retentions), scene)
        key = (*cors, *retentions.values())
        if key != active_key:
            projector.clear()
            active_key = key
        try:
            with _scope(net, record, scene):
                p, roots = projector.project(physical_vector(q), q[2:physical:6])
                chain = prefix.full.model.chain(scene, p, query_frames=queries)
                defer_projection = restoration_geometry_only and restoration_only
                if not defer_projection:
                    counters["trial_projection_evaluations"] += 1
                    pred = prefix.full.exposure.prediction(
                        scene, p, axes, duration, termination_kind=context["termination_kind"]
                    )
            constraints = _block_constraints(chain, queries, first, grounds, target, scene.fps)
            eq, slack = constraints["boundary_equality_m"], constraints["clearance_slacks_m"]
            receipt = dict(
                p=p,
                chain=chain,
                roots=roots,
                boundary_equality_m=eq,
                clearance_slacks_m=slack,
                response=record,
                normals=cors,
                **({"horizontal_retentions": retentions} if horizontal_retention else {}),
            )
            if defer_projection:
                if not np.isfinite(np.r_[eq, slack]).all():
                    raise ValueError("nonfinite interior restoration constraints")
                # Restoration consumes only these physical constraints. A
                # geometrically feasible trial still must evaluate the complete
                # original objective before it can reach either incumbent.
                if not _feasible(receipt):
                    counters["infeasible_restoration_projection_skips"] += 1
                    receipt["objective_evaluated"] = False
                    return np.empty(0), receipt
                with _scope(net, record, scene):
                    counters["trial_projection_evaluations"] += 1
                    pred = prefix.full.exposure.prediction(
                        scene, p, axes, duration, termination_kind=context["termination_kind"]
                    )
            residual, components, starts = block_residual(p, chain, pred)
            receipt.update(components=components, contact_xyz=starts[inner].tolist())
        except NUMERICAL_ERRORS:
            counters["invalid_trials"] += 1
            raise
        cost = float(residual @ residual)
        if _feasible(receipt):
            counters["feasible_trials"] += 1
            if completed is None or cost < completed[0]:
                completed = (cost, np.asarray(q, float).copy(), receipt)
        return residual, receipt

    restoration = dict(status="not_started")
    ladder = dict(status="not_started")
    termination = None
    solver = None
    initial_chart = None
    try:
        try:
            initial_residual, initial_receipt = evaluate(q0)
            initial_chart = dict(
                cost=float(initial_residual @ initial_residual),
                components=initial_receipt["components"],
                feasible=bool(_feasible(initial_receipt)),
                boundary_error_m=initial_receipt["boundary_equality_m"].tolist(),
                minimum_slack=float(np.min(initial_receipt["clearance_slacks_m"])),
            )
        except NUMERICAL_ERRORS as error:
            initial_chart = dict(
                status="chart_not_entered", reason=f"{type(error).__name__}: {error}"
            )
        seeded = _seed_ladder(evaluate, q0, physical, window, ladder)
        restoration_evaluate = (
            (lambda q: evaluate(q, restoration_only=True))
            if restoration_geometry_only
            else evaluate
        )
        restored, restoration = _restore(
            restoration_evaluate, seeded, lo, hi, started, seconds, maxiter, constraint_size
        )
        state, termination = boundary.solve(
            evaluate,
            restored,
            lo,
            hi,
            jacobian=prefix._jacobian,
            feasible=_feasible,
            maxiter=maxiter,
            seconds=seconds,
            started=started,
        )
        solver = {k: state.get(k) for k in ("calls", "invalid", "feasible_trials")}
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

    details.update(
        objective=dict(
            source_cost=source_cost,
            source_components=source_components,
            initial_chart=initial_chart,
            candidate_cost=None if completed is None else completed[0],
            candidate_components=None if completed is None else completed[2]["components"],
            relative_tolerance=INCUMBENT_RELATIVE_TOLERANCE,
        ),
        feasibility=dict(
            seed_ladder=ladder,
            restoration=restoration,
            source_inside_interval=inside,
            seed_epochs=seeds,
            seed_normals=normal_seed.tolist(),
            seed_normal_kinds=seed_kinds,
            solver=solver,
            **counters,
        ),
        termination=termination,
        consumed_check_rows=consumed,
        native_objective_frames=np.concatenate(scene.observation_frames)[image_rows].tolist(),
    )
    if completed is None:
        details.update(reason="no feasible block trial", wall_seconds=time.monotonic() - started)
        return context, details
    cost, q, best = completed
    improved = cost < source_cost - INCUMBENT_RELATIVE_TOLERANCE * max(1.0, source_cost)
    if source_admissibility["admissible"] and not improved:
        details.update(
            reason="feasible candidate not better than the source incumbent",
            wall_seconds=time.monotonic() - started,
        )
        return context, details
    p = best["p"]
    record = best["response"]
    with _scope(net, record, scene):
        replay = prefix.full.model.chain(scene, p, query_frames=queries)
    errors = _replay_errors(replay, original)
    verification = []
    if not np.array_equal(p[frozen], src[frozen]):
        verification.append("candidate changed frozen coordinates")
    if max(errors[:first], default=0) != 0:
        verification.append("candidate changed the fixed prefix")
    if max(errors[first + block_size :], default=0) > SUFFIX_TOLERANCE_M:
        verification.append("candidate changed the propagated suffix")
    for j, i in enumerate(window):
        impacts = replay[i]["bounces"]
        if (
            len(impacts) != 1
            or abs(float(impacts[0]["frame"]) - float(q[6 * j + 2])) > EPOCH_TOLERANCE_FRAMES
            or impacts[0].get("applied_restitution") != q[physical + j]
        ):
            verification.append(f"flight {i} replay did not reproduce the fitted ground route")
        if i in horizontal_flights:
            expected_h = q[physical + block_size + horizontal_flights.index(i)]
            if len(impacts) != 1 or impacts[0].get("applied_horizontal_retention") != expected_h:
                verification.append(f"flight {i} replay did not reproduce horizontal retention")
    if verification:
        details.update(
            reason="candidate replay verification failed: " + "; ".join(verification),
            candidate_replay_errors_m=errors,
            wall_seconds=time.monotonic() - started,
        )
        return context, details
    details.update(
        status="refined",
        feasible_output=True,
        replacement_reason="lower input objective"
        if source_admissibility["admissible"]
        else "restore original block physical constraints",
        full_vector=p.tolist(),
        ground_response=deepcopy(record),
        response_routes=deepcopy(record["routes"]),
        q=q.tolist(),
        roots=best["roots"],
        contact_xyz=best["contact_xyz"],
        boundary_error_m=best["boundary_equality_m"].tolist(),
        minimum_slack=float(np.min(best["clearance_slacks_m"])),
        prefix_max_error_m=max(errors[:first], default=0),
        suffix_max_error_m=max(errors[first + block_size :], default=0),
        wall_seconds=time.monotonic() - started,
    )
    return context, details
