"""Optional two-flight ground/contact coupling on a supplied current connected state.

Both original racket epochs and the prefix stay fixed. Sequential first-impact
charts propagate the preceding endpoint into the terminal launch. One input-only
objective evaluates the exact incumbent and all trials. Original second-impact
intervals guide residuals smoothly; only feasible strict improvements may return.
The shared S6 stage invokes this optional helper; it performs no file IO, gate
selection or model calls.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import ExitStack
from copy import deepcopy
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    interior_contact_epochs,
    feasible_difference,
    labeled_interior_ground_response as ground,
    labeled_interior_normal as interior,
    labeled_prefix_boundary as boundary,
    labeled_terminal_ground_response as terminal,
    model,
    net_constraints,
)
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.experiments.connected_shooting.labeled_net_free_response import response_from_record
from cv.experiments.connected_shooting.labeled_passive_tape import using_response
from physics import bounce_reference

ERRORS = interior.NUMERICAL_ERRORS
EPOCH_TOLERANCE = 1e-7
NORMAL_BOUNDS = (0.05, 1.0)
INTERVAL_SCALE_FRAMES = 0.25


def qualify(
    context: dict,
    duration: float,
    second_normal: bool = True,
    horizontal_retention: bool = False,
    *,
    allow_sparse_wings: bool = False,
) -> dict:
    """Only supplied final ground pair and supported second-rebound observations."""
    if type(allow_sparse_wings) is not bool:
        raise ValueError("allow_sparse_wings must be an explicit boolean")
    rows = interior._flight_rows(context, duration)
    n = len(rows)
    if n < 3:
        raise ValueError("supported preceding non-serve single-ground flight required")
    preceding = rows[-2]
    incoming, rebound = preceding["incoming_frames"], preceding["rebound_frames"]
    # Admit missing local evidence only with the complete terminal branch below.
    # Nothing here supplies the missing rebound or an independently measured COR.
    sparse = bool(
        allow_sparse_wings
        and preceding["reason"] == "fewer than two original native exposures on one ground wing"
        and ((len(incoming) < 2) != (len(rebound) < 2))
        and len(set(incoming + rebound)) >= 4
    )
    if preceding["reason"] and not sparse:
        raise ValueError("supported preceding non-serve single-ground flight required")
    spec = terminal.qualify(context, duration)
    original, events = interior_contact_epochs.original_contact_inventory(context, duration)
    contacts = [e for e in events if e["event_type"] == "contact"]
    first, shared, end = map(float, context["scene"].contact_frames[-3:])
    for nominal, epoch in zip(original[-3:-1], (first, shared), strict=True):
        matches = [e for e in contacts if abs(float(e["frame"]) - nominal) <= 1e-8]
        if len(matches) != 1:
            raise ValueError("two original racket boundaries required")
        low, high = map(float, matches[0]["frame_interval"])
        if not low <= epoch <= high:
            raise ValueError("original racket epoch outside supplied interval")
    if any(
        original[-3] < float(e["frame"]) <= original[-1]
        and abs(float(e["frame"]) - original[-2]) > 1e-8
        for e in contacts
    ):
        raise ValueError("additional original contact inside terminal pair")
    frames = spec.get("second_rebound_frames", [])
    horizontal_frames = [
        float(frame)
        for frame in spec.get("rebound_frames", [])
        if spec["ground_count"] == 1 or frame < spec["intervals"][1][0]
    ]
    horizontal_supported = len(horizontal_frames) >= 2
    return dict(
        **(
            dict(
                sparse_wing_admission=dict(
                    qualified=sparse,
                    preceding_flight=n - 2,
                    incoming_native_epochs=incoming,
                    rebound_native_epochs=rebound,
                    unique_preceding_wing_epochs=len(set(incoming + rebound)),
                    terminal_incoming_native_epochs=spec["incoming_frames"],
                    terminal_rebound_native_epochs=spec["rebound_frames"],
                    minimum_preceding_wing_epochs=4,
                    policy="one deficient preceding ground wing; other wing supported; fully supported terminal ground wings",
                    uncertainty="missing wing remains unobserved; contact depth and normal response are jointly regularized, not independently measured",
                )
            )
            if allow_sparse_wings
            else {}
        ),
        preceding=deepcopy(rows[-2]),
        terminal=spec,
        second_normal_requested=bool(second_normal),
        second_normal_active=bool(
            second_normal
            and spec["ground_count"] == 2
            and len(frames) >= terminal.SECOND_REBOUND_MIN_EXPOSURES
        ),
        second_normal_support=dict(
            native_frames=frames, minimum_distinct_exposures=terminal.SECOND_REBOUND_MIN_EXPOSURES
        ),
        horizontal_retention_requested=bool(horizontal_retention),
        horizontal_retention_active=bool(horizontal_retention and horizontal_supported),
        horizontal_support=dict(native_frames=horizontal_frames, minimum_distinct_exposures=2),
    )


def _laws(scene: model.Scene, registry: dict | None, net: dict | None) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(ground.using_response(registry, scene))
    if net is not None:
        stack.enter_context(using_response(response_from_record(net)))
    return stack


def _interval_terms(chain: list[dict], spec: dict, queries: tuple) -> tuple[np.ndarray, dict]:
    """Out-of-interval events remain differentiable; topology only filters incumbents."""
    groups = [[spec["preceding"]["interval"]], spec["terminal"]["intervals"]]
    residual, slacks, missing, extras = [], [], [], []
    valid = True
    for j, (flight, intervals) in enumerate(zip(chain[-2:], groups, strict=True)):
        impacts = flight["bounces"]
        valid &= not bool(flight.get("net_hits"))
        for k, (lo, hi) in enumerate(intervals):
            if k < len(impacts):
                t = float(impacts[k]["frame"])
                slack = [t - lo, hi - t]
                residual.extend(np.minimum(slack, 0) / INTERVAL_SCALE_FRAMES)
                slacks.extend(slack)
            else:
                # Existing native-height guidance keeps the missing-event branch
                # informative, without fabricating an impact or accepting it.
                z = np.interp((lo + hi) / 2, queries[-2 + j], flight["positions"][:, 2])
                residual.extend([0.0, max(float(z) - model.R_BALL, 0) / 0.05])
                missing.append([j, k])
                valid = False
        additional = impacts[len(intervals) :]
        extra_frames = [float(b["frame"]) for b in additional]
        extras.append(extra_frames)
        if j == 0:
            residual.append(
                max(flight["end_frame"] - extra_frames[0], 0) / INTERVAL_SCALE_FRAMES
                if extra_frames
                else 0.0
            )
            valid &= not additional
        else:
            allowed = spec["terminal"]["continuation"]["enabled"]
            residual.append(
                max(intervals[-1][1] - extra_frames[0], 0) / INTERVAL_SCALE_FRAMES
                if extra_frames
                else 0.0
            )
            valid &= (not additional) or (
                allowed
                and all(
                    intervals[-1][1] < f <= flight["end_frame"] + EPOCH_TOLERANCE
                    for f in extra_frames
                )
            )
    return np.asarray(residual), dict(
        impact_slacks_frames=slacks,
        missing_original_impacts=missing,
        continuation_ground_frames=extras[-1],
        topology_supported=bool(valid),
    )


class PairProblem:
    """One fixed objective and sequential projector, exposed for exact replay tests."""

    def __init__(
        self,
        context: dict,
        source: np.ndarray,
        duration: float,
        registry: dict | None,
        net: dict | None,
        spec: dict,
    ):
        self.context, self.source, self.duration, self.spec = context, source.copy(), duration, spec
        checks, self.consumed = terminal.followup.fit_check_copy(
            context["scene"], context["heldout"]
        )
        self.scene, self.axes, _ = terminal.prefix.full.merge_scene(
            context["scene"], checks, context["bounces"], context["axes"]
        )
        self.n = len(self.scene.pixels)
        self.first = self.n - 2
        self.registry = ground.normalize(registry, self.scene)
        self.net = deepcopy(net)
        self.queries = net_constraints.dense_queries(self.scene, context["bounces"])
        self.target = np.concatenate(self.scene.pixels)
        self.cache = FlightCache(entries_per_flight=64)
        with _laws(self.scene, self.registry, net):
            self.original = model.chain(self.scene, source, query_frames=self.queries)
        self.projector = interior.BlockProjector(
            self.scene,
            source,
            self.first,
            [spec["preceding"], dict(interval=spec["terminal"]["interval"])],
            self.original,
        )
        self.active_normals = None
        self.q0, self.lower, self.upper, self.scale = [], [], [], []
        self.normal_priors = []
        self.selected = []
        for j, i in enumerate(range(self.first, self.n)):
            interval = spec["preceding"]["interval"] if j == 0 else spec["terminal"]["interval"]
            v, w = 3 + 3 * i, 3 + 3 * self.n + 3 * i
            self.selected.extend(range(v, v + 3))
            self.selected.extend(range(w, w + 3))
            impacts = self.original[i]["bounces"]
            normal, center, sigma = self._normal_seed(impacts, i, 0)
            self.normal_priors.append([center, sigma])
            epoch = float(np.clip(impacts[0]["frame"] if impacts else np.mean(interval), *interval))
            self.q0.extend([*source[v : v + 2], epoch, *source[w : w + 3], normal])
            self.lower.extend([-75, -75, interval[0], -6, -6, -6, 0.05])
            self.upper.extend([75, 75, interval[1], 6, 6, 6, 1])
            self.scale.extend([30, 30, 1, 3, 3, 3, 1])
        if spec["second_normal_active"]:
            normal, center, sigma = self._normal_seed(self.original[-1]["bounces"], self.n - 1, 1)
            self.q0.append(normal)
            self.lower.append(0.05)
            self.upper.append(1)
            self.scale.append(1)
            self.normal_priors.append([center, sigma])
        self.horizontal_index = None
        self.horizontal_prior = None
        if len(spec["horizontal_support"]["native_frames"]) >= 2:
            first_ground = self.original[-1]["bounces"]
            if not first_ground:
                raise ValueError("source terminal first ground required for retention prior")
            impact = first_ground[0]
            nominal = float(
                impact.get(
                    "nominal_applied_horizontal_retention", impact["applied_horizontal_retention"]
                )
            )
            center = float(np.clip(nominal, 0.05, 1))
            sigma = max(
                0.10,
                bounce_reference.MEASURED[self.scene.surface][
                    "retention_residual_std_at_model_spin"
                ]
                * float(source[-1]),
            )
            self.horizontal_prior = (center, sigma)
            if spec["horizontal_retention_active"]:
                self.horizontal_index = len(self.q0)
                self.q0.append(float(impact["applied_horizontal_retention"]))
                self.lower.append(0.05)
                self.upper.append(1.0)
                self.scale.append(1.0)
        self.q0, self.lower, self.upper, self.scale = map(
            np.asarray, (self.q0, self.lower, self.upper, self.scale)
        )
        if np.any(self.q0 < self.lower) or np.any(self.q0 > self.upper):
            raise ValueError(
                "source launch coordinates outside shared bounds; no returned clipping"
            )
        self.offsets = np.cumsum([0, *map(len, self.scene.pixels)])

    def _normal_seed(self, impacts: list, flight: int, ordinal: int) -> tuple[float, float, float]:
        row = bounce_reference.MEASURED[self.scene.surface]
        scale = float(self.source[-2])
        if len(impacts) > ordinal:
            b = impacts[ordinal]
            nominal = float(b.get("nominal_applied_restitution", b["applied_restitution"]))
            applied = float(b["applied_restitution"])
        else:
            nominal = float(
                bounce_reference.restitution(row["incidence_deg_median"], self.scene.surface)
                * scale
            )
            applied = nominal
        existing = next(
            (
                r
                for r in (self.registry or {}).get("routes", [])
                if r["flight_index"] == flight and r["ground_ordinal"] == ordinal
            ),
            None,
        )
        seed = float(
            existing["normal_restitution"] if existing else np.clip(applied, *NORMAL_BOUNDS)
        )
        return (
            seed,
            float(np.clip(nominal, *NORMAL_BOUNDS)),
            float(max(0.15, row["restitution_residual_std_at_model_spin"] * scale)),
        )

    def expand(self, q: np.ndarray) -> tuple[np.ndarray, dict, list]:
        q = np.asarray(q, float)
        if (
            q.shape != self.q0.shape
            or not np.isfinite(q).all()
            or np.any(q < self.lower)
            or np.any(q > self.upper)
        ):
            raise ValueError("finite in-bounds chart coordinates required")
        p = self.source.copy()
        routes = []
        for j, i in enumerate(range(self.first, self.n)):
            v, w = 3 + 3 * i, 3 + 3 * self.n + 3 * i
            k = 7 * j
            p[v : v + 2] = q[k : k + 2]
            p[w : w + 3] = q[k + 3 : k + 6]
            routes.append(
                dict(
                    flight_index=i,
                    native_contact_epoch=float(self.scene.contact_frames[i]),
                    ground_ordinal=0,
                    normal_restitution=float(q[k + 6]),
                )
            )
        if self.spec["second_normal_active"]:
            routes.append(
                dict(
                    flight_index=self.n - 1,
                    native_contact_epoch=float(self.scene.contact_frames[-2]),
                    ground_ordinal=1,
                    normal_restitution=float(q[14]),
                )
            )
        if self.horizontal_index is not None:
            routes[1]["horizontal_retention"] = float(q[self.horizontal_index])
        registry = ground.merge(self.registry, routes, self.scene)
        active = tuple(q[[6, 13, 14]] if self.spec["second_normal_active"] else q[[6, 13]])
        if self.horizontal_index is not None:
            active += (q[self.horizontal_index],)
        if active != self.active_normals:
            self.projector.clear()
            self.active_normals = active
        with _laws(self.scene, registry, self.net):
            p, roots = self.projector.project(p, [q[2], q[9]])
        return p, registry, roots

    def evaluate(self, p: np.ndarray, registry: dict | None) -> tuple[np.ndarray, dict, list]:
        with _laws(self.scene, registry, self.net):
            chain = model.chain(
                self.scene, p, query_frames=self.queries, simulation_cache=self.cache
            )
            predicted = terminal.prefix.full.exposure.prediction(
                self.scene,
                p,
                self.axes,
                self.duration,
                self.cache,
                termination_kind=self.context["termination_kind"],
            )
        image = np.asarray(predicted) - self.target
        timing, timing_receipt = _interval_terms(chain, self.spec, self.queries)
        _, clearance = boundary.constraints(chain[-2:], self.queries[-2:], 2, [[], []], None)
        floor = np.array([float(np.min(f["positions"][:, 2])) - model.R_BALL for f in chain[-2:]])
        physical = np.r_[np.minimum(clearance, 0) / 0.025, np.minimum(floor, 0) / 0.025]
        starts = np.asarray([f["start_xyz"] for f in chain])
        players = interior.indexed_player_residuals(starts, self.context["players"], self.n - 1)
        # Fixed priors and weights apply identically to the actual source vector.
        motion = (p[self.selected] - self.source[self.selected]) / np.tile([30, 30, 30, 6, 6, 6], 2)
        impacts = [chain[-2]["bounces"], chain[-1]["bounces"]]
        normals = []
        for j, (center, sigma) in enumerate(self.normal_priors):
            flight = self.n - 2 if j == 0 else self.n - 1
            ordinal = 1 if j == 2 else 0
            rows = impacts[0 if j == 0 else 1]
            route = next(
                (
                    r
                    for r in (registry or {}).get("routes", [])
                    if r["flight_index"] == flight and r["ground_ordinal"] == ordinal
                ),
                None,
            )
            applied = (
                float(rows[ordinal]["applied_restitution"])
                if len(rows) > ordinal
                else float(route["normal_restitution"] if route else center)
            )
            normals.append((applied - center) / sigma)
        rebound = 0.0
        rebound_ok = True
        required = self.spec["terminal"]["ground_count"]
        # An exposure inside the original impact interval is not necessarily a
        # post-dwell rebound picture. Qualify by the original interval's upper
        # edge, using the same support inventory as the normal-response helper.
        frames = self.spec["terminal"].get(
            "second_rebound_frames" if required == 2 else "rebound_frames", []
        )
        if frames is not None and len(frames) and len(impacts[-1]) >= required:
            b = impacts[-1][required - 1]
            delta = float(b["frame"] + b["dwell_seconds"] * self.scene.fps - min(frames))
            rebound = max(delta, 0) / INTERVAL_SCALE_FRAMES
            rebound_ok = delta <= EPOCH_TOLERANCE
        horizontal = 0.0
        applied_horizontal = None
        if self.horizontal_prior is not None:
            center, sigma = self.horizontal_prior
            if impacts[-1]:
                applied_horizontal = float(impacts[-1][0]["applied_horizontal_retention"])
            else:
                row = next(
                    (
                        r
                        for r in (registry or {}).get("routes", [])
                        if r["flight_index"] == self.n - 1 and r["ground_ordinal"] == 0
                    ),
                    {},
                )
                applied_horizontal = float(row.get("horizontal_retention", center))
            horizontal = (applied_horizontal - center) / sigma
        residual = np.r_[
            image.ravel(), timing, physical, players, motion, normals, rebound, horizontal
        ]
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite fixed input objective")
        slacks = timing_receipt["impact_slacks_frames"]
        feasible = bool(
            timing_receipt["topology_supported"]
            and not timing_receipt["missing_original_impacts"]
            and min(slacks, default=0) >= -EPOCH_TOLERANCE
            and np.min(clearance) >= -1e-6
            and np.min(floor) >= -1e-6
            and rebound_ok
        )
        frame = np.concatenate(self.scene.observation_frames)
        windows = {}
        for j, i in enumerate(range(self.first, self.n)):
            bounds = (
                self.spec["preceding"]["interval"] if j == 0 else self.spec["terminal"]["interval"]
            )
            select = np.zeros(len(frame), bool)
            select[self.offsets[i] : self.offsets[i + 1]] = True
            names = {
                "incoming": select & (frame < bounds[0]),
                "first_rebound": select & (frame > bounds[1]),
            }
            if j == 1 and required == 2:
                second = self.spec["terminal"]["intervals"][1]
                names["first_rebound"] &= frame < second[0]
                names["second_rebound"] = select & (frame > second[1])
            for name, mask in names.items():
                windows[f"flight_{i + 1}_{name}"] = dict(
                    frames=frame[mask].tolist(),
                    count=int(mask.sum()),
                    sse_px=float(np.sum(image[mask] ** 2)),
                    rms_px=float(np.sqrt(np.mean(np.sum(image[mask] ** 2, axis=1))))
                    if mask.any()
                    else None,
                )
        return (
            residual,
            dict(
                cost=float(residual @ residual),
                feasible=feasible,
                components=dict(
                    image=float(np.sum(image**2)),
                    interval=float(timing @ timing),
                    physical=float(physical @ physical),
                    player=float(players @ players),
                    motion=float(motion @ motion),
                    normal=float(np.dot(normals, normals)),
                    rebound=float(rebound**2),
                    horizontal=float(horizontal**2),
                ),
                terminal_first_horizontal_retention=applied_horizontal,
                horizontal_prior=self.horizontal_prior,
                native_windows=windows,
                contact_xyz=starts[-1].tolist(),
                minimum_clearance_m=float(np.min(clearance)),
                rebound_support_frames=list(frames),
                rebound_order_residual=float(rebound),
                **timing_receipt,
            ),
            chain,
        )


def fit(
    context: dict,
    source: np.ndarray,
    duration: float,
    *,
    ground_response: dict | None = None,
    net_response: dict | None = None,
    enabled: bool = False,
    second_normal: bool = True,
    horizontal_retention: bool = False,
    allow_sparse_wings: bool = False,
    seconds: float = 120.0,
    maxiter: int = 60,
) -> tuple[dict, dict]:
    """Return exact source on no-op/failure; checkpoints are lower-cost feasible states."""
    started = time.monotonic()
    p0 = np.asarray(source, float).copy()
    n = len(context["scene"].pixels)
    if p0.shape != (5 + 6 * n,) or not np.isfinite(p0).all():
        raise ValueError("finite full source required")
    if (
        type(enabled) is not bool
        or type(second_normal) is not bool
        or type(horizontal_retention) is not bool
        or type(allow_sparse_wings) is not bool
        or type(maxiter) is not int
        or maxiter < 1
        or not np.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("explicit booleans and positive shared budgets required")
    registry = ground.normalize(deepcopy(ground_response), context["scene"])
    details = dict(
        status="disabled" if not enabled else "source_retained",
        full_vector=p0.tolist(),
        initial_source=p0.tolist(),
        ground_response=deepcopy(registry),
        initial_ground_response=deepcopy(registry),
        retained_net_response=deepcopy(net_response),
        gate_selection=False,
        external_fitted_inputs_used=False,
        policy=dict(
            enabled=enabled,
            second_normal=second_normal,
            horizontal_retention=horizontal_retention,
            seconds=float(seconds),
            maxiter=maxiter,
            original_contact_epochs_fixed=True,
            prefix_fixed=True,
            native_rows_unchanged=True,
            strict_source_incumbent=True,
            second_interval_residual="one-sided hinge /0.25 frames; strict feasible checkpoint",
            input_objective="all merged native images + timing/clearance/player/motion/normal/horizontal priors",
        ),
    )
    if not enabled:
        details["wall_seconds"] = time.monotonic() - started
        return context, details
    try:
        options = {"allow_sparse_wings": True} if allow_sparse_wings else {}
        spec = qualify(context, duration, second_normal, horizontal_retention, **options)
        if allow_sparse_wings:
            details["policy"]["allow_sparse_wings"] = True
        details["inventory"] = spec
        problem = PairProblem(context, p0, duration, registry, net_response, spec)
        source_r, source_metrics, _ = problem.evaluate(p0, registry)
        details["source_metrics"] = source_metrics
        best = dict(
            parameters=p0.copy(),
            registry=deepcopy(registry),
            metrics=source_metrics,
            roots=None,
            q=None,
        )
        counters = Counter()
        invalid = Counter()
        cache = OrderedDict()
        derivative_receipts = []
        best_trial_metrics = None

        def evaluate(x):
            nonlocal best_trial_metrics
            if time.monotonic() - started > seconds:
                raise TimeoutError("common terminal ground/contact deadline")
            key = np.asarray(x, float).tobytes()
            if key in cache:
                return cache[key]
            counters["evaluations"] += 1
            p, record, roots = problem.expand(x * problem.scale)
            r, metrics, _ = problem.evaluate(p, record)
            if r.shape != source_r.shape:
                raise ValueError("input residual shape changed")
            if best_trial_metrics is None or metrics["cost"] < best_trial_metrics["cost"]:
                best_trial_metrics = deepcopy(metrics)
            if metrics["feasible"]:
                counters["feasible"] += 1
                if metrics["cost"] < best["metrics"]["cost"] - 1e-9 * max(
                    1.0, best["metrics"]["cost"]
                ):
                    best.update(
                        parameters=p.copy(),
                        registry=deepcopy(record),
                        metrics=deepcopy(metrics),
                        roots=deepcopy(roots),
                        q=(x * problem.scale).tolist(),
                    )
            cache[key] = r
            if len(cache) > 48:
                cache.popitem(last=False)
            return r

        def trust_residual(x):
            try:
                return evaluate(x)
            except ERRORS as error:
                invalid[str(error)] += 1
                return np.full_like(source_r, 1e5)

        def jac(x):
            value, receipt = feasible_difference.feasible_jacobian(
                evaluate,
                x,
                problem.lower / problem.scale,
                problem.upper / problem.scale,
                rejected_errors=ERRORS,
            )
            derivative_receipts.append(receipt)
            return value

        termination = {}
        try:
            # Initial chart must be real; rejection penalties only reject later
            # trust-region trials and are never differentiated.
            evaluate(problem.q0 / problem.scale)
            solved = least_squares(
                trust_residual,
                problem.q0 / problem.scale,
                jac=jac,
                bounds=(problem.lower / problem.scale, problem.upper / problem.scale),
                max_nfev=maxiter,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            termination = dict(
                success=bool(solved.success),
                message=str(solved.message),
                nfev=solved.nfev,
                njev=solved.njev,
            )
        except ERRORS + (TimeoutError,) as error:
            termination = dict(success=False, message=f"{type(error).__name__}: {error}")
        # Re-evaluate the complete selected state under fresh caches before
        # returning. A route/chain mismatch retains the exact source, never a
        # clipped or partially replayed improvement.
        problem.cache = FlightCache(entries_per_flight=64)
        replay_status = dict(status="source_retained")
        if best["q"] is not None:
            try:
                rr, replay_metrics, replay_chain = problem.evaluate(
                    best["parameters"], best["registry"]
                )
                frozen = np.ones(len(p0), dtype=bool)
                frozen[problem.selected] = False
                if not np.array_equal(best["parameters"][frozen], p0[frozen]):
                    raise ValueError("coupling changed a frozen source slot")
                prefix_error = max(
                    (
                        float(np.max(np.abs(a["positions"] - b["positions"])))
                        for a, b in zip(
                            replay_chain[: problem.first], problem.original[: problem.first]
                        )
                    ),
                    default=0.0,
                )
                if prefix_error != 0:
                    raise ValueError("coupling changed the frozen physical prefix")
                if not replay_metrics["feasible"] or not np.isclose(
                    replay_metrics["cost"], best["metrics"]["cost"], rtol=1e-12, atol=1e-9
                ):
                    raise ValueError("selected candidate full-state replay mismatch")
                replay_status = dict(
                    status="verified",
                    prefix_max_position_error_m=prefix_error,
                    cost=replay_metrics["cost"],
                    shared_contact_xyz=replay_chain[-1]["start_xyz"].tolist(),
                    seam_error_m=float(
                        np.linalg.norm(replay_chain[-2]["end_xyz"] - replay_chain[-1]["start_xyz"])
                    ),
                )
            except ERRORS as error:
                replay_status = dict(status="failed_source_retained", reason=str(error))
                best = dict(
                    parameters=p0.copy(),
                    registry=deepcopy(registry),
                    metrics=source_metrics,
                    roots=None,
                    q=None,
                )
        details.update(
            status="improved" if best["q"] is not None else "source_retained",
            full_vector=best["parameters"].tolist(),
            ground_response=best["registry"],
            metrics=best["metrics"],
            replay=replay_status,
            chart=best["roots"],
            selected_q=best["q"],
            initial_q=problem.q0.tolist(),
            coordinate_count=len(problem.q0),
            counters=dict(counters),
            invalid_reasons=dict(invalid),
            termination=termination,
            contact_shift_m=(
                np.asarray(best["metrics"]["contact_xyz"]) - source_metrics["contact_xyz"]
            ).tolist(),
            best_completed_trial_metrics=best_trial_metrics,
            derivative_calls=len(derivative_receipts),
            consumed_native_check_rows=problem.consumed,
        )
    except ERRORS + (TimeoutError, IndexError) as error:
        details["reason"] = f"{type(error).__name__}: {error}"
    details["wall_seconds"] = time.monotonic() - started
    return context, details
