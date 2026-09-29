"""Bounded real-event interval chart and movable three-flight block repair.

Replace one launch Vz coordinate by a latent first-impact epoch inside the
original annotation interval. An inner root uses the unchanged physical model;
its actual second impact must occur after the next racket contact. Shared outer
contacts, unrelated parameters and global rebound coefficients remain fixed.
This is opened human/Astra development, never an automatic inference input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq, minimize

from cv.experiments.connected_shooting import block_flight_repair_probe as block
from cv.experiments.connected_shooting import full_native_continuation as full
from cv.experiments.connected_shooting import labeled_context_census as census
from cv.experiments.connected_shooting import labeled_context_witness_scope as scope
from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import paths, provenance


class ChartError(ValueError):
    pass


class FirstImpactChart:
    """Exact-value root cache; no proxy position is reported as an impact."""

    def __init__(self, scene, flight, interval, maximum_simulations=40000):
        self.scene = scene
        self.flight = flight
        self.interval = np.asarray(interval, float)
        if self.interval.shape != (2,) or self.interval[0] >= self.interval[1]:
            raise ValueError("nonzero original timing interval required")
        if scene.net_hit_frames is not None and len(scene.net_hit_frames[flight]):
            raise ValueError("ground-only chart excludes declared net transitions")
        self.start, self.end = map(float, scene.contact_frames[flight : flight + 2])
        if not self.start < self.interval[0] < self.interval[1] < self.end:
            raise ValueError("impact interval must precede the next racket contact")
        self.slices = full.model.shared_parameter_slices(scene)
        self.vz = self.slices["velocities"].start + 3 * flight + 2
        self.maximum_simulations = maximum_simulations
        self.simulations = 0
        self.cache = {}

    def simulate(self, parameters):
        self.simulations += 1
        if self.simulations > self.maximum_simulations:
            raise TimeoutError("inner simulation budget reached")
        i = self.flight
        s = self.slices
        theta = np.r_[
            parameters[3 * i : 3 * i + 3],
            parameters[s["velocities"]][3 * i : 3 * i + 3],
            parameters[s["spins"]][3 * i : 3 * i + 3],
        ]
        kwargs = dict(
            bounce_profile=self.scene.bounce_profile, rebound_scales=parameters[s["rebound_scales"]]
        )
        if self.scene.bounce_regime_override is not None:
            raise ValueError("this real chart preserves the original free impact regime")
        return measured_dynamics.simulate(
            theta,
            self.start,
            np.asarray([self.start, self.end + 4]),
            self.scene.fps,
            self.scene.surface,
            **kwargs,
        )

    def first_impact_frame(self, parameters):
        """Bracket using only the first event; later trial bounces are irrelevant."""
        self.simulations += 1
        if self.simulations > self.maximum_simulations:
            raise TimeoutError("inner simulation budget reached")
        i, s = self.flight, self.slices
        theta = np.r_[
            parameters[3 * i : 3 * i + 3],
            parameters[s["velocities"]][3 * i : 3 * i + 3],
            parameters[s["spins"]][3 * i : 3 * i + 3],
        ]
        # No cache: this deliberately truncated trace cannot alias a full flight.
        impacts = measured_dynamics.integrate(
            theta,
            self.start,
            (self.end + 4 - self.start) / self.scene.fps,
            self.scene.fps,
            self.scene.surface,
            self.scene.bounce_profile,
            parameters[s["rebound_scales"]],
            None,
            stop_after_first_impact=True,
        )[4]
        return float(impacts[0]["frame"]) if impacts else self.end + 4

    def project(self, parameters, epoch):
        epoch = float(epoch)
        if not self.interval[0] <= epoch <= self.interval[1]:
            raise ChartError("epoch outside original interval")
        p = np.asarray(parameters, float).copy()
        i = self.flight
        s = self.slices
        # Every changed flight input and tau enters the exact key. Ignored Vz is
        # only a root initializer, never an independent physical coordinate.
        relevant = np.r_[
            p[3 * i : 3 * i + 3],
            p[s["velocities"]][3 * i : 3 * i + 2],
            p[s["spins"]][3 * i : 3 * i + 3],
            p[s["rebound_scales"]],
            epoch,
        ]
        key = relevant.tobytes()
        if key in self.cache:
            vz, receipt = self.cache[key]
            p[self.vz] = vz
            return p, dict(receipt)
        reference = float(p[self.vz])

        def residual(vz):
            p[self.vz] = vz
            return self.first_impact_frame(p) - epoch

        low = max(-75.0, reference - 12)
        high = min(75.0, reference + 12)
        left, right = residual(low), residual(high)
        if left * right > 0:
            raise ChartError("no local first-impact Vz bracket")
        vz = brentq(residual, low, high, xtol=1e-10, rtol=1e-12, maxiter=50)
        residual(vz)
        # The chosen root still needs the original complete-flight verification.
        impacts = self.simulate(p)[3]
        error = float(impacts[0]["frame"] - epoch) if impacts else float("inf")
        if not impacts or abs(error) > 1e-7 or impacts[0]["v_in"][2] >= 0:
            raise ChartError("root did not produce the requested descending first impact")
        receipt = dict(
            epoch=epoch,
            actual_first_impact_frame=float(impacts[0]["frame"]),
            error_frames=float(error),
            vz_mps=float(vz),
            second_impact_frame=None if len(impacts) < 2 else float(impacts[1]["frame"]),
            second_after_contact_slack_frames=4.0
            if len(impacts) < 2
            else float(impacts[1]["frame"] - self.end),
            first_regime=impacts[0]["regime"],
            coefficient_clipped=impacts[0]["coefficient_clipped"],
        )
        if len(self.cache) > 256:
            self.cache.clear()
        self.cache[key] = (vz, receipt)
        return p, dict(receipt)


def optimize(
    context, parameters, flight, interval, duration, *, latent=True, maxiter=80, seconds=300
):
    began = time.monotonic()
    scene = replace(context["scene"], parameterization="shared_contact_states")
    initial = full.model.shared_contact_seed(scene, parameters)
    indices = block.block_indices(scene, flight)
    changed = np.arange(flight - 1, flight + 2)
    scale = np.r_[[1.0] * 6, [30.0] * 9, [3.0] * 9]
    low = np.r_[initial[indices[:6]] - 1.5, [-75.0] * 9, [-6.0] * 9]
    high = np.r_[initial[indices[:6]] + 1.5, [75.0] * 9, [6.0] * 9]
    low[[2, 5]] = np.maximum(low[[2, 5]], full.model.R_BALL + 1e-4)
    high[[2, 5]] = np.minimum(high[[2, 5]], 4.0)
    chart = FirstImpactChart(scene, flight, interval)
    vz_local = int(np.flatnonzero(indices == chart.vz)[0])
    z0 = initial[indices].copy()
    original_flights = full.model.chain(context["scene"], parameters)
    source_epoch = original_flights[flight]["bounces"][0]["frame"]
    if latent:
        z0[vz_local] = np.clip(source_epoch, *interval)
        low[vz_local], high[vz_local] = interval
        scale[vz_local] = 1.0
    q0 = z0 / scale
    target = np.concatenate(scene.pixels)
    loss = block.event_constraints.mixed_loss(2 * len(target))
    groups = block.contact_image_groups(scene, flight)
    cache = FlightCache(entries_per_flight=80)
    memo = {}
    errors = Counter()
    best = None
    latest = None

    def expand(q):
        p = initial.copy()
        p[indices] = q * scale
        if latent:
            epoch = p[chart.vz]
            p[chart.vz] = initial[chart.vz]
            return chart.project(p, epoch)
        impacts = chart.simulate(p)[3]
        return p, dict(
            second_after_contact_slack_frames=4.0
            if len(impacts) < 2
            else impacts[1]["frame"] - chart.end,
            actual_first_impact_frame=None if not impacts else impacts[0]["frame"],
        )

    def evaluate(q):
        nonlocal best, latest
        if time.monotonic() - began > seconds:
            raise TimeoutError("outer wall-time budget reached")
        key = np.asarray(q).tobytes()
        if key in memo:
            return memo[key]
        try:
            p, receipt = expand(q)
            flights, physical, _ = block.event_constraints.evaluate(
                scene, p, context["bounces"], 2.0, simulation_cache=cache
            )
            image = (
                full.exposure.prediction(
                    scene,
                    p,
                    context["axes"],
                    duration,
                    cache,
                    termination_kind=context["termination_kind"],
                )
                - target
            )
            residual = np.r_[image.ravel(), physical]
            cost = float(2 * np.sum(loss((residual / 2) ** 2)[0]))
            gaps = full.model.shared_contact_gaps(flights)[changed].ravel()
            budgets = block.image_budgets(image, groups)
            # Continuous timing guidance from the actual second impact. This is
            # not an invented second bounce annotation or a new acceptance gate.
            second = receipt["second_after_contact_slack_frames"] - 0.01
            value = (cost, gaps, budgets, second, p, receipt)
            latest = (np.asarray(q).copy(), value)
            if (
                np.max(np.abs(gaps)) <= 0.0003
                and second >= 0
                and (best is None or cost < best[1][0])
            ):
                best = latest
        except (ValueError, FloatingPointError) as exc:
            errors[str(exc)] += 1
            value = (1e12, np.full(9, 1000.0), np.full(len(groups), 1e12), -1e6, None, {})
        if len(memo) > 256:
            memo.clear()
        memo[key] = value
        return value

    original_image = (
        full.exposure.prediction(
            scene,
            initial,
            context["axes"],
            duration,
            cache,
            termination_kind=context["termination_kind"],
        )
        - target
    )
    image_limits = np.maximum(block.image_budgets(original_image, groups) + 1e-8, 16.0**2)
    baseline = evaluate(q0)
    termination = {}
    chosen = None
    try:
        solved = minimize(
            lambda q: evaluate(q)[0],
            q0,
            method="SLSQP",
            bounds=list(zip(low / scale, high / scale)),
            constraints=[
                dict(type="eq", fun=lambda q: evaluate(q)[1]),
                dict(type="ineq", fun=lambda q: image_limits - evaluate(q)[2]),
                dict(type="ineq", fun=lambda q: evaluate(q)[3]),
            ],
            options=dict(maxiter=maxiter, ftol=1e-8),
        )
        chosen = (solved.x, evaluate(solved.x))
        termination = dict(
            success=bool(solved.success),
            message=str(solved.message),
            iterations=int(solved.nit),
            nfev=int(solved.nfev),
        )
    except TimeoutError as exc:
        termination = dict(success=False, message=str(exc))
        chosen = best or latest
    if chosen is None or chosen[1][4] is None:
        return dict(status="no_physical_candidate", termination=termination, errors=dict(errors))
    q, value = chosen
    p = value[4]
    single = full.model.shared_to_single_parameters(scene, p)
    actual = full.model.chain(context["scene"], single)
    shared = full.model.chain(scene, p)
    export_drift = max(
        float(np.max(np.abs(a["positions"] - b["positions"])))
        for a, b in zip(actual, shared, strict=True)
    )
    return dict(
        status="candidate",
        arm="latent_original_interval" if latent else "ordinary_velocity_control",
        termination=termination,
        parameters=single,
        shared_parameters=p,
        source_parameters=parameters,
        interval=interval,
        first_impact_receipt=value[5],
        initial_cost=baseline[0],
        final_cost=value[0],
        maximum_endpoint_gap_m=float(np.max(np.linalg.norm(value[1].reshape(-1, 3), axis=1))),
        shared_to_single_maximum_native_drift_m=export_drift,
        contact_image_rms_px=np.sqrt(value[2]),
        contact_image_limits_px=np.sqrt(image_limits),
        second_impact_after_contact=bool(value[3] >= 0),
        inner_simulations=chart.simulations,
        errors=dict(errors),
        wall_seconds=time.monotonic() - began,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="rg2017f_pt0004_a01")
    parser.add_argument("--flight", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--arm", choices=["latent", "control"], default="latent")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    basepath = paths.data_root() / "processed/ownerfix/adopted_perflight3/report.json"
    base = json.loads(basepath.read_text())
    item = next(x for x in base["attempts"] if x["key"] == args.key)
    compositionpath = (
        paths.data_root()
        / "processed/lifting_goal_20260910/labeled_fitted_v1/witness_scope_v2/report.json"
    )
    rung = next(
        x["rung"]
        for x in json.loads(Path(item["document"]).read_text())["rungs"]
        if x["rung"].startswith(
            "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f"
        )
    )
    ctx, p, _, _, search, t, d, loaded = full.load_case(dict(item=item, rung=rung))
    before, measurement = census.score(
        ctx, p, t, d, item["metadata"]["ending_kind"], scorer=scope.LOCAL_SCORE
    )
    event = next(
        e
        for e in ctx["events"]
        if e["event_type"] == "bounce"
        and ctx["scene"].contact_frames[args.flight]
        < e["frame"]
        < ctx["scene"].contact_frames[args.flight + 1]
    )
    interval = event["frame_interval"]
    scene = replace(ctx["scene"], parameterization="shared_contact_states")
    seed = full.model.shared_contact_seed(scene, p)
    chart = FirstImpactChart(scene, args.flight, interval)
    preflight = []
    for epoch in sorted(
        set(
            [
                *interval,
                sum(interval) / 2,
                float(
                    np.clip(
                        measurement["dense_flights"][args.flight]["bounces"][0]["frame"], *interval
                    )
                ),
            ]
        )
    ):
        try:
            projected, receipt = chart.project(seed, epoch)
            flights = full.model.chain(scene, projected)
            receipt.update(
                endpoint_gap_m=float(
                    np.linalg.norm(full.model.shared_contact_gaps(flights)[args.flight])
                ),
                status="root_verified",
            )
            preflight.append(receipt)
        except Exception as exc:
            preflight.append(
                dict(epoch=epoch, status="root_failure", error=f"{type(exc).__name__}: {exc}")
            )
    result = dict(
        key=args.key,
        flight=args.flight,
        event=event,
        before=before,
        preflight=preflight,
        source=[
            *loaded["sources"],
            *[
                provenance.file_record(x)
                for x in [
                    Path(__file__),
                    basepath,
                    compositionpath,
                    Path(block.__file__),
                    Path(measured_dynamics.__file__),
                ]
            ],
        ],
        automatic_inference_eligible=False,
    )
    (args.output / "preflight.json").write_text(
        json.dumps(full.profile.jsonable(result), indent=2) + "\n"
    )
    print("preflight", json.dumps(preflight), flush=True)
    if args.preflight_only:
        return
    candidate = optimize(
        ctx,
        p,
        args.flight,
        interval,
        d,
        latent=args.arm == "latent",
        maxiter=args.maxiter,
        seconds=args.seconds,
    )
    result["candidate"] = candidate
    if candidate["status"] == "candidate":
        after, m = census.score(
            ctx,
            np.asarray(candidate["parameters"]),
            t,
            d,
            item["metadata"]["ending_kind"],
            scorer=scope.LOCAL_SCORE,
        )
        candidate.update(
            verdict=after,
            measurement=m,
            accepted_gain=after["accepted_flight_count"] > before["accepted_flight_count"],
            lost_original_flights=sorted(
                set(before["accepted_flight_indices"]) - set(after["accepted_flight_indices"])
            ),
        )
    (args.output / "report.json").write_text(
        json.dumps(full.profile.jsonable(result), indent=2) + "\n"
    )
    print(
        candidate["status"],
        candidate.get("termination"),
        candidate.get("maximum_endpoint_gap_m"),
        candidate.get("verdict", {}).get("accepted_flight_count"),
        flush=True,
    )


if __name__ == "__main__":
    main()
