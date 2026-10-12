"""Bounded terminal-only context/rebound control for the eleven-flight USO point.

Opened labeled development. Earlier flights and native timestamps stay fixed.
Arms: sparse continuation, existing terminal context, or context with explicit
terminal-only rebound scales in the existing [.8,1.2] range. No default changes.
Run --arm sparse|context|rebound --output NEW_DIR.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import labeled_nonnet_triage as triage
from cv.experiments.connected_shooting import labeled_preparation_net_followup as followup
from cv.experiments.connected_shooting import labeled_net_ground_guidance as slots
from cv.experiments.connected_shooting import measured_dynamics
from cv.experiments.connected_shooting.flight_cache import FlightCache
from cv.pipeline import provenance
from scripts.shared_data import repository_root, resolve_shared_root

KEY = "uso2020f_pt0006_a01"


@contextmanager
def terminal_scales(first_frame, scales):
    """Worker-local explicit last-flight scales, with ordinary simulator restored."""
    values = measured_dynamics.validate_rebound_scales(scales)
    original = measured_dynamics.simulate

    def simulate(theta, first, queries, fps, surface, **kwargs):
        if first == first_frame:
            kwargs["rebound_scales"] = values
        return original(theta, first, queries, fps, surface, **kwargs)

    measured_dynamics.simulate = simulate
    try:
        yield
    finally:
        measured_dynamics.simulate = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["sparse", "context", "rebound"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = resolve_shared_root(repository_root(), None)
    inventory_path = root / "processed/ownerfix/adopted_perflight3/report.json"
    item = next(x for x in json.loads(inventory_path.read_text())["attempts"] if x["key"] == KEY)
    original, source, before, _, search, threshold, duration, loaded = triage.full.load_case(
        dict(item=item, rung=triage.RUNG)
    )
    labels = json.loads(loaded["arguments"].labels.read_text())
    camera_doc = json.loads(loaded["arguments"].cameras.read_text())
    context = original
    context_receipt = None
    if args.arm != "sparse":
        context, context_receipt = triage.census.extended_context(
            original, labels, camera_doc, search
        )
        if context is None:
            raise ValueError("original terminal context unavailable")
    check, consumed = followup.fit_check_copy(context["scene"], context["heldout"])
    scene, axes, added = triage.full.merge_scene(
        context["scene"], check, context["bounces"], context["axes"]
    )
    n = len(scene.pixels)
    last = n - 1
    selected = np.r_[
        np.arange(3 + 3 * last, 6 + 3 * last), np.arange(3 + 3 * n + 3 * last, 6 + 3 * n + 3 * last)
    ]
    first = float(scene.contact_frames[-2])
    event = next(
        e
        for e in context["events"]
        if e["event_type"] == "bounce" and e["frame"] == context["bounces"][-1][-1]
    )
    epoch = float(event["frame"])
    interval = event["frame_interval"]
    target = np.concatenate(scene.pixels)
    initial = source[selected].copy()
    scale = np.r_[[30.0] * 3, [3.0] * 3]
    lo = np.r_[[-75.0] * 3, [-6.0] * 3]
    hi = -lo
    if args.arm == "rebound":
        initial = np.r_[initial, source[-2:]]
        scale = np.r_[scale, 1.0, 1.0]
        lo = np.r_[lo, 0.8, 0.8]
        hi = np.r_[hi, 1.2, 1.2]
    initial = np.clip(initial, lo + 1e-9, hi - 1e-9) / scale
    cache = FlightCache(entries_per_flight=64)
    loss = followup.epoch.interval.block.event_constraints.mixed_loss(2 * len(target))
    start = time.monotonic()
    best = None
    calls = 0
    invalid = []
    count = None

    def expand(x):
        values = x * scale
        p = source.copy()
        p[selected] = values[:6]
        return p, values[6:] if args.arm == "rebound" else None

    def evaluate(x):
        nonlocal best, calls, count
        if time.monotonic() - start > args.seconds:
            raise TimeoutError("bounded terminal fit wall budget")
        calls += 1
        try:
            p, rebound = expand(x)
            with terminal_scales(first, rebound) if rebound is not None else nullcontext():
                native, physics, evidence = (
                    followup.epoch.interval.block.event_constraints.evaluate(
                        scene,
                        p,
                        context["bounces"],
                        2.0,
                        simulation_cache=cache,
                        net_clearance_scale_m=0.1,
                        terminal_rebound_frames=context.get("terminal_rebound_frames"),
                    )
                )
                impacts = native[-1]["bounces"]
                if not impacts:
                    theta = np.r_[
                        native[-1]["start_xyz"],
                        p[3 + 3 * last : 6 + 3 * last],
                        p[3 + 3 * n + 3 * last : 6 + 3 * n + 3 * last],
                    ]
                    impacts = measured_dynamics.simulate(
                        theta,
                        first,
                        np.array([first, scene.contact_frames[-1] + 2 * scene.fps]),
                        scene.fps,
                        scene.surface,
                        bounce_profile=scene.bounce_profile,
                        rebound_scales=p[-2:],
                    )[3]
                if not impacts:
                    raise ValueError("no terminal ground impact in bounded physical forecast")
                frame = float(impacts[0]["frame"])
                delta = frame - epoch
                physics = np.array(physics, copy=True)
                physics[slots.terminal_bounce_slot(evidence)] = (
                    np.sign(delta)
                    * max(abs(delta) - 2.0, 0)
                    / followup.epoch.interval.block.event_constraints.TIME_SCALE_FRAMES
                )
                timing = (min(frame - interval[0], 0) + max(frame - interval[1], 0)) / 0.1
                image = (
                    triage.full.exposure.prediction(
                        scene,
                        p,
                        axes,
                        duration,
                        cache,
                        termination_kind=context["termination_kind"],
                    )
                    - target
                )
            prior = (p[selected] - source[selected]) / np.r_[[30.0] * 3, [6.0] * 3]
            residual = np.r_[image.ravel(), physics, timing, prior]
            if rebound is not None:
                residual = np.r_[residual, (rebound - source[-2:]) / 0.2]
            count = len(residual)
            cost = float(2 * np.sum(loss((residual / 2) ** 2)[0]))
            if best is None or cost < best["cost"]:
                best = dict(
                    parameters=p.copy(),
                    terminal_rebound_scales=None if rebound is None else rebound.copy(),
                    cost=cost,
                    impact_frame=frame,
                    all_native_rms_px=float(np.sqrt(np.mean(np.sum(image**2, axis=1)))),
                    calls=calls,
                )
            return residual
        except ValueError as e:
            invalid.append(str(e))
            if count is None:
                raise
            return np.full(count, 1e5)

    records = loaded["sources"] + [provenance.file_record(inventory_path)]
    for path in [
        __file__,
        triage.__file__,
        triage.census.__file__,
        followup.__file__,
        measured_dynamics.__file__,
        triage.full.model.__file__,
    ]:
        records.append(provenance.file_record(path))
        shutil.copy2(path, args.output / Path(path).name)
    result = dict(
        status="running",
        key=KEY,
        arm=args.arm,
        sources=records,
        source_parameters=source,
        source_reproduction=loaded["reproduction"],
        before=before,
        policy=dict(
            maxiter=args.maxiter,
            seconds=args.seconds,
            original_interval=interval,
            original_labels_and_thresholds_unchanged=True,
            terminal_rebound_scales_bounds=[0.8, 1.2] if args.arm == "rebound" else None,
            point_rebound_scales_fixed=source[-2:],
            source_prior_velocity_sigma=30.0,
            source_prior_spin_sigma=6.0,
            interval_guidance_scale_frames=0.1,
            original_first10_fixed=True,
            image_loss="soft_L1_scale2px",
            numerical_impact_forecast_seconds=2,
        ),
        context_receipt=context_receipt,
        fit_only_duplicate_check_rows_removed=consumed,
        added_check_frames=added,
        original_rung=triage.RUNG,
    )

    def write():
        (args.output / "report.json").write_text(
            json.dumps(triage.full.profile.jsonable(result), indent=2) + "\n"
        )

    write()
    evaluate(initial)

    def jac(x):
        columns = []
        for j in range(len(x)):
            a, b = x.copy(), x.copy()
            a[j] = max(lo[j] / scale[j], x[j] - 1e-6)
            b[j] = min(hi[j] / scale[j], x[j] + 1e-6)
            columns.append((evaluate(b) - evaluate(a)) / (b[j] - a[j]))
        return np.column_stack(columns)

    try:
        fit = least_squares(
            evaluate,
            initial,
            jac=jac,
            bounds=(lo / scale, hi / scale),
            loss=loss,
            f_scale=2,
            max_nfev=args.maxiter,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        termination = dict(
            success=bool(fit.success), message=fit.message, nfev=fit.nfev, njev=fit.njev
        )
    except TimeoutError as e:
        termination = dict(success=False, message=str(e))
    p = best["parameters"]
    rebound = best["terminal_rebound_scales"]
    with terminal_scales(first, rebound) if rebound is not None else nullcontext():
        verdict, measurement = triage.census.score(
            context,
            p,
            threshold,
            duration,
            item["metadata"]["ending_kind"],
            scorer=triage.scope.LOCAL_SCORE,
        )
        physical = triage.full.model.chain(context["scene"], p)
    original_path = triage.full.model.chain(original["scene"], source)
    drift = max(
        float(np.max(abs(a["positions"] - b["positions"])))
        for a, b in zip(original_path[:-1], physical[:-1], strict=True)
    )
    assert drift == 0
    result.update(
        status="measured",
        best=best,
        termination=termination,
        seconds=time.monotonic() - start,
        residual_calls=calls,
        invalid_calls=len(invalid),
        invalid_reasons=dict((x, invalid.count(x)) for x in set(invalid)),
        after_original_witness=verdict,
        measurement=measurement,
        prefix_drift_m=drift,
        scene=asdict(context["scene"]),
        physical_terminal=physical[-1],
    )
    write()
    print(args.arm, verdict["accepted_flight_count"], best["impact_frame"], flush=True)


if __name__ == "__main__":
    main()
