"""Does interpreting visible front tips as nominal centres bias complete S6 fits?

Opened broadcast-camera synthetic capacity only. Known duration0.5frame, opening
at the native timestamp, blur1.5px and noiseless image orientation are privileged
conditions, not camera metadata or an automatic producer. Generate directional
silhouette extrema from recorded truth, including contact/impact curves. Retain
the same2px training noise and frozen image-derived baseline initializer in both
arms. Reject an entire condition if any native exposure lacks recorded context;
never fabricate a toss/ending continuation or drop a boundary image to gain yield.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import (
    camera_geometry,
    event_constraints,
    flight_cache,
    model,
    oracle_benchmark as oracle,
    serve_region_capacity as capacity,
    swept_exposure as swept,
    terminal_exposure_context as context,
    terminal_feasibility as terminal,
)
from cv.pipeline import paths, provenance
from cv.validation.s6_owner_ground_camera import resolve_record


DURATION, OPEN_OFFSET, BLUR = 0.5, 0.0, 1.5


def synthetic_labels(scene, truth):
    rows = swept.truth_curves(scene, truth, DURATION, OPEN_OFFSET)
    axes, extents, internal_fronts = [], [], 0
    for row in rows:
        if row["status"] != "covered":
            raise ValueError("full native exposure context required before generating labels")
        camera = scene.cameras[row["flight"]][row["index"]]
        projected = camera_geometry.project(
            np.repeat(camera[None], len(row["positions"]), axis=0), row["positions"]
        )
        axis = projected[-1] - projected[0]
        if np.linalg.norm(axis) < 1e-6:
            raise ValueError("unresolved synthetic image orientation; no invented direction")
        axis /= np.linalg.norm(axis)
        tips, selected = swept.directional_tips(row["positions"], camera, axis, blur_radius_px=BLUR)
        axes.append(axis)
        extents.append(tips)
        internal_fronts += int(selected[1] not in (0, len(row["positions"]) - 1))
    extents, axes = np.array(extents), np.array(axes)
    noisy, offset = [], 0
    for frames, cameras, original, path in zip(
        scene.observation_frames, scene.cameras, scene.pixels, truth, strict=True
    ):
        nominal = camera_geometry.project(cameras, oracle.sample(path, frames))
        noise = original - nominal
        noisy.append(extents[offset : offset + len(frames), 1] + noise)
        offset += len(frames)
    return replace(scene, pixels=tuple(noisy)), axes, extents, internal_fronts


def image_prediction(scene, parameters, axes, exposure, cache=None):
    if not exposure:
        flights = model.chain(scene, parameters, simulation_cache=cache)
        return np.concatenate(
            [
                camera_geometry.project(
                    c,
                    f["positions"],
                    None if scene.camera_distortion is None else scene.camera_distortion[i],
                )
                for i, (c, f) in enumerate(zip(scene.cameras, flights, strict=True))
            ]
        )
    rows = swept.fitted_curves(scene, parameters, DURATION, OPEN_OFFSET, cache=cache)
    return swept.predict(scene, rows, axes, BLUR)[:, 1]


def refine(
    scene,
    initial,
    bounces,
    axes,
    exposure,
    max_nfev,
    *,
    image_context_frames=0.0,
    termination_kind=None,
):
    n = len(scene.pixels)
    initial = np.asarray(initial, float)
    cache = flight_cache.FlightCache()
    target = np.concatenate(scene.pixels)
    loss = event_constraints.mixed_loss(2 * len(target))
    imaging = (
        context.imaging_scene(scene, image_context_frames, termination_kind)
        if image_context_frames
        else scene
    )

    def residual(p):
        image = image_prediction(imaging if exposure else scene, p, axes, exposure, cache) - target
        _, physical, _ = event_constraints.evaluate(scene, p, bounces, 1, simulation_cache=cache)
        return np.r_[image.ravel(), physical, (p[-2:] - 1) / 0.02]

    before = residual(initial)

    def safe(p):
        try:
            r = residual(p)
            if np.isfinite(r).all():
                return r
        except (ValueError, FloatingPointError, OverflowError):
            pass
        return np.full(len(before), 1e6)

    lo = np.r_[[-10, -15, model.R_BALL], [-75] * (3 * n), [-6] * (3 * n), [0.8, 0.8]]
    hi = np.r_[[21, 40, 12], [75] * (3 * n), [6] * (3 * n), [1.2, 1.2]]
    seed = np.clip(initial, lo + 1e-8, hi - 1e-8)
    fit = least_squares(
        safe, seed, bounds=(lo, hi), loss=loss, f_scale=2, x_scale="jac", max_nfev=max_nfev
    )
    after = residual(fit.x)
    return dict(
        parameters=fit.x.tolist(),
        success=bool(fit.success),
        status=int(fit.status),
        nfev=int(fit.nfev),
        initial_cost=float(2 * np.sum(loss((before / 2) ** 2)[0])),
        final_cost=float(2 * np.sum(loss((after / 2) ** 2)[0])),
        native_front_euclidean_rms_px=float(
            np.sqrt(np.mean(np.sum(after[: 2 * len(target)].reshape(-1, 2) ** 2, axis=1)))
        ),
        maximum_seed_adjustment=float(np.max(abs(seed - initial))),
    )


def run_point(point, baseline, old, audit, max_nfev, timeout, image_context_frames=0.0):
    scene, heldout, truth, bounces, native = capacity.prepare(point, baseline)
    initial = np.asarray(old["parameters"])
    before = capacity.score(point, scene, heldout, truth, initial, bounces, native)
    if (
        before["synthetic_correct"] != audit["oracle_physical_diagnostic_met"]
        or not np.allclose(
            [f["trajectory_rms_m"] for f in before["spatial"]["flights"]],
            [f["trajectory_rms_m"] for f in old["flights"]],
            rtol=1e-8,
            atol=1e-8,
        )
        or not np.isclose(
            terminal.objective(
                scene, initial, bounces, float(native[-1][-1]), terminal_guidance=False
            ),
            old["optimizer_evidence"]["cost"],
            rtol=1e-8,
            atol=1e-7,
        )
    ):
        raise ValueError("baseline spatial/physical/objective replay changed")
    native_frames = np.concatenate(native)
    imaging = context.imaging_scene(scene, image_context_frames, point["termination_kind"])
    unsupported = native_frames[
        (native_frames + OPEN_OFFSET < imaging.contact_frames[0])
        | (native_frames + OPEN_OFFSET + DURATION > imaging.contact_frames[-1])
    ]
    row = dict(
        point=point["point"],
        broadcast=point["match_id"],
        status="coverage_hold",
        baseline=before,
        native_frames=native_frames.tolist(),
        training_count=sum(map(len, scene.pixels)),
        unsupported_frames=unsupported.tolist(),
        scored_bounds_frames=scene.contact_frames.tolist(),
        imaging_bounds_frames=imaging.contact_frames.tolist(),
        arms={},
        automatic_inference_eligible=False,
    )
    if len(unsupported):
        return row
    imaging_truth = context.recorded_point(imaging, point) if image_context_frames else truth
    observed, axes, extents, internal = synthetic_labels(imaging, imaging_truth)
    observed = replace(observed, contact_frames=scene.contact_frames)
    row.update(
        status="measured",
        image_axes=axes.tolist(),
        clean_extents=extents.tolist(),
        native_front_labels=[p.tolist() for p in observed.pixels],
        front_extrema_at_interior_samples=internal,
    )

    def deadline(*_):
        raise TimeoutError("bounded leading-edge capacity fit")

    for name, exposure in (("wrong_center", False), ("known_exposure_front", True)):
        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(timeout)
        try:
            fit = refine(
                observed,
                initial,
                bounces,
                axes,
                exposure,
                max_nfev,
                image_context_frames=image_context_frames,
                termination_kind=point["termination_kind"],
            )
            # Score against original dense XYZ and centre-projection diagnostics only
            # after fitting. Neither original centre pixels nor heldout truth enters
            # these front-label objectives; the old centre fit is an explicit warm seed.
            score = capacity.score(
                point, scene, heldout, truth, np.asarray(fit["parameters"]), bounces, native
            )
            compatible = fit["success"] and score["input_compatible"]
            row["arms"][name] = dict(
                status="measured",
                fit=fit,
                score=score,
                compatible=bool(compatible),
                correct=bool(compatible and score["synthetic_correct"]),
            )
        except (ValueError, FloatingPointError, OverflowError, TimeoutError) as error:
            row["arms"][name] = dict(
                status="held", reason=str(error), compatible=False, correct=False
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            row["arms"][name]["wall_seconds"] = time.monotonic() - start
    return row


def summarize(rows):
    return dict(
        denominator=len(rows),
        exposure_covered=sum(r["status"] == "measured" for r in rows),
        coverage_holds=sum(r["status"] == "coverage_hold" for r in rows),
        original_center_baseline_correct=sum(r["baseline"]["synthetic_correct"] for r in rows),
        original_center_baseline_correct_on_covered=sum(
            r["baseline"]["synthetic_correct"] and r["status"] == "measured" for r in rows
        ),
        arms={
            name: dict(
                correct=sum(r["arms"].get(name, {}).get("correct", False) for r in rows),
                spatially_wrong_physical=sum(
                    a.get("compatible", False) and not a.get("correct", False)
                    for r in rows
                    for a in [r["arms"].get(name, {})]
                ),
                converged=sum(
                    r["arms"].get(name, {}).get("fit", {}).get("success", False) for r in rows
                ),
                fit_holds=sum(r["arms"].get(name, {}).get("status") == "held" for r in rows),
            )
            for name in ("wrong_center", "known_exposure_front")
        },
    )


def visualize(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    summary = report["summary"]
    names = ["wrong_center", "known_exposure_front"]
    axes[0].bar(
        [0, 1, 2],
        [
            summary["original_center_baseline_correct_on_covered"],
            *[summary["arms"][n]["correct"] for n in names],
        ],
    )
    axes[0].set(
        xticks=[0, 1, 2],
        xticklabels=["original centre labels", "front as centre", "known exposure front"],
        ylim=(0, summary["denominator"]),
        ylabel="strict correct points / full inventory",
    )
    axes[0].axhline(
        summary["exposure_covered"], color="gray", linestyle="--", label="complete exposure context"
    )
    axes[0].legend(fontsize=8)
    for name in names:
        pairs = [
            (
                r["baseline"]["spatial"]["flights"][0]["trajectory_rms_m"],
                r["arms"][name]["score"]["spatial"]["flights"][0]["trajectory_rms_m"],
            )
            for r in report["results"]
            if r["arms"].get(name, {}).get("status") == "measured"
        ]
        if pairs:
            x, y = np.asarray(pairs).T
            axes[1].scatter(x, y, label=name, s=18)
    limit = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, limit], [0, limit], color="gray")
    axes[1].set(
        xlabel="original centre-label serve RMS (m)", ylabel="front-label fit serve RMS (m)"
    )
    if axes[1].collections:
        axes[1].legend(fontsize=8)
    fig.suptitle("Exposure semantics — known synthetic nuisance, not real-video yield")
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--terminal-image-context-frames",
        type=float,
        default=0.0,
        help="Explicit <=1frame passive imaging context; scoring bounds stay unchanged",
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.jobs, args.max_nfev, args.timeout) < 1:
        raise ValueError("new output and positive budgets required")
    if (
        not np.isfinite(args.terminal_image_context_frames)
        or not 0 <= args.terminal_image_context_frames <= 1
    ):
        raise ValueError("explicit finite image context within [0,1] frame required")
    baseline = json.loads(args.baseline.read_text())
    audit_path = args.baseline.with_name("physical_audit.json")
    audit = json.loads(audit_path.read_text())
    cfg = baseline["configuration"]
    if (
        baseline["status"] != "complete"
        or provenance.file_record(args.baseline) not in audit["inputs"]
        or cfg["arms"] != ["pixels_noise2"]
        or cfg["training_visibility"] != "all"
        or cfg["bounce_time_conditioning"] != "exact"
    ):
        raise ValueError("complete bound all-visible noise2 exact-event baseline/audit required")
    inventory = baseline["point_inventory"]
    for records in (baseline["results"], audit["results"]):
        if (
            len(records) != len(inventory)
            or len(set(inventory)) != len(inventory)
            or {r["point"] for r in records} != set(inventory)
        ):
            raise ValueError("unique complete source inventory required")
    files = [args.baseline, audit_path, Path(__file__)]
    changes = []
    for binding in baseline["input_bindings"]:
        record = binding["record"]
        if record["path_base"] == "repository" and record["path"].endswith(".py"):
            source = (paths.REPO_ROOT / record["path"]).resolve()
            if not source.is_relative_to(paths.REPO_ROOT):
                raise ValueError("producer path escaped repository")
            current = provenance.file_record(source)
            if current != record:
                changes.append(dict(historical=record, current=current))
        else:
            source = resolve_record(record)
        files.append(source)
    files += [
        Path(m.__file__)
        for m in (
            swept,
            context,
            camera_geometry,
            event_constraints,
            flight_cache,
            model,
            oracle,
            capacity,
            terminal,
            capacity.physical_audit,
            capacity.physical_compatibility,
            capacity.terminal_completion,
        )
    ]
    bindings = [provenance.file_record(p) for p in dict.fromkeys(files)]
    points = {p["point"]: p for p in json.loads(Path(cfg["truth"]).read_text())["points"]}
    audits = {p["point"]: p for p in audit["results"]}
    report = dict(
        schema="s6_leading_edge_capacity_v1",
        status="running",
        inputs=bindings,
        code=provenance.git_record(paths.REPO_ROOT),
        scope=__doc__,
        configuration=dict(
            duration_frames=DURATION,
            open_offset_frames=OPEN_OFFSET,
            blur_radius_px=BLUR,
            noiseless_image_axis=True,
            max_nfev=args.max_nfev,
            timeout=args.timeout,
            jobs=args.jobs,
            noise_seed=cfg["noise_seed"],
            terminal_image_context_frames=args.terminal_image_context_frames,
            scoring_bounds_and_physical_objective_unchanged=True,
        ),
        historical_producer_changes=changes,
        inventory=inventory,
        results=[],
        automatic_inference_eligible=False,
        real_yield_established=False,
    )
    args.output.mkdir(parents=True)

    def write():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    write()
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                run_point,
                points[r["point"]],
                baseline,
                r,
                audits[r["point"]],
                args.max_nfev,
                args.timeout,
                args.terminal_image_context_frames,
            )
            for r in baseline["results"]
        ]
        for future in as_completed(futures):
            report["results"].append(future.result())
            write()
            print(f"completed {len(report['results'])}/{len(inventory)}", flush=True)
    for binding in bindings:
        resolve_record(binding)
    report.update(
        summary=summarize(report["results"]),
        wall_seconds=time.monotonic() - start,
    )
    report["by_broadcast"] = {
        name: summarize([r for r in report["results"] if r["broadcast"] == name])
        for name in sorted({r["broadcast"] for r in report["results"]})
    }
    visualize(report, args.output)
    report["status"] = "complete"
    write()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
