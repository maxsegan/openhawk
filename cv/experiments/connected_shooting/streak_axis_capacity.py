"""Remove privileged streak direction with matched native-visibility controls.

Opened synthetic full-point experiment, not an image detector. Preserve frozen
front noise; add independent2px trailing-tip noise. Unknown/out-of-image fronts
do not become pixel measurements. All native times remain in physical/XYZ scoring.
Derive axes from ordered visible endpoints, else an observed same-wing secant over
at most3 native frames. No truth-direction fallback or angular-error selection.
Axes are correlated plug-in estimates, not independent Gaussian likelihoods.
Known shutter/blur, exact events/cameras and centre-derived warm seeds remain.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import least_squares

from cv.experiments.connected_shooting import leading_edge_capacity as leading
from cv.pipeline import paths, provenance
from cv.validation.s6_owner_ground_camera import resolve_record


ARMS = ("known_axis_masked", "clean_pair_axis", "noisy_pair_axis")


def in_image(xy):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1:] != (2,):
        raise ValueError("Nx2 native pixels required")
    return np.isfinite(xy).all(axis=1) & ((xy >= 0) & (xy < [1920, 1080])).all(axis=1)


def observed_axes(front, back, frames, front_visible, back_visible, wings):
    front, back, frames = (
        np.asarray(front, float),
        np.asarray(back, float),
        np.asarray(frames, float),
    )
    fv, bv, wings = (
        np.asarray(front_visible, bool),
        np.asarray(back_visible, bool),
        np.asarray(wings),
    )
    n = len(frames)
    if (
        front.shape != (n, 2)
        or back.shape != (n, 2)
        or fv.shape != (n,)
        or bv.shape != (n,)
        or wings.shape != (n,)
        or not np.isfinite(frames).all()
        or np.any(np.diff(frames) <= 0)
        or not np.isfinite(front[fv]).all()
        or not np.isfinite(back[bv]).all()
    ):
        raise ValueError("aligned finite visible observations and ordered native frames required")
    axes = np.tile([1.0, 0.0], (n, 1))  # masked dummy only, never a direction fallback
    available = np.zeros(n, bool)
    sources = []
    for i in range(n):
        source = "front_unavailable"
        if fv[i]:
            candidates = [(front[i] - back[i], "endpoint_pair")] if bv[i] else []
            neighbors = np.flatnonzero(
                fv & (wings == wings[i]) & (abs(frames - frames[i]) <= 3) & (frames != frames[i])
            )
            order = sorted(
                neighbors, key=lambda j: (frames[j] < frames[i], abs(frames[j] - frames[i]))
            )
            candidates += [
                (
                    (front[j] - front[i]) * np.sign(frames[j] - frames[i]),
                    f"{'future' if j > i else 'past'}_secant:{int(frames[j])}",
                )
                for j in order
            ]
            source = "direction_unresolved"
            for vector, name in candidates:
                length = np.linalg.norm(vector)
                if length > 1e-6:
                    axes[i], available[i], source = vector / length, True, name
                    break
        sources.append(source)
    return axes, available, sources


def inputs(scene, bounces, reference, noise_seed):
    front = np.concatenate(reference["native_front_labels"])
    extents = np.asarray(reference["clean_extents"], float)
    frames = np.concatenate(scene.observation_frames)
    if front.shape != (len(frames), 2) or extents.shape != (len(frames), 2, 2):
        raise ValueError("frozen full native training inventory required")
    seed = int(
        hashlib.sha256(f"axis_tail_v1:{noise_seed}:{reference['point']}".encode()).hexdigest()[:16],
        16,
    )
    back = extents[:, 0] + np.random.default_rng(seed).normal(0, 2, front.shape)
    fv = in_image(extents[:, 1]) & in_image(front)
    clean_bv = in_image(extents[:, 0])
    noisy_bv = clean_bv & in_image(back)
    # Known physical events still condition this single-factor experiment. The
    # source exposure midpoint assigns a wing; no secant crosses a supplied knot.
    boundaries = np.unique(np.r_[scene.contact_frames, *bounces])
    wings = np.searchsorted(boundaries, frames + leading.DURATION / 2, side="right")
    clean_axes, clean_ok, clean_sources = observed_axes(
        extents[:, 1], extents[:, 0], frames, fv, clean_bv, wings
    )
    noisy_axes, noisy_ok, noisy_sources = observed_axes(front, back, frames, fv, noisy_bv, wings)
    mask = fv & clean_ok & noisy_ok
    target = front.copy()
    target[~mask] = 0  # no unavailable coordinates are supplied to optimization
    sizes = np.cumsum([len(f) for f in scene.observation_frames])[:-1]
    observed = replace(scene, pixels=tuple(np.split(target, sizes)))
    axes = {
        "known_axis_masked": np.asarray(reference["image_axes"], float),
        "clean_pair_axis": clean_axes,
        "noisy_pair_axis": noisy_axes,
    }
    if axes["known_axis_masked"].shape != front.shape:
        raise ValueError("frozen axis inventory required")
    angles = {
        name: np.degrees(np.arccos(np.clip(np.sum(a * axes["known_axis_masked"], axis=1), -1, 1)))
        for name, a in axes.items()
        if name != "known_axis_masked"
    }
    receipt = dict(
        frames=frames.tolist(),
        front_visible=fv.tolist(),
        image_residual_mask=mask.tolist(),
        original_native_training_count=len(frames),
        usable_front_count=int(mask.sum()),
        out_of_image_front_count=int((~fv).sum()),
        unresolved_visible_axis_count=int((fv & ~mask).sum()),
        clean_pair_sources=clean_sources,
        noisy_pair_sources=noisy_sources,
        front_pixels=[p.tolist() if v else None for p, v in zip(front, fv)],
        back_pixels=[p.tolist() if v else None for p, v in zip(back, noisy_bv)],
        axes={k: v.tolist() for k, v in axes.items()},
        scoring_only_angle_degrees={k: v[mask].tolist() for k, v in angles.items()},
        low_pair_snr_count=int(
            np.sum(mask & noisy_bv & (np.linalg.norm(front - back, axis=1) < 2 * np.sqrt(8)))
        ),
        whole_points_dropped=0,
        native_frames_discarded=0,
        truth_direction_fallback=False,
    )
    return observed, mask, axes, receipt


def refine(scene, initial, bounces, axes, mask, termination_kind, max_nfev):
    n = len(scene.pixels)
    target, mask = np.concatenate(scene.pixels), np.asarray(mask, bool)
    if mask.shape != (len(target),) or not mask.any():
        raise ValueError("nonempty explicit native observation mask required")
    axes = np.asarray(axes, float).copy()
    if axes.shape != target.shape:
        raise ValueError("one explicit axis per native observation required")
    axes[~mask] = [1, 0]
    imaging = leading.context.imaging_scene(scene, 0.5, termination_kind)
    cache = leading.flight_cache.FlightCache()
    loss = leading.event_constraints.mixed_loss(2 * len(target))

    def residual(p):
        # Preserve vector/frame inventory, but unknown observations supply exactly
        # zero likelihood. All frames still constrain full physical/XYZ evaluation.
        predicted = leading.image_prediction(imaging, p, axes, True, cache)
        image = np.zeros_like(target)
        image[mask] = predicted[mask] - target[mask]
        _, physical, _ = leading.event_constraints.evaluate(
            scene, p, bounces, 1, simulation_cache=cache
        )
        return np.r_[image.ravel(), physical, (p[-2:] - 1) / 0.02]

    initial = np.asarray(initial, float)
    before = residual(initial)

    def safe(p):
        try:
            r = residual(p)
            if np.isfinite(r).all():
                return r
        except (ValueError, FloatingPointError, OverflowError):
            pass
        return np.full(len(before), 1e6)

    lo = np.r_[[-10, -15, leading.model.R_BALL], [-75] * (3 * n), [-6] * (3 * n), [0.8, 0.8]]
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
        visible_front_euclidean_rms_px=float(
            np.sqrt(np.mean(np.sum(after[: 2 * len(target)].reshape(-1, 2)[mask] ** 2, axis=1)))
        ),
        usable_front_count=int(mask.sum()),
        maximum_seed_adjustment=float(np.max(abs(seed - initial))),
    )


def run_point(point, baseline, old, reference, max_nfev, timeout):
    scene, heldout, truth, bounces, native = leading.capacity.prepare(point, baseline)
    old_known = reference["arms"]["known_exposure_front"]
    replay = leading.capacity.score(
        point, scene, heldout, truth, np.asarray(old_known["fit"]["parameters"]), bounces, native
    )
    if replay["synthetic_correct"] != old_known["score"]["synthetic_correct"] or not np.allclose(
        [f["trajectory_rms_m"] for f in replay["spatial"]["flights"]],
        [f["trajectory_rms_m"] for f in old_known["score"]["spatial"]["flights"]],
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("known-axis geometry/correctness replay changed")
    observed, mask, axes, receipt = inputs(
        scene, bounces, reference, baseline["configuration"]["noise_seed"]
    )
    row = dict(
        point=point["point"],
        broadcast=point["match_id"],
        input=receipt,
        arms={},
        uncropped_known_axis_correct=old_known["correct"],
        native_scoring_frames=[f.tolist() for f in native],
    )

    def deadline(*_):
        raise TimeoutError("bounded streak-axis fit")

    for name in ARMS:
        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(timeout)
        try:
            fit = refine(
                observed,
                old["parameters"],
                bounces,
                axes[name],
                mask,
                point["termination_kind"],
                max_nfev,
            )
            score = leading.capacity.score(
                point, scene, heldout, truth, np.asarray(fit["parameters"]), bounces, native
            )
            compatible = bool(fit["success"] and score["input_compatible"])
            row["arms"][name] = dict(
                status="measured",
                fit=fit,
                score=score,
                compatible=compatible,
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
        native_training_frames=sum(r["input"]["original_native_training_count"] for r in rows),
        usable_fronts=sum(r["input"]["usable_front_count"] for r in rows),
        unavailable_fronts=sum(r["input"]["out_of_image_front_count"] for r in rows),
        unresolved_visible_axes=sum(r["input"]["unresolved_visible_axis_count"] for r in rows),
        uncropped_known_axis_correct=sum(r["uncropped_known_axis_correct"] for r in rows),
        arms={
            name: dict(
                correct=sum(r["arms"][name]["correct"] for r in rows),
                spatially_wrong_physical=sum(
                    r["arms"][name]["compatible"] and not r["arms"][name]["correct"] for r in rows
                ),
                converged=sum(r["arms"][name].get("fit", {}).get("success", False) for r in rows),
                held=sum(r["arms"][name]["status"] == "held" for r in rows),
            )
            for name in ARMS
        },
    )


def visualize(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(np.arange(3), [report["summary"]["arms"][n]["correct"] for n in ARMS])
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=["known axis", "clean pair + secant", "noisy pair + secant"],
        ylim=(0, report["summary"]["denominator"]),
        ylabel="strict correct / full point inventory",
    )
    for name in ARMS[1:]:
        values = np.sort(
            np.concatenate(
                [r["input"]["scoring_only_angle_degrees"][name] for r in report["results"]]
            )
        )
        if len(values):
            axes[1].plot(values, np.arange(1, len(values) + 1) / len(values), label=name)
    axes[1].set(
        xlabel="axis error versus generating direction (degrees)",
        ylabel="fraction of usable observations",
        xlim=(0, 180),
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("Native-visibility axis control — synthetic known exposure, not real-video yield")
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.output.exists() or min(args.jobs, args.max_nfev, args.timeout) < 1:
        raise ValueError("new output and positive budgets required")
    reference, baseline = [json.loads(p.read_text()) for p in (args.reference, args.baseline)]
    if (
        reference["schema"] != "s6_leading_edge_capacity_v1"
        or reference["status"] != "complete"
        or reference["configuration"]["terminal_image_context_frames"] != 0.5
        or (
            reference["configuration"]["duration_frames"],
            reference["configuration"]["open_offset_frames"],
            reference["configuration"]["blur_radius_px"],
        )
        != (0.5, 0, 1.5)
        or provenance.file_record(args.baseline) not in reference["inputs"]
    ):
        raise ValueError("matching full-context frozen reference/baseline required")
    inventory = reference["inventory"]
    for rows in (reference["results"], baseline["results"]):
        if (
            len(rows) != len(inventory)
            or len(set(inventory)) != len(inventory)
            or {r["point"] for r in rows} != set(inventory)
        ):
            raise ValueError("complete unique point inventory required")
    files = [
        args.reference,
        args.baseline,
        Path(__file__),
        *[resolve_record(r) for r in reference["inputs"]],
    ]
    bindings = [provenance.file_record(p) for p in dict.fromkeys(files)]
    points = {
        p["point"]: p
        for p in json.loads(Path(baseline["configuration"]["truth"]).read_text())["points"]
    }
    refs = {p["point"]: p for p in reference["results"]}
    report = dict(
        schema="s6_streak_axis_capacity_v1",
        status="running",
        scope=__doc__,
        inputs=bindings,
        code=provenance.git_record(paths.REPO_ROOT),
        inventory=inventory,
        configuration=dict(
            jobs=args.jobs,
            max_nfev=args.max_nfev,
            timeout=args.timeout,
            tail_sigma_px=2,
            noise_seed=baseline["configuration"]["noise_seed"],
            max_secant_gap_frames=3,
            matched_visibility_mask=True,
            angular_error_never_selects_inputs=True,
        ),
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
                refs[r["point"]],
                args.max_nfev,
                args.timeout,
            )
            for r in baseline["results"]
        ]
        for future in as_completed(futures):
            report["results"].append(future.result())
            write()
            print(f"completed {len(report['results'])}/{len(inventory)}", flush=True)
    for record in bindings:
        resolve_record(record)
    report.update(summary=summarize(report["results"]), wall_seconds=time.monotonic() - start)
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
