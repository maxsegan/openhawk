"""Matched frozen-seed continuation with original versus all native labeled pixels.

Opened evaluation only. A separate fitting scene activates the old frame%5
check rows; the original acceptance scene, witnesses and thresholds stay frozen.
All89 attempts remain in the denominator. Activated rows are no longer held out.
Run all source/export preflights before any matched continuation is launched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import (
    local_flight_repair_probe as local,
    model,
    per_flight_rescore as rescore,
    real_exposure_replay as exposure,
    serve_timing_profile as profile,
)
from cv.pipeline import paths, provenance


def merge_scene(scene, heldout, bounces, axes):
    """Activate rows with their exact prior observation metadata and scoring axis."""
    boundaries = np.unique(np.r_[scene.contact_frames, *bounces])
    train_frames = np.concatenate(scene.observation_frames)
    wings = np.searchsorted(boundaries, train_frames, side="right")
    axes = np.asarray(axes)
    groups = []
    offset = 0
    added = []
    for i, (train, check) in enumerate(
        zip(scene.observation_frames, heldout.observation_frames, strict=True)
    ):
        new = []
        for frame in check:
            ix = np.flatnonzero(wings == np.searchsorted(boundaries, frame, side="right"))
            if not len(ix):
                ix = np.flatnonzero(wings == np.searchsorted(boundaries, frame, side="left"))
            if not len(ix):
                raise ValueError("activated frame lacks original training-wing axis")
            j = ix[np.argmin(abs(train_frames[ix] - frame))]
            new.append(axes[j])
        combined = np.r_[train, check]
        order = np.argsort(combined)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("overlapping original train/check membership")
        groups.append(
            (
                order,
                np.concatenate(
                    [axes[offset : offset + len(train)], np.asarray(new).reshape(-1, 2)]
                )[order],
            )
        )
        offset += len(train)
        added.append(check.tolist())

    def merged(name):
        left = getattr(scene, name)
        right = getattr(heldout, name)
        if left is None:
            if right is not None:
                raise ValueError("camera distortion metadata differs between splits")
            return None
        return tuple(
            np.concatenate([np.asarray(a), np.asarray(b)])[order]
            for a, b, (order, _) in zip(left, right, groups, strict=True)
        )

    active = replace(
        scene,
        observation_frames=merged("observation_frames"),
        cameras=merged("cameras"),
        pixels=merged("pixels"),
        camera_distortion=merged("camera_distortion"),
    )
    active.validate()
    return active, np.concatenate([a for _, a in groups]), added


def load_case(job):
    item = job["item"]
    pf = json.loads(Path(item["document"]).read_text())
    saved = next(r for r in pf["rungs"] if r["rung"] == job["rung"])
    searchpath = local._resolve(pf["search_report"])
    search = json.loads(searchpath.read_text())
    source = searchpath.parent.parent
    candidate = next(
        c
        for c in search["refined_candidates"]
        if c["depth_hypothesis_m"] == saved["selected_depth_m"]
    )
    a = SimpleNamespace(
        labels=paths.REPO_ROOT
        / "cv/validation/labels/s6_agent_inputs_v1"
        / item["metadata"]["label_file"],
        packet=source / "inputs/packet.json",
        cameras=source / "inputs/cameras.json",
        pose_csv=paths.data_root() / "processed" / item["metadata"]["player_localization"],
        pose_image_scale=item["metadata"]["pose_image_scale"],
        athlete_prior_mode="stature_pose_soft",
        player_order=[n for n, _ in item["metadata"]["player_statures_m"]],
        observation_fallback="on",
        dense_labels=None,
        player_ledger=None,
        witness_surface=item["metadata"]["surface"],
    )
    ctx = rescore.build_context(a, search)
    receipt = rescore.reproduction_receipt(ctx, candidate, a.athlete_prior_mode)
    if not receipt["identical"]:
        raise ValueError("frozen original scene did not reproduce")
    if ctx.get("terminal_rebound_frames") is not None:
        raise ValueError("external terminal extension is outside this arm")
    params = np.asarray(candidate["measurement"]["fit"]["parameters"])
    threshold = next(r["thresholds"] for r in rescore.ladder() if r["name"] == job["rung"])
    duration = search["configuration"]["exposure_duration_frames"]
    before, measurement = profile.score(
        ctx, params, threshold, duration, item["metadata"]["ending_kind"]
    )
    if [f["checks"] for f in before["flights"]] != [
        f["checks"] for f in saved["verdict"]["flights"]
    ]:
        raise ValueError("original selected acceptance checks did not reproduce")
    configpath = next(
        local._resolve(r)
        for r in search["inputs"]
        if r["path"].endswith("/fixed_configuration.json")
    )
    config = json.loads(configpath.read_text())
    return (
        ctx,
        params,
        before,
        measurement,
        search,
        threshold,
        duration,
        dict(
            sources=[
                provenance.file_record(p)
                for p in [
                    Path(item["document"]),
                    searchpath,
                    configpath,
                    a.labels,
                    a.packet,
                    a.cameras,
                    a.pose_csv,
                ]
            ],
            configuration=config,
            arguments=a,
            reproduction=receipt,
        ),
    )


def endpoint_difference(left, right):
    """Compare the actual contact state, never differently timed last observations."""
    return max(
        float(np.linalg.norm(np.asarray(a["end_xyz"]) - b["end_xyz"]))
        for a, b in zip(left, right, strict=True)
    )


def preflight(job):
    row = dict(
        key=job["item"]["key"], status="pending", labeled_flights=job["item"]["labeled_flights"]
    )
    if not job["item"].get("fitted") or not job["item"].get("document"):
        return row | dict(status="no_selected_fitted_source")
    try:
        ctx, p, before, measurement, search, threshold, duration, loaded = load_case(job)
        active, axes, added = merge_scene(ctx["scene"], ctx["heldout"], ctx["bounces"], ctx["axes"])
        predicted = exposure.prediction(
            active, p, axes, duration, termination_kind=ctx["termination_kind"]
        )
        frames = np.concatenate(active.observation_frames)
        original = {x["frame"]: x for x in measurement["native_projection"]}
        expected = np.asarray([original[int(f)]["predicted"] for f in frames])
        pixel_delta = float(np.max(np.linalg.norm(predicted - expected, axis=1)))
        old = model.chain(ctx["scene"], p)
        new = model.chain(active, p)
        endpoint_delta = endpoint_difference(old, new)
        if pixel_delta > 1e-4 or endpoint_delta > 1e-3:
            raise ValueError(
                f"zero-step prediction/endpoint mismatch: {pixel_delta} px, {endpoint_delta} m"
            )
        row.update(
            status="passed",
            sources=loaded["sources"],
            maximum_zero_step_pixel_delta=pixel_delta,
            maximum_endpoint_delta_m=endpoint_delta,
            training_rows=sum(map(len, ctx["scene"].observation_frames)),
            activated_rows=sum(map(len, added)),
            activated_frames=added,
            original_contact_frames=ctx["scene"].contact_frames.tolist(),
            evaluation_witnesses_hash=hashlib.sha256(
                json.dumps(profile.jsonable(ctx["targets"]), sort_keys=True).encode()
            ).hexdigest(),
            before=before,
            parameters=p.tolist(),
            source_reproduction=loaded["reproduction"],
        )
    except (ValueError, TypeError, KeyError, StopIteration, FloatingPointError) as error:
        row.update(
            status="held_preflight_execution_or_support", reason=f"{type(error).__name__}: {error}"
        )
    return row


def continuation(job):
    result = dict(
        key=job["item"]["key"],
        arms={},
        status="pending",
        human_derived=True,
        activated_rows_independent=False,
    )
    output = Path(job["output"]) / "cases" / job["item"]["key"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        ctx, initial, before, _, search, threshold, duration, loaded = load_case(job)
        active, active_axes, added = merge_scene(
            ctx["scene"], ctx["heldout"], ctx["bounces"], ctx["axes"]
        )
        options = search["configuration"]["solver_experimental_arm"]
        base_arm = loaded["configuration"]["solver_experimental_arm"]
        anchors = exposure.anchor_plan(
            ctx["legacy_targets"],
            exposure.launch_contact_targets(ctx["players"]),
            player_sigma_m=options["anchor_player_sigma_m"],
            closing_contact=exposure.context_closing_anchor(
                ctx, player_sigma_m=options["anchor_player_sigma_m"]
            ),
        )
        result.update(
            before=before,
            sources=loaded["sources"],
            original_contact_frames=ctx["scene"].contact_frames.tolist(),
            activated_frames=added,
            original_witnesses=profile.jsonable(ctx["targets"]),
        )

        def alarm(*_):
            raise TimeoutError("matched continuation arm budget exhausted")

        old_handler = signal.signal(signal.SIGALRM, alarm)
        for name, scene, axes in [
            ("original_training", ctx["scene"], ctx["axes"]),
            ("all_native_pixels", active, active_axes),
        ]:
            start = time.monotonic()
            signal.alarm(job["timeout"])
            try:
                fit = exposure.refine(
                    scene,
                    initial.copy(),
                    ctx["bounces"],
                    ctx["native"],
                    axes,
                    duration,
                    job["maxiter"],
                    first_contact_y_m=float(initial[1]),
                    termination_kind=ctx["termination_kind"],
                    directional_event_frames=[r["frame"] for r in ctx["events"]],
                    directional_rms_limit_px=16.0,
                    inequality_constraints=base_arm["refine_inequalities"] != "out",
                    directional_inequalities=base_arm["refine_inequalities"] != "terminal_only",
                    anchor_targets=anchors if base_arm["anchor_residuals"] == "present" else None,
                    bounce_bracket_frames=base_arm["bounce_bracket_frames"],
                )
                p = np.asarray(fit["parameters"])
                verdict, measurement = profile.score(
                    ctx, p, threshold, duration, job["item"]["metadata"]["ending_kind"]
                )
                chain = model.chain(scene, p)
                gates_chain = model.chain(ctx["scene"], p)
                drift = endpoint_difference(chain, gates_chain)
                eligible = local.improvement_allowed(before, verdict, drift)
                result["arms"][name] = dict(
                    status="measured",
                    fit=fit,
                    verdict=verdict,
                    eligible_improvement=eligible,
                    export_endpoint_drift_m=drift,
                    measurement=measurement,
                    fit_consumed_old_withheld_pixels=name == "all_native_pixels",
                    original_evaluation_scene_preserved=True,
                )
            except (ValueError, TimeoutError, FloatingPointError, OverflowError) as error:
                result["arms"][name] = dict(
                    status="held_execution_or_optimizer", reason=f"{type(error).__name__}: {error}"
                )
            finally:
                signal.alarm(0)
            result["arms"][name]["wall_seconds"] = time.monotonic() - start
            (output / "report.json").write_text(
                json.dumps(profile.jsonable(result), indent=2, allow_nan=False) + "\n"
            )
        signal.signal(signal.SIGALRM, old_handler)
        result["status"] = "complete"
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        result.update(status="held_execution_or_support", reason=str(error))
    (output / "report.json").write_text(
        json.dumps(profile.jsonable(result), indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--run", action="store_true")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(cli.baseline_report.read_text())
    jobs = [
        dict(
            item=a, rung=cli.rung, output=str(cli.output), maxiter=cli.maxiter, timeout=cli.timeout
        )
        for a in base["attempts"]
    ]
    sources = []
    for module in [
        Path(__file__),
        Path(exposure.__file__),
        Path(rescore.__file__),
        Path(profile.__file__),
        Path(local.__file__),
    ]:
        sources.append(provenance.file_record(module))
        (cli.output / module.name).write_bytes(module.read_bytes())
    report = dict(
        schema="matched_full_native_continuation_v1",
        scope=__doc__,
        job={**vars(cli), "baseline_report": str(cli.baseline_report), "output": str(cli.output)},
        baseline=provenance.file_record(cli.baseline_report),
        source=sources,
        promoted=False,
        preflight=[],
        results=[],
        attempt_denominator=len(jobs),
        labeled_flight_denominator=sum(a["labeled_flights"] or 0 for a in base["attempts"]),
    )

    def write():
        (cli.output / "report.json").write_text(
            json.dumps(profile.jsonable(report), indent=2, allow_nan=False) + "\n"
        )

    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        for row in pool.map(preflight, jobs):
            report["preflight"].append(row)
            write()
            print("preflight", row["key"], row["status"], flush=True)
    failed = [
        r for r in report["preflight"] if r["status"] not in ["passed", "no_selected_fitted_source"]
    ]
    report["preflight_complete"] = True
    report["continuation_authorized_by_preflight"] = not failed
    if cli.run and not failed:
        eligible = {r["key"] for r in report["preflight"] if r["status"] == "passed"}
        with ProcessPoolExecutor(max_workers=cli.workers) as pool:
            for row in pool.map(continuation, [j for j in jobs if j["item"]["key"] in eligible]):
                report["results"].append(row)
                write()
                print("continuation", row["key"], row["status"], flush=True)
    report["status"] = (
        "held_failed_preflight" if failed else ("complete" if cli.run else "preflight_complete")
    )
    write()


if __name__ == "__main__":
    main()
