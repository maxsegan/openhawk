"""Opened bounded joint-toss profile at original contact bracket endpoints/midpoint.

Every candidate exports its actual scene and representative event epoch. Native
pictures, original interval, evidence memberships, bounce witnesses, outer right
endpoint and acceptance thresholds remain frozen. No XYZ truth chooses a fit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import signal

import numpy as np

from cv.experiments.connected_shooting import (
    athlete_priors,
    block_flight_repair_probe as block,
    local_flight_repair_probe as local,
    model,
    per_flight_acceptance as acceptance,
    per_flight_rescore as rescore,
    real_exposure_replay as exposure,
    serve_block_repair_probe as serve,
    toss_witness,
)
from cv.pipeline import paths, provenance


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def contact_epoch_bounds(original):
    """Intersect the original contact interval with fixed outgoing row ownership.

    A contact cannot move past an observation already assigned to its outgoing
    flight. Include both train and check rows, without moving or dropping either.
    A singleton domain fixes timing; an empty domain is an input inconsistency.
    """
    first = next(e for e in original["events"] if e["event_type"] == "contact")
    low, high = map(float, first["frame_interval"])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("invalid original contact interval")
    for name in ("scene", "heldout"):
        rows = np.asarray(original[name].observation_frames[0], dtype=float)
        if rows.size:
            if not np.isfinite(rows).all():
                raise ValueError("nonfinite outgoing observation epoch")
            high = min(high, float(np.min(rows)))
    if low > high:
        raise ValueError("empty contact domain under fixed observation ownership")
    return low, high


def profile_context(original, epoch):
    """Change physical contact epoch within its label interval, never native times."""
    ctx = dict(original)
    events = deepcopy(original["events"])
    first = next(e for e in events if e["event_type"] == "contact")
    low, high = first["frame_interval"]
    if not low <= epoch <= high:
        raise ValueError("profile epoch outside original label interval")
    first["original_representative_frame"] = first["frame"]
    first["frame"] = float(epoch)
    first["profiled_within_original_interval"] = True
    ctx["events"] = events
    for name in ["scene", "heldout"]:
        scene = original[name]
        frames = scene.contact_frames.copy()
        frames[0] = epoch
        if any(
            len(f) and (np.min(f) < a or np.max(f) > b)
            for f, a, b in zip(scene.observation_frames, frames[:-1], frames[1:], strict=True)
        ):
            raise ValueError("profile would reassign an original observation across contact")
        ctx[name] = replace(scene, contact_frames=frames)
    return ctx


def score(context, parameters, threshold, duration, ending_kind):
    measured = exposure.measure(
        context["scene"],
        context["heldout"],
        context["bounces"],
        context["native"],
        parameters,
        context["axes"],
        duration,
        [],
        termination_kind=context["termination_kind"],
    )
    athlete = athlete_priors.evaluate(np.asarray(measured["contact_xyz"]), context["players"])
    result = acceptance.measure(
        context["scene"],
        context["heldout"],
        parameters,
        context["bounces"],
        context["native"],
        context["axes"],
        context["targets"],
        context["players"],
        context["events"],
        termination_kind=context["termination_kind"],
        duration=duration,
        athlete=athlete,
        ending_passive_context_frames=threshold.ending_passive_context_frames,
        measurement=measured,
    )
    result["surface"] = context["witness_surface"]
    verdict = acceptance.score(
        result,
        threshold,
        depth_bounds=context["depth_bounds"],
        athlete_prior_mode="stature_pose_soft",
        ending_kind=ending_kind,
    )
    return verdict, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per profiled epoch")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(cli.baseline_report.read_text())
    item = next(a for a in base["attempts"] if a["key"] == cli.case)
    pf = json.loads(Path(item["document"]).read_text())
    saved = next(r for r in pf["rungs"] if r["rung"] == cli.rung)
    searchpath = local._resolve(pf["search_report"])
    search = json.loads(searchpath.read_text())
    source = searchpath.parent.parent
    configuration_path = next(
        local._resolve(r)
        for r in search["inputs"]
        if r["path"].endswith("/fixed_configuration.json")
    )
    configuration = json.loads(configuration_path.read_text())
    bounce_bracket = float(configuration["solver_experimental_arm"]["bounce_bracket_frames"])
    reference = next(
        c
        for c in search["refined_candidates"]
        if c["depth_hypothesis_m"] == saved["selected_depth_m"]
    )
    labels_path = (
        paths.REPO_ROOT / "cv/validation/labels/s6_agent_inputs_v1" / item["metadata"]["label_file"]
    )
    args = SimpleNamespace(
        labels=labels_path,
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
    ctx = rescore.build_context(args, search)
    receipt = rescore.reproduction_receipt(ctx, reference, args.athlete_prior_mode)
    if not receipt["identical"]:
        raise ValueError("original frozen scene failed reproduction")
    parameters = np.asarray(reference["measurement"]["fit"]["parameters"], float)
    threshold = next(r["thresholds"] for r in rescore.ladder() if r["name"] == cli.rung)
    duration = search["configuration"]["exposure_duration_frames"]
    before, _ = score(ctx, parameters, threshold, duration, item["metadata"]["ending_kind"])
    assert [f["accepted"] for f in before["flights"]] == [
        f["accepted"] for f in saved["verdict"]["flights"]
    ]
    ctx.update(repair_thresholds=threshold, repair_reference_verdict=before)
    original_shared = model.shared_contact_seed(
        replace(ctx["scene"], parameterization="shared_contact_states"), parameters
    )
    ctx["repair_initial_shared_parameters"] = original_shared.tolist()
    err = exposure.prediction(
        ctx["scene"], parameters, ctx["axes"], duration, termination_kind=ctx["termination_kind"]
    ) - np.concatenate(ctx["scene"].pixels)
    ctx["repair_contact_image_budgets"] = np.maximum(
        block.image_budgets(err, serve.serve_contact_image_groups(ctx["scene"])) + 1e-8, 16**2
    )
    # Resolve original evidence once. Never acquire an uncertain-interval frame for one profile only.
    labels = json.loads(labels_path.read_text())
    cameras = json.loads(args.cameras.read_text())
    cylinder = serve.serve_reach_cylinder.build(ctx["players"][0])
    low = np.array(
        [original_shared[0] - 1.5, ctx["depth_bounds"][0], cylinder.height_interval_m[0]]
    )
    high = np.array(
        [original_shared[0] + 1.5, ctx["depth_bounds"][1], cylinder.height_interval_m[1]]
    )
    evidence = serve.toss_evidence(
        ctx, labels, cameras, args.pose_csv, args.pose_image_scale, low, high
    )
    first = next(e for e in ctx["events"] if e["event_type"] == "contact")
    interval = first["frame_interval"]
    epochs = sorted(set([float(interval[0]), sum(interval) / 2, float(interval[1])]))
    sources = []
    for file in [
        Path(__file__),
        Path(serve.__file__),
        Path(serve.joint_toss_residual.__file__),
        Path(toss_witness.__file__),
        Path(local.__file__),
        Path(block.__file__),
    ]:
        sources.append(provenance.file_record(file))
        (cli.output / file.name).write_bytes(file.read_bytes())
    report = dict(
        schema="joint_serve_native_contact_profile_v1",
        scope=__doc__,
        job=vars(cli) | {"baseline_report": str(cli.baseline_report), "output": str(cli.output)},
        original_interval=interval,
        original_representative_epoch=first["frame"],
        rung=cli.rung,
        before=before,
        reproduction=receipt,
        profiles=[],
        sources=sources,
        inputs=[
            provenance.file_record(p)
            for p in [
                cli.baseline_report,
                searchpath,
                args.packet,
                args.cameras,
                labels_path,
                args.pose_csv,
            ]
        ],
        promoted=False,
        human_derived=True,
        automatic_inference_eligible=False,
    )

    def timeout(*_):
        raise TimeoutError("bounded timing profile epoch deadline")

    signal.signal(signal.SIGALRM, timeout)
    for epoch in epochs:
        signal.alarm(cli.timeout)
        c = profile_context(ctx, epoch)
        toss = deepcopy(evidence)
        toss["observations"]["contact_frame"] = epoch
        toss["estimate"] = toss_witness.fit_contact(
            toss["observations"],
            toss["feet"],
            c["scene"].fps,
            contact_frame_interval=tuple(interval),
            contact_bounds=(low, high),
            config=toss_witness.CONTACT_CONSTRAINT_CONFIG,
        )
        c["repair_toss_evidence"] = toss
        row = dict(
            epoch=epoch,
            scene=jsonable(asdict(c["scene"])),
            heldout_scene=jsonable(asdict(c["heldout"])),
            events=c["events"],
            frozen_right_contact_xyz_m=original_shared[6:9].tolist(),
            native_memberships_unchanged=True,
            frozen_bounce_witnesses_unchanged=c["targets"] == ctx["targets"],
            trials=[],
        )
        try:
            trials = serve.optimize_serve(
                c,
                parameters,
                0,
                maxiter=cli.maxiter,
                bounce_bracket_frames=bounce_bracket,
                duration=duration,
                labels=labels,
                cameras=cameras,
                pose_csv=args.pose_csv,
                pose_scale=args.pose_image_scale,
                raw_toss=True,
            )
            for trial in trials:
                verdict, measurement = score(
                    c,
                    np.asarray(trial["parameters"]),
                    threshold,
                    duration,
                    item["metadata"]["ending_kind"],
                )
                trial["verdict"] = verdict
                trial["eligible_improvement"] = local.improvement_allowed(
                    before, verdict, trial["maximum_endpoint_gap_m"]
                )
                trial["measurement"] = measurement
            selected = dict(status="measured", before=before, trials=trials)
            serve.finalize_selection(selected)
            row.update(selected)
        except (ValueError, TimeoutError) as error:
            row.update(status="held_execution_or_support", reason=str(error))
        finally:
            signal.alarm(0)
        report["profiles"].append(row)
        (cli.output / "report.json").write_text(
            json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
        )
        print(epoch, row["status"], row.get("selected_source"), flush=True)
    eligible = [
        (i, t)
        for i, row in enumerate(report["profiles"])
        for t in row["trials"]
        if t["eligible_improvement"]
    ]
    chosen = min(
        eligible,
        key=lambda it: (it[1]["cost"], report["profiles"][it[0]]["epoch"], it[1]["start"]),
        default=None,
    )
    report["selection_rule"] = (
        "Minimum unchanged input-only image/physical objective among full-gate and old-neighbor preserving candidates; no XYZ truth."
    )
    report["selected_profile_index"] = None if chosen is None else chosen[0]
    report["selected_trial_start"] = None if chosen is None else chosen[1]["start"]
    report["selected_source"] = "original_reference" if chosen is None else "repair_trial"
    (cli.output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    )
    print("selected", None if chosen is None else report["profiles"][chosen[0]]["epoch"])


if __name__ == "__main__":
    main()
