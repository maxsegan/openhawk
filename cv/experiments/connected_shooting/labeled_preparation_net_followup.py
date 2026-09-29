"""Fit an explicitly prepared post-net attempt with the existing net-epoch chart.

Run --source-dir DIR --output DIR. Reconstructs and checks the frozen net-bounded
witness scene, preserves terminal-rebound semantics, fixes the earlier prefix,
and uses original net physics. Existing post-ending native rows are consumed
input evidence. No labels, gate thresholds or native timestamps are changed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_context_census as census
from cv.experiments.connected_shooting import labeled_net_epoch_fit as epoch
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as bounded
from cv.experiments.connected_shooting import labeled_preparation_net_witness as witness
from cv.experiments.connected_shooting import per_flight_rescore as score
from cv.experiments.connected_shooting.labeled_preparation_recovery_review import RUNG
from cv.pipeline import provenance
from scripts.shared_data import repository_root, resolve_shared_root


def load(source: Path):
    command = json.loads((source / "score_command.json").read_text())

    def arg(name, default=None):
        return command[command.index("--" + name) + 1] if "--" + name in command else default

    args = SimpleNamespace(
        labels=Path(arg("labels")),
        packet=Path(arg("packet")),
        cameras=Path(arg("cameras")),
        pose_csv=Path(arg("pose-csv")),
        pose_image_scale=float(arg("pose-image-scale")),
        athlete_prior_mode="stature_pose_soft",
        observation_fallback="on",
        dense_labels=None,
        player_ledger=None,
        player_order=[command[i + 1] for i, x in enumerate(command) if x == "--player-order"],
        witness_surface=arg("witness-surface"),
    )
    report = json.loads((source / "search/report.json").read_text())
    scored = json.loads((source / "perflight").read_text())
    if report.get("experimental_net_support_policy") != bounded.POLICY:
        raise ValueError("explicit prepared net-support source required")
    if (
        provenance.file_record(source / "search/report.json")["sha256"]
        != scored["search_report"]["sha256"]
    ):
        raise ValueError("scored source changed")
    for record in scored["inputs"]:
        base = (
            resolve_shared_root(repository_root(), None)
            if record["path_base"] == "TENNIS_DATA_ROOT"
            else repository_root()
        )
        if provenance.file_record(base / record["path"])["sha256"] != record["sha256"]:
            raise ValueError("source input changed: " + record["path"])
    labels = json.loads(args.labels.read_text())
    nets = [
        e
        for e in labels["events"]["records"]
        if e["event_type"] == "net_hit" and e["status"] == "labeled"
    ]
    old = whole.event_ground_target

    def target(event, cameras, pixels, radii=None, **kw):
        eligible, _ = bounded.net_bounded_frames(event, kw["eligible_frames"], nets)
        kw["eligible_frames"] = np.asarray(eligible, float)
        if report.get("experimental_net_witness_context_policy") == witness.POLICY:
            pixels, radii, _ = witness.augment_visible(
                pixels, radii, labels["ball"]["records"][0]["frames"], eligible, cameras
            )
        return old(event, cameras, pixels, radii, **kw)

    try:
        whole.event_ground_target = target
        context = score.build_context(args, report)
    finally:
        whole.event_ground_target = old
    rung = next(r for r in scored["rungs"] if r["rung"] == RUNG)
    candidates = [
        c
        for c in report["refined_candidates"]
        if c["depth_hypothesis_m"] == rung["selected_depth_m"]
    ]
    if len(candidates) != 1:
        raise ValueError("unique selected source vector required")
    candidate = candidates[0]
    receipt = score.reproduction_receipt(context, candidate, "stature_pose_soft")
    if not receipt["identical"]:
        raise ValueError("source candidate does not reproduce")
    threshold = next(r["thresholds"] for r in score.ladder() if r["name"] == RUNG)
    return (
        context,
        np.asarray(candidate["measurement"]["fit"]["parameters"]),
        threshold,
        float(report["configuration"]["exposure_duration_frames"]),
        arg("ending-kind"),
        report,
        labels,
        candidate,
        rung,
        receipt,
    )


def fit_check_copy(scene, heldout):
    """Remove only exact already-consumed check rows from the fit merge copy."""
    masks, consumed = [], []
    for i, (train, check) in enumerate(
        zip(scene.observation_frames, heldout.observation_frames, strict=True)
    ):
        mask = np.ones(len(check), bool)
        used = []
        for j, frame in enumerate(check):
            match = np.flatnonzero(train == frame)
            if len(match):
                if len(match) != 1:
                    raise ValueError("duplicate training exposure")
                for field in ["pixels", "cameras", "camera_distortion"]:
                    a, b = getattr(scene, field), getattr(heldout, field)
                    if a is None and b is None:
                        continue
                    if a is None or b is None or not np.array_equal(a[i][match[0]], b[i][j]):
                        raise ValueError("overlapping native observation differs")
                mask[j] = False
                used.append(float(frame))
        masks.append(mask)
        consumed.append(used)
    changes = {
        name: tuple(a[mask] for a, mask in zip(getattr(heldout, name), masks, strict=True))
        for name in ["observation_frames", "pixels", "cameras"]
    }
    if heldout.camera_distortion is not None:
        changes["camera_distortion"] = tuple(
            a[mask] for a, mask in zip(heldout.camera_distortion, masks, strict=True)
        )
    return replace(heldout, **changes), consumed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c, p, threshold, duration, ending, search, labels, candidate, rung, receipt = load(
        args.source_dir
    )
    before, _ = census.score(c, p, threshold, duration, ending)
    source_gate_verdict = {
        k: v
        for k, v in rung["verdict"].items()
        if k not in {"ending_completed", "terminal_completion_reason", "maximum_junction_gap_m"}
    }
    if before != source_gate_verdict:
        raise ValueError("source per-flight gate verdict drift")
    records = [
        provenance.file_record(args.source_dir / "search/report.json"),
        provenance.file_record(args.source_dir / "perflight"),
    ]
    scored = json.loads((args.source_dir / "perflight").read_text())
    records.extend(scored["inputs"])
    for path in [
        __file__,
        epoch.__file__,
        epoch.chart.__file__,
        epoch.tape.__file__,
        bounded.__file__,
        witness.__file__,
    ]:
        records.append(provenance.file_record(path))
        shutil.copy2(path, args.output / Path(path).name)
    result = dict(
        status="running",
        source_dir=str(args.source_dir),
        sources=records,
        source_reproduction=receipt,
        source_verdict_identical=True,
        before=before,
        source_parameters=p,
        human_derived=True,
        automatic_inference_eligible=False,
        independent_xyz_truth=False,
        policy=dict(
            maxiter=args.maxiter,
            seconds=args.seconds,
            original_net_physics=True,
            fixed_earlier_prefix=True,
            terminal_rebound_frames=c["terminal_rebound_frames"],
            activated_pixels_consumed_not_independent_validation=True,
        ),
    )

    def write():
        (args.output / "report.json").write_text(
            json.dumps(epoch.full.profile.jsonable(result), indent=2) + "\n"
        )

    write()
    original = epoch.interval.block.event_constraints.evaluate

    def rebound_evaluate(*a, **kw):
        kw["terminal_rebound_frames"] = c["terminal_rebound_frames"]
        return original(*a, **kw)

    try:
        epoch.interval.block.event_constraints.evaluate = rebound_evaluate
        check_copy, consumed = fit_check_copy(c["scene"], c["heldout"])
        fit_context = {**c, "heldout": check_copy}
        result["fit_merge_already_consumed_frames"] = consumed
        fitted = epoch.fit(fit_context, p, duration, maxiter=args.maxiter, seconds=args.seconds)
        selected = fitted["best"]["parameters"]
        after, measurement = census.score(c, selected, threshold, duration, ending)
        physical = epoch.full.model.chain(c["scene"], selected)
        original_physical = epoch.full.model.chain(c["scene"], p)
        prefix_drift = max(
            float(np.max(abs(a["positions"] - b["positions"])))
            for a, b in zip(original_physical[:-1], physical[:-1], strict=True)
        )
        result.update(
            status="measured",
            fit=fitted,
            after=after,
            measurement=measurement,
            final_net_hits=physical[-1]["net_hits"],
            prefix_drift_m=prefix_drift,
        )
        rendered = deepcopy(search)
        chosen = deepcopy(candidate)
        chosen["measurement"] = measurement
        rendered.update(selected=chosen, diagnostic_candidate=chosen, selected_arms={})
        result["artifacts"] = whole.render(rendered, labels, args.output)
    except (ValueError, TimeoutError) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        epoch.interval.block.event_constraints.evaluate = original
        write()
    print(result["status"], result.get("after", {}).get("accepted_flight_count"), flush=True)


if __name__ == "__main__":
    main()
