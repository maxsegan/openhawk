"""Can supported post-net context and continuous event coordinates recover net attempts?

Opened labeled development, explicit passive mesh and native-reviewed upper-net
height options. Use --case KEY --output DIR; --continuous-ground avoids a known
missing-height/time penalty jump, and --quadratic trusts all native image rows.
Defaults retain the original bounded-control settings for reproducibility.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
from cv.experiments.connected_shooting import labeled_net_epoch_fit as epoch
from cv.experiments.connected_shooting import labeled_preparation_net_followup as followup
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as bounded
from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.pipeline import provenance
from cv.experiments.connected_shooting import labeled_net_ground_guidance as horizon

f, census, scope = epoch.full, epoch.interval.census, epoch.interval.scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mesh", action="store_true")
    p.add_argument(
        "--free-net-velocity",
        action="store_true",
        help="Explicit independent outgoing velocity; incompatible with --mesh",
    )
    p.add_argument("--ground-guidance", action="store_true")
    p.add_argument("--net-band", action="store_true")
    p.add_argument("--joint", action="store_true")
    p.add_argument("--upper-net", action="store_true")
    p.add_argument("--continuous-ground", action="store_true")
    p.add_argument("--quadratic", action="store_true")
    p.add_argument(
        "--free-net-spin", action="store_true", help="Fit independent outgoing spin at net"
    )
    p.add_argument(
        "--first-ground-seed",
        action="store_true",
        help="Initialize free net velocity toward the first original ground witness; no response constraint",
    )
    args = p.parse_args()
    if args.first_ground_seed and not args.free_net_velocity:
        p.error("--first-ground-seed requires --free-net-velocity")
    if args.free_net_spin and not args.free_net_velocity:
        p.error("--free-net-spin requires --free-net-velocity")
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(
        (f.paths.data_root() / "processed/ownerfix/adopted_perflight3/report.json").read_text()
    )
    item = next(r for r in base["attempts"] if r["key"] == args.case)
    rung = "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm_win32_shift1f"
    original, source, _, _, search, t, d, loaded = f.load_case(dict(item=item, rung=rung))
    before, _ = census.score(
        original, source, t, d, item["metadata"]["ending_kind"], scorer=scope.LOCAL_SCORE
    )
    labels = json.loads(loaded["arguments"].labels.read_text())
    cameras_doc = json.loads(loaded["arguments"].cameras.read_text())
    context, receipt = census.extended_context(original, labels, cameras_doc, search)
    if context is None:
        raise ValueError(receipt)
    targets = deepcopy(context["targets"])
    cameras = {
        int(r["frame"]): np.asarray(r["P"])
        for r in cameras_doc["cameras"]
        if "P" in r and r.get("supported", True)
    }
    records = [
        r
        for group in labels["ball"]["records"]
        for r in group["frames"]
        if r["status"] == "visible"
    ]
    pixels = {int(r["frame"]): np.array([r["x1080"], r["y1080"]]) for r in records}
    radii = {int(r["frame"]): r.get("uncertainty_radius_px1080", 2.0) for r in records}
    nets = [e for e in context["events"] if e["event_type"] == "net_hit"]
    target_receipts = []
    for i, group in enumerate(targets):
        if not len(context["scene"].net_hit_frames[i]):
            continue
        available = np.unique(
            np.r_[context["scene"].observation_frames[i], context["heldout"].observation_frames[i]]
        )
        for j, target in enumerate(group):
            event = next(
                e
                for e in context["events"]
                if e["event_type"] == "bounce" and e["frame"] == target["event_frame"]
            )
            eligible, bound = bounded.net_bounded_frames(event, available, nets)
            new = whole.event_ground_target(
                event,
                cameras,
                pixels,
                radii,
                camera_distortion=whole.camera_geometry.camera_radial_map(cameras_doc),
                mode="subframe_graded_circle",
                observation_fallback=True,
                eligible_frames=eligible,
            )
            group[j] = new
            target_receipts.append(dict(flight=i, bounds=bound, before=target, after=new))
    checks, consumed = followup.fit_check_copy(context["scene"], context["heldout"])
    fit_context = context | {"heldout": checks}
    first_ground_seed = None
    if args.first_ground_seed:
        terminal_net = context["scene"].net_hit_frames[-1][0]
        first_ground_seed = min(
            (target for target in context["targets"][-1] if target["event_frame"] > terminal_net),
            key=lambda target: target["event_frame"],
        )
    sources = [*loaded["sources"]] + [
        provenance.file_record(path)
        for path in [
            __file__,
            epoch.__file__,
            epoch.chart.__file__,
            epoch.height_chart.__file__,
            epoch.tape.__file__,
            followup.__file__,
            bounded.__file__,
            horizon.__file__,
        ]
    ]
    if args.free_net_velocity:
        from cv.experiments.connected_shooting import labeled_net_free_response

        sources.append(provenance.file_record(labeled_net_free_response.__file__))
        shutil.copy2(
            labeled_net_free_response.__file__,
            args.output / Path(labeled_net_free_response.__file__).name,
        )
    if args.first_ground_seed:
        from cv.experiments.connected_shooting import labeled_net_ground_seed

        sources.append(provenance.file_record(labeled_net_ground_seed.__file__))
        shutil.copy2(
            labeled_net_ground_seed.__file__,
            args.output / Path(labeled_net_ground_seed.__file__).name,
        )
    if args.free_net_spin:
        from cv.experiments.connected_shooting import labeled_net_spin_response, net_collision

        for module in [labeled_net_spin_response, net_collision]:
            sources.append(provenance.file_record(module.__file__))
            shutil.copy2(module.__file__, args.output / Path(module.__file__).name)
    for path in [
        __file__,
        epoch.__file__,
        epoch.chart.__file__,
        epoch.height_chart.__file__,
        epoch.tape.__file__,
        followup.__file__,
        bounded.__file__,
        horizon.__file__,
    ]:
        shutil.copy2(path, args.output / Path(path).name)
    start = dict(
        key=args.case,
        sources=sources,
        before=before,
        context_receipt=receipt,
        targets=target_receipts,
        fit_only_duplicate_check_rows_removed=consumed,
        original_labels_changed=False,
        automatic_inference_eligible=False,
        human_derived=True,
        explicit_first_ground_seed=first_ground_seed,
    )
    (args.output / "start.json").write_text(json.dumps(f.profile.jsonable(start), indent=2) + "\n")
    original_evaluate = epoch.interval.block.event_constraints.evaluate
    last_event = next(
        e
        for e in context["events"]
        if e["event_type"] == "bounce" and e["frame"] == context["bounces"][-1][-1]
    )

    def guided_evaluate(scene, params, *positional, **keywords):
        native, residuals, evidence = original_evaluate(scene, params, *positional, **keywords)
        if args.continuous_ground:
            return horizon.apply(
                scene, params, native, residuals, evidence, last_event["frame_interval"]
            )
        lo, hi = last_event["frame_interval"]
        impacts = native[-1]["bounces"]
        if impacts:
            frame = float(impacts[0]["frame"])
            extra = (min(frame - lo, 0.0) + max(frame - hi, 0.0)) / 0.1
        else:
            frames = np.asarray(scene.observation_frames[-1])
            z = np.interp(hi, frames, native[-1]["positions"][:, 2])
            extra = max(z - f.model.R_BALL, 0.0) / 0.01
        if args.net_band:
            hits = native[-1]["net_hits"]
            if hits:
                hit = hits[0]
                z = hit["x"][2]
                tape = hit["tape_height_m"]
                net_extra = (
                    min(z - (tape - 0.10), 0.0) + max(z - (tape + f.model.R_BALL + 0.03), 0.0)
                ) / 0.03
            else:
                net_extra = 100.0
        else:
            net_extra = 0.0
        return native, np.r_[residuals, extra, net_extra], evidence

    try:
        if args.ground_guidance:
            epoch.interval.block.event_constraints.evaluate = guided_evaluate
        fit = epoch.fit(
            fit_context,
            source,
            d,
            maxiter=120,
            seconds=300,
            mesh_response=args.mesh,
            joint=args.joint,
            upper_net=args.upper_net,
            jacobian_step=1e-6 if args.continuous_ground else 1e-4,
            quadratic_images=args.quadratic,
            free_net_velocity=args.free_net_velocity,
            first_ground_seed=first_ground_seed,
            free_net_spin=args.free_net_spin,
        )
    finally:
        epoch.interval.block.event_constraints.evaluate = original_evaluate
    fit["policy"]["continuous_terminal_timing_forecast"] = args.continuous_ground
    fit["policy"]["astra_net_band_guidance"] = dict(
        enabled=args.net_band,
        lower_offset_from_tape_m=-0.10,
        upper_offset_from_tape_m=f.model.R_BALL + 0.03,
        scale_m=0.03,
        evidence="original upper-net/tape event notes plus new clean native sequence review; camera projects actual tape closely; not independent airborne truth",
    )
    fit["policy"]["extra_ground_interval_guidance"] = dict(
        enabled=args.ground_guidance,
        interval=last_event["frame_interval"],
        time_scale_frames=0.1,
        missing_height_scale_m=0.01,
        original_numeric_gates_unchanged=True,
    )
    params = fit["best"]["parameters"]
    response = fit["best"]["response"]
    if args.free_net_velocity:
        response_law = labeled_net_free_response.response_from_record(response)
    else:
        response_law = epoch.tape.TapeResponse(**response) if response else None
    with epoch.tape.using_response(response_law) if response_law is not None else nullcontext():
        old_targets, measurement = census.score(
            context, params, t, d, item["metadata"]["ending_kind"], scorer=scope.LOCAL_SCORE
        )
        after, measured = census.score(
            context | {"targets": targets},
            params,
            t,
            d,
            item["metadata"]["ending_kind"],
            scorer=scope.LOCAL_SCORE,
        )
        chain = f.model.chain(context["scene"], params)
    result = start | dict(
        status="measured",
        fit=fit,
        after=after,
        old_targets_verdict=old_targets,
        measurement=measurement,
        final_net_hits=chain[-1]["net_hits"],
        scene_contact_frames=context["scene"].contact_frames,
        terminal_rebound_frames=context["terminal_rebound_frames"],
        parameters=params,
    )
    (args.output / "report.json").write_text(
        json.dumps(f.profile.jsonable(result), indent=2) + "\n"
    )
    print(args.case, "mesh", args.mesh, after["accepted_flight_count"], after["gaps"], flush=True)


if __name__ == "__main__":
    main()
