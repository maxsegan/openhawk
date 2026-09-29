"""No-fit census of supported terminal context on the reviewed labeled cohort.

Use --composition for the reviewed 89-attempt report and --output for artifacts.
The existing terminal-rebound contract is explicit, labels and fitted parameters
stay unchanged, and every original attempt remains in the denominator. This is
opened labeled development, not automatic inference or independent XYZ proof.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from copy import deepcopy
import json
import traceback
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import full_native_continuation as full
from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import terminal_completion
from cv.experiments.connected_shooting import interior_contact_epochs
from cv.pipeline import paths, provenance


# Cold initialization and later replay use the same original-event boundary rule.
before_next_physical_event = full.exposure.before_next_physical_event


# Earlier artifact producers import this name; new callers use the general name.
before_next_contact = before_next_physical_event


def extended_context(
    context, labels, cameras, search, *, last_context_frame=None, preserve_partition=False
):
    """Apply existing terminal semantics only when native post-bounce support exists."""
    if context.get("terminal_context_support") is not None:
        return None, {
            "status": "owned_native_context_preserved",
            "transaction": context["terminal_context_support"],
        }
    if context["attempt"].get("observation_scope") is not None:
        return None, {
            "status": "original_observation_horizon_preserved",
            "observation_scope": context["attempt"]["observation_scope"],
        }
    if not len(context["bounces"][-1]):
        return None, {"status": "no_supplied_terminal_bounce"}
    receipt = full.exposure.terminal_rebound_inventory(
        context["attempt"],
        labels["ball"]["records"],
        context["bounces"][-1][-1],
        labels=labels,
        duration_frames=search["configuration"]["exposure_duration_frames"],
    )
    if receipt["status"] != "supported":
        return None, receipt
    end = receipt["last_postbounce_labeled_frame"]
    if last_context_frame is not None:
        end = min(end, float(last_context_frame))
        receipt["postbounce_labeled_frames"] = [
            f for f in receipt["postbounce_labeled_frames"] if f <= end
        ]
        receipt["last_postbounce_labeled_frame"] = end
        receipt["postbounce_labeled_frame_count"] = len(receipt["postbounce_labeled_frames"])
    attempt = whole.event_recovery.extend_attempt_window(
        context["attempt"], labels["ball"]["records"], end
    )
    scene, heldout, bounces, native, _ = whole.prepare_attempt(
        attempt,
        cameras,
        search["configuration"]["surface"],
        context["events"],
        end,
        observation_partition=search["configuration"].get(
            "observation_partition", "fifth_frame_withheld"
        ),
        ground_settling=search["configuration"].get("ground_settling", "off") == "on",
        observation_fallback=True,
        fallback_receipt=[],
    )
    net_groups = tuple(
        np.asarray(
            [
                event["frame"]
                for event in context["events"]
                if event["event_type"] == "net_hit" and a < event["frame"] < b
            ],
            float,
        )
        for a, b in zip(scene.contact_frames[:-1], scene.contact_frames[1:], strict=True)
    )
    if any(len(group) for group in net_groups):
        scene = replace(scene, net_hit_frames=net_groups)
        heldout = replace(heldout, net_hit_frames=net_groups)
    axes, _ = whole.training_directions(
        scene,
        bounces,
        dict(annotation_status="frozen_agent_reference", records=labels["ball"]["records"]),
        observation_fallback=True,
        fallback_receipt=[],
    )
    if preserve_partition:
        from cv.pipeline import s6_terminal_context

        scene, axes, frames, rebuilt = s6_terminal_context.preserved_segment(
            scene, heldout, bounces, axes
        )
    else:
        scene, axes, frames, rebuilt = full.exposure.terminal_rebound_segment(
            scene, heldout, bounces, axes
        )
    if rebuilt["postbounce_labeled_frames"] != receipt["postbounce_labeled_frames"]:
        raise ValueError("rebuilt context does not reproduce native support inventory")
    if [len(x) for x in bounces] != [len(x) for x in context["bounces"]]:
        raise ValueError("extension changes supplied bounce membership")
    scene.validate()
    heldout.validate()
    extended = {
        **context,
        "attempt": attempt,
        "scene": scene,
        "heldout": heldout,
        "bounces": bounces,
        "native": native,
        "axes": axes,
        "terminal_rebound_frames": frames,
    }
    extended = interior_contact_epochs.preserve_in_extended_context(
        context, extended, search["configuration"]["exposure_duration_frames"]
    )
    return extended, {
        **receipt,
        **rebuilt,
        **(
            {"interior_timing_extension": extended["interior_timing_extension"]}
            if extended.get("interior_timing_extension") is not None
            else {}
        ),
    }


def score(context, parameters, threshold, duration, ending_kind, *, scorer=None):
    measured = full.exposure.measure(
        context["scene"],
        context["heldout"],
        context["bounces"],
        context["native"],
        parameters,
        context["axes"],
        duration,
        [],
        missing_check_direction="abstain",
        termination_kind=context["termination_kind"],
        **(
            {
                "terminal_rebound_frames": context["terminal_rebound_frames"],
                "terminal_ground_event": terminal_completion.original_ground_event(
                    context["scene"], context["bounces"], context["events"]
                ),
            }
            if context.get("terminal_rebound_frames") is not None
            else {}
        ),
    )
    from cv.experiments.connected_shooting import observation_partition as partition

    evaluation_scene, evaluation_check, evaluation_axes = partition.legacy_evaluation(
        context, exposure_duration=duration
    )
    gate_measurement = measured
    if partition.all_native(context["scene"]):
        gate_measurement = full.exposure.measure(
            evaluation_scene,
            evaluation_check,
            context["bounces"],
            context["native"],
            parameters,
            evaluation_axes,
            duration,
            [],
            missing_check_direction="abstain",
            termination_kind=context["termination_kind"],
            **(
                {
                    "terminal_rebound_frames": context["terminal_rebound_frames"],
                    "terminal_ground_event": terminal_completion.original_ground_event(
                        evaluation_scene, context["bounces"], context["events"]
                    ),
                }
                if context.get("terminal_rebound_frames") is not None
                else {}
            ),
        )
        gate_measurement = partition.mark_fitted_evaluation(
            gate_measurement, context["scene"], context["heldout"]
        )
        measured["fixed_frame_evaluation"] = {
            "native_projection": gate_measurement["native_projection"],
            "rms_px": gate_measurement["rms_px"],
            "observation_frames": [row.tolist() for row in evaluation_scene.observation_frames],
            "axes": evaluation_axes.tolist(),
            "policy": "legacy_fit_rows_and_axes_in_sample",
            **(
                {"unscorable_native_projection": gate_measurement["unscorable_native_projection"]}
                if gate_measurement.get("unscorable_native_projection")
                else {}
            ),
        }
    athlete = full.profile.athlete_priors.evaluate(
        np.asarray(measured["contact_xyz"]), context["players"]
    )
    raw = full.profile.acceptance.measure(
        evaluation_scene,
        evaluation_check,
        parameters,
        context["bounces"],
        context["native"],
        evaluation_axes,
        context["targets"],
        context["players"],
        context["events"],
        termination_kind=context["termination_kind"],
        duration=duration,
        athlete=athlete,
        ending_passive_context_frames=threshold.ending_passive_context_frames,
        measurement=gate_measurement,
        terminal_rebound_frames=context.get("terminal_rebound_frames"),
        **(
            {"right_contact_player": context.get("right_contact_player")}
            if context["scene"].right_boundary_kind == "original_contact"
            else {}
        ),
        **(
            {"preserve_observation_horizon": True}
            if context["attempt"].get("observation_scope") is not None
            or context.get("terminal_context_support") is not None
            else {}
        ),
    )
    raw["surface"] = context["witness_surface"]
    verdict = (scorer or full.profile.acceptance.score)(
        raw,
        threshold,
        depth_bounds=context["depth_bounds"],
        athlete_prior_mode="stature_pose_soft",
        ending_kind=ending_kind,
    )
    if context["scene"].terminal_net_tail is not None:
        from cv.experiments.connected_shooting.labeled_terminal_net_tail import mark_conditional

        verdict = mark_conditional(verdict, context["scene"].terminal_net_tail)
    if getattr(context["scene"], "observed_horizon_tail", None) is not None:
        # An unresolved tail can never become a complete point: the verdict says
        # so here, on the real driver, not only in the tail module's own tests.
        from cv.experiments.connected_shooting.observed_horizon_tail import mark_verdict

        verdict = mark_verdict(verdict, context["scene"].observed_horizon_tail)
    from cv.experiments.connected_shooting import observation_scope

    verdict = observation_scope.mark_verdict(
        verdict, context["attempt"].get("observation_scope"), context=context, measurement=measured
    )
    return verdict, measured


def common_path_check(context, extended, parameters):
    ends = np.minimum(context["scene"].contact_frames[1:], extended["scene"].contact_frames[1:])
    queries = tuple(
        np.unique(np.r_[a, np.arange(a, b, 0.25), b])
        for a, b in zip(context["scene"].contact_frames[:-1], ends, strict=True)
    )
    left = full.model.chain(context["scene"], parameters, query_frames=queries)
    right = full.model.chain(extended["scene"], parameters, query_frames=queries)
    drift = max(
        float(np.max(np.abs(np.asarray(a[field]) - b[field])))
        for a, b in zip(left, right, strict=True)
        for field in ["positions", "velocities", "start_xyz"]
    )
    event_equal = True
    for a, b in zip(left, right, strict=True):
        for field in ["bounces", "net_hits"]:
            event_equal &= full.profile.jsonable(a[field]) == full.profile.jsonable(b[field])
    return dict(
        common_dense_position_velocity_maximum_drift=drift,
        common_impacts_and_net_states_equal=event_equal,
    )


def run_case(job):
    key = job["item"]["key"]
    row = dict(key=key, status="pending")
    try:
        ctx, p, before, _, search, threshold, duration, loaded = full.load_case(job)
        assert before["accepted_flight_count"] == job["reviewed"]["after"]["accepted_flights"]
        labels = json.loads(loaded["arguments"].labels.read_text())
        cameras = json.loads(loaded["arguments"].cameras.read_text())
        ext, receipt = extended_context(ctx, labels, cameras, search)
        row.update(receipt=receipt, sources=loaded["sources"], before=before)
        if ext is None:
            row["status"] = "unsupported_native_context"
        else:
            after, measurement = score(
                ext, p, threshold, duration, job["item"]["metadata"]["ending_kind"]
            )
            row.update(
                status="measured",
                after=after,
                measurement=measurement,
                parameters=p,
                original_contact_frames=ctx["scene"].contact_frames,
                extended_contact_frames=ext["scene"].contact_frames,
                terminal_rebound_frames=ext["terminal_rebound_frames"],
                parity=common_path_check(ctx, ext, p),
                accepted_neighbors_preserved=set(before["accepted_flight_indices"])
                <= set(after["accepted_flight_indices"]),
                frozen_witnesses_unchanged=ctx["targets"] == ext["targets"],
                unchanged_parameters=True,
            )
            row["gain"] = after["accepted_flight_count"] > before["accepted_flight_count"]
            row["loss"] = after["accepted_flight_count"] < before["accepted_flight_count"]
    except Exception as exc:
        row.update(
            status="execution_failure",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    output = Path(job["output"]) / key
    output.mkdir(parents=True, exist_ok=True)
    (output / "context.json").write_text(json.dumps(full.profile.jsonable(row), indent=2) + "\n")
    return {
        k: v
        for k, v in full.profile.jsonable(row).items()
        if k not in ["measurement", "parameters"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--controls", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.composition.read_text())
    basepath = paths.data_root() / "processed/ownerfix/adopted_perflight3/report.json"
    base = json.loads(basepath.read_text())
    items = {x["key"]: x for x in base["attempts"]}
    jobs = []
    untested = []
    controls = 0
    for row in prior["attempts"]:
        before = row["before"]
        after = row["after"]
        changed = (
            row.get("changed_from_reviewed39")
            or row.get("changed_from_reviewed38")
            or row.get("changed")
        )
        # Existing derived selected vectors are already reviewed. Incomplete RG
        # already has this context, so do not revert it to its historical seed.
        if not items[row["key"]].get("fitted"):
            untested.append(dict(key=row["key"], status="no_fitted_source"))
        elif changed or after["accepted_flights"] != before["accepted_flights"]:
            untested.append(
                dict(key=row["key"], status="existing_reviewed_derived_vector_retained")
            )
        elif after["complete_point"] and controls >= args.controls:
            untested.append(dict(key=row["key"], status="complete_control_budget_not_selected"))
        else:
            controls += bool(after["complete_point"])
            jobs.append(
                dict(
                    item=items[row["key"]],
                    reviewed=row,
                    rung=prior["rung"],
                    output=str(args.output),
                )
            )
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run_case, jobs):
            results.append(row)
            print(
                row["key"],
                row["status"],
                row.get("before", {}).get("accepted_flight_count"),
                "->",
                row.get("after", {}).get("accepted_flight_count"),
                row.get("error", ""),
                flush=True,
            )
            (args.output / "report.json").write_text(
                json.dumps(dict(status="running", results=results, untested=untested), indent=2)
                + "\n"
            )
    composed = deepcopy(prior["attempts"])
    lookup = {x["key"]: x for x in results}
    for row in composed:
        result = lookup.get(row["key"], {})
        if (
            result.get("gain")
            and result["accepted_neighbors_preserved"]
            and result["parity"]["common_dense_position_velocity_maximum_drift"] < 1e-9
        ):
            verdict = result["after"]
            row["after"].update(
                accepted_flights=verdict["accepted_flight_count"],
                complete_point=verdict["complete_point"],
                partial_point=verdict["partial_point"],
                failure_counts=verdict["failure_counts"],
                gaps=verdict["gaps"],
            )
    result = dict(
        status="complete_native_gain_review_pending",
        automatic_inference_eligible=False,
        input_scope="Original human/Astra labels; prior withheld rebound rows consumed explicitly. Frozen vectors, physics and original witnesses; existing terminal-rebound semantics.",
        source=[
            provenance.file_record(p)
            for p in [
                args.composition,
                basepath,
                Path(__file__),
                Path(full.exposure.__file__),
                Path(full.rescore.__file__),
                Path(full.profile.acceptance.__file__),
            ]
        ],
        workers=args.workers,
        results=results,
        untested=untested,
        attempts=composed,
        summary=dict(
            attempts=len(composed),
            before_complete=41,
            before_flights=292,
            complete_points=sum(r["after"]["complete_point"] for r in composed),
            accepted_flights=sum(r["after"]["accepted_flights"] for r in composed),
            labeled_flights=sum(r["labeled_flights"] or 0 for r in composed),
        ),
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
