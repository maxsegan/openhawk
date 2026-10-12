"""Evidence-qualified preparation for a cold labeled S6 candidate.

This optional candidate is separate from the running baseline. It uses ordinary
Python numerical code only. Known net/ground observations can extend a query
window and bound bounce witnesses; observed contact evidence can repair player
association. No case keys, fitted seeds, new labels or acceptance-ranked choices.

Example: python -m cv.pipeline.s6_preparation_policy --manifest FILE --case KEY
         --output NEW_DIRECTORY --policy on
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_contact_association as association
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as net
from cv.experiments.connected_shooting import labeled_preparation_recovery_probe as preparation
from cv.pipeline import provenance, s6_labeled_stage as stage
from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

POLICIES = ("off", "on")

#: Source net events one declared branch scope treats as absent. The bounded
#: helper below is shared by search, replay and export, so a branch that omits a
#: source net must narrow that helper rather than inherit a captured global net.
_ABSENT_NETS: ContextVar[tuple] = ContextVar("declared_absent_source_net_events", default=())


def _net_key(event: dict) -> tuple:
    return (event["event_type"], float(event["frame"]), tuple(map(float, event["frame_interval"])))


@contextmanager
def declared_absent_nets(events) -> Iterator[None]:
    """Narrow the shared bounded-net helper to one declared branch scope."""
    token = _ABSENT_NETS.set(tuple(_net_key(row) for row in events or ()))
    try:
        yield
    finally:
        _ABSENT_NETS.reset(token)


def attempt_labels(labels: dict, packet: dict) -> dict:
    """Limit physical event qualification to the one explicitly supplied clip."""
    if len(packet.get("attempts", [])) != 1:
        raise ValueError("one prepared attempt required")
    clip = packet["attempts"][0]["point_clip"]
    scoped = deepcopy(labels)
    scoped["events"]["records"] = [
        event for event in labels["events"]["records"] if event.get("clip", clip) == clip
    ]
    from cv.pipeline import s6_contact_prefix_runtime

    return s6_contact_prefix_runtime.labels_for_scope(scoped, packet)


def prepare_packet(packet: dict, labels: dict, policy: str) -> tuple[dict, dict]:
    if policy not in POLICIES:
        raise ValueError("preparation policy must be off or on")
    from cv.experiments.connected_shooting import observation_scope

    observation_scope.validate(packet["attempts"][0], labels)
    scoped = attempt_labels(labels, packet)
    from cv.pipeline import s6_contact_prefix_runtime

    from cv.pipeline import s6_component_scope

    if s6_component_scope.active(packet["attempts"][0]):
        return packet, {
            "policy": policy,
            "event_occurrence": {"status": "not_applicable_contact_component"},
            "resolved_net_events": [],
            "terminal_context": {"status": "not_applicable_contact_component"},
            "observations_or_epochs_manufactured": False,
        }
    if s6_contact_prefix_runtime.active(packet["attempts"][0]):
        return packet, {
            "policy": policy,
            "event_occurrence": {"status": "not_applicable_contact_prefix"},
            "resolved_net_events": [],
            "terminal_context": {"status": "not_applicable_contact_prefix"},
            "observations_or_epochs_manufactured": False,
        }
    occurrence_receipt = {"status": "policy_disabled"}
    if policy == "on":
        packet, occurrence_receipt = occurrence.reconcile_packet(packet, scoped)
        observation_scope.validate(packet["attempts"][0], labels)
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    attempt = packet["attempts"][0]
    barrier = exposure.next_physical_event(
        scoped, attempt["point_clip"], attempt["owner_end_frame"], event_types=("contact",)
    )
    boundary = (
        None if barrier is None else float(barrier.get("frame_interval", [barrier["frame"]])[0])
    )
    resolved_nets = [
        event
        for event in scoped["events"]["records"]
        if event["event_type"] == "net_hit" and occurrence.resolved_membership(event, default=None)
    ]
    nets = [
        event
        for event in resolved_nets
        if boundary is None
        or float(event.get("frame_interval", [event["frame"], event["frame"]])[-1]) < boundary
    ]
    receipt = {
        "policy": policy,
        "event_occurrence": occurrence_receipt,
        "resolved_net_events": nets if policy == "on" else [],
        "net_continuity_scope": {
            "next_original_contact": deepcopy(barrier),
            "excluded_events": [deepcopy(event) for event in resolved_nets if event not in nets],
            "rule": "whole net interval precedes the earliest possible next raw contact",
            "original_event_records_preserved": True,
        },
        "terminal_context": {"status": "not_applied"},
        "observations_or_epochs_manufactured": False,
    }
    if policy == "off" or not nets:
        receipt["terminal_context"]["reason"] = "policy disabled or no resolved net observation"
        return packet, receipt
    try:
        prepared, details = preparation.known_terminal_context(packet, scoped)
    except ValueError as error:
        receipt["terminal_context"] = {"status": "qualification_refused", "reason": str(error)}
        return packet, receipt
    receipt["terminal_context"] = {"status": "applied", **details}
    return prepared, receipt


@contextmanager
def bounded_net_context(net_events: list[dict]) -> Iterator[dict]:
    """Use the same original helper for fitting and replay; restore on every exit."""
    receipt = {"policy": net.POLICY if net_events else "original", "applications": []}
    if not net_events:
        yield receipt
        return
    original = whole.event_ground_target
    seen = set()

    def target(event, cameras, pixels, radii=None, **kwargs):
        available = kwargs.get("eligible_frames")
        if available is None:
            available = sorted(set(cameras).intersection(pixels))
        absent = set(_ABSENT_NETS.get())
        active = (
            net_events
            if not absent
            else [
                row
                for row in net_events
                if not {"event_type", "frame", "frame_interval"} <= row.keys()
                or _net_key(row) not in absent
            ]
        )
        if not active:
            return original(event, cameras, pixels, radii, **kwargs)
        kept, details = net.net_bounded_frames(event, available, active)
        key = json.dumps(details, sort_keys=True)
        if key not in seen:
            receipt["applications"].append(details)
            seen.add(key)
        kwargs["eligible_frames"] = np.asarray(kept, float)
        return original(event, cameras, pixels, radii, **kwargs)

    whole.event_ground_target = target
    try:
        yield receipt
    finally:
        whole.event_ground_target = original


def run_cold(row: dict, output: Path, numerical_policy: dict, policy: str = "on") -> dict:
    """Cold-run preparation/search and replay its input-ranked candidate exactly."""
    from cv.experiments.connected_shooting import labeled_common_source as loader
    from cv.experiments.connected_shooting import labeled_context_census as census
    from cv.experiments.connected_shooting import labeled_context_witness_scope as scope
    from cv.experiments.connected_shooting import labeled_isolated_serve as isolated

    if output.exists():
        raise ValueError("new candidate output required")
    inputs = stage.validate_row(row)
    labels, packet, cameras = (
        json.loads(inputs[name].read_text()) for name in ("labels", "packet", "cameras")
    )
    scoped = attempt_labels(labels, packet)
    prepared, preparation_receipt = prepare_packet(packet, scoped, policy)
    output.mkdir(parents=True)
    from cv.pipeline import s6_player_camera

    inputs, player_camera_receipt = s6_player_camera.prepare(
        inputs,
        row,
        packet,
        cameras,
        output,
        stage.shared_settings(numerical_policy)["player_camera_coordinates"],
    )
    preparation_receipt["player_camera_coordinates"] = player_camera_receipt
    if prepared is not packet:
        path = output / "prepared_packet.json"
        stage.save(path, prepared)
        inputs = {**inputs, "packet": path}
    command = stage.search_arguments(row, inputs, output, numerical_policy)
    nets = preparation_receipt["resolved_net_events"]
    if nets:
        command.extend(["--terminal-rebound", "on"])
    association_policy = "observed-contact" if policy == "on" else "original"
    invocation = {
        "key": row["key"],
        "policy": policy,
        "numerical_policy": numerical_policy,
        "original_inputs": row,
        "executed_inputs": {name: provenance.file_record(path) for name, path in inputs.items()},
        "preparation": preparation_receipt,
        "arguments": command,
        "cold_search": True,
        "external_fitted_parameters_read": False,
        "implementation_files": [
            provenance.file_record(Path(module.__file__))
            for module in (
                sys.modules[__name__],
                stage,
                whole,
                association,
                net,
                preparation,
                s6_player_camera,
            )
        ],
        "code": provenance.git_record(stage.paths.REPO_ROOT),
    }
    stage.save(output / "invocation.json", invocation)
    previous = sys.argv
    try:
        with bounded_net_context(nets) as search_net:
            with association.context(scoped, cameras, association_policy) as search_association:
                sys.argv = [whole.__file__, *command]
                whole.main()
    finally:
        sys.argv = previous
    search_path = output / "search" / "report.json"
    report = json.loads(search_path.read_text())
    if player_camera_receipt["enabled"]:
        report["s6_player_camera_coordinates"] = player_camera_receipt
    if report.get("initialization", {}).get("uses_truth_xyz_or_velocity") is not False:
        raise ValueError("cold initialization attestation missing")
    report["experimental_contact_association"] = search_association
    if nets:
        report["experimental_net_support_policy"] = net.POLICY
    report["s6_preparation_policy"] = {
        **preparation_receipt,
        "net_support": search_net,
        "association": search_association,
    }
    stage.save(search_path, report)
    args = SimpleNamespace(
        **inputs,
        search_report=search_path,
        output=output,
        pose_image_scale=row.get("pose_image_scale", 1.0),
        player_order=list(row.get("player_order") or []),
        athlete_evidence=stage.shared_settings(numerical_policy)["athlete_evidence"],
        athlete_root_reach_loss=stage.shared_settings(numerical_policy)["athlete_root_reach_loss"],
        observation_fallback="on",
        witness_surface=row["surface"],
        source_stage="input-ranked-refined",
        association_policy=association_policy,
    )
    with bounded_net_context(nets) as replay_net:
        # The source loader itself enters the matching association context.
        bundle = loader.load(args)
        context = bundle["context"]
        context_receipt = {"status": "source_context"}
        try:
            extended, context_receipt = census.extended_context(
                context, bundle["labels"], bundle["cameras"], report
            )
            if extended is not None:
                context = extended
        except ValueError as error:
            context_receipt = {"status": "source_context_fallback", "reason": str(error)}
        verdict, measurement = census.score(
            context,
            bundle["parameters"],
            bundle["threshold"],
            bundle["duration"],
            bundle["ending"],
            scorer=scope.LOCAL_SCORE,
        )
    result = {
        "schema": "labeled_s6_preparation_candidate_v1",
        "key": row["key"],
        "status": "measured_requires_native_review",
        "declared_flights": row["declared_flights"],
        "surface": row["surface"],
        "cold_search": True,
        "preparation": report["s6_preparation_policy"],
        "replay_preparation": {
            "net_support": replay_net,
            "association": bundle["association_preparation"],
        },
        "selected_by": {
            key: value
            for key, value in bundle["selection"].items()
            if key != "from_scratch_initialization"
        },
        "selected_search": provenance.file_record(search_path),
        "context": context_receipt,
        "parameters": bundle["parameters"],
        "measurement": measurement,
        "verdict": verdict,
        "replay": bundle["projection_replay"],
        "scorer_rung": isolated.RUNG,
        "correct_complete_yield": None,
        "incorrect_accept_count": None,
    }
    stage.save(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, default="on")
    parser.add_argument("--coarse-iterations", type=int, default=150)
    parser.add_argument("--refine-iterations", type=int, default=200)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != stage.SCHEMA:
        raise ValueError("frozen observation manifest required")
    row = next(row for row in manifest["rows"] if row["key"] == args.case)
    policy = {
        **stage.NUMERICAL_POLICY,
        "coarse_iterations": args.coarse_iterations,
        "refine_iterations": args.refine_iterations,
    }
    result = run_cold(row, args.output, policy, args.policy)
    print(result["status"], result["verdict"]["accepted_flight_count"], flush=True)


if __name__ == "__main__":
    main()
