"""Recover explicit labeled-input preparation without changing labels or defaults.

``prepare -- <agent_attempt_prepare arguments>`` permits six or seven genuinely
visible ground controls through the existing physical camera fitter.
``fit --known-terminal-context -- <agent_whole_point_search arguments>`` extends
only the modeled query window to an already labeled post-net ground impact.
Original competitive ending, native rows, event intervals and provenance remain.
This is opened human/Astra-input research, never automatic inference.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_attempt_prepare as prepare
from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.pipeline import provenance

ORIGINAL_ANCHOR_FIT = prepare.anchor_fit


def partial_ground_anchor(record: dict, width_m: float) -> dict:
    """Use observed controls only; retain the existing six-landmark minimum."""
    labels = {r["target_id"]: r for r in record["frames"]}
    visible = [k for k in prepare.court.GROUND if labels[k]["status"] == "visible"]
    if len(visible) == len(prepare.court.GROUND):
        return ORIGINAL_ANCHOR_FIT(record, width_m)
    if len(visible) < 6:
        raise ValueError("at least six observed ground controls required")
    xyz = np.asarray(list(prepare.court.GROUND.values()), float)
    xyz[4:6, 1] += width_m / 2
    xyz[6:8, 1] -= width_m / 2
    indices = [list(prepare.court.GROUND).index(k) for k in visible]
    pixels = np.asarray([[labels[k]["x1080"], labels[k]["y1080"]] for k in visible])
    fitted = prepare.ground.fit_ground(xyz[indices], pixels)
    net = [k for k in prepare.court.NET if labels.get(k, {}).get("status") == "visible"]
    errors = {}
    if net:
        projected = prepare.ground.project(
            np.asarray(fitted["P"]), np.asarray([prepare.court.NET[k] for k in net])
        )
        errors = {
            k: float(np.linalg.norm(p - [labels[k]["x1080"], labels[k]["y1080"]]))
            for k, p in zip(net, projected, strict=True)
        }
    return dict(
        case_id=record["case_id"],
        frame=int(record["frames"][0]["frame"]),
        service_line_width_m=width_m,
        fit=fitted,
        visible_ground_controls=visible,
        omitted_ground_controls=[k for k in prepare.court.GROUND if k not in visible],
        visible_net_errors_px=errors,
        hidden_net_supports=[k for k in prepare.court.NET if k not in net],
        airborne_metric_accuracy_certified=False,
    )


def known_terminal_context(packet: dict, labels: dict) -> tuple[dict, dict]:
    """Carry existing post-net ground evidence, preserving the competitive stop."""
    result = deepcopy(packet)
    attempt = result["attempts"][0]
    end = float(attempt["owner_end_frame"])
    events = labels["events"]["records"]
    net = [
        e
        for e in events
        if e["event_type"] == "net_hit" and occurrence.resolved_membership(e, default=None)
    ]
    if not any(abs(float(e["frame"]) - end) <= 1e-8 for e in net):
        raise ValueError("competitive ending must be an already labeled net event")
    later = [
        e
        for e in events
        if e["event_type"] == "bounce"
        and occurrence.resolved_membership(e, default=None)
        and e["frame"] > end
    ]
    if not later:
        raise ValueError("no observed post-ending ground impact; do not invent one")
    target = min(later, key=lambda e: e["frame"])
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    barrier = exposure.next_physical_event(
        labels, attempt["point_clip"], end, event_types=("contact",)
    )
    if barrier is not None and float(barrier.get("frame_interval", [barrier["frame"]])[0]) <= float(
        target.get("frame_interval", [target["frame"], target["frame"]])[-1]
    ):
        raise ValueError("post-ending contact interval prevents passive continuation")
    if not any(
        e["event_type"] == "bounce" and e["frame"] == target["frame"] for e in attempt["events"]
    ):
        raise ValueError("source packet lacks the already labeled context impact")
    extended = whole.event_recovery.extend_attempt_window(
        attempt, labels["ball"]["records"], float(target["frame"])
    )
    receipt = dict(
        competitive_ending_frame=end,
        modeled_ground_context_end_frame=float(target["frame"]),
        original_ending_kind=labels["attempt"].get("ending_kind"),
        context_impact=target,
        physical_events_changed=False,
        native_times_changed=False,
        original_rows_unchanged=extended["owner_ball_labels"][: len(attempt["owner_ball_labels"])]
        == attempt["owner_ball_labels"],
        additional_frames=extended["modeled_window_extension"]["native_frames_added"],
        scope="query horizon only; original competitive ending is not relabeled",
    )
    extended["preparation_recovery"] = receipt
    result["attempts"][0] = extended
    return result, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit"])
    parser.add_argument("--known-terminal-context", action="store_true")
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    output = Path(remaining[remaining.index("--output") + 1])
    labels_path = Path(remaining[remaining.index("--labels") + 1])
    manifest = dict(
        human_derived=True,
        automatic_inference_eligible=False,
        producer=provenance.file_record(Path(__file__)),
        labels=provenance.file_record(labels_path),
        original_arguments=remaining.copy(),
        mode=args.mode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        manifest["camera_policy"] = "existing physical fit with six or more visible ground controls"
        prepare.anchor_fit = partial_ground_anchor
    elif args.known_terminal_context:
        index = remaining.index("--packet") + 1
        source = Path(remaining[index])
        packet, receipt = known_terminal_context(
            json.loads(source.read_text()), json.loads(labels_path.read_text())
        )
        destination = output.parent / "recovery_packet.json"
        destination.write_text(json.dumps(packet, indent=2) + "\n")
        manifest.update(
            original_packet=provenance.file_record(source),
            packet=provenance.file_record(destination),
            adaptation=receipt,
        )
        remaining[index] = str(destination)
    (output.parent / (output.name + "_recovery_manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    old = sys.argv
    try:
        sys.argv = [__file__, *remaining]
        (prepare.main if args.mode == "prepare" else whole.main)()
    finally:
        sys.argv = old
        prepare.anchor_fit = ORIGINAL_ANCHOR_FIT


if __name__ == "__main__":
    main()
