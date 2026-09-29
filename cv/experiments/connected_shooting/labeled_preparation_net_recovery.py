"""Use existing post-net context without fitting a bounce wing across net impact.

Opt-in labeled-input research. ``fit -- <whole-point args>`` first applies the
existing known-terminal-context adapter, then the existing --terminal-rebound on
path. ``score -- <per-flight args>`` reproduces the same net-bounded witnesses.
Labels, native timestamps, camera rows, physical laws and gate thresholds remain.
Both the added context and witness support change are explicit experimental inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_preparation_recovery_probe as preparation
from cv.experiments.connected_shooting import per_flight_rescore
from cv.pipeline import provenance

POLICY = "native_bounce_wings_between_labeled_net_brackets_v1"


def net_bounded_frames(event, available, net_events):
    """Exclude impact-uncertain frames and every different net-motion segment."""
    epoch = float(event["frame"])
    low, high = -np.inf, np.inf
    for net in net_events:
        a, b = map(float, net["frame_interval"])
        if a > b or a <= epoch <= b:
            raise ValueError("distinct ordered net and ground impact brackets required")
        if b < epoch:
            low = max(low, b)
        if a > epoch:
            high = min(high, a)
    original = sorted(set(map(int, available)))
    kept = [frame for frame in original if low < frame < high]
    return kept, {
        "ground_event_frame": epoch,
        "previous_net_upper_frame": None if not np.isfinite(low) else low,
        "next_net_lower_frame": None if not np.isfinite(high) else high,
        "original_eligible_frames": original,
        "bounded_eligible_frames": kept,
        "excluded_across_net_or_uncertain_frames": [f for f in original if f not in kept],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    labels_path = Path(remaining[remaining.index("--labels") + 1])
    labels = json.loads(labels_path.read_text())
    net_events = [
        e
        for e in labels["events"]["records"]
        if e["event_type"] == "net_hit" and e["status"] == "labeled"
    ]
    if not net_events:
        raise ValueError("explicit labeled net transition required")
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = None
    if args.mode == "fit":
        if "--terminal-rebound" in remaining:
            if remaining[remaining.index("--terminal-rebound") + 1] != "on":
                raise ValueError("this probe explicitly consumes existing terminal rebound context")
        else:
            remaining.extend(["--terminal-rebound", "on"])
    else:
        search_path = Path(remaining[remaining.index("--search-report") + 1])
        search = json.loads(search_path.read_text())
        if search.get("experimental_net_support_policy") != POLICY:
            raise ValueError("source report must explicitly bind this experimental witness policy")
        source_manifest = provenance.file_record(search_path)
    manifest = {
        "policy": POLICY,
        "mode": args.mode,
        "producer": provenance.file_record(Path(__file__)),
        "labels": provenance.file_record(labels_path),
        "arguments": remaining,
        "net_events": net_events,
        "source_report": source_manifest,
        "observations_or_epochs_manufactured": False,
        "thresholds_changed": False,
        "status": "running",
        "applications": [],
    }
    manifest_path = output.parent / (output.name + "_net_support_manifest.json")
    original = whole.event_ground_target
    seen = set()

    def target(event, cameras, pixels, radii=None, **kwargs):
        available = kwargs.get("eligible_frames")
        if available is None:
            available = sorted(set(cameras).intersection(pixels))
        kept, receipt = net_bounded_frames(event, available, net_events)
        key = json.dumps(receipt, sort_keys=True)
        if key not in seen:
            manifest["applications"].append(receipt)
            seen.add(key)
        kwargs["eligible_frames"] = np.asarray(kept, float)
        return original(event, cameras, pixels, radii, **kwargs)

    old_argv = sys.argv
    try:
        whole.event_ground_target = target
        sys.argv = [
            __file__,
            *(["fit", "--known-terminal-context", "--"] if args.mode == "fit" else []),
            *remaining,
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (preparation.main if args.mode == "fit" else per_flight_rescore.main)()
        if args.mode == "fit":
            path = output / "report.json"
            report = json.loads(path.read_text())
            report["experimental_net_support_policy"] = POLICY
            report["experimental_net_support_producer"] = manifest["producer"]
            path.write_text(json.dumps(report, indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        whole.event_ground_target = original
        sys.argv = old_argv
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
