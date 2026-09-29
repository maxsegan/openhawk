"""Explicitly admit fitted native context to the net-bounded bounce witness.

The existing terminal-rebound path fits post-ground native rows but freezes an
earlier witness lookup. This development control also exposes those same rows
to the witness, only when already eligible in its actual fitted scene and camera.
Run ``fit -- <whole args>`` or ``score -- <per-flight args>``. No labels, pictures,
epochs or gate thresholds change. Earlier wrapper/report receipts remain intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as bounded
from cv.pipeline import provenance

POLICY = "already_fitted_native_context_in_net_bounded_bounce_witness_v1"


def augment_visible(pixels, radii, records, eligible, cameras):
    result, sigma = dict(pixels), dict(radii or {})
    added = []
    allowed = set(map(int, eligible)).intersection(cameras)
    for row in records:
        frame = int(row["frame"])
        if frame not in allowed or row["status"] != "visible":
            continue
        xy = np.array([row["x1080"], row["y1080"]], float)
        if frame in result:
            if not np.array_equal(result[frame], xy):
                raise ValueError("existing witness pixel differs from immutable label row")
            continue
        result[frame] = xy
        if row.get("uncertainty_radius_px1080") is not None:
            sigma[frame] = float(row["uncertainty_radius_px1080"])
        added.append(frame)
    return result, sigma, sorted(added)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    label_path = Path(remaining[remaining.index("--labels") + 1])
    labels = json.loads(label_path.read_text())
    windows = labels["ball"]["records"]
    if len(windows) != 1:
        raise ValueError("one explicit frozen native window required")
    records = windows[0]["frames"]
    output = Path(remaining[remaining.index("--output") + 1])
    if args.mode == "score":
        source = json.loads(Path(remaining[remaining.index("--search-report") + 1]).read_text())
        if source.get("experimental_net_witness_context_policy") != POLICY:
            raise ValueError("source does not bind explicit witness-context augmentation")
    manifest = {
        "policy": POLICY,
        "producer": provenance.file_record(Path(__file__)),
        "labels": provenance.file_record(label_path),
        "arguments": remaining,
        "mode": args.mode,
        "status": "running",
        "applications": [],
        "new_pixel_measurements_or_native_times": False,
        "witness_is_consumed_fitting_evidence_not_independent_3d_truth": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / (output.name + "_witness_context_manifest.json")
    original = whole.event_ground_target
    seen = set()

    def target(event, cameras, pixels, radii=None, **kwargs):
        eligible = kwargs.get("eligible_frames")
        if eligible is None:
            raise ValueError("explicit fitted native eligibility required for context witness")
        augmented, sigmas, added = augment_visible(pixels, radii, records, eligible, cameras)
        receipt = {
            "event_frame": event["frame"],
            "added_native_frames": added,
            "eligible_frames": np.asarray(eligible).tolist(),
        }
        key = json.dumps(receipt, sort_keys=True)
        if key not in seen:
            manifest["applications"].append(receipt)
            seen.add(key)
        return original(event, cameras, augmented, sigmas, **kwargs)

    old = sys.argv
    try:
        whole.event_ground_target = target
        sys.argv = [__file__, args.mode, "--", *remaining]
        bounded.main()
        if args.mode == "fit":
            report_path = output / "report.json"
            report = json.loads(report_path.read_text())
            report["experimental_net_witness_context_policy"] = POLICY
            report["experimental_net_witness_context_producer"] = manifest["producer"]
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        whole.event_ground_target = original
        sys.argv = old
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
