"""Recover an association-only original precontact front after an unpictured launch.

Explicit labeled-input research, with no inference/default changes. A supported
original bracket front may replace a distant post-bounce association pixel only
when at least two later bracket-local contacts independently agree on serve side.
The original net/context adapter and original trajectory observation set remain.
Usage: ``fit|score --association old|recovered -- <original adapter arguments>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as net
from cv.experiments.connected_shooting.labeled_preparation_recovery_association import (
    implied_first_side,
)
from cv.pipeline import provenance


def bracket_frames(event: dict, fronts: dict) -> list[int]:
    low, high = event["frame_interval"]
    return sorted(f for f in fronts if np.floor(low) - 1 <= f <= np.ceil(high) + 1)


def qualify(contacts, fitted_fronts, original_fronts, cameras, ground_events, lookup):
    """Return an observed association pixel only; never manufacture a fit front."""
    first = contacts[0]
    epoch = float(first["frame"])
    if not fitted_fronts or bracket_frames(first, fitted_fronts):
        raise ValueError("requires absent bracket-local fitting fronts")
    nearest = min(fitted_fronts, key=lambda f: (abs(f - epoch), f))
    if nearest - epoch <= 8 or not any(epoch < e["frame"] < nearest for e in ground_events):
        raise ValueError("requires a distant first fitting front after a supplied bounce")

    def supported(frame):
        row = cameras.get(int(frame), {})
        return "P" in row and row.get("supported", True)

    original = bracket_frames(first, original_fronts)
    original = [f for f in original if supported(f)]
    if not original or not supported(np.ceil(epoch)):
        raise ValueError("original bracket front and first-contact cameras must be supported")
    frame = min(original, key=lambda f: (abs(f - epoch), f))
    pixel = np.asarray(original_fronts[frame], float).copy()
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("finite original observed front required")
    first_state = lookup(int(np.ceil(epoch)), pixel)
    anchors = []
    for index, event in enumerate(contacts[1:], 1):
        contact_frame = int(np.ceil(event["frame"]))
        choices = bracket_frames(event, fitted_fronts)
        choices = [f for f in choices if supported(f)]
        if not choices or not supported(contact_frame):
            continue
        observed = min(choices, key=lambda f: (abs(f - event["frame"]), f))
        try:
            state = lookup(contact_frame, fitted_fronts[observed])
        except ValueError:
            continue
        anchors.append(
            dict(
                contact_index=index,
                contact_frame=contact_frame,
                observation_frame=observed,
                state=state,
            )
        )
    side = implied_first_side([(a["contact_index"], a["state"]["side"]) for a in anchors])
    if side != first_state["side"]:
        raise ValueError("original precontact front disagrees with later contact grammar")
    return pixel, dict(
        first_contact_frame=epoch,
        original_front_frame=frame,
        original_front_xy=pixel.tolist(),
        original_front_state=first_state,
        first_fitting_front_frame=nearest,
        later_associations=anchors,
        implied_first_side=side,
        first_contact_camera_supported=True,
        trajectory_observations_added=0,
        original_labels_changed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("--association", choices=("old", "recovered"), required=True)
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    source = Path(remaining[remaining.index("--labels") + 1])
    camera_path = Path(remaining[remaining.index("--cameras") + 1])
    output = Path(remaining[remaining.index("--output") + 1])
    doc = json.loads(source.read_text())
    cameras = {int(r["frame"]): r for r in json.loads(camera_path.read_text())["cameras"]}
    ground_events = [
        e
        for e in doc["events"]["records"]
        if e["event_type"] == "bounce" and e["status"] == "labeled"
    ]
    manifest = dict(
        status="running",
        association=args.association,
        mode=args.mode,
        sources=[provenance.file_record(p) for p in [source, camera_path, Path(__file__)]],
        arguments=remaining,
        applications=[],
        automatic_inference_eligible=False,
    )
    manifest_path = output.parent / (output.name + "_launch_association_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    original_players = whole.alternating_player_states
    original_pixel = whole.contact_association_pixel
    recovered = {}

    def association(event, labels, **kwargs):
        epoch = float(event["frame"])
        if epoch in recovered:
            return recovered[epoch].copy()
        return original_pixel(event, labels, **kwargs)

    def players(pose_csv, clip, contacts, labels, **kwargs):
        fronts = {
            int(x["frame"]): np.array([x["x1080"], x["y1080"]], float)
            for record in doc["ball"]["records"]
            if record["clip"] == clip
            for x in record["frames"]
            if x["status"] == "visible"
        }

        def lookup(frame, pixel):
            return whole.single.server_state(
                pose_csv,
                clip,
                frame,
                pixel,
                image_coordinate_scale=kwargs.get("image_coordinate_scale", 1.0),
            )

        pixel, receipt = qualify(contacts, labels, fronts, cameras, ground_events, lookup)
        recovered[float(contacts[0]["frame"])] = pixel
        result = original_players(pose_csv, clip, contacts, labels, **kwargs)
        receipt["result_sides"] = [r["side"] for r in result]
        manifest["applications"].append(receipt)
        if kwargs.get("fallback_receipt") is not None:
            kwargs["fallback_receipt"].append(
                dict(fallback="original_precontact_association_only", **receipt)
            )
        return result

    old_argv = sys.argv
    try:
        if args.association == "recovered":
            whole.alternating_player_states = players
            whole.contact_association_pixel = association
        sys.argv = [__file__, args.mode, "--", *remaining]
        net.main()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = old_argv
        whole.alternating_player_states = original_players
        whole.contact_association_pixel = original_pixel
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
