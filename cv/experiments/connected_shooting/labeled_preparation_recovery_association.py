"""Recover serve-side association after an unsupported close-up camera.

Explicit labeled-input probe. Two or more later contacts with supported cameras
must independently imply the same serve side under alternating-hitter grammar.
The existing nearest-sided automatic player lookup then handles the missing
serve picture. All whole-point arguments follow ``--``; no label is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_preparation_recovery_inputs as inputs
from cv.pipeline import provenance


def implied_first_side(contacts: list[tuple[int, str]]) -> str:
    if len(contacts) < 2 or len({i for i, _ in contacts}) != len(contacts):
        raise ValueError("at least two distinct later contact associations required")
    sides = []
    for index, side in contacts:
        if index < 1 or side not in {"near", "far"}:
            raise ValueError("later contact index and sided automatic association required")
        sides.append(side if index % 2 == 0 else ("far" if side == "near" else "near"))
    if len(set(sides)) != 1:
        raise ValueError("later contact associations disagree on the serve side")
    return sides[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit", "score"), default="fit")
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    camera_path = Path(remaining[remaining.index("--cameras") + 1])
    cameras = {
        int(row["frame"]): row
        for row in json.loads(camera_path.read_text())["cameras"]
        if "P" in row and row.get("supported", True)
    }
    output = Path(remaining[remaining.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "producer": provenance.file_record(Path(__file__)),
        "camera_source": provenance.file_record(camera_path),
        "arguments": remaining,
        "human_derived": True,
        "automatic_inference_eligible": False,
        "applications": [],
        "status": "running",
    }
    path = output.parent / (output.name + "_association_recovery_manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    original_players = whole.alternating_player_states
    original_server = whole.single.server_state

    def players(pose_csv, clip, contact_events, labels, **kwargs):
        first_frame = int(np.ceil(contact_events[0]["frame"]))
        if first_frame in cameras:
            raise ValueError("this recovery requires an unsupported first-contact camera")
        anchors = []
        for index, event in enumerate(contact_events[1:], 1):
            frame = int(np.ceil(event["frame"]))
            if frame not in cameras:
                continue
            pixel = whole.contact_association_pixel(
                event, labels, observation_fallback=True, fallback_receipt=None
            )
            try:
                state = original_server(
                    pose_csv,
                    clip,
                    frame,
                    pixel,
                    image_coordinate_scale=kwargs.get("image_coordinate_scale", 1.0),
                )
            except ValueError:
                continue
            anchors.append({"contact_index": index, "frame": frame, "state": state})
        side = implied_first_side([(a["contact_index"], a["state"]["side"]) for a in anchors])

        def sided_server(*args, **options):
            if options.get("required_side") is None:
                options["required_side"] = side
            return original_server(*args, **options)

        try:
            whole.single.server_state = sided_server
            result = original_players(pose_csv, clip, contact_events, labels, **kwargs)
        finally:
            whole.single.server_state = original_server
        receipt = {
            "first_contact_frame": first_frame,
            "first_contact_camera_supported": False,
            "later_automatic_contact_associations": anchors,
            "implied_first_side": side,
            "result": result,
            "native_pictures_or_ball_labels_changed": False,
        }
        manifest["applications"].append(receipt)
        if kwargs.get("fallback_receipt") is not None:
            kwargs["fallback_receipt"].append(receipt)
        return result

    old = sys.argv
    try:
        whole.alternating_player_states = players
        if args.mode == "fit":
            sys.argv = [__file__, "--contact-player-feet", "--", *remaining]
            inputs.main()
        else:
            from cv.experiments.connected_shooting import per_flight_rescore

            sys.argv = [__file__, *remaining]
            per_flight_rescore.main()
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = old
        whole.alternating_player_states = original_players
        whole.single.server_state = original_server
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
