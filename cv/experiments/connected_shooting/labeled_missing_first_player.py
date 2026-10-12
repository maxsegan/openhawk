"""Recover a first hitter whose automatic pose row is absent at a pictured serve.

Explicit labeled-input research, with no inference/default changes.  The camera
and the labeled ball front at the first contact are both supported, but the sided
player table holds only the receiver at that picture.  The ordinary association
therefore adopts the sole visible player as server, mislabels every later side
under alternating grammar and pins the serve to the wrong end.

Two or more later bracket-local contacts, each with a supported camera, a
supported original fitting front and *both* sided rows present, must
independently imply the same first side.  The existing nearest-sided fallback
then supplies the missing first row with its declared widened sigma; it remains
an uncertain substitute with no wrist witness and no pre-contact feet track.
The original net/context adapter and observation set remain untouched.
Usage: ``fit|score --association old|recovered -- <original adapter arguments>``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_preparation_net_recovery as net
from cv.experiments.connected_shooting import player_position
from cv.experiments.connected_shooting import player_state_fallback
from cv.experiments.connected_shooting import toss_witness
from cv.experiments.connected_shooting.labeled_missing_launch_association import bracket_frames
from cv.experiments.connected_shooting.labeled_preparation_recovery_association import (
    implied_first_side,
)
from cv.pipeline import provenance

FEET_HISTORY_STATUS = "uncertain_contact_proxy_no_precontact_track"
FALLBACK_NAME = "first_player_from_later_alternating_consensus"


def opposite(side: str) -> str:
    return "far" if side == "near" else "near"


def camera_supported(cameras: dict, frame: int) -> bool:
    row = cameras.get(int(frame), {})
    return "P" in row and row.get("status") == "supported" and row.get("supported", True)


def both_sides_present(rows: dict, frame: int) -> bool:
    return (frame, "near") in rows and (frame, "far") in rows


def later_consensus(contacts, labels, cameras, rows, lookup) -> tuple[str, list[dict]]:
    """Implied first side from later contacts that were each pictured with both players.

    Only contacts with a supported contact camera, a supported bracket-local
    original fitting front and both sided detector rows at the exact contact
    picture qualify.  A contact with one sided row cannot witness a side choice.
    """
    anchors = []
    for index, event in enumerate(contacts[1:], 1):
        contact_frame = int(np.ceil(float(event["frame"])))
        choices = [f for f in bracket_frames(event, labels) if camera_supported(cameras, f)]
        if (
            not choices
            or not camera_supported(cameras, contact_frame)
            or not both_sides_present(rows, contact_frame)
        ):
            continue
        observed = min(choices, key=lambda f: (abs(f - float(event["frame"])), f))
        try:
            state = lookup(contact_frame, labels[observed])
        except ValueError:
            continue
        anchors.append(
            dict(
                contact_index=index,
                contact_frame=contact_frame,
                observation_frame=observed,
                side=state["side"],
                both_sides_present=True,
            )
        )
    side = implied_first_side([(a["contact_index"], a["side"]) for a in anchors])
    return side, anchors


def qualify(contacts, labels, cameras, rows, lookup) -> dict:
    """Decide whether the first row is missing exactly as later consensus implies.

    Raises ``ValueError`` unless the first contact camera and a bracket-local
    original fitting front are supported, at least two later contacts agree,
    the consensus side has no row at the first picture and the opposite side
    does.  Nothing here depends on a case identity.
    """
    first = contacts[0]
    epoch = float(first["frame"])
    first_frame = int(np.ceil(epoch))
    if not camera_supported(cameras, first_frame):
        raise ValueError("first-contact camera must be supported")
    fronts = [f for f in bracket_frames(first, labels) if camera_supported(cameras, f)]
    if not fronts:
        raise ValueError("requires a supported bracket-local original fitting front")
    side, anchors = later_consensus(contacts, labels, cameras, rows, lookup)
    if (first_frame, side) in rows:
        raise ValueError(
            f"{side}-side row present at the first contact picture; ordinary association applies"
        )
    if (first_frame, opposite(side)) not in rows:
        raise ValueError("requires the opposite receiver row at the first contact picture")
    return dict(
        first_contact_frame=epoch,
        first_pose_frame=first_frame,
        first_fitting_front_frame=min(fronts, key=lambda f: (abs(f - epoch), f)),
        first_contact_camera_supported=True,
        implied_first_side=side,
        sole_row_side_at_first_picture=opposite(side),
        later_associations=anchors,
        trajectory_observations_added=0,
        original_labels_changed=False,
        gate_thresholds_changed=False,
    )


def absent_position_state(
    frame: int,
    side: str,
    *,
    player_name: str | None,
    stature_m: float | None,
    image_coordinate_scale: float,
) -> dict:
    """The first-contact state whose court position is explicitly absent.

    The side is the later-contact consensus the qualifier already established and
    is retained; so are the supplied name and stature, which do not depend on a
    root.  ``court_centre_xy_m`` is ``None`` -- not a dummy root, not a NaN -- so
    every root-dependent consumer must abstain deliberately.  This remains the
    serve; no non-serve role is inferred from the missing row.
    """
    return player_position.absent_contact_state(
        frame,
        side,
        player_name=player_name,
        stature_m=stature_m,
        image_coordinate_scale=image_coordinate_scale,
        reason=(
            f"no {side}-side automatic row within "
            f"{player_state_fallback.DEFAULT_SEARCH_RADIUS_FRAMES} frames of the serve picture; "
            "side established independently by later-contact consensus"
        ),
        interpretation=(
            "independently sided first-contact player with no available court root; "
            "root-dependent priors, anchors and witnesses abstain, stature is retained"
        ),
    )


def recovered_player_states(
    pose_csv: Path,
    clip: str,
    contacts: list[dict],
    labels: dict,
    cameras: dict,
    *,
    original_players=None,
    original_server=None,
    missing_player_position: str = "off",
    **kwargs,
) -> tuple[list[dict], dict, dict]:
    """Run the original alternating association with only the first state substituted.

    The first state is the existing nearest-sided fallback of the consensus
    side.  ``single.server_state`` is intercepted for the first picture only and
    restored even when the original association raises.  Under the explicit
    ``abstain`` policy, and only when that ordinary same-side fallback yields
    nothing, the first state instead declares its court position absent.
    """
    if missing_player_position not in whole.MISSING_PLAYER_POSITION_POLICIES:
        raise ValueError("supported missing-player-position policy required")
    original_players = original_players or whole.alternating_player_states
    original_server = original_server or whole.single.server_state
    scale = kwargs.get("image_coordinate_scale", 1.0)
    rows = player_state_fallback.sided_court_rows(pose_csv, clip)

    def lookup(frame, pixel):
        return original_server(pose_csv, clip, frame, pixel, image_coordinate_scale=scale)

    receipt = qualify(contacts, labels, cameras, rows, lookup)
    first_frame = receipt["first_pose_frame"]
    side = receipt["implied_first_side"]
    names = kwargs.get("player_names")
    statures = kwargs.get("player_statures_m")
    name = None if names is None else names[0]
    stature = None if name is None else statures[name]
    first_state = player_state_fallback.contact_state(
        pose_csv,
        clip,
        first_frame,
        side,
        player_name=name,
        stature_m=stature,
        image_coordinate_scale=scale,
    )
    if first_state is None:
        # The qualifier already established this side from later independently
        # associated contacts, and the ball/camera/event evidence at the serve is
        # untouched.  Only the court root is unavailable, so under the explicit
        # shared policy it is declared absent rather than widened or invented.
        if missing_player_position != "abstain":
            raise ValueError(
                f"no {side}-side automatic row within the fallback radius of the serve"
            )
        first_state = absent_position_state(
            first_frame, side, player_name=name, stature_m=stature, image_coordinate_scale=scale
        )
        receipt["first_position"] = first_state["player_position_evidence"]
        receipt["fallback_radius_frames"] = player_state_fallback.DEFAULT_SEARCH_RADIUS_FRAMES

    def intercepted(pose_csv_, clip_, frame, pixel, **options):
        if int(frame) == first_frame and options.get("required_side") is None:
            return copy.deepcopy(first_state)
        return original_server(pose_csv_, clip_, frame, pixel, **options)

    try:
        whole.single.server_state = intercepted
        result = original_players(
            pose_csv,
            clip,
            contacts,
            labels,
            missing_player_position=missing_player_position,
            **kwargs,
        )
    finally:
        whole.single.server_state = original_server
    receipt.update(
        substituted_first_state=copy.deepcopy(first_state),
        result_sides=[r["side"] for r in result],
        pose_source=provenance.file_record(pose_csv),
    )
    return result, receipt, first_state


def recovered_server_feet(first_state: dict) -> dict:
    """Serve-feet proxy from the substituted first state, with its provenance disclosed.

    The original feet witness would associate the sole pictured receiver.  This
    returns the ordinary contact-state fallback shape but refuses to claim any
    observed feet at the contact picture: no history, no pre-contact track, no
    synthetic release position and no hand-height prior.
    """
    substitution = first_state.get("substitution")
    if substitution is None:
        # Absent position: there is no substitute row, so the feet witness has no
        # court position to report either.  Everything the witness could only get
        # from a root abstains; nothing is synthesized in its place.
        evidence = first_state["player_position_evidence"]
        feet = whole.contact_player_feet_fallback(
            first_state,
            ValueError(
                f"{first_state['side']}-side server row absent at contact picture "
                f"{first_state['frame']} and nowhere inside the fallback radius"
            ),
        )
        feet.update(
            history_frames=[],
            history_count=0,
            history_status=FEET_HISTORY_STATUS,
            history_span_frames=0,
            association_frame=None,
            observed_at_contact_picture=False,
            court_position=evidence,
            court_position_source_frame=None,
            court_position_frame_distance=None,
            court_position_sigma_m=None,
            precontact_track_grounding=False,
            release_position_synthesized=False,
            hand_height_prior_applied=False,
            interpretation=(
                "independently sided server with no available court root; the feet witness "
                "abstains entirely rather than naming a position it never observed"
            ),
        )
        return feet
    error = ValueError(
        f"{first_state['side']}-side server row absent at contact picture "
        f"{first_state['frame']}; consensus-side substitute from frame "
        f"{substitution['source_frame']}"
    )
    feet = whole.contact_player_feet_fallback(first_state, error)
    feet.update(
        history_frames=[],
        history_count=0,
        history_status=FEET_HISTORY_STATUS,
        history_span_frames=0,
        association_frame=None,
        observed_at_contact_picture=False,
        court_position_source_frame=substitution["source_frame"],
        court_position_frame_distance=substitution["frame_distance"],
        court_position_sigma_m=first_state["court_position_sigma_m"],
        precontact_track_grounding=False,
        release_position_synthesized=False,
        hand_height_prior_applied=False,
        interpretation=(
            "consensus-side nearest automatic row substituted for the absent server row; "
            "an uncertain contact proxy with no observed feet at the contact picture"
        ),
    )
    return feet


def module_records() -> list[dict]:
    return [
        provenance.file_record(Path(module.__file__), role=role)
        for module, role in (
            (whole, "whole_point_search"),
            (whole.single, "single_flight_search"),
            (player_state_fallback, "player_state_fallback"),
            (toss_witness, "toss_witness"),
            (net, "net_recovery_adapter"),
        )
    ]


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
    cameras = {int(r["frame"]): r for r in json.loads(camera_path.read_text())["cameras"]}
    manifest = dict(
        status="running",
        association=args.association,
        mode=args.mode,
        fallback=FALLBACK_NAME,
        sources=[provenance.file_record(p) for p in [source, camera_path, Path(__file__)]],
        module_dependencies=module_records(),
        arguments=remaining,
        applications=[],
        new_pixels_or_labels_added=False,
        gate_thresholds_changed=False,
        automatic_inference_eligible=False,
    )
    manifest_path = output.parent / (output.name + "_missing_first_player_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    original_players = whole.alternating_player_states
    original_server = whole.single.server_state
    original_feet = toss_witness.server_feet
    recovered: dict = {}

    def players(pose_csv, clip, contacts, labels, **kwargs):
        result, receipt, first_state = recovered_player_states(
            pose_csv,
            clip,
            contacts,
            labels,
            cameras,
            original_players=original_players,
            original_server=original_server,
            **kwargs,
        )
        recovered["first_state"] = first_state
        manifest["applications"].append(dict(application="player_states", **receipt))
        if kwargs.get("fallback_receipt") is not None:
            kwargs["fallback_receipt"].append(dict(fallback=FALLBACK_NAME, **receipt))
        return result

    def feet(pose_csv, clip, contact_frame, contact_pixel, **kwargs):
        first_state = recovered.get("first_state")
        if first_state is None or int(np.ceil(float(contact_frame))) != first_state["frame"]:
            return original_feet(pose_csv, clip, contact_frame, contact_pixel, **kwargs)
        proxy = recovered_server_feet(first_state)
        manifest["applications"].append(dict(application="server_feet", **proxy))
        if kwargs.get("fallback_receipt") is not None:
            kwargs["fallback_receipt"].append(
                dict(fallback=f"{FALLBACK_NAME}_server_feet", **proxy)
            )
        return proxy

    old_argv = sys.argv
    try:
        if args.association == "recovered":
            whole.alternating_player_states = players
            toss_witness.server_feet = feet
        sys.argv = [__file__, args.mode, "--", *remaining]
        net.main()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = old_argv
        whole.alternating_player_states = original_players
        whole.single.server_state = original_server
        toss_witness.server_feet = original_feet
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
