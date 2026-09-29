"""Scope evidence-qualified first-hitter association around one search or rescore.

Explicit labeled-input research, with no inference/default changes.  The
ordinary cached common source misassigns the first hitter in two already
studied shapes: every in-attempt launch front abstains until after the serve
bounce while an original bracket front exists (see
``labeled_missing_launch_association``), and the server's automatic pose row is
absent at a pictured serve while the receiver's row is present (see
``labeled_missing_first_player``).  Both modules already own the qualification
rules and the substitutes; this helper only decides which one, if either,
applies to the contacts a caller actually associates, and scopes the patches.
A third recovery handles an unsupported first-contact camera: explicit original
hitter-end metadata must agree with at least two supported later contacts.

``context(labels_doc, cameras_doc)`` patches ``whole.alternating_player_states``,
``whole.contact_association_pixel`` and ``toss_witness.server_feet`` for the
duration of the block, so it wraps either the whole-point search or
``per_flight_rescore.build_context`` unchanged.  When neither recovery
qualifies the original functions run with the original arguments and the
caller's fallback receipt is untouched.  Qualification refusals are ordinary
fallbacks recorded with their reason for the original two recoveries. In the
unsupported-camera/explicit-end scope, ambiguous or contradictory evidence
raises a preparation refusal instead of restoring the distant-front guess.
Execution errors inside a qualified recovery propagate. Single-process,
single-attempt scoped use only.
"""

from __future__ import annotations

from cv.experiments.connected_shooting import labeled_event_occurrence as occurrence

from contextlib import contextmanager
import copy
from typing import Iterator

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_missing_first_player as first_player
from cv.experiments.connected_shooting import labeled_missing_launch_association as launch
from cv.experiments.connected_shooting import labeled_unsupported_contact_camera as initial_camera
from cv.experiments.connected_shooting import player_state_fallback
from cv.experiments.connected_shooting import toss_witness

POLICIES = ("original", "observed-contact")
MISSING_LAUNCH = "original_precontact_association_only"
MISSING_FIRST_PLAYER = first_player.FALLBACK_NAME


def original_fronts(labels_doc: dict, clip: str) -> dict[int, np.ndarray]:
    """Original visible fronts of one clip, read from the labels document."""
    return {
        int(x["frame"]): np.array([x["x1080"], x["y1080"]], float)
        for record in labels_doc["ball"]["records"]
        if record["clip"] == clip
        for x in record["frames"]
        if x["status"] == "visible"
    }


def labeled_grounds(labels_doc: dict) -> list[dict]:
    return [
        e
        for e in labels_doc["events"]["records"]
        if e["event_type"] == "bounce" and occurrence.resolved_membership(e, default=None)
    ]


def camera_rows(cameras_doc: dict) -> dict[int, dict]:
    return {int(r["frame"]): r for r in cameras_doc["cameras"]}


@contextmanager
def context(
    labels_doc: dict, cameras_doc: dict, policy: str = "observed-contact"
) -> Iterator[dict]:
    """Yield a mutable receipt while the association recoveries are scoped in.

    ``policy='original'`` patches nothing and yields an unapplied receipt so callers
    can use one code path for control runs.
    """
    if policy not in POLICIES:
        raise ValueError(f"association policy must be one of {POLICIES}")
    receipt = dict(policy=policy, applications=[], refusals=[])
    if policy == "original":
        yield receipt
        return
    cameras = camera_rows(cameras_doc)
    grounds = labeled_grounds(labels_doc)
    original_players = whole.alternating_player_states
    original_pixel = whole.contact_association_pixel
    original_server = whole.single.server_state
    original_feet = toss_witness.server_feet
    # Keyed by first-contact epoch: the association pixel and the feet witness
    # receive no clip, and the context is scoped to a single attempt.
    recovered_pixels: dict[float, np.ndarray] = {}
    recovered_first_state: dict[str, dict] = {}

    def record(name: str, application: dict, fallback_receipt) -> None:
        receipt["applications"].append(dict(recovery=name, **application))
        if fallback_receipt is not None:
            fallback_receipt.append(dict(fallback=name, **application))

    def refuse(name: str, contacts: list[dict], error: ValueError) -> None:
        receipt["refusals"].append(
            dict(
                recovery=name,
                first_contact_frame=float(contacts[0]["frame"]),
                reason=f"{type(error).__name__}: {error}",
            )
        )

    def players(pose_csv, clip, contacts, labels, **kwargs):
        # A search can rebuild the contact topology within the same scope.
        # Re-qualification must not inherit a previous call's override.
        recovered_pixels.clear()
        recovered_first_state.clear()
        scale = kwargs.get("image_coordinate_scale", 1.0)
        fallback_receipt = kwargs.get("fallback_receipt")

        def lookup(frame, pixel):
            return original_server(pose_csv, clip, frame, pixel, image_coordinate_scale=scale)

        if not contacts:
            return original_players(pose_csv, clip, contacts, labels, **kwargs)
        epoch = float(contacts[0]["frame"])
        try:
            pixel, application = launch.qualify(
                contacts, labels, original_fronts(labels_doc, clip), cameras, grounds, lookup
            )
        except ValueError as error:
            refuse(MISSING_LAUNCH, contacts, error)
        else:
            # Qualified: the association pixel is shared by the player-state
            # association and the serve-feet pixel lookup.  Errors from here
            # are execution failures and propagate.
            recovered_pixels[epoch] = pixel
            result = original_players(pose_csv, clip, contacts, labels, **kwargs)
            application["result_sides"] = [r["side"] for r in result]
            record(MISSING_LAUNCH, application, fallback_receipt)
            return result
        rows = player_state_fallback.sided_court_rows(pose_csv, clip)
        try:
            application = initial_camera.qualify(
                contacts, labels, cameras, rows, lookup, labels_doc["events"]["records"], clip
            )
        except initial_camera.NotApplicable:
            pass
        else:
            result, application, first_state = initial_camera.recover(
                pose_csv,
                clip,
                contacts,
                labels,
                application,
                original_players=original_players,
                original_server=original_server,
                **kwargs,
            )
            recovered_first_state["state"] = first_state
            recovered_first_state["recovery"] = initial_camera.POLICY
            record(initial_camera.POLICY, application, fallback_receipt)
            return result
        try:
            first_player.qualify(contacts, labels, cameras, rows, lookup)
        except ValueError as error:
            refuse(MISSING_FIRST_PLAYER, contacts, error)
            return original_players(pose_csv, clip, contacts, labels, **kwargs)
        result, application, first_state = first_player.recovered_player_states(
            pose_csv,
            clip,
            contacts,
            labels,
            cameras,
            original_players=original_players,
            original_server=original_server,
            **kwargs,
        )
        recovered_first_state["state"] = first_state
        record(
            MISSING_FIRST_PLAYER, dict(application="player_states", **application), fallback_receipt
        )
        return result

    def pixel(event, labels, **kwargs):
        epoch = float(event["frame"])
        if epoch in recovered_pixels:
            return recovered_pixels[epoch].copy()
        return original_pixel(event, labels, **kwargs)

    def feet(pose_csv, clip, contact_frame, contact_pixel, **kwargs):
        first_state = recovered_first_state.get("state")
        if first_state is None or int(np.ceil(float(contact_frame))) != first_state["frame"]:
            return original_feet(pose_csv, clip, contact_frame, contact_pixel, **kwargs)
        proxy = first_player.recovered_server_feet(first_state)
        record(
            f"{recovered_first_state.get('recovery', MISSING_FIRST_PLAYER)}_server_feet",
            dict(application="server_feet", **copy.deepcopy(proxy)),
            kwargs.get("fallback_receipt"),
        )
        return proxy

    whole.alternating_player_states = players
    whole.contact_association_pixel = pixel
    toss_witness.server_feet = feet
    try:
        yield receipt
    finally:
        whole.alternating_player_states = original_players
        whole.contact_association_pixel = original_pixel
        whole.single.server_state = original_server
        toss_witness.server_feet = original_feet
        recovered_pixels.clear()
        recovered_first_state.clear()
