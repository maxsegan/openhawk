"""Qualify first-hitter side across an unsupported initial broadcast camera.

Labeled-input preparation only. Original observed end metadata must agree with
at least two independently pictured later contacts. No camera is transported
across the cut and no distant outgoing front determines the serving side.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import agent_whole_point_search as whole
from cv.experiments.connected_shooting import labeled_missing_first_player as first
from cv.experiments.connected_shooting import player_state_fallback
from cv.pipeline import provenance

POLICY = "unsupported_contact_camera_observed_end_consensus"


class NotApplicable(ValueError):
    """This input is outside the narrow recovery domain; retain ordinary behavior."""


class MissingOriginalContact(ValueError):
    """No original record exists; distinct from multiple ambiguous matches."""


def original_contact(event: dict, records: list[dict], clip: str) -> dict:
    epoch = float(event.get("original_representative_frame", event["frame"]))
    matches = [
        r
        for r in records
        if r.get("event_type") == "contact"
        and r.get("clip", clip) == clip
        and float(r["frame"]) == epoch
    ]
    if not matches:
        raise MissingOriginalContact("no original contact record for observed end qualification")
    if len(matches) != 1:
        raise ValueError("one original contact record required for observed end qualification")
    return matches[0]


def qualify(contacts, labels, cameras, rows, lookup, records, clip) -> dict:
    frame = int(np.ceil(float(contacts[0]["frame"])))
    if first.camera_supported(cameras, frame):
        raise NotApplicable("first contact camera already supported")
    try:
        original = original_contact(contacts[0], records, clip)
    except MissingOriginalContact as error:
        raise NotApplicable(str(error)) from error
    if "hitter_end" not in original:
        raise NotApplicable("no explicit original first-hitter end metadata")
    if original.get("status") != "labeled" or original["hitter_end"] not in {"near", "far"}:
        raise ValueError("original first-hitter end is ambiguous")
    side, anchors = first.later_consensus(contacts, labels, cameras, rows, lookup)
    if side != original["hitter_end"]:
        raise ValueError("later contact consensus contradicts original first-hitter end")
    semantics = []
    for index, event in enumerate(contacts):
        record = original_contact(event, records, clip)
        expected = side if index % 2 == 0 else first.opposite(side)
        observed = record.get("hitter_end")
        if observed is not None and (record.get("status") != "labeled" or observed != expected):
            raise ValueError("original contact ends contradict the qualified alternating topology")
        semantics.append(
            dict(contact_index=index, original_frame=record["frame"], hitter_end=observed)
        )
    return dict(
        first_pose_frame=frame,
        first_contact_camera_supported=False,
        implied_first_side=side,
        later_supported_contact_associations=anchors,
        original_contact_semantics=semantics,
        trajectory_observations_added=0,
        camera_transport_across_cut=False,
    )


def recover(
    pose_csv: Path,
    clip,
    contacts,
    labels,
    application,
    *,
    original_players,
    original_server,
    **kwargs,
):
    names = kwargs.get("player_names")
    name = None if names is None else names[0]
    state = player_state_fallback.contact_state(
        pose_csv,
        clip,
        application["first_pose_frame"],
        application["implied_first_side"],
        player_name=name,
        stature_m=None if name is None else kwargs["player_statures_m"][name],
        image_coordinate_scale=kwargs.get("image_coordinate_scale", 1.0),
    )
    if state is None:
        raise ValueError("qualified first side has no nearby automatic court-position support")

    def server(path, clip_, frame, pixel, **options):
        if int(frame) == application["first_pose_frame"] and options.get("required_side") is None:
            return deepcopy(state)
        return original_server(path, clip_, frame, pixel, **options)

    try:
        whole.single.server_state = server
        result = original_players(pose_csv, clip, contacts, labels, **kwargs)
    finally:
        whole.single.server_state = original_server
    return (
        result,
        dict(
            **application,
            result_sides=[r["side"] for r in result],
            substituted_first_state=deepcopy(state),
            pose_source=provenance.file_record(pose_csv),
        ),
        state,
    )
