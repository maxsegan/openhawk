"""Reuse the ordinary native player-depth cue at a bound original rally origin.

An interior contact becomes local index zero when a supported component starts.
Its source role, player association and native bracket survive; the existing
axial cue must survive too. This is a soft depth cue, never a measured height.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import csv
import hashlib
import json

import numpy as np

from cv.pipeline import s6_first_contact_role as role

FIELD = "original_rally_origin_depth_cue"
SCHEMA = "s6_original_rally_origin_depth_cue_v1"


def prepare(players, events, labels, cameras, rows, scale):
    """Apply the same qualification and constants as ordinary interior contacts."""
    if not role.rally(players):
        return None
    from cv.experiments.connected_shooting import labeled_interior_normal as normal

    contacts = [e for e in events if e["event_type"] == "contact"]
    proof = players[0][role.FIELD]
    if not contacts or contacts[0]["frame"] != proof["first_contact"]["frame"]:
        raise ValueError("rally depth cue requires the bound original first contact")
    context = dict(
        players=players[:1],
        events=events,
        scene=SimpleNamespace(contact_frames=np.array([e["frame"] for e in contacts])),
    )
    cue = normal.contact_cues(context, labels, cameras, rows, scale)[0]
    return dict(
        schema=SCHEMA,
        original_contact_index=proof["original_contact_index"],
        original_contact_frame=contacts[0]["frame"],
        role_sha256=hashlib.sha256(
            json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        side=players[0]["side"],
        cue=deepcopy(cue),
        source="ordinary interior contact_cues and depth_residuals at local origin zero",
        height_observation=False,
        acceptance_gates_changed=False,
    )


def residuals(xyz, receipt):
    if receipt is None:
        return np.empty(0)
    from cv.experiments.connected_shooting import labeled_interior_normal as normal

    if receipt.get("schema") != SCHEMA or receipt.get("height_observation") is not False:
        raise ValueError("source-qualified ordinary rally depth cue required")
    cue = receipt["cue"]
    if cue["status"] == "supported":
        if cue.get("sigma_m") != 1.0 or cue.get("pixel_weight") != 12.0:
            raise ValueError("ordinary contact cue weight and scale required")
        if not np.isclose(np.linalg.norm(cue["axis"]), 1.0):
            raise ValueError("unit original camera axial direction required")
    return normal.depth_residuals(np.asarray(xyz, float)[None], [cue], [0])


def verify_report(search, labels, cameras, pose_csv, scale):
    """Rebuild a declared source cue; old reports without this addition stay valid."""
    declared = search.get("configuration", {}).get(FIELD)
    candidates = [*search.get("coarse_candidates", []), *search.get("refined_candidates", [])]
    candidates += [search.get("selected"), search.get("diagnostic_candidate")]
    candidates += list(search.get("selected_arms", {}).values())
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fit = candidate.get("measurement", {}).get("fit", {})
        if fit.get(FIELD) != declared:
            raise ValueError(
                "rally origin depth cue differs between declared policy and executed fit"
            )
    if declared is None:
        return
    with pose_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = prepare(search["player_states"], search["events"], labels, cameras, rows, scale)
    if declared != expected:
        raise ValueError("rally origin depth cue differs from original native inputs")
