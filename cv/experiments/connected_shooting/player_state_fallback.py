"""Court position for a contact whose automatic pose row is missing.

The connected fitter needs one court position per labeled contact: the striking
player's root, which the athlete arm turns into a *soft* reach hinge.  The
artifact it reads is a per-frame sided player table, and eight of the eighty-nine
labeled attempts lose the whole point because that table has no row of the
required side at one contact picture -- a detector gap of a few frames inside a
rally the labels describe completely.

A missing detector row is not evidence that the player was absent.  This module
supplies the same quantity from the nearest sided row of the same artifact, and
declares an inflated position uncertainty that grows with how far it reached.
Nothing here reads an evaluation label, a fitted result or a human court anchor;
the substituted row is an automatic detection like the one it replaces, and every
substitution is recorded with its frame distance so a bad reach is visible.

The declared floor, ``BOXES_ONLY_SIGMA_M``, is the sided-box court sigma the
``tennis_player_state_v1`` ledger publishes for boxes-only positions.  The
ledger's own audit measures sided boxes at 0.284 m median and 0.540 m p90
against dense truth, so 0.65 m is a deliberately wide statement of what a box
root is worth, not a fitted value.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from cv.experiments.connected_shooting import athlete_priors

BOXES_ONLY_SIGMA_M = 0.65
# One second of play at 25 fps.  A player crosses a few metres in that time, so
# the reach hinge is widened in proportion below rather than trusted at face
# value; beyond it the substitution says nothing useful and abstains.
DEFAULT_SEARCH_RADIUS_FRAMES = 25
# Court metres per frame of reach, used only to widen the declared sigma.
PLAYER_SPEED_M_PER_FRAME = 0.16


def _frame_number(value: str) -> int | None:
    """Read the ``f_0123.jpg`` picture name the player artifacts use."""
    text = str(value)
    if not text.startswith("f_"):
        return None
    try:
        return int(text[2:6])
    except ValueError:
        return None


def sided_court_rows(path: Path, clip: str) -> dict[tuple[int, str], dict[str, float]]:
    """Index one sided automatic player artifact by native frame and side."""
    rows: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"side", "court_x", "court_y"} <= fields:
            return rows
        for row in reader:
            if row.get("clip") != clip or row.get("side") not in {"near", "far"}:
                continue
            frame = _frame_number(row.get("frame", ""))
            if frame is None or not row.get("court_x") or not row.get("court_y"):
                continue
            try:
                position = (float(row["court_x"]), float(row["court_y"]))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(position).all():
                continue
            rows[(frame, row["side"])] = {
                "court_x": position[0],
                "court_y": position[1],
                "track_id": row.get("track_id"),
                **athlete_priors.root_observation(row),
            }
    return rows


def nearest_sided_position(
    rows: dict[tuple[int, str], dict[str, float]],
    frame: int,
    side: str,
    *,
    radius_frames: int = DEFAULT_SEARCH_RADIUS_FRAMES,
) -> dict[str, Any] | None:
    """The closest sided court position to ``frame``, or ``None`` inside ``radius``."""
    if side not in {"near", "far"}:
        raise ValueError("required player side must be near or far")
    candidates = [key[0] for key in rows if key[1] == side and abs(key[0] - frame) <= radius_frames]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda value: (abs(value - frame), value))
    row = rows[(chosen, side)]
    offset = abs(chosen - frame)
    return {
        "court_x": row["court_x"],
        "court_y": row["court_y"],
        "track_id": row["track_id"],
        "source_frame": chosen,
        "frame_distance": offset,
        "court_sigma_m": float(BOXES_ONLY_SIGMA_M + PLAYER_SPEED_M_PER_FRAME * offset),
        **({"root_observation": row["root_observation"]} if "root_observation" in row else {}),
    }


def contact_state(
    pose_csv: Path,
    clip: str,
    frame: int,
    side: str,
    *,
    player_name: str | None = None,
    stature_m: float | None = None,
    image_coordinate_scale: float = 1.0,
    radius_frames: int = DEFAULT_SEARCH_RADIUS_FRAMES,
) -> dict[str, Any] | None:
    """Build the contact player state the fitter needs from a neighbouring row.

    Returns ``None`` when the artifact says nothing about this side anywhere in
    the bounded window, so the caller can still fail closed with its own reason.
    """
    nearest = nearest_sided_position(
        sided_court_rows(pose_csv, clip), frame, side, radius_frames=radius_frames
    )
    if nearest is None:
        return None
    return dict(
        frame=frame,
        side=side,
        player=player_name,
        stature_m=stature_m,
        court_centre_xy_m=[nearest["court_x"], nearest["court_y"]],
        **(
            {"root_observation": nearest["root_observation"]}
            if "root_observation" in nearest
            else {}
        ),
        court_position_source="nearest_sided_automatic_row",
        court_position_sigma_m=nearest["court_sigma_m"],
        image_association_distance_px=None,
        pixel_nearest_side=side,
        pixel_nearest_distance_px=None,
        sided_file_disagrees_with_pixel_nearest=False,
        unscaled_pixel_nearest_side=side,
        coordinate_scale_changes_pixel_nearest_side=False,
        image_coordinate_scale=image_coordinate_scale,
        pose_wrist_witness={
            "status": "abstained",
            "abstention_reason": "no_automatic_pose_row_at_this_contact_picture",
        },
        state_dof=2,
        substitution={
            "reason": "automatic player row absent at this contact picture",
            "source_frame": nearest["source_frame"],
            "frame_distance": nearest["frame_distance"],
            "declared_sigma_m": nearest["court_sigma_m"],
            "sigma_basis": (
                f"{BOXES_ONLY_SIGMA_M} m boxes-only court sigma widened by "
                f"{PLAYER_SPEED_M_PER_FRAME} m per frame of reach"
            ),
            "track_id": nearest["track_id"],
        },
        interpretation=(
            "nearest sided automatic court position of the required alternating side; "
            "an inflated-uncertainty substitute for an absent detector row, not a pose"
        ),
    )
