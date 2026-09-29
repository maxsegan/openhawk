"""Associate raw person detections with the near and far tennis players.

``--keep-unique-admissible-frames`` is production default-on (ByteTrack unique
frames of other admissible same-side tracks). Rollback: pass
``--no-keep-unique-admissible-frames``, or set
``SelectionConfig.keep_unique_admissible_frames = False``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.camera_cal import COURT_L, COURT_W, NET_Y
from cv.pipeline.player_identity import names_by_clip
from cv.pipeline.pose import (
    load_native_pose_rows,
    pose_foot_pixel,
    pose_hip_pixel,
    vertical_height_from_pixel,
)
from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    DIAGNOSTIC_MODE,
    build_provenance,
    file_record,
    file_sha256,
    load_provenance,
    write_provenance,
)
from cv.pipeline.player_court_v2 import build_tracklets, select_player_tracklets
from cv.pipeline.player_tracker import (
    STANDING_HEIGHT_M,
    Detection,
    RevivalBodyHistoryAudit,
    SelectionConfig,
    Track,
    TrackerConfig,
    gaps,
    ground_scales,
    select_players,
    track_detections,
)

COURT_X_MARGIN_M = 2.5
BASELINE_MARGIN_M = 8.0
# Court depth is noisiest at the net, where a metre of image error flips the side. Inside
# this band the side is carried over from the same player's last confident position.
NET_HYSTERESIS_M = 1.0
# Human sprint ceiling with margin; matches player_court_v3.V_MAX_MS.
V_MAX_MS = 9.0
NATIVE_BOX_COLUMNS = ("x0_native", "y0_native", "x1_native", "y1_native")
PLAYER_TRACKS_FIELDS = [
    "clip",
    "frame",
    "t",
    "side",
    "name",
    "track_id",
    "x0_native",
    "y0_native",
    "x1_native",
    "y1_native",
    "root_x_native",
    "root_y_native",
    "court_x",
    "court_y",
    "conf",
    "occluded",
    "airborne",
]
PLAYER_STATE_SCHEMA = "tennis_player_state_v1"
PLAYER_STATE_FIELDS = [
    "clip",
    "frame",
    "t",
    "side",
    "name",
    "track_id",
    "court_x",
    "court_y",
    "court_sigma_m",
    "position_source",
    "foot_x_native",
    "foot_y_native",
    "hip_x_native",
    "hip_y_native",
    "hip_height_proxy_m",
    "hip_height_sigma_m",
    "hip_source",
    "airborne",
    "occluded",
    "contact_frame",
    "racket_face_x_native",
    "racket_face_y_native",
    "racket_face_court_x",
    "racket_face_court_y",
    "racket_face_sigma_px",
    "racket_face_sigma_m",
    "racket_face_source",
    "stroke_type",
    "stroke_status",
]
# Opened-development error scales from ``player_truth_ledger``. They describe
# observation dispersion, not confidence probabilities.
POSE_FOOT_SIGMA_M = 1.00
BOX_FOOT_SIGMA_M = 0.65
HIP_HEIGHT_SIGMA_M = 0.25


# --- Per-contact striker resolution -------------------------------------------------------
#
# The whole-point tracker answers "where is the near player through this point"; a contact
# asks a narrower question -- "which body did the racket that hit this pixel belong to" -- and
# docs/wk1/point_fit5.md section 3 measures what happens when the fitter reads the first answer
# as if it were the second.  The resolver below answers the narrow question at one frame from
# the raw person detections, with a side constraint, a body-scale gate and a continuity gate
# against the point track, and it reports a confidence so a caller may refuse it.
#
# Distances are measured from the box edge in units of the box height: a person is 1.0, and a
# 1.4 m racket reach on a 1.8 m player is about 0.8.  Measured over 679 owner contact clicks on
# the 40 truth broadcasts, the nearest person detection to the click sits at a median of 0.14
# and a p99 of 0.41, so 0.50 is the band a real contact lives in and 1.00 is the outer limit
# past which no detection is called the striker.
CONTACT_REACH_BAND = 0.50
CONTACT_REACH_LIMIT = 1.00
# The window the fitter itself looks in (reconstruction.CONTACT_SIDE_BOX_WINDOW_SECONDS).
CONTACT_WINDOW_SECONDS = 0.16
# A candidate is a player only if its box is roughly as tall as a standing person at its own
# court position.  A lunging or sliding player is shorter and a jumping one taller, so the band
# is wide; it exists to refuse ball kids, line judges and half-boxes, not to judge posture.
CONTACT_SCALE_BAND = (0.45, 1.80)
PLAYER_HEIGHT_M = 1.8
# How far a body may have moved between the contact and the nearest track row that brackets it.
CONTACT_CONTINUITY_V_MAX_MS = 9.0
CONTACT_CONTINUITY_SLACK_M = 1.0
# Replace the tracker's own row only when the resolver is both confident and better than it.
CONTACT_MIN_CONFIDENCE = 0.35
CONTACT_MIN_REACH_GAIN = 0.10
# The fitter searches the contact time over +/-1 frame (anchor_first_fit's shared-contact
# offsets) and the owner labels contacts on half frames, so the striker is resolved over the
# same span rather than at one frame.  Each frame in the span is resolved on its own body.
CONTACT_PATCH_SPAN_FRAMES = 1
# The detector sometimes returns a partial box of a body it has already boxed whole -- a torso,
# or an arm-and-racket sliver -- and a sliver's edge is nearer the ball than the whole body's
# is, so pure image reach picks it and the box bottom then reads metres off. A box mostly
# inside a larger one whose court position is the same body is suppressed in favour of the
# larger box.
CONTACT_CONTAINMENT = 0.60
CONTACT_SAME_BODY_M = 1.5


def _box_native(row: dict, scale: float) -> tuple[float, float, float, float]:
    return tuple(scale * float(row[key]) for key in ("x0", "y0", "x1", "y1"))


def contact_reach(pixel: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    """Distance from ``pixel`` to the box edge, in units of the box height."""
    x0, y0, x1, y1 = box
    dx = max(x0 - pixel[0], 0.0, pixel[0] - x1)
    dy = max(y0 - pixel[1], 0.0, pixel[1] - y1)
    return float(np.hypot(dx, dy) / max(1.0, y1 - y0))


def standing_height_px(projection: np.ndarray | None, court: np.ndarray) -> float | None:
    """The pixel height of a standing person at ``court``, through the point's camera."""
    if projection is None:
        return None
    matrix = np.asarray(projection, float)
    image = []
    for height in (0.0, PLAYER_HEIGHT_M):
        point = matrix @ np.array([float(court[0]), float(court[1]), height, 1.0])
        if not np.isfinite(point).all() or abs(point[2]) < 1e-9:
            return None
        image.append(point[:2] / point[2])
    return float(abs(image[0][1] - image[1][1]))


def _court_of(row: dict, homography: np.ndarray) -> np.ndarray | None:
    foot = np.float32([[[0.5 * (float(row["x0"]) + float(row["x1"])), float(row["y1"])]]])
    court = cv2.perspectiveTransform(foot, homography)[0, 0]
    return np.asarray(court, float) if np.all(np.isfinite(court)) else None


def _suppress_partial_boxes(candidates: list[dict]) -> list[dict]:
    """Drop a box that is mostly inside a larger box of the same body."""
    order = sorted(candidates, key=lambda c: -c["area"])
    kept: list[dict] = []
    for candidate in order:
        x0, y0, x1, y1 = candidate["box"]
        swallowed = False
        for other in kept:
            a0, b0, a1, b1 = other["box"]
            overlap = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            if overlap < CONTACT_CONTAINMENT * max(candidate["area"], 1e-9):
                continue
            if float(np.linalg.norm(candidate["court"] - other["court"])) <= CONTACT_SAME_BODY_M:
                swallowed = True
                break
        if not swallowed:
            kept.append(candidate)
    return kept


def _continuity_penalty(
    court: np.ndarray,
    frame: float,
    anchors: list[tuple[int, np.ndarray]],
    fps: float,
) -> float:
    """0 when some bracketing track row can reach ``court`` at a run, 1 when none can."""
    if not anchors:
        return 0.0
    best = None
    for anchor_frame, anchor_court in anchors:
        seconds = abs(anchor_frame - frame) / max(fps, 1e-6)
        budget = CONTACT_CONTINUITY_V_MAX_MS * seconds + CONTACT_CONTINUITY_SLACK_M
        excess = float(np.linalg.norm(court - anchor_court)) - budget
        best = excess if best is None else min(best, excess)
    if best is None or best <= 0.0:
        return 0.0
    return float(min(1.0, best / 3.0))


def resolve_contact_striker(
    pixel: tuple[float, float],
    frame: float,
    detections: dict[int, list[dict]],
    homography: np.ndarray,
    *,
    fps: float,
    side: str | None = None,
    projection: np.ndarray | None = None,
    anchors: list[tuple[int, np.ndarray]] | None = None,
    box_scale: float = 2.0,
) -> dict | None:
    """The body the racket at ``pixel`` belonged to, or ``None`` when nothing qualifies.

    ``detections`` maps frame to raw person-box rows in the box artifact's own coordinate
    space; ``box_scale`` takes that space to the space ``pixel`` is in. ``homography`` takes
    the box space to court metres. ``side`` constrains the answer to one half of the court.
    ``anchors`` are (frame, court position) rows of the same side's point track outside the
    contact window, and are what stops a ball kid beside the ball from being called the striker.
    """
    window = max(1, int(np.ceil(CONTACT_WINDOW_SECONDS * fps)))
    centre = int(round(frame))
    anchors = anchors or []
    best: dict | None = None
    # The contact time is known to about a frame, so the body at this frame and its two
    # neighbours are all "the striker now" and the nearest of them to the ball wins.  Beyond
    # that the body has moved, and writing it into this frame would hand the fitter a stale
    # witness dressed as a fresh one, so wider frames are a fallback for a missed detection
    # only: the search stops at the first ring that holds a body.

    def ring(offset: int) -> int:
        return max(0, abs(offset) - CONTACT_PATCH_SPAN_FRAMES)

    offsets = sorted(range(-window, window + 1), key=lambda offset: (ring(offset), abs(offset)))
    for candidate_frame in (centre + offset for offset in offsets):
        if best is not None and ring(best["frame"] - centre) < ring(candidate_frame - centre):
            break
        frame_candidates = []
        for row in detections.get(candidate_frame, []):
            court = _court_of(row, homography)
            if court is None:
                continue
            if not (
                -COURT_X_MARGIN_M <= court[0] <= COURT_W + COURT_X_MARGIN_M
                and -BASELINE_MARGIN_M <= court[1] <= COURT_L + BASELINE_MARGIN_M
            ):
                continue
            box = _box_native(row, box_scale)
            frame_candidates.append(
                {
                    "row": row,
                    "court": court,
                    "box": box,
                    "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                }
            )
        for candidate in _suppress_partial_boxes(frame_candidates):
            court = candidate["court"]
            candidate_side = "near" if court[1] < NET_Y else "far"
            if side is not None and candidate_side != side:
                continue
            box = candidate["box"]
            reach = contact_reach(pixel, box)
            if reach > CONTACT_REACH_LIMIT:
                continue
            expected = standing_height_px(projection, court)
            if expected is not None and expected > 1.0:
                ratio = (box[3] - box[1]) / expected
                if not CONTACT_SCALE_BAND[0] <= ratio <= CONTACT_SCALE_BAND[1]:
                    continue
            else:
                ratio = None
            penalty = _continuity_penalty(court, frame, anchors, fps)
            score = reach + penalty
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "reach": reach,
                    "penalty": penalty,
                    "frame": candidate_frame,
                    "side": candidate_side,
                    "court": court,
                    "row": candidate["row"],
                    "scale_ratio": ratio,
                }
    if best is None:
        return None
    # Confidence: how far inside the reach band the winner sits, discounted by how much the
    # continuity gate had to forgive.  It is a stated rule, not a calibrated probability.
    confidence = max(0.0, 1.0 - best["reach"] / CONTACT_REACH_BAND) * (1.0 - best["penalty"])
    return {
        "side": best["side"],
        "frame": best["frame"],
        "court_x": float(best["court"][0]),
        "court_y": float(best["court"][1]),
        "reach": float(best["reach"]),
        "continuity_penalty": float(best["penalty"]),
        "scale_ratio": best["scale_ratio"],
        "confidence": float(min(1.0, confidence)),
        "row": best["row"],
    }


def patch_contact_strikers(
    sided: list[dict],
    raw: list[dict],
    contacts: list[dict],
    homographies: dict[int, np.ndarray],
    *,
    fps: float,
    image_size: res.FrameSize,
    artifact_size: res.FrameSize,
    frame_homographies: dict[str, dict[int, np.ndarray]] | None = None,
    projections: dict[str, dict[int, np.ndarray]] | None = None,
    point_projections: dict[int, np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Rewrite the sided box at each contact frame with the resolver's answer.

    Only the contact frames move: every other frame of the point track is the tracker's own.
    Returns the new rows and one ledger entry per contact and side.
    """
    native = bool(raw) and all(column in raw[0] for column in NATIVE_BOX_COLUMNS)
    box_size = res.NATIVE_SIZE if native else artifact_size
    box_scale = res.NATIVE_SIZE.width / box_size.width
    has_frame_track = frame_homographies is not None
    frame_homographies = frame_homographies or {}
    projections = projections or {}
    point_projections = point_projections or {}

    detections: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        detections[row["clip"]][frame_number(row["frame"])].append(row)
    indexed: dict[tuple[str, int, str], dict] = {}
    by_side: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    for row in sided:
        key = (str(row["clip"]), frame_number(str(row["frame"])), str(row["side"]))
        indexed[key] = row
        by_side[(str(row["clip"]), str(row["side"]))][key[1]] = np.array(
            [float(row["court_x"]), float(row["court_y"])]
        )

    window = max(1, int(np.ceil(CONTACT_WINDOW_SECONDS * fps)))
    ledger: list[dict] = []
    replaced: dict[tuple[str, int, str], dict] = {}
    spans = []
    for contact in contacts:
        emitted = float(contact["frame"])
        for offset in range(-CONTACT_PATCH_SPAN_FRAMES, CONTACT_PATCH_SPAN_FRAMES + 1):
            spans.append((contact, int(round(emitted)) + offset))
    for contact, centre in spans:
        clip = str(contact["clip"])
        if clip not in detections:
            continue
        point = int(clip.removeprefix("pt"))
        frame = float(centre)
        tracked = frame_homographies.get(clip, {}).get(centre)
        matrix = tracked if has_frame_track else homographies.get(point)
        if matrix is None:
            continue
        homography = res.image_to_world_homography(matrix, image_size, box_size)
        projection = projections.get(clip, {}).get(centre, point_projections.get(point))
        pixel = (float(contact["image_x"]), float(contact["image_y"]))
        for side in ("near", "far"):
            track = by_side.get((clip, side), {})
            anchors = [
                (candidate, track[candidate])
                for candidate in track
                if window < abs(candidate - frame) <= 3 * window
            ]
            resolved = resolve_contact_striker(
                pixel,
                frame,
                detections[clip],
                homography,
                fps=fps,
                side=side,
                projection=projection,
                anchors=anchors,
                box_scale=box_scale,
            )
            incumbent = indexed.get((clip, centre, side))
            incumbent_reach = None
            if incumbent is not None:
                incumbent_reach = contact_reach(pixel, _box_native(incumbent, box_scale))
            entry = {
                "clip": clip,
                "frame": frame,
                "emitted_frame": float(contact["frame"]),
                "side": side,
                "incumbent_reach": incumbent_reach,
                "resolved_reach": None if resolved is None else round(resolved["reach"], 4),
                "confidence": None if resolved is None else round(resolved["confidence"], 4),
                "action": "none",
            }
            if resolved is not None and resolved["confidence"] >= CONTACT_MIN_CONFIDENCE:
                gain = (
                    CONTACT_REACH_LIMIT
                    if incumbent_reach is None
                    else incumbent_reach - resolved["reach"]
                )
                if gain >= CONTACT_MIN_REACH_GAIN:
                    source = resolved["row"]
                    row = {
                        "clip": clip,
                        "frame": f"f_{centre:04d}.jpg",
                        "side": side,
                        **{key: source[key] for key in ("x0", "y0", "x1", "y1") if key in source},
                        **{key: source[key] for key in NATIVE_BOX_COLUMNS if key in source},
                        "conf": source.get("conf", ""),
                        "court_x": round(resolved["court_x"], 3),
                        "court_y": round(resolved["court_y"], 3),
                        "track_id": (
                            incumbent.get("track_id", "") if incumbent is not None else "resolved"
                        ),
                    }
                    replaced[(clip, centre, side)] = row
                    entry["action"] = "replaced" if incumbent is not None else "inserted"
            ledger.append(entry)

    output = [dict(row) for row in sided]
    for index, row in enumerate(output):
        key = (str(row["clip"]), frame_number(str(row["frame"])), str(row["side"]))
        if key in replaced:
            output[index] = replaced.pop(key)
    output.extend(replaced.values())
    return (
        sorted(
            output,
            key=lambda row: (row["clip"], frame_number(str(row["frame"])), str(row["side"])),
        ),
        ledger,
    )


def frame_number(value: str) -> int:
    return int("".join(character for character in value if character.isdigit()))


def load_homographies(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {
        int(point): np.asarray(homography, dtype=float)
        for point, homography in zip(data["pts"], data["H"], strict=True)
        if np.isfinite(homography).all()
    }


def propagate_nearest_registered(
    reliable: dict[str, dict[int, np.ndarray]],
    candidates: dict[str, dict[int, np.ndarray]],
    shot_ids: dict[str, dict[int, int]] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Copy the nearest reliable homography onto finite frames of the same shot.

    A static wide shot stores the one registered camera as ``anchor_static_fallback``
    on the other frames of that shot. Side association that reads only reliable
    rows then emits an empty sided inventory. A different shot, including a
    close-up, is left empty. Without shot ids nothing is copied: the caller has
    not shown that the frames are one shot.
    """
    filled: dict[str, dict[int, np.ndarray]] = {}
    shots = shot_ids or {}
    for clip, frames in candidates.items():
        registered = reliable.get(clip) or {}
        if not registered:
            filled[clip] = {}
            continue
        clip_shots = shots.get(clip) or {}
        keys = [int(frame) for frame in registered]
        clip_filled = dict(registered)
        for frame in frames:
            frame = int(frame)
            if frame in clip_filled:
                continue
            shot = clip_shots.get(frame)
            if shot is None:
                continue
            same = [key for key in keys if clip_shots.get(key) == shot]
            if not same:
                continue
            nearest = min(same, key=lambda key: abs(key - frame))
            clip_filled[frame] = registered[nearest]
        filled[clip] = clip_filled
    for clip, registered in reliable.items():
        filled.setdefault(clip, dict(registered))
    return filled


def load_frame_homographies(
    path: Path,
    *,
    propagate_shot: bool = False,
    shot_ids: dict[str, dict[int, int]] | None = None,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Read only reliable frame geometry; None denotes an absent legacy track.

    An empty mapping is an actual abstention, never permission to resurrect a static
    point camera. Legacy artifacts without reliability do not certify any frames.
    ``propagate_shot`` fills each finite court-track frame with the nearest
    reliable homography of the same shot. Callers leave it off unless the
    production policy names it. Without ``shot_ids`` nothing is copied onto an
    unreliable frame.
    """
    if not path.exists():
        return None
    track: dict[str, dict[int, np.ndarray]] = {}
    finite: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path) as data:
        reliable = data.get("reliable", np.zeros(len(data["frames"]), dtype=bool))
        if reliable.dtype != np.dtype(bool):
            raise ValueError("court frame reliability must be a boolean array")
        for clip, frame, homography, accepted in zip(
            data["clips"], data["frames"], data["H"], reliable, strict=True
        ):
            matrix = np.asarray(homography, dtype=float)
            clip_name = str(clip)
            frame_index = int(frame)
            track.setdefault(clip_name, {})
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                continue
            finite.setdefault(clip_name, {})[frame_index] = matrix
            if accepted:
                track[clip_name][frame_index] = matrix
    if propagate_shot:
        return propagate_nearest_registered(track, finite, shot_ids)
    return track


def _side_with_hysteresis(
    court_y: float,
    frame: int,
    fps: float,
    last: dict[str, tuple[int, float, float]],
    court_x: float,
) -> str:
    """Assign near/far, carrying the previous side through the ambiguous net band."""
    if court_y < NET_Y - NET_HYSTERESIS_M:
        return "near"
    if court_y > NET_Y + NET_HYSTERESIS_M:
        return "far"
    reachable = {}
    for side, (last_frame, last_x, last_y) in last.items():
        gap_seconds = abs(frame - last_frame) / max(fps, 1e-6)
        distance = float(np.hypot(court_x - last_x, court_y - last_y))
        if distance <= V_MAX_MS * gap_seconds + NET_HYSTERESIS_M:
            reachable[side] = distance
    if reachable:
        return min(reachable, key=reachable.__getitem__)
    return "near" if court_y < NET_Y else "far"


def coordinate_sizes(path: Path) -> tuple[res.FrameSize, res.FrameSize]:
    manifest = json.loads(path.with_suffix(path.suffix + ".coordinates.json").read_text())
    image = manifest["image_size"]
    artifact = manifest.get("legacy_artifact_size", manifest["artifact_size"])
    return (
        res.FrameSize(int(image["width"]), int(image["height"])),
        res.FrameSize(int(artifact["width"]), int(artifact["height"])),
    )


def associate_rows(
    rows: list[dict[str, str]],
    homographies: dict[int, np.ndarray],
    *,
    fps: float,
    image_size: res.FrameSize,
    artifact_size: res.FrameSize,
    frame_homographies: dict[str, dict[int, np.ndarray]] | None = None,
    tracker: str = "bytetrack",
    embeddings: np.ndarray | None = None,
    frame_counts: dict[str, int] | None = None,
    tracker_config: TrackerConfig | None = None,
    selection_config: SelectionConfig | None = None,
    audit: RevivalBodyHistoryAudit | None = None,
) -> list[dict[str, str | float | int]]:
    """Sided player boxes.

    ``tracker="bytetrack"`` is the identity-preserving default and returns exactly one
    track per clip-side. ``tracker="greedy"`` retains the week-one court-space
    nearest-neighbour linker for explicit baseline comparisons.
    """
    if tracker == "bytetrack":
        return sided_rows_from_tracks(
            track_clips(
                build_detections(
                    rows,
                    homographies,
                    image_size=image_size,
                    artifact_size=artifact_size,
                    frame_homographies=frame_homographies,
                    embeddings=embeddings,
                ),
                fps=fps,
                frame_counts=frame_counts,
                tracker_config=tracker_config,
                selection_config=selection_config,
                audit=audit,
            )
        )
    if tracker != "greedy":
        raise ValueError(f"unknown tracker {tracker!r}")
    by_clip: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_clip[row["clip"]][frame_number(row["frame"])].append(row)
    has_frame_track = frame_homographies is not None
    frame_homographies = frame_homographies or {}
    native = bool(rows) and all(column in rows[0] for column in NATIVE_BOX_COLUMNS)
    # The stored homography takes native pixels; only the legacy path needs rescaling.
    box_size = res.NATIVE_SIZE if native else artifact_size
    left, _top, right, bottom = NATIVE_BOX_COLUMNS if native else ("x0", "y0", "x1", "y1")

    output = []
    for clip, clip_frames in sorted(by_clip.items()):
        point = int(clip.removeprefix("pt"))
        if point not in homographies:
            continue
        clip_track = frame_homographies.get(clip, {})
        point_homography = res.image_to_world_homography(
            homographies[point],
            image_size,
            box_size,
        )
        candidates: dict[str, dict[int, list[dict]]] = {
            "near": defaultdict(list),
            "far": defaultdict(list),
        }
        last_confident: dict[str, tuple[int, float, float]] = {}
        for frame in sorted(clip_frames):
            tracked = clip_track.get(frame)
            if has_frame_track and tracked is None:
                continue
            homography = (
                res.image_to_world_homography(tracked, image_size, box_size)
                if tracked is not None
                else point_homography
            )
            for row in clip_frames[frame]:
                # The box bottom centre stays the root: 3.53 px540 median on the owner's
                # 85 corrected roots against 11.98 for the ankle mean.
                foot_x = 0.5 * (float(row[left]) + float(row[right]))
                foot_y = float(row[bottom])
                court_x, court_y = cv2.perspectiveTransform(
                    np.float32([[[foot_x, foot_y]]]),
                    homography,
                )[0, 0]
                if not (
                    -COURT_X_MARGIN_M <= court_x <= COURT_W + COURT_X_MARGIN_M
                    and -BASELINE_MARGIN_M <= court_y <= COURT_L + BASELINE_MARGIN_M
                ):
                    continue
                side = _side_with_hysteresis(
                    float(court_y), frame, fps, last_confident, float(court_x)
                )
                if abs(float(court_y) - NET_Y) > NET_HYSTERESIS_M:
                    last_confident[side] = (frame, float(court_x), float(court_y))
                candidates[side][frame].append(
                    {
                        **row,
                        "cx": float(court_x),
                        "cy": float(court_y),
                        "side": side,
                    }
                )
        for side in ("near", "far"):
            tracklets = build_tracklets(candidates[side], fps)
            selected = select_player_tracklets(tracklets, side, fps)
            for tracklet in selected:
                for row in tracklet.rows:
                    output.append(
                        {
                            "clip": clip,
                            "frame": row["frame"],
                            "side": side,
                            "x0": row["x0"],
                            "y0": row["y0"],
                            "x1": row["x1"],
                            "y1": row["y1"],
                            **(
                                {
                                    "x0_native": row["x0_native"],
                                    "y0_native": row["y0_native"],
                                    "x1_native": row["x1_native"],
                                    "y1_native": row["y1_native"],
                                }
                                if "x0_native" in row
                                else {}
                            ),
                            "conf": row["conf"],
                            "court_x": round(float(row["cx"]), 3),
                            "court_y": round(float(row["cy"]), 3),
                            "track_id": tracklet.tid,
                        }
                    )
    return sorted(
        output,
        key=lambda row: (
            row["clip"],
            frame_number(str(row["frame"])),
            str(row["side"]),
        ),
    )


def _clip_homographies(
    clip: str,
    point: int,
    homographies: dict[int, np.ndarray],
    frame_homographies: dict[str, dict[int, np.ndarray]],
    *,
    image_size: res.FrameSize,
    box_size: res.FrameSize,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    point_homography = res.image_to_world_homography(homographies[point], image_size, box_size)
    per_frame = {
        frame: res.image_to_world_homography(matrix, image_size, box_size)
        for frame, matrix in frame_homographies.get(clip, {}).items()
    }
    return point_homography, per_frame


def build_detections(
    rows: list[dict[str, str]],
    homographies: dict[int, np.ndarray],
    *,
    image_size: res.FrameSize,
    artifact_size: res.FrameSize,
    frame_homographies: dict[str, dict[int, np.ndarray]] | None = None,
    embeddings: np.ndarray | None = None,
) -> dict[str, list[Detection]]:
    """Project every person box onto the court and wrap it as a tracker detection.

    ``court_x``/``court_y`` come from that frame's homography when a frame track is
    supplied and reliable; missing/rejected frame geometry abstains. Only an absent
    legacy track uses the point homography. An ``embed_index`` column, when
    present, indexes ``embeddings`` and gives the detection its appearance vector.
    """
    has_frame_track = frame_homographies is not None
    frame_homographies = frame_homographies or {}
    native = bool(rows) and all(column in rows[0] for column in NATIVE_BOX_COLUMNS)
    box_size = res.NATIVE_SIZE if native else artifact_size
    left, top, right, bottom = NATIVE_BOX_COLUMNS if native else ("x0", "y0", "x1", "y1")
    scale = res.NATIVE_SIZE.width / box_size.width

    by_clip: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_clip[row["clip"]][frame_number(row["frame"])].append(row)

    detections: dict[str, list[Detection]] = {}
    for clip, clip_frames in sorted(by_clip.items()):
        point = int(clip.removeprefix("pt"))
        if point not in homographies:
            continue
        point_homography, per_frame = _clip_homographies(
            clip,
            point,
            homographies,
            frame_homographies,
            image_size=image_size,
            box_size=box_size,
        )
        collected: list[Detection] = []
        for frame in sorted(clip_frames):
            homography = per_frame.get(frame) if has_frame_track else point_homography
            if homography is None:
                continue
            frame_rows = clip_frames[frame]
            feet = np.float32(
                [[[0.5 * (float(r[left]) + float(r[right])), float(r[bottom])]] for r in frame_rows]
            )
            court = cv2.perspectiveTransform(feet, homography)[:, 0, :]
            scales = ground_scales(homography, court)
            for row, (court_x, court_y), ground in zip(frame_rows, court, scales, strict=True):
                if not (
                    -COURT_X_MARGIN_M <= court_x <= COURT_W + COURT_X_MARGIN_M
                    and -BASELINE_MARGIN_M <= court_y <= COURT_L + BASELINE_MARGIN_M
                ):
                    continue
                embedding = None
                index = int(float(row.get("embed_index", -1) or -1))
                if embeddings is not None and 0 <= index < len(embeddings):
                    embedding = np.asarray(embeddings[index], dtype=np.float32)
                collected.append(
                    Detection(
                        frame=frame,
                        box=(
                            float(row[left]) * scale,
                            float(row[top]) * scale,
                            float(row[right]) * scale,
                            float(row[bottom]) * scale,
                        ),
                        conf=float(row["conf"]),
                        court=(float(court_x), float(court_y)),
                        ground_scale=float(ground) * scale,
                        embedding=embedding,
                        row=row,
                    )
                )
        detections[clip] = collected
    return detections


def camera_body_scale(
    detections: dict[str, list[Detection]], projections: dict[int, np.ndarray]
) -> dict[str, list[Detection]]:
    """Let the camera's own standing height vouch for a body the ground scale refuses.

    The tracker's size ratio divides box height by ``STANDING_HEIGHT_M`` times the
    horizontal ground scale at the foot. On a steep camera the vertical is foreshortened
    more than the ground, so a standing near player can fall under the floor (panel D
    Monte Carlo, median 0.57 against 0.62). The automatic camera's P gives a second
    estimate, the projected length of a standing body at the foot point, but its height
    is not certified as metric: on flatter cameras it puts near players at 1.4-1.7. So
    neither scale is trusted alone. Where ``projections`` holds a 3x4 P for the frame
    (native pixels, court metres, z up), the detection keeps whichever scale puts its
    ratio nearer 1 (in log), and the gate refuses a body only when both call it the
    wrong size. Frames without P keep the ground scale.
    """
    output: dict[str, list[Detection]] = {}
    for clip, clip_detections in detections.items():
        scaled: list[Detection] = []
        for detection in clip_detections:
            projection = projections.get(detection.frame)
            ground = detection.size_ratio
            if projection is None or not math.isfinite(ground) or ground <= 0:
                scaled.append(detection)
                continue
            x, y = detection.court
            ends = projection @ np.array([[x, y, 0.0, 1.0], [x, y, STANDING_HEIGHT_M, 1.0]]).T
            if not np.all(np.isfinite(ends)) or np.any(ends[2] <= 0):
                scaled.append(detection)
                continue
            pixels = ends[:2] / ends[2]
            standing = float(np.linalg.norm(pixels[:, 1] - pixels[:, 0]))
            camera = detection.height / standing if standing > 0 else float("nan")
            if (
                math.isfinite(camera)
                and camera > 0
                and abs(math.log(camera)) < abs(math.log(ground))
            ):
                detection = replace(detection, ground_scale=standing / STANDING_HEIGHT_M)
            scaled.append(detection)
        output[clip] = scaled
    return output


def track_clips(
    detections: dict[str, list[Detection]],
    *,
    fps: float,
    frame_counts: dict[str, int] | None = None,
    tracker_config: TrackerConfig | None = None,
    selection_config: SelectionConfig | None = None,
    bilateral_overlap: bool = False,
    overlap_recovery: bool = False,
    recovery_frame_homographies: dict[str, dict[int, np.ndarray]] | None = None,
    overlap_audit: list[dict] | None = None,
    audit: RevivalBodyHistoryAudit | None = None,
) -> dict[str, dict[str, Track]]:
    """One near track and one far track per clip (a side is absent when nothing fits)."""
    frame_counts = frame_counts or {}
    out: dict[str, dict[str, Track]] = {}
    for clip, clip_detections in detections.items():
        if not clip_detections:
            out[clip] = {}
            continue
        frames = {d.frame for d in clip_detections}
        n_frames = frame_counts.get(clip) or (max(frames) - min(frames) + 1)
        if audit is not None:
            audit.clip = clip
            audit.observed_frame_range = (min(frames), max(frames))
        tracks = track_detections(
            clip_detections,
            fps=fps,
            config=tracker_config,
            selection=selection_config,
            audit=audit,
        )
        audit_start = len(overlap_audit) if overlap_audit is not None else 0
        out[clip] = select_players(
            tracks,
            fps=fps,
            n_frames=n_frames,
            selection=selection_config,
            config=tracker_config,
            bilateral_overlap=bilateral_overlap,
            overlap_recovery=overlap_recovery,
            reliable_frames=(
                set(recovery_frame_homographies.get(clip, {}))
                if recovery_frame_homographies is not None
                else None
            ),
            overlap_audit=overlap_audit,
        )
        if overlap_audit is not None:
            for record in overlap_audit[audit_start:]:
                record["clip"] = clip
    return out


def track_reliable_views(
    detections: dict[str, list[Detection]],
    frame_homographies: dict[str, dict[int, np.ndarray]],
    *,
    fps: float,
    tracker_config: TrackerConfig | None = None,
    selection_config: SelectionConfig | None = None,
    bilateral_overlap: bool = False,
    overlap_recovery: bool = False,
    recovery_frame_homographies: dict[str, dict[int, np.ndarray]] | None = None,
    overlap_audit: list[dict] | None = None,
    audit: RevivalBodyHistoryAudit | None = None,
) -> dict[str, dict[str, Track]]:
    """Associate independently inside each contiguous reliable-camera span.

    A held/missing camera row ends identity support, even when motion prediction
    could bridge it. Detector misses inside a reliable span do not split it.
    Span tracks are collected for export only; no subsequent association, scoring
    or identity stitching is performed on the combined side container.
    """
    output: dict[str, dict[str, Track]] = {}
    for clip, clip_detections in detections.items():
        spans: list[list[int]] = []
        for frame in sorted(frame_homographies.get(clip, {})):
            if not spans or frame != spans[-1][-1] + 1:
                spans.append([])
            spans[-1].append(frame)
        by_frame: dict[int, list[Detection]] = defaultdict(list)
        for detection in clip_detections:
            by_frame[detection.frame].append(detection)
        selected: dict[str, list[Track]] = defaultdict(list)
        for span_id, frames in enumerate(spans):
            rows = [d for frame in frames for d in by_frame.get(frame, [])]
            scoped = track_clips(
                {clip: rows},
                fps=fps,
                frame_counts={clip: len(frames)},
                tracker_config=tracker_config,
                selection_config=selection_config,
                bilateral_overlap=bilateral_overlap,
                overlap_recovery=overlap_recovery,
                recovery_frame_homographies={clip: dict.fromkeys(frames)},
                overlap_audit=overlap_audit,
                audit=audit,
            )[clip]
            for side, track in scoped.items():
                identity = 2 * span_id + (side == "far")
                selected[side].append(
                    replace(
                        track,
                        track_id=identity,
                        association_ids=dict.fromkeys(track.frames, identity),
                    )
                )
        output[clip] = {}
        for side, parts in selected.items():
            rows = [d for part in parts for d in part.detections]
            output[clip][side] = replace(
                parts[-1],
                detections=rows,
                hits=len(rows),
                start_frame=rows[0].frame,
                association_ids={f: i for part in parts for f, i in part.association_ids.items()},
                stitch_boundaries=[b for part in parts for b in part.stitch_boundaries],
            )
    return output


def sided_rows_from_tracks(tracked: dict[str, dict[str, Track]]) -> list[dict]:
    """The legacy sided-box schema, emitted from the tracker instead of the greedy linker."""
    output: list[dict] = []
    for clip, sides in tracked.items():
        for side, track in sides.items():
            for detection in track.detections:
                row = detection.row or {}
                output.append(
                    {
                        "clip": clip,
                        "frame": row.get("frame", f"f_{detection.frame:04d}.jpg"),
                        "side": side,
                        "x0": row.get("x0", ""),
                        "y0": row.get("y0", ""),
                        "x1": row.get("x1", ""),
                        "y1": row.get("y1", ""),
                        **{column: row[column] for column in NATIVE_BOX_COLUMNS if column in row},
                        "conf": row.get("conf", round(detection.conf, 3)),
                        "court_x": round(detection.court[0], 3),
                        "court_y": round(detection.court[1], 3),
                        "track_id": track.association_ids.get(detection.frame, track.track_id),
                    }
                )
    return sorted(
        output,
        key=lambda row: (row["clip"], frame_number(str(row["frame"])), str(row["side"])),
    )


def track_rows(
    tracked: dict[str, dict[str, Track]],
    *,
    names: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Rows for ``player_tracks_native_v1.csv``: one per tracked player per frame.

    ``names`` maps clip -> {side: player name}; it stays blank when the match has no
    identity artifact. ``occluded`` and ``airborne`` are declared placeholders (always
    blank) so downstream consumers can bind to the column now.
    """
    names = names or {}
    output: list[dict] = []
    for clip, sides in tracked.items():
        for side, track in sides.items():
            for detection in track.detections:
                row = detection.row or {}
                x0, y0, x1, y1 = detection.box
                output.append(
                    {
                        "clip": clip,
                        "frame": detection.frame,
                        # Copy the detector artifact's timestamp. Do not reconstruct it
                        # from frame/fps: source cadence can be fractional or irregular.
                        "t": row.get("t", ""),
                        "side": side,
                        "name": names.get(clip, {}).get(side, ""),
                        "track_id": track.association_ids.get(detection.frame, track.track_id),
                        "x0_native": round(x0, 1),
                        "y0_native": round(y0, 1),
                        "x1_native": round(x1, 1),
                        "y1_native": round(y1, 1),
                        "root_x_native": round(0.5 * (x0 + x1), 1),
                        "root_y_native": round(y1, 1),
                        "court_x": round(detection.court[0], 3),
                        "court_y": round(detection.court[1], 3),
                        "conf": round(detection.conf, 3),
                        "occluded": "",
                        "airborne": "",
                    }
                )
    return sorted(output, key=lambda row: (row["clip"], row["frame"], row["side"]))


def _project_pixel(
    homography: np.ndarray | None, pixel: tuple[float, float] | None
) -> tuple[float, float] | None:
    if homography is None or pixel is None:
        return None
    point = cv2.perspectiveTransform(
        np.float32([[[float(pixel[0]), float(pixel[1])]]]),
        np.asarray(homography, dtype=np.float32),
    )[0, 0]
    if not np.isfinite(point).all():
        return None
    return float(point[0]), float(point[1])


def _pixel_sigma_to_court(
    homography: np.ndarray | None,
    pixel: tuple[float, float] | None,
    sigma_px: float | None,
) -> float | None:
    if homography is None or pixel is None or sigma_px is None:
        return None
    centre = _project_pixel(homography, pixel)
    along_x = _project_pixel(homography, (pixel[0] + sigma_px, pixel[1]))
    along_y = _project_pixel(homography, (pixel[0], pixel[1] + sigma_px))
    if centre is None or along_x is None or along_y is None:
        return None
    return float(
        max(
            np.linalg.norm(np.asarray(along_x) - np.asarray(centre)),
            np.linalg.norm(np.asarray(along_y) - np.asarray(centre)),
        )
    )


def build_player_state_rows(
    tracks: list[dict],
    pose_rows: list[dict],
    *,
    homography_for_frame,
    projection_for_frame,
    contact_strikers: list[dict] | None = None,
    foot_estimator: str = "sided_box",
) -> list[dict]:
    """Compose automatic track, pose and contact evidence into player states.

    All inputs are inference artifacts. Missing pose/camera evidence falls back
    to the sided box root or leaves the corresponding 3-D proxy blank. Racket
    face rows occur only on resolved contact frames.
    """
    pose_index: dict[tuple[str, int, str], dict] = {}
    for row in pose_rows:
        side = str(row.get("side") or "")
        if side not in ("near", "far"):
            continue
        key = (str(row["clip"]), frame_number(str(row["frame"])), side)
        incumbent = pose_index.get(key)
        if incumbent is None or float(row.get("conf", 0.0)) > float(incumbent.get("conf", 0.0)):
            pose_index[key] = row
    contacts: dict[tuple[str, int, str], dict] = {}
    for record in contact_strikers or []:
        if record.get("end") not in ("near", "far"):
            continue
        resolved_frame = record.get("resolved_frame")
        if resolved_frame is None:
            continue
        contacts[(str(record.get("clip") or ""), int(resolved_frame), record["end"])] = record

    output = []
    for track in tracks:
        clip = str(track["clip"])
        frame = frame_number(str(track["frame"]))
        side = str(track["side"])
        pose = pose_index.get((clip, frame, side))
        homography = homography_for_frame(clip, frame)
        projection = projection_for_frame(clip, frame)
        foot = (
            pose_foot_pixel(pose, estimator=foot_estimator)
            if pose and foot_estimator != "sided_box"
            else None
        )
        pose_court = _project_pixel(homography, foot)
        if pose_court is None:
            court = (float(track["court_x"]), float(track["court_y"]))
            foot = (
                float(track["root_x_native"]),
                float(track["root_y_native"]),
            )
            position_source = "sided_box_bottom_center"
            court_sigma = BOX_FOOT_SIGMA_M
        else:
            court = pose_court
            position_source = f"pose_{foot_estimator}"
            court_sigma = POSE_FOOT_SIGMA_M

        hip = pose_hip_pixel(pose) if pose else None
        hip_height = (
            vertical_height_from_pixel(projection, court, hip)
            if projection is not None and hip is not None
            else None
        )
        contact = contacts.get((clip, frame, side))
        racket = None if contact is None else contact.get("racket_face_native")
        racket_pixel = (
            None
            if not isinstance(racket, list) or len(racket) != 2
            else (float(racket[0]), float(racket[1]))
        )
        racket_court = _project_pixel(homography, racket_pixel)
        racket_sigma_px = None if contact is None else contact.get("racket_face_sigma_px")
        racket_sigma_m = _pixel_sigma_to_court(homography, racket_pixel, racket_sigma_px)
        output.append(
            {
                "clip": clip,
                "frame": frame,
                "t": track.get("t", ""),
                "side": side,
                "name": track.get("name", ""),
                "track_id": track.get("track_id", ""),
                "court_x": round(court[0], 3),
                "court_y": round(court[1], 3),
                "court_sigma_m": round(court_sigma, 3),
                "position_source": position_source,
                "foot_x_native": round(foot[0], 2),
                "foot_y_native": round(foot[1], 2),
                "hip_x_native": "" if hip is None else round(hip[0], 2),
                "hip_y_native": "" if hip is None else round(hip[1], 2),
                "hip_height_proxy_m": (
                    "" if hip_height is None else round(max(0.0, hip_height), 3)
                ),
                "hip_height_sigma_m": "" if hip_height is None else HIP_HEIGHT_SIGMA_M,
                "hip_source": "" if hip is None else "pose_hip_mean_metric_vertical",
                "airborne": "",
                "occluded": "",
                "contact_frame": "" if contact is None else contact.get("frame", ""),
                "racket_face_x_native": "" if racket_pixel is None else round(racket_pixel[0], 2),
                "racket_face_y_native": "" if racket_pixel is None else round(racket_pixel[1], 2),
                "racket_face_court_x": "" if racket_court is None else round(racket_court[0], 3),
                "racket_face_court_y": "" if racket_court is None else round(racket_court[1], 3),
                "racket_face_sigma_px": "" if racket_sigma_px is None else racket_sigma_px,
                "racket_face_sigma_m": "" if racket_sigma_m is None else round(racket_sigma_m, 3),
                "racket_face_source": ""
                if contact is None
                else contact.get("racket_face_source", ""),
                "stroke_type": "" if contact is None else contact.get("stroke_type", ""),
                "stroke_status": "" if contact is None else contact.get("stroke_status", ""),
            }
        )
    return sorted(output, key=lambda row: (row["clip"], row["frame"], row["side"]))


def write_player_state_artifact(
    path: Path,
    rows: list[dict],
    *,
    sources: dict[str, Path | None],
    fps: float,
    foot_estimator: str,
    gaps: list[dict] | None = None,
    automatic: bool = True,
    labels_or_reviewed_inputs: list[str] | None = None,
    parent_provenance: dict | None = None,
) -> None:
    """Write ``tennis_player_state_v1`` CSV and its fail-closed sidecar."""
    reviewed_paths = [Path(value) for value in labels_or_reviewed_inputs or []]
    if automatic and reviewed_paths:
        raise ValueError("automatic player state rejects labels or reviewed inputs")
    if automatic and parent_provenance is None:
        raise ValueError("automatic player state requires parent automatic provenance")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_STATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    receipts = {
        name: (
            None
            if source is None or not source.is_file()
            else {"path": str(source), "sha256": file_sha256(source)}
        )
        for name, source in sources.items()
    }
    reused = [
        file_record(source, role=name)
        for name, source in sources.items()
        if source is not None and source.is_file()
    ]
    provenance = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=AUTOMATIC_MODE if automatic else DIAGNOSTIC_MODE,
        source_videos=[] if parent_provenance is None else parent_provenance["source_videos"],
        models=[] if parent_provenance is None else parent_provenance["models"],
        configuration={
            "stage": PLAYER_STATE_SCHEMA,
            "fps": fps,
            "foot_estimator": foot_estimator,
            "labels_loaded": bool(reviewed_paths),
        },
        reused_artifacts=reused,
        fallbacks=[] if parent_provenance is None else parent_provenance["fallbacks"],
        reviewed_inputs=[file_record(label, role="player_truth") for label in reviewed_paths],
    )
    provenance_path = path.with_suffix(path.suffix + ".provenance.json")
    write_provenance(provenance_path, provenance)
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        source="player_side_association",
        extra={
            "artifact": path.name,
            "artifact_identity": PLAYER_STATE_SCHEMA,
            "automatic": automatic,
            "labels_or_reviewed_inputs": labels_or_reviewed_inputs or [],
            "fps": fps,
            "frame_index_origin": 1,
            "timestamp_semantics": (
                "copied_from_player_boxes_t_seconds; blank_when_source_timestamp_absent"
            ),
            "court_units": "metres",
            "position_estimator": (
                "sided_box_bottom_center"
                if foot_estimator == "sided_box"
                else f"pose_{foot_estimator}_then_sided_box_fallback"
            ),
            "hip_height_semantics": "single_view_metric_camera_vertical_proxy",
            "racket_face_semantics": "2d_pose_elbow_wrist_extrapolation_not_face_normal",
            "consumer_contract": {
                "fitter_athlete_arm": {
                    "position": ["court_x", "court_y", "court_sigma_m"],
                    "vertical_proxy": ["hip_height_proxy_m", "hip_height_sigma_m"],
                    "contact_racket": [
                        "racket_face_x_native",
                        "racket_face_y_native",
                        "racket_face_sigma_px",
                    ],
                    "rule": "treat_blank_as_abstention_and_never_ground_an_airborne_player",
                },
                "viewer_3d": {
                    "join": ["clip", "frame", "side"],
                    "clock": "t_seconds_when_present_else_native_frame_with_declared_fps",
                    "rule": "render_position_sigma_and_contact-only_racket_observations",
                },
            },
            "airborne_status_available": False,
            "occlusion_status_available": False,
            "sigmas": {
                "pose_foot_m": POSE_FOOT_SIGMA_M,
                "box_foot_m": BOX_FOOT_SIGMA_M,
                "hip_height_m": HIP_HEIGHT_SIGMA_M,
            },
            "source_receipts": receipts,
            "provenance_manifest": {
                "path": str(provenance_path),
                "sha256": file_sha256(provenance_path),
            },
            "gaps": gaps or [],
            "output_rows": len(rows),
            "abstained": not rows,
            "abstention_reasons": [] if rows else ["no_player_tracks"],
        },
    )


def track_gaps(tracked: dict[str, dict[str, Track]], frame_counts: dict[str, int]) -> list[dict]:
    """Explicit per-side gap spans, so an absent frame is a stated hole not a silence."""
    spans: list[dict] = []
    for clip, sides in sorted(tracked.items()):
        last = frame_counts.get(clip)
        for side, track in sorted(sides.items()):
            first_frame = 1
            last_frame = last or track.detections[-1].frame
            for start, end in gaps(track, first_frame=first_frame, last_frame=last_frame):
                spans.append({"clip": clip, "side": side, "start": start, "end": end})
    return spans


def track_boundary_records(tracked: dict[str, dict[str, Track]]) -> list[dict]:
    return [
        {"clip": clip, "side": side, **record}
        for clip, sides in sorted(tracked.items())
        for side, track in sorted(sides.items())
        for record in track.stitch_boundaries
    ]


def write_track_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_TRACKS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


KEEP_UNIQUE_SIDECAR_KEY = "keep_unique_admissible_frames"


def recorded_keep_unique_admissible_frames(path: Path) -> bool | None:
    """Return the association flag recorded on a sided-box artifact, or None.

    ``None`` means ``path`` is not a sided player-box CSV (derived pose, a test
    fixture, a missing file). Pre-flag sided boxes with no sidecar key are
    unflagged (``False``), never treated as the production default.
    """
    sidecar = path.with_name(path.name + ".coordinates.json")
    space = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if KEEP_UNIQUE_SIDECAR_KEY in space:
        value = space[KEEP_UNIQUE_SIDECAR_KEY]
        if not isinstance(value, bool):
            raise ValueError(f"{KEEP_UNIQUE_SIDECAR_KEY} must be bool, got {value!r}")
        return value
    identity = space.get("artifact_identity")
    sided_name = path.name.startswith("player_boxes_") and "native_sided" in path.name
    if identity == res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY or (
        sided_name and (path.is_file() or sidecar.is_file())
    ):
        return False
    return None


def require_keep_unique_admissible_frames(path: Path, *, expected: bool | None = None) -> None:
    """Refuse a sided-box CSV whose recorded flag does not match the production default.

    Rollback of the default is ``SelectionConfig.keep_unique_admissible_frames = False``;
    pass ``expected`` to assert a specific arm. Non-sided artifacts are ignored.
    """
    if expected is None:
        expected = SelectionConfig().keep_unique_admissible_frames
    recorded = recorded_keep_unique_admissible_frames(path)
    if recorded is None:
        return
    if bool(recorded) != bool(expected):
        how = (
            "player_side_association --keep-unique-admissible-frames"
            if expected
            else "player_side_association --no-keep-unique-admissible-frames"
        )
        raise ValueError(
            f"player boxes {KEEP_UNIQUE_SIDECAR_KEY}={recorded!r} does not match "
            f"production default {expected}; regenerate sided boxes with {how} "
            f"(path={path})"
        )


def write_propagated_sided_boxes(
    *,
    unsided: Path,
    court_point: Path,
    court_frame: Path,
    fps: float,
    clip: str,
    output: Path,
    frames_directory: Path,
    camera_projections: dict[int, np.ndarray] | None = None,
) -> Path:
    """Associate one clip from unsided boxes using same-shot homographies.

    ``camera_projections`` (frame -> native-pixel P) switches the body-size gate to the
    camera's standing height (``camera_body_scale``); None keeps the ground scale.
    """
    from cv.pipeline.shot_segments import clip_propagation_shots

    with unsided.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("clip") == clip]
    image_size, artifact_size = coordinate_sizes(unsided)
    shot_ids = {clip: clip_propagation_shots(frames_directory, fps)}
    detections = build_detections(
        rows,
        load_homographies(court_point),
        image_size=image_size,
        artifact_size=artifact_size,
        frame_homographies=load_frame_homographies(
            court_frame, propagate_shot=True, shot_ids=shot_ids
        )
        or {},
    )
    if camera_projections is not None:
        detections = camera_body_scale(detections, camera_projections)
    sided = sided_rows_from_tracks(track_clips(detections, fps=fps))
    write_rows(output, sided, include_native=bool(rows and "x0_native" in rows[0]))
    res.propagate_coordinate_manifest(
        [unsided],
        output,
        extra={
            "artifact": output.name,
            "artifact_identity": res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
            "source_fps": float(fps),
            "input_rows": len(rows),
            "output_rows": len(sided),
            "court_geometry_mode": "propagated_shot_homography",
            "shot_homography_propagation": True,
            KEEP_UNIQUE_SIDECAR_KEY: True,
            **({"player_body_scale": "camera"} if camera_projections is not None else {}),
            "clip": clip,
            "abstained": not sided,
        },
    )
    return output


def write_rows(path: Path, rows: list[dict], *, include_native: bool = False) -> None:
    fieldnames = [
        "clip",
        "frame",
        "side",
        "x0",
        "y0",
        "x1",
        "y1",
        "conf",
        "court_x",
        "court_y",
        "track_id",
        *(
            ["x0_native", "y0_native", "x1_native", "y1_native"]
            if include_native or (rows and "x0_native" in rows[0])
            else []
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


CONTACT_LEDGER_FIELDS = [
    "clip",
    "frame",
    "emitted_frame",
    "side",
    "incumbent_reach",
    "resolved_reach",
    "confidence",
    "action",
]


def contact_observations(path: Path, match_id: str | None = None) -> list[dict]:
    """Contact emissions as (clip, frame, pixel) rows the resolver can consume.

    A cohort-wide emissions file repeats clip names across broadcasts, so ``match_id`` must be
    given whenever the file covers more than the match being associated.
    """
    payload = json.loads(path.read_text())
    emissions = payload["emissions"] if isinstance(payload, dict) else payload
    rows = []
    for emission in emissions:
        if emission.get("event_type") != "contact" or emission.get("abstain"):
            continue
        if match_id is not None and emission.get("match_id") != match_id:
            continue
        location = emission.get("location") or {}
        if location.get("image_x") is None or location.get("image_y") is None:
            continue
        rows.append(
            {
                "clip": str(emission["clip"]).split("__")[-1],
                "frame": float(emission["frame"]),
                "image_x": float(location["image_x"]),
                "image_y": float(location["image_y"]),
            }
        )
    return rows


def load_projections(path: Path | None):
    """Per-clip and per-point camera matrices; empty when no camera artifact is given."""
    if path is None or not path.exists():
        return {}, {}
    per_frame: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path, allow_pickle=True) as data:
        for clip, frame, projection in zip(data["clips"], data["frames"], data["P"], strict=True):
            matrix = np.asarray(projection, float)
            if np.isfinite(matrix).all():
                per_frame.setdefault(str(clip), {})[int(frame)] = matrix
    per_point: dict[int, np.ndarray] = {}
    point_path = path.parent / "camera_P_per_point.npz"
    if point_path.exists():
        with np.load(point_path, allow_pickle=True) as data:
            for point, projection in zip(data["pts"], data["P"], strict=True):
                matrix = np.asarray(projection, float)
                if np.isfinite(matrix).all():
                    per_point[int(point)] = matrix
    return per_frame, per_point


def write_contact_ledger(path: Path, ledger: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_LEDGER_FIELDS)
        writer.writeheader()
        for entry in ledger:
            writer.writerow({key: entry.get(key, "") for key in CONTACT_LEDGER_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--court", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tracker",
        choices=("greedy", "bytetrack"),
        default="bytetrack",
        help="box linker: greedy (player_court_v2) or bytetrack (player_tracker)",
    )
    parser.add_argument(
        "--overlap-fragment-recovery",
        action="store_true",
        help="default-off: recover one admissible measured player fragment from consistent "
        "original overlap within one reliable camera span",
    )
    parser.add_argument(
        "--lost-revival-body-history",
        action="store_true",
        help="default-off: at lost-track revival only, refuse an identity whose measured "
        "body-size history is an established different size class from the incoming "
        "detection, reusing the existing selection size band and min_hits (bytetrack only)",
    )
    parser.add_argument(
        "--keep-unique-admissible-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="production default-on: after winner-take-all, keep unique frames of other "
        "admissible same-side tracks (a deep-behind-baseline prefix of the same player "
        "that lost the whole-clip score). Line judges still fail the depth/size gates. "
        "Rollback: --no-keep-unique-admissible-frames.",
    )
    parser.add_argument(
        "--tracks-output",
        type=Path,
        default=None,
        help="also write player_tracks_native_v1.csv here (bytetrack only)",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=None,
        help="automatic pose CSV used by --player-state-output",
    )
    parser.add_argument(
        "--contact-strikers",
        type=Path,
        default=None,
        help="contact_striker_resolution_v1 JSON used by --player-state-output",
    )
    parser.add_argument(
        "--player-state-output",
        type=Path,
        default=None,
        help="write the automatic tennis_player_state_v1 per-frame artifact",
    )
    parser.add_argument(
        "--player-state-foot-estimator",
        choices=("sided_box", "ankle_mean", "lower_ankle"),
        default="sided_box",
    )
    parser.add_argument(
        "--parent-provenance",
        type=Path,
        default=None,
        help="required automatic pipeline_provenance_v1 inherited by --player-state-output",
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=None,
        help="player_identity_v1.json; stamps the per-point names onto --tracks-output",
    )
    parser.add_argument(
        "--court-frame-track",
        type=Path,
        default=None,
        help="court_H_per_frame_v1.npz; defaults to the sibling of --court",
    )
    parser.add_argument(
        "--propagate-shot-homography",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="default-off: project boxes with the nearest reliable homography of "
        "the same camera shot when court registration marked the frame unreliable. "
        "A static wide shot stores that camera as anchor_static_fallback. A "
        "close-up is a different shot and is not painted. Requires --frames-root. "
        "Rollback: omit the flag or pass --no-propagate-shot-homography.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="native frame directory whose clip subdirectories are f_*.jpg; "
        "required with --propagate-shot-homography",
    )
    parser.add_argument(
        "--reliable-view-scope",
        action="store_true",
        help="require explicit reliable per-frame court input and restart association "
        "across every unsupported span; never bridge identities across held views",
    )
    parser.add_argument(
        "--bilateral-overlap-recovery",
        action="store_true",
        help="within --reliable-view-scope, fill gaps only from one admissible fragment "
        "agreeing with the original track at both observed boundaries",
    )
    parser.add_argument(
        "--sided-boxes",
        type=Path,
        default=None,
        help="an existing sided artifact to start from instead of re-associating --boxes",
    )
    parser.add_argument(
        "--contact-events",
        type=Path,
        default=None,
        help="event emissions JSON; rewrites the sided box at each contact frame with the "
        "per-contact striker resolver",
    )
    parser.add_argument(
        "--contact-match-id",
        default=None,
        help="restrict --contact-events to one broadcast; required for a cohort-wide file",
    )
    parser.add_argument(
        "--contact-ledger",
        type=Path,
        default=None,
        help="CSV of one resolver decision per contact and side",
    )
    parser.add_argument(
        "--camera",
        type=Path,
        default=None,
        help="camera_P_per_frame_v1.npz; the resolver's body-scale gate is off without it",
    )
    args = parser.parse_args()
    if args.overlap_fragment_recovery and (
        args.tracker != "bytetrack"
        or args.sided_boxes is not None
        or args.contact_events is not None
        or args.court_frame_track is None
    ):
        parser.error(
            "--overlap-fragment-recovery requires fresh bytetrack association "
            "and explicit --court-frame-track"
        )
    if args.lost_revival_body_history and args.tracker != "bytetrack":
        parser.error("--lost-revival-body-history requires --tracker bytetrack")
    if args.lost_revival_body_history and args.sided_boxes is not None:
        parser.error("--lost-revival-body-history requires original detections, not --sided-boxes")
    if args.bilateral_overlap_recovery and not args.reliable_view_scope:
        parser.error("--bilateral-overlap-recovery requires --reliable-view-scope")
    if args.reliable_view_scope and (
        args.court_frame_track is None
        or args.tracker != "bytetrack"
        or args.sided_boxes is not None
        or args.contact_events is not None
    ):
        parser.error(
            "--reliable-view-scope requires explicit --court-frame-track, bytetrack, "
            "and fresh association without --sided-boxes/--contact-events"
        )
    with args.boxes.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_size, artifact_size = coordinate_sizes(args.boxes)
    homographies = load_homographies(args.court)
    frame_track_path = args.court_frame_track or (args.court.parent / "court_H_per_frame_v1.npz")
    if args.court_frame_track is not None and not frame_track_path.is_file():
        raise FileNotFoundError(f"explicit court frame track missing: {frame_track_path}")
    shot_ids = None
    if args.propagate_shot_homography:
        if args.frames_root is None or not args.frames_root.is_dir():
            parser.error("--propagate-shot-homography requires --frames-root")
        from cv.pipeline.shot_segments import propagation_shot_ids

        shot_ids = {}
        for clip_dir in sorted(path for path in args.frames_root.iterdir() if path.is_dir()):
            paths = sorted(clip_dir.glob("f_*.jpg"))
            if len(paths) < 2:
                continue
            shot_ids[clip_dir.name] = propagation_shot_ids(paths, args.fps)
    frame_homographies = load_frame_homographies(
        frame_track_path,
        propagate_shot=args.propagate_shot_homography,
        shot_ids=shot_ids,
    )

    tracker_config = TrackerConfig(lost_revival_body_history=args.lost_revival_body_history)
    selection_config = SelectionConfig(
        keep_unique_admissible_frames=args.keep_unique_admissible_frames
    )
    revival_audit = RevivalBodyHistoryAudit(enabled=args.lost_revival_body_history)
    overlap_audit: list[dict] = []

    def build_tracks():
        detections = build_detections(
            rows,
            homographies,
            image_size=image_size,
            artifact_size=artifact_size,
            frame_homographies=frame_homographies,
        )
        if args.reliable_view_scope:
            assert frame_homographies is not None
            return track_reliable_views(
                detections,
                frame_homographies,
                fps=args.fps,
                bilateral_overlap=args.bilateral_overlap_recovery,
                overlap_recovery=args.overlap_fragment_recovery,
                overlap_audit=overlap_audit,
                tracker_config=tracker_config,
                selection_config=selection_config,
                audit=revival_audit,
            )
        return track_clips(
            detections,
            fps=args.fps,
            tracker_config=tracker_config,
            selection_config=selection_config,
            audit=revival_audit,
            overlap_recovery=args.overlap_fragment_recovery,
            recovery_frame_homographies=frame_homographies,
            overlap_audit=overlap_audit,
        )

    tracked = None
    if args.sided_boxes is not None:
        with args.sided_boxes.open(newline="") as handle:
            output = list(csv.DictReader(handle))
    elif args.tracker == "bytetrack":
        tracked = build_tracks()
        output = sided_rows_from_tracks(tracked)
    else:
        output = associate_rows(
            rows,
            homographies,
            fps=args.fps,
            image_size=image_size,
            artifact_size=artifact_size,
            frame_homographies=frame_homographies,
            tracker=args.tracker,
        )
    resolved_contacts = 0
    if args.contact_events is not None:
        contacts = contact_observations(args.contact_events, args.contact_match_id)
        projections, point_projections = load_projections(args.camera)
        output, ledger = patch_contact_strikers(
            output,
            rows,
            contacts,
            homographies,
            fps=args.fps,
            image_size=image_size,
            artifact_size=artifact_size,
            frame_homographies=frame_homographies,
            projections=projections,
            point_projections=point_projections,
        )
        resolved_contacts = sum(1 for entry in ledger if entry["action"] != "none")
        if args.contact_ledger is not None:
            write_contact_ledger(args.contact_ledger, ledger)
    include_native = bool(rows and "x0_native" in rows[0])
    write_rows(args.output, output, include_native=include_native)
    exported_tracks = None
    names = {}
    if (
        args.tracks_output is not None or args.player_state_output is not None
    ) and args.tracker == "bytetrack":
        if tracked is None:
            tracked = build_tracks()
        if args.identity is not None:
            payload = json.loads(args.identity.read_text())
            names = names_by_clip(payload)
            if not names:
                print(f"identity witnesses disagree, names withheld: {args.identity}")
        exported_tracks = track_rows(tracked, names=names)
    if args.tracks_output is not None and exported_tracks is not None:
        write_track_rows(args.tracks_output, exported_tracks)
        res.write_coordinate_manifest(
            res.coordinate_manifest_path(args.tracks_output),
            image_size=image_size,
            artifact_size=image_size,
            source="player_side_association_bytetrack",
            extra={
                "artifact": args.tracks_output.name,
                "artifact_identity": "player_tracks_native_v1",
                "coordinate_columns": {"x": "root_x_native", "y": "root_y_native"},
                "court_units": "metres",
                "fps": args.fps,
                "frame_index_origin": 1,
                "position_estimator": "box_bottom_center_ground_projection",
                "identity_scope": "anchored_match_names" if names else "point_local_side_track",
                "airborne_status_available": False,
                "occlusion_status_available": False,
                "source_track_selection": "unpatched_bytetrack",
                **(
                    {
                        "overlap_fragment_recovery": True,
                        "overlap_fragment_scope_source": file_record(frame_track_path),
                        "overlap_fragment_recovery_audit": overlap_audit,
                    }
                    if args.overlap_fragment_recovery
                    else {}
                ),
                "lost_revival_body_history": args.lost_revival_body_history,
                KEEP_UNIQUE_SIDECAR_KEY: args.keep_unique_admissible_frames,
                **(
                    {"lost_revival_body_history_audit": revival_audit.summary()}
                    if args.lost_revival_body_history
                    else {}
                ),
                "court_geometry_mode": (
                    "reliable_per_frame"
                    if frame_homographies is not None
                    else "legacy_point_static"
                ),
                "output_rows": len(exported_tracks),
                "stitch_boundary_reconciliation": track_boundary_records(tracked),
                "abstained": not exported_tracks,
            },
        )
    if args.player_state_output is not None:
        if args.tracker != "bytetrack" or exported_tracks is None:
            raise ValueError("--player-state-output requires --tracker bytetrack")
        if args.pose is None:
            raise ValueError("--player-state-output requires --pose")
        if args.parent_provenance is None:
            raise ValueError("--player-state-output requires --parent-provenance")
        parent_provenance = load_provenance(args.parent_provenance, require_automatic=True)
        pose_rows = load_native_pose_rows(str(args.pose))
        contact_rows = []
        if args.contact_strikers is not None:
            contact_payload = json.loads(args.contact_strikers.read_text())
            if contact_payload.get("schema") != "contact_striker_resolution_v1":
                raise ValueError("--contact-strikers has an unsupported schema")
            contact_rows = list(contact_payload.get("strikers") or [])
        projections, point_projections = load_projections(args.camera)

        def state_homography(clip: str, frame: int):
            point = int(clip.removeprefix("pt"))
            matrix = (frame_homographies or {}).get(clip, {}).get(frame)
            if matrix is None and frame_homographies is None:
                matrix = homographies.get(point)
            return (
                None
                if matrix is None
                else res.image_to_world_homography(matrix, image_size, res.NATIVE_SIZE)
            )

        def state_projection(clip: str, frame: int):
            return projections.get(clip, {}).get(
                frame, point_projections.get(int(clip.removeprefix("pt")))
            )

        state_rows = build_player_state_rows(
            exported_tracks,
            pose_rows,
            homography_for_frame=state_homography,
            projection_for_frame=state_projection,
            contact_strikers=contact_rows,
            foot_estimator=args.player_state_foot_estimator,
        )
        write_player_state_artifact(
            args.player_state_output,
            state_rows,
            sources={
                "player_boxes": args.boxes,
                "court": frame_track_path if frame_track_path.is_file() else args.court,
                "camera": args.camera,
                "pose": args.pose,
                "contact_strikers": args.contact_strikers,
                "identity": args.identity,
                "parent_provenance": args.parent_provenance,
            },
            fps=args.fps,
            foot_estimator=args.player_state_foot_estimator,
            gaps=track_gaps(tracked, {}),
            parent_provenance=parent_provenance,
        )
    res.propagate_coordinate_manifest(
        [args.boxes],
        args.output,
        extra={
            "artifact": args.output.name,
            "artifact_identity": res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
            "source_fps": args.fps,
            "input_rows": len(rows),
            "output_rows": len(output),
            "court_homographies": len(homographies),
            "court_frame_track_clips": len(frame_homographies or {}),
            "court_geometry_mode": (
                "reliable_per_frame" if frame_homographies is not None else "legacy_point_static"
            ),
            **(
                {
                    "association_scope": "contiguous_reliable_camera_spans",
                    "association_scope_source": file_record(frame_track_path),
                    "bilateral_overlap_recovery": args.bilateral_overlap_recovery,
                }
                if args.reliable_view_scope
                else {}
            ),
            "linker": args.tracker if args.sided_boxes is None else "sided_boxes_input",
            **(
                {
                    "overlap_fragment_recovery": True,
                    "overlap_fragment_scope_source": file_record(frame_track_path),
                    "overlap_fragment_recovery_audit": overlap_audit,
                }
                if args.overlap_fragment_recovery
                else {}
            ),
            "lost_revival_body_history": args.lost_revival_body_history,
            KEEP_UNIQUE_SIDECAR_KEY: args.keep_unique_admissible_frames,
            "shot_homography_propagation": args.propagate_shot_homography,
            **(
                {"lost_revival_body_history_audit": revival_audit.summary()}
                if args.lost_revival_body_history
                else {}
            ),
            "contact_striker_resolver": args.contact_events is not None,
            "contacts_resolved": resolved_contacts,
            "stitch_boundary_reconciliation": track_boundary_records(tracked or {}),
            "box_coordinate_space": (
                "native" if rows and "x0_native" in rows[0] else "legacy_artifact"
            ),
            "abstained": not output,
            "abstention_reasons": (
                []
                if output
                else [
                    (
                        "no_player_detections"
                        if not rows
                        else (
                            "no_valid_court_homographies"
                            if not homographies
                            else "no_admissible_player_tracklets"
                        )
                    )
                ]
            ),
        },
    )
    print(f"associated {len(output)} near/far player rows -> {args.output}")


if __name__ == "__main__":
    main()
