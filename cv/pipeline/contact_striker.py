"""Which player hit the ball, resolved per emitted contact.

The whole-point tracker answers "where is the near player through this point".  A contact
asks a narrower question -- "which body did the racket that produced this pixel belong to,
and which end of the court is that body playing from" -- and the two answers are not the
same object.  ``player_side_association.resolve_contact_striker`` already answers the first
half at one frame given a side; this module answers the second half, choosing the striker's
end from the image evidence instead of inheriting it from a track label or a court-metre
sign test at the net.

Three things decide an end here.

*Reach* is the distance from the contact pixel to a person box's edge in units of that
box's height, so it is scale-free across near and far court.  *Continuity and body scale*
are the same gates the single-frame resolver uses: a body must be about as tall as a
standing person at its own court position, and must be somewhere the point track could
have reached at a run.  *Alternation* is a soft prior over the whole contact sequence: a
rally alternates ends, so a non-alternating pair pays ``SAME_END_PENALTY``.  It is a prior
and not a rule, because a missed emission genuinely produces two consecutive contacts at
one end, and it only flips a decision when the two ends' reaches are within that penalty of
each other.

Every coordinate that enters is native 1920x1080.  Box artifacts that declare a smaller
space in their sidecar are converted by the caller (``box_scale``); nothing here infers a
space from a file name.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.camera_cal import COURT_L, COURT_W, NET_Y
from cv.pipeline.pose import (
    load_native_pose_rows,
    pose_hip_pixel,
    racket_face_from_pose,
)
from cv.pipeline.provenance import file_sha256
from cv.pipeline.player_side_association import (
    BASELINE_MARGIN_M,
    CONTACT_CONTINUITY_SLACK_M,
    CONTACT_CONTINUITY_V_MAX_MS,
    CONTACT_MIN_CONFIDENCE,
    CONTACT_REACH_BAND,
    CONTACT_REACH_LIMIT,
    CONTACT_SCALE_BAND,
    CONTACT_WINDOW_SECONDS,
    COURT_X_MARGIN_M,
    _continuity_penalty,
    _suppress_partial_boxes,
    contact_reach,
    standing_height_px,
)

ENDS = ("near", "far")

# A same-end pair costs this much in the sequence decode. It is measured in the same units
# as reach (box heights), so the prior can only overturn an end whose reach advantage is
# smaller than it. Set just above the reach band so a body that is clearly the striker
# (inside the band) still wins against alternation, while a marginal one does not.
SAME_END_PENALTY = 0.60
# The cost charged to an end with no admissible body at all. Above CONTACT_REACH_LIMIT so a
# real body always beats an absence, and finite so a missed detection does not force the
# whole sequence to one end.
MISSING_END_COST = 1.50
# How far from the contact frame a sided-track row may sit and still say which end a body
# is continuous with.
END_TRACK_WINDOW_SECONDS = 0.60
# The body-scale reference. The automatic per-frame camera reproduces the court plane
# exactly but its height column is not metrically calibrated -- on the September cohort it
# makes a player at the near baseline the same pixel height as one at the far baseline, which
# is geometrically impossible and rejects the real near-court striker. So the standing-height
# reference is measured, not projected: the same end's own tracked boxes around the contact.
# The projected ratio is still reported, as a diagnostic only.
SCALE_REFERENCE_WINDOW_SECONDS = 0.60
# Keypoints allowed to stand in for the racket hand.
WRIST_KEYPOINTS = ("left_wrist", "right_wrist")
MIN_WRIST_CONFIDENCE = 0.30
SCHEMA = "contact_striker_resolution_v1"
# Calibrated on opened-development player/racket truth by
# ``cv.validation.player_truth_ledger``. It is deliberately conservative and
# remains an observation sigma, not a calibrated probability interval.
RACKET_FACE_SIGMA_PX = 75.0
# Opened-development gates. Coarse stroke type is optional metadata, so false
# type claims cost more than abstention. A serve/overhead contact must be above
# the detected body and a groundstroke must clear the largest wrong-side arm
# margin seen in the dense truth ledger.
STROKE_ABOVE_BODY_RATIO = -0.05
STROKE_LATERAL_MARGIN = 0.16


@dataclass
class Candidate:
    """One admissible body at one frame, with everything the decode needs about it."""

    frame: int
    box: tuple[float, float, float, float]
    court: np.ndarray
    reach: float
    continuity_penalty: float
    scale_ratio: float | None
    scale_reference: str
    camera_scale_ratio: float | None
    end: str
    end_source: str
    row: dict = field(repr=False, default_factory=dict)

    @property
    def cost(self) -> float:
        return self.reach + self.continuity_penalty


def _box_in_native(row: dict, scale: float) -> tuple[float, float, float, float]:
    if all(
        f"{axis}_native" in row and row[f"{axis}_native"] != "" for axis in ("x0", "y0", "x1", "y1")
    ):
        return tuple(float(row[f"{axis}_native"]) for axis in ("x0", "y0", "x1", "y1"))
    return tuple(scale * float(row[axis]) for axis in ("x0", "y0", "x1", "y1"))


def end_of_body(
    court: np.ndarray,
    frame: float,
    track_by_end: dict[str, dict[int, np.ndarray]],
    *,
    fps: float,
) -> tuple[str, str]:
    """The end a body is playing from, and what decided it.

    A player at the net is metres past the net line in court coordinates for a few frames,
    so the sign of ``court_y`` is not the end.  Prefer the end whose point track the body is
    continuous with; fall back to the sign only when no track can claim it.
    """
    window = max(1, int(round(END_TRACK_WINDOW_SECONDS * fps)))
    excess: dict[str, float] = {}
    for end in ENDS:
        rows = track_by_end.get(end) or {}
        best = None
        for track_frame, track_court in rows.items():
            if abs(track_frame - frame) > window:
                continue
            seconds = abs(track_frame - frame) / max(fps, 1e-6)
            budget = CONTACT_CONTINUITY_V_MAX_MS * seconds + CONTACT_CONTINUITY_SLACK_M
            gap = float(np.linalg.norm(court - track_court)) - budget
            best = gap if best is None else min(best, gap)
        if best is not None:
            excess[end] = best
    reachable = {end: value for end, value in excess.items() if value <= 0.0}
    if len(reachable) == 1:
        return next(iter(reachable)), "point_track"
    if reachable:
        return min(reachable, key=reachable.__getitem__), "point_track_nearest"
    if excess:
        return min(excess, key=excess.__getitem__), "point_track_nearest"
    return ("near" if court[1] < NET_Y else "far"), "court_side_fallback"


def measured_standing_heights(
    heights_by_end: dict[str, dict[int, float]],
    frame: float,
    *,
    fps: float,
) -> dict[str, float]:
    """Each end's own tracked box height around ``frame``, as the body-scale reference."""
    window = max(1, int(round(SCALE_REFERENCE_WINDOW_SECONDS * fps)))
    reference = {}
    for end in ENDS:
        local = [
            height
            for track_frame, height in (heights_by_end.get(end) or {}).items()
            if abs(track_frame - frame) <= window and height > 1.0
        ]
        if local:
            reference[end] = float(np.median(local))
    return reference


def candidates_at_contact(
    pixel: tuple[float, float],
    frame: float,
    detections: dict[int, Sequence[dict]],
    *,
    homography_for_frame: Callable[[int], np.ndarray | None],
    projection_for_frame: Callable[[int], np.ndarray | None],
    track_by_end: dict[str, dict[int, np.ndarray]],
    heights_by_end: dict[str, dict[int, float]] | None = None,
    fps: float,
    box_scale: float,
) -> list[Candidate]:
    """Every body near ``pixel`` that passes the court, scale and reach gates."""
    window = max(1, int(np.ceil(CONTACT_WINDOW_SECONDS * fps)))
    centre = int(round(frame))
    anchors = [
        (track_frame, court)
        for end in ENDS
        for track_frame, court in (track_by_end.get(end) or {}).items()
        if window < abs(track_frame - frame) <= 3 * window
    ]
    reference = measured_standing_heights(heights_by_end or {}, frame, fps=fps)
    out: list[Candidate] = []
    for offset in range(-window, window + 1):
        candidate_frame = centre + offset
        homography = homography_for_frame(candidate_frame)
        if homography is None:
            continue
        projection = projection_for_frame(candidate_frame)
        raw = []
        for row in detections.get(candidate_frame, []):
            box = _box_in_native(row, box_scale)
            foot = np.float32([[[0.5 * (box[0] + box[2]), box[3]]]])
            court = np.asarray(_perspective(foot, homography), dtype=float)
            if not np.isfinite(court).all():
                continue
            if not (
                -COURT_X_MARGIN_M <= court[0] <= COURT_W + COURT_X_MARGIN_M
                and -BASELINE_MARGIN_M <= court[1] <= COURT_L + BASELINE_MARGIN_M
            ):
                continue
            raw.append(
                {
                    "row": row,
                    "box": box,
                    "court": court,
                    "area": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
                }
            )
        for entry in _suppress_partial_boxes(raw):
            reach = contact_reach(pixel, entry["box"])
            if reach > CONTACT_REACH_LIMIT:
                continue
            end, end_source = end_of_body(entry["court"], frame, track_by_end, fps=fps)
            height = entry["box"][3] - entry["box"][1]
            projected = standing_height_px(projection, entry["court"])
            camera_ratio = None if projected is None or projected <= 1.0 else height / projected
            ratio, scale_reference = None, "absent"
            if end in reference:
                ratio, scale_reference = height / reference[end], "tracked_end_height"
            elif camera_ratio is not None:
                ratio, scale_reference = camera_ratio, "projected_standing_height"
            if ratio is not None and not (CONTACT_SCALE_BAND[0] <= ratio <= CONTACT_SCALE_BAND[1]):
                continue
            out.append(
                Candidate(
                    frame=candidate_frame,
                    box=entry["box"],
                    court=entry["court"],
                    reach=reach,
                    continuity_penalty=_continuity_penalty(entry["court"], frame, anchors, fps),
                    scale_ratio=ratio,
                    scale_reference=scale_reference,
                    camera_scale_ratio=camera_ratio,
                    end=end,
                    end_source=end_source,
                    row=entry["row"],
                )
            )
    return out


def _perspective(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float32))[0, 0]


def decode_ends(costs: list[dict[str, float]]) -> list[str]:
    """Least-cost end sequence under the soft alternation prior."""
    if not costs:
        return []
    total = {end: costs[0].get(end, MISSING_END_COST) for end in ENDS}
    back: list[dict[str, str]] = []
    for step in costs[1:]:
        pointers: dict[str, str] = {}
        updated: dict[str, float] = {}
        for end in ENDS:
            options = {
                previous: total[previous] + (0.0 if previous != end else SAME_END_PENALTY)
                for previous in ENDS
            }
            best_previous = min(options, key=options.__getitem__)
            pointers[end] = best_previous
            updated[end] = options[best_previous] + step.get(end, MISSING_END_COST)
        back.append(pointers)
        total = updated
    end = min(total, key=total.__getitem__)
    sequence = [end]
    for pointers in reversed(back):
        end = pointers[end]
        sequence.append(end)
    return list(reversed(sequence))


def _wrist_reach(
    pixel: tuple[float, float],
    box: tuple[float, float, float, float],
    pose_rows: Sequence[dict],
) -> dict:
    """Distance from the striker's nearer wrist to the contact pixel, in native pixels."""
    height = max(1.0, box[3] - box[1])
    root = (0.5 * (box[0] + box[2]), box[3])
    result = {
        "root_px": float(np.hypot(pixel[0] - root[0], pixel[1] - root[1])),
        "wrist_px": None,
        "wrist_keypoint": None,
        "wrist_box_heights": None,
        "pose_matched": False,
        "racket_face_native": None,
        "racket_face_sigma_px": None,
        "racket_face_source": None,
        "striking_hand": None,
    }
    result["root_box_heights"] = result["root_px"] / height
    best_overlap, matched = 0.0, None
    for row in pose_rows:
        try:
            other = tuple(float(row[axis]) for axis in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        overlap = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
            0.0, min(box[3], other[3]) - max(box[1], other[1])
        )
        if overlap > best_overlap:
            best_overlap, matched = overlap, row
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    if matched is None or best_overlap < 0.3 * area:
        return result
    result["pose_matched"] = True
    for name in WRIST_KEYPOINTS:
        try:
            confidence = float(matched[f"{name}_confidence"])
            x, y = float(matched[f"{name}_x"]), float(matched[f"{name}_y"])
        except (KeyError, TypeError, ValueError):
            continue
        if confidence < MIN_WRIST_CONFIDENCE or (x <= 0 and y <= 0):
            continue
        distance = float(np.hypot(pixel[0] - x, pixel[1] - y))
        if result["wrist_px"] is None or distance < result["wrist_px"]:
            result["wrist_px"] = distance
            result["wrist_keypoint"] = name
    if result["wrist_px"] is not None:
        result["wrist_box_heights"] = result["wrist_px"] / height
    face = racket_face_from_pose(matched, contact_pixel=pixel)
    if face is not None:
        result.update(
            racket_face_native=[round(face["x"], 2), round(face["y"], 2)],
            racket_face_sigma_px=RACKET_FACE_SIGMA_PX,
            racket_face_source="pose_elbow_wrist_extrapolation_v1",
            striking_hand=face["hand"],
            racket_face_keypoint_confidence=round(face["keypoint_confidence"], 4),
            racket_face_contact_distance_px=round(face["contact_distance_px"], 2),
        )
    return result


def infer_stroke_type(
    pixel: tuple[float, float],
    candidate: Candidate,
    pose_rows: Sequence[dict],
    *,
    contact_index: int,
) -> dict:
    """Conservative coarse stroke type from body/contact geometry.

    The classifier abstains without a matched pose. It distinguishes the
    high-contact serve/overhead cases first, then net volleys, and only calls
    forehand/backhand when the contact lies more clearly on the selected
    racket arm's shoulder side than the opposite shoulder.
    """
    box = candidate.box
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    matched, best_overlap = None, 0.0
    for row in pose_rows:
        try:
            other = tuple(float(row[axis]) for axis in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        overlap = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
            0.0, min(box[3], other[3]) - max(box[1], other[1])
        )
        if overlap > best_overlap:
            matched, best_overlap = row, overlap
    if matched is None or best_overlap < 0.3 * area:
        return {"stroke_type": None, "stroke_status": "abstain_no_pose"}

    height = max(1.0, box[3] - box[1])
    shoulders = []
    for side in ("left", "right"):
        try:
            confidence = float(matched[f"{side}_shoulder_confidence"])
            point = np.array(
                [float(matched[f"{side}_shoulder_x"]), float(matched[f"{side}_shoulder_y"])],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if confidence >= MIN_WRIST_CONFIDENCE and np.all(point > 0):
            shoulders.append((side, point))
    hip = pose_hip_pixel(matched, min_confidence=MIN_WRIST_CONFIDENCE)
    contact_height = (pixel[1] - box[1]) / height
    high_contact = contact_height <= STROKE_ABOVE_BODY_RATIO
    baseline_distance = (
        abs(float(candidate.court[1]))
        if candidate.end == "near"
        else abs(COURT_L - float(candidate.court[1]))
    )
    evidence = {
        "stroke_contact_height_box": round(contact_height, 4),
        "stroke_baseline_distance_m": round(baseline_distance, 3),
    }
    if high_contact and contact_index == 0 and baseline_distance <= 2.5:
        return {
            "stroke_type": "serve",
            "stroke_status": "resolved_high_baseline_contact",
            **evidence,
        }
    if high_contact and baseline_distance > 2.5:
        return {
            "stroke_type": "overhead",
            "stroke_status": "resolved_overhead_geometry",
            **evidence,
        }
    if abs(float(candidate.court[1]) - NET_Y) <= 3.0:
        return {
            "stroke_type": "volley",
            "stroke_status": "resolved_net_position",
            **evidence,
        }

    face = racket_face_from_pose(matched, contact_pixel=pixel)
    if face is None or len(shoulders) < 2 or hip is None:
        return {"stroke_type": None, "stroke_status": "abstain_arm_geometry", **evidence}
    shoulder_by_side = dict(shoulders)
    hand = face["hand"]
    own = shoulder_by_side.get(hand)
    opposite = shoulder_by_side.get("left" if hand == "right" else "right")
    if own is None or opposite is None:
        return {"stroke_type": None, "stroke_status": "abstain_arm_geometry", **evidence}
    own_distance = float(np.linalg.norm(np.asarray(pixel) - own))
    opposite_distance = float(np.linalg.norm(np.asarray(pixel) - opposite))
    margin = abs(own_distance - opposite_distance) / height
    if margin < STROKE_LATERAL_MARGIN:
        return {
            "stroke_type": None,
            "stroke_status": "abstain_lateral_ambiguity",
            **evidence,
        }
    stroke = "forehand" if own_distance < opposite_distance else "backhand"
    return {
        "stroke_type": stroke,
        "stroke_status": "resolved_shoulder_side",
        "stroke_side_margin_box_heights": round(margin, 4),
        **evidence,
    }


def resolve_point_strikers(
    contacts: Sequence[dict],
    detections: dict[int, Sequence[dict]],
    *,
    homography_for_frame: Callable[[int], np.ndarray | None],
    projection_for_frame: Callable[[int], np.ndarray | None],
    track_by_end: dict[str, dict[int, np.ndarray]] | None = None,
    sided_rows_by_frame: dict[int, Sequence[dict]] | None = None,
    pose_by_frame: dict[int, Sequence[dict]] | None = None,
    fps: float,
    box_scale: float = 1.0,
    sided_box_scale: float | None = None,
) -> list[dict]:
    """One striker decision per contact, with its confidence and its disagreements.

    ``contacts`` are emitted contacts ordered in time, each ``{"frame", "image_x",
    "image_y"}`` in native pixels.  ``detections`` are raw person rows by frame in the box
    artifact's own space; ``box_scale`` takes that space to native.  ``track_by_end`` is the
    sided track's court positions, used for continuity only.  ``sided_rows_by_frame`` is the
    same track's boxes and is used only to report what the existing artifact would have
    said, never to choose.
    """
    track_by_end = track_by_end or {}
    sided_rows_by_frame = sided_rows_by_frame or {}
    sided_box_scale = box_scale if sided_box_scale is None else sided_box_scale
    heights_by_end: dict[str, dict[int, float]] = {end: {} for end in ENDS}
    for frame, rows in sided_rows_by_frame.items():
        for row in rows:
            end = str(row.get("side") or "")
            if end in heights_by_end:
                box = _box_in_native(row, sided_box_scale)
                heights_by_end[end][frame] = box[3] - box[1]
    pose_by_frame = pose_by_frame or {}
    window = max(1, int(np.ceil(CONTACT_WINDOW_SECONDS * fps)))

    per_contact: list[dict[str, Candidate]] = []
    for contact in contacts:
        pixel = (float(contact["image_x"]), float(contact["image_y"]))
        found = candidates_at_contact(
            pixel,
            float(contact["frame"]),
            detections,
            homography_for_frame=homography_for_frame,
            projection_for_frame=projection_for_frame,
            track_by_end=track_by_end,
            heights_by_end=heights_by_end,
            fps=fps,
            box_scale=box_scale,
        )
        best: dict[str, Candidate] = {}
        for candidate in found:
            incumbent = best.get(candidate.end)
            if incumbent is None or candidate.cost < incumbent.cost:
                best[candidate.end] = candidate
        per_contact.append(best)

    costs = [{end: best[end].cost for end in ENDS if end in best} for best in per_contact]
    decoded = decode_ends(costs)

    results = []
    for contact_index, (contact, best, cost, end) in enumerate(
        zip(contacts, per_contact, costs, decoded, strict=True)
    ):
        pixel = (float(contact["image_x"]), float(contact["image_y"]))
        frame = float(contact["frame"])
        argmin_end = min(cost, key=cost.__getitem__) if cost else None
        chosen = best.get(end)
        other = MISSING_END_COST
        for candidate_end in ENDS:
            if candidate_end != end:
                other = min(other, cost.get(candidate_end, MISSING_END_COST))
        record = {
            "clip": contact.get("clip", ""),
            "frame": frame,
            "image_x": pixel[0],
            "image_y": pixel[1],
            "end": end if chosen is not None else None,
            "status": "resolved",
            "confidence": 0.0,
            "reach": None,
            "reach_end_margin": None,
            "continuity_penalty": None,
            "scale_ratio": None,
            "scale_reference": None,
            "camera_scale_ratio": None,
            "end_source": None,
            "resolved_frame": None,
            "court_x": None,
            "court_y": None,
            "box_native": None,
            "alternation_applied": bool(argmin_end is not None and argmin_end != end),
            "candidate_ends": sorted(best),
        }
        if chosen is None:
            record["status"] = "abstain"
            record["abstain_reason"] = (
                "no_admissible_body" if not best else "decoded_end_has_no_body"
            )
            results.append(record)
            continue
        margin = other - chosen.cost
        confidence = (
            max(0.0, 1.0 - chosen.reach / CONTACT_REACH_BAND)
            * (1.0 - chosen.continuity_penalty)
            * min(1.0, max(0.0, margin) / CONTACT_REACH_BAND)
        )
        record.update(
            reach=round(chosen.reach, 4),
            reach_end_margin=round(margin, 4),
            continuity_penalty=round(chosen.continuity_penalty, 4),
            scale_ratio=None if chosen.scale_ratio is None else round(chosen.scale_ratio, 4),
            scale_reference=chosen.scale_reference,
            camera_scale_ratio=(
                None if chosen.camera_scale_ratio is None else round(chosen.camera_scale_ratio, 4)
            ),
            end_source=chosen.end_source,
            resolved_frame=chosen.frame,
            court_x=round(float(chosen.court[0]), 3),
            court_y=round(float(chosen.court[1]), 3),
            box_native=[round(value, 2) for value in chosen.box],
            confidence=round(float(min(1.0, confidence)), 4),
        )
        record.update(_wrist_reach(pixel, chosen.box, pose_by_frame.get(chosen.frame, [])))
        record.update(
            infer_stroke_type(
                pixel,
                chosen,
                pose_by_frame.get(chosen.frame, []),
                contact_index=contact_index,
            )
        )
        if record["confidence"] < CONTACT_MIN_CONFIDENCE:
            record["status"] = "low_confidence"
        incumbent = _nearest_sided_end(pixel, frame, sided_rows_by_frame, window, sided_box_scale)
        record["sided_track_end"] = incumbent["end"]
        record["sided_track_reach"] = incumbent["reach"]
        record["disagrees_with_sided_track"] = bool(
            incumbent["end"] is not None and incumbent["end"] != record["end"]
        )
        results.append(record)
    return results


def _nearest_sided_end(
    pixel: tuple[float, float],
    frame: float,
    sided_rows_by_frame: dict[int, Sequence[dict]],
    window: int,
    box_scale: float,
) -> dict:
    """What the existing sided artifact would call the striker: its nearest box's side."""
    centre = int(round(frame))
    best: dict = {"end": None, "reach": None}
    for offset in range(-window, window + 1):
        for row in sided_rows_by_frame.get(centre + offset, []):
            reach = contact_reach(pixel, _box_in_native(row, box_scale))
            if best["reach"] is None or reach < best["reach"]:
                best = {"end": str(row.get("side") or ""), "reach": round(reach, 4)}
    return best


# --- Reading one match's automatic artifacts ------------------------------------------------
#
# Every reader below takes the coordinate space from the artifact's own sidecar. Several
# shipped artifacts are named "native" and declare 960x540; the name is not the contract.

BALL_TRACK_NAMES = (
    "ball_track_joint_native1080_arc_augmented_v2.csv",
    "ball_track_joint_native1080_integrity_v1.csv",
    "ball_track_joint_native1080_availability_v1.csv",
)
POSE_NAME = "player_pose_tracked_crop_native_v1.csv"


def frame_number(value) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        raise ValueError(f"no frame index in {value!r}")
    return int(digits)


def declared_sizes(path: Path) -> tuple[res.FrameSize, res.FrameSize]:
    """(image size, coordinate space) of an artifact, from its sidecar."""
    manifest = res.read_coordinate_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"{path} has no coordinate sidecar")
    image = manifest["image_size"]
    artifact = manifest.get("legacy_artifact_size", manifest["artifact_size"])
    return (
        res.FrameSize(int(image["width"]), int(image["height"])),
        res.FrameSize(int(artifact["width"]), int(artifact["height"])),
    )


def clip_rows(path: Path, clip: str) -> list[dict]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("clip") == clip]


def ball_pixels(path: Path, clip: str) -> dict[int, tuple[float, float]]:
    """The automatic ball track for one clip, in native pixels."""
    _image, artifact = declared_sizes(path)
    scale_x = res.NATIVE_SIZE.width / artifact.width
    scale_y = res.NATIVE_SIZE.height / artifact.height
    out: dict[int, tuple[float, float]] = {}
    for row in clip_rows(path, clip):
        try:
            out[frame_number(row["frame"])] = (
                float(row["x"]) * scale_x,
                float(row["y"]) * scale_y,
            )
        except (KeyError, ValueError):
            continue
    return out


def contact_pixel(track: dict[int, tuple[float, float]], frame: float):
    """The ball pixel at a possibly half-frame contact; None when the track has no row."""
    low, high = int(np.floor(frame)), int(np.ceil(frame))
    if low == high:
        return track.get(low)
    if low in track and high in track:
        return tuple(0.5 * (np.array(track[low]) + np.array(track[high])))
    return track.get(high, track.get(low))


def reliable_projections(path: Path, clip: str) -> dict[int, np.ndarray]:
    """Per-frame camera matrices the camera stage itself marked reliable."""
    if not path.exists():
        return {}
    out: dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        reliable = data["reliable"] if "reliable" in data.files else None
        for index, (row_clip, frame, projection) in enumerate(
            zip(data["clips"], data["frames"], data["P"], strict=True)
        ):
            if str(row_clip) != clip:
                continue
            if reliable is not None and not bool(reliable[index]):
                continue
            matrix = np.asarray(projection, float)
            if np.isfinite(matrix).all():
                out[int(frame)] = matrix
    return out


def point_homography(path: Path, clip: str, image_size: res.FrameSize):
    """The clip's court homography in native pixels, or None when S2 abstained."""
    point = int(clip.removeprefix("pt"))
    with np.load(path) as data:
        for candidate, homography in zip(data["pts"], data["H"], strict=True):
            if int(candidate) != point:
                continue
            matrix = np.asarray(homography, float)
            if not np.isfinite(matrix).all():
                return None
            return res.image_to_world_homography(matrix, image_size, res.NATIVE_SIZE)
    return None


@dataclass
class MatchInputs:
    """One match's automatic S2--S5 artifacts, resolved by coordinate contract."""

    match_dir: Path
    boxes: Path
    sided: Path | None
    court: Path
    camera: Path
    ball: Path | None
    pose: Path | None
    image_size: res.FrameSize
    box_scale: float
    fps: float

    @classmethod
    def resolve(cls, match_dir: Path, *, fps: float | None = None) -> "MatchInputs":
        boxes = sorted(match_dir.glob("player_boxes_*_native_v1.csv"))
        if not boxes:
            raise FileNotFoundError(f"no automatic player boxes in {match_dir}")
        sided = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))
        image_size, artifact_size = declared_sizes(boxes[0])
        ball = next(
            (match_dir / name for name in BALL_TRACK_NAMES if (match_dir / name).is_file()),
            None,
        )
        pose = match_dir / POSE_NAME
        stem = boxes[0].stem.split("_")
        try:
            declared_fps = float(stem[2]) if len(stem) > 2 else None
        except ValueError:
            declared_fps = None
        return cls(
            match_dir=match_dir,
            boxes=boxes[0],
            sided=sided[0] if sided else None,
            court=match_dir / "court_H_per_point.npz",
            camera=match_dir / "camera_P_per_frame_v1.npz",
            ball=ball,
            pose=pose if pose.is_file() else None,
            image_size=image_size,
            box_scale=res.NATIVE_SIZE.width / artifact_size.width,
            fps=fps or declared_fps or 25.0,
        )

    def resolve_clip(self, clip: str, contacts: Sequence[dict]) -> list[dict]:
        """Striker decisions for one clip's emitted contacts, with their abstentions."""
        homography = point_homography(self.court, clip, self.image_size)
        if homography is None:
            return [
                _abstained(clip, contact["frame"], "no_automatic_court_geometry")
                for contact in contacts
            ]
        supplied_pixels = all(
            contact.get("image_x") is not None and contact.get("image_y") is not None
            for contact in contacts
        )
        if self.ball is None and not supplied_pixels:
            return [
                _abstained(clip, contact["frame"], "no_automatic_ball_track")
                for contact in contacts
            ]
        track = {} if self.ball is None else ball_pixels(self.ball, clip)
        projections = reliable_projections(self.camera, clip)
        detections: dict[int, list[dict]] = {}
        for row in clip_rows(self.boxes, clip):
            detections.setdefault(frame_number(row["frame"]), []).append(row)
        sided_by_frame: dict[int, list[dict]] = {}
        track_by_end: dict[str, dict[int, np.ndarray]] = {end: {} for end in ENDS}
        sided_scale = self.box_scale
        if self.sided is not None:
            _image, sided_artifact = declared_sizes(self.sided)
            sided_scale = res.NATIVE_SIZE.width / sided_artifact.width
            for row in clip_rows(self.sided, clip):
                frame = frame_number(row["frame"])
                sided_by_frame.setdefault(frame, []).append(row)
                end = str(row.get("side") or "")
                if end in track_by_end:
                    track_by_end[end][frame] = np.array(
                        [float(row["court_x"]), float(row["court_y"])]
                    )
        pose_by_frame: dict[int, list[dict]] = {}
        if self.pose is not None:
            for row in load_native_pose_rows(str(self.pose)):
                if row.get("clip") == clip:
                    pose_by_frame.setdefault(frame_number(row["frame"]), []).append(row)

        located, missing = [], []
        for contact in contacts:
            # A caller that already knows the contact pixel (an evaluation click, a contact
            # localizer) supplies it; otherwise the automatic ball track locates the contact.
            supplied = contact.get("image_x"), contact.get("image_y")
            pixel = (
                supplied
                if supplied[0] is not None and supplied[1] is not None
                else contact_pixel(track, float(contact["frame"]))
            )
            if pixel is None:
                missing.append(float(contact["frame"]))
                continue
            located.append(
                {
                    "clip": clip,
                    "frame": float(contact["frame"]),
                    "image_x": float(pixel[0]),
                    "image_y": float(pixel[1]),
                }
            )
        resolved = resolve_point_strikers(
            located,
            detections,
            homography_for_frame=lambda _frame: homography,
            projection_for_frame=projections.get,
            track_by_end=track_by_end,
            sided_rows_by_frame=sided_by_frame,
            pose_by_frame=pose_by_frame,
            fps=self.fps,
            box_scale=self.box_scale,
            sided_box_scale=sided_scale,
        )
        resolved.extend(
            _abstained(clip, frame, "no_automatic_ball_pixel_at_contact") for frame in missing
        )
        return sorted(resolved, key=lambda row: row["frame"])


def _abstained(clip: str, frame: float, reason: str) -> dict:
    return {
        "clip": clip,
        "frame": float(frame),
        "end": None,
        "status": "abstain",
        "abstain_reason": reason,
        "confidence": 0.0,
    }


def emitted_contacts(path: Path, match_id: str | None) -> dict[str, list[dict]]:
    """Contacts of one broadcast from an event emissions file, keyed by clip."""
    payload = json.loads(path.read_text())
    emissions = payload["emissions"] if isinstance(payload, dict) else payload
    out: dict[str, list[dict]] = {}
    for emission in emissions:
        if emission.get("event_type") != "contact" or emission.get("abstain"):
            continue
        if match_id is not None and emission.get("match_id") not in (None, match_id):
            continue
        clip = str(emission["clip"]).split("__")[-1]
        location = emission.get("location") or {}
        out.setdefault(clip, []).append(
            {
                "frame": float(emission["frame"]),
                "image_x": location.get("image_x"),
                "image_y": location.get("image_y"),
            }
        )
    return {clip: sorted(rows, key=lambda row: row["frame"]) for clip, rows in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--emissions", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = MatchInputs.resolve(args.match_dir, fps=args.fps)
    contacts = emitted_contacts(args.emissions, args.match_id)
    rows: list[dict] = []
    for clip in sorted(contacts):
        rows.extend(inputs.resolve_clip(clip, contacts[clip]))
    sources = {
        name: (
            None
            if path is None or not Path(path).is_file()
            else {"path": str(path), "sha256": file_sha256(Path(path))}
        )
        for name, path in (
            ("player_boxes", inputs.boxes),
            ("player_boxes_sided", inputs.sided),
            ("court_homography", inputs.court),
            ("camera", inputs.camera),
            ("ball_track", inputs.ball),
            ("pose", inputs.pose),
            ("event_emissions", args.emissions),
        )
    }
    payload = {
        "schema": SCHEMA,
        "match_id": args.match_id,
        "fps": inputs.fps,
        "coordinate_space": "native_1920x1080",
        "box_scale_to_native": inputs.box_scale,
        "sources": sources,
        "counts": {
            "contacts": len(rows),
            "resolved": sum(1 for row in rows if row["status"] == "resolved"),
            "low_confidence": sum(1 for row in rows if row["status"] == "low_confidence"),
            "abstained": sum(1 for row in rows if row["status"] == "abstain"),
            "disagrees_with_sided_track": sum(
                1 for row in rows if row.get("disagrees_with_sided_track")
            ),
            "alternation_applied": sum(1 for row in rows if row.get("alternation_applied")),
        },
        "strikers": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"contact strikers {payload['counts']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
