"""Re-anchor the play-camera test on the frame the court was registered on.

``shot_segments.segment_shots`` registers every shot to the clip's longest shot
and calls a shot the play camera when it keeps enough ORB inliers. The longest
shot is usually the wide court view, but not always: when a player close-up, a
crowd shot or a replay runs longer, that shot becomes the reference, it is called
play camera, and the wide rally is refused (``no_play_camera_shot``) or the
active span lands on the close-up.

The per-point court fit already names one native frame on which the court passed
its topology and surface witnesses (``court_topology_evidence_v1.json``, status
``direct``). That frame is a picture of the court by construction. When it falls
on a sample the shot gate called *not* play camera, the two automatic stages
disagree about which camera is the court camera, and this module lets the court
anchor decide: every sampled frame is registered to the anchor, and the cut
segments are split where the class changes (a missed cut between a close-up and
the wide shot is common). Where they agree, the segmentation is returned
unchanged, so clips the shot gate already reads correctly keep their verdicts.

``ANCHOR_MINIMUM_INLIERS`` is higher than the shot gate's 60, because the score
overlay alone keeps 60-150 inliers across a cut. On the twelve opened-panel clips
where the anchor disagrees with the shot gate, frames that are not the court camera
kept at most 177 inliers against the anchor and court frames at least 390; 250 sits
in that gap. Across all clips the same camera can fall to 150-300 under zoom, which
is why the test runs only where the two stages disagree
(``cv/experiments/active_span/FINAL_REPORT_active_span.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv.pipeline.shot_segments import (
    ShotSegmentation,
    _features,
    _registration_inliers,
    frame_number,
    merge_short_runs,
)

SCHEMA = "court_anchor_camera_v1"
EVIDENCE_NAME = "court_topology_evidence_v1.json"
ANCHOR_MINIMUM_INLIERS = 250.0


def anchor_frame_name(match_dir: Path, clip: str) -> str | None:
    """The court fit's anchor picture for one clip, or None when it abstained."""

    path = match_dir / EVIDENCE_NAME
    if not path.is_file():
        return None
    try:
        number = int(clip.removeprefix("pt"))
    except ValueError:
        return None
    for row in json.loads(path.read_text()).get("rows", []):
        if int(row.get("point", -1)) == number and row.get("status") == "direct":
            frame = row.get("frame")
            return str(frame) if frame else None
    return None


def reanchor(
    segmentation: ShotSegmentation,
    frame_paths: list[Path],
    anchor_path: Path | None,
    fps: float,
    stride: int,
    minimum_shot_seconds: float,
) -> tuple[ShotSegmentation, dict]:
    """Return the segmentation the court anchor supports, and a receipt."""

    receipt: dict = {"schema": SCHEMA, "anchor": None, "applied": False}
    sampled = segmentation.sampled_frames
    if anchor_path is None or not anchor_path.is_file() or len(sampled) < 2:
        receipt["reason"] = "no_anchor"
        return segmentation, receipt
    anchor = frame_number(anchor_path)
    receipt["anchor"] = anchor
    nearest = int(np.argmin(np.abs(sampled - anchor)))
    if abs(int(sampled[nearest]) - anchor) > stride or segmentation.is_play_camera[nearest]:
        receipt["reason"] = "shot_gate_agrees"
        return segmentation, receipt

    by_frame = {frame_number(path): path for path in frame_paths}
    reference = _features(anchor_path)
    registration = np.asarray(
        [_registration_inliers(reference, by_frame[int(frame)]) for frame in sampled], dtype=float
    )
    classes = registration >= ANCHOR_MINIMUM_INLIERS
    change = np.zeros(len(sampled), dtype=bool)
    change[1:] = (np.diff(segmentation.shot_id) != 0) | (classes[1:] != classes[:-1])
    shot_id = np.cumsum(change)
    minimum_samples = max(2, round(minimum_shot_seconds * fps / stride))
    shot_id = merge_short_runs(shot_id, minimum_samples)
    shots = []
    play = np.zeros(len(sampled), dtype=bool)
    for shot in range(int(shot_id.max()) + 1):
        members = np.flatnonzero(shot_id == shot)
        score = float(np.median(registration[members]))
        is_play = bool(score >= ANCHOR_MINIMUM_INLIERS)
        play[members] = is_play
        shots.append(
            {
                "shot_id": shot,
                "start_frame": int(sampled[members[0]]),
                "end_frame": int(sampled[members[-1]]),
                "sampled_frames": int(len(members)),
                "registration_inliers": score,
                "is_play_camera": is_play,
                "reference": "court_anchor",
            }
        )
    receipt.update(
        applied=True,
        reason="shot_gate_disagrees",
        minimum_inliers=ANCHOR_MINIMUM_INLIERS,
        original_shots=segmentation.shots,
    )
    return (
        ShotSegmentation(
            sampled, shot_id, play, registration, shots, segmentation.boundary_continuity
        ),
        receipt,
    )
