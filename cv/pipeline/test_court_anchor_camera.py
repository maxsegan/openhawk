from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv.pipeline import court_anchor_camera as anchor
from cv.pipeline.shot_segments import ShotSegmentation


def _paths(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for frame in range(1, count + 1):
        path = tmp_path / f"f_{frame:04d}.jpg"
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _segmentation(frames: list[int], shot_id: list[int], play: list[bool]) -> ShotSegmentation:
    sampled = np.asarray(frames)
    ids = np.asarray(shot_id)
    flags = np.asarray(play)
    shots = []
    for shot in range(int(ids.max()) + 1):
        members = np.flatnonzero(ids == shot)
        shots.append(
            {
                "shot_id": shot,
                "start_frame": int(sampled[members[0]]),
                "end_frame": int(sampled[members[-1]]),
                "is_play_camera": bool(flags[members[0]]),
            }
        )
    return ShotSegmentation(sampled, ids, flags, np.zeros(len(sampled)), shots)


def test_agreement_returns_the_same_segmentation(tmp_path, monkeypatch):
    paths = _paths(tmp_path, 40)
    frames = list(range(1, 41, 4))
    seg = _segmentation(frames, [0] * 5 + [1] * 5, [True] * 5 + [False] * 5)
    monkeypatch.setattr(anchor, "_features", lambda path: (_ for _ in ()).throw(AssertionError))
    out, receipt = anchor.reanchor(
        seg, paths, paths[4], fps=10.0, stride=4, minimum_shot_seconds=0.48
    )
    assert out is seg
    assert receipt["applied"] is False and receipt["reason"] == "shot_gate_agrees"


def test_missing_anchor_is_left_alone(tmp_path):
    paths = _paths(tmp_path, 40)
    seg = _segmentation(list(range(1, 41, 4)), [0] * 10, [False] * 10)
    out, receipt = anchor.reanchor(seg, paths, None, fps=10.0, stride=4, minimum_shot_seconds=0.48)
    assert out is seg and receipt["reason"] == "no_anchor"


def test_disagreement_reclassifies_and_splits_a_missed_cut(tmp_path, monkeypatch):
    # One uncut "shot": a close-up (frames 1-37) then the wide court (41-77). The
    # longest-shot reference called all of it non-play; the anchor sits in the wide part.
    paths = _paths(tmp_path, 80)
    frames = list(range(1, 81, 4))
    seg = _segmentation(frames, [0] * 20, [False] * 20)
    monkeypatch.setattr(anchor, "_features", lambda path: "reference")
    monkeypatch.setattr(
        anchor,
        "_registration_inliers",
        lambda reference, path: 900.0 if int(path.stem[2:]) >= 41 else 120.0,
    )
    out, receipt = anchor.reanchor(
        seg, paths, paths[60], fps=10.0, stride=4, minimum_shot_seconds=0.48
    )
    assert receipt["applied"] is True
    assert [(s["start_frame"], s["end_frame"], s["is_play_camera"]) for s in out.shots] == [
        (1, 37, False),
        (41, 77, True),
    ]
    assert out.is_play_camera.tolist() == [False] * 10 + [True] * 10


def test_anchor_frame_name_reads_only_a_direct_fit(tmp_path):
    rows = [
        {"point": 1, "status": "direct", "frame": "f_0550.jpg"},
        {"point": 2, "status": "abstained", "frame": None},
    ]
    (tmp_path / anchor.EVIDENCE_NAME).write_text(json.dumps({"rows": rows}))
    assert anchor.anchor_frame_name(tmp_path, "pt0001") == "f_0550.jpg"
    assert anchor.anchor_frame_name(tmp_path, "pt0002") is None
    assert anchor.anchor_frame_name(tmp_path, "pt0003") is None
