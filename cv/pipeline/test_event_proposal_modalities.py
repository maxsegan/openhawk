"""Independent proposal generation through real native-coordinate input readers."""

import csv
import json

import pytest

from cv.pipeline import event_proposals as proposals, resolution


def pose_fixture(tmp_path, *, native=False):
    root = tmp_path / "automatic"
    match = root / "broadcast"
    match.mkdir(parents=True)
    (match / "audit_frames_native_1080.coordinates.json").write_text(json.dumps({"fps": 25.0}))
    pose_root = tmp_path / "poses"
    pose_match = pose_root / "broadcast"
    pose_match.mkdir(parents=True)
    path = pose_match / proposals.POSE_NAMES[0]
    scale = 1 if native else 0.5
    rows = [
        dict(
            clip="pt0002",
            side="near",
            frame=f"f_{frame:04d}.jpg",
            right_wrist_x=x * scale,
            right_wrist_y=y * scale,
            right_wrist_confidence=1,
            right_shoulder_y=70 * scale,
            right_shoulder_confidence=1,
        )
        for frame, x, y in [(9, 10, 80), (10, 30, 40), (11, 10, 80)]
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    resolution.coordinate_manifest_path(path).write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "artifact": path.name,
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {
                    "width": 1920 if native else 960,
                    "height": 1080 if native else 540,
                },
                "source": "test",
                "subnative_flagged": not native,
            }
        )
    )
    return root, match, pose_root, path


@pytest.mark.parametrize("track_present", [False, True])
def test_pose_clip_proposes_without_ball_observations(tmp_path, track_present):
    root, match, poses, pose = pose_fixture(tmp_path)
    if track_present:
        track = match / "track.csv"
        track.write_text("clip,frame,x,y\npt0001,f_0001.jpg,100,200\n")
        resolution.coordinate_manifest_path(track).write_text(
            json.dumps(
                {
                    "schema": "tennis.coordinate-space.v1",
                    "artifact": track.name,
                    "image_size": {"width": 1920, "height": 1080},
                    "artifact_size": {"width": 1920, "height": 1080},
                    "source": "test",
                    "coordinate_columns": {"x": "x", "y": "y"},
                }
            )
        )
    document = proposals.generate_proposal_document(
        root, tmp_path / "out.json", track_name="track.csv", pose_root=poses
    )
    rows = document["proposals"]
    assert len(rows) == 1
    assert rows[0]["source"] == "pose_swing"
    assert rows[0]["clip"] == "broadcast__pt0002"
    assert rows[0]["frame"] == 10
    assert rows[0]["evidence"]["predicted_x"] == 30
    assert rows[0]["evidence"]["predicted_y"] == 40
    assert rows[0]["kinds"] == ["contact"]
    assert document["labels_or_reviewed_inputs"] == []
    assert str(resolution.coordinate_manifest_path(pose).resolve()) in document["inputs"]
    assert (
        str((match / "audit_frames_native_1080.coordinates.json").resolve()) in document["inputs"]
    )


def test_completely_missing_modalities_emit_no_events_or_fabricated_clock(tmp_path):
    root = tmp_path / "automatic"
    (root / "empty").mkdir(parents=True)
    document = proposals.generate_proposal_document(
        root, tmp_path / "out.json", track_name="missing.csv"
    )
    assert document["proposals"] == []
    assert document["inputs"] == []


def test_pose_without_declared_pixel_space_still_refuses(tmp_path):
    root, _, poses, path = pose_fixture(tmp_path)
    resolution.coordinate_manifest_path(path).unlink()
    with pytest.raises(FileNotFoundError, match="coordinate sidecar"):
        proposals.generate_proposal_document(
            root, tmp_path / "out.json", track_name="missing.csv", pose_root=poses
        )


def test_pose_only_still_requires_source_cadence(tmp_path):
    root, match, poses, pose = pose_fixture(tmp_path)
    (match / "audit_frames_native_1080.coordinates.json").unlink()
    with pytest.raises(ValueError, match="source fps"):
        proposals.generate_proposal_document(
            root, tmp_path / "out.json", track_name="missing.csv", pose_root=poses
        )
