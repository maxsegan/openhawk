from __future__ import annotations

import json

import cv2
import numpy as np

from pose import associate_frame_rows, coverage_summary, valid_keypoint


def _image_to_court_homography() -> np.ndarray:
    source = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
    target = np.float32([[0, 0], [10, 0], [0, 30], [10, 30]])
    return cv2.getPerspectiveTransform(source, target)


def _row(y1: float, confidence: float, x: float = 50.0) -> dict:
    row = {
        "clip": "pt0001",
        "frame": "f_0001.jpg",
        "x0": x - 5,
        "y0": y1 - 20,
        "x1": x + 5,
        "y1": y1,
        "conf": confidence,
    }
    for name in (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ):
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y1 - 10
        row[f"{name}_confidence"] = 0.9
    return row


def test_association_selects_highest_confidence_player_per_side() -> None:
    rows = [_row(20, 0.7), _row(25, 0.9), _row(70, 0.8)]

    selected = associate_frame_rows(rows, _image_to_court_homography())

    assert selected["near"]["conf"] == 0.9
    assert selected["far"]["conf"] == 0.8


def test_association_rejects_people_beyond_audited_baseline_margin() -> None:
    selected = associate_frame_rows(
        [_row(20, 0.8), _row(70, 0.8), _row(120, 0.99)],
        _image_to_court_homography(),
    )

    assert set(selected) == {"near", "far"}


def test_arm_coverage_requires_confident_nonzero_keypoints() -> None:
    near, far = _row(20, 0.8), _row(70, 0.8)
    near["side"], far["side"] = "near", "far"
    near["left_wrist_confidence"] = 0.1

    assert not valid_keypoint(near, "left_wrist")
    assert coverage_summary([near, far], total_frames=1) == {
        "total_frames": 1,
        "frames_with_near": 1,
        "frames_with_far": 1,
        "frames_with_both": 1,
        "frames_with_near_arm": 0,
        "frames_with_far_arm": 1,
        "frames_with_both_arms": 0,
    }


def test_native_is_the_default_pose_coordinate_space() -> None:
    import resolution as res
    from pose import result_rows

    class _Values:
        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._values, dtype=float)

        def __getitem__(self, index):
            return _Values(self._values[index])

        def __len__(self):
            return len(self._values)

        def __float__(self):
            return float(self._values)

    class _Boxes:
        def __init__(self):
            self.xyxy = _Values([[100.0, 200.0, 160.0, 380.0]])
            self.conf = _Values([0.9])

        def __len__(self):
            return 1

    class _Keypoints:
        def __init__(self):
            self.data = _Values([[[130.0, 260.0, 0.95]] * 17])

    class _Result:
        orig_shape = (1080, 1920)
        boxes = _Boxes()
        keypoints = _Keypoints()

    rows = result_rows("/tmp/pt0001/f_0001.jpg", _Result())

    assert res.NATIVE_SIZE == res.FrameSize(1920, 1080)
    assert (rows[0]["x0"], rows[0]["y0"]) == (100.0, 200.0)
    assert (rows[0]["nose_x"], rows[0]["nose_y"]) == (130.0, 260.0)


def test_a_declared_half_native_pose_row_is_read_back_at_the_known_native_pixel(tmp_path):
    """The sidecar, not the file name, decides the space a keypoint is in."""
    import csv

    import resolution as res
    from pose import fieldnames, load_native_pose_rows

    native_wrist = (1204.0, 733.0)
    path = tmp_path / "player_pose_native_looking_name_v1.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        row = dict.fromkeys(fieldnames(), 0.0)
        row.update(
            clip="pt0001",
            frame="f_0001.jpg",
            x0=native_wrist[0] / 2 - 20,
            y0=native_wrist[1] / 2 - 60,
            x1=native_wrist[0] / 2 + 20,
            y1=native_wrist[1] / 2 + 30,
            conf=0.9,
            right_wrist_x=native_wrist[0] / 2,
            right_wrist_y=native_wrist[1] / 2,
            right_wrist_confidence=0.9,
        )
        writer.writerow(row)
    # Written by hand on purpose: this reproduces a *shipped* artifact that is named native
    # and declares 960x540.  `write_coordinate_manifest` now refuses to emit that pair, so the
    # only way to still prove the reader honours the sidecar is to forge the stale sidecar.
    res.coordinate_manifest_path(path).write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "artifact": path.name,
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 960, "height": 540},
                "source": "test",
                "subnative_flagged": True,
                "resolution_status": res.LEGACY_BAD_SHOULD_UPDATE,
                "subnative_justification": "stale half-native pose mirror under a native name",
            }
        )
    )

    (recovered,) = load_native_pose_rows(str(path))

    assert (recovered["right_wrist_x"], recovered["right_wrist_y"]) == native_wrist


def test_pose_rows_project_a_known_keypoint_to_its_known_court_position(tmp_path):
    """A keypoint at a known court point survives the artifact space it was written in."""
    import resolution as res
    from pose import associate_rows

    # A 1920x1080 image-to-court homography with the near baseline low in the frame.
    image = np.float32([[420, 980], [1500, 980], [700, 470], [1220, 470]])
    court = np.float32([[0.0, 0.0], [10.97, 0.0], [0.0, 23.77], [10.97, 23.77]])
    homography = cv2.getPerspectiveTransform(image, court)
    known_court = (5.485, 3.0)
    inverse = np.linalg.inv(homography)
    native_root = cv2.perspectiveTransform(
        np.float32([[[known_court[0], known_court[1]]]]), inverse
    )[0, 0]

    def _row(root, size):
        scale = np.array([size.width / 1920.0, size.height / 1080.0])
        x, y = np.asarray(root, dtype=float) * scale
        return {
            "clip": "pt0001",
            "frame": "f_0001.jpg",
            "x0": x - 30 * scale[0],
            "y0": y - 180 * scale[1],
            "x1": x + 30 * scale[0],
            "y1": y,
            "conf": 0.9,
        }

    native = associate_rows(
        [_row(native_root, res.NATIVE_SIZE)],
        {1: homography},
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
    )
    half = associate_rows(
        [_row(native_root, res.LEGACY_TRACKING_SIZE)],
        {1: homography},
        image_size=res.NATIVE_SIZE,
        artifact_size=res.LEGACY_TRACKING_SIZE,
    )

    for associated in (native, half):
        assert associated[0]["side"] == "near"
        assert abs(associated[0]["court_x"] - known_court[0]) < 0.05
        assert abs(associated[0]["court_y"] - known_court[1]) < 0.05


def test_racket_face_extends_the_selected_forearm_by_the_shipped_scale() -> None:
    from pose import RACKET_FACE_FOREARM_SCALE, racket_face_from_pose

    row = {
        "right_elbow_x": 10.0,
        "right_elbow_y": 20.0,
        "right_elbow_confidence": 0.9,
        "right_wrist_x": 20.0,
        "right_wrist_y": 30.0,
        "right_wrist_confidence": 0.8,
    }

    face = racket_face_from_pose(row, hand="right")

    assert RACKET_FACE_FOREARM_SCALE == 1.5
    assert (face["x"], face["y"]) == (35.0, 45.0)
    assert face["keypoint_confidence"] == 0.8


def test_metric_vertical_height_recovers_a_known_hip_height() -> None:
    from pose import vertical_height_from_pixel

    projection = np.array(
        [[100.0, 0.0, 10.0, 0.0], [0.0, 100.0, -50.0, 0.0], [0.0, 0.0, 1.0, 10.0]]
    )
    court = (5.2, 4.0)
    height = 1.05
    homogeneous = projection @ np.array([court[0], court[1], height, 1.0])
    pixel = homogeneous[:2] / homogeneous[2]

    recovered = vertical_height_from_pixel(projection, court, tuple(pixel))

    assert recovered is not None
    assert abs(recovered - height) < 1e-9
