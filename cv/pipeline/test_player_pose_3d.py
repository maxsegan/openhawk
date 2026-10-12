import json
from argparse import Namespace

import numpy as np
import pytest

from cv.pipeline import player_pose_3d, player_pose_portal, pose_lift_batched
from cv.pipeline.pose import KEYPOINT_NAMES


def broadcast_projection(vertical_error: float = 1.0) -> np.ndarray:
    """A pinhole camera 30 m behind the near baseline, 12 m up, looking at the court."""
    centre = np.array([5.485, -30.0, 12.0])
    target = np.array([5.485, 11.885, 0.0])
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    K = np.array([[2200.0, 0, 960], [0, 2200.0, 540], [0, 0, 1]])
    P = K @ np.concatenate([rotation, -(rotation @ centre)[:, None]], axis=1)
    # The production fault: a wrong vertical scale, the court plane untouched.
    P[:, 2] *= vertical_error
    return P


def standing_pose(stature: float, root) -> np.ndarray:
    template = pose_lift_batched._template(stature)
    return template + np.array([root[0], root[1], 0.0])


def project(P, points):
    h = (P @ np.concatenate([points, np.ones((len(points), 1))], axis=1).T).T
    return h[:, :2] / h[:, 2:3]


@pytest.mark.parametrize("side,root", [("near", (4.0, 1.0)), ("far", (6.0, 23.0))])
def test_batched_lift_recovers_upright_body_under_vertical_camera_error(side, root):
    torch = pytest.importorskip("torch")
    del torch
    stature = 1.85
    truth = standing_pose(stature, root)
    true_camera = broadcast_projection()
    wrong_camera = broadcast_projection(vertical_error=0.8)
    n = 12
    keypoints = np.repeat(project(true_camera, truth)[None], n, axis=0)
    confidences = np.full((n, 17), 0.9)
    pixels = keypoints[0]
    box = [
        pixels[:, 0].min() - 5,
        pixels[:, 1].min() - 10,
        pixels[:, 0].max() + 5,
        pixels[:, 1].max() + 3,
    ]
    out = pose_lift_batched.lift(
        keypoints,
        confidences,
        np.repeat([box], n, axis=0),
        np.repeat([root], n, axis=0).astype(float),
        np.repeat(wrong_camera[None], n, axis=0),
        np.zeros(n, dtype=int),
        np.arange(n),
        np.full(n, 1.0 if side == "near" else -1.0),
        stature,
        fps=50.0,
        iterations=300,
        device="cpu",
    )
    joints = out["joints"][n // 2]
    index = {name: k for k, name in enumerate(KEYPOINT_NAMES)}
    hips = joints[[index["left_hip"], index["right_hip"]]].mean(0)
    shoulders = joints[[index["left_shoulder"], index["right_shoulder"]]].mean(0)
    ankles = joints[[index["left_ankle"], index["right_ankle"]]]
    # Upright, on the court at the root, and the right height.
    assert np.linalg.norm(shoulders[:2] - hips[:2]) < 0.2
    assert np.all(np.abs(ankles[:, 2] - 0.08) < 0.08)
    assert np.linalg.norm(ankles[:, :2].mean(0) - np.asarray(root)) < 0.3
    assert abs(joints[index["nose"], 2] - truth[index["nose"], 2]) < 0.15
    # The camera's vertical error is found (the true factor relative to the wrong one).
    assert abs(out["vertical_scale"][0] - 1 / 0.8) < 0.15
    assert np.nanmedian(out["reprojection_px"]) < 3.0


def test_joint_status_interpolates_short_gaps_only_inside_a_run():
    n = 20
    confidences = np.full((n, 17), 0.9)
    confidences[3:5, 0] = 0.0  # 2-frame gap: interpolated
    confidences[8:18, 1] = 0.0  # 10-frame gap: missing
    confidences[12, 2] = 0.3  # low confidence
    confidences[0:2, 3] = 0.0  # leading gap: never extrapolated
    liftable = np.ones(n, dtype=bool)
    status = player_pose_3d.joint_status(
        confidences, liftable, np.zeros(n, dtype=int), np.arange(n), max_gap=8
    )
    S = player_pose_3d.STATUS
    assert list(status[3:5, 0]) == [S["interpolated"]] * 2
    assert set(status[8:18, 1]) == {S["missing"]}
    assert status[12, 2] == S["lifted_low_confidence"]
    assert list(status[0:2, 3]) == [S["missing"]] * 2
    # A frame gap or a track change ends the run.
    frames = np.arange(n)
    frames[4:] += 1
    status = player_pose_3d.joint_status(confidences, liftable, np.zeros(n, dtype=int), frames, 8)
    assert status[3, 0] == S["missing"]


def test_document_layers_and_fill(tmp_path):
    portal = tmp_path / "portal"
    (portal / "data").mkdir(parents=True)
    document = {
        "source_clip": "pt0001",
        "frame_range": [2, 3],
        "fps": 50.0,
        "native_timebase": {"fps": 50.0, "origin_t": 10.0, "origin_frame": 1},
        "players": {"names": {"near": "A", "far": None}},
        "skeletons": {"near": [], "far": []},
    }
    (portal / "data" / "doc__pt0001.json").write_text(json.dumps(document))
    (portal / "data" / "index.json").write_text("{}")
    joints = np.full((4, 17, 3), np.nan, dtype=np.float32)
    joints[:, 0] = [1.0, 2.0, 1.6]
    status = np.zeros((4, 17), dtype=np.uint8)
    status[:, 0] = 1
    status[3, 0] = 0
    npz = tmp_path / "pose.npz"
    np.savez(
        npz,
        joint_names=np.asarray(KEYPOINT_NAMES),
        clip=np.asarray(["pt0001", "pt0001", "pt0001", "pt0002"]),
        frame=np.asarray([1, 2, 3, 2]),
        side=np.asarray([0, 0, 0, 0]),
        track_id=np.asarray(["1"] * 4),
        box_conf=np.full(4, 0.9),
        root_xy=np.full((4, 2), [1.0, 2.0]),
        joints_3d=joints,
        joint_status=status,
    )
    assert player_pose_3d.document_windows(player_pose_3d.portal_documents(portal)) == {
        "pt0001": (2, 3)
    }
    report = player_pose_portal.fill(portal, npz)
    assert report["filled"] == 1 and report["layers"] == 1 and report["skeleton_rows"] == 2
    filled = json.loads((portal / "data" / "doc__pt0001.json").read_text())
    assert filled["pose_layer"]["file"] == "pose/pt0001.json.gz"
    assert filled["skeleton_format"]["joint_names"][0] == "nose"
    assert filled["players"]["names"]["near"] == "A"
    import gzip

    layer = json.loads(gzip.decompress((portal / "data" / "pose" / "pt0001.json.gz").read_bytes()))
    assert layer["players"]["near"]["frame"] == [2, 3]
    assert layer["players"]["near"]["t"] == [10.02, 10.04]
    row = layer["skeletons"]["near"][0]
    assert row["j"][0] == [0, 0, 160, 1] and row["j"][1] is None


def test_product_pose_step_switch_off(tmp_path, monkeypatch):
    from cv.pipeline import product_runner

    monkeypatch.setenv(player_pose_3d.ENV, "off")
    args = Namespace(out=tmp_path, match_id="m", gpu=0, gpu_lease=None)
    assert product_runner.step_pose(args) == {"skipped": "TENNIS_PLAYER_POSE_3D=off"}
    monkeypatch.setenv(player_pose_3d.ENV, "on")
    assert product_runner.step_pose(args) == {"skipped": "no staged portal"}
    (tmp_path / "portal" / "3d" / "data").mkdir(parents=True)
    assert product_runner.step_pose(args) == {"skipped": "native frames already compacted"}
    assert product_runner.STEPS.index("pose") == product_runner.STEPS.index("compact") - 1
