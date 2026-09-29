import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv.pipeline import court
from cv.pipeline.court_topology import NO_TOPOLOGY_SOLUTION, edge_convention_world_transform
from cv.pipeline.court_topology_runner import (
    ARTIFACT_EDGE_CONVENTION,
    AnchorSolution,
    calibrate_clip,
    calibrate_points,
    load_frame_track,
    recover_anchor_from_match_prior,
    register_clip,
    select_anchor,
)


@pytest.mark.parametrize("failure", ["registration", "unreadable"])
def test_registration_never_interpolates_through_a_rejected_sample(tmp_path, monkeypatch, failure):
    frames = [tmp_path / f"f_{index:04d}.jpg" for index in range(11)]
    H = np.eye(3)
    anchor = AnchorSolution(0, frames[0].name, H, "test", 1.0, 1.0, None, {}, {})

    def read(path):
        index = int(Path(path).stem[2:])
        if failure == "unreadable" and index == 5:
            return None
        return np.full((8, 8, 3), index, dtype=np.uint8)

    def transfer(image, *_args):
        if int(image[0, 0, 0]) == 5:
            raise ValueError("different camera view")
        return H, {"inlier_ratio": 1.0, "target_topology_score": 1.0}

    monkeypatch.setattr("cv.pipeline.court_topology_runner.cv2.imread", read)
    monkeypatch.setattr("cv.pipeline.court_topology_runner.transfer_court_homography", transfer)
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._standard_broadcast_view", lambda *_: True
    )
    monkeypatch.setattr("cv.pipeline.court_topology_runner._maximum_topology_score", lambda *_: 0.0)
    _, reliable, source, evidence = register_clip(frames, anchor, stride=5)
    assert reliable == {index: index in {0, 10} for index in range(11)}
    assert all(source[index] == "registration_gap_abstention" for index in range(1, 10))
    assert evidence["unsupported_sample_frames"] == ["f_0005.jpg"]

    if failure == "registration":
        monkeypatch.setattr(
            "cv.pipeline.court_topology_runner._maximum_topology_score", lambda *_: 0.95
        )
        _, recovered, _, recovered_evidence = register_clip(frames, anchor, stride=5)
        assert all(recovered.values())
        assert recovered_evidence["unsupported_sample_frames"] == []


def test_runner_emits_direct_evidence_and_homography(tmp_path):
    frames = tmp_path / "rally_frames" / "pt0001"
    frames.mkdir(parents=True)
    image = np.full((1080, 1920, 3), (70, 125, 170), np.uint8)
    image_corners = np.float32([[330, 890], [1590, 890], [660, 210], [1260, 210]])
    world_corners = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(image_corners, world_corners)
    inverse = np.linalg.inv(homography)
    for start, end in court._court_model():
        pixels = cv2.perspectiveTransform(np.float32([[start, end]]), inverse)[0]
        cv2.line(
            image,
            tuple(np.rint(pixels[0]).astype(int)),
            tuple(np.rint(pixels[1]).astype(int)),
            (250, 250, 250),
            6,
        )
    cv2.imwrite(str(frames / "f_0001.jpg"), image)

    report = calibrate_points(tmp_path, "rally_frames", jobs=2)

    assert report["direct"] == 1
    with np.load(tmp_path / "court_H_per_point.npz") as output:
        assert output["source"].tolist() == ["graded_far_refine:standard"]
        assert np.isfinite(output["H"]).all()
    evidence = json.loads((tmp_path / "court_topology_evidence_v1.json").read_text())
    assert evidence["rows"][0]["fallback_ancestry"] == []
    assert evidence["rows"][0]["graded_band_climb"]["config"]["name"] == "far_k21_r10"


def _synthetic_court_frame(shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A drawn court plus texture, optionally panned by ``shift`` pixels."""
    rng = np.random.default_rng(20260901)
    image = np.full((1080, 1920, 3), (70, 125, 170), np.uint8)
    speckle = rng.integers(0, 40, (1080, 1920, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + speckle, 0, 255).astype(np.uint8)
    image_corners = np.float32(
        [[330 - shift, 890], [1590 - shift, 890], [660 - shift, 210], [1260 - shift, 210]]
    )
    world_corners = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(image_corners, world_corners)
    inverse = np.linalg.inv(homography)
    for start, end in court._court_model():
        pixels = cv2.perspectiveTransform(np.float32([[start, end]]), inverse)[0]
        cv2.line(
            image,
            tuple(np.rint(pixels[0]).astype(int)),
            tuple(np.rint(pixels[1]).astype(int)),
            (250, 250, 250),
            6,
        )
    return image, homography


def test_frame_track_follows_a_pan_and_marks_every_frame(tmp_path):
    frames = tmp_path / "pt0001"
    frames.mkdir(parents=True)
    for index in range(1, 5):
        image, _ = _synthetic_court_frame(shift=(index - 1) * 8)
        cv2.imwrite(str(frames / f"f_{index:04d}.jpg"), image)

    point, row, anchor, track = calibrate_clip(frames, stride=1)

    assert point == 1
    assert row["status"] == "direct"
    assert row["frame_scope"] == "frame_track"
    assert row["frame_track"]["frames"] == 4
    assert track["frames"].tolist() == [1, 2, 3, 4]
    assert len(track["H"]) == len(track["reliable"]) == 4
    # the anchor frame keeps its own solve exactly
    anchor_index = int(row["frame"].removeprefix("f_").removesuffix(".jpg")) - 1
    np.testing.assert_allclose(track["H"][anchor_index], anchor)
    # a panned frame must not reuse the anchor geometry
    moved = [
        index
        for index in range(4)
        if not np.allclose(track["H"][index] / track["H"][index][2, 2], anchor / anchor[2, 2])
    ]
    assert moved


def test_no_frame_track_keeps_the_point_static_artifact(tmp_path):
    frames = tmp_path / "rally_frames" / "pt0001"
    frames.mkdir(parents=True)
    image, _ = _synthetic_court_frame()
    cv2.imwrite(str(frames / "f_0001.jpg"), image)

    report = calibrate_points(tmp_path, "rally_frames", jobs=1, frame_track=False)

    assert report["frame_track_npz"] is None
    assert not (tmp_path / "court_H_per_frame_v1.npz").exists()


def test_frame_track_artifact_round_trips(tmp_path):
    frames = tmp_path / "rally_frames" / "pt0001"
    frames.mkdir(parents=True)
    for index in range(1, 4):
        image, _ = _synthetic_court_frame(shift=(index - 1) * 6)
        cv2.imwrite(str(frames / f"f_{index:04d}.jpg"), image)

    report = calibrate_points(tmp_path, "rally_frames", jobs=1, stride=1)
    track = load_frame_track(Path(report["frame_track_npz"]))

    assert sorted(track["pt0001"]) == [1, 2, 3]
    assert all(np.isfinite(entry[0]).all() for entry in track["pt0001"].values())


def test_anchor_reselection_reaches_fallback_quantiles(tmp_path, monkeypatch):
    frames = [tmp_path / f"f_{index:04d}.jpg" for index in range(11)]
    expected_index = 1

    def solve(index, frame):
        if index != expected_index:
            return {"frame": frame.name, "error": "no solve"}
        return AnchorSolution(
            index=index,
            frame=frame.name,
            homography=np.eye(3),
            source="test",
            topology_score=1.0,
            surface_score=1.0,
            player_score=None,
            refinement_evidence={},
            far_evidence={"accepted": False},
        )

    monkeypatch.setattr("cv.pipeline.court_topology_runner._solve_anchor_frame", solve)

    anchor, attempts = select_anchor(frames)

    assert anchor is not None
    assert anchor.index == expected_index
    assert len(attempts) == 5


def _sun_and_shadow_court_frame() -> tuple[np.ndarray, np.ndarray]:
    """The drawn court with a hard shadow edge along its diagonal."""
    image, homography = _synthetic_court_frame()
    inverse = np.linalg.inv(homography)
    ((near, far),) = cv2.perspectiveTransform(
        np.float32([[[0.0, 0.0], [court.COURT_W, court.COURT_L]]]), inverse
    )
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    side = (grid_x - near[0]) * (far[1] - near[1]) - (grid_y - near[1]) * (far[0] - near[0])
    return (
        np.clip(image * np.where(side > 0, 0.55, 1.0)[..., None], 0, 255).astype(np.uint8),
        homography,
    )


def test_the_surface_witness_policy_recovers_by_default_and_off_still_holds(tmp_path):
    frames = tmp_path / "rally_frames" / "pt0001"
    frames.mkdir(parents=True)
    image, _ = _sun_and_shadow_court_frame()
    cv2.imwrite(str(frames / "f_0001.jpg"), image)

    # Explicitly asking for the strict route still holds the sun-split court.
    held = calibrate_points(tmp_path, "rally_frames", jobs=1, surface_witness_policy="off")
    assert held["direct"] == 0 and held["abstained"] == 1
    assert (
        json.loads((tmp_path / "court_topology_evidence_v1.json").read_text())[
            "surface_witness_policy"
        ]
        == "off"
    )

    # The pipeline default now recovers it, as one extra pass after the strict passes fail.
    report = calibrate_points(tmp_path, "rally_frames", jobs=1)

    assert report["direct"] == 1
    evidence = json.loads((tmp_path / "court_topology_evidence_v1.json").read_text())
    assert evidence["surface_witness_policy"] == "illumination_split"
    assert evidence["illumination_split_recovered"] == 1
    row = evidence["rows"][0]
    assert row["acceptance_route"] == "strong_topology_illumination_split"
    assert row["surface_score"] < 0.50 <= row["illumination_surface_score"]
    assert "illumination_split" in row["source"]
    with np.load(tmp_path / "court_H_per_point.npz") as output:
        assert np.isfinite(output["H"]).all()


def test_the_split_witness_pass_never_replaces_an_ordinary_anchor(tmp_path, monkeypatch):
    frames = [tmp_path / f"f_{index:04d}.jpg" for index in range(11)]
    ordinary_index = 5
    seen = []

    def solve(index, frame, *, proposal_pool=48, surface_witness_policy="off"):
        seen.append((index, surface_witness_policy))
        if index != ordinary_index or surface_witness_policy != "off":
            return {"frame": frame.name, "error": NO_TOPOLOGY_SOLUTION}
        return AnchorSolution(
            index=index,
            frame=frame.name,
            homography=np.eye(3),
            source="test",
            topology_score=1.0,
            surface_score=1.0,
            player_score=None,
            refinement_evidence={},
            far_evidence={"accepted": False},
        )

    monkeypatch.setattr("cv.pipeline.court_topology_runner._solve_anchor_frame", solve)

    anchor, _ = select_anchor(frames, surface_witness_policy="illumination_split")

    assert anchor is not None and anchor.index == ordinary_index
    assert anchor.acceptance_route == "strict_surface"
    # The split witness is never consulted while an ordinary acceptance is still available.
    assert all(policy == "off" for _, policy in seen)


def test_match_prior_requires_repeated_strong_target_line_support(tmp_path):
    frames = []
    image, homography = _synthetic_court_frame()
    for index in range(1, 10):
        path = tmp_path / f"f_{index:04d}.jpg"
        cv2.imwrite(str(path), image)
        frames.append(path)

    recovered = recover_anchor_from_match_prior(frames, 2, [(1, homography)])

    assert recovered is not None
    assert recovered.source == "match_prior_consensus:pt0001"
    assert (
        recovered.refinement_evidence["support_frames"]
        == recovered.refinement_evidence["sampled_frames"]
    )

    blank = np.zeros_like(image)
    for path in frames:
        cv2.imwrite(str(path), blank)
    assert recover_anchor_from_match_prior(frames, 2, [(1, homography)]) is None


def test_registration_line_model_recovers_only_with_strong_target_lines(tmp_path, monkeypatch):
    image, homography = _synthetic_court_frame()
    frames = []
    for index in range(3):
        path = tmp_path / f"f_{index:04d}.jpg"
        cv2.imwrite(str(path), image)
        frames.append(path)
    anchor = AnchorSolution(
        index=1,
        frame=frames[1].name,
        homography=homography,
        source="test",
        topology_score=1.0,
        surface_score=1.0,
        player_score=None,
        refinement_evidence={},
        far_evidence={"accepted": False},
    )
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner.transfer_court_homography",
        lambda *_args: (_ for _ in ()).throw(ValueError("texture registration failed")),
    )
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._standard_broadcast_view", lambda *_: True
    )
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._maximum_topology_score", lambda *_: 0.95
    )

    _, reliable, source, evidence = register_clip(frames, anchor, stride=5)

    assert all(reliable.values())
    assert source == {0: "line_model_registered", 1: "registered", 2: "line_model_registered"}
    assert evidence["line_model_recoveries"] == 2

    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._maximum_topology_score", lambda *_: 0.91
    )
    _, reliable, source, evidence = register_clip(frames, anchor, stride=5)
    assert reliable == {0: False, 1: True, 2: False}
    assert source == {
        0: "anchor_static_fallback",
        1: "registered",
        2: "anchor_static_fallback",
    }
    assert evidence["line_model_recoveries"] == 0


def test_registration_does_not_extrapolate_past_supported_samples(tmp_path, monkeypatch):
    _, homography = _synthetic_court_frame()
    frames = []
    for index in range(5):
        path = tmp_path / f"f_{index:04d}.jpg"
        cv2.imwrite(str(path), np.full((32, 32, 3), index * 30, dtype=np.uint8))
        frames.append(path)
    anchor = AnchorSolution(
        index=2,
        frame=frames[2].name,
        homography=homography,
        source="test",
        topology_score=1.0,
        surface_score=1.0,
        player_score=None,
        refinement_evidence={},
        far_evidence={"accepted": False},
    )

    def transfer(image, *_args):
        index = round(float(np.mean(image)) / 30.0)
        if index in {1, 3}:
            return homography, {"inlier_ratio": 1.0, "target_topology_score": 1.0}
        raise ValueError("unsupported end frame")

    monkeypatch.setattr("cv.pipeline.court_topology_runner.transfer_court_homography", transfer)
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._standard_broadcast_view", lambda *_: True
    )
    monkeypatch.setattr("cv.pipeline.court_topology_runner._maximum_topology_score", lambda *_: 0.0)

    _, reliable, _, _ = register_clip(frames, anchor, stride=1)

    assert reliable == {0: False, 1: True, 2: True, 3: True, 4: False}


def test_emitted_homography_carries_the_declared_edge_convention(tmp_path, monkeypatch):
    frames = tmp_path / "rally_frames" / "pt0001"
    frames.mkdir(parents=True)
    image, homography = _synthetic_court_frame()
    cv2.imwrite(str(frames / "f_0001.jpg"), image)
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._solve_anchor_frame",
        lambda index, path: AnchorSolution(
            index, path.name, homography, "test", 1.0, 1.0, None, {}, {"accepted": False}
        ),
    )
    _, row, shipped, _ = calibrate_clip(frames, frame_track=False)
    identity, evidence = edge_convention_world_transform(ARTIFACT_EDGE_CONVENTION)
    # The shipped artifact stays in the frame the solve produces.
    assert row["edge_convention"] == evidence
    assert np.allclose(identity, np.eye(3), atol=1e-6)
    assert np.allclose(shipped, homography)
    _, row, converted, _ = calibrate_clip(
        frames, frame_track=False, edge_convention="itf_outside_edge"
    )
    assert row["edge_convention"]["edge_convention"] == "itf_outside_edge"
    corner = np.float32([[[0.0, 0.0]]])
    before = cv2.perspectiveTransform(corner, np.linalg.inv(shipped))[0][0]
    after = cv2.perspectiveTransform(corner, np.linalg.inv(converted))[0][0]
    # The near baseline's outside edge is below its paint centre in a broadcast frame.
    assert after[1] > before[1]


def test_registration_retains_a_witness_for_every_sample(tmp_path, monkeypatch):
    frames = [tmp_path / f"f_{index:04d}.jpg" for index in range(6)]
    H = np.eye(3)
    anchor = AnchorSolution(0, frames[0].name, H, "test", 1.0, 1.0, None, {}, {})
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner.cv2.imread",
        lambda path: np.full((8, 8, 3), int(Path(path).stem[2:]), dtype=np.uint8),
    )

    def transfer(image, *_args):
        if int(image[0, 0, 0]) == 5:
            raise ValueError("registered court lacks target support")
        return H, {
            "inlier_ratio": 0.9,
            "target_topology_score": 0.95,
            "target_support_mask": "relaxed_chroma",
        }

    monkeypatch.setattr("cv.pipeline.court_topology_runner.transfer_court_homography", transfer)
    monkeypatch.setattr(
        "cv.pipeline.court_topology_runner._standard_broadcast_view", lambda *_: False
    )
    _, _, _, evidence = register_clip(frames, anchor, stride=2)
    samples = evidence["samples"]
    assert samples[5]["status"] == "held"
    assert "lacks target support" in samples[5]["reason"]
    assert samples[2]["status"] == "registered"
    assert samples[2]["target_support_mask"] == "relaxed_chroma"
