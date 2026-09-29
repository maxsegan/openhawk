from __future__ import annotations

import cv2
import numpy as np
import pytest

import ball_neural
from ball_neural import (
    INPUT_WH,
    affine_for_frame,
    scale_court_gates,
    subpixel_refine,
    temporal_contexts,
    topk_peaks,
)
import resolution


@pytest.mark.parametrize("skip_camera", [False, True])
def test_runtime_counts_native_frames_separately_from_ranked_candidates(
    tmp_path, monkeypatch, capsys, skip_camera
):
    import sys

    clip = tmp_path / "frames/pt0001"
    clip.mkdir(parents=True)
    for frame in range(1, 4):
        cv2.imwrite(str(clip / f"f_{frame:04d}.jpg"), np.zeros((540, 960, 3), np.uint8))
    predictions = [
        {
            "frame": f"f_{frame:04d}.jpg",
            "frame_id": frame,
            "x": 100.0,
            "y": 100.0,
            "score": 0.8,
            "rank": rank,
        }
        for frame, rank in [(1, 0), (1, 1), (2, 0)]
    ]
    outputs = {}

    class Stage:
        def __init__(self, *args, **kwargs):
            pass

        def finish(self, *, outputs):
            captured.update(outputs)

    captured = outputs
    monkeypatch.setattr(ball_neural, "StageRun", Stage)
    monkeypatch.setattr(ball_neural, "build_model", lambda *args: None)
    monkeypatch.setattr(ball_neural, "court_gates_in_artifact_space", lambda *args: {})
    monkeypatch.setattr(ball_neural, "infer_clip", lambda *args: predictions)
    monkeypatch.setattr(ball_neural, "link", lambda *args: [])
    extra = []
    if skip_camera:
        np.savez(
            tmp_path / "camera_P_per_point.npz", pts=np.array([], dtype=int), P=np.empty((0, 3, 4))
        )
        extra = ["--far-native"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ball_neural",
            "--out",
            str(tmp_path),
            "--frames-dir",
            "frames",
            "--model",
            "wasb",
            "--k-best",
            "2",
            *extra,
        ],
    )
    assert ball_neural.main() == 0
    assert outputs["candidate_frames"] == (0 if skip_camera else 2)
    assert outputs["candidate_rows"] == (0 if skip_camera else 3)
    assert outputs["processed_frames"] == (0 if skip_camera else 3)
    assert outputs["input_frames"] == 3
    assert outputs["skipped_camera_clips"] == int(skip_camera)
    assert outputs["processed_clips"] == int(not skip_camera)
    assert outputs["count_contract"] == "tracker_frame_counts_v2"
    if not skip_camera:
        assert "3 processed frames, 3 candidate rows" in capsys.readouterr().out


def test_topk_peaks_returns_argmax_first() -> None:
    hm = np.zeros((20, 30), dtype=np.float32)
    hm[5, 10] = 0.9
    hm[15, 25] = 0.4
    peaks = topk_peaks(hm, k=5, threshold=0.05, radius=3)
    assert peaks[0] == (10, 5, np.float32(0.9)) or (peaks[0][0], peaks[0][1]) == (10, 5)
    assert peaks[0][2] >= peaks[-1][2]  # descending by score
    coords = {(x, y) for x, y, _ in peaks}
    assert (10, 5) in coords and (25, 15) in coords


def test_topk_peaks_nms_suppresses_neighbours() -> None:
    hm = np.zeros((20, 20), dtype=np.float32)
    hm[10, 10] = 0.9
    hm[10, 11] = 0.85  # within radius of the peak -> suppressed
    hm[2, 2] = 0.5  # far -> kept
    peaks = topk_peaks(hm, k=5, threshold=0.05, radius=3)
    coords = [(x, y) for x, y, _ in peaks]
    assert (10, 10) in coords
    assert (11, 10) not in coords
    assert (2, 2) in coords


def test_topk_peaks_k1_matches_global_argmax() -> None:
    rng = np.random.default_rng(0)
    hm = rng.random((36, 64)).astype(np.float32)
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    peaks = topk_peaks(hm, k=1, threshold=0.05, radius=3)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (int(x), int(y))


def test_topk_peaks_empty_map_falls_back_to_argmax() -> None:
    hm = np.full((10, 10), 0.01, dtype=np.float32)
    hm[3, 4] = 0.02
    peaks = topk_peaks(hm, k=5, threshold=0.5, radius=3)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (4, 3)


def test_temporal_contexts_use_past_center_and_future_views() -> None:
    assert temporal_contexts(4, 10) == [
        ([4, 5, 6], 0),
        ([3, 4, 5], 1),
        ([2, 3, 4], 2),
    ]
    assert temporal_contexts(0, 10) == [
        ([0, 1, 2], 0),
        ([0, 0, 1], 1),
        ([0, 0, 0], 2),
    ]


def test_affine_round_trip() -> None:
    forward = affine_for_frame(960, 540)
    inverse = affine_for_frame(960, 540, inverse=True)
    points = np.float32([[[0, 0], [480, 270], [959, 539]]])

    transformed = cv2.transform(points, forward)
    restored = cv2.transform(transformed, inverse)

    np.testing.assert_allclose(restored, points, atol=1e-3)
    center = transformed[0, 1]
    np.testing.assert_allclose(center, [INPUT_WH[0] / 2, INPUT_WH[1] / 2], atol=1e-3)


def test_model_output_coordinate_conversion_is_resolution_invariant() -> None:
    canonical = np.array([351.25, 402.5])
    for size in resolution.SUPPORTED_TEST_SIZES:
        native = resolution.scale_points(canonical, resolution.CANONICAL_SIZE, size)
        restored = resolution.scale_points(native, size, resolution.CANONICAL_SIZE)
        np.testing.assert_allclose(restored, canonical, atol=1e-12)


def test_scale_court_gates_moves_native_polygon_to_artifact_space() -> None:
    native = resolution.FrameSize(1920, 1080)
    artifact = resolution.FrameSize(960, 540)
    gates = {1: np.asarray([[200, 100], [1800, 100], [1800, 1000], [200, 1000]])}

    scaled = scale_court_gates(gates, native, artifact)

    np.testing.assert_allclose(
        scaled[1],
        [[100, 50], [900, 50], [900, 500], [100, 500]],
    )
    assert scaled[1].dtype == np.float32


def test_subpixel_refine_argmax_is_identity_default() -> None:
    # the production default must not move the peak: argmax returns exactly the argmax cell.
    hm = np.zeros((20, 30), dtype=np.float32)
    hm[7, 11] = 1.0
    assert subpixel_refine(hm, 11, 7, "argmax") == (11.0, 7.0)


def test_subpixel_refine_recovers_subcell_peak() -> None:
    ys, xs = np.mgrid[0:40, 0:40]
    hm = np.exp(-(((xs - 15.3) ** 2 + (ys - 22.7) ** 2) / (2 * 1.8**2))).astype(np.float32)
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    for method in ("parabolic", "centroid"):
        rx, ry = subpixel_refine(hm, int(x), int(y), method)
        assert np.hypot(rx - 15.3, ry - 22.7) < np.hypot(int(x) - 15.3, int(y) - 22.7)


def test_subpixel_refine_composes_with_topk_peaks() -> None:
    ys, xs = np.mgrid[0:40, 0:40]
    hm = np.exp(-(((xs - 15.3) ** 2 + (ys - 22.7) ** 2) / (2 * 1.8**2))).astype(np.float32)
    ((px, py, _),) = topk_peaks(hm, k=1, threshold=0.05, radius=3)
    rx, ry = subpixel_refine(hm, px, py, "centroid")
    assert abs(rx - 15.3) < 0.4 and abs(ry - 22.7) < 0.4


def test_subpixel_refine_edge_peak_stays_in_bounds() -> None:
    hm = np.zeros((16, 16), dtype=np.float32)
    hm[0, 0] = 1.0
    for method in ("argmax", "parabolic", "centroid"):
        rx, ry = subpixel_refine(hm, 0, 0, method)
        assert 0 <= rx < 16 and 0 <= ry < 16


def _broadcast_homography() -> np.ndarray:
    """Image->court homography from a measured broadcast camera (native 1920x1080 px)."""
    court = np.float32([[0.0, 0.0], [10.97, 0.0], [0.0, 23.77], [10.97, 23.77]])
    image = np.float32(
        [
            [266.78, 827.12],
            [1721.73, 828.51],
            [629.01, 296.52],
            [1347.49, 296.33],
        ]
    )
    return cv2.getPerspectiveTransform(image, court)


def _court_candidates_in_artifact_space(
    homography: np.ndarray,
    artifact_size: resolution.FrameSize,
) -> np.ndarray:
    """A grid of genuinely on-court points, in the space the candidate CSV uses."""
    xs, ys = np.meshgrid(np.linspace(0.2, 10.77, 12), np.linspace(0.2, 23.57, 24))
    court = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    native = cv2.perspectiveTransform(court.reshape(1, -1, 2), np.linalg.inv(homography))[0]
    return resolution.scale_points(native, resolution.NATIVE_SIZE, artifact_size)


def _on_court_fraction(gate: np.ndarray, points: np.ndarray) -> float:
    inside = [cv2.pointPolygonTest(gate, (float(x), float(y)), False) >= 0 for x, y in points]
    return sum(inside) / len(inside)


def test_court_gate_is_scaled_exactly_once_into_artifact_space(tmp_path) -> None:
    out = tmp_path / "match"
    clip = out / "frames" / "pt0001"
    clip.mkdir(parents=True)
    cv2.imwrite(str(clip / "f_0001.jpg"), np.zeros((1080, 1920, 3), dtype=np.uint8))
    homography = _broadcast_homography()
    np.savez(out / "court_H_per_point.npz", pts=np.array([1]), H=homography[None])
    artifact_size = resolution.FrameSize(960, 540)

    gate = ball_neural.court_gates_in_artifact_space(str(out), "frames", artifact_size)[1]
    candidates = _court_candidates_in_artifact_space(homography, artifact_size)

    assert _on_court_fraction(gate, candidates) == 1.0
    # The defect this replaced applied the native->artifact scale a second time, which
    # contracts the polygon toward the origin and rejects most of the real court.
    double_scaled = resolution.scale_points(gate, resolution.NATIVE_SIZE, artifact_size).astype(
        np.float32
    )
    assert _on_court_fraction(double_scaled, candidates) < 0.5


def test_scale_projection_moves_camera_into_artifact_space() -> None:
    homography = _broadcast_homography()
    court_to_image = np.linalg.inv(homography)
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, 0] = court_to_image[:, 0]
    projection[:, 1] = court_to_image[:, 1]
    projection[:, 3] = court_to_image[:, 2]
    artifact_size = resolution.FrameSize(960, 540)

    scaled = ball_neural.scale_projection(projection, resolution.NATIVE_SIZE, artifact_size)

    world = np.array([5.485, 11.885, 0.0, 1.0])
    native = projection @ world
    artifact = scaled @ world
    assert np.allclose(artifact[:2] / artifact[2], (native[:2] / native[2]) / 2.0)


def test_native_named_candidates_declare_the_native_interface(tmp_path) -> None:
    """The regression a batch run found: ball tracking raised at main because the candidate
    writer declared 960x540 under a ``native``-named artifact."""
    from cv.pipeline import resolution as res
    from cv.pipeline.track_artifact import write_candidate_artifact, write_track_artifact

    candidates = tmp_path / "ball_candidates_wasb_native1080_sliding_k5_v1.csv"
    rows = [
        {
            "clip": "pt0001",
            "frame": "f_0001.jpg",
            "x": 480.0,
            "y": 270.0,
            "score": 0.9,
            "on_court": 1,
            "rank": 0,
        }
    ]
    write_candidate_artifact(
        candidates,
        ["clip", "frame", "x", "y", "score", "on_court", "rank"],
        rows,
        image_size=res.NATIVE_SIZE,
        legacy_size=res.CANONICAL_SIZE,
        source="frames",
    )
    manifest = res.read_coordinate_manifest(candidates)
    assert res.manifest_artifact_size(manifest) == res.NATIVE_SIZE
    assert res.manifest_artifact_size(manifest, legacy_columns=True) == res.CANONICAL_SIZE
    assert res.declared_artifact_space(candidates) == res.CANONICAL_SIZE
    assert (
        res.declared_artifact_space(candidates, columns=("x_native", "y_native")) == res.NATIVE_SIZE
    )
    with candidates.open() as handle:
        header = handle.readline().strip().split(",")
        first = dict(zip(header, handle.readline().strip().split(",")))
    assert (float(first["x_native"]), float(first["y_native"])) == (960.0, 540.0)
    assert (float(first["x"]), float(first["y"])) == (480.0, 270.0)
    # The linked track derives its contract from the candidate sidecar.
    track = tmp_path / "ball_track_wasb_native1080_sliding_k5_v1.csv"
    write_track_artifact(
        track,
        [{"clip": "pt0001", "frame": "f_0001.jpg", "x": 480.0, "y": 270.0, "track_id": 0}],
        [candidates],
    )
    assert res.declared_artifact_space(track, columns=("x_native", "y_native")) == res.NATIVE_SIZE
