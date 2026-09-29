from pathlib import Path

import numpy as np

from cv.pipeline.court_topology_frame_track import interpolate_track_H, nearest_H


def _write_track(
    path: Path,
    *,
    points: list[int],
    frames: list[np.ndarray],
    homographies: list[np.ndarray],
) -> None:
    np.savez(
        path,
        pts=np.asarray(points, dtype=np.int32),
        frames=np.asarray(frames, dtype=object),
        H=np.asarray(homographies, dtype=object),
    )


def test_interpolate_track_H_clamps_to_endpoints() -> None:
    frames = np.asarray([10, 30], dtype=np.int32)
    homographies = np.asarray(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 8.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
        ]
    )

    np.testing.assert_array_equal(interpolate_track_H(frames, homographies, 5), homographies[0])
    np.testing.assert_array_equal(interpolate_track_H(frames, homographies, 40), homographies[-1])


def test_interpolate_track_H_returns_only_sample() -> None:
    frames = np.asarray([20], dtype=np.int32)
    homographies = np.asarray([[[2.0, 0.0, 4.0], [0.0, 2.0, 6.0], [0.0, 0.0, 2.0]]])

    np.testing.assert_array_equal(interpolate_track_H(frames, homographies, 100), homographies[0])


def test_interpolate_track_H_normalizes_before_mid_gap_interpolation() -> None:
    frames = np.asarray([10, 30], dtype=np.int32)
    normalized_a = np.asarray([[1.0, 0.1, 2.0], [0.2, 1.0, 3.0], [0.001, 0.002, 1.0]])
    normalized_b = np.asarray([[1.2, 0.3, 8.0], [0.4, 0.9, 9.0], [0.003, 0.004, 1.0]])
    homographies = np.stack([normalized_a * 2.0, normalized_b * 4.0])

    actual = interpolate_track_H(frames, homographies, 15)

    np.testing.assert_allclose(actual, 0.75 * normalized_a + 0.25 * normalized_b)
    assert actual[2, 2] == 1.0


def test_nearest_H_returns_none_for_missing_point(tmp_path: Path) -> None:
    track_path = tmp_path / "track.npz"
    homography = np.eye(3)
    _write_track(
        track_path,
        points=[1],
        frames=[np.asarray([10], dtype=np.int32)],
        homographies=[np.stack([homography])],
    )

    assert nearest_H(track_path, point=2, frame=10) is None


def test_nearest_H_returns_none_for_empty_frames(tmp_path: Path) -> None:
    track_path = tmp_path / "track.npz"
    _write_track(
        track_path,
        points=[1],
        frames=[np.asarray([], dtype=np.int32)],
        homographies=[np.empty((0, 3, 3), dtype=np.float64)],
    )

    assert nearest_H(track_path, point=1, frame=10) is None


def test_nearest_H_selects_closest_sample(tmp_path: Path) -> None:
    track_path = tmp_path / "track.npz"
    homographies = np.stack(
        [
            np.eye(3),
            np.asarray([[1.0, 0.0, 30.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]),
            np.asarray([[1.0, 0.0, 70.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
        ]
    )
    _write_track(
        track_path,
        points=[4],
        frames=[np.asarray([10, 30, 70], dtype=np.int32)],
        homographies=[homographies],
    )

    np.testing.assert_array_equal(nearest_H(track_path, point=4, frame=52), homographies[2])
