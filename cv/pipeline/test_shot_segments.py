from pathlib import Path

import cv2
import numpy as np

from shot_segments import ShotSegmentation, merge_short_runs, propagation_shot_ids


def test_merge_absorbs_isolated_flicker() -> None:
    identifiers = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2])
    merged = merge_short_runs(identifiers, minimum=3)
    assert len(np.unique(merged)) == 2
    assert merged[5] == merged[4]


def test_merge_keeps_genuine_shots() -> None:
    identifiers = np.array([0] * 10 + [1] * 10 + [2] * 10)
    merged = merge_short_runs(identifiers, minimum=4)
    assert len(np.unique(merged)) == 3


def test_expand_assigns_nearest_sample() -> None:
    segmentation = ShotSegmentation(
        sampled_frames=np.array([1, 3, 5]),
        shot_id=np.array([0, 0, 1]),
        is_play_camera=np.array([True, True, False]),
        registration_inliers=np.array([100.0, 100.0, 10.0]),
    )
    shot_id, is_play, support = segmentation.expand(np.array([1, 2, 3, 4, 5]))
    assert shot_id.tolist() == [0, 0, 0, 0, 1]
    assert is_play.tolist() == [True, True, True, True, False]
    assert support.tolist() == [100.0, 100.0, 100.0, 100.0, 10.0]


def test_expand_empty_segmentation_abstains() -> None:
    segmentation = ShotSegmentation(
        sampled_frames=np.array([], dtype=int),
        shot_id=np.array([], dtype=int),
        is_play_camera=np.array([], dtype=bool),
        registration_inliers=np.array([], dtype=float),
    )
    shot_id, is_play, support = segmentation.expand(np.array([1, 2]))
    assert shot_id.tolist() == [-1, -1]
    assert not is_play.any()
    assert not support.any()


def _write_clip(directory: Path, values: list[int]) -> list[Path]:
    directory.mkdir(parents=True)
    paths = []
    for index, value in enumerate(values, start=1):
        path = directory / f"f_{index:04d}.jpg"
        image = np.full((36, 64), value, np.uint8)
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def test_propagation_floor_ignores_a_static_blip_and_keeps_a_hard_cut(tmp_path: Path) -> None:
    # A few grey levels is a huge z-score on an otherwise still shot, and not a cut.
    still = _write_clip(tmp_path / "still", [10] * 20 + [18] * 20 + [10] * 20)
    shots = propagation_shot_ids(still, fps=25.0, stride=1)
    assert set(shots.values()) == {0}
    # A hard cut (black to white) stays two shots.
    cut = _write_clip(tmp_path / "cut", [0] * 20 + [255] * 20)
    shots = propagation_shot_ids(cut, fps=25.0, stride=1)
    assert shots[1] != shots[40]
    assert shots[1] == shots[20]
    assert shots[21] == shots[40]
