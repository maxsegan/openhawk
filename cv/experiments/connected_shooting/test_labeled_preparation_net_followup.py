from dataclasses import dataclass
import numpy as np
import pytest
from cv.experiments.connected_shooting.labeled_preparation_net_followup import fit_check_copy


@dataclass
class Splits:
    observation_frames: tuple
    pixels: tuple
    cameras: tuple
    camera_distortion: object = None


def test_consumed_rows_removed_only_from_check_copy():
    scene = Splits(
        (np.array([1.0, 2.0]),), (np.array([[10.0, 20.0], [11.0, 21.0]]),), (np.ones((2, 3, 4)),)
    )
    check = Splits(
        (np.array([2.0, 3.0]),), (np.array([[11.0, 21.0], [12.0, 22.0]]),), (np.ones((2, 3, 4)),)
    )
    result, used = fit_check_copy(scene, check)
    assert used == [[2.0]]
    assert result.observation_frames[0].tolist() == [3.0]
    assert check.observation_frames[0].tolist() == [2.0, 3.0]
    assert scene.observation_frames[0].tolist() == [1.0, 2.0]
    check.pixels[0][0, 0] = 900
    with pytest.raises(ValueError, match="differs"):
        fit_check_copy(scene, check)
