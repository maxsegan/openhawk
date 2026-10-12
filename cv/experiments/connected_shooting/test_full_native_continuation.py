"""Activation must preserve observation identity and original scoring directions."""

from dataclasses import dataclass
import numpy as np
import pytest
from cv.experiments.connected_shooting.full_native_continuation import merge_scene


@dataclass
class Scene:
    contact_frames: np.ndarray
    observation_frames: tuple
    cameras: tuple
    pixels: tuple
    camera_distortion: tuple | None

    def validate(self):
        assert len(self.observation_frames[0]) == len(self.cameras[0]) == len(self.pixels[0])


def make(frames, radial=True):
    f = np.array(frames, dtype=float)
    return Scene(
        np.array([0.0, 20.0]),
        (f,),
        (np.repeat(f[:, None, None], 12, axis=1).reshape(-1, 3, 4),),
        (np.c_[f, f + 100],),
        (np.c_[f, f + 1, f + 2],) if radial else None,
    )


def test_merge_keeps_pixels_cameras_radial_metadata_and_original_wing_axes():
    train = make([1, 9, 11, 19])
    check = make([5, 10, 15])
    axes = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    merged, active_axes, added = merge_scene(train, check, (np.array([10.0]),), axes)
    np.testing.assert_array_equal(merged.observation_frames[0], [1, 5, 9, 10, 11, 15, 19])
    np.testing.assert_array_equal(
        active_axes, [[1, 0], [1, 0], [2, 0], [3, 0], [3, 0], [3, 0], [4, 0]]
    )
    np.testing.assert_array_equal(merged.pixels[0][:, 0], merged.observation_frames[0])
    np.testing.assert_array_equal(merged.cameras[0][:, 0, 0], merged.observation_frames[0])
    np.testing.assert_array_equal(merged.camera_distortion[0][:, 0], merged.observation_frames[0])
    np.testing.assert_array_equal(train.observation_frames[0], [1, 9, 11, 19])
    assert added == [[5, 10, 15]]
    assert merged.contact_frames is train.contact_frames


def test_duplicate_exposure_is_not_counted_twice():
    train = make([1, 9, 11, 19])
    check = make([9, 15])
    axes = np.ones((4, 2))
    with pytest.raises(ValueError, match="overlapping"):
        merge_scene(train, check, (np.array([10.0]),), axes)


def test_distortion_cannot_silently_disappear_on_activation():
    with pytest.raises(ValueError, match="distortion"):
        merge_scene(
            make([1, 9, 11, 19], False), make([5, 10, 15]), (np.array([10.0]),), np.ones((4, 2))
        )


def test_endpoint_readback_uses_actual_contact_not_last_observed_picture():
    from cv.experiments.connected_shooting.full_native_continuation import endpoint_difference

    old = [dict(end_xyz=np.array([1.0, 2.0, 3.0]), positions=np.array([[0.0, 0.0, 0.0]]))]
    new = [dict(end_xyz=np.array([1.0, 2.0, 3.0]), positions=np.array([[9.0, 9.0, 9.0]]))]
    assert endpoint_difference(old, new) == 0
    new[0]["end_xyz"][0] += 0.01
    assert endpoint_difference(old, new) == pytest.approx(0.01)
