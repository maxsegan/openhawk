"""A timing profile changes its exported physical epoch, never native pictures."""

from dataclasses import dataclass
import numpy as np
import pytest
from cv.experiments.connected_shooting.serve_timing_profile import (
    contact_epoch_bounds,
    profile_context,
)


@dataclass
class Scene:
    contact_frames: np.ndarray
    observation_frames: tuple


def context():
    scene = Scene(np.array([884.5, 951, 1008]), (np.array([885, 886, 950]), np.array([952, 1008])))
    return dict(
        scene=scene,
        heldout=scene,
        events=[
            dict(event_type="contact", frame=884.5, frame_interval=[884, 885]),
            dict(event_type="bounce", frame=909.5, frame_interval=[909, 910]),
        ],
    )


@pytest.mark.parametrize("epoch", [884, 884.5, 885])
def test_exported_profile_preserves_outer_epoch_native_membership_and_original_interval(epoch):
    old = context()
    new = profile_context(old, epoch)
    assert new["scene"].contact_frames[0] == epoch
    assert new["heldout"].contact_frames[0] == epoch
    np.testing.assert_array_equal(new["scene"].contact_frames[1:], old["scene"].contact_frames[1:])
    assert new["scene"].observation_frames is old["scene"].observation_frames
    assert new["events"][0]["frame"] == epoch
    assert new["events"][0]["frame_interval"] == [884, 885]
    assert old["events"][0]["frame"] == old["scene"].contact_frames[0] == 884.5
    assert new["events"][1] == old["events"][1]


def test_profile_rejects_outside_bracket_or_reassigned_picture():
    old = context()
    with pytest.raises(ValueError, match="outside original"):
        profile_context(old, 885.1)
    old["scene"].observation_frames = (np.array([884.7, 886, 950]), np.array([952, 1008]))
    with pytest.raises(ValueError, match="reassign"):
        profile_context(old, 885)


@pytest.mark.parametrize("limiting_scene", ["scene", "heldout"])
def test_contact_domain_preserves_rows_and_admits_every_bounded_epoch(limiting_scene):
    old = context()
    old[limiting_scene] = Scene(
        old[limiting_scene].contact_frames.copy(),
        (np.array([884.5, 886, 950]), np.array([952, 1008])),
    )
    assert contact_epoch_bounds(old) == (884, 884.5)
    for epoch in np.linspace(*contact_epoch_bounds(old), 5):
        new = profile_context(old, epoch)
        for name in ("scene", "heldout"):
            assert new[name].observation_frames is old[name].observation_frames
    assert old["events"][0]["frame_interval"] == [884, 885]


def test_singleton_and_empty_contact_domains_are_distinct():
    old = context()
    old["scene"].observation_frames = (np.array([884]), np.array([952, 1008]))
    assert contact_epoch_bounds(old) == (884, 884)
    profile_context(old, 884)
    old["scene"].observation_frames = (np.array([883.9]), np.array([952, 1008]))
    with pytest.raises(ValueError, match="empty contact domain"):
        contact_epoch_bounds(old)
