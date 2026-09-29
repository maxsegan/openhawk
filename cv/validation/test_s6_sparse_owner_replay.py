from copy import deepcopy

import numpy as np
import pytest

from cv.validation import s6_sparse_owner_replay as replay


def inputs():
    labels = [dict(frame=f, status="visible", x1080=100 + f, y1080=300 + f) for f in range(2, 26)]
    attempt = dict(
        structurally_ground_replayable=True,
        point_clip="pt1",
        match_id="m",
        owner_ball_labels=labels,
        visible_native_frames=24,
        fps=60,
        owner_end_frame=25.0,
        events=[
            dict(event_type=t, frame=f)
            for t, f in (("contact", 1.5), ("bounce", 8), ("contact", 12.5), ("bounce", 25))
        ],
    )
    cameras = dict(
        clip="pt1",
        match_id="m",
        cameras=[dict(frame=f, status="supported", P=np.eye(3, 4).tolist()) for f in range(2, 26)],
    )
    return attempt, cameras


def test_disjoint_predeclared_split_keeps_every_native_owner_coordinate():
    attempt, cameras = inputs()
    original = deepcopy((attempt, cameras))
    train, check, bounces, native = replay.prepare(attempt, cameras)
    assert np.array_equal(train.contact_frames, [1.5, 12.5, 25.0])
    assert [r.tolist() for r in bounces] == [[8.0], [25.0]]
    fs = np.concatenate(native)
    assert np.array_equal(fs, np.arange(2, 26))
    assert set(np.concatenate(train.observation_frames)).isdisjoint(
        np.concatenate(check.observation_frames)
    )
    for scene in (train, check):
        for frames, pixels in zip(scene.observation_frames, scene.pixels):
            np.testing.assert_array_equal(pixels, np.c_[100 + frames, 300 + frames])
    assert all(f % 5 == 0 for f in np.concatenate(check.observation_frames))
    assert (attempt, cameras) == original


def test_explicit_timing_control_preserves_every_pixel_and_original_event():
    attempt, cameras = inputs()
    original = deepcopy(attempt)
    before = replay.prepare(attempt, cameras)
    after = replay.prepare(attempt, cameras, first_contact_offset_frames=-0.5)
    assert after[0].contact_frames[0] == 1.0
    for a, b in zip(before[:2], after[:2]):
        for key in ("observation_frames", "cameras", "pixels"):
            for x, y in zip(getattr(a, key), getattr(b, key)):
                np.testing.assert_array_equal(x, y)
    assert attempt == original


@pytest.mark.parametrize("offset", [1, -1.1, float("nan"), float("inf")])
def test_timing_control_cannot_drop_observations_or_escape_bound(offset):
    with pytest.raises(ValueError):
        replay.prepare(*inputs(), first_contact_offset_frames=offset)


@pytest.mark.parametrize("offset", [-1, -0.5, 0, 0.5, 1])
def test_interior_timing_reassigns_flight_not_pixels_or_split(offset):
    attempt, cameras = inputs()
    original = deepcopy((attempt, cameras))
    before = replay.prepare(attempt, cameras)
    after = replay.prepare(attempt, cameras, interior_contact_offsets_frames={1: offset})
    assert after[0].contact_frames.tolist() == [1.5, 12.5 + offset, 25.0]
    for a, b in zip(before[:2], after[:2]):
        for key in ("observation_frames", "cameras", "pixels"):
            np.testing.assert_array_equal(
                np.concatenate(getattr(a, key)), np.concatenate(getattr(b, key))
            )
    assert after[0].observation_frames[0][-1] < 12.5 + offset
    assert after[0].observation_frames[1][0] >= 12.5 + offset
    np.testing.assert_array_equal(np.concatenate(after[3]), np.arange(2, 26))
    assert (attempt, cameras) == original


@pytest.mark.parametrize(
    "offsets", [{0: 0}, {2: 0}, {-1: 0}, {True: 0}, {1.0: 0}, {1: True}, {1: 1.01}, {1: np.nan}]
)
def test_invalid_interior_timing_rejected(offsets):
    with pytest.raises(ValueError):
        replay.prepare(*inputs(), interior_contact_offsets_frames=offsets)


@pytest.mark.parametrize(
    "kind", ["held", "missing", "duplicate", "wrong_clip", "grammar", "coverage", "bounce"]
)
def test_unsupported_or_incomplete_evidence_cannot_be_dropped(kind):
    a, c = inputs()
    if kind == "held":
        c["cameras"][0]["status"] = "held"
    elif kind == "missing":
        c["cameras"].pop()
    elif kind == "duplicate":
        c["cameras"].append(c["cameras"][0])
    elif kind == "wrong_clip":
        c["clip"] = "different"
    elif kind == "grammar":
        a["structurally_ground_replayable"] = False
    elif kind == "coverage":
        a["owner_ball_labels"] = a["owner_ball_labels"][:4]
        a["visible_native_frames"] = 4
    else:
        a["events"].pop()
    with pytest.raises(ValueError):
        replay.prepare(a, c)
