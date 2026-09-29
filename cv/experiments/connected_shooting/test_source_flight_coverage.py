"""Source coverage ownership: camera identity, membership and train/check parity."""

import numpy as np
import pytest

from cv.experiments.connected_shooting import source_flight_coverage as coverage

# Original source46 shape: four automatic contacts and an unresolved horizon at 135.
CONTACTS = [45.09228515625, 67.91796875, 99.1689453125, 132.666015625]
HORIZON = 135.0
WINDOW = [1, 135]


def event(kind, frame):
    return dict(
        event_type=kind, frame=frame, frame_interval=[frame - 1.0, frame + 1.0], status="predicted"
    )


def camera(frame, status="supported"):
    return dict(frame=frame, status=status, P=[[1.0] * 4] * 3)


def inputs(contacts=None, end=HORIZON, bounces=(), invisible=(), unsupported=(), window=None):
    contacts = CONTACTS if contacts is None else contacts
    window = WINDOW if window is None else window
    events = sorted(
        [event("contact", frame) for frame in contacts]
        + [event("bounce", frame) for frame in bounces],
        key=lambda e: (e["frame"], e["event_type"]),
    )
    labels = [
        dict(
            frame=frame,
            status="absent" if frame in invisible else "visible",
            x1080=float(frame),
            y1080=2.0 * frame,
        )
        for frame in range(window[0], window[1] + 1)
    ]
    attempt = dict(
        events=events,
        owner_end_frame=end,
        owner_ball_labels=labels,
        point_clip="pt0046",
        match_id="wta_2020_580_f_226",
        fps=25.0,
    )
    cameras = dict(
        clip="pt0046",
        match_id="wta_2020_580_f_226",
        cameras=[
            camera(frame, "unsupported" if frame in unsupported else "supported")
            for frame in range(window[0], window[1] + 1)
        ],
    )
    return attempt, cameras


def test_visible_inputs_requires_one_matching_camera_document():
    attempt, cameras = inputs()
    rows, labels = coverage.visible_inputs(attempt, cameras)
    assert len(rows) == 135 and len(labels) == 135
    assert labels[46]["x1080"] == 46.0
    for change in ("clip", "match_id"):
        broken = dict(cameras, **{change: "other"})
        with pytest.raises(ValueError, match="one matching camera document"):
            coverage.visible_inputs(attempt, broken)
    duplicated = dict(cameras, cameras=[*cameras["cameras"], camera(46)])
    with pytest.raises(ValueError, match="one matching camera document"):
        coverage.visible_inputs(attempt, duplicated)


def test_visible_label_needs_supported_camera_unless_fallback_declared():
    attempt, cameras = inputs(unsupported=(50, 51))
    with pytest.raises(ValueError, match="supported matching camera"):
        coverage.visible_inputs(attempt, cameras)
    receipt = []
    _, labels = coverage.visible_inputs(
        attempt, cameras, observation_fallback=True, fallback_receipt=receipt
    )
    assert 50 not in labels and 51 not in labels and len(labels) == 133
    assert receipt == [
        {
            "fallback": "unsupported_camera_observations_omitted",
            "frames": [50, 51],
            "retained_visible_observations": 133,
        }
    ]
    missing = dict(cameras, cameras=[row for row in cameras["cameras"] if row["frame"] != 60])
    with pytest.raises(ValueError, match="supported matching camera"):
        coverage.visible_inputs(attempt, missing)


def test_absent_label_is_not_an_observation():
    attempt, cameras = inputs(invisible=(46, 47))
    _, labels = coverage.visible_inputs(attempt, cameras)
    assert 46 not in labels and len(labels) == 133


@pytest.mark.parametrize("inclusive_end", [False, True])
def test_frame_partition_membership_and_fifth_frame_ownership(inclusive_end):
    labels = {frame: frame for frame in range(1, 40)}
    native, train, check = coverage.frame_partition(
        labels,
        10.0,
        25.0,
        inclusive_end=inclusive_end,
        observation_partition="fifth_frame_withheld",
    )
    expected = list(range(10, 26 if inclusive_end else 25))
    assert native.tolist() == [float(f) for f in expected]
    assert check.tolist() == [float(f) for f in expected if not f % 5]
    assert train.tolist() == [float(f) for f in expected if f % 5]
    assert (25.0 in check.tolist()) is inclusive_end


def test_all_native_partition_keeps_every_row_and_copies_a_missing_check():
    labels = {frame: frame for frame in range(1, 40)}
    native, train, check = coverage.frame_partition(
        labels, 10.0, 25.0, inclusive_end=False, observation_partition="all_native"
    )
    assert train.tolist() == native.tolist()
    assert check.tolist() == [10.0, 15.0, 20.0]
    _, sparse_train, sparse_check = coverage.frame_partition(
        {frame: frame for frame in (11, 12, 13, 14)},
        10.0,
        16.0,
        inclusive_end=False,
        observation_partition="all_native",
    )
    assert sparse_train.tolist() == sparse_check.tolist() == [11.0, 12.0, 13.0, 14.0]
    with pytest.raises(ValueError, match="observation partition"):
        coverage.frame_partition(
            labels, 10.0, 25.0, inclusive_end=False, observation_partition="every_frame"
        )


def test_inventory_reproduces_the_source46_coverage_shape():
    attempt, cameras = inputs()
    rows = coverage.inventory(attempt, cameras)
    assert [(row["train_count"], row["check_count"]) for row in rows] == [
        (18, 4),
        (26, 6),
        (26, 7),
        (2, 1),
    ]
    assert [row["coverage_qualified"] for row in rows] == [True, True, True, False]
    assert [row["inclusive_end"] for row in rows] == [False, False, False, True]
    assert rows[3]["native_frames"] == [133, 134, 135]
    assert rows[3]["start_frame"] == CONTACTS[3] and rows[3]["end_frame"] == HORIZON
    assert coverage.first_coverage_failure(rows) == 3
    assert all(row["source_bounce_frames"] == [] for row in rows)


def test_inventory_membership_partitions_every_in_window_observation_once():
    attempt, cameras = inputs(bounces=(57.5, 80.25))
    rows = coverage.inventory(attempt, cameras)
    _, labels = coverage.visible_inputs(attempt, cameras, observation_fallback=True)
    inside = [f for f in labels if rows[0]["start_frame"] <= f <= HORIZON]
    assert sum(row["native_count"] for row in rows) == len(inside)
    assert not set.intersection(*[set(row["native_frames"]) for row in rows[:2]])
    assert [row["source_bounce_frames"] for row in rows] == [[57.5], [80.25], [], []]


def test_inventory_needs_an_ordered_contact_endpoint_domain():
    attempt, cameras = inputs()
    with pytest.raises(ValueError, match="at least one contact boundary"):
        coverage.inventory(attempt, cameras, events=[event("bounce", 60.0)])
    with pytest.raises(ValueError, match="ordered distinct"):
        coverage.inventory(attempt, cameras, end_frame=CONTACTS[-1])
    with pytest.raises(ValueError, match="ordered distinct"):
        coverage.inventory(attempt, cameras, end_frame=100.0)


def test_inventory_never_mutates_supplied_rows_or_reads_fitted_state():
    attempt, cameras = inputs()
    before = np.asarray([row["frame"] for row in attempt["owner_ball_labels"]])
    rows = coverage.inventory(attempt, cameras)
    after = np.asarray([row["frame"] for row in attempt["owner_ball_labels"]])
    assert np.array_equal(before, after)
    assert attempt["events"] == inputs()[0]["events"] and rows[0]["train_count"] == 18
    source = __import__("pathlib").Path(coverage.__file__).read_text()
    assert not any(
        token in source for token in ("result.json", "verdict", "candidate_", "scipy", "optimi")
    )
