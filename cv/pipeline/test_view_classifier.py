import json

import numpy as np

from cv.pipeline import view_classifier
from cv.pipeline.view_classifier import (
    MOTION_FEATURES,
    ViewHead,
    frame_times,
    shot_features,
    shot_frame_indices,
)


def _shot(start: float, end: float) -> dict:
    return {
        "t_start": start,
        "t_end": end,
        "frames": 100,
        **{name: 1.0 for name in MOTION_FEATURES},
    }


def test_frame_times_are_sample_centres() -> None:
    times = frame_times(3, fps=2.0, start_seconds=10.0)

    assert np.allclose(times, [10.25, 10.75, 11.25])


def test_shot_frame_indices_stay_inside_the_shot() -> None:
    times = frame_times(20, fps=2.0, start_seconds=0.0)
    shots = [_shot(0.0, 5.0), _shot(5.0, 10.0)]

    picks = shot_frame_indices(shots, times)

    assert all(0.0 < times[index] < 5.0 for index in picks[0])
    assert all(5.0 < times[index] < 10.0 for index in picks[1])


def test_shot_frame_indices_never_return_an_empty_shot() -> None:
    times = frame_times(4, fps=1.0, start_seconds=0.0)
    shots = [_shot(1.6, 1.9)]

    picks = shot_frame_indices(shots, times)

    assert len(picks[0]) == 1


def test_shot_features_pool_embeddings_and_append_motion() -> None:
    embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
    shots = [_shot(0.0, 4.0)]

    features = shot_features(shots, embeddings, [[0, 2]])

    assert features.shape == (1, 3 + len(MOTION_FEATURES) + 4)
    assert np.allclose(features[0, :3], embeddings[[0, 2]].mean(axis=0))


def test_view_head_round_trips_through_json() -> None:
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(4, 1, (40, 4))]).astype(np.float32)
    labels = ["play"] * 40 + ["crowd"] * 40

    head = ViewHead.fit(features, labels)
    restored = ViewHead.from_json(json.loads(json.dumps(head.to_json())))

    assert head.predict(features) == restored.predict(features)
    assert np.allclose(head.probabilities(features).sum(axis=1), 1.0)
    assert head.predict(features)[:5] == ["play"] * 5


def test_view_head_class_weight_lifts_the_rare_class() -> None:
    import numpy as np

    from cv.pipeline.view_classifier import ViewHead

    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 1.0, size=(400, 2))
    rare = rng.normal(1.2, 1.0, size=(12, 2))
    features = np.vstack([common, rare]).astype(np.float32)
    labels = ["play"] * len(common) + ["replay"] * len(rare)
    plain = ViewHead.fit(features, labels)
    balanced = ViewHead.fit(features, labels, class_weight="balanced")
    recalled = lambda head: sum(  # noqa: E731
        prediction == "replay" for prediction in head.predict(features[len(common) :])
    )
    assert recalled(balanced) > recalled(plain)


def _signals(seconds: float, fps: float = 10.0):
    from cv.pipeline.shot_boundaries import Signals

    count = int(seconds * fps)
    rng = np.random.default_rng(0)
    motion = rng.uniform(0.0, 1.0, count).astype(np.float32)
    return Signals(fps, 0.0, motion.copy(), motion, motion, motion)


def test_long_shot_windows_split_only_long_shots_and_keep_parents():
    from cv.pipeline.shot_boundaries import shots_from_cuts

    signals = _signals(400.0)
    shots = shots_from_cuts(signals, np.array([300]))  # a 30 s shot, then a 370 s shot
    windows = view_classifier.view_windows(
        shots, signals, longer_than=180.0, window=20.0, continuous_feed=300.0
    )
    assert windows[0] == {**shots[0], "window_index": 0}
    long = [row for row in windows if row["shot_index"] == 1]
    assert len(long) == 19 and view_classifier.split_any(windows)
    assert long[0]["t_start"] == shots[1]["t_start"] and long[-1]["t_end"] == shots[1]["t_end"]
    assert sum(row["frames"] for row in long) == shots[1]["frames"]
    assert all(a["t_end"] == b["t_start"] for a, b in zip(long, long[1:]))
    probabilities = np.zeros((len(windows), 5))
    probabilities[0, 0] = 1.0
    probabilities[1:10, 0] = 1.0
    probabilities[10:, 4] = 1.0
    rows = view_classifier.shot_rows_from_windows(shots, windows, probabilities)
    assert rows[0, 0] == 1.0 and 0.4 < rows[1, 0] < 0.6
    assert np.allclose(rows.sum(axis=1), 1.0)


def test_shots_under_the_threshold_are_not_split():
    from cv.pipeline.shot_boundaries import shots_from_cuts

    signals = _signals(300.0)
    shots = shots_from_cuts(signals, np.array([1000, 2000]))
    windows = view_classifier.view_windows(shots, signals, longer_than=180.0)
    assert not view_classifier.split_any(windows)
    assert [dict(row) for row in windows] == [{**shot, "window_index": 0} for shot in shots]


def test_a_broadcast_without_a_continuous_feed_shot_is_not_split():
    from cv.pipeline.shot_boundaries import shots_from_cuts

    signals = _signals(400.0)
    shots = shots_from_cuts(signals, np.array([300]))  # a 370 s shot, but no 20-minute shot
    assert not view_classifier.split_any(view_classifier.view_windows(shots, signals))
