import json

import numpy as np

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
