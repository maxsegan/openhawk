import importlib

import numpy as np
import pytest

from cv.pipeline import ball_neural_batched


def test_batched_tracker_is_importable_by_canonical_module_name() -> None:
    module = importlib.import_module("cv.pipeline.ball_neural_batched")

    assert callable(module.infer_clip)


def test_empty_clip_abstains_without_allocating_workers() -> None:
    assert ball_neural_batched.infer_clip(None, [], 32, 0) == []


@pytest.mark.parametrize("implementation", [ball_neural_batched.baseline, ball_neural_batched])
@pytest.mark.parametrize("temporal_ensemble", [False, True])
def test_long_frame_ids_survive_neural_inference(monkeypatch, implementation, temporal_ensemble):
    import torch

    paths = [f"/frames/f_{n:04d}.jpg" for n in (9999, 10000, 10001)]

    def preprocess(_path):
        return (
            np.zeros((3, 4, 4), np.float32),
            np.float32([[1, 0, 0], [0, 1, 0]]),
            ball_neural_batched.res.FrameSize(4, 4),
        )

    def model(inputs):
        return (torch.zeros((len(inputs), 3, 4, 4)),)

    monkeypatch.setattr(ball_neural_batched.baseline, "preprocess", preprocess)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    rows = implementation.infer_clip(model, paths, 9, 0, temporal_ensemble=temporal_ensemble)
    assert [row["frame_id"] for row in rows] == [9999, 10000, 10001]


def test_nonensemble_wrapper_delegates_without_duplicate_preprocessing(monkeypatch):
    calls = []
    monkeypatch.setattr(ball_neural_batched.baseline, "preprocess", lambda path: calls.append(path))
    result = [{"frame_id": 1}]
    monkeypatch.setattr(ball_neural_batched, "_BASELINE_INFER_CLIP", lambda *args, **kwargs: result)
    assert ball_neural_batched.infer_clip(None, ["f_0001.jpg"], 1, 0) is result
    assert calls == [], "the delegated baseline is responsible for its own preprocessing"
