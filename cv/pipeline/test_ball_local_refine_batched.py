from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_local_refine_batched as batched
import resolution as res
from ball_local_refine import RecoveryRegion


def test_prepare_batch_reuses_overlapping_temporal_crops(monkeypatch) -> None:
    region = RecoveryRegion("pt0001", 10, 100.0, 100.0, 80.0, 45.0, "test")
    calls = []

    def fake_crop(frame, region, artifact_size):
        calls.append(int(frame[0, 0, 0]))
        return np.full((3, 2, 2), frame[0, 0, 0], dtype=np.float32)

    monkeypatch.setattr(batched.baseline, "_crop_tensor", fake_crop)
    samples = batched._prepare_batch(
        [region],
        batched._contexts(True),
        {"pt0001": 20},
        lambda clip, frame: np.full((1, 1, 3), frame, dtype=np.uint8),
        res.CANONICAL_SIZE,
    )

    assert samples.shape == (3, 9, 2, 2)
    assert calls == [8, 9, 10, 11, 12]
    assert samples[:, ::3, 0, 0].tolist() == [
        [8.0, 9.0, 10.0],
        [9.0, 10.0, 11.0],
        [10.0, 11.0, 12.0],
    ]


def test_uint8_crop_normalizes_to_the_legacy_tensor() -> None:
    frame = np.arange(120 * 200 * 3, dtype=np.uint8).reshape(120, 200, 3)
    region = RecoveryRegion("pt0001", 10, 80.5, 55.25, 90.0, 50.0, "test")

    crop = batched._crop_uint8(frame, region, res.FrameSize(200, 120))
    normalized = (crop.astype(np.float32) / 255.0 - batched.baseline.MEAN[:, None, None]) / (
        batched.baseline.STD[:, None, None]
    )

    assert crop.dtype == np.uint8
    assert np.array_equal(
        normalized,
        batched.baseline._crop_tensor(frame, region, res.FrameSize(200, 120)),
    )


def test_resident_budget_fails_closed_above_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(batched, "resident_bytes", lambda: 101)
    budget = batched.ResidentMemoryBudget(100)

    with pytest.raises(MemoryError, match="max-resident-gb"):
        budget.sample()


def test_resident_budget_backpressures_until_capacity_is_available(monkeypatch) -> None:
    readings = iter((90, 90, 70))
    sleeps = []
    monkeypatch.setattr(batched, "resident_bytes", lambda: next(readings))
    monkeypatch.setattr(batched.time, "sleep", sleeps.append)
    budget = batched.ResidentMemoryBudget(100)

    budget.wait_for_capacity(20, threading.Event())

    assert sleeps == [0.05, 0.05]
    assert budget.peak_bytes == 90
