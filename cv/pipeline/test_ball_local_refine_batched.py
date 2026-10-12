from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
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


def test_shared_decode_models_yield_each_models_own_rows(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    rng = np.random.default_rng(0)
    clip = tmp_path / "pt0001"
    clip.mkdir()
    for frame in range(1, 9):
        cv2.imwrite(
            str(clip / f"f_{frame:04d}.jpg"),
            rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8),
        )
    regions = [
        RecoveryRegion("pt0001", frame, 300.0 + 20 * frame, 200.0, 256.0, 144.0, "test")
        for frame in range(1, 9)
    ]

    class Heat(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.conv = torch.nn.Conv2d(9, 3, 3, padding=1)

        def forward(self, x):
            return {0: self.conv(x)}

    models = [Heat(1).cuda().eval(), Heat(2).cuda().eval()]
    kwargs = dict(batch_size=6, decode_workers=2, prefetch_batches=2)
    size = res.FrameSize(960, 540)
    alone = [
        [
            row
            for rows in batched.infer_region_batches(m, tmp_path, regions, 0, size, **kwargs)
            for row in rows
        ]
        for m in models
    ]
    together = [[], []]
    for per_model in batched.infer_region_batches_multi(
        models, tmp_path, regions, 0, size, **kwargs
    ):
        for index, rows in enumerate(per_model):
            together[index].extend(rows)
    assert together == alone
    assert all(alone)
