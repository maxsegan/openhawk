"""TENNIS_EVENT_CROPS_DECODE=nvjpeg decodes exactly what cv2.imread does (explicit GPU only)."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

FRAMES = Path(
    os.environ.get("TENNIS_DATA_ROOT", "data"),
    "processed/gpu-crops-20261001/croot/atp_2020_580_r16_212_rafael_nadal_nick_kyrgios/"
    "audit_frames_native_1080/pt0105",
)


@pytest.mark.skipif(
    "TENNIS_EVENT_CROPS_DEVICE" not in os.environ or not FRAMES.is_dir(),
    reason="needs an explicit GPU (TENNIS_EVENT_CROPS_DEVICE) and the copied pilot frames",
)
def test_gpu_decoded_frames_equal_cv2(monkeypatch):
    from cv.pipeline import event_crops

    monkeypatch.setenv(event_crops.DECODE_BACKEND, "nvjpeg")
    paths = sorted(FRAMES.glob("f_*.jpg"))[:150] + [FRAMES / "f_9999.jpg"]  # last one absent
    decoded = event_crops._decode_frames(paths)
    assert decoded[-1] is None
    for path, image in zip(paths[:-1], decoded[:-1], strict=True):
        assert np.array_equal(image, cv2.imread(str(path), cv2.IMREAD_COLOR))


def test_default_decode_is_cv2(monkeypatch):
    from cv.pipeline import event_crops

    monkeypatch.delenv(event_crops.DECODE_BACKEND, raising=False)
    assert event_crops._gpu_decoder() is None
