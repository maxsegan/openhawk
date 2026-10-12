"""gpu_jpeg must reproduce cv2.imread byte for byte (needs a CUDA card with NVJPG engines)."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

#: The card the hardware tests use; unset skips them, so an unpinned suite never lands on GPU0.
DEVICE = os.environ.get("TENNIS_GPU_JPEG_TEST_DEVICE")


def _decoder():
    if DEVICE is None or not torch.cuda.is_available():
        pytest.skip("set TENNIS_GPU_JPEG_TEST_DEVICE to run on a CUDA card")
    from cv.experiments.gpu_decode.gpu_jpeg import NvJpegDecoder

    try:
        return NvJpegDecoder(int(DEVICE), batch_size=4)
    except (OSError, RuntimeError) as error:
        pytest.skip(f"no nvJPEG hardware backend: {error}")


def _picture(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 256, (9, 13, 3), dtype=np.uint8), (width, height))
    noise = rng.integers(-40, 41, (height, width, 3))
    return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("height", "width", "sampling"),
    [
        (1080, 1920, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
        (67, 101, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),  # odd: edge replication and cropping
        (64, 96, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
    ],
)
def test_decode_equals_cv2(tmp_path, height, width, sampling):
    decoder = _decoder()
    paths = []
    for seed in range(3):
        path = tmp_path / f"f_{seed}.jpg"
        cv2.imwrite(
            str(path),
            _picture(height, width, seed),
            [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling],
        )
        paths.append(str(path))
    for path, gpu in zip(paths, decoder.decode_paths(paths), strict=True):
        assert np.array_equal(gpu, cv2.imread(path))
    assert decoder.stats.fallback == 0


def test_unsupported_subsampling_falls_back_to_cv2(tmp_path):
    decoder = _decoder()
    path = tmp_path / "f_422.jpg"
    cv2.imwrite(
        str(path),
        _picture(48, 64, 7),
        [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422],
    )
    (gpu,) = decoder.decode_paths([str(path)])
    assert np.array_equal(gpu, cv2.imread(str(path)))
    assert decoder.stats.fallback == 1


def test_open_decoder_falls_back_when_the_hardware_backend_is_refused(monkeypatch, capsys):
    from cv.experiments.gpu_decode import gpu_jpeg

    def refuse(*args, **kwargs):
        raise RuntimeError("nvJPEG create (hardware backend) failed with status 3")

    monkeypatch.setattr(gpu_jpeg, "NvJpegDecoder", refuse)
    assert gpu_jpeg.open_decoder(0, batch_size=4) is None
    assert "decoding with cv2" in capsys.readouterr().err


def test_several_full_batches_in_one_call_stay_exact(tmp_path):
    """Same-size batches in one call must not overwrite each other's pictures (stream sync).

    Without the sync around nvjpegDecodeBatched, two full 32-picture batches in one call
    corrupted a few pictures of the first batch on the hardware backend (timing-dependent;
    seen within 2 tries on 64 synthetic 1080p frames, 20-40 of 64 real frames per call)."""
    if DEVICE is None or not torch.cuda.is_available():
        pytest.skip("set TENNIS_GPU_JPEG_TEST_DEVICE to run on a CUDA card")
    from cv.experiments.gpu_decode.gpu_jpeg import NvJpegDecoder

    try:
        decoder = NvJpegDecoder(
            int(DEVICE)
        )  # default batch of 32: 64 pictures are two full batches
    except (OSError, RuntimeError) as error:
        pytest.skip(f"no nvJPEG hardware backend: {error}")
    paths = []
    for seed in range(64):
        path = tmp_path / f"f_{seed:04d}.jpg"
        cv2.imwrite(
            str(path),
            _picture(1080, 1920, seed),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
            ],
        )
        paths.append(str(path))
    expected = [cv2.imread(path) for path in paths]
    for _ in range(4):
        decoded = decoder.decode_paths(paths)
        assert all(np.array_equal(a, b) for a, b in zip(decoded, expected, strict=True))
