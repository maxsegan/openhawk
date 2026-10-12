import numpy as np

from cv.pipeline.shot_boundaries import (
    BINS,
    GRID,
    Signals,
    _grid_histograms,
    detect_cuts,
    shots_from_cuts,
)


def _signals(distance: np.ndarray, fps: float = 25.0) -> Signals:
    motion = np.full(len(distance), 1.0, dtype=np.float32)
    return Signals(
        fps=fps,
        start_seconds=100.0,
        distance=distance.astype(np.float32),
        motion=motion,
        motion_top=motion,
        motion_bottom=motion,
    )


def test_grid_histograms_are_normalised_per_cell() -> None:
    frames = np.random.default_rng(0).integers(0, 256, size=(3, 18, 24), dtype=np.uint8)

    histograms = _grid_histograms(frames)

    assert histograms.shape == (3, GRID * GRID * BINS)
    per_cell = histograms.reshape(3, GRID * GRID, BINS).sum(axis=2)
    assert np.allclose(per_cell, 1.0)


def test_detect_cuts_finds_isolated_peaks_and_ignores_drift() -> None:
    distance = np.full(1000, 0.02, dtype=np.float32)
    distance[400:430] = 0.15  # a fast pan: elevated but not a peak
    for index in (200, 600, 850):
        distance[index] = 0.7

    cuts = detect_cuts(_signals(distance))

    assert cuts.tolist() == [200, 600, 850]


def test_detect_cuts_collapses_a_two_frame_blend_into_one_cut() -> None:
    distance = np.full(600, 0.02, dtype=np.float32)
    distance[300] = 0.6
    distance[301] = 0.45

    cuts = detect_cuts(_signals(distance))

    assert cuts.tolist() == [300]


def test_shots_cover_the_span_without_gaps() -> None:
    distance = np.full(500, 0.02, dtype=np.float32)
    distance[250] = 0.8
    signals = _signals(distance)

    shots = shots_from_cuts(signals, detect_cuts(signals))

    assert len(shots) == 2
    assert shots[0]["t_start"] == 100.0
    assert shots[0]["t_end"] == shots[1]["t_start"]
    assert shots[1]["t_end"] == 100.0 + 500 / 25.0


def test_signal_accumulator_does_not_depend_on_chunking() -> None:
    from cv.pipeline.shot_boundaries import SIGNAL_HEIGHT, SIGNAL_WIDTH, _SignalAccumulator

    frames = np.random.default_rng(1).integers(
        0, 256, size=(700, SIGNAL_HEIGHT, SIGNAL_WIDTH), dtype=np.uint8
    )
    whole, pieces = _SignalAccumulator(), _SignalAccumulator()
    whole.add(frames)
    for start in range(0, len(frames), 256):
        pieces.add(frames[start : start + 256])

    a, b = whole.signals(25.0, 0.0), pieces.signals(25.0, 0.0)
    for field in ("distance", "motion", "motion_top", "motion_bottom"):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    assert len(a.distance) == len(frames) and a.distance[0] == 0.0


def test_gpu_grey_matches_the_bt709_to_bt601_grey_of_flat_colours() -> None:
    import torch

    from cv.pipeline.shot_boundaries import gpu_grey

    height, width = 108, 192
    colours = [(16, 128, 128), (235, 128, 128), (126, 128, 128), (81, 90, 240), (145, 54, 34)]
    frames = torch.empty((len(colours), height * 3 // 2, width), dtype=torch.uint8)
    for index, (y, u, v) in enumerate(colours):
        frames[index, :height] = y
        chroma = frames[index, height:].view(height // 2, width // 2, 2)
        chroma[..., 0], chroma[..., 1] = u, v

    grey = gpu_grey(frames, height, width).numpy()

    assert grey.shape == (len(colours), 54, 96)
    for index, (y, u, v) in enumerate(colours):
        luma, cb, cr = (y - 16) * 255 / 219, (u - 128) * 255 / 224, (v - 128) * 255 / 224
        rgb = np.clip(
            [luma + 1.5748 * cr, luma - 0.1873 * cb - 0.4681 * cr, luma + 1.8556 * cb], 0, 255
        )
        expected = round(float(np.dot([0.299, 0.587, 0.114], rgb)))
        assert np.abs(grey[index].astype(int) - expected).max() <= 1


def test_gpu_signals_leave_spans_to_ffmpeg() -> None:
    from cv.pipeline.shot_boundaries import gpu_signals

    assert gpu_signals("missing.mkv", 25.0, 0, start_seconds=10.0) is None
    assert gpu_signals("missing.mkv", 25.0, 0, duration_seconds=5.0) is None
