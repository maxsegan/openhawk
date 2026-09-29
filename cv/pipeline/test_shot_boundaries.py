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
