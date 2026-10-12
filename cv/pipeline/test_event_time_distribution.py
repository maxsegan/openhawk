"""The additive local timing distribution is either the real thing or absent."""

import numpy as np
import pytest

from cv.pipeline import event_time_distribution as timing


def _pmf(rows=3):
    values = np.random.default_rng(0).random((rows, timing.TIME_BINS)).astype(np.float32)
    return values / values.sum(axis=1, keepdims=True)


def _artifact(rows=3):
    return {
        "frames": np.arange(rows),
        timing.PMF_KEY: _pmf(rows),
        timing.GRID_KEY: np.tile(timing.expected_grid(), (rows, 1)),
    }


def test_an_artifact_without_the_distribution_declares_it_unavailable():
    data = {"frames": np.arange(3), "time_offset": np.zeros(3)}
    assert timing.present(data) is False
    assert timing.status(data) == "unavailable_in_prediction_artifact"
    assert timing.arrays(data) is None


@pytest.mark.parametrize("key", timing.KEYS)
def test_half_the_distribution_is_a_malformed_artifact_not_an_old_one(key):
    data = _artifact()
    del data[key]
    with pytest.raises(ValueError, match="needs both"):
        timing.present(data)


def test_a_present_distribution_validates_and_reports_its_status():
    data = _artifact()
    pmf, grid = timing.arrays(data, rows=3)
    assert timing.status(data) == timing.AVAILABLE
    assert pmf.shape == grid.shape == (3, 17)
    assert np.allclose(grid[0], np.arange(-8, 9))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (timing.PMF_KEY, np.zeros((3, 9), dtype=np.float32), "must be \\(rows, 17\\)"),
        (timing.PMF_KEY, np.zeros((3, 17), dtype=np.float32), "must be normalised"),
        (timing.GRID_KEY, np.zeros((3, 17), dtype=np.float32), "strictly increasing"),
        (
            timing.GRID_KEY,
            np.tile(np.arange(-16, 18, 2, dtype=np.float32), (3, 1)),
            "not the exact model grid",
        ),
        (timing.GRID_KEY, np.tile(timing.expected_grid(), (2, 1)), "row-aligned"),
    ],
)
def test_malformed_distributions_are_rejected(field, value, message):
    data = _artifact()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        timing.arrays(data)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.5])
def test_non_finite_or_negative_scores_are_rejected(bad):
    data = _artifact()
    data[timing.PMF_KEY] = data[timing.PMF_KEY].copy()
    data[timing.PMF_KEY][1, 3] = bad
    with pytest.raises(ValueError, match="finite and non-negative|normalised"):
        timing.arrays(data)


def test_a_row_count_mismatch_with_the_rest_of_the_artifact_is_rejected():
    with pytest.raises(ValueError, match="expected 5"):
        timing.arrays(_artifact(3), rows=5)


def test_the_record_keeps_the_grid_and_declares_what_the_scores_are_not():
    data = _artifact()
    pmf, grid = timing.arrays(data)
    record = timing.record(pmf[1], grid[1])
    assert record["schema"] == "event_time_distribution_v1"
    assert record["source"] == "event_video_model_time_softmax"
    assert record["calibrated"] is False
    assert record["offset_reference"] == "native_crop_candidate_frame"
    assert record["offset_frames"] == [float(v) for v in range(-8, 9)]
    assert record["probabilities"] == pytest.approx(pmf[1].tolist())
    assert sum(record["probabilities"]) == pytest.approx(1.0)
