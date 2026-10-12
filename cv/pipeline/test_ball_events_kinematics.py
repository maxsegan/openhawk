import numpy as np

from cv.pipeline.ball_events import _line_fit


def test_line_fit_collapses_duplicate_resampled_frames():
    line_x, line_y, origin = _line_fit(
        [10.0, 10.0, 11.0],
        [2.0, 4.0, 8.0],
        [5.0, 7.0, 9.0],
    )

    assert origin == 10.0
    assert np.all(np.isfinite(line_x))
    assert np.all(np.isfinite(line_y))
    assert np.allclose(np.polyval(line_x, [0.0, 1.0]), [3.0, 8.0])
    assert np.allclose(np.polyval(line_y, [0.0, 1.0]), [6.0, 9.0])
