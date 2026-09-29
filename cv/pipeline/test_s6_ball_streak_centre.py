import pytest

from cv.pipeline import s6_ball_streak_centre as streak
from cv.pipeline.s6_automatic_observations import native_ball_support
from cv.pipeline.resolution import NATIVE_SIZE


def _row(frame, x, y, status="visible"):
    return {"frame": frame, "status": status, "x1080": x, "y1080": y}


def test_off_returns_the_rows_untouched():
    rows = [_row(1, 10.0, 10.0), _row(2, 40.0, 10.0)]
    out, receipt = streak.apply(rows, "off")
    assert out is rows and receipt is None


def test_on_moves_back_along_the_previous_step():
    rows = [_row(1, 100.0, 50.0), _row(2, 130.0, 10.0), _row(3, 160.0, -30.0)]
    out, receipt = streak.apply(rows, "on")
    # frame 1 has no previous row, so it uses the step to the next one
    assert out[0]["x1080"] == pytest.approx(100.0 - 0.2 * 30.0)
    assert out[0]["streak_centre"]["velocity_basis"] == "next_frame"
    assert out[1]["x1080"] == pytest.approx(130.0 - 0.2 * 30.0)
    assert out[1]["y1080"] == pytest.approx(10.0 + 0.2 * 40.0)
    assert out[1]["streak_centre"]["raw_xy"] == [130.0, 10.0]
    assert receipt["shifted_rows"] == 3
    assert rows[1]["x1080"] == 130.0  # input not modified


def test_isolated_and_unsupported_rows_stay_put():
    rows = [
        _row(1, 100.0, 50.0),
        _row(2, 500.0, 500.0, status="unsupported"),
        _row(3, 160.0, 50.0),
        _row(5, 300.0, 50.0),
    ]
    out, receipt = streak.apply(rows, "on")
    assert out == rows
    assert receipt["shifted_rows"] == 0


def test_native_support_receipt_names_the_shift():
    rows = [_row(1, 100.0, 50.0), _row(2, 120.0, 50.0)]
    out, receipt = native_ball_support(rows, NATIVE_SIZE, streak_centre="on")
    assert out[1]["x1080"] == pytest.approx(116.0)
    assert receipt["streak_centre"]["gain_frames"] == streak.GAIN_FRAMES
    _, off = native_ball_support(rows, NATIVE_SIZE)
    assert "streak_centre" not in off


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        streak.apply([], "maybe")
