import numpy as np
import pytest

from play_phase import detect_phase, rally_activity


def trajectory(fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, 8.0, 1.0 / fps)
    frames = times * fps
    rally_time = np.clip(times - 2.0, 0.0, 4.0)
    active = (times >= 2.0) & (times <= 6.0)
    x = 480.0 + 35.0 * np.sin(2.0 * np.pi * rally_time)
    y = np.where(
        active,
        360.0 + 180.0 * np.sin(2.0 * np.pi * rally_time),
        360.0,
    )
    return frames, np.column_stack((x, y)), np.full(len(frames), 360.0)


@pytest.mark.parametrize("fps", [24.0, 25.0, 50.0, 59.94])
def test_detect_phase_is_cadence_equivalent(fps: float) -> None:
    frames, points, net_row = trajectory(fps)

    result = detect_phase(frames, points, net_row, fps=fps)

    assert result.point_valid
    assert len(result.spans) == 1
    start, end = result.spans[0]
    assert start / fps == pytest.approx(1.65, abs=0.06)
    assert end / fps == pytest.approx(6.42, abs=0.06)
    assert np.array_equal(
        result.in_play,
        (frames >= start) & (frames <= end),
    )


def test_rally_activity_matches_across_cadences() -> None:
    reference_frames, reference_points, reference_net = trajectory(50.0)
    reference = rally_activity(
        reference_frames,
        reference_points,
        reference_net,
        fps=50.0,
    )
    reference_times = reference_frames / 50.0

    for fps in (24.0, 25.0, 59.94):
        frames, points, net_row = trajectory(fps)
        activity = rally_activity(frames, points, net_row, fps=fps)
        aligned = np.interp(reference_times, frames / fps, activity)
        assert np.mean(np.abs(reference - aligned)) < 0.04


def test_detect_phase_requires_explicit_valid_cadence() -> None:
    frames, points, net_row = trajectory(25.0)

    with pytest.raises(ValueError, match="fps must be positive"):
        detect_phase(frames, points, net_row, fps=0.0)
