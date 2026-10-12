from __future__ import annotations

from ball import link


def test_link_preserves_original_frame_gaps() -> None:
    detections = [
        (1, [(10.0, 10.0, 2)]),
        (2, [(20.0, 10.0, 2)]),
        (10, [(30.0, 10.0, 2)]),
        (11, [(40.0, 10.0, 2)]),
    ]

    tracks = link(detections, fps=25.0)

    assert all(not ({2, 10} <= {frame for frame, _, _ in track}) for track in tracks)


def test_link_speed_gate_is_fps_normalized() -> None:
    at_25 = [(i, [(10.0 * i, 10.0, 2)]) for i in range(1, 6)]
    at_50 = [(i, [(5.0 * i, 10.0, 2)]) for i in range(1, 6)]

    tracks_25 = link(at_25, fps=25.0)
    tracks_50 = link(at_50, fps=50.0)

    assert len(tracks_25) == len(tracks_50) == 1


def test_link_can_keep_single_neural_detection_without_motion_filter() -> None:
    tracks = link([(1, [(10.0, 20.0, 1)])], 50.0, 1, 0.0, 0.0)

    assert tracks == [[(1, 10.0, 20.0)]]
