from __future__ import annotations

from ball_track_consensus import (
    Candidate,
    DecoderConfig,
    config_for_consumer,
    interpolate_short_gaps,
    merge_candidates,
    select_consistent_path,
    suppress_static_hotspots,
)


def test_config_scales_frame_windows_and_exposes_consumer_tradeoff() -> None:
    integrity = config_for_consumer(25.0, "integrity")
    availability = config_for_consumer(25.0, "availability")

    assert integrity.interpolation_gap == 6
    assert integrity.reacquisition_reset_gap == 6
    assert integrity.hotspot_consecutive_frames == 3
    assert integrity.hotspot_motion_reach == 12
    assert integrity.hotspot_motion_step == 70.0
    assert availability.missing_cost > integrity.missing_cost
    assert availability.restart_cost < integrity.restart_cost


def candidate(x: float, score: float = 0.6, rank: int = 0, source: str = "a") -> Candidate:
    return Candidate(x, 100.0, score, rank, (source,))


def test_decoder_rejects_one_frame_high_score_teleport() -> None:
    frames = {}
    for frame in range(1, 13):
        real = candidate(100.0 + 8.0 * frame, score=0.55, rank=1)
        frames[frame] = [real]
        if frame == 6:
            frames[frame].insert(0, candidate(700.0, score=0.99))
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert selected[6] is not None
    assert selected[6].x == 148.0


def test_decoder_allows_position_continuous_velocity_kink() -> None:
    frames = {}
    for frame in range(1, 11):
        x = 100.0 + 10.0 * frame if frame <= 5 else 150.0 - 9.0 * (frame - 5)
        frames[frame] = [candidate(x)]
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert [selected[frame].x for frame in range(1, 11)] == [
        frames[frame][0].x for frame in range(1, 11)
    ]


def test_cross_model_agreement_beats_single_model_distractor() -> None:
    config = DecoderConfig()
    merged = merge_candidates(
        [
            ("wasb", candidate(200.0, score=0.5, source="wasb")),
            ("tracknet", candidate(203.0, score=0.45, source="tracknet")),
            ("wasb", candidate(600.0, score=0.95, source="wasb")),
        ],
        config,
    )
    agreed = min(merged, key=lambda item: abs(item.x - 200.0))
    assert agreed.sources == ("tracknet", "wasb")


def test_decoder_can_abstain_through_short_gap() -> None:
    frames = {
        frame: [candidate(100.0 + 5.0 * frame)] for frame in range(1, 9) if frame not in {4, 5}
    }
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert selected[4] is None
    assert selected[5] is None


def test_decoder_does_not_force_teleport_after_three_missing_frames() -> None:
    frames = {
        frame: [candidate(100.0 + 5.0 * frame)]
        for frame in range(1, 16)
        if frame not in {6, 7, 8, 9}
    }
    for frame in range(6, 10):
        frames[frame] = [candidate(800.0, score=0.99)]
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert all(selected[frame] is None for frame in range(6, 10))
    assert selected[10] is not None
    assert selected[10].x == 150.0


def test_decoder_can_remain_missing_indefinitely() -> None:
    frames = {
        1: [candidate(100.0)],
        **{frame: [candidate(800.0, score=0.99)] for frame in range(2, 40)},
    }
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert all(selected[frame] is None for frame in range(2, 40))


def test_decoder_reacquires_multidetector_path_after_long_gap() -> None:
    frames = {
        1: [candidate(100.0)],
        **{
            frame: [
                Candidate(
                    600.0 + 5.0 * (frame - 10),
                    100.0,
                    0.6,
                    0,
                    ("tracknet", "wasb"),
                )
            ]
            for frame in range(10, 16)
        },
    }

    selected = dict(select_consistent_path(frames, fps=50.0))

    assert selected[9] is None
    assert selected[10] is not None
    assert selected[10].x == 600.0


def test_decoder_prefers_moving_track_to_static_high_score_distractor() -> None:
    frames = {}
    for frame in range(1, 21):
        frames[frame] = [
            candidate(40.0, score=0.99),
            candidate(100.0 + 6.0 * frame, score=0.55, rank=1),
        ]
    selected = dict(select_consistent_path(frames, fps=50.0))
    assert selected[20] is not None
    assert selected[20].x == 220.0


def test_interpolation_fills_only_bounded_short_gaps() -> None:
    selected = [
        (1, candidate(90.0)),
        (2, candidate(100.0)),
        (3, None),
        (4, None),
        (5, candidate(130.0)),
        (6, candidate(140.0)),
        (7, None),
    ]
    filled = dict(interpolate_short_gaps(selected, maximum_gap=2))
    assert filled[3] is not None
    assert filled[3].x == 110.0
    assert filled[3].sources == ("interpolated",)
    assert filled[4] is not None
    assert filled[4].x == 120.0
    assert filled[7] is None


def test_interpolation_rejects_switch_between_stationary_objects() -> None:
    selected = [
        (1, candidate(100.0)),
        (2, candidate(100.0)),
        (3, None),
        (4, None),
        (5, candidate(180.0)),
        (6, candidate(180.0)),
    ]
    filled = dict(interpolate_short_gaps(selected, maximum_gap=2))
    assert filled[3] is None
    assert filled[4] is None


def test_persistent_detector_hotspot_is_removed() -> None:
    frames = {}
    for frame in range(1, 41):
        frames[frame] = [
            candidate(40.0, score=0.99, source="wasb"),
            candidate(100.0 + 6.0 * frame, score=0.55, rank=1, source="wasb"),
        ]
    filtered, hotspots = suppress_static_hotspots(frames, DecoderConfig())
    assert hotspots
    assert all(all(item.x != 40.0 for item in candidates) for candidates in filtered.values())
    assert all(any(item.x != 40.0 for item in candidates) for candidates in filtered.values())


def test_short_consecutive_static_run_is_removed() -> None:
    frames = {frame: [candidate(40.0, score=0.99, source="wasb")] for frame in range(1, 7)}
    filtered, hotspots = suppress_static_hotspots(frames, DecoderConfig())
    assert hotspots
    assert all(not candidates for candidates in filtered.values())


def test_stationary_apex_with_consensus_motion_is_retained() -> None:
    frames = {}
    for frame in range(1, 31):
        x = 100.0 + 5.0 * min(frame, 10)
        if 11 <= frame <= 18:
            x = 150.0
        elif frame > 18:
            x = 150.0 + 5.0 * (frame - 18)
        frames[frame] = [
            Candidate(x, 100.0, 0.6, 0, ("tracknet", "wasb")),
        ]

    filtered, hotspots = suppress_static_hotspots(frames, DecoderConfig())

    assert hotspots
    assert all(filtered[frame] for frame in range(11, 19))


def test_persistent_multidetector_hotspot_without_motion_is_removed() -> None:
    frames = {
        frame: [Candidate(40.0, 100.0, 0.9, 0, ("tracknet", "wasb"))] for frame in range(1, 31)
    }

    filtered, hotspots = suppress_static_hotspots(frames, DecoderConfig())

    assert hotspots
    assert all(not candidates for candidates in filtered.values())
