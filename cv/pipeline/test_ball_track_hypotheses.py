import numpy as np

from ball_track_hypotheses import (
    HypothesisConfig,
    assess_local_track,
    build_continuous_hypothesis_layers,
    build_continuous_hypotheses,
    enforce_speed_safety,
    evidence_tier,
    hypothesis_segments,
    layer_segments,
)


def linear_track(start: int, end: int, x_offset: float = 0.0) -> np.ndarray:
    frames = np.arange(start, end + 1, dtype=float)
    return np.column_stack([frames, frames * 4.0 + x_offset, frames * 2.0])


def test_assess_local_track_requires_explicit_cadence() -> None:
    try:
        assess_local_track(linear_track(1, 20), 10, 0.0, require_motion=False)
    except ValueError as error:
        assert "fps" in str(error)
    else:
        raise AssertionError("invalid cadence accepted")


def test_evidence_tier_keeps_strict_and_marginal_separate() -> None:
    assert evidence_tier({"usable": True, "reason": "ok", "fit_rmse": 5.0}, None) == "strict"
    assert (
        evidence_tier(
            {"usable": False, "reason": "not_ballistic", "fit_rmse": 6.4},
            None,
        )
        == "marginal"
    )
    assert (
        evidence_tier(
            {"usable": False, "reason": "not_ballistic", "fit_rmse": 7.0},
            2.0,
        )
        == "agreement"
    )


def test_continuous_hypotheses_fill_without_event_frames() -> None:
    primary = np.vstack([linear_track(1, 8), linear_track(13, 20)])
    alternate = linear_track(1, 20)
    motion = linear_track(1, 20)
    rows = build_continuous_hypotheses(
        {
            "primary": primary,
            "alternate": alternate,
            "motion": motion,
        },
        25.0,
        primary_arm="primary",
        evidence_arm="motion",
        arm_priority=("primary", "alternate", "motion"),
        frame_start=1,
        frame_end=20,
    )

    assert rows[2]["source"] == "primary"
    assert rows[9]["source"] == "alternate"
    assert rows[15]["source"] == "primary"
    assert hypothesis_segments(rows, "primary") == [
        {
            "start_frame": 9,
            "end_frame": 12,
            "source": "alternate",
            "quality_tier": "strict",
        }
    ]


def test_continuous_hypotheses_abstain_on_teleporting_alternate() -> None:
    primary = np.vstack([linear_track(1, 8), linear_track(13, 20)])
    alternate = linear_track(1, 20)
    alternate[8:12, 1] += np.asarray([0.0, 500.0, 0.0, 500.0])
    rows = build_continuous_hypotheses(
        {"primary": primary, "motion": alternate},
        25.0,
        primary_arm="primary",
        evidence_arm="motion",
        arm_priority=("primary", "motion"),
        frame_start=1,
        frame_end=20,
        config=HypothesisConfig(maximum_teleport_rate=0.0),
    )

    assert all(row["source"] == "none" for row in rows[8:12])


def test_continuous_layers_preserve_parallel_coherent_tracks() -> None:
    rows = build_continuous_hypothesis_layers(
        {
            "primary": linear_track(1, 20),
            "alternate": linear_track(1, 20, 2.0),
            "motion": linear_track(1, 20, 2.0),
        },
        25.0,
        evidence_arm="motion",
        arm_priority=("primary", "alternate", "motion"),
        frame_start=8,
        frame_end=12,
    )

    at_frame_ten = [row for row in rows if row["frame"] == 10]
    assert [row["arm"] for row in at_frame_ten] == [
        "primary",
        "alternate",
        "motion",
    ]
    assert all(row["quality_tier"] == "strict" for row in at_frame_ten)
    assert layer_segments(rows) == [
        {
            "start_frame": 8,
            "end_frame": 12,
            "arm": "primary",
            "quality_tier": "strict",
        },
        {
            "start_frame": 8,
            "end_frame": 12,
            "arm": "alternate",
            "quality_tier": "strict",
        },
        {
            "start_frame": 8,
            "end_frame": 12,
            "arm": "motion",
            "quality_tier": "strict",
        },
    ]


def test_speed_safety_rejects_entire_disconnected_proposal_run() -> None:
    rows = [
        {
            "frame": 1,
            "source": "primary",
            "quality_tier": "strict",
            "x": 0.0,
            "y": 0.0,
            "proposal_rejection": None,
        },
        {
            "frame": 2,
            "source": "alternate",
            "quality_tier": "strict",
            "x": 500.0,
            "y": 0.0,
            "proposal_rejection": None,
        },
        {
            "frame": 3,
            "source": "alternate",
            "quality_tier": "strict",
            "x": 510.0,
            "y": 0.0,
            "proposal_rejection": None,
        },
        {
            "frame": 4,
            "source": "primary",
            "quality_tier": "strict",
            "x": 30.0,
            "y": 0.0,
            "proposal_rejection": None,
        },
    ]

    filtered = enforce_speed_safety(
        rows,
        25.0,
        "primary",
        maximum_speed_px_s=4000.0,
    )

    assert [row["source"] for row in filtered] == [
        "primary",
        "none",
        "none",
        "primary",
    ]
    assert filtered[1]["proposal_rejection"] == "speed_unsafe"
