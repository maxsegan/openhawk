from __future__ import annotations

import numpy as np

from cv.pipeline.ball_motion_tracker import (
    Geometry,
    MotionConfig,
    Observation,
    _apply_h,
    camera_warp,
    crop_first_config,
    ground_homography_from_projection,
    measurement_covariance,
    merge_observations,
    suppress_static_hotspots,
    track_clip,
)


def observation(x: float, y: float, source: str, *, score: float = 0.8) -> Observation:
    return Observation(x, y, score, 0, (source,))


def test_ground_homography_round_trips_projection_ground_plane() -> None:
    projection = np.asarray(
        [[100.0, 0.0, 2.0, 20.0], [0.0, 80.0, -30.0, 10.0], [0.0, 0.0, 0.0, 1.0]]
    )
    homography = ground_homography_from_projection(projection)
    image = projection[:, [0, 1, 3]] @ np.asarray([2.0, 3.0, 1.0])
    image = image[:2] / image[2]

    court = _apply_h(homography, image)

    np.testing.assert_allclose(court, [2.0, 3.0])


def test_camera_warp_compensates_frame_pan() -> None:
    previous = np.eye(3)
    current = np.asarray([[1.0, 0.0, -10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    warped = _apply_h(camera_warp(previous, current), np.asarray([100.0, 50.0]))

    np.testing.assert_allclose(warped, [110.0, 50.0])


def test_merge_observations_preserves_detector_and_crop_provenance() -> None:
    merged = merge_observations(
        [
            observation(100.0, 100.0, "sliding_wasb"),
            Observation(
                103.0,
                101.0,
                0.7,
                1,
                ("branched_crop_tracknetv2",),
                ("persistent_lock_observed",),
            ),
        ],
        radius=8.0,
    )

    assert len(merged) == 1
    assert merged[0].detector_count == 2
    assert merged[0].is_crop
    assert merged[0].crop_provenance == ("persistent_lock_observed",)


def test_merge_keeps_coarse_guide_separate_from_detector_cluster() -> None:
    merged = merge_observations(
        [
            observation(100.0, 100.0, "coarse_lock"),
            observation(101.0, 100.0, "sliding_wasb"),
            observation(102.0, 100.0, "sliding_tracknetv2"),
        ],
        radius=8.0,
    )

    assert len(merged) == 2
    assert sum(item.is_guide for item in merged) == 1


def test_motion_filter_rejects_high_score_wrong_object_jump() -> None:
    frames = {}
    for frame, x in enumerate((100.0, 108.0, 116.0, 124.0, 132.0), start=1):
        frames[frame] = [
            observation(x, 100.0, "sliding_wasb", score=0.5),
            observation(x + 1.0, 100.5, "sliding_tracknetv2", score=0.5),
            observation(300.0, 280.0, "branched_crop_wasb", score=0.99),
        ]
    geometry = Geometry(
        {("pt0001", frame): np.eye(3) for frame in frames},
        {},
        "synthetic",
    )

    rows, _ = track_clip("pt0001", frames, geometry, 50.0)

    assert len(rows) == 5
    assert all(float(row["x"]) * 2.0 < 150.0 for row in rows)
    assert all(float(row["innovation_cov_xx_native"]) > 0.0 for row in rows)


def test_coarse_lock_can_bootstrap_without_detector_agreement() -> None:
    frames = {
        frame: [observation(100.0 + 5.0 * frame, 100.0, "coarse_lock")] for frame in range(1, 6)
    }
    geometry = Geometry(
        {("pt0001", frame): np.eye(3) for frame in frames},
        {},
        "synthetic",
    )

    rows, _ = track_clip("pt0001", frames, geometry, 50.0)

    assert len(rows) == 5


def test_coarse_lock_reinitializes_after_an_innovation_jump() -> None:
    frames = {
        1: [observation(100.0, 100.0, "coarse_lock")],
        2: [observation(105.0, 100.0, "coarse_lock")],
        3: [observation(500.0, 400.0, "coarse_lock")],
    }
    geometry = Geometry(
        {("pt0001", frame): np.eye(3) for frame in frames},
        {},
        "synthetic",
    )

    rows, _ = track_clip("pt0001", frames, geometry, 50.0)

    assert len(rows) == 3
    assert float(rows[-1]["x"]) * 2.0 == 500.0
    assert rows[-1]["innovation_mahalanobis"] == 0.0


def test_frames_inside_coarse_guide_gap_abstain() -> None:
    frames = {
        1: [observation(100.0, 100.0, "coarse_lock")],
        2: [
            observation(105.0, 100.0, "sliding_wasb"),
            observation(105.5, 100.0, "sliding_tracknetv2"),
        ],
        3: [observation(110.0, 100.0, "coarse_lock")],
    }
    geometry = Geometry(
        {("pt0001", frame): np.eye(3) for frame in frames},
        {},
        "synthetic",
    )

    rows, _ = track_clip("pt0001", frames, geometry, 50.0)

    assert [row["frame"] for row in rows] == ["f_0001.jpg", "f_0003.jpg"]


def test_static_hotspot_is_removed_but_moving_agreement_survives() -> None:
    frames = {}
    for frame in range(1, 21):
        frames[frame] = [
            observation(300.0, 50.0, "far_native_wasb"),
            Observation(
                100.0 + 5.0 * frame,
                100.0,
                0.8,
                0,
                ("sliding_wasb", "sliding_tracknetv2"),
            ),
        ]

    filtered = suppress_static_hotspots(frames, MotionConfig())

    assert all(len(observations) == 1 for observations in filtered.values())
    assert all(observations[0].detector_count == 2 for observations in filtered.values())


def test_motion_filter_emits_impulse_regime_proposal() -> None:
    positions = (100.0, 105.0, 110.0, 115.0, 160.0, 205.0)
    frames = {
        frame: [
            observation(x, 100.0, "sliding_wasb"),
            observation(x + 0.5, 100.0, "sliding_tracknetv2"),
        ]
        for frame, x in enumerate(positions, start=1)
    }
    geometry = Geometry(
        {("pt0001", frame): np.eye(3) for frame in frames},
        {},
        "synthetic",
    )
    config = MotionConfig(impulse_acceleration_noise=30.0)

    _, proposals = track_clip("pt0001", frames, geometry, 50.0, config)

    assert any(proposal["to_regime"] == "impulse" for proposal in proposals)


# --- crop-first association -----------------------------------------------------------------


def _identity_geometry(frames) -> Geometry:
    return Geometry({("pt0001", frame): np.eye(3) for frame in frames}, {}, "synthetic")


def test_default_association_is_unchanged() -> None:
    """Crop-first must be opt-in: the production defaults keep their measured values."""
    config = MotionConfig()

    assert config.crop_first is False
    assert config.crop_prior_bonus == 0.25
    assert config.guide_prior_bonus == 1.4
    assert config.guide_replacement_margin == 2.0
    assert config.guide_candidate_gate_native == 4.0
    assert config.crop_measurement_std_native == 2.5
    assert config.guide_measurement_std_native == 3.0


def test_crop_first_config_puts_the_crop_ahead_of_the_guide() -> None:
    config = crop_first_config()

    assert config.crop_first is True
    assert config.crop_prior_bonus > config.guide_prior_bonus
    assert config.guide_replacement_margin == 0.0
    assert config.crop_candidate_gate_native > config.guide_candidate_gate_native
    # The sigma ratio is the measured accuracy ratio, so the crop must be the tighter one.
    assert config.crop_measurement_std_native < config.guide_measurement_std_native


def test_crop_first_measurement_covariance_follows_the_config() -> None:
    crop = Observation(0.0, 0.0, 0.9, 0, ("branched_crop_wasb",))
    guide = observation(0.0, 0.0, "coarse_lock")

    default_crop = measurement_covariance(crop)[0, 0]
    first_crop = measurement_covariance(crop, crop_first_config())[0, 0]

    assert first_crop < default_crop
    assert first_crop < measurement_covariance(guide, crop_first_config())[0, 0]


def test_lock_is_crop_gives_the_guide_stream_the_crop_sigma() -> None:
    guide = observation(0.0, 0.0, "coarse_lock")
    config = crop_first_config(lock_is_crop=True)

    assert (
        measurement_covariance(guide, config)[0, 0]
        == measurement_covariance(Observation(0.0, 0.0, 0.9, 0, ("branched_crop_wasb",)), config)[
            0, 0
        ]
    )


def _drifting_lock_with_crop() -> dict[int, list[Observation]]:
    """A coarse lock 10 native px off a crop detection that tracks the true path."""
    frames = {}
    for frame in range(1, 8):
        truth = 100.0 + 5.0 * frame
        frames[frame] = [
            observation(truth + 10.0, 100.0, "coarse_lock", score=0.4),
            Observation(truth, 100.0, 0.95, 0, ("branched_crop_wasb",), ("persistent_lock",)),
            Observation(truth, 100.0, 0.93, 0, ("branched_crop_tracknetv2",), ("persistent_lock",)),
        ]
    return frames


def test_default_association_discards_a_crop_outside_the_guide_gate() -> None:
    frames = _drifting_lock_with_crop()

    rows, _ = track_clip("pt0001", frames, _identity_geometry(frames), 50.0)

    assert len(rows) == 7
    assert all("provenance:coarse" in row["sources"] for row in rows)


def test_crop_first_selects_the_crop_over_a_drifted_lock() -> None:
    frames = _drifting_lock_with_crop()

    rows, _ = track_clip("pt0001", frames, _identity_geometry(frames), 50.0, crop_first_config())

    assert len(rows) == 7
    # Frame 1 still bootstraps on the lock -- there is no filter state to prefer a crop with
    # yet -- and every frame after it takes the crop.
    assert "provenance:coarse" in rows[0]["sources"]
    assert all("provenance:crop" in row["sources"] for row in rows[1:])
    # The emitted position is the crop's, in 960x540 artifact units.
    assert float(rows[-1]["x"]) * 2.0 == 135.0


def test_crop_first_carries_a_frame_the_coarse_lock_never_reached() -> None:
    frames = {
        1: [observation(100.0, 100.0, "coarse_lock")],
        2: [
            Observation(105.0, 100.0, 0.95, 0, ("branched_crop_wasb",), ()),
            Observation(105.0, 100.0, 0.94, 0, ("branched_crop_tracknetv2",), ()),
        ],
        3: [observation(110.0, 100.0, "coarse_lock")],
    }
    geometry = _identity_geometry(frames)

    default_rows, _ = track_clip("pt0001", frames, geometry, 50.0)
    first_rows, _ = track_clip("pt0001", frames, geometry, 50.0, crop_first_config())

    assert [row["frame"] for row in default_rows] == ["f_0001.jpg", "f_0003.jpg"]
    assert [row["frame"] for row in first_rows] == [
        "f_0001.jpg",
        "f_0002.jpg",
        "f_0003.jpg",
    ]


def test_crop_first_still_rejects_a_wrong_object_crop_outside_the_imm_gate() -> None:
    frames = {
        frame: [
            observation(100.0 + 5.0 * frame, 100.0, "coarse_lock"),
            *(
                [Observation(700.0, 400.0, 0.99, 0, ("branched_crop_wasb",), ())]
                if frame == 5
                else []
            ),
        ]
        for frame in range(1, 8)
    }

    rows, _ = track_clip("pt0001", frames, _identity_geometry(frames), 50.0, crop_first_config())

    assert all(float(row["x"]) * 2.0 < 200.0 for row in rows)


def test_image_exit_remains_missing_beyond_restart_timeout_and_reacquires() -> None:
    from dataclasses import replace

    frames = {}
    # A well-supported ball leaves the top. A guide then locks to a body inside
    # the image for longer than maximum_misses; it is not a new ball track.
    for f in range(1, 11):
        frames[f] = [observation(300 + 8 * f, 105 - 10 * f, "coarse_lock")]
    for f in range(11, 35):
        frames[f] = [observation(700 + f, 400, "coarse_lock")]
    for f in range(35, 41):
        x, y = 600 + 8 * (f - 35), 5 + 12 * (f - 35)
        frames[f] = [observation(x, y, "sliding_wasb"), observation(x, y, "sliding_tracknetv2")]
    geometry = Geometry({("p", f): np.eye(3) for f in frames}, {}, "fixture")
    before, _ = track_clip("p", frames, geometry, 25)
    after, proposals = track_clip(
        "p", frames, geometry, 25, replace(MotionConfig(), qualified_image_reentry=True)
    )
    emitted = {int(row["frame"][2:6]): row for row in after}
    assert any(11 <= int(row["frame"][2:6]) < 35 for row in before)
    assert not any(f in emitted for f in range(11, 35))
    assert all(f in emitted for f in range(35, 41))
    assert emitted[35]["x"] * 2 == 600
    assert emitted[35]["y"] * 2 == 5
    assert [p["proposal"] for p in proposals if "image_" in p["proposal"]] == [
        "missing_after_image_exit",
        "qualified_image_reentry",
    ]


def test_image_reentry_rejects_single_exposure_and_stationary_edge_distractors() -> None:
    from cv.pipeline.ball_motion_tracker import _supported_image_reentry

    def detectors(x, y):
        return [Observation(x, y, 0.8, 0, ("sliding_wasb", "sliding_tracknetv2"))]

    assert _supported_image_reentry(1, {1: detectors(300, 5)}, MotionConfig()) is None
    assert (
        _supported_image_reentry(1, {f: detectors(300, 5) for f in (1, 2, 3)}, MotionConfig())
        is None
    )
    assert (
        _supported_image_reentry(
            1, {1: detectors(300, 5), 2: detectors(900, 20), 3: detectors(910, 30)}, MotionConfig()
        )
        is None
    )
    assert (
        _supported_image_reentry(
            1, {f: [observation(300, 5 + f * 10, "coarse_lock")] for f in (1, 2, 3)}, MotionConfig()
        )
        is None
    )


def test_image_reentry_option_preserves_interior_contact_and_healthy_path_exactly() -> None:
    from dataclasses import replace

    frames = {f: [observation(200 + f * 6, 300 - f * 2, "coarse_lock")] for f in range(1, 15)}
    # A real contact can cause a large interior change; this bounded option
    # cannot relabel every motion impulse as an image exit.
    frames.update(
        {f: [observation(500 + f * 2, 400 + f * 4, "coarse_lock")] for f in range(15, 25)}
    )
    geometry = Geometry({("p", f): np.eye(3) for f in frames}, {}, "fixture")
    before = track_clip("p", frames, geometry, 25)
    after = track_clip(
        "p", frames, geometry, 25, replace(MotionConfig(), qualified_image_reentry=True)
    )
    assert before == after


def test_reentry_velocity_excludes_camera_pan() -> None:
    from cv.pipeline.ball_motion_tracker import _reentry_velocity

    # Camera moves the image right10px; ball moves right5px between exposures.
    geometry = Geometry(
        {
            ("p", 1): np.eye(3),
            ("p", 2): np.asarray([[1.0, 0.0, -10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        },
        {},
        "pan",
    )
    chain = [observation(300, 5, "sliding_wasb"), observation(315, 17, "sliding_wasb")]
    np.testing.assert_allclose(_reentry_velocity(chain, 1, geometry, "p"), [5, 12])


def test_exit_reference_keeps_observation_camera_across_missing_frames(monkeypatch) -> None:
    from dataclasses import replace

    from cv.pipeline import ball_motion_tracker as tracker

    frames = {1: [observation(100, 50, "coarse_lock")], 2: [], 3: []}
    geometry = Geometry(
        {
            ("p", f): np.asarray([[1.0, 0.0, -10.0 * (f - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            for f in frames
        },
        {},
        "pan",
    )
    references = []

    def record_reference(previous_xy, predicted_xy, config):
        references.append(previous_xy.copy())
        return False

    monkeypatch.setattr(tracker, "_predicted_image_exit", record_reference)
    rows, _ = track_clip(
        "p", frames, geometry, 25, replace(MotionConfig(), qualified_image_reentry=True)
    )
    assert len(rows) == 1  # Missing exposures remain missing.
    np.testing.assert_allclose(references, [[110, 50], [120, 50]])


def test_healthy_visible_edge_graze_does_not_trigger_missing() -> None:
    from dataclasses import replace

    frames = {
        f: [observation(300 + f * 5, 4 + 0.5 * (f - 8) ** 2, "coarse_lock")] for f in range(1, 17)
    }
    geometry = Geometry({("p", f): np.eye(3) for f in frames}, {}, "fixture")
    before = track_clip("p", frames, geometry, 25)
    after = track_clip(
        "p", frames, geometry, 25, replace(MotionConfig(), qualified_image_reentry=True)
    )
    assert before == after


def test_untrusted_guide_cannot_bootstrap_after_reentry_detector_timeout() -> None:
    from dataclasses import replace

    frames = {f: [observation(300 + 8 * f, 105 - 10 * f, "coarse_lock")] for f in range(1, 11)}
    frames.update({f: [observation(700 + f, 400, "coarse_lock")] for f in range(11, 35)})
    for f in range(35, 41):
        frames[f] = [
            Observation(
                600 + 8 * (f - 35),
                5 + 12 * (f - 35),
                0.9,
                0,
                ("sliding_wasb", "sliding_tracknetv2"),
            )
        ]
    frames.update({f: [observation(1700, 1000, "coarse_lock")] for f in range(41, 70)})
    # Later direct evidence may reacquire in the interior after an occlusion;
    # retaining guide distrust does not require another image-edge crossing.
    for f in range(70, 76):
        frames[f] = [
            Observation(
                900 + 8 * (f - 70),
                400 + 12 * (f - 70),
                0.9,
                0,
                ("sliding_wasb", "sliding_tracknetv2"),
            )
        ]
    geometry = Geometry({("p", f): np.eye(3) for f in frames}, {}, "fixture")
    rows, _ = track_clip(
        "p", frames, geometry, 25, replace(MotionConfig(), qualified_image_reentry=True)
    )
    emitted = {int(r["frame"][2:6]) for r in rows}
    assert not emitted.intersection(range(41, 70))
    assert all(f in emitted for f in range(70, 76))


def test_camera_only_inward_motion_remains_ambiguous() -> None:
    from cv.pipeline.ball_motion_tracker import _supported_image_reentry

    frames = {
        f: [Observation(300, 5 + 10 * (f - 1), 0.9, 0, ("sliding_wasb", "sliding_tracknetv2"))]
        for f in (1, 2, 3)
    }
    geometry = Geometry(
        {
            ("p", f): np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -10 * (f - 1)], [0.0, 0.0, 1.0]])
            for f in frames
        },
        {},
        "camera-only pan",
    )
    assert _supported_image_reentry(1, frames, MotionConfig(), geometry, "p", 25.0) is None
