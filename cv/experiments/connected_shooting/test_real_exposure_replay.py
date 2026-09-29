import numpy as np
from dataclasses import replace

from cv.experiments.connected_shooting import model, real_exposure_replay as replay


def test_terminal_comparator_schema_is_the_published_schema():
    assert replay.BASELINE_SCHEMA == "connected_terminal_feasibility_v1"
    assert replay.ARMS == {
        "centre": None,
        "front_d025": 0.25,
        "front_d050": 0.5,
        "front_d075": 0.75,
    }


def test_duration_none_uses_nominal_centre(monkeypatch):
    expected = np.array([[1.0, 2.0]])
    monkeypatch.setattr(
        replay.leading,
        "image_prediction",
        lambda scene, parameters, axes, endpoint, cache: expected,
    )
    actual = replay.prediction(object(), np.zeros(1), np.zeros((1, 2)), None)
    assert np.array_equal(actual, expected)


def test_zero_time_shift_replays_the_same_whole_path_without_refitting():
    scene, parameters = model.control()
    axes = np.tile([1.0, 0.0], (sum(map(len, scene.observation_frames)), 1))
    expected = replay.prediction(scene, parameters, axes, 0.25)
    actual = replay.time_shifted_prediction(scene, parameters, axes, 0.25, 0.0)

    np.testing.assert_allclose(actual, expected)


def test_shifted_checks_respect_an_existing_observation_horizon(monkeypatch):
    scene, parameters = model.control()
    axes = np.tile([1.0, 0.0], (sum(map(len, scene.observation_frames)), 1))
    original = model.chain
    calls = []

    def within_observed_domain(active, params, **kwargs):
        # A valid path need not support a numerical continuation after its
        # declared observation horizon (for example, a later bounce cascade).
        assert active.contact_frames[-1] == scene.contact_frames[-1]
        calls.append(active.contact_frames.copy())
        return original(active, params, **kwargs)

    monkeypatch.setattr(model, "chain", within_observed_domain)
    shifted = replay.time_shifted_prediction(
        scene, parameters, axes, None, 1.0, preserve_observation_horizon=True
    )
    assert calls
    assert np.isfinite(shifted[:-1]).all()
    assert np.isnan(shifted[-1]).all()
    zero = replay.time_shifted_prediction(
        scene, parameters, axes, None, 0.0, preserve_observation_horizon=True
    )
    np.testing.assert_allclose(zero, replay.prediction(scene, parameters, axes, None))
    np.testing.assert_array_equal(scene.contact_frames, [1, 13, 26])


def test_bounce_cap_in_the_terminal_extension_leaves_extension_rows_unsupported(monkeypatch):
    from cv.experiments.connected_shooting.measured_dynamics import BounceCapacityError

    scene, parameters = model.control()
    axes = np.tile([1.0, 0.0], (sum(map(len, scene.observation_frames)), 1))
    original = model.chain

    def capped_after_the_domain(active, params, **kwargs):
        if active.contact_frames[-1] > scene.contact_frames[-1]:
            raise BounceCapacityError(active.contact_frames[-1], 8)
        return original(active, params, **kwargs)

    monkeypatch.setattr(model, "chain", capped_after_the_domain)
    shifted = replay.time_shifted_prediction(scene, parameters, axes, None, 1.0)
    expected = replay.time_shifted_prediction(
        scene, parameters, axes, None, 1.0, preserve_observation_horizon=True
    )
    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(expected))
    np.testing.assert_allclose(shifted[np.isfinite(shifted)], expected[np.isfinite(expected)])
    assert np.isnan(shifted[-1]).all()

    def capped_inside_the_domain(active, params, **kwargs):
        raise BounceCapacityError(active.contact_frames[-1], 8)

    monkeypatch.setattr(model, "chain", capped_inside_the_domain)
    try:
        replay.time_shifted_prediction(scene, parameters, axes, None, 1.0)
    except BounceCapacityError:
        pass
    else:
        raise AssertionError("a bounce cap inside the fitted domain must still raise")


def test_terminal_rebound_activates_only_withheld_postbounce_rows():
    full, _ = model.control()
    full = replace(full, dynamics="measured_240hz")

    def subset(heldout):
        masks = tuple((frames % 5 == 0) == heldout for frames in full.observation_frames)
        return replace(
            full,
            observation_frames=tuple(
                frames[mask] for frames, mask in zip(full.observation_frames, masks, strict=True)
            ),
            cameras=tuple(rows[mask] for rows, mask in zip(full.cameras, masks, strict=True)),
            pixels=tuple(rows[mask] for rows, mask in zip(full.pixels, masks, strict=True)),
        )

    scene, heldout = subset(False), subset(True)
    axes = np.tile([1.0, 0.0], (sum(map(len, scene.observation_frames)), 1))
    active, active_axes, rebound_frames, receipt = replay.terminal_rebound_segment(
        scene, heldout, (np.array([9.0]), np.array([21.0])), axes
    )

    terminal_frames = full.observation_frames[-1]
    assert set(rebound_frames) == set(terminal_frames[terminal_frames > 21])
    expected_active = set(terminal_frames[terminal_frames % 5 != 0]) | {25.0}
    assert set(active.observation_frames[-1]) == expected_active
    assert len(active_axes) == sum(map(len, active.observation_frames))
    assert receipt["withheld_rows_activated_frames"] == [25.0]
    assert receipt["native_timestamps_changed"] is False
    assert receipt["pictures_invented"] == 0


def test_terminal_rebound_inventory_reads_frozen_context_beyond_old_endpoint():
    attempt = {"point_clip": "pt0002"}
    records = [
        {
            "clip": "pt0002",
            "frames": [
                {"frame": 217, "status": "visible"},
                {"frame": 218, "status": "ambiguous"},
                {"frame": 219, "status": "visible"},
                {"frame": 220, "status": "visible"},
            ],
        }
    ]
    receipt = replay.terminal_rebound_inventory(attempt, records, 217.5)

    assert receipt["status"] == "supported"
    assert receipt["postbounce_labeled_frames"] == [219, 220]
    assert receipt["last_postbounce_labeled_frame"] == 220


def test_terminal_rebound_accepts_no_withheld_row_after_bounce():
    full, _ = model.control()
    full = replace(full, dynamics="measured_240hz")
    scene = replace(
        full,
        observation_frames=tuple(frames[frames % 5 != 0] for frames in full.observation_frames),
        cameras=tuple(
            rows[frames % 5 != 0]
            for rows, frames in zip(full.cameras, full.observation_frames, strict=True)
        ),
        pixels=tuple(
            rows[frames % 5 != 0]
            for rows, frames in zip(full.pixels, full.observation_frames, strict=True)
        ),
    )
    heldout = replace(
        full,
        observation_frames=(np.array([5.0, 10.0]), np.array([15.0, 20.0])),
        cameras=(full.cameras[0][[4, 9]], full.cameras[1][[1, 6]]),
        pixels=(full.pixels[0][[4, 9]], full.pixels[1][[1, 6]]),
    )
    axes = np.tile([1.0, 0.0], (sum(map(len, scene.observation_frames)), 1))

    _, active_axes, frames, receipt = replay.terminal_rebound_segment(
        scene, heldout, (np.array([9.0]), np.array([21.0])), axes
    )

    expected = scene.observation_frames[-1][scene.observation_frames[-1] > 21.0]
    assert np.array_equal(frames, expected)
    assert len(active_axes) == len(axes)
    assert receipt["withheld_rows_activated_count"] == 0


def test_anchor_plan_uses_the_projection_derived_graded_radius_with_a_floor():
    plan = replay.anchor_plan(
        [
            [
                {"xyz_m": [1.0, 2.0, 0.0325], "graded_circle_radius_m": 0.2},
                {"xyz_m": [3.0, 4.0, 0.0325], "graded_circle_radius_m": 0.44},
            ],
            [{"xyz_m": [5.0, 6.0, 0.0325], "uncertainty_sigma_m": 0.05}],
        ],
        [[7.0, 8.0], None],
    )

    assert [row[1] for row in plan["bounce"][0]] == [0.2, 0.44]
    # 2 sigma is below the floor, so the floor decides.
    assert plan["bounce"][1][0][1] == replay.ANCHOR_BOUNCE_SIGMA_FLOOR_M
    assert np.array_equal(plan["contact"][0][0], np.array([7.0, 8.0]))
    assert plan["contact"][1] is None
    assert plan["player_sigma_m"] == replay.ANCHOR_PLAYER_SIGMA_M


def test_anchor_plan_refuses_a_mismatched_flight_count():
    try:
        replay.anchor_plan([[{"xyz_m": [1.0, 2.0, 0.0325]}]], [])
    except ValueError as error:
        assert "per connected flight" in str(error)
    else:
        raise AssertionError("mismatched anchor groups must fail closed")


def test_arm_receipt_names_the_arm_without_anchors():
    receipt = replay.arm_receipt(None, None, None, None, False, None)

    assert receipt == {
        "inequality_constraints": "out",
        "anchor_residuals": "absent",
        "acceptance_gates_unchanged": True,
    }


def test_raw_contact_interval_bounds_cold_rebound_even_if_packet_contact_was_removed():
    from copy import deepcopy

    attempt = {"point_clip": "p", "events": [{"event_type": "bounce", "frame": 10}]}
    records = [{"clip": "p", "frames": [{"frame": f, "status": "visible"} for f in range(11, 20)]}]
    labels = {
        "events": {
            "records": [
                {"clip": "other", "event_type": "contact", "frame": 11},
                {"clip": "p", "event_type": "contact", "frame": 12, "occurrence_status": "absent"},
                {
                    "clip": "p",
                    "event_type": "contact",
                    "frame": 16,
                    "frame_interval": [14.25, 17],
                    "status": "ambiguous",
                },
                {"clip": "p", "event_type": "bounce", "frame": 18},
            ]
        }
    }
    original = deepcopy((attempt, records, labels))
    result = replay.terminal_rebound_inventory(
        attempt, records, 10, labels=labels, duration_frames=0.25
    )
    assert result["postbounce_labeled_frames"] == [11, 12, 13]
    assert result["barrier_inventory_source"] == "raw_label_event_records"
    assert result["next_contact_boundary"]["original_contact"]["frame"] == 16
    # Nominal center at14 is before14.25; a swept14 exposure straddles it.
    center = replay.terminal_rebound_inventory(
        attempt, records, 10, labels=labels, duration_frames=None
    )
    assert center["postbounce_labeled_frames"] == [11, 12, 13, 14]
    assert (attempt, records, labels) == original


def test_single_ending_rebound_does_not_cross_unmodeled_net_or_second_ground():
    for kind in ["net_hit", "bounce"]:
        attempt = {"point_clip": "p"}
        records = [
            {"clip": "p", "frames": [{"frame": f, "status": "visible"} for f in range(11, 16)]}
        ]
        labels = {
            "events": {
                "records": [
                    {"clip": "p", "event_type": kind, "frame": 14, "frame_interval": [13.5, 14.5]}
                ]
            }
        }
        result = replay.terminal_rebound_inventory(attempt, records, 10, labels=labels)
        assert result["postbounce_labeled_frames"] == [11, 12, 13]


def test_contact_interval_overlapping_ending_abstains_on_continuation_order():
    attempt = {"point_clip": "p"}
    records = [{"clip": "p", "frames": [{"frame": 11, "status": "visible"}]}]
    labels = {
        "events": {"records": [{"event_type": "contact", "frame": 11, "frame_interval": [9, 12]}]}
    }
    result = replay.terminal_rebound_inventory(attempt, records, 10, labels=labels)
    assert result["status"] == "source_event_order_uncertain"
    assert result["postbounce_labeled_frames"] == []
    assert result["source_order_conflict"]["next_interval_start"] == 9


def test_rebound_after_declared_second_bounce_uses_only_later_raw_barrier():
    attempt = {"point_clip": "p"}
    events = [
        {"event_type": "bounce", "frame": f, "frame_interval": [f - 0.5, f + 0.5]} for f in [10, 14]
    ]
    events.append({"event_type": "contact", "frame": 19, "frame_interval": [18, 20]})
    labels = {"events": {"records": events}}
    records = [{"clip": "p", "frames": [{"frame": f, "status": "visible"} for f in range(11, 21)]}]
    result = replay.terminal_rebound_inventory(attempt, records, 14, labels=labels)
    assert result["postbounce_labeled_frames"] == [15, 16, 17]
    assert result["next_contact_boundary"]["original_contact"]["frame"] == 19
