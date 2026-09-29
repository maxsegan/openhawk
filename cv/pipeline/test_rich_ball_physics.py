from types import SimpleNamespace

import numpy as np
import pytest

from rich_ball_physics import (
    _free_state_bidirectional,
    _rk4,
    _valid_flight_endpoint,
    audio_witness_check,
    bounce_energy_ratio,
    bounce_recovery_screen_passed,
    bounce_velocity,
    bounce_for_shot,
    contact_image_observation,
    contact_image_residual,
    contact_ray_anchor,
    contact_ray_candidates,
    contact_xy_bounds,
    contact_prior,
    fit_anchored_ballistic_bounce_node_shot,
    fixed_bounce_diagnostics,
    full_spin_theta,
    implied_racket,
    retime_closest_approach,
    retime_contacts,
    R_BALL,
    simulate,
    simulate_fixed_bounce,
    simulate_hard_bounce_knot,
    simulate_free,
    simulate_with_spin,
    ShotFit,
    spin_component_rpm,
    spin_vector,
)


def _contacts(*pairs):
    return [{"frame": float(f), "side": s} for f, s in pairs]


_DENSE = {f: (0.0, 0.0) for f in range(0, 400)}  # >=MIN_OBS in any span >= 10 frames


def test_rk4_rejects_divergent_optimizer_state() -> None:
    with pytest.raises(FloatingPointError, match="diverged"):
        _rk4(np.zeros(3), np.array([300.0, 0.0, 0.0]), np.zeros(3), 0.01)


def test_hard_bounce_knot_is_exact_and_uses_the_impact_law() -> None:
    anchor = {"frame": 12.5, "x": np.array([4.0, 15.0, R_BALL])}
    frames = np.array([5.0, 12.5, 20.0])

    positions, _, _, bounces, _ = simulate_hard_bounce_knot(
        np.array([2.0, -18.0, -6.0]),
        np.array([0.0, 120.0, 0.0]),
        5.0,
        frames,
        25.0,
        "hard",
        anchor,
    )

    np.testing.assert_allclose(positions[1], anchor["x"], atol=1e-12)
    np.testing.assert_allclose(bounces[0]["x"], anchor["x"], atol=1e-12)
    assert bounces[0]["v_in"][2] < 0.0
    assert bounces[0]["v_out"][2] > 0.0


def test_terminal_contact_initializes_from_last_ground_projection():
    camera = SimpleNamespace(h_at=lambda _frame: np.eye(3))
    contact = {
        "frame": 20,
        "side": "far",
        "phase": "terminal",
        "terminal": True,
    }

    point, player, pixel = contact_prior(
        contact,
        {20: np.array([4.0, 7.0])},
        {"near": {}, "far": {}},
        camera,
    )

    np.testing.assert_allclose(point, [4.0, 7.0, R_BALL])
    np.testing.assert_allclose(player, [4.0, 7.0])
    np.testing.assert_allclose(pixel, [4.0, 7.0])


def test_contact_bounds_keep_optimizer_corners_within_radial_reach():
    player = np.array([4.0, 24.0])

    lower, upper = contact_xy_bounds(player, 2.0)

    for corner in (
        lower,
        upper,
        np.array([lower[0], upper[1]]),
        np.array([upper[0], lower[1]]),
    ):
        assert np.linalg.norm(corner - player) <= 2.0 + 1e-12


def test_contact_image_observation_interpolates_fractional_frame():
    observed, frames = contact_image_observation(
        {
            10: np.array([390.0, 500.0]),
            11: np.array([410.0, 500.0]),
        },
        10.5,
    )

    np.testing.assert_allclose(observed, [400.0, 500.0])
    assert frames == (10, 11)


def test_contact_image_residual_is_zero_at_observed_ray():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)

    residual, diagnostic = contact_image_residual(
        np.array([4.0, 5.0, 1.5]),
        {"frame": 10.5},
        {
            10: np.array([390.0, 500.0]),
            11: np.array([410.0, 500.0]),
        },
        camera,
    )

    np.testing.assert_allclose(residual, [0.0, 0.0], atol=1e-12)
    assert diagnostic["source_frames"] == [10, 11]
    assert diagnostic["error_px"] == 0.0


def test_contact_image_residual_accepts_label_free_arc_override():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)

    residual, diagnostic = contact_image_residual(
        np.array([4.0, 5.0, 1.5]),
        {
            "frame": 10.0,
            "image_observation_override": [400.0, 500.0],
            "image_observation_source_frames": [10],
            "image_observation_confidence": 0.7,
        },
        {10: np.array([900.0, 900.0])},
        camera,
    )

    np.testing.assert_allclose(residual, [0.0, 0.0], atol=1e-12)
    assert diagnostic["observed_uv"] == [400.0, 500.0]


def test_contact_ray_anchor_is_exact_and_player_reachable():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.5,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }

    anchor, diagnostic = contact_ray_anchor(
        contact,
        np.array([4.0, 5.0, 1.5]),
        {
            10: np.array([400.0, 150.0]),
            11: np.array([400.0, 150.0]),
        },
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
    )

    np.testing.assert_allclose(anchor[:2], [4.0, 5.0], atol=1e-12)
    assert np.linalg.norm(anchor[:2] - [4.0, 5.0]) <= 2.0
    assert diagnostic["reprojection_error_px"] < 1e-9


def test_contact_ray_anchor_uses_pose_wrist_as_soft_racket_witness():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }

    anchor, diagnostic = contact_ray_anchor(
        contact,
        np.array([4.0, 4.0, 1.5]),
        {10: np.array([400.0, 150.0])},
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
        pose_witness={
            "confidence": 0.9,
            "wrist_xyz": [4.0, 5.0, 0.9],
            "source": "test_pose",
        },
    )

    assert anchor is not None
    assert diagnostic["pose_witness"]["source"] == "test_pose"
    assert 0.2 <= diagnostic["wrist_distance_m"] <= 1.2
    assert diagnostic["toward_net_m"] >= -0.35


def test_contact_ray_candidates_span_reachable_exact_ray_depths():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }

    candidates, diagnostic = contact_ray_candidates(
        contact,
        np.array([4.0, 5.0, 1.5]),
        {10: np.array([400.0, 150.0])},
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
    )

    assert diagnostic["available"]
    assert len(candidates) > 20
    assert np.ptp([row["point"][1] for row in candidates]) > 1.0
    for row in candidates:
        np.testing.assert_allclose(
            [row["point"][0] * 100.0, row["point"][2] * 100.0],
            [400.0, 150.0],
            atol=1e-9,
        )


def test_contact_ray_candidates_reject_body_scale_inconsistent_height():
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -100.0, 500.0],
            [0.0, 1.0, 0.0, 10.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }

    candidates, diagnostic = contact_ray_candidates(
        contact,
        np.array([0.0, 5.0, 2.0]),
        {10: np.array([0.0, 20.0])},
        {"near": {10: np.array([0.0, 5.0])}, "far": {}},
        camera,
        pose_witness={
            "confidence": 0.8,
            "apparent_ball_height_m": 1.4,
            "apparent_height_confidence": 0.9,
            "source": "test_body_scale",
        },
    )

    assert diagnostic["available"]
    assert candidates
    assert max(abs(row["point"][2] - 1.4) for row in candidates) <= 0.45 + 1e-9


def test_contact_ray_candidates_softly_prefer_physical_racket_head() -> None:
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }
    common = (
        contact,
        np.array([4.0, 5.5, 1.5]),
        {10: np.array([400.0, 150.0])},
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
    )

    baseline, _ = contact_ray_candidates(*common)
    physical, _ = contact_ray_candidates(
        *common,
        pose_witness={
            "confidence": 1.0,
            "racket_head_hypotheses": [{"head_center_xyz": [4.0, 4.5, 1.5]}],
        },
    )

    assert abs(physical[0]["point"][1] - 4.5) < abs(baseline[0]["point"][1] - 4.5)


def test_physical_racket_does_not_replace_single_flight_geometry_anchor() -> None:
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }
    arguments = (
        contact,
        np.array([4.0, 5.5, 1.5]),
        {10: np.array([400.0, 150.0])},
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
    )

    baseline, _ = contact_ray_anchor(*arguments)
    physical, diagnostic = contact_ray_anchor(
        *arguments,
        pose_witness={
            "confidence": 1.0,
            "racket_head_hypotheses": [{"head_center_xyz": [4.0, 4.5, 1.5]}],
        },
    )

    np.testing.assert_allclose(physical, baseline)
    assert diagnostic["pose_witness"]["racket_head_hypotheses"]


def test_contact_ray_candidates_do_not_let_weak_racket_branch_dominate() -> None:
    projection = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact = {
        "frame": 10.0,
        "side": "near",
        "phase": "rally",
        "side_evidence": {"confidence": 0.9},
    }

    candidates, _ = contact_ray_candidates(
        contact,
        np.array([4.0, 5.5, 1.5]),
        {10: np.array([400.0, 150.0])},
        {"near": {10: np.array([4.0, 5.0])}, "far": {}},
        camera,
        pose_witness={
            "confidence": 0.9,
            "racket_head_hypotheses": [
                {
                    "head_center_xyz": [4.0, 4.5, 1.5],
                    "evidence_confidence": 0.05,
                },
                {
                    "head_center_xyz": [4.0, 5.0, 1.5],
                    "evidence_confidence": 0.90,
                },
            ],
        },
    )

    assert candidates[0]["racket_head_evidence_confidence"] == 0.90
    assert abs(candidates[0]["point"][1] - 5.0) < abs(candidates[0]["point"][1] - 4.5)


def test_bounce_energy_ratio_includes_rotational_energy():
    bounce = {
        "v_in": np.array([0.0, 10.0, -5.0]),
        "v_out": np.array([0.0, 8.0, 3.0]),
        "w_in": np.array([0.0, 0.0, 0.0]),
        "w_out": np.array([0.0, 100.0, 0.0]),
    }

    assert (
        bounce_energy_ratio(bounce)
        > np.linalg.norm(bounce["v_out"]) ** 2 / np.linalg.norm(bounce["v_in"]) ** 2
    )


def test_bounce_recovery_rejects_a_failed_internal_screen():
    candidate = SimpleNamespace(
        _bounce_node_split={"screen_passed": False, "screen_reasons": ["impact_slack"]}
    )

    assert not bounce_recovery_screen_passed(candidate)
    assert bounce_recovery_screen_passed(SimpleNamespace())


def test_endpoint_prefers_nearest_opposite_side():
    # forward from the near contact: the adjacent far contact bounds a cross-net flight
    contacts = _contacts((100, "near"), (180, "far"))
    assert _valid_flight_endpoint(contacts, 0, +1, _DENSE, 50.0) == (1, True)


def test_endpoint_skips_same_side_insertion_to_reach_opposite():
    # a same-side (near) phantom split sits between the real near contact and the far end;
    # it is spanned, not treated as a boundary
    contacts = _contacts((100, "near"), (140, "near"), (200, "far"))
    assert _valid_flight_endpoint(contacts, 0, +1, _DENSE, 50.0) == (2, True)


def test_endpoint_skips_too_short_span():
    # the immediate opposite contact is <MIN_OBS frames away (an unfittable stub); reach on
    contacts = _contacts((180, "far"), (185, "near"), (240, "near"))
    assert _valid_flight_endpoint(contacts, 0, +1, _DENSE, 50.0) == (2, True)


def test_endpoint_accepts_slow_opposite_side_flight():
    # The opposite endpoint is 3.4 seconds away. Slow slices, drop shots, and lobs must
    # reach physics scoring rather than falling back to a same-side boundary.
    contacts = _contacts((40, "far"), (140, "far"), (210, "near"))
    assert _valid_flight_endpoint(contacts, 0, +1, _DENSE, 50.0) == (2, True)


def test_endpoint_none_when_nothing_within_gap():
    contacts = _contacts((40, "near"), (400, "far"))  # 7.2 seconds exceeds candidate scope
    assert _valid_flight_endpoint(contacts, 0, +1, _DENSE, 50.0) == (None, False)


def test_simulation_applies_physical_court_bounce():
    theta = np.array([5.0, 3.0, 1.0, 0.0, 18.0, -2.0, 0.0])
    frames = np.arange(0, 51)
    xs, _, bounces = simulate(theta, 0.0, frames, 50.0, "hard")
    assert bounces
    assert abs(bounces[0]["x"][2] - R_BALL) < 1e-6
    assert bounces[0]["v_in"][2] < 0 < bounces[0]["v_out"][2]
    assert np.isfinite(xs).all()


def test_long_simulation_never_falls_through_after_second_bounce():
    theta = np.array([5.0, 3.0, 0.8, 0.0, 4.0, -1.0, 0.0])
    frames = np.arange(0, 251)

    xs, _, bounces = simulate(theta, 0.0, frames, 50.0, "hard")

    assert len(bounces) > 2
    assert float(np.min(xs[:, 2])) >= R_BALL - 1e-9


def test_simulation_decays_spin_during_flight():
    theta = np.array([5.0, 3.0, 8.0, 0.0, 25.0, 2.0, 2.0])
    _, _, spins, bounces = simulate_with_spin(
        theta,
        0.0,
        np.array([0.0, 25.0]),
        25.0,
        "hard",
    )

    assert not bounces
    assert np.linalg.norm(spins[-1]) < np.linalg.norm(spins[0])
    assert np.linalg.norm(spins[-1]) > 0.90 * np.linalg.norm(spins[0])


def test_bounce_records_decayed_incoming_and_changed_outgoing_spin():
    theta = np.array([5.0, 3.0, 1.0, 0.0, 18.0, -2.0, 2.0])
    _, _, spins, bounces = simulate_with_spin(
        theta,
        0.0,
        np.arange(0, 51),
        50.0,
        "clay",
    )

    bounce = bounces[0]
    assert np.linalg.norm(bounce["w_in"]) < np.linalg.norm(spins[0])
    assert not np.allclose(bounce["w_out"], bounce["w_in"])


def test_full_spin_basis_supports_sidespin_and_legacy_theta():
    legacy = np.array([5.0, 3.0, 5.0, 0.0, 25.0, 2.0, 2.0])
    upgraded = full_spin_theta(legacy)
    np.testing.assert_allclose(upgraded[:7], legacy)
    np.testing.assert_allclose(upgraded[7:], 0.0)

    theta = upgraded.copy()
    theta[7] = 3.0
    top_only = spin_vector(legacy)
    full = spin_vector(theta)
    assert not np.allclose(full, top_only)
    np.testing.assert_allclose(
        spin_component_rpm(theta)[2],
        0.0,
        atol=1e-12,
    )

    vertical = np.array([5.0, 3.0, 5.0, 0.0, 0.0, 12.0, 1.0, 2.0, 0.5])
    assert np.linalg.norm(spin_vector(vertical)) > 0.0


def test_sidespin_curves_opposite_directions_in_flight():
    base = np.array([5.0, 3.0, 8.0, 0.0, 25.0, 1.0, 0.0, 3.0, 0.0])
    frames = np.array([0.0, 12.0])
    left, _, _, _ = simulate_with_spin(base, 0.0, frames, 25.0, "hard")
    opposite = base.copy()
    opposite[7] *= -1
    right, _, _, _ = simulate_with_spin(opposite, 0.0, frames, 25.0, "hard")
    assert left[-1, 0] < base[0] < right[-1, 0]


def test_vector_bounce_preserves_vertical_spin_and_couples_roll_spin():
    velocity = np.array([0.0, 20.0, -7.0])
    vertical_spin = np.array([0.0, 0.0, 250.0])
    v_vertical, w_vertical, _ = bounce_velocity(
        velocity,
        vertical_spin,
        "hard",
    )
    assert abs(v_vertical[0]) < 1e-12
    assert w_vertical[2] == vertical_spin[2]

    roll_spin = np.array([0.0, 180.0, 0.0])
    v_roll, w_roll, _ = bounce_velocity(velocity, roll_spin, "hard")
    assert abs(v_roll[0]) > 0.1
    assert not np.allclose(w_roll, roll_spin)


def test_fixed_bounce_is_an_exact_piecewise_ground_node():
    theta = np.array([5.0, 3.0, 1.2, 1.0, 18.0, -3.0, 0.5])
    anchor = {
        "frame": 10.0,
        "x": np.array([5.2, 6.0, 0.033]),
    }
    frames = np.array([0.0, 9.0, 10.0, 11.0, 20.0])
    xs, _, bounces, continuity = simulate_fixed_bounce(
        theta,
        0.0,
        frames,
        50.0,
        "clay",
        anchor,
    )
    np.testing.assert_allclose(xs[2], anchor["x"])
    np.testing.assert_allclose(bounces[0]["x"], anchor["x"])
    assert bounces[0]["frame"] == 10.0
    assert continuity.shape == (3,)


def test_fixed_bounce_diagnostics_rejects_underground_and_discontinuous_paths():
    bounce = {
        "v_in": np.array([1.0, 15.0, -4.0]),
        "v_out": np.array([1.0, 10.0, 3.0]),
    }
    valid = fixed_bounce_diagnostics(
        np.array([[5.0, 2.0, 1.0], [5.0, 4.0, 0.033]]),
        bounce,
        np.array([0.01, 0.01, 0.0]),
    )
    underground = fixed_bounce_diagnostics(
        np.array([[5.0, 2.0, 1.0], [5.0, 3.0, -0.2], [5.0, 4.0, 0.033]]),
        bounce,
        np.zeros(3),
    )
    discontinuous = fixed_bounce_diagnostics(
        np.array([[5.0, 2.0, 1.0], [5.0, 4.0, 0.033]]),
        bounce,
        np.array([0.0, 0.5, 0.0]),
    )

    assert valid["valid"] is True
    assert underground["valid"] is False
    assert discontinuous["valid"] is False


def test_hard_bounce_can_precede_contact_by_fewer_than_five_frames():
    anchors = [
        {"frame": 97.0, "hard_geometry": True},
        {"frame": 50.0},
    ]
    assert bounce_for_shot(anchors, 60.0, 100.0)["frame"] == 97.0


def test_bidirectional_pixel_intersection_retimes_without_labels():
    contacts = [
        {"frame": 50.0, "frame_detector": 50.0, "side": "near"},
        {"frame": 100.0, "frame_detector": 100.0, "side": "far"},
        {"frame": 150.0, "frame_detector": 150.0, "side": "near"},
    ]
    ball = {}
    # Two image-space branches meet at frame 102, while the central occlusion is absent.
    for frame in range(86, 99):
        t = frame - 102
        ball[frame] = np.array([400 + 2 * t, 150 + 0.1 * t * t])
    for frame in range(102, 115):
        t = frame - 102
        ball[frame] = np.array([400 - 3 * t, 150 + 0.2 * t * t])
    retime_contacts(contacts, ball)
    assert abs(contacts[1]["frame"] - 102.0) < 0.2
    assert contacts[1]["retime_source"] == "bidirectional_pixel_intersection"


def test_simulate_free_matches_simulate_forward_and_round_trips():
    # a high arc with no bounce in the window: bounceless simulate_free must equal simulate
    theta = np.array([5.0, 3.0, 1.5, 2.0, 15.0, 3.0, 0.5])
    frames = np.array([100.0, 103.0, 107.5, 112.0])
    xs_ref, _, bounces = simulate(theta, 100.0, frames, 50.0, "clay")
    assert not bounces
    xs_free = simulate_free(theta, 100.0, frames, 50.0)
    assert np.abs(xs_ref - xs_free).max() < 1e-9
    # backward integration returns exactly to the launch state at f0
    back = simulate_free(theta, 100.0, np.array([94.0, 100.0, 106.0]), 50.0)
    assert np.abs(back[1] - theta[:3]).max() < 1e-9


def test_bidirectional_state_integrator_meets_exact_node():
    node = np.array([5.0, 8.0, 0.033])
    velocity = np.array([1.0, 20.0, -5.0])
    spin = np.array([0.0, 0.0, 0.0])

    positions, _, _ = _free_state_bidirectional(
        node,
        velocity,
        spin,
        np.array([-5.0, 0.0, 5.0]),
        25.0,
    )

    np.testing.assert_allclose(positions[1], node)
    assert positions[0, 2] > node[2]
    assert positions[2, 2] < node[2]


def test_ballistic_bounce_node_state_is_exact_and_continuous():
    fit = ShotFit(
        0,
        1,
        np.zeros(9),
        2.0,
        0,
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        [],
        0,
    )
    fit._f0 = 0.0
    fit._ballistic_bounce_node = {
        "frame": 10.0,
        "xyz": np.array([5.0, 8.0, 0.033]),
        "incoming_velocity": np.array([0.0, 20.0, -4.0]),
        "outgoing_velocity": np.array([0.0, 13.0, 3.0]),
        "incoming_acceleration": np.array([0.0, 0.0, -9.81]),
        "outgoing_acceleration": np.array([0.0, 0.0, -9.81]),
    }

    position, velocity = fit.state(10.0, 25.0, "hard")

    np.testing.assert_allclose(position, fit._ballistic_bounce_node["xyz"])
    np.testing.assert_allclose(velocity, fit._ballistic_bounce_node["outgoing_velocity"])


def test_anchored_ballistic_bounce_node_preserves_contacts_and_bounce():
    fps = 10.0
    start_frame = 0.0
    bounce_frame = 10.0
    end_frame = 20.0
    node = np.array([5.0, 10.0, 0.033])
    incoming_acceleration = np.array([0.0, 0.0, -9.81])
    outgoing_acceleration = np.array([0.0, 0.0, -9.81])
    incoming_velocity = np.array([0.5, 10.0, -7.0])
    outgoing_velocity, _, _ = bounce_velocity(incoming_velocity, np.zeros(3), "hard")
    start_anchor = node - incoming_velocity + 0.5 * incoming_acceleration
    end_anchor = node + outgoing_velocity + 0.5 * outgoing_acceleration
    projection = np.array(
        [
            [80.0, 20.0, 0.0, 0.0],
            [0.0, 25.0, -100.0, 600.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    frames = np.arange(0, 21)
    relative = (frames - bounce_frame) / fps
    positions = np.empty((len(frames), 3))
    before = relative < 0
    positions[before] = (
        node
        + incoming_velocity * relative[before, None]
        + 0.5 * incoming_acceleration * relative[before, None] ** 2
    )
    positions[~before] = (
        node
        + outgoing_velocity * relative[~before, None]
        + 0.5 * outgoing_acceleration * relative[~before, None] ** 2
    )
    ball = {
        int(frame): (projection @ np.r_[position, 1.0])[:2]
        for frame, position in zip(frames, positions)
    }
    fit = fit_anchored_ballistic_bounce_node_shot(
        0,
        [
            {"frame": start_frame, "side": "near"},
            {"frame": end_frame, "side": "far"},
        ],
        ball,
        _StubCam(projection),
        fps,
        "hard",
        {"frame": bounce_frame, "x": node},
        start_anchor,
        end_anchor,
        20,
    )

    assert fit is not None
    np.testing.assert_allclose(fit.state(start_frame, fps, "hard")[0], start_anchor, atol=1e-6)
    np.testing.assert_allclose(fit.state(bounce_frame, fps, "hard")[0], node, atol=1e-6)
    np.testing.assert_allclose(fit.state(end_frame, fps, "hard")[0], end_anchor, atol=1e-6)


class _StubCam:
    def __init__(self, P):
        self._P = P

    def p_at(self, frame):
        return self._P


class _StubFit:
    def __init__(self, theta, f0):
        self.theta = np.asarray(theta, float)
        self.f0 = float(f0)


def test_closest_approach_retimes_two_flights_to_their_junction(monkeypatch):
    import rich_ball_physics as rbp

    monkeypatch.setattr(rbp, "CLOSEST_APPROACH", True)  # lever is OFF by default (measured parity)
    fps = 50.0
    P = np.array([[1200.0, 0.0, 40.0, 10.0], [0.0, -1200.0, 30.0, 900.0], [0.0, 0.0, 0.0, 1.0]])
    inc = _StubFit([2.0, 1.0, 1.2, 3.0, 14.0, 2.0, 0.0], 120.0)
    junction = simulate_free(inc.theta, inc.f0, np.array([150.0]), fps)[0]
    out = _StubFit([junction[0], junction[1], junction[2], -4.0, 18.0, 4.0, 0.0], 150.0)
    contacts = [
        {"frame": 120.0, "frame_detector": 120.0, "side": "near"},
        {"frame": 151.5, "frame_detector": 151.5, "side": "far"},  # detector off by +1.5f
        {"frame": 180.0, "frame_detector": 180.0, "side": "near"},
    ]
    retime_closest_approach(contacts, {0: inc, 1: out}, _StubCam(P), fps, "clay")
    assert abs(contacts[1]["frame"] - 150.0) < 0.2
    assert contacts[1]["retime_source"] == "closest_approach_image"
    assert 0.5 <= contacts[1]["timing_ci_frames"] <= 3.0


def test_audio_witness_flags_only_disagreement_beyond_tolerance():
    onsets = [(97.0, 20.0), (143.0, 15.0)]
    witness = {"mux_frames": -3.0, "witness_tol_frames": 3.0}
    # contact at 100 -> predicted onset 97, observed 97 -> agree, no flag
    onset, delta, abstain = audio_witness_check(100.0, onsets, witness)
    assert onset == 97.0 and abs(delta) < 1e-9 and abstain is False
    # contact at 150 -> predicted 147, nearest onset 143 (within 8f) -> delta -4 -> flag
    onset, delta, abstain = audio_witness_check(150.0, onsets, witness)
    assert onset == 143.0 and abstain is True
    # no onset within window -> not judged
    assert audio_witness_check(500.0, onsets, witness) == (None, None, False)


def test_implied_racket_face_follows_velocity_impulse():
    result = implied_racket(np.array([0.0, -20.0, -4.0]), np.array([3.0, 25.0, 6.0]))
    normal, speed, _, _ = result
    impulse = np.array([3.0, 45.0, 10.0])
    assert np.allclose(normal, impulse / np.linalg.norm(impulse))
    assert speed > 0
