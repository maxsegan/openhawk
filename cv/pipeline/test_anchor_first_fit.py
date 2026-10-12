import math
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from cv.pipeline import anchor_first_fit
from cv.pipeline.anchor_first_fit import (
    contact_adjacent_weights,
    fit_anchor_first_shot,
    held_out_summary,
    refine_anchor_first_point,
    simulate_measured_bounce_knot,
)
from cv.pipeline.flight_anchors import BALL_RADIUS_M, NET_COURT_Y_M
from cv.pipeline.rich_ball_physics import ShotFit, project_one
from physics.bounce_reference import DWELL_SECONDS, court_bounce
from physics import flight as flight_module
from physics.flight import rk4_step, sample_states


@contextmanager
def _striker_arm(*, prior: bool = True, authority: bool = True):
    """Turn the striker-witness arms on for one test, then put the shipped defaults back."""
    anchor_first_fit.configure_striker_witness(prior=prior, authority=authority)
    try:
        yield
    finally:
        anchor_first_fit.configure_striker_witness(prior=False, authority=True)


def test_the_shipped_striker_arms_are_authority_only():
    """The prior costs eight wrong accepted real flights; the authority arm costs none."""
    assert anchor_first_fit._STRIKER_PRIOR is False
    assert anchor_first_fit._STRIKER_AUTHORITY is True


def _projection() -> np.ndarray:
    camera_center = np.array([5.0, -12.0, 9.0])
    target = np.array([5.0, 12.0, 0.0])
    forward = (target - camera_center) / np.linalg.norm(target - camera_center)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsic = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
    return intrinsic @ np.c_[rotation, -rotation @ camera_center]


def _project(projection: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.r_[point, 1.0]
    return homogeneous[:2] / homogeneous[2]


def test_batched_low_spin_free_flight_jacobian_paths_match_scalar() -> None:
    # theta spin units are 100 rad/s: these include zero and both sides of the
    # former batch-only 1 rad/s threshold, including decay through that threshold.
    thetas = np.tile([5.0, 2.0, 3.0, 3.0, 30.0, 5.0, 0.0, 0.0, 0.0], (4, 1))
    thetas[:, 6] = [0.0, 1e-4, 0.005, 0.0101]
    frames = np.array([0.0, 5.0, 12.5, 20.0, 25.0])
    positions, continuities = anchor_first_fit._simulate_positions_batch(
        thetas, 0.0, frames, 25.0, "hard", None
    )
    for i, theta in enumerate(thetas):
        expected, _, _, _, continuity = anchor_first_fit._simulate(
            theta, 0.0, frames, 25.0, "hard", None
        )
        np.testing.assert_allclose(positions[i], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(continuities[i], continuity, rtol=1e-12, atol=1e-12)


def test_anchor_first_fit_scores_only_odd_frames() -> None:
    projection = _projection()
    frames = np.arange(0, 26)
    positions, _, _ = sample_states(
        np.array([5.0, 18.0, 1.2]),
        np.array([0.0, -15.0, 5.0]),
        np.zeros(3),
        frames / 25.0,
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    crossing_index = int(np.argmin(np.abs(positions[:, 1] - NET_COURT_Y_M)))
    before = crossing_index if positions[crossing_index, 1] > NET_COURT_Y_M else crossing_index - 1
    fraction = (NET_COURT_Y_M - positions[before, 1]) / (
        positions[before + 1, 1] - positions[before, 1]
    )
    crossing_frame = before + fraction
    crossing_xyz = (1.0 - fraction) * positions[before] + fraction * positions[before + 1]
    crossing_xyz[1] = NET_COURT_Y_M
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        h_at=lambda _frame: np.eye(3),
    )
    contacts = [
        {
            "frame": 0.0,
            "side": "far",
            "phase": "rally",
            "span": 0,
            "row": {},
        },
        {
            "frame": 25.0,
            "side": "near",
            "phase": "rally",
            "span": 0,
            "row": {},
        },
    ]
    fit = fit_anchor_first_shot(
        0,
        contacts,
        track,
        {"far": {0: np.array([5.0, 18.0])}, "near": {25: np.array([5.0, 3.0])}},
        camera,
        25.0,
        "hard",
        40,
        [
            {
                "type": "net_crossing",
                "frame": float(crossing_frame),
                "xyz": crossing_xyz.tolist(),
                "sigma_m": 0.02,
            }
        ],
    )

    assert fit is not None
    summary = held_out_summary(fit)
    assert summary["held_out_observations"] == 13
    assert summary["held_out_reprojection_median_px"] < 1.0
    assert summary["anchor_satisfied"]
    assert getattr(fit, "_spin_identifiable") is False
    assert np.array_equal(getattr(fit, "_held_out_frames"), frames[frames % 2 == 1])


def test_net_row_is_ignored_by_contact_and_bounce_fit() -> None:
    projection = _projection()
    frames = np.arange(0, 26)
    positions, _, _ = sample_states(
        np.array([5.0, 18.0, 1.2]),
        np.array([0.0, -15.0, 5.0]),
        np.zeros(3),
        frames / 25.0,
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    crossing = next(
        index
        for index in range(len(frames) - 1)
        if (positions[index, 1] - NET_COURT_Y_M) * (positions[index + 1, 1] - NET_COURT_Y_M) <= 0.0
    )
    fraction = (NET_COURT_Y_M - positions[crossing, 1]) / (
        positions[crossing + 1, 1] - positions[crossing, 1]
    )
    true_frame = float(crossing + fraction)
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {"frame": 25.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    fit = fit_anchor_first_shot(
        0,
        contacts,
        track,
        {"far": {0: np.array([5.0, 18.0])}, "near": {25: np.array([5.0, 3.0])}},
        camera,
        25.0,
        "hard",
        60,
        [
            {
                "type": "net_crossing",
                "frame": true_frame + 0.35,
                "crossing_frame_bounds": [float(crossing), float(crossing + 1)],
                "crossing_direction_y": -1,
                # Deliberately impossible bootstrap point: the constraint arm must
                # not use its x/z coordinates as an optimization residual.
                "xyz": [20.0, NET_COURT_Y_M, 7.0],
                "sigma_m": 0.005,
            }
        ],
        net_point_anchor=False,
    )

    assert fit is not None
    without_net = fit_anchor_first_shot(
        0,
        contacts,
        track,
        {"far": {0: np.array([5.0, 18.0])}, "near": {25: np.array([5.0, 3.0])}},
        camera,
        25.0,
        "hard",
        60,
        [],
    )
    assert without_net is not None
    assert getattr(fit, "_net_constraint")["mode"] == "post_fit_plausibility_only"
    np.testing.assert_allclose(fit.theta, without_net.theta, atol=1e-8)
    assert held_out_summary(fit)["anchor_satisfied"]


@pytest.mark.parametrize("fps", [24.0, 25.0, 50.0, 59.94])
def test_measured_fit_state_matches_its_objective_before_during_and_after_dwell(fps):
    bounce_frame = 30.25
    knot = {"frame": bounce_frame, "x": np.array([5.0, 17.0, BALL_RADIUS_M])}
    velocity, spin = np.array([1.0, -25.0, -5.0]), np.array([180.0, 10.0, 30.0])
    query = bounce_frame + fps * np.array([-0.3, 0.0, DWELL_SECONDS / 2, DWELL_SECONDS, 0.6])
    xyz, vel, _spin, bounces, initial = simulate_measured_bounce_knot(
        velocity, spin, query[0], query, fps, "hard", knot
    )
    rebound = court_bounce(velocity, spin, "hard")
    fit = ShotFit(0, 1, np.concatenate(initial), 0.0, len(query), xyz, vel, query, bounces, 1)
    fit._f0 = float(query[0])
    fit._bounce_node_split = {
        "frame": bounce_frame,
        "xyz": knot["x"],
        "incoming_velocity": velocity,
        "incoming_spin": spin,
        "outgoing_velocity": rebound.velocity,
        "outgoing_spin": rebound.spin,
        "sampling_model": "measured_bounce_v1",
        "dwell_seconds": DWELL_SECONDS,
    }
    for index, frame in enumerate(query):
        position, speed = fit.state(float(frame), fps, "hard")
        np.testing.assert_allclose(position, xyz[index], atol=1e-6, rtol=0)
        np.testing.assert_allclose(speed, vel[index], atol=1e-6, rtol=0)


def test_terminal_flight_fits_first_bounce_knot_and_second_bounce_anchor() -> None:
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 25.0
    first_frame = 20.0
    first_xyz = np.array([5.2, 15.0, BALL_RADIUS_M])
    incoming_velocity = np.array([0.5, -11.0, -4.5])
    incoming_spin = np.zeros(3)
    rebound = court_bounce(incoming_velocity, incoming_spin, "hard")
    elapsed = np.linspace(0.05, 2.0, 1000)
    post_positions, _, _ = sample_states(first_xyz, rebound.velocity, rebound.spin, elapsed)
    descending = np.flatnonzero((elapsed > 0.15) & (post_positions[:, 2] <= BALL_RADIUS_M))
    assert len(descending)
    terminal_elapsed = float(elapsed[descending[0]]) + DWELL_SECONDS
    terminal_frame = first_frame + terminal_elapsed * fps
    terminal_xyz = post_positions[descending[0]].copy()
    terminal_xyz[2] = BALL_RADIUS_M
    frames = np.arange(0, math.floor(terminal_frame) + 1)
    positions, _, _, _, _ = simulate_measured_bounce_knot(
        incoming_velocity,
        incoming_spin,
        0.0,
        frames.astype(float),
        fps,
        "hard",
        {"frame": first_frame, "x": first_xyz},
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {
            "frame": terminal_frame,
            "side": "near",
            "phase": "terminal",
            "span": 0,
            "terminal": True,
            "row": {},
        },
    ]

    fit = fit_anchor_first_shot(
        0,
        contacts,
        track,
        {"far": {0: np.array([5.0, 22.0])}},
        camera,
        fps,
        "hard",
        100,
        [
            {"type": "bounce", "frame": first_frame, "xyz": first_xyz, "sigma_m": 0.01},
            {
                "type": "bounce",
                "frame": terminal_frame,
                "xyz": terminal_xyz,
                "sigma_m": 0.01,
            },
        ],
    )

    assert fit is not None
    assert len(fit.bounces) == 2
    assert fit._bounce_node_split["sampling_model"] == "measured_bounce_v1"
    assert fit.bounces[0]["dwell_seconds"] == DWELL_SECONDS
    for index, frame in enumerate(fit.obs_frames):
        np.testing.assert_allclose(
            fit.state(float(frame), fps, "hard")[0], fit.xs_obs[index], atol=1e-6, rtol=0
        )
    assert fit.bounces[1]["termination_anchor"] is True
    terminal_error = next(
        row["error_m"] for row in fit._anchor_errors if row["type"] == "terminal_bounce"
    )
    assert terminal_error < 0.05
    assert held_out_summary(fit)["anchor_satisfied"]

    from cv.pipeline.reconstruction import compact_fit

    exported = compact_fit(fit, fps, "hard", start_frame=0.0, end_frame=terminal_frame)
    assert exported["bounces"][0]["sampling_model"] == "measured_bounce_v1"
    assert exported["bounces"][0]["dwell_seconds"] == DWELL_SECONDS
    for row in exported["trajectory"]:
        np.testing.assert_allclose(
            row["xyz"], fit.state(row["frame"], fps, "hard")[0], atol=1e-12, rtol=0
        )


def test_fixed_start_bounce_refit_tries_endpoint_consistent_seed_first(monkeypatch) -> None:
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 25.0
    fixed_start = np.array([5.0, 18.0, 1.2])
    bounce_frame = 15.0
    bounce_xyz = np.array([5.2, 10.0, BALL_RADIUS_M])
    frames = np.arange(0, 26)
    visible_positions, _, _ = sample_states(
        fixed_start,
        np.array([0.3, -14.0, 4.0]),
        np.zeros(3),
        frames / fps,
    )
    track = {
        int(frame): _project(projection, xyz)
        for frame, xyz in zip(frames, visible_positions, strict=True)
    }
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {"frame": 25.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    wrong_velocity = np.array([40.0, 30.0, -1.0])
    incumbent = SimpleNamespace(
        bounces=[{"v_in": wrong_velocity, "w_in": np.zeros(3)}],
        state=lambda _frame, _fps, _surface: (np.array([20.0, 25.0, 3.0]), wrong_velocity),
    )
    attempted_seeds = []

    def reject_fit(_residual, seed, **_kwargs):
        attempted_seeds.append(np.asarray(seed, float))
        raise ValueError("capture seed only")

    monkeypatch.setattr(anchor_first_fit, "least_squares", reject_fit)

    fit = fit_anchor_first_shot(
        0,
        contacts,
        track,
        {},
        camera,
        fps,
        "hard",
        20,
        [{"type": "bounce", "frame": bounce_frame, "xyz": bounce_xyz}],
        fixed_start_anchor=fixed_start,
        initial_fit=incumbent,
    )

    assert fit is None
    interval = bounce_frame / fps
    expected_velocity = (bounce_xyz - fixed_start) / interval
    expected_velocity[2] -= 0.5 * 9.81 * interval
    np.testing.assert_allclose(attempted_seeds[0][:3], expected_velocity)
    assert any(np.allclose(seed[:3], wrong_velocity) for seed in attempted_seeds)


def test_joint_refit_does_not_treat_shared_contacts_as_outliers(monkeypatch) -> None:
    """The shared state is a structural constraint, unlike a noisy image observation."""
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    frames = np.arange(0, 26)
    start = np.array([5.0, 18.0, 1.2])
    positions, _, _ = sample_states(start, np.array([0.0, -15.0, 5.0]), np.zeros(3), frames / 25.0)
    track = {int(f): _project(projection, x) for f, x in zip(frames, positions, strict=True)}
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally"},
        {"frame": 25.0, "side": "near", "phase": "rally"},
    ]
    losses = []

    def capture(residual, seed, **kwargs):
        values = residual(seed)
        loss = kwargs["loss"]
        assert callable(loss), "a blanket robust loss downweights shared-state violations"
        # Endpoint refits exclude the contact frames; twelve even interior observations.
        shared = slice(24, 30)
        z = np.full(len(values), 10000.0)
        rho = loss(z)
        np.testing.assert_array_equal(rho[0, shared], z[shared])
        np.testing.assert_array_equal(rho[1, shared], np.ones(6))
        np.testing.assert_array_equal(rho[2, shared], np.zeros(6))
        assert rho[1, 0] == pytest.approx(1 / np.sqrt(10001.0))
        losses.append(loss)
        raise ValueError("capture only")

    monkeypatch.setattr(anchor_first_fit, "least_squares", capture)
    with _joint_arm():
        fit_anchor_first_shot(
            0,
            contacts,
            track,
            {},
            camera,
            25.0,
            "hard",
            20,
            [],
            fixed_start_anchor=start,
            fixed_end_anchor=positions[-1],
        )
    assert losses


def test_per_observation_timing_nuisance_reduces_shifted_held_out_error() -> None:
    projection = _projection()
    frames = np.arange(0, 26)
    initial_position = np.array([5.0, 18.0, 1.2])
    initial_velocity = np.array([0.0, -15.0, 5.0])
    positions, _, _ = sample_states(
        initial_position,
        initial_velocity,
        np.zeros(3),
        (frames + 0.35) / 25.0,
    )
    unshifted, _, _ = sample_states(
        initial_position,
        initial_velocity,
        np.zeros(3),
        frames / 25.0,
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    crossing_index = int(np.argmin(np.abs(unshifted[:, 1] - NET_COURT_Y_M)))
    before = crossing_index if unshifted[crossing_index, 1] > NET_COURT_Y_M else crossing_index - 1
    fraction = (NET_COURT_Y_M - unshifted[before, 1]) / (
        unshifted[before + 1, 1] - unshifted[before, 1]
    )
    crossing_frame = before + fraction
    crossing_xyz = (1.0 - fraction) * unshifted[before] + fraction * unshifted[before + 1]
    crossing_xyz[1] = NET_COURT_Y_M
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {"frame": 25.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    arguments = (
        0,
        contacts,
        track,
        {"far": {0: np.array([5.0, 18.0])}, "near": {25: np.array([5.0, 3.0])}},
        camera,
        25.0,
        "hard",
        60,
        [
            {
                "type": "net_crossing",
                "frame": float(crossing_frame),
                "xyz": crossing_xyz.tolist(),
                "sigma_m": 0.02,
            }
        ],
    )
    nominal = fit_anchor_first_shot(*arguments, net_point_anchor=True)
    timed = fit_anchor_first_shot(
        *arguments,
        timing_nuisance="per_observation",
        net_point_anchor=True,
    )

    assert nominal is not None
    assert timed is not None
    offsets = np.asarray(getattr(timed, "_observation_time_offsets_frames"))
    assert np.max(np.abs(offsets)) <= anchor_first_fit.TIMING_OFFSET_LIMIT_FRAMES
    assert held_out_summary(timed)["held_out_reprojection_median_px"] < 1.0
    offsets = getattr(timed, "_observation_time_offsets_frames")
    assert np.max(np.abs(offsets)) <= 0.5
    assert getattr(timed, "_timing_nuisance") == "per_observation"


def test_contact_adjacent_weighting_selects_two_fit_nodes_per_contact() -> None:
    frames = np.asarray([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    baseline = contact_adjacent_weights(frames, (10.0, 20.0), enabled=False)
    weighted = contact_adjacent_weights(frames, (10.0, 20.0), enabled=True)

    assert np.array_equal(baseline, np.ones(6))
    assert np.allclose(weighted, [0.1, 0.1, 1.0, 1.0, 0.1, 0.1])


def test_legacy_height_on_ray_cannot_reactivate_net_anchoring(monkeypatch) -> None:
    direct = object()
    legacy = object()
    monkeypatch.setattr(anchor_first_fit, "_camera_space_fit", lambda **_kwargs: direct)
    monkeypatch.setattr(
        anchor_first_fit,
        "_fit_legacy_height_on_ray_shot",
        lambda **_kwargs: legacy,
    )
    arguments = (0, [{}, {}], {}, {}, object(), 25.0, "hard", 20, [])
    try:
        anchor_first_fit.configure_legacy_height_on_ray(False)
        assert fit_anchor_first_shot(*arguments) is direct
        anchor_first_fit.configure_legacy_height_on_ray(True)
        assert fit_anchor_first_shot(*arguments) is direct
    finally:
        anchor_first_fit.configure_legacy_height_on_ray(False)


class _PairFit:
    def __init__(self, index: int, context: dict, start: np.ndarray, end: np.ndarray):
        self.index = index
        self.theta = np.r_[start, np.zeros(6)]
        self._start = np.asarray(start, float)
        self._end = np.asarray(end, float)
        self._anchor_first_context = context
        self._held_out_errors_px = np.asarray([2.0, 3.0, 4.0])
        self._anchor_errors = [{"error_m": 0.02}]
        self._net_anchor_available = True
        self.rms_px = 2.0

    def state(self, frame: float, _fps: float, _surface: str):
        contact = float(self._anchor_first_context["contacts"][1]["frame"])
        position = self._start if frame <= contact and self.index == 1 else self._end
        return position, np.zeros(3)


def test_contact_authority_distrusts_bounce_free_depth() -> None:
    context = {"contacts": [{}, {"frame": 10.0}]}
    anchored = _PairFit(0, context, np.zeros(3), np.ones(3))
    free = _PairFit(1, context, np.zeros(3), np.ones(3))
    anchored.bounces = [{"frame": 5.0}]
    free.bounces = []

    assert anchor_first_fit._contact_authority_score(
        anchored
    ) < anchor_first_fit._contact_authority_score(free)


def test_stronger_contact_authority_can_replace_wrong_depth_pixel_minimum() -> None:
    context = {"contacts": [{}, {"frame": 10.0}]}
    authority = _PairFit(0, context, np.zeros(3), np.ones(3))
    incumbent = _PairFit(1, context, np.zeros(3), np.ones(3))
    candidate = _PairFit(1, context, np.zeros(3), np.ones(3))
    authority.bounces = [{"frame": 5.0}]
    incumbent.bounces = []
    candidate.bounces = []
    incumbent._held_out_errors_px = np.asarray([0.1, 0.2, 0.3])
    candidate._held_out_errors_px = np.asarray([4.0, 5.0, 6.0])

    safe, reason = anchor_first_fit._one_sided_candidate_safe(authority, incumbent, candidate)

    assert safe
    assert reason == "stronger_authority_absolute_held_out"


def test_shared_contact_preserves_pixel_safe_authority_endpoint() -> None:
    projection = _projection()
    endpoint = np.array([5.0, 10.0, 1.2])
    pixel = _project(projection, endpoint) + np.array([4.0, -3.0])

    shared, error_px = anchor_first_fit._authority_shared_contact(projection, pixel, endpoint)

    np.testing.assert_allclose(shared, endpoint)
    assert math.isclose(error_px, 5.0)


def test_shared_contact_can_use_explicit_marginal_pixel_limit() -> None:
    projection = _projection()
    endpoint = np.array([5.0, 10.0, 1.2])
    pixel = _project(projection, endpoint) + np.array([18.0, 0.0])

    default_shared, _ = anchor_first_fit._authority_shared_contact(projection, pixel, endpoint)
    relaxed_shared, error_px = anchor_first_fit._authority_shared_contact(
        projection, pixel, endpoint, max_error_px=24.0
    )

    assert default_shared is None
    np.testing.assert_allclose(relaxed_shared, endpoint)
    assert math.isclose(error_px, 18.0)


def test_shared_contact_pair_is_adopted_atomically(monkeypatch) -> None:
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contact_points = (
        np.array([5.0, 15.0, 1.0]),
        np.array([5.0, 10.0, 1.0]),
    )
    contacts = [
        {
            "frame": 0.0,
            "event_frame": 0.0,
            "side": "far",
            "phase": "rally",
            "image_observation_override": _project(projection, contact_points[0]),
        },
        {
            "frame": 10.0,
            "event_frame": 10.0,
            "side": "near",
            "phase": "rally",
            "image_observation_sigma_px": 18.0,
            "image_observation_override": _project(projection, contact_points[1]),
        },
        {"frame": 20.0, "terminal": True},
    ]
    context = {
        "contacts": contacts,
        "ball": {
            0: _project(projection, contact_points[0]),
            10: _project(projection, contact_points[1]),
        },
        "players": {
            "far": {0: contact_points[0][:2]},
            "near": {10: contact_points[1][:2]},
        },
        "camera": camera,
        "fps": 25.0,
        "surface": "hard",
    }
    incoming_context = {**context, "index": 0}
    outgoing_context = {**context, "index": 1}
    incoming = _PairFit(0, incoming_context, np.array([5.0, 15.0, 1.0]), np.array([4.0, 9.0, 1.0]))
    outgoing = _PairFit(1, outgoing_context, np.array([6.0, 11.0, 1.0]), np.array([5.0, 5.0, 1.0]))

    calls = []

    def fake_fit(**kwargs):
        calls.append(kwargs)
        index = kwargs["index"]
        if index == 0:
            start = kwargs.get("fixed_start_anchor", incoming._start)
            end = kwargs.get("fixed_end_anchor", incoming._end)
            return _PairFit(index, incoming_context, start, end)
        start = kwargs.get("fixed_start_anchor", outgoing._start)
        end = kwargs.get("fixed_end_anchor", outgoing._end)
        return _PairFit(index, outgoing_context, start, end)

    monkeypatch.setattr(anchor_first_fit, "fit_anchor_first_shot", fake_fit)
    refined = refine_anchor_first_point({0: incoming, 1: outgoing})

    assert refined[0] is not incoming
    assert refined[1] is not outgoing
    incoming_contact = refined[0].state(10.0, 25.0, "hard")[0]
    outgoing_contact = refined[1].state(10.0, 25.0, "hard")[0]
    assert np.allclose(incoming_contact, outgoing_contact)
    incoming_record = next(row for row in refined[0]._shared_contacts if row["contact_index"] == 1)
    outgoing_record = next(row for row in refined[1]._shared_contacts if row["contact_index"] == 1)
    assert incoming_record == outgoing_record
    assert abs(incoming_record["time_offset_frames"]) <= 1.0
    assert incoming_record["observation_sigma_px"] == 18.0
    assert incoming_record["position_prior_sigma_m"] > 0.04
    assert any(
        call.get("fixed_end_sigma_m") == incoming_record["position_prior_sigma_m"] for call in calls
    )
    assert any(
        call.get("fixed_start_sigma_m") == incoming_record["position_prior_sigma_m"]
        for call in calls
    )


def test_failed_soft_contact_refit_preserves_incumbent_flights(monkeypatch) -> None:
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contacts = [
        {
            "frame": 0.0,
            "event_frame": 0.0,
            "side": "far",
            "phase": "rally",
            "image_observation_override": _project(projection, np.array([5.0, 15.0, 1.0])),
        },
        {
            "frame": 10.0,
            "event_frame": 10.0,
            "side": "near",
            "phase": "rally",
            "image_observation_override": _project(projection, np.array([5.0, 10.0, 1.0])),
        },
        {"frame": 20.0, "terminal": True},
    ]
    context = {
        "contacts": contacts,
        "ball": {},
        "players": {},
        "camera": camera,
        "fps": 25.0,
        "surface": "hard",
    }
    incoming = _PairFit(
        0,
        {**context, "index": 0},
        np.array([5.0, 15.0, 1.0]),
        np.array([4.0, 9.0, 1.0]),
    )
    outgoing = _PairFit(
        1,
        {**context, "index": 1},
        np.array([6.0, 11.0, 1.0]),
        np.array([5.0, 5.0, 1.0]),
    )
    diagnostics = []
    monkeypatch.setattr(anchor_first_fit, "fit_anchor_first_shot", lambda **_kwargs: None)

    refined = refine_anchor_first_point({0: incoming, 1: outgoing}, diagnostics)

    assert refined == {0: incoming, 1: outgoing}
    assert any(row["status"] == "no_safe_soft_anchor_refit" for row in diagnostics)


def test_one_sided_serve_is_not_misrepresented_as_shared_contact(monkeypatch) -> None:
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    visible_contact = np.array([5.0, 15.0, 2.75])
    contacts = [
        {
            "frame": 0.0,
            "event_frame": 0.0,
            "side": "far",
            "phase": "serve",
            "image_observation_override": _project(projection, visible_contact),
        },
        {"frame": 12.0, "terminal": True},
    ]
    context = {
        "index": 0,
        "contacts": contacts,
        "ball": {0: _project(projection, visible_contact)},
        "players": {"far": {0: visible_contact[:2]}},
        "camera": camera,
        "fps": 25.0,
        "surface": "hard",
    }

    class ServeFit(_PairFit):
        def state(self, _frame: float, _fps: float, _surface: str):
            return self._start, np.zeros(3)

    initial = ServeFit(0, context, visible_contact, np.array([5.0, 5.0, 1.0]))

    monkeypatch.setattr(
        anchor_first_fit,
        "fit_anchor_first_shot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("one-sided contact refit")),
    )
    refined = refine_anchor_first_point({0: initial})

    assert refined == {0: initial}
    assert not hasattr(refined[0], "_shared_contacts")


def _restarting_signed_states(
    position: np.ndarray,
    velocity: np.ndarray,
    spin: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pre-optimisation sampler: restart the integration at every target."""
    positions = []
    velocities = []
    spins = []
    for target in np.asarray(times, float):
        x = np.asarray(position, float).copy()
        v = np.asarray(velocity, float).copy()
        w = np.asarray(spin, float).copy()
        remaining = float(target)
        while abs(remaining) > 1e-12:
            step = math.copysign(min(0.01, abs(remaining)), remaining)
            x, v, w = rk4_step(x, v, w, step)
            remaining -= step
        positions.append(x)
        velocities.append(v)
        spins.append(w)
    return np.asarray(positions), np.asarray(velocities), np.asarray(spins)


def test_shared_step_grid_reproduces_the_restarting_sampler_exactly() -> None:
    generator = np.random.default_rng(20260903)
    for trial in range(12):
        position = generator.uniform([-10.0, -12.0, 0.1], [10.0, 12.0, 3.0])
        velocity = generator.uniform([-30.0, -30.0, -20.0], [30.0, 30.0, 20.0])
        spin = generator.uniform(-300.0, 300.0, 3)
        times = generator.uniform(-0.9, 0.9, int(generator.integers(1, 30)))
        # Zero, an exact whole step, and a sub-step remainder are the edges of the
        # step schedule the grid has to reproduce.
        times = np.concatenate([times, [0.0, 0.01, -0.01, 0.05, -0.05, 0.1234]])
        expected = _restarting_signed_states(position, velocity, spin, times)
        measured = anchor_first_fit._sample_signed_states(position, velocity, spin, times)
        for reference, value in zip(expected, measured, strict=True):
            assert np.array_equal(reference, value), trial


def test_shared_step_grid_integrates_each_direction_once() -> None:
    calls = {"count": 0}
    original = flight_module.rk4_step

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    times = np.linspace(-0.5, 0.5, 21)
    flight_module.rk4_step = counted
    try:
        anchor_first_fit._sample_signed_states(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 20.0, 3.0]),
            np.zeros(3),
            times,
        )
    finally:
        flight_module.rk4_step = original
    # Restarting at every target costs 550 steps for this query; sharing the whole
    # steps costs at most one grid per direction plus one partial step per target.
    assert calls["count"] <= 2 * 50 + len(times)


def test_endpoint_anchored_bounce_free_flight_fits_spin() -> None:
    """A lob refitted onto a neighbour's contact needs spin to reach the pixels."""
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 50.0
    start = np.array([1.0, 3.0, 1.4])
    velocity = np.array([1.5, 11.0, 9.0])
    spin = np.array([-260.0, 0.0, 0.0])
    frames = np.arange(0, 41, dtype=float)
    positions, _, _ = sample_states(start, velocity, spin, frames / fps)
    ball = {
        int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions, strict=True)
    }
    contacts = [
        {"frame": 0.0, "side": "near", "phase": "rally"},
        {"frame": 40.0, "side": "far", "phase": "rally", "terminal": True},
    ]
    anchored = fit_anchor_first_shot(
        index=0,
        contacts=contacts,
        ball=ball,
        players={},
        camera=camera,
        fps=fps,
        surface="hard",
        max_nfev=40,
        anchors=[],
        fixed_start_anchor=start,
    )
    assert anchored is not None
    assert getattr(anchored, "_spin_identifiable") is True
    assert float(np.linalg.norm(anchored.theta[6:9])) > 0.0
    assert float(np.median(getattr(anchored, "_held_out_errors_px"))) < 2.0

    unanchored = fit_anchor_first_shot(
        index=0,
        contacts=contacts,
        ball=ball,
        players={},
        camera=camera,
        fps=fps,
        surface="hard",
        max_nfev=40,
        anchors=[],
    )
    # Without a fixed endpoint the depth is still free, so the flight keeps the
    # zero-spin parameters the rest of the pipeline was measured with.
    assert unanchored is not None
    assert getattr(unanchored, "_spin_identifiable") is False
    assert float(np.linalg.norm(unanchored.theta[6:9])) == 0.0


def test_the_default_arm_keeps_the_shipped_single_reach_hinge():
    player = np.array([5.0, 3.0])
    # Three metres from the body is 0.9 m past the shipped 2.1 m hinge, and nothing else is said.
    assert np.allclose(
        anchor_first_fit.striker_contact_residuals(
            np.array([5.0, 6.0, 1.1]), player, "rally", "near"
        ),
        [(3.0 - anchor_first_fit.MAX_CONTACT_REACH_M) / 0.25, 0.0],
    )
    # The shared-contact search passes its own tighter serve hinge, and being behind the body is
    # still free.
    assert np.allclose(
        anchor_first_fit.striker_contact_residuals(
            np.array([5.0, 0.0, 2.8]),
            player,
            "serve",
            "near",
            legacy_reach_m=anchor_first_fit.SERVE_CONTACT_REACH_M,
        ),
        [(3.0 - anchor_first_fit.SERVE_CONTACT_REACH_M) / 0.25, 0.0],
    )


def test_striker_contact_residuals_are_silent_inside_the_band():
    with _striker_arm():
        player = np.array([5.0, 3.0])
        # 0.8 m in front of a near-court body: a plain rally contact, so the witness says nothing.
        assert np.allclose(
            anchor_first_fit.striker_contact_residuals(
                np.array([5.0, 3.8, 1.1]), player, "rally", "near"
            ),
            0.0,
        )
        # A serve is struck almost over the body, so the rally band would be wrong for it.
        assert np.allclose(
            anchor_first_fit.striker_contact_residuals(
                np.array([5.0, 3.3, 2.8]), player, "serve", "near"
            ),
            0.0,
        )
        assert np.allclose(
            anchor_first_fit.striker_contact_residuals(
                np.array([5.0, 3.8, 1.1]), None, "rally", "near"
            ),
            0.0,
        )


def test_striker_contact_residuals_charge_reach_and_being_behind_the_body():
    with _striker_arm():
        player = np.array([5.0, 3.0])
        sigma = anchor_first_fit.STRIKER_ROOT_SIGMA_M["near"]
        reach_high = anchor_first_fit.STRIKER_REACH_BAND_M[1]
        slack = anchor_first_fit.STRIKER_BEHIND_SLACK_M
        # Three metres down the contact ray from the body: too far, but still in front.
        far = anchor_first_fit.striker_contact_residuals(
            np.array([5.0, 6.0, 1.1]), player, "rally", "near", sigma
        )
        assert far[0] == pytest.approx((3.0 - reach_high) / sigma)
        assert far[1] == 0.0
        # The same distance on the other side of the body: charged for reach and for being behind.
        behind = anchor_first_fit.striker_contact_residuals(
            np.array([5.0, 0.0, 1.1]), player, "rally", "near", sigma
        )
        assert behind[0] == pytest.approx((3.0 - reach_high) / sigma)
        assert behind[1] == pytest.approx((3.0 - slack) / sigma)
        # "Behind" is toward that striker's own baseline, so the far end has the opposite sign.
        assert anchor_first_fit.striker_contact_residuals(
            np.array([5.0, 6.0, 1.1]), player, "rally", "far", sigma
        )[1] == pytest.approx((3.0 - slack) / sigma)


def test_striker_witness_pulls_a_first_flight_off_a_wrong_depth_branch():
    """A bounce-free first flight has no depth witness but the striker, and must use it."""
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 25.0
    truth_start = np.array([6.0, 2.5, 1.2])
    velocity = np.array([-1.0, 22.0, 4.0])
    frames = np.arange(0.0, 13.0)
    positions, _, _ = sample_states(truth_start, velocity, np.zeros(3), frames / fps)
    ball = {
        int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions, strict=True)
    }
    contacts = [
        {"frame": 0.0, "side": "near", "phase": "rally"},
        {"frame": 12.0, "side": "far", "phase": "rally", "terminal": True},
    ]
    # The body is 0.75 m behind the contact; the same image ray also passes metres further away.
    players = {"near": {frame: np.array([6.1, 1.75]) for frame in range(0, 14)}, "far": {}}
    kwargs = {
        "index": 0,
        "contacts": contacts,
        "ball": ball,
        "camera": camera,
        "fps": fps,
        "surface": "hard",
        "max_nfev": 40,
        "anchors": [],
    }
    without = fit_anchor_first_shot(players={}, **kwargs)
    with _striker_arm():
        with_striker = fit_anchor_first_shot(players=players, **kwargs)
    assert without is not None and with_striker is not None
    witnessed = float(np.linalg.norm(with_striker.theta[:3] - truth_start))
    blind = float(np.linalg.norm(without.theta[:3] - truth_start))
    assert witnessed <= blind


def test_striker_witness_relaxes_the_authority_contact_pixel_gate():
    """An authority the striker agrees with survives a contact pixel the automatic track fumbles."""
    projection = _projection()
    # The authority endpoint is at a racket's reach from the body; the weak one is four metres
    # further down the same court, which no racket reaches.
    player = np.array([6.0, 2.0])
    authority_endpoint = np.array([6.0, 2.8, 1.2])
    weak_endpoint = np.array([6.0, 6.8, 1.2])
    with _striker_arm():
        residuals = [
            float(
                np.linalg.norm(
                    anchor_first_fit.striker_contact_residuals(
                        endpoint,
                        player,
                        "rally",
                        "near",
                        anchor_first_fit.STRIKER_ROOT_SIGMA_M["near"],
                    )
                )
            )
            for endpoint in (authority_endpoint, weak_endpoint)
        ]
    assert residuals[0] == 0.0 < residuals[1]
    # The refined contact observation sits 15 px from the authority: past the 12 px authority gate
    # but inside the 24 px marginal limit the striker agreement unlocks.
    authority_pixel = project_one(projection, authority_endpoint)
    observation = authority_pixel + np.array([15.0, 0.0])
    strict, strict_error = anchor_first_fit._authority_shared_contact(
        projection,
        observation,
        authority_endpoint,
        max_error_px=anchor_first_fit.HELD_OUT_P90_LIMIT_PX,
    )
    relaxed, relaxed_error = anchor_first_fit._authority_shared_contact(
        projection,
        observation,
        authority_endpoint,
        max_error_px=anchor_first_fit.ONE_SIDED_CONTACT_MARGINAL_PIXEL_LIMIT_PX,
    )
    assert strict is None
    assert relaxed is not None
    assert strict_error == pytest.approx(relaxed_error)
    assert (
        anchor_first_fit.HELD_OUT_P90_LIMIT_PX
        < relaxed_error
        <= anchor_first_fit.ONE_SIDED_CONTACT_MARGINAL_PIXEL_LIMIT_PX
    )


def test_a_stale_striker_row_widens_the_prior_and_then_drops_it():
    players = {"near": {40: np.array([6.0, 2.0])}, "far": {}}
    fresh, gap = anchor_first_fit.striker_witness(players, "near", 40.0)
    assert gap == 0.0 and np.allclose(fresh, [6.0, 2.0])
    assert anchor_first_fit.striker_contact_sigma_m(0.0, "near") == pytest.approx(
        anchor_first_fit.STRIKER_ROOT_SIGMA_M["near"]
    )
    # The far half is twice as many metres per pixel, so the same root error is a wider band.
    assert anchor_first_fit.striker_contact_sigma_m(
        0.0, "far"
    ) > anchor_first_fit.striker_contact_sigma_m(0.0, "near")
    # Four frames of unseen body is four frames of possible walking, so the band softens further.
    assert anchor_first_fit.striker_contact_sigma_m(4.0, "near") == pytest.approx(
        anchor_first_fit.STRIKER_ROOT_SIGMA_M["near"]
        + 4.0 * anchor_first_fit.STRIKER_DRIFT_M_PER_FRAME
    )
    # Past the limit the witness is no witness at all.
    assert (
        anchor_first_fit.striker_contact_sigma_m(
            anchor_first_fit.STRIKER_WITNESS_MAX_GAP_FRAMES + 1.0, "near"
        )
        is None
    )
    with _striker_arm():
        assert np.allclose(
            anchor_first_fit.striker_contact_residuals(
                np.array([6.0, 9.0, 1.1]), fresh, "rally", "near", None
            ),
            0.0,
        )
        stale = anchor_first_fit.striker_contact_residuals(
            np.array([6.0, 9.0, 1.1]),
            fresh,
            "rally",
            "near",
            anchor_first_fit.striker_contact_sigma_m(4.0, "near"),
        )
        sharp = anchor_first_fit.striker_contact_residuals(
            np.array([6.0, 9.0, 1.1]),
            fresh,
            "rally",
            "near",
            anchor_first_fit.striker_contact_sigma_m(0.0, "near"),
        )
    assert 0.0 < float(stale[0]) < float(sharp[0])
    assert anchor_first_fit.striker_witness({"near": {}}, "near", 40.0) == (None, math.inf)


@contextmanager
def _objective_arm(*, contact_observation: bool = False, spin_prior: bool = False):
    """Turn the pointfit6 objective arms on for one test, then put the shipped defaults back."""
    anchor_first_fit.configure_contact_observation_witness(contact_observation)
    anchor_first_fit.configure_physical_spin_prior(spin_prior)
    try:
        yield
    finally:
        anchor_first_fit.configure_contact_observation_witness(False)
        anchor_first_fit.configure_physical_spin_prior(False)


def test_the_pointfit6_objective_arms_ship_off() -> None:
    """Both are measured in WK3_REPORT.md section 3 before either is promoted."""
    assert anchor_first_fit._CONTACT_OBSERVATION_WITNESS is False
    assert anchor_first_fit._PHYSICAL_SPIN_PRIOR is False


def _corner_track() -> tuple[dict[int, np.ndarray], float, np.ndarray]:
    """A tracked ball that turns a corner at a fractional contact frame, as a racket makes it."""
    contact_frame = 10.4
    contact_pixel = np.array([500.0, 400.0])
    incoming = np.array([40.0, 12.0])
    outgoing = np.array([-38.0, -20.0])
    track = {}
    for frame in range(6, 11):
        track[frame] = contact_pixel - incoming * (contact_frame - frame)
    for frame in range(11, 16):
        track[frame] = contact_pixel + outgoing * (frame - contact_frame)
    return track, contact_frame, contact_pixel


def test_the_contact_witness_beats_the_chord_across_the_impact() -> None:
    track, contact_frame, contact_pixel = _corner_track()
    chord, sources = anchor_first_fit.interpolate_track(track, contact_frame)
    assert len(sources) == 2

    pixel, frame, sigma, source = anchor_first_fit.contact_observation(
        {"frame": contact_frame}, track
    )

    assert source == "two_sided_tangent"
    assert frame == contact_frame
    # Each side is a straight line here, so both tangents land exactly on the contact.
    assert np.linalg.norm(pixel - contact_pixel) < 1e-6
    assert np.linalg.norm(chord - contact_pixel) > 20.0
    assert sigma == anchor_first_fit.CONTACT_OBSERVATION_MIN_SIGMA_PX


def test_the_contact_witness_prefers_the_contact_row_s_own_observation() -> None:
    track, contact_frame, _ = _corner_track()
    contact = {
        "frame": contact_frame,
        "image_observation_override": [501.0, 402.0],
        "image_observation_frame": 10.0,
        "image_observation_sigma_px": 6.5,
        "image_observation_source": "contact_frame_refiner",
    }

    pixel, frame, sigma, source = anchor_first_fit.contact_observation(contact, track)

    assert source == "contact_frame_refiner"
    assert np.allclose(pixel, [501.0, 402.0])
    assert frame == 10.0
    assert sigma == 6.5


def test_a_contact_on_an_observed_frame_is_believed() -> None:
    track, _, _ = _corner_track()

    pixel, frame, sigma, source = anchor_first_fit.contact_observation({"frame": 12.0}, track)

    assert source == "observed_frame"
    assert np.allclose(pixel, track[12])
    assert sigma == anchor_first_fit.CONTACT_OBSERVATION_MIN_SIGMA_PX


def test_an_observation_with_no_error_bar_is_worth_ten_pixels() -> None:
    track, contact_frame, _ = _corner_track()
    contact = {"frame": contact_frame, "image_observation_override": [501.0, 402.0]}

    _, _, sigma, _ = anchor_first_fit.contact_observation(contact, track)

    assert sigma == anchor_first_fit.CONTACT_OBSERVATION_DEFAULT_SIGMA_PX


def test_the_spin_prior_is_centred_where_the_corpus_measured_it() -> None:
    velocity = np.array([0.0, -25.0, -4.0])
    # ``_spin_components`` reads topspin about ``z x travel``, which is the generator's own axis.
    axis = np.array([1.0, 0.0, 0.0])
    hard_rpm = anchor_first_fit.SPIN_PRIOR_TOPSPIN_RPM["hard"]
    measured = hard_rpm * 2.0 * math.pi / 60.0 * axis

    at_prior = anchor_first_fit.spin_prior_residuals(velocity, measured, "hard")
    at_zero = anchor_first_fit.spin_prior_residuals(velocity, np.zeros(3), "hard")

    assert np.allclose(at_prior, 0.0, atol=1e-6)
    # A ball with no spin is now the unusual one, by the corpus's own centre and this sigma.
    assert float(at_zero[0]) == pytest.approx(
        -hard_rpm / anchor_first_fit.SPIN_PRIOR_TOPSPIN_SIGMA_RPM, rel=1e-6
    )
    # Clay was measured spinning harder than hard court, so the same ball is charged differently.
    assert anchor_first_fit.SPIN_PRIOR_TOPSPIN_RPM["clay"] > hard_rpm
    assert float(anchor_first_fit.spin_prior_residuals(velocity, measured, "clay")[0]) < 0.0


def test_sidespin_and_rifle_stay_centred_on_nothing() -> None:
    velocity = np.array([0.0, -25.0, 0.0])
    rifle = np.array([0.0, -100.0, 0.0])  # spin about the direction of travel

    topspin_only = anchor_first_fit.spin_prior_residuals(velocity, np.zeros(3), "hard")
    residuals = anchor_first_fit.spin_prior_residuals(velocity, rifle, "hard")

    # Only topspin has a centre, so a purely rifling ball is charged for its rifle and nothing
    # else changes.
    assert float(residuals[0]) == pytest.approx(float(topspin_only[0]))
    assert float(residuals[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(residuals[2]) > 0.0


def _bounce_flight_fit(contact_row: dict | None = None, **arms):
    """One bounce-anchored flight, fitted with whichever objective arms are asked for."""
    projection = _projection()
    frames = np.arange(0, 31)
    start = np.array([5.0, 20.0, 1.1])
    velocity = np.array([-0.4, -24.0, 2.4])
    spin = 1900.0 * 2.0 * math.pi / 60.0 * np.array([1.0, 0.0, 0.0])
    positions = [start]
    x, v, w = start.copy(), velocity.copy(), spin.copy()
    bounce_frame = None
    bounce_xyz = None
    for frame in frames[1:]:
        for _ in range(10):
            x, v, w = rk4_step(x, v, w, 1.0 / 250.0)
        if bounce_frame is None and x[2] <= BALL_RADIUS_M and v[2] < 0.0:
            bounce_frame = float(frame)
            bounce_xyz = x.copy()
            rebound = court_bounce(v, w, "hard")
            x = np.array([x[0], x[1], BALL_RADIUS_M])
            v, w = rebound.velocity, rebound.spin
        positions.append(x.copy())
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    contacts = [
        {
            "frame": 0.4,
            "side": "far",
            "phase": "rally",
            "span": 0,
            "row": {},
            **(contact_row or {}),
        },
        {"frame": 30.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    anchors = [{"type": "bounce", "frame": bounce_frame, "xyz": bounce_xyz.tolist()}]
    with _objective_arm(**arms):
        return fit_anchor_first_shot(
            0,
            contacts,
            track,
            {"far": {0: np.array([5.0, 20.0])}, "near": {30: np.array([5.0, 3.0])}},
            camera,
            25.0,
            "hard",
            40,
            anchors,
        )


def test_the_shipped_objective_still_fits_the_chord_at_full_weight() -> None:
    fit = _bounce_flight_fit()

    assert fit is not None
    record = getattr(fit, "_contact_observation")
    assert record["witness"] is False
    assert record["source"] == "track_chord"
    assert record["sigma_px"] == anchor_first_fit.CONTACT_OBSERVATION_MIN_SIGMA_PX
    assert getattr(fit, "_spin_prior") == "zero_spin"


def test_the_contact_arm_moves_the_contact_row_onto_the_witness() -> None:
    row = {
        "image_observation_override": [640.0, 300.0],
        "image_observation_frame": 0.4,
        "image_observation_sigma_px": 6.5,
        "image_observation_source": "contact_frame_refiner",
    }
    shipped = _bounce_flight_fit(row)
    armed = _bounce_flight_fit(row, contact_observation=True)

    assert shipped is not None and armed is not None
    # The shipped objective ignores the contact row's own observation and re-reads the track.
    assert getattr(shipped, "_contact_observation")["source"] == "track_chord"
    record = getattr(armed, "_contact_observation")
    assert record["witness"] is True
    assert record["source"] == "contact_frame_refiner"
    assert record["sigma_px"] == 6.5


def test_the_spin_arm_keeps_the_fit_and_records_which_prior_it_used() -> None:
    """A well-observed bounce flight identifies its own spin, so the prior must not move it."""
    shipped = _bounce_flight_fit()
    armed = _bounce_flight_fit(spin_prior=True)

    assert shipped is not None and armed is not None
    assert getattr(shipped, "_spin_prior") == "zero_spin"
    assert getattr(armed, "_spin_prior") == "measured_surface_topspin"
    shipped_rpm = np.linalg.norm(shipped.bounces[0]["w_in"]) * 60.0 / (2.0 * math.pi)
    armed_rpm = np.linalg.norm(armed.bounces[0]["w_in"]) * 60.0 / (2.0 * math.pi)
    assert abs(armed_rpm - shipped_rpm) < 100.0
    assert held_out_summary(armed)["held_out_reprojection_median_px"] == pytest.approx(
        held_out_summary(shipped)["held_out_reprojection_median_px"], abs=0.2
    )


def test_a_witness_before_the_contact_frame_does_not_break_a_bounce_free_fit() -> None:
    """A bounce-free flight has no state before its own start, so the row must not ask for one."""
    projection = _projection()
    frames = np.arange(0, 26)
    positions, _, _ = sample_states(
        np.array([5.0, 18.0, 1.2]),
        np.array([0.0, -15.0, 5.0]),
        np.zeros(3),
        frames / 25.0,
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    contacts = [
        {
            "frame": 0.6,
            "side": "far",
            "phase": "rally",
            "span": 0,
            "row": {},
            # The contact-frame refiner puts its pixel BEFORE the contact on 206 of the bench's
            # 579 contacts.
            "image_observation_override": list(track[0]),
            "image_observation_frame": 0.0,
            "image_observation_sigma_px": 3.0,
            "image_observation_source": "contact_frame_refiner",
        },
        {"frame": 25.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    with _objective_arm(contact_observation=True):
        fit = fit_anchor_first_shot(
            0,
            contacts,
            track,
            {"far": {0: np.array([5.0, 18.0])}, "near": {25: np.array([5.0, 3.0])}},
            camera,
            25.0,
            "hard",
            40,
            [],
        )

    assert fit is not None
    assert getattr(fit, "_contact_observation")["frame"] == 0.6
    assert held_out_summary(fit)["held_out_reprojection_median_px"] < 1.0


def test_the_weight_only_arm_keeps_the_chord_and_only_changes_what_it_is_worth() -> None:
    """The gate tests the fitted contact against the refiner's pixel.  If the fitter fits that
    same pixel the gate stops being independent evidence, so this arm leaves the pixel alone."""
    track, contact_frame, _ = _corner_track()
    contact = {
        "frame": contact_frame,
        "image_observation_override": [501.0, 402.0],
        "image_observation_sigma_px": 6.5,
        "image_observation_source": "contact_frame_refiner",
    }

    assert anchor_first_fit.contact_row_sigma_px_value(contact, track) == (
        anchor_first_fit.CONTACT_OBSERVATION_CHORD_SIGMA_PX
    )
    # A contact that lands on an observed frame is an observation, not a chord.
    assert anchor_first_fit.contact_row_sigma_px_value({"frame": 12.0}, track) == (
        anchor_first_fit.CONTACT_OBSERVATION_MIN_SIGMA_PX
    )


def test_the_weight_only_arm_ships_off_and_does_not_move_the_contact_pixel() -> None:
    row = {
        "image_observation_override": [640.0, 300.0],
        "image_observation_frame": 0.4,
        "image_observation_sigma_px": 6.5,
        "image_observation_source": "contact_frame_refiner",
    }
    assert anchor_first_fit._CONTACT_OBSERVATION_SIGMA is False

    anchor_first_fit.configure_contact_observation_sigma(True)
    try:
        armed = _bounce_flight_fit(row)
    finally:
        anchor_first_fit.configure_contact_observation_sigma(False)

    assert armed is not None
    record = getattr(armed, "_contact_observation")
    assert record["source"] == "track_chord"
    assert record["sigma_px"] == anchor_first_fit.CONTACT_OBSERVATION_CHORD_SIGMA_PX
    assert getattr(armed, "_contact_observation_sigma_arm") is True


@contextmanager
def _subframe_arm():
    """Turn the sub-frame impact-anchor arm on for one test, then put the default back."""
    anchor_first_fit.configure_subframe_anchors(True)
    try:
        yield
    finally:
        anchor_first_fit.configure_subframe_anchors(False)


def _sub_frame_bounce_case() -> tuple[dict, dict, float, np.ndarray, list[dict], float]:
    """A flight whose true bounce falls between two frames, emitted the way the chain emits.

    The generator's bounce is at a fractional frame; the emission names the nearest whole frame
    and carries the track's own pixel on it, exactly as ``cv/validation/s6_point_bench.py``
    does and as the real event model does.  On that frame the ball is above the court, so the
    ray/plane intersection of the emitted pixel is not the bounce.
    """
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 25.0
    bounce_frame = 12.37
    bounce_xyz = np.array([5.4, 14.0, BALL_RADIUS_M])
    incoming_velocity = np.array([0.4, -13.0, -5.2])
    incoming_spin = np.array([-160.0, 0.0, 0.0])
    frames = np.arange(0, 27)
    positions, _, _, _, _ = simulate_measured_bounce_knot(
        incoming_velocity,
        incoming_spin,
        0.0,
        frames.astype(float),
        fps,
        "hard",
        {"frame": bounce_frame, "x": bounce_xyz},
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}
    emitted_frame = float(round(bounce_frame))
    anchors = [
        {
            "type": "bounce",
            "frame": emitted_frame,
            "observation_frame": emitted_frame,
            "image_xy": track[int(emitted_frame)].tolist(),
            "xyz": (
                anchor_first_fit.ray_at_height(projection, track[int(emitted_frame)], BALL_RADIUS_M)
            ).tolist(),
            "xyz_is_observation": False,
            "plane_z_m": BALL_RADIUS_M,
            "pixel_sigma": 2.0,
            "time_sigma_frames": 0.42,
            "time_bounds_frames": [emitted_frame - 1.0, emitted_frame + 1.0],
            "sigma_m": 0.02,
        }
    ]
    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {"frame": 26.0, "side": "near", "phase": "rally", "span": 0, "row": {}},
    ]
    return (
        {
            "index": 0,
            "contacts": contacts,
            "ball": track,
            "players": {"far": {0: np.array([5.0, 22.0])}, "near": {26: np.array([5.0, 2.0])}},
            "camera": camera,
            "fps": fps,
            "surface": "hard",
            "max_nfev": 200,
            "anchors": anchors,
        },
        camera,
        bounce_frame,
        bounce_xyz,
        anchors,
        emitted_frame,
    )


def test_the_subframe_anchor_arm_ships_off() -> None:
    assert anchor_first_fit._SUBFRAME_ANCHORS is False


def test_the_emitted_bounce_pixel_is_not_the_bounce() -> None:
    """The premise of the arm, measured rather than assumed: the ray/plane intersection of an
    integer-frame pixel sits away from the impact because the ball is above the court there."""
    _, _, _, bounce_xyz, anchors, _ = _sub_frame_bounce_case()
    seed = np.asarray(anchors[0]["xyz"], float)
    assert np.linalg.norm(seed[:2] - bounce_xyz[:2]) > 0.05


def test_the_subframe_arm_recovers_a_bounce_the_exact_knot_gets_wrong() -> None:
    kwargs, _, bounce_frame, bounce_xyz, _, emitted_frame = _sub_frame_bounce_case()

    shipped = fit_anchor_first_shot(**kwargs)
    with _subframe_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert shipped is not None and armed is not None
    shipped_error = float(np.linalg.norm(np.asarray(shipped.bounces[0]["x"], float) - bounce_xyz))
    armed_error = float(np.linalg.norm(np.asarray(armed.bounces[0]["x"], float) - bounce_xyz))
    assert shipped_error > 0.05
    assert armed_error < shipped_error / 2.0
    # The fitted impact is on the court plane, and its time is inside the emission's frame.
    assert armed.bounces[0]["x"][2] == pytest.approx(BALL_RADIUS_M)
    assert abs(float(armed.bounces[0]["frame"]) - emitted_frame) <= 1.0
    assert abs(float(armed.bounces[0]["frame"]) - bounce_frame) < 0.5


def test_the_subframe_arm_records_what_it_moved() -> None:
    kwargs, _, _, _, _, emitted_frame = _sub_frame_bounce_case()
    with _subframe_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert armed is not None
    assert getattr(armed, "_subframe_anchors") is True
    row = next(entry for entry in armed._anchor_errors if entry["type"] == "bounce")
    assert row["role"] == "court_plane_observation"
    assert row["emitted_frame"] == emitted_frame
    assert abs(row["time_offset_frames"]) <= 1.0
    # The anchor error is now what the fit does with the emission, not a constant zero.
    assert row["emission_pixel_error_px"] < 3.0
    assert row["seed_distance_m"] > 0.0


def test_the_subframe_arm_leaves_the_bounce_on_the_court_plane() -> None:
    """The one exact thing about a bounce stays exact: the impact is on the court."""
    kwargs, _, _, _, _, _ = _sub_frame_bounce_case()
    with _subframe_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert armed is not None
    assert float(armed.bounces[0]["x"][2]) == pytest.approx(BALL_RADIUS_M, abs=1e-9)


def test_the_default_arm_still_pins_the_flight_through_the_emitted_knot() -> None:
    kwargs, _, _, _, anchors, _ = _sub_frame_bounce_case()
    shipped = fit_anchor_first_shot(**kwargs)

    assert shipped is not None
    assert np.allclose(
        np.asarray(shipped.bounces[0]["x"], float), np.asarray(anchors[0]["xyz"], float)
    )
    row = next(entry for entry in shipped._anchor_errors if entry["type"] == "bounce")
    assert row["error_m"] == 0.0


def _terminal_subframe_case() -> tuple[dict, float, np.ndarray, float]:
    """A rally-closing flight whose second bounce falls between two frames.

    The flight ends at its own termination, so nothing is observed after that impact and the
    only rows that move with its sub-frame time are the court-plane row and the time prior.
    """
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    fps = 25.0
    first_frame = 20.37
    first_xyz = np.array([5.2, 15.0, BALL_RADIUS_M])
    incoming_velocity = np.array([0.5, -11.0, -4.5])
    incoming_spin = np.zeros(3)
    rebound = court_bounce(incoming_velocity, incoming_spin, "hard")
    elapsed = np.linspace(0.05, 2.0, 2000)
    post, _, _ = sample_states(first_xyz, rebound.velocity, rebound.spin, elapsed)
    descending = np.flatnonzero((elapsed > 0.15) & (post[:, 2] <= BALL_RADIUS_M))
    terminal_frame = first_frame + (float(elapsed[descending[0]]) + DWELL_SECONDS) * fps
    terminal_xyz = post[descending[0]].copy()
    terminal_xyz[2] = BALL_RADIUS_M
    frames = np.arange(0, math.floor(terminal_frame) + 1)
    positions, _, _, _, _ = simulate_measured_bounce_knot(
        incoming_velocity,
        incoming_spin,
        0.0,
        frames.astype(float),
        fps,
        "hard",
        {"frame": first_frame, "x": first_xyz},
    )
    track = {int(frame): _project(projection, xyz) for frame, xyz in zip(frames, positions)}

    def anchor(emitted: float) -> dict:
        pixel = track[int(emitted)]
        return {
            "type": "bounce",
            "frame": float(emitted),
            "observation_frame": float(emitted),
            "image_xy": pixel.tolist(),
            "xyz": anchor_first_fit.ray_at_height(projection, pixel, BALL_RADIUS_M).tolist(),
            "plane_z_m": BALL_RADIUS_M,
            "pixel_sigma": 2.0,
            "time_sigma_frames": 0.42,
            "time_bounds_frames": [emitted - 1.0, emitted + 1.0],
            "sigma_m": 0.02,
        }

    contacts = [
        {"frame": 0.0, "side": "far", "phase": "rally", "span": 0, "row": {}},
        {
            "frame": float(math.floor(terminal_frame)),
            "side": "near",
            "phase": "terminal",
            "span": 0,
            "terminal": True,
            "row": {},
        },
    ]
    return (
        {
            "index": 0,
            "contacts": contacts,
            "ball": track,
            "players": {"far": {0: np.array([5.0, 22.0])}},
            "camera": camera,
            "fps": fps,
            "surface": "hard",
            "max_nfev": 200,
            "anchors": [anchor(round(first_frame)), anchor(math.floor(terminal_frame))],
        },
        terminal_frame,
        terminal_xyz,
        float(math.floor(terminal_frame)),
    )


def test_the_terminal_impact_time_is_seeded_on_both_sides_of_its_frame() -> None:
    """From a zero seed the solver never leaves the emitted frame, because the terminal impact
    has no observations after it; seeding both half-frame edges is what finds it."""
    kwargs, terminal_frame, terminal_xyz, emitted = _terminal_subframe_case()

    shipped = fit_anchor_first_shot(**kwargs)
    with _subframe_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert shipped is not None and armed is not None
    shipped_error = float(
        np.linalg.norm(np.asarray(shipped.bounces[-1]["x"], float) - terminal_xyz)
    )
    armed_error = float(np.linalg.norm(np.asarray(armed.bounces[-1]["x"], float) - terminal_xyz))
    assert shipped_error > 0.25
    assert armed_error < 0.15
    # It moved toward the true instant rather than staying on the frame it was emitted on.
    offset = float(armed.bounces[-1]["frame"]) - emitted
    assert offset > 0.5
    assert abs(float(armed.bounces[-1]["frame"]) - terminal_frame) < 0.3


def test_the_terminal_time_seeds_bracket_the_emitted_frame() -> None:
    assert anchor_first_fit.SUBFRAME_TERMINAL_TIME_SEEDS_FRAMES == (0.0, -0.5, 0.5)


def test_the_plane_anchor_error_arm_ships_off() -> None:
    assert anchor_first_fit._SUBFRAME_PLANE_ANCHOR_ERROR is False


def test_the_plane_anchor_error_arm_reports_the_assertion_the_anchor_makes() -> None:
    """``anchor_max_error_m`` is a gate input.  The default reports the emission ray miss; this
    arm reports the court-plane assertion, which the fit satisfies exactly, so the gate reads the
    same kind of number for the sub-frame arm as it does for the shipped knot."""
    kwargs, _, _, _, _, _ = _sub_frame_bounce_case()

    with _subframe_arm():
        default = fit_anchor_first_shot(**kwargs)
        anchor_first_fit.configure_subframe_plane_anchor_error(True)
        try:
            armed = fit_anchor_first_shot(**kwargs)
        finally:
            anchor_first_fit.configure_subframe_plane_anchor_error(False)

    assert default is not None and armed is not None
    default_row = next(row for row in default._anchor_errors if row["type"] == "bounce")
    armed_row = next(row for row in armed._anchor_errors if row["type"] == "bounce")
    assert default_row["error_m"] > 0.0
    assert armed_row["error_m"] == 0.0
    # The ray miss is still reported, it is just no longer what the gate reads.
    assert armed_row["emission_ray_miss_m"] == pytest.approx(default_row["emission_ray_miss_m"])
    assert getattr(armed, "_subframe_plane_anchor_error") is True


@contextmanager
def _contact_parts(**overrides):
    """Run one test with some halves of the sub-frame contact arm off, then put them back."""
    anchor_first_fit.configure_subframe_contact_parts(**overrides)
    try:
        yield
    finally:
        anchor_first_fit.configure_subframe_contact_parts()


def test_every_half_of_the_subframe_contact_arm_ships_on_inside_the_flag() -> None:
    """``--subframe-contacts`` alone is exactly what ``docs/wk1/point_fit7.md`` measured."""
    assert anchor_first_fit._SUBFRAME_CONTACT_SEAM is True
    assert anchor_first_fit._SUBFRAME_CONTACT_ADVANCE is True
    assert anchor_first_fit._SUBFRAME_CONTACT_SEAM_SKIPS_REFIT is True
    assert anchor_first_fit._SUBFRAME_CONTACTS is False


def test_the_advance_half_can_be_turned_off_on_its_own() -> None:
    """With the advance off the shared contact stays on the emitted pixel's ray."""
    fits = [SimpleNamespace(), SimpleNamespace()]
    contexts = [{"fps": 25.0, "surface": "hard"}, {"fps": 25.0, "surface": "hard"}]
    ray_point = np.array([1.0, 2.0, 1.0])
    velocity = np.array([0.0, 25.0, 0.0])
    fits[0].state = lambda *_args, **_kwargs: (ray_point, velocity)
    fits[1].state = lambda *_args, **_kwargs: (ray_point, velocity)
    advanced = anchor_first_fit._advance_emission_point(ray_point, fits, contexts, 10.0, 10.5)
    assert np.linalg.norm(advanced - ray_point) == pytest.approx(0.5)


def test_a_seam_record_alone_does_not_stand_the_soft_reconciliation_down() -> None:
    """The two halves the flag bundles: recording the seam time, and skipping the refit there.

    With ``seam_skips_refit`` off, a seam-only record no longer blocks the later passes, so the
    contact is still reconciled; every other kind of record still blocks them.
    """
    seam_only = SimpleNamespace(
        _shared_contacts=[{"contact_index": 3, "source": anchor_first_fit.SEAM_ADOPTION_SOURCE}]
    )
    refitted = SimpleNamespace(
        _shared_contacts=[{"contact_index": 3, "source": "one_sided_contact"}]
    )
    refined = {2: seam_only, 3: seam_only, 4: refitted}
    assert anchor_first_fit._contact_already_reconciled(refined, [2, 3], 3) is True
    with _contact_parts(seam_skips_refit=False):
        assert anchor_first_fit._contact_already_reconciled(refined, [2, 3], 3) is False
        # A record that really did refit still blocks the pass.
        assert anchor_first_fit._contact_already_reconciled(refined, [3, 4], 3) is True
    assert anchor_first_fit._contact_already_reconciled(refined, [2, 3], 3) is True


# --------------------------------------------------------------------------
# pass 9: the label-free evidence, the bounce circle, and the joint solve
# --------------------------------------------------------------------------


@contextmanager
def _bounce_witness_arm():
    anchor_first_fit.configure_subframe_anchors(True)
    anchor_first_fit.configure_subframe_bounce_witness(True)
    try:
        yield
    finally:
        anchor_first_fit.configure_subframe_bounce_witness(False)
        anchor_first_fit.configure_subframe_anchors(False)


@contextmanager
def _joint_arm():
    anchor_first_fit.configure_whole_point_joint(True)
    try:
        yield
    finally:
        anchor_first_fit.configure_whole_point_joint(False)


def test_the_pass_nine_arms_ship_off() -> None:
    assert anchor_first_fit._SUBFRAME_BOUNCE_WITNESS is False
    assert anchor_first_fit._WHOLE_POINT_JOINT is False


def test_joint_bounce_parameter_seed_reaches_the_constrained_optimizer():
    kwargs, *_ = _sub_frame_bounce_case()
    kwargs["max_nfev"] = 5
    captured = []
    with _joint_arm(), _subframe_arm():
        initial = fit_anchor_first_shot(**kwargs)
        assert initial is not None
        parameters = initial._joint_payload["parameters"].copy()
        assert initial._joint_payload["parameter_basis"] == "bounce_knot"
        anchor_first_fit.configure_objective_probe(captured.append)
        try:
            refit = fit_anchor_first_shot(
                **kwargs,
                initial_fit=initial,
                parameter_seed=parameters,
                fixed_end_anchor=initial.state(26.0, 25.0, "hard")[0],
            )
        finally:
            anchor_first_fit.configure_objective_probe(None)
    assert refit is not None
    assert refit._joint_parameter_seed_used is True
    assert np.array_equal(captured[0]["seeds"][0], parameters)


def test_joint_bounce_parameter_seed_rejects_a_different_basis_shape():
    kwargs, *_ = _sub_frame_bounce_case()
    with _joint_arm(), _subframe_arm(), pytest.raises(ValueError, match="bounce-knot basis"):
        fit_anchor_first_shot(**kwargs, parameter_seed=np.zeros(2))


def test_the_bounce_circle_is_twenty_centimetres_or_two_sigma_whichever_is_larger() -> None:
    """The owner's shape, exactly: ``max(0.20 m, 2 sigma)``, and no circle without a witness."""
    tight = anchor_first_fit._court_witness_geometry({"xy": [3.0, 9.0], "sigma_m": 0.04})
    wide = anchor_first_fit._court_witness_geometry({"xy": [3.0, 9.0], "sigma_m": 0.35})
    absent = anchor_first_fit._court_witness_geometry(None)

    assert tight["court_witness_radius_m"] == pytest.approx(0.20)
    assert wide["court_witness_radius_m"] == pytest.approx(0.70)
    assert absent["court_witness_xy"] is None
    assert absent["court_witness_radius_m"] is None


def test_the_bounce_witness_is_free_inside_the_circle_and_grows_outside_it() -> None:
    """Zero everywhere inside the rim, zero at the rim, then a growing penalty; never a pin."""
    geometry = anchor_first_fit._court_witness_geometry({"xy": [3.0, 9.0], "sigma_m": 0.10})
    radius = geometry["court_witness_radius_m"]
    assert radius == pytest.approx(0.20)

    at_witness = anchor_first_fit.bounce_witness_residual(np.array([3.0, 9.0]), geometry)
    inside = anchor_first_fit.bounce_witness_residual(np.array([3.0, 9.0 + 0.5 * radius]), geometry)
    at_rim = anchor_first_fit.bounce_witness_residual(np.array([3.0, 9.0 + radius]), geometry)
    one_out = anchor_first_fit.bounce_witness_residual(
        np.array([3.0, 9.0 + 2.0 * radius]), geometry
    )
    two_out = anchor_first_fit.bounce_witness_residual(
        np.array([3.0, 9.0 + 3.0 * radius]), geometry
    )

    assert at_witness == 0.0
    assert inside == 0.0
    assert at_rim == pytest.approx(0.0)
    assert one_out == pytest.approx(1.0)
    assert two_out == pytest.approx(2.0)
    # A witness the anchor never carried leaves the objective exactly as it was.
    assert (
        anchor_first_fit.bounce_witness_residual(
            np.array([50.0, 50.0]), anchor_first_fit._court_witness_geometry(None)
        )
        == 0.0
    )


def test_the_bounce_witness_arm_is_inert_when_the_fit_is_already_inside_the_circle() -> None:
    """The circle costs nothing where the fit already agrees with the track's own corner."""
    kwargs, _, _, bounce_xyz, anchors, _ = _sub_frame_bounce_case()
    anchors[0]["court_witness"] = {"xy": bounce_xyz[:2].tolist(), "sigma_m": 0.02}

    with _subframe_arm():
        plain = fit_anchor_first_shot(**kwargs)
    with _bounce_witness_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert plain is not None and armed is not None
    plain_bounce = np.asarray(plain.bounces[0]["x"], float)
    armed_bounce = np.asarray(armed.bounces[0]["x"], float)
    assert np.linalg.norm(plain_bounce[:2] - bounce_xyz[:2]) < 0.20
    np.testing.assert_allclose(armed_bounce, plain_bounce, atol=1e-6)


def test_the_bounce_witness_arm_pulls_a_bounce_towards_a_displaced_witness() -> None:
    """Outside the circle the penalty is real, and it is a penalty and not a pin."""
    kwargs, _, _, bounce_xyz, anchors, _ = _sub_frame_bounce_case()
    displaced = np.asarray([bounce_xyz[0] + 2.0, bounce_xyz[1]], float)
    anchors[0]["court_witness"] = {"xy": displaced.tolist(), "sigma_m": 0.02}

    with _subframe_arm():
        plain = fit_anchor_first_shot(**kwargs)
    with _bounce_witness_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert plain is not None and armed is not None
    plain_gap = float(np.linalg.norm(np.asarray(plain.bounces[0]["x"], float)[:2] - displaced))
    armed_gap = float(np.linalg.norm(np.asarray(armed.bounces[0]["x"], float)[:2] - displaced))
    assert armed_gap < plain_gap
    # It never pins: a wrong witness two metres away does not drag the impact onto itself.
    assert armed_gap > anchor_first_fit.BOUNCE_WITNESS_FLOOR_M
    assert float(armed.bounces[0]["x"][2]) == pytest.approx(BALL_RADIUS_M)


def test_every_fit_carries_its_own_label_free_consistency_evidence() -> None:
    """The gate lane's missing columns, emitted per flight and never read by the fitter."""
    kwargs, _, _, bounce_xyz, anchors, _ = _sub_frame_bounce_case()
    anchors[0]["court_witness"] = {"xy": bounce_xyz[:2].tolist(), "sigma_m": 0.02}
    with _subframe_arm():
        armed = fit_anchor_first_shot(**kwargs)

    assert armed is not None
    evidence = armed._consistency_evidence
    assert evidence["schema"] == "flight_consistency_evidence_v1"
    # multistart spread, in metres and along the direction the camera cannot see
    assert evidence["multistart_solutions"] >= 1
    assert evidence["multistart_start_spread_m"] >= 0.0
    assert evidence["multistart_depth_spread_m"] >= 0.0
    # the objective's weakest direction and what one unit of cost buys along it
    assert evidence["hessian_min_curvature"] > 0.0
    assert evidence["hessian_max_curvature"] >= evidence["hessian_min_curvature"]
    assert 0.0 < evidence["hessian_curvature_ratio"] <= 1.0
    assert evidence["hessian_weak_start_move_m"] >= 0.0
    assert 0.0 <= evidence["hessian_weak_depth_fraction"] <= 1.0 + 1e-9
    # the court-plane corner the track draws, and the time prior the fit disagreed with
    assert evidence["bounce_court_witness_error_m"] < 0.20
    assert evidence["bounce_court_witness_inside_circle"] is True
    assert abs(evidence["bounce_time_prior_residual_sigmas"]) < 3.0
    # the striker the tracker put at each contact
    assert evidence["start_striker_distance_m"] is not None
    assert evidence["end_striker_distance_m"] is not None


def test_the_weak_direction_is_the_step_that_costs_one_unit_of_the_objective() -> None:
    """The reported number is checked against the fitter's own objective, not asserted.

    ``hessian_weak_start_move_m`` claims to be how far the flight's own start travels for one
    unit of cost along the objective's flattest direction.  Rebuild that step from the reported
    curvature, evaluate the fitter's own residual there, and the cost really does rise by about
    one.
    """
    kwargs, _, _, _, _, _ = _sub_frame_bounce_case()
    captured: list[dict] = []
    anchor_first_fit.configure_objective_probe(captured.append)
    try:
        with _subframe_arm():
            armed = fit_anchor_first_shot(**kwargs)
    finally:
        anchor_first_fit.configure_objective_probe(None)

    assert armed is not None and captured
    payload = captured[-1]
    evidence = armed._consistency_evidence
    curvature = evidence["hessian_min_curvature"]
    assert curvature > 0.0

    def cost(parameters: np.ndarray) -> float:
        residuals = np.asarray(payload["residual"](parameters), float)
        scaled = np.square(residuals / 2.0)
        return float(0.5 * np.sum(2.0 * 2.0**2 * (np.sqrt(1.0 + scaled) - 1.0)))

    here = np.asarray(payload["parameters"], float)
    jacobian = np.asarray(payload["solve"](here).jac, float)
    values, vectors = np.linalg.eigh(jacobian.T @ jacobian)
    step = math.sqrt(2.0 / float(values[0]))
    moved = np.clip(here + step * vectors[:, 0], payload["lower"], payload["upper"])
    start_here = payload["simulate"](here, np.asarray([payload["f0"]]))[0][0]
    start_there = payload["simulate"](moved, np.asarray([payload["f0"]]))[0][0]

    # The probe's re-solve from the recorded solution lands on the same minimum, so its
    # Jacobian is the recorded one to within the optimiser's own convergence.
    assert float(values[0]) == pytest.approx(curvature, rel=0.05)
    assert float(np.linalg.norm(start_there - start_here)) == pytest.approx(
        evidence["hessian_weak_start_move_m"], rel=0.05
    )
    assert cost(moved) - cost(here) == pytest.approx(1.0, abs=0.6)


class _JointFit:
    """A two-parameter straight-line flight with the payload the joint solve reads."""

    def __init__(self, index: int, context: dict, start: np.ndarray, velocity: np.ndarray):
        self.index = index
        self._start = np.asarray(start, float)
        self._velocity = np.asarray(velocity, float)
        self.theta = np.r_[self._start, self._velocity, np.zeros(3)]
        self.rms_px = 2.0
        self.bounces = []
        self._held_out_errors_px = np.asarray([2.0, 3.0, 4.0])
        self._anchor_errors = [{"error_m": 0.02}]
        self._anchor_first_context = context
        self._joint_payload = {
            "residual": lambda parameters: np.asarray(parameters, float) * 0.0,
            "simulate": self._simulate,
            "lower": np.full(6, -100.0),
            "upper": np.full(6, 100.0),
            "parameters": np.r_[self._start, self._velocity],
            "cost": 0.0,
            "f0": float(context["contacts"][index]["frame"]),
            "f1": float(context["contacts"][index + 1]["frame"]),
            "fps": 25.0,
            "surface": "hard",
        }

    def _simulate(self, parameters: np.ndarray, frames: np.ndarray):
        parameters = np.asarray(parameters, float)
        base = float(self._joint_payload["f0"]) if hasattr(self, "_joint_payload") else 0.0
        elapsed = (np.asarray(frames, float) - base)[:, None]
        positions = parameters[:3][None, :] + parameters[3:6][None, :] * elapsed
        velocities = np.repeat(parameters[3:6][None, :], len(frames), axis=0)
        return (
            positions,
            velocities,
            np.zeros_like(positions),
            [],
            (
                parameters[:3],
                parameters[3:6],
                np.zeros(3),
            ),
        )

    def state(self, frame: float, _fps: float, _surface: str):
        positions, velocities, *_ = self._simulate(
            self._joint_payload["parameters"], np.asarray([float(frame)])
        )
        return positions[0], velocities[0]


def _joint_point(monkeypatch, gap: float):
    """Two straight flights whose ends are ``gap`` metres apart at their shared contact."""
    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection)
    contacts = [
        {"frame": 0.0, "event_frame": 0.0, "side": "far", "phase": "rally"},
        {"frame": 10.0, "event_frame": 10.0, "side": "near", "phase": "rally"},
        {"frame": 20.0, "terminal": True},
    ]
    context = {
        "contacts": contacts,
        "ball": {},
        "players": {},
        "camera": camera,
        "fps": 25.0,
        "surface": "hard",
        "max_nfev": 20,
        "anchors": [],
        "observation_weights": None,
    }
    incoming = _JointFit(
        0, {**context, "index": 0}, np.array([5.0, 18.0, 1.0]), np.array([0.0, -0.8, 0.0])
    )
    outgoing = _JointFit(
        1,
        {**context, "index": 1},
        np.array([5.0, 10.0 - gap, 1.0]),
        np.array([0.0, 0.8, 0.0]),
    )

    def fake_fit(**kwargs):
        index = kwargs["index"]
        source = incoming if index == 0 else outgoing
        start = kwargs.get("fixed_start_anchor")
        end = kwargs.get("fixed_end_anchor")
        parameters = np.asarray(source._joint_payload["parameters"], float).copy()
        if start is not None:
            parameters[:3] = np.asarray(start, float)
        if end is not None:
            base = float(source._joint_payload["f0"])
            span = float(source._joint_payload["f1"]) - base
            parameters[3:6] = (np.asarray(end, float) - parameters[:3]) / max(span, 1e-9)
        return _JointFit(
            index,
            kwargs["contacts"] and source._anchor_first_context,
            parameters[:3],
            parameters[3:6],
        )

    monkeypatch.setattr(anchor_first_fit, "fit_anchor_first_shot", fake_fit)
    return incoming, outgoing, contacts


def test_the_joint_solve_closes_the_seam_and_records_that_it_did(monkeypatch) -> None:
    incoming, outgoing, _ = _joint_point(monkeypatch, gap=0.9)
    diagnostics: list[dict] = []

    with _joint_arm():
        refined = refine_anchor_first_point({0: incoming, 1: outgoing}, diagnostics)

    adopted = [row for row in diagnostics if row["status"] == "adopted_whole_point_joint_point"]
    assert len(adopted) == 1
    assert adopted[0]["flights"] == 2
    assert adopted[0]["seams"] == 1
    seam = next(row for row in diagnostics if row["status"] == "adopted_whole_point_joint")
    assert seam["junction_gap_m"] < seam["pre_refit_endpoint_gap_m"]
    assert refined[0] is not incoming and refined[1] is not outgoing
    for index in (0, 1):
        record = next(row for row in refined[index]._shared_contacts if row["contact_index"] == 1)
        assert record["source"] == anchor_first_fit.WHOLE_POINT_JOINT_SOURCE
        assert refined[index]._fit_objective == "anchor_first_whole_point_joint_v1"
        assert refined[index]._whole_point_joint["seams"] == 1


def test_joint_materialization_preserves_the_optimized_bounce_parameters(monkeypatch):
    incoming, outgoing, contacts = _joint_point(monkeypatch, gap=0.9)
    for fit in (incoming, outgoing):
        fit._joint_payload.update(parameter_basis="bounce_knot", timing_nuisance="none")
    original = anchor_first_fit.fit_anchor_first_shot
    captured = []

    def capture(**kwargs):
        captured.append(kwargs["parameter_seed"])
        return original(**kwargs)

    monkeypatch.setattr(anchor_first_fit, "fit_anchor_first_shot", capture)
    with _joint_arm():
        refined = anchor_first_fit.whole_point_joint_solve({0: incoming, 1: outgoing}, contacts, [])
    assert refined is not None
    assert len(captured) == 2
    assert all(seed is not None and np.isfinite(seed).all() for seed in captured)
    assert any(
        not np.allclose(seed, fit._joint_payload["parameters"])
        for seed, fit in zip(captured, (incoming, outgoing), strict=True)
    )


def test_the_joint_solve_stands_aside_for_a_point_with_too_many_flights(monkeypatch) -> None:
    """Past the flight limit the alternating scheme is what runs, and it says so."""
    incoming, outgoing, _ = _joint_point(monkeypatch, gap=0.9)
    diagnostics: list[dict] = []
    with _joint_arm():
        monkeypatch.setattr(anchor_first_fit, "WHOLE_POINT_MAX_FLIGHTS", 1)
        refine_anchor_first_point({0: incoming, 1: outgoing}, diagnostics)

    stood_aside = next(row for row in diagnostics if row["status"] == "joint_not_attempted")
    assert stood_aside["reason"] == "flight_count"
    assert not any(row["status"] == "adopted_whole_point_joint_point" for row in diagnostics)


def test_the_joint_solve_falls_back_when_its_guard_refuses_the_result(monkeypatch) -> None:
    """A refit whose held-out reprojection is worse is refused and the alternating pass runs."""
    incoming, outgoing, _ = _joint_point(monkeypatch, gap=0.9)
    original = anchor_first_fit.fit_anchor_first_shot

    def worse(**kwargs):
        candidate = original(**kwargs)
        candidate._held_out_errors_px = np.asarray([40.0, 50.0, 60.0])
        return candidate

    monkeypatch.setattr(anchor_first_fit, "fit_anchor_first_shot", worse)
    diagnostics: list[dict] = []
    with _joint_arm():
        refined = refine_anchor_first_point({0: incoming, 1: outgoing}, diagnostics)

    assert any(row["status"] == "joint_refit_rejected" for row in diagnostics)
    assert not any(row["status"] == "adopted_whole_point_joint_point" for row in diagnostics)
    assert set(refined) == {0, 1}
