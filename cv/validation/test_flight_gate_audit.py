"""Unit tests for the owner-truth flight gate audit."""

from __future__ import annotations

import numpy as np
import pytest

from cv.validation import flight_gate_audit as audit


class _FixedCamera:
    """A camera that maps world x,z to image u,v with a unit scale."""

    def p_at(self, frame: float) -> np.ndarray:
        del frame
        return np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float
        )


def test_project_and_trajectory_state_interpolate() -> None:
    camera = _FixedCamera()
    assert audit.project(camera.p_at(0.0), [3.0, 99.0, 4.0]) == (3.0, 4.0)
    trajectory = [
        {"frame": 10, "xyz": [0.0, 0.0, 0.0]},
        {"frame": 11, "xyz": [2.0, 0.0, 4.0]},
    ]
    assert audit.trajectory_state(trajectory, 10.5).tolist() == [1.0, 0.0, 2.0]
    # One frame outside the span clamps; further out is refused rather than extrapolated.
    assert audit.trajectory_state(trajectory, 12.0).tolist() == [2.0, 0.0, 4.0]
    assert audit.trajectory_state(trajectory, 13.0) is None


def test_reprojection_errors_use_owner_pixels() -> None:
    fit = {
        "trajectory": [{"frame": 10, "xyz": [0.0, 0.0, 0.0]}, {"frame": 12, "xyz": [0.0, 0.0, 6.0]}]
    }
    rows = audit.reprojection_errors(fit, _FixedCamera(), [(11.0, (4.0, 6.0))])
    assert len(rows) == 1
    assert rows[0]["error_px"] == 5.0


def test_unmatched_truth_events_is_one_to_one() -> None:
    assert audit.unmatched_truth_events([10.0, 20.0], [10.5, 19.0], 3.0) == 0
    assert audit.unmatched_truth_events([10.0, 11.0], [10.5], 3.0) == 1
    assert audit.unmatched_truth_events([10.0], [40.0], 3.0) == 1


def _flight(**overrides: object) -> dict:
    row = {
        "flight_id": "m__pt0001__flight_000",
        "point": "m__pt0001",
        "scope": "other_38",
        "solved": True,
        "terminal_end": False,
        "held_out_median_px": 3.0,
        "held_out_p90_px": 6.0,
        "anchor_max_error_m": 0.02,
        "contact_reprojection_max_px": 5.0,
        "contact_reach_max_m": 1.5,
        "contact_height_min_m": 0.9,
        "contact_height_max_m": 2.0,
        "net_crossing_plausible": True,
        "net_anchor_available": True,
        "minimum_height_m": 0.05,
        "bounce_count": 1,
        "speed_kmh": 120.0,
        "observation_coverage": 0.9,
        "truth_good": True,
        "gate_blind_truth_good": True,
    }
    row.update(overrides)
    return row


def test_gate_accepts_enforces_every_stated_check() -> None:
    gate = {
        "held_out_median_px": 4.0,
        "held_out_p90_px": 8.0,
        "anchor_max_error_m": 0.05,
        "contact_reprojection_px": 12.0,
        "contact_reach_m": 2.1,
        "contact_height": True,
        "net_crossing": True,
    }
    assert audit.gate_accepts(_flight(), gate) is True
    assert audit.gate_accepts(_flight(solved=False), gate) is False
    assert audit.gate_accepts(_flight(held_out_median_px=4.5), gate) is False
    assert audit.gate_accepts(_flight(anchor_max_error_m=0.2), gate) is False
    assert audit.gate_accepts(_flight(contact_height_max_m=3.9), gate) is False
    assert audit.gate_accepts(_flight(net_crossing_plausible=False), gate) is False
    assert audit.gate_accepts(_flight(net_anchor_available=False), gate) is True
    assert audit.gate_accepts(_flight(bounce_count=2), gate) is False


def test_score_gate_counts_wrong_accepts_and_complete_points() -> None:
    rows = [
        _flight(flight_id="a", truth_good=True),
        _flight(flight_id="b", truth_good=False, terminal_end=True),
        _flight(flight_id="c", point="m__pt0002", truth_good=None, terminal_end=True),
    ]
    accepted = {"a": True, "b": True, "c": True}
    score = audit.score_gate(rows, accepted, {"m__pt0001", "m__pt0002"}, "test")
    assert score["accepted_flights"] == 3
    assert score["accepted_witnessed"] == 2
    assert score["wrong_accepts"] == 1
    assert score["wrong_accept_rate"] == 0.5
    # pt0001 has both flights accepted and exactly one terminal; pt0002 has one terminal flight.
    assert score["complete_points"] == 2
    assert audit.score_gate(rows, {"a": True}, {"m__pt0001"}, "t")["complete_points"] == 0


def test_junction_gap_map_pairs_adjacent_fits() -> None:
    point = {
        "fits": [
            {
                "flight_index": 0,
                "start_frame": 10,
                "end_frame": 20,
                "end_xyz": [0.0, 0.0, 0.0],
                "start_xyz": [0.0, 0.0, 0.0],
            },
            {
                "flight_index": 1,
                "start_frame": 20,
                "end_frame": 30,
                "start_xyz": [3.0, 4.0, 0.0],
                "end_xyz": [0.0, 0.0, 0.0],
            },
        ]
    }
    gaps = audit.junction_gap_map(point)
    assert gaps[0]["next"] == 5.0
    assert gaps[1]["prev"] == 5.0


def test_point_cause_walks_the_chain_in_order() -> None:
    def run(point: dict, flights: list[dict], truth: dict, automatic: dict) -> tuple[str, str]:
        row = audit.point_cause(
            point={"point": "m__pt0001", "match_id": "m", **point},
            flight_rows=flights,
            truth_events=truth,
            automatic_events=automatic,
        )
        return row["cause"], row["detail"]

    solved = {"camera_calibration": {"accepted": True}}
    one_contact = {"contact": [{"frame": 10.0}], "bounce": []}
    matched = {"contact": [10.0], "bounce": []}
    flight = {
        "flight_index": 0,
        "solved": True,
        "terminal_end": True,
        "pixel_accepted": True,
        "metric_accepted": True,
        "net_anchor_available": True,
        "owner_bounce_missing": False,
        "ledger_reasons": [],
        "junction_gap_prev_m": None,
        "junction_gap_next_m": None,
    }
    assert (
        run(
            {"camera_calibration": {"accepted": False, "reason": "hard"}}, [], one_contact, matched
        )[0]
        == "camera_abstained"
    )
    assert run(solved, [], one_contact, {"contact": [], "bounce": []})[0] == "event_missing"
    assert run(solved, [], one_contact, matched)[0] == "flight_not_attempted"
    assert (
        run(solved, [{**flight, "solved": False}], one_contact, matched)[0] == "flight_not_solved"
    )
    assert (
        run(solved, [{**flight, "net_anchor_available": False}], one_contact, matched)[0]
        == "complete"
    )
    assert (
        run(solved, [{**flight, "owner_bounce_missing": True}], one_contact, matched)[0]
        == "missing_bounce"
    )
    assert run(
        solved,
        [{**flight, "pixel_accepted": False, "ledger_reasons": ["high"]}],
        one_contact,
        matched,
    ) == ("solved_but_rejected", "1/1:high")
    assert (
        run(solved, [{**flight, "junction_gap_next_m": 2.0}], one_contact, matched)[0]
        == "junction_gap"
    )
    assert run(solved, [flight], one_contact, matched)[0] == "complete"


def test_owner_position_index_is_native_and_nonempty() -> None:
    index = audit.owner_position_index()
    assert index, "owner-positioned frames must load"
    for pixels in index.values():
        for x, y in pixels.values():
            assert 0.0 <= x <= 1920.0 and 0.0 <= y <= 1080.0


# --- the corrected bounce witness ----------------------------------------------------------


class _CourtCamera:
    """A pinhole 10 m above the baseline looking down the court, in court metres."""

    def __init__(self) -> None:
        # World is (x, y, z) in court metres.  The camera sits at (5.5, -8, 10) looking at the
        # far court, with a 1000 px focal length and the principal point at 960, 540.
        centre = np.asarray([5.5, -8.0, 10.0], dtype=float)
        forward = np.asarray([0.0, 1.0, -0.35], dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.asarray([1.0, 0.0, 0.0], dtype=float)
        down = np.cross(forward, right)
        rotation = np.vstack([right, down, forward])
        intrinsics = np.asarray(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float
        )
        self._p = intrinsics @ np.hstack([rotation, (-rotation @ centre).reshape(3, 1)])
        self._h = np.linalg.inv(self._p[:, [0, 1, 3]])

    def p_at(self, frame: float) -> np.ndarray:
        del frame
        return self._p

    def h_at(self, frame: float) -> np.ndarray:
        del frame
        return self._h

    def pixel(self, xyz) -> tuple[float, float]:
        homogeneous = self._p @ np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
        return (float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2]))


def _bounce_scene(bounce_time: float = 40.35):
    """A ball on a parabola that touches z = R_BALL at ``bounce_time``, seen by _CourtCamera."""
    from cv.pipeline.rich_ball_physics import R_BALL

    camera = _CourtCamera()

    def xyz(time: float):
        dt = time - bounce_time
        return np.asarray([5.0 + 0.2 * dt, 12.0 + 0.9 * dt, R_BALL + 0.9 * abs(dt)], dtype=float)

    track = {frame: camera.pixel(xyz(float(frame))) for frame in range(36, 46)}
    return camera, xyz, track, R_BALL


def test_ray_at_height_inverts_the_projection() -> None:
    camera = _CourtCamera()
    target = np.asarray([7.0, 15.0, 0.0325], dtype=float)
    solved = audit.ray_at_height(camera.p_at(0.0), camera.pixel(target), 0.0325)
    assert np.allclose(solved, target[:2], atol=1e-6)
    # At the wrong height the same ray lands somewhere else on the court, which is the whole
    # error the corrected witness removes.
    at_zero = audit.ray_at_height(camera.p_at(0.0), camera.pixel(target), 0.0)
    assert np.linalg.norm(at_zero - target[:2]) > 0.01


def test_metres_per_pixel_is_anisotropic_and_positive() -> None:
    camera = _CourtCamera()
    scale, lateral, depth = audit.metres_per_pixel(
        camera.p_at(0.0), camera.pixel([5.0, 18.0, 0.0325]), 0.0325
    )
    assert 0.0 < lateral <= scale <= depth
    assert depth > lateral  # a pixel of click error costs more in depth than sideways


def test_fit_image_path_is_quadratic_and_reports_its_residual() -> None:
    samples = [(float(f), (2.0 * f + f * f, 3.0)) for f in (10.0, 11.0, 12.0)]
    fitted = audit.fit_image_path(samples, 10.0)
    assert fitted is not None
    path, residual = fitted
    assert residual < 1e-6
    assert np.allclose(path(13.0), [2.0 * 13.0 + 169.0, 3.0], atol=1e-6)
    assert audit.fit_image_path(samples[:1], 10.0) is None


def test_bounce_window_track_prefers_the_owner_and_keeps_the_click() -> None:
    track = {frame: (float(frame), 0.0) for frame in range(36, 46)}
    owner = {41: (999.0, 111.0)}
    rows, owner_used = audit.bounce_window_track(40.0, (500.0, 250.0), track, owner)
    assert owner_used == 1
    assert rows[41] == (999.0, 111.0)
    assert rows[40] == (500.0, 250.0)  # the click wins on its own frame
    assert rows[39] == (39.0, 0.0)  # the automatic track elsewhere


def test_bounce_witness_v2_recovers_a_known_bounce() -> None:
    camera, xyz, track, radius = _bounce_scene()
    truth = xyz(40.35)
    click_frame = 40.0
    click_pixel = camera.pixel(xyz(click_frame))
    rayplane = audit.ray_at_height(camera.p_at(click_frame), click_pixel, 0.0)
    witness = audit.bounce_witness_v2(
        camera=camera,
        click_frame=click_frame,
        click_pixel=click_pixel,
        track=track,
        owner_positions={},
        fps=25.0,
        rayplane_xy=rayplane,
    )
    assert witness["formed"] is True
    # The sub-frame time is recovered, and the position lands on the true bounce.
    assert abs(witness["t_subframe"] - 40.35) < 0.05
    assert np.linalg.norm(np.asarray(witness["court_xy"]) - truth[:2]) < 0.05
    # The shipped construction, on the same click, is much further out.
    assert np.linalg.norm(rayplane - truth[:2]) > 0.20
    # The sigma is reported and decomposes into the three stated terms.
    total = (
        witness["sigma_click_m"] ** 2 + witness["sigma_path_m"] ** 2 + witness["sigma_time_m"] ** 2
    )
    assert abs(witness["sigma_m"] ** 2 - total) < 1e-9
    assert witness["sigma_click_lateral_m"] <= witness["sigma_click_depth_m"]
    assert radius > 0.0


def test_bounce_witness_v2_abstains_without_a_track() -> None:
    camera, xyz, _, _ = _bounce_scene()
    click_pixel = camera.pixel(xyz(40.0))
    witness = audit.bounce_witness_v2(
        camera=camera,
        click_frame=40.0,
        click_pixel=click_pixel,
        track={},
        owner_positions={},
        fps=25.0,
        rayplane_xy=np.asarray([0.0, 0.0]),
    )
    assert witness["formed"] is False
    assert witness["reason"] == "too_few_track_frames"
    assert witness.get("court_xy") is None


def test_bounce_witness_v2_click_offset_moves_the_answer() -> None:
    camera, xyz, track, _ = _bounce_scene()
    click_pixel = camera.pixel(xyz(40.0))
    rayplane = audit.ray_at_height(camera.p_at(40.0), click_pixel, 0.0)
    common = dict(
        camera=camera,
        click_frame=40.0,
        click_pixel=click_pixel,
        track=track,
        owner_positions={},
        fps=25.0,
        rayplane_xy=rayplane,
    )
    at_face_value = audit.bounce_witness_v2(**common, click_offset_frames=0.0)
    leading_blur = audit.bounce_witness_v2(**common, click_offset_frames=-0.5)
    assert at_face_value["formed"] and leading_blur["formed"]
    # Same time, different transport, so a different position: the convention is a real choice.
    assert at_face_value["t_subframe"] == leading_blur["t_subframe"]
    assert leading_blur["transport_px"] > at_face_value["transport_px"]


def test_witness_breakdown_splits_wrong_accepts_per_witness() -> None:
    scoped = [
        _flight(flight_id="a", truth_good=True, truth_good_v2=False),
        _flight(flight_id="b", truth_good=False, truth_good_v2=True),
        _flight(flight_id="c", truth_good=None, truth_good_v2=None),
    ]
    taken = scoped[:2]
    breakdown = audit.witness_breakdown(scoped, taken)
    assert breakdown["truth_good"]["wrong_accepts"] == 1
    assert breakdown["truth_good"]["truth_good_in_scope"] == 1
    assert breakdown["truth_good_v2"]["wrong_accepts"] == 1
    assert breakdown["truth_good_v2"]["witnessed_in_scope"] == 2
    assert breakdown["truth_bounce_v2_pass"]["accepted_witnessed"] == 0


def test_graded_score_is_flat_inside_the_circle_and_decays_outside() -> None:
    # Fully valid anywhere inside the circle, and the decay is smooth, monotone and never
    # negative.  The stated verdict line, score >= 0.5, sits at d = 1.833 r.
    assert audit.graded_score(0.0, 0.20) == 1.0
    assert audit.graded_score(0.20, 0.20) == 1.0
    assert audit.graded_score(0.19, 0.20) == 1.0
    assert 0.5 < audit.graded_score(0.30, 0.20) < 1.0
    assert audit.graded_score(0.20 * 1.8326, 0.20) == pytest.approx(0.5, abs=1e-3)
    assert audit.graded_score(0.60, 0.20) < 0.02
    assert audit.graded_score(0.60, 0.20) > 0.0  # a penalty, not a fail
    assert audit.graded_score(10.0, 0.20) == 0.0  # far enough out it underflows to zero
    values = [audit.graded_score(d, 0.20) for d in (0.2, 0.3, 0.4, 0.5)]
    assert values == sorted(values, reverse=True)
    assert audit.graded_score(0.1, 0.0) == 0.0
    assert audit.graded_score(float("inf"), 0.20) == 0.0


def test_linear_shape_puts_the_same_verdict_at_two_radii() -> None:
    # The owner's other suggestion: a ramp to zero at 3r, which draws the 0.5 line at 2r.
    assert audit.graded_score_linear(0.20, 0.20) == 1.0
    assert audit.graded_score_linear(0.40, 0.20) == pytest.approx(0.5)
    assert audit.graded_score_linear(0.60, 0.20) == pytest.approx(0.0, abs=1e-12)
    assert audit.graded_score_linear(1.00, 0.20) == 0.0


def test_bounce_circle_is_the_wider_of_20cm_and_two_sigma() -> None:
    assert audit.bounce_graded_radius_m(0.02) == 0.20
    assert audit.bounce_graded_radius_m(None) == 0.20
    assert audit.bounce_graded_radius_m(0.25) == 0.50
    # A 0.22 m witness -- the real cohort's median -- draws a 0.44 m circle, not a 0.10 m line.
    assert audit.bounce_graded_radius_m(0.22) == pytest.approx(0.44)


def test_contact_circle_is_the_refiner_radius_floored_at_the_shipped_tolerance() -> None:
    # A straight track with a sharp corner: the refiner answers, and its radius is at least the
    # 12 px the pipeline already holds a contact to.
    track = {frame: (100.0 + 20.0 * (frame - 40), 300.0) for frame in range(30, 41)}
    track.update({frame: (100.0, 300.0 + 20.0 * (frame - 40)) for frame in range(41, 51)})
    radius, abstained = audit.contact_graded_radius_px(track, 40)
    assert radius >= audit.CONTACT_GRADED_FLOOR_PX
    assert isinstance(abstained, bool)
    # With no track at all the refiner supplies nothing, so the circle falls back to 12 px.
    assert audit.contact_graded_radius_px({}, 40) == (audit.CONTACT_GRADED_FLOOR_PX, True)
    # The cache answers for a frame it has already seen without consulting the refiner.
    cache = {40: (77.0, False)}
    assert audit.contact_graded_radius_px(track, 40.0, cache) == (77.0, False)


def test_graded_summary_counts_what_the_circle_changes() -> None:
    rows = [
        # inside the circle: hard-fail at 10 cm, fully valid graded
        _flight(
            flight_id="a",
            truth_bounce_v2_score=1.0,
            truth_bounce_v2_score_linear=1.0,
            truth_bounce_v2_pass=False,
            truth_bounce_v2_graded_pass=True,
            truth_bounce_v2_radius_max_m=0.20,
            truth_bounce_v2_error_max_m=0.15,
        ),
        # far outside it: fails both
        _flight(
            flight_id="b",
            truth_bounce_v2_score=0.05,
            truth_bounce_v2_score_linear=0.0,
            truth_bounce_v2_pass=False,
            truth_bounce_v2_graded_pass=False,
            truth_bounce_v2_radius_max_m=0.20,
            truth_bounce_v2_error_max_m=0.70,
        ),
        # inside 10 cm: passes both
        _flight(
            flight_id="c",
            truth_bounce_v2_score=1.0,
            truth_bounce_v2_score_linear=1.0,
            truth_bounce_v2_pass=True,
            truth_bounce_v2_graded_pass=True,
            truth_bounce_v2_radius_max_m=0.20,
            truth_bounce_v2_error_max_m=0.04,
        ),
    ]
    summary = audit.graded_summary(rows)["bounce"]
    assert summary["flights_scored"] == 3
    assert summary["hard_pass"] == 1
    assert summary["graded_pass"] == 2
    assert summary["graded_only"] == 1
    assert summary["hard_only"] == 0
    assert summary["inside_circle_flights"] == 2
    assert summary["score_histogram"]["1.0"] == 2
    assert summary["graded_only_error_median_p90"][0] == pytest.approx(0.15)


def test_camera_centre_and_transverse_scale_match_the_pinhole() -> None:
    camera = _CourtCamera()
    projection = camera.p_at(0.0)
    centre = audit.camera_centre(projection)
    assert centre == pytest.approx([5.5, -8.0, 10.0])
    # 1000 px focal length, so on the optical axis a point d metres away is d/1000 metres per
    # pixel across the ray, and the scale grows linearly with range.
    forward = np.asarray([0.0, 1.0, -0.35], dtype=float)
    forward /= np.linalg.norm(forward)
    for distance in (12.0, 24.0):
        on_axis = centre + distance * forward
        assert audit.transverse_metres_per_pixel(projection, on_axis) == pytest.approx(
            distance / 1000.0, rel=1e-6
        )
    # Off axis the same rule holds to within the projection's own foreshortening, and range
    # still orders the scale.
    near = np.asarray([5.5, 2.0, 1.0], dtype=float)
    far = np.asarray([5.5, 22.0, 1.0], dtype=float)
    near_scale = audit.transverse_metres_per_pixel(projection, near)
    far_scale = audit.transverse_metres_per_pixel(projection, far)
    assert near_scale == pytest.approx(np.linalg.norm(near - centre) / 1000.0, rel=0.15)
    assert far_scale == pytest.approx(np.linalg.norm(far - centre) / 1000.0, rel=0.15)
    assert far_scale > near_scale


def test_ray_distance_is_zero_on_the_ray_and_grows_off_it() -> None:
    camera = _CourtCamera()
    projection = camera.p_at(0.0)
    on_ray = np.asarray([6.0, 12.0, 1.2], dtype=float)
    pixel = camera.pixel(on_ray)
    assert audit.ray_distance_m(projection, pixel, on_ray) == pytest.approx(0.0, abs=1e-9)
    centre = audit.camera_centre(projection)
    direction = on_ray - centre
    direction /= np.linalg.norm(direction)
    # A point twice as far along the same ray is still on it.
    further = centre + 2.0 * np.linalg.norm(on_ray - centre) * direction
    assert audit.ray_distance_m(projection, pixel, further) == pytest.approx(0.0, abs=1e-9)
    # A point 0.4 m perpendicular to it is 0.4 m off it.
    perpendicular = np.cross(direction, np.asarray([0.0, 0.0, 1.0]))
    perpendicular /= np.linalg.norm(perpendicular)
    assert audit.ray_distance_m(projection, pixel, on_ray + 0.4 * perpendicular) == pytest.approx(
        0.4, abs=1e-6
    )


def _junction_click(camera: _CourtCamera, xyz, frame: float = 100.0) -> dict:
    return {"frame": frame, "image_xy": camera.pixel(xyz)}


def test_junction_witness_passes_two_flights_that_meet_on_the_click_ray() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    record = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=contact + np.asarray([0.0, 0.0, 0.02]),
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    assert record["formed"] is True
    assert record["gap_m"] == pytest.approx(0.02)
    assert record["radius_m"] >= audit.JUNCTION_WITNESS_FLOOR_M
    assert record["hard_pass"] is True
    assert record["graded_pass"] is True
    assert record["score"] == pytest.approx(1.0)


def test_junction_witness_fails_both_flights_on_the_position_gap() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    centre = audit.camera_centre(camera.p_at(100.0))
    direction = contact - centre
    direction /= np.linalg.norm(direction)
    # Both states are on the click ray, so only the depth disagreement can fail this.
    far_along = contact + 6.0 * direction
    record = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=far_along,
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    assert record["formed"] is True
    assert record["ray_max_px"] == pytest.approx(0.0, abs=1e-6)
    assert record["ray_score"] == pytest.approx(1.0)
    assert record["gap_m"] == pytest.approx(6.0)
    assert record["hard_pass"] is False
    assert record["graded_pass"] is False
    assert record["score"] == record["position_score"]


def test_junction_witness_fails_both_flights_off_the_click_ray() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    # The two flights agree with each other to a centimetre and both sit a metre off the ray.
    offset = np.asarray([1.0, 0.0, 0.0], dtype=float)
    record = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact + offset,
        outbound_start_xyz=contact + offset + np.asarray([0.0, 0.0, 0.01]),
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    assert record["formed"] is True
    assert record["position_score"] == pytest.approx(1.0)
    assert record["ray_max_px"] > audit.CONTACT_GRADED_FLOOR_PX
    assert record["hard_pass"] is False
    assert record["graded_pass"] is False


def test_junction_witness_circle_widens_with_the_contact_radius() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 22.0, 1.2], dtype=float)
    narrow = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=contact,
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    wide = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=contact,
        radius_px=120.0,
        refiner_abstained=True,
    )
    # The circle is the wider of the bench's own 0.25 m junction tolerance and 2 sigma, and
    # sigma is the contact circle in pixels carried to metres at the contact's own range.
    for record in (narrow, wide):
        assert record["radius_m"] == pytest.approx(
            max(
                audit.JUNCTION_WITNESS_FLOOR_M,
                audit.JUNCTION_WITNESS_SIGMA_MULTIPLE * record["sigma_m"],
            )
        )
    assert wide["radius_m"] > narrow["radius_m"]
    assert wide["sigma_m"] == pytest.approx(10.0 * narrow["sigma_m"])
    # A 12 px circle on a contact 30 m away is a quarter of a metre, so the floor is the right
    # order of magnitude and the sigma term is what widens it.
    assert 0.15 < narrow["sigma_m"] < 0.5


def test_junction_witness_abstains_without_a_fitted_state() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    record = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=None,
        outbound_start_xyz=contact,
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    assert record == {"formed": False, "reason": "missing_fitted_state"}


def test_junction_witness_map_attaches_one_witness_to_both_flights() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    point = {
        "point": "m__pt0001",
        "fits": [
            {"flight_index": 0, "start_frame": 90.0, "end_frame": 100.0, "end_xyz": list(contact)},
            {
                "flight_index": 1,
                "start_frame": 100.0,
                "end_frame": 110.0,
                "start_xyz": list(contact + np.asarray([3.0, 0.0, 0.0])),
            },
        ],
    }
    events = {"contact": [{"frame": 100.0, "image_xy": camera.pixel(contact)}]}
    witnesses = audit.junction_witness_map(point, camera, events, {}, {})
    assert sorted(witnesses) == [0, 1]
    assert witnesses[0][0]["side"] == "end"
    assert witnesses[1][0]["side"] == "start"
    # The same verdict reaches both flights: a junction failure is a failure of the pair.
    assert witnesses[0][0]["gap_m"] == pytest.approx(witnesses[1][0]["gap_m"])
    assert witnesses[0][0]["hard_pass"] is False
    # No owner click within two frames of the boundary means no witness at all.
    assert (
        audit.junction_witness_map(
            point, camera, {"contact": [{"frame": 90.0, "image_xy": camera.pixel(contact)}]}, {}, {}
        )
        == {}
    )


def test_truth_good_requires_the_junction_and_noj_keeps_the_old_column() -> None:
    camera = _CourtCamera()
    contact = np.asarray([6.0, 12.0, 1.2], dtype=float)
    point = {"point": "m__pt0001", "match_id": "m", "fits": []}
    attempt = {"flight_index": 0, "start_frame": 90.0, "end_frame": 100.0, "terminal_end": True}
    fit = {
        "flight_index": 0,
        "start_frame": 90.0,
        "end_frame": 100.0,
        "bounces": [],
        "trajectory": [
            {"frame": float(frame), "xyz": list(contact + np.asarray([0.0, 0.0, 0.0]))}
            for frame in range(90, 101)
        ],
    }
    metric_row = {"metric_gate_accepted": False}
    good = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=contact,
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )
    bad = audit.junction_witness(
        camera=camera,
        boundary_frame=100.0,
        click=_junction_click(camera, contact),
        inbound_end_xyz=contact,
        outbound_start_xyz=contact + np.asarray([8.0, 0.0, 0.0]),
        radius_px=audit.CONTACT_GRADED_FLOOR_PX,
        refiner_abstained=False,
    )

    def row_for(witnesses):
        row = audit.flight_truth_row(
            point=point,
            attempt=attempt,
            fit=fit,
            camera=camera,
            owner_events={"contact": [], "bounce": [], "net_hit": []},
            owner_positions={95: camera.pixel(contact)},  # the fit sits exactly on it
            court_bounces=[],
            junction_prev=None,
            junction_next=None,
            pixel_accepted=False,
            metric_row=metric_row,
            track={},
            junction_witnesses=witnesses,
        )
        return row

    without = row_for(None)
    assert without["truth_junction_pass"] is None
    assert without["truth_good_v2"] == without["truth_good_v2_noj"]
    assert without["truth_good_graded"] == without["truth_good_graded_noj"]

    passing = row_for([good])
    assert passing["truth_junction_pass"] is True
    assert passing["truth_good_v2"] is True
    assert passing["truth_good_graded"] is True

    failing = row_for([bad])
    assert failing["truth_junction_pass"] is False
    assert failing["truth_junction_graded_pass"] is False
    # The junction turns a flight every other witness calls right into a wrong one, and the
    # ``_noj`` column keeps the reading every earlier audit reported.
    assert failing["truth_good_v2"] is False
    assert failing["truth_good_graded"] is False
    assert failing["truth_good_v2_noj"] is True
    assert failing["truth_good_graded_noj"] is True
