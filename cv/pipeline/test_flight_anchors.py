from types import SimpleNamespace

import numpy as np

from cv.pipeline.flight_anchors import (
    BALL_RADIUS_M,
    NET_COURT_Y_M,
    bounce_anchor,
    build_point_anchors,
    contact_ray_measurement,
    find_net_anchor,
    ray_at_height,
    ray_at_net,
)


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


def test_ray_plane_intersections_recover_exact_metric_points() -> None:
    projection = _projection()
    point = np.array([3.25, NET_COURT_Y_M, 1.35])
    pixel = _project(projection, point)

    np.testing.assert_allclose(ray_at_net(projection, pixel), point, atol=1e-9)
    np.testing.assert_allclose(ray_at_height(projection, pixel, point[2]), point, atol=1e-9)


def test_bounce_anchor_uses_event_pixel_and_propagates_uncertainty() -> None:
    projection = _projection()
    point = np.array([7.0, 17.5, BALL_RADIUS_M])
    pixel = _project(projection, point)
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 1.0, "source": "direct"},
    )

    anchor = bounce_anchor(
        {"frame": 12.5, "event_type": "bounce", "x1080": pixel[0], "y1080": pixel[1]},
        {},
        camera,
    )

    np.testing.assert_allclose(anchor["xyz"], point, atol=1e-9)
    assert anchor["sigma_m"] > 0.0
    assert anchor["source"] == "emission:x1080,y1080"


def test_contact_ray_prefers_native_emission_location_and_has_unknown_depth() -> None:
    projection = _projection()
    point = np.array([4.5, 18.0, 2.7])
    pixel = _project(projection, point)
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 0.9, "source": "direct"},
    )

    ray = contact_ray_measurement(
        {
            "frame": 12.0,
            "event_type": "contact",
            "location": {"image_x": pixel[0], "image_y": pixel[1]},
        },
        {12: pixel + 100.0},
        camera,
    )

    assert ray is not None
    np.testing.assert_allclose(ray["image_xy"], pixel)
    assert ray["source"] == "emission:location.image_x,location.image_y"
    assert "xyz" not in ray
    assert ray["boundary_offset_bounds_frames"] == [-1.0, 1.0]


def test_contact_ray_falls_back_to_track_at_emitted_frame() -> None:
    projection = _projection()
    point = np.array([4.5, 18.0, 1.2])
    pixel = _project(projection, point)
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 1.0, "source": "direct"},
    )

    ray = contact_ray_measurement(
        {"frame": 12.0, "event_type": "contact"},
        {12: pixel},
        camera,
    )

    assert ray is not None
    np.testing.assert_allclose(ray["image_xy"], pixel)
    assert ray["source"] == "track_interpolation"
    assert ray["source_frames"] == [12]


def test_point_anchors_never_construct_a_net_depth_anchor() -> None:
    projection = _projection()
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 1.0, "source": "direct"},
    )
    start = np.array([5.0, 18.0, 1.2])
    bounce = np.array([5.0, 8.0, BALL_RADIUS_M])
    track = {0: _project(projection, start), 10: _project(projection, bounce)}

    artifact = build_point_anchors(
        "match__pt0001",
        [{"flight_index": 0, "start_frame": 0.0, "end_frame": 20.0}],
        [
            {"event_type": "contact", "frame": 0.0},
            {"event_type": "bounce", "frame": 10.0, "x1080": track[10][0], "y1080": track[10][1]},
        ],
        track,
        camera,
    )

    assert artifact["schema"] == "flight_anchors_v2"
    assert [row["type"] for row in artifact["flights"][0]["anchors"]] == ["bounce"]
    assert artifact["flights"][0]["net_role"] == "post_fit_plausibility_only"


def test_terminal_bounce_at_attempt_endpoint_is_an_anchor() -> None:
    projection = _projection()
    point = np.array([5.0, 8.0, BALL_RADIUS_M])
    track = {frame: _project(projection, point) for frame in range(4)}
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 1.0, "source": "direct"},
    )
    event = {
        "event_type": "bounce",
        "frame": 3.0,
        "x1080": track[3][0],
        "y1080": track[3][1],
    }

    terminal = build_point_anchors(
        "match__pt0001",
        [
            {
                "flight_index": 0,
                "start_frame": 0.0,
                "end_frame": 3.0,
                "terminal_end": True,
            }
        ],
        [event],
        track,
        camera,
    )
    ordinary = build_point_anchors(
        "match__pt0001",
        [{"flight_index": 0, "start_frame": 0.0, "end_frame": 3.0}],
        [event],
        track,
        camera,
    )

    assert terminal["flights"][0]["bounce_anchors"] == 1
    assert ordinary["flights"][0]["bounce_anchors"] == 0


def test_net_anchor_interpolates_the_projected_court_crossing() -> None:
    projection = _projection()
    frames = range(8, 15)
    points = {
        frame: np.array(
            [
                4.0,
                NET_COURT_Y_M + (frame - 11) * 0.8,
                1.3 + 4.0 * ((frame - 8) / 25.0) - 0.5 * 9.81 * ((frame - 8) / 25.0) ** 2,
            ]
        )
        for frame in frames
    }
    track = {frame: _project(projection, point) for frame, point in points.items()}
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 1.0, "source": "direct"},
    )

    anchor = find_net_anchor(track, camera, 8.0, 14.0)

    assert anchor is not None
    assert 10.0 < anchor["frame"] < 12.0
    assert abs(anchor["xyz"][1] - NET_COURT_Y_M) < 1e-9
    assert 1.0 < anchor["xyz"][2] < 2.2
    assert anchor["source_frames"] == [10, 11] or anchor["source_frames"] == [11, 12]


def _court_bounce_track(
    projection: np.ndarray, bounce_xy: np.ndarray, bounce_frame: float
) -> dict[int, np.ndarray]:
    """A ball whose court-plane shadow turns a corner at ``bounce_frame``.

    The corner witness reads the track after the image-to-court homography, so a track built
    from a shadow that really does bend at the bounce is what tests the construction rather
    than the projection.
    """
    incoming = np.array([0.3, -9.0])
    outgoing = np.array([2.5, 6.0])
    track = {}
    for frame in range(int(bounce_frame) - 5, int(bounce_frame) + 6):
        elapsed = float(frame) - bounce_frame
        velocity = incoming if elapsed < 0.0 else outgoing
        xy = bounce_xy + velocity * elapsed / 25.0
        track[frame] = _project(projection, np.r_[xy, BALL_RADIUS_M])
    return track


def _image_to_court(projection: np.ndarray) -> np.ndarray:
    """The homography a per-frame court module publishes: image to the ``z = 0`` court plane."""
    ground = np.column_stack((projection[:, 0], projection[:, 1], projection[:, 3]))
    return np.linalg.inv(ground)


def test_the_court_plane_corner_witness_puts_a_bounce_where_the_track_bends() -> None:
    from cv.pipeline.flight_anchors import bounce_court_witness

    projection = _projection()
    homography = _image_to_court(projection)
    bounce_xy = np.array([6.2, 15.4])
    track = _court_bounce_track(projection, bounce_xy, 40.4)
    camera = SimpleNamespace(
        p_at=lambda _frame: projection, h_at=lambda _frame: homography, quality_at=lambda _f: {}
    )

    witness = bounce_court_witness({"event_type": "bounce", "frame": 40.0}, track, camera, 25.0)

    assert witness is not None
    assert witness["source"] == "subframe_timing.witness_kinematic_court"
    # The corner is where the ball really bounced, and its own claimed width knows it.
    assert float(np.linalg.norm(np.asarray(witness["xy"]) - bounce_xy)) < 0.15
    assert witness["sigma_m"] >= 0.02
    assert abs(witness["t_subframe"] - 40.4) < 0.3


def test_a_bounce_with_no_readable_corner_carries_no_witness() -> None:
    from cv.pipeline.flight_anchors import build_bounce_witnesses

    projection = _projection()
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: np.eye(3))
    events = [{"event_type": "bounce", "frame": 40.0}, {"event_type": "contact", "frame": 10.0}]

    witnesses = build_bounce_witnesses(events, {}, camera, 25.0)

    assert witnesses == {}


def test_the_bounce_anchor_carries_the_court_witness_it_was_given() -> None:
    projection = _projection()
    camera = SimpleNamespace(
        p_at=lambda _frame: projection,
        quality_at=lambda _frame: {"reliable": True, "confidence": 0.9, "source": "test"},
    )
    truth = np.array([6.0, 15.0, BALL_RADIUS_M])
    pixel = _project(projection, truth)
    witness = {"xy": [6.05, 15.02], "sigma_m": 0.04}

    anchor = bounce_anchor(
        {"event_type": "bounce", "frame": 40.0, "image_x": pixel[0], "image_y": pixel[1]},
        {40: pixel},
        camera,
        court_witness=witness,
    )

    assert anchor is not None
    assert anchor["court_witness"] == witness
    assert (
        bounce_anchor(
            {"event_type": "bounce", "frame": 40.0, "image_x": pixel[0], "image_y": pixel[1]},
            {40: pixel},
            camera,
        )["court_witness"]
        is None
    )


def test_the_contact_time_prior_is_widened_by_the_bench_calibration() -> None:
    """Section 4's measurement, pinned: the contact claim is 1.15 short of a 90% interval."""
    from cv.pipeline.flight_anchors import SUBFRAME_TIME_PRIOR_SIGMA_SCALE

    assert SUBFRAME_TIME_PRIOR_SIGMA_SCALE["bounce"] == 1.0
    assert SUBFRAME_TIME_PRIOR_SIGMA_SCALE["contact"] == 1.15


def test_the_time_prior_reports_and_applies_its_own_width_scale() -> None:
    from cv.pipeline import flight_anchors

    projection = _projection()
    homography = _image_to_court(projection)
    track = _court_bounce_track(projection, np.array([6.2, 15.4]), 40.4)
    camera = SimpleNamespace(p_at=lambda _frame: projection, h_at=lambda _frame: homography)

    prior = flight_anchors.impact_time_prior(
        {"event_type": "bounce", "frame": 40.0}, track, camera, 25.0
    )

    assert prior is not None
    assert prior["sigma_scale"] == 1.0
    expected = min(
        max(
            prior["sigma_scale"] * prior["raw_sigma_frames"],
            flight_anchors.SUBFRAME_TIME_PRIOR_MIN_SIGMA_FRAMES,
        ),
        flight_anchors.ANCHOR_TIME_SIGMA_FRAMES,
    )
    assert prior["sigma_frames"] == expected
