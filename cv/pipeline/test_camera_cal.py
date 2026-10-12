from __future__ import annotations

import numpy as np
import cv2
import camera_cal

from camera_cal import (
    fit_net_top_envelope,
    fixed_f_projection_from_ground,
    intrinsic_projection_from_ground,
    measure_net_band,
    NET_POST_X,
    NET_Y,
    net_height_at_x,
    project,
    projection_from_ground,
    shared_vertical_column,
    solve_f_from_net,
)


def test_net_top_envelope_does_not_average_tape_thickness() -> None:
    candidates = []
    for image_y in range(400, 421, 2):
        candidates.append((float(image_y), -900.0, (500, image_y, 1400, image_y)))
    candidates.extend(
        [
            (397.0, -300.0, (500, 397, 800, 397)),
            (398.0, -300.0, (800, 398, 1100, 398)),
            (399.0, -300.0, (1100, 399, 1400, 399)),
        ]
    )

    coefficients = fit_net_top_envelope(candidates, 500.0, 1400.0)

    assert coefficients is not None
    predicted = np.polyval(coefficients, [550.0, 950.0, 1350.0])
    assert np.max(predicted) < 403.0


def test_net_measurement_falls_back_when_top_envelope_has_no_focal(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((40, 60, 3), dtype=np.uint8))
    top = [(0.0, 1.0, 10.0, 30.0, 20.0)] * 9
    center = [(0.0, 1.0, 10.0, 30.0, 25.0)] * 9

    def fake_measure(
        _image,
        _homography,
        *,
        prefer_top_envelope=True,
        support="singles_sticks",
        tape_measurement="hough_top",
    ):
        return top if prefer_top_envelope else center

    monkeypatch.setattr(camera_cal, "measure_net_band", fake_measure)
    monkeypatch.setattr(
        camera_cal,
        "solve_f_from_net",
        lambda _homography, observations, **_kwargs: 1800.0 if observations is center else None,
    )

    result = camera_cal.measure_point_net((1, np.eye(3), str(image_path), 60, 40, "singles_sticks"))

    assert result[2] is center
    assert result[3] == 1800.0
    assert result[4] == "hough_net_centerline_fallback"
    assert not camera_cal.net_source_is_tape_top(result[4])


def test_shared_vertical_column_preserves_each_ground_homography() -> None:
    h1 = np.array([[10.0, 0.0, 100.0], [0.0, 8.0, 50.0], [0.0, 0.0, 1.0]])
    h2 = np.array([[11.0, 0.2, 98.0], [0.1, 8.2, 52.0], [0.0, 0.0, 1.0]])
    vertical = np.array([0.5, -20.0, -0.001])
    p1 = projection_from_ground(h1, vertical) * 3.0
    p2 = projection_from_ground(h2, vertical) * 0.25

    estimated = shared_vertical_column({1: p1, 2: p2}, {1: h1, 2: h2})
    fallback = projection_from_ground(h2, estimated)

    np.testing.assert_allclose(estimated, vertical)
    np.testing.assert_allclose(fallback[:, [0, 1, 3]], h2)


def test_intrinsic_fallback_preserves_ground_homography() -> None:
    homography = np.array(
        [
            [43.5382447, 12.5830161, 45.9655886],
            [0.827292695, -6.9192004, 385.040871],
            [0.000140440941, 0.0131021808, 0.299166087],
        ]
    )

    solved = intrinsic_projection_from_ground(homography)

    assert solved is not None
    projection, focal = solved
    np.testing.assert_allclose(
        projection[:, [0, 1, 3]],
        homography / homography[2, 2],
    )
    assert 400.0 <= focal <= 40000.0


def test_measure_net_band_selects_tape_top_at_native_resolution() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    homography = np.array(
        [
            [100.0, 0.0, 300.0],
            [0.0, 20.0, 362.3],
            [0.0, 0.0, 1.0],
        ]
    )
    cv2.line(image, (300, 520), (1397, 520), (255, 255, 255), 5)
    cv2.line(image, (360, 472), (1337, 472), (255, 255, 255), 6)
    for row in range(540, 601, 12):
        cv2.line(image, (300, row), (1397, row), (180, 180, 180), 1)

    observations = measure_net_band(image, homography)

    assert observations is not None
    assert max(abs(top - 517.0) for _, _, _, _, top in observations) <= 5.0
    assert len(observations) == 9
    np.testing.assert_allclose(
        [observations[0][2], observations[-1][2]],
        [300.0, 1397.0],
        atol=1.1,
    )
    inverse = np.linalg.inv(homography)
    recovered = cv2.perspectiveTransform(
        np.float32([[[row[2], row[3]] for row in observations]]),
        inverse,
    )[0]
    np.testing.assert_allclose([row[0] for row in observations], recovered[:, 0], atol=1e-5)
    np.testing.assert_allclose(
        [row[1] for row in observations],
        [net_height_at_x(row[0]) for row in observations],
    )


def test_focal_fit_matches_cord_without_false_ground_x_correspondence() -> None:
    homography = np.array(
        [
            [80.0, 2.0, 300.0],
            [1.0, 18.0, 420.0],
            [0.0001, 0.01, 1.0],
        ]
    )
    expected_focal = 3200.0
    projection = fixed_f_projection_from_ground(
        homography,
        focal=expected_focal,
        w=1920,
        h=1080,
    )
    court_x = np.linspace(NET_POST_X[0], NET_POST_X[1], 41)
    cord = project(
        projection,
        np.stack(
            [
                court_x,
                np.full_like(court_x, NET_Y),
                [net_height_at_x(float(x)) for x in court_x],
            ],
            axis=1,
        ),
    )
    indices = np.linspace(0, len(cord) - 1, 9).round().astype(int)
    observations = [
        (0.0, 0.914, float(cord[index, 0]), 0.0, float(cord[index, 1])) for index in indices
    ]

    fitted = solve_f_from_net(homography, observations, w=1920, h=1080)

    assert fitted is not None
    assert abs(fitted - expected_focal) / expected_focal < 0.08


def test_fixed_f_projection_preserves_ground_homography() -> None:
    homography = np.array(
        [
            [80.0, 2.0, 300.0],
            [1.0, 18.0, 420.0],
            [0.0001, 0.01, 1.0],
        ]
    )

    projection = fixed_f_projection_from_ground(
        homography,
        focal=3200.0,
        w=1920,
        h=1080,
    )

    np.testing.assert_allclose(
        projection[:, [0, 1, 3]],
        homography / homography[2, 2],
    )


def test_net_height_model_holds_the_tape_up_at_the_singles_sticks() -> None:
    stick_x = camera_cal.NET_STICK_X[0]

    assert camera_cal.net_height_at_x(camera_cal.COURT_W / 2.0) == camera_cal.NET_H_CENTER
    assert camera_cal.net_height_at_x(stick_x) == camera_cal.NET_H_POST
    # beyond the stick the cord runs level between two 1.07 m tie points
    assert camera_cal.net_height_at_x(camera_cal.NET_POST_X[0]) == camera_cal.NET_H_POST
    # the retired doubles-post model sits about 6 cm low where the sticks hold the tape
    doubles = camera_cal.net_height_at_x(stick_x, "doubles_posts")
    assert 0.05 <= camera_cal.NET_H_POST - doubles <= 0.07


def _curved_tape_scene(*, tape: bool, occlusion: bool = False):
    image = np.full((1080, 1920, 3), (145, 110, 65), dtype=np.uint8)
    homography = np.array([[86.0, 0.0, 488.0], [0.0, 16.0, 309.84], [0.0, 0.0, 1.0]])
    xs = np.arange(488, 1432)
    top = 416.0 - 0.000035 * (xs - 960.0) ** 2
    for row in range(426, 471, 2):
        cv2.line(image, (488, row), (1431, row), (190, 190, 190), 1)
    if tape:
        for x, y in zip(xs, top, strict=True):
            image[int(round(y)) : int(round(y)) + 6, x] = (235, 240, 238)
    # A bright crossing line and ball must not become the top of the tape.
    cv2.line(image, (960, 385), (960, 472), (255, 255, 255), 4)
    cv2.circle(image, (1040, 401), 4, (80, 250, 200), -1)
    if occlusion:
        image[390:475, 920:980] = (30, 30, 30)
    return image, homography


def test_tape_pixels_recover_curved_band_amid_dense_mesh() -> None:
    image, homography = _curved_tape_scene(tape=True)
    observations = measure_net_band(image, homography, tape_measurement="connected_pixels")
    assert observations is not None
    for _, _, image_x, _, top_y in observations:
        expected = 416.0 - 0.000035 * (image_x - 960.0) ** 2 - 0.5
        assert abs(top_y - expected) < 2.0


def test_tape_pixels_support_two_visible_sides_of_occlusion() -> None:
    image, homography = _curved_tape_scene(tape=True, occlusion=True)
    observations = measure_net_band(image, homography, tape_measurement="connected_pixels")
    assert observations is not None
    assert observations[-1][2] - observations[0][2] > 800
    assert abs(observations[4][4] - 415.5) < 2.0


def test_mesh_only_and_crossing_bright_pixels_do_not_certify_tape() -> None:
    image, homography = _curved_tape_scene(tape=False)
    assert measure_net_band(image, homography, tape_measurement="connected_pixels") is None


def test_sparse_bright_tape_fragment_does_not_extrapolate_whole_net() -> None:
    image, homography = _curved_tape_scene(tape=False)
    cv2.line(image, (820, 415), (920, 415), (255, 255, 255), 6)
    assert measure_net_band(image, homography, tape_measurement="connected_pixels") is None


def test_default_tape_measurement_does_not_fall_back_to_isolated_columns(monkeypatch) -> None:
    image, homography = _curved_tape_scene(tape=False)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *_args, **_kwargs: None)
    assert measure_net_band(image, homography, tape_measurement="connected_pixels") is None


def test_tape_measurement_default_preserves_hough_top() -> None:
    image, homography = _curved_tape_scene(tape=True)
    default = measure_net_band(image, homography)
    explicit = measure_net_band(image, homography, tape_measurement="hough_top")
    np.testing.assert_array_equal(default, explicit)
    pixel = measure_net_band(image, homography, tape_measurement="connected_pixels")
    assert default is not None and pixel is not None
    assert np.median(np.array(default)[:, 4] - np.array(pixel)[:, 4]) > 2.0


def test_connected_top_failure_does_not_become_legacy_tape_in_point_measurement(tmp_path) -> None:
    image, homography = _curved_tape_scene(tape=False)
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), image)
    result = camera_cal.measure_point_net(
        (1, homography, str(image_path), 1920, 1080, "singles_sticks"),
        tape_measurement="connected_pixels",
    )
    assert result[2:] == (None, None, "unavailable")


def test_tape_top_provenance_rejects_both_legacy_centerline_spellings() -> None:
    assert camera_cal.net_source_is_tape_top("observed_connected_tape_pixels_top")
    assert camera_cal.net_source_is_tape_top("observed_connected_tape_top_envelope")
    assert not camera_cal.net_source_is_tape_top("hough_net_centerline_fallback")
    assert not camera_cal.net_source_is_tape_top("observed_connected_tape_centerline_fallback")
    assert not camera_cal.net_source_is_tape_top("unavailable")
