"""Focused tests for measured native streak support on one observed ball pixel.

Every picture here is planted: a background with ordinary 8-bit noise, plus whatever
shape the case is about.  The assertions are on measured geometry in native pixels and
on which abstention the module reaches, never on its internal ordering of checks.
"""

import numpy as np
import pytest
import cv2

from cv.pipeline import event_wing_streak as streak
from cv.pipeline.resolution import FrameSize

ORIGIN = (900, 400)
GEOMETRY = streak.PatchGeometry(origin_native=ORIGIN, image_size=FrameSize(1920, 1080))
FPS = 25.0


def _background(seed, shape=(64, 64), level=80):
    """A flat patch with a few levels of independent compression-like noise."""
    rng = np.random.default_rng(seed)
    return (level + rng.integers(-2, 3, size=shape)).astype(np.uint8)


def _trio(shape=(64, 64), level=80):
    return [_background(seed, shape, level) for seed in (1, 2, 3)]


def _native(current, before, after, indices=(40, 41, 42)):
    frames = [
        streak.NativeFrame(image, index, index / FPS)
        for image, index in zip((current, before, after), (indices[1], indices[0], indices[2]))
    ]
    return frames[0], [frames[1], frames[2]]


def _native_point(x, y):
    return (ORIGIN[0] + x, ORIGIN[1] + y)


def _measure(current, before, after, point, **kwargs):
    frame, neighbours = _native(current, before, after)
    return streak.measure_support(frame, neighbours, point, GEOMETRY, **kwargs)


def _disk(image, cx, cy, radius, value):
    ys, xs = np.ogrid[: image.shape[0], : image.shape[1]]
    image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2] = value


# ---------------------------------------------------------------------------
# what a measurable streak looks like
# ---------------------------------------------------------------------------


def test_thin_moving_streak_is_measured_in_native_pixels():
    current, before, after = _trio()
    current[31:34, 16:45] = 200  # 29 native px long, 3 native px wide
    support = _measure(current, before, after, _native_point(30.5, 32.5))

    assert support["available"] is True
    assert support["reason"] == "measured_streak_support"
    measurement = support["measurement"]
    # The planted extent is 29 x 3 native px and comes back as such: no resampling,
    # no smoothing and no rescaling happens anywhere between the patch and here.
    assert measurement["length_native_px"] == pytest.approx(29.0, abs=0.5)
    assert measurement["transverse_width_native_px"] == pytest.approx(3.0, abs=0.5)
    assert abs(measurement["axis_native"][0]) == pytest.approx(1.0, abs=1e-6)
    assert measurement["centroid_native"] == pytest.approx(list(_native_point(30.5, 32.5)), abs=0.6)
    ends = sorted(measurement["endpoints_native"])
    assert ends[0] == pytest.approx(list(_native_point(16.0, 32.5)), abs=0.6)
    assert ends[1] == pytest.approx(list(_native_point(45.0, 32.5)), abs=0.6)
    assert measurement["is_corrected_ball_position"] is False
    assert measurement["is_event_time"] is False
    assert support["source"]["current"]["frame_index"] == 41
    assert [n["native_time_seconds"] for n in support["source"]["neighbours"]] == [
        40 / FPS,
        42 / FPS,
    ]


def test_round_ball_is_measured_but_leaves_its_axis_undetermined():
    current, before, after = _trio()
    _disk(current, 32, 32, 3, 210)
    support = _measure(current, before, after, _native_point(32.5, 32.5))

    assert support["available"] is True
    measurement = support["measurement"]
    assert measurement["length_native_px"] == pytest.approx(7.0, abs=1.0)
    # A disk has no direction of its own, so no direction may be claimed from it and
    # the tolerance it earns is about the size of the ball.
    assert measurement["axis_determined"] is False
    assert measurement["elongation"] < streak.CONFIG["minimum_directional_elongation"]


@pytest.mark.parametrize("overlap", [False, True])
def test_bright_racket_strings_do_not_enlarge_ball_support(overlap):
    current, before, after = _trio()
    left = 25 if overlap else 42
    for x in range(left, left + 16, 4):
        cv2.line(current, (x, 20), (x, 45), 235, 1)
    for y in range(20, 46, 5):
        cv2.line(current, (left, y), (left + 15, y), 235, 1)
    current[31:34, 10:36] = 210
    support = _measure(current, before, after, _native_point(23, 32))
    if overlap:
        assert not support["available"]
    elif support["available"]:
        # Conservative abstention is allowed; strings must never become ball blur.
        assert support["measurement"]["length_native_px"] <= 27


def test_ball_crossing_a_static_court_line_does_not_borrow_its_extent():
    current, before, after = _trio()
    for picture in (current, before, after):
        picture[:, 31:34] = 230
    current[30:34, 15:47] = 210
    support = _measure(current, before, after, _native_point(26, 32))
    if support["available"]:
        assert support["measurement"]["length_native_px"] <= 33
        assert support["measurement"]["transverse_width_native_px"] <= 5


def test_rgb_and_dark_components_are_admitted_on_contrast_not_on_being_white():
    yellow = [np.repeat(plane[:, :, None], 3, axis=2) for plane in _trio()]
    yellow[0][31:34, 16:45] = (200, 200, 40)
    colour = _measure(*yellow, _native_point(30.5, 32.5))
    assert colour["available"] is True
    assert colour["measurement"]["contrast_polarity"] == "brighter"

    current, before, after = _trio(level=200)
    current[31:34, 16:45] = 60
    dark = _measure(current, before, after, _native_point(30.5, 32.5))
    assert dark["available"] is True
    assert dark["measurement"]["contrast_polarity"] == "darker"
    assert dark["measurement"]["length_native_px"] == pytest.approx(29.0, abs=0.5)


# ---------------------------------------------------------------------------
# what must not be measured
# ---------------------------------------------------------------------------


def test_static_white_patch_carries_no_measured_support():
    current, before, after = _trio()
    for image in (current, before, after):
        image[28:36, 24:40] = 240  # a line junction or a painted logo: bright, and still
    current[10:13, 16:45] = 200  # an unrelated moving streak keeps the pictures distinct
    support = _measure(current, before, after, _native_point(31.5, 31.5))

    assert support["available"] is False
    assert support["reason"] == "no_moving_component_at_observed_point"
    assert support["measurement"] is None


def test_broad_moving_player_shape_is_rejected_on_ball_geometry():
    current, before, after = _trio(shape=(96, 96))
    current[30:56, 20:50] = 150  # 30 x 26 native px of moving torso
    support = _measure(current, before, after, _native_point(34.5, 42.5))

    assert support["available"] is False
    assert support["reason"] == "component_width_outside_ball_geometry"


def test_ball_merged_into_a_racket_shape_is_rejected_on_compactness():
    current, before, after = _trio(shape=(96, 96))
    current[31:34, 16:45] = 200  # the streak
    current[24:34, 38:41] = 200  # a perpendicular bar it has merged with
    support = _measure(current, before, after, _native_point(20.5, 32.5))

    assert support["available"] is False
    assert support["reason"] == "component_not_a_single_compact_width"


def test_two_ball_like_components_are_ambiguous():
    current, before, after = _trio()
    current[31:34, 16:45] = 200
    current[41:44, 16:45] = 200
    support = _measure(current, before, after, _native_point(30.5, 32.5))

    assert support["available"] is False
    assert support["reason"] == "ambiguous_competing_component"


def test_nearby_player_shape_does_not_block_a_clean_ball_measurement():
    """Ambiguity is about rival *balls*; a limb next to the ball is not one.

    Wing frames near a contact sit next to the striker, so a blanket "something else is
    moving nearby" rule would abstain exactly where the extra tolerance is needed.
    """
    current, before, after = _trio(shape=(96, 96))
    current[20:23, 16:45] = 200  # the ball streak
    current[45:71, 16:46] = 150  # a limb 22 native px away, too wide to be a ball
    support = _measure(current, before, after, _native_point(30.5, 21.5))

    assert support["available"] is True
    assert support["measurement"]["length_native_px"] == pytest.approx(29.0, abs=0.5)


def test_component_running_off_the_patch_edge_is_rejected():
    current, before, after = _trio()
    current[31:34, 0:20] = 200
    support = _measure(current, before, after, _native_point(10.5, 32.5))

    assert support["available"] is False
    assert support["reason"] == "component_clipped_by_patch_edge"
    # This patch does not start at the native picture edge, so the truncation is the
    # crop's, and the extent measured from it would have been a lower bound either way.
    assert support["diagnostics"]["clipped_at_native_image_edge"] is False


# ---------------------------------------------------------------------------
# evidence identity
# ---------------------------------------------------------------------------


def test_duplicate_pictures_are_not_two_observations():
    current, before, after = _trio()
    current[31:34, 16:45] = 200
    support = _measure(current, before, current.copy(), _native_point(30.5, 32.5))

    assert support["available"] is False
    assert support["reason"] == "duplicate_native_pictures"


def test_repeated_and_nonmonotonic_exposures_are_rejected():
    current, before, after = _trio()
    current[31:34, 16:45] = 200
    point = _native_point(30.5, 32.5)

    repeated = streak.measure_support(
        streak.NativeFrame(current, 41, 41 / FPS),
        [streak.NativeFrame(before, 40, 40 / FPS), streak.NativeFrame(after, 40, 42 / FPS)],
        point,
        GEOMETRY,
    )
    assert repeated["reason"] == "duplicate_or_nonunique_native_exposures"

    backwards = streak.measure_support(
        streak.NativeFrame(current, 41, 41 / FPS),
        [streak.NativeFrame(before, 40, 43 / FPS), streak.NativeFrame(after, 42, 42 / FPS)],
        point,
        GEOMETRY,
    )
    assert backwards["reason"] == "nonmonotonic_native_timestamps"

    distant = streak.measure_support(
        streak.NativeFrame(current, 41, 41 / FPS),
        [streak.NativeFrame(before, 40, 40 / FPS), streak.NativeFrame(after, 62, 62 / FPS)],
        point,
        GEOMETRY,
    )
    assert distant["reason"] == "neighbour_exposures_not_contemporaneous"


def test_upsampled_or_sub_native_patches_are_refused():
    current, before, after = _trio()
    current[31:34, 16:45] = 200
    frame, neighbours = _native(current, before, after)
    point = _native_point(30.5, 32.5)

    upsampled = streak.measure_support(
        frame,
        neighbours,
        point,
        streak.PatchGeometry(ORIGIN, FrameSize(1920, 1080), native_px_per_patch_px=0.5),
    )
    assert upsampled["available"] is False
    assert upsampled["reason"] == "patch_not_at_native_scale"

    legacy = streak.measure_support(
        frame, neighbours, point, streak.PatchGeometry(ORIGIN, FrameSize(960, 540))
    )
    assert legacy["available"] is False
    assert legacy["reason"] == "source_below_native_tracking_resolution"


# ---------------------------------------------------------------------------
# background that this module does not model
# ---------------------------------------------------------------------------


def test_camera_shifted_background_with_no_ball_abstains():
    """A pan is detected and abstained on; its residual is never read as ball blur."""
    rng = np.random.default_rng(11)
    scene = (80 + rng.integers(-2, 3, size=(64, 80))).astype(np.uint8)
    scene[:, ::9] = 190  # court lines and paint edges across the whole patch
    current = scene[:, 8:72].copy()
    before = scene[:, 6:70].copy()
    after = scene[:, 10:74].copy()
    support = _measure(current, before, after, _native_point(31.5, 31.5))

    assert support["available"] is False
    assert support["reason"] == "unsupported_background_motion_at_patch_rim"
    assert support["reason"] in streak.UNSUPPORTED_BACKGROUND_REASONS


def test_neighbours_that_disagree_about_the_background_abstain():
    current, before, after = _trio()
    current[31:34, 16:45] = 140
    before[31:34, 16:45] = 30  # whatever is behind the component is not static:
    after[31:34, 16:45] = 240  # the two neighbours do not agree about it
    support = _measure(current, before, after, _native_point(30.5, 32.5))

    assert support["available"] is False
    assert support["reason"] == "neighbour_backgrounds_disagree_behind_component"
    assert support["reason"] in streak.UNSUPPORTED_BACKGROUND_REASONS


@pytest.mark.parametrize(
    "current_span, before_span, after_span, point_x",
    [
        ((20, 42), (20, 36), (20, 39), 37.5),  # a shadow or paint boundary advancing
        ((20, 36), (20, 39), (20, 42), 37.5),  # the same boundary receding
    ],
)
def test_interior_structure_shifted_under_a_quiet_rim_abstains(
    current_span, before_span, after_span, point_x
):
    """The dangerous pan: one that never touches the rim and that the neighbours agree on.

    A camera move over an interior edge leaves a sliver of pure difference that has a
    ball's area, width, length and compactness, and the two neighbours agree about what
    is behind it, so neither of the other two background checks sees it.  What gives it
    away is that its surround is two backgrounds with a step between them, not one.
    """
    current, before, after = _trio()
    for image, span in ((current, current_span), (before, before_span), (after, after_span)):
        image[20:45, span[0] : span[1]] = 40
    support = _measure(current, before, after, _native_point(point_x, 32.5))

    assert support["available"] is False
    assert support["reason"] in streak.UNSUPPORTED_BACKGROUND_REASONS + (
        "insufficient_current_frame_contrast",
    )
    assert support["diagnostics"]["rim_motion_fraction"] == 0.0


def test_thin_static_line_displaced_by_the_camera_abstains():
    current, before, after = _trio()
    current[20:45, 34:38] = 200  # a painted line, 4 native px wide, panning 3 px a frame
    before[20:45, 31:35] = 200
    after[20:45, 37:41] = 200
    support = _measure(current, before, after, _native_point(35.5, 32.5))

    assert support["available"] is False
    assert support["reason"] in streak.UNSUPPORTED_BACKGROUND_REASONS


# ---------------------------------------------------------------------------
# direction compatibility
# ---------------------------------------------------------------------------


def test_long_component_across_the_observed_motion_is_rejected_only_when_motion_is_given():
    current, before, after = _trio()
    current[16:45, 31:34] = 200  # a long component square across the travel direction
    point = _native_point(32.5, 30.5)

    blind = _measure(current, before, after, point)
    assert blind["available"] is True
    # Honest limitation: with no measured direction the module cannot tell this from a
    # ball streak, so a caller that can supply one from measured rows should.
    assert blind["measurement"]["direction_gate"] == "not_supplied"

    directed = _measure(current, before, after, point, observed_motion_native_px=(12.0, 0.5))
    assert directed["available"] is False
    assert directed["reason"] == "axis_incompatible_with_observed_motion"

    along = _measure(current, before, after, point, observed_motion_native_px=(0.5, -12.0))
    assert along["available"] is True
    assert along["measurement"]["direction_gate"] == "applied"
    assert (
        along["measurement"]["axis_alignment_cosine"]
        > streak.CONFIG["minimum_axis_alignment_cosine"]
    )


# ---------------------------------------------------------------------------
# the geometric containment helper
# ---------------------------------------------------------------------------


def _support(length=40.0, width=4.0, centre=(120.0, 200.0)):
    half = 0.5 * length
    return {
        "schema": streak.SCHEMA,
        "available": True,
        "measurement": {
            "endpoints_native": [
                [centre[0] - half, centre[1]],
                [centre[0] + half, centre[1]],
            ],
            "transverse_width_native_px": width,
        },
    }


@pytest.mark.parametrize(
    "point, inside",
    [
        ((120.0, 200.0), True),  # the centre
        ((140.0, 200.0), True),  # 2 px past the centre path
        ((142.0, 200.0), False),  # diameter is not position uncertainty
        ((144.0, 200.0), False),  # 4 px past the end
        ((120.0, 202.0), True),
        ((120.0, 204.0), False),
        ((120.0, 206.0), False),  # 6 px across
    ],
)
def test_predicted_point_against_measured_extent(point, inside):
    verdict = streak.within_measured_support(point, _support())
    assert verdict["inside"] is inside
    assert verdict["longitudinal_limit_native_px"] == pytest.approx(21.0)
    assert verdict["transverse_limit_native_px"] == pytest.approx(3.0)


def test_cross_track_allowance_never_grows_with_the_streak_length():
    short = streak.within_measured_support((120.0, 206.0), _support(length=12.0))
    long = streak.within_measured_support((120.0, 206.0), _support(length=120.0))
    assert short["transverse_limit_native_px"] == long["transverse_limit_native_px"]
    assert short["inside"] is False and long["inside"] is False
    # The long streak does buy the longitudinal tolerance it measured, and only that.
    assert streak.within_measured_support((178.0, 200.0), _support(length=120.0))["inside"] is True


def test_unavailable_support_grants_nothing_and_rejects_nothing():
    for absent in ({}, {"available": False, "measurement": None}, {"available": True}):
        verdict = streak.within_measured_support((120.0, 200.0), absent)
        assert verdict["inside"] is False
        assert verdict["reason"] == "support_unavailable"
        # The caller must keep its original 3 px test; this is additive evidence only.
        assert verdict["combine_with_original_test"] == "or_never_and"


def test_round_component_reduces_to_radius_three_disk():
    support = _support(length=6, width=6)
    assert streak.within_measured_support((122, 202), support)["inside"]
    assert not streak.within_measured_support((122.5, 202.5), support)["inside"]


def test_rotated_centre_path_and_endpoint_capsule():
    support = _support(length=14, width=4, centre=(0, 0))
    unit = np.array([1.0, 1.0]) / np.sqrt(2)
    support["measurement"]["endpoints_native"] = [(-7 * unit).tolist(), (7 * unit).tolist()]
    assert streak.within_measured_support(7 * unit, support)["inside"]
    assert not streak.within_measured_support(9 * unit, support)["inside"]


def test_two_earlier_pictures_cannot_supply_a_bracketing_background():
    current, before, after = _trio()
    current[31:34, 16:45] = 200
    frame = streak.NativeFrame(current, 42, 42 / FPS)
    neighbours = [streak.NativeFrame(before, 40, 40 / FPS), streak.NativeFrame(after, 41, 41 / FPS)]
    support = streak.measure_support(frame, neighbours, _native_point(30, 32), GEOMETRY)
    assert support["reason"] == "neighbours_do_not_straddle_current_exposure"


def test_diagonal_streak_is_a_measured_native_extent():
    current, before, after = _trio()
    cv2.line(current, (20, 20), (42, 42), 220, 3)
    support = _measure(
        current, before, after, _native_point(31, 31), observed_motion_native_px=(5, 5)
    )
    assert support["available"]
    assert abs(support["measurement"]["axis_native"][0]) == pytest.approx(2**-0.5, abs=0.01)
    assert 3 <= support["measurement"]["transverse_width_native_px"] <= 5
