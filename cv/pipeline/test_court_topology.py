import cv2
import numpy as np
import pytest

from cv.pipeline import court
from cv.pipeline import court_topology
from cv.pipeline.court_topology import (
    CourtTopologyHypothesis,
    LANDMARKS,
    find_court_h_topology,
    find_court_h_topology_multiframe,
    player_foot_geometry_score,
    refine_visible_painted_edges,
    transfer_court_homography,
)


def _synthetic_court() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((540, 960, 3), (70, 125, 170), np.uint8)
    image_corners = np.float32([[165, 445], [795, 445], [330, 105], [630, 105]])
    world_corners = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(image_corners, world_corners)
    inverse = np.linalg.inv(homography)
    for start, end in court._court_model():
        pixels = cv2.perspectiveTransform(np.float32([[start, end]]), inverse)[0]
        cv2.line(
            image,
            tuple(np.rint(pixels[0]).astype(int)),
            tuple(np.rint(pixels[1]).astype(int)),
            (250, 250, 250),
            3,
        )
    cv2.line(image, (40, 300), (920, 315), (240, 240, 240), 3)
    cv2.line(image, (130, 80), (800, 500), (240, 240, 240), 3)
    return image, homography


def test_topology_solver_recovers_full_court_with_straight_line_clutter():
    image, expected = _synthetic_court()
    actual = find_court_h_topology(image)
    points = np.asarray(LANDMARKS, np.float32)
    expected_pixels = cv2.perspectiveTransform(points.reshape(1, -1, 2), np.linalg.inv(expected))[0]
    actual_pixels = cv2.perspectiveTransform(points.reshape(1, -1, 2), np.linalg.inv(actual))[0]
    residuals = np.linalg.norm(expected_pixels - actual_pixels, axis=1)
    assert np.median(residuals) <= 3.0
    assert np.quantile(residuals, 0.9) <= 6.0


def test_multiframe_recovery_rejects_context_after_camera_cut():
    context, _ = _synthetic_court()
    target = np.full_like(context, (70, 125, 170))
    with pytest.raises(ValueError, match="camera-stable"):
        find_court_h_topology_multiframe(target, [context])


def test_trusted_homography_transfers_across_static_scene_motion():
    reference, reference_homography = _synthetic_court()
    transform = np.float32([[1.0, 0.0, 6.0], [0.0, 1.0, 3.0]])
    target = cv2.warpAffine(reference, transform, (reference.shape[1], reference.shape[0]))
    actual, evidence = transfer_court_homography(target, reference, reference_homography)
    points = np.asarray(LANDMARKS, np.float32)
    expected_pixels = cv2.transform(
        cv2.perspectiveTransform(points.reshape(1, -1, 2), np.linalg.inv(reference_homography)),
        transform,
    )[0]
    actual_pixels = cv2.perspectiveTransform(points.reshape(1, -1, 2), np.linalg.inv(actual))[0]
    assert np.median(np.linalg.norm(expected_pixels - actual_pixels, axis=1)) <= 1.0
    assert evidence["inliers"] >= 40


def test_expanded_solver_continues_after_one_proposal_arm_fails(monkeypatch):
    image, expected = _synthetic_court()

    def hypotheses(_image, *, spatially_balanced, **_kwargs):
        if not spatially_balanced:
            raise ValueError("missing standard lines")
        return (CourtTopologyHypothesis(expected, 0.95),)

    monkeypatch.setattr(court_topology, "court_topology_hypotheses", hypotheses)
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *_args: 0.95)

    solution = court_topology.solve_court_h_topology(image)

    assert solution.source == "standard_balanced"


def test_player_feet_downweight_depth_shifted_court_alias():
    _, expected = _synthetic_court()
    feet = cv2.perspectiveTransform(
        np.float32([[[5.0, 22.0], [6.0, 1.0]]]),
        np.linalg.inv(expected),
    )[0]
    shifted = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -12.0], [0.0, 0.0, 1.0]]) @ expected

    assert player_foot_geometry_score(expected, feet) == 1.0
    assert player_foot_geometry_score(shifted, feet) < 1.0


def test_visible_edge_refinement_is_guarded_by_local_painted_line_evidence():
    image, homography = _synthetic_court()
    translation = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    biased = homography @ np.linalg.inv(translation)

    refined, evidence = refine_visible_painted_edges(image, biased)

    assert evidence["accepted"] is True
    assert evidence["refined_lines"] >= 5
    assert all(
        row.get("edge_offset_px", 0.0) >= 0.0
        for row in evidence["lines"].values()
        if row["refined"]
    )
    assert not np.array_equal(refined, biased)


def test_visible_edge_refinement_falls_back_without_painted_line_evidence():
    image, homography = _synthetic_court()
    image[:] = (70, 125, 170)

    refined, evidence = refine_visible_painted_edges(image, homography)

    assert evidence["accepted"] is False
    assert evidence["reason"] == "insufficient_refined_lines"
    np.testing.assert_array_equal(refined, homography)


def _synthetic_court_native() -> tuple[np.ndarray, np.ndarray]:
    image, homography = _synthetic_court()
    native = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    return native, court_topology.res.image_to_world_homography(
        homography,
        court_topology.res.CANONICAL_SIZE,
        court_topology.res.FrameSize(1920, 1080),
    )


@pytest.mark.parametrize(
    ("orientation", "convention", "expected_sign"),
    [
        ("near_horizontal", "itf_outside_edge", 1.0),
        ("far_horizontal", "itf_outside_edge", -1.0),
        ("near_horizontal", "image_lower_edge", 1.0),
        ("far_horizontal", "image_lower_edge", 1.0),
        ("near_horizontal", "paint_centre", 1.0),
        ("far_horizontal", "paint_centre", 1.0),
    ],
)
def test_horizontal_edge_normal_follows_the_convention(orientation, convention, expected_sign):
    line = np.asarray([0.02, 1.0, -400.0])

    normal = court_topology._oriented_line_normal(line, orientation, convention)

    assert np.sign(normal[1]) == expected_sign


def test_sideline_edge_normals_always_point_away_from_the_court_centre():
    left = np.asarray([1.0, 0.4, -200.0])
    right = np.asarray([1.0, -0.4, -700.0])

    for convention in court_topology.EDGE_CONVENTIONS:
        assert court_topology._oriented_line_normal(left, "left_vertical", convention)[0] < 0
        assert court_topology._oriented_line_normal(right, "right_vertical", convention)[0] > 0


def test_paint_centre_model_insets_each_line_by_half_its_width():
    outside = dict(zip(("near", "far"), court_topology.paint_centre_segments("itf_outside_edge")))
    lower = dict(zip(("near", "far"), court_topology.paint_centre_segments("image_lower_edge")))

    assert outside["near"][0][1] == pytest.approx(0.05)
    assert outside["far"][0][1] == pytest.approx(court.COURT_L - 0.05)
    assert lower["far"][0][1] == pytest.approx(court.COURT_L + 0.05)
    assert court_topology.paint_centre_segments("paint_centre") == court_topology.MODEL_SEGMENTS
    assert court_topology.paint_centre_landmarks("paint_centre") == court_topology.LANDMARKS


def test_unknown_conventions_and_scales_are_rejected():
    with pytest.raises(ValueError, match="edge convention"):
        court_topology.resolve_edge_convention("outer")
    with pytest.raises(ValueError, match="solve scale"):
        court_topology.resolve_solve_scale("1080p")
    with pytest.raises(ValueError, match="refinement witness"):
        court_topology.resolve_refinement_witness("strict")
    with pytest.raises(ValueError, match="line orientation"):
        court_topology._oriented_line_normal(np.asarray([0.0, 1.0, 0.0]), "horizontal")


def test_environment_selects_the_solve_scale_and_conventions(monkeypatch):
    monkeypatch.setenv(court_topology.SOLVE_SCALE_ENVIRONMENT_VARIABLE, "canonical_540")
    monkeypatch.setenv(court_topology.EDGE_CONVENTION_ENVIRONMENT_VARIABLE, "paint_centre")
    monkeypatch.setenv(court_topology.REFINEMENT_WITNESS_ENVIRONMENT_VARIABLE, "paint_inset")

    assert court_topology.resolve_solve_scale(None) == "canonical_540"
    assert court_topology.resolve_edge_convention(None) == "paint_centre"
    assert court_topology.resolve_refinement_witness(None) == "paint_inset"
    assert court_topology.resolve_solve_scale("native") == "native"


def test_topology_score_is_unchanged_by_the_resolution_it_is_measured_at():
    image, homography = _synthetic_court()
    native, native_homography = _synthetic_court_native()

    canonical_score = court_topology.topology_score(homography, court.line_mask(image, 0.12))
    native_score = court_topology.topology_score(native_homography, court.line_mask(native, 0.12))

    assert native_score == pytest.approx(canonical_score, abs=0.03)


def test_native_solve_returns_a_homography_in_native_pixels():
    native, expected = _synthetic_court_native()

    solution = court_topology.solve_court_h_topology_native(native, solve_scale="native")

    assert solution.refinement_evidence["solve_size"] == "1920x1080"
    points = np.asarray(LANDMARKS, np.float32).reshape(1, -1, 2)
    expected_pixels = cv2.perspectiveTransform(points, np.linalg.inv(expected))[0]
    actual_pixels = cv2.perspectiveTransform(points, np.linalg.inv(solution.homography))[0]
    assert np.median(np.linalg.norm(expected_pixels - actual_pixels, axis=1)) <= 6.0


def test_canonical_solve_scale_reports_the_scale_it_ran_at():
    native, _ = _synthetic_court_native()

    solution = court_topology.solve_court_h_topology_native(native, solve_scale="canonical_540")

    assert solution.refinement_evidence["solve_size"] == "960x540"
    assert solution.refinement_evidence["solve_scale"] == "canonical_540"


def test_centre_mode_returns_the_band_centroid_rather_than_its_outer_edge():
    offsets = np.arange(-6.0, 6.5, 0.5)
    profile = np.where(np.abs(offsets - 1.0) <= 1.0, 100.0, 0.0)

    edge, _ = court_topology._visible_edge_offset(offsets, profile, mode="outer_edge")
    centre, _ = court_topology._visible_edge_offset(offsets, profile, mode="centre")

    assert edge == pytest.approx(2.0)
    assert centre == pytest.approx(1.0)


def test_the_witness_scores_a_candidate_against_the_model_its_convention_implies():
    image, homography = _synthetic_court()
    mask = court.line_mask(image, top_fraction=0.12)
    # A court shifted a quarter of a metre down the y axis, so no score saturates at 1.0.
    biased = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, -0.25], [0.0, 0.0, 1.0]]) @ homography

    nominal = court_topology.topology_score(biased, mask)
    scores = {
        convention: court_topology.topology_score(
            biased, mask, convention="paint_inset", edge_convention=convention
        )
        for convention in court_topology.EDGE_CONVENTIONS
    }

    assert scores["paint_centre"] == pytest.approx(nominal)
    assert scores["itf_outside_edge"] != pytest.approx(nominal)
    assert scores["image_lower_edge"] != pytest.approx(nominal)
    assert scores["itf_outside_edge"] != pytest.approx(scores["image_lower_edge"])


def test_edge_convention_transform_moves_lines_by_half_a_paint_width():
    identity, evidence = court_topology.edge_convention_world_transform("paint_centre")
    assert np.allclose(identity, np.eye(3), atol=1e-6)
    assert evidence["corner_residual_maximum_m"] < 1e-5
    transform, evidence = court_topology.edge_convention_world_transform("itf_outside_edge")
    # A per-line offset is not exactly one projective map; the cost of representing it as
    # one must stay far below the offsets themselves.
    assert evidence["corner_residual_maximum_m"] < 0.005
    moved = cv2.perspectiveTransform(
        np.float32([[[0.0, 0.0], [court.COURT_W, court.COURT_L]]]), transform
    )[0]
    assert moved[0] == pytest.approx([0.025, 0.05], abs=0.002)
    assert moved[1] == pytest.approx([court.COURT_W - 0.025, court.COURT_L - 0.05], abs=0.002)


def test_edge_convention_pushes_a_projected_baseline_outwards():
    _, homography = _synthetic_court()
    converted = court_topology.to_edge_convention(homography, "itf_outside_edge")
    corner = np.float32([[[0.0, 0.0]]])
    before = cv2.perspectiveTransform(corner, np.linalg.inv(homography))[0][0]
    after = cv2.perspectiveTransform(corner, np.linalg.inv(converted))[0][0]
    # The near baseline's outside edge is below its paint centre in a broadcast frame.
    assert after[1] > before[1]
    assert 1.0 < np.linalg.norm(after - before) < 12.0


def test_transfer_accepts_clay_paint_only_on_the_stricter_relaxed_witness(monkeypatch):
    reference, reference_homography = _synthetic_court()
    target = reference.copy()
    scores = {"standard": 0.60, "relaxed_chroma": 0.95}
    monkeypatch.setattr(
        court_topology,
        "topology_score",
        lambda homography, mask, **kwargs: scores[mask[0, 0]],
    )
    monkeypatch.setattr(
        court_topology,
        "court_line_masks",
        lambda image: tuple(
            (name, np.full((4, 4), name, dtype=object)) for name in ("standard", "relaxed_chroma")
        ),
    )
    _, evidence = transfer_court_homography(target, reference, reference_homography)
    assert evidence["target_support_mask"] == "relaxed_chroma"
    scores["relaxed_chroma"] = 0.90
    with pytest.raises(ValueError, match="lacks target support"):
        transfer_court_homography(target, reference, reference_homography)


def test_strong_topology_accepts_a_sun_and_shadow_court_the_surface_witness_rejects(monkeypatch):
    image, homography = _synthetic_court()
    hypothesis = CourtTopologyHypothesis(homography=homography, score=0.97)
    monkeypatch.setattr(court_topology, "court_topology_hypotheses", lambda *a, **k: [hypothesis])
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *a, **k: 0.61)
    solution = court_topology.solve_court_h_topology(image)
    assert solution.source.endswith("_strong_topology")
    assert solution.surface_score == pytest.approx(0.61)
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *a, **k: 0.40)
    with pytest.raises(ValueError, match="topology and surface witnesses"):
        court_topology.solve_court_h_topology(image)


def _sunlit_grass_court() -> tuple[np.ndarray, np.ndarray]:
    """A real image of one court under two illuminations, with ordinary surface texture.

    Mown bands and a worn baseline area vary the same surface's luminance; a hard shadow
    edge along the court diagonal then darkens everything on one side of it, as a stand
    roof does. Nothing is mocked: the surface witnesses measure these pixels.
    """
    image, homography = _synthetic_court()
    inverse = np.linalg.inv(homography)
    surface = np.ones(image.shape[:2])
    for index, band in enumerate(np.arange(0.0, court.COURT_L, 2.0)):
        corners = np.float32(
            [
                [-2.0, band],
                [court.COURT_W + 2.0, band],
                [court.COURT_W + 2.0, band + 2.0],
                [-2.0, band + 2.0],
            ]
        )
        pixels = cv2.perspectiveTransform(corners.reshape(1, -1, 2), inverse)[0]
        cv2.fillPoly(surface, [np.rint(pixels).astype(np.int32)], 1.0 if index % 2 else 0.92)
    worn = cv2.perspectiveTransform(np.float32([[[court.COURT_W / 2, 1.0]]]), inverse)[0][0]
    cv2.ellipse(surface, tuple(np.rint(worn).astype(int)), (170, 22), 0, 0, 360, 0.86, -1)
    image = np.clip(image * surface[..., None], 0, 255).astype(np.uint8)
    ((near, far),) = cv2.perspectiveTransform(
        np.float32([[[0.0, 0.0], [court.COURT_W, court.COURT_L]]]), inverse
    )
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    side = (grid_x - near[0]) * (far[1] - near[1]) - (grid_y - near[1]) * (far[0] - near[0])
    shadow = np.where(side > 0, 0.55, 1.0)
    return np.clip(image * shadow[..., None], 0, 255).astype(np.uint8), homography


def _landmark_residuals(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    points = np.asarray(LANDMARKS, np.float32).reshape(1, -1, 2)
    expected_pixels = cv2.perspectiveTransform(points, np.linalg.inv(expected))[0]
    actual_pixels = cv2.perspectiveTransform(points, np.linalg.inv(actual))[0]
    return np.linalg.norm(expected_pixels - actual_pixels, axis=1)


def test_measured_illumination_split_rejects_the_court_the_split_witness_recovers():
    image, expected = _sunlit_grass_court()
    with pytest.raises(ValueError, match="topology and surface witnesses"):
        court_topology.solve_court_h_topology(image)

    solution = court_topology.solve_court_h_topology(
        image, surface_witness_policy="illumination_split"
    )

    assert solution.acceptance_route == court_topology.ILLUMINATION_SPLIT_ROUTE
    assert solution.source.endswith("_strong_topology_illumination_split")
    assert solution.topology_score >= court_topology.STRONG_TOPOLOGY_SCORE
    assert solution.surface_score < court_topology.STRONG_TOPOLOGY_MINIMUM_SURFACE_SCORE
    assert (
        solution.illumination_surface_score >= court_topology.STRONG_TOPOLOGY_MINIMUM_SURFACE_SCORE
    )
    residuals = _landmark_residuals(expected, solution.homography)
    assert np.median(residuals) <= 3.0
    assert np.quantile(residuals, 0.9) <= 6.0


def test_an_evenly_lit_court_solves_identically_under_either_surface_witness():
    image, expected = _synthetic_court()
    baseline = court_topology.solve_court_h_topology(image)
    solution = court_topology.solve_court_h_topology(
        image, surface_witness_policy="illumination_split"
    )
    assert solution.acceptance_route == court_topology.STRICT_SURFACE_ROUTE
    assert solution.illumination_surface_score is None
    assert solution.source == baseline.source
    assert np.allclose(solution.homography, baseline.homography)
    assert np.median(_landmark_residuals(expected, solution.homography)) <= 3.0


def test_a_court_off_the_playing_surface_is_still_rejected_under_the_split_witness():
    image, expected = _sunlit_grass_court()
    displaced = expected @ np.float32([[1, 0, 0], [0, 1, 400], [0, 0, 1]])
    mask = court.line_mask(image, top_fraction=0.12)

    assert court_topology.court_surface_illumination_consistency(image, displaced) == 0.0
    assert court_topology.topology_score(displaced, mask) < 0.80
    solution = court_topology.solve_court_h_topology(
        image, surface_witness_policy="illumination_split"
    )
    assert np.median(_landmark_residuals(expected, solution.homography)) <= 3.0
    assert np.median(_landmark_residuals(displaced, solution.homography)) > 6.0


def test_the_split_witness_is_no_sharper_than_the_guard_it_stands_in_for():
    """A rescaled court passes both colour witnesses; topology is what rejects it."""
    image, expected = _sunlit_grass_court()
    centre = np.float32([[0.75, 0, 120.0], [0, 0.75, 67.5], [0, 0, 1]])
    rescaled = expected @ np.linalg.inv(centre)

    assert court_topology.court_surface_illumination_consistency(image, rescaled) >= (
        court_topology.STRONG_TOPOLOGY_MINIMUM_SURFACE_SCORE
    )
    assert court_topology.topology_score(rescaled, court.line_mask(image, top_fraction=0.12)) < (
        court_topology.STRONG_TOPOLOGY_SCORE
    )
    solution = court_topology.solve_court_h_topology(
        image, surface_witness_policy="illumination_split"
    )
    assert np.median(_landmark_residuals(expected, solution.homography)) <= 3.0


def test_an_unknown_surface_witness_policy_is_rejected():
    image, _ = _synthetic_court()
    with pytest.raises(ValueError, match="unknown surface witness policy"):
        court_topology.solve_court_h_topology(image, surface_witness_policy="chroma_only")


def _low_framed_painted_court():
    corners = np.float32([[150, 490], [810, 490], [330, 160], [630, 160]])
    world = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(corners, world)
    mask = np.zeros((540, 960), np.uint8)
    for start, end in court_topology.MODEL_SEGMENTS:
        pixels = court_topology._project_court_points(homography, np.float32([start, end]))
        cv2.line(
            mask,
            tuple(np.rint(pixels[0]).astype(int)),
            tuple(np.rint(pixels[1]).astype(int)),
            255,
            3,
        )
    return mask, homography


def test_topology_excludes_graphics_pixels_without_penalizing_visible_baseline():
    mask, homography = _low_framed_painted_court()
    observed = court.line_mask_observation_mask(mask.shape, top_fraction=0.12)
    masked = mask.copy()
    masked[~observed] = 0
    original = court_topology.topology_score(homography, masked)
    qualified = court_topology.topology_score(homography, masked, observation_mask=observed)
    assert original < 0.92
    assert qualified > 0.99


def test_topology_still_penalizes_missing_paint_in_observed_region():
    mask, homography = _low_framed_painted_court()
    observed = court.line_mask_observation_mask(mask.shape, top_fraction=0.12)
    mask[~observed] = 0
    mask[480:500, 450:850] = 0
    assert court_topology.topology_score(homography, mask, observation_mask=observed) < 0.92


def test_topology_requires_spatial_support_after_observation_exclusions():
    mask, homography = _low_framed_painted_court()
    observed = np.zeros(mask.shape, dtype=bool)
    observed[480:500] = True
    mask[~observed] = 0
    assert court_topology.topology_score(homography, mask, observation_mask=observed) == 0


def test_topology_observation_mask_requires_matching_boolean_pixels():
    mask, homography = _low_framed_painted_court()
    for observed in [np.ones((1, 1), dtype=bool), np.ones(mask.shape, dtype=np.uint8)]:
        with pytest.raises(ValueError, match="observation_mask"):
            court_topology.topology_score(homography, mask, observation_mask=observed)


def test_transfer_accepts_supported_clipped_baseline_without_lowering_threshold(monkeypatch):
    mask, homography = _low_framed_painted_court()
    observed = court.line_mask_observation_mask(mask.shape, top_fraction=0.12)
    mask[~observed] = 0
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    monkeypatch.setattr(
        court_topology,
        "register_static_scene",
        lambda *_: (np.eye(3), {"inliers": 100, "inlier_ratio": 1.0}),
    )
    monkeypatch.setattr(
        court_topology,
        "court_line_masks",
        lambda _: (("standard", np.zeros_like(mask)), ("relaxed_chroma", mask)),
    )
    actual, evidence = transfer_court_homography(image, image, homography)
    assert np.allclose(actual, homography)
    assert evidence["target_support_mask"] == "relaxed_chroma"
    assert evidence["target_topology_score"] >= court_topology.MINIMUM_RELAXED_TARGET_SCORE
    mask[480:500, 450:850] = 0
    with pytest.raises(ValueError, match="lacks target support"):
        transfer_court_homography(image, image, homography)


@pytest.mark.parametrize("dy,height", [(0, 540), (50, 540), (100, 640), (-40, 480)])
def test_standard_broadcast_geometry_allows_vertical_translation_and_crop(dy, height):
    image_corners = np.float32([[165, 445], [795, 445], [330, 175], [630, 175]])
    image_corners[:, 1] += dy
    world = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(image_corners, world)
    assert court_topology._standard_broadcast_view(homography, 960, height)


@pytest.mark.parametrize(
    "corners",
    [
        [[165, 445], [795, 445], [330, 355], [630, 355]],
        [[165, 175], [795, 175], [330, 445], [630, 445]],
        [[330, 445], [630, 445], [165, 175], [795, 175]],
        [[165, 445], [795, 350], [330, 175], [630, 100]],
        [[165, 745], [795, 745], [330, 475], [630, 475]],
    ],
)
def test_standard_broadcast_geometry_keeps_depth_order_perspective_and_frame_guards(corners):
    world = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(np.float32(corners), world)
    assert not court_topology._standard_broadcast_view(homography, 960, 540)


def test_standard_broadcast_geometry_allows_weak_perspective():
    corners = np.float32([[192.5, 450], [767.5, 450], [193, 175], [767, 175]])
    world = np.float32(
        [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]]
    )
    homography = cv2.getPerspectiveTransform(corners, world)
    assert court_topology._standard_broadcast_view(homography, 960, 540)


def test_relaxed_refinement_rejects_paint_loss_hidden_by_standard_mask(monkeypatch):
    image, homography = _synthetic_court()
    standard = np.zeros(image.shape[:2], dtype=np.uint8)
    relaxed = np.full(image.shape[:2], 255, dtype=np.uint8)
    monkeypatch.setattr(court, "line_mask", lambda *args, **kwargs: standard)
    monkeypatch.setattr(court_topology, "relaxed_chroma_line_mask", lambda *args, **kwargs: relaxed)

    def score(candidate, mask, **kwargs):
        original = np.allclose(candidate, homography)
        return (0.83 if original else 0.78) if mask is relaxed else (0.10 if original else 0.105)

    monkeypatch.setattr(court_topology, "topology_score", score)
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *args: 0.95)
    _, old_witness = refine_visible_painted_edges(image, homography)
    retained, matched_witness = refine_visible_painted_edges(
        image, homography, witness_mask="relaxed_chroma"
    )
    assert old_witness["accepted"]
    assert not matched_witness["accepted"]
    assert matched_witness["reason"] == "independent_witness_regression"
    assert matched_witness["original_topology_score"] == 0.83
    assert matched_witness["refined_topology_score"] == 0.78
    np.testing.assert_array_equal(retained, homography)


@pytest.mark.parametrize("convention,accepted", [("nominal", False), ("paint_inset", True)])
def test_matched_refinement_mask_preserves_explicit_edge_witness(monkeypatch, convention, accepted):
    image, homography = _synthetic_court()
    seen = []

    def score(candidate, mask, **kwargs):
        original = np.allclose(candidate, homography)
        witness = kwargs.get("convention", "nominal")
        seen.append((original, witness, kwargs.get("edge_convention")))
        return 0.83 if original else 0.84 if witness == "paint_inset" else 0.78

    monkeypatch.setattr(court_topology, "topology_score", score)
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *args: 0.95)
    _, evidence = refine_visible_painted_edges(
        image,
        homography,
        witness_mask="relaxed_chroma",
        witness_convention=convention,
        edge_convention="itf_outside_edge",
    )
    assert evidence["accepted"] == accepted
    assert seen[0][:2] == (True, "nominal")
    assert seen[1] == (False, convention, "itf_outside_edge")
    assert evidence["witness_mask"] == "relaxed_chroma"
    assert evidence["refined_topology_convention"] == convention


def test_native_solver_threads_selected_proposal_mask_and_score(monkeypatch):
    image, homography = _synthetic_court()
    monkeypatch.setattr(
        court_topology,
        "solve_court_h_topology",
        lambda *args, **kwargs: court_topology.CourtTopologySolution(
            homography, "relaxed_chroma_balanced", 0.91, 0.95, proposal_mask="relaxed_chroma"
        ),
    )

    def refine(image, candidate, **kwargs):
        assert kwargs["witness_mask"] == "relaxed_chroma"
        return candidate, {"accepted": False, "reason": "independent_witness_regression"}

    monkeypatch.setattr(court_topology, "refine_visible_painted_edges", refine)
    result = court_topology.solve_court_h_topology_native(image, solve_scale="native")
    assert result.proposal_mask == "relaxed_chroma"
    assert result.topology_score == 0.91
    assert result.refinement_evidence["proposal_topology_score"] == 0.91
    assert result.refinement_evidence["proposal_topology_convention"] == "nominal"


def test_expanded_solver_records_relaxed_proposal_mask(monkeypatch):
    image, homography = _synthetic_court()
    calls = []

    def hypotheses(*args, **kwargs):
        calls.append(kwargs)
        return (CourtTopologyHypothesis(homography, 0.95),) if len(calls) == 3 else ()

    monkeypatch.setattr(court_topology, "court_topology_hypotheses", hypotheses)
    monkeypatch.setattr(court_topology, "court_surface_consistency", lambda *args: 0.95)
    result = court_topology.solve_court_h_topology(image)
    assert result.source == "relaxed_chroma_balanced"
    assert result.proposal_mask == "relaxed_chroma"


def test_refinement_rejects_unknown_mask():
    image, homography = _synthetic_court()
    with pytest.raises(ValueError, match="unknown refinement witness mask"):
        refine_visible_painted_edges(image, homography, witness_mask="unspecified")
