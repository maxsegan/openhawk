import cv2
import numpy as np
import pytest

import resolution


@pytest.mark.parametrize("target", resolution.SUPPORTED_TEST_SIZES)
def test_point_box_and_normalized_coordinates_are_resolution_invariant(target):
    source = resolution.CANONICAL_SIZE
    points = np.array([[0.0, 0.0], [137.25, 419.5], [959.0, 539.0]])
    boxes = np.array([[10.0, 20.0, 300.0, 500.0]])
    moved = resolution.scale_points(points, source, target)
    np.testing.assert_allclose(
        resolution.scale_points(moved, target, source), points, atol=1e-12
    )
    np.testing.assert_allclose(
        resolution.normalized_points(moved, target),
        resolution.normalized_points(points, source),
        atol=1e-12,
    )
    moved_boxes = resolution.scale_boxes(boxes, source, target)
    np.testing.assert_allclose(
        resolution.scale_boxes(moved_boxes, target, source), boxes, atol=1e-12
    )


@pytest.mark.parametrize("target", resolution.SUPPORTED_TEST_SIZES)
def test_homography_and_projection_preserve_world_geometry(target):
    source = resolution.CANONICAL_SIZE
    h_source = np.array([
        [0.021, 0.002, -9.0],
        [-0.001, 0.043, -2.5],
        [0.00001, 0.0003, 1.0],
    ])
    p_source = np.array([
        [900.0, 20.0, 480.0, -1600.0],
        [5.0, 880.0, 270.0, -9500.0],
        [0.01, 0.02, 1.0, 8.0],
    ])
    uv_source = np.array([[[351.0, 402.0]]], dtype=np.float64)
    uv_target = resolution.scale_points(uv_source, source, target)
    h_target = resolution.image_to_world_homography(h_source, source, target)
    np.testing.assert_allclose(
        cv2.perspectiveTransform(uv_target, h_target),
        cv2.perspectiveTransform(uv_source, h_source),
        atol=1e-10,
    )
    xyz1 = np.array([4.2, 17.5, 1.37, 1.0])
    a = p_source @ xyz1
    b = resolution.world_to_image_projection(p_source, source, target) @ xyz1
    np.testing.assert_allclose(
        b[:2] / b[2],
        resolution.scale_points(a[:2] / a[2], source, target),
        atol=1e-10,
    )


@pytest.mark.parametrize("target", resolution.SUPPORTED_TEST_SIZES)
def test_isotropic_pixel_rates_scale_with_resolution(target):
    expected = target.width / resolution.CANONICAL_SIZE.width
    assert resolution.pixel_length(500.0, resolution.CANONICAL_SIZE, target) == pytest.approx(
        500.0 * expected
    )


def test_highest_resolution_selection_uses_measured_images(tmp_path):
    roots = []
    for name, size in (("misleading_1080", (960, 540)), ("native", (1920, 1080)),
                       ("medium", (1280, 720))):
        root = tmp_path / name / "pt0001"
        root.mkdir(parents=True)
        width, height = size
        assert cv2.imwrite(str(root / "f_0000.jpg"), np.zeros((height, width, 3), np.uint8))
        roots.append(root.parent)
    selected, size = resolution.select_highest_resolution_frame_dir(roots)
    assert selected.endswith("native")
    assert size == resolution.FrameSize(1920, 1080)


def test_known_contact_frame_twin_includes_native_rg_name(tmp_path):
    candidates = resolution.frame_twin_candidates(
        tmp_path / "rally_frames_50_contact_v2"
    )
    assert str(tmp_path / "rally_frames_50_1080") in candidates


def test_coordinate_manifest_propagation_preserves_contract(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    source.write_text("x\n")
    output.write_text("x\n")
    resolution.write_coordinate_manifest(
        resolution.coordinate_manifest_path(source),
        image_size=resolution.FrameSize(1920, 1080),
        artifact_size=resolution.FrameSize(960, 540),
        source="frames",
        subnative_flagged=True,
        subnative_justification="legacy propagation fixture",
    )
    assert resolution.propagate_coordinate_manifest([source], output)
    manifest = resolution.read_coordinate_manifest(output)
    assert manifest["image_size"] == {"width": 1920, "height": 1080}
    assert manifest["artifact_size"] == {"width": 960, "height": 540}


@pytest.mark.parametrize("fps_token", ["29.97", "59.9401"])
def test_player_boxes_resolve_by_contract_identity_not_fps_token(tmp_path, fps_token):
    artifact = tmp_path / f"player_boxes_{fps_token}_native_sided_v1.csv"
    artifact.write_text("clip,frame,side\n")
    resolution.write_coordinate_manifest(
        resolution.coordinate_manifest_path(artifact),
        image_size=resolution.NATIVE_SIZE,
        artifact_size=resolution.NATIVE_SIZE,
        source="frames",
        extra={
            "artifact_identity": resolution.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
            "source_fps": 60000 / 1001 if fps_token == "59.9401" else 30000 / 1001,
        },
    )

    assert resolution.resolve_player_boxes(tmp_path, sided=True) == artifact


def test_player_boxes_resolution_fails_closed_on_ambiguous_artifacts(tmp_path):
    for token in ("59.94", "59.9401"):
        artifact = tmp_path / f"player_boxes_{token}_native_sided_v1.csv"
        artifact.write_text("clip,frame,side\n")
        resolution.write_coordinate_manifest(
            resolution.coordinate_manifest_path(artifact),
            image_size=resolution.NATIVE_SIZE,
            artifact_size=resolution.NATIVE_SIZE,
            source="frames",
            extra={
                "artifact_identity": resolution.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
            },
        )

    with pytest.raises(ValueError, match="ambiguous player_boxes.native_sided.v1"):
        resolution.resolve_player_boxes(tmp_path, sided=True)
