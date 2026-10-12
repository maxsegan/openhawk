import json
import pathlib

import numpy as np
import pytest

from cv.pipeline import resolution
from cv.pipeline.event_model_v2_features import (
    BASE_FEATURE_NAMES,
    RADIUS,
    TRACK_NAME,
    CAMERA_ELEVATION_FEATURE_NAME,
    COURT_FEATURE_NAMES,
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    _add_physical_features,
    _coordinate_sizes,
    _project,
    feature_names,
    frame_features,
)


def test_project_converts_artifact_pixels_to_native_homography_space() -> None:
    homography = np.asarray(
        [
            [0.03169318084188355, 0.01829832735625702, -24.87591809556843],
            [0.0, -0.044723990665129616, 36.94201628939706],
            [-4.55272867886001e-19, 0.003346247620272605, 1.0],
        ]
    )

    projected = _project(
        homography,
        196.875,
        474.375,
        image_size=resolution.FrameSize(1920, 1080),
        artifact_size=resolution.FrameSize(960, 540),
    )

    assert projected == pytest.approx((1.189007, -1.315017), abs=1e-6)


def test_frame_features_append_corrected_physical_channels() -> None:
    base, court_present = frame_features(
        3,
        {1: (500.0, 250.0)},
        np.eye(3),
        image_size=resolution.FrameSize(1000, 500),
        artifact_size=resolution.FrameSize(1000, 500),
        court_missing="zero",
    )
    features = _add_physical_features(base[None, :, :], -0.25)[0]
    index = {name: position for position, name in enumerate(LEGACY_FEATURE_NAMES)}

    assert base.shape == (3, len(BASE_FEATURE_NAMES))
    assert court_present.tolist() == [False, True, False]
    assert features.shape == (3, len(LEGACY_FEATURE_NAMES))
    assert features[1, index["court_x"]] == pytest.approx(500.0 / 10.97)
    assert features[1, index["court_y"]] == pytest.approx(250.0 / 23.77)
    assert features[1, index["distance_camera_interaction"]] == pytest.approx(0.5 * (250.0 / 23.77))
    assert features[:, index["camera_elevation_proxy"]].tolist() == [-0.25] * 3
    assert features[1, index["court_surface_signed_margin"]] < 0.0


def test_camera_elevation_proxy_is_off_by_default() -> None:
    assert CAMERA_ELEVATION_FEATURE_NAME not in FEATURE_NAMES
    assert CAMERA_ELEVATION_FEATURE_NAME in LEGACY_FEATURE_NAMES
    assert len(FEATURE_NAMES) == 18
    assert len(LEGACY_FEATURE_NAMES) == 19
    assert feature_names(include_camera_elevation=False) == FEATURE_NAMES


def test_missing_court_geometry_is_nan_not_zero() -> None:
    base, court_present = frame_features(
        3,
        {1: (500.0, 250.0)},
        None,
        image_size=resolution.FrameSize(1000, 500),
        artifact_size=resolution.FrameSize(1000, 500),
    )
    features = _add_physical_features(base[None, :, :])[0]
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}

    assert not court_present.any()
    assert features.shape == (3, len(FEATURE_NAMES))
    for name in COURT_FEATURE_NAMES:
        assert np.isnan(features[:, index[name]]).all(), name
    assert features[1, index["image_x"]] == pytest.approx(0.5)
    assert features[1, index["track_presence"]] == 1.0


def test_zero_policy_reproduces_the_legacy_silent_fill() -> None:
    base, _ = frame_features(
        3,
        {1: (500.0, 250.0)},
        None,
        image_size=resolution.FrameSize(1000, 500),
        artifact_size=resolution.FrameSize(1000, 500),
        court_missing="zero",
    )
    index = {name: position for position, name in enumerate(BASE_FEATURE_NAMES)}

    assert base[1, index["court_x"]] == 0.0
    assert base[1, index["court_y"]] == 0.0
    assert base[1, index["net_proximity"]] == 0.0


def test_unsupported_court_missing_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="court_missing"):
        frame_features(
            1,
            {},
            None,
            image_size=resolution.FrameSize(1000, 500),
            artifact_size=resolution.FrameSize(1000, 500),
            court_missing="interpolate",
        )


def test_coordinate_sizes_are_required_from_sidecar(tmp_path) -> None:
    track = tmp_path / "track.csv"
    track.write_text("clip,frame,x,y\n")
    resolution.coordinate_manifest_path(track).write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "image_size": {"width": 2048, "height": 1152},
                "artifact_size": {"width": 1024, "height": 576},
            }
        )
    )

    assert _coordinate_sizes(track) == (
        resolution.FrameSize(2048, 1152),
        resolution.FrameSize(1024, 576),
    )
    resolution.coordinate_manifest_path(track).unlink()
    with pytest.raises(FileNotFoundError, match="explicit coordinate sidecar"):
        _coordinate_sizes(track)


def test_dataset_masks_default_to_all_present() -> None:
    from cv.pipeline.event_model_v2_features import AutomaticDataset

    dataset = AutomaticDataset(
        windows=np.zeros((2, 25, len(FEATURE_NAMES)), dtype=np.float32),
        clips=np.asarray(["m__pt0001"] * 2),
        broadcasts=np.asarray(["m"] * 2),
        frames=np.asarray([1, 2]),
        court_geometry_missing=np.asarray([True, False]),
    )

    assert dataset.missing_court_geometry().tolist() == [True, False]
    assert dataset.missing_court_coordinate().tolist() == [False, False]
    assert dataset.subset(np.asarray([False, True])).missing_court_geometry().tolist() == [False]


def _synthetic_camera(
    *,
    height_m: float = 9.0,
    behind_m: float = 12.0,
    focal_px: float = 1400.0,
    image_size: resolution.FrameSize = resolution.FrameSize(1920, 1080),
) -> tuple[np.ndarray, np.ndarray]:
    """A camera behind the near baseline, and the ground homography it induces.

    Returns ``(P, H)`` where ``P`` maps court metres to native pixels and ``H``
    maps native pixels back onto the court plane, which is the pair the event
    feature builder and the net derivation read.
    """

    from cv.pipeline.event_model_v2_features import COURT_LENGTH_M, COURT_WIDTH_M

    centre = np.asarray([COURT_WIDTH_M / 2.0, -behind_m, height_m])
    forward = np.asarray([0.0, COURT_LENGTH_M / 2.0, 0.0]) + np.asarray(
        [COURT_WIDTH_M / 2.0, 0.0, 0.0]
    )
    forward = forward - centre
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    intrinsics = np.asarray(
        [
            [focal_px, 0.0, image_size.width / 2.0],
            [0.0, focal_px, image_size.height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.column_stack((rotation, -rotation @ centre))
    projection = intrinsics @ extrinsics
    ground = projection[:, [0, 1, 3]]
    return projection, np.linalg.inv(ground)


def test_track_pixel_space_follows_the_sidecars_declared_column_map(tmp_path) -> None:
    from cv.pipeline.event_model_v2_features import track_column_size, track_pixel_space

    track = tmp_path / "track.csv"
    track.write_text("clip,frame,x,y,x_native,y_native\n")
    manifest = {
        "schema": "tennis.coordinate-space.v1",
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {"width": 1920, "height": 1080},
        "legacy_artifact_size": {"width": 960, "height": 540},
        "coordinate_columns": {
            "legacy_960x540": ["x", "y"],
            "native_1920x1080": ["x_native", "y_native"],
        },
    }
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))

    # A dual-column sidecar is read in its native columns, at the authoritative
    # artifact size.  Reading x/y here against that size was the defect.
    space = track_pixel_space(track)
    assert (space.x_column, space.y_column) == ("x_native", "y_native")
    assert space.space == "native_1920x1080"
    assert space.size == resolution.FrameSize(1920, 1080)
    assert track_column_size(track) == resolution.FrameSize(1920, 1080)

    # A legacy-only sidecar is read in its legacy columns at the legacy size.
    manifest["coordinate_columns"] = {"legacy_960x540": ["x", "y"]}
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))
    space = track_pixel_space(track)
    assert (space.x_column, space.y_column) == ("x", "y")
    assert space.size == resolution.FrameSize(960, 540)

    # A sidecar with no column map keeps the historical rule.
    del manifest["coordinate_columns"]
    manifest["artifact_size"] = {"width": 960, "height": 540}
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))
    space = track_pixel_space(track)
    assert (space.x_column, space.y_column, space.space) == ("x", "y", "artifact_size")
    assert space.size == resolution.FrameSize(960, 540)


def _write_point_root(
    tmp_path,
    homography: np.ndarray,
    rows: list[tuple[int, float, float]],
    *,
    n_frames: int = 40,
    dual_columns: bool = True,
) -> pathlib.Path:
    """A one-broadcast, one-point cohort root the feature builder can read.

    ``rows`` are ``(frame, native_x, native_y)``; the track file carries them in
    both the legacy 960x540 columns and the native ones, exactly as
    ``cohort_root_v3`` does.
    """

    root = tmp_path / "root"
    match_root = root / "bc"
    match_root.mkdir(parents=True)
    (root / "active_play_v1.json").write_text(
        json.dumps({"bc/pt0001": {"n_frames": n_frames, "event_spans": [[5, 9]]}})
    )
    track = match_root / TRACK_NAME
    header = "clip,frame,x,y,score,x_native,y_native" if dual_columns else "clip,frame,x,y,score"
    lines = [header]
    for frame, x_native, y_native in rows:
        legacy = f"{x_native / 2.0},{y_native / 2.0}"
        native = f",{x_native},{y_native}"
        lines.append(f"pt0001,f_{frame:05d}.jpg,{legacy},1.0" + (native if dual_columns else ""))
    track.write_text("\n".join(lines) + "\n")
    manifest = {
        "schema": "tennis.coordinate-space.v1",
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {"width": 1920, "height": 1080},
    }
    if dual_columns:
        manifest["legacy_artifact_size"] = {"width": 960, "height": 540}
        manifest["coordinate_columns"] = {
            "legacy_960x540": ["x", "y"],
            "native_1920x1080": ["x_native", "y_native"],
        }
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))
    np.savez(
        match_root / "court_H_per_point.npz",
        pts=np.asarray([1]),
        H=np.asarray([homography]),
    )
    return root


def test_a_known_native_pixel_becomes_its_own_court_fraction(tmp_path) -> None:
    """The contract, end to end: project a known native pixel, get the fraction.

    Three court points of known metric position are projected to native pixels
    by a synthetic camera and written to a dual-column track.  The builder must
    report ``court_x``/``court_y`` equal to ``x_m / COURT_WIDTH_M`` and
    ``y_m / COURT_LENGTH_M``.  Reading the legacy columns against the native
    homography -- the defect ``docs/wk1/net_line.md`` traced -- gives a
    different, projectively wrong number for every one of them.
    """

    from cv.pipeline.event_model_v2_features import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        NET_Y_M,
        build_automatic_dataset,
    )

    projection, homography = _synthetic_camera()
    world = [
        (COURT_WIDTH_M / 2.0, NET_Y_M),
        (1.37, 5.5),
        (COURT_WIDTH_M - 1.37, COURT_LENGTH_M - 4.0),
    ]
    rows = []
    for index, (x_m, y_m) in enumerate(world):
        pixel = projection @ np.asarray([x_m, y_m, 0.0, 1.0])
        rows.append((6 + index, float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])))
    root = _write_point_root(tmp_path, homography, rows)

    dataset, manifest = build_automatic_dataset(root)
    index = {name: position for position, name in enumerate(dataset.feature_names)}
    frames = dataset.frames.tolist()

    for (x_m, y_m), (frame, native_x, native_y) in zip(world, rows, strict=True):
        window = dataset.windows[frames.index(frame)]
        court_x = float(window[RADIUS, index["court_x"]])
        court_y = float(window[RADIUS, index["court_y"]])
        assert court_x == pytest.approx(x_m / COURT_WIDTH_M, abs=1e-6)
        assert court_y == pytest.approx(y_m / COURT_LENGTH_M, abs=1e-6)
        # The image channels are the fraction of the frame the pixel sits at.
        assert float(window[RADIUS, index["image_x"]]) == pytest.approx(native_x / 1920.0, abs=1e-6)
        assert float(window[RADIUS, index["image_y"]]) == pytest.approx(native_y / 1080.0, abs=1e-6)
        # What the legacy columns would have given through the same homography.
        defective = _project(
            homography,
            native_x / 2.0,
            native_y / 2.0,
            image_size=resolution.FrameSize(1920, 1080),
            artifact_size=resolution.FrameSize(1920, 1080),
        )
        assert defective[1] / COURT_LENGTH_M != pytest.approx(court_y, abs=1e-3)

    assert manifest["track_pixel_space"]["bc"]["columns"] == ["x_native", "y_native"]
    assert manifest["track_pixel_space"]["bc"]["size"] == {"width": 1920, "height": 1080}


def test_a_legacy_only_root_is_unchanged_by_the_column_map(tmp_path) -> None:
    """A sidecar without a column map still reads x/y at its artifact size."""

    from cv.pipeline.event_model_v2_features import COURT_LENGTH_M, build_automatic_dataset

    _projection, homography = _synthetic_camera()
    rows = [(6, 900.0, 700.0)]
    root = _write_point_root(tmp_path, homography, rows, dual_columns=False)
    # Declare the artifact itself legacy, which is what wk1_s5/cohort_root does.
    track = root / "bc" / TRACK_NAME
    manifest = json.loads(resolution.coordinate_manifest_path(track).read_text())
    manifest["artifact_size"] = {"width": 960, "height": 540}
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))

    dataset, built = build_automatic_dataset(root)
    index = {name: position for position, name in enumerate(dataset.feature_names)}
    expected = _project(
        homography,
        450.0,
        350.0,
        image_size=resolution.FrameSize(1920, 1080),
        artifact_size=resolution.FrameSize(960, 540),
    )
    position = dataset.frames.tolist().index(6)
    assert float(dataset.windows[position, RADIUS, index["court_y"]]) == pytest.approx(
        expected[1] / COURT_LENGTH_M, abs=1e-6
    )
    assert built["track_pixel_space"]["bc"]["space"] == "artifact_size"


def test_a_sidecar_declaring_columns_the_track_lacks_is_refused(tmp_path) -> None:
    from cv.pipeline.event_model_v2_features import _load_track

    _projection, homography = _synthetic_camera()
    root = _write_point_root(tmp_path, homography, [(6, 900.0, 700.0)], dual_columns=False)
    track = root / "bc" / TRACK_NAME
    manifest = json.loads(resolution.coordinate_manifest_path(track).read_text())
    manifest["coordinate_columns"] = {"native_1920x1080": ["x_native", "y_native"]}
    resolution.coordinate_manifest_path(track).write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="x_native"):
        _load_track(track)


def test_feature_court_xy_reproduces_the_builder_channel() -> None:
    from cv.pipeline.event_model_v2_features import feature_court_xy

    _projection, homography = _synthetic_camera()
    image_size = resolution.FrameSize(1920, 1080)
    artifact_size = resolution.FrameSize(1920, 1080)
    base, present = frame_features(
        3,
        {1: (700.0, 520.0)},
        homography,
        image_size=image_size,
        artifact_size=artifact_size,
    )

    court = feature_court_xy(
        homography, [(700.0, 520.0)], image_size=image_size, artifact_size=artifact_size
    )

    assert present[1]
    assert court[0, 0] == pytest.approx(float(base[1, 8]), abs=1e-6)
    assert court[0, 1] == pytest.approx(float(base[1, 9]), abs=1e-6)


def test_the_nets_own_pixels_are_called_net_inside_the_derived_dead_band() -> None:
    """The test the derivation exists for: project the net, ask for its side.

    Every pixel of the net -- its ground line, its tape top and the band between
    them -- must come back as side 0 from ``side_of`` at the derived net line and
    its derived dead band, in the frame the feature builder actually reports,
    including the half-resolution defect ``court_y`` carries.
    """

    from cv.pipeline.event_grammar_decoder import side_of
    from cv.pipeline.event_model_v2_features import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        NET_TAPE_HEIGHT_M,
        NET_Y_M,
        feature_court_xy,
        net_geometry_in_feature_frame,
    )

    projection, homography = _synthetic_camera()
    image_size = resolution.FrameSize(1920, 1080)
    artifact_size = resolution.FrameSize(1920, 1080)
    track_size = resolution.FrameSize(960, 540)
    cord = []
    for x in np.linspace(0.0, COURT_WIDTH_M, 9):
        point = projection @ np.asarray([x, NET_Y_M, NET_TAPE_HEIGHT_M, 1.0])
        cord.append(point[:2] / point[2])

    geometry = net_geometry_in_feature_frame(
        homography,
        image_size=image_size,
        artifact_size=artifact_size,
        track_size=track_size,
        net_cord_pixels=np.asarray(cord),
    )

    assert geometry["source"] == "observed_net_cord"
    assert geometry["net_span"] > 0.0
    for x in np.linspace(0.0, COURT_WIDTH_M, 5):
        for z in np.linspace(0.0, NET_TAPE_HEIGHT_M, 5):
            pixel = projection @ np.asarray([x, NET_Y_M, z, 1.0])
            native = pixel[:2] / pixel[2]
            court_y = feature_court_xy(
                homography,
                resolution.scale_points(native[None, :], image_size, track_size),
                image_size=image_size,
                artifact_size=artifact_size,
            )[0, 1]
            assert side_of(float(court_y), geometry["net_line"], geometry["net_span"]) == 0

    # A ball well inside each half is still called that half.
    for y_m, expected in ((2.0, -1), (COURT_LENGTH_M - 2.0, 1)):
        pixel = projection @ np.asarray([COURT_WIDTH_M / 2.0, y_m, 1.0, 1.0])
        native = pixel[:2] / pixel[2]
        court_y = feature_court_xy(
            homography,
            resolution.scale_points(native[None, :], image_size, track_size),
            image_size=image_size,
            artifact_size=artifact_size,
        )[0, 1]
        assert side_of(float(court_y), geometry["net_line"], geometry["net_span"]) == expected


def test_the_corrected_frame_puts_the_net_ground_line_at_one_half() -> None:
    """The derivation is not a constant: in the corrected frame it returns 0.5.

    The builder now reads the native columns, so ``court_y`` is the fraction of
    the court it is documented to be and the net's ground line lands on
    ``NET_Y_M / COURT_LENGTH_M`` exactly.  The second call keeps the historical
    half-resolution frame for comparison.
    """

    from cv.pipeline.event_model_v2_features import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        NET_TAPE_HEIGHT_M,
        NET_Y_M,
        net_geometry_in_feature_frame,
    )

    projection, homography = _synthetic_camera()
    image_size = resolution.FrameSize(1920, 1080)
    cord = []
    for x in np.linspace(0.0, COURT_WIDTH_M, 9):
        point = projection @ np.asarray([x, NET_Y_M, NET_TAPE_HEIGHT_M, 1.0])
        cord.append(point[:2] / point[2])

    corrected = net_geometry_in_feature_frame(
        homography,
        image_size=image_size,
        artifact_size=image_size,
        track_size=image_size,
        net_cord_pixels=np.asarray(cord),
    )
    defective = net_geometry_in_feature_frame(
        homography,
        image_size=image_size,
        artifact_size=image_size,
        track_size=resolution.FrameSize(960, 540),
        net_cord_pixels=np.asarray(cord),
    )

    assert corrected["base_court_y"][0] == pytest.approx(NET_Y_M / COURT_LENGTH_M, abs=1e-9)
    assert corrected["base_court_y"][1] == pytest.approx(NET_Y_M / COURT_LENGTH_M, abs=1e-9)
    assert defective["net_line"] > corrected["net_line"] > NET_Y_M / COURT_LENGTH_M


def test_net_geometry_falls_back_to_the_camera_matrix_without_a_cord() -> None:
    from cv.pipeline.event_model_v2_features import net_geometry_in_feature_frame

    projection, homography = _synthetic_camera()
    image_size = resolution.FrameSize(1920, 1080)

    geometry = net_geometry_in_feature_frame(
        homography,
        image_size=image_size,
        artifact_size=image_size,
        track_size=resolution.FrameSize(960, 540),
        net_cord_pixels=None,
        camera_projection=projection,
    )

    assert geometry["source"] == "camera_projection_tape_top"
    assert geometry["net_span"] > 0.0
    assert net_geometry_in_feature_frame(
        homography,
        image_size=image_size,
        artifact_size=image_size,
        track_size=image_size,
    ) is None


def test_derive_net_lines_uses_the_frame_the_builder_actually_reports(tmp_path) -> None:
    """``derive_net_lines`` and ``build_automatic_dataset`` share one frame.

    The net's ground line is at ``y = NET_Y_M``, so in the frame the builder
    reports it must land on 0.5 -- and a ball projected at the same court metre
    must land on the same number.
    """

    from cv.pipeline.event_model_v2_features import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        NET_TAPE_HEIGHT_M,
        NET_Y_M,
        build_automatic_dataset,
        derive_net_lines,
    )

    projection, homography = _synthetic_camera()
    pixel = projection @ np.asarray([COURT_WIDTH_M / 2.0, NET_Y_M, 0.0, 1.0])
    native = (float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]))
    root = _write_point_root(tmp_path, homography, [(6, *native)])
    cord = []
    for x in np.linspace(0.0, COURT_WIDTH_M, 9):
        point = projection @ np.asarray([x, NET_Y_M, NET_TAPE_HEIGHT_M, 1.0])
        cord.append(point[:2] / point[2])
    np.savez(
        root / "bc" / "camera_P_per_point.npz",
        pts=np.asarray([1]),
        P=np.asarray([projection]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
    )

    table = derive_net_lines(root)
    geometry = table["bc__pt0001"]
    dataset, _manifest = build_automatic_dataset(root)
    index = {name: position for position, name in enumerate(dataset.feature_names)}
    position = dataset.frames.tolist().index(6)

    assert geometry["base_court_y"][0] == pytest.approx(NET_Y_M / COURT_LENGTH_M, abs=1e-9)
    assert geometry["base_court_y"][1] == pytest.approx(NET_Y_M / COURT_LENGTH_M, abs=1e-9)
    assert float(dataset.windows[position, RADIUS, index["court_y"]]) == pytest.approx(
        NET_Y_M / COURT_LENGTH_M, abs=1e-6
    )
    assert geometry["net_line"] > NET_Y_M / COURT_LENGTH_M
    assert geometry["net_span"] == pytest.approx(
        geometry["net_line"] - NET_Y_M / COURT_LENGTH_M, abs=1e-9
    )
