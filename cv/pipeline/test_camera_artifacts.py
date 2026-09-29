from pathlib import Path

import numpy as np

from cv.pipeline.camera_artifacts import (
    MatchRigPrior,
    WindowCamera,
    compose_frame_projection,
    expand_point_cameras,
    registered_frame_reliable,
    slice_reconstruction_match_artifacts,
    window_camera_rows,
    write_window_camera,
)

PROJECTION = np.asarray(
    [
        [820.0, 30.0, 12.0, 4800.0],
        [10.0, 700.0, -430.0, 3500.0],
        [0.001, 0.09, 0.004, 10.0],
    ]
)


def _write_point_camera(path: Path) -> None:
    np.savez_compressed(
        path,
        pts=np.asarray([1]),
        P=np.asarray([PROJECTION]),
        reliable=np.asarray([True]),
        source=np.asarray(["direct"]),
        ground_residual_px=np.asarray([0.2]),
        net_residual_px=np.asarray([1.5]),
        confidence=np.asarray([0.8]),
        fallback_ancestry=np.asarray(["[]"]),
        frame_scope=np.asarray(["point_static"]),
        reference_frame=np.asarray(["f_0001.jpg"]),
    )


def _ground_homography(projection: np.ndarray) -> np.ndarray:
    return np.linalg.inv(projection[:, [0, 1, 3]])


def _write_frame_track(path: Path, homographies: dict[int, np.ndarray], reliable: list[bool]):
    frames = sorted(homographies)
    np.savez_compressed(
        path,
        clips=np.asarray(["pt0001"] * len(frames)),
        frames=np.asarray(frames, dtype=np.int32),
        H=np.stack([homographies[frame] for frame in frames]),
        reliable=np.asarray(reliable, dtype=bool),
        source=np.asarray(
            ["registered" if accepted else "anchor_static_fallback" for accepted in reliable],
            dtype=str,
        ),
    )


def test_composition_reproduces_the_frame_ground_plane_exactly() -> None:
    warp = np.asarray([[1.04, 0.01, 26.0], [0.002, 0.98, -9.0], [0.0, 2e-5, 1.0]])
    frame_homography = np.linalg.inv(warp @ PROJECTION[:, [0, 1, 3]])

    composed = compose_frame_projection(PROJECTION, frame_homography)

    np.testing.assert_allclose(
        _ground_homography(composed) / _ground_homography(composed)[2, 2],
        frame_homography / frame_homography[2, 2],
        atol=1e-9,
    )


def test_composition_is_the_identity_at_the_anchor_frame() -> None:
    composed = compose_frame_projection(PROJECTION, _ground_homography(PROJECTION))

    np.testing.assert_allclose(composed / composed[2, 3], PROJECTION / PROJECTION[2, 3], atol=1e-9)


def test_composition_rejects_a_degenerate_frame_homography() -> None:
    assert compose_frame_projection(PROJECTION, np.zeros((3, 3))) is None


def test_expand_follows_the_frame_track_and_reports_its_reliability(tmp_path: Path) -> None:
    _write_point_camera(tmp_path / "camera_P_per_point.npz")
    frames = tmp_path / "frames" / "pt0001"
    frames.mkdir(parents=True)
    for frame in (1, 2):
        (frames / f"f_{frame:04d}.jpg").write_bytes(b"frame identity only")
    warp = np.asarray([[1.02, 0.0, 14.0], [0.0, 0.99, -6.0], [0.0, 1e-5, 1.0]])
    _write_frame_track(
        tmp_path / "court_H_per_frame_v1.npz",
        {
            1: _ground_homography(PROJECTION),
            2: np.linalg.inv(warp @ PROJECTION[:, [0, 1, 3]]),
        },
        [True, False],
    )

    with np.load(expand_point_cameras(tmp_path, "frames")) as output:
        assert output["frame_scope"].tolist() == ["frame_track", "frame_track"]
        assert output["reliable"].tolist() == [True, False]
        assert [str(value) for value in output["source"]] == [
            "direct+registered",
            "direct+anchor_static_fallback",
        ]
        np.testing.assert_allclose(
            output["P"][0] / output["P"][0][2, 3], PROJECTION / PROJECTION[2, 3], atol=1e-9
        )
        assert not np.allclose(output["P"][1], output["P"][0])


def test_registered_fallback_does_not_accept_an_unwitnessed_static_anchor() -> None:
    common = {
        "point_reliable": True,
        "point_source": "direct",
        "registration_reliable": False,
        "registration_source": "anchor_static_fallback",
        "ground_residual_px": 0.2,
        "net_residual_px": 1.5,
    }

    assert not registered_frame_reliable(**common)
    assert not registered_frame_reliable(**{**common, "point_source": "intrinsic_fallback"})
    assert not registered_frame_reliable(**{**common, "net_residual_px": 8.01})


def test_rig_prior_recovers_only_a_registered_unreliable_point(tmp_path: Path, monkeypatch) -> None:
    np.savez_compressed(
        tmp_path / "camera_P_per_point.npz",
        pts=np.asarray([1]),
        P=np.asarray([PROJECTION]),
        reliable=np.asarray([False]),
        source=np.asarray(["shared_net_fallback"]),
        ground_residual_px=np.asarray([0.2]),
        net_residual_px=np.asarray([float("nan")]),
        confidence=np.asarray([0.25]),
        fallback_ancestry=np.asarray(['["shared_vertical_column"]']),
        frame_scope=np.asarray(["point_static"]),
        reference_frame=np.asarray(["f_0001.jpg"]),
    )
    frames = tmp_path / "frames" / "pt0001"
    frames.mkdir(parents=True)
    for frame in (1, 2):
        (frames / f"f_{frame:04d}.jpg").write_bytes(b"frame identity only")
    _write_frame_track(
        tmp_path / "court_H_per_frame_v1.npz",
        {1: _ground_homography(PROJECTION), 2: _ground_homography(PROJECTION)},
        [True, False],
    )
    prior = MatchRigPrior(None, (1920, 1080), 2.0, (2, 3))
    monkeypatch.setattr("cv.pipeline.camera_artifacts.fit_match_rig_prior", lambda *_args: prior)
    monkeypatch.setattr(
        "cv.pipeline.camera_artifacts.rig_prior_projection",
        lambda *_args: (PROJECTION * 1.01, 0.4),
    )

    with np.load(expand_point_cameras(tmp_path, "frames")) as output:
        assert output["reliable"].tolist() == [True, False]
        assert output["source"].tolist() == [
            "rig_prior+registered",
            "shared_net_fallback+anchor_static_fallback",
        ]
        assert output["frame_scope"].tolist() == ["rig_prior_fallback", "frame_track"]
        assert output["fallback_ancestry"].tolist() == [
            "[]",
            '["shared_vertical_column"]',
        ]
        assert output["net_residual_px"].tolist()[0] == 2.0
    report = __import__("json").loads((tmp_path / "camera_rig_fallback_v1.json").read_text())
    assert report["recovered_points"] == [1]


def test_expand_without_a_frame_track_stays_point_static(tmp_path: Path) -> None:
    _write_point_camera(tmp_path / "camera_P_per_point.npz")
    frames = tmp_path / "frames" / "pt0001"
    frames.mkdir(parents=True)
    (frames / "f_0001.jpg").write_bytes(b"frame identity only")

    with np.load(expand_point_cameras(tmp_path, "frames")) as output:
        assert output["frame_scope"].tolist() == ["point_static"]
        assert output["reliable"].tolist() == [True]
        np.testing.assert_allclose(output["P"][0], PROJECTION)


def test_the_observed_net_cord_travels_with_the_camera(tmp_path: Path) -> None:
    cord = np.stack([np.linspace(300.0, 1300.0, 9), np.linspace(510.0, 515.0, 9)], axis=1)
    np.savez_compressed(
        tmp_path / "camera_P_per_point.npz",
        pts=np.asarray([1]),
        P=np.asarray([PROJECTION]),
        reliable=np.asarray([True]),
        source=np.asarray(["direct"]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
        net_cord_source=np.asarray(["observed"]),
    )
    frames = tmp_path / "frames" / "pt0001"
    frames.mkdir(parents=True)
    for frame in (1, 2):
        (frames / f"f_{frame:04d}.jpg").write_bytes(b"frame identity only")
    warp = np.asarray([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _write_frame_track(
        tmp_path / "court_H_per_frame_v1.npz",
        {
            1: _ground_homography(PROJECTION),
            2: np.linalg.inv(warp @ PROJECTION[:, [0, 1, 3]]),
        },
        [True, True],
    )

    with np.load(expand_point_cameras(tmp_path, "frames")) as output:
        np.testing.assert_allclose(output["net_cord_xy"][0], cord, atol=1e-9)
        np.testing.assert_allclose(
            output["net_cord_xy"][1], cord + np.asarray([40.0, 0.0]), atol=1e-6
        )


def test_reconstruction_artifacts_are_sliced_once_into_point_local_schema(tmp_path: Path) -> None:
    match_dir = tmp_path / "source"
    match_dir.mkdir()
    clips = np.asarray(["pt0001", "pt0001", "pt0002"])
    np.savez_compressed(
        match_dir / "camera_P_per_frame_v1.npz",
        clips=clips,
        frames=np.asarray([1, 2, 1]),
        P=np.stack([PROJECTION, PROJECTION, PROJECTION]),
        reliable=np.ones(3, dtype=bool),
        # Reproduce the match-wide dtype inflation caused by one exceptional row.
        fallback_ancestry=np.asarray(["[]", "[]", "x" * 2000]),
    )
    np.savez_compressed(
        match_dir / "court_H_per_point.npz",
        pts=np.asarray([1, 2]),
        H=np.stack([_ground_homography(PROJECTION)] * 2),
    )
    np.savez_compressed(
        match_dir / "court_H_per_frame_v1.npz",
        clips=clips,
        frames=np.asarray([1, 2, 1]),
        H=np.stack([_ground_homography(PROJECTION)] * 3),
    )
    (match_dir / "ball_track_joint_native1080_arc_augmented_v2.csv").write_text(
        "clip,frame,x,y,score,sources\n"
        "pt0001,f_0001.jpg,1,2,0.9,crop\n"
        "pt0002,f_0001.jpg,3,4,0.8,full\n"
    )
    (match_dir / "player_boxes_25_native_sided_v1.csv").write_text(
        "clip,frame,side,court_x,court_y\npt0001,f_0001.jpg,near,1,2\npt0002,f_0001.jpg,far,3,4\n"
    )
    (match_dir / "audit_frames_native_1080.coordinates.json").write_text('{"fps":25}\n')
    # A sliced pixel artifact must carry its coordinate sidecar or reconstruct_3d
    # fails closed (the first full-match yield run aborted on exactly this).
    (match_dir / "ball_track_joint_native1080_arc_augmented_v2.csv.coordinates.json").write_text(
        '{"schema":"tennis.coordinate-space.v1"}\n'
    )
    destinations = {
        "pt0001": tmp_path / "points" / "pt0001" / "match",
        "pt0002": tmp_path / "points" / "pt0002" / "match",
    }

    report = slice_reconstruction_match_artifacts(match_dir, destinations)

    assert report["artifacts"]["camera_P_per_frame_v1.npz"] == {
        "pt0001": 2,
        "pt0002": 1,
    }
    with np.load(destinations["pt0001"] / "camera_P_per_frame_v1.npz") as point:
        assert point["clips"].tolist() == ["pt0001", "pt0001"]
        assert point["fallback_ancestry"].dtype.itemsize < 100
    with np.load(destinations["pt0002"] / "court_H_per_point.npz") as point:
        assert point["pts"].tolist() == [2]
    assert (
        "pt0002"
        not in (
            destinations["pt0001"] / "ball_track_joint_native1080_arc_augmented_v2.csv"
        ).read_text()
    )
    assert (destinations["pt0002"] / "audit_frames_native_1080.coordinates.json").is_file()
    for point in destinations.values():
        assert (
            point / "ball_track_joint_native1080_arc_augmented_v2.csv.coordinates.json"
        ).is_file()


def _window_camera(reliable: tuple[bool, ...]) -> WindowCamera:
    frames = tuple(range(40, 40 + len(reliable)))
    return WindowCamera(
        frames=frames,
        projections=np.stack([PROJECTION] * len(frames)),
        homographies=np.stack([_ground_homography(PROJECTION)] * len(frames)),
        reliable=reliable,
        source=tuple(
            "direct+registered" if value else "direct+anchor_static_fallback" for value in reliable
        ),
        anchor={"frame": frames[0], "image": "f_0040.jpg", "source": "standard"},
        point={
            "source": "direct",
            "reliable": True,
            "ground_residual_px": 0.1,
            "net_residual_px": 2.0,
            "focal_native_px": 2000.0,
            "net_observation_source": "observed_connected_tape_top_envelope",
            "net_observations": 9,
            "fallback_ancestry": [],
            "net_cord_xy": [[float(index), 400.0] for index in range(9)],
        },
        registration={"stride": 5, "samples": {}},
    )


def test_window_rows_hold_every_frame_without_a_supported_registration() -> None:
    rows = window_camera_rows(_window_camera((True, False, True)))
    assert [row["status"] for row in rows] == ["supported", "held", "supported"]
    assert "P" not in rows[1] and rows[1]["reason"] == "direct+anchor_static_fallback"
    assert np.allclose(np.asarray(rows[0]["P"]), PROJECTION)


def test_window_camera_is_written_as_automatic_not_as_agent_evidence(tmp_path: Path) -> None:
    camera = _window_camera((True, True, False))
    written = write_window_camera(
        tmp_path,
        "match",
        "pt0001",
        camera,
        inputs=[{"path": "f_0040.jpg", "sha256": "0" * 64}],
        configuration={"registration_stride": 5},
    )
    assert written["human_derived"] is False
    assert written["annotation_origin"] == "automatic"
    assert (written["supported"], written["total"]) == (2, 3)
    with np.load(tmp_path / "camera_P_per_frame_v1.npz", allow_pickle=True) as artifact:
        assert artifact["P"].shape == (3, 3, 4)
        assert list(artifact["frames"]) == [40, 41, 42]
        assert list(artifact["reliable"]) == [True, True, False]
        assert artifact["net_cord_valid"].all()
        assert artifact["calibration_point"][0] == 1
    with np.load(tmp_path / "court_H_per_frame_v1.npz") as artifact:
        assert artifact["H"].shape == (3, 3, 3)
        assert list(artifact["clips"]) == ["pt0001"] * 3
