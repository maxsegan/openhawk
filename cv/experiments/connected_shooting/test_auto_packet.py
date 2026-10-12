import csv
import json

import numpy as np
import pytest

from cv.experiments.connected_shooting import auto_packet
from cv.pipeline import resolution as res


def _labels(tmp_path, *, clip="pt0001", window=(10, 20)):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "attempt-1",
                "match_id": "match-1",
                "annotation_origin": "agent",
                "attempt": {
                    "clip": clip,
                    "native_window": list(window),
                    "ending_kind": "ground_after_failed_return",
                },
                "source_pack": {"fps": 25.0, "images": []},
                "ball": {
                    "records": [
                        {
                            "clip": clip,
                            "frames": [
                                {
                                    "frame": frame,
                                    "status": "visible",
                                    "x1080": 900.0 + frame,
                                    "y1080": 300.0 + frame,
                                    "uncertainty_radius_px1080": 4,
                                }
                                for frame in range(window[0], window[1] + 1)
                            ],
                        }
                    ]
                },
                "events": {
                    "records": [
                        {
                            "id": "event_01",
                            "status": "labeled",
                            "event_type": "contact",
                            "frame": 10.5,
                            "frame_interval": [10.0, 11.0],
                        },
                        {
                            "id": "event_02",
                            "status": "labeled",
                            "event_type": "bounce",
                            "frame": 19.5,
                            "frame_interval": [19.0, 20.0],
                        },
                        {
                            "id": "event_03",
                            "status": "labeled",
                            "event_type": "ending",
                            "frame": 19.5,
                            "frame_interval": [19.0, 20.0],
                        },
                    ]
                },
            }
        )
    )
    return path


def _ball_csv(tmp_path, *, native=True, clip="pt0001"):
    path = tmp_path / "ball.csv"
    columns = ("x_native", "y_native") if native else ("x", "y")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("clip", "frame", *columns, "track_id", "sources", "confidence")
        )
        writer.writeheader()
        for frame in range(10, 21):
            x, y = (900.0 + frame, 300.0 + frame)
            if not native:
                x, y = x / 2, y / 2
            writer.writerow(
                {
                    "clip": clip,
                    "frame": f"f_{frame:04d}.jpg",
                    columns[0]: x,
                    columns[1]: y,
                    "track_id": 0,
                    "sources": "automatic-test",
                    "confidence": 0.9,
                }
            )
    manifest = {
        "schema": "tennis.coordinate-space.v1",
        "artifact": path.name,
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {"width": 1920, "height": 1080}
        if native
        else {"width": 960, "height": 540},
        "source": "unit-test",
    }
    if native:
        manifest["interface_space"] = "native_1920x1080"
        manifest["legacy_artifact_size"] = {"width": 960, "height": 540}
        manifest["coordinate_columns"] = {
            "native_1920x1080": ["x_native", "y_native"],
            "legacy_960x540": ["x", "y"],
        }
    (tmp_path / f"{path.name}.coordinates.json").write_text(json.dumps(manifest))
    return path


def _events_json(tmp_path, *, subpixel=True):
    path = tmp_path / "events.json"
    rows = []
    for kind, frame, fine in (("contact", 10, 10.4), ("bounce", 20, 19.6), ("point_end", 20, 19.6)):
        row = {
            "match_id": "match-1",
            "clip": "match-1__pt0001",
            "event_type": kind,
            "frame": frame,
            "abstain": False,
            "confidence": 0.99,
        }
        if subpixel:
            row["location"] = {"frame_subpixel": fine}
        if kind == "point_end":
            row["point_end"] = {
                "terminal_event_type": "bounce",
                "termination_kind": "second_ground",
            }
        rows.append(row)
    path.write_text(json.dumps(rows))
    return path


def _camera_npz(tmp_path, *, reliable_from=10):
    path = tmp_path / "camera.npz"
    projection = np.array(
        [[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]]
    )
    frames = np.arange(10, 21)
    np.savez(
        path,
        clips=np.array(["pt0001"] * len(frames)),
        frames=frames,
        P=np.repeat(projection[None], len(frames), axis=0),
        reliable=frames >= reliable_from,
        source=np.array(["registered"] * len(frames)),
        confidence=np.ones(len(frames)),
        frame_scope=np.array(["per_frame"] * len(frames)),
    )
    return path


def _players_csv(tmp_path, *, legacy=True):
    path = tmp_path / "players.csv"
    columns = ("x0", "y0", "x1", "y1") if legacy else auto_packet.res.PLAYER_NATIVE_BOX_COLUMNS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip", "frame", "side", *columns))
        writer.writeheader()
        writer.writerow(
            dict(
                zip(
                    ("clip", "frame", "side", *columns),
                    ("pt0001", "f_0010.jpg", "near", 400.0, 300.0, 450.0, 400.0),
                )
            )
        )
    (tmp_path / f"{path.name}.coordinates.json").write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "artifact": path.name,
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 960, "height": 540}
                if legacy
                else {"width": 1920, "height": 1080},
                "source": "unit-test",
            }
        )
    )
    return path


def _build(tmp_path, output, **origins):
    return auto_packet.build_packet(
        labels_path=_labels(tmp_path),
        output=output,
        automatic_ball_path=_ball_csv(tmp_path),
        automatic_events_path=_events_json(tmp_path),
        label_player_path=_players_csv(tmp_path),
        automatic_player_path=_players_csv(tmp_path),
        automatic_camera_path=_camera_npz(tmp_path),
        label_camera_path=tmp_path / "unused_label_cameras.json",
        **origins,
    )


def test_automatic_packet_keeps_centre_semantics_and_per_stream_hashes(tmp_path):
    output = tmp_path / "packet"
    manifest = _build(
        tmp_path,
        output,
        ball_origin="automatic",
        event_origin="automatic",
        player_origin="automatic",
        camera_origin="automatic",
    )
    packet = json.loads((output / "packet.json").read_text())
    attempt = packet["attempts"][0]
    observation = attempt["ball_observations"][0]
    assert manifest["automatic_rows_rewritten_as_agent_labels"] is False
    assert manifest["stream_origins"] == dict.fromkeys(auto_packet.STREAMS, "automatic")
    assert packet["human_derived"] is False
    assert observation["annotation_origin"] == "automatic"
    assert observation["observation_semantics"] == "detector_heatmap_nominal_centre"
    assert observation["coordinate_columns_read"] == ["x_native", "y_native"]
    # The legacy field name is an alias for the same rows, never a human-origin claim.
    assert attempt["owner_ball_labels"] == attempt["ball_observations"]
    assert attempt["ball_observation_compatibility_alias"]["does_not_assert_human_annotation"]
    assert attempt["labeled_native_frames"] == 0
    for stream in auto_packet.STREAMS:
        assert packet["stream_manifest"][stream]["inputs"][0]["sha256"]


def test_mixed_packet_names_each_stream_origin_separately(tmp_path):
    output = tmp_path / "packet"
    manifest = _build(
        tmp_path,
        output,
        ball_origin="label",
        event_origin="automatic",
        player_origin="label",
        camera_origin="automatic",
    )
    packet = json.loads((output / "packet.json").read_text())
    streams = packet["stream_manifest"]
    assert manifest["stream_origins"] == {
        "ball": "label",
        "events": "automatic",
        "players": "label",
        "camera": "automatic",
    }
    assert streams["ball"]["semantics"] == "visible_blur_leading_edge_in_travel_direction"
    assert packet["attempts"][0]["ball_observations"][0]["annotation_origin"] == "agent"
    assert packet["attempts"][0]["events"][0]["annotation_origin"] == "automatic"
    assert packet["human_derived"] is True


def test_sidecar_resolves_a_half_native_track_instead_of_trusting_the_file_name(tmp_path):
    legacy = _ball_csv(tmp_path, native=False)
    rows = auto_packet.automatic_ball_rows(legacy, "pt0001")
    assert rows[0]["coordinate_columns_read"] == ["x", "y"]
    assert rows[0]["x1080"] == pytest.approx(910.0)
    assert rows[0]["declared_source_space"] == "960x540"


def test_missing_coordinate_sidecar_fails_closed(tmp_path):
    path = tmp_path / "ball.csv"
    path.write_text("clip,frame,x,y\npt0001,f_0010.jpg,1.0,2.0\n")
    with pytest.raises(res.MissingCoordinateContract):
        auto_packet.automatic_ball_rows(path, "pt0001")


def test_automatic_event_epoch_prefers_the_decoder_subframe_value():
    fine = {"event_type": "contact", "frame": 64, "location": {"frame_subpixel": 64.443}}
    coarse = {"event_type": "contact", "frame": 64, "location": {}}
    assert auto_packet.automatic_event_epoch(fine) == pytest.approx(64.443)
    assert auto_packet.automatic_event_epoch(coarse) == 64.0


def test_automatic_events_without_an_accepted_ending_block_the_packet(tmp_path):
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            [
                {
                    "match_id": "match-1",
                    "clip": "match-1__pt0001",
                    "event_type": "contact",
                    "frame": 10,
                    "abstain": False,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="0 accepted point_end"):
        auto_packet.automatic_events(events, "match-1", "pt0001", (10, 20))


def test_camera_abstention_removes_observations_and_is_counted(tmp_path):
    output = tmp_path / "packet"
    manifest = auto_packet.build_packet(
        labels_path=_labels(tmp_path),
        output=output,
        ball_origin="automatic",
        event_origin="automatic",
        player_origin="automatic",
        camera_origin="automatic",
        automatic_ball_path=_ball_csv(tmp_path),
        automatic_events_path=_events_json(tmp_path),
        label_player_path=_players_csv(tmp_path),
        automatic_player_path=_players_csv(tmp_path),
        automatic_camera_path=_camera_npz(tmp_path, reliable_from=15),
        label_camera_path=tmp_path / "unused_label_cameras.json",
    )
    ball = manifest["streams"]["ball"]
    assert ball["observations_removed_for_camera_abstention"] > 0
    assert manifest["packet_observation_count"] == ball["selected_observations"]


def _camera_with_lens(tmp_path, lens, *, reliable_from=10):
    path = _camera_npz(tmp_path, reliable_from=reliable_from)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    np.savez(path, **arrays, **lens)
    return path


def test_camera_missing_diagnostics_serialize_without_changing_projection_support(tmp_path):
    path = _camera_npz(tmp_path, reliable_from=13)
    labels = json.loads(_labels(tmp_path).read_text())
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["ground_residual_px"] = np.array([np.inf, *([2.5] * 10)])
    arrays["net_residual_px"] = np.array([*([np.nan] * 10), 1.75])
    np.savez(path, **arrays)
    source_bytes = path.read_bytes()
    document = auto_packet.automatic_camera_document(path, labels, (10, 20))
    json.dumps(document, allow_nan=False)
    assert path.read_bytes() == source_bytes
    assert document["supported"] == 8
    for index, row in enumerate(document["cameras"]):
        assert row["net_residual_px"] == (1.75 if index == 10 else None)
        assert row["ground_residual_px"] == (None if index == 0 else 2.5)
        if index < 3:
            assert row["P"] is None and row["supported"] is False
        else:
            assert row["P"] == arrays["P"][index].tolist()
            assert row["supported"] is True


def test_automatic_camera_preserves_nonzero_native_radial_projection(tmp_path):
    from cv.experiments.connected_shooting import camera_geometry

    labels = json.loads(_labels(tmp_path).read_text())
    coefficients = np.linspace(1e-8, 3e-8, 11)
    centers = np.column_stack([np.linspace(950, 960, 11), np.linspace(530, 540, 11)])
    path = _camera_with_lens(tmp_path, {"k1": coefficients, "dist_center": centers})
    document = auto_packet.automatic_camera_document(path, labels, (12, 18))
    lenses = camera_geometry.camera_radial_map(document)
    assert set(lenses) == set(range(12, 19))
    cameras = np.asarray([row["P"] for row in document["cameras"]])
    xyz = np.repeat([[2.0, 4.0, 1.5]], len(cameras), axis=0)
    pinhole = camera_geometry.project(cameras, xyz)
    delta = pinhole - centers[2:9]
    expected = centers[2:9] + delta * (
        1 + coefficients[2:9, None] * np.sum(delta**2, axis=1, keepdims=True)
    )
    actual = camera_geometry.project(cameras, xyz, np.asarray(list(lenses.values())))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert np.max(np.abs(actual - pinhole)) > 1.0
    for row in document["cameras"]:
        index = row["frame"] - 10
        assert row["k1"] == coefficients[index]
        assert row["dist_center"] == centers[index].tolist()


def test_automatic_radial_camera_keeps_held_rows_out_of_shared_scene(tmp_path):
    from cv.experiments.connected_shooting import camera_geometry

    labels = json.loads(_labels(tmp_path).read_text())
    lens = {"k1": np.full(11, 1e-8), "dist_center": np.tile([960.0, 540.0], (11, 1))}
    path = _camera_with_lens(tmp_path, lens, reliable_from=13)
    document = auto_packet.automatic_camera_document(path, labels, (10, 20))
    assert document["supported"] == 8
    for row in document["cameras"][:3]:
        assert row["status"] == "held" and row["supported"] is False
        assert row["P"] is None
        assert row["k1"] == 1e-8  # Calibration retained; projection still unsupported.
    assert set(camera_geometry.camera_radial_map(document)) == set(range(13, 21))
    attempt = {
        "match_id": "match-1",
        "point_clip": "pt0001",
        "fps": 25.0,
        "owner_end_frame": 20.0,
        "events": [
            {"event_type": "contact", "frame": 10.0},
            {"event_type": "bounce", "frame": 18.0},
        ],
        "owner_ball_labels": labels["ball"]["records"][0]["frames"],
    }
    receipts = []
    scene, heldout, *_ = auto_packet.search.prepare_attempt(
        attempt, document, "hard", observation_fallback=True, fallback_receipt=receipts
    )
    assert receipts[0]["frames"] == [10, 11, 12]
    for split in (scene, heldout):
        assert np.min(split.observation_frames[0]) >= 13
        np.testing.assert_array_equal(
            split.camera_distortion[0],
            np.tile([1e-8, 960.0, 540.0], (len(split.observation_frames[0]), 1)),
        )


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_automatic_camera_retains_pinhole_projection(tmp_path, explicit_zero):
    from cv.experiments.connected_shooting import camera_geometry

    labels = json.loads(_labels(tmp_path).read_text())
    metadata = (
        {"k1": np.zeros(11), "dist_center": np.tile([960.0, 540.0], (11, 1))}
        if explicit_zero
        else {}
    )
    path = _camera_with_lens(tmp_path, metadata)
    document = auto_packet.automatic_camera_document(path, labels, (10, 20))
    lenses = camera_geometry.camera_radial_map(document)
    if not explicit_zero:
        assert lenses is None
    cameras = np.asarray([row["P"] for row in document["cameras"]])
    xyz = np.repeat([[2.0, 4.0, 1.5]], len(cameras), axis=0)
    np.testing.assert_array_equal(
        camera_geometry.project(cameras, xyz),
        camera_geometry.project(
            cameras, xyz, None if lenses is None else np.asarray(list(lenses.values()))
        ),
    )


@pytest.mark.parametrize("include_canonical_fields", [False, True])
def test_automatic_camera_normalizes_consistent_explicit_lens_row(
    tmp_path, include_canonical_fields
):
    from cv.experiments.connected_shooting import camera_geometry

    labels = json.loads(_labels(tmp_path).read_text())
    lenses = np.tile([1e-8, 960.0, 540.0], (11, 1))
    metadata = {"camera_distortion": lenses, "k2": np.zeros(11)}
    if include_canonical_fields:
        metadata.update(k1=lenses[:, 0], dist_center=lenses[:, 1:])
    path = _camera_with_lens(tmp_path, metadata)
    document = auto_packet.automatic_camera_document(path, labels, (10, 20))
    np.testing.assert_array_equal(camera_geometry.rows_radial(document["cameras"]), lenses)
    assert all("k1" in row and "dist_center" in row for row in document["cameras"])


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"k1": np.zeros(11)}, "both k1 and dist_center"),
        ({"dist_center": np.zeros((11, 2))}, "both k1 and dist_center"),
        ({"k1": np.zeros(11), "dist_center": np.zeros(2)}, "native-frame shape"),
        ({"k1": np.zeros(10), "dist_center": np.zeros((11, 2))}, "native-frame shape"),
        ({"k1": np.full(11, np.nan), "dist_center": np.zeros((11, 2))}, "finite native"),
        ({"k1": np.zeros(11), "dist_center": np.full((11, 2), np.inf)}, "finite native"),
        (
            {
                "k1": np.zeros(11),
                "dist_center": np.zeros((11, 2)),
                "camera_distortion": np.ones((11, 3)),
            },
            "conflicting native",
        ),
        ({"distortion": np.zeros((11, 3))}, "ambiguous radial"),
        ({"radial_distortion": np.zeros((11, 3))}, "ambiguous radial"),
        ({"k2": np.full(11, 1e-12)}, "does not support"),
    ],
)
def test_automatic_camera_refuses_invalid_or_unsupported_lens(tmp_path, metadata, message):
    labels = json.loads(_labels(tmp_path).read_text())
    # An unreliable image does not authorize discarding its malformed lens identity.
    path = _camera_with_lens(tmp_path, metadata, reliable_from=13)
    with pytest.raises(ValueError, match=message):
        auto_packet.automatic_camera_document(path, labels, (10, 20))


def _intrinsic_fallback_npz(tmp_path):
    path = tmp_path / "intrinsic.npz"
    projection = np.array(
        [[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]]
    )
    bad = np.full((3, 4), np.nan)
    frames = np.arange(10, 17)
    sources = np.array(
        [
            "intrinsic_fallback+registered",
            "intrinsic_fallback+registered_interpolated",
            "intrinsic_fallback+line_model_registered",
            "intrinsic_fallback+anchor_static_fallback",
            "intrinsic_fallback+registration_gap_abstention",
            "direct+registered",
            "direct+anchor_static_fallback",
        ]
    )
    projections = np.repeat(projection[None], len(frames), axis=0)
    projections[1] = bad
    ground = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 9.0])
    np.savez(
        path,
        clips=np.array(["pt0001"] * len(frames)),
        frames=frames,
        P=projections,
        reliable=np.array([False, False, False, False, False, True, False]),
        source=sources,
        confidence=np.full(len(frames), 0.25),
        frame_scope=np.array(["frame_track"] * len(frames)),
        ground_residual_px=ground,
        net_residual_px=np.full(len(frames), 54.0),
    )
    return path


def test_intrinsic_fallback_registration_is_admit_by_default(tmp_path):
    """Production packet builders admit a tracked intrinsic fallback without naming it.

    Same shape as ``test_keep_unique_admissible_frames_is_on_by_default``: the
    caller does not pass the mode, the document records admit, and the tracked
    frames are kept. Explicit ``off`` is the rollback and omits the fields.
    """
    labels = json.loads(_labels(tmp_path, window=(10, 16)).read_text())
    path = _intrinsic_fallback_npz(tmp_path)
    document = auto_packet.automatic_camera_document(path, labels, (10, 16))
    by_source = {row["source"]: row for row in document["cameras"]}
    assert document["intrinsic_fallback_registration"] == "admit"
    assert document["admitted_intrinsic_fallback_frames"] == 2
    assert document["supported"] == 3
    for source in (
        "intrinsic_fallback+registered",
        "intrinsic_fallback+line_model_registered",
        "direct+registered",
    ):
        assert by_source[source]["status"] == "supported"
        assert by_source[source]["P"] is not None
    for source in (
        "intrinsic_fallback+registered_interpolated",
        "intrinsic_fallback+anchor_static_fallback",
        "intrinsic_fallback+registration_gap_abstention",
        "direct+anchor_static_fallback",
    ):
        assert by_source[source]["status"] == "held"
        assert by_source[source]["P"] is None


def test_intrinsic_fallback_registration_off_is_the_rollback(tmp_path):
    labels = json.loads(_labels(tmp_path, window=(10, 16)).read_text())
    path = _intrinsic_fallback_npz(tmp_path)
    document = auto_packet.automatic_camera_document(
        path, labels, (10, 16), intrinsic_fallback_registration="off"
    )
    assert "intrinsic_fallback_registration" not in document
    by_source = {row["source"]: row for row in document["cameras"]}
    assert document["supported"] == 1
    assert by_source["direct+registered"]["status"] == "supported"
    assert by_source["direct+registered"]["P"] is not None
    for source in (
        "intrinsic_fallback+registered",
        "intrinsic_fallback+registered_interpolated",
        "intrinsic_fallback+line_model_registered",
        "intrinsic_fallback+anchor_static_fallback",
        "intrinsic_fallback+registration_gap_abstention",
        "direct+anchor_static_fallback",
    ):
        assert by_source[source]["status"] == "held"
        assert by_source[source]["P"] is None
        assert by_source[source]["reason"] == "automatic_camera_unreliable_or_nonfinite"


def test_admit_keeps_tracked_intrinsic_fallback_and_refuses_the_rest(tmp_path):
    labels = json.loads(_labels(tmp_path, window=(10, 16)).read_text())
    path = _intrinsic_fallback_npz(tmp_path)
    document = auto_packet.automatic_camera_document(
        path,
        labels,
        (10, 16),
        intrinsic_fallback_registration="admit",
    )
    by_source = {row["source"]: row for row in document["cameras"]}
    assert document["intrinsic_fallback_registration"] == "admit"
    assert document["admitted_intrinsic_fallback_frames"] == 2
    for source in (
        "intrinsic_fallback+registered",
        "intrinsic_fallback+line_model_registered",
        "direct+registered",
    ):
        assert by_source[source]["status"] == "supported"
        assert by_source[source]["P"] is not None
        assert by_source[source]["reason"] is None
    # Nonfinite projection, static fallback, registration gap, a large ground
    # residual, and a non-direct static frame stay held.
    for source in (
        "intrinsic_fallback+registered_interpolated",
        "intrinsic_fallback+anchor_static_fallback",
        "intrinsic_fallback+registration_gap_abstention",
        "direct+anchor_static_fallback",
    ):
        assert by_source[source]["status"] == "held"
        assert by_source[source]["P"] is None
    with pytest.raises(ValueError, match="intrinsic-fallback registration"):
        auto_packet.automatic_camera_document(
            path, labels, (10, 16), intrinsic_fallback_registration="on"
        )


def _shared_net_fallback_npz(tmp_path):
    path = tmp_path / "shared_net.npz"
    projection = np.array(
        [[1000.0, 0.0, 960.0, 0.0], [0.0, 1000.0, 540.0, 0.0], [0.0, 0.0, 1.0, 10.0]]
    )
    frames = np.arange(10, 14)
    sources = np.array(
        [
            "shared_net_focal_fallback+registered",
            "shared_net_focal_fallback+anchor_static_fallback",
            "shared_net_focal_fallback+registration_gap_abstention",
            "direct+registered",
        ]
    )
    np.savez(
        path,
        clips=np.array(["pt0001"] * len(frames)),
        frames=frames,
        P=np.repeat(projection[None], len(frames), axis=0),
        reliable=np.array([False, False, False, True]),
        source=sources,
        confidence=np.full(len(frames), 0.25),
        frame_scope=np.array(["frame_track"] * len(frames)),
        ground_residual_px=np.array([0.01, 0.01, 0.01, 0.01]),
        net_residual_px=np.full(len(frames), 54.0),
    )
    return path


def test_shot_homography_propagation_is_off_by_default(tmp_path):
    labels = json.loads(_labels(tmp_path, window=(10, 13)).read_text())
    document = auto_packet.automatic_camera_document(
        _shared_net_fallback_npz(tmp_path), labels, (10, 13)
    )
    assert "shot_homography_propagation" not in document
    by_source = {row["source"]: row for row in document["cameras"]}
    assert by_source["direct+registered"]["status"] == "supported"
    for source in (
        "shared_net_focal_fallback+registered",
        "shared_net_focal_fallback+anchor_static_fallback",
        "shared_net_focal_fallback+registration_gap_abstention",
    ):
        assert by_source[source]["status"] == "held"
        assert by_source[source]["P"] is None


def test_shot_homography_propagation_admits_fallback_static_and_tracked(tmp_path):
    labels = json.loads(_labels(tmp_path, window=(10, 13)).read_text())
    document = auto_packet.automatic_camera_document(
        _shared_net_fallback_npz(tmp_path),
        labels,
        (10, 13),
        shot_homography_propagation="on",
        shot_ids={10: 0, 11: 0, 12: 0, 13: 0},
    )
    assert document["shot_homography_propagation"] == "on"
    assert document["admitted_shot_homography_frames"] == 2
    by_source = {row["source"]: row for row in document["cameras"]}
    for source in (
        "shared_net_focal_fallback+registered",
        "shared_net_focal_fallback+anchor_static_fallback",
        "direct+registered",
    ):
        assert by_source[source]["status"] == "supported"
        assert by_source[source]["P"] is not None
    assert by_source["shared_net_focal_fallback+registration_gap_abstention"]["status"] == "held"
    other_shot = auto_packet.automatic_camera_document(
        _shared_net_fallback_npz(tmp_path),
        labels,
        (10, 13),
        shot_homography_propagation="on",
        shot_ids={10: 0, 11: 1, 12: 1, 13: 0},
    )
    other = {row["source"]: row for row in other_shot["cameras"]}
    assert other["shared_net_focal_fallback+registered"]["status"] == "supported"
    assert other["shared_net_focal_fallback+anchor_static_fallback"]["status"] == "held"
    assert other["shared_net_focal_fallback+anchor_static_fallback"]["P"] is None
    unsegmented = auto_packet.automatic_camera_document(
        _shared_net_fallback_npz(tmp_path),
        labels,
        (10, 13),
        shot_homography_propagation="on",
    )
    plain = {row["source"]: row for row in unsegmented["cameras"]}
    assert plain["shared_net_focal_fallback+anchor_static_fallback"]["status"] == "held"
    assert plain["shared_net_focal_fallback+registered"]["status"] == "supported"
    with pytest.raises(ValueError, match="shot-homography propagation"):
        auto_packet.automatic_camera_document(
            _shared_net_fallback_npz(tmp_path),
            labels,
            (10, 13),
            shot_homography_propagation="yes",
        )


def test_player_image_scale_comes_from_the_sidecar_not_a_per_attempt_constant(tmp_path):
    half = _players_csv(tmp_path, legacy=True)
    assert auto_packet.uniform_native_scale(half, ("x0", "y0", "x1", "y1")) == 2.0


def test_predicted_event_interval_preserves_producer_bounds_and_never_claims_exact():
    assert auto_packet.automatic_event_interval({"frame_interval": [9.0, 12.0]}, 10.4) == (
        [9.0, 12.0],
        "producer_supplied_interval",
    )
    interval, origin = auto_packet.automatic_event_interval({}, 10.4)
    assert interval == pytest.approx([9.4, 11.4])
    assert origin == "prediction_search_radius_one_native_frame_v1"
    with pytest.raises(ValueError, match="inside"):
        auto_packet.automatic_event_interval({"frame_interval": [11.0, 12.0]}, 10.4)
    with pytest.raises(ValueError, match="integer"):
        auto_packet.native_frame(10.4)


def test_generated_boundary_search_preserves_epoch_and_original_radius():
    raw = dict(event_type="contact", frame=2, location=dict(frame_subpixel=1.52734375))
    event = auto_packet.automatic_physical_event(raw, prediction_window=(1, 175))
    assert event["frame"] == 1.52734375
    assert event["frame_interval"] == [1, 2.52734375]
    assert event["original_prediction_search_interval"] == [0.52734375, 2.52734375]
    assert event["prediction_search_native_window"] == [1, 175]
    assert event["exact_epoch_observed"] is False
    assert "frame_interval" not in raw


def test_native_window_never_clips_supplied_uncertainty():
    raw = dict(event_type="contact", frame=2, frame_interval=[0.0, 4.0])
    event = auto_packet.automatic_physical_event(raw, prediction_window=(1, 175))
    assert event["frame_interval"] == [0.0, 4.0]
    assert event["interval_origin"] == "producer_supplied_interval"
    assert "original_prediction_search_interval" not in event


@pytest.mark.parametrize("frame", [1, 1.25, 174.75, 175])
def test_generated_search_stays_inside_inclusive_native_window(frame):
    interval, _ = auto_packet.automatic_event_interval({}, frame, prediction_window=(1, 175))
    assert 1 <= interval[0] <= frame <= interval[1] <= 175
    assert interval[0] < interval[1]


def test_interior_event_is_exactly_unchanged_and_outside_prediction_refuses():
    raw = dict(event_type="bounce", frame=20, location=dict(frame_subpixel=20.125))
    assert auto_packet.automatic_physical_event(raw, prediction_window=(1, 175)) == (
        auto_packet.automatic_physical_event(raw)
    )
    with pytest.raises(ValueError, match="inside its native window"):
        auto_packet.automatic_physical_event(raw, prediction_window=(21, 175))


def test_innovation_covariance_is_not_an_observation_radius(tmp_path):
    path = _ball_csv(tmp_path)
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row.update(
            innovation_cov_xx_native="557",
            innovation_cov_xy_native="0",
            innovation_cov_yy_native="557",
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = auto_packet.automatic_ball_rows(path, "pt0001")
    assert output[0]["uncertainty_radius_px1080"] is None
    assert output[0]["automatic_covariance_native_px2"] == [557, 0, 557]
    assert (
        output[0]["automatic_covariance_semantics"] == "innovation_covariance_not_observation_noise"
    )


def test_coarse_lock_keeps_interpolation_ancestry_without_dropping_real_guides(tmp_path):
    path = _ball_csv(tmp_path)
    rows = list(csv.DictReader(path.open()))
    guide = tmp_path / "guide.csv"
    for row in rows:
        row["sources"] = "interpolated" if row["frame"] == "f_0011.jpg" else "crop_wasb"
    with guide.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sidecar = json.loads(path.with_suffix(".csv.coordinates.json").read_text())
    guide.with_suffix(".csv.coordinates.json").write_text(json.dumps(sidecar))
    sidecar["source"] = str(guide)
    path.with_suffix(".csv.coordinates.json").write_text(json.dumps(sidecar))
    for row in rows:
        row["sources"] = "coarse_lock+provenance:coarse"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = auto_packet.automatic_ball_rows(path, "pt0001")
    assert output[0]["status"] == "visible"
    assert output[0]["support_class"] == "guide_observation"
    assert output[1]["status"] == "derived_estimate"
    assert output[1]["x1080"] == 911
    assert output[1]["guide_support"]["sources"] == "interpolated"


def test_sanitized_automatic_events_ignore_all_inactive_event_hints(tmp_path):
    from copy import deepcopy

    labels = json.loads(_labels(tmp_path).read_text())
    labels["source_pack"]["images"] = [
        dict(
            clip="pt0001",
            frame=f,
            native_pts_seconds=100 + (f - 1) / 25,
            clip_seconds=(f - 1) / 25,
            source={"sha256": str(f)},
            image_url=f"frames/f_{f:04d}.jpg",
        )
        for f in range(10, 21)
    ]
    events, ending = auto_packet.automatic_events(
        _events_json(tmp_path), "match-1", "pt0001", (10, 20)
    )
    ball = auto_packet.label_ball_rows(labels)
    first = auto_packet.mixed_observation_document(
        labels, ball, events, ending, ball_origin="label", event_origin="automatic"
    )
    poisoned = deepcopy(labels)
    poisoned["events"] = {"DO_NOT_READ": "wrong contacts"}
    poisoned["attempt"].update(
        ending_kind="serve_fault",
        first_contact_frame=-100,
        ending_frame=1,
        serve_number=2,
        topology="SECRET",
        notes="SECRET",
    )
    poisoned["serve_speed_evidence"] = {"SECRET": 500}
    poisoned["fractional_estimates"] = ["SECRET"]
    poisoned["ball_drafts"] = ["SECRET"]
    ball2 = deepcopy(ball)
    for row in ball2:
        row.update(note="SECRET", event_type="contact", motion_qualification={"SECRET": True})
    second = auto_packet.mixed_observation_document(
        poisoned, ball2, events, ending, ball_origin="label", event_origin="automatic"
    )
    assert second == first
    assert "SECRET" not in json.dumps(second)
    assert second["ball"]["records"][0]["frames"][0]["native_pts_seconds"] == 100.36
    assert second["events"]["records"][0]["exact_epoch_observed"] is False
    assert second["events"]["records"][0]["frame_interval"] == pytest.approx([10, 11.4])
    assert second["events"]["records"][0]["original_prediction_search_interval"] == pytest.approx(
        [9.4, 11.4]
    )


def test_automatic_ball_never_borrows_label_pixels_streaks_or_toss(tmp_path):
    labels = json.loads(_labels(tmp_path).read_text())
    labels["source_pack"]["images"] = [
        dict(clip="pt0001", frame=f, native_pts_seconds=(f - 1) / 25, source={"sha256": str(f)})
        for f in range(10, 21)
    ]
    events, ending = auto_packet.label_events(labels)
    automatic = auto_packet.automatic_ball_rows(_ball_csv(tmp_path), "pt0001")
    first = auto_packet.mixed_observation_document(
        labels, automatic, events, ending, ball_origin="automatic", event_origin="label"
    )
    labels["ball"] = {"DO_NOT_READ": True}
    labels["toss"] = {"contact_frame": -100}
    second = auto_packet.mixed_observation_document(
        labels, automatic, events, ending, ball_origin="automatic", event_origin="label"
    )
    assert first == second
    assert all("streak" not in r for r in second["ball"]["records"][0]["frames"])


def test_automatic_events_preserve_identity_only_support_and_original_location(tmp_path):
    path = _events_json(tmp_path)
    rows = json.loads(path.read_text())
    identity = {
        "status": "supported_by_model_and_observed_impulse",
        "certifies_physical_ending": False,
    }
    certificate = {
        "supported": True,
        "localization_status": "interval_supported_only",
        "pixel_support_offset_frames": -0.2,
    }
    rows[0].update(
        event_identity_support=identity, impulse_support=certificate, frame_interval=[9.8, 11.0]
    )
    rows[0]["location"].update(image_x=123, image_y=456)
    path.write_text(json.dumps(rows))
    events, ending = auto_packet.automatic_events(path, "match-1", "pt0001", (10, 20))
    first = events[0]
    assert first["status"] == first["occurrence_status"] == "predicted"
    assert first["automatic_event_identity_support"] == identity
    assert first["automatic_impulse_support"] == certificate
    assert first["automatic_location"] == rows[0]["location"]
    assert first["frame"] == 10.4 and first["frame_interval"] == [9.8, 11.0]
    assert ending["frame"] == 19.6


def test_strict_automatic_origin_cannot_drop_an_accepted_bounce_before_return(tmp_path):
    path = _events_json(tmp_path)
    events = json.loads(path.read_text())
    events.insert(0, dict(events[0], event_type="bounce", location={"frame_subpixel": 10.1}))
    path.write_text(json.dumps(events))
    # Frozen controlled-input behavior remains unchanged unless explicitly requested.
    physical, ending = auto_packet.automatic_events(path, "match-1", "pt0001", (10, 20))
    assert physical[0]["frame"] == 10.4
    with pytest.raises(ValueError, match="originating contact"):
        auto_packet.automatic_events(
            path, "match-1", "pt0001", (10, 20), require_originating_contact=True
        )


def test_camera_npz_members_are_read_once_and_unrelated_objects_remain_unread(
    tmp_path, monkeypatch
):
    from collections import Counter

    labels = json.loads(_labels(tmp_path).read_text())
    path = _camera_with_lens(
        tmp_path, {"k1": np.zeros(11), "dist_center": np.tile([960.0, 540.0], (11, 1))}
    )
    with np.load(path, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    np.savez_compressed(path, **values, unrelated_notes=np.array([{"unused": True}], dtype=object))
    calls = Counter()
    original = np.lib.npyio.NpzFile.__getitem__

    def read_once(archive, key):
        calls[key] += 1
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read_once)
    result = auto_packet.automatic_camera_document(path, labels, (10, 20))
    assert result["total"] == 11 and result["supported"] == 11
    assert max(calls.values()) == 1
    assert "unrelated_notes" not in calls
    # No state survives a call: changed input bytes are read on the next call.
    values["reliable"] = np.zeros(11, dtype=bool)
    np.savez_compressed(path, **values)
    calls.clear()
    result = auto_packet.automatic_camera_document(path, labels, (10, 20))
    assert result["supported"] == 0 and max(calls.values()) == 1
