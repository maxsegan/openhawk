"""Player preparation preserves native evidence and refuses unsupported roots."""

import csv
import json

import numpy as np
import pytest

from cv.experiments.connected_shooting import camera_geometry, player_state_fallback
from cv.pipeline import provenance, s6_labeled_stage as stage, s6_player_camera as prep


def camera(frame=1):
    return dict(frame=frame, status="supported", P=[[10, 0, 0, 100], [0, 10, 0, 100], [0, 0, 1, 1]])


def pose(frame=1):
    return dict(
        clip="p",
        frame=f"f_{frame:04d}.jpg",
        side="near",
        x0="100",
        y0="100",
        x1="140",
        y1="150",
        court_x="999",
        court_y="888",
        timestamp="0.02",
        left_wrist="120,130,0.8",
    )


def fixture(tmp_path, rows=None, sidecar=True):
    rows = rows or [pose(), pose(2), pose() | {"clip": "other"}]
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if sidecar:
        source.with_name(source.name + ".coordinates.json").write_text(
            json.dumps(
                dict(
                    schema="tennis.coordinate-space.v1",
                    image_size=dict(width=1920, height=1080),
                    artifact_size=dict(width=1920, height=1080),
                    artifact=source.name,
                )
            )
        )
    cameras = dict(clip="p", cameras=[camera()])
    cp = tmp_path / "cameras.json"
    cp.write_text(json.dumps(cameras))
    output = tmp_path / "out"
    output.mkdir()
    return dict(pose_csv=source, cameras=cp), cameras, output


def test_original_tracker_is_native_and_not_pose_box_or_old_root():
    row = pose() | dict(track_x0="120", track_y0="110", track_x1="160", track_y1="160")
    result = prep.derive_row(row, camera(), 2.0)
    assert result["native_bottom_px"] == [140.0, 160.0]
    assert result["court_xy_m"] == [4.0, 6.0]
    assert result["proxy"] == "original_tracker_box_native"
    assert prep.derive_row(row | dict(court_x="-10", court_y="10000"), camera(), 2) == result


def test_radial_ground_projection_reproduces_observation():
    C = camera() | dict(k1=1e-6, dist_center=[100.0, 100.0])
    uv = camera_geometry.project(
        np.array(C["P"])[None], np.array([[2.0, 5.0, 0.0]]), camera_geometry.radial_row(C)[None]
    )[0]
    row = pose() | dict(x0=str(uv[0] - 10), x1=str(uv[0] + 10), y1=str(uv[1]))
    result = prep.derive_row(row, C, 1)
    np.testing.assert_allclose(result["court_xy_m"], [2, 5], atol=1e-10)
    assert result["native_reprojection_max_px"] < 1e-6


@pytest.mark.parametrize(
    "changed,match",
    [
        ({"status": "unsupported"}, "not explicitly supported"),
        ({"k1": 0.1}, "radial camera requires"),
        ({"P": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1]]}, "behind"),
    ],
)
def test_unsupported_camera_is_not_old_calibration(changed, match):
    with pytest.raises(ValueError, match=match):
        prep.derive_row(pose(), camera() | changed, 1)


def test_partial_tracker_box_does_not_silently_use_different_pose_box():
    with pytest.raises(ValueError, match="incomplete player box"):
        prep.derive_row(pose() | {"track_x0": "80"}, camera(), 1)


@pytest.mark.parametrize("sidecar", [True, False])
def test_preparation_binds_derived_artifact_preserves_evidence_and_supported_only_fallback(
    tmp_path, sidecar
):
    inputs, cameras, output = fixture(tmp_path, sidecar=sidecar)
    old_bytes = inputs["pose_csv"].read_bytes()
    original = list(csv.DictReader(inputs["pose_csv"].open()))
    derived, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    assert inputs["pose_csv"].read_bytes() == old_bytes
    rows = list(csv.DictReader(derived["pose_csv"].open()))
    for before, after in zip(original, rows, strict=True):
        assert {k: after[k] for k in before if k not in ("court_x", "court_y")} == {
            k: v for k, v in before.items() if k not in ("court_x", "court_y")
        }
    assert rows[0]["court_x"] == "2.0" and rows[0]["court_y"] == "5.0"
    assert rows[1]["court_x"] == rows[1]["court_y"] == ""
    assert rows[1]["s6_original_court_x"] == "999"
    assert rows[2]["court_x"] == "999"  # Outside this attempt, never consumed.
    supported = player_state_fallback.sided_court_rows(derived["pose_csv"], "p")
    assert list(supported) == [(1, "near")]
    space = stage.pose_space_sidecar(derived["pose_csv"])
    assert ("image_size" in space) is sidecar
    assert space["s6_player_camera_derivation"]["sources"] == receipt["sources"]
    assert receipt["derived_pose"] == provenance.file_record(derived["pose_csv"])
    assert space["record"] == receipt["derived_coordinate_sidecar"]
    assert receipt["counts"] == {"derived": 1, "unsupported": 1}


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("mode", ["off", "on"])
def test_prepare_accepts_flagged_sided_boxes_and_refuses_unflagged_when_default_on(
    tmp_path, flag, mode
):
    from cv.pipeline import player_side_association as psa
    from cv.pipeline import resolution as res

    source = tmp_path / "player_boxes_25_native_sided_v1.csv"
    source.write_text("clip,frame,side,x0,y0,x1,y1,court_x,court_y\np,f_0001.jpg,near,1,1,2,2,0,0\n")
    sidecar = {
        "schema": "tennis.coordinate-space.v1",
        "artifact": source.name,
        "artifact_identity": res.PLAYER_BOXES_NATIVE_SIDED_IDENTITY,
        psa.KEEP_UNIQUE_SIDECAR_KEY: flag,
        "image_size": {"width": 1920, "height": 1080},
        "artifact_size": {"width": 1920, "height": 1080},
    }
    source.with_name(source.name + ".coordinates.json").write_text(json.dumps(sidecar))
    cameras = dict(clip="p", cameras=[camera()])
    cp = tmp_path / "cameras.json"
    cp.write_text(json.dumps(cameras))
    output = tmp_path / "out"
    output.mkdir()
    inputs = dict(pose_csv=source, cameras=cp)
    if flag:
        prep.prepare(inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, mode)
        return
    with pytest.raises(ValueError, match="keep_unique_admissible_frames"):
        prep.prepare(inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, mode)


def test_off_retains_exact_paths_and_writes_nothing(tmp_path):
    inputs = {"pose_csv": tmp_path / "absent"}
    result, receipt = prep.prepare(inputs, {}, {}, {}, tmp_path, "off")
    assert result is inputs and not receipt["enabled"] and not list(tmp_path.iterdir())


def test_duplicate_camera_abstains_and_never_interpolates(tmp_path):
    inputs, cameras, output = fixture(tmp_path)
    cameras["cameras"].append(camera())
    _, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    assert receipt["counts"] == {"unsupported": 2}


def test_original_space_mismatch_is_explicit(tmp_path):
    inputs, cameras, output = fixture(tmp_path)
    with pytest.raises(ValueError, match="scale conflicts"):
        prep.prepare(
            inputs,
            {"pose_image_scale": 2},
            {"attempts": [{"point_clip": "p"}]},
            cameras,
            output,
            "on",
        )


@pytest.mark.parametrize("tamper", ["pose_csv", "sidecar", "cameras"])
def test_replay_requires_derived_csv_and_sidecar_and_exact_camera(tmp_path, tamper):
    inputs, cameras, output = fixture(tmp_path)
    actual, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    report = {"s6_player_camera_coordinates": receipt}
    prep.require_replay_inputs(report, actual["pose_csv"], actual["cameras"])
    with pytest.raises(ValueError, match="pose CSV binding changed"):
        prep.require_replay_inputs(report, inputs["pose_csv"], inputs["cameras"])
    path = (
        actual["pose_csv"].with_name(actual["pose_csv"].name + ".coordinates.json")
        if tamper == "sidecar"
        else actual[tamper]
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="binding changed"):
        prep.require_replay_inputs(report, actual["pose_csv"], actual["cameras"])


def mixed_native_fixture(tmp_path, *, row_change=None, declaration_change=None):
    record = pose() | dict(x0_native="200", y0_native="200", x1_native="280", y1_native="300")
    if row_change:
        record.update(row_change)
    inputs, cameras, output = fixture(tmp_path, rows=[record])
    path = inputs["pose_csv"].with_name("source.csv.coordinates.json")
    document = json.loads(path.read_text()) | dict(
        interface_space="native_1920x1080",
        legacy_artifact_size=dict(width=960, height=540),
        coordinate_columns={
            "legacy_960x540": ["x0", "y0", "x1", "y1"],
            "native_1920x1080": list(prep.NATIVE_BOX_COLUMNS),
        },
    )
    document.update(declaration_change or {})
    path.write_text(json.dumps(document))
    return inputs, cameras, output


def test_declared_native_bbox_is_not_shrunk_to_legacy_mirror_and_replays(tmp_path):
    inputs, cameras, output = mixed_native_fixture(tmp_path)
    original = inputs["pose_csv"].read_bytes()
    packet = {"attempts": [{"point_clip": "p"}]}
    inputs["packet"] = tmp_path / "packet.json"
    inputs["packet"].write_text(json.dumps(packet))
    actual, receipt = prep.prepare(inputs, {"pose_image_scale": 1}, packet, cameras, output, "on")
    applied = receipt["applications"][0]
    assert applied["native_bottom_px"] == [240.0, 300.0]
    assert applied["court_xy_m"] == [14.0, 20.0]
    assert applied["proxy"] == "declared_player_box_native"
    assert inputs["pose_csv"].read_bytes() == original
    before = next(csv.DictReader(inputs["pose_csv"].open()))
    after = next(csv.DictReader(actual["pose_csv"].open()))
    assert all(
        after[k] == v
        for k, v in before.items()
        if k not in ("court_x", "court_y", "x0", "y0", "x1", "y1")
    )
    for alias, native in zip(("x0", "y0", "x1", "y1"), prep.NATIVE_BOX_COLUMNS, strict=True):
        assert after[alias] == before[native]
        assert after["s6_original_box_" + alias] == before[alias]
    assert receipt["sources"]["pose_coordinate_sidecar"] == provenance.file_record(
        inputs["pose_csv"].with_name("source.csv.coordinates.json")
    )
    prep.verify_derivation(inputs, {"pose_image_scale": 1}, actual["pose_csv"], receipt)


def test_original_track_bbox_precedes_declared_native_bbox(tmp_path):
    inputs, cameras, output = mixed_native_fixture(
        tmp_path,
        row_change=dict(track_x0="120", track_y0="110", track_x1="160", track_y1="160"),
    )
    _, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    assert receipt["applications"][0]["native_bottom_px"] == [140.0, 160.0]
    assert receipt["applications"][0]["proxy"] == "original_tracker_box_native"


@pytest.mark.parametrize("value", ["", "nan", "inf"])
def test_invalid_declared_native_row_never_falls_back_to_legacy_bbox(tmp_path, value):
    inputs, cameras, output = mixed_native_fixture(tmp_path, row_change={"x0_native": value})
    actual, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    assert receipt["counts"] == {"unsupported": 1}
    row = next(csv.DictReader(actual["pose_csv"].open()))
    assert row["court_x"] == row["court_y"] == ""
    assert row["s6_original_box_x0"] == "100"
    assert row["x0"] == row["x0_native"] == value


@pytest.mark.parametrize("kind", ["partial", "missing_sidecar", "wrong_size", "wrong_mapping"])
def test_native_headers_require_complete_consistent_coordinate_declaration(tmp_path, kind):
    inputs, cameras, output = mixed_native_fixture(tmp_path)
    source = inputs["pose_csv"]
    sidecar = source.with_name("source.csv.coordinates.json")
    if kind == "partial":
        record = next(csv.DictReader(source.open()))
        del record["y1_native"]
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
    elif kind == "missing_sidecar":
        sidecar.unlink()
    else:
        space = json.loads(sidecar.read_text())
        if kind == "wrong_size":
            space["artifact_size"] = dict(width=960, height=540)
        else:
            space["coordinate_columns"]["native_1920x1080"] = ["x0", "y0", "x1", "y1"]
        sidecar.write_text(json.dumps(space))
    with pytest.raises(ValueError):
        prep.prepare(inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on")
    assert not (output / "prepared_player_pose.csv").exists()


def test_dual_box_mirrors_normalize_once_before_normal_serve_consumers(tmp_path):
    from cv.pipeline import resolution
    from cv.experiments.connected_shooting import agent_single_flight_search as single, toss_witness

    records = []
    for frame in range(1, 5):
        for side, box in [("near", [900, 700, 1000, 1000]), ("far", [450, 300, 550, 450])]:
            row = pose(frame) | dict(
                side=side,
                track_id="0" if side == "near" else "1",
                left_wrist_x="510",
                left_wrist_y="340",
                left_wrist_confidence=".9",
            )
            row.update(
                {
                    key: str(value / 2)
                    for key, value in zip(("x0", "y0", "x1", "y1"), box, strict=True)
                }
            )
            row.update(
                {key: str(value) for key, value in zip(prep.NATIVE_BOX_COLUMNS, box, strict=True)}
            )
            records.append(row)
    inputs, cameras, output = fixture(tmp_path, records)
    cameras["cameras"] = [camera(frame) for frame in range(1, 5)]
    inputs["cameras"].write_text(json.dumps(cameras))
    resolution.write_native_dual_coordinate_manifest(
        resolution.coordinate_manifest_path(inputs["pose_csv"]),
        image_size=resolution.NATIVE_SIZE,
        legacy_size=resolution.LEGACY_TRACKING_SIZE,
        source="original detector",
        native_columns=prep.NATIVE_BOX_COLUMNS,
        legacy_columns=("x0", "y0", "x1", "y1"),
    )
    original_bytes = inputs["pose_csv"].read_bytes()
    assert single.server_state(inputs["pose_csv"], "p", 4, np.array([500, 340]))["side"] == "near"
    derived, receipt = prep.prepare(
        inputs, {}, {"attempts": [{"point_clip": "p"}]}, cameras, output, "on"
    )
    assert inputs["pose_csv"].read_bytes() == original_bytes
    rows = list(csv.DictReader(derived["pose_csv"].open()))
    for before, after in zip(records, rows, strict=True):
        for alias, native in zip(("x0", "y0", "x1", "y1"), prep.NATIVE_BOX_COLUMNS, strict=True):
            assert after[alias] == before[native]
            assert after["s6_original_box_" + alias] == before[alias]
            assert after[native] == before[native]
        assert after["timestamp"] == before["timestamp"]
        assert after["left_wrist_x"] == before["left_wrist_x"]
        expected = prep.derive_row(before, camera(), 1, native_box_columns=prep.NATIVE_BOX_COLUMNS)
        np.testing.assert_array_equal(
            [float(after["court_x"]), float(after["court_y"])], expected["court_xy_m"]
        )
    state = single.server_state(derived["pose_csv"], "p", 4, np.array([500, 340]))
    assert state["side"] == "far"
    assert state["box_height_native_px"] == 150
    feet = toss_witness.server_feet(derived["pose_csv"], "p", 4, np.array([500, 340]))
    assert feet["side"] == "far"
    space = resolution.read_coordinate_manifest(derived["pose_csv"])
    assert resolution.manifest_artifact_size(space, legacy_columns=True) == resolution.NATIVE_SIZE
    assert receipt["box_mirror_normalization"]["rows"] == 8
