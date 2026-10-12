import csv
import json

import pytest

import resolution as res
from track_artifact import rows_with_native_mirror, write_track_artifact


def test_non_native_manifest_requires_flag_and_justification() -> None:
    legacy = {
        "schema": "tennis.coordinate-space.v1",
        "artifact_size": {"width": 960, "height": 540},
    }
    assert "subnative_flagged" in res.coordinate_manifest_errors(legacy)[0]
    legacy["subnative_flagged"] = True
    assert "resolution_status" in res.coordinate_manifest_errors(legacy)[0]
    legacy["resolution_status"] = res.LEGACY_BAD_SHOULD_UPDATE
    assert "subnative_justification" in res.coordinate_manifest_errors(legacy)[0]
    legacy["subnative_justification"] = "frozen model input; migration tail item 1"
    assert res.coordinate_manifest_errors(legacy) == []


def test_native_manifest_needs_no_exception() -> None:
    manifest = {
        "schema": "tennis.coordinate-space.v1",
        "artifact_size": {"width": 1920, "height": 1080},
    }
    assert res.coordinate_manifest_errors(manifest) == []


def test_writer_rejects_unflagged_subnative_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="subnative_flagged"):
        res.write_coordinate_manifest(
            tmp_path / "bad.coordinates.json",
            image_size=res.NATIVE_SIZE,
            artifact_size=res.CANONICAL_SIZE,
            source="frames",
        )


def test_track_writer_adds_native_columns_and_native_sidecar(tmp_path) -> None:
    source = tmp_path / "candidates.csv"
    source.write_text("clip,frame,x,y,score\npt0001,f_0001.jpg,10.25,20.5,0.9\n")
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(source),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.CANONICAL_SIZE,
        source="native frames",
        subnative_flagged=True,
        subnative_justification="legacy candidate fixture",
    )
    output = tmp_path / "track.csv"
    rows = [
        {
            "clip": "pt0001",
            "frame": "f_0001.jpg",
            "x": 10.25,
            "y": 20.5,
            "track_id": 0,
        }
    ]

    write_track_artifact(output, rows, [source])

    written = next(csv.DictReader(output.open(newline="")))
    assert float(written["x"]) == 10.25
    assert float(written["y"]) == 20.5
    assert float(written["x_native"]) == 20.5
    assert float(written["y_native"]) == 41.0
    sidecar = json.loads(res.coordinate_manifest_path(output).read_text())
    assert sidecar["artifact_size"] == {"width": 1920, "height": 1080}
    assert sidecar["legacy_artifact_size"] == {"width": 960, "height": 540}
    assert sidecar["deprecation"]["resolution_status"] == res.LEGACY_BAD_SHOULD_UPDATE
    assert res.coordinate_manifest_errors(sidecar) == []


def test_track_writer_preserves_legacy_column_prefix(tmp_path) -> None:
    source = tmp_path / "candidates.csv"
    source.write_text("clip,frame,x,y,score\npt0001,f_0001.jpg,10,20,0.9\n")
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(source),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.CANONICAL_SIZE,
        source="native frames",
        subnative_flagged=True,
        subnative_justification="legacy candidate fixture",
    )
    output = tmp_path / "track.csv"
    rows = [
        {
            "clip": "pt0001",
            "frame": "f_0001.jpg",
            "x": 10.0,
            "y": 20.0,
            "track_id": 0,
            "score": 0.9,
            "rank": 0,
            "sources": "wasb",
        }
    ]

    write_track_artifact(output, rows, [source])

    with output.open(newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert fields == [*rows[0], "x_native", "y_native"]


def test_empty_track_writer_keeps_base_track_schema(tmp_path) -> None:
    source = tmp_path / "candidates.csv"
    source.write_text("clip,frame,x,y,score\n")
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(source),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.CANONICAL_SIZE,
        source="native frames",
        subnative_flagged=True,
        subnative_justification="legacy candidate fixture",
    )
    output = tmp_path / "track.csv"

    write_track_artifact(output, [], [source])

    with output.open(newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert fields == ["clip", "frame", "x", "y", "track_id", "x_native", "y_native"]


def test_track_mirror_rejects_missing_legacy_values() -> None:
    with pytest.raises(KeyError):
        rows_with_native_mirror([{"x": 1.0}], res.CANONICAL_SIZE)
