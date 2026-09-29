"""Exact native PTS and nested source-image export regressions."""

from copy import deepcopy
from fractions import Fraction
import hashlib

import pytest

from cv.viz import export_connected_3d as connected
from cv.viz import export_local_s6 as local


def native_label(tmp_path):
    pts = [2918616, 2918633, 2918649, 2918666]
    images = []
    for i, epoch in enumerate(pts):
        path = tmp_path / f"pts_{epoch}.jpg"
        path.write_bytes(f"original-picture-{i}".encode())
        images.append(
            {
                "frame": i,
                "clip": "pt0001",
                "source_pts": epoch,
                "source_time_base": "1/1000",
                "timestamp_seconds": epoch / 1000,
                "source": {
                    "path": path.name,
                    "path_base": "TENNIS_DATA_ROOT",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            }
        )
    return {
        "schema": "s6_labeled_validation_observation_document_v1",
        "source_pack": {
            "fps": 60000 / 1001,
            "images": images,
            "native_clock": {
                "modeled_fps_exact": "60000/1001",
                "modeled_fps": 60000 / 1001,
                "original_start_seconds_exact": str(Fraction(pts[0], 1000)),
                "original_pts_preserved": True,
                "pictures_resampled": False,
            },
        },
    }


def test_declared_native_epochs_and_fractional_interpolation(tmp_path):
    label = native_label(tmp_path)
    before = deepcopy(label)
    clock, time = connected._native_timebase(label)
    assert clock["one_based_frames"] is False
    assert time(0) == 2918.616
    assert time(1.5) == pytest.approx(2918.641)
    assert time(2) - time(1) == pytest.approx(0.016)
    assert time(1) - time(0) == pytest.approx(0.017)
    assert local.native_context_bounds(label, "pt0001", (0, 3), 0.5, 2.5, tmp_path) == (0, 3)
    with pytest.raises(ValueError, match="outside"):
        time(-0.1)
    assert label == before


@pytest.mark.parametrize(
    "mutation", ["timestamp", "duplicate", "reverse", "origin", "fps", "resampled"]
)
def test_native_clock_conflicts_refuse(tmp_path, mutation):
    label = native_label(tmp_path)
    pack = label["source_pack"]
    if mutation == "timestamp":
        pack["images"][1]["timestamp_seconds"] += 0.001
    elif mutation == "duplicate":
        pack["images"][1]["frame"] = 0
    elif mutation == "reverse":
        pack["images"][1]["source_pts"] = 2918615
        pack["images"][1]["timestamp_seconds"] = 2918.615
    elif mutation == "origin":
        pack["native_clock"]["original_start_seconds_exact"] = "0"
    elif mutation == "fps":
        pack["fps"] = 25
    elif mutation == "resampled":
        pack["native_clock"]["pictures_resampled"] = True
    with pytest.raises(ValueError):
        connected._native_timebase(label)


def test_nested_asset_binding_renames_only_viewer_file_and_rejects_changed_pixels(tmp_path):
    label = native_label(tmp_path)
    _, time = connected._native_timebase(label)
    out = tmp_path / "viewer"
    manifest, _ = connected._link_frames("case", label, "pt0001", 0, 3, out, tmp_path, time)
    assert [r["file"] for r in manifest["frames"]] == [f"f_{i:04d}.jpg" for i in range(4)]
    for image, row in zip(label["source_pack"]["images"], manifest["frames"], strict=True):
        assert (out / "frames/case" / row["file"]).read_bytes() == (
            tmp_path / image["source"]["path"]
        ).read_bytes()
        assert row["source_pts"] == image["source_pts"]
    with pytest.raises(ValueError, match="lacks declared"):
        connected._link_frames("extra", label, "pt0001", 0, 4, out, tmp_path, time)
    (tmp_path / label["source_pack"]["images"][0]["source"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        connected._link_frames("changed", label, "pt0001", 0, 3, out, tmp_path, time)


@pytest.mark.parametrize("mixed", [False, True])
def test_merging_batches_preserves_prior_observation_only_holds(tmp_path, monkeypatch, mixed):
    import json

    run = tmp_path / "run"
    run.mkdir()
    key = "new_case"
    (run / "manifest.json").write_text(
        json.dumps(
            {"schema": "labeled_s6_observations_v1", "rows": [{"key": key, "declared_flights": 1}]}
        )
    )
    case = run / "cases" / key
    case.mkdir(parents=True)
    (case / "result.json").write_text("{}")
    out = tmp_path / "out"
    (out / "data").mkdir(parents=True)
    prior = {
        "source_groups": [],
        "points": [
            {"file": "local_s6_old_fit.json", "point": "old_fit", "source_group": local.GROUP}
        ],
        "local_s6": {
            "held_before_fitting": [
                {"key": "old_hold", "file": "local_s6_old_hold.json", "frames_available": True}
            ]
        },
    }
    if mixed:
        prior["points"].extend(
            [
                {
                    "file": "automatic_old.json",
                    "point": "automatic_old",
                    "source_group": local.AUTOMATIC_GROUP,
                },
                {
                    "file": "legacy_old.json",
                    "point": "legacy_old",
                    "source_group": "curated_real_attempts",
                },
            ]
        )
        prior["local_s6"]["held_before_fitting"].append(
            {"key": "foreign_auto_hold", "source_group": local.AUTOMATIC_GROUP}
        )
        prior["automatic_s6"] = {"held_before_fitting": [{"key": "automatic_held"}]}
    (out / "data/index.json").write_text(json.dumps(prior))
    monkeypatch.setattr(local, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(local, "resolve_shared_root", lambda *_: tmp_path)
    monkeypatch.setattr(local, "copy_static", lambda _: None)
    monkeypatch.setattr(
        local,
        "export_case",
        lambda *_: {
            "key": key,
            "held": True,
            "file": "local_s6_new_case.json",
            "frames_available": True,
        },
    )
    result = local.export_run(run, out, merge=True)
    assert [r["key"] for r in result["held_before_fitting"]] == ["old_hold", "new_case"]
    assert result["requested_attempts"] == 3
    assert result["exported_attempts"] == 1
    index = json.loads((out / "data/index.json").read_text())
    if mixed:
        assert len(index["points"]) == 3
        assert index["automatic_s6"]["held_before_fitting"] == [{"key": "automatic_held"}]
