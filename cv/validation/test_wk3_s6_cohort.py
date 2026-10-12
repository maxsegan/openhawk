from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv.validation.wk3_s6_cohort import build_mirror, verify


def test_build_mirror_reserves_camera_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    match = source / "example"
    manifests = match / "run_manifests"
    manifests.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps({"matches": [{"id": "example", "point_ids": [1, 2], "source_fps": 25.0}]})
    )
    (match / "ball.csv").write_text("frame,x,y\n")
    (match / "camera_P_per_point.npz").write_text("old")
    (manifests / "frame_extraction.json").write_text("{}")

    report = build_mirror(source, output)

    assert report["points"] == 2
    assert (output / "manifest.json").is_symlink()
    assert (output / "example" / "ball.csv").is_symlink()
    assert not (output / "example" / "camera_P_per_point.npz").exists()
    assert not (output / "example" / "run_manifests").is_symlink()
    assert (output / "example" / "run_manifests" / "frame_extraction.json").is_symlink()


def test_verify_reports_per_point_reliability(tmp_path: Path) -> None:
    root = tmp_path / "root"
    match = root / "example"
    match.mkdir(parents=True)
    (root / "wk3_s6_cohort_manifest.json").write_text(json.dumps({"matches": ["example"]}))
    (root / "manifest.json").write_text(
        json.dumps({"matches": [{"id": "example", "point_ids": [1, 2]}]})
    )
    np.savez_compressed(
        match / "camera_P_per_frame_v1.npz",
        clips=np.asarray(["pt0001", "pt0001", "pt0002"]),
        frames=np.asarray([0, 1, 0]),
        P=np.zeros((3, 3, 4)),
        frame_scope=np.asarray(["frame_track"] * 3),
        reliable=np.asarray([True, False, True]),
        source=np.asarray(["direct+registered"] * 3),
    )
    np.savez_compressed(match / "court_H_per_frame_v1.npz", frames=np.asarray([0]))

    report = verify(root)

    assert report["points"] == 2
    assert report["all_frame_scope_frame_track"] is True
    assert [row["reliable_fraction"] for row in report["rows"]] == [0.5, 1.0]
