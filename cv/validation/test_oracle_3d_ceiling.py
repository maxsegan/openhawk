from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cv.validation.oracle_3d_ceiling import (
    ARTIFACT_CLASS,
    load_owner_events,
    overlay_track_rows,
    write_json,
)


def test_owner_events_preserve_half_frame_and_scale_px540(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    with truth.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip",
                "event_type",
                "seed_id",
                "seed_frame",
                "labeled_frame",
                "labeled_x540",
                "labeled_y540",
                "verdict",
                "note",
            ],
        )
        writer.writeheader()
        for index in range(1_415):
            writer.writerow(
                {
                    "clip": f"match__pt{index % 146:04d}",
                    "event_type": "bounce" if index == 0 else "contact",
                    "seed_id": f"event_{index}",
                    "seed_frame": "10",
                    "labeled_frame": "10.5" if index == 0 else "10",
                    "labeled_x540": "123.25",
                    "labeled_y540": "45.5",
                    "verdict": "confirmed",
                    "note": "",
                }
            )
    points = {f"match__pt{index:04d}" for index in range(146)}

    rows, by_point = load_owner_events(truth, points, {"match": 25.0})

    assert len(rows) == 1_415
    assert len(by_point) == 146
    bounce = next(row for row in rows if row["event_type"] == "bounce")
    assert bounce["frame"] == 10.5
    assert bounce["location"]["frame_subpixel"] == 10.5
    assert bounce["location"]["image_x"] == 246.5
    assert bounce["location"]["image_y"] == 91.0
    assert bounce["location"]["image_coordinate_space"] == "native_1920x1080"
    assert bounce["artifact_class"] == ARTIFACT_CLASS


def test_track_overlay_replaces_adds_and_marks_owner_truth() -> None:
    rows = [
        {
            "clip": "pt0001",
            "frame": "f_0010.jpg",
            "x": "1",
            "y": "2",
            "track_id": "0",
            "score": "0.5",
            "rank": "0",
            "sources": "automatic_detector",
        },
        {
            "clip": "pt0001",
            "frame": "f_0012.jpg",
            "x": "3",
            "y": "4",
            "track_id": "0",
            "score": "0.5",
            "rank": "0",
            "sources": "automatic_detector",
        },
    ]
    truth = {
        "match__pt0001": {
            10: {"x1080": 200.0, "y1080": 100.0, "source": "sequence"},
            11: {"x1080": 220.0, "y1080": 120.0, "source": "trajectory"},
        }
    }

    output, counts = overlay_track_rows(rows, truth)

    assert counts == {"replaced": 1, "added": 1, "owner_rows": 2}
    by_frame = {row["frame"]: row for row in output}
    assert by_frame["f_0010.jpg"]["x"] == "100"
    assert by_frame["f_0011.jpg"]["y"] == "60"
    assert by_frame["f_0010.jpg"]["provenance"] == "owner_truth"
    assert by_frame["f_0012.jpg"]["provenance"] == "automatic"


def test_json_artifacts_normalize_numpy_and_are_marked(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"

    write_json(output, {"value": np.int64(3), "array": np.asarray([1.5])})

    payload = json.loads(output.read_text())
    assert payload["artifact_class"] == ARTIFACT_CLASS
    assert payload["value"] == 3
    assert payload["array"] == [1.5]
