import json

import numpy as np
import pytest

from cv.pipeline import camera_frame_support as support

MATCH = "match"


def write_transport(match_root, rows):
    """rows: (clip, frame, reliable, matrix)."""
    match_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        match_root / "court_H_per_frame_v1.npz",
        clips=np.asarray([clip for clip, _, _, _ in rows]),
        frames=np.asarray([frame for _, frame, _, _ in rows], dtype=np.int32),
        H=np.asarray([matrix for _, _, _, matrix in rows], dtype=float),
        reliable=np.asarray([flag for _, _, flag, _ in rows], dtype=bool),
        source=np.asarray(["direct+registered"] * len(rows)),
    )


def reliable_rows(clip, frames):
    return [(clip, frame, True, np.eye(3)) for frame in frames]


def test_support_collapses_accepted_native_frames_into_closed_runs(tmp_path):
    write_transport(tmp_path / MATCH, reliable_rows("pt0001", [*range(10, 16), *range(40, 43), 90]))

    rows = support.match_rows(tmp_path / MATCH, MATCH, ["pt0001"])

    assert rows[0]["supported_spans"] == [[10, 15], [40, 42], [90, 90]]
    assert rows[0]["supported_frames"] == 10
    assert rows[0]["artifact_present"] is True
    assert "reason" not in rows[0]


def test_an_unreliable_registration_is_not_support(tmp_path):
    # The close-up case: the frames exist in the artifact but the automatic
    # registration did not accept them, so the view is not certified.
    write_transport(
        tmp_path / MATCH,
        [
            *[("pt0001", frame, False, np.eye(3)) for frame in range(1, 20)],
            *reliable_rows("pt0001", range(20, 30)),
        ],
    )

    row = support.match_rows(tmp_path / MATCH, MATCH, ["pt0001"])[0]

    assert row["supported_spans"] == [[20, 29]]
    assert row["supported_frames"] == 10


def test_a_nonfinite_or_singular_homography_is_not_support(tmp_path):
    write_transport(
        tmp_path / MATCH,
        [
            ("pt0001", 1, True, np.full((3, 3), np.nan)),
            ("pt0001", 2, True, np.zeros((3, 3))),
            ("pt0001", 3, True, np.eye(3)),
        ],
    )

    row = support.match_rows(tmp_path / MATCH, MATCH, ["pt0001"])[0]

    assert row["supported_spans"] == [[3, 3]]


def test_a_clip_with_no_accepted_frame_publishes_an_explicit_reason(tmp_path):
    write_transport(tmp_path / MATCH, reliable_rows("pt0001", [1, 2, 3]))

    rows = support.match_rows(tmp_path / MATCH, MATCH, ["pt0001", "pt0002"])

    assert rows[1]["supported_spans"] == []
    assert rows[1]["reason"] == support.NO_SUPPORTED_FRAME_REASON
    assert rows[1]["artifact_present"] is True


def test_a_missing_artifact_is_reported_rather_than_assumed_supported(tmp_path):
    (tmp_path / MATCH).mkdir(parents=True)

    rows = support.match_rows(tmp_path / MATCH, MATCH, ["pt0001"])

    assert rows[0]["supported_spans"] == []
    assert rows[0]["artifact_present"] is False
    assert rows[0]["reason"] == support.MISSING_ARTIFACT_REASON


def test_the_document_declares_the_transport_and_binds_the_artifact(tmp_path, monkeypatch):
    write_transport(tmp_path / MATCH, reliable_rows("pt0001", range(5, 9)))
    manifest = {
        "schema": "broadcast_pipeline_manifest_v1",
        "points_per_match": 1,
        "matches": [{"id": MATCH, "point_ids": [1]}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "support.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "camera_frame_support",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--out",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    support.main()
    document = json.loads(output.read_text())

    assert document["schema"] == support.SCHEMA
    assert document["labels_loaded"] is False
    assert document["transport"] == {
        "mode": "reliable_per_frame",
        "missing": "hold",
        "artifact_name": "court_H_per_frame_v1.npz",
    }
    assert document["frame_index_origin"] == 1
    assert document["points"] == 1
    assert document["supported_points"] == 1
    assert document["rows"][0]["point"] == f"{MATCH}/pt0001"
    assert document["rows"][0]["supported_spans"] == [[5, 8]]
    assert len(document["artifacts"]) == 1
    assert document["artifacts"][0]["sha256"]


def test_the_transport_refuses_a_duplicated_native_frame(tmp_path):
    write_transport(tmp_path / MATCH, reliable_rows("pt0001", [7, 7]))

    with pytest.raises(ValueError, match="duplicate native court frame"):
        support.match_rows(tmp_path / MATCH, MATCH, ["pt0001"])
