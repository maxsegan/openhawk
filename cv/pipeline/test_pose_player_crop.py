import argparse
import csv
import json
import hashlib

import numpy as np
import pytest

from cv.pipeline import artifact_cache, pose_player_crop
from cv.pipeline.pose import KEYPOINT_NAMES
from cv.pipeline.pose_player_crop import (
    coverage_report,
    listed_rows,
    native_box_for_row,
    prefetch_frames,
    prepare_resume_directory,
    read_csv_rows,
    read_frame_cached,
    resume_contract,
    select_rows,
    write_csv_rows_atomic,
)
from cv.pipeline.resolution import FrameSize


def test_native_player_crop_box_uses_coordinate_contract() -> None:
    row = {"x0": "100", "y0": "50", "x1": "500", "y1": "400"}

    box = native_box_for_row(
        row,
        FrameSize(1280, 720),
        FrameSize(1920, 1080),
    )

    np.testing.assert_allclose(box, [150.0, 75.0, 750.0, 600.0])


def test_native_player_crop_prefers_explicit_native_columns() -> None:
    row = {
        "x0": "100",
        "y0": "50",
        "x1": "500",
        "y1": "400",
        "x0_native": "200",
        "y0_native": "100",
        "x1_native": "1000",
        "y1_native": "800",
    }

    box = native_box_for_row(
        row,
        FrameSize(1920, 1080),
        FrameSize(1920, 1080),
    )

    np.testing.assert_allclose(box, [200.0, 100.0, 1000.0, 800.0])


def test_read_frame_cached_decodes_each_path_once(monkeypatch, tmp_path) -> None:
    calls = []
    expected = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_imread(path: str):
        calls.append(path)
        return expected

    monkeypatch.setattr(pose_player_crop.cv2, "imread", fake_imread)
    path = tmp_path / "frame.jpg"
    cache = {}

    assert read_frame_cached(path, cache) is expected
    assert read_frame_cached(path, cache) is expected
    assert calls == [str(path)]


def test_prefetch_frames_decodes_unique_paths_and_preserves_mapping(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_imread(path: str):
        calls.append(path)
        return path

    monkeypatch.setattr(pose_player_crop.cv2, "imread", fake_imread)
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"

    result = prefetch_frames([first, second, first], workers=2)

    assert set(calls) == {str(first), str(second)}
    assert result == {first: str(first), second: str(second)}


def test_resume_directory_reuses_only_matching_contract(tmp_path) -> None:
    path = tmp_path / "chunks"
    contract = {"schema": "test", "value": 1}

    prepare_resume_directory(path, contract, resume=True)
    prepare_resume_directory(path, contract, resume=True)

    with np.testing.assert_raises_regex(ValueError, "contract changed"):
        prepare_resume_directory(path, {"schema": "test", "value": 2}, resume=True)
    prepare_resume_directory(
        path,
        {"schema": "test", "value": 2},
        resume=True,
        reset_incompatible=True,
    )
    assert json.loads((path / "manifest.json").read_text())["value"] == 2


BOX_FIELDS = ["clip", "frame", "side", "x0", "y0", "x1", "y1", "conf", "court_x", "court_y"]


def _box(clip, frame, side="near", conf="0.9"):
    return {
        "clip": clip,
        "frame": frame,
        "side": side,
        "x0": "10",
        "y0": "10",
        "x1": "20",
        "y1": "40",
        "conf": conf,
        "court_x": "1.0",
        "court_y": "2.0",
    }


def _boxes_csv(path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _extraction_receipt(frames, clip, names) -> None:
    source = {"configuration": {"native": True, "preserve_source_fps": True, "fps": 25.0}}
    source["fingerprint"] = artifact_cache._digest_json(source)
    (frames / clip / "extraction_receipt.json").write_text(
        json.dumps(
            {
                "schema": "player_frame_extraction_v1",
                "identity": {
                    "point": {"pt": int(clip[2:])},
                    "source_fingerprint": source["fingerprint"],
                },
                "source_identity": source,
                "frames": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256((frames / clip / name).read_bytes()).hexdigest(),
                    }
                    for name in names
                ],
            }
        )
    )


def _scope_scene(tmp_path):
    """One clip whose box stream reaches beyond the decided active span."""
    boxes = tmp_path / "player_boxes_25_native_sided_v1.csv"
    rows = [
        _box("pt0001", "f_0000000010.jpg"),
        _box("pt0001", "f_0000000040.jpg", conf="0"),
        _box("pt0002", "f_0000000005.jpg", conf=""),
    ]
    _boxes_csv(boxes, rows)
    frames = tmp_path / "audit_frames_native_1080"
    for row in rows:
        (frames / row["clip"]).mkdir(parents=True, exist_ok=True)
        pose_player_crop.cv2.imwrite(
            str(frames / row["clip"] / row["frame"]), np.zeros((1080, 1920, 3), dtype=np.uint8)
        )
    for clip in ("pt0001", "pt0002"):
        _extraction_receipt(frames, clip, sorted(r["frame"] for r in rows if r["clip"] == clip))
    active = {"match/pt0001": {"active_spans": [[5, 20]]}}
    return boxes, frames, active


def test_retained_scope_keeps_actor_rows_outside_the_decided_active_spans(tmp_path) -> None:
    boxes, frames, active = _scope_scene(tmp_path)

    decided = select_rows(
        boxes,
        scope="active_play",
        match_id="match",
        clips=set(),
        active=active,
        frames_root=frames,
    )
    retained = select_rows(
        boxes,
        scope="retained_native",
        match_id="match",
        clips=set(),
        active=None,
        frames_root=frames,
    )

    # The decided scope drops the late row and the clip with no decision; the retained scope
    # keeps every original row, including the zero and undeclared confidences.
    assert [(row["clip"], row["frame"]) for row in decided] == [("pt0001", "f_0000000010.jpg")]
    assert [(row["clip"], row["frame"]) for row in retained] == [
        ("pt0001", "f_0000000010.jpg"),
        ("pt0001", "f_0000000040.jpg"),
        ("pt0002", "f_0000000005.jpg"),
    ]


def test_retained_scope_refuses_rows_naming_pictures_outside_the_inventory(tmp_path) -> None:
    boxes = tmp_path / "boxes.csv"
    _boxes_csv(boxes, [_box("pt0001", "../pt0002/f_0000000001.jpg")])

    with pytest.raises(ValueError, match="outside the frame inventory"):
        select_rows(
            boxes,
            scope="retained_native",
            match_id="match",
            clips=set(),
            active=None,
            frames_root=tmp_path / "audit_frames_native_1080",
        )


def _retained(boxes, frames):
    return select_rows(
        boxes,
        scope="retained_native",
        match_id="match",
        clips=set(),
        active=None,
        frames_root=frames,
    )


def test_retained_scope_reads_only_pictures_the_extraction_declared(tmp_path) -> None:
    boxes, frames, _ = _scope_scene(tmp_path)
    # A picture on disk that the original extraction never listed, and a clip whose receipt
    # this run does not hold at all.
    _extraction_receipt(frames, "pt0001", ["f_0000000010.jpg"])
    (frames / "pt0002" / "extraction_receipt.json").unlink()

    presented, inventory = listed_rows(_retained(boxes, frames), frames)

    assert [row["frame"] for row in presented] == ["f_0000000010.jpg"]
    assert inventory["rows_not_in_extraction_inventory"] == 1
    assert inventory["clips_with_unlisted_rows"] == {"pt0001": 1}
    assert inventory["rows_without_extraction_inventory"] == 1
    assert inventory["clips_without_extraction_inventory"] == ["pt0002"]


def test_retained_coverage_separates_unread_undetected_and_written_rows(tmp_path) -> None:
    boxes, frames, _ = _scope_scene(tmp_path)
    (frames / "pt0002" / "extraction_receipt.json").unlink()
    requested = _retained(boxes, frames)
    presented, inventory = listed_rows(requested, frames)

    report = coverage_report(
        requested,
        presented=presented,
        inventory=inventory,
        execution={"decode_failed": 1, "no_pose": 0},
        output_rows=[{f"{name}_confidence": 0.0 for name in KEYPOINT_NAMES}],
        scope="retained_native",
        boxes=boxes.name,
        frames_directory=frames.name,
        output="player_pose_optional_contact_native_v1.csv",
    )

    assert report["requested_rows"] == 3 and report["requested_frames"] == 3
    assert report["presented_rows"] == 2
    assert report["extraction_inventory"]["rows_without_extraction_inventory"] == 1
    assert report["picture_bytes_verified_against_inventory"] is True
    assert report["rows_zero_confidence"] == 1 and report["rows_unknown_confidence"] == 1
    # One written pose out of three requested rows, and that row carries no keypoint
    # confidence at all: nothing here claims three covered actors.
    assert report["decode_failed_rows"] == 1 and report["no_pose_rows"] == 0
    assert report["pose_rows"] == 1
    assert report["pose_rows_without_any_keypoint_confidence"] == 1
    assert report["rows_without_pose_output"] == 2


def test_pose_resume_chunks_do_not_cross_observation_scopes(tmp_path) -> None:
    boxes, frames, _ = _scope_scene(tmp_path)
    coordinates = boxes.with_suffix(boxes.suffix + ".coordinates.json")
    coordinates.write_text("{}")
    model = tmp_path / "yolo26m-pose.pt"
    model.write_bytes(b"weights")
    active_path = tmp_path / "active_play_v1.json"
    active_path.write_text("{}")

    receipts = sorted(frames.glob("*/extraction_receipt.json"))

    def contract(scope, active, inventory=()):
        arguments = argparse.Namespace(
            out=str(tmp_path),
            boxes=boxes.name,
            model=str(model),
            observation_scope=scope,
            active_play=None if active is None else str(active),
            resume=True,
        )
        return resume_contract(
            arguments,
            boxes_path=boxes,
            active_path=active,
            coordinate_path=coordinates,
            requested_rows=3,
            inventory_paths=list(inventory),
        )

    decided = contract("active_play", active_path)
    retained = contract("retained_native", None, receipts)
    assert "active_play" in decided["inputs"] and "active_play" not in retained["inputs"]
    assert len(retained["inputs"]["extraction_receipts"]) == len(receipts) == 2
    # A legacy interrupted run keeps resuming its own chunks: the default scope serializes
    # exactly as it did before the scope argument existed.
    assert "observation_scope" not in decided["arguments"]
    assert retained["arguments"]["observation_scope"] == "retained_native"

    chunks = tmp_path / ".chunks"
    prepare_resume_directory(chunks, decided, resume=True)
    with pytest.raises(ValueError, match="contract changed"):
        prepare_resume_directory(chunks, retained, resume=True)

    # A re-extracted point invalidates the retained chunks even though the box stream,
    # the model and the arguments are unchanged.
    prepare_resume_directory(chunks, retained, resume=True, reset_incompatible=True)
    changed = json.loads(receipts[0].read_text())
    changed["frames"][0]["sha256"] = "1" * 64
    receipts[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="contract changed"):
        prepare_resume_directory(chunks, contract("retained_native", None, receipts), resume=True)


def _producer_scene(monkeypatch, tmp_path, *, fail_on_call=None):
    """The real producer with only its model boundary and picture decode replaced."""
    import sys
    from types import ModuleType

    from cv.pipeline.resolution import write_coordinate_manifest

    boxes, frames, _ = _scope_scene(tmp_path)
    native = FrameSize(1920, 1080)
    write_coordinate_manifest(
        str(boxes) + ".coordinates.json",
        image_size=native,
        artifact_size=native,
        source=str(frames),
    )
    (tmp_path / "yolo26m-pose.pt").write_bytes(b"weights")
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = lambda path: path
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setattr(
        pose_player_crop.cv2,
        "imread",
        lambda path: (
            np.zeros((1080, 1920, 3), dtype=np.uint8)
            if pose_player_crop.Path(path).is_file()
            else None
        ),
    )
    calls = []

    def pose_windows(model, windows, **kwargs):
        calls.append(len(windows))
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("inference interrupted")
        return [
            {
                "box": [1.0, 2.0, 3.0, 4.0],
                "det_conf": 0.5,
                "n_variants": 2,
                "kpts_xy": [[1.0, 2.0]] * len(KEYPOINT_NAMES),
                "kpts_conf": [0.3] * len(KEYPOINT_NAMES),
            }
            for _ in windows
        ]

    monkeypatch.setattr(pose_player_crop, "pose_windows", pose_windows)
    return boxes, frames


def _producer_arguments(monkeypatch, tmp_path, boxes, *extra) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pose_player_crop",
            "--out",
            str(tmp_path),
            "--match-id",
            "match",
            "--boxes",
            boxes.name,
            "--model",
            str(tmp_path / "yolo26m-pose.pt"),
            "--observation-scope",
            "retained_native",
            "--output",
            "player_pose_optional_contact_native_v1.csv",
            *extra,
        ],
    )


def test_retained_scope_run_writes_its_own_artifact_and_coverage(monkeypatch, tmp_path) -> None:
    boxes, frames = _producer_scene(monkeypatch, tmp_path)
    (frames / "pt0002" / "f_0000000005.jpg").unlink()
    _producer_arguments(monkeypatch, tmp_path, boxes)

    assert pose_player_crop.main() == 0

    output = tmp_path / "player_pose_optional_contact_native_v1.csv"
    assert output.is_file() and (tmp_path / f"{output.name}.coordinates.json").is_file()
    assert not (tmp_path / "player_pose_tracked_crop_native_v1.csv").exists()
    written = read_csv_rows(output)
    # The two readable pictures are inferred, including the row outside the decided active
    # span; the declared picture that will not decode yields no row.
    assert [(row["clip"], row["frame"]) for row in written] == [
        ("pt0001", "f_0000000010.jpg"),
        ("pt0001", "f_0000000040.jpg"),
    ]
    coverage = json.loads(
        (tmp_path / "player_pose_optional_contact_native_v1.coverage.json").read_text()
    )
    assert coverage["requested_rows"] == 3 and coverage["presented_rows"] == 3
    assert coverage["decode_failed_rows"] == 1 and coverage["no_pose_rows"] == 0
    assert coverage["pose_rows"] == 2 and coverage["rows_without_pose_output"] == 1
    manifests = list((tmp_path / "run_manifests").glob("player_pose_retained_native_crop_*.json"))
    assert len(manifests) == 1
    outputs = json.loads(manifests[0].read_text())["outputs"]
    assert outputs["rows"] == 2 and outputs["requested_boxes"] == 3
    assert outputs["coverage_counts"]["decode_failed_rows"] == 1


def test_resumed_chunks_keep_the_outcome_of_the_pictures_they_read(monkeypatch, tmp_path) -> None:
    boxes, frames = _producer_scene(monkeypatch, tmp_path, fail_on_call=2)
    (frames / "pt0001" / "f_0000000010.jpg").unlink()
    _producer_arguments(monkeypatch, tmp_path, boxes, "--chunk", "1")

    with pytest.raises(RuntimeError, match="interrupted"):
        pose_player_crop.main()

    _producer_scene(monkeypatch, tmp_path)
    _producer_arguments(monkeypatch, tmp_path, boxes, "--chunk", "1")
    assert pose_player_crop.main() == 0

    coverage = json.loads(
        (tmp_path / "player_pose_optional_contact_native_v1.coverage.json").read_text()
    )
    # The undecodable picture was read by the first, resumed chunk: the completed run still
    # reports it instead of counting that chunk as clean.
    assert coverage["decode_failed_rows"] == 1
    assert coverage["resumed_chunks_without_counts"] == 0
    assert coverage["pose_rows"] == 2 and coverage["presented_rows"] == 3


def test_chunk_csv_round_trip_supports_empty_chunks(tmp_path) -> None:
    empty = tmp_path / "empty.csv"
    rows = tmp_path / "rows.csv"

    write_csv_rows_atomic(empty, ["frame", "side"], [])
    write_csv_rows_atomic(rows, ["frame", "side"], [{"frame": 3, "side": "near"}])

    assert read_csv_rows(empty) == []
    assert read_csv_rows(rows) == [{"frame": "3", "side": "near"}]


def test_native_inventory_refuses_wrong_clip_signature_and_duplicate_frame(tmp_path):
    _, frames, _ = _scope_scene(tmp_path)
    p = frames / "pt0001" / "extraction_receipt.json"
    original = json.loads(p.read_text())
    for mutate in (
        lambda d: d["identity"]["point"].update(pt=2),
        lambda d: d["source_identity"]["configuration"].update(fps=30),
        lambda d: d["frames"].append(d["frames"][0]),
    ):
        d = json.loads(json.dumps(original))
        mutate(d)
        p.write_text(json.dumps(d))
        assert pose_player_crop.extraction_inventory(frames, "pt0001") is None


def test_verified_native_decode_withholds_replaced_missing_and_corrupt_pictures(tmp_path):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[8:20, 12:30] = 220
    valid = tmp_path / "valid.jpg"
    pose_player_crop.cv2.imwrite(str(valid), image)
    replaced = tmp_path / "replaced.jpg"
    replaced.write_bytes(valid.read_bytes())
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not a jpeg")
    missing = tmp_path / "missing.jpg"
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [valid, replaced, corrupt]}
    digests[missing] = "0" * 64
    replaced.write_bytes(b"different source bytes")
    decoded, reasons = pose_player_crop.prefetch_verified_frames(
        [valid, valid, replaced, corrupt, missing], digests, workers=2
    )
    np.testing.assert_array_equal(decoded[valid], pose_player_crop.cv2.imread(str(valid)))
    assert decoded[replaced] is None and decoded[corrupt] is None and decoded[missing] is None
    assert reasons == {
        replaced: "source_hash_mismatch",
        corrupt: "decode_failed",
        missing: "decode_failed",
    }
