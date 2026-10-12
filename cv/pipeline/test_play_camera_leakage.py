from __future__ import annotations

import json
import cv2
import numpy as np
import pytest

from cv.pipeline import resolution as res
from cv.pipeline import play_camera_leakage as leakage

from cv.pipeline.play_camera_leakage import (
    COURT_LINES,
    FrameEvidence,
    PointReference,
    court_support,
    leak_reasons,
    leak_runs,
    point_reference,
    sample_court_points,
    snap_to_shots,
    _load_boxes,
)

PIXELS_PER_METER = 40.0


def test_match_snapshot_matches_point_loaders_and_proposals(tmp_path, monkeypatch):
    boxes = tmp_path / "player_boxes_25_native_sided_v1.csv"
    boxes.write_text(
        "clip,frame,side,x0,y0,x1,y1,conf\n"
        "pt0001,f_0001.jpg,near,10,20,30,40,0.5\n"
        "pt0001,f_0001.jpg,near,11,21,31,41,0.9\n"
        "pt0002,f_0002.jpg,far,100,200,300,400,0.8\n"
    )
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(boxes),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        source="fixture",
    )
    np.savez(tmp_path / "court_H_per_point.npz", pts=[1, 2], H=[np.eye(3), np.full((3, 3), np.nan)])
    np.savez(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=["pt0001", "pt0001", "pt0002"],
        frames=[1, 2, 2],
        source=["direct", "fallback", "direct"],
        reliable=[True, False, True],
    )
    snapshot = leakage.load_match_inputs(tmp_path)
    seen = []

    def evidence(_root, clip, _frames, boxes, camera):
        seen.append((clip, boxes, camera))
        return [_frame(1)]

    monkeypatch.setattr(leakage, "evidence_for_clip", evidence)
    for clip in ["pt0001", "pt0002", "pt0003"]:
        entry = {"fps": 25, "active_spans": [[1, 2]]}
        old = leakage.propose_for_point(tmp_path, clip, entry)[1]
        new = leakage.propose_for_point(tmp_path, clip, entry, inputs=snapshot)[1]
        assert old == new
        _, old_boxes, old_camera = seen[-2]
        _, new_boxes, new_camera = seen[-1]
        assert old_camera.sources == new_camera.sources
        assert old_camera.reliable_frames == new_camera.reliable_frames
        if old_camera.homography is None:
            assert new_camera.homography is None
        else:
            np.testing.assert_array_equal(old_camera.homography, new_camera.homography)
        for side in ["near", "far"]:
            assert old_boxes[side].keys() == new_boxes[side].keys()
            for frame, old_box in old_boxes[side].items():
                np.testing.assert_array_equal(old_box.box, new_boxes[side][frame].box)
                assert old_box.confidence == new_boxes[side][frame].confidence
    snapshot.assert_unchanged()
    with pytest.raises(ValueError, match="different match"):
        leakage.propose_for_point(tmp_path / "other", "pt0001", {}, inputs=snapshot)


@pytest.mark.parametrize("change", ["new_camera", "boxes", "sidecar", "new_selected_boxes"])
def test_match_snapshot_rejects_concurrent_input_changes(tmp_path, change):
    path = tmp_path / "player_boxes_25_native_sided_v1.csv"
    path.write_text("clip,frame,side,x0,y0,x1,y1,conf\n")
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        source="fixture",
    )
    snapshot = leakage.load_match_inputs(tmp_path)
    if change == "new_camera":
        np.savez(tmp_path / "court_H_per_point.npz", pts=[1], H=[np.eye(3)])
    elif change == "boxes":
        path.write_text(path.read_text() + "pt0001,f_0001.jpg,near,1,2,3,4,0.9\n")
    elif change == "sidecar":
        sidecar = res.coordinate_manifest_path(path)
        payload = json.loads(sidecar.read_text())
        payload["source"] = "changed"
        sidecar.write_text(json.dumps(payload))
    else:
        (tmp_path / "player_boxes_00_native_sided_v1.csv").write_text("new file")
    with pytest.raises(ValueError, match="changed"):
        snapshot.assert_unchanged()


@pytest.mark.parametrize("schema", ["dual", "legacy", "native_only", "declared_legacy_mirror"])
@pytest.mark.parametrize("image_size", [res.NATIVE_SIZE, res.FrameSize(3840, 2160)])
def test_player_columns_and_sidecar_size_are_resolved_together(tmp_path, schema, image_size):
    path = tmp_path / "boxes.csv"
    legacy = "x0,y0,x1,y1"
    native = "x0_native,y0_native,x1_native,y1_native"
    if schema == "dual":
        columns, values = f"{legacy},{native}", "100,200,150,300,200,400,300,600"
    elif schema == "native_only":
        columns, values = native, "200,400,300,600"
    else:
        columns, values = legacy, "100,200,150,300"
    path.write_text(f"clip,frame,side,{columns},conf\npt0001,f_0001.jpg,near,{values},0.9\n")
    if schema == "legacy":
        res.write_coordinate_manifest(
            res.coordinate_manifest_path(path),
            image_size=image_size,
            artifact_size=res.LEGACY_TRACKING_SIZE,
            source="test",
            subnative_flagged=True,
            subnative_justification="legacy coordinate test fixture",
        )
    else:
        res.write_native_dual_coordinate_manifest(
            res.coordinate_manifest_path(path),
            image_size=image_size,
            legacy_size=res.LEGACY_TRACKING_SIZE,
            source="test",
            native_columns=native.split(","),
            legacy_columns=legacy.split(","),
        )
    observed = _load_boxes(path, "pt0001")["near"][1].box
    expected = res.scale_boxes([[200, 400, 300, 600]], res.NATIVE_SIZE, image_size)[0]
    np.testing.assert_allclose(observed, expected)


def _frame(
    frame: int,
    support: float | None = 0.6,
    near: float | None = 1.8,
    far: float | None = 1.7,
    readable: bool = True,
    samples: int = 60,
) -> FrameEvidence:
    return FrameEvidence(
        frame=frame,
        court_support=support,
        court_samples=samples,
        frame_readable=readable,
        apparent_near_m=near,
        apparent_far_m=far,
        near_present=near is not None,
        far_present=far is not None,
        side_conflict=False,
    )


def test_court_support_separates_lines_from_blank():
    homography = np.array(
        [
            [1 / PIXELS_PER_METER, 0.0, -50 / PIXELS_PER_METER],
            [0.0, 1 / PIXELS_PER_METER, -50 / PIXELS_PER_METER],
            [0.0, 0.0, 1.0],
        ]
    )
    width, height = 550, 1050
    image = np.full((height, width), 90, dtype=np.uint8)
    samples = sample_court_points(homography, width, height)
    assert len(samples) > 100
    blank_support, _ = court_support(image, samples)
    assert blank_support < 0.05
    for (x0, y0), (x1, y1) in COURT_LINES:
        start = (int(x0 * PIXELS_PER_METER) + 50, int(y0 * PIXELS_PER_METER) + 50)
        end = (int(x1 * PIXELS_PER_METER) + 50, int(y1 * PIXELS_PER_METER) + 50)
        cv2.line(image, start, end, 255, 3)
    lined_support, _ = court_support(image, samples)
    assert lined_support > 0.8


def test_hard_cut_to_closeup_is_trimmed():
    rows = [_frame(frame) for frame in range(100)]
    rows += [_frame(frame, support=0.05, near=6.0, far=None) for frame in range(100, 140)]
    runs = leak_runs(rows, 25.0, point_reference(rows))
    assert len(runs) == 1
    assert (runs[0]["start_frame"], runs[0]["end_frame"]) == (100, 139)
    assert runs[0]["reasons"] == ["court_support_lost", "giant_player_box"]


def test_transient_box_enlargement_is_not_trimmed():
    rows = [_frame(frame) for frame in range(100)]
    rows[50] = _frame(50, near=4.5)
    rows[51] = _frame(51, near=4.2)
    assert leak_runs(rows, 25.0, point_reference(rows)) == []


def test_missing_or_unhealthy_court_evidence_abstains():
    low_support = [_frame(frame, support=0.03) for frame in range(200)]
    assert not point_reference(low_support).court_arm_enabled
    assert leak_runs(low_support, 25.0, point_reference(low_support)) == []

    unreadable = [_frame(frame, support=None, readable=False) for frame in range(200)]
    assert leak_runs(unreadable, 25.0, point_reference(unreadable)) == []


def test_valid_serve_toss_is_not_trimmed():
    rows = [_frame(frame) for frame in range(60)]
    rows += [_frame(frame, near=2.5, support=0.55) for frame in range(60, 120)]
    reference = point_reference(rows)
    assert reference.box_arm_enabled
    assert leak_runs(rows, 25.0, reference) == []


def test_broken_height_reference_abstains():
    rows = [_frame(frame, near=5.0, far=4.8, support=0.5) for frame in range(100)]
    reference = point_reference(rows)
    assert not reference.box_arm_enabled
    assert all("giant_player_box" not in leak_reasons(row, reference) for row in rows)


def test_gap_bridging_keeps_one_run():
    rows = [_frame(frame) for frame in range(50)]
    leak = [_frame(frame, support=0.05) for frame in range(50, 80)]
    leak[10] = _frame(60)
    rows += leak
    rows += [_frame(frame) for frame in range(80, 130)]
    runs = leak_runs(rows, 25.0, point_reference(rows))
    assert len(runs) == 1
    assert (runs[0]["start_frame"], runs[0]["end_frame"]) == (50, 79)


def test_snap_to_shot_boundaries():
    shots = [
        {"start_frame": 1, "end_frame": 98},
        {"start_frame": 102, "end_frame": 300},
    ]
    run = {
        "start_frame": 100,
        "end_frame": 205,
        "frames": 106,
        "density": 1.0,
        "reasons": [],
    }
    snapped = snap_to_shots(run, shots)
    assert snapped["start_frame"] == 102
    assert snapped["end_frame"] == 205


def test_empty_reference_disables_both_arms():
    reference = point_reference([])
    assert isinstance(reference, PointReference)
    assert not reference.court_arm_enabled
    assert not reference.box_arm_enabled
