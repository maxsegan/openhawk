import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cv.pipeline.active_play import (
    NET_GUARDED_PHASE_PADDING_SECONDS,
    ActivePlayResult,
    resolve_active_play,
)
from cv.pipeline.active_play_gate import (
    load_leakage_proposals,
    load_track_artifact,
    main,
    net_rows_for_track,
)
from cv.pipeline import resolution as res
from cv.pipeline.broadcast_runner import write_runner_parent_provenance


def test_net_rows_use_observed_cord_from_camera_artifact(tmp_path: Path) -> None:
    track_path = tmp_path / "track.csv"
    track_path.write_text("clip,frame,x,y\n")
    (tmp_path / "track.csv.coordinates.json").write_text(
        '{"schema":"tennis.coordinate-space.v1","image_size":{"width":960,"height":540},'
        '"artifact_size":{"width":960,"height":540}}'
    )
    projection = np.asarray(
        [[100.0, 0.0, 0.0, 300.0], [0.0, 20.0, -80.0, 362.0], [0.0, 0.0, 0.0, 1.0]]
    )
    cord = np.stack([np.linspace(100.0, 900.0, 9), np.linspace(250.0, 258.0, 9)], axis=1)
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=np.asarray(["pt0001"]),
        frames=np.asarray([10]),
        P=np.asarray([projection]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
        net_cord_source=np.asarray(["observed_connected_tape_segments"]),
    )
    track = np.asarray([[10.0, 500.0, 200.0]])

    rows = net_rows_for_track(tmp_path, "pt0001", track_path, track)

    np.testing.assert_allclose(rows, [254.0])


def test_native_ball_and_cord_use_identical_phase_units(tmp_path):
    path = tmp_path / "track.csv"
    # Deliberately inconsistent mirrors establish that native columns are authoritative.
    path.write_text("clip,frame,x,y,x_native,y_native\npt0001,f_0010.jpg,1,1,1000,400\n")
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        legacy_size=res.LEGACY_TRACKING_SIZE,
        source="test",
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )
    cord = np.stack([np.linspace(200, 1800, 9), np.linspace(500, 516, 9)], axis=1)
    projection = np.asarray([[200.0, 0, 0, 600], [0, 40, -160, 724], [0, 0, 0, 1]])
    np.savez_compressed(
        tmp_path / "camera_P_per_frame_v1.npz",
        clips=np.asarray(["pt0001"]),
        frames=np.asarray([10]),
        P=np.asarray([projection]),
        net_cord_xy=np.asarray([cord]),
        net_cord_valid=np.asarray([True]),
    )
    track = load_track_artifact(path)["pt0001"]
    np.testing.assert_allclose(track, [[10, 500, 200]])
    np.testing.assert_allclose(net_rows_for_track(tmp_path, "pt0001", path, track), [254.0])


def test_phase_track_requires_a_declared_coordinate_space(tmp_path):
    path = tmp_path / "track.csv"
    path.write_text("clip,frame,x,y\n")
    with pytest.raises(ValueError, match="missing track coordinate manifest"):
        load_track_artifact(path)


def _fake_frames(tmp_path: Path, count: int) -> list[Path]:
    frame_dir = tmp_path / "pt0001"
    frame_dir.mkdir()
    frames = []
    for frame in range(count):
        path = frame_dir / f"f_{frame:06d}.jpg"
        path.touch()
        frames.append(path)
    return frames


def test_baseline_profile_ignores_optional_net_rows(monkeypatch, tmp_path: Path) -> None:
    frames = _fake_frames(tmp_path, 20)
    track = np.column_stack([np.arange(20, dtype=float), np.zeros(20), np.arange(20, dtype=float)])

    class Segmentation:
        shots = [{"start_frame": 0, "end_frame": 19, "is_play_camera": True}]

    class Phase:
        point_valid = True
        spans = [(4.0, 12.0)]
        reason = "ok"

    calls = []
    monkeypatch.setattr(
        "cv.pipeline.active_play.segment_shots", lambda *args, **kwargs: Segmentation()
    )
    monkeypatch.setattr(
        "cv.pipeline.active_play._extend_final_span_to_track",
        lambda spans, *args, **kwargs: spans,
    )

    def fake_detect(frames, points, net_row, **kwargs):
        calls.append(net_row)
        return Phase()

    monkeypatch.setattr("cv.pipeline.active_play.detect_phase", fake_detect)
    result = resolve_active_play(
        frames,
        track,
        fps=25.0,
        net_row=np.zeros(20),
        phase_profile="baseline",
    )

    assert calls == [None]
    assert result.active_spans == [(4.0, 12.0)]


def test_net_guarded_profile_pads_accepted_phase(monkeypatch, tmp_path: Path) -> None:
    frames = _fake_frames(tmp_path, 30)
    track = np.column_stack([np.arange(30, dtype=float), np.zeros(30), np.arange(30, dtype=float)])

    class Segmentation:
        shots = [{"start_frame": 0, "end_frame": 29, "is_play_camera": True}]

    class Phase:
        point_valid = True
        spans = [(10.0, 20.0)]
        reason = "ok"

    monkeypatch.setattr(
        "cv.pipeline.active_play.segment_shots", lambda *args, **kwargs: Segmentation()
    )
    monkeypatch.setattr(
        "cv.pipeline.active_play._extend_final_span_to_track",
        lambda spans, *args, **kwargs: spans,
    )
    monkeypatch.setattr("cv.pipeline.active_play.detect_phase", lambda *args, **kwargs: Phase())
    result = resolve_active_play(
        frames,
        track,
        fps=25.0,
        net_row=np.zeros(30),
        phase_profile="net_guarded_v1",
    )

    padding = NET_GUARDED_PHASE_PADDING_SECONDS * 25.0
    assert result.active_spans == [(10.0 - padding, 20.0 + padding)]


def test_explicit_leakage_proposals_preserve_source_paths(tmp_path: Path) -> None:
    path = tmp_path / "match__play_camera_leakage_v1.json"
    path.write_text(
        '{"schema":"play_camera_leakage_v1","match_id":"match",'
        '"points":{"pt0001":{"proposed_trims":[]}}}'
    )

    proposals, paths = load_leakage_proposals(tmp_path)

    assert proposals["match/pt0001"]["schema"] == "play_camera_leakage_v1"
    assert paths == [path]


def test_explicit_leakage_proposals_reject_wrong_schema(tmp_path: Path) -> None:
    (tmp_path / "match__play_camera_leakage_v1.json").write_text(
        '{"schema":"reviewed_leakage","match_id":"match","points":{}}'
    )

    with pytest.raises(RuntimeError, match="unsupported leakage proposal schema"):
        load_leakage_proposals(tmp_path)


def test_explicit_leakage_proposals_reject_duplicate_match(tmp_path: Path) -> None:
    document = '{"schema":"play_camera_leakage_v1","match_id":"match","points":{}}'
    (tmp_path / "first__play_camera_leakage_v1.json").write_text(document)
    (tmp_path / "second__play_camera_leakage_v1.json").write_text(document)

    with pytest.raises(RuntimeError, match="duplicate leakage proposal match_id"):
        load_leakage_proposals(tmp_path)


def _runner_shaped_processed(tmp_path: Path) -> Path:
    """One runner-shaped processed cohort with its parent automatic record."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    model = tmp_path / "person.pt"
    model.write_bytes(b"model")
    processed = tmp_path / "processed"
    match_dir = processed / "match"
    frame_dir = match_dir / "audit_frames_native_1080" / "pt0001"
    frame_dir.mkdir(parents=True)
    (frame_dir / "f_0001.jpg").touch()
    (frame_dir / "f_0002.jpg").touch()
    (match_dir / "audit_frames_native_1080.coordinates.json").write_text('{"fps":25}\n')
    (match_dir / "ball_track_joint_native1080_arc_augmented_v2.csv").write_text(
        "clip,frame,x,y\npt0001,f_0001.jpg,10,10\n"
    )
    res.write_coordinate_manifest(
        res.coordinate_manifest_path(
            match_dir / "ball_track_joint_native1080_arc_augmented_v2.csv"
        ),
        image_size=res.NATIVE_SIZE,
        artifact_size=res.NATIVE_SIZE,
        source="test",
    )
    write_runner_parent_provenance(
        video=source,
        out_root=processed,
        match_id="match",
        surface="hard",
        person_model=model,
    )
    return processed


def test_point_gate_runs_on_runner_shaped_output(tmp_path: Path, monkeypatch) -> None:
    """The raw-broadcast runner writes the parent automatic record before point gates."""
    processed = _runner_shaped_processed(tmp_path)

    class Decision:
        trim = None
        trimmed_fraction = 0.0

        @staticmethod
        def as_dict() -> dict:
            return {
                "point_valid": True,
                "active_spans": [[1.0, 2.0]],
                "event_spans": [[1.0, 2.0]],
                "fps": 25.0,
                "n_frames": 2,
                "reasons": [],
            }

    monkeypatch.setattr(
        "cv.pipeline.active_play_gate.resolve_active_play", lambda *args, **kwargs: Decision()
    )
    monkeypatch.setattr(
        "cv.pipeline.active_play_gate.propose_for_point",
        lambda *args, **kwargs: (None, {"proposed_trims": []}),
    )
    output = processed / "active_play_v1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["active_play_gate", "--processed", str(processed), "--out", str(output)],
    )

    main()

    assert json.loads(output.read_text())["match/pt0001"]["point_valid"] is True
    assert output.with_name("active_play_v1.provenance.json").is_file()


def test_default_segmentation_call_keeps_its_current_arguments(monkeypatch, tmp_path: Path) -> None:
    """An unselected run must still reach segment_shots with its existing arguments."""
    frames = _fake_frames(tmp_path, 20)
    track = np.column_stack([np.arange(20, dtype=float), np.zeros(20), np.arange(20, dtype=float)])

    class Segmentation:
        shots = [{"start_frame": 0, "end_frame": 19, "is_play_camera": True}]

    class Phase:
        point_valid = True
        spans = [(4.0, 12.0)]
        reason = "ok"

    def old_signature_segment_shots(paths, *, fps, stride):
        return Segmentation()

    monkeypatch.setattr("cv.pipeline.active_play.segment_shots", old_signature_segment_shots)
    monkeypatch.setattr("cv.pipeline.active_play.detect_phase", lambda *a, **kw: Phase())

    result = resolve_active_play(frames, track, fps=25.0)

    assert result.boundary_continuity == []
    assert "native_cut_continuity" not in result.as_dict()


def test_selected_native_cut_continuity_forwards_and_publishes_receipts(
    monkeypatch, tmp_path: Path
) -> None:
    frames = _fake_frames(tmp_path, 20)
    track = np.column_stack([np.arange(20, dtype=float), np.zeros(20), np.arange(20, dtype=float)])
    receipts = [
        {
            "schema": "native_shot_boundary_continuity_v1",
            "before_frame": 8,
            "after_frame": 12,
            "continuous": True,
        }
    ]

    class Segmentation:
        shots = [{"start_frame": 0, "end_frame": 19, "is_play_camera": True}]
        boundary_continuity = receipts

    class Phase:
        point_valid = True
        spans = [(4.0, 12.0)]
        reason = "ok"

    seen = {}

    def fake_segment_shots(paths, **kwargs):
        seen.update(kwargs)
        return Segmentation()

    monkeypatch.setattr("cv.pipeline.active_play.segment_shots", fake_segment_shots)
    monkeypatch.setattr(
        "cv.pipeline.active_play._extend_final_span_to_track",
        lambda spans, *args, **kwargs: spans,
    )
    monkeypatch.setattr("cv.pipeline.active_play.detect_phase", lambda *a, **kw: Phase())

    result = resolve_active_play(frames, track, fps=25.0, native_cut_continuity=True)

    assert seen["native_cut_continuity"] is True
    entry = result.as_dict()
    assert entry["native_cut_continuity"] is True
    assert entry["boundary_continuity"] == receipts
    # The option revises segmentation only; the trim decision stays the existing one.
    assert result.active_spans == [(4.0, 12.0)]


def _run_gate(tmp_path: Path, monkeypatch, extra_arguments: list[str]) -> tuple[dict, dict]:
    """Run the gate command once, returning its entries and declared configuration."""
    processed = _runner_shaped_processed(tmp_path)
    selected = []

    def fake_resolve(*args, native_cut_continuity: bool = False, **kwargs):
        selected.append(native_cut_continuity)
        return ActivePlayResult(
            clip="pt0001",
            fps=25.0,
            n_frames=2,
            active_spans=[(1.0, 2.0)],
            event_spans=[(1.0, 2.0)],
            native_cut_continuity=native_cut_continuity,
            boundary_continuity=(
                [{"schema": "native_shot_boundary_continuity_v1", "continuous": False}]
                if native_cut_continuity
                else []
            ),
        )

    monkeypatch.setattr("cv.pipeline.active_play_gate.resolve_active_play", fake_resolve)
    monkeypatch.setattr(
        "cv.pipeline.active_play_gate.propose_for_point",
        lambda *args, **kwargs: (None, {"proposed_trims": []}),
    )
    output = processed / "active_play_v1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["active_play_gate", "--processed", str(processed), "--out", str(output), *extra_arguments],
    )

    main()

    provenance = json.loads(output.with_name("active_play_v1.provenance.json").read_text())
    entries = json.loads(output.read_text())
    assert selected == [bool(extra_arguments)]
    return entries, provenance["configuration"]


def test_gate_defaults_leave_native_cut_continuity_unselected(tmp_path: Path, monkeypatch) -> None:
    entries, configuration = _run_gate(tmp_path, monkeypatch, [])

    assert configuration["native_cut_continuity"] is False
    assert "native_cut_continuity" not in entries["match/pt0001"]
    assert "boundary_continuity" not in entries["match/pt0001"]


def test_gate_flag_selects_native_cut_continuity(tmp_path: Path, monkeypatch) -> None:
    entries, configuration = _run_gate(tmp_path, monkeypatch, ["--native-cut-continuity"])

    assert configuration["native_cut_continuity"] is True
    assert entries["match/pt0001"]["native_cut_continuity"] is True
    assert entries["match/pt0001"]["boundary_continuity"] == [
        {"schema": "native_shot_boundary_continuity_v1", "continuous": False}
    ]
