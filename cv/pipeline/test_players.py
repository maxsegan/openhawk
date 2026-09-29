import csv
import json
from pathlib import Path

import numpy as np
import pytest

import resolution as res
from players import (
    extract_clip_frames,
    frame_path_number,
    native_extract_command,
    native_frame_window,
    player_detection_row,
    valid_jpeg_frame,
    write_player_artifact,
)


def _video(tmp_path: Path) -> str:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"test video identity")
    return str(path)


@pytest.mark.parametrize(
    "change", ["none", "window", "source_contents", "source_path", "config", "output", "receipt"]
)
def test_frame_reuse_requires_source_window_config_and_output_identity(
    tmp_path, monkeypatch, change
):
    video = _video(tmp_path)
    root = tmp_path / "frames"
    point_dir = root / "pt0001"
    calls = []

    def extract(command, check):
        calls.append(command)
        for index in (1, 2):
            (point_dir / f"f_{index:04d}.jpg").write_bytes(b"\xff\xd8" + bytes([len(calls), index]))

    monkeypatch.setattr("players.subprocess.run", extract)
    points = [{"pt": 1, "t0": 0.0, "t1": 0.2}]
    first = extract_clip_frames(video, str(root), points, 10.0, preserve_source_fps=True)
    native = True
    if change == "window":
        points = [{"pt": 1, "t0": 10.0, "t1": 10.2}]
    elif change == "source_contents":
        Path(video).write_bytes(b"changed video identity")
    elif change == "source_path":
        video = str(tmp_path / "other.mp4")
        Path(video).write_bytes(b"other video")
    elif change == "config":
        native = False
    elif change == "output":
        Path(first[0][1]).write_bytes(b"\xff\xd8different but still JPEG magic")
    elif change == "receipt":
        (point_dir / "extraction_receipt.json").write_text("not json")
    second = extract_clip_frames(
        video, str(root), points, 10.0, native=native, preserve_source_fps=True
    )
    assert len(calls) == (1 if change == "none" else 2)
    assert second[0][2] == points[0]["t0"]
    if change == "window":
        assert Path(second[0][1]).read_bytes() != b"\xff\xd8\x01\x01"


def test_existing_valid_jpegs_without_receipt_are_not_relabelled_with_new_times(
    tmp_path, monkeypatch
):
    root = tmp_path / "frames"
    point_dir = root / "pt0001"
    point_dir.mkdir(parents=True)
    for index in (1, 2):
        (point_dir / f"f_{index:04d}.jpg").write_bytes(b"\xff\xd8old window")
    calls = []

    def extract(command, check):
        calls.append(command)
        for index in (1, 2):
            (point_dir / f"f_{index:04d}.jpg").write_bytes(b"\xff\xd8new window")

    monkeypatch.setattr("players.subprocess.run", extract)
    result = extract_clip_frames(
        _video(tmp_path),
        str(root),
        [{"pt": 1, "t0": 10.0, "t1": 10.2}],
        10.0,
        preserve_source_fps=True,
    )
    assert len(calls) == 1
    assert Path(result[0][1]).read_bytes() == b"\xff\xd8new window"


def test_frame_paths_sort_numerically_past_four_digits() -> None:
    paths = ["/frames/f_9999.jpg", "/frames/f_10000.jpg", "/frames/f_0001.jpg"]

    assert sorted(paths, key=frame_path_number) == [
        "/frames/f_0001.jpg",
        "/frames/f_9999.jpg",
        "/frames/f_10000.jpg",
    ]


def test_corrupt_resumed_frames_are_regenerated(monkeypatch, tmp_path: Path) -> None:
    point_dir = tmp_path / "frames" / "pt0001"
    point_dir.mkdir(parents=True)
    for frame in range(1, 11):
        (point_dir / f"f_{frame:04d}.jpg").write_bytes(b"\x00" * 32)
    calls = []

    def regenerate(command, check):
        calls.append(command)
        assert check is True
        for frame in range(1, 11):
            (point_dir / f"f_{frame:04d}.jpg").write_bytes(b"\xff\xd8frame")

    monkeypatch.setattr("players.subprocess.run", regenerate)

    index = extract_clip_frames(
        _video(tmp_path),
        str(tmp_path / "frames"),
        [{"pt": 1, "t0": 0.0, "t1": 1.0}],
        10.0,
        preserve_source_fps=True,
    )

    assert len(calls) == 1
    assert len(index) == 10
    assert all(valid_jpeg_frame(path) for _, path, _ in index)


def test_native_frame_window_partitions_fractional_broadcast_cadence() -> None:
    fps = 60000 / 1001
    boundaries = (0.0, 10.010, 26.009, 51.018, 66.016)

    windows = [
        native_frame_window(start, end, fps)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]

    assert windows == [
        (0, 599),
        (600, 1558),
        (1559, 3057),
        (3058, 3956),
    ]
    assert all(left[1] + 1 == right[0] for left, right in zip(windows, windows[1:]))


def test_native_frame_window_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        native_frame_window(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        native_frame_window(1.0, 1.0, 25.0)


def test_native_extract_command_seeks_and_limits_decode() -> None:
    command, start_seconds, frame_count = native_extract_command(
        "video.mp4", "frames/f_%04d.jpg", 4230, 4324, 25.0, native=True
    )

    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "169.2"
    assert command[command.index("-frames:v") + 1] == "95"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert start_seconds == 169.2
    assert frame_count == 95


def test_native_extraction_accepts_contiguous_terminal_source_truncation(
    monkeypatch, tmp_path: Path
) -> None:
    point_dir = tmp_path / "frames" / "pt0001"
    dispositions = []

    def truncated_extract(command, check):
        assert check is True
        point_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(1, 9):
            (point_dir / f"f_{frame:04d}.jpg").touch()

    monkeypatch.setattr("players.subprocess.run", truncated_extract)
    index = extract_clip_frames(
        _video(tmp_path),
        str(tmp_path / "frames"),
        [{"pt": 1, "t0": 9.0, "t1": 10.2}],
        10.0,
        preserve_source_fps=True,
        video_duration=9.8,
        dispositions=dispositions,
    )

    assert len(index) == 8
    assert dispositions == [
        {
            "point": 1,
            "decision": "terminal_source_truncation",
            "requested_frames": 12,
            "observed_frames": 8,
            "requested_end_seconds": 10.2,
            "source_duration_seconds": 9.8,
        }
    ]


def test_native_extraction_rejects_nonterminal_short_decode(monkeypatch, tmp_path: Path) -> None:
    point_dir = tmp_path / "frames" / "pt0001"

    def truncated_extract(command, check):
        point_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(1, 9):
            (point_dir / f"f_{frame:04d}.jpg").touch()

    monkeypatch.setattr("players.subprocess.run", truncated_extract)
    with pytest.raises(RuntimeError, match="8/12 frames"):
        extract_clip_frames(
            _video(tmp_path),
            str(tmp_path / "frames"),
            [
                {"pt": 1, "t0": 1.0, "t1": 2.2},
                {"pt": 2, "t0": 3.0, "t1": 4.0},
            ],
            10.0,
            preserve_source_fps=True,
            video_duration=4.0,
        )


def test_native_extraction_rejects_short_decode_before_source_end(
    monkeypatch, tmp_path: Path
) -> None:
    point_dir = tmp_path / "frames" / "pt0001"

    def truncated_extract(command, check):
        point_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(1, 4):
            (point_dir / f"f_{frame:04d}.jpg").touch()

    monkeypatch.setattr("players.subprocess.run", truncated_extract)
    with pytest.raises(RuntimeError, match="3/22 frames"):
        extract_clip_frames(
            _video(tmp_path),
            str(tmp_path / "frames"),
            [{"pt": 1, "t0": 8.0, "t1": 10.2}],
            10.0,
            preserve_source_fps=True,
            video_duration=9.8,
        )


def test_player_detection_row_dual_emits_without_changing_legacy_box() -> None:
    row = player_detection_row(
        point=1,
        timestamp=0.123,
        frame_path="/tmp/pt0001/f_0001.jpg",
        box=np.asarray([200.0, 100.0, 800.0, 900.0]),
        confidence=0.91234,
        image_size=res.NATIVE_SIZE,
    )

    assert [row[key] for key in ("x0", "y0", "x1", "y1")] == [100.0, 50.0, 400.0, 450.0]
    assert [row[key] for key in ("x0_native", "y0_native", "x1_native", "y1_native")] == [
        200.0,
        100.0,
        800.0,
        900.0,
    ]


def test_native_player_box_uses_detector_precision_not_rounded_legacy_values() -> None:
    row = player_detection_row(
        point=1,
        timestamp=0.0,
        frame_path="/tmp/pt0001/f_0001.jpg",
        box=np.asarray([200.06, 100.06, 800.06, 900.06]),
        confidence=0.9,
        image_size=res.NATIVE_SIZE,
    )

    assert row["x0"] == 100.0
    assert row["x0_native"] == 200.1


def test_player_writer_declares_native_columns(tmp_path) -> None:
    path = tmp_path / "players.csv"
    row = player_detection_row(
        point=1,
        timestamp=0.0,
        frame_path="/tmp/pt0001/f_0001.jpg",
        box=np.asarray([200.0, 100.0, 800.0, 900.0]),
        confidence=0.9,
        image_size=res.NATIVE_SIZE,
    )

    write_player_artifact(
        str(path),
        [row],
        image_size=res.NATIVE_SIZE,
        legacy_size=res.CANONICAL_SIZE,
        source="frames",
    )

    written = next(csv.DictReader(path.open(newline="")))
    assert written["x0"] == "100.0"
    assert written["x0_native"] == "200.0"
    sidecar = json.loads((tmp_path / "players.csv.coordinates.json").read_text())
    assert sidecar["artifact_size"] == {"width": 1920, "height": 1080}
    assert sidecar["artifact_identity"] == res.PLAYER_BOXES_NATIVE_IDENTITY
    assert sidecar["coordinate_columns"]["legacy_960x540"] == ["x0", "y0", "x1", "y1"]

    with path.open(newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert fields == [
        "pt",
        "t",
        "clip",
        "frame",
        "x0",
        "y0",
        "x1",
        "y1",
        "conf",
        "x0_native",
        "y0_native",
        "x1_native",
        "y1_native",
    ]
