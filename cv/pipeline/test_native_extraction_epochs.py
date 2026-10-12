"""Actual FFmpeg picture/clock parity and rejection of fabricated/reindexed epochs."""

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from cv.pipeline import players, provenance
from cv.pipeline.s6_automatic_observations import _native_inventory
from cv.pipeline.s6_broadcast_backend import source_pts_cache
from cv.pipeline.source_timebase import frame_pts


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))


def _video(tmp_path, offset):
    path = tmp_path / f"source{offset}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=25:duration=1.6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            "-output_ts_offset",
            str(offset),
            str(path),
        ],
        check=True,
    )
    return path


def _extraction(tmp_path, offset):
    video = _video(tmp_path, offset)
    root = tmp_path / "captured"
    indices = players.extract_clip_frames(
        str(video),
        str(root),
        [{"pt": 1, "t0": 0.4, "t1": 1.0}],
        25,
        preserve_source_fps=True,
        measured_native_epochs=True,
    )
    frames = root / "pt0001"
    receipt = frames / "extraction_receipt.json"
    pts, _ = source_pts_cache(video, 25, tmp_path / "clock")
    inputs = SimpleNamespace(
        extraction_receipt=receipt,
        source_video=video,
        source_pts=pts,
        frames_directory=frames,
        clip="pt0001",
    )
    ledger = dict(pt=1, rally_t_start=0.4, rally_t_end=1.0)
    return indices, receipt, inputs, ledger


def test_zero_origin_measured_epochs_preserve_every_jpeg_and_legacy_mapping(tmp_path):
    indices, receipt, inputs, ledger = _extraction(tmp_path, 0)
    legacy = players.extract_clip_frames(
        str(inputs.source_video),
        str(tmp_path / "legacy"),
        [{"pt": 1, "t0": 0.4, "t1": 1.0}],
        25,
        preserve_source_fps=True,
    )
    assert [provenance.file_sha256(Path(r[1])) for r in indices] == [
        provenance.file_sha256(Path(r[1])) for r in legacy
    ]
    images = _native_inventory(inputs, 25, ledger)
    assert [r["source_frame_index"] for r in images] == list(range(11, 26))
    assert [row[2] for row in indices] == [r["native_pts_seconds"] for r in images]
    old = json.loads((tmp_path / "legacy/pt0001/extraction_receipt.json").read_text())
    with inputs.source_pts.open() as handle:
        epochs = list(frame_pts(handle))
    assert players.source_clock_ordinals(old, epochs, 25) == list(range(10, 25))


def test_nonzero_origin_uses_measured_source_ordinals_and_actual_player_times(tmp_path):
    indices, receipt, inputs, ledger = _extraction(tmp_path, 0.08)
    images = _native_inventory(inputs, 25, ledger)
    # Source picture0 is at.08; the selected .4s picture is ordinal8, not nominal10.
    assert images[0]["source_frame_index"] == 9
    assert images[0]["native_pts_seconds"] == 0.4
    assert indices[0][2] == 0.4
    document = json.loads(receipt.read_text())
    with inputs.source_pts.open() as handle:
        epochs = list(frame_pts(handle))
    legacy = deepcopy(document)
    del legacy["native_clock"]
    del legacy["source_identity"]["configuration"]["native_epoch_capture"]
    with pytest.raises(ValueError, match="nonzero-origin"):
        players.source_clock_ordinals(legacy, epochs, 25)
    for change in ("shift", "noncontiguous", "missing", "names"):
        wrong = deepcopy(document)
        if change == "shift":
            for row in wrong["native_clock"]["frames"]:
                row["source_pts"] += 512  # one actual native picture tick at25fps
        elif change == "noncontiguous":
            wrong["native_clock"]["frames"][3]["source_pts"] = wrong["native_clock"]["frames"][2][
                "source_pts"
            ]
        elif change == "missing":
            del wrong["native_clock"]
        else:
            wrong["native_clock"]["frames"][0]["name"] = "f_0002.jpg"
        with pytest.raises(ValueError):
            players.source_clock_ordinals(wrong, epochs, 25)


def test_measured_output_reuse_preserves_clock_and_jpeg_identity(tmp_path, monkeypatch):
    original, receipt, inputs, ledger = _extraction(tmp_path, 0.08)
    before = receipt.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("reusable native extraction must not decode again")

    monkeypatch.setattr(players.subprocess, "run", unexpected)
    actual = players.extract_clip_frames(
        str(inputs.source_video),
        str(inputs.frames_directory.parent),
        [{"pt": 1, "t0": 0.4, "t1": 1.0}],
        25,
        preserve_source_fps=True,
        measured_native_epochs=True,
    )
    assert original == actual
    assert receipt.read_bytes() == before


def _halfway_receipt(start_tick_delta: int, *, frames: int = 4):
    """Nominal 60000/1001 seek that rounds halfway between 90 kHz ticks."""
    fps = 60000 / 1001
    tb = "1/90000"
    lo = 50049
    seek = lo / fps
    threshold_tick = math.floor(seek / float(Fraction(tb)) + 0.5)
    step = 1502
    pts = [threshold_tick + start_tick_delta + i * step for i in range(frames)]
    names = [f"f_{i + 1:04d}.jpg" for i in range(frames)]
    epochs = [float(Fraction(p, 90000)) for p in pts]
    timestamps = [epochs[0] + (i - lo) / fps for i in range(lo + frames + 1)]
    for i, epoch in enumerate(epochs):
        timestamps[lo + i] = epoch
    timestamps[lo - 1] = float(Fraction(pts[0] - step, 90000))
    receipt = {
        "identity": {"point": {"t0": seek, "t1": (lo + frames) / fps}},
        "source_identity": {"configuration": {"native_epoch_capture": "ffmpeg_copyts_showinfo_v1"}},
        "frames": [{"name": name} for name in names],
        "native_clock": {
            "schema": "native_extracted_picture_clock_v1",
            "method": "ffmpeg_copyts_showinfo",
            "seek_timestamp": True,
            "seek_seconds": seek,
            "time_base": tb,
            "frames": [{"name": name, "source_pts": p} for name, p in zip(names, pts, strict=True)],
        },
    }
    return receipt, timestamps, fps, lo


def test_halfway_timebase_tick_keeps_the_nominal_index_frame():
    receipt, timestamps, fps, lo = _halfway_receipt(-1)
    assert players.source_clock_ordinals(receipt, timestamps, fps) == list(range(lo, lo + 4))


def test_a_frame_early_halfway_seek_is_still_outside_the_window():
    receipt, timestamps, fps, _ = _halfway_receipt(-1502)
    with pytest.raises(ValueError, match="outside its requested seek window"):
        players.source_clock_ordinals(receipt, timestamps, fps)


def test_missing_or_nonmonotone_measured_log_is_not_nominally_filled(tmp_path):
    log = tmp_path / "ffmpeg.log"
    log.write_text("config in time_base: 1/1000\nn: 0 pts: 123 pts_time:0.123\n")
    with pytest.raises(ValueError, match="lacks measured"):
        players.captured_native_clock(log, ["f_0001.jpg", "f_0002.jpg"], 0.12)
    log.write_text(log.read_text() + "n: 1 pts: 123 pts_time:0.123\n")
    with pytest.raises(ValueError, match="increase"):
        players.captured_native_clock(log, ["f_0001.jpg", "f_0002.jpg"], 0.12)


def test_invalid_cached_clock_is_remeasured_instead_of_retaining_stale_epochs(tmp_path):
    original, receipt, inputs, _ = _extraction(tmp_path, 0.08)
    expected = receipt.read_bytes()
    document = json.loads(expected)
    for change in ("count", "time_base", "names", "pts_type"):
        invalid = deepcopy(document)
        if change == "count":
            invalid["native_clock"]["frames"].pop()
        elif change == "time_base":
            del invalid["native_clock"]["time_base"]
        elif change == "names":
            invalid["native_clock"]["frames"][0]["name"] = "f_9999.jpg"
        else:
            invalid["native_clock"]["frames"][0]["source_pts"] = "not_a_measured_integer"
        receipt.write_text(json.dumps(invalid))
        actual = players.extract_clip_frames(
            str(inputs.source_video),
            str(inputs.frames_directory.parent),
            [{"pt": 1, "t0": 0.4, "t1": 1.0}],
            25,
            preserve_source_fps=True,
            measured_native_epochs=True,
        )
        assert actual == original
        assert receipt.read_bytes() == expected


def _fake_native_decoder(epochs_by_point: dict[str, list[int]]):
    """Stand-in FFmpeg: writes JPEG-headed pictures and the showinfo epochs it 'measured'."""

    def run(command, check, stdout=None, stderr=None):
        pattern = command[-1]
        point = Path(pattern).parent.name
        count = int(command[command.index("-frames:v") + 1])
        epochs = epochs_by_point[point][:count]
        for index in range(1, len(epochs) + 1):
            Path(pattern % index).write_bytes(b"\xff\xd8" + f"{point}:{index}".encode())
        stderr.write("config in time_base: 1/90000\n")
        stderr.writelines(
            f"n: {n} pts: {value} pts_time:{value / 90000:.6f}\n" for n, value in enumerate(epochs)
        )
        return subprocess.CompletedProcess(command, 0)

    return run


def test_non_increasing_epochs_cut_only_the_affected_clip(tmp_path, monkeypatch):
    """Fix list 22: a source epoch reset inside one point cuts and holds that point only."""
    from cv.pipeline import frame_cadence

    video = tmp_path / "source.mp4"
    video.write_bytes(b"not decoded by the stand-in")
    fps = 60000 / 1001
    step = 1501
    points = [{"pt": 1, "t0": 10.0, "t1": 10.1}, {"pt": 2, "t0": 20.0, "t1": 20.1}]
    windows = [players.native_frame_window(p["t0"], p["t1"], fps) for p in points]
    counts = [hi - lo + 1 for lo, hi in windows]
    clean = [900_000 + i * step for i in range(counts[0])]
    # Third picture repeats its predecessor's epoch, then the clock resets backwards.
    broken = [1_800_000, 1_800_000 + step, 1_800_000 + step] + [
        1_700_000 + i * step for i in range(counts[1] - 3)
    ]
    monkeypatch.setattr(
        players.subprocess, "run", _fake_native_decoder({"pt0001": clean, "pt0002": broken})
    )
    root = tmp_path / "frames"
    dispositions: list[dict] = []

    def extract():
        return players.extract_clip_frames(
            str(video),
            str(root),
            points,
            fps,
            preserve_source_fps=True,
            measured_native_epochs=True,
            dispositions=dispositions,
        )

    index = extract()
    by_point = {pt: [row for row in index if row[0] == pt] for pt in (1, 2)}
    # The clean clip is complete; the affected clip keeps its measured increasing prefix.
    assert [row[2] for row in by_point[1]] == [float(Fraction(v, 90000)) for v in clean]
    assert [row[2] for row in by_point[2]] == [float(Fraction(v, 90000)) for v in broken[:2]]
    assert sorted(p.name for p in (root / "pt0002").glob("f_*.jpg")) == [
        "f_0001.jpg",
        "f_0002.jpg",
    ]
    clean_receipt = json.loads((root / "pt0001/extraction_receipt.json").read_text())
    cut_receipt = json.loads((root / "pt0002/extraction_receipt.json").read_text())
    assert "native_epoch_cut" not in clean_receipt
    cut = cut_receipt["native_epoch_cut"]
    assert cut["decision"] == players.NATIVE_EPOCH_CUT
    assert (cut["emitted_frames"], cut["kept_frames"]) == (counts[1], 2)
    assert (cut["first_dropped_frame"], cut["previous_source_pts"], cut["source_pts"]) == (
        "f_0003.jpg",
        broken[1],
        broken[2],
    )
    assert [row["source_pts"] for row in cut_receipt["native_clock"]["frames"]] == broken[:2]
    assert dispositions == [{"point": 2, **cut}]

    # Resume reuses the declared cut without decoding again or rewriting either receipt.
    receipts = [root / pt / "extraction_receipt.json" for pt in ("pt0001", "pt0002")]
    before = [path.read_bytes() for path in receipts]

    def unexpected(*args, **kwargs):
        raise AssertionError("a declared cut must be reused, not decoded again")

    monkeypatch.setattr(players.subprocess, "run", unexpected)
    dispositions.clear()
    assert extract() == index
    assert dispositions == [{"point": 2, **cut}]
    assert [path.read_bytes() for path in receipts] == before

    # Downstream: cadence holds only the cut point; S6 refuses it with the declared reason.
    assert not frame_cadence.extraction_epoch_cut(root / "pt0001")
    assert frame_cadence.extraction_epoch_cut(root / "pt0002")
    with pytest.raises(ValueError, match="cut at a non-increasing source epoch"):
        _native_inventory(SimpleNamespace(extraction_receipt=receipts[1]), fps, {})


def test_epoch_cut_matches_the_strict_clock_when_epochs_increase(tmp_path):
    log = tmp_path / "ffmpeg.log"
    log.write_text(
        "config in time_base: 1/1000\n"
        "n: 0 pts: 100 pts_time:0.1\nn: 1 pts: 140 pts_time:0.14\nn: 2 pts: 90 pts_time:0.09\n"
    )
    names = ["f_0001.jpg", "f_0002.jpg", "f_0003.jpg"]
    clock, cut = players.captured_native_clock_with_cut(log, names[:2], 0.1)
    assert cut is None and clock == players.captured_native_clock(log, names[:2], 0.1)
    clock, cut = players.captured_native_clock_with_cut(log, names, 0.1)
    assert [row["name"] for row in clock["frames"]] == names[:2]
    assert (cut["kept_frames"], cut["source_pts"]) == (2, 90)
    assert players.reusable_native_clock(clock, names[:2], 0.1)
