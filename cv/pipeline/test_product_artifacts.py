import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from cv.pipeline import product_artifacts as pa

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("zstd")), reason="needs ffmpeg and zstd"
)
MATCH = "m1"


def _frame(n: int) -> np.ndarray:
    """A picture whose content encodes its frame number (a square moving 9 px per frame)."""
    picture = np.full((180, 320, 3), 40, np.uint8)
    x = 5 + 9 * (n % 30)
    cv2.rectangle(picture, (x, 60), (x + 30, 110), (230, 230, 230), -1)
    return picture


def _portal_folder(frames_root, native, key, first, last):
    folder = frames_root / key
    folder.mkdir(parents=True)
    for n in range(first, last + 1):
        os.link(native / f"f_{n:04d}.jpg", folder / f"f_{n:04d}.jpg")
    (folder / "manifest.json").write_text(json.dumps({"frame_range": [first, last]}))


def _match(tmp_path):
    out = tmp_path / MATCH
    native = out / "upstream" / MATCH / "audit_frames_native_1080" / "pt0001"
    native.mkdir(parents=True)
    for n in range(1, 61):
        cv2.imwrite(str(native / f"f_{n:04d}.jpg"), _frame(n))
    portal = out / "portal" / "3d"
    data = portal / "data"
    data.mkdir(parents=True)
    keys = {"a_pt0001": 3, "a_pt0002": 0, "a_pt0003": None}
    for key, flights in keys.items():
        _portal_folder(portal / "frames", native, key, 1, 60)
        (data / f"{key}.json").write_text(
            json.dumps({"clip": key, "fps": 50.0, "frames_available": True, "frame_range": [1, 60]})
        )
    _portal_folder(portal / "frames", native, "a_pt0001__component_0", 1, 60)
    (data / "a_pt0001__component_0.json").write_text(json.dumps({"clip": "c"}))
    (data / "index.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "clip": k,
                        "file": f"{k}.json",
                        "accepted_flights": f,
                        "frames_available": True,
                    }
                    for k, f in keys.items()
                    if f is not None
                ],
                "automatic_s6": {
                    "held_before_fitting": [
                        {"clip": "a_pt0003", "file": "a_pt0003.json", "frames_available": True}
                    ]
                },
            }
        )
    )
    for tree in (
        "fit/attempts/a/stage_run",
        "refit/a/stage",
        "beam/a/r1/b0/stage_run",
        "cascade/packets",
    ):
        (out / tree).mkdir(parents=True)
        (out / tree / "result.json").write_text(json.dumps({"x": list(range(2000))}))
    (out / "fit" / "summary.json").write_text('{"attempts": []}')
    (out / "fit" / "attempts" / "a" / "stage").symlink_to("stage_run")
    (out / "refit" / "REFIT.json").write_text('{"runs": []}')
    for name in ("BEAM.json", "CALIBRATION.json", "STATE.json"):
        (out / "beam" / name).write_text('{"changed": {}}')
    evidence = out / "cascade" / "adjudicate" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "e.jpg").write_bytes(b"jpeg")
    (out / "cascade" / "COST.json").write_text('{"receipt_usd": 1.0}')
    crops = out / "upstream" / "event_crops_v1"
    crops.mkdir(parents=True)
    (crops / f"{MATCH}.npz").write_bytes(b"0" * 4096)
    (crops / "manifest.json").write_text("{}")
    (out / "upstream" / MATCH / "automatic_point_ledger_v1.csv").write_text("a,b\n")
    (out / "upstream" / MATCH / "ball_track.csv").write_text("f,x\n1,2\n")
    (out / "MATCH_RESULT.json").write_text("{}")
    return out, native


def _stub_viewer(monkeypatch):
    monkeypatch.setattr(pa, "refresh_viewer", lambda portal: "stub")


def test_compact_keeps_accepted_clips_archives_and_deletes_bulk(tmp_path, monkeypatch):
    _stub_viewer(monkeypatch)
    out, native = _match(tmp_path)
    original = tmp_path / "original_frame.jpg"
    os.link(native / "f_0010.jpg", original)
    before_bytes = original.read_bytes()
    document = pa.compact(out, MATCH, workers=2)
    portal = out / "portal" / "3d"
    assert sorted(p.name for p in (portal / "frames").iterdir()) == ["a_pt0001"]
    kept = portal / "frames" / "a_pt0001"
    assert sorted(p.name for p in kept.iterdir()) == [
        "clip.mp4",
        "frames_video.json",
        "manifest.json",
    ]
    video = json.loads((portal / "data" / "a_pt0001.json").read_text())["frames_video"]
    assert (video["first_frame"], video["last_frame"], video["frames"]) == (1, 60, 60)
    assert video["alignment"] == "aligned" and video["rate"] == 50.0
    index = json.loads((portal / "data" / "index.json").read_text())
    assert [p["frames_available"] for p in index["points"]] == [True, False]
    assert index["automatic_s6"]["held_before_fitting"][0]["frames_available"] is False
    assert json.loads((portal / "data" / "a_pt0002.json").read_text())["frames_available"] is False
    # Hardlinked originals outside the match are untouched.
    assert original.read_bytes() == before_bytes
    # Archives exist, list their trees, and the kept summaries/ledger are unpacked.
    for name in ("s6_stages", "upstream"):
        assert (out / "archive" / f"{name}.tar.zst").is_file()
    listed = subprocess.run(
        ["tar", "-I", "zstd -d --long=30", "-tf", str(out / "archive" / "s6_stages.tar.zst")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (
        "fit/attempts/a/stage_run/result.json" in listed and "refit/a/stage/result.json" in listed
    )
    assert (out / "fit" / "summary.json").read_text() == '{"attempts": []}'
    assert (out / "refit" / "REFIT.json").is_file()
    assert "beam/a/r1/b0/stage_run/result.json" in listed
    assert sorted(p.name for p in (out / "beam").iterdir()) == [
        "BEAM.json",
        "CALIBRATION.json",
        "STATE.json",
    ]
    assert not (out / "fit" / "attempts").exists() and not (out / "cascade" / "packets").exists()
    assert sorted(p.name for p in (out / "upstream" / MATCH).iterdir()) == [
        "automatic_point_ledger_v1.csv"
    ]
    assert not (out / "upstream" / "event_crops_v1").exists()
    assert not (out / "cascade" / "adjudicate" / "evidence").exists()
    hashes = json.loads((out / "cascade" / "adjudicate" / "evidence_hashes.json").read_text())
    assert list(hashes["files"]) == ["e.jpg"]
    assert (out / "cascade" / "COST.json").is_file()
    assert document["status"] == "done" and document["total_after"] < document["total_before"]
    # Idempotent: a second run changes nothing and keeps the first inventory.
    again = pa.compact(out, MATCH, workers=2)
    assert again["total_before"] == document["total_before"]
    assert again["bytes_after"] == document["bytes_after"]


def test_interrupted_compaction_resumes(tmp_path, monkeypatch):
    _stub_viewer(monkeypatch)
    out, _ = _match(tmp_path)
    calls = {"n": 0}
    real = pa.archive_trees

    def crash_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(pa, "archive_trees", crash_once)
    with pytest.raises(KeyboardInterrupt):
        pa.compact(out, MATCH, workers=2)
    assert json.loads((out / "ARTIFACTS.json").read_text())["status"] == "running"
    document = pa.compact(out, MATCH, workers=2)
    assert document["status"] == "done"
    assert (out / "archive" / "s6_stages.tar.zst").is_file()
    assert (out / "portal" / "3d" / "frames" / "a_pt0001" / "clip.mp4").is_file()


def test_verify_clip_rejects_a_shifted_clip(tmp_path):
    folder = tmp_path / "k"
    folder.mkdir()
    for n in range(1, 81):
        cv2.imwrite(str(folder / f"f_{n:04d}.jpg"), _frame(n))
    (folder / "manifest.json").write_text(json.dumps({"frame_range": [2, 80]}))
    video = pa.encode_clip(folder, folder / "clip.mp4", rate=25.0, ffmpeg=shutil.which("ffmpeg"))
    assert pa.verify_clip(folder, video)["alignment"] == "aligned"
    shifted = {**video, "first_frame": 1, "last_frame": 79}
    with pytest.raises(ValueError, match="shifted"):
        pa.verify_clip(folder, shifted)


def test_keep_full_deletes_nothing_and_lean_drops_s6_trees(tmp_path, monkeypatch):
    _stub_viewer(monkeypatch)
    out, _ = _match(tmp_path / "full")
    pa.compact(out, MATCH, keep="full")
    assert (out / "fit" / "attempts").is_dir() and (out / "upstream" / "event_crops_v1").is_dir()
    assert not (out / "archive").exists()
    lean, _ = _match(tmp_path / "lean")
    document = pa.compact(lean, MATCH, keep="lean", workers=2)
    assert document["s6_deleted"] == ["fit", "refit", "beam", "cascade/packets"]
    assert (lean / "beam" / "BEAM.json").is_file() and not (lean / "beam" / "a").exists()
    assert not (lean / "archive" / "s6_stages.tar.zst").exists()
    assert (lean / "archive" / "upstream.tar.zst").is_file()
    assert (lean / "fit" / "summary.json").is_file()


def test_verify_clip_calls_a_nearly_still_clip_inconclusive(tmp_path):
    folder = tmp_path / "k"
    folder.mkdir()
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)

    def still(n):
        picture = base.copy()
        picture[100:110, 100 + n // 8 : 110 + n // 8] = 255  # a slow speck, no real motion
        return picture

    for n in range(1, 81):
        cv2.imwrite(str(folder / f"f_{n:04d}.jpg"), still(n))
    (folder / "manifest.json").write_text(json.dumps({"frame_range": [1, 80]}))
    video = pa.encode_clip(folder, folder / "clip.mp4", rate=25.0, ffmpeg=shutil.which("ffmpeg"))
    cv2.imwrite(str(folder / "f_0081.jpg"), still(81))
    shifted = {**video, "first_frame": 2, "last_frame": 81}
    assert pa.verify_clip(folder, shifted)["alignment"] == "inconclusive_static"


def test_a_tree_added_after_compaction_gets_its_own_verified_archive(tmp_path, monkeypatch):
    _stub_viewer(monkeypatch)
    out, _ = _match(tmp_path)
    beam = out / "beam"
    shutil.move(beam, tmp_path / "beam_aside")
    pa.compact(out, MATCH, workers=2)
    shutil.move(tmp_path / "beam_aside", beam)  # an old compaction left beam/ unpacked
    first = (out / "archive" / "s6_stages.tar.zst").read_bytes()
    report = pa.rearchive_s6(out)
    extra = out / "archive" / "s6_stages.beam.tar.zst"
    assert report["archived"] == ["beam"] and report["written"][0]["verify"] == "compare"
    assert (out / "archive" / "s6_stages.tar.zst").read_bytes() == first
    listed = subprocess.run(
        ["tar", "-I", "zstd -d --long=30", "-tf", str(extra)], capture_output=True, text=True
    ).stdout
    assert "beam/a/r1/b0/stage_run/result.json" in listed
    assert sorted(p.name for p in beam.iterdir()) == ["BEAM.json", "CALIBRATION.json", "STATE.json"]
    record = json.loads((out / "ARTIFACTS.json").read_text())
    assert record["s6_rearchived"][0]["archived"] == ["beam"]
    assert pa.rearchive_s6(out)["archive"] == "archive/s6_stages.tar.zst"  # nothing left to do


def test_portal_clip_encode_is_single_threaded_by_default(tmp_path, monkeypatch):
    # Two libx264 threads were not byte-reproducible for the same pictures (stage-scheduler
    # REPORT.md); one thread is the default, two the rollback.
    seen = []
    monkeypatch.setattr(pa.subprocess, "run", lambda command, **_: seen.append(command))
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"frame_range": [1, 1]}')
    (folder / "f_0001.jpg").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):  # the stub writes no clip
        pa.encode_clip(folder, folder / "clip.mp4", rate=25.0, ffmpeg="ffmpeg")
    assert seen[-1][seen[-1].index("-threads") + 1] == "1"
    monkeypatch.setenv(pa.CLIP_THREADS_ENV, "2")
    with pytest.raises(FileNotFoundError):
        pa.encode_clip(folder, folder / "clip.mp4", rate=25.0, ffmpeg="ffmpeg")
    assert seen[-1][seen[-1].index("-threads") + 1] == "2"
