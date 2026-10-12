"""Failure bundles: selection, stored listing and receipts, exact picture regeneration."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from cv.pipeline import artifact_cache, players
from cv.pipeline import reprocess_bundle as rb

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("zstd")), reason="needs ffmpeg and zstd"
)
MATCH = "m1"


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNIS_DATA_ROOT", str(tmp_path))


def _video(tmp_path):
    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=25:duration=2.4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
    )
    return path


def _attempt(clip, reason, complete, flights):
    return {
        "key": f"{MATCH}__{clip}",
        "parent_key": f"{MATCH}__{clip}",
        "clip": clip,
        "reason": reason,
        "complete_point": complete,
        "first_fit": {"status": "completed" if reason != "held" else "preparation_held"},
        "accepted_flights": sum(f["accepted"] for f in flights),
        "flights": flights,
    }


def _match(tmp_path):
    video = _video(tmp_path)
    out = tmp_path / "processed" / "run" / MATCH
    upstream = out / "upstream" / MATCH
    frames = upstream / rb.FRAMES_DIR
    points = [
        {"pt": 1, "t0": 0.0, "t1": 0.6},
        {"pt": 2, "t0": 0.8, "t1": 1.4},
        {"pt": 3, "t0": 1.6, "t1": 2.2},
    ]
    players.extract_clip_frames(
        str(video),
        str(frames),
        points,
        25,
        preserve_source_fps=True,
        measured_native_epochs=True,
    )
    receipts = upstream / "run_manifests" / "stage_receipts"
    receipts.mkdir(parents=True)
    (receipts / "frame_extraction.json").write_text(
        json.dumps({"outputs": [artifact_cache.path_record(frames)]})
    )
    ok = {"accepted": True}
    out.joinpath("OUTCOMES.json").write_text(
        json.dumps(
            {
                "match_id": MATCH,
                "source": {"video": str(video)},
                "attempts": [
                    _attempt("pt0001", "accepted", True, [ok, ok]),
                    _attempt("pt0002", "gate:bounce_rays_agree", False, [{"accepted": False}]),
                    _attempt("pt0003", "accepted", True, [ok]),
                ],
            }
        )
    )
    out.joinpath("MATCH_RESULT.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"key": f"{MATCH}__pt0001", "accepted_flights": [{"confirmed": True}] * 2},
                    {"key": f"{MATCH}__pt0003", "accepted_flights": [{"after_point_end": True}]},
                ]
            }
        )
    )
    crops = out / "upstream" / rb.CROPS_DIR
    crops.mkdir(parents=True)
    rows = 6
    np.savez(
        crops / f"{MATCH}.npz",
        tight=np.arange(rows * 8, dtype=np.uint8).reshape(rows, 8),
        context=np.ones((rows, 4), np.uint8),
        mel=np.zeros((rows, 2), np.float16),
        clips=np.asarray(
            [f"{MATCH}__pt0001"] * 2 + [f"{MATCH}__pt0002"] * 3 + [f"{MATCH}__pt0003"]
        ),
        frames=np.arange(rows, dtype=np.int32),
        centres=np.zeros((rows, 2), np.float32),
        track_observed=np.ones(rows, bool),
        audio_present=np.ones(rows, bool),
        proposal_source=np.asarray(["track"] * rows),
    )
    return out, video


def test_selection_classes():
    assert (
        rb.attempt_classes({"reason": "accepted", "complete_point": True, "flights": []}, None)
        == []
    )
    assert rb.attempt_classes(
        {"reason": "events_no_contact_in_scope", "first_fit": {"status": "completed"}}, None
    ) == ["abstained"]
    assert rb.attempt_classes(
        {"reason": "x", "first_fit": {"status": "preparation_held"}}, None
    ) == ["held"]
    assert rb.attempt_classes(
        {"reason": "accepted", "complete_point": True, "flights": [{"accepted": True}]},
        {"accepted_flights": [{"confirmed": False, "validation": {"ok": False}}]},
    ) == ["flagged"]
    assert rb.attempt_classes(
        {"reason": "not_run_cap", "first_fit": {"status": "not_run_cap"}}, None
    ) == ["not_run"]


def test_bundle_stores_receipts_listing_and_crop_digests_and_regenerates_exactly(tmp_path):
    out, video = _match(tmp_path)
    report = rb.write(out, MATCH, workers=2)
    assert report["bundled_attempts"] == 2 and report["bundled_clips"] == 2
    index = json.loads((out / rb.INDEX).read_text())
    assert [a["clip"] for a in index["attempts"]] == ["pt0002", "pt0003"]
    assert index["attempts"][0]["classes"] == ["failed"]
    assert index["attempts"][1]["classes"] == ["tagged"]
    assert index["frames_listing"]["matches_extraction_receipt"] is True
    assert index["clips"]["pt0002"]["crop_rows"] == [2, 5]
    assert rb.write(out, MATCH)["already"] is True

    frames = out / "upstream" / MATCH / rb.FRAMES_DIR
    binding = rb.recorded_frames_record(out / "upstream" / MATCH)
    original = (frames / "pt0002" / "f_0003.jpg").read_bytes()
    shutil.rmtree(frames)
    bundle = rb.extract_bundle(out, tmp_path / "work" / "bundle")
    stored = json.loads((bundle / "clips" / "pt0002" / "event_crops.json").read_text())
    assert set(stored["sha256"]) == set(rb.CROP_MEMBERS)
    with np.load(out / "upstream" / rb.CROPS_DIR / f"{MATCH}.npz") as shard:
        for name in ("tight", "clips", "proposal_source"):
            expected = rb._sha256_bytes(np.ascontiguousarray(shard[name][2:5]))
            assert stored["sha256"][name] == expected
    result = rb.regenerate_frames(bundle, ["pt0002"], frames, video)
    assert result["clips"]["pt0002"]["identical"] and result["pictures"] == 15
    assert (frames / "pt0002" / "f_0003.jpg").read_bytes() == original
    assert not artifact_cache.directory_matches(frames, binding)
    rb.write_listing_sidecar(bundle, out / "upstream" / MATCH)
    assert artifact_cache.directory_matches(frames, binding)

    # A picture that differs from the listing is refused.
    listing = bundle / rb.LISTING_NAME
    listing.write_text(
        "".join(
            line if not line.startswith("pt0003/f_0002.jpg") else "pt0003/f_0002.jpg\t1\t00\n"
            for line in listing.read_text().splitlines(keepends=True)
        )
    )
    with pytest.raises(ValueError, match="differ"):
        rb.regenerate_frames(bundle, ["pt0003"], frames, video)


def test_a_fused_shard_takes_its_pixel_digests_from_the_inference_sidecar(tmp_path):
    from cv.pipeline.event_video_model import FUSED_DIGESTS, fused_digest_path

    out, _ = _match(tmp_path)
    full = out / "upstream" / rb.CROPS_DIR / f"{MATCH}.npz"
    stored = rb.crop_slices(full)
    with np.load(full) as shard:
        arrays = {k: shard[k] for k in shard.files}
    fused = tmp_path / "fused" / f"{MATCH}.npz"
    fused.parent.mkdir()
    np.savez(fused, **{k: v for k, v in arrays.items() if k not in ("tight", "context", "mel")})
    assert {c: e["pixels"] for c, e in rb.crop_slices(fused).items()} == dict.fromkeys(
        stored, "absent"
    )
    clips = {
        clip: {
            "rows": entry["rows"],
            "sha256": {m: entry["sha256"][m] for m in ("tight", "context", "mel")},
        }
        for clip, entry in stored.items()
    }
    fused_digest_path(fused).write_text(json.dumps({"schema": FUSED_DIGESTS, "clips": clips}))
    from_sidecar = rb.crop_slices(fused)
    assert {c: e["sha256"] for c, e in from_sidecar.items()} == {
        c: e["sha256"] for c, e in stored.items()
    }
    assert {e["pixels"] for e in from_sidecar.values()} == {"fused_digests"}
    clips["pt0002"]["rows"] = [0, 1]
    fused_digest_path(fused).write_text(json.dumps({"schema": FUSED_DIGESTS, "clips": clips}))
    with pytest.raises(ValueError, match="rows of pt0002"):
        rb.crop_slices(fused)


def test_an_unbundled_clip_regenerates_from_its_point_window(tmp_path):
    """A court start needs every rally; a complete point has no bundled receipt."""
    out, video = _match(tmp_path)
    rb.write(out, MATCH, workers=2)
    frames = out / "upstream" / MATCH / rb.FRAMES_DIR
    original = (frames / "pt0001" / "f_0004.jpg").read_bytes()
    shutil.rmtree(frames)
    bundle = rb.extract_bundle(out, tmp_path / "work" / "bundle")
    assert rb.listing_clips(bundle) == ["pt0001", "pt0002", "pt0003"]
    assert not (bundle / "clips" / "pt0001").exists()
    with pytest.raises(FileNotFoundError, match="no bundled extraction receipt"):
        rb.regenerate_frames(bundle, ["pt0001"], frames, video)

    windows = {"pt0001": {"pt": 1, "t0": 0.0, "t1": 0.6}}
    result = rb.regenerate_frames(bundle, rb.listing_clips(bundle), frames, video, windows)
    assert result["clips"]["pt0001"] == {
        "pictures": 15,
        "identical": True,
        "receipt": "re-extracted",
    }
    assert (frames / "pt0001" / "f_0004.jpg").read_bytes() == original
    assert (frames / "pt0001" / "extraction_receipt.json").is_file()

    # A wrong window cannot pass: its pictures differ from the listing.
    shutil.rmtree(frames)
    with pytest.raises(ValueError):
        rb.regenerate_frames(
            bundle, ["pt0001", "pt0002"], frames, video, {"pt0001": {"pt": 1, "t0": 0.8, "t1": 1.4}}
        )
