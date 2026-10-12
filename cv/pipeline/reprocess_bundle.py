"""Failure reprocess bundles: what a compacted match keeps so a failed rally can rerun alone.

Owner request 2026-10-01: cache intermediate results for every failed flight so later
pipeline improvements can reprocess any aspect of those clips cheaply. ``compact`` (``--keep
standard``; ``lean`` skips this) calls :func:`write` before the native frame folders and the
event crops are deleted. The bundle covers every attempt that did not end as a complete
accepted point with every flight confirmed: failed, partial, held, refused, abstained,
flagged (a flight failing post-validation) and tagged (a flight after the point-end call).
On the pilot that is nearly every rally (Goffin-Tabilo: 256 of 259 attempts), so each item
is either small or regenerated exactly.

Stored (``archive/reprocess_bundles.tar.zst``, index ``archive/REPROCESS_BUNDLES.json``):

- per bundled clip, the native extraction receipt (every picture's sha256 and source PTS)
  and the sha256 of each event-crop array slice of that clip;
- per match, the listing (name, bytes, sha256) of every file of the native frame folder.
  Its digest recomputes the directory sha256 the frame-extraction receipt recorded, so a
  folder holding only some regenerated clips still proves those clips are the producer's
  output (:func:`cv.pipeline.artifact_cache.directory_matches`);
- the source video's identity and path, the pinned ffmpeg, the code commit.

``torso_lock_inputs_v1`` is no longer deleted (it is archived with upstream): the composed
ball track's sidecar declares it as the guide, and S6 preparation reads and binds it.

Regenerated exactly (:func:`regenerate`): native pictures (``players.extract_clip_frames``
on the retained source video with the pinned ffmpeg, each picture checked against the
listing), event crops (``event_crops.build_broadcast_shard`` on the clip's rows, each array
checked against the stored slice digests), portal pictures (hardlinks of the native ones).
Not regenerable per rally: the whole-match point-finding samples (``frames_native``,
``frames_small``, ``overlay_crops``, ``score_crops_native``); they come back only with the
match's source pass, and no per-rally stage reads them.

    python -m cv.pipeline.reprocess_bundle show MATCH_DIR
    python -m cv.pipeline.reprocess_bundle regenerate MATCH_DIR --clips pt0005,pt0012 \\
        --out SCRATCH [--video PATH]

No evaluation label is read.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cv.pipeline import artifact_cache

SCHEMA = "reprocess_bundles_v1"
INDEX = "archive/REPROCESS_BUNDLES.json"
ARCHIVE = "archive/reprocess_bundles.tar.zst"
FRAMES_DIR = "audit_frames_native_1080"
LISTING_NAME = f"{FRAMES_DIR}.listing.tsv"  # artifact_cache.LISTING_SUFFIX
CROPS_DIR = "event_crops_v1"
#: Event-crop shard members, each hashed per clip slice.
CROP_MEMBERS = (
    "tight",
    "context",
    "mel",
    "clips",
    "frames",
    "centres",
    "track_observed",
    "audio_present",
    "proposal_source",
)
UNZSTD = ("zstd", "-d", "--long=30")
#: Classes in the order a reason maps to them (an attempt can carry several).
CLASSES = (
    "held",
    "refused",
    "abstained",
    "failed",
    "partial",
    "flagged",
    "tagged",
    "not_run",
    "indexed",
)
#: The per-flight failure index (``cv.pipeline.failure_index``); an attempt it lists that the
#: classes above miss is bundled as ``indexed``, so every listed slot can be regenerated.
FAILURE_INDEX = "FAILED_FLIGHTS.json"
#: First-fit statuses of attempts a capped or scoped run never tried (class ``not_run``).
NOT_RUN = ("not_run_cap", "not_run_scope", "not_run_selection")


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, default=str) + "\n")
    temporary.replace(path)


def _sha256_bytes(data) -> str:
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


# --------------------------------------------------------------------------- selection


def attempt_classes(attempt: dict, result_row: dict | None) -> list[str]:
    """Why an attempt needs a bundle; empty for a complete accepted, fully confirmed point."""
    classes = []
    reason = attempt.get("reason") or ""
    first = (attempt.get("first_fit") or {}).get("status")
    if first in NOT_RUN or reason in NOT_RUN:
        return ["not_run"]
    if reason != "accepted":
        if first not in (None, "completed"):
            classes.append("held")
        elif "refused" in reason:
            classes.append("refused")
        elif reason.startswith("events_no_contact"):
            classes.append("abstained")
        else:
            classes.append("failed")
    elif not attempt.get("complete_point") or any(
        not flight.get("accepted") for flight in attempt.get("flights") or []
    ):
        classes.append("partial")
    flights = (result_row or {}).get("accepted_flights") or []
    if any(
        not flight.get("confirmed", True)
        or (flight.get("validation") or {}).get("ok") is False
        or (flight.get("validation") or {}).get("tags")
        for flight in flights
    ):
        classes.append("flagged")
    if any(flight.get("after_point_end") for flight in flights):
        classes.append("tagged")
    return classes


def selection(out: Path) -> list[dict]:
    """Every attempt of ``OUTCOMES.json`` that needs a bundle, with its classes."""
    out = Path(out)
    outcomes = _load(out / "OUTCOMES.json")
    rows = {}
    if (out / "MATCH_RESULT.json").is_file():
        rows = {r["key"]: r for r in _load(out / "MATCH_RESULT.json").get("attempts") or []}
    selected = []
    for attempt in outcomes.get("attempts") or []:
        classes = attempt_classes(attempt, rows.get(attempt["key"]))
        if classes:
            selected.append(
                {
                    "key": attempt["key"],
                    "parent_key": attempt.get("parent_key"),
                    "clip": attempt.get("clip"),
                    "classes": classes,
                    "reason": attempt.get("reason"),
                    "stage": attempt.get("stage"),
                    "accepted_flights": attempt.get("accepted_flights"),
                    "first_fit_status": (attempt.get("first_fit") or {}).get("status"),
                    "fit_job": (attempt.get("reprocess") or {}).get("fit_job"),
                }
            )
    index = out / FAILURE_INDEX
    if index.is_file():
        chosen = {a["key"] for a in selected}
        listed = {r["key"] for r in _load(index).get("rows") or []}
        for attempt in outcomes.get("attempts") or []:
            if attempt["key"] in listed - chosen:
                selected.append(
                    {
                        "key": attempt["key"],
                        "parent_key": attempt.get("parent_key"),
                        "clip": attempt.get("clip"),
                        "classes": ["indexed"],
                        "reason": attempt.get("reason"),
                        "stage": attempt.get("stage"),
                        "accepted_flights": attempt.get("accepted_flights"),
                        "first_fit_status": (attempt.get("first_fit") or {}).get("status"),
                        "fit_job": (attempt.get("reprocess") or {}).get("fit_job"),
                    }
                )
    return selected


# --------------------------------------------------------------------------- frame listing


def read_listing(text: str) -> list[tuple[str, int, str]]:
    rows = []
    for line in text.splitlines():
        if line:
            relative, size, sha = line.split("\t")
            rows.append((relative, int(size), sha))
    return rows


def format_listing(rows: list[tuple[str, int, str]]) -> str:
    return "".join(f"{r}\t{s}\t{h}\n" for r, s, h in sorted(rows))


def recorded_frames_record(match_upstream: Path) -> dict | None:
    """The frame folder's directory record in the frame-extraction stage receipt."""
    receipt = match_upstream / "run_manifests" / "stage_receipts" / "frame_extraction.json"
    if not receipt.is_file():
        return None
    for output in _load(receipt).get("outputs") or []:
        if output.get("kind") == "directory" and str(output.get("path", "")).endswith(FRAMES_DIR):
            return output
    return None


def frames_listing(frames: Path, workers: int = 8) -> list[tuple[str, int, str]]:
    """(name, bytes, sha256) of every file under the frame folder.

    JPEG digests come from each clip's extraction receipt when its size and name match (the
    folder digest check below catches any stale receipt); other files are hashed.
    """
    from cv.pipeline.provenance import file_sha256

    known: dict[str, str] = {}
    for receipt in frames.glob("pt*/extraction_receipt.json"):
        try:
            for frame in _load(receipt).get("frames") or []:
                known[f"{receipt.parent.name}/{frame['name']}"] = frame["sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    files = sorted(p for p in frames.rglob("*") if p.is_file())

    def one(path: Path) -> tuple[str, int, str]:
        relative = path.relative_to(frames).as_posix()
        return relative, path.stat().st_size, known.get(relative) or file_sha256(path)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(one, files))
    return rows


def verified_frames_listing(frames: Path, recorded: dict | None, workers: int = 8) -> dict:
    """The listing, checked against the receipt's directory sha256 (fully hashed if stale)."""
    rows = frames_listing(frames, workers)
    digest = artifact_cache.listing_digest(rows)
    if recorded is not None and digest != recorded.get("sha256"):
        from cv.pipeline.provenance import file_sha256

        # A receipt digest that no longer matches its picture: hash every file instead.
        rows = [
            (p.relative_to(frames).as_posix(), p.stat().st_size, file_sha256(p))
            for p in sorted(q for q in frames.rglob("*") if q.is_file())
        ]
        digest = artifact_cache.listing_digest(rows)
    return {
        "rows": rows,
        "sha256": digest,
        "files": len(rows),
        "bytes": sum(r[1] for r in rows),
        "matches_receipt": None if recorded is None else digest == recorded.get("sha256"),
    }


# --------------------------------------------------------------------------- crops


def crop_slices(shard: Path, clips: set[str] | None = None) -> dict[str, dict]:
    """Per clip (``clips``: only these): row range in the shard, sha256 of each member slice."""
    import numpy as np

    from cv.pipeline.event_crops import memmap_member

    with np.load(shard) as archive:
        names = [str(c) for c in archive["clips"]]
    ranges: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        clip = name.split("__", 1)[1] if "__" in name else name
        span = ranges.get(clip)
        if span is None:
            ranges[clip] = [index, index + 1]
        elif span[1] == index:
            span[1] = index + 1
        else:
            raise ValueError(f"{shard}: rows of {clip} are not contiguous")
    wanted = {clip: span for clip, span in ranges.items() if clips is None or clip in clips}
    digests: dict[str, dict[str, str]] = {clip: {} for clip in wanted}
    with shard.open("rb") as handle:
        for member in CROP_MEMBERS:
            try:
                mapped = memmap_member(shard, member)
            except (KeyError, ValueError):
                with np.load(shard) as archive:
                    if member not in archive.files:
                        continue
                    array = np.ascontiguousarray(archive[member])
                for clip, (lo, hi) in wanted.items():
                    digests[clip][member] = _sha256_bytes(np.ascontiguousarray(array[lo:hi]))
                continue
            # Streamed from the file (each clip's slice is one contiguous byte range of a
            # C-order member), dropping the read pages: an 82 GB shard stays out of memory.
            row_bytes = mapped.dtype.itemsize * int(np.prod(mapped.shape[1:], dtype=np.int64))
            base = mapped.offset
            del mapped
            for clip, (lo, hi) in sorted(wanted.items(), key=lambda item: item[1][0]):
                digest = hashlib.sha256()
                position, end = base + lo * row_bytes, base + hi * row_bytes
                while position < end:
                    block = os.pread(handle.fileno(), min(1 << 24, end - position), position)
                    if not block:
                        raise ValueError(f"{shard}:{member} ends early")
                    digest.update(block)
                    position += len(block)
                digests[clip][member] = digest.hexdigest()
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(
                        handle.fileno(),
                        base + lo * row_bytes,
                        end - base - lo * row_bytes,
                        os.POSIX_FADV_DONTNEED,
                    )
    pixels = "stored"
    from cv.pipeline.event_crops import PIXEL_MEMBERS

    missing = [m for m in PIXEL_MEMBERS if not any(m in d for d in digests.values())]
    if missing:
        # A fused (metadata-only) shard: event inference cut the pixels in memory and left
        # the digests the stored slices would have beside the shard.
        from cv.pipeline.event_video_model import fused_digest_path

        sidecar = fused_digest_path(shard)
        pixels = "fused_digests" if sidecar.is_file() else "absent"
        recorded = json.loads(sidecar.read_text())["clips"] if sidecar.is_file() else {}
        for clip, span in wanted.items():
            entry = recorded.get(clip)
            if entry is None:
                continue
            if list(entry["rows"]) != list(span):
                raise ValueError(f"{sidecar}: rows of {clip} differ from the shard's")
            for member in missing:
                if member in entry["sha256"]:
                    digests[clip][member] = entry["sha256"][member]
    return {
        clip: {"rows": list(span), "sha256": digests[clip], "pixels": pixels}
        for clip, span in wanted.items()
    }


def clip_crop_rows(manifest: dict, broadcast: str, clip: str) -> list[tuple[str, int]]:
    """The (clip, frame) rows the runtime crop build cut for one clip, in shard order."""
    key = f"{broadcast}__{clip}"
    return [
        (clip, int(row["frame"]))
        for row in manifest.get("rows_index") or []
        if row["broadcast"] == broadcast and row["clip"] == key
    ]


# --------------------------------------------------------------------------- write


def _source_video(out: Path, match_upstream: Path) -> dict:
    """The broadcast file: path, size and sha256 as the extraction receipt bound."""
    path = None
    for name in ("QUEUE_ITEM.json", "OUTCOMES.json"):
        if (out / name).is_file():
            document = _load(out / name)
            path = document.get("video") or (document.get("source") or {}).get("video")
            if path:
                break
    record = {"path": path, "exists": bool(path) and Path(path).is_file()}
    receipt = match_upstream / "run_manifests" / "stage_receipts" / "frame_extraction.json"
    if receipt.is_file():
        inputs = _load(receipt).get("identity", {}).get("inputs") or []
        videos = [i for i in inputs if i.get("path_base") == "unconfigured_external"]
        if videos:
            record.update(sha256=videos[0].get("sha256"), bytes=videos[0].get("bytes"))
    if record["exists"]:
        record["bytes_on_disk"] = Path(path).stat().st_size
    # A box that ran from a staged copy (product_queue stage_video) names that copy, which is
    # deleted later; the queue item's original stays (regeneration checks every picture).
    item = out / "QUEUE_ITEM.json"
    original = _load(item).get("video") if item.is_file() else None
    if original and original != path:
        record["original"] = {
            "path": original,
            "exists": Path(original).is_file(),
            "bytes_on_disk": Path(original).stat().st_size if Path(original).is_file() else None,
        }
    return record


def source_video_path(record: dict | None, *fallbacks: str | None) -> str | None:
    """The first of the bundle's video, its original and ``fallbacks`` that exists (else the
    first named): the bytes are proven by the picture listing, not by the path."""
    record = record or {}
    candidates = [record.get("path"), (record.get("original") or {}).get("path"), *fallbacks]
    candidates = [c for c in candidates if c]
    return next((c for c in candidates if Path(c).is_file()), candidates[0] if candidates else None)


def write(out: Path, match_id: str, *, workers: int = 8) -> dict:
    """Store this match's failure bundles (idempotent; before the bulk is deleted)."""
    from cv.pipeline import ffmpeg_pin, provenance

    out = Path(out)
    index_path, archive = out / INDEX, out / ARCHIVE
    if index_path.is_file() and archive.is_file():
        return {"index": INDEX, "already": True}
    if not (out / "OUTCOMES.json").is_file():
        return {"index": None, "reason": "no OUTCOMES.json"}
    started = time.time()
    upstream = out / "upstream" / match_id
    frames = upstream / FRAMES_DIR
    attempts = selection(out)
    clips = sorted({a["clip"] for a in attempts if a.get("clip")})
    staging = out / "archive" / "reprocess_bundles.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    stored: dict[str, list[str]] = {clip: [] for clip in clips}
    listing = None
    if frames.is_dir():
        listing = verified_frames_listing(frames, recorded_frames_record(upstream), workers)
        (staging / LISTING_NAME).write_text(format_listing(listing["rows"]))
        for clip in clips:
            receipt = frames / clip / "extraction_receipt.json"
            if receipt.is_file():
                target = staging / "clips" / clip / "extraction_receipt.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(receipt, target)
                stored[clip].append("extraction_receipt.json")
    shard = out / "upstream" / CROPS_DIR / f"{match_id}.npz"
    crops = {}
    if shard.is_file():
        slices = crop_slices(shard, set(clips))
        for clip in clips:
            if clip in slices:
                crops[clip] = slices[clip]
                _save(staging / "clips" / clip / "event_crops.json", slices[clip])
                stored[clip].append("event_crops.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        tar = subprocess.Popen(["tar", "-C", str(staging), "-cf", "-", "."], stdout=subprocess.PIPE)
        subprocess.run(
            ["zstd", "-q", "-9", "-T4", "-c"], stdin=tar.stdout, stdout=handle, check=True
        )
        if tar.wait() != 0:
            raise RuntimeError("tar failed")
    members = subprocess.run(
        ["tar", "-I", " ".join(UNZSTD), "-tf", str(temporary)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    expected = sum(1 for _ in staging.rglob("*")) + 1
    if len(members) != expected:
        raise RuntimeError(f"bundle archive lists {len(members)} members, expected {expected}")
    temporary.replace(archive)
    shutil.rmtree(staging)
    recorded = recorded_frames_record(upstream)
    document = {
        "schema": SCHEMA,
        "match_id": match_id,
        "written": time.time(),
        "seconds": round(time.time() - started, 1),
        "archive": ARCHIVE,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": provenance.file_sha256(archive),
        "selection_rule": (
            "every OUTCOMES attempt that is not a complete accepted point with every flight "
            "accepted and confirmed: " + ", ".join(CLASSES)
        ),
        "counts": {
            "attempts": len(_load(out / "OUTCOMES.json").get("attempts") or []),
            "bundled_attempts": len(attempts),
            "bundled_clips": len(clips),
            "by_class": {c: sum(c in a["classes"] for a in attempts) for c in CLASSES},
        },
        "attempts": attempts,
        "clips": {
            clip: {
                "stored": stored[clip],
                "crop_rows": (crops.get(clip) or {}).get("rows"),
            }
            for clip in clips
        },
        "frames_listing": None
        if listing is None
        else {
            "member": LISTING_NAME,
            "files": listing["files"],
            "bytes": listing["bytes"],
            "sha256": listing["sha256"],
            "matches_extraction_receipt": listing["matches_receipt"],
            "receipt_directory_sha256": (recorded or {}).get("sha256"),
        },
        "source_video": _source_video(out, upstream),
        "ffmpeg": ffmpeg_pin.identity(),
        "code": provenance.git_record(Path(__file__).resolve().parents[2]),
        "stored": {
            "extraction_receipt.json": "per clip: every native picture's sha256 and source PTS",
            "event_crops.json": "per clip: crop shard row range, sha256 of each member slice",
            LISTING_NAME: "every file of the native frame folder (name, bytes, sha256)",
            "upstream archive": "all upstream outputs incl. torso_lock_inputs_v1, receipts",
            "s6_stages archive": "fit/refit/beam jobs, inputs and results (standard keep)",
        },
        "regenerate": {
            "native pictures": "players.extract_clip_frames on the source video with the "
            "pinned ffmpeg; every picture checked against the listing",
            "event crops": "event_crops.build_broadcast_shard on the clip's manifest rows "
            "from regenerated pictures; every member slice checked against event_crops.json",
            "portal pictures": "hardlinks of the regenerated native pictures",
        },
        "not_regenerable_per_rally": {
            "frames_native, frames_small, overlay_crops, score_crops_native": (
                "whole-match point-finding samples; only the match's source pass "
                "recreates them, and no per-rally stage reads them"
            ),
            "cascade evidence pictures": "hashes kept; recut from native pictures if needed",
        },
    }
    _save(index_path, document)
    return {
        "index": INDEX,
        "archive": ARCHIVE,
        "bytes": document["archive_bytes"],
        "bundled_attempts": len(attempts),
        "bundled_clips": len(clips),
        "seconds": document["seconds"],
    }


# --------------------------------------------------------------------------- regenerate


def bundle_member(match: Path, member: str) -> bytes | None:
    """One file of the bundle archive (``clips/pt0005/extraction_receipt.json``)."""
    archive = Path(match) / ARCHIVE
    if not archive.is_file():
        return None
    data = subprocess.run([*UNZSTD, "-c", str(archive)], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        try:
            handle = tar.extractfile(f"./{member}")
        except KeyError:
            return None
        return handle.read() if handle else None


def extract_bundle(match: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-I", " ".join(UNZSTD), "-xf", str(Path(match) / ARCHIVE), "-C", str(destination)],
        check=True,
    )
    return destination


def listing_clips(bundle: Path) -> list[str]:
    """Every clip the match's native frame folder held (bundled or not)."""
    rows = read_listing((bundle / LISTING_NAME).read_text())
    return sorted({row[0].split("/", 1)[0] for row in rows if "/" in row[0]})


def regenerate_frames(
    bundle: Path,
    clips: list[str],
    frames_root: Path,
    video: Path,
    points: dict[str, dict] | None = None,
) -> dict:
    """Re-extract the clips' native pictures and check each against the stored listing.

    ``bundle`` is an extracted bundle folder. The clips' original extraction receipts are put
    back byte for byte (the listing binds them), so S6 preparation sees the producer's own
    receipt. Raises on any picture that differs.

    A clip without a bundled receipt (a complete accepted point, needed by a court start)
    is extracted with a bundled clip's configuration and its own ``points[clip]`` window
    (``{"pt", "t0", "t1"}`` of the point map); its pictures are checked against the listing
    the same way and it keeps the receipt the extraction writes.
    """
    from cv.pipeline import ffmpeg_pin, players
    from cv.pipeline.provenance import file_sha256

    listing = {r[0]: (r[1], r[2]) for r in read_listing((bundle / LISTING_NAME).read_text())}
    receipts = {}
    rebuilt = set()
    template = None
    for clip in clips:
        stored = bundle / "clips" / clip / "extraction_receipt.json"
        if stored.is_file():
            receipts[clip] = _load(stored)
            template = template or receipts[clip]
    for clip in clips:
        if clip in receipts:
            continue
        if template is None or not points or clip not in points:
            raise FileNotFoundError(f"{clip}: no bundled extraction receipt and no point window")
        receipts[clip] = {
            "source_identity": template["source_identity"],
            "identity": {**template["identity"], "point": points[clip]},
        }
        rebuilt.add(clip)
    groups: dict[str, list[str]] = {}
    for clip, receipt in receipts.items():
        configuration = receipt["source_identity"]["configuration"]
        key = json.dumps([configuration, receipt["identity"].get("video_duration")], sort_keys=True)
        groups.setdefault(key, []).append(clip)
    started = time.time()
    scratch = frames_root.parent / f".{frames_root.name}.regenerate"
    if scratch.exists():
        shutil.rmtree(scratch)
    for key, members in groups.items():
        configuration, duration = json.loads(key)
        players.extract_clip_frames(
            str(video),
            str(scratch),
            [receipts[clip]["identity"]["point"] for clip in members],
            configuration["fps"],
            native=configuration["native"],
            preserve_source_fps=configuration["preserve_source_fps"],
            video_duration=duration,
            measured_native_epochs=bool(configuration.get("native_epoch_capture")),
        )
    extraction_seconds = time.time() - started
    report = {}
    for clip in clips:
        produced = scratch / clip
        wanted = {
            name.split("/", 1)[1]: entry
            for name, entry in listing.items()
            if name.startswith(f"{clip}/") and name.endswith(".jpg")
        }
        actual = {p.name: p for p in produced.glob("f_*.jpg")}
        if set(actual) != set(wanted):
            raise ValueError(
                f"{clip}: regenerated {len(actual)} pictures, listing has {len(wanted)}"
            )
        bad = [
            name
            for name, path in actual.items()
            if (path.stat().st_size, file_sha256(path)) != wanted[name]
        ]
        if bad:
            raise ValueError(f"{clip}: {len(bad)} regenerated pictures differ, e.g. {bad[:3]}")
        target = frames_root / clip
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name, path in actual.items():
            path.replace(target / name)
        if clip in rebuilt:
            shutil.copyfile(
                produced / "extraction_receipt.json", target / "extraction_receipt.json"
            )
            report[clip] = {"pictures": len(actual), "identical": True, "receipt": "re-extracted"}
            continue
        original = bundle / "clips" / clip / "extraction_receipt.json"
        shutil.copyfile(original, target / "extraction_receipt.json")
        entry = listing.get(f"{clip}/extraction_receipt.json")
        if entry and entry[1] != file_sha256(target / "extraction_receipt.json"):
            raise ValueError(f"{clip}: stored extraction receipt differs from the listing")
        report[clip] = {"pictures": len(actual), "identical": True}
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "clips": report,
        "pictures": sum(r["pictures"] for r in report.values()),
        "seconds": round(time.time() - started, 1),
        "extraction_seconds": round(extraction_seconds, 1),
        "ffmpeg": ffmpeg_pin.identity(),
    }


def write_listing_sidecar(bundle: Path, match_upstream: Path) -> Path:
    """Place the stored listing beside the frame folder (read by ``directory_matches``)."""
    target = match_upstream / LISTING_NAME
    shutil.copyfile(bundle / LISTING_NAME, target)
    return target


def regenerate_crops(
    upstream_root: Path, match_id: str, clips: list[str], output: Path, bundle: Path
) -> dict:
    """Cut the clips' event crops from the regenerated pictures and check every slice."""
    import numpy as np

    from cv.pipeline import event_crops

    manifest = _load(upstream_root / CROPS_DIR / "manifest.json")
    event_track = (manifest.get("feature_manifest") or {}).get("ball_track", {})
    track = event_track.get("event_track") if isinstance(event_track, dict) else None
    started = time.time()
    report = {}
    for clip in clips:
        rows = clip_crop_rows(manifest, match_id, clip)
        stored_path = bundle / "clips" / clip / "event_crops.json"
        if not rows or not stored_path.is_file():
            report[clip] = {"rows": len(rows), "checked": False}
            continue
        shard = output / f"{match_id}__{clip}.npz"
        event_crops.build_broadcast_shard(
            upstream_root,
            match_id,
            rows,
            shard,
            **({"event_track": track} if track else {}),
        )
        stored = _load(stored_path)["sha256"]
        with np.load(shard) as archive:
            actual = {
                name: _sha256_bytes(np.ascontiguousarray(archive[name]))
                for name in stored
                if name in archive.files
            }
        differing = sorted(k for k in stored if actual.get(k) != stored[k])
        report[clip] = {
            "rows": len(rows),
            "checked": True,
            "pixels_checked": all(m in stored for m in event_crops.PIXEL_MEMBERS),
            "differing_members": differing,
        }
        if differing:
            raise ValueError(f"{clip}: regenerated crops differ in {differing}")
    return {"clips": report, "seconds": round(time.time() - started, 1)}


def regenerate(match: Path, clips: list[str], out: Path, video: Path | None = None) -> dict:
    """Pictures, then crops, of ``clips`` from a compacted match into ``out`` (verified)."""
    match = Path(match)
    index = _load(match / INDEX)
    item = match / "QUEUE_ITEM.json"
    video = Path(
        video
        or source_video_path(
            index["source_video"], _load(item).get("video") if item.is_file() else None
        )
    )
    bundle = extract_bundle(match, out / "bundle")
    frames_root = out / "upstream" / index["match_id"] / FRAMES_DIR
    frames = regenerate_frames(bundle, clips, frames_root, video)
    return {"frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("show")
    one.add_argument("match", type=Path)
    one = sub.add_parser("regenerate")
    one.add_argument("match", type=Path)
    one.add_argument("--clips", required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.command == "show":
        document = _load(args.match / INDEX)
        print(
            json.dumps(
                {k: document[k] for k in ("counts", "frames_listing", "source_video")}, indent=1
            )
        )
        return
    report = regenerate(args.match, args.clips.split(","), args.out, args.video)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
