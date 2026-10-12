"""Per-match artifact compaction for the product runner: bounded disk at scale.

A finished product-runner match directory is about 100 GB (Rome Paul-Vukic: event crops
64 GB, native audit frames 21 GB, S6 stage JSON 12 GB). ``compact`` reduces it to the
set a processed match keeps (Rome: 0.88 GB), in an order that is safe to interrupt and rerun:

1. Portal pictures. Each portal attempt folder ``portal/3d/frames/<key>`` holds hardlinks
   to the native audit frames. ``video`` mode encodes the kept attempts' frames into one
   H.264 clip per attempt (``clip.mp4``: picture ``i`` is native frame ``first_frame + i``,
   checked by decoding) and declares it in the attempt document as ``frames_video``; the
   viewer seeks the clip instead of loading JPEGs. Component folders are removed (the
   viewer shows a component with its parent's pictures). Attempts not kept lose their
   pictures and are marked ``frames_available: false``. Frame numbers and native timestamps
   are unchanged; the pictures are display derivatives, recorded as such.
2. Cascade evidence images (the pictures sent to the adjudicating models) are hashed into
   ``cascade/adjudicate/evidence_hashes.json`` and deleted; replies, receipts, the spend
   ledger and decisions stay.
3. Upstream bulk (event crop tensors, native frame folders, crop caches) is deleted, after the
   native pictures an S6 refit reads (near-baseline refinement candidates, a few per
   camera-fallback attempt) and every clip's extraction receipt go into a verified
   ``archive/s6_frames.tar.zst`` (Rome: 404 pictures, ~164 MB), and after the failure
   reprocess bundles (``cv.pipeline.reprocess_bundle``: for every attempt that is not a
   complete accepted point, what regenerates its pictures and crops exactly and proves
   them; ``lean`` skips them) are written.
4. ``fit``, ``refit``, ``beam`` and the cascade packets/manifests go into a verified
   ``archive/s6_stages.tar.zst`` (``lean``: deleted instead), the rest of upstream (ball
   track, camera, players, events, receipts) into ``archive/upstream.tar.zst``. The fit,
   refit and beam summaries and the automatic point ledger stay unpacked. (Before
   2026-10-01 ``beam`` was left in place: 45.7 GB on a production match.)

``keep="full"`` records the inventory and deletes nothing. A compacted match can be
refitted from its retained fit inputs (``archive/s6_stages.tar.zst`` plus the native pictures
S6 reads, ``archive/s6_frames.tar.zst``; ``cv.pipeline.product_reprocess``); anything upstream
reruns from the video.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cv.pipeline import reprocess_bundle

SCHEMA = "product_match_artifacts_v1"
FRAMES_SCHEMA = "portal_display_frames_v1"
FRAME_MODES = ("video", "native", "none")
FRAME_ATTEMPTS = ("accepted", "all")
KEEP = ("standard", "lean", "full")
VIDEO_WIDTH = 960
VIDEO_CRF = 26
VIDEO_GOP = 10
#: libx264 threads for a portal clip. Two threads gave different bytes (and 5-10 different
#: pictures) for the same input pictures in about 1 clip per match: one Nakashima-Rune folder
#: encoded 6 times gave 3 different clips; one thread gave 6 identical clips for every folder,
#: at +37% encode CPU (cv/experiments/stage_scheduler/REPORT.md). A run-twice byte check
#: relies on this. Rollback: TENNIS_PORTAL_CLIP_THREADS=2.
CLIP_THREADS_ENV = "TENNIS_PORTAL_CLIP_THREADS"
#: zstd with a 1 GiB window: component results repeat their parent's documents, so the
#: long window takes Rome's 12 GB of stage JSON to 0.33 GB (level 3 alone: 0.94 GB).
ZSTD_LEVEL = ("-9", "--long=30")
#: Archive zstd level (default 9). On Swiatek-Yuan's 21.3 GB S6 tree (dry run 3) level 1 took
#: 30 CPU-s for 1.074 GB against level 9's 117 CPU-s for 1.106 GB (same --long=30 window,
#: same extracted bytes); level 3: 44 CPU-s, 1.155 GB. Rollback: unset.
ZSTD_LEVEL_ENV = "TENNIS_ARCHIVE_ZSTD_LEVEL"
#: x264 preset of the portal clips (default medium); a display derivative of the native
#: frames, which keep their sha256 in the clip's manifest.
CLIP_PRESET_ENV = "TENNIS_PORTAL_CLIP_PRESET"


def _zstd_level() -> tuple[str, ...]:
    level = os.environ.get(ZSTD_LEVEL_ENV)
    if not level:
        return ZSTD_LEVEL
    if not level.isdigit() or not 1 <= int(level) <= 19:
        raise ValueError(f"{ZSTD_LEVEL_ENV} must be a zstd level 1-19")
    return (f"-{int(level)}", "--long=30")


UNZSTD = ("zstd", "-d", "--long=30")
#: Archived whole, then deleted; the listed files are kept unpacked beside the archive.
S6_TREES = ("fit", "refit", "beam", "cascade/packets", "cascade/manifests")
#: The beam's summaries stay readable: ``assemble`` and the outcome backfill read BEAM.json.
S6_UNPACKED = (
    "fit/summary.json",
    "refit/REFIT.json",
    "beam/BEAM.json",
    "beam/CALIBRATION.json",
    "beam/STATE.json",
)
#: Native pictures S6-only reprocessing reads (near-baseline refinement candidates).
FRAMES_ARCHIVE = "archive/s6_frames.tar.zst"
UPSTREAM_UNPACKED_NAMES = (
    "automatic_point_ledger_v1.csv",
    "automatic_point_ledger_v1.json",
    "source_integrity.json",
)
#: Upstream bulk under ``upstream/<match>/`` and ``upstream/``: intermediate pictures and
#: tensors no product output reads once the portal clips exist.
UPSTREAM_MATCH_BULK = (
    "audit_frames_native_1080",
    "frames_native",
    "frames_small",
    "overlay_crops",
    "score_crops_native",
)
#: ``lean`` also deletes the torso-lock guide streams. ``standard`` archives them with
#: upstream: the composed ball track's sidecar declares them as its guide, and S6
#: preparation of a reprocessed attempt reads and binds them (~80 MB compressed per match).
LEAN_UPSTREAM_MATCH_BULK = ("torso_lock_inputs_v1",)
UPSTREAM_BULK_GLOBS = ("event_crops_v1/*.npz",)


def _save(path: Path, document: dict) -> None:
    """Write through a new inode (never in place: a match may be a hardlink clone)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, default=str) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_bytes(root: Path) -> int:
    """Allocated bytes under ``root``, each inode once, symlinks not followed."""
    return _allocated(Path(root), set())


def _allocated(root: Path, seen: set) -> int:
    total = 0
    if not root.exists() and not root.is_symlink():
        return 0
    if not root.is_dir() or root.is_symlink():
        stat = os.lstat(root)
        if (stat.st_dev, stat.st_ino) in seen:
            return 0
        seen.add((stat.st_dev, stat.st_ino))
        return stat.st_blocks * 512
    for folder, directories, files in os.walk(root):
        for name in files + directories:
            try:
                stat = os.lstat(os.path.join(folder, name))
            except FileNotFoundError:
                continue
            if (stat.st_dev, stat.st_ino) in seen:
                continue
            seen.add((stat.st_dev, stat.st_ino))
            total += stat.st_blocks * 512
    return total


def inventory(out: Path) -> dict[str, int]:
    """Allocated bytes per top-level entry, each inode counted once for the whole match."""
    seen: set = set()
    return {entry.name: _allocated(entry, seen) for entry in sorted(Path(out).iterdir())}


# --------------------------------------------------------------------------- portal


def _index(portal: Path) -> dict:
    return _load(portal / "data" / "index.json")


def kept_portal_keys(portal: Path, attempts: str) -> set[str]:
    """Frame folders the compacted portal keeps: index attempts, optionally only accepted."""
    if attempts not in FRAME_ATTEMPTS:
        raise ValueError(f"unknown frame attempts policy {attempts}")
    return {
        point["clip"]
        for point in _index(portal)["points"]
        if attempts == "all" or (point.get("accepted_flights") or 0) > 0
    }


# Mean grey-level change between consecutive native thumbnails below which a clip window is
# too still to show a one-frame shift through compression error (~1.2 grey levels at CRF).
STATIC_MOTION = 0.5
# A real shift of k pictures raises the zero-shift error by about the window's picture-to-picture
# motion; a best shift that beats zero by less than this share of it is compression noise.
SHIFT_MARGIN = 0.25


def encode_clip(folder: Path, destination: Path, *, rate: float, ffmpeg: str) -> dict:
    """One H.264 clip of a frame folder's contiguous ``f_%04d.jpg`` range."""
    manifest = _load(folder / "manifest.json")
    first, last = (int(v) for v in manifest["frame_range"])
    expected = [folder / f"f_{frame:04d}.jpg" for frame in range(first, last + 1)]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{folder}: missing {missing[:3]}")
    temporary = destination.with_suffix(".tmp.mp4")
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{rate:g}",
        "-start_number",
        str(first),
        "-i",
        str(folder / "f_%04d.jpg"),
        "-vf",
        f"scale={VIDEO_WIDTH}:-2",
        "-c:v",
        "libx264",
        "-preset",
        os.environ.get(CLIP_PRESET_ENV, "medium"),
        "-crf",
        str(VIDEO_CRF),
        "-g",
        str(VIDEO_GOP),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        os.environ.get(CLIP_THREADS_ENV, "1"),
        str(temporary),
    ]
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
    temporary.replace(destination)
    return {
        "schema": FRAMES_SCHEMA,
        "file": destination.name,
        "first_frame": first,
        "last_frame": last,
        "frames": last - first + 1,
        "rate": rate,
        "time_of_frame": "(frame - first_frame + 0.5) / rate seconds into the clip",
        "width": VIDEO_WIDTH,
        "codec": "h264",
        "crf": VIDEO_CRF,
        "gop": VIDEO_GOP,
        # Recorded only when the switch moves it, so default documents stay as they were.
        **(
            {"preset": preset}
            if (preset := os.environ.get(CLIP_PRESET_ENV, "medium")) != "medium"
            else {}
        ),
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "source_manifest_sha256": _sha256(folder / "manifest.json"),
        "derivation": (
            "display derivative of the native frames listed in manifest.json (their sha256 "
            "stays there); frame numbers and native timestamps are unchanged"
        ),
    }


def verify_clip(
    folder: Path, video: dict, window: int = 60, shifts: int = 2, candidates: int = 5
) -> dict:
    """Decode the clip: exact picture count, and picture i aligned with native frame first+i.

    Alignment is the shift in ``-shifts..shifts`` with the lowest mean absolute error over a
    ``window`` of consecutive pictures (small grey thumbnails). The window is the one with the
    most native picture-to-picture motion among ``candidates`` spread over the clip: on a nearly
    still window compression error swamps the frame-to-frame change and the best shift is noise
    (a Sinner-Rublev pilot clip read -2 at 0.1 grey levels of motion, 0 elsewhere). A window
    whose motion is below ``STATIC_MOTION`` grey levels, or whose shifts all score within 0.05,
    is reported as inconclusive, not failed; so is a best shift that beats zero by less than
    ``SHIFT_MARGIN`` of the window motion.
    """
    import cv2
    import numpy as np

    def thumb(picture):
        grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey, (240, 135), interpolation=cv2.INTER_AREA).astype(np.int16)

    capture = cv2.VideoCapture(str(folder / video["file"]))
    decoded = []
    while True:
        ok, picture = capture.read()
        if not ok:
            break
        decoded.append(thumb(picture))
    capture.release()
    count = video["frames"]
    if len(decoded) != count:
        raise ValueError(f"{folder}: clip has {len(decoded)} pictures, expected {count}")
    span = min(window, count - 2 * shifts)
    if span <= 0:
        return {"alignment": "too_short", "pictures": len(decoded)}
    last_start = count - shifts - span
    starts = sorted({int(round(v)) for v in np.linspace(shifts, last_start, max(1, candidates))})
    native: dict[int, np.ndarray] = {}

    def native_at(index: int) -> np.ndarray:
        if index not in native:
            path = folder / f"f_{video['first_frame'] + index:04d}.jpg"
            native[index] = thumb(cv2.imread(str(path)))
        return native[index]

    def motion(lo: int) -> float:
        return float(
            np.mean(
                [np.mean(np.abs(native_at(i) - native_at(i - 1))) for i in range(lo + 1, lo + span)]
            )
        )

    moves = {lo: motion(lo) for lo in starts}
    lo = max(moves, key=moves.get)
    hi = lo + span
    errors = {
        shift: float(
            np.mean([np.mean(np.abs(decoded[i] - native_at(i + shift))) for i in range(lo, hi)])
        )
        for shift in range(-shifts, shifts + 1)
    }
    best = min(errors, key=errors.get)
    spread = max(errors.values()) - min(errors.values())
    if spread < 0.05:
        status = "inconclusive_static"
    elif best == 0:
        status = "aligned"
    elif moves[lo] < STATIC_MOTION:
        status = "inconclusive_static"
    elif errors[0] - errors[best] < SHIFT_MARGIN * moves[lo]:
        # Low-contrast motion (a Kenin-Gauff dry-run rally: 1.1 grey levels of motion, shift -1
        # beating 0 by 0.1) is within compression error of zero; not evidence of a shift.
        status = "inconclusive_noise"
    else:
        raise ValueError(f"{folder}: clip pictures are shifted by {best} frames")
    return {
        "alignment": status,
        "pictures": len(decoded),
        "window": [video["first_frame"] + lo, video["first_frame"] + hi - 1],
        "window_motion": round(moves[lo], 3),
        "mae_by_shift": {str(k): round(v, 3) for k, v in errors.items()},
    }


def refresh_viewer(portal: Path) -> str:
    """Install the current viewer (it reads ``frames_video``), never writing through a link."""
    from cv.viz.portal_3d_src import build

    for name in (*build.STATIC, *(f"vendor/{name}" for name in build.VENDOR)):
        (portal / name).unlink(missing_ok=True)
    return build.copy_static(portal)


def _documents_for(portal: Path, key: str) -> list[Path]:
    return [portal / "data" / f"{key}.json"]


def _set_document(path: Path, **fields) -> None:
    if not path.is_file():
        return
    document = _load(path)
    if all(document.get(k) == v for k, v in fields.items()):
        return
    document.update(fields)
    _save(path, document)


def compact_portal_frames(
    portal: Path,
    *,
    mode: str = "video",
    attempts: str = "accepted",
    workers: int = 8,
    ffmpeg: str | None = None,
) -> dict:
    """Replace hardlinked native frame folders with the kept portal pictures (idempotent)."""
    if mode not in FRAME_MODES:
        raise ValueError(f"unknown portal frame mode {mode}")
    frames = portal / "frames"
    if mode == "native" or not frames.is_dir():
        return {"mode": mode, "skipped": True}
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if mode == "video" and not ffmpeg:
        raise RuntimeError("ffmpeg is required for portal video frames")
    keep = kept_portal_keys(portal, attempts) if mode == "video" else set()
    index = _index(portal)
    rates = {
        point["clip"]: float(_load(portal / "data" / point["file"]).get("fps") or 25.0)
        for point in index["points"]
        if point["clip"] in keep
    }

    def one(key: str) -> dict:
        folder = frames / key
        declared = folder / "frames_video.json"
        if declared.is_file():
            video = _load(declared)
        else:
            video = encode_clip(folder, folder / "clip.mp4", rate=rates[key], ffmpeg=ffmpeg)
            video.update(verify_clip(folder, video))
            _save(declared, video)
        for picture in folder.glob("f_*.jpg"):
            picture.unlink()
        for document in _documents_for(portal, key):
            _set_document(document, frames_video=video)
        return {"key": key, "bytes": video["bytes"], "frames": video["frames"]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        encoded = list(pool.map(one, sorted(k for k in keep if (frames / k).is_dir())))
    dropped = []
    for folder in sorted(p for p in frames.iterdir() if p.is_dir() and p.name not in keep):
        shutil.rmtree(folder)
        dropped.append(folder.name)
    kept_index = {point["clip"] for point in index["points"]} & keep
    changed = False
    for point in index["points"]:
        available = point["clip"] in kept_index
        if point.get("frames_available") != available:
            point["frames_available"] = available
            changed = True
        if not available:
            for document in _documents_for(portal, point["clip"]):
                _set_document(document, frames_available=False)
    for row in (index.get("automatic_s6") or {}).get("held_before_fitting") or []:
        if row.get("frames_available"):
            row["frames_available"] = False
            changed = True
            for document in _documents_for(portal, row["clip"]):
                _set_document(document, frames_available=False)
    if changed:
        _save(portal / "data" / "index.json", index)
    return {
        "mode": mode,
        "viewer_version": refresh_viewer(portal),
        "attempts": attempts,
        "clips": len(encoded),
        "clip_frames": sum(e["frames"] for e in encoded),
        "clip_bytes": sum(e["bytes"] for e in encoded),
        "dropped_frame_folders": len(dropped),
    }


# --------------------------------------------------------------------------- bulk


def hash_and_drop_evidence(out: Path) -> dict:
    evidence = out / "cascade" / "adjudicate" / "evidence"
    listing = out / "cascade" / "adjudicate" / "evidence_hashes.json"
    if not evidence.is_dir():
        return {"evidence_files": 0, "listing": listing.is_file()}
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    _save(
        listing,
        {
            "schema": "cascade_evidence_hashes_v1",
            "note": "pictures sent to the adjudicating models, deleted after hashing",
            "files": {
                str(p.relative_to(evidence)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
                for p in files
            },
        },
    )
    shutil.rmtree(evidence)
    return {"evidence_files": len(files), "listing": True}


def _members(path: Path) -> int:
    """Entries ``tar`` lists for ``path``: itself, every folder, file and symlink below."""
    if not path.is_dir() or path.is_symlink():
        return 1
    count = 1
    for _, directories, files in os.walk(path):
        count += len(directories) + len(files)
    return count


def _archived_already(out: Path, name: str, unpacked: tuple[str, ...]) -> bool:
    """A tree already archived: nothing left of it but the files kept unpacked."""
    left = {str(p.relative_to(out)) for p in (out / name).rglob("*") if not p.is_dir()}
    return left <= set(unpacked)


def _listed(archive: Path) -> list[str]:
    return subprocess.run(
        ["tar", "-I", " ".join(UNZSTD), "-tf", str(archive)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


def _write_archive(out: Path, archive: Path, trees: list[str], verify: str) -> None:
    """``trees`` into ``archive``: written under a temporary name, renamed once verified.

    ``count``: the archive lists every member; ``compare``: in addition ``tar --compare``
    reads every archived file back against the tree, byte for byte.
    """
    members = sum(_members(out / tree) for tree in trees)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        tar = subprocess.Popen(["tar", "-C", str(out), "-cf", "-", *trees], stdout=subprocess.PIPE)
        subprocess.run(
            ["zstd", "-q", *_zstd_level(), "-T8", "-c"], stdin=tar.stdout, stdout=handle, check=True
        )
        if tar.wait() != 0:
            raise RuntimeError("tar failed")
    listed = _listed(temporary)
    if len(listed) != members:
        raise RuntimeError(f"{archive.name}: lists {len(listed)} members, expected {members}")
    if verify == "compare":
        compared = subprocess.run(
            ["tar", "-I", " ".join(UNZSTD), "--compare", "-f", str(temporary), "-C", str(out)],
            capture_output=True,
            text=True,
        )
        if compared.returncode != 0:
            raise RuntimeError(f"{archive.name}: differs from the tree: {compared.stdout[:500]}")
    temporary.replace(archive)


def archive_trees(
    out: Path,
    name: str,
    trees: tuple[str, ...],
    unpacked: tuple[str, ...],
    *,
    verify: str = "count",
) -> dict:
    """Tar+zstd ``trees`` into ``archive/<name>.tar.zst``, verify, delete, restore ``unpacked``.

    Safe to rerun at any point: the archive is written under a temporary name and renamed
    only after it verifies; deletion happens only once it exists. A tree an existing archive
    does not hold (a tree added to the policy after the match was compacted, e.g. ``beam``)
    goes into its own ``archive/<name>.<tree>.tar.zst`` first, never deleted unarchived.
    """
    archive = out / "archive" / f"{name}.tar.zst"
    present = [
        tree
        for tree in trees
        if (out / tree).exists()
        and not ((out / tree).is_dir() and _archived_already(out, tree, unpacked))
    ]
    if not present:
        return {"archive": str(archive.relative_to(out)) if archive.is_file() else None}
    kept = {path: (out / path).read_bytes() for path in unpacked if (out / path).is_file()}
    members = sum(_members(out / tree) for tree in present)
    written = []
    if not archive.is_file():
        _write_archive(out, archive, present, verify)
        written.append(archive)
    else:
        listed = _listed(archive)
        held = {
            tree
            for tree in present
            if any(m.rstrip("/") == tree or m.startswith(f"{tree}/") for m in listed)
        }
        for tree in [t for t in present if t not in held]:
            extra = out / "archive" / f"{name}.{tree.replace('/', '_')}.tar.zst"
            if not extra.is_file():
                _write_archive(out, extra, [tree], verify)
            written.append(extra)
    for tree in present:
        path = out / tree
        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    for path, content in kept.items():
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(content)
    return {
        "archive": str(archive.relative_to(out)),
        "archived": present,
        "kept_unpacked": sorted(kept),
        "members": members,
        "written": [
            {
                "archive": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "verify": verify,
            }
            for path in written
        ],
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "extract": f"tar -I '{' '.join(UNZSTD)}' -xf {archive.relative_to(out)}",
    }


def drop_trees(out: Path, trees: tuple[str, ...], unpacked: tuple[str, ...]) -> list[str]:
    """``lean``: delete the S6 stage trees outright, keeping only ``unpacked``."""
    kept = {path: (out / path).read_bytes() for path in unpacked if (out / path).is_file()}
    dropped = []
    for tree in trees:
        path = out / tree
        if path.exists() and not _archived_already(out, tree, unpacked):
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
            dropped.append(tree)
    for path, content in kept.items():
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(content)
    return dropped


def refinement_frames(out: Path) -> list[str]:
    """Data-relative pictures an S6 rerun of this match's fit jobs may read, plus receipts.

    Near-baseline camera refinement reads up to eight native pictures per intrinsic-fallback
    camera group (``court_near_baseline_refinement.candidate_frames``); nothing else in an S6
    fit reads pixels. Each referenced clip's ``extraction_receipt.json`` (source PTS and
    sha256 of every native picture) is kept too, so any other picture can be re-extracted
    from the video and verified.
    """
    from cv.pipeline import paths
    from cv.pipeline.court_near_baseline_refinement import candidate_frames

    root = paths.data_root()
    jobs = sorted((out / "fit" / "attempts").glob("*/job.json"))
    jobs += sorted((out / "fit" / "attempts").glob("*/child_*/job.json"))
    jobs += sorted((out / "refit").glob("*/job.json"))
    jobs += sorted((out / "beam").glob("*/*/*/job.json"))
    wanted: set[str] = set()
    for job in jobs:
        document = _load(job)
        if (document.get("policy") or {}).get("near_baseline_refinement") != "on":
            continue
        row = document["row"]
        records = [row.get("cameras"), row.get("labels")]
        if any(not r or r.get("path_base") != "TENNIS_DATA_ROOT" for r in records):
            continue
        cameras_path, labels_path = (root / r["path"] for r in records)
        if not cameras_path.is_file() or not labels_path.is_file():
            continue
        frames = candidate_frames(_load(cameras_path))
        images = (_load(labels_path).get("source_pack") or {}).get("images") or []
        for image in images:
            source = image.get("source") or {}
            if source.get("path_base") != "TENNIS_DATA_ROOT" or not source.get("path"):
                continue
            receipt = str(Path(source["path"]).parent / "extraction_receipt.json")
            if (root / receipt).is_file():
                wanted.add(receipt)
            if int(image["frame"]) in frames and (root / source["path"]).is_file():
                wanted.add(source["path"])
    return sorted(wanted)


def retain_refinement_frames(out: Path, destination: Path | None = None) -> dict:
    """``archive/s6_frames.tar.zst``: the pictures S6-only reprocessing needs (verified).

    ``destination``: write the archive under another match directory (read-only sources).
    """
    from cv.pipeline import paths

    archive = (destination or out) / FRAMES_ARCHIVE
    if archive.is_file():
        return {"archive": FRAMES_ARCHIVE, "already": True}
    members = refinement_frames(out)
    if not members:
        return {"archive": None, "members": 0}
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp")
    listing = archive.parent / "s6_frames.members"
    listing.write_text("\n".join(members) + "\n")
    try:
        with temporary.open("wb") as handle:
            tar = subprocess.Popen(
                ["tar", "-C", str(paths.data_root()), "-cf", "-", "-T", str(listing)],
                stdout=subprocess.PIPE,
            )
            subprocess.run(
                ["zstd", "-q", "-3", "-T4", "-c"], stdin=tar.stdout, stdout=handle, check=True
            )
            if tar.wait() != 0:
                raise RuntimeError("tar failed")
        listed = subprocess.run(
            ["tar", "-I", " ".join(UNZSTD), "-tf", str(temporary)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if sorted(listed) != members:
            raise RuntimeError(
                f"s6 frames archive lists {len(listed)} members, expected {len(members)}"
            )
        temporary.replace(archive)
    finally:
        listing.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return {
        "archive": FRAMES_ARCHIVE,
        "members": len(members),
        "pictures": sum(m.endswith(".jpg") for m in members),
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "paths": "TENNIS_DATA_ROOT-relative (extract with -C <data root>)",
    }


def drop_upstream_bulk(out: Path, match_id: str, keep: str = "standard") -> list[dict]:
    upstream = out / "upstream"
    names = UPSTREAM_MATCH_BULK + (LEAN_UPSTREAM_MATCH_BULK if keep == "lean" else ())
    targets = [upstream / match_id / name for name in names]
    for pattern in UPSTREAM_BULK_GLOBS:
        targets.extend(sorted(upstream.glob(pattern)))
    dropped = []
    for path in targets:
        if not path.exists() or path.is_symlink():
            continue
        size = disk_bytes(path)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        dropped.append({"path": str(path.relative_to(out)), "bytes": size})
    return dropped


def compact(
    out: Path,
    match_id: str,
    *,
    frames: str = "video",
    frame_attempts: str = "accepted",
    keep: str = "standard",
    workers: int = 8,
) -> dict:
    """Compact one finished match directory in place; ``ARTIFACTS.json`` records the result."""
    if keep not in KEEP:
        raise ValueError(f"unknown keep policy {keep}")
    out = Path(out)
    record_path = out / "ARTIFACTS.json"
    previous = _load(record_path) if record_path.is_file() else {}
    before = previous.get("bytes_before") or inventory(out)
    _save(record_path, {**previous, "schema": SCHEMA, "status": "running", "bytes_before": before})
    report: dict = {"keep": keep}
    if keep in ("standard", "lean"):
        report["portal_frames"] = compact_portal_frames(
            out / "portal" / "3d", mode=frames, attempts=frame_attempts, workers=workers
        )
        report["cascade_evidence"] = hash_and_drop_evidence(out)
        if keep == "standard":
            # Before the native frame folders, the event crops and the fit trees go.
            report["s6_frames"] = retain_refinement_frames(out)
            report["reprocess_bundles"] = reprocess_bundle.write(out, match_id, workers=workers)
        report["upstream_dropped"] = previous.get("upstream_dropped", []) + drop_upstream_bulk(
            out, match_id, keep
        )
        for partial in out.glob("fit.partial.*"):
            shutil.rmtree(partial)
        if keep == "standard":
            report["s6_archive"] = archive_trees(out, "s6_stages", S6_TREES, S6_UNPACKED)
        else:
            report["s6_deleted"] = drop_trees(out, S6_TREES, S6_UNPACKED)
        report["upstream_archive"] = archive_trees(
            out,
            "upstream",
            ("upstream",),
            tuple(f"upstream/{match_id}/{name}" for name in UPSTREAM_UNPACKED_NAMES),
        )
    after = inventory(out)
    document = {
        "schema": SCHEMA,
        "status": "done",
        "match_id": match_id,
        "bytes_before": before,
        "bytes_after": after,
        "total_before": sum(before.values()),
        "total_after": sum(after.values()),
        **report,
        "kept": {
            "MATCH_RESULT.json": "per attempt final flights, 3D states, cascade $, step times",
            "RECEIPT.json": "queue receipt: models, tokens, $, GPU/CPU/wall per step, bytes",
            "OUTCOMES.json": "per stage/attempt/component/flight status and reason codes",
            "steps/": "per-step markers (resume)",
            "portal/3d/": "viewer, attempt documents, one H.264 clip per kept attempt",
            "cascade/": "candidates, decisions, replies, spend ledger, COST, evidence hashes",
            "archive/s6_stages.tar.zst": "fit/refit/beam stage trees, cascade packets, manifests",
            "archive/s6_frames.tar.zst": "native pictures an S6 refit reads, extraction receipts",
            "archive/upstream.tar.zst": "ball track (and its torso-lock guide), camera, players, "
            "events, receipts, manifests",
            "archive/reprocess_bundles.tar.zst": "per non-complete attempt: native extraction "
            "receipt, event crop slice digests; the native frame folder's listing",
            "archive/REPROCESS_BUNDLES.json": "which attempts are bundled and why, source video, "
            "what is stored and what is regenerated (cv.pipeline.reprocess_bundle)",
            "upstream/<match>/": "automatic point ledger and source integrity, unpacked",
        },
        "deleted": {
            "portal frame folders": "hardlinked native JPEGs; components reuse parent pictures",
            "cascade/adjudicate/evidence": "model input pictures (sha256 kept)",
            "upstream bulk": list(UPSTREAM_MATCH_BULK)
            + list(UPSTREAM_BULK_GLOBS)
            + (list(LEAN_UPSTREAM_MATCH_BULK) if keep == "lean" else []),
        },
    }
    _save(record_path, document)
    return document


def rearchive_s6(out: Path) -> dict:
    """An already compacted match: archive the S6 trees its compaction left unpacked.

    For matches compacted before ``beam`` was archived (2026-10-01). Each such tree goes
    into its own archive, read back byte for byte against the tree (``tar --compare``)
    before the tree is deleted; ``ARTIFACTS.json`` records it under ``s6_rearchived``.
    Nothing else of the match is touched.
    """
    out = Path(out)
    before = disk_bytes(out)
    report = archive_trees(out, "s6_stages", S6_TREES, S6_UNPACKED, verify="compare")
    record_path = out / "ARTIFACTS.json"
    document = _load(record_path) if record_path.is_file() else {"schema": SCHEMA}
    entry = {**report, "at": time.time(), "bytes_before": before, "bytes_after": disk_bytes(out)}
    document["s6_rearchived"] = [*document.get("s6_rearchived", []), entry]
    document["total_after"] = entry["bytes_after"]
    _save(record_path, document)
    return entry


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--frames", choices=FRAME_MODES, default="video")
    parser.add_argument("--frame-attempts", choices=FRAME_ATTEMPTS, default="accepted")
    parser.add_argument("--keep", choices=KEEP, default="standard")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rearchive-s6",
        action="store_true",
        help="already compacted match: archive S6 trees left unpacked (verified), nothing else",
    )
    args = parser.parse_args()
    if args.rearchive_s6:
        print(json.dumps(rearchive_s6(args.out), indent=1, default=str))
        return
    result = compact(
        args.out,
        args.match_id,
        frames=args.frames,
        frame_attempts=args.frame_attempts,
        keep=args.keep,
        workers=args.workers,
    )
    print(json.dumps({k: result[k] for k in ("total_before", "total_after")}, indent=1))


if __name__ == "__main__":
    main()
