"""Run pose inference inside tracked-player crops for selected player boxes.

Full-frame pose inference commonly misses the far player even at 1080p. This stage uses
the established near/far player boxes as a localization prior, presents a larger local
crop to the pose model, and writes native-resolution COCO keypoints for both players.

``--observation-scope`` selects which actor rows are presented. ``active_play`` keeps the
established behaviour: only rows inside a point's decided active spans. ``retained_native``
presents every retained actor row of the original box stream whose picture the clip's own
``player_frame_extraction_v1`` receipt lists, whatever its side, confidence or active-span
status, so evidence outside the decided spans is not silently dropped. Membership is decided
by that receipt and not by a directory listing, so a picture the original extraction never
declared is recorded as unlisted rather than read. Neither scope invents rows: an unlisted or
undecodable picture, and a read picture the model returned nothing for, are counted apart from
the rows actually written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import artifact_cache, provenance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_crop_infer import pose_windows, window_for_box  # noqa: E402
from pose import KEYPOINT_NAMES, fieldnames, write_pose_manifest  # noqa: E402
import resolution as res  # noqa: E402
from run_manifest import StageRun  # noqa: E402


# The original per-clip native extraction receipt written by cv.pipeline.players, and read
# here for its source-signed inventory and the exact bytes decoded for pose inference.
EXTRACTION_RECEIPT = "extraction_receipt.json"
EXTRACTION_SCHEMA = "player_frame_extraction_v1"


def frame_number(value: str) -> int:
    return int(Path(value).stem.rsplit("_", 1)[-1])


def active_frame(
    frame: int,
    spans: list[list[float]],
) -> bool:
    return any(start <= frame <= end for start, end in spans)


def inventory_frame_path(frames_root: Path, clip: str, frame: str) -> Path:
    """The native picture an actor row was detected in, refusing names outside the inventory.

    The retained scope reads the box stream rather than a decided span list, so the frame
    inventory is the only thing bounding it: a row naming a directory component, an absolute
    path or a parent reference would reach pictures this run never extracted.
    """
    for part in (clip, frame):
        if not part or Path(part).name != part:
            raise ValueError(
                f"actor row names a picture outside the frame inventory: {clip}/{frame}"
            )
    return frames_root / clip / frame


def confidence_status(row: dict[str, str]) -> str:
    """``unknown`` when the box stream declares no confidence, ``zero`` when it declares none."""
    value = row.get("conf", "")
    if value in ("", None):
        return "unknown"
    try:
        number = float(value)
    except ValueError:
        return "unknown"
    return "zero" if number == 0.0 else "positive"


def extraction_inventory(frames_root: Path, clip: str) -> dict[str, str] | None:
    """The picture names the clip's original extraction receipt declares, or ``None``.

    ``None`` means this run holds no usable declaration of what was extracted for the clip.
    That is unavailable evidence, not a claim that the pictures on disk are original, so the
    retained scope reads none of them and counts the rows instead.
    """
    path = inventory_frame_path(frames_root, clip, EXTRACTION_RECEIPT)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("schema") != EXTRACTION_SCHEMA:
        return None
    source, identity = receipt.get("source_identity"), receipt.get("identity")
    if not isinstance(source, dict) or not isinstance(identity, dict):
        return None
    try:
        provenance.assert_automatic_document(source, context="retained native pose pictures")
        configuration = source["configuration"]
        fingerprint = source["fingerprint"]
        unsigned = {key: value for key, value in source.items() if key != "fingerprint"}
        if (
            configuration.get("native") is not True
            or configuration.get("preserve_source_fps") is not True
            or not np.isfinite(float(configuration["fps"]))
            or float(configuration["fps"]) <= 0
            or artifact_cache._digest_json(unsigned) != fingerprint
            or identity.get("source_fingerprint") != fingerprint
            or clip != f"pt{int(identity['point']['pt']):04d}"
        ):
            return None
    except (KeyError, TypeError, ValueError, provenance.ProvenanceError):
        return None
    frames = receipt.get("frames")
    if not isinstance(frames, list):
        return None
    inventory: dict[str, str] = {}
    for entry in frames:
        name = entry.get("name") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or name in inventory
        ):
            return None
        inventory[name] = digest
    return inventory or None


def extraction_receipts(frames_root: Path, clips: list[str]) -> list[Path]:
    return [
        path
        for path in (inventory_frame_path(frames_root, clip, EXTRACTION_RECEIPT) for clip in clips)
        if path.is_file()
    ]


def select_rows(
    boxes_path: Path,
    *,
    scope: str,
    match_id: str,
    clips: set[str],
    active: dict | None,
    frames_root: Path,
) -> list[dict[str, str]]:
    """The scope's actor rows: its decided active spans, or the whole retained box stream."""
    rows = []
    with boxes_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            clip = row["clip"]
            if clips and clip not in clips:
                continue
            if scope == "active_play":
                decision = active.get(f"{match_id}/{clip}")
                if decision is None:
                    continue
                if not active_frame(frame_number(row["frame"]), decision["active_spans"]):
                    continue
            else:
                inventory_frame_path(frames_root, clip, row["frame"])
            rows.append(row)
    rows.sort(key=lambda row: (row["clip"], frame_number(row["frame"]), row["side"]))
    return rows


def listed_rows(
    rows: list[dict[str, str]],
    frames_root: Path,
) -> tuple[list[dict[str, str]], dict]:
    """Split retained rows into those the original extraction declared, and why the rest are not.

    The row universe stays the box stream. A picture the extraction receipt does not list is
    not read even when a file of that name exists, because this stage cannot say such a file
    is the original exposure.
    """
    inventories: dict[str, dict[str, str] | None] = {}
    presented: list[dict[str, str]] = []
    undeclared_clips: dict[str, int] = {}
    unlisted_frames: dict[str, int] = {}
    for row in rows:
        clip = row["clip"]
        if clip not in inventories:
            inventories[clip] = extraction_inventory(frames_root, clip)
        inventory = inventories[clip]
        if inventory is None:
            undeclared_clips[clip] = undeclared_clips.get(clip, 0) + 1
        elif row["frame"] not in inventory:
            unlisted_frames[clip] = unlisted_frames.get(clip, 0) + 1
        else:
            presented.append(row)
    return presented, {
        "clips": len(inventories),
        "clips_without_extraction_inventory": sorted(undeclared_clips),
        "rows_without_extraction_inventory": sum(undeclared_clips.values()),
        "rows_not_in_extraction_inventory": sum(unlisted_frames.values()),
        "clips_with_unlisted_rows": dict(sorted(unlisted_frames.items())),
    }


def coverage_report(
    rows: list[dict[str, str]],
    *,
    presented: list[dict[str, str]],
    inventory: dict,
    execution: dict,
    output_rows: list[dict],
    scope: str,
    boxes: str,
    frames_directory: str,
    output: str,
) -> dict:
    """What the scope asked for, what of it was actually read, and what came back.

    Requested rows are not evidence. Rows the extraction never declared are never read; a
    declared picture can still fail to decode; a decoded crop can return no pose; and a
    written row can carry no keypoint confidence at all. Each stays counted on its own
    instead of being folded into the written row count.
    """
    statuses = {"unknown": 0, "zero": 0}
    for row in rows:
        status = confidence_status(row)
        if status in statuses:
            statuses[status] += 1
    keypoint_confidences = [f"{name}_confidence" for name in KEYPOINT_NAMES]
    without_confidence = sum(
        all(float(row.get(name, 0) or 0) == 0.0 for name in keypoint_confidences)
        for row in output_rows
    )
    return {
        "schema": "tracked_crop_pose_coverage_v1",
        "observation_scope": scope,
        "boxes": boxes,
        "native_frames": frames_directory,
        "output": output,
        "requested_rows": len(rows),
        "requested_frames": len({(row["clip"], row["frame"]) for row in rows}),
        "extraction_inventory": inventory,
        "picture_bytes_verified_against_inventory": True,
        "source_hash_mismatch_rows": execution.get("source_hash_mismatch", 0),
        "rows_zero_confidence": statuses["zero"],
        "rows_unknown_confidence": statuses["unknown"],
        "presented_rows": len(presented),
        "decode_failed_rows": execution["decode_failed"],
        "no_pose_rows": execution["no_pose"],
        "resumed_chunks_without_counts": execution.get("chunks_without_counts", 0),
        "pose_rows": len(output_rows),
        "pose_rows_without_any_keypoint_confidence": without_confidence,
        "rows_without_pose_output": len(rows) - len(output_rows),
    }


def native_box_for_row(
    row: dict[str, str],
    artifact_size: res.FrameSize,
    image_size: res.FrameSize,
):
    native_keys = ("x0_native", "y0_native", "x1_native", "y1_native")
    if all(row.get(key, "") != "" for key in native_keys):
        return [float(row[key]) for key in native_keys]
    return res.scale_boxes(
        [float(row[key]) for key in ("x0", "y0", "x1", "y1")],
        artifact_size,
        image_size,
    )


def read_frame_cached(path: Path, cache: dict[Path, object]):
    """Decode a native frame once per chunk even when both player rows use it."""
    if path not in cache:
        cache[path] = cv2.imread(str(path))
    return cache[path]


def prefetch_frames(paths: list[Path], workers: int) -> dict[Path, object]:
    """Decode unique native frames concurrently without changing row or inference order."""
    unique = list(dict.fromkeys(paths))
    if workers <= 1:
        return {path: cv2.imread(str(path)) for path in unique}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        images = executor.map(lambda path: cv2.imread(str(path)), unique)
        return dict(zip(unique, images, strict=True))


def prefetch_verified_frames(paths: list[Path], digests: dict[Path, str], workers: int):
    """Decode only bytes belonging to the extraction, without a check/read race."""

    def read(path):
        try:
            data = path.read_bytes()
        except OSError:
            return path, None, "decode_failed"
        if hashlib.sha256(data).hexdigest() != digests[path]:
            return path, None, "source_hash_mismatch"
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return path, image, None if image is not None else "decode_failed"

    unique = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        decoded = list(executor.map(read, unique))
    return (
        {path: image for path, image, _ in decoded},
        {path: reason for path, _, reason in decoded if reason is not None},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_contract(
    args: argparse.Namespace,
    *,
    boxes_path: Path,
    active_path: Path | None,
    coordinate_path: Path,
    requested_rows: int,
    inventory_paths: list[Path] | None = None,
) -> dict:
    """Partial chunks are only reusable for the same scope over the same inputs.

    The default scope serializes exactly as it did before the scope existed, so a legacy
    interrupted run still resumes its own chunks. The retained scope declares itself and the
    extraction receipts that decide which pictures it may read, because its rows depend on
    that inventory and not only on the box stream.
    """
    model_path = Path(args.model).resolve()
    arguments = {key: value for key, value in vars(args).items() if key not in {"resume"}}
    if args.observation_scope == "active_play":
        arguments.pop("observation_scope", None)
    else:
        arguments["native_picture_verification"] = "extraction_sha256_v1"
    return {
        "schema": "tracked_crop_pose_resume_v1",
        "arguments": arguments,
        "requested_rows": requested_rows,
        "inputs": {
            "boxes": {"path": str(boxes_path.resolve()), "sha256": sha256_file(boxes_path)},
            **(
                {
                    "active_play": {
                        "path": str(active_path.resolve()),
                        "sha256": sha256_file(active_path),
                    }
                }
                if active_path is not None
                else {}
            ),
            "coordinates": {
                "path": str(coordinate_path.resolve()),
                "sha256": sha256_file(coordinate_path),
            },
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            **(
                {
                    "extraction_receipts": [
                        {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for path in inventory_paths
                    ]
                }
                if inventory_paths
                else {}
            ),
        },
    }


def prepare_resume_directory(
    path: Path,
    contract: dict,
    *,
    resume: bool,
    reset_incompatible: bool = False,
) -> None:
    manifest_path = path / "manifest.json"
    if path.exists() and not resume:
        shutil.rmtree(path)
    if path.exists():
        if not manifest_path.is_file():
            raise ValueError(f"pose resume directory has no manifest: {path}")
        if json.loads(manifest_path.read_text()) == contract:
            return
        if not reset_incompatible:
            raise ValueError(
                f"pose resume contract changed: {path}; rerun with --no-resume to reset"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows_atomic(path: Path, names: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument(
        "--active-play",
        help="decided active spans; required by, and only read under, --observation-scope "
        "active_play",
    )
    parser.add_argument(
        "--observation-scope",
        choices=("active_play", "retained_native"),
        default="active_play",
        help="which retained actor rows to present: the decided active spans (default), or "
        "every retained native row of the original box stream",
    )
    parser.add_argument("--point", action="append", default=[])
    parser.add_argument("--frames-dir", default="audit_frames_native_1080")
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--scales", default="1.0,1.5")
    parser.add_argument("--flips", default="0,1")
    parser.add_argument("--crop-factor", type=float, default=3.5)
    parser.add_argument(
        "--half",
        action="store_true",
        help="candidate FP16 inference path; enable only after output-equivalence benchmarking",
    )
    parser.add_argument("--output", default="player_pose_tracked_crop_native_v1.csv")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse contract-matched completed inference chunks after interruption",
    )
    parser.add_argument(
        "--reset-incompatible-resume",
        action="store_true",
        help="discard partial chunks when their input contract no longer matches",
    )
    args = parser.parse_args()

    out = Path(args.out)
    boxes_path = out / args.boxes
    coordinate_manifest = res.read_coordinate_manifest(boxes_path)
    if coordinate_manifest is None:
        raise ValueError(f"missing coordinate-space contract for {boxes_path}")
    artifact_size = res.manifest_artifact_size(coordinate_manifest)
    image_record = coordinate_manifest.get("image_size", {})
    image_size = res.FrameSize(
        int(image_record.get("width", 0)),
        int(image_record.get("height", 0)),
    )
    frames_root = out / args.frames_dir
    if args.observation_scope == "active_play" and args.active_play is None:
        raise ValueError("--observation-scope active_play requires --active-play")
    active_path = Path(args.active_play) if args.observation_scope == "active_play" else None
    active = json.loads(active_path.read_text()) if active_path is not None else None
    requested = select_rows(
        boxes_path,
        scope=args.observation_scope,
        match_id=args.match_id,
        clips=set(args.point),
        active=active,
        frames_root=frames_root,
    )
    inventory: dict = {}
    rows = requested
    inventory_paths: list[Path] = []
    native_digests: dict[Path, str] = {}
    if args.observation_scope == "retained_native":
        rows, inventory = listed_rows(requested, frames_root)
        clips = sorted({row["clip"] for row in requested})
        inventory_paths = extraction_receipts(frames_root, clips)
        native_digests = {
            frames_root / clip / name: digest
            for clip in clips
            for name, digest in (extraction_inventory(frames_root, clip) or {}).items()
        }

    from ultralytics import YOLO

    model = YOLO(args.model)
    scales = tuple(float(value) for value in args.scales.split(","))
    flips = tuple(bool(int(value)) for value in args.flips.split(","))
    output_rows = []
    names = fieldnames() + [
        "court_x",
        "court_y",
        "side",
        "track_id",
        "n_variants",
        "track_x0",
        "track_y0",
        "track_x1",
        "track_y1",
    ]
    output_path = out / args.output
    resume_dir = out / f".{args.output}.chunks"
    prepare_resume_directory(
        resume_dir,
        resume_contract(
            args,
            boxes_path=boxes_path,
            active_path=active_path,
            coordinate_path=boxes_path.with_suffix(boxes_path.suffix + ".coordinates.json"),
            requested_rows=len(rows),
            inventory_paths=inventory_paths,
        ),
        resume=args.resume,
        reset_incompatible=args.reset_incompatible_resume,
    )
    stage = StageRun(
        str(out),
        "player_pose_tracked_crop"
        if args.observation_scope == "active_play"
        else "player_pose_retained_native_crop",
        args,
        gpu_devices=[int(str(args.device).split(":")[-1])],
    )
    execution = {
        "decode_failed": 0,
        "source_hash_mismatch": 0,
        "no_pose": 0,
        "chunks_without_counts": 0,
    }
    for start in range(0, len(rows), args.chunk):
        chunk = rows[start : start + args.chunk]
        chunk_path = resume_dir / f"chunk_{start:09d}_{start + len(chunk):09d}.csv"
        counts_path = chunk_path.with_suffix(".counts.json")
        if chunk_path.is_file():
            output_rows.extend(read_csv_rows(chunk_path))
            # A resumed chunk keeps the outcome of the pictures it actually read, so the
            # coverage of an interrupted run is the same as that of an uninterrupted one. A
            # chunk whose counts are gone is declared, not counted as a clean chunk.
            if counts_path.is_file():
                resumed = json.loads(counts_path.read_text())
                for key in ("decode_failed", "source_hash_mismatch", "no_pose"):
                    execution[key] += int(resumed.get(key, 0))
            else:
                execution["chunks_without_counts"] += 1
            print(f"pose crops {start + len(chunk)}/{len(rows)} (resumed)", flush=True)
            continue
        chunk_execution = {"decode_failed": 0, "source_hash_mismatch": 0, "no_pose": 0}
        windows = []
        focus_boxes = []
        metadata = []
        frame_paths = [out / args.frames_dir / row["clip"] / row["frame"] for row in chunk]
        unread = {}
        if args.observation_scope == "retained_native":
            frame_cache, unread = prefetch_verified_frames(
                frame_paths, native_digests, max(args.decode_workers, 1)
            )
        else:
            frame_cache = prefetch_frames(frame_paths, max(args.decode_workers, 1))
        for row in chunk:
            path = out / args.frames_dir / row["clip"] / row["frame"]
            image = read_frame_cached(path, frame_cache)
            if image is None:
                chunk_execution[unread.get(path, "decode_failed")] += 1
                continue
            if image.shape[1] != image_size.width or image.shape[0] != image_size.height:
                raise ValueError(
                    f"frame {path} is {image.shape[1]}x{image.shape[0]}, expected "
                    f"{image_size.label} from {boxes_path}.coordinates.json"
                )
            native_box = native_box_for_row(row, artifact_size, image_size)
            x0, y0, side = window_for_box(
                native_box,
                image.shape[1],
                image.shape[0],
                args.crop_factor,
            )
            windows.append(image[y0 : y0 + side, x0 : x0 + side])
            focus_boxes.append(
                [
                    native_box[0] - x0,
                    native_box[1] - y0,
                    native_box[2] - x0,
                    native_box[3] - y0,
                ]
            )
            metadata.append((row, x0, y0, native_box))
        inferred = pose_windows(
            model,
            windows,
            scales=scales,
            flips=flips,
            device=args.device,
            batch=args.batch,
            focus_boxes=focus_boxes,
            half=args.half,
        )
        chunk_output_rows = []
        for (row, offset_x, offset_y, native_box), pose in zip(metadata, inferred):
            if pose is None:
                chunk_execution["no_pose"] += 1
                continue
            result = {
                "clip": row["clip"],
                "frame": row["frame"],
                "x0": round(float(pose["box"][0] + offset_x), 2),
                "y0": round(float(pose["box"][1] + offset_y), 2),
                "x1": round(float(pose["box"][2] + offset_x), 2),
                "y1": round(float(pose["box"][3] + offset_y), 2),
                "conf": round(float(pose["det_conf"]), 4),
                "court_x": row["court_x"],
                "court_y": row["court_y"],
                "side": row["side"],
                "track_id": row.get("track_id", ""),
                "n_variants": pose.get("n_variants", 0),
                "track_x0": round(float(native_box[0]), 2),
                "track_y0": round(float(native_box[1]), 2),
                "track_x1": round(float(native_box[2]), 2),
                "track_y1": round(float(native_box[3]), 2),
            }
            for index, name in enumerate(KEYPOINT_NAMES):
                result[f"{name}_x"] = round(
                    float(pose["kpts_xy"][index][0] + offset_x),
                    2,
                )
                result[f"{name}_y"] = round(
                    float(pose["kpts_xy"][index][1] + offset_y),
                    2,
                )
                result[f"{name}_confidence"] = round(
                    float(pose["kpts_conf"][index]),
                    4,
                )
            chunk_output_rows.append(result)
        # Counts first: a crash before the chunk CSV lands re-runs the chunk and rewrites them.
        counts_path.write_text(json.dumps(chunk_execution, sort_keys=True) + "\n")
        write_csv_rows_atomic(chunk_path, names, chunk_output_rows)
        for key, value in chunk_execution.items():
            execution[key] += value
        output_rows.extend(chunk_output_rows)
        print(f"pose crops {min(start + args.chunk, len(rows))}/{len(rows)}", flush=True)

    write_csv_rows_atomic(output_path, names, output_rows)
    # The crop stage writes whatever the source frames are, so a sub-native frame set would
    # otherwise emit a sub-native pose artifact that declares itself native by name. Route the
    # sidecar through the contract writer, which refuses an undeclared sub-native space.
    coordinate_path = Path(
        write_pose_manifest(
            str(output_path),
            image_size=image_size,
            artifact_size=image_size,
            source=str(out / args.frames_dir),
            subnative=image_size != res.NATIVE_SIZE,
        )
    )
    coverage = None
    if args.observation_scope == "retained_native":
        # The retained scope is bounded by the frame inventory rather than by decided spans,
        # so its execution coverage is written next to the artifact and stays readable when a
        # later run reuses the stage receipt instead of re-running the producer.
        coverage = out / f"{Path(args.output).stem}.coverage.json"
        document = coverage_report(
            requested,
            presented=rows,
            inventory=inventory,
            execution=execution,
            output_rows=output_rows,
            scope=args.observation_scope,
            boxes=args.boxes,
            frames_directory=args.frames_dir,
            output=args.output,
        )
        temporary = coverage.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(coverage)
    stage.finish(
        outputs={
            "pose_csv": str(output_path),
            "rows": len(output_rows),
            "requested_boxes": len(requested),
            "coordinate_manifest": str(coordinate_path),
            **({"coverage": str(coverage), "coverage_counts": document} if coverage else {}),
        }
    )
    shutil.rmtree(resume_dir)
    print(f"wrote {len(output_rows)}/{len(rows)} poses to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
