"""Player detection over aligned rally windows — Phase B, first stage.

For every point in ``point_video_map.csv`` with a rally window: extract frames at --fps,
run YOLO person detection (batched, GPU), and store per-frame boxes. Downstream, court
homography turns boxes into court positions and near/far player identities; here we only
persist raw detections plus a crude near/far split by box size (far player appears small).

    .venv/bin/python cv/pipeline/players.py --out data/processed/rg2025f \
        --video MATCH.mp4 --fps 8 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import glob
from fractions import Fraction
import bisect
import math
import re
import json
import os
import subprocess
from pathlib import Path

from cv.pipeline import resolution as res
from cv.pipeline.artifact_cache import stage_identity
from cv.pipeline.provenance import file_sha256
from cv.pipeline.broadcast_source import probe
from cv.pipeline.run_manifest import StageRun

PLAYER_LEGACY_COLUMNS = ("x0", "y0", "x1", "y1")
PLAYER_NATIVE_COLUMNS = ("x0_native", "y0_native", "x1_native", "y1_native")
PLAYER_FIELDS = (
    "pt",
    "t",
    "clip",
    "frame",
    *PLAYER_LEGACY_COLUMNS,
    "conf",
    *PLAYER_NATIVE_COLUMNS,
)


def rally_points(out_dir: str, point_map: str = "point_video_map.csv") -> list[dict]:
    pts = []
    with open(os.path.join(out_dir, point_map), newline="") as f:
        for r in csv.DictReader(f):
            if r["rally_t_start"]:
                pts.append(
                    {
                        "pt": int(r["pt"]),
                        "t0": float(r["rally_t_start"]),
                        "t1": float(r["rally_t_end"]),
                    }
                )
    return pts


def load_point_filter(path: str) -> set[int]:
    if not path:
        return set()
    with open(path) as f:
        return {int(line) for raw in f if (line := raw.strip()) and not line.startswith("#")}


def native_frame_window(start_seconds: float, end_seconds: float, fps: float) -> tuple[int, int]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    start_frame = round(start_seconds * fps)
    end_frame = round(end_seconds * fps)
    if end_frame <= start_frame:
        raise ValueError("native frame window must contain at least one frame")
    return start_frame, end_frame - 1


def frame_path_number(path: str) -> int:
    """Sort extracted frames numerically after the four-digit filename width is exceeded."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[-1])


def valid_jpeg_frame(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"\xff\xd8"
    except OSError:
        return False


def native_extract_command(
    video: str,
    output_pattern: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    *,
    native: bool,
    measured_native_epochs: bool = False,
) -> tuple[list[str], float, int]:
    """Build an exact native-frame seek that does not decode from frame zero."""
    start_seconds = start_frame / fps
    frame_count = end_frame - start_frame + 1
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        video,
    ]
    if measured_native_epochs:
        command[command.index("-v") + 1] = "info"
        command[command.index("-ss") + 1] = str(max(0.0, start_seconds - 1.0))
        command[command.index("-ss") : command.index("-ss")] = ["-seek_timestamp", "1"]
        command[command.index("-i") : command.index("-i")] = ["-copyts", "-noaccurate_seek"]
        # Input accurate-seek can trim relative to container start even with copyts.
        # Seek cheaply near the target, then select on original source PTS explicitly.
        selection = f"select=gte(pts\\,floor({start_seconds:.17g}/TB+0.5))"
        filters = [selection, *([] if native else ["scale=960:540"]), "showinfo"]
        command.extend(["-vf", ",".join(filters)])
    elif not native:
        command.extend(["-vf", "scale=960:540"])
    command.extend(
        [
            "-frames:v",
            str(frame_count),
            "-fps_mode",
            "passthrough",
            "-q:v",
            "2" if native else "4",
            output_pattern,
        ]
    )
    return command, start_seconds, frame_count


def captured_native_clock(log_path: Path, names: list[str], seek_seconds: float) -> dict:
    """Read actual output-picture epochs; extra decoder look-ahead is not an output."""
    text = log_path.read_text()
    clocks = re.findall(r"config in time_base:\s*(\d+/\d+)", text)
    if len(set(clocks)) != 1:
        raise ValueError("native extraction lacks one measured source timebase")
    clock = Fraction(clocks[0])
    if clock <= 0:
        raise ValueError("native extraction timebase must be positive")
    matches = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", text)
    if len(matches) < len(names):
        raise ValueError("native extraction lacks measured epochs for output pictures")
    rows = matches[: len(names)]
    if [int(index) for index, _ in rows] != list(range(len(names))):
        raise ValueError("native extraction picture epoch ordinals are incomplete")
    pts = [int(value) for _, value in rows]
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("native extraction epochs must increase")
    return dict(
        schema="native_extracted_picture_clock_v1",
        method="ffmpeg_copyts_showinfo",
        seek_seconds=seek_seconds,
        seek_timestamp=True,
        time_base=str(clock),
        frames=[dict(name=name, source_pts=value) for name, value in zip(names, pts, strict=True)],
    )


def reusable_native_clock(clock: object, names: list[str], seek_seconds: float) -> bool:
    """A malformed stored clock must trigger measurement, never stale-clock reuse."""
    try:
        if not isinstance(clock, dict) or not names:
            return False
        if (
            clock.get("schema") != "native_extracted_picture_clock_v1"
            or clock.get("method") != "ffmpeg_copyts_showinfo"
            or clock.get("seek_timestamp") is not True
            or clock.get("seek_seconds") != seek_seconds
        ):
            return False
        tb = Fraction(clock["time_base"])
        rows = clock["frames"]
        if tb <= 0 or [row["name"] for row in rows] != names:
            return False
        pts = [row["source_pts"] for row in rows]
        return all(type(value) is int for value in pts) and all(
            right > left for left, right in zip(pts, pts[1:])
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def source_clock_ordinals(receipt: dict, timestamps: list[float], fps: float) -> list[int]:
    """Authenticate measured picture epochs against actual full-source presentation order."""
    point = receipt["identity"]["point"]
    lo, hi = native_frame_window(point["t0"], point["t1"], fps)
    clock = receipt.get("native_clock")
    measured = receipt["source_identity"]["configuration"].get("native_epoch_capture")
    if not measured:
        if clock is not None:
            raise ValueError("undeclared native picture clock")
        # Historical zero-origin receipts retain their established mapping. An unmeasured
        # nonzero stream start cannot be authenticated by a requested nominal seek.
        if abs(timestamps[0]) > 0.00000051:
            raise ValueError("legacy native extraction lacks measured nonzero-origin epochs")
        return list(range(lo, min(hi + 1, len(timestamps))))
    if measured != "ffmpeg_copyts_showinfo_v1" or not isinstance(clock, dict):
        raise ValueError("native extraction measured clock missing or unsupported")
    if (
        clock.get("schema") != "native_extracted_picture_clock_v1"
        or clock.get("method") != "ffmpeg_copyts_showinfo"
    ):
        raise ValueError("unsupported measured native picture clock")
    if clock.get("seek_timestamp") is not True or clock.get("seek_seconds") != lo / fps:
        raise ValueError("native measured clock differs from requested absolute seek")
    tb = Fraction(clock["time_base"])
    if tb <= 0:
        raise ValueError("native picture timebase must be positive")
    rows = clock["frames"]
    if [row["name"] for row in rows] != [row["name"] for row in receipt["frames"]]:
        raise ValueError("native clock/picture inventories differ")
    epochs = []
    for row in rows:
        if type(row.get("source_pts")) is not int:
            raise ValueError("native picture requires an integer measured PTS")
        epochs.append(float(row["source_pts"] * tb))
    if not epochs:
        raise ValueError("native picture clock is empty")
    # FFprobe compact PTS is decimal to six places. This tolerance only matches
    # the serialization, never a picture shift or inferred cadence timestamp.
    tolerance = 0.00000051
    found = bisect.bisect_left(timestamps, epochs[0] - tolerance)
    if found >= len(timestamps) or abs(timestamps[found] - epochs[0]) > tolerance:
        raise ValueError("measured native picture epoch is absent from source")
    indices = list(range(found, found + len(epochs)))
    if indices[-1] >= len(timestamps) or any(
        abs(timestamps[i] - epoch) > tolerance for i, epoch in zip(indices, epochs, strict=True)
    ):
        raise ValueError("native picture epochs do not match contiguous source ordinals")
    seek = lo / fps
    source_threshold = math.floor(seek / float(tb) + 0.5) * float(tb)
    # index/fps can land exactly halfway between timebase ticks. Half-up then
    # sits one tick after the source frame at that index. A slice of an earlier
    # extraction still begins on that frame. One tick is not a picture shift
    # when the timebase is finer than half a frame; a coarser tick keeps the
    # strict threshold so a whole frame cannot sneak in.
    tick = float(tb)
    earliest = source_threshold - tick - tolerance if tick < (0.5 / fps) else source_threshold - tolerance
    if epochs[0] < earliest or (
        found > 0 and timestamps[found - 1] >= source_threshold - tolerance
    ):
        raise ValueError("native extraction begins outside its requested seek window")
    expected = min(hi - lo + 1, len(timestamps) - found)
    if len(epochs) != expected or any(not math.isfinite(t) for t in epochs):
        raise ValueError("native extraction output count differs from requested window")
    return indices


def player_detection_row(
    *,
    point: int,
    timestamp: float,
    frame_path: str,
    box,
    confidence: float,
    image_size: res.FrameSize,
    legacy_size: res.FrameSize = res.CANONICAL_SIZE,
) -> dict[str, str | int | float]:
    """Return byte-compatible legacy box values plus their exact native mirror."""
    legacy_box = res.scale_boxes(box, image_size, legacy_size)
    rounded_legacy = [round(float(value), 1) for value in legacy_box]
    native_box = res.scale_boxes(box, image_size, res.NATIVE_SIZE)
    rounded_native = [round(float(value), 1) for value in native_box]
    return {
        "pt": point,
        "t": round(timestamp, 2),
        "clip": os.path.basename(os.path.dirname(frame_path)),
        "frame": os.path.basename(frame_path),
        **dict(zip(PLAYER_LEGACY_COLUMNS, rounded_legacy, strict=True)),
        **dict(zip(PLAYER_NATIVE_COLUMNS, rounded_native, strict=True)),
        "conf": round(float(confidence), 3),
    }


def write_player_artifact(
    path: str,
    rows: list[dict],
    *,
    image_size: res.FrameSize,
    legacy_size: res.FrameSize,
    source: str,
    fps: float | None = None,
) -> None:
    """Write the dual player-box schema and its native-authoritative sidecar."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    res.write_native_dual_coordinate_manifest(
        f"{path}.coordinates.json",
        image_size=image_size,
        legacy_size=legacy_size,
        source=source,
        native_columns=PLAYER_NATIVE_COLUMNS,
        legacy_columns=PLAYER_LEGACY_COLUMNS,
        extra={
            "artifact": os.path.basename(path),
            "artifact_identity": res.PLAYER_BOXES_NATIVE_IDENTITY,
            **({"source_fps": fps} if fps is not None else {}),
        },
    )


def extract_clip_frames(
    video: str,
    out_dir: str,
    pts: list[dict],
    fps: float,
    native: bool = True,
    preserve_source_fps: bool = False,
    video_duration: float | None = None,
    dispositions: list[dict] | None = None,
    measured_native_epochs: bool = False,
) -> list[tuple]:
    """One ffmpeg call per point (seek + short decode). Returns (pt, frame_path, t)."""
    if measured_native_epochs and not preserve_source_fps:
        raise ValueError("measured native epochs require source-cadence extraction")
    # Hash the video once for the whole extraction, not once per point in a long match.
    source_identity = stage_identity(
        stage="player_frame_extraction",
        command=["python", str(Path(__file__).resolve())],
        configuration={
            "entrypoint": "cv.pipeline.players.extract_clip_frames",
            "fps": fps,
            "native": native,
            "preserve_source_fps": preserve_source_fps,
            **(
                {"native_epoch_capture": "ffmpeg_copyts_showinfo_v1"}
                if measured_native_epochs
                else {}
            ),
        },
        inputs=[Path(video)],
    )
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for p in pts:
        pdir = os.path.join(out_dir, f"pt{p['pt']:04d}")
        os.makedirs(pdir, exist_ok=True)
        timestamp_start = p["t0"]
        native_clock = None
        native_log = Path(pdir) / "native_extraction.log"
        existing = sorted(glob.glob(os.path.join(pdir, "f_*.jpg")), key=frame_path_number)
        receipt_path = Path(pdir) / "extraction_receipt.json"
        identity = {
            "source_fingerprint": source_identity["fingerprint"],
            "point": p,
            "video_duration": video_duration,
        }
        try:
            receipt = json.loads(receipt_path.read_text())
            reusable = (
                receipt.get("schema") == "player_frame_extraction_v1"
                and receipt.get("identity") == identity
                and receipt.get("frames")
                == [
                    {"name": Path(path).name, "sha256": file_sha256(Path(path))}
                    for path in existing
                ]
            )
            if measured_native_epochs and reusable:
                native_clock = receipt.get("native_clock")
                start_frame, _ = native_frame_window(p["t0"], p["t1"], fps)
                reusable = reusable_native_clock(
                    native_clock, [Path(path).name for path in existing], start_frame / fps
                )
        except (OSError, ValueError, TypeError):
            reusable = False
        if not reusable:
            native_clock = None
            # Only this extractor's generated JPEGs are invalidated. Preserve other files,
            # the source video, and all old worker artifacts outside this exact point folder.
            for path in existing:
                os.unlink(path)
            existing = []
            receipt_path.unlink(missing_ok=True)
        if existing and not all(valid_jpeg_frame(path) for path in existing):
            native_clock = None
            for path in existing:
                os.unlink(path)
            existing = []
        if preserve_source_fps:
            start_frame, end_frame = native_frame_window(p["t0"], p["t1"], fps)
            command, timestamp_start, expected_count = native_extract_command(
                video,
                os.path.join(pdir, "f_%04d.jpg"),
                start_frame,
                end_frame,
                fps,
                native=native,
                measured_native_epochs=measured_native_epochs,
            )
            expected_names = [f"f_{frame:04d}.jpg" for frame in range(1, expected_count + 1)]
            if not existing or (
                not reusable and [os.path.basename(path) for path in existing] != expected_names
            ):
                for path in existing:
                    os.unlink(path)
                if measured_native_epochs:
                    with native_log.open("w") as log:
                        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=log)
                else:
                    subprocess.run(command, check=True)
                existing = sorted(glob.glob(os.path.join(pdir, "f_*.jpg")), key=frame_path_number)
            actual_names = [os.path.basename(path) for path in existing]
            if actual_names != expected_names:
                contiguous_names = [f"f_{frame:04d}.jpg" for frame in range(1, len(existing) + 1)]
                terminal_window = (
                    video_duration is not None
                    and p["t1"] == max(point["t1"] for point in pts)
                    and p["t1"] >= video_duration - (1.0 / fps)
                    and abs(video_duration - (timestamp_start + len(existing) / fps))
                    <= max(2.0 / fps, 0.1)
                )
                if existing and actual_names == contiguous_names and terminal_window:
                    if dispositions is not None:
                        dispositions.append(
                            {
                                "point": int(p["pt"]),
                                "decision": "terminal_source_truncation",
                                "requested_frames": expected_count,
                                "observed_frames": len(existing),
                                "requested_end_seconds": float(p["t1"]),
                                "source_duration_seconds": float(video_duration),
                            }
                        )
                    expected_names = actual_names
                else:
                    raise RuntimeError(
                        f"native extraction emitted {len(existing)}/{expected_count} frames for "
                        f"point {p['pt']}"
                    )
        elif not existing:
            dur = max(1.0, p["t1"] - p["t0"])
            command = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                str(p["t0"]),
                "-i",
                video,
                "-t",
                str(dur),
            ]
            filters = [f"fps={fps}"]
            if not native:
                filters.append("scale=960:540")
            command.extend(["-vf", ",".join(filters)])
            command.extend(
                [
                    "-fps_mode",
                    "vfr" if preserve_source_fps else "cfr",
                    "-q:v",
                    "2" if native else "4",
                    os.path.join(pdir, "f_%04d.jpg"),
                ]
            )
            subprocess.run(command, check=True)
            existing = sorted(glob.glob(os.path.join(pdir, "f_*.jpg")), key=frame_path_number)
        if not existing:
            raise RuntimeError(f"extraction emitted no frames for point {p['pt']}")
        if measured_native_epochs:
            if native_clock is None:
                native_clock = captured_native_clock(
                    native_log, [Path(path).name for path in existing], timestamp_start
                )
            timestamp_start = float(
                Fraction(native_clock["time_base"]) * native_clock["frames"][0]["source_pts"]
            )
        document = {
            **({"native_clock": native_clock} if measured_native_epochs else {}),
            "schema": "player_frame_extraction_v1",
            "identity": identity,
            "source_identity": source_identity,
            "frames": [
                {"name": Path(path).name, "sha256": file_sha256(Path(path))} for path in existing
            ],
        }
        pending = receipt_path.with_suffix(".json.pending")
        pending.write_text(json.dumps(document, sort_keys=True) + "\n")
        pending.replace(receipt_path)
        for fp in existing:
            k = frame_path_number(fp)
            timestamp = (
                float(
                    Fraction(native_clock["time_base"])
                    * native_clock["frames"][k - 1]["source_pts"]
                )
                if measured_native_epochs
                else timestamp_start + (k - 1) / fps
            )
            index.append((p["pt"], fp, timestamp))
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument(
        "--preserve-source-fps",
        action="store_true",
        help="extract each decoded source frame once; --fps records the source timebase",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1920,
        help="YOLO inference size; the default keeps native 1920x1080 frames native "
        "(the previous implicit 640 letterboxed them to 640x384). 1920 finds about twice "
        "as many people per frame, but on the owner's 85 corrected roots the sided "
        "box-bottom root moved 3.530 to 3.803 px540 median, so pass --imgsz 640 to "
        "reproduce the previous artifact exactly.",
    )
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--frames-dir", default="rally_frames")
    ap.add_argument("--point-map", default="point_video_map.csv")
    ap.add_argument(
        "--canonical-frames",
        action="store_true",
        help="legacy escape hatch: extract 960x540 instead of the native video resolution",
    )
    ap.add_argument("--artifact-width", type=int, default=960)
    ap.add_argument("--artifact-height", type=int, default=540)
    ap.add_argument("--points-file", default="", help="optional newline-delimited point ids")
    ap.add_argument("--extract-only", action="store_true", help="extract frames without YOLO")
    ap.add_argument("--output", default="player_boxes.csv")
    args = ap.parse_args()
    video = sorted(glob.glob(args.video))[0] if any(c in args.video for c in "*?[") else args.video

    pts = rally_points(args.out, args.point_map)
    point_filter = load_point_filter(args.points_file)
    if point_filter:
        pts = [point for point in pts if point["pt"] in point_filter]
    if args.max_points:
        pts = pts[: args.max_points]
    stage = "frame_extraction" if args.extract_only else "player_detection"
    gpu_devices = [] if args.extract_only else [int(str(args.device).split(":")[-1])]
    stage_run = StageRun(args.out, stage, args, gpu_devices=gpu_devices)
    print(f"{len(pts)} points with rally windows")
    extraction_dispositions: list[dict] = []
    source_metadata = probe(Path(video)) if args.preserve_source_fps else None
    index = extract_clip_frames(
        video,
        os.path.join(args.out, args.frames_dir),
        pts,
        args.fps,
        native=not args.canonical_frames,
        preserve_source_fps=args.preserve_source_fps,
        measured_native_epochs=args.preserve_source_fps,
        video_duration=(source_metadata or {}).get("duration_seconds"),
        dispositions=extraction_dispositions,
    )
    print(f"{len(index)} frames extracted")
    image_size = res.frame_size_for_path(index[0][1]) if index else None
    artifact_size = res.FrameSize(args.artifact_width, args.artifact_height)
    if image_size is None:
        raise SystemExit("no rally frames extracted")
    frames_manifest = os.path.join(args.out, f"{args.frames_dir}.coordinates.json")
    res.write_coordinate_manifest(
        frames_manifest,
        image_size=image_size,
        artifact_size=image_size,
        source=os.path.abspath(video),
        extra={
            "frames_dir": args.frames_dir,
            "fps": args.fps,
            "extraction_dispositions": extraction_dispositions,
        },
        subnative_flagged=image_size != res.NATIVE_SIZE,
        subnative_justification=(
            "explicit legacy --canonical-frames extraction"
            if image_size != res.NATIVE_SIZE
            else None
        ),
    )
    if args.extract_only:
        stage_run.finish(
            outputs={
                "points": len(pts),
                "frames": len(index),
                "image_size": [image_size.width, image_size.height],
                "coordinate_manifest": frames_manifest,
                "extraction_dispositions": extraction_dispositions,
            }
        )
        return 0

    from ultralytics import YOLO

    model = YOLO(args.model)
    rows = []
    for i in range(0, len(index), args.batch):
        chunk = index[i : i + args.batch]
        results = model(
            [fp for _, fp, _ in chunk],
            classes=[0],
            verbose=False,
            device=args.device,
            conf=0.35,
            imgsz=args.imgsz,
        )
        for (pt, fp, t), result in zip(chunk, results):
            b = result.boxes
            for j in range(len(b)):
                rows.append(
                    player_detection_row(
                        point=pt,
                        timestamp=t,
                        frame_path=fp,
                        box=b.xyxy[j].detach().cpu().numpy(),
                        confidence=float(b.conf[j]),
                        image_size=image_size,
                        legacy_size=artifact_size,
                    )
                )
        if i % (args.batch * 20) == 0:
            print(f"  detect {i}/{len(index)}", flush=True)

    out_csv = os.path.join(args.out, args.output)
    write_player_artifact(
        out_csv,
        rows,
        image_size=image_size,
        legacy_size=artifact_size,
        source=os.path.join(args.out, args.frames_dir),
        fps=args.fps,
    )
    # quick sanity: detections per frame distribution
    from collections import Counter

    per_frame = Counter()
    for r in rows:
        per_frame[(r["clip"], r["frame"])] += 1
    dist = Counter(per_frame.values())
    print(f"boxes: {len(rows)} | per-frame count dist: {dict(sorted(dist.items()))}")
    print(f"-> {out_csv}")
    stage_run.finish(
        outputs={
            "points": len(pts),
            "frames": len(index),
            "boxes": len(rows),
            "image_size": [image_size.width, image_size.height],
            "artifact_size": [res.NATIVE_SIZE.width, res.NATIVE_SIZE.height],
            "legacy_artifact_size": [artifact_size.width, artifact_size.height],
            "coordinate_manifest": f"{out_csv}.coordinates.json",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
