"""3D player pose for every exported attempt window: 2D crop pose, batched temporal 3D lift.

    python -m cv.pipeline.player_pose_3d --match-out OUT/upstream/MATCH --portal OUT/portal/3d \\
        --out OUT/pose --match-id MATCH [--device 0]

The product runner runs this as its ``pose`` step (``TENNIS_PLAYER_POSE_3D``: ``on`` default,
``off`` rollback) after ``portal`` and before ``compact`` deletes the native frame folders.

1. Frames: the union of every portal document's ``frame_range`` per source clip, so a parent
   attempt and its components share one pass. Every native frame in that range is read.
2. 2D: for each frame and side (``near``/``far``), the highest-confidence sided player box
   (``player_boxes_*_native_sided_v1.csv``) is cut as a square native window of ``factor``
   box sizes, resampled to a ``canvas`` px image and given to the COCO pose model
   (yolo26m-pose); the detection overlapping the box most is kept and its 17 keypoints are
   mapped back to native pixels. Pictures are decoded and cut on the GPU (nvJPEG,
   byte-identical to cv2) or with cv2 threads (``--decode cv2``, rollback). These 2D
   keypoints are model observations.
3. 3D: :mod:`cv.pipeline.pose_lift_batched` fits all rows of the match together: joint offsets
   from the (median/mean smoothed) box court root under a local camera at the root whose
   vertical scale is calibrated per player track from stature (the per-frame projection's
   own vertical scale is off by 10-30%), with limb-length, ground, upright, forward and
   second-difference smoothness terms. A track whose raw vertical-scale estimate is outside
   0.5-2.0 is withheld (camera implausible), as are both ends of a >2 m one-frame root jump.
   Every 3D joint is FITTED, not observed.
4. Status: a joint without a usable keypoint is ``interpolated`` only between two of its own
   keypoints at most ``max_gap`` native frames apart in the same run; otherwise ``missing``.
   Nothing is extrapolated past the first/last observed keypoint.

Outputs (``--out``): ``player_pose_3d_v1.npz`` (every frame/side row: native 2D keypoints and
confidences, metric court-frame 3D joints, per-joint status, root, camera reliability, vertical
scale, active-play flag; packed int16, read with :func:`load_pose`) and
``player_pose_3d_v1.json`` (receipt: inputs with sha256, model, settings, counts, timing).
``--fill-documents`` (default) writes the portal pose layers (``player_pose_portal``).

Per-joint status codes (``STATUS``): 0 missing, 1 lifted from a confident 2D keypoint, 2 lifted
from a low-confidence keypoint, 3 interpolated across a short gap. No evaluation label or
reviewed choice is read; the inputs are automatic upstream outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import pose_lift, pose_lift_batched, provenance
from cv.pipeline.pose import KEYPOINT_NAMES
from cv.pipeline.pose_crop_infer import select_detection

SCHEMA = "player_pose_3d_v1"
ENV = "TENNIS_PLAYER_POSE_3D"
SIDES = ("near", "far")
NPZ_NAME = f"{SCHEMA}.npz"
RECEIPT_NAME = f"{SCHEMA}.json"
STATUS = {"missing": 0, "lifted": 1, "lifted_low_confidence": 2, "interpolated": 3}
#: Keypoints below this confidence do not enter the lift (pose_lift's own floor).
LIFT_MIN_CONFIDENCE = pose_lift.MIN_KEYPOINT_CONFIDENCE
#: Lifted joints below this 2D confidence are marked low-confidence.
CONFIDENT = 0.5
#: Tour-average statures (m) when the player is not identified.
DEFAULT_STATURE_M = {"atp": 1.86, "wta": 1.74}
FALLBACK_STATURE_M = 1.80
HEAD_EXTRAS = ("left_eye", "right_eye", "left_ear", "right_ear")


#: Stored integer units: 2D pixels and reprojection in 0.1 px, 3D joints in mm (int16, NaN as
#: ``MISSING``). Read with :func:`load_pose`.
KEYPOINT_SCALE = 0.1
JOINT_SCALE = 0.001
MISSING = np.iinfo(np.int16).min
PACKED = ("keypoints_2d", "joints_3d", "reprojection_px")


def pack(values: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.round(np.asarray(values, dtype=float) / scale)
    return np.where(
        np.isfinite(scaled), np.clip(scaled, MISSING + 1, -MISSING - 1), MISSING
    ).astype(np.int16)


def load_pose(path: Path) -> dict:
    """``player_pose_3d_v1.npz`` with packed fields back in float metres/pixels (NaN missing)."""
    data = np.load(path, allow_pickle=False)
    pose = {key: data[key] for key in data.files}
    scales = {
        "keypoints_2d": KEYPOINT_SCALE,
        "joints_3d": JOINT_SCALE,
        "reprojection_px": KEYPOINT_SCALE,
    }
    for key in PACKED:
        if key in pose and pose[key].dtype == np.int16:
            values = pose[key].astype(np.float64) * scales[key]
            values[pose[key] == MISSING] = np.nan
            pose[key] = values
    return pose


def enabled() -> bool:
    value = os.environ.get(ENV, "on") or "on"
    if value not in ("on", "off"):
        raise SystemExit(f"{ENV}={value!r}: expected on or off")
    return value == "on"


def stature_for(match_id: str) -> float:
    return DEFAULT_STATURE_M.get(match_id.split("_", 1)[0].lower(), FALLBACK_STATURE_M)


# --------------------------------------------------------------------------- inputs


def portal_documents(portal: Path) -> list[Path]:
    """Every staged attempt document (parents and components) with a native frame range."""
    paths = []
    for path in sorted((portal / "data").glob("*.json")):
        if path.name in ("index.json", "best_effort.json"):
            continue
        paths.append(path)
    return paths


def document_windows(documents: list[Path]) -> dict[str, tuple[int, int]]:
    """Union frame range per source clip over every document."""
    windows: dict[str, tuple[int, int]] = {}
    for path in documents:
        document = json.loads(path.read_text())
        clip, frames = document.get("source_clip"), document.get("frame_range")
        if not clip or not frames:
            continue
        lo, hi = int(frames[0]), int(frames[1])
        if clip in windows:
            lo, hi = min(lo, windows[clip][0]), max(hi, windows[clip][1])
        windows[clip] = (lo, hi)
    return dict(sorted(windows.items()))


def frame_number(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[-1])


def load_boxes(path: Path, windows: dict[str, tuple[int, int]]) -> dict[tuple[str, int, str], dict]:
    """Highest-confidence sided box per (clip, frame, side) inside the windows."""
    best: dict[tuple[str, int, str], dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            clip, side = row["clip"], row.get("side")
            if clip not in windows or side not in SIDES:
                continue
            frame = frame_number(row["frame"])
            lo, hi = windows[clip]
            if not lo <= frame <= hi:
                continue
            try:
                box = [float(row[k]) for k in ("x0_native", "y0_native", "x1_native", "y1_native")]
                conf = float(row.get("conf") or 0.0)
            except (KeyError, ValueError):
                continue
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            key = (clip, frame, side)
            if key in best and best[key]["conf"] >= conf:
                continue
            try:
                court = (float(row["court_x"]), float(row["court_y"]))
            except (KeyError, ValueError):
                court = (math.nan, math.nan)
            best[key] = {
                "box": box,
                "conf": conf,
                "court": court,
                "track_id": row.get("track_id", ""),
                "frame_name": row["frame"],
            }
    return best


def active_rows(path: Path, match_id: str, keys: list) -> np.ndarray:
    """Whether each (clip, frame, side) row lies inside its point's decided active-play spans.

    Rows outside them (dead-ball context inside the exported window) are kept and flagged:
    their sided boxes are more often someone else (umpire, ball kid). All False without the file.
    """
    flags = np.zeros(len(keys), dtype=bool)
    if not path.is_file():
        return flags
    decisions = json.loads(path.read_text())
    for i, (clip, frame, _side) in enumerate(keys):
        spans = (decisions.get(f"{match_id}/{clip}") or {}).get("active_spans") or []
        flags[i] = any(start <= frame <= end for start, end in spans)
    return flags


def load_cameras(path: Path, clips: set[str]) -> dict[tuple[str, int], tuple[np.ndarray, bool]]:
    data = np.load(path, allow_pickle=False)
    cameras = {}
    for clip, frame, projection, reliable in zip(
        data["clips"], data["frames"], data["P"], data["reliable"], strict=True
    ):
        clip = str(clip)
        if clip in clips and np.isfinite(projection).all():
            cameras[(clip, int(frame))] = (projection.astype(float), bool(reliable))
    return cameras


# --------------------------------------------------------------------------- 2D pose


def crop_transform(box, canvas: int, factor: float) -> tuple[float, float, float]:
    """Scale and offset mapping native pixels to the square crop canvas."""
    x0, y0, x1, y1 = box
    side = max(y1 - y0, x1 - x0) * factor
    scale = canvas / side
    return scale, canvas / 2 - scale * (x0 + x1) / 2, canvas / 2 - scale * (y0 + y1) / 2


def cut(image: np.ndarray, transform, canvas: int) -> np.ndarray:
    scale, tx, ty = transform
    matrix = np.float32([[scale, 0, tx], [0, scale, ty]])
    return cv2.warpAffine(
        image, matrix, (canvas, canvas), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)
    )


def gpu_crops(frames, owners_frame: list[int], transforms: list, canvas: int):
    """Bilinear crops of CUDA RGB frames (F,3,H,W uint8): the same map as :func:`cut`."""
    import torch
    import torch.nn.functional as F

    height, width = frames.shape[2], frames.shape[3]
    out = []
    grid_u = torch.arange(canvas, device=frames.device, dtype=torch.float32)
    for start in range(0, len(owners_frame), 128):
        index = torch.as_tensor(owners_frame[start : start + 128], device=frames.device)
        params = torch.as_tensor(
            transforms[start : start + 128], device=frames.device, dtype=torch.float32
        )
        scale, tx, ty = params[:, 0:1], params[:, 1:2], params[:, 2:3]
        x = (grid_u[None, :] - tx) / scale  # (B, canvas) source column per output column
        y = (grid_u[None, :] - ty) / scale
        gx = (2 * x + 1) / width - 1
        gy = (2 * y + 1) / height - 1
        grid = torch.stack(
            [gx[:, None, :].expand(-1, canvas, -1), gy[:, :, None].expand(-1, -1, canvas)], dim=-1
        )
        source = frames[index].float() - 114.0
        sampled = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        out.append(((sampled + 114.0).clamp(0, 255).round() / 255.0))
    return torch.cat(out) if out else None


def infer_2d(
    model,
    items: list[tuple[Path, list[float]]],
    *,
    canvas: int,
    factor: float,
    batch: int,
    device: str,
    workers: int,
    decoder=None,
    chunk_frames: int = 64,
    half: bool = False,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pose every (frame picture, native box): keypoints (N,17,2), confidences (N,17), det conf (N).

    With ``decoder`` (``gpu_jpeg.NvJpegDecoder``) pictures are decoded and cropped on the GPU;
    otherwise ``cv2.imread`` in a thread pool and ``cv2.warpAffine`` (rollback). Either way a
    chunk is prepared while the previous one runs. An unreadable picture or a crop without an
    overlapping person yields NaN keypoints.
    """
    n = len(items)
    keypoints = np.full((n, 17, 2), np.nan, dtype=np.float32)
    confidences = np.zeros((n, 17), dtype=np.float32)
    detections = np.zeros(n, dtype=np.float32)
    counts = {"unreadable": 0, "no_person": 0, "posed": 0}
    order = sorted(range(n), key=lambda i: str(items[i][0]))
    chunks: list[list[int]] = []
    current: list[int] = []
    seen: set[Path] = set()
    for index in order:
        path = items[index][0]
        if path not in seen and len(seen) >= chunk_frames:
            chunks.append(current)
            current, seen = [], set()
        seen.add(path)
        current.append(index)
    if current:
        chunks.append(current)

    pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def focus_box(i, transform):
        scale, tx, ty = transform
        x0, y0, x1, y1 = items[i][1]
        return np.clip(
            [x0 * scale + tx, y0 * scale + ty, x1 * scale + tx, y1 * scale + ty], 0, canvas
        )

    def prepare(chunk: list[int]):
        paths = list(dict.fromkeys(items[i][0] for i in chunk))
        owners, transforms, focus = [], [], []
        if decoder is not None:
            present = [path for path in paths if path.is_file()]
            position = {path: k for k, path in enumerate(present)}
            frames = (
                decoder.decode_paths_rgb_chw([str(path) for path in present]) if present else None
            )
            frame_index = []
            for i in chunk:
                if items[i][0] not in position:
                    counts["unreadable"] += 1
                    continue
                transform = crop_transform(items[i][1], canvas, factor)
                owners.append(i)
                transforms.append(transform)
                frame_index.append(position[items[i][0]])
                focus.append(focus_box(i, transform))
            crops = gpu_crops(frames, frame_index, transforms, canvas) if owners else None
            return crops, owners, transforms, focus
        images = dict(zip(paths, pool.map(lambda p: cv2.imread(str(p)), paths), strict=True))
        crops = []
        for i in chunk:
            image = images[items[i][0]]
            if image is None:
                counts["unreadable"] += 1
                continue
            transform = crop_transform(items[i][1], canvas, factor)
            crops.append(cut(image, transform, canvas))
            owners.append(i)
            transforms.append(transform)
            focus.append(focus_box(i, transform))
        return crops, owners, transforms, focus

    prepared: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        try:
            for chunk in chunks:
                prepared.put(prepare(chunk))
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            prepared.put(error)
            return
        prepared.put(None)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    done = 0
    pending: list = []

    def flush():
        crops = [entry[0] for entry in pending]
        if decoder is not None:
            import torch

            crops = torch.cat(crops)
        options = {"imgsz": canvas, "conf": 0.05, "device": device, "verbose": False}
        if half:
            options["half"] = True
        results = model.predict(crops, **options)
        for (_crop, i, transform, box), result in zip(
            [(None, *entry[1:]) for entry in pending], results, strict=True
        ):
            selected = (
                select_detection(result, canvas, box)
                if box[2] > box[0] and box[3] > box[1]
                else None
            )
            if selected is None:
                counts["no_person"] += 1
                continue
            scale, tx, ty = transform
            kpts = selected["kpts"]
            keypoints[i, :, 0] = (kpts[:, 0] - tx) / scale
            keypoints[i, :, 1] = (kpts[:, 1] - ty) / scale
            confidences[i] = kpts[:, 2]
            detections[i] = selected["conf"]
            counts["posed"] += 1
        pending.clear()

    while True:
        payload = prepared.get()
        if payload is None:
            break
        if isinstance(payload, BaseException):
            raise payload
        crops, owners, transforms, focus = payload
        for k, i in enumerate(owners):
            crop = crops[k : k + 1] if decoder is not None else crops[k]
            pending.append((crop, i, transforms[k], focus[k]))
            if len(pending) >= batch:
                flush()
        done += len(owners)
        if progress:
            progress(done, n)
    if pending:
        flush()
    thread.join()
    pool.shutdown()
    return keypoints, confidences, detections, counts


# --------------------------------------------------------------------------- 3D lift


def smooth_roots(roots: np.ndarray, frames: np.ndarray, window: int) -> np.ndarray:
    """Median-then-mean smoothing of court roots along contiguous native-frame runs."""
    output = roots.copy()
    if window <= 1 or len(roots) < 3:
        return output
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    for run in np.split(np.arange(len(frames)), breaks):
        if len(run) < 3:
            continue
        values = roots[run]
        half = window // 2
        padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
        median = np.stack([np.median(padded[k : k + window], axis=0) for k in range(len(run))])
        padded = np.pad(median, ((half, half), (0, 0)), mode="edge")
        output[run] = np.stack([padded[k : k + window].mean(axis=0) for k in range(len(run))])
    return output


def joint_status(
    confidences: np.ndarray,
    liftable: np.ndarray,
    track: np.ndarray,
    frames: np.ndarray,
    max_gap: int,
) -> np.ndarray:
    """Per-joint status: lifted from a (low-)confident keypoint, interpolated, or missing.

    A joint without a usable keypoint in its own frame is ``interpolated`` only when the same
    joint has one within ``max_gap`` native frames on both sides in the same lifted run.
    """
    seen = (confidences >= LIFT_MIN_CONFIDENCE) & liftable[:, None]
    status = np.where(
        seen,
        np.where(confidences >= CONFIDENT, STATUS["lifted"], STATUS["lifted_low_confidence"]),
        STATUS["missing"],
    ).astype(np.uint8)
    run_break = np.ones(len(frames), dtype=bool)
    run_break[1:] = ~(
        (track[1:] == track[:-1]) & (frames[1:] == frames[:-1] + 1) & liftable[1:] & liftable[:-1]
    )
    run_id = np.cumsum(run_break)
    for run in np.unique(run_id[liftable]):
        rows = np.flatnonzero(run_id == run)
        for k in range(seen.shape[1]):
            hits = np.flatnonzero(seen[rows, k])
            for a, b in zip(hits[:-1], hits[1:], strict=True):
                if 1 < b - a <= max_gap + 1:
                    status[rows[a + 1 : b], k] = STATUS["interpolated"]
    return status


def document_fps(documents: list[Path]) -> float:
    for path in documents:
        fps = json.loads(path.read_text()).get("fps")
        if isinstance(fps, (int, float)) and fps > 0:
            return float(fps)
    return 25.0


# --------------------------------------------------------------------------- run


def run(args) -> dict:
    started = time.time()
    match_out = args.match_out
    portal = args.portal
    documents = portal_documents(portal)
    windows = document_windows(documents)
    boxes_path = next(iter(sorted(match_out.glob("player_boxes_*_native_sided_v1.csv"))), None)
    camera_path = match_out / "camera_P_per_frame_v1.npz"
    frames_root = match_out / args.frames_dir
    if boxes_path is None or not camera_path.is_file():
        raise FileNotFoundError("player_pose_3d needs sided player boxes and per-frame cameras")
    boxes = load_boxes(boxes_path, windows)
    cameras = load_cameras(camera_path, set(windows))
    keys = sorted(boxes, key=lambda key: (key[0], SIDES.index(key[2]), key[1]))
    items = [(frames_root / key[0] / boxes[key]["frame_name"], boxes[key]["box"]) for key in keys]
    timing = {"inputs_seconds": round(time.time() - started, 2)}

    t0 = time.time()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    decoder = None
    if args.decode == "nvjpeg":
        from cv.experiments.gpu_decode import gpu_jpeg

        decoder = gpu_jpeg.open_decoder(int(str(args.device).split(":")[-1]))
    keypoints, confidences, detections, counts_2d = infer_2d(
        model,
        items,
        canvas=args.canvas,
        factor=args.factor,
        batch=args.batch,
        device=args.device,
        workers=args.decode_workers,
        decoder=decoder,
        half=args.half,
        progress=(lambda done, total: print(f"pose 2d {done}/{total}", flush=True))
        if args.verbose
        else None,
    )
    timing["pose_2d_seconds"] = round(time.time() - t0, 2)

    t0 = time.time()
    stature = args.stature or stature_for(args.match_id)
    n = len(keys)
    roots = np.full((n, 2), np.nan)
    projections = np.full((n, 3, 4), np.nan)
    camera_reliable = np.zeros(n, dtype=bool)
    withheld = np.zeros(n, dtype=bool)
    track = np.zeros(n, dtype=np.int64)
    frames = np.asarray([key[1] for key in keys], dtype=np.int64)
    forward = np.asarray([1.0 if key[2] == "near" else -1.0 for key in keys])
    groups: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault((key[0], key[2]), []).append(i)
    next_track = 0
    for (clip, _side), rows in groups.items():
        rows = np.asarray(rows)
        raw = np.asarray([boxes[keys[i]]["court"] for i in rows], dtype=float)
        # An over-limit one-frame root jump is an identity/association break: both ends are
        # withheld and the track restarts there.
        step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        jump = (np.diff(frames[rows]) == 1) & ~(step <= pose_lift.MAXIMUM_INPUT_ROOT_JUMP_M)
        withheld[rows[1:][jump]] = True
        withheld[rows[:-1][jump]] = True
        segment = np.concatenate([[0], np.cumsum(jump)])
        track[rows] = next_track + segment
        next_track += int(segment[-1]) + 1
        for value in np.unique(segment):
            part = rows[segment == value]
            roots[part] = smooth_roots(raw[segment == value], frames[part], args.root_window)
        for i in rows:
            camera = cameras.get((clip, keys[i][1]))
            if camera is not None:
                projections[i], camera_reliable[i] = camera
    active_path = args.active_play or match_out.parent / "active_play_v1.json"
    active = active_rows(Path(active_path), args.match_id, keys)
    has_camera = np.isfinite(projections).all(axis=(1, 2))
    liftable = has_camera & np.isfinite(roots).all(axis=1) & ~withheld
    rows = np.flatnonzero(liftable)
    lifted = pose_lift_batched.lift(
        keypoints[rows],
        confidences[rows],
        np.asarray([boxes[keys[i]]["box"] for i in rows]).reshape(-1, 4),
        roots[rows],
        projections[rows],
        track[rows],
        frames[rows],
        forward[rows],
        stature,
        fps=args.fps or document_fps(documents),
        iterations=args.iterations,
    )
    joints = np.full((n, 17, 3), np.nan)
    joints[rows] = lifted["joints"]
    reprojection = np.full((n, 17), np.nan)
    reprojection[rows] = lifted["reprojection_px"]
    vertical_scale = np.full(n, np.nan)
    vertical_scale[rows] = lifted["vertical_scale"]
    implausible = np.zeros(n, dtype=bool)
    implausible[rows] = ~lifted["scale_plausible"]
    liftable &= ~implausible
    status = joint_status(confidences, liftable, track, frames, args.max_gap)
    joints[status == 0] = np.nan
    reproj = np.nanmedian(np.where(confidences >= CONFIDENT, reprojection, np.nan), axis=1)
    lift_counts = {
        "no_camera": int((~has_camera).sum()),
        "no_root": int((has_camera & ~np.isfinite(roots).all(axis=1)).sum()),
        "root_jump_withheld": int((has_camera & withheld).sum()),
        "lifted_rows": int(len(rows) - implausible.sum()),
        "camera_vertical_scale_withheld": int(implausible.sum()),
        "device": lifted.get("device"),
        "loss_history": lifted.get("loss_history"),
        "raw_vertical_scale_by_track": lifted.get("raw_scales"),
        "vertical_scale_by_track": {
            f"{keys[np.flatnonzero(track == t)[0]][0]}:{keys[np.flatnonzero(track == t)[0]][2]}:{t}": round(
                v, 3
            )
            for t, v in sorted(lifted.get("scales", {}).items())
        },
    }
    timing["lift_seconds"] = round(time.time() - t0, 2)

    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / NPZ_NAME
    temporary = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray(SCHEMA),
        joint_names=np.asarray(KEYPOINT_NAMES),
        clip=np.asarray([key[0] for key in keys]),
        frame=np.asarray([key[1] for key in keys], dtype=np.int32),
        side=np.asarray([SIDES.index(key[2]) for key in keys], dtype=np.uint8),
        track_id=np.asarray([str(boxes[key]["track_id"]) for key in keys]),
        box_native=np.asarray([boxes[key]["box"] for key in keys], dtype=np.float32),
        box_conf=np.asarray([boxes[key]["conf"] for key in keys], dtype=np.float32),
        keypoints_2d=pack(keypoints, KEYPOINT_SCALE),
        keypoint_conf=np.round(confidences * 255).astype(np.uint8),
        pose_det_conf=np.round(detections * 255).astype(np.uint8),
        joints_3d=pack(joints, JOINT_SCALE),
        joint_status=status,
        root_xy=np.round(roots, 3).astype(np.float32),
        camera_reliable=camera_reliable,
        active_play=active,
        vertical_scale=np.round(vertical_scale, 3).astype(np.float32),
        reprojection_px=pack(reprojection, KEYPOINT_SCALE),
        scales=np.asarray([KEYPOINT_SCALE, JOINT_SCALE]),
    )
    temporary.replace(npz_path)
    present = status > 0
    rows_with_pose = int(present.any(axis=1).sum())
    observed = int(
        ((status == STATUS["lifted"]) | (status == STATUS["lifted_low_confidence"])).sum()
    )
    receipt = {
        "schema": SCHEMA,
        "match_id": args.match_id,
        "observation_origin": "automatic",
        "evaluation_labels_read": False,
        "fitted_not_observed": "joints_3d are fitted on camera rays; keypoints_2d are model observations",
        "status_codes": STATUS,
        "inputs": {
            "boxes": provenance.file_record(boxes_path),
            "cameras": provenance.file_record(camera_path),
            "frames": str(frames_root),
            "documents": len(documents),
            **(
                {"active_play": provenance.file_record(Path(active_path))}
                if Path(active_path).is_file()
                else {}
            ),
        },
        "model": provenance.file_record(Path(args.model)),
        "settings": {
            "canvas": args.canvas,
            "factor": args.factor,
            "half": args.half,
            "decode": "nvjpeg" if decoder is not None else "cv2",
            "stature_m": stature,
            "root_window": args.root_window,
            "max_gap": args.max_gap,
            "lift": "pose_lift_batched local-camera stature-calibrated temporal fit",
            "iterations": args.iterations,
        },
        "windows": {clip: list(window) for clip, window in windows.items()},
        "counts": {
            "rows": n,
            "frames": len({(key[0], key[1]) for key in keys}),
            "pose_2d": counts_2d,
            "lift": lift_counts,
            "rows_with_3d_pose": rows_with_pose,
            "rows_in_active_play": int(active.sum()),
            "active_play_rows_with_3d_pose": int((present.any(axis=1) & active).sum()),
            "joints_lifted": observed,
            "joints_interpolated": int((status == STATUS["interpolated"]).sum()),
            "joints_missing": int((status == 0).sum()),
            "median_reprojection_px": float(np.nanmedian(reproj))
            if np.isfinite(reproj).any()
            else None,
        },
        "output": provenance.file_record(npz_path),
        "timing": {**timing, "total_seconds": round(time.time() - started, 2)},
        "code": provenance.git_record(Path(__file__).resolve().parents[2]),
    }
    (args.out / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
    if args.fill_documents:
        from cv.pipeline import player_pose_portal

        receipt["documents"] = player_pose_portal.fill(portal, npz_path, receipt)
        (args.out / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--match-out", type=Path, required=True, help="upstream match directory")
    p.add_argument("--portal", type=Path, required=True, help="staged portal (OUT/portal/3d)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--match-id", required=True)
    p.add_argument("--frames-dir", default="audit_frames_native_1080")
    p.add_argument(
        "--active-play", type=Path, default=None, help="default: MATCH_OUT/../active_play_v1.json"
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
        / "models/pipeline/yolo26m-pose.pt",
    )
    p.add_argument("--device", default="0")
    p.add_argument("--canvas", type=int, default=320)
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--half", action="store_true")
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument(
        "--decode",
        choices=("nvjpeg", "cv2"),
        default=os.environ.get("TENNIS_PLAYER_POSE_DECODE", "nvjpeg"),
        help="picture decode and crop: on the GPU (nvjpeg, default) or cv2 threads (rollback)",
    )
    p.add_argument("--stature", type=float, default=None)
    p.add_argument("--root-window", type=int, default=5)
    p.add_argument("--max-gap", type=int, default=8)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--fps", type=float, default=None, help="default: the documents' native fps")
    p.add_argument("--fill-documents", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    from cv.pipeline import math_threads

    math_threads.apply()
    args = parser().parse_args()
    receipt = run(args)
    print(json.dumps({"counts": receipt["counts"], "timing": receipt["timing"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
