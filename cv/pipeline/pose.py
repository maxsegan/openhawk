"""Batched player pose extraction over an existing fixed frame set.

The output retains all COCO body keypoints and explicitly associates the highest-confidence
on-court near and far players.  That makes temporal arm motion available to contact detection
without rerunning inference or relying on a single contact frame.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_manifest import StageRun  # noqa: E402
import resolution as res  # noqa: E402

COURT_W = 10.97
COURT_L = 23.77
NET_Y = COURT_L / 2
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
ARM_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
FOOT_KEYPOINTS = ("left_ankle", "right_ankle")
HIP_KEYPOINTS = ("left_hip", "right_hip")
RACKET_ARM_KEYPOINTS = {
    "left": ("left_elbow", "left_wrist"),
    "right": ("right_elbow", "right_wrist"),
}
# A tennis racket's face centre is roughly one forearm beyond the wrist in the
# image.  This is an observation model, not a 3-D racket normal.  The player
# truth ledger owns its measured error and can demote it without changing the
# pose detector.
RACKET_FACE_FOREARM_SCALE = 1.5
RACKET_FACE_MIN_CONFIDENCE = 0.30
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12))


def load_point_filter(path: str) -> set[int]:
    if not path:
        return set()
    with open(path) as handle:
        return {int(line) for raw in handle if (line := raw.strip()) and not line.startswith("#")}


def frame_index(
    out_dir: str, frames_dir: str, point_filter: set[int], max_frame: int = 0
) -> list[str]:
    frames = sorted(glob.glob(os.path.join(out_dir, frames_dir, "pt*", "f_*.jpg")))
    if point_filter:
        frames = [
            path
            for path in frames
            if int(os.path.basename(os.path.dirname(path))[2:]) in point_filter
        ]
    if max_frame:
        frames = [path for path in frames if int(os.path.basename(path)[2:-4]) <= max_frame]
    return frames


def load_homographies(out_dir: str) -> dict[int, np.ndarray]:
    data = np.load(os.path.join(out_dir, "court_H_per_point.npz"))
    return dict(zip(data["pts"].tolist(), data["H"]))


def court_xy(homography: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = cv2.perspectiveTransform(np.float32([[[x, y]]]), homography)[0, 0]
    return float(point[0]), float(point[1])


def valid_keypoint(row: dict, name: str, min_confidence: float = 0.2) -> bool:
    return (
        float(row.get(f"{name}_confidence", 0.0)) >= min_confidence
        and float(row.get(f"{name}_x", 0.0)) > 0
        and float(row.get(f"{name}_y", 0.0)) > 0
    )


def keypoint_xy(row: dict, name: str, min_confidence: float = 0.2) -> tuple[float, float] | None:
    """Return one declared pose keypoint or ``None`` when it is not usable."""
    if not valid_keypoint(row, name, min_confidence):
        return None
    return float(row[f"{name}_x"]), float(row[f"{name}_y"])


def pose_foot_pixel(
    row: dict,
    *,
    estimator: str = "ankle_mean",
    min_confidence: float = 0.2,
) -> tuple[float, float] | None:
    """A source-image foot/root observation from the visible COCO ankles.

    ``ankle_mean`` represents the midpoint of both feet and is the default
    state observation. ``lower_ankle`` is retained as a measured comparison.
    Neither one asserts that an airborne foot touches the court.
    """
    points = [
        point
        for name in FOOT_KEYPOINTS
        if (point := keypoint_xy(row, name, min_confidence)) is not None
    ]
    if not points:
        return None
    if estimator == "lower_ankle":
        return max(points, key=lambda point: point[1])
    if estimator != "ankle_mean":
        raise ValueError(f"unknown pose foot estimator: {estimator}")
    values = np.asarray(points, dtype=float)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def pose_hip_pixel(row: dict, *, min_confidence: float = 0.2) -> tuple[float, float] | None:
    """Mean of the visible anatomical hip keypoints in source pixels."""
    points = [
        point
        for name in HIP_KEYPOINTS
        if (point := keypoint_xy(row, name, min_confidence)) is not None
    ]
    if not points:
        return None
    values = np.asarray(points, dtype=float)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def racket_face_from_pose(
    row: dict,
    *,
    contact_pixel: tuple[float, float] | None = None,
    hand: str | None = None,
    forearm_scale: float = RACKET_FACE_FOREARM_SCALE,
    min_confidence: float = RACKET_FACE_MIN_CONFIDENCE,
) -> dict | None:
    """Estimate racket-face centre by extending elbow→wrist in the image.

    When the striking hand is not known, the arm whose extrapolated face is
    closest to the automatic contact pixel is selected. With no contact pixel,
    the higher-confidence complete arm wins. The returned point is a 2-D
    observation for a downstream fitter; it is not a face normal.
    """
    requested = [hand] if hand in RACKET_ARM_KEYPOINTS else list(RACKET_ARM_KEYPOINTS)
    candidates = []
    for candidate_hand in requested:
        elbow_name, wrist_name = RACKET_ARM_KEYPOINTS[candidate_hand]
        elbow = keypoint_xy(row, elbow_name, min_confidence)
        wrist = keypoint_xy(row, wrist_name, min_confidence)
        if elbow is None or wrist is None:
            continue
        elbow_array = np.asarray(elbow, dtype=float)
        wrist_array = np.asarray(wrist, dtype=float)
        face = wrist_array + forearm_scale * (wrist_array - elbow_array)
        confidence = min(
            float(row[f"{elbow_name}_confidence"]),
            float(row[f"{wrist_name}_confidence"]),
        )
        contact_distance = (
            None
            if contact_pixel is None
            else float(np.linalg.norm(face - np.asarray(contact_pixel, dtype=float)))
        )
        candidates.append(
            {
                "x": float(face[0]),
                "y": float(face[1]),
                "hand": candidate_hand,
                "elbow": [float(value) for value in elbow],
                "wrist": [float(value) for value in wrist],
                "forearm_scale": float(forearm_scale),
                "keypoint_confidence": confidence,
                "contact_distance_px": contact_distance,
            }
        )
    if not candidates:
        return None
    if contact_pixel is not None:
        return min(candidates, key=lambda item: item["contact_distance_px"])
    return max(candidates, key=lambda item: item["keypoint_confidence"])


def vertical_height_from_pixel(
    projection: np.ndarray,
    court_xy: tuple[float, float],
    pixel: tuple[float, float],
) -> float | None:
    """Solve height on the vertical line through ``court_xy`` for one pixel.

    The two image equations are solved together in least squares. This is a
    metric-camera hip-height proxy, not multi-view anatomical 3-D truth.
    """
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        return None
    x, y = (float(value) for value in court_xy)
    u, v = (float(value) for value in pixel)
    base = np.array([x, y, 0.0, 1.0])
    constant = matrix @ base
    z_column = matrix[:, 2]
    coefficients = np.array(
        [z_column[0] - u * z_column[2], z_column[1] - v * z_column[2]],
        dtype=float,
    )
    targets = -np.array(
        [constant[0] - u * constant[2], constant[1] - v * constant[2]],
        dtype=float,
    )
    denominator = float(coefficients @ coefficients)
    if denominator < 1e-12:
        return None
    height = float(coefficients @ targets / denominator)
    return height if np.isfinite(height) else None


def associate_frame_rows(
    rows: list[dict], homography: np.ndarray, baseline_margin_m: float = 8.0
) -> dict[str, dict]:
    """Select the highest-confidence near/far detection inside the audited court margin."""
    candidates: dict[str, list[dict]] = {"near": [], "far": []}
    for row in rows:
        foot_x = (float(row["x0"]) + float(row["x1"])) / 2
        foot_y = float(row["y1"])
        cx, cy = court_xy(homography, foot_x, foot_y)
        if not (
            -2.5 <= cx <= COURT_W + 2.5 and -baseline_margin_m <= cy <= COURT_L + baseline_margin_m
        ):
            continue
        side = "near" if cy < NET_Y else "far"
        candidates[side].append({**row, "court_x": cx, "court_y": cy, "side": side})
    return {
        side: max(detections, key=lambda row: float(row["conf"]))
        for side, detections in candidates.items()
        if detections
    }


def result_rows(
    frame_path: str, result, artifact_size: res.FrameSize = res.NATIVE_SIZE
) -> list[dict]:
    clip = os.path.basename(os.path.dirname(frame_path))
    frame = os.path.basename(frame_path)
    rows = []
    if result.boxes is None or result.keypoints is None:
        return rows
    for detection_index in range(len(result.boxes)):
        source_size = res.FrameSize(result.orig_shape[1], result.orig_shape[0])
        x0, y0, x1, y1 = res.scale_boxes(
            result.boxes.xyxy[detection_index].detach().cpu().numpy(),
            source_size,
            artifact_size,
        ).tolist()
        keypoints = result.keypoints.data[detection_index].detach().cpu().numpy()
        row = {
            "clip": clip,
            "frame": frame,
            "x0": round(float(x0), 2),
            "y0": round(float(y0), 2),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "conf": round(float(result.boxes.conf[detection_index]), 4),
        }
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            values = keypoints[keypoint_index] if keypoint_index < len(keypoints) else []
            point = (
                res.scale_points(values[:2], source_size, artifact_size)
                if len(values) >= 2
                else (0, 0)
            )
            x, y = float(point[0]), float(point[1])
            confidence = float(values[2]) if len(values) >= 3 else float(x > 0 and y > 0)
            row[f"{name}_x"] = round(x, 2)
            row[f"{name}_y"] = round(y, 2)
            row[f"{name}_confidence"] = round(confidence, 4)
        rows.append(row)
    return rows


def fieldnames() -> list[str]:
    names = ["clip", "frame", "x0", "y0", "x1", "y1", "conf"]
    for keypoint in KEYPOINT_NAMES:
        names.extend([f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_confidence"])
    return names


def associate_rows(
    rows: list[dict],
    homographies: dict[int, np.ndarray],
    *,
    image_size: res.FrameSize = res.NATIVE_SIZE,
    artifact_size: res.FrameSize = res.NATIVE_SIZE,
) -> list[dict]:
    """Near/far association of pose rows written in ``artifact_size`` coordinates.

    ``court_H_per_point.npz`` takes ``image_size`` pixels to court metres, so a row space
    that is not the image space must move the homography, never the court metres.  Reading
    a half-native row with a native homography silently moves a player metres up the court
    and is what put a near-court volley on the far side.
    """
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[(row["clip"], row["frame"])].append(row)
    in_row_space = {
        point: res.image_to_world_homography(homography, image_size, artifact_size)
        for point, homography in homographies.items()
    }
    associated = []
    for (clip, _), detections in by_frame.items():
        point = int(clip[2:])
        if point not in in_row_space:
            continue
        selected = associate_frame_rows(detections, in_row_space[point])
        associated.extend(selected.values())
    return sorted(associated, key=lambda row: (row["clip"], row["frame"], row["side"]))


def load_native_pose_rows(path: str) -> list[dict]:
    """Pose rows in native 1920x1080, whichever space the artifact itself declares.

    The sidecar is the contract: a consumer never infers the space from the file name or
    hardcodes a factor of two.  A pose artifact without a sidecar is refused rather than
    assumed native.
    """
    manifest = res.read_coordinate_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"{path} has no coordinate sidecar; its pixel space is undeclared")
    artifact_size = res.manifest_artifact_size(manifest)
    scale_x = res.NATIVE_SIZE.width / artifact_size.width
    scale_y = res.NATIVE_SIZE.height / artifact_size.height
    columns_x = {"x0", "x1", *(f"{name}_x" for name in KEYPOINT_NAMES)}
    columns_y = {"y0", "y1", *(f"{name}_y" for name in KEYPOINT_NAMES)}
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    scaled = []
    for row in rows:
        record = dict(row)
        for column, factor in ((columns_x, scale_x), (columns_y, scale_y)):
            for name in column & set(record):
                value = record[name]
                if value == "" or value is None:
                    continue
                record[name] = float(value) * factor
        scaled.append(record)
    return scaled


def coverage_summary(associated: list[dict], total_frames: int) -> dict:
    sides_by_frame = defaultdict(set)
    arm_by_frame = defaultdict(set)
    for row in associated:
        key = (row["clip"], row["frame"])
        sides_by_frame[key].add(row["side"])
        if all(valid_keypoint(row, name) for name in ARM_KEYPOINTS):
            arm_by_frame[key].add(row["side"])
    return {
        "total_frames": total_frames,
        "frames_with_near": sum("near" in sides for sides in sides_by_frame.values()),
        "frames_with_far": sum("far" in sides for sides in sides_by_frame.values()),
        "frames_with_both": sum(sides == {"near", "far"} for sides in sides_by_frame.values()),
        "frames_with_near_arm": sum("near" in sides for sides in arm_by_frame.values()),
        "frames_with_far_arm": sum("far" in sides for sides in arm_by_frame.values()),
        "frames_with_both_arms": sum(sides == {"near", "far"} for sides in arm_by_frame.values()),
    }


def draw_pose(image: np.ndarray, row: dict, artifact_size: res.FrameSize = res.NATIVE_SIZE) -> None:
    image_size = res.FrameSize(image.shape[1], image.shape[0])
    color = (30, 210, 30) if row["side"] == "near" else (30, 170, 240)
    x0, y0, x1, y1 = np.round(
        res.scale_boxes(
            [float(row[name]) for name in ("x0", "y0", "x1", "y1")],
            artifact_size,
            image_size,
        )
    ).astype(int)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        f"{row['side']} {float(row['conf']):.2f}",
        (x0, max(15, y0 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    points = {}
    for index, name in enumerate(KEYPOINT_NAMES):
        if valid_keypoint(row, name):
            point = res.scale_points(
                [float(row[f"{name}_x"]), float(row[f"{name}_y"])],
                artifact_size,
                image_size,
            )
            points[index] = tuple(np.round(point).astype(int))
    for start, end in SKELETON:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], color, 2, cv2.LINE_AA)
    for index in (5, 6, 7, 8, 9, 10):
        if index in points:
            cv2.circle(image, points[index], 3, color, -1, cv2.LINE_AA)


def write_audit(
    out_dir: str,
    frames_dir: str,
    associated: list[dict],
    frame_paths: list[str],
    model_tag: str,
    sample_count: int,
    seed: int,
) -> str:
    if sample_count <= 0 or not frame_paths:
        return ""
    rng = random.Random(seed)
    selected_paths = rng.sample(frame_paths, min(sample_count, len(frame_paths)))
    by_frame = defaultdict(list)
    for row in associated:
        by_frame[(row["clip"], row["frame"])].append(row)
    tiles = []
    for path in selected_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        clip, frame = os.path.basename(os.path.dirname(path)), os.path.basename(path)
        for row in by_frame.get((clip, frame), []):
            draw_pose(image, row)
        cv2.putText(
            image,
            f"{clip} {frame}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(image, (480, 270)))
    if not tiles:
        return ""
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    montage = np.vstack([np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)])
    output = os.path.join(out_dir, f"pose_audit_{model_tag}_n{sample_count}_seed{seed}.jpg")
    cv2.imwrite(output, montage)
    return output


def write_pose_manifest(
    artifact_path: str,
    *,
    image_size: res.FrameSize,
    artifact_size: res.FrameSize,
    source: str,
    subnative: bool,
) -> str:
    """Declare the pose artifact's own pixel space, with the native identity when it is native."""
    manifest_path = f"{artifact_path}.coordinates.json"
    res.write_coordinate_manifest(
        manifest_path,
        image_size=image_size,
        artifact_size=artifact_size,
        source=source,
        extra={
            "artifact": os.path.basename(artifact_path),
            "artifact_identity": (
                res.PLAYER_POSE_NATIVE_IDENTITY if not subnative else "player_pose.subnative.v1"
            ),
            "coordinate_columns": {
                "box": ["x0", "y0", "x1", "y1"],
                "keypoints": [f"{name}_x/{name}_y" for name in KEYPOINT_NAMES],
            },
        },
        subnative_flagged=subnative,
        subnative_justification=(
            "explicit --artifact-width/--artifact-height escape hatch; the pose default is native"
            if subnative
            else None
        ),
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir", default="rally_frames_50_contact_v2")
    parser.add_argument("--points-file", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="pose inference size; far-court players are too small at the detector's 640 default",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--max-frame",
        type=int,
        default=0,
        help="only process frames with index <= this per clip (0 = all)",
    )
    parser.add_argument("--io-workers", type=int, default=16)
    parser.add_argument("--audit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--artifact-width",
        type=int,
        default=res.NATIVE_SIZE.width,
        help="pose coordinate space; native is the contract and a smaller space must be "
        "declared in the sidecar and justified",
    )
    parser.add_argument("--artifact-height", type=int, default=res.NATIVE_SIZE.height)
    args = parser.parse_args()
    gpu_device = int(str(args.device).split(":")[-1])
    stage_run = StageRun(
        args.out,
        "player_pose",
        args,
        gpu_devices=[gpu_device],
        seed=args.seed,
    )

    points = load_point_filter(args.points_file)
    requested_root = os.path.join(args.out, args.frames_dir)
    frames_root, image_size = res.select_highest_resolution_frame_dir(
        res.frame_twin_candidates(requested_root)
    )
    resolved_frames_dir = os.path.relpath(frames_root, args.out)
    artifact_size = res.FrameSize(args.artifact_width, args.artifact_height)
    frames = frame_index(args.out, resolved_frames_dir, points, args.max_frame)
    print(
        f"pose model={args.model} frames={len(frames)} points={len(points) or 'all'} "
        f"source={image_size.label} artifact={artifact_size.label}"
    )
    from concurrent.futures import ThreadPoolExecutor

    from ultralytics import YOLO

    model = YOLO(args.model)
    rows = []
    # Sequential cv2.imread in the predict loop is the throughput bottleneck (single core);
    # prefetch the next chunk's images on a thread pool while the GPU works.
    chunks = [frames[i : i + args.batch] for i in range(0, len(frames), args.batch)]
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:

        def load_chunk(chunk):
            return list(pool.map(cv2.imread, chunk))

        pending = pool.submit(load_chunk, chunks[0]) if chunks else None
        for index, chunk in enumerate(chunks):
            images = pending.result()
            pending = (
                pool.submit(load_chunk, chunks[index + 1]) if index + 1 < len(chunks) else None
            )
            valid = [(path, img) for path, img in zip(chunk, images) if img is not None]
            if not valid:
                continue
            results = model(
                [img for _, img in valid],
                classes=[0],
                verbose=False,
                device=args.device,
                conf=args.conf,
                imgsz=args.imgsz,
            )
            for (path, _), result in zip(valid, results):
                rows.extend(result_rows(path, result, artifact_size))
            if index % 20 == 0:
                print(f"  pose {index * args.batch}/{len(frames)}", flush=True)

    output_path = os.path.join(args.out, args.output)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    subnative = artifact_size != res.NATIVE_SIZE
    write_pose_manifest(
        output_path,
        image_size=image_size,
        artifact_size=artifact_size,
        source=frames_root,
        subnative=subnative,
    )

    homographies = load_homographies(args.out)
    associated = associate_rows(
        rows, homographies, image_size=image_size, artifact_size=artifact_size
    )
    associated_path = output_path.removesuffix(".csv") + "_on_court.csv"
    with open(associated_path, "w", newline="") as handle:
        names = fieldnames() + ["court_x", "court_y", "side"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(associated)
    write_pose_manifest(
        associated_path,
        image_size=image_size,
        artifact_size=artifact_size,
        source=frames_root,
        subnative=subnative,
    )
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    audit = write_audit(
        args.out,
        resolved_frames_dir,
        associated,
        frames,
        model_tag,
        args.audit,
        args.seed,
    )
    coverage = coverage_summary(associated, len(frames))
    print(f"detections={len(rows)} associated={len(associated)} coverage={coverage}")
    print(f"pose -> {output_path} | on-court -> {associated_path} | audit -> {audit}")
    stage_run.finish(
        outputs={
            "pose_csv": output_path,
            "on_court_csv": associated_path,
            "audit": audit,
            "detections": len(rows),
            "associated_detections": len(associated),
            "coverage": coverage,
            "image_size": [image_size.width, image_size.height],
            "artifact_size": [artifact_size.width, artifact_size.height],
            "coordinate_manifest": f"{output_path}.coordinates.json",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
