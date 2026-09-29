"""Segment broadcast camera shots and identify the dominant play-camera view.

Abrupt image differences nominate shot boundaries. Each resulting shot is then registered
to the dominant shot by ORB features and a RANSAC homography. Same-camera tennis footage
retains many inliers despite pan and zoom; closeups, crowd shots, and replays do not.

This is a camera-view gate, not a rally-phase classifier. Between-point footage from the
normal court camera remains in scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

THUMBNAIL_SIZE = (320, 180)
DEFAULT_CUT_THRESHOLD = 8.0
DEFAULT_MINIMUM_SHOT_SECONDS = 0.48
DEFAULT_MINIMUM_INLIERS = 60.0
# Mean absolute thumbnail difference. A hard cut on these broadcasts is about
# 50. A static wide shot has a tiny deviation, so the z-score alone cuts on a
# few pixels of noise. Propagation uses this floor. The play-camera gate does not.
PROPAGATION_CUT_DIFFERENCE = 30.0

_DETECTOR = cv2.ORB_create(1200)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def frame_number(path: Path | str) -> int:
    match = re.search(r"\d+", Path(path).stem)
    if match is None:
        raise ValueError(f"frame has no numeric index: {path}")
    return int(match.group())


@dataclass
class ShotSegmentation:
    sampled_frames: np.ndarray
    shot_id: np.ndarray
    is_play_camera: np.ndarray
    registration_inliers: np.ndarray
    shots: list[dict] = field(default_factory=list)
    boundary_continuity: list[dict] = field(default_factory=list)

    def play_rate(self) -> float:
        return float(np.mean(self.is_play_camera)) if len(self.is_play_camera) else 0.0

    def expand(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.sampled_frames) == 0:
            count = len(frames)
            return (
                np.full(count, -1, dtype=int),
                np.zeros(count, dtype=bool),
                np.zeros(count, dtype=float),
            )
        indices = np.searchsorted(self.sampled_frames, frames)
        indices = np.clip(indices, 0, len(self.sampled_frames) - 1)
        prior = np.maximum(indices - 1, 0)
        choose_prior = np.abs(frames - self.sampled_frames[prior]) <= np.abs(
            frames - self.sampled_frames[indices]
        )
        nearest = np.where(choose_prior, prior, indices)
        return (
            self.shot_id[nearest],
            self.is_play_camera[nearest],
            self.registration_inliers[nearest],
        )


def _features(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    return _DETECTOR.detectAndCompute(image, None)


def _registration_inliers(reference, path: Path) -> float:
    if reference is None or reference[1] is None:
        return 0.0
    probe = _features(path)
    if probe is None or probe[1] is None or len(probe[0]) < 10:
        return 0.0
    matches = _MATCHER.match(reference[1], probe[1])
    if len(matches) < 10:
        return 0.0
    source = np.float32([reference[0][match.queryIdx].pt for match in matches])
    target = np.float32([probe[0][match.trainIdx].pt for match in matches])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    return 0.0 if mask is None else float(mask.sum())


def merge_short_runs(shot_id: np.ndarray, minimum: int) -> np.ndarray:
    output = shot_id.copy()
    start = 0
    for index in range(1, len(output) + 1):
        if index == len(output) or output[index] != output[start]:
            if index - start < minimum and start > 0:
                output[start:index] = output[start - 1]
            start = index
    _, renumbered = np.unique(output, return_inverse=True)
    return renumbered


def segment_shots(
    frame_paths: list[Path],
    fps: float,
    stride: int = 1,
    minimum_inliers: float = DEFAULT_MINIMUM_INLIERS,
    cut_threshold: float = DEFAULT_CUT_THRESHOLD,
    minimum_shot_seconds: float = DEFAULT_MINIMUM_SHOT_SECONDS,
    native_cut_continuity: bool = False,
    minimum_cut_difference: float = 0.0,
) -> ShotSegmentation:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    sampled_paths = frame_paths[::stride]
    sampled_frames = np.asarray([frame_number(path) for path in sampled_paths], dtype=int)
    thumbnails = []
    readable_paths = []
    readable_frames = []
    for path, frame in zip(sampled_paths, sampled_frames, strict=True):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        thumbnails.append(cv2.resize(image, THUMBNAIL_SIZE).astype(np.float32))
        readable_paths.append(path)
        readable_frames.append(frame)
    sampled_frames = np.asarray(readable_frames, dtype=int)
    if len(thumbnails) < 2:
        count = len(thumbnails)
        return ShotSegmentation(
            sampled_frames,
            np.zeros(count, dtype=int),
            np.zeros(count, dtype=bool),
            np.zeros(count, dtype=float),
        )

    difference = np.asarray(
        [
            float(np.abs(thumbnails[index + 1] - thumbnails[index]).mean())
            for index in range(len(thumbnails) - 1)
        ]
    )
    median = float(np.median(difference))
    deviation = 1.4826 * float(np.median(np.abs(difference - median))) + 1e-6
    # Same rule as the z-score cut, plus an absolute floor when the caller sets one.
    threshold = max(float(minimum_cut_difference), median + cut_threshold * deviation)
    boundaries = set((np.flatnonzero(difference > threshold) + 1).tolist())
    shot_id = np.zeros(len(thumbnails), dtype=int)
    current_shot = 0
    for index in range(len(thumbnails)):
        if index in boundaries:
            current_shot += 1
        shot_id[index] = current_shot
    minimum_samples = max(2, round(minimum_shot_seconds * fps / stride))
    shot_id = merge_short_runs(shot_id, minimum_samples)

    dominant_shot = int(np.bincount(shot_id).argmax())
    dominant_members = np.flatnonzero(shot_id == dominant_shot)
    reference_index = int(dominant_members[len(dominant_members) // 2])
    reference = _features(readable_paths[reference_index])

    registration = np.zeros(len(thumbnails), dtype=float)
    shots = []
    for shot in range(int(shot_id.max()) + 1):
        members = np.flatnonzero(shot_id == shot)
        probe = members[:: max(1, len(members) // 3)][:3]
        scores = [_registration_inliers(reference, readable_paths[index]) for index in probe]
        score = float(np.median(scores)) if scores else 0.0
        registration[members] = score
        shots.append(
            {
                "shot_id": shot,
                "start_frame": int(sampled_frames[members[0]]),
                "end_frame": int(sampled_frames[members[-1]]),
                "sampled_frames": int(len(members)),
                "registration_inliers": score,
                "is_play_camera": bool(score >= minimum_inliers),
            }
        )
    continuity = []
    if native_cut_continuity:
        from cv.pipeline.native_shot_continuity import qualify_native_continuity

        existing_boundaries = set((np.flatnonzero(np.diff(shot_id)) + 1).tolist())
        all_frames = np.asarray([frame_number(path) for path in frame_paths], dtype=int)
        for index in sorted(existing_boundaries):
            before = int(sampled_frames[index - 1])
            after = int(sampled_frames[index])
            boundary_registration = _registration_inliers(
                _features(readable_paths[index - 1]), readable_paths[index]
            )
            receipt = qualify_native_continuity(
                frame_paths, all_frames, before, after, registration_inliers=boundary_registration
            )
            receipt["original_camera_classes"] = [
                bool(registration[index - 1] >= minimum_inliers),
                bool(registration[index] >= minimum_inliers),
            ]
            if receipt["continuous"] and len(set(receipt["original_camera_classes"])) > 1:
                receipt.update(
                    continuous=False,
                    local_motion_continuous=True,
                    reason="original_camera_class_disagreement",
                )
            continuity.append(receipt)
        existing_boundaries.difference_update(
            index
            for index, receipt in zip(sorted(existing_boundaries), continuity, strict=True)
            if receipt["continuous"]
        )
        # Refine the established segmentation monotonically. Qualifying raw peaks before
        # short-run merging can expose previously absorbed fragments and create new cuts.
        shot_id = np.cumsum(np.isin(np.arange(len(shot_id)), list(existing_boundaries)))

        # Preserve original per-sample camera eligibility. A local continuity witness
        # must not replace the dominant reference and upgrade held close-up pictures.
        shots = []
        for shot in range(int(shot_id.max()) + 1):
            members = np.flatnonzero(shot_id == shot)
            classes = registration[members] >= minimum_inliers
            assert np.all(classes == classes[0])
            shots.append(
                {
                    "shot_id": shot,
                    "start_frame": int(sampled_frames[members[0]]),
                    "end_frame": int(sampled_frames[members[-1]]),
                    "sampled_frames": int(len(members)),
                    "registration_inliers": float(np.median(registration[members])),
                    "is_play_camera": bool(classes[0]),
                }
            )

    return ShotSegmentation(
        sampled_frames,
        shot_id,
        registration >= minimum_inliers,
        registration,
        shots,
        continuity,
    )


def propagation_shot_ids(
    frame_paths: list[Path],
    fps: float,
    stride: int = 2,
) -> dict[int, int]:
    """Frame number to shot id for homography propagation.

    Uses the same thumbnail cut as ``segment_shots``, with an absolute floor so
    a static wide shot stays one shot. A close-up is a different shot. Frames
    that cannot be placed are omitted.
    """
    if len(frame_paths) < 2:
        return {frame_number(path): 0 for path in frame_paths}
    segmentation = segment_shots(
        frame_paths,
        fps,
        stride=stride,
        minimum_cut_difference=PROPAGATION_CUT_DIFFERENCE,
    )
    frames = np.asarray([frame_number(path) for path in frame_paths], dtype=int)
    shot_id, _, _ = segmentation.expand(frames)
    return {
        int(frame): int(shot)
        for frame, shot in zip(frames, shot_id, strict=True)
        if int(shot) >= 0
    }


def clip_propagation_shots(frames_directory: Path, fps: float) -> dict[int, int]:
    """Shot ids for one clip directory. Raises when the frames are not there."""
    paths = sorted(Path(frames_directory).glob("f_*.jpg"))
    if len(paths) < 2:
        raise ValueError("shot-homography propagation requires the clip's native frames")
    return propagation_shot_ids(paths, fps)


def build_shot_mask(
    frames_root: Path,
    output_path: Path,
    report_path: Path,
    fps: float,
    stride: int = 2,
    minimum_inliers: float = DEFAULT_MINIMUM_INLIERS,
) -> dict:
    rows = []
    clips = []
    for clip_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        frame_paths = sorted(clip_dir.glob("f_*.jpg"))
        if not frame_paths:
            continue
        segmentation = segment_shots(
            frame_paths,
            fps,
            stride=stride,
            minimum_inliers=minimum_inliers,
        )
        frames = np.asarray([frame_number(path) for path in frame_paths], dtype=int)
        shot_ids, is_play_camera, registration = segmentation.expand(frames)
        for frame, shot_id, is_play, inliers in zip(
            frames,
            shot_ids,
            is_play_camera,
            registration,
            strict=True,
        ):
            margin = float(inliers - minimum_inliers)
            rows.append(
                {
                    "clip": clip_dir.name,
                    "frame": f"f_{frame:04d}.jpg",
                    "shot_id": int(shot_id),
                    "is_play_camera": int(is_play),
                    "registration_inliers": float(inliers),
                    "classification_margin": margin,
                    "classification_confidence": min(
                        1.0,
                        abs(margin) / max(minimum_inliers, 1.0),
                    ),
                }
            )
        clips.append(
            {
                "clip": clip_dir.name,
                "frames": len(frame_paths),
                "play_camera_rate": float(np.mean(is_play_camera)),
                "shots": segmentation.shots,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "camera_shot_mask_v1",
        "fps": fps,
        "stride": stride,
        "minimum_inliers": minimum_inliers,
        "clips": clips,
        "frames": len(rows),
        "play_camera_rate": (
            sum(row["is_play_camera"] for row in rows) / len(rows) if rows else 0.0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--minimum-inliers", type=float, default=DEFAULT_MINIMUM_INLIERS)
    args = parser.parse_args()
    report = build_shot_mask(
        args.frames_root,
        args.output,
        args.report,
        args.fps,
        args.stride,
        args.minimum_inliers,
    )
    print(
        f"camera shots: clips={len(report['clips'])}, frames={report['frames']}, "
        f"play_camera_rate={report['play_camera_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
