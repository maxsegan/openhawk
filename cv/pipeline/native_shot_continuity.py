"""Conservatively qualify smooth native motion across a coarse cut proposal.

This does not discover cuts or repair time. An unavailable or ambiguous neighborhood leaves
its original boundary intact. Native ordering, positive registration and distributed motion
are all required before removing a sampled-difference proposal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.provenance import file_record
from cv.pipeline.shot_boundaries import GRID, _grid_histograms

# Existing native shot detector's histogram floor and immediate-neighbour ratio.
HISTOGRAM_CUT_DISTANCE = 0.22
LOCAL_CHANGE_RATIO = 1.8
MINIMUM_REGISTRATION_INLIERS = 60.0


def qualify_native_continuity(
    paths: list[Path],
    frames: np.ndarray,
    before: int,
    after: int,
    *,
    registration_inliers: float,
) -> dict:
    """Return positive continuity evidence, otherwise preserve the original boundary."""
    result = {
        "schema": "native_shot_boundary_continuity_v1",
        "before_frame": int(before),
        "after_frame": int(after),
        "continuous": False,
        "registration_inliers": float(registration_inliers),
        "reason": "unavailable_native_neighborhood",
    }
    if (
        len(paths) != len(frames)
        or len(set(map(int, frames))) != len(frames)
        or not np.isfinite(registration_inliers)
        or registration_inliers < MINIMUM_REGISTRATION_INLIERS
    ):
        result["reason"] = "registration_or_native_identity_unavailable"
        return result
    width = after - before
    if width < 3:
        result["reason"] = "insufficient_native_steps"
        return result
    indexed = {int(frame): path for frame, path in zip(frames, paths, strict=True)}
    required = list(range(before - width, after + width + 1))
    if not all(frame in indexed for frame in required):
        return result
    images = []
    evidence = []
    for frame in required:
        image = cv2.imread(str(indexed[frame]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return result
        images.append(cv2.resize(image, (320, 180)))
        evidence.append(file_record(indexed[frame], role="native_boundary_picture"))
    result["native_picture_evidence"] = evidence
    receipt_paths = sorted(
        {indexed[frame].parent / "extraction_receipt.json" for frame in required}
    )
    result["native_extraction_receipts"] = [
        file_record(path, role="native_extraction_receipt")
        for path in receipt_paths
        if path.is_file()
    ]
    result["cadence_validation"] = "ordinary_frame_cadence_gate_separate"
    data = np.asarray(images, dtype=np.float32)
    changes = np.mean(np.abs(np.diff(data, axis=0)), axis=(1, 2))
    middle = changes[width : 2 * width]
    outer = np.concatenate((changes[:width], changes[2 * width :]))
    mean = float(middle.mean())
    background = float(np.median(outer))
    result.update(
        native_frames=required,
        native_changes=changes.tolist(),
        middle_mean_change=mean,
        context_median_change=background,
    )
    if mean <= 0.0 or background <= 0.0 or float(middle.min()) <= 0.0:
        result["reason"] = "duplicate_or_static_native_steps"
        return result
    peak_ratio = float(middle.max()) / mean
    trough_ratio = mean / float(middle.min())
    context_ratio = mean / background
    result.update(
        peak_to_mean=peak_ratio,
        mean_to_minimum=trough_ratio,
        mean_to_context=context_ratio,
    )
    if max(peak_ratio, trough_ratio, context_ratio) > LOCAL_CHANGE_RATIO:
        result["reason"] = "temporally_localized_or_ambiguous_change"
        return result
    histograms = _grid_histograms(np.asarray(images, dtype=np.uint8))
    # Check both each native transition and accumulated pre/post appearance. A gradual
    # cross-view blend must not pass merely because no one native increment dominates.
    adjacent = np.abs(np.diff(histograms, axis=0)).sum(axis=1) / (2.0 * GRID * GRID)
    endpoint = float(np.abs(histograms[2 * width] - histograms[width]).sum() / (2.0 * GRID * GRID))
    maximum = max(float(adjacent.max()), endpoint)
    result["maximum_histogram_distance"] = maximum
    if maximum >= HISTOGRAM_CUT_DISTANCE:
        result["reason"] = "native_appearance_discontinuity"
        return result
    result.update(continuous=True, reason="registered_distributed_native_motion")
    return result
