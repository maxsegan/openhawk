"""Parallel CPU runtime for full-frame neural ball tracking.

The canonical :mod:`ball_neural` implementation supplies the shared model semantics. This wrapper preserves its
model batches and candidate semantics while parallelizing JPEG preprocessing that otherwise
stalls the GPU before every clip.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from cv.pipeline import ball_neural as baseline
from cv.pipeline import resolution as res

_BASELINE_INFER_CLIP = baseline.infer_clip


def infer_clip(
    model,
    frame_paths: list[str],
    batch_size: int,
    device: int,
    artifact_size: res.FrameSize = res.CANONICAL_SIZE,
    k_best: int = 1,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = False,
    subpixel: str = "argmax",
) -> list[dict]:
    import torch

    if not frame_paths:
        return []
    if not temporal_ensemble:
        return _BASELINE_INFER_CLIP(
            model,
            frame_paths,
            batch_size,
            device,
            artifact_size,
            k_best,
            peak_threshold,
            nms_radius,
            temporal_ensemble=False,
            subpixel=subpixel,
        )
    workers = min(8, os.cpu_count() or 1, len(frame_paths))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="ball-frame-preprocess",
    ) as executor:
        prepared = list(executor.map(baseline.preprocess, frame_paths))
    rows = []

    def append_prediction(index: int, heatmap: np.ndarray) -> None:
        frame_name = os.path.basename(frame_paths[index])
        frame_id = baseline.frame_number_from_name(frame_name)
        if k_best <= 1:
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peaks = [(int(x), int(y), float(heatmap[y, x]))]
        else:
            peaks = baseline.topk_peaks(
                heatmap,
                k_best,
                peak_threshold,
                nms_radius,
            )
        for rank, (x, y, score) in enumerate(peaks):
            rx, ry = baseline.subpixel_refine(heatmap, int(x), int(y), subpixel)
            original = cv2.transform(
                np.float32([[[rx, ry]]]),
                prepared[index][1],
            )[0, 0]
            artifact = res.scale_points(
                original,
                prepared[index][2],
                artifact_size,
            )
            row = {
                "frame": frame_name,
                "frame_id": frame_id,
                "x": float(artifact[0]),
                "y": float(artifact[1]),
                "score": float(score),
            }
            if k_best > 1:
                row["rank"] = rank
            rows.append(row)

    targets_per_batch = max(1, batch_size // 3)
    for target_offset in range(0, len(frame_paths), targets_per_batch):
        targets = list(
            range(
                target_offset,
                min(target_offset + targets_per_batch, len(frame_paths)),
            )
        )
        samples = []
        for target in targets:
            for indices, target_channel in baseline.temporal_contexts(
                target,
                len(frame_paths),
            ):
                tensor = np.concatenate(
                    [prepared[index][0] for index in indices],
                    axis=0,
                )
                samples.append((target, target_channel, tensor))
        inputs = torch.from_numpy(np.stack([item[2] for item in samples])).to(f"cuda:{device}")
        with torch.inference_mode():
            heatmaps = model(inputs)[0].sigmoid().cpu().numpy()
        grouped: dict[int, list[np.ndarray]] = {target: [] for target in targets}
        for (target, target_channel, _), sample in zip(
            samples,
            heatmaps,
            strict=True,
        ):
            grouped[target].append(sample[target_channel])
        for target in targets:
            append_prediction(target, np.mean(grouped[target], axis=0))
    return rows


def main() -> int:
    baseline.infer_clip = infer_clip
    return baseline.main()


if __name__ == "__main__":
    raise SystemExit(main())
